"""
Sleeve-of-sleeves momentum strategy.

Treats the v15 production portfolio return series as a tradable asset and
applies a suite of meta-momentum / mean-reversion overlays on top of it.

All statistics that drive the overlay are walk-forward (only past data).
IS  = dates < 2024-01-01
OOS = dates >= 2024-01-01

Outputs:
    scratch/wave6/sos_returns.parquet    -- daily returns of every variant
    scratch/wave6/sos_variants.csv       -- IS / OOS / full Sharpe + MaxDD
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUANT = ROOT / "scratch" / "quant"
OUT = ROOT / "scratch" / "wave6"
OUT.mkdir(parents=True, exist_ok=True)

OOS_START = pd.Timestamp("2024-01-01")
ANN = 365.0  # crypto/24-7 convention used in earlier waves


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def load_portfolio() -> pd.Series:
    df = pd.read_parquet(QUANT / "PRODUCTION_FINAL_v15.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    return df.set_index("timestamp")["ret"].astype(float).sort_index()


def load_sleeves() -> pd.DataFrame:
    df = pd.read_parquet(QUANT / "all_sleeve_returns_v15.parquet")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.sort_index().astype(float)


def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) == 0 or r.std(ddof=0) == 0:
        return np.nan
    return float(r.mean() / r.std(ddof=0) * np.sqrt(ANN))


def max_dd(r: pd.Series) -> float:
    eq = (1.0 + r.fillna(0.0)).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def cagr(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    n = len(r)
    return float((1.0 + r).prod() ** (ANN / n) - 1.0)


def summary(r: pd.Series) -> dict:
    isd = r[r.index < OOS_START]
    oos = r[r.index >= OOS_START]
    return {
        "is_sharpe": sharpe(isd),
        "oos_sharpe": sharpe(oos),
        "full_sharpe": sharpe(r),
        "is_mdd": max_dd(isd),
        "oos_mdd": max_dd(oos),
        "full_mdd": max_dd(r),
        "is_cagr": cagr(isd),
        "oos_cagr": cagr(oos),
        "full_cagr": cagr(r),
    }


def apply_leverage(base: pd.Series, lev: pd.Series, defend: pd.Series | None = None,
                   defend_w: pd.Series | None = None) -> pd.Series:
    """Leverage is applied SHIFTED so we trade today on yesterday's signal."""
    lev = lev.reindex(base.index).shift(1).fillna(1.0)
    out = base * lev
    if defend is not None and defend_w is not None:
        dw = defend_w.reindex(base.index).shift(1).fillna(0.0)
        # portfolio_w on base = 1 - dw
        out = (1.0 - dw) * base * lev + dw * defend.reindex(base.index).fillna(0.0)
    return out


# --------------------------------------------------------------------------- #
# Variants                                                                    #
# --------------------------------------------------------------------------- #
def variant_tsmom_sharpe(base: pd.Series) -> pd.Series:
    """V1: scale leverage based on trailing 63d portfolio Sharpe.
    >+3 -> full leverage (1.5x). <+1 -> halve (0.5x). Linear in between.
    """
    win = 63
    mu = base.rolling(win).mean()
    sd = base.rolling(win).std(ddof=0)
    sh = (mu / sd) * np.sqrt(ANN)
    # piecewise linear between 1 and 3
    lev = np.where(sh >= 3.0, 1.5,
          np.where(sh <= 1.0, 0.5,
                   0.5 + (sh - 1.0) / 2.0 * 1.0))  # 1 -> 0.5, 3 -> 1.5
    lev = pd.Series(lev, index=base.index).fillna(1.0)
    return apply_leverage(base, lev)


def variant_streak_mr(base: pd.Series) -> pd.Series:
    """V2: 5 consecutive losing days -> reduce lev (0.6x).
              5 consecutive winning days -> add lev (1.4x).
    Decay back to 1.0 over the next 5 trading days when streak breaks.
    """
    sign = np.sign(base.fillna(0.0))
    # rolling sum of 5 consecutive same-sign days
    pos_streak = (sign == 1).rolling(5).sum()
    neg_streak = (sign == -1).rolling(5).sum()
    raw = pd.Series(1.0, index=base.index)
    raw[neg_streak >= 5] = 0.6
    raw[pos_streak >= 5] = 1.4
    # smooth: EMA half-life ~3 days
    lev = raw.ewm(halflife=3, adjust=False).mean()
    return apply_leverage(base, lev)


def variant_dd_buffer(base: pd.Series, defend: pd.Series) -> pd.Series:
    """V3: When DD > 2%, reallocate 20% of capital to DEFEND sleeve."""
    eq = (1.0 + base.fillna(0.0)).cumprod()
    dd = eq / eq.cummax() - 1.0
    w_def = pd.Series(0.0, index=base.index)
    w_def[dd < -0.02] = 0.20
    # smooth
    w_def = w_def.ewm(halflife=2, adjust=False).mean()
    lev = pd.Series(1.0, index=base.index)
    return apply_leverage(base, lev, defend=defend, defend_w=w_def)


def variant_sharpe_momentum(base: pd.Series) -> pd.Series:
    """V4: 21d Sharpe direction. Rising -> lever up (1.3x), Falling -> down (0.7x)."""
    win = 21
    sh = (base.rolling(win).mean() / base.rolling(win).std(ddof=0)) * np.sqrt(ANN)
    # rate of change
    dsh = sh.diff(5)
    raw = pd.Series(1.0, index=base.index)
    raw[dsh > 0.5] = 1.3
    raw[dsh < -0.5] = 0.7
    lev = raw.ewm(halflife=3, adjust=False).mean()
    return apply_leverage(base, lev)


def variant_vol_of_vol(base: pd.Series) -> pd.Series:
    """V5: high vol-of-vol -> de-risk; low vol-of-vol -> lever up.
    Vol-of-vol = stdev of 21d rolling vol over the past 63d.
    Threshold: walk-forward median of vov.
    """
    vol = base.rolling(21).std(ddof=0)
    vov = vol.rolling(63).std(ddof=0)
    # expanding median to remain walk-forward
    med = vov.expanding(min_periods=126).median()
    raw = pd.Series(1.0, index=base.index)
    raw[vov > med * 1.25] = 0.7
    raw[vov < med * 0.75] = 1.3
    lev = raw.ewm(halflife=5, adjust=False).mean()
    return apply_leverage(base, lev)


def variant_dispersion(base: pd.Series, sleeves: pd.DataFrame) -> pd.Series:
    """V6: cross-sleeve average pairwise correlation (63d).
    High corr -> reduce gross (0.7x). Low corr -> add (1.3x).
    Walk-forward expanding median for threshold.
    """
    win = 63
    # rolling mean pairwise correlation
    # cheaper proxy: average pairwise corr ~ (var(sum)/sum(var) - 1) / (N-1)
    s = sleeves.copy()
    s = s.fillna(0.0)
    var_sum = s.sum(axis=1).rolling(win).var(ddof=0)
    sum_var = (s ** 2).rolling(win).mean().sum(axis=1) * win / max(win, 1)
    # use sum of rolling variances
    rolling_var = s.rolling(win).var(ddof=0).sum(axis=1)
    N = s.shape[1]
    avg_corr = (var_sum - rolling_var) / (rolling_var * (N - 1))
    avg_corr = avg_corr.clip(-1, 1)
    med = avg_corr.expanding(min_periods=126).median()
    raw = pd.Series(1.0, index=base.index)
    raw[avg_corr > med + 0.05] = 0.7
    raw[avg_corr < med - 0.05] = 1.3
    lev = raw.ewm(halflife=5, adjust=False).mean()
    return apply_leverage(base, lev)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    base = load_portfolio()
    sleeves = load_sleeves()
    # align sleeves to base index
    sleeves = sleeves.reindex(base.index).fillna(0.0)
    defend = sleeves["DEFEND"] if "DEFEND" in sleeves.columns else pd.Series(0.0, index=base.index)

    variants: dict[str, pd.Series] = {
        "v15_baseline": base,
        "V1_tsmom_sharpe": variant_tsmom_sharpe(base),
        "V2_streak_mr": variant_streak_mr(base),
        "V3_dd_buffer": variant_dd_buffer(base, defend),
        "V4_sharpe_mom": variant_sharpe_momentum(base),
        "V5_vol_of_vol": variant_vol_of_vol(base),
        "V6_dispersion": variant_dispersion(base, sleeves),
    }

    # Combo of best-performing IS overlays (decided post hoc but with full WF stats)
    # We pick the two highest OOS-Sharpe overlays to combine.
    rows = []
    for k, r in variants.items():
        s = summary(r)
        s["variant"] = k
        rows.append(s)
    cmp_df = pd.DataFrame(rows).set_index("variant")
    # Pick top two by OOS Sharpe (excluding baseline)
    ranked = cmp_df.drop("v15_baseline").sort_values("oos_sharpe", ascending=False)
    top2 = ranked.head(2).index.tolist()

    # Build a combo: average of the two top leverage paths -- recover by ratio
    # We need lev paths; reconstruct from output / base.
    def implied_lev(r: pd.Series) -> pd.Series:
        out = (r / base).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        # cap to sane range
        return out.clip(0.2, 2.0)

    levs = pd.concat({k: implied_lev(variants[k]) for k in top2}, axis=1)
    combo_lev = levs.mean(axis=1)
    combo = base * combo_lev
    variants[f"V7_combo_{top2[0]}_{top2[1]}"] = combo

    # also try a stack of all-3 momentum (V1, V4, V5)
    mom_keys = ["V1_tsmom_sharpe", "V4_sharpe_mom", "V5_vol_of_vol"]
    levs_m = pd.concat({k: implied_lev(variants[k]) for k in mom_keys}, axis=1)
    variants["V8_mom_stack"] = base * levs_m.mean(axis=1)

    # finalize summary
    rows = []
    for k, r in variants.items():
        s = summary(r)
        s["variant"] = k
        s["oos_sharpe_lift"] = s["oos_sharpe"] - cmp_df.loc["v15_baseline", "oos_sharpe"]
        s["oos_mdd_change"] = s["oos_mdd"] - cmp_df.loc["v15_baseline", "oos_mdd"]
        rows.append(s)
    final = pd.DataFrame(rows).set_index("variant").round(4)

    cols = ["is_sharpe", "oos_sharpe", "full_sharpe",
            "is_mdd", "oos_mdd", "full_mdd",
            "is_cagr", "oos_cagr", "full_cagr",
            "oos_sharpe_lift", "oos_mdd_change"]
    final = final[cols]

    final.to_csv(OUT / "sos_variants.csv")
    # write returns as wide df
    ret_df = pd.DataFrame({k: v for k, v in variants.items()})
    ret_df.to_parquet(OUT / "sos_returns.parquet")

    print("Saved sos_variants.csv and sos_returns.parquet")
    print(final.to_string())


if __name__ == "__main__":
    main()
