"""Wave-5 regime-adaptive vol-targeting layer.

Variants:
  1. Static 18% target (baseline = current production).
  2. Linear vol-target ramp on SPX 30d realized vol (10..22%).
  3. Trailing-Sharpe-aware target (lever up when 3m Sharpe > 3, halve when < 1).
  4. Drawdown-aware target (18% * (1 - DD/0.10), zero at -10%).
  5. Combined regime + DD.
  6. Time-of-month seasonality.
  7. Vol-target-of-vol-target (60-day MA smoothing).

Build baseline from all_sleeve_returns_v11.parquet -> TOP14 EW + regime gate + tripwire.
Apply each variant's dynamic vol-target, then DD-control as final step.
IS <= 2024-01-01, OOS >= 2024-01-01.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scratch" / "quant"))

from master_v4 import (
    fast_decay_tripwire, vol_target_overlay, drawdown_control,
    build_regime_mask, _bpy, GATES_HIGH, SPLIT,
)

OUT = Path(__file__).resolve().parent
QUANT = ROOT / "scratch" / "quant"
DATA = ROOT / "data"

TOP14 = ["RISKPAR", "TREND_NEW", "EVE_XAU", "D1REV_UK", "XSMOM",
         "D1REV_NAS", "WED_BTC", "DEFEND", "PAIRS_EXP",
         "VOLFORECAST", "H4_SLEEVE", "CRYPTO_vs_SPX",
         "CORR_REGIME", "SESSION_MOM"]

STATIC_TARGET = 0.18
MAX_LEV = 15.0
VT_LOOKBACK = 60


# -------------- helpers --------------
def build_baseline_returns() -> pd.Series:
    """TOP14 equal-weight after regime-gate + decay tripwire. NO vol-target, NO DD ctrl."""
    panel = pd.read_parquet(QUANT / "all_sleeve_returns_v11.parquet")
    panel.index = pd.to_datetime(panel.index, utc=True)

    gates = dict(GATES_HIGH)
    for s in ["PAIRS_EXP", "CRYPTO_vs_SPX", "CORR_REGIME"]:
        gates[s] = 1.5
    gates["VOLFORECAST"] = 1.0
    gates["H4_SLEEVE"] = 1.5
    gates["TREND_NEW"] = 0.5
    gates["SESSION_MOM"] = 1.0

    high_vol, _ = build_regime_mask(panel.index, 80)
    g = panel.copy()
    for s, m in gates.items():
        if s in g.columns:
            g.loc[high_vol, s] *= m

    after_decay, _ = fast_decay_tripwire(g[TOP14], TOP14)
    return after_decay.mean(axis=1)


def spx_rv30(idx: pd.DatetimeIndex) -> pd.Series:
    spx = pd.read_parquet(DATA / "SPX500_USD" / "D1.parquet").copy()
    spx["timestamp"] = pd.to_datetime(spx["timestamp"], utc=True)
    spx["timestamp"] = spx["timestamp"].dt.floor("D")
    spx = spx.sort_values("timestamp").drop_duplicates("timestamp")
    spx["ret"] = np.log(spx["close"] / spx["close"].shift(1))
    spx["rv30"] = spx["ret"].rolling(30).std() * np.sqrt(252)
    aligned = pd.merge_asof(
        pd.DataFrame({"timestamp": idx}).sort_values("timestamp"),
        spx[["timestamp", "rv30"]].sort_values("timestamp"),
        on="timestamp", direction="backward",
    )
    return pd.Series(aligned["rv30"].values, index=idx)


def dynamic_voltarget_apply(returns: pd.Series, target_series: pd.Series,
                            lookback: int = VT_LOOKBACK, max_lev: float = MAX_LEV) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Apply a *time-varying* target vol. Walk-forward — uses trailing realized vol shifted by 1."""
    rolling_vol = returns.rolling(lookback).std(ddof=0).shift(1) * np.sqrt(365.25)
    tgt = target_series.shift(1).reindex(returns.index).ffill()
    lev = (tgt / rolling_vol).clip(upper=max_lev).fillna(1.0)
    lev = lev.where(rolling_vol > 0, 1.0)
    lev = lev.where(tgt > 0, 0.0)  # zero-target -> flat
    return returns * lev, lev, tgt


def lever_change_freq(lev: pd.Series) -> float:
    """Fraction of months where target leverage changed > 25% from previous month-end."""
    me = lev.resample("ME").last().dropna()
    if len(me) < 2:
        return 0.0
    pct = (me / me.shift(1) - 1).abs()
    return float((pct > 0.25).mean())


def variant_metrics(label: str, ret: pd.Series, lev: pd.Series) -> dict:
    oos = ret[ret.index >= SPLIT]
    is_part = ret[ret.index < SPLIT]
    bpy = _bpy(oos.index)
    ar = float(oos.mean()) * bpy
    av = float(oos.std(ddof=0)) * np.sqrt(bpy)
    sh = ar / av if av > 0 else 0.0
    eq = (1 + oos).cumprod()
    dd = float((eq / eq.cummax() - 1).min())

    # IS stats
    bpy_is = _bpy(is_part.index)
    ar_is = float(is_part.mean()) * bpy_is
    av_is = float(is_part.std(ddof=0)) * np.sqrt(bpy_is)
    sh_is = ar_is / av_is if av_is > 0 else 0.0
    eq_is = (1 + is_part).cumprod()
    dd_is = float((eq_is / eq_is.cummax() - 1).min())

    # OOS monthly distribution
    monthly = oos.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    med = float(monthly.median()) if len(monthly) else 0.0
    p5 = float(monthly.quantile(0.05)) if len(monthly) else 0.0
    p95 = float(monthly.quantile(0.95)) if len(monthly) else 0.0

    return {
        "variant": label,
        "IS_sharpe": sh_is, "IS_ret": ar_is, "IS_vol": av_is, "IS_maxdd": dd_is,
        "OOS_sharpe": sh,  "OOS_ret": ar,   "OOS_vol": av,   "OOS_maxdd": dd,
        "OOS_monthly_med": med, "OOS_monthly_p5": p5, "OOS_monthly_p95": p95,
        "OOS_avg_lev": float(lev[lev.index >= SPLIT].mean()),
        "lev_change_freq_25pct": lever_change_freq(lev[lev.index >= SPLIT]),
    }


# -------------- target series for each variant --------------

def target_static(idx) -> pd.Series:
    return pd.Series(STATIC_TARGET, index=idx)


def target_linear_ramp(idx) -> pd.Series:
    """10% at SPX rv30 > IS p80, 22% at SPX rv30 < IS p20, linear in-between.
    Walk-forward: IS percentiles computed from IS rv30 only."""
    rv = spx_rv30(idx)
    rv_is = rv[idx < SPLIT].dropna()
    p20 = float(rv_is.quantile(0.20))
    p80 = float(rv_is.quantile(0.80))
    # Map: rv <= p20 -> 0.22, rv >= p80 -> 0.10, linear between
    def f(v):
        if pd.isna(v):
            return STATIC_TARGET
        if v <= p20: return 0.22
        if v >= p80: return 0.10
        # linear: at p20 -> 0.22, at p80 -> 0.10
        frac = (v - p20) / (p80 - p20)
        return 0.22 + frac * (0.10 - 0.22)
    out = rv.apply(f)
    # Smooth with 10-day MA to prevent flicker
    return out.rolling(10, min_periods=1).mean()


def target_sharpe_aware(returns: pd.Series, idx) -> pd.Series:
    """Trailing-3m (63 trading-day) Sharpe of *baseline* returns.
       Sh > 3 -> target *= 1.5, Sh < 1 -> target *= 0.5, else 1.0.
       Walk-forward: shifted by 1."""
    lookback = 63
    rmean = returns.rolling(lookback).mean()
    rstd = returns.rolling(lookback).std(ddof=0)
    sh = (rmean / rstd) * np.sqrt(365.25)
    sh = sh.shift(1)
    mult = pd.Series(1.0, index=returns.index)
    mult = mult.where(~(sh > 3.0), 1.5)
    mult = mult.where(~(sh < 1.0), 0.5)
    out = (STATIC_TARGET * mult).reindex(idx).fillna(STATIC_TARGET)
    return out


def target_dd_aware(returns: pd.Series, idx) -> pd.Series:
    """Target = 18% * (1 - DD / 0.10).
       Use trailing peak from full returns history (walk-forward — uses past only)."""
    eq = (1 + returns).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1   # negative or zero
    raw_mult = 1.0 + dd / 0.10  # at dd=0 -> 1.0, at dd=-0.10 -> 0
    raw_mult = raw_mult.clip(lower=0.0, upper=1.0)
    # Walk-forward — shift
    raw_mult = raw_mult.shift(1).fillna(1.0)
    return (STATIC_TARGET * raw_mult).reindex(idx).fillna(STATIC_TARGET)


def target_combined(returns: pd.Series, idx) -> pd.Series:
    """Regime scaler (from linear ramp normalized to 1.0 around 18%) * DD scaler."""
    ramp = target_linear_ramp(idx)
    regime_scaler = ramp / STATIC_TARGET
    dd = target_dd_aware(returns, idx)
    dd_scaler = dd / STATIC_TARGET
    return (STATIC_TARGET * regime_scaler * dd_scaler).clip(lower=0.0)


def target_tom_seasonality(idx) -> pd.Series:
    """First 10 days of month -> 20%, days 11-20 -> 16%, days 21+ -> 18%."""
    day = pd.Series(idx, index=idx).dt.day
    out = pd.Series(STATIC_TARGET, index=idx)
    out.loc[day <= 10] = 0.20
    out.loc[(day > 10) & (day <= 20)] = 0.16
    out.loc[day > 20] = 0.18
    return out


def target_smoothed(base: pd.Series) -> pd.Series:
    """60-day MA of an underlying target series — applied as wrapper around an existing target."""
    return base.rolling(60, min_periods=1).mean()


# -------------- main --------------
def main():
    print("Building baseline returns (TOP14 EW + regime + tripwire, no vol-target, no DD)")
    baseline = build_baseline_returns()
    idx = baseline.index
    print(f"  rows: {len(baseline)}  range: {idx[0].date()} -> {idx[-1].date()}")

    # Pre-build each target series.
    targets = {
        "V1_static_18": target_static(idx),
        "V2_linear_ramp": target_linear_ramp(idx),
        "V3_sharpe_aware": target_sharpe_aware(baseline, idx),
        "V4_dd_aware": target_dd_aware(baseline, idx),
        "V5_combined_regime_dd": target_combined(baseline, idx),
        "V6_tom_seasonality": target_tom_seasonality(idx),
    }
    # V7 = smoothed version of V5 (the most-dynamic variant) — demonstrates turnover suppression
    targets["V7_smoothed_target"] = target_smoothed(targets["V5_combined_regime_dd"])

    rows = []
    series_store: dict[str, pd.Series] = {}
    for label, tgt in targets.items():
        vt_ret, lev, _ = dynamic_voltarget_apply(baseline, tgt)
        dd_ret, _ = drawdown_control(vt_ret)   # final-step DD control on ALL variants for parity
        m = variant_metrics(label, dd_ret, lev)
        rows.append(m)
        series_store[label] = dd_ret

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "regime_voltarget_variants.csv", index=False)

    # Display
    print("\n=== Variant comparison ===")
    cols = ["variant", "IS_sharpe", "IS_maxdd",
            "OOS_sharpe", "OOS_ret", "OOS_vol", "OOS_maxdd",
            "OOS_monthly_med", "OOS_monthly_p5",
            "OOS_avg_lev", "lev_change_freq_25pct"]
    print(df[cols].to_string(index=False,
        formatters={
            "IS_sharpe": "{:+.2f}".format, "IS_maxdd": "{:+.1%}".format,
            "OOS_sharpe": "{:+.2f}".format, "OOS_ret": "{:+.1%}".format,
            "OOS_vol": "{:.1%}".format, "OOS_maxdd": "{:+.1%}".format,
            "OOS_monthly_med": "{:+.2%}".format, "OOS_monthly_p5": "{:+.2%}".format,
            "OOS_avg_lev": "{:.2f}".format, "lev_change_freq_25pct": "{:.1%}".format,
        }))

    # Pick best: highest OOS Sharpe with MaxDD reduction >= 1pp vs static
    static_dd = df.loc[df["variant"] == "V1_static_18", "OOS_maxdd"].iloc[0]
    static_sh = df.loc[df["variant"] == "V1_static_18", "OOS_sharpe"].iloc[0]

    # Filter: DD must be at least 1pp better (i.e., less negative)
    candidates = df[(df["OOS_maxdd"] >= static_dd + 0.01) &
                    (df["OOS_sharpe"] >= static_sh - 0.1) &
                    (df["variant"] != "V1_static_18")].copy()
    print(f"\nStatic baseline: OOS_Sh={static_sh:+.2f}, MaxDD={static_dd:+.1%}")
    print(f"Candidates meeting DD>=+1pp AND Sharpe drop <=0.1: {len(candidates)}")

    if len(candidates) > 0:
        # rank by OOS Sharpe
        candidates = candidates.sort_values("OOS_sharpe", ascending=False)
        best_label = candidates.iloc[0]["variant"]
        print(f"Best constrained variant: {best_label}")
    else:
        # Fall back: best OOS Sharpe overall
        best_label = df.sort_values("OOS_sharpe", ascending=False).iloc[0]["variant"]
        print(f"No variant meets DD criterion; recommending best Sharpe: {best_label}")

    best = series_store[best_label]
    eq_best = (1 + best).cumprod()
    pd.DataFrame({"timestamp": best.index, "ret": best.values, "equity": eq_best.values}).to_parquet(
        OUT / "regime_voltarget_returns.parquet", index=False)
    print(f"\nSaved best variant ({best_label}) -> regime_voltarget_returns.parquet")

    # Also report yearly for best
    print(f"\n=== {best_label} yearly OOS ===")
    for year, sub in best.groupby(best.index.year):
        if len(sub) < 30:
            continue
        bpy = _bpy(sub.index)
        ar = float(sub.mean()) * bpy
        av = float(sub.std(ddof=0)) * np.sqrt(bpy)
        shy = ar / av if av > 0 else 0
        eq_y = (1 + sub).cumprod()
        ddy = float((eq_y / eq_y.cummax() - 1).min())
        print(f"  {year}  Sh={shy:+.2f}  Ret={ar:+.1%}  Vol={av:.1%}  DD={ddy:+.1%}")


if __name__ == "__main__":
    main()
