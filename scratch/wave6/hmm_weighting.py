"""Wave 6 — HMM-state-conditional sleeve weighting.

The wave-10 2-state HMM identified 2022 as 91.9% bear regime vs. only 43.4%
for the simple SPX-vol p80 gate.  That is a materially cleaner regime
signal, so it is worth re-using as the *weighting* lever (not just for
the three regime sleeves already in v15).

Variants built on top of the v15 TOP23 sleeve panel, ALL kept inside the
existing v15 production stack (high-vol regime gate -> fast-decay tripwire
-> time-of-month vol-target @ 18% -> DD control):

  A  HARD_SWITCH    p_bull > 0.6 -> 1.5x DIRECTIONAL, 0.7x MEAN_REV
                    p_bull < 0.4 -> 0.7x DIRECTIONAL, 1.5x MEAN_REV
                    else baseline.
  B  SMOOTH_PROB    per-sleeve weight = bull_aff * p_bull
                                       + bear_aff * (1 - p_bull)
                    where bull_aff/bear_aff are 1.5 for sleeves in the
                    matching family list, 0.7 otherwise.  Clamped to
                    [0.5, 1.6].
  C  IS_COND_SHARPE Per-sleeve IS Sharpe computed separately in bull and
                    bear bars.  Weight at bar t equals max(0, IS_Sharpe in
                    the CURRENT (p_bull-side) state), scaled to mean 1.0
                    across all active sleeves.  Strictly causal: only the
                    IS bucket Sharpes are used for the OOS window.
  D  COMBO_HMM_DD   Variant B plus the v15 fast-decay tripwire stacked.

Inputs
------
  scratch/quant/all_sleeve_returns_v15.parquet
  scratch/wave6/hmm_states.parquet  (causal filtered probabilities)

Outputs
-------
  scratch/wave6/hmm_weighting.py
  scratch/wave6/hmm_weighting_returns.parquet   (best variant daily ret)
  scratch/wave6/hmm_weighting_variants.csv      (per-variant comparison)

Baseline reference: v15 TOP23 -> OOS Sharpe +4.13, +5.62%/mo @18% vol,
                                 2022 Sharpe +4.21.
A candidate must beat +4.13 by at least +0.10 OOS Sharpe to be selected.
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
    fast_decay_tripwire, drawdown_control, build_regime_mask,
    stats, _bpy, GATES_HIGH,
)
from master_v12 import time_of_month_vol_target

OUT = Path(__file__).resolve().parent
QUANT = ROOT / "scratch" / "quant"
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
BASE_VOL_TARGET = 0.18

# ---------------------------------------------------------------------------
# Sleeve set / families (mirrors v15 TOP23)
# ---------------------------------------------------------------------------
TOP23 = ["RISKPAR", "TREND_NEW", "EVE_XAU", "D1REV_UK", "XSMOM",
         "D1REV_NAS", "WED_BTC", "DEFEND", "PAIRS_EXP",
         "VOLFORECAST", "H4_SLEEVE", "CRYPTO_vs_SPX",
         "CORR_REGIME", "SESSION_MOM",
         "W1_STRATS", "EVENT_VOLSPIKE",
         "STATARB_XS", "MICROSTR_D1", "VOL_BREAKOUT",
         "TERM_SPREADS", "EURGBP_MR", "MULTIDAY",
         "HMM_BULL_TSMOM"]

# Directional / momentum / beta sleeves -> boost in bull.
DIRECTIONAL = {"RISKPAR", "TREND_NEW", "XSMOM", "WED_BTC", "TSMOM",
               "HMM_BULL_TSMOM", "MOM_QUALITY", "VOL_BREAKOUT",
               "SESSION_MOM", "CRYPTO_DOM", "H4_SLEEVE"}

# Mean-reversion + defensive + crisis -> boost in bear.
MEAN_REV = {"D1REV_NAS", "D1REV_UK", "DEFEND", "PAIRS_EXP", "MULTIDAY",
            "MICROSTR_D1", "STATARB_XS", "EURGBP_MR", "TERM_SPREADS",
            "CORR_REGIME", "EVENT_VOLSPIKE", "CRYPTO_vs_SPX",
            "HMM_BEAR_REV"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def perf(name: str, ret: pd.Series) -> dict:
    s = stats(name, ret)
    out = {
        "variant": name,
        "FULL_sharpe": s.get("FULL_sharpe", 0.0),
        "IS_sharpe":   s.get("IS_sharpe",   0.0),
        "OOS_sharpe":  s.get("OOS_sharpe",  0.0),
        "OOS_ret":     s.get("OOS_ret",     0.0),
        "OOS_vol":     s.get("OOS_vol",     0.0),
        "OOS_dd":      s.get("OOS_dd",      0.0),
    }
    out["OOS_monthly"] = out["OOS_ret"] / 12.0
    y22 = ret[ret.index.year == 2022]
    if len(y22) > 10 and y22.std() > 0:
        bpy = _bpy(y22.index)
        out["Y2022_sharpe"] = float(y22.mean() * bpy / (y22.std(ddof=0) * np.sqrt(bpy)))
        eq22 = (1 + y22).cumprod()
        out["Y2022_dd"] = float((eq22 / eq22.cummax() - 1).min())
    else:
        out["Y2022_sharpe"] = 0.0
        out["Y2022_dd"] = 0.0
    return out


def overlay_stack(port: pd.Series) -> pd.Series:
    vt, _ = time_of_month_vol_target(port, base_target=BASE_VOL_TARGET)
    final, _ = drawdown_control(vt)
    return final


def portfolio_from_weights(panel: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    """sum(w * r) / sum(|w|) — so w=1 everywhere recovers equal-weight mean."""
    denom = weights.abs().sum(axis=1).replace(0.0, np.nan)
    port = (weights * panel).sum(axis=1) / denom
    return port.fillna(0.0)


def load_hmm_signal(index: pd.DatetimeIndex) -> pd.Series:
    """Causal HMM bull-state probability aligned to the panel index.

    The HMM in wave 10 was already trained walk-forward with online
    (filter-only) probabilities, i.e. p_bull[t] uses only information
    through close of bar t.  To be safely usable for weighting decisions
    on bar t we additionally lag by one bar.
    """
    hmm = pd.read_parquet(OUT / "hmm_states.parquet")
    hmm["timestamp"] = pd.to_datetime(hmm["timestamp"], utc=True)
    hmm["date"] = hmm["timestamp"].dt.floor("D")
    # de-dup any same-day entries (rare): take the last as latest filter.
    hmm = hmm.sort_values("date").drop_duplicates("date", keep="last")
    s = hmm.set_index("date")["p_bull"].sort_index()
    # ffill to panel index, then lag by one bar (avoid look-ahead).
    aligned = s.reindex(index).ffill().bfill()
    aligned = aligned.shift(1).fillna(0.5)
    return aligned.astype(float)


# ---------------------------------------------------------------------------
# Variant A: hard regime switch
# ---------------------------------------------------------------------------
def w_hard_switch(panel: pd.DataFrame, sleeves: list[str],
                   p_bull: pd.Series,
                   bull_thr: float = 0.6, bear_thr: float = 0.4,
                   boost: float = 1.5, cut: float = 0.7) -> pd.DataFrame:
    w = pd.DataFrame(1.0, index=panel.index, columns=sleeves)
    in_bull = p_bull > bull_thr
    in_bear = p_bull < bear_thr
    for s in sleeves:
        if s in DIRECTIONAL:
            w.loc[in_bull, s] = boost
            w.loc[in_bear, s] = cut
        elif s in MEAN_REV:
            w.loc[in_bull, s] = cut
            w.loc[in_bear, s] = boost
        # neutral sleeves stay at 1.0
    return w


# ---------------------------------------------------------------------------
# Variant B: smooth probability-weighted
# ---------------------------------------------------------------------------
def w_smooth_prob(panel: pd.DataFrame, sleeves: list[str],
                   p_bull: pd.Series,
                   high_aff: float = 1.5, low_aff: float = 0.7,
                   floor: float = 0.5, cap: float = 1.6) -> pd.DataFrame:
    w = pd.DataFrame(1.0, index=panel.index, columns=sleeves)
    for s in sleeves:
        if s in DIRECTIONAL:
            bull_aff, bear_aff = high_aff, low_aff
        elif s in MEAN_REV:
            bull_aff, bear_aff = low_aff, high_aff
        else:
            bull_aff, bear_aff = 1.0, 1.0
        col = bull_aff * p_bull + bear_aff * (1 - p_bull)
        w[s] = col.clip(floor, cap)
    return w


# ---------------------------------------------------------------------------
# Variant C: IS-conditional Sharpe weights
# ---------------------------------------------------------------------------
def compute_is_bucket_sharpes(panel: pd.DataFrame, sleeves: list[str],
                                p_bull: pd.Series,
                                bull_thr: float = 0.6,
                                bear_thr: float = 0.4) -> dict:
    """Compute per-sleeve IS Sharpe separately in bull / bear / neutral
    state buckets.  Returns dict mapping (sleeve, bucket) -> sharpe.

    Walk-forward safety: only uses bars with index < SPLIT.  These are
    constants applied to the OOS window, so no look-ahead from OOS into
    weighting decisions.
    """
    out = {}
    is_mask = panel.index < SPLIT
    pb = p_bull.loc[is_mask]
    is_bull = (pb > bull_thr)
    is_bear = (pb < bear_thr)
    is_neut = ~(is_bull | is_bear)
    for s in sleeves:
        r = panel.loc[is_mask, s]
        for label, mask in [("bull", is_bull), ("bear", is_bear), ("neut", is_neut)]:
            r_b = r[mask.values]
            if len(r_b) > 20 and r_b.std() > 0:
                bpy = _bpy(r_b.index)
                sh = float(r_b.mean() * bpy / (r_b.std(ddof=0) * np.sqrt(bpy)))
            else:
                sh = 0.0
            out[(s, label)] = sh
    return out


def w_is_cond_sharpe(panel: pd.DataFrame, sleeves: list[str],
                      p_bull: pd.Series,
                      bull_thr: float = 0.6, bear_thr: float = 0.4,
                      floor: float = 0.2, cap: float = 2.5) -> pd.DataFrame:
    buckets = compute_is_bucket_sharpes(panel, sleeves, p_bull,
                                         bull_thr=bull_thr, bear_thr=bear_thr)
    w = pd.DataFrame(1.0, index=panel.index, columns=sleeves)
    in_bull = p_bull > bull_thr
    in_bear = p_bull < bear_thr
    in_neut = ~(in_bull | in_bear)
    for s in sleeves:
        b_sh = max(0.0, buckets[(s, "bull")])
        e_sh = max(0.0, buckets[(s, "bear")])
        n_sh = max(0.0, buckets[(s, "neut")])
        col = pd.Series(1.0, index=panel.index)
        col.loc[in_bull] = b_sh
        col.loc[in_bear] = e_sh
        col.loc[in_neut] = n_sh
        w[s] = col
    # Per-bar mean-1 normalisation: scale row so the average weight across
    # sleeves is 1.0; floor/cap clip extreme tilts.
    row_mean = w.mean(axis=1).replace(0.0, np.nan)
    w = w.div(row_mean, axis=0).fillna(1.0).clip(floor, cap)
    return w, buckets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    panel = pd.read_parquet(QUANT / "all_sleeve_returns_v15.parquet")
    panel.index = pd.to_datetime(panel.index, utc=True)

    missing = [s for s in TOP23 if s not in panel.columns]
    if missing:
        raise RuntimeError(f"Missing sleeves in panel: {missing}")
    print(f"Loaded panel: {panel.shape}, using TOP23 = {len(TOP23)} sleeves")

    # ---- v15 prep: high-vol regime gate -> fast-decay tripwire ----
    gates = dict(GATES_HIGH)
    for s in ["PAIRS_EXP", "CRYPTO_vs_SPX", "CORR_REGIME", "STATARB_XS",
              "MICROSTR_D1", "EURGBP_MR", "TERM_SPREADS", "EVENT_VOLSPIKE",
              "MULTIDAY"]:
        gates[s] = 1.5
    for s in ["VOLFORECAST", "W1_STRATS", "SESSION_MOM"]:
        gates[s] = 1.0
    gates["H4_SLEEVE"] = 1.5
    gates["TREND_NEW"] = 0.5
    gates["HMM_BULL_TSMOM"] = 0.7
    gates["VOL_BREAKOUT"] = 1.2

    high_vol, _ = build_regime_mask(panel.index, 80)
    g = panel.copy()
    for s, m in gates.items():
        if s in g.columns:
            g.loc[high_vol, s] *= m

    sub = g[TOP23]
    after_decay, _ = fast_decay_tripwire(sub, TOP23)

    # ---- HMM bull-state probability (causal, 1-bar lagged) ----
    p_bull = load_hmm_signal(panel.index)
    print(f"HMM p_bull alignment: {p_bull.isna().sum()} NaN / {len(p_bull)}")
    print(f"  bars in bull (>0.6): {float((p_bull > 0.6).mean()):.2%}")
    print(f"  bars in bear (<0.4): {float((p_bull < 0.4).mean()):.2%}")
    y22_mask = p_bull.index.year == 2022
    print(f"  2022 bars in bear: {float((p_bull[y22_mask] < 0.4).mean()):.2%}")
    print(f"  2022 mean p_bull:  {float(p_bull[y22_mask].mean()):.3f}")

    # ---- Build variant portfolios ----
    base_w = pd.DataFrame(1.0, index=sub.index, columns=TOP23)
    baseline_port = portfolio_from_weights(after_decay, base_w)

    w_A = w_hard_switch(sub, TOP23, p_bull)
    w_B = w_smooth_prob(sub, TOP23, p_bull)
    w_C, bucket_sh = w_is_cond_sharpe(sub, TOP23, p_bull)
    # D: smooth prob weights applied to the decay-filtered sleeve panel
    # (i.e. stack with the v15 fast-decay tripwire).
    w_D = w_B  # weights identical to B; difference is panel below.

    ports = {
        "V0_BASELINE_v15": baseline_port,
        "A_HARD_SWITCH":   portfolio_from_weights(sub, w_A),
        "B_SMOOTH_PROB":   portfolio_from_weights(sub, w_B),
        "C_IS_COND_SH":    portfolio_from_weights(sub, w_C),
        "D_COMBO_HMM_DD":  portfolio_from_weights(after_decay, w_D),
    }

    # ---- Apply v15 overlay stack and score ----
    rows = []
    final_by_variant: dict[str, pd.Series] = {}
    for name, raw in ports.items():
        final = overlay_stack(raw)
        final_by_variant[name] = final
        rows.append(perf(name, final))

    df = pd.DataFrame(rows)
    base_oos_sh = df.loc[df["variant"] == "V0_BASELINE_v15", "OOS_sharpe"].iloc[0]
    base_oos_mo = df.loc[df["variant"] == "V0_BASELINE_v15", "OOS_monthly"].iloc[0]
    df["dSharpe_vs_v15"] = df["OOS_sharpe"] - base_oos_sh
    df["dMonthly_vs_v15_pct"] = (df["OOS_monthly"] - base_oos_mo) * 100

    df = df[["variant",
             "FULL_sharpe", "IS_sharpe", "OOS_sharpe", "Y2022_sharpe",
             "OOS_ret", "OOS_vol", "OOS_dd", "OOS_monthly",
             "Y2022_dd",
             "dSharpe_vs_v15", "dMonthly_vs_v15_pct"]]

    print("\n=== Variant comparison (overlay stack: TOM-vol@18% + DD ctrl) ===")
    print(f"v15 baseline reference: OOS Sharpe +4.13, +5.62%/mo @ 18% vol, 2022 +4.21")
    print(f"  Computed baseline here: OOS Sharpe {base_oos_sh:+.2f}, "
          f"+{base_oos_mo*100:.2f}%/mo")
    print()
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    df.to_csv(OUT / "hmm_weighting_variants.csv", index=False, float_format="%.4f")

    # ---- Per-sleeve regime affinity table ----
    aff_rows = []
    for s in TOP23:
        b = bucket_sh[(s, "bull")]
        e = bucket_sh[(s, "bear")]
        n = bucket_sh[(s, "neut")]
        aff_rows.append({"sleeve": s, "IS_bull_Sh": b, "IS_bear_Sh": e,
                         "IS_neut_Sh": n, "bull_minus_bear": b - e})
    aff_df = pd.DataFrame(aff_rows).sort_values("bull_minus_bear", ascending=False)
    print("\n=== IS regime affinity (per-sleeve Sharpe by HMM state) ===")
    print(aff_df.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
    aff_df.to_csv(OUT / "hmm_weighting_affinity.csv", index=False, float_format="%.4f")

    # ---- Pick the best variant under the gate (>= +0.10 OOS Sharpe over v15) ----
    eligible = df[(df["variant"] != "V0_BASELINE_v15")
                  & (df["dSharpe_vs_v15"] >= 0.10)]
    if len(eligible) > 0:
        eligible = eligible.sort_values("OOS_sharpe", ascending=False)
        best_variant = eligible.iloc[0]["variant"]
        print(f"\nBest variant chosen ({len(eligible)} pass +0.10 gate): {best_variant}")
    else:
        df2 = df[df["variant"] != "V0_BASELINE_v15"].copy().sort_values("OOS_sharpe", ascending=False)
        best_variant = df2.iloc[0]["variant"]
        print(f"\nNo variant beat v15 by +0.10. Best alternative: {best_variant}")

    best_ret = final_by_variant[best_variant]
    eq = (1 + best_ret).cumprod()
    out_df = pd.DataFrame({
        "timestamp": best_ret.index,
        "ret":       best_ret.values,
        "equity":    eq.values,
    })
    out_df.to_parquet(OUT / "hmm_weighting_returns.parquet", index=False)
    print(f"Saved {best_variant} returns -> {OUT / 'hmm_weighting_returns.parquet'}")

    # ---- Year-by-year for the best variant ----
    print(f"\n=== Year-by-year ({best_variant}) @ 18% vol ===")
    for year, sub_r in best_ret.groupby(best_ret.index.year):
        if len(sub_r) < 50:
            continue
        bpy = _bpy(sub_r.index)
        ar = sub_r.mean() * bpy
        av = sub_r.std(ddof=0) * np.sqrt(bpy)
        sh = ar / av if av > 0 else 0
        eq_y = (1 + sub_r).cumprod()
        dd = (eq_y / eq_y.cummax() - 1).min()
        print(f"  {year}  Sh={sh:+.2f}  Ret={ar:+.1%}  DD={dd:+.1%}  Mo={ar/12:+.2%}")


if __name__ == "__main__":
    main()
