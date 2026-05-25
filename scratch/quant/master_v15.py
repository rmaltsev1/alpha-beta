"""Master v15 — final production build.

Adds on top of v14:
  + Stops on DEFEND/EVENT_VOLSPIKE/CRYPTO_vs_SPX (wave 9)
  + HMM regime sleeves (wave 10) — HMM_BULL_TSMOM, HMM_SLEEVE_MIX, HMM_BEAR_REV
  + V8 COMBO_LIGHT DD-conditional refinement

Drops considered but rejected:
  - Multi-leg spreads (duplicates PAIRS_EXP)
  - Bayesian weighting (doesn't beat EW)
  - Bucket weighting (within noise)

Note: capacity analysis flagged $20M as breaking point due to UK100 bottleneck.
For $100M+ deployment, drop 14 sleeves (see capacity report).
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
    build_regime_mask, stats, _bpy, GATES_HIGH,
)
from master_v12 import time_of_month_vol_target

OUT = Path(__file__).resolve().parent
WAVE6 = ROOT / "scratch" / "wave6"
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
TARGET_VOL = 0.05


def normalize_idx(s):
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    else:
        s.index = s.index.tz_convert("UTC")
    s = s.groupby(s.index.floor("D")).sum()
    s.index = pd.to_datetime(s.index, utc=True)
    return s


def rescale_is(s, target):
    is_part = s[s.index < SPLIT]
    av = float(is_part.std(ddof=0)) * np.sqrt(365.25)
    return s * (target / av) if av > 1e-9 else s * 0


def main():
    panel = pd.read_parquet(OUT / "all_sleeve_returns_v14.parquet")
    panel.index = pd.to_datetime(panel.index, utc=True)

    # Add extended stops for 3 more sleeves
    stops_df = pd.read_parquet(WAVE6 / "stops_extended_returns.parquet")
    if stops_df.index.tz is None:
        stops_df.index = stops_df.index.tz_localize("UTC")
    for stops_col, target_name in [("DEFEND_time_3bar", "DEFEND"),
                                     ("EVENT_VOLSPIKE_time_3bar", "EVENT_VOLSPIKE"),
                                     ("CRYPTO_vs_SPX_atr_1.5x", "CRYPTO_vs_SPX")]:
        if stops_col in stops_df.columns:
            s = pd.Series(stops_df[stops_col].values, index=stops_df.index)
            s = normalize_idx(s.rename(target_name))
            s = rescale_is(s, TARGET_VOL)
            panel[target_name] = s.reindex(panel.index, fill_value=0.0)
            print(f"Replaced {target_name} with stops")

    # Add HMM regime sleeves
    hmm_path = WAVE6 / "hmm_returns.parquet"
    if hmm_path.exists():
        hmm_df = pd.read_parquet(hmm_path)
        if "timestamp" not in hmm_df.columns:
            hmm_df["timestamp"] = hmm_df.index
        for hmm_col in ["HMM_BULL_TSMOM", "HMM_SLEEVE_MIX", "HMM_BEAR_REV"]:
            if hmm_col in hmm_df.columns:
                ts = pd.to_datetime(hmm_df["timestamp"], utc=True)
                s = pd.Series(hmm_df[hmm_col].values, index=ts)
                s = normalize_idx(s.rename(hmm_col))
                s = rescale_is(s, TARGET_VOL)
                panel[hmm_col] = s.reindex(panel.index, fill_value=0.0)
                is_part = panel[hmm_col][panel.index < SPLIT]
                oos_part = panel[hmm_col][panel.index >= SPLIT]
                is_sh = is_part.mean()*365/(is_part.std(ddof=0)*np.sqrt(365)) if is_part.std() > 0 else 0
                oos_sh = oos_part.mean()*365/(oos_part.std(ddof=0)*np.sqrt(365)) if oos_part.std() > 0 else 0
                print(f"Added {hmm_col:<18}: IS_Sh={is_sh:+.2f}  OOS_Sh={oos_sh:+.2f}")

    panel.to_parquet(OUT / "all_sleeve_returns_v15.parquet")

    # TOP25 — v14 TOP22 + 3 HMM
    TOP25 = ["RISKPAR", "TREND_NEW", "EVE_XAU", "D1REV_UK", "XSMOM",
             "D1REV_NAS", "WED_BTC", "DEFEND", "PAIRS_EXP",
             "VOLFORECAST", "H4_SLEEVE", "CRYPTO_vs_SPX",
             "CORR_REGIME", "SESSION_MOM",
             "W1_STRATS", "EVENT_VOLSPIKE",
             "STATARB_XS", "MICROSTR_D1", "VOL_BREAKOUT",
             "TERM_SPREADS", "EURGBP_MR", "MULTIDAY",
             "HMM_BULL_TSMOM", "HMM_SLEEVE_MIX", "HMM_BEAR_REV"]

    gates = dict(GATES_HIGH)
    for s in ["PAIRS_EXP", "CRYPTO_vs_SPX", "CORR_REGIME", "STATARB_XS",
              "MICROSTR_D1", "EURGBP_MR", "TERM_SPREADS", "EVENT_VOLSPIKE", "MULTIDAY",
              "HMM_BEAR_REV"]:
        gates[s] = 1.5
    for s in ["VOLFORECAST", "W1_STRATS", "SESSION_MOM", "HMM_SLEEVE_MIX"]:
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

    # Compare
    print(f"\n{'Variant':<30} {'OOS_Sh':>7} {'2022_Sh':>8} {'OOS_DD':>8}")
    print("-" * 60)
    for name, sleeves in [
        ("TOP22 v14", TOP25[:-3]),
        ("TOP23 + HMM_BULL", TOP25[:-3] + ["HMM_BULL_TSMOM"]),
        ("TOP24 + HMM_BULL + MIX", TOP25[:-3] + ["HMM_BULL_TSMOM", "HMM_SLEEVE_MIX"]),
        ("TOP25 (all v15)", TOP25),
    ]:
        port = g[sleeves].mean(axis=1)
        s_st = stats(name, port)
        y22 = port[port.index.year == 2022]
        bpy = _bpy(y22.index) if len(y22) > 0 else 252
        sh22 = (y22.mean()*bpy)/(y22.std(ddof=0)*np.sqrt(bpy)) if y22.std() > 0 else 0
        print(f"{name:<30} {s_st.get('OOS_sharpe',0):>+7.2f} {sh22:>+8.2f} {s_st.get('OOS_dd',0):>+8.1%}")

    after_decay, _ = fast_decay_tripwire(g[TOP25], TOP25)
    full_baseline = after_decay.mean(axis=1)

    print(f"\n=== Leverage sweep v15 ===")
    print(f"{'VolTgt':<8} {'OOS_Ret':>8} {'OOS_Vol':>8} {'Sharpe':>7} {'MaxDD':>8} {'Mo':>7}")
    for tv in [0.10, 0.15, 0.18, 0.22, 0.27, 0.32]:
        vt, lev = time_of_month_vol_target(full_baseline, base_target=tv)
        dd, _ = drawdown_control(vt)
        oos = dd[dd.index >= SPLIT]
        bpy = _bpy(oos.index)
        ar = oos.mean()*bpy
        av = oos.std(ddof=0)*np.sqrt(bpy)
        sh = ar/av if av > 0 else 0
        eq = (1+oos).cumprod()
        ddv = (eq/eq.cummax()-1).min()
        print(f"{tv:<8.0%} {ar:>+8.1%} {av:>8.1%} {sh:>+7.2f} {ddv:>+8.1%} {ar/12:>+7.2%}")

    vt, _ = time_of_month_vol_target(full_baseline, base_target=0.18)
    prod, _ = drawdown_control(vt)
    eq = (1 + prod).cumprod()
    pd.DataFrame({"timestamp": prod.index, "ret": prod.values, "equity": eq.values}).to_parquet(
        OUT / "PRODUCTION_v15.parquet", index=False)

    print(f"\n=== Year-by-year v15 PRODUCTION (18% vol) ===")
    for year, sub in prod.groupby(prod.index.year):
        if len(sub) < 50: continue
        bpy = _bpy(sub.index)
        ar = sub.mean()*bpy
        av = sub.std(ddof=0)*np.sqrt(bpy)
        sh = ar/av if av > 0 else 0
        eq_y = (1+sub).cumprod()
        dd = (eq_y/eq_y.cummax()-1).min()
        print(f"  {year}  Sh={sh:+.2f}  Ret={ar:+.1%}  DD={dd:+.1%}  Mo={ar/12:+.2%}")


if __name__ == "__main__":
    main()
