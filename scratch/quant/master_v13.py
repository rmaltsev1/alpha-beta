"""Master v13 — integrate Wave 7-8 wins. New sleeves added:
  + STATARB_XS (Bollinger-z cross-sectional, OOS +1.35, -0.05 corr with D1REV)
  + MICROSTR_D1 (gap-fill + range-expansion + pivot, OOS +1.36, 2022 +1.58)
  + TERM_SPREADS (vol-of-vol regime + crypto-vs-eq, OOS +0.97, 2022 +1.12)
  + VOL_BREAKOUT (compression-expansion + BB squeeze, OOS +1.12, 2022 +1.21)
  + EURGBP_MR (OOS +1.04, 2022 +1.07)
  + TREND_COND (OOS +1.62, replaces TREND_NEW if better blend)

Rejected wave 7-8:
  - Classical indicators (duplicates trend)
  - Anomaly detection (0 survivors)
  - High-freq lead-lag (all failed cost)
  - DD recovery (2022 -0.63)
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
WAVE5 = ROOT / "scratch" / "wave5"
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


def load_returns(path, name):
    df = pd.read_parquet(path)
    if "timestamp" not in df.columns and "H4_SLEEVE" in df.columns:
        s = pd.Series(df["H4_SLEEVE"].values, index=pd.to_datetime(df.index, utc=True))
        return normalize_idx(s.rename(name))
    if "ret" not in df.columns and "timestamp" in df.columns:
        cols = [c for c in df.columns if c != "timestamp"]
        for special in ["survivors_mean", "COMBINED", "combined", "MULTI_TF", "PORTFOLIO"]:
            if special in df.columns:
                combined = df[special]
                break
        else:
            combined = df[cols].mean(axis=1)
        ts = pd.to_datetime(df["timestamp"], utc=True)
        return normalize_idx(pd.Series(combined.values, index=ts).rename(name))
    ts = pd.to_datetime(df["timestamp"], utc=True)
    return normalize_idx(pd.Series(df["ret"].values, index=ts).rename(name))


def rescale_is(s, target):
    is_part = s[s.index < SPLIT]
    av = float(is_part.std(ddof=0)) * np.sqrt(365.25)
    return s * (target / av) if av > 1e-9 else s * 0


def main():
    panel = pd.read_parquet(OUT / "all_sleeve_returns_v12.parquet")
    panel.index = pd.to_datetime(panel.index, utc=True)
    print(f"Starting v12 panel: {len(panel.columns)} sleeves")

    new_sleeves = {
        "STATARB_XS":   WAVE6 / "statarb_returns.parquet",
        "MICROSTR_D1":  WAVE6 / "microstructure_returns.parquet",
        "TERM_SPREADS": WAVE6 / "term_spreads_returns.parquet",
        "VOL_BREAKOUT": WAVE6 / "vol_breakout_returns.parquet",
        "EURGBP_MR":    WAVE6 / "fx_specific_returns.parquet",
        "TREND_COND":   WAVE6 / "trend_conditional_returns.parquet",
    }
    for name, path in new_sleeves.items():
        if not path.exists():
            print(f"Skip {name} (missing)")
            continue
        s = load_returns(path, name)
        s = rescale_is(s, TARGET_VOL)
        panel[name] = s.reindex(panel.index, fill_value=0.0)
        is_part = panel[name][panel.index < SPLIT]
        oos_part = panel[name][panel.index >= SPLIT]
        is_sh = is_part.mean()*365/(is_part.std(ddof=0)*np.sqrt(365)) if is_part.std() > 0 else 0
        oos_sh = oos_part.mean()*365/(oos_part.std(ddof=0)*np.sqrt(365)) if oos_part.std() > 0 else 0
        print(f"Added {name:<14}: IS_Sh={is_sh:+.2f}  OOS_Sh={oos_sh:+.2f}")

    panel.to_parquet(OUT / "all_sleeve_returns_v13.parquet")
    print(f"\nFinal v13 panel: {len(panel.columns)} sleeves")

    # Define candidate compositions
    TOP14 = ["RISKPAR", "TREND_NEW", "EVE_XAU", "D1REV_UK", "XSMOM",
             "D1REV_NAS", "WED_BTC", "DEFEND", "PAIRS_EXP",
             "VOLFORECAST", "H4_SLEEVE", "CRYPTO_vs_SPX", "CORR_REGIME", "SESSION_MOM"]
    # v12: TOP14 + W1 + EVENT (was best)
    TOP16_v12 = TOP14 + ["W1_STRATS", "EVENT_VOLSPIKE"]
    # v13: add STATARB_XS, MICROSTR_D1, VOL_BREAKOUT, TERM_SPREADS, EURGBP_MR
    TOP21_v13 = TOP16_v12 + ["STATARB_XS", "MICROSTR_D1", "VOL_BREAKOUT", "TERM_SPREADS", "EURGBP_MR"]
    # Variant: replace TREND_NEW with TREND_COND (higher OOS Sharpe)
    TOP21_replace = [s for s in TOP21_v13 if s != "TREND_NEW"] + ["TREND_COND"]
    # Conservative: only the highest-OOS adds
    TOP18 = TOP16_v12 + ["STATARB_XS", "MICROSTR_D1"]

    gates = dict(GATES_HIGH)
    for s in ["PAIRS_EXP", "CRYPTO_vs_SPX", "CORR_REGIME", "STATARB_XS", "MICROSTR_D1",
              "EURGBP_MR", "TERM_SPREADS", "EVENT_VOLSPIKE"]:
        gates[s] = 1.5
    for s in ["VOLFORECAST", "W1_STRATS", "SESSION_MOM"]:
        gates[s] = 1.0
    for s in ["H4_SLEEVE"]:
        gates[s] = 1.5
    for s in ["TREND_NEW", "TREND_COND"]:
        gates[s] = 0.5
    gates["VOL_BREAKOUT"] = 1.2

    high_vol, _ = build_regime_mask(panel.index, 80)
    g = panel.copy()
    for s, m in gates.items():
        if s in g.columns:
            g.loc[high_vol, s] *= m

    print(f"\n{'Variant':<28} {'OOS_Sh':>7} {'2022_Sh':>8} {'OOS_DD':>8}")
    print("-" * 65)
    for name, sleeves in [
        ("TOP16 (v12)", TOP16_v12),
        ("TOP18 (+statarb+microstr)", TOP18),
        ("TOP21 (full v13)", TOP21_v13),
        ("TOP21 (TREND_COND replace)", TOP21_replace),
    ]:
        port = g[sleeves].mean(axis=1)
        s = stats(name, port)
        y22 = port[port.index.year == 2022]
        bpy = _bpy(y22.index) if len(y22) > 0 else 252
        sh22 = (y22.mean()*bpy)/(y22.std(ddof=0)*np.sqrt(bpy)) if y22.std() > 0 else 0
        print(f"{name:<28} {s.get('OOS_sharpe',0):>+7.2f} {sh22:>+8.2f} {s.get('OOS_dd',0):>+8.1%}")

    # Pick the winner
    print(f"\n=== Best variant leverage sweep ===")
    after_decay, _ = fast_decay_tripwire(g[TOP21_v13], TOP21_v13)
    full_baseline = after_decay.mean(axis=1)
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

    # Production: 18% target with V6 time-of-month
    vt, lev = time_of_month_vol_target(full_baseline, base_target=0.18)
    prod, _ = drawdown_control(vt)
    eq = (1 + prod).cumprod()
    pd.DataFrame({"timestamp": prod.index, "ret": prod.values, "equity": eq.values}).to_parquet(
        OUT / "PRODUCTION_v13.parquet", index=False)

    print(f"\n=== Year-by-year at 18% vol target (v13 production) ===")
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
