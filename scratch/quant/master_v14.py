"""Master v14 — apply stops to TREND_NEW/PAIRS_EXP/RISKPAR + add MULTIDAY.

Wave 8 wins:
  - Stops on TREND_NEW (+0.34 OOS Sharpe), PAIRS_EXP (+0.32), RISKPAR (+0.23)
  - MULTIDAY_PATTERNS (OOS +0.64, 2022 +2.80, ~0 corr with MICROSTRUCTURE)

Wave 8 rejections:
  - ML non-linear (still doesn't beat EW)
  - Strategy crowding (none beat EW by 0.15)
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


def rescale_is(s, target):
    is_part = s[s.index < SPLIT]
    av = float(is_part.std(ddof=0)) * np.sqrt(365.25)
    return s * (target / av) if av > 1e-9 else s * 0


def main():
    panel = pd.read_parquet(OUT / "all_sleeve_returns_v13.parquet")
    panel.index = pd.to_datetime(panel.index, utc=True)

    # Load stops returns and replace
    stops_df = pd.read_parquet(WAVE6 / "stops_returns.parquet")
    stops_df.index = pd.to_datetime(stops_df.index, utc=True)
    replacements = {
        "TREND_NEW":  "TREND_NEW_tpsl_3.0x_1.5x",
        "PAIRS_EXP":  "PAIRS_EXP_atr_2.0x",
        "RISKPAR":    "RISKPAR_tpsl_3.0x_1.5x",
    }
    for old_name, stops_col in replacements.items():
        s = pd.Series(stops_df[stops_col].values, index=stops_df.index)
        s = normalize_idx(s.rename(old_name))
        s = rescale_is(s, TARGET_VOL)
        panel[old_name] = s.reindex(panel.index, fill_value=0.0)
        oos = panel[old_name][panel.index >= SPLIT]
        oos_sh = oos.mean()*365/(oos.std(ddof=0)*np.sqrt(365)) if oos.std() > 0 else 0
        print(f"Replaced {old_name:<12} with stops: OOS_Sh={oos_sh:+.2f}")

    # Add MULTIDAY_PATTERNS
    md_path = WAVE6 / "multiday_returns.parquet"
    if md_path.exists():
        md_df = pd.read_parquet(md_path)
        if "timestamp" in md_df.columns:
            cols = [c for c in md_df.columns if c != "timestamp"]
            # use 'survivors_mean' if present, else mean
            if "survivors_mean" in md_df.columns:
                combined = md_df["survivors_mean"]
            else:
                combined = md_df[cols].mean(axis=1)
            ts = pd.to_datetime(md_df["timestamp"], utc=True)
            s = normalize_idx(pd.Series(combined.values, index=ts).rename("MULTIDAY"))
            s = rescale_is(s, TARGET_VOL)
            panel["MULTIDAY"] = s.reindex(panel.index, fill_value=0.0)
            oos = panel["MULTIDAY"][panel.index >= SPLIT]
            oos_sh = oos.mean()*365/(oos.std(ddof=0)*np.sqrt(365)) if oos.std() > 0 else 0
            print(f"Added MULTIDAY: OOS_Sh={oos_sh:+.2f}")

    panel.to_parquet(OUT / "all_sleeve_returns_v14.parquet")

    # TOP22 = TOP21 + MULTIDAY
    TOP22 = ["RISKPAR", "TREND_NEW", "EVE_XAU", "D1REV_UK", "XSMOM",
             "D1REV_NAS", "WED_BTC", "DEFEND", "PAIRS_EXP",
             "VOLFORECAST", "H4_SLEEVE", "CRYPTO_vs_SPX",
             "CORR_REGIME", "SESSION_MOM",
             "W1_STRATS", "EVENT_VOLSPIKE",
             "STATARB_XS", "MICROSTR_D1", "VOL_BREAKOUT",
             "TERM_SPREADS", "EURGBP_MR", "MULTIDAY"]

    gates = dict(GATES_HIGH)
    for s in ["PAIRS_EXP", "CRYPTO_vs_SPX", "CORR_REGIME", "STATARB_XS",
              "MICROSTR_D1", "EURGBP_MR", "TERM_SPREADS", "EVENT_VOLSPIKE", "MULTIDAY"]:
        gates[s] = 1.5
    for s in ["VOLFORECAST", "W1_STRATS", "SESSION_MOM"]:
        gates[s] = 1.0
    gates["H4_SLEEVE"] = 1.5
    gates["TREND_NEW"] = 0.5
    gates["VOL_BREAKOUT"] = 1.2

    high_vol, _ = build_regime_mask(panel.index, 80)
    g = panel.copy()
    for s, m in gates.items():
        if s in g.columns:
            g.loc[high_vol, s] *= m

    # Compare v13 → v14
    print(f"\n{'Variant':<26} {'OOS_Sh':>7} {'2022_Sh':>8} {'OOS_DD':>8}")
    print("-" * 60)
    for name, sleeves in [
        ("TOP21 v13 (no stops)", [s for s in TOP22 if s != "MULTIDAY"]),
        ("TOP21 v14 (with stops)", [s for s in TOP22 if s != "MULTIDAY"]),
        ("TOP22 v14 + MULTIDAY", TOP22),
    ]:
        port = g[sleeves].mean(axis=1)
        s_st = stats(name, port)
        y22 = port[port.index.year == 2022]
        bpy = _bpy(y22.index) if len(y22) > 0 else 252
        sh22 = (y22.mean()*bpy)/(y22.std(ddof=0)*np.sqrt(bpy)) if y22.std() > 0 else 0
        print(f"{name:<26} {s_st.get('OOS_sharpe',0):>+7.2f} {sh22:>+8.2f} {s_st.get('OOS_dd',0):>+8.1%}")

    after_decay, _ = fast_decay_tripwire(g[TOP22], TOP22)
    full_baseline = after_decay.mean(axis=1)

    print(f"\n=== Leverage sweep v14 ===")
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

    # Production at 18%
    vt, lev = time_of_month_vol_target(full_baseline, base_target=0.18)
    prod, _ = drawdown_control(vt)
    eq = (1 + prod).cumprod()
    pd.DataFrame({"timestamp": prod.index, "ret": prod.values, "equity": eq.values}).to_parquet(
        OUT / "PRODUCTION_v14.parquet", index=False)

    print(f"\n=== Year-by-year (PRODUCTION v14 at 18% vol) ===")
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
