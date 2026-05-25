"""Master v7 — production model.

Composition: TOP8 + PAIRS_v2 (cointegration-filtered pairs).
Overlays: regime gate → fast decay tripwire → vol-target 10% → DD control.

Why this version is the production model:
  * +0.46 Sharpe IS improvement in 2022 (+0.52 → +0.98) by adding PAIRS_v2.
  * Same OOS Sharpe (+2.89) as TOP8 alone — pairs added diversification without OOS drag.
  * Cointegration ADF gate prevents the v1 pairs-trade catastrophe (β regime breaks).
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

OUT = Path(__file__).resolve().parent
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")

# Composition: TOP8 + PAIRS
TOP9 = ["RISKPAR", "TSMOM", "EVE_XAU", "D1REV_UK", "XSMOM",
        "D1REV_NAS", "WED_BTC", "DEFEND", "PAIRS_v2"]


def main():
    panel = pd.read_parquet(OUT / "all_sleeve_returns_v5.parquet")
    panel.index = pd.to_datetime(panel.index, utc=True)
    panel = panel[TOP9]

    # Regime mask
    high_vol, cutoff = build_regime_mask(panel.index, 80)
    print(f"SPX regime cutoff: {cutoff:.4f}, OOS high-vol bars: "
          f"{high_vol[panel.index >= SPLIT].sum()}/{(panel.index >= SPLIT).sum()}")

    # Apply per-sleeve regime gates
    gates = dict(GATES_HIGH)
    gates["PAIRS_v2"] = 1.5  # pairs are mean-rev — amplify in high vol
    g = panel.copy()
    for sleeve, mult in gates.items():
        if sleeve in g.columns:
            g.loc[high_vol, sleeve] *= mult

    # Baseline portfolio (equal-weight, regime-gated)
    baseline = g.mean(axis=1)

    # Step 1: fast decay tripwire
    after_decay, decay_log = fast_decay_tripwire(g, TOP9)
    decay_log.to_csv(OUT / "v7_decay_log.csv", index=False)
    tripwired = after_decay.mean(axis=1)

    # Step 2: vol-target 10%
    vt, lev = vol_target_overlay(tripwired, target_vol=0.10)

    # Step 3: drawdown control
    final, dd_mult = drawdown_control(vt)

    variants = {
        "TOP9 baseline (regime-gated)": baseline,
        "+ decay tripwire":              tripwired,
        "+ vol-target 10%":              vt,
        "+ DD control (PRODUCTION)":     final,
    }

    print(f"\n{'Variant':<35} {'FULL':>6} {'IS':>6} {'OOS':>6} {'2022':>7} {'OOS_vol':>8} {'OOS_DD':>7}")
    print("-" * 80)
    rows = []
    for name, r in variants.items():
        s = stats(name, r)
        y22 = r[r.index.year == 2022]
        bpy = _bpy(y22.index) if len(y22) > 0 else 252
        sh22 = (y22.mean() * bpy) / (y22.std(ddof=0) * np.sqrt(bpy)) if y22.std() > 0 else 0
        rows.append({**s, "Sh2022": sh22})
        print(f"{name:<35} {s.get('FULL_sharpe',0):>+6.2f} {s.get('IS_sharpe',0):>+6.2f} "
              f"{s.get('OOS_sharpe',0):>+6.2f} {sh22:>+7.2f} {s.get('OOS_vol',0):>8.1%} {s.get('OOS_dd',0):>+7.1%}")
    pd.DataFrame(rows).to_csv(OUT / "master_v7_variants.csv", index=False)

    # Year-by-year
    yr = {}
    for name, r in variants.items():
        yr_row = {}
        for year, sub in r.groupby(r.index.year):
            if len(sub) < 50: continue
            bpy = _bpy(sub.index)
            sh = sub.mean() * bpy / (sub.std(ddof=0) * np.sqrt(bpy)) if sub.std() > 0 else 0
            yr_row[year] = sh
        yr[name] = yr_row
    yr_df = pd.DataFrame(yr).T
    print(f"\nYear-by-year Sharpe:")
    print(yr_df.round(2).to_string())
    yr_df.to_csv(OUT / "master_v7_yearly.csv")

    # Save PRODUCTION
    eq = (1 + final).cumprod()
    pd.DataFrame({"timestamp": final.index, "ret": final.values, "equity": eq.values}).to_parquet(
        OUT / "PRODUCTION_FINAL.parquet", index=False)
    print(f"\nPRODUCTION FINAL saved.")
    print(f"  Total return:    {(eq.iloc[-1]-1)*100:+.1f}%")
    print(f"  Annualized:      {(eq.iloc[-1]**(365.25/((eq.index[-1]-eq.index[0]).days))-1)*100:+.1f}%")
    print(f"  OOS slice ret:   {(1+final[final.index>=SPLIT]).cumprod().iloc[-1]*100-100:+.1f}%")
    print(f"  Avg leverage:    {lev.mean():.2f}x")

    # Save sleeve weights timeline
    decay_summary = decay_log.groupby("sleeve").apply(
        lambda d: (d["state_before"] == "ON").mean()).rename("pct_ON_time")
    print(f"\nPercent of monthly checks each sleeve was ON:")
    for sleeve, pct in decay_summary.items():
        print(f"  {sleeve:<12} {pct:.1%}")


if __name__ == "__main__":
    main()
