"""Master v6 — last iteration. Test TOP8 vs TOP9 (add only one new sleeve),
also test 60/40 mix and 70/30 mix of TOP8 vs the new sleeves.

Goal: find the optimum point where new sleeves' 2022 benefit outweighs
their lower Sharpe.
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
    build_regime_mask, apply_regime_gate, stats, _bpy, GATES_HIGH,
)

OUT = Path(__file__).resolve().parent
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")


def main():
    panel = pd.read_parquet(OUT / "all_sleeve_returns_v5.parquet")
    panel.index = pd.to_datetime(panel.index, utc=True)
    high_vol, _ = build_regime_mask(panel.index, 80)

    # Extended gates
    gates = dict(GATES_HIGH)
    gates["PAIRS_v2"] = 1.5
    gates["ROTATION"] = 0.5
    g = panel.copy()
    for sleeve, mult in gates.items():
        if sleeve in g.columns:
            g.loc[high_vol, sleeve] *= mult

    TOP8 = ["RISKPAR","TSMOM","EVE_XAU","D1REV_UK","XSMOM","D1REV_NAS","WED_BTC","DEFEND"]
    TOP9_pairs = TOP8 + ["PAIRS_v2"]
    TOP9_rot   = TOP8 + ["ROTATION"]
    TOP10      = TOP8 + ["PAIRS_v2", "ROTATION"]

    variants = {
        "TOP8 + regime":              g[TOP8].mean(axis=1),
        "TOP9 + PAIRS":               g[TOP9_pairs].mean(axis=1),
        "TOP9 + ROTATION":            g[TOP9_rot].mean(axis=1),
        "TOP10 + both":               g[TOP10].mean(axis=1),
        # Mixes: 90% TOP8, 10% rotation (smaller weight)
        "TOP8(90%) + ROT(10%)":       0.9 * g[TOP8].mean(axis=1) + 0.1 * g["ROTATION"],
        "TOP8(85%) + PAIRS(15%)":     0.85 * g[TOP8].mean(axis=1) + 0.15 * g["PAIRS_v2"],
        "TOP8(80%) + ROT(10%) + PAIRS(10%)": 0.8 * g[TOP8].mean(axis=1) + 0.1 * g["ROTATION"] + 0.1 * g["PAIRS_v2"],
    }

    # Apply full overlay stack (tripwire + vol-target + DD) to each variant
    print(f"{'Variant':<40} {'FULL':>6} {'IS':>6} {'OOS':>6} {'2022':>7} {'OOS_vol':>8} {'OOS_DD':>7}")
    print("-" * 80)
    rows = []
    for name, r in variants.items():
        # No tripwire here, just regime-gated baseline (we're comparing sleeve mix)
        s = stats(name, r)
        y22 = r[r.index.year == 2022]
        bpy = _bpy(y22.index) if len(y22) > 0 else 252
        sh22 = (y22.mean() * bpy) / (y22.std(ddof=0) * np.sqrt(bpy)) if y22.std() > 0 else 0
        rows.append({**s, "Sh2022": sh22})
        print(f"{name:<40} {s.get('FULL_sharpe',0):>+6.2f} {s.get('IS_sharpe',0):>+6.2f} "
              f"{s.get('OOS_sharpe',0):>+6.2f} {sh22:>+7.2f} {s.get('OOS_vol',0):>8.1%} {s.get('OOS_dd',0):>+7.1%}")
    pd.DataFrame(rows).to_csv(OUT / "master_v6_variants.csv", index=False)

    print("\nYear-by-year Sharpe:")
    yr = {}
    for name, r in variants.items():
        yr_row = {}
        for year, sub in r.groupby(r.index.year):
            if len(sub) < 50: continue
            bpy = _bpy(sub.index)
            sh = sub.mean() * bpy / (sub.std(ddof=0) * np.sqrt(bpy)) if sub.std() > 0 else 0
            yr_row[year] = sh
        yr[name] = yr_row
    print(pd.DataFrame(yr).T.round(2).to_string())


if __name__ == "__main__":
    main()
