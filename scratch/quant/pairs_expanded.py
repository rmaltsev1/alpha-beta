"""Expanded pairs search — test all 78 unique pairs among the 13 symbols.

Same machinery as pairs_v2.py: cointegration ADF gate + walk-forward β +
z-score entries. Faster: vectorize the inner loop where possible.

Reports IS/OOS Sharpe + 2022 Sharpe per pair, then aggregates survivors.
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scratch" / "quant"))

from alphabeta import get_candles, ALL_SYMBOLS
from pairs_v2 import pair_strategy, stats, _bpy, _scale_to_target

OUT = Path(__file__).resolve().parent
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
TARGET_VOL = 0.05


def main():
    pairs = list(combinations(ALL_SYMBOLS, 2))
    print(f"Testing {len(pairs)} pair combinations...")
    print(f"{'Pair':<28} {'IS_Sh':>6} {'OOS_Sh':>7} {'2022_Sh':>8} {'#Trd':>5} {'scale':>6}")
    print("-" * 75)
    rows = []
    streams = {}
    for i, (a, b) in enumerate(pairs):
        try:
            rets, diag = pair_strategy(a, b)
            scaled, k = _scale_to_target(rets, TARGET_VOL)
            s = stats(f"{a}-{b}", scaled)
            n_trades = int((np.abs(np.diff(diag["pos_spread"], prepend=0)) > 0).sum() // 2)
            # 2022 sharpe
            y22 = scaled[scaled.index.year == 2022]
            if len(y22) > 10:
                bpy = _bpy(y22.index)
                ar = float(y22.mean()) * bpy
                av = float(y22.std(ddof=0)) * np.sqrt(bpy)
                sh22 = ar / av if av > 0 else 0
            else:
                sh22 = 0
            rows.append({"pair": f"{a}-{b}", "scale": k, "n_trades": n_trades,
                         "2022_sharpe": sh22, **s})
            streams[f"{a}-{b}"] = scaled
            mark = " *" if (s.get("IS_sharpe", 0) > 0.3 and s.get("OOS_sharpe", 0) > 0) else ""
            print(f"{a}-{b:<14} {s.get('IS_sharpe',0):>+6.2f} {s.get('OOS_sharpe',0):>+7.2f} "
                  f"{sh22:>+8.2f} {n_trades:>5d} {k:>6.2f}{mark}")
        except Exception as e:
            print(f"{a}-{b}: ERROR {type(e).__name__}: {e}")

    df = pd.DataFrame(rows).sort_values("OOS_sharpe", ascending=False)
    df.to_csv(OUT / "pairs_expanded_results.csv", index=False)

    # Survivors: IS Sharpe > 0.3 AND OOS > 0 AND #trades >= 3
    survivors = df[(df["IS_sharpe"] > 0.3) & (df["OOS_sharpe"] > 0) & (df["n_trades"] >= 3)]
    print(f"\nSurvivors (IS_Sh > 0.3, OOS > 0, ≥3 trades): {len(survivors)} pairs")
    print(survivors[["pair", "IS_sharpe", "OOS_sharpe", "2022_sharpe", "n_trades"]].to_string(index=False))

    # Combine survivors equal-weight
    if len(survivors) > 0:
        keep = list(survivors["pair"])
        combined = pd.concat({k: streams[k] for k in keep}, axis=1, sort=True).fillna(0).mean(axis=1)
        s = stats("PAIRS_EXP", combined)
        print(f"\nCombined sleeve ({len(keep)} pairs):")
        for tag in ["FULL", "IS", "OOS"]:
            print(f"  {tag:<4} Sharpe={s.get(f'{tag}_sharpe',0):+.2f}  Return={s.get(f'{tag}_ret',0):+.2%}")
        # Year-by-year
        for year, sub in combined.groupby(combined.index.year):
            if len(sub) < 50: continue
            bpy = _bpy(sub.index)
            sh = sub.mean() * bpy / (sub.std(ddof=0) * np.sqrt(bpy)) if sub.std() > 0 else 0
            print(f"  {year}  Sharpe={sh:+.2f}")
        pd.DataFrame({"timestamp": combined.index, "ret": combined.values}).to_parquet(
            OUT / "pairs_expanded_returns.parquet", index=False)


if __name__ == "__main__":
    main()
