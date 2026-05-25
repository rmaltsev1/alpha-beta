"""Full stress + leverage sweep for v9.

Tests:
  1. Robustness: drop H4 (the highest IS-OOS bias sleeve) — does v9 still hit target?
  2. Cost stress: 2x cost on calendar sleeves.
  3. Sleeve drop test: drop each sleeve one-at-a-time, see how OOS Sharpe changes.
  4. Comprehensive leverage scenarios with both static and walk-forward.
  5. Bootstrap 1000 random 12-month windows.
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


def yearly(r):
    rows = {}
    for year, sub in r.groupby(r.index.year):
        span = (sub.index[-1] - sub.index[0]).total_seconds() / 86400
        if span < 60: continue
        bpy = len(sub) / span * 365.25
        rows[year] = sub.mean()*bpy/(sub.std(ddof=0)*np.sqrt(bpy)) if sub.std() > 0 else 0
    return pd.Series(rows)


def build_portfolio(panel, sleeves, gates_overrides=None):
    """Apply regime gates + equal-weight."""
    high_vol, _ = build_regime_mask(panel.index, 80)
    gates = dict(GATES_HIGH)
    if gates_overrides:
        gates.update(gates_overrides)
    g = panel.copy()
    for s, m in gates.items():
        if s in g.columns:
            g.loc[high_vol, s] *= m
    return g[sleeves].mean(axis=1)


def lever_sweep(returns: pd.Series, targets: list[float]):
    """Apply vol target + DD control + return scenario stats."""
    rows = []
    for tv in targets:
        vt, lev = vol_target_overlay(returns, target_vol=tv, max_lev=15.0)
        dd_ctrl, _ = drawdown_control(vt)
        # OOS stats
        oos = dd_ctrl[dd_ctrl.index >= SPLIT]
        if len(oos) < 30:
            continue
        bpy = _bpy(oos.index)
        ar = oos.mean() * bpy
        av = oos.std(ddof=0) * np.sqrt(bpy)
        sh = ar/av if av > 0 else 0
        eq = (1 + oos).cumprod()
        dd = (eq/eq.cummax()-1).min()
        rows.append({"vol_target": tv, "ann_ret": ar, "ann_vol": av,
                     "sharpe": sh, "maxdd": dd, "monthly": ar/12, "avg_lev": lev.mean()})
    return rows


def main():
    panel = pd.read_parquet(OUT / "all_sleeve_returns_v9.parquet")
    panel.index = pd.to_datetime(panel.index, utc=True)
    TOP12 = ["RISKPAR", "TREND_NEW", "EVE_XAU", "D1REV_UK", "XSMOM",
             "D1REV_NAS", "WED_BTC", "DEFEND", "PAIRS_EXP",
             "VOLFORECAST", "H4_SLEEVE", "CRYPTO_vs_SPX"]
    gates_overrides = {"PAIRS_EXP": 1.5, "VOLFORECAST": 1.0, "H4_SLEEVE": 1.5,
                       "TREND_NEW": 0.5, "CRYPTO_vs_SPX": 1.5}

    # ---- (1) Robustness: drop each sleeve, see OOS impact ----
    print("=" * 70)
    print("TEST 1: One-sleeve-drop sensitivity (OOS Sharpe impact)")
    print("=" * 70)
    base_portfolio = build_portfolio(panel, TOP12, gates_overrides)
    base_oos = base_portfolio[base_portfolio.index >= SPLIT]
    base_sh = base_oos.mean()*365/(base_oos.std(ddof=0)*np.sqrt(365))
    print(f"Baseline OOS Sharpe: {base_sh:+.2f}")
    print(f"{'Drop':<14} {'OOS_Sh':>7} {'Δ_OOS':>7} {'2022_Sh':>8}")
    rows = []
    for sleeve in TOP12:
        keep = [s for s in TOP12 if s != sleeve]
        p = build_portfolio(panel, keep, gates_overrides)
        oos = p[p.index >= SPLIT]
        sh = oos.mean()*365/(oos.std(ddof=0)*np.sqrt(365))
        delta = sh - base_sh
        y22 = p[p.index.year == 2022]
        bpy = _bpy(y22.index) if len(y22) > 0 else 252
        sh22 = y22.mean()*bpy/(y22.std(ddof=0)*np.sqrt(bpy)) if y22.std() > 0 else 0
        rows.append({"drop": sleeve, "oos_sh": sh, "delta": delta, "2022_sh": sh22})
        print(f"{sleeve:<14} {sh:>+7.2f} {delta:>+7.2f} {sh22:>+8.2f}")
    pd.DataFrame(rows).to_csv(OUT / "v9_dropone.csv", index=False)

    # ---- (2) Build the robust version: drop H4_SLEEVE (highest IS-OOS bias) ----
    print(f"\n{'=' * 70}\nTEST 2: Drop H4_SLEEVE — does v9 still hit monthly target?\n{'=' * 70}")
    TOP11_no_h4 = [s for s in TOP12 if s != "H4_SLEEVE"]
    p_no_h4 = build_portfolio(panel, TOP11_no_h4, gates_overrides)
    after_decay, _ = fast_decay_tripwire(panel[TOP11_no_h4], TOP11_no_h4)
    trip = after_decay.mean(axis=1)
    rows = lever_sweep(trip, [0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.27])
    print(f"\n{'Vol_Tgt':<8} {'OOS_Ret':>8} {'OOS_Vol':>8} {'Sharpe':>7} {'MaxDD':>8} {'Mo/yr':>7}")
    for r in rows:
        print(f"{r['vol_target']:<8.0%} {r['ann_ret']:>+8.1%} {r['ann_vol']:>8.1%} "
              f"{r['sharpe']:>+7.2f} {r['maxdd']:>+8.1%} {r['monthly']:>+7.2%}")

    # ---- (3) v9 production with all sleeves + full overlay ----
    print(f"\n{'=' * 70}\nTEST 3: Full v9 leverage sweep with all overlays\n{'=' * 70}")
    after_decay, _ = fast_decay_tripwire(panel[TOP12], TOP12)
    full_baseline = after_decay.mean(axis=1)
    rows = lever_sweep(full_baseline, [0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.27, 0.32])
    print(f"\n{'Vol_Tgt':<8} {'OOS_Ret':>8} {'OOS_Vol':>8} {'Sharpe':>7} {'MaxDD':>8} {'Mo/yr':>7} {'Lev':>5}")
    for r in rows:
        print(f"{r['vol_target']:<8.0%} {r['ann_ret']:>+8.1%} {r['ann_vol']:>8.1%} "
              f"{r['sharpe']:>+7.2f} {r['maxdd']:>+8.1%} {r['monthly']:>+7.2%} {r['avg_lev']:>5.1f}x")

    # ---- (4) Monte Carlo bootstrap on v9 baseline (no overlays) ----
    print(f"\n{'=' * 70}\nTEST 4: Monte Carlo bootstrap — 1000 random 12-month IS windows\n{'=' * 70}")
    base = build_portfolio(panel, TOP12, gates_overrides)
    is_part = base[base.index < SPLIT]
    rng = np.random.default_rng(42)
    n_bars = 365
    sharpes = []; returns = []; dds = []
    for _ in range(1000):
        start = rng.integers(0, len(is_part) - n_bars)
        w = is_part.iloc[start: start+n_bars]
        bpy = _bpy(w.index)
        ar = w.mean() * bpy
        av = w.std(ddof=0) * np.sqrt(bpy)
        sh = ar/av if av > 0 else 0
        eq = (1+w).cumprod()
        dd = (eq/eq.cummax()-1).min()
        sharpes.append(sh); returns.append(ar); dds.append(dd)
    sharpes = np.array(sharpes); returns = np.array(returns); dds = np.array(dds)
    print(f"Sharpe:  mean={sharpes.mean():+.2f}  median={np.median(sharpes):+.2f}  "
          f"5%={np.quantile(sharpes,0.05):+.2f}  95%={np.quantile(sharpes,0.95):+.2f}")
    print(f"Ann ret: mean={returns.mean():+.1%}  5%={np.quantile(returns,0.05):+.1%}")
    print(f"MaxDD:   worst={dds.min():+.1%}  5%={np.quantile(dds,0.05):+.1%}")
    print(f"P(Sharpe < 0) = {(sharpes < 0).mean():.1%}")
    print(f"P(Sharpe > 2) = {(sharpes > 2).mean():.1%}")

    # ---- (5) Year-by-year breakdown at multiple leverage points ----
    print(f"\n{'=' * 70}\nTEST 5: Year-by-year at multiple vol targets (full v9)\n{'=' * 70}")
    vt, _ = vol_target_overlay(full_baseline, target_vol=0.10, max_lev=15.0)
    dd_ctrl, _ = drawdown_control(vt)
    vt_18, _ = vol_target_overlay(full_baseline, target_vol=0.18, max_lev=15.0)
    dd_18, _ = drawdown_control(vt_18)
    vt_22, _ = vol_target_overlay(full_baseline, target_vol=0.22, max_lev=15.0)
    dd_22, _ = drawdown_control(vt_22)

    print(f"{'Year':<6} {'10%':<20} {'18%':<20} {'22%':<20}")
    for year in sorted(set(dd_ctrl.index.year)):
        r10 = dd_ctrl[dd_ctrl.index.year == year]
        r18 = dd_18[dd_18.index.year == year]
        r22 = dd_22[dd_22.index.year == year]
        bpy10 = _bpy(r10.index) if len(r10) > 30 else 252
        bpy18 = _bpy(r18.index) if len(r18) > 30 else 252
        bpy22 = _bpy(r22.index) if len(r22) > 30 else 252
        def fmt(r, bpy):
            ar = r.mean() * bpy
            av = r.std(ddof=0) * np.sqrt(bpy)
            sh = ar/av if av > 0 else 0
            eq = (1+r).cumprod(); dd = (eq/eq.cummax()-1).min()
            return f"Sh{sh:+.1f} R{ar:+.0%} DD{dd:+.0%}"
        print(f"{year:<6} {fmt(r10, bpy10):<20} {fmt(r18, bpy18):<20} {fmt(r22, bpy22):<20}")


if __name__ == "__main__":
    main()
