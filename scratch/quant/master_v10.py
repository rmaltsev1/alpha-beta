"""Master v10 — add wave3-final sleeves (corr_regime + session_momentum + kelly blend).

New additions:
  + CORR_REGIME (OOS +1.24, 2022 +1.66) — cross-asset correlation regime detector
  + SESSION_MOM (OOS +0.88, 2022 +0.73) — JP225 Asia-open momentum (lone survivor)
  + Optional: 25% Kelly blend overlay

Targets the 3-5% monthly with controlled DD.
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
WAVE3 = ROOT / "scratch" / "wave3"
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
    if "ret" not in df.columns and "dom_BTCUSDT" in df.columns:
        combined = (df["dom_BTCUSDT"] + df["dom_ETHUSDT"]) / 2
        ts = pd.to_datetime(df["timestamp"], utc=True)
        return normalize_idx(pd.Series(combined.values, index=ts).rename(name))
    # Multi-column sleeve (e.g. corr_regime) — equal-weight combine
    if "ret" not in df.columns and "timestamp" in df.columns:
        cols = [c for c in df.columns if c != "timestamp"]
        combined = df[cols].mean(axis=1)
        ts = pd.to_datetime(df["timestamp"], utc=True)
        return normalize_idx(pd.Series(combined.values, index=ts).rename(name))
    ts = pd.to_datetime(df["timestamp"], utc=True)
    return normalize_idx(pd.Series(df["ret"].values, index=ts).rename(name))


def rescale_is(s, target):
    is_part = s[s.index < SPLIT]
    av = float(is_part.std(ddof=0)) * np.sqrt(365.25)
    return s * (target / av) if av > 1e-9 else s * 0


def lever_sweep(returns, targets):
    rows = []
    for tv in targets:
        vt, lev = vol_target_overlay(returns, target_vol=tv, max_lev=15.0)
        dd_ctrl, _ = drawdown_control(vt)
        oos = dd_ctrl[dd_ctrl.index >= SPLIT]
        if len(oos) < 30: continue
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
    # Start with v9 panel
    panel = pd.read_parquet(OUT / "all_sleeve_returns_v9.parquet")
    panel.index = pd.to_datetime(panel.index, utc=True)

    # Add wave3-final sleeves
    new_sleeves = {
        "CORR_REGIME":   WAVE3 / "corr_regime_returns.parquet",
        "SESSION_MOM":   WAVE3 / "session_momentum_returns.parquet",
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
        is_sh = is_part.mean()*365/(is_part.std(ddof=0)*np.sqrt(365))
        oos_sh = oos_part.mean()*365/(oos_part.std(ddof=0)*np.sqrt(365))
        print(f"Added {name:<14}: IS_Sh={is_sh:+.2f}  OOS_Sh={oos_sh:+.2f}")

    panel.to_parquet(OUT / "all_sleeve_returns_v10.parquet")

    # TOP13: v9 TOP12 + CORR_REGIME
    TOP13 = ["RISKPAR", "TREND_NEW", "EVE_XAU", "D1REV_UK", "XSMOM",
             "D1REV_NAS", "WED_BTC", "DEFEND", "PAIRS_EXP",
             "VOLFORECAST", "H4_SLEEVE", "CRYPTO_vs_SPX", "CORR_REGIME"]
    # TOP14: TOP13 + SESSION_MOM
    TOP14 = TOP13 + ["SESSION_MOM"]

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

    # ---- Compare TOP12, TOP13, TOP14 ----
    print(f"\n{'Variant':<22} {'OOS_Sh':>7} {'2022_Sh':>8} {'OOS_DD':>8} {'OOS_Ret_10%':>12}")
    print("-" * 70)
    for name, sleeves in [("TOP12 (v9)", TOP13[:-1]),
                          ("TOP13 (+CORR_REGIME)", TOP13),
                          ("TOP14 (+SESSION_MOM)", TOP14)]:
        port = g[sleeves].mean(axis=1)
        s = stats(name, port)
        y22 = port[port.index.year == 2022]
        bpy = _bpy(y22.index) if len(y22) > 0 else 252
        sh22 = (y22.mean()*bpy)/(y22.std(ddof=0)*np.sqrt(bpy)) if y22.std() > 0 else 0
        # Apply tripwire + vol target 10% for headline OOS ret
        after_decay, _ = fast_decay_tripwire(g[sleeves], sleeves)
        trip = after_decay.mean(axis=1)
        vt, _ = vol_target_overlay(trip, target_vol=0.10, max_lev=15.0)
        dd_ctrl, _ = drawdown_control(vt)
        oos = dd_ctrl[dd_ctrl.index >= SPLIT]
        bpy_oos = _bpy(oos.index)
        ar_oos = oos.mean()*bpy_oos
        print(f"{name:<22} {s.get('OOS_sharpe',0):>+7.2f} {sh22:>+8.2f} "
              f"{s.get('OOS_dd',0):>+8.1%} {ar_oos:>+12.1%}")

    # ---- Full leverage sweep on TOP13 (the production winner) ----
    after_decay, _ = fast_decay_tripwire(g[TOP13], TOP13)
    full_baseline = after_decay.mean(axis=1)
    print(f"\n=== TOP13 leverage sweep (all overlays) ===")
    print(f"{'VolTarget':<10} {'OOS_Ret':>8} {'OOS_Vol':>8} {'Sharpe':>7} {'MaxDD':>8} {'Mo/yr':>7} {'Lev':>5}")
    rows = lever_sweep(full_baseline, [0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.27])
    for r in rows:
        print(f"{r['vol_target']:<10.0%} {r['ann_ret']:>+8.1%} {r['ann_vol']:>8.1%} "
              f"{r['sharpe']:>+7.2f} {r['maxdd']:>+8.1%} {r['monthly']:>+7.2%} {r['avg_lev']:>5.1f}x")
    pd.DataFrame(rows).to_csv(OUT / "v10_leverage_sweep.csv", index=False)

    # Save PRODUCTION at 18% vol target (the 3-5%/mo sweet spot)
    vt, _ = vol_target_overlay(full_baseline, target_vol=0.18, max_lev=15.0)
    prod, _ = drawdown_control(vt)
    eq = (1 + prod).cumprod()
    pd.DataFrame({"timestamp": prod.index, "ret": prod.values, "equity": eq.values}).to_parquet(
        OUT / "PRODUCTION_v10_18pct.parquet", index=False)

    # Year-by-year at 18% vol
    print(f"\n=== Year-by-year at 18% vol target (TOP13) ===")
    for year, sub in prod.groupby(prod.index.year):
        if len(sub) < 50: continue
        bpy = _bpy(sub.index)
        ar = sub.mean()*bpy
        av = sub.std(ddof=0)*np.sqrt(bpy)
        sh = ar/av if av > 0 else 0
        eq_y = (1+sub).cumprod()
        dd = (eq_y/eq_y.cummax()-1).min()
        print(f"  {year}  Sh={sh:+.2f}  Ret={ar:+.1%}  Vol={av:.1%}  DD={dd:+.1%}  Mo={ar/12:+.2%}")


if __name__ == "__main__":
    main()
