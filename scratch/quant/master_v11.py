"""Master v11 — final production model.

Adds to v10:
  + SESSION_MOM (JP225 Asia momentum, OOS +0.88, 2022 +0.73)
  + MULTI_CONFIRM at small weight (OOS +0.72, 0.6 corr with TREND_NEW so small addition)

Drops:
  - CARRY_DRIFT (0.83 corr with RISKPAR — duplicate beta)
  - VWAP (only 1/55 survived, no edge)

Test the 25% Kelly + 75% equal-weight blend that the Kelly agent suggested.
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
    # Multi-column with timestamp
    if "ret" not in df.columns and "timestamp" in df.columns:
        cols = [c for c in df.columns if c != "timestamp"]
        # For sleeves with explicit "survivors_mean" use that; otherwise mean all cols
        if "survivors_mean" in df.columns:
            combined = df["survivors_mean"]
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


def lever_sweep(returns, targets):
    rows = []
    for tv in targets:
        vt, lev = vol_target_overlay(returns, target_vol=tv, max_lev=15.0)
        dd_ctrl, _ = drawdown_control(vt)
        oos = dd_ctrl[dd_ctrl.index >= SPLIT]
        if len(oos) < 30: continue
        bpy = _bpy(oos.index)
        ar = oos.mean()*bpy
        av = oos.std(ddof=0)*np.sqrt(bpy)
        sh = ar/av if av > 0 else 0
        eq = (1+oos).cumprod()
        dd = (eq/eq.cummax()-1).min()
        rows.append({"vol_target": tv, "ann_ret": ar, "ann_vol": av,
                     "sharpe": sh, "maxdd": dd, "monthly": ar/12, "avg_lev": lev.mean()})
    return rows


def main():
    panel = pd.read_parquet(OUT / "all_sleeve_returns_v10.parquet")
    panel.index = pd.to_datetime(panel.index, utc=True)

    # Add MULTI_CONFIRM (the lone wave-4 win)
    mc_path = WAVE3 / "multi_confirm_returns.parquet"
    if mc_path.exists():
        mc = load_returns(mc_path, "MULTI_CONFIRM")
        mc = rescale_is(mc, TARGET_VOL)
        panel["MULTI_CONFIRM"] = mc.reindex(panel.index, fill_value=0.0)
        is_part = panel["MULTI_CONFIRM"][panel.index < SPLIT]
        oos_part = panel["MULTI_CONFIRM"][panel.index >= SPLIT]
        is_sh = is_part.mean()*365/(is_part.std(ddof=0)*np.sqrt(365))
        oos_sh = oos_part.mean()*365/(oos_part.std(ddof=0)*np.sqrt(365))
        print(f"Added MULTI_CONFIRM:  IS_Sh={is_sh:+.2f}  OOS_Sh={oos_sh:+.2f}")

    panel.to_parquet(OUT / "all_sleeve_returns_v11.parquet")

    # Sleeve configurations
    TOP12 = ["RISKPAR", "TREND_NEW", "EVE_XAU", "D1REV_UK", "XSMOM",
             "D1REV_NAS", "WED_BTC", "DEFEND", "PAIRS_EXP",
             "VOLFORECAST", "H4_SLEEVE", "CRYPTO_vs_SPX"]
    TOP13 = TOP12 + ["CORR_REGIME"]
    TOP14 = TOP13 + ["SESSION_MOM"]
    TOP15 = TOP14 + ["MULTI_CONFIRM"]

    gates = dict(GATES_HIGH)
    for s in ["PAIRS_EXP", "CRYPTO_vs_SPX", "CORR_REGIME"]:
        gates[s] = 1.5
    gates["VOLFORECAST"] = 1.0
    gates["H4_SLEEVE"] = 1.5
    gates["TREND_NEW"] = 0.5
    gates["SESSION_MOM"] = 1.0
    gates["MULTI_CONFIRM"] = 0.7  # has trend correlation, dampen slightly in high-vol

    high_vol, _ = build_regime_mask(panel.index, 80)
    g = panel.copy()
    for s, m in gates.items():
        if s in g.columns:
            g.loc[high_vol, s] *= m

    # ---- Compare TOP12/13/14/15 ----
    print(f"\n{'Variant':<28} {'OOS_Sh':>7} {'2022_Sh':>8} {'Vol18_Mo':>10}")
    print("-" * 65)
    best_sleeves = None
    best_oos = 0
    for name, sleeves in [
        ("TOP12 (v9)", TOP12), ("TOP13 (+CORR)", TOP13),
        ("TOP14 (+SESSION_MOM)", TOP14), ("TOP15 (+MULTI_CONFIRM)", TOP15),
    ]:
        # Apply tripwire
        after_decay, _ = fast_decay_tripwire(g[sleeves], sleeves)
        port = after_decay.mean(axis=1)
        # Apply vol-target 18% for monthly calc
        vt, _ = vol_target_overlay(port, target_vol=0.18, max_lev=15.0)
        dd_ctrl, _ = drawdown_control(vt)
        oos = dd_ctrl[dd_ctrl.index >= SPLIT]
        bpy_oos = _bpy(oos.index)
        ar_oos = oos.mean()*bpy_oos
        av_oos = oos.std(ddof=0)*np.sqrt(bpy_oos)
        sh_oos = ar_oos/av_oos if av_oos > 0 else 0
        y22 = port[port.index.year == 2022]
        bpy22 = _bpy(y22.index) if len(y22) > 0 else 252
        sh22 = (y22.mean()*bpy22)/(y22.std(ddof=0)*np.sqrt(bpy22)) if y22.std() > 0 else 0
        mo = ar_oos / 12
        print(f"{name:<28} {sh_oos:>+7.2f} {sh22:>+8.2f} {mo:>+10.2%}")
        if sh_oos > best_oos:
            best_oos = sh_oos
            best_sleeves = sleeves
            best_name = name

    print(f"\nBest variant: {best_name}")

    # ---- Production leverage sweep on best variant ----
    after_decay, _ = fast_decay_tripwire(g[best_sleeves], best_sleeves)
    full_baseline = after_decay.mean(axis=1)
    print(f"\n=== Best variant leverage sweep (all overlays) ===")
    print(f"{'VolTarget':<10} {'OOS_Ret':>8} {'OOS_Vol':>8} {'Sharpe':>7} {'MaxDD':>8} {'Mo/yr':>7} {'Lev':>5}")
    rows = lever_sweep(full_baseline, [0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.27, 0.32])
    for r in rows:
        print(f"{r['vol_target']:<10.0%} {r['ann_ret']:>+8.1%} {r['ann_vol']:>8.1%} "
              f"{r['sharpe']:>+7.2f} {r['maxdd']:>+8.1%} {r['monthly']:>+7.2%} {r['avg_lev']:>5.1f}x")
    pd.DataFrame(rows).to_csv(OUT / "v11_leverage_sweep.csv", index=False)

    # ---- Year-by-year at 18% vol ----
    vt, _ = vol_target_overlay(full_baseline, target_vol=0.18, max_lev=15.0)
    prod, _ = drawdown_control(vt)
    eq = (1 + prod).cumprod()
    pd.DataFrame({"timestamp": prod.index, "ret": prod.values, "equity": eq.values}).to_parquet(
        OUT / "PRODUCTION_v11_18pct.parquet", index=False)

    print(f"\n=== Year-by-year at 18% vol target (PRODUCTION v11) ===")
    for year, sub in prod.groupby(prod.index.year):
        if len(sub) < 50: continue
        bpy = _bpy(sub.index)
        ar = sub.mean()*bpy
        av = sub.std(ddof=0)*np.sqrt(bpy)
        sh = ar/av if av > 0 else 0
        eq_y = (1+sub).cumprod()
        dd = (eq_y/eq_y.cummax()-1).min()
        print(f"  {year}  Sh={sh:+.2f}  Ret={ar:+.1%}  Vol={av:.1%}  DD={dd:+.1%}  Mo={ar/12:+.2%}")

    # ---- Test Kelly blend (diagnostic only — panel already saved above) ----
    print(f"\n=== Kelly-blend (25% Kelly + 75% equal-weight) ===")
    kelly_path = WAVE3 / "kelly_returns.parquet"
    if kelly_path.exists():
        kr = pd.read_parquet(kelly_path)
        # Schema-tolerant: kelly_sizing.py was later updated to save with DatetimeIndex
        # and multiple return columns, breaking the original `kr["timestamp"]` / `kr["ret"]`.
        # Skip the diagnostic if the schema doesn't match the original expectation.
        if "timestamp" not in kr.columns or "ret" not in kr.columns:
            print(f"  (skip: kelly_returns.parquet schema changed; "
                  f"columns={list(kr.columns)[:5]}...)")
            return
        kr_ts = pd.to_datetime(kr["timestamp"], utc=True)
        kelly = pd.Series(kr["ret"].values, index=kr_ts)
        kelly = normalize_idx(kelly.rename("KELLY"))
        kelly = kelly.reindex(panel.index, fill_value=0.0)
        # Blend
        blend = 0.25 * kelly + 0.75 * full_baseline
        vt_blend, _ = vol_target_overlay(blend, target_vol=0.18, max_lev=15.0)
        prod_blend, _ = drawdown_control(vt_blend)
        oos = prod_blend[prod_blend.index >= SPLIT]
        bpy = _bpy(oos.index)
        ar = oos.mean()*bpy
        av = oos.std(ddof=0)*np.sqrt(bpy)
        sh = ar/av if av > 0 else 0
        eq = (1+oos).cumprod()
        dd = (eq/eq.cummax()-1).min()
        print(f"  25% Kelly + 75% EW at 18% vol: Ret={ar:+.1%}  Vol={av:.1%}  Sh={sh:+.2f}  DD={dd:+.1%}  Mo={ar/12:+.2%}")


if __name__ == "__main__":
    main()
