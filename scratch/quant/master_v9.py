"""Master v9 — integrate wave3 wins.

Wave3 additions:
  + VOLFORECAST  (OOS +1.25, 13/13 symbols positive — EWMA vol-target)
  + H4_SLEEVE    (2022 Sharpe +1.91, uncorrelated to D1 — pure diversifier)
  + TREND_NEW    (OOS +1.08, beats existing TSMOM — REPLACE TSMOM)
  + CRYPTO_DOM   (OOS +0.51, BTC dominance regime)
  + CRYPTO_vs_SPX (2022 +0.75, market-neutral)

Wave3 rejections:
  - ML_META (indistinguishable from equal-weight)
  - SESSION_REVERSION (sessions are MOMENTUM events, not reversion!)
  - REL_STRENGTH (worse than rejected ROTATION; same idea as XSMOM)
  - 4 of 6 synthetic spreads (only crypto vs spx survived)

Production: 9 sleeves with overlays. Test multiple vol-target scenarios.
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
    # H4 sleeve has timestamp as index + multiple columns; use H4_SLEEVE column
    if "timestamp" not in df.columns and "H4_SLEEVE" in df.columns:
        s = pd.Series(df["H4_SLEEVE"].values, index=pd.to_datetime(df.index, utc=True))
        return normalize_idx(s.rename(name))
    # crypto_alpha has 2 sleeve columns — combine equal-weight
    if "ret" not in df.columns and "dom_BTCUSDT" in df.columns:
        combined = (df["dom_BTCUSDT"] + df["dom_ETHUSDT"]) / 2
        ts = pd.to_datetime(df["timestamp"], utc=True)
        return normalize_idx(pd.Series(combined.values, index=ts).rename(name))
    # Standard schema
    ts = pd.to_datetime(df["timestamp"], utc=True)
    return normalize_idx(pd.Series(df["ret"].values, index=ts).rename(name))


def rescale_is(s, target):
    is_part = s[s.index < SPLIT]
    av = float(is_part.std(ddof=0)) * np.sqrt(365.25)
    return s * (target / av) if av > 1e-9 else s * 0


def main():
    # Load v5 panel (12 sleeves) + PAIRS_EXP
    panel = pd.read_parquet(OUT / "all_sleeve_returns_v3.parquet")
    panel.index = pd.to_datetime(panel.index, utc=True)
    pe = load_returns(OUT / "pairs_expanded_returns.parquet", "PAIRS_EXP")
    pe = rescale_is(pe, TARGET_VOL)
    panel["PAIRS_EXP"] = pe.reindex(panel.index, fill_value=0.0)

    # Add wave3 sleeves
    new_sleeves = {
        "VOLFORECAST":     WAVE3 / "volforecast_returns.parquet",
        "H4_SLEEVE":       WAVE3 / "h4_returns.parquet",
        "TREND_NEW":       WAVE3 / "trend_returns.parquet",
        "CRYPTO_DOM":      WAVE3 / "crypto_alpha_returns.parquet",
        "CRYPTO_vs_SPX":   WAVE3 / "synthetic_returns.parquet",
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
        is_sh = is_part.mean() * 365 / (is_part.std(ddof=0) * np.sqrt(365))
        oos_sh = oos_part.mean() * 365 / (oos_part.std(ddof=0) * np.sqrt(365))
        print(f"Added {name:<14}: IS_Sh={is_sh:+.2f}  OOS_Sh={oos_sh:+.2f}")

    print(f"\nFinal panel: {len(panel.columns)} sleeves, {len(panel)} bars")
    panel.to_parquet(OUT / "all_sleeve_returns_v9.parquet")

    # Define TOP12 — drop TSMOM in favor of TREND_NEW (replaces); add VOLFORECAST, H4_SLEEVE, CRYPTO_DOM, CRYPTO_vs_SPX
    # Note: TSMOM kept also since they have ρ=0.68 but each may have unique periods of strength
    TOP12 = ["RISKPAR", "TREND_NEW", "EVE_XAU", "D1REV_UK", "XSMOM",
             "D1REV_NAS", "WED_BTC", "DEFEND", "PAIRS_EXP",
             "VOLFORECAST", "H4_SLEEVE", "CRYPTO_vs_SPX"]
    # Replace TSMOM with TREND_NEW; add 3 wave3 wins; drop CRYPTO_DOM (correlated with WED_BTC + lower Sharpe + bad 2022)

    # Regime gates
    gates = dict(GATES_HIGH)
    gates["PAIRS_EXP"] = 1.5
    gates["VOLFORECAST"] = 1.0  # already vol-adaptive internally
    gates["H4_SLEEVE"] = 1.5  # 2022 hedge
    gates["TREND_NEW"] = 0.5  # trend
    gates["CRYPTO_vs_SPX"] = 1.5  # market-neutral

    high_vol, cutoff = build_regime_mask(panel.index, 80)
    g = panel.copy()
    for sleeve, mult in gates.items():
        if sleeve in g.columns:
            g.loc[high_vol, sleeve] *= mult

    baseline = g[TOP12].mean(axis=1)

    # Per-sleeve breakdown
    print(f"\n{'Sleeve':<14} {'IS':>6} {'OOS':>7} {'2022':>7} {'OOS_DD':>7}")
    for c in TOP12:
        sl = panel[c]
        s = stats(c, sl)
        y22 = sl[sl.index.year == 2022]
        bpy = _bpy(y22.index) if len(y22) > 0 else 252
        sh22 = (y22.mean()*bpy)/(y22.std(ddof=0)*np.sqrt(bpy)) if y22.std() > 0 else 0
        print(f"{c:<14} {s.get('IS_sharpe',0):>+6.2f} {s.get('OOS_sharpe',0):>+7.2f} "
              f"{sh22:>+7.2f} {s.get('OOS_dd',0):>+7.1%}")

    # Apply overlays
    after_decay, decay_log = fast_decay_tripwire(g, TOP12)
    tripwired = after_decay[TOP12].mean(axis=1)
    decay_log.to_csv(OUT / "v9_decay_log.csv", index=False)

    # Test multiple vol targets
    print(f"\n=== Vol-target scenarios ===")
    print(f"{'Target':<8} {'AnnRet':>8} {'AnnVol':>8} {'Sharpe':>7} {'MaxDD':>8} {'Mo/yr':>7} {'Lev':>5}")
    print("-" * 60)
    results = {}
    for tv in [0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.27, 0.32]:
        vt, lev = vol_target_overlay(tripwired, target_vol=tv, max_lev=15.0)
        dd_ctrl, _ = drawdown_control(vt)
        results[tv] = dd_ctrl
        bpy = _bpy(dd_ctrl.index)
        ar = dd_ctrl.mean() * bpy
        av = dd_ctrl.std(ddof=0) * np.sqrt(bpy)
        sh = ar/av if av > 0 else 0
        eq = (1 + dd_ctrl).cumprod()
        dd = (eq / eq.cummax() - 1).min()
        monthly_ret = ar / 12
        avg_lev = lev.mean()
        print(f"{tv:<8.0%} {ar:>+8.1%} {av:>8.1%} {sh:>+7.2f} {dd:>+8.1%} {monthly_ret:>+7.2%} {avg_lev:>5.1f}x")
        # OOS specific
        oos = dd_ctrl[dd_ctrl.index >= SPLIT]
        bpy_oos = _bpy(oos.index)
        ar_oos = oos.mean() * bpy_oos
        av_oos = oos.std(ddof=0) * np.sqrt(bpy_oos)
        sh_oos = ar_oos/av_oos if av_oos > 0 else 0
        eq_oos = (1 + oos).cumprod()
        dd_oos = (eq_oos / eq_oos.cummax() - 1).min()
        print(f"  ↳ OOS:  {ar_oos:>+7.1%} {av_oos:>7.1%} {sh_oos:>+6.2f} {dd_oos:>+7.1%} {ar_oos/12:>+6.2%}/mo")

    # Save 10% vol target (sane production)
    prod = results[0.10]
    eq = (1 + prod).cumprod()
    pd.DataFrame({"timestamp": prod.index, "ret": prod.values, "equity": eq.values}).to_parquet(
        OUT / "PRODUCTION_v9.parquet", index=False)
    # Also save the 18% target ("aggressive growth")
    aggr = results[0.18]
    eq_a = (1 + aggr).cumprod()
    pd.DataFrame({"timestamp": aggr.index, "ret": aggr.values, "equity": eq_a.values}).to_parquet(
        OUT / "PRODUCTION_v9_AGGRESSIVE.parquet", index=False)

    # Year-by-year for 10% vol target
    print(f"\n=== Year-by-year (10% vol target production) ===")
    for year, sub in prod.groupby(prod.index.year):
        if len(sub) < 50: continue
        bpy = _bpy(sub.index)
        ar = sub.mean() * bpy
        av = sub.std(ddof=0) * np.sqrt(bpy)
        sh = ar/av if av > 0 else 0
        eq_y = (1 + sub).cumprod()
        dd = (eq_y / eq_y.cummax() - 1).min()
        print(f"  {year}  Sh={sh:+.2f}  Ret={ar:+.1%}  DD={dd:+.1%}  Mo={ar/12:+.2%}")


if __name__ == "__main__":
    main()
