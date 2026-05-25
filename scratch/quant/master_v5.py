"""Master v5 — adds PAIRS_v2 and ROTATION sleeves to the v3 panel,
then applies the v4 production overlays (regime gate + tripwire + vol-target + DD control).

New sleeves (from this iteration):
  * PAIRS_v2: cointegration-filtered (ADF gate). 2 survivors (NAS-US30, EUR-GBP). OOS Sharpe +0.34, 2022 +0.92.
  * ROTATION: top-2 basket rotation with risk-off override. OOS Sharpe +0.48, +5.9% alpha vs eq-wt.

Dropped after testing:
  * Tail sleeve (vol-of-vol short SPX): cost-benefit negative; DEFEND already handles 2022.
  * Volume signals: only S1_SOLUSDT survived; sleeve too narrow.
  * Crypto hour-of-week: data-mined, OOS -4.55 to -0.83.
  * Crypto day-of-month: a priori hypothesis, IS +2.13 → OOS -1.32. Calendar effects in
    crypto flipped post-2024 — same regime-shift signature as Monday-23 indices.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from alphabeta import get_candles

# Re-use the v4 overlay helpers
sys.path.insert(0, str(ROOT / "scratch" / "quant"))
from master_v4 import (
    fast_decay_tripwire, vol_target_overlay, drawdown_control,
    build_regime_mask, apply_regime_gate, stats, _bpy, GATES_HIGH,
)

OUT = Path(__file__).resolve().parent
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
    ts = pd.to_datetime(df["timestamp"], utc=True)
    return normalize_idx(pd.Series(df["ret"].values, index=ts).rename(name))


def rescale_is(s, target):
    is_part = s[s.index < SPLIT]
    av = float(is_part.std(ddof=0)) * np.sqrt(365.25)
    return s * (target / av) if av > 1e-9 else s * 0


def main():
    # Load existing panel
    panel = pd.read_parquet(OUT / "all_sleeve_returns_v3.parquet")
    panel.index = pd.to_datetime(panel.index, utc=True)

    # ---- Add new sleeves ----
    new_streams = {}
    for name, path in [
        ("PAIRS_v2", OUT / "pairs_v2_returns.parquet"),
        ("ROTATION", OUT / "rotation_returns.parquet"),
    ]:
        if path.exists():
            s = load_returns(path, name)
            s = rescale_is(s, TARGET_VOL)
            new_streams[name] = s
            sh_is = s[s.index < SPLIT]
            sh_oos = s[s.index >= SPLIT]
            print(f"Added {name:<10}: bars={len(s)}, IS_Sh={sh_is.mean()*365/(sh_is.std()*np.sqrt(365)):+.2f}, "
                  f"OOS_Sh={sh_oos.mean()*365/(sh_oos.std()*np.sqrt(365)):+.2f}")

    panel = panel.join(pd.DataFrame(new_streams), how="outer").fillna(0.0)
    panel.to_parquet(OUT / "all_sleeve_returns_v5.parquet")

    # ---- Define expanded TOP10 (TOP8 + 2 new) ----
    TOP10 = ["RISKPAR", "TSMOM", "EVE_XAU", "D1REV_UK", "XSMOM",
             "D1REV_NAS", "WED_BTC", "DEFEND",
             "PAIRS_v2", "ROTATION"]
    # Extend regime gates for new sleeves: pairs are mean-rev-ish, rotation is beta-ish
    gates = dict(GATES_HIGH)
    gates["PAIRS_v2"] = 1.5  # amplify like other mean-rev sleeves
    gates["ROTATION"] = 0.5  # halve like beta sleeves

    # Regime
    high_vol, cutoff = build_regime_mask(panel.index, 80)
    g = panel.copy()
    for sleeve, mult in gates.items():
        if sleeve in g.columns:
            g.loc[high_vol, sleeve] *= mult

    # ---- Baseline portfolios ----
    portfolios = {
        "TOP8 (v3)":          panel[["RISKPAR","TSMOM","EVE_XAU","D1REV_UK","XSMOM","D1REV_NAS","WED_BTC","DEFEND"]].mean(axis=1),
        "TOP8 + regime":      g[["RISKPAR","TSMOM","EVE_XAU","D1REV_UK","XSMOM","D1REV_NAS","WED_BTC","DEFEND"]].mean(axis=1),
        "TOP10 (v5)":         panel[TOP10].mean(axis=1),
        "TOP10 + regime":     g[TOP10].mean(axis=1),
    }

    # Add tripwire + vol target + DD control variants on the headline TOP10 + regime
    after_decay, _ = fast_decay_tripwire(g, TOP10)
    tripwired = after_decay[TOP10].mean(axis=1)
    portfolios["TOP10 + regime + tripwire"] = tripwired
    vt, lev = vol_target_overlay(tripwired, target_vol=0.10)
    portfolios["+ vol-target 10%"] = vt
    dd_ctrl, _ = drawdown_control(vt)
    portfolios["+ DD control (FINAL)"] = dd_ctrl

    # ---- Stats ----
    rows = []
    print(f"\n{'Variant':<32} {'FULL':>6} {'IS':>6} {'OOS':>6} {'2022':>7} {'OOS_vol':>8} {'OOS_DD':>7}")
    print("-" * 75)
    for name, r in portfolios.items():
        s = stats(name, r)
        y22 = r[r.index.year == 2022]
        bpy = _bpy(y22.index) if len(y22) > 0 else 252
        sh22 = (y22.mean() * bpy) / (y22.std(ddof=0) * np.sqrt(bpy)) if y22.std() > 0 else 0
        rows.append({**s, "2022_sharpe": sh22})
        print(f"{name:<32} {s.get('FULL_sharpe',0):>+6.2f} {s.get('IS_sharpe',0):>+6.2f} "
              f"{s.get('OOS_sharpe',0):>+6.2f} {sh22:>+7.2f} {s.get('OOS_vol',0):>8.1%} {s.get('OOS_dd',0):>+7.1%}")
    pd.DataFrame(rows).to_csv(OUT / "master_v5_variants.csv", index=False)

    # Year-by-year
    yr = {}
    for name, r in portfolios.items():
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
    yr_df.to_csv(OUT / "master_v5_yearly.csv")

    # Save FINAL (vol-targeted + dd-controlled + tripwired)
    final = dd_ctrl
    eq = (1 + final).cumprod()
    pd.DataFrame({"timestamp": final.index, "ret": final.values, "equity": eq.values}).to_parquet(
        OUT / "PRODUCTION_v5.parquet", index=False)
    print(f"\nProduction v5 (TOP10 + regime + tripwire + vol-target + DD) saved.")
    print(f"  Total return: {(eq.iloc[-1]-1)*100:+.1f}%")
    print(f"  OOS slice:    {(1+final[final.index>=SPLIT]).cumprod().iloc[-1]*100-100:+.1f}%")


if __name__ == "__main__":
    main()
