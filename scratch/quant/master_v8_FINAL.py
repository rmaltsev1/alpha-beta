"""Master v8 — TOP8 + PAIRS_EXPANDED (11 cross-asset pairs) + full overlay stack.

The expanded pair search lifted PAIRS' OOS Sharpe from +0.34 to +0.63 and
its 2022 Sharpe from +0.92 to +2.06 — pairs is now the strongest 2022 sleeve
in the portfolio.

Key new pairs (cross-asset FX↔Index):
  GBP_USD-US30_USD       OOS +1.06
  GBP_USD-SPX500_USD     OOS +0.94
  GBP_USD-DE30_EUR       OOS +0.86
  USD_JPY-US30_USD       OOS +0.73   2022 +0.73
  USD_JPY-UK100_GBP      OOS +0.39   2022 +0.83
  NAS100-US30            OOS +0.31   2022 +1.08
  EUR_USD-DE30_EUR       OOS +0.49   2022 +1.36
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

TOP9 = ["RISKPAR", "TSMOM", "EVE_XAU", "D1REV_UK", "XSMOM",
        "D1REV_NAS", "WED_BTC", "DEFEND", "PAIRS_EXP"]


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
    panel = pd.read_parquet(OUT / "all_sleeve_returns_v3.parquet")
    panel.index = pd.to_datetime(panel.index, utc=True)

    # Swap in PAIRS_EXPANDED for the old PAIRS_v2
    pairs_exp = load_returns(OUT / "pairs_expanded_returns.parquet", "PAIRS_EXP")
    pairs_exp = rescale_is(pairs_exp, 0.05)
    panel = panel.join(pairs_exp.rename("PAIRS_EXP"), how="outer").fillna(0.0)
    panel = panel[TOP9]

    # Regime gate
    high_vol, cutoff = build_regime_mask(panel.index, 80)
    gates = dict(GATES_HIGH)
    gates["PAIRS_EXP"] = 1.5  # amplify in high vol
    g = panel.copy()
    for sleeve, mult in gates.items():
        if sleeve in g.columns:
            g.loc[high_vol, sleeve] *= mult

    baseline = g.mean(axis=1)

    after_decay, decay_log = fast_decay_tripwire(g, TOP9)
    decay_log.to_csv(OUT / "v8_decay_log.csv", index=False)
    tripwired = after_decay.mean(axis=1)
    vt, lev = vol_target_overlay(tripwired, target_vol=0.10)
    final, _ = drawdown_control(vt)

    variants = {
        "TOP9_EXP baseline":         baseline,
        "+ decay tripwire":           tripwired,
        "+ vol-target 10%":           vt,
        "+ DD control (PRODUCTION)":  final,
    }

    print(f"{'Variant':<35} {'FULL':>6} {'IS':>6} {'OOS':>6} {'2022':>7} {'OOS_vol':>8} {'OOS_DD':>7}")
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
    pd.DataFrame(rows).to_csv(OUT / "master_v8_variants.csv", index=False)

    print(f"\nYear-by-year Sharpe:")
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
    print(yr_df.round(2).to_string())
    yr_df.to_csv(OUT / "master_v8_yearly.csv")

    eq = (1 + final).cumprod()
    pd.DataFrame({"timestamp": final.index, "ret": final.values, "equity": eq.values}).to_parquet(
        OUT / "PRODUCTION_v8_FINAL.parquet", index=False)
    print(f"\nPRODUCTION v8 saved.")
    print(f"  Total return:   {(eq.iloc[-1]-1)*100:+.1f}%")
    days = (eq.index[-1] - eq.index[0]).days
    print(f"  Annualized:     {(eq.iloc[-1]**(365.25/days)-1)*100:+.1f}%")
    print(f"  OOS slice:      {(1+final[final.index>=SPLIT]).cumprod().iloc[-1]*100-100:+.1f}%")
    print(f"  Avg leverage:   {lev.mean():.2f}x")


if __name__ == "__main__":
    main()
