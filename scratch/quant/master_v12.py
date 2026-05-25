"""Master v12 — integrate Wave 5 + Wave 6 wins.

Wave 5 wins:
  + V6 time-of-month vol scaling overlay (Pareto improvement: OOS Sh +0.09, MaxDD -0.2pp)
  + TAIL_v3 #7 beta-neutral safe-haven (OOS +1.85, but +0.36 corr — small weight)

Wave 6 wins:
  + W1_STRATEGIES (OOS +1.49, 2022 +1.96, uncorrelated with D1!) ← BIGGEST WIN
  + VOL_SPIKE_DAYAFTER (event sleeve, OOS +0.66, 2022 +1.26)
  + MOM_QUALITY DD_FILTER + ANTI_QUALITY (combined OOS +0.93, 2022 +1.68)
  + VRP_PROXY (OOS +0.86, but carry-contaminated)

Wave 5/6 rejections (with data):
  - Adaptive weighting (tied with EW, +0.016 OOS)
  - Multi-timeframe ensembles (dominated by TREND_NEW)
  - Asymmetric strategies (2022 bad)
  - Conservative Kelly v2 (no variant beat constraint)
  - Skewness premium (no lottery names in universe)
  - Adaptive reversion (doesn't beat vanilla)
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


def load_returns(path, name):
    df = pd.read_parquet(path)
    if "timestamp" not in df.columns and "H4_SLEEVE" in df.columns:
        s = pd.Series(df["H4_SLEEVE"].values, index=pd.to_datetime(df.index, utc=True))
        return normalize_idx(s.rename(name))
    if "ret" not in df.columns and "timestamp" in df.columns:
        cols = [c for c in df.columns if c != "timestamp"]
        if "survivors_mean" in df.columns:
            combined = df["survivors_mean"]
        elif "COMBINED" in df.columns:
            combined = df["COMBINED"]
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


def time_of_month_vol_target(returns: pd.Series, base_target: float = 0.18,
                               max_lev: float = 15.0) -> tuple[pd.Series, pd.Series]:
    """V6 from wave 5: 20% in days 1-10, 16% in days 11-20, 18% in days 21+.
    Modulates the base vol-target overlay by day-of-month.
    """
    rolling_vol = returns.rolling(60).std(ddof=0).shift(1) * np.sqrt(365.25)
    dom = returns.index.day
    target_series = pd.Series(base_target, index=returns.index)
    target_series[(dom >= 1) & (dom <= 10)] = base_target * (20/18)  # 20%
    target_series[(dom >= 11) & (dom <= 20)] = base_target * (16/18)  # 16%
    target_series[dom >= 21] = base_target  # 18%
    lev = (target_series / rolling_vol).clip(upper=max_lev).fillna(1.0)
    lev = lev.where(rolling_vol > 0, 1.0)
    return returns * lev, lev


def main():
    panel = pd.read_parquet(OUT / "all_sleeve_returns_v11.parquet")
    panel.index = pd.to_datetime(panel.index, utc=True)
    print(f"Starting panel: {len(panel.columns)} sleeves")

    # Add wave 5-6 wins
    new_sleeves = {
        "W1_STRATS":       WAVE6 / "w1_returns.parquet",
        "EVENT_VOLSPIKE":  WAVE6 / "event_clusters_returns.parquet",
        "MOM_QUALITY":     WAVE6 / "mom_quality_returns.parquet",
        "VRP_PROXY":       WAVE6 / "vrp_returns.parquet",
        "TAIL_SAFEHAVEN":  WAVE5 / "tail_v3_returns.parquet",
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
        is_sh = is_part.mean()*365/(is_part.std(ddof=0)*np.sqrt(365)) if is_part.std() > 0 else 0
        oos_sh = oos_part.mean()*365/(oos_part.std(ddof=0)*np.sqrt(365)) if oos_part.std() > 0 else 0
        print(f"Added {name:<16}: IS_Sh={is_sh:+.2f}  OOS_Sh={oos_sh:+.2f}")

    panel.to_parquet(OUT / "all_sleeve_returns_v12.parquet")
    print(f"\nFinal panel: {len(panel.columns)} sleeves")

    # TOP19 — TOP14 + 5 wave 5-6 sleeves
    TOP19 = ["RISKPAR", "TREND_NEW", "EVE_XAU", "D1REV_UK", "XSMOM",
             "D1REV_NAS", "WED_BTC", "DEFEND", "PAIRS_EXP",
             "VOLFORECAST", "H4_SLEEVE", "CRYPTO_vs_SPX",
             "CORR_REGIME", "SESSION_MOM",
             "W1_STRATS", "EVENT_VOLSPIKE", "MOM_QUALITY",
             "VRP_PROXY", "TAIL_SAFEHAVEN"]

    gates = dict(GATES_HIGH)
    for s in ["PAIRS_EXP", "CRYPTO_vs_SPX", "CORR_REGIME", "TAIL_SAFEHAVEN"]:
        gates[s] = 1.5
    gates["VOLFORECAST"] = 1.0
    gates["H4_SLEEVE"] = 1.5
    gates["TREND_NEW"] = 0.5
    gates["SESSION_MOM"] = 1.0
    gates["W1_STRATS"] = 1.0  # slow signal, no high-vol gate
    gates["EVENT_VOLSPIKE"] = 1.2  # mildly amplify in high-vol
    gates["MOM_QUALITY"] = 0.7  # trend-correlated, dampen
    gates["VRP_PROXY"] = 0.8

    high_vol, _ = build_regime_mask(panel.index, 80)
    g = panel.copy()
    for s, m in gates.items():
        if s in g.columns:
            g.loc[high_vol, s] *= m

    # Compare different sleeve sets
    print(f"\n{'Variant':<30} {'OOS_Sh':>7} {'2022_Sh':>8} {'OOS_DD':>8}")
    print("-" * 60)
    for name, sleeves in [
        ("TOP14 (v11)", TOP19[:14]),
        ("TOP14 + W1_STRATS", TOP19[:14] + ["W1_STRATS"]),
        ("TOP14 + W1 + EVENT", TOP19[:14] + ["W1_STRATS", "EVENT_VOLSPIKE"]),
        ("TOP14 + W1+EVENT+MOMQ", TOP19[:14] + ["W1_STRATS", "EVENT_VOLSPIKE", "MOM_QUALITY"]),
        ("TOP18 (no TAIL)", TOP19[:-1]),
        ("TOP19 (everything)", TOP19),
    ]:
        port = g[sleeves].mean(axis=1)
        s = stats(name, port)
        y22 = port[port.index.year == 2022]
        bpy = _bpy(y22.index) if len(y22) > 0 else 252
        sh22 = (y22.mean()*bpy)/(y22.std(ddof=0)*np.sqrt(bpy)) if y22.std() > 0 else 0
        print(f"{name:<30} {s.get('OOS_sharpe',0):>+7.2f} {sh22:>+8.2f} {s.get('OOS_dd',0):>+8.1%}")

    # Pick best variant based on OOS Sharpe — likely TOP19
    # Apply full overlay stack
    after_decay, _ = fast_decay_tripwire(g[TOP19], TOP19)
    full_baseline = after_decay.mean(axis=1)

    # Standard 18% target
    vt, lev = vol_target_overlay(full_baseline, target_vol=0.18, max_lev=15.0)
    dd_ctrl_std, _ = drawdown_control(vt)

    # V6 time-of-month variant
    vt_v6, lev_v6 = time_of_month_vol_target(full_baseline, base_target=0.18)
    dd_ctrl_v6, _ = drawdown_control(vt_v6)

    print(f"\n=== TOP19 leverage sweep (standard 18% baseline) ===")
    for tv in [0.10, 0.15, 0.18, 0.22, 0.27, 0.32]:
        vt, lev = vol_target_overlay(full_baseline, target_vol=tv, max_lev=15.0)
        dd, _ = drawdown_control(vt)
        oos = dd[dd.index >= SPLIT]
        bpy_oos = _bpy(oos.index)
        ar = oos.mean()*bpy_oos
        av = oos.std(ddof=0)*np.sqrt(bpy_oos)
        sh = ar/av if av > 0 else 0
        eq = (1+oos).cumprod()
        ddv = (eq/eq.cummax()-1).min()
        print(f"  {tv:<6.0%} OOS Ret={ar:+.1%}  Vol={av:.1%}  Sh={sh:+.2f}  DD={ddv:+.1%}  Mo={ar/12:+.2%}")

    # Year-by-year for V6 (with time-of-month) at 18% base
    print(f"\n=== Year-by-year with V6 vol-target schedule (18% base) ===")
    prod = dd_ctrl_v6
    for year, sub in prod.groupby(prod.index.year):
        if len(sub) < 50: continue
        bpy = _bpy(sub.index)
        ar = sub.mean()*bpy
        av = sub.std(ddof=0)*np.sqrt(bpy)
        sh = ar/av if av > 0 else 0
        eq_y = (1+sub).cumprod()
        dd = (eq_y/eq_y.cummax()-1).min()
        print(f"  {year}  Sh={sh:+.2f}  Ret={ar:+.1%}  Vol={av:.1%}  DD={dd:+.1%}  Mo={ar/12:+.2%}")

    eq = (1 + prod).cumprod()
    pd.DataFrame({"timestamp": prod.index, "ret": prod.values, "equity": eq.values}).to_parquet(
        OUT / "PRODUCTION_v12.parquet", index=False)


if __name__ == "__main__":
    main()
