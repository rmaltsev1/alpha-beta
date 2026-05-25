"""Master v16 — push returns higher.

Adds on top of v15:
  + WKND_FUND_SPIKE (crypto weekend funding proxy, OOS +0.90, 2022 +1.60)
  + Aggressive leverage variants V4 (floor+ceiling) and V5 (SPX regime ramp)

V4 baseline: floor 25% vol target, halve when 5d realized portfolio vol > 30%.
V5 aggressive: SPX-vol-regime-conditional 12-30% target.

Both target +8-12%/month at MaxDD ~10%.
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
from master_v12 import time_of_month_vol_target

from alphabeta import get_candles

OUT = Path(__file__).resolve().parent
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


def rescale_is(s, target):
    is_part = s[s.index < SPLIT]
    av = float(is_part.std(ddof=0)) * np.sqrt(365.25)
    return s * (target / av) if av > 1e-9 else s * 0


def floor_ceiling_lever(returns: pd.Series, base_target: float = 0.25,
                        cap_vol: float = 0.30, max_lev: float = 15.0) -> pd.Series:
    """V4 aggressive: target 25% vol normally; when 5d realized port vol > 30%, halve gross."""
    rolling_vol_60 = returns.rolling(60).std(ddof=0).shift(1) * np.sqrt(365.25)
    rolling_vol_5 = returns.rolling(5).std(ddof=0).shift(1) * np.sqrt(365.25)
    lev = (base_target / rolling_vol_60).clip(upper=max_lev).fillna(1.0)
    # Cut leverage in half when short-term vol exceeds cap
    cut = rolling_vol_5 > cap_vol
    lev = lev * np.where(cut, 0.5, 1.0)
    return returns * lev


def spx_regime_ramp_lever(returns: pd.Series, base_target: float = 0.22,
                          max_lev: float = 15.0) -> pd.Series:
    """V5 SPX-regime ramp: 30% target when SPX vol low, 12% when high."""
    spx = get_candles("SPX500_USD", "D1").copy()
    spx["ret"] = np.log(spx["close"] / spx["close"].shift(1))
    spx["rv30"] = spx["ret"].rolling(30).std()
    is_part = spx.loc[spx["timestamp"] < SPLIT, "rv30"].dropna()
    p25, p75 = float(is_part.quantile(0.25)), float(is_part.quantile(0.75))
    spx["timestamp"] = pd.to_datetime(spx["timestamp"], utc=True)
    aligned = pd.merge_asof(
        pd.DataFrame({"timestamp": returns.index}).sort_values("timestamp"),
        spx[["timestamp", "rv30"]].sort_values("timestamp"),
        on="timestamp", direction="backward",
    )
    rv = aligned["rv30"].ffill().values
    # Ramp target between 12% (high vol) and 30% (low vol)
    target_series = np.where(rv < p25, 0.30,
                    np.where(rv > p75, 0.12, base_target))
    target_series = pd.Series(target_series, index=returns.index)
    rolling_vol = returns.rolling(60).std(ddof=0).shift(1) * np.sqrt(365.25)
    lev = (target_series / rolling_vol).clip(upper=max_lev).fillna(1.0)
    return returns * lev


def main():
    panel = pd.read_parquet(OUT / "all_sleeve_returns_v15.parquet")
    panel.index = pd.to_datetime(panel.index, utc=True)

    # Add WKND_FUND_SPIKE
    fund_path = WAVE6 / "funding_proxy_returns.parquet"
    if fund_path.exists():
        fund_df = pd.read_parquet(fund_path)
        if "WKND_FUND_SPIKE" in fund_df.columns:
            ts = pd.to_datetime(fund_df["timestamp"], utc=True)
            s = pd.Series(fund_df["WKND_FUND_SPIKE"].values, index=ts)
            s = normalize_idx(s.rename("WKND_FUND"))
            s = rescale_is(s, TARGET_VOL)
            panel["WKND_FUND"] = s.reindex(panel.index, fill_value=0.0)
            oos = panel["WKND_FUND"][panel.index >= SPLIT]
            oos_sh = oos.mean()*365/(oos.std(ddof=0)*np.sqrt(365)) if oos.std() > 0 else 0
            print(f"Added WKND_FUND: OOS_Sh={oos_sh:+.2f}")

    panel.to_parquet(OUT / "all_sleeve_returns_v16.parquet")

    # TOP24 = v15 TOP23 + WKND_FUND
    TOP24 = ["RISKPAR","TREND_NEW","EVE_XAU","D1REV_UK","XSMOM","D1REV_NAS","WED_BTC",
             "DEFEND","PAIRS_EXP","VOLFORECAST","H4_SLEEVE","CRYPTO_vs_SPX",
             "CORR_REGIME","SESSION_MOM","W1_STRATS","EVENT_VOLSPIKE",
             "STATARB_XS","MICROSTR_D1","VOL_BREAKOUT","TERM_SPREADS","EURGBP_MR",
             "MULTIDAY","HMM_BULL_TSMOM","WKND_FUND"]

    gates = dict(GATES_HIGH)
    for s in ["PAIRS_EXP","CRYPTO_vs_SPX","CORR_REGIME","STATARB_XS","MICROSTR_D1",
              "EURGBP_MR","TERM_SPREADS","EVENT_VOLSPIKE","MULTIDAY"]:
        gates[s] = 1.5
    for s in ["VOLFORECAST","W1_STRATS","SESSION_MOM"]:
        gates[s] = 1.0
    gates["H4_SLEEVE"] = 1.5
    gates["TREND_NEW"] = 0.5
    gates["HMM_BULL_TSMOM"] = 0.7
    gates["VOL_BREAKOUT"] = 1.2
    gates["WKND_FUND"] = 1.2

    high_vol, _ = build_regime_mask(panel.index, 80)
    g = panel.copy()
    for s, m in gates.items():
        if s in g.columns:
            g.loc[high_vol, s] *= m

    after_decay, _ = fast_decay_tripwire(g[TOP24], TOP24)
    base = after_decay.mean(axis=1)

    # Three production variants:
    #   v15-style: time-of-month 18% target + DD control (conservative)
    #   v16-V4: 25% floor + 30% ceiling (production aggressive)
    #   v16-V5: SPX-regime ramp 12-30% (max aggressive)

    print(f"\n{'Variant':<28} {'OOS_Ret':>8} {'OOS_Vol':>8} {'Sharpe':>7} {'MaxDD':>8} {'Mo':>7} {'Mo_p5':>8}")
    print("-" * 80)

    # v15 baseline (18% target)
    vt, _ = time_of_month_vol_target(base, base_target=0.18)
    v15_prod, _ = drawdown_control(vt)

    # v16-V4: floor + ceiling
    v16_v4 = floor_ceiling_lever(base, base_target=0.25, cap_vol=0.30)

    # v16-V5: SPX regime ramp
    v16_v5 = spx_regime_ramp_lever(base, base_target=0.22)

    for name, r in [("v15 baseline (18%)", v15_prod),
                     ("v16-V4 (floor+ceiling 25%)", v16_v4),
                     ("v16-V5 (SPX regime 12-30%)", v16_v5)]:
        oos = r[r.index >= SPLIT]
        if len(oos) < 30: continue
        bpy = _bpy(oos.index)
        ar = oos.mean()*bpy
        av = oos.std(ddof=0)*np.sqrt(bpy)
        sh = ar/av if av > 0 else 0
        eq = (1+oos).cumprod()
        ddv = (eq/eq.cummax()-1).min()
        # Monthly distribution
        monthly = oos.groupby(oos.index.to_period('M')).sum()
        mo_mean = monthly.mean()
        mo_p5 = monthly.quantile(0.05)
        print(f"{name:<28} {ar:>+8.1%} {av:>8.1%} {sh:>+7.2f} {ddv:>+8.1%} "
              f"{mo_mean:>+7.2%} {mo_p5:>+8.2%}")

    # Year-by-year for V4 (the recommended)
    print(f"\n=== Year-by-year v16-V4 ===")
    for year, sub in v16_v4.groupby(v16_v4.index.year):
        if len(sub) < 50: continue
        bpy = _bpy(sub.index)
        ar = sub.mean()*bpy
        av = sub.std(ddof=0)*np.sqrt(bpy)
        sh = ar/av if av > 0 else 0
        eq_y = (1+sub).cumprod()
        dd = (eq_y/eq_y.cummax()-1).min()
        print(f"  {year}  Sh={sh:+.2f}  Ret={ar:+.1%}  Vol={av:.1%}  DD={dd:+.1%}  Mo={ar/12:+.2%}")

    # Save production V4 and V5
    eq4 = (1 + v16_v4).cumprod()
    pd.DataFrame({"timestamp": v16_v4.index, "ret": v16_v4.values,
                   "equity": eq4.values}).to_parquet(OUT / "PRODUCTION_v16_V4.parquet", index=False)
    eq5 = (1 + v16_v5).cumprod()
    pd.DataFrame({"timestamp": v16_v5.index, "ret": v16_v5.values,
                   "equity": eq5.values}).to_parquet(OUT / "PRODUCTION_v16_V5.parquet", index=False)


if __name__ == "__main__":
    main()
