"""Cost stress test on the v3 calendar sleeves.

Re-runs the 7 calendar sleeves at 1x / 2x / 3x the assumed per-side cost,
shows how the headline TOP8 portfolio's OOS Sharpe degrades.

For the quant sleeves (TSMOM/XSMOM/VOLMGD/RISKPAR/DEFEND) we keep the 1x
return stream from upstream — re-running them at higher costs would require
re-invoking the per-sleeve scripts, but their turnover is bounded by the
weekly rebalance cycle so the drag is mechanically smaller than the calendar
sleeves' single-bar holds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scratch" / "models"))

from alphabeta import get_candles
from alphabeta.backtest import backtest, cost_for
from alphabeta.symbols import SYMBOL_TYPE
import strategies_v2 as S2

OUT = Path(__file__).resolve().parent
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
TARGET_VOL = 0.05


def _bpy(idx) -> float:
    idx = pd.DatetimeIndex(idx)
    span = (idx[-1] - idx[0]).total_seconds() / 86400
    return len(idx) / span * 365.25 if span > 0 else 252.0


def _scale_is(df_full, build_position, target):
    is_df = df_full[df_full["timestamp"] < SPLIT].reset_index(drop=True)
    is_pos = pd.Series(build_position(is_df).values, index=is_df.index)
    ret = np.log(is_df["close"] / is_df["close"].shift(1)).fillna(0)
    raw = is_pos.values * ret.values
    av = float(np.std(raw, ddof=0) * np.sqrt(_bpy(is_df["timestamp"].values)))
    return target / av if av > 1e-9 else 0


def stats(label, r):
    out = {"label": label}
    for tag, m in [("FULL", pd.Series(True, index=r.index)),
                   ("IS",   r.index < SPLIT),
                   ("OOS",  r.index >= SPLIT)]:
        sub = r[m]
        if len(sub) < 2: continue
        bpy = _bpy(sub.index)
        ar = float(sub.mean()) * bpy
        av = float(sub.std(ddof=0)) * np.sqrt(bpy)
        out[f"{tag}_sharpe"] = ar / av if av > 0 else 0
        eq = (1 + sub).cumprod()
        out[f"{tag}_dd"] = float((eq / eq.cummax() - 1).min())
    return out


def normalize_index(s):
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    else:
        s.index = s.index.tz_convert("UTC")
    s = s.groupby(s.index.floor("D")).sum()
    s.index = pd.to_datetime(s.index, utc=True)
    return s


def calendar_sleeve_with_cost(name, symbol, timeframe, build_position, cost_mult):
    df = get_candles(symbol, timeframe)
    scale = _scale_is(df, build_position, TARGET_VOL)
    full_pos = pd.Series(build_position(df).values, index=df.index) * scale
    boosted = cost_for(symbol) * cost_mult
    res = backtest(df, full_pos, symbol=symbol, timeframe=timeframe,
                   name=name, cost_per_side=boosted)
    idx = pd.to_datetime(df["timestamp"].values, utc=True)
    return normalize_index(pd.Series(res.returns.values, index=idx))


def main():
    sleeves = [
        ("EVE_XAU",   "XAU_USD",    "H1",  S2.evening_long),
        ("WED_BTC",   "BTCUSDT",    "D1",  S2.crypto_wed_long),
        ("WED_ETH",   "ETHUSDT",    "D1",  S2.crypto_wed_long),
        ("WED_SOL",   "SOLUSDT",    "D1",  S2.crypto_wed_long),
        ("D1REV_NAS", "NAS100_USD", "D1",  lambda df: S2.d1_reversion(df, threshold_bps=50)),
        ("D1REV_UK",  "UK100_GBP",  "D1",  lambda df: S2.d1_reversion(df, threshold_bps=50)),
        ("D1REV_SPX", "SPX500_USD", "D1",  lambda df: S2.d1_reversion(df, threshold_bps=50)),
    ]

    # Load quant sleeves (1x cost only — kept as baseline)
    quant_panel = pd.read_parquet(OUT / "all_sleeve_returns_v3.parquet")
    quant_panel.index = pd.to_datetime(quant_panel.index, utc=True)
    quant_cols = ["TSMOM", "XSMOM", "VOLMGD", "RISKPAR", "DEFEND"]
    quants = quant_panel[quant_cols]

    print(f"{'Cost mult':<10} {'sleeve':<14} {'IS_Sh':>6} {'OOS_Sh':>7}")
    print("-" * 50)
    by_mult = {}
    for mult in [1.0, 2.0, 3.0]:
        cal_streams = {}
        for name, sym, tf, fn in sleeves:
            s = calendar_sleeve_with_cost(name, sym, tf, fn, mult)
            cal_streams[name] = s
            stt = stats(name, s)
            print(f"{mult:<10.1f} {name:<14} {stt.get('IS_sharpe', 0):>+6.2f} {stt.get('OOS_sharpe', 0):>+7.2f}")
        # Build panel with calendar sleeves at this cost + quants at 1x
        # Re-scale calendar sleeves to 5% IS vol so the panel is comparable
        cal_df = pd.concat(cal_streams, axis=1, sort=True).fillna(0.0)
        from master_v3 import rescale_to_target_vol  # re-use
        rescaled = pd.concat({n: rescale_to_target_vol(cal_df[n], TARGET_VOL)
                              for n in cal_df.columns}, axis=1)
        panel = rescaled.join(quants, how="outer").fillna(0.0)
        by_mult[mult] = panel

    # Build the production portfolio (TOP8 + regime gate) at each cost level
    from master_v3 import build_regime_mask
    print(f"\n=== Portfolio variants at different calendar-sleeve cost multipliers ===")
    print(f"{'Cost mult':<10} {'IS_Sh':>6} {'OOS_Sh':>7} {'2022_Sh':>8} {'OOS_DD':>7}")
    print("-" * 55)
    rows = []
    for mult, panel in by_mult.items():
        high_vol, _ = build_regime_mask(panel.index, percentile=80)
        top8 = ["RISKPAR", "TSMOM", "EVE_XAU", "D1REV_UK", "XSMOM",
                "D1REV_NAS", "WED_BTC", "DEFEND"]
        gates = {  # high-vol multipliers
            "RISKPAR": 0.5, "TSMOM": 0.5, "XSMOM": 0.5, "EVE_XAU": 0.5,
            "WED_BTC": 0.5,
            "D1REV_NAS": 1.5, "D1REV_UK": 1.5, "DEFEND": 1.5,
        }
        gated = panel.copy()
        for k, g in gates.items():
            gated.loc[high_vol, k] *= g
        port = gated[top8].mean(axis=1)
        s = stats(f"mult={mult}", port)
        # 2022 sub
        y22 = port[port.index.year == 2022]
        bpy = _bpy(y22.index)
        sh22 = (y22.mean() * bpy) / (y22.std(ddof=0) * np.sqrt(bpy)) if y22.std() > 0 else 0
        rows.append({"cost_mult": mult, **s, "2022_sharpe": sh22})
        print(f"{mult:<10.1f} {s.get('IS_sharpe',0):>+6.2f} {s.get('OOS_sharpe',0):>+7.2f} "
              f"{sh22:>+8.2f} {s.get('OOS_dd',0):>+7.1%}")

    pd.DataFrame(rows).to_csv(OUT / "cost_stress.csv", index=False)
    print(f"\nSaved → {OUT / 'cost_stress.csv'}")


if __name__ == "__main__":
    main()
