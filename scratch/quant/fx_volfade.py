"""Test the volume agent's hypothesis: OANDA FX "volume" (tick count) is a
FADE signal, not a follow signal. The agent's quick check showed post-spike
H1 bars revert by ~0.5 bps, implying we should fade tick spikes rather than
follow them.

Strategy: on H1 forex (EUR/GBP/JPY/XAU), when the prior bar's tick count is
in the top 10% of trailing 60-bar window AND the prior bar moved in some
direction, take the OPPOSITE direction for the next 1-3 bars.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from alphabeta import get_candles
from alphabeta.backtest import backtest

OUT = Path(__file__).resolve().parent
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
TARGET_VOL = 0.05


def _bpy(idx):
    idx = pd.DatetimeIndex(idx)
    span = (idx[-1] - idx[0]).total_seconds() / 86400
    return len(idx) / span * 365.25 if span > 0 else 252.0


def stats(label, r):
    out = {"label": label}
    for tag, mask in [("FULL", pd.Series(True, index=r.index)),
                      ("IS",   r.index < SPLIT),
                      ("OOS",  r.index >= SPLIT)]:
        sub = r[mask]
        if len(sub) < 2: continue
        bpy = _bpy(sub.index)
        ar = float(sub.mean()) * bpy
        av = float(sub.std(ddof=0)) * np.sqrt(bpy)
        out[f"{tag}_sharpe"] = ar/av if av > 0 else 0
        out[f"{tag}_ret"] = ar
    return out


def fade_signal(df, hold_bars=1, vol_pct=0.90, lookback=60):
    """Fade FX tick spikes. Walk-forward rolling pct threshold."""
    df = df.copy()
    df["ret"] = np.log(df["close"] / df["close"].shift(1))
    df["vol_pct_thresh"] = df["volume"].rolling(lookback).quantile(vol_pct).shift(1)
    df["high_vol_prev"] = (df["volume"].shift(1) > df["vol_pct_thresh"].shift(1)).fillna(False)
    df["prev_ret_sign"] = np.sign(df["ret"].shift(1))
    pos = pd.Series(0.0, index=df.index)
    # Enter opposite direction for `hold_bars` after high-vol bar
    fired = df["high_vol_prev"] & (df["prev_ret_sign"] != 0)
    fade_dir = -df["prev_ret_sign"]
    pos[fired] = fade_dir[fired]
    # Forward-fill the position for hold_bars
    for h in range(1, hold_bars):
        prev = pos.shift(h).fillna(0)
        # Only fill where we don't have a fresh signal
        pos = pos.where(pos != 0, prev)
    return pos


def _scale_is(df_full, build_position, target):
    is_df = df_full[df_full["timestamp"] < SPLIT].reset_index(drop=True)
    is_pos = pd.Series(build_position(is_df).values, index=is_df.index)
    ret = np.log(is_df["close"] / is_df["close"].shift(1)).fillna(0)
    raw = is_pos.values * ret.values
    av = float(np.std(raw, ddof=0) * np.sqrt(_bpy(is_df["timestamp"].values)))
    return target/av if av > 1e-9 else 0


def main():
    print(f"{'Symbol':<10} {'Hold':>4} {'Pct':>5} {'IS_Sh':>6} {'OOS_Sh':>7} {'scale':>6}")
    print("-" * 50)
    rows = []
    sleeves = {}
    for sym in ["EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD"]:
        df = get_candles(sym, "H1")
        for hold in [1, 2, 3]:
            for pct in [0.90, 0.95]:
                builder = lambda df, h=hold, p=pct: fade_signal(df, hold_bars=h, vol_pct=p)
                scale = _scale_is(df, builder, TARGET_VOL)
                pos = pd.Series(builder(df).values, index=df.index) * scale
                res = backtest(df, pos, symbol=sym, timeframe="H1",
                               name=f"FX_FADE_{sym}_h{hold}_p{int(pct*100)}")
                idx = pd.to_datetime(df["timestamp"].values, utc=True)
                rets = pd.Series(res.returns.values, index=idx)
                # Collapse to D1
                daily = rets.groupby(rets.index.floor("D")).sum()
                daily.index = pd.to_datetime(daily.index, utc=True)
                s = stats(sym, daily)
                rows.append({"symbol": sym, "hold": hold, "pct": pct, "scale": scale, **s})
                key = f"{sym}_h{hold}_p{int(pct*100)}"
                sleeves[key] = daily
                print(f"{sym:<10} {hold:>4} {pct:>5.2f} {s.get('IS_sharpe',0):>+6.2f} "
                      f"{s.get('OOS_sharpe',0):>+7.2f} {scale:>6.2f}")

    pd.DataFrame(rows).to_csv(OUT / "fx_volfade_results.csv", index=False)
    survivors = [r for r in rows if r.get("IS_sharpe", 0) > 0.3 and r.get("OOS_sharpe", 0) > 0]
    print(f"\nSurvivors: {len(survivors)}")
    for r in survivors:
        print(f"  {r['symbol']} hold={r['hold']} pct={r['pct']:.2f}  IS={r['IS_sharpe']:+.2f}  OOS={r['OOS_sharpe']:+.2f}")


if __name__ == "__main__":
    main()
