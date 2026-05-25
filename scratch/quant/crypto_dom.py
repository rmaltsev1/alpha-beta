"""Crypto turn-of-month + day-21 short.

A priori hypothesis (not picked from data — these are *known* calendar
effects in equities, tested in crypto):
  - Turn-of-month: long crypto on days {1, 29, 30, 31} (TOM effect)
  - Options-expiry hangover: short crypto around day 21 (monthly options
    expiry on or near day 21 in retail crypto markets historically)

Sleeve runs across BTC + ETH + SOL.
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

TOM_DAYS = {1, 29, 30, 31}
SHORT_DAYS = {21}


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


def crypto_dom_signal(df):
    dom = df["timestamp"].dt.day
    pos = pd.Series(0.0, index=df.index)
    pos[dom.isin(TOM_DAYS)] = +1.0
    pos[dom.isin(SHORT_DAYS)] = -1.0
    return pos


def _scale_is(df_full, build_position, target):
    is_df = df_full[df_full["timestamp"] < SPLIT].reset_index(drop=True)
    is_pos = pd.Series(build_position(is_df).values, index=is_df.index)
    ret = np.log(is_df["close"] / is_df["close"].shift(1)).fillna(0)
    raw = is_pos.values * ret.values
    av = float(np.std(raw, ddof=0) * np.sqrt(_bpy(is_df["timestamp"].values)))
    return target/av if av > 1e-9 else 0


def main():
    sleeves = {}
    rows = []
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        df = get_candles(sym, "D1")
        scale = _scale_is(df, crypto_dom_signal, TARGET_VOL)
        pos = pd.Series(crypto_dom_signal(df).values, index=df.index) * scale
        res = backtest(df, pos, symbol=sym, timeframe="D1", name=f"DOM_{sym}")
        idx = pd.to_datetime(df["timestamp"].values, utc=True)
        rets = pd.Series(res.returns.values, index=idx)
        sleeves[sym] = rets
        s = stats(sym, rets)
        rows.append({"symbol": sym, "scale": scale, **s})
        print(f"{sym:<10}  scale={scale:5.2f}  IS_Sh={s.get('IS_sharpe',0):+5.2f}  "
              f"OOS_Sh={s.get('OOS_sharpe',0):+5.2f}")

    # Combine
    combined = pd.concat(sleeves, axis=1, sort=True).fillna(0.0).mean(axis=1)
    s_combined = stats("CRYPTO_DOM", combined)
    print(f"\nCombined sleeve:")
    for tag in ["FULL", "IS", "OOS"]:
        print(f"  {tag:<4} Sharpe={s_combined.get(f'{tag}_sharpe',0):+.2f}  "
              f"Return={s_combined.get(f'{tag}_ret',0):+.2%}")

    # Year-by-year
    for year, sub in combined.groupby(combined.index.year):
        if len(sub) < 50: continue
        bpy = _bpy(sub.index)
        sh = sub.mean() * bpy / (sub.std(ddof=0) * np.sqrt(bpy)) if sub.std() > 0 else 0
        print(f"  {year}  Sharpe={sh:+.2f}")

    pd.DataFrame({"timestamp": combined.index, "ret": combined.values}).to_parquet(
        OUT / "crypto_dom_returns.parquet", index=False)
    pd.DataFrame(rows).to_csv(OUT / "crypto_dom_breakdown.csv", index=False)


if __name__ == "__main__":
    main()
