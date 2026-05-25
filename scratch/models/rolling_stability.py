"""Year-by-year Sharpe per sleeve + portfolio — does any sleeve fall apart in a single calendar year?"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from alphabeta import get_candles
from alphabeta.backtest import backtest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strategies_v2 as S2

SPLIT_DATE = "2024-01-01"
TARGET_VOL = 0.05


def _bpy(df):
    s = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 86400
    return len(df) / s * 365.25 if s > 0 else 252.0


def _scale_is(df_full, build_pos, target):
    split = pd.Timestamp(SPLIT_DATE, tz="UTC")
    is_df = df_full[df_full["timestamp"] < split].reset_index(drop=True)
    is_pos = pd.Series(build_pos(is_df).values, index=is_df.index)
    ret = np.log(is_df["close"] / is_df["close"].shift(1)).fillna(0)
    raw = is_pos.values * ret.values
    av = float(np.std(raw, ddof=0) * np.sqrt(_bpy(is_df)))
    return target/av if av > 1e-9 else 0


def sleeve_returns(symbol, timeframe, build_pos):
    df = get_candles(symbol, timeframe)
    scale = _scale_is(df, build_pos, TARGET_VOL)
    full_pos = pd.Series(build_pos(df).values, index=df.index) * scale
    res = backtest(df, full_pos, symbol=symbol, timeframe=timeframe, name="")
    idx = pd.to_datetime(df["timestamp"].values, utc=True)
    return pd.Series(res.returns.values, index=idx)


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
    streams = {}
    for name, sym, tf, fn in sleeves:
        streams[name] = sleeve_returns(sym, tf, fn)
    df = pd.concat(streams, axis=1, sort=True).fillna(0.0)
    df["PORTFOLIO"] = df.mean(axis=1)

    # Group by calendar year
    annual = {}
    for year, sub in df.groupby(df.index.year):
        span = (sub.index[-1] - sub.index[0]).total_seconds() / 86400
        if span < 60:
            continue
        bpy = len(sub) / span * 365.25
        row = {}
        for col in df.columns:
            r = sub[col]
            ann_ret = float(r.mean()) * bpy
            ann_vol = float(r.std(ddof=0)) * np.sqrt(bpy)
            row[col] = ann_ret / ann_vol if ann_vol > 1e-9 else 0.0
        annual[year] = row
    annual_df = pd.DataFrame(annual).T
    annual_df.index.name = "year"

    print("Sharpe by calendar year:")
    pd.set_option("display.width", 200)
    print(annual_df.round(2).to_string())
    annual_df.to_csv(Path(__file__).resolve().parent / "annual_sharpes.csv")


if __name__ == "__main__":
    main()
