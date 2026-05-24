"""High-level data loader for backtests.

This is the API a strategy file imports — it doesn't care whether the data
lives on disk because we pulled it from prod or from the exchange APIs.

    from alphabeta.data import get_candles
    df = get_candles("BTCUSDT", "H1", start="2023-01-01", end="2024-01-01")
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import pandas as pd

from .storage import load_candles, parquet_path
from .symbols import ALL_SYMBOLS, TIMEFRAMES


def _coerce_ts(x: str | datetime | pd.Timestamp | None) -> pd.Timestamp | None:
    if x is None:
        return None
    ts = pd.to_datetime(x, utc=True)
    return ts


def get_candles(
    symbol: str,
    timeframe: str,
    start: str | datetime | pd.Timestamp | None = None,
    end: str | datetime | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return a candle dataframe sliced to [start, end].

    Raises FileNotFoundError if no local data — the caller should run
    `python -m alphabeta fetch` first.
    """
    if symbol not in ALL_SYMBOLS:
        raise ValueError(f"unknown symbol {symbol!r}; valid: {ALL_SYMBOLS}")
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"unknown timeframe {timeframe!r}; valid: {TIMEFRAMES}")

    p = parquet_path(symbol, timeframe)
    if not p.exists():
        raise FileNotFoundError(
            f"no local data for {symbol} {timeframe} — run "
            f"`python -m alphabeta fetch --symbol {symbol} --timeframe {timeframe}` first"
        )

    df = load_candles(symbol, timeframe)
    s = _coerce_ts(start)
    e = _coerce_ts(end)
    if s is not None:
        df = df[df["timestamp"] >= s]
    if e is not None:
        df = df[df["timestamp"] < e]
    return df.reset_index(drop=True)


def get_many(
    symbols: Iterable[str],
    timeframe: str,
    start=None,
    end=None,
) -> dict[str, pd.DataFrame]:
    """Same as get_candles but for several symbols at once."""
    return {s: get_candles(s, timeframe, start, end) for s in symbols}
