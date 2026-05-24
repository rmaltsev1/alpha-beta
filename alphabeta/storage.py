"""Local parquet store for OHLCV candles.

Layout: data/<symbol>/<timeframe>.parquet
Schema: timestamp (datetime64[ns, UTC]), open/high/low/close/volume (float64)
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import settings


COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def parquet_path(symbol: str, timeframe: str) -> Path:
    return settings.data_dir / symbol / f"{timeframe}.parquet"


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, deduplicate, coerce to the canonical schema."""
    if df.empty:
        return df.astype({c: "float64" for c in COLUMNS if c != "timestamp"})

    df = df.loc[:, COLUMNS].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype("float64")
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.sort_values("timestamp", ignore_index=True)
    return df


def load_candles(symbol: str, timeframe: str) -> pd.DataFrame:
    """Return the local candle dataframe, or an empty one if nothing saved."""
    p = parquet_path(symbol, timeframe)
    if not p.exists():
        return pd.DataFrame(columns=COLUMNS).astype(
            {"timestamp": "datetime64[ns, UTC]", "open": "float64", "high": "float64",
             "low": "float64", "close": "float64", "volume": "float64"}
        )
    return pd.read_parquet(p)


def save_candles(symbol: str, timeframe: str, df: pd.DataFrame) -> Path:
    """Write the candle dataframe to parquet, replacing any existing file."""
    df = _normalize(df)
    p = parquet_path(symbol, timeframe)
    _ensure_parent(p)
    df.to_parquet(p, compression="snappy", index=False)
    return p


def append_candles(symbol: str, timeframe: str, df_new: pd.DataFrame) -> tuple[Path, int]:
    """Merge `df_new` into the existing local file. Returns (path, rows_after).

    Idempotent on the timestamp column — re-fetching overlapping ranges is safe.
    """
    df_new = _normalize(df_new)
    if df_new.empty:
        return parquet_path(symbol, timeframe), len(load_candles(symbol, timeframe))

    existing = load_candles(symbol, timeframe)
    merged = pd.concat([existing, df_new], ignore_index=True) if not existing.empty else df_new
    merged = _normalize(merged)
    return save_candles(symbol, timeframe, merged), len(merged)


def last_timestamp(symbol: str, timeframe: str) -> datetime | None:
    """Latest timestamp present locally, or None if no file."""
    p = parquet_path(symbol, timeframe)
    if not p.exists():
        return None
    # Cheap path: only read the timestamp column.
    ts = pd.read_parquet(p, columns=["timestamp"])
    if ts.empty:
        return None
    return ts["timestamp"].max().to_pydatetime().astimezone(timezone.utc)


def list_local() -> pd.DataFrame:
    """Inventory of what's on disk: rows, first/last timestamp, file size."""
    rows = []
    if not settings.data_dir.exists():
        return pd.DataFrame(columns=["symbol", "timeframe", "rows", "first", "last", "mb"])
    for sym_dir in sorted(settings.data_dir.iterdir()):
        if not sym_dir.is_dir():
            continue
        for f in sorted(sym_dir.glob("*.parquet")):
            ts = pd.read_parquet(f, columns=["timestamp"])
            size_mb = f.stat().st_size / 1_048_576
            rows.append({
                "symbol": sym_dir.name,
                "timeframe": f.stem,
                "rows": len(ts),
                "first": ts["timestamp"].min() if len(ts) else None,
                "last": ts["timestamp"].max() if len(ts) else None,
                "mb": round(size_mb, 2),
            })
    return pd.DataFrame(rows)
