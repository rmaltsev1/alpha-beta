"""Refetch candles directly from Binance / OANDA — no DB or SSH tunnel needed.

Mirrors the pagination logic in rektfree-backend's scripts/backfill_to_2020.py
so the local store stays consistent whether we pulled from prod or from the
original upstream APIs.

Crypto (BTCUSDT/ETHUSDT/SOLUSDT) → Binance (public, no key).
Forex + indices → OANDA (needs OANDA_API_KEY in .env).
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

import httpx
import pandas as pd

from .config import settings
from .storage import append_candles, last_timestamp
from .symbols import (
    ALL_SYMBOLS, BINANCE_TF, CRYPTO, OANDA_TF, SYMBOL_TYPE, TF_SECONDS,
    TIMEFRAMES, AssetType,
)

logger = logging.getLogger(__name__)

BACKFILL_START = datetime(2020, 1, 1, tzinfo=timezone.utc)
BINANCE_REST = "https://api.binance.com/api/v3"
BINANCE_LIMIT = 1000
OANDA_LIMIT = 5000


def _binance_paginated(client: httpx.Client, symbol: str, tf: str, since: datetime) -> pd.DataFrame:
    """Fetch all Binance klines from `since` to now, paginated 1000 at a time."""
    interval = BINANCE_TF[tf]
    start_ms = int(since.timestamp() * 1000)
    rows: list[list] = []
    while True:
        resp = client.get(
            f"{BINANCE_REST}/klines",
            params={"symbol": symbol, "interval": interval, "startTime": start_ms, "limit": BINANCE_LIMIT},
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        rows.extend(data)
        last_open = data[-1][0]
        start_ms = last_open + TF_SECONDS[tf] * 1000
        if len(data) < BINANCE_LIMIT:
            break
        time.sleep(0.1)
    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame({
        "timestamp": pd.to_datetime([r[0] for r in rows], unit="ms", utc=True),
        "open":   [float(r[1]) for r in rows],
        "high":   [float(r[2]) for r in rows],
        "low":    [float(r[3]) for r in rows],
        "close":  [float(r[4]) for r in rows],
        "volume": [float(r[5]) for r in rows],
    })
    return df


def _oanda_paginated(client: httpx.Client, instrument: str, tf: str, since: datetime) -> pd.DataFrame:
    """Fetch all OANDA candles from `since` to now, paginated 5000 at a time."""
    if not settings.oanda_api_key:
        raise RuntimeError("OANDA_API_KEY not set in .env")
    headers = {"Authorization": f"Bearer {settings.oanda_api_key}"}
    granularity = OANDA_TF[tf]
    start = since
    rows: list[dict] = []
    while True:
        url = (
            f"{settings.oanda_base_url}/v3/instruments/{instrument}/candles"
            f"?granularity={granularity}&from={start.strftime('%Y-%m-%dT%H:%M:%SZ')}"
            f"&count={OANDA_LIMIT}&price=M"
        )
        resp = client.get(url, headers=headers)
        resp.raise_for_status()
        candles = resp.json().get("candles", [])
        if not candles:
            break
        for c in candles:
            if not c.get("complete", False):
                continue
            mid = c["mid"]
            rows.append({
                "timestamp": c["time"],
                "open": float(mid["o"]),
                "high": float(mid["h"]),
                "low":  float(mid["l"]),
                "close": float(mid["c"]),
                "volume": float(c.get("volume", 0)),
            })
        last_time = datetime.fromisoformat(
            candles[-1]["time"].replace("000Z", "+00:00").replace("Z", "+00:00")
        )
        start = last_time + timedelta(seconds=TF_SECONDS[tf])
        if len(candles) < OANDA_LIMIT:
            break
        time.sleep(0.2)
    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"].str.replace("000Z", "+00:00").str.replace("Z", "+00:00"), utc=True)
    return df


def fetch_one(symbol: str, timeframe: str, *, full: bool = False) -> dict:
    """Refresh a single (symbol, timeframe) directly from the upstream API."""
    since = BACKFILL_START if full else (last_timestamp(symbol, timeframe) or BACKFILL_START)
    kind = SYMBOL_TYPE[symbol]
    with httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
        if kind == AssetType.CRYPTO:
            df = _binance_paginated(client, symbol, timeframe, since)
        else:
            df = _oanda_paginated(client, symbol, timeframe, since)
    append_candles(symbol, timeframe, df)
    return {"symbol": symbol, "timeframe": timeframe, "fetched": len(df), "since": since}


def fetch_all(
    *,
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    full: bool = False,
) -> list[dict]:
    symbols = symbols or ALL_SYMBOLS
    timeframes = timeframes or TIMEFRAMES
    results: list[dict] = []
    for tf in timeframes:
        for sym in symbols:
            try:
                r = fetch_one(sym, tf, full=full)
                logger.info("  %s %s: fetched=%d since=%s",
                            sym, tf, r["fetched"], r["since"].isoformat())
                results.append(r)
            except Exception as e:
                logger.exception("  %s %s: FAILED — %s", sym, tf, type(e).__name__)
                results.append({"symbol": sym, "timeframe": tf, "fetched": -1, "error": str(e)})
            time.sleep(0.5)  # be nice to upstreams
    return results
