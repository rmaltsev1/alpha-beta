"""Pull candles from prod postgres into local parquet.

Requires the SSH tunnel to be open: `./scripts/tunnel-prod.sh -b`.

Two modes:
  * incremental (default) — fetch rows newer than the latest local timestamp
  * full — re-fetch everything from BACKFILL_START

Uses a server-side cursor so a 3M-row M1 table doesn't load all at once.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterable, Iterator

import pandas as pd
import psycopg
from psycopg.rows import dict_row

from .config import settings
from .storage import append_candles, last_timestamp, load_candles, save_candles
from .symbols import ALL_SYMBOLS, TIMEFRAMES

logger = logging.getLogger(__name__)

BACKFILL_START = datetime(2020, 1, 1, tzinfo=timezone.utc)
CHUNK_SIZE = 50_000  # rows per cursor fetch


@contextmanager
def connect() -> Iterator[psycopg.Connection]:
    """Open a connection to prod postgres via the local tunnel."""
    conn = psycopg.connect(
        host=settings.db_host,
        port=settings.db_port,
        user=settings.db_user,
        password=settings.db_password,
        dbname=settings.db_name,
        connect_timeout=10,
        row_factory=dict_row,
    )
    try:
        yield conn
    finally:
        conn.close()


def _asset_id_map(conn: psycopg.Connection) -> dict[str, int]:
    with conn.cursor() as cur:
        cur.execute("SELECT id, symbol FROM assets")
        return {r["symbol"]: r["id"] for r in cur.fetchall()}


def _iter_candles(
    conn: psycopg.Connection,
    asset_id: int,
    timeframe: str,
    since: datetime,
) -> Iterable[list[dict]]:
    """Yield chunks of candle dicts from a server-side cursor.

    Server-side cursor is mandatory here — the M1 tables are 1-3M rows and
    pulling them all into client memory would blow the stack.
    """
    sql = (
        "SELECT timestamp, open, high, low, close, volume "
        "FROM candles "
        "WHERE asset_id = %s AND timeframe = %s::timeframe_enum AND timestamp >= %s "
        "ORDER BY timestamp ASC"
    )
    with conn.cursor(name=f"cur_{asset_id}_{timeframe}") as cur:
        cur.itersize = CHUNK_SIZE
        cur.execute(sql, (asset_id, timeframe, since))
        chunk: list[dict] = []
        for row in cur:
            chunk.append(row)
            if len(chunk) >= CHUNK_SIZE:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def fetch_one(
    symbol: str,
    timeframe: str,
    *,
    full: bool = False,
    conn: psycopg.Connection | None = None,
    asset_id_map: dict[str, int] | None = None,
) -> dict:
    """Refresh a single (symbol, timeframe) from prod.

    Returns a stats dict so the CLI / caller can log per-symbol progress.
    """
    own_conn = conn is None
    if own_conn:
        ctx = connect()
        conn = ctx.__enter__()  # noqa: PLR1704
    try:
        if asset_id_map is None:
            asset_id_map = _asset_id_map(conn)
        if symbol not in asset_id_map:
            raise KeyError(f"asset {symbol!r} not found in prod")

        if full:
            since = BACKFILL_START
        else:
            local_last = last_timestamp(symbol, timeframe)
            # Re-fetch the last bar in case it was a still-forming partial when we cached it.
            since = local_last if local_last else BACKFILL_START

        asset_id = asset_id_map[symbol]
        # Buffer all chunks in memory and write the parquet once at the end.
        # Appending per-chunk would re-read+rewrite the growing file on every
        # iteration (O(n²) on the M1 tables — 3M rows × 60+ rewrites).
        buffered: list[pd.DataFrame] = []
        total = 0
        for chunk in _iter_candles(conn, asset_id, timeframe, since):
            buffered.append(
                pd.DataFrame(chunk, columns=["timestamp", "open", "high", "low", "close", "volume"])
            )
            total += len(chunk)
            logger.debug("  %s %s +%d (running=%d)", symbol, timeframe, len(chunk), total)

        if buffered:
            df_new = pd.concat(buffered, ignore_index=True)
            if full:
                # Re-fetch overrides whatever was on disk.
                save_candles(symbol, timeframe, df_new)
            else:
                # Incremental: merge with existing.
                existing = load_candles(symbol, timeframe)
                merged = pd.concat([existing, df_new], ignore_index=True) if not existing.empty else df_new
                save_candles(symbol, timeframe, merged)

        return {"symbol": symbol, "timeframe": timeframe, "fetched": total, "since": since}
    finally:
        if own_conn:
            ctx.__exit__(None, None, None)


# Fetch order: from least to most rows. If the tunnel drops mid-job,
# the small-tf data we use most often is already on disk.
_TF_ORDER = ["W1", "D1", "H4", "H1", "M15", "M5", "M1"]


def fetch_all(
    *,
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    full: bool = False,
) -> list[dict]:
    """Refresh every (symbol, timeframe) combo, or a filtered subset."""
    symbols = symbols or ALL_SYMBOLS
    timeframes = sorted(timeframes or TIMEFRAMES, key=lambda t: _TF_ORDER.index(t) if t in _TF_ORDER else 99)
    results: list[dict] = []
    with connect() as conn:
        ids = _asset_id_map(conn)
        for tf in timeframes:
            for sym in symbols:
                if sym not in ids:
                    logger.warning("skip %s — asset not found", sym)
                    continue
                try:
                    r = fetch_one(sym, tf, full=full, conn=conn, asset_id_map=ids)
                    logger.info("  %s %s: fetched=%d since=%s",
                                sym, tf, r["fetched"], r["since"].isoformat())
                    results.append(r)
                except Exception as e:
                    logger.exception("  %s %s: FAILED — %s", sym, tf, type(e).__name__)
                    results.append({"symbol": sym, "timeframe": tf, "fetched": -1, "error": str(e)})
    return results
