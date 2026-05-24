"""Live streaming: Binance WebSocket for crypto, OANDA REST-stream for FX / indices.

Each closed candle is appended to the local parquet store, so a backtester
re-reading the file picks up the bar the moment it closes.

Run from the CLI:

    python -m alphabeta stream                           # all 13 symbols, M1 + H1
    python -m alphabeta stream --symbol BTCUSDT          # one symbol
    python -m alphabeta stream --timeframe M1 M5 H1      # subset of timeframes
    python -m alphabeta stream --no-oanda                # crypto only
    python -m alphabeta stream --no-binance              # forex / indices only

OANDA's stream only emits ticks — we aggregate to a 1-minute candle here.
Higher timeframes are produced by resampling closed M1 bars. (Same trick the
upstream backend uses; OANDA doesn't ship 5m/15m candles over the stream.)
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone

import httpx
import pandas as pd
import websockets

from .config import settings
from .storage import append_candles
from .symbols import (
    ALL_SYMBOLS, BINANCE_TF, CRYPTO, FOREX, INDEX, SYMBOL_TYPE, TF_SECONDS,
    TIMEFRAMES, AssetType,
)

logger = logging.getLogger(__name__)

BINANCE_WS = "wss://stream.binance.com:9443/ws"
RECONNECT_DELAY = 5.0


# ---------------------------------------------------------------------------
# Binance (crypto) — native multi-timeframe streams per symbol
# ---------------------------------------------------------------------------

async def _stream_binance_symbol(symbol: str, timeframes: list[str]) -> None:
    """Subscribe to Binance kline streams for one symbol across many TFs.

    Binance emits an update per partial candle and again when it closes
    (`k.x == True`). We persist only on close so the parquet store contains
    finalized bars only.
    """
    streams = [f"{symbol.lower()}@kline_{BINANCE_TF[tf]}" for tf in timeframes]
    url = f"{BINANCE_WS}/{'/'.join(streams)}"
    tf_by_interval = {BINANCE_TF[tf]: tf for tf in timeframes}

    while True:
        try:
            async with websockets.connect(
                url, close_timeout=10, ping_timeout=30, ping_interval=20,
            ) as ws:
                logger.info("binance WS connected: %s tfs=%s", symbol, timeframes)
                async for msg in ws:
                    data = json.loads(msg)
                    k = data.get("k")
                    if not k or not k.get("x"):
                        continue  # not closed yet
                    tf = tf_by_interval.get(k["i"])
                    if not tf:
                        continue
                    df = pd.DataFrame([{
                        "timestamp": pd.to_datetime(k["t"], unit="ms", utc=True),
                        "open": float(k["o"]),
                        "high": float(k["h"]),
                        "low":  float(k["l"]),
                        "close": float(k["c"]),
                        "volume": float(k["v"]),
                    }])
                    append_candles(symbol, tf, df)
                    logger.info("  + %s %s @ %s close=%s",
                                symbol, tf, df["timestamp"].iloc[0].isoformat(), k["c"])
        except (websockets.ConnectionClosed, OSError) as e:
            logger.warning("binance WS dropped for %s (%s) — reconnect in %.1fs",
                           symbol, type(e).__name__, RECONNECT_DELAY)
            await asyncio.sleep(RECONNECT_DELAY)
        except Exception:
            logger.exception("binance WS error for %s — reconnect in %.1fs", symbol, RECONNECT_DELAY)
            await asyncio.sleep(RECONNECT_DELAY)


async def stream_binance(symbols: list[str], timeframes: list[str]) -> None:
    """Run a binance stream task per symbol, all sharing the event loop."""
    tasks = [asyncio.create_task(_stream_binance_symbol(s, timeframes)) for s in symbols]
    await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# OANDA (forex + indices) — single REST stream multiplexed across symbols
# ---------------------------------------------------------------------------

# Mapping of higher TF -> floor function. After we close an M1, we check
# whether we just crossed a 5/15/60/240/1440-minute boundary and, if so,
# resample the relevant slice and append.
_HIGHER_TFS = ["M5", "M15", "H1", "H4", "D1"]


def _floor_minute(ts: datetime, tf_seconds: int) -> datetime:
    epoch = int(ts.timestamp())
    return datetime.fromtimestamp(epoch - (epoch % tf_seconds), tz=timezone.utc)


def _emit_higher_tf(symbol: str, closed_minute_ts: datetime, m1_buffer: list[dict]) -> None:
    """When an M1 close crosses a higher-TF boundary, build that candle from
    the accumulated buffer and append it. The buffer holds enough M1 bars to
    cover the largest active TF (1 day = 1440 bars).
    """
    if not m1_buffer:
        return
    df_m1 = pd.DataFrame(m1_buffer)
    df_m1["timestamp"] = pd.to_datetime(df_m1["timestamp"], utc=True)
    df_m1 = df_m1.set_index("timestamp").sort_index()

    for tf in _HIGHER_TFS:
        secs = TF_SECONDS[tf]
        # Did the M1 close just complete a higher-TF bar?
        next_min = closed_minute_ts + timedelta(seconds=60)
        if int(next_min.timestamp()) % secs != 0:
            continue
        bucket_start = _floor_minute(closed_minute_ts, secs)
        chunk = df_m1.loc[bucket_start: bucket_start + timedelta(seconds=secs - 1)]
        if chunk.empty:
            continue
        agg = {
            "timestamp": bucket_start,
            "open": chunk["open"].iloc[0],
            "high": chunk["high"].max(),
            "low": chunk["low"].min(),
            "close": chunk["close"].iloc[-1],
            "volume": chunk["volume"].sum(),
        }
        append_candles(symbol, tf, pd.DataFrame([agg]))
        logger.info("  + %s %s @ %s close=%s",
                    symbol, tf, bucket_start.isoformat(), agg["close"])


async def stream_oanda(symbols: list[str]) -> None:
    """Open one OANDA pricing stream over all symbols, aggregate ticks to M1,
    then derive higher timeframes from closed M1 bars.
    """
    if not settings.oanda_api_key:
        raise RuntimeError("OANDA_API_KEY not set in .env — cannot stream OANDA")
    headers = {"Authorization": f"Bearer {settings.oanda_api_key}"}
    url = (
        f"{settings.oanda_base_url}/v3/accounts/{settings.oanda_account_id}"
        f"/pricing/stream?instruments={','.join(symbols)}"
    )

    # Per-instrument M1 aggregation state.
    cur_minute: dict[str, datetime] = {}
    cur_candle: dict[str, dict] = {}
    # Rolling buffer of recent M1 bars for higher-TF resampling.
    m1_buf: dict[str, list[dict]] = {s: [] for s in symbols}
    BUF_MAX = 60 * 24  # 1 day of M1 bars

    while True:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=120.0)) as client:
                async with client.stream("GET", url, headers=headers) as resp:
                    logger.info("oanda stream connected: %s", ",".join(symbols))
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        msg = json.loads(line)
                        if msg.get("type") != "PRICE":
                            continue
                        instr = msg.get("instrument")
                        if instr not in symbols:
                            continue
                        bids, asks = msg.get("bids", []), msg.get("asks", [])
                        if not bids or not asks:
                            continue
                        mid = (float(bids[0]["price"]) + float(asks[0]["price"])) / 2
                        tick_t = datetime.fromisoformat(
                            msg["time"].replace("000Z", "+00:00").replace("Z", "+00:00")
                        )
                        minute = tick_t.replace(second=0, microsecond=0)

                        cm = cur_minute.get(instr)
                        cc = cur_candle.get(instr)
                        if cm is None or minute > cm:
                            # Close out the previous minute, then start fresh.
                            if cc is not None:
                                append_candles(instr, "M1", pd.DataFrame([cc]))
                                m1_buf[instr].append(cc)
                                if len(m1_buf[instr]) > BUF_MAX:
                                    m1_buf[instr] = m1_buf[instr][-BUF_MAX:]
                                _emit_higher_tf(instr, cc["timestamp"], m1_buf[instr])
                                logger.debug("  + %s M1 @ %s close=%s",
                                             instr, cc["timestamp"].isoformat(), cc["close"])
                            cur_candle[instr] = {
                                "timestamp": minute,
                                "open": mid, "high": mid, "low": mid, "close": mid,
                                "volume": 1.0,
                            }
                            cur_minute[instr] = minute
                        else:
                            cc["high"] = max(cc["high"], mid)
                            cc["low"] = min(cc["low"], mid)
                            cc["close"] = mid
                            cc["volume"] += 1
        except (httpx.ReadTimeout, httpx.ConnectTimeout) as e:
            logger.warning("oanda stream timeout (%s) — reconnect in %.1fs",
                           type(e).__name__, RECONNECT_DELAY)
            await asyncio.sleep(RECONNECT_DELAY)
        except Exception:
            logger.exception("oanda stream error — reconnect in %.1fs", RECONNECT_DELAY)
            await asyncio.sleep(RECONNECT_DELAY)


# ---------------------------------------------------------------------------
# Combined runner
# ---------------------------------------------------------------------------

async def run(
    *,
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    binance: bool = True,
    oanda: bool = True,
) -> None:
    """Start streaming for the selected symbols. Runs forever until killed."""
    symbols = symbols or ALL_SYMBOLS
    timeframes = timeframes or ["M1", "M5", "M15", "H1", "H4", "D1"]

    crypto = [s for s in symbols if SYMBOL_TYPE[s] == AssetType.CRYPTO]
    fx_idx = [s for s in symbols if SYMBOL_TYPE[s] in (AssetType.FOREX, AssetType.INDEX)]

    tasks: list[asyncio.Task] = []
    if binance and crypto:
        tasks.append(asyncio.create_task(stream_binance(crypto, timeframes)))
    if oanda and fx_idx:
        tasks.append(asyncio.create_task(stream_oanda(fx_idx)))
    if not tasks:
        logger.warning("nothing to stream — check --symbol / --no-binance / --no-oanda flags")
        return
    logger.info("streaming binance=%d crypto, oanda=%d fx+idx, tfs=%s",
                len(crypto) if binance else 0, len(fx_idx) if oanda else 0, timeframes)
    await asyncio.gather(*tasks)
