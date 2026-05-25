"""Discrete paper-trade tracker.

Layers a discrete-trade view on top of the continuous exposure betas:

  - Open a paper trade when |β| crosses ABOVE OPEN_THRESHOLD from below.
  - Close it when |β| crosses BELOW CLOSE_THRESHOLD.
  - On sign flip: close existing + open new in opposite direction.

This gives the user a clean "trades-since-start" P&L track to compare against
the continuous portfolio P&L.

Per-trade P&L is computed against the instrument's spot price at open / close
(latest D1 close from the local parquet store). The position is treated as
unit-sized (1 unit of the instrument per unit of |β|); P&L is a return %.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from alphabeta import get_candles


OPEN_THRESHOLD = 0.05   # |β| must exceed this to open
CLOSE_THRESHOLD = 0.02  # |β| below this = close


SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    instrument TEXT NOT NULL,
    side TEXT NOT NULL,             -- 'LONG' | 'SHORT'
    open_bar_iso TEXT NOT NULL,
    open_price REAL NOT NULL,
    open_beta REAL NOT NULL,
    close_bar_iso TEXT,
    close_price REAL,
    close_beta REAL,
    return_pct REAL,                 -- signed (long: +ve when price up; short: +ve when price down)
    hold_days INTEGER,
    status TEXT NOT NULL,            -- 'OPEN' | 'CLOSED'
    created_ts REAL NOT NULL,
    closed_ts REAL
);
CREATE INDEX IF NOT EXISTS idx_pt_instr ON paper_trades(instrument, status);
CREATE INDEX IF NOT EXISTS idx_pt_status ON paper_trades(status);
"""


@dataclass(frozen=True)
class TradeEvent:
    """A discrete trade lifecycle event derived from an exposure shift."""
    kind: str              # 'OPEN' | 'CLOSE' | 'FLIP' | 'IGNORE'
    instrument: str
    side: Optional[str]    # 'LONG'|'SHORT' for OPEN/FLIP; None for CLOSE/IGNORE
    prev_beta: float
    new_beta: float
    ref_price: Optional[float] = None
    return_pct: Optional[float] = None  # filled in on CLOSE/FLIP-close-leg
    hold_days: Optional[int] = None


def _ensure_schema(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as c:
        c.executescript(SCHEMA)


def _current_price(instrument: str) -> Optional[float]:
    """Latest D1 close for the instrument."""
    try:
        df = get_candles(instrument, "D1")
    except Exception:
        return None
    if df is None or len(df) == 0:
        return None
    return float(df["close"].iloc[-1])


def _get_open_trade(db_path: Path, instrument: str) -> Optional[dict]:
    with sqlite3.connect(db_path) as c:
        c.row_factory = sqlite3.Row
        row = c.execute(
            "SELECT * FROM paper_trades WHERE instrument=? AND status='OPEN' "
            "ORDER BY id DESC LIMIT 1",
            (instrument,),
        ).fetchone()
    return dict(row) if row else None


def _open_trade(
    db_path: Path,
    instrument: str,
    side: str,
    bar_iso: str,
    price: float,
    beta: float,
) -> int:
    with sqlite3.connect(db_path) as c:
        cur = c.execute(
            "INSERT INTO paper_trades "
            "(instrument, side, open_bar_iso, open_price, open_beta, status, created_ts) "
            "VALUES (?, ?, ?, ?, ?, 'OPEN', ?)",
            (instrument, side, bar_iso, price, beta, time.time()),
        )
        return cur.lastrowid


def _close_trade(
    db_path: Path,
    trade_id: int,
    bar_iso: str,
    price: float,
    beta: float,
) -> tuple[float, int]:
    """Returns (return_pct, hold_days)."""
    with sqlite3.connect(db_path) as c:
        c.row_factory = sqlite3.Row
        row = c.execute(
            "SELECT side, open_bar_iso, open_price FROM paper_trades WHERE id=?",
            (trade_id,),
        ).fetchone()
        side = row["side"]
        open_bar = pd.Timestamp(row["open_bar_iso"])
        open_price = float(row["open_price"])
        close_bar = pd.Timestamp(bar_iso)
        hold_days = max(1, int((close_bar - open_bar).total_seconds() / 86400))
        raw_return = (price - open_price) / open_price if open_price > 0 else 0.0
        return_pct = raw_return if side == "LONG" else -raw_return
        c.execute(
            "UPDATE paper_trades "
            "SET close_bar_iso=?, close_price=?, close_beta=?, "
            "    return_pct=?, hold_days=?, status='CLOSED', closed_ts=? "
            "WHERE id=?",
            (bar_iso, price, beta, return_pct, hold_days, time.time(), trade_id),
        )
        return return_pct, hold_days


def process_exposure_shift(
    db_path: Path,
    instrument: str,
    prev_beta: float,
    new_beta: float,
    bar_iso: str,
) -> TradeEvent:
    """Process a single per-instrument exposure shift into a discrete-trade event.

    Returns a TradeEvent describing what happened.
    """
    _ensure_schema(db_path)
    open_now = abs(new_beta) >= OPEN_THRESHOLD
    closed_now = abs(new_beta) < CLOSE_THRESHOLD
    existing = _get_open_trade(db_path, instrument)
    price = _current_price(instrument)

    # Case 1: no existing trade, new exposure too small → ignore
    if existing is None and not open_now:
        return TradeEvent(
            kind="IGNORE", instrument=instrument, side=None,
            prev_beta=prev_beta, new_beta=new_beta, ref_price=price,
        )

    # Case 2: no existing trade, new exposure exceeds OPEN → open
    if existing is None and open_now:
        side = "LONG" if new_beta > 0 else "SHORT"
        if price is not None:
            _open_trade(db_path, instrument, side, bar_iso, price, new_beta)
        return TradeEvent(
            kind="OPEN", instrument=instrument, side=side,
            prev_beta=prev_beta, new_beta=new_beta, ref_price=price,
        )

    # Case 3: existing trade, exposure dropped below CLOSE → close
    if existing is not None and closed_now:
        if price is not None:
            ret, hd = _close_trade(db_path, existing["id"], bar_iso, price, new_beta)
            return TradeEvent(
                kind="CLOSE", instrument=instrument, side=None,
                prev_beta=prev_beta, new_beta=new_beta, ref_price=price,
                return_pct=ret, hold_days=hd,
            )
        return TradeEvent(kind="CLOSE", instrument=instrument, side=None,
                          prev_beta=prev_beta, new_beta=new_beta)

    # Case 4: existing trade, sign flip while still material → close + open
    if existing is not None and open_now and (
        (existing["side"] == "LONG" and new_beta < 0) or
        (existing["side"] == "SHORT" and new_beta > 0)
    ):
        if price is not None:
            ret, hd = _close_trade(db_path, existing["id"], bar_iso, price, new_beta)
            new_side = "LONG" if new_beta > 0 else "SHORT"
            _open_trade(db_path, instrument, new_side, bar_iso, price, new_beta)
            return TradeEvent(
                kind="FLIP", instrument=instrument, side=new_side,
                prev_beta=prev_beta, new_beta=new_beta, ref_price=price,
                return_pct=ret, hold_days=hd,
            )

    # Case 5: same side, just a resize → no discrete event
    return TradeEvent(
        kind="IGNORE", instrument=instrument, side=existing["side"] if existing else None,
        prev_beta=prev_beta, new_beta=new_beta, ref_price=price,
    )


def list_open_trades(db_path: Path) -> list[dict]:
    if not db_path.exists():
        return []
    with sqlite3.connect(db_path) as c:
        c.row_factory = sqlite3.Row
        rows = c.execute(
            "SELECT * FROM paper_trades WHERE status='OPEN' ORDER BY id DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def closed_today(db_path: Path, bar_iso: str) -> list[dict]:
    if not db_path.exists():
        return []
    with sqlite3.connect(db_path) as c:
        c.row_factory = sqlite3.Row
        rows = c.execute(
            "SELECT * FROM paper_trades WHERE close_bar_iso=? ORDER BY id DESC",
            (bar_iso,),
        ).fetchall()
    return [dict(r) for r in rows]


def trade_stats(db_path: Path) -> dict:
    """Aggregate stats over all closed trades."""
    if not db_path.exists():
        return {"n_closed": 0, "n_open": 0}
    with sqlite3.connect(db_path) as c:
        c.row_factory = sqlite3.Row
        closed = c.execute(
            "SELECT return_pct, hold_days FROM paper_trades WHERE status='CLOSED'"
        ).fetchall()
        n_open = c.execute(
            "SELECT COUNT(*) AS n FROM paper_trades WHERE status='OPEN'"
        ).fetchone()["n"]
    if not closed:
        return {"n_closed": 0, "n_open": n_open}
    rets = [float(r["return_pct"]) for r in closed]
    return {
        "n_closed": len(closed),
        "n_open": n_open,
        "avg_return_pct": float(np.mean(rets)),
        "median_return_pct": float(np.median(rets)),
        "hit_rate": float(sum(1 for r in rets if r > 0) / len(rets)),
        "best": max(rets),
        "worst": min(rets),
        "sum_return_pct": float(sum(rets)),
    }


def format_trades_block_html(db_path: Path, bar_iso: str) -> str:
    """Format a 'paper trades' summary block for the daily digest."""
    closed = closed_today(db_path, bar_iso)
    open_trades = list_open_trades(db_path)
    stats = trade_stats(db_path)

    lines = ["<b>📒 Paper trades</b>"]
    if stats["n_closed"] == 0 and not open_trades:
        lines.append("<i>No discrete trades yet (waiting for first material shift).</i>")
        return "\n".join(lines)

    lines.append("<pre>")
    if open_trades:
        lines.append(f"Open positions: {len(open_trades)}")
        for t in open_trades[:8]:
            since = pd.Timestamp(t["open_bar_iso"]).strftime("%m-%d")
            lines.append(
                f"  {t['side']:<5} {t['instrument']:<10} "
                f"opened {since} @ {t['open_price']:.2f} (β={t['open_beta']:+.2f})"
            )
    if closed:
        lines.append(f"\nClosed today: {len(closed)}")
        for t in closed[:8]:
            ret = float(t["return_pct"] or 0) * 100
            sign = "+" if ret >= 0 else ""
            lines.append(
                f"  {t['side']:<5} {t['instrument']:<10} "
                f"held {t['hold_days']}d, {sign}{ret:.2f}%"
            )
    if stats["n_closed"] > 0:
        lines.append(
            f"\nLifetime: {stats['n_closed']} closed, "
            f"hit-rate {stats['hit_rate']:.1%}, "
            f"avg/trade {stats['avg_return_pct']:+.2%}, "
            f"sum {stats['sum_return_pct']:+.2%}"
        )
    lines.append("</pre>")
    return "\n".join(lines)
