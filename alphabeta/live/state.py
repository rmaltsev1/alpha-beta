"""SQLite-backed state for the live paper-trading layer.

Tables:
  positions      — current target position per (sleeve, instrument)
  signals        — append-only log of every position-change event
  fills          — paper "fills" with reference price + cost applied
  mtm            — daily mark-to-market snapshots for equity curve
  bar_fires      — log of each bar-close fire (for idempotency / catch-up)
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from alphabeta.config import settings
from alphabeta.live.signals import Signal


DEFAULT_DB = settings.data_dir / "live" / "paper_state.sqlite"


SCHEMA = """
CREATE TABLE IF NOT EXISTS positions (
    sleeve TEXT NOT NULL,
    instrument TEXT NOT NULL,
    position REAL NOT NULL DEFAULT 0.0,
    updated_ts REAL NOT NULL,
    bar_close_iso TEXT NOT NULL,
    PRIMARY KEY (sleeve, instrument)
);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bar_close_iso TEXT NOT NULL,
    sleeve TEXT NOT NULL,
    instrument TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    prev_position REAL NOT NULL,
    new_position REAL NOT NULL,
    direction TEXT NOT NULL,
    ref_price REAL,
    notional_usd REAL,
    note TEXT,
    payload_json TEXT,
    created_ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_signals_bar ON signals(bar_close_iso);
CREATE INDEX IF NOT EXISTS idx_signals_sleeve ON signals(sleeve, instrument);

CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER REFERENCES signals(id),
    bar_close_iso TEXT NOT NULL,
    instrument TEXT NOT NULL,
    delta_position REAL NOT NULL,
    ref_price REAL NOT NULL,
    cost_bps REAL NOT NULL,
    created_ts REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS mtm (
    bar_close_iso TEXT PRIMARY KEY,
    equity REAL NOT NULL,
    portfolio_return REAL NOT NULL,
    daily_pnl_usd REAL NOT NULL,
    open_positions_json TEXT NOT NULL,
    sleeve_attribution_json TEXT,
    created_ts REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS bar_fires (
    fire_key TEXT PRIMARY KEY,        -- "<timeframe>:<bar_iso>"
    timeframe TEXT NOT NULL,
    bar_close_iso TEXT NOT NULL,
    fired_ts REAL NOT NULL,
    status TEXT NOT NULL,             -- "ok" | "skipped" | "error"
    signals_emitted INTEGER NOT NULL DEFAULT 0,
    error_msg TEXT
);
CREATE INDEX IF NOT EXISTS idx_fires_tf ON bar_fires(timeframe, bar_close_iso);

CREATE TABLE IF NOT EXISTS exposure_snapshots (
    bar_close_iso TEXT PRIMARY KEY,
    window_days INTEGER NOT NULL,
    r_squared REAL NOT NULL,
    betas_json TEXT NOT NULL,
    created_ts REAL NOT NULL
);
"""


class State:
    """Persistence layer for the live signal pipeline."""

    def __init__(self, path: Optional[Path] = None):
        self.path = path or DEFAULT_DB
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, isolation_level=None)  # autocommit
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # ---------------- positions ----------------

    def get_position(self, sleeve: str, instrument: str) -> float:
        with self._conn() as c:
            row = c.execute(
                "SELECT position FROM positions WHERE sleeve=? AND instrument=?",
                (sleeve, instrument),
            ).fetchone()
        return float(row["position"]) if row else 0.0

    def get_all_positions(self) -> dict[tuple[str, str], float]:
        with self._conn() as c:
            rows = c.execute("SELECT sleeve, instrument, position FROM positions").fetchall()
        return {(r["sleeve"], r["instrument"]): float(r["position"]) for r in rows}

    def get_portfolio_positions(self) -> dict[str, float]:
        """Net per-instrument exposure summed across sleeves."""
        out: dict[str, float] = {}
        for (sleeve, instr), p in self.get_all_positions().items():
            out[instr] = out.get(instr, 0.0) + p
        return out

    def set_position(
        self, sleeve: str, instrument: str, position: float, bar_close_iso: str
    ) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO positions (sleeve, instrument, position, updated_ts, bar_close_iso) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(sleeve, instrument) DO UPDATE SET "
                "position=excluded.position, updated_ts=excluded.updated_ts, "
                "bar_close_iso=excluded.bar_close_iso",
                (sleeve, instrument, position, time.time(), bar_close_iso),
            )

    # ---------------- signals ----------------

    def append_signal(self, sig: Signal) -> int:
        d = sig.to_dict()
        payload = json.dumps(d, default=str)
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO signals "
                "(bar_close_iso, sleeve, instrument, timeframe, prev_position, new_position, "
                " direction, ref_price, notional_usd, note, payload_json, created_ts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    sig.bar_close.isoformat(),
                    sig.sleeve,
                    sig.instrument,
                    sig.timeframe,
                    sig.prev_position,
                    sig.new_position,
                    sig.direction.value,
                    sig.ref_price,
                    sig.notional_usd,
                    sig.note,
                    payload,
                    time.time(),
                ),
            )
            return cur.lastrowid

    def recent_signals(self, limit: int = 50) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM signals ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ---------------- fills ----------------

    def record_fill(
        self,
        signal_id: int,
        bar_close_iso: str,
        instrument: str,
        delta_position: float,
        ref_price: float,
        cost_bps: float,
    ) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO fills "
                "(signal_id, bar_close_iso, instrument, delta_position, ref_price, cost_bps, created_ts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (signal_id, bar_close_iso, instrument, delta_position, ref_price, cost_bps, time.time()),
            )
            return cur.lastrowid

    # ---------------- mtm ----------------

    def record_mtm(
        self,
        bar_close_iso: str,
        equity: float,
        portfolio_return: float,
        daily_pnl_usd: float,
        open_positions: dict[str, float],
        sleeve_attribution: Optional[dict[str, float]] = None,
    ) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO mtm "
                "(bar_close_iso, equity, portfolio_return, daily_pnl_usd, "
                " open_positions_json, sleeve_attribution_json, created_ts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(bar_close_iso) DO UPDATE SET "
                "equity=excluded.equity, portfolio_return=excluded.portfolio_return, "
                "daily_pnl_usd=excluded.daily_pnl_usd, "
                "open_positions_json=excluded.open_positions_json, "
                "sleeve_attribution_json=excluded.sleeve_attribution_json",
                (
                    bar_close_iso,
                    equity,
                    portfolio_return,
                    daily_pnl_usd,
                    json.dumps(open_positions),
                    json.dumps(sleeve_attribution) if sleeve_attribution else None,
                    time.time(),
                ),
            )

    def latest_equity(self) -> Optional[float]:
        with self._conn() as c:
            row = c.execute(
                "SELECT equity FROM mtm ORDER BY bar_close_iso DESC LIMIT 1"
            ).fetchone()
        return float(row["equity"]) if row else None

    # ---------------- bar fires (idempotency) ----------------

    @staticmethod
    def fire_key(timeframe: str, bar_close_iso: str) -> str:
        return f"{timeframe}:{bar_close_iso}"

    def already_fired(self, timeframe: str, bar_close_iso: str) -> bool:
        with self._conn() as c:
            row = c.execute(
                "SELECT 1 FROM bar_fires WHERE fire_key=? AND status='ok'",
                (self.fire_key(timeframe, bar_close_iso),),
            ).fetchone()
        return row is not None

    def record_exposure(
        self,
        bar_close_iso: str,
        window_days: int,
        r_squared: float,
        betas: dict[str, float],
    ) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO exposure_snapshots "
                "(bar_close_iso, window_days, r_squared, betas_json, created_ts) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(bar_close_iso) DO UPDATE SET "
                "window_days=excluded.window_days, r_squared=excluded.r_squared, "
                "betas_json=excluded.betas_json",
                (bar_close_iso, window_days, r_squared, json.dumps(betas), time.time()),
            )

    def latest_exposure_betas(self) -> Optional[dict[str, float]]:
        """Get most recent betas (excluding the bar at `before_iso` if given)."""
        with self._conn() as c:
            row = c.execute(
                "SELECT betas_json FROM exposure_snapshots "
                "ORDER BY bar_close_iso DESC LIMIT 1"
            ).fetchone()
        if not row:
            return None
        return json.loads(row["betas_json"])

    def exposure_before(self, bar_iso: str) -> Optional[dict[str, float]]:
        with self._conn() as c:
            row = c.execute(
                "SELECT betas_json FROM exposure_snapshots "
                "WHERE bar_close_iso < ? ORDER BY bar_close_iso DESC LIMIT 1",
                (bar_iso,),
            ).fetchone()
        if not row:
            return None
        return json.loads(row["betas_json"])

    def record_fire(
        self,
        timeframe: str,
        bar_close_iso: str,
        status: str,
        signals_emitted: int = 0,
        error_msg: Optional[str] = None,
    ) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO bar_fires "
                "(fire_key, timeframe, bar_close_iso, fired_ts, status, signals_emitted, error_msg) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(fire_key) DO UPDATE SET "
                "fired_ts=excluded.fired_ts, status=excluded.status, "
                "signals_emitted=excluded.signals_emitted, error_msg=excluded.error_msg",
                (
                    self.fire_key(timeframe, bar_close_iso),
                    timeframe,
                    bar_close_iso,
                    time.time(),
                    status,
                    signals_emitted,
                    error_msg,
                ),
            )
