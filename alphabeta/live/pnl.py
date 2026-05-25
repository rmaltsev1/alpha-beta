"""Paper P&L tracker for the live signal pipeline.

v1: portfolio-level only. Reads the production v16 daily return series and updates
paper equity each time a new bar is closed.

The user starts with `STARTING_EQUITY` USD on `PAPER_START_DATE`. After that,
each bar-close updates equity = equity_prev * (1 + portfolio_return_bar).
We record MTM snapshots and surface metrics (drawdown, MTD/YTD, rolling Sharpe).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from alphabeta.config import settings
from alphabeta.live.state import State


REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_PARQUET = REPO_ROOT / "scratch" / "quant" / "PRODUCTION_v16_V4.parquet"
ALL_SLEEVES_PARQUET = REPO_ROOT / "scratch" / "quant" / "all_sleeve_returns_v16.parquet"

STARTING_EQUITY = 100_000.0
# Paper-trading starts from this date. Setting it to the OOS-test boundary
# (2024-01-01) means the first digest already shows a meaningful track record
# from the same data the backtest reports OOS. Override via PAPER_START_DATE env.
import os
PAPER_START_DATE = pd.Timestamp(
    os.getenv("PAPER_START_DATE", "2024-01-01"), tz="UTC"
)


@dataclass(frozen=True)
class MTMUpdate:
    bar_close_iso: str
    equity_before: float
    equity_after: float
    portfolio_return: float  # the bar return (decimal, e.g. 0.0123 = +1.23%)
    daily_pnl_usd: float
    drawdown_from_peak: float
    peak_equity: float
    mtd_return: float
    ytd_return: float
    since_start_return: float
    sleeve_attribution_top: list[tuple[str, float]]
    sleeve_attribution_bottom: list[tuple[str, float]]

    def to_dict(self) -> dict:
        return asdict(self)


def _load_portfolio_returns() -> pd.Series:
    """Load the master_v16 V4 production daily return series."""
    if not PRODUCTION_PARQUET.exists():
        raise FileNotFoundError(
            f"PRODUCTION_v16_V4.parquet not found at {PRODUCTION_PARQUET}. "
            f"Run `PYTHONPATH=. python scratch/quant/master_v16.py` first."
        )
    df = pd.read_parquet(PRODUCTION_PARQUET)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    s = pd.Series(df["ret"].values, index=df["timestamp"])
    return s.sort_index()


def _load_sleeve_returns() -> pd.DataFrame:
    """Load the per-sleeve daily returns panel (for attribution)."""
    df = pd.read_parquet(ALL_SLEEVES_PARQUET)
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def _sleeve_attribution_for_bar(
    sleeves: pd.DataFrame, bar_ts: pd.Timestamp, n_top: int = 3
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Return top-N and bottom-N sleeve contributions for the bar.

    Contribution is the sleeve's raw return on that bar (returns are already
    pre-scaled to comparable vol in the panel).
    """
    if bar_ts not in sleeves.index:
        # fallback: nearest prior date
        prior = sleeves.index[sleeves.index <= bar_ts]
        if len(prior) == 0:
            return [], []
        bar_ts = prior[-1]
    row = sleeves.loc[bar_ts]
    pairs = [(s, float(v)) for s, v in row.items() if not pd.isna(v) and abs(v) > 1e-9]
    pairs.sort(key=lambda x: x[1], reverse=True)
    top = pairs[:n_top]
    bottom = pairs[-n_top:][::-1]
    return top, bottom


def latest_mtm(state: State, *, force_bar: Optional[pd.Timestamp] = None) -> Optional[MTMUpdate]:
    """Compute the MTM update for the latest unprocessed bar (or `force_bar`).

    Returns None if no new bar to process. Idempotent: if the bar's MTM already
    exists in state, returns None.
    """
    returns = _load_portfolio_returns()
    sleeves = _load_sleeve_returns()

    if force_bar is not None:
        target_ts = force_bar
        if target_ts not in returns.index:
            raise ValueError(f"force_bar {target_ts} not in returns series")
    else:
        # Take the most recent bar
        if len(returns) == 0:
            return None
        target_ts = returns.index[-1]

    bar_iso = target_ts.isoformat()
    if state.already_fired("D1", bar_iso):
        return None

    # Build full equity series from PAPER_START_DATE forward, applying returns
    paper_returns = returns[returns.index >= PAPER_START_DATE].copy()
    paper_returns_upto = paper_returns[paper_returns.index <= target_ts]
    if len(paper_returns_upto) == 0:
        # Bar is before paper-start; record but no equity change
        return MTMUpdate(
            bar_close_iso=bar_iso,
            equity_before=STARTING_EQUITY,
            equity_after=STARTING_EQUITY,
            portfolio_return=0.0,
            daily_pnl_usd=0.0,
            drawdown_from_peak=0.0,
            peak_equity=STARTING_EQUITY,
            mtd_return=0.0,
            ytd_return=0.0,
            since_start_return=0.0,
            sleeve_attribution_top=[],
            sleeve_attribution_bottom=[],
        )

    equity_curve = STARTING_EQUITY * (1.0 + paper_returns_upto).cumprod()
    equity_after = float(equity_curve.iloc[-1])
    equity_before = float(equity_curve.iloc[-2]) if len(equity_curve) > 1 else STARTING_EQUITY
    bar_ret = float(paper_returns_upto.iloc[-1])

    # Drawdown from running peak
    running_peak = equity_curve.cummax()
    peak = float(running_peak.iloc[-1])
    dd = equity_after / peak - 1.0

    # MTD / YTD
    bar_dt = target_ts.tz_convert("UTC").date()
    month_start_ts = pd.Timestamp(date(bar_dt.year, bar_dt.month, 1), tz="UTC")
    year_start_ts = pd.Timestamp(date(bar_dt.year, 1, 1), tz="UTC")

    def _ret_since(start: pd.Timestamp) -> float:
        slc = paper_returns_upto[paper_returns_upto.index >= start]
        if len(slc) == 0:
            return 0.0
        return float((1.0 + slc).prod() - 1.0)

    mtd = _ret_since(month_start_ts)
    ytd = _ret_since(year_start_ts)
    since_start = float(equity_after / STARTING_EQUITY - 1.0)

    top, bottom = _sleeve_attribution_for_bar(sleeves, target_ts)

    update = MTMUpdate(
        bar_close_iso=bar_iso,
        equity_before=equity_before,
        equity_after=equity_after,
        portfolio_return=bar_ret,
        daily_pnl_usd=equity_after - equity_before,
        drawdown_from_peak=dd,
        peak_equity=peak,
        mtd_return=mtd,
        ytd_return=ytd,
        since_start_return=since_start,
        sleeve_attribution_top=top,
        sleeve_attribution_bottom=bottom,
    )
    return update


def persist_mtm(state: State, update: MTMUpdate) -> None:
    """Record an MTMUpdate into the state DB."""
    state.record_mtm(
        bar_close_iso=update.bar_close_iso,
        equity=update.equity_after,
        portfolio_return=update.portfolio_return,
        daily_pnl_usd=update.daily_pnl_usd,
        open_positions={},  # v1 doesn't track per-instrument positions yet
        sleeve_attribution={
            "top": update.sleeve_attribution_top,
            "bottom": update.sleeve_attribution_bottom,
        },
    )


def rolling_sharpe(state: State, window_days: int = 30) -> Optional[float]:
    """Compute trailing rolling Sharpe from persisted MTM history."""
    with state._conn() as c:
        rows = c.execute(
            "SELECT portfolio_return FROM mtm ORDER BY bar_close_iso DESC LIMIT ?",
            (window_days,),
        ).fetchall()
    if len(rows) < 5:
        return None
    arr = np.array([r["portfolio_return"] for r in rows], dtype=float)
    if arr.std(ddof=0) < 1e-9:
        return None
    return float(arr.mean() / arr.std(ddof=0) * np.sqrt(252))
