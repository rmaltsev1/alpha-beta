"""On-demand console report for the live paper-trading layer.

Renders a one-shot terminal dashboard summarizing:
  - Portfolio P&L (equity, MTD, YTD, DD, Sharpe)
  - Currently open paper trades with mark-to-market unrealized P&L
  - Recently closed paper trades with realized P&L
  - Lifetime per-trade stats (hit rate, avg, sum, best/worst)
  - Tracking error vs the backtest's predicted return for the same window

Invoked via `python -m alphabeta live --report`. No Telegram send, no DB writes.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from alphabeta import get_candles
from alphabeta.config import settings
from alphabeta.live.pnl import (
    PAPER_START_DATE,
    PRODUCTION_PARQUET,
    STARTING_EQUITY,
    _load_portfolio_returns,
    rolling_sharpe,
)
from alphabeta.live.state import State


PAPER_TRADES_DB = settings.data_dir / "live" / "paper_trades.sqlite"


def _current_price(instrument: str) -> Optional[float]:
    try:
        df = get_candles(instrument, "D1")
        return float(df["close"].iloc[-1]) if df is not None and len(df) else None
    except Exception:
        return None


def _h_rule(width: int = 70) -> str:
    return "─" * width


def _portfolio_block() -> list[str]:
    """Render the portfolio P&L block."""
    returns = _load_portfolio_returns()
    paper = returns[returns.index >= PAPER_START_DATE]
    if len(paper) == 0:
        return ["  (no paper-trade data yet)"]
    equity_curve = STARTING_EQUITY * (1 + paper).cumprod()
    equity = float(equity_curve.iloc[-1])
    bar_iso = paper.index[-1].isoformat()
    bar_short = bar_iso[:10]

    peak = float(equity_curve.cummax().iloc[-1])
    dd = equity / peak - 1.0
    since_start = equity / STARTING_EQUITY - 1.0

    today_iso = paper.index[-1].date()
    month_start = pd.Timestamp(today_iso.replace(day=1), tz="UTC")
    year_start = pd.Timestamp(f"{today_iso.year}-01-01", tz="UTC")
    mtd = float((1 + paper[paper.index >= month_start]).prod() - 1.0)
    ytd = float((1 + paper[paper.index >= year_start]).prod() - 1.0)

    state = State()
    sharpe_30d = rolling_sharpe(state, window_days=30)

    annualized_ret = float(paper.mean() * 252)
    annualized_vol = float(paper.std(ddof=0) * np.sqrt(252))
    overall_sharpe = annualized_ret / annualized_vol if annualized_vol > 1e-9 else 0.0

    lines = [
        "PAPER PORTFOLIO".center(70),
        _h_rule(),
        f"  As of bar:        {bar_short}",
        f"  Starting equity:  ${STARTING_EQUITY:>14,.2f}    (paper-start: {PAPER_START_DATE.date()})",
        f"  Current equity:   ${equity:>14,.2f}    ({since_start:+.2%} since start)",
        f"  Peak equity:      ${peak:>14,.2f}",
        f"  Current DD:       {dd:>+14.2%}",
        "",
        f"  MTD return:       {mtd:>+14.2%}",
        f"  YTD return:       {ytd:>+14.2%}",
        "",
        f"  Sharpe (lifetime): {overall_sharpe:>+13.2f}",
        f"  Sharpe (30d):      "
        + (f"{sharpe_30d:>+13.2f}" if sharpe_30d is not None else "      (need more data)"),
        f"  Annualized vol:   {annualized_vol:>+14.2%}",
        f"  Annualized ret:   {annualized_ret:>+14.2%}",
    ]
    return lines


def _open_trades_block() -> list[str]:
    """Render the open paper trades table with mark-to-market unrealized P&L."""
    if not PAPER_TRADES_DB.exists():
        return ["OPEN TRADES".center(70), _h_rule(), "  (no paper-trades DB)", ""]
    with sqlite3.connect(PAPER_TRADES_DB) as c:
        c.row_factory = sqlite3.Row
        rows = c.execute(
            "SELECT * FROM paper_trades WHERE status='OPEN' ORDER BY id"
        ).fetchall()
    if not rows:
        return ["OPEN TRADES".center(70), _h_rule(), "  (no open trades)", ""]

    lines = [
        "OPEN TRADES".center(70),
        _h_rule(),
        f"  {'ID':>3} {'Instrument':<11} {'Side':<5} {'Opened':<10} "
        f"{'Entry':>10} {'Now':>10} {'β':>6} {'Unreal':>8} {'Days':>5}",
        f"  {'-'*3} {'-'*11} {'-'*5} {'-'*10} {'-'*10} {'-'*10} {'-'*6} {'-'*8} {'-'*5}",
    ]
    total_unreal = 0.0
    n = 0
    now = datetime.now(timezone.utc)
    for r in rows:
        instr = r["instrument"]
        side = r["side"]
        entry = float(r["open_price"])
        beta = float(r["open_beta"])
        opened = pd.Timestamp(r["open_bar_iso"])
        now_price = _current_price(instr)
        if now_price is None:
            unreal_pct = 0.0
            now_str = "n/a"
        else:
            raw = (now_price - entry) / entry if entry > 0 else 0.0
            unreal_pct = raw if side == "LONG" else -raw
            now_str = f"{now_price:>10,.2f}"
        hold_days = max(1, int((now - opened).total_seconds() / 86400))
        unreal_str = f"{unreal_pct*100:+.2f}%"
        total_unreal += unreal_pct
        n += 1
        lines.append(
            f"  {r['id']:>3} {instr:<11} {side:<5} {opened.strftime('%Y-%m-%d'):<10} "
            f"{entry:>10,.2f} {now_str} {beta:>+6.2f} {unreal_str:>8} {hold_days:>5}"
        )
    lines.append("")
    if n > 0:
        avg = total_unreal / n
        lines.append(f"  Total open: {n}    Avg unrealized: {avg*100:+.2f}%    Sum: {total_unreal*100:+.2f}%")
    return lines + [""]


def _closed_trades_block(limit: int = 20) -> list[str]:
    """Render recently closed paper trades with realized P&L."""
    if not PAPER_TRADES_DB.exists():
        return []
    with sqlite3.connect(PAPER_TRADES_DB) as c:
        c.row_factory = sqlite3.Row
        rows = c.execute(
            "SELECT * FROM paper_trades WHERE status='CLOSED' "
            "ORDER BY closed_ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
    if not rows:
        return [
            f"CLOSED TRADES (last {limit})".center(70),
            _h_rule(),
            "  (no closed trades yet — they appear when |β| drops below 0.02)",
            "",
        ]
    lines = [
        f"CLOSED TRADES (last {limit})".center(70),
        _h_rule(),
        f"  {'ID':>3} {'Instrument':<11} {'Side':<5} {'Opened':<10} {'Closed':<10} "
        f"{'Entry':>10} {'Exit':>10} {'Return':>8} {'Days':>5}",
        f"  {'-'*3} {'-'*11} {'-'*5} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*5}",
    ]
    for r in rows:
        ret = (r["return_pct"] or 0.0) * 100
        opened = pd.Timestamp(r["open_bar_iso"]).strftime("%Y-%m-%d")
        closed = pd.Timestamp(r["close_bar_iso"]).strftime("%Y-%m-%d") if r["close_bar_iso"] else "—"
        lines.append(
            f"  {r['id']:>3} {r['instrument']:<11} {r['side']:<5} {opened:<10} {closed:<10} "
            f"{r['open_price']:>10,.2f} {r['close_price']:>10,.2f} "
            f"{ret:>+7.2f}% {r['hold_days'] or 0:>5}"
        )
    return lines + [""]


def _lifetime_stats_block() -> list[str]:
    if not PAPER_TRADES_DB.exists():
        return []
    with sqlite3.connect(PAPER_TRADES_DB) as c:
        c.row_factory = sqlite3.Row
        closed = c.execute(
            "SELECT return_pct, hold_days, instrument FROM paper_trades WHERE status='CLOSED'"
        ).fetchall()
    if not closed:
        return ["LIFETIME PER-TRADE STATS".center(70), _h_rule(),
                "  (no closed trades yet)", ""]
    rets = np.array([float(r["return_pct"]) for r in closed])
    holds = np.array([int(r["hold_days"] or 1) for r in closed])
    n = len(rets)
    hit = (rets > 0).mean()
    # Per-instrument breakdown
    per_instr: dict[str, list[float]] = {}
    for r in closed:
        per_instr.setdefault(r["instrument"], []).append(float(r["return_pct"]))
    lines = [
        "LIFETIME PER-TRADE STATS".center(70),
        _h_rule(),
        f"  Closed trades:       {n}",
        f"  Hit rate:            {hit:.1%}",
        f"  Mean return / trade: {rets.mean()*100:+.2f}%",
        f"  Median:              {np.median(rets)*100:+.2f}%",
        f"  Best:                {rets.max()*100:+.2f}%",
        f"  Worst:               {rets.min()*100:+.2f}%",
        f"  Sum (cumulative):    {rets.sum()*100:+.2f}%",
        f"  Mean hold:           {holds.mean():.1f} days",
        "",
        "  Per-instrument:",
        f"  {'Instrument':<11} {'N':>4} {'Hit':>6} {'Avg':>8} {'Sum':>8}",
        f"  {'-'*11} {'-'*4} {'-'*6} {'-'*8} {'-'*8}",
    ]
    for instr, vals in sorted(per_instr.items(), key=lambda x: -abs(sum(x[1]))):
        arr = np.array(vals)
        lines.append(
            f"  {instr:<11} {len(arr):>4} {(arr>0).mean():>6.1%} "
            f"{arr.mean()*100:>+7.2f}% {arr.sum()*100:>+7.2f}%"
        )
    return lines + [""]


def _tracking_error_block() -> list[str]:
    """Estimate tracking error: paper portfolio vs the master_v16 backtest.

    Since paper = master_v16 series in v1, this will be ~0 until we have
    true paper fills diverging. Surfaced anyway to make the framework visible.
    """
    state = State()
    with state._conn() as c:
        rows = c.execute(
            "SELECT bar_close_iso, portfolio_return FROM mtm ORDER BY bar_close_iso"
        ).fetchall()
    if len(rows) < 5:
        return [
            "TRACKING ERROR vs BACKTEST".center(70), _h_rule(),
            f"  Need at least 5 MTM rows to compute (have {len(rows)}).", "",
        ]
    rets_paper = np.array([float(r["portfolio_return"]) for r in rows])
    # Compare to backtest at the same bars
    bt = _load_portfolio_returns()
    bt_aligned = []
    for r in rows:
        ts = pd.Timestamp(r["bar_close_iso"])
        if ts in bt.index:
            bt_aligned.append(float(bt.loc[ts]))
        else:
            bt_aligned.append(0.0)
    bt_arr = np.array(bt_aligned)
    diff = rets_paper - bt_arr
    te_ann = float(diff.std(ddof=0) * np.sqrt(252))
    return [
        "TRACKING ERROR vs BACKTEST".center(70),
        _h_rule(),
        f"  MTM rows observed:  {len(rows)}",
        f"  Mean drift:         {diff.mean()*10000:+.2f} bps/day",
        f"  Tracking error:     {te_ann*100:.2f}% annualized",
        f"  (target: < 0.50%/yr — anything higher indicates signal drift)",
        "",
    ]


def render() -> str:
    """Compose the full report as a single string."""
    blocks: list[list[str]] = [
        ["", "═" * 70,
         "  ALPHA-BETA PAPER TRADING REPORT".ljust(70),
         f"  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}".ljust(70),
         "═" * 70, ""],
        _portfolio_block(),
        [""],
        _open_trades_block(),
        _closed_trades_block(limit=20),
        _lifetime_stats_block(),
        _tracking_error_block(),
        ["═" * 70, ""],
    ]
    return "\n".join(line for block in blocks for line in block)


def main() -> int:
    print(render())
    return 0
