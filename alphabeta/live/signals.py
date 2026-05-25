"""Signal data model + Telegram message formatting.

A "signal" represents a *change* in target position for a single instrument,
emitted at a known bar-close timestamp by a named sleeve.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from alphabeta.live.telegram_client import escape_markdown_v2 as e


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


def position_to_direction(p: float, eps: float = 1e-6) -> Direction:
    if p > eps:
        return Direction.LONG
    if p < -eps:
        return Direction.SHORT
    return Direction.FLAT


@dataclass(frozen=True)
class Signal:
    """One position-change event for one instrument."""

    bar_close: datetime  # UTC, tz-aware
    sleeve: str  # e.g. "TREND_NEW", "PORTFOLIO" (the net aggregate)
    instrument: str  # e.g. "BTCUSDT", "EUR_USD"
    timeframe: str  # "D1", "H4", "H1", "W1"

    prev_position: float  # last known position [-1, +1] (or larger after leverage)
    new_position: float
    direction: Direction  # derived from new_position

    # Optional context fields
    ref_price: Optional[float] = None  # the bar-close price we recorded the change at
    notional_usd: Optional[float] = None  # how much $ this position represents
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    note: Optional[str] = None

    def is_open(self) -> bool:
        return abs(self.prev_position) < 1e-9 and abs(self.new_position) > 1e-9

    def is_close(self) -> bool:
        return abs(self.prev_position) > 1e-9 and abs(self.new_position) < 1e-9

    def is_flip(self) -> bool:
        return self.prev_position * self.new_position < -1e-12

    def is_resize(self) -> bool:
        return (
            not self.is_open()
            and not self.is_close()
            and abs(self.new_position - self.prev_position) > 1e-9
        )

    def event_label(self) -> str:
        if self.is_open():
            return f"OPEN {self.direction.value}"
        if self.is_close():
            return "CLOSE"
        if self.is_flip():
            return f"FLIP → {self.direction.value}"
        if self.is_resize():
            diff = self.new_position - self.prev_position
            return f"{'ADD' if abs(self.new_position) > abs(self.prev_position) else 'TRIM'} {self.direction.value}"
        return "HOLD"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bar_close"] = self.bar_close.isoformat()
        d["direction"] = self.direction.value
        return d


def format_signal_markdown(sig: Signal) -> str:
    """Format a single signal as a Telegram MarkdownV2 message.

    Example:
        ⚡ *OPEN LONG* BTCUSDT (D1)
        Sleeve: TREND_NEW
        Position: 0.00 → +0.42
        Ref price: 67,420.50
        Bar close: 2026-05-25 21:00 UTC
    """
    arrow_emoji = {
        Direction.LONG: "🟢",
        Direction.SHORT: "🔴",
        Direction.FLAT: "⚪",
    }[sig.direction]

    lines = [
        f"{arrow_emoji} *{e(sig.event_label())}* `{e(sig.instrument)}` \\({e(sig.timeframe)}\\)",
        f"Sleeve: `{e(sig.sleeve)}`",
        f"Position: `{sig.prev_position:+.3f}` → `{sig.new_position:+.3f}`",
    ]
    if sig.ref_price is not None:
        lines.append(f"Ref price: `{e(f'{sig.ref_price:,.2f}')}`")
    if sig.notional_usd is not None:
        lines.append(f"Notional: `${e(f'{sig.notional_usd:,.0f}')}`")
    if sig.stop_loss is not None:
        lines.append(f"SL: `{e(f'{sig.stop_loss:,.2f}')}`")
    if sig.take_profit is not None:
        lines.append(f"TP: `{e(f'{sig.take_profit:,.2f}')}`")
    lines.append(f"Bar close: `{e(sig.bar_close.strftime('%Y-%m-%d %H:%M UTC'))}`")
    if sig.note:
        lines.append(f"_{e(sig.note)}_")
    return "\n".join(lines)


def format_batch_digest(
    signals: list[Signal],
    *,
    portfolio_equity: Optional[float] = None,
    day_pnl_pct: Optional[float] = None,
    open_positions: Optional[dict[str, float]] = None,
) -> str:
    """Format multiple signals in a single batched message (one Telegram send).

    Used when several sleeves fire on the same bar-close — keeps notification noise down.
    """
    if not signals:
        return ""
    bar = signals[0].bar_close
    tf = signals[0].timeframe
    header = [
        f"📊 *alpha\\-beta signals* \\({e(tf)} bar\\)",
        f"Bar close: `{e(bar.strftime('%Y-%m-%d %H:%M UTC'))}`",
        "",
    ]
    if portfolio_equity is not None:
        header.append(f"Equity: `${e(f'{portfolio_equity:,.0f}')}`")
    if day_pnl_pct is not None:
        sign = "+" if day_pnl_pct >= 0 else ""
        header.append(f"Day P&L: `{e(sign)}{day_pnl_pct:.2%}`")

    body = ["", "*Position changes:*"]
    for s in signals:
        emoji = {"LONG": "🟢", "SHORT": "🔴", "FLAT": "⚪"}[s.direction.value]
        body.append(
            f"{emoji} `{e(s.instrument):<10}` `{e(s.sleeve):<14}` "
            f"`{s.prev_position:+.2f}`→`{s.new_position:+.2f}` "
            f"\\[{e(s.event_label())}\\]"
        )

    if open_positions:
        body.append("")
        body.append("*Current portfolio positions:*")
        for instr, pos in sorted(open_positions.items()):
            if abs(pos) < 1e-3:
                continue
            emoji = "🟢" if pos > 0 else "🔴"
            body.append(f"{emoji} `{e(instr):<10}` `{pos:+.3f}`")

    return "\n".join(header + body)


def _h(s: str) -> str:
    """Escape HTML special chars for Telegram HTML parse mode."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def format_daily_digest_html(
    *,
    bar_iso: str,
    equity_after: float,
    equity_before: float,
    portfolio_return: float,
    daily_pnl_usd: float,
    drawdown_from_peak: float,
    peak_equity: float,
    mtd_return: float,
    ytd_return: float,
    since_start_return: float,
    rolling_sharpe_30d: Optional[float],
    sleeve_top: list[tuple[str, float]],
    sleeve_bottom: list[tuple[str, float]],
    starting_equity: float,
    next_bar_hint: Optional[str] = None,
) -> str:
    """Format the daily portfolio digest as Telegram HTML.

    HTML is preferred over MarkdownV2: fewer reserved chars, more robust against
    numeric content with dots/dashes, supports <pre> for monospaced tables.
    """
    bar_dt = datetime.fromisoformat(bar_iso).strftime("%Y-%m-%d %H:%M UTC")
    sign_day = "+" if daily_pnl_usd >= 0 else ""
    sign_ret = "+" if portfolio_return >= 0 else ""
    sign_since = "+" if since_start_return >= 0 else ""
    sign_mtd = "+" if mtd_return >= 0 else ""
    sign_ytd = "+" if ytd_return >= 0 else ""

    lines = [
        f"<b>📊 alpha-beta paper book — {_h(bar_dt)}</b>",
        f"<i>Starting equity: ${_h(f'{starting_equity:,.0f}')}</i>",
        "",
        "<pre>",
        f"Equity:        ${equity_after:>11,.2f}  ({sign_since}{since_start_return:.2%})",
        f"Day P&amp;L:       {sign_day}${daily_pnl_usd:>10,.2f}  ({sign_ret}{portfolio_return:.2%})",
        f"MTD / YTD:     {sign_mtd}{mtd_return:.2%}  /  {sign_ytd}{ytd_return:.2%}",
        f"Drawdown:      {drawdown_from_peak:>+.2%}  (peak ${peak_equity:,.0f})",
    ]
    if rolling_sharpe_30d is not None:
        lines.append(f"30d Sharpe:    {rolling_sharpe_30d:>+.2f}")
    lines.append("</pre>")

    if sleeve_top:
        lines.append("")
        lines.append("<b>Top contributors today (bps):</b>")
        lines.append("<pre>")
        for name, val in sleeve_top:
            bps = val * 10000
            lines.append(f"  {name:<18}  {bps:>+7.1f} bps")
        lines.append("</pre>")

    if sleeve_bottom:
        lines.append("<b>Top drags today (bps):</b>")
        lines.append("<pre>")
        for name, val in sleeve_bottom:
            bps = val * 10000
            lines.append(f"  {name:<18}  {bps:>+7.1f} bps")
        lines.append("</pre>")

    lines.append("")
    if next_bar_hint:
        lines.append(f"<i>Next bar close: {_h(next_bar_hint)}</i>")
    lines.append("<i>Paper-trade only. Not investment advice.</i>")
    lines.append("#alphabeta #digest #daily")

    return "\n".join(lines)


def format_boot_message_html(bot_username: str, repo_url: Optional[str] = None) -> str:
    lines = [
        f"<b>🟢 alpha-beta signal pipeline online</b>",
        "",
        f"Bot: <code>@{_h(bot_username)}</code>",
        f"Mode: paper-trade (no real orders)",
        f"Cadence: D1 close 21:00 UTC daily digest",
        "",
        "<i>Will start emitting from next D1 close.</i>",
    ]
    if repo_url:
        lines.append(f"Repo: {_h(repo_url)}")
    return "\n".join(lines)
