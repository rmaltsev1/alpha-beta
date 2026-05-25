"""Mirror-mode rebalance plan.

Given the latest exposure β vector and a target portfolio NAV, compute the
concrete rebalance instructions ("hold $X notional in instrument Y").

Used to produce an actionable message: 'to mirror this strategy at $100K NAV,
your book should be: UK100 +$25,000, JP225 +$19,300, GBP -$8,100, ...'.

Currencies: all betas are interpreted as fractions of NAV in USD-equivalent
notional. For OANDA non-USD-quoted instruments (UK100_GBP, DE30_EUR, JP225_USD)
we still report USD notional — the user's broker will handle the FX leg.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from alphabeta import get_candles


@dataclass(frozen=True)
class RebalanceLine:
    instrument: str
    beta: float
    target_notional_usd: float
    spot_price: Optional[float]
    target_units: Optional[float]    # target shares/contracts/coins
    side: str                         # 'LONG' | 'SHORT' | 'FLAT'


def _spot(instrument: str) -> Optional[float]:
    try:
        df = get_candles(instrument, "D1")
        return float(df["close"].iloc[-1]) if df is not None and len(df) else None
    except Exception:
        return None


def build_rebalance_plan(
    betas: dict[str, float],
    *,
    nav_usd: float,
    leverage: float = 1.0,
    min_abs_beta: float = 0.005,
) -> list[RebalanceLine]:
    """Convert a beta vector into concrete per-instrument allocations.

    `leverage` scales the gross target (e.g., 1.0 = mirror raw betas;
    raising it amplifies for paper-trading at higher gross).
    """
    out: list[RebalanceLine] = []
    for instr, beta in betas.items():
        if abs(beta) < min_abs_beta:
            continue
        target_usd = nav_usd * beta * leverage
        spot = _spot(instr)
        units = (target_usd / spot) if (spot and spot > 0) else None
        side = "LONG" if beta > 0 else "SHORT"
        out.append(RebalanceLine(
            instrument=instr,
            beta=beta,
            target_notional_usd=target_usd,
            spot_price=spot,
            target_units=units,
            side=side,
        ))
    out.sort(key=lambda r: -abs(r.target_notional_usd))
    return out


def format_rebalance_html(
    plan: list[RebalanceLine],
    *,
    nav_usd: float,
    bar_iso: str,
    leverage: float = 1.0,
) -> str:
    """Format the rebalance plan as a Telegram HTML message.

    Shows: instrument, side, target $ notional, target units (where computable).
    """
    bar_short = bar_iso[:10]
    lines = [
        f"<b>📐 Mirror-mode rebalance — {bar_short}</b>",
        f"<i>NAV: ${nav_usd:,.0f}  ·  Gross-target leverage: {leverage:.1f}×</i>",
        "",
        "<pre>",
        f"{'Instrument':<12} {'Side':<6} {'$ Target':>11}  Units",
    ]
    gross = 0.0
    net = 0.0
    for r in plan:
        gross += abs(r.target_notional_usd)
        net += r.target_notional_usd
        units_str = f"{r.target_units:+,.4f}" if r.target_units is not None else "n/a"
        sign = "+" if r.target_notional_usd >= 0 else ""
        lines.append(
            f"{r.instrument:<12} {r.side:<6} "
            f"{sign}${r.target_notional_usd:>9,.0f}  {units_str}"
        )
    lines.append("</pre>")
    lines.append(
        f"<i>Gross: ${gross:,.0f} ({gross/nav_usd:.0%} of NAV)  ·  "
        f"Net: {'+' if net >= 0 else ''}${net:,.0f} ({net/nav_usd:+.0%})</i>"
    )
    lines.append("<i>This is the target book. Diff vs your current book = trades to place.</i>")
    return "\n".join(lines)
