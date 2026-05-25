"""One-shot bar-close runner.

Triggered by cron (or manually) — it:
  1. Optionally re-runs master_v16.py to refresh the production parquet
  2. Computes the MTM update for the latest bar (idempotent)
  3. Persists state
  4. Sends a Telegram daily digest
  5. Records the bar_fire as ok / skipped / error

Designed to be safe to invoke repeatedly: second call on the same bar is a no-op.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd

from alphabeta.config import settings
from alphabeta.live.attribution import compute_attribution, format_attribution_lines
from alphabeta.live.exposure import (
    WINDOW_DAYS,
    compute_exposure,
    detect_signal_events,
    format_exposure_html,
    format_signal_events_html,
)
from alphabeta.live.paper_trades import (
    process_exposure_shift,
    format_trades_block_html,
)
from alphabeta.live.pnl import (
    latest_mtm,
    persist_mtm,
    rolling_sharpe,
    STARTING_EQUITY,
    PRODUCTION_PARQUET,
)
from alphabeta.live.rebalance import build_rebalance_plan, format_rebalance_html
from alphabeta.live.signals import (
    format_boot_message_html,
    format_daily_digest_html,
)
from alphabeta.live.state import State
from alphabeta.live.telegram_client import TelegramClient

PAPER_TRADES_DB = Path(__file__).resolve().parents[2].parent / "alpha-beta" / "data" / "live" / "paper_trades.sqlite"
# Resolve via settings instead
from alphabeta.config import settings
PAPER_TRADES_DB = settings.data_dir / "live" / "paper_trades.sqlite"


REPO_ROOT = Path(__file__).resolve().parents[2]
MASTER_V16 = REPO_ROOT / "scratch" / "quant" / "master_v16.py"


def refresh_master_v16(verbose: bool = True) -> None:
    """Re-run master_v16.py to refresh PRODUCTION_v16_V4.parquet.

    No-op if the file is fresh (within last 6 hours). Use --force to bypass.
    """
    cmd = [sys.executable, str(MASTER_V16)]
    env = {**__import__("os").environ, "PYTHONPATH": str(REPO_ROOT)}
    if verbose:
        print(f"[runner] refreshing master_v16: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"master_v16 failed (exit {result.returncode}):\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    if verbose:
        print(result.stdout[-1500:] if len(result.stdout) > 1500 else result.stdout)


def fire_once(*, dry_run: bool = False, refresh: bool = True, verbose: bool = True) -> dict:
    """Fire one D1 bar-close cycle. Returns a result dict suitable for logging."""
    if refresh:
        try:
            refresh_master_v16(verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"[runner] master_v16 refresh failed: {e}", file=sys.stderr)
            # Continue with whatever production parquet we have

    state = State()
    update = latest_mtm(state)
    if update is None:
        return {"status": "skipped", "reason": "no new bar or already fired"}

    if dry_run:
        print(f"[DRY-RUN] would update equity to ${update.equity_after:,.2f}")
        print(f"[DRY-RUN] daily return: {update.portfolio_return:+.4%}")
        print(f"[DRY-RUN] top: {update.sleeve_attribution_top}")
        print(f"[DRY-RUN] bottom: {update.sleeve_attribution_bottom}")

    # Sharpe needs a few days of history first
    sharpe_30d = rolling_sharpe(state, window_days=30) if not dry_run else None

    digest_html = format_daily_digest_html(
        bar_iso=update.bar_close_iso,
        equity_after=update.equity_after,
        equity_before=update.equity_before,
        portfolio_return=update.portfolio_return,
        daily_pnl_usd=update.daily_pnl_usd,
        drawdown_from_peak=update.drawdown_from_peak,
        peak_equity=update.peak_equity,
        mtd_return=update.mtd_return,
        ytd_return=update.ytd_return,
        since_start_return=update.since_start_return,
        rolling_sharpe_30d=sharpe_30d,
        sleeve_top=update.sleeve_attribution_top,
        sleeve_bottom=update.sleeve_attribution_bottom,
        starting_equity=STARTING_EQUITY,
        next_bar_hint="D1 close + 1 day @ 21:00 UTC",
    )

    if verbose:
        print("[runner] digest preview:")
        print(digest_html)

    # v2: compute per-instrument exposure via rolling regression
    prev_betas = state.exposure_before(update.bar_close_iso)
    exposure_snap = None
    signal_events = []
    try:
        exposure_snap = compute_exposure(
            window_days=WINDOW_DAYS, last_known=prev_betas
        )
        signal_events = detect_signal_events(exposure_snap)
    except Exception as e:
        if verbose:
            print(f"[runner] exposure computation failed: {e}", file=sys.stderr)

    tc = TelegramClient(dry_run=dry_run)
    try:
        # 1) main daily digest (P&L)
        msg_id = tc.send(
            digest_html,
            parse_mode="HTML",
            idempotency_key=f"digest:{update.bar_close_iso}",
        )
        # 2) exposure snapshot (per-instrument net beta)
        if exposure_snap is not None:
            exposure_html = format_exposure_html(exposure_snap)
            tc.send(
                exposure_html,
                parse_mode="HTML",
                idempotency_key=f"exposure:{exposure_snap.bar_iso}",
            )
            if verbose:
                print("[runner] exposure preview:")
                print(exposure_html)
        # 3) material exposure-change events + sleeve attribution + paper-trade lifecycle
        trade_events = []
        if signal_events:
            # Build enhanced events message with attribution + trade-event tags
            bar_ts = pd.Timestamp(exposure_snap.bar_iso)
            event_lines = [
                f"<b>🚦 Material exposure shifts — {exposure_snap.bar_iso[:10]}</b>",
                f"<i>|Δbeta| ≥ 0.05 (R² {exposure_snap.r_squared:.2f})</i>",
                "",
            ]
            for sym, b, d in signal_events:
                prev = b - d
                # Discrete trade lifecycle
                trade_evt = process_exposure_shift(
                    PAPER_TRADES_DB,
                    instrument=sym,
                    prev_beta=prev,
                    new_beta=b,
                    bar_iso=exposure_snap.bar_iso,
                )
                trade_events.append(trade_evt)
                # Tag inside the message
                if trade_evt.kind == "OPEN":
                    action = f"OPEN {trade_evt.side}"
                elif trade_evt.kind == "CLOSE":
                    rp = (trade_evt.return_pct or 0) * 100
                    action = f"CLOSE  ({rp:+.2f}%, held {trade_evt.hold_days}d)"
                elif trade_evt.kind == "FLIP":
                    rp = (trade_evt.return_pct or 0) * 100
                    action = f"FLIP→{trade_evt.side}  (prior leg {rp:+.2f}%, held {trade_evt.hold_days}d)"
                else:
                    action = "resize"
                event_lines.append(
                    f"<code>{sym:<10}</code>  "
                    f"β {prev:+.2f} → {b:+.2f}   [<b>{action}</b>]"
                )
                # Attribution: top sleeves driving this instrument
                try:
                    contribs = compute_attribution(sym, bar_ts=bar_ts, n_top=3)
                    for line in format_attribution_lines(sym, contribs, indent="    "):
                        event_lines.append(line)
                except Exception as e:
                    if verbose:
                        print(f"[runner] attribution failed for {sym}: {e}", file=sys.stderr)
                event_lines.append("")  # blank line between instruments

            event_lines.append(
                "<i>Estimates from rolling regression — not exact positions. "
                "Open/Close use trade thresholds |β|≥0.05 / &lt;0.02.</i>"
            )
            events_html = "\n".join(event_lines)
            tc.send(
                events_html,
                parse_mode="HTML",
                idempotency_key=f"events:{exposure_snap.bar_iso}",
            )
            if verbose:
                print(f"[runner] {len(signal_events)} signal event(s) emitted")

        # 4) mirror-mode rebalance plan — sized to STARTING_EQUITY (fixed NAV),
        # NOT the compounded backfilled equity, so the rebalance dollar amounts
        # are stable and meaningful for a real paper-trader sizing to $100K.
        if exposure_snap is not None:
            plan = build_rebalance_plan(
                exposure_snap.betas,
                nav_usd=STARTING_EQUITY,
                leverage=1.0,
            )
            rebalance_html = format_rebalance_html(
                plan,
                nav_usd=STARTING_EQUITY,
                bar_iso=exposure_snap.bar_iso,
            )
            tc.send(
                rebalance_html,
                parse_mode="HTML",
                idempotency_key=f"rebalance:{exposure_snap.bar_iso}",
            )

        # 5) paper-trades summary block
        if exposure_snap is not None:
            trades_html = format_trades_block_html(PAPER_TRADES_DB, exposure_snap.bar_iso)
            tc.send(
                trades_html,
                parse_mode="HTML",
                idempotency_key=f"trades:{exposure_snap.bar_iso}",
            )
    except Exception as e:
        state.record_fire("D1", update.bar_close_iso, status="error", error_msg=str(e))
        return {
            "status": "error",
            "stage": "telegram_send",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

    if not dry_run:
        persist_mtm(state, update)
        if exposure_snap is not None:
            state.record_exposure(
                exposure_snap.bar_iso,
                exposure_snap.window_days,
                exposure_snap.r_squared,
                exposure_snap.betas,
            )
        state.record_fire("D1", update.bar_close_iso, status="ok",
                          signals_emitted=1 + (1 if exposure_snap else 0) + len(signal_events))

    return {
        "status": "ok",
        "dry_run": dry_run,
        "bar_close_iso": update.bar_close_iso,
        "equity_after": update.equity_after,
        "portfolio_return": update.portfolio_return,
        "telegram_message_id": msg_id,
        "exposure_r2": exposure_snap.r_squared if exposure_snap else None,
        "signal_events_count": len(signal_events),
    }


def ping(*, dry_run: bool = False) -> dict:
    """Send a connectivity-test Telegram message."""
    tc = TelegramClient(dry_run=dry_run)
    me = tc.get_me() if not dry_run else {"result": {"username": "(dry-run)"}}
    bot_username = me.get("result", {}).get("username", "unknown")
    text = format_boot_message_html(bot_username, repo_url="github.com/rmaltsev1/alpha-beta")
    msg_id = tc.send(text, parse_mode="HTML", idempotency_key=f"boot:{__import__('time').strftime('%Y-%m-%d-%H')}")
    return {"status": "ok", "telegram_message_id": msg_id, "bot": bot_username}


def status() -> dict:
    """Return a dict summarizing current paper state."""
    state = State()
    equity = state.latest_equity()
    positions = state.get_portfolio_positions()
    recent = state.recent_signals(limit=10)
    sharpe_30d = rolling_sharpe(state, window_days=30)
    return {
        "equity": equity if equity is not None else STARTING_EQUITY,
        "starting_equity": STARTING_EQUITY,
        "since_start_return": (equity / STARTING_EQUITY - 1.0) if equity else 0.0,
        "open_positions": positions,
        "rolling_sharpe_30d": sharpe_30d,
        "recent_signals_count": len(recent),
        "production_parquet_exists": PRODUCTION_PARQUET.exists(),
    }


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="alphabeta live")
    ap.add_argument("--once", action="store_true", help="Fire one D1 bar-close cycle and exit.")
    ap.add_argument("--ping", action="store_true", help="Send a connectivity-test Telegram message.")
    ap.add_argument("--status", action="store_true", help="Print current paper state as JSON.")
    ap.add_argument("--report", action="store_true", help="Print full paper-trading dashboard (portfolio + trades + stats).")
    ap.add_argument("--dry-run", action="store_true", help="Print would-be actions; do not send Telegram or mutate state.")
    ap.add_argument("--no-refresh", action="store_true", help="Skip the master_v16 refresh step (use existing parquet).")
    ap.add_argument("--quiet", action="store_true", help="Minimal stdout output.")
    args = ap.parse_args(argv)

    if args.report:
        from alphabeta.live import report as report_mod
        return report_mod.main()

    if args.status:
        print(json.dumps(status(), indent=2, default=str))
        return 0

    if args.ping:
        try:
            r = ping(dry_run=args.dry_run)
            print(json.dumps(r, indent=2))
            return 0
        except Exception as e:
            print(f"ping failed: {e}", file=sys.stderr)
            return 2

    if args.once or args.dry_run:
        r = fire_once(
            dry_run=args.dry_run,
            refresh=not args.no_refresh,
            verbose=not args.quiet,
        )
        print(json.dumps(r, indent=2, default=str))
        return 0 if r.get("status") in {"ok", "skipped"} else 1

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
