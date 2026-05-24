"""CLI: `python -m alphabeta <command>`.

Commands:
  list                       Show local parquet inventory.
  fetch                      Pull data from prod DB (default) or upstream APIs.
                             Default is incremental. Pass --full to re-download.
  status                     Show local vs. prod row counts side-by-side
                             (requires the SSH tunnel to be open).
  stream                     Live-stream new bars into the parquet store
                             (binance WS for crypto, oanda for fx/indices).

Examples:
  python -m alphabeta fetch                        # incremental refresh, all symbols
  python -m alphabeta fetch --full                 # nuke and re-download everything
  python -m alphabeta fetch --symbol BTCUSDT       # just one symbol
  python -m alphabeta fetch --timeframe H1 H4 D1   # subset of timeframes
  python -m alphabeta fetch --source api           # bypass DB, hit Binance/OANDA directly
  python -m alphabeta list                         # what's on disk?
  python -m alphabeta stream                       # live crypto + fx + indices, runs forever
  python -m alphabeta stream --symbol BTCUSDT      # one crypto only
  python -m alphabeta stream --no-oanda            # crypto only
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime

from . import fetch_api, fetch_db, storage
from .symbols import ALL_SYMBOLS, TIMEFRAMES


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_list(_args: argparse.Namespace) -> int:
    df = storage.list_local()
    if df.empty:
        print("no local data yet — run `python -m alphabeta fetch`")
        return 0
    with __import__("pandas").option_context("display.max_rows", None, "display.width", 200):
        print(df.to_string(index=False))
    print(f"\ntotal: {df['rows'].sum():,} rows, {df['mb'].sum():.1f} MB on disk")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    syms = args.symbol or ALL_SYMBOLS
    tfs = args.timeframe or TIMEFRAMES
    unknown = [s for s in syms if s not in ALL_SYMBOLS]
    if unknown:
        print(f"unknown symbol(s): {unknown}\nvalid: {ALL_SYMBOLS}", file=sys.stderr)
        return 2

    backend = fetch_api if args.source == "api" else fetch_db
    print(f"fetching: source={args.source} full={args.full} "
          f"symbols={len(syms)} timeframes={len(tfs)} -> {storage.settings.data_dir}")
    t0 = datetime.now()
    results = backend.fetch_all(symbols=syms, timeframes=tfs, full=args.full)
    elapsed = (datetime.now() - t0).total_seconds()

    ok = sum(1 for r in results if r["fetched"] >= 0)
    failed = sum(1 for r in results if r["fetched"] < 0)
    total = sum(r["fetched"] for r in results if r["fetched"] > 0)
    print(f"\ndone in {elapsed:.1f}s  ok={ok}  failed={failed}  fetched_rows={total:,}")
    return 0 if failed == 0 else 1


def cmd_stream(args: argparse.Namespace) -> int:
    from . import stream as stream_mod  # imports websockets lazily
    syms = args.symbol or ALL_SYMBOLS
    tfs = args.timeframe or ["M1", "M5", "M15", "H1", "H4", "D1"]
    unknown = [s for s in syms if s not in ALL_SYMBOLS]
    if unknown:
        print(f"unknown symbol(s): {unknown}\nvalid: {ALL_SYMBOLS}", file=sys.stderr)
        return 2
    try:
        asyncio.run(stream_mod.run(
            symbols=syms, timeframes=tfs,
            binance=not args.no_binance, oanda=not args.no_oanda,
        ))
    except KeyboardInterrupt:
        print("\nstream stopped (Ctrl-C)")
    return 0


def cmd_status(_args: argparse.Namespace) -> int:
    """Compare local row counts to prod row counts (needs the tunnel)."""
    local = storage.list_local().set_index(["symbol", "timeframe"])
    with fetch_db.connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT a.symbol, c.timeframe::text AS tf, count(*) AS rows, "
            "max(c.timestamp) AS last "
            "FROM candles c JOIN assets a ON a.id = c.asset_id "
            "GROUP BY 1, 2 ORDER BY 1, 2"
        )
        prod_rows = cur.fetchall()
    print(f"{'symbol':<12} {'tf':<4} {'local_rows':>10} {'prod_rows':>10} {'delta':>10}  {'last':<25}")
    for r in prod_rows:
        sym, tf, prod, last = r["symbol"], r["tf"], r["rows"], r["last"]
        loc = int(local.loc[(sym, tf), "rows"]) if (sym, tf) in local.index else 0
        delta = prod - loc
        flag = "" if delta == 0 else (" !" if delta > 0 else " (extra local)")
        print(f"{sym:<12} {tf:<4} {loc:>10,} {prod:>10,} {delta:>+10,}  {last}{flag}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="alphabeta", description="local price store")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="show local parquet inventory")
    sub.add_parser("status", help="compare local vs prod row counts (requires tunnel)")

    pf = sub.add_parser("fetch", help="pull / refresh candles")
    pf.add_argument("--source", choices=["db", "api"], default="db",
                    help="db = prod postgres via tunnel (default); api = Binance/OANDA direct")
    pf.add_argument("--full", action="store_true",
                    help="re-download from 2020-01-01 (default: incremental from latest local ts)")
    pf.add_argument("--symbol", nargs="+", help=f"subset of {ALL_SYMBOLS}")
    pf.add_argument("--timeframe", nargs="+", help=f"subset of {TIMEFRAMES}")

    ps = sub.add_parser("stream", help="live stream new bars into the local store")
    ps.add_argument("--symbol", nargs="+", help=f"subset of {ALL_SYMBOLS}")
    ps.add_argument("--timeframe", nargs="+", help="binance TFs to subscribe to (default: M1..D1)")
    ps.add_argument("--no-binance", action="store_true", help="skip crypto WebSocket")
    ps.add_argument("--no-oanda", action="store_true", help="skip OANDA REST stream")

    args = p.parse_args(argv)
    _setup_logging(args.verbose)

    if args.cmd == "list":
        return cmd_list(args)
    if args.cmd == "fetch":
        return cmd_fetch(args)
    if args.cmd == "status":
        return cmd_status(args)
    if args.cmd == "stream":
        return cmd_stream(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
