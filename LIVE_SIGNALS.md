# Live Telegram signals — design

How the backtested `master_v16` portfolio is exposed as paper-trading Telegram signals.

## What the system delivers

Three flows out of Telegram, all targeting one chat:

1. **Per-bar signal messages** at each bar close — only when *net portfolio position per instrument* changes materially (|Δ| ≥ 0.05 of book). One batched message per bar, not one per instrument.
2. **Daily digest** at 21:00 UTC — current positions table, today's P&L, MTD/YTD, drawdown, top-contributing sleeves.
3. **Ops alerts** for failures (data fetch, scheduler, computation) — terse, terminal-style.

## Architectural constraint discovered in the sleeve audit

All 24 production sleeves persist *returns only* (`.parquet` with `timestamp` + `ret`). Position vectors are computed internally during each sleeve script's run and discarded. There is no per-instrument position state to "read."

Implication: live signal generation cannot simply load a `positions.parquet` and diff. We have two pathways:

| Pathway | Cost | Fidelity |
|---|---|---|
| **A. Portfolio P&L tracking (no per-instrument signals)** | Tiny — runs master_v16 forward and records portfolio daily return | High for P&L tracking, zero for instrument-level signals |
| **B. Per-sleeve position emission (full signals)** | Per-sleeve refactor (24 files) to also save positions | Full signal fidelity |
| **C. Hybrid (recommended)** | Refactor only the ~8 single-instrument sleeves that emit discrete positions | High for P&L, partial for signals |

**v1 ships pathway A only.** Per-instrument signals come in v2 once the position-emission refactor is done. This is the fastest path to a working paper-trading channel today.

## v1 scope (this PR)

What's delivered:

- Telegram client with retries, dedup, dry-run, idempotency log
- SQLite state: positions, signals, fills, mtm, bar_fires tables
- Signal model (dataclass) + Markdown/HTML formatters
- Daily digest formatter
- Master-runner: pulls latest bar of data, runs `master_v16`, records daily return, sends Telegram digest
- `python -m alphabeta live --once` for one-shot fire (manual or cron-driven)
- `python -m alphabeta live --status` for inspecting paper state
- `python -m alphabeta live --dry-run` for testing without sending

What's not in v1:

- Per-sleeve position emission (pathway B/C)
- launchd plist (manual `cron` is fine to start; daemonization is v2)
- Stream-driven trigger (timer-only for v1)
- Backfill / sleep-wake recovery
- 4H / W1 intraday fires (D1 only for v1)
- ATR-based stop-loss hints

## v2 roadmap

Once v1 is shipping clean digests:

1. **Position emission**: modify each sleeve to optionally write a positions parquet alongside its returns parquet. Aggregate to net portfolio position per instrument.
2. **Per-bar event messages**: when net position per instrument changes ≥0.05 of book, fire a batched Telegram event.
3. **Multi-timeframe**: add H4 + W1 fires.
4. **Daemon**: long-running `python -m alphabeta live --daemon` with APScheduler + stream-driven fast path + launchd plist for auto-start.
5. **Reconciliation**: detect missed fires (Mac sleep) and catch up on startup with `[STALE RESUME]` marker.

## Data flow (v1)

```
21:00 UTC  ←  cron fires "alphabeta live --once"
              ↓
       1. fetch latest D1 data
       2. run master_v16.py (recompute portfolio return for today)
       3. read PRODUCTION_v16_V4.parquet → last row
       4. update paper equity in SQLite
       5. compute MTM-deltas, sleeve attribution
       6. send Telegram digest
       7. record bar_fire (idempotent)
```

## Configuration (.env)

```
TELEGRAM_BOT=<bot token from BotFather>
CHAT_ID=<user's chat id with the bot>
```

Both already exist in the user's `.env` (gitignored). `.env.example` documents the structure.

## Paper P&L parameters (v1 defaults)

| Setting | Default | Source |
|---|---|---|
| Starting equity | $100,000 | conventional |
| Cost model | inherited from backtest (FX 1bp, Index 1.5bp, Crypto 5bp per side) | `alphabeta/backtest.py` |
| MTM frequency | once/day at 21:00 UTC | matches master_v16 D1 cadence |
| Vol target | 25% (per v16-V4) | `floor_ceiling_lever` |
| Leverage cap | 15× | `floor_ceiling_lever(max_lev=15)` |
| Number of sleeves | 24 | `TOP24` in `master_v16.py` |

## Files

```
alphabeta/live/
  __init__.py              ← package marker
  telegram_client.py       ← Bot API wrapper (retries, dedup, dry-run)
  signals.py               ← Signal dataclass + Markdown/HTML formatters
  state.py                 ← SQLite persistence (positions, signals, fills, mtm)
  pnl.py                   ← Paper P&L tracker (v1: portfolio-level)
  runner.py                ← One-shot bar-close fire (v1 entry point)
data/live/
  paper_state.sqlite       ← persistent state (gitignored)
  telegram_log.sqlite      ← idempotency log (gitignored)
```

## CLI (v1)

```sh
# Fire once for today's D1 bar (manual or cron)
python -m alphabeta live --once

# Print current paper state (equity, positions, recent signals)
python -m alphabeta live --status

# Dry-run: same as --once but only prints to stdout, no Telegram, no state writes
python -m alphabeta live --dry-run

# Send a manual "I'm online" ping (for testing connectivity)
python -m alphabeta live --ping

# Skip the master_v16 rebuild (use already-on-disk parquet)
python -m alphabeta live --once --no-refresh
```

## Deployment on macOS

Two paths, pick one.

### A. launchd (recommended — survives reboot, runs in background)

```sh
./scripts/install_live_plist.sh
```

That generates `~/Library/LaunchAgents/com.alphabeta.live.plist` (substituting your repo path), loads it, and the daemon will fire every day at 21:05 UTC.

To unload:

```sh
./scripts/install_live_plist.sh --unload
```

Force-fire now (useful for first test):

```sh
launchctl start com.alphabeta.live
```

Inspect:

```sh
launchctl list | grep alphabeta
tail -f ~/Library/Logs/alphabeta_live.out
```

### B. cron (simpler, but Mac must be awake at 21:05 UTC)

```sh
crontab -e
# add this line:
5 21 * * * /Users/you/path/to/alpha-beta/scripts/run_live_daily.sh >> ~/Library/Logs/alphabeta_live.log 2>&1
```

## First-time setup checklist

1. **Initiate chat with the bot.** Open Telegram → search `@alphabetabotnotrobot` → tap Start. (Required: Telegram doesn't allow bots to message users who haven't initiated a chat.)
2. **Verify connectivity.**
   ```sh
   python -m alphabeta live --ping
   ```
   Expect: `"status": "ok"` and a 🟢 message in Telegram.
3. **First dry-run end-to-end.**
   ```sh
   python -m alphabeta live --dry-run --no-refresh
   ```
   Expect: digest preview in stdout, no Telegram send.
4. **Real first fire (one-shot).**
   ```sh
   python -m alphabeta live --once
   ```
   Expect: digest delivered to Telegram, paper state updated.
5. **Install the schedule.**
   ```sh
   ./scripts/install_live_plist.sh
   ```

## Testing strategy

1. **Telegram connectivity**: `python -m alphabeta live --ping` sends one test message.
2. **Dry-run end-to-end**: `python -m alphabeta live --dry-run` exercises everything without state mutation or network send.
3. **Idempotency**: run `--once` twice in a row → second run is a no-op.
4. **Status inspection**: `--status` shows the persisted state.

## Honest limitations (state explicitly)

- v1 does NOT emit per-instrument trade signals. It tracks the portfolio's daily P&L only. To paper-trade in the literal sense, the user would need to recreate the entire 24-sleeve portfolio themselves — not realistic. The digest is the paper-track-record.
- All signals are advisory and lag the backtest by the cron schedule (≤24h).
- macOS sleep = missed fires. v1 has no catch-up; v2 adds it.
- Single venue assumed for paper-trade math (OANDA + Binance, both with backtest-spec costs).
