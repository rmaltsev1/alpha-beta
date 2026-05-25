#!/usr/bin/env bash
# Fire the alpha-beta live signal pipeline once for the current D1 bar.
# Designed to be invoked by cron or launchd at ~21:05 UTC daily.
#
# Usage:
#   ./scripts/run_live_daily.sh              # fire for real
#   ./scripts/run_live_daily.sh --dry-run    # preview only
#
# Cron example (every day at 21:05 UTC):
#   5 21 * * * /Users/you/path/to/alpha-beta/scripts/run_live_daily.sh >> ~/Library/Logs/alphabeta_live.log 2>&1

set -euo pipefail

# Resolve repo root regardless of cwd
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

# Activate the venv
if [[ ! -d ".venv" ]]; then
    echo "ERROR: .venv not found in $REPO_ROOT" >&2
    echo "Run: python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt" >&2
    exit 2
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# Ensure logs dir exists
LOG_DIR="${HOME}/Library/Logs"
mkdir -p "$LOG_DIR"

TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================"
echo "[$TS] alphabeta live fire starting"
echo "Repo: $REPO_ROOT"
echo "Python: $(which python)"

# Step 1: incremental fetch of fresh candles from Binance + OANDA APIs.
# Uses --source api so no SSH tunnel is required. Incremental (only new bars
# since last fetch), so typically ~30s-2min depending on how stale we are.
# If fetch fails (network), continue anyway with whatever's on disk.
echo "[$(date -u +%H:%M:%S)] fetching latest candles via API..."
if ! python -m alphabeta fetch --source api; then
    echo "[$(date -u +%H:%M:%S)] WARNING: fetch failed, proceeding with stale data" >&2
fi

# Step 2: run the live signal pipeline. The --once command refreshes the
# master_v16 integration script (which reads pre-computed sleeve returns).
# NOTE: individual sleeve scripts are NOT re-run by --once. The sleeve panel
# `all_sleeve_returns_v15.parquet` is frozen until you manually rebuild it.
# For day-over-day signal freshness, run rebuild_sleeves.sh weekly.
echo "[$(date -u +%H:%M:%S)] running live --once..."
python -m alphabeta live --once "$@"
EXIT_CODE=$?

TS_END="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[$TS_END] alphabeta live fire done (exit $EXIT_CODE)"
echo "================================"
exit $EXIT_CODE
