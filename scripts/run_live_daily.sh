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

# Step 1: full pipeline rebuild — fetches candles, re-runs all 50 sleeve scripts,
# then runs the master chain v3 → v16. ~5-7 min total. This is what makes the
# signals truly reflect today's data instead of replaying frozen historicals.
# Use --skip-rebuild as an arg to bypass (e.g. for fast testing).
SKIP_REBUILD=0
NEW_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --skip-rebuild) SKIP_REBUILD=1 ;;
        *) NEW_ARGS+=("$arg") ;;
    esac
done

if [[ $SKIP_REBUILD -eq 0 ]]; then
    echo "[$(date -u +%H:%M:%S)] running full pipeline rebuild (~5-7 min)..."
    if ! "$SCRIPT_DIR/rebuild_all.sh"; then
        echo "[$(date -u +%H:%M:%S)] WARNING: rebuild had failures, continuing" >&2
    fi
else
    echo "[$(date -u +%H:%M:%S)] --skip-rebuild: using existing parquets"
fi

# Step 2: run the signal pipeline. --no-refresh because rebuild_all already
# ran master_v16; --once means just process the latest bar and exit.
echo "[$(date -u +%H:%M:%S)] running live --once --no-refresh..."
python -m alphabeta live --once --no-refresh "${NEW_ARGS[@]}"
EXIT_CODE=$?

TS_END="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[$TS_END] alphabeta live fire done (exit $EXIT_CODE)"
echo "================================"
exit $EXIT_CODE
