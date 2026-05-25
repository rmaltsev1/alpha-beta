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

set -uo pipefail   # NOT -e — we handle errors explicitly so we can alert

# Resolve repo root regardless of cwd
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HOSTNAME_SHORT=$(hostname -s 2>/dev/null || hostname)

cd "$REPO_ROOT"

# --- failure alert ---
# Any unhandled error → Telegram message + non-zero exit. Captures stage so
# the alert tells you whether it was rebuild that failed or signal emission.
LAST_STAGE="boot"
on_failure() {
    local exit_code=$?
    local err_line="${BASH_LINENO[0]:-?}"
    local msg="🚨 alphabeta live FAILED on ${HOSTNAME_SHORT}
Stage: ${LAST_STAGE}
Exit code: ${exit_code}
Line: ${err_line}
Repo: ${REPO_ROOT}
Time: $(date -u +%Y-%m-%dT%H:%M:%SZ)

Logs: ~/Library/Logs/alphabeta_live.{out,err}"
    "$SCRIPT_DIR/notify_telegram.sh" "$msg" 2>/dev/null || true
    exit "$exit_code"
}
trap on_failure ERR

# Activate the venv
if [[ ! -d ".venv" ]]; then
    "$SCRIPT_DIR/notify_telegram.sh" "🚨 alphabeta: .venv missing at $REPO_ROOT" 2>/dev/null || true
    echo "ERROR: .venv not found in $REPO_ROOT" >&2
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

LAST_STAGE="rebuild"
if [[ $SKIP_REBUILD -eq 0 ]]; then
    echo "[$(date -u +%H:%M:%S)] running full pipeline rebuild (~5-7 min)..."
    if ! "$SCRIPT_DIR/rebuild_all.sh"; then
        # Soft warning — continue with stale data rather than skip the day
        echo "[$(date -u +%H:%M:%S)] WARNING: rebuild had failures, continuing with existing parquets" >&2
        "$SCRIPT_DIR/notify_telegram.sh" "⚠️ alphabeta: rebuild had failures on ${HOSTNAME_SHORT} — continuing with existing parquets. Check ~/Library/Logs/alphabeta_live.err" 2>/dev/null || true
    fi
else
    echo "[$(date -u +%H:%M:%S)] --skip-rebuild: using existing parquets"
fi

# Step 2: run the signal pipeline. --no-refresh because rebuild_all already
# ran master_v16; --once means just process the latest bar and exit.
LAST_STAGE="live_signal"
echo "[$(date -u +%H:%M:%S)] running live --once --no-refresh..."
python -m alphabeta live --once --no-refresh "${NEW_ARGS[@]}"
EXIT_CODE=$?

LAST_STAGE="done"

TS_END="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[$TS_END] alphabeta live fire done (exit $EXIT_CODE)"
echo "================================"
exit $EXIT_CODE
