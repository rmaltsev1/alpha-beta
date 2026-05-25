#!/usr/bin/env bash
# Send a one-off Telegram message via curl. Independent of Python so it works
# even if the venv is broken or imports are failing.
#
# Reads TELEGRAM_BOT and CHAT_ID from the repo's .env file.
#
# Usage:
#   ./scripts/notify_telegram.sh "your message here"
#   echo "your message" | ./scripts/notify_telegram.sh -
#
# Exits 0 on success, non-zero on failure. Failures are silent (this is the
# bottom of the alerting stack — nowhere to escalate to).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "notify_telegram: missing $ENV_FILE" >&2
    exit 2
fi

# Extract TELEGRAM_BOT and CHAT_ID without sourcing the whole .env
TELEGRAM_BOT=$(grep -E '^TELEGRAM_BOT=' "$ENV_FILE" | head -1 | cut -d= -f2- | tr -d '"' | tr -d "'")
CHAT_ID=$(grep -E '^CHAT_ID=' "$ENV_FILE" | head -1 | cut -d= -f2- | tr -d '"' | tr -d "'")

if [[ -z "$TELEGRAM_BOT" || -z "$CHAT_ID" ]]; then
    echo "notify_telegram: TELEGRAM_BOT or CHAT_ID not set in $ENV_FILE" >&2
    exit 2
fi

# Read message from arg or stdin
if [[ "${1:-}" == "-" || -z "${1:-}" ]]; then
    MSG=$(cat)
else
    MSG="$1"
fi

if [[ -z "$MSG" ]]; then
    echo "notify_telegram: empty message" >&2
    exit 2
fi

# Telegram limits at 4096 chars; truncate just in case
if [[ ${#MSG} -gt 4000 ]]; then
    MSG="${MSG:0:3900}\n\n…[truncated]"
fi

# Use curl with timeouts so we never block the calling script for long
HTTP_CODE=$(curl -s -o /tmp/notify_telegram_resp.json -w "%{http_code}" \
    --max-time 15 \
    --connect-timeout 5 \
    -X POST "https://api.telegram.org/bot${TELEGRAM_BOT}/sendMessage" \
    -d "chat_id=${CHAT_ID}" \
    --data-urlencode "text=${MSG}" 2>/dev/null) || HTTP_CODE=0

if [[ "$HTTP_CODE" == "200" ]]; then
    exit 0
else
    echo "notify_telegram: HTTP $HTTP_CODE" >&2
    [[ -f /tmp/notify_telegram_resp.json ]] && cat /tmp/notify_telegram_resp.json >&2
    exit 1
fi
