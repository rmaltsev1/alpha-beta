#!/usr/bin/env bash
# Open an SSH tunnel to prod postgres on the rektfree VM so the data
# fetcher can read the candles table without touching the production app.
#
# Usage:
#   ./scripts/tunnel-prod.sh            # foreground, Ctrl+C closes
#   ./scripts/tunnel-prod.sh -b         # background; kill with: pkill -f 'ssh.*-N.*deploy@'
set -euo pipefail

# Load .env if present so VM_HOST / SSH_KEY / LOCAL_PG_PORT can be overridden.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
if [[ -f "$ROOT/.env" ]]; then
  # shellcheck disable=SC1091
  set -a; . "$ROOT/.env"; set +a
fi

VM_HOST="${VM_HOST:-62.238.28.3}"
VM_USER="${VM_USER:-deploy}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
SSH_KEY="${SSH_KEY/#\~/$HOME}"
LOCAL_PG_PORT="${LOCAL_PG_PORT:-15432}"

ARGS=(-N -L "${LOCAL_PG_PORT}:localhost:5432" -i "$SSH_KEY" -o ServerAliveInterval=30 "${VM_USER}@${VM_HOST}")

if [[ "${1:-}" == "-b" ]]; then
  ssh -fN "${ARGS[@]}"
  echo "tunnel running in background. kill with: pkill -f 'ssh.*-N.*${VM_USER}@${VM_HOST}'"
else
  echo "tunneling localhost:${LOCAL_PG_PORT} -> ${VM_HOST}:5432 (postgres)"
  echo "Ctrl+C to close."
  exec ssh "${ARGS[@]}"
fi
