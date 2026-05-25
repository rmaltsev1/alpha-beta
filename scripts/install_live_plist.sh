#!/usr/bin/env bash
# Install the alpha-beta live signal launchd plist on macOS.
#
# This generates ~/Library/LaunchAgents/com.alphabeta.live.plist from the
# template and loads it. The job fires daily at 21:05 UTC.
#
# Usage:
#   ./scripts/install_live_plist.sh           # install + load
#   ./scripts/install_live_plist.sh --unload  # unload + delete

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEMPLATE="$SCRIPT_DIR/com.alphabeta.live.plist.template"
INSTALL_PATH="$HOME/Library/LaunchAgents/com.alphabeta.live.plist"

if [[ "${1:-}" == "--unload" ]]; then
    if [[ -f "$INSTALL_PATH" ]]; then
        launchctl unload "$INSTALL_PATH" 2>/dev/null || true
        rm -f "$INSTALL_PATH"
        echo "unloaded and removed: $INSTALL_PATH"
    else
        echo "no plist at $INSTALL_PATH"
    fi
    exit 0
fi

if [[ ! -f "$TEMPLATE" ]]; then
    echo "ERROR: template not found: $TEMPLATE" >&2
    exit 2
fi

mkdir -p "$HOME/Library/LaunchAgents"
mkdir -p "$HOME/Library/Logs"

# Substitute placeholders
sed \
    -e "s|ABS_REPO_PATH|$REPO_ROOT|g" \
    -e "s|ABS_HOME|$HOME|g" \
    "$TEMPLATE" > "$INSTALL_PATH"

# Reload if already loaded
launchctl unload "$INSTALL_PATH" 2>/dev/null || true
launchctl load "$INSTALL_PATH"

echo "installed: $INSTALL_PATH"
echo
echo "fires daily at 21:05 UTC"
echo "logs: ~/Library/Logs/alphabeta_live.{out,err}"
echo
echo "to force-fire now:    launchctl start com.alphabeta.live"
echo "to remove:            ./scripts/install_live_plist.sh --unload"
