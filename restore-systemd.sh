#!/bin/bash
# restore-systemd.sh — restore the bare-metal llama.cpp systemd services from
# the source tree. Run after a fresh git clone or a system update.
#
# Installs:
#   - llama-backend.service        (the model server)
#   - llama-router.service         (the OpenAI-compatible multiplexer)
#   - llama-backend-watcher.service (the health-probe + auto-restart)
# Plus the watcher script (llama-backend-watcher.sh) is referenced from
# its absolute path in the unit, so no separate copy is needed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$SCRIPT_DIR/systemd"
DEST_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

if [ ! -d "$SRC_DIR" ]; then
    echo "ERROR: source dir not found at $SRC_DIR" >&2
    exit 1
fi

mkdir -p "$DEST_DIR"
# Copy all *.service files from the source (backend, router, watcher).
cp -v "$SRC_DIR"/*.service "$DEST_DIR/"
systemctl --user daemon-reload

# Enable lingering so the services survive logout
loginctl enable-linger "$USER" 2>/dev/null || echo "(note: enable-linger needs sudo)"

echo
echo "=== enabled services ==="
systemctl --user is-enabled llama-backend.service llama-router.service llama-backend-watcher.service
echo
echo "=== start them now ==="
systemctl --user start llama-backend.service llama-backend-watcher.service llama-router.service
echo
echo "verify with:"
echo "  systemctl --user status llama-backend.service llama-router.service llama-backend-watcher.service"
echo "  curl -s http://127.0.0.1:64000/health"
echo "  curl -s http://127.0.0.1:64010/health"
