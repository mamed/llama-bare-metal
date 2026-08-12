#!/bin/bash
# restore-systemd.sh — restore the bare-metal llama.cpp systemd services from
# the backup directory. Run after a system update or fresh user session.

set -euo pipefail

BACKUP_DIR="$(cd "$(dirname "$0")" && pwd)/systemd-backup"
DEST_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

if [ ! -d "$BACKUP_DIR" ]; then
    echo "ERROR: backup dir not found at $BACKUP_DIR" >&2
    exit 1
fi

mkdir -p "$DEST_DIR"
cp -v "$BACKUP_DIR"/*.service "$DEST_DIR/"
systemctl --user daemon-reload

# Enable lingering so the services survive logout
loginctl enable-linger "$USER" 2>/dev/null || echo "(note: enable-linger needs sudo)"

echo
echo "=== enabled services ==="
systemctl --user is-enabled llama-backend.service llama-router.service
echo
echo "=== start them now ==="
systemctl --user start llama-backend.service llama-router.service
echo
echo "verify with:"
echo "  systemctl --user status llama-backend.service llama-router.service"
echo "  curl -s -X POST http://localhost:64010/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"unsloth-gemma-4-26b-a4b-it-ud-iq2-m\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":10}'"
