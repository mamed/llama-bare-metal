#!/bin/bash
# llama-backend.sh — systemd-friendly launcher wrapper.
#
# - Reads MODEL_NAME from .env
# - Sources it into the launcher's environment
# - Writes $STATE_FILE before exec, so the router can read it
# - Removes $STATE_FILE on exit (any signal)
set -euo pipefail

# D2: Structured logging — emit to stderr with timestamps so the systemd
# journal captures them in a parseable format.
log_info()  { printf '%s INFO: %s\n'  "$(date -Iseconds)" "$*" >&2; }
log_warn()  { printf '%s WARN: %s\n'  "$(date -Iseconds)" "$*" >&2; }
log_error() { printf '%s ERROR: %s\n' "$(date -Iseconds)" "$*" >&2; }

ENV_FILE="${ENV_FILE:-/home/fekry/Projects/llama-bare-metal/.env}"
STATE_FILE="${STATE_FILE:-${XDG_RUNTIME_DIR:-/run/user/$UID}/llama-backend.model}"

if [[ ! -f "$ENV_FILE" ]]; then
    log_error "FATAL: env file not found: $ENV_FILE"
    exit 1
fi

# Source the env file (just MODEL_NAME=...)
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

if [[ -z "${MODEL_NAME:-}" ]]; then
    log_error "FATAL: MODEL_NAME not set in $ENV_FILE"
    exit 2
fi

# Publish current model name BEFORE exec so the router can read it
# immediately on first request after backend start
echo "$MODEL_NAME" > "$STATE_FILE"

# Clean up state file on any exit (SIGTERM from systemctl, etc.)
cleanup() {
    rm -f "$STATE_FILE"
}
trap cleanup EXIT INT TERM

# Hand off to the actual launcher
exec /home/fekry/Projects/llama-bare-metal/launcher.sh