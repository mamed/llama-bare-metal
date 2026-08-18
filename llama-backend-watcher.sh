#!/bin/bash
# llama-backend-watcher.sh — health probe + auto-restart for the backend.
#
# Runs as a sibling systemd unit (llama-backend-watcher.service). Every
# PING_INTERVAL seconds it probes the backend /health endpoint. When
# /health fails for FAIL_THRESHOLD consecutive checks, it explicitly
# triggers `systemctl --user restart llama-backend.service` and waits
# for the backend to come back.
#
# Why explicit restart instead of a systemd watchdog? Because the
# backend's own process is alive and well (a CUDA hang is invisible to
# systemd's cgroup-based health probe). The only way to detect a hang
# is from outside via /health, and the only way to recover is to
# restart. The router's L-3 backoff + E2 circuit breaker throttle
# repeated restart attempts if the backend is fundamentally broken.
#
# Environment:
#   HEALTH_URL         — full /health URL (default: http://127.0.0.1:64000/health)
#   PING_INTERVAL      — seconds between probes (default: 10)
#   HEALTH_TIMEOUT     — curl timeout per probe (default: 3)
#   FAIL_THRESHOLD     — consecutive failures before restart (default: 3)
#   BACKEND_SERVICE    — systemd unit to restart (default: llama-backend.service)
#   STARTUP_GRACE_SEC  — seconds after THIS script starts during which
#                        failed checks are IGNORED. Defends against the
#                        backend startup race: backend is Type=simple and
#                        the model takes ~10s to load, so /health returns
#                        connection-refused for the first ~10s after the
#                        backend unit starts. Without the grace window,
#                        the watcher would trip on every boot/restart and
#                        restart the backend mid-load. (audit 2026-08-18)

set -u

HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:64000/health}"
PING_INTERVAL="${PING_INTERVAL:-10}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-3}"
FAIL_THRESHOLD="${FAIL_THRESHOLD:-3}"
BACKEND_SERVICE="${BACKEND_SERVICE:-llama-backend.service}"
STARTUP_GRACE_SEC="${STARTUP_GRACE_SEC:-60}"  # 60s covers model load + slack

# Trap SIGTERM so systemd's stop is clean (exit 0) instead of leaving the
# `sleep` interrupted. Without this, the watcher exits non-zero on stop
# and Restart=on-failure spins it back up — log noise only, but easy to fix.
trap 'log "received SIGTERM, exiting cleanly"; exit 0' TERM INT

log() {
    printf '%s backend-watcher: %s\n' "$(date -Iseconds)" "$*" >&2
}

log "starting (HEALTH_URL=$HEALTH_URL PING_INTERVAL=${PING_INTERVAL}s FAIL_THRESHOLD=$FAIL_THRESHOLD STARTUP_GRACE=${STARTUP_GRACE_SEC}s)"

# Don't start counting failures until the startup grace window has elapsed.
# Without this, the watcher would fire FAIL_THRESHOLD restarts on every
# boot because the backend is Type=simple and takes ~10s to bind /health.
start_time=$(date +%s)

fail_count=0

while true; do
    if curl -fsS --max-time "$HEALTH_TIMEOUT" -o /dev/null "$HEALTH_URL" 2>/dev/null; then
        # Healthy — reset fail counter.
        if [[ "$fail_count" -gt 0 ]]; then
            log "backend /health recovered (after $fail_count failed checks)"
        fi
        fail_count=0
    else
        elapsed=$(( $(date +%s) - start_time ))
        if [[ "$elapsed" -lt "$STARTUP_GRACE_SEC" ]]; then
            # In startup grace — log only once per threshold crossing so the
            # journal isn't spammed, but don't increment fail_count.
            if [[ "$fail_count" -eq 0 ]]; then
                log "backend /health not ready yet (in startup grace, ${elapsed}s/${STARTUP_GRACE_SEC}s elapsed)"
            fi
        else
            fail_count=$((fail_count + 1))
            log "backend /health failed ($fail_count/$FAIL_THRESHOLD)"
        fi
        if [[ "$fail_count" -ge "$FAIL_THRESHOLD" ]]; then
            log "threshold reached — restarting $BACKEND_SERVICE"
            if systemctl --user restart "$BACKEND_SERVICE" 2>&1; then
                log "restart succeeded; waiting for /health to recover"
                # After restart, give the backend 30s to come back. During
                # this window, don't trigger another restart.
                sleep 30
                # Reset the counter so we don't immediately re-trigger if
                # /health is still down immediately after restart.
                fail_count=0
            else
                log "restart FAILED — backing off"
                # Sleep longer than usual to avoid hot loop.
                sleep $((PING_INTERVAL * 5))
            fi
        fi
    fi

    sleep "$PING_INTERVAL"
done
