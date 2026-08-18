#!/bin/bash
# final-check.sh — single-shot health + perf verification for the bare-metal
# llama.cpp stack. Run anytime to confirm the system is at its tuned state.
#
# Note: deliberately no `set -e` because some checks (e.g., python | curl)
# may transiently fail and we want the script to keep going.

# Find the repo root. Defaults to /home/fekry/Projects/llama-bare-metal but
# can be overridden via REPO_ROOT env var (useful for CI).
REPO_ROOT="${REPO_ROOT:-/home/fekry/Projects/llama-bare-metal}"

YELLOW='\033[1;33m'
GREEN='\033[1;32m'
RED='\033[1;31m'
CYAN='\033[1;36m'
NC='\033[0m'

ok()  { echo -e "${GREEN}✓${NC} $*"; }
warn(){ echo -e "${YELLOW}⚠${NC} $*"; }
fail(){ echo -e "${RED}✗${NC} $*"; }
hdr(){ echo -e "\n${CYAN}━━━ $* ━━━${NC}"; }

hdr "SERVICE HEALTH"
for svc in llama-backend llama-router llama-backend-watcher; do
  if systemctl --user is-active --quiet "$svc.service"; then
    ok "$svc.service is active"
  else
    fail "$svc.service is NOT active"
  fi
done

hdr "PORTS (127.0.0.1 only – LAN exposure check for llama-bare-metal)"
# Only check the ports we own. Other services (Grafana, Prometheus, n8n,
# open-webui, etc.) bind to 0.0.0.0 deliberately as part of the monitoring
# stack.
local_ports="64000 64010"
LISTEN=$(ss -tln 2>/dev/null | grep -E ':64[0-9]{2}' | awk '{print $4}' | sort -u)
for entry in $LISTEN; do
    addr=$(echo "$entry" | cut -d: -f1)
    port=$(echo "$entry" | cut -d: -f2)
    # Strip [] from IPv6 display
    addr=${addr#[}; addr=${addr%]}
    if [[ " $local_ports " == *" $port "* ]]; then
        if [[ "$addr" == "127.0.0.1" || "$addr" == "::1" ]]; then
            ok "port $port bound to $addr (loopback only)"
        else
            fail "port $port bound to $addr — LAN EXPOSED (our service)"
        fi
    fi
done

hdr "GPU + VRAM"
nvidia-smi --query-gpu=memory.used,memory.free,memory.reserved,driver_version --format=csv,noheader,nounits 2>/dev/null | awk -F',' '{printf "  used=%s MiB  free=%s MiB  reserved=%s MiB  driver=%s\n", $1, $2, $3, $4}'

hdr "CURRENT MODEL"
if [ -f /run/user/1000/llama-backend.model ]; then
  echo "  $(cat /run/user/1000/llama-backend.model)"
else
  warn "model state file not present"
fi
ctx=$(curl -s http://localhost:64000/props 2>/dev/null | grep -oE '"n_ctx":[0-9]+' | head -1 | awk -F: '{print $2}')
if [ -z "$ctx" ]; then
  ctx=$(journalctl --user -u llama-backend.service -n 1000 --no-pager 2>&1 | grep -oE 'n_ctx_slot = [0-9]+' | tail -1 | awk '{print $3}')
fi
echo "  context_size: ${ctx:-unknown}"

hdr "FLAG CHECK (against locked-in production config)"
PID=$(pgrep -x llama-server | head -1)
if [ -z "$PID" ]; then
  fail "llama-server is not running"
  exit 1
fi
CMD=$(ps -o cmd= -p $PID)
EXPECTED_FLAGS=(
  "-m" "--host" "--port 64000" "-ngl 99" "--ctx-size 262144"
  "--threads 16" "--parallel 1" "-ctk q4_0" "-ctv q4_0"
  "--reasoning on" "--reasoning-budget 32768" "--cont-batching"
  "-fa on" "--spec-type ngram-cache"
  "--no-warmup"
  "--no-mmproj-offload" "--no-kv-offload" "--poll 0"
  "--cache-ram 16384" "--slot-save-path" "--metrics"
)
# Optional flags (only present if not disabled via DISABLE_* env vars)
OPTIONAL_FLAGS=(
  "--kv-unified"        # disabled by DISABLE_KV_UNIFIED=true
  "--cache-idle-slots"  # disabled by DISABLE_KV_UNIFIED=true
)
missing=0
for flag in "${EXPECTED_FLAGS[@]}"; do
  if echo "$CMD" | grep -qF -- "$flag"; then
    ok "$flag"
  else
    fail "$flag (NOT FOUND)"
    missing=$((missing+1))
  fi
done
for flag in "${OPTIONAL_FLAGS[@]}"; do
  if echo "$CMD" | grep -qF -- "$flag"; then
    ok "$flag (optional)"
  else
    warn "$flag (optional, disabled by env)"
  fi
done
[ $missing -eq 0 ] && ok "all required flags present" || warn "$missing required flags missing"

hdr "DRIVER TUNING"
PCIE=$(nvidia-smi --query-gpu=pcie.link.gen.current --format=csv,noheader,nounits 2>/dev/null)
if [ "$PCIE" = "5" ]; then
  ok "PCIe Gen5 active"
elif [ "$PCIE" = "1" ]; then
  warn "PCIe Gen$PCIE (downshifted; check ASPM policy)"
else
  ok "PCIe Gen$PCIE"
fi
CLKS=$(nvidia-smi --query-gpu=clocks.current.graphics,clocks.current.memory --format=csv,noheader,nounits 2>/dev/null)
echo "  clocks: $CLKS"
PERSIST=$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader,nounits 2>/dev/null)
[ "$PERSIST" = "Enabled" ] && ok "persistence_mode enabled" || warn "persistence_mode: $PERSIST"

hdr "CPU PINNING"
PID=$(pgrep -x llama-server | head -1)
AFFINITY=$(taskset -cp $PID 2>/dev/null | awk -F': ' '{print $2}')
echo "  llama-server CPU affinity: $AFFINITY (expected: 0-7 for 3D V-Cache CCD)"

hdr "CAPABILITY CHECK"
getcap $REPO_ROOT/bin/llama-server-bare/llama-server 2>/dev/null | grep -q nice && ok "CAP_SYS_NICE granted on llama-server binary" || warn "CAP_SYS_NICE not granted"

hdr "PERFORMANCE (live decode test)"
t0=$(date +%s%N)
curl -s -X POST http://localhost:64010/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"unsloth-qwen3.8-27b-ud-q2_k_xl","messages":[{"role":"user","content":"Write a detailed story"}],"max_tokens":200,"temperature":0}' > /tmp/final-check-resp.json
python3 - <<'PYEOF'
import json
with open("/tmp/final-check-resp.json") as f:
    d = json.load(f)
if "choices" in d:
    t = d.get("timings", {})
    print(f'  decode: {t.get("predicted_per_second", 0):.1f} tok/s')
    print(f'  prompt: {t.get("prompt_per_second", 0):.1f} tok/s')
    print(f'  tokens: {d.get("usage", {}).get("completion_tokens")}')
else:
    print("  ERROR:", d)
PYEOF
t1=$(date +%s%N)
echo "  wall: $(( (t1 - t0) / 1000000 ))ms"

hdr "JOURNAL HYGIENE"
SIZE=$(journalctl --disk-usage 2>&1 | grep -oE '[0-9.]+[KMGT]' | head -1)
echo "  journal size: ${SIZE:-unknown}"
COUNT=$(journalctl --user -u llama-router.service --no-pager 2>&1 | grep -cE 'API token \(auto-generated\): [A-Za-z0-9_-]{40,}')
if [ "$COUNT" -gt 0 ]; then
    fail "$COUNT full API tokens leaked in the journal (run journalctl --vacuum-time=1s + SIGUSR2 to clean)"
else
    ok "no full API tokens in the journal"
fi

hdr "OBSERVABILITY"
curl -sf http://localhost:64000/metrics > /dev/null && ok "llama-server /metrics responds" || warn "llama-server /metrics not reachable"
curl -sf http://localhost:64011/-/ready > /dev/null && ok "Prometheus ready" || warn "prometheus not reachable"
curl -sf http://localhost:64012/api/health > /dev/null && ok "Grafana healthy" || warn "grafana not reachable"

hdr "AUTOMATION SAFETY"
[ -f /etc/apt/apt.conf.d/51llama-blacklist.conf ] && ok "unattended-upgrades blacklist present" || warn "blacklist missing"
[ -d "$REPO_ROOT" ] && ok "repo root present ($REPO_ROOT)" || fail "repo root missing"
[ -f "$REPO_ROOT/restore-systemd.sh" ] && ok "restore script exists" || warn "no restore script"
[ -f "$REPO_ROOT/systemd/llama-backend.service" ] && ok "systemd/llama-backend.service tracked" || fail "systemd unit missing"
[ -f "$REPO_ROOT/systemd/llama-router.service" ] && ok "systemd/llama-router.service tracked" || fail "systemd unit missing"
[ -f "$REPO_ROOT/systemd/llama-backend-watcher.service" ] && ok "systemd/llama-backend-watcher.service tracked" || fail "watcher unit missing"

hdr "TEST SUITE"
cd "$REPO_ROOT" || { warn "REPO_ROOT missing; skipping tests"; exit 0; }
if PYTHONPATH=src python3 -m pytest tests/ -q 2>&1 | tail -15; then
  ok "tests pass"
else
  fail "test suite has failures"
fi

hdr "DONE"
echo "  Expected VRAM:  ~11,400 MiB"
echo "  Expected ctx:   262144 (model's max)"
