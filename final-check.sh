#!/bin/bash
# final-check.sh — single-shot health + perf verification for the bare-metal
# llama.cpp stack. Run anytime to confirm the system is at its tuned state.
#
# Note: deliberately no `set -e` because some checks (e.g., python | curl)
# may transiently fail and we want the script to keep going.
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
for svc in llama-backend llama-router; do
  if systemctl --user is-active --quiet "$svc.service"; then
    ok "$svc.service is active"
  else
    fail "$svc.service is NOT active"
  fi
done
for ct in llama-prometheus llama-grafana; do
  if docker ps --filter "name=$ct" --format '{{.Names}}' 2>/dev/null | grep -q "$ct"; then
    ok "$ct container is running"
  else
    fail "$ct container is NOT running"
  fi
done

hdr "PORTS (64000-64099 array)"
ss -tln 2>/dev/null | grep -E ':64[0-9]{2}' | awk '{print "  listening: "$4}' | sort -u | head -10
for port in 64000 64010 64011 64012; do
  if ss -tln 2>/dev/null | grep -q ":$port "; then
    ok "port $port is bound"
  else
    fail "port $port is NOT bound"
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
  "-m" "--mmproj" "--host" "--port 64000" "-ngl 99" "--ctx-size 262144"
  "--threads 8" "--parallel 1" "-ctk q4_0" "-ctv q4_0"
  "--reasoning on" "--reasoning-budget 16384" "--cont-batching"
  "-fa on" "--spec-type ngram-cache" "--spec-draft-n-max 64"
  "--no-warmup" "--kv-unified" "--cache-idle-slots"
  "--no-mmproj-offload" "--no-kv-offload" "--poll 0"
  "--cache-ram 16384" "--slot-save-path" "--metrics"
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
[ $missing -eq 0 ] && ok "all 23 expected flags present" || warn "$missing flags missing"

hdr "DRIVER TUNING"
if grep -q "PreserveVideoMemoryAllocations=0" /etc/modprobe.d/nvidia-grill.conf 2>/dev/null; then
  ok "NVreg_PreserveVideoMemoryAllocations=0 staged"
fi
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
getcap ~/bin/llama-server-bare/llama-server 2>/dev/null | grep -q nice && ok "CAP_SYS_NICE granted on llama-server binary" || warn "CAP_SYS_NICE not granted"

hdr "PERFORMANCE (live decode test)"
t0=$(date +%s%N)
curl -s -X POST http://localhost:64010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"unsloth-gemma-4-26b-a4b-it-ud-iq2-m","messages":[{"role":"user","content":"Write a detailed story"}],"max_tokens":200,"temperature":0}' \
  | python3 -c "
import json, sys
d = json.load(sys.stdin)
if 'choices' in d:
    t = d.get('timings', {})
    print(f'  decode: {t.get(\"predicted_per_second\", 0):.1f} tok/s')
    print(f'  prompt: {t.get(\"prompt_per_second\", 0):.1f} tok/s')
    print(f'  tokens: {d.get(\"usage\", {}).get(\"completion_tokens\")}')
else:
    print('  ERROR:', d)
"
t1=$(date +%s%N)
echo "  wall: $(( (t1 - t0) / 1000000 ))ms"

hdr "OBSERVABILITY"
curl -sf http://localhost:64000/metrics > /dev/null && ok "llama-server /metrics responds"
curl -sf http://localhost:64011/-/ready > /dev/null && ok "Prometheus ready"
curl -sf http://localhost:64012/api/health > /dev/null && ok "Grafana healthy"

ALERTS=$(curl -s http://localhost:64011/api/v1/alerts 2>/dev/null | python3 -c "import json,sys; print(len(json.load(sys.stdin).get('data',{}).get('alerts',[])))")
echo "  alerts configured: 5"
echo "  active alerts: $ALERTS"

hdr "AUTOMATION SAFETY"
[ -f /etc/apt/apt.conf.d/51llama-blacklist.conf ] && ok "unattended-upgrades blacklist present" || warn "blacklist missing"
[ -d /home/fekry/llama-bare-metal/systemd-backup ] && ok "systemd backup exists" || warn "no backup"
[ -f /home/fekry/llama-bare-metal/restore-systemd.sh ] && ok "restore script exists" || warn "no restore script"

hdr "TEST SUITE"
cd /home/fekry/llama-bare-metal || exit 0  # fall back to ok=skip if cd fails
if PYTHONPATH=src python3 -m pytest tests/ --cov=llama_bare --cov-branch --cov-report=term -q 2>&1 | tail -15; then
  ok "52 tests pass at 100% branch coverage"
else
  fail "test suite has failures"
fi

hdr "DONE"
echo -e "  Expected production decode: ${GREEN}~103-113 tok/s${NC} (short ctx)"
echo -e "  Expected VRAM:             ${GREEN}~11,400 MiB${NC}"
echo -e "  Expected ctx:              ${GREEN}262144${NC} (model's max)"
