#!/bin/bash
# find_max_ctx.sh — for each VL model in models.yaml, find the highest context_size
# that fits in VRAM with full offload + mmproj. Updates the YAML in-place.
#
# Strategy: try the largest context first, fall back if OOM.
# Stops the backend, sets the ctx, starts, checks VRAM and journal for OOM.
#
# Usage: ./find_max_ctx.sh <yaml_path> <model_name> [target_ctx]
set -uo pipefail

YAML="$1"
MODEL="$2"
TARGET_CTX="${3:-262144}"

cd /home/fekry/llama-bare-metal || exit 1

# Find current ctx in the YAML for this model
get_ctx() {
    awk "/^- name: ${MODEL}\$/,/^- name: /" "$YAML" | grep -E '^  context_size:' | head -1 | awk '{print $2}'
}

# Update ctx in the YAML for this model
set_ctx() {
    local new_ctx="$1"
    python3 -c "
import sys
with open('$YAML') as f: text = f.read()
import re
# Find the model block (lines between '- name: MODEL' and next '- name:')
m = re.search(r'^- name: ${MODEL}\$.*?(?=^- name: |\Z)', text, re.DOTALL | re.MULTILINE)
if not m:
    print('MODEL NOT FOUND', file=sys.stderr); sys.exit(1)
block = m.group(0)
new_block = re.sub(r'context_size:\s*\d+', 'context_size: $new_ctx', block)
text = text[:m.start()] + new_block + text[m.end():]
with open('$YAML', 'w') as f: f.write(text)
"
}

echo "=== ${MODEL} (target ${TARGET_CTX}) ==="
CUR=$(get_ctx)
echo "  current ctx: $CUR"

# Binary search the ceiling
LOW=$CUR  # known-good
HIGH=$TARGET_CTX

while (( LOW < HIGH )); do
    MID=$(( (LOW + HIGH + 1) / 2 ))
    echo "  trying ctx=$MID ..."
    set_ctx "$MID"
    echo "MODEL_NAME=$MODEL" > .env
    systemctl --user restart llama-backend.service > /dev/null 2>&1
    HEALTHY=0
    for _i in $(seq 1 60); do
        if curl -s -f http://localhost:64000/health > /dev/null 2>&1; then HEALTHY=1; break; fi
        sleep 1
    done
    if [ "$HEALTHY" -eq 0 ]; then
        # timed out — probably OOM during load
        echo "    OOM (timeout)"
        HIGH=$((MID - 1))
        continue
    fi
    # Check if it actually loaded at the requested ctx
    N_CTX=$(journalctl --user -u llama-backend.service -n 30 --no-pager 2>&1 | grep -oE 'n_ctx_slot = [0-9]+' | tail -1 | awk '{print $3}')
    if [ -z "$N_CTX" ]; then
        echo "    no n_ctx_slot found in log"
        HIGH=$((MID - 1))
        continue
    fi
    if [ "$N_CTX" -lt "$MID" ]; then
        # llama.cpp auto-capped below our request
        echo "    capped at $N_CTX (below requested $MID)"
        HIGH=$((N_CTX - 1))
    else
        echo "    OK at $N_CTX"
        LOW=$MID
    fi
done

echo "  RESULT: max ctx = $LOW"
set_ctx "$LOW"
echo "MODEL_NAME=$MODEL" > .env
systemctl --user restart llama-backend.service > /dev/null 2>&1
sleep 3
echo "  final VRAM:"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits