#!/bin/bash
# tune_max_ctx.py — for each entry in models.yaml, find the highest context_size
# that fits in VRAM with the current server-level optimizations applied.
# This assumes --no-kv-offload is enabled (KV on RAM, so ctx is only limited
# by compute-buffer VRAM, not KV cache).
#
# Strategy: try the largest requested ctx, halve if OOM. Stop on success.
# Updates each entry's context_size in-place.

import re
import subprocess
import sys
import time
from pathlib import Path

YAML = Path("/home/fekry/llama-cpp-docker/llama-unified/models.yaml")
BACKEND_PORT = 64000
HEALTH_URL = f"http://localhost:{BACKEND_PORT}/health"
LOG = "/tmp/tune_max_ctx.log"

# Load the YAML once
text = YAML.read_text()

# Find every model entry. Each starts with `- name: ...` and ends at next `- name:` or EOF.
entries = []
for m in re.finditer(r'(- name: [^\n]+\n(?:  [^\n]+\n)*)', text):
    block = m.group(1)
    name_m = re.match(r'- name: (\S+)', block)
    if not name_m:
        continue
    name = name_m.group(1)
    ctx_m = re.search(r'context_size:\s*(\d+)', block)
    if not ctx_m:
        continue
    current_ctx = int(ctx_m.group(1))
    entries.append((name, current_ctx))

print(f"Found {len(entries)} entries with context_size")
print(f"Log: {LOG}")
with open(LOG, "w") as f:
    pass

results = []
for name, current_ctx in entries:
    # Binary search the max ctx that loads
    low = current_ctx
    high = 1048576  # 1M = absolute max for llama.cpp

    # First test: does the model load at all at current_ctx?
    # (skip if already failing)
    print(f"\n=== {name} (current ctx={current_ctx:,}) ===")

    # Set MODEL_NAME, restart, check
    def try_ctx(ctx):
        # Update YAML
        new_text = re.sub(
            rf'(- name: {re.escape(name)}\n(?:  [^\n]+\n)*?  context_size:)\s*\d+',
            rf'\g<1> {ctx}',
            text,
            count=1,
        )
        YAML.write_text(new_text)
        # Restart
        subprocess.run(
            ["systemctl", "--user", "restart", "llama-backend.service"],
            capture_output=True,
        )
        # Wait for health or timeout
        for _ in range(120):  # 2 minutes
            try:
                r = subprocess.run(
                    ["curl", "-s", "-f", HEALTH_URL],
                    capture_output=True,
                    timeout=2,
                )
                if r.returncode == 0:
                    return True
            except subprocess.TimeoutExpired:
                pass
            time.sleep(1)
        return False

    # Quick sanity check at current ctx
    print(f"  sanity check at current ctx...")
    if not try_ctx(current_ctx):
        print(f"  WARN: model fails to load even at current_ctx; skipping")
        results.append((name, current_ctx, "FAIL"))
        continue

    # Now binary search upward
    last_good = current_ctx
    while low < high:
        mid = (low + high + 1) // 2
        print(f"  trying ctx={mid:,}...", end=" ", flush=True)
        if try_ctx(mid):
            print("OK")
            low = mid
            last_good = mid
        else:
            print("OOM")
            high = mid - 1
            # restore the last good ctx for the next try
            try_ctx(last_good)

    print(f"  >>> max ctx = {last_good:,}")
    results.append((name, last_good, "OK"))

# Final report
print("\n\n=== SUMMARY ===")
for name, ctx, status in results:
    print(f"  {name:55}  ctx={ctx:>10,}  {status}")
