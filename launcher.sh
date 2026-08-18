#!/bin/bash
# llama-launcher — generic llama.cpp runtime launcher
#
# Reads models.yaml, finds the entry whose `name:` matches $MODEL_NAME, builds
# the corresponding llama-server arg list, and execs it. All arg-list logic
# lives in llama_bare.launcher_config.build_args (covered by tests at 100%
# branch coverage). This script is just the systemd-friendly glue:
# sourcing profile.d, sourcing .env, calling the Python builder, exec'ing.
#
# Required env:
#   MODEL_NAME    — the YAML entry's `name:` field
# Optional env (override YAML values):
#   PORT          — port to bind (default: 64000)
#   HOST          — bind address (default: 127.0.0.1)
# Anything else is taken from the YAML.

set -eo pipefail

# D2: Structured logging. Three levels, all emit to stderr with timestamps so
# the systemd journal captures them. Use these instead of bare `echo`/`>&2`.
log_info()  { printf '%s INFO: %s\n'  "$(date -Iseconds)" "$*" >&2; }
log_warn()  { printf '%s WARN: %s\n'  "$(date -Iseconds)" "$*" >&2; }
log_error() { printf '%s ERROR: %s\n' "$(date -Iseconds)" "$*" >&2; }

# Source the perf env (CUDA or Vulkan, whichever baked it)
# Use a subshell with -u disabled because some profile.d scripts (e.g.
# cedilla-portuguese.sh on Ubuntu 26.04) reference unbound vars.
for _envd in /etc/profile.d/*.sh; do
    # shellcheck disable=SC1090
    (set +u; source "$_envd")
done
unset _envd

CONFIG="${CONFIG_FILE:-/home/fekry/Projects/llama-cpp-unified/models.yaml}"
LLAMA_BARE_SRC="${LLAMA_BARE_SRC:-/home/fekry/Projects/llama-bare-metal/src}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-/home/fekry/bin/llama-server-bare/llama-server}"

if [[ ! -f "$CONFIG" ]]; then
    log_error "FATAL: config not found at $CONFIG"
    log_error "Set CONFIG_FILE to point at a valid models.yaml"
    exit 1
fi

if [[ -z "${MODEL_NAME:-}" ]]; then
    log_error "FATAL: MODEL_NAME env var is required (set it in .env)"
    exit 1
fi

# E5: Pre-flight binary check — fail fast with a clear error if the binary
# is missing or not executable, instead of letting `exec` produce a cryptic
# "exec format error" or "permission denied".
if [[ ! -x "$LLAMA_SERVER_BIN" ]]; then
    if [[ -e "$LLAMA_SERVER_BIN" ]]; then
        log_error "FATAL: llama-server binary at $LLAMA_SERVER_BIN is not executable (check permissions)"
    else
        log_error "FATAL: llama-server binary not found at $LLAMA_SERVER_BIN"
        log_error "Set LLAMA_SERVER_BIN to point at the llama-server executable, or build llama.cpp"
    fi
    exit 3
fi

# E7: Pre-flight GPU check — fail fast with exit 4 if nvidia-smi is broken
# or returns no devices. Set DISABLE_GPU_CHECK=true to bypass.
if [[ "${DISABLE_GPU_CHECK:-}" != "true" ]]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        log_error "FATAL: nvidia-smi not found — GPU driver not installed?"
        log_error "Set DISABLE_GPU_CHECK=true to bypass this check"
        exit 4
    fi
    if ! nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -q .; then
        log_error "FATAL: nvidia-smi returned no GPUs — CUDA may be misconfigured"
        log_error "Set DISABLE_GPU_CHECK=true to bypass this check"
        exit 4
    fi
    log_info "GPU check passed: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
fi

PORT="${PORT:-64000}"
HOST="${HOST:-127.0.0.1}"

# Build the llama-server argv list using the tested Python module.
# The module is the single source of truth for the YAML→flag mapping
# (same logic the test suite covers at 100% branch coverage). Pass HOST/PORT
# as overrides so the systemd unit's environment wins over YAML defaults.
#
# Server-level memory/VRAM optimizations (always applied unless disabled
# via env var). These are process-wide defaults — every model benefits:
#   --no-warmup        skip the post-load warmup (~100 MiB transient,
#                      faster cold start; safe for inference)
#   --kv-unified       single unified KV buffer across slots (~200-400 MiB
#                      persistent; safe when parallel=1)
#   --cache-idle-slots save idle slots to prompt cache (~50-100 MiB during
#                      low traffic; only works with --kv-unified)
#   --no-mmproj-offload keep mmproj on CPU instead of GPU (saves ~572 MiB
#                      VRAM on a Q4_0 mmproj; vision still works, just slower)
#   --no-kv-offload    keep KV cache on CPU/RAM (saves 1-3 GB VRAM at high
#                      ctx; inference still fast — model weights stay on GPU,
#                      only KV crosses PCIe on each token)
#   --cache-ram 16384  16 GiB host-RAM prompt cache (default 8 GiB).
#                      Helps repeated long-context requests hit near-zero
#                      TTFT after the first call.
#   --slot-save-path   persist slot KV state to disk so restart preserves
#                      any in-flight conversation context.
#   --poll 0           disable busy-wait polling (default 50). Sleep mode
#                      reduces CPU pressure and stops CUDA stream warmup.
#
# Universal model-load flags (apply to every model — they're infrastructure
# defaults, not model-specific data, so they live here and not in
# models.yaml):
#   -ctk/-ctv q4_0      q4_0 KV cache quant on 12 GB GPU. Halves VRAM vs q8_0.
#   --cont-batching     continuous batching; required for slot reuse.
#   --parallel 1        single-user setup; the router queues everything serially.
#   --threads 8         8 cores = 3D V-Cache CCD on 5070 Ti; sweet spot
#                       verified at 11,454 MiB VRAM, 106 tok/s decode.
#   -fa on              Flash Attention (RTX 5070 Ti sm_120 supports it).
#
# NOTE: --fit-target 256 was tested and BROKE loads at high ctx — let
# llama.cpp use its default 1024 MiB safety margin instead.
#
# Disable any of them by setting the matching env var to "true":
#   DISABLE_NO_WARMUP=true    -> drops --no-warmup
#   DISABLE_KV_UNIFIED=true   -> drops --kv-unified + --cache-idle-slots
#   DISABLE_MMPROJ_OFFLOAD=true -> drops --no-mmproj-offload
#   DISABLE_NO_KV_OFFLOAD=true  -> drops --no-kv-offload
#   DISABLE_POLL_ZERO=true    -> drops --poll 0
#   DISABLE_CACHE_RAM=true    -> drops --cache-ram + --slot-save-path
#   DISABLE_METRICS=true      -> drops --metrics
EXTRA_ARGS=()
[[ "${DISABLE_NO_WARMUP:-}"    != "true" ]] && EXTRA_ARGS+=(--no-warmup)
[[ "${DISABLE_KV_UNIFIED:-}"   != "true" ]] && EXTRA_ARGS+=(--kv-unified --cache-idle-slots)
[[ "${DISABLE_MMPROJ_OFFLOAD:-}" != "true" ]] && EXTRA_ARGS+=(--no-mmproj-offload)
[[ "${DISABLE_NO_KV_OFFLOAD:-}"  != "true" ]] && EXTRA_ARGS+=(--no-kv-offload)
[[ "${DISABLE_POLL_ZERO:-}"    != "true" ]] && EXTRA_ARGS+=(--poll 0)
[[ "${DISABLE_CACHE_RAM:-}"   != "true" ]] && EXTRA_ARGS+=(--cache-ram 16384 --slot-save-path /tmp/llama-slots)
[[ "${DISABLE_METRICS:-}"     != "true" ]] && EXTRA_ARGS+=(--metrics)
[[ "${DISABLE_SPEC_NGRAM:-}"  != "true" ]] && EXTRA_ARGS+=(--spec-type ngram-cache)
# Universal model-load infrastructure (applies to every model — see comment
# block above at "Universal model-load flags"). Hardcoded rather than in
# models.yaml so they're a single source of truth.
[[ "${DISABLE_CTK_Q4:-}"        != "true" ]] && EXTRA_ARGS+=(-ctk q4_0 -ctv q4_0)
[[ "${DISABLE_CONT_BATCHING:-}" != "true" ]] && EXTRA_ARGS+=(--cont-batching)
[[ "${DISABLE_PARALLEL:-}"      != "true" ]] && EXTRA_ARGS+=(--parallel 1)
[[ "${DISABLE_THREADS:-}"       != "true" ]] && EXTRA_ARGS+=(--threads 16)
[[ "${DISABLE_FLASH_ATTN:-}"    != "true" ]] && EXTRA_ARGS+=(-fa on)
# ----- Universal batch defaults (every model uses these values) -----
[[ "${DISABLE_THREADS_BATCH:-}" != "true" ]] && EXTRA_ARGS+=(--threads-batch 16)
[[ "${DISABLE_BATCH_SIZE:-}"    != "true" ]] && EXTRA_ARGS+=(--batch-size 2048)
# --reasoning-preserve: keep thinking trace in full history, not just last assistant message
[[ "${DISABLE_REASONING_PRESERVE:-}" != "true" ]] && EXTRA_ARGS+=(--reasoning-preserve)
# ----- Observability + speculative decode tuning -----
[[ "${DISABLE_SPEC_PSPLIT:-}"        != "true" ]] && EXTRA_ARGS+=(--spec-draft-p-split 0.10)
[[ "${DISABLE_LOG_VERBOSITY:-}"      != "true" ]] && EXTRA_ARGS+=(--log-verbosity 0)
[[ "${DISABLE_LOG_TIMESTAMPS:-}"    != "true" ]] && EXTRA_ARGS+=(--log-timestamps)
# Note: --prio-batch 2 and --prio 2 are SLOT-SCHEDULING optimizations for
# multi-request concurrent serving. With parallel:1 (our config) they add
# batch-fill overhead with no other request to batch with = −20% throughput
# on single-request workloads. Reverted based on bench data.
#
# Keep the flag off. If we ever switch to parallel:4+ for web serving, revisit.
[[ "${ENABLE_PRIO_BATCH:-}"  == "true" ]] && EXTRA_ARGS+=(--prio-batch 2 --prio 2)

# Marshal EXTRA_ARGS to a temp file (avoids shell-quoting hell with Python)
EXTRA_ARGS_FILE="$(mktemp)"
trap 'rm -f "$EXTRA_ARGS_FILE"' EXIT
printf '%s\n' "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" > "$EXTRA_ARGS_FILE"

read_args() {
    # Capture the python exit code on its own stderr line so the caller can
    # detect build_args failures. The previous shape `mapfile -t ARGS < <(read_args)`
    # silently swallowed the python child's exit code (mapfile's exit status
    # is 0 even when the process-substitution wrote no lines because the
    # python script raised). Failure of build_args used to fall through to
    # `exec llama-server "${ARGS[@]}"` with an empty ARGS array, which
    # caused llama-server to start in router mode with no preset — exactly
    # the broken state we hit on 2026-08-18.
    PYTHONPATH="$LLAMA_BARE_SRC" python3 -c '
import sys
from llama_bare.launcher_config import build_args
extra = [line.strip() for line in open(sys.argv[1]) if line.strip()] or None
args = build_args(sys.argv[2], sys.argv[3], host=sys.argv[4], port=sys.argv[5], extra_args=extra)
for a in args:
    print(a)
' "$EXTRA_ARGS_FILE" "$CONFIG" "$MODEL_NAME" "$HOST" "$PORT"
    PY_RC=$?
    echo "PYTHON_EXIT_CODE=$PY_RC" >&2
    return "$PY_RC"
}

# Capture the python exit code on stderr (via the PYTHON_EXIT_CODE marker
# emitted by read_args) so build_args failures actually halt the script.
# The previous shape `mapfile -t ARGS < <(read_args)` silently swallowed
# the python child's exit code (mapfile's exit status is 0 even when the
# process-substitution wrote no lines because the python script raised).
# Failure of build_args used to fall through to `exec llama-server` with
# an empty ARGS array, which caused llama-server to start in router mode
# with no preset — exactly the broken state we hit on 2026-08-18.
RC_FILE="$(mktemp)"
trap 'rm -f "$EXTRA_ARGS_FILE" "$RC_FILE"' EXIT

if ! ARGS_OUTPUT="$(read_args 2>"$RC_FILE")"; then
    PY_RC=$(grep -oE 'PYTHON_EXIT_CODE=[0-9]+' "$RC_FILE" | tail -1 | cut -d= -f2)
    PY_RC="${PY_RC:-1}"
    log_error "FATAL: build_args failed (python exit $PY_RC)"
    cat "$RC_FILE" >&2
    exit "$PY_RC"
fi

mapfile -t ARGS <<< "$ARGS_OUTPUT"
if [[ ${#ARGS[@]} -eq 0 ]]; then
    log_error "FATAL: build_args returned no arguments"
    exit 1
fi

log_info "--- launcher: loaded config for MODEL_NAME=$MODEL_NAME ---"
printf '  %s\n' "${ARGS[@]}" >&2

log_info "--- launcher: exec llama-server ---"
# Clean up tmpfiles BEFORE exec. Trap-based cleanup does not fire here
# because exec replaces the bash process (the EXIT trap is per-process).
# Without this, every backend restart leaves an EXTRA_ARGS_FILE and
# RC_FILE in /tmp (audit 2026-08-18 finding FFF).
rm -f "$EXTRA_ARGS_FILE" "$RC_FILE"
exec "$LLAMA_SERVER_BIN" "${ARGS[@]}"