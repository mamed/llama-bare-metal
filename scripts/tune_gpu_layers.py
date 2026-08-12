#!/usr/bin/env python3
"""tune_gpu_layers.py — for each entry in models.yaml, ensure file_size +
estimated compute buffers fit in available VRAM (12.2 GB minus driver
reservation). If they don't fit, reduce gpu_layers proportionally.

Strategy:
  1. For each model, read GGUF metadata (n_layer, n_embd) and file size
  2. Compute the per-layer VRAM cost
  3. Available VRAM = 12.2 GB - 453 MB driver reservation = 11.77 GB
  4. For full offload (gpu_layers=99): VRAM_needed = file_size + compute_buffers
     If VRAM_needed > 11.77 GB, reduce gpu_layers so it fits
  5. Update YAML with the new gpu_layers value

Compute buffers at ctx=262144 for full offload ~ 1 GB for small models,
3 GB for 26B, 5 GB for 31B+. Empirically: ~ 0.05 GB per layer per 100k ctx.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

YAML = Path("/home/fekry/llama-cpp-docker/llama-unified/models.yaml")
MODELS_DIR = Path("/home/fekry/llama-models/LLM-Models")
AVAILABLE_VRAM_GB = 11.77  # 12.23 total - 0.45 driver reservation
TARGET_CTX = 262144

# Extract metadata from GGUF
def get_gguf_meta(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        from gguf import GGUFReader
        r = GGUFReader(str(path))
        meta = {}
        for f in r.fields:
            ln = f.name.lower()
            if "context_length" in ln and "vision" not in ln:
                meta["ctx_train"] = f.parts[f.data[0]]
            elif "embedding_length" in ln and "vision" not in ln and "audio" not in ln:
                meta["n_embd"] = f.parts[f.data[0]]
            elif ln == "block_count.int32" or ln.endswith(".block_count"):
                meta["n_layer"] = f.parts[f.data[0]]
            elif "head_count" in ln and "vision" not in ln and "kv" not in ln:
                meta["n_head"] = f.parts[f.data[0]]
        meta["file_size_gb"] = path.stat().st_size / 1024 / 1024 / 1024
        return meta
    except Exception as e:
        return None


def estimate_vram_needed(file_size_gb: float, n_layer: int, ctx: int, n_embd: int) -> float:
    """VRAM needed for full offload (gpu_layers = n_layer) at given ctx.

    Includes:
      - file_size_gb (the model weights)
      - compute_buffers (estimate, scales with ctx and model size)
      - 0.5 GB overhead for CUDA context + driver
    """
    # Empirically validated: ~0.05 GB per layer per 100k ctx for compute scratch
    # Plus a baseline 0.3 GB that doesn't scale with ctx
    compute_buf = 0.3 + 0.05 * n_layer * (ctx / 100000)
    return file_size_gb + compute_buf + 0.5


def recommended_gpu_layers(file_size_gb: float, n_layer: int, ctx: int, n_embd: int) -> int:
    """How many layers can fit at full GPU offload? Returns <= n_layer."""
    per_layer_gb = file_size_gb / n_layer  # rough: file_size / layers
    # reserve compute_buffers + overhead
    compute_buf = 0.3 + 0.05 * n_layer * (ctx / 100000)
    available_for_weights = AVAILABLE_VRAM_GB - compute_buf - 0.5
    if per_layer_gb <= 0:
        return n_layer
    fit_layers = int(available_for_weights / per_layer_gb)
    return min(fit_layers, n_layer)


def main():
    text = YAML.read_text()
    entries = list(re.finditer(r'(- name: ([^\n]+)\n((?:  [^\n]+\n)*))', text))

    updates = []
    skipped = []
    for m in entries:
        full_name = m.group(2)
        block = m.group(3)

        # Get model path
        model_m = re.search(r'  model: (\S+)', block)
        if not model_m:
            continue
        ypath = model_m.group(1)
        # Translate to disk path
        if ypath.startswith("/models/"):
            disk_path = MODELS_DIR / ypath[len("/models/"):]
        else:
            disk_path = Path(ypath)

        # Get current gpu_layers
        ngl_m = re.search(r'  gpu_layers:\s*(\d+)', block)
        current_ngl = int(ngl_m.group(1)) if ngl_m else 99

        # Get current ctx (we set all to 262144, but be safe)
        ctx_m = re.search(r'  context_size:\s*(\d+)', block)
        current_ctx = int(ctx_m.group(1)) if ctx_m else TARGET_CTX

        meta = get_gguf_meta(disk_path)
        if not meta or "n_layer" not in meta:
            skipped.append((full_name, "no GGUF metadata"))
            continue

        # Cap ctx at model's training ctx
        ctx = min(current_ctx, meta.get("ctx_train", TARGET_CTX))

        # Estimate VRAM needed at full offload
        vram_needed = estimate_vram_needed(
            meta["file_size_gb"],
            meta["n_layer"],
            ctx,
            meta.get("n_embd", 0),
        )

        if vram_needed <= AVAILABLE_VRAM_GB:
            # Fits at full offload
            new_ngl = meta["n_layer"]
            status = "FIT" if new_ngl == current_ngl else f"FIT (was {current_ngl})"
            updates.append((full_name, new_ngl, ctx, meta["file_size_gb"], vram_needed, status))
        else:
            # Need to reduce gpu_layers
            new_ngl = recommended_gpu_layers(
                meta["file_size_gb"], meta["n_layer"], ctx, meta.get("n_embd", 0)
            )
            updates.append((
                full_name, new_ngl, ctx, meta["file_size_gb"], vram_needed,
                f"REDUCE {current_ngl} -> {new_ngl}"
            ))

    # Show results
    print(f"\n{'='*100}")
    print(f"{'Model':50} {'Size':>8} {'EstVRAM':>8} {'ngl':>4}  Status")
    print(f"{'='*100}")
    for name, ngl, ctx, size, vram, status in updates:
        print(f"{name:50} {size:>6.2f}GB {vram:>6.2f}GB {ngl:>4}  {status}")

    print(f"\n{len(skipped)} skipped:")
    for name, reason in skipped:
        print(f"  {name}: {reason}")


if __name__ == "__main__":
    main()