#!/usr/bin/env python3
"""Rewrite mmproj: lines in models.yaml to point at Q4_0-quantized sidecars.

For each unique mmproj path in YAML, check the corresponding Q4_0 file exists
on disk. If yes, rewrite the line. If no, skip and warn.

Run from anywhere — uses absolute paths. Backs up the YAML first.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

YAML_PATH = Path("/home/fekry/llama-cpp-docker/llama-unified/models.yaml")
MODELS_DIR = Path("/home/fekry/llama-models/LLM-Models")

# Strip prefix /models/ from the YAML value to get the real disk path
def ypath_to_disk(ypath: str) -> Path:
    if ypath.startswith("/models/"):
        return MODELS_DIR / ypath[len("/models/"):]
    return Path(ypath)


def q4_0_for(disk_path: Path) -> Path | None:
    """Given a disk path to an mmproj file, return the Q4_0 counterpart.

    Naming rules (from the actual quantize batch outputs):
      - mmproj-F32.gguf          -> mmproj-Q4_0.gguf
      - mmproj-BF16.gguf         -> mmproj-Q4_0.gguf
      - mmproj-F16.gguf          -> mmproj-Q4_0.gguf
      - mmproj-model-f16.gguf    -> mmproj-model-f16-Q4_0.gguf  (openbmb)
      - mmproj-f16.gguf          -> mmproj-f16-Q4_0.gguf  (empero-ai)
      - mmproj-Q6_K.gguf         -> mmproj-Q4_0.gguf
      - mmproj-Q5_K_M.gguf       -> mmproj-Q4_0.gguf
      - mmproj-Q5_K_S.gguf       -> mmproj-Q4_0.gguf
      - mmproj-Q4_K_M.gguf       -> mmproj-Q4_0.gguf
      - mmproj-Q4_K_S.gguf       -> mmproj-Q4_0.gguf
      - mmproj-<model>-BF16.gguf -> mmproj-<model>-Q4_0.gguf  (prism Bonsai)
      - gemma-4-26B-it-mmproj.gguf -> gemma-4-26B-it-mmproj-Q4_0.gguf  (google, custom)

    Strips F32/F16/BF16/Q*_K* suffixes from the basename, appends -Q4_0.gguf.
    Special cases: openbmb's "mmproj-model-f16" and prism Bonsai's pattern.
    """
    name = disk_path.name
    parent = disk_path.parent

    # Special case: openbmb mmproj-model-f16.gguf
    if name == "mmproj-model-f16.gguf":
        candidate = parent / "mmproj-model-f16-Q4_0.gguf"
        return candidate if candidate.exists() else None

    # Special case: empero-ai mmproj-Qwable-9B-Claude-Fable-5-f16.gguf
    if "f16.gguf" in name and name.startswith("mmproj-Qwable"):
        candidate = parent / name.replace("f16.gguf", "f16-Q4_0.gguf")
        return candidate if candidate.exists() else None

    # Special case: google gemma-4-26B-it-mmproj.gguf (no quant suffix)
    if name == "gemma-4-26B-it-mmproj.gguf":
        candidate = parent / "gemma-4-26B-it-mmproj-Q4_0.gguf"
        return candidate if candidate.exists() else None

    # General: strip -F32/-F16/-BF16/-Q?_K? suffix, append -Q4_0.gguf.
    # Some files (google gemma-4-12b, lmstudio gemma-4-26B QAT, mistral/ornith,
    # qwen etc.) don't have a quant suffix in the original name. In that case
    # the Q4_0 file lives alongside with the same name + "-Q4_0".
    for suffix_pat in [
        r'-(F32|F16|BF16|Q\d_K(?:_[A-Z])?|Q\d_[A-Z](?:_[A-Z])?)\.gguf$',
        r'\.gguf$',  # no-quant suffix: mmproj-<name>.gguf -> mmproj-<name>-Q4_0.gguf
    ]:
        base = re.sub(suffix_pat, '', name)
        if base != name:
            candidate = parent / f"{base}-Q4_0.gguf"
            if candidate.exists():
                return candidate
            name = base  # try next pattern


def main():
    if not YAML_PATH.exists():
        sys.exit(f"YAML not found: {YAML_PATH}")

    text = YAML_PATH.read_text()
    backup = YAML_PATH.with_suffix(".yaml.bak-pre-q4_0-mmproj")
    backup.write_text(text)
    print(f"Backup: {backup}")

    # Find every mmproj: line and rewrite
    rewrites = 0
    skipped = []
    out_lines = []
    mmproj_re = re.compile(r'^(\s*mmproj:\s*)(.+)$')

    for line in text.splitlines(keepends=True):
        m = mmproj_re.match(line)
        if not m:
            out_lines.append(line)
            continue
        prefix, old_ypath = m.group(1), m.group(2).strip()
        disk_path = ypath_to_disk(old_ypath)
        new_disk = q4_0_for(disk_path)
        if new_disk is None:
            skipped.append((old_ypath, "no Q4_0 candidate found on disk"))
            out_lines.append(line)
            continue
        # Translate back to /models/ prefix to keep YAML format consistent
        try:
            new_rel = new_disk.relative_to(MODELS_DIR)
            new_ypath = "/models/" + str(new_rel)
        except ValueError:
            new_ypath = str(new_disk)
        new_line = f"{prefix}{new_ypath}\n"
        out_lines.append(new_line)
        rewrites += 1
        print(f"  REWRITE: {old_ypath}")
        print(f"       ->: {new_ypath}")

    YAML_PATH.write_text("".join(out_lines))
    print()
    print(f"Done. Rewrites: {rewrites}, skipped: {len(skipped)}")
    if skipped:
        print("Skipped entries (need manual review):")
        for old, reason in skipped:
            print(f"  - {old}: {reason}")


if __name__ == "__main__":
    main()