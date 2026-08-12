"""Tests that all bash scripts in the project pass shellcheck.

Bash scripts in this project:
  - launcher.sh
  - llama-backend.sh
  - restore-systemd.sh
  - final-check.sh
  - scripts/find_max_ctx.sh

These tests run shellcheck with project-appropriate severity. We allow info-level
notes to pass (they're not actionable) but fail on warnings or errors.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

# Scripts to lint (relative to repo root)
REPO_ROOT = Path(__file__).resolve().parents[1]

BASH_SCRIPTS = [
    REPO_ROOT / "launcher.sh",
    REPO_ROOT / "llama-backend.sh",
    REPO_ROOT / "restore-systemd.sh",
    REPO_ROOT / "final-check.sh",
    REPO_ROOT / "scripts" / "find_max_ctx.sh",
]


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
@pytest.mark.parametrize("script_path", BASH_SCRIPTS, ids=lambda p: p.name)
def test_bash_script_passes_shellcheck(script_path):
    """Each bash script must pass shellcheck with no warnings or errors."""
    assert script_path.exists(), f"Script not found: {script_path}"
    # -x: allow following source files (none here, but harmless)
    # -S warning: only fail on warnings+ (info-level is OK)
    result = subprocess.run(
        ["shellcheck", "-S", "warning", str(script_path)],
        capture_output=True,
        text=True,
    )
    # shellcheck exits 0 if no issues at the requested severity
    assert result.returncode == 0, (
        f"shellcheck failed for {script_path.name}:\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_all_bash_scripts_have_shebangs():
    """Every bash script starts with #!/bin/bash or #!/usr/bin/env bash."""
    for script in BASH_SCRIPTS:
        if not script.exists():
            continue
        first_line = script.read_text().splitlines()[0]
        assert first_line.startswith("#!"), f"{script.name} missing shebang"
        assert "bash" in first_line, f"{script.name} shebang is not bash: {first_line}"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
@pytest.mark.parametrize("script_path", BASH_SCRIPTS, ids=lambda p: p.name)
def test_bash_scripts_have_valid_syntax(script_path):
    """Every bash script must pass `bash -n` (syntax check)."""
    if not script_path.exists():
        pytest.skip(f"{script_path.name} does not exist")
    result = subprocess.run(
        ["bash", "-n", str(script_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"bash syntax error in {script_path.name}: {result.stderr}"
    )
