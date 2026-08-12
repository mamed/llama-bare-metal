"""Tests for scripts/rewrite_mmproj_to_q4_0.py helper functions.

The script itself touches the live models.yaml and is run once-off, but the
pure helper functions (ypath_to_disk, q4_0_for) benefit from unit tests so
they don't regress if someone edits the rewrite rules.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add the scripts directory to sys.path so we can import the script as a module
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

# Import the rewrite module
import rewrite_mmproj_to_q4_0 as rw  # noqa: E402


@pytest.fixture
def tmp_models_dir(tmp_path):
    """Create a fake MODELS_DIR with a few mmproj files for testing."""
    return tmp_path


# ============================================================
# ypath_to_disk tests
# ============================================================

class TestYpathToDisk:
    """Tests for translating a YAML /models/... path to a real disk path."""

    def test_strips_models_prefix(self):
        result = rw.ypath_to_disk("/models/foo/bar.gguf")
        # The MODELS_DIR is appended after stripping
        assert str(result).endswith("foo/bar.gguf")
        assert not str(result).startswith("/models/")

    def test_passthrough_for_non_models_path(self):
        result = rw.ypath_to_disk("/absolute/path/file.gguf")
        assert str(result) == "/absolute/path/file.gguf"

    def test_empty_string_returns_current_dir(self):
        # Path("") resolves to Path(".") — the script doesn't have special
        # handling for empty input. Documented behavior.
        result = rw.ypath_to_disk("")
        assert result == Path(".")

    def test_relative_path_passes_through(self):
        result = rw.ypath_to_disk("relative/path.gguf")
        assert str(result) == "relative/path.gguf"


# ============================================================
# q4_0_for tests (uses a temp MODELS_DIR with fake files)
# ============================================================

class TestQ4PathForF32:
    """Standard pattern: mmproj-F32.gguf -> mmproj-Q4_0.gguf."""

    def test_f32_to_q4_0(self, tmp_models_dir):
        src = tmp_models_dir / "mmproj-F32.gguf"
        src.touch()  # source doesn't need to exist; only the candidate
        cand = tmp_models_dir / "mmproj-Q4_0.gguf"
        cand.touch()
        # We have to monkeypatch MODELS_DIR since the script uses it as a global
        original = rw.MODELS_DIR
        rw.MODELS_DIR = tmp_models_dir
        try:
            result = rw.q4_0_for(src)
            assert result == cand
        finally:
            rw.MODELS_DIR = original


class TestQ4PathForBF16:
    """mmproj-BF16.gguf -> mmproj-Q4_0.gguf."""

    def test_bf16_to_q4_0(self, tmp_models_dir):
        src = tmp_models_dir / "mmproj-BF16.gguf"
        cand = tmp_models_dir / "mmproj-Q4_0.gguf"
        cand.touch()
        original = rw.MODELS_DIR
        rw.MODELS_DIR = tmp_models_dir
        try:
            assert rw.q4_0_for(src) == cand
        finally:
            rw.MODELS_DIR = original


class TestQ4PathForF16:
    """mmproj-F16.gguf -> mmproj-Q4_0.gguf."""

    def test_f16_to_q4_0(self, tmp_models_dir):
        src = tmp_models_dir / "mmproj-F16.gguf"
        cand = tmp_models_dir / "mmproj-Q4_0.gguf"
        cand.touch()
        original = rw.MODELS_DIR
        rw.MODELS_DIR = tmp_models_dir
        try:
            assert rw.q4_0_for(src) == cand
        finally:
            rw.MODELS_DIR = original


class TestQ4PathForOpenBMB:
    """Special case: openbmb mmproj-model-f16.gguf -> mmproj-model-f16-Q4_0.gguf."""

    def test_openbmb_special(self, tmp_models_dir):
        src = tmp_models_dir / "mmproj-model-f16.gguf"
        cand = tmp_models_dir / "mmproj-model-f16-Q4_0.gguf"
        cand.touch()
        original = rw.MODELS_DIR
        rw.MODELS_DIR = tmp_models_dir
        try:
            assert rw.q4_0_for(src) == cand
        finally:
            rw.MODELS_DIR = original


class TestQ4PathForGoogleCustom:
    """Special case: google gemma-4-26B-it-mmproj.gguf -> gemma-4-26B-it-mmproj-Q4_0.gguf."""

    def test_google_custom_name(self, tmp_models_dir):
        src = tmp_models_dir / "gemma-4-26B-it-mmproj.gguf"
        cand = tmp_models_dir / "gemma-4-26B-it-mmproj-Q4_0.gguf"
        cand.touch()
        original = rw.MODELS_DIR
        rw.MODELS_DIR = tmp_models_dir
        try:
            assert rw.q4_0_for(src) == cand
        finally:
            rw.MODELS_DIR = original


class TestQ4PathForEmperoF16:
    """Special case: empero-ai mmproj-Qwable-9B-Claude-Fable-5-f16.gguf."""

    def test_empero_ai_qwable(self, tmp_models_dir):
        src = tmp_models_dir / "mmproj-Qwable-9B-Claude-Fable-5-f16.gguf"
        cand = tmp_models_dir / "mmproj-Qwable-9B-Claude-Fable-5-f16-Q4_0.gguf"
        cand.touch()
        original = rw.MODELS_DIR
        rw.MODELS_DIR = tmp_models_dir
        try:
            assert rw.q4_0_for(src) == cand
        finally:
            rw.MODELS_DIR = original


class TestQ4PathNoCandidate:
    """When no Q4_0 candidate exists, return None."""

    def test_returns_none_when_no_candidate(self, tmp_models_dir):
        src = tmp_models_dir / "mmproj-F32.gguf"
        # No Q4_0 file exists in the dir
        original = rw.MODELS_DIR
        rw.MODELS_DIR = tmp_models_dir
        try:
            assert rw.q4_0_for(src) is None
        finally:
            rw.MODELS_DIR = original

    def test_returns_none_for_openbmb_when_missing(self, tmp_models_dir):
        src = tmp_models_dir / "mmproj-model-f16.gguf"
        # Q4_0 candidate missing
        original = rw.MODELS_DIR
        rw.MODELS_DIR = tmp_models_dir
        try:
            assert rw.q4_0_for(src) is None
        finally:
            rw.MODELS_DIR = original


class TestQ4PathForQuantizedSource:
    """If source is already quantized (Q5_K, Q6_K, etc.), use --allow-requantize pattern.

    The script's behavior here is: strip the existing quant suffix and try to find
    a Q4_0 counterpart. If the user already has a Q5_K_M file, the script tries
    Q4_0 by stripping that suffix.
    """

    def test_q6_k_to_q4_0(self, tmp_models_dir):
        src = tmp_models_dir / "mmproj-Q6_K.gguf"
        cand = tmp_models_dir / "mmproj-Q4_0.gguf"
        cand.touch()
        original = rw.MODELS_DIR
        rw.MODELS_DIR = tmp_models_dir
        try:
            assert rw.q4_0_for(src) == cand
        finally:
            rw.MODELS_DIR = original

    def test_q5_k_m_to_q4_0(self, tmp_models_dir):
        src = tmp_models_dir / "mmproj-Q5_K_M.gguf"
        cand = tmp_models_dir / "mmproj-Q4_0.gguf"
        cand.touch()
        original = rw.MODELS_DIR
        rw.MODELS_DIR = tmp_models_dir
        try:
            assert rw.q4_0_for(src) == cand
        finally:
            rw.MODELS_DIR = original


class TestQ4PathFallbackForUnquantized:
    """When the mmproj has no quant suffix (e.g. lmstudio-community pattern),
    the script falls back to the `\\.gguf$` pattern which appends -Q4_0.
    """

    def test_unquantized_basename(self, tmp_models_dir):
        src = tmp_models_dir / "mmproj-foo.gguf"
        cand = tmp_models_dir / "mmproj-foo-Q4_0.gguf"
        cand.touch()
        original = rw.MODELS_DIR
        rw.MODELS_DIR = tmp_models_dir
        try:
            assert rw.q4_0_for(src) == cand
        finally:
            rw.MODELS_DIR = original
