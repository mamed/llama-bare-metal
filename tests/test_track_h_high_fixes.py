"""Tests for the Phase H HIGH fixes (H-1: SIGTERM handler, H-3: MODELS_YAML
consistency, H-4: find_max_ctx.sh concurrency guard).

We don't import the full llama-router.py module for these tests — we exercise
the new pieces in isolation so the test doesn't depend on module-level
state set up by other tests.

For H-1: the SIGTERM handler is wired into `main()` in llama-router.py.
We test it indirectly by inspecting that the handler is installed, and
directly by simulating a SIGTERM delivery to a controlled ThreadingHTTPServer
instance.

For H-3: we verify both systemd units point at the same MODELS_YAML path.

For H-4: we exercise the find_max_ctx.sh lockfile via subprocess so we hit
the real bash code path.
"""

from __future__ import annotations

import importlib.util
import io
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_router(**env_overrides):
    """Load llama-router.py fresh, with the given env overrides applied."""
    sys.modules.pop("llama_router_loaded", None)
    for key, value in env_overrides.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    spec = importlib.util.spec_from_file_location("llama_router_loaded", REPO_ROOT / "llama-router.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ===========================================================================
# H-1: SIGTERM handler
# ===========================================================================


def test_sigterm_handler_is_installed_in_main(tmp_path):
    """H-1: main() must install a SIGTERM handler before serve_forever().

    We run main() in a real subprocess (not a thread) so signal.signal()
    actually works (Python only delivers signals to the main thread of the
    main interpreter). The subprocess writes a sentinel file when the
    SIGTERM handler runs — which proves the handler was installed by main().
    """
    sentinel_dir = tmp_path / "sentinel"
    sentinel_dir.mkdir()
    handler_ran = sentinel_dir / "handler_ran"
    driver = tmp_path / "driver.py"
    driver.write_text(
        "import sys, os, importlib.util, signal, time\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        f"os.environ['ROUTER_API_TOKEN'] = 'test-token-for-sigterm-handler-test'\n"
        f"os.environ['ROUTER_PORT'] = '0'\n"
        f"os.environ['HEALTH_PROBE_INTERVAL'] = '3600'\n"
        f"spec = importlib.util.spec_from_file_location('llama_router_loaded', {str(REPO_ROOT / 'llama-router.py')!r})\n"
        "r = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(r)\n"
        f"handler_ran = {str(handler_ran)!r}\n"
        # Wrap the handler so we can prove main() installed one.
        "orig_signal = signal.signal\n"
        "def wrapped_signal(signum, handler):\n"
        "    if signum == signal.SIGTERM:\n"
        "        # Wrap the handler to record it was called.\n"
        "        def wrapped(signum_, frame):\n"
        "            open(handler_ran, 'w').write('ran')\n"
        "            handler(signum_, frame)\n"
        "        return orig_signal(signum, wrapped)\n"
        "    return orig_signal(signum, handler)\n"
        "signal.signal = wrapped_signal\n"
        "class FakeServer:\n"
        "    def __init__(self, *a, **kw): self._stopped = False\n"
        "    def serve_forever(self):\n"
        "        while not self._stopped: time.sleep(0.05)\n"
        "    def shutdown(self): self._stopped = True\n"
        "    def __getattr__(self, name): return lambda *a, **kw: None\n"
        "r.ThreadingHTTPServer = FakeServer\n"
        "try:\n"
        "    r.main()\n"
        "except SystemExit:\n"
        "    pass\n"
    )
    proc = subprocess.Popen(
        [sys.executable, str(driver)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        # Give main() time to install the SIGTERM handler and enter
        # serve_forever().
        time.sleep(2)
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
    assert handler_ran.exists(), (
        f"main() did not install a SIGTERM handler (or handler never ran); "
        f"stderr={proc.stderr.read()!r}"
    )


def test_sigterm_handler_triggers_drain_and_shutdown(tmp_path):
    """H-1: sending SIGTERM must trigger drain + server.shutdown().

    Run main() in a real subprocess (not a thread) so signal.signal() is
    actually honored (Python only delivers signals to the main thread of
    the main interpreter). The subprocess writes a sentinel file when the
    SIGTERM handler completes both the drain and the server.shutdown().
    """
    sentinel_dir = tmp_path / "sentinel"
    sentinel_dir.mkdir()
    drained_path = sentinel_dir / "drained"
    shutdown_path = sentinel_dir / "shutdown"
    driver = tmp_path / "driver.py"
    # Load llama-router.py via importlib (it's a script, not a package).
    driver.write_text(
        "import sys, os, importlib.util\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        f"os.environ['ROUTER_API_TOKEN'] = 'test-token-for-sigterm-handler-test'\n"
        f"os.environ['ROUTER_PORT'] = '0'\n"
        f"os.environ['HEALTH_PROBE_INTERVAL'] = '3600'\n"
        f"spec = importlib.util.spec_from_file_location('llama_router_loaded', {str(REPO_ROOT / 'llama-router.py')!r})\n"
        "r = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(r)\n"
        f"drained_path = {str(drained_path)!r}\n"
        f"shutdown_path = {str(shutdown_path)!r}\n"
        "def fake_drain(timeout=None):\n"
        "    open(drained_path, 'w').write('drained')\n"
        "    return True\n"
        "r._inflight_drain = fake_drain\n"
        "class FakeServer:\n"
        "    def __init__(self, *a, **kw):\n"
        "        self._stopped = False\n"
        "    def serve_forever(self):\n"
        "        import time as _t\n"
        "        while not self._stopped:\n"
        "            _t.sleep(0.05)\n"
        "    def shutdown(self):\n"
        "        self._stopped = True\n"
        "        open(shutdown_path, 'w').write('shutdown')\n"
        "    def __getattr__(self, name):\n"
        "        return lambda *a, **kw: None\n"
        "r.ThreadingHTTPServer = FakeServer\n"
        "try:\n"
        "    r.main()\n"
        "except SystemExit:\n"
        "    pass\n"
    )
    proc = subprocess.Popen(
        [sys.executable, str(driver)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        # Give main() time to install the SIGTERM handler and enter
        # serve_forever().
        time.sleep(2)
        # Send SIGTERM if main() is still running (it almost certainly is).
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
        # Wait up to 10s for the process to exit.
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
    assert drained_path.exists(), (
        f"SIGTERM handler did not call _inflight_drain; stderr={proc.stderr.read()!r}"
    )
    assert shutdown_path.exists(), (
        f"SIGTERM handler did not call server.shutdown; stderr={proc.stderr.read()!r}"
    )


def test_inflight_drain_returns_true_when_no_inflight():
    """E3: drain returns True immediately when no in-flight requests."""
    mod = _load_router(ROUTER_API_TOKEN="test-token-for-drain-empty-test")
    mod._inflight_value[0] = 0
    start = time.monotonic()
    result = mod._inflight_drain(timeout=5)
    elapsed = time.monotonic() - start
    assert result is True
    assert elapsed < 0.1, f"drain should be instant when nothing in-flight, took {elapsed:.3f}s"


def test_inflight_drain_returns_true_when_count_drops_to_zero():
    """E3: drain waits for the in-flight count to drop to zero."""
    mod = _load_router(ROUTER_API_TOKEN="test-token-for-drain-decrement-test")
    mod._inflight_value[0] = 1

    def release_after_delay():
        time.sleep(0.2)
        mod._inflight_value[0] = 0

    threading.Thread(target=release_after_delay, daemon=True).start()
    start = time.monotonic()
    result = mod._inflight_drain(timeout=5)
    elapsed = time.monotonic() - start
    assert result is True
    assert 0.15 < elapsed < 1.0, f"drain should wait ~0.2s, took {elapsed:.3f}s"


def test_inflight_acquire_release_roundtrip():
    """E3: in-flight counter is correctly incremented and decremented."""
    mod = _load_router(ROUTER_API_TOKEN="test-token-for-inflight-roundtrip")
    mod._inflight_value[0] = 0
    mod._inflight_acquire()
    assert mod._inflight_value[0] == 1
    mod._inflight_acquire()
    assert mod._inflight_value[0] == 2
    mod._inflight_release()
    assert mod._inflight_value[0] == 1
    mod._inflight_release()
    assert mod._inflight_value[0] == 0


# ===========================================================================
# H-3: MODELS_YAML path consistency between systemd units
# ===========================================================================


def test_router_systemd_unit_uses_correct_models_yaml():
    """H-3: systemd/llama-router.service MODELS_YAML must match backend."""
    router_unit = (REPO_ROOT / "systemd" / "llama-router.service").read_text()
    backend_unit = (REPO_ROOT / "systemd" / "llama-backend.service").read_text()
    router_match = re.search(r"Environment=MODELS_YAML=(\S+)", router_unit)
    backend_match = re.search(r"Environment=CONFIG_FILE=(\S+)", backend_unit)
    assert router_match, "router unit missing MODELS_YAML"
    assert backend_match, "backend unit missing CONFIG_FILE"
    assert router_match.group(1) == backend_match.group(1), (
        f"MODELS_YAML mismatch: router={router_match.group(1)!r} "
        f"backend={backend_match.group(1)!r}"
    )


def test_no_root_level_service_file():
    """I-2: there must be no top-level llama-router.service. Otherwise a
    fresh clone where someone `cp llama-router.service ~/.config/systemd/user/`
    would deploy the UNHARDENED version (audit 2026-08-18). The canonical
    location is systemd/llama-router.service; the watcher and backend
    units also live in systemd/. Always install via restore-systemd.sh.
    """
    root_path = REPO_ROOT / "llama-router.service"
    assert not root_path.exists(), (
        f"{root_path.name} is present at the repo root. The hardened version "
        f"lives in systemd/. Delete this file or people will deploy the "
        f"unhardened version by mistake. See audit 2026-08-18 finding F."
    )


# ===========================================================================
# H-4: find_max_ctx.sh concurrency guard + atomic write
# ===========================================================================


def _run_find_max_ctx_setup_lock(tmp_path):
    """Run the script's set_ctx logic by invoking the script in a way that
    exercises only the lock + atomic write (not the actual model load).
    Returns the yaml_path used for the test."""
    yaml_path = tmp_path / "models.yaml"
    yaml_path.write_text(
        "models:\n"
        "  - name: test-model\n"
        "    model: /models/test/file.gguf\n"
        "    context_size: 4096\n"
    )
    return yaml_path


def test_find_max_ctx_has_lock_logic():
    """H-4: find_max_ctx.sh must create a lock dir at startup and clean up on exit.

    We inspect the source directly (rather than racing against subprocess.Popen)
    so the test is deterministic and doesn't depend on timing or a real systemd.
    """
    script = REPO_ROOT / "scripts" / "find_max_ctx.sh"
    text = script.read_text()
    # Verify a LOCK_DIR variable is defined.
    assert re.search(r'^\s*LOCK_DIR=', text, re.MULTILINE), (
        "no LOCK_DIR variable found in find_max_ctx.sh"
    )
    # Verify the lock is acquired via mkdir (POSIX-portable flock).
    assert re.search(r'mkdir\s+"?\$LOCK_DIR"?', text) or "mkdir \"$LOCK_DIR\"" in text, (
        "no mkdir $LOCK_DIR call found — script does not acquire the lock"
    )
    # Verify a cleanup trap is installed (rmdir on EXIT).
    assert re.search(r"trap\s+['\"]?rmdir\s+['\"]?\$LOCK_DIR", text), (
        "no `trap rmdir $LOCK_DIR` cleanup found — lock will leak on exit"
    )
    # The trap should fire on EXIT (and ideally also INT/TERM).
    m_trap = re.search(r"trap\s+[^#\n]*\$LOCK_DIR[^#\n]*?((?:EXIT|INT|TERM|HUP|QUIT)(?:\s+\S+)*)", text)
    assert m_trap, "could not parse trap signal list"
    trap_signals = m_trap.group(1).split()
    assert "EXIT" in trap_signals, (
        f"trap must include EXIT so lock is always released; got {trap_signals!r}"
    )


def test_find_max_ctx_lock_is_unique_and_in_tmp():
    """H-4: the lock dir path should live in /tmp and be find_max_ctx-specific."""
    script = REPO_ROOT / "scripts" / "find_max_ctx.sh"
    text = script.read_text()
    m = re.search(r'^\s*LOCK_DIR=([^\s#]+)', text, re.MULTILINE)
    assert m, "could not parse LOCK_DIR path"
    # Strip any leading double-quote.
    lock_path = m.group(1).strip('"').strip("'")
    assert lock_path.startswith("/tmp/"), (
        f"lock dir should be in /tmp/, got {lock_path!r}"
    )
    assert "find_max_ctx" in lock_path, (
        f"lock dir should be find_max_ctx-specific, got {lock_path!r}"
    )


def test_find_max_ctx_uses_atomic_write():
    """H-4: the script must use temp + rename for atomic YAML writes."""
    script = REPO_ROOT / "scripts" / "find_max_ctx.sh"
    text = script.read_text()
    # Look for temp-file pattern (.tmp suffix, atomic rename, or os.replace()).
    has_temp = ".tmp" in text or "temp" in text.lower()
    has_atomic_rename = (
        "os.replace" in text
        or re.search(r"\bmv\s+", text) is not None
        or re.search(r"\bos\.rename\(", text) is not None
    )
    assert has_temp, "no temp-file pattern found for atomic write"
    assert has_atomic_rename, (
        "no atomic rename (mv / os.replace / os.rename) found for YAML write"
    )


def test_find_max_ctx_exits_zero_when_already_locked(tmp_path):
    """H-4: a second invocation while the lock is held exits 0 immediately."""
    lock_dir = Path("/tmp/find_max_ctx.lock")
    if lock_dir.exists():
        try:
            lock_dir.rmdir()
        except OSError:
            pytest.skip("could not clear stale lock dir")
    lock_dir.mkdir()
    try:
        yaml_path = _run_find_max_ctx_setup_lock(tmp_path)
        script = REPO_ROOT / "scripts" / "find_max_ctx.sh"
        start = time.monotonic()
        result = subprocess.run(
            ["bash", str(script), str(yaml_path), "test-model", "8192"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        elapsed = time.monotonic() - start
        assert result.returncode == 0, (
            f"expected exit 0 (skip), got {result.returncode}: {result.stderr}"
        )
        assert elapsed < 1.0, f"second invocation should exit fast, took {elapsed:.2f}s"
        assert "another invocation is in progress" in result.stdout + result.stderr
    finally:
        if lock_dir.exists():
            try:
                lock_dir.rmdir()
            except OSError:
                pass
