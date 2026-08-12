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


def test_sigterm_handler_is_installed_in_main():
    """H-1: main() must install a SIGTERM handler before serve_forever()."""
    mod = _load_router(ROUTER_API_TOKEN="test-token-for-sigterm-handler-test")
    # Capture the signal handler as it stands before main() runs.
    original = signal.getsignal(signal.SIGTERM)
    try:
        # Stand up a fake server so we don't bind a real port.
        class _FakeServer:
            def __init__(self, *a, **k):
                pass

            def serve_forever(self):
                # Block until shutdown is called.
                while not getattr(self, "_stopped", False):
                    time.sleep(0.01)

            def shutdown(self):
                self._stopped = True

        mod.ThreadingHTTPServer = _FakeServer
        # Spawn main() in a thread because it blocks on serve_forever.
        server_ref = []

        original_main = mod.main

        def main_wrapper():
            # We don't want the actual logger to write to disk for this test.
            import logging
            logging.getLogger("llama_router").addHandler(logging.NullHandler())
            mod._configure_logger()
            # Call main, but inject our fake server so we control it.
            server = _FakeServer()
            server_ref.append(server)
            mod.ThreadingHTTPServer = lambda *a, **k: server
            original_main()

        t = threading.Thread(target=main_wrapper, daemon=True)
        t.start()
        # Give main() a moment to install the SIGTERM handler and start serving.
        time.sleep(0.3)
        # The handler must now NOT be the Python default.
        installed = signal.getsignal(signal.SIGTERM)
        assert installed != signal.SIG_DFL, (
            "main() did not install a SIGTERM handler — H-1 fix is missing"
        )
    finally:
        signal.signal(signal.SIGTERM, original)
        # Re-raise SIGTERM in the thread so it exits cleanly.
        if server_ref:
            server_ref[0].shutdown()


def test_sigterm_handler_triggers_drain_and_shutdown():
    """H-1: sending SIGTERM must trigger drain + server.shutdown()."""
    mod = _load_router(ROUTER_API_TOKEN="test-token-for-sigterm-handler-test")

    shutdown_called = threading.Event()
    drain_called = threading.Event()

    class _FakeServer:
        def __init__(self, *a, **k):
            pass

        def serve_forever(self):
            # Block until shutdown.
            shutdown_called.wait(timeout=5)

        def shutdown(self):
            shutdown_called.set()

    mod.ThreadingHTTPServer = _FakeServer
    mod._inflight_value[0] = 0

    # Monkeypatch _inflight_drain to record that it was called.
    original_drain = mod._inflight_drain
    def fake_drain(timeout=None):
        drain_called.set()
        return True
    mod._inflight_drain = fake_drain

    # Spy on the SIGTERM handler by installing it through main().
    original = signal.getsignal(signal.SIGTERM)
    try:
        # Run main() in a thread; immediately raise SIGTERM to ourselves.
        t = threading.Thread(target=mod.main, daemon=True)
        t.start()
        time.sleep(0.2)
        os.kill(os.getpid(), signal.SIGTERM)
        # Give it a moment to process.
        time.sleep(0.2)
        assert drain_called.is_set(), "SIGTERM handler did not call _inflight_drain"
        assert shutdown_called.is_set(), "SIGTERM handler did not call server.shutdown"
    finally:
        signal.signal(signal.SIGTERM, original)


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


def test_root_router_service_matches_systemd_copy():
    """I-2: the root-level llama-router.service must match the systemd/ copy."""
    root_copy = (REPO_ROOT / "llama-router.service").read_text()
    systemd_copy = (REPO_ROOT / "systemd" / "llama-router.service").read_text()
    root_match = re.search(r"Environment=MODELS_YAML=(\S+)", root_copy)
    systemd_match = re.search(r"Environment=MODELS_YAML=(\S+)", systemd_copy)
    assert root_match, "root llama-router.service missing MODELS_YAML"
    assert systemd_match, "systemd/llama-router.service missing MODELS_YAML"
    assert root_match.group(1) == systemd_match.group(1), (
        f"root copy points at {root_match.group(1)!r} but systemd copy points at "
        f"{systemd_match.group(1)!r}"
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


def test_find_max_ctx_creates_lock_dir_on_invocation(tmp_path):
    """H-4: the script must create the lock directory while running."""
    # Remove the lock if a previous run left it.
    lock_dir = Path("/tmp/find_max_ctx.lock")
    if lock_dir.exists():
        # Another test is running or it was left over; remove to start fresh.
        try:
            lock_dir.rmdir()
        except OSError:
            pass
    yaml_path = _run_find_max_ctx_setup_lock(tmp_path)
    # Start the script in the background and check the lock appears.
    script = REPO_ROOT / "scripts" / "find_max_ctx.sh"
    # The script will try to `systemctl --user restart llama-backend.service`
    # which will fail in test environments — but we only need to verify the
    # lock is created at the START of the script (before systemctl runs).
    # Use `set +e` semantics by not caring about the script's exit code.
    proc = subprocess.Popen(
        ["bash", str(script), str(yaml_path), "test-model", "8192"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Give the script a moment to mkdir the lock.
    time.sleep(0.2)
    lock_present = lock_dir.exists()
    proc.kill()
    proc.wait(timeout=5)
    # Cleanup the lock (the trap should do this on exit; we kill -9 above so
    # it may not have run).
    if lock_dir.exists():
        try:
            lock_dir.rmdir()
        except OSError:
            pass
    assert lock_present, "find_max_ctx.sh did not create /tmp/find_max_ctx.lock"


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
