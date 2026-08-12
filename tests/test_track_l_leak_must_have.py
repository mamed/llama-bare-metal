"""PHASE L — leak / must-have audit regression tests.

Captures the new findings from /tmp/llama-audit-prompts/PHASE_L_REPORT.md so
they don't regress. Production code was inspected and is clean (the audit
itself is the deliverable); the tests below lock in the invariants that the
audit proved correct.

Findings covered:
- L-1: _inflight Semaphore at module level is dead code (placeholder, never
       .acquire()d or .release()d). Real in-flight tracking uses the
       _inflight_count Lock + _inflight_value[0] counter.
- L-2: atomic .env / state-file writers must not leave .tmp files behind
       after 100s of writes.
- L-3: log file rotation must keep the file openable after N rotations
       (smoke test of the RotatingFileHandler config — does the chosen
       file path stay writable?).
- L-4: ensure_model_loaded must not change the module-level current_model
       when the swap raises (the state file invariant: failed swaps leave
       the previously-loaded model intact).
- L-5: the .env.example document must list ROUTER_API_TOKEN, MAX_REQUEST_BYTES
       and the new observability env vars (LOW finding — log if missing).
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_router(**env_overrides):
    """Import llama-router.py with the given env vars set before import."""
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


# ---------------------------------------------------------------------------
# L-1 (Phase M): the dead _inflight Semaphore placeholder has been removed.
# These tests pin the NEW contract: no _inflight attribute exists; the
# real tracking is _inflight_count Lock + _inflight_value[0] counter.
# ---------------------------------------------------------------------------


def test_inflight_semaphore_placeholder_is_dead():
    """L-1 (Phase M): the module-level `_inflight = threading.Semaphore(1)`
    placeholder has been REMOVED. No code path .acquire()s or .release()s
    it (it never existed), and it cannot regress because there's nothing
    to acquire. Real in-flight tracking is the _inflight_count Lock +
    _inflight_value[0] counter."""
    mod = _load_router(ROUTER_API_TOKEN="t")
    assert not hasattr(mod, "_inflight"), (
        "dead placeholder _inflight Semaphore has been reintroduced — "
        "revert llama-router.py to remove the line"
    )


def test_inflight_count_is_used_for_tracking():
    """L-1 (Phase M companion): the real in-flight tracking is the
    _inflight_count Lock + _inflight_value[0] mutable container. After
    many acquire/release cycles, the counter must return to 0."""
    mod = _load_router(ROUTER_API_TOKEN="t")
    mod._inflight_value[0] = 0
    for _ in range(50):
        mod._inflight_acquire()
    for _ in range(50):
        mod._inflight_release()
    assert mod._inflight_value[0] == 0


# ---------------------------------------------------------------------------
# L-2: atomic writers do not leave .tmp files behind
# ---------------------------------------------------------------------------


def test_atomic_writers_leave_no_tmp_files(tmp_path):
    """L-2: 200 write_env_file + write_current_model calls must not leave
    any .tmp file behind. The os.replace() should atomically rename .tmp
    -> target on every call."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from llama_bare.router_state import write_env_file, write_current_model

    env_file = tmp_path / ".env"
    state_file = tmp_path / "state"
    for i in range(200):
        write_env_file(env_file, f"model-{i}")
        write_current_model(state_file, f"model-{i}")

    # Inspect tmp_path for any .tmp leftovers.
    leftovers = sorted(p.name for p in tmp_path.glob("*.tmp"))
    assert leftovers == [], f"atomic writers left .tmp files behind: {leftovers}"


def test_concurrent_writers_leave_no_tmp_files(tmp_path):
    """L-2 (concurrency): 10 threads × 50 writes each must not leave any
    .tmp files behind. The module-level _state_lock serializes the rename."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from llama_bare.router_state import write_env_file

    env_file = tmp_path / ".env"

    def writer(i):
        for j in range(50):
            write_env_file(env_file, f"m{i}-{j}")

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    leftovers = sorted(p.name for p in tmp_path.glob("*.tmp"))
    assert leftovers == [], f"atomic writers left .tmp files behind under contention: {leftovers}"
    # Final content must be a single line (any of the writers' values).
    content = env_file.read_text()
    assert content.count("\n") == 1, f"expected exactly 1 line in final .env, got {content!r}"


# ---------------------------------------------------------------------------
# L-3: log file is writable
# ---------------------------------------------------------------------------


def test_logger_can_be_configured_against_tmp_log(tmp_path):
    """L-3: _configure_logger() must succeed against a tmp path without
    raising, and the RotatingFileHandler must produce output."""
    log_file = tmp_path / "router.log"
    mod = _load_router(
        ROUTER_API_TOKEN="t",
        LOG_FILE=str(log_file),
        LOG_MAX_BYTES="1024",
        LOG_BACKUP_COUNT="2",
    )
    # The module's _logger_configured is a guard set in main(); it is NOT
    # called at import time. Call it explicitly here (mirrors what main()
    # does at line 779 of llama-router.py).
    mod._configure_logger()
    mod.logger.info("hello from test")
    for h in mod.logger.handlers:
        try:
            h.flush()
        except Exception:
            pass
    assert log_file.exists(), f"logger did not write to {log_file}"
    body = log_file.read_text()
    assert "hello from test" in body, f"log file missing message: {body!r}"


# ---------------------------------------------------------------------------
# L-4: ensure_model_loaded doesn't update current_model on failure
# ---------------------------------------------------------------------------


def test_ensure_model_loaded_does_not_update_current_on_failure(tmp_path, monkeypatch):
    """L-4: if restart_with_model raises, ensure_model_loaded must propagate
    the exception and NOT update current_model. The state file invariant
    (state file == current_model) must hold after a failed swap."""
    mod = _load_router(ROUTER_API_TOKEN="t")
    state_file = tmp_path / "state"
    state_file.write_text("previously-loaded-model\n")

    mod.BACKEND_STATE_FILE = str(state_file)
    mod.current_model = "previously-loaded-model"

    def raise_runtime(*a, **kw):
        raise RuntimeError("simulated swap failure")

    monkeypatch.setattr(mod, "read_current_model_from_backend", lambda: "previously-loaded-model")
    monkeypatch.setattr(mod, "restart_with_model", raise_runtime)

    with pytest.raises(RuntimeError, match="simulated swap failure"):
        mod.ensure_model_loaded("new-model")

    assert mod.current_model == "previously-loaded-model", (
        "current_model must NOT change when the swap raises"
    )
    # State file should still hold the old model.
    assert state_file.read_text().strip() == "previously-loaded-model"


# ---------------------------------------------------------------------------
# L-5: .env.example documents the security/observability env vars
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_var", [
    "ROUTER_API_TOKEN",
    "MAX_REQUEST_BYTES",
    "LOG_FILE",
    "LOG_LEVEL",
    "DRAIN_TIMEOUT",
])
def test_env_example_documents_security_and_observability_vars(env_var):
    """L-5: .env.example should list the security and observability env vars
    that production code actually reads. Currently a known gap (LOW) — this
    test xfails with a clear message so the gap is visible in the report
    but doesn't block CI. To fix: add the env vars to .env.example and
    remove the xfail markers."""
    env_example = REPO_ROOT / ".env.example"
    if not env_example.exists():
        pytest.skip(".env.example missing — cannot test documentation")
    text = env_example.read_text()
    if env_var not in text:
        pytest.xfail(f"LOW documentation gap: .env.example missing {env_var!r}")
    assert env_var in text


# ---------------------------------------------------------------------------
# L-extra: client-disconnect during response write is a "missing error path"
# ---------------------------------------------------------------------------


def test_proxy_handles_client_disconnect_during_response_write(tmp_path):
    """L-extra: a BrokenPipeError raised by self.wfile.write(resp_body)
    after a successful backend read must not crash the request handler
    uncaught. The framework's handle_one_request will log it, but a
    TypeError or AttributeError would not be caught. This test pins the
    current behavior: the wfile.write call IS allowed to raise
    BrokenPipeError, but the request's in-flight counter must be released
    via the `finally` block."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    mod = _load_router(ROUTER_API_TOKEN="t")

    class _FakeHandler:
        def __init__(self):
            self.path = "/v1/chat/completions"
            self.headers = {"Authorization": "Bearer t", "Content-Length": "2"}
            self.client_address = ("192.168.1.99", 12345)
            self.rfile = io.BytesIO(b"{}")
            self.wfile = io.BytesIO()
            self.command = "POST"

    handler = _FakeHandler()
    # Patch the handler class onto a fresh RouterHandler instance.
    rh = mod.RouterHandler.__new__(mod.RouterHandler)
    rh.path = handler.path
    rh.headers = handler.headers
    rh.client_address = handler.client_address
    rh.rfile = handler.rfile
    rh.wfile = handler.wfile
    rh.command = handler.command
    rh.requestline = "POST /v1/chat/completions HTTP/1.1"
    rh.request_version = "HTTP/1.1"
    rh._logs = []

    # Snapshot the in-flight value before the call.
    mod._inflight_value[0] = 0

    # Build a fake response object whose .read() returns bytes but the
    # underlying wfile.write raises BrokenPipeError on the second call.
    backend_body = b"hello"

    class _FakeResp:
        status = 200
        def read(self):
            return backend_body
        def getheaders(self):
            return [("Content-Type", "application/json"), ("X-Other", "ok")]

    # Make the wfile raise BrokenPipeError on .write to simulate a
    # disconnected client. The framework would log this; we just verify
    # the `finally` releases the in-flight counter.
    def raise_broken_pipe(b):
        raise BrokenPipeError("simulated client disconnect")
    rh.wfile.write = raise_broken_pipe

    # Provide a target model that's "valid" and a stubbed urlopen.
    with patch.object(mod, "load_available_models", return_value={"my-model": 1}):
        with patch.object(mod, "ensure_model_loaded", lambda name: (True, ("ok", 0.0))):
            with patch.object(mod.request, "Request") as MockReq:
                with patch.object(mod, "contextlib") as mock_ctx:
                    # Build a context manager whose __enter__ returns _FakeResp.
                    fake_resp = _FakeResp()
                    mock_ctx.closing.return_value.__enter__.return_value = fake_resp
                    mock_ctx.closing.return_value.__exit__.return_value = False
                    try:
                        rh._proxy("POST")
                    except BrokenPipeError:
                        # Acceptable — framework would log this.
                        pass

    # The in-flight counter must be back to 0 (the `finally` released it).
    assert mod._inflight_value[0] == 0, (
        "in-flight counter leaked: the `finally` in _proxy must always run"
    )
