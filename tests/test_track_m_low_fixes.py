"""PHASE M — regression tests for the 5 LOW findings from Phase L.

Findings covered:
- L-1: dead `threading.Semaphore(1)` placeholder is gone from llama-router.py
- L-2: .env.example documents every env var the code reads
- L-3: restart_with_model uses exponential backoff (reset on success,
       capped at SWAP_BACKOFF_MAX, gates ensure_model_loaded)
- L-4: wfile.write() during response/JSON write swallows BrokenPipeError
       and ConnectionResetError (client-disconnect safety)
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

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
    spec = importlib.util.spec_from_file_location(
        "llama_router_loaded", REPO_ROOT / "llama-router.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# L-1: dead Semaphore placeholder is gone
# ---------------------------------------------------------------------------


def test_no_dead_semaphore_in_router_module():
    """L-1: the placeholder `threading.Semaphore(1)` at the top of
    llama-router.py is dead code — no path acquires or releases it.
    Phase M removed the line entirely. This test pins the new contract:
    the module no longer has `_inflight`."""
    mod = _load_router(ROUTER_API_TOKEN="t")
    assert not hasattr(mod, "_inflight"), (
        "dead `_inflight` Semaphore placeholder has been reintroduced; "
        "delete the line in llama-router.py"
    )


# ---------------------------------------------------------------------------
# L-2: .env.example documents every env var the code reads
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_var", [
    "ROUTER_API_TOKEN",
    "MAX_REQUEST_BYTES",
    "LOG_FILE",
    "LOG_LEVEL",
    "DRAIN_TIMEOUT",
])
def test_env_example_documents_var(env_var):
    """L-2: .env.example should list each env var the router code reads.
    Phase M rewrote .env.example to document them all with defaults."""
    env_example = REPO_ROOT / ".env.example"
    assert env_example.exists(), f".env.example missing at {env_example}"
    text = env_example.read_text()
    assert env_var in text, (
        f".env.example missing documentation for {env_var!r}; "
        f"add a line for this variable with its default and a comment"
    )


# ---------------------------------------------------------------------------
# L-3: exponential backoff on restart_with_model
# ---------------------------------------------------------------------------


def test_backoff_resets_on_successful_swap(monkeypatch):
    """L-3: after 2 failed swaps, a successful swap must reset the
    consecutive-failure counter to 0 and clear the next-allowed time."""
    mod = _load_router(ROUTER_API_TOKEN="t")
    # Reset module-level state.
    mod._swap_failure_count = 0
    mod._next_swap_allowed_at = 0.0

    # Stub write_env_file + wait_for_health BEFORE the failure loop so
    # restart_with_model never touches the real filesystem or HTTP probe.
    monkeypatch.setattr(mod, "write_env_file", lambda path, name: None)
    monkeypatch.setattr(mod, "wait_for_health", lambda timeout=120: True)

    # Track which swap ATTEMPT we are on (1 or 2 = fail; 3 = succeed).
    # restart_with_model may invoke subprocess.run 1x (primary rc=0 path)
    # or 2x (primary rc!=0, then sudo fallback). Both must fail for the
    # swap attempt to raise RuntimeError. We use a call_counter that
    # fails for swap attempts 1 and 2, succeeds for swap attempt 3.
    swap_attempt = [0]
    call_counter = [0]

    class _FakeResult:
        def __init__(self, rc):
            self.returncode = rc
            self.stderr = ""
            self.stdout = ""

    def fake_run(*a, **kw):
        call_counter[0] += 1
        # Fail first 4 subprocess calls (covers 2 swap attempts x up to 2 calls each).
        if call_counter[0] <= 4:
            return _FakeResult(1)
        return _FakeResult(0)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    # 2 failed swap attempts.
    for _ in range(2):
        with pytest.raises(RuntimeError):
            mod.restart_with_model("failing-model")

    assert mod._swap_failure_count == 2, (
        f"expected 2 failures, got {mod._swap_failure_count}"
    )
    assert mod._next_swap_allowed_at > time.monotonic(), (
        "backoff window must be set after failures"
    )

    # 3rd swap attempt: subprocess.run now returns rc=0 and wait_for_health
    # returns True, so the swap succeeds and resets the backoff state.
    mod.restart_with_model("good-model")  # must not raise

    assert mod._swap_failure_count == 0, (
        f"successful swap must reset _swap_failure_count to 0; "
        f"got {mod._swap_failure_count}"
    )
    assert mod._next_swap_allowed_at == 0.0, (
        f"successful swap must clear _next_swap_allowed_at; "
        f"got {mod._next_swap_allowed_at}"
    )


def test_backoff_caps_at_max(monkeypatch):
    """L-3: after many consecutive failures, the backoff delay must
    cap at SWAP_BACKOFF_MAX (60s), not grow without bound."""
    mod = _load_router(ROUTER_API_TOKEN="t")
    mod._swap_failure_count = 0
    mod._next_swap_allowed_at = 0.0

    # Stub write_env_file + subprocess.run so we don't actually call
    # systemctl (which would hang waiting on systemd).
    monkeypatch.setattr(mod, "write_env_file", lambda path, name: None)

    class _FakeResult:
        returncode = 1  # non-zero so the "else: if result.returncode != 0"
                        # branch raises RuntimeError on the next line.
        stderr = "stub-failure"
        stdout = ""

    # All subprocess.run calls (including fallback stop/start) go through
    # this stub; the function reads `returncode` and raises on != 0.
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **kw: _FakeResult())

    # 10 failures is more than enough to exceed 2**n > SWAP_BACKOFF_MAX
    # at the SWAP_BACKOFF_INITIAL=2.0 doubling rate (2*2^9 = 1024 > 60).
    for _ in range(10):
        with pytest.raises(RuntimeError):
            mod.restart_with_model("failing-model")

    assert mod._swap_failure_count == 10
    expected_max = mod.SWAP_BACKOFF_MAX  # 60.0
    # The _next_swap_allowed_at is the monotonic timestamp when the next
    # swap is allowed; the delay is _next_swap_allowed_at - now.
    delay = mod._next_swap_allowed_at - time.monotonic()
    # Allow a small slack for clock skew / test execution time.
    assert delay <= expected_max + 0.5, (
        f"backoff {delay:.1f}s exceeds SWAP_BACKOFF_MAX ({expected_max}s)"
    )


def test_backoff_gate_rejects_swap_during_cooldown():
    """L-3: ensure_model_loaded must return (False, ("in_backoff", wait))
    while a backoff cooldown is active, so the proxy can return 503 to
    the client instead of hammering a broken backend."""
    mod = _load_router(ROUTER_API_TOKEN="t")
    mod._swap_failure_count = 0
    mod._next_swap_allowed_at = 0.0

    # Force a backoff window in the future.
    mod._next_swap_allowed_at = time.monotonic() + 30.0

    # ensure_model_loaded must short-circuit and return a backoff tuple.
    ok, info = mod.ensure_model_loaded("any-model")
    assert ok is False, "ensure_model_loaded must return False during backoff"
    assert info[0] == "in_backoff"
    assert isinstance(info[1], float)
    assert info[1] > 0, f"backoff wait must be positive, got {info[1]}"


def test_circuit_breaker_trips_after_threshold(monkeypatch):
    """E2: after SWAP_CIRCUIT_TRIP_AFTER consecutive failures, the circuit
    breaker must trip and ensure_model_loaded must short-circuit with a
    'circuit_open' reason instead of attempting another swap. This prevents
    the router from hammering a fundamentally broken backend indefinitely."""
    # Set the trip threshold low (3) so the test is fast.
    mod = _load_router(
        ROUTER_API_TOKEN="t",
        SWAP_CIRCUIT_TRIP_AFTER="3",
        SWAP_CIRCUIT_OPEN_SECONDS="60",
    )
    mod._swap_failure_count = 0
    mod._next_swap_allowed_at = 0.0
    mod._circuit_open_until = 0.0

    monkeypatch.setattr(mod, "write_env_file", lambda path, name: None)

    class _FakeResult:
        returncode = 1
        stderr = "stub"
        stdout = ""

    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **kw: _FakeResult())

    # 3 failures must trip the circuit.
    for _ in range(3):
        with pytest.raises(RuntimeError):
            mod.restart_with_model("failing-model")

    assert mod._swap_failure_count == 3
    assert mod._circuit_open_until > time.monotonic(), (
        "circuit_open_until must be set in the future after trip"
    )

    # ensure_model_loaded must short-circuit with 'circuit_open'.
    ok, info = mod.ensure_model_loaded("any-model")
    assert ok is False
    assert info[0] == "circuit_open", (
        f"expected 'circuit_open', got {info[0]!r}"
    )
    assert info[1] > 0, f"circuit-open wait must be positive, got {info[1]}"


def test_circuit_breaker_resets_on_success(monkeypatch):
    """E2: a successful swap must reset the circuit breaker, not just the
    backoff counter. Without this, the breaker would stay tripped forever
    after a single transient failure sequence."""
    mod = _load_router(
        ROUTER_API_TOKEN="t",
        SWAP_CIRCUIT_TRIP_AFTER="3",
        SWAP_CIRCUIT_OPEN_SECONDS="60",
    )
    mod._swap_failure_count = 0
    mod._next_swap_allowed_at = 0.0
    mod._circuit_open_until = 0.0

    monkeypatch.setattr(mod, "write_env_file", lambda path, name: None)
    monkeypatch.setattr(mod, "wait_for_health", lambda timeout=120: True)

    # First failed swap.
    class _FakeResult:
        def __init__(self, rc):
            self.returncode = rc
            self.stderr = ""
            self.stdout = ""

    calls = [0]

    def fake_run(*a, **kw):
        calls[0] += 1
        return _FakeResult(1)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError):
        mod.restart_with_model("failing-model")

    assert mod._swap_failure_count == 1
    assert mod._circuit_open_until == 0.0, (
        "circuit must NOT trip on a single failure"
    )

    # Now make subprocess succeed.
    def fake_run_success(*a, **kw):
        return _FakeResult(0)
    monkeypatch.setattr(mod.subprocess, "run", fake_run_success)

    mod.restart_with_model("good-model")  # must not raise

    assert mod._swap_failure_count == 0
    assert mod._circuit_open_until == 0.0, (
        "successful swap must reset the circuit breaker"
    )


def test_backoff_gate_clears_after_cooldown():
    """L-3: once the cooldown elapses, ensure_model_loaded must
    return True again (subject to the swap succeeding)."""
    mod = _load_router(ROUTER_API_TOKEN="t")
    mod._swap_failure_count = 1
    mod._next_swap_allowed_at = 0.0  # already elapsed

    # Stub the heavy lifting: model not loaded, swap succeeds.
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(mod, "read_current_model_from_backend", lambda: "other-model")
        monkeypatch.setattr(mod, "restart_with_model", lambda name: None)
        ok, info = mod.ensure_model_loaded("m")
    finally:
        monkeypatch.undo()

    assert ok is True, "ensure_model_loaded must return True when backoff has elapsed"
    assert info[0] == "swapped"


# ---------------------------------------------------------------------------
# L-4: BrokenPipeError / ConnectionResetError handled gracefully
# ---------------------------------------------------------------------------


def test_send_json_catches_broken_pipe(monkeypatch):
    """L-4: _send_json must swallow BrokenPipeError raised by wfile.write
    (client disconnected mid-response) without re-raising."""
    mod = _load_router(ROUTER_API_TOKEN="t")

    class _FakeHandler:
        def __init__(self):
            self.path = "/health"
            self.headers = {}
            self.command = "GET"
            self.client_address = ("127.0.0.1", 12345)
            self.request_version = "HTTP/1.1"
            # wfile raises BrokenPipeError on .write().
            self._buf = io.BytesIO()
            self._broken = False

            class _Wfile:
                def __init__(self, outer):
                    self._outer = outer

                def write(self, data):
                    if not self._outer._broken:
                        self._outer._broken = True
                        raise BrokenPipeError("simulated client disconnect")
                    return len(data)

            self.wfile = _Wfile(self)
            self._logs = []

        # The handler uses self.send_response / send_header / end_headers.
        def send_response(self, *a, **kw):
            pass

        def send_header(self, *a, **kw):
            pass

        def end_headers(self, *a, **kw):
            pass

    rh = _FakeHandler()
    # Call the bound method directly. Must NOT raise.
    mod.RouterHandler._send_json(rh, 200, {"ok": True})


def test_send_json_catches_connection_reset(monkeypatch):
    """L-4: _send_json must also swallow ConnectionResetError."""
    mod = _load_router(ROUTER_API_TOKEN="t")

    class _FakeHandler:
        def __init__(self):
            self._broken = False

            class _Wfile:
                def __init__(self, outer):
                    self._outer = outer

                def write(self, data):
                    if not self._outer._broken:
                        self._outer._broken = True
                        raise ConnectionResetError("simulated RST")
                    return len(data)

            self.wfile = _Wfile(self)

        def send_response(self, *a, **kw):
            pass

        def send_header(self, *a, **kw):
            pass

        def end_headers(self, *a, **kw):
            pass

    rh = _FakeHandler()
    mod.RouterHandler._send_json(rh, 500, {"error": "x"})


def test_proxy_catches_broken_pipe_on_response_write(monkeypatch):
    """L-4: the proxy path (after a successful backend read) must catch
    BrokenPipeError from wfile.write(resp_body) and not propagate it.
    The in-flight counter must still be released via the `finally`."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    mod = _load_router(ROUTER_API_TOKEN="t")

    class _FakeHandler:
        def __init__(self):
            self.path = "/v1/chat/completions"
            self.headers = {"Authorization": "Bearer t", "Content-Length": "2"}
            self.client_address = ("192.168.1.99", 12345)
            self.rfile = io.BytesIO(b"{}")
            self.command = "POST"
            self.requestline = "POST /v1/chat/completions HTTP/1.1"
            self.request_version = "HTTP/1.1"
            # Track call counts to differentiate header writes from body writes.
            self._writes = 0
            self._broken = False
            # Buffer to capture output for any writes after the simulated break.
            self._buf = io.BytesIO()

            class _Wfile:
                def __init__(self, outer):
                    self._outer = outer

                def write(self, data):
                    self._outer._writes += 1
                    if not self._outer._broken:
                        self._outer._broken = True
                        raise BrokenPipeError("simulated client disconnect")
                    # After the simulated break, swallow further writes.
                    return len(data)

            self.wfile = _Wfile(self)
            self._logs = []

        # Header methods are no-ops — we only care about wfile.write.
        def send_response(self, *a, **kw):
            pass

        def send_header(self, *a, **kw):
            pass

        def end_headers(self, *a, **kw):
            pass

    handler = _FakeHandler()
    rh = mod.RouterHandler.__new__(mod.RouterHandler)
    rh.__dict__.update(handler.__dict__)
    # Replace header methods on the instance — `__dict__.update` does not
    # override methods already bound on the parent class.
    rh.send_response = handler.send_response
    rh.send_header = handler.send_header
    rh.end_headers = handler.end_headers

    mod._inflight_value[0] = 0

    # Fake backend response.
    backend_body = b"hello"

    class _FakeResp:
        status = 200

        def read(self):
            return backend_body

        def getheaders(self):
            return [("Content-Type", "application/json")]

    fake_resp = _FakeResp()

    with patch.object(mod, "load_available_models", return_value={"m": 1}):
        with patch.object(mod, "ensure_model_loaded", lambda name: (True, ("ok", 0.0))):
            with patch.object(mod, "contextlib") as mock_ctx:
                mock_ctx.closing.return_value.__enter__.return_value = fake_resp
                mock_ctx.closing.return_value.__exit__.return_value = False
                # Must not raise even though wfile.write raises BrokenPipeError.
                rh._proxy("POST")

    # In-flight counter must be back to 0 (the `finally` released it).
    assert mod._inflight_value[0] == 0, (
        "in-flight counter leaked: BrokenPipeError must not bypass the `finally`"
    )
