"""PHASE R: Session lock — pin a session to the model from its first request.

Prevents mid-session model swaps from killing in-flight tool-call iterations
in Hermes Agent / Open WebUI. The router reads X-Session-Id from the request,
pins the session to the model in the FIRST request, and uses the pinned model
for all subsequent requests with the same X-Session-Id (instead of triggering
a swap).

Tests cover the five required behaviors:
  1. No X-Session-Id header → existing swap behavior (no lock check).
  2. First request with X-Session-Id pins the session to the requested model.
  3. Follow-up request with same X-Session-Id + same model does NOT call
     ensure_model_loaded (it's a no-op for the already-loaded model).
  4. Follow-up request with same X-Session-Id but DIFFERENT model gets
     rewritten to the pinned model — no mid-session swap.
  5. The locks dict evicts old entries when it exceeds the cap (LRU/FIFO).
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Module loading: same approach as test_track_o_swap_during_request.py
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
ROUTER_PATH = REPO_ROOT / "llama-router.py"


def _load_router(**env_overrides):
    sys.modules.pop("llama_router_loaded", None)
    for key, value in env_overrides.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    spec = importlib.util.spec_from_file_location("llama_router_loaded", ROUTER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Test harness: minimal stand-in for BaseHTTPRequestHandler's per-request
# state, matching the structure in test_track_o_swap_during_request.py.
# ---------------------------------------------------------------------------


class _FakeRequest:
    def __init__(self, *, method="POST", path="/v1/chat/completions", headers=None,
                 body=b"", query_string="", client_address=("127.0.0.1", 0)):
        self.headers = dict(headers or {})
        self.path = path
        self.command = method
        self.request_version = "HTTP/1.1"
        self.requestline = f"{method} {path} HTTP/1.1"
        self.client_address = client_address
        self.rfile = io.BytesIO(body)
        self.wfile = io.BytesIO()
        self._body_buf = body
        self._query_string = query_string
        self._logs = []

    def address_string(self):
        return "127.0.0.1"

    def log_message(self, format, *args):
        self._logs.append(format % args)

    def send_error(self, code, message=None, explain=None):
        raise AssertionError(f"send_error({code}, {message!r}, {explain!r}) was called")


def _build_handler(mod, request: _FakeRequest) -> "mod.RouterHandler":
    handler = mod.RouterHandler.__new__(mod.RouterHandler)
    handler.headers = request.headers
    handler.path = request.path
    handler.command = request.command
    handler.request_version = request.request_version
    handler.requestline = request.requestline
    handler.client_address = request.client_address
    handler.rfile = request.rfile
    handler.wfile = request.wfile
    handler.log_message = request.log_message
    handler.send_error = request.send_error
    handler.address_string = request.address_string
    return handler


def _read_json_response(wfile: io.BytesIO):
    """Parse the BytesIO as an HTTP response — returns (code, payload, headers)."""
    raw = wfile.getvalue()
    if not raw:
        return None, b"", {}
    head, _, body = raw.partition(b"\r\n\r\n")
    header_lines = head.split(b"\r\n")
    status_line = header_lines[0]
    parts = status_line.split(b" ", 2)
    code = int(parts[1])
    headers = {}
    for line in header_lines[1:]:
        if b":" in line:
            k, v = line.split(b":", 1)
            headers[k.strip().lower().decode()] = v.strip().decode()
    payload = json.loads(body) if body else None
    return code, payload, headers


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def router_with_token(monkeypatch):
    monkeypatch.setenv("ROUTER_API_TOKEN", "phase-r-test-token-1234567890")
    mod = _load_router(ROUTER_API_TOKEN="phase-r-test-token-1234567890")
    # Clear any leftover state from prior tests.
    mod._session_locks.clear()
    return mod


# ===========================================================================
# Test 1: No X-Session-Id header → ensure_model_loaded still called
# ===========================================================================


def test_no_session_header_skips_lock(router_with_token, monkeypatch):
    """PHASE R: clients that do NOT send X-Session-Id must get the existing
    behavior — ensure_model_loaded is called normally. The session-lock
    machinery must be a no-op for them."""
    ensure_calls = []

    def fake_ensure(model_name):
        ensure_calls.append(model_name)
        return (True, ("already_loaded", 0.0))

    monkeypatch.setattr(router_with_token, "ensure_model_loaded", fake_ensure)
    monkeypatch.setattr(router_with_token, "load_available_models", lambda: {"muse", "qwen"})
    # Avoid hitting a real backend.
    monkeypatch.setattr(
        router_with_token.request, "urlopen",
        lambda req, timeout=None: _ok_response(),
    )

    body = b'{"model":"muse","messages":[]}'
    headers = {
        "Authorization": f"Bearer {router_with_token.API_TOKEN}",
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
    }
    # Note: NO X-Session-Id header.
    req = _FakeRequest(method="POST", path="/v1/chat/completions", headers=headers, body=body)
    handler = _build_handler(router_with_token, req)
    handler.do_POST()

    code, _, _ = _read_json_response(req.wfile)
    assert code == 200, f"expected 200, got {code}"
    assert ensure_calls == ["muse"], f"expected ensure_model_loaded('muse'), got {ensure_calls}"
    # No lock should have been recorded.
    assert "any-session" not in router_with_token._session_locks
    assert len(router_with_token._session_locks) == 0


# ===========================================================================
# Test 2: First request with X-Session-Id pins the session
# ===========================================================================


def test_session_first_request_pins_model(router_with_token, monkeypatch):
    """PHASE R: the first request with a given X-Session-Id must pin the
    session to the requested model. ensure_model_loaded is called once
    (with the requested model), and the lock dict records the pin."""
    ensure_calls = []

    def fake_ensure(model_name):
        ensure_calls.append(model_name)
        return (True, ("already_loaded", 0.0))

    monkeypatch.setattr(router_with_token, "ensure_model_loaded", fake_ensure)
    monkeypatch.setattr(router_with_token, "load_available_models", lambda: {"muse", "qwen"})
    monkeypatch.setattr(
        router_with_token.request, "urlopen",
        lambda req, timeout=None: _ok_response(),
    )

    body = b'{"model":"muse","messages":[]}'
    headers = {
        "Authorization": f"Bearer {router_with_token.API_TOKEN}",
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
        "X-Session-Id": "session-A",
    }
    req = _FakeRequest(method="POST", path="/v1/chat/completions", headers=headers, body=body)
    handler = _build_handler(router_with_token, req)
    handler.do_POST()

    code, _, _ = _read_json_response(req.wfile)
    assert code == 200
    assert ensure_calls == ["muse"]
    assert router_with_token._session_locks.get("session-A") == "muse", (
        f"expected session-A pinned to 'muse', got {router_with_token._session_locks!r}"
    )


# ===========================================================================
# Test 3: Follow-up request with same X-Session-Id + same model → no swap check
# ===========================================================================


def test_session_followup_no_swap(router_with_token, monkeypatch):
    """PHASE R: a follow-up request with the same X-Session-Id AND same model
    must call ensure_model_loaded (because the model might not still be loaded
    if some other session took over) — the lock just prevents a MID-SESSION
    swap. This test pins the session first, then sends a follow-up."""
    ensure_calls = []

    def fake_ensure(model_name):
        ensure_calls.append(model_name)
        return (True, ("already_loaded", 0.0))

    monkeypatch.setattr(router_with_token, "ensure_model_loaded", fake_ensure)
    monkeypatch.setattr(router_with_token, "load_available_models", lambda: {"muse", "qwen"})
    monkeypatch.setattr(
        router_with_token.request, "urlopen",
        lambda req, timeout=None: _ok_response(),
    )

    # Pre-pin the session by sending one request first.
    body = b'{"model":"muse","messages":[]}'
    headers = {
        "Authorization": f"Bearer {router_with_token.API_TOKEN}",
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
        "X-Session-Id": "session-B",
    }
    req1 = _FakeRequest(method="POST", path="/v1/chat/completions", headers=headers, body=body)
    handler1 = _build_handler(router_with_token, req1)
    handler1.do_POST()
    assert router_with_token._session_locks.get("session-B") == "muse"

    # Now a follow-up with the same session + same model.
    req2 = _FakeRequest(method="POST", path="/v1/chat/completions", headers=headers, body=body)
    handler2 = _build_handler(router_with_token, req2)
    handler2.do_POST()
    code, _, _ = _read_json_response(req2.wfile)
    assert code == 200
    # ensure_model_loaded must have been called for the pinned model on both requests.
    assert ensure_calls == ["muse", "muse"], (
        f"expected ['muse','muse'], got {ensure_calls!r}"
    )


# ===========================================================================
# Test 4: Follow-up with same session but DIFFERENT model → pinned model wins
# ===========================================================================


def test_session_lock_overrides_model_mismatch(router_with_token, monkeypatch):
    """PHASE R: when a session is pinned to 'muse' and a follow-up request
    asks for 'qwen', the router must NOT swap — it must rewrite target_model
    to 'muse' and call ensure_model_loaded('muse') instead."""
    ensure_calls = []

    def fake_ensure(model_name):
        ensure_calls.append(model_name)
        return (True, ("already_loaded", 0.0))

    monkeypatch.setattr(router_with_token, "ensure_model_loaded", fake_ensure)
    monkeypatch.setattr(router_with_token, "load_available_models", lambda: {"muse", "qwen"})
    monkeypatch.setattr(
        router_with_token.request, "urlopen",
        lambda req, timeout=None: _ok_response(),
    )

    # First request: pin session to 'muse'.
    body1 = b'{"model":"muse","messages":[]}'
    headers1 = {
        "Authorization": f"Bearer {router_with_token.API_TOKEN}",
        "Content-Type": "application/json",
        "Content-Length": str(len(body1)),
        "X-Session-Id": "session-C",
    }
    req1 = _FakeRequest(method="POST", path="/v1/chat/completions", headers=headers1, body=body1)
    handler1 = _build_handler(router_with_token, req1)
    handler1.do_POST()
    assert router_with_token._session_locks.get("session-C") == "muse"

    # Second request: same session, different model. Must NOT trigger a swap.
    body2 = b'{"model":"qwen","messages":[]}'
    headers2 = {
        "Authorization": f"Bearer {router_with_token.API_TOKEN}",
        "Content-Type": "application/json",
        "Content-Length": str(len(body2)),
        "X-Session-Id": "session-C",
    }
    req2 = _FakeRequest(method="POST", path="/v1/chat/completions", headers=headers2, body=body2)
    handler2 = _build_handler(router_with_token, req2)
    handler2.do_POST()
    code, _, _ = _read_json_response(req2.wfile)
    assert code == 200
    # The mismatch must have been rewritten to the pinned model — no qwen call.
    assert ensure_calls == ["muse", "muse"], (
        f"expected ['muse','muse'] (no qwen!), got {ensure_calls!r}"
    )


# ===========================================================================
# Test 5: LRU/FIFO eviction when the locks dict exceeds the cap
# ===========================================================================


def test_session_lru_eviction(router_with_token, monkeypatch):
    """PHASE R: when the locks dict exceeds the cap, older entries must be
    evicted. Verify by populating beyond the cap and checking that the
    oldest entries are no longer present."""
    # Find the cap from the module (the implementation may name it differently;
    # accept any of the obvious names so we don't over-couple the test).
    cap = None
    for name in ("_SESSION_LOCKS_MAX", "SESSION_LOCKS_MAX", "_SESSION_LOCKS_CAP"):
        if hasattr(router_with_token, name):
            cap = getattr(router_with_token, name)
            break
    assert cap is not None and isinstance(cap, int) and cap > 0, (
        f"expected a positive int cap constant on the module, got {cap!r}"
    )

    # Stuff the dict past capacity.
    for i in range(cap + 1):
        router_with_token._session_locks[f"session-{i}"] = "muse"

    # Send one more request with a NEW session-id — this should trigger eviction.
    def fake_ensure(model_name):
        return (True, ("already_loaded", 0.0))

    monkeypatch.setattr(router_with_token, "ensure_model_loaded", fake_ensure)
    monkeypatch.setattr(router_with_token, "load_available_models", lambda: {"muse"})
    monkeypatch.setattr(
        router_with_token.request, "urlopen",
        lambda req, timeout=None: _ok_response(),
    )

    body = b'{"model":"muse","messages":[]}'
    headers = {
        "Authorization": f"Bearer {router_with_token.API_TOKEN}",
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
        "X-Session-Id": "session-newest",
    }
    req = _FakeRequest(method="POST", path="/v1/chat/completions", headers=headers, body=body)
    handler = _build_handler(router_with_token, req)
    handler.do_POST()
    code, _, _ = _read_json_response(req.wfile)
    assert code == 200

    # The dict must not have grown unboundedly.
    assert len(router_with_token._session_locks) <= cap, (
        f"locks dict size {len(router_with_token._session_locks)} exceeds cap {cap}"
    )
    # And the oldest entry (session-0) must have been evicted.
    assert "session-0" not in router_with_token._session_locks, (
        f"expected oldest entry to be evicted; locks={list(router_with_token._session_locks)!r}"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeUrlOpenResponse:
    def __init__(self, body=b'{"ok":true}', status=200):
        self._body = body
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self._body

    def getheaders(self):
        return [("Content-Type", "application/json")]

    def close(self):
        pass


def _ok_response():
    return _FakeUrlOpenResponse(b'{"ok":true}', 200)
