"""PHASE O: Backend stream cut mid-response (model swap during request).

Covers the user-facing bug where an in-flight request is killed by its own
model swap, surfacing as HTTP 500 with a leaked traceback. The fix is to
catch IncompleteRead / ConnectionError / BrokenPipeError in _proxy and
return a sanitized 503 + Retry-After, plus add a SwapInProgressError path
so concurrent swaps don't double-restart the backend.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest


# ---------------------------------------------------------------------------
# Module loading: same approach as test_router_security.py
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
# state, matching the structure in test_router_security.py.
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
    """Parse the BytesIO as an HTTP response — returns (code, payload, headers).
    headers is a dict of header-name -> value (lowercased keys).
    """
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
    monkeypatch.setenv("ROUTER_API_TOKEN", "phase-o-test-token-1234567890")
    mod = _load_router(ROUTER_API_TOKEN="phase-o-test-token-1234567890")
    return mod


# ===========================================================================
# Test 1: IncompleteRead from urlopen → 503 (not 500)
# ===========================================================================


def test_proxy_returns_503_on_incomplete_read(router_with_token, monkeypatch):
    """PHASE O F1: when the backend cuts the response stream mid-stream
    (http.client.IncompleteRead), the proxy must NOT surface HTTP 500 with a
    leaked traceback. It must return a sanitized 503 + Retry-After."""
    import http.client

    def fake_urlopen(req, timeout=None):
        # Simulate the backend dying mid-stream — the model was just swapped.
        raise http.client.IncompleteRead(b'{"partial":', 16)

    monkeypatch.setattr(router_with_token.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(router_with_token, "ensure_model_loaded", lambda m: (True, ("ok", 0.0)))
    monkeypatch.setattr(router_with_token, "load_available_models", lambda: {"m"})

    body = b'{"model":"m","messages":[]}'
    headers = {
        "Authorization": f"Bearer {router_with_token.API_TOKEN}",
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
    }
    req = _FakeRequest(method="POST", path="/v1/chat/completions", headers=headers, body=body)
    handler = _build_handler(router_with_token, req)
    handler.do_POST()
    code, payload, hdrs = _read_json_response(req.wfile)
    assert code == 503, f"expected 503, got {code}"
    assert payload is not None
    assert "request_id" in payload
    assert "backend" in payload["error"].lower() or "stream" in payload["error"].lower()
    assert hdrs.get("retry-after") == "5"


# ===========================================================================
# Test 2: ConnectionResetError from urlopen → 503
# ===========================================================================


def test_proxy_returns_503_on_connection_reset(router_with_token, monkeypatch):
    """PHASE O F1: ConnectionResetError (peer closed the socket unexpectedly)
    must also return 503 + Retry-After, not 500."""
    def fake_urlopen(req, timeout=None):
        raise ConnectionResetError("backend peer closed socket")

    monkeypatch.setattr(router_with_token.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(router_with_token, "ensure_model_loaded", lambda m: (True, ("ok", 0.0)))
    monkeypatch.setattr(router_with_token, "load_available_models", lambda: {"m"})

    body = b'{"model":"m","messages":[]}'
    headers = {
        "Authorization": f"Bearer {router_with_token.API_TOKEN}",
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
    }
    req = _FakeRequest(method="POST", path="/v1/chat/completions", headers=headers, body=body)
    handler = _build_handler(router_with_token, req)
    handler.do_POST()
    code, payload, hdrs = _read_json_response(req.wfile)
    assert code == 503, f"expected 503, got {code}"
    assert "request_id" in payload
    assert hdrs.get("retry-after") == "5"


# ===========================================================================
# Test 3: BrokenPipeError from urlopen → 503
# ===========================================================================


def test_proxy_returns_503_on_broken_pipe(router_with_token, monkeypatch):
    """PHASE O F1: BrokenPipeError (writing to a closed socket) from urlopen
    must also return 503 + Retry-After, not 500."""
    def fake_urlopen(req, timeout=None):
        raise BrokenPipeError("write to closed pipe")

    monkeypatch.setattr(router_with_token.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(router_with_token, "ensure_model_loaded", lambda m: (True, ("ok", 0.0)))
    monkeypatch.setattr(router_with_token, "load_available_models", lambda: {"m"})

    body = b'{"model":"m","messages":[]}'
    headers = {
        "Authorization": f"Bearer {router_with_token.API_TOKEN}",
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
    }
    req = _FakeRequest(method="POST", path="/v1/chat/completions", headers=headers, body=body)
    handler = _build_handler(router_with_token, req)
    handler.do_POST()
    code, payload, hdrs = _read_json_response(req.wfile)
    assert code == 503, f"expected 503, got {code}"
    assert "request_id" in payload
    assert hdrs.get("retry-after") == "5"


# ===========================================================================
# Test 4: ensure_model_loaded raises SwapInProgressError → 503
# ===========================================================================


def test_proxy_returns_503_when_swap_in_progress(router_with_token, monkeypatch):
    """PHASE O F2: when another model swap is already in progress, the proxy
    must not block on it nor crash — return 503 + Retry-After immediately."""
    def raise_swap_in_progress(m):
        # The SwapInProgressError class must exist on the module.
        raise router_with_token.SwapInProgressError(
            f"another swap to {m!r} is in progress"
        )

    monkeypatch.setattr(router_with_token, "ensure_model_loaded", raise_swap_in_progress)
    monkeypatch.setattr(router_with_token, "load_available_models", lambda: {"m"})

    body = b'{"model":"m","messages":[]}'
    headers = {
        "Authorization": f"Bearer {router_with_token.API_TOKEN}",
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
    }
    req = _FakeRequest(method="POST", path="/v1/chat/completions", headers=headers, body=body)
    handler = _build_handler(router_with_token, req)
    handler.do_POST()
    code, payload, hdrs = _read_json_response(req.wfile)
    assert code == 503, f"expected 503, got {code}"
    assert "request_id" in payload
    assert hdrs.get("retry-after") == "5"
    assert "swap" in payload["error"].lower()


# ===========================================================================
# Test 5: 503 body does NOT leak internals (Traceback, type names, paths)
# ===========================================================================


def test_proxy_sanitizes_503_error_message(router_with_token, monkeypatch):
    """PHASE O F1: the 503 body must NEVER leak tracebacks, exception class
    names, file paths, or any other internal state — same A4 invariant as
    the existing 500 path."""
    import http.client

    def fake_urlopen(req, timeout=None):
        # Raise with a value that includes words that should NOT appear in
        # the client-facing 503 body.
        raise http.client.IncompleteRead(b"", 0)

    monkeypatch.setattr(router_with_token.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(router_with_token, "ensure_model_loaded", lambda m: (True, ("ok", 0.0)))
    monkeypatch.setattr(router_with_token, "load_available_models", lambda: {"m"})

    body = b'{"model":"m","messages":[]}'
    headers = {
        "Authorization": f"Bearer {router_with_token.API_TOKEN}",
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
    }
    req = _FakeRequest(method="POST", path="/v1/chat/completions", headers=headers, body=body)
    handler = _build_handler(router_with_token, req)
    handler.do_POST()
    code, payload, hdrs = _read_json_response(req.wfile)

    # The body, serialized as JSON, must not contain these strings.
    assert code == 503
    assert payload is not None
    body_str = json.dumps(payload)
    forbidden = [
        "Traceback",
        "IncompleteRead",
        "ConnectionResetError",
        "BrokenPipeError",
        "/home/fekry",  # any local path
        "llama-router.py",  # any production file path
        "_proxy",  # function name leak
    ]
    for needle in forbidden:
        assert needle not in body_str, (
            f"503 body leaks internal state: contains {needle!r}\n"
            f"body: {body_str!r}"
        )
    # Sanity: the body IS a small, sanitized error envelope.
    assert "request_id" in payload
    assert "error" in payload