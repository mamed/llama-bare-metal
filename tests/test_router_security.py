"""Security tests for llama-router.py.

PHASE A coverage:
- A1: Body size limit (MAX_REQUEST_BYTES)
- A2: Header whitelist on proxy + X-Router-Forwarded injection
- A3: Shared-secret auth via ROUTER_API_TOKEN
- A4: Error message sanitization (no path leak to client, full to stderr)

We exercise RouterHandler by giving it a stub BaseHTTPRequestHandler instance
whose rfile / wfile / headers are in-memory buffers we control. The handler's
methods are called directly — no real socket is opened, so tests are fast
and don't bind a port.
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
# Module loading: llama-router.py is a script, not a package, so we load it
# by file path. We also force a fresh import per test (parametrized reload)
# so monkeypatched env vars take effect on the module-level constants.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
ROUTER_PATH = REPO_ROOT / "llama-router.py"


def _load_router(**env_overrides):
    """Import llama-router.py with the given env vars set before import.
    Reuses sys.modules cache only if no env overrides requested.
    """
    # Strip cached module if it was loaded before, so module-level constants
    # are re-evaluated against the new env.
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
# Test harness: build a RouterHandler instance backed by in-memory streams.
# We bypass __init__ (which expects a real socket) and set the attributes the
# handler reads from directly.
# ---------------------------------------------------------------------------


class _FakeRequest:
    """Minimal stand-in for BaseHTTPRequestHandler's per-request state."""

    def __init__(self, *, method="POST", path="/v1/chat/completions", headers=None,
                 body=b"", query_string="", client_address=("127.0.0.1", 0)):
        # self.headers is an .items()-iterable mapping.
        self.headers = dict(headers or {})
        self.path = path
        self.command = method
        self.request_version = "HTTP/1.1"
        # requestline is what send_response → log_request reads.
        self.requestline = f"{method} {path} HTTP/1.1"
        self.client_address = client_address
        # self.rfile: BytesIO so rfile.read(n) is bounded
        self.rfile = io.BytesIO(body)
        # self.wfile: capture what the handler writes back to the client
        self.wfile = io.BytesIO()
        self._body_buf = body
        self._query_string = query_string
        # Capture log_message output (the handler prints on every request).
        self._logs = []

    def address_string(self):
        return "127.0.0.1"

    def log_message(self, format, *args):
        self._logs.append(format % args)

    # BaseHTTPRequestHandler.send_error is sometimes called when send_response
    # sees a status it doesn't like — we don't expect it here, but stub it
    # just in case so tests fail loudly if it's invoked.
    def send_error(self, code, message=None, explain=None):
        raise AssertionError(f"send_error({code}, {message!r}, {explain!r}) was called")


def _build_handler(mod, request: _FakeRequest) -> "mod.RouterHandler":
    """Attach a RouterHandler to a _FakeRequest and return the instance."""
    # Bypass __init__ — it would try to read from a real socket.
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
    """Parse the BytesIO written to as an HTTP response."""
    raw = wfile.getvalue()
    if not raw:
        return None, b""
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
    return code, payload


class _FakeUrlOpenResponse:
    """Minimal http.client.HTTPResponse replacement that supports the
    context manager protocol used by urllib.request.urlopen()."""

    def __init__(self, body=b"{}", status=200, headers=None):
        self._body = body
        self.status = status
        self._headers = list(headers or [])

    def read(self):
        return self._body

    def getheaders(self):
        return self._headers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Module-level: load with default env (no ROUTER_API_TOKEN), and capture the
# auto-generated token so we can use it in tests that DO need auth.
# ---------------------------------------------------------------------------

@pytest.fixture
def router_default(monkeypatch):
    """Load the router with no ROUTER_API_TOKEN → uses auto-generated token."""
    monkeypatch.delenv("ROUTER_API_TOKEN", raising=False)
    mod = _load_router(ROUTER_API_TOKEN=None)
    return mod


@pytest.fixture
def router_with_env_token(monkeypatch):
    """Load the router with an explicit ROUTER_API_TOKEN."""
    monkeypatch.setenv("ROUTER_API_TOKEN", "explicit-token-from-env-1234567890")
    mod = _load_router(ROUTER_API_TOKEN="explicit-token-from-env-1234567890")
    return mod


# ===========================================================================
# Test 20: MAX_REQUEST_BYTES default value is 32 MiB.
# ===========================================================================

def test_max_request_bytes_default_value():
    """Default MAX_REQUEST_BYTES must be exactly 32 MiB."""
    assert 32 * 1024 * 1024 == 32 * 1024 * 1024  # sanity
    # Load with no env override → default kicks in.
    os.environ.pop("MAX_REQUEST_BYTES", None)
    mod = _load_router(MAX_REQUEST_BYTES=None)
    assert mod.MAX_REQUEST_BYTES == 32 * 1024 * 1024


# ===========================================================================
# Test 1: Oversized body via Content-Length → 413.
# ===========================================================================

def test_request_body_size_limit(router_default, monkeypatch):
    """POST with Content-Length > MAX_REQUEST_BYTES → 413."""
    # Send a tiny body but lie about Content-Length via a custom header.
    # The handler reads self.headers.get('Content-Length') which is the
    # parsed header — set it to a value above the cap.
    oversized_length = router_default.MAX_REQUEST_BYTES + 1
    req = _FakeRequest(
        method="POST",
        path="/v1/chat/completions",
        headers={"Content-Length": str(oversized_length), "Content-Type": "application/json"},
        body=b'{"model": "x"}',  # actual body irrelevant — we reject on header
    )
    handler = _build_handler(router_default, req)
    handler.do_POST()
    code, payload = _read_json_response(req.wfile)
    assert code == 413, f"expected 413, got {code} payload={payload}"
    assert payload["limit_bytes"] == router_default.MAX_REQUEST_BYTES


# ===========================================================================
# Test 2: Body larger than declared Content-Length → 413 (chunked-like).
# ===========================================================================

def test_request_body_size_limit_chunked(router_default):
    """PHASE G: Stream more bytes than declared Content-Length → silently
    truncated to declared length. We no longer probe for trailing bytes
    because that probe (`rfile.read(1)` after the body) hangs on
    Connection: close requests, which is the default for urllib, curl,
    and most clients. Tradeoff: clients that lie about Content-Length
    no longer get a 413 — but neither does the router hang on legit
    requests. The request proceeds with the declared-length prefix and
    extra bytes are either consumed by the next pipelined request or
    dropped on socket close."""
    declared = 16
    extra = b'x' * 32  # extra bytes past declared length
    actual = b'{"model":"x"}' + extra
    # PHASE I: Use a non-loopback IP so the auth gate still fires (we
    # want this test to exercise the auth path, not the loopback bypass).
    req = _FakeRequest(
        method="POST",
        path="/v1/chat/completions",
        headers={"Content-Length": str(declared), "Content-Type": "application/json"},
        body=actual,
        client_address=("192.168.1.5", 54321),
    )
    handler = _build_handler(router_default, req)
    handler.do_POST()
    code, _payload = _read_json_response(req.wfile)
    # We expect 401 (auth-required, body was truncated and forwarded).
    # NOT 413 (the chunked-probe path is gone), NOT a hang.
    assert code == 401, (
        f"expected 401 (silently truncated, no hang, non-loopback hits auth), "
        f"got {code} payload={_payload}"
    )


# ===========================================================================
# Test 3: Client sets X-Forwarded-For — backend never sees it.
# ===========================================================================

def test_header_whitelist_blocks_x_forwarded(router_default, monkeypatch):
    """A2: x-forwarded-* and all x-* are stripped before forwarding."""
    captured: dict = {}

    # Stub the upstream call: capture the headers that would be sent.
    def fake_urlopen(req, timeout=None):
        # urllib.request.Request stores headers in req.header_items() in
        # original case, but req.headers is normalized (lowercased keys).
        # Normalize both into a single lowercased dict for assertion.
        for k, v in req.header_items():
            captured[k.lower()] = v
        for k, v in req.headers.items():
            captured.setdefault(k.lower(), v)
        return _FakeUrlOpenResponse()

    monkeypatch.setattr(router_default.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(router_default, "ensure_model_loaded", lambda m: None)
    monkeypatch.setattr(router_default, "load_available_models", lambda: {"m"})

    body = b'{"model":"m"}'
    req = _FakeRequest(
        method="POST",
        path="/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
            "Authorization": "Bearer " + router_default.API_TOKEN,
            "X-Forwarded-For": "1.2.3.4",
            "X-Real-IP": "5.6.7.8",
            "X-Custom": "evil",
            "Host": "evil.example.com",
        },
        body=body,
    )
    handler = _build_handler(router_default, req)
    handler.do_POST()
    # No x-* of any kind should have been forwarded (except our injected marker).
    for k in list(captured.keys()):
        if k == "x-router-forwarded":
            continue
        assert not k.startswith("x-"), f"unexpected header forwarded: {k}={captured[k]!r}"
    assert "host" not in captured
    assert "x-forwarded-for" not in captured
    assert "x-real-ip" not in captured
    # The marker we inject must be present.
    assert captured.get("x-router-forwarded") == "llama-bare-router/1.0"


# ===========================================================================
# Test 4: Client sets Host header — backend never sees it.
# ===========================================================================

def test_header_whitelist_blocks_host(router_default, monkeypatch):
    """A2: Host header is stripped (urllib would otherwise rewrite it)."""
    captured: dict = {}

    def fake_urlopen(req, timeout=None):
        for k, v in req.header_items():
            captured[k.lower()] = v
        for k, v in req.headers.items():
            captured.setdefault(k.lower(), v)
        return _FakeUrlOpenResponse()

    monkeypatch.setattr(router_default.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(router_default, "ensure_model_loaded", lambda m: None)
    monkeypatch.setattr(router_default, "load_available_models", lambda: {"m"})

    body = b'{"model":"m"}'
    req = _FakeRequest(
        method="POST",
        path="/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
            "Authorization": "Bearer " + router_default.API_TOKEN,
            "Host": "evil.example.com",
        },
        body=body,
    )
    handler = _build_handler(router_default, req)
    handler.do_POST()
    assert "host" not in captured
    # Connection / Transfer-Encoding are never present in our whitelist:
    assert "connection" not in captured
    assert "transfer-encoding" not in captured


# ===========================================================================
# Test 5: /health works without token.
# ===========================================================================

def test_no_auth_required_for_health(router_default):
    """Public endpoint — no Authorization header required."""
    req = _FakeRequest(
        method="GET",
        path="/health",
        headers={},  # no auth headers
    )
    handler = _build_handler(router_default, req)
    handler.do_GET()
    code, payload = _read_json_response(req.wfile)
    assert code == 200
    assert payload["status"] == "ok"


# ===========================================================================
# Test 6: /v1/models works without token.
# ===========================================================================

def test_no_auth_required_for_models_list(router_default, monkeypatch):
    """Public endpoint — no Authorization header required."""
    monkeypatch.setattr(router_default, "cached_list_models_sorted_by_basename",
                        lambda p: [{"name": "a"}, {"name": "b"}])
    req = _FakeRequest(method="GET", path="/v1/models", headers={})
    handler = _build_handler(router_default, req)
    handler.do_GET()
    code, payload = _read_json_response(req.wfile)
    assert code == 200
    assert payload["object"] == "list"
    assert [m["id"] for m in payload["data"]] == ["a", "b"]


# ===========================================================================
# Test 7: /v1/models also works without auth (alias to test 6 — explicit).
# ===========================================================================

def test_no_auth_required_for_v1_models(router_default, monkeypatch):
    """Same as test 6 but checks the alternative path form for completeness."""
    monkeypatch.setattr(router_default, "cached_list_models_sorted_by_basename",
                        lambda p: [{"name": "only"}])
    req = _FakeRequest(method="GET", path="/api/v1/models", headers={})  # endswith /v1/models
    handler = _build_handler(router_default, req)
    handler.do_GET()
    code, payload = _read_json_response(req.wfile)
    assert code == 200
    assert payload["data"][0]["id"] == "only"


# ===========================================================================
# Test 8: /v1/chat/completions without token → 401.
# ===========================================================================

def test_auth_required_for_chat_completions(router_default):
    """A3: POST /v1/chat/completions without auth → 401."""
    # PHASE I: non-loopback client — auth gate still fires.
    req = _FakeRequest(
        method="POST",
        path="/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        body=b'{"model":"x"}',
        client_address=("192.168.1.5", 54321),
    )
    handler = _build_handler(router_default, req)
    handler.do_POST()
    code, payload = _read_json_response(req.wfile)
    assert code == 401
    assert "authentication required" in payload["error"]


# ===========================================================================
# Test 9: /v1/completions without token → 401.
# ===========================================================================

def test_auth_required_for_completions(router_default):
    """A3: POST /v1/completions without auth → 401."""
    # PHASE I: non-loopback client — auth gate still fires.
    req = _FakeRequest(
        method="POST",
        path="/v1/completions",
        headers={"Content-Type": "application/json"},
        body=b'{"model":"x"}',
        client_address=("192.168.1.5", 54321),
    )
    handler = _build_handler(router_default, req)
    handler.do_POST()
    code, payload = _read_json_response(req.wfile)
    assert code == 401


# ===========================================================================
# Test 10: /v1/embeddings without token → 401.
# ===========================================================================

def test_auth_required_for_embeddings(router_default):
    """A3: POST /v1/embeddings without auth → 401."""
    # PHASE I: non-loopback client — auth gate still fires.
    req = _FakeRequest(
        method="POST",
        path="/v1/embeddings",
        headers={"Content-Type": "application/json"},
        body=b'{"model":"x","input":"hi"}',
        client_address=("192.168.1.5", 54321),
    )
    handler = _build_handler(router_default, req)
    handler.do_POST()
    code, payload = _read_json_response(req.wfile)
    assert code == 401


# ===========================================================================
# Test 11: Authorization: Bearer <valid token> works.
# ===========================================================================

def test_auth_accepts_authorization_bearer(router_default, monkeypatch):
    """A3: Bearer token with the right value passes auth."""
    def fake_urlopen(req, timeout=None):
        return _FakeUrlOpenResponse()

    monkeypatch.setattr(router_default.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(router_default, "ensure_model_loaded", lambda m: None)
    monkeypatch.setattr(router_default, "load_available_models", lambda: {"m"})

    body = b'{"model":"m"}'
    req = _FakeRequest(
        method="POST",
        path="/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {router_default.API_TOKEN}",
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        },
        body=body,
    )
    handler = _build_handler(router_default, req)
    handler.do_POST()
    code, _ = _read_json_response(req.wfile)
    assert code == 200


# ===========================================================================
# Test 12: X-API-Token: <valid token> works.
# ===========================================================================

def test_auth_accepts_x_api_token(router_default, monkeypatch):
    """A3: X-API-Token header is also accepted."""
    def fake_urlopen(req, timeout=None):
        return _FakeUrlOpenResponse()

    monkeypatch.setattr(router_default.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(router_default, "ensure_model_loaded", lambda m: None)
    monkeypatch.setattr(router_default, "load_available_models", lambda: {"m"})

    body = b'{"model":"m"}'
    req = _FakeRequest(
        method="POST",
        path="/v1/chat/completions",
        headers={
            "X-API-Token": router_default.API_TOKEN,
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        },
        body=body,
    )
    handler = _build_handler(router_default, req)
    handler.do_POST()
    code, _ = _read_json_response(req.wfile)
    assert code == 200


# ===========================================================================
# Test 13: Wrong token → 401 (constant-time via secrets.compare_digest).
# ===========================================================================

def test_auth_rejects_wrong_token(router_default):
    """A3: Any token other than the configured one → 401."""
    # Even a token that's "almost right" should fail.
    almost_right = router_default.API_TOKEN[:-1] + "X"
    for bad in ("definitely-wrong", "", almost_right, router_default.API_TOKEN + "x"):
        # PHASE I: non-loopback client — auth gate still fires.
        req = _FakeRequest(
            method="POST",
            path="/v1/chat/completions",
            headers={"Authorization": f"Bearer {bad}"},
            body=b'{"model":"x"}',
            client_address=("192.168.1.5", 54321),
        )
        handler = _build_handler(router_default, req)
        handler.do_POST()
        code, _ = _read_json_response(req.wfile)
        assert code == 401, f"bad token {bad!r} should have been rejected, got {code}"


# ===========================================================================
# Test 14: ROUTER_API_TOKEN env var is respected.
# ===========================================================================

def test_auth_token_from_env_var(router_with_env_token):
    """A3: If $ROUTER_API_TOKEN is set, that's the active token."""
    assert router_with_env_token.API_TOKEN == "explicit-token-from-env-1234567890"
    # And requests with that exact token pass auth (don't fully proxy here,
    # just verify authenticate returns True).
    assert router_with_env_token.authenticate(
        {"Authorization": "Bearer explicit-token-from-env-1234567890"}
    )
    assert not router_with_env_token.authenticate(
        {"Authorization": "Bearer wrong"}
    )


# ===========================================================================
# Test 15: If env var unset, token is auto-generated and is non-trivial.
# ===========================================================================

def test_auth_token_autogenerated(router_default, monkeypatch, capsys):
    """A3: When env var is unset, a token is generated and printed at startup."""
    # Run main() — but we don't want it to actually bind the port. Patch it.
    class _Server:
        def __init__(self, *a, **k): pass
        def serve_forever(self): pass
        def shutdown(self): pass

    monkeypatch.setattr(router_default, "ThreadingHTTPServer", _Server)
    monkeypatch.setattr(router_default, "load_available_models", lambda: set())
    monkeypatch.setattr(router_default, "read_current_model_from_backend", lambda: None)

    router_default.main()
    captured = capsys.readouterr()
    # Auto-generated token is logged exactly once at startup.
    assert router_default.API_TOKEN in captured.out
    assert "auto-generated" in captured.out
    # It must look like a token_urlsafe(32) output (≥ 32 chars).
    assert len(router_default.API_TOKEN) >= 32


# ===========================================================================
# Test 16: Backend 500 → client sees only request_id, no path leak.
# ===========================================================================

def test_error_messages_no_path_leak(router_default, monkeypatch):
    """A4: Upstream HTTPError body with a path is sanitized in the response."""
    secret_path = "/home/fekry/secret/model-path/Q4_K_M.gguf"
    upstream_body = json.dumps({
        "error": f"failed to read model at {secret_path}",
        "stack": f"File \"{secret_path}\", line 1",
    }).encode()

    def fake_urlopen(req, timeout=None):
        raise HTTPError(req.full_url, 500, "Internal Server Error", {},
                        io.BytesIO(upstream_body))

    monkeypatch.setattr(router_default.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(router_default, "ensure_model_loaded", lambda m: None)
    monkeypatch.setattr(router_default, "load_available_models", lambda: {"m"})

    body = b'{"model":"m"}'
    req = _FakeRequest(
        method="POST",
        path="/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {router_default.API_TOKEN}",
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        },
        body=body,
    )
    handler = _build_handler(router_default, req)
    handler.do_POST()
    code, payload = _read_json_response(req.wfile)
    assert code == 500
    # The path must NOT leak to the client.
    serialized = json.dumps(payload)
    assert secret_path not in serialized, f"path leaked to client: {serialized}"
    # But the request_id IS in the response so support can correlate.
    assert "request_id" in payload
    assert len(payload["request_id"]) == 32  # uuid4 hex


# ===========================================================================
# Test 17: Server-side stderr log has the full upstream error (including path).
# ===========================================================================

def test_error_messages_log_full_to_stderr(router_default, monkeypatch, capfd):
    """A4: Full upstream error including the path is written to stderr."""
    secret_path = "/home/fekry/secret/model-path/Q4_K_M.gguf"
    upstream_body = json.dumps({"error": f"failed at {secret_path}"}).encode()

    def fake_urlopen(req, timeout=None):
        raise HTTPError(req.full_url, 500, "Internal Server Error", {},
                        io.BytesIO(upstream_body))

    monkeypatch.setattr(router_default.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(router_default, "ensure_model_loaded", lambda m: None)
    monkeypatch.setattr(router_default, "load_available_models", lambda: {"m"})

    body = b'{"model":"m"}'
    req = _FakeRequest(
        method="POST",
        path="/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {router_default.API_TOKEN}",
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        },
        body=body,
    )
    handler = _build_handler(router_default, req)
    handler.do_POST()
    captured = capfd.readouterr()
    # Path MUST be in the stderr log for operators.
    assert secret_path in captured.err, f"path missing from stderr: {captured.err!r}"
    # The log line should be tagged with a request_id.
    assert "request_id=" in captured.err


# ===========================================================================
# Test 18: Client sets Connection: keep-alive — backend doesn't see it.
# ===========================================================================

def test_proxy_strips_connection_header(router_default, monkeypatch):
    """A2: Connection / Transfer-Encoding are hop-by-hop and never forwarded."""
    captured: dict = {}

    def fake_urlopen(req, timeout=None):
        for k, v in req.header_items():
            captured[k.lower()] = v
        for k, v in req.headers.items():
            captured.setdefault(k.lower(), v)
        return _FakeUrlOpenResponse()

    monkeypatch.setattr(router_default.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(router_default, "ensure_model_loaded", lambda m: None)
    monkeypatch.setattr(router_default, "load_available_models", lambda: {"m"})

    body = b'{"model":"m"}'
    req = _FakeRequest(
        method="POST",
        path="/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {router_default.API_TOKEN}",
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
            "Connection": "keep-alive",
            "Transfer-Encoding": "chunked",
        },
        body=body,
    )
    handler = _build_handler(router_default, req)
    handler.do_POST()
    assert "connection" not in captured
    assert "transfer-encoding" not in captured


# ===========================================================================
# Test 19: Backend receives X-Router-Forwarded header on every proxy.
# ===========================================================================

def test_proxy_adds_x_router_forwarded(router_default, monkeypatch):
    """A2: X-Router-Forwarded is injected so the backend knows it's proxied."""
    captured: dict = {}

    def fake_urlopen(req, timeout=None):
        for k, v in req.header_items():
            captured[k.lower()] = v
        for k, v in req.headers.items():
            captured.setdefault(k.lower(), v)
        return _FakeUrlOpenResponse()

    monkeypatch.setattr(router_default.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(router_default, "ensure_model_loaded", lambda m: None)
    monkeypatch.setattr(router_default, "load_available_models", lambda: {"m"})

    # Client sends zero headers (other than auth).
    body = b'{"model":"m"}'
    req = _FakeRequest(
        method="POST",
        path="/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {router_default.API_TOKEN}",
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        },
        body=body,
    )
    handler = _build_handler(router_default, req)
    handler.do_POST()
    # Regardless of what the client sent, the marker is there.
    assert captured.get("x-router-forwarded") == "llama-bare-router/1.0"


# ===========================================================================
# Bonus: ensure_model_loaded failure path also produces a sanitized error.
# This isn't on the required 20, but it's the other half of A4 and an easy
# regression to introduce.
# ===========================================================================

def test_ensure_model_loaded_failure_sanitizes(router_default, monkeypatch):
    """A4: If ensure_model_loaded raises (e.g., systemctl fails), the
    response body doesn't contain the exception message."""
    def boom(name):
        raise RuntimeError(f"systemctl restart failed for path /etc/systemd/system/{name}")
    monkeypatch.setattr(router_default, "ensure_model_loaded", boom)
    monkeypatch.setattr(router_default, "load_available_models", lambda: {"m"})

    body = b'{"model":"m"}'
    req = _FakeRequest(
        method="POST",
        path="/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {router_default.API_TOKEN}",
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        },
        body=body,
    )
    handler = _build_handler(router_default, req)
    handler.do_POST()
    code, payload = _read_json_response(req.wfile)
    assert code == 500
    assert "/etc/systemd/system/m" not in json.dumps(payload)
    assert payload["error"] == "internal error"
    assert "request_id" in payload


# ===========================================================================
# B5/B6: connection cleanup on exception, urlopen timeout, .env auto-close.
# ===========================================================================

def test_urlopen_response_closed_on_exception(router_default, monkeypatch):
    """B5: When resp.read() raises mid-stream, the urlopen response object
    must still get its close() called (context manager exit covers this).
    Verify via a fake response that records whether close() was invoked."""
    close_called = {"n": 0}

    class _LeakyResponse(_FakeUrlOpenResponse):
        def read(self):
            raise IOError("simulated mid-stream read error")

        def close(self):
            close_called["n"] += 1

    def fake_urlopen(req, timeout=None):
        return _LeakyResponse()

    monkeypatch.setattr(router_default.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(router_default, "ensure_model_loaded", lambda m: None)
    monkeypatch.setattr(router_default, "load_available_models", lambda: {"m"})

    body = b'{"model":"m"}'
    req = _FakeRequest(
        method="POST",
        path="/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {router_default.API_TOKEN}",
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        },
        body=body,
    )
    handler = _build_handler(router_default, req)
    handler.do_POST()
    # The with-block guarantees close() is called, so we expect at least 1.
    assert close_called["n"] >= 1, (
        f"urlopen response not closed after exception: "
        f"close_called={close_called['n']}"
    )
    # The client gets a sanitized 500 — no path leak.
    code, payload = _read_json_response(req.wfile)
    assert code == 500
    assert payload["error"] == "internal error"


def test_urlopen_timeout_propagates_from_urlopen_call(router_default, monkeypatch):
    """B5: When the urlopen call itself blows up (e.g., socket timeout
    during connect), the request still gets a sanitized 500 response —
    not a hung connection or leaked file descriptor."""
    def fake_urlopen(req, timeout=None):
        raise TimeoutError(f"connect timeout after {timeout}s")

    monkeypatch.setattr(router_default.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(router_default, "ensure_model_loaded", lambda m: None)
    monkeypatch.setattr(router_default, "load_available_models", lambda: {"m"})

    body = b'{"model":"m"}'
    req = _FakeRequest(
        method="POST",
        path="/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {router_default.API_TOKEN}",
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        },
        body=body,
    )
    handler = _build_handler(router_default, req)
    handler.do_POST()
    code, payload = _read_json_response(req.wfile)
    assert code == 500
    assert "request_id" in payload


def test_proxy_uses_urlopen_timeout_300(router_default, monkeypatch):
    """B5: Sanity check that the timeout=300 argument is actually passed
    to urlopen (no regressions if a future refactor drops it)."""
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["timeout"] = timeout
        return _FakeUrlOpenResponse(body=b"{}", status=200, headers=[])

    monkeypatch.setattr(router_default.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(router_default, "ensure_model_loaded", lambda m: None)
    monkeypatch.setattr(router_default, "load_available_models", lambda: {"m"})

    body = b'{"model":"m"}'
    req = _FakeRequest(
        method="POST",
        path="/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {router_default.API_TOKEN}",
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        },
        body=body,
    )
    handler = _build_handler(router_default, req)
    handler.do_POST()
    assert captured["timeout"] == 300, (
        f"expected timeout=300 forwarded to urlopen, got {captured['timeout']!r}"
    )


# ===========================================================================
# B2: subprocess.run timeout on systemctl restart.
# ===========================================================================

def test_subprocess_restart_uses_timeout(router_default, monkeypatch):
    """B2: subprocess.run(...) for `systemctl --user restart` MUST be
    called with a finite timeout — a hung systemctl blocks every request.
    We patch subprocess.run to assert it was called with timeout=30."""
    captured: list = []

    def fake_run(argv, *args, **kwargs):
        captured.append({"argv": argv, "kwargs": kwargs})
        # First call: the primary restart. Return success.
        r = type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        return r

    monkeypatch.setattr(router_default.subprocess, "run", fake_run)

    # Stub health-check so the loop returns immediately.
    monkeypatch.setattr(router_default, "wait_for_health", lambda timeout=120, interval=2: True)
    # Stub write_env_file so it doesn't touch the disk.
    monkeypatch.setattr(router_default, "write_env_file", lambda path, name: None)

    router_default.restart_with_model("dummy")
    assert len(captured) >= 1, f"subprocess.run was never called: {captured}"
    first_kwargs = captured[0]["kwargs"]
    assert "timeout" in first_kwargs, f"subprocess.run had no timeout: {first_kwargs}"
    assert first_kwargs["timeout"] == 30, (
        f"expected timeout=30 (systemctl restart must not block forever), "
        f"got {first_kwargs['timeout']}"
    )


def test_subprocess_restart_timeout_falls_back_to_stop_start(router_default, monkeypatch):
    """B2: When the primary `systemctl restart` exceeds 30s (TimeoutExpired),
    the router must attempt a `systemctl stop` + `systemctl start` fallback
    instead of giving up.
    """
    import subprocess as real_subprocess

    calls: list = []
    response_index = {"i": 0}

    responses = [
        # First call: primary restart times out.
        real_subprocess.TimeoutExpired(cmd=["systemctl", "--user", "restart"], timeout=30),
        # Second call: stop succeeds.
        type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
        # Third call: start succeeds.
        type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
    ]

    def fake_run(argv, *args, **kwargs):
        calls.append(argv)
        resp = responses[len(calls) - 1]
        if isinstance(resp, Exception):
            raise resp
        return resp

    monkeypatch.setattr(router_default.subprocess, "run", fake_run)
    monkeypatch.setattr(router_default, "wait_for_health", lambda timeout=120, interval=2: True)
    monkeypatch.setattr(router_default, "write_env_file", lambda path, name: None)

    router_default.restart_with_model("dummy")

    # Exactly one restart attempt that timed out, then stop+start as fallback.
    assert len(calls) == 3, f"expected 3 subprocess calls (restart+stop+start), got {calls}"
    assert "restart" in calls[0]
    assert "stop" in calls[1], f"second call should be stop, got {calls[1]}"
    assert "start" in calls[2], f"third call should be start, got {calls[2]}"


def test_subprocess_restart_raises_when_all_paths_fail(router_default, monkeypatch):
    """B2: If the primary restart times out AND the fallback stop+start
    also fails, the caller must see a RuntimeError so the request returns 503."""
    import subprocess as real_subprocess

    responses = [
        # Primary restart times out.
        real_subprocess.TimeoutExpired(cmd=["systemctl", "--user", "restart"], timeout=30),
        # Stop fails (rc=1).
        type("R", (), {"returncode": 1, "stdout": "", "stderr": "stop failed"})(),
    ]

    def fake_run(argv, *args, **kwargs):
        resp = responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp

    monkeypatch.setattr(router_default.subprocess, "run", fake_run)
    monkeypatch.setattr(router_default, "write_env_file", lambda path, name: None)

    with pytest.raises(RuntimeError, match="(stop|start|restart)"):
        router_default.restart_with_model("dummy")


# ===========================================================================
# PHASE G: Connection: close / keep-alive hang regression tests.
#
# Before the fix: `_read_body_capped` did `self.rfile.read(1)` after
# reading the declared body to probe for chunked-encoded trailing data.
# On `Connection: close` requests (the default for urllib, curl, and
# most clients) the client had already closed the socket — so `read(1)`
# blocked until the client timed out, at which point BrokenPipe
# occurred when the router tried to write a 413. The router hung on
# every legit POST.
#
# The fix drops the probe entirely (we trust Content-Length; extra
# bytes are silently truncated or dropped on socket close).
#
# These tests exercise the handler via the in-memory `_FakeRequest`
# harness with `request_version` set to HTTP/1.0 and `Connection: close`
# headers to verify the handler does NOT block and returns immediately.
# ===========================================================================


def test_router_chat_completions_with_connection_close_returns_immediately(
    router_default, monkeypatch
):
    """PHASE G #1: A POST with `Connection: close` (the default for
    urllib/curl) must return within 1s — not hang on the now-removed
    `next_byte = self.rfile.read(1)` probe."""
    import time

    # Stub everything downstream so the test exercises ONLY the body read.
    monkeypatch.setattr(router_default, "ensure_model_loaded", lambda m: None)
    monkeypatch.setattr(router_default, "load_available_models", lambda: {"m"})

    def fake_urlopen(req, timeout=None):
        return _FakeUrlOpenResponse(
            body=b'{"id":"r","choices":[]}', status=200, headers=[]
        )

    monkeypatch.setattr(router_default.request, "urlopen", fake_urlopen)

    body = b'{"model":"m","messages":[{"role":"user","content":"x"}],"max_tokens":4}'
    req = _FakeRequest(
        method="POST",
        path="/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
            "Connection": "close",
            "Authorization": f"Bearer {router_default.API_TOKEN}",
        },
        body=body,
    )
    req.request_version = "HTTP/1.1"  # version alone isn't the trigger
    handler = _build_handler(router_default, req)

    t0 = time.monotonic()
    handler.do_POST()
    elapsed = time.monotonic() - t0
    assert elapsed < 1.0, (
        f"PHASE G REGRESSION: handler took {elapsed:.3f}s — the body probe "
        f"is hanging on Connection: close"
    )
    code, _payload = _read_json_response(req.wfile)
    assert code == 200, f"expected 200 (auth ok, upstream stubbed), got {code}"


def test_router_chat_completions_with_keep_alive_returns_immediately(
    router_default, monkeypatch
):
    """PHASE G #2: HTTP/1.1 with `Connection: keep-alive` must also
    return within 1s — the probe was wrong on every path, not just close."""
    import time

    monkeypatch.setattr(router_default, "ensure_model_loaded", lambda m: None)
    monkeypatch.setattr(router_default, "load_available_models", lambda: {"m"})

    def fake_urlopen(req, timeout=None):
        return _FakeUrlOpenResponse(
            body=b'{"id":"r","choices":[]}', status=200, headers=[]
        )

    monkeypatch.setattr(router_default.request, "urlopen", fake_urlopen)

    body = b'{"model":"m","messages":[{"role":"user","content":"x"}],"max_tokens":4}'
    req = _FakeRequest(
        method="POST",
        path="/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
            "Connection": "keep-alive",
            "Authorization": f"Bearer {router_default.API_TOKEN}",
        },
        body=body,
    )
    handler = _build_handler(router_default, req)

    t0 = time.monotonic()
    handler.do_POST()
    elapsed = time.monotonic() - t0
    assert elapsed < 1.0, f"keep-alive handler took {elapsed:.3f}s"
    code, _payload = _read_json_response(req.wfile)
    assert code == 200, f"expected 200, got {code}"


def test_router_no_hang_on_small_body(router_default, monkeypatch):
    """PHASE G #3: Small (~200 byte) body — must respond in <1s."""
    import time

    monkeypatch.setattr(router_default, "ensure_model_loaded", lambda m: None)
    monkeypatch.setattr(router_default, "load_available_models", lambda: {"m"})

    def fake_urlopen(req, timeout=None):
        return _FakeUrlOpenResponse(
            body=b'{"id":"r","choices":[]}', status=200, headers=[]
        )

    monkeypatch.setattr(router_default.request, "urlopen", fake_urlopen)

    # Realistic small chat-completions body — well under 200 bytes.
    body = (
        b'{"model":"m","messages":[{"role":"user","content":"hello"}],'
        b'"max_tokens":4}'
    )

    req = _FakeRequest(
        method="POST",
        path="/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
            "Connection": "close",
            "Authorization": f"Bearer {router_default.API_TOKEN}",
        },
        body=body,
    )
    handler = _build_handler(router_default, req)
    t0 = time.monotonic()
    handler.do_POST()
    elapsed = time.monotonic() - t0
    assert elapsed < 1.0, f"small-body handler took {elapsed:.3f}s"
    code, _ = _read_json_response(req.wfile)
    assert code == 200


def test_router_no_hang_on_large_body(router_default, monkeypatch):
    """PHASE G #4: Body declared at MAX_REQUEST_BYTES - 1 (just under the
    32 MiB cap) must return in <2s. Since the in-memory BytesIO can't
    actually hold 32 MiB-1, we declare large but rfile returns a small
    body — the handler reads whatever it gets and proceeds to auth +
    parse. The point is to assert NO HANG, regardless of status code."""
    import time

    monkeypatch.setattr(router_default, "ensure_model_loaded", lambda m: None)
    monkeypatch.setattr(router_default, "load_available_models", lambda: {"m"})

    def fake_urlopen(req, timeout=None):
        return _FakeUrlOpenResponse(
            body=b'{"id":"r","choices":[]}', status=200, headers=[]
        )

    monkeypatch.setattr(router_default.request, "urlopen", fake_urlopen)

    declared = router_default.MAX_REQUEST_BYTES - 1  # just under the cap
    # We don't actually allocate that much; rfile just returns b"{}".
    req = _FakeRequest(
        method="POST",
        path="/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(declared),
            "Connection": "close",
            "Authorization": f"Bearer {router_default.API_TOKEN}",
        },
        body=b'{"model":"m"}',
    )
    handler = _build_handler(router_default, req)
    t0 = time.monotonic()
    handler.do_POST()
    elapsed = time.monotonic() - t0
    assert elapsed < 2.0, f"large-body handler took {elapsed:.3f}s"
    # No hang regardless of downstream result; status is 200 (auth ok,
    # upstream stubbed) since the length is under the cap.
    code, _ = _read_json_response(req.wfile)
    assert code == 200


def test_router_no_hang_on_oversized_body(router_default, monkeypatch):
    """PHASE G #5: 35 MiB declared (above the 32 MiB cap) must return
    413 within 2s. The length pre-check rejects before any body read,
    so this should be instant."""
    import time

    monkeypatch.setattr(router_default, "ensure_model_loaded", lambda m: None)
    monkeypatch.setattr(router_default, "load_available_models", lambda: {"m"})

    declared = 35 * 1024 * 1024
    req = _FakeRequest(
        method="POST",
        path="/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(declared),
            "Connection": "close",
            "Authorization": f"Bearer {router_default.API_TOKEN}",
        },
        body=b"{}",
    )
    handler = _build_handler(router_default, req)
    t0 = time.monotonic()
    handler.do_POST()
    elapsed = time.monotonic() - t0
    assert elapsed < 2.0, f"oversized-body handler took {elapsed:.3f}s"
    code, payload = _read_json_response(req.wfile)
    assert code == 413, f"expected 413, got {code} payload={payload}"
    assert payload["limit_bytes"] == router_default.MAX_REQUEST_BYTES


# ===========================================================================
# PHASE G: Unit test for the in-process body-read timeout guard. Without a
# real socket we can't reproduce the broken-pipe path end-to-end, but we
# CAN verify the probe is gone by inspecting the source of _read_body_capped.
# ===========================================================================


def test_read_body_capped_no_probe(router_default):
    """PHASE G: The fix removes the `self.rfile.read(1)` probe entirely.
    This test reads the source and asserts the probe byte is absent."""
    import inspect
    src = inspect.getsource(router_default.RouterHandler._read_body_capped)
    assert "next_byte" not in src, (
        "PHASE G REGRESSION: _read_body_capped still references next_byte — "
        "the Connection: close hang fix has been undone"
    )
    # And the simpler path is in place.
    assert "self.rfile.read(length)" in src, (
        "expected plain `self.rfile.read(length)` body read in _read_body_capped"
    )


# ===========================================================================
# PHASE I: Loopback auth bypass — local clients (127.0.0.1 / ::1) hit the
# service without setting an API token. Non-loopback clients still must
# present a valid token. This keeps the service safe if it ever gets exposed
# to a non-loopback network while letting Hermes Agent, open-webui, OpenClaw,
# and ad-hoc scripts talk to localhost without ceremony.
# ===========================================================================


def _stub_loopback_forward(mod, monkeypatch):
    """Common stubs for the loopback-bypass tests so each test only has to
    assert on the response code — not on the proxy plumbing."""
    monkeypatch.setattr(mod, "ensure_model_loaded", lambda m: None)
    monkeypatch.setattr(mod, "load_available_models", lambda: {"m"})

    def fake_urlopen(req, timeout=None):
        return _FakeUrlOpenResponse(
            body=b'{"id":"r","choices":[]}', status=200, headers=[]
        )

    monkeypatch.setattr(mod.request, "urlopen", fake_urlopen)


def test_loopback_bypasses_auth(router_default, monkeypatch):
    """PHASE I #1: POST /v1/chat/completions from 127.0.0.1 with NO auth
    headers → 200 (auth is bypassed because the client is loopback)."""
    _stub_loopback_forward(router_default, monkeypatch)

    body = b'{"model":"m","messages":[{"role":"user","content":"hi"}]}'
    req = _FakeRequest(
        method="POST",
        path="/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
            # Deliberately no Authorization / X-API-Token header.
        },
        body=body,
        client_address=("127.0.0.1", 54321),
    )
    handler = _build_handler(router_default, req)
    handler.do_POST()
    code, _payload = _read_json_response(req.wfile)
    # Must NOT be 401 — loopback bypass wins.
    assert code != 401, f"loopback request was rejected with 401; payload={_payload}"
    assert code == 200, f"expected 200 from stubbed upstream, got {code}"


def test_loopback_bypasses_auth_ipv6(router_default, monkeypatch):
    """PHASE I #2: POST from ::1 (IPv6 loopback) with no auth → not 401."""
    _stub_loopback_forward(router_default, monkeypatch)

    body = b'{"model":"m","messages":[{"role":"user","content":"hi"}]}'
    req = _FakeRequest(
        method="POST",
        path="/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        },
        body=body,
        client_address=("::1", 54321),
    )
    handler = _build_handler(router_default, req)
    handler.do_POST()
    code, _payload = _read_json_response(req.wfile)
    assert code != 401, f"IPv6 loopback request was rejected with 401; payload={_payload}"
    assert code == 200


def test_non_loopback_requires_auth(router_default, monkeypatch):
    """PHASE I #3: POST from a non-loopback IP (192.168.1.5) with no auth
    → 401. The auth gate still works for non-loopback clients so the
    service is safe if it ever gets exposed to a network."""
    # No stubs needed — auth check rejects before any upstream call.
    body = b'{"model":"m"}'
    req = _FakeRequest(
        method="POST",
        path="/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        },
        body=body,
        client_address=("192.168.1.5", 54321),
    )
    handler = _build_handler(router_default, req)
    handler.do_POST()
    code, payload = _read_json_response(req.wfile)
    assert code == 401, f"non-loopback without auth must be 401, got {code}"
    assert "authentication required" in payload["error"]


def test_loopback_with_wrong_token_still_works(router_default, monkeypatch):
    """PHASE I #4: Loopback + a wrong token → still passes (loopback
    bypass wins; the bad token is irrelevant)."""
    _stub_loopback_forward(router_default, monkeypatch)

    body = b'{"model":"m","messages":[{"role":"user","content":"hi"}]}'
    req = _FakeRequest(
        method="POST",
        path="/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
            "Authorization": "Bearer this-token-is-deliberately-wrong",
        },
        body=body,
        client_address=("127.0.0.1", 54321),
    )
    handler = _build_handler(router_default, req)
    handler.do_POST()
    code, _payload = _read_json_response(req.wfile)
    assert code != 401, (
        f"loopback request with wrong token was rejected with 401; "
        f"loopback bypass should make the token irrelevant"
    )
    assert code == 200


def test_non_loopback_with_valid_token_passes(router_default, monkeypatch):
    """PHASE I #5: Non-loopback + the right token → 200. Auth gate is
    intact for non-loopback clients who DO have a token."""
    _stub_loopback_forward(router_default, monkeypatch)

    body = b'{"model":"m","messages":[{"role":"user","content":"hi"}]}'
    req = _FakeRequest(
        method="POST",
        path="/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
            "Authorization": f"Bearer {router_default.API_TOKEN}",
        },
        body=body,
        client_address=("192.168.1.5", 54321),
    )
    handler = _build_handler(router_default, req)
    handler.do_POST()
    code, _payload = _read_json_response(req.wfile)
    assert code != 401, (
        f"non-loopback with valid token was rejected; payload={_payload}"
    )
    assert code == 200
