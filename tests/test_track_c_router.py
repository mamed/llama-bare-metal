"""Comprehensive behavioral tests for llama-router.py (PHASE C).

Coverage targets:
- C1: do_GET routing (/health, /v1/models, other GET → proxy)
- C2: do_POST routing (/v1/chat/completions, /v1/completions, /v1/embeddings,
                   missing model, unknown model, auth required)
- C3: ensure_model_loaded behavior (no-op, restart, lock serialization)
- C4: restart_with_model behavior (write_env first, timeout, fallback,
                                   RuntimeError on full failure)
- C5: wait_for_health behavior (success, timeout, configurable timeout)
- C6: load_available_models / read_current_model_from_backend env wiring
- C7: main() / __main__ block + module-level defaults

We exercise RouterHandler end-to-end by giving it an in-memory rfile/wfile
and stubbed subprocess / urllib.request.urlopen. No real sockets, no real
network, no real subprocess.

This file is the PHASE C test track. We deliberately keep test fixtures and
helpers local to this file to avoid cross-track coupling with the security
tests (PHASE A) or state tests (PHASE B).
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
from urllib.error import HTTPError

import pytest


# ---------------------------------------------------------------------------
# Module loading: llama-router.py is a script, not a package, so we load it
# by file path. Reload on demand so module-level constants reflect the
# monkeypatched env vars in each test.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
ROUTER_PATH = REPO_ROOT / "llama-router.py"


def _load_router(**env_overrides):
    """Import llama-router.py with the given env vars set before import."""
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
# Test harness: build a RouterHandler with in-memory streams.
# Mirrors the structure used by PHASE A's test_router_security.py, but kept
# inline here so PHASE C owns its own helpers (per track-isolation rule).
# ---------------------------------------------------------------------------


class _FakeRequest:
    """Minimal BaseHTTPRequestHandler per-request state, in-memory."""

    def __init__(self, *, method="POST", path="/v1/chat/completions", headers=None,
                 body=b"", client_address=("127.0.0.1", 0)):
        self.headers = dict(headers or {})
        self.path = path
        self.command = method
        self.request_version = "HTTP/1.1"
        self.requestline = f"{method} {path} HTTP/1.1"
        self.client_address = client_address
        self.rfile = io.BytesIO(body)
        self.wfile = io.BytesIO()
        self._logs = []

    def address_string(self):
        return "127.0.0.1"

    def log_message(self, format, *args):
        self._logs.append(format % args)

    def send_error(self, code, message=None, explain=None):
        raise AssertionError(f"send_error({code}, {message!r}, {explain!r}) was called")


def _build_handler(mod, request):
    """Bypass __init__ and attach a RouterHandler to a _FakeRequest."""
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
    """Parse an HTTP response from the handler's in-memory wfile."""
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
    """Stub http.client.HTTPResponse compatible with the router's `with` block."""

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
# Fixtures: load router with a known token, no token, and a temp models.yaml.
# ---------------------------------------------------------------------------


@pytest.fixture
def router_with_token(monkeypatch):
    """Load router with an explicit ROUTER_API_TOKEN (deterministic auth)."""
    monkeypatch.setenv("ROUTER_API_TOKEN", "phase-c-test-token-1234567890")
    return _load_router(ROUTER_API_TOKEN="phase-c-test-token-1234567890")


@pytest.fixture
def router_no_token(monkeypatch):
    """Load router with no ROUTER_API_TOKEN (uses auto-generated token)."""
    monkeypatch.delenv("ROUTER_API_TOKEN", raising=False)
    return _load_router(ROUTER_API_TOKEN=None)


@pytest.fixture
def temp_models_yaml(tmp_path):
    """Write a small models.yaml and point MODELS_YAML at it."""
    yaml_path = tmp_path / "models.yaml"
    yaml_path.write_text(
        "models:\n"
        "  - name: alpha\n"
        "    model: /models/A/file-Q4.gguf\n"
        "  - name: beta\n"
        "    model: /models/B/thing-Q8.gguf\n"
    )
    return yaml_path


# ===========================================================================
# C1: do_GET routing
# ===========================================================================


def test_do_get_health_no_auth_required(router_with_token):
    """C1: GET /health returns 200 with the documented payload, no auth."""
    req = _FakeRequest(method="GET", path="/health", headers={})
    handler = _build_handler(router_with_token, req)
    handler.do_GET()
    code, payload = _read_json_response(req.wfile)
    assert code == 200, f"expected 200, got {code}"
    assert payload == {"status": "ok", "router": "cuda-router"}


def test_do_get_v1_models_returns_openai_format(router_with_token, monkeypatch):
    """C1: GET /v1/models returns OpenAI-compatible {object, data} shape."""
    monkeypatch.setattr(
        router_with_token,
        "cached_list_models_sorted_by_basename",
        lambda p: [{"name": "alpha"}, {"name": "beta"}, {"name": "gamma"}],
    )
    req = _FakeRequest(method="GET", path="/v1/models", headers={})
    handler = _build_handler(router_with_token, req)
    handler.do_GET()
    code, payload = _read_json_response(req.wfile)
    assert code == 200, f"expected 200, got {code}"
    assert payload["object"] == "list"
    assert [m["id"] for m in payload["data"]] == ["alpha", "beta", "gamma"]
    assert all(m["object"] == "model" for m in payload["data"])


def test_do_get_v1_models_sorted_by_basename(router_with_token, monkeypatch):
    """C1: /v1/models reflects whatever order cached_list returns — the
    router itself does no additional sorting, so verify it passes entries
    through in the order the cache provides (sorting is unit-tested in
    test_router_state)."""
    sorted_entries = [
        {"name": "alpha", "model": "/models/A/file-Q4.gguf"},
        {"name": "beta", "model": "/models/B/thing-Q8.gguf"},
        {"name": "gamma", "model": "/models/C/other.gguf"},
    ]
    monkeypatch.setattr(
        router_with_token,
        "cached_list_models_sorted_by_basename",
        lambda p: sorted_entries,
    )
    req = _FakeRequest(method="GET", path="/v1/models", headers={})
    handler = _build_handler(router_with_token, req)
    handler.do_GET()
    code, payload = _read_json_response(req.wfile)
    assert code == 200
    assert [m["id"] for m in payload["data"]] == ["alpha", "beta", "gamma"]


# ===========================================================================
# C2: do_POST routing — proxies for all three endpoints + auth + validation
# ===========================================================================


@pytest.mark.parametrize("endpoint", [
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
])
def test_do_post_proxies_to_backend(router_with_token, monkeypatch, endpoint):
    """C2: All three POST endpoints forward to the backend when auth + model valid."""
    captured = {"called": False, "url": None, "method": None, "body": None}

    def fake_urlopen(req, timeout=None):
        captured["called"] = True
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["body"] = req.data
        return _FakeUrlOpenResponse(body=b'{"ok":true}', status=200, headers=[("X-Upstream", "yes")])

    monkeypatch.setattr(router_with_token.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(router_with_token, "ensure_model_loaded", lambda m: (True, ("ok", 0.0)))
    monkeypatch.setattr(router_with_token, "load_available_models", lambda: {"m"})

    body = b'{"model":"m","messages":[]}'
    headers = {
        "Authorization": f"Bearer {router_with_token.API_TOKEN}",
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
    }
    req = _FakeRequest(method="POST", path=endpoint, headers=headers, body=body)
    handler = _build_handler(router_with_token, req)
    handler.do_POST()

    code, payload = _read_json_response(req.wfile)
    assert captured["called"], "urlopen was never called"
    assert captured["url"].endswith(endpoint)
    assert captured["method"] == "POST"
    assert captured["body"] == body
    assert code == 200


def test_do_post_missing_model_returns_400(router_with_token, monkeypatch):
    """C2: POST with a body that has no 'model' field → 400."""
    monkeypatch.setattr(router_with_token, "load_available_models", lambda: {"m"})
    body = b'{"messages":[]}'  # no model
    headers = {
        "Authorization": f"Bearer {router_with_token.API_TOKEN}",
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
    }
    req = _FakeRequest(method="POST", path="/v1/chat/completions", headers=headers, body=body)
    handler = _build_handler(router_with_token, req)
    handler.do_POST()
    code, payload = _read_json_response(req.wfile)
    assert code == 400
    assert "model" in payload["error"]


def test_do_post_unknown_model_returns_400(router_with_token, monkeypatch):
    """C2: POST with a model name not in models.yaml → 400 + available list."""
    monkeypatch.setattr(router_with_token, "load_available_models", lambda: {"alpha", "beta"})
    body = b'{"model":"nonexistent"}'
    headers = {
        "Authorization": f"Bearer {router_with_token.API_TOKEN}",
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
    }
    req = _FakeRequest(method="POST", path="/v1/chat/completions", headers=headers, body=body)
    handler = _build_handler(router_with_token, req)
    handler.do_POST()
    code, payload = _read_json_response(req.wfile)
    assert code == 400
    assert "nonexistent" in payload["error"]
    assert sorted(payload["available"]) == ["alpha", "beta"]


def test_do_post_without_auth_returns_401(router_with_token):
    """C2: POST /v1/chat/completions with no auth headers → 401."""
    body = b'{"model":"m"}'
    headers = {
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
    }
    # PHASE I: non-loopback client — auth gate still fires.
    req = _FakeRequest(
        method="POST",
        path="/v1/chat/completions",
        headers=headers,
        body=body,
        client_address=("192.168.1.5", 54321),
    )
    handler = _build_handler(router_with_token, req)
    handler.do_POST()
    code, payload = _read_json_response(req.wfile)
    assert code == 401
    assert "authentication" in payload["error"]


def test_do_post_with_bearer_auth_proxies(router_with_token, monkeypatch):
    """C2: Bearer auth with the right token → proxy succeeds."""
    monkeypatch.setattr(
        router_with_token.request, "urlopen",
        lambda req, timeout=None: _FakeUrlOpenResponse(body=b'{"ok":1}', status=200, headers=[]),
    )
    monkeypatch.setattr(router_with_token, "ensure_model_loaded", lambda m: (True, ("ok", 0.0)))
    monkeypatch.setattr(router_with_token, "load_available_models", lambda: {"m"})
    body = b'{"model":"m"}'
    headers = {
        "Authorization": f"Bearer {router_with_token.API_TOKEN}",
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
    }
    req = _FakeRequest(method="POST", path="/v1/chat/completions", headers=headers, body=body)
    handler = _build_handler(router_with_token, req)
    handler.do_POST()
    code, _ = _read_json_response(req.wfile)
    assert code == 200


def test_do_post_with_x_api_token_proxies(router_with_token, monkeypatch):
    """C2: X-API-Token header is also accepted for auth."""
    monkeypatch.setattr(
        router_with_token.request, "urlopen",
        lambda req, timeout=None: _FakeUrlOpenResponse(body=b'{"ok":1}', status=200, headers=[]),
    )
    monkeypatch.setattr(router_with_token, "ensure_model_loaded", lambda m: (True, ("ok", 0.0)))
    monkeypatch.setattr(router_with_token, "load_available_models", lambda: {"m"})
    body = b'{"model":"m"}'
    headers = {
        "X-API-Token": router_with_token.API_TOKEN,
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
    }
    req = _FakeRequest(method="POST", path="/v1/chat/completions", headers=headers, body=body)
    handler = _build_handler(router_with_token, req)
    handler.do_POST()
    code, _ = _read_json_response(req.wfile)
    assert code == 200


def test_do_post_with_wrong_token_returns_401(router_with_token):
    """C2: Wrong token → 401, never reaches the backend."""
    body = b'{"model":"m"}'
    headers = {
        "Authorization": "Bearer definitely-not-the-right-token",
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
    }
    # PHASE I: non-loopback client — auth gate still fires.
    req = _FakeRequest(
        method="POST",
        path="/v1/chat/completions",
        headers=headers,
        body=body,
        client_address=("192.168.1.5", 54321),
    )
    handler = _build_handler(router_with_token, req)
    handler.do_POST()
    code, payload = _read_json_response(req.wfile)
    assert code == 401


# ===========================================================================
# C3: ensure_model_loaded behavior
# ===========================================================================


def test_ensure_model_loaded_no_op_when_already_loaded(router_with_token, monkeypatch):
    """C3: When the backend state file says the requested model is already
    loaded, ensure_model_loaded returns without calling restart_with_model."""
    # Backend state file claims "m" is already loaded.
    monkeypatch.setattr(router_with_token, "read_current_model_from_backend", lambda: "m")

    restart_calls = []

    def fail_restart(model):
        restart_calls.append(model)
        raise AssertionError("restart_with_model should NOT have been called")

    monkeypatch.setattr(router_with_token, "restart_with_model", fail_restart)

    router_with_token.ensure_model_loaded("m")
    assert restart_calls == [], f"restart should not have been triggered: {restart_calls}"


def test_ensure_model_loaded_triggers_restart_when_different(router_with_token, monkeypatch):
    """C3: When the loaded model differs from requested, restart_with_model
    is called with the requested name."""
    monkeypatch.setattr(router_with_token, "read_current_model_from_backend", lambda: "old-model")

    captured = []

    def fake_restart(model):
        captured.append(model)

    monkeypatch.setattr(router_with_token, "restart_with_model", fake_restart)

    router_with_token.ensure_model_loaded("new-model")
    assert captured == ["new-model"], f"expected one restart with 'new-model', got {captured}"


def test_ensure_model_loaded_lock_serializes_concurrent(router_with_token, monkeypatch):
    """C3: Two concurrent callers must serialize through restart_lock —
    the second caller waits for the first to finish before entering."""
    monkeypatch.setattr(router_with_token, "read_current_model_from_backend", lambda: "old")
    in_flight = {"count": 0, "max_concurrent": 0, "current": 0}
    completion_event = threading.Event()
    second_started = threading.Event()

    def slow_restart(model):
        in_flight["current"] += 1
        in_flight["count"] += 1
        if in_flight["current"] > in_flight["max_concurrent"]:
            in_flight["max_concurrent"] = in_flight["current"]
        if in_flight["count"] == 1:
            # First call: signal we're inside, wait for the second to attempt.
            time.sleep(0.3)
        else:
            # Second call: we're past the lock; mark that we got here.
            second_started.set()
        in_flight["current"] -= 1
        completion_event.set()

    monkeypatch.setattr(router_with_token, "restart_with_model", slow_restart)

    t1 = threading.Thread(target=router_with_token.ensure_model_loaded, args=("m1",))
    t1.start()
    # Give t1 a moment to acquire the lock and enter restart.
    time.sleep(0.05)

    t2 = threading.Thread(target=router_with_token.ensure_model_loaded, args=("m2",))
    t2.start()

    t1.join(timeout=5)
    t2.join(timeout=5)

    # At no point should two restarts have been in flight concurrently.
    assert in_flight["max_concurrent"] <= 1, (
        f"expected serialized restarts, got max_concurrent={in_flight['max_concurrent']}"
    )
    # Both threads must have completed.
    assert not t1.is_alive() and not t2.is_alive()


# ===========================================================================
# C4: restart_with_model behavior
# ===========================================================================


def test_restart_with_model_calls_write_env_first(router_with_token, monkeypatch):
    """C4: write_env_file is called BEFORE any subprocess invocation."""
    call_order = []

    def fake_write_env(path, name):
        call_order.append(("write_env_file", path, name))

    def fake_run(argv, *args, **kwargs):
        call_order.append(("subprocess.run", tuple(argv)))
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(router_with_token, "write_env_file", fake_write_env)
    monkeypatch.setattr(router_with_token.subprocess, "run", fake_run)
    monkeypatch.setattr(router_with_token, "wait_for_health", lambda timeout=120, interval=2: True)

    router_with_token.restart_with_model("newmodel")

    assert len(call_order) >= 2
    assert call_order[0][0] == "write_env_file", f"write_env_file was not first: {call_order}"
    assert call_order[1][0] == "subprocess.run", f"subprocess.run did not follow write_env_file: {call_order}"
    # Specifically: write_env_file got our model name.
    assert call_order[0][2] == "newmodel"


def test_restart_with_model_calls_subprocess_run_with_timeout(router_with_token, monkeypatch):
    """C4: The primary subprocess.run call uses timeout=30."""
    captured = []

    def fake_run(argv, *args, **kwargs):
        captured.append({"argv": list(argv), "kwargs": dict(kwargs)})
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(router_with_token.subprocess, "run", fake_run)
    monkeypatch.setattr(router_with_token, "write_env_file", lambda p, n: None)
    monkeypatch.setattr(router_with_token, "wait_for_health", lambda timeout=120, interval=2: True)

    router_with_token.restart_with_model("m")

    # The very first subprocess call is the primary restart.
    first = captured[0]
    assert first["kwargs"].get("timeout") == 30, f"primary restart had no timeout=30: {first}"


def test_restart_with_model_raises_runtime_error_when_both_paths_fail(router_with_token, monkeypatch):
    """C4: If primary restart returns non-zero AND the sudo fallback also
    returns non-zero, the caller must see a RuntimeError."""
    responses = [
        # Primary restart fails.
        type("R", (), {"returncode": 1, "stdout": "", "stderr": "primary failed"})(),
        # Sudo fallback also fails.
        type("R", (), {"returncode": 1, "stdout": "", "stderr": "sudo failed"})(),
    ]

    def fake_run(argv, *args, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr(router_with_token.subprocess, "run", fake_run)
    monkeypatch.setattr(router_with_token, "write_env_file", lambda p, n: None)

    with pytest.raises(RuntimeError):
        router_with_token.restart_with_model("m")


# ===========================================================================
# C5: wait_for_health behavior
# ===========================================================================


def test_wait_for_health_returns_true_on_first_200(router_with_token, monkeypatch):
    """C5: When urlopen returns a 200 status, wait_for_health returns True."""
    import urllib.request

    class _Resp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=None: _Resp())

    assert router_with_token.wait_for_health(timeout=5, interval=0.1) is True


def test_wait_for_health_returns_false_after_timeout(router_with_token, monkeypatch):
    """C5: When urlopen keeps failing (no 200), wait_for_health returns False."""
    import urllib.request

    def always_fail(url, timeout=None):
        raise IOError("backend down")

    monkeypatch.setattr(urllib.request, "urlopen", always_fail)

    assert router_with_token.wait_for_health(timeout=1, interval=0.2) is False


def test_wait_for_health_respects_configurable_timeout(router_with_token, monkeypatch):
    """C5: wait_for_health honors the timeout argument — short timeout
    returns False fast even when the backend never comes up."""
    import urllib.request

    def always_fail(url, timeout=None):
        raise IOError("backend down")

    monkeypatch.setattr(urllib.request, "urlopen", always_fail)

    start = time.time()
    result = router_with_token.wait_for_health(timeout=0.5, interval=0.1)
    elapsed = time.time() - start

    assert result is False
    # Allow generous slack (CI jitter); the key assertion is "did not run forever".
    assert elapsed < 5.0, f"wait_for_health ran too long: {elapsed:.2f}s"


# ===========================================================================
# C6: load_available_models / read_current_model_from_backend env wiring
# ===========================================================================


def test_load_available_models_uses_MODELS_YAML_env_var(tmp_path, monkeypatch):
    """C6: load_available_models reads MODELS_YAML from env at call time."""
    yaml_path = tmp_path / "models.yaml"
    yaml_path.write_text("models:\n  - name: alpha\n  - name: beta\n")

    # Load fresh router with this MODELS_YAML path set in env.
    monkeypatch.setenv("MODELS_YAML", str(yaml_path))
    monkeypatch.setenv("ROUTER_API_TOKEN", "phase-c-test-token-1234567890")
    mod = _load_router(
        MODELS_YAML=str(yaml_path),
        ROUTER_API_TOKEN="phase-c-test-token-1234567890",
    )

    result = mod.load_available_models()
    assert result == {"alpha", "beta"}


def test_read_current_model_from_backend_uses_BACKEND_STATE_FILE_env_var(
    tmp_path, monkeypatch
):
    """C6: read_current_model_from_backend reads BACKEND_STATE_FILE from env."""
    state_file = tmp_path / "backend.model"
    state_file.write_text("currently-loaded-model\n")

    monkeypatch.setenv("BACKEND_STATE_FILE", str(state_file))
    monkeypatch.setenv("ROUTER_API_TOKEN", "phase-c-test-token-1234567890")
    mod = _load_router(
        BACKEND_STATE_FILE=str(state_file),
        ROUTER_API_TOKEN="phase-c-test-token-1234567890",
    )

    result = mod.read_current_model_from_backend()
    assert result == "currently-loaded-model"


# ===========================================================================
# C7: main() / defaults / imports
# ===========================================================================


def test_default_ROUTER_PORT_is_64010():
    """C7: With no ROUTER_PORT in env, the module default is 64010."""
    # Clear all relevant env so the module-level defaults are exercised.
    for key in ("ROUTER_PORT", "ROUTER_API_TOKEN"):
        os.environ.pop(key, None)
    mod = _load_router(
        ROUTER_PORT=None,
        ROUTER_API_TOKEN=None,
    )
    assert mod.ROUTER_PORT == 64010


def test_default_CUDA_SERVE_URL_is_localhost_64000():
    """C7: With no CUDA_SERVE_URL in env, the module default is :64000."""
    os.environ.pop("CUDA_SERVE_URL", None)
    mod = _load_router(CUDA_SERVE_URL=None, ROUTER_API_TOKEN=None)
    assert mod.CUDA_SERVE_URL == "http://127.0.0.1:64000"


def test_main_smoke_test(router_with_token, monkeypatch, capsys):
    """C7: main() runs without crashing when its side-effects are stubbed
    (server bind + health probe), and prints the documented startup banner."""
    class _Server:
        def __init__(self, *a, **k): pass
        def serve_forever(self): pass
        def shutdown(self): pass

    monkeypatch.setattr(router_with_token, "ThreadingHTTPServer", _Server)
    monkeypatch.setattr(router_with_token, "load_available_models", lambda: {"alpha", "beta"})
    monkeypatch.setattr(router_with_token, "read_current_model_from_backend", lambda: None)

    router_with_token.main()
    out = capsys.readouterr().out
    # Banner lines that operators rely on.
    assert "starting on" in out
    assert "backend:" in out
    assert "env file:" in out
    assert "known models:" in out
    assert "API token" in out
    # Security (audit 2026-08-18): the token is REDACTED in the log line —
    # only a fingerprint (first 4 + ... + last 4) is emitted. The full token
    # is NOT in the log. Operators verify the token is set by comparing
    # the fingerprint against their secrets manager.
    fp = f"{router_with_token.API_TOKEN[:4]}...{router_with_token.API_TOKEN[-4:]}"
    assert fp in out, f"expected fingerprint {fp!r} in log output"
    assert router_with_token.API_TOKEN not in out, (
        "security regression: full API token leaked into log output"
    )
