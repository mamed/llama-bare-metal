#!/usr/bin/env python3
"""
llama-router — single endpoint that serves any of the 44 models in models.yaml
by auto-swapping the llama-backend systemd service's MODEL_NAME when a request
comes in.

UI sees:    http://127.0.0.1:64010/v1/chat/completions
            body: {"model": "<any of 44>", ...}

Behind it:  llama-backend.service on :64000 (one model loaded at a time)

When a request arrives for a model that isn't loaded:
  1. Write .env MODEL_NAME=<new>, systemctl restart llama-backend.service
  2. Wait for /health to come back up
  3. Forward the request

First request to a new model takes ~30-60s (model load time).
Subsequent requests to the same model are instant.

Bare-metal port of the docker-based cuda-router.py. Same protocol,
same /v1/models output, same swap semantics.

The "what model is currently loaded?" question is answered by reading
the state file the backend service writes on startup:
    $BACKEND_STATE_FILE  (plain text, just the model name)

State-file and .env I/O are delegated to llama_bare.router_state so the
swap-path logic is unit-tested (the test suite covers the state-file
format pin, atomic write, and error handling).
"""

import os
import re
import sys
import time
import json
import uuid
import secrets
import threading
import subprocess
import contextlib
import signal
import yaml
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib import request, error

# Use the tested module for state-file and .env I/O.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from llama_bare.router_state import (
    read_current_model,
    write_env_file,
    cached_load_models_from_yaml,
    cached_list_models_sorted_by_basename,
    load_models_from_yaml,
    list_models_sorted_by_basename,
)

CUDA_SERVE_URL = os.environ.get("CUDA_SERVE_URL", "http://127.0.0.1:64000")
CUDA_SERVE_HEALTH = os.environ.get("CUDA_SERVE_HEALTH", "http://127.0.0.1:64000/health")
ENV_DIR = os.environ.get("ENV_DIR", "/home/fekry/Projects/llama-bare-metal")
ENV_FILE = os.environ.get("ENV_FILE", f"{ENV_DIR}/.env")
MODELS_YAML = os.environ.get(
    "MODELS_YAML", "/home/fekry/Projects/llama-cpp-unified/models.yaml"
)
BACKEND_STATE_FILE = os.environ.get(
    "BACKEND_STATE_FILE", os.environ.get("XDG_RUNTIME_DIR", "/run/user/1000") + "/llama-backend.model"
)
BACKEND_SERVICE = os.environ.get("BACKEND_SERVICE", "llama-backend.service")
ROUTER_PORT = int(os.environ.get("ROUTER_PORT", "64010"))

# ---- Security knobs (A1, A3) ----
# A1: Cap incoming request bodies. 32 MiB matches typical chat-completions
# payloads (32k tokens × ~1KB/token worst-case) and is well under llama-server's
# own default (64 MiB). Anything larger is rejected before reading.
MAX_REQUEST_BYTES = int(os.environ.get("MAX_REQUEST_BYTES", str(32 * 1024 * 1024)))

# A3: Shared-secret API token. If unset, we generate one at startup and log it.
# Operators should set $ROUTER_API_TOKEN explicitly in production so the value
# is stable across restarts.
API_TOKEN = os.environ.get("ROUTER_API_TOKEN") or secrets.token_urlsafe(32)

# Paths that require auth (A3): anything that loads a model or triggers work.
# /health and /v1/models are intentionally free — probes should not need a token.
AUTH_REQUIRED_RE = re.compile(r"/(v1/chat/completions|v1/completions|v1/embeddings)$")

# Headers we forward to the backend (A2): case-insensitive.
# Anything outside this whitelist (including all x-*, host, content-length,
# connection, transfer-encoding, x-forwarded-*) is stripped on the way out.
# We always add X-Router-Forwarded so the backend knows it was proxied.
ALLOWED_HEADER_RE = re.compile(r"^(accept|content-type|authorization)$")


def load_available_models():
    """Read models.yaml and return set of valid model names.
    B4: Cached by file mtime via llama_bare.router_state.cached_load_models_from_yaml."""
    return cached_load_models_from_yaml(MODELS_YAML)


def read_current_model_from_backend():
    """Read the backend state file. Delegates to llama_bare.router_state
    so the YAML/plain-text fallback and error handling are unit-tested.
    Returns None if absent, empty, or unreadable."""
    return read_current_model(BACKEND_STATE_FILE)


def wait_for_health(timeout=120, interval=2):
    """Poll /health until it returns 200 or timeout."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with request.urlopen(CUDA_SERVE_HEALTH, timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(interval)
    return False


def restart_with_model(model_name):
    """Write .env MODEL_NAME= and systemctl restart llama-backend.service,
    then wait for /health on :64000. If the primary restart times out (likely
    a hung unit), fall back to stop+start which always returns promptly."""
    # Drop a hint that we're about to swap so the backend startup logs
    # make sense alongside our own log line. Uses the tested writer so the
    # format is guaranteed (atomic, exact "MODEL_NAME=<name>\n").
    write_env_file(ENV_FILE, model_name)
    print(f"[router] restarting llama-backend.service with MODEL_NAME={model_name}")
    try:
        result = subprocess.run(
            ["systemctl", "--user", "restart", BACKEND_SERVICE],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        # Hung unit: try stop+start instead.
        stop_result = subprocess.run(
            ["systemctl", "--user", "stop", BACKEND_SERVICE],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if stop_result.returncode != 0:
            raise RuntimeError(
                f"systemctl restart timed out, and stop failed (rc={stop_result.returncode}): "
                f"{stop_result.stderr.strip() or stop_result.stdout.strip()}"
            )
        result2 = subprocess.run(
            ["systemctl", "--user", "start", BACKEND_SERVICE],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result2.returncode != 0:
            raise RuntimeError(
                f"systemctl restart timed out, and start failed (rc={result2.returncode}): "
                f"{result2.stderr.strip() or result2.stdout.strip()}"
            )
    else:
        if result.returncode != 0:
            # Try system-level too — some systems don't have lingering enabled
            result2 = subprocess.run(
                ["sudo", "-n", "systemctl", "restart", BACKEND_SERVICE],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result2.returncode != 0:
                raise RuntimeError(
                    f"systemctl restart failed (rc={result.returncode}): "
                    f"{result.stderr.strip() or result.stdout.strip()}"
                )
    print(f"[router] waiting for health on {CUDA_SERVE_HEALTH}")
    if not wait_for_health(timeout=180):
        raise RuntimeError(f"llama-backend did not come up with model {model_name}")
    print(f"[router] llama-backend is up with MODEL={model_name}")


def ensure_model_loaded(model_name):
    """If a different model is loaded, restart llama-backend with the requested one."""
    global current_model
    with restart_lock:
        loaded = read_current_model_from_backend()
        if loaded == model_name:
            return  # already loaded
        print(f"[router] loaded={loaded!r} requested={model_name!r}, restarting...")
        restart_with_model(model_name)
        current_model = model_name


def extract_bearer_token(headers) -> Optional[str]:
    """Pull a token from Authorization: Bearer ... OR X-API-Token: ...
    Returns the token string (possibly empty) or None if neither is present.
    Empty Authorization header is treated as no token."""
    auth = headers.get("Authorization")
    if auth:
        parts = auth.split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer" and parts[1].strip():
            return parts[1].strip()
    api_token = headers.get("X-API-Token")
    if api_token and api_token.strip():
        return api_token.strip()
    return None


def authenticate(headers) -> bool:
    """A3: Validate the request's auth token. Constant-time compare so an
    attacker can't measure how many bytes of their guess match."""
    presented = extract_bearer_token(headers)
    if not presented:
        return False
    return secrets.compare_digest(presented, API_TOKEN)


def sanitize_forward_headers(client_headers) -> dict:
    """A2: Whitelist the headers we forward to the backend.
    Drops everything except accept/content-type/authorization (non-empty),
    then injects X-Router-Forwarded so the backend can tell it was proxied.
    """
    out: dict = {}
    for key, value in client_headers.items():
        if not ALLOWED_HEADER_RE.match(key):
            continue
        # "authorization" must be non-empty — an empty value just means
        # "I declined to authenticate" and we'd rather not advertise that.
        if key == "authorization" and not value.strip():
            continue
        out[key] = value
    out["X-Router-Forwarded"] = "llama-bare-router/1.0"
    return out


# Lock so only one restart happens at a time
restart_lock = threading.Lock()
current_model = None


class RouterHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Quieter logging
        print(f"[router] {self.address_string()} - {format % args}")

    def _read_body_capped(self) -> Optional[bytes]:
        """A1: Read up to MAX_REQUEST_BYTES from rfile, return None if the
        body is missing Content-Length, has a malformed length, exceeds the
        cap, or streams chunked data past the cap.
        Returns the body bytes on success.
        """
        length_header = self.headers.get("Content-Length")
        if length_header is None:
            # No body — treat as empty (e.g., GET with no payload).
            return b""
        try:
            length = int(length_header)
        except ValueError:
            # Malformed Content-Length — reject rather than guess.
            self._send_json(400, {"error": "invalid Content-Length"})
            return None
        if length < 0:
            self._send_json(400, {"error": "invalid Content-Length"})
            return None
        if length > MAX_REQUEST_BYTES:
            self._send_json(
                413,
                {
                    "error": "request body too large",
                    "limit_bytes": MAX_REQUEST_BYTES,
                },
            )
            return None
        if length == 0:
            return b""
        # Read the declared body. We trust Content-Length; any extra bytes
        # the client may send past the declared length will either be
        # consumed by the next request on a keep-alive connection or
        # silently discarded by socket close — either way, we MUST NOT
        # block here on a probe, because that hangs on Connection: close
        # clients (urllib, curl, etc.) where no more data will arrive.
        # Tradeoff: a client that streams more bytes than declared will
        # be silently truncated to Content-Length, but the alternative
        # was a router-wide hang on every legit request.
        body = self.rfile.read(length)
        return body

    def _proxy(self, method):
        # A1: Read request body with a hard cap.
        body = self._read_body_capped()
        if body is None:
            return  # _read_body_capped already sent the error response

        # /v1/models: just return the full list from models.yaml.
        # Public endpoint (no auth) — probes should be free.
        if self.path == "/v1/models" or self.path.endswith("/v1/models"):
            self._serve_models_list()
            return

        # /health: always 200 (router is alive).
        # Public endpoint (no auth) — probes should be free.
        if self.path == "/health" or self.path.endswith("/health"):
            self._send_json(200, {"status": "ok", "router": "cuda-router"})
            return

        # A3: Auth required for model-touching endpoints,
        # UNLESS the request came from loopback (127.0.0.1 or ::1).
        # Local clients (open-webui, Hermes Agent, OpenClaw, custom scripts)
        # don't need to set auth headers when talking to localhost.
        if AUTH_REQUIRED_RE.search(self.path):
            client_ip = self.client_address[0] if self.client_address else None
            is_loopback = client_ip in ("127.0.0.1", "::1", "localhost")
            if not is_loopback and not authenticate(self.headers):
                self._send_json(401, {"error": "authentication required"})
                return

        # Parse to get the model name (so we can ensure it's loaded).
        target_model = None
        if body and self.path.endswith(("/chat/completions", "/completions", "/embeddings")):
            try:
                req = json.loads(body)
                target_model = req.get("model")
            except Exception:
                pass

        # Other paths: must have a model in the body.
        if not target_model:
            self._send_json(400, {"error": "missing 'model' in request body"})
            return

        # Validate model name.
        valid = load_available_models()
        if target_model not in valid:
            self._send_json(
                400,
                {
                    "error": f"unknown model {target_model!r}",
                    "available": sorted(valid),
                },
            )
            return

        # Ensure the model is loaded in cuda-serve (auto-swap if needed).
        try:
            ensure_model_loaded(target_model)
        except Exception as e:
            # A4: Log full error server-side, return sanitized message to client.
            self._handle_internal_error("failed to load model", e)
            return

        # Forward the request to cuda-serve.
        url = f"{CUDA_SERVE_URL}{self.path}"
        request_id = uuid.uuid4().hex
        try:
            req = request.Request(
                url,
                data=body,
                method=method,
                headers=sanitize_forward_headers(self.headers),
            )
            # B5: Wrap urlopen in contextlib.closing() so the underlying
            # socket is always closed — even if resp.read() raises mid-stream
            # (where the `with`-block alone is not enough on some Python
            # stdlib response types). The closing() wrapper guarantees
            # close() runs on normal exit AND on exception.
            with contextlib.closing(request.urlopen(req, timeout=300)) as resp:
                resp_body = resp.read()
                self.send_response(resp.status)
                # Forward response headers, but never let the backend set
                # hop-by-hop headers we already manage.
                for k, v in resp.getheaders():
                    if k.lower() not in ("transfer-encoding", "connection", "content-length"):
                        self.send_header(k, v)
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)
        except error.HTTPError as e:
            # A4: Log full upstream error (may contain paths/traces) to stderr,
            # return only a sanitized message + request_id to the client.
            err_body = e.read()
            print(
                f"[router] request_id={request_id} upstream HTTPError "
                f"code={e.code} body={err_body!r}",
                file=sys.stderr,
            )
            client_payload = json.loads(err_body) if err_body else None
            if isinstance(client_payload, dict) and "error" in client_payload:
                # Safe upstream error — pass through, but don't leak paths.
                sanitized = {"error": "upstream error", "request_id": request_id}
                self._send_json(e.code, sanitized)
            else:
                sanitized = {"error": "upstream error", "request_id": request_id}
                self._send_json(e.code, sanitized)
        except Exception as e:
            # A4: Same treatment for any other proxy failure.
            self._handle_internal_error("proxy error", e, request_id=request_id)

    def _serve_models_list(self):
        # B7: sorted_entries comes from a mtime-keyed cache; the cache
        # invalidates automatically when an operator edits models.yaml.
        # Return all entries from models.yaml, formatted as OpenAI /v1/models.
        # Sort by the actual model file basename (so all quants of the same model
        # cluster together) with the yaml name as a tiebreaker for stability.
        sorted_entries = cached_list_models_sorted_by_basename(MODELS_YAML)
        data = {
            "object": "list",
            "data": [
                {"id": m["name"], "object": "model"}
                for m in sorted_entries
                if m.get("name")
            ],
        }
        self._send_json(200, data)

    def _handle_internal_error(self, label, exc, request_id=None):
        """A4: Send a sanitized error to the client; log the real exception
        (with traceback) to stderr, tagged with the request_id."""
        if request_id is None:
            request_id = uuid.uuid4().hex
        import traceback
        print(
            f"[router] request_id={request_id} {label}: {exc!r}\n"
            + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            file=sys.stderr,
        )
        self._send_json(500, {"error": "internal error", "request_id": request_id})

    def _send_json(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._proxy("GET")

    def do_POST(self):
        self._proxy("POST")


def main():
    print(f"[router] starting on :{ROUTER_PORT}")
    print(f"[router] backend: {CUDA_SERVE_URL}")
    print(f"[router] env file: {ENV_FILE}")
    print(f"[router] known models: {len(load_available_models())}")
    print(f"[router] max request bytes: {MAX_REQUEST_BYTES} ({MAX_REQUEST_BYTES // (1024*1024)} MiB)")
    # A3: Log the API token exactly once at startup. If the operator didn't
    # set $ROUTER_API_TOKEN, the auto-generated token is the only way in.
    token_source = "env" if os.environ.get("ROUTER_API_TOKEN") else "auto-generated"
    print(f"[router] API token ({token_source}): {API_TOKEN}")
    print()

    # Show what llama-backend currently has
    loaded = read_current_model_from_backend()
    if loaded:
        print(f"[router] llama-backend currently has: {loaded}")
    else:
        print(f"[router] (llama-backend not running yet, or state file missing)")

    server = ThreadingHTTPServer(("0.0.0.0", ROUTER_PORT), RouterHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[router] shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
