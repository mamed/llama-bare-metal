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
import sys
import time
import json
import threading
import subprocess
import yaml
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import request, error

# Use the tested module for state-file and .env I/O.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from llama_bare.router_state import (
    read_current_model,
    write_env_file,
    load_models_from_yaml,
    list_models_sorted_by_basename,
)

CUDA_SERVE_URL = os.environ.get("CUDA_SERVE_URL", "http://127.0.0.1:64000")
CUDA_SERVE_HEALTH = os.environ.get("CUDA_SERVE_HEALTH", "http://127.0.0.1:64000/health")
ENV_DIR = os.environ.get("ENV_DIR", "/home/fekry/llama-bare-metal")
ENV_FILE = os.environ.get("ENV_FILE", f"{ENV_DIR}/.env")
MODELS_YAML = os.environ.get(
    "MODELS_YAML", "/home/fekry/llama-cpp-docker/llama-unified/models.yaml"
)
BACKEND_STATE_FILE = os.environ.get(
    "BACKEND_STATE_FILE", os.environ.get("XDG_RUNTIME_DIR", "/run/user/1000") + "/llama-backend.model"
)
BACKEND_SERVICE = os.environ.get("BACKEND_SERVICE", "llama-backend.service")
ROUTER_PORT = int(os.environ.get("ROUTER_PORT", "64010"))

# Lock so only one restart happens at a time
restart_lock = threading.Lock()
current_model = None


def load_available_models():
    """Read models.yaml and return set of valid model names.
    Delegates to llama_bare.router_state.load_models_from_yaml (tested)."""
    return load_models_from_yaml(MODELS_YAML)


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
    then wait for /health on :64000."""
    # Drop a hint that we're about to swap so the backend startup logs
    # make sense alongside our own log line. Uses the tested writer so the
    # format is guaranteed (atomic, exact "MODEL_NAME=<name>\n").
    write_env_file(ENV_FILE, model_name)
    print(f"[router] restarting llama-backend.service with MODEL_NAME={model_name}")
    result = subprocess.run(
        ["systemctl", "--user", "restart", BACKEND_SERVICE],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # Try system-level too — some systems don't have lingering enabled
        result2 = subprocess.run(
            ["sudo", "-n", "systemctl", "restart", BACKEND_SERVICE],
            capture_output=True,
            text=True,
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


class RouterHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Quieter logging
        print(f"[router] {self.address_string()} - {format % args}")

    def _proxy(self, method):
        # Read request body
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""

        # Parse to get the model name (so we can ensure it's loaded)
        target_model = None
        if body and self.path.endswith(("/chat/completions", "/completions", "/embeddings")):
            try:
                req = json.loads(body)
                target_model = req.get("model")
            except Exception:
                pass

        # /v1/models: just return the full list from models.yaml
        if self.path == "/v1/models" or self.path.endswith("/v1/models"):
            self._serve_models_list()
            return

        # /health: always 200 (router is alive)
        if self.path == "/health" or self.path.endswith("/health"):
            self._send_json(200, {"status": "ok", "router": "cuda-router"})
            return

        # Other paths: must have a model in the body
        if not target_model:
            self._send_json(400, {"error": "missing 'model' in request body"})
            return

        # Validate model name
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

        # Ensure the model is loaded in cuda-serve (auto-swap if needed)
        try:
            ensure_model_loaded(target_model)
        except Exception as e:
            self._send_json(503, {"error": f"failed to load model: {e}"})
            return

        # Forward the request to cuda-serve
        url = f"{CUDA_SERVE_URL}{self.path}"
        try:
            req = request.Request(
                url,
                data=body,
                method=method,
                headers={
                    k: v for k, v in self.headers.items()
                    if k.lower() not in ("host", "content-length")
                },
            )
            with request.urlopen(req, timeout=300) as resp:
                resp_body = resp.read()
                self.send_response(resp.status)
                # Forward response headers
                for k, v in resp.getheaders():
                    if k.lower() not in ("transfer-encoding", "connection"):
                        self.send_header(k, v)
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)
        except error.HTTPError as e:
            err_body = e.read()
            self._send_json(e.code, json.loads(err_body) if err_body else {"error": str(e)})
        except Exception as e:
            self._send_json(500, {"error": f"proxy error: {e}"})

    def _serve_models_list(self):
        # Return all entries from models.yaml, formatted as OpenAI /v1/models.
        # Sort by the actual model file basename (so all quants of the same model
        # cluster together) with the yaml name as a tiebreaker for stability.
        sorted_entries = list_models_sorted_by_basename(MODELS_YAML)
        data = {
            "object": "list",
            "data": [
                {"id": m["name"], "object": "model"}
                for m in sorted_entries
                if m.get("name")
            ],
        }
        self._send_json(200, data)

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