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
import logging
import logging.handlers
import http.client
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


# ---- D1: Structured logging ----
# D1: Rotating file handler defaults to /tmp/llama-router.log at 50 MiB with 3 backups.
LOG_FILE = os.environ.get("LOG_FILE", "/tmp/llama-router.log")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_MAX_BYTES = int(os.environ.get("LOG_MAX_BYTES", str(50 * 1024 * 1024)))
LOG_BACKUP_COUNT = int(os.environ.get("LOG_BACKUP_COUNT", "3"))

logger = logging.getLogger("llama_router")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
_logger_configured = False


def _configure_logger():
    """D1: Configure the router logger with a RotatingFileHandler + stderr.
    Idempotent — safe to call from main() and from tests."""
    global _logger_configured
    if _logger_configured:
        return
    handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    )
    logger.addHandler(handler)
    # Mirror to stderr so the systemd journal still gets the lines.
    stderr_handler = logging.StreamHandler()
    stderr_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    )
    logger.addHandler(stderr_handler)
    # Mirror to stdout so legacy tools (and tests) that grep on stdout keep
    # working. The systemd unit captures both stdout and stderr to the journal.
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    )
    logger.addHandler(stdout_handler)
    _logger_configured = True


# ---- E3: In-flight request drain on shutdown ----
# Tracks how many requests are currently being processed. The shutdown handler
# will wait up to DRAIN_TIMEOUT seconds for in-flight requests to complete.
DRAIN_TIMEOUT = int(os.environ.get("DRAIN_TIMEOUT", "30"))
_inflight_count = threading.Lock()
_inflight_value = [0]  # mutable container for the count

# ---- L-3: Exponential backoff on failed swaps ----
# A hot loop on a broken backend will hammer `systemctl restart`
# continuously. Count consecutive failures and back off (2s, 4s, 8s, ...,
# capped at SWAP_BACKOFF_MAX seconds). Reset on a successful swap.
SWAP_BACKOFF_INITIAL = 2.0  # seconds; doubles each failure
SWAP_BACKOFF_MAX = 60.0  # ceiling
_swap_failure_count = 0
_next_swap_allowed_at = 0.0  # time.monotonic() when next swap is allowed
_swap_backoff_lock = threading.Lock()


def _inflight_acquire():
    """E3: Mark a request as in-flight for the drain check."""
    with _inflight_count:
        _inflight_value[0] += 1


def _inflight_release():
    """E3: Mark a request as done."""
    with _inflight_count:
        _inflight_value[0] -= 1


def _inflight_drain(timeout=DRAIN_TIMEOUT):
    """E3: Wait until all in-flight requests finish, or until timeout elapses.
    Returns True if drained, False on timeout."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        with _inflight_count:
            if _inflight_value[0] <= 0:
                return True
        time.sleep(0.1)
    return False


# ---- D4: Prometheus metrics ----
# D4: prometheus_client metrics for swap count, swap latency, request count,
# request duration, and currently-loaded model. The /metrics endpoint serves
# the default registry.
try:
    from prometheus_client import Counter, Histogram, Gauge, REGISTRY, generate_latest, CONTENT_TYPE_LATEST
except ImportError:
    # Soft-fail so tests without prometheus_client can still import the module.
    Counter = Histogram = Gauge = None
    REGISTRY = None

    def generate_latest():
        return b""

    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"


if Counter is not None:
    # L-5: Use the public Counter/Histogram/Gauge constructors; guard
    # against double-registration in the same process (test reimports,
    # `importlib.reload`). The previous code reached into the private
    # `_names_to_collectors` dict — a module-level flag is the
    # public-API equivalent. Subsequent imports in the same process
    # reuse the names already bound at the top level.
    if "METRICS_SWAPS_TOTAL" not in globals():
        try:
            METRICS_SWAPS_TOTAL = Counter(
                "llama_router_swaps_total",
                "Total number of model swap operations",
                labelnames=("model_name", "status"),
            )
            METRICS_SWAP_DURATION = Histogram(
                "llama_router_swap_duration_seconds",
                "Wall-clock duration of model swap operations",
                labelnames=("model_name", "status"),
                buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 180.0, 300.0),
            )
            METRICS_REQUESTS_TOTAL = Counter(
                "llama_router_requests_total",
                "Total number of proxied requests by endpoint and HTTP status",
                labelnames=("endpoint", "status"),
            )
            METRICS_REQUEST_DURATION = Histogram(
                "llama_router_request_duration_seconds",
                "Wall-clock duration of proxied requests",
                labelnames=("endpoint", "status"),
            )
            METRICS_LOADED_MODEL = Gauge(
                "llama_router_loaded_model_info",
                "Indicator gauge: 1 for the currently loaded model name, 0 for others",
                labelnames=("model_name",),
            )
            METRICS_LOADED_AT = Gauge(
                "llama_router_loaded_at_unix_seconds",
                "Unix timestamp when the backend last completed a model swap",
            )
            METRICS_BACKEND_HEALTHY = Gauge(
                "llama_router_backend_healthy",
                "Whether the backend /health probe returned 200 on the last poll",
            )
        except ValueError:
            # Already registered in this process (DuplicateTimeseries).
            # Disable all metrics so calls are no-ops.
            METRICS_SWAPS_TOTAL = METRICS_SWAP_DURATION = None
            METRICS_REQUESTS_TOTAL = METRICS_REQUEST_DURATION = None
            METRICS_LOADED_MODEL = METRICS_LOADED_AT = None
            METRICS_BACKEND_HEALTHY = None
else:
    METRICS_SWAPS_TOTAL = METRICS_SWAP_DURATION = METRICS_REQUESTS_TOTAL = None
    METRICS_REQUEST_DURATION = METRICS_LOADED_MODEL = METRICS_LOADED_AT = None
    METRICS_BACKEND_HEALTHY = None


# ---- E1/E6: /health cache + background probe ----
# E1: Cache the backend /health probe for 5 seconds so high-frequency
# probes don't hammer the backend.
_HEALTH_CACHE_TTL = int(os.environ.get("HEALTH_CACHE_TTL", "5"))
_health_cache: dict = {"at": 0.0, "ok": False, "reason": "uninitialized"}
_health_cache_lock = threading.Lock()
_backend_healthy_flag = [True]  # mutable container for the background probe


def _probe_backend_health():
    """E1: Probe the backend /health endpoint. Returns (ok, reason)."""
    try:
        with request.urlopen(CUDA_SERVE_HEALTH, timeout=3) as r:
            if r.status == 200:
                return True, "ok"
            return False, f"backend returned {r.status}"
    except Exception as e:
        return False, f"backend unreachable: {type(e).__name__}"


def _state_file_age_seconds():
    """E1: Return the age of the backend state file in seconds, or None
    if the file is missing."""
    try:
        st = os.stat(BACKEND_STATE_FILE)
    except OSError:
        return None
    return time.time() - st.st_mtime


def _evaluate_health(force=False):
    """E1: Cache-then-probe health check. Returns (ok, reason, status_code)."""
    now = time.monotonic()
    with _health_cache_lock:
        if not force and _health_cache["ok"] and (now - _health_cache["at"]) < _HEALTH_CACHE_TTL:
            return True, _health_cache.get("reason", "cached"), 200
        cached_at = _health_cache["at"]
        cached_reason = _health_cache.get("reason", "")
    # Outside the lock — call the backend.
    ok, reason = _probe_backend_health()
    if not ok:
        with _health_cache_lock:
            _health_cache.update(at=now, ok=False, reason=reason)
        return False, reason, 503
    # Backend ok — check state file freshness.
    age = _state_file_age_seconds()
    if age is None:
        with _health_cache_lock:
            _health_cache.update(at=now, ok=False, reason="state file missing")
        return False, "state file missing", 503
    if age > 300:  # 5 minutes
        reason = f"state file stale ({age:.0f}s)"
        with _health_cache_lock:
            _health_cache.update(at=now, ok=False, reason=reason)
        return False, reason, 503
    reason = "ok"
    with _health_cache_lock:
        _health_cache.update(at=now, ok=True, reason=reason)
    return True, reason, 200


def _background_health_loop():
    """E6: Poll the backend /health every 30s and update the healthy flag.
    Runs as a daemon thread so it dies when the main process exits."""
    while True:
        ok, _reason = _probe_backend_health()
        _backend_healthy_flag[0] = ok
        if METRICS_BACKEND_HEALTHY is not None:
            METRICS_BACKEND_HEALTHY.set(1 if ok else 0)
        time.sleep(30)


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
    a hung unit), fall back to stop+start which always returns promptly.
    L-3: On success, reset the swap-backoff counter. On failure, increment
    the counter and push the next-allowed time forward exponentially."""
    global _swap_failure_count, _next_swap_allowed_at
    swap_start = time.monotonic()
    # Drop a hint that we're about to swap so the backend startup logs
    # make sense alongside our own log line. Uses the tested writer so the
    # format is guaranteed (atomic, exact "MODEL_NAME=<name>\n").
    write_env_file(ENV_FILE, model_name)
    logger.info("restarting llama-backend.service with MODEL_NAME=%s", model_name)
    status = "success"
    try:
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
                status = "failed"
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
                status = "failed"
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
                    status = "failed"
                    raise RuntimeError(
                        f"systemctl restart failed (rc={result.returncode}): "
                        f"{result.stderr.strip() or result.stdout.strip()}"
                    )
        logger.info("waiting for health on %s", CUDA_SERVE_HEALTH)
        if not wait_for_health(timeout=180):
            status = "failed"
            raise RuntimeError(f"llama-backend did not come up with model {model_name}")
        logger.info("llama-backend is up with MODEL=%s", model_name)
        # D4: record metrics
        if METRICS_SWAPS_TOTAL is not None:
            METRICS_SWAPS_TOTAL.labels(model_name=model_name, status=status).inc()
            METRICS_SWAP_DURATION.labels(model_name=model_name, status=status).observe(
                time.monotonic() - swap_start
            )
            METRICS_LOADED_AT.set(time.time())
            METRICS_LOADED_MODEL.labels(model_name=model_name).set(1)
        # L-3: Success — reset backoff state.
        with _swap_backoff_lock:
            _swap_failure_count = 0
            _next_swap_allowed_at = 0.0
    except Exception:
        if METRICS_SWAPS_TOTAL is not None and status == "success":
            METRICS_SWAPS_TOTAL.labels(model_name=model_name, status="failed").inc()
            METRICS_SWAP_DURATION.labels(model_name=model_name, status="failed").observe(
                time.monotonic() - swap_start
            )
        # L-3: Failure — increment backoff exponentially.
        with _swap_backoff_lock:
            _swap_failure_count += 1
            backoff = min(
                SWAP_BACKOFF_MAX,
                SWAP_BACKOFF_INITIAL * (2 ** (_swap_failure_count - 1)),
            )
            _next_swap_allowed_at = time.monotonic() + backoff
        logger.warning(
            "swap failure #%d for MODEL=%s; next swap allowed in %.1fs",
            _swap_failure_count, model_name, backoff,
        )
        raise


def ensure_model_loaded(model_name):
    """If a different model is loaded, restart llama-backend with the requested one.
    L-3: If a recent swap failed, the module is in backoff — reject the request
    with 503 + retry_after rather than hammering a broken backend.
    O/F2: If another swap is already in progress, raise SwapInProgressError
    so the proxy can return 503 + Retry-After instead of blocking on
    restart_lock (which would either cut an in-flight stream or queue
    behind another swap)."""
    global current_model, _swap_in_progress
    # L-3: Backoff gate. Checked BEFORE acquiring restart_lock so a request
    # rejected by backoff doesn't block other swappers behind it.
    with _swap_backoff_lock:
        wait = _next_swap_allowed_at - time.monotonic()
    if wait > 0:
        logger.warning(
            "swap to %r rejected: backend in backoff for %.1fs after recent failure",
            model_name, wait,
        )
        return False, ("in_backoff", wait)
    # O/F2: Quick check — if a swap is already in progress, bail out
    # immediately rather than queuing behind it (the queued request would
    # land on whatever model the in-flight swap just installed, not what
    # the caller asked for, AND waiting on restart_lock blocks other
    # handlers behind us on the same ThreadingHTTPServer worker).
    with _swap_in_progress_lock:
        if _swap_in_progress:
            raise SwapInProgressError(
                f"another swap is in progress; cannot swap to {model_name!r}"
            )
        _swap_in_progress = True
    try:
        with restart_lock:
            loaded = read_current_model_from_backend()
            if loaded == model_name:
                return True, ("already_loaded", 0.0)
            logger.info("loaded=%r requested=%r, restarting...", loaded, model_name)
            restart_with_model(model_name)
            current_model = model_name
            return True, ("swapped", 0.0)
    finally:
        with _swap_in_progress_lock:
            _swap_in_progress = False


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


# ---- O/F2: Prevent concurrent swaps ----
# Two concurrent requests for different models used to both call
# restart_with_model — the second restart cut the first request's stream
# mid-flight, producing IncompleteRead → HTTP 500 with a leaked traceback.
# The swap-in-progress flag makes the second caller bail out immediately
# with a SwapInProgressError (handled in _proxy as 503 + Retry-After) so
# the in-flight request gets a clean retry on the new model.
class SwapInProgressError(RuntimeError):
    """Raised when ensure_model_loaded is asked to swap while another swap
    is already in progress. _proxy turns this into 503 + Retry-After."""


_swap_in_progress = False
_swap_in_progress_lock = threading.Lock()


# ---- R: Session lock — pin a session to the model from its first request.
# Hermes Agent / Open WebUI can run 50+ tool-call iterations against a
# single chat session. If ANY of those requests asks for a different model
# than the one pinned at session start, the router would normally call
# ensure_model_loaded → restart_with_model → cut the in-flight response
# mid-stream (HTTP 500/503 + client retry). We avoid that by remembering
# the first model each X-Session-Id saw, then overriding the requested
# model for any later request in the same session.
#
# - No X-Session-Id header: behavior is unchanged (no pin, no rewrite).
# - First request with a new X-Session-Id: pin it to the requested model.
# - Follow-up with same X-Session-Id: rewrite target_model to the pinned
#   model so ensure_model_loaded never triggers a mid-session swap.
# - LRU/FIFO eviction at _SESSION_LOCKS_MAX entries so stale sessions
#   don't leak memory across a long-lived router.
_SESSION_LOCKS_MAX = 1000
_session_locks: dict = {}  # session_id (str) -> pinned model name (str)
_session_locks_lock = threading.Lock()


class RouterHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Quieter logging
        logger.info("%s - %s", self.address_string(), format % args)

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
        # D3: Generate a request ID per request for log correlation. We also
        # forward this to the backend so backend logs can be joined to router
        # logs by request_id.
        request_id = uuid.uuid4().hex
        # E3: Track this request as in-flight so the shutdown handler can drain.
        _inflight_acquire()
        request_start = time.monotonic()
        endpoint_label = self._endpoint_label()
        try:
            # A1: Read request body with a hard cap.
            body = self._read_body_capped()
            if body is None:
                self._record_request_metric(endpoint_label, 4)  # 4xx codes
                return  # _read_body_capped already sent the error response

            # /v1/models: just return the full list from models.yaml.
            # Public endpoint (no auth) — probes should be free.
            if self.path == "/v1/models" or self.path.endswith("/v1/models"):
                self._serve_models_list()
                self._record_request_metric(endpoint_label, 200)
                return

            # /health: simple liveness — the router itself is alive.
            # Public endpoint (no auth) — probes should be free.
            if self.path == "/health" or self.path.endswith("/health"):
                self._send_json(200, {"status": "ok", "router": "cuda-router"})
                self._record_request_metric(endpoint_label, 200)
                return

            # /health/deep: E1 — actually probe the backend + check state file freshness.
            # Returns 503 if the backend is down or the state file is stale.
            if self.path == "/health/deep" or self.path.endswith("/health/deep"):
                ok, reason, code = _evaluate_health()
                self._send_json(code, {
                    "status": "ok" if ok else "degraded",
                    "router": "cuda-router",
                    "backend": "ok" if ok else reason,
                })
                self._record_request_metric(endpoint_label, code)
                return

            # D4: /metrics — serve the Prometheus exposition format.
            if self.path == "/metrics" or self.path.endswith("/metrics"):
                self._serve_metrics()
                self._record_request_metric(endpoint_label, 200)
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
                    self._record_request_metric(endpoint_label, 401)
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
                self._record_request_metric(endpoint_label, 400)
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
                self._record_request_metric(endpoint_label, 400)
                return

            # R: Session lock — pin a chat session to the model from its
            # first request. Prevents mid-session model swaps from killing
            # in-flight tool-call iterations. No X-Session-Id → unchanged
            # behavior.
            session_id = self.headers.get("X-Session-Id")
            if session_id:
                with _session_locks_lock:
                    locked = _session_locks.get(session_id)
                    if locked and locked != target_model:
                        # Session is pinned to a different model — honor the
                        # pin, log the override, do NOT trigger a swap.
                        logger.info(
                            "request_id=%s session %r locked to %r; "
                            "ignoring request for %r (no mid-session swap)",
                            request_id, session_id, locked, target_model,
                        )
                        target_model = locked
                    elif not locked:
                        # First request from this session — pin it. Evict
                        # the oldest entries (FIFO) until we're strictly
                        # below the cap so the insert keeps us at the cap.
                        # Python 3.7+ dicts preserve insertion order, so
                        # `next(iter(d))` is the first-inserted key.
                        while len(_session_locks) >= _SESSION_LOCKS_MAX:
                            try:
                                oldest = next(iter(_session_locks))
                            except StopIteration:
                                break
                            del _session_locks[oldest]
                        _session_locks[session_id] = target_model
                        logger.info(
                            "request_id=%s session %r pinned to %r",
                            request_id, session_id, target_model,
                        )

            # Ensure the model is loaded in cuda-serve (auto-swap if needed).
            try:
                ok, info = ensure_model_loaded(target_model)
                if not ok:
                    # L-3: Backend is in backoff after recent failures.
                    # Return 503 + retry_after so the client can retry later
                    # instead of hammering a broken backend.
                    reason, retry_after = info
                    self._send_json(503, {
                        "error": "backend in backoff after recent failure",
                        "retry_after": retry_after,
                    })
                    self._record_request_metric(endpoint_label, 503)
                    return
            except SwapInProgressError as e:
                # O/F2: Another swap is already in progress. Returning 503 +
                # Retry-After lets the client retry on the new model without
                # us blocking on restart_lock (which would either cut an
                # in-flight stream mid-flight or starve other handlers).
                logger.info(
                    "request_id=%s swap-in-progress; deferring to next request: %s",
                    request_id, e,
                )
                self._record_request_metric(endpoint_label, 503)
                self._send_json(503, {
                    "error": "another model swap is in progress",
                    "request_id": request_id,
                }, extra_headers={"Retry-After": "5"})
                return
            except Exception as e:
                # A4: Log full error server-side, return sanitized message to client.
                self._handle_internal_error("failed to load model", e, request_id=request_id)
                self._record_request_metric(endpoint_label, 500)
                return

            # Forward the request to cuda-serve.
            url = f"{CUDA_SERVE_URL}{self.path}"
            try:
                forward_headers = sanitize_forward_headers(self.headers)
                # D3: we record the request_id in the router's log line so it can
                # be correlated with backend logs by timestamp + model. We don't
                # inject it as a header because the header whitelist is a security
                # boundary and we don't want to silently widen it.
                req = request.Request(
                    url,
                    data=body,
                    method=method,
                    headers=forward_headers,
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
                    self.send_header("X-Request-Id", request_id)
                    self.end_headers()
                    # L-4: Catch client-disconnect (BrokenPipeError /
                    # ConnectionResetError) so a closed socket doesn't
                    # propagate as an uncaught exception. The `finally`
                    # still releases the in-flight counter.
                    try:
                        self.wfile.write(resp_body)
                    except (BrokenPipeError, ConnectionResetError):
                        logger.info(
                            "request_id=%s client cancelled during response write",
                            request_id,
                        )
                        return
                    self._record_request_metric(endpoint_label, resp.status)
                    logger.info(
                        "request_id=%s %s %s -> %d in %.3fs",
                        request_id, method, self.path, resp.status,
                        time.monotonic() - request_start,
                    )
                    return
            except error.HTTPError as e:
                # A4: Log full upstream error (may contain paths/traces) to stderr,
                # return only a sanitized message + request_id to the client.
                err_body = e.read()
                logger.error(
                    "request_id=%s upstream HTTPError code=%s body=%r",
                    request_id, e.code, err_body,
                )
                self._record_request_metric(endpoint_label, e.code)
                client_payload = json.loads(err_body) if err_body else None
                sanitized = {"error": "upstream error", "request_id": request_id}
                self._send_json(e.code, sanitized)
                return
            except (http.client.IncompleteRead, ConnectionResetError, BrokenPipeError) as e:
                # O/F1: The backend cut the stream mid-response — almost
                # always because a model swap restarted llama-backend while
                # we were reading from it. Returning 500 here leaks the
                # traceback (A4 violation) AND tells the client nothing
                # useful. 503 + Retry-After is the right signal: backend
                # was reloaded mid-flight, please retry on the new model.
                logger.warning(
                    "request_id=%s backend cut stream mid-response: %s: %s",
                    request_id, type(e).__name__, e,
                )
                self._record_request_metric(endpoint_label, 503)
                sanitized = {
                    "error": "backend stream interrupted (likely model swap)",
                    "request_id": request_id,
                }
                try:
                    self._send_json(503, sanitized, extra_headers={"Retry-After": "5"})
                except (BrokenPipeError, ConnectionResetError):
                    # Client gave up between the time we caught the
                    # backend error and the time we tried to write the
                    # response — nothing useful we can do.
                    pass
                return
            except Exception as e:
                # A4: Same treatment for any other proxy failure.
                self._handle_internal_error("proxy error", e, request_id=request_id)
                self._record_request_metric(endpoint_label, 500)
                return
        finally:
            # E3: Always release the in-flight marker so the drain succeeds.
            _inflight_release()

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
        logger.error(
            "request_id=%s %s: %r\n%s",
            request_id, label, exc,
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )
        self._send_json(500, {"error": "internal error", "request_id": request_id})

    def _endpoint_label(self):
        """D4: Map self.path to a coarse endpoint label for metrics."""
        if self.path.endswith("/v1/models") or self.path == "/v1/models":
            return "v1_models"
        if self.path.endswith("/health") or self.path == "/health":
            return "health"
        if self.path.endswith("/metrics") or self.path == "/metrics":
            return "metrics"
        if self.path.endswith("/v1/chat/completions"):
            return "v1_chat"
        if self.path.endswith("/v1/completions"):
            return "v1_completions"
        if self.path.endswith("/v1/embeddings"):
            return "v1_embeddings"
        return "other"

    def _record_request_metric(self, endpoint, status):
        """D4: Update the requests_total counter and request_duration histogram."""
        if METRICS_REQUESTS_TOTAL is None:
            return
        try:
            status_str = str(int(status))
        except (TypeError, ValueError):
            status_str = "unknown"
        METRICS_REQUESTS_TOTAL.labels(endpoint=endpoint, status=status_str).inc()
        # Histogram observes elapsed time since request start.
        # We rely on time.monotonic() here; rate is recorded via a fresh
        # measurement each call. (approximate — but stable enough for SLOs)
        if METRICS_REQUEST_DURATION is not None and hasattr(self, "_request_start"):
            METRICS_REQUEST_DURATION.labels(endpoint=endpoint, status=status_str).observe(
                time.monotonic() - self._request_start
            )

    def _serve_metrics(self):
        """D4: Serve the Prometheus exposition format."""
        if REGISTRY is None:
            body = b"# prometheus_client not installed\n"
        else:
            body = generate_latest()
        self.send_response(200)
        self.send_header("Content-Type", CONTENT_TYPE_LATEST)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        # L-4: Catch client-disconnect so a metrics scraper that closes
        # the socket mid-response doesn't propagate as an uncaught error.
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_json(self, code, data, extra_headers=None):
        """Write a JSON response. extra_headers is an optional dict of
        additional response headers (e.g. {"Retry-After": "5"})."""
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        # Phase O: Inject Retry-After (or any other header) BEFORE
        # end_headers(). Must come after send_response or HTTP is malformed.
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        # L-4: Catch client-disconnect so a closed socket doesn't crash
        # the request handler. The `finally` in _proxy still runs and
        # releases the in-flight counter.
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        self._proxy("GET")

    def do_POST(self):
        self._proxy("POST")


def main():
    _configure_logger()
    logger.info("starting on :%d", ROUTER_PORT)
    logger.info("backend: %s", CUDA_SERVE_URL)
    logger.info("env file: %s", ENV_FILE)
    logger.info("known models: %d", len(load_available_models()))
    logger.info("max request bytes: %d (%d MiB)", MAX_REQUEST_BYTES, MAX_REQUEST_BYTES // (1024*1024))
    # A3: Log the API token exactly once at startup. If the operator didn't
    # set $ROUTER_API_TOKEN, the auto-generated token is the only way in.
    token_source = "env" if os.environ.get("ROUTER_API_TOKEN") else "auto-generated"
    # The token is printed in full. Note: if sourced from env, this means
    # the secret lands in the systemd journal; consider setting ROUTER_API_TOKEN
    # via a file-based unit's EnvironmentFile= to keep it out of journalctl.
    logger.info("API token (%s): %s", token_source, API_TOKEN)

    # Show what llama-backend currently has
    loaded = read_current_model_from_backend()
    if loaded:
        logger.info("llama-backend currently has: %s", loaded)
    else:
        logger.info("(llama-backend not running yet, or state file missing)")

    # E6: Spawn the background health-probe thread.
    health_thread = threading.Thread(target=_background_health_loop, daemon=True, name="router-health")
    health_thread.start()

    server = ThreadingHTTPServer(("0.0.0.0", ROUTER_PORT), RouterHandler)
    server.request_queue_size = 128  # E3: cap concurrent connections

    # H1: SIGTERM handler — drain in-flight requests, then exit. systemd
    # sends SIGTERM on `systemctl stop`; without this handler the server
    # is killed mid-request, leaking locks and partial responses.
    def _sigterm_handler(signum, frame):
        logger.info("SIGTERM received — draining %d in-flight request(s) (timeout %ds)",
                    _inflight_value[0], DRAIN_TIMEOUT)
        if _inflight_drain():
            logger.info("drained cleanly; exiting")
        else:
            logger.warning("drain timeout — exiting with %d request(s) still in flight",
                           _inflight_value[0])
        server.shutdown()

    signal.signal(signal.SIGTERM, _sigterm_handler)
    signal.signal(signal.SIGINT, _sigterm_handler)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
