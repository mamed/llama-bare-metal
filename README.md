# llama-bare-metal

Bare-metal systemd services that host a single llama.cpp model at a time.
The active model is selected by writing `MODEL_NAME=...` into `.env`
(`.env.example` shows the format). A companion router proxies requests
to whichever backend is up and handles model swaps; a watchdog pinger
auto-restarts the backend if `/health` fails for 3 consecutive checks.

The three services run as user-level systemd (no root required):

| Service | Port | Purpose |
|---|---|---|
| `llama-backend.service` | `:64000` (127.0.0.1) | llama-server with the model from `.env` |
| `llama-router.service` | `:64010` (127.0.0.1) | OpenAI-compatible multiplexer (handles swaps) |
| `llama-backend-watcher.service` | (no port) | Health probe + auto-restart on hang |

> **Security:** all ports bind to `127.0.0.1` by default. To expose to
> LAN, set `HOST=0.0.0.0` (backend), `ROUTER_BIND=0.0.0.0` (router), and
> `ROUTER_API_TOKEN=<secret>` (router). Loopback clients bypass auth.

## Layout

```
.env                         # active MODEL_NAME=... (gitignored, mode 600)
.env.example                 # template (documents all env vars)
launcher.sh                  # builds the universal flag list (DISABLE_* envs)
llama-backend.sh             # env wrapper, publishes state file, exec launcher
llama-backend-watcher.sh     # health probe + auto-restart loop
llama-router.py              # HTTP router on 64010 -> backend 64000
src/llama_bare/              # Python module (build_args, path_translate, router_state)
systemd/                     # unit files (3 services, hardened)
restore-systemd.sh           # install all 3 services from systemd/ in one command
scripts/                     # tune_max_ctx, tune_gpu_layers, rewrite_mmproj_to_q4_0
tests/                       # pytest (218 tests passing)
monitoring/                  # prometheus + grafana provisioning
final-check.sh               # one-shot health + perf verification
```

## Install

```bash
# 1. Run the install script (installs all 3 services, enables linger)
./restore-systemd.sh

# 2. Set up the model registry + active model
cp .env.example .env
echo "MODEL_NAME=unsloth-gemma-4-26b-a4b-it-ud-iq2-m" > .env

# 3. Swap to a different model
echo "MODEL_NAME=unsloth-qwen3.6-35b-a3b-ud-iq2-m" > .env
systemctl --user restart llama-backend.service
```

`restore-systemd.sh` does this:

```bash
cp systemd/*.service ~/.config/systemd/user/
systemctl --user daemon-reload
loginctl enable-linger $USER
systemctl --user start llama-backend.service llama-backend-watcher.service llama-router.service
```

## Hardening audit (2026-08-18)

The stack has been through nine rounds of security/reliability hardening.
All fixes are documented in commit `4576c0d` ("audit: 2026-08-18 incident recovery + 9 hardening rounds").

Key changes:

- **Bound to localhost** — backend (`HOST=127.0.0.1`) and router (`ROUTER_BIND=127.0.0.1`) bind loopback only.
- **Mount guard** — `RequiresMountsFor=/home/fekry/llama-models` makes the backend unit fail-fast if the model mount is missing.
- **Boot race fixed** — the watcher has a 60-second startup grace window so it doesn't restart the backend mid-load.
- **Auto-restart on hang** — the watcher probes `/health` every 10 seconds and triggers `systemctl restart llama-backend.service` after 3 consecutive failures.
- **Swap circuit breaker** — `SWAP_CIRCUIT_TRIP_AFTER=10` failures opens the breaker for 5 minutes, preventing hot loops on broken backends.
- **API token redaction** — the auto-generated router token is logged as a fingerprint (first 4 + "..." + last 4), never the full secret.
- **Journal hygiene** — `journald` cap is 500 MB / 2-week retention. No more unbounded disk growth.
- **Fail-fast on YAML errors** — `launcher.sh` propagates Python errors so a misconfigured model file can't silently fall through to a broken state.

## Environment variables

See `.env.example` for the full list. The most important ones:

| Var | Where | Default | Purpose |
|---|---|---|---|
| `MODEL_NAME` | `.env` | (none) | which model to load (must be in `models.yaml`) |
| `HOST` | backend unit | `127.0.0.1` | backend bind address |
| `ROUTER_BIND` | router unit | `127.0.0.1` | router bind address |
| `ROUTER_API_TOKEN` | router unit | auto-generated | shared-secret bearer token (fingerprint only in logs) |
| `LOG_FILE` | router unit | `/tmp/llama-router.log` | disable with `/dev/null` to log only to the journal |
| `SWAP_CIRCUIT_TRIP_AFTER` | router unit | `10` | consecutive swap failures before the circuit breaker trips |
| `BACKEND_SERVICE` | watcher unit | `llama-backend.service` | which unit the watcher restarts |
| `HEALTH_URL` | watcher unit | `http://127.0.0.1:64000/health` | watcher's probe target |
| `STARTUP_GRACE_SEC` | watcher script | `60` | grace window after watcher start (defends against boot race) |

## Universal flags (apply to every model — see launcher.sh)

`-ngl 99 --ctx-size <from yaml> --threads 16 --threads-batch 16 --batch-size 2048`
`--no-warmup --kv-unified --cache-idle-slots --no-mmproj-offload --no-kv-offload`
`--poll 0 --cache-ram 16384 --slot-save-path /tmp/llama-slots --metrics`
`--spec-type ngram-cache -ctk q4_0 -ctv q4_0 --cont-batching --parallel 1 -fa on`
`--log-verbosity 0 --log-timestamps --spec-draft-p-split 0.10 --reasoning-preserve`

Server-level defaults: `--reasoning on --reasoning-budget 32768 --agent --jinja`

Every flag has a `DISABLE_*=true` env-var to turn it off per launch.

## Tests

```bash
pytest tests/ -v
```

218 tests pass. Coverage on `launcher_config.py` is 100% branch.

The test suite also runs shellcheck + bash -n on every bash script:

```bash
tests/test_bash_scripts.py::test_bash_script_passes_shellcheck[launcher.sh]
tests/test_bash_scripts.py::test_bash_script_passes_shellcheck[llama-backend.sh]
tests/test_bash_scripts.py::test_bash_script_passes_shellcheck[llama-backend-watcher.sh]
tests/test_bash_scripts.py::test_bash_script_passes_shellcheck[restore-systemd.sh]
```

## Verification

```bash
./final-check.sh           # one-shot health check (service, ports, GPU, tests)
```

## Files NOT in the repo

- `.env` — gitignored, mode 600 (secret).
- `.venv/` — virtual environment, gitignored.
- `__pycache__/` — Python bytecode cache.
- `~/.llama-models/` symlink target — external mount at `/run/media/fekry/OS-Shared/LLM-Models`.
