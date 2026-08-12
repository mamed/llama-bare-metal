# llama-bare-metal

Bare-metal systemd service that hosts a single llama.cpp model at a time. The active
model is selected by writing `MODEL_NAME=...` into `.env` (use `.env.example`
for the format). A companion router (`llama-router.py` + `llama-router.service`)
proxies requests to whichever backend is up and handles model swaps.

## Layout

```
.env                     # active MODEL_NAME=... (gitignored)
.env.example             # template
launcher.sh              # builds the universal flag list (DISABLE_* envs)
llama-backend.sh         # env wrapper, publishes state file, exec launcher
llama-router.py          # HTTP router on 64010 -> backend 64000
src/llama_bare/          # Python module (build_args, path_translate, router_state)
systemd/                 # unit files (llama-backend.service, llama-router.service)
scripts/                 # tune_max_ctx, tune_gpu_layers, rewrite_mmproj_to_q4_0
tests/                   # pytest (100% branch coverage on launcher_config)
monitoring/              # prometheus + grafana provisioning
```

## Install

```bash
# 1. Copy service files into place
mkdir -p ~/.config/systemd/user
cp systemd/llama-backend.service systemd/llama-router.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now llama-backend.service llama-router.service

# 2. Point CONFIG_FILE and MODELS_DIR at your registry
cp .env.example .env
echo "MODEL_NAME=unsloth-gemma-4-26b-a4b-it-ud-iq2-m" > .env

# 3. Swap to a model
echo "MODEL_NAME=unsloth-qwen3.6-35b-a3b-ud-iq2-m" > .env
systemctl --user restart llama-backend.service
```

## Universal flags (apply to every model — see launcher.sh)

`-ngl 99 --ctx-size <from yaml> --threads 16 --threads-batch 16 --batch-size 2048`
`--no-warmup --kv-unified --cache-idle-slots --no-mmproj-offload --no-kv-offload`
`--poll 0 --cache-ram 16384 --slot-save-path /tmp/llama-slots --metrics`
`--spec-type ngram-cache -ctk q4_0 -ctv q4_0 --cont-batching --parallel 1 -fa on`
`--log-verbosity 0 --log-timestamps --spec-draft-p-split 0.10 --reasoning-preserve`

Server-level defaults: `--reasoning on --reasoning-budget 16384 --agent --jinja`

Every flag has a `DISABLE_*=true` env-var to turn it off per launch.

## Tests

```bash
pytest tests/ -v
```

Coverage on `launcher_config.py` is 100% branch.
