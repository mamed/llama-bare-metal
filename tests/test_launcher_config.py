from pathlib import Path
import pytest
from llama_bare.launcher_config import build_args

# Path to the production yaml — used for integration-style tests
PROD_YAML = Path("/home/fekry/Projects/llama-cpp-unified/models.yaml")
DISK_ROOT = Path("/home/fekry/llama-models/LLM-Models")


def make_yaml(tmp_path, text):
    path = tmp_path / 'models.yaml'
    path.write_text(text)
    return path


def make_model(monkeypatch, tmp_path, rel='model.gguf'):
    model = tmp_path / rel
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_text('dummy')
    monkeypatch.setenv('MODELS_DIR', str(tmp_path))
    return model


# ---- Required-field and error paths ----

def test_missing_yaml(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_args(tmp_path / 'missing.yaml', 'x')


def test_unknown_model_name(monkeypatch, tmp_path):
    make_model(monkeypatch, tmp_path)
    yaml = make_yaml(tmp_path, 'models:\n- name: x\n  model: model.gguf\n')
    with pytest.raises(KeyError):
        build_args(yaml, 'missing')


def test_dict_model_shape(monkeypatch, tmp_path):
    """Old yaml format with single 'model:' dict (not list of models) is accepted."""
    make_model(monkeypatch, tmp_path)
    yaml = make_yaml(tmp_path, 'model:\n  name: x\n  model: model.gguf\n')
    args = build_args(yaml, 'x')
    assert args[0] == '-m'


def test_model_file_missing(monkeypatch, tmp_path):
    """Resolving a model whose .gguf doesn't exist on disk raises FileNotFoundError."""
    yaml = make_yaml(tmp_path, 'models:\n- name: x\n  model: /nope/missing.gguf\n')
    with pytest.raises(FileNotFoundError):
        build_args(yaml, 'x')


def test_entry_missing_model_field(monkeypatch, tmp_path):
    """If an entry has 'name' but no 'model' field, KeyError is raised."""
    yaml = make_yaml(tmp_path, 'models:\n- name: x\n')
    with pytest.raises(KeyError):
        build_args(yaml, 'x')


# ---- Required per-launch flags ----

def test_required_argv_basics(monkeypatch, tmp_path):
    """Every call must produce -m, --host, --port as the first 6 args."""
    model = make_model(monkeypatch, tmp_path)
    yaml = make_yaml(tmp_path, 'models:\n- name: x\n  model: model.gguf\n')
    args = build_args(yaml, 'x')
    assert args[:6] == ['-m', str(model), '--host', '0.0.0.0', '--port', '64000']


def test_host_and_port_overrides(monkeypatch, tmp_path):
    """host/port kwargs override defaults."""
    make_model(monkeypatch, tmp_path)
    yaml = make_yaml(tmp_path, 'models:\n- name: x\n  model: model.gguf\n')
    args = build_args(yaml, 'x', host='127.0.0.1', port=12345)
    assert args[:6] == ['-m', args[1], '--host', '127.0.0.1', '--port', '12345']


# ---- Numeric per-model fields ----

def test_gpu_layers_and_context_size(monkeypatch, tmp_path):
    """gpu_layers and context_size produce -ngl and --ctx-size flags."""
    make_model(monkeypatch, tmp_path)
    yaml = make_yaml(tmp_path, 'models:\n- name: x\n  model: model.gguf\n  gpu_layers: 40\n  context_size: 8192\n')
    args = build_args(yaml, 'x')
    assert ['-ngl', '40'] in [args[i:i+2] for i in range(len(args)-1)]
    assert ['--ctx-size', '8192'] in [args[i:i+2] for i in range(len(args)-1)]


def test_threads_and_parallel(monkeypatch, tmp_path):
    """threads and parallel produce --threads and --parallel flags."""
    make_model(monkeypatch, tmp_path)
    yaml = make_yaml(tmp_path, 'models:\n- name: x\n  model: model.gguf\n  threads: 8\n  parallel: 4\n')
    args = build_args(yaml, 'x')
    assert ['--threads', '8'] in [args[i:i+2] for i in range(len(args)-1)]
    assert ['--parallel', '4'] in [args[i:i+2] for i in range(len(args)-1)]


def test_ubatch_size(monkeypatch, tmp_path):
    make_model(monkeypatch, tmp_path)
    yaml = make_yaml(tmp_path, 'models:\n- name: x\n  model: model.gguf\n  ubatch_size: 256\n')
    args = build_args(yaml, 'x')
    assert ['--ubatch-size', '256'] in [args[i:i+2] for i in range(len(args)-1)]


def test_numeric_empty_values_omitted(monkeypatch, tmp_path):
    """Empty string or null numeric values produce NO flag."""
    make_model(monkeypatch, tmp_path)
    yaml = make_yaml(tmp_path, 'models:\n- name: x\n  model: model.gguf\n  gpu_layers: ""\n  threads:\n')
    args = build_args(yaml, 'x')
    assert '-ngl' not in args
    assert '--threads' not in args


# ---- Per-model optional paths and overrides ----

def test_ctk_ctv_when_present(monkeypatch, tmp_path):
    make_model(monkeypatch, tmp_path)
    yaml = make_yaml(tmp_path, 'models:\n- name: x\n  model: model.gguf\n  ctk: q4_0\n  ctv: q4_0\n')
    args = build_args(yaml, 'x')
    assert ['-ctk', 'q4_0'] in [args[i:i+2] for i in range(len(args)-1)]
    assert ['-ctv', 'q4_0'] in [args[i:i+2] for i in range(len(args)-1)]


def test_ctk_ctv_empty_omitted(monkeypatch, tmp_path):
    make_model(monkeypatch, tmp_path)
    yaml = make_yaml(tmp_path, 'models:\n- name: x\n  model: model.gguf\n  ctk: ""\n  ctv: ""\n')
    args = build_args(yaml, 'x')
    assert '-ctk' not in args and '-ctv' not in args


def test_mmproj_translates_path(monkeypatch, tmp_path):
    """mmproj field goes through translate_path and emits --mmproj flag."""
    rel = 'unsloth/gemma-4-26B/mmproj-Q4_0.gguf'
    make_model(monkeypatch, tmp_path)
    mp = tmp_path / rel
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text('x')
    yaml = make_yaml(tmp_path, 'models:\n- name: x\n  model: model.gguf\n  mmproj: /models/' + rel + '\n')
    args = build_args(yaml, 'x')
    assert ['--mmproj', str(mp)] in [args[i:i+2] for i in range(len(args)-1)]


def test_mtp_translates_path(monkeypatch, tmp_path):
    rel = 'google/gemma-4-12b/mtp.gguf'
    make_model(monkeypatch, tmp_path)
    mt = tmp_path / rel
    mt.parent.mkdir(parents=True, exist_ok=True)
    mt.write_text('x')
    yaml = make_yaml(tmp_path, 'models:\n- name: x\n  model: model.gguf\n  mtp: /models/' + rel + '\n')
    args = build_args(yaml, 'x')
    assert ['--mtp', str(mt)] in [args[i:i+2] for i in range(len(args)-1)]


def test_chat_template_file_alias(monkeypatch, tmp_path):
    """chat_template_file and jinja_file both produce --chat-template-file."""
    rel = 'qwen3-coder-native.jinja'
    make_model(monkeypatch, tmp_path)
    ct = tmp_path / rel
    ct.parent.mkdir(parents=True, exist_ok=True)
    ct.write_text('x')
    for key in ('chat_template_file', 'jinja_file'):
        yaml = make_yaml(tmp_path, f'models:\n- name: x\n  model: model.gguf\n  {key}: /models/{rel}\n')
        args = build_args(yaml, 'x')
        assert ['--chat-template-file', str(ct)] in [args[i:i+2] for i in range(len(args)-1)]


def test_override_kv_alias(monkeypatch, tmp_path):
    make_model(monkeypatch, tmp_path)
    yaml = make_yaml(tmp_path, 'models:\n- name: x\n  model: model.gguf\n  override_kv: gemma4.expert_used_count=int:8\n')
    args = build_args(yaml, 'x')
    assert ['--override-kv', 'gemma4.expert_used_count=int:8'] in [args[i:i+2] for i in range(len(args)-1)]


def test_override_kv_alias_override(monkeypatch, tmp_path):
    """'override' is an alias for 'override_kv'."""
    make_model(monkeypatch, tmp_path)
    yaml = make_yaml(tmp_path, 'models:\n- name: x\n  model: model.gguf\n  override: foo=bar\n')
    args = build_args(yaml, 'x')
    assert '--override-kv' in args
    assert 'foo=bar' in args


def test_override_tensor(monkeypatch, tmp_path):
    make_model(monkeypatch, tmp_path)
    yaml = make_yaml(tmp_path, 'models:\n- name: x\n  model: model.gguf\n  override_tensor: blk\\.1[0-9]\\.ffn_.*=CPU\n')
    args = build_args(yaml, 'x')
    assert '--override-tensor' in args


# ---- Boolean per-model fields (still allowed per-model) ----

def test_cont_batching_true_emits_flag(monkeypatch, tmp_path):
    make_model(monkeypatch, tmp_path)
    yaml = make_yaml(tmp_path, 'models:\n- name: x\n  model: model.gguf\n  cont_batching: true\n')
    args = build_args(yaml, 'x')
    assert args.count('--cont-batching') == 1


def test_cont_batching_false_omits_flag(monkeypatch, tmp_path):
    make_model(monkeypatch, tmp_path)
    yaml = make_yaml(tmp_path, 'models:\n- name: x\n  model: model.gguf\n  cont_batching: false\n')
    args = build_args(yaml, 'x')
    assert '--cont-batching' not in args


def test_flash_attention_true_emits_on(monkeypatch, tmp_path):
    make_model(monkeypatch, tmp_path)
    yaml = make_yaml(tmp_path, 'models:\n- name: x\n  model: model.gguf\n  flash_attention: true\n')
    args = build_args(yaml, 'x')
    assert ['-fa', 'on'] in [args[i:i+2] for i in range(len(args)-1)]


def test_no_mmap_true_emits_flag(monkeypatch, tmp_path):
    make_model(monkeypatch, tmp_path)
    yaml = make_yaml(tmp_path, 'models:\n- name: x\n  model: model.gguf\n  no_mmap: true\n')
    args = build_args(yaml, 'x')
    assert '--no-mmap' in args


def test_no_mmap_false_omits_flag(monkeypatch, tmp_path):
    make_model(monkeypatch, tmp_path)
    yaml = make_yaml(tmp_path, 'models:\n- name: x\n  model: model.gguf\n  no_mmap: false\n')
    args = build_args(yaml, 'x')
    assert '--no-mmap' not in args


# ---- Per-model extra_args ----

def test_extra_args_as_list(monkeypatch, tmp_path):
    make_model(monkeypatch, tmp_path)
    yaml = make_yaml(tmp_path, 'models:\n- name: x\n  model: model.gguf\n  extra_args:\n    - --foo\n    - --bar=1\n')
    args = build_args(yaml, 'x')
    assert '--foo' in args and '--bar=1' in args


def test_extra_args_as_csv_string(monkeypatch, tmp_path):
    make_model(monkeypatch, tmp_path)
    yaml = make_yaml(tmp_path, 'models:\n- name: x\n  model: model.gguf\n  extra_args: "--foo,--bar=1"\n')
    args = build_args(yaml, 'x')
    assert '--foo' in args and '--bar=1' in args


def test_function_extra_args_appended(monkeypatch, tmp_path):
    """The 'extra_args' function kwarg appends to the argv list (before server-level defaults)."""
    make_model(monkeypatch, tmp_path)
    yaml = make_yaml(tmp_path, 'models:\n- name: x\n  model: model.gguf\n')
    args = build_args(yaml, 'x', extra_args=['--tail', '--value=42'])
    # extra_args come BEFORE server-level defaults (which always come last)
    assert '--tail' in args and '--value=42' in args
    # Find the index of --tail; everything from there until first server-default flag is the extras
    tail_idx = args.index('--tail')
    # Next item should be --value=42 (extras are consecutive)
    assert args[tail_idx + 1] == '--value=42'


# ---- Server-level defaults (apply to every model) ----

def test_server_level_reasoning_default(monkeypatch, tmp_path):
    """Default: --reasoning on --reasoning-budget 16384 present in args (at end of server defaults)."""
    make_model(monkeypatch, tmp_path)
    yaml = make_yaml(tmp_path, 'models:\n- name: x\n  model: model.gguf\n')
    args = build_args(yaml, 'x')
    # Server-level defaults are appended at the very end of argv
    assert '--reasoning' in args
    assert '--reasoning-budget' in args
    # They appear together (last 4 items unless extra_args added)
    reasoning_idx = args.index('--reasoning')
    assert args[reasoning_idx:reasoning_idx + 4] == ['--reasoning', 'on', '--reasoning-budget', '16384']


def test_server_level_agent_default(monkeypatch, tmp_path):
    """Default: --agent flag present exactly once."""
    make_model(monkeypatch, tmp_path)
    yaml = make_yaml(tmp_path, 'models:\n- name: x\n  model: model.gguf\n')
    args = build_args(yaml, 'x')
    assert args.count('--agent') == 1


def test_server_level_jinja_default(monkeypatch, tmp_path):
    """Default: --jinja flag present exactly once."""
    make_model(monkeypatch, tmp_path)
    yaml = make_yaml(tmp_path, 'models:\n- name: x\n  model: model.gguf\n')
    args = build_args(yaml, 'x')
    assert args.count('--jinja') == 1


def test_disable_reasoning_env_var(monkeypatch, tmp_path):
    """DISABLE_REASONING=true suppresses --reasoning and --reasoning-budget."""
    make_model(monkeypatch, tmp_path)
    monkeypatch.setenv('DISABLE_REASONING', 'true')
    yaml = make_yaml(tmp_path, 'models:\n- name: x\n  model: model.gguf\n')
    args = build_args(yaml, 'x')
    assert '--reasoning' not in args
    assert '--reasoning-budget' not in args


def test_disable_agent_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv('DISABLE_AGENT', 'true')
    make_model(monkeypatch, tmp_path)
    yaml = make_yaml(tmp_path, 'models:\n- name: x\n  model: model.gguf\n')
    args = build_args(yaml, 'x')
    assert '--agent' not in args


def test_disable_jinja_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv('DISABLE_JINJA', 'true')
    make_model(monkeypatch, tmp_path)
    yaml = make_yaml(tmp_path, 'models:\n- name: x\n  model: model.gguf\n')
    args = build_args(yaml, 'x')
    assert '--jinja' not in args


def test_disable_env_var_must_be_exactly_true(monkeypatch, tmp_path):
    """'DISABLE_REASONING=1' should NOT disable (only 'true' is honored)."""
    make_model(monkeypatch, tmp_path)
    monkeypatch.setenv('DISABLE_REASONING', '1')
    yaml = make_yaml(tmp_path, 'models:\n- name: x\n  model: model.gguf\n')
    args = build_args(yaml, 'x')
    assert '--reasoning' in args  # still emitted


def test_all_disables_simultaneously(monkeypatch, tmp_path):
    """All three DISABLE_* env vars set → server defaults stripped."""
    make_model(monkeypatch, tmp_path)
    monkeypatch.setenv('DISABLE_REASONING', 'true')
    monkeypatch.setenv('DISABLE_AGENT', 'true')
    monkeypatch.setenv('DISABLE_JINJA', 'true')
    yaml = make_yaml(tmp_path, 'models:\n- name: x\n  model: model.gguf\n')
    args = build_args(yaml, 'x')
    assert '--reasoning' not in args
    assert '--agent' not in args
    assert '--jinja' not in args


def test_server_defaults_not_in_yaml(monkeypatch, tmp_path):
    """Even if yaml entries have reasoning/agent/jinja, they're ignored (server-level only)."""
    make_model(monkeypatch, tmp_path)
    # Set reasoning/agent/jinja in yaml — these should be IGNORED now
    yaml = make_yaml(tmp_path, 'models:\n- name: x\n  model: model.gguf\n  reasoning: false\n  reasoning_budget: 0\n  agent: false\n  jinja: false\n')
    args = build_args(yaml, 'x')
    # Server-level defaults still applied — yaml values ignored
    assert '--reasoning' in args
    assert '--reasoning-budget' in args
    assert '--agent' in args
    assert '--jinja' in args


# ---- production yaml integration tests ----

@pytest.mark.skipif(not PROD_YAML.exists(), reason="production yaml not available")
def test_prod_yaml_parses_cleanly():
    import yaml
    with open(PROD_YAML) as f:
        data = yaml.safe_load(f)
    assert "models" in data
    assert isinstance(data["models"], list)
    assert len(data["models"]) > 0


@pytest.mark.skipif(not PROD_YAML.exists(), reason="production yaml not available")
def test_prod_yaml_no_duplicate_names():
    import yaml
    with open(PROD_YAML) as f:
        data = yaml.safe_load(f)
    names = [m["name"] for m in data["models"] if "name" in m]
    duplicates = [n for n in names if names.count(n) > 1]
    assert not duplicates, f"duplicate names: {set(duplicates)}"


@pytest.mark.skipif(not PROD_YAML.exists(), reason="production yaml not available")
def test_prod_yaml_required_fields_present():
    """Every entry must have name + model + gpu_layers + context_size."""
    import yaml
    REQUIRED = ("name", "model", "gpu_layers", "context_size")
    with open(PROD_YAML) as f:
        data = yaml.safe_load(f)
    bad = []
    for m in data["models"]:
        for r in REQUIRED:
            if r not in m:
                bad.append(f"{m.get('name', '<unnamed>')}: missing {r}")
    assert not bad, "missing required fields:\n  " + "\n  ".join(bad)


@pytest.mark.skipif(not PROD_YAML.exists(), reason="production yaml not available")
def test_prod_yaml_no_removed_keys():
    """The 4 keys moved to server-level must NOT appear in any model entry."""
    import yaml
    REMOVED = {"reasoning", "reasoning_budget", "agent", "jinja"}
    with open(PROD_YAML) as f:
        data = yaml.safe_load(f)
    leaks = []
    for m in data["models"]:
        for field in REMOVED & set(m.keys()):
            leaks.append(f"{m.get('name', '<unnamed>')}: {field}={m[field]}")
    assert not leaks, f"server-level keys leaked into yaml:\n  " + "\n  ".join(leaks)


@pytest.mark.skipif(not PROD_YAML.exists(), reason="production yaml not available")
@pytest.mark.parametrize("model_name", [
    # Sample one entry per category to prove the schema works for all model families
    "google-gemma-4-12b-it-qat-q4_0",                                # google (multimodal)
    "unsloth-gemma-4-26b-a4b-it-ud-iq2-m",                           # unsloth gemma (no mmproj, big MoE)
    "lmstudio-bonsai-27b-q1_0",                                       # lmstudio with mmproj
    "unsloth-deepseek-r1-distill-qwen-32b-q2-k",                     # unsloth (medium)
    "prism-bonsai-1.7b-q1_0",                                         # prism (small)
])
def test_prod_yaml_sample_entries_build(monkeypatch, tmp_path, model_name):
    """Each sampled production entry must produce a valid argv with no duplicates."""
    monkeypatch.setenv("MODELS_DIR", str(DISK_ROOT))
    args = build_args(PROD_YAML, model_name)
    assert args[0] == "-m"
    assert "--host" in args and "--port" in args
    assert "-ngl" in args
    assert "--ctx-size" in args
    # server-level defaults always present
    assert "--reasoning" in args
    assert "--reasoning-budget" in args
    assert "--agent" in args
    assert "--jinja" in args
    # server-level defaults appear exactly once
    assert args.count("--reasoning") == 1
    assert args.count("--agent") == 1
    assert args.count("--jinja") == 1


@pytest.mark.skipif(not PROD_YAML.exists(), reason="production yaml not available")
def test_prod_yaml_all_entries_build(monkeypatch):
    """Smoke test: every entry in production yaml must build a valid argv without crashing."""
    import yaml
    monkeypatch.setenv("MODELS_DIR", str(DISK_ROOT))
    with open(PROD_YAML) as f:
        data = yaml.safe_load(f)
    failed = []
    for m in data["models"]:
        name = m.get("name", "<unnamed>")
        try:
            args = build_args(PROD_YAML, name)
            assert isinstance(args, list)
            assert args[0] == "-m"
        except Exception as e:
            failed.append(f"{name}: {e}")
    assert not failed, "entries failed to build:\n  " + "\n  ".join(failed)