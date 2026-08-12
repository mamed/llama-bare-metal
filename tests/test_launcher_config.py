from pathlib import Path
import pytest
from llama_bare.launcher_config import build_args

# Path to the production yaml — used for integration-style tests
PROD_YAML = Path("/home/fekry/llama-cpp-docker/llama-unified/models.yaml")
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


def test_fixture_text(monkeypatch, tmp_path):
    model = make_model(monkeypatch, tmp_path, 'prism-ml/Bonsai-1.7B-gguf/Bonsai-1.7B-Q1_0.gguf')
    args = build_args(Path(__file__).parent / 'fixtures/models.yaml', 'text-only-tiny')
    assert args[:6] == ['-m', str(model), '--host', '0.0.0.0', '--port', '64000']
    assert ['-ngl', '99'] == args[6:8]
    assert '--cont-batching' in args and ['-fa', 'on'] in [args[i:i+2] for i in range(len(args)-1)] and '--agent' in args


def test_fixture_vl(monkeypatch, tmp_path):
    make_model(monkeypatch, tmp_path, 'unsloth/gemma-4-26B-A4B-it-UD-IQ2_XXS/gemma-4-26B-it-UD-IQ2_XXS.gguf')
    (tmp_path / 'unsloth/gemma-4-26B-A4B-it-UD-IQ2_XXS/mmproj-Q5_K_M.gguf').write_text('x')
    args = build_args(Path(__file__).parent / 'fixtures/models.yaml', 'vl-with-mmproj', host='127.0.0.1', port=1234, extra_args=['--end'])
    assert '--mmproj' in args and '--reasoning' in args and 'on' in args and '--jinja' in args
    assert args[-1] == '--end'


def test_fixture_extras(monkeypatch, tmp_path):
    make_model(monkeypatch, tmp_path, 'foo/bar.gguf')
    args = build_args(Path(__file__).parent / 'fixtures/models.yaml', 'with-extras')
    assert args[-2:] == ['--no-mmap', '--special'] and 'auto' in args


def test_all_optional_branches_and_aliases(monkeypatch, tmp_path):
    make_model(monkeypatch, tmp_path)
    yaml = make_yaml(tmp_path, '''models:\n- name: all\n  model: model.gguf\n  threads_batch: 2\n  batch_size: 3\n  ubatch_size: 4\n  ctk: q4\n  ctv: q5\n  mtp: missing.mtp\n  jinja_file: template.jinja\n  override: x=y\n  reasoning: false\n  reasoning_format: deep\n  no_mmap: true\n  jinja: true\n  extra_args: a,b\n''')
    args = build_args(yaml, 'all', extra_args=['tail'])
    for value in ['--threads-batch', '--batch-size', '--ubatch-size', '-ctk', '-ctv', '--mtp', '--chat-template-file', '--override-kv', '--reasoning', 'off', '--reasoning-format', '--no-mmap', '--jinja', 'a', 'b', 'tail']:
        assert value in args


def test_dict_models_and_missing_model(tmp_path):
    yaml = make_yaml(tmp_path, 'model:\n  name: x\n  model: absent\n')
    with pytest.raises(FileNotFoundError):
        build_args(yaml, 'x')
    missing = make_yaml(tmp_path, 'models:\n- name: x\n')
    with pytest.raises(KeyError):
        build_args(missing, 'x')


def test_missing_yaml_and_unknown(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_args(tmp_path / 'missing.yaml', 'x')
    yaml = make_yaml(tmp_path, 'models:\n- name: x\n  model: absent\n')
    with pytest.raises(KeyError):
        build_args(yaml, 'missing')


def test_bool_false_and_empty_values(monkeypatch, tmp_path):
    make_model(monkeypatch, tmp_path)
    yaml = make_yaml(tmp_path, '''models:\n- name: x\n  model: model.gguf\n  ctk: ''\n  ctv: ''\n  reasoning: true\n  cont_batching: false\n  flash_attention: false\n  no_mmap: false\n  jinja: false\n  agent: false\n  extra_args: []\n''')
    args = build_args(yaml, 'x')
    assert args[-2:] == ['--reasoning', 'on']
    assert '-ctk' not in args and '-ctv' not in args


def test_numeric_empty_values(monkeypatch, tmp_path):
    make_model(monkeypatch, tmp_path)
    yaml = make_yaml(tmp_path, 'models:\n- name: x\n  model: model.gguf\n  gpu_layers: ""\n  threads: null\n')
    args = build_args(yaml, 'x')
    assert '-ngl' not in args and '--threads' not in args


def test_optional_none_reasoning(monkeypatch, tmp_path):
    make_model(monkeypatch, tmp_path)
    yaml = make_yaml(tmp_path, 'models:\n- name: x\n  model: model.gguf\n  reasoning: null\n  mmproj: null\n')
    assert '--reasoning' not in build_args(yaml, 'x')


# ---- New tests: jinja behavior + production yaml coverage ----

def test_jinja_default_unset_emits_no_flag(monkeypatch, tmp_path):
    """jinja absent from yaml → --jinja flag NOT emitted."""
    make_model(monkeypatch, tmp_path)
    yaml = make_yaml(tmp_path, 'models:\n- name: x\n  model: model.gguf\n')
    args = build_args(yaml, 'x')
    assert '--jinja' not in args


def test_jinja_true_emits_flag(monkeypatch, tmp_path):
    """jinja: true → --jinja flag IS emitted."""
    make_model(monkeypatch, tmp_path)
    yaml = make_yaml(tmp_path, 'models:\n- name: x\n  model: model.gguf\n  jinja: true\n')
    args = build_args(yaml, 'x')
    assert '--jinja' in args
    # Should appear exactly once
    assert args.count('--jinja') == 1


def test_jinja_false_does_not_emit_flag(monkeypatch, tmp_path):
    """jinja: false explicitly → --jinja flag NOT emitted."""
    make_model(monkeypatch, tmp_path)
    yaml = make_yaml(tmp_path, 'models:\n- name: x\n  model: model.gguf\n  jinja: false\n')
    args = build_args(yaml, 'x')
    assert '--jinja' not in args


# ---- Production yaml coverage: every entry must build a valid argv ----

@pytest.mark.skipif(not PROD_YAML.exists(), reason="production yaml not available")
def test_prod_yaml_parses_cleanly():
    """The production models.yaml must parse without error."""
    import yaml
    with open(PROD_YAML) as f:
        data = yaml.safe_load(f)
    assert "models" in data
    assert isinstance(data["models"], list)
    assert len(data["models"]) > 0


@pytest.mark.skipif(not PROD_YAML.exists(), reason="production yaml not available")
def test_prod_yaml_no_duplicate_names():
    """Every name in production yaml must be unique."""
    import yaml
    with open(PROD_YAML) as f:
        data = yaml.safe_load(f)
    names = [m["name"] for m in data["models"] if "name" in m]
    duplicates = [n for n in names if names.count(n) > 1]
    assert not duplicates, f"duplicate names: {set(duplicates)}"


@pytest.mark.skipif(not PROD_YAML.exists(), reason="production yaml not available")
def test_prod_yaml_no_launcher_owned_fields():
    """The 6 fields now hardcoded in launcher.sh must NOT appear in any model entry."""
    import yaml
    LAUNCHER_OWNED = {"flash_attention", "cont_batching", "parallel", "ctk", "ctv", "threads"}
    with open(PROD_YAML) as f:
        data = yaml.safe_load(f)
    leaks = []
    for m in data["models"]:
        for field in LAUNCHER_OWNED & set(m.keys()):
            leaks.append(f"{m.get('name', '<unnamed>')}: {field}={m[field]}")
    assert not leaks, f"launcher-owned fields leaked into yaml:\n  " + "\n  ".join(leaks)


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
@pytest.mark.parametrize("model_name", [
    # Sample one entry from each publisher category — proves the schema works
    "google-gemma-4-12b-it-qat-q4_0",        # google
    "unsloth-qwen3-coder-30b-a3b-instruct-ud-iq2_xxs",  # unsloth (jinja now true)
    "lmstudio-deepseek-r1-8b",               # lmstudio (jinja now true)
    "openbmb-minicpm-v-4_6-q8_0",            # openbmb (no jinja)
    "microsoft-Phi-3-mini-4k-instruct-q4",   # microsoft (no jinja, agent false)
    "prism-bonsai-1.7b-q1_0",                # prism (no jinja)
    "lmstudio-nemotron-3-nano-30b",          # unsloth/lmstudio with no_mmap + override_tensor
])
def test_prod_yaml_sample_entries_build(monkeypatch, tmp_path, model_name):
    """Each sampled production entry must produce a valid argv with no duplicates of the
    launcher-owned flags."""
    import yaml
    monkeypatch.setenv("MODELS_DIR", str(DISK_ROOT))
    args = build_args(PROD_YAML, model_name)
    # Required basics
    assert args[0] == "-m"
    assert "--host" in args and "--port" in args
    assert "-ngl" in args
    assert "--ctx-size" in args
    # No launcher-owned flag should appear in build_args output (launcher hardcodes them)
    for flag in ("-ctk", "-ctv", "--cont-batching", "--parallel", "--threads", "-fa"):
        assert flag not in args, f"{flag} leaked from yaml into argv"


@pytest.mark.skipif(not PROD_YAML.exists(), reason="production yaml not available")
def test_prod_yaml_jinja_distribution():
    """Verify the post-patch jinja distribution matches expectations."""
    import yaml
    from collections import Counter
    with open(PROD_YAML) as f:
        data = yaml.safe_load(f)
    dist = Counter(m.get("jinja") for m in data["models"])
    # After the 2026-08-09 patch: 57 with jinja, 19 without
    assert dist[True] >= 50, f"expected ≥50 entries with jinja: true, got {dist[True]}"
    assert dist[None] <= 25, f"expected ≤25 entries without jinja, got {dist[None]}"


@pytest.mark.skipif(not PROD_YAML.exists(), reason="production yaml not available")
def test_prod_yaml_no_redundant_reasoning_budget():
    """Every entry with reasoning: false must not have reasoning_budget set (launcher ignores it)."""
    import yaml
    with open(PROD_YAML) as f:
        data = yaml.safe_load(f)
    bad = []
    for m in data["models"]:
        if m.get("reasoning") is False and "reasoning_budget" in m:
            bad.append(m["name"])
    assert not bad, f"reasoning: false entries with reasoning_budget (redundant):\n  " + "\n  ".join(bad)


@pytest.mark.skipif(not PROD_YAML.exists(), reason="production yaml not available")
def test_prod_yaml_no_redundant_reasoning_budget_when_unset():
    """Entries with reasoning: None (unset) shouldn't carry reasoning_budget — pure noise."""
    import yaml
    with open(PROD_YAML) as f:
        data = yaml.safe_load(f)
    bad = []
    for m in data["models"]:
        if m.get("reasoning") is None and "reasoning_budget" in m:
            bad.append(m["name"])
    assert not bad, f"reasoning: None entries with reasoning_budget (redundant):\n  " + "\n  ".join(bad)


@pytest.mark.skipif(not PROD_YAML.exists(), reason="production yaml not available")
def test_prod_yaml_reasoning_budget_consistent_with_reasoning():
    """Every reasoning: true entry must have reasoning_budget: 16384 (canonical cap).
    Every reasoning: false or None entry must NOT have reasoning_budget."""
    import yaml
    with open(PROD_YAML) as f:
        data = yaml.safe_load(f)
    bad = []
    for m in data["models"]:
        name = m.get("name", "<unnamed>")
        r = m.get("reasoning")
        rb = m.get("reasoning_budget")
        if r is True:
            if rb != 16384:
                bad.append(f"{name}: reasoning=true but budget={rb} (expected 16384)")
        elif r in (False, None):
            if rb is not None:
                bad.append(f"{name}: reasoning={r} but budget={rb} set (redundant)")
    assert not bad, "reasoning/reasoning_budget inconsistency:\n  " + "\n  ".join(bad)


@pytest.mark.skipif(not PROD_YAML.exists(), reason="production yaml not available")
def test_prod_yaml_reasoning_distribution():
    """Verify the post-patch reasoning distribution matches expectations:
    ~25 entries with reasoning: true + budget: 16384 (the reasoning models)."""
    import yaml
    from collections import Counter
    with open(PROD_YAML) as f:
        data = yaml.safe_load(f)
    dist = Counter(m.get("reasoning") for m in data["models"])
    # After the 2026-08-09 patch: 25 reasoning models, 34 explicitly off, 17 unset
    assert dist[True] >= 20, f"expected ≥20 reasoning models, got {dist[True]}"
    assert dist[False] >= 30, f"expected ≥30 reasoning:false entries, got {dist[False]}"
