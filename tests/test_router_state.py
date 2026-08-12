import os
from pathlib import Path
import pytest
from llama_bare.router_state import (read_current_model, write_current_model, write_env_file, load_models_from_yaml, resolve_model_path, list_models_sorted_by_basename)


def test_state_roundtrip_and_atomic(tmp_path):
    state = tmp_path / 'nested/state.yaml'
    write_current_model(state, 'tiny')
    assert read_current_model(state) == 'tiny'
    assert not (tmp_path / 'nested/state.yaml.tmp').exists()
    write_env_file(tmp_path / 'env/file', 'tiny')
    assert (tmp_path / 'env/file').read_text() == 'MODEL_NAME=tiny\n'


def test_write_state_plain_text_format(tmp_path):
    # write_current_model MUST emit plain text matching the legacy bash
    # wrapper's `echo "$MODEL_NAME" > "$STATE_FILE"` — not YAML. This test
    # pins the exact format so future refactors don't drift back to YAML.
    state = tmp_path / 'state'
    write_current_model(state, 'prism-bonsai-1.7b-q1_0')
    raw = state.read_text()
    assert raw == 'prism-bonsai-1.7b-q1_0\n'
    assert 'model:' not in raw  # explicitly not YAML


def test_write_state_accepts_pathlike(tmp_path):
    # Type hint says PathLike — confirm both str and Path work.
    state_str = tmp_path / 'str-state'
    state_path = tmp_path / 'path-state'
    write_current_model(str(state_str), 'a')
    write_current_model(state_path, 'b')
    assert read_current_model(state_str) == 'a'
    assert read_current_model(state_path) == 'b'


def test_read_variants_and_errors(tmp_path):
    assert read_current_model(tmp_path / 'missing') is None
    empty = tmp_path / 'empty'; empty.write_text('')
    assert read_current_model(empty) is None
    scalar = tmp_path / 'scalar'; scalar.write_text('abc\n')
    assert read_current_model(scalar) == 'abc'
    named = tmp_path / 'named'; named.write_text('model_name: named\n')
    assert read_current_model(named) == 'named'
    bad = tmp_path / 'bad'; bad.write_text(': bad: yaml')
    assert read_current_model(bad) is None
    restricted = tmp_path / 'restricted'; restricted.write_text('model: no\n'); restricted.chmod(0)
    try:
        assert read_current_model(restricted) is None
    finally:
        restricted.chmod(0o600)


def test_load_models_all_shapes(tmp_path):
    yaml = tmp_path / 'models.yaml'; yaml.write_text('models:\n- name: a\n- name: b\n- nope: x\n')
    assert load_models_from_yaml(yaml) == {'a', 'b'}
    one = tmp_path / 'one.yaml'; one.write_text('model:\n  name: one\n')
    assert load_models_from_yaml(one) == {'one'}
    assert load_models_from_yaml(tmp_path / 'missing') == set()
    bad = tmp_path / 'bad.yaml'; bad.write_text(': bad: yaml')
    assert load_models_from_yaml(bad) == set()


def test_resolve_model_path(monkeypatch, tmp_path):
    monkeypatch.setenv('MODELS_DIR', str(tmp_path))
    model = tmp_path / 'x.gguf'; model.write_text('x')
    yaml = tmp_path / 'models.yaml'; yaml.write_text('models:\n- name: x\n  model: x.gguf\n')
    assert resolve_model_path(yaml, 'x') == str(model)
    with pytest.raises(KeyError): resolve_model_path(yaml, 'missing')
    missing = tmp_path / 'missing.yaml'; missing.write_text('models:\n- name: x\n  model: absent\n')
    with pytest.raises(FileNotFoundError): resolve_model_path(missing, 'x')
    one = tmp_path / 'one-model.yaml'; one.write_text('model:\n  name: x\n  model: x.gguf\n')
    assert resolve_model_path(one, 'x') == str(model)


def test_write_env_overwrites(tmp_path):
    target = tmp_path / 'env'
    write_env_file(target, 'first'); write_env_file(target, 'second')
    assert target.read_text() == 'MODEL_NAME=second\n'


def test_load_non_mapping(tmp_path):
    path = tmp_path / 'scalar.yaml'; path.write_text('- just-a-string\n')
    assert load_models_from_yaml(path) == set()


def test_resolve_missing_model_key(tmp_path):
    path = tmp_path / 'bad.yaml'; path.write_text('models:\n- name: x\n')
    with pytest.raises(KeyError): resolve_model_path(path, 'x')


def test_read_non_dict_empty(tmp_path):
    path = tmp_path / 'list.yaml'; path.write_text('- one\n')
    assert read_current_model(path) is None


# ---- list_models_sorted_by_basename ----

def test_list_models_sorted_by_basename_clusters_variants(tmp_path):
    """Three quants of the same model should cluster together regardless of yaml name."""
    yaml = tmp_path / 'models.yaml'
    yaml.write_text('''models:
- name: zzz-third-quant
  model: /models/Devstral-Small-2-24B-Instruct-2512-GGUF/Devstral-Small-2-24B-Instruct-2512-UD-IQ2_XXS.gguf
- name: aaa-first-quant
  model: /models/Devstral-Small-2-24B-Instruct-2512-GGUF/Devstral-Small-2-24B-Instruct-2512-UD-Q2_K_XL.gguf
- name: middle-quant
  model: /models/Devstral-Small-2-24B-Instruct-2512-GGUF/Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf
- name: unrelated-model
  model: /models/somewhere/gemma-4-12b-it-qat-q4_0.gguf
''')
    result = list_models_sorted_by_basename(yaml)
    names = [m['name'] for m in result]
    # Devstral variants (3 of them) must cluster before gemma
    devstral_idxs = [i for i, n in enumerate(names) if 'Devstral' in n]
    gemma_idx = names.index('unrelated-model')
    assert all(i < gemma_idx for i in devstral_idxs), \
        f"Devstral variants should cluster before gemma, got: {names}"


def test_list_models_sorted_by_basename_tiebreaker(tmp_path):
    """When multiple entries share the same basename (rare), yaml name breaks ties."""
    yaml = tmp_path / 'models.yaml'
    yaml.write_text('''models:
- name: z-second
  model: /models/foo/Same-Basename-Name-Q4_K_M.gguf
- name: a-first
  model: /models/foo/Same-Basename-Name-Q4_K_M.gguf
''')
    result = list_models_sorted_by_basename(yaml)
    names = [m['name'] for m in result]
    assert names == ['a-first', 'z-second'], f"yaml name should break ties: {names}"


def test_list_models_sorted_by_basename_empty_and_errors(tmp_path):
    """Missing / malformed files → empty list, no exception."""
    assert list_models_sorted_by_basename(tmp_path / 'missing.yaml') == []
    bad = tmp_path / 'bad.yaml'; bad.write_text(': bad: yaml')
    assert list_models_sorted_by_basename(bad) == []
    # Empty yaml
    empty = tmp_path / 'empty.yaml'; empty.write_text('')
    assert list_models_sorted_by_basename(empty) == []


def test_list_models_sorted_by_basename_strips_quant_suffix_in_basename(tmp_path):
    """Different quants of the same base model share a basename prefix → cluster."""
    yaml = tmp_path / 'models.yaml'
    yaml.write_text('''models:
- name: other-yaml-name-1
  model: /models/unsloth/Qwen3-Coder-30B-A3B-Instruct-UD-IQ2_XXS/Qwen3-Coder-30B-A3B-Instruct-UD-IQ2_XXS.gguf
- name: other-yaml-name-2
  model: /models/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-UD-IQ2_XXS.gguf
''')
    result = list_models_sorted_by_basename(yaml)
    # Both files have the SAME basename "Qwen3-Coder-30B-A3B-Instruct-UD-IQ2_XXS"
    # so they're adjacent in sort order
    basenames = [m['model'].split('/')[-1].rsplit('.gguf', 1)[0] for m in result]
    assert basenames[0] == basenames[1], f"Expected both files to have same basename: {basenames}"


def test_list_models_sorted_by_basename_single_model_dict_shape(tmp_path):
    """Single-model yaml uses top-level 'model:' dict (not list) — branch coverage."""
    yaml = tmp_path / 'one.yaml'
    yaml.write_text('''model:
  name: single
  model: /models/foo/Single-Model-Q4_K_M.gguf
''')
    result = list_models_sorted_by_basename(yaml)
    assert len(result) == 1
    assert result[0]['name'] == 'single'
