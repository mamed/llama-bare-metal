import os
from pathlib import Path
from llama_bare.path_translate import translate_path, file_exists_on_host


def test_translate_models_prefix():
    assert translate_path('/models/a.gguf', '/host') == '/host/a.gguf'


def test_translate_basename():
    assert translate_path('a.gguf', '/host') == '/host/a.gguf'


def test_translate_absolute_unchanged():
    assert translate_path('/host/a.gguf', '/other') == '/host/a.gguf'


def test_translate_env_default(monkeypatch):
    monkeypatch.setenv('MODELS_DIR', '/env-models')
    assert translate_path('a.gguf') == '/env-models/a.gguf'


def test_translate_empty_string():
    # Empty string must NOT be decorated into "{models_dir}/" — that would
    # silently turn a missing/edge-case input into a directory reference.
    assert translate_path('', '/host') == ''
    assert translate_path('', None) == ''  # also with env-default fallback


def test_translate_models_root_no_slash():
    # `/models` (no trailing slash, no path after) returns unchanged.
    # Consistent with the bash launcher glob behavior — root namespace
    # is not a file path, so leave it alone.
    assert translate_path('/models', '/host') == '/models'


def test_translate_default_and_exists(tmp_path, monkeypatch):
    monkeypatch.delenv('MODELS_DIR', raising=False)
    assert translate_path('a.gguf') == '/models/a.gguf'
    file_path = tmp_path / 'x.gguf'
    file_path.write_text('x')
    assert file_exists_on_host(str(file_path))
    assert not file_exists_on_host(str(tmp_path / 'none'))
    assert file_exists_on_host('x.gguf', str(tmp_path))
    assert not file_exists_on_host('none.gguf', str(tmp_path))
    assert Path(translate_path('/models/x', str(tmp_path))).is_absolute()
