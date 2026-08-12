from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Optional, Union

import yaml

from .path_translate import translate_path

PathLike = Union[str, os.PathLike]


# B3: Module-level lock guarding state-file and .env operations. Both files
# are small (one line) and contended by 1-2 concurrent writers plus N readers,
# so a single lock is enough — no need for RLock or finer-grained locks.
_state_lock = threading.Lock()


# B4/B7: Module-level caches for models.yaml reads. Keyed by absolute path;
# value is (mtime, parsed). On every call we stat the file and serve from
# cache if the mtime is unchanged. This avoids re-parsing on every request
# when models.yaml has 1000+ entries.
_yaml_cache: dict[str, tuple[float, set[str]]] = {}
_yaml_list_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_yaml_cache_lock = threading.Lock()


def read_current_model(state_file: PathLike) -> Optional[str]:
    """Read the currently-loaded model name from the backend state file.

    Accepts BOTH plain-text (legacy: just the model name on one line) and
    YAML format (e.g. `model: <name>`). Returns None if the file is missing,
    unreadable, empty, or contains a non-scalar value (defensive against
    garbage content).
    """
    try:
        with _state_lock, open(state_file) as stream:
            data = yaml.safe_load(stream)
    except (OSError, ValueError, yaml.YAMLError):
        return None

    if isinstance(data, dict):
        value = data.get("model") or data.get("model_name")
    else:
        value = data

    if isinstance(value, (list, dict)):
        return None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def write_current_model(state_file: PathLike, model_name: str) -> None:
    """Atomically write the model name to the state file.

    Format: plain text, just `<model_name>` followed by a newline. Matches
    the legacy bash wrapper's `echo "$MODEL_NAME" > "$STATE_FILE"` format
    exactly, so hand-edits and existing tooling see consistent content.

    Atomic via temp-file + os.replace() under the module-level lock so
    concurrent writers don't collide on the .tmp name and readers don't
    see a half-written file. Creates parent directories as needed.
    """
    target = Path(state_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    # B3: explicit open()/write()/replace() so we hold the file only long
    # enough to write+rename atomically, all under the module lock. We do
    # NOT use Path.write_text() because that opens/closes the target file
    # directly (a crash mid-truncate would leave .env empty).
    with _state_lock:
        with open(temporary, "w") as stream:
            stream.write(f"{model_name}\n")
        os.replace(temporary, target)


def write_env_file(env_file: PathLike, model_name: str) -> None:
    """Write the .env file with `MODEL_NAME=<model_name>\\n`.

    Overwrites any existing content. Creates parent directories as needed.
    Atomic via temp-file + os.replace() (B3) so a crash mid-write never
    leaves .env partially written — readers either see the previous
    contents or the new contents, never a torn half-line.
    """
    target = Path(env_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    # B3: explicit open()/write()/replace() so the rename is atomic.
    # Path.write_text() uses `with` semantics to auto-close the file, but
    # that pattern truncates the target directly on failure. Using a temp
    # file means the target stays intact if the rename never completes.
    with _state_lock:
        with open(temporary, "w") as stream:
            stream.write(f"MODEL_NAME={model_name}\n")
        os.replace(temporary, target)


def load_models_from_yaml(yaml_path: PathLike) -> set[str]:
    """Return the set of valid model names registered in the YAML.

    Returns an empty set if the YAML is missing, malformed, or shaped
    in a way this loader doesn't recognize (defensive — callers can still
    iterate `models` if they want stricter validation).
    """
    try:
        with open(yaml_path) as stream:
            config = yaml.safe_load(stream) or {}
    except (OSError, TypeError, ValueError, yaml.YAMLError, AttributeError):
        return set()

    if not isinstance(config, dict):
        return set()
    models = config.get("models") or config.get("model") or []
    if isinstance(models, dict):
        models = [models]
    return {item["name"] for item in models if isinstance(item, dict) and item.get("name")}


def resolve_model_path(yaml_path: PathLike, model_name: str) -> str:
    """Look up the YAML entry and return the host-resolved absolute model path.

    Translates /models/... → MODELS_DIR using path_translate.translate_path.
    Raises KeyError if model_name not found in YAML.
    Raises FileNotFoundError if the resolved path doesn't exist on disk.
    """
    with open(yaml_path) as stream:
        config = yaml.safe_load(stream) or {}
    models = config.get("models") or config.get("model") or []
    if isinstance(models, dict):
        models = [models]
    entry = next((item for item in models if isinstance(item, dict) and item.get("name") == model_name), None)
    if entry is None:
        raise KeyError(model_name)
    path = translate_path(str(entry["model"]))
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    return path


def list_models_sorted_by_basename(yaml_path: PathLike) -> list[dict]:
    """Return model entries from the YAML sorted by file basename (so quants
    of the same model cluster together), with yaml name as tiebreaker.

    Returns an empty list if the YAML is missing or malformed.
    Each entry in the returned list is the raw dict from the YAML.
    """
    try:
        with open(yaml_path) as stream:
            config = yaml.safe_load(stream) or {}
    except (OSError, TypeError, ValueError, yaml.YAMLError, AttributeError):
        return []
    entries = config.get("models") or config.get("model") or []
    if isinstance(entries, dict):
        entries = [entries]
    entries = [m for m in entries if isinstance(m, dict)]
    def sort_key(m):
        path = m.get("model") or ""
        basename = path.split("/")[-1].rsplit(".gguf", 1)[0].lower() if path else ""
        name = m.get("name") or ""
        return (basename, name.lower())
    return sorted(entries, key=sort_key)


# ---- B4/B7: mtime-keyed caches ----

def cached_load_models_from_yaml(yaml_path: PathLike) -> set[str]:
    """B4: Return the set of valid models, cached by file mtime.

    Re-uses the parsed result while the file's mtime is unchanged. When
    an operator edits models.yaml the mtime advances and we reparse.
    Missing files are NOT cached (they're cheap; we don't want a stale
    empty set serving requests after the file is created).
    """
    key = str(Path(yaml_path).resolve())
    try:
        mtime = os.stat(key).st_mtime
    except OSError:
        # Can't cache a file we can't stat — fall through to a direct read.
        return load_models_from_yaml(key)
    with _yaml_cache_lock:
        cached = _yaml_cache.get(key)
        if cached is not None and cached[0] == mtime:
            return set(cached[1])  # copy so callers can't mutate cache
    result = load_models_from_yaml(key)
    with _yaml_cache_lock:
        _yaml_cache[key] = (mtime, set(result))
    return result


def cached_list_models_sorted_by_basename(yaml_path: PathLike) -> list[dict]:
    """B7: Return sorted list of model entries, cached by file mtime.

    Same mtime-keyed strategy as ``cached_load_models_from_yaml`` above.
    """
    key = str(Path(yaml_path).resolve())
    try:
        mtime = os.stat(key).st_mtime
    except OSError:
        return list_models_sorted_by_basename(key)
    with _yaml_cache_lock:
        cached = _yaml_list_cache.get(key)
        if cached is not None and cached[0] == mtime:
            return list(cached[1])
    result = list_models_sorted_by_basename(key)
    with _yaml_cache_lock:
        _yaml_list_cache[key] = (mtime, list(result))
    return result
