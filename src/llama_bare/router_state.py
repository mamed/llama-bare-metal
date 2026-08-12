from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import yaml

from .path_translate import translate_path

PathLike = Union[str, os.PathLike]


def read_current_model(state_file: PathLike) -> Optional[str]:
    """Read the currently-loaded model name from the backend state file.

    Accepts BOTH plain-text (legacy: just the model name on one line) and
    YAML format (e.g. `model: <name>`). Returns None if the file is missing,
    unreadable, empty, or contains a non-scalar value (defensive against
    garbage content).
    """
    try:
        with open(state_file) as stream:
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

    Atomic via temp-file + os.replace(). Creates parent directories as needed.
    """
    target = Path(state_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with open(temporary, "w") as stream:
        stream.write(f"{model_name}\n")
    os.replace(temporary, target)


def write_env_file(env_file: PathLike, model_name: str) -> None:
    """Write the .env file with `MODEL_NAME=<model_name>\\n`.

    Overwrites any existing content. Creates parent directories as needed.
    """
    target = Path(env_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"MODEL_NAME={model_name}\n")


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
