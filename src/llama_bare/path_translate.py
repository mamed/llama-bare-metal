"""Path translation between Docker-stack /models/... and bare-metal host paths.

The Docker stack mounted the model directory at /models/ inside the container
and YAML entries reference `/models/...` paths. On bare-metal that path
doesn't exist — host models live under $MODELS_DIR (defaults to
/home/fekry/llama-models/LLM-Models).

This module is the single source of truth for the translation. The launcher
delegates here, so a bug fix is one diff instead of bash-string-grepping.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, os.PathLike]


def translate_path(path: str, models_dir: Optional[str] = None) -> str:
    """Translate a path from the Docker /models/ namespace to the host namespace.

    Rules (in order):
      1. If path starts with `/models/`, strip the prefix and prepend `models_dir`.
      2. If path is a basename (no leading slash), prepend `models_dir`.
      3. Otherwise return path unchanged (caller passed a real absolute host path,
         OR an empty/edge-case string we should not decorate).

    `models_dir` defaults to the `MODELS_DIR` env var, falling back to `/models`
    (which is what the Docker stack used — keeps behavior identical if a caller
    forgets to set the env var).

    Returns the resolved string (never raises; callers verify existence).
    Empty/edge-case inputs are returned unchanged rather than decorated.
    """
    if models_dir is None:
        models_dir = os.environ.get("MODELS_DIR", "/models")

    if not path:
        return path  # empty string: don't decorate
    if path.startswith("/models/"):
        return models_dir + path[len("/models"):]
    if not path.startswith("/"):
        return f"{models_dir}/{path}"
    return path


def file_exists_on_host(path: str, models_dir: Optional[str] = None) -> bool:
    """Check whether the translated path resolves to a real file."""
    return Path(translate_path(path, models_dir)).is_file()