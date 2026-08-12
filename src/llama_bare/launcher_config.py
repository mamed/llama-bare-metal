from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import yaml

from .path_translate import translate_path

PathLike = Union[str, Path]


def build_args(
    yaml_path: PathLike,
    model_name: str,
    *,
    host: Optional[str] = None,
    port: Optional[Union[str, int]] = None,
    extra_args: Optional[Sequence[str]] = None,
) -> list[str]:
    """Build the full llama-server argv list for the given YAML entry.

    Returns the argv as a list of strings, ready for subprocess exec.

    Raises FileNotFoundError if yaml_path doesn't exist.
    Raises KeyError if model_name is not in the YAML.
    Raises FileNotFoundError if the resolved model file doesn't exist on disk
    (use llama_bare.path_translate.translate_path to resolve).
    """
    with open(yaml_path) as stream:
        config = yaml.safe_load(stream) or {}
    models = config.get("models") or config.get("model") or []
    if isinstance(models, dict):
        models = [models]
    entry = next((item for item in models if isinstance(item, dict) and item.get("name") == model_name), None)
    if entry is None:
        raise KeyError(model_name)
    model = entry.get("model")
    if not model:
        raise KeyError("model")
    model_path = translate_path(str(model))
    if not Path(model_path).is_file():
        raise FileNotFoundError(model_path)
    args = ["-m", model_path, "--host", "0.0.0.0" if host is None else str(host), "--port", "64000" if port is None else str(port)]
    numeric = (("gpu_layers", "-ngl"), ("context_size", "--ctx-size"), ("threads", "--threads"), ("ubatch_size", "--ubatch-size"), ("parallel", "--parallel"))
    for key, flag in numeric:
        if entry.get(key) is not None and entry.get(key) != "":
            args.extend([flag, str(entry[key])])
    for key, flag in (("ctk", "-ctk"), ("ctv", "-ctv")):
        if entry.get(key):
            args.extend([flag, str(entry[key])])
    for key, flag in (("mmproj", "--mmproj"), ("mtp", "--mtp")):
        if entry.get(key):
            args.extend([flag, translate_path(str(entry[key]))])
    chat_template = entry.get("chat_template_file", entry.get("jinja_file"))
    if chat_template:
        args.extend(["--chat-template-file", translate_path(str(chat_template))])
    override = entry.get("override_kv", entry.get("override"))
    if override:
        args.extend(["--override-kv", str(override)])
    if "override_tensor" in entry and entry["override_tensor"]:
        args.extend(["--override-tensor", str(entry["override_tensor"])])
    for key, flag, value in (("cont_batching", "--cont-batching", None), ("flash_attention", "-fa", "on"), ("no_mmap", "--no-mmap", None)):
        if entry.get(key) is True:
            args.append(flag)
            if value is not None:
                args.append(value)
    yaml_extra = entry.get("extra_args")
    if isinstance(yaml_extra, list):
        args.extend(yaml_extra)
    elif isinstance(yaml_extra, str):
        args.extend(yaml_extra.split(","))
    if extra_args:
        args.extend(extra_args)

    # ----- Server-level defaults (apply to every model unless overridden) -----
    # Per-model yaml can no longer set these (per-model handlers removed).
    # Disable per-launch via env var.
    import os as _os
    if _os.environ.get("DISABLE_REASONING") != "true":
        args.extend(["--reasoning", "on", "--reasoning-budget", "16384"])
    if _os.environ.get("DISABLE_AGENT") != "true":
        args.append("--agent")
    if _os.environ.get("DISABLE_JINJA") != "true":
        args.append("--jinja")

    return args
