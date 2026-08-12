"""llama_bare: bare-metal llama.cpp + router — Python modules extracted from bash.

Everything testable lives here. The bash scripts in this directory are thin
wrappers that call into these modules (so the bash layer only needs a few
integration tests, not unit coverage).
"""
from __future__ import annotations

__version__ = "0.1.0"