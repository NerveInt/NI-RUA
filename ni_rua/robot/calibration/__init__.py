"""Robot calibration workflows and offline verification evidence packages."""
from __future__ import annotations

import importlib

__all__ = ["evidence", "workflow"]


def __getattr__(name: str):
    if name in __all__:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
