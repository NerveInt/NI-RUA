"""Offline visual and contact trajectory evidence extractors for robot calibration."""
from __future__ import annotations

import importlib

__all__ = ["config", "contact_trajectory", "vggt_evidence", "video_evidence"]


def __getattr__(name: str):
    if name in __all__:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
