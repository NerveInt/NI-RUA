"""Operator-gated S0--S6 calibration state machine, capture, solver, and bundle management."""
from __future__ import annotations

import importlib

__all__ = [
    "bundle",
    "capture",
    "charuco",
    "defaults",
    "hand_eye_solve",
    "joint_calibration",
    "runner",
    "state_machine",
    "touch_off",
    "video",
    "xy_verification",
]


def __getattr__(name: str):
    if name in __all__:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
