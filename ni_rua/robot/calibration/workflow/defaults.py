"""Packaged calibration defaults, independent of drivers, models and deployment validation."""
from copy import deepcopy
import json
from pathlib import Path

_DEFAULTS = json.loads(
    (Path(__file__).resolve().parents[2] / "backends" / "so101" / "calibration_defaults.json").read_text(encoding="utf-8")
)


def calibration_default(name):
    """Return a copy of a named default; unknown names raise KeyError."""
    return deepcopy(_DEFAULTS[name])
