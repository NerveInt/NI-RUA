"""Shared file-backed defaults for offline SO101 evidence commands; no hardware imports.

Numeric meanings/units/bounds belong to the analysis function contracts and
so101-configuration.md. This loader validates section/key/type identity, not physics.
"""
from copy import deepcopy
import json
import math
from pathlib import Path

_DEFAULTS = json.loads(Path(__file__).with_name("defaults.json").read_text())
_DEFAULTS = {k: v for k, v in _DEFAULTS.items() if not k.startswith("_")}


def evidence_settings(section, path=None, overrides=None):
    """Resolve packaged defaults < JSON file < explicit caller/CLI overrides.

    section: sam31_video, video_evidence, vggt_evidence or contact_fit (unitless name).
    path: optional JSON file path; omitted uses packaged defaults. The file may contain
        any documented section; keys beginning '_' are comments. Unknown keys fail.
    overrides: optional settings mapping for this section; values keep their documented
        units. Empty/omitted makes no changes; null is not an omission.
    Returns a new effective settings dict, saved by the consuming analysis in its output.
    Example: evidence_settings('video_evidence', 'review-settings.json', {'max_frame_gap_s': .2}).
    """
    if section not in _DEFAULTS:
        raise ValueError(f"unknown evidence section: {section}")
    document = {} if path is None else json.loads(Path(path).read_text())
    if not isinstance(document, dict):
        raise ValueError("evidence config must be an object")
    def checked(name, values):
        if not isinstance(values, dict):
            raise ValueError(f"evidence section must be an object: {name}")
        result = {}
        for key, value in values.items():
            if key.startswith('_'):
                continue
            if key not in _DEFAULTS[name]:
                raise ValueError(f"unknown evidence setting: {name}.{key}")
            default = _DEFAULTS[name][key]
            if isinstance(default, bool):
                valid = type(value) is bool
            elif isinstance(default, int):
                valid = type(value) is int
            elif isinstance(default, float):
                valid = type(value) in (int, float) and math.isfinite(value)
            else:
                valid = isinstance(value, list)
            if not valid:
                raise ValueError(f"invalid evidence setting type: {name}.{key}")
            result[key] = value
        return result
    sections = {}
    for name, values in document.items():
        if name.startswith('_'):
            continue
        if name not in _DEFAULTS:
            raise ValueError(f"unknown evidence section: {name}")
        sections[name] = checked(name, values)
    return deepcopy(_DEFAULTS[section] | sections.get(section, {}) | checked(section, {} if overrides is None else overrides))
