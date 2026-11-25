"""Registered hooks for configuration loading and launch validation (no upward imports)."""
from __future__ import annotations

_profile_expander = None
_task_notes_validator = None


def register_profile_expander(fn):
    """Register `(config) -> config` to resolve ``source="profile"`` upstream entries during ``Config.load``."""
    global _profile_expander
    _profile_expander = fn


def register_task_notes_validator(fn):
    """Register `(task_notes, task_notes_source) -> None` to validate launch-time task notes."""
    global _task_notes_validator
    _task_notes_validator = fn


def expand_profiles(config):
    if _profile_expander is None:
        raise RuntimeError("profile_expander_not_registered")
    return _profile_expander(config)


def validate_task_notes(task_notes, task_notes_source):
    if _task_notes_validator is None:
        raise RuntimeError("task_notes_validator_not_registered")
    _task_notes_validator(task_notes, task_notes_source)
