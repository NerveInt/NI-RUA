"""Wire cortex profile expansion and task-notes validation into ``ni_rua.core.config.hooks``."""
from __future__ import annotations

from ..core.config.hooks import register_profile_expander, register_task_notes_validator


def expand_profile_references_hook(config):
    """Expand ``source="profile"`` upstream sections via ``ni_rua.cortex.agents.profiles``."""
    from .agents import profiles
    return profiles.expand_profile_references(config, strict=False)


def validate_task_notes_hook(task_notes, task_notes_source):
    """Validate launch-time custom task notes via ``ni_rua.cortex.context.task_notes``."""
    from .context import task_notes as task_notes_module
    task_notes_module.resolve_task_notes(False, task_notes, task_notes_source)


def register_config_hooks():
    """Register cortex configuration hooks on ``ni_rua.core.config.hooks``."""
    register_profile_expander(expand_profile_references_hook)
    register_task_notes_validator(validate_task_notes_hook)


register_config_hooks()
