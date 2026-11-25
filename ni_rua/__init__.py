"""NI-RUA: one actor, one observed-world execution gateway."""
from __future__ import annotations

from importlib import import_module
from pathlib import Path


def _wire_core_hooks() -> None:
    """Register lazy cross-layer adapters on ``ni_rua.core`` without eager upward imports."""
    root = Path(__file__).resolve().parent

    if (
        (root / "core" / "embodiment.py").is_file()
        and (root / "robot" / "model" / "embodiments.py").is_file()
    ):
        from .core.embodiment import register_default_embodiment

        def _lazy_default_embodiment():
            return import_module("ni_rua.robot.model.embodiments").ARX_X5_PROFILE

        register_default_embodiment(_lazy_default_embodiment)

    has_arm_queue = (root / "robot" / "motion" / "arm_queue_contract.py").is_file()
    has_batch = (root / "robot" / "motion" / "batch_contract.py").is_file()
    if (root / "core" / "contracts.py").is_file() and (has_arm_queue or has_batch):
        from .core.contracts import register_arm_queue_validator, register_batch_validator

        if has_arm_queue:
            def _lazy_arm_queue_validator(action, capabilities=None):
                mod = import_module("ni_rua.robot.motion.arm_queue_contract")
                return mod.validate_arm_queue_action(action, capabilities)

            register_arm_queue_validator(_lazy_arm_queue_validator)

        if has_batch:
            def _lazy_batch_validator(action, capabilities=None, *, profile=None):
                mod = import_module("ni_rua.robot.motion.batch_contract")
                return mod.validate_batch_action(action, capabilities, profile=profile)

            register_batch_validator(_lazy_batch_validator)

    has_task_planning = (root / "cortex" / "context" / "task_planning.py").is_file()
    if (root / "core" / "tool_errors.py").is_file() and has_task_planning:
        from .core.tool_errors import register_plan_schema_provider

        def _lazy_plan_schema_provider(plan):
            return import_module("ni_rua.cortex.context.task_planning")._plan_schema_for_error(plan)

        register_plan_schema_provider(_lazy_plan_schema_provider)

    if (
        (root / "core" / "presentation" / "feedback_delivery.py").is_file()
        and (root / "runtime" / "actions" / "trajectory_feedback.py").is_file()
    ):
        from .core.presentation.feedback_delivery import register_trajectory_feedback

        def _lazy_trajectory_feedback(harness, action, receipt, correlation, *, diagnostic=False):
            mod = import_module("ni_rua.runtime.actions.trajectory_feedback")
            if diagnostic:
                return mod.build_action_trajectory_feedback(harness, action, receipt, correlation, diagnostic=True)
            return mod.build_action_trajectory_feedback(harness, action, receipt, correlation)

        register_trajectory_feedback(_lazy_trajectory_feedback)

    if (
        (root / "core" / "config" / "hooks.py").is_file()
        and (root / "cortex" / "launch_config.py").is_file()
    ):
        from .core.config.hooks import register_profile_expander, register_task_notes_validator

        def _lazy_profile_expander(config):
            mod = import_module("ni_rua.cortex.launch_config")
            return mod.expand_profile_references_hook(config)

        def _lazy_task_notes_validator(task_notes, task_notes_source):
            mod = import_module("ni_rua.cortex.launch_config")
            return mod.validate_task_notes_hook(task_notes, task_notes_source)

        register_profile_expander(_lazy_profile_expander)
        register_task_notes_validator(_lazy_task_notes_validator)


_wire_core_hooks()
