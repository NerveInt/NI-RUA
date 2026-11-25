"""Formal runtime-checkable protocols and clock-domain contracts for ``ni_rua/robot`` backends.

Every physical robot action still enters exclusively through ``Harness.execute``;
the capability sub-protocols below allow callers (such as Gateway capability
routers and evaluation runners) to check optional backend surfaces via
``isinstance(robot, EvaluatorBackend)`` instead of duck-typed ``getattr`` probes
or hardcoded environment-name branches.
"""
from __future__ import annotations

from enum import StrEnum
from typing import Any, Callable, Protocol, runtime_checkable

from ..core.embodiment import EmbodimentProfile


class ClockDomain(StrEnum):
    """Physical or simulated progression clock domain of an embodiment/backend.

    - ``SIMULATED_STEPS`` (``"simulated_steps"``): discrete physics stepping owned by
      a simulator worker (e.g., RoboDojo / Isaac Sim at ``control_hz``).
    - ``REAL_TIME`` (``"real_time"``): monotonic wall-clock hardware control loop
      (e.g., SO-101 real follower arm).
    """

    SIMULATED_STEPS = "simulated_steps"
    REAL_TIME = "real_time"


def clock_domain_of(target: Any) -> ClockDomain:
    """Resolve the ``ClockDomain`` of an ``EmbodimentProfile`` or ``RobotBackend`` instance."""
    if isinstance(target, ClockDomain):
        return target
    if isinstance(target, EmbodimentProfile):
        return ClockDomain(target.clock)
    profile_fn = getattr(target, "embodiment_profile", None)
    if callable(profile_fn):
        return ClockDomain(profile_fn().clock)
    raise TypeError(f"cannot_resolve_clock_domain:{type(target).__name__}")


@runtime_checkable
class RobotBackend(Protocol):
    """Core lifecycle, observation, batch preview, and action execution contract for all backends."""

    compile_action: Callable[..., dict[str, Any]]

    def embodiment_profile(self) -> EmbodimentProfile:
        """Return the frozen ``EmbodimentProfile`` describing this backend's robot and control limits."""
        ...

    def start(self) -> dict[str, Any]:
        """Start the backend worker (idempotent if already started) and return public capabilities."""
        ...

    def observe(self) -> dict[str, Any]:
        """Capture one synchronized raw observation bundle from the backend."""
        ...

    def preview_batch(self, action: dict[str, Any]) -> dict[str, Any]:
        """Compile and preview a ``BatchAction`` without advancing physics or hardware state."""
        ...

    def execute(
        self,
        action: dict[str, Any],
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        cancel_flag: Any = None,
    ) -> dict[str, Any]:
        """Validate and dispatch a physical action, returning a typed raw action receipt."""
        ...

    def validate_action(self, action: dict[str, Any]) -> Any:
        """Validate a compiled backend action dict without starting a worker or moving the robot."""
        ...

    def validate_reset(self, seed: int = 0, layout_id: int = 0) -> Any:
        """Read-only reset preflight; refuses invalid ``seed`` / ``layout_id`` before touching world state."""
        ...

    def reset(self, seed: int = 0, layout_id: int = 0) -> dict[str, Any]:
        """Reset episode state for ``(seed, layout_id)`` and return the post-reset receipt."""
        ...

    def close(self) -> dict[str, Any]:
        """Gracefully close the backend worker and return exit/cleanup verification facts."""
        ...


@runtime_checkable
class EvaluatorBackend(Protocol):
    """Privileged simulator evaluation and observer-only ground-truth snapshot capability."""

    def evaluate(self, final_check: bool = True) -> dict[str, Any]:
        """Return native task evaluation result (observer-only; never exposed to actor context)."""
        ...

    def observer_snapshot(self) -> dict[str, Any]:
        """Return privileged scene snapshot for independent observer verification only."""
        ...


@runtime_checkable
class VideoBackend(Protocol):
    """Observer/motion video recording and MP4 finalization capability."""

    def record(self, enabled: bool = True) -> dict[str, Any]:
        """Enable or pause camera frame recording (`enabled`: bool, default True)."""
        ...

    def save_video(self) -> dict[str, Any]:
        """Seal recorded video streams and return observer-only file metadata."""
        ...


@runtime_checkable
class OperatorBackend(Protocol):
    """Interactive hardware operator control and nominal joint-limit reporting capability."""

    def operator(
        self,
        command: str,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        **arguments: Any,
    ) -> dict[str, Any]:
        """Execute an operator-only command (e.g., ``hold``, ``relax``, ``torque_on``, ``go_home``)."""
        ...

    def joint_limits_rad(self) -> dict[str, list[float]]:
        """Return nominal URDF joint bounds ``{joint_name: [lower_rad, upper_rad]}`` in ``rad``."""
        ...


@runtime_checkable
class CalibratableBackend(Protocol):
    """Operator-guided hardware calibration workflow and deployment hot-reload capability."""

    def operator(
        self,
        command: str,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        **arguments: Any,
    ) -> dict[str, Any]:
        """Dispatch calibration workflow commands (``calib_workflow_*``, ``calib_step_*``, ``calib_apply``)."""
        ...

    def calibrate_timeout_s(self) -> float:
        """Return parent IPC timeout in ``s`` for a full ``calibrate`` run."""
        ...

    def calibration_step_timeout_s(self) -> float:
        """Return parent IPC timeout in ``s`` for a single ``calib_step_start`` execution."""
        ...

    def reload_deployment(self) -> dict[str, Any]:
        """Reload deployment configuration from disk after ``calib_apply`` or ``calib_rollback``."""
        ...


__all__ = [
    "CalibratableBackend",
    "ClockDomain",
    "EvaluatorBackend",
    "OperatorBackend",
    "RobotBackend",
    "VideoBackend",
    "clock_domain_of",
]
