"""Pluggable end-effector grasp-reference and aperture geometry hooks (``GraspReferenceHook``).

Decouples ``ni_rua.robot.model.geometry`` from robot-specific CAD modules (such as SO-101's
swinging-jaw mesh derivation) so new parallel-jaw, linkage-jaw, or suction end-effectors
register their own ``GraspReferenceHook`` inside ``ni_rua/robot/embodiments/<arm>/``.
NumPy is imported lazily at call time so embodiment profile metadata modules stay NumPy-free at import.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np


@runtime_checkable
class GraspReferenceHook(Protocol):
    """Protocol locating an end-effector's grasp centre and jaw normal from a measured TCP pose.

    Units:
        - ``origin``: ``(3,)`` TCP position in reference frame (``m``).
        - ``rotation``: ``(3, 3)`` TCP rotation matrix in reference frame.
        - ``jaw_rad``: Optional measured jaw joint angle in ``rad`` (or ``m`` for prismatic jaws).
    """

    def locate(
        self,
        gripper: Any,
        origin: np.ndarray,
        rotation: np.ndarray,
        jaw_rad: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any] | None] | None:
        """Return ``(grasp_position_m, jaw_normal_unit, tip_details_or_None)`` or ``None`` if unavailable."""
        ...

    def aperture_m(self, gripper: Any, jaw_rad: float) -> float | None:
        """Return geometric fingertip opening in ``m`` at ``jaw_rad`` (``rad``), or ``None`` if unsupported."""
        ...


class TcpOffsetGraspReference:
    """Rigid TCP-offset grasp reference (e.g., ARX-X5 pad centre at ``+0.1501 m`` along ``approach_axis_local``)."""

    def locate(
        self,
        gripper: Any,
        origin: np.ndarray,
        rotation: np.ndarray,
        jaw_rad: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any] | None] | None:
        import numpy as np

        approach = rotation @ np.asarray(gripper.approach_axis_local, dtype=float)
        normal = rotation @ np.asarray(gripper.jaw_opening_axis_local, dtype=float)
        return origin + float(gripper.grasp_reference_offset_m) * approach, normal, None

    def aperture_m(self, gripper: Any, jaw_rad: float) -> float | None:
        return None


_GRASP_REFERENCE_HOOKS: dict[str, GraspReferenceHook] = {
    "tcp_offset": TcpOffsetGraspReference(),
}
_PACKAGED_TCP_FACTORIES: dict[str, Callable[[], tuple[Any, str | None]]] = {}
_BUILTINS_BOOTSTRAPPED = False


def register_grasp_reference(kind: str, hook: GraspReferenceHook) -> None:
    """Register a ``GraspReferenceHook`` implementation for ``GripperProfile.grasp_reference == kind``."""
    if not isinstance(kind, str) or not kind:
        raise ValueError("invalid_grasp_reference_kind")
    _GRASP_REFERENCE_HOOKS[kind] = hook


def register_packaged_tcp(tcp_config_id: str, factory: Callable[[], tuple[Any, str | None]]) -> None:
    """Register a fallback ``(gripper_spec, jaw_joint_name_or_None)`` factory for ``GeometryTools(profile=None)``."""
    if not isinstance(tcp_config_id, str) or not tcp_config_id:
        raise ValueError("invalid_tcp_config_id")
    _PACKAGED_TCP_FACTORIES[tcp_config_id] = factory


def _ensure_builtin_embodiments() -> None:
    global _BUILTINS_BOOTSTRAPPED
    if _BUILTINS_BOOTSTRAPPED:
        return
    _BUILTINS_BOOTSTRAPPED = True
    from ..embodiments import arx_x5, libero_franka_panda, so101  # noqa: F401


def locate_grasp_reference(
    gripper: Any,
    origin: np.ndarray,
    rotation: np.ndarray,
    jaw_rad: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any] | None] | None:
    """Locate ``(grasp_position_m, jaw_normal_unit, tips)`` via the registered ``GraspReferenceHook``."""
    import numpy as np

    origin_arr = np.asarray(origin, dtype=float)
    rot_arr = np.asarray(rotation, dtype=float)
    kind = getattr(gripper, "grasp_reference", None)
    if not isinstance(kind, str):
        return None
    hook = _GRASP_REFERENCE_HOOKS.get(kind)
    if hook is None:
        _ensure_builtin_embodiments()
        hook = _GRASP_REFERENCE_HOOKS.get(kind)
    if hook is None:
        return None
    return hook.locate(gripper, origin_arr, rot_arr, jaw_rad=jaw_rad)


def packaged_gripper_references() -> dict[str, tuple[Any, str | None]]:
    """Return ``{tcp_config_id: (gripper_spec, jaw_joint_name_or_None)}`` for packaged embodiments."""
    _ensure_builtin_embodiments()
    return {tcp_id: factory() for tcp_id, factory in _PACKAGED_TCP_FACTORIES.items()}


__all__ = [
    "GraspReferenceHook",
    "TcpOffsetGraspReference",
    "locate_grasp_reference",
    "packaged_gripper_references",
    "register_grasp_reference",
    "register_packaged_tcp",
]
