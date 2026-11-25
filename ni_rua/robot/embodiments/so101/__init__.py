"""SO-101 embodiment package: 5-DoF kinematics, swinging-jaw CAD geometry, and ``GraspReferenceHook`` registration."""
import importlib
from types import SimpleNamespace

from .profile import (
    ARM_JOINTS,
    BASE_LINK,
    GRIPPER_JOINT,
    SO101,
    TCP_LINK,
    WRIST_CAMERA_PARENT_LINK,
    fk,
    fk_all,
    ik,
    ik_position_priority,
    joint_limits_rad,
    load_model,
    so101_chain,
)


def _register_so101_end_effector() -> None:
    from ...model.end_effectors import register_grasp_reference, register_packaged_tcp
    from ..arx_x5.profile import physical_facts

    class _LazySO101Hook:
        """Defers CAD STL parsing until ``jaw_fingertip_midpoint`` is actually queried."""

        def locate(self, gripper, origin, rotation, jaw_rad=None):
            from .gripper_cad import SO101JawFingertipHook

            return SO101JawFingertipHook().locate(gripper, origin, rotation, jaw_rad=jaw_rad)

        def aperture_m(self, gripper, jaw_rad):
            from .gripper_cad import SO101JawFingertipHook

            return SO101JawFingertipHook().aperture_m(gripper, jaw_rad)

    register_grasp_reference("jaw_fingertip_midpoint", _LazySO101Hook())
    register_packaged_tcp(
        SO101.tcp_config_id,
        lambda: (
            SimpleNamespace(grasp_reference="jaw_fingertip_midpoint"),
            physical_facts("so101")["kinematics"]["gripper_joint_order"][0],
        ),
    )


_register_so101_end_effector()


def so101_profile(deployment):
    """Build the resolved ``EmbodimentProfile`` for a validated SO-101 deployment mapping."""
    from ...backends.so101.deployment import embodiment_profile

    return embodiment_profile(deployment)


__all__ = [
    "ARM_JOINTS",
    "BASE_LINK",
    "GRIPPER_JOINT",
    "SO101",
    "TCP_LINK",
    "WRIST_CAMERA_PARENT_LINK",
    "fk",
    "fk_all",
    "ik",
    "ik_position_priority",
    "joint_limits_rad",
    "load_model",
    "so101_chain",
    "so101_profile",
]


def __getattr__(name: str):
    if name == "gravity":
        return importlib.import_module(".gravity", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
