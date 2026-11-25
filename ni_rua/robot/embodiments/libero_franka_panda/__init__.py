"""Simulated LIBERO Franka Emika Panda embodiment package (simulation-only: robosuite MuJoCo)."""
from .profile import (
    ARM_ID,
    ARM_JOINTS,
    BASE_LINK,
    GRIPPER_JOINTS,
    LIBERO_FRANKA_PANDA,
    PANDA_CAMERA_EXAMPLES,
    PANDA_DEFAULT_ACTION_TIMEOUT_S,
    PANDA_DEFAULT_WAIT_TIMEOUT_S,
    PANDA_EMPTY_CLOSE_APERTURE_M,
    PANDA_FULL_OPEN_APERTURE_M,
    PANDA_GRIPPER_SPEED_SCALE,
    PANDA_LIMITS_PROFILES,
    PANDA_MASK_LINK,
    PANDA_MASK_REACH_M,
    PANDA_MAX_TIMEOUT_S,
    PANDA_OVERVIEW_CAMERA,
    PANDA_STALL_GUARD_SEMANTICS,
    PANDA_SUPPORTS,
    PANDA_VERTICAL_WINDOW_SEMANTICS,
    PANDA_WRIST_CAMERAS,
    TCP_LINK,
    WRIST_CAMERA_PARENT_LINK,
    _default_panda_profile,
    fk,
    fk_all,
    ik,
    ik_position_priority,
    joint_limits_rad,
    libero_franka_panda_profile,
    load_model,
    panda_chain,
    panda_defaults,
    panda_facts,
)


def _register_libero_panda_end_effector() -> None:
    from ...model.end_effectors import register_packaged_tcp

    register_packaged_tcp(
        LIBERO_FRANKA_PANDA.tcp_config_id,
        lambda: (_default_panda_profile().gripper, None),
    )


_register_libero_panda_end_effector()


def __getattr__(name: str):
    if name == "LIBERO_FRANKA_PANDA_PROFILE":
        return _default_panda_profile()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ARM_ID",
    "ARM_JOINTS",
    "BASE_LINK",
    "GRIPPER_JOINTS",
    "LIBERO_FRANKA_PANDA",
    "PANDA_CAMERA_EXAMPLES",
    "PANDA_DEFAULT_ACTION_TIMEOUT_S",
    "PANDA_DEFAULT_WAIT_TIMEOUT_S",
    "PANDA_EMPTY_CLOSE_APERTURE_M",
    "PANDA_FULL_OPEN_APERTURE_M",
    "PANDA_GRIPPER_SPEED_SCALE",
    "PANDA_LIMITS_PROFILES",
    "PANDA_MASK_LINK",
    "PANDA_MASK_REACH_M",
    "PANDA_MAX_TIMEOUT_S",
    "PANDA_OVERVIEW_CAMERA",
    "PANDA_STALL_GUARD_SEMANTICS",
    "PANDA_SUPPORTS",
    "PANDA_VERTICAL_WINDOW_SEMANTICS",
    "PANDA_WRIST_CAMERAS",
    "TCP_LINK",
    "WRIST_CAMERA_PARENT_LINK",
    "fk",
    "fk_all",
    "ik",
    "ik_position_priority",
    "joint_limits_rad",
    "libero_franka_panda_profile",
    "load_model",
    "panda_chain",
    "panda_defaults",
    "panda_facts",
]
