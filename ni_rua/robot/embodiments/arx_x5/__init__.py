"""ARX-X5 embodiment package: profile, kinematics chain, mounts, and end-effector registration."""
from .profile import (
    ARX_X5,
    ROBODOJO_REQUEST_DURATION_LIMIT_S,
    RoboDojoRobotBinding,
    X5_CAMERA_EXAMPLES,
    X5_CAPSULE_RADII_M,
    X5_DEFAULT_ACTION_TIMEOUT_S,
    X5_DEFAULT_WAIT_TIMEOUT_S,
    X5_FEEDBACK_UNSUPPORTED,
    X5_GRIPPER_SPEED_SCALE,
    X5_LIMITS_PROFILES,
    X5_MASK_REACH_M,
    X5_MAX_TIMEOUT_S,
    X5_OVERVIEW_CAMERA,
    X5_STALL_GUARD_SEMANTICS,
    X5_SUPPORTS,
    X5_VERTICAL_WINDOW_SEMANTICS,
    X5_WRIST_CAMERAS,
    _default_profile,
    arx_x5_chain,
    arx_x5_default_mounts,
    arx_x5_profile,
    get_robodojo_robot,
    physical_facts,
)


def _register_arx_x5_end_effector() -> None:
    from ...model.end_effectors import register_packaged_tcp

    register_packaged_tcp(ARX_X5.tcp_config_id, lambda: (_default_profile().gripper, None))


_register_arx_x5_end_effector()


def __getattr__(name: str):
    if name == "ARX_X5_PROFILE":
        return _default_profile()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ARX_X5",
    "ROBODOJO_REQUEST_DURATION_LIMIT_S",
    "RoboDojoRobotBinding",
    "X5_CAMERA_EXAMPLES",
    "X5_CAPSULE_RADII_M",
    "X5_DEFAULT_ACTION_TIMEOUT_S",
    "X5_DEFAULT_WAIT_TIMEOUT_S",
    "X5_FEEDBACK_UNSUPPORTED",
    "X5_GRIPPER_SPEED_SCALE",
    "X5_LIMITS_PROFILES",
    "X5_MASK_REACH_M",
    "X5_MAX_TIMEOUT_S",
    "X5_OVERVIEW_CAMERA",
    "X5_STALL_GUARD_SEMANTICS",
    "X5_SUPPORTS",
    "X5_VERTICAL_WINDOW_SEMANTICS",
    "X5_WRIST_CAMERAS",
    "arx_x5_chain",
    "arx_x5_default_mounts",
    "arx_x5_profile",
    "get_robodojo_robot",
    "physical_facts",
]
