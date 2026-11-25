"""Per-robot embodiment packages (1 self-contained package per robot arm model)."""
from . import arx_x5, libero_franka_panda, so101
from .arx_x5 import (
    ARX_X5,
    RoboDojoRobotBinding,
    arx_x5_chain,
    arx_x5_default_mounts,
    arx_x5_profile,
    get_robodojo_robot,
    physical_facts,
)
from .libero_franka_panda import LIBERO_FRANKA_PANDA, libero_franka_panda_profile, panda_chain
from .so101 import SO101, so101_profile


def __getattr__(name: str):
    if name == "ARX_X5_PROFILE":
        return arx_x5.ARX_X5_PROFILE
    if name == "LIBERO_FRANKA_PANDA_PROFILE":
        return libero_franka_panda.LIBERO_FRANKA_PANDA_PROFILE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ARX_X5",
    "LIBERO_FRANKA_PANDA",
    "RoboDojoRobotBinding",
    "SO101",
    "arx_x5",
    "arx_x5_chain",
    "arx_x5_default_mounts",
    "arx_x5_profile",
    "get_robodojo_robot",
    "libero_franka_panda",
    "libero_franka_panda_profile",
    "panda_chain",
    "physical_facts",
    "so101",
    "so101_profile",
]
