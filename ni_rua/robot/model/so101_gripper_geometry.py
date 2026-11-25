"""Backward-compatible alias for ``ni_rua.robot.embodiments.so101.gripper_cad``."""
import sys

from ..embodiments.so101 import gripper_cad as _impl

sys.modules[__name__] = _impl
