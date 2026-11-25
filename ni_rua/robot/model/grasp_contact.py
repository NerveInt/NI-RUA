"""Backward-compatible alias for ``ni_rua.robot.embodiments.arx_x5.grasp_contact``."""
import sys

from ..embodiments.arx_x5 import grasp_contact as _impl

sys.modules[__name__] = _impl
