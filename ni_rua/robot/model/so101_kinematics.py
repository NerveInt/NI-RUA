"""Backward-compatible alias for ``ni_rua.robot.embodiments.so101.profile``."""
import sys

from ..embodiments.so101 import profile as _impl

sys.modules[__name__] = _impl
