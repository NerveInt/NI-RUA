"""RoboDojo worker JSONL framing and dispatch over shared ``ni_rua.robot.ipc``."""
from ...ipc.protocol import WIRE_PREFIX, serve_worker

__all__ = ["WIRE_PREFIX", "serve_worker"]
