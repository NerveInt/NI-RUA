"""Own one RoboDojo JSONL worker, cancellation channel, and GPU OOM exit evidence.

Generic subprocess framing and lifecycle live in ``ni_rua.robot.ipc.process``;
this module specializes native CUDA/PhysX/Warp OOM classification for RoboDojo.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import select
import subprocess
import threading
import time
import uuid

from ...ipc.process import WorkerProcess
from ...ipc.protocol import WIRE_PREFIX

GPU_OOM_FAILURE = "robot_gpu_out_of_memory: native GPU allocation failed; robot status unknown; do not replay action"
NATIVE_GPU_OOM_SIGNATURES = (
    "ERROR_OUT_OF_DEVICE_MEMORY",
    "Out of GPU memory allocating resource",
    "CUDA error: out of memory",
    "CUDA out of memory",
    "CUDA_ERROR_OUT_OF_MEMORY",
    "torch.OutOfMemoryError:",
    "torch.cuda.OutOfMemoryError:",
    "Warp CUDA error 2: out of memory",
)
WARP_CUDA_ALLOC_OOM = re.compile(r"Failed to allocate \d+ bytes on device 'cuda(?::\d+)?'")
# 仅匹配 worker 返回的原生异常头或 Warp CUDA 显存分配失败签名，绝不匹配被引用的普通文本。
GPU_OOM_REPLY = re.compile(
    r"^(?:backend_error: )?(?:(?:torch\.)?(?:cuda\.)?OutOfMemoryError: "
    r"|RuntimeError: Failed to allocate \d+ bytes on device 'cuda(?::\d+)?')"
)


def native_gpu_oom_signature(text: str) -> str | None:
    for signature in NATIVE_GPU_OOM_SIGNATURES:
        if signature in text:
            return signature
    if "PxgCudaDeviceMemoryAllocator failed to allocate memory" in text and "Result = 2" in text:
        return "PxgCudaDeviceMemoryAllocator allocation failed; Result = 2"
    if WARP_CUDA_ALLOC_OOM.search(text):
        return "Warp CUDA allocation failed on device cuda"
    return None


def gpu_oom_reply(message: dict) -> bool:
    """A planner/torch OOM is caught by the worker and returned as an error or receipt reason."""
    reply = message.get("result")
    reason = reply.get("reason") if isinstance(reply, dict) else None
    return any(isinstance(text, str) and GPU_OOM_REPLY.match(text) for text in (message.get("error"), reason))


def _classify_robodojo_log_line(line: str) -> str | None:
    return GPU_OOM_FAILURE if native_gpu_oom_signature(line) else None


def _classify_robodojo_reply(message: dict) -> str | None:
    return GPU_OOM_FAILURE if gpu_oom_reply(message) else None


class RobotProcess(WorkerProcess):
    """RoboDojo worker transport with CUDA/Omniverse/Warp OOM failure classification."""

    log_failure_classifier = staticmethod(_classify_robodojo_log_line)
    reply_failure_classifier = staticmethod(_classify_robodojo_reply)


__all__ = [
    "GPU_OOM_FAILURE",
    "GPU_OOM_REPLY",
    "NATIVE_GPU_OOM_SIGNATURES",
    "RobotProcess",
    "WARP_CUDA_ALLOC_OOM",
    "WIRE_PREFIX",
    "gpu_oom_reply",
    "native_gpu_oom_signature",
]
