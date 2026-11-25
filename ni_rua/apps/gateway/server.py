"""Process-level run supervision facade delegating to :class:`RunSupervisor`."""
from __future__ import annotations

import os
import signal

from ..library.experiments import allocate_run, finish_run, register_run, write_runtime
from ..remote.manager import RemoteManager, start_services
from . import supervisor as _supervisor
from .supervisor import (
    ProcessSignals,
    RunSupervisor,
    rebind_remote_record as _rebind_remote_record,
)
from .workbench import DEFAULT_PROMPT, Workbench, _UNSET


def shutdown_signals():
    return _supervisor.shutdown_signals(signal_module=signal)


def _say(text):
    return _supervisor._say(text, os_module=os)


def _supervisor_instance(*, use_serve_patch=True) -> RunSupervisor:
    return RunSupervisor(
        workbench_factory=Workbench,
        remote_manager_cls=RemoteManager,
        start_services_fn=start_services,
        say_fn=_say,
        serve_run_fn=_serve if use_serve_patch else None,
        allocate_run_fn=allocate_run,
        register_run_fn=register_run,
        finish_run_fn=finish_run,
        write_runtime_fn=write_runtime,
    )


async def serve(args):
    """Serve one run, then the successor run of every handoff it accepts (same process, same port)."""
    return await _supervisor_instance().serve(args)


async def _serve_successor(successor, remote):
    """Serve the successor run of a handoff, falling back once on startup failure."""
    return await _supervisor_instance().serve_successor(successor, remote)


async def _serve(args, remote, launch=None):
    """Serve one run through :class:`RunSupervisor`."""
    return await _supervisor_instance(use_serve_patch=False).serve_run(args, remote, launch=launch)


__all__ = [
    "DEFAULT_PROMPT",
    "ProcessSignals",
    "RemoteManager",
    "RunSupervisor",
    "Workbench",
    "_UNSET",
    "_rebind_remote_record",
    "_say",
    "_serve",
    "_serve_successor",
    "allocate_run",
    "finish_run",
    "os",
    "register_run",
    "serve",
    "shutdown_signals",
    "signal",
    "start_services",
    "write_runtime",
]
