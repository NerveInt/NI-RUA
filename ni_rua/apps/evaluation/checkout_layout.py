"""Checkout layout and queue process detection for NI-RUA evaluation queues."""
from __future__ import annotations

from pathlib import Path

NI_RUA_CHECKOUT_LAYOUT = {
    "gateway": "ni_rua/apps/gateway/cli.py",
    "config": "ni_rua/core/config.py",
    "module": "ni_rua.apps.gateway",
    "package": "ni_rua",
    "env_prefix": "NI_RUA_",
}

CHECKOUT_LAYOUTS = (NI_RUA_CHECKOUT_LAYOUT,)
QUEUE_MODULES = ("ni_rua.apps.evaluation.queue",)
QUEUE_SCRIPTS = ("benchmark_queue.py", "ni-rua-queue")


def checkout_layout(root):
    """Return the entry-point layout of a checkout, or None when it has no gateway."""
    root_path = Path(root)
    for layout in CHECKOUT_LAYOUTS:
        if (root_path / layout["gateway"]).is_file():
            return layout
    return None


def checkout_environment(environment, layout):
    """Return the launch environment for an NI-RUA checkout."""
    return environment


def is_queue_process_command(command) -> bool:
    """Return True when ``command`` argv (after Python flags) launches an NI-RUA queue."""
    if not command:
        return False
    if command[:2] in (["-m", module] for module in QUEUE_MODULES):
        return True
    return Path(command[0]).name in QUEUE_SCRIPTS


__all__ = [
    "CHECKOUT_LAYOUTS",
    "NI_RUA_CHECKOUT_LAYOUT",
    "QUEUE_MODULES",
    "QUEUE_SCRIPTS",
    "checkout_environment",
    "checkout_layout",
    "is_queue_process_command",
]
