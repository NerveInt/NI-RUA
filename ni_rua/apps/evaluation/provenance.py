"""Bind queued attempts to source bytes, including non-Git frozen checkouts."""

import hashlib
from pathlib import Path

from ni_rua.core.contracts import digest
from .checkout_layout import (
    CHECKOUT_LAYOUTS,
    checkout_environment,
    checkout_layout,
)


def source_binding(root):
    root = Path(root).resolve()
    files = {}
    for directory in ("ni_rua", "scripts", "configs", "third_party", "benchmarks"):
        if (root / directory).is_symlink():
            raise ValueError("source_binding_symlink_denied:" + str(root / directory))
        for path in sorted((root / directory).rglob("*")):
            if "__pycache__" in path.parts:
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError("source_binding_symlink_denied:" + str(path))
            if path.is_file():
                with path.open("rb") as stream:
                    files[path.relative_to(root).as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
    for name in ("pyproject.toml", "requirements.lock.txt", "environment.yml"):
        path = root / name
        if path.exists():
            if path.is_symlink():
                raise ValueError("source_binding_symlink_denied:" + str(path))
            files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not any(layout["gateway"] in files for layout in CHECKOUT_LAYOUTS):
        raise ValueError("source_binding_gateway_missing:" + str(root))
    return {"version": 1, "files": files, "sha256": digest(files)}


def verify_source(root, expected):
    actual = source_binding(root)
    if actual != expected:
        raise ValueError("queue_source_changed:" + str(root))
    return actual["sha256"]


def bind_sources(cases, expected=None):
    roots = {str(Path(case["cwd"]).resolve()) for case in cases}
    if expected is not None:
        if not isinstance(expected, dict) or not roots <= expected.keys():
            raise ValueError("queue_source_binding_missing")
        for root in roots:
            verify_source(root, expected[root])
        return expected
    return {root: source_binding(root) for root in sorted(roots)}
