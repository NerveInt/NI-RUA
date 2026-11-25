"""SO-101 calibration bundle: every workflow result stored together, versioned, with an active pointer.

Layout under ``deployment.calibration_bundle_dir(deployment)`` (default ``calibration/`` next to the
deployment ``camera_calibration_file``)::

    versions/<UTC>/manifest.json            version record (mode, steps done and when, joint offsets, previous)
    versions/<UTC>/camera_calibration.json  the camera file this version installs
    versions/<UTC>/joint_calibration.json   table height + joint zero offsets (also inside the camera file)
    versions/<UTC>/touches.json             touch-off contacts the joint solve used
    versions/<UTC>/reports/<step>.json      per-step result (measures, criteria, pass/fail)
    runs/<UTC>-<id>/                        workflow working directories (sessions, candidate, run.json)
    active                                  name of the active version (one line, replaced atomically)

The deployment ``camera_calibration_file`` stays the single file the worker reads: a version is installed by
copying its camera file onto that path with one ``os.replace`` and reloading; a reload failure restores the
previous bytes. A version directory is written under a temporary name and renamed once complete, so a
half-written version never appears. Operator/audit records only; nothing here enters model context.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import uuid

MANIFEST_SCHEMA = "ni_rua.so101.calibration_version.v1"
ACTIVE_FILE = "active"
CAMERA_FILE = "camera_calibration.json"


def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _write_atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_json(path, value):
    _write_atomic(path, json.dumps(value, indent=2, ensure_ascii=False, default=_plain) + "\n")


def _plain(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(type(value).__name__)


class Bundle:
    """Versioned calibration store. ``root``: bundle directory; ``camera_file``: deployment camera file path."""

    def __init__(self, root, camera_file):
        self.root = Path(root)
        self.camera_file = Path(camera_file) if camera_file else None
        self.versions_dir = self.root / "versions"
        self.runs_dir = self.root / "runs"

    # ---- read side ----
    def active_version(self):
        """Name of the active version, or None (no pointer, or it names a missing version)."""
        try:
            name = (self.root / ACTIVE_FILE).read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return name if name and (self.versions_dir / name / "manifest.json").is_file() else None

    def manifest(self, name):
        return json.loads((self.versions_dir / name / "manifest.json").read_text(encoding="utf-8"))

    def version_file(self, name, relative):
        return self.versions_dir / name / relative

    def read_version_json(self, name, relative, default=None):
        try:
            return json.loads(self.version_file(name, relative).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return default

    def versions(self, limit=None):
        """Version manifests, newest first by ``sequence`` (each with ``name``)."""
        if not self.versions_dir.is_dir():
            return []
        out = []
        for path in self.versions_dir.iterdir():
            if path.name.startswith(".") or not (path / "manifest.json").is_file():
                continue
            try:
                out.append(dict(self.manifest(path.name), name=path.name))
            except (OSError, ValueError):
                continue
        out.sort(key=lambda item: (item.get("sequence") or 0, item["name"]), reverse=True)
        return out if limit is None else out[:limit]

    # ---- write side ----
    def create_version(self, *, camera_file, manifest, files=None, stamp=None):
        """Write a complete version directory and return its name (not activated).

        ``camera_file``: path of the camera calibration to store; ``manifest``: dict (schema/name/created_at
        are filled in); ``files``: ``{relative_path: json-able}`` extra records.
        """
        stamp = stamp or utc_stamp()
        name, suffix = stamp, 1
        while (self.versions_dir / name).exists():
            name, suffix = f"{stamp}-{suffix}", suffix + 1
        staging = self.versions_dir / f".staging-{uuid.uuid4().hex}"
        staging.mkdir(parents=True)
        try:
            shutil.copy2(camera_file, staging / CAMERA_FILE)
            for relative, value in (files or {}).items():
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_plain) + "\n", encoding="utf-8")
            sequence = 1 + max((item.get("sequence") or 0 for item in self.versions()), default=0)
            record = dict(manifest, schema=MANIFEST_SCHEMA, name=name, sequence=sequence,
                          created_at=manifest.get("created_at") or utc_now())
            (staging / "manifest.json").write_text(json.dumps(record, indent=2, ensure_ascii=False, default=_plain) + "\n",
                                                   encoding="utf-8")
            os.rename(staging, self.versions_dir / name)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return name

    def set_active(self, name):
        if not (self.versions_dir / name / "manifest.json").is_file():
            raise ValueError(f"calibration_version_unknown:{name}")
        _write_atomic(self.root / ACTIVE_FILE, name + "\n")

    def import_existing(self, joint_offsets_rad, stamp=None):
        """Store the current camera file as an ``imported`` version when no version is active yet.

        Gives the first workflow apply a rollback target. Returns the version name, or None when there is
        nothing to import (no camera file, or a version is already active).
        """
        if self.active_version() is not None or self.camera_file is None or not self.camera_file.is_file():
            return None
        try:
            document = json.loads(self.camera_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        joint = document.get("joint_calibration")
        name = self.create_version(camera_file=self.camera_file, stamp=(stamp or utc_stamp()) + "-imported", manifest={
            "source": "imported_existing_file", "mode": None, "changes": [], "steps": {},
            "calibration_created_at": document.get("created_at"), "joint_offsets_rad": dict(joint_offsets_rad),
            "table_z_m": (joint or {}).get("table_z_m"), "previous": None},
            files={"joint_calibration.json": joint} if joint else None)
        self.set_active(name)
        return name


def install_camera_file(adapter, source, active_path):
    """Make ``source`` the active camera calibration: validate, ``os.replace``, reload; restore on failure.

    Unlike the one-click ``calibrate`` install this never carries records over from the old file: a version
    is installed exactly as stored. Returns the reload receipt; raises ``ValueError`` and leaves the previous
    file and the loaded calibration in place when the file would not load.
    """
    active_path = Path(active_path)
    adapter.validate_calibration_file(source)  # same checks as reload, before touching the active file
    previous = active_path.read_bytes() if active_path.exists() else None
    temporary = active_path.with_name(f".{active_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, active_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    reload = adapter.reload_calibration()
    if reload["status"] != "reloaded":
        if previous is None:
            active_path.unlink(missing_ok=True)
        else:
            restore = active_path.with_name(f".{active_path.name}.{uuid.uuid4().hex}.tmp")
            try:
                restore.write_bytes(previous)
                os.replace(restore, active_path)
            except BaseException:
                restore.unlink(missing_ok=True)
                raise
        adapter.reload_calibration()
        raise ValueError(f"calibration_reload_failed_after_install: {reload.get('reason')}")
    return reload
