"""SO-101 calibration workflow runner (worker side): executes the steps of ``calibration_workflow``.

Runs inside the worker interpreter against a live ``SO101Adapter``. Motion steps (S1, S2, S4, S6) run on
the worker main thread like ``go_home``; ``hold``/E-STOP and ``calib_step_cancel`` preempt them through the
adapter's stop generation. Status, readiness and the staleness hint are answered off the main thread and
never touch the bus (latest frames and the telemetry joint sample only).

State of one workflow run lives in ``<bundle>/runs/<UTC>-<id>/run.json`` (operator/audit record); results
are stored in the calibration bundle only by :meth:`CalibrationWorkflow.apply`. Every threshold is a
deployment ``calibration`` field. Nothing here is model-visible.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import threading
import time
import uuid

import numpy as np

from . import state_machine as wf
from .bundle import Bundle, install_camera_file, utc_now, utc_stamp, write_json
from ni_rua.robot.backends.so101.camera_devices import binding_summary
from ni_rua.robot.backends.so101.deployment import calibration_bundle_dir, write_joint_offsets
from ni_rua.robot.backends.so101.driver import ARM_JOINTS

RUN_SCHEMA = "ni_rua.so101.calibration_run.v1"
PROGRESS_KEEP = 40
OVERLAY_FILES = {"wrist": "calib_wrist.jpg", "fixed": "calib_fixed.jpg"}  # telemetry_dir, operator preview only
STATUS_FILE = "calibration_workflow.json"  # telemetry_dir: last status snapshot (console badge without IPC)
BOARD_FIELDS = ("squares_x", "squares_y", "square_m", "marker_m", "dictionary")


class StepRefused(ValueError):
    """A step cannot start; ``reason`` is stable and shown to the operator."""


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _rotation_angle(R):
    return float(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))


class CalibrationWorkflow:
    """One adapter's calibration workflow: run state, step execution, apply and rollback."""

    def __init__(self, adapter):
        self.adapter = adapter
        self.settings = adapter.deployment["calibration"]
        root = calibration_bundle_dir(adapter.deployment)
        self.bundle = None if root is None else Bundle(root, adapter.calibration_path)
        self.lock = threading.RLock()
        self.run = None
        self.capture = None  # HandheldCapture of S3
        self.running_step = None
        self._stale_hint = None
        self._stale_checked_at = None
        self._overlay_at = None
        self._load_latest_run()

    # ------------------------------------------------------------ run state ---
    def _load_latest_run(self):
        if self.bundle is None or not self.bundle.runs_dir.is_dir():
            return
        for path in sorted(self.bundle.runs_dir.iterdir(), reverse=True):
            try:
                run = json.loads((path / "run.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if run.get("schema") != RUN_SCHEMA or run.get("closed"):
                return
            for step in run["steps"].values():
                if step["status"] == "running":  # the worker died mid-step: never shown as running
                    step.update(status="failed", reason="worker_restarted_during_step", finished_at=utc_now())
            self.run = run
            return

    def _run_dir(self):
        return Path(self.run["directory"])

    def _save(self):
        if self.run is not None:
            write_json(self._run_dir() / "run.json", _plain(self.run))

    def _run_settings_binding(self):
        """Compare this run's recorded calibration controls with the loaded deployment; no I/O or motion."""
        recorded = (self.run or {}).get("calibration_settings")
        if not isinstance(recorded, dict):
            return {"status": "unknown", "reason": "calibration_settings_unrecorded", "changed_keys": []}
        changed = sorted(key for key in set(recorded) | set(self.settings)
                         if key not in recorded or key not in self.settings or recorded[key] != self.settings[key])
        return {"status": "changed" if changed else "matched",
                "reason": "calibration_settings_changed" if changed else None, "changed_keys": changed}

    def _require_run_settings(self):
        binding = self._run_settings_binding()
        if binding["status"] != "matched":
            raise StepRefused(binding["reason"] + ": start a new run with the reviewed deployment settings")

    def default_board(self):
        """Board configured in deployment (else board.json / active calibration file, else packaged board) and thickness."""
        return self._default_board_with_source()[0]

    def _default_board_with_source(self):
        """Resolve defaults plus operator-only source evidence, preserving per-field precedence."""
        from . import charuco as cal
        spec = dict(cal.DEFAULT_BOARD_SPEC)
        thickness = self.settings["board_thickness_m"]
        source = {"geometry": "packaged_board", "thickness": "deployment.calibration.board_thickness_m",
                  "fields": {key: "packaged_board" for key in BOARD_FIELDS}, "fallback_errors": []}
        try:
            if self.adapter.calibration_path and Path(self.adapter.calibration_path).is_file():
                spec = json.loads(Path(self.adapter.calibration_path).read_text())["board"]
                source["geometry"] = "active_camera_calibration.board"
                source["fields"].update({key: source["geometry"] for key in BOARD_FIELDS})
        except (OSError, ValueError, KeyError) as error:
            source["fallback_errors"].append("active_camera_calibration:" + type(error).__name__)
        board_json = Path(self.adapter.calibration_path).parent / "board.json" if self.adapter.calibration_path else None
        if board_json is not None and board_json.is_file():
            try:
                raw = json.loads(board_json.read_text(encoding="utf-8"))
                spec = dict(spec, **{k: raw[k] for k in BOARD_FIELDS if k in raw})
                if any(key in raw for key in BOARD_FIELDS):
                    source["geometry"] = "adjacent_board.json"
                    source["fields"].update({key: "adjacent_board.json" for key in BOARD_FIELDS if key in raw})
                if raw.get("thickness_m") is not None:
                    thickness = float(raw["thickness_m"])
                    source["thickness"] = "adjacent_board.json"
            except (OSError, ValueError, TypeError) as error:
                source["fallback_errors"].append("adjacent_board.json:" + type(error).__name__)
        deployed = self.adapter.deployment.get("board")
        if isinstance(deployed, dict):
            spec = dict(spec, **{k: deployed[k] for k in BOARD_FIELDS if k in deployed})
            source["geometry"] = "deployment.board"
            source["fields"].update({key: "deployment.board" for key in BOARD_FIELDS if key in deployed})
            if deployed.get("thickness_m") is not None:
                thickness = float(deployed["thickness_m"])
                source["thickness"] = "deployment.board"
        source["configured_board_file"] = self.adapter.deployment.get("board_file")
        source["fields"]["thickness_m"] = source["thickness"]
        return dict({key: spec[key] for key in BOARD_FIELDS}, thickness_m=thickness), source

    @staticmethod
    def validate_board(board):
        """``{squares_x, squares_y, square_m (m), marker_m (m), dictionary, thickness_m (m)}`` -> normalized."""
        from . import charuco as cal
        if not isinstance(board, dict) or set(board) - {*BOARD_FIELDS, "thickness_m", "type"}:
            raise StepRefused("board_spec_invalid: expected squares_x, squares_y, square_m, marker_m, dictionary, thickness_m")
        thickness = board.get("thickness_m")
        if type(thickness) not in (int, float) or not 0 <= thickness <= 0.05:
            raise StepRefused("board_spec_invalid: thickness_m must be in [0, 0.05] m")
        try:
            spec = cal.validate_board_spec({key: board[key] for key in BOARD_FIELDS})
        except (KeyError, ValueError, TypeError) as error:
            raise StepRefused(f"board_spec_invalid: {error}") from error
        return dict({key: spec[key] for key in BOARD_FIELDS}, thickness_m=float(thickness))

    def begin(self, mode, changes=(), board=None):
        """Start a new workflow run (closes the previous one). Returns the status snapshot."""
        if self.bundle is None:
            raise StepRefused("calibration_bundle_not_configured: set camera_calibration_file or calibration_bundle_dir")
        if self.running_step is not None or (self.capture is not None and self.capture.running):
            raise StepRefused("calibration_step_running")
        plan = wf.plan_for(mode, changes)
        if board is not None:
            board = self.validate_board(board)
            board_source = {"geometry": "operator_begin_request", "thickness": "operator_begin_request"}
        else:
            board, board_source = self._default_board_with_source()
        with self.lock:
            if self.run is not None:
                self.run["closed"] = True
                self._save()
            stamp = utc_stamp()
            run_id = f"{stamp}-{uuid.uuid4().hex[:6]}"
            directory = self.bundle.runs_dir / run_id
            directory.mkdir(parents=True)
            self.run = {"schema": RUN_SCHEMA, "run_id": run_id, "directory": str(directory), "created_at": utc_now(),
                        "mode": mode, "changes": list(changes or ()), "plan": plan, "board": board, "closed": False,
                        "board_source": board_source,
                        "calibration_settings": deepcopy(self.settings),
                        "steps": {step: {"status": "not_started", "started_at": None, "finished_at": None, "reason": None,
                                         "progress": [], "result": None} for step in wf.STEP_IDS},
                        "joint_offsets_at_start_rad": self._offsets(), "candidate": None, "applied_version": None,
                        "previous_version": None}
            self._save()
        return self.status()

    def _offsets(self):
        return {name: float(self.adapter.deployment["joint_mapping"][name]["offset_rad"]) for name in ARM_JOINTS}

    def _step(self, step):
        return self.run["steps"][step]

    def _usable(self, step):
        """True when ``step`` passed in this run and is not stale."""
        return self.run is not None and self._step(step)["status"] == "passed"

    def _source_evidence(self, step, source):
        """Describe a calibration input's workflow source and reject invalid active reuse.

        ``step`` is S1/S2/S3; ``source`` is that step or ``active``. Failed/stale/running stages
        cannot be replaced by an older active input. Recalibration stages selected by the change
        plan must be completed. First-run manual skips remain supported but are explicitly marked
        planned and not_started; they cannot be mistaken for completion of the full plan.
        """
        status = self._step(step)["status"] if self.run else "not_started"
        planned = step in (self.run or {}).get("plan", [])
        if source == "active":
            if status in ("failed", "stale", "running"):
                raise StepRefused(f"calibration_source_unusable:{step}:{status}; repeat this stage")
            if status != "passed" and planned and (self.run or {}).get("mode") == "recalibrate":
                raise StepRefused(f"calibration_source_required_by_plan:{step}; complete this stage")
        return {"source": source, "step": step, "step_status": status, "planned": planned}

    def _progress(self, step):
        def push(update):
            with self.lock:
                record = self._step(step)
                record["progress"] = (record["progress"] + [dict(_plain(update), at=utc_now())])[-PROGRESS_KEEP:]
        return push

    # ------------------------------------------------------------ readiness ---
    def readiness(self, step, *, fresh=False, cache=None):
        """``{check_id: {"ok": bool|None, "label", "kind", "detail"}}`` for ``step``. ``fresh``: board check on new
        frames (start time) instead of the latest preview frames. Operator checks report ``ok=None``.
        ``cache``: dict shared across calls so one status snapshot measures each check once."""
        out = {}
        cache = {} if cache is None else cache
        for check in wf.STEP_BY_ID[step]["readiness"]:
            spec = wf.READINESS[check]
            ok, detail = None, None
            if spec["kind"] == "auto":
                if check not in cache:
                    cache[check] = self._auto_check(check, fresh)
                ok, detail = cache[check]
            out[check] = dict(ok=ok, label=spec["label"], kind=spec["kind"], detail=detail)
        return out

    def _sample_age(self):
        sample = self.adapter.last_sample
        return None if sample is None else self.adapter.clock.now() - sample["t"]

    def _camera_ok(self, name):
        camera = self.adapter.cameras.get(name)
        config = self.adapter.deployment["cameras"].get(name)
        if camera is None or config is None:
            return False, "not_configured"
        latest = camera.latest()
        if latest is None:
            return False, "no_frame_yet"
        image, stamp = latest
        shape = np.asarray(image).shape
        if shape[:2] != (config["height"], config["width"]):
            return False, f"frame {shape[1]}x{shape[0]} px != configured {config['width']}x{config['height']} px"
        age = self.adapter.clock.now() - stamp
        if age > self.settings["selfcheck_max_telemetry_age_s"]:
            return False, f"last frame {age:.1f} s old"
        return True, None

    def _auto_check(self, check, fresh):
        adapter = self.adapter
        if check == "torque_on":
            ok = adapter.torque_known is True and not adapter.relaxed
            return ok, None if ok else ("torque_off" if adapter.relaxed or adapter.torque_known is False else "torque_unknown")
        if check == "telemetry_fresh":
            age = self._sample_age()
            ok = age is not None and age <= self.settings["selfcheck_max_telemetry_age_s"]
            return ok, None if age is None else f"{age:.2f} s"
        if check in ("cameras_streaming", "fixed_streaming"):
            names = ["fixed"] if check == "fixed_streaming" else sorted(adapter.deployment["cameras"])
            results = {name: self._camera_ok(name) for name in names}
            bad = {name: detail for name, (ok, detail) in results.items() if not ok}
            return not bad, "; ".join(f"{name}: {detail}" for name, detail in bad.items()) or None
        if check == "board_both_visible":
            board = (self.run or {}).get("board") or self.default_board()
            result = adapter.board_check(fresh=fresh, board_spec={key: board[key] for key in BOARD_FIELDS})
            return bool(result["ok"]), " ".join(f"{name}={count}" for name, count in result["corners"].items()) + \
                f" (需要 ≥ {result['min_board_corners']})"
        if check == "touches_available":
            try:
                report, source = self._touch_report()
                return bool(report.get("touches")), source
            except (StepRefused, OSError, ValueError, KeyError, TypeError) as error:
                return False, str(error)
        if check == "wrist_calibration_available":
            try:
                _, source = self._wrist()
                evidence = self._source_evidence("S2", source)
                return True, f"source={source}; S2={evidence['step_status']}; planned={evidence['planned']}"
            except (StepRefused, OSError, ValueError, KeyError, TypeError) as error:
                return False, str(error)
        if check == "fixed_calibration_available":
            try:
                _, source = self._fixed_intrinsics()
                evidence = self._source_evidence("S3", source)
                return True, f"source={source}; S3={evidence['step_status']}; planned={evidence['planned']}"
            except (StepRefused, OSError, ValueError, KeyError, TypeError) as error:
                return False, str(error)
        if check == "joint_inputs_ready":
            return self._usable("S4"), None
        if check == "review_ready":
            return self._usable("S5"), None
        if check == "applied":
            try:
                binding = self._verification_binding()
                return True, binding["version"]
            except (StepRefused, OSError, ValueError, KeyError, TypeError) as error:
                return False, str(error)
        raise ValueError(f"unknown readiness check {check}")

    def _verification_binding(self):
        """Read-only S6 binding to the installed version and effective joint mapping.

        Returns an internal comparison snapshot, including table height in base-frame metres.
        No default version or mapping is inferred for old runs. Missing mapping-sign evidence
        requires a new apply; the old archive remains readable. No motor reads or writes.
        """
        version = (self.run or {}).get("applied_version")
        binding = self._version_binding(version)
        if binding["table_z_m"] is None:
            raise StepRefused("verification_calibration_mismatch:table_height_not_calibrated")
        return binding

    def _version_binding(self, version):
        """Validate ``version`` (bundle name, no default) against current loaded and persisted state.

        Shared by S6 and operator status; it does not require the newest workflow to own the version.
        Returns the internal camera/mapping snapshot and base-frame table height in metres (None for
        camera-only calibration, which cannot pass S6). Raises on missing or mismatched binding
        evidence. Never reads the motor bus or modifies calibration state.
        """
        if not version or self.bundle is None or self.bundle.active_version() != version:
            raise StepRefused("verification_calibration_mismatch:active_version")
        path = self.bundle.version_file(version, "camera_calibration.json")
        stored = path.read_bytes()
        if not self.adapter.calibration_path or Path(self.adapter.calibration_path).read_bytes() != stored:
            raise StepRefused("verification_calibration_mismatch:installed_file")
        expected, identity = self.adapter._load_calibration(path)
        if self.adapter.calibration_error or self.adapter.calibration_identity is None:
            raise StepRefused("verification_calibration_mismatch:not_loaded")
        if self.adapter.calibrated and _plain(self.adapter.camera_calibration) != _plain(expected):
            raise StepRefused("verification_calibration_mismatch:loaded_camera")
        if any(self.adapter.calibration_identity.get(k) != v for k, v in identity.items() if k != "file"):
            raise StepRefused("verification_calibration_mismatch:loaded_identity")
        manifest = self.bundle.manifest(version)
        joint = json.loads(stored).get("joint_calibration")
        offsets = self._offsets()
        if offsets != manifest.get("joint_offsets_rad") or (joint is not None and offsets != joint.get("joint_offsets_rad")):
            raise StepRefused("verification_calibration_mismatch:joint_offsets")
        mapping = {name: dict(self.adapter.deployment["joint_mapping"][name]) for name in ARM_JOINTS}
        if manifest.get("joint_mapping") is None:
            raise StepRefused("verification_calibration_mismatch:joint_mapping_not_recorded; reapply required")
        if mapping != manifest["joint_mapping"]:
            raise StepRefused("verification_calibration_mismatch:joint_mapping")
        driver_mapping = getattr(self.adapter.driver, "mapping", None)
        if isinstance(driver_mapping, dict) and any(driver_mapping.get(n) != m for n, m in mapping.items()):
            raise StepRefused("verification_calibration_mismatch:driver_mapping")
        if self.adapter.deployment_path:
            from ni_rua.robot.backends.so101.deployment import load_deployment
            disk = load_deployment(self.adapter.deployment_path)["joint_mapping"]
            if any(disk[n] != m for n, m in mapping.items()):
                raise StepRefused("verification_calibration_mismatch:deployment_file")
        table = None if joint is None else float(joint["table_z_m"])
        if (table is not None and not math.isfinite(table)) or manifest.get("table_z_m") != table:
            raise StepRefused("verification_calibration_mismatch:table_height")
        return {"version": version, "camera": _plain(expected), "joint_mapping": mapping, "table_z_m": table}

    def _binding_diagnostics(self, manifest):
        """Explain offset identity failures without granting compatibility or modifying state.

        Differences are effective-minus-recorded radians. The six-decimal test reproduces the old
        writer's exact serialization rule; it is not an angular tolerance or physical acceptance.
        Missing historical signs stay unknown. Re-running acquisition/solve/apply creates a fully
        recorded version while retaining the original archive and its verification result.
        """
        manifest = manifest if isinstance(manifest, dict) else {}
        recorded = manifest.get("joint_offsets_rad")
        current = self._offsets()
        known = isinstance(recorded, dict) and all(type(recorded.get(n)) in (int, float)
                    and math.isfinite(recorded[n]) for n in ARM_JOINTS)
        delta = {n: current[n] - recorded[n] for n in ARM_JOINTS} if known else None
        exact = current == recorded if known else None
        rounded = not exact and all(current[n] == round(recorded[n], 6) for n in ARM_JOINTS) if known else None
        mapping_recorded = isinstance((manifest or {}).get("joint_mapping"), dict)
        return {"offsets_exact": exact, "joint_mapping_recorded": mapping_recorded,
                "matches_legacy_six_decimal_serialization": rounded,
                "offset_difference_rad": delta,
                "max_abs_offset_difference_rad": None if delta is None else max(abs(v) for v in delta.values()),
                "recovery": "new_acquisition_solve_apply" if rounded or not mapping_recorded else None}

    def _active_document(self):
        from . import charuco as cal
        try:
            return cal.read_calibration_file(self.adapter.calibration_path)
        except (OSError, ValueError, KeyError, TypeError):
            return None

    # --------------------------------------------------------------- status ---
    def status(self):
        """Operator snapshot: definition, current run, bundle status, staleness hint. No bus access."""
        with self.lock:
            run = deepcopy(self.run)
        readiness, cache = {}, {}
        if run is not None:
            for step in wf.STEP_IDS:
                try:
                    readiness[step] = self.readiness(step, cache=cache)
                except Exception as error:  # readiness is advisory; never fail the status
                    readiness[step] = {"error": {"ok": False, "label": "检查失败", "kind": "auto",
                                                 "detail": f"{type(error).__name__}: {error}"}}
        capture = self.capture.snapshot() if self.capture is not None else None
        from .xy_verification import XY_SETTING_KEYS
        default_board, board_source = self._default_board_with_source()
        snapshot = _plain({"definition": wf.describe(self.settings), "run": run, "readiness": readiness,
                           "settings_binding": self._run_settings_binding(),
                           "xy_defaults": {key: self.settings[setting] for key, setting in XY_SETTING_KEYS.items()},
                           "plan_completion": None if run is None else {
                               "all_steps_passed": all(run["steps"][step]["status"] == "passed" for step in run["plan"]),
                               "not_passed_steps": [step for step in run["plan"] if run["steps"][step]["status"] != "passed"]},
                           "running_step": self.running_step, "handheld": capture, "bundle": self.bundle_status(),
                           "staleness": self.staleness_hint(), "default_board": default_board, "default_board_source": board_source,
                           "settings": {key: value for key, value in self.settings.items()
                                        if key in ("touch_points_xy_m", "verify_points_xy_m", "board_thickness_m",
                                                   "touch_video_fps", "touch_video_max_frames", "touch_video_max_duration_s")}})
        try:
            self._write_overlays(run)
            self.adapter._write_atomic(self.adapter.telemetry_dir / STATUS_FILE,
                                       json.dumps({"bundle": snapshot["bundle"], "staleness": snapshot["staleness"],
                                                   "running_step": snapshot["running_step"]}, allow_nan=False).encode())
        except Exception:
            pass
        return snapshot

    def bundle_status(self):
        identity = self.adapter.calibration_identity
        out = {"configured": self.bundle is not None, "present": identity is not None,
               "calibration_created_at": (identity or {}).get("created_at"), "residuals": (identity or {}).get("residuals"),
               "load_error": self.adapter.calibration_error, "active_version": None, "versions": [],
               "matches_active_version": None, "binding_error": None, "table_z_m": None, "has_joint_calibration": False,
               "fixed_intrinsics_source": None, "joint_offsets_rad": self._offsets(), "verification": None}
        try:
            document = json.loads(Path(self.adapter.calibration_path).read_text()) if self.adapter.calibration_path else {}
            out["table_z_m"] = (document.get("joint_calibration") or {}).get("table_z_m")
            out["has_joint_calibration"] = isinstance(document.get("joint_calibration"), dict)
            out["fixed_intrinsics_source"] = ((document.get("cameras") or {}).get("fixed") or {}).get("intrinsics_source")
        except (OSError, ValueError):
            pass
        if self.bundle is None:
            return out
        active = self.bundle.active_version()
        out["active_version"] = active
        manifest = self.bundle.read_version_json(active, "manifest.json") if active else None
        out["binding_diagnostics"] = self._binding_diagnostics(manifest) if manifest else None
        out["verification"] = manifest.get("verification") if isinstance(manifest, dict) else None
        s6 = ((self.run or {}).get("steps") or {}).get("S6") or {}
        result = s6.get("result") or {}
        if active and (self.run or {}).get("applied_version") == active and s6.get("status") == "failed" \
                and result.get("evidence_persisted") is False:
            # A successful earlier manifest write is historical evidence, not permission to
            # display this run's incompletely archived latest attempt as an accepted step.
            out["historical_verification"] = deepcopy(out["verification"])
            out["verification"] = dict(out["verification"] or {}, passed=False, evidence_persisted=False,
                reason="latest_verification_evidence_not_persisted", archive_error=result.get("archive_error"),
                at=s6.get("finished_at"))
        out["versions"] = [{key: version.get(key) for key in ("name", "created_at", "source", "mode", "changes", "steps",
                                                             "table_z_m", "joint_offsets_rad", "previous", "verification")}
                           for version in self.bundle.versions(limit=20)]
        if active:
            try:
                self._version_binding(active)
                out["matches_active_version"] = True
            except (StepRefused, OSError, ValueError, KeyError, TypeError) as error:
                out["matches_active_version"] = False
                out["binding_error"] = str(error)
        return out

    def staleness_hint(self):
        """Cheap check, throttled by ``staleness_check_period_s``: when both cameras see the board, the board pose
        from the wrist chain (latest telemetry joints) is projected into the fixed camera with the active
        calibration and compared with the fixed detection (RMS px vs ``staleness_max_board_shift_px``)."""
        now = time.monotonic()
        if self._stale_checked_at is not None and now - self._stale_checked_at < self.settings["staleness_check_period_s"]:
            return self._stale_hint
        self._stale_checked_at = now
        self._stale_hint = dict(self._check_staleness(), checked_at=utc_now(),
                                threshold_px=self.settings["staleness_max_board_shift_px"])
        return self._stale_hint

    def _check_staleness(self):
        import cv2
        from . import charuco as cal
        adapter = self.adapter
        document = self._active_document()
        if document is None or not {"wrist", "fixed"} <= set(document["cameras"]):
            return {"status": "unavailable", "reason": "no_active_two_camera_calibration"}
        if adapter.motion_phase not in ("idle", "holding") or adapter.last_sample is None:
            return {"status": "unavailable", "reason": "arm_moving_or_no_joint_sample"}
        frames = {name: adapter.cameras[name].latest() for name in ("wrist", "fixed") if name in adapter.cameras}
        if any(frame is None for frame in frames.values()) or len(frames) < 2:
            return {"status": "unavailable", "reason": "no_frames"}
        board = cal.make_board(document["board"])
        detections = {name: cal.detect_charuco(cv2.cvtColor(np.asarray(frame[0], np.uint8), cv2.COLOR_RGB2BGR), board,
                                               min_corners=self.settings["min_board_corners"]) for name, frame in frames.items()}
        if any(detection is None for detection in detections.values()):
            return {"status": "board_not_visible", "corners": {name: 0 if d is None else len(d["ids"]) for name, d in detections.items()}}
        wrist, fixed = document["cameras"]["wrist"], document["cameras"]["fixed"]
        T_wb, _ = cal.board_pose(detections["wrist"]["corners"], detections["wrist"]["ids"], wrist["K"], wrist["distortion"], board)
        T_link = np.asarray(adapter.kin.fk(adapter.model, adapter.last_sample["q_rad"], link=adapter.kin.WRIST_CAMERA_PARENT_LINK))
        T_base_board = T_link @ wrist["T_link_camera"] @ T_wb
        det = detections["fixed"]
        predicted = cal.project_board_corners(cal.invert_transform(fixed["T_base_camera"]) @ T_base_board, det["ids"],
                                              fixed["K"], fixed["distortion"], board)
        rms = float(np.sqrt(np.mean(np.sum((predicted - det["corners"]) ** 2, axis=1))))
        return {"status": "shifted" if rms > self.settings["staleness_max_board_shift_px"] else "ok", "rms_px": rms}

    def _write_overlays(self, run):
        """Operator previews: detected board corners on both cameras; touch targets projected into the fixed
        camera with the active calibration. Throttled to telemetry.frame_hz; never raises."""
        import cv2
        from . import charuco as cal
        now = time.monotonic()
        if self._overlay_at is not None and now - self._overlay_at < 1 / self.adapter.deployment["telemetry"]["frame_hz"]:
            return
        self._overlay_at = now
        board_spec = ((run or {}).get("board") or self.default_board())
        board = cal.make_board({key: board_spec[key] for key in BOARD_FIELDS})
        document = self._active_document()
        side = self.adapter.deployment["telemetry"]["preview_max_side_px"]
        for name, filename in OVERLAY_FILES.items():
            camera = self.adapter.cameras.get(name)
            latest = camera.latest() if camera is not None else None
            if latest is None:
                continue
            image = cv2.cvtColor(np.asarray(latest[0], np.uint8), cv2.COLOR_RGB2BGR)
            detection = cal.detect_charuco(image, board, min_corners=4)
            if detection is not None:
                for x, y in detection["corners"]:
                    cv2.circle(image, (int(round(x)), int(round(y))), 4, (0, 200, 0), 2)
            if name == "fixed" and document is not None and "fixed" in document["cameras"]:
                self._draw_touch_targets(image, document)
            scale = min(1.0, side / max(image.shape[:2]))
            if scale < 1.0:
                image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
            ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                self.adapter._write_atomic(self.adapter.telemetry_dir / filename, buffer.tobytes())

    def _draw_touch_targets(self, image, document):
        import cv2
        fixed = document["cameras"]["fixed"]
        table = (json.loads(Path(self.adapter.calibration_path).read_text()).get("joint_calibration") or {}).get("table_z_m")
        z = 0.0 if table is None else float(table)
        T_cam_base = np.linalg.inv(fixed["T_base_camera"])
        for colour, points in (((255, 120, 0), self.settings["touch_points_xy_m"]), ((200, 0, 200), self.settings["verify_points_xy_m"])):
            for index, (x, y) in enumerate(points):
                p = T_cam_base @ np.array([x, y, z, 1.0])
                if p[2] <= 0:
                    continue
                uv, _ = cv2.projectPoints(p[:3].reshape(1, 1, 3), np.zeros(3), np.zeros(3), fixed["K"], fixed["distortion"])
                u, v = (int(round(c)) for c in uv.reshape(2))
                cv2.circle(image, (u, v), 9, colour, 2)
                cv2.putText(image, str(index + 1), (u + 10, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
        sample = self.adapter.last_sample
        if sample is not None:
            tcp = np.asarray(self.adapter.kin.fk(self.adapter.model, sample["q_rad"]))[:4, 3]
            p = T_cam_base @ tcp
            if p[2] > 0:
                uv, _ = cv2.projectPoints(p[:3].reshape(1, 1, 3), np.zeros(3), np.zeros(3), fixed["K"], fixed["distortion"])
                u, v = (int(round(c)) for c in uv.reshape(2))
                cv2.drawMarker(image, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)

    # ---------------------------------------------------------- step start ---
    def _xy_binding(self):
        """Internal binding: camera calibration and joint mapping of an XY capture."""
        from ni_rua.core.contracts import digest
        return digest(dict(camera=self.adapter.camera_calibration, offsets=self._offsets()))

    def _xy_verification(self, params):
        """Capture without goals, or evaluate operator-selected frozen pixels.

        Units/defaults: validate_xy_request. Evidence is separate from the S6
        height verdict. This never fits calibration or sends a motor command.
        """
        from .xy_verification import validate_xy_request, evaluate_xy, XY_SETTING_KEYS
        p = validate_xy_request(params, defaults={key: self.settings[setting] for key, setting in XY_SETTING_KEYS.items()})
        adapter = self.adapter
        if self.run.get("closed"):
            raise StepRefused("calibration_run_closed")
        if not adapter.camera_calibration:
            raise StepRefused("xy_requires_camera_calibration")
        if p["action"] == "xy_capture":
            from PIL import Image
            self.running_step = "S6"
            adapter.active_action_id = "operator_calibration_S6_xy"
            try:
                motion = adapter.deployment["motion"]
                run = adapter._new_run(threading.Event(), motion["arrival_timeout_s"], "xy_capture")
                settled, sample, reason = adapter._settle(run, motion["arrival_timeout_s"])
                if not settled:
                    raise StepRefused("xy_capture_not_settled:" + str(reason))
                binding = self._xy_binding()
                document = deepcopy(adapter.camera_calibration)
                identifier = uuid.uuid4().hex
                cameras, images, alignment = {}, {}, {}
                for name in ("fixed", "wrist"):
                    rgb, stamp = adapter.cameras[name].read_after(sample["t"], adapter.deployment["camera"]["frame_timeout_s"])
                    spec = document["cameras"][name]
                    T = (adapter.kin.fk(adapter.model, sample["q_rad"], link=adapter.kin.WRIST_CAMERA_PARENT_LINK)
                         @ np.asarray(spec["T_link_camera"])) if name == "wrist" else spec["T_base_camera"]
                    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
                    images[name] = image
                    cameras[name] = dict(K=spec["K"], D=spec["distortion"], T=np.asarray(T).tolist(),
                                         width=image.width, height=image.height)
                    alignment[name] = stamp - sample["t"]
                after = adapter.read()
                interrupted = adapter._interrupt(run, "xy_capture")
                if interrupted:
                    raise StepRefused("xy_capture_interrupted:" + interrupted)
                dt = max(after["t"] - sample["t"], 1 / motion["command_rate_hz"])
                velocity = float(np.abs(np.subtract(after["q_rad"], sample["q_rad"])).max()) / dt
                if velocity >= motion["settle_velocity_rad_s"] or binding != self._xy_binding():
                    raise StepRefused("xy_capture_moved_or_calibration_changed")
                snapshot = _plain(dict(snapshot_id=identifier, at=utc_now(), cameras=cameras,
                    T_base_tcp=adapter.tcp(sample["q_rad"]).tolist(), q_rad=sample["q_rad"],
                    binding=binding, active_version=self.bundle.active_version(),
                    joint_to_frame_offset_s=alignment, capture_alignment="Asynchronous frames; exposure latency unknown. Arm settled before capture; endpoint drift checked, not continuous motion proof."))
                directory = self._run_dir() / "xy" / identifier
                directory.mkdir(parents=True)
                for name, image in images.items():
                    image.save(directory / f"{name}.jpg", quality=95)
                    adapter._write_atomic(adapter.telemetry_dir / f"xy-{identifier}-{name}.jpg", (directory / f"{name}.jpg").read_bytes())
                write_json(directory / "snapshot.json", snapshot)
                with self.lock:
                    self.run["xy_capture"] = {key: snapshot[key] for key in ("snapshot_id", "at", "joint_to_frame_offset_s", "capture_alignment")}
                    self._save()
                return dict(status="captured", snapshot=self.run["xy_capture"])
            finally:
                adapter.active_action_id = None
                self.running_step = None
        identifier = (self.run.get("xy_capture") or {}).get("snapshot_id")
        if p["snapshot_id"] != identifier:
            raise StepRefused("xy_snapshot_mismatch: capture again")
        snapshot = json.loads((self._run_dir() / "xy" / identifier / "snapshot.json").read_text())
        if snapshot["binding"] != self._xy_binding():
            raise StepRefused("xy_calibration_changed: capture again")
        result = evaluate_xy(snapshot, p)
        result.update(at=utc_now(), q_rad=snapshot["q_rad"], active_version=snapshot["active_version"])
        with self.lock:
            self.run.setdefault("xy_verifications", []).append(result)
            self._save()
        return dict(status="evaluated", result=result)

    def start_step(self, step, params=None, progress=None):
        """Run ``step`` (main thread for motion; S3 ``capture`` only starts the background sampler).

        ``params``: ``confirmed`` (operator readiness ids), ``board`` (S2: edited board spec), S3 ``action``
        (``capture``/``solve``/``stop``). Returns ``{"step", "status", "reason", "result", ...}``; a refused start
        returns ``status="rejected"`` with nothing moved.
        """
        params = dict(params or {})
        base = {"command": "calib_step_start", "step": step, "physics_advanced": False}
        if step not in wf.STEP_BY_ID:
            return dict(base, status="rejected", reason=f"calibration_step_unknown:{step}")
        if self.run is None:
            return dict(base, status="rejected", reason="calibration_run_not_started")
        if self.running_step is not None:
            return dict(base, status="rejected", reason=f"calibration_step_running:{self.running_step}")
        if self.adapter.active_action_id is not None:
            return dict(base, status="rejected", reason="action_active")
        if step != "S3" and self.capture is not None and self.capture.running:
            return dict(base, status="rejected", reason="S3_capture_running: stop the hand-held capture first")
        if step == "S6" and params.get("action") in ("xy_capture", "xy_evaluate"):
            try:
                return dict(base, **self._xy_verification(params))
            except (StepRefused, RuntimeError, ValueError, KeyError, OSError) as error:
                return dict(base, status="rejected", reason=str(error))
        try:
            # A stop remains available even if configuration changed after capture began.
            if not (step == "S3" and params.get("action") == "stop"):
                self._require_run_settings()
            if step == "S2" and params.get("board") is not None:
                board = self.validate_board(params["board"])
                with self.lock:
                    self.run["board"] = board
                    self.run["board_source"] = {"geometry": "operator_S2_request", "thickness": "operator_S2_request"}
                    self._save()
            if step == "S3" and params.get("action", "capture") in ("solve", "stop"):
                return dict(base, **self._s3(params["action"], progress))
            self._check_readiness(step, params.get("confirmed") or [])
            if step == "S5":
                report, source = self._touch_report()
                from ni_rua.core.contracts import digest
                self.run["contact_review"] = {"at": utc_now(), "source": source,
                    "report_digest": digest(report), "basis": report.get("joint_offsets_rad"),
                    "joint_mapping": report.get("joint_mapping"),
                    "kind": "operator_attestation", "check": "touches_physically_confirmed",
                    "acquisition_evidence": "unverified_obstruction" if all(
                        t.get("measurement_kind") == "obstruction_candidate" for t in report.get("touches", []))
                        else "legacy_unverified"}
                self._save()
        except StepRefused as error:
            return dict(base, status="rejected", reason=str(error))
        if step == "S3":
            return dict(base, **self._s3("capture", progress))
        runner = {"S0": self._s0, "S1": self._s1, "S2": self._s2, "S4": self._s4, "S5": self._s5, "S6": self._s6}[step]
        return dict(base, **self._execute(step, runner, progress))

    def _check_readiness(self, step, confirmed):
        if wf.STEP_BY_ID[step]["motion"] and self.adapter._gate():
            raise StepRefused(self.adapter._gate())
        checks = self.readiness(step, fresh=True)
        failed = [check for check, value in checks.items() if value["kind"] == "auto" and not value["ok"]]
        if failed:
            raise StepRefused("readiness_failed:" + ",".join(failed))
        missing = [check for check, value in checks.items() if value["kind"] == "operator" and check not in confirmed]
        if missing:
            raise StepRefused("readiness_unconfirmed:" + ",".join(missing))

    def _begin_step(self, step):
        with self.lock:
            statuses = {name: record["status"] for name, record in self.run["steps"].items()}
            for name, status in wf.mark_stale(statuses, step).items():
                if status == "stale" and self.run["steps"][name]["status"] != "stale":
                    self.run["steps"][name].update(status="stale", reason=f"{step}_rerun")
            if step in ("S1", "S2", "S3", "S4", "S5") and self.run.get("candidate"):
                self.run["candidate"]["stale"] = True
            if step == "S1":
                self.run.pop("contact_review", None)
            self._step(step).update(status="running", started_at=utc_now(), finished_at=None, reason=None, progress=[],
                                    result=None, effective_calibration_settings=deepcopy(self.settings))
            self._save()

    def _finish_step(self, step, status, reason, result):
        with self.lock:
            self._step(step).update(status=status, reason=reason, finished_at=utc_now(), result=_plain(result))
            self._save()
            write_json(self._run_dir() / "reports" / f"{step}.json", _plain(dict(self._step(step), step=step)))

    def _execute(self, step, runner, progress):
        self._begin_step(step)
        push = self._progress(step)

        def both(update):
            push(update)
            if progress is not None:
                progress(dict(update, step=step))
        self.running_step = step
        self.adapter.active_action_id = f"operator_calibration_{step}"
        start_step = getattr(self.adapter, "control_step", 0)
        is_motion_step = bool(wf.STEP_BY_ID[step].get("motion"))
        try:
            outcome = runner(both)
        except Exception as error:
            moved = getattr(self.adapter, "control_step", 0) != start_step
            stop = None
            if is_motion_step:
                try:
                    stop = self.adapter._stop()
                except Exception as stop_err:
                    stop = {"confirmed": False, "state": "unknown", "reason": f"{type(stop_err).__name__}: {stop_err}"}
            uncertain = moved or (stop is not None and stop.get("confirmed") is False)
            outcome = {
                "status": "unknown" if uncertain else "failed",
                "motion_status": ("unknown" if uncertain else "failed") if is_motion_step else None,
                "physics_advanced": moved,
                "reason": f"{type(error).__name__}: {error}",
                "measures": {},
                **({"stop_confirmation": stop} if stop is not None else {}),
            }
        finally:
            self.adapter.active_action_id = None
            self.running_step = None
        measures = outcome.get("measures") or {}
        verdict = wf.evaluate(step, measures, self.settings)
        motion = outcome.get("motion_status")  # completed / stopped / failed / unknown for motion steps
        status = "passed" if verdict["passed"] and outcome.get("status") in (None, "completed") else "failed"
        reason = outcome.get("reason")
        if status == "failed" and reason is None:
            reason = "; ".join(f"{row['label']} 未通过" for row in verdict["criteria"] if not row["passed"]) or "failed"
        result = dict(outcome, verdict=verdict,
                      effective_calibration_settings=deepcopy(self._step(step)["effective_calibration_settings"]))
        try:
            self._finish_step(step, status, reason, result)
        except Exception as error:
            # Physical work already returned a typed outcome. An archive failure must never
            # replace it with a generic no-motion error or erase an unconfirmed stop.
            status, reason = "failed", f"step_evidence_archive_failed: {type(error).__name__}: {error}"
            result.update(evidence_persisted=False, archive_error=reason)
            with self.lock:
                self._step(step).update(status=status, reason=reason, finished_at=utc_now(), result=_plain(result))
                if step == "S5" and self.run.get("candidate"):
                    self.run["candidate"]["stale"] = True
                try:
                    self._save()  # e.g. report file failed but run.json is still writable
                except Exception:
                    pass  # the returned receipt still carries the physical facts and archive error
        receipt = {"status": status, "reason": reason, "result": _plain(result),
                   "physics_advanced": bool(outcome.get("physics_advanced"))}
        if motion == "unknown" or (outcome.get("stop_confirmation") or {}).get("confirmed") is False:
            receipt["status"] = "unknown"
            receipt["stop_confirmation"] = outcome.get("stop_confirmation") or {"confirmed": False}
        if outcome.get("status") == "stopped" or motion == "stopped":
            receipt["stopped"] = True
        return receipt

    # ---------------------------------------------------------------- S0 ---
    def _s0(self, progress):
        adapter, limits = self.adapter, self.adapter.deployment["preflight"]
        age = self._sample_age()
        measures = {"telemetry_age_s": age}
        try:
            preflight = json.loads((adapter.run_dir / "preflight.json").read_text())
            measures["preflight_ok"] = preflight.get("ok") is True
        except (OSError, ValueError):
            measures["preflight_ok"], preflight = False, None
        detail = {"preflight": None if preflight is None else {"ok": preflight.get("ok"), "checked_at": preflight.get("checked_at"),
                                                               "error": preflight.get("error"), "detail": preflight.get("detail")}}
        progress({"phase": "bus"})
        try:
            adapter.read()
            with adapter.bus_lock:
                motors = adapter.driver.motor_status()
                torque = adapter.driver.torque_enabled()
            adapter.torque_known = torque
            voltages = {n: m.get("voltage_v") for n, m in motors.items()}
            temperatures = {n: m.get("temperature_c") for n, m in motors.items()}
            measures["power_ok"] = all(v is not None and limits["min_voltage_v"] <= v <= limits["max_voltage_v"] for v in voltages.values()) \
                and all(t is not None and t <= limits["max_temperature_c"] for t in temperatures.values()) and bool(motors)
            measures["torque_ok"] = torque is True
            detail.update(voltage_v=voltages, temperature_c=temperatures, torque_enabled=torque,
                          voltage_range_v=[limits["min_voltage_v"], limits["max_voltage_v"]],
                          max_temperature_c=limits["max_temperature_c"])
        except Exception as error:
            measures.update(power_ok=False, torque_ok=False)
            detail["bus_error"] = f"{type(error).__name__}: {error}"
        progress({"phase": "cameras"})
        cameras = {}
        for name, camera in sorted(adapter.cameras.items()):
            config = adapter.deployment["cameras"].get(name, {})
            try:
                image, _ = camera.read_after(adapter.clock.now(), adapter.deployment["camera"]["frame_timeout_s"])
                shape = np.asarray(image).shape
                ok = shape[:2] == (config.get("height"), config.get("width"))
                cameras[name] = {"ok": ok, "size_px": [int(shape[1]), int(shape[0])],
                                 "configured_px": [config.get("width"), config.get("height")]}
            except Exception as error:
                cameras[name] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
            # Which physical device this camera opened (mode, OpenCV target, name, unique ids, transport).
            device = binding_summary(((getattr(adapter, "camera_binding", None) or {}).get("cameras") or {}).get(name))
            if device is not None:
                cameras[name]["device"] = device
        measures["cameras_ok"] = bool(cameras) and all(item["ok"] for item in cameras.values()) and \
            set(cameras) >= set(adapter.deployment["cameras"])
        detail["cameras"] = cameras
        missing_deps = []
        for mod in ("cv2.aruco", "scipy.optimize"):
            try:
                __import__(mod)
            except ImportError:
                missing_deps.append(mod)
        detail["solver_dependencies_ok"] = not missing_deps
        if missing_deps:
            return {"status": "failed", "reason": "missing_worker_dependency:" + ",".join(missing_deps),
                    "measures": measures, "detail": detail}
        return {"measures": measures, "detail": detail}

    # ----------------------------------------------------------- S1 / S6 ---
    def _touch(self, points, progress, label):
        """Acquire touch evidence and bounded operator video for one S1/S6 attempt.

        ``points`` are base-frame XY metres; ``label`` identifies S1 or S6; ``progress`` receives
        physical motion updates. Recording limits come from deployment FIELDS. The relative video
        directory is rooted at this workflow run. Camera/recorder failures never replace a physical
        receipt, start inference, promote contact confidence or bypass the motion stop mechanism.
        """
        import ni_rua.robot.calibration.workflow.touch_off as touch_off_mod
        settings = {"fps": self.settings["touch_video_fps"], "max_frames": self.settings["touch_video_max_frames"],
                    "max_duration_s": self.settings["touch_video_max_duration_s"]}
        video = {"status": "disabled", "directory": None, "settings": settings, "contact_evidence": "unknown",
                 "manifest_sha256": None}
        recorder, directory = None, None
        errors = []
        if settings["fps"] > 0:
            if not all(self.adapter.cameras.get(name) is not None for name in ("wrist", "fixed")):
                video.update(status="unavailable", termination_reason="both_cameras_required")
            else:
                directory = self._run_dir() / "touch_video" / f"{label}-{uuid.uuid4().hex}"
                video["directory"] = str(directory.relative_to(self._run_dir()))
                try:
                    from ni_rua.robot.calibration.workflow.video import CalibrationVideoRecorder
                    directory.parent.mkdir(parents=True, exist_ok=True)
                    recorder = CalibrationVideoRecorder(self.adapter, directory, **settings)
                    recorder.start()
                except Exception as error:
                    errors.append(f"start: {type(error).__name__}: {error}")
                    video.update(status="failed", termination_reason="recorder_start_failed")
        report = None
        try:
            report = touch_off_mod.touch_points(self.adapter, points, self.settings, progress=progress, label=label)
        finally:
            if recorder is not None:
                try:
                    summary = recorder.stop()
                    video.update(summary)
                    video["directory"] = str(directory.relative_to(self._run_dir()))
                    video["contact_evidence"] = "unknown"  # video availability is not contact validation
                    # The operator summary uses run-relative paths; the raw manifest remains local.
                    if video.get("manifest_path"):
                        path = Path(video["manifest_path"])
                        if path.is_absolute():
                            video["manifest_path"] = str(path.relative_to(self._run_dir()))
                except Exception as error:
                    errors.append(f"stop: {type(error).__name__}: {error}")
                    video.update(status="failed", termination_reason="recorder_stop_failed")
            if not errors and video.get("manifest_path") and video.get("sealed") is True:
                try:
                    import hashlib
                    manifest = self._run_dir() / video["manifest_path"]
                    video["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
                except Exception as error:
                    errors.append(f"manifest: {type(error).__name__}: {error}")
                    video.update(status="failed", termination_reason="manifest_binding_failed")
            if errors:
                video.update(status="failed", errors=errors)
            if directory is not None:
                try:
                    if directory.is_dir():
                        write_json(directory / "video_summary.json", _plain(video))
                except Exception as error:
                    video.setdefault("errors", []).append(f"summary: {type(error).__name__}: {error}")
                    video["status"] = "failed"
            if report is not None:
                report["video"] = video
        report["joint_offsets_rad"] = self._offsets()  # basis of q_measured_rad (deployment joint_mapping)
        report["evidence_persisted"] = True
        try:
            write_json(self._run_dir() / f"{label}_touches.json", _plain(report))
        except Exception as error:
            report.update(evidence_persisted=False, archive_error=f"{type(error).__name__}: {error}")
        return report

    def _s1(self, progress):
        report = self._touch(self.settings["touch_points_xy_m"], progress, "S1")
        measures = {"contacts": report["contacts"], "planned_points": report["planned_points"]}
        archived = report.get("evidence_persisted") is not False
        return {"status": report["status"] if archived else "failed", "motion_status": report["status"],
                "reason": report.get("reason") if archived else "touch_evidence_archive_failed: " + report["archive_error"],
                "evidence_persisted": archived, "archive_error": report.get("archive_error"),
                "measures": measures, "physics_advanced": report["physics_advanced"],
                "stop_confirmation": report.get("stop_confirmation"), "table_z_nominal_m": report["table_z_nominal_m"],
                "evidence_quality": "unverified_obstruction", "physical_contact_verified": False,
                "video": report.get("video"),
                "touches": [{key: touch.get(key) for key in ("index", "xy", "status", "contact_z_m", "trigger", "lag_rad", "confirmation")}
                            for touch in report["touches"]]}

    def _s6(self, progress):
        binding = self._verification_binding()  # recheck after readiness, before any command
        table = binding["table_z_m"]
        report = self._touch(self.settings["verify_points_xy_m"], progress, "S6")
        if report.get("evidence_persisted") is False:
            return {"status": "failed", "motion_status": report["status"],
                    "reason": "touch_evidence_archive_failed: " + report["archive_error"],
                    "physics_advanced": report["physics_advanced"], "stop_confirmation": report.get("stop_confirmation"),
                    "measures": {}, "verification_recorded": False, "evidence_persisted": False,
                    "archive_error": report["archive_error"], "captured_report": report, "video": report.get("video")}
        try:
            if self._verification_binding() != binding:
                raise StepRefused("verification_calibration_mismatch:changed_during_capture")
        except (StepRefused, OSError, ValueError, KeyError, TypeError) as error:
            return {"status": "failed", "motion_status": report["status"], "reason": str(error),
                    "physics_advanced": report["physics_advanced"], "stop_confirmation": report.get("stop_confirmation"),
                    "measures": {}, "verification_recorded": False, "captured_report": report, "video": report.get("video")}
        errors = [touch["contact_z_m"] - table for touch in report["touches"] if touch.get("status") == "contact"]
        measures = {"contacts": report["contacts"], "planned_points": report["planned_points"],
                    "max_verify_error_m": max((abs(e) for e in errors), default=None)}
        with self.lock:
            try:
                if self._verification_binding() != binding:
                    raise StepRefused("verification_calibration_mismatch:changed_before_verdict")
            except (StepRefused, OSError, ValueError, KeyError, TypeError) as error:
                return {"status": "failed", "motion_status": report["status"], "reason": str(error),
                        "physics_advanced": report["physics_advanced"], "stop_confirmation": report.get("stop_confirmation"),
                        "measures": measures, "verification_recorded": False, "captured_report": report, "video": report.get("video")}
            if self.bundle is not None and self.run.get("applied_version"):
                verification = {"at": utc_now(), "errors_m": errors, "max_error_m": measures["max_verify_error_m"],
                                "scope": "holdout_model_consistency", "independent_physical_accuracy": None,
                                "version": binding["version"], "joint_offsets_rad": self._offsets(),
                                "passed": wf.evaluate("S6", measures, self.settings)["passed"] and report["status"] == "completed"}
                try:
                    manifest_path = self.bundle.version_file(binding["version"], "manifest.json")
                    manifest = json.loads(manifest_path.read_text())
                    manifest["verification"] = verification
                    write_json(manifest_path, manifest)
                except Exception as error:
                    reason = f"verification_evidence_archive_failed: {type(error).__name__}: {error}"
                    return {"status": "failed", "motion_status": report["status"], "reason": reason,
                            "physics_advanced": report["physics_advanced"], "stop_confirmation": report.get("stop_confirmation"),
                            "measures": measures, "verification_recorded": False, "evidence_persisted": False,
                            "archive_error": reason, "captured_report": report, "video": report.get("video")}
        return {"status": report["status"], "motion_status": report["status"], "reason": report.get("reason"),
                "measures": measures, "physics_advanced": report["physics_advanced"],
                "stop_confirmation": report.get("stop_confirmation"), "table_z_m": table, "errors_m": errors,
                "verification_scope": "holdout_model_consistency", "independent_physical_accuracy": None,
                "video": report.get("video"),
                "touches": [{key: touch.get(key) for key in ("index", "xy", "status", "contact_z_m", "trigger", "lag_rad", "confirmation")}
                            for touch in report["touches"]]}

    # ---------------------------------------------------------------- S2 ---
    def _board_spec(self):
        return {key: self.run["board"][key] for key in BOARD_FIELDS}

    def _fk_link(self, delta=None):
        adapter = self.adapter
        delta = np.zeros(5) if delta is None else np.asarray(delta, float)
        link = adapter.kin.WRIST_CAMERA_PARENT_LINK
        return lambda q: np.asarray(adapter.kin.fk(adapter.model, list(np.asarray(q, float) + delta), link=link))

    def _s2(self, progress):
        from .capture import capture_session, report_residuals
        from .hand_eye_solve import solve_session
        session = self._run_dir() / "S2_session"
        if session.exists():
            session = self._run_dir() / f"S2_session-{uuid.uuid4().hex[:6]}"
        captured = capture_session(self.adapter, session, settings=self.settings, progress=progress, board_spec=self._board_spec())
        out = {"status": captured["status"], "motion_status": captured["status"], "reason": captured.get("reason"),
               "physics_advanced": captured.get("physics_advanced"), "stop_confirmation": captured.get("stop_confirmation"),
               "session_dir": str(session), "measures": {"kept": captured["kept"]},
               "capture": {key: captured.get(key) for key in ("kept", "pose_count", "park")}}
        if captured["status"] != "completed" or captured["kept"] < self.settings["min_samples"]:
            return out
        progress({"phase": "solving", "kept_count": captured["kept"]})
        thresholds = {key: self.settings[key] for key in ("max_intrinsics_rms_px", "max_hand_eye_translation_m", "max_hand_eye_rotation_rad")}
        try:
            report = solve_session(session, output=session / "camera_calibration.json", thresholds=thresholds, force=True,
                                   overwrite=True, max_pnp_reprojection_px=self.settings["max_pnp_reprojection_px"],
                                   hand_eye_score_lever_m=self.settings["hand_eye_score_lever_m"],
                                   fixed_intrinsics_from_wrist=True, fk_link=self._fk_link())
        except Exception as error:
            return dict(out, status="failed", reason=f"solve_failed: {type(error).__name__}: {error}")
        residuals = report_residuals(report)
        wrist = report["cameras"]["wrist"]
        out["measures"].update(wrist_intrinsics_rms_px=residuals["wrist_intrinsics_rms_px"],
                               hand_eye_translation_rms_m=residuals["hand_eye_translation_rms_m"],
                               hand_eye_rotation_rms_rad=residuals["hand_eye_rotation_rms_rad"])
        from . import charuco as cal
        document = cal.read_calibration_file(session / "camera_calibration.json")
        out["wrist"] = {"K": wrist["K"], "distortion": wrist["distortion"], "image_size": wrist["image_size"],
                        "T_link_camera": document["cameras"]["wrist"]["T_link_camera"],
                        "intrinsics_rms_px": residuals["wrist_intrinsics_rms_px"]}
        out["T_base_board"] = report.get("T_base_board")
        out["hand_eye_samples"] = residuals["hand_eye_samples"]
        return out

    # ---------------------------------------------------------------- S3 ---
    @staticmethod
    def _handheld_writer_stopped(capture, snapshot):
        """True only after a handheld writer's explicit stop confirmation and actual thread exit.

        ``capture`` is the current HandheldCapture; ``snapshot`` is its stop receipt. This confirms
        image-writer shutdown, not robot stopping, capture quality, or sufficient calibration views.
        """
        thread = getattr(capture, "thread", None)
        return snapshot.get("stop_confirmed") is True and not capture.running \
            and thread is not None and not thread.is_alive()

    def _s3_writer_pending(self, snapshot):
        """Fail this S3 attempt without claiming an uncertain image writer moved the robot."""
        reason = "handheld_writer_not_stopped: " + str(snapshot.get("reason") or "stop not confirmed")
        result = {"capture": snapshot, "capture_stop_confirmed": False, "physics_advanced": False,
                  "measures": {}, "reason": reason}
        self._finish_step("S3", "failed", reason, result)
        return dict(result, status="failed")

    def _s3(self, action, progress):
        from .capture import HandheldCapture, handheld_intrinsics
        if action == "capture":
            if self.capture is not None:
                snapshot = self.capture.stop("restarted")
                if not self._handheld_writer_stopped(self.capture, snapshot):
                    return self._s3_writer_pending(snapshot)
            if self.adapter.handheld is not None and self.adapter.handheld is not self.capture:
                snapshot = self.adapter.handheld.stop("stopped_for_workflow")
                if not self._handheld_writer_stopped(self.adapter.handheld, snapshot):
                    return self._s3_writer_pending(snapshot)
            self._begin_step("S3")
            directory = self._run_dir() / f"S3_handheld-{uuid.uuid4().hex[:6]}"
            self.capture = HandheldCapture(self.adapter, directory, board_spec=self._board_spec())
            self.adapter.handheld = self.capture  # same telemetry preview/status files as the console capture
            snapshot = self.capture.start()
            return {"status": "running", "reason": None, "capture": snapshot}
        if self.capture is None or self._step("S3")["status"] not in ("running", "failed", "passed", "stale"):
            raise StepRefused("S3_capture_not_started")
        if action == "stop":
            snapshot = self.capture.stop()
            if not self._handheld_writer_stopped(self.capture, snapshot):
                return self._s3_writer_pending(snapshot)
            return {"status": "running", "reason": None, "capture": snapshot, "capture_stop_confirmed": True}
        # solve
        snapshot = self.capture.stop("stopped_for_solve")
        if not self._handheld_writer_stopped(self.capture, snapshot):
            return self._s3_writer_pending(snapshot)
        self.adapter.active_action_id = "operator_calibration_S3"
        self.running_step = "S3"
        try:
            K, dist, rms, used = handheld_intrinsics(self.capture, self.settings["handheld_min_views"])
            outcome = {"measures": {"views": snapshot["kept"], "fixed_intrinsics_rms_px": rms},
                       "fixed": {"K": K, "distortion": dist, "image_size": snapshot["image_size"], "intrinsics_rms_px": rms,
                                 "used_views": used}, "capture": snapshot}
        except Exception as error:
            outcome = {"status": "failed", "reason": f"{type(error).__name__}: {error}",
                       "measures": {"views": snapshot["kept"]}, "capture": snapshot}
        finally:
            self.adapter.active_action_id = None
            self.running_step = None
        verdict = wf.evaluate("S3", outcome["measures"], self.settings)
        status = "passed" if verdict["passed"] and outcome.get("status") is None else "failed"
        reason = outcome.get("reason") or (None if status == "passed" else "; ".join(
            f"{row['label']} 未通过" for row in verdict["criteria"] if not row["passed"]))
        result = dict(outcome, verdict=verdict)
        self._finish_step("S3", status, reason, result)
        return {"status": status, "reason": reason, "result": _plain(result)}

    # ---------------------------------------------------------------- S4 ---
    def _wrist(self):
        """Wrist intrinsics + hand-eye: this run's S2 when usable, else the active file."""
        if self._usable("S2"):
            wrist = self._step("S2")["result"]["wrist"]
            return {key: np.asarray(wrist[key], float) if key in ("K", "distortion", "T_link_camera") else wrist[key]
                    for key in wrist}, "S2"
        self._source_evidence("S2", "active")
        document = self._active_document()
        if document is None or "wrist" not in document["cameras"]:
            raise StepRefused("wrist_calibration_unavailable")
        return dict(document["cameras"]["wrist"]), "active"

    def _fixed_intrinsics(self):
        """Fixed K/distortion: this run's S3 when usable, else the active file."""
        if self._usable("S3"):
            fixed = self._step("S3")["result"]["fixed"]
            return dict(fixed, K=np.asarray(fixed["K"], float), distortion=np.asarray(fixed["distortion"], float),
                        intrinsics_source="handheld"), "S3"
        self._source_evidence("S3", "active")
        document = self._active_document()
        if document is None or "fixed" not in document["cameras"]:
            raise StepRefused("fixed_intrinsics_unavailable: run S3 first")
        return dict(document["cameras"]["fixed"]), "active"

    def _s4(self, progress):
        from .capture import (calibration_poses, capture_session,
                                                                     fixed_extrinsic_from_session)
        from . import charuco as cal
        wrist, wrist_source = self._wrist()
        fixed, fixed_source = self._fixed_intrinsics()
        all_poses = calibration_poses(self.adapter.deployment)
        count = self.settings["extrinsic_pose_count"]
        if count <= 1 or len(all_poses) <= count:
            poses = all_poses[:count]
        else:
            poses = [all_poses[0]] + [all_poses[int(i)] for i in np.linspace(1, len(all_poses) - 1, count - 1, dtype=int)]
        session = self._run_dir() / f"S4_session-{uuid.uuid4().hex[:6]}"
        captured = capture_session(self.adapter, session, settings=self.settings, poses=poses, progress=progress,
                                   board_spec=self._board_spec())
        out = {"status": captured["status"], "motion_status": captured["status"], "reason": captured.get("reason"),
               "physics_advanced": captured.get("physics_advanced"), "stop_confirmation": captured.get("stop_confirmation"),
               "session_dir": str(session), "wrist_source": wrist_source, "fixed_intrinsics_source": fixed_source,
               "calibration_sources": {"wrist": self._source_evidence("S2", wrist_source),
                                       "fixed_intrinsics": self._source_evidence("S3", fixed_source)},
               "measures": {}, "capture": {key: captured.get(key) for key in ("kept", "pose_count", "park")}}
        if captured["status"] != "completed":
            return out
        progress({"phase": "solving", "kept_count": captured["kept"]})
        try:
            solved = fixed_extrinsic_from_session(session, wrist, fixed["K"], fixed["distortion"],
                                                  cal.make_board(self._board_spec()), self._fk_link())
        except LookupError as error:
            return dict(out, status="failed", reason=str(error), measures={"fixed_samples": 0})
        out["measures"] = {"fixed_samples": solved["fixed_samples"],
                           "fixed_chain_reprojection_px": solved["chain_reprojection_px"]}
        out.update(T_base_camera=solved["T_base_camera"], T_base_board=solved["T_base_board"],
                   board_spread=solved["board_spread"], wrist_samples=solved["wrist_samples"])
        return out

    # ---------------------------------------------------------------- S5 ---
    def _candidate_source_binding(self):
        """Snapshot the current mapping and calibration used to solve/review a candidate.

        Mapping offsets are radians, signs dimensionless. File digest and version are internal
        identity checks, never model-visible calibration information. No default source is inferred
        for old candidates; they require a new S5 solve before apply.
        """
        import hashlib
        from ni_rua.robot.backends.so101.deployment import load_deployment
        self._require_run_settings()
        path = Path(self.adapter.calibration_path) if self.adapter.calibration_path else None
        deployment_path = getattr(self.adapter, "deployment_path", None)
        disk_deployment = load_deployment(deployment_path) if deployment_path else None
        disk = disk_deployment["joint_mapping"] if disk_deployment else None
        return {"active_version": self.bundle.active_version() if self.bundle else None,
                "calibration_settings": deepcopy(self.settings),
                "persisted_calibration_settings": None if disk_deployment is None else disk_deployment["calibration"],
                "joint_mapping": {n: dict(self.adapter.deployment["joint_mapping"][n]) for n in ARM_JOINTS},
                "persisted_joint_mapping": None if disk is None else {n: dict(disk[n]) for n in ARM_JOINTS},
                "camera_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path and path.is_file() else None,
                "loaded_camera": _plain(self.adapter.camera_calibration), "board": deepcopy(self.run["board"]),
                "board_source": deepcopy(self.run.get("board_source"))}

    @staticmethod
    def _candidate_artifact_binding(candidate):
        """Internal digest of reviewed camera/touch artifacts and installed numeric targets.

        ``candidate`` contains file path, joint offsets (rad) and table height (m). A modified
        candidate must be solved/reviewed again; this identity never enters actor context.
        """
        from ni_rua.core.contracts import digest
        path = Path(candidate["file"])
        return digest({"camera": json.loads(path.read_text()),
                       "touches": json.loads((path.parent / "touches.json").read_text()),
                       "joint_offsets_rad": candidate["joint_offsets_rad"], "table_z_m": candidate["table_z_m"]})

    def _touch_report(self):
        """Return this run's completed S1 report, or the active archive report, and its source label."""
        if self._usable("S1"):
            if (self._step("S1").get("result") or {}).get("evidence_persisted") is False:
                raise StepRefused("touches_unavailable: current S1 evidence was not persisted")
            report = json.loads((self._run_dir() / "S1_touches.json").read_text())
            source = "S1"
        else:
            if self.run is not None and self._step("S1")["status"] != "not_started":
                raise StepRefused("touches_unavailable: current S1 did not pass; old contacts cannot replace it")
            self._source_evidence("S1", "active")
            active = self.bundle.active_version() if self.bundle else None
            report = self.bundle.read_version_json(active, "touches.json") if active else None
            source = f"version:{active}"
            if not report:
                raise StepRefused("touches_unavailable: run S1 first")
        if report.get("evidence_persisted") is False:
            raise StepRefused("touches_unavailable: contact evidence archive incomplete")
        return report, source

    def _touch_inputs(self):
        """Reviewed touches in the current mapping basis: (q rad, link, point m, jaw rad).

        Operator attestation is required after acquisition, including for legacy reports. It is
        evidence review, not an independently calibrated sensor or proof of the CAD contact model.
        Complete recorded sign/offset mapping is mandatory; old offset-only records remain readable
        but require new S1 acquisition before fitting. Re-expression preserves raw encoder radians:
        q_new = (q_old - offset_old) * sign_new / sign_old + offset_new.
        """
        from ni_rua.core.contracts import digest
        report, source = self._touch_report()
        review = self.run.get("contact_review") or {}
        if review.get("source") != source or review.get("report_digest") != digest(report):
            raise StepRefused("touches_unreviewed: confirm acquired contact evidence before S5")
        now = self.adapter.deployment["joint_mapping"]
        then = report.get("joint_mapping")
        if not isinstance(then, dict) or any(
                not isinstance(then.get(name), dict) or type(then[name].get("sign")) not in (int, float)
                or then[name]["sign"] not in (-1, 1)
                or type(then[name].get("offset_rad")) not in (int, float)
                or not math.isfinite(then[name]["offset_rad"]) for name in ARM_JOINTS):
            raise StepRefused("touch_basis_unknown: acquire S1 with complete joint mapping")
        offsets = report.get("joint_offsets_rad")
        if offsets is not None and offsets != {name: then[name]["offset_rad"] for name in ARM_JOINTS}:
            raise StepRefused("touch_basis_mismatch: offsets disagree with recorded mapping")
        touches = []
        for touch in report["touches"]:
            if touch.get("status") != "contact":
                continue
            # driver._to_rad: q = sign * radians(raw encoder degrees) + offset_rad.
            # Invert the recorded mapping, then apply the current mapping to the same raw reading.
            q = [(float(touch["q_measured_rad"][i]) - then[name]["offset_rad"])
                 * now[name]["sign"] / then[name]["sign"] + now[name]["offset_rad"]
                 for i, name in enumerate(ARM_JOINTS)]
            touches.append((q, touch["contact_link"], touch["contact_point_in_link_m"], float(touch.get("gripper_rad", 0.0))))
        if not touches:
            raise StepRefused("touches_unavailable: no contacts")
        return touches, report, source

    def _session_samples(self, session_dir, board, wrist, board_index):
        """Wrist detections of a capture session -> (Sample list, PnP poses) in the current mapping basis."""
        from . import charuco as cal
        from .hand_eye_solve import _detect, load_session
        from .joint_calibration import Sample
        _, samples = load_session(session_dir)
        mapping = cal.normalize_joint_mapping({name: dict(self.adapter.deployment["joint_mapping"][name]) for name in ARM_JOINTS})
        out, poses = [], []
        for sample in samples:
            if not sample.get("settled", True) or "wrist" not in (sample.get("images") or {}):
                continue
            detection, _ = _detect(Path(session_dir) / sample["images"]["wrist"], board)
            if detection is None:
                continue
            T_cb, rms = cal.board_pose(detection["corners"], detection["ids"], wrist["K"], wrist["distortion"], board)
            if rms > self.settings["max_pnp_reprojection_px"]:
                continue
            q = cal.lerobot_deg_to_urdf_rad(sample["joints_deg"], mapping)
            out.append(Sample(q, detection["corners"], detection["ids"], board=board_index))
            poses.append(T_cb)
        return out, poses

    def _s5(self, progress):
        from .capture import fixed_extrinsic_from_session
        from . import charuco as cal
        from ...embodiments.so101.contact import lowest_gripper_point
        from .joint_calibration import DEFAULT_ESTIMATED, solve_joint_calibration
        source_binding = self._candidate_source_binding()
        adapter, settings, kin, model = self.adapter, self.settings, self.adapter.kin, self.adapter.model
        wrist, wrist_source = self._wrist()
        fixed, fixed_source = self._fixed_intrinsics()
        touches_raw, touch_report, touch_source = self._touch_inputs()
        board = cal.make_board(self._board_spec())
        thickness = self.run["board"]["thickness_m"]
        progress({"phase": "loading"})
        placements, samples, pnp = [], [], []
        sessions = []
        if self._usable("S2"):
            sessions.append(("S2", self._step("S2")["result"]["session_dir"]))
        elif self.bundle is not None and self.bundle.active_version():
            s2_report = self.bundle.read_version_json(self.bundle.active_version(), "reports/S2.json")
            s2_dir = ((s2_report or {}).get("result") or {}).get("session_dir")
            if s2_dir and Path(s2_dir).is_dir():
                sessions.append(("S2", s2_dir))
        sessions.append(("S4", self._step("S4")["result"]["session_dir"]))
        fk_link = self._fk_link()
        for name, directory in sessions:
            rows, poses = self._session_samples(directory, board, wrist, len(placements))
            if not rows:
                continue
            chain = [fk_link(row.q_rad) @ np.asarray(wrist["T_link_camera"], float) @ T for row, T in zip(rows, poses)]
            placements.append({"session": name, "session_dir": directory, "T0": cal.mean_transform(chain), "samples": len(rows)})
            samples += rows
            pnp += [(row, T) for row, T in zip(rows, poses)]
        if not samples:
            return {"status": "failed", "reason": "no_wrist_board_views_for_joint_solve", "measures": {}}

        def as_gripper_link(q, link, point, jaw):
            if link == kin.WRIST_CAMERA_PARENT_LINK:
                return list(point)
            poses = kin.fk_all(model, q, gripper_rad=jaw)
            return (np.linalg.inv(poses[kin.WRIST_CAMERA_PARENT_LINK]) @ poses[link] @ np.append(point, 1.0))[:3].tolist()
        touches = [(q, kin.WRIST_CAMERA_PARENT_LINK, as_gripper_link(q, link, point, jaw), jaw)
                   for q, link, point, jaw in touches_raw]
        fk_all = lambda q: kin.fk_all(model, q)  # noqa: E731
        lowest_fn = lambda q, jaw: lowest_gripper_point(model, q, jaw)  # noqa: E731
        nominal = [float(lowest_fn(q, jaw)["z_m"]) for q, _link, _point, jaw in touches]
        partial_fixed_only = not self._usable("S2") and not self._usable("S1")
        active_doc = self._active_document()
        active_table = ((active_doc or {}).get("joint_calibration") or {}).get("table_z_m")
        if partial_fixed_only:
            estimated_joints = ()
            freeze_gripper_camera = True
            table_z_init = float(active_table) if active_table is not None else float(np.mean(nominal))
            table_prior_sigma = None
        else:
            estimated_joints = DEFAULT_ESTIMATED
            freeze_gripper_camera = False
            table_z_init = float(np.mean(nominal))
            table_prior_sigma = settings["table_prior_sigma_m"]
        progress({"phase": "solving", "samples": len(samples), "touches": len(touches), "placements": len(placements)})
        result = solve_joint_calibration(samples, board, wrist["K"], wrist["distortion"], fk_link,
                                         T_gripper_camera0=np.asarray(wrist["T_link_camera"], float),
                                         T_base_board0=[p["T0"] for p in placements], table_z_m=table_z_init,
                                         board_thickness_m=thickness, estimated_joints=estimated_joints,
                                         offset_prior_rad=settings["joint_offset_prior_rad"],
                                         plane_sigma_m=settings["joint_plane_sigma_m"], pixel_sigma=settings["joint_pixel_sigma_px"],
                                         touches=touches, fk_all=fk_all, touch_sigma_m=settings["joint_touch_sigma_m"],
                                         table_prior_sigma_m=table_prior_sigma,
                                         initial_offsets_rad=self._offsets(),
                                         lowest_point_fn=lowest_fn,
                                         freeze_gripper_camera=freeze_gripper_camera,
                                         max_nfev=settings["joint_solver_max_nfev"],
                                         pixel_robust_scale=settings["joint_pixel_robust_scale"],
                                         ftol=settings["joint_solver_ftol"], xtol=settings["joint_solver_xtol"],
                                         gtol=settings["joint_solver_gtol"], x_scale=settings["joint_solver_x_scale"])
        delta = np.array([result.joint_offsets_rad[name] for name in ARM_JOINTS])
        current = self._offsets()
        new_offsets = {name: current[name] + float(delta[i]) for i, name in enumerate(ARM_JOINTS)}
        X = result.T_gripper_camera
        progress({"phase": "cross_check"})
        # Hand-eye spread of the new chain on the first placement (S2 when present).
        first = [(row, T) for row, T in pnp if row.board == 0]
        fk_new = self._fk_link(delta)
        spread = cal.hand_eye_consistency([fk_new(row.q_rad) for row, _ in first], X, [T for _, T in first],
                                         score_lever_m=settings["hand_eye_score_lever_m"])
        s4_index = next(i for i, p in enumerate(placements) if p["session"] == "S4") if any(
            p["session"] == "S4" for p in placements) else None
        fixed_solution, fixed_reason = None, None
        try:
            wrist_new = dict(wrist, T_link_camera=X)
            fixed_solution = fixed_extrinsic_from_session(
                self._step("S4")["result"]["session_dir"], wrist_new, fixed["K"], fixed["distortion"], board, fk_new,
                mapping=cal.normalize_joint_mapping({name: dict(self.adapter.deployment["joint_mapping"][name]) for name in ARM_JOINTS}),
                T_base_board=None if s4_index is None else result.T_base_boards[s4_index])
        except LookupError as error:
            fixed_reason = str(error)
        outline = cal.board_outline_points(board)
        board_errors = [abs(float(cz) - (result.table_z_m + thickness))
                        for B in result.T_base_boards
                        for cz in ((B[:3, :3] @ outline.T).T[:, 2] + B[2, 3])]
        measures = {"joint_reprojection_px": result.reprojection_rms_px,
                    "board_height_error_m": max(board_errors) if board_errors else None,
                    "max_touch_residual_m": max((abs(v) for v in result.touch_residuals_m), default=None),
                    "fixed_chain_reprojection_px": None if fixed_solution is None else fixed_solution["chain_reprojection_px"],
                    "max_joint_offset_rad": max(abs(v) for v in new_offsets.values())}
        image_sizes = {name: [config["width"], config["height"]] for name, config in adapter.deployment["cameras"].items()}
        cameras = {"wrist": {"K": wrist["K"], "distortion": wrist["distortion"], "image_size": image_sizes["wrist"],
                             "T_link_camera": X, "intrinsics_rms_px": wrist.get("intrinsics_rms_px", 0.0),
                             "hand_eye_residual": {"translation_m": spread["translation_rms_m"], "rotation_rad": spread["rotation_rms_rad"]}}}
        if fixed_solution is not None:
            cameras["fixed"] = {"K": fixed["K"], "distortion": fixed["distortion"], "image_size": image_sizes["fixed"],
                                "T_base_camera": fixed_solution["T_base_camera"],
                                "intrinsics_rms_px": fixed.get("intrinsics_rms_px", 0.0),
                                "extrinsic_residual": {"reprojection_px": fixed_solution["chain_reprojection_px"] or 0.0}}
            if fixed.get("intrinsics_source"):
                cameras["fixed"]["intrinsics_source"] = fixed["intrinsics_source"]
        joint_record = {"table_z_m": result.table_z_m, "joint_offsets_rad": new_offsets,
                        "joint_offset_changes_rad": {name: float(delta[i]) for i, name in enumerate(ARM_JOINTS)},
                        "estimated_joints": list(result.estimated_joints), "unobservable_joints": list(result.unobservable_joints),
                        "reprojection_rms_px": result.reprojection_rms_px, "touch_residuals_m": result.touch_residuals_m,
                        "board_thickness_m": thickness, "board_top_z_m": result.board_top_z_list_m,
                        "calibrated_at": utc_now(), "note": "extrinsics valid only with the matching deployment joint_mapping offsets"}
        out = {"measures": measures, "wrist_source": wrist_source, "fixed_intrinsics_source": fixed_source,
               "calibration_sources": {"wrist": self._source_evidence("S2", wrist_source),
                                       "fixed_intrinsics": self._source_evidence("S3", fixed_source),
                                       "touches": self._source_evidence("S1", "S1" if touch_source == "S1" else "active")},
               "touch_source": touch_source, "contact_review": deepcopy(self.run["contact_review"]),
               "samples": len(samples), "touches": len(touches),
               "placements": [{key: p[key] for key in ("session", "samples")} for p in placements],
               "solver": {"success": result.success, "message": result.message,
                          "effective_settings": result.effective_solver_settings, "nfev": result.nfev,
                          "initial_reprojection_rms_px": result.initial_reprojection_rms_px,
                          "board_tilt_rad": result.board_tilt_rad},
               "joint_calibration": joint_record, "hand_eye_spread": {key: spread[key] for key in ("translation_rms_m", "rotation_rms_rad")},
               "fixed_extrinsic_reason": fixed_reason}
        if not result.success:
            return dict(out, status="failed", reason=f"joint_solver_did_not_converge: {result.message}")
        if fixed_solution is None:
            return dict(out, status="failed", reason=f"fixed_extrinsic_unavailable: {fixed_reason}")
        if self._candidate_source_binding() != source_binding:
            return dict(out, status="failed", reason="calibration_source_changed_during_solve: rerun S5")
        candidate = self._run_dir() / "candidate" / "camera_calibration.json"
        cal.write_calibration_file(candidate, board=self._board_spec(), cameras=cameras, samples=len(samples), overwrite=True,
                                   joint_calibration=joint_record)
        adapter.validate_calibration_file(candidate)  # the candidate must load exactly like a reload
        diff = self._diff(cal.read_calibration_file(candidate), new_offsets, result.table_z_m)
        with self.lock:
            self.run["candidate"] = {"file": str(candidate), "joint_offsets_rad": new_offsets, "table_z_m": result.table_z_m,
                                     "diff": diff, "stale": False, "created_at": utc_now(),
                                     "touch_source": touch_source, "source_binding": source_binding}
            self.run["applied_version"] = None  # a new candidate must be reviewed and applied again
            write_json(self._run_dir() / "candidate" / "touches.json", _plain(touch_report))
            self.run["candidate"]["artifact_binding"] = self._candidate_artifact_binding(self.run["candidate"])
            self._save()
        out["diff"] = diff
        return out

    def _diff(self, new, new_offsets, new_table):
        """Old (active file + deployment offsets) vs new numbers for the review screen (m, rad, px)."""
        old = self._active_document()
        old_offsets = self._offsets()
        try:
            old_table = (json.loads(Path(self.adapter.calibration_path).read_text()).get("joint_calibration") or {}).get("table_z_m")
        except (OSError, ValueError, TypeError):
            old_table = None
        rows = [{"key": "table_z_m", "label": "桌面高度", "unit": "m", "old": old_table, "new": new_table}]
        rows += [{"key": f"joint_offset.{name}", "label": f"{name} 零位 offset_rad", "unit": "rad",
                  "old": old_offsets[name], "new": new_offsets[name]} for name in ARM_JOINTS]
        for camera, field in (("wrist", "T_link_camera"), ("fixed", "T_base_camera")):
            new_T = np.asarray(new["cameras"][camera][field])
            old_T = None if old is None or camera not in old["cameras"] else np.asarray(old["cameras"][camera][field])
            rows.append({"key": f"{camera}.{field}.translation_change_m", "label": f"{camera} {field} 平移变化", "unit": "m",
                         "old": None, "new": None if old_T is None else float(np.linalg.norm(new_T[:3, 3] - old_T[:3, 3]))})
            rows.append({"key": f"{camera}.{field}.rotation_change_rad", "label": f"{camera} {field} 旋转变化", "unit": "rad",
                         "old": None, "new": None if old_T is None else _rotation_angle(old_T[:3, :3].T @ new_T[:3, :3])})
        for camera in ("wrist", "fixed"):
            for index, label in ((0, "fx"), (4, "fy")):
                old_value = None if old is None or camera not in old["cameras"] else float(np.asarray(old["cameras"][camera]["K"]).flat[index])
                rows.append({"key": f"{camera}.K.{label}", "label": f"{camera} 焦距 {label}", "unit": "px", "old": old_value,
                             "new": float(np.asarray(new["cameras"][camera]["K"]).flat[index])})
        residual = lambda doc, path: None if doc is None else _dig(doc, path)  # noqa: E731
        for path, label, unit in ((("cameras", "wrist", "hand_eye_residual", "translation_m"), "手眼平移 RMS", "m"),
                                  (("cameras", "wrist", "hand_eye_residual", "rotation_rad"), "手眼旋转 RMS", "rad"),
                                  (("cameras", "fixed", "extrinsic_residual", "reprojection_px"), "固定相机链路重投影", "px"),
                                  (("cameras", "fixed", "intrinsics_rms_px"), "固定相机内参 RMS", "px")):
            rows.append({"key": ".".join(path[1:]), "label": label, "unit": unit, "old": residual(old, path), "new": residual(new, path)})
        return _plain(rows)

    # -------------------------------------------------------- apply/rollback ---
    def _set_offsets_live(self, offsets):
        """Switch the in-memory joint mapping to ``offsets`` and re-express the held goal in the new basis so the
        arm does not move (goal_new = goal_old + (offset_new - offset_old)). Cached plans are dropped."""
        adapter = self.adapter
        with adapter.bus_lock:
            previous = self._offsets()
            change = np.array([offsets[name] - previous[name] for name in ARM_JOINTS])
            for name in ARM_JOINTS:
                adapter.deployment["joint_mapping"][name]["offset_rad"] = float(offsets[name])
            mapping = getattr(adapter.driver, "mapping", None)
            if isinstance(mapping, dict) and mapping is not adapter.deployment["joint_mapping"]:
                for name in ARM_JOINTS:
                    mapping[name]["offset_rad"] = float(offsets[name])
            adapter.arm_goal = (np.asarray(adapter.arm_goal) + change).tolist()
            adapter._plans.clear()
            if adapter.last_sample is not None:
                adapter.last_sample = None

    def _install(self, version, offsets, previous_offsets):
        """Write deployment offsets (backup) and install the version's camera file; undo the offsets on failure."""
        adapter = self.adapter
        changed = any(offsets[name] != previous_offsets[name] for name in ARM_JOINTS)
        written = None
        # Refuse before anything is written: the stored camera file must load exactly like a reload.
        adapter.validate_calibration_file(self.bundle.version_file(version, "camera_calibration.json"))
        installed_document = json.loads(self.bundle.version_file(version, "camera_calibration.json").read_text())
        # reload_calibration.calibration is camera identity, not the calibrated joint geometry.
        # Keep absent geometry unknown for historical camera-only versions.
        table_z_m = (installed_document.get("joint_calibration") or {}).get("table_z_m")
        if changed:
            if not getattr(adapter, "deployment_path", None):
                raise StepRefused("deployment_path_unknown: cannot write joint_mapping offsets")
            written = write_joint_offsets(adapter.deployment_path, offsets)
        try:
            reload = install_camera_file(adapter, self.bundle.version_file(version, "camera_calibration.json"), adapter.calibration_path)
        except Exception as error:
            if written is not None:
                try:
                    write_joint_offsets(adapter.deployment_path, previous_offsets)
                except Exception as rb_err:
                    error.add_note(f"offset_rollback_failed: {type(rb_err).__name__}: {rb_err}")
            raise
        if changed:
            self._set_offsets_live(offsets)
        self.bundle.set_active(version)
        reload = dict(reload, table_z_m=table_z_m, table_z_unit="m", table_z_frame="so101_base")
        return reload, written

    def apply(self):
        """Store this run's candidate as a new bundle version and make it active (camera file + joint offsets)."""
        base = {"command": "calib_apply", "physics_advanced": False}
        if self.run is None or not self.run.get("candidate"):
            return dict(base, status="rejected", reason="no_candidate: run S5 first")
        if self.run["candidate"].get("stale") or not self._usable("S5"):
            return dict(base, status="rejected", reason="candidate_stale_or_S5_not_passed")
        if self.run.get("applied_version"):
            return dict(base, status="rejected", reason=f"already_applied:{self.run['applied_version']}")
        if self.running_step is not None or self.adapter.active_action_id is not None:
            return dict(base, status="rejected", reason="action_active")
        if not self.adapter.calibration_path:
            return dict(base, status="rejected", reason="camera_calibration_file_not_configured")
        candidate = self.run["candidate"]
        try:
            if candidate.get("source_binding") != self._candidate_source_binding():
                return dict(base, status="rejected", reason="candidate_source_changed_or_unrecorded: rerun S5")
        except (OSError, ValueError, KeyError, TypeError) as error:
            return dict(base, status="rejected", reason=f"candidate_source_unavailable: {error}")
        try:
            if candidate.get("artifact_binding") != self._candidate_artifact_binding(candidate):
                return dict(base, status="rejected", reason="candidate_artifact_changed_or_unrecorded: rerun S5")
        except (OSError, ValueError, KeyError, TypeError) as error:
            return dict(base, status="rejected", reason=f"candidate_evidence_unavailable: {error}")
        previous_offsets = self._offsets()
        imported = self.bundle.import_existing(previous_offsets)
        previous = self.bundle.active_version()
        reports = {}
        for step in wf.STEP_IDS:
            path = self._run_dir() / "reports" / f"{step}.json"
            if path.is_file():
                reports[f"reports/{step}.json"] = json.loads(path.read_text())
        touches = json.loads((Path(candidate["file"]).parent / "touches.json").read_text())
        joint = json.loads(Path(candidate["file"]).read_text())["joint_calibration"]
        steps = {step: {"status": record["status"], "finished_at": record["finished_at"]}
                 for step, record in self.run["steps"].items() if record["status"] != "not_started"}
        version = self.bundle.create_version(camera_file=candidate["file"], files=dict(reports, **{
            "joint_calibration.json": joint, "touches.json": touches}), manifest={
            "source": "workflow", "run_id": self.run["run_id"], "mode": self.run["mode"], "changes": self.run["changes"],
            "steps": steps, "board": self.run["board"], "contact_review": self.run.get("contact_review"),
            "joint_offsets_rad": candidate["joint_offsets_rad"],
            "joint_mapping": {name: dict(self.adapter.deployment["joint_mapping"][name],
                                          offset_rad=candidate["joint_offsets_rad"][name]) for name in ARM_JOINTS},
            "previous_joint_offsets_rad": previous_offsets, "table_z_m": candidate["table_z_m"], "previous": previous,
            "diff": candidate["diff"]})
        try:
            reload, written = self._install(version, candidate["joint_offsets_rad"], previous_offsets)
        except Exception as error:
            return dict(base, status="install_failed", reason=f"{type(error).__name__}: {error}", version=version,
                        active_version=self.bundle.active_version(), imported_version=imported)
        with self.lock:
            self.run.update(applied_version=version, previous_version=previous)
            self._save()
        return dict(base, status="applied", reason=None, version=version, previous_version=previous, imported_version=imported,
                    calibration=reload["calibration"], deployment_backup=None if written is None else written["backup"],
                    table_z_m=reload["table_z_m"], table_z_unit=reload["table_z_unit"], table_z_frame=reload["table_z_frame"],
                    joint_offsets_rad=candidate["joint_offsets_rad"], applies_to="observations captured after this command")

    def rollback(self, version=None):
        """Make ``version`` (default: the active version's ``previous``) active again: camera file + offsets."""
        base = {"command": "calib_rollback", "physics_advanced": False}
        if self.bundle is None:
            return dict(base, status="rejected", reason="calibration_bundle_not_configured")
        if self.running_step is not None or self.adapter.active_action_id is not None:
            return dict(base, status="rejected", reason="action_active")
        active = self.bundle.active_version()
        if version is None:
            version = (self.bundle.manifest(active).get("previous") if active else None)
        if not version or not (self.bundle.versions_dir / version / "manifest.json").is_file():
            return dict(base, status="rejected", reason=f"rollback_target_unknown:{version}")
        if version == active:
            return dict(base, status="rejected", reason="rollback_target_is_active")
        target = self.bundle.manifest(version)
        offsets = target.get("joint_offsets_rad") or self._offsets()
        previous_offsets = self._offsets()
        try:
            reload, written = self._install(version, {name: float(offsets[name]) for name in ARM_JOINTS}, previous_offsets)
        except Exception as error:
            return dict(base, status="install_failed", reason=f"{type(error).__name__}: {error}", active_version=active)
        with self.lock:
            if self.run is not None and self.run.get("applied_version") == active:
                # The run's result is no longer live: S6 verification needs a fresh apply.
                self.run.update(rolled_back_to=version, applied_version=None)
                self._save()
        return dict(base, status="rolled_back", reason=None, version=version, previous_active=active,
                    calibration=reload["calibration"], joint_offsets_rad=offsets,
                    table_z_m=reload["table_z_m"], table_z_unit=reload["table_z_unit"], table_z_frame=reload["table_z_frame"],
                    deployment_backup=None if written is None else written["backup"])

    def cancel(self):
        """Stop the running step: S3 capture stops; a motion step is held like ``hold`` (preemptive)."""
        base = {"command": "calib_step_cancel", "physics_advanced": False}
        if self.capture is not None and self.capture.running:
            snapshot = self.capture.stop("operator_cancel")
            if not self._handheld_writer_stopped(self.capture, snapshot):
                return dict(base, **self._s3_writer_pending(snapshot), step="S3")
            return dict(base, status="cancelled", step="S3", capture=snapshot, capture_stop_confirmed=True)
        if self.running_step is None:
            return dict(base, status="idle", step=None)
        step = self.running_step
        with self.adapter.state_lock:
            self.adapter.stop_generation += 1
            self.adapter.stop_reason = "operator_calibration_cancel"
        stop = self.adapter._stop()
        return dict(base, status="cancelled" if stop["confirmed"] else "unknown", step=step, stop_confirmation=stop)


def _dig(document, path):
    value = document
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return None if value is None else float(value)
