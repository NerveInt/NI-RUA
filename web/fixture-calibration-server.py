"""Synthetic /calibration fixture: the real gateway Workbench over a fake SO-101 worker. No hardware.

The routes, gating, lease, idempotency and admission handling are the real gateway code
(``ni_rua.apps.gateway``); only the robot behind ``Workbench.robot.operator`` is fake. It answers the
``calib_*`` worker operations with the same typed shapes as ``ni_rua.robot.calibration.workflow.runner.CalibrationWorkflow``
(definition from ``state_machine.describe``, verdicts from ``state_machine.evaluate``,
thresholds from ``configs/so101.example.json``) and draws synthetic preview JPEGs. Never use as runtime
evidence: every number below is invented.

Run: mamba run -n ni-rua python web/fixture-calibration-server.py --port 8791   (never 8781)
Then open http://127.0.0.1:8791/calibration or /console (after ``npm --prefix web run build``).

Scenario knobs (fixture only, ``POST /fixture/calibration`` with a JSON object, ``GET`` reads it back):
``fail_steps`` (steps whose measures violate a threshold), ``step_seconds`` (how long a motion/solve step
runs, with progress updates), ``bad_readiness`` (auto readiness ids reported red), ``bundle_present``,
``staleness`` (``ok`` / ``shifted`` / ``board_not_visible`` / ``unavailable``), ``unknown_step`` (step
whose receipt is unconfirmed), ``stall_claims_during_step`` (a ``claim_input`` of the lease's last holder waits
until the running step ends, as the gateway before the lease-lock split queued it behind the step, then is answered
503 so the page claims again; other windows' claims reach the gateway, which refuses them with input_owner_busy_step),
``state_delay_s`` (seconds every ``GET /v1/calibration/state`` waits before it is answered: a stuck worker status
call). ``POST /fixture/reset`` restores the initial state. ``POST /fixture/expire-lease`` lets the writer lease lapse
now, as the gateway's 30 s lease does during a long real step (time warp, fixture only).

Like the real worker (``CalibrationWorkflow.status`` -> ``_write_overlays``), every full status call redraws the
wrist/fixed preview JPEGs (here with a frame counter), so the live preview only moves while the page polls state.
"""
import argparse
import asyncio
import json
import uuid
import sys
import tempfile
import threading
import time
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from aiohttp import web

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ni_rua.apps.gateway.workbench import Workbench  # noqa: E402
from ni_rua.robot.calibration.workflow import state_machine as wf  # noqa: E402

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
SETTINGS = json.loads((ROOT / "configs" / "so101.example.json").read_text())["calibration"]
DEFAULT_BOARD = dict(squares_x=7, squares_y=5, square_m=0.03, marker_m=0.022, dictionary="DICT_5X5_100", thickness_m=SETTINGS["board_thickness_m"])
DEFAULT_SCENARIO = dict(fail_steps=[], step_seconds=1.2, bad_readiness=[], bundle_present=True, staleness="ok", unknown_step=None,
                        default_board_thickness_m=DEFAULT_BOARD["thickness_m"], stall_claims_during_step=False, state_delay_s=0)

# Passing measures of each step (SI units), and the failing override applied when the step is in ``fail_steps``.
GOOD = {
    "S0": dict(preflight_ok=True, power_ok=True, torque_ok=True, telemetry_age_s=0.04, cameras_ok=True),
    "S1": dict(contacts=3, planned_points=3),
    "S2": dict(kept=21, wrist_intrinsics_rms_px=0.42, hand_eye_translation_rms_m=0.0031, hand_eye_rotation_rms_rad=0.011),
    "S3": dict(views=14, fixed_intrinsics_rms_px=0.55),
    "S4": dict(fixed_samples=5, fixed_chain_reprojection_px=6.1),
    "S5": dict(joint_reprojection_px=0.61, board_height_error_m=0.0011, max_touch_residual_m=0.0019, fixed_chain_reprojection_px=2.2, max_joint_offset_rad=0.05),
    "S6": dict(contacts=2, planned_points=2, max_verify_error_m=0.0012),
}
BAD = {
    "S0": dict(power_ok=False),
    "S1": dict(contacts=2),
    "S2": dict(hand_eye_translation_rms_m=0.014),
    "S3": dict(fixed_intrinsics_rms_px=3.4),
    "S4": dict(fixed_samples=0),
    "S5": dict(max_touch_residual_m=0.012),
    "S6": dict(max_verify_error_m=0.011),
}
AUTO_STEP_READINESS = {check: True for check in wf.READINESS if wf.READINESS[check]["kind"] == "auto"}


def utc(offset_s=0.0):
    return (datetime.now(timezone.utc) + timedelta(seconds=offset_s)).isoformat(timespec="seconds").replace("+00:00", "Z")


def jpeg(label, colour):
    image = np.full((240, 320, 3), 40, np.uint8)
    for x in range(40, 280, 30):
        for y in range(40, 200, 30):
            if (x // 30 + y // 30) % 2 == 0:
                cv2.rectangle(image, (x, y), (x + 30, y + 30), (220, 220, 220), -1)
            cv2.circle(image, (x, y), 4, colour, 2)
    cv2.putText(image, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return cv2.imencode(".jpg", image)[1].tobytes()


class FakeCalibrationRobot:
    """Stateful stand-in for the SO-101 worker's calibration operations (``Workbench.robot``)."""

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        telemetry = self.run_dir / "telemetry"
        telemetry.mkdir(parents=True, exist_ok=True)
        for name, label, colour in (("calib_wrist.jpg", "fixture wrist", (0, 200, 0)), ("calib_fixed.jpg", "fixture fixed", (0, 200, 0)),
                                    ("fixed_corners.jpg", "fixture hand-held", (0, 165, 255))):
            (telemetry / name).write_bytes(jpeg(label, colour))
        self.lock = threading.Lock()
        self.preview_frames = 0
        self.reset()

    def reset(self):
        with self.lock:
            self.scenario = deepcopy(DEFAULT_SCENARIO)
            self.run = None
            self.running_step = None
            self.capture = None
            self.offsets = {name: 0.0 for name in JOINTS}
            self.table_z_m = -0.0185
            self.versions = [dict(name="20261101T000000Z-imported", created_at="2026-11-01T00:00:00Z", source="imported_existing_file", mode=None,
                                  changes=[], steps={}, table_z_m=-0.0185, joint_offsets_rad=dict(self.offsets), previous=None,
                                  verification=dict(passed=True, max_error_m=0.0012))]
            self.active = self.versions[0]["name"]
            self.calibrated_at = "2026-11-01T00:00:00Z"
            self.cancelled = threading.Event()

    # ---- worker operations ------------------------------------------------
    def operator(self, command, progress_callback=None, **arguments):
        handler = getattr(self, command, None)
        if handler is None:
            raise ValueError(f"unknown_operator_command:{command}")
        return dict(handler(**arguments), command=command)

    def close(self):
        return None

    def calib_workflow_status(self, summary=False):
        with self.lock:
            snapshot = dict(bundle=self._bundle(), staleness=self._staleness(), running_step=self.running_step)
            if summary:
                return snapshot
            self._write_previews()
            run = deepcopy(self.run)
            readiness = {step: self._readiness(step) for step in wf.STEP_IDS} if run else {}
            return dict(snapshot, definition=wf.describe(SETTINGS), run=run, readiness=readiness, handheld=self._handheld(),
                        default_board=dict(DEFAULT_BOARD, thickness_m=self.scenario["default_board_thickness_m"]),
                        xy_defaults={"max_xy_error_m": SETTINGS["xy_max_error_m"],
                                     "max_reprojection_error_px": SETTINGS["xy_max_reprojection_error_px"],
                                     "min_parallax_deg": SETTINGS["xy_min_parallax_deg"]},
                        settings={key: SETTINGS[key] for key in ("touch_points_xy_m", "verify_points_xy_m", "board_thickness_m")})

    def calib_step_status(self, step):
        with self.lock:
            return dict(status="ok", step=step, state=(self.run or {}).get("steps", {}).get(step), readiness=self._readiness(step))

    def calib_workflow_begin(self, mode, changes=(), board=None):
        with self.lock:
            if self.running_step:
                return dict(status="rejected", reason="calibration_step_running")
            plan = wf.plan_for(mode, changes)
            if self.run is not None:
                self.run["closed"] = True
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-fixture"
            self.run = dict(schema="ni_rua.so101.calibration_run.v1", run_id=run_id, created_at=utc(), mode=mode, changes=list(changes), plan=plan,
                            board=dict(DEFAULT_BOARD, **(board or {})), closed=False, candidate=None, applied_version=None, previous_version=None,
                            steps={step: dict(status="not_started", started_at=None, finished_at=None, reason=None, progress=[], result=None)
                                   for step in wf.STEP_IDS})
            self.capture = None
        return dict(status="started", run_id=run_id, plan=plan)

    def calib_step_cancel(self):
        self.cancelled.set()
        with self.lock:
            running = self.running_step
            if self.capture and self.capture["status"] == "running":
                self.capture.update(status="cancelled", reason="operator_cancel")
                return dict(status="cancelled", step="S3")
        return dict(status="cancelled" if running else "idle", step=running)

    def calib_step_start(self, step, params=None):
        params = dict(params or {})
        with self.lock:
            if self.run is None:
                return dict(status="rejected", reason="calibration_run_not_started")
            if self.running_step:
                return dict(status="rejected", reason=f"calibration_step_running:{self.running_step}")
            if step == "S6" and params.get("action") == "xy_capture":
                identifier = uuid.uuid4().hex
                self.run["xy_capture"] = dict(snapshot_id=identifier, at=utc())
                for name in ("fixed", "wrist"):
                    (self.run_dir / "telemetry" / f"xy-{identifier}-{name}.jpg").write_bytes(jpeg(f"XY fixture {name}", (0, 200, 0)))
                return dict(status="captured", snapshot=self.run["xy_capture"], physics_advanced=False)
            if step == "S6" and params.get("action") == "xy_evaluate":
                if params["snapshot_id"] != self.run.get("xy_capture", {}).get("snapshot_id"):
                    return dict(status="rejected", reason="xy_snapshot_mismatch")
                result = dict(status="failed", passed=False, xy_error_m=.012,
                              observed_minus_predicted_m=[.012, 0., .001], parameters=params)
                self.run.setdefault("xy_verifications", []).append(result)
                return dict(status="evaluated", result=result, physics_advanced=False)
            if step == "S3" and params.get("action") in ("stop", "solve"):
                return self._s3_followup(params["action"])
            failed = [check for check, value in self._readiness(step).items() if value["kind"] == "auto" and not value["ok"]]
            if failed:
                return dict(status="rejected", reason="readiness_failed:" + ",".join(failed))
            missing = [check for check, value in self._readiness(step).items() if value["kind"] == "operator" and check not in params.get("confirmed", [])]
            if missing:
                return dict(status="rejected", reason="readiness_unconfirmed:" + ",".join(missing))
            if step == "S2" and params.get("board"):
                self.run["board"] = dict(self.run["board"], **params["board"])
            self._begin(step)
            if step == "S3":
                self.capture = dict(status="running", started=time.monotonic(), reason=None)
                return dict(status="running", reason=None, capture=self._handheld())
            self.running_step = step
            self.cancelled.clear()
        return self._run_step(step)

    def calib_apply(self):
        with self.lock:
            run = self.run
            if not run or not run["candidate"]:
                return dict(status="rejected", reason="no_candidate: run S5 first")
            if run["candidate"]["stale"] or run["steps"]["S5"]["status"] != "passed":
                return dict(status="rejected", reason="candidate_stale_or_S5_not_passed")
            if run["applied_version"]:
                return dict(status="rejected", reason=f"already_applied:{run['applied_version']}")
            name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            previous = self.active
            self.versions.insert(0, dict(name=name, created_at=utc(), source="workflow", mode=run["mode"], changes=run["changes"],
                                         steps={}, table_z_m=run["candidate"]["table_z_m"],
                                         joint_offsets_rad=run["candidate"]["joint_offsets_rad"], previous=previous, verification=None))
            self.active, self.calibrated_at = name, utc()
            self.offsets, self.table_z_m = dict(run["candidate"]["joint_offsets_rad"]), run["candidate"]["table_z_m"]
            run.update(applied_version=name, previous_version=previous)
        return dict(status="applied", reason=None, version=name, previous_version=previous, applies_to="observations captured after this command")

    def calib_rollback(self, version=None):
        with self.lock:
            active = next(item for item in self.versions if item["name"] == self.active)
            version = version or active["previous"]
            target = next((item for item in self.versions if item["name"] == version), None)
            if target is None:
                return dict(status="rejected", reason=f"rollback_target_unknown:{version}")
            if version == self.active:
                return dict(status="rejected", reason="rollback_target_is_active")
            previous_active, self.active = self.active, version
            self.offsets, self.table_z_m, self.calibrated_at = dict(target["joint_offsets_rad"]), target["table_z_m"], target["created_at"]
            if self.run and self.run.get("applied_version") == previous_active:
                self.run.update(rolled_back_to=version, applied_version=None)
        return dict(status="rolled_back", reason=None, version=version, previous_active=previous_active)

    # ---- helpers (call with the lock held unless noted) --------------------
    def _bundle(self):
        present = self.scenario["bundle_present"]
        if not present:
            return dict(configured=True, present=False, calibration_created_at=None, residuals=None, load_error=None, active_version=None,
                        versions=[], matches_active_version=None, table_z_m=None, joint_offsets_rad=dict(self.offsets))
        return dict(configured=True, present=True, calibration_created_at=self.calibrated_at, load_error=None, active_version=self.active,
                    residuals=dict(wrist_intrinsics_rms_px=0.31, hand_eye_translation_rms_m=0.0024, fixed_chain_reprojection_px=3.7),
                    versions=deepcopy(self.versions), matches_active_version=True, table_z_m=self.table_z_m, joint_offsets_rad=dict(self.offsets))

    def _write_previews(self):
        self.preview_frames += 1
        telemetry = self.run_dir / "telemetry"
        for name, label in (("calib_wrist.jpg", "fixture wrist"), ("calib_fixed.jpg", "fixture fixed")):
            temporary = telemetry / f".{name}.tmp"
            temporary.write_bytes(jpeg(f"{label} #{self.preview_frames}", (0, 200, 0)))
            temporary.replace(telemetry / name)

    def _staleness(self):
        status = self.scenario["staleness"]
        out = dict(status=status, checked_at=utc(), threshold_px=SETTINGS["staleness_max_board_shift_px"])
        if status in ("ok", "shifted"):
            out["rms_px"] = 3.1 if status == "ok" else 41.7
        elif status == "unavailable":
            out["reason"] = "arm_moving_or_no_joint_sample"
        return out

    def _readiness(self, step):
        out = {}
        for check in wf.STEP_BY_ID[step]["readiness"]:
            spec = wf.READINESS[check]
            ok = None
            if spec["kind"] == "auto":
                ok = check not in self.scenario["bad_readiness"]
                if ok and check == "joint_inputs_ready":
                    ok = bool(self.run and self.run["steps"]["S4"]["status"] == "passed")
                elif ok and check == "review_ready":
                    ok = bool(self.run and self.run["steps"]["S5"]["status"] == "passed")
                elif ok and check == "applied":
                    ok = bool(self.run and self.run.get("applied_version"))
            detail = None if ok in (None, True) else "fixture: 未满足"
            if check == "board_both_visible":
                detail = f"wrist={26 if ok else 2} fixed={31 if ok else 0} (需要 ≥ {SETTINGS['min_board_corners']})"
            out[check] = dict(ok=ok, label=spec["label"], kind=spec["kind"], detail=detail)
        return out

    def _begin(self, step):
        statuses = {name: record["status"] for name, record in self.run["steps"].items()}
        for name, status in wf.mark_stale(statuses, step).items():
            if status == "stale" and self.run["steps"][name]["status"] != "stale":
                self.run["steps"][name].update(status="stale", reason=f"{step}_rerun")
        if step in ("S1", "S2", "S3", "S4", "S5") and self.run["candidate"]:
            self.run["candidate"]["stale"] = True
        self.run["steps"][step].update(status="running", started_at=utc(), finished_at=None, reason=None, progress=[], result=None)

    def _progress(self, step, update):
        with self.lock:
            record = self.run["steps"][step]
            record["progress"] = (record["progress"] + [dict(update, at=utc())])[-40:]

    def _run_step(self, step):
        """Runs outside the lock: the page polls status meanwhile, like the real worker."""
        phases = {"S0": [dict(phase="bus"), dict(phase="cameras")], "S1": [dict(phase="moving", point_index=i, point_count=3) for i in (1, 2, 3)],
                  "S2": [dict(phase="moving", pose_index=i, pose_count=21, kept_count=i - 1) for i in range(1, 22)]
                  + [dict(phase="parking", kept_count=21), dict(phase="solving", kept_count=21)],
                  "S4": [dict(phase="moving", pose_index=i, pose_count=3) for i in (1, 2, 3)],
                  "S5": [dict(phase="loading"), dict(phase="solving"), dict(phase="cross_check")],
                  "S6": [dict(phase="moving", point_index=i, point_count=2) for i in (1, 2)]}[step]
        pause = self.scenario["step_seconds"] / len(phases)
        for update in phases:
            self._progress(step, update)
            if self.cancelled.wait(pause):
                break
        measures = dict(GOOD[step], **(BAD[step] if step in self.scenario["fail_steps"] else {}))
        cancelled = self.cancelled.is_set()
        with self.lock:
            verdict = wf.evaluate(step, measures, SETTINGS)
            status = "passed" if verdict["passed"] and not cancelled else "failed"
            reason = "operator_cancel" if cancelled else None if status == "passed" else "; ".join(f"{r['label']} 未通过" for r in verdict["criteria"] if not r["passed"])
            result = dict(status="completed" if status == "passed" else "failed", measures=measures, verdict=verdict)
            if step in ("S1", "S6"):
                table = self.table_z_m if step == "S6" else -0.0178
                points = SETTINGS["touch_points_xy_m" if step == "S1" else "verify_points_xy_m"]
                result["touches"] = [dict(index=i + 1, xy=xy, status="contact", contact_z_m=table + 0.0004 * i, trigger="lag", lag_rad=0.031)
                                     for i, xy in enumerate(points)]
            if step == "S5" and status == "passed":
                self.run["candidate"] = dict(stale=False, created_at=utc(), table_z_m=-0.0179, joint_offsets_rad={name: 0.01 * i for i, name in enumerate(JOINTS)},
                                             diff=self._diff(), touch_source="S1")
                self.run["applied_version"] = None
            self.run["steps"][step].update(status=status, reason=reason, finished_at=utc(), result=result)
            if step == "S6":
                version = next(v for v in self.versions if v["name"] == self.active)
                version["verification"] = dict(passed=status == "passed", max_error_m=measures["max_verify_error_m"])
            self.running_step = None
            receipt = dict(status=status, reason=reason, result=deepcopy(result), physics_advanced=False)
            if self.scenario["unknown_step"] == step:
                receipt.update(status="unknown", stop_confirmation=dict(confirmed=False))
        return receipt

    def _diff(self):
        rows = [dict(key="table_z_m", label="桌面高度", unit="m", old=self.table_z_m, new=-0.0179)]
        rows += [dict(key=f"joint_offset.{name}", label=f"{name} 零位 offset_rad", unit="rad", old=self.offsets[name], new=0.01 * i) for i, name in enumerate(JOINTS)]
        rows += [dict(key="wrist.T_link_camera.translation_change_m", label="wrist T_link_camera 平移变化", unit="m", old=None, new=0.0021),
                 dict(key="fixed.T_base_camera.rotation_change_rad", label="fixed T_base_camera 旋转变化", unit="rad", old=None, new=0.0087),
                 dict(key="wrist.K.fx", label="wrist 焦距 fx", unit="px", old=612.4, new=615.1),
                 dict(key="fixed.extrinsic_residual.reprojection_px", label="固定相机链路重投影", unit="px", old=3.7, new=2.2)]
        return rows

    def _handheld(self):
        if self.capture is None:
            return None
        elapsed = time.monotonic() - self.capture["started"]
        kept = 14 if self.capture["status"] != "running" else min(14, int(elapsed * 4))
        grid = [[min(3, kept // 5 + (r + c) % 2) for c in range(3)] for r in range(3)]
        return dict(camera="fixed", status=self.capture["status"], reason=self.capture["reason"], kept=kept, target_views=SETTINGS["handheld_target_views"],
                    min_views=SETTINGS["handheld_min_views"], ready_to_solve=kept >= SETTINGS["handheld_min_views"], frames=kept * 9, dropped_wrong_size=0,
                    image_size=[640, 480], guidance=["再靠近一些", "倾斜标定板"] if kept < SETTINGS["handheld_min_views"] else ["覆盖足够，可以求解"],
                    coverage=dict(grid=grid, distance=dict(far=kept // 3, mid=kept // 2, near=kept // 4), tilt=dict(frontal=kept // 2, small=kept // 3, large=kept // 5)),
                    last=dict(corners=24), timeout_s=SETTINGS["handheld_timeout_s"])

    def _s3_followup(self, action):
        if self.capture is None:
            return dict(status="rejected", reason="S3_capture_not_started")
        if action == "stop":
            self.capture.update(status="stopped")
            return dict(status="running", reason=None, capture=self._handheld())
        self.capture.update(status="stopped")
        measures = dict(GOOD["S3"], **(BAD["S3"] if "S3" in self.scenario["fail_steps"] else {}))
        verdict = wf.evaluate("S3", measures, SETTINGS)
        status = "passed" if verdict["passed"] else "failed"
        reason = None if status == "passed" else "; ".join(f"{r['label']} 未通过" for r in verdict["criteria"] if not r["passed"])
        result = dict(measures=measures, verdict=verdict, capture=self._handheld())
        self.run["steps"]["S3"].update(status=status, reason=reason, finished_at=utc(), result=result)
        return dict(status=status, reason=reason, result=deepcopy(result), physics_advanced=False)


def build(port, directory):
    robot = FakeCalibrationRobot(Path(directory) / "robot-run")
    workbench = Workbench(Path(directory) / "gateway", robot=robot)
    workbench.starting, workbench.status = False, "running"
    session = SimpleNamespace(thread_id="fixture-thread", turn_id=None, state="completed", effort="high")
    session.interrupt = lambda: setattr(session, "state", "interrupted")
    session.close = lambda: setattr(session, "state", "closed")
    workbench.session = session
    counter = [0]

    def observe(correlation=None):
        counter[0] += 1
        observation = dict(observation_id=f"obs-fixture-{counter[0]}")
        workbench.harness.latest = observation
        return observation

    workbench.harness.observe = observe
    workbench.harness.capabilities = {"robot_model_id": "so101"}
    workbench.harness.accepting = True
    app = workbench.app()

    async def scenario(request):
        if request.method == "POST":
            body = await request.json()
            unknown = set(body) - set(DEFAULT_SCENARIO)
            if unknown:
                raise web.HTTPBadRequest(text="unknown_scenario_keys:" + ",".join(sorted(unknown)))
            robot.scenario.update(body)
        return web.json_response(robot.scenario)

    async def reset(request):
        robot.reset()
        workbench.controller = None  # a previous browser run may have left its writer lease behind
        workbench.operator_gate["unknown"].clear()  # an UNKNOWN receipt from the previous run gates every later call
        workbench.harness.accepting = True
        return web.json_response(robot.scenario)

    @web.middleware
    async def stalls(request, handler):
        """Fixture-only delays in front of the real routes (scenario ``stall_claims_during_step``, ``state_delay_s``)."""
        if request.method == "GET" and request.path == "/v1/calibration/state" and robot.scenario["state_delay_s"]:
            await asyncio.sleep(robot.scenario["state_delay_s"])
        if request.method == "POST" and request.path == "/v1/control" and robot.scenario["stall_claims_during_step"] \
                and robot.running_step is not None and (body := await request.json()).get("command") == "claim_input" \
                and body.get("client_id") == workbench.lease_client:
            while robot.running_step is not None:
                await asyncio.sleep(0.05)
            return web.json_response(dict(error="fixture_claim_stalled_until_step_end"), status=503)
        return await handler(request)

    app.middlewares.append(stalls)

    async def expire_lease(request):
        if workbench.controller:
            workbench.controller["expires_at"] = 0
        return web.json_response(dict(expired=workbench.controller is not None))

    app.router.add_route("*", "/fixture/calibration", scenario)
    app.router.add_post("/fixture/reset", reset)
    app.router.add_post("/fixture/expire-lease", expire_lease)
    return workbench, app


async def serve(port):
    with tempfile.TemporaryDirectory() as directory:
        workbench, app = build(port, directory)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", port).start()
        print(f"synthetic calibration fixture on http://127.0.0.1:{port}/calibration", flush=True)
        try:
            await asyncio.Event().wait()
        finally:
            await runner.cleanup()
            await workbench.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8791)
    arguments = parser.parse_args()
    if arguments.port == 8781:
        parser.error("port 8781 belongs to the real gateway")
    try:
        asyncio.run(serve(arguments.port))
    except KeyboardInterrupt:
        pass
