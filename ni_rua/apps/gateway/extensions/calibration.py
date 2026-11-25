"""Guided calibration workflow capability extension (``/v1/calibration/*``, ``/calibration``)."""
from __future__ import annotations

import asyncio
import re

from aiohttp import web

from ....core.presentation.feedback_delivery import execution_unconfirmed
from ....cortex.agents.base import session_turn_blocks_operator
from ....robot.calibration.workflow import state_machine as workflow
from ....robot.calibration.workflow.xy_verification import validate_xy_request
from ....robot.contracts import CalibratableBackend
from ...library.static_assets import static_directory

CALIBRATION_COMMANDS = frozenset(
    {
        "calibration_begin",
        "calibration_step_start",
        "calibration_cancel",
        "calibration_apply",
        "calibration_rollback",
    }
)
CALIBRATION_LEASE_FREE = frozenset({"calibration_cancel"})
CALIBRATION_PREEMPTIVE = frozenset({"calibration_cancel"})
PREVIEW_FILES = {"wrist": "calib_wrist.jpg", "fixed": "calib_fixed.jpg", "fixed_corners": "fixed_corners.jpg"}
VERSION_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}$")
STEP_PARAMS = {"confirmed", "board", "action"}
S3_ACTIONS = ("capture", "solve", "stop")
BADGE_LABELS = {
    "valid": "标定有效",
    "needed": "需要标定",
    "maybe_stale": "可能已过期",
    "unknown": "标定状态未知",
    "verification_failed": "高度验证未通过",
    "unverified": "标定已加载，未验收",
}


def active_verification(bundle):
    """S6 height verdict for the exact active version; missing evidence stays None."""
    if "verification" in bundle:
        value = bundle["verification"]
    else:
        version = next(
            (item for item in bundle.get("versions", []) if item.get("name") == bundle.get("active_version")),
            {},
        )
        value = version.get("verification")
    return value if isinstance(value, dict) else None


def calibration_badge(summary):
    """Classify the worker's ``calib_workflow_status(summary=True)`` reply for the header badge."""
    if not isinstance(summary, dict) or summary.get("available") is False or "bundle" not in summary:
        return dict(
            status="unknown",
            label=BADGE_LABELS["unknown"],
            calibrated_at=None,
            active_version=None,
            reason=(summary or {}).get("reason") if isinstance(summary, dict) else None,
            staleness=None,
            verification=None,
        )
    bundle, staleness = summary.get("bundle") or {}, summary.get("staleness") or {}
    verification = active_verification(bundle)
    present = bool(bundle.get("present")) and not bundle.get("load_error")
    if not present:
        status, reason = "needed", bundle.get("load_error") or "no_calibration_loaded"
    elif bundle.get("matches_active_version") is False:
        status, reason = "maybe_stale", bundle.get("binding_error") or "file_diverged_from_active_version"
    elif bundle.get("has_joint_calibration") is False:
        status, reason = "maybe_stale", "missing_joint_calibration"
    elif bundle.get("fixed_intrinsics_source") == "copied_from_wrist_same_model_unverified":
        status, reason = "maybe_stale", "fixed_intrinsics_unverified"
    elif verification is not None and verification.get("passed") is False:
        status, reason = "verification_failed", "active_height_verification_failed"
    elif staleness.get("status") == "shifted":
        status, reason = "maybe_stale", "board_shift_exceeds_threshold"
    elif verification is None or verification.get("passed") is not True:
        status, reason = "unverified", "active_height_verification_missing"
    else:
        status, reason = "valid", None
    return dict(
        status=status,
        label=BADGE_LABELS[status],
        reason=reason,
        calibrated_at=bundle.get("calibration_created_at"),
        active_version=bundle.get("active_version"),
        running_step=summary.get("running_step"),
        staleness=staleness or None,
        verification=verification,
        binding_diagnostics=bundle.get("binding_diagnostics"),
    )


def validate_begin(value):
    """``mode`` (+ ``changes`` for recalibrate, + optional ``board``) -> worker ``calib_workflow_begin`` arguments."""
    mode = value.get("mode")
    if mode not in workflow.MODES:
        raise ValueError("invalid_calibration_mode: expected " + " or ".join(workflow.MODES))
    changes = value.get("changes", [])
    if not isinstance(changes, list) or any(not isinstance(item, str) for item in changes):
        raise ValueError("invalid_calibration_changes: expected a list of change ids")
    if mode == "first_run" and changes:
        raise ValueError("calibration_first_run_takes_no_changes")
    workflow.plan_for(mode, changes)
    arguments = dict(mode=mode, changes=changes)
    if value.get("board") is not None:
        if not isinstance(value["board"], dict):
            raise ValueError("invalid_calibration_board: expected an object")
        arguments["board"] = value["board"]
    return arguments


def validate_step_start(value):
    """``step`` + optional ``params`` -> worker ``calib_step_start`` arguments."""
    step, params = value.get("step"), value.get("params", {})
    if step not in workflow.STEP_BY_ID:
        raise ValueError("calibration_step_unknown:" + str(step))
    if params is None:
        params = {}
    if step == "S6" and isinstance(params, dict) and params.get("action") in ("xy_capture", "xy_evaluate"):
        return dict(step=step, params=validate_xy_request(params))
    if not isinstance(params, dict) or set(params) - STEP_PARAMS:
        raise ValueError("invalid_step_params: allowed keys " + ", ".join(sorted(STEP_PARAMS)))
    out = {}
    if "confirmed" in params:
        confirmed = params["confirmed"]
        operator_checks = {key for key, spec in workflow.READINESS.items() if spec["kind"] == "operator"}
        if not isinstance(confirmed, list) or any(item not in operator_checks for item in confirmed):
            raise ValueError("invalid_step_confirmed: expected operator readiness ids " + ", ".join(sorted(operator_checks)))
        out["confirmed"] = confirmed
    if "board" in params:
        if step != "S2":
            raise ValueError("step_board_only_for_S2")
        if not isinstance(params["board"], dict):
            raise ValueError("invalid_calibration_board: expected an object")
        out["board"] = params["board"]
    if "action" in params:
        if step != "S3" or params["action"] not in S3_ACTIONS:
            raise ValueError("invalid_step_action: only S3 takes action " + "/".join(S3_ACTIONS))
        out["action"] = params["action"]
    return dict(step=step, params=out)


class CalibrationExtension:
    """Capability-mounted calibration workflow router for ``Workbench``."""

    name = "calibration"

    def __init__(self, workbench):
        self.workbench = workbench
        self.init_state()

    def init_state(self):
        self.workbench.calibration_workflow_state = dict(running_step=None, last=None)

    def supports(self, workbench=None) -> bool:
        target = workbench if workbench is not None else self.workbench
        robot = getattr(target, "robot", None)
        return isinstance(robot, CalibratableBackend) or callable(getattr(robot, "operator", None))

    def blockers(self, motion):
        wb = self.workbench
        blockers = []
        if wb.starting or session_turn_blocks_operator(wb.session):
            blockers.append("codex_turn_running")
        if wb.harness.active_action:
            blockers.append("action_in_progress")
        if wb.has_active_programs():
            blockers.append("program_active")
        if wb.calibration_workflow_state["running_step"] is not None:
            blockers.append("calibration_step_running")
        if motion:
            if wb.unknown_items():
                blockers.append("unknown_requires_acknowledgement_via_resume")
            if wb.operator_gate["torque"] == "off":
                blockers.append("torque_off")
        return blockers

    def _robot_ready(self):
        robot = self.workbench.robot
        if not (isinstance(robot, CalibratableBackend) or callable(getattr(robot, "operator", None))):
            raise ValueError("operator_command_unsupported:calibration")

    async def _status(self, command, **arguments):
        operator = getattr(self.workbench.robot, "operator", None)
        if not callable(operator):
            return dict(available=False, reason="operator_command_unsupported:" + command)
        try:
            result = await asyncio.to_thread(operator, command, **arguments)
        except (ValueError, RuntimeError, TimeoutError, OSError) as error:
            return dict(available=False, reason=str(error))
        return dict(result if isinstance(result, dict) else {"value": result}, available=True)

    async def snapshot(self):
        wb = self.workbench
        result = await self._status("calib_workflow_status")
        return dict(
            result,
            gateway=dict(
                running_step=wb.calibration_workflow_state["running_step"],
                last=wb.calibration_workflow_state["last"],
                blockers=dict(idle=self.blockers(False), motion=self.blockers(True)),
                accepting_actions=wb.harness.accepting,
                torque=wb.operator_gate["torque"],
                preview_names=list(PREVIEW_FILES),
            ),
        )

    async def summary(self):
        return calibration_badge(await self._status("calib_workflow_status", summary=True))

    async def handle_command(self, command, identifier, value):
        wb = self.workbench
        if command not in CALIBRATION_COMMANDS:
            raise ValueError("unsupported_control_command")
        self._robot_ready()
        if command == "calibration_cancel":
            return await self._cancel(identifier)
        if command == "calibration_begin":
            op, arguments, motion, closes = "calib_workflow_begin", validate_begin(value), False, False
        elif command == "calibration_step_start":
            checked = validate_step_start(value)
            motion = workflow.STEP_BY_ID[checked["step"]]["motion"]
            if checked["step"] == "S6" and checked["params"].get("action") in ("xy_capture", "xy_evaluate"):
                motion = False
            op, arguments, closes = "calib_step_start", checked, motion
        elif command == "calibration_apply":
            op, arguments, motion, closes = "calib_apply", {}, False, True
        else:
            version = value.get("version")
            if version is not None and (not isinstance(version, str) or not VERSION_NAME.match(version)):
                raise ValueError("invalid_calibration_version")
            op, arguments, motion, closes = "calib_rollback", {} if version is None else dict(version=version), False, True
        blockers = self.blockers(motion)
        if blockers:
            raise ValueError("calibration_blocked:" + ",".join(blockers))
        step = arguments.get("step")
        ext_source = f"calibration_step:{step}" if step is not None else command
        ext_motion = getattr(wb.harness, "external_motion", None)
        was_accepting = wb.harness.accepting
        stop_revision = wb.harness.stop_revision
        if motion:
            wb.calibration_workflow_state["running_step"] = step
        try:
            if closes:
                wb.harness.request_stop()
                stop_revision = wb.harness.stop_revision
                if callable(ext_motion):
                    await asyncio.to_thread(ext_motion, "begin", ext_source)
            result = await wb._operator(op, identifier, robot_lock=True, **arguments)
        except BaseException as error:
            wb.calibration_workflow_state.update(
                running_step=None,
                last=dict(command=command, step=step, status="unknown", reason=str(error)),
            )
            if closes and callable(ext_motion):
                await asyncio.to_thread(
                    ext_motion,
                    "end",
                    ext_source,
                    result=dict(status="unknown", reason=str(error)),
                    refresh_observation=False,
                )
            raise
        wb.calibration_workflow_state.update(
            running_step=None,
            last=dict(command=command, step=step, status=result.get("status"), reason=result.get("reason")),
        )
        unconfirmed = execution_unconfirmed(result)
        if not unconfirmed and result.get("status") in ("applied", "rolled_back"):
            if callable(getattr(wb.robot, "reload_deployment", None)):
                wb.robot.reload_deployment()
        response = dict(
            request_id=identifier,
            command=command,
            status="unknown" if unconfirmed else result.get("status", "acknowledged"),
            operator=result,
        )
        if step is not None:
            response["step"] = step
        if closes:
            will_reopen = (
                was_accepting
                and not unconfirmed
                and wb.harness.stop_revision == stop_revision
                and not wb.resume_blockers()
                and not wb.unknown_items()
            )
            if callable(ext_motion):
                await asyncio.to_thread(
                    ext_motion,
                    "end",
                    ext_source,
                    result=result,
                    refresh_observation=(not will_reopen and not unconfirmed),
                )
            admission, observation_id = await wb._reopen_after_operator(identifier, was_accepting, stop_revision, unconfirmed)
            response.update(admission=admission, observation_id=observation_id)
        if result.get("physics_advanced"):
            response["world"] = "objects_not_reset"
        return response

    async def _cancel(self, identifier):
        wb = self.workbench
        running = wb.calibration_workflow_state["running_step"]
        if running is not None and workflow.STEP_BY_ID[running]["motion"]:
            wb.harness.request_stop()
        result = await wb._operator("calib_step_cancel", identifier)
        status = "unknown" if execution_unconfirmed(result) else result.get("status", "acknowledged")
        return dict(
            request_id=identifier,
            command="calibration_cancel",
            status=status,
            operator=result,
            **({"admission": "closed"} if running is not None and workflow.STEP_BY_ID[running]["motion"] else {}),
        )

    def mount(self, app, control_dispatch=None):
        wb = self.workbench
        dispatch = control_dispatch or wb.control_dispatch

        def no_store(body):
            return web.json_response(body, headers={"Cache-Control": "no-store"})

        async def state(request):
            return no_store(await self.snapshot())

        async def summary(request):
            return no_store(await self.summary())

        async def step_status(request):
            step = request.match_info["step"]
            if step not in workflow.STEP_BY_ID:
                raise web.HTTPNotFound(text="calibration_step_unknown")
            return no_store(await self._status("calib_step_status", step=step))

        async def preview(request):
            name = request.match_info["name"]
            if name not in PREVIEW_FILES:
                raise web.HTTPNotFound(text="unknown_calibration_preview")
            path = wb.telemetry_directory() / PREVIEW_FILES[name]
            if path.is_symlink() or not path.is_file():
                raise web.HTTPNotFound(text="calibration_preview_unavailable")
            body = await asyncio.to_thread(path.read_bytes)
            return web.Response(body=body, content_type="image/jpeg", headers={"Cache-Control": "no-store"})

        async def start_step(request):
            value = await request.json()
            if not isinstance(value, dict):
                raise ValueError("control_requires_request_id")
            value = dict(value, command="calibration_step_start", step=request.match_info["step"])
            return await dispatch(value, request.headers.get("Origin"))

        async def xy_preview(request):
            identifier, camera = request.match_info["snapshot_id"], request.match_info["camera"]
            if not re.fullmatch(r"[a-f0-9]{32}", identifier) or camera not in ("fixed", "wrist"):
                raise web.HTTPNotFound(text="unknown_xy_preview")
            path = wb.telemetry_directory() / f"xy-{identifier}-{camera}.jpg"
            if path.is_symlink() or not path.is_file():
                raise web.HTTPNotFound(text="xy_preview_unavailable")
            return web.Response(
                body=await asyncio.to_thread(path.read_bytes),
                content_type="image/jpeg",
                headers={"Cache-Control": "no-store"},
            )

        async def simple(request, command):
            value = await request.json()
            if not isinstance(value, dict):
                raise ValueError("control_requires_request_id")
            return await dispatch(dict(value, command=command), request.headers.get("Origin"))

        async def begin(request):
            return await simple(request, "calibration_begin")

        async def cancel(request):
            return await simple(request, "calibration_cancel")

        async def apply(request):
            return await simple(request, "calibration_apply")

        async def rollback(request):
            return await simple(request, "calibration_rollback")

        async def page(request):
            path = static_directory() / "calibration.html"
            if not path.is_file():
                raise web.HTTPNotFound(text="calibration_page_not_built: run npm --prefix web run build")
            return web.FileResponse(path)

        app.router.add_get("/v1/calibration/state", state)
        app.router.add_get("/v1/calibration/summary", summary)
        app.router.add_get("/v1/calibration/steps/{step}", step_status)
        app.router.add_get("/v1/calibration/preview/{name}.jpg", preview)
        app.router.add_get("/v1/calibration/xy/{snapshot_id}/{camera}.jpg", xy_preview)
        app.router.add_post("/v1/calibration/begin", begin)
        app.router.add_post("/v1/calibration/steps/{step}/start", start_step)
        app.router.add_post("/v1/calibration/cancel", cancel)
        app.router.add_post("/v1/calibration/apply", apply)
        app.router.add_post("/v1/calibration/rollback", rollback)
        app.router.add_get("/calibration", page)


__all__ = [
    "BADGE_LABELS",
    "CALIBRATION_COMMANDS",
    "CALIBRATION_LEASE_FREE",
    "CALIBRATION_PREEMPTIVE",
    "CalibrationExtension",
    "PREVIEW_FILES",
    "S3_ACTIONS",
    "STEP_PARAMS",
    "VERSION_NAME",
    "active_verification",
    "calibration_badge",
    "validate_begin",
    "validate_step_start",
]
