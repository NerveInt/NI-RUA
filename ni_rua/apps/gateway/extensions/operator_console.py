"""Operator console capability extension: live telemetry, safety/operator commands, and console routes."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from aiohttp import web

from ....core.contracts import IDENTITY
from ....core.presentation.feedback_delivery import execution_unconfirmed
from ....cortex.agents.base import session_capabilities, session_turn_blocks_operator
from ....cortex.agents.reasoning_options import SERVICE_TIERS
from ....robot.contracts import OperatorBackend, VideoBackend
from ...evaluation.episode import record_video
from ...library.static_assets import static_directory
from .calibration import CALIBRATION_COMMANDS, CALIBRATION_LEASE_FREE, CALIBRATION_PREEMPTIVE

TELEMETRY_SCHEMA = "ni_rua.so101.telemetry.v1"
TELEMETRY_STALE_AFTER_S = 2.0
CAMERA_ID = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
OPERATOR_COMMANDS = (
    frozenset(
        {
            "hold",
            "estop",
            "relax",
            "torque_on",
            "set_runtime_limits",
            "go_home",
            "resume",
            "set_effort",
            "set_model",
            "set_service_tier",
            "calibrate",
            "reload_calibration",
            "handheld_capture",
            "handheld_solve",
            "save_video",
            "retry_model_requests",
            "apply_agent_settings",
            "handoff_run",
        }
    )
    | CALIBRATION_COMMANDS
)
HANDHELD_COMMANDS = frozenset({"handheld_capture", "handheld_solve"})
HANDHELD_CAPTURE_ACTIONS = ("start", "stop")
HANDHELD_STATUS_FILE = "handheld.json"
HANDHELD_PREVIEW_FILE = "fixed_corners.jpg"
IDLE_ROBOT_COMMANDS = frozenset({"go_home", "calibrate", "reload_calibration"})
CALIBRATION_PROGRESS_KEEP = 60
MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@+\[\]-]{0,199}\Z")
NO_IMAGE_INPUT_LABEL = " · 无图像输入"
LEASE_FREE_COMMANDS = frozenset({"hold", "estop"}) | CALIBRATION_LEASE_FREE
PREEMPTIVE_COMMANDS = frozenset({"hold", "estop", "relax"}) | CALIBRATION_PREEMPTIVE
LEASE_COMMANDS = frozenset({"claim_input", "renew_input", "release_input"})
LEASE_BUSY_CODE = "input_owner_busy_step"
LEASE_BUSY_REFUSAL = (
    LEASE_BUSY_CODE
    + ": a long command (e.g. a calibration step or go_home) is running for the window "
    "that held control, so control cannot change hands until it ends. hold, estop and calibration_cancel "
    "still work from any window"
)
DEFAULT_SPEED_SCALE_CAP = 0.9
SPEED_SCALE_CAP_RANGE = (0.1, 1.0)
SECRET_KEY = re.compile(r"(api[_-]?key|token|secret|password|credential|auth)", re.I)
RECEIPT_FIELDS = (
    "status",
    "reason",
    "stop_confirmed",
    "stop_confirmation",
    "execution_uncertain",
    "effective_tolerances",
    "effective_limits",
    "result_observation_id",
    "reset_strategy",
)


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_utc(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def telemetry_view(directory, now=None, stale_after_s=TELEMETRY_STALE_AFTER_S):
    """Read ``state.json`` and classify freshness from ``captured_at`` only."""
    now = now or datetime.now(timezone.utc)
    view = dict(available=False, stale=True, age_s=None, stale_after_s=stale_after_s, reason=None, state=None)
    path = Path(directory) / "state.json"
    if path.is_symlink() or not path.is_file():
        view["reason"] = "telemetry_unavailable"
        return view
    try:
        state = json.loads(path.read_text())
    except (OSError, ValueError):
        view["reason"] = "telemetry_unreadable"
        return view
    if not isinstance(state, dict) or state.get("schema") != TELEMETRY_SCHEMA:
        view["reason"] = "telemetry_schema_mismatch"
        return view
    captured = _parse_utc(state.get("captured_at"))
    view.update(available=True, state=state)
    if captured is None:
        view["reason"] = "telemetry_captured_at_missing"
        return view
    age = (now - captured).total_seconds()
    view["age_s"] = round(age, 3)
    if age < -stale_after_s:
        view["reason"] = "telemetry_clock_ahead"
    elif age > stale_after_s:
        view["reason"] = "telemetry_stale"
    else:
        view["stale"] = False
    return view


def redact(value):
    if isinstance(value, dict):
        return {
            key: "[redacted]" if SECRET_KEY.search(str(key)) and item not in (None, "", [], {}) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def _receipt_projection(receipt):
    view = {key: receipt[key] for key in RECEIPT_FIELDS if key in receipt}
    for key, item in receipt.items():
        if "residual" in key or key.endswith("_error_m") or key.endswith("_error_rad"):
            view[key] = item
    return view


def _compose_motion_video(robot, export, *, run_dir=None, config_dict=None):
    composer = getattr(robot, "compose_video", None)
    if callable(composer):
        return composer(export)
    from ....robot.backends.so101.motion_video_composite import compose
    return compose(export, run_dir=run_dir, config_dict=config_dict)


class OperatorConsoleExtension:
    """Capability-mounted operator console extension for ``Workbench``."""

    name = "operator_console"

    def __init__(self, workbench):
        self.workbench = workbench
        self.init_state()

    def init_state(self):
        wb = self.workbench
        wb.operator_gate = dict(torque=None, unknown=[], acknowledged_actions=[], speed_scale_cap=None, last=None)
        wb.safety_lock = asyncio.Lock()
        wb.lease_lock = asyncio.Lock()
        wb.calibration_state = dict(running=False, request_id=None, started_at=None, finished_at=None, progress=[], result=None)
        wb.handheld_state = dict(solving=False, solve_request_id=None, solve_result=None)
        wb.video_export_state = None
        if hasattr(wb, "_calibration_ext"):
            wb._calibration_ext.init_state()

    def supports(self, workbench=None) -> bool:
        return True

    def telemetry_directory(self):
        wb = self.workbench
        run_dir = getattr(wb.robot, "run_dir", None)
        return Path(run_dir if isinstance(run_dir, (str, Path)) else wb.root / "private" / "robot") / "telemetry"

    def joint_limits(self):
        wb = self.workbench
        cached = getattr(wb, "_console_joint_limits", False)
        if cached is not False:
            return cached
        try:
            limits = getattr(wb.robot, "joint_limits_rad", lambda: None)()
        except Exception:
            limits = None
        wb._console_joint_limits = limits
        return limits

    def unknown_action_ids(self):
        wb = self.workbench
        reconciled = getattr(wb.harness, "_reconciled_late_actions", ())
        ids = []
        for identifier, state in wb.harness.states.items():
            if identifier in reconciled or not execution_unconfirmed(state):
                continue
            try:
                request = wb.store.read("action_request", identifier)
            except KeyError:
                ids.append(identifier)
                continue
            if all(request.get(key) == wb.identity[key] for key in IDENTITY):
                ids.append(identifier)
        return ids

    def unknown_items(self):
        wb = self.workbench
        acknowledged = set(wb.operator_gate["acknowledged_actions"])
        items = [
            dict(id=identifier, source="action_receipt", reason=wb.harness.states[identifier].get("reason"))
            for identifier in self.unknown_action_ids()
            if identifier not in acknowledged
        ]
        programs = getattr(wb, "programs", None) or getattr(wb.harness, "programs", None)
        if programs is not None:
            unresolved_fn = getattr(programs, "unresolved_jobs", None)
            if callable(unresolved_fn):
                try:
                    prog_ids = unresolved_fn(identity=wb.identity, acknowledged=acknowledged)
                except TypeError:
                    prog_ids = [
                        jid for jid in unresolved_fn()
                        if jid not in acknowledged and f"program:{jid}" not in acknowledged
                    ]
                status_fn = getattr(programs, "status", None)
                for jid in prog_ids:
                    p_reason = "program_stop_unconfirmed"
                    if callable(status_fn):
                        try:
                            p_reason = status_fn(jid).get("reason") or p_reason
                        except Exception:
                            pass
                    items.append(dict(id="program:" + jid, source="program_job", reason=p_reason))
        return items + [dict(entry) for entry in wb.operator_gate["unknown"]]

    def resume_blockers(self):
        wb = self.workbench
        blockers = []
        if wb.replay:
            blockers.append("replay_is_read_only")
        if wb.starting:
            blockers.append("run_starting")
        if session_turn_blocks_operator(wb.session):
            blockers.append("codex_turn_running")
        if wb.harness.active_action:
            blockers.append("action_in_progress")
        programs = getattr(wb, "programs", None) or getattr(wb.harness, "programs", None)
        if programs and programs.snapshot()["active_jobs"]:
            blockers.append("program_active_or_unresolved")
        if wb.operator_gate["torque"] == "off":
            blockers.append("torque_off_requires_torque_on")
        if getattr(wb.harness, "episode_terminal", None):
            blockers.append("native_episode_ended")
        return blockers

    def console_state(self):
        wb = self.workbench
        recent = []
        for identifier in list(wb.harness.states)[-12:]:
            receipt = wb.harness.states[identifier]
            try:
                kind = wb.store.read("action_request", identifier).get("type")
            except KeyError:
                kind = None
            recent.append(dict(action_id=identifier, type=kind, **_receipt_projection(receipt)))
        try:
            plan = wb.harness.task_planning.decision_context(wb.identity)
        except Exception:
            plan = None
        config = redact(wb.config.to_dict())
        model, features, backend = config.get("model", {}), config.get("features", {}), config.get("backend", {})
        current_model = getattr(wb.session, "model", None) or model.get("name")
        model_options = wb.annotate_model_options(self._model_options(model.get("name")))
        proxy = getattr(wb.session, "model_proxy", None)
        probes = (
            dict(proxy.select_model(current_model) or {})
            if callable(getattr(proxy, "select_model", None))
            else dict(getattr(proxy, "service_tier_probes", {}) or {})
        )
        operator_supported = isinstance(wb.robot, OperatorBackend) or callable(getattr(wb.robot, "operator", None))
        return dict(
            **wb.identity,
            status=wb.status,
            codex_status=wb.session.state,
            accepting_actions=wb.harness.accepting,
            active_action=wb.harness.active_action,
            operator_gate=dict(wb.operator_gate, unknown=self.unknown_items()),
            resume=dict(
                blockers=self.resume_blockers(),
                requires_acknowledgement=[item["id"] for item in self.unknown_items()],
            ),
            latest_observation_id=(wb.harness.latest or {}).get("observation_id"),
            recent_actions=list(reversed(recent)),
            task_plan=plan,
            operator_supported=operator_supported,
            calibration=dict(
                wb.calibration_state,
                progress=list(wb.calibration_state["progress"]),
                handheld=self.handheld_view(),
            ),
            video_export=wb.video_export_state,
            model_transport=self.model_transport_view(),
            telemetry_directory_configured=True,
            joint_limits_rad=self.joint_limits(),
            settings=dict(
                runtime=dict(
                    speed_scale_cap=dict(
                        value=self._effective_speed_scale_cap(),
                        default=DEFAULT_SPEED_SCALE_CAP,
                        range=list(SPEED_SCALE_CAP_RANGE),
                        command="set_runtime_limits",
                        applies="later motion; can only lower deployment speed",
                    ),
                    model=dict(
                        value=current_model,
                        choices=[item["value"] for item in model_options],
                        options=model_options,
                        free_text=True,
                        command="set_model",
                        applies="next agent turn",
                        image_input=next(
                            (item.get("image_input") for item in model_options if item["value"] == current_model),
                            None,
                        ),
                    ),
                    effort=dict(
                        value=getattr(wb.session, "effort", None),
                        choices=list(self._effort_choices()),
                        command="set_effort",
                        applies="next Codex turn" if self._session_supports("live_effort") else "next session start",
                    ),
                    service_tier=dict(
                        value=(getattr(proxy, "service_tier", None) or "default"),
                        choices=list(SERVICE_TIERS),
                        command="set_service_tier",
                        probes=probes,
                        applies="next model request (also mid-turn)",
                    ),
                ),
                new_run=dict(
                    model=model.get("name"),
                    observation_profile=features.get("observation_profile"),
                    environment=backend.get("environment"),
                    robot_model_id=backend.get("robot_model_id"),
                    backend=backend,
                ),
                agent=dict(wb.agent_settings_view(), served=wb.served_view()),
            ),
            effective_config=config,
        )

    def model_transport_view(self):
        proxy = getattr(getattr(self.workbench, "session", None), "model_proxy", None)
        error = getattr(proxy, "blocked_error", None)
        if not isinstance(error, dict):
            return dict(blocked=False)
        evidence = error.get("evidence") if isinstance(error.get("evidence"), dict) else {}
        return dict(
            blocked=True,
            error_type=error.get("error_type"),
            manual_retry_allowed=bool(error.get("manual_retry_allowed")),
            **{key: evidence.get(key) for key in ("http_status", "classification", "code", "type", "message")},
        )

    def _effective_speed_scale_cap(self):
        wb = self.workbench
        if wb.operator_gate["speed_scale_cap"] is not None:
            return float(wb.operator_gate["speed_scale_cap"])
        cap = ((getattr(wb.harness, "capabilities", None) or {}).get("runtime_limits") or {}).get("speed_scale_cap")
        if type(cap) in (int, float):
            return float(cap)
        return DEFAULT_SPEED_SCALE_CAP

    def _model_options(self, fallback=None):
        wb = self.workbench
        options = self._session_model_options() or self._labelled_model_options(wb.listed_model_items()) or []
        current = getattr(wb.session, "model", None) or fallback
        if isinstance(current, str) and current and current not in {item["value"] for item in options}:
            options.append(dict(value=current, label=current))
        return options

    def _session_model_options(self):
        hook = getattr(self.workbench.session, "model_options", None)
        try:
            items = hook() if callable(hook) else None
        except Exception:  # noqa: BLE001
            return None
        return self._labelled_model_options(items)

    @staticmethod
    def _labelled_model_options(items):
        options, seen = [], set()
        for item in items if isinstance(items, (list, tuple)) else ():
            value = item.get("value") if isinstance(item, dict) else None
            if not isinstance(value, str) or not MODEL_ID.match(value) or value in seen:
                continue
            seen.add(value)
            name, image_input = item.get("name"), item.get("image_input")
            label = f"{name.strip()} ({value})" if isinstance(name, str) and name.strip() and name.strip() != value else value
            if image_input is False:
                label += NO_IMAGE_INPUT_LABEL
            options.append(
                dict(
                    value=value,
                    label=label,
                    image_input=image_input if isinstance(image_input, bool) else None,
                )
            )
        return options or None

    def _effort_choices(self):
        declared = getattr(getattr(self.workbench, "session", None), "effort_choices", None)
        if isinstance(declared, (tuple, list)) and declared:
            return tuple(declared)
        from dataclasses import fields
        from ....core.config import ModelConfig
        return next(item.metadata["choices"] for item in fields(ModelConfig) if item.name == "effort")

    def _session_supports(self, name):
        session = self.workbench.session
        capabilities = session_capabilities(session, None)
        if name == "service_tier":
            proxy = getattr(session, "model_proxy", None)
            wire = getattr(proxy, "wire", None) or getattr(session, "wire", None)
            if wire is not None and getattr(wire, "supports_service_tier", None) is not None:
                wire_ok = bool(wire.supports_service_tier)
            else:
                wire_api = getattr(session, "wire_api", None)
                if wire_api is None:
                    wire_ok = True
                else:
                    from ....cortex.agents.registry import wire_supports_service_tier
                    wire_ok = wire_supports_service_tier(wire_api)
            if not wire_ok:
                return False
            if capabilities is None:
                return True
            if getattr(capabilities, "service_tier", False):
                return True
            return getattr(session, "backend_kind", None) == "opencode" and (
                wire is not None or getattr(session, "wire_api", None) is not None
            )
        return True if capabilities is None else bool(getattr(capabilities, name, False))

    def _operator_unknown(self, command, identifier, reason):
        wb = self.workbench
        entry = dict(id="operator:" + identifier, source="operator_" + command, reason=reason)
        if entry["id"] not in {item["id"] for item in wb.operator_gate["unknown"]}:
            wb.operator_gate["unknown"].append(entry)
        wb.store.event("operator.status_unknown", dict(command=command, request_id=identifier, reason=reason))

    async def _operator(self, command, identifier, robot_lock=False, progress=None, event_origin=None, **arguments):
        wb = self.workbench
        operator = getattr(wb.robot, "operator", None)
        if not callable(operator):
            raise ValueError("operator_command_unsupported:" + command)
        extra = {} if progress is None else {"progress_callback": progress}

        def call():
            if not robot_lock:
                return operator(command, **arguments, **extra)
            with wb.harness.robot_lock:
                return operator(command, **arguments, **extra)

        try:
            result = await asyncio.to_thread(call)
        except (RuntimeError, TimeoutError, OSError) as error:
            self._operator_unknown(command, identifier, str(error))
            raise
        result = result if isinstance(result, dict) else {"value": result}
        if (
            command in ("hold", "go_home", "relax", "torque_on", "calibrate", "calib_step_start", "calib_step_cancel")
            and execution_unconfirmed(result)
        ):
            self._operator_unknown(command, identifier, result.get("reason") or "operator_confirmation_missing")
        wb.operator_gate["last"] = dict(
            command=command,
            request_id=identifier,
            status="unknown" if execution_unconfirmed(result) else result.get("status", "acknowledged"),
        )
        wb.store.event(
            "operator." + command,
            dict(
                request_id=identifier,
                arguments=arguments,
                result=result,
                **({"origin": event_origin} if event_origin else {}),
            ),
        )
        return result

    async def execute_operator(self, value, origin):
        wb = self.workbench
        command, identifier = value["command"], value["request_id"]
        if command not in LEASE_FREE_COMMANDS:
            wb.authorize_writer(value, origin)
        if command == "hold":
            wb.harness.request_stop()
            result = await self._operator("hold", identifier)
            return dict(
                request_id=identifier,
                command=command,
                status=wb.operator_gate["last"]["status"],
                admission="closed",
                operator=result,
            )
        if command == "estop":
            wb.harness.request_stop()
            try:
                hold = await self._operator("hold", identifier)
            except ValueError as error:
                hold = dict(status="unsupported", reason=str(error))
            except (RuntimeError, TimeoutError, OSError) as error:
                hold = dict(status="unknown", reason=str(error))
            stop = await wb.stop(identifier)
            return dict(
                request_id=identifier,
                command=command,
                status="unknown" if "unknown" in (stop["status"], hold.get("status")) or execution_unconfirmed(hold) else stop["status"],
                stop=stop,
                hold=hold,
                admission="closed",
            )
        if command == "relax":
            if value.get("confirm") != "relax":
                raise ValueError("relax_requires_confirm:relax")
            wb.harness.request_stop()
            ext_motion = getattr(wb.harness, "external_motion", None)
            if callable(ext_motion):
                try:
                    ext_motion("begin", "torque_off_manual_motion")
                except Exception:
                    wb.harness.accepting = False
                    wb.harness.latest = None
            try:
                result = await self._operator("relax", identifier)
            except (RuntimeError, TimeoutError, OSError):
                wb.operator_gate["torque"] = "unknown"
                raise
            wb.operator_gate["torque"] = "unknown" if execution_unconfirmed(result) else "off"
            return dict(
                request_id=identifier,
                command=command,
                status=wb.operator_gate["last"]["status"],
                torque=wb.operator_gate["torque"],
                admission="closed",
                operator=result,
            )
        if command == "torque_on":
            try:
                result = await self._operator("torque_on", identifier)
            except (RuntimeError, TimeoutError, OSError):
                wb.operator_gate["torque"] = "unknown"
                raise
            ext_motion = getattr(wb.harness, "external_motion", None)
            if callable(ext_motion):
                await asyncio.to_thread(
                    ext_motion,
                    "end",
                    "torque_off_manual_motion",
                    result=result,
                    refresh_observation=False,
                )
            wb.operator_gate["torque"] = "unknown" if execution_unconfirmed(result) else "on"
            return dict(
                request_id=identifier,
                command=command,
                status=wb.operator_gate["last"]["status"],
                torque=wb.operator_gate["torque"],
                admission="closed" if not wb.harness.accepting else "open",
                operator=result,
            )
        if command == "set_runtime_limits":
            cap = value.get("speed_scale_cap")
            low, high = SPEED_SCALE_CAP_RANGE
            if type(cap) not in (int, float) or not low <= cap <= high:
                raise ValueError("invalid_speed_scale_cap: expected %.1f..%.1f" % SPEED_SCALE_CAP_RANGE)
            result = await self._operator("set_runtime_limits", identifier, speed_scale_cap=float(cap))
            wb.operator_gate["speed_scale_cap"] = result.get("speed_scale_cap", float(cap))
            return dict(
                request_id=identifier,
                command=command,
                status="applied",
                speed_scale_cap=wb.operator_gate["speed_scale_cap"],
                operator=result,
            )
        if command == "set_service_tier":
            return await self.apply_service_tier(identifier, value.get("tier"))
        if command == "set_model":
            return await self.apply_model(identifier, value.get("model"))
        if command == "apply_agent_settings":
            return await wb.apply_agent_settings(identifier, value)
        if command == "handoff_run":
            return await wb.handoff_run(identifier, value)
        if command == "retry_model_requests":
            retry = getattr(getattr(wb.session, "model_proxy", None), "manual_retry", None)
            if not callable(retry):
                raise ValueError("model_proxy_unavailable")
            previous = self.model_transport_view()
            if not previous["blocked"]:
                raise ValueError("model_transport_not_blocked")
            await asyncio.to_thread(retry)
            wb.store.event(
                "operator.model_retry_authorized",
                dict(request_id=identifier, previous=previous, applies="next_model_request"),
            )
            return dict(
                request_id=identifier,
                command=command,
                status="retry_authorized",
                previous=previous,
                applies="next_model_request",
            )
        if command == "set_effort":
            return await self.apply_effort(identifier, value.get("effort"))
        if command == "save_video":
            return await self._save_video(identifier)
        if command in CALIBRATION_COMMANDS:
            return await wb.calibration_workflow_command(command, identifier, value)
        if command in HANDHELD_COMMANDS:
            return await self._handheld_command(command, identifier, value)
        if command in IDLE_ROBOT_COMMANDS:
            if (
                wb.starting
                or session_turn_blocks_operator(wb.session)
                or wb.harness.active_action
                or wb.has_active_programs()
            ):
                raise ValueError(command + "_requires_idle_turn_and_motion")
            if self.unknown_items():
                raise ValueError(command + "_requires_unknown_acknowledged_via_resume")
            if wb.operator_gate["torque"] == "off":
                raise ValueError(command + "_requires_torque_on")
        if command in ("calibrate", "reload_calibration"):
            return await self._calibration_command(command, identifier)
        if command == "go_home":
            wb.harness.request_stop()
            ext_motion = getattr(wb.harness, "external_motion", None)
            if callable(ext_motion):
                await asyncio.to_thread(ext_motion, "begin", "operator_go_home")
            try:
                result = await self._operator("go_home", identifier, robot_lock=True)
            except BaseException:
                if callable(ext_motion):
                    await asyncio.to_thread(
                        ext_motion,
                        "end",
                        "operator_go_home",
                        result=dict(status="unknown"),
                        refresh_observation=False,
                    )
                raise
            unconfirmed = execution_unconfirmed(result)
            if callable(ext_motion):
                await asyncio.to_thread(
                    ext_motion,
                    "end",
                    "operator_go_home",
                    result=result,
                    refresh_observation=not unconfirmed,
                )
            wb.status = "stopped"
            return dict(
                request_id=identifier,
                command=command,
                status=wb.operator_gate["last"]["status"],
                world="objects_not_reset",
                admission="closed_until_resume",
                operator=result,
            )
        if command == "resume":
            programs = getattr(wb, "programs", None) or getattr(wb.harness, "programs", None)
            reconcile_report = None
            if programs is not None and callable(getattr(programs, "reconcile", None)):
                reconcile_report = await asyncio.to_thread(programs.reconcile)
            blockers = self.resume_blockers()
            if blockers:
                raise ValueError("resume_blocked:" + ",".join(blockers))
            required = sorted(item["id"] for item in self.unknown_items())
            acknowledged = value.get("acknowledge_unknown", [])
            reconciled_prog_ids = {
                f"program:{jid}" for jid in (reconcile_report or {}).get("reconciled", ())
            }
            valid_ack = (
                isinstance(acknowledged, list)
                and all(isinstance(x, str) for x in acknowledged)
                and len(acknowledged) == len(set(acknowledged))
                and set(required).issubset(acknowledged)
                and set(acknowledged).issubset(set(required) | reconciled_prog_ids)
            )
            if not valid_ack:
                raise ValueError(
                    "resume_requires_unknown_acknowledgement:" + ",".join(required)
                    if required
                    else "resume_acknowledgement_mismatch"
                )
            recorded_ack = sorted(set(required) | (set(acknowledged) & reconciled_prog_ids))
            if recorded_ack:
                wb.store.event(
                    "operator.unknown_acknowledged",
                    dict(
                        request_id=identifier,
                        acknowledged=recorded_ack,
                        note="operator inspected the arm; receipts remain unknown in the archive",
                    ),
                )
            observation = await asyncio.to_thread(wb.harness.reopen_admission, {"operator_request_id": identifier})
            wb.operator_gate["unknown"] = []
            non_operator_ack = [item for item in recorded_ack if not item.startswith("operator:")]
            wb.operator_gate["acknowledged_actions"] += non_operator_ack
            op_ack_set = getattr(wb.harness, "_operator_acknowledged_actions", None)
            if isinstance(op_ack_set, set):
                op_ack_set.update(item for item in non_operator_ack if not item.startswith("program:"))
            op_prog_set = getattr(wb.harness, "_operator_acknowledged_programs", None)
            if isinstance(op_prog_set, set):
                op_prog_set.update(item for item in non_operator_ack if item.startswith("program:"))
            wb.status = "running"
            return dict(
                request_id=identifier,
                command=command,
                status="resumed",
                admission="open",
                observation_id=observation["observation_id"],
                acknowledged=recorded_ack,
            )
        raise ValueError("unsupported_control_command")

    async def apply_service_tier(self, identifier, tier):
        wb = self.workbench
        command = "set_service_tier"
        if tier not in SERVICE_TIERS:
            raise ValueError("invalid_service_tier")
        if not self._session_supports("service_tier"):
            backend = getattr(wb.session, "backend_kind", "unknown")
            return dict(
                request_id=identifier,
                command=command,
                status="unsupported",
                tier=tier,
                reason="service_tier_unsupported_by_backend",
                message=f"当前 agent 后端（{backend}）不支持服务档位设置",
            )
        proxy = getattr(wb.session, "model_proxy", None)
        if proxy is None:
            raise ValueError("model_proxy_unavailable")
        probe = None
        if tier != "default":
            model = getattr(wb.session, "model", None)
            probe = await asyncio.to_thread(proxy.probe_service_tier, model, tier)
            if probe.get("error"):
                message = (
                    f"供应商当前不提供模型 {model}（{probe['error'].split(':')[0]}），与档位无关；请先切换到可用模型再选 {tier}"
                    if (probe.get("provider_error") or {}).get("code") == "model_not_found"
                    else f"无法确认供应商是否支持 {tier}（模型 {model}）：{probe['error']}"
                )
                return dict(
                    request_id=identifier,
                    command=command,
                    status="unknown",
                    tier=tier,
                    model=model,
                    probe=probe,
                    message=message,
                )
            if not probe["supported"]:
                return dict(
                    request_id=identifier,
                    command=command,
                    status="unsupported",
                    tier=tier,
                    probe=probe,
                    current=proxy.service_tier or "default",
                    message=f"当前供应商不支持 {tier}（{probe.get('rejection') or '请求被拒绝'}）",
                )
        previous, proxy.service_tier = proxy.service_tier, None if tier == "default" else tier
        wb.store.event(
            "operator.service_tier_changed",
            dict(
                request_id=identifier,
                previous=previous or "default",
                tier=tier,
                probe=probe,
                applies="next_model_request",
                **wb.agent_event_context(),
            ),
        )
        wb.agent_setting_changed("service_tier", previous or "default", tier, identifier, "next_model_request")
        served = (probe or {}).get("served_tier")
        receipt = dict(
            request_id=identifier,
            command=command,
            status="applied",
            tier=tier,
            probe=probe,
            served_tier=served,
            applies="next_model_request",
        )
        if probe is not None and served != tier:
            receipt["message"] = (
                f"供应商接受了 {tier}，但探测请求实际以 {served or '未报告的'} 档位服务"
                "（以每次响应为准）"
            )
        return receipt

    async def apply_model(self, identifier, selected):
        wb = self.workbench
        command = "set_model"
        if not isinstance(selected, str) or not MODEL_ID.match(selected):
            raise ValueError("invalid_model")
        if getattr(wb.session, "price_card", None) is not None and selected != getattr(wb.session, "model", None):
            raise ValueError("set_model_conflicts_with_price_card")
        previous = getattr(wb.session, "model", None)
        setter = getattr(wb.session, "set_model", None)
        if callable(setter):
            await asyncio.to_thread(setter, selected)
        else:
            wb.session.model = selected
        proxy = getattr(wb.session, "model_proxy", None)
        if self._session_supports("service_tier") and callable(getattr(proxy, "select_model", None)):
            proxy.select_model(selected)
        wb.store.event(
            "operator.model_changed",
            dict(
                request_id=identifier,
                previous=previous,
                model=selected,
                applies="next_agent_turn",
                **wb.agent_event_context(),
            ),
        )
        wb.agent_setting_changed("model", previous, selected, identifier, "next_agent_turn")
        return dict(
            request_id=identifier,
            command=command,
            status="applied",
            model=selected,
            service_tier=(getattr(proxy, "service_tier", None) or "default"),
            applies="next_agent_turn",
        )

    async def apply_effort(self, identifier, effort):
        wb = self.workbench
        command = "set_effort"
        if effort not in self._effort_choices():
            raise ValueError("invalid_effort")
        previous = getattr(wb.session, "effort", None)
        setter = getattr(wb.session, "set_effort", None)
        if callable(setter):
            setter(effort)
        else:
            wb.session.effort = effort
        applies = "next_codex_turn" if self._session_supports("live_effort") else "next_session_start"
        wb.store.event(
            "operator.effort_changed",
            dict(
                request_id=identifier,
                previous=previous,
                effort=effort,
                applies=applies,
                **wb.agent_event_context(),
            ),
        )
        wb.agent_setting_changed("effort", previous, effort, identifier, applies)
        return dict(request_id=identifier, command=command, status="applied", effort=effort, applies=applies)

    async def _reopen_after_operator(self, identifier, was_accepting, stop_revision, unconfirmed):
        wb = self.workbench
        admission, observation_id = "closed_until_resume", None
        if (
            was_accepting
            and not unconfirmed
            and wb.harness.stop_revision == stop_revision
            and not self.resume_blockers()
            and not self.unknown_items()
        ):
            try:
                observation = await asyncio.to_thread(wb.harness.reopen_admission, {"operator_request_id": identifier})
                admission, observation_id = "open", observation["observation_id"]
            except ValueError as error:
                admission = "closed_until_resume:" + str(error)
        wb.status = "running" if admission == "open" else "stopped"
        return admission, observation_id

    def _seal_recording(self):
        wb = self.workbench
        with wb.harness.robot_lock:
            wb.require_observer_idle()
            export = record_video(wb.store, wb.identity, wb.robot.save_video())
            try:
                rearm = wb.robot.record(enabled=wb.config.features.record_video)
            except (RuntimeError, TimeoutError, OSError) as error:
                rearm = dict(enabled=False, error=f"{type(error).__name__}: {error}")
        return export, rearm

    def _register_composite(self, composite):
        wb = self.workbench
        composite["video_ref"] = wb.store.put_artifact(
            Path(composite["video_path"]).read_bytes(),
            "native_video",
            "video/mp4",
            "observer",
        )

    async def _save_video(self, identifier):
        wb = self.workbench
        has_video = isinstance(wb.robot, VideoBackend) or callable(getattr(wb.robot, "save_video", None))
        if not (wb.harness.capabilities or {}).get("observer_video") or not has_video:
            raise ValueError("observer_video_unavailable")
        if not wb.config.features.record_video:
            raise ValueError("record_video_disabled")
        try:
            export, rearm = await asyncio.to_thread(self._seal_recording)
        except (RuntimeError, TimeoutError, OSError) as error:
            wb.video_export_state = dict(
                request_id=identifier,
                saved_at=_utc_now(),
                status="failed",
                error=f"seal_failed: {type(error).__name__}: {error}",
            )
            wb.store.event("operator.save_video", dict(request_id=identifier, result=wb.video_export_state))
            return dict(wb.video_export_state, command="save_video")
        try:
            config_dict = wb.config.to_dict() if hasattr(wb.config, "to_dict") else None
            composite = await asyncio.to_thread(
                _compose_motion_video,
                wb.robot,
                export,
                run_dir=wb.root,
                config_dict=config_dict,
            )
            if composite.get("video_path"):
                await asyncio.to_thread(self._register_composite, composite)
                wb.store.event("observer.video_composed", {**wb.identity, **composite}, source="private_evaluator")
        except (RuntimeError, TimeoutError, OSError, ValueError) as error:
            composite = dict(status="failed", error=f"compose_failed: {type(error).__name__}: {error}")
        video_json = wb.store.root / "video.json"
        if export.get("segments") or not video_json.exists():
            try:
                video_json.write_text(json.dumps(dict(export, composite=composite), indent=2))
            except OSError as error:
                composite = dict(composite, video_json_error=f"{type(error).__name__}: {error}")
        mappings = [camera.get("frame_mapping_path") for camera in export.get("cameras", []) if camera.get("frame_mapping_path")]
        recording = dict(
            directory=str(Path(mappings[0]).parent) if mappings else None,
            segments=len(export.get("segments", [])),
            error=export.get("error"),
            cameras=[
                {key: camera.get(key) for key in ("camera_id", "status", "frames", "video_ref")}
                for camera in export.get("cameras", [])
            ],
        )
        wb.video_export_state = dict(
            request_id=identifier,
            saved_at=_utc_now(),
            status=composite["status"],
            error=composite.get("error"),
            composite=composite,
            recording=recording,
            recording_rearmed=bool(rearm.get("enabled")),
            rearm_error=rearm.get("error"),
        )
        wb.store.event("operator.save_video", dict(request_id=identifier, result=wb.video_export_state))
        return dict(wb.video_export_state, command="save_video")

    async def _calibration_command(self, command, identifier):
        wb = self.workbench
        was_accepting = wb.harness.accepting
        wb.harness.request_stop()
        stop_revision = wb.harness.stop_revision
        source = "operator_" + command
        ext_motion = getattr(wb.harness, "external_motion", None)
        progress = None
        if command == "calibrate":
            wb.calibration_state.update(
                running=True,
                request_id=identifier,
                started_at=_utc_now(),
                finished_at=None,
                progress=[],
                result=None,
            )

            def progress(update):
                if isinstance(update, dict):
                    entries = wb.calibration_state["progress"] + [dict(update, at=_utc_now())]
                    wb.calibration_state["progress"] = entries[-CALIBRATION_PROGRESS_KEEP:]

        try:
            if callable(ext_motion):
                await asyncio.to_thread(ext_motion, "begin", source)
            result = await self._operator(command, identifier, robot_lock=True, progress=progress)
        except BaseException as error:
            if command == "calibrate":
                wb.calibration_state.update(
                    running=False,
                    finished_at=_utc_now(),
                    result=dict(status="unknown", reason=str(error)),
                )
            if callable(ext_motion):
                await asyncio.to_thread(
                    ext_motion,
                    "end",
                    source,
                    result=dict(status="unknown", reason=str(error)),
                    refresh_observation=False,
                )
            raise
        unconfirmed = execution_unconfirmed(result)
        if command == "calibrate":
            wb.calibration_state.update(running=False, finished_at=_utc_now(), result=result)
        will_reopen = (
            was_accepting
            and not unconfirmed
            and wb.harness.stop_revision == stop_revision
            and not self.resume_blockers()
            and not self.unknown_items()
        )
        if callable(ext_motion):
            await asyncio.to_thread(
                ext_motion,
                "end",
                source,
                result=result,
                refresh_observation=(not will_reopen and not unconfirmed),
            )
        admission, observation_id = await self._reopen_after_operator(identifier, was_accepting, stop_revision, unconfirmed)
        status = "unknown" if unconfirmed else result.get("status", "acknowledged")
        return dict(
            request_id=identifier,
            command=command,
            status=status,
            admission=admission,
            observation_id=observation_id,
            operator=result,
            **({"world": "objects_not_reset"} if command == "calibrate" and result.get("physics_advanced") else {}),
        )

    def handheld_view(self):
        path = self.telemetry_directory() / HANDHELD_STATUS_FILE
        capture = None
        try:
            if not path.is_symlink() and path.is_file():
                capture = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            capture = None
        if not isinstance(capture, dict):
            capture = None
        elif capture.get("status") == "running":
            age = None
            try:
                age = datetime.now(timezone.utc).timestamp() - path.stat().st_mtime
            except OSError:
                pass
            capture["stale"] = age is None or age > max(TELEMETRY_STALE_AFTER_S, 5.0)
        wb = self.workbench
        return dict(
            capture=capture,
            solving=wb.handheld_state["solving"],
            solve_request_id=wb.handheld_state["solve_request_id"],
            solve_result=wb.handheld_state["solve_result"],
        )

    async def _handheld_command(self, command, identifier, value):
        wb = self.workbench
        if (
            wb.starting
            or session_turn_blocks_operator(wb.session)
            or wb.harness.active_action
            or wb.has_active_programs()
        ):
            raise ValueError(command + "_requires_idle_turn_and_motion")
        if wb.handheld_state["solving"]:
            raise ValueError(command + "_refused_while_handheld_solve_running")
        if command == "handheld_capture":
            action = value.get("action")
            if action not in HANDHELD_CAPTURE_ACTIONS:
                raise ValueError("invalid_handheld_capture_action: expected start or stop")
            result = await self._operator(command, identifier, action=action)
            return dict(
                request_id=identifier,
                command=command,
                action=action,
                status=result.get("status", "acknowledged"),
                operator=result,
            )
        wb.handheld_state.update(solving=True, solve_request_id=identifier, solve_result=None)
        try:
            response = await self._calibration_command(command, identifier)
        except BaseException as error:
            wb.handheld_state.update(solving=False, solve_result=dict(status="unknown", reason=str(error)))
            raise
        wb.handheld_state.update(solving=False, solve_result=response["operator"])
        return response

    async def board_check(self):
        operator = getattr(self.workbench.robot, "operator", None)
        if not callable(operator):
            return dict(available=False, reason="operator_command_unsupported:board_check")
        try:
            result = await asyncio.to_thread(operator, "board_check")
        except (ValueError, RuntimeError, TimeoutError, OSError) as error:
            return dict(available=False, reason=str(error))
        return dict(result if isinstance(result, dict) else {"value": result}, available=True)

    def mount(self, app):
        async def telemetry(request):
            view = await asyncio.to_thread(telemetry_view, self.telemetry_directory())
            view["joint_limits_rad"] = self.joint_limits()
            return web.json_response(view, headers={"Cache-Control": "no-store"})

        async def camera(request):
            camera_id = request.match_info["camera_id"]
            if not CAMERA_ID.match(camera_id):
                raise web.HTTPNotFound(text="unknown_camera")
            path = self.telemetry_directory() / f"{camera_id}.jpg"
            if path.is_symlink() or not path.is_file():
                raise web.HTTPNotFound(text="camera_frame_unavailable")
            body = await asyncio.to_thread(path.read_bytes)
            return web.Response(body=body, content_type="image/jpeg", headers={"Cache-Control": "no-store"})

        async def fixed_corners(request):
            path = self.telemetry_directory() / HANDHELD_PREVIEW_FILE
            if path.is_symlink() or not path.is_file():
                raise web.HTTPNotFound(text="fixed_corner_preview_unavailable")
            body = await asyncio.to_thread(path.read_bytes)
            return web.Response(body=body, content_type="image/jpeg", headers={"Cache-Control": "no-store"})

        async def state(request):
            return web.json_response(self.console_state(), headers={"Cache-Control": "no-store"})

        async def board_check(request):
            return web.json_response(await self.board_check(), headers={"Cache-Control": "no-store"})

        async def page(request):
            path = static_directory() / "console.html"
            if not path.is_file():
                raise web.HTTPNotFound(text="console_not_built: run npm --prefix web run build")
            return web.FileResponse(path)

        app.router.add_get("/v1/console/telemetry", telemetry)
        app.router.add_get("/v1/console/cameras/fixed_corners.jpg", fixed_corners)
        app.router.add_get("/v1/console/cameras/{camera_id}.jpg", camera)
        app.router.add_get("/v1/console/state", state)
        app.router.add_get("/v1/console/board_check", board_check)
        app.router.add_get("/console", page)


__all__ = [
    "CALIBRATION_PROGRESS_KEEP",
    "CAMERA_ID",
    "DEFAULT_SPEED_SCALE_CAP",
    "HANDHELD_CAPTURE_ACTIONS",
    "HANDHELD_COMMANDS",
    "HANDHELD_PREVIEW_FILE",
    "HANDHELD_STATUS_FILE",
    "IDLE_ROBOT_COMMANDS",
    "LEASE_BUSY_CODE",
    "LEASE_BUSY_REFUSAL",
    "LEASE_COMMANDS",
    "LEASE_FREE_COMMANDS",
    "MODEL_ID",
    "NO_IMAGE_INPUT_LABEL",
    "OPERATOR_COMMANDS",
    "OperatorConsoleExtension",
    "PREEMPTIVE_COMMANDS",
    "RECEIPT_FIELDS",
    "SECRET_KEY",
    "SERVICE_TIERS",
    "SPEED_SCALE_CAP_RANGE",
    "TELEMETRY_SCHEMA",
    "TELEMETRY_STALE_AFTER_S",
    "_parse_utc",
    "_receipt_projection",
    "_utc_now",
    "redact",
    "telemetry_view",
]
