"""K-arm, N-DoF command-row layout and backend-neutral batch scheduler port (``batch_scheduler.py``).

Replaces hardcoded ``offset = 0 if arm == "left" else 7`` and ``[:6]`` slices across motion planning,
cached-row inspection, and contact-stop reanchoring with explicit ``ArmSlot`` and ``ArmLayout`` specs
derived from ``EmbodimentProfile`` (supporting 1-arm 5-DoF SO-101, 2-arm 6-DoF ARX-X5, 1/2-arm 7-DoF
arms, and arbitrary multi-arm layouts).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ...core.embodiment import EmbodimentProfile
from .grip_guard import (
    CARRY_SLIP_REASON,
    CLOSING_JAW_GOAL_BELOW,
    CarrySlipMonitor,
    after_stage_grip_guard,
    format_effective_grip_guard,
    jaw_checkpoints,
)


@dataclass(frozen=True)
class ArmSlot:
    """Contiguous joint and optional jaw index assignment for one arm inside a flat native command row.

    Attributes:
        arm: Canonical arm identifier (e.g., ``'left'``, ``'right'``).
        joints: Contiguous ``slice(start, stop)`` of ``N`` arm joint angles (``rad``) within a row.
        jaw: Optional column index of the arm's scalar jaw/gripper command within a row (``None`` if no jaw).
        tip_link: End-effector / TCP link name (e.g., ``'link6'``, ``'gripper_frame_link'``).
    """

    arm: str
    joints: slice
    jaw: int | None
    tip_link: str = "link6"

    def __post_init__(self) -> None:
        start = 0 if self.joints.start is None else int(self.joints.start)
        stop = int(self.joints.stop)
        step = 1 if self.joints.step is None else int(self.joints.step)
        if step != 1 or start < 0 or stop <= start:
            raise ValueError(f"invalid_arm_slot_slice:{self.arm}:{self.joints!r}")
        if self.jaw is not None and (int(self.jaw) < 0 or start <= int(self.jaw) < stop):
            raise ValueError(f"invalid_arm_slot_jaw_index:{self.arm}:{self.jaw}")

    @property
    def offset(self) -> int:
        """Starting column index of ``joints`` in the flat command row."""
        return 0 if self.joints.start is None else int(self.joints.start)

    @property
    def dof(self) -> int:
        """Number of active arm joints ``N`` in this slot."""
        return int(self.joints.stop) - self.offset


@dataclass(frozen=True)
class ArmLayout:
    """Complete ``K``-arm, ``N_k``-DoF row layout mapping arm names to ``ArmSlot`` descriptors."""

    slots: Mapping[str, ArmSlot]
    row_width: int

    def __post_init__(self) -> None:
        if not self.slots or self.row_width < 1:
            raise ValueError("invalid_arm_layout")
        occupied: set[int] = set()
        for arm, slot in self.slots.items():
            if slot.arm != arm:
                raise ValueError(f"arm_slot_key_mismatch:{arm}!={slot.arm}")
            cols = set(range(slot.offset, slot.offset + slot.dof))
            if slot.jaw is not None:
                cols.add(int(slot.jaw))
            if max(cols) >= self.row_width or (occupied & cols):
                raise ValueError(f"overlapping_or_oob_arm_slot:{arm}")
            occupied |= cols

    @property
    def arms(self) -> tuple[str, ...]:
        """Ordered tuple of arm identifiers in this layout."""
        return tuple(self.slots.keys())

    def slot(self, arm: str) -> ArmSlot:
        """Return ``ArmSlot`` for ``arm``, raising ``ValueError`` on unknown arm names."""
        try:
            return self.slots[arm]
        except KeyError:
            raise ValueError(f"invalid_arm_name:{arm}") from None

    def offset(self, arm: str) -> int:
        """Return starting joint column offset for ``arm``."""
        return self.slot(arm).offset

    def dof(self, arm: str) -> int:
        """Return joint count ``N`` for ``arm``."""
        return self.slot(arm).dof

    def joint_slice(self, arm: str) -> slice:
        """Return ``slice(offset, offset + dof)`` for ``arm``."""
        return self.slot(arm).joints

    def jaw_index(self, arm: str) -> int | None:
        """Return jaw column index for ``arm`` (or ``None``)."""
        return self.slot(arm).jaw

    def extract_arm_q(self, row: Sequence[float], arm: str) -> list[float]:
        """Extract ``arm``'s ``N`` joint values (``rad``) from a flat command row."""
        sl = self.slot(arm)
        return [float(v) for v in row[sl.joints]]

    def write_arm_q(self, row: list[float], arm: str, q_rad: Sequence[float]) -> None:
        """Write ``q_rad`` (length ``N``, ``rad``) into ``arm``'s slice of ``row`` in-place."""
        sl = self.slot(arm)
        vals = [float(v) for v in q_rad]
        if len(vals) != sl.dof:
            raise ValueError(f"arm_q_length_mismatch:expected_{sl.dof}_got_{len(vals)}")
        row[sl.joints] = vals

    @classmethod
    def x5_default(cls) -> ArmLayout:
        """Default dual-arm 6-DoF + 1-jaw ARX-X5 layout (14 columns: ``left[0:6]+jaw[6]``, ``right[7:13]+jaw[13]``)."""
        return cls(
            slots={
                "left": ArmSlot("left", slice(0, 6), 6, tip_link="link6"),
                "right": ArmSlot("right", slice(7, 13), 13, tip_link="link6"),
            },
            row_width=14,
        )

    @classmethod
    def from_specs(
        cls,
        arm_dofs: Mapping[str, int],
        *,
        with_jaw: bool = True,
        tip_links: Mapping[str, str] | None = None,
    ) -> ArmLayout:
        """Construct an ``ArmLayout`` sequentially from ``{arm_name: dof}``."""
        slots: dict[str, ArmSlot] = {}
        cursor = 0
        for arm, dof in arm_dofs.items():
            n = int(dof)
            if n < 1:
                raise ValueError(f"invalid_arm_dof:{arm}:{dof}")
            j_slice = slice(cursor, cursor + n)
            cursor += n
            jaw_col = cursor if with_jaw else None
            if with_jaw:
                cursor += 1
            tip = (tip_links or {}).get(arm, "link6")
            slots[str(arm)] = ArmSlot(str(arm), j_slice, jaw_col, tip_link=tip)
        return cls(slots=slots, row_width=cursor)

    @classmethod
    def from_profile(cls, profile: EmbodimentProfile | None) -> ArmLayout:
        """Derive ``ArmLayout`` from an ``EmbodimentProfile`` (falling back to ``x5_default()`` if ``None``)."""
        if profile is None:
            return cls.x5_default()
        groups = getattr(profile, "kinematic_groups", None) or {}
        end_effectors = getattr(profile, "end_effectors", None) or {}
        arm_joints_map = getattr(profile, "arm_joint_names", None) or {}
        jaw_joints_map = getattr(profile, "jaw_joint_names", None) or {}
        slots: dict[str, ArmSlot] = {}
        cursor = 0
        for arm in profile.arms:
            group = groups.get(arm)
            arm_prof = (profile.arm_profiles or {}).get(arm)
            if group is not None and getattr(group, "dof", 0) > 0:
                dof = int(group.dof)
                tip_link = str(group.tip_link or getattr(profile, "tcp_link", "link6"))
            elif arm in arm_joints_map and arm_joints_map[arm]:
                dof = len(arm_joints_map[arm])
                tip_link = str(getattr(arm_prof, "tcp_link", None) or getattr(profile, "tcp_link", "link6"))
            elif arm_prof is not None and getattr(arm_prof, "joint_names", None):
                dof = len(arm_prof.joint_names)
                tip_link = str(getattr(arm_prof, "tcp_link", None) or getattr(profile, "tcp_link", "link6"))
            else:
                dof = 6
                tip_link = str(getattr(profile, "tcp_link", "link6"))
            has_jaw = True
            if end_effectors and arm in end_effectors:
                ee = end_effectors[arm]
                has_jaw = getattr(ee, "kind", "none") != "none"
            elif arm in jaw_joints_map:
                has_jaw = bool(jaw_joints_map[arm])
            j_slice = slice(cursor, cursor + dof)
            cursor += dof
            jaw_col = cursor if has_jaw else None
            if has_jaw:
                cursor += 1
            slots[arm] = ArmSlot(arm, j_slice, jaw_col, tip_link=tip_link)
        return cls(slots=slots, row_width=cursor)


@dataclass(frozen=True)
class StageScheduleItem:
    """Normalized stage descriptor within a scheduled ``BatchAction``.

    Attributes:
        id: Unique stage identifier within the batch.
        kind: Stage kind (``'move'``, ``'gripper'``, or ``'observe'``).
        arm: Owning arm identifier (or ``None`` for arm-neutral ``observe`` stages).
        depends_on: Explicit prerequisite stage IDs.
        start: Planned starting native step cursor (``>= 0``).
        end: Planned ending native step cursor (``>= start``).
        stage: Optional raw stage dictionary.
    """

    id: str
    kind: str
    arm: str | None = None
    depends_on: tuple[str, ...] = ()
    start: int = 0
    end: int = 0
    stage: Mapping[str, Any] | None = None


def order_stages(stages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return a topological ordering of ``stages`` preserving request order among ready stages."""
    order: list[dict[str, Any]] = []
    done: set[str] = set()
    pending = [dict(stage) for stage in stages]
    while pending:
        for idx, stage in enumerate(pending):
            if set(stage.get("depends_on", ())) <= done:
                order.append(stage)
                done.add(stage["id"])
                pending.pop(idx)
                break
        else:
            raise ValueError("batch_dependency_cycle_or_unknown")
    return order


def jaw_pauses(
    action: Mapping[str, Any],
    stages: Sequence[Mapping[str, Any]],
    records: Mapping[str, Mapping[str, Any]],
    costs: Mapping[str, Mapping[str, Any]],
    checkpoints: list[dict[str, Any]],
    previous_goals: Mapping[str, float],
) -> dict[str, dict[str, Any]]:
    """Compute ``checkpoint_after_gripper`` stage-boundary pauses and update ``checkpoints`` sorted by cursor.

    Returns ``{dependent_move_stage_id: {"stage_ids": [...], "reasons": {...}}}``.
    """
    windows: dict[str, dict[str, Any]] = {}
    dependencies: dict[str, set[str]] = {}
    last_by_arm: dict[str | None, str] = {}
    for stage in stages:
        identifier = stage["id"]
        arm = stage.get("arm")
        prev = last_by_arm.get(arm)
        dependencies[identifier] = set(stage.get("depends_on", ())) | ({prev} if prev else set())
        last_by_arm[arm] = identifier
        windows[identifier] = {
            "kind": stage["kind"],
            "arm": arm,
            "start": costs[identifier]["start_native_step"],
            "end": costs[identifier]["end_native_step"],
            "action": stage.get("action", {}),
            "checkpoint_after_gripper": stage.get("checkpoint_after_gripper"),
            "previous_goal": previous_goals[identifier],
        }
    markers = [
        {"cursor": item["cursor"], "stage_ids": [item["stage_id"]], "reasons": {item["stage_id"]: item["reason"]}}
        for item in checkpoints
    ]
    observed = {item["stage_id"] for item in checkpoints}
    jaw_checkpoints(
        windows,
        list(stages),
        dependencies,
        markers,
        bool(action.get("checkpoint_after_gripper", False)),
        lambda record: record["previous_goal"],
    )
    pauses: dict[str, dict[str, Any]] = {}
    for marker in markers:
        jaw_ids = [identifier for identifier in marker["stage_ids"] if identifier not in observed]
        if not jaw_ids:
            continue
        move_id = next(
            (
                stage["id"]
                for stage in stages
                if stage["kind"] == "move" and windows[stage["id"]]["start"] == marker["cursor"]
            ),
            None,
        )
        if move_id is None:
            continue
        pauses[move_id] = {
            "stage_ids": jaw_ids,
            "reasons": {identifier: marker["reasons"][identifier] for identifier in jaw_ids},
        }
        checkpoints.extend(
            {
                "stage_id": identifier,
                "cursor": marker["cursor"],
                "reason": marker["reasons"][identifier],
                "pause_before_stage_id": move_id,
            }
            for identifier in jaw_ids
        )
    checkpoints.sort(key=lambda item: item["cursor"])
    return pauses


def pause_receipt(
    stage_ids: Sequence[str],
    reasons: Mapping[str, str],
    stop_confirmation: Mapping[str, Any] | None,
    *,
    checkpoint_reason: str | None = None,
    checkpoint_reason_code: str | None = None,
    checkpoint_stage_id: str | None = None,
    effective_grip_guard: Mapping[str, Any] | None = None,
    slip_detected_at: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a canonical checkpoint-pause receipt update dictionary across all backends.

    ``checkpoint_reason_code`` is always populated with a typed machine code:
    ``"observation_checkpoint"``, ``"jaw_checkpoint"``, or the guard's typed code
    (e.g., ``"aperture_below_min_m"``, ``"aperture_above_max_m"``, ``"jaw_not_settled"``,
    ``"carry_slip_aperture_below_min_m"``).
    """
    code = checkpoint_reason_code or checkpoint_reason or "observation_checkpoint"
    payload: dict[str, Any] = {
        "status": "stopped",
        "reason": "observation_checkpoint",
        "paused_at_checkpoint": True,
        "checkpoint_stage_ids": list(stage_ids),
        "checkpoint_reasons": dict(reasons),
        "checkpoint_reason_code": code,
        "stop_confirmation": stop_confirmation,
    }
    if checkpoint_stage_id is not None:
        payload["checkpoint_stage_id"] = checkpoint_stage_id
    if checkpoint_reason is not None:
        payload["checkpoint_reason"] = checkpoint_reason
    if effective_grip_guard is not None:
        payload["effective_grip_guard"] = deepcopy(dict(effective_grip_guard))
    if slip_detected_at is not None:
        payload["slip_detected_at"] = deepcopy(dict(slip_detected_at))
    return payload


def skip_dependents(
    stage_id: str,
    stages: Sequence[Mapping[str, Any]],
    dependencies: Mapping[str, set[str]] | None = None,
    *,
    stage_results: dict[str, dict[str, Any]] | None = None,
    status: str = "pending",
) -> list[str]:
    """Return ordered stage IDs transitively depending on ``stage_id`` (via ``depends_on`` or same-arm order).

    When ``stage_results`` is provided, unexecuted dependent entries still marked ``'pending'`` are set to ``status``
    (default ``'pending'``, preserving historical stage_results receipt invariants while skipping execution).
    """
    if dependencies is None:
        computed: dict[str, set[str]] = {}
        last_by_arm: dict[Any, str] = {}
        known_arms = [s.get("arm") for s in stages if s.get("arm") is not None]
        arms = tuple(dict.fromkeys(known_arms))
        for st in stages:
            sid = st["id"]
            arm = st.get("arm")
            deps = set(st.get("depends_on", ()))
            owners = arms if arm is None and arms else (arm,)
            for side in owners:
                if side in last_by_arm:
                    deps.add(last_by_arm[side])
                last_by_arm[side] = sid
            computed[sid] = deps
        dependencies = computed
    blocked: set[str] = {stage_id}
    changed = True
    while changed:
        changed = False
        for st in stages:
            sid = st["id"]
            if sid not in blocked and (dependencies.get(sid, set()) & blocked):
                blocked.add(sid)
                changed = True
    skipped = [st["id"] for st in stages if st["id"] != stage_id and st["id"] in blocked]
    if stage_results is not None:
        for sid in skipped:
            if sid in stage_results and stage_results[sid].get("status") == "pending":
                stage_results[sid]["status"] = status
    return skipped


def summarize_batch(
    plan_id: str,
    plan: dict[str, Any],
    stage_order: Sequence[str],
    stage_results: dict[str, dict[str, Any]],
    receipt: dict[str, Any],
    *,
    complete_batch_native_steps: int | None = None,
) -> dict[str, Any]:
    """Finalize batch plan status, stage running->paused/stopped transitions, and receipt summary fields."""
    if plan.get("status") == "executing":
        plan["status"] = "failed"
    completed_set = set(plan.get("completed") or ())
    completed_ids = [sid for sid in stage_order if sid in completed_set]
    pending_ids = [sid for sid in stage_order if sid not in completed_set]
    plan["executable"] = plan.get("status") in ("planned", "awaiting_observation")
    plan["completed_stage_ids"] = completed_ids
    plan["pending_stage_ids"] = pending_ids
    plan["failed_stage_id"] = receipt.get("failed_stage_id")
    for st_res in stage_results.values():
        if st_res.get("status") == "running":
            if st_res.get("execution_uncertain"):
                st_res["status"] = "unknown"
            elif plan.get("status") == "awaiting_observation":
                st_res["status"] = "paused"
            else:
                st_res["status"] = "stopped"
    summary: dict[str, Any] = {
        "batch_plan_id": plan_id,
        "batch_status": plan["status"],
        "batch_cursor": plan.get("cursor", 0),
        "batch_revision": plan.get("revision", 0),
        "completed_stage_ids": completed_ids,
        "pending_stage_ids": pending_ids,
    }
    if complete_batch_native_steps is not None:
        summary["complete_batch_native_steps"] = int(complete_batch_native_steps)
    receipt.update(summary)
    return summary


class _MotionControllerPortMeta(type(Protocol)):
    """Runtime protocol check accepting both full ``MotionControllerPort`` adapters and minimal ``state+capabilities`` stubs."""

    def __instancecheck__(cls, instance: Any) -> bool:
        if super().__instancecheck__(instance):
            return True
        return callable(getattr(instance, "state", None)) and callable(getattr(instance, "capabilities", None))


@runtime_checkable
class MotionControllerPort(Protocol, metaclass=_MotionControllerPortMeta):
    """Worker motion controller port consumed by ``BatchScheduler``."""

    arm_layout: ArmLayout

    def state(self) -> dict[str, Any]: ...
    def capabilities(self) -> dict[str, Any]: ...
    def read_state(self) -> dict[str, Any]: ...
    def hold(self) -> dict[str, Any]: ...
    def dispatch_row(self, row: Sequence[float], *, command_phase: str = "batch") -> bool: ...
    def step_budget_remaining(self) -> int | None: ...
    def settle_and_correct(self, *args: Any, **kwargs: Any) -> tuple[bool, str | None, dict[str, Any] | None]: ...


class BatchScheduler:
    """Backend-neutral K-arm, N-DoF cached-batch coordinator bound to an ``ArmLayout``."""

    def __init__(
        self,
        controller: Any,
        *,
        layout: ArmLayout | None = None,
        plans: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.controller = controller
        if layout is None:
            ctrl_layout = getattr(controller, "arm_layout", None)
            if isinstance(ctrl_layout, ArmLayout):
                layout = ctrl_layout
            else:
                from .batch_motion import _embodiment

                layout = ArmLayout.from_profile(_embodiment(controller))
        self.layout = layout
        try:
            controller.arm_layout = layout
        except (AttributeError, TypeError):
            pass
        if plans is not None:
            controller._batch_plans = plans
        elif not hasattr(controller, "_batch_plans"):
            fallback_plans = getattr(controller, "_plans", None)
            try:
                controller._batch_plans = fallback_plans if isinstance(fallback_plans, dict) else {}
            except (AttributeError, TypeError):
                pass

    @property
    def arm_layout(self) -> ArmLayout:
        return self.layout

    @property
    def plans(self) -> dict[str, dict[str, Any]]:
        return self.controller._batch_plans

    def read_state(self) -> dict[str, Any]:
        fn = getattr(self.controller, "read_state", None)
        if callable(fn):
            return fn()
        return self.controller.state()

    def hold(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        fn = getattr(self.controller, "hold", None)
        if callable(fn):
            return fn(*args, **kwargs)
        priv = getattr(self.controller, "_hold", None)
        if callable(priv):
            return priv(*args, **kwargs)
        stop_fn = getattr(self.controller, "_stop", None)
        if callable(stop_fn):
            return stop_fn(*args, **kwargs)
        return {"confirmed": True, "settling_steps": 0}

    def dispatch_row(self, row: Sequence[float], *, command_phase: str = "batch", **kwargs: Any) -> bool:
        ctrl_dispatch = getattr(self.controller, "dispatch_row", None)
        if callable(ctrl_dispatch):
            return bool(ctrl_dispatch(row, command_phase=command_phase, **kwargs))
        step_fn = getattr(self.controller, "step", None)
        if callable(step_fn):
            return step_fn(row) is not False
        native_fn = getattr(self.controller, "_native_action", None)
        if callable(native_fn):
            return bool(native_fn(row))
        raise TypeError("controller_missing_dispatch_row_or_step")

    def step_budget_remaining(self) -> int | None:
        fn = getattr(self.controller, "step_budget_remaining", None)
        if callable(fn):
            return fn()
        limit = getattr(self.controller, "native_action_budget_limit", None)
        env = getattr(self.controller, "env", None)
        if limit is not None and env is not None and hasattr(env, "take_action_cnt"):
            return max(0, int(limit) - int(env.take_action_cnt[0]))
        return None

    def settle_and_correct(self, *args: Any, **kwargs: Any) -> tuple[bool, str | None, dict[str, Any] | None]:
        fn = getattr(self.controller, "settle_and_correct", None)
        if callable(fn):
            return fn(*args, **kwargs)
        arrive_fn = getattr(self.controller, "_arrive", None)
        if callable(arrive_fn):
            return arrive_fn(*args, **kwargs)
        return True, None, None

    @staticmethod
    def admit_cached_plan(
        plan: dict[str, Any] | None,
        action: Mapping[str, Any],
        *,
        digest_value: str,
        max_age_s: float | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Validate cached plan existence, status, cursor, revision, command digest, and optional real-time age."""
        if plan is None:
            raise ValueError("batch_plan_unknown_or_expired")
        if plan.get("status") not in ("planned", "awaiting_observation"):
            raise ValueError("batch_plan_already_consumed_or_failed")
        stored_digest = plan.get("action_digest") if "action_digest" in plan else plan.get("digest")
        if action.get("batch_expected_cursor", plan["cursor"]) != plan["cursor"]:
            raise ValueError("batch_plan_cursor_changed_since_review")
        if action.get("batch_expected_revision", plan["revision"]) != plan["revision"]:
            raise ValueError("batch_plan_revision_changed_since_review")
        if stored_digest != digest_value:
            raise ValueError("batch_plan_commands_changed")
        if max_age_s is not None and now is not None and "created_at" in plan:
            if float(now) - float(plan["created_at"]) > float(max_age_s):
                plan["status"] = "invalidated"
                plan["executable"] = False
                raise ValueError("batch_plan_expired_real_world_moved")
        return plan

    @staticmethod
    def prepare_stage_carry_slip(
        carry_monitors: dict[str, CarrySlipMonitor],
        arm: str,
        stage_id: str,
        kind: str,
        goal: float | None,
        stage_result: dict[str, Any],
        record_guard: Mapping[str, Any] | None = None,
    ) -> CarrySlipMonitor | None:
        """Disarm on jaw opening (``goal >= 0.5``), echo ``effective_grip_guard``, and return active move monitor."""
        if isinstance(record_guard, Mapping) and "carry_slip_guard" in record_guard:
            stage_result["effective_grip_guard"] = format_effective_grip_guard(record_guard)
        if goal is not None and float(goal) >= CLOSING_JAW_GOAL_BELOW:
            carry_monitors.pop(arm, None)
            return None
        monitor = carry_monitors.get(arm)
        if monitor is not None and kind == "move":
            monitor.reset_streak()
            stage_result["effective_grip_guard"] = deepcopy(monitor.effective_grip_guard)
            return monitor
        return None

    @staticmethod
    def check_carry_slip_step(
        monitor: CarrySlipMonitor | None,
        stage_id: str,
        step_index: int,
        measured_aperture_m: float | None,
    ) -> dict[str, Any] | None:
        """Check one step's measured fingertip aperture against an active ``CarrySlipMonitor``."""
        if monitor is None:
            return None
        return monitor.observe_step(stage_id, step_index, measured_aperture_m)

    @staticmethod
    def finish_stage_carry_slip(
        carry_monitors: dict[str, CarrySlipMonitor],
        arm: str,
        kind: str,
        goal: float | None,
    ) -> None:
        """Consume ``next_move`` monitor after a covered ``move`` stage completes, or disarm on opening jaw."""
        if goal is not None and float(goal) >= CLOSING_JAW_GOAL_BELOW:
            carry_monitors.pop(arm, None)
            return
        monitor = carry_monitors.get(arm)
        if monitor is not None and kind == "move" and monitor.scope == "next_move":
            carry_monitors.pop(arm, None)

    @staticmethod
    def record_carry_slip_pause(
        plan: dict[str, Any],
        receipt: dict[str, Any],
        stage_results: dict[str, dict[str, Any]],
        stage_id: str,
        trigger: Mapping[str, Any],
        stop_confirmation: Mapping[str, Any] | None,
        *,
        stages: Sequence[Mapping[str, Any]] | None = None,
        dependencies: Mapping[str, set[str]] | None = None,
        carry_monitors: dict[str, CarrySlipMonitor] | None = None,
    ) -> None:
        """Record a mid-move ``carry_slip_aperture_below_min_m`` checkpoint pause on stage, plan, and receipt."""
        slip_at = dict(trigger["slip_detected_at"])
        eff_guard = dict(trigger["effective_grip_guard"])
        arm = str(slip_at["arm"])
        if carry_monitors is not None:
            monitor = carry_monitors.get(arm)
            if monitor is not None and monitor.scope == "until_release":
                monitor.reset_streak()
            else:
                carry_monitors.pop(arm, None)
        for armed_key in ("armed_grip_guard", "active_arm_grip_guard"):
            armed_map = plan.get(armed_key)
            if isinstance(armed_map, dict):
                armed_map.pop(arm, None)
        stage_res = stage_results[stage_id]
        stage_res.update(
            status="paused_at_checkpoint",
            checkpoint_reason=CARRY_SLIP_REASON,
            checkpoint_reason_code=CARRY_SLIP_REASON,
            effective_grip_guard=deepcopy(eff_guard),
            slip_detected_at=deepcopy(slip_at),
        )
        if stages is not None:
            skip_dependents(stage_id, stages, dependencies, stage_results=stage_results)
        confirmed = (
            not isinstance(stop_confirmation, Mapping)
            or (stop_confirmation.get("confirmed") is not False and stop_confirmation.get("status") != "unknown")
        )
        plan["status"] = "awaiting_observation" if confirmed else "failed"
        receipt.update(
            pause_receipt(
                [stage_id],
                {stage_id: str(trigger["message"])},
                stop_confirmation,
                checkpoint_reason=CARRY_SLIP_REASON,
                checkpoint_reason_code=CARRY_SLIP_REASON,
                checkpoint_stage_id=stage_id,
                effective_grip_guard=eff_guard,
                slip_detected_at=slip_at,
            )
        )

    @staticmethod
    def evaluate_post_stage_grip_guard(
        armed_grip_guard: dict[str, Any],
        carry_monitors: dict[str, CarrySlipMonitor],
        arm: str,
        stage_id: str,
        kind: str,
        goal: float | None,
        guard: Mapping[str, Any] | None,
        grip_facts: Mapping[str, Any],
        *,
        has_remaining_stages: bool,
        has_arm_remaining: bool,
        is_downward_move: bool,
        post_grasp_recheck: bool = True,
        adapt_evaluation: Callable[[dict[str, Any], bool], tuple[dict[str, Any], bool]] | None = None,
        **gripper: Any,
    ) -> tuple[dict[str, Any] | None, bool]:
        """Evaluate shared post-stage ``grip_guard`` and arm/disarm ``CarrySlipMonitor`` consistently."""
        closing = goal is not None and float(goal) < CLOSING_JAW_GOAL_BELOW
        if goal is not None and not closing:
            carry_monitors.pop(arm, None)
        evaluation, triggered = after_stage_grip_guard(
            armed_grip_guard,
            arm,
            kind,
            goal,
            dict(guard) if isinstance(guard, Mapping) else guard,
            grip_facts,
            has_remaining_stages=has_remaining_stages,
            has_arm_remaining=has_arm_remaining,
            is_downward_move=is_downward_move,
            post_grasp_recheck=post_grasp_recheck,
            **gripper,
        )
        if evaluation is not None and adapt_evaluation is not None:
            evaluation, triggered = adapt_evaluation(evaluation, triggered)
        if triggered:
            carry_monitors.pop(arm, None)
        elif (
            closing
            and isinstance(guard, Mapping)
            and guard.get("carry_slip_guard") is True
            and guard.get("min_aperture_m") is not None
            and float(guard["min_aperture_m"]) > 0.0
        ):
            carry_monitors[arm] = CarrySlipMonitor(
                arm,
                float(guard["min_aperture_m"]),
                scope=str(guard.get("carry_slip_scope") or "next_move"),
                slip_consecutive_steps=int(guard.get("slip_consecutive_steps") or 1),
                source_stage_id=stage_id,
                effective_grip_guard=dict(guard),
            )
        return evaluation, triggered

    def run_stage_plan(
        self,
        *,
        plan_id: str,
        plan: dict[str, Any],
        action: Mapping[str, Any],
        receipt: dict[str, Any],
        default_arm: str,
        failed_reasons: Sequence[str],
        on_stage_progress: Callable[[str], None],
        on_pause: Callable[..., None],
        on_carry_slip_hold: Callable[[], tuple[dict[str, Any], Callable[[], None]]],
        run_stage: Callable[[str, dict[str, Any], dict[str, Any], dict[str, Any], CarrySlipMonitor | None], tuple[bool, str | None, Any, Any]],
        compute_facts: Callable[[Any, Any], dict[str, Any]],
        is_downward_move: Callable[[dict[str, Any], Any], bool],
        post_grasp_recheck: bool,
        guard_parameters: Mapping[str, Any],
        on_after_stage_facts: Callable[[str, dict[str, Any], dict[str, Any], Any], None] | None = None,
        adapt_grip_evaluation: Callable[[str, dict[str, Any], dict[str, Any], dict[str, Any], list[str], dict[str, Any], bool], tuple[dict[str, Any], bool]] | None = None,
        on_stage_failed: Callable[[str, str | None, dict[str, Any]], None] | None = None,
        on_batch_failed: Callable[[], None] | None = None,
    ) -> None:
        """Execute a stage-ordered cached plan with unified checkpoints, ``grip_guard``, ``carry_slip_guard``, and skipping."""
        by_id = {stage["id"]: stage for stage in action["stages"]}
        results = receipt.setdefault("stage_results", {})
        for identifier in plan["order"]:
            rec = plan["records"][identifier]
            results[identifier] = {
                "kind": rec["kind"],
                "arm": None if rec["kind"] == "observe" else (rec.get("arm") or default_arm),
                "status": "completed" if identifier in plan["completed"] else "pending",
            }
        carry_monitors: dict[str, CarrySlipMonitor] = plan.setdefault("active_carry_slip_monitors", {})
        armed_grip = plan.setdefault("armed_grip_guard", {})
        failed: str | None = None
        while plan["index"] < len(plan["order"]):
            identifier = plan["order"][plan["index"]]
            plan["active_stage_id"] = identifier
            record, stage = plan["records"][identifier], by_id[identifier]
            arm = record.get("arm") or default_arm
            marker = plan["jaw_pauses"].pop(identifier, None)
            if marker is not None:
                on_pause(
                    marker["stage_ids"],
                    marker["reasons"],
                    checkpoint_reason=None,
                    checkpoint_reason_code="jaw_checkpoint",
                )
                break
            on_stage_progress(identifier)
            result = results[identifier]
            result["status"] = "running"
            if record["kind"] == "observe":
                if plan["index"] + 1 < len(plan["order"]):
                    on_pause(
                        [identifier],
                        {identifier: stage["reason"]},
                        checkpoint_reason=None,
                        checkpoint_reason_code="observation_checkpoint",
                    )
                    plan["index"] += 1
                    plan["completed"].append(identifier)
                    result["status"] = "completed"
                    break
                plan["index"] += 1
                plan["completed"].append(identifier)
                result["status"] = "completed"
                continue
            goal = record.get("goal_normalized")
            goal_f = None if goal is None else float(goal)
            monitor = self.prepare_stage_carry_slip(
                carry_monitors, arm, identifier, record["kind"], goal_f, result, record.get("grip_guard")
            )
            ok, reason, begin_sample, end_sample = run_stage(identifier, stage, record, result, monitor)
            if not ok:
                if reason == CARRY_SLIP_REASON and "carry_slip" in result:
                    slip_trigger = result.pop("carry_slip")
                    stop_conf, sync_paused = on_carry_slip_hold()
                    sync_paused()
                    self.record_carry_slip_pause(
                        plan,
                        receipt,
                        results,
                        identifier,
                        slip_trigger,
                        stop_conf,
                        stages=action["stages"],
                        carry_monitors=carry_monitors,
                    )
                    break
                result["status"] = "failed" if reason in failed_reasons else "stopped"
                failed = identifier
                skip_dependents(identifier, action["stages"], stage_results=results)
                if on_stage_failed is not None:
                    on_stage_failed(identifier, reason, result)
                break
            self.finish_stage_carry_slip(carry_monitors, arm, record["kind"], goal_f)
            facts = result["grip_facts"] = compute_facts(begin_sample, end_sample)
            if on_after_stage_facts is not None:
                on_after_stage_facts(identifier, record, result, end_sample)
            remaining = plan["order"][plan["index"] + 1 :]
            downward = is_downward_move(record, begin_sample)
            has_arm_rem = any(
                plan["records"][sid]["kind"] != "observe" or plan["records"][sid].get("arm") == arm
                for sid in remaining
            )

            def _adapt(ev: dict[str, Any], trig: bool) -> tuple[dict[str, Any], bool]:
                if adapt_grip_evaluation is None:
                    return ev, trig
                return adapt_grip_evaluation(identifier, record, result, facts, remaining, ev, trig)

            evaluation, triggered = self.evaluate_post_stage_grip_guard(
                armed_grip,
                carry_monitors,
                arm,
                identifier,
                record["kind"],
                goal_f,
                record.get("grip_guard"),
                facts,
                has_remaining_stages=bool(remaining),
                has_arm_remaining=has_arm_rem,
                is_downward_move=downward,
                post_grasp_recheck=post_grasp_recheck,
                adapt_evaluation=_adapt if adapt_grip_evaluation is not None else None,
                **guard_parameters,
            )
            if evaluation is not None:
                result["grip_guard"] = evaluation
            if triggered and evaluation is not None:
                if remaining:
                    plan["jaw_pauses"].pop(remaining[0], None)
                on_pause(
                    [identifier],
                    {identifier: evaluation["message"]},
                    checkpoint_reason=evaluation["reason"],
                    checkpoint_reason_code=evaluation["reason"],
                )
                result.update(
                    status="paused_at_checkpoint",
                    checkpoint_reason=evaluation["reason"],
                    checkpoint_reason_code=evaluation["reason"],
                )
                plan["completed"].append(identifier)
                plan["index"] += 1
                break
            result["status"] = "completed"
            plan["completed"].append(identifier)
            plan["index"] += 1
        else:
            plan["status"] = "completed"
            receipt.update(status="completed", reason=None)
        plan.pop("active_stage_id", None)
        if failed is not None:
            plan["status"] = "failed"
            if on_batch_failed is not None:
                on_batch_failed()

    def preview(self, action: dict[str, Any]) -> dict[str, Any]:
        preview_fn = getattr(self.controller, "preview_batch", None)
        if callable(preview_fn) and not hasattr(self.controller, "env"):
            return preview_fn(action)
        from .batch_motion import preview_batch

        return preview_batch(self.controller, action)

    def inspect(self, plan_id: str) -> dict[str, Any]:
        from .batch_motion import inspect_batch

        return inspect_batch(self.controller, plan_id)

    def execute(
        self,
        action: dict[str, Any],
        progress_or_cache: Any,
        cancellation_or_receipt: Any = None,
        segments: list[dict[str, Any]] | None = None,
        progress: Callable[[dict[str, Any]], None] | None = None,
        interrupted: Callable[[str], bool] | None = None,
        dispatch: Callable[..., bool] | None = None,
    ) -> dict[str, Any]:
        """Execute a cached batch via either high-level ``(action, progress, cancellation)`` or full worker args."""
        from .batch_motion import execute_batch

        if segments is not None and progress is not None and interrupted is not None and dispatch is not None:
            cache = progress_or_cache
            receipt = cancellation_or_receipt
            execute_batch(self.controller, action, cache, receipt, segments, progress, interrupted, dispatch)
            return receipt

        prog_cb: Callable[[dict[str, Any]], None] = progress_or_cache if callable(progress_or_cache) else (lambda _ev: None)
        cancel_token = cancellation_or_receipt
        try:
            self.controller.arm_layout = self.layout
        except (AttributeError, TypeError):
            object.__setattr__(self.controller, "arm_layout", self.layout)
        try:
            self.controller._batch_plans = self.plans
        except (AttributeError, TypeError):
            object.__setattr__(self.controller, "_batch_plans", self.plans)
        ctrl_exec = getattr(self.controller, "execute", None)
        if (
            callable(ctrl_exec)
            and not callable(getattr(self.controller, "dispatch_row", None))
            and not callable(getattr(self.controller, "step", None))
        ):
            return ctrl_exec(action, prog_cb, cancel_token)
        if callable(ctrl_exec) and hasattr(self.controller, "env") and hasattr(self.controller, "observe"):
            return ctrl_exec(action, prog_cb, cancel_token)
        plan_id = action.get("batch_plan_id") or action.get("plan_id")
        if not isinstance(plan_id, str) or plan_id not in self.plans:
            raise ValueError("robot_action_plan_unknown_or_expired")
        cache = self.plans[plan_id]
        eff_action = dict(action, batch_plan_id=plan_id)
        receipt: dict[str, Any] = {"executed_steps": 0, "status": "unknown"}
        seg_list: list[dict[str, Any]] = []

        def _is_interrupted(phase: str = "batch") -> bool:
            if callable(cancel_token):
                return bool(cancel_token(phase))
            is_set = getattr(cancel_token, "is_set", None)
            return bool(is_set()) if callable(is_set) else False

        def _dispatch_row(row: Sequence[float], *, command_phase: str = "batch") -> bool:
            if _is_interrupted(command_phase):
                receipt.update(status="cancelled", reason="operator_interrupted")
                return False
            ctrl_dispatch = getattr(self.controller, "dispatch_row", None)
            if callable(ctrl_dispatch):
                ok = bool(ctrl_dispatch(row, command_phase=command_phase, receipt=receipt))
            else:
                step_fn = getattr(self.controller, "step", None)
                if not callable(step_fn):
                    raise TypeError("controller_missing_dispatch_row_or_step")
                ok = step_fn(row) is not False
            if ok:
                receipt["executed_steps"] = int(receipt.get("executed_steps", 0)) + 1
            return ok

        execute_batch(self.controller, eff_action, cache, receipt, seg_list, prog_cb, _is_interrupted, _dispatch_row)
        receipt.setdefault("segments", seg_list)
        return receipt


__all__ = [
    "ArmLayout",
    "ArmSlot",
    "BatchScheduler",
    "MotionControllerPort",
    "StageScheduleItem",
    "jaw_pauses",
    "order_stages",
    "pause_receipt",
    "skip_dependents",
    "summarize_batch",
]
