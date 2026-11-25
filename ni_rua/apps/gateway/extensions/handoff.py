"""Handoff capability extension: successor run allocation, gating, and speed-cap carry-over."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from aiohttp import web

from ....core.config import namespace_values
from ....core.contracts import utc_now
from ....cortex.agents import profiles
from ....cortex.agents.factory import EXTERNAL_BACKEND, actor_upstream
from ...library.experiments import allocate_run, finish_run
from ...library.scoring import classify_episode_result
from .agent_settings import HANDOFF_BLOCKERS, _UNSET, _code
from .operator_console import LEASE_COMMANDS, LEASE_FREE_COMMANDS, MODEL_ID, PREEMPTIVE_COMMANDS

HANDOFF_SCHEMA = "ni_rua.handoff.v1"
HANDOFF_OBSERVE_PROMPT = (
    "Operator console session start. Inspect robot_capabilities and robot_observe and, using "
    "only those RGB images and the measured robot state, briefly report what you see and the "
    "robot's current state. Do not move the robot or change its state in this turn; wait for "
    "the operator's next instruction."
)
PROMPT_MAX_CHARS = 16000
REASON_MAX_CHARS = 1000
SAFE_DURING_HANDOFF = frozenset({"stop"}) | LEASE_COMMANDS | PREEMPTIVE_COMMANDS | LEASE_FREE_COMMANDS


def failure_reason(error):
    text = str(error).strip()
    return (f"{type(error).__name__}: {text}" if text else type(error).__name__)[:REASON_MAX_CHARS]


def base_config(args, workbench):
    config = (
        getattr(workbench, "config", None)
        if getattr(args, "resume", False)
        else (getattr(args, "resolved_config", None) or getattr(workbench, "config", None))
    )
    if config is None or not hasattr(config, "with_overrides"):
        return None
    return config.with_overrides({"run": dict(run_dir=None, resume=False, replay=False, prompt=None)})


def run_config(base, root, port, prompt):
    return base.with_overrides(
        {
            "run": dict(
                run_dir=str(root),
                resume=False,
                replay=False,
                port=int(port),
                prompt=prompt,
                benchmark=False,
                exit_after_turn=False,
            )
        }
    )


def launch_args(config):
    return argparse.Namespace(**namespace_values(config), resolved_config=config)


def allocate(config):
    run = config.run
    return allocate_run(
        run.experiment_root,
        category=run.category,
        task=run.task,
        title=run.title,
        tags=list(run.tag or ()),
        kind="workbench",
    )


def finish_unstarted(root, launch, *, status, reason):
    if root is None:
        return
    summary = dict(
        task_success=None,
        normalized_score=None,
        termination_reason=reason,
        infrastructure_error=reason if status == "failed" else None,
        handoff=launch.link if launch is not None else None,
    )
    try:
        finish_run(Path(root), status=status, summary=classify_episode_result(summary, status))
    except (OSError, ValueError):
        pass


@dataclass(frozen=True)
class HandoffHost:
    base_config: object
    port: int
    done: asyncio.Event


@dataclass(frozen=True)
class HandoffRequest:
    request_id: str
    requested_at: str
    predecessor_run_id: str
    successor_root: Path
    successor_run_id: str
    config: object = field(repr=False)
    base_config: object = field(repr=False)
    port: int
    selection: dict
    previous: dict
    prompt: str | None
    carry: dict

    def receipt(self):
        return dict(
            request_id=self.request_id,
            command="handoff_run",
            status="handoff_accepted",
            successor_run_id=self.successor_run_id,
            predecessor_run_id=self.predecessor_run_id,
        )

    def record(self):
        return dict(
            schema=HANDOFF_SCHEMA,
            request_id=self.request_id,
            requested_at=self.requested_at,
            predecessor_run_id=self.predecessor_run_id,
            successor_run_id=self.successor_run_id,
            successor_run_dir=str(self.successor_root),
            port=self.port,
            **{"from": dict(self.previous)},
            to=dict(self.selection),
            prompt=self.prompt,
            prompt_source="operator" if self.prompt else "observe_only_default",
            carried=dict(self.carry),
        )


@dataclass
class Launch:
    signals: object
    request: HandoffRequest | None = None
    kind: str | None = None
    link: dict | None = None
    carry: dict = field(default_factory=dict)
    last: dict | None = None
    event: tuple | None = None
    started: bool = False

    @property
    def born(self):
        return self.request is not None


@dataclass
class Successor:
    args: argparse.Namespace
    launch: Launch

    def fallback(self, error):
        launch = self.launch
        request = launch.request
        if launch.kind != "successor" or launch.started or launch.signals.received is not None:
            return None
        reason = failure_reason(error)
        root = allocate(request.base_config)
        config = run_config(previous_settings(request), root, request.port, HANDOFF_OBSERVE_PROMPT)
        fallback = dict(failed_run_id=request.successor_run_id, reason=reason)
        link = dict(
            schema=HANDOFF_SCHEMA,
            predecessor_run_id=request.predecessor_run_id,
            request_id=request.request_id,
            requested_at=request.requested_at,
            fallback=fallback,
            carried=dict(request.carry),
            prompt_source="observe_only_default",
        )
        last = dict(
            status="fallback",
            request_id=request.request_id,
            predecessor_run_id=request.predecessor_run_id,
            successor_run_id=request.successor_run_id,
            failed_run_id=request.successor_run_id,
            reason=reason,
            fallback=fallback,
            at=utc_now(),
        )
        event = (
            "run.handoff_fallback",
            dict(
                request_id=request.request_id,
                failed_run_id=request.successor_run_id,
                reason=reason,
                predecessor_run_id=request.predecessor_run_id,
                carried=dict(request.carry),
            ),
        )
        return Successor(
            launch_args(config),
            Launch(
                launch.signals,
                request=request,
                kind="fallback",
                link=link,
                carry=dict(request.carry),
                last=last,
                event=event,
            ),
        )


def previous_settings(request):
    previous = request.previous or {}
    model = {
        key: previous.get(name)
        for key, name in (("name", "model"), ("effort", "effort"))
        if isinstance(previous.get(name), str) and previous.get(name)
    }
    if not model:
        return request.base_config
    try:
        return request.base_config.with_overrides({"model": model})
    except ValueError:
        return request.base_config


def successor_of(request, signals):
    link = dict(
        schema=HANDOFF_SCHEMA,
        predecessor_run_id=request.predecessor_run_id,
        request_id=request.request_id,
        requested_at=request.requested_at,
        fallback=None,
        carried=dict(request.carry),
        prompt_source="operator" if request.prompt else "observe_only_default",
    )
    last = dict(
        status="completed",
        request_id=request.request_id,
        predecessor_run_id=request.predecessor_run_id,
        successor_run_id=request.successor_run_id,
        at=utc_now(),
    )
    event = (
        "run.handoff_started",
        dict(
            request_id=request.request_id,
            predecessor_run_id=request.predecessor_run_id,
            **{"from": dict(request.previous)},
            to=dict(request.selection),
            carried=dict(request.carry),
        ),
    )
    return Successor(
        launch_args(request.config),
        Launch(
            signals,
            request=request,
            kind="successor",
            link=link,
            carry=dict(request.carry),
            last=last,
            event=event,
        ),
    )


def adopt(workbench, launch):
    if launch is None or not launch.born:
        return
    workbench.handoff_link = dict(launch.link)
    workbench.handoff_carry = dict(launch.carry)
    workbench.handoff_last = dict(launch.last) if launch.last else None
    if launch.event is not None:
        kind, payload = launch.event
        workbench.store.event(kind, dict(payload))


def summary_links(summary, workbench, launch, final_status):
    summary = dict(summary)
    if launch is not None and launch.born:
        summary["handoff"] = dict(launch.link)
    request = getattr(workbench, "successor", None)
    if isinstance(request, HandoffRequest):
        abandoned = final_status == "unknown"
        summary["successor_run_id"] = request.successor_run_id
        summary["successor_handoff"] = dict(
            request.record(),
            status="abandoned" if abandoned else "handed_off",
            reason="predecessor_cleanup_unconfirmed" if abandoned else None,
        )
    return summary


def next_launch(workbench, signals, final_status):
    request = getattr(workbench, "successor", None)
    if not isinstance(request, HandoffRequest):
        return None
    successor = successor_of(request, signals)
    reason = (
        "gateway_shutdown_before_start"
        if signals.received is not None
        else "predecessor_cleanup_unconfirmed"
        if final_status == "unknown"
        else None
    )
    if reason is not None:
        finish_unstarted(request.successor_root, successor.launch, status="cancelled", reason=reason)
        return None
    return successor


class HandoffExtension:
    """Capability-mounted handoff extension for ``Workbench``."""

    name = "handoff"

    def __init__(self, workbench):
        self.workbench = workbench
        self.init_state()

    def init_state(self):
        wb = self.workbench
        wb.handoff_host = None
        wb.successor = None
        wb.handoff_link = None
        wb.handoff_carry = None
        wb.handoff_last = None

    def supports(self, workbench=None) -> bool:
        target = workbench if workbench is not None else self.workbench
        return not getattr(target, "replay", False)

    def mount(self, app) -> None:
        # Handoff commands enter through POST /v1/control (command="handoff_run").
        return None

    def available(self):
        wb = self.workbench
        return getattr(wb, "handoff_host", None) is not None and not getattr(wb, "replay", False)

    def state(self):
        wb = self.workbench
        link = getattr(wb, "handoff_link", None)
        request = getattr(wb, "successor", None)
        if link is None and not isinstance(request, HandoffRequest):
            return None
        view = dict(link) if link else dict(predecessor_run_id=None, request_id=None, fallback=None)
        if isinstance(request, HandoffRequest):
            view["successor"] = dict(
                request_id=request.request_id,
                successor_run_id=request.successor_run_id,
                status="accepted",
            )
        return view

    def refuse_during_handoff(self, command, identifier):
        request = getattr(self.workbench, "successor", None)
        if not isinstance(request, HandoffRequest) or command in SAFE_DURING_HANDOFF:
            return
        if command == "handoff_run" and identifier == request.request_id:
            return
        raise ValueError(
            f"handoff_in_progress: this run is closing for successor run {request.successor_run_id}; "
            "nothing was changed"
        )

    @staticmethod
    def _refuse(codes, details=None):
        parts = [f"{code}: {(details or {}).get(code) or HANDOFF_BLOCKERS.get(code) or code}" for code in codes]
        return ValueError("; ".join(parts) + "; nothing was changed")

    def _selection(self, value):
        wb = self.workbench
        current = wb.agent_settings_current()
        if value.get("backend") == EXTERNAL_BACKEND:
            raise ValueError(
                "handoff_unsupported_by_backend: an external agent run starts only from the command line "
                "(--agent-backend external); a handoff cannot start one; nothing was changed"
            )
        selection = {}
        for name in ("backend", "profile", "wire_api"):
            item = value.get(name)
            if item is None and name == "wire_api":
                selection[name] = None
                continue
            if item is None:
                raise ValueError(f"handoff_requires_{name}: name the {name} of the successor run")
            wb._validate_session_field(name, item)
            selection[name] = item
        model = value.get("model") or current.get("model")
        if not isinstance(model, str) or not MODEL_ID.match(model):
            raise ValueError(
                "invalid_model: a model id starts with a letter or digit (letters, digits, . _ : / @ + - "
                "[ ]; at most 200 characters)"
            )
        selection["model"] = model
        effort = value.get("effort") or current.get("effort")
        if effort is not None and not isinstance(effort, str):
            raise ValueError("invalid_effort: effort is a string")
        selection["effort"] = effort
        thinking = value["thinking_display"] if "thinking_display" in value else _UNSET
        if thinking is not _UNSET:
            wb._validate_session_field("thinking_display", thinking)
            if selection["backend"] != "opencode":
                thinking = _UNSET
        selection["thinking_display"] = thinking
        prompt = value.get("prompt")
        if prompt is not None and (not isinstance(prompt, str) or not prompt.strip() or len(prompt) > PROMPT_MAX_CHARS):
            raise ValueError(
                f"invalid_task_prompt: a non-empty prompt of at most {PROMPT_MAX_CHARS} characters, or none"
            )
        selection["prompt"] = prompt.strip() if isinstance(prompt, str) else None
        return selection

    def price_card_problem(self, candidate, backend):
        card = getattr(getattr(self.workbench, "session", None), "price_card", None)
        if not isinstance(card, dict):
            return None
        upstream = actor_upstream(candidate, backend)
        provider, model = getattr(upstream, "provider", None), candidate.model.name
        if any(
            isinstance(rate, dict) and rate.get("provider") == provider and rate.get("model") == model
            for rate in card.get("rates") or ()
        ):
            return None
        return (
            f"the run's price card (source {card.get('source')!r}, version {card.get('version')!r}) has no rate "
            f"for provider {provider!r} and model {model!r}; start a run with a matching price card instead"
        )

    def candidate(self, backend, profile, wire, model, effort, thinking=_UNSET):
        wb = self.workbench
        host = getattr(wb, "handoff_host", None)
        base = host.base_config if host is not None else wb.config
        options = {} if thinking is _UNSET else {"thinking_display": thinking}
        return profiles.apply_profile(base, backend, profile, wire, model=model, effort=effort, **options)

    def selection_blockers(self, backend, profile, wire, model, effort, thinking=_UNSET):
        wb = self.workbench
        current = wb.agent_settings_current()
        try:
            candidate = self.candidate(
                backend,
                profile,
                wire,
                model or current.get("model"),
                effort or current.get("effort"),
                thinking,
            )
        except ValueError:
            return []
        problem = self.price_card_problem(candidate, backend)
        return [("price_card_mismatch", problem)] if problem else []

    async def request_handoff(self, identifier, value):
        wb = self.workbench
        link = getattr(wb, "handoff_link", None) or {}
        if link.get("request_id") is not None and identifier == link["request_id"]:
            failed = (link.get("fallback") or {}).get("failed_run_id")
            return dict(
                request_id=identifier,
                command="handoff_run",
                status="handoff_accepted",
                successor_run_id=failed or wb.identity["run_id"],
                predecessor_run_id=link.get("predecessor_run_id"),
                replayed=True,
            )
        if isinstance(wb.successor, HandoffRequest):
            if identifier == wb.successor.request_id:
                return dict(wb.successor.receipt(), replayed=True)
            raise self._refuse(["handoff_in_progress"])
        if not self.available():
            raise ValueError(
                "handoff_not_available: this run has no run supervisor that could start a successor "
                "(replay, benchmark or a Workbench outside the gateway); nothing was changed"
            )
        blockers = wb.handoff_blockers()
        if blockers:
            raise self._refuse(blockers)
        if not isinstance(value, dict):
            raise ValueError("handoff_requires_object")
        selection = self._selection(value)
        backend, profile_id, wire = selection["backend"], selection["profile"], selection["wire_api"]
        try:
            profile, mtime = await asyncio.to_thread(wb._load_profile, profile_id)
        except web.HTTPNotFound:
            raise ValueError(
                f"profile_not_found: no profile {profile_id!r} in {wb.profiles_dir()}; nothing was changed"
            ) from None
        except ValueError as error:
            raise ValueError(f"profile_invalid:{_code(error)}: {error}; nothing was changed") from None
        thinking = selection["thinking_display"]
        check = await asyncio.to_thread(
            wb._check_selection_cached,
            backend,
            str(profile.path),
            mtime,
            wire,
            selection["model"],
            selection["effort"],
            thinking,
        )
        if not check.ok:
            message = wb._readable(backend, check.reason, check.message, profile) or check.message
            raise ValueError(f"{check.reason}: {message}; nothing was changed")
        wire = check.wire_api
        host = wb.handoff_host
        try:
            candidate = self.candidate(
                backend,
                profile,
                wire,
                selection["model"],
                selection["effort"],
                thinking,
            )
        except ValueError as error:
            raise ValueError(f"config_invalid:{_code(error)}: {error}; nothing was changed") from None
        problem = self.price_card_problem(candidate, backend)
        if problem:
            raise self._refuse(["price_card_mismatch"], {"price_card_mismatch": problem})
        try:
            listing = await asyncio.to_thread(wb.read_catalogue, profile, mtime, wire, True)
        except Exception:  # noqa: BLE001
            listing = None
        if listing is not None and listing.failure == "key_refused":
            raise self._refuse(
                ["handoff_profile_key_refused"],
                {
                    "handoff_profile_key_refused": (
                        f"the provider refused the key of profile {profile.id!r} when its model listing was read now "
                        f"({listing.reason}); fix the key in the profile first"
                    )
                },
            )
        blockers = wb.handoff_blockers()
        if blockers:
            raise self._refuse(blockers)
        current = wb.agent_settings_current()
        upstream = actor_upstream(candidate, backend)
        cap = (getattr(wb, "operator_gate", None) or {}).get("speed_scale_cap")
        prompt = selection["prompt"]
        root = await asyncio.to_thread(allocate, candidate)
        request = HandoffRequest(
            request_id=identifier,
            requested_at=utc_now(),
            predecessor_run_id=wb.identity["run_id"],
            successor_root=root,
            successor_run_id=root.name,
            config=run_config(candidate, root, host.port, prompt or HANDOFF_OBSERVE_PROMPT),
            base_config=host.base_config,
            port=host.port,
            selection=dict(
                backend=backend,
                profile=profile.id,
                provider=getattr(upstream, "provider", None),
                wire_api=wire,
                model=candidate.model.name,
                effort=candidate.model.effort,
                thinking_display=(candidate.agent.opencode.thinking_display if backend == "opencode" else None),
            ),
            previous={
                key: current.get(key)
                for key in ("backend", "profile", "provider", "wire_api", "model", "effort", "service_tier", "thinking_display")
            },
            prompt=prompt,
            carry={"speed_scale_cap": float(cap)} if type(cap) in (int, float) else {},
        )
        record = request.record()
        wb.store.save("handoff", identifier, record)
        wb.store.event("run.handoff_requested", record)
        wb.successor = request
        wb.handoff_last = dict(
            status="accepted",
            request_id=identifier,
            successor_run_id=request.successor_run_id,
            at=request.requested_at,
        )
        wb.status = "handoff_requested"
        wb.harness.request_stop()
        host.done.set()
        return request.receipt()

    async def apply_carry(self):
        wb = self.workbench
        cap = (getattr(wb, "handoff_carry", None) or {}).get("speed_scale_cap")
        if cap is None:
            return None
        link = getattr(wb, "handoff_link", None) or {}
        identifier = f"handoff-{link.get('request_id')}-speed-cap"
        result = await wb._operator(
            "set_runtime_limits",
            identifier,
            event_origin="handoff",
            speed_scale_cap=float(cap),
        )
        if result.get("status") != "applied":
            raise RuntimeError(
                f"handoff_speed_cap_not_applied: {result.get('status')} {result.get('reason') or ''}".strip()
            )
        wb.operator_gate["speed_scale_cap"] = result.get("speed_scale_cap", float(cap))
        return result


__all__ = [
    "HANDOFF_OBSERVE_PROMPT",
    "HANDOFF_SCHEMA",
    "HandoffExtension",
    "HandoffHost",
    "HandoffRequest",
    "Launch",
    "PROMPT_MAX_CHARS",
    "REASON_MAX_CHARS",
    "SAFE_DURING_HANDOFF",
    "Successor",
    "adopt",
    "allocate",
    "base_config",
    "failure_reason",
    "finish_unstarted",
    "launch_args",
    "next_launch",
    "previous_settings",
    "run_config",
    "successor_of",
    "summary_links",
]
