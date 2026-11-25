"""Agent-settings capability extension: compatibility matrix, profile CRUD, live settings, and options API."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import tempfile
import threading

from aiohttp import web

from ....core.contracts import utc_now
from ....cortex.agents import factory, model_catalogue, profiles
from ....cortex.agents.base import AgentRejected, session_backend_kind
from ....cortex.agents.factory import BACKEND_WIRES, BACKENDS, actor_upstream
from ....cortex.agents.model_catalogue import CACHE_DIR_NAME, merge_model, native_codex_model
from ....cortex.agents.model_listing import shared_catalogue_cache
from ....cortex.agents.reasoning_options import backend_traits, reasoning_options
from .operator_console import MODEL_ID, SERVICE_TIERS, _utc_now

AGENT_SETTINGS_SCHEMA = "ni_rua.agent_settings.v1"
PRESETS_SCHEMA = "ni_rua.agent_presets.v1"
PRESETS_FILE = "presets.json"
BACKEND_LABELS = {"codex": "Codex", "claude_code": "Claude Code", "opencode": "OpenCode"}
SETTING_FIELDS = ("backend", "profile", "provider", "wire_api", "model", "effort", "service_tier", "thinking_display")
LIVE_FIELDS = ("model", "effort", "service_tier")
SESSION_FIELDS = ("backend", "profile", "wire_api", "thinking_display")
FIELD_LABELS = {
    "backend": "后端",
    "profile": "profile",
    "wire_api": "接口",
    "model": "模型",
    "effort": "推理强度",
    "service_tier": "速度",
    "thinking_display": "思考显示",
}
HANDOFF_BLOCKERS = {
    "replay_is_read_only": "回放只读，不能切换",
    "benchmark_run_owned_by_queue": "评测 run 由队列管理，不能在页面上切换",
    "run_starting": "run 仍在启动",
    "agent_turn_running": "需要 Agent 空闲（当前一轮仍在运行）",
    "action_in_progress": "有动作正在执行",
    "program_active_or_unresolved": "有程序在运行或状态未确认",
    "unknown_unacknowledged": "有未确认的动作结果，请先在恢复面板确认",
    "torque_not_on": "力矩已关闭或状态未知，请先上力矩",
    "handoff_in_progress": "切换已被接受，当前 run 正在收尾",
    "handoff_not_available": "当前 run 不能切换（回放、评测或没有网关监管）",
    "handoff_unsupported_by_backend": "外部 Agent 的 run 只能从命令行启动，不能切换",
    "price_card_mismatch": "本 run 的价格表没有新 provider / 模型的价格，切换后的费用估计会与之矛盾",
    "handoff_profile_key_refused": "供应商拒绝了该 profile 的密钥（HTTP 401/403）",
}
_CHECK_CACHE_MAX = 512
_UNSET = object()
_MAX_PROFILE_BODY_BYTES = 64 * 1024

_RUNTIME_PROBES = {}
_RUNTIME_LOCK = threading.Lock()


def reset_runtime_probes():
    with _RUNTIME_LOCK:
        _RUNTIME_PROBES.clear()


def runtime_problem(backend, config):
    if backend == "codex":
        return None
    agent = config.agent
    path = agent.claude_code.cli_path if backend == "claude_code" else agent.opencode.executable
    key = (backend, path)
    with _RUNTIME_LOCK:
        if key in _RUNTIME_PROBES:
            return _RUNTIME_PROBES[key]
    probe = factory.claude_code_runtime_problem if backend == "claude_code" else factory.opencode_runtime_problem
    problem = probe(path)
    with _RUNTIME_LOCK:
        _RUNTIME_PROBES[key] = problem
    return problem


def catalogue_cache(directory):
    return shared_catalogue_cache(Path(directory) / CACHE_DIR_NAME)


def _code(error):
    return str(error).split(":", 1)[0].strip()


def _profile_id_of(path, fallback=None):
    name = Path(path).name
    return name[: -len(profiles.PROFILE_SUFFIX)] if name.endswith(profiles.PROFILE_SUFFIX) else fallback


def _write_json_atomic(location, document):
    handle, temporary = tempfile.mkstemp(prefix="." + location.name + ".", suffix=".tmp", dir=location.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, location)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def read_presets(directory):
    try:
        document = json.loads((Path(directory) / PRESETS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    presets = document.get("presets") if isinstance(document, dict) else None
    return {key: value for key, value in (presets or {}).items() if isinstance(key, str) and isinstance(value, dict)}


class AgentSettingsExtension:
    """Capability-mounted agent-settings extension for ``Workbench``."""

    name = "agent_settings"

    def __init__(self, workbench):
        self.workbench = workbench
        self.init_state()

    def init_state(self):
        wb = self.workbench
        session = getattr(wb, "session", None)
        wb._agent_session_effort = (session, getattr(session, "effort", None))

    def supports(self, workbench=None) -> bool:
        return True

    def _meta(self):
        wb = self.workbench
        meta = getattr(wb, "_agent_settings_meta", None)
        if meta is None:
            stored = None
            store = getattr(wb, "store", None)
            if store is not None:
                try:
                    stored = store.read_state("runtime", "agent_settings")
                except Exception:  # noqa: BLE001
                    stored = None
            stored = stored if isinstance(stored, dict) else {}
            meta = wb._agent_settings_meta = dict(
                revision=stored.get("revision") if type(stored.get("revision")) is int else 0,
                changed_at=stored.get("changed_at"),
                turn_count=stored.get("turn_count") if type(stored.get("turn_count")) is int else 0,
                last_turn_id=stored.get("last_turn_id"),
                mixed=stored.get("mixed") is True,
            )
        return meta

    def _agent_backend(self):
        return session_backend_kind(getattr(self.workbench, "session", None))

    def _agent_profile(self):
        wb = self.workbench
        backend, config = self._agent_backend(), getattr(wb, "config", None)
        if backend not in BACKENDS or config is None:
            return None, None, None
        upstream = actor_upstream(config, backend)
        if upstream is None:
            return None, None, "codex_home" if backend == "codex" else None
        if upstream.source != "profile" or not upstream.settings:
            return None, None, upstream.source
        session = getattr(wb, "session", None)
        block = getattr(session, "upstream_profile_block", None)
        try:
            block = block() if callable(block) else getattr(session, "upstream_profile", None)
        except Exception:  # noqa: BLE001
            block = None
        identifier = block.get("id") if isinstance(block, dict) and isinstance(block.get("id"), str) else None
        return (
            identifier or _profile_id_of(upstream.settings, upstream.provider),
            os.path.realpath(upstream.settings),
            "profile",
        )

    def _agent_wire(self, backend):
        wb = self.workbench
        wire = getattr(getattr(wb, "session", None), "wire_api", None)
        if isinstance(wire, str) and wire:
            return wire
        if backend in BACKEND_WIRES and len(BACKEND_WIRES[backend]) == 1:
            return BACKEND_WIRES[backend][0]
        config = getattr(wb, "config", None)
        upstream = actor_upstream(config, backend) if config is not None and backend in BACKENDS else None
        if upstream is None:
            return None
        from ....cortex.agents.opencode.upstream_file import configured_wire
        return configured_wire(upstream)

    def _effective_effort(self):
        wb = self.workbench
        session = getattr(wb, "session", None)
        effort = getattr(session, "effort", None)
        if wb._session_supports("live_effort"):
            return effort
        started = getattr(wb, "_agent_session_effort", None)
        return started[1] if isinstance(started, tuple) and started[0] is session else effort

    def _current_service_tier(self):
        wb = self.workbench
        if not wb._session_supports("service_tier"):
            return None
        return getattr(getattr(wb.session, "model_proxy", None), "service_tier", None) or "default"

    def current(self):
        wb = self.workbench
        session, config = getattr(wb, "session", None), getattr(wb, "config", None)
        backend = self._agent_backend()
        profile_id, _, source = self._agent_profile()
        model = getattr(session, "model", None) or (
            getattr(getattr(config, "model", None), "name", None) if backend != factory.EXTERNAL_BACKEND else None
        )
        thinking = None
        if backend == "opencode" and config is not None:
            thinking = config.agent.opencode.thinking_display
        effort = self._effective_effort()
        current = dict(
            backend=backend,
            profile=profile_id,
            provider=getattr(session, "provider", None),
            wire_api=self._agent_wire(backend),
            model=model,
            effort=effort,
            service_tier=self._current_service_tier(),
            thinking_display=thinking,
            upstream_source=source,
        )
        selected = getattr(session, "effort", None)
        if selected is not None and selected != effort:
            current["effort_next_session"] = selected
        return current

    def event_context(self):
        current = self.current()
        return {key: current[key] for key in ("backend", "profile", "wire_api")}

    def _write_agent_state(self, current=None):
        wb = self.workbench
        if getattr(wb, "replay", False):
            return
        meta = self._meta()
        current = current or self.current()
        wb.store.set_state("runtime", "agent_settings", dict(schema=AGENT_SETTINGS_SCHEMA, **current, **meta))

    def setting_changed(self, field, previous, value, request_id, applies):
        if previous == value:
            return
        meta = self._meta()
        meta["revision"] += 1
        meta["changed_at"] = utc_now()
        if meta["turn_count"] and applies != "next_session_start":
            meta["mixed"] = True
        current = self.current()
        self._write_agent_state(current)
        self._remember_preset(current)

    def record_turn_settings(self, turn_id, origin):
        wb = self.workbench
        try:
            session, store = wb.session, wb.store
            current = self.current()
            turn_model = getattr(session, "active_turn_model", None)
            if isinstance(turn_model, str) and turn_model:
                current["model"] = turn_model
            meta = self._meta()
            served = (store.read_state("runtime", "model_requests") or {}).get("served_models")
            payload = dict(
                turn_id=turn_id,
                thread_id=getattr(session, "thread_id", None),
                origin=origin,
                revision=meta["revision"],
                **current,
                served_models_at_admission=dict(served) if isinstance(served, dict) else {},
            )
            store.event("run.turn_settings", payload, {"thread_id": payload["thread_id"], "turn_id": turn_id})
            meta["turn_count"] += 1
            meta["last_turn_id"] = turn_id
            self._write_agent_state(current)
        except Exception as error:  # noqa: BLE001
            try:
                wb.store.event(
                    "run.turn_settings_unrecorded",
                    dict(turn_id=turn_id, origin=origin, error=type(error).__name__),
                )
            except Exception:  # noqa: BLE001
                pass

    def profiles_dir(self):
        return profiles.profiles_directory(getattr(self.workbench, "config", None))

    def _remember_preset(self, current):
        wb = self.workbench
        if getattr(wb, "replay", False) or not current.get("profile") or not current.get("model"):
            return
        directory = self.profiles_dir()
        try:
            if not directory.is_dir():
                return
            presets = read_presets(directory)
            key = f"{current['profile']}/{current['backend']}/{current['model']}"
            presets[key] = dict(
                effort=current.get("effort_next_session") or current.get("effort"),
                service_tier=current.get("service_tier"),
                updated_at=_utc_now(),
            )
            _write_json_atomic(directory / PRESETS_FILE, dict(schema=PRESETS_SCHEMA, presets=presets))
        except OSError:
            pass

    def served_view(self):
        wb = self.workbench
        store = getattr(wb, "store", None)
        if store is None:
            return None
        try:
            models = (store.read_state("runtime", "model_requests") or {}).get("served_models")
            latest = next(
                (
                    event
                    for event in store.events_by_type(("model.model_request_finished",), limit=50, newest_first=True)
                    if isinstance(event["payload"].get("served_model"), str)
                ),
                None,
            )
        except Exception:  # noqa: BLE001
            return None
        tier = self._served_tier()
        if latest is None and not models and tier is None:
            return None
        return dict(
            model=latest["payload"]["served_model"] if latest else None,
            at=latest["occurred_at"] if latest else None,
            models=models if isinstance(models, dict) else None,
            service_tier=tier,
            service_tier_source="probe" if tier is not None else None,
        )

    def _served_tier(self):
        wb = self.workbench
        if not wb._session_supports("service_tier"):
            return None
        proxy = getattr(getattr(wb, "session", None), "model_proxy", None)
        tier = getattr(proxy, "service_tier", None)
        probe = (getattr(proxy, "service_tier_probes", None) or {}).get(tier) if tier else None
        served = probe.get("served_tier") if isinstance(probe, dict) else None
        return served if isinstance(served, str) else None

    def view(self):
        wb = self.workbench
        meta = self._meta()
        view = dict(
            self.current(),
            revision=meta["revision"],
            changed_at=meta["changed_at"],
            turn_count=meta["turn_count"],
            mixed=meta["mixed"],
        )
        client_view = getattr(getattr(wb, "session", None), "client_view", None)
        external = client_view() if callable(client_view) else None
        if external is not None:
            view["external"] = external
        return view

    def handoff_blockers(self):
        wb = self.workbench
        blockers = []
        config = getattr(wb, "config", None)
        run = getattr(config, "run", None)
        if getattr(wb, "replay", False):
            blockers.append("replay_is_read_only")
        if getattr(run, "benchmark", False) or getattr(run, "exit_after_turn", False):
            blockers.append("benchmark_run_owned_by_queue")
        if getattr(wb, "starting", False):
            blockers.append("run_starting")
        if getattr(getattr(wb, "session", None), "state", None) == "running":
            blockers.append("agent_turn_running")
        harness = getattr(wb, "harness", None)
        if getattr(harness, "active_action", None):
            blockers.append("action_in_progress")
        programs = getattr(harness, "programs", None)
        try:
            if programs and (programs.snapshot()["active_jobs"] or programs.unresolved_jobs()):
                blockers.append("program_active_or_unresolved")
        except Exception:  # noqa: BLE001
            blockers.append("program_active_or_unresolved")
        try:
            if wb.unknown_items():
                blockers.append("unknown_unacknowledged")
        except Exception:  # noqa: BLE001
            pass
        if (getattr(wb, "operator_gate", None) or {}).get("torque") in ("off", "unknown"):
            blockers.append("torque_not_on")
        if getattr(wb, "successor", None) is not None:
            blockers.append("handoff_in_progress")
        if "handoff_run" in getattr(getattr(wb, "session", None), "console_commands_unsupported", ()):
            blockers.append("handoff_unsupported_by_backend")
        available = getattr(wb, "handoff_available", None)
        if not (callable(available) and available()):
            blockers.append("handoff_not_available")
        return blockers

    def handoff_view(self, selection_blockers=()):
        wb = self.workbench
        blockers = [dict(code=code, message=HANDOFF_BLOCKERS.get(code)) for code in self.handoff_blockers()]
        blockers += [
            dict(code=code, message=HANDOFF_BLOCKERS.get(code), detail=detail)
            for code, detail in selection_blockers
        ]
        robot = getattr(wb, "robot", None)
        torque = getattr(robot, "disable_torque_on_disconnect", None)
        if not isinstance(torque, bool):
            deployment = getattr(robot, "deployment", None)
            torque = deployment.get("disable_torque_on_disconnect") if isinstance(deployment, dict) else None
        return dict(
            allowed=not blockers,
            blockers=blockers,
            disable_torque_on_disconnect=torque if isinstance(torque, bool) else None,
            last=getattr(wb, "handoff_last", None),
        )

    def _in_use_profiles(self):
        config, used = getattr(self.workbench, "config", None), set()
        for kind in BACKENDS if config is not None else ():
            upstream = actor_upstream(config, kind)
            if upstream is not None and upstream.source == "profile" and upstream.settings:
                used.update((os.path.abspath(upstream.settings), os.path.realpath(upstream.settings)))
                identifier = _profile_id_of(upstream.settings)
                if identifier:
                    used.add(identifier)
        return used

    @staticmethod
    def profile_view(info, used):
        view = dict(info.to_dict(), in_use=info.id in used or os.path.realpath(info.path) in used)
        error = view.get("error") if isinstance(view.get("error"), dict) else None
        view.update(error=error.get("code") if error else None, error_message=error.get("message") if error else None)
        return view

    @staticmethod
    def _backend_entry(backend, problem, current_backend):
        traits = {}
        for wire in BACKEND_WIRES[backend]:
            try:
                traits[wire] = backend_traits(backend, wire)
            except Exception:  # noqa: BLE001
                traits[wire] = None
        known = [item for item in traits.values() if item is not None]
        union = {name for item in known for name in item.effort_choices}
        label = BACKEND_LABELS[backend]
        return dict(
            id=backend,
            label=label,
            wires=list(BACKEND_WIRES[backend]),
            available=problem is None,
            reason=None if problem is None else f"backend_unavailable:{problem}",
            message=None if problem is None else f"{label} 运行时不可用（{problem}）",
            live_effort=any(item.live_effort for item in known),
            service_tier=any(item.service_tier for item in known),
            effort_choices=[name for name in profiles.EFFORT_CHOICES if name in union],
            effort_choices_by_wire={
                wire: list(item.effort_choices) if item else [] for wire, item in traits.items()
            },
            thinking_display_wires=[
                wire for wire, item in traits.items() if item is not None and item.thinking_display_choices
            ],
            current=backend == current_backend,
        )

    async def _availability(self):
        config = self.workbench.config
        problems = await asyncio.gather(*(asyncio.to_thread(runtime_problem, backend, config) for backend in BACKENDS))
        return dict(zip(BACKENDS, problems))

    def _check_cache(self):
        wb = self.workbench
        cache = getattr(wb, "_agent_check_cache", None)
        if cache is None or len(cache) > _CHECK_CACHE_MAX:
            cache = wb._agent_check_cache = {}
        return cache

    def _check_selection_cached(self, backend, path, mtime_ns, wire, model, effort, thinking=_UNSET):
        key = (backend, path, mtime_ns, wire, model, effort, "unset" if thinking is _UNSET else thinking)
        cache = self._check_cache()
        if key in cache:
            return cache[key]
        options = {} if thinking is _UNSET else {"thinking_display": thinking}
        check = profiles.check_selection(
            self.workbench.config,
            backend,
            path,
            wire,
            model=model,
            effort=effort,
            **options,
        )
        cache[key] = check
        return check

    @staticmethod
    def _matrix_effort(backend, wire, info, current_effort):
        try:
            choices = backend_traits(backend, wire).effort_choices
        except Exception:  # noqa: BLE001
            return current_effort
        for candidate in ((info.defaults or {}).get("effort"), current_effort):
            if candidate in choices:
                return candidate
        return "high" if "high" in choices else (choices[0] if choices else current_effort)

    @staticmethod
    def _readable(backend, reason, message, info=None):
        label = BACKEND_LABELS.get(backend, backend)
        if reason == "no_common_wire" and info is not None:
            return (
                f"该 profile 只提供 {'、'.join(info.wires) or '（无接口）'}，{label} 需要 "
                f"{' 或 '.join(BACKEND_WIRES[backend])}"
            )
        if reason == "profile_missing_key":
            return "该 profile 还没有保存密钥"
        if reason and reason.startswith("profile_invalid:"):
            return f"profile 文件无效（{reason.split(':', 1)[1]}）：{message}"
        if reason == "wire_not_in_profile":
            return f"该 profile 不提供 {label} 所需的接口"
        return message

    async def agent_matrix(self, infos, availability, current):
        cells, jobs = [], []
        for backend in BACKENDS:
            problem = availability.get(backend)
            for info in infos:
                cell = dict(
                    backend=backend,
                    profile=info.id,
                    compatible=False,
                    wires=[],
                    reason=None,
                    message=None,
                    checks=[],
                )
                cells.append(cell)
                if problem is not None:
                    cell.update(
                        reason=f"backend_unavailable:{problem}",
                        message=f"{BACKEND_LABELS[backend]} 运行时不可用（{problem}）",
                    )
                    continue
                base = profiles.matrix_cell(backend, info)
                if not base["compatible"]:
                    cell.update(
                        reason=base["reason"],
                        message=self._readable(backend, base["reason"], base["message"], info),
                    )
                    continue
                model = (info.defaults or {}).get("model") or current.get("model")
                for wire in base["wires"]:
                    effort = self._matrix_effort(backend, wire, info, current.get("effort"))
                    jobs.append(
                        (
                            cell,
                            wire,
                            effort,
                            asyncio.to_thread(
                                self._check_selection_cached,
                                backend,
                                info.path,
                                info.mtime_ns,
                                wire,
                                model,
                                effort,
                            ),
                        )
                    )
        results = await asyncio.gather(*(job[3] for job in jobs), return_exceptions=True)
        for (cell, wire, effort, _), check in zip(jobs, results):
            if isinstance(check, BaseException):
                row = dict(
                    wire=wire,
                    ok=False,
                    reason="check_failed:" + type(check).__name__,
                    message=str(check)[:300],
                    effort=effort,
                )
            else:
                row = dict(wire=wire, ok=check.ok, reason=check.reason, message=check.message, effort=effort)
            cell["checks"].append(row)
        for cell in cells:
            if not cell["checks"]:
                continue
            passing = [row["wire"] for row in cell["checks"] if row["ok"]]
            if passing:
                cell.update(compatible=True, wires=passing, reason=None, message=None)
            else:
                first = cell["checks"][0]
                info = next(item for item in infos if item.id == cell["profile"])
                cell.update(
                    reason=first["reason"],
                    message=self._readable(cell["backend"], first["reason"], first["message"], info),
                )
        return cells

    def _load_profile(self, profile_id):
        path = profiles.profile_path(self.profiles_dir(), profile_id)
        if path.is_symlink() or not path.is_file():
            raise web.HTTPNotFound(text="profile_not_found")
        profile = profiles.load_profile(path)
        try:
            mtime = path.stat().st_mtime_ns
        except OSError:
            mtime = None
        return profile, mtime

    def read_catalogue(self, profile, mtime_ns, wire, refresh=False):
        entry = profile.wires[wire]
        return catalogue_cache(self.profiles_dir()).read(
            profile.id,
            wire,
            credential=lambda: profiles.credential(profile, wire),
            profile_mtime=mtime_ns,
            listing_url=entry.models_url,
            refresh=refresh,
        )

    def _catalogue_entry(self, view, model_id):
        if view is None or not model_id:
            return None
        return next((model for model in view.models if model.id == model_id), None)

    def _peek_catalogue(self, profile_id, wire, mtime_ns):
        try:
            return catalogue_cache(self.profiles_dir()).peek(profile_id, wire, profile_mtime=mtime_ns)
        except ValueError:
            return None

    def listed_model_items(self):
        try:
            backend = self._agent_backend()
            profile_id, path, source = self._agent_profile()
            if profile_id:
                wire = self._agent_wire(backend)
                view = self._peek_catalogue(profile_id, wire, os.stat(path).st_mtime_ns) if wire else None
                models = view.usable(wire) if view is not None else ()
            elif backend == "codex" and source == "codex_home":
                entries = sorted(
                    (entry for entry in model_catalogue._codex_models().values() if entry.get("visibility") == "list"),
                    key=lambda entry: (
                        entry.get("priority") if type(entry.get("priority")) is int else 1 << 30,
                        entry["slug"],
                    ),
                )
                models = [native_codex_model(entry["slug"]) for entry in entries]
            else:
                return None
        except (OSError, ValueError):
            return None
        return [dict(value=model.id, name=model.name, image_input=model.image_input) for model in models if model] or None

    def annotate_model_options(self, options):
        try:
            backend = self._agent_backend()
            profile_id, path, _ = self._agent_profile()
            wire = self._agent_wire(backend)
            known = {}
            if profile_id and wire is not None:
                view = self._peek_catalogue(profile_id, wire, os.stat(path).st_mtime_ns)
                known = {model.id: model for model in view.models} if view is not None else {}
        except (OSError, ValueError):
            return options
        for option in options:
            model = known.get(option.get("value"))
            if backend == "codex":
                model = merge_model(model, native_codex_model(option.get("value")))
            if model is not None:
                option.update(
                    {
                        key: value
                        for key, value in model.to_dict().items()
                        if key
                        in (
                            "name",
                            "protocols",
                            "tools",
                            "context_window",
                            "max_output_tokens",
                            "reasoning",
                            "reasoning_efforts",
                        )
                    }
                )
                if option.get("image_input") is None:
                    option["image_input"] = model.image_input
        return options

    def _reasoning(self, backend, wire, entry, model_id, effort, thinking, *, live):
        wb = self.workbench
        if backend not in BACKENDS or wire is None:
            return None
        tier, probes = None, None
        if live:
            proxy = getattr(wb.session, "model_proxy", None)
            tier = self._current_service_tier()
            if callable(getattr(proxy, "select_model", None)) and wb._session_supports("service_tier"):
                probes = dict(getattr(proxy, "service_tier_probes", {}) or {})
        try:
            controls = reasoning_options(
                backend,
                wire,
                entry,
                model_id=model_id,
                effort=effort,
                service_tier=tier,
                thinking_display=thinking,
                tier_probes=probes,
                service_tiers=SERVICE_TIERS,
            )
        except ValueError:
            return None
        if live and controls["service_tier"].get("available"):
            controls["service_tier"]["served_tier"] = self._served_tier()
        return controls

    async def agent_options(self, refresh=False):
        directory = self.profiles_dir()
        infos = await asyncio.to_thread(profiles.list_profiles, directory)
        availability = await self._availability()
        current = self.view()
        matrix = await self.agent_matrix(infos, availability, current)
        used = self._in_use_profiles()
        profile_views = [self.profile_view(info, used) for info in infos]
        view = None
        running_path = self._agent_profile()[1]
        info = next((item for item in infos if item.valid and os.path.realpath(item.path) == running_path), None)
        if info is not None and current["wire_api"] in info.wires:
            try:
                profile, mtime = await asyncio.to_thread(self._load_profile, info.id)
                view = await asyncio.to_thread(self.read_catalogue, profile, mtime, current["wire_api"], refresh)
            except (ValueError, web.HTTPNotFound):
                view = None
        catalogues = []
        for item in infos:
            for wire in (item.wires if item.valid else ()):
                cached = (
                    view
                    if (item is info and wire == current["wire_api"] and view)
                    else self._peek_catalogue(item.id, wire, item.mtime_ns)
                )
                catalogues.append(
                    dict(
                        profile=item.id,
                        wire_api=wire,
                        status=cached.status if cached else None,
                        reason=cached.reason if cached else None,
                        failure=cached.failure if cached else None,
                        fetched_at=cached.fetched_at if cached else None,
                        age_s=cached.age_s if cached else None,
                        stale=bool(cached and cached.stale),
                        refreshing=bool(cached and cached.refreshing),
                        count=len(cached.models) if cached else None,
                    )
                )
        reasoning = self._reasoning(
            current["backend"],
            current["wire_api"],
            self._catalogue_entry(view, current["model"]),
            current["model"],
            current["effort"],
            current["thinking_display"],
            live=True,
        )
        backends = [self._backend_entry(backend, availability.get(backend), current["backend"]) for backend in BACKENDS]
        return dict(
            current=current,
            backends=backends,
            profiles=profile_views,
            matrix=matrix,
            reasoning=reasoning,
            catalogues=catalogues,
            handoff=self.handoff_view(),
            served=self.served_view(),
            presets=await asyncio.to_thread(read_presets, directory),
            profiles_dir=str(directory),
            profile_schema=dict(
                schema=profiles.SCHEMA,
                wires=list(profiles.PROFILE_WIRES),
                auth_schemes=list(profiles.AUTH_SCHEMES),
                tool_result_images=list(profiles.TOOL_RESULT_IMAGES),
                thinking_display=list(profiles.THINKING_DISPLAY_CHOICES),
                efforts=list(profiles.EFFORT_CHOICES),
                fields=[dict(name=name, help=text) for name, text in profiles.FIELD_HELP],
            ),
        )

    async def profile_models(self, profile_id, wire, refresh=False):
        profile, mtime = await asyncio.to_thread(self._load_profile, profile_id)
        if wire is None and len(profile.wires) == 1:
            wire = next(iter(profile.wires))
        if wire not in profile.wires:
            raise ValueError(
                f"profile_wire_not_served: the profile serves {', '.join(profile.wires)}; name one with ?wire="
            )
        view = await asyncio.to_thread(self.read_catalogue, profile, mtime, wire, refresh)
        body = dict(view.to_dict(), profile=profile.id, wire_api=wire)
        if wire == "openai_responses":
            body["models"] = [merge_model(model, native_codex_model(model.id)).to_dict() for model in view.models]
        return body

    async def check_agent_selection(self, value):
        wb = self.workbench
        if not isinstance(value, dict):
            raise ValueError("agent_selection_requires_object")
        backend, profile_id = value.get("backend"), value.get("profile")
        wire, model, effort = value.get("wire_api"), value.get("model"), value.get("effort")
        result = dict(
            ok=False,
            reason=None,
            message=None,
            wire_api=wire,
            reasoning=None,
            backend=backend,
            profile=profile_id,
            model=model,
            effort=effort,
        )
        if backend not in BACKENDS:
            return dict(
                result,
                reason="unknown_agent_backend",
                message=f"未知后端 {backend!r}；可选 {', '.join(BACKENDS)}",
            )
        if model is not None and (not isinstance(model, str) or not MODEL_ID.match(model)):
            return dict(
                result,
                reason="invalid_model",
                message="模型名格式无效：字母或数字开头，只含字母、数字和 . _ : / @ + - [ ]，最多 200 字符",
            )
        if profile_id is None:
            return dict(result, reason="profile_required", message="请选择 profile")
        availability = await asyncio.to_thread(runtime_problem, backend, wb.config)
        if availability is not None:
            return dict(
                result,
                reason=f"backend_unavailable:{availability}",
                message=f"{BACKEND_LABELS[backend]} 运行时不可用（{availability}）",
            )
        try:
            profile, mtime = await asyncio.to_thread(self._load_profile, profile_id)
        except web.HTTPNotFound:
            return dict(result, reason="profile_not_found", message=f"没有 profile {profile_id!r}")
        except ValueError as error:
            return dict(result, reason="profile_invalid:" + _code(error), message=str(error))
        thinking = _UNSET
        if backend == "opencode" and "thinking_display" in value:
            thinking = value["thinking_display"]
        check = await asyncio.to_thread(
            self._check_selection_cached,
            backend,
            str(profile.path),
            mtime,
            wire,
            model,
            effort,
            thinking,
        )
        result.update(
            ok=check.ok,
            reason=check.reason,
            wire_api=check.wire_api or wire,
            message=self._readable(backend, check.reason, check.message, profile) if check.reason else None,
        )
        selection = getattr(wb, "handoff_selection_blockers", None)
        refusals = selection(backend, profile, check.wire_api, model, effort, thinking) if check.ok and callable(selection) else []
        if refusals:
            code, detail = refusals[0]
            result.update(ok=False, reason=code, message=f"{HANDOFF_BLOCKERS.get(code, code)}（{detail}）")
        result["handoff"] = self.handoff_view(refusals)
        wire = result["wire_api"]
        if wire in profile.wires:
            current = self.current()
            live = (
                backend == current["backend"]
                and profile.id == current["profile"]
                and wire == current["wire_api"]
                and (model or current["model"]) == current["model"]
            )
            if thinking is _UNSET:
                thinking = (
                    profile.opencode_thinking_display
                    if (backend == "opencode" and wire == "anthropic_messages")
                    else None
                )
            model_id = model or current["model"]
            entry = self._catalogue_entry(self._peek_catalogue(profile.id, wire, mtime), model_id)
            result["reasoning"] = self._reasoning(
                backend,
                wire,
                entry,
                model_id,
                effort or (current["effort"] if live else None),
                thinking if backend == "opencode" else None,
                live=live,
            )
        return result

    def _profile_lock(self):
        wb = self.workbench
        lock = getattr(wb, "_agent_profile_lock", None)
        if lock is None:
            lock = wb._agent_profile_lock = asyncio.Lock()
        return lock

    def _forget_profile(self, profile_id):
        catalogue_cache(self.profiles_dir()).invalidate(profile_id)
        cache = self._check_cache()
        for key in [key for key in cache if _profile_id_of(key[1]) == profile_id]:
            cache.pop(key, None)

    async def save_profile_request(self, value, origin):
        wb = self.workbench
        if getattr(wb, "replay", False):
            raise ValueError("replay_is_read_only")
        if not isinstance(value, dict):
            raise ValueError("profile_request_requires_object")
        wb.authorize_writer(value, origin)
        profile_id, document = value.get("id"), value.get("document")
        directory = self.profiles_dir()
        async with self._profile_lock():
            location = profiles.profile_path(directory, profile_id)
            created = not location.exists()
            if document is None and "api_key" in value:
                if not isinstance(value["api_key"], str):
                    raise ValueError("profile_invalid_key: the key must be a string")
                info = await asyncio.to_thread(profiles.save_profile_key, directory, profile_id, value["api_key"])
                key_changed = True
            else:
                if not isinstance(document, dict):
                    raise ValueError("profile_document_required: send the profile document as a JSON object")
                keep_key = value.get("keep_key", True)
                if type(keep_key) is not bool:
                    raise ValueError("profile_invalid_keep_key: keep_key must be true or false")
                credential = document.get("credential")
                supplied = isinstance(credential, dict) and credential.get("api_key") not in (None, "")
                before = False
                if not created and not supplied and not keep_key:
                    try:
                        before = (await asyncio.to_thread(profiles.load_profile, location)).key_present
                    except ValueError:
                        before = False
                info = await asyncio.to_thread(profiles.save_profile, directory, profile_id, document, keep_key=keep_key)
                key_changed = supplied or before
            self._forget_profile(profile_id)
        view = self.profile_view(info, self._in_use_profiles())
        wb.store.event(
            "operator.profile_saved",
            dict(
                id=info.id,
                wires=sorted(info.wires),
                compat=dict(info.compat),
                key_changed=bool(key_changed),
                key_present=info.key_present,
                created=created,
                in_use=view["in_use"],
            ),
        )
        return view

    async def remove_profile_request(self, profile_id, value, origin):
        wb = self.workbench
        if getattr(wb, "replay", False):
            raise ValueError("replay_is_read_only")
        wb.authorize_writer(value if isinstance(value, dict) else {}, origin)
        directory = self.profiles_dir()
        async with self._profile_lock():
            target = await asyncio.to_thread(
                profiles.remove_profile,
                directory,
                profile_id,
                in_use=tuple(self._in_use_profiles()),
            )
            self._forget_profile(profile_id)
        wb.store.event("operator.profile_removed", dict(id=profile_id, renamed_to=target.name))
        return dict(status="removed", id=profile_id, renamed_to=target.name)

    @staticmethod
    def _validate_session_field(field, value):
        if field == "backend" and value not in BACKENDS:
            raise ValueError(f"invalid_agent_backend: use one of {', '.join(BACKENDS)}")
        if field == "profile" and (not isinstance(value, str) or not profiles.PROFILE_ID.fullmatch(value)):
            raise ValueError(f"profile_invalid_id: a profile id matches {profiles.PROFILE_ID.pattern}")
        if field == "wire_api" and value not in profiles.PROFILE_WIRES:
            raise ValueError(f"invalid_wire_api: use one of {', '.join(profiles.PROFILE_WIRES)}")
        if field == "thinking_display" and value is not None and value not in profiles.THINKING_DISPLAY_CHOICES:
            raise ValueError("invalid_thinking_display: null, " + " or ".join(profiles.THINKING_DISPLAY_CHOICES))

    async def apply_agent_settings(self, identifier, value):
        wb = self.workbench
        requested = {field: value[field] for field in (*SESSION_FIELDS, *LIVE_FIELDS) if field in value}
        if not requested:
            raise ValueError("apply_agent_settings_requires_a_setting: send model, effort and/or service_tier")
        for field in SESSION_FIELDS:
            if field in requested:
                self._validate_session_field(field, requested[field])
        current = self.current()
        if current["service_tier"] is None and requested.get("service_tier") == "default":
            current["service_tier"] = "default"
        diff = {
            field: dict(current=current.get(field), requested=requested[field])
            for field in (*SESSION_FIELDS, *LIVE_FIELDS)
            if field in requested and requested[field] != current.get(field)
        }
        session_fields = [field for field in SESSION_FIELDS if field in diff]
        if "effort" in diff and not wb._session_supports("live_effort"):
            session_fields.append("effort")
        if session_fields:
            labels = "、".join(FIELD_LABELS[field] for field in session_fields)
            receipt = dict(
                request_id=identifier,
                command="apply_agent_settings",
                status="handoff_required",
                reason="handoff_required",
                session_fields=session_fields,
                diff=diff,
                results={},
                handoff=dict(
                    command="handoff_run",
                    available=bool(wb.handoff_available()) if callable(getattr(wb, "handoff_available", None)) else False,
                ),
                message=f"{labels} 需要新会话：当前 run 不变，用 handoff_run 收尾并启动后继 run；本次未应用任何设置",
            )
            wb.store.event(
                "operator.agent_settings_applied",
                dict(
                    request_id=identifier,
                    status="handoff_required",
                    session_fields=session_fields,
                    diff=diff,
                    results={},
                    **self.event_context(),
                ),
            )
            return receipt
        appliers = dict(model=wb.apply_model, effort=wb.apply_effort, service_tier=wb.apply_service_tier)
        results, stopped, outcome_unknown = {}, None, False
        for field in LIVE_FIELDS:
            if field not in requested:
                continue
            if stopped is not None:
                results[field] = dict(
                    status="skipped",
                    reason="previous_setting_not_applied",
                    message=f"{FIELD_LABELS[stopped]}未应用，本项未执行",
                )
                continue
            if field not in diff:
                results[field] = dict(status="applied", changed=False, **{field: requested[field]})
                continue
            try:
                receipt = await appliers[field](identifier, requested[field])
            except (ValueError, KeyError, AgentRejected) as error:
                receipt = dict(status="rejected", reason=_code(error), message=str(error))
            except Exception as error:  # noqa: BLE001
                receipt, outcome_unknown = dict(status="unknown", reason=str(error)), True
            receipt = {key: item for key, item in receipt.items() if key != "request_id"}
            receipt.setdefault("changed", receipt.get("status") == "applied")
            results[field] = receipt
            if receipt.get("status") != "applied":
                stopped = field
        statuses = [result["status"] for result in results.values()]
        status = (
            "applied"
            if all(item == "applied" for item in statuses)
            else "partial"
            if "applied" in statuses
            else "unknown"
            if outcome_unknown
            else "rejected"
        )
        wb.store.event(
            "operator.agent_settings_applied",
            dict(request_id=identifier, status=status, results=results, **self.event_context()),
        )
        response = dict(request_id=identifier, command="apply_agent_settings", status=status, results=results)
        if outcome_unknown:
            wb.status = "unknown"
            response["http_status"] = 409
            wb.store.event(
                "control.status_unknown",
                dict(request_id=identifier, status="unknown", reason="apply_agent_settings_outcome_unknown"),
            )
        return response

    async def handoff_run(self, identifier, value):
        handoff = getattr(self.workbench, "request_handoff", None)
        if callable(handoff):
            return await handoff(identifier, value)
        raise ValueError("handoff_not_available: this gateway cannot start a successor run yet; nothing was changed")

    def mount(self, app):
        headers = {"Cache-Control": "no-store"}

        async def options(request):
            return web.json_response(
                await self.agent_options(refresh=request.query.get("refresh") == "1"),
                headers=headers,
            )

        async def models(request):
            body = await self.profile_models(
                request.match_info["id"],
                request.query.get("wire"),
                refresh=request.query.get("refresh") == "1",
            )
            return web.json_response(body, headers=headers)

        async def check(request):
            return web.json_response(await self.check_agent_selection(await request.json()), headers=headers)

        async def body_of(request):
            raw = await request.read()
            if len(raw) > _MAX_PROFILE_BODY_BYTES:
                raise ValueError("profile_request_too_large")
            try:
                return json.loads(raw or b"{}")
            except ValueError:
                raise ValueError("profile_request_invalid_json") from None

        async def save(request):
            info = await self.save_profile_request(await body_of(request), request.headers.get("Origin"))
            return web.json_response(info, headers=headers)

        async def remove(request):
            result = await self.remove_profile_request(
                request.match_info["id"],
                await body_of(request),
                request.headers.get("Origin"),
            )
            return web.json_response(result, headers=headers)

        app.router.add_get("/v1/console/agent-options", options)
        app.router.add_post("/v1/console/agent-options/check", check)
        app.router.add_get("/v1/console/profiles/{id}/models", models)
        app.router.add_post("/v1/console/profiles", save)
        app.router.add_post("/v1/console/profiles/{id}/remove", remove)


__all__ = [
    "AGENT_SETTINGS_SCHEMA",
    "AgentSettingsExtension",
    "BACKEND_LABELS",
    "FIELD_LABELS",
    "HANDOFF_BLOCKERS",
    "LIVE_FIELDS",
    "PRESETS_FILE",
    "PRESETS_SCHEMA",
    "SESSION_FIELDS",
    "SETTING_FIELDS",
    "_CHECK_CACHE_MAX",
    "_MAX_PROFILE_BODY_BYTES",
    "_UNSET",
    "_code",
    "_profile_id_of",
    "_write_json_atomic",
    "catalogue_cache",
    "read_presets",
    "reset_runtime_probes",
    "runtime_problem",
]
