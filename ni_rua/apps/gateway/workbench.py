"""Loopback workbench, persistent SSE, read-only PTY, and run supervision."""

import asyncio
import json
import secrets
import subprocess
import time
import fcntl
import hashlib
from pathlib import Path


from ...cortex.agents.base import (AgentCapabilityError, read_agent_identity, session_backend_kind, session_capabilities,
                                   session_turn_blocks_operator)
from ...cortex.agents.factory import (EVIDENCE_CLIENT_RELAY, EXTERNAL_BACKEND, agent_backend, backend_requirements,
                                      create_session, evidence_client)
from ...cortex.relay.server import DEFAULT_MAX_RETRIES, RETRY_POLICY
from ...core.contracts import canonical, new_id, digest, IDENTITY
from ...core.presentation.actor_payload import ACTOR_PAYLOAD_PROFILES
from ...runtime.harness import Harness, REMINDER_IN_COMPACT_REQUEST, REMINDER_IN_TOOL_RESULT
from ...runtime.tools.builtins import create_registry, CONFIG_TOOL_NAMES
from ...runtime.tools.plugins import configured_tool_specs
from ...robot.backends.factory import create_backend
from ...core.storage.store import Store
from ...cortex.session.terminal import Terminal
from ...runtime.programs.runner import ProgramRunner
from ...perception.scene.scene_service import SceneService, TERMINAL as JOB_TERMINAL
from ...cortex.session.usage import load_price_card
from ...core.observation_policy import ACTOR_PROFILES, LEGACY_ACTOR_PROFILE, actor_policy, has_calibration, allows_depth
from ...perception.depth.depth_runtime import INSTRUCTIONS as DEPTH_INSTRUCTIONS
from ...cortex.supervision.completion_guidance import (COMPLETION_GUIDANCE_MODES, COMPLETION_GUIDANCE_SOURCE,
                                  COMPLETION_GUIDANCE_TRIGGER, INITIAL_COMPLETION_GUIDANCE,
                                  hook_completion_guidance)
from ...cortex.context.task_notes import resolve_task_notes, embodiment_handoff_policy
from ...runtime.actions.home_check import HOME_CHECK_POLICY
from ...cortex.supervision.actor_stall import ACTOR_STALL_POLICY, ACTOR_PROGRESS_POLICY, ACTOR_EVIDENCE_POLICY
from ..library.scoring import SCORING_POLICY
from ...runtime.supervision import INACTIVITY_TIMEOUT_S
from ..evaluation.episode import BENCHMARK_CONTINUATION, record_evaluation, record_video

from ...core.config import FEATURE_ABSENT_VALUES, Config, cli_overrides, namespace_values
from ...core.storage.images import store_image_settings

from ...cortex.context.task_planning import PLANNING_VERSION
from ...cortex.context.public_task_rules import public_task_rules
from ...cortex.context.task_instruction_overrides import apply_task_instruction_override, task_instruction_override
from ...cortex.context.actor_instructions import (
    CALIBRATED_OBSERVATION_INSTRUCTIONS,
    EXTERNAL_URDF_ON_DEMAND_CLAUSE,
    EXTERNAL_URDF_ON_DEMAND_GUIDANCE,
    legacy_observation_instructions,
)
# archived_embodiment_record: (record_id, record) of a run's archived profile (defined next to profile_record).
from ...robot.contracts import ClockDomain, EvaluatorBackend, VideoBackend
from ...robot.model.embodiments import EmbodimentProfile, archived_embodiment_record
from .extensions import (
    AgentSettingsExtension,
    CalibrationExtension,
    HandoffExtension,
    OperatorConsoleExtension,
)
from .extensions.operator_console import LEASE_BUSY_REFUSAL, LEASE_COMMANDS, OPERATOR_COMMANDS
from .routes import GatewayRouter

try:
    from ...cortex.agents.codex.session import CODEX_CLI_VERSION, CODEX_SDK_VERSION, CodexSession
    _CODEX_CAPABILITIES = CodexSession.capabilities
except ModuleNotFoundError:
    from ...cortex.agents.base import AgentCapabilities

    CodexSession = None  # type: ignore[assignment]
    CODEX_SDK_VERSION = None
    CODEX_CLI_VERSION = None
    _CODEX_CAPABILITIES = AgentCapabilities(
        steer=True,
        native_requests=True,
        deferred_tools=True,
        account_relay=True,
        native_evidence_agent=True,
        sandbox=True,
        service_tier=True,
        live_effort=True,
        fork=True,
        terminal_attach=True,
        compaction_signal="relay",
    )

_UNSET = object()

_DEFAULT_PROMPT_HEAD = ("Development acceptance: inspect robot_capabilities and robot_observe. Using only those RGB "
                        "images and measured robot state, ")
_DEFAULT_PROMPT_MOTION = (" a small safe distance of at most 10 mm in free space, preserving orientation and gripper. "
                          "Set timeout_s=60 for this one movement. Execute at most one movement. Inspect the actual "
                          "final feedback and RGB. Report request versus measurement, errors, and any uncertainty. ")
# The tail per EmbodimentProfile.clock: a simulated RoboDojo world, or real hardware that keeps moving.
_DEFAULT_PROMPT_TAILS = {
    "simulated_steps": ("This is an interface smoke test, not completion of the RoboDojo task. Do not inspect "
                        "simulator task files or evaluation state."),
    "real_time": ("This is an interface smoke test on real hardware, not a task. The world keeps moving in real time: "
                  "re-observe before acting on an old image."),
}


def default_prompt(profile=None):
    """First actor turn of an interactive run started without a prompt, for ``profile`` (``EmbodimentProfile``;
    None = ARX-X5): one movement of at most 10 mm of one arm's TCP (the only arm is named when there is one), and the
    clock-specific tail (``profile.clock``). The ARX-X5 rendering is ``DEFAULT_PROMPT``."""
    if profile is None:
        from ...robot.model.embodiments import ARX_X5_PROFILE as profile
    arms = profile.arms
    motion = ("choose one arm and move its EE" if len(arms) > 1
              else f'move the TCP of the only arm "{arms[0]}"')
    return _DEFAULT_PROMPT_HEAD + motion + _DEFAULT_PROMPT_MOTION + _DEFAULT_PROMPT_TAILS[profile.clock]


#: ARX-X5 / RoboDojo rendering of ``default_prompt``. ``Workbench.start`` resolves this value (what
#: ``apps/gateway/server.py`` passes without ``--prompt``) and None to ``default_prompt`` of the run's embodiment.
#: File name of the robot URDF in the actor's working directory (``features.robot_urdf_delivery="file"``).
ROBOT_URDF_NAME = "robot.urdf"
#: The robot briefings' clause about inline URDF delivery, and its wording for file delivery.
URDF_INLINE_CLAUSE = "The original URDF is supplied inline,"
URDF_FILE_CLAUSE = f"The original URDF is supplied as the file {ROBOT_URDF_NAME} in your working directory,"
DEFAULT_PROMPT = (_DEFAULT_PROMPT_HEAD + "choose one arm and move its EE" + _DEFAULT_PROMPT_MOTION
                  + _DEFAULT_PROMPT_TAILS["simulated_steps"])

# Manifest clock_policy per EmbodimentProfile.clock (observer label; the capability document carries the prose).
CLOCK_POLICIES = {"simulated_steps": "paused_between_gateway_operations", "real_time": "real_time_not_paused"}


def _process_command_line(pid):
    """Command line of the live process ``pid``: the exact argv list from ``/proc/<pid>/cmdline`` (Linux), else the
    argv joined by spaces from ``ps -ww -o command= -p <pid>`` (macOS and other POSIX hosts). None when no such
    process is visible or the command line cannot be read."""
    proc = Path(f"/proc/{pid}/cmdline")
    if proc.exists():
        try:
            arguments = proc.read_bytes().decode(errors="replace").split("\0")
        except OSError:
            return None
        return [argument for argument in arguments if argument] or None
    try:
        result = subprocess.run(["ps", "-ww", "-o", "command=", "-p", str(int(pid))],
                                capture_output=True, text=True, timeout=5)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def owned_robot_worker_running(worker, default_run_dir):
    """True when the robot worker recorded in runtime state ``robot`` (``{"pid": int, "run_dir": str}``, written by
    ``Workbench.start`` for a backend with a worker process) is still running: a live process ``pid`` whose command
    line passes that run's private robot directory (``run_dir``, else ``default_run_dir``) as one argument. Both
    worker kinds take it (RoboDojo ``robot_worker`` and SO-101 ``worker``, ``--run-dir``), so no module name is
    assumed; a reused pid of another process has another command line. Unreadable or absent processes: False."""
    pid = (worker or {}).get("pid")
    if type(pid) is not int or pid <= 0:
        return False
    run_dir = str((worker or {}).get("run_dir") or default_run_dir)
    command = _process_command_line(pid)
    if isinstance(command, list):
        return run_dir in command
    return isinstance(command, str) and (f" {run_dir} " in command + " ")


def _configured_tool_specs(config, tool_specs=(), owner=None):
    """Compatibility wrapper forwarding to public ``ni_rua.runtime.tools.plugins.configured_tool_specs``."""
    return configured_tool_specs(config, tool_specs=tool_specs, owner=owner)


class Workbench:
    remote = None

    def __init__(self, root, task=_UNSET, gpu=_UNSET, max_actions=_UNSET, robot=None,
                 resume=_UNSET, replay=_UNSET, observation_profile=_UNSET, seed=_UNSET, layout_id=_UNSET,
                 price_card=None, pi05=_UNSET, visual_review_model=_UNSET, actor_payload_profile=_UNSET,
                 enable_wait=_UNSET, tracking_guard=_UNSET, joint_replan=_UNSET, completion_guidance=_UNSET,
                 motion_review=_UNSET, default_task_notes=_UNSET, vision_compare=_UNSET,
                 robotic_view=_UNSET, stereo_guide=_UNSET,
                 depth_skills=_UNSET, depth_auto_feedback=_UNSET, depth_event_payload=_UNSET, config=None, tool_specs=()):
        if observation_profile is not _UNSET and replay is not True:
            actor_policy(observation_profile)
        supplied = {key: value for key, value in locals().copy().items()
                    if value is not _UNSET and key in namespace_values(Config()) and key != 'price_card'}
        for name in ('enable_wait', 'tracking_guard', 'joint_replan', 'motion_review', 'pi05', 'default_task_notes', 'vision_compare',
                     'robotic_view', 'stereo_guide', 'depth_skills', 'depth_auto_feedback', 'depth_event_payload'):
            if name in supplied and type(supplied[name]) is not bool:
                raise ValueError('invalid_control_feature_flag')
        if actor_payload_profile is not _UNSET and actor_payload_profile not in ACTOR_PAYLOAD_PROFILES:
            raise ValueError('invalid_actor_payload_profile')
        if completion_guidance is not _UNSET and completion_guidance not in COMPLETION_GUIDANCE_MODES:
            raise ValueError('invalid_completion_guidance')
        selected = (config if config is not None else Config.load()).with_overrides(cli_overrides(supplied))
        requested_backend = selected.agent.backend
        resume, replay = selected.run.resume, selected.run.replay
        invocation_port = selected.run.port
        # The invocation's --prompt / run.prompt; an external agent's resumed run takes it as its new task (below).
        invocation_prompt = selected.run.prompt
        self.root = Path(root).resolve()
        occupied = self.root.exists() and any(path.name not in {'experiment.json', 'runtime.json'} for path in self.root.iterdir())
        if occupied and not (resume or replay):
            raise ValueError('run_directory_already_used: choose a fresh --run-dir or --resume')
        if (resume or replay) and not occupied:
            raise ValueError('resume_requires_existing_run')
        manifest = json.loads((self.root / 'manifest.json').read_text()) if (resume or replay) else {}
        if resume or replay:
            if manifest.get('effective_config') is not None:
                archived = dict(manifest['effective_config'])
                archived_features = dict(archived.get('features', {}))
                archived_features.setdefault('default_task_notes', False)
                archived_features.setdefault('vision_compare', True)
                archived_features.setdefault('actor_payload_profile', 'full_v1')
                archived_features.setdefault('completion_guidance', manifest.get('completion_guidance', {}).get('mode', 'none'))
                archived_features.setdefault('airborne_handoff', True)
                archived_features.setdefault('joint_replan', manifest.get('joint_replan', {}).get('enabled', False))
                for flag in ('bounded_continuation', 'motion_safety_guard', 'trajectory_preview', 'manipulation_protocol'):
                    archived_features.setdefault(flag, False)
                for flag in ('task_scheduling', 'operation_candidates', 'preview_intent', 'recovery_evidence', 'robotic_view', 'stereo_guide', 'self_briefing', 'startup_inline_grounding'):
                    archived_features.setdefault(flag, False)
                archived_features.setdefault('startup_grounding_views', 'overview')
                archived_features.setdefault('action_trajectory_delivery', 'automatic')
                # An archive written before the reply-delivery, tool catalogue or URDF delivery settings existed keeps
                # the form it ran with, key by key (tool_catalog_profile is older than robot_urdf_delivery).
                for name, absent in FEATURE_ABSENT_VALUES.items():
                    archived_features.setdefault(name, absent)
                archived['features'] = archived_features
                # 兼容未记录 action_feedback_policy 的旧归档，回退至其保存的 feedback_policy（默认 all）
                archived_images = dict(archived.get('images', {}))
                archived_images.setdefault('action_feedback_policy', archived_images.get('feedback_policy', 'all'))
                archived['images'] = archived_images
                # 兼容历史非 Codex 归档的 context_delivery 语义，避免翻默认或移除 subagents 硬覆写后触发 tools_digest 冲突：
                # 1) 缺少 context_delivery（含因默认省略 agent.external 的旧归档）：优先回填 manifest.agent_backend.context_delivery，缺失回退 "full"；
                # 2) 旧版 subagents=True 硬覆写期归档（configured="delta" 但实际生效 recorded_delivery="full"）：回填 "full"（现代归档两者一致不误命中）。
                archived_agent = dict(archived.get('agent', {}))
                archived_backend = archived_agent.get('backend', 'codex')
                if archived_backend in ('claude_code', 'opencode', 'external'):
                    archived_section = dict(archived_agent.get(archived_backend, {}))
                    recorded_delivery = (manifest.get('agent_backend') or {}).get('context_delivery')
                    if 'context_delivery' not in archived_section:
                        archived_section['context_delivery'] = (
                            recorded_delivery if recorded_delivery in ('full', 'delta') else 'full'
                        )
                        archived_agent[archived_backend] = archived_section
                        archived['agent'] = archived_agent
                    elif (archived_backend in ('claude_code', 'opencode')
                          and recorded_delivery == 'full'
                          and archived_section.get('subagents', True)
                          and archived_section.get('context_delivery') == 'delta'):
                        archived_section['context_delivery'] = 'full'
                        archived_agent[archived_backend] = archived_section
                        archived['agent'] = archived_agent
                selected = Config.from_dict(archived)
            else:
                # Translate the actual historical fields without consulting a new
                # config file for features absent from old experiments.
                saved = {'run': {key: manifest[key] for key in ('task', 'seed', 'layout_id', 'max_actions', 'benchmark', 'task_notes', 'task_notes_source') if key in manifest},
                         'features': dict(observation_profile=manifest.get('observation_profile') if manifest.get('observation_profile') in ACTOR_PROFILES else LEGACY_ACTOR_PROFILE,
                            motion_review=manifest.get('motion_review', False),
                            vision_compare=manifest.get('vision_compare', True),
                            default_task_notes=manifest.get('task_notes_config', {}).get('default_enabled', False),
                            actor_payload_profile=manifest.get('actor_payload_profile', 'full_v1'),
                            enable_wait=manifest.get('enable_wait', False),
                            tracking_guard=manifest.get('tracking_guard', {}).get('enabled', False),
                            joint_replan=manifest.get('joint_replan', {}).get('enabled', False),
                            airborne_handoff=True,
                            bounded_continuation=False, motion_safety_guard=False,
                            trajectory_preview=False, manipulation_protocol=False,
                            task_scheduling=False, operation_candidates=False,
                            preview_intent=False, recovery_evidence=False,
                            action_trajectory_delivery='automatic',
                            robotic_view=False, stereo_guide=False, self_briefing=False,
                            startup_inline_grounding=False,
                            completion_guidance=manifest.get('completion_guidance', {}).get('mode', 'none'),
                            pi05=manifest.get('pi05_enabled', False), visual_review_model=manifest.get('visual_review_model'),
                            **FEATURE_ABSENT_VALUES)}
                if manifest.get('grasp_memory_dir'):
                    saved['features']['grasp_memory_dir'] = manifest['grasp_memory_dir']
                if 'task_notes_config' in manifest:
                    saved['run']['task_notes'] = manifest['task_notes_config'].get('custom_text')
                    saved['run']['task_notes_source'] = manifest['task_notes_config'].get('custom_source')
                saved['run']['wall_budget_s'] = manifest.get('actor_wait_budget_s')
                saved['model'] = {key: manifest[old] for key, old in (('name', 'model'), ('effort', 'effort')) if manifest.get(old)}
                if manifest.get('image_settings'):
                    saved['images'] = dict(manifest['image_settings'])
                    saved['images'].setdefault('reuse_exact_images', False)
                # 早期无 effective_config 的归档保持原默认全图动作回执，避免新默认值覆盖历史回放语义
                saved.setdefault('images', {}).setdefault(
                    'action_feedback_policy',
                    saved.get('images', {}).get('feedback_policy', 'all'),
                )
                # Host paths can be selected on legacy runs that never recorded
                # deployment settings. Their experiment semantics stay frozen.
                saved.update(backend=selected.backend.to_dict(), policy=selected.policy.to_dict())
                saved['model']['codex_home'] = selected.model.codex_home
                selected = Config.from_dict(saved)
        selected = selected.with_overrides({'run': dict(run_dir=str(self.root), resume=resume, replay=replay, port=invocation_port)})
        if resume and invocation_prompt and agent_backend(selected) == EXTERNAL_BACKEND:
            # agent.backend "external": the task is what run_begin delivers, so a resume --prompt replaces the archived
            # task (manifest and briefing); without one the archived task stays. Hosted agents never get a task on
            # resume, so their archived configuration is kept unchanged.
            selected = selected.with_overrides({'run': dict(prompt=invocation_prompt)})
        if not has_calibration(selected.features.observation_profile):
            selected = selected.with_overrides({'features': {'motion_review': False, 'robotic_view': False, 'stereo_guide': False}})
        configured_specs = _configured_tool_specs(selected, tool_specs, self)
        unknown_tools = set(selected.features.disabled_tools) - create_registry(configured_specs).names - CONFIG_TOOL_NAMES
        if unknown_tools:
            raise ValueError('unknown_disabled_tools:' + ','.join(sorted(unknown_tools)))
        if not replay and "task_plan" in selected.features.disabled_tools:
            raise ValueError("required_task_plan_tool_cannot_be_disabled")
        if not replay:
            # Refuse a backend/feature combination before any run state, robot or native process exists.
            backend_requirements(selected)
        self.config = selected
        run, features = selected.run, selected.features
        task, gpu, seed, layout_id, max_actions = run.task, run.gpu, run.seed, run.layout_id, run.max_actions
        observation_profile = features.observation_profile
        actor_payload_profile, enable_wait = features.actor_payload_profile, features.enable_wait
        tracking_guard, joint_replan, motion_review = features.tracking_guard, features.joint_replan, features.motion_review
        pi05, visual_review_model, completion_guidance = features.pi05, features.visual_review_model, features.completion_guidance
        if not replay:
            actor_policy(observation_profile)
        if price_card is None and selected.model.price_card and not (resume or replay):
            price_card = load_price_card(selected.model.price_card)
        default_task_notes, vision_compare = features.default_task_notes, features.vision_compare
        self.default_task_notes = default_task_notes
        self.archived_task_notes = ({
            "text": manifest.get("task_notes"), "source": manifest.get("task_notes_source"),
            "config": manifest.get("task_notes_config", {
                "default_enabled": False, "default_version": None,
                "custom_text": manifest.get("task_notes"), "custom_source": manifest.get("task_notes_source"),
            }),
        } if resume or replay else None)
        self.archived_public_task_rules = manifest.get("public_task_rules") if resume or replay else None
        self.archived_instruction_override = manifest.get("task_instruction_override") if resume or replay else None
        self.archived_embodiment_record_id = manifest.get("embodiment_record_id") if resume or replay else None
        self.completion_guidance = completion_guidance
        self.initial_observation_id = None
        self.actor_payload_profile, self.enable_wait, self.tracking_guard = actor_payload_profile, enable_wait, tracking_guard
        self.joint_replan = joint_replan
        self.vision_compare = vision_compare
        self.observation_profile = observation_profile
        self.motion_review = motion_review
        if resume and (manifest.get("observation_profile") not in ACTOR_PROFILES or
                       manifest.get("actor_information_policy") != actor_policy(observation_profile)):
            raise ValueError("resume_requires_clean_actor_history: create a fresh run and thread")
        if resume:
            previous_card = manifest.get("price_card")
            if price_card is not None and price_card != previous_card:
                raise ValueError("resume_price_card_conflict: use offline report for a different schedule")
            price_card = previous_card
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_lock = (self.root/"gateway.lock").open("a")
        try: fcntl.flock(self.run_lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise ValueError("run_gateway_already_owned")
        self.resume, self.replay = resume, replay
        self.identity = {key:manifest[key] for key in IDENTITY} if (resume or replay) else dict(run_id=self.root.name, episode_id=new_id("ep"), env_instance_id=new_id("env"), env_generation=1)
        self.store = Store(self.root, self.identity)
        self.identity.update(self.store.read_state("runtime", "identity", self.identity))
        # Identity of the agent session that recorded this run (runtime/agent; runtime/codex for older archives).
        self.previous_agent = read_agent_identity(self.store) if resume else None
        recorded_backend = (self.previous_agent or {}).get("backend")
        # Both the backend this invocation asked for and the archived configuration must match the recorded session.
        if recorded_backend is not None and {requested_backend, self.config.agent.backend} != {recorded_backend}:
            self.store.close()
            self.run_lock.close()
            raise ValueError(f"resume_agent_backend_changed: the run was recorded with agent backend {recorded_backend}; "
                             f"resume with --agent-backend {recorded_backend} or start a fresh run")
        if resume:
            # The previous gateway's robot worker (any backend) must have exited before a new one owns the robot.
            if owned_robot_worker_running(self.store.read_state("runtime", "robot"), self.root / "private" / "robot"):
                self.store.close()
                self.run_lock.close()
                raise ValueError("resume_requires_previous_owned_robot_worker_exit")
            previous = dict(self.identity)
            self.identity.update(episode_id=new_id("ep"), env_generation=self.identity["env_generation"]+1)
            self.store.set_state("runtime", "identity", dict(self.identity))
            self.store.event("environment.recovery_reset", {"previous_identity":previous,"physics_restore":False})
        self.task, self.seed, self.layout_id = task, seed, layout_id
        self.pi05_enabled, self.gpu = pi05, gpu
        self.visual_review_model = visual_review_model
        # Snapshot every local implementation module, including optional adapters,
        # so adding a module cannot silently omit it from run provenance.
        source_paths = sorted(Path(__file__).parents[2].rglob("*.py"))
        source_paths += [Path(__file__).parents[2] / "cortex/context/robodojo_task_rules.json",
                         Path(__file__).parents[2] / "cortex/context/task_instruction_overrides.json",
                         Path(__file__).parents[2] / "cortex/session/codex_responses.json"]
        self.source_hashes = {str(path.relative_to(Path(__file__).parents[2])): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in source_paths}
        self.source_ref = None
        if not replay:
            sources = {name: (Path(__file__).parents[2] / name).read_text() for name in self.source_hashes}
            self.source_ref = self.store.put_artifact(canonical(sources).encode(), "runtime_source", "application/json", "observer")
        self.robot = robot if robot is not None else create_backend(task=task, gpu_id=gpu, seed=seed, layout_id=layout_id, run_dir=self.root / "private" / "robot", observation_profile=observation_profile, tracking_guard=tracking_guard, joint_replan=joint_replan, config=self.config)
        actual_images = store_image_settings(self.store, self.config.images.to_dict())
        self.config = self.config.with_overrides({"images": actual_images})
        self.grasp_memory_store = None
        if not replay:
            memory_path = Path(self.config.features.grasp_memory_dir).expanduser().resolve()
            self.grasp_memory_store = Store(memory_path, self.identity)
        self.harness = Harness(self.store, self.robot, max_actions, actor_payload_profile=actor_payload_profile, enable_wait=enable_wait,
                               observation_profile=observation_profile, motion_review=self.motion_review,
                               grasp_memory_store=self.grasp_memory_store, vision_compare=vision_compare, config=self.config,
                               task_plan_required=True, tool_specs=configured_specs)
        self.embodiment = self.harness.embodiment
        if resume: self.harness.accepting=True  # Fresh environment generation; old unknown receipts remain immutable.
        self.session = create_session(self.config, self.root, self.store, self.harness.handle_tool,
                                      price_card=price_card, task_planning=self.harness.task_planning)
        self._router = GatewayRouter(self)
        self._console_ext = OperatorConsoleExtension(self)
        self._calibration_ext = CalibrationExtension(self)
        self._agent_settings_ext = AgentSettingsExtension(self)
        self._handoff_ext = HandoffExtension(self)
        self.extensions = (
            self._console_ext,
            self._agent_settings_ext,
            self._calibration_ext,
            self._handoff_ext,
        )
        self.mounted_extensions = ()
        codex_backend = self._backend_kind() == "codex"
        self.harness.delivery_channel = "codex_tool" if codex_backend else "agent_tool"
        # Codex re-injects the decision context into its own compaction request; every other backend gets it from
        # the Harness on the next actor tool response after the session reports a context reset (design §4.5).
        self.harness.decision_reminder_channel = (REMINDER_IN_COMPACT_REQUEST if codex_backend
                                                  else REMINDER_IN_TOOL_RESULT)
        if not (resume or replay):
            (self.root / 'effective_config.json').write_text(json.dumps(self.config.to_dict(), indent=2) + '\n')
        # The host's own limits decide how actor replies are shaped (features.budgeted_action_reply and
        # reply_static_once read them per reply): its text limit for one tool output and whether it loads
        # deferLoading tools on demand.
        self.harness.action_reply_budget = self.session.tool_output_text_budget_bytes
        self.harness.host_truncation_limit = getattr(self.session, "host_tool_output_truncation_limit", None)
        self.harness.deferred_tool_loading = bool(
            session_capabilities(self.session, _CODEX_CAPABILITIES).deferred_tools
            or getattr(self.harness, "require_tool_search_unlock", False)
        )
        self.session.on_disconnect=self.harness.request_stop
        self.session.on_context_reset = self.harness.model_context_reset
        self.session.transport_recovery_gate = self.transport_recovery_block
        self.session.tool_parallel_safe = self.harness.tool_registry.parallel_safe
        # Declared effect per call (ToolRegistry.effect_of); read only by the external backend's session bracket.
        self.session.tool_effect = self.harness.tool_registry.effect_of
        self.session.embodiment = self.embodiment
        # Published ToolSpec.parallel per actor tool (True | False | "request_dependent"), for MCP readOnlyHint.
        self.session.tool_parallel_hints = {tool["name"]: tool["parallel"]
                                            for tool in self.harness.tool_registry.manifest(self.harness)["tools"]}
        self.terminal = None
        self.status = "starting"
        self.starting = True
        self.control_responses = {}
        self.control_lock = asyncio.Lock()
        self.scene_service = None
        self.scene_forwarder = None
        self.controller = None
        self.lease_client = None  # client_id of the lease's last holder (kept after expiry and release)
        self.loop = None
        self.closed = False

    def _embodiment(self):
        """This run's ``EmbodimentProfile``: ``harness.embodiment`` (resolved once by ``Harness.__init__``).
        Process-free harness doubles without it get the profile of the Harness this Workbench built, else ARX-X5."""
        for profile in (getattr(getattr(self, "harness", None), "embodiment", None), getattr(self, "embodiment", None)):
            if isinstance(profile, EmbodimentProfile):
                return profile
        from ...robot.model.embodiments import ARX_X5_PROFILE
        return ARX_X5_PROFILE

    def embodiment_record(self):
        """The archived ``embodiments.profile_record`` of this run's embodiment, or None (observer data, never model
        context). Replay: the record the archive names, never the live backend's profile (that follows the current
        configuration and deployment files); live and resumed runs: the record ``Harness.start`` saved."""
        return archived_embodiment_record(self.store, getattr(self.harness, "embodiment_record_id", None))[1]

    async def start(self, prompt, benchmark=False, task_notes=None, task_notes_source=None, wall_budget_s=None):
        """``prompt``: the first actor turn of an interactive run; None or ``DEFAULT_PROMPT`` (the gateway's value
        without ``--prompt``) is ``default_prompt`` of this run's embodiment. Benchmark runs build their own. With
        the external backend it is the operator's task for ``run_begin`` and None or ``DEFAULT_PROMPT`` means no task
        (never the development-acceptance default prompt, which moves the robot); a resumed external run ignores it and
        keeps its archived task, or the resume invocation's ``--prompt`` that replaced it."""
        external = session_backend_kind(getattr(self, "session", None)) == EXTERNAL_BACKEND
        if external and getattr(self, "resume", False) and hasattr(self, "config"):
            # The archived task, or the resume invocation's --prompt that replaced it in __init__.
            prompt = self.config.run.prompt
        elif prompt is None or prompt == DEFAULT_PROMPT:
            prompt = None if external else default_prompt(self._embodiment())
        if self.resume and hasattr(self, 'config'):
            benchmark = self.config.run.benchmark
            task_notes, task_notes_source = self.config.run.task_notes, self.config.run.task_notes_source
            wall_budget_s = self.config.run.wall_budget_s
        if not self.replay and hasattr(self, 'config'):
            self.config = self.config.with_overrides({'run': dict(prompt=None if benchmark else prompt, benchmark=benchmark,
                task_notes=task_notes, task_notes_source=task_notes_source, wall_budget_s=wall_budget_s)})
            (self.root / 'effective_config.json').write_text(json.dumps(self.config.to_dict(), indent=2) + '\n')
        if self.replay:
            self.starting=False
            self.status="replay"
            self.harness.accepting=False
            self.scene_service=SceneService(self.root/"twin-service", image_settings=getattr(self.harness, "image_settings", self.config.images.to_dict()), **{key: value for key, value in getattr(self, "config", Config()).scene.to_dict().items() if key not in {"root", "port"}})
            self.harness.scene=self.scene_service.scene
            observations=self.store.records("observation")
            self.harness.latest=observations[-1] if observations else None
            self.harness.capabilities=self.store.read_state("runtime","capabilities") or self.store.read("configuration","robot_capabilities")
            # The embodiment the run used is the archived record, not the live backend's profile.
            self.harness.embodiment_record_id = archived_embodiment_record(
                self.store, getattr(self, "archived_embodiment_record_id", None))[0]
            info=read_agent_identity(self.store) or {}
            enter_replay=getattr(self.session, "enter_replay", None)
            if callable(enter_replay):
                enter_replay(info.get("thread_id"))
            else:
                self.session.thread_id=info.get("thread_id")
                self.session.state="replay"
            self.terminal=Terminal(self.root,self.root)
            saved_terminal=json.loads((self.root/"terminal.json").read_text()) if (self.root/"terminal.json").exists() else {}
            self.terminal.session_id=saved_terminal.get("session_id",self.terminal.session_id)
            self.terminal.thread_id=self.session.thread_id
            self.terminal._closed=True
            return
        is_robodojo = (
            getattr(getattr(self, "config", None), "backend", None) is not None
            and self.config.backend.environment == "robodojo"
        )
        # 外部模式虽禁用 --benchmark，但运行 RoboDojo 时仍需向 run_begin 透传原生任务指令、公开任务规则与操作笔记
        use_native_task_context = benchmark or (external and is_robodojo)
        notes = self.archived_task_notes if self.resume else resolve_task_notes(
            default_enabled=use_native_task_context and self.default_task_notes,
            custom_text=task_notes, custom_source=task_notes_source,
            manipulation_protocol=self.config.features.manipulation_protocol,
            task_scheduling=self.config.features.task_scheduling, operation_candidates=self.config.features.operation_candidates)
        embodiment = self._embodiment()
        from ...cortex.context.actor_instructions import task_notes as current_task_notes
        if not self.resume:
            notes = current_task_notes(notes, profile=embodiment)
        task_notes, task_notes_source = notes["text"], notes["source"]
        # None for an embodiment without a cross-arm handoff: no section, prompt-record or manifest policy.
        handoff = embodiment_handoff_policy(embodiment, self.config.features.airborne_handoff,
                                            handoff_preference=self.config.features.handoff_preference)
        rules = (self.archived_public_task_rules if self.resume
                 else public_task_rules(self.task) if use_native_task_context else None)
        instruction_override = (self.archived_instruction_override if self.resume
                                else task_instruction_override(self.task, self.config.features.task_instruction_overrides)
                                if use_native_task_context else None)
        self.loop = asyncio.get_running_loop()
        self.self_briefing = None
        self.startup_grounding = None
        try:
            if isinstance(self.harness, Harness):
                # Scene reconstruction is not implemented for actor tasks. Keep the
                # local geometry store, but do not start an unused scene service.
                self.harness.programs = ProgramRunner(self.harness.program_services())
            if not self.harness.accepting:
                return
            await asyncio.to_thread(self.harness.start)
            self.initial_observation_id = (getattr(self.harness, "latest", None) or {}).get("observation_id")
            # A handoff-born run re-applies the predecessor's operator speed cap before any agent turn (handoff.py).
            await self.apply_handoff_carry()
            if self.pi05_enabled:
                from ...runtime.policy.action_policy import OpenPIProcessPolicy
                self.harness.action_policy = await asyncio.to_thread(OpenPIProcessPolicy,
                    Path(self.config.policy.python),
                    Path(self.config.policy.checkpoint),
                    gpu=self.gpu, log_path=self.root / "private/pi05.log", timeout=120,
                    server_url=self.config.policy.server_url)
            native_task_context = None
            if benchmark:
                instruction = self.harness.latest.get("task_instruction")
                if not instruction: raise ValueError("native_task_instruction_missing")
                instruction, rules_text = apply_task_instruction_override(instruction, rules, instruction_override)
                env_name = (
                    "LIBERO"
                    if getattr(getattr(self, "config", None), "backend", None) is not None
                    and self.config.backend.environment == "libero"
                    else "RoboDojo"
                )
                prompt = (f"Complete the following native {env_name} task: {instruction}\n"
                          "Use permitted RGB, measured state and tools; inspect actual feedback. "
                          "The environment ends execution automatically on its terminal condition. "
                          "Do not read simulator files or evaluator state.")
                if rules_text:
                    prompt += "\n" + rules_text
                prompt += "\nCurrent native execution budget: " + canonical(self.harness.action_budget())
            elif external:
                native_instruction = (getattr(self.harness, "latest", None) or {}).get("task_instruction")
                if native_instruction:
                    eff_instruction, rules_text = apply_task_instruction_override(
                        native_instruction, rules, instruction_override
                    )
                    native_task_context = dict(
                        task=self.task,
                        instruction=eff_instruction,
                        public_task_rules=rules_text,
                    )
            has_home_check = embodiment.home_check is not None
            initial_guidance = (INITIAL_COMPLETION_GUIDANCE
                                if benchmark and not self.resume and has_home_check and self.completion_guidance in {"initial", "both"}
                                else None)
            if initial_guidance:
                prompt = (prompt + f"\nPublic completion guidance: {initial_guidance}") if prompt else f"Public completion guidance: {initial_guidance}"
            if task_notes and not external:
                prompt = (prompt + "\nTask notes:\n" + task_notes) if prompt else ("Task notes:\n" + task_notes)
            home_check = getattr(self.harness, "home_check", None)
            state_aware_hook = has_home_check and self.completion_guidance in {"hook", "both"}
            guidance_record = {
                "mode": self.completion_guidance, "source": COMPLETION_GUIDANCE_SOURCE,
                "initial_trigger": "initial_actor_prompt" if initial_guidance else None,
                "initial_prompt": initial_guidance,
                "initial_observation_id": self.initial_observation_id,
                "continuation_trigger": COMPLETION_GUIDANCE_TRIGGER if benchmark else None,
                "continuation_condition": "actor_turn_completed_before_native_terminal" if benchmark else None,
                "continuation_policy": HOME_CHECK_POLICY if benchmark and state_aware_hook else "generic",
                "home_reference": home_check.reference_record() if home_check is not None else None,
                # The selected prompt depends on the current measured state;
                # actual decisions and delivery attempts are recorded per turn.
                "continuation_prompt": BENCHMARK_CONTINUATION if benchmark and not state_aware_hook else None,
                "continuation_prompts": {
                    "away": hook_completion_guidance(self.initial_observation_id),
                    "home": BENCHMARK_CONTINUATION, "unknown": BENCHMARK_CONTINUATION,
                } if benchmark and state_aware_hook else None}
            if benchmark and self.config.features.bounded_continuation:
                from ...cortex.supervision import recovery_guidance as recovery_module
                recovery_version = (recovery_module.EVIDENCE_VERSION if self.config.features.recovery_evidence
                                    else recovery_module.VERSION)
                recovery_instructions = (recovery_module.EVIDENCE_INSTRUCTIONS if self.config.features.recovery_evidence
                                         else recovery_module.INSTRUCTIONS)
                guidance_record["recovery_guidance"] = dict(version=recovery_version,
                    instructions=recovery_instructions, context="current actor observation and native budget; measured home status")
            prompt_record = {"prompt": prompt, "native_instruction": (getattr(self.harness, "latest", None) or {}).get("task_instruction"), "benchmark": benchmark,
                             "public_task_rules": rules,
                             "task_instruction_override": instruction_override,
                             "task_planning": {"version": PLANNING_VERSION, "required_before_robot_actions": True,
                                               "tool_choices": "advisory", "revisions": "actor_discretion"},
                             "task_notes": task_notes, "task_notes_source": task_notes_source,
                             "task_notes_config": notes["config"],
                             "handoff_policy": handoff,
                             "completion_guidance": guidance_record}
            self.store.save("configuration", "task_prompt-" + digest(prompt_record), prompt_record)
            if not self.harness.accepting:
                return
            if getattr(self.robot, "process", None):
                self.store.set_state("runtime", "robot", {"pid":self.robot.process.pid,"run_dir":str(self.root/"private"/"robot")})
            # Published Harness source for every backend (design R4.4): mounted at /harness by the sandbox, named by
            # the instructions below otherwise. CodexSession._prepare refreshes the same copy.
            from ...cortex.session.actor_source import snapshot_actor_source
            (self.root / "private").mkdir(mode=0o700, exist_ok=True)
            snapshot_actor_source(self.root / "private" / "actor-source")
            from ...cortex.context.actor_instructions import instructions as actor_instructions
            observation_contract = (DEPTH_INSTRUCTIONS if allows_depth(self.observation_profile) else
                CALIBRATED_OBSERVATION_INSTRUCTIONS if has_calibration(self.observation_profile)
                else legacy_observation_instructions(embodiment))
            backend_kind = agent_backend(self.config)
            oc_cfg = getattr(getattr(self.config, "agent", None), "opencode", None)
            allow_shell = bool(getattr(oc_cfg, "allow_shell", True)) if backend_kind == "opencode" and oc_cfg is not None else True
            allow_external_read = (
                bool(getattr(oc_cfg, "allow_external_read", True)) if backend_kind == "opencode" and oc_cfg is not None else True
            )
            instructions = actor_instructions(observation_contract, handoff["text"] if handoff else None,
                source_root=None if self.config.features.sandbox_enabled else self.root / "private" / "actor-source",
                operation_memory="operation_memory" not in self.config.features.disabled_tools,
                embodiment_motion=(self.robot.actor_motion_instructions()
                                   if callable(getattr(self.robot, "actor_motion_instructions", None)) else None),
                profile=embodiment, external_agent=backend_kind == EXTERNAL_BACKEND,
                allow_shell=allow_shell, allow_external_read=allow_external_read)
            if self.visual_review_model and not {"vision_review", "operation_verify"}.intersection(self.config.features.disabled_tools):
                from ...cortex.context.operation_verification import INSTRUCTIONS as verification_instructions
                instructions += "\n" + verification_instructions.replace("operation_verify", "vision_review(evidence_mode='operation')")
            from ...perception.external.obs3d import developer_instructions as obs3d_instructions
            from ...perception.external import pose_skill, held_perception
            profile = getattr(self.harness, "embodiment", None)
            instructions += obs3d_instructions(self.config.features, profile)
            instructions += pose_skill.instructions(self.config.features, profile)
            held = held_perception.instruction(self.config.features, profile)  # features.held_perception: one sentence
            if held:
                instructions += "\n" + held
            if self.config.images.feedback_policy == "adaptive":
                instructions += "\nAutomatic RGB may omit unchanged views; this does not prove unchanged object state. Read a needed saved view with history_view_observation."
            # Automatic initialization uses the same capability delivery path as a read.
            # 启动拼装在 session.start() 之前执行（尚无真实 thread_id），显式标记 isolated 避免写入 thread_id=None 的脏基线。
            if isinstance(self.harness, Harness):
                capability_result = self.harness.invoke("robot_capabilities", {"detail": "summary"},
                    {"tool_call_id": "initialize-capabilities", "context_scope": "isolated"}, role="actor")
                capability_result = self.harness.actor_output.present(capability_result)
                instructions += "\nRobot runtime capabilities\n" + capability_result["contentItems"][0]["text"]
                self.store.event("robot_capabilities.initialized", {"delivered": True})
            # Read the same robot-only API exposed to the actor. This costs no
            # actor turn, advances no simulation time and opens no task assets.
            if callable(getattr(self.harness, "robot_information", None)):
                robot_information = self.harness.robot_information()
            else:
                from ...robot.model.robot_information import robot_information as read_robot_information
                robot_information = read_robot_information(
                    (self.harness.capabilities or {}).get("robot_model_id") or "unknown")
            self.store.set_state("runtime", "robot_information", robot_information)
            self.store.event("robot_information.initialized", dict(
                robot_id=robot_information.get("robot_id"), available=robot_information.get("available"),
                section="default", order=["robot_information", "harness_instructions", "task", "task_notes"]))
            if isinstance(self.harness, Harness):
                robot_information = self.harness.actor_output.present(robot_information)
            from ...robot.model.robot_information import robot_information_text
            geometry_guidance = ("The original URDF is supplied above. Read its STL/GLB meshes from the native ZIP with client.artifact using resource_ref when you need surface geometry. "
                                 if "urdf" in robot_information else "")
            # features.robot_urdf_delivery="file": 托管模式写入 actor 工作目录 robot.urdf；外部模式无托管目录，
            # 改为省略内联大 URDF 并引导按需调用只读工具 robot_information（绝不暴露任何本地路径）。
            if "urdf" in robot_information and self.config.features.robot_urdf_delivery == "file":
                if external:
                    robot_information = {key: value for key, value in robot_information.items() if key != "urdf"}
                    briefing = robot_information.get("briefing")
                    if isinstance(briefing, str) and briefing.count(URDF_INLINE_CLAUSE) == 1:
                        robot_information["briefing"] = briefing.replace(URDF_INLINE_CLAUSE, EXTERNAL_URDF_ON_DEMAND_CLAUSE)
                    geometry_guidance = EXTERNAL_URDF_ON_DEMAND_GUIDANCE
                elif self._write_actor_file(ROBOT_URDF_NAME, robot_information["urdf"]):
                    robot_information = {key: value for key, value in robot_information.items() if key != "urdf"}
                    # The briefing's own statement of where the URDF is (one sentence; left alone when a briefing words
                    # it differently, where it then makes no claim about inline delivery this code could falsify).
                    briefing = robot_information.get("briefing")
                    if isinstance(briefing, str) and briefing.count(URDF_INLINE_CLAUSE) == 1:
                        robot_information["briefing"] = briefing.replace(URDF_INLINE_CLAUSE, URDF_FILE_CLAUSE)
                    geometry_guidance = (f"The original URDF (metres, radians, kilograms) is the file {ROBOT_URDF_NAME} in your working directory; read it when you need joint or inertial records. "
                                         "Read its STL/GLB meshes from the native ZIP with client.artifact using resource_ref when you need surface geometry. ")
            instructions = (robot_information_text(robot_information)
                + "\n\nHarness instructions\n" + instructions
                + "\n" + geometry_guidance
                + "Calibration, when available, is anchored to its observation, not a substitute for current measured motion. robot_information can repeat these inputs, calibration rereads the latest published camera/base calibration, and section=all returns the detailed physical reference. "
                  "geometry_image is an optional illustrative drawing, not live visual evidence.")
            # 在启动 MCP 端点前注入 briefing 上下文与首轮接地回调，避免 --resume 或首条 run_begin 竞争
            if callable(getattr(self.session, "set_briefing_context", None)):
                self.session.set_briefing_context(native_task=native_task_context, task_notes=task_notes)
            if (external and isinstance(self.harness, Harness)
                    and callable(getattr(self.session, "set_startup_grounding_provider", None))):
                def _external_startup_grounding(correlation, *, opened):
                    # 严格 5 重门禁：仅新鲜模拟时钟首次 run_begin 且动作前、最新观测仍为 obs_1 时内联首轮接地
                    if not getattr(self.config.features, "startup_inline_grounding", False) or self.resume:
                        return None
                    if self.harness.embodiment.clock != "simulated_steps":
                        return None
                    if not opened or getattr(self.session, "_startup_grounding_delivered", False):
                        return None
                    if self.harness.action_count != 0:
                        return None
                    latest = getattr(self.harness, "latest", None)
                    if latest is None or latest.get("observation_id") != self.initial_observation_id:
                        return None
                    from ...cortex.context import startup_grounding as grounding
                    grounded = grounding.build_startup_grounding(
                        self.harness,
                        bundle=latest,
                        correlation=correlation,
                        include_images=True,
                    )
                    self.startup_grounding = grounded.record
                    self.store.save("configuration", "startup_grounding-" + digest(grounded.record), grounded.record)
                    return grounded
                self.session.set_startup_grounding_provider(_external_startup_grounding)
            # features.tool_catalog_profile="layered_v1": the full text of the fields the descriptions list with
            # type and bounds only (catalog_guide.reference_document), readable in the actor's working directory.
            from ...runtime.tools.builtins import catalog_profile_of
            from ...runtime.tools.catalog_guide import LAYERS, REFERENCE_NAME, on_demand_index, reference_document
            layer = LAYERS.get(catalog_profile_of(self.config))
            if layer is not None:
                reference = reference_document(self.actor_tools(), embodiment.robot_model_id)
                if reference is not None and not self._write_actor_file(REFERENCE_NAME, reference):
                    # The descriptions name this file; a session without a working directory must not start with them.
                    raise ValueError("tool_reference_requires_actor_directory")
                # A host that loads deferLoading tools on demand shows the model none of them; name them once.
                show_index = (layer["index"] and self.harness.deferred_tool_loading) or getattr(
                    self.harness, "require_tool_search_unlock", False
                )
                index = on_demand_index(self.actor_tools()) if show_index else None
                if index is not None:
                    instructions += "\n\n" + index
            elif getattr(self.harness, "require_tool_search_unlock", False):
                index = on_demand_index(self.actor_tools())
                if index is not None:
                    instructions += "\n\n" + index
            # A recorded identity without a thread (empty native record) starts a fresh thread, as before.
            if self.resume and (self.previous_agent or {}).get("thread_id"):
                self.harness.thread_id = await asyncio.to_thread(self.session.start, self.actor_tools(), instructions, self.previous_agent["thread_id"])
            else:
                self.harness.thread_id = await asyncio.to_thread(self.session.start, self.actor_tools(), instructions)
            self.harness.model_proxy = getattr(self.session, "model_proxy", None)
            # Forced full delivery (design §4.3) for a backend that cannot yet prove what its model still holds;
            # read after start() so a session may settle its delivery mode while starting. Codex: "delta".
            self.harness.context_epoch_volatile = getattr(self.session, "context_delivery", "delta") == "full"
            if isinstance(self.harness, Harness) and not self.harness.context_epoch_volatile and not external:
                # 仅在托管 delta 模式下于启动阶段预播种能力基线；full 模式每轮自包含，external 延后至 run_begin 送达简报时播种。
                self.harness._capability_delivery.project(self.harness.capabilities,
                    epoch=self.harness._model_context_epoch(), current_action_budget=self.harness.action_budget())
            if not self.harness.accepting:
                return
            if self.visual_review_model:
                from ...cortex.session.visual_review import VisualReviewer
                options = {}
                if self._evidence_client() == EVIDENCE_CLIENT_RELAY:
                    # agent.evidence.upstream (when set) becomes the relay's "evidence" binding (design §9.4).
                    from ...cortex.agents.evidence import bind_evidence_upstream
                    upstream = bind_evidence_upstream(self.session, self.config)
                    if upstream != "actor":
                        options["upstream"] = upstream
                self.harness.visual_reviewer = VisualReviewer(self._evidence_agent(**options))
            startup_images = ()
            startup_grounded = False
            from ...cortex.context import startup_grounding as grounding
            if (benchmark and not self.resume and isinstance(self.harness, Harness)
                    and bool(getattr(session_capabilities(self.session, None), "turn_images", False))
                    and getattr(self.config.features, "startup_inline_grounding", False)):
                supports_images_fn = getattr(self.session, "supports_turn_images", None)
                include_images = bool(supports_images_fn()) if callable(supports_images_fn) else False
                # 首轮统一内联 obs_1 结构化 3D 状态；按当前会话与模型能力附带概览/全相机图
                grounded = await asyncio.to_thread(
                    grounding.build_startup_grounding,
                    self.harness,
                    include_images=include_images,
                )
                prompt = prompt + grounded.section_text
                startup_images = grounded.image_items
                startup_grounded = True
                grounding_record = dict(grounded.record, actor_prompt=prompt)
                self.startup_grounding = grounded.record
                self.store.save("configuration", "startup_grounding-" + digest(grounding_record), grounding_record)
                if not self.harness.accepting:
                    return
            # An empty thread has no rollout; start the task once before attaching TUI.
            if not self.resume:
                try:
                    if startup_images:
                        turn_id = await asyncio.to_thread(self.session.run_turn, prompt, images=startup_images)
                    else:
                        turn_id = await asyncio.to_thread(self.session.run_turn, prompt)
                except BaseException:
                    if startup_grounded and isinstance(self.harness, Harness):
                        self.harness._invalidate_actor_delivery("startup-grounding")
                    raise
                if turn_id is not None:   # the external backend admits no turn here (its agent opens it)
                    self.record_turn_settings(turn_id, "startup")  # run.turn_settings (observer data, best effort)
            if not self.harness.accepting and not getattr(self.harness, "episode_terminal", None):
                return
            attach = self._terminal_attach_spec() if getattr(self, 'config', Config()).features.terminal else None
            if attach is not None:
                # ``command`` (complete argv) is present only for backends whose TUI is not Codex's
                # ``resume --remote``; the Codex construction stays exactly as before.
                command = {"command": attach["command"]} if attach.get("command") is not None else {}
                self.terminal = Terminal(self.root, attach["cwd"], command_prefix=attach["command_prefix"],
                                         env=attach["env"], **command)
                await self.terminal.start(attach["attach_uri"], attach["thread_id"])
            if not self.harness.accepting and not getattr(self.harness, "episode_terminal", None):
                return
            self.status = "running"
            codex_backend = self._backend_kind() == "codex"
            capabilities = self.harness.capabilities or {}
            manifest = dict(schema_version="0.1", **self.identity, data_origin=capabilities.get("data_origin") or "unknown",
                            actor_information_policy=actor_policy(self.observation_profile),
                            motion_review=self.motion_review, vision_compare=self.vision_compare,
                            task_planning=dict(version=PLANNING_VERSION, required_before_robot_actions=True,
                                               tool_choices="advisory", revisions="actor_discretion"),
                            grasp_memory_dir=str(self.grasp_memory_store.root) if self.grasp_memory_store else None,
                            observation_profile=(getattr(self.harness,"latest",None) or {}).get("observation_profile", "synthetic_fixture"),
                            clock_policy=CLOCK_POLICIES[embodiment.clock],
                            model=self.session.model, provider=self.session.provider, effort=self.session.effort,
                            price_card=self.session.price_card, codex_sdk=CODEX_SDK_VERSION if codex_backend else None,
                            codex_cli=CODEX_CLI_VERSION if codex_backend else None,
                            codex_cli_executable=str(self.session.cli_executable) if codex_backend else None,
                            capabilities=self.harness.capabilities,
                            image_settings=self.store.read_state("runtime", "image_settings"),
                            model_image_policy=self.store.read_state("runtime", "model_image_policy"),
                            actor_payload_profile=self.actor_payload_profile, enable_wait=self.enable_wait,
                            # RoboDojo worker switches as the backend applies them (robot.tracking_guard /
                            # robot.joint_replan, passed to its worker); SO-101 and robot doubles have neither.
                            tracking_guard={"enabled": getattr(self.robot, "tracking_guard", None) is True,
                                            "threshold_rad": .5,
                                            "consecutive_native_steps": 5, "scope": "active_arm_ee_motion"},
                            joint_replan={"enabled": getattr(self.robot, "joint_replan", None) is True,
                                          "scope": "backend_free_ee",
                                          "candidate": "nearest_limit_valid_periodic_endpoint",
                                          "selection": "native_success_and_shorter_resampled_path",
                                          "minimum_remaining_wall_s": 15},
                            task=self.task, seed=self.seed, layout_id=self.layout_id,
                            max_actions=self.harness.max_actions if isinstance(self.harness, Harness) else None,
                            actor_wait_budget_s=wall_budget_s,
                            # The backend's native action protocol: its limits profile (RoboDojo robodojo-native-v1).
                            runtime_protocol=embodiment.limits_profiles[0], model_request_limit=180,
                            model_request_count_basis="delivered usable completed actor, visual-helper and compaction responses; failed/incomplete attempts excluded",
                            model_attempt_max_retries=DEFAULT_MAX_RETRIES,
                            model_request_timeout_s=None, model_auto_retries=DEFAULT_MAX_RETRIES, model_retry_policy=RETRY_POLICY,
                            no_progress_timeout_s=INACTIVITY_TIMEOUT_S, startup_timeout_s=900,
                            progress_clock="last_model_code_robot_or_simulator_activity",
                            read_only_queries_between_actions=None,
                            pi05_enabled=self.pi05_enabled, visual_review_model=self.visual_review_model,
                            source_sha256=self.source_hashes, benchmark=benchmark,
                            benchmark_terminal_policy="native_terminal_and_task_limits_v3" if benchmark else None,
                            actor_stall_policy=ACTOR_STALL_POLICY if benchmark else None,
                            actor_progress_policy=(ACTOR_EVIDENCE_POLICY if self.config.features.recovery_evidence else ACTOR_PROGRESS_POLICY)
                                if benchmark and self.config.features.bounded_continuation else None,
                            scoring_policy=SCORING_POLICY,
                            completion_guidance=guidance_record,
                            source_ref=self.source_ref,
                            task_notes=task_notes, task_notes_source=task_notes_source,
                            task_notes_config=notes["config"],
                            handoff_policy=handoff,
                            public_task_rules=rules,
                            task_instruction_override=instruction_override,
                            thread_id=self.session.thread_id,
                            app_server_pid=self.session.process.pid if codex_backend else None,
                            terminal_session_id=self.terminal.session_id if self.terminal else None,
                            terminal_pid=self.terminal.process.pid if self.terminal else None,
                            effective_config=getattr(self, "config", Config()).to_dict(),
                            tool_catalog=self.harness.tool_registry.manifest(self.harness),
                            local_scene=self.harness.scene.capabilities())
            if not codex_backend:
                # Codex-only provenance keys are absent for other backends; agent_backend carries theirs.
                for key in ("codex_sdk", "codex_cli", "codex_cli_executable", "app_server_pid"):
                    manifest.pop(key)
            for key in getattr(self.session, "manifest_null_fields", ()):
                # Host-mode values that do not describe this backend (external: no model, relay or inactivity stop).
                manifest[key] = None
            manifest["agent_backend"] = self._agent_provenance()
            if self.visual_review_model:
                # Effective evidence helper client (design §9.6); absent without an evidence feature, so
                # default Codex manifests keep their key set.
                manifest["evidence_client"] = self._evidence_client()
            if getattr(self, "handoff_link", None):
                # How this run was started by a handoff: predecessor, request, fallback reason (observer data).
                manifest["handoff"] = dict(self.handoff_link)
            record_id = getattr(self.harness, "embodiment_record_id", None)
            if record_id is not None:
                # The archived embodiment profile of this start (Harness.start); harness doubles save none.
                manifest["embodiment_record_id"] = record_id
            if self.startup_grounding is not None:
                manifest["startup_grounding"] = self.startup_grounding
            self.store.set_state("runtime", "identity", dict(self.identity))
            (self.root / "manifest.json").write_text(json.dumps(manifest, indent=2))
            self.store.event("run.started", {"manifest_ref": "manifest.json"}, {"thread_id": self.session.thread_id})
        finally:
            try:
                if not self.harness.accepting and not getattr(self.harness, "episode_terminal", None):
                    self.status = "unknown"
                    try:
                        await asyncio.to_thread(self.robot.close)
                    finally:
                        try:
                            if self.terminal:
                                await self.terminal.close()
                        finally:
                            await asyncio.to_thread(self.session.close)
                    unresolved = self.current_unknown_actions()
                    self.status = "unknown" if self.harness.active_action or unresolved else "stopped"
                    self.store.event("run." + self.status, {"reason": "stop_during_startup"})
            finally:
                self.starting = False

    # ---- agent session protocol, read tolerantly -------------------------------------------------------------
    # Sessions implement ni_rua.cortex.agents.base.AgentSession. Duck-typed, Codex-shaped sessions (test fakes,
    # pre-protocol objects) lack the protocol members; they are served by the Codex implementation of the member,
    # i.e. exactly the attribute probing the gateway used before the protocol existed.

    def _backend_kind(self):
        """The session's backend kind; duck-typed sessions without ``backend_kind`` are Codex-shaped."""
        return session_backend_kind(self.session)

    def _session_capability(self, name):
        """``session.capabilities.<name>``; duck-typed sessions without capabilities get the Codex value."""
        return bool(getattr(session_capabilities(self.session, CodexSession.capabilities), name, False))

    def _terminal_attach_spec(self):
        """Live native TUI attachment, or None when the backend has no terminal (skipped, not an error)."""
        if not self._session_capability("terminal_attach"):
            return None
        spec = getattr(self.session, "terminal_attach_spec", None)
        return spec() if callable(spec) else CodexSession.terminal_attach_spec(self.session)

    def _owned_session_processes(self):
        owned = getattr(self.session, "owned_processes", None)
        return list(owned()) if callable(owned) else CodexSession.owned_processes(self.session)

    def _evidence_client(self):
        """Effective evidence helper client of this run: "codex_agent" (Codex default) or "relay"."""
        return evidence_client(self.config, self._backend_kind())

    def _evidence_agent(self, **options):
        """Evidence helper for visual review / self-briefing from ``session.evidence_agent(**options)``.

        Codex: ``CodexEvidenceAgent`` unless ``agent.evidence.client="relay"``; other backends: ``RelayEvidenceClient``.
        Duck-typed, Codex-shaped sessions without the protocol member get the Codex selection for this run's config.
        """
        factory = getattr(self.session, "evidence_agent", None)
        if callable(factory):
            return factory(**options)
        if self._evidence_client() == EVIDENCE_CLIENT_RELAY:
            from ...cortex.agents.evidence import RelayEvidenceClient
            return RelayEvidenceClient(self.session, **options)
        from ...cortex.session.codex_evidence_agent import CodexEvidenceAgent
        return CodexEvidenceAgent(self.session, **options)

    def _agent_provenance(self):
        provenance = getattr(self.session, "provenance", None)
        return provenance() if callable(provenance) else CodexSession.provenance(self.session)

    def _write_actor_file(self, name, text):
        """Write the reference file ``name`` (UTF-8 ``text``) into the actor's working directory before the session
        starts. Returns False, writing nothing, when the session has no working directory of its own (external agent)."""
        directory = getattr(self.session, "actor_dir", None)
        if directory is None or session_backend_kind(self.session) == EXTERNAL_BACKEND:
            return False
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_text(text, encoding="utf-8")
        return True

    def actor_tools(self):
        if hasattr(self, 'harness'):
            return self.harness.tool_registry.catalog(self.harness, role='actor')
        from types import SimpleNamespace
        config = getattr(self, 'config', Config())
        runtime = SimpleNamespace(config=config, disabled_tools=config.features.disabled_tools,
            observation_profile=getattr(self, 'observation_profile', config.features.observation_profile),
            motion_review=getattr(self, 'motion_review', config.features.motion_review),
            enable_wait=getattr(self, 'enable_wait', False),
            vision_compare=getattr(self, 'vision_compare', config.features.vision_compare),
            visual_reviewer=getattr(self, 'visual_review_model', None),
            action_policy=True if getattr(self, 'pi05_enabled', False) else None,
            capabilities=None, programs=True)
        return create_registry(runtime=runtime).catalog(runtime, role='actor')

    def drain_scene_events(self):
        if not self.scene_service: return
        cursor=self.store.read_state("runtime","scene_cursor",0)
        end = self.scene_service.store.event_cursor()
        while cursor < end:
            page = self.scene_service.event_page(cursor)
            if not page["events"]:
                break
            for event in page["events"]:
                self.store.ingest_event(event)
            cursor = page["next_cursor"]
            self.store.set_state("runtime", "scene_cursor", cursor)

    async def forward_scene_events(self):
        while True:
            self.drain_scene_events()
            await asyncio.sleep(.05)

    def submit_scene_thread(self, request, correlation):
        async def submit():
            job = self.scene_service.submit(request, "build_scene", identifier="job-"+digest(correlation)[:32])
            self.store.event("scene.job_linked", {"job_id":job["job_id"]}, correlation, world_kind="internal")
            return job
        return asyncio.run_coroutine_threadsafe(submit(),self.loop).result(15)

    def snapshot(self):
        if self.controller and self.controller["expires_at"] < time.time(): self.controller=None
        snapshot = dict(**self.identity, status=self.status, thread_id=self.session.thread_id, turn_id=self.session.turn_id,
                    codex_status=self.session.state, agent_backend=self._backend_kind(), agent_status=self.session.state,
                    action_count=self.harness.action_count, accepting_actions=self.harness.accepting,
                    active_action=self.harness.active_action,
                    # True while a command holds the control lock: only the lease's last holder may claim it meanwhile.
                    control_busy=self.control_lock.locked(),
                    motion_status=self.harness.states.get(self.harness.active_action,{}).get("status","idle"),
                    controller={"owner_client_id":self.controller["client_id"] if self.controller else None,"expires_at":self.controller["expires_at"] if self.controller else None,"can_claim":self.controller is None},
                    pending_requests=[r for r in self.store.states("native_request") if r["status"]=="pending"],
                    token_accounting=self.store.read_state("runtime", "token_accounting"),
                    terminal=self.terminal.status() if self.terminal else None, capabilities=self.harness.capabilities)
        handoff = self.handoff_state() if callable(getattr(self, "handoff_state", None)) else None
        if handoff is not None:
            snapshot["handoff"] = handoff   # predecessor link / fallback reason / accepted successor (handoff.py)
        return snapshot

    def authorize_writer(self, value, origin):
        self.snapshot()
        if not self.controller:
            if origin: raise ValueError("input_owner_claim_required")
            return
        if value.get("client_id") != self.controller["client_id"] or not secrets.compare_digest(str(value.get("writer_token","")),self.controller["token"]):
            raise ValueError("input_owner_conflict")
        self.controller["expires_at"]=time.time()+30

    async def execute_control(self, value, origin=None):
        command, identifier = value.get("command"), value["request_id"]
        if self.replay: raise ValueError("replay_is_read_only")
        self.refuse_during_handoff(command, identifier)  # after handoff_run was accepted, only safety commands remain
        if command in LEASE_COMMANDS:
            client_id=value.get("client_id")
            if not isinstance(client_id,str) or not 0<len(client_id)<=128: raise ValueError("invalid_client_id")
            self.snapshot()
            if command=="claim_input":
                if self.controller and self.controller["client_id"]!=client_id: raise ValueError("input_owner_conflict")
                # Lease commands do not wait for the control lock (LEASE_COMMANDS), so a step may outlive its owner's
                # lease. Only that owner may claim it again before the command ends.
                if self.control_lock.locked() and client_id!=self.lease_client: raise ValueError(LEASE_BUSY_REFUSAL)
                if self.controller: self.authorize_writer(value,origin)
                if not self.controller: self.controller={"client_id":client_id,"token":secrets.token_urlsafe(32),"expires_at":time.time()+30}
                self.lease_client=client_id
            else:
                self.authorize_writer(value,origin)
                if command=="release_input": self.controller=None
            return {"request_id":identifier,"status":"released" if not self.controller else "owned","writer_token":self.controller["token"] if self.controller else None,"expires_at":self.controller["expires_at"] if self.controller else None}
        if command=="stop": return await self.stop(identifier)
        if command in getattr(self.session, "console_commands_unsupported", ()):
            # The backend has no agent NI-RUA runs (external): nothing to prompt, switch, unblock or hand off.
            raise AgentCapabilityError(command + "_unsupported_by_backend")
        if command in OPERATOR_COMMANDS: return await self.execute_operator(value,origin)
        self.authorize_writer(value,origin)
        if command=="reply":
            result=await asyncio.to_thread(self.session.reply,value["native_request_id"],value["response"])
            return {**result,"control_request_id":identifier}
        if command=="reset":
            # An open turn of an agent NI-RUA does not run (external) does not block the reset; the reset closes it.
            if self.starting or session_turn_blocks_operator(self.session) or self.harness.active_action: raise ValueError("reset_requires_idle_turn_and_motion")
            if self.harness.programs:
                programs = self.harness.programs.snapshot()
                if programs['active_jobs'] or programs['unresolved_jobs']:
                    raise ValueError("reset_requires_program_stop")
            if self.scene_service:
                for job in self.scene_service.jobs():
                    if job["status"] not in JOB_TERMINAL: self.scene_service.transition(job,job["status"],cancel_requested=True,cancel_reason="environment_reset")
                deadline=time.monotonic()+10
                while self.scene_service.processes and time.monotonic()<deadline: await asyncio.sleep(.05)
                if self.scene_service.processes: raise ValueError("scene_workers_stop_unconfirmed")
            observation=await asyncio.to_thread(self.harness.reset,value.get("seed",0),value.get("layout_id",0))
            if getattr(self.session, "operator_turn_release", False):
                # Operator authority over the external agent's turn, only once the reset happened (a refused reset
                # changes nothing): agent.turn_closed_by_operator {reason: operator_reset}, turn interrupted; its next
                # call is refused with external_turn_closed (the environment was reset) until run_begin.
                await asyncio.to_thread(self.session.release_turn, "operator_reset")
            if self.scene_service: self.scene_service.closing=False
            self.status="running" if self.harness.accepting else "stopped"
            return {"request_id":identifier,"status":"reset","observation_id":observation["observation_id"],**self.identity}
        if command=="steer":
            # Mid-turn operator input: joins the running turn; stop/hold remain the way to halt motion.
            text=value.get("prompt")
            if not isinstance(text,str) or not text.strip() or len(text)>16000: raise ValueError("invalid_task_prompt")
            if not self._session_capability("steer"): raise AgentCapabilityError("steer_unsupported_by_backend")
            if self.starting or self.session.state!="running": raise ValueError("no_active_turn_to_steer")
            turn_id=await asyncio.to_thread(self.session.steer,text)
            return dict(request_id=identifier,thread_id=self.session.thread_id,turn_id=turn_id,status="steered")
        if command=="prompt":
            if self.starting or not self.harness.accepting or self.session.state=="running" or self.harness.active_action: raise ValueError("run_not_ready_for_new_task")
            prompt=value.get("prompt")
            if not isinstance(prompt,str) or not prompt.strip() or len(prompt)>16000: raise ValueError("invalid_task_prompt")
            turn_id=await asyncio.to_thread(self.session.run_turn,prompt)
            self.record_turn_settings(turn_id,"prompt")  # run.turn_settings (observer data, best effort)
            return dict(request_id=identifier,thread_id=self.session.thread_id,turn_id=turn_id,status="running")
        raise ValueError("unsupported_control_command")

    def current_unknown_actions(self):
        unresolved_fn = getattr(self.harness, "unresolved_execution", None)
        if callable(unresolved_fn):
            return unresolved_fn()
        from ...core.presentation.feedback_delivery import execution_unconfirmed
        programs = getattr(self.harness, 'programs', None)
        if programs and programs.unresolved_jobs():
            return True
        reset=self.store.read_state("runtime","reset")
        if reset and reset["status"]=="unknown" and hasattr(self,"identity") and all(reset.get(key)==self.identity[key] for key in IDENTITY): return True
        for identifier,state in self.harness.states.items():
            if not execution_unconfirmed(state): continue
            if not hasattr(self,"identity"): return True
            try: request=self.store.read("action_request",identifier)
            except KeyError: return True
            if all(request.get(key)==self.identity[key] for key in IDENTITY): return True
        return False

    def transport_recovery_block(self):
        """A model resend is safe only while the paused world has no pending action."""
        harness = self.harness
        if self.closed or not harness.accepting:
            return "harness_not_accepting"
        if harness.episode_done.is_set():
            return "native_episode_ended"
        if harness.active_action:
            return "action_in_progress"
        acknowledged = (getattr(self, "operator_gate", None) or {}).get("acknowledged_actions", ())
        unresolved_fn = getattr(harness, "unresolved_execution", None)
        has_unknown = (
            unresolved_fn(acknowledged=acknowledged)
            if callable(unresolved_fn)
            else self.current_unknown_actions()
        )
        if has_unknown:
            return "action_status_unknown"
        if self.has_active_programs():
            return "program_active"
        return None

    def has_active_programs(self):
        programs = getattr(self.harness, "programs", None)
        return bool(programs and programs.snapshot()['active_jobs'])

    def program_continuation_state(self):
        """Only an acknowledged review pause permits another benchmark turn."""
        programs = getattr(self.harness, "programs", None)
        return programs.continuation_state() if programs else dict(blocked=False, pending_reviews=[])

    async def abort_startup(self, starting):
        """Unblock the original startup owner without leaving late-created children."""
        signalled = {}
        while not starting.done():
            processes = (("robot", getattr(self.robot, "process", None)),
                         *Workbench._owned_session_processes(self),
                         ("terminal", getattr(self.terminal, "process", None)))
            for role, process in processes:
                if process is None or process.poll() is not None:
                    continue
                previous = signalled.get(process)
                signal_name = "SIGTERM" if previous is None else "SIGKILL"
                if previous is not None and (previous[2] == "SIGKILL" or time.monotonic() - previous[1] < 5):
                    continue
                try:
                    process.terminate() if previous is None else process.kill()
                except ProcessLookupError:
                    continue
                signalled[process] = (role, time.monotonic(), signal_name)
                if role == "robot":
                    self.robot.startup_forced_termination = signal_name
                self.store.event("run.startup_process_signalled", dict(role=role, pid=process.pid, signal=signal_name))
            await asyncio.sleep(.05)
        for process, (role, _, signal_name) in signalled.items():
            self.store.event("run.startup_process_exit", dict(role=role, pid=process.pid,
                signal=signal_name, process_exit_confirmed=process.poll() is not None, returncode=process.returncode))

    async def stop(self, request_id):
        if request_id in self.control_responses:
            return self.control_responses[request_id]
        self.status = "stop_requested"
        self.harness.request_stop()  # Close motion admission before cancelling any owned program.
        if getattr(self.harness,"programs",None):
            self.harness.programs.cancel_all()
        if getattr(self,"scene_service",None):
            await self.scene_service.stop_jobs()
            self.drain_scene_events()
        if self.starting:
            response = dict(request_id=request_id, status="stop_requested", reason="waiting_for_startup_to_release_owned_resources")
            self.control_responses[request_id] = response
            self.store.event("run.stop_requested", response)
            return response
        try:
            await asyncio.wait_for(asyncio.to_thread(self.session.interrupt), timeout=45)
            deadline = asyncio.get_running_loop().time() + 45
            # An external agent may open a new turn right after interrupt() closed its turn; that never blocks the stop.
            while (self.harness.active_action or session_turn_blocks_operator(self.session) or self.has_active_programs()) and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.05)
            unresolved = self.current_unknown_actions()
            self.status = "stopped" if self.harness.active_action is None and not session_turn_blocks_operator(self.session) and not self.has_active_programs() and not unresolved else "unknown"
        except Exception as error:
            self.status = "unknown"
            self.store.event("run.stop_unknown", {"reason": str(error)})
            deadline = asyncio.get_running_loop().time() + 45
            while (self.harness.active_action or self.has_active_programs()) and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(.05)
        if (getattr(self.harness, "episode_terminal", None) and self.harness.active_action is None
                and not self.has_active_programs() and not self.current_unknown_actions()):
            # Native completion is already latched and motion admission closed.
            # An actor interrupt failure must not discard the robot's result.
            if self.session.state == "running":
                try:
                    await asyncio.wait_for(asyncio.to_thread(self.session.close), timeout=45)
                except Exception as error:
                    self.store.event("codex.close_after_native_terminal_failed", {"error_type":type(error).__name__})
            self.status = "stopped"
        response = dict(request_id=request_id, status=self.status)
        self.control_responses[request_id] = response
        self.store.event("run." + self.status, response)
        return response

    async def close(self):
        if self.closed: return getattr(self, "cleanup_report", None)
        self.closed=True
        report = self.cleanup_report = {"steps": {}, "errors": [], "robot": None}

        async def cleanup(name, operation):
            try:
                result = await operation()
                report["steps"][name] = "completed"
                return result
            except Exception as error:
                report["steps"][name] = "failed"
                report["errors"].append({"step": name, "error": f"{type(error).__name__}: {error}"})

        await cleanup("motion_admission", lambda: asyncio.to_thread(self.harness.request_stop))
        if self.session.state == "running" or self.harness.active_action or self.has_active_programs():
            await cleanup("stop", lambda: self.stop(new_id("shutdown")))
        if self._supports_shutdown_video_export():
            async def export_motion_video():
                result = await asyncio.to_thread(self.observer_video)
                # A recording re-armed by the console "save video" button may hold no action;
                # it must not replace the video.json of the export that the operator saved.
                if result.get("segments") or not (self.store.root / "video.json").exists():
                    (self.store.root / "video.json").write_text(json.dumps(result, indent=2))
                return result
            await cleanup("motion_video", export_motion_video)
        # Release the worker before session.close waits for tool-handler threads.
        # Every cleanup step is independent: a broken PTY/policy/model must not
        # bypass the owned simulator's native close and bounded process fallback.
        report["robot"] = await cleanup("robot", lambda: asyncio.to_thread(self.robot.close))
        if self.terminal:
            await cleanup("terminal", self.terminal.close)
        if getattr(self.harness, "action_policy", None) is not None:
            await cleanup("policy", lambda: asyncio.to_thread(self.harness.action_policy.close))
        await cleanup("session", lambda: asyncio.to_thread(self.session.close))
        if getattr(self.harness,"programs",None):
            await cleanup("programs", lambda: asyncio.to_thread(self.harness.programs.close))
        if self.scene_service:
            await cleanup("scene_jobs", self.scene_service.stop_jobs)
            await cleanup("scene_events", lambda: asyncio.to_thread(self.drain_scene_events))
        if self.scene_forwarder:
            self.scene_forwarder.cancel()
            await asyncio.gather(self.scene_forwarder,return_exceptions=True)
        if self.scene_service:
            await cleanup("scene_service", self.scene_service.close)
        if getattr(self, "grasp_memory_store", None) is not None:
            await cleanup("grasp_memory_store", lambda: asyncio.to_thread(self.grasp_memory_store.close))
        await cleanup("store", lambda: asyncio.to_thread(self.store.close))
        await cleanup("run_lock", lambda: asyncio.to_thread(self.run_lock.close))
        return report

    def require_observer_idle(self):
        if (self.replay or self.starting or self.harness.active_action or (session_turn_blocks_operator(self.session) and not getattr(self.harness, "episode_terminal", None))
                or self.current_unknown_actions() or self.has_active_programs()):
            raise ValueError("observer_operation_requires_idle_run")

    def observer_evaluate(self):
        with self.harness.robot_lock:
            self.require_observer_idle()
            return record_evaluation(self.store, self.identity, self.robot.evaluate(final_check=True))

    def observer_video(self):
        with self.harness.robot_lock:
            self.require_observer_idle()
            return record_video(self.store, self.identity, self.robot.save_video())

    def _supports_shutdown_video_export(self):
        config = getattr(self, "config", None)
        run_cfg = getattr(config, "run", None)
        features = getattr(config, "features", None)
        if (
            getattr(self, "replay", False)
            or getattr(run_cfg, "benchmark", False)
            or not getattr(features, "record_video", False)
            or not getattr(features, "export_video", False)
            or self.harness.active_action
            or self.has_active_programs()
            or self.current_unknown_actions()
        ):
            return False
        caps = getattr(self.robot, "capabilities", None) or {}
        if not caps.get("observer_video"):
            return False
        is_video = isinstance(self.robot, VideoBackend) or callable(getattr(self.robot, "save_video", None))
        is_evaluator = isinstance(self.robot, EvaluatorBackend) or callable(getattr(self.robot, "evaluate", None))
        if not is_video or is_evaluator:
            return False
        return (
            self._embodiment().clock == ClockDomain.REAL_TIME
            or not callable(getattr(self.robot, "embodiment_profile", None))
        )

    # ---- Composed router & capability extension delegation (backward-compatible surface) ----
    def _get_ext(self, attr, factory):
        ext = self.__dict__.get(attr)
        if ext is None:
            ext = factory(self)
            self.__dict__[attr] = ext
        return ext

    def app(self):
        return self._get_ext("_router", GatewayRouter).app()

    async def control_dispatch(self, value, origin):
        return await self._get_ext("_router", GatewayRouter).control_dispatch(value, origin)

    def telemetry_directory(self):
        return self._get_ext("_console_ext", OperatorConsoleExtension).telemetry_directory()

    def joint_limits(self):
        return self._get_ext("_console_ext", OperatorConsoleExtension).joint_limits()

    def unknown_action_ids(self):
        return self._get_ext("_console_ext", OperatorConsoleExtension).unknown_action_ids()

    def unknown_items(self):
        return self._get_ext("_console_ext", OperatorConsoleExtension).unknown_items()

    def resume_blockers(self):
        return self._get_ext("_console_ext", OperatorConsoleExtension).resume_blockers()

    def console_state(self):
        return self._get_ext("_console_ext", OperatorConsoleExtension).console_state()

    def model_transport_view(self):
        return self._get_ext("_console_ext", OperatorConsoleExtension).model_transport_view()

    def _effective_speed_scale_cap(self):
        return self._get_ext("_console_ext", OperatorConsoleExtension)._effective_speed_scale_cap()

    def _model_options(self, fallback=None):
        return self._get_ext("_console_ext", OperatorConsoleExtension)._model_options(fallback)

    def _session_model_options(self):
        return self._get_ext("_console_ext", OperatorConsoleExtension)._session_model_options()

    @staticmethod
    def _labelled_model_options(items):
        return OperatorConsoleExtension._labelled_model_options(items)

    def _effort_choices(self):
        return self._get_ext("_console_ext", OperatorConsoleExtension)._effort_choices()

    def _session_supports(self, name):
        return self._get_ext("_console_ext", OperatorConsoleExtension)._session_supports(name)

    def _operator_unknown(self, command, identifier, reason):
        return self._get_ext("_console_ext", OperatorConsoleExtension)._operator_unknown(command, identifier, reason)

    async def _operator(self, command, identifier, robot_lock=False, progress=None, event_origin=None, **arguments):
        return await self._get_ext("_console_ext", OperatorConsoleExtension)._operator(
            command,
            identifier,
            robot_lock=robot_lock,
            progress=progress,
            event_origin=event_origin,
            **arguments,
        )

    async def execute_operator(self, value, origin):
        return await self._get_ext("_console_ext", OperatorConsoleExtension).execute_operator(value, origin)

    async def apply_service_tier(self, identifier, tier):
        return await self._get_ext("_console_ext", OperatorConsoleExtension).apply_service_tier(identifier, tier)

    async def apply_model(self, identifier, selected):
        return await self._get_ext("_console_ext", OperatorConsoleExtension).apply_model(identifier, selected)

    async def apply_effort(self, identifier, effort):
        return await self._get_ext("_console_ext", OperatorConsoleExtension).apply_effort(identifier, effort)

    async def _reopen_after_operator(self, identifier, was_accepting, stop_revision, unconfirmed):
        return await self._get_ext("_console_ext", OperatorConsoleExtension)._reopen_after_operator(
            identifier, was_accepting, stop_revision, unconfirmed
        )

    def handheld_view(self):
        return self._get_ext("_console_ext", OperatorConsoleExtension).handheld_view()

    async def board_check(self):
        return await self._get_ext("_console_ext", OperatorConsoleExtension).board_check()

    def console_routes(self, app):
        self._get_ext("_console_ext", OperatorConsoleExtension).mount(app)
        self._get_ext("_agent_settings_ext", AgentSettingsExtension).mount(app)
        self._get_ext("_calibration_ext", CalibrationExtension).mount(app, control_dispatch=self.control_dispatch)

    def calibration_blockers(self, motion):
        return self._get_ext("_calibration_ext", CalibrationExtension).blockers(motion)

    async def calibration_workflow_snapshot(self):
        return await self._get_ext("_calibration_ext", CalibrationExtension).snapshot()

    async def calibration_summary(self):
        return await self._get_ext("_calibration_ext", CalibrationExtension).summary()

    async def calibration_workflow_command(self, command, identifier, value):
        return await self._get_ext("_calibration_ext", CalibrationExtension).handle_command(command, identifier, value)

    def calibration_routes(self, app, control_dispatch=None):
        self._get_ext("_calibration_ext", CalibrationExtension).mount(app, control_dispatch=control_dispatch)

    def agent_settings_current(self):
        return self._get_ext("_agent_settings_ext", AgentSettingsExtension).current()

    def agent_event_context(self):
        return self._get_ext("_agent_settings_ext", AgentSettingsExtension).event_context()

    def agent_setting_changed(self, field, previous, value, request_id, applies):
        return self._get_ext("_agent_settings_ext", AgentSettingsExtension).setting_changed(
            field, previous, value, request_id, applies
        )

    def record_turn_settings(self, turn_id, origin):
        return self._get_ext("_agent_settings_ext", AgentSettingsExtension).record_turn_settings(turn_id, origin)

    def profiles_dir(self):
        return self._get_ext("_agent_settings_ext", AgentSettingsExtension).profiles_dir()

    def served_view(self):
        return self._get_ext("_agent_settings_ext", AgentSettingsExtension).served_view()

    def agent_settings_view(self):
        return self._get_ext("_agent_settings_ext", AgentSettingsExtension).view()

    def handoff_blockers(self):
        return self._get_ext("_agent_settings_ext", AgentSettingsExtension).handoff_blockers()

    def handoff_view(self, selection_blockers=()):
        return self._get_ext("_agent_settings_ext", AgentSettingsExtension).handoff_view(selection_blockers)

    def _load_profile(self, profile_id):
        return self._get_ext("_agent_settings_ext", AgentSettingsExtension)._load_profile(profile_id)

    def _check_selection_cached(self, backend, path, mtime_ns, wire, model, effort, thinking=_UNSET):
        return self._get_ext("_agent_settings_ext", AgentSettingsExtension)._check_selection_cached(
            backend, path, mtime_ns, wire, model, effort, thinking
        )

    @staticmethod
    def _readable(backend, reason, message, info=None):
        return AgentSettingsExtension._readable(backend, reason, message, info)

    def read_catalogue(self, profile, mtime_ns, wire, refresh=False):
        return self._get_ext("_agent_settings_ext", AgentSettingsExtension).read_catalogue(
            profile, mtime_ns, wire, refresh
        )

    def listed_model_items(self):
        return self._get_ext("_agent_settings_ext", AgentSettingsExtension).listed_model_items()

    def annotate_model_options(self, options):
        return self._get_ext("_agent_settings_ext", AgentSettingsExtension).annotate_model_options(options)

    async def agent_options(self, refresh=False):
        return await self._get_ext("_agent_settings_ext", AgentSettingsExtension).agent_options(refresh)

    async def profile_models(self, profile_id, wire, refresh=False):
        return await self._get_ext("_agent_settings_ext", AgentSettingsExtension).profile_models(
            profile_id, wire, refresh
        )

    async def check_agent_selection(self, value):
        return await self._get_ext("_agent_settings_ext", AgentSettingsExtension).check_agent_selection(value)

    async def save_profile_request(self, value, origin):
        return await self._get_ext("_agent_settings_ext", AgentSettingsExtension).save_profile_request(value, origin)

    async def remove_profile_request(self, profile_id, value, origin):
        return await self._get_ext("_agent_settings_ext", AgentSettingsExtension).remove_profile_request(
            profile_id, value, origin
        )

    @staticmethod
    def _validate_session_field(field, value):
        return AgentSettingsExtension._validate_session_field(field, value)

    async def apply_agent_settings(self, identifier, value):
        return await self._get_ext("_agent_settings_ext", AgentSettingsExtension).apply_agent_settings(identifier, value)

    async def handoff_run(self, identifier, value):
        return await self._get_ext("_agent_settings_ext", AgentSettingsExtension).handoff_run(identifier, value)

    def agent_settings_routes(self, app):
        self._get_ext("_agent_settings_ext", AgentSettingsExtension).mount(app)

    def handoff_available(self):
        return self._get_ext("_handoff_ext", HandoffExtension).available()

    def handoff_state(self):
        return self._get_ext("_handoff_ext", HandoffExtension).state()

    def refuse_during_handoff(self, command, identifier):
        return self._get_ext("_handoff_ext", HandoffExtension).refuse_during_handoff(command, identifier)

    def _handoff_selection(self, value):
        return self._get_ext("_handoff_ext", HandoffExtension)._selection(value)

    def handoff_price_card_problem(self, candidate, backend):
        return self._get_ext("_handoff_ext", HandoffExtension).price_card_problem(candidate, backend)

    def handoff_candidate(self, backend, profile, wire, model, effort, thinking=_UNSET):
        return self._get_ext("_handoff_ext", HandoffExtension).candidate(
            backend, profile, wire, model, effort, thinking
        )

    def handoff_selection_blockers(self, backend, profile, wire, model, effort, thinking=_UNSET):
        return self._get_ext("_handoff_ext", HandoffExtension).selection_blockers(
            backend, profile, wire, model, effort, thinking
        )

    async def request_handoff(self, identifier, value):
        return await self._get_ext("_handoff_ext", HandoffExtension).request_handoff(identifier, value)

    async def apply_handoff_carry(self):
        return await self._get_ext("_handoff_ext", HandoffExtension).apply_carry()
