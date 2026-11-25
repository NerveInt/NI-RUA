"""Immutable configuration section schemas, type validation, and recursive merging."""
from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import re
import sys
import types
from typing import Union, get_args, get_origin, get_type_hints
from urllib.parse import urlsplit

from .thresholds import MOTION_THRESHOLD_BOUNDS, ThresholdMap, _DEFAULT_MOTION_THRESHOLDS


class ConfigError(ValueError):
    pass


# features.vggt_support_z_m: the default "take the support height from the run's EmbodimentProfile" (see there);
# explicit numbers (m) must lie in SUPPORT_Z_RANGE_M.
SUPPORT_Z_FROM_PROFILE = 'profile'
SUPPORT_Z_RANGE_M = (-3.0, 3.0)


def option(
    default=None,
    *,
    choices=None,
    minimum=None,
    maximum=None,
    path=False,
    cli: bool | str = False,
    cli_help: str | None = None,
    queue_case: bool = False,
    queue_required: bool = False,
    queue_cli: bool = False,
):
    return field(
        default=default,
        metadata=dict(
            choices=choices,
            minimum=minimum,
            maximum=maximum,
            path=path,
            cli=cli,
            cli_help=cli_help,
            queue_case=queue_case,
            queue_required=queue_required,
            queue_cli=queue_cli,
        ),
    )


def _type_ok(value, annotation):
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        return any(_type_ok(value, item) for item in get_args(annotation))
    if origin is tuple:
        return isinstance(value, tuple) and all(_type_ok(item, get_args(annotation)[0]) for item in value)
    if annotation is float:
        return type(value) in (int, float) and math.isfinite(value)
    if annotation in (int, bool, str):
        return type(value) is annotation
    if annotation is type(None):
        return value is None
    return isinstance(value, annotation)


@lru_cache(maxsize=None)
def _section_hints(cls):
    return types.MappingProxyType(get_type_hints(cls))


class Section:
    def __post_init__(self):
        hints = _section_hints(type(self))
        for item in fields(self):
            value = getattr(self, item.name)
            key = type(self).__name__ + '.' + item.name
            if not _type_ok(value, hints[item.name]):
                raise ConfigError(f'invalid_config_type:{key}')
            if value is None:
                continue
            metadata = item.metadata
            if metadata.get('choices') is not None and value not in metadata['choices']:
                raise ConfigError(f'invalid_config_choice:{key}: {value!r}')
            if metadata.get('minimum') is not None and value < metadata['minimum']:
                raise ConfigError(f'config_below_minimum:{key}')
            if metadata.get('maximum') is not None and value > metadata['maximum']:
                raise ConfigError(f'config_above_maximum:{key}')
            if isinstance(value, str) and (not value.strip() or '\x00' in value):
                raise ConfigError(f'invalid_config_string:{key}')
            if isinstance(value, tuple) and any(isinstance(item, str) and not item.strip() for item in value):
                raise ConfigError(f'invalid_config_list:{key}')

    def to_dict(self):
        # JSON-compatible arrays rather than Python tuples in exported snapshots.
        return json.loads(json.dumps(asdict(self)))


_EXTERNAL_PERCEPTION_SWITCHES = (
    'vggt_height', 'object_table', 'scene3d', 'object_table_observe', 'scene3d_observe',
)
_EXTERNAL_PERCEPTION_OPT_IN_SWITCHES = (
    'pose_skill', 'held_perception', 'persistent_object_tracking', 'multiview_scene3d',
)
_GRIP_MARGIN_FIELDS = (
    'grip_non_empty_min_aperture_m', 'grip_thin_min_aperture_m', 'grip_rigid_min_aperture_m',
    'grip_wide_min_aperture_m',
)
# What a run did before the reply-delivery fields existed. An archive, manifest or frozen runtime that lacks these
# fields had none of the features, so they are restored / projected with these values, never with the current
# defaults (apps/gateway/workbench.py resume and replay; apps/evaluation/queue.py project_config_for_runtimes).
REPLY_DELIVERY_ABSENT = {
    "budgeted_action_reply": False,
    "reply_same_as": False,
    "reply_float32": False,
    "reply_static_once": False,
    "action_reply_profile": "full",
    "reply_compact_encoding": False,
}
# The tool catalogue and URDF delivery settings (features.tool_catalog_profile, features.robot_urdf_delivery) as the run
# behaved before their defaults changed to layered_v3 / file: an archive or frozen runtime that lacks a key ran with
# this value (the full prose catalogue, the URDF inline in the startup instructions). ``tool_catalog_profile`` is older
# than ``robot_urdf_delivery``, so each key is restored on its own: an archive that records one keeps it.
CATALOGUE_DELIVERY_ABSENT = {
    "tool_catalog_profile": "full_v1",
    "robot_urdf_delivery": "inline",
}
# Every feature value that a missing key restores as "the behaviour of the time" (resume, replay, turn_probe slices,
# projection to a frozen runtime), never the current default.
FEATURE_ABSENT_VALUES = {**REPLY_DELIVERY_ABSENT, **CATALOGUE_DELIVERY_ABSENT}

_RETIRED_BLOAT_DEFAULTS = {
    'self_briefing': False,
    'operation_candidates': False,
    'lean_projection': False,
    'lean_projection_k': 1,
    'lean_projection_block': 8,
    'lean_request_dump': 0,
    'lean_compact': False,
}
# features.* keys retired by design 13 V-E (unread threshold fields) and Wave-2 Config v2 (disabled cortex/presentation
# bloat flags). Older configuration files and archived effective_config.json snapshots that still carry them load
# unchanged: Config.from_dict / Config.load accept these keys and drop them (no validation, no effect, absent from to_dict()).
RETIRED_FEATURE_FIELDS = frozenset({
    'soft_position_tolerance_m', 'grip_thin_max_aperture_m', 'grip_rigid_max_aperture_m', 'grip_wide_max_aperture_m',
    'max_min_grip_m', 'closed_jaw_goal_max', 'checkpoint_empty_jaw_measured_max',
    *_RETIRED_BLOAT_DEFAULTS,
})
EXTERNAL_PERCEPTION_FIELDS = (
    'external_perception', 'vggt_height', 'vggt_height_mode', 'vggt_endpoint',
    'vggt_support_z_m', 'vggt_timeout_s', 'object_table', 'object_table_max_objects',
    'object_table_max_height_m', 'object_table_head_exclude_px', 'scene3d',
    'scene3d_voxel_m', 'scene3d_z_above_m', 'scene3d_z_below_m',
    'object_table_observe', 'scene3d_observe', 'persistent_object_tracking', 'multiview_scene3d',
    'obs3d_timeout_s', 'obs3d_max_objects', 'obs3d_action_overlay', 'pose_skill', 'held_perception',
)


@dataclass(frozen=True)
class EmbodimentConfig(Section):
    """Sparse threshold overrides overlaid onto ARX-X5 packaged defaults (`arx_x5/defaults.json`)."""

    thresholds: dict = field(default_factory=dict)

    def __post_init__(self):
        raw = self.thresholds
        if not isinstance(raw, dict):
            raise ConfigError('invalid_config_type:EmbodimentConfig.thresholds')
        overrides = dict(raw._overrides) if isinstance(raw, ThresholdMap) else dict(raw)
        unknown = set(overrides) - set(MOTION_THRESHOLD_BOUNDS)
        if unknown:
            raise ConfigError(f'unknown_config_keys:EmbodimentConfig.thresholds: {", ".join(sorted(unknown))}')
        for key, val in overrides.items():
            expected_type, minimum, maximum = MOTION_THRESHOLD_BOUNDS[key]
            if not _type_ok(val, expected_type):
                raise ConfigError(f'invalid_config_type:EmbodimentConfig.thresholds.{key}')
            if minimum is not None and val < minimum:
                raise ConfigError(f'config_below_minimum:EmbodimentConfig.thresholds.{key}')
            if maximum is not None and val > maximum:
                raise ConfigError(f'config_above_maximum:EmbodimentConfig.thresholds.{key}')
        resolved = {**_DEFAULT_MOTION_THRESHOLDS, **overrides}
        for name in _GRIP_MARGIN_FIELDS:
            if resolved[name] > resolved['max_gripper_aperture_m']:
                raise ConfigError(f'config_above_maximum:EmbodimentConfig.thresholds.{name}')
        object.__setattr__(self, 'thresholds', ThresholdMap(resolved, overrides=overrides))

    def to_dict(self):
        return {'thresholds': json.loads(json.dumps(dict(self.thresholds._overrides)))}


@dataclass(frozen=True)
class RunConfig(Section):
    run_dir: str | None = option(path=True, cli=True, queue_case=True, queue_required=True)
    experiment_root: str = option('~/exp/ni-rua', path=True, cli=True, cli_help='Archive root; default ~/exp/ni-rua')
    category: str = option('main', cli=True)
    title: str | None = option(None, cli=True, queue_case=True, queue_required=True)
    tag: tuple[str, ...] = option((), cli=True, queue_case=True, queue_required=True)
    replay: bool = option(False, cli=True, cli_help='Offline read-only viewer: no GPU, app-server, model or actions')
    resume: bool = option(False, cli=True)
    task: str = option('stack_bowls', cli=True, queue_case=True, queue_required=True)
    seed: int = option(0, minimum=0, cli=True, queue_case=True, queue_required=True)
    layout_id: int = option(0, minimum=0, cli=True, queue_case=True, queue_required=True)
    gpu: int = option(4, minimum=0, cli=True)
    port: int = option(
        0, minimum=0, maximum=65535, cli=True,
        cli_help='Per-run loopback port; 0 chooses a free port. The permanent library uses 8780.',
    )
    max_actions: int | None = option(minimum=1, cli=True, cli_help='Development smoke only; benchmark uses native task steps')
    wall_budget_s: float | None = option(
        minimum=0.001, cli=True, cli_help='Development smoke only; no total wall deadline in benchmark',
    )
    prompt: str | None = option(None, cli=True)
    exit_after_turn: bool = option(False, cli=True)
    benchmark: bool = option(
        False, cli=True,
        cli_help='Use the native public task instruction; score and export actual video after the actor stops, then release resources',
    )
    task_notes: str | None = option(
        None, cli=True, queue_case=True,
        cli_help='Additional public guidance appended to default operation notes; never hidden thresholds, object truth or scripted actions',
    )
    task_notes_source: str | None = option(
        None, cli=True, queue_case=True, cli_help='Public source for --task-notes; saved with the actual prompt',
    )


@dataclass(frozen=True)
class FeaturesConfig(Section):
    sandbox_enabled: bool = False
    program_seccomp_enabled: bool = False
    programs_dir: str | None = None
    observation_profile: str = option(
        'rgb_proprio_calibrated_v1',
        choices=('rgb_proprio_v1', 'rgb_proprio_calibrated_v1', 'rgbd_proprio_calibrated_v1'),
        cli=True, queue_case=True,
        cli_help='Calibrated RGB/proprio by default; explicit rgb_proprio_v1 retains the legacy control boundary',
    )
    depth_skills: bool = option(False, cli=True, queue_case=True)
    depth_auto_feedback: bool = option(False, cli=True, queue_case=True)
    depth_event_payload: bool = option(False, cli=True, queue_case=True)
    actor_payload_profile: str = option(
        'lean_v2', choices=('full_v1', 'compact_v1', 'lean_v1', 'lean_v2'),
        cli=True, queue_case=True,
        cli_help='Model-facing JSON presentation; programs and immutable evidence retain full data',
    )
    # How tool descriptions state the input contract (runtime/tools/definitions.py:_publish): full_v1 the prose field
    # guide; compact_v1 the same contract in typed notation; layered_v1..v3 typed notation with rarely used fields
    # shortened (catalog_guide.LAYERS). guide_only_v1: full_v1 descriptions, and Codex dynamic-tool schemas omit property descriptions
    # the tool's field guide already carries verbatim (cortex/agents/codex/session.py:_native_dynamic_tools).
    tool_catalog_profile: str = option(
        'layered_v3', choices=('full_v1', 'guide_only_v1', 'compact_v1', 'layered_v1', 'layered_v2', 'layered_v3'), cli=True, queue_case=True,
        cli_help='Tool catalogue form: layered_v3 (default) typed notation with rarely used fields shortened and their '
                 'full text in a reference file (layered_v1, layered_v2: its earlier forms); full_v1 the prose field '
                 'guide; guide_only_v1 sends each input field '
                 'description once (in the tool description) instead of also in the schema; compact_v1 the same '
                 'contract in typed notation',
    )
    robot_urdf_delivery: str = option(
        'file', choices=('inline', 'file'), cli=True, queue_case=True,
        cli_help='Where the actor gets the robot URDF text: as the file robot.urdf in its working directory (default; '
                 'inline where the session has none) or inline in the startup instructions',
    )
    default_task_notes: bool = option(
        True, cli=True, queue_case=True,
        cli_help='Include versioned operation experience in new benchmark prompts (default: enabled)',
    )
    # Per-task corrections of the actor-facing task text (ni_rua/task_instruction_overrides.json).
    task_instruction_overrides: bool = True
    # 首轮内联 obs_1 结构化 3D 状态与概览图开关（默认开启，省去首轮空跑 robot_observe）。
    startup_inline_grounding: bool = option(
        True, cli=True, queue_case=True,
        cli_help='Inline initial obs_1 3D state and calibrated camera views into turn-1 prompt (default: enabled)',
    )
    # 首轮内联视觉视图模式：overview（默认单张概览图，保持单测兼容）或 all_cameras（附带全部原始标定相机图 + 3D ID 标注图）。
    startup_grounding_views: str = option(
        'overview', choices=('overview', 'all_cameras'),
        cli=True, queue_case=True,
        cli_help='Turn-1 inline camera views: overview (single view, default) or all_cameras (all calibrated raw RGB views plus 3D overlay)',
    )
    vision_compare: bool = option(
        True, cli=True, queue_case=True, cli_help='Enable RGB feature comparison and comparison-based skills',
    )
    robotic_view: bool = option(
        False, cli=True, queue_case=True,
        cli_help='Expose read-only wrist-camera look-at pose suggestions (default: disabled)',
    )
    stereo_guide: bool = option(
        False, cli=True, queue_case=True,
        cli_help='Expose read-only epipolar guidance between calibrated RGB cameras (default: disabled)',
    )
    enable_wait: bool = option(
        True, cli=True, queue_case=True,
        cli_help='Enable bounded native waiting with chronological RGB samples (default: enabled; --no-enable-wait disables)',
    )
    motion_review: bool = option(
        True, cli=True, queue_case=True,
        cli_help='Expose motion preview, reviewed execution and grasp planning tools; legacy observation profile disables these',
    )
    tracking_guard: bool = option(
        False, cli=True, queue_case=True,
        cli_help='Stop EE motion after sustained measured joint tracking divergence; no automatic correction',
    )
    joint_replan: bool = option(
        True, cli=True, queue_case=True,
        cli_help='Try native planning to a closer equivalent joint endpoint and retain only a validated shorter path',
    )
    motion_execution_mode: str = option(
        'resampled', choices=('resampled', 'feedback'),
        cli=True, queue_case=True,
        cli_help='Use resampled unified actions (default); feedback is for trusted diagnostics and cannot run benchmarks',
    )
    # Speed multiplier (dimensionless, 1 = nominal) of a move that names no speed. [0.25, 2.0] is the widest range of
    # any embodiment; the run's embodiment range (EmbodimentProfile.speed_scale: ARX-X5 [0.25, 2.0], SO-101
    # [0.25, 1.0]) is enforced when actions are resolved (robot/motion/motion_speed.resolve_action_speed).
    default_motion_speed_scale: float = option(
        1.0, minimum=.25, maximum=2.0, cli=True, queue_case=True,
        cli_help='Arm command spacing when a motion request omits speed selection (0.25..2; default 1)',
    )
    arm_queue: bool = option(True, cli=True, queue_case=True, cli_help='Expose bounded per-arm stage queues')
    motion_cost_preview: bool = option(
        True, cli=True, queue_case=True, cli_help='Expose bounded planning-only candidate cost comparison',
    )
    airborne_handoff: bool = option(
        True, cli=True, queue_case=True,
        cli_help='Permit direct airborne handoff; --no-airborne-handoff enforces surface-only transfer. Does not disable dual-arm motion',
    )
    handoff_preference: str = option(
        'surface', choices=('surface', 'airborne'), cli=True, queue_case=True,
        cli_help='Soft planning preference when airborne handoff is allowed; new runs prefer a stable physical surface while leaving the choice to the actor',
    )
    bounded_continuation: bool = option(
        True, cli=True, queue_case=True,
        cli_help='Bound repeated final-answer continuations without native progress',
    )
    motion_safety_guard: bool = option(
        True, cli=True, queue_case=True,
        cli_help='Reject abnormal winding or over-budget compiled segments before dispatch',
    )
    trajectory_preview: bool = option(
        True, cli=True, queue_case=True,
        cli_help='Disclose candidate joints and calibrated robot geometry for actor path review',
    )
    manipulation_protocol: bool = option(
        True, cli=True, queue_case=True,
        cli_help='Use explicit contact, transfer and placement evidence in planning and bounded templates',
    )
    task_scheduling: bool = option(True, cli=True, queue_case=True)
    preview_intent: bool = option(False, cli=True, queue_case=True)
    recovery_evidence: bool = option(True, cli=True, queue_case=True)
    # 动作后轨迹反馈策略：默认 automatic（动作完成后自动返回实测轨迹图与数值摘要，保持闭环视觉反馈高信噪比）
    action_trajectory_delivery: str = option(
        'automatic', choices=('automatic', 'diagnostic'), cli=True, queue_case=True,
        cli_help='Automatic sampled joints and images after actions, or compact routine summaries with full diagnostics on exceptions',
    )
    pi05: bool = option(
        False, cli=True, queue_case=True,
        cli_help='Enable matched ARX X5 pi0.5 short-prefix tool; NI_RUA_PI05_SERVER_URL reuses a running service',
    )
    visual_review_model: str | None = option(
        None, choices=('gpt-6.1-sol',), cli=True, queue_case=True,
        cli_help='Optional counted RGB second-opinion model',
    )
    completion_guidance: str = option(
        'hook', choices=('none', 'initial', 'hook', 'both'), cli=True, queue_case=True,
        cli_help='Benchmark completion guidance: state-aware hook by default; initial/both add an initial reminder; none disables the home check. Hook is dormant outside benchmark.',
    )
    disabled_tools: tuple[str, ...] = option(
        (), cli=True, queue_case=True,
        cli_help='Tool names blocked in both the model interface and program dispatch; empty clears config list',
    )
    external_perception: bool = option(True, cli=True, queue_case=True)
    vggt_height: bool = option(True, cli=True, queue_case=True)
    vggt_height_mode: str = option('optional', choices=('optional', 'default'), cli=True, queue_case=True)
    vggt_endpoint: str = option('http://127.0.0.1:8331', cli=True, queue_case=True)
    vggt_support_z_m: float | str | None = option(
        SUPPORT_Z_FROM_PROFILE, cli=True, queue_case=True,
        cli_help="Support-plane z in observation-frame m [-3,3]; 'profile' (default) for the embodiment's support height (ARX-X5 0.765, SO-101 the observation's); or 'observation' to require that frozen observation's calibrated table height; no fallback",
    )
    vggt_camera_id: str = 'cam_head'
    vggt_timeout_s: float = option(60.0, minimum=1.0, maximum=300.0, cli=True, queue_case=True)
    object_table: bool = option(True, cli=True, queue_case=True)
    object_table_max_objects: int = option(20, minimum=1, maximum=40, cli=True, queue_case=True)
    object_table_max_height_m: float = option(0.30, minimum=0.05, maximum=1.0, cli=True, queue_case=True)
    object_table_head_exclude_px: str = option('200,405,440,480', cli=True, queue_case=True)
    scene3d: bool = option(True, cli=True, queue_case=True)
    scene3d_voxel_m: float = option(0.01, minimum=0.002, maximum=0.05, cli=True, queue_case=True)
    scene3d_z_above_m: float = option(0.60, minimum=0.05, maximum=2.0, cli=True, queue_case=True)
    scene3d_z_below_m: float = option(0.05, minimum=0.0, maximum=1.0, cli=True, queue_case=True)
    object_table_observe: bool = option(True, cli=True, queue_case=True)
    scene3d_observe: bool = option(True, cli=True, queue_case=True)
    persistent_object_tracking: bool = option(False, cli=True, queue_case=True)
    multiview_scene3d: bool = option(False, cli=True, queue_case=True)
    obs3d_timeout_s: float = option(60.0, minimum=1.0, maximum=120.0, cli=True, queue_case=True)
    obs3d_max_objects: int = option(10, minimum=1, maximum=40, cli=True, queue_case=True)
    obs3d_action_overlay: bool = option(False, cli=True, queue_case=True)
    pose_skill: bool = option(False, cli=True, queue_case=True)
    held_perception: bool = option(False, cli=True, queue_case=True)
    grasp_memory_dir: str = option(
        '~/.local/share/ni-rua/grasp-memory', path=True, cli=True, queue_case=True,
        cli_help='Persistent reviewed grasp memory directory',
    )
    record_video: bool = option(True, cli=True, queue_case=True)
    export_video: bool = option(True, cli=True, queue_case=True)
    terminal: bool = option(True, cli=True, queue_case=True)
    # --- Actor reply delivery (all agent backends; docs/development/tool-output-contract.md, "Reply shaping order") ---
    # budgeted_action_reply: kind="action" replies are laid out by decision value (execution result, scene change,
    # post-action observation, per-stage detail, trajectory feedback); when the agent host has a known text limit for
    # one tool output (AgentSession.tool_output_text_budget_bytes) and the text exceeds it, blocks are replaced, least
    # valuable first, by a reference naming the existing call that returns them. False: no layout, no replacement.
    budgeted_action_reply: bool = option(
        True, cli=True, queue_case=True,
        cli_help='Order action replies by decision value and fit their text into the host tool-output budget; '
                 'replaced blocks name the existing call that returns them (default: enabled)',
    )
    # Lossless compaction of finished actor replies (core/presentation/reply_compaction.py); programs are unaffected.
    # reply_same_as: a value repeated inside one text item is printed once, later copies point to it by JSON Pointer.
    # reply_float32: float32-exact numbers are printed with the shortest decimal that keeps the same float32.
    reply_same_as: bool = option(
        True, cli=True, queue_case=True,
        cli_help='Print a value repeated inside one reply text item once; later copies point to it by JSON Pointer '
                 '(default: enabled)',
    )
    reply_float32: bool = option(
        True, cli=True, queue_case=True,
        cli_help='Print float32-exact numbers with the shortest decimal that keeps the same float32 (default: enabled)',
    )
    # reply_static_once: explanatory strings whose text never changes are stated once in the robot_action description
    # and left out of routine replies, and the ways to read omitted detail become one {"detail": {"tool",
    # "arguments"}} form. Values, states, residuals and diagnostics are untouched.
    reply_static_once: bool = option(
        True, cli=True, queue_case=True,
        cli_help='State fixed explanatory strings once in the tool catalogue instead of in every routine reply and '
                 'give one detail-read form (default: enabled)',
    )
    # action_reply_profile: "summary_first" (default) gives an action reply one fixed shape: result, scene change,
    # observation and compact_3d_trajectory inline, joint samples, passing safety rows and the detail of cleanly
    # completed stages by reference, with the measured samples around the affected stage inline after a failure,
    # guard pause or slip; a stop at a checkpoint the model requested is routine. "full" delivers every block
    # (core/presentation/reply_budget.py).
    action_reply_profile: str = option(
        'summary_first', choices=('full', 'summary_first'), cli=True, queue_case=True,
        cli_help='Shape of action replies: a fixed summary with joint samples, passing safety rows and clean stage '
                 'detail by reference (summary_first, default) or every block inline (full)',
    )
    # reply_compact_encoding: the structures listed in reply_encoding.TABLE_STRUCTURES (trajectory samples, stage
    # rows, keyframes, object deltas ...) are printed as {"columns", "rows"}, an arm that did not move is stated
    # once, and finite-difference quantities are printed with 4 significant digits. Measurements keep their precision.
    reply_compact_encoding: bool = option(
        True, cli=True, queue_case=True,
        cli_help='Print trajectory samples and the other listed numeric structures of tool replies as column tables, '
                 'state an unchanged arm once and print finite-difference quantities with 4 significant digits '
                 '(default: enabled)',
    )

    def __getattr__(self, name: str):
        if name in _RETIRED_BLOAT_DEFAULTS:
            return _RETIRED_BLOAT_DEFAULTS[name]
        if name in _DEFAULT_MOTION_THRESHOLDS:
            thresholds = self.__dict__.get('_embodiment_thresholds')
            if thresholds is not None:
                return thresholds[name]
            return _DEFAULT_MOTION_THRESHOLDS[name]
        raise AttributeError(f'{type(self).__name__!r} object has no attribute {name!r}')

    def __post_init__(self):
        super().__post_init__()
        support = self.vggt_support_z_m
        if isinstance(support, str) and support != SUPPORT_Z_FROM_PROFILE:
            raise ConfigError(f'invalid_config_choice:FeaturesConfig.vggt_support_z_m: {support!r}')
        if not isinstance(support, str) and support is not None:
            if support < SUPPORT_Z_RANGE_M[0]:
                raise ConfigError('config_below_minimum:FeaturesConfig.vggt_support_z_m')
            if support > SUPPORT_Z_RANGE_M[1]:
                raise ConfigError('config_above_maximum:FeaturesConfig.vggt_support_z_m')
        if self.external_perception and not any(getattr(self, flag) for flag in _EXTERNAL_PERCEPTION_SWITCHES):
            for flag in _EXTERNAL_PERCEPTION_SWITCHES:
                object.__setattr__(self, flag, True)
        elif not self.external_perception and all(getattr(self, flag) for flag in _EXTERNAL_PERCEPTION_SWITCHES):
            for flag in _EXTERNAL_PERCEPTION_SWITCHES:
                object.__setattr__(self, flag, False)
            for flag in _EXTERNAL_PERCEPTION_OPT_IN_SWITCHES:
                object.__setattr__(self, flag, False)

    def to_dict(self):
        value = super().to_dict()
        if not value.get('program_seccomp_enabled'):
            value.pop('program_seccomp_enabled', None)
        if not value.get('programs_dir'):
            value.pop('programs_dir', None)
        return value


@dataclass(frozen=True)
class AnalysisConfig(Section):
    scene_truth_log: bool = False


@dataclass(frozen=True)
class TransportRecoveryConfig(Section):
    max_rounds: int = option(10, minimum=0, maximum=20)
    initial_delay_s: float = option(60, minimum=1)
    backoff_multiplier: float = option(2, minimum=1)
    max_delay_s: float = option(600, minimum=1)
    max_episode_delay_s: float = option(3600, minimum=0)
    transport_incomplete: bool = True


@dataclass(frozen=True)
class ModelConfig(Section):
    name: str = option('gpt-6.1-sol', cli='model', queue_case=True, cli_help='Actor model name')
    effort: str = option(
        'high', choices=('none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'ultra'),
        cli=True, queue_case=True, cli_help='Actor reasoning effort',
    )
    verbosity: str = option(
        'high', choices=('low', 'medium', 'high'),
        cli=True, queue_case=True, cli_help='Actor output verbosity',
    )
    codex_home: str = option(
        '~/.codex', path=True, cli=True, queue_case=True,
        cli_help='Source Codex configuration/auth directory (credentials are never embedded)',
    )
    price_card: str | None = option(
        path=True, cli=True,
        cli_help='JSON price schedule for an explicitly labelled estimate; defaults to NI_RUA_PRICE_CARD',
    )
    native_memories: bool = True
    transport_recovery: TransportRecoveryConfig = field(default_factory=TransportRecoveryConfig)


@dataclass(frozen=True)
class BackendConfig(Section):
    environment: str = 'robodojo'
    robot_model_id: str = 'arx-x5'
    root: str = option('~/code/robodojo', path=True)
    python: str = option('~/miniforge3/envs/RoboDojo/bin/python', path=True)
    deployment: str | None = option(None, path=True)

    def to_dict(self):
        # Unset deployment is omitted so RoboDojo/X5 manifests and snapshots stay byte-identical.
        value = super().to_dict()
        if value.get('deployment') is None:
            value.pop('deployment', None)
        return value


@dataclass(frozen=True)
class PolicyConfig(Section):
    python: str = option('~/.local/share/ni-rua-openpi-arx/bin/python', path=True)
    checkpoint: str = option('/home/ckpt/robodojo_pi05_official/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999', path=True)
    server_url: str | None = None


@dataclass(frozen=True)
class ImagesConfig(Section):
    feedback_policy: str = option(
        'all', choices=('all', 'adaptive'),
        cli='image_feedback_policy', queue_case=True,
        cli_help='Automatic action camera delivery; explicit observations retain all requested views',
    )
    action_feedback_policy: str = option(
        'all', choices=('on_anomaly', 'none', 'focus_roi', 'head_only', 'adaptive', 'all'),
        cli='action_image_feedback_policy', queue_case=True,
        cli_help='Action/wait camera image delivery policy (default all: deliver updated camera views; on_anomaly omits routine images on clean completion)',
    )
    transport_format: str = option('webp', choices=('webp', 'jpeg'))
    webp_quality: int = option(85, minimum=1, maximum=100)
    webp_method: int = option(4, minimum=0, maximum=6)
    jpeg_quality: int = option(95, minimum=1, maximum=100)
    storage_jpeg_quality: int = option(95, minimum=1, maximum=100)
    reuse_exact_images: bool = True
    trim_oversized_requests: bool = True
    trim_max_inline_images: int = option(60, minimum=0, maximum=1000)


@dataclass(frozen=True)
class LibraryConfig(Section):
    root: str = option('~/exp/ni-rua', path=True)
    port: int = option(8780, minimum=0, maximum=65535)
    scan_interval: float = option(1, minimum=0.1, maximum=60)
    public_origin: str | None = None
    task_source_root: str | None = option(path=True)


@dataclass(frozen=True)
class OomRetryConfig(Section):
    max_retries: int = option(2, minimum=0, maximum=20)
    initial_delay_s: float = option(60, minimum=1)
    backoff_multiplier: float = option(2, minimum=1)
    max_delay_s: float = option(300, minimum=1)


@dataclass(frozen=True)
class SceneConfig(Section):
    root: str = option('runs/local_scene', path=True)
    port: int = option(8770, minimum=0, maximum=65535)
    max_workers: int = option(2, minimum=1)
    max_queue: int = option(4, minimum=1)
    max_blob_store_bytes: int = option(536870912, minimum=1)


@dataclass(frozen=True)
class QueueConfig(Section):
    max_parallel: int | None = option(60, minimum=1)
    eligible_gpu_indices: tuple[int, ...] | None = None
    memory_per_case_mib: int = option(10240, minimum=1)
    gpu_headroom_mib: int = option(0, minimum=0)
    wait_timeout_s: float = option(1800, minimum=0.001, queue_cli=True)
    oom_retry: OomRetryConfig = field(default_factory=OomRetryConfig)
    wake_enabled: bool = option(True, queue_cli=True)
    wake_thread: str | None = option(
        None, queue_cli=True, cli_help='Main Codex thread UUID; defaults to ~/.config/ni-rua/completion-thread',
    )
    wake_codex: str = option(
        'codex', queue_cli=True,
        cli_help='Codex executable for completion notification; use an absolute path to bypass CLI wrappers',
    )
    wake_remote: str | None = option(
        None, queue_cli=True,
        cli_help="Completion notification app-server endpoint; defaults to the user's shared socket",
    )

    def __post_init__(self):
        super().__post_init__()
        if self.eligible_gpu_indices is not None:
            indices = self.eligible_gpu_indices
            if (not indices or any(type(i) is not int or i < 0 for i in indices)
                    or len(set(indices)) != len(indices)):
                raise ConfigError('invalid_eligible_gpu_indices')


_REMOTE_NAME = re.compile(r'^[A-Za-z0-9_-]{1,32}$')
_REMOTE_ENV = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*=')
_REMOTE_GPU = re.compile(r'^(auto|[0-9]+(,[0-9]+)*)$')


@dataclass(frozen=True)
class RemoteServiceConfig(Section):
    """One HTTP service on the remote host, reached through an SSH forward 127.0.0.1:local_port -> host 127.0.0.1:remote_port."""
    name: str
    remote_port: int = field(metadata=dict(choices=None, minimum=1, maximum=65535, path=False))
    local_port: int | None = option(None, minimum=1, maximum=65535)
    health_path: str = '/health'
    command: str | None = None
    workdir: str | None = None
    env: tuple[str, ...] = ()
    gpu: str | None = None
    gpu_min_free_mib: int = option(15360, minimum=1, maximum=1048576)
    startup_timeout_s: float = option(300.0, minimum=0.001, maximum=3600.0)
    reuse_if_healthy: bool = True
    stop: str | None = None
    stop_timeout_s: float = option(60.0, minimum=0.001, maximum=600.0)

    def __post_init__(self):
        super().__post_init__()
        if not _REMOTE_NAME.match(self.name):
            raise ConfigError(f'invalid_remote_service_name: {self.name!r}')
        if not self.health_path.startswith('/'):
            raise ConfigError(f'remote_health_path_must_start_with_slash:{self.name}')
        if any(not _REMOTE_ENV.match(item) for item in self.env):
            raise ConfigError(f'remote_env_must_be_NAME=value:{self.name}')
        if self.gpu is not None and not _REMOTE_GPU.match(self.gpu):
            raise ConfigError(f'remote_gpu_must_be_auto_or_index_list:{self.name}: {self.gpu!r}')
        if self.gpu is not None and self.command is None:
            raise ConfigError(f'remote_gpu_requires_command:{self.name}')

    @property
    def effective_local_port(self):
        return self.local_port or self.remote_port


@dataclass(frozen=True)
class RemoteConfig(Section):
    """Remote GPU services driven by the config (docs/development/so101-integration.md section 9)."""
    enabled: bool = option(
        False, cli='remote',
        cli_help='Override remote.enabled: connect/start the remote GPU services declared in the config (--no-remote runs without them)',
    )
    host: str | None = None
    services: tuple[RemoteServiceConfig, ...] = ()
    required: bool = False
    stop_on_exit: bool = False
    health_interval_s: float = option(30.0, minimum=1.0, maximum=3600.0)
    poll_interval_s: float = option(2.0, minimum=0.001, maximum=60.0)
    connect_timeout_s: float = option(10.0, minimum=1.0, maximum=120.0)
    command_timeout_s: float = option(60.0, minimum=1.0, maximum=600.0)
    log_dir: str = '~/.cache/ni-rua/remote'
    ssh: str = 'ssh'
    remote_repo: str | None = None
    record: str | None = option(None, path=True)

    def __post_init__(self):
        super().__post_init__()
        if not self.enabled:
            return
        if self.host is None:
            raise ConfigError('remote_enabled_requires_host')
        if not self.services:
            raise ConfigError('remote_enabled_requires_services')
        names = [item.name for item in self.services]
        if len(set(names)) != len(names):
            raise ConfigError('remote_service_names_must_be_unique')
        ports = [item.effective_local_port for item in self.services]
        if len(set(ports)) != len(ports):
            raise ConfigError('remote_local_ports_must_be_unique')


ENV_NAME_PATTERN = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
ANTHROPIC_BASE_URL = 'https://api.anthropic.com'


@dataclass(frozen=True)
class UpstreamConfig(Section):
    """Where an agent backend's model requests are forwarded by the local accounting relay."""
    source: str = option('explicit', choices=('explicit', 'codex_home', 'claude_settings', 'opencode_config', 'profile'))
    settings: str | None = option(None, path=True)
    wire_api: str | None = option(None, choices=('anthropic_messages', 'openai_responses', 'openai_chat'))
    base_url: str | None = None
    api_key_env: str | None = None
    auth_scheme: str = option('bearer', choices=('bearer', 'x-api-key'))
    provider: str | None = None
    tool_result_images: str = option('inline', choices=('inline', 'user_blocks'))
    tool_schema_max_depth: int | None = option(None, minimum=3)

    def __post_init__(self):
        super().__post_init__()
        if self.base_url is not None:
            try:
                parts = urlsplit(self.base_url)
                parts.port  # Raises ValueError for a malformed port.
            except ValueError as error:
                raise ConfigError('invalid_upstream_base_url') from error
            if (parts.scheme not in ('http', 'https') or not parts.hostname or parts.username is not None
                    or parts.password is not None or parts.query or parts.fragment or '?' in self.base_url
                    or '#' in self.base_url):
                raise ConfigError('invalid_upstream_base_url: http(s) URL without credentials, query or fragment')
        if self.api_key_env is not None and not ENV_NAME_PATTERN.match(self.api_key_env):
            raise ConfigError('invalid_upstream_api_key_env: expected an environment variable NAME, never the key')
        if self.source == 'claude_settings' and not self.settings:
            raise ConfigError('upstream_claude_settings_requires_settings: set upstream.settings to a Claude Code '
                              'settings.json path')
        if self.source == 'opencode_config' and not self.settings:
            raise ConfigError('upstream_opencode_config_requires_settings: set upstream.settings to an OpenCode '
                              'config JSON path')
        if self.source == 'profile' and not self.settings:
            raise ConfigError('upstream_profile_requires_settings: set upstream.settings to a provider profile path '
                              '(<profiles_dir>/<id>.local.json)')


def _effective_context_delivery(section):
    """返回 claude_code / opencode 会话生效的上下文投递模式（``AgentSession.context_delivery``）。
    主链与子链已通过作用域隔离，不再因 ``subagents=True`` 强制覆写为 ``'full'``。"""
    return section.context_delivery


def _anthropic_upstream():
    return UpstreamConfig(wire_api='anthropic_messages', base_url=ANTHROPIC_BASE_URL, api_key_env='ANTHROPIC_API_KEY',
                          auth_scheme='x-api-key', provider='anthropic')


@dataclass(frozen=True)
class ClaudeCodeConfig(Section):
    """Settings of the Claude Code agent backend (agent.backend = "claude_code"; optional extra ni-rua-harness[claude])."""
    cli_path: str | None = option(path=True)
    upstream: UpstreamConfig = field(default_factory=_anthropic_upstream)
    context_delivery: str = option('delta', choices=('full', 'delta'))
    subagents: bool = True
    tool_search: bool = False
    auto_compact_window_tokens: int | None = option(None, minimum=100000, maximum=1000000)
    effective_context_delivery = property(_effective_context_delivery)

    def to_dict(self):
        value = super().to_dict()
        if not value.get('tool_search'):
            value.pop('tool_search', None)
        return value


@dataclass(frozen=True)
class OpenCodeConfig(Section):
    """Settings of the OpenCode agent backend (agent.backend = "opencode"; binary installed outside the Python env)."""
    executable: str | None = option(path=True)
    upstream: UpstreamConfig = field(default_factory=UpstreamConfig)
    context_limit_tokens: int | None = option(None, minimum=1)
    output_limit_tokens: int | None = option(None, minimum=1)
    thinking_display: str | None = option(None, choices=('summarized', 'omitted'))
    context_delivery: str = option('delta', choices=('full', 'delta'))
    subagents: bool = True
    tool_search: bool = False
    allow_shell: bool = True
    allow_external_read: bool = True
    effective_context_delivery = property(_effective_context_delivery)

    def __post_init__(self):
        super().__post_init__()
        if self.thinking_display is not None and self.upstream.wire_api != 'anthropic_messages':
            raise ConfigError('opencode_thinking_display_requires_anthropic_messages: set '
                              'agent.opencode.thinking_display=null on another wire')

    def to_dict(self):
        value = super().to_dict()
        if not value.get('tool_search'):
            value.pop('tool_search', None)
        if value.get('allow_shell') is True:
            value.pop('allow_shell', None)
        if value.get('allow_external_read') is True:
            value.pop('allow_external_read', None)
        return value


@dataclass(frozen=True)
class ExternalConfig(Section):
    """Settings of the external agent backend (agent.backend = "external")."""
    mcp_port: int = option(
        8790, minimum=0, maximum=65535, cli='external_mcp_port',
        cli_help='Override agent.external.mcp_port: loopback MCP port for --agent-backend external (default 8790; 0 chooses a free port)',
    )
    context_delivery: str = option('delta', choices=('full', 'delta'))


EVIDENCE_UPSTREAM_WIRES = ('openai_responses', 'anthropic_messages')


@dataclass(frozen=True)
class EvidenceConfig(Section):
    """Client of the evidence helper (features.visual_review_model; design 04 section 9)."""
    client: str = option('native', choices=('native', 'relay'))
    upstream: UpstreamConfig | None = None

    def __post_init__(self):
        super().__post_init__()
        upstream = self.upstream
        if upstream is None:
            return
        if upstream.wire_api == 'openai_chat':
            raise ConfigError('evidence_upstream_wire_not_implemented:openai_chat: choose openai_responses or '
                              'anthropic_messages for agent.evidence.upstream.wire_api')
        if upstream.tool_result_images != 'inline' or upstream.tool_schema_max_depth is not None:
            raise ConfigError('evidence_upstream_tool_options_unsupported: tool_result_images and '
                              'tool_schema_max_depth apply to the actor upstream only')
        if upstream.source not in ('explicit', 'codex_home'):
            raise ConfigError(f'evidence_upstream_source_unsupported:{upstream.source}: use "explicit" or "codex_home" '
                              'for agent.evidence.upstream')
        if upstream.source == 'codex_home':
            if upstream.wire_api not in (None, 'openai_responses'):
                raise ConfigError('evidence_upstream_codex_home_requires_openai_responses')
        elif upstream.wire_api not in EVIDENCE_UPSTREAM_WIRES or upstream.base_url is None:
            raise ConfigError('evidence_upstream_requires_wire_api_and_base_url: set agent.evidence.upstream.wire_api '
                              'and base_url, or source "codex_home"')


@dataclass(frozen=True)
class AgentConfig(Section):
    """Which agent runtime drives the actor. Default codex; the section is omitted from snapshots while unchanged."""
    backend: str = option(
        'codex', choices=('codex', 'claude_code', 'opencode', 'external'),
        cli='agent_backend', queue_case=True,
        cli_help='Override agent.backend: the agent runtime that drives the actor (default codex); external = NI-RUA starts no agent, your own agent connects to http://127.0.0.1:<agent.external.mcp_port>/mcp (interactive runs only); other agent.* settings besides --external-mcp-port are config-file only',
    )
    claude_code: ClaudeCodeConfig = field(default_factory=ClaudeCodeConfig)
    opencode: OpenCodeConfig = field(default_factory=OpenCodeConfig)
    external: ExternalConfig = field(default_factory=ExternalConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    codex_upstream: UpstreamConfig | None = None
    profiles_dir: str | None = option(None, path=True)

    def __post_init__(self):
        super().__post_init__()
        if self.backend == 'codex' and self.evidence.client == 'native' and self.evidence.upstream is not None:
            raise ConfigError('evidence_upstream_requires_relay_client: set agent.evidence.client="relay" or remove '
                              'agent.evidence.upstream (the native Codex evidence agent uses the actor provider)')
        upstream = self.codex_upstream
        if upstream is not None:
            if upstream.source != 'profile':
                raise ConfigError(f'codex_upstream_source_unsupported:{upstream.source}: agent.codex_upstream takes '
                                  'source "profile" only; null keeps model.codex_home')
            if upstream.wire_api not in (None, 'openai_responses'):
                raise ConfigError('codex_upstream_requires_openai_responses: Codex runs on the openai_responses wire')
            if upstream.tool_result_images != 'inline' or upstream.tool_schema_max_depth is not None:
                raise ConfigError('codex_upstream_tool_options_unsupported: tool_result_images and '
                                  'tool_schema_max_depth apply to the claude_code / opencode relay only')

    @property
    def context_delivery(self):
        """返回当前所选后端的上下文投递模式（'delta' 或 'full'；对应 ``AgentSession.context_delivery``）：
        codex 固定为 'delta'，claude_code / opencode 取各节 ``effective_context_delivery``，external 取
        ``self.external.context_delivery``（四后端默认均为 'delta'）。工具目录据此决定是否声明 delta 投递。"""
        if self.backend in ('claude_code', 'opencode'):
            return getattr(self, self.backend).effective_context_delivery
        if self.backend == 'external':
            return self.external.context_delivery
        return 'delta'

    def to_dict(self):
        # Default sections are omitted so agent snapshots of earlier checkouts stay identical.
        return _omit_agent_defaults(self, super().to_dict())


def _omit_agent_defaults(agent, value):
    """``value`` without members omitted while holding defaults."""
    if 'opencode' in value:
        value['opencode'] = agent.opencode.to_dict()
    if 'claude_code' in value:
        value['claude_code'] = agent.claude_code.to_dict()
    if agent.evidence == EvidenceConfig():
        value.pop('evidence', None)
    if agent.external == ExternalConfig():
        value.pop('external', None)
    if agent.codex_upstream is None:
        value.pop('codex_upstream', None)
    if agent.profiles_dir is None:
        value.pop('profiles_dir', None)
    return value


def _section_type(annotation):
    """The Section dataclass of a field annotated ``S`` or ``S | None``; None for any other annotation."""
    if is_dataclass(annotation):
        return annotation
    if get_origin(annotation) in (Union, types.UnionType):
        options = [item for item in get_args(annotation) if item is not type(None)]
        if len(options) == 1 and is_dataclass(options[0]):
            return options[0]
    return None


def _tuple_item_type(annotation):
    """The element type ``T`` of ``tuple[T, ...]`` or ``tuple[T, ...] | None``; None otherwise."""
    if get_origin(annotation) is tuple:
        return get_args(annotation)[0]
    if get_origin(annotation) in (Union, types.UnionType):
        tuple_args = [item for item in get_args(annotation) if get_origin(item) is tuple]
        if len(tuple_args) == 1:
            return get_args(tuple_args[0])[0]
    return None


def _construct(cls, value, prefix='config', base=None):
    if not isinstance(value, dict):
        raise ConfigError(f'config_object_required:{prefix}')
    if cls is FeaturesConfig:
        value = {key: item for key, item in value.items() if key not in RETIRED_FEATURE_FIELDS}
    known = {item.name: item for item in fields(cls)}
    unknown = set(value) - known.keys()
    if unknown:
        raise ConfigError(f'unknown_config_keys:{prefix}: {", ".join(sorted(unknown))}')
    required = [name for name, item in known.items() if item.default is MISSING and item.default_factory is MISSING]
    missing = [name for name in required if name not in value]
    if missing:
        raise ConfigError(f'missing_config_keys:{prefix}: {", ".join(missing)}')
    hints = _section_hints(cls)
    values = {}
    for key, item in known.items():
        annotation = hints[key]
        if key not in value:
            continue
        part = value[key]
        section = _section_type(annotation)
        tuple_inner = _tuple_item_type(annotation)
        if section is not None and (part is not None or is_dataclass(annotation)):
            part = _construct(section, part, prefix + '.' + key, base)
        elif tuple_inner is not None and isinstance(part, list):
            part = tuple(_construct(tuple_inner, entry, f'{prefix}.{key}[{index}]', base)
                         if isinstance(tuple_inner, type) and is_dataclass(tuple_inner) else entry
                         for index, entry in enumerate(part))
        if item.metadata.get('path') and isinstance(part, str) and part.strip():
            path = Path(part).expanduser()
            part = _absolute_path((base or Path.cwd()) / path)
        values[key] = part
    return cls(**values)


def _absolute_path(candidate):
    """Absolute, symlink-resolved path string of a config path option.

    macOS mounts ``/home`` as an ``autofs`` map (``auto_home``): ``realpath`` on a Linux-style ``/home/...`` default
    (remote-host checkpoints, caches) makes ``automountd`` answer every stat (~25 ms each, serialised across
    processes). ``/home`` is a synthetic firmlink to ``/System/Volumes/Data/home`` there, so that exact result is
    composed lexically without touching the mount. Every other path (relative, ``/tmp`` -> ``/private/tmp``,
    symlinked homes) keeps ``Path.resolve``.
    """
    if sys.platform == 'darwin' and str(candidate).startswith('/home/'):
        lexical = os.path.abspath(str(candidate))
        if lexical.startswith('/home/'):  # a '..' that climbs out of /home must still resolve on the real filesystem
            return '/System/Volumes/Data' + lexical
    return str(candidate.resolve())


def _merge(base, overrides):
    result = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _handoff_compat(value):
    """An explicit legacy bool or master external_perception switch keeps its strategy across config layers."""
    features = value.get('features')
    if isinstance(features, dict):
        updated = dict(features)
        if type(updated.get('airborne_handoff')) is bool and 'handoff_preference' not in updated:
            updated['handoff_preference'] = 'airborne' if updated['airborne_handoff'] else 'surface'
        if type(updated.get('external_perception')) is bool:
            master = updated['external_perception']
            for flag in _EXTERNAL_PERCEPTION_SWITCHES:
                updated.setdefault(flag, master)
            if not master:
                for flag in _EXTERNAL_PERCEPTION_OPT_IN_SWITCHES:
                    updated.setdefault(flag, False)
            if 'vggt_height_mode' not in updated:
                updated['vggt_height_mode'] = 'optional'
        if updated != features:
            value = {**value, 'features': updated}
    return value
