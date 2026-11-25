"""Static registrations; a new ToolSpec requires no dispatcher edits."""
from copy import deepcopy
from functools import partial

from ...perception.depth.depth_runtime import DEPTH_KINDS, DEPTH_TOOLS, GEOMETRY_KINDS as DEPTH_MEASUREMENT_KINDS
from ...core.observation_policy import DEVELOPMENT_TOOLS, allows_depth, has_calibration
from . import handlers
from .definitions import public_definitions
from .output_contracts import CONTRACTS, ENVELOPE, UNCONSTRAINED
from .registry import ToolRegistry, ToolSpec

# 几何/深度副作用四分法：纯读 vs 写测量（depth_runtime._measure_locked）、改目标表、重锚轨迹
_DEPTH_TARGET_WRITE_KINDS = frozenset({'register_target', 'refresh_target', 'forget_target'})
_GEOMETRY_REANCHOR_KINDS = frozenset({'reanchor_tracks'})
_DEPTH_WRITE_KINDS = DEPTH_MEASUREMENT_KINDS | _DEPTH_TARGET_WRITE_KINDS
_GEOMETRY_WRITE_KINDS = _DEPTH_WRITE_KINDS | _GEOMETRY_REANCHOR_KINDS


# Retired tool names remain valid disabled_tools entries so archived configurations still launch.
CONFIG_TOOL_NAMES = frozenset({
    'depth_move', 'grasp_memory', 'grasp_plan', 'manipulation_run', 'operation_verify', 'robot_execute_ee',
    'robot_execute_plan', 'robot_execute_queue', 'robot_gripper', 'robot_inspect_trajectory',
    'robot_move_dual_grasp', 'robot_move_ee', 'robot_move_grasp', 'robot_preview_cost', 'robot_preview_motion',
    'scene_capabilities', 'scene_job_status', 'scene_query', 'scene_snapshot', 'scene_update', 'skill_run',
})


def unavailable(name, runtime, args):
    profile = runtime.observation_profile
    if name in DEVELOPMENT_TOOLS | {'robot_contact_overlay', 'robot_action'} and not has_calibration(profile):
        # A backend whose action poses come from its own proprioceptive frame
        # (not image geometry) may declare that robot_action needs no camera
        # calibration. Absent declaration keeps the historical gate (ARX X5).
        unified = (runtime.capabilities or {}).get('unified_action', {})
        if not (name == 'robot_action' and isinstance(unified, dict)
                and unified.get('requires_camera_calibration') is False):
            return 'tool_unavailable_without_permitted_calibration'
    depth_query = name == 'depth_query'
    if not depth_query and name == 'geometry_query':
        leaves = handlers.expand_geometry_requests(args.get('request'), strict=False)
        depth_query = bool(leaves) and any(
            isinstance(leaf, dict) and isinstance(leaf.get('kind'), str) and leaf.get('kind') in DEPTH_KINDS
            for leaf in leaves
        )
    if name in DEPTH_TOOLS or depth_query:
        if not allows_depth(profile):
            return 'depth_not_enabled_for_run'
        if depth_query and 'depth_query' in runtime.disabled_tools:
            return 'tool_disabled_by_config:depth_query'
    if name == 'robot_action':
        capability = (runtime.capabilities or {}).get('unified_action', {})
        if capability.get('available') is False or runtime.features.motion_execution_mode != 'resampled':
            return 'unified_action_unavailable_for_backend'
    if name == 'robot_wait' and not runtime.enable_wait:
        return 'bounded_wait_not_enabled_for_run'
    if name == 'vision_compare' and not runtime.vision_compare:
        return 'vision_compare_not_enabled_for_run'
    if name in {'vision_review', 'operation_verify'} and not runtime.visual_review:
        return 'visual_review_not_enabled_for_run'
    if name == 'vision_review' and args.get('evidence_mode') == 'operation' and 'operation_verify' in runtime.disabled_tools:
        return 'tool_disabled_by_config:operation_verify'
    if name == 'robot_policy_step' and not runtime.policy:
        return 'policy_not_enabled_for_run'
    if name in {'program_run', 'program_status'} and not runtime.programs:
        return 'program_runner_not_started'
    return None


def action_effect(args):
    mode = args.get('mode', 'preview')
    if mode in ('preview', 'preflight') and args.get('auto_execute_if_clean'):
        return 'physical'
    # reach: IK queries from the measured joints; no plan record, no motion (unified_actions._reach).
    return {
        'preview': 'plan',
        'preflight': 'plan',
        'inspect': 'read',
        'reach': 'read',
        'execute': 'physical',
        'direct': 'physical',
        'execute_if_clean': 'physical',
    }[mode]


def memory_effect(args):
    request = args.get('request', {})
    if request.get('kind') in {'help', 'query'}:
        return 'read'
    return 'shared_memory' if request.get('scope') == 'shared' else 'record'


def plan_effect(args):
    return 'read' if args.get('operation') == 'read' else 'record'


def geometry_effect(args):
    request = args.get('request', {})
    if not isinstance(request, dict):
        raise TypeError('geometry_request_must_be_object')
    leaves = handlers.expand_geometry_requests(request, strict=False)
    if leaves is None:
        return 'record'
    for leaf in leaves:
        kind = leaf.get('kind') if isinstance(leaf, dict) else None
        if isinstance(kind, str) and kind in _GEOMETRY_WRITE_KINDS:
            return 'record'
    return 'read'


def pose_plan_effect(args):
    """pose_plan 仅 grasp 分支写入回合槽 POSE_SKILL_GRASPS_SLOT，其余为纯读。"""
    request = args.get('request')
    return 'record' if isinstance(request, dict) and request.get('kind') == 'grasp' else 'read'


# 按请求判定的 ToolSpec.parallel 分类器：True 表示不修改任何 Harness/世界/回合状态，可与其他只读调用并发。
def geometry_parallel(args):
    request = args.get('request')
    if not isinstance(request, dict):
        return False
    leaves = handlers.expand_geometry_requests(request, strict=False)
    if leaves is None:
        return False
    for leaf in leaves:
        if not isinstance(leaf, dict):
            return False
        kind = leaf.get('kind')
        if kind is not None and not isinstance(kind, str):
            return False
        if kind in _GEOMETRY_WRITE_KINDS:
            return False
    return True


def depth_parallel(args):
    # target_status / skill_help / help 为纯读；写测量与改目标表的深度子项必须串行。
    request = args.get('request')
    if not isinstance(request, dict):
        return False
    kind = request.get('kind')
    if kind is not None and not isinstance(kind, str):
        return False
    return kind not in _DEPTH_WRITE_KINDS


def pose_plan_parallel(args):
    # grasp 会写入回合槽，必须串行；targets 与 place（只读回合槽）允许并发。
    request = args.get('request')
    return isinstance(request, dict) and request.get('kind') in {'targets', 'place'}


def plan_parallel(args):
    return args.get('operation') == 'read'


def memory_parallel(args):
    request = args.get('request')
    return isinstance(request, dict) and request.get('kind') in {'help', 'query'}


def output_schemas(name):
    # Field contracts describe typed results; they are not a second robot/job
    # state machine. Owners still validate observations, plans and receipts.
    if name in UNCONSTRAINED:
        return dict(data=dict(description='Compatibility: ' + UNCONSTRAINED[name] + '; only JSON serialization is required.'))
    if name in CONTRACTS:
        return deepcopy(CONTRACTS[name])
    return dict(data=deepcopy(ENVELOPE))


def catalog_profile_of(config):
    """How the run's tool descriptions state the input contract: ``config.features.tool_catalog_profile``
    (``full_v1`` for a runtime without a config object, which also has none of the other feature settings; every
    Harness holds a Config, so this is never the production default). A layered profile refers the actor to a reference file in its working directory;
    an agent NI-RUA does not run (``agent.backend`` ``"external"``) has no such directory, so it gets ``compact_v1``,
    which describes every field in the tool description itself."""
    from .catalog_guide import LAYERS
    profile = getattr(getattr(config, 'features', None), 'tool_catalog_profile', 'full_v1')
    if profile in LAYERS and getattr(getattr(config, 'agent', None), 'backend', None) == 'external':
        return 'compact_v1'
    return profile


def create_registry(extra_specs=(), *, runtime=None):
    from ...perception.external.vggt_height import mode as vggt_height_mode
    from ...perception.external.object_table import enabled as object_table_enabled
    from ...perception.external.reanchor_tracks import enabled as reanchor_tracks_enabled
    from ...perception.external.scene3d import enabled as scene3d_enabled
    from ...perception.external.obs3d import flags as obs3d_flags, action_overlay as obs3d_action_overlay
    from ...perception.external.pose_skill import enabled as pose_skill_enabled
    selected, object_table, scene3d = vggt_height_mode(runtime), object_table_enabled(runtime), scene3d_enabled(runtime)
    reanchor_tracks = reanchor_tracks_enabled(runtime)
    observe3d = obs3d_flags(runtime)
    if any(observe3d):
        observe3d = (*observe3d, obs3d_action_overlay(runtime))
    pose_skill = pose_skill_enabled(runtime)
    # The run's embodiment (Harness sets it before creating the registry; None = the default robot, ARX-X5): every
    # published definition, including the robot_action schema dispatch validates against, is rendered from it.
    from ...core.embodiment import EmbodimentProfile
    profile = getattr(runtime, 'embodiment', None)
    profile = profile if isinstance(profile, EmbodimentProfile) else None
    # The run's delivery mode (config.agent.context_delivery; 'delta' without a config): a description claims that
    # summary replies leave out unchanged static metadata only where the Harness does that.
    agent = getattr(getattr(runtime, 'config', None), 'agent', None)
    delivery = 'full' if getattr(agent, 'context_delivery', None) == 'full' else 'delta'
    catalog_profile = catalog_profile_of(getattr(runtime, 'config', None))
    definitions = {item['name']: item for item in (public_definitions(profile=profile, vggt_height=selected or False,
                                                                      object_table=object_table, scene3d=scene3d,
                                                                      observe3d=observe3d, pose_skill=pose_skill,
                                                                      reanchor_tracks=reanchor_tracks,
                                                                      context_delivery=delivery,
                                                                      catalog_profile=catalog_profile)
                   if (selected or object_table or scene3d or any(observe3d) or reanchor_tracks)
                   else public_definitions(profile=profile, pose_skill=pose_skill, context_delivery=delivery,
                                           catalog_profile=catalog_profile))}
    effects = dict(robot_action=action_effect, robot_wait='physical', robot_policy_step='physical',
        program_run='physical', task_plan=plan_effect, operation_memory=memory_effect,
        history_annotate_action='record', depth_query=geometry_effect, geometry_query=geometry_effect,
        pose_plan=pose_plan_effect)
    parallel = dict(geometry_query=geometry_parallel, depth_query=depth_parallel,
        task_plan=plan_parallel, operation_memory=memory_parallel, pose_plan=pose_plan_parallel,
        robot_information=True, robot_contact_overlay=True, robot_action_status=True, robot_capabilities=True,
        history_list_actions=True, history_view_observation=True, history_read_trace=True, program_status=True,
        vision_compare=True)
    specs = []
    for name, definition in definitions.items():
        roles = {'actor', 'program', 'trusted_legacy'}
        if name in {'program_run', 'program_status', 'robot_policy_step'}:
            roles.remove('program')
        legacy_schema = None
        if name == 'task_plan':
            # These trusted adapters already provide their historical field diagnostics.
            legacy_schema = {'type': 'object'}
        specs.append(ToolSpec(definition, getattr(handlers, name), frozenset(roles),
            effects.get(name, 'read'), partial(unavailable, name), legacy_schema,
            api_version='1.1' if name == 'robot_action' else '1.0',
            output_schemas=output_schemas(name), parallel=parallel.get(name, False)))
    return ToolRegistry([*specs, *extra_specs])
