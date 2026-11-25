"""The readable Harness API implementation, without backends or experiment data."""
from pathlib import Path
import shutil


# Explicit publication keeps evaluators, backends, apps, and observer artifacts private,
# while forming a complete import-time closure for every published module in /harness.
SOURCE_FILES = (
    "__init__.py",
    "core/__init__.py",
    "core/action_validation.py",
    "core/config/__init__.py",
    "core/config/hooks.py",
    "core/config/loader.py",
    "core/config/sections.py",
    "core/config/thresholds.py",
    "core/contracts.py",
    "core/embodiment.py",
    "core/enrichment.py",
    "core/episode_state.py",
    "core/observation_policy.py",
    "core/pose_math.py",
    "core/schema_guide.py",
    "core/presentation/__init__.py",
    "core/presentation/actor_output.py",
    "core/presentation/actor_payload.py",
    "core/presentation/actor_presentation.py",
    "core/presentation/capability_output.py",
    "core/presentation/feedback_delivery.py",
    "core/presentation/history_output.py",
    "core/presentation/observation_output.py",
    "core/presentation/program_output_delivery.py",
    "core/presentation/reply_budget.py",
    "core/presentation/reply_compaction.py",
    "core/presentation/reply_encoding.py",
    "core/storage/__init__.py",
    "core/storage/images.py",
    "core/storage/store.py",
    "core/tool_errors.py",
    "core/tool_output.py",
    "cortex/__init__.py",
    "cortex/agents/__init__.py",
    "cortex/agents/identity.py",
    "cortex/context/__init__.py",
    "cortex/context/actor_instructions.py",
    "cortex/context/operation_verification.py",
    "cortex/context/task_notes.py",
    "cortex/context/task_planning.py",
    "cortex/session/__init__.py",
    "cortex/session/system_runtime.py",
    "perception/__init__.py",
    "perception/depth/__init__.py",
    "perception/depth/depth_runtime.py",
    "perception/external/__init__.py",
    "perception/external/_service_http.py",
    "perception/external/clearance_oracle.py",
    "perception/external/contact_height.py",
    "perception/external/field_ops.py",
    "perception/external/geometry_ops.py",
    "perception/external/held_perception.py",
    "perception/external/ik_prefilter.py",
    "perception/external/local_kdtree_field.py",
    "perception/external/object_table.py",
    "perception/external/obs3d.py",
    "perception/external/pointcloud_field.py",
    "perception/external/pose_math.py",
    "perception/external/pose_skill.py",
    "perception/external/rectify.py",
    "perception/external/robot_view.py",
    "perception/external/scene3d_helpers.py",
    "perception/external/sensor_frame_contract.py",
    "perception/external/vectorized_grasp.py",
    "perception/external/virtual_field.py",
    "perception/external/wrist_progress_cache.py",
    "perception/focus_3d.py",
    "perception/scene/__init__.py",
    "perception/scene/scene_contracts.py",
    "perception/scene/store.py",
    "perception/world_model.py",
    "robot/__init__.py",
    "robot/embodiments/__init__.py",
    "robot/embodiments/arx_x5/__init__.py",
    "robot/embodiments/arx_x5/defaults.json",
    "robot/embodiments/arx_x5/grasp_contact.py",
    "robot/embodiments/arx_x5/pose_convention.py",
    "robot/embodiments/arx_x5/profile.py",
    "robot/embodiments/libero_franka_panda/__init__.py",
    "robot/embodiments/libero_franka_panda/defaults.json",
    "robot/embodiments/libero_franka_panda/grasp_contact.py",
    "robot/embodiments/libero_franka_panda/pose_convention.py",
    "robot/embodiments/libero_franka_panda/profile.py",
    "robot/embodiments/so101/__init__.py",
    "robot/embodiments/so101/gripper_cad.py",
    "robot/embodiments/so101/profile.py",
    "robot/model/__init__.py",
    "robot/model/arx_contract.py",
    "robot/model/calibration_correction.py",
    "robot/model/embodiments.py",
    "robot/model/end_effectors.py",
    "robot/model/geometry.py",
    "robot/model/grasp_contact.py",
    "robot/model/kinematics.py",
    "robot/model/roboprobe_pose.py",
    "robot/model/robot_information_data/arx_x5/physical.json",
    "robot/model/robot_information_data/libero_franka_panda/physical.json",
    "robot/model/robot_information_data/so101/physical.json",
    "robot/model/robot_model.py",
    "robot/model/so101_gripper_geometry.py",
    "robot/model/so101_kinematics.py",
    "robot/motion/__init__.py",
    "robot/motion/action_compilation.py",
    "robot/motion/arx_actions.py",
    "robot/motion/batch_continuation.py",
    "robot/motion/batch_contract.py",
    "robot/motion/batch_geometry.py",
    "robot/motion/batch_motion.py",
    "robot/motion/batch_scheduler.py",
    "robot/motion/contact_diagnostics.py",
    "robot/motion/contact_reanchor.py",
    "robot/motion/feedback_motion.py",
    "robot/motion/grip_guard.py",
    "robot/motion/motion_review.py",
    "robot/motion/motion_safety.py",
    "robot/motion/motion_speed.py",
    "robot/motion/motion_tolerance.py",
    "robot/motion/proxy_contract.py",
    "robot/motion/reachability.py",
    "robot/motion/scene_obstacles.py",
    "runtime/__init__.py",
    "runtime/actions/__init__.py",
    "runtime/actions/action_state_ledger.py",
    "runtime/actions/batch_repair.py",
    "runtime/actions/grasp_preview_3d.py",
    "runtime/actions/home_check.py",
    "runtime/actions/inspection.py",
    "runtime/actions/obstacle_registry.py",
    "runtime/actions/plan_continuation.py",
    "runtime/actions/robot_action_schema.py",
    "runtime/actions/rollout_tracker.py",
    "runtime/actions/unified_actions.py",
    "runtime/decision_context.py",
    "runtime/harness.py",
    "runtime/observation_pipeline.py",
    "runtime/programs/__init__.py",
    "runtime/programs/ni_rua_client.py",
    "runtime/programs/program_result.py",
    "runtime/programs/program_review.py",
    "runtime/programs/runner.py",
    "runtime/tools/__init__.py",
    "runtime/tools/actions.py",
    "runtime/tools/builtins.py",
    "runtime/tools/catalog_guide.py",
    "runtime/tools/context.py",
    "runtime/tools/definitions.py",
    "runtime/tools/handlers.py",
    "runtime/tools/output_contracts.py",
    "runtime/tools/program_tools.py",
    "runtime/tools/registry.py",
    "runtime/tools/tool_search.py",
)

RETIRED_SOURCE_FILES = (
    "cortex/context/legacy_task_planning.py",
    "cortex/context/operation_candidates.py",
    "perception/external/spatial_field_shims.py",
)


def snapshot_actor_source(destination):
    """Copy actual files; the sandbox mounts this snapshot read-only at /harness."""
    destination = Path(destination)
    source = Path(__file__).resolve().parents[2]
    allowed = {"ni_rua/" + name for name in SOURCE_FILES}
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise ValueError("actor_source_snapshot_contains_unpublished_files")
    if destination.exists():
        if any(path.is_symlink() for path in destination.rglob("*")):
            raise ValueError("actor_source_snapshot_contains_unpublished_files")
        # Modules retired since an earlier version of this run published them may be removed; any other
        # file is unpublished. The decision is taken before anything is removed, so a refused snapshot
        # is left exactly as it was found.
        removable = {"ni_rua/" + name for name in RETIRED_SOURCE_FILES}
        present = {path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file()}
        if present - allowed - removable:
            raise ValueError("actor_source_snapshot_contains_unpublished_files")
        for name in sorted(present & removable):
            (destination / name).unlink()
    for relative in SOURCE_FILES:
        origin = source / relative
        if origin.is_symlink() or not origin.is_file():
            raise ValueError("actor_source_requires_regular_file: " + relative)
        target = destination / "ni_rua" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            raise ValueError("actor_source_snapshot_symlink")
        shutil.copyfile(origin, target)
    return destination
