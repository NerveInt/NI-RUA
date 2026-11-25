"""Robot control identities, explicit environment asset bindings and resolved embodiment profiles.

Leaf data contracts live in ``ni_rua/core/embodiment.py`` and robot-specific definitions
live under ``ni_rua/robot/embodiments/{arx_x5,so101}/``. Everything is re-exported here
for 100% caller compatibility, keeping ``ARX_X5_PROFILE`` lazy so importing this module
stays free of NumPy.
"""
from ...core.embodiment import (
    ArmProfile,
    ArrivalCorrectionLimits,
    CLOCKS,
    EMBODIMENT_RECORD_PREFIX,
    EmbodimentProfile,
    EndEffectorSpec,
    FrozenDict,
    GRASP_REFERENCES,
    GRIPPER_SPEED_PRESETS,
    GripperProfile,
    HoldSqueeze,
    HomeCheckProfile,
    JAW_GRASP_CENTRE_FIELDS,
    JawGraspGeometry,
    KinematicGroupSpec,
    LinkSegment,
    MAX_ARRIVAL_CORRECTION_OFFSET_RAD,
    MAX_ARRIVAL_CORRECTION_ROUNDS,
    MAX_BATCH_STAGES,
    MAX_STAGES_PER_ARM,
    ORIENTATION_FREE_TOLERANCE_RAD,
    POSE_INPUTS,
    PREVIEW_REVIEW_CLEAN,
    PREVIEW_REVIEW_DOWNGRADES,
    PROFILE_RECORD_SCHEMA,
    PROFILE_RECORD_SCHEMA_V2,
    PUBLISHED_STALL_GUARD_PRESETS,
    REACH_REQUEST_FIELDS,
    REACH_STAGE_FIELDS,
    RobotDescription,
    SPEED_PRESETS,
    SPLIT_ORIENTATION_FIELDS,
    SUPPORT_TAGS,
    ScaleRange,
    TOLERANCE_FIELDS,
    Tolerance,
    _LATER_RECORD_FIELDS,
    archived_embodiment_record,
    grip_guard_preset_margins,
    presets_within,
    preview_reviews,
    profile_from_record,
    profile_record,
    profile_to_record,
    register_default_embodiment,
    stall_guard_presets,
    tolerance_presets,
)
from ..embodiments.arx_x5.profile import (
    ARX_X5,
    ROBODOJO_REQUEST_DURATION_LIMIT_S,
    RoboDojoRobotBinding,
    X5_CAMERA_EXAMPLES,
    X5_DEFAULT_ACTION_TIMEOUT_S,
    X5_DEFAULT_WAIT_TIMEOUT_S,
    X5_FEEDBACK_UNSUPPORTED,
    X5_GRIPPER_SPEED_SCALE,
    X5_LIMITS_PROFILES,
    X5_MASK_REACH_M,
    X5_MAX_TIMEOUT_S,
    X5_OVERVIEW_CAMERA,
    X5_STALL_GUARD_SEMANTICS,
    X5_SUPPORTS,
    X5_VERTICAL_WINDOW_SEMANTICS,
    X5_WRIST_CAMERAS,
    _ROBODOJO_ROBOTS,
    _default_profile,
    arx_x5_profile,
    get_robodojo_robot,
    physical_facts,
)
from ..embodiments.libero_franka_panda.profile import (
    LIBERO_FRANKA_PANDA,
    _default_panda_profile,
    libero_franka_panda_profile,
)
from ..embodiments.so101.profile import SO101


def __getattr__(name):
    if name == "ARX_X5_PROFILE":
        return _default_profile()
    if name == "LIBERO_FRANKA_PANDA_PROFILE":
        return _default_panda_profile()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
