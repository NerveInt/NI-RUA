"""Leaf embodiment profile contract and deterministic archive codec (stdlib only).

Defines the frozen ``EmbodimentProfile`` schema shared across ``core``, ``robot``, ``perception``,
``cortex``, and ``runtime``, with zero dependencies on higher layers or third-party array libraries.
Concrete robot profiles (such as ARX-X5 and SO-101) register the default embodiment via
``register_default_embodiment``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


class FrozenDict(dict):
    """Read-only, hashable, insertion-ordered dict for the mapping fields of a profile.

    Compares equal to a plain ``dict`` with the same items; ``dict(value)`` gives a mutable copy; ``json.dumps``
    accepts it. Every mutating method raises ``TypeError``. Values must be hashable (str, float, None, tuples,
    frozen dataclasses or nested ``FrozenDict``).
    """

    __slots__ = ()

    def _immutable(self, *args, **kwargs):
        raise TypeError("embodiment_profile_is_immutable")

    __setitem__ = __delitem__ = __ior__ = clear = pop = popitem = setdefault = update = _immutable

    def __hash__(self):
        return hash(tuple(self.items()))

    def __reduce__(self):
        return FrozenDict, (dict(self),)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


@dataclass(frozen=True)
class RobotDescription:
    """Identity triple for an embodiment's model, pose reference frame, and TCP config."""

    robot_model_id: str
    frame_id: str
    tcp_config_id: str


@dataclass(frozen=True)
class ScaleRange:
    """Accepted range of a dimensionless speed multiplier (1 = the backend's nominal speed).

    ``min`` / ``max``: inclusive bounds a request may set (dimensionless).
    ``default``: the backend's nominal value when a request omits the field (dimensionless).
    """

    min: float
    max: float
    default: float


@dataclass(frozen=True)
class Tolerance:
    """Arrival tolerance of a move stage: TCP position error ``position_m`` (m, Euclidean), orientation error
    ``orientation_rad`` (rad, rotation angle), and optional upward +Z compliance window ``max_upward_dz_m`` (m,
    in [0, 0.15]; None when unset). An orientation at or above ``ORIENTATION_FREE_TOLERANCE_RAD`` leaves the
    orientation unconstrained.
    """

    position_m: float
    orientation_rad: float
    max_upward_dz_m: float | None = None


@dataclass(frozen=True)
class LinkSegment:
    """A segment from the origin of URDF link ``link`` along the unit vector ``axis_local`` (link frame) of length
    ``length_m`` (m).
    """

    link: str
    axis_local: tuple
    length_m: float


@dataclass(frozen=True)
class JawGraspGeometry:
    """CAD numbers of the grasp-centre pose input of a gripper with one fixed and one moving jaw (pose input tag
    ``jaw_grasp_centre``; SO-101: ``so101_gripper_geometry.jaw_grasp_facts``). All in the TCP link frame, metres:

    ``fixed_face_x_m``: coordinate of the fixed fingertip inner face along the jaw line (the TCP axis opposite to
        ``GripperProfile.jaw_opening_axis_local``; SO-101 0.0: the TCP lies on that face).
    ``min_grasp_depth_m`` / ``max_grasp_depth_m``: accepted range of ``grasp_depth_m`` (distance of the grasp centre
        behind the TCP plane, against ``approach_axis_local``): the fixed fingertip inner point (SO-101 -0.0053, i.e.
        5.3 mm beyond the TCP plane) and the pocket floor (SO-101 0.0611).
    ``tcp_plane_gap_m``: the jaw gap (width along the jaw line between the fingertips) at which the moving fingertip
        crosses the TCP plane; above it the moving fingertip is behind that plane (SO-101 0.0618).
    ``max_gap_m``: the widest jaw gap the deployment's jaw range opens to, the bound of ``object_width_m +
        fixed_jaw_clearance_m``.
    """

    fixed_face_x_m: float
    min_grasp_depth_m: float
    max_grasp_depth_m: float
    tcp_plane_gap_m: float
    max_gap_m: float


@dataclass(frozen=True)
class HoldSqueeze:
    """Bounded holding squeeze of a position-controlled jaw (SO-101: ``SO101Adapter._hold_squeeze``).

    ``default_rad`` (rad of the jaw joint, in [0, max_rad]): the squeeze of a closing jaw stage that sets no
        ``grip_squeeze_rad`` (0 holds the measured contact angle).
    ``max_rad`` (rad of the jaw joint, > 0): the largest ``grip_squeeze_rad`` a stage may request.
    """

    default_rad: float
    max_rad: float


@dataclass(frozen=True)
class GripperProfile:
    """The gripper specification of an embodiment (units: lengths m, angles rad, axes dimensionless unit vectors).

    ``empty_close_aperture_m`` (m or None): measured aperture after an empty full close (goal 0); None when unmeasured.
    ``full_open_aperture_m`` (m): measured/configured aperture at goal 1.
    ``full_open_source`` (str): provenance label explaining where ``full_open_aperture_m`` came from.
    ``preset_margins_m`` (FrozenDict): ``{preset: (min_m or None, max_m or None)}`` margins (m) above empty-close.
    ``aperture_reading`` (str): physical meaning of ``measured_aperture_m``.
    ``fingertip_beyond_reference_m`` (m): protrusion of fingertips beyond the grasp reference along approach axis.
    ``grasp_reference`` (``"tcp_offset"`` | ``"jaw_fingertip_midpoint"``): grasp center locator mode.
    ``grasp_reference_offset_m`` (m or None): axial offset from TCP to grasp center for ``"tcp_offset"``.
    ``approach_axis_local`` (tuple): unit vector outward along fingers in TCP link frame.
    ``jaw_opening_axis_local`` (tuple): unit vector along moving jaw opening direction in TCP link frame.
    ``mask_segment`` (LinkSegment or None): optional robot-pixel mask extension beyond URDF chain.
    ``contact_sections`` (FrozenDict or None): optional ``{region: contact_depth_m}`` longitudinal contact offsets (m).
    ``jaw_grasp`` (JawGraspGeometry or None): CAD bounds of ``jaw_grasp_centre`` pose input when supported.
    ``hold_squeeze`` (HoldSqueeze or None): bounded holding squeeze parameters (rad) when supported.
    """

    empty_close_aperture_m: float | None
    full_open_aperture_m: float
    full_open_source: str
    preset_margins_m: FrozenDict
    aperture_reading: str
    fingertip_beyond_reference_m: float
    grasp_reference: str
    grasp_reference_offset_m: float | None
    approach_axis_local: tuple
    jaw_opening_axis_local: tuple
    mask_segment: LinkSegment | None
    contact_sections: FrozenDict | None
    jaw_grasp: JawGraspGeometry | None = None
    hold_squeeze: HoldSqueeze | None = None


@dataclass(frozen=True)
class HomeCheckProfile:
    """Identity part of completion-guidance home check (``runtime/actions/home_check``)."""

    version: str
    frame_id: str
    tcp_config_id: str
    arms: tuple


#: Largest bounds any embodiment may publish for its arrival correction (``ArrivalCorrectionLimits``): count / rad.
MAX_ARRIVAL_CORRECTION_ROUNDS = 10
MAX_ARRIVAL_CORRECTION_OFFSET_RAD = 0.5


@dataclass(frozen=True)
class ArrivalCorrectionLimits:
    """Bounds of the goal shift after a move settles short of its target (``supports`` tag ``arrival_correction``).

    ``max_rounds`` (count, integer in [1, MAX_ARRIVAL_CORRECTION_ROUNDS]): most goal shifts one move may make.
    ``max_offset_rad`` (rad per joint, in (0, MAX_ARRIVAL_CORRECTION_OFFSET_RAD]): largest total shift of any joint goal.
    """

    max_rounds: int
    max_offset_rad: float


@dataclass(frozen=True)
class KinematicGroupSpec:
    """Per-arm kinematic chain identity and rest posture for heterogeneous v2 embodiments (angles in rad)."""

    joint_names: tuple
    tip_link: str
    tip_label: str
    tcp_config_id: str
    base_link: str = ""
    rest_posture_rad: tuple = ()


@dataclass(frozen=True)
class EndEffectorSpec:
    """Per-arm end-effector specification for heterogeneous v2 embodiments (jaw joints and gripper geometry)."""

    jaw_joint_names: tuple
    gripper: GripperProfile
    kind: str = "parallel_jaw"


@dataclass(frozen=True)
class ArmProfile:
    """Complete per-arm descriptor for heterogeneous multi-arm embodiments (v2 schema)."""

    arm: str
    kinematics: KinematicGroupSpec
    end_effector: EndEffectorSpec
    wrist_camera: str | None = None


# Vocabulary of the tag sets (validated at construction) and shared stage/request field sets.
POSE_INPUTS = frozenset({
    "tcp_quaternion",    # position_m + quaternion_xyzw of the TCP link in frame_id
    "grasp_angles",      # grasp_position_m + angles_deg in the RoboProbe top-down convention (X5 only)
    "jaw_grasp_centre",  # grasp_centre_m + object_width_m (+ fixed_jaw_clearance_m, grasp_depth_m) on fixed/moving jaw
})
JAW_GRASP_CENTRE_FIELDS = (
    "grasp_centre_m",
    "object_width_m",
    "fixed_jaw_clearance_m",
    "grasp_depth_m",
    "fixed_jaw_yaw_deg",
)
SPLIT_ORIENTATION_FIELDS = ("approach_tolerance_rad", "roll_tolerance_rad")
TOLERANCE_FIELDS = (
    "tolerance_preset",
    "position_tolerance_m",
    "orientation_tolerance_rad",
    "max_upward_dz_m",
    "max_downward_dz_m",
    *SPLIT_ORIENTATION_FIELDS,
)
REACH_TOLERANCE_FIELDS = (
    "tolerance_preset",
    "position_tolerance_m",
    "orientation_tolerance_rad",
    "max_upward_dz_m",
    *SPLIT_ORIENTATION_FIELDS,
)
REACH_REQUEST_FIELDS = frozenset({"mode", "observation_id", "stages", "timeout_s", *REACH_TOLERANCE_FIELDS})
REACH_STAGE_FIELDS = frozenset({"id", "kind", "arm", "position_m", "quaternion_xyzw", *REACH_TOLERANCE_FIELDS})

SUPPORT_TAGS = frozenset({
    "gripper_motion",        # jaw ramp during a move (gripper_motion start/end fractions, during_motion events)
    "joint_travel",          # joint_travel_preset / max_joint_travel_rad / max_backtracking_rad admission
    "max_upward_dz",         # max_upward_dz_m vertical arrival window
    "stall_guard",           # stall_guard_preset / max_tracking_divergence_m (meaning: stall_guard_semantics)
    "clearance",             # clearance_preset / clearance_height_m obstacle-crossing arcs
    "review_proxies",        # obstacles / held_object(s) / contact_entity_id / initial_contact / placement_contact
    "obstacle_object_ids",   # obstacle_object_ids resolved from the object table
    "settle",                # bounded settle checks (runtime/actions/stability)
    "execute_if_clean",      # robot_action mode execute_if_clean
    "dual_arm",              # simultaneous motion of two arms in one request
    "held_perception",       # held-object perception from an FK render of the arm meshes
    "reach",                 # robot_action mode=reach: per-stage IK queries without a plan or motion
    "approach_roll",         # move-stage approach_tolerance_rad / roll_tolerance_rad split orientation bounds
    "cross_view",            # object_table target_camera_ids also report the two-view (stereo) check value.cross_view
    "path_check",            # preview checks the joint-space path between targets for collision (path_check review)
    "collision_world",       # obstacles / held_object boxes enter the planner's collision world (IK targets + path check)
    "arrival_correction",    # arrival_correction_rounds / arrival_correction_max_offset_rad on move stages and the batch
    "robot_occlusion",       # object tables report value.robot_occlusion: table share under the arm's FK silhouette
    "contact_top",           # cross_view rows add contact_top_z: two-view fit of a prism standing on the table
    "cross_view_attach",     # the attached obs3d object table carries cross_view against the wrist camera
})
PREVIEW_REVIEW_CLEAN = {
    "motion_safety": "motion_safety is pass",
    "grasp_3d_review": "grasp_3d_review has no warnings/risk_flags",
    "geometry_review": "geometry_review has no risks",
    "path_check": "path_check.status is clear",
}
PREVIEW_REVIEW_DOWNGRADES = {
    "motion_safety": ("motion_safety_reject",),
    "grasp_3d_review": ("grasp_3d_review_warnings",),
    "geometry_review": ("geometry_review_risks",),
    "path_check": ("path_check_collisions", "path_check_unavailable"),
}


def preview_reviews(profile):
    """The preview reviews ``profile`` produces, which robot_action ``execute_if_clean`` requires clean, in published
    order: ``[(review, scope)]`` where ``scope`` names stage coverage or is None when batch-wide.
    """
    centre = "jaw_grasp_centre" in profile.pose_inputs
    return [
        (name, scope)
        for name, present, scope in (
            ("motion_safety", "joint_travel" in profile.supports, None),
            (
                "grasp_3d_review",
                bool({"grasp_angles", "jaw_grasp_centre"} & profile.pose_inputs),
                "grasp-centre stages only; TCP-pose stages are not reviewed" if centre else None,
            ),
            ("geometry_review", "review_proxies" in profile.supports, None),
            ("path_check", "path_check" in profile.supports, "the joint path of every move"),
        )
        if present
    ]


PUBLISHED_STALL_GUARD_PRESETS = ("off", "contact_stop", "strict")
CLOCKS = ("simulated_steps", "real_time")
GRASP_REFERENCES = ("tcp_offset", "jaw_fingertip_midpoint")


@dataclass(frozen=True)
class EmbodimentProfile:
    """Resolved, frozen embodiment description for one run.

    All v1 flat fields remain real dataclass fields so ``dataclasses.replace``, ``profile_from_record``,
    and golden digests remain byte-identical. Heterogeneous v2 multi-arm profiles optionally set
    ``arm_profiles`` (mapping ``arm -> ArmProfile``), emitting ``ni_rua.embodiment_profile.v2`` records.
    """

    robot: RobotDescription
    tcp_link: str
    tcp_label: str
    limits_profiles: tuple
    arms: tuple
    max_simultaneous_arms: int
    arm_joint_names: FrozenDict
    jaw_joint_names: FrozenDict
    stages_per_arm: int
    max_stages: int
    gripper: GripperProfile
    pose_inputs: frozenset
    supports: frozenset
    stall_guard_semantics: str
    speed_scale: ScaleRange
    gripper_speed_scale: ScaleRange
    speed_presets: FrozenDict
    gripper_speed_presets: FrozenDict
    tolerance_presets: FrozenDict
    default_tolerance_preset: str
    default_tolerance: Tolerance
    max_position_tolerance_m: float
    max_orientation_tolerance_rad: float
    stall_guard_presets_m: FrozenDict
    tracking_divergence_range_m: tuple
    clock: str
    control_hz: float
    step_domain: str
    step_label: str
    per_request_step_limit: int | None
    default_action_timeout_s: float
    default_wait_timeout_s: float
    max_timeout_s: float
    cameras: tuple
    overview_camera: str | None
    wrist_cameras: FrozenDict
    camera_examples: tuple
    support_surface_z_m: float | None
    handoff: bool
    home_check: HomeCheckProfile | None
    policy_step: bool
    vertical_window_semantics: str | None = None
    arrival_correction_limits: ArrivalCorrectionLimits | None = None
    arm_profiles: FrozenDict | None = None

    @property
    def robot_model_id(self):
        return self.robot.robot_model_id

    @property
    def frame_id(self):
        return self.robot.frame_id

    @property
    def tcp_config_id(self):
        return self.robot.tcp_config_id

    def __post_init__(self):
        def require(condition, name):
            if not condition:
                raise ValueError(f"invalid_embodiment_profile:{name}")

        def positive(value):
            return type(value) in (int, float) and math.isfinite(value) and value > 0

        if self.arm_profiles is not None and isinstance(self.arm_profiles, dict) and not isinstance(self.arm_profiles, FrozenDict):
            object.__setattr__(self, "arm_profiles", FrozenDict(self.arm_profiles))

        arms = self.arms
        require(isinstance(arms, tuple) and arms and len(set(arms)) == len(arms)
                and all(isinstance(arm, str) and arm for arm in arms), "arms")
        require(type(self.max_simultaneous_arms) is int and 1 <= self.max_simultaneous_arms <= len(arms),
                "max_simultaneous_arms")
        for name in ("arm_joint_names", "jaw_joint_names"):
            value = getattr(self, name)
            require(isinstance(value, FrozenDict) and tuple(value) == arms
                    and all(isinstance(names, tuple) and names for names in value.values()), name)
        require(type(self.stages_per_arm) is int and type(self.max_stages) is int
                and 1 <= self.stages_per_arm <= self.max_stages, "stages")
        require(isinstance(self.limits_profiles, tuple) and self.limits_profiles, "limits_profiles")
        require(isinstance(self.pose_inputs, frozenset) and self.pose_inputs and self.pose_inputs <= POSE_INPUTS,
                "pose_inputs")
        require(isinstance(self.supports, frozenset) and self.supports <= SUPPORT_TAGS, "supports")
        require(("dual_arm" in self.supports) == (self.max_simultaneous_arms > 1), "supports.dual_arm")
        require(("max_upward_dz" not in self.supports) or (isinstance(self.vertical_window_semantics, str)
                                                          and self.vertical_window_semantics),
                "vertical_window_semantics")
        correction = self.arrival_correction_limits
        require(correction is None or ("arrival_correction" in self.supports
                                       and isinstance(correction, ArrivalCorrectionLimits)
                                       and type(correction.max_rounds) is int
                                       and 1 <= correction.max_rounds <= MAX_ARRIVAL_CORRECTION_ROUNDS
                                       and positive(correction.max_offset_rad)
                                       and correction.max_offset_rad <= MAX_ARRIVAL_CORRECTION_OFFSET_RAD),
                "arrival_correction_limits")
        gripper = self.gripper
        require(positive(gripper.full_open_aperture_m), "gripper.full_open_aperture_m")
        require(gripper.empty_close_aperture_m is None or 0 <= gripper.empty_close_aperture_m
                < gripper.full_open_aperture_m, "gripper.empty_close_aperture_m")
        require(gripper.grasp_reference in GRASP_REFERENCES
                and (gripper.grasp_reference == "tcp_offset") == (gripper.grasp_reference_offset_m is not None),
                "gripper.grasp_reference")
        require(isinstance(gripper.preset_margins_m, FrozenDict) and "none" in gripper.preset_margins_m,
                "gripper.preset_margins_m")
        require(("grasp_angles" in self.pose_inputs) <= (gripper.grasp_reference == "tcp_offset"), "pose_inputs")
        jaw_grasp = gripper.jaw_grasp
        require(("jaw_grasp_centre" in self.pose_inputs) == (jaw_grasp is not None)
                and (jaw_grasp is None or gripper.grasp_reference == "jaw_fingertip_midpoint"), "pose_inputs")
        require(jaw_grasp is None or (isinstance(jaw_grasp, JawGraspGeometry)
                                      and jaw_grasp.min_grasp_depth_m <= 0 <= jaw_grasp.max_grasp_depth_m
                                      and 0 < jaw_grasp.tcp_plane_gap_m <= jaw_grasp.max_gap_m), "gripper.jaw_grasp")
        squeeze = gripper.hold_squeeze
        require(squeeze is None or (isinstance(squeeze, HoldSqueeze) and positive(squeeze.max_rad)
                                    and type(squeeze.default_rad) in (int, float)
                                    and 0 <= squeeze.default_rad <= squeeze.max_rad), "gripper.hold_squeeze")
        for name in ("speed_scale", "gripper_speed_scale"):
            scale = getattr(self, name)
            require(positive(scale.min) and scale.min <= scale.default <= scale.max, name)
        for name, scale in (("speed_presets", self.speed_scale), ("gripper_speed_presets", self.gripper_speed_scale)):
            require(all(scale.min <= value <= scale.max for value in getattr(self, name).values()), name)
        require(self.default_tolerance_preset in self.tolerance_presets, "default_tolerance_preset")
        for name, tolerance in (*self.tolerance_presets.items(), ("default_tolerance", self.default_tolerance)):
            require(isinstance(tolerance, Tolerance)
                    and positive(tolerance.position_m) and positive(tolerance.orientation_rad)
                    and tolerance.position_m <= self.max_position_tolerance_m
                    and tolerance.orientation_rad <= self.max_orientation_tolerance_rad
                    and (tolerance.max_upward_dz_m is None
                         or (type(tolerance.max_upward_dz_m) in (int, float)
                             and math.isfinite(tolerance.max_upward_dz_m)
                             and 0.0 <= tolerance.max_upward_dz_m <= 0.15)), f"tolerance.{name}")
        low, high = self.tracking_divergence_range_m
        require(positive(low) and low <= high, "tracking_divergence_range_m")
        require(self.clock in CLOCKS, "clock")
        require(positive(self.control_hz), "control_hz")
        require(self.per_request_step_limit is None
                or type(self.per_request_step_limit) is int and self.per_request_step_limit > 0,
                "per_request_step_limit")
        require(positive(self.default_action_timeout_s) and positive(self.default_wait_timeout_s)
                and max(self.default_action_timeout_s, self.default_wait_timeout_s) <= self.max_timeout_s, "timeouts")
        require(isinstance(self.cameras, tuple) and len(set(self.cameras)) == len(self.cameras), "cameras")
        require(self.overview_camera is None or self.overview_camera in self.cameras, "overview_camera")
        require(set(self.wrist_cameras) <= set(arms) and set(self.wrist_cameras.values()) <= set(self.cameras),
                "wrist_cameras")
        require(set(self.camera_examples) <= set(self.cameras), "camera_examples")
        require(self.home_check is None or set(self.home_check.arms) <= set(arms), "home_check")
        if self.arm_profiles is not None:
            require(isinstance(self.arm_profiles, FrozenDict) and tuple(self.arm_profiles) == arms, "arm_profiles")
            for arm, arm_profile in self.arm_profiles.items():
                require(isinstance(arm_profile, ArmProfile) and arm_profile.arm == arm, f"arm_profiles.{arm}")
                require(isinstance(arm_profile.kinematics, KinematicGroupSpec)
                        and arm_profile.kinematics.joint_names == self.arm_joint_names[arm],
                        f"arm_profiles.{arm}.kinematics")
                require(isinstance(arm_profile.end_effector, EndEffectorSpec)
                        and arm_profile.end_effector.jaw_joint_names == self.jaw_joint_names[arm],
                        f"arm_profiles.{arm}.end_effector")
                require(arm_profile.wrist_camera is None or arm_profile.wrist_camera == self.wrist_cameras.get(arm),
                        f"arm_profiles.{arm}.wrist_camera")


#: Orientation tolerances at or above this value (rad; pi rounded to 5 decimals) leave the TCP orientation free.
ORIENTATION_FREE_TOLERANCE_RAD = round(math.pi, 5)
#: Named arm speed multipliers (dimensionless, 1 = nominal); a profile publishes those inside its speed_scale.
SPEED_PRESETS = FrozenDict(slower=.5, slow=.8, careful=.9, normal=1., fast=1.25, faster=1.5)
#: Named jaw speed multipliers (dimensionless); a profile publishes those inside its gripper_speed_scale.
GRIPPER_SPEED_PRESETS = FrozenDict(slower=.5, slow=.8, normal=1.)
#: robot_action stage limits shared by both robots (per arm, per batch).
MAX_STAGES_PER_ARM = 16
MAX_BATCH_STAGES = 32


def tolerance_presets(thresholds, *, upward_dz_m=None):
    """``{preset: Tolerance}`` from resolved thresholds (``core.config.motion_thresholds`` keys, m / rad)."""
    t = thresholds
    dz_map = dict(upward_dz_m) if upward_dz_m else {}

    def pair(name, position, orientation):
        dz = float(dz_map[name]) if name in dz_map and dz_map[name] is not None else None
        return Tolerance(t[position], t[orientation] if orientation else ORIENTATION_FREE_TOLERANCE_RAD, dz)

    return FrozenDict(
        tight=pair("tight", "tight_position_tolerance_m", "tight_orientation_tolerance_rad"),
        fine=pair("fine", "tight_position_tolerance_m", "tight_orientation_tolerance_rad"),
        strict=pair("strict", "default_position_tolerance_m", "orientation_tolerance_rad"),
        normal=pair("normal", "default_position_tolerance_m", "orientation_tolerance_rad"),
        grasp=pair("grasp", "default_position_tolerance_m", "orientation_tolerance_rad"),
        relaxed=pair("relaxed", "relaxed_position_tolerance_m", "relaxed_orientation_tolerance_rad"),
        transit=pair("transit", "coarse_position_tolerance_m", "coarse_orientation_tolerance_rad"),
        compliant=pair("compliant", "compliant_position_tolerance_m", "relaxed_orientation_tolerance_rad"),
        seating=pair("seating", "relaxed_position_tolerance_m", "relaxed_orientation_tolerance_rad"),
        coarse=pair("coarse", "coarse_position_tolerance_m", "coarse_orientation_tolerance_rad"),
        unconstrained_orientation=pair("unconstrained_orientation", "relaxed_position_tolerance_m", None))


def grip_guard_preset_margins(thresholds):
    """``{preset: (min_margin_m, max_margin_m)}`` (m above the empty-close reading, None = unbounded)."""
    t = thresholds
    return FrozenDict(
        none=(None, None),
        non_empty=(t["grip_non_empty_min_aperture_m"], None),
        normal=(t["grip_non_empty_min_aperture_m"], None),
        thin_object=(t["grip_thin_min_aperture_m"], None),
        thin=(t["grip_thin_threshold_mm"] / 1000.0, None),
        rigid_object=(t["grip_rigid_min_aperture_m"], None),
        solid=(t["grip_solid_threshold_mm"] / 1000.0, None),
        wide_object=(t["grip_wide_min_aperture_m"], None),
        empty_only=(None, t["grip_empty_threshold_mm"] / 1000.0))


def stall_guard_presets(thresholds):
    """``{preset: max_tracking_divergence_m (m) or None (off)}`` from resolved thresholds."""
    t = thresholds
    return FrozenDict(
        off=None,
        contact_stop=t["stall_guard_contact_stop_divergence_m"],
        firm=t["stall_guard_contact_stop_divergence_m"],
        normal=t["stall_guard_contact_stop_divergence_m"],
        sensitive=t["stall_guard_strict_divergence_m"],
        compliant=t["stall_guard_compliant_divergence_m"],
        strict=t["stall_guard_strict_divergence_m"])


def presets_within(presets, scale):
    """The presets of ``presets`` whose multiplier lies inside ``scale`` (ScaleRange), in order."""
    return FrozenDict((name, value) for name, value in presets.items() if scale.min <= value <= scale.max)


# ---- default embodiment registration hook -----------------------------------------------------------------------
_default_embodiment_fn = None


def register_default_embodiment(fn):
    """Register the callable ``() -> EmbodimentProfile`` returning the default embodiment profile."""
    global _default_embodiment_fn
    _default_embodiment_fn = fn


def default_embodiment() -> EmbodimentProfile:
    """Return the registered default ``EmbodimentProfile``."""
    if _default_embodiment_fn is None:
        raise RuntimeError("default_embodiment_not_registered")
    return _default_embodiment_fn()


# ---- archive record ---------------------------------------------------------------------------------------------
PROFILE_RECORD_SCHEMA = "ni_rua.embodiment_profile.v1"
PROFILE_RECORD_SCHEMA_V2 = "ni_rua.embodiment_profile.v2"
EMBODIMENT_RECORD_PREFIX = "embodiment_profile-"
_LATER_RECORD_FIELDS = ("jaw_grasp", "arrival_correction_limits", "hold_squeeze", "max_upward_dz_m")
_LEGACY_V1_ARX_X5_UPWARD_DZ_M = {"compliant": 0.04, "seating": 0.04}
_V1_PROFILE_FIELDS = frozenset(name for name in EmbodimentProfile.__dataclass_fields__ if name != "arm_profiles")
_V2_PROFILE_FIELDS = frozenset(EmbodimentProfile.__dataclass_fields__)


def _plain(value, *, include_arm_profiles=True):
    if isinstance(value, EmbodimentProfile):
        fields = (
            name
            for name in value.__dataclass_fields__
            if (include_arm_profiles or name != "arm_profiles")
            and not (name in _LATER_RECORD_FIELDS and getattr(value, name) is None)
        )
        return {name: _plain(getattr(value, name), include_arm_profiles=include_arm_profiles) for name in fields}
    if isinstance(
        value,
        (
            GripperProfile,
            ScaleRange,
            Tolerance,
            LinkSegment,
            HomeCheckProfile,
            RobotDescription,
            JawGraspGeometry,
            ArrivalCorrectionLimits,
            HoldSqueeze,
            KinematicGroupSpec,
            EndEffectorSpec,
            ArmProfile,
        ),
    ):
        return {
            name: _plain(getattr(value, name), include_arm_profiles=include_arm_profiles)
            for name in value.__dataclass_fields__
            if not (name in _LATER_RECORD_FIELDS and getattr(value, name) is None)
        }
    if isinstance(value, dict):
        return {key: _plain(item, include_arm_profiles=include_arm_profiles) for key, item in value.items()}
    if isinstance(value, frozenset):
        return sorted(value)
    if isinstance(value, (tuple, list)):
        return [_plain(item, include_arm_profiles=include_arm_profiles) for item in value]
    return value


def profile_record(profile):
    """JSON-ready audit record of ``profile`` (operator/archive fact; never model context).

    Emits byte-identical ``ni_rua.embodiment_profile.v1`` records when ``profile.arm_profiles is None``,
    and ``ni_rua.embodiment_profile.v2`` when heterogeneous ``arm_profiles`` is provided.
    """
    if not isinstance(profile, EmbodimentProfile):
        raise TypeError("profile_record_requires_embodiment_profile")
    is_v2 = profile.arm_profiles is not None
    schema = PROFILE_RECORD_SCHEMA_V2 if is_v2 else PROFILE_RECORD_SCHEMA
    return {"schema": schema, "profile": _plain(profile, include_arm_profiles=is_v2)}


profile_to_record = profile_record


def archived_embodiment_record(store, record_id=None):
    """``(record_id, record)`` of an archived embodiment profile in ``store``."""
    if record_id:
        try:
            return record_id, store.read("configuration", record_id)
        except KeyError:
            return record_id, None
    records = [(name, value) for name, value in store.record_items("configuration").items()
               if name.startswith(EMBODIMENT_RECORD_PREFIX)]
    return records[-1] if records else (None, None)


def _tuples(value):
    """JSON arrays back to (nested) tuples."""
    return tuple(_tuples(item) for item in value) if isinstance(value, list) else value


def _frozen(value, convert=_tuples):
    return FrozenDict((key, convert(item)) for key, item in value.items())


def _gripper_from_dict(raw):
    gripper = dict(raw)
    segment = gripper["mask_segment"]
    gripper.update(
        preset_margins_m=_frozen(gripper["preset_margins_m"]),
        approach_axis_local=_tuples(gripper["approach_axis_local"]),
        jaw_opening_axis_local=_tuples(gripper["jaw_opening_axis_local"]),
        mask_segment=None if segment is None else LinkSegment(
            segment["link"], _tuples(segment["axis_local"]), segment["length_m"]),
        contact_sections=None if gripper["contact_sections"] is None else _frozen(gripper["contact_sections"]),
        jaw_grasp=None if gripper.get("jaw_grasp") is None else JawGraspGeometry(**gripper["jaw_grasp"]),
        hold_squeeze=None if gripper.get("hold_squeeze") is None else HoldSqueeze(**gripper["hold_squeeze"]),
    )
    return GripperProfile(**gripper)


def _arm_profile_from_dict(raw):
    kin = dict(raw["kinematics"])
    kin["joint_names"] = _tuples(kin["joint_names"])
    kin["rest_posture_rad"] = _tuples(kin.get("rest_posture_rad", ()))
    ee = dict(raw["end_effector"])
    ee["jaw_joint_names"] = _tuples(ee["jaw_joint_names"])
    ee["gripper"] = _gripper_from_dict(ee["gripper"])
    return ArmProfile(
        arm=raw["arm"],
        kinematics=KinematicGroupSpec(**kin),
        end_effector=EndEffectorSpec(**ee),
        wrist_camera=raw.get("wrist_camera"),
    )


def profile_from_record(record):
    """Reconstruct ``EmbodimentProfile`` from an archived ``profile_record`` (inverse of ``profile_record``)."""
    if not isinstance(record, dict) or not isinstance(record.get("profile"), dict):
        raise ValueError("unsupported_embodiment_profile_record")
    schema = record.get("schema")
    if schema == PROFILE_RECORD_SCHEMA:
        expected_fields = _V1_PROFILE_FIELDS
    elif schema == PROFILE_RECORD_SCHEMA_V2:
        expected_fields = _V2_PROFILE_FIELDS
    else:
        raise ValueError("unsupported_embodiment_profile_record")
    fields = dict(record["profile"])
    try:
        if not expected_fields - {"arrival_correction_limits"} <= set(fields) <= expected_fields:
            raise KeyError("fields")
        home = fields["home_check"]
        correction = fields.get("arrival_correction_limits")
        robot = RobotDescription(**fields["robot"])
        raw_presets = fields["tolerance_presets"]
        if not isinstance(raw_presets, dict):
            raise TypeError("tolerance_presets")
        if (
            schema == PROFILE_RECORD_SCHEMA
            and robot.robot_model_id == "arx-x5"
            and not any(isinstance(item, dict) and "max_upward_dz_m" in item for item in raw_presets.values())
        ):
            raw_presets = {
                name: (
                    dict(item, max_upward_dz_m=_LEGACY_V1_ARX_X5_UPWARD_DZ_M[name])
                    if name in _LEGACY_V1_ARX_X5_UPWARD_DZ_M and isinstance(item, dict)
                    else item
                )
                for name, item in raw_presets.items()
            }
        updates = dict(
            arrival_correction_limits=None if correction is None else ArrivalCorrectionLimits(**correction),
            robot=robot,
            gripper=_gripper_from_dict(fields["gripper"]),
            pose_inputs=frozenset(fields["pose_inputs"]),
            supports=frozenset(fields["supports"]),
            speed_scale=ScaleRange(**fields["speed_scale"]),
            gripper_speed_scale=ScaleRange(**fields["gripper_speed_scale"]),
            tolerance_presets=_frozen(raw_presets, lambda item: Tolerance(**item)),
            default_tolerance=Tolerance(**fields["default_tolerance"]),
            home_check=None if home is None else HomeCheckProfile(**dict(home, arms=_tuples(home["arms"]))),
            **{name: _frozen(fields[name]) for name in ("arm_joint_names", "jaw_joint_names", "speed_presets",
                                                        "gripper_speed_presets", "stall_guard_presets_m",
                                                        "wrist_cameras")},
            **{name: _tuples(fields[name]) for name in ("limits_profiles", "arms", "tracking_divergence_range_m",
                                                        "cameras", "camera_examples")},
        )
        if schema == PROFILE_RECORD_SCHEMA_V2:
            raw_arms = fields["arm_profiles"]
            if not isinstance(raw_arms, dict):
                raise TypeError("arm_profiles")
            updates["arm_profiles"] = FrozenDict(
                (arm, _arm_profile_from_dict(entry)) for arm, entry in raw_arms.items()
            )
        fields.update(updates)
        return EmbodimentProfile(**fields)
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("invalid_embodiment_profile:"):
            raise
        raise ValueError("unsupported_embodiment_profile_record") from error
