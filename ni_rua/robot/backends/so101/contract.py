"""Pure SO-101 request compilation and validation (no worker or LeRobot; numpy only through the CAD geometry).

Every robot fact and bound used here is read from the run's ``EmbodimentProfile`` built from the deployment file
(``embodiment(deployment)`` = ``deployment.embodiment_profile``, design 13 section 5.1): arm ids, frame, TCP, limits
profile, speed ranges, tolerance and stall-guard presets and bounds, the gripper's empty-close zero point, full opening
and preset margins, default timeouts and the stage features this backend honours. This module defines none of them.
The profile builder never calls this module (no recursion).

The public action language is unchanged: absolute ``position_m`` + ``quaternion_xyzw`` of the TCP
``gripper_frame_link`` in ``so101_base``, normalized gripper goals, and bounded waits. A move may give the grasp
centre instead (``grasp_centre_m`` + ``object_width_m``, pose input tag ``jaw_grasp_centre``): ``compile_action``
converts it to that TCP target with the gripper CAD (``grasp_centre_pose``), keeps the request on the action as
``input_grasp_centre``, ``validate_leaf`` re-derives the target from it, and ``annotate_preview`` / ``annotate_reach``
/ ``annotate_receipt`` report the conversion in the results. A during-motion jaw ramp
(``gripper_motion``) and the vertical arrival window (``max_upward_dz_m``) are honoured when the profile's ``supports``
carries ``gripper_motion`` / ``max_upward_dz`` (design 13 R5, R6); the worker implements them (``SO101Adapter._move``).
Fields this backend does not honour (X5 grasp-angle poses and contact regions, dual-arm fields, joint-travel admission,
clearance arcs, unknown path fields, and the two above on a profile without their tag) are refused by
name (``unsupported_capability: <field>``), never accepted and ignored. With the cuRobo IK service (profile tag
``collision_world``) the compiled ``review_proxies`` of a batch are accepted with ``obstacles`` and this arm's
``held_objects`` box only: they become the collision world of the preview (``ik_plan.collision_world``); contact
declarations stay refused. ``arrival_correction_rounds`` / ``arrival_correction_max_offset_rad`` on a move path bound
the sag correction (``SO101Adapter._arrive``) up to the deployment's own limits. ``grip_squeeze_rad`` on a jaw stage
(a gripper stage, or a move with a during-motion jaw ramp) sets the holding squeeze of that jaw command
(``SO101Adapter._hold_squeeze``) within ``[0, gripper.hold_squeeze.max_rad]``; it is refused on a deployment without
``gripper.hold_squeeze_rad`` (no squeeze is applied there, and the field is not offered).
"""
from __future__ import annotations

import functools
import json
import math
import threading

from ....core.action_validation import validate_gripper_motion
from ....core.contracts import new_id
from ...model.embodiments import JAW_GRASP_CENTRE_FIELDS, PUBLISHED_STALL_GUARD_PRESETS, SO101, physical_facts
from ...motion.grip_guard import GRIP_SQUEEZE_FIELD, validate_grip_squeeze
from ...motion.motion_speed import SPEED_INPUT_FIELDS
from ...motion.motion_tolerance import MAX_UPWARD_DZ_M, arrival_correction_bounds
from .deployment import GRIP_GUARD_PRESET_NAMES, LIMITS_PROFILE as DEPLOYMENT_LIMITS_PROFILE
from .deployment import active_motion_compensation, check_workspace, embodiment_profile, hold_squeeze

FRAME_ID = SO101.frame_id
TCP_CONFIG_ID = SO101.tcp_config_id
#: ``limits_profile`` id of this backend (``deployment.LIMITS_PROFILE``; the profile's ``limits_profiles[0]``).
LIMITS_PROFILE = DEPLOYMENT_LIMITS_PROFILE
#: Arm ids of the packaged SO-101 facts (``physical.json`` ``identity.public_arm_slots``; the profile's ``arms``).
ARMS = tuple(physical_facts("so101")["identity"]["public_arm_slots"])
#: URDF link whose pose ``position_m`` / ``quaternion_xyzw`` target (the profile's ``tcp_link``).
TCP_LINK = physical_facts("so101")["kinematics"]["tcp_link"]
UNSUPPORTED_FIELDS = ("angles_deg", "grasp_position_m", "via_grasp_poses", "contact_region", "contact_depth_m",
                      "dual_arm_progression", "arm_options", "waypoint_speed_presets", "waypoint_speed_scales")
#: What to use instead of a refused ARX-X5 grasp field (appended to ``unsupported_capability: <field>``).
_GRASP_FIELD_ALTERNATIVES = {
    "angles_deg": "use quaternion_xyzw, or fixed_jaw_yaw_deg with grasp_centre_m",
    "grasp_position_m": "use grasp_centre_m + object_width_m",
    "via_grasp_poses": "use successive move stages",
    "contact_region": "use grasp_depth_m with grasp_centre_m",
    "contact_depth_m": "use grasp_depth_m with grasp_centre_m",
}
#: Move-stage fields of the grasp-centre pose input (``embodiments.JAW_GRASP_CENTRE_FIELDS``; ``grasp_centre_pose``).
GRASP_CENTRE_FIELDS = JAW_GRASP_CENTRE_FIELDS
#: Key of a compiled move leaf that keeps its grasp-centre request (``grasp_centre_request`` output).
GRASP_CENTRE_INPUT = "input_grasp_centre"
#: Largest difference (m; quaternion components likewise) between a leaf's waypoint and the pose re-derived from its
#: ``input_grasp_centre``: float round-off of one JSON round trip, not a tolerance on the grasp.
GRASP_CENTRE_CONSISTENCY = 1e-9
# Batch fields the SO-101 cannot honour (no joint-travel admission, no X5 grasp-angle symmetry/tilt or downward window).
# ``max_upward_dz_m`` joins them on a profile without ``max_upward_dz`` and ``review_proxies`` without ``collision_world``.
UNSUPPORTED_BATCH_FIELDS = (
    "joint_travel_preset",
    "max_joint_travel_rad",
    "max_backtracking_rad",
    "allow_jaw_symmetry",
    "allow_approach_tilt",
    "max_downward_dz_m",
)
#: Keys of a batch's ``review_proxies`` this backend reads (``collision_world`` tag): obstacle boxes and the held box.
REVIEW_PROXY_FIELDS = ("obstacles", "held_objects")
#: Path fields of a compiled move or hold that the worker reads (``kind``, ``orientation``, ``blend_radius_m``,
#: ``waypoints``, the arrival tolerances, the vertical arrival window ``max_upward_dz_m`` when the profile supports it,
#: and the stall guard) or reports (the two preset names). Any other path field is refused: the worker would ignore it.
PATH_FIELDS = ("kind", "orientation", "blend_radius_m", "waypoints", "position_tolerance_m", "orientation_tolerance_rad",
               "max_upward_dz_m", "approach_tolerance_rad", "roll_tolerance_rad", "tolerance_preset",
               "stall_guard_preset", "max_tracking_divergence_m", "arrival_correction_rounds",
               "arrival_correction_max_offset_rad")
#: Arrival-correction path fields (count / rad), honoured with the profile's ``arrival_correction`` tag.
ARRIVAL_CORRECTION_PATH_FIELDS = ("arrival_correction_rounds", "arrival_correction_max_offset_rad")
#: Split orientation path fields (rad), honoured with the profile's ``approach_roll`` tag (cuRobo IK service).
SPLIT_ORIENTATION_PATH_FIELDS = ("approach_tolerance_rad", "roll_tolerance_rad")
#: Fields of one during-motion jaw event (``gripper_motion`` compiled onto a move; fractions of the move's commands).
DURING_MOTION_EVENT_FIELDS = frozenset({"arm", "phase", "goal_normalized", "start_fraction", "end_fraction"})
# Why a path field the Harness can add is refused here (the text of ``unsupported_capability``).
_PATH_FIELD_REASONS = {
    "max_upward_dz_m": "no vertical arrival window in this SO-101 profile: arrival is checked against position_tolerance_m",
    "max_downward_dz_m": "no downward arrival window on SO-101: arrival is checked against position_tolerance_m",
    "approach_tolerance_rad": "split orientation bounds need the cuRobo IK service (deployment ik_service)",
    "roll_tolerance_rad": "split orientation bounds need the cuRobo IK service (deployment ik_service)",
    "clearance_height_m": "no clearance arc is planned on SO-101: a move is a joint-space interpolation to the IK target",
    "target_exemption_radius_m": "no obstacle review on SO-101",
    "arrival_correction_rounds": "this deployment applies no arrival correction (motion.arrival_correction_rounds "
                                 "or arrival_correction_max_offset_rad is 0), so there is nothing to bound",
    "arrival_correction_max_offset_rad": "this deployment applies no arrival correction (motion.arrival_correction_rounds "
                                         "or arrival_correction_max_offset_rad is 0), so there is nothing to bound",
}
GRIP_GUARD_FIELDS = ("grip_guard_preset", "min_aperture_m", "max_aperture_m")
#: Presets offered by the robot_action schema (``deployment.GRIP_GUARD_PRESET_NAMES``; the profile's preset margins
#: besides ``none``); each resolves relative to the measured empty-close gap.
SCHEMA_GRIP_GUARD_PRESETS = GRIP_GUARD_PRESET_NAMES
#: Stall-guard preset names the robot_action schema publishes (``embodiments.PUBLISHED_STALL_GUARD_PRESETS``). The
#: profile's other names (X5 legacy aliases firm / normal / sensitive / compliant) are refused at stage and batch level
#: alike, so the text, the schema enum and the contract offer the same names (``offered_stall_guard_presets``).
SCHEMA_STALL_GUARD_PRESETS = PUBLISHED_STALL_GUARD_PRESETS
#: Largest ``native_steps`` of one bounded wait (command periods of 1 / motion.command_rate_hz s each).
MAX_WAIT_STEPS = 50
# Measurements on the development SO-101 (run archives 2026-11-10..13), quoted to the model as
# measured facts about this gripper. No decision uses them; re-measure on another unit.
MEASURED_EMPTY_CLOSE_RANGE_M = (0.0042, 0.0051)  # empty goal-0 closes across days and recalibrations
VERIFIED_OBJECT_READING_M = (0.029, 0.0289)  # (object width, measured_aperture_m): the only verified point

_PROFILES = {}
_PROFILES_LOCK = threading.Lock()  # the worker reads profiles from its main and operator threads


def embodiment(deployment):
    """The ``EmbodimentProfile`` of one validated deployment (``deployment.embodiment_profile``), cached by content.

    The cache key is the deployment's canonical JSON, so an edited deployment (calibration apply, reload) gets a new
    profile; building one takes about 10 ms (CAD geometry), reading the cache about 0.05 ms."""
    key = json.dumps(deployment, sort_keys=True, default=str)
    with _PROFILES_LOCK:
        profile = _PROFILES.get(key)
    if profile is None:
        profile = embodiment_profile(deployment)
        with _PROFILES_LOCK:
            while len(_PROFILES) >= 8:
                _PROFILES.pop(next(iter(_PROFILES)))
            _PROFILES[key] = profile
    return profile


def _arm(args, arms=ARMS):
    arm = args.get("arm")
    if arm not in arms:
        raise ValueError(f"unsupported_arm:{arm}")
    return arm


def compile_action(kind, args, identity, capabilities):
    """Build one EEAction from explicit arguments; never reads live state.

    Defaults come from the worker capabilities, which carry the profile's values: ``timeout_s`` from
    ``bounded_wait.default_timeout_s`` for a wait (profile ``default_wait_timeout_s``) and ``default_timeout_s``
    otherwise (profile ``default_action_timeout_s``); move tolerances from ``motion_tolerance_defaults`` (profile
    ``default_tolerance``, the deployment's ``motion.default_*_tolerance_*``)."""
    if kind not in ("move", "grasp", "gripper", "wait"):
        raise ValueError("unsupported_so101_action_kind")
    if kind == "grasp" or "angles_deg" in args or "grasp_position_m" in args:
        raise ValueError(f"unsupported_capability: angles_deg (SO-101 accepts position_m + quaternion_xyzw of "
                         f"{TCP_LINK} in {FRAME_ID}, or grasp_centre_m + object_width_m with quaternion_xyzw or "
                         "fixed_jaw_yaw_deg)")
    for field in UNSUPPORTED_FIELDS:
        if field in args:
            raise ValueError(_unsupported_field(field))
    grasp = None
    if any(field in args for field in GRASP_CENTRE_FIELDS):
        if kind != "move":
            raise ValueError("invalid_grasp_centre: grasp_centre_m and its fields belong to a move stage")
        grasp = grasp_centre_request(args)
    if "gripper_motion" in args:
        # Offered only when the worker publishes the during_motion jaw phase (profile tag gripper_motion).
        if "during_motion" not in (capabilities.get("gripper_phases") or ()):
            raise ValueError("unsupported_capability: gripper_motion")
        if kind != "move" or "gripper_goal_normalized" not in args:
            raise ValueError("invalid_gripper_motion: requires a move and gripper_goal_normalized")
    arms = tuple(capabilities.get("arms") or ARMS)
    defaults = capabilities.get("motion_tolerance_defaults", {})
    default_timeout = capabilities.get("default_timeout_s")
    if kind == "wait":
        default_timeout = (capabilities.get("bounded_wait") or {}).get("default_timeout_s", default_timeout)
    timeout = args.get("timeout_s", default_timeout)
    path = dict(kind="hold", orientation="backend_free", blend_radius_m=0.0, waypoints=[])
    if kind == "move":
        arm = _arm(args, arms)
        if grasp is not None:
            # The TCP target of the grasp centre (CAD conversion); the request stays on the action for the receipts.
            target = grasp_centre_pose(grasp)
            pose = dict(position_m=target["position_m"], quaternion_xyzw=target["quaternion_xyzw"])
        else:
            pose = dict(position_m=args["position_m"], quaternion_xyzw=args["quaternion_xyzw"])
        path.update(kind="backend_free", waypoints=[{arm: pose}])
        if "tolerance_preset" in args:
            path["tolerance_preset"] = args["tolerance_preset"]
            if "position_tolerance_m" in args:
                path["position_tolerance_m"] = args["position_tolerance_m"]
            if "orientation_tolerance_rad" in args:
                path["orientation_tolerance_rad"] = args["orientation_tolerance_rad"]
        else:
            path["position_tolerance_m"] = args.get("position_tolerance_m", defaults.get("position_tolerance_m"))
            path["orientation_tolerance_rad"] = args.get(
                "orientation_tolerance_rad", defaults.get("orientation_tolerance_rad")
            )
        for extra_path_key in (
            "max_upward_dz_m",
            "max_downward_dz_m",
            "approach_tolerance_rad",
            "roll_tolerance_rad",
            "stall_guard_preset",
            "max_tracking_divergence_m",
            "arrival_correction_rounds",
            "arrival_correction_max_offset_rad",
        ):
            if extra_path_key in args:
                path[extra_path_key] = args[extra_path_key]
    action = dict(schema_version="0.1", **identity, world_kind="observed",
                  action_id=args.get("action_id", new_id("action")), based_on_observation_id=args["observation_id"],
                  frame_id=capabilities["frame_id"], tcp_config_id=capabilities["tcp_config_id"], mode="absolute",
                  path=path, gripper_events=[], limits_profile=capabilities.get("limits_profile", LIMITS_PROFILE),
                  timeout_s=timeout)
    if grasp is not None:
        action[GRASP_CENTRE_INPUT] = grasp
    for key in SPEED_INPUT_FIELDS:
        if key in args:
            action[key] = args[key]
    if kind == "gripper":
        action["gripper_events"] = [dict(arm=_arm(args, arms), phase="after_motion", goal_normalized=args["goal_normalized"])]
    elif kind == "wait":
        action["hold_native_steps"] = args["native_steps"]
    if "gripper_goal_normalized" in args:
        action["gripper_events"] = [dict(arm=_arm(args, arms), phase="after_motion",
                                         goal_normalized=args["gripper_goal_normalized"])]
    if "gripper_motion" in args:
        # Jaw ramp over start_fraction..end_fraction of this move's commands (SO101Adapter._segment).
        timing = validate_gripper_motion(args["gripper_motion"])
        action["gripper_events"][0].update(phase="during_motion", start_fraction=timing["start_fraction"],
                                           end_fraction=timing["end_fraction"])
    return action


def _unsupported_field(field):
    """``unsupported_capability: <field>``, naming the SO-101 input that replaces a refused ARX-X5 grasp field."""
    alternative = _GRASP_FIELD_ALTERNATIVES.get(field)
    return f"unsupported_capability: {field}" + (f" (on SO-101 {alternative})" if alternative else "")


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def grasp_centre_request(args):
    """The grasp-centre request of one move stage, checked for form only (bounds: ``validate_leaf``, which has the
    deployment). ``args`` fields (``GRASP_CENTRE_FIELDS`` plus the orientation):

    * ``grasp_centre_m`` ([x, y, z], m in so101_base; required): where the centre of the object sits between the jaws.
    * ``object_width_m`` (m, >= 0; required, no default): the object's extent along the jaw line (TCP X).
    * ``fixed_jaw_clearance_m`` (m, >= 0; default 0): gap kept between the fixed fingertip face and the object.
    * ``grasp_depth_m`` (m; default 0): distance of the centre behind the TCP plane, positive toward the palm.
    * exactly one of ``quaternion_xyzw`` (unit quaternion of the TCP in so101_base) and ``fixed_jaw_yaw_deg`` (deg in
      [-180, 180]: straight-down approach with TCP +X, the direction from the moving to the fixed jaw, at this yaw
      about base +z from +x). ``position_m`` is refused beside them (the centre replaces it).

    Returns ``{"grasp_centre_m", "object_width_m", "fixed_jaw_clearance_m", "grasp_depth_m", "quaternion_xyzw"}``
    plus ``"fixed_jaw_yaw_deg"`` when that was the orientation input (the quaternion is then derived from it).
    Raises ``ValueError("invalid_grasp_centre: ...")``."""
    def fail(text):
        raise ValueError("invalid_grasp_centre: " + text)
    if "grasp_centre_m" not in args:
        fail(", ".join(field for field in GRASP_CENTRE_FIELDS if field in args) + " require grasp_centre_m")
    if "position_m" in args:
        fail("give grasp_centre_m or position_m, not both (grasp_centre_m is converted to the TCP position)")
    centre = args["grasp_centre_m"]
    if not isinstance(centre, list) or len(centre) != 3 or not all(_finite(value) for value in centre):
        fail("grasp_centre_m must be three finite numbers [x, y, z] in metres")
    if "object_width_m" not in args:
        fail("object_width_m (m, the object's extent along the jaw line) is required with grasp_centre_m")
    request = {"grasp_centre_m": [float(value) for value in centre]}
    for field, default in (("object_width_m", None), ("fixed_jaw_clearance_m", 0.0), ("grasp_depth_m", 0.0)):
        value = args.get(field, default)
        if not _finite(value) or (field != "grasp_depth_m" and value < 0):
            fail(f"{field} must be a finite number in metres" + ("" if field == "grasp_depth_m" else ", >= 0"))
        request[field] = float(value)
    if ("quaternion_xyzw" in args) == ("fixed_jaw_yaw_deg" in args):
        fail("give exactly one orientation: quaternion_xyzw or fixed_jaw_yaw_deg")
    if "fixed_jaw_yaw_deg" in args:
        yaw = args["fixed_jaw_yaw_deg"]
        if not _finite(yaw) or not -180.0 <= yaw <= 180.0:
            fail("fixed_jaw_yaw_deg must be in [-180, 180] degrees")
        request.update(quaternion_xyzw=straight_down_quaternion(yaw), fixed_jaw_yaw_deg=float(yaw))
    else:
        quaternion = args["quaternion_xyzw"]
        if (not isinstance(quaternion, list) or len(quaternion) != 4 or not all(_finite(value) for value in quaternion)
                or abs(math.sqrt(sum(value * value for value in quaternion)) - 1) > 1e-3):
            fail("quaternion_xyzw must be a unit quaternion [x, y, z, w]")
        request["quaternion_xyzw"] = [float(value) for value in quaternion]
    return request


def straight_down_quaternion(fixed_jaw_yaw_deg):
    """``quaternion_xyzw`` of the TCP for a straight-down approach: TCP +Z along base -z and TCP +X (from the moving
    jaw toward the fixed jaw) at ``fixed_jaw_yaw_deg`` (deg) about base +z from +x; TCP +Y is then
    (sin yaw, -cos yaw, 0). Example: 0 -> [1, 0, 0, 0]; 90 -> [0.7071, 0.7071, 0, 0]."""
    half = math.radians(float(fixed_jaw_yaw_deg)) / 2
    return [math.cos(half), math.sin(half), 0.0, 0.0]


def _rotate(quaternion_xyzw, vector):
    """``vector`` (TCP axes) rotated into the base frame by the unit-normalized ``quaternion_xyzw``."""
    x, y, z, w = (float(value) for value in quaternion_xyzw)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    rows = ((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)))
    return [sum(row[index] * float(vector[index]) for index in range(3)) for row in rows]


def grasp_centre_pose(request):
    """TCP target of a grasp-centre request (``grasp_centre_request`` output); pure CAD, no live state.

    ``position_m = grasp_centre_m + R(quaternion_xyzw) @ tcp_offset_m`` with ``tcp_offset_m =
    so101_gripper_geometry.grasp_centre_tcp_offset_m(object_width_m, fixed_jaw_clearance_m, grasp_depth_m)`` =
    ``[object_width_m / 2 + fixed_jaw_clearance_m - fixed_face_x, 0, grasp_depth_m]`` in TCP axes (m): the fixed
    fingertip face ends ``object_width_m / 2 + fixed_jaw_clearance_m`` from the centre on its TCP +X side, the TCP
    plane ``grasp_depth_m`` beyond the centre along the approach. The jaw angle does not enter (only the -X jaw
    moves). Returns ``{"position_m", "quaternion_xyzw", "tcp_offset_m"}``; the quaternion is the request's.
    Example: centre [0.24, -0.06, 0.015], width 0.03, yaw 0 (straight down) -> position [0.255, -0.06, 0.015]."""
    from ...model.so101_gripper_geometry import grasp_centre_tcp_offset_m
    offset = grasp_centre_tcp_offset_m(request["object_width_m"], request["fixed_jaw_clearance_m"],
                                       request["grasp_depth_m"])
    shift = _rotate(request["quaternion_xyzw"], offset)
    return {"position_m": [centre + delta for centre, delta in zip(request["grasp_centre_m"], shift)],
            "quaternion_xyzw": list(request["quaternion_xyzw"]), "tcp_offset_m": offset}


def _validate_grasp_centre(action, pose, profile):
    """A leaf's ``input_grasp_centre``: its form, the profile's CAD bounds (``gripper.jaw_grasp``) and that the
    waypoint ``pose`` is exactly the conversion of the request (nothing may carry a request that was not executed)."""
    request, bounds = action[GRASP_CENTRE_INPUT], profile.gripper.jaw_grasp
    if "jaw_grasp_centre" not in profile.pose_inputs or bounds is None:
        raise ValueError("unsupported_capability: grasp_centre_m")
    allowed = {*GRASP_CENTRE_FIELDS, "quaternion_xyzw"}
    if not isinstance(request, dict) or set(request) - allowed:
        raise ValueError("invalid_grasp_centre: input_grasp_centre carries only the grasp-centre fields")
    checked = grasp_centre_request({key: value for key, value in request.items()
                                    if not (key == "quaternion_xyzw" and "fixed_jaw_yaw_deg" in request)})
    if checked != request:
        raise ValueError("invalid_grasp_centre: input_grasp_centre is not a compiled grasp-centre request")
    width, clearance, depth = (request[key] for key in ("object_width_m", "fixed_jaw_clearance_m", "grasp_depth_m"))
    if width + clearance > bounds.max_gap_m:
        raise ValueError(f"invalid_grasp_centre: object_width_m + fixed_jaw_clearance_m = {width + clearance:.4f} m "
                         f"exceeds the widest jaw gap {bounds.max_gap_m:g} m; refused, never clamped")
    if not bounds.min_grasp_depth_m <= depth <= bounds.max_grasp_depth_m:
        raise ValueError(f"invalid_grasp_centre: grasp_depth_m must be in [{bounds.min_grasp_depth_m:g}, "
                         f"{bounds.max_grasp_depth_m:g}] m (fixed fingertip .. pocket floor)")
    target = grasp_centre_pose(request)
    if any(abs(a - b) > GRASP_CENTRE_CONSISTENCY for field in ("position_m", "quaternion_xyzw")
           for a, b in zip(pose[field], target[field])):
        raise ValueError("invalid_grasp_centre: the move target is not the conversion of input_grasp_centre")


def grasp_centre_report(request, deployment, *, compact=False, planned_tcp=None):
    """Model-visible account of one grasp-centre conversion (``stage_plans[<stage>].grasp_centre`` at preview,
    ``stage_results[<stage>].grasp_centre`` of a reach query and of an executed stage). Pure: CAD + ``deployment``'s
    jaw angles. Lengths m in so101_base unless named ``*_tcp_*``; angles rad; rounded to 0.1 mm / 1e-4.

    Always: the request (``grasp_centre_m``, ``object_width_m``, ``fixed_jaw_clearance_m``, ``grasp_depth_m`` and
    ``fixed_jaw_yaw_deg`` when given) and ``tcp_target_position_m``, the converted gripper_frame_link target.
    Unless ``compact`` (receipts of executed stages), also:

    * ``tcp_target_quaternion_xyzw``; ``tcp_offset_tcp_m``: target minus centre in TCP axes [x, y, z].
    * ``fixed_jaw_direction`` / ``approach_direction``: TCP +X (moving jaw -> fixed jaw) and TCP +Z as unit vectors.
    * ``jaw``: ``straddle_gap_m`` = object_width_m + fixed_jaw_clearance_m and ``straddle_goal_normalized``, the
      jaw goal whose fingertip gap along the jaw line equals it (arrive with a larger goal, or the moving fingertip
      comes down on the object); ``closed_on_object_goal_normalized`` / ``closed_on_object_jaw_rad``, where the
      closing jaw meets an object of that width (a jaw goal below it squeezes); ``moving_fingertip_depth_m``, how
      far behind the TCP plane the moving fingertip is there (compare with grasp_depth_m: larger means it meets
      the object behind the commanded centre plane). A value is null when the width is outside the jaw's range.
    * ``object_shift_on_closing_m``: = fixed_jaw_clearance_m, how far the closing jaw slides a free object along
      ``fixed_jaw_direction`` before the fixed jaw stops it.
    * ``planned`` (with ``planned_tcp`` = ``(position_m, quaternion_xyzw)`` of the planned joints): where the
      grasp centre ends with that TCP pose, ``grasp_centre_m`` = planned position - R(planned) @ tcp_offset_tcp_m,
      and ``centre_error_m``, its distance from the request. The target is converted with the requested
      orientation, so an orientation residual the tolerances accept moves the centre although the TCP position
      matches; this is that effect in metres."""
    from ...model import so101_gripper_geometry as cad
    r4 = lambda values: [round(float(value), 4) + 0.0 for value in values]
    target = grasp_centre_pose(request)
    report = {"grasp_centre_m": r4(request["grasp_centre_m"]),
              **{key: round(request[key], 4) for key in ("object_width_m", "fixed_jaw_clearance_m", "grasp_depth_m")},
              **({"fixed_jaw_yaw_deg": round(request["fixed_jaw_yaw_deg"], 2)} if "fixed_jaw_yaw_deg" in request else {}),
              "tcp_target_position_m": r4(target["position_m"])}
    if compact:
        return report
    gripper = deployment["gripper"]
    closed, opened = gripper["jaw_closed_rad"], gripper["jaw_open_rad"]

    def goal(jaw_rad):
        value = None if jaw_rad is None else (jaw_rad - closed) / (opened - closed)
        return None if value is None or not -1e-9 <= value <= 1 + 1e-9 else round(min(1.0, max(0.0, value)), 3)
    width, straddle = request["object_width_m"], request["object_width_m"] + request["fixed_jaw_clearance_m"]
    contact = cad.jaw_rad_for_gap(width)
    report.update(
        tcp_target_quaternion_xyzw=[round(float(value), 5) + 0.0 for value in target["quaternion_xyzw"]],
        tcp_offset_tcp_m=r4(target["tcp_offset_m"]),
        fixed_jaw_direction=r4(_rotate(request["quaternion_xyzw"], (1.0, 0.0, 0.0))),
        approach_direction=r4(_rotate(request["quaternion_xyzw"], (0.0, 0.0, 1.0))),
        jaw={"straddle_gap_m": round(straddle, 4), "straddle_goal_normalized": goal(cad.jaw_rad_for_gap(straddle)),
             "closed_on_object_goal_normalized": goal(contact),
             "closed_on_object_jaw_rad": None if contact is None else round(contact, 4),
             "moving_fingertip_depth_m": None if contact is None else round(cad.moving_fingertip_depth_m(contact), 4)},
        object_shift_on_closing_m=round(request["fixed_jaw_clearance_m"], 4))
    if planned_tcp is not None:
        position, quaternion = planned_tcp
        centre = [value - delta for value, delta in zip(position, _rotate(quaternion, target["tcp_offset_m"]))]
        report["planned"] = {"grasp_centre_m": r4(centre),
                             "centre_error_m": round(math.dist(centre, request["grasp_centre_m"]), 4)}
    return report


@functools.lru_cache(maxsize=1)
def _kinematic_model():
    """The packaged SO-101 URDF model (``so101_kinematics.load_model()``), parsed once per process."""
    from ...model import so101_kinematics as kinematics
    return kinematics.load_model()


def _planned_tcp(entry):
    """``(position_m, quaternion_xyzw)`` of the joints one preview / reach entry planned, or None: FK of ``q_rad``
    (preview ``stage_plans``), else the entry's ``reached`` pose (reach verdicts, also of unreachable stages)."""
    reached = entry.get("reached")
    if isinstance(reached, dict) and "position_m" in reached and "quaternion_xyzw" in reached:
        return reached["position_m"], reached["quaternion_xyzw"]
    if isinstance(entry.get("q_rad"), list):
        from ...model import so101_kinematics as kinematics
        from .ik_plan import matrix_to_quaternion
        T = kinematics.fk(_kinematic_model(), entry["q_rad"])
        return [float(value) for value in T[:3, 3]], matrix_to_quaternion(T[:3, :3])
    return None


def _grasp_centre_requests(action):
    """``{stage id: input_grasp_centre}`` of the move stages of a compiled BatchAction that carry one."""
    return {stage["id"]: stage["action"][GRASP_CENTRE_INPUT] for stage in action.get("stages") or []
            if isinstance(stage, dict) and isinstance(stage.get("action"), dict) and GRASP_CENTRE_INPUT in stage["action"]}


def annotate_grasp_centres(result, action, deployment, *, key, compact=False, ran_only=False):
    """Add ``grasp_centre`` (``grasp_centre_report``) to ``result[key][<stage id>]`` for every stage of ``action``
    compiled from a grasp-centre request; returns ``result`` (changed in place). ``key``: ``stage_plans`` (preview)
    or ``stage_results`` (reach, execution). ``ran_only``: skip stages still ``pending`` (execution receipts).
    A result without that map (failed preview, rejected action) is returned unchanged."""
    entries = result.get(key) if isinstance(result, dict) else None
    if not isinstance(entries, dict):
        return result
    for identifier, request in _grasp_centre_requests(action).items():
        entry = entries.get(identifier)
        if isinstance(entry, dict) and not (ran_only and entry.get("status") == "pending"):
            entry["grasp_centre"] = grasp_centre_report(request, deployment, compact=compact,
                                                        planned_tcp=None if compact else _planned_tcp(entry))
    return result


def _number(value, low, high, name, *, low_open=True):
    if type(value) not in (int, float) or not math.isfinite(value) or value > high or value < low or (low_open and value <= low):
        raise ValueError(f"invalid_action: {name} must be in {'(' if low_open else '['}{low}, {high}]")
    return float(value)


def arrival_correction_limits(deployment):
    """``(rounds, max_offset_rad)``: the deployment's sag-correction bounds (``motion.arrival_correction_rounds``,
    count; ``motion.arrival_correction_max_offset_rad``, rad per joint). They are the default of a move and the
    largest value its ``arrival_correction_rounds`` / ``arrival_correction_max_offset_rad`` may request."""
    motion = deployment["motion"]
    return int(motion["arrival_correction_rounds"]), float(motion["arrival_correction_max_offset_rad"])


def arrival_correction(path, deployment):
    """``(rounds, max_offset_rad)`` one compiled move path uses: its own fields, else ``arrival_correction_limits``."""
    rounds, offset = arrival_correction_limits(deployment)
    if path.get("arrival_correction_rounds") is not None:
        rounds = int(path["arrival_correction_rounds"])
    if path.get("arrival_correction_max_offset_rad") is not None:
        offset = float(path["arrival_correction_max_offset_rad"])
    return rounds, offset


def _validate_path_fields(path, profile, deployment=None):
    """Refuse path fields the worker would ignore; check preset names, the vertical window, the stall-guard bound
    and the arrival-correction bounds against ``profile`` and the deployment limits (``deployment`` None: the
    profile's ``arrival_correction_limits``, which the deployment's profile builder copies from the same fields)."""
    honoured = (set(PATH_FIELDS) - (set() if "max_upward_dz" in profile.supports else {"max_upward_dz_m"})
                - (set() if "approach_roll" in profile.supports else set(SPLIT_ORIENTATION_PATH_FIELDS))
                - (set() if "arrival_correction" in profile.supports else set(ARRIVAL_CORRECTION_PATH_FIELDS)))
    for field in sorted(set(path) - honoured):
        reason = _PATH_FIELD_REASONS.get(field, "not read by the SO-101 worker")
        raise ValueError(f"unsupported_capability: {field} ({reason})")
    if path.get("max_upward_dz_m") is not None:
        _number(path["max_upward_dz_m"], 0.0, MAX_UPWARD_DZ_M, "max_upward_dz_m", low_open=False)
    for field in SPLIT_ORIENTATION_PATH_FIELDS:
        if path.get(field) is not None:
            _number(path[field], 0.0, math.pi + 1e-9, field)
    preset = path.get("tolerance_preset")
    if preset is not None and preset not in profile.tolerance_presets:
        raise ValueError(f"unsupported_capability: tolerance_preset {preset} (presets of this deployment: "
                         f"{', '.join(profile.tolerance_presets)})")
    stall, divergence = path.get("stall_guard_preset"), path.get("max_tracking_divergence_m")
    if "stall_guard" not in profile.supports and (stall not in (None, "off") or divergence is not None):
        raise ValueError("unsupported_capability: stall_guard_preset / max_tracking_divergence_m")
    offered = offered_stall_guard_presets(profile)
    if stall is not None and stall not in offered:
        # Also reached by a batch-wide stall_guard_preset, which the Harness resolves into every move stage's path.
        raise ValueError(f"unsupported_capability: stall_guard_preset {stall} (presets on this SO-101: "
                         f"{', '.join(offered)})")
    if divergence is not None:
        _number(divergence, *profile.tracking_divergence_range_m, "max_tracking_divergence_m", low_open=False)
    max_rounds, max_offset = (arrival_correction_limits(deployment) if deployment is not None else
                              arrival_correction_bounds(profile))
    rounds, offset = path.get("arrival_correction_rounds"), path.get("arrival_correction_max_offset_rad")
    if rounds is not None and (type(rounds) is not int or not 0 <= rounds <= max_rounds):
        raise ValueError(f"invalid_action: arrival_correction_rounds must be an integer in [0, {max_rounds}] "
                         "(this deployment's motion.arrival_correction_rounds)")
    if offset is not None:
        if max_rounds == 0 or max_offset <= 0:
            raise ValueError("invalid_action: arrival_correction_max_offset_rad cannot be set: this deployment applies "
                             "no arrival correction")
        _number(offset, 0.0, max_offset, "arrival_correction_max_offset_rad")


def validate_leaf(action, deployment):
    """One EEAction (single request or batch stage). Returns its arm id (``"right"``) or ``None`` for waits."""
    profile = embodiment(deployment)
    arms = profile.arms
    if action.get("type", "EEAction") != "EEAction":
        raise ValueError("unsupported_capability: " + str(action.get("type")))
    if action.get("mode") != "absolute" or action.get("frame_id") != profile.frame_id:
        raise ValueError(f"unsupported_capability: absolute pose in {profile.frame_id} required")
    if action.get("tcp_config_id") != profile.tcp_config_id:
        raise ValueError(f"unsupported_capability: TCP must be {profile.tcp_config_id}")
    if action.get("limits_profile") not in profile.limits_profiles:
        raise ValueError(f"unsupported_capability: limits_profile must be {profile.limits_profiles[0]}")
    for field in (*UNSUPPORTED_FIELDS, "input_grasp_pose"):
        if field in action:
            raise ValueError(_unsupported_field(field))
    for field, scale in (("speed_scale", profile.speed_scale), ("gripper_speed_scale", profile.gripper_speed_scale)):
        if field in action:
            _number(action[field], scale.min, scale.max, field, low_open=False)
    motion = deployment["motion"]
    _number(action.get("timeout_s"), 0.0, profile.max_timeout_s, "timeout_s")
    path = action.get("path")
    if not isinstance(path, dict):
        raise ValueError("invalid_action: path must be an object")
    points = path.get("waypoints", [])
    events = action.get("gripper_events", [])
    if not isinstance(points, list) or not isinstance(events, list):
        raise ValueError("invalid_action: waypoints and gripper_events must be lists")
    if path.get("blend_radius_m", 0) != 0 or path.get("orientation", "backend_free") != "backend_free":
        raise ValueError("unsupported_capability: blending or orientation path constraints")
    _validate_path_fields(path, profile, deployment)
    for event in events:
        phase = event.get("phase") if isinstance(event, dict) else None
        if phase == "during_motion" and "gripper_motion" in profile.supports:
            if set(event) != DURING_MOTION_EVENT_FIELDS:
                raise ValueError("invalid_gripper_motion: a during_motion event has exactly arm, phase, goal_normalized, "
                                 "start_fraction, end_fraction")
            validate_gripper_motion({key: event[key] for key in ("start_fraction", "end_fraction")})
            if path.get("kind") != "backend_free" or not points:
                raise ValueError("invalid_gripper_motion: requires a moving arm")
        elif phase != "after_motion":
            raise ValueError("unsupported_capability:gripper_events:during_motion")
        if event.get("arm") not in arms:
            raise ValueError(f"unsupported_arm:{event.get('arm')}")
        _number(event.get("goal_normalized"), 0.0, 1.0, "goal_normalized", low_open=False)
    if len(events) > 1:
        raise ValueError("unsupported_capability: at most one gripper event")
    if path.get("kind") == "hold":
        if points:
            raise ValueError("unsupported_capability: hold requires empty waypoints")
        if GRASP_CENTRE_INPUT in action:
            raise ValueError("invalid_grasp_centre: grasp_centre_m and its fields belong to a move stage")
        if "hold_native_steps" in action:
            steps = action["hold_native_steps"]
            if type(steps) is not int or not 1 <= steps <= MAX_WAIT_STEPS or events:
                raise ValueError(f"invalid_action: hold_native_steps must be an integer in [1, {MAX_WAIT_STEPS}] "
                                 "without gripper events")
            return None
        if len(events) != 1:
            raise ValueError("invalid_action: hold requires a standalone gripper event")
        return events[0]["arm"]
    if path.get("kind") != "backend_free" or not 1 <= len(points) <= motion["max_waypoints"]:
        raise ValueError(f"unsupported_capability: 1..{motion['max_waypoints']} backend_free waypoints required")
    if "hold_native_steps" in action:
        raise ValueError("invalid_action: hold_native_steps requires a hold path")
    has_centre = GRASP_CENTRE_INPUT in action
    if has_centre and len(points) != 1:
        raise ValueError("invalid_grasp_centre: a grasp-centre move has one target")
    for point in points:
        if not isinstance(point, dict) or len(point) != 1 or not set(point) <= set(arms):
            raise ValueError("unsupported_arm:" + ",".join(sorted(set(point) - set(arms))) if isinstance(point, dict)
                             else "invalid_pose: waypoint must be an object")
        pose = next(iter(point.values()))
        if not isinstance(pose, dict):
            raise ValueError("invalid_pose: pose must be an object")
        for field, size in (("position_m", 3), ("quaternion_xyzw", 4)):
            value = pose.get(field)
            if (not isinstance(value, list) or len(value) != size
                    or any(type(x) not in (int, float) or not math.isfinite(x) for x in value)):
                raise ValueError("invalid_pose: finite position_m[3] and quaternion_xyzw[4] required")
        if abs(math.sqrt(sum(x * x for x in pose["quaternion_xyzw"])) - 1) > 1e-3:
            raise ValueError("invalid_pose: quaternion must be normalized")
        if not has_centre:
            check_workspace(deployment, pose["position_m"])
            continue
        # Grasp-centre move: its own bounds first (they name the request's fields), then the workspace box on the
        # converted TCP target, saying where that target came from.
        _validate_grasp_centre(action, pose, profile)
        centre = action[GRASP_CENTRE_INPUT]
        try:
            check_workspace(deployment, pose["position_m"])
        except ValueError as error:
            raise ValueError(f"{error} (the {profile.tcp_label} target converted from grasp_centre_m "
                             f"{[round(float(value), 4) for value in centre['grasp_centre_m']]} with object_width_m "
                             f"{centre['object_width_m']:g}, fixed_jaw_clearance_m {centre['fixed_jaw_clearance_m']:g} "
                             f"and grasp_depth_m {centre['grasp_depth_m']:g}; move the centre or change the "
                             "orientation)") from None
    _number(path.get("position_tolerance_m"), 0.0, profile.max_position_tolerance_m, "position_tolerance_m")
    _number(path.get("orientation_tolerance_rad"), 0.0, profile.max_orientation_tolerance_rad, "orientation_tolerance_rad")
    return next(iter(points[0]))


def grip_guard_gripper(deployment):
    """Zero point and range of the SO-101 jaw for grip guards (m), from the profile (``profile.gripper``).

    Returns ``{"empty_close_aperture_m", "full_open_aperture_m", "full_open_source"}``:
    the deployment's measured empty-close gap (None when not measured: presets are then
    rejected) and the full-open gap (deployment value, else the CAD opening at
    ``gripper.jaw_open_rad``). Both are CAD fingertip openings of measured jaw angles.
    """
    gripper = embodiment(deployment).gripper
    return {"empty_close_aperture_m": gripper.empty_close_aperture_m, "full_open_aperture_m": gripper.full_open_aperture_m,
            "full_open_source": gripper.full_open_source}


def explicit_aperture_limit_m(deployment):
    """Largest explicit min/max_aperture_m (m): the full opening of this jaw (``profile.gripper.full_open_aperture_m``;
    development unit 0.1361 m measured, CAD 0.1372 m)."""
    return embodiment(deployment).gripper.full_open_aperture_m


def guard_parameters(deployment):
    """Keyword arguments for the shared ``grip_guard`` rules on this deployment.

    ``profile``: the deployment's ``EmbodimentProfile`` (accepted presets and their margins: ``none`` and the four
    schema presets). ``empty_close_aperture_m`` (m or None = not measured) and ``max_aperture_m`` (m, the full
    opening) are the profile's values, passed explicitly so every reader of these parameters sees the numbers in
    use. An unmeasured zero point resolves no bounded preset (presets are rejected before this) and leaves explicit
    bounds absolute; the post-grasp re-check, which depends on it, is switched by ``post_grasp_recheck_available``.
    """
    profile = embodiment(deployment)
    return {"profile": profile, "empty_close_aperture_m": profile.gripper.empty_close_aperture_m,
            "max_aperture_m": profile.gripper.full_open_aperture_m}


RECHECK_NOT_MEASURED = "gripper.empty_close_aperture_m not measured"
RECHECK_HOLD_SQUEEZE = ("holding squeeze (gripper.hold_squeeze_rad): a jaw that loses its object stops at its hold goal "
                        "instead of closing to the empty-close gap")


def post_grasp_recheck_unavailable_reason(deployment):
    """None when the post-grasp slip re-check can work on this deployment, else why it cannot (text).

    The re-check pauses when the jaw closed further during the first move after a grasp and ends within 0.002 m of
    the measured empty-close gap. That is undefined while ``gripper.empty_close_aperture_m`` is unmeasured, and
    cannot happen with a holding squeeze (``gripper.hold_squeeze_rad`` set): the jaw goal is then at most the
    squeeze beyond the contact angle, so a jaw that loses its object stops there. Stage results of later moves
    report ``grip_hold.squeeze_remaining_rad`` instead."""
    if embodiment(deployment).gripper.empty_close_aperture_m is None:
        return RECHECK_NOT_MEASURED
    return RECHECK_HOLD_SQUEEZE if hold_squeeze(deployment) is not None else None


def post_grasp_recheck_available(deployment):
    """True when the post-grasp slip re-check is armed on this deployment (``post_grasp_recheck_unavailable_reason``
    is None): a measured empty-close gap and no holding squeeze."""
    return post_grasp_recheck_unavailable_reason(deployment) is None


def closed_limit_opening_m(deployment):
    """CAD fingertip opening (m) at ``gripper.jaw_closed_rad``: no goal-0 close can read less (0.0031 m for -0.174533 rad)."""
    from ...model.so101_gripper_geometry import opening_width_m
    return round(opening_width_m(deployment["gripper"]["jaw_closed_rad"]), 5)


def detects_empty_close(min_aperture_m, deployment):
    """Reported fact for one resolved guard: can its minimum gap catch an EMPTY close on this jaw?

    False when there is no minimum, or it is at or below the measured empty-close gap or the CAD
    closed-limit floor (the guard can never fire on an empty close); True above the measured gap;
    None (unknown) when the gap is unmeasured and the minimum is above the CAD floor. The bound is
    kept as given either way.
    """
    if min_aperture_m is None or min_aperture_m <= closed_limit_opening_m(deployment):
        return False
    zero = embodiment(deployment).gripper.empty_close_aperture_m
    return None if zero is None else min_aperture_m > zero


def validate_grip_guard(mapping, deployment):
    """Validate grip-guard fields of one stage or batch against the SO-101 jaw; returns the resolved bounds or None.

    Explicit ``min_aperture_m`` / ``max_aperture_m`` (m) must lie in [0, full opening of this jaw]; presets must be
    presets of the profile (``none`` and the four schema presets) and, except ``none``, need the measured zero point.
    Bounds are refused, never clamped."""
    from ...motion.grip_guard import validate_grip_guard_fields
    if not any(field in mapping for field in GRIP_GUARD_FIELDS):
        return None
    gripper = embodiment(deployment).gripper
    limit = gripper.full_open_aperture_m
    for field in ("min_aperture_m", "max_aperture_m"):
        value = mapping.get(field)
        if type(value) in (int, float) and math.isfinite(value) and not 0.0 <= value <= limit:
            raise ValueError(f"invalid_{field}: {value} m is outside [0, {limit}] m (full opening of this SO-101 jaw, "
                             f"{gripper.full_open_source}); bounds are refused, never clamped")
    preset = mapping.get("grip_guard_preset")
    if preset is not None and preset not in gripper.preset_margins_m:
        raise ValueError(f"invalid_grip_guard_preset: {preset} (presets on this SO-101: "
                         f"{', '.join(gripper.preset_margins_m)})")
    if preset not in (None, "none") and gripper.empty_close_aperture_m is None:
        raise ValueError(
            "unsupported_capability: grip_guard_preset requires the measured empty-close fingertip gap "
            "gripper.empty_close_aperture_m (m) in the SO-101 deployment (measure with an empty goal-0 close and set "
            "gripper.empty_close_aperture_m in so101.local.json, or use explicit min_aperture_m / max_aperture_m (m))"
        )
    return validate_grip_guard_fields(mapping, **guard_parameters(deployment))


def validate_action(action, deployment):
    """Backend validator: the SO-101 numeric gate after shared Harness checks."""
    profile = embodiment(deployment)
    arms = profile.arms
    kind = action.get("type", "EEAction")
    if kind == "BatchAction":
        stages = action.get("stages")
        if not isinstance(stages, list) or not stages:
            raise ValueError("batch_requires_stages")
        if action.get("frame_id") != profile.frame_id or action.get("tcp_config_id") != profile.tcp_config_id:
            raise ValueError("unsupported_batch_frame_or_tcp")
        if action.get("limits_profile") not in profile.limits_profiles:
            raise ValueError("unsupported_batch_limits")
        _number(action.get("timeout_s"), 0.0, profile.max_timeout_s, "timeout_s")
        for field in _unsupported_batch_fields(profile):
            if field in action or any(field in stage for stage in stages if isinstance(stage, dict)):
                raise ValueError(f"unsupported_capability: {field}")
        if "review_proxies" in action:
            validate_review_proxies(action["review_proxies"], arms)
        if action.get("max_upward_dz_m") is not None:
            # Batch-wide window: the Harness resolves it into every move stage's path, where the worker reads it.
            _number(action["max_upward_dz_m"], 0.0, MAX_UPWARD_DZ_M, "max_upward_dz_m", low_open=False)
        if "checkpoint_after_gripper" in action and not isinstance(action["checkpoint_after_gripper"], bool):
            raise ValueError("invalid_checkpoint_after_gripper")
        validate_grip_guard(action, deployment)
        for stage in stages:
            if not isinstance(stage, dict):
                raise ValueError("invalid_batch_stage")
            if stage.get("kind") == "observe":
                if stage.get("arm") not in (None, *arms):
                    raise ValueError(f"unsupported_arm:{stage.get('arm')}")
                if any(field in stage for field in (*GRIP_GUARD_FIELDS, "checkpoint_after_gripper", GRIP_SQUEEZE_FIELD)):
                    raise ValueError("unsupported_capability: grip_guard_* / min_aperture_m / max_aperture_m / "
                                     "checkpoint_after_gripper / grip_squeeze_rad apply to jaw stages, not observe stages")
                continue
            if stage.get("arm") not in arms:
                raise ValueError(f"unsupported_arm:{stage.get('arm')}")
            leaf = stage.get("action")
            if not isinstance(leaf, dict):
                raise ValueError("batch_command_requires_arm_action")
            validate_leaf(leaf, deployment)
            if stage.get("kind") == "move" and (leaf["path"]["kind"] != "backend_free" or len(leaf["path"]["waypoints"]) != 1):
                raise ValueError("batch_move_requires_one_owned_target")
            if stage.get("kind") == "move" and any(event.get("phase") != "during_motion"
                                                   for event in leaf.get("gripper_events", [])):
                # An after-arrival jaw goal of a move is compiled into its own gripper stage by the Harness.
                raise ValueError("batch_after_motion_jaw_requires_separate_stage")
            if stage.get("kind") == "gripper" and (leaf["path"]["kind"] != "hold" or len(leaf.get("gripper_events", [])) != 1):
                raise ValueError("batch_gripper_requires_one_jaw_event")
            jaw_fields = [field for field in (*GRIP_GUARD_FIELDS, "checkpoint_after_gripper", GRIP_SQUEEZE_FIELD)
                          if field in stage]
            if jaw_fields and stage.get("kind") != "gripper" and not leaf.get("gripper_events"):
                # Only a move with a during-motion jaw ramp (gripper_motion) commands the jaw inside a move stage.
                raise ValueError(f"unsupported_capability: {jaw_fields[0]} applies to jaw stages (a gripper stage, or a "
                                 "move with gripper_goal_normalized); this move stage has no jaw command")
            if "checkpoint_after_gripper" in stage and not isinstance(stage["checkpoint_after_gripper"], bool):
                raise ValueError("invalid_checkpoint_after_gripper")
            validate_grip_guard(stage, deployment)
            validate_grip_squeeze(stage, profile=profile)
        return arms[0], None
    if kind != "EEAction":
        raise ValueError(f"unsupported_capability: {kind}")
    return validate_leaf(action, deployment), None


def validate_review_proxies(proxies, arms):
    """The ``review_proxies`` of a compiled batch on a ``collision_world`` profile: only ``obstacles`` and
    ``held_objects`` of this arm, in the shared proxy form (``proxy_contract.validate_proxies``: finite boxes with
    ``max_m > min_m`` inside +-3 m, ``margin_m`` in [0, 0.2] m, ``evidence`` text, unique ids). Contact declarations
    (``contact_entity_id``, ``initial_contact``, ``placement_contact``) are refused: nothing here reads them."""
    from ...motion.proxy_contract import validate_proxies
    if not isinstance(proxies, dict):
        raise ValueError("invalid_batch_review_proxies")
    for field in sorted(set(proxies) - set(REVIEW_PROXY_FIELDS)):
        raise ValueError(f"unsupported_capability: {field}")
    for arm in sorted(set(proxies.get("held_objects") or {}) - set(arms)):
        raise ValueError(f"unsupported_arm:{arm}")
    return validate_proxies(proxies)


def offered_stall_guard_presets(profile):
    """``{name: max_tracking_divergence_m (m) or None (off)}`` of the stall-guard presets this SO-101 offers: the
    schema's names (``SCHEMA_STALL_GUARD_PRESETS``) that ``profile.stall_guard_presets_m`` defines, in schema order."""
    return {name: profile.stall_guard_presets_m[name] for name in SCHEMA_STALL_GUARD_PRESETS
            if name in profile.stall_guard_presets_m}


def _unsupported_batch_fields(profile):
    """Batch / stage fields refused on ``profile``: ``UNSUPPORTED_BATCH_FIELDS`` plus ``max_upward_dz_m`` without the
    ``max_upward_dz`` tag, ``review_proxies`` without the ``collision_world`` tag and the split orientation bounds
    without the ``approach_roll`` tag."""
    return (*UNSUPPORTED_BATCH_FIELDS, *(() if "max_upward_dz" in profile.supports else ("max_upward_dz_m",)),
            *(() if "collision_world" in profile.supports else ("review_proxies",)),
            *(() if "approach_roll" in profile.supports else SPLIT_ORIENTATION_PATH_FIELDS))


def _stall_guard_sentence(deployment, profile):
    """Honest stall-guard semantics on this backend (findings A13, R7): presets by threshold, bound, what a stop does
    (``profile.stall_guard_semantics``), the stop signals every move has, and the ``contact_block_summary`` fields."""
    motion = deployment["motion"]
    always = (f"Every move also stops when any joint lags its command by more than {motion['tracking_error_limit_rad']} "
              f"rad, or is commanded at least {motion['stall_min_commanded_rad']} rad within {motion['stall_window_s']} s "
              f"but moves less than {motion['stall_min_progress_rad']} rad.")
    summary = ("A failed move (any stop, or arrival outside the rule) and a contact_stop hold report contact_block_summary "
               f"(stage result; also top-level when the batch fails there): measured_position_m and target_position_m (TCP, "
               f"m in {profile.frame_id}), vertical_shortfall_m (measured z minus target z), horizontal_offset_m, "
               "approach_shortfall_m (target minus measured along the target's TCP +Z approach axis; > 0: stopped short), "
               "blocked_above_target (vertical_shortfall_m > position_tolerance_m), completed_steps / planned_steps "
               "(commands).")
    if "stall_guard" not in profile.supports:
        return f"There is no stall guard on this arm. {always} Such a stop fails the stage with tracking_error. {summary}"
    offered = offered_stall_guard_presets(profile)
    presets = "; ".join(f"{name} {value:g} m" for name, value in offered.items() if value is not None)
    low, high = profile.tracking_divergence_range_m
    contact = ""
    if "contact_stop" in offered:
        window = (" and a max_upward_dz_m covering how far above the target the contact may stop" if
                  "max_upward_dz" in profile.supports else "")
        contact = (" A contact_stop hold pins the joint goals to the measured pose (no sag offset, no arrival "
                   "correction), so it neither pushes on toward the target nor lifts off the support; a later pause at the "
                   "held pose (observe stage, checkpoint) keeps that goal. For a descent "
                   f"onto a support, set contact_stop{window}; otherwise the held pose must be within "
                   "position_tolerance_m of the target.")
    return (f"Stall guard (move stages): stall_guard_preset (default off; {presets}) or max_tracking_divergence_m in "
            f"[{low:g}, {high:g}] m {profile.stall_guard_semantics}. {always}{contact} {summary}")


def _vertical_window_sentence(deployment, profile):
    """The ``max_upward_dz_m`` arrival rule of this backend (design 13 R6), or "" without the ``max_upward_dz`` tag.

    States that the window changes only the verdict: before it, ``SO101Adapter._arrive`` runs the bounded sag correction
    (``arrival_correction`` tag; its bounds are documented once, in the robot_action field help of
    ``arrival_correction_rounds``), which moves the joint goals further toward a target the arm rests short of, i.e.
    pushes harder on a support above the target."""
    if "max_upward_dz" not in profile.supports:
        return ""
    correction = ("before the verdict the arrival correction (robot_action arrival_correction_rounds) shifts the joint "
                  "goals further toward the target, so an arm resting on a support above the target presses harder on "
                  "it; only a contact_stop hold (a stop signal during the move) or arrival_correction_rounds=0 on the "
                  "stage skips the correction" if "arrival_correction" in profile.supports else
                  "this deployment applies no arrival correction (motion.arrival_correction_rounds or "
                  "arrival_correction_max_offset_rad is 0), so the joint goals stay at the planned target")
    return (f"Vertical arrival window: max_upward_dz_m (m, 0..{MAX_UPWARD_DZ_M:g}; default 0 = no window; on a move stage "
            "or batch-wide) also accepts an arrival whose horizontal error sqrt(dx_m^2 + dy_m^2) <= position_tolerance_m "
            "with -position_tolerance_m <= dz_m <= max_upward_dz_m and the orientation within tolerance; the stage "
            "result then has vertical_compliance_accepted=true and xy_error_m. It is an acceptance rule only: the servos "
            f"still drive toward the target (no vertical compliance or force control), and {correction}. "
            "effective_tolerances.max_upward_dz_m reports the window used.")


def _grasp_sentence(deployment, profile):
    """The grasp-centre move input of this backend (``grasp_centre_request`` / ``grasp_centre_pose``): a pointer to its
    contract, one worked example with this deployment's numbers, its reports and the measured
    ``robot_state.arms.<arm>.jaw_grasp``; "" without the ``jaw_grasp_centre`` pose input. The conversion, units,
    defaults and bounds are explained once, in the robot_action field help of the grasp-centre fields
    (``robot_action_schema``: ``grasp_centre_m``, ``object_width_m``, ``fixed_jaw_clearance_m``, ``grasp_depth_m``,
    ``fixed_jaw_yaw_deg``). Every number is the profile's ``gripper.jaw_grasp`` (CAD) or computed from the deployment's
    jaw angles."""
    bounds = profile.gripper.jaw_grasp
    if "jaw_grasp_centre" not in profile.pose_inputs or bounds is None:
        return ""
    arm, tcp, frame = profile.arms[0], profile.tcp_label, profile.frame_id
    example = grasp_centre_request({"grasp_centre_m": [0.24, -0.06, 0.015], "object_width_m": 0.03,
                                    "fixed_jaw_clearance_m": 0.004, "fixed_jaw_yaw_deg": 90})
    report = grasp_centre_report(example, deployment)
    numbers = lambda values: "[" + ", ".join(f"{value:g}" for value in values) + "]"
    return (
        f"Grasp centre: the {tcp} origin is on the FIXED jaw's inner fingertip face, not between the jaws, and only the "
        f"other jaw moves. To grasp or carry an object a move stage may give grasp_centre_m (where the object's centre "
        f"sits between the jaws) + object_width_m with quaternion_xyzw or fixed_jaw_yaw_deg instead of position_m; the "
        f"Harness converts it with the gripper CAD to the {tcp} target, so add no jaw offset yourself. The robot_action "
        "field help of grasp_centre_m, object_width_m, fixed_jaw_clearance_m, grasp_depth_m and fixed_jaw_yaw_deg is "
        "the contract (conversion, units, defaults, bounds). Tolerances, speed, gripper_goal_normalized, gripper_motion "
        "and the stall guard apply to the converted target as on any move; an accepted orientation residual moves the "
        f"grasp centre about the {tcp} origin (planned.centre_error_m below).\n"
        f"Example: {{\"id\": \"descend\", \"kind\": \"move\", \"arm\": \"{arm}\", \"grasp_centre_m\": [0.24, -0.06, 0.015], "
        "\"object_width_m\": 0.03, \"fixed_jaw_clearance_m\": 0.004, \"fixed_jaw_yaw_deg\": 90} targets "
        f"{tcp} at {numbers(report['tcp_target_position_m'])} (fixed jaw toward +y); it must arrive with a jaw goal above "
        f"{report['jaw']['straddle_goal_normalized']:g} (fingertips {report['jaw']['straddle_gap_m']:g} m apart) and the jaw "
        f"meets the 0.03 m object at goal {report['jaw']['closed_on_object_goal_normalized']:g}, so a lower goal grips it.\n"
        "Preview stage_plans[<stage>].grasp_centre and reach stage_results[<stage>].grasp_centre report "
        "tcp_target_position_m / tcp_target_quaternion_xyzw, tcp_offset_tcp_m, fixed_jaw_direction and approach_direction "
        "(unit vectors), jaw {straddle_gap_m, straddle_goal_normalized, closed_on_object_goal_normalized, "
        "closed_on_object_jaw_rad, moving_fingertip_depth_m}, object_shift_on_closing_m and planned {grasp_centre_m, "
        "centre_error_m} (where the planned joints put the centre); executed stage_results[<stage>].grasp_centre repeats "
        "the request and tcp_target_position_m. A move stage's own id is \"<id>/motion\" when it also carries an "
        "after-arrival gripper_goal_normalized. Preview grasp_3d_review audits grasp_centre_m stages after which the jaw "
        "closes (advisory).\n"
        f"robot_state.arms.{arm}.jaw_grasp is the measured counterpart (m in {frame}, CAD fingertips at the measured jaw "
        "angle): grasp_centre_m (midway between the fixed fingertip face and the moving fingertip along the jaw line, on "
        f"the {tcp} origin plane), gap_m (their distance along the jaw line: about the object's width while it is "
        "gripped), fixed_tip_m, moving_tip_m and fixed_jaw_direction (unit vector from the moving to the fixed jaw). To "
        "put a held object's centre at P, move with grasp_centre_m = P, object_width_m = gap_m and the same grasp_depth_m "
        "you grasped it with.")


def _jaw_ramp_sentence(deployment, profile):
    """The during-motion jaw ramp of this backend (design 13 R5), or "" without the ``gripper_motion`` tag."""
    if "gripper_motion" not in profile.supports:
        return ""
    g, rate = deployment["gripper"], deployment["motion"]["command_rate_hz"]
    return ("Jaw during a move: a move stage with gripper_goal_normalized and gripper_motion {start_fraction, end_fraction} "
            "ramps the jaw goal linearly from its last commanded value over that fraction of the move's commands "
            "(without gripper_motion the jaw moves after arrival). The move is slowed (each interpolation interval "
            f"subdivided) until no jaw command step exceeds gripper.max_speed_rad_s ({g['max_speed_rad_s']:g} rad/s) x "
            "gripper_speed_scale x the operator speed cap; preview stage_costs and stage_plans[<move>].gripper_motion "
            f"show the schedule, and one needing more than timeout_s x {rate:g} commands is refused. If a contact_stop "
            "hold ends the move first, the rest of the ramp runs at the held pose (finished_at_held_pose). After arrival the "
            f"worker waits for the jaw to settle (gripper.settle_timeout_s {g['settle_timeout_s']:g} s); grip_guard_preset, "
            "min_aperture_m / max_aperture_m and checkpoint_after_gripper apply to that move as to a gripper stage.")


def _arrival_correction_sentence(deployment, profile):
    """A pointer to the per-stage sag-correction bounds (``arrival_correction`` tag), or "" without the tag. The full
    explanation (meaning, units, defaults, bounds, receipts) is the robot_action field help of
    ``arrival_correction_rounds`` / ``arrival_correction_max_offset_rad`` (``robot_action_schema``), from the same
    profile limits."""
    if "arrival_correction" not in profile.supports:
        return ""
    return ("Arrival correction: after a move settles short of its target the worker shifts the joint goal by the "
            "settled shortfall (servo sag) and settles again; arrival_correction_rounds / "
            "arrival_correction_max_offset_rad bound it per move stage or batch-wide (their robot_action field help "
            "gives the limits and defaults); set arrival_correction_rounds=0 when the arm is to rest on a support or set "
            "a held object down, so it does not push harder.")


def _compensation_sentence(deployment):
    """What the deployment's ``motion_compensation`` adds to executed moves (``deployment.active_motion_compensation``),
    or "" when every mechanism is off. Names only the receipt fields the model then sees."""
    active = active_motion_compensation(deployment)
    if not active:
        return ""
    parts = []
    if active.get("carry_arrival_offset"):
        parts.append("the sag offset the last free-space arrival ended with is added to every goal of the next move "
                     "(path_compensation.carried_offset_rad, rad per arm joint; path_compensation.carry says where it "
                     "came from: last_arrival, or zeros with correction_disabled, contact_stage or unavailable), and "
                     "that move's arrival correction continues from it (arrival_correction.start_goal_offset_rad)")
    if active.get("gravity_feedforward"):
        parts.append("each goal also gets a gravity feedforward offset for the arm's own weight, at most "
                     f"{active['gravity_feedforward']['max_offset_rad']:g} rad per joint "
                     "(path_compensation.gravity_feedforward_rad at the target; a held object's weight is not "
                     "included), so arrival correction only handles what is left; it also acts while pressing on a "
                     "support, so a stopped press pushes harder than without it by about the arm's own weight")
    if active.get("no_response_stop"):
        parts.append("an arrival-correction round the joints did not answer (responded=false) ends the correction with "
                     "termination_reason no_response and its step is taken back")
    if active.get("stop_correction_within_tolerance"):
        parts.append("arrival correction stops as soon as a settled pose is inside the move's own tolerances "
                     "(termination_reason within_tolerance) and that pose is judged, so the final error can be as "
                     "large as the tolerance you set; set a tighter tolerance where you need more accuracy")
    if active.get("arrival_breakaway"):
        parts.append("a correction step below a joint's identified static-friction band is enlarged to that band "
                     "(breakaway_joints in the round)")
    return (" Motion compensation on this deployment: " + "; ".join(parts) + ". Guards, residuals and transit are "
            "always measured against the commanded path without these offsets.")


def _path_sentence(deployment, profile):
    """What a move does between two targets and what preview reports about it: FK path facts on every SO-101, the
    cuRobo path collision check with the ``path_check`` tag, obstacle and held-object boxes with ``collision_world``."""
    text = ("Path between targets: a move runs on a straight line in JOINT space from the measured start to the planned "
            "joint target, so the TCP does not follow the straight Cartesian line between them; on a long low move it can "
            "pass centimetres below or beside that line. Preview reports stage_plans[<move>].path from forward kinematics of "
            f"the exact commands the move will send (m in {profile.frame_id}): min_tcp_z_m, min_fingertip_z_m (lowest CAD "
            "fingertip point), lowest_at_fraction (0 = start, 1 = target), max_line_deviation_m (largest TCP distance from "
            "the straight segment start->target) and dip_below_endpoints_m (how far the TCP passes under the lower "
            "endpoint); the trajectory overlay is drawn through samples of those commands. If a move passes too low or too "
            "far from the line, split it with via move stages (lift, traverse, descend). Execution starts each move from "
            "the measured joints; its stage result gives path_start_deviation_rad (largest joint difference from the "
            "previewed start). The preview describes the COMMANDED path; the loaded arm sags below it. Each "
            "executed move reports transit: max_below_path_m (largest amount the measured TCP was lower than the "
            "commanded path, m), max_deviation_m (largest 3D distance, m), their *_at_fraction (0 = start, 1 = "
            "target), and duration_s."
            + _compensation_sentence(deployment))
    if "path_check" not in profile.supports:
        return text + " Nothing checks the path for collision on this deployment."
    service = deployment["ik_service"]
    table = service.get("table_top_z_m")
    world = "self-collision" + (f", the table box (top z={table:g} m)" if table is not None else "")
    boxes = ""
    if "collision_world" in profile.supports:
        ids = ", obstacle_object_ids (object_table rows)" if "obstacle_object_ids" in profile.supports else ""
        world += ", the request's obstacle boxes and its held_object box"
        boxes = (f" Collision world: obstacles (boxes you estimate){ids} and held_object (the box of what the jaw holds, "
                 "carried with the gripper until the first stage that opens the jaw) apply to this request only. Planned "
                 "joint targets prefer solutions clear of them; ik.obstacle_free=false marks a target that is not. Objects "
                 "you do not pass are not checked.")
    return (text + f" Path collision check (advisory): preview samples every move's joint path and every standalone "
            "gripper stage's jaw sweep (the arm still, the jaw from its angle before the stage to the commanded goal), at most "
            f"{service['path_check_step_rad']:g} rad of joint change apart, in cuRobo's sphere model of the arm (the jaw "
            "at its measured angle until the plan commands it, then on the commanded schedule, a gripper_motion ramp "
            f"included) against {world}. It checks the commanded path (path_check.checked_path), not the measured "
            "one (transit). path_check.status is clear, collisions or unavailable (with reason); colliding_stage_ids lists the "
            "stages, and stage_plans[<stage>].path.collision gives the first colliding sample: fraction (0 = stage start, "
            "1 = its end), tcp_position_m, jaw_rad (jaw angle at that sample, rad), "
            "world [{link, penetration_m, obstacles}] (obstacle \"table\" = the table box, link attached_object = the held "
            "box) and self [{links, penetration_m}]. Nothing is blocked or re-routed: add a via stage, raise the path, or "
            "accept a contact you intend (fingers closing on the object you grasp, a held object set on its support). "
            "mode=execute_if_clean executes only when status is clear." + boxes)


def actor_motion_instructions(deployment):
    """Model-facing motion/coordinate section for SO-101, built from the deployment file and its profile.

    Replaces the ARX X5 "Motion planning" and "Coordinates and timing" text. Every
    number below is the value the worker enforces, so prompt and execution agree.
    """
    profile = embodiment(deployment)
    m, w = deployment["motion"], deployment["workspace"]
    arm = profile.arms[0]
    joints = len(profile.arm_joint_names[arm])
    deg = lambda rad: round(math.degrees(rad), 1)
    extra = "".join("\n" + text for text in (_vertical_window_sentence(deployment, profile),
                                                     _arrival_correction_sentence(deployment, profile),
                                                     _jaw_ramp_sentence(deployment, profile),
                                                     _grasp_sentence(deployment, profile),
                                                     _path_sentence(deployment, profile))
                    if text)
    return f"""## Motion planning
You control ONE real SO-101 arm ({joints} revolute joints + a single-jaw gripper). Its only arm id is "{arm}"; any other arm id is rejected.
robot_action is the interface for arm and gripper motion. Default preview solves IK for every move stage without moving
and reports per-stage IK residuals; inspect them, then execute its plan_id. Execution replays the cached joint targets and
never re-solves IK; it is refused if the measured arm drifted more than {deg(deployment['plan']['max_start_drift_rad'])} deg since
preview or the plan is older than {deployment['plan']['max_age_s']:.0f} s. An observe stage pauses with a confirmed hold; resume with
mode='execute' and the same plan_id. Stages run sequentially on one arm.
{_continuation_sentence()}
{_ik_sentence(deployment, profile)}
{_stall_guard_sentence(deployment, profile)}{extra}

## Coordinates and timing
Positions are metres in {profile.frame_id} (robot base: +x forward, +z up). position_m targets describe the TCP {profile.tcp_label}
(on the fixed jaw's inner fingertip face, not between the jaws; see gripper_geometry) with quaternion_xyzw; no extra
offset is applied to them.{_centre_pointer(profile)} Start from the measured ee_pose in robot_state.
{_top_down_text(deployment)}
Accepted target box: x,y,z in [{w['min_m']}, {w['max_m']}] m. Joint speed is capped at
{m['max_joint_speed_rad_s']} rad/s ({deg(m['max_joint_speed_rad_s'])} deg/s) times speed_scale (max {profile.speed_scale.max:g}) and the operator speed cap.
Arrival is verified on the measured pose; defaults are {profile.default_tolerance.position_m * 1000:.0f} mm and
{profile.default_tolerance.orientation_rad} rad unless the stage sets position_tolerance_m / orientation_tolerance_rad. Low-gain servos sag
under gravity; {_sag_clause(profile)}. Expect
millimetre-level residuals, not sub-millimetre. Gripper goal_normalized: 0 closed, 1 open; measured jaw angle is reported, but
jaw closure is not proof of a grasp. There is no force sensing or vertical compliance: approach contact slowly and from above.
This is the real world in real time: objects and people can move while you think. Re-observe before acting on old images.
{_table_line(deployment, w, profile.frame_id)}{clear_view_sentence(deployment, profile)}
{_gripper_text(deployment)}
{_release_text(deployment)}

{grip_guard_text(deployment)}
"""


def clear_view_sentence(deployment, profile=None):
    """The clear-view move target of deployment ``perception.clear_view_q_rad`` ("" when it is null): the TCP pose
    (FK of the packaged URDF, m and quaternion_xyzw in frame_id) of a joint configuration whose silhouette left the
    table visible to the fixed camera when the operator chose it (scripts/so101_clear_view_pose.py). A fact the model
    may use as an ordinary robot_action move target; nothing moves there on its own. Starts with a newline."""
    q = (deployment.get("perception") or {}).get("clear_view_q_rad")
    if not q:
        return ""
    from ...model import so101_kinematics as kinematics
    from .ik_plan import matrix_to_quaternion
    profile = profile or embodiment(deployment)
    T = kinematics.fk(_kinematic_model(), q)
    position = [round(float(v), 4) for v in T[:3, 3]]
    quaternion = [round(float(v), 4) for v in matrix_to_quaternion(T[:3, :3])]
    return (f"\nClear fixed-camera view: a move of the {profile.tcp_label} to position_m {position} quaternion_xyzw "
            f"{quaternion} leaves the table visible to the fixed camera (chosen offline for this calibration; "
            "robot_occlusion in each object_table reports the coverage seen). An ordinary move target: IK may use "
            "other joints; nothing moves there on its own.")


def _centre_pointer(profile):
    """The other move pose input of the Coordinates paragraph: grasp_centre_m (``jaw_grasp_centre``), or ""."""
    if "jaw_grasp_centre" not in profile.pose_inputs:
        return ""
    return (" A move may instead give grasp_centre_m, which the Harness converts to the TCP target with the gripper CAD "
            "(Grasp centre below).")


def _sag_clause(profile):
    """What happens to the servo sag after a move: the bounded arrival correction (``arrival_correction`` tag) or none."""
    if "arrival_correction" in profile.supports:
        return "the worker applies bounded goal corrections after each move (arrival correction, each reported in the receipt)"
    return ("this deployment applies no goal correction (motion.arrival_correction_rounds or "
            "arrival_correction_max_offset_rad is 0), so the arm settles where the servos hold it")


def _continuation_sentence():
    """How a paused or failed SO-101 batch continues (the shared plan_continuation / batch_repair paths)."""
    return ("After a stage fails or pauses, continue from the current measured joints instead of resubmitting completed "
            "stages: preview\nwith source_plan_id and start_from_stage_id (optional stage_overrides, e.g. a new "
            "position_m or tolerance for that stage)\nto recompile that stage and the rest, or plan_id with "
            "skip_failed_stage=true to drop the failed stage.")


def _wrist_roll_limits_rad(deployment):
    """(lower, upper) rad of wrist_roll the planner uses: the URDF limits shrunk by ``motion.joint_limit_margin_rad``
    (the worker's ``SO101Adapter.limits``)."""
    from ...model import so101_kinematics as kin
    lower, upper = kin.joint_limits_rad(kin.load_model())["wrist_roll"]
    margin = deployment["motion"]["joint_limit_margin_rad"]
    return lower + margin, upper - margin


def _top_down_text(deployment):
    """Straight-down quaternion convention and the wrist_roll-limited yaw band (``top_down_roll_facts``: FK of the
    packaged URDF with this deployment's wrist_roll range)."""
    from ...model.so101_gripper_geometry import top_down_roll_facts
    lower, upper = _wrist_roll_limits_rad(deployment)
    facts = top_down_roll_facts((lower, upper))
    centre, half = facts["unreachable_centre_rad"], facts["unreachable_half_width_rad"]
    (px, py), offset = (round(v, 4) + 0.0 for v in facts["pan_axis_xy_m"]), facts["roll_axis_offset_m"]
    sign = "+" if facts["roll_sign"] > 0 else "-"
    wrapped = lambda angle: math.remainder(angle, 2 * math.pi)  # noqa: E731
    direction = (" (TCP +X pointing radially away from the base lies in it)" if abs(wrapped(centre)) <= half else
                 " (TCP +X pointing toward the base lies in it)" if abs(wrapped(centre - math.pi)) <= half else "")
    pair = (" and at most one of them falls in the band" if half < math.pi / 2 else "")
    return (
        "Straight-down poses: quaternion_xyzw [cos(a/2), sin(a/2), 0, 0] points the approach axis (TCP +Z) at world -z\n"
        "with TCP +X (from the jaw gap toward the fixed jaw) at world yaw a (rad about +z from +x; a = 0 is [1, 0, 0, 0]);\n"
        "TCP +Y (wrist-camera side) then points along (sin a, -cos a, 0). With the approach vertical only wrist_roll sets\n"
        f"the yaw: a = phi + {facts['yaw_at_zero_roll_rad']:.4f} rad {sign} wrist_roll (wrapped), phi = azimuth of the "
        f"wrist_roll axis seen from the\nshoulder_pan axis at x {px:.4f}, y {py:.4f} m. The planner's wrist_roll range "
        f"[{lower:.4f}, {upper:.4f}] rad (URDF limits minus\nmotion.joint_limit_margin_rad) leaves a - phi in "
        f"({math.degrees(centre - half):+.1f}, {math.degrees(centre + half):+.1f}) deg unreachable straight down at any "
        f"position{direction}.\nThe TCP is {1000 * offset:.1f} mm off the roll axis, so measured from the TCP's own "
        f"azimuth the band edges shift by up to\natan({offset:.4f} m / r), r = horizontal distance from the shoulder_pan "
        f"axis ({math.degrees(math.atan(offset / 0.2)):.1f} deg at r = 0.2 m). An object's\njaw_line_yaw_deg is an "
        "undirected line: yaw a and a + 180 deg are different grasps on this one-sided jaw (fixed jaw, TCP\nand wrist "
        f"camera on opposite sides, wrist_roll 180 deg apart){pair}.")


def _ik_sentence(deployment, profile):
    """How preview plans a TCP target on this deployment: the packaged local IK, or the cuRobo service (``ik_service``),
    plus the reach query (design: ``ik_plan``)."""
    reach = (f"robot_action mode=reach solves up to {min(profile.max_stages, profile.stages_per_arm * len(profile.arms))} "
             "candidate move poses in one call, each from the "
             "measured joints at the call, without a plan or motion, and answers every one (preview stops at the first "
             "unreachable stage): use it to choose grasp, hover and place poses before preview." if "reach" in profile.supports else "")
    if not deployment.get("ik_service"):
        return ("A 5-DOF arm cannot reach every 6D pose: the approach axis (TCP +Z) tilts only in the arm's vertical plane through\n"
                "shoulder_pan, so a tilted quaternion copied from another position is usually unreachable; the roll about the approach\n"
                "comes from wrist_roll (straight down: see Coordinates). Preview solves the weighted pose first; if that misses either tolerance it\n"
                "re-solves position first (exact position, smallest orientation error) and uses it when orientation_error_rad is within\n"
                "orientation_tolerance_rad (ik.solve=\"position_priority\"). orientation_tolerance_rad >= 3.14159 (pi, or tolerance_preset\n"
                "\"unconstrained_orientation\") solves position only. On planning_failed, ik.position_priority gives the quaternion_xyzw\n"
                "reached at that position and its residuals: reuse it, relax orientation_tolerance_rad, or choose another reachable\n"
                "orientation instead of retrying the same pose." + (" " + reach if reach else ""))
    table = deployment["ik_service"].get("table_top_z_m")
    collision = "self-collision" + (f" and a table box with its top at z={table:g} m" if table is not None else "")
    return (f"IK is NVIDIA cuRobo multi-seed GPU IK (ik.solver=\"curobo\"): it searches many seeds, keeps planned targets free of\n"
            f"{collision} in its sphere model of the arm (the path between targets is checked separately, see Path\n"
            "collision check), and every solution is\n"
            "re-checked with this arm's FK against your tolerances; the valid one with the least joint travel is used. A 5-DOF arm\n"
            "cannot reach every 6D pose: the approach axis (TCP +Z) tilts only in the arm's vertical plane through shoulder_pan,\n"
            "its reachable tilt shrinks with reach and height, and the roll about it comes from wrist_roll. If the full pose misses\n"
            "a tolerance, a position-first solve follows (ik.solve=\"position_priority\"). orientation_tolerance_rad >= 3.14159 (or\n"
            "tolerance_preset \"unconstrained_orientation\") solves position only. approach_tolerance_rad / roll_tolerance_rad bound\n"
            "the approach tilt and the jaw roll separately (rad; e.g. a straight-down quaternion with approach_tolerance_rad 0.35 and\n"
            "roll_tolerance_rad 0.1 accepts up to 20 deg tilt with the jaw kept aligned); stage receipts then add approach_error_rad /\n"
            "roll_error_rad. On planning_failed, ik.position_priority is the quaternion_xyzw reached at that position (reuse it or\n"
            "relax a tolerance), ik.limit_joints names joints of the closest solution at a joint limit (wrist_roll: the jaw yaw\n"
            "is outside the wrist's range, turn it; wrist_flex: tilt the approach or come closer/lower),\n"
            "ik.in_collision_candidates > 0 means reachable solutions exist but collide, and ik.reason\n"
            "ik_service_unreachable means the GPU IK service or its tunnel is down (tell the operator; do not loop)."
            + ("\n" + reach if reach else ""))


def _table_line(deployment, workspace, frame_id=FRAME_ID):
    """Measured table height (touch-off calibration) if the calibration file carries it, and where the fingertips
    meet a support under a straight-down gripper (CAD reach beyond the TCP, ``jaw_reach_facts``)."""
    from pathlib import Path
    from ...model.so101_gripper_geometry import jaw_reach_facts
    table = None
    path = deployment.get("camera_calibration_file")
    try:
        table = json.loads(Path(path).read_text()).get("joint_calibration", {}).get("table_z_m") if path else None
    except (OSError, ValueError):
        table = None
    floor = workspace["min_m"][2]
    g = deployment["gripper"]
    reach = jaw_reach_facts(g["jaw_closed_rad"], g["jaw_open_rad"])
    fixed, peak = reach["fixed_reach_m"], reach["peak_reach_m"]
    fingers = (f"With the approach straight down the fixed fingertip is {1000 * fixed:.1f} mm below the TCP at every "
               f"opening and the moving one up to {1000 * peak:.1f} mm (near goal {reach['peak_goal']:.2f}; CAD)")
    if table is None:
        return (f"Table height is not measured in this deployment; the TCP may not go below z={floor:.3f} m. {fingers}. "
                "Judge support contact from images and receipts.")
    return (f"The table surface is at z={table:.3f} m in {frame_id} (measured by touching it). The TCP may not go below "
            f"z={floor:.3f} m. {fingers}, so the fixed fingertip touches the table at TCP z={table + fixed:.4f} m and the "
            f"moving one at up to TCP z={table + peak:.4f} m. Objects resting on the table have their bottom at "
            f"z={table:.3f} m.")


def _gripper_text(deployment):
    """CAD ``gripper_geometry`` block plus the readings of this deployment's jaw: one name per reading and the
    measured empty close against the CAD closed limit (the full opening is in ``grip_guard_text``)."""
    from ni_rua.robot.model.so101_gripper_geometry import gripper_geometry_text
    g = deployment["gripper"]
    zero, cad_closed = embodiment(deployment).gripper.empty_close_aperture_m, closed_limit_opening_m(deployment)
    empty = (f"an empty goal-0 close reads {_metres(zero)} on this unit (deployment gripper.empty_close_aperture_m; "
             f"CAD at the commanded closed limit {_metres(cad_closed)})" if zero is not None else
             f"an empty goal-0 close reads at least the CAD closed-limit opening {_metres(cad_closed)} (not measured "
             "in this deployment)")
    return (gripper_geometry_text(g["jaw_closed_rad"], g["jaw_open_rad"]) + "\n"
            "- Readings of this unit: gripper_execution.measured_opening_m (gripper results) and measured_aperture_m "
            "(grip_guard, grip_facts) are one reading, the CAD opening of the measured jaw angle after it settles; a jaw "
            f"stopped by an object reads about the object's width; {empty}; goal 1: see Grip guard."
            + _hold_squeeze_text(deployment))


def squeeze_fingertip_travel_m(squeeze_rad, opening_m):
    """Change of the CAD fingertip gap (m) when the jaw joint closes by ``squeeze_rad`` (rad) from the jaw angle whose
    fingertip gap is ``opening_m`` (m): how far the fingertips would travel if nothing resisted the squeeze
    (``so101_gripper_geometry.opening_width_m``). None when ``opening_m`` is outside the CAD jaw range.
    Example: ``squeeze_fingertip_travel_m(0.01, 0.029)`` -> about ``0.0008`` (0.8 mm)."""
    from ...model.so101_gripper_geometry import jaw_rad_for_opening, opening_width_m
    jaw = jaw_rad_for_opening(opening_m)
    return None if jaw is None else float(opening_width_m(jaw) - opening_width_m(jaw - squeeze_rad))


def hold_squeeze_capability(deployment):
    """``{"hold_squeeze": {...}}`` for ``robot_capabilities.gripper_semantics``, or ``{}`` when the deployment applies
    none: ``default_rad`` / ``max_rad`` (rad of the jaw joint; the default and the range of ``grip_squeeze_rad``) and
    ``default_fingertip_travel_m`` (m: ``squeeze_fingertip_travel_m`` of the default at the verified 0.029 m opening)."""
    limits = hold_squeeze(deployment)
    if limits is None:
        return {}
    travel = squeeze_fingertip_travel_m(limits[0], VERIFIED_OBJECT_READING_M[0])
    return {"hold_squeeze": {"field": GRIP_SQUEEZE_FIELD, "default_rad": limits[0], "max_rad": limits[1],
                             "default_fingertip_travel_m": None if travel is None else round(travel, 5),
                             "applies": "closing jaw command settled short of its goal by more than the squeeze"}}


def _hold_squeeze_text(deployment):
    """The holding-squeeze line of the gripper section ("" when the deployment applies none): what the worker does
    (``SO101Adapter._hold_squeeze``), the default in rad and in mm of fingertip travel, and where it is reported."""
    limits = hold_squeeze(deployment)
    if limits is None:
        return ""
    default, maximum = limits
    width = VERIFIED_OBJECT_READING_M[0]
    travel = squeeze_fingertip_travel_m(default, width)
    return ("\n- Holding squeeze: a closing jaw that settles on contact (short of its goal by more than the squeeze) then "
            "holds at the measured jaw angle minus the squeeze instead of pushing at the servo limit. Default "
            f"{default:g} rad of the jaw joint (about {1000 * travel:.1f} mm of fingertip gap near a {1000 * width:g} mm "
            f"opening); grip_squeeze_rad on the jaw stage sets it in [0, {maximum:g}] rad. gripper_execution.hold_squeeze "
            "reports it; measured_opening_m, grip_facts and grip_guard are read after it. A later move without a jaw "
            "command reports grip_hold.squeeze_remaining_rad: about the squeeze while a rigid object resists; if the "
            "object slips out or compresses, an unloaded STS3215 jaw closes only down to its static-friction deadband "
            "above hold_goal (about 0.03-0.04 rad at P=16, not 0—so a squeeze inside ~0.04 rad neither holds firmly nor "
            "drops on object loss; verify retention from camera images and grip_facts.aperture_delta_m).")


def _release_text(deployment):
    """Release and placement facts of the one-sided jaw (CAD reach in goal units, ``jaw_reach_facts``); no procedure."""
    from ...model.so101_gripper_geometry import jaw_reach_facts
    g = deployment["gripper"]
    reach = jaw_reach_facts(g["jaw_closed_rad"], g["jaw_open_rad"])
    swing = (f"Opening from below goal {reach['peak_goal']:.2f} first swings the moving fingertip further beyond the "
             f"TCP ({1000 * reach['moving_reach_at_goal0_m']:.1f} mm at goal 0, {1000 * reach['peak_reach_m']:.1f} mm at "
             f"goal {reach['peak_goal']:.2f}), i.e. toward a support under a downward gripper, before it "
             + (f"retracts behind the TCP plane above goal {reach['behind_tcp_plane_above_goal']:.2f}."
                if reach["behind_tcp_plane_above_goal"] is not None else "retracts."))
    return ("Release and placement (geometry facts): opening moves only the -X jaw, so the fixed jaw's inner face stays "
            "on the held object's\n+X face. Right after opening, a lift slides that face along the object (friction can "
            "lift, drag or tip a light object)\nand a move toward TCP -X pushes the object; a move toward TCP +X separates "
            f"the fixed jaw from it first. {swing}\nA held object that does not slip keeps its grasp offset: held straight "
            "down, its bottom stays (TCP z - bottom z, both at the\ngrasp) below the TCP, so it reaches a surface at "
            "height h when the TCP is at h plus that offset.")


def _metres(value):
    return f"{value:.4f} m"


def _schema_presets(profile):
    """Bounded grip-guard presets of ``profile`` (its preset margins besides ``none``), in profile order."""
    return tuple(name for name in profile.gripper.preset_margins_m if name != "none")


def recheck_min_object_width_m(deployment):
    """Narrowest object (m) whose slip-out the post-grasp re-check can see on this jaw.

    The re-check needs the jaw to close by ``SLIP_MIN_JAW_DROP`` (0.08 normalized) down to the
    empty-close gap, so the object must be at least the opening reached 0.08 normalized above the
    empty-close jaw angle (about 0.017 m with a 0.0051 m zero point; the jaw drop alone is about
    0.012 m there, against 0.007 m on ARX-X5). None when the zero point is unmeasured.
    """
    from ...model.so101_gripper_geometry import jaw_rad_for_opening, opening_width_m
    from ...motion.grip_guard import SLIP_MIN_JAW_DROP
    g, zero = deployment["gripper"], embodiment(deployment).gripper.empty_close_aperture_m
    if zero is None:
        return None
    start = jaw_rad_for_opening(zero)
    start = g["jaw_closed_rad"] if start is None else start
    return round(opening_width_m(start + SLIP_MIN_JAW_DROP * (g["jaw_open_rad"] - g["jaw_closed_rad"])), 4)


def grip_guard_capability(deployment):
    """Model-visible grip-guard facts for ``robot_capabilities``; the numbers the worker enforces (m)."""
    from ...motion.grip_guard import (CLOSING_JAW_GOAL_BELOW, BATCH_DEFAULT_JAW_GOAL_BELOW, DOWNWARD_MOVE_DZ_M,
                                      SLIP_MAX_GAP_ABOVE_EMPTY_CLOSE_M, SLIP_MIN_JAW_DROP, preset_aperture_bounds)
    profile = embodiment(deployment)
    gripper = profile.gripper
    zero = gripper.empty_close_aperture_m
    presets = None if zero is None else {
        name: dict(zip(("min_aperture_m", "max_aperture_m"),
                       (None if v is None else round(v, 4) for v in preset_aperture_bounds(name, profile=profile))))
        for name in _schema_presets(profile)}
    recheck = ({"available": True, "closing_goal_below": CLOSING_JAW_GOAL_BELOW, "min_jaw_drop_normalized": SLIP_MIN_JAW_DROP,
                "min_object_width_m": recheck_min_object_width_m(deployment),
                "min_jaw_drop_m_near_empty_close": round(recheck_min_object_width_m(deployment) - zero, 4),
                "max_gap_above_empty_close_m": SLIP_MAX_GAP_ABOVE_EMPTY_CLOSE_M,
                "scope": "first move without a jaw command after a passing guarded close; skipped on moves ending more "
                         f"than {-DOWNWARD_MOVE_DZ_M:g} m lower"}
               if post_grasp_recheck_available(deployment) else
               {"available": False, "reason": post_grasp_recheck_unavailable_reason(deployment)})
    return {"available": True, "fields": [*GRIP_GUARD_FIELDS, "checkpoint_after_gripper"], "unit": "m",
            "measured_aperture": "fingertip gap: CAD opening of the measured jaw angle after the jaw settles",
            "reading_accuracy": (f"verified at one point: a {VERIFIED_OBJECT_READING_M[0]} m cube read {VERIFIED_OBJECT_READING_M[1]} m; "
                                 "near closure the reading is offset by the empty-close gap (a true gap of 0 reads that "
                                 "gap); between them unverified. Below about 0.010 m prefer a preset or "
                                 "checkpoint_after_gripper over a bound taken from the object width"),
            "empty_close_spread_m": list(MEASURED_EMPTY_CLOSE_RANGE_M),
            "empty_close_aperture_m": zero, "closed_limit_cad_opening_m": closed_limit_opening_m(deployment),
            "full_open_aperture_m": gripper.full_open_aperture_m,
            "full_open_source": gripper.full_open_source, "presets_available": zero is not None,
            "presets": presets, "explicit_bounds_m": [0.0, gripper.full_open_aperture_m],
            "batch_default_applies_below_goal_normalized": BATCH_DEFAULT_JAW_GOAL_BELOW,
            "preview_fact": "stage_plans[<jaw stage>].grip_guard.detects_empty_close: true, false (an empty close can "
                            "never fire this guard) or null (unknown: empty-close gap not measured)",
            "on_violation": "pause before the next stage with a confirmed hold: status stopped, reason "
                            "observation_checkpoint, checkpoint_reason aperture_below_min_m | aperture_above_max_m | "
                            "jaw_not_settled (jaw still moving after gripper.settle_timeout_s "
                            f"{deployment['gripper']['settle_timeout_s']} s)",
            "post_grasp_recheck": recheck}


def grip_guard_text(deployment):
    """Model-facing grip-guard section built from the deployment's profile (zero point, full opening, preset margins)."""
    from ...motion.grip_guard import (BATCH_DEFAULT_JAW_GOAL_BELOW, DOWNWARD_MOVE_DZ_M, SLIP_MAX_GAP_ABOVE_EMPTY_CLOSE_M,
                                      SLIP_MIN_JAW_DROP, preset_aperture_bounds)
    from ...model.so101_gripper_geometry import opening_width_m
    profile = embodiment(deployment)
    gripper = profile.gripper
    zero, full = gripper.empty_close_aperture_m, gripper.full_open_aperture_m
    full_source_measured = deployment["gripper"]["full_open_aperture_m"] is not None
    lines = [
        "## Grip guard (empty-grasp check)",
        "After a jaw stage the worker measures the fingertip gap measured_aperture_m (m): the CAD opening of the measured",
        f"jaw angle once it settles. It is verified at one point only: a {VERIFIED_OBJECT_READING_M[0]} m cube read",
        f"{VERIFIED_OBJECT_READING_M[1]} m. Near closure it is offset by the empty-close gap (a true gap of 0 reads that gap,",
        "not 0); in between it is not verified, so for objects under about 0.010 m prefer a preset or",
        f"checkpoint_after_gripper over a bound taken from the object width. Goal 1 reads {_metres(full)} ({gripper.full_open_source}"
        + (f"; CAD {_metres(opening_width_m(deployment['gripper']['jaw_open_rad']))}" if full_source_measured else "") + ").",
    ]
    if zero is None:
        lines += [
            "grip_guard_preset is unavailable here: the deployment has no measured empty-close gap",
            "(gripper.empty_close_aperture_m). Explicit min_aperture_m / max_aperture_m still work, but an empty close never",
            f"reads 0 on this jaw: it reads at least the CAD closed-limit opening {_metres(closed_limit_opening_m(deployment))}, and measured",
            f"SO-101 empty closes read more ({MEASURED_EMPTY_CLOSE_RANGE_M[0]}-{MEASURED_EMPTY_CLOSE_RANGE_M[1]} m on the unit where this was measured). A min_aperture_m at or below",
            "the empty-close reading never fires; preview stage_plans[<jaw stage>].grip_guard.detects_empty_close says false",
            "for that, null when it cannot be known here.",
        ]
    else:
        bounds = ", ".join(f"{name} >= {_metres(preset_aperture_bounds(name, profile=profile)[0])}"
                           for name in _schema_presets(profile))
        spread = MEASURED_EMPTY_CLOSE_RANGE_M[1] - MEASURED_EMPTY_CLOSE_RANGE_M[0]
        lines += [
            f"An EMPTY full close on this unit reads {_metres(zero)} (gripper.empty_close_aperture_m), not 0, so each preset's",
            f"minimum gap is that value plus the preset margin: {bounds}; none disables.",
        ]
        narrow = [(name, low) for name in _schema_presets(profile)
                  if (low := gripper.preset_margins_m[name][0]) is not None and low < spread]
        if narrow:
            names = " and ".join(f"{name}'s" for name, _ in narrow)
            margins = " / ".join(f"{1000 * low:g} mm" for _, low in narrow)
            lines += [
                f"Empty closes measured on this gripper spread {MEASURED_EMPTY_CLOSE_RANGE_M[0]}-{MEASURED_EMPTY_CLOSE_RANGE_M[1]} m"
                f" ({1000 * spread:.1f} mm), more than {names}",
                f"{margins} margin: {' and '.join(name for name, _ in narrow)} cannot tell a very thin object from an "
                "empty close here.",
            ]
    lines += ([
        "Choose the guard from the object: for a rigid object of known width, a bounded preset (rigid_object, wide_object,",
        "non_empty) or a min_aperture_m below that width lets an empty close pause while a real grasp continues into the",
        "lift; for thin or deformable objects, whose valid pinch reads near the empty-close gap, use thin_object, none or",
        "checkpoint_after_gripper."] if zero is not None else [
        "Choose the guard from the object: for a rigid object of known width, a min_aperture_m below that width (and above",
        "the empty-close reading) lets an empty close pause while a real grasp continues into the lift; for thin or",
        "deformable objects, whose valid pinch reads near the empty-close gap, omit it or use checkpoint_after_gripper."])
    lines += [
        "Set grip_guard_preset and/or min_aperture_m / max_aperture_m on a gripper stage, on a move stage together with",
        "gripper_goal_normalized, or on the whole batch (then it applies to each jaw stage with goal_normalized below",
        f"{BATCH_DEFAULT_JAW_GOAL_BELOW} that has none of its own). Explicit min_aperture_m / max_aperture_m are absolute gaps in",
        f"[0, {full:.4f}] m and replace the preset's bound. When the jaw closed during the stage, later stages remain and the",
        "gap is outside [min, max], the batch pauses before the next stage with a confirmed hold: status \"stopped\",",
        "reason \"observation_checkpoint\", checkpoint_reason \"aperture_below_min_m\" or \"aperture_above_max_m\"; the jaw",
        "stage result has grip_guard {min_aperture_m, max_aperture_m, measured_aperture_m} and grip_facts. Check the images:",
        "resume the same plan_id (mode='execute') only for a real grasp, otherwise re-plan. If the jaw is still moving when",
        f"gripper.settle_timeout_s ({deployment['gripper']['settle_timeout_s']} s) expires, a guarded jaw stage pauses the same way with",
        "checkpoint_reason \"jaw_not_settled\" and grip_guard.jaw_settled=false: that reading is not final.",
        *(["After a guarded close passes, the next move without a jaw command re-checks once (not on a move ending more than",
           f"{-1000 * DOWNWARD_MOVE_DZ_M:g} mm lower): if the jaw closed by >= {SLIP_MIN_JAW_DROP} normalized during that move and ends within",
           f"{SLIP_MAX_GAP_ABOVE_EMPTY_CLOSE_M} m of the empty-close gap (object slipped out), it pauses the same way. On this jaw",
           f"{SLIP_MIN_JAW_DROP} normalized is a gap drop of about {recheck_min_object_width_m(deployment) - zero:.3f} m, so an object narrower than about",
           f"{recheck_min_object_width_m(deployment):.3f} m can slip out unnoticed."]
          if post_grasp_recheck_available(deployment) else
          ["There is no post-grasp slip re-check here until gripper.empty_close_aperture_m is measured: an object that slips",
           "out during a later move is not caught automatically."]
          if zero is None else
          ["There is no post-grasp slip re-check with the holding squeeze: a jaw that loses its object closes toward its",
           "hold goal (stopping at the servo's ~0.03-0.04 rad static-friction deadband above hold_goal, not at the",
           "empty-close gap). After a lift check the camera images together with grip_facts.aperture_delta_m and",
           "grip_hold.squeeze_remaining_rad (which drops only when the squeeze exceeds that deadband)."]),
        "checkpoint_after_gripper=true (gripper stage, move with",
        "gripper_goal_normalized, or the batch for closing jaw stages) always pauses after the jaw stage, before the next",
        "dependent move.",
    ]
    return "\n".join(lines)
