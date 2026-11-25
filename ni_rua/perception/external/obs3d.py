"""features.object_table_observe / features.scene3d_observe (both on by default with external_perception): 3D scene information as part of
every model observation reply instead of an on-demand geometry_query kind.

With a switch on, a full head-camera observation in a reply to the model (not to a program job) gets, after the
observation metadata and camera images, one extra JSON text part and at most one image. Replies covered:

* robot_observe without camera_id (a full observation, not an ROI view): text part + overlay image;
* the final observation of robot_action (mode execute / direct) and robot_wait replies: text part only; the overlay
  image is added there too only with features.obs3d_action_overlay (default false). Temporal wait samples get nothing.

Parts:

* object_table_observe: the object table of the head frame, computed exactly as geometry_query kind object_table
  (same service call POST /object_table, same rows: ids o1..oN, centre, top/bottom z, length, width, yaw, tilt,
  support, quality; no names, no truth), limited to features.obs3d_max_objects rows, and one overlay image of the head
  frame with the ids drawn (same size as the camera frame). Wrist cameras get nothing here; on an embodiment with
  the ``cross_view`` support tag (SO-101) the on-demand geometry_query kind object_table with target_camera_ids
  checks the rows against a second camera (object_table.cross_view_rows), and the developer instructions say so.
* scene3d_observe: the VGGT point cloud of the head frame saved as a scene3d_points artifact, exactly as
  geometry_query kind scene3d (POST /points); the reply carries only the artifact handle, the point count, table_z and
  a one-line pointer to scene3d_helpers (copied into program_run workspaces while the switch is on). No table, no image.

One service call per kind and observation: /object_table first, then /points (the service reuses its per-image
reconstruction). A successful result is kept for the most recent observation, so a robot_observe of an observation
that an action reply already covered reuses it (same rows, same artifact, no new call). The /height path is
unchanged. A service error or timeout (features.obs3d_timeout_s) never fails the observation or the action reply: the
part then says "unavailable: <reason>" in one field. Everything lives inside the tool result, so the request
projection (lean_projection block compression, old-result summaries, image placeholders) treats it like any other
tool output. The stored ObservationBundle is unchanged; per-observation diagnostics are written as event
geometry.obs3d_attached.

With analysis.scene_truth_log on (analysis only, off by default), every attached object table (also a reused one,
and an unavailable one) is additionally written as internal event analysis.obs3d_object_table: observation id, head
frame id, reply kind, reused flag and the rows exactly as attached. It holds the model-visible estimate only (no
truth), is never returned to the model, and is read offline by scripts/score_geometry_truth.py (kind
object_table_obs). With the switch off nothing is written.

With both switches off only flags() is consulted: tool definitions and replies are unchanged.

Embodiment (design 13 section 5.3): the attached camera, the note's camera and frame, the catalogue sentence and the
developer instructions are rendered from the run's EmbodimentProfile (perception.external.embodiment_of; None = the
default ARX-X5 robot, whose rendering is the historical text). "Head camera" in this docstring is the profile's
overview camera (ARX-X5 cam_head, SO-101 fixed).
"""
from __future__ import annotations

from copy import deepcopy
import threading
import time

_monotonic = time.monotonic

from ...core.enrichment import Enrichment, EnrichmentRequest, ObservationEnricher
from ...core.episode_state import (
    StateSlot,
    episode_state_of,
    slot_get,
    slot_peek,
    slot_reset,
    slot_set,
)
from . import embodiment_of

OBS3D_MEMO_SLOT: StateSlot = StateSlot("obs3d_memo", lambda: None, scope="episode")
OBS3D_HISTORY_SLOT: StateSlot[dict] = StateSlot("obs3d_history", dict, scope="episode")
PENDING_OBSERVE_OBS3D_SLOT: StateSlot = StateSlot("pending_observe_obs3d", lambda: None, scope="episode")
PENDING_ACTION_OBS3D_SLOT: StateSlot = StateSlot("pending_action_obs3d", lambda: None, scope="episode")
VIRTUAL_FIELD_SLOT: StateSlot = StateSlot("virtual_spatial_field", lambda: None, scope="episode")
VIRTUAL_FIELD_ERROR_SLOT: StateSlot = StateSlot("virtual_field_last_error", lambda: None, scope="episode")
LAST_OBS3D_DELTA_SLOT: StateSlot = StateSlot("last_obs3d_delta", lambda: None, scope="episode")
SCENE3D_USE_EPOCH_SLOT: StateSlot = StateSlot("obs3d_scene3d_use_epoch", lambda: None, scope="episode")

_OBJECT_TABLE_NOTE = ("{camera} camera, VGGT monocular-depth estimates in {frame_id} m (not measurements); "
                      "{id_scope}{overlay}; low:<flags> = implausible depth")
_NOTE_IDS_PER_OBS = "ids are per observation"
_NOTE_IDS_PERSISTENT = "ids are persistent episode tracks (see track_meta)"
_NOTE_OVERLAY = " and drawn on the next image"   # only in a reply that carries the id overlay image
SCENE3D_USE = "program_run: import scene3d_helpers as s3; s3.load(artifact); print(s3.API) lists the queries"
_CATALOGUE_PREFIX = " {camera}-camera observations here and in robot_action / robot_wait replies also carry the frame's "
# Default-robot (ARX-X5) renderings under their historical names; the code renders per profile.
CAMERA = "cam_head"
CATALOGUE_PREFIX = _CATALOGUE_PREFIX.format(camera="Head")
CATALOGUE_TABLE = "VGGT-estimated object table (ids o1..oN: centre, top/bottom z, footprint, tilt, support, quality)"
CATALOGUE_CLOUD = "VGGT point cloud as a saved artifact (handle, point count, table_z) for scene3d_helpers in program_run."
CATALOGUE_BOTH = " plus a point-cloud artifact handle for scene3d_helpers"
CATALOGUE_OVERLAY = ("; robot_observe adds an id overlay image.", ", with an id overlay image.")
ANALYSIS_EVENT = "analysis.obs3d_object_table"


def flags(runtime):
    cfg = getattr(runtime, "config", runtime)
    features = getattr(cfg, "features", cfg)
    return (bool(getattr(features, "object_table_observe", False)),
            bool(getattr(features, "scene3d_observe", False)))


def action_overlay(runtime):
    cfg = getattr(runtime, "config", runtime)
    features = getattr(cfg, "features", cfg)
    return bool(getattr(features, "obs3d_action_overlay", False))


def _camera_label(camera_id):
    """Camera name used in text: the id without a ``cam_`` prefix (cam_head -> head, fixed -> fixed)."""
    return str(camera_id).removeprefix("cam_")


def object_table_note(profile=None, camera_id=None, *, overlay=False, persistent_ids=False):
    """``note`` of an attached object table: the attached camera (``camera_id``, default the profile's
    ``overview_camera``) and the frame of every value (``profile.frame_id``). None = the default ARX-X5 robot.
    ``overlay``: True only when the id overlay image follows the table in the same reply (default False).
    ``persistent_ids``: True when ``track_meta.persistent_ids`` is active for this table (default False)."""
    profile = profile if profile is not None else embodiment_of(None)
    camera = camera_id or profile.overview_camera
    return _OBJECT_TABLE_NOTE.format(
        camera=_camera_label(camera) if camera else "overview",
        frame_id=profile.frame_id,
        id_scope=_NOTE_IDS_PERSISTENT if persistent_ids else _NOTE_IDS_PER_OBS,
        overlay=_NOTE_OVERLAY if overlay else "",
    )


# Default robot's rendering in a reply with the overlay image (robot_observe).
OBJECT_TABLE_NOTE = object_table_note(None, "cam_head", overlay=True)


def catalogue_sentence(object_table=False, scene3d=False, action_overlay=False, profile=None):
    """Text appended to the robot_observe description ('' when both switches are off); ``profile`` = the run's
    EmbodimentProfile (None = the default ARX-X5 robot) names the overview camera."""
    if not (object_table or scene3d):
        return ""
    profile = profile if profile is not None else embodiment_of(None)
    label = _camera_label(profile.overview_camera or "overview")
    prefix = _CATALOGUE_PREFIX.format(camera=label[:1].upper() + label[1:])
    if not object_table:
        return prefix + CATALOGUE_CLOUD
    return (prefix + CATALOGUE_TABLE + (CATALOGUE_BOTH if scene3d else "")
            + CATALOGUE_OVERLAY[bool(action_overlay)])


def _mm(value_m):
    """Millimetres as %g (0.088 -> '88', 0.0075 -> '7.5')."""
    return "%g" % round(float(value_m) * 1000.0, 3)


def _object_table_observe_text(profile, *, persistent_tracking=False):
    """The object_table_observe sentence for ``profile`` (grasp-hint fields as object_table.grasp_hints_from emits)."""
    gripper = profile.gripper
    label = _camera_label(profile.overview_camera or "overview")
    id_desc = (
        "persistent episode `track_id`s o1..oN with `track_meta`"
        if persistent_tracking
        else "ids o1..oN"
    )
    text = (label[:1].upper() + label[1:] + "-camera observations in robot_observe and the final observation of "
            f"robot_action / robot_wait automatically attach a VGGT 3D object_table ({id_desc}: centre_m, top_z, "
            "bottom_z, length, width, yaw_deg in [-90,90), tilt_deg, support, quality, label_px; robot_observe also attaches an "
            "id overlay image) plus inline `grasp_hints` per object (")
    angles = "grasp_angles" in profile.pose_inputs
    jaw_grasp = gripper.jaw_grasp if "jaw_grasp_centre" in profile.pose_inputs else None
    if jaw_grasp is not None:   # the commanded grasp_centre_m is the object centre between the two jaws (SO-101)
        text += _jaw_grasp_hints_text(profile, jaw_grasp)
    elif angles:   # the commanded grasp_position_m is the pad centre (ARX-X5)
        text += ("`pad_z_m`: the object's centre z, the pad height that puts the whole "
                 + _mm(2 * gripper.fingertip_beyond_reference_m) + " mm gripping face on the object; "
                 "`fingertip_support_clearance_m`: how far the fingertips, " + _mm(gripper.fingertip_beyond_reference_m)
                 + " mm below the pad centre, stay above the support at that height; negative for thin objects, where "
                 "a contact_stop descent touches the support and stops there, which is normal, with the pads as low as "
                 "the object allows; ")
    else:        # the commanded position_m is the TCP link
        text += ("`tcp_z_m`: the object's centre z, the " + profile.tcp_label + " height level with the object's "
                 "centre; `fingertip_support_clearance_m`: how far the fingertips, up to "
                 + _mm(gripper.fingertip_beyond_reference_m) + " mm beyond " + profile.tcp_label + " along the "
                 "approach axis, stay above the support at that height with the approach axis pointing down; negative "
                 "for thin objects, where the fingertips reach the support first; ")
    if jaw_grasp is None:
        text += "`opening_check` against the " + _mm(gripper.full_open_aperture_m) + " mm max opening, `elongated` flag"
    if jaw_grasp is not None:
        pass    # _jaw_grasp_hints_text already named opening_check, elongated and jaw_lines
    elif angles:
        text += ("; for horizontally elongated graspable objects, `across_yaw_deg` gives the two angles_deg yaws "
                 "closing perpendicular to the long axis `yaw_deg +/- 90`, and for lying elongated objects "
                 "`stand_upright_angles_deg` gives the 3D `[pitch,roll,yaw]` release orientation to stand the object "
                 "upright")
    else:
        text += ("; for horizontally elongated graspable objects, `jaw_line_yaw_deg` gives the horizontal line across "
                 "the long axis (`yaw_deg + 90` in [-90,90), degrees about " + profile.frame_id + " +z from +x), along "
                 "which the fingertips close to span the narrow side; it is a direction in " + profile.frame_id
                 + ", not a tool angle")
    text += ("). For `too_wide` objects/fixtures or a grasp on a part of an object, choose the grasp centre, width "
             "and orientation " if jaw_grasp is not None else
             "). For compact (`elongated:false`) or `too_wide` objects/fixtures, choose the grasp point and orientation ")
    text += ("directly from the RGB image (such as a handle, rim, prongs, or slot) and execute complete pick-and-place "
             "sequences in a single continuous batch relying on `grip_guard` rather than splitting batches or inserting "
             "redundant mid-batch `:observe` stages. ")
    # grasp_3d_review audits angles_deg grasp poses; contact_block_summary comes with the collision-proxy executor.
    text += ("`robot_action(mode='preview')` automatically returns `grasp_3d_review`, and after execution"
             if angles else
             "`robot_action(mode='preview')` returns `grasp_3d_review` for grasp_centre_m stages after which the jaw "
             "closes (jaw opening against the width, grasp height, fixed jaw against the object, jaw line against a long "
             "axis; advisory, computed from these estimates), and after execution" if jaw_grasp is not None else
             "After execution")
    if "review_proxies" in profile.supports:
        text += (" the reply reports `object_state_delta` and any contact stop's `contact_object_3d` in "
                 "`contact_block_summary` / `episode_ledger`.")
    else:
        text += (" the reply reports `object_state_delta`; a stopped or failed batch's `episode_ledger` can attribute "
                 "the stop position to a table object (`contact_object_3d`).")
    return text


def _jaw_grasp_hints_text(profile, jaw_grasp):
    """The grasp-hint part of the object_table_observe sentence for a profile with the ``jaw_grasp_centre`` pose
    input: the fields ``object_table.grasp_hints_from`` emits there, as inputs of a robot_action grasp_centre_m move."""
    gripper = profile.gripper
    return ("`grasp_centre_z_m`: the object's centre z, the height of a grasp_centre_m level with it; "
            "`fingertip_support_clearance_m`: how far the fingertips, up to " + _mm(gripper.fingertip_beyond_reference_m)
            + " mm beyond " + profile.tcp_label + " along the approach axis, stay above the support at that height "
            "with the approach pointing down and grasp_depth_m 0; negative for thin objects, where the fingertips reach "
            "the support first; `opening_check` of the row width: `ok`, `tight` (within 10 mm of the limit), "
            "`beyond_parallel_range` (wider than " + _mm(jaw_grasp.tcp_plane_gap_m) + " mm: the moving fingertip has "
            "swung behind the " + profile.tcp_label + " origin plane, so grasp deeper with grasp_depth_m) or `too_wide` "
            "(wider than the " + _mm(jaw_grasp.max_gap_m) + " mm widest jaw gap); `elongated` flag; `jaw_lines`: for "
            "each face axis of the object's box, first across its long axis, the two `fixed_jaw_yaw_deg` (the same "
            "line with the fixed jaw on either side, degrees about " + profile.frame_id + " +z from +x) and the "
            "`object_width_m` the object spans along that line. A row's `centre_m` with one `jaw_lines` entry are the "
            "inputs of a straight-down robot_action move (`grasp_centre_m`, `object_width_m`, `fixed_jaw_yaw_deg`); "
            "they describe the object's estimated box, not what the arm reaches: check candidates with mode='reach'")


def developer_instructions(features, profile=None) -> str:
    """Developer-instruction summary for active 3D perception switches; '' when all are off.

    ``profile``: the run's EmbodimentProfile (None = the default ARX-X5 robot, the historical text): overview
    camera, grasp-hint fields and numbers (gripper opening, fingertip reach), wrist cameras and the stage features
    the text mentions (``obstacle_object_ids``, ``review_proxies``, ``grasp_angles``) are read from it."""
    if features is None:
        return ""
    from .scene3d import ik_prefilter_staged

    profile = profile if profile is not None else embodiment_of(None)
    parts = []
    persistent_tracking = bool(getattr(features, "persistent_object_tracking", False))
    angles = "grasp_angles" in profile.pose_inputs
    ik_staged = ik_prefilter_staged(profile)
    dual_arm = len(profile.arms) > 1
    has_obstacles = "obstacle_object_ids" in profile.supports
    if getattr(features, "object_table_observe", False):
        if angles and ik_staged:
            id_desc = (
                "persistent episode `track_id`s o1..oN with `track_meta`"
                if persistent_tracking
                else "per-observation ids o1..oN"
            )
            gripper = profile.gripper
            cam_label = _camera_label(profile.overview_camera or "overview")
            cam_title = cam_label[:1].upper() + cam_label[1:]
            reach_clause = (
                "check single-arm source-to-destination reachability before picking, "
                if dual_arm
                else ""
            )
            obstacle_clause = (
                " (pass `obstacle_object_ids` only for non-target intermediate objects to route over, never for the target object "
                "or hollow racks/slots/containers being entered)"
                if has_obstacles
                else ""
            )
            parts.append(
                f"{cam_title}-camera observations in robot_observe and the final observation of robot_action / robot_wait "
                f"automatically attach a VGGT 3D object_table ({id_desc}: "
                "centre_m, top_z, bottom_z, length, width, yaw_deg in [-90,90), tilt_deg, support, quality; "
                "robot_observe also attaches an id overlay image) "
                f"plus inline `grasp_hints` per object (`pad_z_m`: centre z that seats the {_mm(2 * gripper.fingertip_beyond_reference_m)} mm gripping face on the object; "
                f"`fingertip_support_clearance_m`: clearance of fingertips {_mm(gripper.fingertip_beyond_reference_m)} mm below `pad_z_m` above support — when <= 0.010 m "
                "or negative on thin/surface-resting parts, descend to `pad_z_m` with `stall_guard_preset='contact_stop'` or "
                "`tolerance_preset='compliant'` so support contact stops cleanly instead of aborting with tracking_error; "
                f"`opening_check` against {_mm(gripper.full_open_aperture_m)} mm max opening, `elongated` flag, optional `cavity_depth_m` in m when >= 0.004 m; "
                "for horizontally elongated graspable objects, `across_yaw_deg` gives both 180-deg symmetric yaws `yaw_deg +/- 90` deg "
                "— `allow_jaw_symmetry=true` is the default on open-jaw grasp moves so exact 180-deg parallel-jaw symmetry resolves automatically "
                "(opt into `allow_approach_tilt=true` when approach tilts are desired) while preserving closed-jaw/holding rigidity, "
                "and `stand_upright_angles_deg` gives `[pitch,roll,yaw]` deg to stand a lying elongated object upright). "
                "For compact (`elongated:false`) or `too_wide` objects/fixtures, choose the grasp point and 3D orientation directly "
                f"from the RGB views (handle, rim, prongs, or slot), {reach_clause}"
                "execute clear open-area batches in one turn (`mode='execute_if_clean'` or `mode='direct'`) with conditional "
                "`grip_guard_preset`, and stage closed-loop visual verification (`robot_observe(focus_3d=...)`, wrist RGB, two-view "
                "`triangulate`, or a pre-entry hover `observe` / `checkpoint_after_gripper=true`) before sub-centimetre contact"
                f"{obstacle_clause}. "
                "Post-action replies report `object_state_delta` and any contact stop's `contact_object_3d`."
            )
        else:
            parts.append(_object_table_observe_text(profile, persistent_tracking=persistent_tracking))
    if getattr(features, "object_table", False):
        wrist = list(profile.wrist_cameras.values())
        reanchor_clause = (
            "; to re-anchor low-confidence persistent tracks, call "
            "request={'kind':'reanchor_tracks', 'observation_id':..., 'target_id':'o3', "
            "'phrases':['...'], 'pos_pixels':[[u,v]], 'neg_pixels':[[u,v]]}"
            if persistent_tracking
            else ""
        )
        parts.append(
            "For custom segmentation (including thin/flat parts and containers omitted from the default table) "
            + ("or wrist-camera projection, " if wrist else "")
            + "call geometry_query with "
            "request={'kind':'object_table', 'observation_id':..., 'labels':['noun phrase',...], "
            "'pixels':[[u,v],...]"
            + (", 'target_camera_ids':[" + ",".join("'%s'" % camera for camera in wrist) + "]" if wrist else "")
            + f"}}{reanchor_clause}."
            + (" Table rows are single-view monocular estimates. With target_camera_ids the result also carries "
               "`cross_view`: each row paired with the same object in the target camera's own table "
               "(`target_view_centre_m`, `target_view_top_z`, `target_view_minus_row_m`) and triangulated from the "
               "two views "
               "(`stereo_centre_m`, `stereo_minus_row_m`, `ray_gap_m`, `parallax_deg`); no row is replaced, so "
               "compare the estimates before relying on a row for contact-accuracy positioning."
               if wrist and "cross_view" in profile.supports else "")
            + (" Paired rows standing on the table, or on rows that do, also get `contact_top_z` (m): the top of a "
               "prism standing on the table fitted to both views' outlines, independent of monocular depth; "
               "`contact_top_z_range` is that fit with the outlines 1 px larger and smaller (calibration error not "
               "included, see `ray_gap_m`) and `contact_fit_iou` how well a prism explains both outlines "
               "(`contact_reason` says why a row has none). It needs the object inside both views."
               if wrist and {"cross_view", "contact_top"} <= profile.supports else "")
            + (" The attached table already carries `cross_view` against the " + "/".join(wrist) + " camera."
               if wrist and {"cross_view", "cross_view_attach"} <= profile.supports
               and getattr(features, "object_table_observe", False) else "")
        )
    if "robot_occlusion" in profile.supports and (getattr(features, "object_table_observe", False)
                                                  or getattr(features, "object_table", False)):
        parts.append(
            "Every object_table also carries `robot_occlusion`: `table_covered` is the share of the table within the "
            "arm's reach seen by that camera that the arm's silhouette (FK of the measured joints, robot base "
            "excluded) covers; `hidden` lists rows of that camera's previous table (and entries it already listed "
            "as hidden) whose estimated box now lies under the arm, with `covered` (share of the box's image area) "
            "and `seen_in` (the observation that listed it). In action deltas, `unmatched_under_arm` lists "
            "unmatched rows from the before observation whose earlier box overlaps the arm; their movement is unknown.")
    if getattr(features, "scene3d_observe", False) or getattr(features, "scene3d", False):
        if ik_staged:
            parts.append(
                "Use `import scene3d_helpers as s3` in `program_run` on demand for dense 3D point-cloud queries, 6R kinematic "
                "screening, rigid feature alignment, or multi-step 3D geometry and action coordination "
                "(`s3.load(artifact)` loads the point cloud in m: `s3.top_z`, `s3.table_z`, `s3.points_in_box`, "
                "`s3.points_in_cylinder`, `s3.nearest`, `s3.cluster_at_pixel`, `s3.surface_extrema(region, radius=0.04, min_conf=0.05)` for slot/hole/rim "
                "height extrema, `s3.cavity_profile(region_or_obj_id, radius=0.06, min_conf=0.05)` for hollow rim_z_m/floor_z_m/cavity_depth_m, "
                "`s3.in_hand_axis(grasp_center_m, radius=0.065, cloud=None, min_conf=0.1)` with `robot=True` for held-object 3D "
                "axis/bottom_point_m, and `s3.triangulate_rays(...)` for two-camera stereo depth). "
                "Optional 6R kinematics & rigid SE(3) helpers in `program_run`: "
                "`s3.check_source_to_dest(source_pos_m, source_angles_deg, dest_pos_m, dest_angles_deg=None)` (verify whether one single arm "
                "reaches both source and destination before picking), "
                "`s3.check_reachability(arm, position_m, angles_deg, contact_region='pad_center')` (6R IK in m/deg/rad; "
                "check `yaw +- 180`), `s3.solve_reachable_pose(arm, position_m, angles_deg, contact_region='pad_center')` "
                "(prioritizes zero-tilt `[0,0,yaw]` and SO(3) jaw symmetry `s3.jaw_symmetric_angles_deg` before approach tilts), "
                "`s3.plan_joint_safe_return(target_robot_state)`, "
                "`s3.prefilter_stages(stages)` (batch IK, travel rad, `min_bimanual_clearance_m`), "
                "`s3.align_feature_to_target(grasp_pos_m, grasp_angles_deg, feature_point_m, feature_axis, target_point_m, target_axis)` "
                "(returns `{target_grasp_position_m, target_pos_m, target_angles_deg, flange_position_m, quaternion_xyzw, rotation_angle_deg, feature_pos_err_m, feature_axis_err_deg}` in m/deg), "
                "and `s3.se3_screw(start_pose, axis, pivot_m=None, angle_rad=0.0, distance_m=0.0, pitch_m_per_rad=0.0, n_steps=4)` "
                "(samples rigid arc/helical waypoints `[{s, grasp_position_m, angles_deg}]` in m/rad/deg). "
                + ("Pass `obstacle_object_ids=['o1',...]` in `robot_action` only for non-target intermediate `object_table` obstacles (observation-scoped solid AABBs)."
                   if "obstacle_object_ids" in profile.supports else "")
            )
        else:
            parts.append(
                "When a scene3d point-cloud artifact handle is attached (or created via geometry_query kind='scene3d'), "
                "you can inspect local 3D surface geometry in program_run with `import scene3d_helpers as s3; s3.load(artifact)` "
                "(queries: s3.top_z, s3.table_z, s3.points_in_box, s3.points_in_cylinder, s3.nearest, s3.cluster_at_pixel, "
                "s3.surface_extrema(region, radius=0.04, min_conf=0.05) for unfiltered slot/hole/rim height extrema, "
                "s3.cavity_profile(region_or_obj_id, radius=0.06, min_conf=0.05) for hollow rim_z_m/floor_z_m/cavity_depth_m, "
                "s3.in_hand_axis(grasp_center_m, radius=0.065, cloud=None, min_conf=0.1) for 3D PCA yaw/tilt/posture and bottom_point_m / tip_offset_from_grasp_m "
                "of a held object near the fingertips using robot=True, "
                "and s3.triangulate_rays(...) for two-camera stereo depth anchoring)."
                + (" Pass `obstacle_object_ids=['o1',...]` in `robot_action` when you want specific 3D objects from "
                   "`object_table` converted into AABB obstacles." if "obstacle_object_ids" in profile.supports else "")
            )
    if getattr(features, "vggt_height", False):
        mode = getattr(features, "vggt_height_mode", "optional")
        if mode == "default":
            parts.append(
                "geometry_query(kind='pixel_plane') automatically uses VGGT-estimated object centre height at the "
                "pixel unless you set plane_source='caller' (your plane_origin_m is retained as the alternate plane)."
            )
        else:
            parts.append(
                "geometry_query(kind='pixel_plane') accepts plane_source='vggt' to intersect the ray with the "
                "VGGT-estimated object centre height at that pixel."
            )
    return ("\n\n## 3D Perception (VGGT + SAM)\n" + "\n".join("- " + p for p in parts)) if parts else ""


# 模型侧 3D 感知使用说明统一入口别名（与 developer_instructions 指向同一纯函数）
actor_instructions = developer_instructions


def short_reason(reason):
    text = str(reason or "")
    text = text.split("_unavailable:", 1)[-1]
    lowered = text.lower()
    if "timed out" in lowered or "timeout" in lowered:
        return "timeout"
    if "refused" in lowered or "connectionreset" in lowered or "remotedisconnected" in lowered:
        return "service_down"
    return text[:80] or "unknown"


def analysis_log_enabled(runtime):
    """analysis.scene_truth_log: the offline-scoring switch (the truth log itself is written by the worker only)."""
    analysis = getattr(getattr(runtime, "config", None), "analysis", None)
    return bool(getattr(analysis, "scene_truth_log", False))


def _analysis_log(runtime, correlation, observation_id, reply, table, result, reused, *, camera_id=None):
    """Internal event with the attached table for offline scoring; never part of a reply, never raises."""
    if not analysis_log_enabled(runtime):
        return
    result = result if isinstance(result, dict) else {}
    payload = dict(observation_id=observation_id, observation_ids=[observation_id], camera_id=camera_id or camera_of(runtime), reply=reply,
                   reused=bool(reused), frame_ids=list(result.get("frame_ids") or []), available=isinstance(table, dict),
                   service_version=(result.get("diagnostics") or {}).get("service_version"))
    if isinstance(table, dict):
        payload.update(columns=list(table["columns"]), rows=table["rows"], table_z=table.get("table_z"),
                       omitted=table.get("omitted", 0))
    else:
        payload["unavailable"] = str(table)
    try:
        runtime.store.event(ANALYSIS_EVENT, payload, correlation, world_kind="internal")
    except Exception:
        pass


def camera_of(runtime, bundle=None):
    """features.vggt_camera_id (object_table.default_camera: cam_head on X5, fixed on SO-101): the camera whose
    table / cloud is attached."""
    from .object_table import default_camera
    return default_camera(runtime, bundle) or embodiment_of(runtime).overview_camera


def _head_camera(bundle, camera_id):
    for camera in (bundle or {}).get("cameras") or []:
        if isinstance(camera, dict) and camera.get("camera_id") == camera_id:
            return camera
    return None


#: Profile tag: the attached table also carries cross_view against the wrist camera (SO-101 deployment
#: perception.attach_cross_view, default off).
CROSS_VIEW_ATTACH_TAG = "cross_view_attach"


def attached_targets(profile, camera_id, bundle_cameras):
    """target_camera_ids of the attached table: with the ``cross_view_attach`` tag the profile's wrist cameras that
    this observation has (``bundle_cameras``: its camera ids), other than ``camera_id``; else none."""
    if CROSS_VIEW_ATTACH_TAG not in profile.supports:
        return []
    return [camera for camera in profile.wrist_cameras.values() if camera != camera_id and camera in bundle_cameras]


def _object_table(runtime, observation_id, timeout_s, max_objects, *, camera_id=None, targets=(), on_before_send=None):
    from .object_table import object_table_query
    cam_id = camera_id or camera_of(runtime)
    request = {"kind": "object_table", "observation_id": observation_id, "camera_id": cam_id}
    if targets:
        request["target_camera_ids"] = list(targets)
    result, overlays = object_table_query(
        runtime,
        request,
        timeout_s=timeout_s,
        max_objects=max_objects,
        slack_s=0.0,
        on_before_send=on_before_send,
    )
    value = result.get("value") if result.get("status") == "conditional" else None
    if not isinstance(value, dict):
        return "unavailable: " + short_reason(result.get("reason")), None, result
    shown = attached_table_view(value)
    persistent_ids = bool((shown.get("track_meta") or {}).get("persistent_ids"))
    shown["note"] = object_table_note(embodiment_of(runtime), cam_id, persistent_ids=persistent_ids)
    overlay = next(((data, original) for cid, data, original in overlays if cid == cam_id), None)
    return shown, overlay, result


def attached_table_view(value):
    """Formal table estimates shared by inline replies, cached geometry queries and checkpoint replay."""
    shown = dict(columns=value["columns"], rows=value["rows"], table_z=value.get("support_z_m"))
    if isinstance(value.get("track_meta"), dict):
        shown["track_meta"] = value["track_meta"]
    if value.get("grasp_hints"):
        # 内联观测只附带无偏客观几何属性（pad_z_m / across_yaw_deg 双候选 / cavity_depth_m），剥离无臂无点云的中心钦定抓取点 safe_grasp_pose
        shown["grasp_hints"] = {
            oid: ({k: v for k, v in hint.items() if k != "safe_grasp_pose"} if isinstance(hint, dict) else hint)
            for oid, hint in value["grasp_hints"].items()
        }
    if value.get("spatial_meta"):
        shown["spatial_meta"] = value["spatial_meta"]
    if value.get("omitted"):
        shown["omitted"] = value["omitted"]
    if value.get("held"):   # features.held_perception only
        shown["held"] = value["held"]
    if value.get("robot_occlusion") is not None:   # profile tag robot_occlusion (SO-101)
        shown["robot_occlusion"] = value["robot_occlusion"]
    if value.get("robot_mask_degraded"):
        shown["robot_mask_degraded"] = value["robot_mask_degraded"]
    if value.get("cross_view"):   # profile tag cross_view_attach (SO-101)
        shown["cross_view"] = value["cross_view"]
    return shown


def _scene3d_fetch(runtime, observation_id, timeout_s, *, camera_id=None):
    from .scene3d import scene3d_query
    features = getattr(getattr(runtime, "config", runtime), "features", runtime)
    include_wrist = bool(getattr(features, "multiview_scene3d", False))
    return scene3d_query(runtime, {"kind": "scene3d", "observation_id": observation_id,
                                   "camera_id": camera_id or camera_of(runtime),
                                   "include_wrist_views": include_wrist},
                         timeout_s=timeout_s, slack_s=0.0, auto_filter_wrist_views=True)


def _scene3d_store(runtime, observation_id, correlation, result, data, *, camera_id=None):
    from .scene3d import ARTIFACT_KIND, ARTIFACT_MIME
    if data is None or result.get("status") != "conditional":
        return "unavailable: " + short_reason(result.get("reason")), result
    reference = runtime.store.put_artifact(data, ARTIFACT_KIND, ARTIFACT_MIME)
    result["value"]["artifact"] = reference
    runtime.store.event("geometry.scene3d_stored", {"artifact": reference, "bytes": len(data),
        "points": result["value"]["points"], "observation_id": observation_id, "camera_id": camera_id or camera_of(runtime),
        "source_kind": "obs3d"}, correlation, world_kind="internal")
    value = result["value"]
    shown_cloud = dict(artifact=reference, points=value["points"], table_z=value["table_z"], use=SCENE3D_USE)
    if isinstance(value.get("fused_views"), list) and len(value["fused_views"]) > 1:
        shown_cloud["fused_views"] = list(value["fused_views"])
    return shown_cloud, result


def _scene3d(runtime, observation_id, timeout_s, correlation, *, camera_id=None):
    result, data = _scene3d_fetch(runtime, observation_id, timeout_s, camera_id=camera_id)
    return _scene3d_store(runtime, observation_id, correlation, result, data, camera_id=camera_id)


def _memo(runtime, observation_id):
    """Successful results for observation_id (checks latest _obs3d_memo and bounded _obs3d_history): {'object_table': (...), 'scene3d': (...)}."""
    memo = slot_peek(runtime, OBS3D_MEMO_SLOT, None, legacy_attr="_obs3d_memo")
    if isinstance(memo, tuple) and len(memo) == 2 and memo[0] == observation_id:
        return memo[1]
    history = slot_peek(runtime, OBS3D_HISTORY_SLOT, None, legacy_attr="_obs3d_history")
    if isinstance(history, dict) and observation_id in history:
        return history[observation_id]
    return {}


def attached_table(runtime, observation_id):
    """Object table attached to ``observation_id`` by obs3d (``features.object_table_observe``); ``None`` when absent."""
    from ..world_model import attached_table as _world_attached_table
    found = _world_attached_table(runtime, observation_id)
    if found is not None:
        return found
    entry = _memo(runtime, observation_id).get("object_table")
    return entry[0] if isinstance(entry, tuple) and entry else (entry if isinstance(entry, dict) else None)


def _sync_world_model(runtime, observation_id, entry, *, bundle=None):
    """Synchronize immutable ``WorldObservation3D`` snapshot for ``observation_id``."""
    if not observation_id or not isinstance(entry, dict):
        return
    from ..world_model import WorldObservation3D, get_world_observation, record_world_observation
    ot_entry = entry.get("object_table")
    table = None
    raw_objects = None
    cam_id = None
    if isinstance(ot_entry, tuple) and ot_entry:
        table = ot_entry[0] if isinstance(ot_entry[0], dict) else None
        if len(ot_entry) >= 3 and isinstance(ot_entry[2], dict):
            raw_objects = (ot_entry[2].get("diagnostics") or {}).get("objects")
            cam_id = (ot_entry[2].get("value") or {}).get("camera_id")
    elif isinstance(ot_entry, dict):
        table = ot_entry
    s3_entry = entry.get("scene3d")
    cloud = (
        (s3_entry[0] if isinstance(s3_entry[0], dict) else None)
        if isinstance(s3_entry, tuple) and s3_entry
        else (s3_entry if isinstance(s3_entry, dict) else None)
    )
    if table is None and cloud is None:
        return
    robot_state = _extract_bundle_robot_state(runtime, bundle) if isinstance(bundle, dict) else None
    prev = get_world_observation(runtime, observation_id)
    if prev is not None and prev.observation_id == str(observation_id):
        wobs = prev.with_updates(
            table=table,
            raw_objects=raw_objects,
            camera_id=cam_id,
            bundle=bundle,
            cloud=cloud,
            robot_state=robot_state,
        )
    else:
        wobs = WorldObservation3D.from_parts(
            observation_id,
            table=table,
            raw_objects=raw_objects,
            camera_id=cam_id,
            bundle=bundle,
            cloud=cloud,
            robot_state=robot_state,
        )
    record_world_observation(runtime, wobs)


def _remember(runtime, observation_id, key, value, *, bundle=None):
    entry = dict(_memo(runtime, observation_id))
    entry[key] = value
    try:
        slot_set(runtime, OBS3D_MEMO_SLOT, (observation_id, entry), legacy_attr="_obs3d_memo")
        history = slot_get(runtime, OBS3D_HISTORY_SLOT, legacy_attr="_obs3d_history")
        if not isinstance(history, dict):
            history = {}
            slot_set(runtime, OBS3D_HISTORY_SLOT, history, legacy_attr="_obs3d_history")
        history[observation_id] = entry
        while len(history) > 8:
            history.pop(next(iter(history)))
    except Exception:
        pass
    _sync_world_model(runtime, observation_id, entry, bundle=bundle)


def _extract_bundle_robot_state(runtime, bundle):
    """从当前观测 bundle 或其引用 artifact 提取 robot_state；读取或解析失败时安全返回 `None`。"""
    if not isinstance(bundle, dict):
        return None
    robot_state = bundle.get("robot_state")
    if not isinstance(robot_state, dict) and bundle.get("robot_state_ref"):
        resolve_art = getattr(getattr(runtime, "store", None), "resolve_artifact", None)
        if callable(resolve_art):
            import json
            try:
                _, raw_bytes = resolve_art(bundle["robot_state_ref"])
                robot_state = json.loads(raw_bytes)
            except Exception:
                robot_state = None
    return robot_state if isinstance(robot_state, dict) else None


def _project_snapshot_robot_state(robot_state, arms=None):
    """仅投影模型合法可见的公共本体状态字段（位姿与夹爪开度/指令），剔除一切内部或特权字段。"""
    if not isinstance(robot_state, dict) or not isinstance(robot_state.get("arms"), dict):
        return None
    arm_keys = tuple(arms) if arms is not None else tuple(robot_state["arms"].keys())
    arms_out = {}
    for arm in arm_keys:
        src = robot_state["arms"].get(arm)
        if not isinstance(src, dict):
            continue
        dst = {}
        ee = src.get("ee_pose")
        if isinstance(ee, dict):
            ee_out = {}
            if isinstance(ee.get("position_m"), (list, tuple)) and len(ee["position_m"]) == 3:
                ee_out["position_m"] = [float(v) for v in ee["position_m"]]
            if isinstance(ee.get("quaternion_xyzw"), (list, tuple)) and len(ee["quaternion_xyzw"]) == 4:
                ee_out["quaternion_xyzw"] = [float(v) for v in ee["quaternion_xyzw"]]
            if ee_out:
                dst["ee_pose"] = ee_out
        gp = src.get("grasp_pose")
        if "ee_pose" not in dst and isinstance(gp, dict) and isinstance(gp.get("position_m"), (list, tuple)) and len(gp["position_m"]) == 3:
            ang = gp.get("angles_deg") or [0.0, 0.0, 0.0]
            dst["grasp_pose"] = {
                "position_m": [float(v) for v in gp["position_m"]],
                "angles_deg": [float(v) for v in ang[:3]],
            }
        grip = src.get("gripper")
        if isinstance(grip, dict):
            grip_out = {}
            for key in ("goal_normalized", "command_normalized"):
                if isinstance(grip.get(key), (int, float)):
                    grip_out[key] = float(grip[key])
            if isinstance(grip.get("position_m"), (list, tuple)):
                grip_out["position_m"] = [float(v) for v in grip["position_m"]]
            if grip_out:
                dst["gripper"] = grip_out
        if dst:
            arms_out[arm] = dst
    return {"arms": arms_out} if arms_out else None


def _sync_virtual_field(runtime, bundle, table, *, cloud_npz=None, raw_objects=None):
    """基于 Copy-on-Write 构建新实例后原子替换虚拟空间场槽位。"""
    profile = embodiment_of(runtime)
    if "grasp_angles" not in profile.pose_inputs:
        return
    if not isinstance(table, dict) and cloud_npz is None:
        return
    try:
        from .pointcloud_field import PointCloudSpatialField
        obs_id = (bundle or {}).get("observation_id") if isinstance(bundle, dict) else None
        robot_state = _extract_bundle_robot_state(runtime, bundle)
        prev_vf = slot_peek(runtime, VIRTUAL_FIELD_SLOT, None, legacy_attr="_virtual_spatial_field")
        if not isinstance(prev_vf, PointCloudSpatialField):
            next_vf = PointCloudSpatialField(
                objects=table if isinstance(table, dict) else None,
                robot_state=robot_state,
                cloud_npz=cloud_npz,
                raw_objects=raw_objects,
                profile=profile,
            )
            if isinstance(table, dict) and obs_id is not None:
                next_vf._ingested_table_obs_id = obs_id
        else:
            next_vf = prev_vf.clone()
            prev_ingested = getattr(prev_vf, "_ingested_table_obs_id", None)
            same_obs_ingested = bool(obs_id is not None and prev_ingested == obs_id)
            eff_table = None if same_obs_ingested else (table if isinstance(table, dict) else None)
            next_vf.update_from_observation(
                eff_table,
                robot_state=robot_state,
                cloud_npz=cloud_npz,
                raw_objects=raw_objects,
            )
            next_vf._ingested_table_obs_id = obs_id if (same_obs_ingested or isinstance(table, dict)) else prev_ingested
        next_vf.observation_id = obs_id
        slot_set(runtime, VIRTUAL_FIELD_SLOT, next_vf, legacy_attr="_virtual_spatial_field")
        slot_set(runtime, VIRTUAL_FIELD_ERROR_SLOT, None, legacy_attr="_virtual_field_last_error")
    except Exception as error:
        slot_set(
            runtime,
            VIRTUAL_FIELD_ERROR_SLOT,
            f"{type(error).__name__}:{str(error)[:200]}",
            legacy_attr="_virtual_field_last_error",
        )


def _persist_spatial_snapshot(runtime, bundle, shown):
    """将与当前 `observation_id` 严格对齐的轻量空间快照写入 Store 状态，供 `program_run` 零手抄预载。"""
    set_state = getattr(getattr(runtime, "store", None), "set_state", None)
    if not callable(set_state) or not isinstance(shown, dict):
        return
    obs_id = shown.get("observation_id")
    if not obs_id:
        return
    table = shown.get("object_table") if isinstance(shown.get("object_table"), dict) else None
    cloud = shown.get("scene3d") if isinstance(shown.get("scene3d"), dict) else None
    if table is None and cloud is None:
        return
    raw_rs = _extract_bundle_robot_state(runtime, bundle)
    public_rs = _project_snapshot_robot_state(raw_rs, arms=embodiment_of(runtime).arms)
    cur_vf = slot_peek(runtime, VIRTUAL_FIELD_SLOT, None, legacy_attr="_virtual_spatial_field")
    vf_aligned = cur_vf is not None and getattr(cur_vf, "observation_id", None) == obs_id
    visible_ids = set()
    if isinstance(table, dict):
        cols = table.get("columns") or ()
        id_idx = cols.index("id") if "id" in cols else 0
        for row in table.get("rows") or ():
            if isinstance(row, (list, tuple)) and len(row) > id_idx:
                visible_ids.add(str(row[id_idx]))
    persisted_objects = []
    cavity_depths_m = {}
    surface_slices_m = {}
    table_z = table.get("table_z") if isinstance(table, dict) else (cloud.get("table_z") if isinstance(cloud, dict) else None)
    if vf_aligned:
        if cur_vf.table_z is not None:
            table_z = float(cur_vf.table_z)
        cavity_depths_m = {str(k): round(float(v), 4) for k, v in getattr(cur_vf, "cavity_depths_m", {}).items()}
        surface_slices_m = deepcopy(getattr(cur_vf, "surface_slices_m", {}))
        for oid, obj in getattr(cur_vf, "objects", {}).items():
            if oid not in visible_ids:
                persisted_objects.append(obj.to_dict())
        historical_unobserved = [
            obj.to_dict() for obj in getattr(cur_vf, "historical_unobserved_objects", {}).values()
        ]
    else:
        historical_unobserved = []
    snap_table = deepcopy(table) if table is not None else None
    if isinstance(snap_table, dict) and historical_unobserved:
        snap_table["historical_unobserved_objects"] = historical_unobserved
    snapshot = {
        "observation_id": obs_id,
        "object_table": snap_table,
        "cloud_artifact": cloud.get("artifact") if isinstance(cloud, dict) else None,
        "table_z": table_z,
        "robot_state": public_rs,
        "cavity_depths_m": cavity_depths_m,
        "surface_slices_m": surface_slices_m,
        "persisted_objects": persisted_objects,
        "historical_unobserved_objects": historical_unobserved,
    }
    set_state("perception", "spatial_snapshot", snapshot)


def is_obs3d_healthy(runtime, shown):
    """客观判定当前观测的开启项 3D 感知是否全部成功返回结构化字典（不引入任何开度或几何魔法阈值）。"""
    object_table_on, scene3d_on = flags(runtime)
    if not (object_table_on or scene3d_on) or not isinstance(shown, dict):
        return False
    if object_table_on and not isinstance(shown.get("object_table"), dict):
        return False
    if scene3d_on and not isinstance(shown.get("scene3d"), dict):
        return False
    return True


def _record_postprocess_error(runtime, diagnostics: dict, correlation: dict, stage: str, error: BaseException) -> None:
    detail = f"{stage}:{type(error).__name__}:{str(error)[:160]}"
    diagnostics.setdefault("postprocess_errors", []).append(detail)
    try:
        runtime.store.event(
            "obs3d.postprocess_error",
            {"observation_id": diagnostics.get("observation_id"), "stage": stage, "error": detail},
            correlation,
            world_kind="internal",
        )
    except Exception:
        pass


def _live_context_epoch(runtime):
    """读取当前活跃模型上下文纪元；若测试桩未提供上下文纪元接口则返回 None。"""
    dc = getattr(runtime, "decision_context", None)
    live_fn = getattr(dc, "live_context_epoch", None)
    if callable(live_fn):
        return live_fn()
    epoch_fn = getattr(runtime, "_model_context_epoch", None)
    if callable(epoch_fn):
        return epoch_fn()
    dc_fn = getattr(dc, "model_context_epoch", None)
    return dc_fn() if callable(dc_fn) else None


def _context_epoch(runtime, correlation=None):
    """读取本次调用可用于去重的模型上下文纪元；非主链作用域、易变模式或测试桩缺失接口时返回 None。"""
    corr = correlation or {}
    if corr.get("context_scope", "main") != "main":
        return None
    dc = getattr(runtime, "decision_context", None)
    actor_fn = getattr(dc, "actor_delivery_epoch", None)
    if callable(actor_fn):
        return actor_fn(corr)
    bound = corr.get("context_epoch")
    if bound is not None:
        return bound
    return _live_context_epoch(runtime)


def _record_scene3d_use_epoch(runtime, shown, *, is_program: bool, correlation=None) -> None:
    """在面向主模型且实际投递了 scene3d.use 且纪元仍活跃时记录当前上下文纪元。"""
    if is_program or not isinstance(shown, dict):
        return
    cloud = shown.get("scene3d")
    if isinstance(cloud, dict) and "use" in cloud:
        epoch = _context_epoch(runtime, correlation)
        live_epoch = _live_context_epoch(runtime)
        if epoch is not None and epoch == live_epoch:
            slot_set(runtime, SCENE3D_USE_EPOCH_SLOT, epoch, legacy_attr="_obs3d_scene3d_use_epoch")


def attachments(
    runtime,
    bundle,
    correlation,
    *,
    reply="observe",
    before_observation_id=None,
    suppress_overlay=False,
    basis_valid=None,
    record_use_epoch=True,
):
    """Extra display parts for one model or program reply: [] or [StructuredText, (inputImage)].

    reply 'observe' (robot_observe, full observation): text + overlay (model only);
    'action' (final observation of robot_action / robot_wait): text, plus the overlay only with features.obs3d_action_overlay.
    program_run 子调用正常计算并更新 _obs3d_memo / _virtual_spatial_field，返回纯结构化文本且不生成叠加图。"""
    correlation = correlation or {}
    object_table_on, scene3d_on = flags(runtime)
    if not (object_table_on or scene3d_on) or not isinstance(bundle, dict):
        return []
    is_program = bool(correlation.get("program_id"))
    with_overlay = (not is_program) and (not suppress_overlay) and (reply == "observe" or action_overlay(runtime))
    if reply == "observe":
        pending_obs = slot_peek(runtime, PENDING_OBSERVE_OBS3D_SLOT, None, legacy_attr="_pending_observe_obs3d")
        slot_reset(runtime, PENDING_OBSERVE_OBS3D_SLOT, legacy_attr="_pending_observe_obs3d")
        if (
            isinstance(pending_obs, dict)
            and pending_obs.get("observation_id") == bundle.get("observation_id")
            and pending_obs.get("is_program") == is_program
            and pending_obs.get("tool_call_id") == correlation.get("tool_call_id")
            and pending_obs.get("with_overlay") == with_overlay
        ):
            parts = list(pending_obs["parts"])
            if record_use_epoch and parts and hasattr(parts[0], "value"):
                _record_scene3d_use_epoch(runtime, parts[0].value, is_program=is_program, correlation=correlation)
            return parts
    cam_id = camera_of(runtime, bundle)
    camera = _head_camera(bundle, cam_id)
    if camera is None:
        return []
    from ...core.tool_output import StructuredText
    features = runtime.config.features
    timeout_s = float(getattr(features, "obs3d_timeout_s", 10.0))
    observation_id = bundle["observation_id"]
    shown = {"observation_id": observation_id}
    images, diagnostics = [], {"observation_id": observation_id, "camera_id": cam_id, "reply": reply}
    started = _monotonic()
    memo = _memo(runtime, observation_id)
    table = cloud = None
    ot_result = {}
    diagnostics.update(object_table={"status": "disabled"}, scene3d={"status": "disabled"})
    s3_prefetch = None
    ot_dispatched = threading.Event()
    if object_table_on and scene3d_on and "object_table" not in memo and "scene3d" not in memo:
        # 待 /object_table 请求字节已通过 TCP 写入套接字后零睡眠触发 /points 并发飞行，严格保障服务端 FIFO 顺序
        s3_box = {}
        def _run_s3():
            ot_dispatched.wait(timeout=max(0.05, timeout_s))
            t_s3 = _monotonic()
            s3_box["t_start"] = t_s3
            try:
                s3_box["res"] = _scene3d_fetch(runtime, observation_id, timeout_s, camera_id=cam_id)
            except Exception as exc:
                s3_box["err"] = exc
            s3_box["elapsed_s"] = _monotonic() - t_s3
        s3_thread = threading.Thread(target=_run_s3, daemon=True)
        s3_thread.start()
        s3_prefetch = (s3_thread, s3_box)
    try:
        table_for_vf = None
        raw_objs = None
        if object_table_on:
            t0 = _monotonic()
            reused = "object_table" in memo
            if reused:
                ot_dispatched.set()
                table, overlay, result = memo["object_table"]
                diagnostics.setdefault("reused", []).append("object_table")
                if isinstance(table, dict):
                    table_for_vf = table
                    raw_objs = (result.get("diagnostics") or {}).get("objects") if isinstance(result, dict) else None
            else:
                try:
                    table, overlay, result = _object_table(runtime, observation_id, timeout_s,
                                                           int(getattr(features, "obs3d_max_objects", 10)),
                                                           camera_id=cam_id,
                                                           targets=attached_targets(
                                                               embodiment_of(runtime),
                                                               cam_id,
                                                               [c.get("camera_id") for c in bundle.get("cameras") or []
                                                                if isinstance(c, dict)],
                                                           ),
                                                           on_before_send=ot_dispatched.set)
                except Exception as error:  # an optional attachment never hides the observation
                    ot_dispatched.set()
                    table, overlay, result = ("unavailable: " + short_reason(type(error).__name__ + ":" + str(error)),
                                              None, {})
                if isinstance(table, dict):
                    table_for_vf = table
                    raw_objs = (result.get("diagnostics") or {}).get("objects") if isinstance(result, dict) else None
                    try:
                        _remember(runtime, observation_id, "object_table", (table, overlay, result), bundle=bundle)
                    except Exception as error:
                        _record_postprocess_error(runtime, diagnostics, correlation, "remember_object_table", error)
                    _sync_virtual_field(runtime, bundle, table, raw_objects=raw_objs)
            ot_result = result
            shown["object_table"] = table
            if isinstance(table, dict) and before_observation_id and before_observation_id != observation_id:
                try:
                    ep_ledger = episode_state_of(runtime, create_fallback=False)
                    allow_delta = bool(basis_valid) if basis_valid is not None else True
                    before_obs = before_observation_id
                    if allow_delta and ep_ledger is not None and not ep_ledger.is_basis(before_obs):
                        allow_delta = False
                    before_entry = _memo(runtime, before_obs).get("object_table") if allow_delta else None
                    if allow_delta and not (isinstance(before_entry, tuple) and before_entry):
                        history = slot_peek(runtime, OBS3D_HISTORY_SLOT, None, legacy_attr="_obs3d_history")
                        if isinstance(history, dict):
                            for hist_obs_id in reversed(list(history)):
                                if hist_obs_id != observation_id and "object_table" in history[hist_obs_id]:
                                    if ep_ledger is not None and not ep_ledger.is_basis(hist_obs_id):
                                        continue
                                    before_obs = hist_obs_id
                                    before_entry = history[hist_obs_id]["object_table"]
                                    break
                    before_table = before_entry[0] if isinstance(before_entry, tuple) and before_entry else None
                    if allow_delta and isinstance(before_table, dict):
                        from .object_table import compute_object_table_delta
                        delta = compute_object_table_delta(before_table, table, before_observation_id=before_obs)
                        if delta:
                            shown["object_state_delta"] = delta
                            slot_set(runtime, LAST_OBS3D_DELTA_SLOT, (before_obs, observation_id, deepcopy(delta)))
                            delta_sink = getattr(runtime, "on_object_table_delta", None)
                            if callable(delta_sink):
                                delta_sink(before_obs, observation_id, delta)
                except Exception as error:
                    _record_postprocess_error(runtime, diagnostics, correlation, "object_state_delta", error)
            _analysis_log(runtime, correlation, observation_id, reply, table, result, reused, camera_id=cam_id)
            diagnostics["object_table"] = dict(status=result.get("status") or "unavailable", reason=result.get("reason") or (table if isinstance(table, str) else None),
                rows=len(table["rows"]) if isinstance(table, dict) else None, latency_s=round(_monotonic() - t0, 3),
                service=(result.get("diagnostics") or {}).get("service_version"),
                service_timing=(result.get("diagnostics") or {}).get("service_timing"))
            if overlay is not None:
                try:
                    from ...core.storage.images import image_settings, image_url
                    data, original = overlay
                    settings = runtime.image_settings if runtime.image_settings is not None else image_settings(None)
                    reference = runtime.store.put_artifact(data, "geometry_overlay", "image/jpeg")
                    diagnostics["object_table"]["overlay_ref"] = reference
                    if with_overlay:
                        url, encoding = image_url(data, settings)
                        images.append(dict(type="inputImage", imageUrl=url))
                        if isinstance(table, dict):   # the memoised table keeps the note without the image claim
                            persistent_ids = bool((table.get("track_meta") or {}).get("persistent_ids"))
                            shown["object_table"] = dict(
                                table,
                                note=object_table_note(
                                    embodiment_of(runtime), cam_id, overlay=True, persistent_ids=persistent_ids
                                ),
                            )
                        runtime.store.event("geometry.overlay_delivered", {"image_ref": reference, "source_sha256": encoding["sha256"],
                            "sha256": encoding["sha256"], "encoding": encoding, "original_rgb_ref": original, "camera_id": cam_id,
                            "observation_id": observation_id,
                            "source_kind": "obs3d_object_table_overlay", "delivered_to_model": True},
                            correlation, world_kind="internal")
                except Exception as error:
                    diagnostics["object_table"]["overlay_error"] = short_reason(type(error).__name__ + ":" + str(error))
        else:
            ot_dispatched.set()
        if scene3d_on:
            t0 = _monotonic()
            s3_data_bytes = None
            if "scene3d" in memo:
                cloud, result = memo["scene3d"]
                diagnostics.setdefault("reused", []).append("scene3d")
                s3_latency_s = _monotonic() - t0
                cur_vf = slot_peek(runtime, VIRTUAL_FIELD_SLOT, None, legacy_attr="_virtual_spatial_field")
                if isinstance(cloud, dict) and cloud.get("artifact") and getattr(cur_vf, "_cloud_data", None) is None:
                    resolve_art = getattr(getattr(runtime, "store", None), "resolve_artifact", None)
                    if callable(resolve_art):
                        try:
                            _, s3_data_bytes = resolve_art(cloud["artifact"])
                        except Exception:
                            s3_data_bytes = None
            elif s3_prefetch is not None:
                s3_thread, s3_box = s3_prefetch
                s3_thread.join(timeout=max(0.05, timeout_s + 0.5))
                t_joined = _monotonic()
                try:
                    if s3_thread.is_alive():
                        raise TimeoutError("scene3d_prefetch_timeout")
                    if "err" in s3_box:
                        raise s3_box["err"]
                    s3_result, s3_data_bytes = s3_box["res"]
                    cloud, result = _scene3d_store(runtime, observation_id, correlation, s3_result, s3_data_bytes, camera_id=cam_id)
                except Exception as error:
                    s3_data_bytes = None
                    cloud, result = "unavailable: " + short_reason(type(error).__name__ + ":" + str(error)), {}
                if isinstance(cloud, dict):
                    try:
                        _remember(runtime, observation_id, "scene3d", (cloud, result), bundle=bundle)
                    except Exception as error:
                        _record_postprocess_error(runtime, diagnostics, correlation, "remember_scene3d", error)
                elapsed = s3_box.get("elapsed_s")
                if elapsed is not None and not s3_thread.is_alive():
                    s3_latency_s = float(elapsed) + max(0.0, _monotonic() - t_joined)
                else:
                    s3_latency_s = max(0.0, _monotonic() - float(s3_box.get("t_start", t0)))
            else:
                try:
                    s3_result, s3_data_bytes = _scene3d_fetch(runtime, observation_id, timeout_s, camera_id=cam_id)
                    cloud, result = _scene3d_store(runtime, observation_id, correlation, s3_result, s3_data_bytes, camera_id=cam_id)
                except Exception as error:
                    s3_data_bytes = None
                    cloud, result = "unavailable: " + short_reason(type(error).__name__ + ":" + str(error)), {}
                if isinstance(cloud, dict):
                    try:
                        _remember(runtime, observation_id, "scene3d", (cloud, result), bundle=bundle)
                    except Exception as error:
                        _record_postprocess_error(runtime, diagnostics, correlation, "remember_scene3d", error)
                s3_latency_s = _monotonic() - t0
            if isinstance(cloud, dict) and s3_data_bytes is not None:
                _sync_virtual_field(runtime, bundle, table_for_vf, cloud_npz=s3_data_bytes, raw_objects=raw_objs)
            if isinstance(cloud, dict):
                epoch = None if is_program else _context_epoch(runtime, correlation)
                live_epoch = None if epoch is None else _live_context_epoch(runtime)
                is_live = epoch is not None and epoch == live_epoch
                prev_epoch = (
                    slot_peek(runtime, SCENE3D_USE_EPOCH_SLOT, None, legacy_attr="_obs3d_scene3d_use_epoch")
                    if is_live
                    else None
                )
                if is_live and prev_epoch == epoch:
                    shown["scene3d"] = {k: v for k, v in cloud.items() if k != "use"}
                else:
                    shown["scene3d"] = dict(cloud)
                    if record_use_epoch and is_live and "use" in cloud:
                        slot_set(runtime, SCENE3D_USE_EPOCH_SLOT, epoch, legacy_attr="_obs3d_scene3d_use_epoch")
            else:
                shown["scene3d"] = cloud
            diagnostics["scene3d"] = dict(status=result.get("status") or "unavailable", reason=result.get("reason") or (cloud if isinstance(cloud, str) else None),
                artifact=cloud.get("artifact") if isinstance(cloud, dict) else None,
                points=cloud.get("points") if isinstance(cloud, dict) else None,
                latency_s=round(s3_latency_s, 3),
                npz_bytes=(result.get("diagnostics") or {}).get("npz_bytes"),
                cache_hit=(result.get("diagnostics") or {}).get("cache_hit"))
    finally:
        ot_dispatched.set()
        if s3_prefetch is not None:
            s3_prefetch[0].join(timeout=max(0.05, min(1.0, timeout_s)))
    diagnostics["latency_s"] = round(_monotonic() - started, 3)
    try:
        _persist_spatial_snapshot(runtime, bundle, shown)
    except Exception as error:
        _record_postprocess_error(runtime, diagnostics, correlation, "persist_spatial_snapshot", error)
    try:
        runtime.store.event("geometry.obs3d_attached", diagnostics, correlation, world_kind="internal")
    except Exception:
        pass
    try:
        ot_diag = diagnostics.get("object_table") or {}
        res_diag = ot_result.get("diagnostics") or {}
        perception_record = {
            "observation_id": observation_id,
            "camera_id": cam_id,
            "reply": reply,
            "overlay_ref": ot_diag.get("overlay_ref"),
            "table": table if isinstance(table, dict) else None,
            "scene3d": cloud if isinstance(cloud, dict) else None,
            "delta": shown.get("object_state_delta"),
            "backend": res_diag.get("backend"),
            "service_version": res_diag.get("service_version"),
            "service_timing": res_diag.get("service_timing"),
            "scale": res_diag.get("scale"),
            "support_fit": res_diag.get("support_fit"),
            "objects": res_diag.get("objects") or [],
            "diagnostics": diagnostics,
            "latency_s": diagnostics.get("latency_s"),
        }
        runtime.store.set_state("perception", observation_id, perception_record)
    except Exception as error:
        try:
            runtime.store.event("geometry.perception_archive_failed", {
                "observation_id": observation_id, "reason": short_reason(type(error).__name__ + ":" + str(error))},
                correlation, world_kind="internal")
        except Exception:
            pass
    return [StructuredText(shown, projection="data"), *images]


def prefetch_observe_attachments(runtime, bundle, correlation, *, cache_pending=False):
    """在观测构建阶段预计算 obs3d 结构化结果；仅当紧邻同调用即将消费 attachments 时按完整受众键缓存。"""
    if not any(flags(runtime)) or not isinstance(bundle, dict):
        return None
    correlation = correlation or {}
    try:
        parts = attachments(
            runtime,
            bundle,
            correlation,
            reply="observe",
            suppress_overlay=True,
            record_use_epoch=False,
        )
    except Exception:
        return None
    if not parts:
        return None
    if cache_pending:
        slot_set(
            runtime,
            PENDING_OBSERVE_OBS3D_SLOT,
            dict(
                observation_id=bundle.get("observation_id"),
                is_program=bool(correlation.get("program_id")),
                tool_call_id=correlation.get("tool_call_id"),
                with_overlay=False,
                parts=parts,
            ),
            legacy_attr="_pending_observe_obs3d",
        )
    return parts[0].value if hasattr(parts[0], "value") and isinstance(parts[0].value, dict) else None


def prefetch_action_attachments(runtime, bundle, correlation, *, before_observation_id=None, suppress_overlay=False):
    """在动作后观测构建阶段预计算 obs3d 并返回客观健康状态，按完整上下文键缓存供紧随其后的 action_attachments 消费。"""
    if not any(flags(runtime)) or not isinstance(bundle, dict):
        return False
    correlation = correlation or {}
    is_program = bool(correlation.get("program_id"))
    effective_with_overlay = (not is_program) and (not suppress_overlay) and bool(action_overlay(runtime))
    try:
        parts = attachments(
            runtime,
            bundle,
            correlation,
            reply="action",
            before_observation_id=before_observation_id,
            suppress_overlay=suppress_overlay,
            record_use_epoch=False,
        )
    except Exception:
        return False
    slot_set(
        runtime,
        PENDING_ACTION_OBS3D_SLOT,
        dict(
            observation_id=bundle.get("observation_id"),
            before_observation_id=before_observation_id,
            is_program=is_program,
            tool_call_id=correlation.get("tool_call_id"),
            suppress_overlay=bool(suppress_overlay),
            with_overlay=effective_with_overlay,
            parts=parts,
        ),
        legacy_attr="_pending_action_obs3d",
    )
    shown = parts[0].value if parts and hasattr(parts[0], "value") else None
    return is_obs3d_healthy(runtime, shown)


def action_attachments(runtime, bundle, correlation, *, before_observation_id=None):
    """Parts for the final observation of a robot_action / robot_wait reply ([] when off); never raises."""
    if not any(flags(runtime)) or not isinstance(bundle, dict):
        return []
    correlation = correlation or {}
    is_program = bool(correlation.get("program_id"))
    pending = slot_peek(runtime, PENDING_ACTION_OBS3D_SLOT, None, legacy_attr="_pending_action_obs3d")
    slot_reset(runtime, PENDING_ACTION_OBS3D_SLOT, legacy_attr="_pending_action_obs3d")
    inherit_suppress = False
    if (
        isinstance(pending, dict)
        and pending.get("observation_id") == bundle.get("observation_id")
        and pending.get("is_program") == is_program
        and pending.get("tool_call_id") == correlation.get("tool_call_id")
    ):
        if pending.get("before_observation_id") == before_observation_id:
            parts = list(pending["parts"])
            if parts and hasattr(parts[0], "value"):
                _record_scene3d_use_epoch(runtime, parts[0].value, is_program=is_program, correlation=correlation)
            return parts
        # 同一次调用内起点观测不同（如多步 program_run 汇总首步起点）时重算增量，但继承本轮图像抑制策略
        inherit_suppress = bool(pending.get("suppress_overlay"))
    try:
        return attachments(
            runtime,
            bundle,
            correlation,
            reply="action",
            before_observation_id=before_observation_id,
            suppress_overlay=inherit_suppress,
        )
    except Exception as error:  # the action already executed; report unavailable 3D attachment transparently instead of silent []
        from ...core.tool_output import StructuredText
        reason = short_reason(type(error).__name__ + ":" + str(error))
        obs_id = bundle.get("observation_id")
        try:
            runtime.store.event(
                "obs3d.action_attachments_error",
                {"observation_id": obs_id, "error": f"{type(error).__name__}: {str(error)[:200]}"},
                correlation,
                world_kind="internal",
            )
        except Exception:
            pass
        object_table_on, scene3d_on = flags(runtime)
        fallback_shown = {"observation_id": obs_id, "3d_perception": f"unavailable: {reason}"}
        if object_table_on:
            fallback_shown["object_table"] = f"unavailable: {reason}"
        if scene3d_on:
            fallback_shown["scene3d"] = f"unavailable: {reason}"
        return [StructuredText(fallback_shown, projection="data")]


def consume_last_object_table_delta(runtime, observation_id=None):
    """Pop the latest ``(before_observation_id, after_observation_id, delta)`` tuple recorded for ``observation_id``."""
    item = slot_peek(runtime, LAST_OBS3D_DELTA_SLOT, None)
    if not isinstance(item, tuple) or len(item) != 3:
        return None
    if observation_id is not None and item[1] != observation_id:
        return None
    slot_reset(runtime, LAST_OBS3D_DELTA_SLOT)
    return item


class Obs3dEnricher(ObservationEnricher):
    """Typed ``ObservationEnricher`` adapter over ``obs3d`` without upward runtime imports."""

    def enrich(self, host, request: EnrichmentRequest) -> Enrichment:
        obs = request.observation
        obs_id = obs.get("observation_id") if isinstance(obs, dict) else None
        if request.reply == "action":
            parts = action_attachments(
                host,
                obs,
                request.correlation,
                before_observation_id=request.before_observation_id if request.basis_valid else None,
            )
        else:
            parts = attachments(
                host,
                obs,
                request.correlation,
                reply=request.reply,
                before_observation_id=request.before_observation_id,
                suppress_overlay=request.suppress_overlay,
                basis_valid=request.basis_valid,
            )
        shown = parts[0].value if parts and hasattr(parts[0], "value") and isinstance(parts[0].value, dict) else {}
        facts = {}
        if isinstance(shown, dict):
            for key in ("object_table", "scene3d", "object_state_delta"):
                if key in shown:
                    facts[key] = shown[key]
        delta_tuple = consume_last_object_table_delta(host, obs_id)
        if delta_tuple is not None:
            facts["delta_record"] = dict(
                before_observation_id=delta_tuple[0],
                after_observation_id=delta_tuple[1],
                delta=delta_tuple[2],
            )
        return Enrichment(parts=tuple(parts), facts=facts)
