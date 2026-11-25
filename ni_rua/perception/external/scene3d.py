"""features.scene3d (on by default with external_perception): geometry_query kind 'scene3d' and the program helper module scene3d_helpers.

With the switch on, geometry_query accepts {kind:'scene3d', observation_id, camera_id?, voxel_m?}. The harness sends
the saved RGB image of camera_id (default: object_table.default_camera, the embodiment's overview camera), its
calibrated K and T_world_camera, the configured support (table) height and the robot link segments of the same
observation (public FK of the measured joints; used only to flag points near the robot) to the local VGGT service
(features.vggt_endpoint, POST /points). The service returns the VGGT-Omega point cloud of that image in the
observation's reference frame (EmbodimentProfile.frame_id), scaled to the support plane and voxel-downsampled, with a per-point
depth-confidence rank, the source pixel and a near-robot flag, as an npz file. The harness stores it as an artifact
(kind scene3d_points) and returns only a short summary: the artifact handle, point count, bounding boxes, table
height, a usage note and the helper API. Program jobs get scene3d_helpers.py in their workspace and query the cloud
there (nearest points, points in a box/cylinder, top surface height of a region, local table height, cluster at a
pixel); the cloud itself never enters the model context unless a program prints it. Inputs are permitted RGB,
calibration and proprioception only; no simulator truth is read. Values are monocular-depth estimates: no per-object
depth correction is applied (thin or white objects can appear above their true position; see conf and table_z).

With the switch off only enabled() is consulted: tool definitions, handler behaviour and program workspaces are
unchanged.
"""
from __future__ import annotations

import base64
import json
import math
import time
import urllib.error
import urllib.request

import numpy as np

from . import embodiment_of
from ._service_http import (
    close_pool,
    evict_cached_image_key,
    get_endpoint_sha_mode,
    make_image_cache_key,
    note_image_uploaded,
    note_server_response,
    parse_http_endpoint,
    post_json_keepalive,
    should_try_points_sha256,
)
from .object_table import DEFAULT_CAMERA, default_camera, effective_support_z_m
from .rectify import Rectified, map_npz_uv, rectified_cached

KIND = "scene3d"
SOURCE = "vggt_scene3d"
ARTIFACT_KIND = "scene3d_points"
ARTIFACT_MIME = "application/x-npz"
WORKSPACE_MODULE = "scene3d_helpers.py"
WORKSPACE_COMPANION_MODULES = ("pose_math.py", "field_ops.py", "ik_prefilter.py")

_KIND_SENTENCE = "scene3d ({camera}): VGGT point cloud of one RGB image in {frame_id} m for scene3d_helpers in program_run (or in-process via queries)."
KIND_SENTENCE = _KIND_SENTENCE.format(camera="default " + DEFAULT_CAMERA, frame_id="env_origin")   # kind_sentence(None)
FIELD_DESCRIPTIONS = {
    "voxel_m": "3D voxel size in m (0.002..0.05, default 0.01).",
    "include_wrist_views": "Fuse calibrated wrist RGB views (cam_left_wrist/cam_right_wrist) via dynamic FK T_world_camera (default false).",
    "include_progress_wrist_views": (
        "Also fuse quasi-static (<=0.8 rad/s, >=0.04 m baseline, <=0.004 m jaw change) 5 Hz wrist RGB views "
        "cached during the last action for occluded cavities/surfaces (default false)."
    ),
}
USAGE = ("In program_run: import scene3d_helpers as s3; s3.load(artifact). Print only needed numbers. "
         "Points are VGGT monocular-depth estimates scaled to table_z (check conf/table_z on thin, white or transparent parts). "
         "robot=True includes points within 6 cm of the arm (held objects).")
API = [
    "load(artifact)->cloud; info()->n,bbox,table_z,voxel_m; table_z(xy=None,radius=0.08)->z (m)",
    "points_in_box(lo,hi) / points_in_cylinder(center_xy,radius,z_min=None,z_max=None) -> n,centroid,min,max,top_z,conf_p50",
    "top_z(region,q=98) -> pct z (m) above table; region=(lo,hi)|{center:[x,y],radius:r}|{polygon:[[x,y],...]} (+z_min,z_max)",
    "nearest(p,k=5) -> xyz,dist,conf,uv; cluster_at_pixel(u,v,radius=0.015) -> n,centroid,min,max,top_z,bottom_z,seed",
    "surface_extrema(region,radius=0.04,min_conf=0.05) -> floor_z,rim_z,lowest_point_m,highest_point_m; triangulate_rays(u1,v1,K1,T1,u2,v2,K2,T2) -> point_m,skew_gap_m",
    "cavity_profile(region_or_obj_id,radius=0.06,min_conf=0.05) -> rim_z_m,floor_z_m,cavity_depth_m,opening_center_xy_m,inner_clearance_radius_m,n_points",
    "in_hand_axis(grasp_center_m,radius=0.065,cloud=None,min_conf=0.1) -> yaw_deg,tilt_deg,posture,elongation,bottom_point_m,tip_offset_from_grasp_m (robot=True)",
    "check_reachability(arm,position_m,angles_deg,contact_region='pad_center') -> reachable,joint_q_rad,pos_err_m,rot_err_rad,min_joint_margin_rad (<1ms 6R IK; yaw+-180 deg)",
    "prefilter_stages(stages) -> valid,blocking_stage_ids,reasons,min_bimanual_clearance_m (<3ms 6R IK, travel rad, clearance m)",
    "solve_reachable_pose(arm,position_m,angles_deg)->selected_angles_deg; check_source_to_dest(source_pos_m,source_angles_deg,dest_pos_m,dest_angles_deg=None)->recommended_arm,requires_handoff; plan_joint_safe_return(target_robot_state)->valid,stages",
    "align_feature_to_target(grasp_pos_m,grasp_angles_deg,feature_point_m,feature_axis,target_point_m,target_axis) -> target_grasp_position_m,target_pos_m,target_angles_deg,feature_pos_err_m,feature_axis_err_deg (m,deg)",
    "se3_screw(start_pose,axis,pivot_m=None,angle_rad=0.0,distance_m=0.0,pitch_m_per_rad=0.0,n_steps=4) -> list[{s,grasp_position_m,angles_deg}] (m,rad,deg)",
    "kwargs: min_conf=0.1 (surface_extrema/cavity_profile=0.05; 0-1), robot=False, xyz=False",
]


def enabled(runtime) -> bool:
    cfg = getattr(runtime, "config", runtime)
    features = getattr(cfg, "features", cfg)
    return bool(getattr(features, "scene3d", False))


def applies(runtime, request):
    cfg = getattr(runtime, "config", runtime)
    features = getattr(cfg, "features", cfg)
    active = enabled(runtime) or bool(getattr(features, "scene3d_observe", False))
    return active and isinstance(request, dict) and request.get("kind") == KIND


def workspace_modules(runtime):
    """Files copied into each program job workspace when the switch (or features.scene3d_observe) is on."""
    cfg = getattr(runtime, "config", runtime)
    features = getattr(cfg, "features", cfg)
    return (WORKSPACE_MODULE,) if enabled(runtime) or bool(getattr(features, "scene3d_observe", False)) else ()


def _ik_staging(profile):
    """``(profile, takes grasp_angles, packaged kinematics facts or None)`` of ``profile`` (an EmbodimentProfile, or
    a runtime / None resolved by ``embodiment_of``): the inputs of the ik_prefilter.py staging decision."""
    from ...core.embodiment import EmbodimentProfile
    from ...robot.embodiments.arx_x5.profile import physical_facts

    eff_profile = profile if isinstance(profile, EmbodimentProfile) else embodiment_of(profile)
    kin_facts = None
    try:
        raw_facts = physical_facts(str(eff_profile.robot_model_id).replace("-", "_"))
        if isinstance(raw_facts, dict) and isinstance(raw_facts.get("kinematics"), dict):
            kin_facts = raw_facts["kinematics"]
    except Exception:
        kin_facts = None
    supports_6r_ik = (
        "grasp_angles" in eff_profile.pose_inputs
        and isinstance(kin_facts, dict)
        and len(kin_facts.get("arm_joint_order") or ()) == 6
    )
    return eff_profile, supports_6r_ik, kin_facts


def ik_prefilter_staged(profile=None) -> bool:
    """Whether ``copy_workspace_modules`` stages ik_prefilter.py with scene3d_helpers for ``profile``: it takes
    grasp_angles (6R IK input) and has packaged kinematics. False = the helpers' IK functions raise there, so the
    scene3d reply lists ``scene3d_helpers.api_for(False)``."""
    _, supports_6r_ik, kin_facts = _ik_staging(profile)
    return supports_6r_ik and kin_facts is not None


def copy_workspace_modules(modules, workspace, *, profile=None) -> bool:
    """Copy requested perception helper modules (plus ``scene3d_helpers`` and profile-matched kinematics companions) into ``workspace``."""
    from pathlib import Path
    import shutil

    workspace = Path(workspace)
    ext_dir = Path(__file__).parent
    ni_rua_root = Path(__file__).resolve().parents[2]
    core_pose_math = ni_rua_root / "core" / "pose_math.py"
    eff_profile, supports_6r_ik, kin_facts = _ik_staging(profile)
    kin_json_name = "_x5_kinematics.json" if eff_profile.robot_model_id == "arx-x5" else "_kinematics.json"
    has_packaged_kinematics = kin_facts is not None
    copied_scene3d = False
    for module in modules:
        shutil.copyfile(ext_dir / module, workspace / module)
        if module == WORKSPACE_MODULE:
            copied_scene3d = True
            for extra_mod in WORKSPACE_COMPANION_MODULES:
                if (
                    extra_mod == "ik_prefilter.py"
                    and not (supports_6r_ik and has_packaged_kinematics)
                    and "ik_prefilter.py" not in modules
                ):
                    continue
                extra_path = ext_dir / extra_mod
                if extra_path.is_file():
                    shutil.copyfile(extra_path, workspace / extra_mod)
                if extra_mod == "pose_math.py" and core_pose_math.is_file():
                    shutil.copyfile(core_pose_math, workspace / "_pose_math_core.py")
                if extra_mod == "ik_prefilter.py" and (workspace / "ik_prefilter.py").is_file():
                    kin_src = ni_rua_root / "robot" / "model" / "kinematics.py"
                    reach_src = ni_rua_root / "robot" / "motion" / "reachability.py"
                    if kin_src.is_file():
                        shutil.copyfile(kin_src, workspace / "kinematics.py")
                    if reach_src.is_file():
                        shutil.copyfile(reach_src, workspace / "_reachability.py")
                    if kin_facts is not None:
                        payload = {
                            "joints": kin_facts["joints"],
                            "arm_joint_order": kin_facts["arm_joint_order"],
                            "end_effector_link": kin_facts.get("end_effector_link") or kin_facts.get("tcp_link"),
                        }
                        if "base_link" in kin_facts:
                            payload["base_link"] = kin_facts["base_link"]
                        if "default_mounts" in kin_facts:
                            payload["default_mounts"] = kin_facts["default_mounts"]
                        (workspace / kin_json_name).write_text(json.dumps(payload), encoding="utf-8")
    return copied_scene3d


def schema_branch():
    return dict(type="object", properties=dict(
        kind=dict(type="string", const=KIND), observation_id=dict(type="string", minLength=1),
        camera_id=dict(type="string", minLength=1), voxel_m=dict(type="number", minimum=0.002, maximum=0.05),
        include_wrist_views=dict(type="boolean"),
        include_progress_wrist_views=dict(type="boolean"),
        artifact=dict(type="string", minLength=1),
        queries=dict(type="array", minItems=1, maxItems=32, items=dict(type="object"))),
        required=["kind"], minProperties=2, additionalProperties=False)


def extend_definition(geometry_schema, descriptions):
    """Model-facing schema change, applied only when the switch is on."""
    geometry_schema["oneOf"].append(schema_branch())
    return descriptions


def kind_sentence(profile=None):
    """Catalogue help of kind scene3d for ``profile`` (None = the default ARX-X5 robot): default camera =
    ``overview_camera``, frame = ``frame_id``."""
    profile = profile if profile is not None else embodiment_of(None)
    camera = profile.overview_camera
    return _KIND_SENTENCE.format(camera="default " + camera if camera else "camera_id required", frame_id=profile.frame_id)


def describe_branch(geometry_schema, profile=None):
    """Branch-specific field help; ``profile`` = the run's EmbodimentProfile (None = the default ARX-X5 robot)."""
    p = profile if profile is not None else embodiment_of(None)
    wrists = list(p.wrist_cameras.values())
    wrist_text = (
        f"Fuse calibrated wrist RGB views ({'/'.join(wrists)}) via dynamic FK T_world_camera (default false)."
        if wrists
        else "Fuse calibrated wrist RGB views via dynamic FK T_world_camera (default false)."
    )
    period_hz = f"{int(p.control_hz) if float(p.control_hz).is_integer() else p.control_hz} Hz" if p.clock != "simulated_steps" else "5 Hz"
    progress_text = (
        f"Also fuse quasi-static (<=0.8 rad/s, >=0.04 m baseline, <=0.004 m jaw change) {period_hz} wrist RGB views "
        "cached during the last action for occluded cavities/surfaces (default false)."
    )
    for branch in geometry_schema["oneOf"]:
        if branch.get("properties", {}).get("kind", {}).get("const") == KIND:
            branch["properties"]["kind"]["description"] = kind_sentence(p)
            for field, text in FIELD_DESCRIPTIONS.items():
                if field == "include_wrist_views":
                    branch["properties"][field]["description"] = wrist_text
                elif field == "include_progress_wrist_views":
                    branch["properties"][field]["description"] = progress_text
                else:
                    branch["properties"][field]["description"] = text


class ServiceError(RuntimeError):
    pass


class ImageCacheMiss(ServiceError):
    """Raised when a sha256-only ``/points`` request misses ``CACHE`` or hits a server requiring ``image_b64``."""

    def __init__(self, message: str = "image_cache_miss", *, legacy: bool = False):
        super().__init__(message)
        self.legacy = bool(legacy)


_STDLIB_URLOPEN = urllib.request.urlopen


def _is_legacy_missing_image_error(endpoint: str, http_status: int | None, answer: object) -> bool:
    if isinstance(answer, dict):
        err_text = str(answer.get("error") or "")
        reason = str(answer.get("reason") or "")
        if "image_b64" in err_text or "image_b64" in reason:
            return True
    return http_status == 500 and get_endpoint_sha_mode(str(endpoint or "")) == "unknown"


def _validate_service_points_payload(answer: dict) -> None:
    """校验上游 `/points` 返回字典的必要字段与有限数值类型。"""
    if not isinstance(answer.get("npz_b64"), str) or not answer["npz_b64"]:
        raise ServiceError("missing_points")
    for int_field in ("n_points", "n_above_support", "n_robot"):
        val = answer.get(int_field)
        if val is not None and (isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val))):
            raise ServiceError(f"malformed_service_payload:{int_field}")
    for float_field in ("voxel_m", "support_z_m", "multiview_residual_m"):
        val = answer.get(float_field)
        if val is not None and (isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val))):
            raise ServiceError(f"malformed_service_payload:{float_field}")
    for box_field in ("bbox_m", "bbox_above_support_m"):
        box = answer.get(box_field)
        if box is not None:
            if (
                not isinstance(box, (list, tuple))
                or len(box) != 2
                or any(
                    not isinstance(pt, (list, tuple))
                    or len(pt) != 3
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) for v in pt)
                    for pt in box
                )
            ):
                raise ServiceError(f"malformed_service_payload:{box_field}")


def query_service(endpoint, body, timeout_s, slack_s=5.0):
    import contextlib
    sha_only = isinstance(body, dict) and "image_b64" not in body and bool(body.get("image_sha256"))
    payload = json.dumps(body).encode()
    http_status: int | None = None
    target_info = (
        parse_http_endpoint(endpoint, "/points")
        if urllib.request.urlopen is _STDLIB_URLOPEN and isinstance(endpoint, str)
        else None
    )
    if target_info is not None:
        host, port, target = target_info
        try:
            http_status, answer = post_json_keepalive(
                host,
                port,
                target,
                payload,
                timeout_s + slack_s,
                service_error_cls=ServiceError,
            )
        except ServiceError as error:
            if sha_only and str(error) == "http_500" and get_endpoint_sha_mode(str(endpoint or "")) == "unknown":
                raise ImageCacheMiss("http_500", legacy=True) from None
            raise
    else:
        request = urllib.request.Request(
            endpoint.rstrip("/") + "/points",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s + slack_s) as response:
                http_status = getattr(response, "status", 200)
                answer = json.loads(response.read())
        except urllib.error.HTTPError as error:
            http_status = error.code
            with contextlib.closing(error):
                raw_err = error.read()
            try:
                answer = json.loads(raw_err)
            except ValueError:
                if sha_only and error.code == 500 and get_endpoint_sha_mode(str(endpoint or "")) == "unknown":
                    raise ImageCacheMiss("http_500", legacy=True) from None
                raise ServiceError("http_" + str(error.code)) from None
        except (urllib.error.URLError, OSError, ValueError) as error:
            raise ServiceError(type(error).__name__ + ":" + str(getattr(error, "reason", error))[:200]) from None
    if not isinstance(answer, dict) or answer.get("status") != "ok":
        if sha_only:
            if isinstance(answer, dict) and (
                answer.get("reason") == "image_cache_miss" or answer.get("error") == "image_cache_miss"
            ):
                raise ImageCacheMiss("image_cache_miss", legacy=False)
            if _is_legacy_missing_image_error(str(endpoint or ""), http_status, answer):
                raise ImageCacheMiss(
                    str((answer if isinstance(answer, dict) else {}).get("error", "legacy_missing_image_b64"))[:200],
                    legacy=True,
                )
        raise ServiceError(str((answer if isinstance(answer, dict) else {}).get("error", "service_error"))[:200])
    _validate_service_points_payload(answer)
    note_server_response(endpoint, answer)
    return answer


_DEFAULT_QUERY_SERVICE = query_service


def _r(values, digits=3):
    return None if values is None else [round(float(v), digits) for v in values]


def _resolve_cloud_artifact(runtime, ref):
    """解析点云 artifact 引用（支持完整 artifact:<sha256>-... 或短别名 artifact:ref_N / ref_N）。"""
    if not isinstance(ref, str) or not ref.strip():
        raise ValueError("invalid_scene3d_artifact_ref")
    store = getattr(runtime, "store", None)
    scene = getattr(runtime, "scene", None)
    resolver = getattr(store, "resolve_artifact", None) or getattr(scene, "resolve_artifact", None)
    if resolver is None:
        raise ValueError("scene3d_artifact_resolver_unavailable")
    candidate = ref.strip()
    try:
        info, data = resolver(candidate)
    except Exception:
        short = candidate.split(":", 1)[-1] if candidate.startswith("artifact:") else candidate
        read_state = getattr(store, "read_state", None)
        record = read_state("actor_reference", short) if callable(read_state) else None
        target = (record.get("reference") or record.get("value")) if isinstance(record, dict) else None
        if not isinstance(target, str) or not target:
            raise ValueError(f"unknown_scene3d_artifact:{ref}") from None
        info, data = resolver(target)
    return data


def _extract_region(q):
    if "region" in q and q["region"] is not None:
        return q["region"]
    if "lo" in q and "hi" in q:
        return (q["lo"], q["hi"])
    region = {k: q[k] for k in ("center", "radius", "polygon", "z_min", "z_max") if k in q}
    if "center" not in region and ("center_xy" in q or "xy" in q):
        region["center"] = q.get("center_xy") if "center_xy" in q else q.get("xy")
    return region


def _resolve_observation_id(runtime, observation_id=None, q=None):
    if isinstance(q, dict) and q.get("observation_id"):
        return q["observation_id"]
    if observation_id:
        return observation_id
    latest = getattr(runtime, "latest", None)
    if isinstance(latest, dict):
        if latest.get("observation_id"):
            return latest["observation_id"]
        obs = latest.get("observation")
        if isinstance(obs, dict) and obs.get("observation_id"):
            return obs["observation_id"]
        if isinstance(obs, str) and obs:
            return obs
    store = getattr(runtime, "store", None)
    records_fn = getattr(store, "records", None)
    if callable(records_fn):
        try:
            obs_list = records_fn("observation")
            if obs_list and isinstance(obs_list[-1], dict):
                return obs_list[-1].get("observation_id")
        except Exception:
            pass
    return None


def _resolve_arm_grasp_center(runtime, arm, observation_id=None, q=None):
    """Grasp centre (m, observation frame) of ``arm`` for in_hand_axis(arm=...): the observation's measured
    ``grasp_pose.position_m`` when present, else the profile's grasp reference of the measured TCP pose
    (``geometry.grasp_reference``: X5 pad centre 0.1501 m along link6 +x; SO-101 CAD inner-fingertip midpoint at the
    measured jaw angle). Never substitutes a different observation's state or a non-finite/offset flange pose."""
    from ...robot.model.geometry import grasp_reference
    profile = embodiment_of(runtime)
    explicit_obs_id = (q.get("observation_id") if isinstance(q, dict) and q.get("observation_id") else None) or observation_id
    obs_id = _resolve_observation_id(runtime, observation_id=observation_id, q=q)
    latest = getattr(runtime, "latest", None)
    latest_obs = None
    if isinstance(latest, dict):
        latest_obs = latest if ("observation_id" in latest or "robot_state" in latest or "robot_state_ref" in latest) else latest.get("observation")
    latest_obs_id = latest_obs.get("observation_id") if isinstance(latest_obs, dict) else None
    strict_historical = bool(explicit_obs_id and explicit_obs_id != latest_obs_id)

    obs = None
    if obs_id and getattr(runtime, "scene", None) is not None:
        try:
            obs = runtime.scene.observation(runtime.identity, obs_id)
        except Exception:
            obs = None
    if not isinstance(obs, dict):
        if strict_historical:
            return None
        obs = latest_obs
    if not isinstance(obs, dict):
        return None

    robot_state = obs.get("robot_state")
    if not isinstance(robot_state, dict) and obs.get("robot_state_ref"):
        resolve_json = getattr(getattr(runtime, "scene", None), "resolve_json", None)
        if callable(resolve_json):
            try:
                robot_state = resolve_json(obs["robot_state_ref"])
            except Exception:
                robot_state = None
        if not isinstance(robot_state, dict):
            store = getattr(runtime, "store", None)
            resolve_art = getattr(store, "resolve_artifact", None)
            if callable(resolve_art):
                try:
                    _, state_bytes = resolve_art(obs["robot_state_ref"])
                    robot_state = json.loads(state_bytes)
                except Exception:
                    robot_state = None
    # 仅当目标观测即当前最新观测且存储记录未携带 arms 时，才允许回退读取同观测的内联 robot_state
    if (
        not strict_historical
        and not isinstance((robot_state or {}).get("arms"), dict)
        and isinstance(latest_obs, dict)
        and isinstance(latest_obs.get("robot_state"), dict)
        and (latest_obs_id is None or obs_id is None or latest_obs_id == obs_id)
    ):
        robot_state = latest_obs["robot_state"]

    arm_state = ((robot_state or {}).get("arms") or {}).get(arm) or {}
    gp = (arm_state.get("grasp_pose") or {}).get("position_m")
    if gp is not None:
        try:
            gp_arr = np.asarray(gp, dtype=np.float64)
        except (ValueError, TypeError):
            return None
        if gp_arr.shape != (3,) or not np.isfinite(gp_arr).all():
            return None
        return [float(v) for v in gp_arr]

    ee = arm_state.get("ee_pose") or {}
    pos = ee.get("position_m")
    if not isinstance(pos, (list, tuple)) or len(pos) != 3:
        return None
    try:
        origin = np.asarray(pos, dtype=np.float64)
    except (ValueError, TypeError):
        return None
    if origin.shape != (3,) or not np.isfinite(origin).all():
        return None

    same_tcp = ee.get("tcp_config_id") in (None, profile.tcp_config_id)
    if not same_tcp:
        return None

    requires_tcp_offset = (
        profile.gripper.grasp_reference == "tcp_offset"
        and float(getattr(profile.gripper, "grasp_reference_offset_m", 0.0) or 0.0) > 0.0
    )
    quat = ee.get("quaternion_xyzw")
    if isinstance(quat, (list, tuple)) and len(quat) == 4:
        try:
            q_arr = np.asarray(quat, dtype=np.float64)
            q_norm = float(np.linalg.norm(q_arr))
            if q_arr.shape == (4,) and np.isfinite(q_arr).all() and q_norm > 1e-6:
                from ...robot.model.roboprobe_pose import quat_to_matrix
                x, y, z, w = (q_arr / q_norm).tolist()
                rotation = quat_to_matrix([w, x, y, z])
                jaw_rad = None
                jaw = (profile.jaw_joint_names.get(arm) or (None,))[0]
                names, joints = arm_state.get("joint_names") or [], arm_state.get("q_rad") or []
                if (
                    profile.gripper.grasp_reference != "tcp_offset"
                    and arm_state.get("source_kind") == "measurement"
                    and len(names) == len(joints)
                    and names.count(jaw) == 1
                ):
                    cand_jaw = float(joints[names.index(jaw)])
                    if math.isfinite(cand_jaw):
                        jaw_rad = cand_jaw
                located = grasp_reference(profile.gripper, origin, rotation, jaw_rad)
                if located is not None:
                    loc_pt = np.asarray(located[0], dtype=np.float64)
                    if loc_pt.shape == (3,) and np.isfinite(loc_pt).all():
                        return [float(v) for v in loc_pt]
        except Exception:
            if requires_tcp_offset:
                return None
    if requires_tcp_offset:
        # 严禁在指尖前移偏移量未知或姿态四元数缺失时将 link6 法兰坐标（偏后 15cm）冒充指尖抓取中心返回
        return None
    return [float(v) for v in origin]


def _execute_one_cloud_query(runtime, cloud, q, observation_id=None):
    from . import scene3d_helpers as s3
    if not isinstance(q, dict):
        raise ValueError("scene3d_query_item_must_be_object")
    op = str(q.get("op") or q.get("fn") or "info").strip()
    common = {}
    if "min_conf" in q:
        common["min_conf"] = float(q["min_conf"])
    if "robot" in q:
        common["robot"] = bool(q["robot"])
    if "xyz" in q:
        common["xyz"] = bool(q["xyz"])
    if op == "info":
        return dict(op=op, **cloud.info())
    if op == "table_z":
        xy = q.get("xy") if "xy" in q else (q.get("center") if "center" in q else q.get("point"))
        radius = float(q.get("radius", 0.08))
        res = cloud.table_z(xy=xy, radius=radius)
        return dict(op=op, **(res if isinstance(res, dict) else {"table_z": res}))
    if op == "points_in_box":
        lo = q.get("lo") if "lo" in q else (q.get("min_m") if "min_m" in q else q.get("min"))
        hi = q.get("hi") if "hi" in q else (q.get("max_m") if "max_m" in q else q.get("max"))
        if lo is None or hi is None:
            raise ValueError("points_in_box_requires_lo_and_hi")
        return dict(op=op, **cloud.points_in_box(lo, hi, **common))
    if op == "points_in_cylinder":
        center = q.get("center_xy") if "center_xy" in q else (q.get("center") if "center" in q else q.get("xy"))
        if center is None or "radius" not in q:
            raise ValueError("points_in_cylinder_requires_center_and_radius")
        return dict(op=op, **cloud.points_in_cylinder(
            center, float(q["radius"]), z_min=q.get("z_min"), z_max=q.get("z_max"), **common))
    if op == "top_z":
        region = _extract_region(q)
        if not region:
            raise ValueError("top_z_requires_region")
        pct = float(q.get("q", 98))
        return dict(op=op, **cloud.top_z(
            region, q=pct, min_conf=float(q.get("min_conf", 0.1)), robot=bool(q.get("robot", False))))
    if op == "nearest":
        p = q.get("p") if "p" in q else (q.get("point") if "point" in q else (q.get("xyz") if "xyz" in q else (q.get("xy") if "xy" in q else q.get("center"))))
        if p is None:
            raise ValueError("nearest_requires_point_p")
        return dict(op=op, **cloud.nearest(
            p, k=int(q.get("k", 5)), min_conf=float(q.get("min_conf", 0.1)), robot=bool(q.get("robot", False))))
    if op == "cluster_at_pixel":
        pixel = q.get("pixel") if "pixel" in q else q.get("uv")
        if isinstance(pixel, (list, tuple)) and len(pixel) == 2:
            u, v = pixel[0], pixel[1]
        elif "u" in q and "v" in q:
            u, v = q["u"], q["v"]
        else:
            raise ValueError("cluster_at_pixel_requires_pixel_or_u_v")
        return dict(op=op, **cloud.cluster_at_pixel(u, v, radius=float(q.get("radius", 0.015)), **common))
    if op in ("in_hand_axis", "in_hand_geometry"):
        center = (q.get("grasp_center_m") if "grasp_center_m" in q else
                  (q.get("grasp_position_m") if "grasp_position_m" in q else
                   (q.get("center") if "center" in q else q.get("p"))))
        if center is None and q.get("arm") in embodiment_of(runtime).arms:
            center = _resolve_arm_grasp_center(runtime, q["arm"], observation_id=observation_id, q=q)
        if center is None:
            raise ValueError("in_hand_axis_requires_grasp_center_m_or_arm")
        return dict(op=op, **cloud.in_hand_axis(
            center, radius=float(q.get("radius", 0.065)), min_conf=float(q.get("min_conf", 0.1)),
            above=float(q.get("above", q.get("z_clearance_above_table", 0.015))), xyz=bool(q.get("xyz", False))))
    if op == "surface_extrema":
        region = _extract_region(q)
        if not region:
            region = q.get("center") if "center" in q else (q.get("xy") if "xy" in q else q.get("p"))
        if region is None:
            raise ValueError("surface_extrema_requires_region_or_center")
        return dict(op=op, **cloud.surface_extrema(
            region, radius=float(q.get("radius", 0.04)), min_conf=float(q.get("min_conf", 0.05)),
            robot=bool(q.get("robot", False)), xyz=bool(q.get("xyz", False))))
    if op == "triangulate_rays":
        p1 = q.get("pixel_a") or q.get("uv1") or ([q["u1"], q["v1"]] if "u1" in q and "v1" in q else None)
        p2 = q.get("pixel_b") or q.get("uv2") or ([q["u2"], q["v2"]] if "u2" in q and "v2" in q else None)
        if not (isinstance(p1, (list, tuple)) and len(p1) == 2 and isinstance(p2, (list, tuple)) and len(p2) == 2):
            raise ValueError("triangulate_rays_requires_pixel_pairs")
        K1, T1, K2, T2 = q.get("K1"), q.get("T1"), q.get("K2"), q.get("T2")
        eff_obs_id = _resolve_observation_id(runtime, observation_id=observation_id, q=q)
        if (K1 is None or T1 is None or K2 is None or T2 is None) and eff_obs_id:
            from ...robot.model.geometry import GeometryTools
            from .sensor_frame_contract import list_wrist_camera_ids, select_primary_fixed_camera_id
            profile = embodiment_of(runtime)
            tools = GeometryTools(runtime.scene, settings=runtime.image_settings, profile=profile)
            full = {"observation_id": eff_obs_id, **runtime.identity, "schema_version": "0.1"}
            obs_doc = runtime.scene.observation(runtime.identity, eff_obs_id) if runtime.scene is not None else None
            resolve_json = getattr(runtime.scene, "resolve_json", None)
            cam_a = q.get("camera_id") or default_camera(runtime, obs_doc) or select_primary_fixed_camera_id(obs_doc, resolve_json=resolve_json)
            wrist_ids = list_wrist_camera_ids(obs_doc, resolve_json=resolve_json)
            profile_wrists = [profile.wrist_cameras[arm] for arm in profile.arms if arm in profile.wrist_cameras]
            cam_b = q.get("camera_id_b") or q.get("camera_id_2") or (wrist_ids[-1] if wrist_ids else (profile_wrists[-1] if profile_wrists else None))
            if cam_b is None:
                raise ValueError("triangulate_rays_requires_camera_id_b")
            c1 = tools.camera(full, eff_obs_id, cam_a)
            c2 = tools.camera(full, eff_obs_id, cam_b)
            K1, T1, K2, T2 = c1["K"], c1["T"], c2["K"], c2["T"]
        if K1 is None or T1 is None or K2 is None or T2 is None:
            raise ValueError("triangulate_rays_requires_K_T_or_observation_cameras")
        return dict(op=op, **s3.triangulate_rays(p1[0], p1[1], K1, T1, p2[0], p2[1], K2, T2))
    if op in ("cavity_profile", "inspect_cavity"):
        target = q.get("target_id") or q.get("object") or _extract_region(q)
        if not target:
            target = q.get("center") if "center" in q else (q.get("xy") if "xy" in q else q.get("p"))
        if not target:
            raise ValueError("cavity_profile_requires_target_id_or_region")
        eff_obs_id = _resolve_observation_id(runtime, observation_id=observation_id, q=q)
        objs = q.get("objects") or q.get("object_table")
        if objs is None and eff_obs_id:
            from .obs3d import attached_table
            objs = attached_table(runtime, eff_obs_id)
        return dict(
            op=op,
            **s3.cavity_profile(
                target,
                radius=float(q.get("radius", q.get("radius_m", 0.06))),
                cloud=cloud,
                objects=objs,
            ),
        )
    raise ValueError(f"unsupported_scene3d_op:{op}")


def execute_cloud_queries(runtime, data_bytes, raw_queries, observation_id=None):
    from . import scene3d_helpers as s3
    cloud = data_bytes if isinstance(data_bytes, s3.Cloud) else s3.from_bytes(data_bytes)
    results = []
    for item in raw_queries:
        try:
            results.append(_execute_one_cloud_query(runtime, cloud, item, observation_id=observation_id))
        except Exception as error:
            err_text = str(error)
            formatted = err_text[:200] if isinstance(error, ValueError) and err_text else f"{type(error).__name__}: {err_text}"[:200]
            results.append(dict(op=(item.get("op") if isinstance(item, dict) else None),
                                status="error", error=formatted))
    return cloud, results


WRIST_CAMERA_IDS = ("cam_left_wrist", "cam_right_wrist")


def _extract_wrist_camera_from_bundle(runtime, observation: dict, wrist_id: str) -> dict | None:
    """从 `ObservationBundle` 解析腕部相机的无畸变内参 `K`、RGB 图像与动态外参 `T_world_camera`。

    同时支持 `camera_calibration["cameras"]` 与 `camera_pose_sample["cameras"]` 以 `camera_id` 为键的 `dict` 结构，
    优先取动态位姿采样 `camera_pose_sample` / `pose_sample_ref`，缺失时兼容回退 `camera_calibration`。
    """
    frame = next(
        (item for item in (observation.get("cameras") or []) if isinstance(item, dict) and item.get("camera_id") == wrist_id),
        None,
    )
    if frame is None:
        return None
    if frame.get("rgb_ref"):
        _, image_bytes = runtime.scene.resolve_artifact(frame["rgb_ref"])
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
    elif isinstance(frame.get("image_b64"), str) and frame["image_b64"]:
        image_b64 = frame["image_b64"]
    else:
        return None

    resolve_json = getattr(runtime.scene, "resolve_json", None)
    cal_doc = frame.get("calibration") or observation.get("camera_calibration")
    cal_ref = frame.get("calibration_ref") or observation.get("camera_calibration_ref")
    if not isinstance(cal_doc, dict) and cal_ref and callable(resolve_json):
        cal_doc = resolve_json(cal_ref)
    if not isinstance(cal_doc, dict):
        return None

    cams_cal = cal_doc.get("cameras")
    if isinstance(cams_cal, dict):
        cal_entry = cams_cal.get(wrist_id)
    elif isinstance(cams_cal, list):
        cal_entry = next((c for c in cams_cal if isinstance(c, dict) and c.get("camera_id") == wrist_id), None)
    else:
        cal_entry = cal_doc if (cal_doc.get("camera_id") in (None, wrist_id) and "K" in cal_doc) else None
    if not isinstance(cal_entry, dict) or cal_entry.get("K") is None:
        return None

    dist = cal_entry.get("distortion")
    d_coeffs = (
        dist.get("coefficients", [])
        if isinstance(dist, dict)
        else (dist if isinstance(dist, (list, tuple, np.ndarray)) else [])
    )
    if len(d_coeffs) > 0 and np.any(np.abs(np.asarray(d_coeffs, dtype=float)) > 1e-12):
        return None

    K = np.asarray(cal_entry["K"], dtype=float).reshape(3, 3)
    if not np.all(np.isfinite(K)) or K[0, 0] <= 0.0 or K[1, 1] <= 0.0:
        return None

    T_raw = None
    pose_doc = frame.get("pose_sample") or observation.get("camera_pose_sample")
    pose_ref = frame.get("pose_sample_ref") or observation.get("camera_pose_sample_ref")
    if not isinstance(pose_doc, dict) and pose_ref and callable(resolve_json):
        pose_doc = resolve_json(pose_ref)
    if isinstance(pose_doc, dict):
        cams_pose = pose_doc.get("cameras")
        if isinstance(cams_pose, dict) and isinstance(cams_pose.get(wrist_id), dict):
            T_raw = cams_pose[wrist_id].get("T_world_camera")
        elif isinstance(cams_pose, list):
            pe = next((p for p in cams_pose if isinstance(p, dict) and p.get("camera_id") == wrist_id), None)
            T_raw = pe.get("T_world_camera") if isinstance(pe, dict) else None
        elif pose_doc.get("camera_id") in (None, wrist_id):
            T_raw = pose_doc.get("T_world_camera")

    if T_raw is None:
        state = observation.get("robot_state")
        if not isinstance(state, dict) and observation.get("robot_state_ref") and callable(resolve_json):
            state = resolve_json(observation["robot_state_ref"])
        if isinstance(state, dict):
            cp = (state.get("camera_poses") or {}).get(wrist_id)
            if isinstance(cp, dict):
                T_raw = cp.get("T_world_camera")

    if T_raw is None:
        T_raw = cal_entry.get("T_world_camera")
    if T_raw is None:
        return None

    T = np.asarray(T_raw, dtype=float).reshape(4, 4)
    if not np.all(np.isfinite(T)) or not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-4):
        return None
    if abs(float(np.linalg.det(T[:3, :3])) - 1.0) > 1e-2:
        return None

    return dict(
        camera_id=wrist_id,
        image_b64=image_b64,
        K=K.tolist(),
        T_world_camera=T.tolist(),
    )


def _wrist_view_overlaps_workspace(
    wrist_T: np.ndarray,
    main_K: np.ndarray | None = None,
    main_T: np.ndarray | None = None,
    main_hw: tuple[int, int] | None = None,
    support_z_m: float | None = None,
    *,
    max_range_m: float = 0.40,
) -> bool:
    """校验腕部相机外参有限正交且近场有效测距视线落在主相机前方工作扇区内（零台面高度走廊硬锁）。

    参数:
        wrist_T: 腕部相机位姿矩阵 `T_world_camera`，形状 `(4, 4)`，平移单位 `m`。
        main_K: 可选主相机内参矩阵 `(3, 3)`，像素单位 `px`。
        main_T: 可选主相机外参矩阵 `T_world_camera`，形状 `(4, 4)`，平移单位 `m`。
        main_hw: 可选主相机图像分辨率 `(height, width)`，单位 `px`。
        support_z_m: 保留兼容签名（单位 `m`），不再对腕部高度或视线高程做任何台面走廊裁剪。
        max_range_m: 腕部深度融合有效近场测距上限，单位 `m` (`> 0.05`)，默认 `0.40`。

    返回:
        若相机外参有限有效且近场视线 `[0.05, max_range_m]` 位于有效测距工作扇区内返回 `True`，否则返回 `False`。
    """
    _ = support_z_m
    Tw = np.asarray(wrist_T, dtype=np.float64).reshape(4, 4)
    max_r = float(max_range_m)
    if not (np.isfinite(Tw).all() and math.isfinite(max_r) and max_r > 0.05):
        return False
    rw = Tw[:3, :3]
    if abs(float(np.linalg.det(rw)) - 1.0) > 5e-2:
        return False
    cw = Tw[:3, 3]
    fwd = rw[:, 2]
    fwd_norm = float(np.linalg.norm(fwd))
    if fwd_norm < 1e-8:
        return False
    fwd = fwd / fwd_norm

    ds = np.linspace(0.05, max_r, 8)
    pts = cw[None, :] + ds[:, None] * fwd[None, :]
    if not np.isfinite(pts).all():
        return False

    if main_T is None or main_K is None or main_hw is None:
        return True

    Th = np.asarray(main_T, dtype=np.float64).reshape(4, 4)
    Kh = np.asarray(main_K, dtype=np.float64).reshape(3, 3)
    if not (np.isfinite(Th).all() and np.isfinite(Kh).all()):
        return False
    mh, mw = int(main_hw[0]), int(main_hw[1])
    if mh <= 0 or mw <= 0:
        return False

    # 沿主相机水平前向与右向校验方位角视锥重叠，不施加任何竖直高度走廊裁剪（支持高塔、立柜、挂架操作）
    c_main = Th[:3, 3]
    main_fwd = Th[:3, 2]
    horiz_fwd = np.array([main_fwd[0], main_fwd[1]], dtype=np.float64)
    horiz_norm = float(np.linalg.norm(horiz_fwd))
    if horiz_norm >= 1e-6:
        horiz_fwd = horiz_fwd / horiz_norm
        horiz_right = np.array([Th[0, 0], Th[1, 0]], dtype=np.float64)
        right_norm = float(np.linalg.norm(horiz_right))
        if right_norm >= 1e-6:
            horiz_right = horiz_right / right_norm
        else:
            horiz_right = np.array([horiz_fwd[1], -horiz_fwd[0]], dtype=np.float64)
        rel_xy = pts[:, :2] - c_main[None, :2]
        depth_fwd = rel_xy @ horiz_fwd
        lat_x = rel_xy @ horiz_right
        in_range = (depth_fwd >= 0.08) & (depth_fwd <= 2.00) & (np.abs(pts[:, 2] - float(c_main[2])) <= 2.00)
        d_safe = np.maximum(depth_fwd, 1e-6)
        u = Kh[0, 0] * (lat_x / d_safe) + Kh[0, 2]
        in_azimuth = in_range & (u >= 0.0) & (u < float(mw))
        return int(np.sum(in_azimuth)) >= 2

    # 主相机纯竖直俯视时（水平光轴投影接近 0），仅校验水平工作区半径与测距深度范围，不裁剪高于主相机的立柜/高架视角
    xy_dist = np.linalg.norm(pts[:, :2] - c_main[None, :2], axis=1)
    in_workspace = (xy_dist <= 1.20) & (np.abs(pts[:, 2] - float(c_main[2])) <= 2.00)
    return int(np.sum(in_workspace)) >= 2


def _collect_wrist_extra_views(
    runtime,
    tools,
    full: dict,
    observation: dict,
    main_camera_id: str,
    *,
    include_current_wrist_views: bool = True,
    include_progress_wrist_views: bool = False,
    filter_idle_by_frustum: bool = False,
    main_K: np.ndarray | None = None,
    main_T: np.ndarray | None = None,
    main_hw: tuple[int, int] | None = None,
    support_z_m: float | None = None,
) -> tuple[list[dict], list[str]]:
    """收集同观测随动腕部视图，并仅在显式启用 `include_progress_wrist_views=True` 时合并动作期准静态腕部视图。"""
    from .sensor_frame_contract import list_wrist_camera_ids
    from .wrist_progress_cache import (
        WristProgressCache,
        make_artifact_bytes_resolver,
        make_artifact_json_resolver,
    )

    extra_views: list[dict] = []
    progress_obs_ids: list[str] = []
    obs_id = full.get("observation_id")
    resolve_json = getattr(getattr(runtime, "scene", None), "resolve_json", None)

    if include_current_wrist_views:
        wrist_ids = list_wrist_camera_ids(observation, resolve_json=resolve_json)
        for wrist_id in wrist_ids:
            if wrist_id == main_camera_id:
                continue
            frame_doc = next(
                (item for item in (observation.get("cameras") or []) if isinstance(item, dict) and item.get("camera_id") == wrist_id),
                None,
            )
            if frame_doc is None:
                continue
            view = None
            has_formal_cal = bool(frame_doc.get("calibration_ref") or observation.get("camera_calibration_ref"))
            if obs_id and has_formal_cal:
                # 正式存储观测严格经过 GeometryTools.camera 验证（时钟对齐、FK 一致性、坐标系校验）；拒绝时绝不绕过
                try:
                    w_cam = tools.camera(full, obs_id, wrist_id)
                    if np.all(np.abs(w_cam["D"]) <= 1e-12) and w_cam["frame"].get("rgb_ref"):
                        w_T = np.asarray(w_cam["T"], dtype=np.float64)
                        if (
                            filter_idle_by_frustum
                            and main_K is not None
                            and main_T is not None
                            and main_hw is not None
                            and not _wrist_view_overlaps_workspace(w_T, main_K, main_T, main_hw, support_z_m)
                        ):
                            continue
                        _, w_bytes = runtime.scene.resolve_artifact(w_cam["frame"]["rgb_ref"])
                        view = dict(
                            camera_id=wrist_id,
                            image_b64=base64.b64encode(w_bytes).decode("ascii"),
                            K=np.asarray(w_cam["K"]).tolist(),
                            T_world_camera=w_T.tolist(),
                        )
                except (ValueError, KeyError):
                    view = None
            elif not has_formal_cal:
                try:
                    cand = _extract_wrist_camera_from_bundle(runtime, observation, wrist_id)
                    if (
                        cand is not None
                        and filter_idle_by_frustum
                        and main_K is not None
                        and main_T is not None
                        and main_hw is not None
                        and not _wrist_view_overlaps_workspace(
                            np.asarray(cand["T_world_camera"], dtype=np.float64),
                            main_K,
                            main_T,
                            main_hw,
                            support_z_m,
                        )
                    ):
                        cand = None
                    view = cand
                except (ValueError, KeyError):
                    view = None
            if view is not None:
                extra_views.append(view)

    run_profile = embodiment_of(runtime)
    allow_progress_cache = (run_profile.clock == "simulated_steps" and "grasp_angles" in run_profile.pose_inputs)
    if include_progress_wrist_views and allow_progress_cache and len(extra_views) < 6:
        progress_cache = getattr(runtime, "_wrist_progress_cache", None)
        if isinstance(progress_cache, WristProgressCache):
            overlap_fn = (
                (lambda t_mat: _wrist_view_overlaps_workspace(t_mat, main_K, main_T, main_hw, support_z_m))
                if (main_K is not None and main_T is not None and main_hw is not None)
                else None
            )
            cached_views, progress_obs_ids = progress_cache.select_extra_views(
                observation,
                resolve_bytes=make_artifact_bytes_resolver(runtime),
                resolve_json=make_artifact_json_resolver(runtime),
                main_camera_id=main_camera_id,
                existing_views=extra_views,
                overlap_fn=overlap_fn,
                max_views=6 - len(extra_views),
            )
            if cached_views:
                extra_views.extend(cached_views)
    return extra_views, progress_obs_ids


def scene3d_query(runtime, request, *, timeout_s=None, slack_s=5.0, auto_filter_wrist_views=False):
    """Returns (result, npz bytes or None). The caller stores the bytes and fills value.artifact."""
    from ...robot.model.geometry import GeometryTools
    from .object_table import robot_segments, LINK_RADIUS_M, persistent_tracking_enabled
    from .sensor_frame_contract import select_primary_fixed_camera_id
    features = runtime.config.features
    profile = embodiment_of(runtime)
    request = dict(request)
    requested_camera_id = request.get("camera_id")
    default_cam = default_camera(runtime)
    camera_id = requested_camera_id or default_cam
    obs_id = request.get("observation_id")
    include_wrist_views = bool(request.get("include_wrist_views", False))
    include_progress_wrist_views = bool(request.get("include_progress_wrist_views", False))
    raw_queries = list(request.get("queries") or ([] if request.get("query") is None else [request["query"]]))
    started = time.monotonic()
    result = {"schema_version": "0.1", **runtime.identity, "world_kind": "internal", "query_type": KIND,
              "observation_ids": [obs_id] if obs_id else [], "status": "supported", "frame_id": profile.frame_id,
              "unit": "m", "value": None}
    # Fast path: in-process query against an existing scene3d artifact or memoized obs3d point cloud.
    artifact_ref = request.get("artifact")
    if (
        artifact_ref is None
        and raw_queries
        and obs_id
        and requested_camera_id in (None, default_cam)
        and not request.get("voxel_m")
        and "include_wrist_views" not in request
        and "include_progress_wrist_views" not in request
    ):
        from .obs3d import _memo
        cached = _memo(runtime, obs_id).get("scene3d")
        if isinstance(cached, tuple) and len(cached) == 2 and isinstance(cached[0], dict):
            artifact_ref = cached[0].get("artifact")
    if artifact_ref is not None:
        import zipfile
        try:
            data = _resolve_cloud_artifact(runtime, artifact_ref)
            cloud, query_results = execute_cloud_queries(runtime, data, raw_queries or [{"op": "info"}], observation_id=obs_id)
            info = cloud.info()
        except (ServiceError, ValueError, OSError, zipfile.BadZipFile, EOFError) as error:
            result.update(status="unknown", reason="scene3d_unavailable:" + str(error)[:200])
            result["duration_s"] = time.monotonic() - started
            return result, None
        value = dict(source=SOURCE, measured=False, camera_id=camera_id, artifact=artifact_ref,
                     points=info["n"], voxel_m=round(float(info["voxel_m"]), 4),
                     table_z=round(float(info["table_z"]), 4), bbox=info["bbox"],
                     query_results=query_results)
        if request.get("query") is not None and len(query_results) == 1:
            value["query_result"] = query_results[0]
        if raw_queries and all(isinstance(r, dict) and r.get("status") == "error" for r in query_results):
            result.update(
                status="unknown",
                value=value,
                reason="all_cloud_queries_failed:" + str(query_results[0].get("error", "error"))[:160],
                assumptions=["VGGT-Omega monocular depth point cloud queried in-process via scene3d_helpers."],
            )
        else:
            result.update(status="conditional", value=value,
                          assumptions=["VGGT-Omega monocular depth point cloud queried in-process via scene3d_helpers."])
        result["duration_s"] = time.monotonic() - started
        return result, None
    full = {**request, "camera_id": camera_id, **runtime.identity, "schema_version": "0.1"}
    tools = GeometryTools(runtime.scene, settings=runtime.image_settings, profile=profile)
    limit = float(features.vggt_timeout_s if timeout_s is None else timeout_s)
    try:
        obs_id_req = request.get("observation_id")
        if not isinstance(obs_id_req, str) or not obs_id_req:
            raise ValueError("invalid_observation_id")
        try:
            observation = runtime.scene.observation(runtime.identity, obs_id_req)
        except KeyError as exc:
            raise ValueError("unknown_observation") from exc
        if not requested_camera_id:
            preferred = default_camera(runtime, observation)
            obs_cam_ids = {c.get("camera_id") for c in (observation.get("cameras") or []) if isinstance(c, dict)}
            camera_id = (
                preferred
                if preferred in obs_cam_ids
                else select_primary_fixed_camera_id(
                    observation, resolve_json=getattr(runtime.scene, "resolve_json", None)
                )
            )
            full["camera_id"] = camera_id
        camera = tools.camera(full, full["observation_id"], camera_id)
        try:
            _, image_bytes = runtime.scene.resolve_artifact(camera["frame"]["rgb_ref"])
        except KeyError as exc:
            raise ValueError("unknown_artifact") from exc
        rectified = rectified_cached(camera, image_bytes)   # pinhole service: distorted cameras are sent undistorted
        image_bytes = rectified.image_bytes
        segments = robot_segments(tools, full, observation, profile.gripper.mask_segment)
        voxel = float(request.get("voxel_m") or features.scene3d_voxel_m)
        support_z = effective_support_z_m(features, camera, profile)
        cache_key = make_image_cache_key(rectified.sha256, camera["K"], camera["T"], support_z)
        base_body = dict(
            K=np.asarray(camera["K"]).tolist(),
            T_world_camera=np.asarray(camera["T"]).tolist(),
            support_z_m=support_z,
            voxel_m=voxel,
            robot_segments_m=segments,
            robot_radius_m=LINK_RADIUS_M,
            params=dict(z_above=float(features.scene3d_z_above_m), z_below=float(features.scene3d_z_below_m)),
            timeout_s=limit,
        )
        ep_id = (getattr(runtime, "identity", None) or {}).get("episode_id")
        if persistent_tracking_enabled(runtime) and ep_id:
            base_body["session_id"] = str(ep_id)
        if include_wrist_views or include_progress_wrist_views:
            main_K_arr = np.asarray(camera["K"], dtype=np.float64)
            main_T_arr = np.asarray(camera["T"], dtype=np.float64)
            frame_doc = camera.get("frame") if isinstance(camera.get("frame"), dict) else {}
            main_h = int(frame_doc.get("height") or max(1, round(2.0 * float(main_K_arr[1, 2]))))
            main_w = int(frame_doc.get("width") or max(1, round(2.0 * float(main_K_arr[0, 2]))))
            extra_views, progress_obs_ids = _collect_wrist_extra_views(
                runtime,
                tools,
                full,
                observation,
                camera_id,
                include_current_wrist_views=include_wrist_views,
                include_progress_wrist_views=include_progress_wrist_views,
                filter_idle_by_frustum=bool(auto_filter_wrist_views),
                main_K=main_K_arr,
                main_T=main_T_arr,
                main_hw=(main_h, main_w),
                support_z_m=support_z,
            )
            if extra_views:
                base_body["extra_views"] = extra_views
            if progress_obs_ids:
                merged_segments = list(segments)
                for hist_obs_id in progress_obs_ids:
                    hist_obs = runtime.scene.observation(runtime.identity, hist_obs_id)
                    merged_segments.extend(
                        robot_segments(tools, {**full, "observation_id": hist_obs_id}, hist_obs, profile.gripper.mask_segment)
                    )
                unique_segs = {
                    json.dumps(np.round(np.asarray(seg), 4).tolist()): seg
                    for seg in merged_segments
                }
                segments = list(unique_segs.values())
                base_body["robot_segments_m"] = segments
        if base_body.get("session_id"):
            mark_active = getattr(runtime, "mark_perception_session_active", None)
            if callable(mark_active):
                mark_active(base_body["session_id"])
        use_std_transport = urllib.request.urlopen is _STDLIB_URLOPEN and query_service is _DEFAULT_QUERY_SERVICE
        if use_std_transport and should_try_points_sha256(features.vggt_endpoint, cache_key):
            sha_body = dict(base_body, image_sha256=rectified.sha256)
            t_call = time.monotonic()
            try:
                answer = query_service(features.vggt_endpoint, sha_body, limit, slack_s)
            except ImageCacheMiss as miss:
                evict_cached_image_key(features.vggt_endpoint, cache_key, legacy_unsupported=miss.legacy)
                rem_limit = max(0.05, limit - (time.monotonic() - t_call))
                full_body = dict(
                    base_body,
                    image_b64=base64.b64encode(image_bytes).decode("ascii"),
                    timeout_s=rem_limit,
                )
                answer = query_service(features.vggt_endpoint, full_body, rem_limit, slack_s)
                note_image_uploaded(features.vggt_endpoint, cache_key, answer=answer)
        else:
            full_body = dict(base_body, image_b64=base64.b64encode(image_bytes).decode("ascii"))
            answer = query_service(features.vggt_endpoint, full_body, limit, slack_s)
            if use_std_transport:
                note_image_uploaded(features.vggt_endpoint, cache_key, answer=answer)
        data = map_npz_uv(base64.b64decode(answer["npz_b64"]), rectified)   # uv in the original image
        from . import scene3d_helpers as s3
        parsed_cloud = s3.from_bytes(data)
        bbox, above = answer.get("bbox_m"), answer.get("bbox_above_support_m")
        value = dict(source=SOURCE, measured=False, camera_id=camera_id, artifact=None, points=int(answer.get("n_points", 0)),
                     points_above_table=int(answer.get("n_above_support", 0)), near_robot=int(answer.get("n_robot", 0)),
                     voxel_m=round(float(answer.get("voxel_m", voxel)), 4),
                     table_z=round(float(answer.get("support_z_m", support_z)), 4),
                     bbox=[_r(bbox[0]), _r(bbox[1])] if bbox else None,
                     bbox_above_table=[_r(above[0]), _r(above[1])] if above else None,
                     usage=USAGE, api=list(s3.api_for(ik_prefilter_staged(profile))))
        if isinstance(answer.get("fused_views"), list):
            value["fused_views"] = [str(v) for v in answer["fused_views"]]
        if answer.get("multiview_residual_m") is not None:
            value["multiview_residual_m"] = round(float(answer["multiview_residual_m"]), 4)
        if raw_queries:
            _, query_results = execute_cloud_queries(runtime, parsed_cloud, raw_queries, observation_id=obs_id)
            value["query_results"] = query_results
            if request.get("query") is not None and len(query_results) == 1:
                value["query_result"] = query_results[0]
    except (ServiceError, ValueError, OSError, EOFError) as error:
        err_text = str(error)[:200]
        if isinstance(error, ValueError) and (err_text.startswith("invalid_") or err_text.startswith("voxel_m")):
            result.update(status="invalid_input", reason=err_text)
        else:
            result.update(status="unknown", reason="scene3d_unavailable:" + err_text)
        result["duration_s"] = time.monotonic() - started
        return result, None
    if raw_queries and all(isinstance(r, dict) and r.get("status") == "error" for r in value.get("query_results", ())):
        result.update(
            status="unknown",
            value=value,
            reason="all_cloud_queries_failed:" + str(value["query_results"][0].get("error", "error"))[:160],
            frame_ids=[camera["frame"]["frame_id"]],
            assumptions=["VGGT-Omega monocular depth of one image, back-projected with the calibrated camera and "
                         "scaled so the fitted table plane lies at table_z; no per-object correction."],
        )
    else:
        result.update(status="conditional", value=value, frame_ids=[camera["frame"]["frame_id"]],
                      assumptions=["VGGT-Omega monocular depth of one image, back-projected with the calibrated camera and "
                                   "scaled so the fitted table plane lies at table_z; no per-object correction."])
    result["diagnostics"] = dict(service_version=answer.get("version"), latency_s=time.monotonic() - started,
                                 service_timing=answer.get("timing"), npz_bytes=len(data), cache_hit=answer.get("cache_hit"),
                                 n_pixels_kept=answer.get("n_pixels_kept"), support_fit=answer.get("support_fit"),
                                 scale=answer.get("scale"), robot_segments=len(segments),
                                 fused_views=answer.get("fused_views"),
                                 multiview_residual_m=answer.get("multiview_residual_m"))
    result["duration_s"] = time.monotonic() - started
    return result, data
