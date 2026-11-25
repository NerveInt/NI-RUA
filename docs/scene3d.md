# `scene3d` Point Cloud (`features.scene3d`, Enabled by Default)

Calling `geometry_query` with `kind="scene3d"` saves the VGGT-estimated 3D point cloud of a camera view as a compressed NPZ artifact and returns a compact summary handle. The model can inspect the point cloud either inline via the `queries` field or inside `program_run` using the pre-installed `scene3d_helpers` module (automatically copied into every program workspace while `scene3d` or `scene3d_observe` is enabled). Raw point-cloud arrays never enter the model context unless explicitly printed by a program.

`scene3d` complements `object_table`: whereas `object_table` provides per-object bounding geometry from instance segmentation, `scene3d` answers segmentation-free regional and surface queries—such as local tabletop height, container floor and rim elevations, highest obstacle surface in a region, 3D principal axis of a held object in the gripper, or stereo ray triangulation.

## Request Format

```json
{
  "kind": "scene3d",
  "observation_id": "obs-...",
  "camera_id": "cam_head",
  "voxel_m": 0.01,
  "include_wrist_views": false,
  "artifact": "artifact:ref_7",
  "queries": [
    {"op": "top_z", "region": {"center": [0.25, -0.10], "radius": 0.05}},
    {"op": "cavity_profile", "region_or_obj_id": "o2", "radius": 0.06}
  ]
}
```

- **`include_wrist_views`** (optional boolean, default `false`): When `true`, fuses calibrated `cam_left_wrist` and `cam_right_wrist` VGGT reconstructions (via measured joint FK `T_world_camera` and residual-gated alignment in [`services/vggt_height/multiview_cluster.py`](../services/vggt_height/multiview_cluster.py)) into the `env_origin` voxel cloud to fill self-occluded cavities and side walls, caching fused clouds in `MULTIVIEW_CACHE`.
- **`artifact`** (optional): Handle of an existing `scene3d_points` artifact (`artifact:ref_N`), allowing the caller to run helper queries against an already stored point cloud without a new VGGT service call.
- **`queries`** (optional, 1 to 32 items): In-process helper operations executed directly on the point cloud, each formatted as `{"op": "<helper_name>", ...arguments}`. Supported operations include `info`, `table_z`, `points_in_box`, `points_in_cylinder`, `top_z`, `nearest`, `cluster_at_pixel`, `in_hand_axis`, `surface_extrema`, `cavity_profile`, and `triangulate_rays`. Results are returned in `value.query_results` (with any failed query reporting `status: "error"`). When querying the default `cam_head` without a custom `voxel_m`, any point cloud already attached to that observation by `features.scene3d_observe` is reused automatically.

## Model-Facing Result

`value` contains:
- `artifact`: Short artifact reference (e.g., `"artifact:ref_7"`).
- `points`, `points_above_table`, `near_robot`: Point counts.
- `voxel_m`, `table_z`: Voxel resolution and fitted tabletop elevation in meters.
- `bbox` and `bbox_above_table`: `[min_xyz, max_xyz]` bounding boxes in `env_origin` meters.
- `usage` and `api`: Concise usage example and function signatures for `scene3d_helpers`. `api` (and `s3.API` in the program workspace) lists only what the run's embodiment can call: the 6R IK entries (`scene3d_helpers.IK_API`) are present only when `ik_prefilter.py` is staged (`scene3d.ik_prefilter_staged`: the embodiment takes `grasp_angles` and has packaged kinematics; not SO-101).

Internal service diagnostics (latency, NPZ byte size, and RANSAC support-plane fit metrics) remain in the stored record and are omitted from model-facing output.

## NPZ Artifact Schema (`kind="scene3d_points"`, `application/x-npz`)

| Array / Scalar | Dtype & Shape | Description |
|---|---|---|
| `xyz` | `float32 (N, 3)` | 3D coordinates in `env_origin` meters; one representative point per voxel (the pixel with highest depth confidence in that voxel) |
| `conf` | `uint8 (N,)` | Per-image depth-confidence percentile rank (`0..255` mapped to `0.0..1.0`) |
| `uv` | `uint16 (N, 2)` | Source image pixel coordinates `[u, v]` (column, row) |
| `count` | `uint16 (N,)` | Number of raw image pixels falling into that voxel |
| `robot` | `bool (N,)` | `True` if the 3D point lies within `6 cm` of a robot link segment (computed from public FK of the measured joint angles) |
| `obj_id`, `view_mask` | `int16 (N,)`, `uint8 (N,)` | Optional instance index (`0` for background) and bitmask of contributing camera views (`cam_head`, `cam_left_wrist`, `cam_right_wrist`) when multi-view fusion is active |
| `support_z`, `voxel_m`, `image_size` | Scalars | Nominal tabletop height the cloud was scaled to, voxel size in meters, and `[width, height]` |

Points are retained within `[table_z - scene3d_z_below_m, table_z + scene3d_z_above_m]` vertically and within `3.0 m` of the camera optical center.

## Helper Module (`import scene3d_helpers as s3` in `program_run`)

- **`load(artifact)`**, **`info()`**, **`table_z(xy=None, radius=0.08)`**: Returns the global tabletop plane height, or the local median $z$ of points within `1.5 cm` of the tabletop plane inside `radius` around `xy`.
- **`points_in_box(lo, hi)`**, **`points_in_cylinder(center_xy, radius, z_min, z_max)`**: Returns `n`, `centroid`, `min`, `max`, `top_z` (98th percentile), and `conf_p50`.
- **`top_z(region, q=98)`**: Accepts `region` as `(lo, hi)`, `{"center": [x, y], "radius": r}`, or `{"polygon": [[x, y], ...]}` (plus optional `z_min`, `z_max`), returning the $q$-th percentile height of points lying more than `4 mm` above the table.
- **`nearest(p, k=5)`** (2D XY or 3D XYZ) and **`cluster_at_pixel(u, v, radius=0.015)`**: Finds above-table points connected within `radius` to the cloud point whose source pixel is closest to `(u, v)` (bounded to a `0.4 m` growth radius).
- **`in_hand_axis(grasp_center_m, radius=0.065, cloud=None, min_conf=0.1)`**: Computes the 3D PCA principal axis of held-object points near the gripper (`yaw_deg`, `tilt_deg`, `posture`, `elongation`, `bottom_point_m`, `tip_offset_from_grasp_m`; automatically includes `robot=True` points near the fingers).
- **`surface_extrema(region, radius=0.04, min_conf=0.05)`**: Returns unfiltered local surface heights (`floor_z`, `rim_z`, and lowest/highest 3D points) without the `4 mm` tabletop cutoff, useful for recessed slots or shallow trays.
- **`cavity_profile(region_or_obj_id, radius=0.06, min_conf=0.05)`**: Profiles an open-top container, tube rack, or slot around a region or `object_id` (such as `"o2"`), returning `rim_z_m`, `floor_z_m`, `cavity_depth_m`, `opening_center_xy_m`, `inner_clearance_radius_m`, and `n_points`.
- **`align_feature_to_target(grasp_pos_m, grasp_angles_deg, feature_point_m, feature_axis, target_point_m, target_axis, ...)`**: Computes the rigid $SE(3)$ grasp pose (`target_grasp_position_m`, `target_pos_m`, `target_angles_deg`, `flange_position_m`, `quaternion_xyzw`, `rotation_angle_deg`, `feature_pos_err_m`, `feature_axis_err_deg`) that aligns a held-object feature point and axis to a 3D target ([`field_ops.py`](../ni_rua/perception/external/field_ops.py), [`tests/runtime/actions/test_spatial_field_and_ik.py`](../tests/runtime/actions/test_spatial_field_and_ik.py)).
- **`se3_screw(start_pose, axis, pivot_m=None, angle_rad=0.0, distance_m=0.0, pitch_m_per_rad=0.0, n_steps=4, ...)`**: Stateless pure-function rigid hinge/circular-arc/helical screw pose sampler returning sampled waypoint dicts (`[{s, grasp_position_m, angles_deg}]`) around an arbitrary 3D axis ([`tests/runtime/actions/test_spatial_field_and_ik.py`](../tests/runtime/actions/test_spatial_field_and_ik.py)).
- **`check_reachability(arm, position_m, angles_deg, contact_region='pad_center', ...)`**, **`check_source_to_dest(source_pos_m, source_angles_deg, dest_pos_m, dest_angles_deg=None, ...)`**, **`prefilter_stages(stages, initial_robot_state=None, ...)`**, **`jaw_symmetric_angles_deg(angles_deg)`**, **`solve_reachable_pose(arm, position_m, angles_deg, contact_region='pad_center', ...)`**, **`plan_joint_safe_return(target_robot_state, current_robot_state=None, *, arms=('left', 'right'), ...)`**: Fast `< 2 ms` 6R kinematic prefilter, single-arm source-to-destination reachability screening, exact $SO(3)$ parallel-jaw $180^\circ$ symmetric twin (`R_local(x, 180 deg)`), jaw-axis-preserving approach-tilt reachability search, and branch-safe bimanual home return stage builder ([`ik_prefilter.py`](../ni_rua/perception/external/ik_prefilter.py), [`tests/runtime/actions/test_spatial_field_and_ik.py`](../tests/runtime/actions/test_spatial_field_and_ik.py)).
- **Pose & quaternion helpers** (re-exported from `pose_math` in both package and standalone sandbox workspaces): explicit `xyzw` convention matching `robot_state` / `robot_action` (`angles_deg_to_quat_xyzw`, `quat_xyzw_to_angles_deg`, `quat_xyzw_to_matrix`, `matrix_to_quat_xyzw`, `quat_wxyz_to_xyzw`, `quat_xyzw_to_wxyz`) alongside existing `wxyz` helpers (`angles_deg_to_quat_wxyz`, `quat_wxyz_to_angles_deg`, `quat_wxyz_to_matrix`, `matrix_to_quat_wxyz`, `gripper_frame`, `wrap180`).
- **`triangulate_rays(u1, v1, K1, T1, u2, v2, K2, T2)`**: Computes the 3D midpoint `point_m` and ray-to-ray `skew_gap_m` from pixel coordinates in two calibrated camera views (accepting flexible parameter orderings).
- **Common keyword arguments**: `min_conf=0.1` (`surface_extrema` and `cavity_profile` default to `min_conf=0.05`), `robot=False` (set `robot=True` to include points within `6 cm` of the robot arms, such as a held object), and `xyz=False` (set `xyz=True` to include the matched `(M, 3)` point array in the result).

> [!NOTE]
> Like `object_table`, raw VGGT depth can place thin, white, or transparent objects too close to the camera so their points appear elevated above the table (e.g., `14–21 cm` for thin test tubes). Because `scene3d` does not apply per-object connected-component ray rescaling, `scene3d_helpers` exposes `conf` and `table_z` directly so callers can inspect point confidence and support height.

## Service Pipeline (`POST /points`)

Shares the exact same VGGT-Omega depth reconstruction and in-memory per-image cache as `/height` and `/object_table` (back-projecting depth with calibrated `K` and `T_world_camera` and scaling the RANSAC tabletop plane to `vggt_support_z_m`), then performs 3D voxel downsampling, computes confidence ranks and the `6 cm` robot-link proximity mask, and returns the compressed NPZ payload encoded in base64 JSON.

## Configuration

| Configuration Key | Default | Description |
|---|---|---|
| `features.scene3d` | `true` | Publishes `geometry_query` `kind="scene3d"` and provisions `scene3d_helpers.py` into program workspaces |
| `features.multiview_scene3d` | `false` | Opt-in calibrated wrist-camera view fusion for `scene3d_observe` (`multiview_cluster.py`), caching fused clouds in `MULTIVIEW_CACHE` |
| `features.scene3d_voxel_m` | `0.01` | Default voxel grid resolution in meters (`1 cm`) |
| `features.scene3d_z_above_m`, `scene3d_z_below_m` | `0.60`, `0.05` | Vertical clipping window (`[table_z - 0.05, table_z + 0.60]` m) retained in the NPZ artifact |
| `features.vggt_endpoint`, `vggt_timeout_s` | Shared | VGGT HTTP service URL and request timeout (`60 s`) |
| `features.vggt_support_z_m` | `0.765` | Support-plane z in observation-frame m, `[-3,3]`; every number is literal. JSON `null` requires that frozen observation's calibrated `joint_calibration.table_z_m`, with no guessed fallback. CLI: `--vggt-support-z-m observation` |

For SO101 following calibration, persist `{"features":{"vggt_support_z_m":null}}` in the actual launch JSON. Numeric values remain authoritative across apply/rollback; the former implicit `0.765` calibration sentinel is no longer supported. The shared resolver binds support height to the same observation as RGB/K/T, preserving historical observations after newer calibrations are applied. Missing/invalid calibrated height in null mode returns `scene3d_unavailable:observation_support_z_unavailable_or_invalid` before inference. See [VGGT height configuration](vggt_height.md#information-boundary) for queue and replay semantics.

When `features.scene3d` is `false` (and `scene3d_observe` is `false`), the tool catalog and program workspaces remain completely unchanged (`tests/test_scene3d.py`). If the VGGT service is unreachable, queries return `status: "unknown"` with `reason: "scene3d_unavailable:<error>"`.
