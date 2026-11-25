# Object Table (`features.object_table`, Enabled by Default)

Calling `geometry_query` with `kind="object_table"` returns an estimated 3D geometric table of objects on the support surface so the model does not have to eyeball object heights and footprints from 2D images alone. The tool reports 3D geometry only; it never names objects on its own or selects manipulation targets.

## Request Format

```json
{
  "kind": "object_table",
  "observation_id": "obs-...",
  "camera_id": "cam_head",
  "pixels": [[u, v], ...],
  "labels": ["..."],
  "target_camera_ids": ["cam_left_wrist", "cam_right_wrist"]
}
```

- **`camera_id`**: Camera view to segment (defaults to `"cam_head"`).
- **`pixels`** (optional, up to 16 `[u, v]` pairs): Specific image coordinates on objects the caller wants guaranteed in the table even if automatic segmentation missed them. Each pixel triggers a SAM 2.1 point-prompt mask, and `value.requested` maps each requested pixel to its resulting object ID.
- **`labels`** (optional, up to 16 noun phrases, truncated to 80 characters each): Free-text noun phrases chosen by the model and forwarded verbatim to the VGGT service (`object-table-v5`), which segments each phrase using SAM 3.1 text prompts. Automatic unlabelled rows come from the fixed class-agnostic prompt `"object"`; labelled instances are inserted first, and any generic `"object"` instance that duplicates or lies inside a labelled instance is suppressed. Returned rows include an extra `label` column (the highest-scoring matching phrase for that object, or `null` for pixel-only objects), and `value.labels` reports `{label, status: "ok" | "no_match" | "filtered" | "backend_unavailable" | "unsupported_backend", ids}` for each phrase. One phrase may match multiple object IDs, and one object ID may match multiple phrases (displaying the highest-scoring phrase in the row). The Harness never invents object labels itself. If the SAM 3.1 sidecar is unreachable, the service automatically falls back to the SAM 2.1 point grid (`status: "backend_unavailable"`).
- **`target_camera_ids`** (optional, up to 3 cameras): Additional synchronized cameras from the same observation (typically wrist cameras). Each object's estimated 3D center is projected into those views using their calibrated intrinsics and extrinsics (this is a geometric 3D-to-2D center projection without occlusion checking), and `value.projected_ids[camera_id]` lists the object IDs whose projected centers fall within that camera's image bounds.

## Model-Facing Result

`value.columns` and `value.rows` contain one row per detected object with IDs `o1..oN` ($N \le$ `object_table_max_objects`, default `20`; `value.omitted` reports the count of additional objects dropped by the cap):

| Column | Description |
|---|---|
| `centre_m` | `[x, y, z]` in `env_origin` meters: horizontal footprint center of the visible mask points, with $z = (\text{top\_z} + \text{bottom\_z}) / 2$ |
| `top_z`, `bottom_z` | `top_z` is the 98th-percentile height of visible mask points; `bottom_z` is the table height, the top of a supporting object underneath, or the lowest visible point when no support is found |
| `length`, `width`, `yaw_deg` | Horizontal extents along and across the 2D principal axis (PCA) of visible points, and the yaw angle of that principal axis in $[-90^\circ, 90^\circ)$ |
| `tilt_deg` | Angle of the 3D principal axis from vertical (populated only for elongated point sets with aspect ratio $\ge 1.8$, otherwise `null`) |
| `support` | `"table"`, the object ID (`oK`) of the supporting object directly underneath, `"held_<arm>"`, or `"unknown"` |
| `quality` | `"ok"`, `"held_fk_depth"`, or `"low:<flags>"` (see quality flags below) |

Alongside `columns` and `rows`, `value.grasp_hints[id]` maps each object ID to pure geometric grasp references—including `pad_z_m`, `fingertip_support_clearance_m`, `opening_check`, `across_yaw_deg` / `stand_upright_angles_deg`, and optional `cavity_depth_m` (in meters, populated when interior recess depth $\ge 0.004\text{ m}$ inside open-top holders/cups/racks)—while `value.track_meta` reports cross-frame tracking status (`track_meta.persistent_ids`, active session state, and track counts).

Each call also returns one overlay image per camera: object contours and IDs drawn on `camera_id`, plus projected object IDs drawn at their 3D centers on each requested `target_camera_id`. Internal service diagnostics (mask pixel counts, 3D point counts, depth-confidence percentiles, and stage latencies) are saved in the Store under `diagnostics` and omitted from model-facing responses.

### Quality Flags

Computed purely from visual and geometric consistency without simulator ground truth:
- `height_above_max`: `top_z` exceeds the table height by more than `object_table_max_height_m` (default `0.30 m`).
- `low_depth_confidence`: Median VGGT depth confidence of mask pixels falls in the bottom 25% of the image.
- `contact_mismatch`: The lowest mask pixels, when back-projected onto the table plane, lie more than `3 cm` farther from the camera than the estimated 3D center (indicating that monocular depth placed the object too high or too close to the camera).
- `no_support_below`: The lowest visible point lies more than `2 cm` above the table with no supporting object detected underneath.
- `few_points` / `fragmented_points`: Sparse or spatially disconnected 3D point clusters.

## Service Pipeline (`POST /object_table`)

1. **VGGT-Omega Reconstruction**: Runs a single VGGT-Omega forward pass on the image, back-projects depth using the calibrated camera intrinsics `K` and rotation `T_world_camera`, and scales the point cloud so the RANSAC-fitted tabletop plane sits at `vggt_support_z_m` (sharing the per-image reconstruction cache with `/height` and `/points`).
2. **Candidate Segmentation**:
   - **v5 default (`seg_backend = "sam31"`)**: Queries the SAM 3.1 text-prompt sidecar (`sam31_service.py` on `127.0.0.1:8333`) with any caller-supplied `labels` plus the generic prompt `"object"` (score threshold `0.4`).
   - **v2–v4 fallback (`seg_backend = "sam21"`)**: Evaluates SAM 2.1 point prompts on an `8 px` grid restricted to pixels whose VGGT height lies between `4 mm` and `45 cm` above the table (3 candidate masks per point; keeping masks with predicted IoU $\ge 0.5$, stability $\ge 0.8$, area between `40 px` and 8% of the image, $\ge 60\%$ of pixels above the table, and not touching image borders or the head camera mount exclusion box).
3. **Filtering and Deduplication**: Selects masks greedily from largest to smallest, skipping masks with $> 30\%$ overlap against already selected masks. Discards masks overlapping the robot arm (via the rendered FK pixel mask in `v5-hp1`, or within `6 cm` of FK link capsules in fallback mode), background masks whose top exceeds `45 cm` above the table, fragmented clusters, and flat table-texture masks with less than `4 mm` height relief over their surrounding border ring.
4. **Per-Object 3D Geometry**: Erodes each mask, keeps pixels with depth confidence above the image 10th percentile, extracts the largest `1 cm` 3D voxel connected component (with `cavity_depth_m` interior floor/rim estimation for recessed containers), and computes the table columns above.
5. **Floating-Object Re-Anchoring (`object-table-v2`)**: Monocular VGGT depth often places thin, reflective, or transparent tabletop objects (such as test tubes) too close to the camera so their 3D points appear to float in mid-air. When an unheld object's lowest visible point lies more than `3 cm` above the table with no supporting object underneath **and** that vertical gap exceeds the object's visible thickness ($\text{top\_z} - \text{lowest\_z}$), the point cluster is rescaled along camera rays until its lowest point rests on the table (annotated with `floating_rescaled_to_table`). Objects whose vertical gap is smaller than their visible thickness (such as spheres whose bottom hemisphere is self-occluded) keep their unscaled geometry and receive the `floating` flag.

For a rescaled row, `spatial_meta[id].before_table_rescale` preserves the original monocular footprint centre as `centre_xy_m` and its visible height extent as `visible_z_range_m` ([2nd, 98th] percentiles of mask-point z, in the same frame and metres as the row). For example, `{"centre_xy_m":[0.22,0.05],"visible_z_range_m":[0.042,0.061]}` describes the depth estimate before applying the table-support assumption. A partially hidden support can make that assumption ambiguous. This field is omitted when the service lacks the original estimate.

Typical latency on an RTX 4090 is **~0.5 s** with SAM 3.1 (**~1–2 s** with the SAM 2.1 grid), and faster when the VGGT reconstruction is already cached.

### Evolution from v4 to v5

- **v4 (Connected-component point matching)**: A caller-requested `[u, v]` pixel reuses an existing segmented object only when the pixel lies inside that mask's largest connected component and the SAM 2.1 point-prompt mask at `[u, v]` has IoU $\ge 0.5$ with it; otherwise, the point mask is added as a separate row (fixing cases in v1–v3 where clicking a test tube inside a rack resolved to the rack's mask). Two requested pixels resolving to the same object receive status `ok_same_object`.
- **v5 (SAM 3.1 text backend)**: Switching default segmentation from the SAM 2.1 point grid to SAM 3.1 text prompting improved offline recall on 28 benchmark frames from `138/191` to `146/191` task objects, reduced false-positive background IDs from `24` to `0`, improved center XY p90 error from `3.7 cm` to `2.0 cm` and top-Z p90 error from `3.0 cm` to `2.5 cm`, and reduced median latency from `2.07 s` to `0.46 s`. Passing `seg_backend: "sam21"` reproduces v4 behavior exactly.

### Cross-Frame Persistent Tracking (`features.persistent_object_tracking=true`) and `reanchor_tracks`

When `features.persistent_object_tracking` is explicitly enabled (`false` by default), the harness passes a per-episode `session_id` to `:8331/object_table` backed by a bounded sliding-window SAM 3.1 streaming video tracker ([`services/vggt_height/sam31/video_session_manager.py`](../services/vggt_height/sam31/video_session_manager.py), using `288×288` logit `mask_input` prompts and closing the session automatically on episode termination):

- **Stable Cross-Frame IDs (`track_meta.persistent_ids=true`)**: Objects retain their `o1..oN` IDs across arm movements, temporary occlusions, and large-displacement tabletop transfers, enabling [`action_state_ledger.py`](../ni_rua/runtime/actions/action_state_ledger.py) to compute exact workspace-wide `object_state_delta` (`dx_m`, `dy_m`, `dz_m`, `dyaw_deg`) without a nearest-neighbor distance cutoff.
- **Zero-Image Semantic Re-Anchoring (`geometry_query(kind="reanchor_tracks")`)**: If an object ID drifts or splits after heavy occlusion or deformable manipulation, the model can call `geometry_query` with `request={"kind": "reanchor_tracks", "observation_id": "obs-...", "target_id": "o3", "phrases": [...], "pos_pixels": [[u, v]], "neg_pixels": [[u, v]], "reset": false, "include_overlay": false}` ([`reanchor_tracks.py`](../ni_rua/perception/external/reanchor_tracks.py)). By default (`include_overlay=false`), it updates the persistent SAM 3.1 track, refreshes `object_table` + `track_meta` and the `PointCloudSpatialField`, and returns `0` images (`include_overlay=true` attaches an updated ID overlay on `camera_id`). See [`tests/perception/external/test_perception_services.py`](../tests/perception/external/test_perception_services.py) and [`tests/perception/external/test_pointcloud_field_and_reanchor.py`](../tests/perception/external/test_pointcloud_field_and_reanchor.py).

## Configuration

| Configuration Key | Default | Description |
|---|---|---|
| `features.object_table` | `true` | Publishes and serves `geometry_query` `kind="object_table"` |
| `features.persistent_object_tracking` | `false` | Opt-in SAM 3.1 cross-frame persistent track IDs (`session_id`) and publishes `geometry_query` `kind="reanchor_tracks"` |
| `features.object_table_max_objects` | `20` | Maximum number of object rows returned per call |
| `features.object_table_max_height_m` | `0.30` | Height threshold above the table for the `height_above_max` quality flag |
| `features.object_table_head_exclude_px` | `"200,405,440,480"` | Pixel bounding box `[u_min, v_min, u_max, v_max]` excluded in `cam_head` (only) to mask out the X5 head-camera mount |
| `features.vggt_camera_id` | `"cam_head"` | Camera used when a request omits `camera_id`, and by the automatic `robot_observe` attachments (`obs3d`). Set `"fixed"` on SO-101 |
| `features.vggt_endpoint`, `vggt_support_z_m`, `vggt_timeout_s` | Shared with `vggt_height` | VGGT HTTP service URL, nominal tabletop height (`0.765 m`), and request timeout (`60 s`) |

Cameras with non-zero OpenCV distortion (for example the SO-101 USB cameras) are sent to the pinhole service as an undistorted image on the same `K` (JPEG quality 95 when the source is JPEG, lossless PNG otherwise; `ni_rua/perception/external/rectify.py`); request pixels and returned pixels (`label_px`, `contour_px`, `bbox_px`, `requested[].pixel`, `scene3d` `uv`) stay in the original image. Cameras with zero distortion are sent unchanged (`tests/perception/external/test_perception_rectify.py`).

When `features.object_table` is `false`, the published tool catalog is byte-identical to a build without this module (`tests/test_object_table.py`). If the VGGT service is unreachable at runtime, the query degrades gracefully by returning `status: "unknown"` with `reason: "object_table_unavailable:<error>"`. When `analysis.scene_truth_log` is enabled, `scripts/score_geometry_truth.py` evaluates every reported object center (`plane_source = "object_table"`) and top-height error (`ot_top_minus_truth_top_z_m`) offline.

## Cross-view check

Embodiments whose profile lists `cross_view` in `supports` (SO-101 with both the `fixed` and the `wrist` camera; not the ARX-X5, whose outputs are unchanged). A table is a single-view monocular estimate. With `target_camera_ids` the result additionally carries `value.cross_view`, so that a disagreement between two calibrated views is visible instead of silent. No row is replaced or corrected: the row, the other view's own monocular estimate and the two-view point are three estimates with their provenance, and the model chooses.

```json
"cross_view": {"wrist": {"method": "mask_centre_rays", "target_table": "computed",
  "rows": [{"id": "o2", "target_view_centre_m": [0.204, 0.017, -0.001], "target_view_minus_row_m": [-0.025, -0.007, 0.007],
            "stereo_centre_m": [0.217, 0.02, 0.002], "stereo_minus_row_m": [-0.012, -0.004, 0.01],
            "ray_gap_m": 0.0237, "parallax_deg": 67.5}],
  "unmatched": ["o3"], "skipped": {"o4": "held_row"}}}
```

| Field | Unit | Meaning |
|---|---|---|
| `target_table` | | `attached`: the target camera's table was already attached to this observation (no service call). `computed`: one more `/object_table` call for the target camera, with the same `labels` |
| `rows[].id` | | Row id of this table |
| `target_view_centre_m` | m, table frame | The target camera's own monocular centre of the same object |
| `target_view_top_z` | m | The target camera's own monocular top height of the same object (its row's `top_z`; `null` when that row has none) |
| `target_view_minus_row_m` | m | That centre minus the row centre |
| `stereo_centre_m` | m | Midpoint of the closest approach of the two mask-centre rays (`scene3d_helpers.triangulate_rays`; distorted pixels are undistorted first). Independent of monocular depth and of the table plane. `null` when the rays are parallel, meet behind a camera or their angle is below 5 deg |
| `stereo_minus_row_m` | m | `stereo_centre_m` minus the row centre |
| `ray_gap_m` | m | Closest distance between the two rays. The rays pass through the area centroids of two masks that show different faces of the object, so they need not meet even with exact calibrations; the gap that geometry leaves grows with the object's size (see *What the ray gap measures* below). A gap beyond what the size explains can also come from a calibration that no longer holds, a mask covering only part of the object or more than it, a wrong pairing, or the arm moving between the two images. No threshold is applied; the model compares |
| `parallax_deg` | deg | Angle between the two rays |
| `ambiguous` | | `true` when a second target row lies within the pairing distance |
| `unmatched` | | Ids of compared rows without a partner in the target table (outside that view, occluded, not segmented, or further than the pairing distance) |
| `skipped` | | `{id: reason}` of the rows that were not compared: `held_row` (a held-object row, whose centre is the gripper-kinematics estimate rather than a monocular one), `no_centre` (no 3D centre), `no_mask_pixel` (no mask outline with area and no label pixel, hence no ray). Every row id is in exactly one of `rows`, `unmatched` and `skipped`; target-table rows of these kinds are never partners |
| `contact_top_z` | m | Profile tag `contact_top` (SO-101 deployment `perception.contact_top_estimate`, default on): top height of a vertical prism standing on the table fitted to the outlines of both views (see *Two-view top height* below); for a row standing on other rows, the stack down to the table |
| `contact_top_z_range` | m | `[low, high]` of the same fit with both outlines grown and shrunk by 1 px, `null` when a changed outline cannot be fitted (too small); calibration error is not included (`ray_gap_m` shows it) |
| `contact_fit_iou` | 0..1 | Mean IoU of the fitted prism's outlines with the two outlines (1: a prism standing on the table explains both) |
| `contact_unpaired_support` | | Only for a stack row whose lower rows have no partner in the target view: their ids. The target outline then has only the paired rows, which is the stack's outline when that camera cannot see the lower rows and too small when it sees them but they were not segmented |
| `contact_reason` | | Instead of the three fields above: `not_on_table`, `mask_at_image_border:<source or target>` (an outline within 2 px of the image edge), `outline_missing:<view>`, `views_do_not_overlap_on_table`, `no_consistent_footprint`, `fit_failed` (the fit raised; the rest of the row stays) |
| `unavailable` | | Instead of the fields above when the target table could not be computed; the table itself stays valid |

**Pairing.** A row and a target row are the same object when each is the other's nearest row by the 3D distance between their monocular centres and that distance is at most `CROSS_VIEW_MAX_CENTRE_M` = 0.06 m. The pairing uses no ray geometry, so two views that disagree by a few centimetres are still paired and the disagreement is reported. The mask centre is the area centroid of the outline `contour_px` (`mask_centre_px`). Held rows and rows without a centre or a mask pixel are not paired and are listed in `skipped`. The bounds are stated in the result's last `assumptions` entry.

**What the ray gap measures.** Each ray goes from a camera through the area centroid of that camera's mask. The two masks show different faces of the object, so the two centroids are projections of different 3D points and the rays need not meet even when both calibrations are exact. Synthetic check with exact calibrations (`CrossView.test_exact_calibrations_leave_a_size_dependent_ray_gap`: boxes from a 30 mm cube to a 120 x 30 mm bar, 30 x 30 x 100 mm and 80 x 80 x 10 mm, random yaw, the overview camera plus a wrist-like camera 0.10 to 0.25 m away at 20 to 70 deg elevation; 895 pairs in the test, 2 385 in a larger run of the same set-up):

| Ratio to the object's longest side | p90 | max |
|---|---|---|
| `ray_gap_m` | 0.08 (`CROSS_VIEW_EXACT_GAP_P90`; 0.076 / 0.077) | about 0.25 (`CROSS_VIEW_EXACT_GAP_MAX`; 0.20 / 0.26) |
| `stereo_centre_m` to the true centre | 0.16 (`CROSS_VIEW_EXACT_MIDPOINT_P90`; 0.161 / 0.159); p90 0.42 at a parallax of 5 to 15 deg | |

In absolute terms: at most about 2 mm for a 30 mm cube, but up to 22 to 25 mm for a 120 x 30 mm bar, so no absolute gap separates a good calibration from a bad one. The assumption text quotes these ratios, and the model weighs the gap against the row's own size. No code decides with them.

**Where it is and why.** On demand, through the existing `target_camera_ids` of `geometry_query` kind `object_table`, and named once in the developer instructions. It is not attached to every observation: that would add one `/object_table` call for the wrist camera to every observation and action reply (2.75 s for one call through the SSH tunnel in the run below) for information that matters only when the gripper is near an object.

**Real observations (2026-11-15).** The three wrist-camera object tables of a real SO-101 run, each compared offline with the fixed-camera table attached to the same observation (SAM masks of both views from the archive):

| Object | Fixed row centre (m) | Wrist-view centre (m) | Distance | Stereo midpoint (m) | `ray_gap_m` |
|---|---|---|---|---|---|
| cube, step 1266 | 0.235, 0.082, -0.006 | 0.206, 0.069, -0.001 | 32 mm | 0.221, 0.072, 0.004 | 27.3 mm |
| cube A, step 1552 | 0.281, 0.002, -0.007 | 0.255, 0.000, 0.001 | 27 mm | 0.268, -0.003, 0.003 | 22.9 mm |
| cube B, step 1552 | 0.229, 0.024, -0.008 | 0.204, 0.017, -0.001 | 27 mm | 0.217, 0.020, 0.002 | 23.7 mm |

The fixed table is 25 to 29 mm further along base +x than the wrist view, and the rays miss each other by 23 to 27 mm on 30 mm cubes, about ten times the at most 2 mm that exact calibrations leave for that size: the two calibrations were not consistent in that run. The arm silhouette gives the same answer independently (the fixed camera had rotated about 5.7 deg; [held_perception](held_perception.md#offline-check-on-real-observations-2026-11-15)), as does the calibration report's VGGT plane normal 5 deg off base z. A first pairing rule that required the two rays to meet rejected all three true pairs and produced one false pair; that is why the pairing uses the monocular centres and reports the ray gap.

**Not a model-facing table.** When the target camera's table has to be computed, that table is used only for the check and is not returned to the model, so held perception neither numbers it nor keeps it as a candidate pre-grasp table (`object_table_query(..., track_held=False)`; [held_perception](held_perception.md)).

**Two-view top height.** A monocular table scales its points to the table plane, so a low-texture object can come out flattened or raised (on the real SO-101, 29 mm cubes read up to 11 mm low or 10 mm high). `contact_height.contact_top` uses only the two calibrated outlines and the table plane: each outline (the masks' outer contours, undistorted) is replaced by its convex hull; on a grid over the table where both hulls' back-projections overlap, a column is alive at height z when its vertical segment from the table up to z projects inside both hulls; for each z the alive columns are the largest footprint both outlines allow, and the z whose prism (that footprint from the table to z) reprojects best onto both hulls (mean IoU) is `contact_top_z`. Grid cell: the coarser ground sample distance of the two views at the object, at most 100 x 100 columns; ceiling `features.object_table_max_height_m`. Synthetic check (exact box outlines, the real fixed calibration, a wrist-like camera 0.15 to 0.25 m away, 1.5 px pose noise, 16 cases): error p50 0.6 mm, p90 2.8 mm, larger below 15 deg parallax; the 1 px range contained the truth in 13 of 16; about 0.1 s per row. The assumption (vertical sides, horizontal top, convex outline, both views see the whole object) is stated in the result's `assumptions`; a tilted, overhanging or partly hidden object breaks it, and `contact_fit_iou` drops.

Tests: `tests/test_so101_held_perception.py` (`CrossView`, `CrossViewQuery`), `tests/test_so101_perception_view.py` (`ContactTop`, `ContactQuery`).

## Arm occlusion (`robot_occlusion`)

Embodiments whose profile lists `robot_occlusion` (SO-101 by default; not the ARX-X5). Every table the model receives also carries `value.robot_occlusion`: `table_covered` (share of the table within the arm's reach in this camera under the arm's FK silhouette, robot base excluded) and `hidden` (rows of the newest earlier table of the same camera attached to an observation reply, and the entries it listed as hidden, whose estimated box now lies under the silhouette, with `id`, `seen_in`, `centre_m`, `top_z` and `covered`). No row changes. Fields, ROI definition, real-run numbers: [SO-101 §13.1](development/so101-integration.md#131-arm-occlusion-report-robot_occlusion). Implementation `ni_rua/perception/external/robot_view.py`; tests `tests/test_so101_perception_view.py` (`OcclusionGeometry`, `OcclusionQuery`).

The action delta names unmatched before rows whose original boxes overlap the arm in `unmatched_under_arm`, for example `[{"before_id":"o2","covered":0.93}]` for 93% projected box coverage. Their location after the action is unknown. Matching uses the source observation, row ID and centre in metres because row IDs may be reused and the occlusion report can carry rows from older observations. Rows matched across observations keep their normal displacement evidence, even if their old boxes overlap the arm.

## Overlay size

Overlays are drawn on the camera frame at its size. Above 640 px width the ids, outlines and markers are drawn `width / 640` times larger (`object_table.overlay_scale`), so they keep their share of the image when the overlay is viewed downscaled; frames up to 640 px wide are drawn exactly as before.
