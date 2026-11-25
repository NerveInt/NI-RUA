# `features.vggt_height`: Monocular Object Height Estimation for `pixel_plane`

Enabled by default in `optional` mode (as part of `features.external_perception`). When `features.vggt_height` is `false`, the published tool catalog, developer instructions, and all `geometry_query` results remain byte-for-byte unchanged (`tests/test_vggt_height.py::test_off_*`).

## What the Model Sees (When Enabled)

- **`geometry_query` description**: Gains one sentence noting that `pixel_plane` accepts `plane_source="vggt"`, which replaces the caller's horizontal plane height with an image-based estimate of the object's center height at that pixel (using the estimated object top as the alternate plane; callers should inspect `value.plane_estimate.quality`).
- **`pixel_plane` request schema**: Adds the optional field `plane_source` (enum `["caller", "vggt"]`, defaulting to `"caller"` for unchanged behavior). `plane_origin_m` and `plane_normal` become optional in the JSON schema when `plane_source` is supplied (while `plane_source="caller"` without an explicit plane is still rejected at runtime by `pixel_plane`).
- **Response payload**: Returns the standard `pixel_plane` fields (`point_m`, `alternate_point_m`, `assumption_delta_m`, `cross_camera_projections`, and optional overlay image) intersected with the horizontal plane at the VGGT-estimated object center height `center_z_m`, plus `value.plane_estimate`:
  ```json
  {
    "source": "vggt_estimate",
    "measured": false,
    "center_z_m": 0.792,
    "top_z_m": 0.819,
    "support_z_m": 0.765,
    "object_height_m": 0.054,
    "method": "sam_point_mask",
    "quality": "ok",
    "quality_reasons": [],
    "n_points": 412,
    "mask_px": 680,
    "depth_scale": 1.04,
    "support_fit": {...},
    "latency_s": 0.34
  }
  ```
  The first entry in `assumptions` explicitly states that the plane height is a monocular visual estimate rather than measured sensor depth.
- **Alternate plane selection**: Uses the caller's `plane_origin_alt_m` if supplied, else the caller's `plane_origin_m`, else the estimated object top plane (`top_z_m`).
- **Graceful fallback on service failure**: If the caller also supplied a fallback `plane_origin_m`, `pixel_plane` computes the intersection with the caller's plane and sets `plane_estimate.status = "unavailable"`; if no caller plane was supplied, the query returns `status: "unknown"` with `reason: "vggt_estimate_unavailable:<error>"`.

## `features.vggt_height_mode`

- **`"optional"` (default — respects caller parameters)**: Only requests that explicitly pass `plane_source="vggt"` use the VGGT height estimate; requests that pass `plane_origin_m` without `plane_source="vggt"` use the caller's exact plane without silent overrides.
- **`"default"` (legacy experimental override)**: Uses the VGGT-estimated center height on every near-horizontal `pixel_plane` query unless the caller explicitly sets `plane_source="caller"`, relegating any caller-supplied `plane_origin_m` to the alternate plane (`assumption_delta_m`). Planes whose normal is non-vertical ($|n_z| < 0.95$, such as vertical walls) are never overridden (`plane_estimate.status = "not_applied"`). Note: Keep `"optional"` as the shipped default so caller-supplied planes (such as explicit tabletop heights `z = 0.765 m`) are never silently overwritten.

## Why This Lives on `pixel_plane` Rather Than `depth_query`

The RGB-D `depth_query` tool contract is strictly bound to the `rgbd_proprio_calibrated_v1` profile (which requires ground-truth metric z-depth sensor artifacts) and is disabled under the default `rgb_proprio_calibrated_v1` profile. Exposing monocular height estimates through `pixel_plane` keeps the RGB-only observation profile boundary clean, clearly labels the output as a visual estimate (`measured: false`), and integrates directly into the ray-plane intersection tool the model already uses.

## Information Boundary

The service receives only permitted inputs from `rgb_proprio_calibrated_v1`: the archived RGB frame of the selected camera, its calibrated intrinsics `K` and extrinsics `T_world_camera`, and the support-plane elevation resolved from `features.vggt_support_z_m`. No simulator state or object ground truth is ever accessed.

`features.vggt_support_z_m` has three forms. The default `"profile"` (CLI `--vggt-support-z-m profile`) takes the support height of the run's embodiment (`EmbodimentProfile.support_surface_z_m`): the ARX-X5 table top `0.765 m`; an embodiment without a fixed support height (SO-101) uses the observation's calibration, as with `null` below. A number is a finite height in metres in the observation's reference frame, range `[-3, 3]`; negative heights are valid. Every numeric value is authoritative for every robot, including `0.765` (configurations and archives written before `"profile"` existed carry that number and keep it). JSON `null` explicitly selects the same frozen observation's `joint_calibration.table_z_m`. That mode requires a finite, in-range calibrated height; missing/invalid evidence makes the perception result unavailable/unknown before a service request, without guessing a default or consulting the current active calibration. Explicitly supplied geometry planes retain their existing fallback behavior. The optional held-wrist query uses the same resolver.

For SO101 following its calibration, use `{"features": {"vggt_support_z_m": null}}` in the actual launch configuration, or `--vggt-support-z-m observation`. A fixed height uses e.g. `--vggt-support-z-m -0.02`. Queue cases accept the same numeric, `"profile"` or null JSON field and encode null as the CLI token `observation`. Config snapshots preserve null across reload/replay: old observations retain their own calibrated height, while newly acquired observations after apply/rollback carry the newly loaded calibration. Apply/rollback does not rewrite this preference or override explicit numbers. Existing configurations that relied on `0.765` implicitly selecting SO101 calibration must explicitly migrate to null; `0.765` is now always literal. VGGT scales its reconstruction to this input, so agreement with that plane is not an independent height validation.

## Service Overview and Configuration

`features.vggt_endpoint` (default `http://127.0.0.1:8331`) runs as an isolated local HTTP service backed by **VGGT-Omega (1B-512)** and **SAM 2.1 (`hiera-large`)**; see [`services/vggt_height/README.md`](../services/vggt_height/README.md) for setup and weights download instructions.

For each image:
1. Runs one VGGT-Omega forward pass and back-projects the predicted depth map using the calibrated `K` and camera rotation.
2. Fits a near-horizontal tabletop plane via RANSAC and computes a per-image global scale factor so the fitted plane aligns with `vggt_support_z_m`.
3. Runs a SAM 2.1 point prompt at the requested pixel `[u, v]` to segment the object mask, erodes the mask, filters VGGT points above the 10th-percentile depth confidence and at least `4 mm` above the support plane, extracts the largest `1 cm` DBSCAN cluster, sets `top_z_m` to the 98th-percentile height, and sets `center_z_m = (top_z_m + support_z_m) / 2` (falling back to a `21 × 21 px` local window if no valid mask is produced).
4. Caches the scaled 3D reconstruction in memory by image digest so subsequent queries on the same camera frame only run the lightweight SAM point decoder.

Default configuration:

```json
{
  "features": {
    "vggt_height": true,
    "vggt_endpoint": "http://127.0.0.1:8331",
    "vggt_support_z_m": "profile",
    "vggt_timeout_s": 60
  }
}
```

> [!NOTE]
> VGGT-Omega model weights are distributed under the **FAIR Noncommercial Research License v1** (non-commercial research use only).
