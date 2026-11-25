# SO-101 Physical Robot Integration & Calibration Specification

> **Implementation Reference**: This document serves as the technical implementation reference for the SO-101 real-robot hardware backend, geometric kinematics, and calibration system. For end-user deployment and step-by-step setup, see the user-facing [SO-101 Quickstart Guide](../guides/so101-quickstart.md) and [SO-101 Calibration Guide](../guides/so101-calibration.md).

## 0. macOS host setup

- `mamba env create -f environment.yml` creates `NI-RUA`. If pip is configured with a mirror that lacks `openai-codex` / `openai-codex-cli-bin` (for example, the Tsinghua mirror), install the pinned pair from official PyPI: `mamba run -n ni-rua python -m pip install --index-url https://pypi.org/simple openai-codex-cli-bin==0.160.0 openai-codex==0.160.0`. PyPI publishes a `macosx_11_0_arm64` CLI wheel.
- `mamba env create -f environment.so101.yml` (or `mamba run -n so101 pip install -r requirements-so101.txt`) creates the isolated `so101` hardware worker interpreter (`lerobot==0.6.1`, `feetech-servo-sdk`, `pyserial`, `numpy`, `scipy`, `opencv-python-headless` with `cv2.aruco`). Step S0 (`calib_step_start(S0)`) verifies that both `cv2.aruco` and `scipy.optimize` are importable in the `so101` interpreter before any motion step starts.
- Codex process cleanup uses `pidfd.open_process` / `process_argv`: a pidfd on Linux, and on macOS a kqueue `NOTE_EXIT` watch plus `KERN_PROCARGS2` argv.
- The bubblewrap sandbox (`features.sandbox_enabled=true`) is Linux-only. Keep it disabled on macOS (the default). Tests that force it on fail there with `codex_sandbox_bubblewrap_not_found_on_path_or_os_defpath`.
- The benchmark queue and completion hook (`apps/evaluation`) still require Linux pidfd/`/proc`.
- **Camera permission (TCC):** macOS charges camera access to the *responsible app* of the process tree. A Claude Code session started from `ClaudeCode.app` (`com.anthropic.claude-code`, no `NSCameraUsageDescription`) is denied silently (`OpenCV: not authorized to capture video`), even when a terminal has camera access. Start the gateway / robot worker from a terminal app that has camera permission (e.g. Terminal.app, or `claude` launched from such a terminal).
- **Servo power:** the USB serial port enumerates on USB power alone. `Missing motor IDs: 1-6` or an empty `broadcast_ping()` means the 12 V servo supply is off or disconnected; it is not a software fault.

## 1. Supported combination

| Item | Value |
| --- | --- |
| `backend.environment` | `so101_real` |
| `backend.robot_model_id` | `so101` |
| Arm | One TheRobotStudio SO-101 follower: 5 revolute arm joints + 1 revolute gripper jaw, Feetech STS3215 servos on one serial bus |
| Driver | LeRobot 0.6.1 `SOFollower` (`lerobot.robots.so_follower`), run in its own interpreter (mamba env `so101`), like RoboDojo's isolated interpreter |
| Cameras | `wrist` (eye-in-hand, mounted on `gripper_link`), `fixed` (eye-to-hand, static w.r.t. robot base), OpenCV/UVC |
| Public arm slot | `right` only (`EmbodimentProfile.arms`). `left` is rejected with `unsupported_arm:left` (no fabricated second arm) and is absent from the published `robot_action` schema |
| Clock | Real time. The world keeps evolving while the model reasons; nothing is paused |

## 2. Frames, units, TCP

- `frame_id = "so101_base"`: URDF `base_link` frame. World == robot base for this single-arm deployment (`T_world_base = I`).
- `tcp_config_id = "so101-gripper-frame-v1"`: TCP is URDF link `gripper_frame_link` (fixed joint `gripper_frame_joint` from `gripper_link`). It lies **on the fixed jaw's inner fingertip face** (TCP x = 0 to 10 nm in the CAD), not between the jaws: the whole jaw gap is on TCP −X and the fingertip midpoint moves with the jaw angle ([§2.1](#21-gripper-geometry-for-grasping)).
- Public poses: `position_m` (m) + `quaternion_xyzw` of `gripper_frame_link` in `so101_base`, or the grasp centre between the jaws (`grasp_centre_m` + `object_width_m`, [§4.9](#49-grasp-centre-input-grasp_centre_m)), which the Harness converts to that TCP pose. X5's `angles_deg` top-down convention is **not** implemented and is rejected (`unsupported_capability: angles_deg`).
- The arm has 5 DOF: an arbitrary 6D pose is generally unreachable. The approach axis (TCP +Z) tilts only in the arm's vertical plane through `shoulder_pan`, so a tilted quaternion copied from another position usually is not reachable; the roll about the approach axis comes from `wrist_roll`. A straight-down approach reaches every TCP yaw except the band the `wrist_roll` limits exclude ([§2.1](#21-gripper-geometry-for-grasping)). Planning (`SO101Adapter._ik`) works as follows when the deployment has no `ik_service` (the packaged local solver, `ik_plan.local_solve`; with the cuRobo service the solver, candidate selection and the split orientation bounds are described in [§12](#12-ik-planning-the-curobo-ik-service-and-modereach)):
  - `orientation_tolerance_rad >= 3.14159` (`ORIENTATION_FREE_TOLERANCE_RAD` in `ni_rua/robot/motion/motion_tolerance.py`; this covers the schema maximum `3.14159265`, `math.pi` and the `unconstrained_orientation` preset) solves position only, and the arrival check skips orientation (`orientation_checked=false`).
  - Otherwise the weighted position + orientation IK (`ik`) runs first. If it misses either requested tolerance, `ik_position_priority` (exact position, smallest orientation error reachable without moving the TCP) is tried and used when it meets both. The stage `ik` record then shows `solve: "position_priority"`.
  - On `planning_failed`, `ik.position_priority` reports the `quaternion_xyzw` reached at the requested position, with its `position_error_m` / `orientation_error_rad`. Reusing that quaternion plans with the same tolerance.
  - The receipt reports the true achieved position/orientation residuals, and stage tolerances decide `completed` vs `tracking_error`. No silent widening. Before 2026-11-11, values below `math.pi` (including the schema maximum and the preset) never reached position-only IK, and a relaxed-but-bounded `orientation_tolerance_rad` did not change the weighted solution.
- Joint order (public & native): `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll` (rad), then `gripper` (rad, jaw joint).
- Gripper public command `goal_normalized`: `0` = closed, `1` = open, mapped linearly onto the jaw joint range in the deployment file. Measured jaw angle is reported separately and is not a grasp-success signal.

### 2.1 Gripper geometry for grasping

All numbers below are CAD facts of the packaged URDF meshes (`ni_rua/robot/model/so101_gripper_geometry.py`), verified by probes on 2026-11-15 with the example deployment's jaw range (`jaw_closed_rad` −0.174533, `jaw_open_rad` 1.74533) and `motion.joint_limit_margin_rad` 0.05. The model reads them from text generated at load from the CAD and the deployment (`contract._top_down_text`, `_table_line`, `_gripper_text`, `_release_text`, `so101_gripper_geometry.gripper_geometry_text`), so another jaw range or margin changes the published numbers with the code; nothing below is hard-coded in the prompt. Programs read the same facts typed in `robot_capabilities.tool_geometry.gripper_geometry` (`gripper_geometry_payload`: `tcp_definition`, `wrist_roll_axis_tcp`, `fixed_jaw.inner_face_profile_m`, the 11 `goal_to_opening` rows with `moving_tip_x_m` / `moving_tip_z_m`, `moving_jaw_reach`). Tests: `tests/test_so101_model_facts.py`, `tests/test_so101_gripper_geometry.py`.

| Fact | Value (TCP frame `gripper_frame_link`, m unless stated) | Source |
| --- | --- | --- |
| TCP location | On the fixed jaw's inner fingertip face: fixed inner fingertip point (x 0.0000, y 0.0000, z +0.0053); the jaw gap is entirely on −X | `derive_gripper_geometry()["fixed_jaw"]` |
| Fixed inner face by depth | `inner_face_profile_m` rows `{z_m, x_m}`: z +0.0043 → x 0.0000, −0.0057 → 0.0020, −0.0157/−0.0257 → 0.0037, −0.0357/−0.0457 → 0.0054, −0.0557 → 0.0071 (finger length 66.4 mm to the pocket floor at z −0.0611). Renamed from the unlabeled `inner_face_x_m_by_z_m` `[x, z]` pairs | same |
| Moving inner fingertip at jaw 0 rad | (−0.0158, +0.0001, +0.0063); the old `physical.json` text said 0.028 m | `fingertip_inner_point_at_zero_m` |
| Fingertip midpoint | (fixed tip + moving tip) / 2: x −0.0079 at jaw 0 rad, −0.0202 at 0.3 rad; about −opening/2 while both tips are at the same depth | `fingertip_reference(jaw_rad)` |
| wrist_roll axis | Parallel to TCP Z through x −0.0079, y +0.0002 (7.9 mm from the TCP), so turning the roll swings the TCP on that radius | `wrist_roll_axis_tcp` |
| Opening by goal | goal → jaw rad, opening, moving tip (x, z) mm: 0 → −0.174, 3.1, (−1.9, +2.9); 0.1 → 17.3, (−17.2, +6.5); 0.2 → 33.0, (−32.9, +7.1); 0.3 → 48.4, (−48.4, +4.7); 0.4 → 63.5, (−63.2, −0.7); 0.5 → 78.0, (−76.7, −8.7); 0.7 → 104.7, (−97.9, −31.7); 1 → 1.745, 137.2, (−110.0, −76.7) | `gripper_geometry_payload()["goal_to_opening"]` (`GOAL_TABLE`, 11 rows) |
| Reach beyond the TCP (+Z) | Fixed jaw 6.3 mm at every opening; moving jaw 4.5 mm at goal 0, peak 8.1 mm at jaw 0.138 rad (goal 0.16), further than the fixed jaw for goal 0.05..0.27, wholly behind the TCP plane above jaw 0.586 rad (goal 0.40, opening 62.9 mm) | `jaw_reach_facts`, `moving_jaw_reach_m` |
| Fingertips on a support (straight down) | Fixed fingertip touches at TCP z = support + 0.0063; the moving one at up to support + 0.0081 near goal 0.16 (replaces the hard-coded `table + 0.007`) | `_table_line` |
| Measured vs CAD readings | Goal 0 at the commanded closed limit: CAD 0.0031 m; empty close measured on the development unit 0.0042–0.0051 m (deployment `gripper.empty_close_aperture_m`). Goal 1: CAD 0.1372 m at `jaw_open_rad`; measured 0.1361 m (deployment `gripper.full_open_aperture_m`). `gripper_execution.measured_opening_m` and `grip_guard` / `grip_facts` `measured_aperture_m` are the same reading (CAD opening of the settled measured jaw angle) | `_gripper_text`, worker `_jaw_measurement` |

**Straight-down quaternion convention.** `quaternion_xyzw = [cos(a/2), sin(a/2), 0, 0]` points the approach axis (TCP +Z) at world −z with TCP +X at world yaw `a` (rad about +z from +x; `a = 0` is `[1, 0, 0, 0]`); TCP +Y (wrist-camera side) then points along `(sin a, −cos a, 0)`. Verified against FK at several pan/roll configurations (`GeometryAgainstKinematicsTests.test_straight_down_quaternion_matches_fk`).

**wrist_roll-limited yaw band.** With the approach vertical, `a = φ + 3.0938 rad + wrist_roll` (wrapped), where φ is the azimuth of the wrist_roll axis seen from the shoulder_pan axis at (0.0388, 0.0000) m in `so101_base`. Derivation (`so101_gripper_geometry.top_down_roll_facts`): FK of a straight-down configuration at pan 0 gives the yaw at roll 0 and the sign of the roll; pan rotates the whole arm about its vertical axis, so the relation holds at every reachable position. The planner's wrist_roll range is the URDF limits minus `motion.joint_limit_margin_rad` (the worker's `SO101Adapter.limits`): [−2.6939, 2.7912] rad with the example margin, i.e. 314.3° of the 360° roll circle, so `a − φ` in (−22.8°, +22.9°), TCP +X pointing radially away from the base, has no straight-down solution at any position (±20.0° with the bare URDF limits). The TCP is 7.9 mm off the roll axis, so measured from the TCP's own azimuth the band edges shift by up to atan(0.0079 m / r) (r = horizontal distance from the pan axis; 2.3° at 0.2 m). Cross-checked with the local IK at (0.25, 0.06, 0.03) m (`test_band_agrees_with_ik`) and, earlier, at four positions (edges ±20.0° to ±22.0° from the TCP azimuth). An object-table `jaw_line_yaw_deg` is an undirected line: yaw `a` and `a + 180°` are different grasps on this one-sided jaw (fixed jaw, TCP and wrist camera on opposite sides, wrist_roll 180° apart), and at most one of them falls in the band.

**Centred grasp.** Only the −X jaw moves, so closing pushes an object along +X until its +X face rests on the fixed inner face. An object of width `w` (along TCP X) is grasped where it stands only with that face already on it: `TCP = object_centre + (w/2 − x_face) · TCP_+X`, `x_face` = fixed inner-face x at the contact depth (0 at the tip). The published formula no longer prints a numeric `− +0.0000`.

**Release and placement facts** (`contract._release_text`, no procedure): after opening, the fixed jaw's inner face still touches the object's +X face, so a lift slides along it (friction can drag or tip a light object) and a move toward TCP −X pushes the object, while a move toward TCP +X separates the fixed jaw first; opening from below goal 0.16 first swings the moving fingertip further beyond the TCP (4.5 → 8.1 mm), toward a support under a downward gripper, before it retracts behind the TCP plane above goal 0.40; a held object that does not slip keeps its grasp offset: held straight down, its bottom stays (TCP z − bottom z, both at the grasp) below the TCP.

## 3. Shared module APIs (owned by separate work packages)

### 3.1 Kinematics: `ni_rua/robot/model/so101_kinematics.py` (pure numpy, no lerobot import)

```python
ARM_JOINTS: tuple[str, ...]          # 5 names above
GRIPPER_JOINT = "gripper"
BASE_LINK = "base_link"; TCP_LINK = "gripper_frame_link"
WRIST_CAMERA_PARENT_LINK = "gripper_link"

def load_model(urdf_path=None) -> dict          # urdf_manifest() of the packaged SO-101 URDF
def joint_limits_rad(model) -> dict[str, tuple[float, float]]
def fk(model, q_arm_rad, link=TCP_LINK) -> np.ndarray   # 4x4 T_base_link
def fk_all(model, q_arm_rad) -> dict[str, np.ndarray]
def ik(model, target_T_base_tcp, q_seed_rad, *, position_weight=1.0,
       orientation_weight=0.3, max_iterations=200, limits=None,
       position_tolerance_m=1e-3, orientation_tolerance_rad=1e-2, restarts=8) -> dict
    # -> {"q_rad": [5], "position_error_m": float, "orientation_error_rad": float,
    #     "converged": bool, "iterations": int}
def ik_position_priority(model, target_T_base_tcp, q_seed_rad, *, max_iterations=200, limits=None,
                         position_tolerance_m=1e-3, orientation_tolerance_rad=1e-2, restarts=8) -> dict
    # Same keys. Reaches the position first, then minimises rotation error in the
    # position Jacobian's null space; converged = both tolerances met.
```

### 3.2 Calibration file: `<deployment_dir>/camera_calibration.json` and `CalibrationBundle`

Produced by the `/calibration` guided wizard (§6.5), `scripts/so101_calibrate.py` (see section 5), or the legacy one-click recalibration (§6.4), consumed by the worker. The deployment field `camera_calibration_file` is the **single source** of the active camera calibration: the deployment numbers never contain calibration values, and the worker reads this path at start, on operator `reload_calibration`, and after a passing calibration apply. A reload validates the file (`so101_calibration.read_calibration_file`, one entry per configured camera, `image_size` equal to the configured `[width, height]` px). On any failure it keeps the previous calibration and reports why. Offline solve/verify code: `ni_rua/robot/calibration/workflow/hand_eye_solve.py` (the script is a CLI over it).

The guided calibration workflow manages full calibration history using a **versioned `CalibrationBundle`** stored at `deployment.calibration_bundle_dir` (default `<deployment_dir>/calibration/` next to `camera_calibration_file`):
```
versions/<UTC>/manifest.json            # Version metadata (mode, steps completed, joint offsets, previous version)
versions/<UTC>/camera_calibration.json  # Camera calibration file installed by this version
versions/<UTC>/joint_calibration.json   # Solved table_z_m and joint zero offsets
versions/<UTC>/touches.json             # Touch contact points from S1 touch-off
versions/<UTC>/reports/<step>.json      # Per-step verdict and criteria records
runs/<UTC>-<id>/                        # Working directories for active/historical runs
active                                  # Plaintext name of currently active version (updated atomically)
```
- **Atomic Apply & Rollback**: Applying a calibration installs `camera_calibration.json` via atomic file replacement, writes estimated joint zero offsets into `configs/so101.local.json` under `joint_mapping.<joint>.offset_rad` while keeping a timestamped backup (`so101.local.json.bak-<UTC>`), points `active` to the new version, triggers `reload_deployment()` on the robot backend to refresh joint mapping offsets live in the parent process, and preserves the configured VGGT support-height mode. With `features.vggt_support_z_m=null`, each new observation supplies its frozen calibrated `table_z_m`; explicit numeric heights remain unchanged. Rollback reverses both camera parameters and joint offsets to any chosen historical version, also hot-reloading deployment without moving the arm.

```json
{
  "schema": "ni_rua.so101.camera_calibration.v1",
  "reference_frame_id": "so101_base",
  "created_at": "UTC ISO-8601",
  "board": {"type": "charuco", "squares_x": 7, "squares_y": 5, "square_m": 0.030, "marker_m": 0.022, "dictionary": "DICT_5X5_100"},
  "cameras": {
    "wrist": {"K": [[...]], "distortion": [k1,k2,p1,p2,k3], "distortion_model": "opencv_plumb_bob",
              "image_size": [w,h], "optical_axes": "opencv: +x right, +y down, +z forward",
              "arm": "right", "link_name": "gripper_link", "T_link_camera": [[4x4]],
              "intrinsics_rms_px": 0.0, "hand_eye_residual": {"translation_m": 0.0, "rotation_rad": 0.0}},
    "fixed": {"K": ..., "distortion": ..., "distortion_model": ..., "image_size": ..., "optical_axes": ...,
              "T_base_camera": [[4x4]], "extrinsic_residual": {"reprojection_px": 0.0}}
  },
  "samples": 0
}
```

At observation time the worker computes `T_world_camera` for `wrist` = `T_base_link(q_measured) @ T_link_camera` and for `fixed` = `T_base_camera`, and publishes them as the existing `camera_calibration` / `camera_pose_sample` artifacts (`robot_calibration_information.py` contract). The published document carries `calibrated_at` and `calibration_residuals` (px, m, rad: the measurement precision of the active calibration). It never carries the file path, hashes or provenance. A reload changes `calibration_id` from the next observation on. With no calibration file, the worker uses the uncalibrated profile `rgb_proprio_v1`; it never fabricates intrinsics.

### 3.3 Deployment file: `configs/so101.local.json` (git-ignored, machine specific)

Serial port, LeRobot calibration id, camera bindings and orientations, calibration file path, motion limits (max joint speed rad/s, max per-command delta rad), the optional `ik_service` section (cuRobo IK service URL, timeout, solutions, table height; [§12.2](#122-deployment-section-ik_service)), the optional `perception` section (occlusion report, two-view top height, attached cross view, clear-view pose; [§13.4](#134-top-height-from-two-views-contact_top_z-and-the-deployment-perception-section)), gripper jaw range and measured jaw apertures (§4.1), and the optional `motion_compensation` section (in-transit sag mechanisms, all off when absent; [§4.12](#412-in-transit-sag-transit-telemetry-and-deployment-motion_compensation)). No credentials.

- **Camera Configuration & Orientation**: Each camera under `cameras.<name>` (`wrist` and `fixed`) specifies:
  - `unique_id` (optional string, default `null`): the stable identity of the camera's physical device ([§3.4](#34-camera-binding-unique_id-device_name-and-index_or_path)). macOS: the AVFoundation `uniqueID`, the same string `system_profiler SPCameraDataType -json` prints as `spcamera_unique-id` (USB camera `0x1340001b170209`: USB location + vendor + product id, so it follows the USB port; built-in camera: a UUID). Linux: a `/dev/v4l/by-id/...` or `/dev/v4l/by-path/...` link, full path. It must match exactly one present device or the camera refuses to start; two cameras may not share one.
  - `device_name` (optional string, default `null`): case-insensitive substring of the device name or model (macOS `localizedName` / `modelID`, Linux sysfs `name`). It must match exactly one present device (together with `unique_id` when both are set) or the camera refuses to start. Two cameras of the same model report the same name (here both USB cameras are `2USB camer`, model `UVC Camera VendorID_6935 ProductID_521`), so bind those by `unique_id`.
  - `index_or_path`: OpenCV device index (e.g. `0`, `1`) or device path (e.g. `"/dev/video0"`). Used only when `unique_id` and `device_name` are both `null` (index binding, kept for existing deployments). Still required; ignored when an identity is set.
  - `rotate_deg`: `0 | 90 | 180 | 270` clockwise rotation applied to incoming camera frames. Allows mounting cameras in portrait or inverted orientations (e.g. 180° for upside-down wrist camera mounts, 90°/270° for portrait orientation) before passing to detector, preview, and observation pipelines.
  - `width`, `height`, `fps`, and optional `fourcc`.

### 3.4 Camera binding: `unique_id`, `device_name` and `index_or_path`

An OpenCV index is a position in the list of present video devices, not an identity. On 2026-11-16 the deployment bound `wrist` to index 0 and `fixed` to index 1; the wrist USB camera came loose, the other devices moved up, index 1 became the MacBook's built-in FaceTime camera, and the worker fed images of the room to the model as the `fixed` camera (and calibration ran on the wrong devices). The previous `device_name` resolver did not prevent this: it fell back to `index_or_path` when the name matched nothing, and on macOS it took the `system_profiler` list position as the OpenCV index, which is not OpenCV's order.

Binding now works as follows (`ni_rua/robot/backends/so101/camera_devices.py`, used by the worker, `scripts/so101_cameras.py` and `scripts/so101_calibrate.py`):

- **Device list, no camera opened.** macOS: `[AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo]` followed by `AVMediaTypeMuxed`, read through the Objective-C runtime, which is the list OpenCV's AVFoundation backend opens from. OpenCV 4.7.0 and later (`modules/videoio/src/cap_avfoundation_mac.mm`, "Preserve devices ordering on the system") sort that list by `uniqueID` (`NSString compare:`) and open `devices[index]`; the rule was checked in the 4.7.0 to 4.13.0 and 5.0.0 sources (4.6.0 and older use the unsorted list), and the `so101` env has OpenCV 4.13.0. So the OpenCV index of a device is its position in the list sorted by unique ID: today `0x1340001b170209` (USB) is index 0 and the FaceTime UUID index 1, although `system_profiler` lists FaceTime first. The worker opens that index with `cv2.CAP_AVFOUNDATION`. With an older or missing OpenCV, or a non-ASCII unique ID (where code-point order and `NSString compare:` could differ), the index rule is unverified and identity binding is refused (`so101_camera_index_rule_unverified`). Linux: one entry per `/sys/class/video4linux/videoN` node with the `/dev/v4l/by-id` and `by-path` links that point at it; an identity binding opens the node `/dev/videoN` with `cv2.CAP_V4L2`. A UVC camera also has a metadata node with the same name, so on Linux bind by the `...-video-index0` link. Other platforms have no device list: identity binding is refused, index binding works as before.
- **Strict resolution, no fallback.** When `unique_id` or `device_name` is set, it must match exactly one present device. Otherwise startup stops with `so101_camera_not_found:<camera>:<identity>` or `so101_camera_ambiguous:<camera>:<identity>` (the message lists the present devices with their unique IDs). An unreadable device list stops it with `so101_camera_device_list_unavailable`, and two cameras resolving to one device with `so101_camera_duplicate_device`. The binding runs before the serial bus is connected, so a missing camera refuses before the arm is touched. After opening, the device list is read again; if it changed in between (a camera plugged or unplugged, which shifts the index), the camera refuses with `so101_camera_devices_changed`.
- **Index binding is reported.** With neither identity set, `index_or_path` is opened as before, and the worker names the device that index resolved to.
- **Operator record only.** `<robot run_dir>/camera_binding.json` (device list, index rule, OpenCV version, per-camera mode, OpenCV target, device name, unique IDs, transport such as `usb` or `bltn` for built-in, and any refusal), `so101_camera_binding ...` lines in `robot-worker.log`, the S0 self-check (`detail.cameras.<name>.device`) and calibration `session.json` (`camera_devices`) carry it. None of it is published to the actor.

**Operator helper.** `scripts/so101_cameras.py` lists the present devices, their OpenCV index and unique IDs, and with `--deployment` shows what the worker would bind each configured camera to, or why it would refuse (exit status 1). It opens no camera. Run it with the worker's interpreter, whose OpenCV version decides the macOS index rule:

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/so101/bin/python scripts/so101_cameras.py --deployment configs/so101.local.json
```

To bind identical cameras, plug them in one at a time (or unplug one) and note which unique ID appears or disappears, or cover one lens and watch the console preview. A macOS USB unique ID follows the USB port, so keep each camera on its own port; after moving a camera to another port, update its `unique_id`. Two identical cameras that swap ports also swap unique IDs, which the binding cannot notice; the S0 camera images and the calibration staleness hint (`calibration.staleness_max_board_shift_px`) are the remaining checks.

## 4. Backend layering

- `ni_rua/robot/backends/so101/backend.py`: parent-process `RobotBackend` subclassing the existing `RobotProcess` (JSONL worker, cancellation, exit confirmation). It exposes `start/observe/execute/reset/close/preview_batch`, `compile_action`, and `validate_action`.
- `ni_rua/robot/backends/so101/worker.py`: runs in the `so101` interpreter. It owns the serial bus and cameras and reuses `worker_protocol.serve_worker`.
- `ni_rua/robot/backends/factory.py`: `"so101_real"` entry. `ni_rua/robot/model/embodiments.py`: `SO101` description.
- Embodiment profile: every robot fact and bound of the contract and worker (arm ids, frame, TCP, speed ranges, tolerance and stall-guard presets, gripper zero point, full opening, preset margins, timeouts) comes from `deployment.embodiment_profile(deployment)`, built from the same deployment file in the parent and in the worker; the model-visible `robot_action` schema, tool text and instructions are rendered from it, so this arm's catalogue names only its own numbers and omits the fields it cannot honour. See [Embodiment Profile](embodiment-profile.md).
- `reset` on real hardware cannot teleport the world. It is limited to moving the arm to a recorded home joint configuration, reported as `reset_strategy="arm_home_only_world_unchanged"`.
- Stop: send the present measured position as the goal (hold). Confirmation means the measured velocity has settled below a threshold. Otherwise `unknown`.
- Step counter: `sim_step` carries the adapter's dispatched-command counter with `step_domain="so101_control_step"`. It is not simulation time.
- Worker lifetime: the worker is a plain `Popen` child in the terminal's process group. It ignores SIGHUP, and also SIGINT when its stdin is a pipe (started by the gateway), so closing the terminal or pressing Ctrl-C cannot kill it before the gateway sends `close` (torque state kept, cleanup recorded); stdin EOF (owner gone) still holds the arm and exits 0. A worker started by hand from a terminal keeps the default Ctrl-C behaviour. Its first `robot-worker.log` line records `pid`, interpreter and the imported `ni_rua` package path (operator-only provenance). Test: `tests/test_so101_worker_signals.py`.

### 4.1 Grip guard and `checkpoint_after_gripper`

SO-101 batches accept the shared grip-guard contract (`ni_rua/robot/motion/grip_guard.py`; the same rules as the ARX-X5 `batch_motion`, not a parallel implementation):

| Field | Where | Meaning, unit, default |
| --- | --- | --- |
| `grip_guard_preset` | gripper stage, move with `gripper_goal_normalized`, or batch | `none`, `non_empty`, `thin_object`, `rigid_object`, `wide_object`. Minimum fingertip gap = `gripper.empty_close_aperture_m` + the preset margin from `gripper.preset_margins_m` (defaults: `non_empty` 1.5 mm, `thin_object` 0.5 mm, `rigid_object` 2.0 mm, `wide_object` 3.0 mm; none has a maximum). Rejected while the deployment has no measured `empty_close_aperture_m`. The X5 legacy names (`normal`, `thin`, `solid`, `empty_only`) are refused. |
| `min_aperture_m` / `max_aperture_m` | same | Absolute fingertip gaps in m, in `[0, full_open_aperture_m]` (this jaw's full opening, 0.1361 m measured on the development unit, 0.1372 m CAD; the ARX-X5 0.088 m cap no longer applies); replace the preset's bound; values outside are refused with the limit in the error, never clamped. Default none. |
| `checkpoint_after_gripper` | same (bool) | `true` pauses after the jaw stage, before its first dependent move (shared `jaw_checkpoints` rule). Default `false`. |

- **Measured gap:** `measured_aperture_m` is the CAD opening (`so101_gripper_geometry.opening_width_m`) of the measured jaw angle read after the stage settles; `grip_facts` has the shared shape (`measured_jaw_before/after`, `jaw_delta`, `measured_aperture_m`, `aperture_delta_m`).
- **Zero point and range** (deployment `gripper` section, both nullable, m): `empty_close_aperture_m` is the gap an empty full close (goal 0) reports on this unit; set the largest value seen over several empty closes. `full_open_aperture_m` is the gap at goal 1 (null: CAD opening at `jaw_open_rad`, 0.1372 m). **Measure and set both before real runs** (close and open the empty jaw a few times and read `gripper_execution.measured_opening_m`, the same reading that `grip_guard` / `grip_facts` call `measured_aperture_m`; the actor text names both). While `empty_close_aperture_m` is null: presets are rejected, explicit bounds still work, the post-grasp re-check is off (`post_grasp_recheck.available=false` in preview and capabilities), and the actor text states that an empty close reads at least the CAD closed-limit opening (0.0031 m at `jaw_closed_rad` -0.174533) and about 0.004-0.005 m on real units. Measured on the arm 2026-11-13 (run `run-b75b67a38f5a4ef3bf468f2fa919a76c`): empty close 0.004731 m three times (jaw -0.14201 rad), full open 0.1361 m (jaw 1.72062 rad), a 29 mm cube 0.02891 m. Earlier archives on this unit read 0.0042-0.0051 m for empty closes, so the zero point can move after recalibration; re-measure it then.
- **Reading accuracy:** verified at one point (a 29 mm cube read 0.0289 m); near closure the reading is offset by the empty-close gap (a true gap of 0 reads that gap); between them it is unverified, so the actor text advises a preset or `checkpoint_after_gripper` below about 10 mm. Empty closes on the development unit spread 0.0042-0.0051 m (0.9 mm), more than `thin_object`'s 0.5 mm margin.
- **Rule:** a jaw stage with a guard is evaluated when the jaw closed during the stage (`jaw_delta < -0.001`) and later stages remain. A gap outside `[min, max]` pauses the batch with a confirmed hold before the next stage: `status="stopped"`, `reason="observation_checkpoint"`, `checkpoint_reason` `aperture_below_min_m` / `aperture_above_max_m`, the stage result `grip_guard` with `min_aperture_m`, `max_aperture_m`, `measured_aperture_m`. Resume the same `plan_id` or re-plan. If the jaw is still moving when `gripper.settle_timeout_s` expires, a guarded closing stage pauses the same way with `checkpoint_reason="jaw_not_settled"` and `grip_guard.jaw_settled=false` (an unsettled reading is neither a pass nor a final miss). Batch-level guard fields apply to jaw stages with `goal_normalized < 0.85` that have none of their own. With a measured zero point, after a passing guarded close (goal < 0.5) the first following move without a jaw command re-checks once (skipped when it ends more than 5 mm lower): it pauses only if the jaw closed by at least 0.08 normalized during that move and ends within 0.002 m of the empty-close gap. On this jaw 0.08 normalized near the empty close is about 0.012 m of gap (0.007 m on ARX-X5), so an object narrower than about 0.017 m can slip out unnoticed (`post_grasp_recheck.min_object_width_m`).
- **Reporting:** preview `stage_plans[<jaw stage>].grip_guard` gives the effective `{grip_guard_preset, min_aperture_m, max_aperture_m, empty_close_aperture_m, source, detects_empty_close, post_grasp_recheck}`. `detects_empty_close` is a reported fact, the bound is kept: `false` when the minimum is absent or at/below the measured empty-close gap or the CAD floor (the guard can never fire on an empty close), `true` above the measured gap, `null` when the gap is unmeasured and the minimum is above the CAD floor. The `robot_action` field help, `robot_capabilities.grip_guard` and the actor motion instructions (`contract.grip_guard_text`) are all rendered from the same profile, so the published preset bounds equal the enforced ones (there is no longer a disclaimer about another robot's gripper); jaw-checkpoint entries in preview `checkpoints` carry `pause_before_stage_id`.
- **Guard fields on a move:** accepted on a move stage that carries `gripper_goal_normalized` (a jaw stage or a during-motion ramp, §4.3); refused on a move without a jaw command. Everything else that is refused is listed in §4.8.

Tests: `tests/test_so101_feedback_grip_guard.py` and `tests/test_so101_motion_features.py` (empty close pauses before the lift, a 29 mm grasp passes and lifts, batch defaults and `aperture_above_max_m`, the one-shot slip re-check, `checkpoint_after_gripper`, an unsettled jaw, rejections, and the unmeasured zero point: a bound at the CAD floor reported as `detects_empty_close=false`, no slip re-check); ARX-X5 behaviour: `tests/test_grip_guard_and_trajectory.py`, `tests/test_batch_motion.py`.

### 4.2 Trajectory feedback evidence

Every executed action receipt carries what `trajectory_feedback.py` reads (Harness stores each as an actor artifact):

- `before_state`: measured `robot_state` before the first command.
- `actual_trace`: one `{sim_step, t_s, state}` per control step (joints, TCP position and quaternion rounded to 1e-6; samples after the first omit joint names, top-level frame ids and the constant gripper notes, which feedback takes from `before_state`); the state is the last joint read taken while that step's command was the latest (one control period after an interpolation command, the settled read after the last command of a stage), plus a final read when the last command had none. `t_s` is host monotonic seconds since the action started; feedback timing (`elapsed_time_s`, derived velocities) uses these read times (`measurement_timing.source_kind="recorded_host_read_time"`), not step counts.
- `planned_trace`: `{arm, joint_names, q_rad, completed_native_steps, row_kinds, ik_targets}`: one commanded row per dispatched command (5 arm goals and the commanded jaw angle, rad, exact), including arrival corrections and holds, plus the cached IK targets. `dispatched_trace` is `{scope, steps}`: the control step of each row (the rows are stored once).
- `trace_sampling.actual_boundary_samples` is true only when every control step of the action has its measured sample, this action dispatched exactly the steps `start+1..end` (an operator hold from another thread makes it false) and the status is not `unknown` (a failed write may have partially executed); then feedback aligns planned and measured rows and reports `measured_command_error` (encoder minus commanded goal; the jaw column shows a blocked jaw as error).

- `measured_command_error` names the joint (`joint`) because the jaw is one of the compared joints: a jaw stopped by a grasped object is usually the largest error (about 0.33 rad on a 29 mm cube).
- Size: about 0.45 KB per control step for `actual_trace`, 0.08 KB for the commanded rows and 1.3 KB for the shared feedback artifact. The bound is `motion.max_timeout_s` x `motion.command_rate_hz` x about 1.9 KB: about 11 MB for a 300 s action at 20 Hz; a 20 s batch is about 0.7 MB. The actor sees at most six samples per path (about 10 KB).
- A batch execute reads its start sample before it marks the plan executing, so a read fault leaves the plan executable.

Post-action geometry and overlays are available (§4.6): the worker captures an observation at the start of every executed action, so geometry and the measured/planned overlay images have their `execution_start_observation_id`. Each measured sample also has the encoder-FK `measured_ee_pose`. When that capture fails, geometry and overlays report `execution_start_observation_unavailable` and `delivery` reads `automatic_sampled_joints` (no image). Operator and calibration moves (`move_joints`) keep their earlier `trace` / `dispatched` format.

### 4.3 Jaw ramp during a move (`gripper_motion`)

A move stage with `gripper_goal_normalized` and `gripper_motion` `{start_fraction, end_fraction}` (fractions in [0, 1] of the move's command count, `start_fraction < end_fraction`) compiles to one `during_motion` jaw event and the worker ramps the jaw goal inside the move (`SO101Adapter._segment` / `_jaw_schedule`, shared `gripper_motion.synchronize`). Without `gripper_motion` the jaw command runs after arrival, as before.

- **Speed limit:** the ramp is subdivided until no command step moves the jaw faster than `gripper.max_speed_rad_s` (1.0 rad/s in the example file) x `gripper_speed_scale` x the operator `speed_scale_cap`. The slowdown is part of the preview `stage_costs`, and `stage_plans` report the schedule.
- **Refusal:** a schedule longer than `timeout_s` x `motion.command_rate_hz` commands is refused at preview with `gripper_motion_schedule_exceeds_timeout`, before any command.
- **Guard:** after arrival the jaw is measured once it settles (`gripper.settle_timeout_s`), and the grip guard and `checkpoint_after_gripper` apply to that move as to a jaw stage (§4.1).
- **Cut short by a stop:** when a `contact_stop` hold (§4.5) ends the move before the ramp reaches its goal, the worker finishes the rest of the ramp at the held pose at the same jaw speed limit with the arm goal kept, and reports `gripper_execution.finished_at_held_pose`.

### 4.4 `max_upward_dz_m`: an arrival rule only

`max_upward_dz_m` (m, [0, 0.15], stage path or batch-wide) changes only the arrival verdict (`worker.arrival_verdict`). With the horizontal error `sqrt(dx_m^2 + dy_m^2) <= position_tolerance_m` and the orientation within tolerance, a measured z from `position_tolerance_m` below the target up to `max_upward_dz_m` above it counts as arrived; the stage result then carries `vertical_compliance_accepted=true` and `xy_error_m`, and the window used is reported as `effective_tolerances.max_upward_dz_m` in `stage_plans` and `stage_results`.

- **Default 0 m** (no window) when omitted, also with `stall_guard_preset="contact_stop"`: there is no hidden default table as on ARX-X5, and no preset implies a window.
- **No compliance:** the servos still drive toward the target; this is neither vertical compliance nor force control. The bounded sag correction (`motion.arrival_correction_rounds`, `arrival_correction_max_offset_rad`: 3 rounds and 0.1 rad in the example file) still runs before the verdict, shifting the joint goals toward a target the arm rests short of, so an arm resting on a support above the target presses harder on it. A `contact_stop` hold (§4.5) skips the correction, and a move stage can switch it off or bound it with `arrival_correction_rounds` / `arrival_correction_max_offset_rad` (§4.10).
- Tests: `tests/test_so101_motion_features.py` (`test_arrival_rule`, `test_default_window_is_zero_and_reported`, `test_resting_above_the_target_arrives_only_inside_an_explicit_window`).

### 4.5 Stall guard presets, `contact_stop` hold and `contact_block_summary`

`stall_guard_preset` is `off` (no TCP-divergence stop), `contact_stop` (stop when the TCP diverges from the commanded joint path by more than 0.02 m) or `strict` (0.01 m); an explicit `max_tracking_divergence_m` is accepted in 0.005 to 0.15 m. The thresholds are the shared run defaults; other X5 alias names are refused at stage and batch level. The joint tracking limit (`motion.tracking_error_limit_rad`) and the stall window (`motion.stall_window_s`, `stall_min_progress_rad`, `stall_min_commanded_rad`) stop every move whatever the preset.

- **Any stop signal** (TCP divergence, joint tracking limit, stall window) on a `contact_stop` stage holds the measured pose: the measured joints are commanded exactly, without the free-space sag offset (so the TCP does not lift off a support), and the hold is confirmed by settling. The stage is then judged by its arrival rule (`position_tolerance_m`, `orientation_tolerance_rad`, `max_upward_dz_m`). Inside it, the stage completes with `contact_stopped_early` and later stages, gripper stages included, run from the held pose. Outside it, and with every other preset, the stage fails with `tracking_error` and later stages do not run. A descent that stalls without `contact_stop` fails even inside a window.
- At a checkpoint the worker sends one hold goal and confirms the arm's measured joint speed stayed below `settle_velocity_rad_s` for `settle_hold_s` under that goal. The goal includes the bounded `hold_offset_rad` computed at the start of the hold. A contact stop uses zero offset for its hold. Confirmation covers the sampled window. The arm can drift during the subsequent pause.
- Resuming a cached plan checks the current measured joints against those measured at the checkpoint. The same check applies between preview and first execution. When the maximum joint change exceeds `plan.max_start_drift_rad`, the worker rejects the plan before dispatch and returns `tcp_drift`. Its `reference` is `preview` or `checkpoint`. `reference_position_m` and `measured_position_m` are the TCP positions from forward kinematics of the two measured joint configurations in `frame_id`; `delta_position_m` is measured minus reference, in metres along the base x, y and z axes. For example, `[0.001, 0.002, -0.010]` reports a 10 mm drop. A camera capture or a measurement of object movement requires a fresh observation.
- **`contact_block_summary`:** every failed move, and every contact hold, reports `{arm, stage_id, measured_position_m, target_position_m, vertical_shortfall_m, horizontal_offset_m, approach_shortfall_m, blocked_above_target, completed_steps, planned_steps}`. Positions are TCP positions in `so101_base` (m); `vertical_shortfall_m` is measured z minus target z (> 0 above); `approach_shortfall_m` is the distance still to go along the TCP approach axis (+z of `gripper_frame_link`, > 0 stopped short); `blocked_above_target` is true when `vertical_shortfall_m` exceeds the stage's own `position_tolerance_m` (no other threshold).
- **No force sensor:** a stop shows obstructed or lagging motion (joint lag, `Present_Load` is not calibrated), not proven contact (`ni_rua/robot/calibration/workflow/touch_off.py`). The summary is a measurement of where the arm ended, not contact evidence. The text published to the model says so.
- Tests: `tests/test_so101_motion_features.py` (`test_contact_stop_inside_the_window_holds_and_the_gripper_stage_runs`, `test_contact_hold_pins_the_measured_pose_without_the_sag_offset`, `test_contact_stop_outside_the_window_fails_with_one_hold`, `test_uncompliant_descent_stall_fails_even_inside_a_window`).

### 4.6 Trajectory feedback geometry, overlays, contact overlay, home check and start frame

- **Start frame:** every executed action (batch, single request, wait) returns `before_observation`, one observation (joints, then one frame per camera) taken before the first command; the Harness publishes it as a historical observation and the receipt names it as `execution_start_observation_id`. A failed capture never blocks the motion (`execution_start_observation_error` instead); a rejected plan captures nothing. Cost: one observation per executed action.
- **Geometry and overlays:** with the start frame, trajectory feedback computes FK geometry from the profile's TCP link and joint names (`gripper_frame_link`, five arm joints plus the jaw) and draws the planned (preview) and measured overlays on the `fixed` camera frame at execution start. Overlay colours are keyed by the profile's arm (`right`), the legend names the arm and the TCP link, and the preview `link_positions_m` no longer carries a forged `link6` key.
- **`robot_contact_overlay`:** draws the measured TCP, the CAD fixed and moving fingertips at the measured jaw angle, their midpoint and the approach axis. `selections` (X5 contact regions) are refused with `contact_overlay_contact_selection_unavailable`. It is a geometric reference from joint encoders, not contact evidence.
- **Home check:** the completion-guidance home check (`features.completion_guidance` `hook` or `both`) records the first measured `ee_pose` in `so101_base` / `so101-gripper-frame-v1` for arm `right` and compares later observations with the shared tolerances (0.15 m per position axis, 20 deg rotation, inclusive). It describes home geometry only, not task success; it was inert on SO-101 before the profile.
- Tests: `tests/test_embodiment_evidence.py`, `tests/test_so101_motion_features.py` (start frame, `test_harness_publishes_the_start_frame_as_a_historical_observation`).

### 4.7 Deployment defaults of the action language

Defaults come from the deployment file, never from ARX-X5 constants (`configs/so101.example.json`): a move stage that omits tolerances uses `motion.default_position_tolerance_m` (0.01 m) and `default_orientation_tolerance_rad` (0.15 rad); a request without `timeout_s` uses `motion.default_timeout_s` (60 s, at most `max_timeout_s` 300 s); a bounded wait without `timeout_s` uses `motion.default_wait_timeout_s` (30 s; a wait of n steps lasts about n / `command_rate_hz` s, 50 steps at 20 Hz = 2.5 s). `motion.tolerance_presets` sets the named presets as `[position m, orientation rad]` (defaults `tight` [0.003, 0.05], `normal` [0.01, 0.15], `relaxed` and `compliant` [0.015, 0.2], `coarse` [0.025, 0.35], `unconstrained_orientation` [0.015, 3.14159]). `speed_scale` and `gripper_speed_scale` accept 0.25 to 1.0 (1.0 is `motion.max_joint_speed_rad_s` times the operator cap); the presets `slower` 0.5, `slow` 0.8, `careful` 0.9 and `normal` 1.0 are published, `fast` and `faster` are not. A move that names no preset reports its `effective_tolerances` and may carry no `tolerance_preset`.

**Catalogue wording on a real-time profile** (`ni_rua/runtime/actions/robot_action_schema.py`, gated on the profile, ARX-X5 byte-identical): the tool description says that `execute` runs the cached IK joint targets re-interpolated from the joints measured at execution (no "exact trajectory" or "exact cached rows"); `execute_if_clean` executes when the preview is planned, every review SO-101 runs is clean and there are no other `downgrade_reasons`; the reviews and their scope are named from one list (`embodiments.preview_reviews`): `grasp_3d_review` (grasp-centre stages only; TCP-pose stages are not reviewed) and, with `ik_service`, `path_check` (the joint path of every move), and `closed_jaw_before_grasp:<arm>` is named when the deployment has a measured empty-close gap (`unified_actions._preview_downgrade_reasons`). The worker's capability document says the same (`SO101Adapter._execute_if_clean_text`: each review's clean condition and scope and the `downgrade_reasons` codes), and its `tool_geometry.conversion` names both move pose inputs (`position_m` + `quaternion_xyzw`, and `grasp_centre_m` + `object_width_m` converted by the Harness; `_pose_input_text`); the `normal` tolerance preset carries no "vertical contact/load compliance" note (that describes the X5 batch executor's implicit vertical windows; SO-101's window defaults to 0, §4.4).

**Recovery.** The motion text names the shared continuation paths, which work on SO-101 (`tests/test_so101_model_facts.py` `ContinuationTests`): after a pause or a failed stage, preview with `source_plan_id` + `start_from_stage_id` (optional `stage_overrides`) to recompile from the current measured joints, or `plan_id` + `skip_failed_stage=true`. The grip-guard text tells the model to choose the guard from the object (bounded preset or `min_aperture_m` for rigid objects of known width; `thin_object`, none or `checkpoint_after_gripper` for thin or deformable ones).

### 4.8 What SO-101 still refuses, and why

Each of these is refused by name (`unsupported_capability: <field>`) and is absent from the published schema; the instructions list the absent features once. The **Kind** column separates what the hardware cannot do from what the software has not implemented yet.

| Refused | Kind | Why |
| --- | --- | --- |
| A second arm (`left`), `dual_arm_progression`, `arm_options`, paired waypoints, `waypoint_speed_*`, handoff | Hardware | One arm |
| `speed_scale` above 1, presets `fast` / `faster` | Deployment policy | Deployment safety cap |
| Contact feedback, contact witnesses | Hardware | No force or tactile sensor (a stop is obstruction or lag, not contact proof) |
| `contact_region`, `contact_depth_m` | Replaced (SO-101 input) | X5 names for a finger section measured from its pad centre. SO-101 has the same choice as `grasp_depth_m` (m behind the TCP origin plane, from the fixed fingertip end to the pocket floor) on a grasp-centre move ([§4.9](#49-grasp-centre-input-grasp_centre_m)); the refusal names it |
| `angles_deg`, `grasp_position_m`, `via_grasp_poses` | Replaced (SO-101 input) | X5 field names whose meaning (a symmetric gripper, a fixed 0.1501 m offset, RoboProbe angles) does not hold here. SO-101 has its own grasp-centre input: `grasp_centre_m` + `object_width_m` with `quaternion_xyzw` or `fixed_jaw_yaw_deg` ([§4.9](#49-grasp-centre-input-grasp_centre_m)), with `grasp_3d_review` at preview; the refusal names it. The `pose_skill` planning tool stays X5-only |
| `joint_travel_preset`, `max_joint_travel_rad`, `max_backtracking_rad` | Not applicable | Moves are straight joint interpolations inside the URDF limits; `stage_plans[move].joint_travel_rad` reports the travel as a fact |
| `obstacles`, `held_object`, `obstacle_object_ids` without the cuRobo IK service | Needs `ik_service` | They feed the service's collision world ([§12.10](#1210-collision-world-obstacles-obstacle_object_ids-and-held_object)); without `ik_service` nothing could check them, so they are absent from the schema |
| `held_objects`, `contact_entity_id`, `initial_contact`, `placement_contact`, `obstacle_policy`, `ignore_obstacle_ids`, `clearance_preset`, `clearance_height_m`, `target_exemption_*` | Implementation gap | No proxy geometry review, no per-episode obstacle registry and no clearance arc planning; a request with these fails the schema and nothing is stored |
| Other path fields the worker would ignore | Contract | Refused so that nothing is accepted and silently dropped |
| `motion_review` / `preview_motion`, motion cost preview | Implementation gap | No preview digital twin |
| `robot_policy_step`, depth profile (`rgbd_*`) | Not available | No policy trained for this arm; USB RGB cameras without depth |
| Explicit `blend_radius_m`, orientation path constraints, more than one gripper event per stage | Contract | Rejected on both robots |

### 4.9 Grasp-centre input (`grasp_centre_m`)

The TCP `gripper_frame_link` lies on the **fixed** jaw's inner fingertip face (CAD x = 0), not between the jaws: the whole jaw gap is on its -X side and only the -X jaw moves. A model that plans grasps with `position_m` must add `object_width_m / 2` along TCP +X with the right sign for every jaw-line yaw; in the 2026-11-15 run that derivation failed repeatedly (sign, magnitude, an assumed dependence on the opening). The grasp-centre input moves that conversion into the Harness (profile pose input `jaw_grasp_centre`; fields and result keys in [Unified Actions](unified-actions.md#grasp-centre-move-input-so-101-pose-input-jaw_grasp_centre)).

Semantics, chosen for a single moving jaw:

- **The fixed jaw is the reference.** The fixed fingertip face is placed `object_width_m / 2 + fixed_jaw_clearance_m` from `grasp_centre_m` on the TCP +X side: `TCP = grasp_centre_m + R @ [object_width_m/2 + fixed_jaw_clearance_m, 0, grasp_depth_m]`. The jaw opening is not an input: at any opening the fixed face is where it is, and the moving jaw closes the remaining gap. Centring on the open-jaw midpoint instead would make the closing jaw push the object by (opening - width) / 2.
- **`fixed_jaw_clearance_m` is the only push.** With 0 the fixed face touches the object's face on the way in, so a centre or width error of a few millimetres lands the fixed jaw on the object. With a clearance c the object is slid c toward the fixed jaw when the jaw closes (`object_shift_on_closing_m`). The default is 0 and nothing adds a margin: the model chooses it from how well it knows the object.
- **`grasp_depth_m` is the finger section** (the SO-101 form of X5's `contact_region` / `contact_depth_m`): the centre lies that far behind the TCP origin plane, between -0.0053 m (the fixed fingertip end) and 0.0611 m (the pocket floor; finger length 66.4 mm). No named sections are offered: the fixed finger has no pad, and the three X5 names would be numbers without a physical feature behind them.
- **Orientation**: `quaternion_xyzw`, or `fixed_jaw_yaw_deg` for a straight-down approach (`[cos(a/2), sin(a/2), 0, 0]`: TCP +X at yaw a, TCP +Y = (sin a, -cos a, 0), TCP +Z down). Yaw a and a + 180 are the same jaw line with the jaws swapped, and the wrist usually reaches only one; `mode=reach` answers both in one call.
- **Bounds** (`gripper.jaw_grasp`, from `so101_gripper_geometry.jaw_grasp_facts(jaw_open_rad)`): `object_width_m + fixed_jaw_clearance_m <= max_gap_m` (0.1101 m: the widest gap between the fingertips along TCP X, smaller than the 0.137 m fingertip distance at full open because the open moving fingertip has swung back); above `tcp_plane_gap_m` (0.0618 m) the moving fingertip is behind the TCP plane, so a wider object is met behind a centre commanded at depth 0 and needs a positive `grasp_depth_m`. Out-of-range values are refused, never clamped. The request's own bounds are checked first; then the workspace box is checked on the converted TCP target, and its `out_of_workspace` refusal says that target was converted from `grasp_centre_m` and repeats the request (centre, width, clearance, depth).

**Where the model reads it.** The robot_action field help of `grasp_centre_m`, `object_width_m`, `fixed_jaw_clearance_m`, `grasp_depth_m` and `fixed_jaw_yaw_deg` is the one full explanation (conversion, units, defaults, bounds). The motion text (`contract._grasp_sentence`) points there and keeps what the field help does not carry: that tolerances and the other stage fields apply to the converted target, a worked example with this deployment's jaw goals, the report keys and `robot_state.arms.<arm>.jaw_grasp`; the Coordinates paragraph says `position_m` gets no offset and `grasp_centre_m` is converted; the tool description and the `gripper_geometry` text (`so101_gripper_geometry.gripper_geometry_text`) are short pointers, the latter keeping the CAD fact that the fixed face lies further toward +X deeper in the jaw.

Code: `contract.grasp_centre_request` (form), `grasp_centre_pose` (conversion), `_validate_grasp_centre` (bounds and target = conversion of the request), `grasp_centre_report` / `annotate_grasp_centres` (results; called by `RobotBackend.preview_batch`, `reach_batch` and `execute`, parent-side, so the worker is unchanged), CAD helpers at the end of `so101_gripper_geometry.py` (`jaw_gap_m`, `jaw_rad_for_gap`, `moving_fingertip_depth_m`, `jaw_gap_reaching_depth_m`, `grasp_centre_tcp_offset_m`, `grasp_centre_reference`, `jaw_grasp_facts`). The X5 fields stay refused by name and the message names the SO-101 field (`unsupported_capability: contact_region (on SO-101 use grasp_depth_m with grasp_centre_m)`).

What surrounds the input:

- **`robot_state.arms.right.jaw_grasp`** (`geometry.measured_jaw_grasp`, added in `Harness.observation_content`): the measured counterpart in `so101_base`, from the measured TCP pose and jaw angle: `grasp_centre_m` (midway between the fixed fingertip face and the moving fingertip along the jaw line, on the TCP origin plane), `gap_m`, `fixed_tip_m`, `moving_tip_m`, `fixed_jaw_direction`. Commanding `grasp_centre_m = P` with `object_width_m = gap_m` brings that point to P, which is how a held object is carried to a place. `geometry_query` `gripper_relative.grasp_position_m` is a different point (the 3D midpoint of the two fingertip points, which leaves the TCP plane as the jaw opens) and keeps its name.
- **Object-table `grasp_hints`** (`object_table.grasp_hints_from`): `grasp_centre_z_m`, `fingertip_support_clearance_m`, `elongated`, `opening_check` of the row width against the jaw-gap range (`ok`, `tight` within 10 mm below 0.0618 m, `beyond_parallel_range` up to 0.1101 m, `too_wide` above; formerly judged against the 0.137 m CAD full opening) and `jaw_lines`: for each face axis of the row's box, the two `fixed_jaw_yaw_deg` and the `object_width_m` along that line. With the row's `centre_m` these are the inputs of a straight-down grasp-centre move. They describe the estimated box; reachability is not implied.
- **`grasp_3d_review`** at preview (`grasp_preview_3d.audit_batch_jaw_grasp`): see [Unified Actions](unified-actions.md#grasp-centre-move-input-so-101-pose-input-jaw_grasp_centre). Flags: `warn_jaw_not_open_enough`, `warn_closing_goal_above_object_width`, `warn_guard_exceeds_object_width`, `warn_planned_centre_outside_position_tolerance`, `warn_centre_above_object_top`, `warn_centre_below_object_bottom`, `warn_fingertip_below_support`, `warn_fixed_jaw_inside_object`, `warn_wider_than_max_gap`, `warn_along_long_axis`.

Limits to know:

- The TCP target is converted with the **requested** orientation. On this 5-DOF arm the planner often accepts an orientation residual (position-first solve, split tolerances), which swings the grasp centre about the TCP by up to offset x angle (19 mm x 0.35 rad = 6.6 mm). `planned.centre_error_m` reports it and the review warns when it exceeds the stage's `position_tolerance_m`; nothing re-solves for the centre. Solving IK for the grasp centre as the tool point would remove it and needs a worker / IK-service change.
- The numbers are CAD, not measured on a printed unit; the fixed inner face is taken at the fingertip (x = 0) although it steps back up to 7.1 mm toward the pocket floor.
- Tests: `tests/test_so101_grasp_centre.py`. Not yet run on the real arm.

### 4.10 Arrival-correction bounds per move

After a move's last command the worker waits for the joints to settle, then shifts the joint goal past the planned target by the settled shortfall (gravity sag of the low-gain servos) and settles again, before the arrival verdict (`SO101Adapter._arrive`). Until now the deployment alone bounded this (`motion.arrival_correction_rounds`, `arrival_correction_max_offset_rad`), so a move that set a held object on a support, or rested on one inside a `max_upward_dz_m` window, was pushed harder by every round. Two fields on a move stage, or batch-wide as the default of every move stage that sets none (profile tag `arrival_correction`, on every SO-101 deployment that corrects at all: both deployment values above 0), let the model bound it:

| Field | Meaning | Unit | Default | Range |
| --- | --- | --- | --- | --- |
| `arrival_correction_rounds` | Largest number of goal shifts after settling; 0 = none: the goal stays the planned target and the verdict judges where the arm settles | count | deployment `motion.arrival_correction_rounds` (3 in the example file) | integer 0 to the deployment value (the schema maximum) |
| `arrival_correction_max_offset_rad` | Largest total shift of any joint goal from the planned target | rad per joint | deployment `motion.arrival_correction_max_offset_rad` (0.1 in the example file) | (0, deployment value] (the schema maximum) |

- **The deployment values are the default and the ceiling, and the schema publishes them.** `deployment.embodiment_profile` copies them into `EmbodimentProfile.arrival_correction_limits` (`max_rounds`, `max_offset_rad`; the deployment fields stay the single source, their own ranges bounded by `embodiments.MAX_ARRIVAL_CORRECTION_ROUNDS` 10 and `MAX_ARRIVAL_CORRECTION_OFFSET_RAD` 0.5 rad), and `robot_action_schema._build` uses them as the stage and batch `maximum`s and in the field help ("integer count, 0..3; default when omitted 3"). A larger request fails the schema (`4 is greater than the maximum of 3`); `motion_tolerance.resolve_arrival_correction` and the backend contract (`invalid_action: arrival_correction_rounds must be an integer in [0, 3] (this deployment's motion.arrival_correction_rounds)`) refuse the same values by name. Nothing is clamped silently. `effective_limits` (capabilities and receipts) carries both deployment values. A deployment with 0 rounds or a 0 rad offset corrects nothing: its profile has no `arrival_correction` tag, the fields are absent from the schema and a request carrying them is refused (`unsupported_capability`), and the motion text says that no correction runs.
- **One explanation.** The meaning, units, defaults and bounds are in the field help of the two fields; the motion text (`contract._arrival_correction_sentence`, `_vertical_window_sentence`, the Coordinates paragraph) only points there and keeps the use of 0 rounds. An archive written before `arrival_correction_limits` existed restores with the tag and no limits; its schema then shows the shared maxima without calling them the default.
- **Reported:** `effective_tolerances.arrival_correction_rounds` / `arrival_correction_max_offset_rad` in `stage_plans` and `stage_results` are the bounds the move uses. The stage result's `arrival_correction` reports `max_rounds` and `max_offset_rad` (the same bounds), `rounds` (each shift with its residual), `offset_limit_reached` and `termination_reason` (`"disabled"` with 0 rounds).
- **What 0 rounds does not do:** the servos still drive toward the planned target, so a blocked arm keeps pushing with its position gain; only a `contact_stop` hold (§4.5) re-commands the measured pose. In free space an uncorrected move keeps the servo sag, so its tolerance must allow for it.
- The hold offset learned at arrival (`hold_offset`, re-applied by later pauses) is still limited by the deployment bound. With `motion_compensation.gravity_feedforward` (§4.12) the bound grows by the feedforward's `max_offset_rad`.
- **Reach:** `mode=reach` never moves, so its `stage_results[<id>].tolerances` carry no arrival-correction bounds (nor window or stall guard; `worker.REACH_EXCLUDED_TOLERANCES`).
- Code: `motion_tolerance.resolve_arrival_correction` / `arrival_correction_bounds` (stage over batch, profile bounds), `contract.arrival_correction` / `_validate_path_fields`, `unified_actions.compile_request`. Tests: `tests/test_so101_path_check.py` (`ArrivalCorrectionTests`, `ArrivalCorrectionProfileTests`, `ReachToleranceTests`, `test_arrival_correction_fields_resolve_stage_over_batch_into_the_worker`).

### 4.11 Receipt-level residual of a failed move

A batch (or single request) that fails at a move with `tracking_error` now carries, at receipt level next to `failed_stage_id` and `contact_block_summary`:

- `endpoint_residual_m`: `{dx_m, dy_m, dz_m, position_error_m}` (m, measured minus target TCP in `so101_base`) and `orientation_error_rad` (rad), plus `approach_error_rad` / `roll_error_rad` when the stage uses split bounds. It is the arrival residual when the move reached its arrival check (or a `contact_stop` hold was judged), else the residual of a joint read taken after the stop. The ARX-X5 receipt uses the same field name for the same quantity. It is stored once: the failed stage result keeps its own flattened residual fields only.
- `in_hand_tracking_error: true` when the last jaw command was a close (`goal_normalized` below 0.5, the shared closing bound) and the measured CAD fingertip opening is above the measured empty close by more than the `non_empty` preset margin (`gripper.preset_margins_m.non_empty`, 1.5 mm by default), i.e. something is still between the fingers at the failed pose. The field is absent otherwise, and always absent when `gripper.empty_close_aperture_m` is not measured (unknown).

Condensed history (`lean_projection._summary_robot_action`) keeps receipt-level keys only, so these fields, `contact_block_summary` and the preview's `path_check` verdict survive it; stage results do not. Tests: `tests/test_so101_path_check.py` (`ReceiptResidualTests`).

### 4.12 In-transit sag: `transit` telemetry and deployment `motion_compensation`

**The problem (measured on the real arm, 2026-11-16).** The STS3215 servos run a position loop without integral gain (`servo.p_coefficient` 16, `i_coefficient` 0, `d_coefficient` 32), so a joint loaded by gravity rests short of its goal. A move interpolates in joint space from the *measured* start to the IK target, so at its first command the goal fell back to near the sagged pose and the offset the previous arrival correction had built up was lost: the TCP ran 10 to 23 mm below the commanded path during transit (one via move dropped 13.5 mm within 0.2 s of its first command), and a carried cube was dragged into a stack. Arrival correction (§4.10) only repairs the endpoint. The preview's path facts and `path_check` (§12.9) describe the commanded joint path, never the measured one. ARX-X5 simulation runs without gravity at stiffness 4400, so "commanded path = real path" was never exercised there.

**Always reported (telemetry, no behaviour change).** Every executed move's stage result carries:

| Field | Meaning | Unit |
| --- | --- | --- |
| `transit.max_below_path_m` | Largest amount the measured TCP was lower in Z than the nearest point of the commanded path while the move was streaming; 0 when never below | m |
| `transit.max_below_at_fraction` | Where that happened: (command index) / (planned commands), 0 = start, 1 = target | ratio |
| `transit.max_deviation_m` | Largest 3D distance of the measured TCP from the commanded path | m |
| `transit.max_deviation_at_fraction` | Where that happened | ratio |
| `transit.samples` | Measured joint samples used (one per command sent) | count |
| `duration_s` | Wall-clock time of the move, first command to verdict | s |

- **How it is computed:** `motion_compensation.transit_deviation`: FK of the joints read one command period after each command, compared with the TCP polyline through the measured start and the commanded rows already sent (without any offset). Each sample is compared with the *nearest* point of that polyline, so servo lag along the path is not counted as deviation. It runs after the verdict, so it never delays a command. A stopped move reports the transit it had.
- **Preview:** `path_check.checked_path` is always `"commanded"`; the collision-scope capability text and the motion instructions (`contract._path_sentence`) say that the loaded arm runs below the previewed path by the amount executed moves report.

**Deployment section `motion_compensation`** (optional; absent, `null` or `{}` = every mechanism off, the behaviour before the section existed; `deployment.validate_motion_compensation`, texts in `deployment.MOTION_COMPENSATION_FIELDS`). Each key switches one mechanism on its own, so they can be compared on hardware one at a time:

| Key | Meaning | Unit / range | Default | When to use |
| --- | --- | --- | --- | --- |
| `carry_arrival_offset` | A move streams its interpolated path plus the arrival-correction offset the last settled free-space arrival ended with, and its own arrival correction starts from that offset instead of 0 | bool | `false` | `transit.max_below_path_m` of several mm right at the start of moves (`max_below_at_fraction` near 0) |
| `gravity_feedforward` | Adds to every streamed goal `-gravity_torque(goal) / joint stiffness` | object or `null` | `null` (off) | Sag that changes along a move (reach changes), after identification |
| `gravity_feedforward.joint_stiffness_nm_per_rad` | Identified servo position stiffness per arm joint: restoring torque per rad of position error at the deployed `servo.p_coefficient`. Joints left out get no feedforward | N·m/rad, (0, 1000] | required | from `scripts/so101_gravity_identify.py` |
| `gravity_feedforward.max_offset_rad` | Bound of each joint's feedforward offset | rad, (0, 0.3] | required | at least the proposal's `largest_feedforward_offset_rad` |
| `gravity_feedforward.provenance` | Which identification the values came from (file, date, servo gains) | non-empty string | required | audit only; never shown to the model |
| `no_response_stop` | An arrival-correction round after which no corrected joint moved by more than one encoder tick ends the correction with `termination_reason` `"no_response"` and takes its step back | bool | `false` | rounds 2 to 3 that change nothing (`settled_joint_error_rad` identical between rounds) |
| `stop_correction_within_tolerance` | A `robot_action` move's arrival correction sends a round only while the settled TCP is outside the move's own arrival rule; a settled pose inside it ends the correction (`termination_reason` `"within_tolerance"`) and is the pose judged | bool | `false` | a stage that settled inside its tolerance and was then failed by the correction; a planned residual close to its bound |
| `arrival_breakaway_rad` | Per-joint smallest correction step that moves the joint from rest; a smaller step is enlarged to it (sign kept) | `{joint: rad}`, each (0, 0.1], or `null` | `null` (off) | only with values identified on this unit; expect some overshoot |

- **M1 `carry_arrival_offset`** (`SO101Adapter._carry`, `_segment`, `_arrive`). The offset is the correction offset of the last arrival (goal − feedforward − target, rad per arm joint), clipped to the move's own `arrival_correction_max_offset_rad`. It is **zero** when: the move disables correction (`arrival_correction_rounds` 0 or a 0 rad bound: `carry` = `correction_disabled`); the move asks for contact or compliance (`stall_guard_preset` `contact_stop`, `tolerance_preset` `compliant`, `max_upward_dz_m` > 0: `contact_stage`; rule `motion_compensation.contact_stage`); or no valid offset exists (`unavailable`): the first move, after a contact / compliant / uncorrected move (its own arrival offset may come from the support, not from gravity, and is not kept), after any stop signal or contact hold, after `torque_on`, and after operator or calibration motion. Validity is not a threshold: the offset is usable only while the arm goal is still exactly the one that arrival left; a pause between stages (`_stop` with the sag offset) uses the confirmed hold goal as its new reference, every other goal write ends it. An arrival keeps its offset for the next move only when it **passed its verdict and did not reach its offset bound** (`offset_limit_reached` false): an arm that stopped short against something has an offset that is contact, not sag, and the recovery move must not carry it toward the obstacle. **Limit:** the offset is constant along the move and the arrival corrects what is left at the target. When the gravity torque of a joint changes sign between the pose where the offset was learned and the target (a retracted high pose against an extended low one: the shoulder's torque goes from about −0.2 to +0.26 N·m), the carried offset has the wrong sign over part of the move and transit can be worse than without it. The bench's `gravity_sign` transit measures exactly this pair; gravity feedforward does not have the problem.
- **Guards never see the offsets.** The joint tracking limit, the stall window and the stall guard's TCP divergence compare the measured joints with the commanded row itself; residuals, `transit` and the arrival verdict use the same reference. The offsets exist so the arm follows that row (the same rule the touch-off uses for its hover sag offset, `calibration.touch_contact_lag_rad`).
- **M3 `gravity_feedforward`** (`ni_rua/robot/model/so101_gravity.py`). `gravity_torques(model, inertials, q)` sums, for each arm joint, the torque of the weight of every link below it about the joint axis, from the packaged URDF's `<inertial>` masses and centres of mass (total 0.667 kg; gravity 9.80665 m/s² along −Z of `so101_base`). Positive = gravity pushes the joint toward larger angles. A pure-P servo of stiffness K rests at `goal − measured = −torque / K`, so the feedforward offset is `clip(−torque / K, ±max_offset_rad)`, evaluated at every streamed row and at the target. It is applied to every move the worker streams, operator and calibration joint moves included, and to contact stages. **Consequence on contact:** without feedforward a sagging arm leaves part of its weight to the support it lands on; with it the servos carry that weight, so when a guard stops a press the arm pushes harder than before by about the gravity torque, and the stop trips at the same TCP divergence. The motion text says so when the mechanism is on, and the bench's contact test reports it as `peak_load_pct` / `below_contact_m`. **Driver clamp:** LeRobot clamps every goal to `max_relative_target_deg` from the present position, so a deployment with feedforward must satisfy `motion.arrival_correction_max_offset_rad + gravity_feedforward.max_offset_rad + motion.max_command_delta_rad <= radians(max_relative_target_deg)` (the largest offset a goal can carry plus one command step) or it is rejected at load: a clamped goal would silently lose the compensation the receipts report. With the template's 10° clamp and 0.1 rad arrival bound that leaves 0.039 rad of feedforward; raise `max_relative_target_deg` deliberately for more. **Slew:** offsets are never switched on in one command: `_segment` starts from the offset the present goal already holds over the measured start (0 after a contact hold) and moves toward carry + feedforward by at most what `motion.max_command_delta_rad` leaves after the row's own step, so no sent goal changes by more than that published bound. **Not modelled:** a held object's weight, a tilted base, friction, and CAD mass errors.
- **How M1 and M3 combine.** Goal = commanded row + feedforward(row) + carried offset. Arrival correction then measures only the residual the feedforward leaves, and that residual is what the next move carries. The bounds stay separate: `arrival_correction_max_offset_rad` limits correction and carry, `gravity_feedforward.max_offset_rad` limits the feedforward; the hold offset (`hold_offset`, re-applied by pauses) stores only the residual after the feedforward and keeps the arrival bound; a hold adds the feedforward of the pose it holds (`_stop`), not of the pose where the offset was learned, and adds none after a contact hold (whose goal is the measured pose). `torque_on` and `relax` clear the hold offset and the carried offset.
- **M5 `no_response_stop`.** Each round used to add only the measured error (about 0.007 rad, 5 encoder ticks, on the real arm), which stays inside the gear's static-friction band (0.015 to 0.02 rad observed): rounds 2 and 3 did not move the arm and the accumulated goal offset was then stored for holds. With the switch on, the round is judged per joint: a corrected joint that moved by at most one encoder tick (`driver.ENCODER_RESOLUTION_RAD` = 2π/4096 rad; one tick is reading noise) gets its step taken back (`unanswered_joints`), the goal is re-sent (`arrival_correction_revert` in `planned_trace.row_kinds`) and the arm settles again; joints that answered keep their step. When no joint answered the round has `responded: false` and the correction ends with `no_response`. A discarded step therefore enters neither `hold_offset` nor the carried offset.
- **M5 `arrival_breakaway_rad`** is experimental. The enlarged step is the band itself, never more, so the goal passes the measured error by at most band − error and the joint can overshoot by up to that: band × (distance from the joint to the TCP) at the tool, e.g. 0.02 rad × 0.3 m = 6 mm. Two preconditions follow. Validation **requires `stop_correction_within_tolerance`** with a band, so a round that lands inside the move's tolerance is not answered by another full band in the other direction. And a set-down or approach target must keep a clearance of at least band × reach from the support; the bench's `m5-breakaway` condition says so for `--place-clearance-m`. The identification script proposes the larger of the two directions and adds the required switch to its proposal. The round lists the enlarged joints in `breakaway_joints`.
- **M6 `stop_correction_within_tolerance`** (observed on the real arm, 2026-11-16). A stage with `tolerance_preset` normal (0.015 m, 0.15 rad) was planned by the position-priority IK with an orientation residual of 0.1488 rad. The arm first settled inside both tolerances (0.0104 m, 0.104 rad); the correction then drove the joints to the IK target, the position improved to 0.0011 m, the orientation became 0.1507 rad, and the stage failed with `tracking_error` while holding a cube. Correction works on joint errors toward the IK target, so it improves position but cannot be better in orientation than the IK solution itself. With the switch on, `_arrive` evaluates the arrival rule (`arrival_verdict`: position, orientation or approach / roll, `max_upward_dz_m`) on every settled pose before sending a round; inside it, the correction ends and the verdict judges that same pose, so correction can never turn an in-tolerance arrival into a failure, and an arm resting on a support inside a `max_upward_dz_m` window is no longer pushed harder. **Trade-off:** a guaranteed pass against accuracy beyond the tolerance. The arm stops anywhere inside the tolerance the move asked for (0.0104 m in the case above instead of 0.0011 m) and correction rounds are skipped more often, so moves are faster; a move that needs more accuracy asks for a tighter tolerance. The alternative, correcting first and stepping back when a round leaves the tolerance, keeps the accuracy in most cases but cannot guarantee that the arm returns to the earlier pose through gear friction, so it was not chosen. Operator and calibration joint moves (`move_joints`) keep their full correction. With `carry_arrival_offset`, an arrival that stops early carries the offset it had. **Preview:** no margin field was added; `stage_plans[<move>].ik.position_error_m` / `orientation_error_rad` already sit next to `tolerances.position_tolerance_m` / `orientation_tolerance_rad` (0.1488 against 0.15 in the case above).
- **Receipts.** `effective_limits.motion_compensation` (receipts and `runtime_limits` in capabilities) lists only what the model can use: the mechanisms switched on (`carry_arrival_offset`, `no_response_stop`, `stop_correction_within_tolerance`, `arrival_breakaway`: `true`) and `gravity_feedforward.max_offset_rad`; it is absent when all are off. The identified joint stiffness, the per-joint breakaway bands and the provenance note are tuning data: they stay in the deployment file (and the bench report's `settings`) and never enter capabilities, receipts or the motion text (`deployment.active_motion_compensation`; test `test_identified_servo_values_never_reach_the_model`). Per move: `path_compensation.carried_offset_rad` + `carry` (M1), `path_compensation.gravity_feedforward_rad` at the target (M3), `arrival_correction.start_goal_offset_rad` (M1) and `final_goal_offset_rad` (the offset in force at the end; with any mechanism on). `planned_trace.q_rad` stays the offset-free plan (the interpolated rows, and for correction commands the target shifted by the correction made at this arrival, as before), so trajectory feedback draws the plan where it was planned; `planned_trace.goal_offset_rad` (present only with a mechanism on) is the compensation sent on top of each row, so row + offset is the goal the servo received. With every mechanism off none of these fields appears and the commands and traces are the ones of before.
- **M4 servo integral gain.** `servo.i_coefficient` (0 to 254, register units, written at connect to every servo) is the servo's own answer to steady-state sag. The code default and the template stay 0. **Contact risk:** against an obstacle the position error does not shrink, so the integral keeps raising the motor torque up to the servo's limits for as long as the goal stays beyond the obstacle (integral windup); the stall window and stall guard still stop a move only from measured positions, and a `contact_stop` hold (goal = measured pose) releases it. Measure both effects with the bench below before real runs, starting small (2, 4, 8).

**Identification** (`scripts/so101_gravity_identify.py`, so101 interpreter, gateway stopped, about 4 minutes, supervised). It holds the arm at a grid of free-space joint poses with arrival correction and all compensation off, pairs the settled `goal − measured` with the gravity torque at the measured joints, and fits each joint by least squares (`motion_compensation.fit_stiffness`: `error = compliance · (−torque) + intercept`; joints whose gravity torque varies less than `--min-torque-range-nm`, default 0.05 N·m, are not fitted, which leaves out `shoulder_pan` and `wrist_roll` on a level base). Each pose is visited in forward and reverse order; `approach_spread_rad` is the difference between the two approaches (friction). With `--breakaway` it also steps each joint's goal one encoder tick at a time until the joint moves, in both directions. It requires `servo.i_coefficient` 0 and writes a **proposal file only** (`proposal.motion_compensation`); it never edits a deployment file. **Height screen** (forward kinematics; not a collision check): the lowest of every moving link origin and the two CAD fingertip points, at both ends of the jaw range, must stay at or above `--min-z-m` (default 0.06 m) at every pose, along the joint-space line between consecutive poses (sampled every `--screen-step-rad`, default 0.02 rad), from the start pose to the first pose, from every pose back to `home.q_rad` (the abort path) and on the lines to and from `--breakaway-pose`. The start pose, home and the breakaway pose are given: when one is itself below the floor, its lines only must not dip below it. `--dry-run` prints every pose and line with its lowest point without opening the bus (assuming a start at home; the real run screens again from the measured pose and refuses before moving). **Acceptance:** a joint's stiffness enters the proposal only when `r_squared >= --min-r-squared` (default 0.9: the gravity model explains at least 90 % of the variance of the settled error) and the mean difference between the two approach directions is at most `--max-approach-spread-ratio` (default 0.5) of the error range the fit explains; a rejected joint is reported with `accepted: false` and the reasons. **Safety:** it refuses to run with `disable_torque_on_disconnect` true, returns to home after an abort once the stop is confirmed (not after Ctrl-C), and sends its one-tick breakaway steps through the worker's goal writer with the admission gate checked and the stop generation bound, so a hold ends them.

**Bench** (`scripts/so101_motion_bench.py`, NI-RUA environment, gateway stopped). One fixed protocol under one deployment file and a condition label, N repetitions, through the agent's own path (`Harness` over the real backend, every motion a `robot_action` preview + execute via `unified_actions.dispatch`; no model, no perception service). Tests: empty-gripper transits (lateral sweeps and radial moves at several heights and reaches, and the `gravity_sign` pair across the sign change of the shoulder's gravity torque, both poses at least 12 cm above the base plane), a cube carry, and a `contact_stop` press with Present_Load sampled from the worker telemetry (§6.1). It writes `bench-<condition>.json` with per-stage metrics and their mean / sample standard deviation over the repetitions; `compare` prints conditions side by side. The script's docstring is the protocol: the exact commands per condition, what each condition changes in the deployment copy, and how to read the comparison. Conditions differ by deployment settings only. `--speed-scale` defaults to 0.5 for first runs. A deployment with `ik_service` needs the cuRobo tunnel up; otherwise remove the section from every condition file so all of them plan with the local IK. The contact test presses on a rigid block (its commanded target must stay inside the deployment `workspace` box, whose floor is just above the table).

- Code: `ni_rua/robot/backends/so101/motion_compensation.py`, `ni_rua/robot/model/so101_gravity.py`, `worker.SO101Adapter._carry` / `_feedforward` / `_segment` / `_arrive` / `_move`, `deployment.validate_motion_compensation` / `active_motion_compensation`, `contract._compensation_sentence`.
- Tests: `tests/test_so101_motion_compensation.py` (carry, zero under contact, guards without the offset, transit from scripted sag, gravity torque against a hand-computed point mass and the potential-energy gradient, feedforward, no-response, deployment validation), `tests/test_so101_motion_bench.py` (bench metrics on recorded data; both scripts end to end on servo models).
- **Not yet run on the real arm.** The servo models in the tests (constant sag, pure-P joints under URDF gravity, a static-friction band) prove the software contract only. Which mechanism to enable is decided by the bench.

### 4.13 Bounded holding squeeze (`gripper.hold_squeeze_rad`, `grip_squeeze_rad`)

**Problem (measured on the development unit, 2026-11-16).** The jaw servo (STS3215) is a position-P servo. A closing stage commands `goal_normalized` 0, i.e. `gripper.jaw_closed_rad`; the jaw stops on the object, or on the other fingertip when empty, well before that goal, and the servo then pushes at its torque limit (LeRobot `configure()` sets the gripper to `Max_Torque_Limit` 50 %, `Protection_Current` 50 %, `Overload_Torque` 25 %) for as long as the goal is held. An empty close with 1 mm rubber pads stopped 15-19 encoder ticks short of the closed limit and held `Present_Load` 18-22 % continuously; a 29 mm cube stalls far deeper. After about ten carry / press cycles the gripper read 46 °C (other joints 31-33 °C) and servo id 6 stopped answering (`servo_missing`) until a power cycle.

**Mechanism (`SO101Adapter._hold_squeeze`).** After a closing jaw command has settled (`_jaw_measurement`), if the jaw rests further than the squeeze from its commanded goal, the worker writes one more goal: the measured jaw angle minus the squeeze, toward closed. The steady holding torque is then about (servo P gain × squeeze) instead of the torque limit. The decision uses the request's own numbers only: commanded goal, measured angle, squeeze. Nothing is remembered across stages except the goal the servo holds.

| Key / field | Where | Meaning, unit, range, default |
| --- | --- | --- |
| `gripper.hold_squeeze_rad` | deployment | Squeeze in **rad of the jaw joint** (URDF angle, the unit of `jaw_closed_rad`). Range `[0, |jaw_open_rad − jaw_closed_rad|]`. `0` holds the measured contact angle. **Default `null`: no squeeze, a closed jaw keeps pushing toward its commanded goal** (the earlier behaviour, kept so the change can be ablated). |
| `gripper.max_hold_squeeze_rad` | deployment | Largest `grip_squeeze_rad` a stage may request (rad of the jaw joint); at least `hold_squeeze_rad`. Default `null`: the jaw range. Only valid together with `hold_squeeze_rad`. |
| `grip_squeeze_rad` | `robot_action` gripper stage, or move stage with `gripper_goal_normalized` | Squeeze of this jaw command (rad of the jaw joint), `[0, max_hold_squeeze_rad]`, default the deployment value. Larger = firmer hold (heavy or slippery objects), smaller = gentler (fragile ones). Refused above the maximum, on a move without a jaw command and on observe stages; **absent from the schema, and refused, on a deployment without `hold_squeeze_rad`**. Part of the plan digest. |

In fingertip travel (CAD, `so101_gripper_geometry.opening_width_m`; `contract.squeeze_fingertip_travel_m`): 0.010 rad closes the fingertip gap by about 0.8 mm at 10-50 mm openings (0.65 mm near 5 mm, 0.73 mm near 80 mm); one encoder tick (0.00153 rad) is about 0.12 mm. A squeeze of 0.020 rad therefore asks for about 1.6 mm less gap than the object left. With compliant pads part of that is taken up by the pads, so the servo's remaining error (and torque) is smaller than P × squeeze: the bench measures what is left (`squeeze_remaining_rad`).

**Interactions.**

- **`goal_normalized`.** Only a command that moved the jaw goal toward closed can be squeezed, and only when the settled jaw rests more than the squeeze short of that goal. An opening command, a close that reaches its goal (within the squeeze of it: the commanded goal already holds no harder than the squeeze would) and a jaw still moving at `gripper.settle_timeout_s` keep the commanded goal. The hold goal is never beyond the commanded goal.
- **Grip guard.** The guard judges the reading taken **after** the jaw settled at the hold goal: `gripper_execution.measured_opening_m`, `grip_facts.measured_aperture_m` and `grip_guard.measured_aperture_m` are that one reading. The reading at contact, before the goal change, is `gripper_execution.hold_squeeze.contact_opening_m`. A rigid object reads the same at both; a soft one may read less after the squeeze, and a guard minimum then applies to what the jaw actually holds. `jaw_not_settled` is unchanged (no squeeze is applied to an unsettled jaw).
- **`checkpoint_after_gripper`.** Unchanged: the pause comes after the jaw stage, so after the squeeze. The pause's hold commands resend the hold goal. The rule compares a stage's goal with the last *commanded* goal, not with the hold goal.
- **Post-grasp slip re-check.** Not available with a holding squeeze, and reported so (`stage_plans[<jaw stage>].grip_guard.post_grasp_recheck = {available: false, reason}`, `robot_capabilities.grip_guard.post_grasp_recheck`, the grip-guard text). The re-check needs a jaw that collapses to the empty-close gap when the object leaves; with a bounded goal the jaw closes toward its hold goal and stops at the STS3215 servo's static-friction deadband above `hold_goal` (`~0.03-0.04 rad` at `P=16`, not `0`). Every later move without a jaw command reports `grip_hold = {hold_goal_normalized, squeeze_remaining_rad}`: the measured jaw angle minus the hold goal. While a rigid object resists it stays near the squeeze; when `squeeze_rad` exceeds the friction deadband (e.g. `0.08 rad`), losing the object drops `squeeze_remaining_rad` toward `~0.03-0.04 rad` and shrinks `grip_facts.measured_aperture_m`, whereas a squeeze inside the deadband neither holds firmly nor drops on object loss. It is a reported fact; nothing pauses on it.
- **Later moves, waits, pauses.** Every command resends `jaw_goal`, which is the hold goal until the next jaw command. The next jaw ramp starts from the hold goal (no goal jump). A preview made while a hold is in effect counts that ramp's commands from the hold goal; inside one batch the preview cannot know whether a closing stage will meet contact, so it counts a later jaw stage from the commanded goal and the executed ramp can be a few commands shorter than `stage_costs`.
- **Stop / hold / resume.** `_stop` (operator hold, checkpoint pause, interruption) resends the hold goal; a resumed plan continues with it. A stop that arrives while the jaw settles at the hold goal fails the stage with the hold goal kept. `torque_on` (after `relax`) pins the jaw goal to the measured jaw, as it pins the arm: no squeeze is in effect afterwards.
- **During-motion jaw ramp (`gripper_motion`).** The ramp runs to the commanded goal inside the move, so between the end of the ramp and the arrival the jaw pushes as before; the squeeze is applied after arrival and settling, like a gripper stage. For a long transport prefer closing in a gripper stage.
- **Single `robot_gripper` requests and operator joint moves** use the deployment squeeze.
- **Observed state.** `robot_state.arms.<arm>.gripper.goal_normalized` / `command_normalized` stay the commanded goal, so their readers (held-object row, grasp review, `in_hand_tracking_error`) still see a jaw commanded closed. The servo's goal is in the receipts and in telemetry `q_goal_rad.gripper`.

**Receipts** (`gripper_execution.hold_squeeze` of a closing jaw stage; absent on a deployment without the key and on opening commands): `squeeze_rad`, `applied`; when not applied `reason` (`jaw_within_squeeze_of_goal` | `jaw_not_settled`); when applied `contact_jaw_rad` (rad) and `contact_opening_m` (m) at contact, `hold_goal_normalized` / `hold_goal_rad` (the commanded hold goal) and, when the driver reads it, `contact_load_pct` / `hold_load_pct` (jaw `Present_Load` before the goal change and after settling at the hold goal, percent of maximum torque; uncalibrated, not a force). The hold command is one dispatched command (`planned_trace.row_kinds` entry `hold_squeeze`). Preview: `stage_plans[<jaw stage>].hold_squeeze_rad`.

**Rubber pads.** The fitted 1 mm pads need no offset here. The squeeze is relative to the measured contact angle, so it is independent of where contact happens; and the empty-close reading did not grow with the pads (3.9-4.2 mm against 4.2-5.1 mm before), so `empty_close_aperture_m` and the CAD opening map stay as they are. Re-measure `empty_close_aperture_m` as after any hardware change (§4.1).

**Bench** (`scripts/so101_grip_bench.py`, NI-RUA environment, gateway stopped; same Harness / `robot_action` path and helpers as the motion bench). Per condition (a deployment copy differing in `gripper.hold_squeeze_rad` only) and repetition: an empty close held `--hold-s` (default 60 s), then a cube grasped at `--grasp-centre-m` / `--object-width-m` / `--fixed-jaw-yaw-deg`, held, lifted, moved laterally at `--transport-speed-scale` (default 1), set down and released. It samples the gripper's `Present_Load`, `Present_Current` and `Present_Temperature` from worker telemetry and reports the jaw's closing between grasp and release, `grip_hold.squeeze_remaining_rad` per transport stage and any guard pause. Every hold starts behind a temperature gate (`--start-temperature-c`, default 36 °C, jaw open while waiting) and ends at once at `--abort-temperature-c` (default 42 °C). Run the gentlest condition first and the baseline (`null`) last:

```bash
D=~/.cache/ni-rua/so101/grip
mamba run -n ni-rua python scripts/so101_grip_bench.py run --deployment $D/so101.sq-010.json \
    --condition sq-010 --out $D/sq-010 --repetitions 3 --hold-s 60 \
    --grasp-centre-m 0.22 0.06 -0.0055 --object-width-m 0.029 --fixed-jaw-yaw-deg 90 \
    --carry-over-xy-m 0.24 -0.08 --carry-height-m 0.08
mamba run -n ni-rua python scripts/so101_grip_bench.py compare $D/*/grip-*.json
```

The script's docstring is the protocol (conditions, gate, how to read the comparison). The jaw reading cannot tell a lost cube from one squeezed further: watch the transport.

- Code: `worker.SO101Adapter._hold_squeeze` / `_squeeze` / `_gripper` / `_ramp_jaw` / `_grip_hold` / `_send` (jaw bookkeeping), `deployment.hold_squeeze` and `FIELDS["gripper"]`, `embodiments.HoldSqueeze`, `grip_guard.validate_grip_squeeze`, `robot_action_schema` (`grip_squeeze_rad`), `contract._hold_squeeze_text` / `post_grasp_recheck_unavailable_reason`, `driver.sts_current_ma` / `sts_load_pct` / `decode_sts_status_block` / `lerobot_max_relative_target`.
- Tests: `tests/test_so101_grip_squeeze.py` (default off, hold goal after contact, no squeeze without contact / on opening / unsettled, zero-delta jaw command clearing prior squeeze, stage override and validation, guard after the squeeze, re-check reported unavailable and `grip_hold`, hold goal across moves, waits, pauses, operator hold and `torque_on`, during-motion ramp, receipts, 11-byte status block vs stale 2-byte reply, per-motor `max_relative_target`), `tests/test_so101_grip_bench.py` (metrics on recorded data, the bench end to end on the fake arm, the temperature gate and confirmed abort).
- **Measured on the development unit (2026-11-17).** At `P_coefficient = 16`, `0.03 rad` (`C1`) stayed inside the unloaded STS3215 static-friction deadband (`~0.03-0.04 rad`), held with only ~13 % load, let the 28.5 mm cube slip during carry, and left `squeeze_remaining_rad` at `0.038 rad` on an empty jaw. `0.08 rad` (`sq-080`, ~6.4 mm fingertip travel) held the 29 mm cube across 60 s holds and full-speed transports with ~20.2 % mean load (vs 50 % limit), ~125 mA mean / 240 mA peak current (vs 806 mA peak without squeeze), 0-1 °C temperature rise and zero slip (`jaw_closed_further_m <= 1e-6 m`).

## 5. Calibration workflow

Everything in this section is **operator-only**: workflow state, previews, thresholds and bundle contents never enter model context. The model only sees the resulting calibrated profile (§3.2) and, through the actor instructions, the measured table height (`joint_calibration.table_z_m`, `ni_rua/robot/backends/so101/contract.py`).

### 5.1 Purpose and what it produces

The cameras and the arm base can be moved, and the servo zero points are only as good as the last calibration. The guided workflow (page `/calibration`, §5.4) replaces the earlier hand-run procedure and the console's one-click recalibration buttons. It measures everything the actor's geometry depends on and installs it as one versioned, reversible unit:

- **Wrist camera:** intrinsics (`K`, distortion) and the hand-eye transform `T_link_camera` (eye-in-hand, parent link `gripper_link`).
- **Fixed camera:** intrinsics and the extrinsic `T_base_camera` (eye-to-hand).
- **Table height** `table_z_m` (m, in `so101_base`) and the **joint zero offsets** (`joint_mapping.<joint>.offset_rad`, rad). These come from one joint least-squares solve (S5) that combines three kinds of evidence: board views seen by the wrist camera, fingertip touches on the table (S1), and the prior "board top face = table + board thickness". Solving them together matters because a joint zero error otherwise leaks into the hand-eye and fixed-camera numbers.

Physical setup: one printed ChArUco board (render it with `scripts/so101_charuco_board.py`; the board spec of the active calibration file is the default) lying flat on the table where **both** cameras see it, and the measured board thickness (`calibration.board_thickness_m`, m: printed face above the table surface; editable per run on S2). The workflow never moves the board; the operator does, between steps, when the page says so.

Safety is the existing motion stack: motion steps run on the worker main thread through `move_joints`, at `calibration.speed_scale` (S2/S4) or `calibration.touch_speed_scale` (S1/S6) times the effective speed (deployment maximum times the operator `speed_scale_cap`), with the usual tracking and stall guards. `hold`, E-STOP and **Cancel** preempt any step, and the arm then stays held where it stopped.

Code: steps, modes, criteria and dependencies are declared as data in `ni_rua/robot/calibration/workflow/state_machine.py` (pure module shared by worker, gateway and page); execution is `ni_rua/robot/calibration/workflow/runner.py` (`CalibrationWorkflow`); storage is `ni_rua/robot/calibration/workflow/bundle.py` (§5.6); the gateway is `ni_rua/apps/gateway/extensions/calibration.py` (§5.5); the page is `web/calibration.{html,css,js}` (§5.4).

### 5.2 Two entry points

`calib_workflow_begin` starts a **run** (`<bundle>/runs/<UTC>-<id>/run.json`) with one of two modes (`state_machine.MODES`, `plan_for`). Starting a run closes the previous one. Steps that are in the plan are the ones the page asks for; every other step can still be run by hand, and data a run does not recreate (wrist calibration, fixed intrinsics, touches) is taken from the active calibration instead.

- **Full initial calibration (`first_run`)**: first use, or start over. Plan is S0 to S6 in order.
- **Recalibration (`recalibrate`)**: the operator ticks what changed, and the plan is the union of the mapped steps in S0 to S6 order (`CHANGES`). At least one change is required. S0 is included when “everything” is selected; otherwise the page still shows it as a manual check.

| Change id | Page label | Steps redone |
| --- | --- | --- |
| `fixed_moved` | Fixed camera moved (`fixed_moved`) | S4, S5, S6 |
| `fixed_replaced` | Fixed camera replaced / refocused (`fixed_replaced`) | S3, S4, S5, S6 |
| `wrist_bumped` | Wrist camera knocked (`wrist_bumped`) | S2, S4, S5, S6 |
| `table_or_base_moved` | Table or base moved (`table_or_base_moved`) | S1, S2, S4, S5, S6 |
| `board_changed` | Calibration board replaced (`board_changed`) | S2, S4, S5, S6 |
| `everything` | Complete recalibration (`everything`) | S0 to S6 |

Example: `plan_for("recalibrate", ["fixed_moved", "wrist_bumped"])` is `["S2", "S4", "S5", "S6"]`. S5 and S6 are in every plan because any camera or geometry change must be re-solved, reviewed and verified.

### 5.3 Steps S0 to S6

Step ids are stable. Every pass threshold is a field of the deployment `calibration` section (units and ranges in `deployment.py` `FIELDS`, starting values in `configs/so101.example.json`); `state_machine.evaluate` compares the runner's measures with them. A missing measure fails its criterion, so an incomplete step is never `passed`; `info` criteria are shown but never fail a step. Measures are SI (m, rad, px, s); the page shows m as mm.

| Step | Motion | Operator action | Machine action | Pass criteria (measure, rule, deployment field) |
| --- | --- | --- | --- | --- |
| **S0 Device self-check** (`self_check`) | no | Servo 12 V supply on, USB connected, both cameras connected and unobstructed. | Reads the power-on `preflight.json`; reads servo voltage, temperature and torque from the bus; grabs a fresh frame per camera and compares its size with the configured one; checks joint telemetry age. | `preflight_ok`, `power_ok` (voltage and temperature inside the `preflight` section), `torque_ok`, `cameras_ok` all true. `telemetry_age_s <= selfcheck_max_telemetry_age_s` (s). |
| **S1 Joint zero and table height** (`table_touch`) | yes | Clear the board and every object from the table in front of the arm; keep hands away and stay next to the E-STOP. | At each `touch_points_xy_m` point (m, `[x, y]` in `so101_base`): hover at `touch_hover_z_m`, close the jaw (`touch_jaw_normalized`), descend `touch_step_m` every `touch_step_period_s`, record an obstruction candidate when the joint lag (hover sag compensation kept) exceeds `touch_contact_lag_rad` or the shoulder_lift/elbow_flex/wrist_flex load rises `touch_contact_load_delta` above free air, for `touch_confirm_reads` reads after `touch_warmup_steps`. Records joint angles and the lowest gripper mesh point, lifts, then returns home. Core: `ni_rua/robot/calibration/workflow/touch_off.py`. | `contacts == planned points` (count of acquired obstruction candidates equals the number of `touch_points_xy_m`; this is not independent physical verification). A point that never touches before `touch_min_z_m` ends with `no_contact_before_min_z`. |
| **S2 Wrist camera intrinsics and hand-eye** (`wrist_grid`) | yes | Lay the board flat where both cameras see it and fix it; check the board form on this step (squares, marker edge, thickness) against the real board; do not move the board afterwards. | Checks the board in both cameras (`min_board_corners`), then visits `home.q_rad` plus the grid of `pan/flex/roll/lift_offsets_rad` (rad) at `speed_scale`; after settling (`arrival_tolerance_rad`, `settle_timeout_s`) keeps a sample when the wrist camera sees `sample_min_wrist_corners` corners, and photographs both cameras. Parks at home, then solves; samples whose board PnP error exceeds `max_pnp_reprojection_px` are dropped from hand-eye. Budgets: `capture_timeout_s`, `solve_timeout_s` (s). | `kept >= min_samples`. `wrist_intrinsics_rms_px <= max_intrinsics_rms_px` (px). `hand_eye_translation_rms_m <= max_hand_eye_translation_m` (m). `hand_eye_rotation_rms_rad <= max_hand_eye_rotation_rad` (rad). |
| **S3 Fixed-camera intrinsics** (`handheld_fixed`) | no | Hold the board in front of the fixed camera, pause about 1 s at different positions, also near and far and tilted; press **Solve**; then lay the board flat again for S4. | Worker-side `HandheldCapture` reads the worker's own fixed `CameraSource` (never a second process; frames whose size differs from `cameras.fixed.width/height` are dropped). `HandheldViewSelector` keeps views that are still (`handheld_still_px`) and novel (`handheld_min_novelty`), reports a 3x3 image-grid coverage plus near/far and tilt buckets and guidance, and stops at `handheld_target_views` or `handheld_timeout_s`. Solve runs only with at least `handheld_min_views`. | `views >= handheld_min_views` (count). `fixed_intrinsics_rms_px <= max_handheld_intrinsics_rms_px` (px). |
| **S4 Fixed-camera extrinsic** (`fixed_extrinsic`) | yes | Board flat where both cameras see it, fixed; do not move it. | Visits the first `extrinsic_pose_count` poses of the S2 grid order (home first) with the board static, both cameras photographing at each. The wrist chain gives `T_base_board`; the fixed camera's board view then gives `T_base_camera = T_base_board @ inv(T_cam_board)`. Uses this run's S3 intrinsics, else the active ones; uses this run's S2 wrist calibration, else the active one. | `fixed_samples >= 1`. `fixed_chain_reprojection_px` (px) is reported for information only, because it still contains any joint zero error; S5 re-solves it with corrected offsets and gates it. |
| **S5 Joint solve and cross-checks** (`joint_solve`) | no | Review captured contact evidence and confirm `touches_physically_confirmed`; then solve. | Joint solve of joint zero offsets, hand-eye and table height from the S2 and S4 board views plus the S1 touches (this run's, else the active bundle's `touches.json`), weighted by `joint_offset_prior_rad`, `joint_plane_sigma_m`, `joint_touch_sigma_m`, `joint_pixel_sigma_px`, `table_prior_sigma_m`. Re-solves the fixed extrinsic with the new offsets, writes the candidate camera file, validates it like a reload and computes the old/new diff for the review screen. Nothing is installed. | `joint_reprojection_px <= max_joint_reprojection_px` (px). `board_height_error_m <= max_board_height_error_m` (m, board top versus table plus thickness). `max_touch_residual_m <= max_touch_residual_m` (m). `fixed_chain_reprojection_px <= max_fixed_reprojection_px` (px). `max_joint_offset_rad <= max_joint_offset_rad` (rad; largest `abs(offset_rad)` after the change). |
| **S6 Apply and verify** (`verify_and_apply`) | yes (verification only) | Review the old/new table, tick the confirmation and press **Apply**; then clear the table again and start the verification touches; roll back if verification fails. | **Apply** (no motion): stores a new bundle version, switches `active`, installs the camera file, writes the joint offsets into the deployment file (with `.bak-<UTC>` backup) and switches them live (§5.6). **Verification:** touches the table at `verify_points_xy_m` (m, choose points not in `touch_points_xy_m`) with the same `touch_*` parameters and compares each contact height with the new `table_z_m`; the result is stored in the version manifest as `verification`. | `contacts == planned points`. `max_verify_error_m <= max_verify_touch_error_m` (m). |

**Readiness checks** (`state_machine.READINESS`, listed per step in the step definition). The worker measures the `auto` ones live without touching the bus: `torque_on`, `telemetry_fresh` (age `<= selfcheck_max_telemetry_age_s`), `cameras_streaming`, `fixed_streaming`, `board_both_visible` (at least `min_board_corners` per camera, checked on fresh frames at start time), `touches_available`, `wrist_calibration_available`, `joint_inputs_ready` (S4 passed and not stale), `review_ready` (S5 passed and not stale) and `applied`. The `operator` ones (`table_clear`, `workspace_clear`, `board_matches`) cannot be seen by the machine: the page asks for a tick and sends them as `params.confirmed` with the start request. A step that fails a check is refused with `readiness_failed:<ids>` or `readiness_unconfirmed:<ids>` and nothing moves.

S6 verifies contact **height**, not independent XY alignment with visual targets. Low board reprojection error and a passing S6 result do not establish current lateral grasp accuracy. Diagnose XY misses using archived camera observations, requested TCP poses, encoder-derived FK residuals, and the single-moving-jaw geometry separately. FK residuals are not an external measurement of fingertip location. An independent XY check requires known visual targets at multiple workspace positions after the active task finishes; do not derive a global calibration offset from one missed grasp. `geometry_query` now derives the SO-101 `gripper_relative` reference from the CAD inner fingertips and measured named jaw angle. The reference is the current fingertip midpoint, not a constant flange offset or a measured contact point.

**XY consistency on the S6 page.** After the actor task ends, freeze a stationary pose with "Freeze Current Frame (Stationary)". Select the same fixed inner fingertip centre in the two original-image views, then run "Check Point XY". Repeat across positions, heights and orientations. Capture waits for measured settling, checks endpoint joint drift, and saves camera matrices, joint state and immutable image pairs under the workflow run's `xy/<snapshot_id>/`; it sends no motor goals. Captures remain asynchronous, with recorded joint-to-frame offsets and unknown exposure latency. Evaluation triangulates the selected pixels and compares against the CAD fixed tip transformed by captured FK. Large reprojection residual or insufficient parallax yields an unknown measurement, not a pass. The UI labels a pass as conditional for that sample; annotation uncertainty is `null`. Wrist extrinsics depend on FK, so shared calibration errors can remain invisible: this is camera/arm consistency, not an independent absolute XY certificate or an automatic calibration update. Existing known-target physical checks remain necessary.

Both actions reuse `POST /v1/calibration/steps/S6/start`, with the normal writer lease, request ID replay and idle actor/action/program gate. `params={action:"xy_capture"}` returns an opaque `snapshot_id`; the fixed image URLs are `/v1/calibration/xy/<snapshot_id>/{fixed,wrist}.jpg`. `params={action:"xy_evaluate",snapshot_id:"...",fixed_pixel:[u,v],wrist_pixel:[u,v],max_xy_error_m:0.005,max_reprojection_error_px:3,min_parallax_deg:1}` evaluates that capture. Pixels are original RGB coordinates, `max_xy_error_m` is the allowed XY Euclidean error in m (0..0.1, exclusive), `max_reprojection_error_px` is the maximum stereo reprojection residual in pixels (0..100, exclusive), and `min_parallax_deg` is the minimum ray angle in degrees (0..90, exclusive). All are required by the API; the editable UI starts at 5 mm, 3 px and 1 deg. Stale capture IDs or changed calibration/joint offsets are rejected. Results accumulate in `run.json:xy_verifications` separately from S6's height verdict, so old height passes never become XY passes. Measurement `result.status=unknown` is returned under operator `status=evaluated`, preserving the distinction from uncertain physical execution. Tests: `test_so101_xy_verification.py`, `test_gateway_calibration.py`, `web/calibration-xy.test.mjs`.

**Arrival correction evidence.** Motion receipts retain the configured bounded correction algorithm and exact requested tolerances. `arrival_correction.initial_tcp_residual` and each round's `tcp_residual` report encoder-FK residuals (`dx_m/dy_m/dz_m/position_error_m` in m, `orientation_error_rad` in rad). `termination_reason` records `round_limit`, `not_settled` or `joint_error_below_correction_threshold`; `offset_limit_reached` reports whether a joint goal offset reached the configured radian bound. These fields explain convergence and limits; they neither establish real fingertip accuracy nor justify pushing harder into contact. VGGT depth/object geometry is perception input, not a motor feedback loop: image-based correction requires fresh target and fingertip observations and explicit subsequent motion through the existing action tool.

**Staleness (`DEPENDS_ON`).** Data flows S2, S3 to S4; S1, S2, S4 to S5; S5 to S6. Re-running a step marks every transitive dependent that has a result as `stale`, and marks the pending candidate stale. `calib_apply` then refuses with `candidate_stale_or_S5_not_passed`; re-run S5. Step statuses are `not_started`, `running`, `passed`, `failed`, `stale`. A worker restart during a step turns it into `failed` with `worker_restarted_during_step`.

S3 novelty and image-grid/distance coverage use a fixed full set of ChArUco inner corners projected by the detected ID-to-pixel homography, not the currently visible corner subset. Missing or newly detected corners alone therefore do not count as a new view. Tilt uses the board outline projected by that same homography; degenerate or horizon-crossing projections are rejected. Existing novelty and coverage thresholds are unchanged. These are approximate image-view descriptors (lens distortion is not yet calibrated), not an intrinsic-observability certificate: low reprojection RMS and many accepted views do not prove accurate intrinsics. The operator must vary distance and tilt about both board axes, rather than only translate/rotate a board lying flat. CPU regressions: `tests/test_so101_handheld_geometry.py`.

**Full-chain source and stop checks.** Failed, stale or running S2/S3 results cannot silently fall back to the active calibration. Recalibration plans require their selected upstream stages to finish; manual first-run reuse of an unstarted stage remains visible in `calibration_sources`, readiness and `plan_completion`, rather than claiming that all stages ran. S5 candidates bind their camera/board/mapping inputs and candidate artifacts; changed or missing bindings require a new solve before apply. Offset installation preserves exact values even below `1e-9 rad`; the old six-decimal mismatch is diagnosed without weakening S6 identity or the physical acceptance threshold.

All poses in a capture/touch sequence, including parking, retain the original stop-generation token and absolute adapter-monotonic deadline (seconds). `move_joints(expected_generation=None, deadline_s=None)` retains standalone behavior by default; internal multi-move callers provide the sequence token and earlier deadline. Stable position does not override failed settling or interruption. S2/S4 samples archive joint-before/after times, host capture intervals and per-camera read timing; camera exposure time and simultaneity remain unknown. S3 stop waits at most 5 s for its writer; `stop_confirmed`, `writer_stopped` and `sealed` stay false while it is alive, and solve/restart cannot consume or replace that writer. These are acquisition-thread facts, separate from physical stop confirmation. See `test_so101_capture_safety.py` and `test_so101_calibration_runner.py`.

**VGGT participation.** Existing `/points` geometry is scaled using the supplied support height and transformed using supplied camera calibration. It cannot independently validate either input. The offline VGGT evidence workflow checks same-image support sensitivity, retained camera-depth shape and optional scale from known board lengths, with disjoint held-out landmarks. It reports uncertainty and remaining intrinsic/extrinsic dependencies, does not infer contact, and does not automatically add learned-depth constraints to S5.

**Staleness of the live calibration (hint only).** Every `staleness_check_period_s` (s) at most, when both cameras see the board and the arm is idle, the worker projects the board pose obtained through the wrist chain into the fixed camera with the active calibration and compares it with the detected corners. Above `staleness_max_board_shift_px` (px, RMS) the status is `shifted`. Other results are `ok`, `board_not_visible` and `unavailable`. The hint never moves the arm and never changes the calibration; it only feeds the console badge (§5.4).

### 5.4 Page flow

**Contact evidence.** S1 completion means obstruction candidates were collected, not independently measured table contacts. S5 now requires the post-capture `touches_physically_confirmed` operator readiness check: review the actual traces and independent contact evidence before checking it. The attestation is bound to the input report and joint basis; changing either invalidates it. It is not a calibrated contact sensor or proof of the lowest-CAD-point hypothesis. S6 remains a held-out-position model-consistency test with the unchanged 5 mm example threshold, not absolute external metrology.

The detector repeats the same actual joint goal during confirmation, including wrist flex, and never counts residual against an obsolete frozen goal. Descent begins at measured hover FK. Requested Cartesian endpoints are solved within limits that also keep the fixed sag-compensated command legal; joint interpolation respects the effective speed and per-command angular bound. `touch_step_m / touch_step_period_s` is therefore an upper nominal rate, not a promised TCP speed. Traces distinguish the IK endpoint, expected goal FK, sent goal, measured joints, first candidate and confirmation. The fixed hover sag estimate is recorded and may be imperfect away from hover. Sequential joint/load read intervals and write attempts are retained, including uncertain writes; they do not imply simultaneous sensing.

**Video evidence.** S1/S6 record the existing camera streams as bounded full-resolution PNG sequences, with frozen calibration/mapping and host read intervals. Deployment controls are `touch_video_fps` (10 Hz, 0 disables), `touch_video_max_frames` (600 per camera), and `touch_video_max_duration_s` (60 s per attempt). Recording failures remain separate from physical receipts. SAM 3.1 video masks, separate feature tracking and offline review are bounded outside the control loop. SAM point prompts select masks; they do not provide material keypoint identities. Exposure times, contact and metric accuracy remain unverified. Neither inference nor automatic S5 admission occurs in the control loop.

Optional sliding is configured in the operator deployment `calibration` section and defaults to disabled. At a confirmed obstruction candidate it requests a finite outward/return XY path at constant base-frame z, without further commanded descent. It is a sampling experiment, not surface following or force control. Unknown contact, changing contact feature, structural deflection and motion of the table require independent observation and explicit segment review.

| Parameter | Default | Meaning and applicability |
| --- | --- | --- |
| `touch_slide_distance_m` | 0 m | Outward distance, followed by a return; 0 disables, maximum 0.05 m. |
| `touch_slide_direction_rad` | 0 rad | Direction in base XY from +X; ±π supported. |
| `touch_slide_step_m` | 0.001 m | Maximum requested XY step; the final step is shortened to the endpoint. |
| `touch_slide_max_duration_s` | 10 s | Budget for the whole out-and-back path, also limited by `touch_timeout_s`. Infeasible step counts are refused before sliding. |
| `touch_max_tracking_error_rad` | 0.15 rad | Stop if any sent-goal-minus-fixed-sag residual exceeds this bound during descent, confirmation or sliding. |
| `touch_max_load_rise` | 300 raw units | Stop on this rise of absolute servo load above the fixed measured-hover baseline, never warmup-adapted; not calibrated force or torque. Sliding requires finite load readings. |

All effective `touch_*` values are archived in the report. The deployment schema supplies only missing new fields for compatibility, never replaces explicit values, and validates units/ranges. Slide commands also obey joint/speed/command-delta and IK-residual bounds. Missing load, excessive residual/load, elapsed budget or disappearance of obstruction evidence stops the sequence without lift/park. Loss of obstruction evidence is not asserted to be physical release. Failed stop confirmation remains `unknown`; a measured hold is not a guarantee of zero contact force. `touch_speed_scale` still applies to hover/lift/park; descent interpolation and slide guards use the effective deployment joint-speed ceiling.

`contact_trajectories` preserves measured rad/rad/s arrays with `contact_evidence=unknown` and no inferred material contact point. The offline estimator uses separately reviewed intervals and explicitly identified link features, normalizes total weight per segment and rejects rank-deficient parameter choices. It does not automatically install its result or feed uncertain rows into S5. First-candidate CAD heights remain clearly conditional legacy S5 inputs after review; do not treat the same data's small fitting residual as validation. The developer camera callback now measures a distinct post-hold pose and records endpoint drift and asynchronous image timing.

S5 preserves explicit board thickness, including exactly 0.003 m. S6 verifies active/applied identity, installed bytes, loaded camera calibration and full manifest/live/driver/on-disk joint mapping both before movement and before recording a verdict; its table reference comes from the installed version. Mismatch prevents verification publication. Status uses the same current binding and never borrows an old pass for a changed mapping. Old versions without a recorded complete mapping remain readable but require a newly reviewed solve/apply before S6. Offset-only historical touch reports cannot enter a new S5 solve; acquire S1 again to record the full sign/offset basis. Process-local checks detect ordinary changes but cannot make non-cooperating external writes across several files transactional. Joint offsets are written without decimal truncation so install and reload use identical floats. No version was applied or rolled back while implementing this change.

`GET /calibration` (`web/calibration.html`, `calibration.js`, `calibration.css`; pure helpers in `calibration-model.js`, covered by `calibration-model.test.mjs`). The page polls `/v1/calibration/state` (1 s, 0.5 s while a step runs) and renders only typed fields; it never parses display text. The live preview depends on that poll: the worker redraws `calib_wrist.jpg`/`calib_fixed.jpg` inside `calib_workflow_status`. So the poll loop never waits on a write: a lease re-claim runs beside it, one at a time. It keeps polling after a page exception, which it shows as a banner until a refresh completes. Each poll is bounded by `POLL_TIMEOUT_MS` (8 s; a normal reply takes well under 1 s). A poll without an HTTP answer shows "Connection to gateway lost... retrying" and keeps the last state on screen until a poll is answered. Controls need the writer lease (**Acquire Control**); **Emergency Stop** (E-STOP) works without it. The lease (30 s) is renewed every 10 s, also while a step runs. `claim_input`/`renew_input`/`release_input` have their own gateway lock (`LEASE_COMMANDS` in `ni_rua/apps/gateway/extensions/operator_console.py`), so they never wait behind a step that holds the control lock. The lease still never changes hands mid-command. While any command holds the control lock, a claim from a window that is not the lease's last holder is refused at once with HTTP 409 `input_owner_busy_step`, also after that holder's lease lapsed; hold, estop and calibration_cancel still work from any window. The last holder keeps renewing and re-claiming. `GET /v1/run` reports `control_busy` (the control lock is held); after a busy refusal a page claims again only once `control_busy` is false. Before this, a 74 s S2 on the real arm lost the lease after 30 s. The page's re-claim then waited behind the step, and progress and preview froze around pose 9 of 21 while the arm kept moving. Anything that blocks a call is listed from the gateway's `blockers` (`BLOCKERS` in `calibration-model.js`).

1. **Start screen.** A card with the current calibration (valid or needed, age, active version, table height, residuals, the staleness check) and **two mode cards**: Full initial calibration (one button) and Recalibration (the six change checkboxes of §5.2 with a live "Will redo: ..." plan preview; the button stays disabled until at least one change is ticked). An unfinished run is offered as **Resume**. Below: the version history (§5.6) with per-version rollback.
2. **Stepper** (left, S0 to S6). Each entry shows one of the five statuses (Not Started, In Progress, Passed, Failed, Expired); steps outside the plan are shown dimmed as skipped but stay clickable.
3. **Per-step section.** Purpose, what the operator must do, what the machine will do, the live readiness list (auto checks with detail, operator checks as checkboxes), a start button, live progress phase and bar with the board-corner or touch-target overlay preview, cancel, and the **result card** (each criterion with measured value, limit and pass or fail, plus the failure reason). Motion steps say that admission closes while they run. S2 additionally shows the **board spec form** (`squares_x`, `squares_y`, `square_m` and `marker_m` in mm, `dictionary`, `thickness_m` in mm), prefilled from the active file and `board_thickness_m`, validated by the worker (`board_spec_invalid`; thickness in 0 to 0.05 m). S3 has start capture, a live preview with detected corners, coverage and guidance, **Solve** and stop. S6 has the review screen and the verification button.
4. **Review before apply (S6).** Available only when S5 passed on this run and is not stale. It shows the S5 cross-check results and an old versus new table: table height, each joint zero offset, hand-eye and fixed-camera translation and rotation change, focal lengths, and the residuals. The **Apply New Calibration** button stays disabled until the operator ticks "I have verified the old and new values". No number is written before that.
5. **Verification and rollback.** After apply, the page shows the new version and its predecessor and offers **Rollback to Previous Version**. Verification touches (S6) record `verification` on the version, which the history table displays (Passed or Failed with the largest error in mm).
6. **Console badge.** `/console` has a header badge (`web/calibration-badge.js`, covered by `calibration-badge.test.mjs`) linking to `/calibration`. It polls `GET /v1/calibration/summary` every 10 s. Calibration Valid (`valid`) requires a passing S6 record for the exact active version; a loadable file alone is insufficient. Failed S6 shows Height Verification Failed (`verification_failed`), and missing/nonterminal S6 shows Calibration Loaded, Unverified (`unverified`). Other states are Calibration Needed (`needed`, no loadable calibration), Possibly Stale (`maybe_stale`, staleness `shifted` or calibration consistency problems), and Calibration Status Unknown (`unknown`). The homepage uses the same presentation helper and shows the maximum verification error in mm. Older running gateways without summary verification are supported through a read-only state lookup; failed lookups never retain a green badge. It replaces the earlier loose recalibration, reload and hand-held buttons.

### 5.5 Gateway endpoints and worker operations

Gateway (`ni_rua/apps/gateway/extensions/calibration.py`, mixed into `Workbench`; tests `tests/test_gateway_calibration.py`):

| Endpoint | Maps to | Worker op (`robot.operator`) | Lease | Notes |
| --- | --- | --- | --- | --- |
| `GET /calibration` | page | n/a | no | 404 `calibration_page_not_built` until `npm --prefix web run build`. |
| `GET /v1/calibration/state` | snapshot | `calib_workflow_status` | no | Definition, run, readiness, bundle, staleness, plus gateway `blockers` (idle and motion), `torque`, `accepting_actions`. Answered while a step runs. |
| `GET /v1/calibration/summary` | badge | `calib_workflow_status(summary=true)` | no | Classified by `calibration_badge`: `valid`, `needed`, `maybe_stale`, `unknown`, `verification_failed`, `unverified`; includes active-version `verification` or null. |
| `GET /v1/calibration/steps/{step}` | step status | `calib_step_status` | no | |
| `GET /v1/calibration/preview/{name}.jpg` | previews | none (reads telemetry dir) | no | Whitelist `wrist`, `fixed`, `fixed_corners`; symlinks refused; `Cache-Control: no-store`. |
| `POST /v1/calibration/begin` | `calibration_begin` | `calib_workflow_begin` (`mode`, `changes`, `board`) | yes | |
| `POST /v1/calibration/steps/{step}/start` | `calibration_step_start` | `calib_step_start` (`step`, `params`: `confirmed`, `board` for S2, `action` capture/solve/stop for S3) | yes | Motion steps close admission. |
| `POST /v1/calibration/apply` | `calibration_apply` | `calib_apply` | yes | Closes admission. |
| `POST /v1/calibration/rollback` | `calibration_rollback` | `calib_rollback` (optional `version`) | yes | Default target is the active version's `previous`. Closes admission. |
| `POST /v1/calibration/cancel` | `calibration_cancel` | `calib_step_cancel` | **no** | Stops the S3 capture, or holds the arm like `hold`. |

The POST routes are aliases of `/v1/control` commands (`calibration_*`) with the same `request_id` idempotency, same-origin check and writer lease. An operator call that raises, or returns an unconfirmed stop, becomes `status: unknown` (HTTP 409, never resent), which needs `resume` with `acknowledge_unknown` exactly as in §6.3.

Gating (`CalibrationRoutes.calibration_blockers`, one function for refusal and page): every mutating call is refused with `calibration_blocked:<ids>` while a Codex turn (`codex_turn_running`), an action (`action_in_progress`), a program (`program_active`) or another calibration step (`calibration_step_running`) is active. Motion steps additionally need torque on (`torque_off`) and no unacknowledged unknown (`unknown_requires_acknowledgement_via_resume`). Motion steps, apply and rollback **close motion admission** (`harness.request_stop`) before the worker runs, because the earlier observation is stale afterwards; when the result is confirmed, admission was open before and no stop arrived meanwhile, it is **reopened with a fresh `robot_observe`**, otherwise it stays closed until `resume`. Cancel needs no lease and uses the safety lock, because cancelling only holds the arm; lease commands use their own lock, so neither waits behind the step's control lock (a claim from another window is refused with `input_owner_busy_step` until the step ends). Cancelling a motion step leaves admission closed.

Worker (`ni_rua/robot/backends/so101/worker.py` `operator` commands, parent side `backend.py`; tests `tests/test_so101_calibration_runner.py`): `calib_workflow_status`, `calib_step_status` and `calib_step_cancel` are preemptive (they read state and the latest frames, never the bus, so they answer while a step runs). `calib_workflow_begin`, `calib_step_start`, `calib_apply` and `calib_rollback` are serialized like `go_home`; the parent waits for `calib_step_start` up to the longest step budget (`capture_timeout_s + solve_timeout_s`, or `touch_timeout_s`, plus two `motion.max_timeout_s` and `ipc.response_margin_s`). A refused start returns `status: rejected` with a stable `reason` (`calibration_run_not_started`, `calibration_step_running:<step>`, `action_active`, `readiness_failed:...`, `S3_capture_running`) and nothing moved. Every receipt carries `physics_advanced`; the gateway adds `world: objects_not_reset` when it is true.

### 5.6 Bundle, apply and rollback

Everything is stored under the bundle directory, `calibration_bundle_dir` of the deployment, or `calibration/` next to `camera_calibration_file` when null. With the local deployment that is `~/.cache/ni-rua/so101/calibration/`:

```
versions/<UTC>/manifest.json            mode, changes, steps and times, board, joint offsets (and previous), table_z_m, previous version, diff, verification
versions/<UTC>/camera_calibration.json  the camera file this version installs
versions/<UTC>/joint_calibration.json   table height and joint zero offsets (also inside the camera file)
versions/<UTC>/touches.json             the S1 touches the joint solve used
versions/<UTC>/reports/<step>.json      per-step measures, criteria, pass or fail
runs/<UTC>-<id>/                        working directory of one run: run.json, sessions, candidate/
active                                  one line: name of the active version, replaced atomically
```

A version directory is written under a temporary name and renamed when complete, so a half-written version never appears; `-1`, `-2` suffixes avoid collisions. The deployment `camera_calibration_file` stays the **single file the worker reads** (§3.2); `active` only names which version it was installed from, and `matches_active_version` in the status says whether the file still equals that version byte for byte (it does not after a legacy install, §5.7).

**Apply** (`calib_apply`, refused with `no_candidate`, `candidate_stale_or_S5_not_passed`, `already_applied:<version>` or `action_active`):
1. If no version is active yet, the current camera file is imported as an `imported` version first, so the first apply has a rollback target.
2. The candidate becomes a new version with the S5 diff in its manifest.
3. The stored camera file is validated exactly like a reload, the joint offsets are written into the deployment file (`deployment.write_joint_offsets`: only `joint_mapping.<joint>.offset_rad` changes, comments are kept, the file is validated like `load_deployment`, the old file is copied to `<name>.bak-<UTC>`, replacement is one `os.replace`), and the camera file is copied onto `camera_calibration_file` with one `os.replace` and reloaded (`bundle.install_camera_file`). A reload failure restores the previous bytes and the previous deployment offsets, and the result is `install_failed`; nothing partial stays active.
4. The in-memory joint mapping is switched and the held goal is re-expressed in the new basis (`goal_new = goal_old + (offset_new - offset_old)`), so **the arm does not move**. Cached plans are dropped. The new calibration applies to observations captured after the command, and `calibration_id` changes from then on.
5. `active` is set to the new version. The version's camera file carries the `joint_calibration` record (`table_z_m`, offsets, residuals), so the table height reaches the actor instructions from the next start or reload of the profile.

Offsets and extrinsics are only valid together: the camera file notes that its extrinsics belong to the matching deployment offsets, and apply and rollback always switch both.

Successful apply/rollback receipts also return `table_z_m` from the installed camera file's `joint_calibration` (table-plane base-frame z coordinate, metres), `table_z_unit="m"`, and `table_z_frame="so101_base"`. A historical camera-only version returns `table_z_m=null`; the runner never invents a height from the camera reload identity. The gateway retains this value as installation evidence and reloads deployment before reopening admission; it does not mutate the frozen configuration. Set `features.vggt_support_z_m=null` to follow the calibrated table height frozen into each observation. Missing or invalid observation calibration makes that height unavailable; it never falls back to a guessed numeric height or reads the currently active file during replay. An explicit number in metres, including `0.765`, remains authoritative across apply, rollback and restart. Apply/rollback do not rewrite NI-RUA configuration files; choosing the support-height mode is an explicit configuration edit.

S6 runs after apply. Failure records the failed verdict on that active version; it does not automatically roll back the offsets or table height. Inspect the failed measurements before deciding on remeasurement or rollback.

S1/S6 raw touch reports retain `descent_traces` for operator diagnosis only. Each point records every descent/confirmation read: elapsed time (s), expected/measured/sent joints (rad), measured jaw (rad), commanded TCP z (m in `so101_base`), joint lag (rad), absolute load and warm-up reference (raw STS3215 units, shoulder_lift/elbow_flex/wrist_flex order), and consecutive trigger count. Unavailable loads remain null. The primary contact height, joints, lag and trigger all refer to the first hit in the confirmed sequence; `confirmation` separately retains the final hit. This adds no samples or motor commands and changes no contact thresholds. Lag/load detection and CAD/FK height remain estimates, not independent proof of physical table contact; visual or physical measurements are required to distinguish tracking lag from real contact.

**Rollback** (`calib_rollback`, optional `version`, default the active version's `previous`; refused with `rollback_target_unknown:<v>`, `rollback_target_is_active` or `action_active`): installs that version's camera file and its recorded joint offsets through the same path as apply, with the same failure restore, and no arm motion. If the run's own applied version was rolled back, the run's `applied_version` is cleared, so S6 verification needs a fresh apply. The page's history table can roll back to any listed version.

### 5.7 Developer tools and the earlier one-click path

The workflow supersedes these; they remain for development and are not part of the operator flow:

- `scripts/so101_touch_table.py`: CLI over `ni_rua/robot/calibration/workflow/touch_off.py` (the core of S1 and S6). Runs the real worker adapter in-process, so the gateway must not be running.
- `scripts/so101_calib_auto.py`: CLI over `ni_rua/robot/calibration/workflow/capture.py` (the S2/S4 pose-grid capture, optionally the gated solve).
- `scripts/so101_calibrate.py`: board render, manual and hand-held capture, offline `solve` and `verify` (`ni_rua/robot/calibration/workflow/hand_eye_solve.py`). Its offline code is also what the worker uses for the solves above.
- `scripts/so101_charuco_board.py` is **not** superseded: it renders the printable board.
- `scripts/so101_commission.py` and `scripts/so101_speed_probe.py` are commissioning tools (§10), not calibration.

The earlier console operations are still accepted by the worker and by `/v1/control` (`calibrate`, `reload_calibration`, `handheld_capture`, `handheld_solve`, `GET /v1/console/board_check`; code `ni_rua/robot/calibration/workflow/capture.py` `run_calibration`, `run_handheld_solve`; tests `tests/test_so101_recalibration.py`, `tests/test_so101_handheld.py`, `tests/test_gateway_console.py`) but no console button calls them any more. They use the same deployment gates and carry the old file's `joint_calibration` over to the new file, but they do not create bundle versions, touch the joint offsets or use S1 touches. Their install writes `camera_calibration-<UTC>.json` and `camera_calibration-previous-<UTC>.json` next to the active file and a `<stem>.robot_session.json` record, and their session images go to `<robot run_dir>/calibration/session-<UTC>-<id>/`. `reload_calibration` remains the way to re-read a camera file produced elsewhere.

### 5.8 Tests and the browser check

CPU tests (no hardware, no gateway on 8781):

| Layer | File | Covers |
| --- | --- | --- |
| Definition | `tests/test_so101_calibration_workflow.py` | Mode and change to step mapping; dependency staleness; every threshold is a deployment field and `evaluate` is strict; new deployment fields validated; bundle directory default; `write_joint_offsets` keeps comments, backs up, never touches the file on invalid input. |
| Worker | `tests/test_so101_calibration_runner.py` | Fake arm with hidden joint zero errors, a table and cameras rendering the true board: S0 to S6 end to end (injected offsets and table height recovered), apply, rollback, install failure leaving file, offsets and pointer unchanged, hold during a touch, re-run marks dependents stale, refusals, S3 capture and solve feeding S4. |
| Touch-off | `tests/test_so101_touch_off.py` | Contact logic (lag and load triggers, warm-up, confirm reads), report shape, stop semantics. |
| Solve | `tests/test_so101_joint_calibration.py`, `tests/test_so101_calibration.py` | Joint offsets, hand-eye and board pose from synthetic views; schema and offline solve. |
| Gateway | `tests/test_gateway_calibration.py` | Lease, gating, admission close and reopen, unknown handling, validation, route to worker op mapping, whitelisted previews, badge classification. |
| Web helpers | `web/calibration-model.test.mjs`, `web/calibration-badge.test.mjs` | Formatting, plan preview, step view, badge view. Part of `npm run check --prefix web`. |

```sh
PYTHONPATH=tests mamba run -n ni-rua python -m unittest test_so101_calibration_workflow \
  test_so101_calibration_runner test_gateway_calibration test_so101_touch_off test_so101_joint_calibration
npm run check --prefix web
```

**Browser check** (`web/check-calibration.mjs` against `web/fixture-calibration-server.py`): the real gateway `Workbench` over a fake SO-101 worker that answers `calib_*` with the real definition and `evaluate` verdicts but invented numbers, so it proves the page and gateway behavior, never calibration quality. It drives an S2 during which the page must renew its lease and, with claims stalled and the lease lapsed, keep progress and preview moving while a `/console` window is refused once (`input_owner_busy_step`) and does not claim again during the step; then a stalled state poll must show the connection-lost banner. It also drives the start screen (two cards, change checklist, plan preview), recalibration stepper skips, readiness gating, running steps, failure and stale states, the review screen with its confirmation checkbox, apply, verification, history, rollback, the console badge and a narrow layout. It refuses port 8781 and resets the fixture on start. It needs a `playwright` module (`PLAYWRIGHT_MODULE`, path to its `index.mjs`) and a Chrome executable (`NI_RUA_BROWSER_EXECUTABLE`; `NI_RUA_BROWSER` selects the engine, default `chromium`):

```sh
npm --prefix web ci && npm --prefix web run build      # builds web/dist including /calibration
mamba run -n ni-rua python web/fixture-calibration-server.py --port 8791 &
NI_RUA_URL=http://127.0.0.1:8791 PLAYWRIGHT_MODULE=/path/to/playwright/index.mjs \
NI_RUA_BROWSER_EXECUTABLE="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
NI_RUA_SCREENSHOT_DIR=/tmp/calibration-check node web/check-calibration.mjs
```

The fixture scenario knobs (`POST /fixture/calibration`: `fail_steps`, `step_seconds`, `bad_readiness`, `bundle_present`, `staleness`, `unknown_step`, `stall_claims_during_step`, `state_delay_s`), `POST /fixture/reset` and `POST /fixture/expire-lease` (lapses the writer lease now, as the 30 s lease does during a long real step) are documented in the fixture's docstring. Like the worker, the fixture redraws the previews on each full status call. Screenshots go to `NI_RUA_SCREENSHOT_DIR` (default `runs/web-calibration-check`). Software checks do not replace a run on the real arm: only a full S0 to S6 pass on hardware, with its residuals and verification touches, is evidence about calibration quality.


## 6. Operator console interface (shared by backend and console work)

The console runs on the local Mac and attaches to the existing gateway on 127.0.0.1. Everything in this section is **operator-only**. Telemetry frames, live joint streams, and console settings never enter model context or actor observations. The model sees the world only through `robot_observe` and receipts.

### 6.1 Worker telemetry (backend owns the producer)

The worker owns the serial bus and the cameras, so the console must never open them. The worker runs a telemetry thread at `telemetry_hz` (deployment config, default 10 Hz for joints and 5 Hz for frames). It shares one bus lock with execution, so reads never interleave with goal writes. It writes the following atomically (write to a temp file, then `os.replace`) under `<robot run_dir>/telemetry/`:

- `state.json`:
  - `{"schema":"ni_rua.so101.telemetry.v1","seq":int,"captured_at":UTC ISO,"monotonic_s":float}`
  - `"connected":bool,"torque_enabled":bool|null,"q_rad":{name:float},"q_goal_rad":{name:float}|null,"gripper_normalized":float|null`
  - `"tcp_position_m":[3]|null,"tcp_quaternion_xyzw":[4]|null` (FK of measured q)
  - `"motor":{name:{"temperature_c":float|null,"load":float|null,"voltage_v":float|null,"current_ma":float|null}}` (fields the bus can read; null otherwise. `load`: `Present_Load`, percent of maximum torque, signed; `current_ma`: `Present_Current`, mA, signed, 6.5 mA per register count)
  - `"motion":{"active_action_id":str|null,"phase":"idle|moving|holding|stopping|fault","tracking_error_rad":float|null}`
  - `"cameras":{id:{"ok":bool,"fps":float|null,"last_frame_at":ISO|null,"error":str|null}}`
  - `"faults":[str]`
- `<camera_id>.jpg`: latest frame, downscaled (longest side ≤ 640 px), operator preview only.

Telemetry failure never affects action receipts. If telemetry stops, the console shows the data as stale using `captured_at`; it does not invent values.

### Motion-only experiment video

SO-101 now supports the existing `features.record_video` and `features.export_video`
switches (both default `true`). Every configured camera records its native RGB
frames during Harness action execution. Recording starts at the first motor
command after validation/planning and ends before the terminal observation; model
requests, thinking, planning, observation delivery and checkpoint pauses add no
frames. Gripper execution, explicit physical `robot_wait`, settling and stop/stall
handling inside an action remain visible. This is an execution interval, not a
velocity threshold or an image-based motion detector. Operator calibration/home
motion is outside these Harness action intervals; calibration evidence keeps its
separate existing recorder.

Each camera appends retained frames directly into one H.264 MP4, effectively
concatenating action clips without saving idle footage, PNG sequences, or a second
encoding pass. Frames come from the existing UVC acquisition callback, not the
downscaled console preview. Width/height are unchanged. Playback uses the camera's
negotiated FPS (configured FPS when the device cannot report it); no deliberate
frame decimation or duplicate filler frames is applied. Normal even dimensions
use browser-compatible `yuv420p`; odd dimensions use `yuv444p` to avoid resizing,
which some browsers cannot decode. Output uses MP4 fast-start for browser seeking.

The deployment `camera` section accepts these backward-compatible defaults,
validated and documented in `deployment.FIELDS`:

| Field | Meaning / unit | Default / range | Example |
| --- | --- | --- | --- |
| `video_crf` | H.264 constant-quality factor; lower retains more detail with larger files | 23; 0–51, typically 18–28 | `20` for more texture detail |
| `video_queue_mb` | Pending raw RGB memory per camera, MiB; excludes one frame being encoded | 64; 1–1024, typically 32–128 | `128` to absorb longer encoder stalls |

Encoding uses `libx264`, preset `medium`, two encoder threads per camera, and no
audio. CRF 23 favors smaller files while retaining useful visual detail; actual
size and visual quality depend on scene motion/noise. Both `ffmpeg` (with libx264)
and `ffprobe` must be available to the SO-101 worker interpreter's environment.
Encoding never runs on the motor-control or capture thread. Queue overflow,
wrong-sized frames and encoder failures are reported independently of the action
receipt. They never cause action replay or change physical completion status.

On normal gateway shutdown, idle SO-101 runs automatically seal/register videos
before closing the worker and save `<run_dir>/video.json`. The existing observer
`POST /v1/observer/video` also seals videos while idle; repeated exports return
the same files. `POST /v1/observer/record` with `{"enabled":false}` pauses capture
without discarding frames; `true` resumes, or starts a new uniquely named recording
after a sealed export. `export_video=false` disables automatic Store registration;
worker close still finalizes already captured video bytes. `record_video=false`
disables automatic capture entirely. Failed export never skips robot cleanup.

Console **Save task video** (`save_video`, §6.3) seals the current recording while idle,
registers the per-camera MP4s like the observer export, composes them into one
side-by-side `combined.mp4`, registers that as an observer `native_video` artifact
(`observer.video_composed` event), writes `<run_dir>/video.json` (the export plus a
`composite` entry) and immediately re-arms recording into a new recording directory,
so later tasks of the same run keep being recorded. A later empty export (shutdown
or another click with no action since) does not replace a `video.json` that already
lists recorded segments.

Composite (`ni_rua/robot/backends/so101/motion_video_composite.py`, HUD renderer
`motion_video_hud.py`, metadata/usage resolver `motion_video_metadata.py`, tests
`tests/robot/so101/test_so101_video_runtime.py`): tiles in camera report order, each at
native size (canvas width = sum of widths, tile height = max height, padded to even px),
with a crisp pre-rendered black camera-id pill at each tile's top-left. When run
metadata is present (every NI-RUA run with `effective_config.json`), a `40 px`
solid-black header band is placed above the camera tiles showing
`Backend-Model-Effort` on the left and cumulative `Tokens` / `Cost` on the right
(read strictly read-only from `effective_config.json`, `agent-raw.jsonl` for OpenCode
`session.usage.updated` / Claude Code `ResultMessage`, `events.sqlite`
`runtime/token_accounting` / `runtime/token_usage`, and `summary.json`, showing `n/a`
for missing values without guessing). A `132 px` native-resolution FreeType timeline
footer sits strictly below the camera tiles (`Task Elapsed`, `Motion` vs `Skipped
Thinking` totals, `Clip` clock, `ACTIVE MOVE` and preceding skipped thinking,
proportional `MOVE`/`THINK` bar with playhead and wall-clock ticks). Per action segment
the camera with the most frames is the reference; every other tile shows its latest
frame captured at or before the reference frame's host capture time (its first segment
frame before that), or a black `no frame` tile when it has no frame in that segment.
Alignment is by host read time only, so tiles agree to about one frame interval.
Playback FPS is the highest source FPS; encoding is single-pass libx264 `medium`
(`-threads 2`, `yuv420p`) at the sources' CRF with `+faststart`, and `combined.mp4` is
published only after ffprobe checks codec, frame count and size. `combined.frames.jsonl`
maps every output frame to the reference camera, action and the source frame index shown
per camera (`null` = black tile); `combined.json` is the report (`status` `complete` |
`empty` | `failed`, `header_summary`, `timeline_summary`). Composition runs in the
gateway (`ni-rua` interpreter, numpy + Pillow + FFmpeg) and holds the console control
lock meanwhile (`hold`/`estop` are not affected). A sealed recording or run directory
can also be composed or previewed offline:
`mamba run -n ni-rua python -m ni_rua.robot.backends.so101.motion_video_composite <run_dir_or_recording_dir> [--overwrite] [--output OUT.mp4] [--preview-frame N --preview-out preview.png]`.

Files are in `<run_dir>/private/robot/videos/<recording-id>/`: one `<camera>.mp4`,
`<camera>.frames.jsonl`, encoder log and `manifest.json`. Experiment Library's
existing media browser finds the MP4s; registered videos remain observer-scoped.
These files/timelines are never delivered as actor observations. The JSONL maps
zero-based `video_frame` and `pts_s` (seconds from video start) to action identity,
segment index and host read interval in adapter monotonic seconds. Segment records
contain start/end host time and control step. Camera exposure timing, hardware
frame loss and cross-camera synchronization remain unknown; `max_capture_gap_s`
reports the largest within-segment host-frame interval without inferring hardware
drop counts. `complete` means received/queued frames were encoded, not that source
capture was physically lossless. Zero-frame intervals, invalid frames and queue
drops remain explicit. Encoding verification checks codec, frame count, dimensions
and average FPS before publishing `.mp4`; a failed `.partial` is not advertised as
a valid video. Forced process death may leave partial files and no sealed manifest.

Call path: `Harness.start → RobotBackend.record → worker MotionVideoRecorder`;
`Harness.execute → SO101Adapter.execute/_send → begin → camera sink → bounded
queue/encoder → end`; `Workbench.close → observer_video → save_video → Store`.
Tests: `test_so101_motion_video.py` (real FFmpeg, synthetic cameras),
`test_so101_backend.py` (admission, checkpoints and JSONL stop/export), and
`test_gateway.py` (shutdown export and cleanup on failure). RoboDojo retains its
existing simulation-clock recorder, which already omits model wall-clock waits.

### 6.2 Operator commands (backend owns the handlers; gateway forwards them)

Worker JSONL op `operator`, with `{"command": ...}`:
- `hold`: send the measured position as the goal for every joint (same as stop). Returns a stop confirmation as in §4.
- `relax`: torque off on all joints. **The arm falls under gravity.** The gateway accepts it only with `confirm:"relax"`. It latches a stop gate: physical admission stays blocked until `torque_on` and a fresh observation.
- `torque_on`: torque on, with the goal set to the present position.
- `set_runtime_limits`: `{"speed_scale_cap": 0.1..1.0}` (initial worker cap comes from `operator.default_speed_scale_cap`, default `0.9`). It multiplies the deployment max joint speed. It can only lower speed from the deployment value, never raise it. It is reported in the `effective_limits` of later receipts and in `robot_capabilities`, so the model is not misled.
- `go_home`: same motion as `reset` (arm home only).
- `board_check`: read-only ChArUco corner count per camera on the latest preview frames (no bus access, no motion; handled off the execution thread).
- `reload_calibration`: re-read `camera_calibration_file` (§3.2); refused while an action is active.
- `calibrate`: one-click recalibration (§6.4); main thread, serialized like `go_home`, with progress messages.

Stall detection: a joint counts as stalled only when the command moved at least `stall_min_commanded_rad` within `stall_window_s` and the measured motion is below `stall_min_progress_rad` (both in the deployment `motion` section). Slow final approaches inside servo deadband plus sag therefore never trip it.

These go through the backend object (`harness.robot.operator(command, **args)`), not a side channel into the worker. `hold` and `relax` must be callable while an action is executing, through the same path as cancel. Execution must not block them.

### 6.3 Console (gateway side, implemented)

Code: `ni_rua/apps/gateway/extensions/operator_console.py` (mixed into `Workbench`), page `web/console.{html,js,css}`, served at `GET /console`. Tests: `tests/test_gateway_console.py`.

Read endpoints (no lease, loopback only, `Cache-Control: no-store`, never stored as artifacts or events):
- `GET /v1/console/telemetry`: `{available, stale, age_s, stale_after_s, reason, state, joint_limits_rad}`. Staleness comes only from `captured_at` (`> 2 s` old, or more than 2 s in the future, means `stale`). A missing file, a wrong schema, or bad JSON means `available=false`. The UI dims stale frames and labels them `STALE`, and suffixes values with `(stale)`.
- `GET /v1/console/cameras/{camera_id}.jpg`: the latest worker preview frame from `<robot.run_dir>/telemetry/`. Symlinks and other path names are refused.
- `GET /v1/console/state`: operator gate (torque latch, unacknowledged unknowns, speed cap), resume blockers, recent receipts with status/reason/residual fields, current `task_plan`, runtime vs new-run settings, and the redacted effective config.

Commands, all through `POST /v1/control` (same `request_id` idempotency, same-origin, and writer lease):

| command | lease | effect |
| --- | --- | --- |
| `hold` | no | `harness.request_stop()` (admission closed), then `robot.operator("hold")` |
| `estop` | no | `request_stop` + `hold` + existing `stop` (interrupt the Codex turn) |
| `relax` | yes, `confirm:"relax"` | admission closed, torque off; latch `torque=off` blocks `go_home`/`resume` until `torque_on` |
| `torque_on` | yes | `robot.operator("torque_on")` |
| `set_runtime_limits` | yes | `{speed_scale_cap: 0.1..1.0}` forwarded (default `0.9` from `operator.default_speed_scale_cap`) |
| `set_model` | yes | `session.set_model(model)` for the next agent turn (`applies: next_agent_turn`). Free text: any id matching the console's `MODEL_ID` (letter/digit first, then letters, digits and `._:/@+-[]`, at most 200 characters) is accepted; the backend refuses what it cannot run and that error is shown. The console's Agent bar (§6.4) sends live changes as `apply_agent_settings`; `set_model` stays for other clients. Suggestions (`settings.runtime.model.options`, with the listing's capability fields) come from the session's `model_options()` (the provider's model listing of the running profile or legacy source), else the running profile's cached listing, else on a Codex-home run the pinned native Codex catalogue's listed models; there is no hardcoded model list. Refused with `set_model_conflicts_with_price_card` when a `price_card` is configured |
| `apply_agent_settings` | yes | `{model?, effort?, service_tier?}`: the live fields in that order through the `set_model` / `set_effort` / `set_service_tier` code paths, one receipt `{status, results: {field: {status, ...}}}`; a session-scoped difference (backend, profile, wire, `thinking_display`, Claude Code effort) applies nothing and answers `status: handoff_required` with the diff ([Agent Backends](agent-backends.md#apply_agent_settings)) |
| `handoff_run` | yes | `{backend, profile, wire_api, model, effort, thinking_display?, prompt?}`: finalise this run and start a successor run in the same gateway process on the same port (§6.4, [Agent Backends](agent-backends.md#handoff_run)); refused with a reason code (`agent_turn_running`, `torque_not_on`, `unknown_unacknowledged`, `price_card_mismatch`, ...) and nothing changed |
| `retry_model_requests` | yes | `ModelRequestProxy.manual_retry()`: lifts the relay's transport block after an explicit provider error (`model_transport.manual_retry_allowed`); never automatic. Typical use: the block shows the provider message (e.g. HTTP 400 `IMAGE_INPUT_UNSUPPORTED`), the operator switches to a model that can serve the request with `set_model`, then retries; the next model request goes upstream, no gateway restart. `model_transport_not_blocked` / `manual_retry_requires_explicit_provider_error` / `model_proxy_unavailable` otherwise. `GET /v1/console/state` reports `model_transport` (`blocked`, `error_type`, `manual_retry_allowed`, `http_status`, `classification`, `code`, `type`, `message`) |
| `set_effort` | yes | `session.effort` for the next `turn_start` (ModelConfig choices); backends without `capabilities.live_effort` report `applies="next_session_start"` |
| `go_home` | yes | idle turn/motion and no unknowns required; admission closed until `resume`; objects are not reset |
| `calibrate` | yes | same idle/unknown/torque gates as `go_home`; admission closed while it runs; afterwards reopened with a fresh observation only if it was open before, the result is confirmed and no stop arrived meanwhile (§5.7) |
| `reload_calibration` | yes | same gates and admission handling as `calibrate`; no motion |
| `resume` | yes | `harness.reopen_admission()`: fresh `robot_observe`, then reopen unless a stop arrived meanwhile |
| `steer` | yes | add operator text to the running turn (native `turn/steer`, `CodexSession.steer`); `no_active_turn_to_steer` when idle |
| `set_service_tier` | yes | `{tier: default\|priority\|ultrafast}` for the next model call (see below) |
| `save_video` | yes | idle run required (`observer_operation_requires_idle_run`); seal the motion recording, compose one side-by-side MP4 of all cameras, re-arm recording (§6.1). Returns `status` `complete` (with `composite.video_ref`) \| `empty` \| `failed`; a failure never changes run, admission or unknown state. The console shows a download link; `GET /v1/console/state` keeps the latest result as `video_export` |

Console UX: the header calibration badge links to `/calibration` (§5); the old individual calibration buttons have been removed. The writer lease is claimed automatically (one claim in flight, never awaited by the poll) and renewed every 10 s; `claim_input`/`renew_input`/`release_input` use their own lock, so they are answered while a long command such as `go_home` or a calibration step holds the control lock. Meanwhile only the lease's last holder may claim it; others get `input_owner_busy_step` (shown in Chinese with the code) and wait for `control_busy` to clear before claiming again. Enter sends. While a turn runs, sending becomes `steer`; when idle it starts a new turn. The native Codex TUI is embedded in `/console` (`web/console-tui.js`).

Service tier: `ModelRequestProxy` rewrites `service_tier` in each `/responses` request, so a change applies from the next model call, also mid-turn. Before accepting a non-default tier the gateway probes the provider with one tiny request identified as `ni-rua-native-model-relay/1` rather than Python urllib's default client. HTTP 2xx plus `response.completed` establishes that the request was accepted; the probe stops at that terminal event without waiting for the connection to close. `served_tier` retains the provider's reported tier, including `default` when the requested tier was `priority`. Acceptance does not prove acceleration, priority scheduling, or billing.

Only a structured HTTP 400/422 error explicitly naming `service_tier` is cached as unsupported. HTTP 401/403, rate limits, HTML/WAF blocks, unrelated parameter errors, incomplete streams and transport failures return `status: unknown` with `probe.supported: null`, leave the selected tier unchanged, and are not cached, so the operator can retry. Raw HTML error pages are not copied into the console. A structured JSON error body adds `probe.provider_error` (`http_status`, `code`, `type`, `classification`; sanitized identifiers only, never the provider message) and the code to `probe.error`; the reply also carries the probed `model`. The probe runs on the current model, so `model_not_found` (for example a relay answering an unknown model ID with HTTP 503) is reported as an unavailable model rather than a tier verdict: switch `set_model` to a model the provider serves, then select the tier again. Successful probes and explicit tier rejections are cached for the proxy's lifetime. Regression coverage: `tests/test_model_proxy_service_tier.py` and `tests/test_gateway_console.py`.

`hold`/`estop`/`relax` run under a separate `safety_lock`, so a long `go_home` never delays them. Neither `hold` nor `relax` takes `harness.robot_lock`, which execution holds for a whole action. `go_home` does take it. A backend without a callable `operator` gets `operator_command_unsupported:<command>`, and admission still closes on `hold`. An operator call that raises `RuntimeError`/`TimeoutError`/`OSError`, or returns an unconfirmed stop, becomes `status: unknown` (HTTP 409, never resent). `resume` then requires `acknowledge_unknown` to list exactly every unknown id (unconfirmed action receipts plus `operator:<request_id>`). Acknowledgement is recorded as an `operator.unknown_acknowledged` event. Receipts stay `unknown` in the archive and in run finalization.

Continue after stop: `stop` closes admission and nothing else reopened it before. The flow is now `stop` → (inspect) → `resume` (fresh observation) → `prompt` on the same thread.

New run: one run directory holds one agent session. From the console, `handoff_run` ("Switch and create new run" in the Agent bar, §6.4) finalises the current run and starts a successor run in the same gateway process and on the same port, also to another backend or profile. From the terminal, stop the gateway (Ctrl-C finalizes it) and start it again without `--run-dir`. A fresh directory is allocated under `run.experiment_root`. On shutdown, browser SSE and terminal WebSocket handlers are cancelled before HTTP cleanup waits for requests, so leaving the console open does not consume aiohttp's graceful-shutdown timeouts. This closes only the read-only browser streams; the supervisor still owns robot stop, native terminal/session cleanup and archive finalization. Ctrl-C and `SIGHUP` (terminal closed) request orderly shutdown rather than immediate forced termination. The robot worker shares the terminal's process group, so a hangup also reaches it directly; it exits without a native close, the gateway records `cleanup_status: worker_exit_failed` and the arm keeps its torque when `disable_torque_on_disconnect` is false.

```sh
npm --prefix web ci && npm --prefix web run build      # once, builds web/dist incl. console
mamba run -n ni-rua python -m ni_rua.apps.gateway --print-config --config configs/ni-rua.json   # check effective config
mamba run -n ni-rua python -m ni_rua.apps.gateway --config configs/ni-rua.json \
  --port 8781 --category so101 --title "desk-test" \
  --prompt "Call robot_capabilities and robot_observe, report the scene; do not move."
# open http://127.0.0.1:8781/console
```

`configs/ni-rua.json` (or a local copy) must set `backend.environment="so101_real"` and `backend.robot_model_id="so101"` once that backend is registered in `factory.py`. Keep a physical power switch for the servo supply within reach. Software stop depends on a healthy serial bus and worker.

### 6.4 Agent bar, model switching and handoff

The Agent bar at the top of the console's Agent panel selects backend, provider profile, model, effort and (Codex) speed ([Agent Backends](agent-backends.md#console-agent-bar-live-settings-and-handoff)). A model or a live effort/speed change applies to the running session (`apply_agent_settings`, from the next turn or model request). A backend, profile, wire or Claude Code effort change is a handoff: the tray shows the diff and the dry run, and "Switch and create new run" sends `handoff_run`. On the arm this means:

- The current run is finalised like Ctrl-C: motion admission closes, the worker closes the bus through its normal close op, the session and Store close. No motion is commanded. Torque follows the deployment's `disable_torque_on_disconnect` (false on the development arm: the servos keep holding their last goal while no worker owns the bus). The tray states which.
- The successor run starts a new worker on the same serial port and cameras (`private/robot` of the new run directory), re-applies the operator's speed cap before any agent turn and starts the new agent with the operator's prompt, or the observe-only default prompt. The remote services (VGGT tunnel) stay up and are not restarted.
- Refused unless the run is idle: no turn, action or program, unknowns acknowledged, torque on (`torque_not_on` after `relax`). A benchmark run never hands off.
- To use a profile for the arm's first run, point the run config at it, for example `agent.claude_code.upstream = {"source": "profile", "settings": "<profiles_dir>/<id>.local.json"}` with `agent.profiles_dir` set (the handoff reads the target profile from that directory). A provider that caps the number of images per request (e.g. a 10-image limit) does not fit the arm's multi-camera observations; pick another profile there.

Verified on the real arm (`disable_torque_on_disconnect: false`). Claude Code ran the observe-only startup turn (`robot_observe` only, no action). Then `handoff_run` went through `POST /v1/control` on port 8781 to Codex with no prompt:

| Measure | Value |
| --- | --- |
| `handoff_run` call (gate, dry run, listing key check) | 1.44 s |
| acceptance (`run.handoff_requested`) → successor Workbench built (`run.handoff_started`) | 1.32 s (predecessor finalised, worker exit confirmed) |
| acceptance → successor ready (`run.started`: new worker on the serial port and both cameras, Codex started, first turn admitted) | 7.92 s |
| joints before (predecessor telemetry) → after (successor telemetry), rad | all six unchanged except `elbow_flex` +0.0046 (0.26°, three encoder steps: the new worker latched the goal at the measured position, which sat 0.003 rad off the old goal under load); `torque_enabled` true in both |
| successor observation | `fixed` and `wrist`, 640×480 each |
| `cleanup_status` | `completed` for both runs (`native_close_acknowledged`, `process_exit_confirmed`, return code 0) |
| chain | predecessor `summary.successor_run_id` + `successor_handoff.status: handed_off`; successor `manifest.handoff.predecessor_run_id` + `request_id`, `prompt_source: observe_only_default` |

The gateway's `gateway_process ... ni_rua=<worktree>/ni_rua/__init__.py` line (same pid for both runs) and each worker's `so101_worker_start ... ni_rua=<worktree>/ni_rua/__init__.py` line in `private/robot/robot-worker.log` showed that both runs used the worktree's code. The gateway was stopped with SIGINT to its `python -u -m ni_rua.apps.gateway` pid (exit within 1 s), then its Ghostty instance was closed.


## 7. Reuse of existing tools (decisions and evidence)

| Need | Reused | Notes |
| --- | --- | --- |
| Servo bus, motor calibration, cameras, leader-arm teleop | LeRobot 0.6.1 (`SOFollower`, `OpenCVCamera`, `SOLeader`) | The driver wraps LeRobot; it never talks to Feetech registers itself. |
| Robot model | TheRobotStudio official `so101_new_calib_camera.urdf` + meshes | Packaged verbatim (comments and STL headers stripped; see `provenance.json`). |
| Camera intrinsics, ChArUco, hand-eye, PnP | OpenCV (`cv2.aruco.CharucoDetector`, `calibrateCamera`, `calibrateHandEye`, `solvePnP`) | We write only the orchestration (pose generation, sampling, residual gates). |
| FK | `urdf_manifest` + `so101_kinematics.fk` | Cross-checked against placo (LeRobot's `RobotKinematics`): max matrix difference 1.1e-15 over 200 random configurations. |
| IK | Our DLS + restarts (`so101_kinematics.ik`) by default; optional cuRobo GPU service ([§12](#12-ik-planning-the-curobo-ik-service-and-modereach)); placo 0.11.0 available as an alternative | LeRobot's placo wrapper does one solver step per call and has no restarts. Iterated 100× from seed = truth + N(0, 0.3 rad) on 200 targets: median position error 2e-16 m, but p95 0.15 m (local minima). Our solver converged 500/500. Revisit if we need placo's self-collision or velocity-limit tasks. |
| Isaac Sim / cuRobo (RoboDojo planner) | **Not usable on this Mac**; cuRobo IK runs on the GPU host as a service | Both require an NVIDIA GPU and CUDA on Linux. cuRobo is reached over the SSH tunnel for IK only ([§12](#12-ik-planning-the-curobo-ik-service-and-modereach)); Isaac Sim is not used. |
| Preview / collision digital twin | Candidate: MuJoCo 3.x (macOS arm64 wheel) + official `so101_new_calib.xml` | Would back `preview_batch` with table and self-collision checks. Not yet integrated. |
| Learned policies | Candidate: LeRobot ACT / SmolVLA / pi0 policies trained on SO-101 | Analogue of the X5 `robot_policy_step`. Needs data collected with the leader arm. Not yet integrated. |

## 8. Delivery plan

Each phase ends with CPU tests green and X5 regression checks unchanged. Refactor items are scheduled where they unblock SO-101. Each one is a separate change with its own regression tests.

| Phase | Work | Refactor items pulled in | Exit check |
| --- | --- | --- | --- |
| P0 | Robot information package, kinematics, calibration tooling, macOS Codex | — | done (CPU tests) |
| P1 | Merge and cross-review the real backend + console; full CPU suite | Backlog #1 (arms from capabilities), #2 (no X5 identity branches) | single-arm `robot_action` preview/execute passes through Harness with the fake driver; X5 tests unchanged |
| P2 | Camera calibration on hardware (manual capture → solve → optional automatic capture) | — | `camera_calibration.json` saved under the residual gates |
| P3 | First live motion: hold, small lift-and-return, gripper open/close, stop/cancel, telemetry in the console | Backlog #4 (thresholds from config) as SO-101 tolerances get tuned | receipts report honest residuals; stop confirmed |
| P4 | Model-facing correctness: SO-101 does not see X5 text | Backlog #3 (embodiment/gripper geometry), #5 (schemas/prompts from capabilities) | `robot_action` schema and instructions mention one arm, the SO-101 TCP and real-time clock only |
| P5 | End-to-end: a natural-language task typed in the console, executed by gpt-6.1-sol | Remaining backlog items when they block | task completed on the real arm, run archived |
| Later | MuJoCo preview twin; LeRobot learned policies | — | separate decisions |

### Motion authorization in development sessions

The agent may run **small, slow** motions without asking first. "Small and slow" means all of the following:
- each joint moves ≤ 15° from its measured start;
- the TCP moves ≤ 5 cm;
- joint speed is ≤ 20°/s;
- the gripper moves within its range;
- the motion is announced before it runs.

Anything larger or faster needs explicit user confirmation, and so do these regardless of size: torque-off (`relax`), the first motion of a new control path, and autonomous model-driven runs. Enforce these limits with the deployment speed cap. Do not rely on discipline alone.

## 9. Distributed deployment (Mac + remote Linux GPU server)

Host `<gpu-host>` (read-only check, 2026-11-04): Ubuntu 24.04, 8× RTX PRO 6000 Blackwell (96 GB each), 256 cores. Most GPUs were busy with other people's jobs at that time, so it is shared. It already has `~/miniforge3/envs/{NI-RUA,RoboDojo,lerobot,...}` and `~/code/NI-RUA` at the same HEAD.

Split by latency and safety:

| Component | Where | Why |
| --- | --- | --- |
| Robot worker (serial bus, cameras), Harness, gateway, console | **Mac** | Stop and hold must not depend on a network link; the cameras and USB are local |
| Codex / gpt-6.1-sol | Mac → model API | No GPU needed |
| VGGT/SAM perception (`features.vggt_endpoint`), learned policies (`policy.server_url`), MuJoCo/Isaac preview twin | **Server**, reached over an SSH tunnel | Need CUDA or heavy compute; already HTTP/WebSocket services with configurable URLs |

Design: no new RPC framework. Remote services keep binding to `127.0.0.1` on the server (the policy gRPC uses pickle and must never be exposed). The Mac reaches them through `ssh -N -L` forwards, and the `remote` section of the NI-RUA config declares which server and which programs:

```json
"remote": {
  "enabled": true, "host": "<gpu-host>", "remote_repo": "~/code/NI-RUA",
  "services": [{
    "name": "vggt", "remote_port": 8331,
    "workdir": "~/code/NI-RUA/services/vggt_height",
    "env": ["VGGT_HOME=<vggt-model-dir>", "VGGT_RUN_DIR=<perception-run-dir>"],
    "command": "bash start.sh", "stop": "bash stop.sh", "gpu": "auto"
  }]
}
```

(`configs/so101.remote.example.json` is the same block as a config overlay; it goes into the NI-RUA config such as `~/.cache/ni-rua/so101/ni-rua.json`, **not** into the strictly validated SO-101 deployment file `configs/so101.local.json`.) `features.vggt_endpoint` must point at the local end of the forward (the default `http://127.0.0.1:8331` matches `remote_port: 8331`); a config whose `vggt_endpoint` is loopback but not served by any `remote.services` entry is rejected at load. Only 8331 is forwarded: the VGGT process reaches the SAM 3.1 sidecar on 8333 on the server itself. The cuRobo IK service for the planner is a further `remote.services` entry (`curobo_ik`, port 8341) with its own forward and the deployment's `ik_service.url`; example and rules in [§12.3](#123-the-service-and-how-the-gateway-starts-it).

Fields (`ni_rua/core/config/sections.py` `RemoteConfig` / `RemoteServiceConfig` carry the same descriptions; unknown keys, wrong types and out-of-range values are rejected). Section: `enabled` (default `false`: simulation and X5 never touch it, and an unused section stays out of exported config snapshots), `host` (ssh alias, key login, `BatchMode` so it never prompts), `services` (one or more), `required` (default `false`: `true` makes the gateway refuse to start when a service cannot be made healthy, `false` starts anyway and perception reports unavailable), `stop_on_exit` (default `false`, see below), `health_interval_s` (30 s between health/tunnel checks), `poll_interval_s` (2 s between health polls while a started service loads), `connect_timeout_s` (10 s ssh `ConnectTimeout`), `command_timeout_s` (60 s per remote probe/start/stop command), `log_dir` (remote, default `~/.cache/ni-rua/remote`), `ssh` (executable), `remote_repo` (remote checkout whose source revision and dirty flag go into the provenance record), `record` (local provenance JSON; default `<run_dir>/private/remote_services.json` in the gateway). Per service: `name`, `remote_port`, `local_port` (default = `remote_port`; a busy local port is never taken over), `health_path` (`/health`), `command` (remote shell command run detached with `setsid nohup`; `null` = never start, attach only), `workdir`, `env` (`"NAME=value"`, a leading `~/` is the remote home; no credentials, the config is archived), `gpu` (`null` = untouched, `"auto"` = the GPU with the most free memory by `nvidia-smi`, or an index list such as `"4"` / `"0,1"`; exported as `CUDA_VISIBLE_DEVICES`), `gpu_min_free_mib` (default 15360, the README figure of about 15 GB for VGGT plus SAM 3.1; every selected GPU, `auto` or explicit, must have this much free or nothing is started), `startup_timeout_s` (300 s), `reuse_if_healthy` (default `true`), `stop` (remote command that stops what `command` started; else the process group of the detached command is signalled), `stop_timeout_s` (60 s).

What happens when the gateway starts (`python -m ni_rua.apps.gateway --config ~/.cache/ni-rua/so101/ni-rua.json ...`, also `python -m ni_rua.apps.remote --config <file>`; `--no-remote` skips it), before the first actor turn:

1. Open one `ssh -N -L 127.0.0.1:L:127.0.0.1:R` for all services (`ExitOnForwardFailure`, `ServerAliveInterval=15`). A busy local port is refused, except that an existing forward whose services already answer healthy is used as is (reported `external`) and never closed by us.
2. Per service: `GET health_path` through the forward. Healthy and `reuse_if_healthy` -> **reused** (never stopped by us). Otherwise, if `command` is set: pick/verify the GPU, refuse when the remote port is already bound by something unhealthy, run `command` detached in `workdir` with `env` (log `<log_dir>/<name>.<remote_port>.start.log`), record the GPU and the remote wrapper PID, then poll health until `startup_timeout_s` (an early failure such as a non-zero exit of the command is reported at once, with the log tail as the reason) -> **started**. `remote_pid` is the detached wrapper; for `start.sh` it exits after launching the two services, so stop them with the configured `stop` command, not with that PID.
3. A monitor thread reopens a dropped tunnel with backoff and refreshes health every `health_interval_s`; it never restarts a service (loading is slow and the host is shared). Services that were pending because the tunnel was down are ensured when it comes back.
4. On gateway exit the tunnel we opened is closed. A service we **started** is stopped only if `stop_on_exit` is `true`; the default is `false` because loading takes about 20 s and the box is shared, so it stays up for the next launch (stop it by hand: `ssh <gpu-host> 'cd ~/code/NI-RUA/services/vggt_height && VGGT_HOME=<vggt-model-dir> VGGT_RUN_DIR=<perception-run-dir> bash stop.sh'`). A reused service is never stopped, whatever `stop_on_exit` says.

Live status (operator only, never model context, no hashes): `GET /v1/remote/status` on the gateway (`{"enabled": false}` when disabled) and the same object as the `remote` key of the run's `runtime.json`. Shape: `{enabled, host, state (starting|ready|degraded|unavailable|stopped), required, stop_on_exit, tunnel: {state (up|down|external|closed|not_started), restarts, error, forwards}, services: {<name>: {name, remote_port, local_port, available, reason, origin (reused|started|none), started_by_us, gpu, remote_pid, remote_log, checked_at (UTC), health: {version fields of /health}}}, checked_at}`. `reason` is `null` when available, otherwise a short code such as `no_gpu_with_15360_MiB_free: most free is GPU 4 with 9000 MiB`, `startup_timeout_after_300s; log tail: ...`, `remote_port_busy_but_not_healthy`, `not_running_and_no_start_command_configured` or `tunnel_unavailable: ...`. The original console header exposes the reported service state and details through the shared perception component; saved snapshots are explicitly historical. Provenance (host, forwards, `/health` versions, remote checkout revision, started/reused, GPU, stop outcome, UTC times) goes to the record file only.

The standalone form is unchanged and starts nothing: `python -m ni_rua.apps.remote --host <gpu-host> --record ~/.cache/ni-rua/remote/<gpu-host>.json --remote-repo '~/code/NI-RUA'` (`ni_rua/apps/remote/tunnel.py`, `tests/test_remote_tunnel.py`). Code: `ni_rua/apps/remote/manager.py` (`RemoteManager`), tests `tests/test_remote_manager.py`, `tests/test_remote_config.py`, `tests/test_gateway_remote.py` (all with a fake ssh runner; no network). SO-101 config: `features.vggt_camera_id="fixed"`, `features.vggt_support_z_m=null` follows the frozen observation calibration, or an explicit number fixes the table height in `so101_base` metres; distorted SO-101 images are rectified before upload (`docs/object_table.md`).

Live check (2026-11-09, `<gpu-host>`, temporary config copies, never the user's file): with the existing standalone forward on 8331 and the service of 2026-11-08 running on GPU 4, `python -m ni_rua.apps.remote --config <tmp> --once` reported the forward `external`, the service `reused` and `/health` ok (`vggt-height-v2`, SAM 3.1 backend). A second instance on 18331/18333 (`VGGT_PORT`/`SAM31_PORT` in `env`, `stop_on_exit` true) was started on the freest GPU at that moment (GPU 7), became healthy after about 30 s through the forward, and was stopped again with `stop.sh`; the first instance and the standalone forward were untouched.

A reused local forward is also checked for disappearance; if its listener vanishes, the manager opens and owns a replacement. Shutdown or cancellation during startup prevents later services from launching and waits for the startup owner before teardown. A missing SSH executable is reported as unavailable (fatal only with `required: true`).

Starting or stopping services on the shared server is outward-facing: it now happens only from this config (`enabled: true` plus a `command`), local tests never trigger it, and it never disturbs processes this code did not start.

### 6.5 Guided calibration wizard (`/calibration`, S0–S6)

The `/calibration` wizard (`ni_rua/apps/gateway/extensions/calibration.py`, `ni_rua/robot/calibration/workflow/runner.py`, `web/calibration.{html,js,css}`) provides an end-to-end guided operator workflow replacing loose recalibrate/reload buttons on the console with a clean status badge in the header linking directly to `/calibration`.

- **Run Modes**:
  - `first_run`: Sequential execution of all steps S0 → S6 on a newly assembled or commissioned robot.
  - `recalibrate`: Targeted execution for specific hardware changes (`wrist_moved`, `fixed_moved`, `fixed_replaced`, `arm_bumped`, `board_replaced`, `table_changed`). Re-running an upstream step automatically marks downstream dependent steps stale (`DEPENDS_ON` DAG).
- **Configurable Board Spec**:
  - ChArUco board parameters (`squares_x`, `squares_y`, `square_mm`, `marker_mm`, `thickness_mm`, `dictionary`) can be viewed and edited on the start screen before initiating a run (essential for partial recalibrations like `fixed_moved` that skip S2), and sent with `beginRun`.
  - Non-S2 step screens display a read-only metadata summary of the active board specification (`${b.squares_x}×${b.squares_y} · square ...`) so the operator can always verify parameters on hardware.
- **Workflow Steps (S0–S6)**:
  - **S0 (Device self-check / Self-check)**: Read-only check verifying serial communication, 6 STS3215 servos, EEPROM calibration, supply voltage, telemetry freshness (`selfcheck_max_telemetry_age_s`), and camera streaming without moving the arm.
  - **S1 (Joint zero and table height / Table touch-off & plane/gripper check)**: Arm descends with fingertips closed to configured table touch points (`touch_points_xy_m`), detecting contact via joint lag and servo load; measures raw table height and verifies table clearance.
  - **S2 (Wrist camera intrinsics and hand-eye / Wrist grid capture & hand-eye)**: Automated motion across a 3D pose grid observing the ChArUco board; captures paired poses to solve wrist camera intrinsics and hand-eye transformation ($T_{\text{gripper}}^{\text{camera}}$).
  - **S3 (Fixed camera intrinsics / Hand-held fixed camera intrinsics)**: Operator holds the board in front of the static camera while an automated sampler selects still, diverse views across distance (near/mid/far) and tilt (frontal/small/large) buckets to solve fixed camera intrinsics ($K, D$).
  - **S4 (Fixed camera extrinsic / Fixed extrinsic)**: Static board observed by both cameras; using the solved wrist chain and board detections, solves the fixed camera pose ($T_{\text{base}}^{\text{camera}}$).
  - **S5 (Joint solve and cross-checks / Lie-algebra joint calibration solve)**: Joint optimization over Lie algebra $\mathfrak{se}(3)$ estimating joint zero-offsets (`joint_mapping.<joint>.offset_rad`), refined table height, and hand-eye transformations simultaneously, subject to touch contacts and prior constraints, followed by rigorous reprojection cross-checks.
  - **S6 (Apply and verify / Verification touch-off & apply)**: Displays parameter diffs, executes verification touches at distinct targets (`verify_points_xy_m`), atomically installs the new bundle version into `camera_calibration.json`, updates `so101.local.json` joint offsets with timestamped backup (`.bak-<UTC>`), and reloads the gateway deployment (`reload_deployment()`). VGGT follows new observation calibration only when `features.vggt_support_z_m=null`; explicit numeric support heights remain unchanged.

## 10. Power-on self-check and commissioning (implemented)

**Camera binding, on every start, read-only and before the bus is connected**: each camera is bound to its physical device by `unique_id` / `device_name` or reported for an index binding; the record is `<robot run_dir>/camera_binding.json` ([§3.4](#34-camera-binding-unique_id-device_name-and-index_or_path)).

**Self-check, on every start, read-only** (`ni_rua/robot/backends/so101/preflight.py`, run inside `LeRobotSO101Driver.connect` before any torque or goal write; record saved to `<robot run_dir>/preflight.json`). Checks run in this order: serial port present; broadcast ping finds exactly the 6 configured STS3215 ids; servo EEPROM calibration matches the LeRobot file; supply voltage, temperature and servo status bits; every raw position inside the calibration range (allowing `preflight.calibration_range_tolerance_rad`, default 0.05 rad of servo shaft, about 33 ticks: a relaxed arm resting on a mechanical end stop reads a few ticks past the hand-recorded LeRobot range, as the elbow did on 2026-11-16 at 2 ticks; the excess per joint is recorded as `beyond_calibration_ticks`, 0 restores the strict check) and every mapped joint inside the URDF limits. The first failure stops startup with a stable code: `serial_port_missing`, `servo_power_off`, `servo_missing`, `servo_model_mismatch`, `motor_calibration_mismatch`, `supply_voltage_out_of_range`, `servo_overtemperature`, `servo_error_status`, `joint_outside_calibration_range` or `joint_outside_urdf_limits`. There is no simulated fallback. Thresholds are in the deployment section `preflight`. After servo power cycles the Goal_Position registers read 0, so the connect step must write goal = present before enabling torque; that order is implemented and tested.

**Commissioning, once per arm or after recalibration** (`scripts/so101_commission.py`, real worker in-process, about 1.5 min). Steps: rest to ready pose; ±0.1 rad per joint, first uncorrected then with sag correction; shoulder_pan ±0.3 rad at the speed cap; jaw 1, then 0, then park; ±2 cm Cartesian moves in x, y and z through `execute`; park at ready. Output goes to `commissioning.json` plus camera images.

First run, 2026-11-08 (`~/.cache/ni-rua/so101/commission-20261108T061608Z`):
- All joint directions match the URDF with identity `joint_mapping`. This is verified by motion; camera calibration is still pending.
- Uncorrected settled error at P=16 is up to 0.042 rad (elbow, moving against gravity), 0.009–0.018 rad elsewhere. With correction, final error is ≤ 0.008 rad.
- Settle time is t90 ≈ 0.35–0.76 s after the ramp.
- Jaw measured 0.985 at goal 1, 0.018 at goal 0, and 0.489 at goal 0.5.
- Cartesian ±2 cm (position-only IK): residual 0.2–1.8 mm, needing 1–3 correction rounds. Lateral y moves change the TCP orientation by about 0.11 rad (5-DOF).
- Ready pose `[0, -0.95, 0.90, 1.00, 0]` rad puts the TCP at (0.216, 0.000, 0.084) m pointing down. It is now `home.q_rad`.
- The folded rest pose rests the elbow on its hard stop (raw 3100 of 3104), so a joint move into it stalls with `tracking_error`. Park at the ready pose instead.

Speed probe, 2026-11-08 (`scripts/so101_speed_probe.py`, shoulder_pan; raw results were written to `/tmp/so101_speed_probe.json`, not archived):

| commanded rad/s | peak measured rad/s | tracking lag rad | result |
| --- | --- | --- | --- |
| 0.3 / 0.6 / 1.0 | 0.36 / 0.70 / 1.14 | ≤ 0.04 | tracks |
| 1.5 | ~1.5 | 0.06–0.10 | tracks with growing lag |
| 2.5 | saturates at ~1.47 | — | fails arrival |

The earlier 0.3 rad/s was our deployment cap, not a driver limit. The local deployment now uses `max_joint_speed_rad_s=1.0` and `max_command_delta_rad=0.06`; the operator `speed_scale_cap` can only lower it. LeRobot's `max_relative_target_deg` additionally clamps each command.

Time budget: in a task-3 run, wall time was 1093 s, of which robot execution was 196 s. Model time dominates, not motion speed.

macOS `program_run`: `RLIMIT_AS` cannot be set on macOS. It is recorded as `unenforced_limits` in the isolation spec instead of aborting the launch (`ni_rua/runtime/programs/runner.py`).

## 11. Known gaps

- Camera calibration done 2026-11-08 (`scripts/so101_calib_auto.py`, 19/21 poses): wrist intrinsics 0.24 px, hand-eye 3.1 mm RMS / 0.014 rad. The fixed camera's intrinsics are copied from the wrist camera (same model, static board gives only one view; marked `copied_from_wrist_same_model_unverified`), and its chain reprojection is 4.0 px. Improve this with workflow S3 (§5), or the developer `capture-handheld` tool. This historical result does not validate the new S0–S6 workflow on hardware.
- The SO-101 backend has no `preview_motion`, so with a calibrated profile, set `features.motion_review=false`; otherwise Harness refuses to start (`motion_review_backend_unavailable`).
- No collision checking of the motion path, objects or obstacles before execution; see the MuJoCo twin candidate in §7. Without `ik_service` there is no collision checking at all; with it, planned joint targets are checked for self-collision and an optional table box only (§12.8). Obstacle, held-object and clearance fields are refused (§4.8).
- Trajectory feedback geometry and overlays exist only when the start-frame capture succeeds (§4.6); they are encoder FK of the TCP and CAD fingertips, not contact evidence.
- 5-DOF arm: TCP orientation cannot be set independently of position (for example, lateral y moves rotate the TCP by about 0.11 rad).
- Perception (§13): the mesh robot mask and the cross-view check were verified offline on archived real observations and with a fake service; the live service path on the real arm (SAM masks against the mesh mask, the wrist `/held` request) has not been run yet. The mask depends on the camera calibration, which the capsule path used to hide.
- `stop` (the console's Stop button) is not preemptive: it waits on the control lock behind a running calibration step or `go_home`. During such a command, use Emergency Stop (`estop`), `hold` or the wizard's cancel, which are answered at once.

### Viewing VGGT and SAM results

Open the original console’s **Model-visible observation → VGGT / SAM perception results** panel (or **Single observation evidence** in the trajectory workspace) and select the source camera (`fixed` for the usual SO-101 setup), then **Perception overlay** or **Side-by-side comparison**. The object table reports estimates with units and component failure states; the wrist image never receives the fixed camera’s overlay. Saved runs use the same read-only display. See [operator visualization](../obs3d.md#operator-visualization-and-saved-replay) for data semantics and limits. Local code changes take effect at the next normal gateway launch; do not restart a gateway during an active motion merely to refresh this UI.

### Workspace entry and navigation

For a live SO-101 gateway, `GET /` redirects to the original `/console` workbench. The header keeps three consistent entries: **Robot Workbench** (`/console`), **Calibration** (`/calibration`) and **Trajectory & Records** (`/index.html`). The console retains its telemetry cameras, joint/motor state, operator controls, Codex terminal and settings; perception appears within its saved-observation section. Calibration remains the existing S0–S6 wizard, including XY consistency checks. The explicit records URL does not redirect back to the console. Simulation and archive home pages keep their original review UI, and archive pages do not offer hardware navigation.

`web/workspace-navigation.js` also redirects a live SO-101 root page when an already-running gateway still has the old root route. This allows a frontend rebuild and browser refresh without restarting the active robot process. Navigation itself issues no physical command; the console's existing writer-lease behavior is unchanged. The two perception views share `web/perception-view.js`, not a nested workbench or a second control session.

`web/build.mjs` keeps the public `console.js` and `calibration.js` script names, but writes imported styles into `console-bundle.css` and `calibration-bundle.css`. Their handcrafted page styles retain `console.css` and `calibration.css`; these must never overwrite the bundled xterm/shared styles. Test the root, all three links, archived/simulation routing, desktop/mobile layouts and imported CSS with `tests/test_gateway_console.py`, `web/check-workspace.mjs` and the existing `web/check-calibration.mjs`.

## 12. IK planning, the cuRobo IK service and `mode=reach`

Code: `ni_rua/robot/backends/so101/ik_plan.py` (geometry, both solvers), `ik_service.py` (HTTP client), `deployment.py` (`ik_service` section), `worker.py` (`SO101Adapter._ik`, `reach_batch`), `contract.py` (model-facing IK sentence). The service is in `services/curobo_ik/` ([README](../../services/curobo_ik/README.md)). Tests: `tests/test_so101_ik_plan.py`, `tests/test_so101_ik_service.py`, `tests/test_curobo_ik_build_model.py`, `tests/test_robot_action_reach.py`, `tests/test_embodiment_profile.py`.

### 12.1 Why there are two solvers

The arm has five joints, so a move stage's `position_m` + `quaternion_xyzw` is solved as inverse kinematics (IK) with residuals. The packaged damped-least-squares solver (`so101_kinematics.ik`, §2 and §7) is local: from the measured joints plus deterministic restarts it can stop in a local minimum. Measured 2026-11-15: a top-down pose above the blue cube at 15 to 30 deg yaw failed with 48 restarts, while cuRobo solved it. In the first real-arm OpenCode run (stack three cubes, 30 min, 0 actions) the first top-down preview failed at a hover pose; preview stops at the first failed stage, so the reachable grasp pose behind it was never evaluated, and the model then rebuilt FK/IK in its sandbox. Two changes address this: a better solver (cuRobo, optional) and a query that answers many poses at once (`mode=reach`, §12.6).

| Solver | Selected when | Where it runs | Orientation tolerances |
| --- | --- | --- | --- |
| `local` (`ik_plan.local_solve`) | deployment has no `ik_service` (default) | worker process on the Mac | one angle, `orientation_tolerance_rad` (§2) |
| `curobo` (`ik_plan.curobo_solve`) | deployment has `ik_service` | NVIDIA GPU host, loopback HTTP behind the SSH tunnel | one angle, or `approach_tolerance_rad` and `roll_tolerance_rad` separately (§12.5) |

Exactly one solver plans per deployment. `robot_capabilities.unified_action.ik_solver` reports `"curobo"`, `"pink"` (the `ik_service.solver` in use) or `"local"`, and the model-facing IK sentence of the `robot_action` instructions (`contract._ik_sentence`) and `kinematic_reachability` are generated for the active solver. There is no automatic fallback from `curobo` to `local` (§12.7).

### 12.2 Deployment section `ik_service`

An optional object in `configs/so101.local.json`; absent or `null` means the local solver. Unknown keys and out-of-range values are rejected at load (`so101_deployment_invalid:ik_service...`). Field meanings are declared once in `deployment.IK_SERVICE_FIELDS`; `deployment.validate_ik_service` normalizes them.

| Field | Meaning | Unit | Default | Range |
| --- | --- | --- | --- | --- |
| `solver` | IK solver behind the service | n/a | required | `"curobo"` |
| `url` | Service base URL. Loopback only, normally the local end of the SSH forward of a `remote.services` entry (§9) | n/a | required | `http://127.0.0.1:<port>` or `http://localhost:<port>`, explicit port |
| `timeout_s` | Wall-clock limit of one HTTP request; a slower answer fails planning with `ik_service_unreachable` | s | 10 | (0, 60] |
| `num_solutions` | Best joint solutions the service returns per pose; the planner picks among them (§12.4) | count | 8 | 1 to 32, integer (the service accepts at most its `--num-seeds`, default 64) |
| `table_top_z_m` | Top face of a table box the service checks for collision, in `so101_base` | m | `null` (no table) | [-0.3, 0.3] |
| `path_check_step_rad` | Largest joint change between two samples of the preview's path collision check (§12.9); 0.01 rad is about 4 mm of fingertip travel at 0.4 m reach | rad | 0.01 | [0.002, 0.2] |

```json
"ik_service": {"solver": "curobo", "url": "http://127.0.0.1:8341", "timeout_s": 10, "num_solutions": 8, "table_top_z_m": -0.02,
               "path_check_step_rad": 0.01}
```

The planning joint limits (URDF limits shrunk by `motion.joint_limit_margin_rad`, rad, default 0.05) are sent with every request. The service refuses limits that differ from the ones its model was built with by more than 1e-6 rad (HTTP 409, client reason `ik_service_http_409:joint_limits_mismatch`), so the margin set for the service (`SO101_JOINT_LIMIT_MARGIN_RAD`, §12.3) must equal `motion.joint_limit_margin_rad`. The worker builds the client at startup but contacts the service lazily (first solve calls `GET /health`), so a worker starts while the service is down and fails only when a pose is planned.

The section also changes the embodiment profile: with `ik_service` the profile carries the `approach_roll`, `path_check`, `collision_world` and `obstacle_object_ids` tags in addition to `reach` and `arrival_correction` ([embodiment-profile.md](embodiment-profile.md#supports-tags)). `table_top_z_m` appears in the collision-scope text (§12.8) but is not otherwise reported to the model.

### 12.3 The service and how the gateway starts it

`services/curobo_ik/` holds a standalone cuRobo v2 service for a GPU host (it needs CUDA; on the lab host it runs in the RoboDojo interpreter):

| File | Role |
| --- | --- |
| `build_model.py` | Builds `model/` from the packaged SO-101 geometry (`geometry_model.zip`, `physical.json`): the URDF with the five arm joint limits shrunk by `--margin` rad, MorphIt collision spheres (55 to 56 spheres on 9 links; the fit is stochastic and seeded with `--seed`, default 42; the base link is clipped at z = 0), the self-collision ignore matrix, the jaw joint locked at 0 rad, and `model.json`. The thin camera-mount fit overlaps `wrist_link` in poses the real arm reaches, so that pair is ignored. |
| `curobo_ik_service.py` | Loopback HTTP server: `GET /health`, `POST /ik` and `POST /path_check` (request fields and units are in the module docstring). One IK solver with `--num-seeds` (default 64) parallel seeds and `--max-batch` (default 32) poses per request, and one collision checker for joint-space paths (§12.9); requests are serialised. At startup it solves the commissioned ready pose as a self-check and refuses to serve (exit 2, `SELF_CHECK_FAILED`) if that pose is not solvable or its solution collides in the path checker with the jaw at 0 rad or at `--open-jaw-rad` (the deployment's `gripper.jaw_open_rad`; `start.sh` passes `SO101_JAW_OPEN_RAD`, default 1.74533; `self_check_failure`). |
| `start.sh`, `stop.sh` | Detached start (builds the model first when `model/model.json` is missing or built with another margin, then waits for `/health`, default 300 s) and stop of its own PID. Defaults: port 8341, `SO101_JOINT_LIMIT_MARGIN_RAD` 0.05, `CUROBO_IK_SEEDS` 64, `CUROBO_IK_MAX_BATCH` 32. |
| `deploy.sh` | `bash services/curobo_ik/deploy.sh <ssh host> <remote dir>` copies the service and the packaged geometry to the host. |

The gateway starts it through the generic `remote` section of the NI-RUA config (§9), not through the SO-101 deployment file. Add a second service next to `vggt`:

```json
"remote": {
  "enabled": true, "host": "<gpu-host>",
  "services": [
    {"name": "curobo_ik", "remote_port": 8341, "workdir": "~/ni_rua_services/curobo_ik",
     "env": ["CUROBO_PYTHON=~/miniforge3/envs/RoboDojo/bin/python", "SO101_JOINT_LIMIT_MARGIN_RAD=0.05"],
     "command": "bash start.sh", "stop": "bash stop.sh", "health_path": "/health", "gpu": "auto"}
  ]
}
```

The remote manager applies the rules of §9 unchanged: it reuses a healthy service (`reuse_if_healthy`), otherwise picks a GPU with `gpu_min_free_mib` free (default 15360 MiB, the VGGT figure; the cuRobo service's own footprint has not been measured here, so lower it only against a measurement), exports `CUDA_VISIBLE_DEVICES`, starts `command` detached and waits up to `startup_timeout_s` (300 s) for `health_path`. It then holds the SSH forward `local_port` (default = `remote_port`) to the server's loopback. `ik_service.url` must be that local end; unlike `features.vggt_endpoint`, no check ties `ik_service.url` to a `remote.services` entry, so a mismatch surfaces only as `ik_service_unreachable` when planning. Measured 2026-11-15 on the RTX PRO 6000 Blackwell host: model build about 25 s (once per margin), service ready about 6 s after start, a batch of 4 to 6 poses with 64 seeds solved in 3 to 45 ms on the GPU, 0.4 to 1.2 s round trip from the Mac through the tunnel including the position-first retry.

Build quirks handled in the service: this cuRobo build pads a smaller batch but not the jerk buffer, so the service always sends a full `max_batch` request (the first pose repeated) and returns only the real ones.

### 12.4 Candidate selection (`ik_plan.curobo_solve`)

cuRobo only proposes joint solutions. For every pose the planner:

1. Sends the pose, the seed (the measured joints, or for chained preview stages the previous stage's planned joints), goal-frame axis weights (`ik_plan.axis_weights`) and fixed convergence bounds `SOLVER_CONVERGENCE` = 0.0005 m and 0.005 rad. The bounds are independent of the stage tolerances so a loose tolerance never stops cuRobo with millimetres left; the tolerances only decide acceptance here. Poses with equal weights go in one batch.
2. Checks each returned candidate: five finite joints, inside the planning limits (1e-6 rad slack), cuRobo `feasible` (joint limits, self-collision, table box), and position and orientation residuals within the stage tolerances, where the residuals are computed with the packaged forward kinematics, never the service's numbers.
3. Among valid candidates, prefers the precise tier (position error at most 0.0005 m by that FK), then takes the one with the smallest largest-joint travel from the seed. Planned grasp poses came back within 0.0 to 0.6 mm and under 0.001 rad.
4. If none is valid, or the best is valid but not precise, and the orientation is constrained, sends one position-first request. It lowers only the two tilt weights (rotations about the goal x and y axes) by `POSITION_PRIORITY_TILT_SCALE` = 0.1 and keeps the roll weight: on this arm the roll comes from `wrist_roll`, while the approach tilt is coupled to the position. The result is reported as `ik.solve = "position_priority"`. Example (2026-11-15): a hover 0.27 m out at z 0.09 m needs 8 to 15 deg of tilt; 0.40 m straight down is unreachable.
5. Free orientation (tolerance >= 3.14159, §2) sets the rotation weights to 0 and needs no retry.

Failure record (`ik.ok = false`): the residuals of the closest candidate (`position_error_m` in m, `orientation_error_rad` in rad, plus `approach_error_rad` and `roll_error_rad` when split bounds are used), `orientation_checked`, and two hints. `position_priority` holds the `quaternion_xyzw` the position-first request reached at this position (among cuRobo-feasible candidates within the position tolerance) with its residuals; the next request can reuse it. `in_collision_candidates` (count) is the number of first-request candidates that meet the tolerances by FK but cuRobo marked infeasible; a value above 0 means reachable solutions exist but collide with the arm itself, the table box, or a joint limit.

### 12.5 Orientation tolerances and the split bounds

Without `ik_service`, orientation is one angle `orientation_tolerance_rad` (rad, (0, pi]; >= 3.14159 = free). With `ik_service` the move stage and the batch also accept (profile tag `approach_roll`; the fields are absent from the schema and refused as `unsupported_capability` otherwise):

| Field | Meaning | Unit | Default | Range |
| --- | --- | --- | --- | --- |
| `approach_tolerance_rad` | Largest angle between the achieved and the requested approach axis (TCP local +Z, outward along the jaws) | rad | unset | (0, pi]; >= 3.14159 = free |
| `roll_tolerance_rad` | Largest rotation about the approach axis left after aligning the achieved axis with the requested one (the jaw-direction error) | rad | unset | (0, pi]; >= 3.14159 = free |

When neither is set the orientation is judged as one angle. When either is set, orientation is judged per component, a component left unset uses the stage's resolved `orientation_tolerance_rad`, and the same split applies to the arrival check after the move (`ik_plan.orientation_within`, `arrival_verdict(..., split=...)`). Receipts then add `approach_error_rad` and `roll_error_rad` (rad) to the IK and arrival residuals and `orientation_check: "approach_and_roll"` to `effective_tolerances`; `orientation_error_rad` (the single total angle) is still reported. Batch-level values are inherited by stages that set none, and `tolerance_preset` sets neither bound. Example: a straight-down quaternion with `approach_tolerance_rad` 0.35 (about 20 deg of tilt) and `roll_tolerance_rad` 0.1 accepts a tilted approach while keeping the jaws aligned. The fields are also covered by stage overrides and repair (`plan_continuation.py`, `batch_repair.py`).

### 12.6 `robot_action mode=reach`

Profile tag `reach` (always on SO-101, with either solver; not on ARX X5). A stage may give the grasp-centre fields instead of `position_m` ([§4.9](#49-grasp-centre-input-grasp_centre_m)); its verdict then carries `grasp_centre`. The mode solves IK for 1 to 16 move stages (`min(max_stages, stages_per_arm * arms)`) independently, each from the joints measured at the call (not chained through the previous stage), answers every stage (no stop at the first unreachable one), and plans, caches and moves nothing. It needs `observation_id` and an idle active run. Accepted fields: `mode`, `observation_id`, `stages`, `timeout_s` (s; bounds the whole query) and the tolerance fields; each stage only `id`, `kind="move"`, `arm`, `position_m`, `quaternion_xyzw` and tolerance fields. Anything else is refused by name (`robot_action_reach_has_unsupported_fields`, `robot_action_reach_accepts_move_poses_only`).

Result (`output_contracts.REACH`): `status` `"solved"`, `observation_id`, `stage_count`, `reachable_stage_count`, `physics_advanced` (false), `executed_native_steps` (0) and `stage_results[<stage id>]` with `ok`, `ik` (the §12.4 record), `tolerances` (the effective acceptance bounds; arrival window and guard facts do not apply), `reached` (`position_m` in m, `quaternion_xyzw`, `approach_axis` the TCP +Z unit vector, all in `so101_base`, of the planned or closest joints by FK) and, when `ok`, `q_rad` and `joint_travel_rad` (|target - measured| per arm joint, rad). A failure of the whole query (service unreachable, unreadable joints) is a tool error carrying the reason. Use it to choose grasp, hover and place poses before a preview. The tool registry classifies the mode as a read (`runtime/tools/builtins.py`).

### 12.7 When the service is unreachable

A connection failure, tunnel loss or timeout raises `IKServiceError("ik_service_unreachable")`; other reasons are `ik_service_http_<code>:<service reason>` and `ik_service_bad_response`. Nothing falls back to the local solver, because two solvers with different feasible sets would make a plan depend on the network. The failure surfaces as follows:

- `mode=reach`: the query fails and the tool error code is the reason (for `ik_service_unreachable` the suggestion tells the model that its pose is not at fault, to retry at most once and then tell the operator).
- `preview` and `direct`: the move stage fails as `planning_failed` with `ik` = `{ok: false, solver: "curobo", reason: ...}` in the failure feedback and no residuals.

No motion has started in either case. The operator restarts the service or the tunnel; the gateway's `GET /v1/remote/status` shows the service health (§9).

### 12.8 Collision scope and its limits

With the local solver the capability text says collision scope is none: joint limits and the deployment workspace box only, and the motion text says that nothing checks the path. With `ik_service` it becomes (`SO101Adapter._collision_scope_text`):

- **IK targets:** each planned joint target is free of self-collision, and of a table box with its top at `table_top_z_m` when configured, in cuRobo's sphere model of the arm (the solver's own constraints, §12.4).
- **Paths:** preview samples the joint-space path of every move against self-collision, the table box and the request's obstacle and held-object boxes and reports the first collision per move (§12.9, §12.10).
- **Advisory only:** a colliding plan is still `planned` and executes when the model executes it. Nothing is re-routed, and execution checks nothing; the grip, stall and tracking guards of §4 are unchanged.

Limits of the model: 57 spheres fitted to the CAD meshes, so thin parts are under-covered. Measured 2026-11-15 on the dev service: with the gripper pointing down and the CAD fingertips at z 0.040 m, a table top at 0.045 m did not touch a `gripper_link` sphere and one at 0.050 m did (by 3.9 mm), i.e. the lowest gripper sphere ends about 6 mm above the CAD fingertips. `stage_plans[<move>].path.min_fingertip_z_m` (§12.9) is the CAD fingertip height and does not depend on the spheres. Objects that are not passed as `obstacles` / `held_object` are not in the world.

### 12.9 The path between targets: FK facts and the collision check

**What a move does.** Between two planned targets the arm runs on a straight line in joint space (`ik_plan.joint_interpolation`: `steps = ceil(largest joint change / (max_joint_speed_rad_s x speed_scale x cap) x command_rate_hz)` equal joint steps). The TCP therefore does not travel the straight Cartesian line between the two poses. Probe (audit 2026-11-15): 14 mm TCP dip on a 15 cm radial move at z 0.05 m and up to 24 mm off the line; with the cuRobo solutions of the live check below, the same move bulged 21 mm off the line without dipping. The preview used to store one sample per move stage, so the overlay joined the stage targets with straight lines the arm does not follow.

**Path facts (every SO-101 deployment).** `preview_batch` now evaluates forward kinematics over exactly the joint goals the move will send, including the subdivision of a during-motion jaw ramp, and reports per move stage `stage_plans[<move>].path` (`ik_plan.path_facts`, m in `so101_base`):

| Field | Meaning |
| --- | --- |
| `samples` | Evaluated configurations: the start plus every command of the move |
| `min_tcp_z_m` | Lowest TCP z on the path |
| `min_fingertip_z_m` | Lowest z of the two CAD fingertip points: the fixed jaw's inner tip at its furthest protrusion beyond the TCP (6.3 mm along TCP +Z) and the moving jaw's inner tip at the jaw angle of each command (`so101_gripper_geometry`; CAD, not measured on the unit) |
| `lowest_at_fraction` | Where that lowest point lies, 0 = start, 1 = target |
| `max_line_deviation_m` | Largest distance of a TCP sample from the straight segment between the start and target TCP positions |
| `dip_below_endpoints_m` | How far `min_tcp_z_m` lies below the lower of the two endpoints (0 when it does not) |

The candidate trajectory keeps up to `PATH_OVERLAY_SAMPLES` = 8 evenly spaced commands of each move (its last one included), each with all link positions, so the preview overlay polyline and `mode=inspect` follow the interpolated path; the count is bounded because the artifact stores every link of every sample.

**Preview and execution use the same rule, not the same start.** `_segment` calls the same two functions from the joints it measures when the move starts; preview chains the planned targets from the joints measured at preview. Each executed move reports the difference as `stage_results[<move>].path_start_deviation_rad` (largest joint difference between the previewed and the measured start, rad). It is bounded by `plan.max_start_drift_rad` for the first stage and by the arrival tolerance of the previous stage afterwards.

**Path collision check (`ik_service` only; tag `path_check`).** After planning, `SO101Adapter._path_check` sends the segments of every move and of every standalone gripper stage (`start_q_rad`, `end_q_rad`, jaw angle at both ends, whether the held object is attached) to `POST /path_check`. A standalone gripper stage is one segment: the arm stays at the joints it has reached and the jaw sweeps from its angle before the stage to the commanded goal; a held box stays attached during the sweep that opens the jaw. A batch of gripper stages only is therefore checked as well. The service samples each segment at `1 + ceil(largest joint change / path_check_step_rad)` evenly spaced configurations, both ends included, interpolating the arm and the jaw linearly, and evaluates all of them in one batch with cuRobo's `RobotCollisionChecker` (robot spheres against the cuboid world, and the self-collision pairs). The jaw angle is the measured one until the plan commands the jaw, then the commanded goal. A move without a during-motion jaw is one segment. A move with `gripper_motion` `{start_fraction, end_fraction}` is cut at those fractions into up to three segments (`ik_plan.path_check_segments`): the jaw holds its start angle before `start_fraction`, ramps over the middle segment and holds the goal after `end_fraction`, exactly the schedule `_jaw_schedule` streams (arm and jaw are both linear in the move's command progress), so the checked jaw angle at every arm configuration is the commanded one; before this the jaw was interpolated over the whole move. The move's result merges its segments: `fraction` is mapped back to the whole move and `samples` / `colliding_samples` are summed (a cut point is sampled by both segments). Result:

- `path_check`: `status` `clear` (no sample collides), `collisions` or `unavailable` (with `reason`, e.g. `ik_service_unreachable`, `ik_service_path_check_unavailable` for a service deployed before this endpoint), `colliding_stage_ids`, `max_joint_step_rad`, `obstacle_ids` and, with a held box, `held_object` (`spheres`, `sphere_radius_m`, `attached_through_stage_id`).
- `stage_plans[<stage>].path.collision` on a colliding move or gripper stage: `fraction` (0 = start, 1 = target), `tcp_position_m` (m) and `jaw_rad` (jaw angle, rad) of the first colliding sample, `colliding_samples` / `samples`, `world` `[{link, penetration_m, obstacles}]` and `self` `[{links, penetration_m}]`, each at most six entries, deepest first. `obstacles` names the request's box ids or `"table"`; link `attached_object` is the held box. The verdict and `penetration_m` of `world` are cuRobo's; the obstacle and link-pair names come from the same spheres checked against the boxes and each other in the service.
- `mode=execute_if_clean` executes only when `status` is `clear`; otherwise the plan is returned with `downgrade_reasons` `path_check_collisions` or `path_check_unavailable`, and the model may still execute it by `plan_id`.
- No timing and no joint vector reach the model; the service answer carries `check_ms`.

Tests: `tests/test_so101_path_check.py` (`InterpolationTests`, `PreviewPathTests`, `ClientTests`, `ServiceHelperTests`, `JawRampPathCheckTests`, `ServiceSelfCheckTests`, `test_execute_if_clean_runs_a_clear_plan_and_keeps_a_colliding_one_for_inspection`).

### 12.10 Collision world: `obstacles`, `obstacle_object_ids` and `held_object`

With `ik_service` (tags `collision_world`, `obstacle_object_ids`) `robot_action` preview / direct / execute_if_clean accept three batch fields of the shared action language. They compile into the batch's `review_proxies` (`proxy_contract.validate_proxies`: finite boxes inside +-3 m with `max_m > min_m`, `margin_m` in [0, 0.2] m default 0.01, `evidence` text, unique ids), which the SO-101 contract accepts with `obstacles` and this arm's held box only; contact declarations stay refused.

| Field | Meaning |
| --- | --- |
| `obstacles` | Up to 40 axis-aligned boxes `{id, min_m, max_m, evidence, margin_m?}` in `so101_base` (m) estimated by the model |
| `obstacle_object_ids` | `object_table` row ids of the current observation; each becomes a box (`scene_obstacles.obstacles_from_object_table`: yaw-aware footprint from bottom to top, `margin_m` 0.005, size rules below). Held rows, unknown ids and rows without a centre add no box, and the preview says so |
| `held_object` | The box `{min_m, max_m, evidence, margin_m?}` of the object the jaw holds now, in `so101_base` |

- **Stateless:** the boxes apply to that request only. The per-episode obstacle registry (`obstacle_policy`, `ignore_obstacle_ids`) belongs to the ARX-X5 proxy review and is not offered; a continuation (`source_plan_id`) inherits `obstacle_object_ids` but not explicit boxes.
- **Obstacles** are grown by their `margin_m` on every face and added to the cuboid world next to the table box (`ik_plan.collision_world`).
- **Object-table rows, transparently.** The conversion applies size rules (`scene_obstacles.ROW_*`, m): a length or width below 0.015 is raised to 0.015 and a missing one is 0.03; each half extent of the yawed footprint along x and y is at least 0.008; the box is at least 0.010 tall (its bottom is lowered); a missing top is centre z + 0.02, a missing bottom the table height (a `table_z` of exactly 0 m counts as a height), else top - 0.010. The SO-101 field help of `obstacle_object_ids` publishes these numbers (generated from the constants, gated on `collision_world`; the ARX-X5 text is unchanged), and the preview adds `path_check.object_row_boxes` when the conversion did anything besides copying a row: `skipped` `[{id, reason}]` (`not_in_table`, `held_row`, `no_centre`, `no_object_table`) and `size_floors` `[{id, field, row_value, used_value}]` (m; `row_value` null when the row had none; fields `length_m`, `width_m`, `half_extent_x_m`, `half_extent_y_m`, `top_z_m`, `bottom_z_m`). The key is absent when nothing was skipped or resized. The boxes themselves are unchanged (`unified_actions.compile_request(..., obstacle_report=)` collects the report). The obstacle `id` help on SO-101 says the box id is what `path_check` names (there are no contact declarations here).
- **Held object:** the box, grown by its margin, is filled with up to 32 equal spheres that stay inside it (`ik_plan.box_spheres`: a grid as fine as three cells along the shortest side; a 3 cm cube gives 27 spheres of 5 mm radius, so edges and corners are under-covered by up to 0.73 x radius). The spheres are fixed to the TCP pose measured at preview and travel with the gripper (the checker's `attached_object` link). They are checked against the world and the arm links, not against the gripper links. The object counts as attached up to and including the first stage whose jaw command opens the jaw (a goal above the goal commanded before it); later stages are planned and checked without it, and the released object is not added as an obstacle. `path_check.held_object.attached_through_stage_id` names that stage (null when none opens the jaw).
- **IK targets:** the service labels every candidate `obstacle_free` (the same checker at the candidate joints with the stage's jaw angle). `curobo_solve` prefers valid candidates that are not labelled colliding and still plans when every valid candidate collides; `stage_plans[<move>].ik.obstacle_free` reports the chosen one. The solver itself keeps planning against self-collision and the table only, so obstacles never make a pose "unreachable".
- **Use:** leave out the object a stage is about to grasp or push, or read its name in the report of that stage. All of this is advice about boxes the model supplied, never proof of clearance.

Tests: `tests/test_so101_path_check.py` (`CollisionWorldTests`, `ObstacleAwareSolveTests`, `CollisionWorldPreviewTests`, `RobotActionTests` incl. `test_object_row_size_rules_are_reported_and_published`, `SchemaGatingTests`, `CapabilityTextTests`).

### 12.11 Live check against the GPU service (2026-11-15)

Dev copy of the service on the lab host (RTX PRO 6000 Blackwell, cuRobo `0.0.post1.dev1`, 57 spheres + 32 attached slots), reached from the Mac through the SSH tunnel; planning with the example deployment, `table_top_z_m` -0.02, step 0.01 rad. No arm moved.

| Case | Result |
| --- | --- |
| Free radial move, TCP (0.15, 0, 0.05) to (0.30, 0, 0.05) | 184 to 227 samples, 0 colliding; GPU 1.8 to 2.7 ms, round trip 290 to 630 ms |
| Same move through a box x 0.21..0.24, y +-0.02, z up to 0.12 | first collision at fraction 0.29 (`gripper_link` vs `o1`), 84 of 184 samples colliding |
| Arm folded onto the base (shoulder_lift -1.6, elbow 1.6, wrist 1.6) | first collision at fraction 0.39, `gripper_link` vs `shoulder_link` (self) |
| Table box raised to 0.05 m under the same move | every sample colliding from fraction 0 (`shoulder_link`, `base_link`, `gripper_link` vs `table`) |
| Held 4 cm cube carried at TCP z 0.05 over a 4 cm slab, then released and retreat | `carry` collides at fraction 0.64 (`attached_object` vs `shelf`), `retreat` clear; without `held_object` both clear |
| 16 segments, 2944 to 3632 samples, one obstacle | GPU 2.7 to 12 ms, round trip 300 to 480 ms |
| Worker preview of three moves (three IK calls + one path check) | 1.7 to 2.7 s through the tunnel |
| 75 mixed `/ik` and `/path_check` requests from three threads, up to 7600 samples and 40 obstacles | 0 errors; a 64-segment check with 52 colliding segments took 54 to 62 ms on the GPU |

One build quirk was found and is handled in the service: the IK solver's first solve captures a CUDA graph, and a warp collision kernel first loaded after that capture corrupts the graph (the next solve fails with `CUDA error: misaligned address`). The service therefore runs every kind of checker query once before the first IK solve.

## 13. Perception: robot mask, held row and the cross-view check

The SO-101 profile lists `held_perception` (always) and `cross_view` (with both the `fixed` and the `wrist` camera) in `supports` (`deployment.embodiment_profile`). Both act only with `features.external_perception` / `features.object_table` and a reachable VGGT service (§9). The robot mask and the held row additionally need `features.held_perception: true`. `configs/so101_gateway.json` still sets it to `false`, so the capsule path stays in use until the operator switches it on; do that after the fixed camera is recalibrated and one live object table has been checked (see the last paragraph and §11).

**Robot pixels.** Object tables no longer drop everything within 6 cm of the joint chain. The harness renders the packaged SO-101 meshes at the observation's measured joints (jaw included) into the requested camera, dilates the silhouette by 5 px and sends it as the robot mask without capsule segments; the service drops a mask only when at least half of its pixels are robot pixels. A cube beside or between the jaws therefore keeps its row, in the overview and in the wrist view. Details, the SO-101 / ARX-X5 differences and the numbers measured on real observations: [held_perception](../held_perception.md#embodiments-arx-x5-and-so-101).

**Held row.** When the jaws are commanded closed (`goal_normalized < 0.5`) and the measured fingertip opening stays more than the `non_empty` grip-guard margin above the commanded opening, the table gets a `held_right` row (`quality: held_fk_depth`) at the CAD fingertip midpoint of the measured jaw angle, with the extents of the pre-grasp table row when one was seen within 6 cm. It is a geometric estimate conditioned on a rigid grasp (`retention_status: unverified`); no centre-error statistics are quoted for this robot. It needs a measured `gripper.empty_close_aperture_m` in the deployment.

**Cross-view check.** `geometry_query` `{"kind": "object_table", "observation_id": ..., "target_camera_ids": ["wrist"]}` (or `camera_id: "wrist"` with `target_camera_ids: ["fixed"]`) also returns `cross_view`: each row paired with the same object in the other camera's table, that camera's monocular centre, the two-view point from the two mask-centre rays, the ray gap and the parallax. Rows are not changed. Fields, pairing rule and the real-run numbers: [object_table](../object_table.md#cross-view-check).

**What the operator should read from it.** A `ray_gap_m` well beyond what the objects' size explains on every object (with exact calibrations the gap reaches about 0.25 x the object's longest side at most, a few millimetres for a 30 mm cube), or arm parts listed as table objects in the overview table, means the fixed camera no longer matches its calibration. In the 2026-11-15 run the fixed-camera rows were 25 to 29 mm from the wrist view and the rays missed by 23 to 27 mm; aligning the arm silhouette offline gave a fixed-camera rotation of about 5.7 deg since calibration. Recalibrate (§5) rather than compensate in the task.

Not changed by this section: the `scene3d` point cloud still marks points within 6 cm of the joint chain as `robot` (a flag, the points stay in the artifact), and `pixel_plane` with `plane_source: "vggt"` is unaffected.

### 13.1 Arm occlusion report (`robot_occlusion`)

After a place the arm often parks over the stack and the fixed-camera table comes back empty or short. Every model-facing object table (`geometry_query` kind `object_table` and the table attached to each observation) now says so: `value.robot_occlusion` (attached tables: `object_table.robot_occlusion`), computed by `ni_rua/perception/external/robot_view.py` from the same FK silhouette as the robot mask (the packaged meshes at the measured joints, the camera's calibration; rendered here when `features.held_perception` is off).

```json
"robot_occlusion": {"table_covered": 0.32,
  "hidden": [{"id": "o1", "seen_in": "obs-...", "centre_m": [0.328, 0.009, 0.001], "top_z": 0.017, "covered": 1.0}]}
```

| Field | Unit | Meaning |
|---|---|---|
| `table_covered` | fraction 0..1 | Share of the table ROI under the arm silhouette. ROI: pixels whose ray meets the table plane (`support_z_m` of the table) in front of the camera within the arm's horizontal TCP reach of the shoulder_pan axis (`robot_view.reach_facts`: 0.439 m on the packaged URDF), minus the static `base_link` silhouette. `null` when the camera sees no table inside the reach |
| `hidden` | list, only when non-empty | Rows of the newest earlier table of the same camera attached to an observation reply (before this observation; obs3d keeps the last 8, on-demand tables are not kept), plus the entries that table itself listed as hidden unless one of its rows lists the same object (the entry's centre inside the row's footprint), whose estimated box (centre, length x width at yaw, bottom to top) now projects at least partly under the silhouette; most covered first. Held rows are never listed |
| `hidden[].id`, `seen_in` | | The row id in the table that listed it and that table's observation id |
| `hidden[].centre_m`, `top_z` | m | That table's estimate (not re-measured) |
| `hidden[].covered` | fraction 0..1 | Share of the box's projected outline under the silhouette (listed when it rounds above 0.00) |

Without meshes or FK the field is `{"unavailable": "robot_meshes_or_fk_unavailable"}`, after a failed render `{"unavailable": "render_failed"}` (the detail goes to the operator diagnostics only). No row is added, removed or changed, and nothing moves. `object_state_delta` still lists a hidden object as vacated; `hidden` says why. Recomputed offline on the 21 attached tables of the 2026-11-16 run (joints, calibration and rows from the archive; six of them in tests/fixtures/so101_occlusion_real.json, `OcclusionGeometry.test_real_run_empty_tables_are_the_arm_covering_the_cubes`): the three empty tables came with `table_covered` 0.32 to 0.34 and the previous table's cubes under the gripper covered 0.37 to 1.0; with the arm hovering about 18 cm above the table `table_covered` was 0.06 to 0.15. Switch: deployment `perception.robot_occlusion_report` (default true; profile tag `robot_occlusion`, absent on the ARX-X5).

Limits: the report is only as good as the fixed-camera calibration (a rotated camera shifts the silhouette, see the end of §13); a box from an earlier, wrong estimate gives a wrong `covered`; an object that moved since the earlier table is still listed at its old place.

### 13.2 Clear fixed-camera pose (`perception.clear_view_q_rad`)

`scripts/so101_clear_view_pose.py --deployment <file>` (offline, reads the deployment and its calibration) searches joint configurations within the planner's limits (URDF minus `motion.joint_limit_margin_rad`), with the TCP inside `workspace` and no moving link mesh below the table plane, ranks them by `table_covered` of the fixed camera and prints the Pareto front of (coverage, largest joint travel from `home.q_rad`) plus the local IK of the chosen TCP pose from home (robot_action plans the TCP pose and may reach it with other joints). On the active calibration of 2026-11-15 it found, among others:

| coverage | travel rad | q_rad | TCP m |
|---|---|---|---|
| 0.0000 | 1.247 | [1.2466, -1.6953, 0.0, 0.0, 0.0] (arm upright) | [0.001, 0.112, 0.423] |
| 0.0047 | 0.623 | [0.6233, -1.2715, 0.41, 0.536, 0.0] | [0.176, -0.098, 0.326] |
| 0.0085 | 0.490 | [0.3116, -1.2715, 0.41, 0.536, 0.0] | [0.199, -0.052, 0.326] |
| 0.1666 | | home [0.0, -0.95, 0.9, 1.0, 0.0] | [0.216, 0.0, 0.084] |

The operator copies one line into the deployment, `"perception": {"clear_view_q_rad": [...]}` (validated against the same limits and with its TCP inside `workspace`), and restarts the gateway. The motion instructions then carry one sentence naming its TCP target (`contract.clear_view_sentence`: position_m and quaternion_xyzw of the TCP; "an ordinary move target ... nothing moves there on its own"); the joints stay in the deployment, since the model commands TCP poses. The model decides when to use it through `robot_action` like any move; its path is checked by preview like any other (path check with `ik_service`). Default `null`: no sentence. Re-run the search after a fixed-camera recalibration.

### 13.3 Fixed-camera resolution (deployment only)

The cameras deliver up to 1920x1080 at 30 Hz; the deployment captures 640x480. A different mode is a deployment change only (`cameras.fixed.width` / `height`) plus intrinsics for that mode: the worker refuses a calibration whose `image_size` differs from the configured size (`so101_camera_calibration_mismatch`), and `GeometryTools.camera` refuses a frame whose size differs from its calibration. A 16:9 mode is a crop and/or another scaling of the sensor, so K cannot be scaled by the size ratio. `scripts/so101_remap_calibration.py` measures the map instead: two images of one static ChArUco board, one per mode (`scripts/so101_perception_bench.py snapshot --size WxH`), corners matched by id, `new = scale x old + offset` per axis (`so101_calibration.fit_pixel_mapping`, RMS reported and gated by `calibration.max_intrinsics_rms_px`), K rewritten (`remap_camera_intrinsics`; distortion and extrinsic unchanged, residuals carried over, `intrinsics_source` naming the derivation) into a new file. A fit above the gate means the modes are not one affine map of the same pixels: recalibrate the camera in the new mode (/calibration S3, S4).

What changes downstream, measured on synthetic 1280x720 and 1920x1080 observations (`Resolution` tests) and from the code:

| Component | Behaviour at 1280x720 / 1920x1080 |
|---|---|
| Robot mask, held row, occlusion report | Rendered at the frame size with its K; the silhouette dilation scales with the width (`SO101Body.dilate_for`: 5 px at 640, 10 at 1280, 15 at 1920), keeping the angular margin it was chosen with |
| VGGT service input | The full frame (rectified when distorted). The service runs VGGT natively when both sides are multiples of 16 (640x480, 1280x720: about 3x the tokens of 640x480, so more GPU time and memory) and otherwise its balanced 512 px preprocessing with depth resized back (1920x1080: no more depth detail than a 512 px input, only finer SAM masks). Pixel thresholds of the service (SAM grid stride 8 px, minimum mask 40 px, held-mask search 40 px) are absolute, so they mean smaller objects at higher resolution |
| Object table rows, cross view, `contact_top_z` | Pixel coordinates in the frame; metric results unchanged in meaning |
| Overlay image | Frame size; ids and outlines drawn `width / 640` times larger above 640 px (`object_table.overlay_scale`), unchanged up to 640 px |
| Model-facing images | The existing encoding preserves resolution (not changed): each full observation delivers the fixed frame at the new size, and robot_observe adds the overlay at that size. Estimated input tokens per image (`token_estimates` in the bench): 640x480 425 / 410, 1280x720 1105 / 1229, 1920x1080 1105 / 1844 (512-tile rule / pixels-750 rule); the provider's usage report is the measurement. Images above 640x480 bypass the small encoded-image cache, so each is re-encoded |

Trade-off: finer masks and outlines (helps `contact_top_z`, which works on outlines) against 2.6 to 4.5x the image tokens per fixed frame and more service latency. The wrist camera can stay at 640x480. Not the default; measure with the bench (§13.5) before switching.

### 13.4 Top height from two views (`contact_top_z`) and the deployment `perception` section

VGGT monocular tops of 29 mm cubes were off by up to a third of a cube on the real arm (2026-11-16: the white cube's top read z 0.002 to 0.006 m against about 0.013 m, the table at -0.016 m; others read up to 10 mm high), and the model spent its turns fitting silhouettes in `program_run`. The cross-view check now carries two more estimates per paired row, beside (never instead of) the row:

* `target_view_top_z` (m): the other camera's own monocular top of the same object (the wrist view is much closer).
* `contact_top_z`, `contact_top_z_range` ([low, high] m), `contact_fit_iou` (0..1), or `contact_reason`: the two-view fit of a vertical prism standing on the table to both views' outlines (`ni_rua/perception/external/contact_height.py`; for a row on other rows the stack down to the table is fitted). No monocular depth and no footprint guess; the range is the fit with the outlines 1 px larger and smaller, calibration error not included (`ray_gap_m` of the same row shows it). Synthetic check (exact box outlines 20-80 x 20-50 x 15-75 mm, the real fixed calibration, a wrist-like camera 0.15-0.25 m away, 1.5 px pose noise on the second view, 16 cases): top error p50 0.6 mm, p90 2.8 mm; the 1 px range contained the truth in 13 of 16. About 0.1 s per row. Reasons: `not_on_table`, `mask_at_image_border:<view>`, `outline_missing:<view>`, `views_do_not_overlap_on_table`, `no_consistent_footprint`, `fit_failed`; `contact_top_z_range` is null when a 1 px change of the outline cannot be fitted; a stack row whose lower rows the wrist table did not list carries `contact_unpaired_support` (their ids).

Deployment section `perception` (optional; `deployment.PERCEPTION_FIELDS`, defaults `PERCEPTION_DEFAULTS`; a deployment without it keeps working):

| Key | Type, default | Effect |
|---|---|---|
| `robot_occlusion_report` | bool, true | §13.1 (profile tag `robot_occlusion`) |
| `contact_top_estimate` | bool, true | `contact_top_z` fields in cross_view rows (tag `contact_top`; needs both cameras) |
| `attach_cross_view` | bool, false | The table attached to every observation also carries `cross_view` against the wrist camera (tag `cross_view_attach`): one more `/object_table` call per observation (about 0.5 s local, 2-3 s through the tunnel). false keeps cross_view on demand (`target_camera_ids`) |
| `clear_view_q_rad` | null or 5 rad, null | §13.2 |

The ARX-X5 profile has none of these tags; its outputs are unchanged (`tests/test_embodiment_x5_golden.py`).

### 13.5 Perception bench

`scripts/so101_perception_bench.py` (snapshot, capture, perceive, score) measures detection, xy and top errors of each estimator, range calibration, latency, image and text size and the occlusion report against cubes placed at known positions. Protocol and exact commands: [so101-perception-bench](so101-perception-bench.md).
