[English](so101-calibration.md) · [简体中文](so101-calibration.zh-CN.md)

# SO-101 Calibration Guide

This guide covers choosing and printing a ChArUco calibration board, running the end-to-end guided calibration wizard (`/calibration`) on a real SO-101 arm, verifying height and stereo XY consistency, and troubleshooting common issues.

For initial hardware wiring, environment installation, `lerobot-calibrate`, and camera binding, complete the [SO-101 Quickstart](so101-quickstart.md) first. For low-level schemas, APIs, and math details, see [SO-101 Integration](../development/so101-integration.md).

---

## 1. Board Selection

### Why ChArUco?

NI-RUA uses an OpenCV **ChArUco** board (a chessboard with ArUco markers embedded in its white squares):

- **Sub-pixel corner precision**: camera intrinsics and PnP pose estimation use chessboard inner corners, giving sharper localization than ArUco marker corners alone.
- **Unique IDs & partial-occlusion robustness**: embedded ArUco markers assign every inner corner a unique ID. Even when the arm or camera FOV cuts off part of the board, OpenCV still detects the remaining corners reliably with zero 180° orientation ambiguity.

### Validated Board Specifications

Two specifications are validated in this repository. Pick **Spec A** if you are printing on A4 paper yourself, or **Spec B** for high corner density / commercial laser-etched or matte UV-printed rigid plates:

| Parameter | Spec A — Default A4 DIY (Recommended) | Spec B — High-Density / Commercial Rigid Plate |
| :--- | :--- | :--- |
| Config Source | Built-in default (`calibration_defaults.json`) | [`configs/so101_board.json`](../../configs/so101_board.json) |
| Type | `charuco` | `charuco` |
| Grid (`squares_x × squares_y`) | `7 × 5` (24 inner corners) | `12 × 9` (88 inner corners) |
| Square Edge (`square_m` / `square_mm`) | `0.030 m` (`30.0 mm`) | `0.015 m` (`15.0 mm`) |
| Marker Edge (`marker_m` / `marker_mm`) | `0.022 m` (`22.0 mm`) | `0.01125 m` (`11.25 mm`) |
| ArUco Dictionary (`dictionary`) | `DICT_5X5_100` | `DICT_5X5_100` |
| Pattern Outer Size | `210 × 150 mm` | `180 × 135 mm` |
| Thickness (`thickness_m` / `board_thickness_m`) | Measure after mounting (default `0.003 m` / `3.0 mm`) | `0.0032 m` (`3.2 mm`) |
| Best For | General tabletop views and A4 home/office printing | Close-range high-density corners or off-the-shelf rigid boards |

### Material, Flatness & Thickness Requirements

- **Matte, non-reflective finish**: Use matte paper, matte UV-printed aluminum/acrylic, or anodized/ceramic calibration targets. Never use glossy photo paper or glossy lamination — overhead glare washes out ArUco markers and biases corner extraction.
- **Rigid and dead-flat**: Any bowing, curling, or wave violates the planar PnP model and inflates intrinsics and hand-eye residuals.
- **White quiet zone**: Keep at least **10 mm** of clean white margin on all four edges outside the outermost squares so outer markers detect cleanly.
- **Measure the true board thickness (`board_thickness_m`)**: Measure the total vertical distance from the tabletop surface to the printed top face with vernier calipers or a precision ruler. For example, if you glue matte paper onto a `3.0 mm` aluminum or acrylic plate, measure the **combined** stack thickness (backing + adhesive + paper) and enter that exact value in `configs/so101.local.json` (`calibration.board_thickness_m`) or in the **S2** form (`thickness_mm`). Step **S5** directly relies on the physical prior:

  $$\text{board top height} = \text{table height} + \text{board thickness}$$

---

## 2. DIY Printing & Mounting

### Generate Print Files

Use [`scripts/so101_charuco_board.py`](../../scripts/so101_charuco_board.py) in the `ni-rua` environment to render exact 1:1 physical-scale A4 landscape PDF and PNG files:

```bash
# Spec A: default 7x5 board (30.0 mm square, 22.0 mm marker, DICT_5X5_100)
mamba run -n ni-rua python scripts/so101_charuco_board.py --output-dir /tmp/so101_board

# Spec B: 12x9 board from configs/so101_board.json (15.0 mm square, 11.25 mm marker)
mamba run -n ni-rua python scripts/so101_charuco_board.py \
  --board-config configs/so101_board.json --output-dir /tmp/so101_board_12x9

# Custom board parameters (all lengths in mm; square_mm * px_per_mm must be an integer)
mamba run -n ni-rua python scripts/so101_charuco_board.py \
  --squares-x 8 --squares-y 6 --square-mm 25 --marker-mm 18 \
  --dictionary DICT_5X5_100 --output-dir /tmp/so101_board_custom
```

Both `so101_charuco_board.pdf` and `so101_charuco_board.png` are written to `--output-dir`:

- **Resolution**: `24 px/mm` (`609.6 DPI`). Every `30.0 mm` square is an exact integer `720 px` (and every `15.0 mm` square is `360 px`), eliminating sub-pixel resampling blur.
- **Page size & built-in verification ruler**: The PDF page is exact **A4 landscape (`297 × 210 mm`)**, centered with a **100.0 mm verification scale bar** (with `10 mm` ticks) beneath the board and a parameter label above it.

### Printer Settings (Critical Rules)

1. **Print the `.pdf` directly** (e.g., via Preview, Adobe Acrobat, or `lp`). Never print the `.png` from an image viewer that silently rescales images to fit margins.
2. **Paper size & orientation**: **A4**, **Landscape**.
3. **Scale**: **100% / Actual Size**. Strictly disable *"Fit to Page"*, *"Shrink Oversized Pages"*, and *"Borderless"* printing (borderless mode secretly enlarges output by 2–5%).
4. **Media & quality**: Use **>= 120 g/m² matte** inkjet or laser paper, high-contrast black & white, and disable toner/ink-saver mode.

### Mounting & Post-Print Measurement

1. **Mount flat onto a rigid backing plate**: Attach the sheet cleanly with thin double-sided tape or spray adhesive onto a flat aluminum-composite panel, acrylic sheet, or thick rigid board. Smooth out all air bubbles and ensure corners cannot lift.
2. **Measure the printed scale before calibrating**:
   - Measure the bottom **`100.0 mm` scale bar** with calipers or a steel ruler.
   - Also measure the total span across multiple squares (e.g., all `7` squares = `210.0 mm`, then divide by `7`) to get the true single-square edge length with higher precision.
3. **Compensate if the printer rescaled the page**:
   - Printers commonly shrink output by ~0.5–1.5%. For example, if `100.0 mm` measures `99.0 mm` (`0.99×`), then a nominal `30.0 mm` square is **`29.7 mm`** and the `22.0 mm` marker scales by the same factor to **`21.78 mm`** (`22.0 × 29.7 / 30.0`).
   - **Web wizard (`/calibration`)**: Enter the measured `square_mm` (`29.7`), proportionally scaled `marker_mm` (`21.78`), and measured mounted `thickness_mm` directly in the board form on the Start screen or Step **S2**.
   - **Offline CLI (`scripts/so101_calibrate.py`)**: Pass `--square-m 0.0297` to `solve` (the CLI automatically scales `marker_m` by the same ratio unless `--marker-m` is also specified).
4. **Re-measure total mounted thickness**: Measure the final mounted board thickness from table to printed face and enter it as `board_thickness_m` (in `configs/so101.local.json`) or `thickness_mm` (in the wizard).

---

## 3. End-to-End Calibration Workflow

### Prerequisites

Before opening the wizard, confirm:

- The SO-101 follower arm has **12 V servo power ON** and USB serial connected.
- Motor midpoint calibration (`lerobot-calibrate`) has already been run for this arm.
- Both `wrist` and `fixed` cameras have stable `unique_id` bindings and correct `rotate_deg` (`0`, `90`, `180`, or `270`) in `configs/so101.local.json` (verify with `scripts/so101_cameras.py`).
- The web frontend is built (`npm --prefix web run build`).
- Your ChArUco board is mounted and measured.

### Launch the Gateway & Open `/calibration`

Start the NI-RUA gateway using the standalone CPU config `configs/so101_standalone.json` (no GPU or remote perception service required). **For a first-time calibration when no `camera_calibration.json` exists yet, pass `--observation-profile rgb_proprio_v1`:**

```bash
# First-time calibration (no camera_calibration.json yet)
mamba run -n ni-rua python -u -m ni_rua.apps.gateway --agent-backend external \
  --config configs/so101_standalone.json --observation-profile rgb_proprio_v1 \
  --port 8781 --run-dir runs/so101-calib

# Recalibration (or daily operation once calibration has been applied)
mamba run -n ni-rua python -u -m ni_rua.apps.gateway --agent-backend external \
  --config configs/so101_standalone.json --port 8781 --run-dir runs/so101-calib
```

> If you connect to the optional remote GPU 3D perception / cuRobo IK services, replace `configs/so101_standalone.json` with `configs/so101_gateway.json` (still adding `--observation-profile rgb_proprio_v1` before the first `camera_calibration.json` is created).

Open the calibration wizard in your browser:

- English UI: <http://127.0.0.1:8781/calibration?lang=en>
- Default (Chinese UI): <http://127.0.0.1:8781/calibration>

Top-bar controls:

- **Claim Control**: Acquires the 30 s operator writer lease (auto-renewed every 10 s, including during long steps). Required to begin a run, start steps, apply, or roll back.
- **E-STOP**: Works immediately **without** holding the control lease. Preempts any motion step, holds the arm in place, and closes motion admission.
- **Safety admission lock**: Every motion step (`S1`, `S2`, `S4`, `S6`), apply, and rollback automatically closes agent action admission while running so no external agent can move the arm concurrently.

### Choose Calibration Mode

The start screen offers two modes:

1. **First-time Full Calibration (`first_run`)**: Runs `S0 → S1 → S2 → S3 → S4 → S5 → S6` in order. Use this on a new setup or whenever starting fresh.
2. **Recalibrate (`recalibrate`)**: Tick only the hardware components that changed; the wizard builds the minimal ordered plan and reuses valid upstream data from the active calibration:

| Change ID | What Changed | Steps Re-Run |
| :--- | :--- | :--- |
| `fixed_moved` | Fixed camera moved or bumped | `S4 → S5 → S6` |
| `fixed_replaced` | Fixed camera replaced or refocused | `S3 → S4 → S5 → S6` |
| `wrist_bumped` | Wrist camera bumped or remounted | `S2 → S4 → S5 → S6` |
| `table_or_base_moved` | Table changed or robot base moved | `S1 → S2 → S4 → S5 → S6` |
| `board_changed` | Calibration board replaced | `S2 → S4 → S5 → S6` |
| `everything` | Redo everything | `S0 → S1 → S2 → S3 → S4 → S5 → S6` |

> **Dependency staleness**: Data flows `S2, S3 → S4 → S5 → S6` and `S1 → S5 → S6`. Re-running an earlier step automatically marks all downstream steps as `stale` until re-executed.

---

### Step-by-Step Walkthrough (S0 → S6)

All default thresholds below come directly from [`configs/so101.example.json`](../../configs/so101.example.json) (`calibration` and `preflight` sections).

#### S0 · Device Self-Check *(No motion)*

- **Operator**: Confirm 12 V servo power is ON, USB cable is seated, and both camera lenses are unobstructed.
- **Machine**: Reads power-on `preflight.json`, queries live servo voltage/temperature/torque, grabs fresh frames from both cameras to check configured resolution (default `640 × 480`), verifies joint telemetry freshness, and checks worker solver dependencies (`cv2.aruco`, `scipy.optimize`).
- **Pass criteria**:
  - `preflight_ok = true`
  - `power_ok = true` (voltage `10.5–13.5 V`, temperature `<= 60.0 °C`)
  - `torque_ok = true`
  - `telemetry_age_s <= 2.0 s` (`selfcheck_max_telemetry_age_s`)
  - `cameras_ok = true`

#### S1 · Joint Zero & Table Height *(Motion)*

- **Operator**: **Remove the calibration board and every object from the table in front of the arm.** Keep hands clear of the workspace and stay near **E-STOP**. Tick the safety checkboxes and start.
- **Machine**: Closes the gripper (`touch_jaw_normalized = 0.0`), hovers at `0.06 m` above the 4 default XY points (`touch_points_xy_m`: `[0.20, 0.00]`, `[0.24, 0.06]`, `[0.24, -0.06]`, `[0.17, -0.05]`), descends slowly (`touch_step_m = 0.002 m` per `0.12 s` at `touch_speed_scale = 0.5`) down to `touch_min_z_m = -0.06 m`, and records obstruction candidates when joint lag exceeds `0.03 rad` (`touch_contact_lag_rad`) or servo load rises by `> 120.0` raw units (`touch_contact_load_delta`) for `2` consecutive reads after `3` warm-up steps. Full-resolution touch video (`10 Hz`, up to `600` frames / `60 s`) is archived for review. Returns to `home`.
- **Pitfalls**: Leaving the board or mat on the table shifts the measured tabletop height; touching an obstacle produces a false contact. S1 passing only means obstruction candidates were collected — independent review is required before S5.
- **Pass criteria**:
  - `contacts == planned_points` (`4 / 4` by default).

#### S2 · Wrist Camera Intrinsics & Hand-Eye *(Motion)*

- **Operator**: **Lay the ChArUco board flat on the table where BOTH cameras see it, and tape/fix it firmly.** Verify (and edit if needed) the board form fields (`squares_x`, `squares_y`, `square_mm`, `marker_mm`, `dictionary`, `thickness_mm`) to match your physical board. **Do not touch or move the board after starting S2.**
- **Machine**: Checks live `board_both_visible` (`>= 8` corners per camera, `min_board_corners`), moves through `home.q_rad` plus the configured joint-offset grid (`pan/flex/roll/lift_offsets_rad`, 21 poses by default at `speed_scale = 0.6`), settles and photographs both cameras at each pose, keeps samples with `>= 8` wrist corners (`sample_min_wrist_corners`), parks at `home`, and solves wrist intrinsics and `T_link_camera` (dropping samples with PnP RMS `> 2.0 px`).
- **Pitfalls**: Board slipping mid-capture, wrong `square_mm` / `marker_mm` / `dictionary` in the form, or harsh specular glare on the printed face.
- **Pass criteria**:
  - Valid samples `kept >= 8` (`min_samples`)
  - Wrist intrinsics RMS `<= 1.0 px` (`max_intrinsics_rms_px`)
  - Hand-eye translation RMS `<= 5.0 mm` (`max_hand_eye_translation_m = 0.005 m`)
  - Hand-eye rotation RMS `<= 0.02 rad` (`max_hand_eye_rotation_rad`)

#### S3 · Fixed Camera Intrinsics *(No motion)*

- **Operator**: The arm stays still. Click **Start Capture**, pick up the board, and hold it in front of the **fixed** camera, pausing steadily for **~1 s** at each pose. Follow the on-screen 3×3 image grid, distance (`far / mid / near`), and tilt (`frontal / small / large`) coverage guidance. **Be sure to vary distance and tilt the board about BOTH axes** — translating flat in one plane leaves focal length and distortion poorly constrained. Once at least `10` well-distributed views are kept (auto-stops at `handheld_target_views = 20` or `300 s`), click **Solve**. Afterwards, **lay the board flat back on the table and fix it** for S4.
- **Machine**: Samples directly from the worker's fixed camera stream, automatically keeping frames that are still (`< 1.5 px` corner motion, `handheld_still_px`) and sufficiently novel (`>= 0.08`, `handheld_min_novelty`), then calibrates fixed camera `K` and plumb-bob distortion.
- **Pass criteria**:
  - Kept views `views >= 10` (`handheld_min_views`)
  - Fixed intrinsics RMS `<= 1.0 px` (`max_handheld_intrinsics_rms_px`)

#### S4 · Fixed Camera Extrinsic *(Motion)*

- **Operator**: Confirm the board is flat, fixed on the table, and visible to **both** cameras. Do not move it during capture.
- **Machine**: Moves through `extrinsic_pose_count` poses (default **5** poses spread across the grid, starting from `home`), transfers the board pose `T_base_board` via the wrist chain, and computes the fixed camera pose `T_base_camera` in `so101_base`.
- **Pass criteria**:
  - `fixed_samples >= 1`
  - `fixed_chain_reprojection_px` is shown for reference only here (it still contains uncorrected joint zero errors; S5 re-solves and gates it).

#### S5 · Joint Solve & Cross-Check *(No motion)*

- **Operator**: Review the captured S1 touch images/videos or external measurements to verify that the fingertips actually touched the bare table as modeled, then tick the physical-contact confirmation checkbox and start S5 (the arm does not move). Never tick confirmation based solely on encoder lag or raw servo load.
- **Machine**: Runs a Lie-algebra joint nonlinear least-squares optimization over S2 + S4 wrist board views, S1 touch contacts, and the `board top = table_z_m + board_thickness_m` prior to jointly solve joint zero offsets (`offset_rad`), hand-eye `T_link_camera`, and `table_z_m`. Re-solves `T_base_camera` under the corrected offsets, builds the candidate calibration, and computes the old-vs-new diff. **Nothing is installed yet.**
- **Pass criteria (5 hard gates)**:
  - Wrist joint reprojection RMS `<= 1.5 px` (`max_joint_reprojection_px`)
  - Board height error `|board_top_z - (table_z + thickness)| <= 3.0 mm` (`max_board_height_error_m = 0.003 m`)
  - Max touch residual `<= 6.0 mm` (`max_touch_residual_m = 0.006 m`)
  - Fixed camera chain reprojection RMS `<= 6.0 px` (`max_fixed_reprojection_px`)
  - Max joint zero offset `|offset_rad| <= 0.15 rad` (`max_joint_offset_rad`)

#### S6 · Apply, Height Verification & Stereo XY Consistency *(Motion for verification only)*

S6 has three distinct stages:

1. **Review & Apply *(No motion)***:
   - Inspect the old-vs-new comparison table (table height, per-joint `offset_rad`, wrist/fixed translation and rotation deltas, focal lengths, and residuals).
   - Tick the review checkbox and click **Apply New Calibration**.
   - Atomically creates a new version in `<calibration_bundle_dir>/versions/<UTC>/` (importing any pre-existing calibration file first as a rollback target), atomically replaces `camera_calibration.json`, writes updated `joint_mapping.<joint>.offset_rad` into `configs/so101.local.json` while saving a timestamped backup (`so101.local.json.bak-<UTC>`), updates the `active` pointer, and hot-reloads the new offsets in memory **without moving the arm**.
2. **Holdout Height Verification *(Motion)***:
   - **Clear the calibration board and all objects from the table again.**
   - Start S6 verification touch-off on independent holdout points (`verify_points_xy_m`, default 2 points: `[0.22, 0.03]` and `[0.30, -0.04]`) and compare measured contact heights against the newly installed `table_z_m`.
   - **Pass criteria**:
     - `contacts == planned_points` (`2 / 2` by default)
     - Max height error `|contact_z - table_z| <= 5.0 mm` (`max_verify_touch_error_m = 0.005 m`)
   - **Note on failure**: If S6 verification fails, the newly applied calibration **stays loaded and is NOT automatically rolled back**. Inspect the failed touch measurements first, then either re-verify or click **Roll Back to Previous Version**.
3. **Stereo XY Camera–Fingertip Consistency Check *(No motion)***:
   - S6 touch-off verifies **height (`Z`)**, not lateral (`XY`) visual alignment. Use the **XY Camera–Fingertip Consistency Check** card on the S6 screen once the arm is stationary:
   - Click **Freeze Current Frame (No Motion)** — waits for the arm to settle and captures a frozen pair of original RGB frames from `fixed` and `wrist` along with FK state (sends zero motor commands).
   - Click the exact same **fixed inner fingertip tip center** in both the `fixed` and `wrist` images (marked with a red `+`; skip if occluded or blurry), then click **Check This Point XY**.
   - Editable default thresholds (from [`configs/so101.example.json`](../../configs/so101.example.json)):
     - XY tolerance: **`5.0 mm`** (`xy_max_error_m = 0.005 m`)
     - Max stereo reprojection residual: **`3.0 px`** (`xy_max_reprojection_error_px = 3.0`)
     - Min ray parallax angle: **`1.0°`** (`xy_min_parallax_deg = 1.0`)
   - Repeat across several workspace positions, heights, and orientations. Results are logged in `run.json` (`xy_verifications`) separately from S6 height verification. Excessive reprojection residual or insufficient parallax yields `unknown` (never a false pass). Because wrist extrinsics share FK, treat a pass as a conditional camera–arm consistency check rather than an external absolute metrology certificate.

---

## 4. Maintenance, Rollback & Troubleshooting

### Console Calibration Status Badge

The header badge on `/console` and `/calibration` polls `GET /v1/calibration/summary` every 10 s:

| Badge Status | Meaning | Recommended Action |
| :--- | :--- | :--- |
| `valid` *(Green)* | Active calibration loaded, matches bundle version, includes joint calibration, staleness OK, and the exact active version passed S6 height verification. | Ready for operation. |
| `unverified` *(Yellow)* | Calibration is loaded, but the active version has no passing S6 height verification record. | Run S6 verification touch-off. |
| `verification_failed` *(Red)* | Active version failed S6 height verification. | Inspect S6 errors, recalibrate, or roll back. |
| `maybe_stale` *(Yellow)* | Staleness check detected `shifted` (`> 15.0 px`), active file differs from bundle version, `joint_calibration` is missing, or fixed intrinsics were copied unverified from wrist. | Re-run affected steps via **Recalibrate**. |
| `needed` *(Red)* | No loadable `camera_calibration.json` or validation error on load. | Run **First-time Full Calibration**. |
| `unknown` *(Gray)* | Gateway/worker unavailable or status query failed. | Check gateway and `so101` worker logs. |

### Automatic Staleness Check (`staleness`)

At most once every `staleness_check_period_s` (`5.0 s`), when the arm is idle and both cameras see the ChArUco board, the worker projects the board pose from the wrist kinematic chain into the fixed camera using the active calibration and compares against the fixed camera's detected corners. If the RMS discrepancy exceeds `staleness_max_board_shift_px` (`15.0 px`), status becomes `shifted` and the badge turns yellow. This check is read-only and never moves the arm or modifies calibration.

### Version History & One-Click Rollback

Every applied calibration is stored atomically under `calibration_bundle_dir` (default `calibration/` next to `camera_calibration_file`, e.g., `~/.cache/ni-rua/so101/calibration/versions/<UTC>/`). From the **Version History** table on the `/calibration` home screen (or the rollback button on S6), click **Roll Back** on any historical version to atomically restore both `camera_calibration.json` and `configs/so101.local.json` joint offsets and hot-reload without moving the arm.

### Common Failure Modes & Fixes

| Symptom / Failure | Likely Cause | How to Fix |
| :--- | :--- | :--- |
| `board_both_visible` fails (`< 8` corners) | Board outside one camera's FOV; specular glare; board form spec (`7×5` vs `12×9`) does not match physical board; wrong `rotate_deg` or swapped camera bindings. | Reposition board into shared view; reduce glare; update S2 board form to match actual board; verify cameras with `scripts/so101_cameras.py`. |
| S2 intrinsics (`> 1.0 px`) or hand-eye RMS (`> 5.0 mm` / `> 0.02 rad`) fails | Board slipped during capture; paper warped/curled; printer scaling mismatch; loose wrist camera mount; wrong `joint_mapping.<joint>.sign`. | Tape board firmly onto a rigid plate; measure printed square size and update S2 form; tighten wrist mount; verify joint sign directions against URDF. |
| S3 fixed intrinsics RMS `> 1.0 px` or coverage stalls | Only translating the board flat without tilting; hand jitter/motion blur; board too close and out of focus. | Pause ~1 s per pose, keep focus sharp, cover all 9 grid cells, vary near/far distance, and tilt about both horizontal and vertical axes. |
| S5 `board_height_error_m > 3.0 mm` | **Most common: wrong `thickness_mm` / `board_thickness_m`** (e.g., forgot backing plate thickness, or left `3.0 mm` default for bare paper); or S1 touched debris instead of the bare table. | Measure true tabletop-to-print-face thickness with calipers, update S2 board form, re-run S2/S4/S5 (or re-run S1 if the table was not bare). |
| S5 `max_joint_offset_rad > 0.15 rad` | Servo horn slipped or `lerobot-calibrate` midpoint is far from URDF zero. | Re-run `lerobot-calibrate` for the arm, verify `joint_mapping` signs, then restart full calibration. |
| S6 height verification fails (`> 5.0 mm`) | Debris left on table, unstable contact trigger, or inaccurate upstream S1/S5 fit. Remember: **S6 does NOT auto-rollback.** | Clear table, inspect `S6_touches.json` / touch video; re-run S6 verification, or roll back to the previous version in **Version History** and recalibrate. |

---

## Appendix: Offline CLI Calibration (`scripts/so101_calibrate.py`)

For headless/developer workflows outside the Web UI, [`scripts/so101_calibrate.py`](../../scripts/so101_calibrate.py) provides a standalone CLI. Hardware capture commands run in the `so101` mamba environment (with the gateway stopped); `board`, `solve`, and `verify` are CPU-only and also run in `NI-RUA`:

```bash
# 1. Render printable ChArUco board (defaults to 7x5; use scripts/so101_charuco_board.py for custom configs)
mamba run -n ni-rua python scripts/so101_calibrate.py board --output-dir /tmp/so101_board

# 2. Manual capture (torque OFF; pose arm by hand, press Enter per view, q to finish)
mamba run -n so101 python scripts/so101_calibrate.py capture \
  --config configs/so101.local.json --output-root /tmp/so101_calib --manual

# 3. Hand-held fixed camera intrinsics capture (automatic still-and-novel view selection)
mamba run -n so101 python scripts/so101_calibrate.py capture-handheld \
  --config configs/so101.local.json --camera fixed --session /tmp/so101_calib/<session_dir> --auto

# 4. Offline solve (pass --square-m if your printed squares deviate from nominal, e.g. 0.0297 m)
mamba run -n ni-rua python scripts/so101_calibrate.py solve \
  --config configs/so101.local.json --session /tmp/so101_calib/<session_dir> --square-m 0.030

# 5. Cross-camera reprojection verification on a held-out capture session
mamba run -n ni-rua python scripts/so101_calibrate.py verify \
  --session /tmp/so101_calib/<heldout_session> \
  --calibration /tmp/so101_calib/<session_dir>/camera_calibration.json
```
