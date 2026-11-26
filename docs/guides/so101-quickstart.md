# SO-101 Real-Robot Quickstart

[English](so101-quickstart.md) · [简体中文](so101-quickstart.zh-CN.md)

This guide walks you from unboxing an **SO-101** desktop follower arm to your first verified motion and interactive agent run—either **purely locally on a laptop (macOS or Linux, CPU-only)** or connected to an **optional remote NVIDIA GPU server** for 3D perception and cuRobo IK.

---

## 1. Hardware & Power Checklist

Before touching software, verify the physical setup:

- **Arm:** TheRobotStudio **SO-101 Follower** (5 arm joints + 1 gripper jaw, Feetech STS3215 servos on one serial bus, IDs 1–6).
- **12 V Servo Power Supply:** Connected and switched on before running any motion or preflight script.
  > [!IMPORTANT]
  > The USB serial adapter enumerates on USB power alone. If startup reports `servo_power_off`, `Missing motor IDs: 1-6`, or an empty `broadcast_ping()`, check that the **12 V servo power** is turned on—this is not a software bug. Preflight also verifies supply voltage (`10.5–13.5 V`) and motor temperature (`<= 60 °C`).
- **USB Serial Connection:** Connect the motor controller board to your computer via USB.
- **Two UVC Cameras:**
  - `wrist` (eye-in-hand camera mounted on `gripper_link`)
  - `fixed` (eye-to-hand static overview camera overlooking the tabletop workspace)
- **Physical E-STOP / Power Switch:** Keep an inline physical power switch for the 12 V servo supply within reach at all times. Software `hold` / `estop` relies on a live serial bus and worker process.
- **macOS Camera Permission (TCC):** macOS grants camera access to the *responsible app* of the process tree. Always launch the NI-RUA gateway and scripts from a terminal app that has macOS camera permission (e.g., `Terminal.app`, `iTerm2`, or `Ghostty`), rather than from an IDE or GUI launcher without `NSCameraUsageDescription`.

---

## 2. Environment Installation

### Option A — One-Line Zero-Clone Setup (Recommended)

Requires **Node >= 18** and **Python >= 3.12** (or `uv`). This installs `@nerveint/ni-rua-plugin` with the lazy stdio MCP bridge (`ni-rua-plugin mcp`) and bundled **`ni-rua-setup`** / **`ni-rua-robot`** skills, and **by default** creates two isolated Python 3.12 environments under `~/.local/share/ni-rua/envs/{gateway,so101}` (avoiding upstream `packaging`/`numpy` dependency conflicts between `openai-codex` and `lerobot`; pass `--no-so101` to skip SO-101 hardware deps):

```bash
npx -y @nerveint/ni-rua-plugin setup codex            # or: claude | opencode | all
npm i -g @nerveint/ni-rua-plugin                      # optional: keep `ni-rua-plugin` on PATH
ni-rua-plugin doctor --so101
```

### Option B — From Source (Mamba / Conda)

From a git checkout, NI-RUA isolates the main Harness environment (`ni-rua`) from the SO-101 LeRobot driver environment (`so101`). Do **not** create a `.venv` inside the repo.

```bash
git clone https://github.com/NerveInt/NI-RUA.git
cd NI-RUA

# 1. Create the main Harness environment (Python 3.12 + FFmpeg)
mamba env create -f environment.yml

# 2. Create the isolated SO-101 hardware worker environment (LeRobot 0.6.1, SciPy, OpenCV ArUco)
mamba env create -f environment.so101.yml
# Or inside an existing Python >= 3.12 environment:
# pip install -r requirements-so101.txt

# 3. Build the Web operator console & calibration UI (Node 22)
npm --prefix web ci && npm --prefix web run build
```

> [!NOTE]
> Make sure `ffmpeg` and `ffprobe` (with `libx264`) are available inside the `so101` environment (or on `PATH`) if you want automatic motion-only H.264 video recording. On macOS, if a configured pip mirror lacks `openai-codex` / `openai-codex-cli-bin`, install them from official PyPI:
> `mamba run -n ni-rua python -m pip install --index-url https://pypi.org/simple openai-codex-cli-bin==0.160.0 openai-codex==0.160.0`.

### Optional: Build & Install the Standalone `ni-rua-harness` Wheel

If you prefer calling `ni-rua` / `ni-rua-so101` directly from source, build the complete wheel (which bundles web assets and config presets) using `scripts/build_package.py` (never plain `pip wheel .`):

```bash
mamba run -n ni-rua python scripts/build_package.py --output dist
mamba run -n ni-rua python scripts/check_package.py dist/*.whl
mamba run -n ni-rua python -m pip install dist/*.whl
```

This installs `ni-rua`, `ni-rua-so101`, `ni-rua-queue`, `ni-rua-report`, and `ni-rua-audit` into your `ni-rua` environment.

---

## 3. First-Time Hardware Preparation

### 3.1 LeRobot Servo Calibration (`lerobot-calibrate`)

Power on the 12 V supply and run LeRobot's motor calibration once so each STS3215 servo's range and zero center match your assembly (ensure the `robot_id` matches `"so101_follower"` or whatever you set in `configs/so101.local.json`):

```bash
# Replace <serial_port> with e.g. /dev/tty.usbmodemXXXX (macOS) or /dev/ttyACM0 (Linux)
mamba run -n so101 lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=<serial_port> \
  --robot.id=so101_follower
```

### 3.2 Inspect and Bind Deterministic Camera `unique_id`s

OpenCV numeric indices (`0`, `1`) shift whenever a USB camera is replugged or enumerates in a different order. Always bind cameras by physical **`unique_id`** using `ni-rua-so101 cameras` (or [`scripts/so101_cameras.py`](../../scripts/so101_cameras.py) from a checkout), run with the **`so101`** interpreter (because OpenCV's AVFoundation index rule on macOS depends on the worker's OpenCV version):

```bash
# Installed wheel / one-line setup:
ni-rua-so101 cameras
# Or from a source checkout:
mamba run -n so101 python scripts/so101_cameras.py
```

The command prints every camera's `opencv` index, `transport`, `unique_id`, and `name / model`:

- **Two identical cameras:** Unplug one (or plug them in one at a time) and re-run `ni-rua-so101 cameras` to see which `unique_id` belongs to `wrist` vs. `fixed`.
- **macOS:** `unique_id` is the AVFoundation `uniqueID` (e.g., `0x1340001b170209`). For USB cameras it encodes the USB port—keep each camera plugged into the same USB port.
- **Linux:** Use the `/dev/v4l/by-id/...-video-index0` (or `/dev/v4l/by-path/...-video-index0`) symlink printed by the script.

### 3.3 Create and Configure `so101.local.json`

Use the interactive wizard from `ni-rua-so101` (detects USB serial ports and UVC cameras and writes `~/.config/ni-rua/so101.local.json`), or copy the annotated template [`configs/so101.example.json`](../../configs/so101.example.json) to the Git-ignored `configs/so101.local.json` in a checkout:

```bash
# Zero-clone / wheel workflow (writes ~/.config/ni-rua/so101.local.json):
ni-rua-so101 init-config

# Or from a source checkout:
cp configs/so101.example.json configs/so101.local.json
```

Edit `configs/so101.local.json` for your machine (all numeric fields carry strict units and bounds validated at load time by `ni_rua/robot/backends/so101/deployment.py`):

1. **Interpreter & Serial Port:**
   - `"python"`: path to your `so101` environment's Python executable (e.g., `~/miniforge3/envs/so101/bin/python` or `/opt/homebrew/Caskroom/miniforge/base/envs/so101/bin/python`).
   - `"port"`: serial port of the follower arm (`/dev/tty.usbmodemXXXX` or `/dev/ttyACM0`).
   - `"robot_id"`: `"so101_follower"` (must match `lerobot-calibrate`).
2. **Cameras (`cameras.wrist`, `cameras.fixed`):**
   - Set `"unique_id"` to the exact string from `scripts/so101_cameras.py` (leave `"device_name": null` unless matching a unique model name).
   - Set `"rotate_deg"` (`0 | 90 | 180 | 270`) if a camera is mounted inverted or rotated.
   - Keep `"width": 640`, `"height": 480`, `"fps": 30` as starting defaults.
3. **Gripper Measured Apertures (`gripper`):**
   - `"empty_close_aperture_m"`: **Measure and set before real runs.** Command several empty full closes (`goal_normalized: 0`) and record the largest `gripper_execution.measured_opening_m` (typically `0.0042–0.0051 m` on real hardware). This is the zero point for `grip_guard_preset` (`non_empty`, `thin_object`, `rigid_object`, `wide_object`) and the post-grasp slip re-check; while `null`, presets are rejected and slip re-check is disabled.
   - `"full_open_aperture_m"`: **Measure and set** at `goal_normalized: 1` (around `0.136 m`; when `null`, uses the CAD opening `0.1372 m`).
4. **Safe Rest Pose & Workspace (`home`, `workspace`):**
   - Verify `home.q_rad` (default `[0.0, -1.5, 1.45, 0.9, 0.0]` in template, or a clear ready pose such as `[0.0, -0.95, 0.90, 1.00, 0.0]`) and ensure `workspace.min_m[2]` stays safely above your table surface before calibration.

Verify your camera bindings without opening cameras or touching the arm:

```bash
mamba run -n so101 python scripts/so101_cameras.py --deployment configs/so101.local.json
```

Exit code `0` confirms both `wrist` and `fixed` resolve deterministically without ambiguity or duplicates.

---

## 4. Pure Local CPU Mode vs. Optional Remote GPU Services

NI-RUA provides two gateway configurations for SO-101:

- **[`configs/so101_standalone.json`](../../configs/so101_standalone.json) (Default for standalone no-GPU machines):** Disables external perception (`"external_perception": false`) and runs pure CPU local IK (`Pink + DAQP` via `so101_kinematics.ik`) with zero GPU or port `8331` dependencies.
- **[`configs/so101_gateway.json`](../../configs/so101_gateway.json) (For optional remote GPU services):** Enables `"external_perception": true` pointing at `http://127.0.0.1:8331` for VGGT/SAM 3D perception (and can be paired with cuRobo IK on `8341`).

| Capability | Standalone CPU Mode (`configs/so101_standalone.json`) | Optional Remote GPU Mode (`configs/so101_gateway.json`) |
| :--- | :--- | :--- |
| **Compute Needed** | CPU only (no GPU required) | Local laptop + remote NVIDIA GPU host (`bfloat16`, CUDA 12.8) |
| **3D Perception (VGGT + SAM)** | **Off** (`"external_perception": false`) | **On** (`services/vggt_height/` on remote port `8331` / `8333`) |
| **Available Perception Tools** | RGB images, proprioception, `robot_contact_overlay`, calibrated `geometry_query` (`ray`, `triangulate`, `pixel_plane`, `project`) | Adds `object_table`, `scene3d`, `obs3d` startup grounding, `vggt_height`, optional `held_perception` & `contact_top_z` |
| **IK Solver** | Packaged local CPU `Pink + DAQP` IK (`so101_kinematics.ik`, default when `ik_service` is absent) | Optional **cuRobo v2** GPU multi-seed IK + `path_check` collision advisor (`services/curobo_ik/`, port `8341`), enabling `approach_tolerance_rad` / `roll_tolerance_rad` & `obstacle_object_ids` |
| **Best For** | Out-of-the-box bring-up, calibration, RGB+proprioception manipulation, interactive testing anywhere | Dense tabletop object segmentation/3D point clouds and obstacle-aware IK |

### Option 1: Standalone CPU Run (Out-of-the-Box, No GPU)

Use `--config configs/so101_standalone.json` and leave `ik_service` absent (or `null`) in `configs/so101.local.json`:

- **Already calibrated (`camera_calibration.json` exists):** run with `--config configs/so101_standalone.json` directly.
- **First-time cold start (no `camera_calibration.json` yet):** add `--observation-profile rgb_proprio_v1` so the gateway starts in uncalibrated mode and lets you open `/calibration`.

### Option 2: Connect Optional Remote GPU 3D Perception & cuRobo IK

1. **Deploy & test on your GPU server:**
   - **VGGT + SAM 3.1 service:** follow [`services/vggt_height/README.md`](../../services/vggt_height/README.md) (`setup.sh`, `download_weights.sh`, `start.sh` on `127.0.0.1:8331`).
   - **cuRobo IK service (optional):** follow [`services/curobo_ik/README.md`](../../services/curobo_ik/README.md) (`deploy.sh <gpu-host> <remote-dir>`, runs on `127.0.0.1:8341` in the `RoboDojo` CUDA environment).
2. **Configure SSH tunneling & remote management in your gateway config:**
   Copy `configs/so101_gateway.json` plus the `remote` block from [`configs/so101.remote.example.json`](../../configs/so101.remote.example.json) into your NI-RUA run config (e.g. `~/.cache/ni-rua/so101/ni-rua.local.json`, replacing `<gpu-host>` and paths with your server's SSH alias and directories—**never** put `remote` inside `configs/so101.local.json`), and keep `"external_perception": true` and `"vggt_support_z_m": null` (which automatically uses each observation's calibrated `table_z_m`). (If you start from `configs/so101_gateway.json` without a GPU server, pass `--no-external-perception`.)
3. **Enable cuRobo IK in `configs/so101.local.json` (if using `curobo_ik`):**
   Add the `ik_service` block (ensure `SO101_JOINT_LIMIT_MARGIN_RAD` on the server matches `motion.joint_limit_margin_rad`, default `0.05`):

   ```json
   "ik_service": {
     "solver": "curobo",
     "url": "http://127.0.0.1:8341",
     "timeout_s": 10,
     "num_solutions": 8,
     "table_top_z_m": -0.02,
     "path_check_step_rad": 0.01
   }
   ```

See [SO-101 Integration §9](../development/so101-integration.md#9-distributed-deployment-mac--remote-linux-gpu-server) and [§12](../development/so101-integration.md#12-ik-planning-the-curobo-ik-service-and-modereach) for full lifecycle and `path_check` details.

---

## 5. Commissioning, Running, and Safety Controls

### 5.1 First-Power-On Commissioning (~1.5 min)

Clear the tabletop, keep a hand near your physical 12 V cutoff switch, and run `ni-rua-so101 commission` (or [`scripts/so101_commission.py`](../../scripts/so101_commission.py) from a checkout) from a terminal with camera access:

```bash
# Installed wheel / one-line setup:
ni-rua-so101 commission \
  --deployment ~/.config/ni-rua/so101.local.json \
  --out ~/.cache/ni-rua/so101/commission

# Or from a source checkout:
PYTHONPATH=. mamba run -n so101 python scripts/so101_commission.py \
  --deployment configs/so101.local.json \
  --out ~/.cache/ni-rua/so101/commission
```

This runs the real `SO101Adapter` through preflight, moves to the ready pose, steps each joint (`±0.1 rad`) with and without sag correction, sweeps `shoulder_pan`, exercises the gripper (`1.0 → 0.0 → 0.5`), tests `±2 cm` Cartesian moves in X/Y/Z, and writes `commissioning.json` plus camera snapshots. Any failure halts immediately with a confirmed hold.

### 5.2 Run Mode A: Agent Plugin Mode (Zero-Friction Single Window or Manual Gateway)

**Recommended single-window flow (lazy stdio MCP bridge `ni-rua-plugin mcp`):**

1. Install once (includes the isolated gateway and SO-101 runtimes by default) and verify:
   ```bash
   npx -y @nerveint/ni-rua-plugin setup codex         # or: claude | opencode | all
   ni-rua-plugin doctor --so101
   ```
2. Open `codex` (or `claude` / `opencode`) directly — the bundled **`ni-rua-setup`** skill can walk you through `ni-rua-so101 init-config`, `doctor`, `commission`, and S0–S6 `/calibration`, and **`ni-rua-robot`** drives the arm directly with no second gateway terminal needed:
   ```text
   $ codex
   > Use NI-RUA to inspect the table, lift the SO-101 gripper 2 cm, verify, and return.
   ```
   Normal coding sessions have zero background gateway processes and zero MCP errors. On the first `run_begin` call, `ni-rua-plugin mcp` automatically spawns the singleton local SO-101 gateway (auto-enabling `--observation-profile rgb_proprio_v1` if `camera_calibration.json` does not exist yet; logs under `~/.local/share/ni-rua/logs/gateway.log`) and surfaces the live Web console URL (`http://127.0.0.1:8781/console?lang=en`, plus `/calibration` if uncalibrated).

**Optional manual gateway mode** (for custom `--run-dir`, explicit `--prompt`, `--dry-run`, remote GPU configs, or proxy-only `NI_RUA_AUTO_START=0` / custom `NI_RUA_MCP_URL`):

```bash
ni-rua-plugin gateway --config so101 \
  --port 8781 --run-dir runs/so101-demo \
  --prompt "Describe the table, move the gripper straight up 2 cm, verify it, and move it back."
```
*(From a checkout, you can also run `mamba run -n ni-rua python -u -m ni_rua.apps.gateway --agent-backend external --config configs/so101_standalone.json ...`. When a manual gateway is already listening on `NI_RUA_MCP_URL`, default `http://127.0.0.1:8790/mcp`, the stdio bridge transparently proxies to it without starting a second gateway.)*

### 5.3 Run Mode B: Full Hosted Harness Mode

Let NI-RUA launch and supervise the agent directly (uses `model` in `configs/so101_standalone.json` and credentials in `~/.codex`, or an explicit provider profile for Claude Code / OpenCode):

```bash
mamba run -n ni-rua python -u -m ni_rua.apps.gateway \
  --config configs/so101_standalone.json \
  --port 8781 --run-dir runs/so101-hosted \
  --prompt "Describe the table, move the gripper straight up 2 cm, verify it, and move it back."
```

*(If you installed the `ni-rua-harness` wheel, `mamba run -n ni-rua python -u -m ni_rua.apps.gateway` can be replaced with `ni-rua` and `--config so101`.)*

### 5.4 Console Safety Controls (`http://127.0.0.1:8781/console?lang=en`)

Open **<http://127.0.0.1:8781/console?lang=en>** in your browser during any run:

- **Claim Control:** Acquires the single-writer operator lease (new tabs open read-only).
- **Hold / Stop:** Closes action admission and commands every joint to hold its measured position. (`Stop` waits for the control lock if `go_home` or a calibration step is active; **`E-STOP`** is preemptive.)
- **E-STOP:** Immediately cuts action admission, holds joints under a dedicated safety lock, and interrupts any running agent turn.
- **Relax (Torque Off):** Cuts servo torque on all joints (**the arm will drop under gravity—physically support the arm before confirming `relax`**). Latches a safety gate until you click **Torque On** and capture a fresh observation.
- **Go Home / Resume:** Moves safely to `home.q_rad` (when idle) or reopens motion admission after a stop with a fresh `robot_observe`.

> [!WARNING]
> **Closing the browser tab or pressing `Esc` in an external agent does NOT stop the physical arm.** Always use **Hold / E-STOP** in `/console` or your physical 12 V power switch.

---

## 6. Next Step: Calibrate Your Arm & Cameras

Before trusting 3D coordinates, `pixel_plane`, `triangulate`, or `robot_action` table tasks, complete the guided S0–S6 calibration:

1. Start the gateway. On a **first uncalibrated run** (no `camera_calibration.json` yet), both the lazy MCP bridge auto-start and `ni-rua-plugin gateway --config so101` automatically enable `--observation-profile rgb_proprio_v1` (pass it explicitly when invoking `ni-rua` directly):
   ```bash
   ni-rua-plugin gateway --config so101 --port 8781 --run-dir runs/so101-calib
   ```
2. Open **<http://127.0.0.1:8781/calibration?lang=en>** in your browser.
3. Generate the exact 1:1 printable ChArUco board with `ni-rua-so101 charuco-board` and follow the **[SO-101 Calibration Guide](so101-calibration.md)** to step through S0–S6.

Once S6 applies cleanly, your deployment automatically switches from uncalibrated `rgb_proprio_v1` to `rgb_proprio_calibrated_v1` with calibrated intrinsics, hand-eye transforms, fixed-camera extrinsics, joint zero-offsets, and table height. Subsequent runs only need `--config so101` (or `--config configs/so101_standalone.json`).
