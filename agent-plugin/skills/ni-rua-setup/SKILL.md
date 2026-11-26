---
name: ni-rua-setup
description: Use when the user wants to install, bootstrap, configure, bind hardware, calibrate, or troubleshoot NI-RUA and an SO-101 robot arm (setup, doctor, init-config, cameras, charuco-board, commission, S0-S6 calibration, dry-run) before handing off to ni-rua-robot.
---

# Setting up & Commissioning NI-RUA (SO-101)

Guide the user end-to-end from fresh machine to calibrated SO-101 arm, then hand off to `ni-rua-robot` for live tasks. Run read-only checks first; before any physical arm motion, confirm the user has a physical **12V power cutoff** within reach and knows the live Web console **Hold / E-STOP** URL (`http://127.0.0.1:8781/console`). Pressing `Esc` in the agent does NOT stop the physical arm.

## 1. Install & Diagnostics

1. Bootstrap plugin + isolated Python 3.12 gateway & SO-101 envs (if not yet installed):
   ```sh
   npx -y @nerveint/ni-rua-plugin setup codex    # or: claude | opencode | all (add --no-so101 to skip SO-101 deps)
   ```
2. Run full-stack diagnostics:
   ```sh
   ni-rua-plugin doctor --so101
   ```
3. Optional zero-hardware MCP loop check (verifies host <-> bridge <-> MCP without touching an arm):
   ```sh
   ni-rua-plugin gateway --dry-run
   ```

## 2. SO-101 Hardware Binding & Commissioning

1. Detect USB serial port and UVC cameras and write `~/.config/ni-rua/so101.local.json`:
   ```sh
   ni-rua-so101 init-config
   ni-rua-so101 cameras                          # read-only camera inventory & binding verification
   ```
2. Print a 1:1 A4 ChArUco board for camera calibration (if calibrating):
   ```sh
   ni-rua-so101 charuco-board
   ```
3. Run first-power-on commissioning (confirm clear workspace, 12V cutoff ready, and user approval before energizing servos):
   ```sh
   ni-rua-so101 commission
   ```
   Verify `gripper.empty_close_aperture_m` is measured in `~/.config/ni-rua/so101.local.json` (if `null`, `grip_guard_preset` is refused).

## 3. Guided S0–S6 Calibration (`/calibration`)

- Before `camera_calibration.json` exists, both `ni-rua-plugin gateway --config so101` and the `ni_rua` stdio MCP bridge auto-append `--observation-profile rgb_proprio_v1` for safe cold start.
- Open `http://127.0.0.1:8781/calibration` and guide the user through **S0–S6** (S0 hardware & camera binding check -> S1–S3 intrinsics & ChArUco capture -> S4–S5 hand-eye / base-frame alignment -> S6 verification & save `camera_calibration.json`).
- Re-run `ni-rua-plugin doctor --so101` to confirm calibration and deployment config are green.

## 4. Common Troubleshooting

- **`Missing motor IDs: 1-6`** (or partial IDs): 12V servo power is off, loose power/daisy-chain cable, or wrong USB serial port. Check 12V supply and re-run `ni-rua-so101 init-config`.
- **Linux serial permission denied**: user is not in `dialout` group — run `sudo usermod -aG dialout $USER` and start a new login session.
- **macOS camera black/denied (TCC)**: launch the host/gateway from a terminal app granted Camera permission in System Settings -> Privacy & Security -> Camera (e.g. Terminal.app, iTerm2, Ghostty).
- **Auto-start gateway failed**: inspect `~/.local/share/ni-rua/logs/gateway.log` for the exact traceback or missing config, fix the root cause, and retry.

## 5. Handoff to Operation

Once `doctor --so101` is clean (and the user confirms 12V cutoff + `http://127.0.0.1:8781/console` E-STOP readiness), switch to the **`ni-rua-robot`** skill and call `run_begin` to start operating the arm.
