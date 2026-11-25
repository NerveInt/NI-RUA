// `ni-rua-plugin doctor` 扩展体检：Python/uv/网关/本地 CPU IK/ffmpeg 与 SO-101 硬件就绪度检查
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import {
  checkLocalIkImports,
  captureVersion,
  findGatewayBinary,
  findOnPath,
  findPython312,
  findSo101DeploymentPath,
  findSo101Python,
  listSerialDevices,
  resolveSo101CalibrationPath,
} from "./runtime.mjs";

export function runStackChecks({ line, so101 = false, cwd = process.cwd() }) {
  const uvBin = findOnPath("uv");
  if (uvBin) {
    line("ok", `uv ${captureVersion(uvBin) ?? "installed"} at ${uvBin}`);
  } else {
    line("info", "uv not found on PATH (optional; recommended for fast Python 3.12 setup: curl -LsSf https://astral.sh/uv/install.sh | sh)");
  }

  const py312 = findPython312();
  if (py312) {
    line("ok", `python ${py312.version} at ${py312.bin} (needs >= 3.12)`);
  } else {
    line("info", "python >= 3.12 not found on PATH (needed only if bootstrapping gateway without uv)");
  }

  const gateway = findGatewayBinary({ cwd });
  if (gateway) {
    line("ok", `ni-rua gateway (${gateway.kind}): ${gateway.bin}`);
    if (gateway.python) {
      const ik = checkLocalIkImports(gateway.python);
      if (ik.ok) {
        line("ok", "local CPU IK stack (pin, pink, coal, daqp, ruckig) importable");
      } else if (ik.missing.length) {
        line("warn", `local CPU IK packages missing in gateway python: ${ik.missing.join(", ")}`);
      } else {
        line("warn", `local CPU IK check failed (${ik.error})`);
      }
    }
  } else {
    line("info", "ni-rua gateway binary not installed yet (run: ni-rua-plugin setup <claude|codex|opencode>, or test with: ni-rua-plugin gateway --dry-run)");
  }

  const ffmpegBin = findOnPath("ffmpeg");
  if (ffmpegBin) {
    line("ok", `ffmpeg at ${ffmpegBin}`);
  } else {
    line("info", "ffmpeg not found on PATH (optional; install via brew/apt for motion H.264 video recording)");
  }

  if (!so101) return;

  const so101Py = findSo101Python();
  if (so101Py) {
    line("ok", `so101 worker python: ${so101Py}`);
  } else {
    line("warn", "so101 isolated Python 3.12 env not found (run: ni-rua-plugin setup <host> --so101)");
  }

  const serials = listSerialDevices();
  if (serials.length) {
    line("ok", `SO-101 USB serial device(s) detected: ${serials.join(", ")}`);
  } else {
    line("warn", "no USB serial devices found (/dev/tty.usbmodem*, /dev/ttyACM*, /dev/ttyUSB*); check USB cable and 12V servo power");
  }

  if (process.platform === "linux") {
    const groupsRes = spawnSync("id", ["-Gn"], { encoding: "utf8", timeout: 5000 });
    const groups = (groupsRes.stdout || "").trim().split(/\s+/).filter(Boolean);
    if (groups.includes("dialout")) {
      line("ok", "Linux serial permission: current user is in dialout group");
    } else {
      line("warn", "Linux serial permission: current user is not in `dialout` group (run: sudo usermod -aG dialout $USER && re-login)");
    }
  } else if (process.platform === "darwin") {
    line("info", "macOS camera permission (TCC): run gateway from a terminal app granted Camera access (Terminal.app / iTerm2 / Ghostty)");
  }

  const depPath = findSo101DeploymentPath({ cwd });
  if (!depPath) {
    line("warn", "so101.local.json not found in ./configs/ or ~/.config/ni-rua/ (run: ni-rua-so101 init-config)");
    return;
  }
  line("ok", `so101 deployment config: ${depPath}`);
  try {
    const dep = JSON.parse(fs.readFileSync(depPath, "utf8"));
    if (dep?.gripper?.empty_close_aperture_m === null || dep?.gripper?.empty_close_aperture_m === undefined) {
      line("warn", `so101: gripper.empty_close_aperture_m is null in ${depPath} (grip_guard_preset will be refused until measured)`);
    } else {
      line("ok", `so101: gripper.empty_close_aperture_m = ${dep.gripper.empty_close_aperture_m} m`);
    }
  } catch (error) {
    line("warn", `so101: failed to parse ${depPath} (${error.message})`);
  }

  const calibPath = resolveSo101CalibrationPath({ cwd, deploymentPath: depPath });
  if (calibPath) {
    line("ok", `so101 camera calibration: ${calibPath}`);
  } else {
    line("warn", "so101: camera_calibration.json missing; cold start needs --observation-profile rgb_proprio_v1 (auto-added by `ni-rua-plugin gateway --config so101`), then complete /calibration");
  }
}
