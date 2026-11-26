// 共享运行环境、网关二进制与 SO-101 硬件路径探测辅助模块
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

export const ESTOP_WARNING =
  "SAFETY: Pressing Esc in Claude Code / Codex / OpenCode does NOT stop a physical robot arm. " +
  "Always use Hold / E-STOP in the Web console (/console) or your physical 12V power cutoff.";

export const DEFAULT_SO101_DEPS = [
  "lerobot==0.6.1",
  "feetech-servo-sdk==1.0.0",
  "pyserial==3.5",
  "numpy>=2.0,<2.3",
  "scipy==1.18.1",
  "opencv-python-headless>=4.10",
  "Pillow==12.1.1",
  "pin==4.1.0",
  "coal==3.0.3",
  "pin-pink==4.4.0",
  "qpsolvers==4.13.0",
  "daqp==0.11.2",
  "ruckig==0.19.4",
];

export const niRuaShareRoot = () => path.join(os.homedir(), ".local", "share", "ni-rua");
export const gatewayEnvDir = () => path.join(niRuaShareRoot(), "envs", "gateway");
export const so101EnvDir = () => path.join(niRuaShareRoot(), "envs", "so101");
export const gatewayLockPath = () => path.join(niRuaShareRoot(), "gateway.lock");
export const gatewayLogPath = () => path.join(niRuaShareRoot(), "logs", "gateway.log");
export const gatewayRunsDir = () => path.join(niRuaShareRoot(), "runs");
export const niRuaConfigDir = () => path.join(process.env.XDG_CONFIG_HOME || path.join(os.homedir(), ".config"), "ni-rua");

export function isSo101Preset(configArg) {
  if (!configArg) return false;
  const normalized = String(configArg).trim();
  return normalized === "so101"
    || normalized === "so101_standalone"
    || normalized.endsWith("/so101_standalone.json")
    || normalized === "so101_standalone.json";
}

export function findOnPath(name, pathEnv = process.env.PATH) {
  for (const dir of (pathEnv || "").split(path.delimiter)) {
    if (!dir) continue;
    const candidate = path.join(dir, name);
    try {
      fs.accessSync(candidate, fs.constants.X_OK);
      if (fs.statSync(candidate).isFile()) return candidate;
    } catch { /* 继续查找下一个目录 */ }
  }
  return null;
}

export function captureVersion(bin, args = ["--version"]) {
  const res = spawnSync(bin, args, { encoding: "utf8", timeout: 10000 });
  if (res.error || res.status !== 0) return null;
  const text = `${res.stdout || ""}\n${res.stderr || ""}`;
  return /\d+\.\d+(?:\.\d+)?/.exec(text)?.[0] ?? null;
}

// 仅接受 Python >= 3.12（scipy==1.18.1 与 lerobot==0.6.1 均要求 Python >= 3.12）
export function findPython312(pathEnv = process.env.PATH) {
  for (const name of ["python3.12", "python3", "python"]) {
    const bin = findOnPath(name, pathEnv);
    if (!bin) continue;
    const version = captureVersion(bin, ["--version"]);
    if (!version) continue;
    const [major, minor] = version.split(".").map(Number);
    if (major > 3 || (major === 3 && minor >= 12)) return { bin, version };
  }
  return null;
}

export function findCheckoutRoot(startDir = process.cwd()) {
  let current = path.resolve(startDir);
  while (true) {
    if (fs.existsSync(path.join(current, "pyproject.toml")) && fs.existsSync(path.join(current, "ni_rua", "apps", "gateway"))) {
      return current;
    }
    const parent = path.dirname(current);
    if (parent === current) return null;
    current = parent;
  }
}

export function findGatewayBinary({ cwd = process.cwd(), pathEnv = process.env.PATH } = {}) {
  const managedBin = path.join(gatewayEnvDir(), "bin", "ni-rua");
  const managedPy = path.join(gatewayEnvDir(), "bin", "python");
  if (fs.existsSync(managedBin)) {
    return { kind: "managed", bin: managedBin, argsPrefix: [], python: fs.existsSync(managedPy) ? managedPy : null };
  }
  const pathBin = findOnPath("ni-rua", pathEnv);
  if (pathBin) {
    const siblingPy = path.join(path.dirname(pathBin), "python");
    const siblingPy3 = path.join(path.dirname(pathBin), "python3");
    const py = fs.existsSync(siblingPy) ? siblingPy : (fs.existsSync(siblingPy3) ? siblingPy3 : null);
    return { kind: "path", bin: pathBin, argsPrefix: [], python: py };
  }
  const checkout = findCheckoutRoot(cwd);
  if (checkout) {
    const py = findPython312(pathEnv);
    if (py) {
      return { kind: "checkout", bin: py.bin, argsPrefix: ["-u", "-m", "ni_rua.apps.gateway"], python: py.bin, cwd: checkout };
    }
  }
  return null;
}

export function findSo101Python() {
  const managed = path.join(so101EnvDir(), "bin", "python");
  if (fs.existsSync(managed)) return managed;
  const miniforge = path.join(os.homedir(), "miniforge3", "envs", "so101", "bin", "python");
  if (fs.existsSync(miniforge)) return miniforge;
  return null;
}

export function findSo101DeploymentPath({ cwd = process.cwd() } = {}) {
  const candidates = [
    path.resolve(cwd, "configs", "so101.local.json"),
    path.join(niRuaConfigDir(), "so101.local.json"),
  ];
  return candidates.find((file) => fs.existsSync(file)) ?? null;
}

export function resolveSo101CalibrationPath({ cwd = process.cwd(), deploymentPath = null } = {}) {
  const depPath = deploymentPath ?? findSo101DeploymentPath({ cwd });
  const candidates = [];
  if (depPath && fs.existsSync(depPath)) {
    try {
      const parsed = JSON.parse(fs.readFileSync(depPath, "utf8"));
      if (typeof parsed?.camera_calibration_file === "string" && parsed.camera_calibration_file.trim()) {
        const raw = parsed.camera_calibration_file.trim();
        candidates.push(path.isAbsolute(raw) ? raw : path.resolve(path.dirname(depPath), raw));
      }
    } catch { /* 留给上层配置体检报告 JSON 语法错误 */ }
  }
  candidates.push(path.resolve(cwd, "configs", "camera_calibration.json"));
  candidates.push(path.join(niRuaConfigDir(), "camera_calibration.json"));
  for (const file of candidates) {
    if (fs.existsSync(file) && fs.statSync(file).isFile()) return file;
  }
  return null;
}

export function listSerialDevices() {
  if (process.env.NI_RUA_SERIAL_DEVICES !== undefined) {
    return process.env.NI_RUA_SERIAL_DEVICES.split(path.delimiter).filter(Boolean);
  }
  if (!fs.existsSync("/dev")) return [];
  try {
    return fs.readdirSync("/dev")
      .filter((name) => /^(?:tty\.usbmodem|ttyACM|ttyUSB)/.test(name))
      .sort()
      .map((name) => path.join("/dev", name));
  } catch {
    return [];
  }
}

export function checkLocalIkImports(pythonBin) {
  if (!pythonBin || !fs.existsSync(pythonBin)) return { ok: false, missing: [], error: "python not found" };
  const code = [
    "import importlib, json",
    "mods = ['pin', 'pink', 'coal', 'daqp', 'ruckig']",
    "missing = []",
    "for m in mods:",
    "    try: importlib.import_module(m)",
    "    except Exception: missing.append(m)",
    "print(json.dumps(missing))",
  ].join("\n");
  const res = spawnSync(pythonBin, ["-c", code], { encoding: "utf8", timeout: 15000 });
  if (res.error || res.status !== 0) {
    return { ok: false, missing: [], error: (res.stderr || res.error?.message || `exit ${res.status}`).trim() };
  }
  try {
    const missing = JSON.parse((res.stdout || "").trim());
    return { ok: Array.isArray(missing) && missing.length === 0, missing: Array.isArray(missing) ? missing : [], error: null };
  } catch (error) {
    return { ok: false, missing: [], error: error.message };
  }
}
