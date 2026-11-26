// `ni-rua-plugin setup`：一键配置宿主插件 + 隔离 Python 3.12 网关环境（及可选 SO-101 硬件环境）
import fs from "node:fs";
import path from "node:path";
import {
  DEFAULT_SO101_DEPS,
  ESTOP_WARNING,
  findGatewayBinary,
  findOnPath,
  findPython312,
  gatewayEnvDir,
  so101EnvDir,
} from "./runtime.mjs";

function ensureIsolatedVenv({ label, dir, packages, force, out, run, UserError }) {
  const uvBin = findOnPath("uv");
  const py312 = uvBin ? null : findPython312();
  if (!uvBin && !py312) {
    throw new UserError(
      `Cannot bootstrap ${label}: neither \`uv\` nor Python >= 3.12 was found on PATH.\n` +
      "Install uv in one line and re-run setup:\n" +
      "  curl -LsSf https://astral.sh/uv/install.sh | sh",
    );
  }
  const pyPath = path.join(dir, "bin", "python");
  if (force && fs.existsSync(dir)) {
    out(`removing existing ${label} env (${dir}) due to --force`);
    fs.rmSync(dir, { recursive: true, force: true });
  }
  fs.mkdirSync(path.dirname(dir), { recursive: true });
  if (!fs.existsSync(pyPath)) {
    if (uvBin) {
      run(uvBin, ["venv", "--python", "3.12", dir]);
    } else {
      run(py312.bin, ["-m", "venv", dir]);
    }
  }
  if (uvBin) {
    run(uvBin, ["pip", "install", "--python", pyPath, ...packages]);
  } else {
    const pipPath = path.join(dir, "bin", "pip");
    run(pipPath, ["install", ...packages]);
  }
  return pyPath;
}

export function runSetup({ target, opts, hosts, findHostExecutable, installHost, out, run, UserError }) {
  const targets = target === "all"
    ? hosts.filter((h) => Boolean(findHostExecutable(h, opts)))
    : [target];

  if (!targets.length) {
    throw new UserError(`No supported host CLI (${hosts.join(", ")}) was found on PATH; install one first or pass --claude-bin / --codex-bin / --opencode-bin.`);
  }

  for (const host of targets) {
    out(`== Installing plugin for ${host}`);
    installHost(host, opts);
  }

  if (opts["skip-gateway"]) {
    out("== Skipping Python gateway setup (--skip-gateway)");
  } else {
    const existing = findGatewayBinary();
    const managedBin = path.join(gatewayEnvDir(), "bin", "ni-rua");
    const spec = (process.env.NI_RUA_HARNESS_SPEC || "ni-rua-harness").trim();
    if (!opts.force && existing && (existing.kind === "managed" || existing.kind === "path")) {
      out(`== Gateway runtime already available (${existing.kind}): ${existing.bin}`);
    } else {
      out(`== Setting up isolated Python 3.12 gateway runtime at ${gatewayEnvDir()}`);
      ensureIsolatedVenv({
        label: "NI-RUA gateway",
        dir: gatewayEnvDir(),
        packages: [spec],
        force: Boolean(opts.force),
        out,
        run,
        UserError,
      });
      out(`gateway runtime ready: ${managedBin}`);
    }
  }

  // 默认包含 SO-101 硬件运行环境；--no-so101 可跳过，--so101 保留为无害兼容别名
  const includeSo101 = !opts["no-so101"] && (!opts["skip-gateway"] || Boolean(opts.so101));
  if (includeSo101) {
    const so101Py = path.join(so101EnvDir(), "bin", "python");
    const deps = process.env.NI_RUA_SO101_DEPS
      ? process.env.NI_RUA_SO101_DEPS.trim().split(/\s+/).filter(Boolean)
      : DEFAULT_SO101_DEPS;
    if (!opts.force && fs.existsSync(so101Py)) {
      out(`== SO-101 hardware env already exists: ${so101Py} (pass --force to recreate)`);
    } else {
      out(`== Setting up isolated Python 3.12 SO-101 hardware runtime at ${so101EnvDir()}`);
      ensureIsolatedVenv({
        label: "SO-101 hardware worker",
        dir: so101EnvDir(),
        packages: deps,
        force: Boolean(opts.force),
        out,
        run,
        UserError,
      });
      out(`SO-101 hardware runtime ready: ${so101Py}`);
    }
  } else if (opts["no-so101"]) {
    out("== Skipping SO-101 hardware env (--no-so101)");
  }

  const hostHint = targets.join(" / ");
  out("");
  out("Next steps:");
  if (includeSo101) {
    out("  1. One-time SO-101 config: ni-rua-so101 init-config   # detect USB serial & cameras");
    out(`  2. Launch ${hostHint} directly and ask it to control your SO-101 arm`);
    out("     (the ni_rua MCP stdio bridge auto-starts the local gateway on first `run_begin`)");
    out("  3. Optional diagnostics:   ni-rua-plugin doctor --so101");
  } else {
    out(`  1. Launch ${hostHint} directly, or run a zero-hardware test: ni-rua-plugin gateway --dry-run`);
    out("  2. Verify setup:           ni-rua-plugin doctor");
  }
  out("  Web console (when local gateway runs): http://127.0.0.1:8781/console");
  out(ESTOP_WARNING);
}
