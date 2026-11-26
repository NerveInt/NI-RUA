// `ni-rua-plugin gateway`：启动零硬件干跑 MCP 网关或本地 NI-RUA Python 网关（含 SO-101 冷启动保护）
import { spawn } from "node:child_process";
import { startDryRunServer } from "./dry-run-server.mjs";
import {
  ESTOP_WARNING,
  findGatewayBinary,
  isSo101Preset,
  resolveSo101CalibrationPath,
} from "./runtime.mjs";

function hasObservationProfileArg(args) {
  return args.some((a) => a === "--observation-profile" || a.startsWith("--observation-profile="));
}

export async function runGateway({ opts, extraArgs = [], defaultMcpPort = 8790, version = "0.1.1", out, UserError }) {
  if (opts["dry-run"]) {
    const port = opts.port ?? defaultMcpPort;
    const srv = await startDryRunServer({ port, version });
    out(`NI-RUA dry-run MCP gateway listening on ${srv.url}`);
    out("Tools: run_begin, robot_observe, robot_action, run_finish (zero hardware, simulated in memory)");
    out("Open Claude Code / Codex / OpenCode in another terminal and ask it to run a dry-run check via ni_rua MCP tools.");
    out("Press Ctrl+C to stop.");
    await new Promise((resolve) => {
      const stop = () => { srv.close().finally(resolve); };
      process.once("SIGINT", stop);
      process.once("SIGTERM", stop);
    });
    return 0;
  }

  const found = findGatewayBinary();
  if (!found) {
    throw new UserError(
      "NI-RUA Python gateway (`ni-rua`) was not found.\n" +
      "Install it with:\n" +
      "  ni-rua-plugin setup claude   # or: codex | opencode\n" +
      "Or test the MCP loop right now without Python or hardware:\n" +
      "  ni-rua-plugin gateway --dry-run",
    );
  }

  const args = [...found.argsPrefix, "--agent-backend", "external"];
  if (opts.config) args.push("--config", opts.config);
  if (opts.port !== undefined) args.push("--port", String(opts.port));
  if (opts["run-dir"]) args.push("--run-dir", opts["run-dir"]);
  if (opts.prompt) args.push("--prompt", opts.prompt);
  args.push(...extraArgs);

  const webPort = opts.port ?? (isSo101Preset(opts.config) ? 8781 : 8767);
  if (isSo101Preset(opts.config) && !hasObservationProfileArg(args)) {
    const calib = resolveSo101CalibrationPath();
    if (!calib) {
      out(
        `[warn] No camera_calibration.json found for SO-101 cold start; automatically adding --observation-profile rgb_proprio_v1. ` +
        `Complete guided calibration at http://127.0.0.1:${webPort}/calibration`,
      );
      args.push("--observation-profile", "rgb_proprio_v1");
    }
  }

  out(`+ ${[found.bin, ...args].map((a) => (/\s/.test(a) ? JSON.stringify(a) : a)).join(" ")}`);
  out(`Web console: http://127.0.0.1:${webPort}/console`);
  out(ESTOP_WARNING);

  return await new Promise((resolve, reject) => {
    const child = spawn(found.bin, args, { cwd: found.cwd || process.cwd(), stdio: "inherit" });
    const forward = (sig) => { if (!child.killed) child.kill(sig); };
    process.once("SIGINT", forward);
    process.once("SIGTERM", forward);
    child.on("error", (error) => reject(new UserError(`failed to launch ${found.bin}: ${error.message}`)));
    child.on("close", (code, signal) => {
      process.removeListener("SIGINT", forward);
      process.removeListener("SIGTERM", forward);
      resolve(signal ? 128 : (code ?? 0));
    });
  });
}
