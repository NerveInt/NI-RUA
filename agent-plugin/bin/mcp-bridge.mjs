// Stdio-to-HTTP 懒启动 MCP 桥接器（`ni-rua-plugin mcp`）
// 职责：
//  1. 零开销响应 initialize / ping（不探测、不拉起任何子进程）
//  2. tools/list：快速探测上游 HTTP 网关（~300ms）；已运行则透传实时目录，未运行则返回打包静态目录
//  3. tools/call：上游可达则透传（保留 Mcp-Session-Id，支持 JSON 与 SSE）；未运行且允许自动启动时，
//     持单例锁拉起后台本地网关，发送 notifications/tools/list_changed，透传调用并在 run_begin 顶部注入安全横幅
import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import {
  ESTOP_WARNING,
  findGatewayBinary,
  findSo101DeploymentPath,
  gatewayLockPath,
  gatewayLogPath,
  gatewayRunsDir,
  isSo101Preset,
  resolveSo101CalibrationPath,
} from "./runtime.mjs";

const SUPPORTED_VERSIONS = ["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"];
const DEFAULT_PROTOCOL_VERSION = "2025-06-18";

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isConnRefused(error) {
  const cause = error?.cause;
  if (!cause) return false;
  if (cause.code === "ECONNREFUSED") return true;
  if (Array.isArray(cause.errors)) return cause.errors.some((e) => e?.code === "ECONNREFUSED");
  return false;
}

function pidAlive(pid) {
  if (!Number.isInteger(pid) || pid <= 0) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return error?.code === "EPERM";
  }
}

function parsePortFromUrl(urlStr, fallback) {
  try {
    const u = new URL(urlStr);
    if (u.port) return Number(u.port);
    return u.protocol === "https:" ? 443 : 80;
  } catch {
    return fallback;
  }
}

function makeFreshRunDir() {
  const runsRoot = gatewayRunsDir();
  fs.mkdirSync(runsRoot, { recursive: true });
  const stamp = new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d+Z$/, "Z");
  const runDir = path.join(runsRoot, `run-${stamp}-${process.pid}`);
  const latestLink = path.join(runsRoot, "latest");
  try {
    fs.rmSync(latestLink, { force: true });
    fs.symlinkSync(runDir, latestLink);
  } catch {
    // 软链接失败不影响网关正常启动
  }
  return runDir;
}

function readLogTail(logFile, startOffset, maxLines = 15) {
  try {
    const fd = fs.openSync(logFile, "r");
    try {
      const size = fs.fstatSync(fd).size;
      const from = Math.max(0, Math.min(startOffset, size));
      const len = Math.min(size - from, 16384);
      if (len <= 0) return "";
      const buf = Buffer.alloc(len);
      fs.readSync(fd, buf, 0, len, from);
      return buf.toString("utf8").trim().split(/\r?\n/).slice(-maxLines).join("\n");
    } finally {
      fs.closeSync(fd);
    }
  } catch {
    return "";
  }
}

function mcpToolError(message) {
  return { content: [{ type: "text", text: message }], isError: true };
}

function formatStartupFailure({ reason, logFile, tail }) {
  const lines = [`NI-RUA local gateway failed to start: ${reason}`];
  if (/so101\.local\.json/i.test(tail)) {
    lines.push("Fix: run `ni-rua-so101 init-config` to detect your SO-101 USB serial port and cameras.");
  }
  if (tail) {
    lines.push("", `Recent gateway log (${logFile}):`, tail);
  } else if (logFile) {
    lines.push(`Log file: ${logFile}`);
  }
  return lines.join("\n");
}

function resolveGatewayLaunch({ gatewayConfig, webPort, mcpPort }) {
  const customCmd = process.env.NI_RUA_GATEWAY_CMD?.trim();
  if (customCmd) {
    if (customCmd.startsWith("[")) {
      const argv = JSON.parse(customCmd);
      if (!Array.isArray(argv) || !argv.length) {
        throw new Error("NI_RUA_GATEWAY_CMD JSON array must be non-empty");
      }
      return { bin: String(argv[0]), args: argv.slice(1).map(String), cwd: process.cwd() };
    }
    return {
      bin: customCmd,
      args: [
        "--agent-backend", "external",
        "--config", gatewayConfig,
        "--port", String(webPort),
        "--external-mcp-port", String(mcpPort),
        "--run-dir", makeFreshRunDir(),
      ],
      cwd: process.cwd(),
    };
  }

  const found = findGatewayBinary();
  if (!found) {
    throw new Error(
      "NI-RUA gateway binary (`ni-rua`) was not found.\n" +
      "Install it with:\n" +
      "  npx -y @nerveint/ni-rua-plugin setup codex   # or: claude | opencode\n" +
      "Or start a zero-hardware test gateway manually:\n" +
      "  npx -y @nerveint/ni-rua-plugin gateway --dry-run",
    );
  }

  if (isSo101Preset(gatewayConfig) && !findSo101DeploymentPath()) {
    throw new Error(
      "SO-101 deployment config (`so101.local.json`) was not found in ./configs/ or ~/.config/ni-rua/.\n" +
      "Run one-time hardware detection first:\n" +
      "  ni-rua-so101 init-config\n" +
      "Or start a zero-hardware test gateway manually:\n" +
      "  npx -y @nerveint/ni-rua-plugin gateway --dry-run",
    );
  }

  const args = [
    ...found.argsPrefix,
    "--agent-backend", "external",
    "--config", gatewayConfig,
    "--port", String(webPort),
    "--external-mcp-port", String(mcpPort),
    "--run-dir", makeFreshRunDir(),
  ];
  if (isSo101Preset(gatewayConfig) && !resolveSo101CalibrationPath()) {
    args.push("--observation-profile", "rgb_proprio_v1");
  }
  return { bin: found.bin, args, cwd: found.cwd || process.cwd() };
}

function tryAcquireLock(lockFile) {
  fs.mkdirSync(path.dirname(lockFile), { recursive: true });
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const fd = fs.openSync(lockFile, "wx");
      fs.writeFileSync(fd, JSON.stringify({ pid: process.pid, startedAt: Date.now() }));
      fs.closeSync(fd);
      return true;
    } catch (error) {
      if (error?.code !== "EEXIST") throw error;
      try {
        const raw = JSON.parse(fs.readFileSync(lockFile, "utf8"));
        if (!pidAlive(raw?.pid)) {
          fs.rmSync(lockFile, { force: true });
          continue;
        }
      } catch {
        fs.rmSync(lockFile, { force: true });
        continue;
      }
      return false;
    }
  }
  return false;
}

export function createMcpBridge({ manifest, pkg, opts = {}, writeMessage }) {
  const defaultMcpUrl = manifest.mcp_url || `http://127.0.0.1:${manifest.default_port || 8790}/mcp`;
  const mcpUrl = opts["gateway-url"]
    || (opts.port !== undefined ? `http://127.0.0.1:${opts.port}/mcp` : null)
    || process.env.NI_RUA_MCP_URL
    || defaultMcpUrl;
  const mcpPort = parsePortFromUrl(mcpUrl, manifest.default_port || 8790);
  const webPort = Number(process.env.NI_RUA_WEB_PORT || manifest.web_port || 8781);
  const webConsoleUrl = `http://127.0.0.1:${webPort}/console`;
  const calibrationUrl = `http://127.0.0.1:${webPort}/calibration`;
  const autoStart = process.env.NI_RUA_AUTO_START !== "0";
  const gatewayConfig = (process.env.NI_RUA_GATEWAY_CONFIG || "so101").trim();
  const probeTimeoutMs = Number(process.env.NI_RUA_MCP_PROBE_MS || 300);
  const startTimeoutMs = Number(process.env.NI_RUA_GATEWAY_START_TIMEOUT_MS || 30000);
  const staticTools = Array.isArray(manifest.tool_definitions)
    ? manifest.tool_definitions
    : (manifest.tool_names || []).map((name) => ({
      name,
      description: `NI-RUA tool ${name}`,
      inputSchema: { type: "object", properties: {} },
    }));

  let protocolVersion = DEFAULT_PROTOCOL_VERSION;
  let clientInfo = { name: "ni-rua-mcp-bridge", version: pkg.version };
  let upstreamSessionId = null;
  let upstreamReady = false;
  let nextUpstreamId = 1;
  let startingPromise = null;

  function resetUpstream() {
    upstreamReady = false;
    upstreamSessionId = null;
  }

  async function postUpstream(body, { timeoutMs = null, onNotification = null } = {}) {
    const headers = {
      "content-type": "application/json",
      accept: "application/json, text/event-stream",
      "mcp-protocol-version": protocolVersion,
    };
    if (upstreamSessionId) headers["mcp-session-id"] = upstreamSessionId;

    const init = { method: "POST", headers, body: JSON.stringify(body) };
    if (timeoutMs !== null) init.signal = AbortSignal.timeout(timeoutMs);

    const response = await fetch(mcpUrl, init);
    const sid = response.headers.get("mcp-session-id");
    if (sid) upstreamSessionId = sid;

    if (!response.ok && response.status !== 202) {
      const errText = await response.text().catch(() => "");
      const err = new Error(`upstream HTTP ${response.status}${errText ? `: ${errText.slice(0, 200)}` : ""}`);
      err.httpStatus = response.status;
      throw err;
    }
    if (response.status === 202) return null;

    const contentType = response.headers.get("content-type") || "";
    if (!contentType.includes("event-stream")) {
      const text = await response.text();
      return text.trim() ? JSON.parse(text) : null;
    }

    // 流式解析 SSE：透传实时进度通知，并提取匹配请求 id 的最终 JSON-RPC 响应
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let matched = null;
    let firstMsg = null;

    const handleBlock = (block) => {
      const data = block
        .split(/\r?\n/)
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (!data) return;
      let msg;
      try { msg = JSON.parse(data); } catch { return; }
      if (msg.id === undefined && msg.method && onNotification) {
        onNotification(msg);
      } else {
        if (!firstMsg) firstMsg = msg;
        if (body.id !== undefined && msg.id === body.id) matched = msg;
      }
    };

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split(/\r?\n\r?\n/);
      buffer = parts.pop() || "";
      for (const block of parts) handleBlock(block);
    }
    buffer += decoder.decode();
    if (buffer.trim()) handleBlock(buffer);
    return matched ?? firstMsg;
  }

  async function initUpstreamSession(timeoutMs) {
    const initResp = await postUpstream(
      {
        jsonrpc: "2.0",
        id: nextUpstreamId++,
        method: "initialize",
        params: {
          protocolVersion,
          capabilities: {},
          clientInfo,
        },
      },
      { timeoutMs },
    );
    if (!initResp?.result) {
      throw new Error(initResp?.error?.message || "upstream initialize returned no result");
    }
    await postUpstream({ jsonrpc: "2.0", method: "notifications/initialized" }, { timeoutMs }).catch(() => null);
    upstreamReady = true;
    return initResp.result;
  }

  async function probeUpstream(timeoutMs = probeTimeoutMs) {
    try {
      await initUpstreamSession(timeoutMs);
      return true;
    } catch {
      resetUpstream();
      return false;
    }
  }

  async function waitForGatewayReady({ child, exitState, deadline, logFile, logStartOffset }) {
    while (Date.now() < deadline) {
      if (exitState.error) {
        throw new Error(formatStartupFailure({ reason: exitState.error.message, logFile, tail: readLogTail(logFile, logStartOffset) }));
      }
      if (exitState.exited) {
        const detail = exitState.signal ? `signal ${exitState.signal}` : `exit code ${exitState.code}`;
        throw new Error(formatStartupFailure({ reason: `gateway process exited early (${detail})`, logFile, tail: readLogTail(logFile, logStartOffset) }));
      }
      if (await probeUpstream(Math.min(500, Math.max(100, deadline - Date.now())))) {
        return;
      }
      await sleep(150);
    }
    throw new Error(
      formatStartupFailure({
        reason: `timed out after ${Math.round(startTimeoutMs / 1000)}s waiting for ${mcpUrl}`,
        logFile,
        tail: readLogTail(logFile, logStartOffset),
      }),
    );
  }

  async function startLocalGatewaySingleton() {
    const lockFile = gatewayLockPath();
    const deadline = Date.now() + startTimeoutMs;

    if (!tryAcquireLock(lockFile)) {
      // 另一个桥接实例正在拉起网关：轮询等待网关就绪或锁释放
      while (Date.now() < deadline) {
        if (await probeUpstream(Math.min(500, Math.max(100, deadline - Date.now())))) return false;
        if (!fs.existsSync(lockFile)) {
          if (tryAcquireLock(lockFile)) break;
        }
        await sleep(150);
      }
      if (upstreamReady) return false;
      if (!fs.existsSync(lockFile) && !tryAcquireLock(lockFile)) {
        throw new Error(`Timed out waiting for peer gateway startup lock (${lockFile})`);
      }
    }

    try {
      // 持锁后二次探测，避免与刚启动完的网关竞争
      if (await probeUpstream(probeTimeoutMs)) return false;

      const launch = resolveGatewayLaunch({ gatewayConfig, webPort, mcpPort });
      const logFile = gatewayLogPath();
      fs.mkdirSync(path.dirname(logFile), { recursive: true });
      const logStartOffset = fs.existsSync(logFile) ? fs.statSync(logFile).size : 0;
      const logFd = fs.openSync(logFile, "a");
      fs.writeSync(logFd, `\n--- NI-RUA auto-start ${new Date().toISOString()}: ${[launch.bin, ...launch.args].join(" ")} ---\n`);

      const exitState = { exited: false, code: null, signal: null, error: null };
      let child;
      try {
        child = spawn(launch.bin, launch.args, {
          cwd: launch.cwd,
          detached: true,
          stdio: ["ignore", logFd, logFd],
        });
      } finally {
        fs.closeSync(logFd);
      }
      child.once("error", (error) => { exitState.error = error; });
      child.once("exit", (code, signal) => { exitState.exited = true; exitState.code = code; exitState.signal = signal; });
      child.unref();

      await waitForGatewayReady({ child, exitState, deadline, logFile, logStartOffset });
      return true;
    } finally {
      try { fs.rmSync(lockFile, { force: true }); } catch { /* 忽略清理竞态 */ }
    }
  }

  async function ensureUpstreamForCall() {
    if (upstreamReady) return;
    if (await probeUpstream(probeTimeoutMs)) return;
    if (!autoStart) {
      throw new Error(
        `NI-RUA gateway is not reachable at ${mcpUrl} and NI_RUA_AUTO_START=0 disables auto-start.\n` +
        "Start a gateway manually:\n" +
        "  ni-rua-plugin gateway --config so101   # or --dry-run",
      );
    }
    if (!startingPromise) {
      startingPromise = startLocalGatewaySingleton().finally(() => { startingPromise = null; });
    }
    const startedByUs = await startingPromise;
    if (startedByUs && writeMessage) {
      writeMessage({ jsonrpc: "2.0", method: "notifications/tools/list_changed" });
    }
  }

  function buildSafetyBanner() {
    const parts = [
      `[NI-RUA Operator Console] Live Web console (Stop / Hold / E-STOP): ${webConsoleUrl}`,
      ESTOP_WARNING,
    ];
    if (isSo101Preset(gatewayConfig) && !resolveSo101CalibrationPath()) {
      parts.push(`[Calibration] No camera_calibration.json detected — complete guided calibration at ${calibrationUrl}`);
    }
    return parts.join("\n");
  }

  async function handleToolsCall(message) {
    const toolName = message.params?.name;
    try {
      await ensureUpstreamForCall();
    } catch (error) {
      return { jsonrpc: "2.0", id: message.id, result: mcpToolError(error.message) };
    }

    const forwardCall = () => postUpstream(
      { jsonrpc: "2.0", id: nextUpstreamId++, method: "tools/call", params: message.params },
      { onNotification: (notif) => writeMessage?.(notif) },
    );

    let upstreamResp;
    try {
      upstreamResp = await forwardCall();
    } catch (error) {
      // 仅在连接被拒（请求未送达旧网关）时重置会话并尝试重连/拉起一次，严禁对已送达的物理动作静默重试
      if (isConnRefused(error)) {
        resetUpstream();
        try {
          await ensureUpstreamForCall();
          upstreamResp = await forwardCall();
        } catch (retryError) {
          return { jsonrpc: "2.0", id: message.id, result: mcpToolError(retryError.message) };
        }
      } else {
        return { jsonrpc: "2.0", id: message.id, result: mcpToolError(`NI-RUA gateway call failed: ${error.message}`) };
      }
    }

    if (upstreamResp?.error) {
      return { jsonrpc: "2.0", id: message.id, error: upstreamResp.error };
    }
    const result = upstreamResp?.result;
    if (toolName === "run_begin" && result && !result.isError && Array.isArray(result.content)) {
      return {
        jsonrpc: "2.0",
        id: message.id,
        result: {
          ...result,
          content: [{ type: "text", text: buildSafetyBanner() }, ...result.content],
        },
      };
    }
    return { jsonrpc: "2.0", id: message.id, result: result ?? { content: [] } };
  }

  async function handleMessage(message) {
    if (!message || typeof message !== "object" || typeof message.method !== "string") {
      if (message?.id !== undefined) {
        return { jsonrpc: "2.0", id: message.id, error: { code: -32600, message: "Invalid Request" } };
      }
      return null;
    }

    // 通知类消息无需响应
    if (message.id === undefined) {
      if (message.method === "notifications/cancelled" && upstreamReady) {
        postUpstream(message, { timeoutMs: 2000 }).catch(() => null);
      }
      return null;
    }

    if (message.method === "initialize") {
      const requested = message.params?.protocolVersion;
      if (typeof requested === "string" && SUPPORTED_VERSIONS.includes(requested)) {
        protocolVersion = requested;
      }
      if (message.params?.clientInfo && typeof message.params.clientInfo === "object") {
        clientInfo = message.params.clientInfo;
      }
      return {
        jsonrpc: "2.0",
        id: message.id,
        result: {
          protocolVersion,
          capabilities: { tools: { listChanged: true } },
          serverInfo: { name: manifest.server_name || "ni_rua", version: pkg.version },
          instructions:
            "NI-RUA robot control MCP server. Call `run_begin` first before any other NI-RUA tool; " +
            "the local gateway auto-starts on first call if not already running. " +
            `Live Web console (Stop/Hold/E-STOP): ${webConsoleUrl}. ${ESTOP_WARNING}`,
        },
      };
    }

    if (message.method === "ping") {
      return { jsonrpc: "2.0", id: message.id, result: {} };
    }

    if (message.method === "tools/list") {
      try {
        if (!upstreamReady) {
          await initUpstreamSession(probeTimeoutMs);
        }
        const live = await postUpstream(
          { jsonrpc: "2.0", id: nextUpstreamId++, method: "tools/list", params: message.params || {} },
          { timeoutMs: Math.max(probeTimeoutMs, 1000) },
        );
        if (live?.result && Array.isArray(live.result.tools)) {
          return { jsonrpc: "2.0", id: message.id, result: live.result };
        }
      } catch {
        resetUpstream();
      }
      return { jsonrpc: "2.0", id: message.id, result: { tools: staticTools } };
    }

    if (message.method === "tools/call") {
      return handleToolsCall(message);
    }

    return {
      jsonrpc: "2.0",
      id: message.id,
      error: { code: -32601, message: `Method not found: ${message.method}` },
    };
  }

  return { handleMessage, mcpUrl, webConsoleUrl };
}

export async function runMcpBridge({ opts = {}, manifest, pkg, stdin = process.stdin, stdout = process.stdout }) {
  const writeMessage = (msg) => {
    if (!stdout.writableEnded) stdout.write(`${JSON.stringify(msg)}\n`);
  };
  const bridge = createMcpBridge({ manifest, pkg, opts, writeMessage });

  return await new Promise((resolve) => {
    let buffer = "";
    const inFlight = new Set();
    let ended = false;

    const finishIfDone = () => {
      if (ended && inFlight.size === 0) resolve(0);
    };

    const dispatchLine = (line) => {
      const trimmed = line.trim();
      if (!trimmed) return;
      let msg;
      try {
        msg = JSON.parse(trimmed);
      } catch {
        writeMessage({ jsonrpc: "2.0", id: null, error: { code: -32700, message: "Parse error" } });
        return;
      }
      const task = bridge.handleMessage(msg)
        .then((reply) => { if (reply) writeMessage(reply); })
        .catch((error) => {
          if (msg?.id !== undefined) {
            writeMessage({ jsonrpc: "2.0", id: msg.id, error: { code: -32603, message: error.message } });
          }
        })
        .finally(() => {
          inFlight.delete(task);
          finishIfDone();
        });
      inFlight.add(task);
    };

    stdin.setEncoding("utf8");
    stdin.on("data", (chunk) => {
      buffer += chunk;
      const lines = buffer.split(/\r?\n/);
      buffer = lines.pop() || "";
      for (const line of lines) dispatchLine(line);
    });
    stdin.on("end", () => {
      if (buffer.trim()) dispatchLine(buffer);
      ended = true;
      finishIfDone();
    });
    stdin.on("error", () => {
      ended = true;
      finishIfDone();
    });
  });
}
