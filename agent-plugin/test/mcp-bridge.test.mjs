// 覆盖 Stdio-to-HTTP 懒启动 MCP 桥接器（`ni-rua-plugin mcp`）全链路契约测试
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import fs from "node:fs";
import http from "node:http";
import net from "node:net";
import path from "node:path";
import { test } from "node:test";
import { CLI, PACKAGE_ROOT, sandbox } from "./helpers.mjs";

function freePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.once("error", reject);
    srv.listen(0, "127.0.0.1", () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
  });
}

// 启动真实 `node bin/cli.mjs mcp` 子进程并封装行级 JSON-RPC 请求/通知收发
function spawnBridge({ args = ["mcp"], env = {}, cwd = PACKAGE_ROOT } = {}) {
  const child = spawn(process.execPath, [CLI, ...args], {
    cwd,
    env: { ...process.env, ...env },
    stdio: ["pipe", "pipe", "pipe"],
  });
  let stdoutBuf = "";
  let stderr = "";
  const notifications = [];
  const pending = new Map();

  child.stderr.on("data", (chunk) => { stderr += chunk; });
  child.stdout.on("data", (chunk) => {
    stdoutBuf += chunk.toString("utf8");
    const lines = stdoutBuf.split(/\r?\n/);
    stdoutBuf = lines.pop() || "";
    for (const line of lines) {
      if (!line.trim()) continue;
      const msg = JSON.parse(line);
      if (msg.id === undefined) {
        notifications.push(msg);
      } else if (pending.has(msg.id)) {
        const { resolve } = pending.get(msg.id);
        pending.delete(msg.id);
        resolve(msg);
      }
    }
  });

  let nextId = 1;
  return {
    child,
    notifications,
    get stderr() { return stderr; },
    sendNotification(method, params) {
      child.stdin.write(`${JSON.stringify({ jsonrpc: "2.0", method, ...(params ? { params } : {}) })}\n`);
    },
    request(method, params = {}, timeoutMs = 10000) {
      const id = nextId++;
      return new Promise((resolve, reject) => {
        const timer = setTimeout(() => {
          pending.delete(id);
          reject(new Error(`RPC timeout for ${method} (stderr: ${stderr})`));
        }, timeoutMs);
        pending.set(id, {
          resolve: (msg) => { clearTimeout(timer); resolve(msg); },
        });
        child.stdin.write(`${JSON.stringify({ jsonrpc: "2.0", id, method, params })}\n`);
      });
    },
    close() {
      return new Promise((resolve) => {
        if (child.exitCode !== null) return resolve(child.exitCode);
        child.once("close", (code) => resolve(code));
        child.stdin.end();
      });
    },
  };
}

test("mcp bridge offline: initialize/ping immediate, tools/list returns bundled static schemas without spawning", async () => {
  const box = sandbox();
  const closedPort = await freePort();
  const bridge = spawnBridge({
    env: {
      ...box.env,
      NI_RUA_MCP_URL: `http://127.0.0.1:${closedPort}/mcp`,
      NI_RUA_MCP_PROBE_MS: "150",
    },
  });
  try {
    const pkg = JSON.parse(fs.readFileSync(path.join(PACKAGE_ROOT, "package.json"), "utf8"));
    const manifest = JSON.parse(fs.readFileSync(path.join(PACKAGE_ROOT, "manifest.generated.json"), "utf8"));

    const init = await bridge.request("initialize", {
      protocolVersion: "2025-11-25",
      capabilities: {},
      clientInfo: { name: "bridge-test", version: "1" },
    });
    assert.equal(init.result.serverInfo.name, "ni_rua");
    assert.equal(init.result.serverInfo.version, pkg.version);
    assert.equal(init.result.capabilities.tools.listChanged, true);
    assert.equal(init.result.protocolVersion, "2025-11-25");

    bridge.sendNotification("notifications/initialized");
    const ping = await bridge.request("ping");
    assert.deepEqual(ping.result, {});

    const listed = await bridge.request("tools/list");
    assert.equal(listed.error, undefined);
    const names = listed.result.tools.map((t) => t.name);
    assert.deepEqual(names, manifest.tool_names);
    for (const tool of listed.result.tools) {
      assert.equal(tool.inputSchema.type, "object", tool.name);
    }
    assert.ok(!fs.existsSync(path.join(box.home, ".local", "share", "ni-rua", "gateway.lock")), "tools/list must never spawn gateway");
  } finally {
    await bridge.close();
    box.cleanup();
  }
});

test("mcp bridge transparent HTTP proxy: preserves Mcp-Session-Id, streams SSE progress, and prepends safety banner on run_begin", async () => {
  const box = sandbox();
  const seenSessionHeaders = [];
  const server = http.createServer((req, res) => {
    if (req.method !== "POST" || req.url !== "/mcp") { res.writeHead(404).end(); return; }
    let body = "";
    req.on("data", (c) => { body += c; });
    req.on("end", () => {
      const msg = JSON.parse(body);
      seenSessionHeaders.push({ method: msg.method, sid: req.headers["mcp-session-id"] || null });
      if (msg.id === undefined) { res.writeHead(202).end(); return; }
      if (msg.method === "initialize") {
        res.writeHead(200, { "content-type": "application/json", "mcp-session-id": "sess-xyz-42" });
        res.end(JSON.stringify({
          jsonrpc: "2.0",
          id: msg.id,
          result: { protocolVersion: "2025-06-18", capabilities: { tools: {} }, serverInfo: { name: "live-gw", version: "1" } },
        }));
        return;
      }
      if (msg.method === "tools/list") {
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify({
          jsonrpc: "2.0",
          id: msg.id,
          result: {
            tools: [
              { name: "run_begin", description: "live run_begin", inputSchema: { type: "object", properties: {} } },
              { name: "robot_observe", description: "live observe", inputSchema: { type: "object", properties: {} } },
            ],
          },
        }));
        return;
      }
      if (msg.method === "tools/call") {
        res.writeHead(200, { "content-type": "text/event-stream" });
        res.write(`event: message\ndata: ${JSON.stringify({ jsonrpc: "2.0", method: "notifications/progress", params: { progress: 1, total: 2 } })}\n\n`);
        const payload = msg.params.name === "run_begin"
          ? { content: [{ type: "text", text: "LIVE_BRIEFING" }], isError: false }
          : { content: [{ type: "text", text: "LIVE_OBS" }], isError: false };
        res.end(`event: message\ndata: ${JSON.stringify({ jsonrpc: "2.0", id: msg.id, result: payload })}\n\n`);
        return;
      }
      res.writeHead(404).end();
    });
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const port = server.address().port;

  const bridge = spawnBridge({
    env: {
      ...box.env,
      NI_RUA_MCP_URL: `http://127.0.0.1:${port}/mcp`,
      NI_RUA_WEB_PORT: "8781",
      NI_RUA_AUTO_START: "0",
    },
  });
  try {
    await bridge.request("initialize", { protocolVersion: "2025-06-18", capabilities: {}, clientInfo: { name: "test", version: "1" } });

    const listed = await bridge.request("tools/list");
    assert.deepEqual(listed.result.tools.map((t) => t.name), ["run_begin", "robot_observe"], "live gateway schema proxied");

    const begin = await bridge.request("tools/call", { name: "run_begin", arguments: {} });
    assert.equal(begin.result.isError, false);
    assert.equal(begin.result.content.length, 2);
    assert.match(begin.result.content[0].text, /http:\/\/127\.0\.0\.1:8781\/console/);
    assert.match(begin.result.content[0].text, /http:\/\/127\.0\.0\.1:8781\/calibration/);
    assert.match(begin.result.content[0].text, /SAFETY: Pressing Esc/);
    assert.equal(begin.result.content[1].text, "LIVE_BRIEFING");

    const obs = await bridge.request("tools/call", { name: "robot_observe", arguments: {} });
    assert.deepEqual(obs.result.content, [{ type: "text", text: "LIVE_OBS" }]);

    // SSE 进度通知已实时透传
    assert.ok(bridge.notifications.some((n) => n.method === "notifications/progress"));
    // Mcp-Session-Id 在 initialize 之后的所有请求中均保持透传
    const callHeaders = seenSessionHeaders.filter((h) => h.method === "tools/call");
    assert.ok(callHeaders.length >= 2 && callHeaders.every((h) => h.sid === "sess-xyz-42"));
  } finally {
    await bridge.close();
    await new Promise((done) => server.close(done));
    box.cleanup();
  }
});

test("mcp bridge lazy singleton auto-start spawns mock gateway once, emits list_changed, and logs to gateway.log", async () => {
  const box = sandbox();
  const port = await freePort();
  const pidFile = path.join(box.root, "mock-gw.pid");
  const countFile = path.join(box.root, "mock-gw.count");
  const mockGwScript = path.join(box.root, "mock-gw.mjs");

  // 轻量模拟本地网关：记录启动次数与 PID，并监听指定 MCP 端口
  fs.writeFileSync(
    mockGwScript,
    `import fs from "node:fs";
import http from "node:http";
const port = Number(process.argv[2]);
const pidFile = process.argv[3];
const countFile = process.argv[4];
fs.writeFileSync(pidFile, String(process.pid));
fs.appendFileSync(countFile, "start\\n");
const tools = [
  { name: "run_begin", description: "mock live begin", inputSchema: { type: "object", properties: {} } },
  { name: "robot_observe", description: "mock live observe", inputSchema: { type: "object", properties: {} } },
];
http.createServer((req, res) => {
  if (req.method !== "POST" || req.url !== "/mcp") { res.writeHead(404).end(); return; }
  let body = "";
  req.on("data", (c) => { body += c; });
  req.on("end", () => {
    const msg = JSON.parse(body || "{}");
    if (msg.id === undefined) { res.writeHead(202).end(); return; }
    let result = {};
    if (msg.method === "initialize") {
      result = { protocolVersion: "2025-06-18", capabilities: { tools: {} }, serverInfo: { name: "mock-lazy-gw", version: "1" } };
    } else if (msg.method === "tools/list") {
      result = { tools };
    } else if (msg.method === "tools/call") {
      result = { content: [{ type: "text", text: "AUTO_STARTED_OK:" + msg.params.name }], isError: false };
    }
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({ jsonrpc: "2.0", id: msg.id, result }));
  });
}).listen(port, "127.0.0.1", () => console.log("mock gateway listening on", port));
`,
  );

  const env = {
    ...box.env,
    NI_RUA_MCP_URL: `http://127.0.0.1:${port}/mcp`,
    NI_RUA_MCP_PROBE_MS: "150",
    NI_RUA_GATEWAY_START_TIMEOUT_MS: "8000",
    NI_RUA_GATEWAY_CMD: JSON.stringify([process.execPath, mockGwScript, String(port), pidFile, countFile]),
  };
  const bridgeA = spawnBridge({ env });
  const bridgeB = spawnBridge({ env });
  try {
    await Promise.all([
      bridgeA.request("initialize", { protocolVersion: "2025-06-18", capabilities: {}, clientInfo: { name: "a", version: "1" } }),
      bridgeB.request("initialize", { protocolVersion: "2025-06-18", capabilities: {}, clientInfo: { name: "b", version: "1" } }),
    ]);

    // 两个桥接器并发首次 tools/call 仅能拉起一个网关实例
    const [resA, resB] = await Promise.all([
      bridgeA.request("tools/call", { name: "run_begin", arguments: {} }),
      bridgeB.request("tools/call", { name: "run_begin", arguments: {} }),
    ]);
    assert.equal(resA.result.isError, false);
    assert.equal(resB.result.isError, false);
    assert.equal(resA.result.content[1].text, "AUTO_STARTED_OK:run_begin");
    assert.equal(resB.result.content[1].text, "AUTO_STARTED_OK:run_begin");

    const starts = fs.readFileSync(countFile, "utf8").trim().split("\n");
    assert.equal(starts.length, 1, "singleton lock must ensure only one gateway is spawned");

    const allNotifs = [...bridgeA.notifications, ...bridgeB.notifications];
    assert.ok(allNotifs.some((n) => n.method === "notifications/tools/list_changed"), "list_changed emitted after auto-start");

    const listed = await bridgeA.request("tools/list");
    assert.deepEqual(listed.result.tools.map((t) => t.name), ["run_begin", "robot_observe"]);

    const logFile = path.join(box.home, ".local", "share", "ni-rua", "logs", "gateway.log");
    assert.ok(fs.existsSync(logFile));
    assert.match(fs.readFileSync(logFile, "utf8"), /mock gateway listening on/);
  } finally {
    await Promise.all([bridgeA.close(), bridgeB.close()]);
    if (fs.existsSync(pidFile)) {
      try { process.kill(Number(fs.readFileSync(pidFile, "utf8")), "SIGTERM"); } catch { /* 已退出 */ }
    }
    box.cleanup();
  }
});

test("mcp bridge returns actionable errors on missing runtime, missing so101.local.json, early exit, and NI_RUA_AUTO_START=0", async () => {
  const box = sandbox();
  const closedPort = await freePort();
  const baseEnv = {
    ...box.env,
    PATH: box.bin,
    NI_RUA_MCP_URL: `http://127.0.0.1:${closedPort}/mcp`,
    NI_RUA_MCP_PROBE_MS: "150",
    NI_RUA_GATEWAY_START_TIMEOUT_MS: "5000",
  };
  try {
    // 1) NI_RUA_AUTO_START=0
    const bDisabled = spawnBridge({ env: { ...baseEnv, NI_RUA_AUTO_START: "0" }, cwd: box.root });
    await bDisabled.request("initialize");
    const rDisabled = await bDisabled.request("tools/call", { name: "run_begin", arguments: {} });
    assert.equal(rDisabled.result.isError, true);
    assert.match(rDisabled.result.content[0].text, /NI_RUA_AUTO_START=0/);
    await bDisabled.close();

    // 2) 缺少 ni-rua 可执行文件
    const bNoBin = spawnBridge({ env: baseEnv, cwd: box.root });
    await bNoBin.request("initialize");
    const rNoBin = await bNoBin.request("tools/call", { name: "run_begin", arguments: {} });
    assert.equal(rNoBin.result.isError, true);
    assert.match(rNoBin.result.content[0].text, /npx -y @nerveint\/ni-rua-plugin setup codex/);
    await bNoBin.close();

    // 3) ni-rua 存在但缺少 so101.local.json
    const gwBinDir = path.join(box.home, ".local", "share", "ni-rua", "envs", "gateway", "bin");
    fs.mkdirSync(gwBinDir, { recursive: true });
    fs.writeFileSync(path.join(gwBinDir, "ni-rua"), "#!/bin/sh\nexit 0\n", { mode: 0o755 });
    const bNoCfg = spawnBridge({ env: baseEnv, cwd: box.root });
    await bNoCfg.request("initialize");
    const rNoCfg = await bNoCfg.request("tools/call", { name: "run_begin", arguments: {} });
    assert.equal(rNoCfg.result.isError, true);
    assert.match(rNoCfg.result.content[0].text, /ni-rua-so101 init-config/);
    await bNoCfg.close();

    // 4) 子进程提前崩溃：立即快速失败并附带最后日志行
    const failScript = path.join(box.root, "fail-gw.mjs");
    fs.writeFileSync(failScript, 'console.error("BOOM_SERIAL_PORT_BUSY"); process.exit(1);\n');
    const bCrash = spawnBridge({
      env: { ...baseEnv, NI_RUA_GATEWAY_CMD: JSON.stringify([process.execPath, failScript]) },
      cwd: box.root,
    });
    await bCrash.request("initialize");
    const startedAt = Date.now();
    const rCrash = await bCrash.request("tools/call", { name: "run_begin", arguments: {} });
    assert.ok(Date.now() - startedAt < 4000, "early exit must fail fast without waiting for full timeout");
    assert.equal(rCrash.result.isError, true);
    assert.match(rCrash.result.content[0].text, /exited early/);
    assert.match(rCrash.result.content[0].text, /BOOM_SERIAL_PORT_BUSY/);
    await bCrash.close();
  } finally {
    box.cleanup();
  }
});
