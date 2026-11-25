// Shared fixtures: isolated HOME, fake host executables that record their argv, a fake MCP gateway.
import { spawn, spawnSync } from "node:child_process";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const PACKAGE_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
export const CLI = path.join(PACKAGE_ROOT, "bin", "cli.mjs");

const FAKE = `#!${process.execPath}
const fs = require("fs");
const name = require("path").basename(process.argv[1]);
const argv = process.argv.slice(2);
fs.appendFileSync(process.env.FAKE_LOG, JSON.stringify({ bin: name, argv, cwd: process.cwd() }) + "\\n");
if (argv[0] === "--version") {
  if (process.env["FAKE_STDERR_" + name]) console.error(process.env["FAKE_STDERR_" + name]);
  console.log(process.env["FAKE_VERSION_" + name] || ({ claude: "2.1.321 (Claude Code)", codex: "codex-cli 0.160.0", opencode: "opencode v2.2.7" })[name]);
  process.exit(0);
}
if (name === "claude" && argv[0] === "plugin" && (argv[1] === "install" || argv[1] === "uninstall")) {
  // like the real host: it keeps its own keys (enabledPlugins) in the settings file of the scope
  const scope = argv[argv.indexOf("-s") + 1] || "user";
  const path = require("path");
  const file = scope === "user" ? path.join(process.env.CLAUDE_CONFIG_DIR, "settings.json") : path.join(process.cwd(), ".claude", scope === "project" ? "settings.json" : "settings.local.json");
  const settings = fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, "utf8")) : {};
  settings.enabledPlugins = argv[1] === "install" ? { ...settings.enabledPlugins, "ni_rua@ni_rua": true } : {};
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, JSON.stringify(settings, null, 2));
}
if (name === "codex" && process.env.FAKE_CODEX_WRITE_CONFIG === "1" && argv[0] === "plugin") {
  // 模拟真实 Codex CLI 在 plugin add / marketplace add 时创建并写入 config.toml，以及 remove 时清空对应表段
  const path = require("path");
  const file = path.join(process.env.CODEX_HOME, "config.toml");
  const cur = fs.existsSync(file) ? fs.readFileSync(file, "utf8") : "";
  const stripSection = (text, header) => {
    const out = [];
    let skipping = false;
    for (const line of text.split("\\n")) {
      const t = line.trim();
      if (t === header) { skipping = true; continue; }
      if (skipping && t.startsWith("[")) skipping = false;
      if (!skipping) out.push(line);
    }
    const body = out.join("\\n");
    return body && !body.endsWith("\\n") ? body + "\\n" : body;
  };
  if (argv[1] === "marketplace" && argv[2] === "add") {
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, stripSection(cur, "[marketplaces.ni_rua]") + '[marketplaces.ni_rua]\\nsource_type = "local"\\n');
  } else if (argv[1] === "add") {
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, stripSection(cur, '[plugins."ni_rua@ni_rua"]') + '[plugins."ni_rua@ni_rua"]\\nenabled = true\\n');
  } else if (argv[1] === "remove" && fs.existsSync(file)) {
    fs.writeFileSync(file, stripSection(cur, '[plugins."ni_rua@ni_rua"]'));
  } else if (argv[1] === "marketplace" && argv[2] === "remove" && fs.existsSync(file)) {
    fs.writeFileSync(file, stripSection(cur, "[marketplaces.ni_rua]"));
  }
}
const failing = (process.env.FAKE_FAIL || "").split(",").filter(Boolean);
if (failing.some((prefix) => argv.join(" ").startsWith(prefix))) { console.error(process.env.FAKE_FAIL_MSG || "fake failure: boom"); process.exit(1); }
if (argv.join(" ") === "plugin list --json" && process.env["FAKE_LIST_" + name]) {
  if (process.env["FAKE_STDERR_" + name]) console.error(process.env["FAKE_STDERR_" + name]);
  console.log(fs.readFileSync(process.env["FAKE_LIST_" + name], "utf8"));
}
`;

export function sandbox() {
  const root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "ni-rua-plugin-test-")));
  const bin = path.join(root, "bin");
  fs.mkdirSync(bin);
  for (const name of ["claude", "codex", "opencode"]) {
    fs.writeFileSync(path.join(bin, name), FAKE, { mode: 0o755 });
  }
  const home = path.join(root, "home");
  fs.mkdirSync(home);
  const log = path.join(root, "calls.jsonl");
  fs.writeFileSync(log, "");
  const env = {
    PATH: `${bin}${path.delimiter}${process.env.PATH}`,
    HOME: home,
    CLAUDE_CONFIG_DIR: path.join(home, ".claude"),
    XDG_CONFIG_HOME: path.join(home, ".config"),
    FAKE_LOG: log,
  };
  const box = {
    root, bin, home, log, env,
    durable: path.join(home, ".local", "share", "ni-rua", "agent-plugin"),
    claudeSettings: path.join(home, ".claude", "settings.json"),
    opencodeConfig: path.join(home, ".config", "opencode", "opencode.json"),
    run(args, { cli = CLI, cwd = root, extraEnv = {} } = {}) {
      const result = spawnSync(process.execPath, [cli, ...args], { cwd, env: { ...env, ...extraEnv }, encoding: "utf8" });
      return { status: result.status, stdout: result.stdout, stderr: result.stderr, out: result.stdout + result.stderr };
    },
    // For tests that serve a gateway from this very process: spawnSync would block its event loop.
    runAsync(args, { cwd = root, extraEnv = {} } = {}) {
      return new Promise((resolve) => {
        const child = spawn(process.execPath, [CLI, ...args], { cwd, env: { ...env, ...extraEnv } });
        let stdout = "", stderr = "";
        child.stdout.on("data", (d) => { stdout += d; });
        child.stderr.on("data", (d) => { stderr += d; });
        child.on("close", (status) => resolve({ status, stdout, stderr, out: stdout + stderr }));
      });
    },
    calls() {
      return fs.readFileSync(log, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l)).filter((c) => c.argv[0] !== "--version");
    },
    commands() { return box.calls().map((c) => `${c.bin} ${c.argv.join(" ")}`); },
    clearCalls() { fs.writeFileSync(log, ""); },
    write(file, value) { fs.mkdirSync(path.dirname(file), { recursive: true }); fs.writeFileSync(file, typeof value === "string" ? value : JSON.stringify(value, null, 2)); },
    // settings without the keys the (fake) host keeps for itself
    own(file) { const value = JSON.parse(fs.readFileSync(file, "utf8")); delete value.enabledPlugins; return value; },
    read(file) { return JSON.parse(fs.readFileSync(file, "utf8")); },
    cleanup() { fs.rmSync(root, { recursive: true, force: true }); },
  };
  return box;
}

// Copy this package to `dir` as another release (version bumped, optional content change) for upgrade tests.
export function copyPackage(dir, version, mutate) {
  fs.cpSync(PACKAGE_ROOT, dir, { recursive: true, filter: (src) => !src.includes("node_modules") });
  const file = path.join(dir, "package.json");
  const pkg = JSON.parse(fs.readFileSync(file, "utf8"));
  pkg.version = version;
  fs.writeFileSync(file, JSON.stringify(pkg, null, 2));
  mutate?.(dir);
  return path.join(dir, "bin", "cli.mjs");
}

// A tiny MCP-over-HTTP gateway double: initialize as JSON, tools/list as SSE, like NiRuaMcpServer's tools/call.
export function fakeGateway(toolNames) {
  const requests = [];
  const server = http.createServer((req, res) => {
    if (req.method !== "POST" || req.url !== "/mcp") { res.writeHead(404).end(); return; }
    let body = "";
    req.on("data", (chunk) => { body += chunk; });
    req.on("end", () => {
      let message;
      try { message = JSON.parse(body); } catch { res.writeHead(400).end(); return; }
      if (!message || typeof message !== "object" || typeof message.method !== "string") { res.writeHead(400).end(); return; }
      requests.push(message);
      if (message.method === "notifications/initialized") { res.writeHead(202).end(); return; }
      if (message.method === "initialize") {
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify({ jsonrpc: "2.0", id: message.id, result: { protocolVersion: "2025-06-18", serverInfo: { name: "ni-rua-test", version: "0" }, capabilities: { tools: {} }, instructions: "call run_begin first" } }));
        return;
      }
      if (message.method === "tools/list") {
        res.writeHead(200, { "content-type": "text/event-stream" });
        res.end(`event: message\ndata: ${JSON.stringify({ jsonrpc: "2.0", id: message.id, result: { tools: toolNames.map((name) => ({ name, inputSchema: { type: "object" } })) } })}\n\n`);
        return;
      }
      res.writeHead(404).end();
    });
  });
  return new Promise((resolve) => server.listen(0, "127.0.0.1", () => resolve({
    port: server.address().port, requests, close: () => new Promise((done) => server.close(done)),
  })));
}
