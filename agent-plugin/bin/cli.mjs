#!/usr/bin/env node
// ni-rua-plugin: install / uninstall / doctor for the NI-RUA agent plugin (Claude Code, Codex, OpenCode).
//
// Rules this installer follows:
//  - It installs from a durable copy of this package (~/.local/share/ni-rua/agent-plugin/<version>/), because
//    `npx` caches are not stable and Codex copies a local marketplace into its own cache.
//  - Every host command and every settings change is printed before it runs and recorded in installed.json next to
//    the durable copy, so `uninstall` removes exactly what was added and leaves values the user set alone.
//  - It never installs Python, starts a gateway, touches provider keys or calls a model.
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { runStackChecks } from "./doctor-checks.mjs";
import { runGateway } from "./gateway.mjs";
import { ESTOP_WARNING } from "./runtime.mjs";
import { runSetup } from "./setup.mjs";

const PACKAGE_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const PKG = JSON.parse(fs.readFileSync(path.join(PACKAGE_ROOT, "package.json"), "utf8"));
const MANIFEST = JSON.parse(fs.readFileSync(path.join(PACKAGE_ROOT, "manifest.generated.json"), "utf8"));
const HOSTS = ["claude", "codex", "opencode"];
const DEFAULT_PORT = MANIFEST.default_port;
const MCP_URL_RE = /http:\/\/127\.0\.0\.1:\d+\/mcp/g;
const PORT_FILES = [".mcp.json", ".codex-mcp.json", "manifest.generated.json"];
const PLUGIN_ID = MANIFEST.plugin_id;
const MARKETPLACE = MANIFEST.marketplace;
const CLAUDE_ALLOW = MANIFEST.claude_allow;
const CLAUDE_ENV = MANIFEST.claude_env;
const OPENCODE_TOOL_OUTPUT = MANIFEST.opencode_tool_output;
const OPENCODE_MEDIA_IMAGE = MANIFEST.opencode_media_image;
const CODEX_DIRECT_CATALOG = MANIFEST.codex_direct_catalog || "codex-models.direct.json";
const CODEX_CODE_MODE_ONLY = new Set(MANIFEST.codex_code_mode_only_models || []);
const VERIFIED = MANIFEST.verified_hosts;
const GATEWAY_HINT = [
  "ni-rua-plugin gateway --dry-run          # zero-hardware MCP connectivity test",
  "       ni-rua-plugin gateway --config so101     # local CPU SO-101 arm",
].join("\n");

class UserError extends Error {}

const out = (text = "") => console.log(text);
const home = () => os.homedir();
const dataRoot = () => path.join(home(), ".local", "share", "ni-rua", "agent-plugin");
const recordPath = () => path.join(dataRoot(), "installed.json");

function realPathOrResolve(target) {
  const abs = path.resolve(target);
  try { return fs.realpathSync(abs); } catch { return abs; }
}

// ---------------------------------------------------------------- arguments
function parseArgs(argv) {
  const opts = { _: [], extra: [] };
  const valued = new Set([
    "scope", "port", "claude-bin", "codex-bin", "opencode-bin",
    "config", "run-dir", "prompt", "gateway-url",
  ]);
  const flags = new Set([
    "no-host-env", "no-allow", "no-direct-catalog", "force", "help",
    "so101", "skip-gateway", "dry-run", "json",
  ]);
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--") { opts.extra.push(...argv.slice(i + 1)); break; }
    if (!arg.startsWith("--")) { opts._.push(arg); continue; }
    const [name, inline] = arg.slice(2).split(/=(.*)/s, 2);
    if (flags.has(name)) { opts[name] = true; continue; }
    if (!valued.has(name)) throw new UserError(`unknown option --${name}`);
    const value = inline ?? argv[++i];
    if (value === undefined) throw new UserError(`--${name} needs a value`);
    opts[name] = value;
  }
  if (opts.port !== undefined) {
    const port = Number(opts.port);
    if (!Number.isInteger(port) || port < 1 || port > 65535) throw new UserError(`--port must be an integer 1..65535, got ${opts.port}`);
    opts.port = port;
  }
  if (opts.scope !== undefined && !["user", "project", "local"].includes(opts.scope)) {
    throw new UserError(`--scope must be user, project or local, got ${opts.scope}`);
  }
  if (opts["gateway-url"] !== undefined) {
    try { new URL(opts["gateway-url"]); } catch { throw new UserError(`--gateway-url must be a valid URL, got ${opts["gateway-url"]}`); }
  }
  return opts;
}

// ---------------------------------------------------------------- host executables and commands
function findExecutable(host, opts) {
  const override = opts[`${host}-bin`];
  if (override) return fs.existsSync(override) ? path.resolve(override) : null;
  for (const dir of (process.env.PATH || "").split(path.delimiter)) {
    if (!dir) continue;
    const candidate = path.join(dir, host);
    try {
      fs.accessSync(candidate, fs.constants.X_OK);
      if (fs.statSync(candidate).isFile()) return candidate;
    } catch { /* next */ }
  }
  return null;
}

function requireExecutable(host, opts) {
  const bin = findExecutable(host, opts);
  if (!bin) {
    throw new UserError(opts[`${host}-bin`]
      ? `${opts[`${host}-bin`]} does not exist (--${host}-bin)`
      : `${host} was not found on PATH; install it or pass --${host}-bin <path>`);
  }
  return bin;
}

// Print the command, run it, print its output. `allowFail` returns the result instead of aborting.
function run(bin, args, { cwd, env, allowFail = false } = {}) {
  out(`+ ${[bin, ...args].map((a) => (/\s/.test(a) ? JSON.stringify(a) : a)).join(" ")}${cwd ? `   (in ${cwd})` : ""}`);
  const result = spawnSync(bin, args, { cwd, env: { ...process.env, ...env }, encoding: "utf8" });
  const text = `${result.stdout || ""}${result.stderr || ""}`.trim();
  if (text) out(text.split("\n").map((line) => `  ${line}`).join("\n"));
  if (result.error) {
    out(`  ${result.error.message}`);
    if (allowFail) return { ok: false, text: `${text} ${result.error.message}`.trim() };
    throw new UserError(`could not run ${bin}: ${result.error.message}`);
  }
  if (result.status !== 0 && !allowFail) throw new UserError(`${path.basename(bin)} ${args.slice(0, 3).join(" ")} failed with exit code ${result.status}`);
  return { ok: result.status === 0, text };
}

// 保持 stdout 与 stderr 分离，避免宿主向 stderr 打印警告时污染 JSON 输出或版本解析
function capture(bin, args, { cwd, env } = {}) {
  const result = spawnSync(bin, args, {
    cwd,
    env: env ? { ...process.env, ...env } : process.env,
    encoding: "utf8",
    timeout: 20000,
  });
  const stdout = (result.stdout || "").trim();
  const stderr = (result.stderr || "").trim();
  return {
    ok: !result.error && result.status === 0,
    stdout,
    stderr,
    text: [stdout, stderr].filter(Boolean).join("\n"),
  };
}

// ---------------------------------------------------------------- JSON files
function readJson(file, what) {
  if (!fs.existsSync(file)) return null;
  const text = fs.readFileSync(file, "utf8");
  if (!text.trim()) return {};
  let value;
  try { value = JSON.parse(text); } catch (error) { throw new UserError(`${file} is not strict JSON (${error.message}); ${what}`); }
  if (value === null || typeof value !== "object" || Array.isArray(value)) throw new UserError(`${file} does not hold a JSON object; ${what}`);
  return value;
}

function resolveWriteTarget(file) {
  let current = file;
  const seen = new Set();
  for (let hops = 0; hops < 16; hops++) {
    let st;
    try { st = fs.lstatSync(current); } catch (error) {
      if (error && error.code === "ENOENT") return { target: current, mode: undefined };
      throw error;
    }
    if (!st.isSymbolicLink()) {
      return { target: fs.realpathSync(current), mode: st.mode & 0o7777 };
    }
    if (seen.has(current)) break;
    seen.add(current);
    current = path.resolve(path.dirname(current), fs.readlinkSync(current));
  }
  const resolved = fs.realpathSync(file);
  return { target: resolved, mode: fs.statSync(resolved).mode & 0o7777 };
}

function removeCreatedFile(file) {
  const real = fs.realpathSync(file);
  fs.rmSync(real, { force: true });
  try {
    if (!fs.lstatSync(file).isSymbolicLink()) fs.rmSync(file, { force: true });
  } catch { /* already gone */ }
}

// Write through a symlink (dotfile managers link these files) and keep the file mode (settings may hold secrets).
function writeText(file, content) {
  const { target, mode } = resolveWriteTarget(file);
  fs.mkdirSync(path.dirname(target), { recursive: true });
  const temp = `${target}.ni-rua-tmp-${process.pid}`;
  const effectiveMode = mode ?? 0o600;
  try {
    fs.writeFileSync(temp, content, { mode: effectiveMode });
    fs.chmodSync(temp, effectiveMode);
    fs.renameSync(temp, target);
  } catch (error) {
    try { fs.rmSync(temp, { force: true }); } catch { /* best-effort cleanup */ }
    throw error;
  }
}

function writeJson(file, value) {
  writeText(file, `${JSON.stringify(value, null, 2)}\n`);
}

function sameValue(a, b) {
  if (a === b) return true;
  if (a === null || b === null || typeof a !== "object" || typeof b !== "object") return false;
  if (Array.isArray(a) || Array.isArray(b)) {
    return Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((item, i) => sameValue(item, b[i]));
  }
  const keysA = Object.keys(a);
  const keysB = Object.keys(b);
  return keysA.length === keysB.length && keysA.every((key) => Object.prototype.hasOwnProperty.call(b, key) && sameValue(a[key], b[key]));
}
const isEmpty = (object) => Object.keys(object).length === 0;

// ---------------------------------------------------------------- install record and lock
// installed.json: {schema: 2, port, copies: [versions this tool created], installs: {<key>: entry}}. One entry per
// host AND config home (Claude: config dir + scope [+ project dir]; Codex: CODEX_HOME; OpenCode: config file), so
// two homes or projects never overwrite each other, and one port is shared by all of them (they load the same copy).
function loadRecord() {
  const record = readJson(recordPath(), "fix or delete it to continue") ?? {};
  if (record.schema !== undefined && record.schema !== 2) throw new UserError(`${recordPath()} has an unknown schema; remove it after uninstalling by hand`);
  record.schema = 2;
  record.copies ??= [];
  record.installs ??= {};
  return record;
}

function saveRecord(record) {
  if (isEmpty(record.installs) && record.copies.length === 0) { fs.rmSync(recordPath(), { force: true }); return; }
  writeJson(recordPath(), record);
}

// A lock directory serialises install and uninstall (two at once lost record entries). Remove it by hand only when
// no ni-rua-plugin is running.
function withLock(fn) {
  fs.mkdirSync(dataRoot(), { recursive: true });
  const lock = path.join(dataRoot(), ".lock");
  try { fs.mkdirSync(lock); } catch (error) {
    if (error.code === "EEXIST") throw new UserError(`another ni-rua-plugin is running (${lock} exists); wait for it, or remove that directory if none is`);
    throw error;
  }
  try { return fn(); } finally {
    fs.rmSync(lock, { recursive: true, force: true });
    try { fs.rmdirSync(dataRoot()); } catch { /* not empty: something is installed */ }
  }
}

const claudeConfigDir = () => process.env.CLAUDE_CONFIG_DIR || path.join(home(), ".claude");
const codexHome = () => process.env.CODEX_HOME || path.join(home(), ".codex");

function installKey(host, opts) {
  if (host === "claude") {
    const scope = opts.scope ?? "user";
    return `claude|${claudeConfigDir()}|${scope}${scope === "user" ? "" : `|${realPathOrResolve(process.cwd())}`}`;
  }
  if (host === "codex") return `codex|${codexHome()}`;
  return `opencode|${opencodeConfigPath()}`;
}

// ---------------------------------------------------------------- durable copy
function listFiles(root) {
  const files = [];
  const walk = (dir) => {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      if (entry.name === ".DS_Store" || entry.name === "node_modules") continue;
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) walk(full);
      else if (entry.isFile()) files.push(path.relative(root, full).split(path.sep).join("/"));
    }
  };
  walk(root);
  return files.sort();
}

// The package's files set (package.json "files" + package.json), with `port` written into the three MCP entries.
function packageContent(port) {
  const map = new Map();
  for (const item of [...PKG.files, "package.json"]) {
    const full = path.join(PACKAGE_ROOT, item);
    if (!fs.existsSync(full)) throw new UserError(`package file ${item} is missing from ${PACKAGE_ROOT}`);
    const names = fs.statSync(full).isDirectory() ? listFiles(full).map((name) => `${item}/${name}`) : [item];
    for (const name of names) map.set(name, fs.readFileSync(path.join(PACKAGE_ROOT, name)));
  }
  for (const name of PORT_FILES) {
    if (!map.has(name)) throw new UserError(`package file ${name} is missing`);
    const text = map.get(name).toString("utf8");
    if (!text.match(MCP_URL_RE)) throw new UserError(`${name} has no MCP URL to rewrite`);
    map.set(name, Buffer.from(text.replace(MCP_URL_RE, `http://127.0.0.1:${port}/mcp`)));
  }
  return map;
}

const withDefaultPort = (name, buffer) => (PORT_FILES.includes(name) ? buffer.toString("utf8").replace(MCP_URL_RE, `http://127.0.0.1:${DEFAULT_PORT}/mcp`) : buffer.toString("utf8"));

// One port for all installs: the copy is shared (Claude Code and OpenCode load it in place).
function resolvePort(opts, record, key) {
  const others = Object.keys(record.installs).filter((k) => k !== key);
  if (opts.port !== undefined) {
    if (others.length && record.port !== undefined && opts.port !== record.port) {
      throw new UserError(`port ${record.port} is in use by other installs (${others.join(", ")}); they share one copy. Rerun with --port ${record.port}, or uninstall them first.`);
    }
    return opts.port;
  }
  return record.port ?? DEFAULT_PORT;
}

// Check (no writes): same version with different content, other than the port, is refused unless --force.
function planCopy(port, force) {
  const target = path.join(dataRoot(), PKG.version);
  const content = packageContent(port);
  let replace = false;
  if (fs.existsSync(target)) {
    const existing = listFiles(target);
    const differs = existing.length !== content.size
      || existing.some((name) => !content.has(name) || withDefaultPort(name, fs.readFileSync(path.join(target, name))) !== withDefaultPort(name, content.get(name)));
    if (differs && !force) throw new UserError(`${target} already holds version ${PKG.version} with different content; nothing was changed. Bump the version, or rerun with --force to replace it.`);
    replace = differs;
  }
  return { target, content, replace, port };
}

function writeCopy(plan, record) {
  const { target, content, replace, port } = plan;
  if (replace) { out(`replacing ${target} (--force)`); fs.rmSync(target, { recursive: true, force: true }); }
  let written = 0;
  for (const [name, buffer] of content) {
    const file = path.join(target, name);
    if (fs.existsSync(file) && fs.readFileSync(file).equals(buffer)) continue;
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, buffer);
    written++;
  }
  fs.chmodSync(path.join(target, "bin", "cli.mjs"), 0o755);
  if (!record.copies.includes(PKG.version)) record.copies.push(PKG.version);
  record.port = port;
  out(`durable copy: ${target} (${written ? `${written} file${written === 1 ? "" : "s"} written` : "up to date"}, MCP port ${port})`);
  return target;
}

// Delete only version directories this tool recorded as its own and no install uses any more.
function dropUnusedCopies(record) {
  const used = new Set(Object.values(record.installs).map((entry) => entry.version));
  for (const version of [...record.copies]) {
    if (used.has(version)) continue;
    const dir = path.join(dataRoot(), version);
    if (path.dirname(dir) === dataRoot() && fs.existsSync(dir)) { out(`removing durable copy ${dir}`); fs.rmSync(dir, { recursive: true, force: true }); }
    record.copies.splice(record.copies.indexOf(version), 1);
  }
  if (isEmpty(record.installs)) delete record.port;
}

// ---------------------------------------------------------------- Claude Code settings
function claudeSettingsPath(scope, cwd) {
  if (scope === "user") return path.join(claudeConfigDir(), "settings.json");
  return path.join(cwd, ".claude", scope === "project" ? "settings.json" : "settings.local.json");
}

// Values that defeat the plugin if they stay: below what it needs (a lower description cap truncates the field
// guides, a lower timeout cuts long calls), or auto-backgrounding on.
function lowEnvValues(env) {
  return Object.entries(CLAUDE_ENV).filter(([key, required]) => {
    if (!(key in env)) return false;
    if (key === "CLAUDE_CODE_MCP_AUTO_BACKGROUND_MS") return String(env[key]) !== required;
    const value = Number(env[key]);
    return !(value >= Number(required));
  }).map(([key, required]) => `${key}=${JSON.stringify(env[key])} (${key.endsWith("BACKGROUND_MS") ? `needs exactly ${required}` : `needs ${required} or more`})`);
}

// Remove, in memory, exactly what `change` added; anything the user changed since is left alone.
function revertClaude(settings, change, messages) {
  for (const [key, value] of Object.entries(change.env_added || {})) {
    if (settings.env?.[key] === value) { messages.push(`settings ${change.path}: remove env.${key}`); delete settings.env[key]; }
    else if (settings.env && key in settings.env) messages.push(`settings ${change.path}: env.${key} was changed since install; left as is`);
  }
  if (change.allow_added && Array.isArray(settings.permissions?.allow)) {
    const index = settings.permissions.allow.indexOf(CLAUDE_ALLOW);
    if (index >= 0) { messages.push(`settings ${change.path}: remove permissions.allow ${CLAUDE_ALLOW}`); settings.permissions.allow.splice(index, 1); }
  }
  if (change.created_allow && settings.permissions?.allow?.length === 0) delete settings.permissions.allow;
  if (change.created_permissions && settings.permissions && isEmpty(settings.permissions)) delete settings.permissions;
  if (change.created_env && settings.env && isEmpty(settings.env)) delete settings.env;
}

// Plan the settings change of an install (no writes): revert the earlier install's additions, then merge ours.
function planClaudeSettings(file, opts, prior) {
  const loaded = readJson(file, "fix it or pass --no-host-env --no-allow");
  const settings = loaded ?? {};
  const messages = [];
  let existed = loaded !== null;
  if (prior) {
    revertClaude(settings, prior, messages);
    if (prior.created_file && isEmpty(settings)) existed = false;
  }
  const change = { path: file, created_file: !existed, env_added: {}, allow_added: false };
  if (!opts["no-host-env"]) {
    if (settings.env !== undefined && (settings.env === null || typeof settings.env !== "object" || Array.isArray(settings.env))) throw new UserError(`${file}: "env" is not an object`);
    change.created_env = settings.env === undefined;
    settings.env ??= {};
    for (const [key, value] of Object.entries(CLAUDE_ENV)) {
      if (key in settings.env) { messages.push(`settings ${file}: env.${key} is already set (${JSON.stringify(settings.env[key])}); kept`); continue; }
      messages.push(`settings ${file}: set env.${key} = ${JSON.stringify(value)}`);
      settings.env[key] = value;
      change.env_added[key] = value;
    }
    for (const low of lowEnvValues(settings.env)) messages.push(`WARNING ${file}: ${low}; kept as you set it, long results or calls may be cut`);
  }
  if (!opts["no-allow"]) {
    const permissions = settings.permissions;
    if (permissions !== undefined && (permissions === null || typeof permissions !== "object" || Array.isArray(permissions))) throw new UserError(`${file}: "permissions" is not an object`);
    if (permissions?.allow !== undefined && !Array.isArray(permissions.allow)) throw new UserError(`${file}: "permissions.allow" is not an array`);
    change.created_permissions = permissions === undefined;
    settings.permissions ??= {};
    change.created_allow = settings.permissions.allow === undefined;
    settings.permissions.allow ??= [];
    if (settings.permissions.allow.includes(CLAUDE_ALLOW)) messages.push(`settings ${file}: permissions.allow already has ${CLAUDE_ALLOW}; kept`);
    else { messages.push(`settings ${file}: add permissions.allow ${CLAUDE_ALLOW}`); settings.permissions.allow.push(CLAUDE_ALLOW); change.allow_added = true; }
  }
  return { settings, messages, change, nothing: isEmpty(settings) && !existed };
}

function commitSettings(plan) {
  plan.messages.forEach(out);
  if (plan.nothing) return undefined;     // reinstall without options after an install that created the file: nothing left
  writeJson(plan.change.path, plan.settings);
  return plan.change;
}

function undoClaudeSettings(change) {
  if (!change || !fs.existsSync(change.path)) return;
  const settings = readJson(change.path, "cannot restore it automatically");
  const messages = [];
  revertClaude(settings, change, messages);
  messages.forEach(out);
  if (change.created_file && isEmpty(settings)) { out(`settings ${change.path}: remove the file this installer created`); removeCreatedFile(change.path); }
  else writeJson(change.path, settings);
}

// 检测 Claude Code 工作区信任状态（未信任目录会忽略项目级/本地级 permissions.allow 与 env）
function findGitRoot(startDir) {
  let current = realPathOrResolve(startDir);
  while (true) {
    if (fs.existsSync(path.join(current, ".git"))) return current;
    const parent = path.dirname(current);
    if (parent === current) return null;
    current = parent;
  }
}

function claudeStateFiles() {
  if (process.env.CLAUDE_CONFIG_DIR) return [path.join(process.env.CLAUDE_CONFIG_DIR, ".claude.json")];
  return [path.join(home(), ".claude.json"), path.join(claudeConfigDir(), ".claude.json")];
}

function checkClaudeProjectTrust(dir) {
  const realDir = realPathOrResolve(dir);
  const gitRoot = findGitRoot(realDir);
  const isSubdirOfGit = Boolean(gitRoot && gitRoot !== realDir);
  // Claude Code 在 git 仓库子目录中仅以最外层 .git 根目录作为 trust key；仅信任子目录本身不会生效
  const requiredKey = gitRoot ?? realDir;
  const files = claudeStateFiles();
  let stateFile = files.find((f) => fs.existsSync(f)) ?? files[0];
  let subdirTrustedOnly = false;
  for (const file of files) {
    if (!fs.existsSync(file)) continue;
    let parsed;
    try { parsed = JSON.parse(fs.readFileSync(file, "utf8")); } catch { continue; }
    const projects = parsed?.projects;
    if (!projects || typeof projects !== "object") continue;
    if (projects[requiredKey]?.hasTrustDialogAccepted === true) {
      return { trusted: true, trustedKey: requiredKey, requiredKey, subdirTrustedOnly: false, realDir, gitRoot, isSubdirOfGit, stateFile: file };
    }
    if (isSubdirOfGit && projects[realDir]?.hasTrustDialogAccepted === true) {
      subdirTrustedOnly = true;
    }
    stateFile = file;
  }
  return { trusted: false, trustedKey: null, requiredKey, subdirTrustedOnly, realDir, gitRoot, isSubdirOfGit, stateFile };
}

function selectClaudePluginEntry(entries, cwd) {
  if (!Array.isArray(entries)) return null;
  const matches = entries.filter((p) => p && p.id === PLUGIN_ID);
  if (!matches.length) return null;
  const isCurrentScope = (p) => p.scope === "user" || (typeof p.projectPath === "string" && realPathOrResolve(p.projectPath) === cwd);
  return matches.find((p) => p.enabled && isCurrentScope(p))
    ?? matches.find((p) => p.enabled)
    ?? matches.find(isCurrentScope)
    ?? matches[0];
}

// ---------------------------------------------------------------- Codex config.toml (direct model catalog)
function codexConfigPath(dir = codexHome()) {
  return path.join(dir, "config.toml");
}

function unquoteTomlString(raw) {
  const trimmed = raw.trim();
  if (trimmed.startsWith('"') && trimmed.endsWith('"')) {
    try { return JSON.parse(trimmed); } catch { return trimmed.slice(1, -1); }
  }
  if (trimmed.startsWith("'") && trimmed.endsWith("'")) return trimmed.slice(1, -1);
  return null;
}

// 仅解析 config.toml 顶层表之前（首个 [section] 之前）的 model 与 model_catalog_json 字段
function parseCodexTopLevel(text) {
  const outMap = { model: null, model_catalog_json: null };
  for (const rawLine of text.split(/\r?\n/)) {
    const trimmed = rawLine.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    if (trimmed.startsWith("[")) break;
    // 先匹配完整的带引号字符串字面量再忽略行尾注释，避免路径中的 `#` 被误切为注释
    const match = /^(model|model_catalog_json)\s*=\s*("(?:\\.|[^"\\])*"|'[^']*')\s*(?:#.*)?$/.exec(trimmed);
    if (!match) continue;
    const val = unquoteTomlString(match[2]);
    if (val !== null) outMap[match[1]] = val;
  }
  return outMap;
}

function stripExactLine(text, targetLine) {
  if (!targetLine) return { text, removed: false };
  const hasBom = text.startsWith("\uFEFF");
  const body = hasBom ? text.slice(1) : text;
  const lines = body.split("\n");
  const index = lines.findIndex((line) => line.replace(/\r$/, "") === targetLine);
  if (index < 0) return { text, removed: false };
  lines.splice(index, 1);
  const joined = lines.join("\n");
  return { text: hasBom && joined ? `\uFEFF${joined}` : joined, removed: true };
}

function planCodexConfig(codexDir, catalogFile, opts, prior, { existedBeforeHost } = {}) {
  const file = codexConfigPath(codexDir);
  if (fs.existsSync(file) && fs.statSync(file).isDirectory()) {
    throw new UserError(`${file} is a directory, not a config.toml file`);
  }
  const existedOriginally = fs.existsSync(file);
  let text = existedOriginally ? fs.readFileSync(file, "utf8") : "";
  let existed = existedBeforeHost ?? existedOriginally;
  const messages = [];
  if (prior?.line) {
    const stripped = stripExactLine(text, prior.line);
    if (stripped.removed) {
      messages.push(`config ${file}: remove earlier model_catalog_json entry`);
      text = stripped.text;
    } else if (parseCodexTopLevel(text).model_catalog_json) {
      messages.push(`config ${file}: model_catalog_json was changed since install; kept`);
    }
    if (prior.created_file && !text.trim()) existed = false;
  }
  const top = parseCodexTopLevel(text);
  if (opts["no-direct-catalog"]) {
    if (!top.model_catalog_json && (process.platform === "linux" || (top.model && CODEX_CODE_MODE_ONLY.has(top.model)))) {
      messages.push(`WARNING ${file}: --no-direct-catalog skipped model_catalog_json; on Linux Codex code_mode_only models crash on MCP calls unless you pass -c 'model_catalog_json="${catalogFile}"'`);
    }
    return {
      path: file,
      text,
      messages,
      change: undefined,
      write: Boolean(prior?.line && text !== (existedOriginally ? fs.readFileSync(file, "utf8") : "")),
      removeEmpty: Boolean(prior?.created_file && !text.trim()),
    };
  }
  if (top.model_catalog_json) {
    messages.push(`config ${file}: model_catalog_json is already set (${JSON.stringify(top.model_catalog_json)}); kept`);
    return { path: file, text, messages, change: undefined, write: false, removeEmpty: false };
  }
  const line = `model_catalog_json = ${JSON.stringify(catalogFile)}`;
  messages.push(`config ${file}: set ${line} (prevents Linux code-mode host crashes on code_mode_only models)`);
  const hasBom = text.startsWith("\uFEFF");
  const body = hasBom ? text.slice(1) : text;
  const nextText = body ? `${hasBom ? "\uFEFF" : ""}${line}\n${body}` : `${line}\n`;
  return {
    path: file,
    text: nextText,
    messages,
    change: { path: file, created_file: !existed, line },
    write: true,
    removeEmpty: false,
  };
}

function commitCodexConfig(plan) {
  plan.messages.forEach(out);
  if (plan.removeEmpty) {
    if (fs.existsSync(plan.path)) removeCreatedFile(plan.path);
    return undefined;
  }
  if (plan.write) writeText(plan.path, plan.text);
  return plan.change;
}

function undoCodexConfig(change) {
  if (!change || !fs.existsSync(change.path)) return;
  const text = fs.readFileSync(change.path, "utf8");
  const stripped = stripExactLine(text, change.line);
  if (stripped.removed) {
    out(`config ${change.path}: remove model_catalog_json entry`);
  } else {
    out(`config ${change.path}: model_catalog_json was changed since install; left as is`);
    return;
  }
  if (change.created_file && !stripped.text.trim()) {
    out(`config ${change.path}: remove the file this installer created`);
    removeCreatedFile(change.path);
  } else {
    writeText(change.path, stripped.text);
  }
}

// ---------------------------------------------------------------- OpenCode config
function opencodeConfigDir() {
  return process.env.OPENCODE_CONFIG_DIR || path.join(process.env.XDG_CONFIG_HOME || path.join(home(), ".config"), "opencode");
}

function opencodeConfigPath() {
  return path.join(opencodeConfigDir(), "opencode.json");
}

function opencodeToolOutputSufficient(value) {
  return value !== null
    && typeof value === "object"
    && !Array.isArray(value)
    && typeof value.max_lines === "number"
    && typeof value.max_bytes === "number"
    && value.max_lines >= OPENCODE_TOOL_OUTPUT.max_lines
    && value.max_bytes >= OPENCODE_TOOL_OUTPUT.max_bytes;
}

function opencodeMediaImageSufficient(value) {
  return value !== null
    && typeof value === "object"
    && !Array.isArray(value)
    && value.auto_resize === false
    && typeof value.max_width === "number"
    && typeof value.max_height === "number"
    && typeof value.max_base64_bytes === "number"
    && value.max_width >= OPENCODE_MEDIA_IMAGE.max_width
    && value.max_height >= OPENCODE_MEDIA_IMAGE.max_height
    && value.max_base64_bytes >= OPENCODE_MEDIA_IMAGE.max_base64_bytes;
}

function manualOpencodeEdit(dir) {
  return `add this to your opencode config by hand: ${JSON.stringify({ plugins: [dir], tool_output: OPENCODE_TOOL_OUTPUT, media: { image: OPENCODE_MEDIA_IMAGE } })}`;
}

function revertOpencode(config, change, messages) {
  if (Array.isArray(config.plugins)) {
    for (const entry of change.plugins_added || []) {
      const index = config.plugins.indexOf(entry);
      if (index >= 0) { messages.push(`config ${change.path}: remove plugins entry ${entry}`); config.plugins.splice(index, 1); }
    }
    if (change.created_plugins && config.plugins.length === 0) delete config.plugins;
  }
  if (change.tool_output_added) {
    if (sameValue(config.tool_output, OPENCODE_TOOL_OUTPUT)) { messages.push(`config ${change.path}: remove tool_output`); delete config.tool_output; }
    else messages.push(`config ${change.path}: tool_output was changed since install; left as is`);
  }
  if (change.media_image_added && config.media && typeof config.media === "object" && !Array.isArray(config.media)) {
    if (sameValue(config.media.image, OPENCODE_MEDIA_IMAGE)) {
      messages.push(`config ${change.path}: remove media.image`);
      delete config.media.image;
      if (change.created_media && isEmpty(config.media)) delete config.media;
    } else {
      messages.push(`config ${change.path}: media.image was changed since install; left as is`);
    }
  }
}

function planOpencodeConfig(dir, prior) {
  const file = opencodeConfigPath();
  const sibling = path.join(path.dirname(file), "opencode.jsonc");
  if (fs.existsSync(sibling)) throw new UserError(`${sibling} exists (JSONC); this installer edits only strict JSON. ${manualOpencodeEdit(dir)}`);
  const loaded = readJson(file, manualOpencodeEdit(dir));
  const config = loaded ?? {};
  const messages = [];
  let existed = loaded !== null;
  if (prior) {
    revertOpencode(config, prior, messages);
    if (prior.created_file && isEmpty(config)) existed = false;
  }
  if (config.plugins !== undefined && !Array.isArray(config.plugins)) throw new UserError(`${file}: "plugins" is not an array; ${manualOpencodeEdit(dir)}`);
  if (config.media !== undefined && (config.media === null || typeof config.media !== "object" || Array.isArray(config.media))) {
    throw new UserError(`${file}: "media" is not an object; ${manualOpencodeEdit(dir)}`);
  }
  const change = {
    path: file,
    created_file: !existed,
    plugins_added: [],
    tool_output_added: false,
    created_plugins: config.plugins === undefined,
    media_image_added: false,
    created_media: config.media === undefined,
  };
  config.plugins ??= [];
  if (config.plugins.includes(dir)) messages.push(`config ${file}: plugins already has ${dir}`);
  else { messages.push(`config ${file}: add plugins entry ${dir}`); config.plugins.push(dir); change.plugins_added.push(dir); }
  if (config.tool_output === undefined) {
    messages.push(`config ${file}: set tool_output ${JSON.stringify(OPENCODE_TOOL_OUTPUT)}`);
    config.tool_output = { ...OPENCODE_TOOL_OUTPUT };
    change.tool_output_added = true;
  } else if (opencodeToolOutputSufficient(config.tool_output)) {
    messages.push(`config ${file}: tool_output is already set; kept`);
  } else messages.push(`config ${file}: tool_output is already set; kept (results over about 50 KB may reach the model cut unless you raise it)`);
  if (config.media?.image === undefined) {
    messages.push(`config ${file}: set media.image ${JSON.stringify(OPENCODE_MEDIA_IMAGE)}`);
    config.media ??= {};
    config.media.image = { ...OPENCODE_MEDIA_IMAGE };
    change.media_image_added = true;
  } else if (opencodeMediaImageSufficient(config.media.image)) {
    messages.push(`config ${file}: media.image is already set; kept`);
  } else {
    messages.push(`config ${file}: media.image is already set; kept (large PNG tool images may be re-encoded to JPEG unless auto_resize is false)`);
  }
  return { config, messages, change };
}

function undoOpencodeConfig(change) {
  if (!change || !fs.existsSync(change.path)) return;
  const config = readJson(change.path, "cannot restore it automatically");
  const messages = [];
  revertOpencode(config, change, messages);
  messages.forEach(out);
  if (change.created_file && isEmpty(config)) { out(`config ${change.path}: remove the file this installer created`); removeCreatedFile(change.path); }
  else writeJson(change.path, config);
}

// ---------------------------------------------------------------- install
// Every step that can refuse (executable, copy conflict, port conflict, unparsable settings) runs before the first
// write, so a refused reinstall leaves the working configuration untouched.
function rollbackUnrecordedCopy(record, prevPort) {
  if (prevPort === undefined) delete record.port;
  else record.port = prevPort;
  dropUnusedCopies(record);
  saveRecord(record);
}

function installClaude(opts, record) {
  const bin = requireExecutable("claude", opts);
  const scope = opts.scope ?? "user";
  const cwd = realPathOrResolve(process.cwd());
  const key = installKey("claude", opts);
  const prior = record.installs[key];
  const port = resolvePort(opts, record, key);
  const copy = planCopy(port, opts.force);
  const wantsSettings = !opts["no-host-env"] || !opts["no-allow"];
  const settingsFile = claudeSettingsPath(scope, cwd);
  const planSettings = () => (wantsSettings || prior?.settings ? planClaudeSettings(settingsFile, opts, prior?.settings) : null);
  planSettings();     // validation only: an unparsable settings file refuses before anything is written
  if (prior) out(`replacing the earlier Claude Code install (version ${prior.version}, ${scope})`);
  const prevPort = record.port;
  const dir = writeCopy(copy, record);
  const entry = {
    host: "claude", version: PKG.version, dir, scope, cwd, bin, config_dir: claudeConfigDir(),
    // 保留旧安装记录中的 settings 变更，防止后续 plugin install 失败时丢失可回滚记录
    ...(prior?.settings ? { settings: prior.settings } : {}),
  };
  const env = { CLAUDE_CONFIG_DIR: entry.config_dir };
  try {
    run(bin, ["plugin", "marketplace", "add", dir], { cwd, env });      // an existing "ni_rua" marketplace is repointed (upgrade)
  } catch (error) {
    rollbackUnrecordedCopy(record, prevPort);
    throw error;
  }
  record.installs[key] = entry;
  saveRecord(record);
  run(bin, ["plugin", "install", PLUGIN_ID, "-s", scope], { cwd, env });
  // plan again now: `claude plugin install` wrote enabledPlugins into this very file, which must not be overwritten
  const settingsPlan = planSettings();
  if (settingsPlan) {
    const settingsChange = commitSettings(settingsPlan);
    if (settingsChange) entry.settings = settingsChange;
    else delete entry.settings;
    saveRecord(record);
  }
  if (scope !== "user") {
    const trust = checkClaudeProjectTrust(cwd);
    if (!trust.trusted) {
      const rootHint = trust.isSubdirOfGit
        ? (trust.subdirTrustedOnly
          ? ` (${cwd} is trusted, but Claude Code checks the enclosing .git root ${trust.gitRoot})`
          : ` (Claude Code resolves workspace trust at the enclosing .git root ${trust.gitRoot})`)
        : "";
      out(`WARNING: workspace ${cwd} is not trusted by Claude Code (${trust.stateFile} has no hasTrustDialogAccepted=true for ${trust.requiredKey})${rootHint}. Claude Code ignores project/local permissions.allow and env in untrusted workspaces; open interactive \`claude\` there once to accept trust, or install with --scope user.`);
    }
  }
  out(`Claude Code: installed ${PLUGIN_ID} (scope ${scope}). Start the gateway first, then a Claude Code session.`);
  out("Next: run `ni-rua-plugin doctor` and `ni-rua-plugin gateway --dry-run` (or `--config so101`).");
  out(ESTOP_WARNING);
}

function installCodex(opts, record) {
  const bin = requireExecutable("codex", opts);
  const ch = codexHome();
  if (fs.existsSync(ch) && !fs.statSync(ch).isDirectory()) {
    throw new UserError(`CODEX_HOME ${ch} exists and is not a directory`);
  }
  const key = installKey("codex", opts);
  const prior = record.installs[key];
  const port = resolvePort(opts, record, key);
  const copy = planCopy(port, opts.force);
  const catalogFile = path.join(copy.target, CODEX_DIRECT_CATALOG);
  // 预检 config.toml 可读写且合法，避免写入半残状态
  planCodexConfig(ch, catalogFile, opts, prior?.config);
  if (prior) out(`replacing the earlier Codex install (version ${prior.version})`);
  fs.mkdirSync(ch, { recursive: true });
  const prevPort = record.port;
  const dir = writeCopy(copy, record);
  const env = { CODEX_HOME: ch };
  // Codex copies a local marketplace into its cache and refuses a marketplace of the same name from another source:
  // an upgrade (or a changed port) is remove, then add. This removes any "ni_rua" marketplace already configured.
  out(`removing any existing ${MARKETPLACE} plugin and marketplace first (Codex cannot repoint one)`);
  run(bin, ["plugin", "remove", PLUGIN_ID], { env, allowFail: true });
  run(bin, ["plugin", "marketplace", "remove", MARKETPLACE], { env, allowFail: true });
  // 在 codex plugin marketplace/add 写入 config.toml 之前判定该文件原本是否存在，避免空 CODEX_HOME 卸载残留 0 字节 config.toml
  const cfgPath = codexConfigPath(ch);
  let existedBeforeHost = fs.existsSync(cfgPath);
  if (existedBeforeHost && prior?.config?.created_file) {
    const remaining = stripExactLine(fs.readFileSync(cfgPath, "utf8"), prior.config.line).text;
    if (!remaining.trim()) existedBeforeHost = false;
  }
  try {
    run(bin, ["plugin", "marketplace", "add", dir], { env });
  } catch (error) {
    rollbackUnrecordedCopy(record, prevPort);
    throw error;
  }
  const entry = {
    host: "codex", version: PKG.version, dir, bin, codex_home: ch,
    // 保留旧安装记录中的 config.toml 变更，防止后续 plugin add 失败时丢失可回滚记录
    ...(prior?.config ? { config: prior.config } : {}),
  };
  record.installs[key] = entry;
  saveRecord(record);
  run(bin, ["plugin", "add", PLUGIN_ID], { env });
  // 重新规划并写入 model_catalog_json，避免覆盖 codex plugin add 刚写入的 config.toml 表
  const codexPlan = planCodexConfig(ch, path.join(dir, CODEX_DIRECT_CATALOG), opts, prior?.config, { existedBeforeHost });
  const codexChange = commitCodexConfig(codexPlan);
  if (codexChange) { entry.config = codexChange; saveRecord(record); }
  else if (entry.config) { delete entry.config; saveRecord(record); }
  out(`Codex: installed ${PLUGIN_ID}. Start the gateway before the Codex session: Codex does not retry a session that started without it.`);
  out("Next: run `ni-rua-plugin doctor` and `ni-rua-plugin gateway --dry-run` (or `--config so101`).");
  out(ESTOP_WARNING);
}

function installOpencode(opts, record) {
  const bin = opts["opencode-bin"] ? requireExecutable("opencode", opts) : findExecutable("opencode", opts);
  if (bin) {
    const version = hostVersion(bin);
    const major = version ? Number(version.split(".")[0]) : null;
    if (major !== null && major < 2) {
      throw new UserError(
        `opencode ${version} at ${bin} is 1.x; this plugin requires OpenCode 2.x (verified: ${VERIFIED.opencode.join(", ")}), because 1.x rejects top-level "plugins" and "tool_output" in opencode.json. Install OpenCode 2.2.7 (for example into ~/.local/share/ni-rua/opencode-2.2.7/opencode) and pass --opencode-bin <path>.`,
      );
    }
  } else {
    out("note: opencode was not found on PATH (--opencode-bin to point at it); writing OpenCode 2.x config anyway (OpenCode 1.x is unsupported).");
  }
  const key = installKey("opencode", opts);
  const prior = record.installs[key];
  const port = resolvePort(opts, record, key);
  const copy = planCopy(port, opts.force);
  const plan = planOpencodeConfig(copy.target, prior?.config);
  if (prior) out(`replacing the earlier OpenCode install (version ${prior.version})`);
  const dir = writeCopy(copy, record);
  plan.messages.forEach(out);
  writeJson(plan.change.path, plan.config);
  record.installs[key] = { host: "opencode", version: PKG.version, dir, config: plan.change };
  saveRecord(record);
  out("OpenCode: configured. The plugin loads at the first session in a project; restart OpenCode if it was running.");
  out("Next: run `ni-rua-plugin doctor` and `ni-rua-plugin gateway --dry-run` (or `--config so101`).");
  out(ESTOP_WARNING);
}

function install(host, opts) {
  withLock(() => {
    const record = loadRecord();
    ({ claude: installClaude, codex: installCodex, opencode: installOpencode })[host](opts, record);
    dropUnusedCopies(record);      // an upgrade leaves the previous version's copy unused
    saveRecord(record);
  });
}

// ---------------------------------------------------------------- uninstall
// A host command that fails for a reason other than "not there" keeps the record (and the copy) and exits non-zero.
const ABSENT = /not (configured|installed|found|present)|no such|does(n't| not) exist|unknown (plugin|marketplace)|not a known/i;

function uninstall(host, opts) {
  withLock(() => {
    const record = loadRecord();
    const key = installKey(host, opts);
    const entry = record.installs[key];
    if (!entry) {
      const others = Object.keys(record.installs).filter((k) => record.installs[k].host === host);
      out(`${host}: nothing recorded for ${key.split("|").slice(1).join(" ")}.`);
      if (others.length) out(`recorded ${host} installs (run uninstall with the same CLAUDE_CONFIG_DIR / CODEX_HOME / OPENCODE_CONFIG_DIR / XDG_CONFIG_HOME and scope):\n  ${others.join("\n  ")}`);
      return;
    }
    const failures = [];
    const step = (bin, args, options, manual) => {
      const result = run(bin, args, { ...options, allowFail: true });
      if (!result.ok && !ABSENT.test(result.text)) failures.push(manual);
    };
    if (host === "claude") {
      // 同一个 CLAUDE_CONFIG_DIR 下若仍有其他作用域/项目安装，则保留共享 marketplace，避免连带破坏其他项目
      const otherSameConfigDir = Object.entries(record.installs).some(
        ([k, v]) => k !== key && v.host === "claude" && realPathOrResolve(v.config_dir) === realPathOrResolve(entry.config_dir),
      );
      const bin = findExecutable("claude", opts) ?? (fs.existsSync(entry.bin || "") ? entry.bin : null);
      const manual = otherSameConfigDir
        ? `claude plugin uninstall ${PLUGIN_ID} -s ${entry.scope}`
        : `claude plugin uninstall ${PLUGIN_ID} -s ${entry.scope} && claude plugin marketplace remove ${MARKETPLACE}`;
      if (!bin) failures.push(manual);
      else {
        // user 作用域卸载不依赖安装时目录；project/local 目录已删除时回退当前目录，避免 ENOENT 永久阻塞卸载
        const uninstallCwd = entry.scope === "user"
          ? process.cwd()
          : (entry.cwd && fs.existsSync(entry.cwd) ? entry.cwd : process.cwd());
        const options = { cwd: uninstallCwd, env: { CLAUDE_CONFIG_DIR: entry.config_dir } };
        step(bin, ["plugin", "uninstall", PLUGIN_ID, "-s", entry.scope], options, manual);
        if (!otherSameConfigDir) step(bin, ["plugin", "marketplace", "remove", MARKETPLACE], options, manual);
      }
      // 仅在宿主 CLI 卸载完成后再撤销 settings.json 改动，防止卸载失败留下半残配置
      if (!failures.length) {
        undoClaudeSettings(entry.settings);
        delete entry.settings;
      }
    } else if (host === "codex") {
      const bin = findExecutable("codex", opts) ?? (fs.existsSync(entry.bin || "") ? entry.bin : null);
      const manual = `CODEX_HOME=${entry.codex_home} codex plugin remove ${PLUGIN_ID} && codex plugin marketplace remove ${MARKETPLACE}`;
      if (!bin) failures.push(manual);
      else {
        step(bin, ["plugin", "remove", PLUGIN_ID], { env: { CODEX_HOME: entry.codex_home } }, manual);
        step(bin, ["plugin", "marketplace", "remove", MARKETPLACE], { env: { CODEX_HOME: entry.codex_home } }, manual);
      }
      if (!failures.length) {
        undoCodexConfig(entry.config);
        delete entry.config;
      }
    } else undoOpencodeConfig(entry.config);
    if (failures.length) {
      saveRecord(record);
      throw new UserError(`${host}: could not finish removing; the record and the copy are kept. Run by hand, then uninstall again:\n  ${[...new Set(failures)].join("\n  ")}`);
    }
    delete record.installs[key];
    dropUnusedCopies(record);
    saveRecord(record);
    out(`${host}: removed.`);
  });
}

// ---------------------------------------------------------------- doctor
const semver = (text) => /\d+\.\d+\.\d+/.exec(text)?.[0] ?? null;

function hostVersion(bin) {
  const result = capture(bin, ["--version"]);
  return semver(result.stdout) ?? semver(result.stderr);
}

function parseSse(text) {
  const events = text.split(/\r?\n\r?\n/).map((block) => block.split(/\r?\n/).filter((l) => l.startsWith("data:")).map((l) => l.slice(5).trimStart()).join("\n")).filter(Boolean);
  return events.map((e) => { try { return JSON.parse(e); } catch { return null; } }).filter(Boolean);
}

async function mcpPost(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json", accept: "application/json, text/event-stream", "mcp-protocol-version": "2025-06-18" },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(5000),
  });
  const text = await response.text();
  if (!response.ok && response.status !== 202) throw new Error(`HTTP ${response.status}`);
  if (!text.trim()) return null;
  const messages = (response.headers.get("content-type") || "").includes("event-stream") ? parseSse(text) : [JSON.parse(text)];
  return messages.find((m) => m.id === body.id) ?? messages[0] ?? null;
}

async function probeGateway(target) {
  const url = typeof target === "string" && /^https?:\/\//i.test(target) ? target : `http://127.0.0.1:${target}/mcp`;
  const init = await mcpPost(url, { jsonrpc: "2.0", id: 1, method: "initialize", params: { protocolVersion: "2025-06-18", capabilities: {}, clientInfo: { name: "ni-rua-plugin-doctor", version: PKG.version } } });
  if (!init?.result) throw new Error("initialize returned no result");
  await mcpPost(url, { jsonrpc: "2.0", method: "notifications/initialized" }).catch(() => null);
  const listed = await mcpPost(url, { jsonrpc: "2.0", id: 2, method: "tools/list", params: {} });
  const tools = listed?.result?.tools;
  if (!Array.isArray(tools)) throw new Error("tools/list returned no tools");
  return { names: tools.map((t) => t.name), instructions: Boolean(init.result.instructions), server: init.result.serverInfo };
}

function checkCodexDirectCatalog(line, record) {
  const cfgPath = codexConfigPath();
  const text = fs.existsSync(cfgPath) && fs.statSync(cfgPath).isFile() ? fs.readFileSync(cfgPath, "utf8") : "";
  const top = parseCodexTopLevel(text);
  const durableCatalog = path.join(dataRoot(), PKG.version, CODEX_DIRECT_CATALOG);
  if (top.model_catalog_json) {
    const resolved = path.isAbsolute(top.model_catalog_json)
      ? top.model_catalog_json
      : path.resolve(path.dirname(cfgPath), top.model_catalog_json);
    if (!fs.existsSync(resolved)) {
      line("warn", `codex: model_catalog_json in ${cfgPath} points to missing file ${resolved}`);
      return;
    }
    try {
      const parsed = JSON.parse(fs.readFileSync(resolved, "utf8"));
      const models = Array.isArray(parsed?.models) ? parsed.models : [];
      const bad = top.model
        ? models.find((m) => m?.slug === top.model && m?.tool_mode === "code_mode_only")
        : models.find((m) => m?.tool_mode === "code_mode_only");
      const searchEnabled = top.model
        ? models.find((m) => m?.slug === top.model && m?.supports_search_tool === true)
        : models.find((m) => m?.supports_search_tool === true);
      const missingCodeModeOverride = Boolean(
        top.model
        && CODEX_CODE_MODE_ONLY.has(top.model)
        && !models.some((m) => m?.slug === top.model),
      );
      if (bad) {
        line("warn", `codex: model_catalog_json (${resolved}) still marks ${bad.slug} as tool_mode="code_mode_only"; on Linux this triggers code-mode host crashes`);
      } else if (searchEnabled) {
        line("warn", `codex: model_catalog_json (${resolved}) still sets supports_search_tool=true on ${searchEnabled.slug}; MCP tools stay hidden behind tool_search (re-run \`ni-rua-plugin install codex --force\`)`);
      } else if (missingCodeModeOverride) {
        line("warn", `codex: model_catalog_json (${resolved}) has no entry for ${top.model}; built-in tool_mode="code_mode_only" may still apply on Linux`);
      } else if (!models.length) {
        line("warn", `codex: model_catalog_json (${resolved}) has an empty models list`);
      } else {
        line("ok", `codex: model_catalog_json set (${resolved}) for direct MCP tool delivery`);
      }
    } catch (error) {
      line("warn", `codex: model_catalog_json (${resolved}) could not be parsed (${error.message})`);
    }
    return;
  }
  const hasCodeModeModel = Boolean(top.model && CODEX_CODE_MODE_ONLY.has(top.model));
  if (hasCodeModeModel || process.platform === "linux") {
    const detail = hasCodeModeModel ? ` (model=${top.model} uses built-in tool_mode="code_mode_only")` : "";
    const hintPath = Object.values(record.installs).find((e) => e.host === "codex")?.dir
      ? path.join(Object.values(record.installs).find((e) => e.host === "codex").dir, CODEX_DIRECT_CATALOG)
      : durableCatalog;
    line("warn", `codex: ${cfgPath} has no model_catalog_json${detail}; on Linux code_mode_only models crash the code-mode host on plugin MCP calls. Re-run \`ni-rua-plugin install codex\` or pass -c 'model_catalog_json="${hintPath}"'`);
  }
}

async function doctor(opts) {
  let failed = false;
  const jsonMode = Boolean(opts.json);
  const checks = [];
  const line = (status, text) => {
    checks.push({ status, message: text });
    if (!jsonMode) out(`[${status}] ${text}`);
    if (status === "FAIL") failed = true;
  };
  const record = loadRecord();
  const nodeMajor = Number(process.versions.node.split(".")[0]);
  line(nodeMajor >= 18 ? "ok" : "FAIL", `node ${process.versions.node} (needs >= 18)`);
  if (process.platform === "win32") line("warn", "Windows is supported only through WSL (the gateway uses fcntl)");
  line("info", `package ${PKG.name} ${PKG.version}; durable copies in ${dataRoot()}`);

  let opencodeMajor = null;
  let opencodeVersionStr = null;
  for (const host of HOSTS) {
    const bin = findExecutable(host, opts);
    if (!bin) { line("warn", `${host}: not found on PATH (--${host}-bin <path>)`); continue; }
    const version = hostVersion(bin);
    const verified = VERIFIED[host];
    if (host === "opencode") {
      opencodeVersionStr = version;
      opencodeMajor = version ? Number(version.split(".")[0]) : null;
      if (opencodeMajor !== null && opencodeMajor < 2) {
        line("warn", `opencode ${version} at ${bin} is 1.x (unsupported; requires OpenCode 2.x, verified versions: ${verified.join(", ")}; 1.x rejects top-level plugins/tool_output)`);
      } else {
        line(version && verified.includes(version) ? "ok" : "warn", `${host} ${version ?? "unknown version"} at ${bin}${version && verified.includes(version) ? " (verified)" : ` (verified versions: ${verified.join(", ")})`}`);
      }
      continue;
    }
    line(version && verified.includes(version) ? "ok" : "warn", `${host} ${version ?? "unknown version"} at ${bin}${version && verified.includes(version) ? " (verified)" : ` (verified versions: ${verified.join(", ")})`}`);
    if (host === "claude") {
      const cwd = realPathOrResolve(process.cwd());
      const listed = capture(bin, ["plugin", "list", "--json"], { env: { CLAUDE_CONFIG_DIR: claudeConfigDir() } });
      let found = null;
      try { found = selectClaudePluginEntry(JSON.parse(listed.stdout), cwd); } catch { /* reported below */ }
      line(found && found.enabled ? "ok" : "warn", found
        ? `claude: ${PLUGIN_ID} listed (scope ${found.scope}, ${found.enabled ? "enabled" : "disabled"})`
        : `claude: ${PLUGIN_ID} is not listed; run: ni-rua-plugin install claude`);
      const projectFile = claudeSettingsPath("project", cwd);
      const localFile = claudeSettingsPath("local", cwd);
      const files = [claudeSettingsPath("user", cwd), projectFile, localFile].filter((f) => fs.existsSync(f));
      const merged = { env: {}, allow: [] };
      for (const file of files) {
        try { const s = readJson(file, ""); Object.assign(merged.env, s.env ?? {}); merged.allow.push(...(s.permissions?.allow ?? [])); } catch (error) { line("warn", `claude: ${error.message}`); }
      }
      const missing = Object.keys(CLAUDE_ENV).filter((key) => !(key in merged.env));
      const low = lowEnvValues(merged.env);
      if (missing.length) line("warn", `claude: env keys not set: ${missing.join(", ")} (long results or calls may be cut; install without --no-host-env)`);
      if (low.length) line("warn", `claude: env values lower than the plugin needs (kept as you set them): ${low.join("; ")}`);
      if (!missing.length && !low.length) line("ok", "claude: the four MCP env keys are set");
      line(merged.allow.includes(CLAUDE_ALLOW) ? "ok" : "warn", merged.allow.includes(CLAUDE_ALLOW) ? `claude: permissions.allow has ${CLAUDE_ALLOW}` : `claude: permissions.allow lacks ${CLAUDE_ALLOW} (robot calls will ask for approval or be refused)`);
      if ((found && found.scope !== "user") || fs.existsSync(projectFile) || fs.existsSync(localFile)) {
        const trust = checkClaudeProjectTrust(cwd);
        if (trust.trusted) {
          line("ok", `claude: workspace trust accepted (${trust.trustedKey})`);
        } else {
          const rootHint = trust.isSubdirOfGit
            ? (trust.subdirTrustedOnly
              ? `; ${cwd} is trusted, but Claude Code checks the enclosing .git root ${trust.gitRoot}`
              : `; Claude Code resolves trust at enclosing .git root ${trust.gitRoot}`)
            : "";
          line("warn", `claude: workspace ${cwd} is not trusted in ${trust.stateFile} (missing hasTrustDialogAccepted=true for ${trust.requiredKey}; project/local permissions.allow and env are ignored until you open interactive claude there and accept trust${rootHint})`);
        }
      }
    } else if (host === "codex") {
      const listed = capture(bin, ["plugin", "list", "--json"], { env: { CODEX_HOME: codexHome() } });
      let found = null;
      try { found = JSON.parse(listed.stdout).installed.find((p) => p.pluginId === PLUGIN_ID); } catch { /* reported below */ }
      line(found && found.enabled ? "ok" : "warn", found
        ? `codex: ${PLUGIN_ID} listed (${found.enabled ? "enabled" : "disabled"})`
        : `codex: ${PLUGIN_ID} is not listed; run: ni-rua-plugin install codex`);
      checkCodexDirectCatalog(line, record);
    }
  }
  // OpenCode's own `plugin list` is not a reliable load check; the config entry is.
  try {
    const ocPath = opencodeConfigPath();
    const siblingJsonc = path.join(opencodeConfigDir(), "opencode.jsonc");
    if (fs.existsSync(siblingJsonc)) {
      line("warn", `opencode: ${siblingJsonc} exists (JSONC); doctor checks only strict JSON ${ocPath} — verify plugins, tool_output and media.image in ${siblingJsonc} by hand`);
    }
    const config = readJson(ocPath, "") ?? {};
    const mine = (config.plugins ?? []).filter((p) => typeof p === "string" && p.startsWith(dataRoot() + path.sep));
    line(mine.length ? "ok" : "warn", mine.length ? `opencode: config ${ocPath} lists ${mine.join(", ")}` : `opencode: no NI-RUA plugin in ${ocPath}; run: ni-rua-plugin install opencode`);
    if (mine.length && !opencodeToolOutputSufficient(config.tool_output)) {
      line("info", `opencode: tool_output is ${JSON.stringify(config.tool_output)}; results over about 50 KB are cut unless it is at least ${JSON.stringify(OPENCODE_TOOL_OUTPUT)}`);
    }
    if (mine.length && !opencodeMediaImageSufficient(config.media?.image)) {
      line("info", `opencode: media.image is ${JSON.stringify(config.media?.image)}; large PNG tool images may be re-encoded to JPEG unless it matches ${JSON.stringify(OPENCODE_MEDIA_IMAGE)}`);
    }
    if (mine.length && opencodeMajor !== null && opencodeMajor < 2) {
      line("warn", `opencode: ${ocPath} uses 2.x keys (plugins/tool_output) which opencode ${opencodeVersionStr} (1.x) rejects; run with OpenCode 2.2.7`);
    }
  } catch (error) { line("warn", `opencode: ${error.message}`); }

  runStackChecks({ line, so101: Boolean(opts.so101), cwd: process.cwd() });

  const port = opts.port ?? record.port ?? DEFAULT_PORT;
  const customUrl = opts["gateway-url"];
  const gatewayUrl = customUrl ?? `http://127.0.0.1:${port}/mcp`;
  const gatewayLabel = customUrl ?? `127.0.0.1:${port}`;
  try {
    const probe = await probeGateway(gatewayUrl);
    line("ok", `gateway on ${gatewayLabel}: MCP initialize + tools/list answered, ${probe.names.length} tools${probe.instructions ? ", instructions present" : ""}`);
    line(probe.names.includes("run_begin") ? "ok" : "warn", probe.names.includes("run_begin") ? "gateway offers run_begin (external backend)" : "gateway has no run_begin tool: it was not started with --agent-backend external");
  } catch (error) {
    line("warn", `no MCP gateway answered on ${gatewayLabel} (${error.cause?.code ?? error.message}). Start one before the agent session:\n       ${GATEWAY_HINT}\n       (the MCP port is agent.external.mcp_port / --external-mcp-port, default ${DEFAULT_PORT})`);
  }
  if (jsonMode) {
    out(JSON.stringify({ ok: !failed, package: PKG.name, version: PKG.version, gateway_url: gatewayUrl, checks }, null, 2));
  }
  return failed ? 1 : 0;
}

// ---------------------------------------------------------------- main
const USAGE = `ni-rua-plugin ${PKG.version} (${PKG.name})
usage:
  ni-rua-plugin setup <claude|codex|opencode|all> [--so101] [--skip-gateway] [--scope user|project|local] [--port N] [--force]
  ni-rua-plugin gateway [--dry-run] [--config <preset|path>] [--port N] [--run-dir D] [--prompt T] [-- ...extra]
  ni-rua-plugin install <claude|codex|opencode> [--scope user|project|local] [--port N] [--no-host-env] [--no-allow] [--no-direct-catalog] [--force]
  ni-rua-plugin uninstall <claude|codex|opencode> [--scope user|project|local]
  ni-rua-plugin doctor [--port N] [--gateway-url URL] [--so101] [--json]
  host executables: --claude-bin P, --codex-bin P, --opencode-bin P (default: first on PATH)
  --scope applies to Claude Code only; --no-host-env skips the four Claude Code env keys; --no-allow skips the
  permissions.allow entry; --no-direct-catalog skips writing model_catalog_json into Codex config.toml; --force
  replaces a durable copy or recreates managed Python envs.`;

async function main(argv) {
  const opts = parseArgs(argv);
  const [command, host] = opts._;
  if (opts.help || !command) { out(USAGE); return opts.help ? 0 : 1; }
  if (command === "doctor") return doctor(opts);
  if (command === "gateway") {
    return runGateway({ opts, extraArgs: opts.extra, defaultMcpPort: DEFAULT_PORT, version: PKG.version, out, UserError });
  }
  if (command === "setup") {
    if (![...HOSTS, "all"].includes(host)) throw new UserError(`setup needs a host: ${HOSTS.join(", ")}, or all`);
    if (process.platform === "win32") throw new UserError("Windows is supported only through WSL");
    if (opts.scope !== undefined && host !== "claude" && host !== "all") throw new UserError("--scope applies to Claude Code only");
    if (opts["no-direct-catalog"] && host !== "codex" && host !== "all") throw new UserError("--no-direct-catalog applies to Codex only");
    runSetup({ target: host, opts, hosts: HOSTS, findHostExecutable: findExecutable, installHost: install, out, run, UserError });
    return 0;
  }
  if (command !== "install" && command !== "uninstall") throw new UserError(`unknown command ${command}\n${USAGE}`);
  if (!HOSTS.includes(host)) throw new UserError(`${command} needs a host: ${HOSTS.join(", ")}`);
  if (process.platform === "win32") throw new UserError("Windows is supported only through WSL");
  if (opts.scope !== undefined && host !== "claude") throw new UserError("--scope applies to Claude Code only");
  if (opts["no-direct-catalog"] && host !== "codex") throw new UserError("--no-direct-catalog applies to Codex only");
  if (command === "install") install(host, opts); else uninstall(host, opts);
  return 0;
}

main(process.argv.slice(2)).then((code) => { process.exitCode = code; }, (error) => {
  if (error instanceof UserError) { console.error(`error: ${error.message}`); process.exitCode = 2; }
  else { console.error(error); process.exitCode = 1; }
});
