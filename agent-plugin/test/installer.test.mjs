import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { pathToFileURL } from "node:url";
import { copyPackage, sandbox } from "./helpers.mjs";

const ENV_KEYS = ["MAX_MCP_OUTPUT_TOKENS", "MCP_TOOL_TIMEOUT", "CLAUDE_CODE_MCP_AUTO_BACKGROUND_MS", "CLAUDE_CODE_MAX_MCP_DESCRIPTION_LENGTH"];
const ALLOW = "mcp__plugin_ni_rua_ni_rua";
const version = JSON.parse(fs.readFileSync(new URL("../package.json", import.meta.url), "utf8")).version;

function withBox(fn) {
  return async (t) => {
    const box = sandbox();
    try { await fn(box, t); } finally { box.cleanup(); }
  };
}

test("claude install: host commands, four env keys, allow entry, record", withBox((box) => {
  const result = box.run(["install", "claude"]);
  assert.equal(result.status, 0, result.out);
  const durable = path.join(box.durable, version);
  assert.deepEqual(box.commands(), [
    `claude plugin marketplace add ${durable}`,
    "claude plugin install ni_rua@ni_rua -s user",
  ]);
  assert.ok(result.stdout.includes(`+ ${path.join(box.bin, "claude")} plugin marketplace add`), "commands are printed before they run");
  const settings = box.read(box.claudeSettings);
  assert.deepEqual(Object.keys(settings.env).sort(), [...ENV_KEYS].sort());
  assert.ok(!("CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT" in settings.env));
  assert.deepEqual(settings.permissions.allow, [ALLOW]);
  const record = box.read(path.join(box.durable, "installed.json"));
  assert.equal(Object.values(record.installs)[0].dir, durable);
  assert.deepEqual(record.copies, [version]);
  for (const file of ["package.json", ".mcp.json", ".codex-mcp.json", "manifest.generated.json", ".claude-plugin/plugin.json", "opencode/plugin.js", "skills/ni-rua-robot/SKILL.md", "bin/cli.mjs"]) {
    assert.ok(fs.existsSync(path.join(durable, file)), file);
  }
  assert.ok(!fs.existsSync(path.join(durable, "test")), "tests are not part of the package");
}));

test("claude settings merge keeps user values and uninstall restores the file exactly", withBox((box) => {
  const before = { theme: "dark", env: { MCP_TOOL_TIMEOUT: "5", OTHER: "x" }, permissions: { allow: ["Bash(ls)"], deny: ["Bash(rm:*)"] } };
  box.write(box.claudeSettings, before);
  assert.equal(box.run(["install", "claude"]).status, 0);
  const merged = box.read(box.claudeSettings);
  assert.equal(merged.env.MCP_TOOL_TIMEOUT, "5", "a key the user set is skipped");
  assert.equal(merged.env.MAX_MCP_OUTPUT_TOKENS, "2000000");
  assert.deepEqual(merged.permissions.allow, ["Bash(ls)", ALLOW]);
  assert.equal(merged.theme, "dark");
  const result = box.run(["uninstall", "claude"]);
  assert.equal(result.status, 0, result.out);
  assert.deepEqual(box.own(box.claudeSettings), before);
  assert.ok(box.commands().slice(-2).join("|").includes("plugin uninstall ni_rua@ni_rua -s user"));
  assert.ok(box.commands().at(-1).endsWith("plugin marketplace remove ni_rua"));
  assert.ok(!fs.existsSync(box.durable), "durable copy and record are removed with the last host");
}));

test("claude install into a home without settings leaves no settings file after uninstall", withBox((box) => {
  assert.equal(box.run(["install", "claude"]).status, 0);
  assert.ok(fs.existsSync(box.claudeSettings));
  assert.equal(box.run(["uninstall", "claude"]).status, 0);
  assert.deepEqual(box.own(box.claudeSettings), {}, "only the host's own keys are left");
}));

test("a value the user changed after install is left alone by uninstall", withBox((box) => {
  assert.equal(box.run(["install", "claude"]).status, 0);
  const settings = box.read(box.claudeSettings);
  settings.env.MCP_TOOL_TIMEOUT = "42";
  box.write(box.claudeSettings, settings);
  const result = box.run(["uninstall", "claude"]);
  assert.match(result.out, /MCP_TOOL_TIMEOUT was changed since install; left as is/);
  assert.deepEqual(box.own(box.claudeSettings), { env: { MCP_TOOL_TIMEOUT: "42" } });
}));

test("install is idempotent", withBox((box) => {
  box.write(box.claudeSettings, { env: { KEEP: "1" } });
  assert.equal(box.run(["install", "claude"]).status, 0);
  const first = fs.readFileSync(box.claudeSettings, "utf8");
  const record = fs.readFileSync(path.join(box.durable, "installed.json"), "utf8");
  assert.equal(box.run(["install", "claude"]).status, 0);
  assert.equal(fs.readFileSync(box.claudeSettings, "utf8"), first);
  assert.equal(fs.readFileSync(path.join(box.durable, "installed.json"), "utf8"), record);
  assert.equal(box.read(box.claudeSettings).permissions.allow.length, 1);
  assert.equal(box.run(["uninstall", "claude"]).status, 0);
  assert.deepEqual(box.own(box.claudeSettings), { env: { KEEP: "1" } });
}));

test("--no-host-env and --no-allow", withBox((box) => {
  assert.equal(box.run(["install", "claude", "--no-host-env"]).status, 0);
  let settings = box.read(box.claudeSettings);
  assert.equal(settings.env, undefined);
  assert.deepEqual(settings.permissions.allow, [ALLOW]);
  assert.equal(box.run(["uninstall", "claude"]).status, 0);
  assert.equal(box.run(["install", "claude", "--no-allow"]).status, 0);
  settings = box.read(box.claudeSettings);
  assert.equal(settings.permissions, undefined);
  assert.equal(Object.keys(settings.env).length, 4);
  assert.equal(box.run(["uninstall", "claude"]).status, 0);
  assert.equal(box.run(["install", "claude", "--no-allow", "--no-host-env"]).status, 0);
  assert.deepEqual(box.own(box.claudeSettings), {}, "the installer writes no settings at all");
}));

test("claude --scope project writes the project settings and runs the host command there", withBox((box) => {
  const project = path.join(box.root, "project");
  fs.mkdirSync(project);
  const result = box.run(["install", "claude", "--scope", "project"], { cwd: project });
  assert.equal(result.status, 0, result.out);
  assert.equal(box.calls()[1].cwd, project);
  assert.deepEqual(box.commands()[1], "claude plugin install ni_rua@ni_rua -s project");
  assert.ok(box.read(path.join(project, ".claude", "settings.json")).permissions.allow.includes(ALLOW));
  assert.ok(!fs.existsSync(box.claudeSettings));
  assert.match(box.run(["uninstall", "claude"], { cwd: box.root }).out, /nothing recorded/, "another scope or project is another install");
  assert.equal(box.run(["uninstall", "claude", "--scope", "project"], { cwd: project }).status, 0);
  assert.equal(box.calls().at(-2).cwd, project, "uninstall runs in the recorded project");
  assert.deepEqual(box.own(path.join(project, ".claude", "settings.json")), {});
}));

test("codex install is remove, marketplace remove, marketplace add, add; uninstall removes both", withBox((box) => {
  const result = box.run(["install", "codex"], { extraEnv: { FAKE_FAIL: "plugin remove,plugin marketplace remove" } });
  assert.equal(result.status, 0, result.out);
  const durable = path.join(box.durable, version);
  assert.deepEqual(box.commands(), [
    "codex plugin remove ni_rua@ni_rua",
    "codex plugin marketplace remove ni_rua",
    `codex plugin marketplace add ${durable}`,
    "codex plugin add ni_rua@ni_rua",
  ]);
  assert.ok(!fs.existsSync(box.claudeSettings), "codex touches no Claude settings");
  box.clearCalls();
  assert.equal(box.run(["uninstall", "codex"]).status, 0);
  assert.deepEqual(box.commands(), ["codex plugin remove ni_rua@ni_rua", "codex plugin marketplace remove ni_rua"]);
  assert.ok(!fs.existsSync(box.durable));
}));

test("opencode install writes plugins, tool_output and media.image; uninstall restores the file", withBox((box) => {
  const before = { theme: "x", tool_output: undefined, plugins: ["some-other-plugin"] };
  delete before.tool_output;
  box.write(box.opencodeConfig, before);
  const result = box.run(["install", "opencode"]);
  assert.equal(result.status, 0, result.out);
  const durable = path.join(box.durable, version);
  const config = box.read(box.opencodeConfig);
  assert.deepEqual(config.plugins, ["some-other-plugin", durable]);
  assert.deepEqual(config.tool_output, { max_lines: 100000, max_bytes: 16000000 });
  assert.deepEqual(config.media, {
    image: { auto_resize: false, max_width: 1000000, max_height: 1000000, max_base64_bytes: 1073741824 },
  });
  assert.equal(box.run(["uninstall", "opencode"]).status, 0);
  assert.deepEqual(box.read(box.opencodeConfig), before);
}));

test("opencode keeps user tool_output/media.image and removes the file it created", withBox((box) => {
  box.write(box.opencodeConfig, { tool_output: { max_lines: 5, max_bytes: 5 }, media: { image: { auto_resize: true } } });
  assert.equal(box.run(["install", "opencode"]).status, 0);
  assert.deepEqual(box.read(box.opencodeConfig).tool_output, { max_lines: 5, max_bytes: 5 });
  assert.deepEqual(box.read(box.opencodeConfig).media, { image: { auto_resize: true } });
  assert.equal(box.run(["uninstall", "opencode"]).status, 0);
  assert.deepEqual(box.read(box.opencodeConfig), { tool_output: { max_lines: 5, max_bytes: 5 }, media: { image: { auto_resize: true } } });
  fs.rmSync(box.opencodeConfig);
  assert.equal(box.run(["install", "opencode"]).status, 0);
  assert.equal(box.run(["uninstall", "opencode"]).status, 0);
  assert.ok(!fs.existsSync(box.opencodeConfig));
}));

test("opencode refuses JSONC and unparsable config, with the manual edit", withBox((box) => {
  const jsonc = '{\n  // my plugins\n  "plugins": []\n}\n';
  box.write(box.opencodeConfig, jsonc);
  const result = box.run(["install", "opencode"]);
  assert.equal(result.status, 2);
  assert.match(result.stderr, /not strict JSON/);
  assert.match(result.stderr, /"plugins":\["[^"]*agent-plugin[^"]*"\]/, "names the manual edit");
  assert.match(result.stderr, /"tool_output":\{"max_lines":100000,"max_bytes":16000000\}/);
  assert.equal(fs.readFileSync(box.opencodeConfig, "utf8"), jsonc, "file untouched");
  assert.ok(!fs.existsSync(path.join(box.durable, "installed.json")), "no record of a failed install");
  fs.rmSync(box.opencodeConfig);
  box.write(path.join(path.dirname(box.opencodeConfig), "opencode.jsonc"), "{}");
  const sibling = box.run(["install", "opencode"]);
  assert.equal(sibling.status, 2);
  assert.match(sibling.stderr, /opencode\.jsonc exists/);
}));

test("--port rewrites the URL of the three entries in the durable copy only", withBox(async (box) => {
  assert.equal(box.run(["install", "codex", "--port", "9123"], { extraEnv: { FAKE_FAIL: "plugin remove,plugin marketplace remove" } }).status, 0);
  const durable = path.join(box.durable, version);
  const url = "http://127.0.0.1:9123/mcp";
  assert.equal(box.read(path.join(durable, ".mcp.json")).mcpServers.ni_rua.url, url);
  assert.equal(box.read(path.join(durable, ".codex-mcp.json")).mcpServers.ni_rua.url, url);
  assert.equal(box.read(path.join(durable, "manifest.generated.json")).mcp_url, url);
  const loaded = await import(pathToFileURL(path.join(durable, "opencode", "plugin.js")).href);
  const mcpEntries = new Map();
  const addedSkills = [];
  await loaded.default.setup({
    mcp: { transform: async (fn) => fn(mcpEntries) },
    skill: { transform: async (fn) => fn({ add: (s) => addedSkills.push(s.id) }) },
  });
  assert.equal(mcpEntries.get("ni_rua").url, url);
  assert.deepEqual(addedSkills, ["ni-rua-robot", "ni-rua-experiments"]);
  assert.equal(box.read(new URL("../.mcp.json", import.meta.url).pathname).mcpServers.ni_rua.url, "http://127.0.0.1:8790/mcp", "the source package is not modified");
  // a reinstall without --port keeps the recorded port; an explicit --port is not a content conflict
  assert.equal(box.run(["install", "codex"], { extraEnv: { FAKE_FAIL: "plugin remove" } }).status, 0);
  assert.equal(box.read(path.join(durable, ".mcp.json")).mcpServers.ni_rua.url, url);
  assert.equal(box.run(["install", "codex", "--port", "8790"], { extraEnv: { FAKE_FAIL: "plugin remove" } }).status, 0);
  assert.equal(box.read(path.join(durable, ".mcp.json")).mcpServers.ni_rua.url, "http://127.0.0.1:8790/mcp");
  assert.equal(box.run(["install", "codex", "--port", "70000"]).status, 2);
}));

test("same version with different content is refused unless --force", withBox((box) => {
  assert.equal(box.run(["install", "opencode"]).status, 0);
  const skill = path.join(box.durable, version, "skills", "ni-rua-robot", "SKILL.md");
  fs.appendFileSync(skill, "\nlocal edit\n");
  const refused = box.run(["install", "opencode"]);
  assert.equal(refused.status, 2);
  assert.match(refused.stderr, /different content.*--force/);
  assert.match(fs.readFileSync(skill, "utf8"), /local edit/, "nothing was overwritten");
  assert.equal(box.run(["install", "opencode", "--force"]).status, 0);
  assert.ok(!/local edit/.test(fs.readFileSync(skill, "utf8")));
}));

test("upgrade N to N+1 replaces the host entries and drops the old copy", withBox((box) => {
  const failing = { FAKE_FAIL: "plugin remove,plugin marketplace remove" };
  assert.equal(box.run(["install", "claude"]).status, 0);
  assert.equal(box.run(["install", "codex"], { extraEnv: failing }).status, 0);
  assert.equal(box.run(["install", "opencode"]).status, 0);
  const next = copyPackage(path.join(box.root, "pkg-next"), "0.1.1", (dir) => fs.appendFileSync(path.join(dir, "skills", "ni-rua-robot", "SKILL.md"), "\nnext release\n"));
  box.clearCalls();
  for (const host of ["claude", "codex", "opencode"]) {
    const result = box.run(["install", host], { cli: next, extraEnv: failing });
    assert.equal(result.status, 0, result.out);
  }
  const newDir = path.join(box.durable, "0.1.1");
  assert.deepEqual(box.commands(), [
    `claude plugin marketplace add ${newDir}`,
    "claude plugin install ni_rua@ni_rua -s user",
    "codex plugin remove ni_rua@ni_rua",
    "codex plugin marketplace remove ni_rua",
    `codex plugin marketplace add ${newDir}`,
    "codex plugin add ni_rua@ni_rua",
  ]);
  assert.ok(!fs.existsSync(path.join(box.durable, version)), "previous durable copy removed once nothing points at it");
  assert.deepEqual(box.read(box.opencodeConfig).plugins, [newDir]);
  assert.equal(box.read(box.claudeSettings).permissions.allow.length, 1);
  const record = box.read(path.join(box.durable, "installed.json"));
  assert.deepEqual(Object.values(record.installs).map((h) => h.version), ["0.1.1", "0.1.1", "0.1.1"]);
  assert.deepEqual(record.copies, ["0.1.1"]);
  for (const host of ["claude", "codex", "opencode"]) assert.equal(box.run(["uninstall", host], { cli: next }).status, 0);
  assert.ok(!fs.existsSync(box.durable));
  assert.ok(!fs.existsSync(box.opencodeConfig));
}));

test("missing host executable is a clear error and records nothing", withBox((box) => {
  const result = box.run(["install", "claude"], { extraEnv: { PATH: "/usr/bin:/bin" } });
  assert.equal(result.status, 2);
  assert.match(result.stderr, /claude was not found on PATH.*--claude-bin/);
  assert.ok(!fs.existsSync(box.durable));
  const override = box.run(["install", "claude", "--claude-bin", path.join(box.bin, "claude")], { extraEnv: { PATH: "/usr/bin:/bin" } });
  assert.equal(override.status, 0, override.out);
}));

test("a failing host command aborts the install but the record allows an exact uninstall", withBox((box) => {
  const result = box.run(["install", "claude"], { extraEnv: { FAKE_FAIL: "plugin install" } });
  assert.equal(result.status, 2);
  assert.deepEqual(box.own(box.claudeSettings), {}, "settings are written only after the plugin installed");
  assert.equal(box.run(["uninstall", "claude"]).status, 0);
  assert.ok(!fs.existsSync(box.durable));
}));

test("the package files set excludes tests and the manifests agree on version and URL", () => {
  const pkg = JSON.parse(fs.readFileSync(new URL("../package.json", import.meta.url), "utf8"));
  assert.ok(!pkg.files.some((f) => f.startsWith("test")));
  for (const manifest of [".claude-plugin/plugin.json", ".codex-plugin/plugin.json"]) {
    assert.equal(JSON.parse(fs.readFileSync(new URL(`../${manifest}`, import.meta.url), "utf8")).version, pkg.version, manifest);
  }
  assert.equal(pkg.name, "ni-rua-agent-plugin");
  assert.deepEqual(pkg.bin, { "ni-rua-plugin": "bin/cli.mjs" });
  assert.deepEqual(pkg.exports, { ".": "./opencode/plugin.js" });
  assert.equal(pkg.type, "module");
  assert.ok(!pkg.dependencies && !pkg.devDependencies);
  assert.equal(pkg.engines.node, ">=18");
});

const stat = (file) => fs.statSync(file).mode & 0o7777;

test("M1: a second host never repoints the first host's port", withBox((box) => {
  assert.equal(box.run(["install", "opencode", "--port", "9000"]).status, 0);
  assert.equal(box.run(["install", "claude"]).status, 0);
  const manifestPath = path.join(box.durable, version, "manifest.generated.json");
  assert.equal(box.read(manifestPath).mcp_url, "http://127.0.0.1:9000/mcp");
  assert.equal(box.read(path.join(box.durable, "installed.json")).port, 9000);
  const conflict = box.run(["install", "codex", "--port", "9001"]);
  assert.equal(conflict.status, 2);
  assert.match(conflict.stderr, /port 9000 is in use by other installs.*--port 9000/);
  assert.equal(box.read(manifestPath).mcp_url, "http://127.0.0.1:9000/mcp");
}));

test("M2: a refused reinstall leaves settings, config and record byte-identical", withBox((box) => {
  assert.equal(box.run(["install", "opencode"]).status, 0);
  assert.equal(box.run(["install", "claude"]).status, 0);
  fs.appendFileSync(path.join(box.durable, version, "skills", "ni-rua-robot", "SKILL.md"), "\nedit\n");
  const files = [box.claudeSettings, box.opencodeConfig, path.join(box.durable, "installed.json")];
  const before = files.map((f) => fs.readFileSync(f, "utf8"));
  for (const host of ["opencode", "claude"]) {
    const result = box.run(["install", host]);
    assert.equal(result.status, 2);
    assert.match(result.stderr, /nothing was changed/);
  }
  assert.deepEqual(files.map((f) => fs.readFileSync(f, "utf8")), before);
  const broken = '{"theme": // comment\n}';
  box.write(box.claudeSettings, broken);
  fs.writeFileSync(path.join(box.durable, "extra"), "");   // different content again is irrelevant: the settings parse fails first or the copy does
  assert.equal(box.run(["install", "claude", "--force"]).status, 2);
  assert.equal(fs.readFileSync(box.claudeSettings, "utf8"), broken);
}));

test("M3: two config homes are two installs; uninstall under another home touches nothing", withBox((box) => {
  const homeA = path.join(box.root, "claude-a");
  const homeB = path.join(box.root, "claude-b");
  const env = (dir) => ({ extraEnv: { CLAUDE_CONFIG_DIR: dir } });
  assert.equal(box.run(["install", "claude"], env(homeA)).status, 0);
  assert.equal(box.run(["install", "claude"], env(homeB)).status, 0);
  assert.ok(box.read(path.join(homeA, "settings.json")).permissions.allow.includes(ALLOW), "A keeps its settings");
  assert.ok(box.read(path.join(homeB, "settings.json")).permissions.allow.includes(ALLOW));
  assert.equal(Object.keys(box.read(path.join(box.durable, "installed.json")).installs).length, 2);
  const other = box.run(["uninstall", "claude"], env(path.join(box.root, "claude-c")));
  assert.match(other.out, /nothing recorded/);
  assert.match(other.out, /claude-a/);
  assert.ok(fs.existsSync(path.join(box.durable, version)));
  box.clearCalls();
  assert.equal(box.run(["uninstall", "claude"], env(homeA)).status, 0);
  assert.equal(box.calls()[0].argv[0], "plugin");
  assert.deepEqual(box.own(path.join(homeA, "settings.json")), {});
  assert.ok(fs.existsSync(path.join(homeB, "settings.json")));
  assert.ok(fs.existsSync(path.join(box.durable, version)), "B still uses the copy");
  assert.equal(box.run(["uninstall", "claude"], env(homeB)).status, 0);
  assert.ok(!fs.existsSync(box.durable));
}));

test("M3: codex uninstall runs against the CODEX_HOME it was installed with", withBox((box) => {
  const codexHome = path.join(box.root, "codex-x");
  assert.equal(box.run(["install", "codex"], { extraEnv: { CODEX_HOME: codexHome } }).status, 0);
  assert.match(box.run(["uninstall", "codex"]).out, /nothing recorded/);
  assert.ok(fs.existsSync(box.durable));
  assert.equal(box.run(["uninstall", "codex"], { extraEnv: { CODEX_HOME: codexHome } }).status, 0);
}));

test("M4: settings written through a symlink keep the link and the mode", withBox((box) => {
  const real = path.join(box.root, "dot", "settings.json");
  box.write(real, { theme: "dark" });
  fs.chmodSync(real, 0o600);
  fs.mkdirSync(path.dirname(box.claudeSettings), { recursive: true });
  fs.symlinkSync(real, box.claudeSettings);
  assert.equal(box.run(["install", "claude"]).status, 0);
  assert.ok(fs.lstatSync(box.claudeSettings).isSymbolicLink());
  assert.equal(stat(real), 0o600);
  assert.ok(box.read(real).permissions.allow.includes(ALLOW));
  assert.equal(box.run(["uninstall", "claude"]).status, 0);
  assert.ok(fs.lstatSync(box.claudeSettings).isSymbolicLink());
  assert.deepEqual(box.own(real), { theme: "dark" });
  assert.equal(stat(real), 0o600);
}));

test("CORTEX-10: new config files default to 0o600, dangling dotfile symlinks survive, and failed writes leave no temp files", withBox((box) => {
  // 1) Newly created opencode.json defaults to 0o600 (not world-readable 0o644).
  assert.equal(box.run(["install", "opencode"]).status, 0);
  assert.equal(stat(box.opencodeConfig), 0o600);
  assert.equal(box.run(["uninstall", "opencode"]).status, 0);

  // 2) Dangling dotfile symlink (~/.config/opencode/opencode.json -> dot/opencode.json where dot/ does not exist yet).
  const dotTarget = path.join(box.root, "dotfiles-new", "opencode.json");
  fs.mkdirSync(path.dirname(box.opencodeConfig), { recursive: true });
  fs.symlinkSync(dotTarget, box.opencodeConfig);
  assert.equal(box.run(["install", "opencode"]).status, 0);
  assert.ok(fs.lstatSync(box.opencodeConfig).isSymbolicLink(), "dangling symlink preserved on install");
  assert.ok(fs.existsSync(dotTarget), "target file created through dangling symlink");
  assert.equal(stat(dotTarget), 0o600);
  assert.equal(box.run(["uninstall", "opencode"]).status, 0);
  assert.ok(fs.lstatSync(box.opencodeConfig).isSymbolicLink(), "user dotfile symlink kept on uninstall");
  assert.ok(!fs.existsSync(dotTarget), "installer-created target removed on uninstall");
  fs.unlinkSync(box.opencodeConfig);

  // 3) If target is a directory (making renameSync throw EISDIR), .ni-rua-tmp-<pid> is cleaned up with zero secret residue.
  fs.mkdirSync(box.opencodeConfig, { recursive: true });
  const parentDir = path.dirname(box.opencodeConfig);
  const failed = box.run(["install", "opencode"]);
  assert.notEqual(failed.status, 0);
  const leftovers = fs.readdirSync(parentDir).filter((name) => name.includes(".ni-rua-tmp-"));
  assert.deepEqual(leftovers, []);
}));

test("M5: directories the installer did not create are never deleted", withBox((box) => {
  fs.mkdirSync(path.join(box.durable, "my-notes"), { recursive: true });
  fs.writeFileSync(path.join(box.durable, "my-notes", "a.txt"), "keep");
  fs.symlinkSync(path.join(box.root, "nowhere"), path.join(box.durable, "dangling"));
  assert.equal(box.run(["install", "opencode"]).status, 0);
  const next = copyPackage(path.join(box.root, "pkg-next"), "0.1.1");
  assert.equal(box.run(["install", "opencode"], { cli: next }).status, 0);
  assert.ok(!fs.existsSync(path.join(box.durable, version)), "its own old copy is dropped");
  assert.equal(fs.readFileSync(path.join(box.durable, "my-notes", "a.txt"), "utf8"), "keep");
  assert.equal(box.run(["uninstall", "opencode"], { cli: next }).status, 0);
  assert.ok(fs.existsSync(path.join(box.durable, "my-notes", "a.txt")), "and the data dir stays while foreign content is in it");
}));

test("concurrent runs are refused by the lock instead of corrupting the record", withBox((box) => {
  fs.mkdirSync(path.join(box.durable, ".lock"), { recursive: true });
  const result = box.run(["install", "opencode"]);
  assert.equal(result.status, 2);
  assert.match(result.stderr, /another ni-rua-plugin is running/);
  fs.rmdirSync(path.join(box.durable, ".lock"));
  assert.equal(box.run(["install", "opencode"]).status, 0);
  assert.ok(!fs.existsSync(path.join(box.durable, ".lock")), "the lock is released");
}));

test("uninstall keeps the record and exits non-zero when a host command fails", withBox((box) => {
  assert.equal(box.run(["install", "claude"]).status, 0);
  const failed = box.run(["uninstall", "claude"], { extraEnv: { FAKE_FAIL: "plugin uninstall" } });
  assert.equal(failed.status, 2);
  assert.match(failed.stderr, /record and the copy are kept/);
  assert.match(failed.stderr, /claude plugin uninstall ni_rua@ni_rua -s user/);
  assert.ok(fs.existsSync(path.join(box.durable, "installed.json")));
  assert.ok(fs.existsSync(path.join(box.durable, version)));
  assert.equal(box.run(["uninstall", "claude"]).status, 0, "retry succeeds");
  assert.ok(!fs.existsSync(box.durable));
  // a host that says "not installed" is not a failure
  assert.equal(box.run(["install", "codex"]).status, 0);
  assert.equal(box.run(["uninstall", "codex"], { extraEnv: { FAKE_FAIL: "plugin marketplace remove" } }).status, 2, "unexplained failure is reported");
  assert.equal(box.run(["uninstall", "codex"], { extraEnv: { FAKE_FAIL: "plugin remove,plugin marketplace remove", FAKE_FAIL_MSG: "Error: marketplace `ni_rua` is not configured or installed" } }).status, 0);
}));

test("low user values are kept and reported by install and doctor", async () => {
  const box = sandbox();
  try {
    box.write(box.claudeSettings, { env: { MCP_TOOL_TIMEOUT: "60000", CLAUDE_CODE_MAX_MCP_DESCRIPTION_LENGTH: "2048", CLAUDE_CODE_MCP_AUTO_BACKGROUND_MS: "5000" } });
    const result = box.run(["install", "claude"]);
    assert.match(result.out, /WARNING .*MCP_TOOL_TIMEOUT="60000"/);
    assert.equal(box.read(box.claudeSettings).env.MCP_TOOL_TIMEOUT, "60000", "not overwritten");
    const doctor = await box.runAsync(["doctor"]);
    assert.match(doctor.stdout, /\[warn\] claude: env values lower than the plugin needs[^\n]*MCP_TOOL_TIMEOUT[^\n]*DESCRIPTION_LENGTH[^\n]*BACKGROUND_MS|\[warn\] claude: env values lower than the plugin needs[^\n]*BACKGROUND_MS[^\n]*DESCRIPTION_LENGTH/);
    assert.ok(!/four MCP env keys are set/.test(doctor.stdout));
  } finally { box.cleanup(); }
});

test("keys the host writes into settings.json during `plugin install` survive our merge", withBox((box) => {
  assert.equal(box.run(["install", "claude"]).status, 0);
  const settings = box.read(box.claudeSettings);
  assert.deepEqual(settings.enabledPlugins, { "ni_rua@ni_rua": true }, "the host's enabledPlugins entry is not overwritten");
  assert.ok(settings.permissions.allow.includes(ALLOW));
}));

test("opencode 1.x is refused before writing any config or durable copy", withBox((box) => {
  const result = box.run(["install", "opencode"], { extraEnv: { FAKE_VERSION_opencode: "1.1.51" } });
  assert.equal(result.status, 2);
  assert.match(result.stderr, /1\.1\.51.*requires OpenCode 2\.x/);
  assert.ok(!fs.existsSync(box.opencodeConfig), "1.x config must never be written");
  assert.ok(!fs.existsSync(box.durable), "no durable copy or record left on version refusal");
}));

test("codex install writes and reverts model_catalog_json, supports --no-direct-catalog, and handles missing CODEX_HOME atomically", withBox((box) => {
  const codexHome = path.join(box.root, "fresh-codex-home");
  const configToml = path.join(codexHome, "config.toml");
  // 未预先创建 CODEX_HOME 时可自动创建并成功写入 model_catalog_json
  assert.equal(box.run(["install", "codex"], { extraEnv: { CODEX_HOME: codexHome } }).status, 0);
  const durableCatalog = path.join(box.durable, version, "codex-models.direct.json");
  assert.ok(fs.existsSync(durableCatalog));
  assert.match(fs.readFileSync(configToml, "utf8"), new RegExp(`^model_catalog_json = ${JSON.stringify(durableCatalog).replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\n`));
  assert.equal(box.run(["uninstall", "codex"], { extraEnv: { CODEX_HOME: codexHome } }).status, 0);
  assert.ok(!fs.existsSync(configToml), "installer-created config.toml is removed on uninstall");

  // 已有 config.toml 时卸载需按字节完全恢复
  const originalToml = 'model = "gpt-6.1-sol"\n\n[model_providers.demo]\nname = "demo"\n';
  fs.mkdirSync(codexHome, { recursive: true });
  fs.writeFileSync(configToml, originalToml);
  assert.equal(box.run(["install", "codex"], { extraEnv: { CODEX_HOME: codexHome } }).status, 0);
  assert.match(fs.readFileSync(configToml, "utf8"), /^model_catalog_json = /);
  assert.equal(box.run(["uninstall", "codex"], { extraEnv: { CODEX_HOME: codexHome } }).status, 0);
  assert.equal(fs.readFileSync(configToml, "utf8"), originalToml);

  // --no-direct-catalog 不写入 model_catalog_json
  assert.equal(box.run(["install", "codex", "--no-direct-catalog"], { extraEnv: { CODEX_HOME: codexHome } }).status, 0);
  assert.equal(fs.readFileSync(configToml, "utf8"), originalToml);
  assert.equal(box.run(["uninstall", "codex"], { extraEnv: { CODEX_HOME: codexHome } }).status, 0);

  // marketplace add 失败前不锁死 installed.json 或端口
  const failAdd = box.run(["install", "codex", "--port", "9444"], {
    extraEnv: { CODEX_HOME: codexHome, FAKE_FAIL: "plugin marketplace add" },
  });
  assert.equal(failAdd.status, 2);
  assert.ok(!fs.existsSync(path.join(box.durable, "installed.json")), "no phantom install record left when marketplace add fails");
  assert.equal(box.run(["install", "opencode", "--port", "9555"]).status, 0, "port was not locked by failed codex install");
  assert.equal(box.run(["uninstall", "opencode"]).status, 0);

  // 已有含 `#` 字符路径的 model_catalog_json 时不得被误判为注释而重复写入导致非法 TOML
  const hashCatalogToml = 'model_catalog_json = "/opt/dir#1/models.json" # keep comment\nmodel = "gpt-6.1-sol"\n';
  fs.writeFileSync(configToml, hashCatalogToml);
  assert.equal(box.run(["install", "codex"], { extraEnv: { CODEX_HOME: codexHome } }).status, 0);
  assert.equal(fs.readFileSync(configToml, "utf8"), hashCatalogToml);
  assert.equal(box.run(["uninstall", "codex"], { extraEnv: { CODEX_HOME: codexHome } }).status, 0);

  // 重装时若 codex plugin add 失败，不得擦除前次安装的 config.toml 回滚记录
  fs.writeFileSync(configToml, originalToml);
  assert.equal(box.run(["install", "codex"], { extraEnv: { CODEX_HOME: codexHome } }).status, 0);
  assert.match(fs.readFileSync(configToml, "utf8"), /^model_catalog_json = /);
  const failReinstall = box.run(["install", "codex"], { extraEnv: { CODEX_HOME: codexHome, FAKE_FAIL: "plugin add" } });
  assert.equal(failReinstall.status, 2);
  assert.equal(box.run(["uninstall", "codex"], { extraEnv: { CODEX_HOME: codexHome } }).status, 0);
  assert.equal(fs.readFileSync(configToml, "utf8"), originalToml, "prior config.toml edit still reverted after failed reinstall");

  // 真实 Codex 行为：空 CODEX_HOME 下 plugin marketplace/add 会先创建 config.toml，重装与卸载后仍须彻底删除空 config.toml
  const realBehaviorHome = path.join(box.root, "codex-real-write");
  const realBehaviorCfg = path.join(realBehaviorHome, "config.toml");
  const realEnv = { CODEX_HOME: realBehaviorHome, FAKE_CODEX_WRITE_CONFIG: "1" };
  assert.equal(box.run(["install", "codex"], { extraEnv: realEnv }).status, 0);
  assert.equal(box.run(["install", "codex"], { extraEnv: realEnv }).status, 0, "idempotent reinstall preserves created_file=true");
  assert.equal(box.run(["uninstall", "codex"], { extraEnv: realEnv }).status, 0);
  assert.ok(!fs.existsSync(realBehaviorCfg), "no 0-byte config.toml left after uninstall when Codex created it during install");

  // UTF-8 BOM 开头的 config.toml 写入时必须保持 BOM 在第 0 字节，卸载按字节完整还原
  const bomToml = '\uFEFFmodel = "gpt-6.1-sol"\n';
  fs.writeFileSync(configToml, bomToml);
  assert.equal(box.run(["install", "codex"], { extraEnv: { CODEX_HOME: codexHome } }).status, 0);
  const withBomInstalled = fs.readFileSync(configToml, "utf8");
  assert.ok(withBomInstalled.startsWith("\uFEFFmodel_catalog_json = "), "BOM must remain at byte 0");
  assert.ok(!withBomInstalled.slice(1).includes("\uFEFF"), "no BOM pushed onto line 2");
  assert.equal(box.run(["uninstall", "codex"], { extraEnv: { CODEX_HOME: codexHome } }).status, 0);
  assert.equal(fs.readFileSync(configToml, "utf8"), bomToml, "BOM-prefixed config.toml restored byte-for-byte");
}));

test("claude multi-project uninstall preserves shared marketplace and keeps settings if CLI uninstall fails", withBox((box) => {
  const projA = path.join(box.root, "proj-a");
  const projB = path.join(box.root, "proj-b");
  fs.mkdirSync(projA);
  fs.mkdirSync(projB);
  const resA = box.run(["install", "claude", "--scope", "project"], { cwd: projA });
  assert.equal(resA.status, 0, resA.out);
  assert.match(resA.out, /WARNING: workspace .* is not trusted by Claude Code/);
  assert.equal(box.run(["install", "claude", "--scope", "project"], { cwd: projB }).status, 0);

  // 重装时若 claude plugin install 失败，不得擦除前次安装的 settings 回滚记录
  const failReinstallA = box.run(["install", "claude", "--scope", "project"], { cwd: projA, extraEnv: { FAKE_FAIL: "plugin install" } });
  assert.equal(failReinstallA.status, 2);

  // CLI 卸载失败时不得提前删掉 projA 的 settings.json
  const failed = box.run(["uninstall", "claude", "--scope", "project"], { cwd: projA, extraEnv: { FAKE_FAIL: "plugin uninstall" } });
  assert.equal(failed.status, 2);
  assert.ok(box.read(path.join(projA, ".claude", "settings.json")).permissions.allow.includes(ALLOW), "settings stay intact when CLI uninstall fails");

  // 卸载 projA 时因同 CLAUDE_CONFIG_DIR 下仍有 projB，不得移除共享 marketplace，且前次 settings 仍被干净清理
  box.clearCalls();
  assert.equal(box.run(["uninstall", "claude", "--scope", "project"], { cwd: projA }).status, 0);
  assert.deepEqual(box.commands(), ["claude plugin uninstall ni_rua@ni_rua -s project"]);
  assert.deepEqual(box.read(path.join(projA, ".claude", "settings.json")), { enabledPlugins: {} }, "prior settings still reverted after failed reinstall");
  assert.ok(box.read(path.join(projB, ".claude", "settings.json")).permissions.allow.includes(ALLOW));

  // 卸载最后一个项目 projB 时才移除共享 marketplace
  box.clearCalls();
  assert.equal(box.run(["uninstall", "claude", "--scope", "project"], { cwd: projB }).status, 0);
  assert.deepEqual(box.commands(), [
    "claude plugin uninstall ni_rua@ni_rua -s project",
    "claude plugin marketplace remove ni_rua",
  ]);
  assert.ok(!fs.existsSync(box.durable));
}));

test("claude project installKey normalizes symlinked cwd and opencode honours OPENCODE_CONFIG_DIR", withBox((box) => {
  const realProj = path.join(box.root, "real-proj");
  const linkProj = path.join(box.root, "link-proj");
  fs.mkdirSync(realProj);
  fs.symlinkSync(realProj, linkProj);
  assert.equal(box.run(["install", "claude", "--scope", "project"], { cwd: linkProj }).status, 0);
  assert.equal(box.run(["uninstall", "claude", "--scope", "project"], { cwd: realProj }).status, 0);
  assert.ok(!fs.existsSync(box.durable));

  const customOcDir = path.join(box.root, "custom-oc");
  assert.equal(box.run(["install", "opencode"], { extraEnv: { OPENCODE_CONFIG_DIR: customOcDir } }).status, 0);
  assert.ok(fs.existsSync(path.join(customOcDir, "opencode.json")));
  assert.equal(box.run(["uninstall", "opencode"], { extraEnv: { OPENCODE_CONFIG_DIR: customOcDir } }).status, 0);
  assert.ok(!fs.existsSync(path.join(customOcDir, "opencode.json")));

  // Claude user 作用域在临时目录安装后，即便该目录随后被删除，uninstall 仍必须成功
  const ephemeralCwd = path.join(box.root, "ephemeral-cwd");
  fs.mkdirSync(ephemeralCwd);
  assert.equal(box.run(["install", "claude"], { cwd: ephemeralCwd }).status, 0);
  fs.rmSync(ephemeralCwd, { recursive: true, force: true });
  const uninstallGone = box.run(["uninstall", "claude"], { cwd: box.root });
  assert.equal(uninstallGone.status, 0, uninstallGone.out);
  assert.ok(!fs.existsSync(box.durable));
}));

test("opencode compaction hook guards only active NI-RUA turns and claude compact hook fires only for active turns", withBox(async (box) => {
  const { spawnSync } = await import("node:child_process");
  const pluginMod = await import(new URL("../opencode/plugin.js", import.meta.url).href);
  const hooks = new Map();
  await pluginMod.default.setup({
    mcp: { transform: async (fn) => fn(new Map()) },
    skill: { transform: async (fn) => fn({ add: () => {} }) },
    session: { hook: async (name, fn) => hooks.set(name, fn) },
  });
  const onCompact = hooks.get("compaction");
  assert.equal(typeof onCompact, "function");

  // 活跃 NI-RUA 回合触发压缩：向 system 追加防自毒化与重调 run_begin 提示（且不重复注入）
  const activeEvent = {
    sessionID: "ses-ni-rua-1",
    system: [{ type: "text", text: "base" }],
    messages: [
      { role: "assistant", content: [{ type: "tool-call", id: "1", name: "ni_rua_run_begin", input: {} }] },
      { role: "tool", content: [{ type: "tool-result", id: "1", name: "ni_rua_run_begin", result: {} }] },
    ],
  };
  onCompact(activeEvent);
  onCompact(activeEvent);
  assert.equal(activeEvent.system.length, 2);
  assert.equal(activeEvent.system[1].type, "text");
  assert.match(activeEvent.system[1].text, /run_begin/);

  // 同一活跃会话在连续第二次压缩且窗口内尚无新 ni_rua_* 工具调用时，仍须按 sessionID 续注提醒；无关会话不得串扰
  const secondCompactSameSession = {
    sessionID: "ses-ni-rua-1",
    system: [{ type: "text", text: "base" }],
    messages: [{ role: "user", content: [{ type: "text", text: "summary from prior compaction" }] }],
  };
  onCompact(secondCompactSameSession);
  assert.equal(secondCompactSameSession.system.length, 2);

  const otherSessionEvent = {
    sessionID: "ses-other-2",
    system: [{ type: "text", text: "base" }],
    messages: [{ role: "user", content: [{ type: "text", text: "plain coding chat" }] }],
  };
  onCompact(otherSessionEvent);
  assert.equal(otherSessionEvent.system.length, 1);

  // 已调用 run_finish 关闭回合后，当次与后续压缩均不得注入提示
  const closedEvent = {
    sessionID: "ses-ni-rua-1",
    system: [{ type: "text", text: "base" }],
    messages: [
      { role: "assistant", content: [{ type: "tool-call", id: "1", name: "ni_rua_run_begin", input: {} }] },
      { role: "assistant", content: [{ type: "tool-call", id: "2", name: "ni_rua_run_finish", input: { summary: "ok" } }] },
    ],
  };
  onCompact(closedEvent);
  assert.equal(closedEvent.system.length, 1);

  const postFinishCompact = {
    sessionID: "ses-ni-rua-1",
    system: [{ type: "text", text: "base" }],
    messages: [{ role: "user", content: [{ type: "text", text: "normal follow-up chat" }] }],
  };
  onCompact(postFinishCompact);
  assert.equal(postFinishCompact.system.length, 1);

  const unrelatedEvent = {
    system: [{ type: "text", text: "base" }],
    messages: [{ role: "assistant", content: [{ type: "tool-call", id: "1", name: "bash", input: {} }] }],
  };
  onCompact(unrelatedEvent);
  assert.equal(unrelatedEvent.system.length, 1);

  // Claude Code hooks/on-compact.mjs 契约验证
  const hookScript = new URL("../hooks/on-compact.mjs", import.meta.url).pathname;
  const runHook = (payload) => {
    const res = spawnSync(process.execPath, [hookScript], { input: JSON.stringify(payload), encoding: "utf8" });
    assert.equal(res.status, 0, res.stderr);
    return JSON.parse(res.stdout);
  };
  const activeTranscript = path.join(box.root, "active.jsonl");
  // 包含 api-request-shape 中完整的工具定义（含 run_finish）时，仍必须准确识别真实 tool_use 状态
  fs.writeFileSync(
    activeTranscript,
    [
      JSON.stringify({ type: "api-request-shape", shape: { tools: [{ name: "mcp__plugin_ni_rua_ni_rua__run_begin" }, { name: "mcp__plugin_ni_rua_ni_rua__run_finish" }] } }),
      JSON.stringify({ type: "assistant", message: { role: "assistant", content: [{ type: "tool_use", name: "mcp__plugin_ni_rua_ni_rua__run_begin" }] } }),
      JSON.stringify({ type: "api-request-shape", shape: { tools: [{ name: "mcp__plugin_ni_rua_ni_rua__run_finish" }] } }),
    ].join("\n"),
  );
  const activeOut = runHook({ hook_event_name: "SessionStart", source: "compact", transcript_path: activeTranscript });
  assert.equal(activeOut.hookSpecificOutput?.hookEventName, "SessionStart");
  assert.match(activeOut.hookSpecificOutput?.additionalContext || "", /run_begin/);

  // 非 compact 触发不得注入
  assert.deepEqual(runHook({ hook_event_name: "SessionStart", source: "startup", transcript_path: activeTranscript }), {});

  // 已 run_finish 结束的会话（即使后续出现带旧 assistant tool_use 的 api-request-blob 记录）不得误判为活跃
  const finishedTranscript = path.join(box.root, "finished.jsonl");
  fs.writeFileSync(
    finishedTranscript,
    [
      JSON.stringify({ type: "assistant", message: { role: "assistant", content: [{ type: "tool_use", name: "mcp__plugin_ni_rua_ni_rua__run_begin" }] } }),
      JSON.stringify({ type: "assistant", message: { role: "assistant", content: [{ type: "tool_use", name: "mcp__plugin_ni_rua_ni_rua__run_finish" }] } }),
      JSON.stringify({ type: "api-request-blob", message: { role: "assistant", content: [{ type: "tool_use", name: "mcp__plugin_ni_rua_ni_rua__run_begin" }] } }),
    ].join("\n"),
  );
  assert.deepEqual(runHook({ hook_event_name: "SessionStart", source: "compact", transcript_path: finishedTranscript }), {});

  // 非 NI-RUA 会话不得注入
  const otherTranscript = path.join(box.root, "other.jsonl");
  fs.writeFileSync(otherTranscript, `${JSON.stringify({ type: "assistant", message: { role: "assistant", content: [{ type: "tool_use", name: "Bash" }] } })}\n`);
  assert.deepEqual(runHook({ hook_event_name: "SessionStart", source: "compact", transcript_path: otherTranscript }), {});
}));
