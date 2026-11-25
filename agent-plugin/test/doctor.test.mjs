import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { fakeGateway, sandbox } from "./helpers.mjs";

test("doctor reports the tool count of a running gateway and the host state", async () => {
  const box = sandbox();
  const gateway = await fakeGateway(["run_begin", "run_finish", "robot_observe"]);
  try {
    assert.equal(box.run(["install", "claude", "--port", String(gateway.port)]).status, 0);
    box.write(path.join(box.root, "claude-list.json"), JSON.stringify([{ id: "ni_rua@ni_rua", scope: "user", enabled: true }]));
    const result = await box.runAsync(["doctor"], { extraEnv: { FAKE_LIST_claude: path.join(box.root, "claude-list.json"), FAKE_VERSION_codex: "codex-cli 0.150.0" } });
    assert.equal(result.status, 0, result.out);
    assert.match(result.stdout, /\[ok\] node /);
    assert.match(result.stdout, /\[ok\] claude 2\.1\.321 .*\(verified\)/);
    assert.match(result.stdout, /\[warn\] codex 0\.150\.0 .*verified versions: 0\.160\.0, 0\.167\.0/);
    assert.match(result.stdout, /\[ok\] claude: ni_rua@ni_rua listed \(scope user, enabled\)/);
    assert.match(result.stdout, /\[ok\] claude: the four MCP env keys are set/);
    assert.match(result.stdout, /\[ok\] claude: permissions\.allow has mcp__plugin_ni_rua_ni_rua/);
    assert.match(result.stdout, /\[warn\] codex: ni_rua@ni_rua is not listed/);
    assert.match(result.stdout, new RegExp(`\\[ok\\] gateway on 127\\.0\\.0\\.1:${gateway.port}: .*3 tools, instructions present`));
    assert.match(result.stdout, /gateway offers run_begin/);
    assert.deepEqual(gateway.requests.map((r) => r.method), ["initialize", "notifications/initialized", "tools/list"]);
    assert.equal(gateway.requests[0].params.clientInfo.name, "ni-rua-plugin-doctor");
  } finally {
    await gateway.close();
    box.cleanup();
  }
});

test("doctor without a gateway says how to start one", async () => {
  const box = sandbox();
  try {
    const gateway = await fakeGateway([]);
    const port = gateway.port;
    await gateway.close();                          // a port that was just free and now refuses connections
    const result = box.run(["doctor", "--port", String(port)]);
    assert.equal(result.status, 0, result.out);
    assert.match(result.stdout, new RegExp(`\\[warn\\] no MCP gateway answered on 127\\.0\\.0\\.1:${port}`));
    assert.match(result.stdout, /ni-rua-plugin gateway --dry-run/);
    assert.match(result.stdout, /ni-rua-plugin gateway --config so101/);
    assert.match(result.stdout, /\[warn\] claude: ni_rua@ni_rua is not listed/);
    assert.match(result.stdout, /\[warn\] opencode: no NI-RUA plugin in /);
  } finally {
    box.cleanup();
  }
});

test("doctor flags a gateway that is not in external mode and a missing host", async () => {
  const box = sandbox();
  const gateway = await fakeGateway(["robot_observe"]);
  try {
    fs.rmSync(path.join(box.bin, "opencode"));
    const result = await box.runAsync(["doctor", "--port", String(gateway.port)], { extraEnv: { PATH: `${box.bin}:/usr/bin:/bin` } });
    assert.match(result.stdout, /\[warn\] opencode: not found on PATH/);
    assert.match(result.stdout, /no run_begin tool: it was not started with --agent-backend external/);
  } finally {
    await gateway.close();
    box.cleanup();
  }
});

test("doctor separates stderr from stdout, flags disabled plugins, checks workspace trust, opencode 1.x/jsonc/tool_output and codex direct catalog", async () => {
  const box = sandbox();
  const gateway = await fakeGateway(["run_begin", "run_finish"]);
  try {
    // 1) stderr 警告不破坏 JSON.parse 与版本解析，且 enabled: false 报 [warn]
    const codexHome = path.join(box.root, "codex-doc");
    fs.mkdirSync(codexHome, { recursive: true });
    fs.writeFileSync(path.join(codexHome, "config.toml"), 'model = "gpt-6.1-sol"\n');
    const proj = path.join(box.root, "git-repo", "sub");
    fs.mkdirSync(path.join(box.root, "git-repo", ".git"), { recursive: true });
    fs.mkdirSync(proj, { recursive: true });
    box.write(path.join(proj, ".claude", "settings.json"), { permissions: { allow: ["mcp__plugin_ni_rua_ni_rua"] } });
    box.write(path.join(box.root, "claude-disabled.json"), JSON.stringify([{ id: "ni_rua@ni_rua", scope: "project", enabled: false }]));
    box.write(path.join(box.root, "codex-enabled.json"), JSON.stringify({ installed: [{ pluginId: "ni_rua@ni_rua", enabled: true }] }));
    const ocDir = path.join(box.root, "oc-dir");
    const durablePath = path.join(box.durable, JSON.parse(fs.readFileSync(new URL("../package.json", import.meta.url), "utf8")).version);
    // 键序相反且高于要求的 tool_output 不应触发提示
    box.write(path.join(ocDir, "opencode.json"), { plugins: [durablePath], tool_output: { max_bytes: 32000000, max_lines: 200000 } });
    box.write(path.join(ocDir, "opencode.jsonc"), "{}\n");

    const res = await box.runAsync(["doctor", "--port", String(gateway.port)], {
      cwd: proj,
      extraEnv: {
        CODEX_HOME: codexHome,
        OPENCODE_CONFIG_DIR: ocDir,
        FAKE_LIST_claude: path.join(box.root, "claude-disabled.json"),
        FAKE_LIST_codex: path.join(box.root, "codex-enabled.json"),
        FAKE_STDERR_codex: "WARNING: refusing to create helper binaries under /tmp (node v99.0.0)",
        FAKE_STDERR_claude: "stderr notice 9.9.9",
        FAKE_VERSION_opencode: "1.1.51",
      },
    });
    assert.equal(res.status, 0, res.out);
    assert.match(res.stdout, /\[ok\] claude 2\.1\.321 .*\(verified\)/);
    assert.match(res.stdout, /\[warn\] claude: ni_rua@ni_rua listed \(scope project, disabled\)/);
    assert.match(res.stdout, /\[warn\] claude: workspace .* is not trusted.*enclosing \.git root/);
    assert.match(res.stdout, /\[ok\] codex 0\.160\.0 .*\(verified\)/);
    assert.match(res.stdout, /\[ok\] codex: ni_rua@ni_rua listed \(enabled\)/);
    assert.match(res.stdout, /\[warn\] codex: .*has no model_catalog_json.*gpt-6\.1-sol/);
    assert.match(res.stdout, /\[warn\] opencode 1\.1\.51 .*is 1\.x/);
    assert.match(res.stdout, /\[warn\] opencode: .*opencode\.jsonc exists/);
    assert.ok(!/results over about 50 KB are cut/.test(res.stdout), "higher tool_output with reversed key order must not warn");

    // 2) 仅信任子目录本身（未信任父 .git 根目录）仍须报 [warn]（与 Claude Code 真实规则一致）
    box.write(path.join(box.env.CLAUDE_CONFIG_DIR, ".claude.json"), {
      projects: { [proj]: { hasTrustDialogAccepted: true } },
    });
    const resSubdirOnly = await box.runAsync(["doctor", "--port", String(gateway.port)], {
      cwd: proj,
      extraEnv: {
        CODEX_HOME: codexHome,
        OPENCODE_CONFIG_DIR: ocDir,
        FAKE_LIST_claude: path.join(box.root, "claude-disabled.json"),
      },
    });
    assert.match(resSubdirOnly.stdout, /\[warn\] claude: workspace .* is not trusted.*checks the enclosing \.git root/);

    // 3) 信任父 .git 根目录后 doctor 报 [ok]；多作用域列表中首项为他项目 disabled 时应命中当前/已启用项
    box.write(path.join(box.env.CLAUDE_CONFIG_DIR, ".claude.json"), {
      projects: { [path.join(box.root, "git-repo")]: { hasTrustDialogAccepted: true } },
    });
    box.write(path.join(box.root, "claude-multi.json"), JSON.stringify([
      { id: "ni_rua@ni_rua", scope: "project", projectPath: path.join(box.root, "other-proj"), enabled: false },
      { id: "ni_rua@ni_rua", scope: "project", projectPath: proj, enabled: true },
    ]));
    const resTrusted = await box.runAsync(["doctor", "--port", String(gateway.port)], {
      cwd: proj,
      extraEnv: {
        CODEX_HOME: codexHome,
        OPENCODE_CONFIG_DIR: ocDir,
        FAKE_LIST_claude: path.join(box.root, "claude-multi.json"),
      },
    });
    assert.match(resTrusted.stdout, /\[ok\] claude: ni_rua@ni_rua listed \(scope project, enabled\)/);
    assert.match(resTrusted.stdout, /\[ok\] claude: workspace trust accepted/);

    // 4) model_catalog_json 缺少当前 code_mode_only 模型条目或 models 为空时必须报 [warn]，不得误报 [ok]
    const incompleteCatalog = path.join(box.root, "incomplete-models.json");
    box.write(incompleteCatalog, { models: [{ slug: "other-model", tool_mode: "direct" }] });
    fs.writeFileSync(path.join(codexHome, "config.toml"), `model_catalog_json = ${JSON.stringify(incompleteCatalog)}\nmodel = "gpt-6.1-sol"\n`);
    const resMissingModel = await box.runAsync(["doctor", "--port", String(gateway.port)], {
      cwd: proj,
      extraEnv: { CODEX_HOME: codexHome, OPENCODE_CONFIG_DIR: ocDir, FAKE_VERSION_claude: "2.1.328 (Claude Code)" },
    });
    assert.match(resMissingModel.stdout, /\[ok\] claude 2\.1\.328 .*\(verified\)/);
    assert.match(resMissingModel.stdout, /\[warn\] codex: model_catalog_json .*has no entry for gpt-6\.1-sol/);

    const emptyCatalog = path.join(box.root, "empty-models.json");
    box.write(emptyCatalog, { models: [] });
    fs.writeFileSync(path.join(codexHome, "config.toml"), `model_catalog_json = ${JSON.stringify(emptyCatalog)}\n`);
    const resEmptyCatalog = await box.runAsync(["doctor", "--port", String(gateway.port)], {
      cwd: proj,
      extraEnv: { CODEX_HOME: codexHome, OPENCODE_CONFIG_DIR: ocDir },
    });
    assert.match(resEmptyCatalog.stdout, /\[warn\] codex: model_catalog_json .*has an empty models list/);

    // 5) model_catalog_json 仍启用 supports_search_tool=true 时须报 [warn]，且 opencode 缺少 media.image 时报 [info]
    const searchCatalog = path.join(box.root, "search-models.json");
    box.write(searchCatalog, { models: [{ slug: "gpt-6.1-sol", tool_mode: "direct", supports_search_tool: true }] });
    fs.writeFileSync(path.join(codexHome, "config.toml"), `model_catalog_json = ${JSON.stringify(searchCatalog)}\nmodel = "gpt-6.1-sol"\n`);
    const resSearchCatalog = await box.runAsync(["doctor", "--port", String(gateway.port)], {
      cwd: proj,
      extraEnv: { CODEX_HOME: codexHome, OPENCODE_CONFIG_DIR: ocDir },
    });
    assert.match(resSearchCatalog.stdout, /\[warn\] codex: model_catalog_json .*supports_search_tool=true on gpt-6\.1-sol/);
    assert.match(resSearchCatalog.stdout, /\[info\] opencode: media\.image is undefined/);
  } finally {
    await gateway.close();
    box.cleanup();
  }
});
