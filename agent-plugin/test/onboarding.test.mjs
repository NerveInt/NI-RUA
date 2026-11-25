// 覆盖 setup / gateway (--dry-run 与 SO-101 冷启动保护) / doctor (--so101 / --json / --gateway-url) 契约测试
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { startDryRunServer } from "../bin/dry-run-server.mjs";
import { fakeGateway, sandbox } from "./helpers.mjs";

function writeFakeUv(binDir, logFile) {
  const uvPath = path.join(binDir, "uv");
  const script = `#!${process.execPath}
const fs = require("fs");
const path = require("path");
const argv = process.argv.slice(2);
fs.appendFileSync(${JSON.stringify(logFile)}, JSON.stringify(argv) + "\\n");
if (argv[0] === "--version") { console.log("uv 0.9.10"); process.exit(0); }
if (argv[0] === "venv") {
  const dir = argv[argv.length - 1];
  fs.mkdirSync(path.join(dir, "bin"), { recursive: true });
  fs.writeFileSync(path.join(dir, "bin", "python"), "#!/bin/sh\\nexit 0\\n", { mode: 0o755 });
  fs.writeFileSync(path.join(dir, "bin", "pip"), "#!/bin/sh\\nexit 0\\n", { mode: 0o755 });
  process.exit(0);
}
if (argv[0] === "pip" && argv[1] === "install") {
  const pyIdx = argv.indexOf("--python");
  const binDir = path.dirname(argv[pyIdx + 1]);
  if (argv.includes("ni-rua-harness") || argv.some((a) => a.includes("local-ni-rua"))) {
    fs.writeFileSync(path.join(binDir, "ni-rua"), "#!/bin/sh\\necho FAKE_NI_RUA $@\\n", { mode: 0o755 });
  }
  process.exit(0);
}
`;
  fs.writeFileSync(uvPath, script, { mode: 0o755 });
  return uvPath;
}

test("setup installs plugin + isolated Python 3.12 gateway & so101 envs via uv, and respects --skip-gateway", () => {
  const box = sandbox();
  try {
    const uvLog = path.join(box.root, "uv.jsonl");
    writeFakeUv(box.bin, uvLog);

    // 1) --skip-gateway 仅装插件不建 Python 环境
    const skipRes = box.run(["setup", "claude", "--skip-gateway"], {
      extraEnv: { PATH: `${box.bin}:/usr/bin:/bin` },
    });
    assert.equal(skipRes.status, 0, skipRes.out);
    assert.match(skipRes.stdout, /Skipping Python gateway setup/);
    assert.match(skipRes.stdout, /SAFETY: Pressing Esc/);
    assert.ok(!fs.existsSync(uvLog), "uv must not run under --skip-gateway");

    // 2) setup all --so101 创建两个隔离 venv 并支持 NI_RUA_HARNESS_SPEC
    const fullRes = box.run(["setup", "all", "--so101"], {
      extraEnv: {
        PATH: `${box.bin}:/usr/bin:/bin`,
        NI_RUA_HARNESS_SPEC: "/tmp/local-ni-rua.whl",
      },
    });
    assert.equal(fullRes.status, 0, fullRes.out);
    const uvCalls = fs.readFileSync(uvLog, "utf8").trim().split("\n").map((l) => JSON.parse(l));
    const gwDir = path.join(box.home, ".local", "share", "ni-rua", "envs", "gateway");
    const soDir = path.join(box.home, ".local", "share", "ni-rua", "envs", "so101");
    assert.ok(uvCalls.some((a) => a[0] === "venv" && a.includes("3.12") && a.includes(gwDir)), "gateway venv created with Python 3.12");
    assert.ok(uvCalls.some((a) => a[0] === "pip" && a.includes("/tmp/local-ni-rua.whl")), "NI_RUA_HARNESS_SPEC honored");
    assert.ok(uvCalls.some((a) => a[0] === "venv" && a.includes("3.12") && a.includes(soDir)), "separate so101 venv created");
    assert.ok(uvCalls.some((a) => a[0] === "pip" && a.includes("lerobot==0.6.1")), "so101 deps installed in isolated env");
    assert.ok(fs.existsSync(path.join(gwDir, "bin", "ni-rua")));
    assert.ok(fs.existsSync(path.join(soDir, "bin", "python")));
  } finally {
    box.cleanup();
  }
});

test("setup fails fast with exact 1-line uv install command when neither uv nor python3.12 exists", () => {
  const box = sandbox();
  try {
    // 只保留宿主桩目录，不含 uv / python3.12
    const res = box.run(["setup", "claude"], {
      extraEnv: { PATH: box.bin },
    });
    assert.equal(res.status, 2);
    assert.match(res.stderr, /curl -LsSf https:\/\/astral\.sh\/uv\/install\.sh \| sh/);
  } finally {
    box.cleanup();
  }
});

test("dry-run MCP server supports full initialize -> tools/list -> run_begin -> robot_observe -> robot_action -> run_finish lifecycle", async () => {
  const srv = await startDryRunServer({ port: 0, version: "0.1.0" });
  try {
    const rpc = async (body) => {
      const res = await fetch(srv.url, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      });
      return res.status === 202 ? null : res.json();
    };

    const init = await rpc({ jsonrpc: "2.0", id: 1, method: "initialize", params: {} });
    assert.equal(init.result.serverInfo.name, "ni-rua-dry-run");
    assert.match(init.result.instructions, /run_begin/);

    const list = await rpc({ jsonrpc: "2.0", id: 2, method: "tools/list", params: {} });
    assert.deepEqual(list.result.tools.map((t) => t.name), ["run_begin", "robot_observe", "robot_action", "run_finish"]);

    // 未调用 run_begin 时调用 robot_observe 必须被拒
    const refused = await rpc({ jsonrpc: "2.0", id: 3, method: "tools/call", params: { name: "robot_observe", arguments: {} } });
    assert.equal(refused.result.isError, true);

    const begin = await rpc({ jsonrpc: "2.0", id: 4, method: "tools/call", params: { name: "run_begin", arguments: {} } });
    assert.equal(begin.result.isError, false);
    assert.equal(JSON.parse(begin.result.content[0].text).kind, "run_started");

    const obs = await rpc({ jsonrpc: "2.0", id: 5, method: "tools/call", params: { name: "robot_observe", arguments: {} } });
    const pos0 = JSON.parse(obs.result.content[0].text).robot_state.ee_pose.position_m;

    const act = await rpc({
      jsonrpc: "2.0",
      id: 6,
      method: "tools/call",
      params: { name: "robot_action", arguments: { mode: "execute", delta_xyz_m: [0, 0, 0.02], gripper_goal: 0.8 } },
    });
    const receipt = JSON.parse(act.result.content[0].text);
    assert.equal(receipt.status, "completed");
    assert.equal(receipt.ee_pose.position_m[2], Number((pos0[2] + 0.02).toFixed(4)));
    assert.equal(receipt.gripper_normalized, 0.8);

    const fin = await rpc({ jsonrpc: "2.0", id: 7, method: "tools/call", params: { name: "run_finish", arguments: { summary: "ok" } } });
    assert.equal(JSON.parse(fin.result.content[0].text).kind, "run_finished");

    // 结束后再次 observe 必须被拒
    const afterFin = await rpc({ jsonrpc: "2.0", id: 8, method: "tools/call", params: { name: "robot_observe", arguments: {} } });
    assert.equal(afterFin.result.isError, true);
  } finally {
    await srv.close();
  }
});

test("gateway locates managed ni-rua binary and auto-adds --observation-profile rgb_proprio_v1 on SO-101 cold start", () => {
  const box = sandbox();
  try {
    const gwBinDir = path.join(box.home, ".local", "share", "ni-rua", "envs", "gateway", "bin");
    fs.mkdirSync(gwBinDir, { recursive: true });
    const argLog = path.join(box.root, "gw-args.json");
    fs.writeFileSync(
      path.join(gwBinDir, "ni-rua"),
      `#!${process.execPath}\nrequire("fs").writeFileSync(${JSON.stringify(argLog)}, JSON.stringify(process.argv.slice(2)));\n`,
      { mode: 0o755 },
    );

    const res = box.run(["gateway", "--config", "so101", "--port", "8781", "--", "--extra-flag"], {
      extraEnv: { PATH: `${box.bin}:/usr/bin:/bin` },
    });
    assert.equal(res.status, 0, res.out);
    assert.match(res.stdout, /No camera_calibration\.json found for SO-101 cold start/);
    assert.match(res.stdout, /\/calibration/);
    assert.match(res.stdout, /SAFETY: Pressing Esc/);
    const spawnedArgs = JSON.parse(fs.readFileSync(argLog, "utf8"));
    assert.deepEqual(spawnedArgs, [
      "--agent-backend", "external",
      "--config", "so101",
      "--port", "8781",
      "--extra-flag",
      "--observation-profile", "rgb_proprio_v1",
    ]);

    // 一旦存在 ~/.config/ni-rua/camera_calibration.json，则不再自动追加 --observation-profile
    box.write(path.join(box.home, ".config", "ni-rua", "camera_calibration.json"), { version: 1 });
    const resCalib = box.run(["gateway", "--config", "so101"], {
      extraEnv: { PATH: `${box.bin}:/usr/bin:/bin` },
    });
    assert.equal(resCalib.status, 0, resCalib.out);
    const spawnedCalib = JSON.parse(fs.readFileSync(argLog, "utf8"));
    assert.ok(!spawnedCalib.includes("--observation-profile"));
  } finally {
    box.cleanup();
  }
});

test("doctor supports --json, --gateway-url, and --so101 hardware/config checks", async () => {
  const box = sandbox();
  const gateway = await fakeGateway(["run_begin", "run_finish", "robot_observe"]);
  try {
    const depPath = path.join(box.home, ".config", "ni-rua", "so101.local.json");
    box.write(depPath, {
      camera_calibration_file: "camera_calibration.json",
      gripper: { empty_close_aperture_m: null },
    });
    const customUrl = `http://127.0.0.1:${gateway.port}/mcp`;
    const res = await box.runAsync(["doctor", "--so101", "--json", "--gateway-url", customUrl], {
      extraEnv: {
        PATH: `${box.bin}:/usr/bin:/bin`,
        NI_RUA_SERIAL_DEVICES: "/dev/ttyACM0",
      },
    });
    assert.equal(res.status, 0, res.out);
    const report = JSON.parse(res.stdout);
    assert.equal(report.ok, true);
    assert.equal(report.package, "@nerveint/ni-rua-plugin");
    assert.equal(report.gateway_url, customUrl);
    const msgs = report.checks.map((c) => `[${c.status}] ${c.message}`);
    assert.ok(msgs.some((m) => m.includes(`gateway on ${customUrl}`)), "custom --gateway-url probed");
    assert.ok(msgs.some((m) => m.includes("[ok] SO-101 USB serial device(s) detected: /dev/ttyACM0")));
    assert.ok(msgs.some((m) => m.includes("empty_close_aperture_m is null")));
    assert.ok(msgs.some((m) => m.includes("camera_calibration.json missing")));
  } finally {
    await gateway.close();
    box.cleanup();
  }
});
