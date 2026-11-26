// 零硬件本地 MCP 干跑网关（用于验证 Agent <-> MCP 工具闭环，不触碰任何真实硬件）
import http from "node:http";

const DRY_RUN_INSTRUCTIONS = [
  "NI-RUA dry-run MCP gateway (zero-hardware simulation).",
  "Lifecycle: call `run_begin` first, then `robot_observe` / `robot_action`, and end with `run_finish`.",
  "All robot state and motion receipts are simulated in memory for connectivity verification.",
].join("\n");

const DRY_RUN_TOOLS = [
  {
    name: "run_begin",
    description: "Open a dry-run NI-RUA robot turn and receive the operating briefing and budget.",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
  },
  {
    name: "robot_observe",
    description: "Read the current simulated SO-101 joint angles (rad), TCP pose (m), and gripper opening (m).",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
  },
  {
    name: "robot_action",
    description: "Preview or execute a simulated motion delta (m) or gripper goal [0..1] in dry-run mode.",
    inputSchema: {
      type: "object",
      properties: {
        mode: { type: "string", enum: ["preview", "execute", "reach"], default: "execute" },
        delta_xyz_m: {
          type: "array",
          items: { type: "number" },
          minItems: 3,
          maxItems: 3,
          description: "Optional Cartesian displacement [dx, dy, dz] in meters.",
        },
        gripper_goal: { type: "number", minimum: 0, maximum: 1, description: "Optional normalized gripper target (0=closed, 1=open)." },
      },
      additionalProperties: true,
    },
  },
  {
    name: "run_finish",
    description: "Close the active dry-run turn with a concise summary.",
    inputSchema: {
      type: "object",
      properties: { summary: { type: "string", description: "Short factual summary of the dry-run turn." } },
      additionalProperties: false,
    },
  },
];

function toolResponse(payload, isError = false) {
  return {
    content: [{ type: "text", text: JSON.stringify(payload, null, 2) }],
    isError,
  };
}

export function startDryRunServer({ port = 8790, host = "127.0.0.1", version = "0.1.1" } = {}) {
  const state = {
    activeTurn: false,
    turnCount: 0,
    stepCount: 0,
    tcp_m: [0.22, 0.0, 0.14],
    q_rad: [0.0, -0.95, 0.9, 1.0, 0.0],
    gripper_normalized: 0.5,
  };

  function callTool(name, args = {}) {
    if (name === "run_begin") {
      state.activeTurn = true;
      state.turnCount += 1;
      return toolResponse({
        kind: "run_started",
        mode: "dry_run",
        turn: state.turnCount,
        robot_model_id: "so101_dry_run",
        instructions: DRY_RUN_INSTRUCTIONS,
        budget: { max_steps: 20, remaining_steps: 20 - state.stepCount },
      });
    }
    if (!state.activeTurn) {
      return toolResponse({ kind: "error", code: "turn_not_open", message: "Call `run_begin` before using other NI-RUA tools." }, true);
    }
    if (name === "robot_observe") {
      return toolResponse({
        kind: "observation",
        mode: "dry_run",
        observation_id: `dry-obs-${state.stepCount}`,
        robot_state: {
          arm: "right",
          q_rad: [...state.q_rad],
          ee_pose: { position_m: state.tcp_m.map((v) => Number(v.toFixed(4))), quaternion_xyzw: [0, 1, 0, 0] },
          gripper_normalized: Number(state.gripper_normalized.toFixed(3)),
          measured_opening_m: Number((0.005 + state.gripper_normalized * 0.13).toFixed(4)),
        },
      });
    }
    if (name === "robot_action") {
      const mode = args.mode || "execute";
      const delta = Array.isArray(args.delta_xyz_m) && args.delta_xyz_m.length === 3 ? args.delta_xyz_m.map(Number) : [0, 0, 0];
      const nextTcp = state.tcp_m.map((v, i) => Number((v + delta[i]).toFixed(4)));
      const nextGripper = typeof args.gripper_goal === "number" ? Math.max(0, Math.min(1, args.gripper_goal)) : state.gripper_normalized;
      if (mode === "execute") {
        state.stepCount += 1;
        state.tcp_m = nextTcp;
        state.gripper_normalized = nextGripper;
      }
      return toolResponse({
        kind: "action_receipt",
        mode,
        dry_run: true,
        status: "completed",
        step: state.stepCount,
        ee_pose: { position_m: nextTcp, quaternion_xyzw: [0, 1, 0, 0] },
        gripper_normalized: Number(nextGripper.toFixed(3)),
      });
    }
    if (name === "run_finish") {
      state.activeTurn = false;
      return toolResponse({
        kind: "run_finished",
        mode: "dry_run",
        steps_executed: state.stepCount,
        summary: typeof args.summary === "string" ? args.summary : "",
      });
    }
    return toolResponse({ kind: "error", code: "unknown_tool", message: `Unknown tool: ${name}` }, true);
  }

  const server = http.createServer((req, res) => {
    if (req.method === "GET" && (req.url === "/" || req.url === "/health")) {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ status: "ok", mode: "dry_run", mcp_url: `http://${host}:${server.address().port}/mcp`, tools: DRY_RUN_TOOLS.map((t) => t.name) }));
      return;
    }
    if (req.method !== "POST" || req.url !== "/mcp") {
      res.writeHead(req.method === "GET" && req.url === "/mcp" ? 405 : 404).end();
      return;
    }
    let body = "";
    req.on("data", (chunk) => { body += chunk; });
    req.on("end", () => {
      let message;
      try { message = JSON.parse(body); } catch {
        res.writeHead(400, { "content-type": "application/json" });
        res.end(JSON.stringify({ jsonrpc: "2.0", id: null, error: { code: -32700, message: "Invalid JSON" } }));
        return;
      }
      if (message.id === undefined || message.method === "notifications/initialized") {
        res.writeHead(202).end();
        return;
      }
      let result;
      if (message.method === "initialize") {
        result = {
          protocolVersion: "2025-06-18",
          capabilities: { tools: {} },
          serverInfo: { name: "ni-rua-dry-run", version },
          instructions: DRY_RUN_INSTRUCTIONS,
        };
      } else if (message.method === "tools/list") {
        result = { tools: DRY_RUN_TOOLS };
      } else if (message.method === "tools/call") {
        result = callTool(message.params?.name, message.params?.arguments || {});
      } else {
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify({ jsonrpc: "2.0", id: message.id, error: { code: -32601, message: `Method not found: ${message.method}` } }));
        return;
      }
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ jsonrpc: "2.0", id: message.id, result }));
    });
  });

  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, host, () => {
      const boundPort = server.address().port;
      resolve({
        server,
        port: boundPort,
        url: `http://${host}:${boundPort}/mcp`,
        close: () => new Promise((done) => server.close(done)),
      });
    });
  });
}
