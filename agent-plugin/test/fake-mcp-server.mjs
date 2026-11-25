// Minimal MCP-over-HTTP server for the tarball check when the spike's NiRuaMcpServer fixture is not available.
// usage: node fake-mcp-server.mjs <port>      (initialize with instructions, tools/list with three tools)
import http from "node:http";

const port = Number(process.argv[2]);
const tools = ["run_begin", "robot_observe", "run_finish"].map((name) => ({ name, description: `fake ${name}`, inputSchema: { type: "object", properties: {} } }));
http.createServer((req, res) => {
  let body = "";
  req.on("data", (chunk) => { body += chunk; });
  req.on("end", () => {
    if (req.method !== "POST") { res.writeHead(405).end(); return; }
    const message = JSON.parse(body || "{}");
    if (message.id === undefined) { res.writeHead(202).end(); return; }
    const result = message.method === "initialize"
      ? { protocolVersion: "2025-06-18", capabilities: { tools: {} }, serverInfo: { name: "fake-ni-rua", version: "0" }, instructions: "call run_begin first" }
      : message.method === "tools/list" ? { tools } : {};
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({ jsonrpc: "2.0", id: message.id, result }));
  });
}).listen(port, "127.0.0.1", () => console.log("listening", port));
