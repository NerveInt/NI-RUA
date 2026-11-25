#!/usr/bin/env node
// Claude Code SessionStart (compact) 钩子：仅在当前会话存在未关闭的 NI-RUA 机器人回合时，提醒模型先调用 run_begin 重读简报。
import fs from "node:fs";

const NI_RUA_PREFIX_RE = /^mcp__(?:plugin_ni_rua_ni_rua|ni_rua)__([a-zA-Z0-9_.-]+)$/;
const COMPACT_REMINDER =
  "Context was compacted during an active NI-RUA robot turn. Call `run_begin` now (before any other NI-RUA tool) " +
  "to reload the operating instructions, task briefing, and current decision context, then continue the task.";

function hasActiveNiRuaTurn(transcriptPath) {
  if (typeof transcriptPath !== "string" || !transcriptPath || !fs.existsSync(transcriptPath)) return false;
  let text;
  try {
    text = fs.readFileSync(transcriptPath, "utf8");
  } catch {
    return false;
  }
  let active = false;
  for (const line of text.split("\n")) {
    // 仅解析包含 assistant tool_use 与 NI-RUA MCP 前缀的行，跳过 api-request-shape 中的工具 Schema 声明
    if (!line.includes('"tool_use"') || !line.includes("mcp__")) continue;
    let row;
    try {
      row = JSON.parse(line);
    } catch {
      continue;
    }
    // 跳过 api-request-blob/progress 等非助手回合记录，避免已关闭回合被旧请求镜像误判为活跃
    if (row?.type !== undefined && row.type !== "assistant") continue;
    const content = row?.message?.content;
    if (row?.message?.role !== "assistant" || !Array.isArray(content)) continue;
    for (const block of content) {
      if (block?.type !== "tool_use" || typeof block.name !== "string") continue;
      const match = NI_RUA_PREFIX_RE.exec(block.name);
      if (!match) continue;
      active = match[1] !== "run_finish";
    }
  }
  return active;
}

let raw = "";
for await (const chunk of process.stdin) raw += chunk;
let event = null;
try {
  event = raw.trim() ? JSON.parse(raw) : null;
} catch {
  event = null;
}

if (event?.source === "compact" && hasActiveNiRuaTurn(event.transcript_path)) {
  process.stdout.write(
    JSON.stringify({
      hookSpecificOutput: {
        hookEventName: "SessionStart",
        additionalContext: COMPACT_REMINDER,
      },
    }),
  );
} else {
  process.stdout.write("{}");
}
