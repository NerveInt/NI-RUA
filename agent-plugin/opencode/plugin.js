// OpenCode 2.x entry for the NI-RUA agent plugin. Reads single-source defaults from ../manifest.generated.json.
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

async function loadManifest() {
  const manifestFile = fileURLToPath(new URL("../manifest.generated.json", import.meta.url));
  return JSON.parse(await readFile(manifestFile, "utf8"));
}

// Skill.Info of @opencode/schema 2.2.7: {id, name, description?, autoinvoke?, path (absolute), content}.
// `content` is the skill body; the YAML frontmatter of SKILL.md carries only `name` and `description`.
function parseSkill(text) {
  const match = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/.exec(text);
  if (!match) return { description: undefined, body: text };
  const description = /^description:\s*(.*)$/m.exec(match[1]);
  return { description: description ? description[1].trim().replace(/^"(.*)"$/, "$1") : undefined, body: match[2] };
}

async function loadSkills(skillNames) {
  const skills = [];
  for (const name of skillNames) {
    const file = fileURLToPath(new URL(`../skills/${name}/SKILL.md`, import.meta.url));
    const { description, body } = parseSkill(await readFile(file, "utf8"));
    skills.push({ id: name, name, description, path: file, content: body });
  }
  return skills;
}

const COMPACTION_NOTE = [
  "Compaction note for an active NI-RUA robot session:",
  "- This summarization prompt is internal context compaction, NOT a new user request and NOT an instruction to stop or merely preserve work state.",
  "- Keep the user's original NI-RUA robot task as the Objective and Active work. Never rewrite the latest user request as summarizing or preserving state.",
  "- In Next Move, put step 1 as: call `run_begin` first after compaction to reload the NI-RUA operating instructions and decision context, then continue the robot task until it completes or `run_finish` is called.",
].join("\n");

// 判断消息窗口内 NI-RUA 回合状态：active（存在未关闭调用）、closed（最后为 run_finish）或 none（窗口内无 NI-RUA 调用）
function niRuaTurnState(messages, serverName) {
  if (!Array.isArray(messages)) return "none";
  const prefix = `${serverName}_`;
  const finishName = `${prefix}run_finish`;
  let state = "none";
  for (const message of messages) {
    if (!Array.isArray(message?.content)) continue;
    for (const part of message.content) {
      if (part?.type !== "tool-call" && part?.type !== "tool-result") continue;
      if (typeof part.name !== "string" || !part.name.startsWith(prefix)) continue;
      state = part.name === finishName ? "closed" : "active";
    }
  }
  return state;
}

export default {
  id: "ni_rua",
  setup: async (ctx) => {
    const manifest = await loadManifest();
    await ctx.mcp.transform((entries) =>
      entries.set(manifest.server_name, {
        type: "remote",
        url: manifest.mcp_url,
        oauth: false,
        codemode: false,
        timeout: { ...manifest.opencode_timeout_ms },
      }),
    );
    const skills = await loadSkills(manifest.skills);
    await ctx.skill.transform((editor) => {
      for (const skill of skills) editor.add(skill);
    });
    if (typeof ctx.session?.hook === "function") {
      // 按 sessionID 记住已进入活跃 NI-RUA 回合的会话，防止连续两次压缩间未再调用 ni_rua_* 时漏注提醒
      const activeSessions = new Set();
      await ctx.session.hook("compaction", (event) => {
        if (!Array.isArray(event?.system)) return;
        const sessionId = typeof event.sessionID === "string" && event.sessionID ? event.sessionID : null;
        const state = niRuaTurnState(event.messages, manifest.server_name);
        if (state === "closed") {
          if (sessionId) activeSessions.delete(sessionId);
          return;
        }
        if (state === "active") {
          if (sessionId) activeSessions.add(sessionId);
        } else if (!sessionId || !activeSessions.has(sessionId)) {
          return;
        }
        if (!event.system.some((item) => item?.type === "text" && item?.text === COMPACTION_NOTE)) {
          event.system.push({ type: "text", text: COMPACTION_NOTE });
        }
      });
    }
  },
};
