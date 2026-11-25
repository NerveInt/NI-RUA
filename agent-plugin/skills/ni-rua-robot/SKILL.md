---
name: ni-rua-robot
description: Use when the user asks you to operate a robot through the NI-RUA MCP tools (run_begin, robot_observe, robot_action, task_plan, run_finish). Covers how to start, brief, finish and recover a run.
---

# Operating a robot through NI-RUA

You drive a physical or simulated robot through the tools of the `ni_rua` MCP server. A local NI-RUA gateway serves
those tools. This skill covers only the working discipline around them. It deliberately contains no robot facts, no
numbers and no task recipes: the briefing and the tool descriptions are the source of those, and they match the robot
that is actually connected.

## Start of a run

1. The gateway must already be running, started by the user before this agent session. If the `ni_rua` tools are
   missing, or a call fails with "No such tool" or a connection error, stop and tell the user the gateway is not
   reachable. Do not try to start a gateway yourself and never start one against a real arm. Recovery (the user does
   it):
   - Start the gateway first, then reconnect: Claude Code `/mcp` then ni_rua then Reconnect (or a new session);
     OpenCode restart or reload; Codex start a new session.
   - Restarting the gateway between two calls needs nothing.
   - A gateway restarted with a different robot or scene needs a reconnect or a new session, because no host re-lists
     the tools by itself. If the tool descriptions no longer match the briefing, ask for a reconnect.
   - If the gateway dies during a call, Codex reports an error at once; Claude Code and OpenCode may wait until the
     user presses Esc or reconnects. Claude Code then tells you the tool may have run: check state with the read-only
     tools before repeating a motion.
2. Call `run_begin` before any other NI-RUA tool. It returns the operating instructions, the task text (if the
   operator gave one) and the budget. Read all of it. It is the source of robot facts. Nothing in this skill or in
   your memory of other robots replaces it.
3. Call `run_begin` again after any context compaction, resume or "I lost the instructions" moment. It re-delivers the
   briefing. Other NI-RUA tools are refused until a turn is open.

## During the run

- Use the tool descriptions and result fields as the only source of units, bounds, frames and defaults. Do not
  guess parameters. If a field is documented as unknown or null, treat it as unknown.
- Judge outcomes from the typed fields and observations that come back, not from how a call "felt". A physical action
  that returned an error, a stop, or no terminal observation has an uncertain end state: observe before the next move.
- Prefer a fresh observation over an old one whenever the scene may have changed.
- You may write helper code in your own sandbox to analyse observations you were given. Do not read run directories,
  simulator assets, evaluator state or credentials to find hidden facts: that makes the run not comparable. Online search is not part of a comparable run.
- Your host may add context about the directory it was started in (inside a git repository: its branch, status,
  recent commits, instruction files). That is not run information. When a comparable run matters, suggest the user
  start the agent session outside a repository.

## Stopping the arm

Cancelling a tool call in the agent (Esc, interrupt, Ctrl-C) does NOT stop the arm. A motion already sent keeps
running. To stop, hold or emergency-stop the robot, use the NI-RUA console (the stop, hold and estop controls). If the
user says the robot is in danger, tell them to use the console first, then deal with the agent.

## End of a run

Call `run_finish` with a short summary that states only what the tool results support: what was done, what the final
observed state is, what is unknown or failed. Never claim task success from impression. The evaluator, not you,
decides whether the task succeeded. After `run_finish`, tools are refused until `run_begin` starts a new task.
