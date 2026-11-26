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

1. The `ni_rua` MCP stdio bridge connects to any already-running local NI-RUA gateway (SO-101, RoboDojo, LIBERO, or
   `--dry-run`) and automatically starts the local SO-101 gateway on first `run_begin` when none is running yet—so
   the user does not need a second terminal for standard SO-101 control.
   - If `run_begin` returns a startup or configuration error (for example missing `ni-rua` runtime, missing
     `so101.local.json`, or gateway log errors), stop and report the exact fix command from the error message to the
     user. Never try to bypass safety gates or fabricate robot state.
   - If the `ni_rua` tools are completely missing from the session, ask the user to verify the plugin install with
     `ni-rua-plugin doctor` and restart or reconnect the session.
   - A gateway manually restarted with a different robot or scene may need a reconnect or a new session if the host
     does not automatically apply `tools/list_changed`. If tool descriptions no longer match the briefing, ask for a
     reconnect.
   - If the gateway dies during a call, check state with read-only tools before repeating any motion.
2. Call `run_begin` before any other NI-RUA tool. It returns the live Web console URL (and calibration link if
   uncalibrated), the operating instructions, the task text (if the operator gave one) and the budget. Read all of it.
   It is the source of robot facts. Nothing in this skill or in your memory of other robots replaces it.
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
running. Never bypass E-STOP or safety interlocks. To stop, hold or emergency-stop the robot, use the NI-RUA Web
console `/console` (the stop, hold and estop controls) or the physical 12 V power cutoff. If the user says the robot
is in danger, tell them to hit `/console` E-STOP or cut 12 V power first, then deal with the agent.

## End of a run

Call `run_finish` with a short summary that states only what the tool results support: what was done, what the final
observed state is, what is unknown or failed. Never claim task success from impression. The evaluator, not you,
decides whether the task succeeded. After `run_finish`, tools are refused until `run_begin` starts a new task.
