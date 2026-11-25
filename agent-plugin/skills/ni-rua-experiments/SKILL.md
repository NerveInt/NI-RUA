---
name: ni-rua-experiments
description: Use when the user asks you to plan, launch, resume, replay, report on or audit NI-RUA benchmark experiments from an NI-RUA checkout (gateway runs, the evaluation queue, ni-rua-report, ni-rua-audit, the experiment library).
---

# Operating NI-RUA experiments

This skill is for the person-side half of NI-RUA: scheduling and analysing runs. It points at the rules; it does not
restate them. The rules live in the checkout and change with it.

## Read the rules first

Work inside an NI-RUA checkout (the directory with `AGENTS.md`). Before launching or organising anything, read in that
checkout:

- `AGENTS.md`, section "实验评测、60 并发队列与防选集偏差" (and "交付、封版默认对齐与发布");
- `docs/development/experiment-library.md` and `docs/development/experiment-completion-hook.md`;
- `docs/development/running.md` for session resume, batch scheduling, reports and audits.

If there is no such checkout, say so and stop. Do not work from memory of these rules.

## Prerequisites (check, never install silently)

- the NI-RUA checkout, with the `ni-rua` Miniforge environment: run everything as `mamba run -n ni-rua python ...`;
- a provider profile in `configs/profiles/` (git-ignored, mode 0600, key never echoed or printed);
- the pinned agent runtime for the chosen backend (see `docs/development/agent-backends.md`);
- for RoboDojo benchmarks, a GPU host with the simulator. A laptop without it can still replay, report and audit.

## Which runs the queue can do

Queue cases have NI-RUA-hosted actors (`agent.backend` `codex`, `claude_code` or `opencode`). A run where your own
agent drives the robot through MCP (the "external" mode behind the `ni-rua-robot` skill) is interactive only and
cannot be queued. Do not put external runs in a manifest.

## Commands

Run from the checkout root. Flags and arguments are in `--help`; check them there, do not guess.

```sh
# one run (fresh run directory every time)
mamba run -n ni-rua python -u -m ni_rua.apps.gateway --benchmark --task <task> --seed <n> --layout-id <n> \
  --gpu <n> --port <port> --run-dir <fresh-dir>
# read-only replay of an archived run
mamba run -n ni-rua python -m ni_rua.apps.gateway --replay --run-dir <run-dir> --port <port>
# batches: one manifest, one queue, fixed max_parallel
mamba run -n ni-rua ni-rua-queue <manifest.json> --output <fresh-queue-dir> --wait-timeout-s <s> [wake flags below]
# analysis
mamba run -n ni-rua ni-rua-report <run-dir>... [--output <new.json>]
mamba run -n ni-rua ni-rua-audit <run-dir> [--profile integrity|native]
# browse runs, comparisons, replays (http://127.0.0.1:8780)
mamba run -n ni-rua python -m ni_rua.apps.library
```

The manifest, source binding, concurrency and completion hook are fixed before submitting. Multiple cases go in one
manifest, never in ad-hoc background processes.

## Waiting for a queue: the wake recipe per host

The queue must not be polled. End your turn after launching it; something must wake you when it finishes.

- **Codex**: use the existing hook: `ni-rua-queue ... --wake-thread <this thread UUID> --wake-codex <absolute codex
  path> --wake-enabled`. The queue sends one `codex queue` message to the thread when it finishes.
- **Claude Code and OpenCode**: run the queue as your own background shell task with the hook off:
  `ni-rua-queue ... --no-wake-enabled`. Use the host's background mode (Claude Code Bash `run_in_background`;
  OpenCode `shell` with `background`), then end the turn. The host re-invokes you when the process exits.
  This is the recipe for Claude Code 2.1.321 and OpenCode 2.2.7. In an interactive session the user may need to approve the command. If the
  wake does not come, say the wake is unavailable and ask the user to tell you when the queue is done. Do not loop
  with `sleep`, tail logs or spawn watcher agents.

## After the wake

"Queue finished" only means the scheduler is done. Read the final queue state, each case's `summary.json` and
`evaluation.json`, the request identity, native cleanup status and failure evidence. Scores and success rates come
from the official evaluator (`ni-rua-report`), never from the queue state or from the agent's own claim. Keep
unstarted, cancelled, invalid and unscored cases in the fixed denominator. If the completion notice is missing or
incomplete, report the status as unknown.
