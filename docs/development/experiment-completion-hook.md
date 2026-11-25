# Experiment Completion Hook

Experiments are scheduled by a primary Codex session. Once a batch of experiments finishes, `scripts/experiment_hook.py` (implemented in `ni_rua/apps/evaluation/completion.py`) sends a single notification line back to that same primary Codex session:

```text
Experiment finished, results directory: /absolute/path/to/queue
```

Using its existing conversation context, the primary session reads the results, decides on improvements, and plans the next batch of experiments. The hook does not inject research strategies, output formatting rules, iteration counts, or development time limits. After launching a queue, the primary session ends its current turn; while the queue runs, only normal background processes execute, with zero model polling.

If the queue process has exited but its records are incomplete, the message instead reports that the stop status of some experiments remains unconfirmed. If scheduling has finished but clean shutdown was not confirmed or required forced reclamation, the message prompts the session to inspect the cleanup records. Both fallback messages still include the results directory path.

## Sending Notifications from the Queue

The target primary session UUID is resolved in the following order:
1. The CLI flag `--wake-thread UUID`
2. The unified configuration field `queue.wake_thread`
3. The UUID saved in `~/.config/ni-rua/completion-thread` (if present)

Passing `--wake-thread ''` explicitly disables notifications without reading `~/.config/ni-rua/completion-thread`; `--no-wake-enabled` or `queue.wake_enabled=false` likewise disables notifications. When no recipient is configured, the queue runs standalone. Experiment actors continue to read `codex_home` and `price_card` from `experiments.json` with credentials kept in the local profile; the notification recipient is stored separately so the two never interfere.

```sh
mamba run -n ni-rua python scripts/benchmark_queue.py \
  /absolute/path/cases.json --output /absolute/path/new-queue \
  --wait-timeout-s 86400 --wake-thread MAIN_THREAD_UUID
```

The queue resolves and freezes the recipient before launch and sends a single notification after execution and cleanup complete. If the notification is not accepted (i.e., the receipt status is not `sent`), the queue exits with a non-zero code. GPU wait timeouts and per-benchmark budgets belong to the experiment configuration and are independent of the primary session's research turns.

`--wake-codex /absolute/path/to/codex` (config field `queue.wake_codex`, defaulting to `codex` on `PATH`) specifies the Codex executable used to send the notification, and `--wake-remote unix:///path/to/app-server.sock` (config field `queue.wake_remote`) specifies the app-server endpoint hosting the recipient session. Both options apply exclusively to completion notifications and do not alter the experiment actor's model or environment. If a wrapper script on `PATH` injects `--no-daemon`, it will conflict with the notification's `--remote` flag; pass the native binary path explicitly in that case. When an interactive TUI runs against a standalone daemon, notifications can resume the same thread UUID through a verified endpoint, though this does not directly wake an unattached TUI process. Always verify notification delivery after switching machines or endpoints.

GPU scheduling is handled by the background queue process: by default, every 90 seconds (configurable via `sample_interval_s` in the manifest) it inspects all visible GPUs (`eligible_gpu_indices=null`), schedules up to `60` pending cases per window based on utilization and available memory (allowing multiple cases per GPU), and re-verifies availability immediately before launching each case. By default, each case reserves `10240` MiB (10 GiB) with `0` MiB extra GPU headroom, adjustable via `memory_per_case_mib` and `gpu_headroom_mib` in the manifest. New queues default to `max_parallel=60` across all available GPUs, with actual concurrency bounded by pending cases and resource admission; explicit config or manifest settings can override this ceiling. Completed case reclamation and stop signals are processed immediately without waiting for the next sampling interval.

Messages are submitted to the current user's shared app-server via native `codex queue --thread UUID --message TEXT --remote unix://...`. The default socket is `~/.codex/app-server-control/app-server-control.sock`, and the hook CLI accepts `--remote` to target another verified endpoint. Specifying an explicit endpoint prevents an experiment actor's isolated `CODEX_HOME` from routing the message to the wrong daemon. An idle session starts a new turn automatically, while an active session processes the queued message once its current turn finishes.

## Attaching to an Already Running Queue

For a queue that was started without notification flags, you can attach a one-shot watcher:

```sh
mamba run -n ni-rua python scripts/experiment_hook.py \
  --queue-output /absolute/path/existing-queue --thread MAIN_THREAD_UUID \
  --pid QUEUE_PID --start-ticks PROC_START_TICKS
```

`--pid` and `--start-ticks` must be provided together; when both are omitted, the hook does not wait for a process and notifies immediately only if the queue has already written a terminal state. `--codex` and `--remote` have the same meaning as `--wake-codex` and `--wake-remote` on the queue CLI.

The watcher verifies the PID, `/proc` start ticks, queue command line (`benchmark_queue.py` or `-m ni_rua.apps.evaluation.queue`), and output directory before blocking on process exit via Linux `pidfd`. If the process has already exited, the watcher sends the normal notification only when the queue left a terminal state and no cases remain running. Normal terminal states include failure, interruption, and GPU wait timeout—missing individual case summaries do not prevent the primary session from being notified of a failure. If process exit is confirmed but queue records are incomplete, the watcher sends the fallback alert noting that stop status is unconfirmed for some runs.

Each queue uses an exclusive `.main-session-wake.json` claim file to prevent duplicate notifications, and writes the prompt, queue snapshot, command, CLI output, and receipt under `completion-hook/`. A `sent` status only confirms that the native queue accepted the message; whether the primary session executed a turn is determined by the native turn records. If delivery is interrupted or its outcome is ambiguous, `delivery_unknown` is recorded without automatic retries. No separate background research thread is created, and `codex exec resume` is never used to contend with the primary session.

## Waking the Agent That Launched the Queue (Per Host)

The hook above is Codex-specific: it sends one `codex queue` message to the primary Codex thread. When your own Claude Code or OpenCode session launches the queue through the [agent plugin](agent-plugin.md) (`ni-rua-experiments` skill), the wake recipe differs per host. The queue must never be polled; the agent ends its turn after launching it and something must wake it.

| Host | Recipe | Status |
| --- | --- | --- |
| Codex | The hook above: `ni-rua-queue ... --wake-thread <this thread UUID> --wake-codex <absolute codex path> --wake-enabled`. The queue sends one `codex queue --thread` message when it finishes. | Existing, covered by `tests.test_experiment_hook`; `codex queue --thread` exists on 0.160.0 and 0.167.0 (spike [report](reports/agent-plugin-spikes.md#3-http-first-gono-go)). |
| Claude Code | Run `ni-rua-queue ... --no-wake-enabled` as the agent's own background shell task (Bash `run_in_background`) and end the turn; Claude Code starts a new turn when the process exits (the last user message is a `<task-notification>`). | Verified headless on 2.1.321 (`claude -p`). In an interactive session a real queue command is not read-only, so the user may have to approve it; not verified interactively. |
| OpenCode | The same with the `shell` tool and `background: true`; OpenCode sends a new model request by itself when the shell completes (`<shell ... state="completed">`). | Verified headless on 2.2.7 (`opencode serve`); not verified in the TUI. |

Where the wake does not come, the agent says the wake is unavailable and asks the user to tell it when the queue is done; it must not loop with `sleep`, tail logs or spawn watcher subagents. After the wake the rules are unchanged: read the final queue state, each case's `summary.json` and `evaluation.json` and the cleanup evidence; scores come from the official evaluator. Queue cases keep NI-RUA-hosted actors; an `external` run cannot be queued.

## Verification

```sh
mamba run -n ni-rua python -m unittest tests.test_experiment_hook -v
```

The test suite covers exact primary session targeting, post-success and post-failure notifications, duplicate/ambiguous delivery handling, real `pidfd` waiting, and rejection of still-running queue records.
