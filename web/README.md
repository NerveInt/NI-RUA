# Workbench and Experiment Library Web UI

The web interface attaches to a supervisor-owned Codex PTY and displays synchronized camera videos, stage timelines, and structured Harness records. Selecting a stage updates the video position and recorded context pane, and can scroll the terminal view to a unique matching string—without ever replacing the PTY process or injecting terminal input. Submitting new tasks or resolving native approval prompts requires acquiring a Gateway writer lease. Closing a browser tab releases its writer lease when reachable and never terminates the underlying run; newly opened tabs start in read-only monitoring mode and must claim input ownership explicitly.

## Bilingual Interface (`zh-CN` / `en`)

All four pages (`/console`, `/calibration`, `/index.html`, and `/library.html`) support full Simplified Chinese (`zh-CN`) and English (`en`) localization with zero external runtime dependencies (`web/shared/i18n.js` and `web/i18n/*.js`):

- **Locale resolution priority**: URL query parameter `?lang=en` or `?lang=zh-CN` (validated and immediately persisted to `localStorage['ni-rua.lang']`) > `localStorage['ni-rua.lang']` > default fallback `'zh-CN'`.
- **In-place language switching**: Clicking the header `#lang-switch` toggle button switches between English and Simplified Chinese immediately without reloading the page, updating `<html lang>`, document titles, reversible static DOM labels/attributes, and dynamic views while preserving steady-state polling DOM identity.
- **Direct English links**: Append `?lang=en` to any route (for example `http://127.0.0.1:8778/console?lang=en` or `http://127.0.0.1:8778/calibration?lang=en`) to open the UI in English and persist that preference across pages.

## Real-Robot Operator Console

`GET /console` serves `web/console.html`, a local operator page for real-robot runs. It shows live telemetry previews (operator-only, never model context), joints, motors, faults, run and Codex state, receipts, the model-visible observation, and settings. It also exposes hold/E-STOP/relax/torque/go_home/resume controls through `/v1/control`. See [SO-101 integration §6.3](../docs/development/so101-integration.md). With the synthetic fixture: `mamba run -n ni-rua python web/fixture-server.py --port 8778`, then open `http://127.0.0.1:8778/console`.

The console and `/calibration` share one design system, `web/design-tokens.css` (palette, type scale, spacing, radii, elevation, motion, focus ring; light and dark through `prefers-color-scheme`), imported by `console.js` and `calibration.js`. The safety header is sticky at every width; when it wraps (tablets, phones) `web/sticky-header.js` lets the rows above the E-STOP scroll away and keeps the E-STOP row pinned. Polls repaint in place (rows are rebuilt only when their content changes), the first load shows skeletons, focus is always visible, and `prefers-reduced-motion` turns transitions off. The Agent bar (`agent-settings-*.js`, design 16) reads `GET /v1/console/agent-options`; its profile form takes every field's help text from the gateway's `profile_schema.fields`.

`web/check-console-shell.mjs` checks the whole console at 1440, 768, 390 and 360 px in light and dark (no horizontal scroll, E-STOP visible and clickable before and after scrolling, no row rebuilt by polling, skeletons, focus ring, reduced motion, the Agent bar follow-ups of design 16 §16) against the synthetic fixture, or with `NI_RUA_TARGET=real NI_RUA_REAL_MODE=claude_code|codex|opencode` against a real gateway app (`Workbench.app()` over a fake robot and agent). `web/check-agent-settings.mjs` covers the Agent bar interactions on the fixture. Both refuse port 8781.

## Building and Running Synthetic Checks

Install pinned frontend dependencies, run static checks, build the bundle, and run the Python terminal unit tests:

```sh
npm --prefix web ci
npm --prefix web run check
npm --prefix web run build
mamba run -n ni-rua python -m unittest discover -s tests -p test_terminal.py -v
```

To launch the deterministic synthetic browser fixture (which requires no model or GPU and tags every emitted record with `synthetic_fixture`):

```sh
mamba run -n ni-rua python web/fixture-server.py --port 8778
```

In a second shell, run the automated Playwright check against the fixture server (using either a Playwright-managed browser or a system Chromium/Chrome binary via `NI_RUA_BROWSER=chromium NI_RUA_BROWSER_EXECUTABLE=/usr/bin/google-chrome`):

```sh
NI_RUA_URL=http://127.0.0.1:8778 \
NI_RUA_SYNTHETIC=1 \
NI_RUA_BROWSER=firefox \
PLAYWRIGHT_MODULE=/path/to/playwright/index.mjs \
NI_RUA_SCREENSHOT_DIR=runs/web-synthetic-check \
node web/check-browser.mjs
```

Start a fresh `fixture-server.py` process before each full synthetic run because native approval requests are intentionally one-shot and cannot be reopened once resolved. The suite verifies independent motion and turn states, two-window writer lease handoffs (including rejecting a duplicate client ID lacking its lease token), native approvals and questions, task submission idempotency, trace identities and missing-feedback states, missing media handling, keyboard navigation, SSE deduplication and gap recovery, HTTP/WebSocket origin checks, and PTY session continuity across page refreshes, resizes, and tab close/reopen cycles. It also verifies that program-internal observations are distinguished from images delivered to the primary actor model, saving screenshots at `320`, `736`, `1024`, and `1440` pixel widths alongside a JSON report.

### Live / Recorded Gateway Checks

To validate against a running real Gateway service, omit `NI_RUA_SYNTHETIC`, point `NI_RUA_URL` at the gateway, and set `NI_RUA_REQUIRE_ACTION=1` to require at least one completed physical action. In this mode, the script verifies the action's input, start, and terminal RoboDojo observations, cross-checks run and action IDs, verifies readable planned/dispatched/actual trajectory artifacts (recorded under `action_evidence`), and—when an immutable scene version contains a tracked proxy—queries its conditional geometry and creates a verified geometry snapshot (recorded under `scene_evidence`), without submitting any new task or robot action.

## Phase Playlist and Recorded Video Verification

The primary review interface is a **phase playlist**: each native tool call groups its correlated Harness actions, the active turn prompt, and the model messages and reasoning immediately preceding and following the call. Multiple physical actions inside a single tool call remain individually selectable clips, while non-moving or unmapped stages retain their text records and are skipped during continuous video playback. Raw observations, events, and video exports remain accessible in collapsible evidence sections showing each camera's run, episode, generation, sampling metadata, and full `frame_mapping`.

Selecting an action stage enables start/end video seeking only when the recording contains exact simulation-step-to-PTS boundary records matching the action's physical identity (or an exact terminal-observation frame anchor when a declared step domain is absent; otherwise the UI explicitly reports the missing mapping). Container FPS metadata is displayed for reference only and is never used to synthesize playback PTS timestamps. If the browser lacks a required video codec, a direct download link to the original MP4 remains available.

Run the read-only video playback check against an existing replay server (uses only `GET`/`HEAD` requests, decodes all three camera streams, seeks across action start/end PTS boundaries, captures native canvas frames, verifies HTTP Range headers, and checks cross-generation rejection):

```sh
NI_RUA_URL=http://127.0.0.1:8767 \
NI_RUA_BROWSER=firefox \
PLAYWRIGHT_MODULE=/path/to/playwright/index.mjs \
NI_RUA_SCREENSHOT_DIR=runs/web-replay-video-check \
node web/check-replay-video.mjs
```

Set `NI_RUA_VIDEO_EXPORT_EVENT_ID` to target a specific immutable video export when multiple exports exist.

## Permanent Experiment Library

The experiment library (typically served on `http://127.0.0.1:8780` over `~/exp/ni-rua`; see [Experiment Library](../docs/development/experiment-library.md)) organizes runs into collapsible research directions. Selecting a research direction or experimental scheme displays an unlimited-column comparison matrix where every row is a benchmark case/scene and every column is a scheme version or registered baseline, showing the score, exact UTC-backed start/end timestamps, and inline video preview in each cell. Sticky row/column headers and independent scrolling keep large matrices navigable. Video players load lazily near the viewport, while clicking row playback loads and synchronizes all cells in that row by recording-relative elapsed seconds. Explicitly registered recovery runs merge into their logical parent experiment while preserving every attempt's timestamps, detail links, and the first formal score (including `0`). Switching to the **All Runs** tab displays the full searchable run catalog and detail drawer.

When loading the comparison matrix, the browser first fetches the compact group summary (with exact paired score intersections), then fetches full case lists for selected groups, and requests individual media manifests on demand for visible video cells. Switching to the **All Runs** tab fetches the paginated run catalog. JSON endpoints use private conditional caching (`ETag` / `If-None-Match`) and compression, while live SSE notifications drive real-time catalog updates (falling back to a 5-second poll only while SSE is disconnected and pausing polling while the tab is hidden).

To build an isolated preview bundle without touching a running server's `web/dist` directory and run the synthetic comparison matrix check:

```sh
node web/build.mjs --outdir /tmp/ni-rua-comparison-preview
PLAYWRIGHT_MODULE=/path/to/playwright/index.mjs \
NI_RUA_BROWSER=firefox \
NI_RUA_PREVIEW_ASSETS=/tmp/ni-rua-comparison-preview \
NI_RUA_SCREENSHOT_DIR=/tmp/ni-rua-comparison-check \
node web/check-comparisons.mjs
```

To run the read-only library check against a live library server:

```sh
PLAYWRIGHT_MODULE=/path/to/playwright/index.mjs \
NI_RUA_BROWSER=firefox NI_RUA_URL=http://127.0.0.1:8780 \
NI_RUA_SCREENSHOT_DIR=~/exp/ni-rua/service/validation/library \
node web/check-library.mjs
```

Completed original runs also provide a **Download Raw Trajectory** action, exporting a portable ZIP archive that can be passed directly to the [HD Replay CLI](../docs/development/hd-replay.md) for high-resolution re-rendering. The trajectory download check verifies HTTP attachment headers, `HEAD`/`Range` support, SHA-256 integrity, and `hd_replay --inspect` import on two completed runs:

```sh
PLAYWRIGHT_MODULE=/path/to/playwright/index.mjs \
NI_RUA_BROWSER=firefox NI_RUA_URL=http://127.0.0.1:8780 \
NI_RUA_PYTHON=/path/to/ni-rua/bin/python \
NI_RUA_RUN_IDS=<catalog_id_1>,<catalog_id_2> \
NI_RUA_SCREENSHOT_DIR=/path/to/new-validation-output \
node web/check-trajectory-download.mjs
```

## Phase Review, Camera Players, and Native TUI Recovery

The run detail view features two synchronized video players: the head camera (`cam_head`) as the primary view, plus a toggleable left or right wrist camera (`cam_left_wrist` / `cam_right_wrist`). Phase navigation and the **Observations & Actions** timeline share selection state: switching wrist cameras preserves the active phase, and scrubbing the video timeline updates the highlighted phase. Task display labels are defined in `task-labels.js` (leaving unrecognized task identifiers untranslated).

The context pane displays only saved prompts, visible assistant messages and reasoning, exact tool call arguments, tool outputs, and Harness receipts (leaving empty reasoning blocks explicitly empty). For legacy terminal logs lacking an item-to-byte-offset index, the UI scrolls to a unique matching visible text line once history synchronization completes, or notes when a match is non-unique or absent.

Run the read-only phase and camera player verification suite against an archived run with:

```sh
PLAYWRIGHT_MODULE=/path/to/playwright/index.mjs \
NI_RUA_BROWSER=chromium NI_RUA_BROWSER_EXECUTABLE=/path/to/google-chrome \
NI_RUA_URL=http://127.0.0.1:8780 NI_RUA_RUN_ID=<catalog_id> \
node web/check-phases.mjs
```

### Asset Cache Busting and Packaging

`node web/build.mjs` stamps every HTML `<script>` and `<link>` asset URL with the SHA-256 digest of its file bytes. The server delivers HTML documents with `Cache-Control: no-store` and static assets with `no-cache, max-age=0, must-revalidate` so browser and CDN caches never serve stale bundles across deployments. For wheel distribution builds, running `node web/build.mjs --outdir ../ni_rua/web_assets` packages the built assets directly into the Python wheel without maintaining a duplicate source tree.

### Observer Task Review, Native Codex TUI Replay, and Incremental Pagination

- **Observer Task Review (`/v1/task-review`)**: Both the library sidebar and the standalone run viewer display the official task instruction, statically parsed scoring rules, and persisted evaluator summary (rendering human-readable predicate breakdowns when the repository commit and diff fingerprint match the reviewed source hash). Native process scores, normalized scores, and binary success are reported separately, and unscored episodes never fabricate a formal score from partial snapshots.
- **Native Codex TUI Replay (`check-native-terminal.mjs`)**: Opens the native ANSI TUI tab by default. When an original PTY recording is healthy, it streams directly; for completed archives where the PTY log is missing or failed account bootstrap, the library renders a fresh read-only ANSI capture by running the installed Codex CLI against an isolated copy of the saved session (with no credentials or network access) and labels it as a post-hoc rendering alongside a download link for the original log.
- **Incremental Record Cursors (`check-task-review.mjs`)**: `/v1/records` paginates large runs using a 2,000-item current window (retaining up to 5,000 stage events and 96 camera exports in browser memory). Clicking **Earlier Records** loads a 200-item historical window with up to 5,000 correlated events and provides a direct link to the complete `/v1/events?after=0` stream (resumable from any `server_seq`) whenever event limits are reached.
