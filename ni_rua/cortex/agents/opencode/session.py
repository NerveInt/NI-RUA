"""OpenCode agent backend session for the pinned ``opencode serve`` 2.2.7 (design 11; design 04 §2.2-§2.3, §3.7,
R4.2-R4.4; plan 07 §1, §3).

``OpenCodeSession`` drives one private ``opencode serve`` for one run over its ``/api/*`` HTTP API
(``OpenCodeClient``). Registry tools reach OpenCode only through the host MCP endpoint (``McpToolBridge``, model-visible
prefix ``ni_rua_``, Code Mode off); model requests only through the run's relay (``ModelRequestProxy`` with the
``opencode`` client profile of ``AnthropicMessagesWire`` or ``OpenAIResponsesWire`` per
``agent.opencode.upstream.wire_api``; egress scrub = the wire's ``project_request``). OpenCode sees a placeholder
credential, a clean environment and one read-only config dir whose generated ``opencode.json`` the host rewrites to
declare a model selected during the run (``config.build_launch`` / ``add_model`` / ``write_config``). Instructions
travel in the config ``instructions`` file, never in a prompt.

Start order: resume identity check -> upstream credentials (and the provider listing, read in the background) ->
relay -> MCP endpoint -> model declaration (limits from the explicit config, else the provider listing, else the
fallbacks; ``config.resolve_limits``) -> launch (managed-config check, exact ``--version``, sha256, ``rg``, private
dirs, config file) -> spawn on a free loopback port -> ``GET /api/info`` (503 while starting; pinned version) ->
``GET /openapi.json`` route list -> ``GET /api/model`` = the declaration -> ``GET /api/mcp`` reports ``ni_rua``
connected -> session create (with the model ref) or existence check -> history adoption (``projector.sync``) ->
identity record -> event, reconcile, watch and tree threads -> ``idle``.

Threads
-------
* callers (gateway ``asyncio.to_thread``): the synchronous facade; blocking HTTP calls are never made under the
  session lock.
* ``opencode-events`` (daemon): the ``/api/event`` reader (``OpenCodeClient.event_stream``, reconnecting). Events are
  hints (no replay): live transcript (``project_event``), the early compaction signal, subagent discovery, permission
  auto-rejection, adoption of a native execution and reconcile requests; nothing on this thread finishes a turn.
* ``opencode-reconcile`` (daemon): the single worker that decides turns from an authoritative snapshot:
  ``GET /api/session/active`` -> main ``GET .../inbox`` -> main ``GET .../message`` (every page) -> adoption ->
  ``projector.sync`` (main + subagents) -> ``submissions()`` under the session lock (``steer()`` writes under it) ->
  ``verdict.turn_verdict`` -> ``SettleGuard`` -> ``IncompleteTurnGuard``. Requests are merged while waiting; a
  watchdog reconcile runs every ``watchdog_s`` (a silently lost event stream; with no turn running it reads only
  ``GET /api/session/active`` and goes on only when a session is active, so missed native work is still adopted), and
  a recheck is scheduled while a guard waits for its second observation.
* ``opencode-watch`` (daemon): ``process.wait()``; an exit that is not part of ``close()`` -> ``disconnected``.
* ``opencode-tree`` (daemon): periodic snapshot of the server's process tree (teardown after a crash).
* MCP handler threads + ``agent-tools`` workers (``McpToolBridge``), relay threads (``_model_event``,
  ``_relay_discontinuity``, ``_note_request_tools``) and the provider-listing reader.

Turn contract (design 11 §1.5-§1.8)
-----------------------------------
* Admission: ``POST /api/session/{id}/prompt`` answered 200 with the user message id (the turn boundary). A 4xx is a
  definite rejection (state restored, ``AgentRejected``); a 5xx, timeout or socket error is unknown admission ->
  ``disconnected``, never retried. The host ``userMessage`` row is written BEFORE the request, bound to the returned
  id on 200 and withdrawn on a 4xx.
* Model and effort: session state of OpenCode (``POST .../model`` with ``{id, providerID: relay, variant}``), switched
  only at the start of a host turn when ``set_model`` / ``set_effort`` selected something else; steers and a joined
  native turn never switch, so no switch happens mid-turn.
* Turn end: the last ``idle`` message after the boundary while no session of the private server is active and the
  inbox is empty (``verdict``); subagent turns are confirmed by ``SettleGuard`` (the same quiescent snapshot twice
  ``settle_gap_s`` apart); ``IncompleteTurnGuard`` turns a verdict that stays non-terminal while nothing is active into
  ``failed: opencode_turn_incomplete`` (a server that died mid-turn writes no ``idle``).
* Steer: ``prompt`` while running (joins at the next step boundary; on an idle main session held open by a background
  subagent it starts an execution of the same turn).
* Interrupt: relay block first (``cancel_pending`` plus a reopenable tombstone on the main session and every known
  subagent session) -> ``POST .../interrupt`` on the main session (synchronous, cascades to foreground subagents) ->
  the same on every subagent session still active (a background child is not reached once its parent is idle).
* Native wake-ups (``capabilities.native_wakeup``, design R4.2): a background subagent keeps the turn open while it
  runs (rule 1 of the verdict: every registry call it makes runs inside that turn); a native execution that starts
  while no host turn runs (a background shell's or subagent's completion report) is adopted as a NI-RUA turn
  (``agent.turn_started {origin: "native_wakeup"}``) whose boundary is the last message of the snapshot that ended the
  previous turn; a ``run_turn`` during it joins it.
* Permission requests (none with the generated config; a future default ``ask`` would wait for a human): answered
  ``reject`` at once (``agent.permission_auto_rejected``); OpenCode then ends the execution without an ``idle`` and the
  turn fails ``opencode_permission_rejected``.

Subagents (design R4.3): child sessions (``session.created`` with a known ``parentID``, or a ``subagent`` tool item
whose ``state.metadata.sessionID`` names them) are followed: their rows carry ``thread_id`` = the child session id and
``parent_item_id`` = the ``subagent`` tool row that started them, so they are never final-answer text.

Nothing written here is model-visible: transcript rows, events, raw logs, identity and close reports are observer data.
"""
import dataclasses
import hashlib
import os
import secrets
import socket
import stat
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

from ....core.contracts import digest, new_id
from ....core.storage.images import store_image_settings
from ...session.actor_source import snapshot_actor_source
from ...session.agent_sandbox import ACTOR_ISOLATION, AgentSandbox
from ...session.egress_cache import EgressProjectionCache
from ...relay.payload import upstream_auth_headers_copy
from ...session.wire import AnthropicMessagesWire, OpenAIResponsesWire
from ..base import (AgentCapabilities, AgentRejected, AgentSession, AgentTransportUnknown, STATE_CLOSED,
                    STATE_DISCONNECTED, STATE_IDLE, STATE_NOT_STARTED, STATE_REPLAY, STATE_RUNNING, UNCONNECTED_STATES,
                    read_agent_identity)
from ..identity import write_identity
from ..isolation import build_snapshot, prepare_ipc_dir, process_table, stop_agent_tree
from ..mcp_session import GENERATION_COUNTING, STALE_TOOL_STATES, McpSessionMixin, McpToolBridge
from ..transcript import TranscriptWriter
from . import config as opencode_config
from .client import OpenCodeHttpError, OpenCodeClient, _checked_file_uris, is_definite_rejection, model_ref
from .projector import TOOL_NAME_PREFIX, OpenCodeProjector, subagent_links
from .upstream_file import configured_wire, read_opencode_config_upstream
from .verdict import IncompleteTurnGuard, SettleGuard, active_sessions, inbox_ids, timeline, turn_verdict

__all__ = ["OpenCodeSession", "BACKEND_KIND", "MCP_SERVER_NAME", "NATIVE_DATA_DIR", "WIRES", "naming_note",
           "compose_instructions"]

BACKEND_KIND = "opencode"
MCP_SERVER_NAME = "ni_rua"          # OpenCode names MCP tools "<server>_<tool>" -> TOOL_NAME_PREFIX "ni_rua_"
# Run-relative directory holding OpenCode's own session database (needed for resume; hidden from the actor).
NATIVE_DATA_DIR = "private/opencode/xdg/data"
# Upstream wires the relay can serve for OpenCode (``openai_chat`` has no relay wire adapter in this checkout).
WIRES = {"anthropic_messages": AnthropicMessagesWire, "openai_responses": OpenAIResponsesWire}
SESSION_TITLE = "ni-rua-actor"      # explicit non-default title (no title-model call)
# Message types that belong to an execution; one listed after the accounted boundary while no turn runs is native
# work to adopt (``verdict`` uses the same set).
_EXECUTION_TYPES = frozenset({"user", "synthetic", "assistant", "compaction"})
# Event frames after which a reconcile is requested (an execution ended, or a step boundary delivered input).
_RECONCILE_EVENTS = frozenset({"session.execution.succeeded", "session.execution.failed",
                               "session.execution.interrupted", "session.inbox.delivered"})
# Reconcile reasons that also read the pending permission requests of the active sessions (a missed
# ``permission.asked`` frame would otherwise leave the execution waiting forever).
_PERMISSION_POLL_REASONS = frozenset({"watchdog", "event_stream_gap", "event_stream_reconnected"})
# Frames after which a native shell may have started (the tree refresher re-reads the process table soon).
_SHELL_HINTS = frozenset({"session.tool.called", "session.shell.started"})


def naming_note():
    """The one-sentence tool-name-prefix note appended to the instructions (design §3.5)."""
    return (f"Tool names in these instructions are written without the prefix '{TOOL_NAME_PREFIX}' that your tool "
            "list shows; they refer to the same tools.")


def compose_instructions(instructions):
    """Instructions file text: the (already presented) ``instructions`` followed by the naming note, separated by a
    blank line. An empty ``instructions`` yields the note alone."""
    text = (instructions or "").rstrip()
    return (text + "\n\n" if text else "") + naming_note() + "\n"


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _free_loopback_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _error_code(error):
    return f"status_{error.status}" if isinstance(error, OpenCodeHttpError) else type(error).__name__


def _request_tool_names(request):
    """Names of the tools one forwarded model request declares (Anthropic ``tools[].name``, Responses function tools'
    ``name``; hosted tools without a name are skipped)."""
    tools = request.get("tools") if isinstance(request, dict) else None
    return [tool["name"] for tool in tools or () if isinstance(tool, dict) and isinstance(tool.get("name"), str)]


class _SubagentTranscript(TranscriptWriter):
    """Transcript writer of one OpenCode subagent (child) session (design R4.3).

    Rows carry ``thread_id`` = the child session id, ``parent_item_id`` = the ``subagent`` tool row of the parent
    session that started the child (unless the projector supplies a more specific parent, e.g. a compaction row) and
    the main writer's CURRENT turn when first written. A row keeps the parent it was first written with: a subagent
    continued by a later ``subagent`` call (``sessionID``) hangs its new rows under the new tool row
    (``parent_item_id`` is re-pointed by the session). They are never stamped as final answer (``stamp_final_answer``
    reads only the main thread's top-level rows). ``OpenCodeProjector(_SubagentTranscript(...))`` projects the child.
    """

    def __init__(self, main, child_id, parent_item_id):
        self._main = main
        super().__init__(main.store, backend=main.backend, thread_id=child_id)
        self.parent_item_id = parent_item_id

    @property
    def turn_id(self):
        return self._main.turn_id

    @turn_id.setter
    def turn_id(self, value):
        """Turns are owned by the main writer; the base constructor's reset is ignored."""

    def _row(self, identifier, fields, *, parent_item_id=None, **keywords):
        with self.lock:
            if parent_item_id is None:
                previous = self.store.transcript_item(identifier, None)
                if not (previous and previous.get("parent_item_id")):
                    parent_item_id = self.parent_item_id
            return super()._row(identifier, fields, parent_item_id=parent_item_id, **keywords)


class OpenCodeSession(McpSessionMixin, AgentSession):
    """OpenCode backend (``agent.backend = "opencode"``). Public contract: ``AgentSession``.

    Constructor (called by ``factory.create_session``): ``run_dir`` (run directory), ``store`` (run Store),
    ``tool_handler`` (``Harness.handle_tool``), ``price_card`` (frozen card or None), ``config`` (``Config``; default
    ``Config.load()``), ``task_planning`` (unused: the decision reminder reaches this backend through
    ``on_context_reset``). Test seams: ``client_factory`` (``(base_url, password[, unix_socket]) ->
    OpenCodeClient``-like; default ``OpenCodeClient``) and ``launch_options`` (extra keywords for
    ``config.build_launch``: ``ripgrep``, ``managed_config_paths``, ``check_version``, ``base_env``). Constructing a
    session starts nothing (a replay viewer of an OpenCode archive needs no binary).
    """
    backend_kind = BACKEND_KIND
    counting = GENERATION_COUNTING
    # steer: prompt while running. live_effort: the effort is the variant of the session's model ref, switched at the
    # next host turn. native_wakeup: background subagents/shells report back and start an execution by themselves
    # (adopted, design R4.2). terminal_attach: ``opencode --server`` renders the live main session (not sandboxed).
    capabilities = AgentCapabilities(steer=True, live_effort=True, native_wakeup=True, sandbox=True,
                                     terminal_attach=True, turn_images=True, compaction_signal="native+relay")
    context_delivery = "delta"
    tool_name_prefix = TOOL_NAME_PREFIX

    def host_tool_output_truncation_limit(self):
        """OpenCode 工具输出受 tool_output.max_bytes / max_lines 双阈值约束。"""
        return (opencode_config.TOOL_OUTPUT["max_bytes"], opencode_config.TOOL_OUTPUT["max_lines"], False)

    def supports_turn_images(self) -> bool:
        """仅当会话支持多模态输入且当前声明模型未标记 image_input=False 时返回 True。"""
        if not self.capabilities.turn_images:
            return False
        lock = getattr(self, "lock", None)
        if lock is not None:
            with lock:
                models = list(getattr(self, "_models", ()) or ())
                active = getattr(self, "model", None)
        else:
            models = list(getattr(self, "_models", ()) or ())
            active = getattr(self, "model", None)
        current = next((spec for spec in models if spec.id == active), None)
        return current is None or current.image_input is not False

    def _sync_capabilities(self):
        """按协议线与 tool_search 配置同步动态能力（service_tier / deferred_tools）。"""
        if self.wire is not None:
            supports_tier = bool(self.wire.supports_service_tier)
        else:
            wire_cls = WIRES.get(self.wire_api)
            supports_tier = bool(wire_cls.supports_service_tier) if wire_cls is not None else False
        self.capabilities = dataclasses.replace(
            type(self).capabilities,
            service_tier=supports_tier,
            deferred_tools=bool(self.oc_config.tool_search),
        )
    # Provider listing (``model_options`` suggestions, limits and image flag of a declared model): read at start for
    # these ``upstream.source`` values (``ModelListingMixin``; ``profile`` through the profile catalogue cache at the
    # wire's ``models_url``); ``codex_home`` has no listing to read.
    model_catalogue_sources = ("explicit", "opencode_config", "profile")
    # Timeouts and periods in seconds (plan 07 §3). startup: readiness poll cap; health_poll: poll interval;
    # health_poll_request: socket timeout of one readiness poll (a connection made while the server is still coming up
    # can stall far longer, observed ~30 s on Linux); request: every non-streaming API call; interrupt: POST
    # .../interrupt from interrupt() (below the gateway's 45 s bound); close_interrupt: the same during close();
    # watchdog: reconcile period (with no turn: one GET /api/session/active); incomplete_gap: minimum gap between the
    # two idle observations of IncompleteTurnGuard; settle_gap: SettleGuard's gap for a turn with subagents;
    # thread_join: per daemon thread at close; events_ready: wait for the first /api/event connection at start
    # (events before it are recovered by reconcile); mcp_connect: wait for GET /api/mcp to report the ni_rua server connected (OpenCode's own MCP startup
    # timeout is 30 s); mcp_poll: its poll interval; model_declare: wait for GET /api/model to list a model written into
    # the config file (once more after POST /api/location/reload); model_declare_poll: its poll interval;
    # model_catalogue_wait: how long the start waits for the provider listing before it declares the configured model
    # without it (fallback limits).
    startup_timeout_s = 30.0
    health_poll_s = 0.05
    health_poll_request_timeout_s = 1.0
    request_timeout_s = 30.0
    interrupt_timeout_s = 15.0
    close_interrupt_timeout_s = 2.0
    watchdog_s = 10.0
    incomplete_gap_s = 2.0
    settle_gap_s = 0.5
    thread_join_timeout_s = 2.0
    events_ready_timeout_s = 2.0
    mcp_connect_timeout_s = 30.0
    mcp_poll_s = 0.05
    model_declare_timeout_s = 2.0
    model_declare_poll_s = 0.1
    model_catalogue_wait_s = 5.0
    # Messages per GET .../message page (1..200). A page carries the tool results' inline images and one answer is
    # capped at 256 MiB, so the binary's own default page size is kept rather than the 200 maximum.
    message_page_size = 50
    # Interval (s) at which the server's process tree is re-snapshotted while it runs (also right after a native shell
    # starts), so the detached tool shells of a server that dies can still be found and stopped at close.
    tree_refresh_s = 2.0
    # Consecutive failed reconciles after which a dead server is declared disconnected (plan 07 §3.9).
    reconcile_failure_limit = 3

    def __init__(self, run_dir, store, tool_handler, *, price_card=None, config=None, task_planning=None,
                 client_factory=None, launch_options=None):
        from ....core.config import Config
        from ....core.presentation.actor_presentation import ActorPresentation
        self.config = config if config is not None else Config.load()
        self.oc_config = self.config.agent.opencode
        self.root = Path(run_dir).resolve()
        self.store, self.tool_handler, self.task_planning = store, tool_handler, task_planning
        self.presentation = ActorPresentation(store)
        self.model = self.config.model.name
        self.active_turn_model = self.model
        self.effort = self.config.model.effort
        upstream = self.oc_config.upstream
        self.provider = upstream.provider
        self.wire_api = configured_wire(upstream)
        self.price_card = price_card
        self.sandbox_enabled = bool(self.config.features.sandbox_enabled)
        self.isolation = ACTOR_ISOLATION if self.sandbox_enabled else "disabled"
        self.sandbox = None
        self.ipc = None
        # 主链与子智能体已通过作用域隔离，按配置采用 effective_context_delivery。
        self.context_delivery = self.oc_config.effective_context_delivery
        self.actor_dir, self.private = self.root / "actor", self.root / "private"
        self.wire = None
        self._egress_cache = EgressProjectionCache()
        self._sync_capabilities()
        self.lock = threading.RLock()
        self._close_lock = threading.Lock()
        self._projector_lock = threading.Lock()
        self._model_lock = threading.Lock()
        self.state, self.thread_id, self.turn_id = STATE_NOT_STARTED, None, None
        self.turn_done, self.turn_result = threading.Event(), None
        self.turn_origin = None
        self.last_activity_at = time.monotonic()
        self.model_limit_reached = threading.Event()
        self.model_proxy = self.bridge = self.transcript = self.projector = self.launch = self.client = None
        self.process = None
        self.closing, self.close_report = False, None
        self._client_factory = client_factory or (
            lambda base_url, password, unix_socket=None: OpenCodeClient(
                base_url, password, timeout_s=self.request_timeout_s, unix_socket=unix_socket))
        self._launch_options = dict(launch_options or {})
        self._started = False
        self._identity = None
        self._instance_id = None          # runtime/agent.instance_id; also the process marker of the server env
        self._teardown_confirmed = True
        self._process_gone = False
        self._interrupt_requested = False
        self._submitting = 0
        self._guard = IncompleteTurnGuard(min_gap_s=self.incomplete_gap_s)
        self._settle = SettleGuard(min_gap_s=self.settle_gap_s)
        self._children = {}             # child session id -> {parent_session, parent_item_id, turn_id, projector, items}
        self._compaction_keys = set()
        self._route_digest = None
        self._instructions_sha256 = None
        self._tool_names = frozenset()
        self._deferred_tool_names = frozenset()
        denied_native = set()
        if not getattr(self.oc_config, "allow_shell", True):
            denied_native.update(("shell", "bash", "execute"))
        if not getattr(self.oc_config, "subagents", True):
            denied_native.update(("subagent", "task"))
        self._denied_native_tools = frozenset(denied_native)
        if not getattr(self.oc_config, "allow_shell", True) and not getattr(self.oc_config, "subagents", True):
            self.tree_refresh_s = 15.0
        self._catalogue = None
        # Model declarations: the ModelSpec list declared on the relay provider (first = the configured model) and the
        # config document last written; ``_native_ref`` = (model, variant) of the native session's model ref.
        self._models = []
        self._config_doc = None
        self._native_ref = None
        # Native tools (provenance): tool names of the main thread's first tools-declaring request the relay forwarded;
        # ``_call_tools`` holds candidate names per logical call until the relay reports the call's purpose.
        self._native_tools = None
        self._call_tools = {}
        # Turn bookkeeping. ``_accounted``: id of the last message of the snapshot that ended the previous NI-RUA turn
        # (or of the history at start): native work listed after it while no turn runs is adopted.
        # ``_turn_boundary``: the boundary of an adopted turn. ``_host_prompted``: the running turn carries a host prompt.
        self._accounted = None
        self._turn_boundary = None
        self._host_prompted = False
        self._permission_rejected = False
        self._rejected_permissions = set()
        self._unsolicited_noted = False
        self._native_busy = False          # a session of the server was last seen active (events / reconcile)
        self._activity_seq = 0             # monotonic counter bumped on session.execution.started under self.lock
        self._stop = threading.Event()
        self._tree_wake = threading.Event()
        self._known_tree = {}             # pid -> ProcessInfo of every server-tree member seen while the server ran
        self._recheck_at = None
        self._events_ready = threading.Event()
        self._stream_connections = 0
        self._gap_open = False
        self._reconcile_cond = threading.Condition()
        self._reconcile_reasons = []
        self._reconcile_failures = 0
        self._threads = []

    # ---- start ---------------------------------------------------------------------------------------------
    def start(self, tools, instructions, resume_thread_id=None):
        """Start ``opencode serve`` and create (or resume ``resume_thread_id``) the main session; return its id.

        Order: see the module doc. Any failure closes the session and re-raises (not ``disconnected``: no turn was
        admitted)."""
        with self.lock:
            if self._started or self.closing:
                raise ValueError("session_already_started")
            self._started = True
        try:
            return self._start(tools, instructions, resume_thread_id)
        except BaseException:
            try:
                self.close()
            except Exception:  # noqa: BLE001 - the start failure is the error to report
                pass
            raise

    def _start(self, tools, instructions, resume_thread_id):
        resume = resume_thread_id is not None
        previous = read_agent_identity(self.store)
        executable = self.oc_config.executable
        if not executable:
            raise ValueError("opencode_executable_not_set: set agent.opencode.executable")
        if resume:
            self._validate_resume(previous, resume_thread_id, tools)
            self._stop_previous_instance(previous)     # a crashed gateway leaves its detached server running
        # Minted before the launch: the server environment carries it as the process marker of this session.
        self._instance_id = new_id("agent")
        upstream_url, upstream_headers = self._upstream()
        if self.effort not in self.effort_choices:
            raise ValueError(f"agent_effort_unsupported:{self.backend_kind}:{self.effort}: choose model.effort from "
                             f"{', '.join(self.effort_choices)}")
        self._start_model_catalogue(upstream_url, upstream_headers)
        self._tool_names = frozenset(tool["name"] for tool in tools)
        self._deferred_tool_names = (
            frozenset(TOOL_NAME_PREFIX + tool["name"] for tool in tools if tool.get("deferLoading"))
            if self.oc_config.tool_search
            else frozenset()
        )
        self.store.set_state("runtime", "price_card", self.price_card)
        upstream = self.oc_config.upstream
        self._start_relay(upstream_url, upstream_auth_headers=upstream_headers,
                          request_projection=self._project_request, default_thread_id=resume_thread_id,
                          image_settings=store_image_settings(self.store, self.config.images.to_dict()),
                          tool_result_images=upstream.tool_result_images, request_context=self._note_request_tools)
        self.bridge = McpToolBridge(self, tools, parallel_hints=self.tool_parallel_hints,
                                    server_name=MCP_SERVER_NAME, schema_max_depth=upstream.tool_schema_max_depth).start()
        text = compose_instructions(self.presentation.present_text(instructions))
        self._instructions_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        self._await_model_catalogue(self.model_catalogue_wait_s)
        self._models = [self._model_spec(self.model)]
        self.root.mkdir(parents=True, exist_ok=True)
        options = dict(self._launch_options)
        options.setdefault("base_env", {"LANG": "C.UTF-8"})
        options.setdefault("allow_shell", getattr(self.oc_config, "allow_shell", True))
        options.setdefault("allow_external_read", getattr(self.oc_config, "allow_external_read", True))
        max_spawn_attempts = 1 if self.sandbox_enabled else 3
        for attempt in range(max_spawn_attempts):
            port, password = _free_loopback_port(), secrets.token_urlsafe(32)
            self.launch = opencode_config.build_launch(
                self.root, executable=executable, models=self._models, wire_api=self.wire_api,
                relay_base_url=self.model_proxy.base_url, instructions_text=text, port=port, password=password,
                mcp_url=self.bridge.url, mcp_headers=self.bridge.headers, subagents=self.oc_config.subagents,
                thinking_display=self.oc_config.thinking_display, verbosity=self.config.model.verbosity,
                sandbox=self.sandbox_enabled, instance_id=self._instance_id, **options)
            self._config_doc = self.launch.config
            argv, env, unix_socket = list(self.launch.argv), self.launch.env, None
            if self.sandbox_enabled:
                prefix, env, unix_socket = self._prepare_sandbox(port)
                argv = prefix + argv
            else:
                os.chmod(self.launch.config_dir if hasattr(self.launch, "config_dir") else self.launch.config_path.parent, 0o555)
            with open(self.private / "agent.log", "ab") as log:     # the server keeps its own descriptor
                self.process = subprocess.Popen(
                    argv, env=env, cwd=self.launch.cwd, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
            base_url = f"http://127.0.0.1:{port}"
            self.client = (self._client_factory(base_url, password) if unix_socket is None
                           else self._client_factory(base_url, password, unix_socket=unix_socket))
            try:
                self._await_health()
                break
            except RuntimeError as err:
                if attempt + 1 >= max_spawn_attempts or not str(err).startswith("opencode_exited_during_startup:"):
                    raise
                try:
                    os.chmod(self.launch.config_path.parent, 0o700)
                except OSError:
                    pass
                self.process = None
        self._route_digest = opencode_config.check_route_list(self.client.routes())
        # The running server must serve exactly the declaration (models, limits, input, variants); OpenCode would
        # otherwise silently use its own defaults for a variant or limit it did not read.
        opencode_config.check_served_models(self.client.models(), self._config_doc)
        self._await_mcp_connected()
        variant = self._variant()
        if resume:
            if not self.client.session_exists(resume_thread_id):
                raise ValueError("resume_native_session_missing: the OpenCode session of this run is absent")
            thread_id = resume_thread_id
            native = _mapping(self.client.session_info(thread_id).get("model"))
            self._native_ref = (native.get("id"), native.get("variant"))
        else:
            thread_id = self.client.create_session(SESSION_TITLE, model=model_ref(
                opencode_config.RELAY_PROVIDER_ID, self.model, variant))
            self._native_ref = (self.model, variant)
            # The relay had to start before the session id existed (the id comes from the server, whose config
            # names the relay); no model request can precede the session, so the attribution fallback is set now.
            self.model_proxy.default_thread_id = thread_id
        self.thread_id = thread_id
        self.model_proxy.register_thread(thread_id, "actor", thread_id)
        self.transcript = TranscriptWriter(self.store, backend=self.backend_kind, thread_id=thread_id)
        self.projector = OpenCodeProjector(self.transcript)
        history = timeline(self.client.messages(thread_id, page_size=self.message_page_size))
        self._link_children(history, self.thread_id)
        children = self._fetch_children(all_turns=True)
        child_projectors = self._child_projectors(children)
        with self._projector_lock:
            self.projector.sync(history)       # model switches of the adopted history are not reported again
            self._sync_children(children, child_projectors)
        self._accounted = history[-1]["id"] if history else None
        identity = write_identity(
            self.store,
            backend=self.backend_kind, thread_id=thread_id, instance_id=self._instance_id, pid=self.process.pid,
            model=self.model, provider=self.provider, effort=self.effort, isolation=self.isolation,
            tools_digest=digest(tools), instructions_digest=self._instructions_sha256, counting=self.counting,
            wire_api=self.wire_api, cli_version=opencode_config.OPENCODE_VERSION,
            cli_executable=self.launch.executable, cli_sha256=self.launch.sha256,
            native=dict(data_dir=NATIVE_DATA_DIR), resumed=resume, clean_shutdown=False)
        self._identity = identity
        if resume:
            clean = (previous or {}).get("clean_shutdown") is True
            if not clean:
                # OpenCode repairs its own dangling tool call ("Tool execution interrupted"); evidence delivered
                # before the crash can no longer be assumed in context. The Harness unknown-action gate stays
                # authoritative for physical state.
                self._invalidate_context("agent_resume_after_unclean_exit", marker=new_id("resume"))
            self.transcript.event("agent.history_recovered", dict(clean=clean, delta_history_complete=False,
                                                                  messages=len(history)))
        self._start_threads()
        self._events_ready.wait(self.events_ready_timeout_s)
        with self.lock:
            if self.state == STATE_NOT_STARTED:     # a native execution seen while connecting may already be adopted
                self.state = STATE_IDLE
        return thread_id

    def _prepare_sandbox(self, port):
        """Build and prepare the bubblewrap sandbox of the server (design R4.4); return
        ``(command prefix, environment, host path of the server's inbound unix socket)``.

        Outbound: the relay and the MCP endpoint (same loopback port numbers inside and outside). Inbound: the
        server's own ``port`` only, as ``<ipc>/inbound/<port>.sock``; no host TCP listener. ``/harness`` is the
        ``actor-source`` snapshot (the Workbench writes it too; the copy is idempotent)."""
        ports = [urlsplit(self.model_proxy.base_url).port, urlsplit(self.bridge.url).port]
        if port in ports:
            raise ValueError("opencode_server_port_conflicts_with_relay_port")
        self._prepare_ipc()
        source = snapshot_actor_source(self.private / "actor-source")
        spec = opencode_config.sandbox_spec(self.launch, harness_source=source)
        self.sandbox = AgentSandbox(self.private / "sandbox", self.ipc, self.actor_dir, ports=ports,
                                    inbound_ports=[port], packages=True, **spec)
        prefix, overrides = self.sandbox.prepare()
        return list(prefix), {"PATH": os.defpath, **overrides}, str(self.sandbox.inbound_socket_path(port))

    def _prepare_ipc(self):
        """Short run-owned IPC directory (Unix socket paths fit sockaddr_un), same layout as Codex/Claude Code."""
        self.ipc = prepare_ipc_dir(
            self.root,
            prefix="opencode",
            longest="sandbox-owner.sock",
            ignore_relative_runtime_dir=True,
            require_runtime_dir_private=False,
        )

    def _validate_resume(self, previous, thread_id, tools):
        if not previous:
            raise ValueError("resume_agent_identity_missing: the run has no recorded agent session")
        expected = dict(backend=self.backend_kind, thread_id=thread_id, isolation=self.isolation,
                        tools_digest=digest(tools), model=self.model, provider=self.provider, counting=self.counting,
                        wire_api=self.wire_api, cli_version=opencode_config.OPENCODE_VERSION)
        for key, value in expected.items():
            if previous.get(key) != value:
                raise ValueError(f"resume_agent_identity_conflict:{key}: resume with the recorded configuration "
                                 "or start a fresh run")

    def _actor_upstream(self):
        """``McpSessionMixin`` hook: the actor upstream (``agent.opencode.upstream``)."""
        return self.oc_config.upstream

    def _upstream(self):
        """``(base_url, host credential headers)`` of ``agent.opencode.upstream`` and the relay wire (``self.wire``).

        ``source="profile"``: endpoint, key and wire from the provider profile ``settings``
        (``profiles.resolve_profile_upstream``: the file must load, hold a key, serve the configured wire and still
        match the configuration's non-secret fields, else ``ValueError("profile_<code>:<file>: ...")``); the loaded
        profile is kept in ``_profile`` for the listing and the provenance. The profile's compat opt-ins and
        ``backends.opencode.thinking_display`` reach the relay, the MCP catalogue and the declared variants through the
        configuration fields ``Config.load`` / ``profiles.apply_profile`` wrote from it, as with every other source.
        ``source="explicit"``: ``wire_api`` (anthropic_messages | openai_responses), ``base_url`` and the key read
        from the gateway's environment variable ``api_key_env`` (never stored, never given to the agent).
        ``source="codex_home"``: the Codex credential source (``model.codex_home``) on the Responses wire, resolved
        exactly as the Codex backend resolves it. ``source="opencode_config"``: endpoint, key and wire from the
        OpenCode config file ``settings`` (``upstream_file.read_opencode_config_upstream``; V1 or V2 shape)."""
        upstream = self.oc_config.upstream
        resolved = None
        if upstream.source == "profile":
            from ..profiles import resolve_profile_upstream
            resolved = resolve_profile_upstream(upstream, self.backend_kind)
            self._profile = resolved.profile
            self.wire_api = resolved.wire_api
        elif upstream.source == "opencode_config":
            resolved = read_opencode_config_upstream(upstream.settings, provider=upstream.provider,
                                                     wire_api=upstream.wire_api)
            self.wire_api = resolved.wire_api
        if self.wire_api is None:
            raise ValueError("opencode_upstream_incomplete: set agent.opencode.upstream.wire_api, base_url and "
                             "api_key_env, or source=codex_home")
        if self.wire_api not in WIRES:
            raise ValueError(f"opencode_upstream_wire_unsupported:{self.wire_api}: set agent.opencode.upstream."
                             "wire_api to anthropic_messages or openai_responses")
        self.wire = WIRES[self.wire_api](client=BACKEND_KIND)
        self._sync_capabilities()
        if resolved is not None:
            return resolved.base_url, upstream_auth_headers_copy(resolved.headers, self.wire.auth_header_names)
        if upstream.source == "codex_home":
            import tomllib
            from ..codex.session import _model_provider, _upstream_auth
            if self.wire_api != "openai_responses":
                raise ValueError("opencode_upstream_codex_home_requires_openai_responses")
            source = Path(self.config.model.codex_home).expanduser()
            native = tomllib.loads((source / "config.toml").read_text())
            _name, provider, url = _model_provider(native, source)
            headers = _upstream_auth(provider, source)
            return url.rstrip("/"), upstream_auth_headers_copy(headers, self.wire.auth_header_names)
        if not upstream.base_url:
            raise ValueError("opencode_upstream_base_url_missing: set agent.opencode.upstream.base_url")
        if not upstream.api_key_env or not os.environ.get(upstream.api_key_env):
            raise ValueError(f"opencode_upstream_key_missing: set the environment variable {upstream.api_key_env}")
        key = os.environ[upstream.api_key_env]
        headers = {"Authorization": "Bearer " + key} if upstream.auth_scheme == "bearer" else {"x-api-key": key}
        return upstream.base_url.rstrip("/"), upstream_auth_headers_copy(headers, self.wire.auth_header_names)

    def _project_request(self, request):
        """Relay ``request_projection``: the wire's egress scrub (design §4.4) for this session's tools."""
        strip_names = self._deferred_tool_names | getattr(self, "_denied_native_tools", frozenset())
        if strip_names and isinstance(request, dict) and isinstance(request.get("tools"), list):
            request = {
                **request,
                "tools": [
                    tool
                    for tool in request["tools"]
                    if not (isinstance(tool, dict) and tool.get("name") in strip_names)
                ],
            }
        return self.wire.project_request(request, self.presentation, registered_tools=self._tool_names,
                                         tool_name_prefix=TOOL_NAME_PREFIX, on_unknown_part=self._egress_unknown_part,
                                         cache=getattr(self, "_egress_cache", None))

    def _note_request_tools(self, request, identity):
        """Relay ``request_context`` (relay handler thread, before the egress projection, which never changes tool
        names): while the native tool list is unknown, keep the tool names of each tools-declaring call by logical
        call id; ``_model_event`` adopts them once the relay classified that call as the main thread's main chain.
        Returns ``request`` unchanged."""
        if self._native_tools is None:
            names = _request_tool_names(request)
            call = identity.get("logical_call_id")
            if names and call is not None:
                self._call_tools[call] = names
        return request

    def _model_event(self, event):
        """``McpSessionMixin._model_event`` plus the native tool list of provenance (design 11 §1.3): the tool names
        of the first call the relay attributed to the main session with ``call_purpose: "main"`` (what the model
        really got; OpenCode 2.x has no tool-listing route). The manifest is written at start, before any model call,
        so the list is also recorded once as ``agent.native_tools`` (``enabled``: native tools, as
        ``provenance()["native_tools"]["enabled"]``; ``registry``: the registry tools of that request without the
        ``ni_rua_`` prefix); observer data, never model context."""
        McpSessionMixin._model_event(self, event)
        call = event.get("logical_call_id")
        if call is None or call not in self._call_tools:
            return
        names = None
        if (self._native_tools is None and event.get("call_purpose") == "main"
                and event.get("native_thread_id") == self.thread_id):
            names = self._call_tools[call]
            self._native_tools = opencode_config.native_tool_ids(names, tool_name_prefix=TOOL_NAME_PREFIX)
        if self._native_tools is not None:
            self._call_tools.clear()
        elif event.get("kind") in ("model_call_finished", "model_call_cancelled") or event.get("call_purpose"):
            self._call_tools.pop(call, None)
        if names is not None:
            registry = sorted({name[len(TOOL_NAME_PREFIX):] for name in names if name.startswith(TOOL_NAME_PREFIX)})
            try:
                self.store.event("agent.native_tools", dict(enabled=list(self._native_tools), registry=registry,
                                                            source="first_main_request"), source="agent_backend")
            except Exception:  # noqa: BLE001 - observer record only
                pass

    def _egress_unknown_part(self, type_name):
        """A content block of a type the egress walker does not know (still scrubbed generically): one
        ``model.egress_unknown_part`` event per type, so a client change is visible."""
        known = self.__dict__.setdefault("_unknown_parts", set())
        if type_name in known:
            return
        known.add(type_name)
        try:
            self.store.event("model.egress_unknown_part", dict(type=type_name), source="private_model_transport")
        except Exception:  # noqa: BLE001 - observer record only
            pass

    def _await_health(self):
        """Poll ``GET /api/info`` every ``health_poll_s`` until it answers 200 (``startup_timeout_s`` cap; each poll's
        socket timeout is ``health_poll_request_timeout_s``, capped by the remaining budget; HTTP 503 = still
        starting); require the pinned version. A server that exits first -> ``opencode_exited_during_startup:<code>``."""
        deadline = time.monotonic() + self.startup_timeout_s
        while True:
            code = self.process.poll()
            if code is not None:
                raise RuntimeError(f"opencode_exited_during_startup:{code}")
            remaining = deadline - time.monotonic()
            poll_timeout = max(0.001, min(self.health_poll_request_timeout_s, remaining))
            try:
                info = self.client.health(timeout=poll_timeout)
            except OSError:
                info = None
            except OpenCodeHttpError as error:
                raise RuntimeError(f"opencode_health_http_{error.status}") from error
            if info is not None:
                opencode_config.check_health_version(info)
                return
            if time.monotonic() >= deadline:
                raise RuntimeError("opencode_startup_timeout")
            time.sleep(self.health_poll_s)

    def _await_mcp_connected(self):
        """Poll ``GET /api/mcp`` every ``mcp_poll_s`` until the ``ni_rua`` server is ``connected`` (right after start
        it is missing, then ``pending``), at most ``mcp_connect_timeout_s``; ``failed`` / ``needs_auth`` /
        ``disabled`` -> ``RuntimeError("opencode_mcp_<status>")``, the time cap -> ``opencode_mcp_connect_timeout``."""
        deadline = time.monotonic() + self.mcp_connect_timeout_s
        while True:
            code = self.process.poll()
            if code is not None:
                raise RuntimeError(f"opencode_exited_during_startup:{code}")
            try:
                status = _mapping(self.client.mcp_status().get(MCP_SERVER_NAME)).get("status")
            except OpenCodeHttpError as error:
                raise RuntimeError(f"opencode_mcp_status_http_{error.status}") from error
            except OSError:
                status = None
            if status == "connected":
                return
            if status in ("failed", "needs_auth", "disabled"):
                raise RuntimeError(f"opencode_mcp_{status}")
            if time.monotonic() >= deadline:
                raise RuntimeError("opencode_mcp_connect_timeout")
            time.sleep(self.mcp_poll_s)

    # ---- models and effort -----------------------------------------------------------------------------------
    @property
    def effort_choices(self):
        """``model.effort`` values this session accepts: the effort variants the generated config declares on every
        model for the configured wire (``config.EFFORT_CHOICES``), () for a wire without them (start refuses it)."""
        return opencode_config.EFFORT_CHOICES.get(self.wire_api, ())

    def _variant(self):
        """Model-ref ``variant`` of the current effort (a declared variant id), or None when it declares none."""
        return self.effort if self.effort in self.effort_choices else None

    def _model_spec(self, model):
        """``config.model_spec`` of ``model``: the explicit ``agent.opencode`` limits (exactly, for every model),
        else the provider listing's entry (limits, image input), else the fallbacks. An id OpenCode cannot carry in
        a model ref (whitespace, ``#``) raises ``ValueError("invalid_model")``."""
        try:
            return opencode_config.model_spec(model, context_limit_tokens=self.oc_config.context_limit_tokens,
                                              output_limit_tokens=self.oc_config.output_limit_tokens,
                                              catalogue_model=self._catalogue_entry(model))
        except opencode_config.OpenCodeConfigError as error:
            if str(error) == "opencode_model_id_invalid":
                raise ValueError("invalid_model") from None
            raise

    def set_model(self, model):
        """Select ``model`` (any provider model id) for the NEXT host turn (design 11 §2.4).

        Once the server runs, a model the run has not declared is declared first: limits and image input
        (``_model_spec``), entry added to the config file (``config.add_model`` + atomic ``write_config``; OpenCode
        hot-reloads it), then ``GET /api/model`` must list it with the declaration within ``model_declare_timeout_s``,
        else once more after ``POST /api/location/reload``; otherwise the file is restored, ``agent.model_declare_failed``
        is recorded and ``ValueError("opencode_model_declare_failed:<model>")`` raised (the selection is unchanged).
        Success records ``agent.model_declared``. The native session's model is switched at the next ``run_turn``
        (never mid-turn). Blocking (HTTP); call it outside the event loop. Before start (and after a disconnect) only
        the selection changes."""
        if not isinstance(model, str) or not model:
            raise ValueError("invalid_model")
        with self._model_lock:
            with self.lock:
                live = (self._connected() and self.transcript is not None and self._config_doc is not None
                        and not self.closing and self.state not in UNCONNECTED_STATES
                        and self.state not in (STATE_NOT_STARTED, STATE_REPLAY))
                declared = any(spec.id == model for spec in self._models)
            if live and not declared:
                self._declare_model(model)
            elif not declared:
                self._model_spec(model)                  # validates the id for OpenCode's model ref
            with self.lock:
                self.model = model

    def _declare_model(self, model):
        """Add ``model`` to the running server's declaration (see ``set_model``). Call with ``_model_lock`` held."""
        spec = self._model_spec(model)
        before = self._config_doc
        document = opencode_config.add_model(before, spec, wire_api=self.wire_api,
                                             thinking_display=self.oc_config.thinking_display,
                                             verbosity=self.config.model.verbosity)
        path = self.launch.config_path
        opencode_config.write_config(path, document)
        method, problems = "hot_reload", self._await_served(document)
        if problems:
            method = "reload"
            try:
                self.client.reload(timeout_s=self.request_timeout_s)
            except Exception as error:  # noqa: BLE001 - the listing below decides
                self._handler_error("model_reload", error)
            problems = self._await_served(document)
        if problems:
            try:
                opencode_config.write_config(path, before)
            except Exception as error:  # noqa: BLE001
                self._handler_error("model_declare_restore", error)
            self.transcript.event("agent.model_declare_failed", dict(model=model, problems=problems[:20]))
            raise ValueError(f"opencode_model_declare_failed:{model}")
        with self.lock:
            self._config_doc = document
            self._models.append(spec)
        self.transcript.event("agent.model_declared", dict(spec.record(), method=method))

    def _await_served(self, document):
        """Poll ``GET /api/model`` until it serves exactly ``document``'s declaration (``config.check_served_models``)
        or ``model_declare_timeout_s`` passed; returns ``[]`` when it does, else the last problem codes."""
        deadline = time.monotonic() + self.model_declare_timeout_s
        while True:
            try:
                opencode_config.check_served_models(self.client.models(), document)
                return []
            except opencode_config.OpenCodeConfigError as error:
                problems = list(getattr(error, "problems", None) or [str(error)])
            except Exception as error:  # noqa: BLE001 - unreadable listing: try again until the deadline
                problems = [f"models_unreadable:{type(error).__name__}"]
            if time.monotonic() >= deadline:
                return problems
            time.sleep(self.model_declare_poll_s)

    def set_effort(self, effort):
        """Select the effort for the NEXT host turn: the variant of the session's model ref, switched with the model
        at the next ``run_turn`` (``capabilities.live_effort``). Validated against ``effort_choices``."""
        AgentSession.set_effort(self, effort)

    def _apply_pending_model(self, turn_id):
        """At the start of a host turn: switch the native session to the selected ``(model, variant)`` when it differs
        (``POST /api/session/{id}/model``). A 4xx keeps the native model (selection reverted,
        ``agent.model_change_rejected``) and the turn continues on it; no answer is unknown -> ``disconnected`` +
        ``AgentTransportUnknown`` (the turn is not submitted). Success records ``agent.model_changed``."""
        with self.lock:
            wanted, native = (self.model, self._variant()), self._native_ref
        if wanted == native:
            return
        self.store.raw_agent("sent", dict(command="switch_model", thread_id=self.thread_id, turn_id=turn_id,
                                          model=wanted[0], variant=wanted[1]), channel="control")
        try:
            self.client.switch_model(self.thread_id, model_ref(opencode_config.RELAY_PROVIDER_ID, *wanted))
        except Exception as error:  # noqa: BLE001
            if is_definite_rejection(error):
                with self.lock:
                    if native is not None and native[0]:
                        self.model = self.active_turn_model = native[0]
                    if native is not None and native[1] in self.effort_choices:
                        self.effort = native[1]
                self.transcript.event("agent.model_change_rejected", dict(
                    model=wanted[0], variant=wanted[1], kept=native[0] if native else None, code=error.status),
                    turn_id=turn_id)
                return
            self._disconnected(error)
            raise AgentTransportUnknown(f"opencode_switch_model_unknown:{_error_code(error)}") from error
        with self.lock:
            self._native_ref = wanted
            self.active_turn_model = wanted[0]
        self.transcript.event("agent.model_changed", dict(model=wanted[0], variant=wanted[1],
                                                          previous=native[0] if native else None,
                                                          previous_variant=native[1] if native else None),
                              turn_id=turn_id)

    # ---- threads ---------------------------------------------------------------------------------------------
    def _start_threads(self):
        for name, target in (("opencode-events", self._events_loop), ("opencode-reconcile", self._reconcile_loop),
                             ("opencode-watch", self._watch_loop), ("opencode-tree", self._tree_loop)):
            thread = threading.Thread(target=target, name=name, daemon=True)
            self._threads.append(thread)
            thread.start()

    def _handler_error(self, where, error):
        try:
            if self.transcript is not None:
                self.transcript.event("agent.native_handler_error", dict(handler=where, error=type(error).__name__,
                                                                         detail=str(error)[:200]))
        except Exception:  # noqa: BLE001
            pass

    def _watch_loop(self):
        process = self.process
        try:
            code = process.wait()
        except Exception:  # noqa: BLE001
            return
        if self.closing or self._stop.is_set():
            return
        try:
            with self._projector_lock:
                for projector in self._projectors():
                    projector.abandon_open_items()
        except Exception as error:  # noqa: BLE001
            self._handler_error("abandon_open_items", error)
        self._disconnected(ProcessLookupError(f"opencode_exited:{code}"))
        self._halt_threads()

    def _halt_threads(self):
        self._stop.set()
        self._tree_wake.set()
        with self._reconcile_cond:
            self._reconcile_cond.notify_all()

    def _refresh_tree(self):
        """Add the server's current descendants to ``_known_tree`` and drop members that no longer exist (one process
        table read). No-op once the server exited (its detached children are no longer reachable by ppid)."""
        process = self.process
        if process is None or process.poll() is not None:
            return
        try:
            table = process_table()
        except Exception:  # noqa: BLE001 - a failed read only loses one refresh
            return
        present = {(row.pid, row.pgid) for row in table}
        known = {pid: row for pid, row in self._known_tree.items() if (row.pid, row.pgid) in present}
        for row in build_snapshot(process.pid, table).members:
            known[row.pid] = row
        self._known_tree = known

    def _tree_loop(self):
        last = 0.0
        while not self._stop.is_set():
            pause = 0.25 - (time.monotonic() - last)     # at most four table reads per second when woken often
            if pause > 0 and self._stop.wait(pause):
                return
            self._tree_wake.clear()
            self._refresh_tree()
            last = time.monotonic()
            self._tree_wake.wait(self.tree_refresh_s)

    def _projectors(self):
        with self.lock:
            children = [entry["projector"] for entry in self._children.values() if entry["projector"] is not None]
        return ([self.projector] if self.projector is not None else []) + children

    # ---- events (hints only) ---------------------------------------------------------------------------------
    def _events_loop(self):
        try:
            for frame in self.client.event_stream(self._stop, on_gap=self._event_gap):
                try:
                    self._on_event(frame)
                except Exception as error:  # noqa: BLE001 - never kill the reader
                    self._handler_error("event", error)
        except Exception as error:  # noqa: BLE001 - the reader ends; reconcile and the watchdog continue
            if not self._stop.is_set():
                self._handler_error("event_stream", error)

    def _event_gap(self, error):
        """``/api/event`` dropped or a reconnect failed (no replay, so events are being lost): reconcile from the API.

        The first failure of an outage records ``agent.event_stream_gap``; later failed reconnects (every <= 2 s) are
        not recorded again but still request a reconcile while a turn runs; the reconnect requests one more
        (``server.connected``)."""
        if self._stop.is_set() or self.closing:
            return
        first, self._gap_open = not self._gap_open, True
        if first:
            try:
                self.transcript.event("agent.event_stream_gap", dict(error=type(error).__name__))
            except Exception:  # noqa: BLE001
                pass
        if first or self.state == STATE_RUNNING:
            self._request_reconcile("event_stream_gap")

    def _event_target(self, session):
        """Resolve ``session`` to ``(known, projector)`` under ``self.lock``, binding active-turn subagents."""
        with self.lock:
            if session == self.thread_id:
                return True, self.projector
            entry = self._children.get(session)
            if entry is None:
                return False, None
            if self.state == STATE_RUNNING:          # a subagent active in this turn (also a continued one)
                entry["turn_id"] = self.turn_id
            return True, entry["projector"]

    def _on_event(self, frame):
        kind = frame.get("type")
        self.store.raw_agent("received", frame, channel="native")
        data = _mapping(frame.get("data"))
        if kind == "server.connected":
            if self._stream_connections:
                self._request_reconcile("event_stream_reconnected")
            self._stream_connections += 1
            self._gap_open = False
            self._events_ready.set()
            return
        if kind == "permission.asked":
            self._auto_reject(data)
            return
        session = data.get("sessionID")
        if kind == "session.created":
            parent = data.get("parentID")
            if isinstance(session, str) and parent:
                with self.lock:
                    if ((parent == self.thread_id or parent in self._children)
                            and session not in self._children and session != self.thread_id):
                        self._children[session] = dict(parent_session=parent, parent_item_id=None,
                                                       turn_id=self.turn_id, projector=None, items=set())
            return
        if not isinstance(session, str):
            return
        known, projector = self._event_target(session)
        if known:
            self._on_tree_event(session, kind, frame, projector)

    def _on_execution_started(self):
        with self.lock:
            self._activity_seq += 1
            self._native_busy = True
            self._guard.reset()
            self._settle.reset()
            self._adopt_locked("execution_started")  # before any row of the execution is projected

    def _on_tree_event(self, session, kind, frame, projector):
        """One frame of the main session or a known subagent session (``projector`` None: the child's rows wait until
        its ``subagent`` tool row is known; the next sync writes them)."""
        self.last_activity_at = time.monotonic()
        if kind == "session.execution.started":
            self._on_execution_started()
        projected = None
        if projector is not None:
            with self._projector_lock:
                projected = projector.project_event(frame)
        if projected is not None:
            if projected.kind == "subagent":
                self._map_child(projected.data["session_id"], projected.data["item_id"], session)
            elif (projected.kind == "compaction" and projected.data.get("new") and session == self.thread_id
                  and projected.data.get("phase") == "started"):
                self._compaction_signal("compaction_started", key=projected.data.get("message_id"))
        if kind in _SHELL_HINTS:
            self._tree_wake.set()
        if kind in _RECONCILE_EVENTS:
            phase = kind.rsplit(".", 1)[1]
            self._request_reconcile(phase if session == self.thread_id else "subagent_" + phase)

    def _native_active(self):
        """``McpSessionMixin`` hook: a session of the server was last seen active (a close now is not a clean
        boundary)."""
        return self._native_busy

    def _auto_reject(self, request):
        """``permission.asked`` (or a pending request found by a reconcile): answer ``reject`` at once (a headless run
        has nobody to approve; design 11 §1.3), record ``agent.permission_auto_rejected`` and mark the running turn
        (OpenCode then ends the execution without an ``idle``: the turn fails ``opencode_permission_rejected``)."""
        session, request_id = request.get("sessionID"), request.get("id")
        if not isinstance(session, str) or not isinstance(request_id, str):
            return
        with self.lock:
            if request_id in self._rejected_permissions:
                return
            self._rejected_permissions.add(request_id)
            if self.state == STATE_RUNNING:
                self._permission_rejected = True      # set first: the reply ends the execution at once
            turn_id = self.turn_id if self.state == STATE_RUNNING else None
        resources = request.get("resources")
        record = dict(session_id=session, request_id=request_id, action=request.get("action"),
                      resources=[item for item in resources if isinstance(item, str)][:20]
                      if isinstance(resources, list) else None)
        try:
            self.client.permission_reply(session, request_id, "reject")
        except Exception as error:  # noqa: BLE001 - the watchdog's permission poll tries again
            with self.lock:
                self._rejected_permissions.discard(request_id)
            self._handler_error("permission_reply", error)
            return
        try:
            self.transcript.event("agent.permission_auto_rejected", record, turn_id=turn_id)
        except Exception:  # noqa: BLE001
            pass
        self._request_reconcile("permission_rejected")

    def _poll_permissions(self, active, *, tree_sessions=None):
        """Pending permission requests of the active sessions of the tree (``GET .../permission``), auto-rejected."""
        if tree_sessions is None:
            with self.lock:
                tree_sessions = sorted(active_sessions(active) & {self.thread_id, *self._children})
        for session in tree_sessions:
            try:
                pending = self.client.permissions(session)
            except Exception as error:  # noqa: BLE001
                self._handler_error("permission_list", error)
                continue
            for request in pending:
                if isinstance(request, dict):
                    self._auto_reject(dict(request, sessionID=request.get("sessionID") or session))

    # ---- native wake-ups (design R4.2) -----------------------------------------------------------------------
    def _adopt(self, signal):
        """A native execution started while no host turn runs: open an adopted NI-RUA turn
        (``agent.turn_started {origin: "native_wakeup"}``) whose boundary is ``_accounted``. Never adopts while a
        submission is on its way (its own execution), closing, disconnected, closed, not started or in replay
        (recorded once as ``agent.unsolicited_native_turn`` instead). A Store failure leaves it unadopted (recorded;
        its tool calls are then refused as stale and ``interrupt()`` still reaches it). Returns True when adopted."""
        with self.lock:
            return self._adopt_locked(signal)

    def _adopt_locked(self, signal):
        """Adopt a native execution while ``self.lock`` is held; see ``_adopt``."""
        if self.state == STATE_RUNNING or self.transcript is None or self._submitting:
            return False
        if self.closing or self.state in UNCONNECTED_STATES or self.state in (STATE_NOT_STARTED, STATE_REPLAY):
            if not self._unsolicited_noted:
                self._unsolicited_noted = True
                try:
                    self.transcript.event("agent.unsolicited_native_turn", dict(host_state=self.state,
                                                                                signal=signal))
                except Exception:  # noqa: BLE001
                    pass
            return False
        try:
            self._begin_turn(origin="native_wakeup")
        except Exception as error:  # noqa: BLE001
            self._handler_error("adopt", error)
            return False
        self._reset_turn_state()
        self._turn_boundary = self._accounted
        return True

    def _reset_turn_state(self):
        """Per-turn bookkeeping at a turn start (call with the lock held)."""
        self._guard.reset()
        self._settle.reset()
        self._schedule_recheck(None)
        self._permission_rejected = False
        self._host_prompted = False
        self._unsolicited_noted = False
        self._turn_boundary = None

    def _unaccounted(self, messages, active, inbox):
        """Native work the host has not accounted to a turn: an active session, a waiting inbox item, or an
        execution message after ``_accounted`` (a report that was delivered and answered between two reads)."""
        if active_sessions(active) or inbox_ids(inbox):
            return True
        ids = [message["id"] for message in messages]
        start = ids.index(self._accounted) + 1 if self._accounted in ids else 0
        return any(message.get("type") in _EXECUTION_TYPES for message in messages[start:])

    def _joinable(self):
        """An adopted turn without a host prompt is running: ``run_turn`` joins it (design R4.2 rule 3)."""
        return (self.state == STATE_RUNNING and self.turn_origin == "native_wakeup" and not self._host_prompted
                and not self.closing)

    # ---- subagent sessions (design R4.3) ---------------------------------------------------------------------
    def _register_child(self, child_id, parent_session):
        with self.lock:
            if child_id in self._children or child_id == self.thread_id:
                return
            self._children[child_id] = dict(parent_session=parent_session, parent_item_id=None,
                                            turn_id=self.turn_id, projector=None, items=set())

    def _map_child(self, child_id, item_id, owner_session):
        """Link subagent session ``child_id`` to the ``subagent`` tool row ``item_id`` of ``owner_session`` and create
        its projector (rows: ``thread_id`` = child, ``parent_item_id`` = that row). A known child named by another
        tool row (continued with ``sessionID``) gets that row as the parent of its later rows and joins the current
        turn. Returns True when the link is new."""
        with self.lock:
            if child_id == self.thread_id or self.transcript is None or not isinstance(child_id, str):
                return False
            entry = self._children.setdefault(child_id, dict(parent_session=owner_session, parent_item_id=None,
                                                             turn_id=self.turn_id, projector=None, items=set()))
            if item_id in entry["items"]:
                return False
            entry["items"].add(item_id)
            if entry["projector"] is None:
                entry["projector"] = OpenCodeProjector(_SubagentTranscript(self.transcript, child_id, item_id))
            else:
                entry["projector"].transcript.parent_item_id = item_id
                entry["turn_id"] = self.turn_id
            entry.update(parent_session=owner_session, parent_item_id=item_id)
        try:
            self.transcript.event("agent.subagent_session", dict(session_id=child_id, parent_session_id=owner_session),
                                  item_id=item_id)
        except Exception:  # noqa: BLE001
            pass
        return True

    def _link_children(self, messages, owner_session):
        """Link every subagent session named by the ``subagent`` tool items of ``owner_session``'s ``messages``."""
        for link in subagent_links(messages):
            self._map_child(link["session_id"], link["item_id"], owner_session)

    def _fetch_children(self, *, all_turns=False):
        """Snapshots of the linked subagent sessions of the current turn (every linked child with ``all_turns``),
        following nested subagents named in those snapshots. Returns ``[(child id, messages)]``; a child whose read
        fails is recorded and skipped (observer rows only: the verdict reads the main session)."""
        with self.lock:
            queue = [child for child, entry in self._children.items() if entry["projector"] is not None
                     and (all_turns or entry["turn_id"] == self.turn_id)]
        fetched, done = [], set()
        while queue:
            child = queue.pop(0)
            if child in done:
                continue
            done.add(child)
            try:
                snapshot = timeline(self.client.messages(child, page_size=self.message_page_size))
            except Exception as error:  # noqa: BLE001
                self._handler_error("subagent_messages", error)
                continue
            fetched.append((child, snapshot))
            for link in subagent_links(snapshot):
                self._map_child(link["session_id"], link["item_id"], child)
                queue.append(link["session_id"])
        return fetched

    def _child_projectors(self, fetched):
        """Snapshot the active projectors for ``fetched`` subagent sessions under a single lock."""
        if not fetched:
            return {}
        with self.lock:
            return {child: entry["projector"] for child, _ in fetched
                    if (entry := self._children.get(child)) is not None and entry["projector"] is not None}

    def _sync_children(self, fetched, projectors=None):
        """Project fetched subagent snapshots (call with ``_projector_lock`` held)."""
        if projectors is None:
            projectors = self._child_projectors(fetched)
        for child, snapshot in fetched:
            projector = projectors.get(child)
            if projector is not None:
                projector.sync(snapshot)

    # ---- compaction ------------------------------------------------------------------------------------------
    def _compaction_signal(self, signal, key):
        """Native compaction of the main session, once per compaction message (``key``: its id; the
        ``session.compaction.started`` frame and the snapshot name the same id): new ``runtime/image_context_epoch``,
        ``agent.compaction`` and the decision reminder (``on_context_reset``)."""
        with self.lock:
            if key is None or key in self._compaction_keys:
                return
            self._compaction_keys.add(key)
        try:
            self._invalidate_context("native_context_compaction", reason="native_context_compaction", signal=signal,
                                     marker=new_id("compaction"))
            self.transcript.event("agent.compaction", dict(signal=signal, source="native", item_id=key))
        except Exception as error:  # noqa: BLE001
            self._handler_error("compaction", error)

    # ---- reconciliation (authoritative) ----------------------------------------------------------------------
    def _request_reconcile(self, reason):
        with self._reconcile_cond:
            if reason not in self._reconcile_reasons:
                self._reconcile_reasons.append(reason)
            self._reconcile_cond.notify_all()

    def _schedule_recheck(self, delay_s):
        """Set or clear (``delay_s is None``) the next guard/retry recheck deadline under ``_reconcile_cond``."""
        with self._reconcile_cond:
            if delay_s is None:
                self._recheck_at = None
            elif not self._stop.is_set():
                self._recheck_at = time.monotonic() + max(0.0, delay_s)
                self._reconcile_cond.notify_all()

    def _reconcile_loop(self):
        last_run = time.monotonic()
        while not self._stop.is_set():
            with self._reconcile_cond:
                while not self._reconcile_reasons and not self._stop.is_set():
                    now = time.monotonic()
                    if ((self._recheck_at is not None and now >= self._recheck_at)
                            or now - last_run >= self.watchdog_s):
                        break
                    timeout = self.watchdog_s - (now - last_run)
                    if self._recheck_at is not None:
                        timeout = min(timeout, self._recheck_at - now)
                    self._reconcile_cond.wait(max(0.0, timeout))
                if self._stop.is_set():
                    return
                reasons, self._reconcile_reasons = self._reconcile_reasons, []
                if not reasons:
                    now = time.monotonic()
                    if self._recheck_at is not None and now >= self._recheck_at:
                        reasons = ["recheck"]
                    elif now - last_run >= self.watchdog_s:
                        reasons = ["watchdog"]
                if reasons:
                    self._recheck_at = None
            if not reasons:
                continue
            last_run = time.monotonic()
            try:
                self._reconcile(reasons)
                self._reconcile_failures = 0
            except Exception as error:  # noqa: BLE001 - retried by the next request or the watchdog
                self._reconcile_failed(error)

    def _reconcile_failed(self, error):
        if self.closing or self._stop.is_set():
            return
        self._reconcile_failures += 1
        try:
            self.transcript.event("agent.reconcile_failed", dict(error=type(error).__name__, detail=str(error)[:200],
                                                                 consecutive=self._reconcile_failures))
        except Exception:  # noqa: BLE001
            pass
        process = self.process
        if (self._reconcile_failures >= self.reconcile_failure_limit and process is not None
                and process.poll() is not None):
            self._disconnected(ProcessLookupError(f"opencode_exited:{process.returncode}"))
            self._halt_threads()
        elif self.state == STATE_RUNNING:
            self._schedule_recheck(self.incomplete_gap_s)

    def _apply_native_snapshot(self, messages, active, inbox, reasons, *, seq_before=None):
        """Update cached native busyness, adopt unaccounted native work, and read permission targets under one lock."""
        busy = bool(active_sessions(active))
        check_permissions = bool(_PERMISSION_POLL_REASONS & set(reasons))
        tree_sessions = None
        with self.lock:
            if seq_before is not None and self._activity_seq != seq_before:
                busy = self._native_busy or busy
                self._native_busy = busy
            else:
                self._native_busy = busy
            if self.state != STATE_RUNNING and self._unaccounted(messages, active, inbox):
                self._adopt_locked("reconcile")
            running, turn_id = self.state == STATE_RUNNING, self.turn_id
            if running and check_permissions:
                tree_sessions = sorted(active_sessions(active) & {self.thread_id, *self._children})
        return busy, running, turn_id, tree_sessions

    def _emit_snapshot_signals(self, snapshot):
        for key in snapshot.data.get("compactions") or ():
            self._compaction_signal("compaction_message", key=key)
        for switch in snapshot.data.get("model_switches") or ():
            try:
                self.transcript.event("agent.model_switched", switch)
            except Exception as error:  # noqa: BLE001
                self._handler_error("model_switched", error)

    def _evaluate_verdict(self, messages, active, inbox, turn_id, *, seq_before=None):
        """Evaluate ``turn_verdict`` + settle/incomplete guards under ``self.lock``.

        Returns ``(verdict, finished, recheck_delay_s)``.
        """
        with self.lock:
            if self.state != STATE_RUNNING or self.turn_id != turn_id or self.closing:
                return None, False, None
            if self._submitting:            # a prompt/steer is on its way: its message may not exist yet
                self._guard.reset()
                self._settle.reset()
                return None, False, None
            if seq_before is not None and self._activity_seq != seq_before:
                self._guard.reset()
                self._settle.reset()
                return None, False, 0.0
            boundary, pending = self.projector.submissions()
            if boundary is None and not pending and self.turn_origin == "native_wakeup" and not self._host_prompted:
                boundary = self._turn_boundary or (messages[0]["id"] if messages else None)
            interrupt = self._interrupt_requested
            subagents = any(entry["turn_id"] == turn_id for entry in self._children.values())
            verdict = turn_verdict(messages, boundary=boundary, active=active, inbox=inbox, sessions=None,
                                   children=subagents, interrupt_requested=interrupt,
                                   permission_rejected=self._permission_rejected)
            now = time.monotonic()
            verdict = self._settle.observe(verdict, now)
            verdict = self._guard.observe(verdict, now, interrupt_requested=interrupt)
            if not verdict.terminal:
                delay = (self._settle.min_gap_s if verdict.reason == "settling" else self._guard.min_gap_s
                         ) if verdict.idle else None
                return verdict, False, delay
            finished = self._finish_turn(verdict.status, verdict.error)
            if finished:
                self._accounted = messages[-1]["id"] if messages else self._accounted
            return verdict, finished, None

    def _reconcile(self, reasons):
        """One authoritative pass (module doc): active -> inbox -> messages -> adoption -> projection (main +
        subagents) -> verdict -> guards. While no turn runs the snapshot is still projected, and native work the host
        has not accounted (``_unaccounted``) is adopted before its rows are written."""
        with self.lock:
            if self.closing or self.state in STALE_TOOL_STATES or self.thread_id is None:
                return
            running = self.state == STATE_RUNNING
            seq_before = self._activity_seq
        active = self.client.status()
        if not running and reasons == ["watchdog"] and not active_sessions(active):
            with self.lock:
                stale = self._activity_seq != seq_before
            if stale:
                self._schedule_recheck(0.0)
            return                          # the idle watchdog: one cheap read, nothing runs natively
        inbox = self.client.inbox(self.thread_id)
        messages = timeline(self.client.messages(self.thread_id, page_size=self.message_page_size))
        busy, running, turn_id, tree_sessions = self._apply_native_snapshot(
            messages, active, inbox, reasons, seq_before=seq_before)
        if busy:
            self.last_activity_at = time.monotonic()
        if tree_sessions:
            self._poll_permissions(active, tree_sessions=tree_sessions)
        self._link_children(messages, self.thread_id)
        children = self._fetch_children()
        child_projectors = self._child_projectors(children)
        with self._projector_lock:
            snapshot = self.projector.sync(messages)
            self._sync_children(children, child_projectors)
        self._emit_snapshot_signals(snapshot)
        if not running:
            with self.lock:
                stale = self._activity_seq != seq_before
            if stale:
                self._schedule_recheck(0.0)
            return
        verdict, finished, recheck_delay = self._evaluate_verdict(
            messages, active, inbox, turn_id, seq_before=seq_before)
        if recheck_delay is not None:
            self._schedule_recheck(recheck_delay)
        if finished:
            # Written after the terminal, so a Store failure here can never delay it (it is recorded instead).
            try:
                self.transcript.event("agent.turn_verdict", dict(
                    status=verdict.status, reason=verdict.reason, message_id=verdict.message_id,
                    idle_id=verdict.idle_id, signals=list(reasons)), turn_id=turn_id)
            except Exception as error:  # noqa: BLE001
                self._handler_error("turn_verdict", error)

    # ---- turns -----------------------------------------------------------------------------------------------
    def _connected(self):
        process = self.process
        return self.client is not None and process is not None and process.poll() is None

    def run_turn(self, prompt, *, images=()):
        """Submit ``prompt`` (and optional inline ``images``) as a new turn, or into a running adopted turn
        (``capabilities.native_wakeup``); returns the turn id once ``POST .../prompt`` answered 200 (admission), not
        at completion. A new host turn first switches the native model/variant when ``set_model`` / ``set_effort``
        selected another one. 4xx: definite rejection (``AgentRejected``, previous state restored; a joined adopted
        turn keeps running); any other failure: unknown admission (``disconnected`` + ``AgentTransportUnknown``),
        never retried."""
        with self.lock:
            if not self._joinable():
                self._check_turn_admissible()
        # An interrupt tombstoned the relay threads (non-Codex profiles); this turn's model calls must be admitted.
        # They are lifted before the prompt is sent (the execution may call the model before the answer reaches us)
        # and put back if nothing was admitted.
        lifted = self._reopen_relay_threads()
        unknown = False
        try:
            with self.lock:
                joined, previous = self._joinable(), None
                if joined:
                    if not self._connected():
                        raise AgentRejected("not_connected", "opencode_not_connected")
                else:
                    previous = self._begin_turn()
                    self._reset_turn_state()
                    if not self._connected():
                        self._reject_turn(previous, "not_connected", "opencode_not_connected")
                turn_id = self.turn_id
                self._submitting += 1
            try:
                if not joined:
                    try:
                        self._apply_pending_model(turn_id)
                    except AgentTransportUnknown:
                        unknown = True
                        raise
                with self.lock:
                    try:
                        text = self.presentation.present_text(prompt)
                        presented_images = self.presentation.present(list(images)) if images else []
                        if presented_images:
                            raw_uris = []
                            for image in presented_images:
                                url = image.get("url") if isinstance(image, dict) else None
                                if not isinstance(url, str):
                                    raise ValueError("opencode_prompt_file_uri_invalid")
                                raw_uris.append(url)
                            files = tuple(_checked_file_uris(raw_uris))
                        else:
                            files = ()
                        self.store.raw_agent("sent", dict(command="submit_prompt", thread_id=self.thread_id,
                                                          turn_id=turn_id,
                                                          input=[dict(type="text", text=text), *presented_images],
                                                          joined=joined), channel="control")
                        item = self.transcript.user_message(text, turn_id=turn_id)
                    except Exception as error:  # noqa: BLE001 - nothing was sent: a definite rejection
                        code, message = "submit_not_recorded", f"opencode_submit_not_recorded:{type(error).__name__}"
                        if previous is not None:
                            self._reject_turn(previous, code, message)
                        raise AgentRejected(code, message) from error
                try:
                    receipt = (self.client.prompt(self.thread_id, text, files=files) if files
                               else self.client.prompt(self.thread_id, text))
                except Exception as error:  # noqa: BLE001
                    self._withdraw(item)
                    if is_definite_rejection(error):
                        if previous is not None:
                            self._reject_turn(previous, error.status, "opencode_prompt_rejected")
                        raise AgentRejected(error.status, "opencode_prompt_rejected") from error
                    unknown = True                  # _disconnected blocks the relay again
                    self._disconnected(error)
                    raise AgentTransportUnknown(f"opencode_prompt_unknown:{_error_code(error)}") from error
                self._bind(item, receipt["id"])
                with self.lock:
                    self._host_prompted = True
                if joined:
                    self.transcript.event("agent.turn_joined", dict(origin=self.turn_origin), item_id=item)
            finally:
                with self.lock:
                    self._submitting -= 1
        except BaseException:
            if not unknown:
                self._restore_tombstones(lifted)
            raise
        self._request_reconcile("submitted")
        return turn_id

    def _bind(self, item, native_id):
        try:
            with self._projector_lock:
                self.projector.bind_submission(item, native_id)
        except Exception as error:  # noqa: BLE001 - the verdict then waits; the incomplete guard ends the turn
            self._handler_error("bind_submission", error)

    def _withdraw(self, item):
        try:
            with self._projector_lock:
                self.projector.withdraw_submission(item)
        except Exception as error:  # noqa: BLE001
            self._handler_error("withdraw_submission", error)

    def steer(self, text):
        """Add operator ``text`` to the ACTIVE turn (``prompt`` while running: it joins at the next step boundary, or
        starts an execution of the same turn on a main session held open by a background subagent). Returns
        ``turn_id`` once OpenCode answered 200; ``ValueError("no_active_turn_to_steer")`` unless running,
        ``ValueError("turn_interrupt_requested")`` once ``interrupt()`` was called for the turn. A 4xx is a definite
        rejection (``AgentRejected``; the turn continues); any other failure is unknown delivery -> ``disconnected``
        + ``AgentTransportUnknown``. Steers never switch the model."""
        with self.lock:
            if self.state != STATE_RUNNING or not self.turn_id or self.closing or not self._connected():
                raise ValueError("no_active_turn_to_steer")
            if self._interrupt_requested:
                # A steer now would start a new native execution after the interrupt; the turn is ending.
                raise ValueError("turn_interrupt_requested: the turn is being interrupted; send the text as the next "
                                 "prompt")
            turn_id = self.turn_id
            text = self.presentation.present_text(text)
            self.store.raw_agent("sent", dict(command="steer", thread_id=self.thread_id, turn_id=turn_id,
                                              input=[dict(type="text", text=text)]), channel="control")
            item = self.transcript.steer(text, turn_id=turn_id)
            self._submitting += 1
        decremented = False
        try:
            try:
                receipt = self.client.prompt(self.thread_id, text)
            except Exception as error:  # noqa: BLE001
                self._withdraw(item)
                with self.lock:
                    self._submitting -= 1
                    decremented = True
                self._request_reconcile("submission_withdrawn")
                if is_definite_rejection(error):
                    try:
                        self.transcript.event("agent.steer_rejected", dict(code=error.status), item_id=item)
                    except Exception:  # noqa: BLE001
                        pass
                    raise AgentRejected(error.status, "opencode_steer_rejected") from error
                self._disconnected(error)
                raise AgentTransportUnknown(f"opencode_steer_unknown:{_error_code(error)}") from error
            self._bind(item, receipt["id"])
        finally:
            if not decremented:
                with self.lock:
                    self._submitting -= 1
        self._request_reconcile("steered")
        return turn_id

    def interrupt(self):
        """Interrupt the active turn, or stop native work that runs without one; returns before it ends (the
        reconcile worker finishes a host turn as ``interrupted``).

        Acts when a turn runs or, with no turn, when a live ``GET /api/session/active`` shows a session busy (an
        unreadable answer counts as busy), also after an unknown admission (``disconnected``) while the server lives.
        Order (relay block first, as Codex): mark the interrupt (the verdict takes it first); ``cancel_pending`` on
        the relay plus a reopenable tombstone on the main session and every known subagent session; ``POST
        .../interrupt`` on the main session (synchronous; cascades to foreground subagents and running shells); then
        the same on every subagent session still active. The next ``run_turn`` reopens the relay threads. Nothing is
        sent when nothing runs (in-flight relay calls are still cancelled). A failed main interrupt is unknown ->
        ``disconnected`` + ``AgentTransportUnknown``; a failed subagent interrupt is recorded only."""
        with self.lock:
            running = self.state == STATE_RUNNING
            if running:
                self._interrupt_requested = True
            turn_id = self.turn_id
            closing = self.closing
            reachable = not closing and self.thread_id is not None and self._connected()
        # With no turn the live status decides (the cached one may lag the events either way).
        busy = False if running or not reachable else self._live_native_busy(self.interrupt_timeout_s)
        act = running or busy
        self._block_relay("turn_interrupted", block=act)
        if not act or not reachable:
            return
        with self.lock:
            children = sorted(self._children)
        self.store.raw_agent("sent", dict(command="interrupt", thread_id=self.thread_id, turn_id=turn_id,
                                          host_turn=running), channel="control")
        try:
            accepted = self.client.interrupt(self.thread_id, timeout_s=self.interrupt_timeout_s)
        except Exception as error:  # noqa: BLE001
            try:
                self.transcript.event("agent.interrupt_failed", dict(error=type(error).__name__, host_turn=running))
            except Exception:  # noqa: BLE001 - safety stop in _disconnected must still execute
                pass
            self._disconnected(error)
            raise AgentTransportUnknown(f"opencode_interrupt_unknown:{_error_code(error)}") from error
        if not accepted:
            self.transcript.event("agent.interrupt_not_accepted", {})
        self._interrupt_children(children, self.interrupt_timeout_s)
        self._request_reconcile("interrupt")

    def _interrupt_children(self, children, timeout_s):
        """``POST .../interrupt`` on every known subagent session the server still lists as active (best effort;
        failures recorded as ``agent.subagent_interrupt_failed``)."""
        if not children:
            return
        try:
            active = active_sessions(self.client.status())
        except Exception:  # noqa: BLE001 - unknown: try every known child
            active = frozenset(children)
        for child in children:
            if child not in active:
                continue
            try:
                self.client.interrupt(child, timeout_s=timeout_s)
            except Exception as error:  # noqa: BLE001
                try:
                    self.transcript.event("agent.subagent_interrupt_failed", dict(session_id=child,
                                                                                  error=type(error).__name__))
                except Exception:  # noqa: BLE001
                    pass

    def _live_native_busy(self, timeout_s):
        """``GET /api/session/active`` now (``timeout_s`` s): True when any session of the server is active or the
        answer cannot be read (unknown counts as busy); updates the cached native status."""
        try:
            busy = bool(active_sessions(self.client._request("GET", "/api/session/active", timeout=timeout_s)))
        except Exception:  # noqa: BLE001 - unknown: treat as busy (stop it rather than let it run)
            return True
        with self.lock:
            self._native_busy = busy
        return busy

    def _block_relay(self, reason, *, block=True):
        """``model_proxy.cancel_pending(reason)``; with ``block`` also a reopenable tombstone on the main session and
        every known subagent session (an unregistered child id falls back to the main session for attribution, so
        the main tombstone alone already denies a child created later)."""
        proxy = self.model_proxy
        if proxy is None:
            return
        threads = ()
        if block and self.thread_id is not None:
            with self.lock:
                threads = (self.thread_id, *self._children)
        proxy.cancel_pending(reason, threads=threads)

    def _reopen_relay_threads(self):
        """Before a new turn: lift the interrupt tombstones of the main session and of every known subagent session
        (a continued subagent keeps its session id). Returns ``{thread: tombstone reason}`` of the threads lifted
        (for ``_restore_tombstones``)."""
        proxy = self.model_proxy
        if proxy is None or self.thread_id is None:
            return {}
        with self.lock:
            threads = (self.thread_id, *self._children)
        lifted = {}
        for thread in threads:
            reason = proxy.cancelled_threads.get(thread)
            if proxy.reopen_thread(thread):
                lifted[thread] = reason or "turn_interrupted"
        return lifted

    def _reopen_relay_thread(self):
        """``McpSessionMixin`` hook: ``_reopen_relay_threads``; True when one tombstone was lifted."""
        return bool(self._reopen_relay_threads())

    def _restore_tombstones(self, lifted):
        """Put back the tombstones ``_reopen_relay_threads`` lifted for a turn that was not admitted (``{thread:
        reason}``), without touching other threads' calls. Never raises."""
        proxy = self.model_proxy
        if proxy is None or not lifted:
            return
        by_reason = {}
        for thread, reason in lifted.items():
            by_reason.setdefault(reason, []).append(thread)
        for reason, threads in by_reason.items():
            try:
                proxy.block_threads(threads, reason)
            except Exception as error:  # noqa: BLE001
                self._handler_error("restore_tombstones", error)

    def _disconnected(self, error):
        """``McpSessionMixin._disconnected`` after a relay block of the main session and every known subagent
        session: an admitted turn (unknown admission) or native work still running on the server cannot reach the
        model any more. ``interrupt()`` and ``close()`` still interrupt it while the server lives."""
        with self.lock:
            if self.closing or self.state in (STATE_DISCONNECTED, STATE_CLOSED):
                return
        try:
            self._block_relay("agent_disconnected")
        except Exception as failure:  # noqa: BLE001 - the disconnect below must still happen
            self._handler_error("relay_block", failure)
        McpSessionMixin._disconnected(self, error)

    # ---- protocol members ------------------------------------------------------------------------------------
    def owned_processes(self):
        """``[("opencode", the server Popen)]`` from spawn until close confirmed it gone, else ``[]``."""
        process = self.process
        return [("opencode", process)] if process is not None and not self._process_gone else []

    def terminal_attach_spec(self):
        """Read-only native TUI of the live main session (``capabilities.terminal_attach``): the command line of
        ``config.tui_attach_command`` (2.2.7: ``opencode --server <url> --session <id> <dir>``) on this session's
        server URL, main session and actor directory, run in the actor directory with the server's own environment
        (the password travels in ``OPENCODE_PASSWORD``, never on the argv).

        The attached TUI is a client of the running server: it reads the session and follows its events; it creates
        no session (the id exists), sends no prompt and makes no model or title request (verified with the pinned
        binary), and the ``Terminal`` never writes a keystroke to it. It runs on the host, outside the server's
        process tree; it carries the session's process marker, so a TUI the gateway failed to close is still stopped
        by ``close()``.

        None before ``start`` connected (no server or session yet), after ``close`` and when the server is sandboxed
        (``features.sandbox_enabled``: the server is reachable only through its inbound Unix socket, which the TUI
        cannot use, and no host TCP listener is opened for it; the terminal is skipped, ``terminal: null``).
        """
        launch, client, thread_id = self.launch, self.client, self.thread_id
        if (launch is None or client is None or thread_id is None or self.sandbox_enabled
                or self.closing or self.state in UNCONNECTED_STATES):
            return None
        url = f"http://{client.host}:{client.port}"
        command, env = opencode_config.tui_attach_command(launch, server_url=url, session_id=thread_id,
                                                          directory=self.actor_dir)
        return dict(cwd=self.actor_dir, command_prefix=command[:1], command=command, env=env, attach_uri=url,
                    thread_id=thread_id)

    def provenance(self):
        """Manifest ``agent_backend`` block (design §7.3, design 11 §1.10) from ``config.provenance_block`` (declared
        models, effective limits of the selected model, the selected variant's body, native tools of the first main
        request); observer data only."""
        from ..factory import evidence_client
        features = self.config.features
        common = dict(upstream_provider=self.provider, model_configured=self.model, effort=self.effort,
                      isolation=self.isolation, counting=self.counting, context_delivery=self.context_delivery,
                      context_delivery_policy=dict(effective=self.context_delivery,
                                                   configured=self.oc_config.context_delivery,
                                                   subagents=self.oc_config.subagents),
                      capabilities=self.capabilities.to_dict(), subagents=self.oc_config.subagents,
                      evidence_client=(evidence_client(self.config, self.backend_kind)
                                       if features.visual_review_model else None))
        profile = self.upstream_profile_block()
        if self.launch is None:
            block = dict(AgentSession.provenance(self), wire_api=self.wire_api, **common)
            if profile is not None:
                block["upstream_profile"] = profile
            return block
        with self.lock:
            models = list(self._models)
        block = opencode_config.provenance_block(
            self.launch, route_list_sha256=self._route_digest, wire_api=self.wire_api, model=self.model,
            effort=self._variant(), instructions_sha256=self._instructions_sha256, tool_name_prefix=TOOL_NAME_PREFIX,
            subagents=self.oc_config.subagents, enabled_tools=self._native_tools, models=models,
            thinking_display=self.oc_config.thinking_display)
        block.update(common)
        block["prompt_channel"].update(tool_name_prefix=TOOL_NAME_PREFIX, naming_note=naming_note())
        block["provider_compat"] = self.provider_compat_block(self.oc_config.upstream)
        if profile is not None:
            block["upstream_profile"] = profile
        if self.bridge is not None:
            block["tool_transport"]["read_only_tools"] = self.bridge.read_only_tools
        return block

    # ---- close -----------------------------------------------------------------------------------------------
    def _close_sandbox(self, report):
        """Remove the sandbox IPC (inbound socket, relay forwarders) after the process tree is gone. Closing the
        owner socket also makes a surviving in-namespace supervisor end its tree (defence in depth)."""
        sandbox, self.sandbox = self.sandbox, None
        if sandbox is None:
            return
        try:
            sandbox.close()
            if self.ipc is not None:
                self.ipc.rmdir()
        except OSError as error:
            if not isinstance(error, FileNotFoundError):
                report["sandbox_close_error"] = type(error).__name__
        except Exception as error:  # noqa: BLE001
            report["sandbox_close_error"] = type(error).__name__

    def close(self):
        """``McpSessionMixin.close`` after one live read of ``GET /api/session/active`` (so native work the events did
        not report still makes the close unclean; an unreadable answer of a live server counts as busy)."""
        with self.lock:
            check = (not self.closing and self.state not in (STATE_CLOSED, STATE_NOT_STARTED)
                     and self.thread_id is not None and self.client is not None)
        if check and self._connected():
            self._live_native_busy(self.close_interrupt_timeout_s)
        return McpSessionMixin.close(self)

    def _stop_backend(self):
        """Stop the server and every process it started, then verify (design §3.7).

        While the server lives: relay block (main + known subagent sessions) and ``POST .../interrupt`` on the main
        session and every active subagent session whatever the host state (kills running tool shells; a no-op on an
        idle session). Native tool rows left open by a turn that was running (host or native) become failed +
        interrupted. Then the event stream ends and the process tree is stopped: a fresh process table right before
        the kill (members re-validated, so a recycled pid or group is not signalled) plus every descendant and
        process group seen in the periodic refreshes, plus every process carrying this session's marker
        (``isolation.AGENT_MARKER_ENV``, inherited by every tool shell and its detached children; on macOS also
        orphans in the actor dir whose environment the OS withholds), all through the shared
        ``isolation.stop_agent_tree`` -> killed -> re-scanned. Teardown is confirmed only when nothing targeted
        survived, the final re-scan found no marked process and every scan succeeded; this also holds after a server
        crash (``agent_exited_before_teardown`` is recorded as a fact). The daemon threads are joined last."""
        report = {}
        self._halt_threads()
        process, client = self.process, self.client
        alive = process is not None and process.poll() is None
        if alive:
            self._refresh_tree()
        elif process is not None and self.thread_id is not None:
            report["agent_exited_before_teardown"] = True
        at_close = (self.close_report or {}).get("state_at_close")
        native_active = self._native_busy            # refreshed by close() right before the mixin's close
        if alive and client is not None and self.thread_id is not None:
            try:   # relay block before the interrupt (see interrupt): a model call made after it is denied
                self._block_relay("session_closed")
            except Exception as error:  # noqa: BLE001
                report["relay_block"] = "failed:" + type(error).__name__
            try:
                client.interrupt(self.thread_id, timeout_s=self.close_interrupt_timeout_s)
                report["interrupt"] = "completed"
            except Exception as error:  # noqa: BLE001 - the process-tree teardown follows anyway
                report["interrupt"] = "failed:" + type(error).__name__
            if native_active:
                with self.lock:
                    children = sorted(self._children)
                self._interrupt_children(children, self.close_interrupt_timeout_s)
        if at_close in (STATE_RUNNING, STATE_DISCONNECTED) or native_active:
            # The turn ends with the session: native tool rows it left open become failed + interrupted (registry
            # rows keep what the MCP transport wrote).
            try:
                with self._projector_lock:
                    report["abandoned_items"] = sum(len(projector.abandon_open_items())
                                                    for projector in self._projectors())
            except Exception as error:  # noqa: BLE001
                report["abandon_error"] = type(error).__name__
        if client is not None:
            try:
                client.close_streams()
            except Exception:  # noqa: BLE001
                pass
        confirmed = True
        if process is not None or self._instance_id is not None:
            try:
                result = stop_agent_tree(process.pid if process is not None else None,
                                         process is not None and process.poll() is None,
                                         list(self._known_tree.values()), self._instance_id,
                                         cwd_roots=[self.actor_dir])
                report["agent_processes"] = result.to_dict()
                confirmed = result.confirmed
            except Exception as error:  # noqa: BLE001
                report["teardown_error"] = type(error).__name__
                confirmed = False
            try:
                if process is not None:
                    process.poll()   # the teardown may have reaped it: let the Popen record its end
            except Exception:  # noqa: BLE001
                pass
        if not confirmed:
            report["agent_process_exit_unconfirmed"] = True
        elif process is not None:
            self._process_gone = True
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(self.thread_join_timeout_s)
        alive_threads = [thread.name for thread in self._threads if thread.is_alive()]
        if alive_threads:
            report["threads_alive"] = alive_threads
        self._teardown_confirmed = confirmed
        self._close_sandbox(report)
        if self.close_report is not None:
            self.close_report["backend"] = report
