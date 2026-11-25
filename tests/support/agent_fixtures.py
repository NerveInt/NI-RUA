"""Shared cortex, MCP session, tool-schema, and gateway workbench doubles across zones."""
from __future__ import annotations

import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock
from unittest.mock import AsyncMock, Mock
from urllib.parse import urlsplit
import zlib
import struct

from ni_rua.apps.gateway.workbench import Workbench
from ni_rua.core.config import Config
from ni_rua.core.observation_policy import ACTOR_PROFILE
from ni_rua.core.presentation.actor_presentation import ActorPresentation
from ni_rua.core.storage.store import Store
from ni_rua.cortex.agents.base import AgentCapabilities, AgentSession
from ni_rua.cortex.agents.mcp_session import GENERATION_COUNTING, McpSessionMixin, McpToolBridge
from ni_rua.cortex.agents.opencode import config as opencode_cfg
from ni_rua.cortex.agents.opencode.client import OpenCodeClient
from ni_rua.cortex.agents.opencode.session import OpenCodeSession
from ni_rua.cortex.agents.transcript import TranscriptWriter
from ni_rua.cortex.session.wire import AnthropicMessagesWire
from ni_rua.runtime.tools.builtins import create_registry
from ni_rua.runtime.tools.plugins.robotic_view import make_robotic_view_tool
from ni_rua.runtime.tools.plugins.stereo_guide import make_stereo_guide_tool
from support.fake_agent import FakeAgentSession
from support.fake_claude_sdk import FakeClaudeSDK, SDK_AVAILABLE, assistant_message, echo_message, init_message, requires_sdk, result_message, state_message
from support.fake_mcp_client import FakeMcpClient
from support.fake_opencode import DIRECTORY, PASSWORD, FakeOpenCodeServer, model_info
from support.paths import FIXTURES_ROOT

if SDK_AVAILABLE:
    from ni_rua.cortex.agents.claude_code import options as claude_options
    from ni_rua.cortex.agents.claude_code.session import ClaudeCodeSession


class FakeGatewayRobot:
    def __init__(self, run_dir, supported=True):
        self.run_dir = run_dir
        self.calls = []
        self.results = {}
        self.block = None
        if not supported:
            self.operator = None

    def operator(self, command, progress_callback=None, **arguments):
        if progress_callback is not None:
            progress_callback({"phase": "moving", "pose_index": 1, "pose_count": 13, "kept_count": 0})
        if self.block is not None and command in ("go_home", "calibrate"):
            self.block.wait(5)
        self.calls.append((command, arguments))
        result = self.results.get(command)
        if isinstance(result, Exception):
            raise result
        return dict(result or dict(status="completed", stop_confirmed=True), **arguments)

    def close(self):
        return None


CODEX_MANIFEST_KEYS_BEFORE = {
    "schema_version", "run_id", "episode_id", "env_instance_id", "env_generation", "data_origin",
    "actor_information_policy", "motion_review", "vision_compare", "task_planning", "grasp_memory_dir",
    "observation_profile", "clock_policy", "model", "provider", "effort", "price_card", "codex_sdk", "codex_cli",
    "codex_cli_executable", "capabilities", "image_settings", "model_image_policy", "actor_payload_profile",
    "enable_wait", "tracking_guard", "joint_replan", "task", "seed", "layout_id", "max_actions", "actor_wait_budget_s",
    "runtime_protocol", "model_request_limit", "model_request_count_basis", "model_attempt_max_retries",
    "model_request_timeout_s", "model_auto_retries", "model_retry_policy", "no_progress_timeout_s", "startup_timeout_s",
    "progress_clock", "read_only_queries_between_actions", "pi05_enabled", "visual_review_model", "source_sha256",
    "benchmark", "benchmark_terminal_policy", "actor_stall_policy", "actor_progress_policy", "scoring_policy",
    "completion_guidance", "source_ref", "task_notes", "task_notes_source", "task_notes_config", "handoff_policy",
    "public_task_rules", "task_instruction_override", "thread_id", "app_server_pid", "terminal_session_id",
    "terminal_pid", "effective_config", "tool_catalog", "local_scene",
}
CODEX_ONLY_MANIFEST_KEYS = ("codex_sdk", "codex_cli", "codex_cli_executable", "app_server_pid")


def fixture_harness(workbench):
    """Process-free harness double with the real registry, as the existing startup tests build it."""
    harness = SimpleNamespace(
        accepting=True,
        active_action=None,
        states={},
        action_count=0,
        latest={"task_instruction": "Fixture native task.", "observation_id": "initial", "observation_profile": ACTOR_PROFILE},
        capabilities={},
        episode_done=threading.Event(),
        episode_terminal=None,
        native_action_count=0,
        started_at=time.monotonic(),
        last_native_progress_at=time.monotonic(),
        scene=SimpleNamespace(capabilities=lambda: {}),
        start=Mock(),
        action_budget=lambda: dict(native_steps_used=0, native_step_limit=700, native_steps_remaining=700),
    )
    for field in (
        "tool_registry",
        "config",
        "disabled_tools",
        "observation_profile",
        "motion_review",
        "enable_wait",
        "vision_compare",
        "visual_reviewer",
        "action_policy",
        "programs",
    ):
        setattr(harness, field, getattr(workbench.harness, field))
    harness.request_stop = lambda: setattr(harness, "accepting", False)
    return harness


def fake_session(capabilities=AgentCapabilities(), **options):
    created = []

    def create(config, run_dir, store, tool_handler, *, price_card=None, task_planning=None):
        session = FakeAgentSession(
            run_dir, store, tool_handler, capabilities=capabilities, price_card=price_card, **options
        )
        created.append(session)
        return session

    return create, created


def terminal_double():
    return SimpleNamespace(
        start=AsyncMock(),
        close=AsyncMock(),
        session_id="fixture-terminal",
        process=SimpleNamespace(pid=-1),
        status=lambda: dict(mode="read_only"),
    )


def fixture_workbench(root, **options):
    robot = SimpleNamespace(close=Mock(), execute=Mock(), reset=Mock(), evaluate=Mock())
    options.setdefault("visual_review_model", None)
    workbench = Workbench(root, robot=robot, **options)
    done = threading.Event()
    done.set()
    harness = fixture_harness(workbench)
    session = SimpleNamespace(
        thread_id="fixture-thread",
        turn_id="first-turn",
        state="completed",
        turn_done=done,
        model="fixture",
        provider="fixture",
        effort="high",
        price_card=None,
        cli_executable=Path("/fixture/codex"),
        command_prefix=["fixture-sandbox"],
        actor_dir=workbench.root,
        socket_path=workbench.root / "fixture.sock",
        env={},
        process=SimpleNamespace(pid=-1),
        start=Mock(return_value="fixture-thread"),
        run_turn=Mock(),
        close=Mock(),
        model_proxy=SimpleNamespace(limit_notified=False, request_count=1),
    )
    workbench.harness, workbench.session = harness, session
    terminal = SimpleNamespace(
        start=AsyncMock(), close=AsyncMock(), session_id="fixture-terminal", process=SimpleNamespace(pid=-1)
    )
    return workbench, terminal


class Listing:
    """Loopback ``GET <path>`` server answering ``body`` with ``status``; counts requests; ``delay_s`` per answer."""

    def __init__(self, body, *, status=200, path="/v1/models", delay_s=0.0):
        from support.fake_model_servers import is_host_port_probe

        self.body, self.status, self.path, self.delay_s, self.requests = body, status, path, delay_s, []
        self.port_probes = []
        listing = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                # 过滤共享宿主机对新开回环端口的 GET / 探测，不计入业务 requests 亦不触发 delay_s
                if listing.path != "/" and is_host_port_probe(self.command, self.path, b"", self.headers):
                    listing.port_probes.append(
                        dict(path=urlsplit(self.path).path, headers={k.lower(): v for k, v in self.headers.items()})
                    )
                    data = json.dumps({"error": "not found"}).encode()
                    self.send_response(404)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    try:
                        self.wfile.write(data)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                listing.requests.append(
                    dict(path=urlsplit(self.path).path, headers={k.lower(): v for k, v in self.headers.items()})
                )
                if listing.delay_s > 0:
                    time.sleep(listing.delay_s)
                st, b = (
                    (listing.status, listing.body)
                    if urlsplit(self.path).path == listing.path
                    else (404, {"error": "not found"})
                )
                data = json.dumps(b).encode()
                self.send_response(st)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


SESSION = "9e7c96f2-1111-4222-8333-444455556666"
FAKE_CLI = """#!/bin/sh
if [ "$1" = "--version" ]; then echo "2.1.321 (Claude Code)"; exit 0; fi
exec /usr/bin/env
"""


def write_cli(directory, body=FAKE_CLI, name="fake-claude"):
    path = Path(directory) / name
    path.write_text(body)
    path.chmod(0o755)
    return path


PIN = opencode_cfg.OPENCODE_VERSION


def fake_opencode_version_binary(directory: Path, output: str = f"opencode v{PIN}") -> str:
    path = Path(directory) / "opencode"
    path.write_text(f"#!/bin/sh\necho '{output}'\n")
    path.chmod(0o755)
    return str(path)


def spec(model_id="claude-sonnet-4-5", **kw):
    return opencode_cfg.model_spec(model_id, **kw)


NUMERIC_KEYWORDS = (
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
)
NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
NUMBER_WORDS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
)


def numbers_in(text):
    values = []
    for token in NUMBER.findall(text):
        try:
            values.append(float(token))
        except ValueError:
            pass
    return values


def has_number(text, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if any(math.isclose(found, float(value), rel_tol=1e-9, abs_tol=1e-12) for found in numbers_in(text)):
        return True
    return float(value) in range(len(NUMBER_WORDS)) and (
        re.search(r"(?<![A-Za-z])" + NUMBER_WORDS[int(value)] + r"(?![A-Za-z])", text, re.I) is not None
    )


def has_word(text, word):
    return re.search(r"(?<![\w])" + re.escape(word) + r"(?![\w])", text) is not None


def has_value(text, value):
    if isinstance(value, bool) or value is None:
        return has_word(text, json.dumps(value))
    if isinstance(value, (int, float)):
        return has_number(text, value)
    return has_word(text, str(value))


def walk(schema, path, root):
    if not isinstance(schema, dict):
        return
    ref = schema.get("$ref")
    if ref and ref.startswith("#/"):
        target = root
        for part in ref[2:].split("/"):
            target = target[part]
        yield from walk(target, path, root)
    yield path, schema
    for key, child in (schema.get("properties") or {}).items():
        yield from walk(child, path + "/properties/" + key, root)
    for key in ("items", "additionalProperties", "not", "if", "then", "else"):
        if isinstance(schema.get(key), dict):
            yield from walk(schema[key], path + "/" + key, root)
    for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
        for index, child in enumerate(schema.get(key) or []):
            yield from walk(child, path + "/%s/%d" % (key, index), root)
    for key in ("$defs", "definitions"):
        for name, child in (schema.get(key) or {}).items():
            yield from walk(child, path + "/%s/%s" % (key, name), root)


def gaps(tool):
    """Return (path, keyword, detail) for every schema constraint missing from the visible prose."""
    found = []
    schema = tool["inputSchema"]
    for path, node in walk(schema, "", schema):
        text = (node.get("description") or "") + "\n" + (tool.get("description") or "")
        for keyword in NUMERIC_KEYWORDS:
            if keyword in node and not has_number(text, node[keyword]):
                found.append((path, keyword, node[keyword]))
        if "enum" in node:
            for value in node["enum"]:
                if not has_value(text, value):
                    found.append((path, "enum", value))
        if "default" in node:
            value = node["default"]
            scalar = isinstance(value, (bool, int, float)) or value is None or (isinstance(value, str) and value)
            if not re.search(r"defaults?\b|defaulting", text, re.I) or (scalar and not has_value(text, value)):
                found.append((path, "default", value))
    return found


def full_registry():
    """Every tool the gateway can register: built-ins (pose_skill on) and both optional plugins."""
    return create_registry(
        [make_robotic_view_tool(lambda *args: None), make_stereo_guide_tool(lambda *args: None)],
        runtime=Config.from_dict({"features": {"pose_skill": True}}),
    )


def png_url(width, height, shade=0):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    rows = b"".join(b"\x00" + bytes([shade, shade, shade]) * width for _ in range(height))
    data = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(data).decode()


def lean_observation(observation_id, step):
    return {
        "bundle": {
            "observation_id": observation_id,
            "sim_step": step,
            "cameras": [{"camera_id": "cam_head"}, {"camera_id": "cam_left_wrist"}],
        },
        "robot_state": {
            "arms": {
                "left": {
                    "grasp_pose": {"position_m": [0.123456, -0.2, 0.9], "angles_deg": [0.0, 0.0, 90.0]},
                    "gripper": {"position_m": [0.03, 0.03]},
                }
            }
        },
        "current_action_budget": {"native_steps_remaining": 900 - step, "native_steps_limit": 1100},
        "camera_calibration": {"cameras": {"cam_head": {"fx": 288.13253784}}},
        "observation_budget_semantics": "long repeated prose",
    }


def lean_text(value):
    return {"type": "input_text", "text": json.dumps(value)}


def lean_image(width, height, shade=0):
    return {"type": "input_image", "image_url": png_url(width, height, shade), "detail": "high"}


STEPS = [
    ("robot_observe", {}, [lean_text(lean_observation("obs_2", 0)), lean_image(8, 6, 1), lean_image(8, 6, 2)]),
    (
        "robot_action",
        {"mode": "execute", "plan_id": "p1"},
        [
            lean_text(
                {
                    "status": "completed",
                    "plan_id": "p1",
                    "action_id": "a2",
                    "receipt": {
                        "status": "completed",
                        "executed_native_steps": 87,
                        "result_observation_id": "obs_3",
                        "pending_stage_ids": [],
                    },
                    "action_budget": {"native_steps_remaining": 1013, "native_steps_limit": 1100},
                    "trajectory_feedback": {
                        "final_measured_jaws": {"arms": {"left": {"position_m": [0.0288, 0.0291]}}},
                        "measured": {"samples": [[0.1] * 50] * 20},
                    },
                }
            ),
            lean_image(24, 6, 3),
            lean_text(lean_observation("obs_3", 870)),
            lean_image(8, 6, 4),
            lean_image(8, 6, 5),
        ],
    ),
    (
        "program_run",
        {"source": "print(1)"},
        [
            lean_text(
                {
                    "job_id": "job_1",
                    "status": "succeeded",
                    "returncode": 0,
                    "calls": 0,
                    "stdout_text": "dollLeft [-0.4145 -0.1761 0.78]\n",
                    "result_semantics": "x" * 300,
                }
            )
        ],
    ),
]


def body(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


MCP_IDENTITY = dict(run_id="run", episode_id="ep", env_instance_id="env", env_generation=1)
MCP_SCHEMA = {"type": "object", "properties": {"x": {"type": "number", "description": "x in m, default 0"}}}
MCP_TOOLS = [
    {"type": "function", "name": "look", "description": "Read-only view.", "inputSchema": MCP_SCHEMA},
    {"type": "function", "name": "move", "description": "Stateful motion.", "inputSchema": MCP_SCHEMA},
]


def text_result(text, success=True):
    return {"success": success, "contentItems": [{"type": "inputText", "text": text}]}


def wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class StubSession(McpSessionMixin, AgentSession):
    """Minimal MCP-backend session over the mixin: real Store/transcript/presentation, no native process."""

    backend_kind = "stub_backend"
    counting = GENERATION_COUNTING

    def __init__(self, root, store, handler, *, parallel_safe=None):
        self.lock = threading.RLock()
        self.root, self.store, self.private = root, store, root / "private"
        self.private.mkdir(parents=True, exist_ok=True)
        self.presentation = ActorPresentation(store)
        self.tool_handler, self.tool_parallel_safe = handler, parallel_safe
        self.state, self.thread_id, self.turn_id = "idle", "thread-1", None
        self.turn_done, self.turn_result = threading.Event(), None
        self.transcript = TranscriptWriter(store, backend=self.backend_kind, thread_id=self.thread_id)
        self.closing, self.close_report = False, None
        self.model, self.provider, self.price_card = "model-a", "provider-a", None
        self.config, self.wire = Config(), AnthropicMessagesWire(client="claude_code")
        self.model_proxy, self.model_limit_reached, self.bridge = None, threading.Event(), None
        self.last_activity_at = 0.0
        self.resets, self.disconnects, self.order = [], [], []
        self.on_context_reset = self.resets.append
        self.on_disconnect = lambda: self.disconnects.append(self.state)
        self._identity = None
        self._teardown_confirmed = True

    def _stop_backend(self):
        self.order.append("backend")


class RecordingHandler:
    def __init__(self, results=None, error=None, gate=None):
        self.calls, self.results, self.error, self.gate = [], results or {}, error, gate

    def __call__(self, params):
        self.calls.append(params)
        if self.gate is not None:
            self.gate.wait(5)
        if self.error is not None:
            raise self.error
        return self.results.get(params["tool"], text_result("ok " + params["tool"]))


class McpSessionBase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / "run"
        self.store = Store(self.root, MCP_IDENTITY)
        self.addCleanup(self.store.close)

    def session(self, handler=None, **kwargs):
        self.handler = handler or RecordingHandler()
        return StubSession(self.root, self.store, self.handler, **kwargs)

    def bridge(self, session, hints=None):
        bridge = McpToolBridge(session, MCP_TOOLS, parallel_hints=hints).start()
        session.bridge = bridge
        self.addCleanup(bridge.join)
        self.addCleanup(bridge.stop_accepting)
        client = FakeMcpClient(bridge.url, headers=bridge.headers)
        client.handshake()
        return bridge, client

    def events(self, kind):
        return self.store.events_by_type([kind], limit=1000)


class FakeProxy:
    def __init__(self, order):
        self.order, self.request_count, self.attempt_count = order, 0, 0
        self.transport_blocked, self.blocked_error = False, None
        self.transport_recovery_summary = dict(policy=None, rounds=0)

    def cancel_pending(self, reason):
        self.order.append(("cancel_pending", reason))

    def reopen_thread(self, thread_id):
        self.order.append(("reopen_thread", thread_id))
        return True

    def close(self):
        self.order.append("relay_close")


CLAUDE_FIXTURES = FIXTURES_ROOT / "claude_sdk"
CLAUDE_KEY_ENV = "NI_RUA_TEST_CLAUDE_UPSTREAM_KEY"
CLAUDE_CONFIG = {
    "agent": {"backend": "claude_code", "claude_code": {"upstream": {"base_url": "http://127.0.0.1:9", "api_key_env": CLAUDE_KEY_ENV}}},
    "model": {"name": "claude-test-model", "effort": "high"},
    "features": {"self_briefing": False, "visual_review_model": None},
}


def plain_turn(answer="FINAL"):
    def build(sid, text):
        return [
            state_message("running", sid),
            init_message(sid),
            echo_message(text),
            assistant_message(answer, session_id=sid),
            result_message(result=answer, session_id=sid),
            state_message("idle", sid),
        ]

    return build


def opening(sid, text):
    return [state_message("running", sid), init_message(sid), echo_message(text)]


def ending(sid, *, answer="FINAL", **result):
    return [assistant_message(answer, session_id=sid), result_message(session_id=sid, **result), state_message("idle", sid)]


@requires_sdk
class ClaudeSessionBase(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve() / "run"
        self.root.mkdir()
        self.store = Store(self.root, MCP_IDENTITY)
        self.addCleanup(self.store.close)
        environment = mock.patch.dict(os.environ, {CLAUDE_KEY_ENV: "host-secret"})
        environment.start()
        self.addCleanup(environment.stop)
        self.config = Config.from_dict(CLAUDE_CONFIG)
        self.fake = FakeClaudeSDK()
        self.scripts = []
        self.fake.on_query.append(self._on_query)
        self.handler_calls, self.disconnects, self.resets = [], [], []
        self.runtime = claude_options.ClaudeRuntime(
            sdk_version=claude_options.SDK_VERSION,
            cli_version=claude_options.CLI_VERSION,
            executable="/usr/bin/true",
            sha256="0" * 64,
        )

    def _on_query(self, client, text):
        if self.scripts:
            build = self.scripts.pop(0)
            messages = build(self.session.thread_id, text) if callable(build) else build
            if messages:
                client.emit(*messages)

    def script(self, *builds):
        self.scripts.extend(builds)

    def tool_handler(self, params):
        self.handler_calls.append(params)
        gate = getattr(self, "tool_gate", None)
        if gate is not None:
            gate.wait(5)
        return text_result("ok " + params["tool"])

    def make(self, **overrides):
        session = ClaudeCodeSession(
            self.root, self.store, self.tool_handler, config=self.config, client_factory=self.fake.factory, runtime=self.runtime
        )
        session.on_disconnect = lambda: self.disconnects.append(session.state)
        session.on_context_reset = self.resets.append
        session.tool_parallel_hints = {"look": True, "move": False}
        session.tool_parallel_safe = lambda tool, arguments: tool == "look"
        session.submit_timeout_s = session.interrupt_timeout_s = session.set_model_timeout_s = 2.0
        session.result_quiet_guard_s = 0.5
        for key, value in overrides.items():
            setattr(session, key, value)
        self.session = session
        self.addCleanup(session.close)
        return session

    def started(self, resume=None, **overrides):
        session = self.make(**overrides)
        session.start(MCP_TOOLS, "Be careful.", resume)
        return session

    @property
    def client(self):
        return self.fake.client

    def events(self, kind):
        return self.store.events_by_type([kind], limit=1000)

    def raw(self, channel):
        path = self.root / "agent-raw.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        return [record for record in records if record["channel"] == channel]

    def finish(self, session, timeout=5.0):
        self.assertTrue(session.turn_done.wait(timeout), "turn did not end")
        return session.state

    def mcp(self, session):
        client = FakeMcpClient(session.bridge.url, headers=session.bridge.headers)
        client.handshake()
        return client


OPENCODE_KEY_ENV = "NI_RUA_TEST_OPENCODE_UPSTREAM_KEY"
OPENCODE_PROVIDER_400 = {"type": "provider.invalid-request", "message": "provider text that must not be stored", "status": 400}
OPENCODE_MODEL = "claude-test-model"

SERVE_MODES = {
    "sleep": "time.sleep(3600)",
    "exit": "sys.exit(3)",
    "orphans": (
        "subprocess.run(['/bin/sh', '-c', 'sleep 3600 >/dev/null 2>&1 & echo $! > orphan.pid; '"
        " + sys.executable + ' -c \"import os, time; os.setsid(); time.sleep(3600)\" >/dev/null 2>&1 &'"
        " echo $! > detached.pid'], start_new_session=True)\ntime.sleep(3600)"
    ),
    "detached": (
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(3600)'], start_new_session=True,"
        " stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\ntime.sleep(3600)"
    ),
}


def fake_opencode_server_binary(directory, *, version="2.2.7", serve="sleep"):
    path = Path(directory) / f"opencode-{version}-{serve}"
    path.write_text(
        f"#!{sys.executable}\nimport subprocess, sys, time\n"
        f"if sys.argv[1:2] == ['--version']:\n    print('opencode v' + {version!r})\n    sys.exit(0)\n"
        f"{SERVE_MODES[serve]}\n"
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def served_models(document):
    provider = document["providers"]["relay"]
    return [
        model_info(
            model_id,
            package=provider["package"],
            variants=entry["variants"],
            input_modalities=tuple(entry["capabilities"]["input"]),
            context=entry["limit"]["context"],
            output=entry["limit"]["output"],
            base_url=provider["settings"]["baseURL"],
        )
        for model_id, entry in provider["models"].items()
    ]


class ScriptedFake(FakeOpenCodeServer):
    config_path = None
    hot_reload = True
    models_override = None
    _loaded = None

    def _read_config(self):
        self._loaded = json.loads(Path(self.config_path).read_text())

    def _r_models(self, handler, query, body_arg, session_id, rest, outcome):
        with self.lock:
            override = self.models_override
            if override is None and self.config_path is not None:
                if self.hot_reload or self._loaded is None:
                    self._read_config()
                models = served_models(self._loaded)
            else:
                models = list(override if override is not None else self.models)
        return self._json(handler, 200, {"location": {"directory": DIRECTORY}, "data": models})

    def _r_reload(self, handler, query, body_arg, session_id, rest, outcome):
        with self.lock:
            if self.config_path is not None:
                self._read_config()
        return super()._r_reload(handler, query, body_arg, session_id, rest, outcome)


class OpenCodeSessionBase(unittest.TestCase):
    maxDiff = None
    client_timeout = 2.0

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.dir = Path(directory.name).resolve()
        self.root = self.dir / "run"
        self.root.mkdir()
        self.store = Store(self.root, MCP_IDENTITY)
        self.addCleanup(self.store.close)
        environment = mock.patch.dict(os.environ, {OPENCODE_KEY_ENV: "host-secret"})
        environment.start()
        self.addCleanup(environment.stop)
        self.fake = ScriptedFake().start()
        self.addCleanup(self.fake.close)
        self.executable = fake_opencode_server_binary(self.dir)
        self.handler_calls, self.disconnects, self.resets = [], [], []
        self.clients = []

    def config(self, *, executable=None, effort="high", upstream=None, **opencode):
        return Config.from_dict(
            {
                "agent": {
                    "backend": "opencode",
                    "opencode": {
                        "executable": executable or self.executable,
                        "upstream": upstream
                        or {
                            "wire_api": "anthropic_messages",
                            "base_url": "http://127.0.0.1:9",
                            "api_key_env": OPENCODE_KEY_ENV,
                            "auth_scheme": "x-api-key",
                            "provider": "fake",
                        },
                        **opencode,
                    },
                },
                "model": {"name": OPENCODE_MODEL, "effort": effort},
                "features": {"self_briefing": False, "visual_review_model": None},
            }
        )

    def client_factory(self, base_url, password, unix_socket=None):
        self.clients.append((base_url, password))
        self.fake.config_path = self.session.launch.config_path
        return OpenCodeClient(self.fake.url, PASSWORD, timeout_s=self.client_timeout)

    def tool_handler(self, params):
        self.handler_calls.append(params)
        return {"success": True, "contentItems": [{"type": "inputText", "text": "ok " + params["tool"]}]}

    def make(self, config=None, *, launch_options=None, client_factory=None, **overrides):
        options = dict(ripgrep="/bin/echo", managed_config_paths=[], check_version=False)
        options.update(launch_options or {})
        session = OpenCodeSession(
            self.root,
            self.store,
            self.tool_handler,
            config=config or self.config(),
            client_factory=client_factory or self.client_factory,
            launch_options=options,
        )
        session.on_disconnect = lambda: self.disconnects.append(session.state)
        session.on_context_reset = self.resets.append
        session.tool_parallel_hints = {"look": True, "move": False}
        session.tool_parallel_safe = lambda tool, arguments: tool == "look"
        session.watchdog_s = 30.0
        for key, value in overrides.items():
            setattr(session, key, value)
        if "incomplete_gap_s" in overrides:
            session._guard.min_gap_s = overrides["incomplete_gap_s"]
        self.session = session
        self.addCleanup(session.close)
        return session

    def started(self, resume=None, *, launch_options=None, **overrides):
        session = self.make(launch_options=launch_options, **overrides)
        session.start(MCP_TOOLS, "Be careful.", resume)
        return session

    def events(self, kind):
        return self.store.events_by_type([kind], limit=1000)

    def raw(self, channel):
        path = self.root / "agent-raw.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        return [record for record in records if record["channel"] == channel]

    def rows(self, kind=None, thread=None):
        return [
            row
            for row in self.store.states(self.store.TRANSCRIPT_KIND)
            if (kind is None or row["type"] == kind) and (thread is None or row.get("thread_id") == thread)
        ]

    def requests(self, suffix, method="POST"):
        with self.fake.lock:
            return [item for item in self.fake.requests if item["method"] == method and item["path"].endswith(suffix)]

    def prompts(self):
        return self.requests("/prompt")

    def interrupts(self):
        return [item["path"] for item in self.requests("/interrupt")]

    def finish(self, session, timeout=5.0):
        self.assertTrue(session.turn_done.wait(timeout), "turn did not end")
        return session.state

    def running_turn(self, session, text="Do it."):
        turn = session.run_turn(text)
        self.assertTrue(wait_until(lambda: session.thread_id in self.fake.active), "prompt never admitted")
        return turn

    def mcp(self, session):
        client = FakeMcpClient(session.bridge.url, headers=session.bridge.headers)
        client.handshake()
        return client

    def generated(self, session):
        return json.loads(session.launch.config_path.read_text())

    def wake(self, sid, text='<shell id="bg" state="completed">BG-DONE</shell>'):
        self.fake.enqueue(sid, "synthetic", {"text": text, "description": "background shell"})
        if self.fake.start_execution(sid):
            self.fake.deliver(sid)
