"""Per-host end-to-end conformance of the external agent backend (design 20 section 5, "Per-host end-to-end").

A real gateway ``--agent-backend external`` serves the run's MCP endpoint over the SO-101 scene double
(``support.so101_scene``: the real ``so101_real`` backend and adapter over ``FakeArmDriver`` and rendered cameras; no
serial port, no real arm). A real host binary (Claude Code, Codex, OpenCode) connects to it through the plugin the
real installer (``agent-plugin/bin/cli.mjs install <host>``) wrote into an isolated home, in its default permission /
approval mode, and is driven headless by a scripted loopback model (``ConformanceModel`` over
``support.fake_model_servers``). The test compares what the gateway sent (its MCP raw log, HTTP header log, Store and
``summary.json``) with what reached the model (every recorded model request).

Pieces, each usable alone:

* ``TARGETS`` / ``HostTarget``: the five host binaries (path overrides by environment variable), their versions.
* ``HostHome``: a fresh ``mktemp -d`` home with only ``PATH``, ``HOME``, the host's config-dir variables and the
  outside-traffic switches; ``installer`` / ``doctor`` run the real installer (the host classes' ``install``).
* ``ExternalGateway``: the real gateway CLI (``python -m support.host_e2e``: ``support.so101_scene.main`` plus an
  HTTP header log of the MCP endpoint) on a given MCP port.
* ``ConformanceModel``: scripted fake model that replays a ``Script`` by request content and records every request.
* ``ClaudeHost`` / ``CodexHost`` / ``OpenCodeHost``: one headless session (``claude -p`` stream-json, ``codex exec
  --json``, ``opencode serve`` API).
* ``request_view``, ``mcp_exchanges``, ``codex_projection_problems``: pure helpers the checks use.
* Live mode (opt-in, a real provider): ``Upstream`` (the endpoint a host driver is configured with; the scripted
  model's is ``fake_upstream``), ``live_upstream`` (from a provider profile), ``LIVE_TASK`` / ``LIVE_USER_MESSAGE``, the
  drivers' ``live_turn`` / ``live_facts`` with the pure transcript readers ``claude_live_facts``, ``codex_live_facts``,
  ``opencode_live_facts``, and ``files_containing`` / ``scrub_files`` for the key scan of kept evidence.

Gateway entry (``PYTHONPATH=tests:<checkout>``, cwd = checkout): ``python -m support.host_e2e <gateway CLI
arguments>``; ``$NI_RUA_SO101_SCENE`` names the scene JSON, ``$NI_RUA_HOST_E2E_HTTP_LOG`` (optional) the JSONL file that
receives one record per MCP HTTP request: ``{t, method, path, status, headers}`` (all request headers, the values of
``Authorization`` and ``Cookie`` replaced by ``"<present>"``). The header log is test instrumentation of the endpoint
class (a subclass of its ``BaseHTTPRequestHandler``), never production code.
"""
from __future__ import annotations

import base64
import binascii
import importlib.util
import json
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from support import so101_scene
from support.fake_model_servers import FakeAnthropicServer, FakeResponsesServer, Image, Step, ToolCall

REPOSITORY = Path(__file__).resolve().parents[2]
PLUGIN_CLI = REPOSITORY / "agent-plugin" / "bin" / "cli.mjs"
HTTP_LOG_ENV = "NI_RUA_HOST_E2E_HTTP_LOG"
SYSTEM_PATH = ("/usr/bin", "/bin", "/usr/sbin", "/sbin")
FAKE_KEY = "sk-host-e2e-fake"          # placeholder credential of the loopback fake model (never a real key)
# Codex appends this to every plugin MCP tool description (spike 21 section 2, 2b).
CODEX_PLUGIN_SUFFIX = " This tool is part of plugin `NI-RUA`."
# JSON-schema keywords Codex's MCP client drops from tool input schemas (spike 21 section 2, 2c; the same over stdio,
# so it is Codex's client, not the transport). A node whose combinator Codex cannot represent becomes ``{}``.
CODEX_DROPPED_KEYWORDS = frozenset({
    "description", "minimum", "maximum", "minLength", "maxLength", "maxItems", "minProperties", "examples",
    "exclusiveMinimum", "exclusiveMaximum", "pattern", "default", "if", "then", "else", "oneOf", "anyOf", "not",
    "const", "uniqueItems", "prefixItems", "dependentRequired", "$schema"})


# ---------------------------------------------------------------------------------------------------- targets
def _bundled(package, relative):
    """Path of a binary shipped inside an installed Python package of this interpreter, or None."""
    spec = importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations))) / relative


@dataclass(frozen=True)
class HostTarget:
    """One host binary under test.

    ``key``: test id (``claude-pinned``...); ``host``: ``claude`` | ``codex`` | ``opencode``; ``variable``: environment
    variable that overrides ``default`` (absolute path); ``named_version``: the version the design names for it (the
    test reports the version it actually ran); ``pinned``: the pinned binary of the NI-RUA environment (full matrix:
    long call, ``doctor``, wake) rather than the user's own installation (items 1-4, no long call);
    ``long_call_s``: seconds the long ``robot_action`` lasts with ``NI_RUA_HOST_E2E_LONG=1`` (None = no long call:
    the user versions); ``wake``: run the operator background-shell wake (Claude Code and OpenCode, spike 21 W);
    ``which``: executable name looked up on ``PATH`` when neither the override variable nor ``default`` names a file
    (the user installations: one machine's default location is not everyone's)."""
    key: str
    host: str
    variable: str
    default: Path | None
    named_version: str
    pinned: bool
    long_call_s: float | None
    wake: bool
    which: str | None = None

    @property
    def path(self):
        value = os.environ.get(self.variable)
        if value:
            candidate = Path(value).expanduser()
            return candidate if candidate.is_file() and os.access(candidate, os.X_OK) else None
        candidates = (
            self.default if isinstance(self.default, (tuple, list))
            else ((self.default,) if self.default is not None else ())
        )
        for candidate in candidates:
            if candidate is not None and candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
        if self.which:
            # 非 pinned 目标（*-user）回退查 PATH 时跳过当前 Python/NI-RUA 环境内的绑定二进制
            env_prefix = Path(sys.prefix).resolve()
            for directory in (os.environ.get("PATH") or "").split(os.pathsep):
                if not directory:
                    continue
                candidate = Path(directory) / self.which
                if not (candidate.is_file() and os.access(candidate, os.X_OK)):
                    continue
                if not self.pinned:
                    try:
                        candidate.resolve().relative_to(env_prefix)
                        continue
                    except ValueError:
                        pass
                return candidate
        return None

    def version(self):
        path = self.path
        if path is None:
            return None
        try:
            output = subprocess.run([str(path), "--version"], capture_output=True, text=True, timeout=60,
                                    env=dict(PATH=host_path(), HOME=os.environ.get("HOME", "/"), TERM="dumb")).stdout
        except (OSError, subprocess.TimeoutExpired):
            return None
        match = re.search(r"\d+\.\d+\.\d+", output)
        return match.group(0) if match else None


TARGETS = {target.key: target for target in (
    # Claude Code: > 300 s, past its MCP idle timer (spike 21 section 1, item 5: 330 s).
    HostTarget("claude-pinned", "claude", "NI_RUA_HOST_E2E_CLAUDE_PINNED",
               _bundled("claude_agent_sdk", "_bundled/claude"), "2.1.321", True, 330.0, True),
    HostTarget("claude-user", "claude", "NI_RUA_HOST_E2E_CLAUDE_USER", Path.home() / ".local/bin/claude", "2.1.325",
               False, None, False, "claude"),
    HostTarget("codex-pinned", "codex", "NI_RUA_HOST_E2E_CODEX_PINNED", _bundled("codex_cli_bin", "bin/codex"),
               "0.160.0", True, 90.0, False),
    HostTarget("codex-user", "codex", "NI_RUA_HOST_E2E_CODEX_USER",
               (Path("/opt/homebrew/bin/codex"), Path.home() / ".local/bin/codex"), "0.169.0",
               False, None, False, "codex"),
    HostTarget("opencode-pinned", "opencode", "NI_RUA_HOST_E2E_OPENCODE",
               Path.home() / ".local/share/ni-rua/opencode-2.2.7/opencode", "2.2.7", True, 90.0, True),
)}


def free_port():
    """A loopback TCP port nothing listens on (bound once, then released)."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def node_executable():
    return shutil.which("node")


def host_path():
    """``PATH`` of every host and installer process: the directories of ``node`` (the installer; an npm-installed
    host such as Homebrew's ``codex`` is a Node launcher) and ``rg`` (OpenCode), then the system directories."""
    directories = [str(Path(tool).parent) for tool in (node_executable(), shutil.which("rg")) if tool]
    return os.pathsep.join(dict.fromkeys(directories + list(SYSTEM_PATH)))


def wait_for(predicate, timeout, what, interval=0.2):
    """Poll ``predicate()`` until it returns a truthy value (returned) or ``timeout`` (s) passes (AssertionError)."""
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out after {timeout:.0f} s waiting for {what}")
        time.sleep(interval)


# ---------------------------------------------------------------------------------------------- isolated home
class HostHome:
    """An isolated home for one host: ``root / "home"`` with a ``work`` directory (the session's cwd).

    ``env(**extra)`` is the whole environment of every host and installer process (``env -i`` style): ``PATH``
    (``host_path``), ``HOME``, ``TERM=dumb``, the host's config-dir
    variables (Claude ``CLAUDE_CONFIG_DIR``; Codex ``CODEX_HOME``; OpenCode ``XDG_CONFIG_HOME`` / ``XDG_DATA_HOME`` /
    ``XDG_CACHE_HOME`` / ``XDG_STATE_HOME``) and its outside-traffic switches (Claude
    ``CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC``, ``DISABLE_AUTOUPDATER``, ``DISABLE_TELEMETRY``,
    ``DISABLE_ERROR_REPORTING``; OpenCode ``OPENCODE_DISABLE_AUTOUPDATE``, ``OPENCODE_DISABLE_MODELS_FETCH``). None of
    these touches MCP, permissions or tool output. The user's real ``~/.claude``, ``~/.codex`` and
    ``~/.config/opencode`` are never read or written."""

    def __init__(self, root, host):
        self.host = host
        self.home = Path(root) / "home"
        self.work = self.home / "work"
        self.work.mkdir(parents=True)
        if host == "codex":
            # Codex refuses a CODEX_HOME that does not exist ("failed to resolve CODEX_HOME"), which a fresh home
            # would otherwise be; every Codex user has it once Codex ran.
            (self.home / ".codex").mkdir()

    def env(self, **extra):
        env = dict(PATH=host_path(), HOME=str(self.home), TERM="dumb")
        if self.host == "claude":
            env.update(CLAUDE_CONFIG_DIR=str(self.home / ".claude"), CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
                       DISABLE_AUTOUPDATER="1", DISABLE_TELEMETRY="1", DISABLE_ERROR_REPORTING="1")
        elif self.host == "codex":
            env.update(CODEX_HOME=str(self.home / ".codex"))
        else:
            xdg = self.home / "xdg"
            env.update(XDG_CONFIG_HOME=str(xdg / "config"), XDG_DATA_HOME=str(xdg / "data"),
                       XDG_CACHE_HOME=str(xdg / "cache"), XDG_STATE_HOME=str(xdg / "state"),
                       OPENCODE_DISABLE_AUTOUPDATE="1", OPENCODE_DISABLE_MODELS_FETCH="1")
        env.update(extra)
        return env

    def installer(self, *arguments, env=None, timeout=180):
        """Run ``node agent-plugin/bin/cli.mjs <arguments>`` in this home; returns ``(exit code, output)``."""
        result = subprocess.run([node_executable(), str(PLUGIN_CLI), *arguments], cwd=self.work, env=env or self.env(),
                                capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout + result.stderr

    def doctor(self, binary, port, env=None):
        """``doctor --port <port>`` for this host only (the other two host binaries are named absent)."""
        options = []
        for host in ("claude", "codex", "opencode"):
            options += [f"--{host}-bin", str(binary) if host == self.host else "/nonexistent"]
        return self.installer("doctor", "--port", str(port), *options, env=env)


# ---------------------------------------------------------------------------------------------- gateway
def install_http_log(path):
    """Log every MCP HTTP request of this process to the JSONL file ``path`` (see the module docstring): replaces the
    ``BaseHTTPRequestHandler`` the endpoint's handler class derives from, before any endpoint starts."""
    from ni_rua.cortex.agents import mcp_server
    lock = threading.Lock()
    masked = {"authorization", "cookie"}

    class Logged(mcp_server.BaseHTTPRequestHandler):
        def send_response(self, code, message=None):
            headers = {name: ("<present>" if name.lower() in masked else value)
                       for name, value in (getattr(self, "headers", None) or {}).items()}
            entry = dict(t=time.time(), method=getattr(self, "command", None), path=getattr(self, "path", None),
                         status=code, headers=headers)
            with lock, open(path, "a") as stream:
                stream.write(json.dumps(entry) + "\n")
            super().send_response(code, message)
    mcp_server.BaseHTTPRequestHandler = Logged


def gateway_main():
    """``python -m support.host_e2e <gateway arguments>``: the SO-101 scene gateway with the MCP header log."""
    path = os.environ.get(HTTP_LOG_ENV)
    if path:
        install_http_log(path)
    so101_scene.main()


def _shipped_so101_features():
    """``features`` of the shipped SO-101 gateway layer (``configs/so101_gateway.json``), the real SO-101 actor
    catalogue. Its deployment (the real arm's ``configs/so101.local.json``) is never used."""
    return dict(json.loads((REPOSITORY / "configs" / "so101_gateway.json").read_text())["features"])


class ExternalGateway:
    """The real gateway process ``--agent-backend external`` over the SO-101 scene double.

    ``root``: its directory (scene, config, logs, ``run`` = the run directory); ``mcp_port``: ``agent.external.
    mcp_port`` (the port the installer wrote with ``--port``); ``prompt``: the operator task (``--prompt``; it reaches
    the agent only through ``run_begin``); ``execute_hold_s``: the scene's test-only slow mode (s; 0 = off);
    ``colour``: the cube colour. The features are the shipped SO-101 gateway layer's, with ``vggt_endpoint`` on a
    closed loopback port (every VGGT-backed part reports ``unavailable`` exactly as with the service down), grasp
    memory in ``root`` and no PTY terminal."""

    def __init__(self, root, mcp_port, *, prompt, execute_hold_s=0.0, colour="red"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.mcp_port, self.prompt = mcp_port, prompt
        self.run_dir = self.root / "run"
        self.http_log = self.root / "mcp-http.jsonl"
        self.stdout_path, self.stderr_path = self.root / "gateway.stdout.log", self.root / "gateway.stderr.log"
        deployment, self.scene_path = so101_scene.write_scene(
            self.root / "scene", so101_scene.scene_document(colour, execute_hold_s=execute_hold_s))
        features = _shipped_so101_features()
        features.update(vggt_endpoint=f"http://127.0.0.1:{free_port()}", grasp_memory_dir=str(self.root / "memory"),
                        terminal=False)
        self.config_path = self.root / "ni-rua.json"
        self.config_path.write_text(json.dumps(dict(
            version=1, backend=dict(environment="so101_real", robot_model_id="so101", deployment=str(deployment)),
            features=features, agent=dict(backend="external", external=dict(mcp_port=mcp_port)),
            run=dict(experiment_root=str(self.root / "exp"), category="host_e2e", task="desk_free_task", port=0),
            remote=dict(enabled=False)), indent=2))
        self.process = None
        self.url = None

    def start(self, timeout=180):
        env = dict(os.environ, **{so101_scene.SCENE_ENV: str(self.scene_path), HTTP_LOG_ENV: str(self.http_log)})
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPOSITORY / "tests"), str(REPOSITORY),
                                                          os.environ.get("PYTHONPATH")]))
        command = [sys.executable, "-m", "support.host_e2e", "--config", str(self.config_path), "--agent-backend",
                   "external", "--run-dir", str(self.run_dir), "--port", "0", "--no-remote", "--prompt", self.prompt]
        with self.stdout_path.open("w") as stdout, self.stderr_path.open("w") as stderr:
            self.process = subprocess.Popen(command, cwd=REPOSITORY, env=env, stdout=stdout, stderr=stderr,
                                            start_new_session=True)

        def ready():
            if self.process.poll() is not None:
                raise AssertionError(f"gateway exited ({self.process.returncode}) before serving:\n{self.tail()}")
            match = re.search(r"External agent MCP endpoint (\S+)", self.stdout_path.read_text())
            return match.group(1) if match else None
        self.url = wait_for(ready, timeout, "the gateway's external MCP endpoint")
        if self.url != f"http://127.0.0.1:{self.mcp_port}/mcp":
            raise AssertionError(f"gateway serves {self.url}, not the installer's port {self.mcp_port}")
        return self

    def tail(self, lines=40):
        parts = []
        for path in (self.stdout_path, self.stderr_path):
            try:
                parts.append(f"--- {path.name}\n" + "\n".join(path.read_text().splitlines()[-lines:]))
            except OSError:
                pass
        return "\n".join(parts)

    def stop(self, timeout=240):
        """SIGTERM (the operator's Ctrl-C equivalent) and wait; the gateway finalises ``summary.json``."""
        if self.process is None or self.process.poll() is not None:
            return None if self.process is None else self.process.returncode
        self.process.send_signal(signal.SIGTERM)
        try:
            return self.process.wait(timeout)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(30)
            raise AssertionError(f"gateway did not stop within {timeout} s after SIGTERM:\n{self.tail()}")

    def kill(self):
        if self.process is not None and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(30)

    def mcp_records(self):
        """The MCP raw log (``agent-raw.jsonl``, channel ``mcp``): ``[{received_at, direction, message}]``."""
        path = self.run_dir / "agent-raw.jsonl"
        if not path.is_file():
            return []
        return [record for record in map(json.loads, path.read_text().splitlines()) if record.get("channel") == "mcp"]

    def http_records(self):
        if not self.http_log.is_file():
            return []
        return [json.loads(line) for line in self.http_log.read_text().splitlines() if line.strip()]

    def read_store(self, function):
        from ni_rua.core.storage.store import open_read_store
        with open_read_store(self.run_dir) as store:
            return function(store)

    def summary(self):
        path = self.run_dir / "summary.json"
        return json.loads(path.read_text()) if path.is_file() else None


# ---------------------------------------------------------------------------------------------- MCP raw log
@dataclass
class Exchange:
    """One JSON-RPC request the gateway received and its response: ``request`` / ``response`` messages (response
    None when none was completely written), ``received_at`` / ``sent_at`` (POSIX s, from the raw log)."""
    method: str
    request: dict
    response: dict | None
    received_at: float
    sent_at: float | None

    @property
    def duration_s(self):
        return None if self.sent_at is None else self.sent_at - self.received_at

    @property
    def tool(self):
        return (self.request.get("params") or {}).get("name") if self.method == "tools/call" else None

    @property
    def result(self):
        return (self.response or {}).get("result")


def _posix(stamp):
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()


def mcp_exchanges(records):
    """Pair the raw log's received requests with their sent responses (same JSON-RPC id, first unanswered request
    first); notifications are skipped."""
    exchanges, pending = [], {}
    for record in records:
        message = record.get("message") or {}
        if record.get("direction") == "received" and "method" in message and "id" in message:
            exchange = Exchange(message["method"], message, None, _posix(record["received_at"]), None)
            exchanges.append(exchange)
            pending.setdefault(json.dumps(message["id"]), []).append(exchange)
        elif record.get("direction") == "sent" and "id" in message:
            waiting = pending.get(json.dumps(message["id"])) or []
            if waiting:
                exchange = waiting.pop(0)
                exchange.response, exchange.sent_at = message, _posix(record["received_at"])
    return exchanges


def result_parts(result):
    """``(texts, images)`` of an MCP ``CallToolResult``: text items in order, image items decoded (``Image``)."""
    texts, images = [], []
    for item in (result or {}).get("content") or []:
        if item.get("type") == "text":
            texts.append(item.get("text", ""))
        elif item.get("type") == "image":
            images.append(Image(item.get("mimeType", ""), base64.b64decode(item.get("data", ""))))
    return texts, images


# ---------------------------------------------------------------------------------------------- model requests
@dataclass
class ResultView:
    """A tool result as the model received it: text parts in order, decoded images in order, the error flag
    (Anthropic ``is_error``; None on the Responses wire)."""
    call_id: str
    texts: list
    images: list
    is_error: bool | None


@dataclass
class RequestView:
    """The parts of one model request the conformance checks read (wire-neutral).

    ``tools``: ``{name: tool definition as sent}`` (Responses: function tools; namespace tools are in
    ``namespaces``); ``system``: system / instructions texts; ``user_texts``: every user-role text in order;
    ``calls``: ``[{"id", "name", "namespace", "arguments"}]`` the assistant made earlier in the conversation;
    ``results``: ``{call id: ResultView}``; ``last_user_text``: the texts of the conversation's last user-role entry
    joined (Responses: the input items after the last assistant item)."""
    tools: dict = field(default_factory=dict)
    namespaces: dict = field(default_factory=dict)
    system: list = field(default_factory=list)
    user_texts: list = field(default_factory=list)
    calls: list = field(default_factory=list)
    results: dict = field(default_factory=dict)
    last_user_text: str = ""

    @property
    def has_tools(self):
        return bool(self.tools or self.namespaces)


def _blocks(content):
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, dict):
        return [content]
    return [block for block in content or [] if isinstance(block, dict)]


def _decode(media_type, data):
    try:
        return Image(media_type or "application/octet-stream", base64.b64decode(data or "", validate=True))
    except (binascii.Error, ValueError, TypeError):
        return None


def anthropic_view(body):
    view = RequestView(tools={tool["name"]: tool for tool in body.get("tools") or []
                              if isinstance(tool, dict) and isinstance(tool.get("name"), str)})
    system = body.get("system")
    view.system = [block.get("text", "") for block in _blocks(system) if block.get("type") == "text"]
    for message in body.get("messages") or []:
        role, texts = message.get("role"), []
        for block in _blocks(message.get("content")):
            kind = block.get("type")
            if kind == "text" and role == "user":
                texts.append(block.get("text", ""))
            elif kind == "tool_use":
                view.calls.append(dict(id=block.get("id"), name=block.get("name"), namespace=None,
                                       arguments=block.get("input")))
            elif kind == "tool_result":
                parts, images = [], []
                for part in _blocks(block.get("content")):
                    if part.get("type") == "text":
                        parts.append(part.get("text", ""))
                    elif part.get("type") == "image" and (part.get("source") or {}).get("type") == "base64":
                        image = _decode(part["source"].get("media_type"), part["source"].get("data"))
                        if image is not None:
                            images.append(image)
                view.results[block.get("tool_use_id")] = ResultView(block.get("tool_use_id"), parts, images,
                                                                    bool(block.get("is_error")))
        if role == "user":
            view.user_texts.extend(texts)
            view.last_user_text = "\n".join(texts)
    return view


def responses_view(body):
    view = RequestView()
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "namespace":
            view.namespaces[tool.get("name")] = tool
        elif isinstance(tool.get("name"), str):
            view.tools[tool["name"]] = tool
    if isinstance(body.get("instructions"), str):
        view.system = [body["instructions"]]
    trailing = []
    for item in body.get("input") or []:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "message" or (kind is None and "role" in item):
            texts = [block.get("text", "") for block in _blocks(item.get("content"))
                     if block.get("type") in ("input_text", "text")]
            if item.get("role") == "user":
                view.user_texts.extend(texts)
                trailing.extend(texts)
            elif item.get("role") == "assistant":
                trailing = []
        elif kind == "function_call":
            try:
                arguments = json.loads(item.get("arguments") or "{}")
            except ValueError:
                arguments = item.get("arguments")
            view.calls.append(dict(id=item.get("call_id"), name=item.get("name"), namespace=item.get("namespace"),
                                   arguments=arguments))
            trailing = []
        elif kind == "function_call_output":
            output, parts, images = item.get("output"), [], []
            for part in _blocks(output):
                if part.get("type") in ("input_text", "text"):
                    parts.append(part.get("text", ""))
                elif part.get("type") == "input_image":
                    url = part.get("image_url")
                    url = url.get("url") if isinstance(url, dict) else url
                    if isinstance(url, str) and url.startswith("data:") and ";base64," in url:
                        head, _, data = url.partition(";base64,")
                        image = _decode(head[5:], data)
                        if image is not None:
                            images.append(image)
            view.results[item.get("call_id")] = ResultView(item.get("call_id"), parts, images, None)
    view.last_user_text = "\n".join(trailing)
    return view


def request_view(wire, body):
    """``RequestView`` of one request body on ``wire`` (``"anthropic"`` | ``"responses"``)."""
    return anthropic_view(body or {}) if wire == "anthropic" else responses_view(body or {})


def strings_in(node):
    """Every string anywhere inside a JSON value."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from strings_in(value)
    elif isinstance(node, list):
        for value in node:
            yield from strings_in(value)


# ---------------------------------------------------------------------------------------------- scripted model
@dataclass(frozen=True)
class ScriptCall:
    """One tool call of a script step: ``tool`` is the registry name of a NI-RUA tool (the host adds its own prefix
    or namespace) or, with ``native=True``, the name of one of the host's own tools (``Bash``, ``shell``)."""
    tool: str
    arguments: dict
    native: bool = False


# A step is a list of ScriptCall (one assistant message; several = parallel calls), a str (final text, the turn ends)
# or a callable ``step(view, conversation) -> list | str`` that reacts to earlier results (``Conversation``).
Script = list
CALL_ID = re.compile(r"(?:toolu|call)_e2e_([A-Za-z0-9]+)_(\d+)_(\d+)")
SCRIPT_MARKER = re.compile(r"NI-RUA-E2E-SCRIPT=([A-Za-z0-9]+)")


def script_of(view):
    """The script a request belongs to: the last ``NI-RUA-E2E-SCRIPT=<name>`` marker among its user texts, or None."""
    markers = [match.group(1) for text in view.user_texts for match in SCRIPT_MARKER.finditer(text)]
    return markers[-1] if markers else None


class Conversation:
    """What a reactive step sees: ``view`` (the current ``RequestView``) and ``result(step, k=0)``, the
    ``ResultView`` of call ``k`` of an earlier ``step`` of the same script (None when the model has not received it)."""

    def __init__(self, name, view):
        self.name, self.view = name, view

    def result(self, step, k=0):
        for call in self.view.calls:
            match = CALL_ID.fullmatch(call["id"] or "")
            if match and match.group(1) == self.name and (int(match.group(2)), int(match.group(3))) == (step, k):
                return self.view.results.get(call["id"])
        return None


class ConformanceModel:
    """Scripted loopback model for one host: ``FakeAnthropicServer`` (``/v1/messages``: Claude Code, OpenCode) or
    ``FakeResponsesServer`` (``/v1/responses``: Codex) whose every answer is chosen from the request content.

    ``add(name, script)`` registers a script; ``prompt(name, text)`` is the user prompt that selects it (it ends with
    ``NI-RUA-E2E-SCRIPT=<name>``). A request without tools (a host's side request: titles, probes) gets the text
    ``ok``. Otherwise the last script marker among the user texts names the script, the step is one past the highest
    step whose call ids (``toolu_e2e_<name>_<step>_<k>`` / ``call_e2e_...``, minted here) are in the request, and an
    exhausted script answers ``NI-RUA-E2E-DONE <name>`` (``done[name]`` is set). ``host`` maps a ``ScriptCall`` to the
    host's tool call (``host.tool_call(call, call_id)``). ``served``: ``[{t, script, step, kind}]`` per answer;
    ``requests()``: every recorded generation request."""

    def __init__(self, host):
        self.host, self.wire = host, host.wire
        server_class = FakeAnthropicServer if self.wire == "anthropic" else FakeResponsesServer
        self.server = server_class(default=self._respond)
        if self.wire == "responses":
            self.server.local_routes[("GET", "/v1/models")] = (
                200, {"object": "list", "data": [{"id": "fake-model", "object": "model"}], "models": []})
        self.scripts, self.done, self.served = {}, {}, []
        self.lock = threading.Lock()

    def start(self):
        self.server.start()
        return self

    def close(self):
        self.server.close()

    @property
    def base_url(self):
        return self.server.base_url

    def add(self, name, script):
        if not re.fullmatch(r"[A-Za-z0-9]+", name):
            raise ValueError("script names are alphanumeric")
        self.scripts[name], self.done[name] = list(script), threading.Event()

    @staticmethod
    def prompt(name, text):
        return f"{text}\nNI-RUA-E2E-SCRIPT={name}"

    def requests(self):
        return self.server.generation_requests()

    def views(self):
        return [(request, request_view(self.wire, request.json)) for request in self.requests()]

    def _record(self, **entry):
        with self.lock:
            self.served.append(dict(t=time.time(), **entry))

    def _respond(self, request):
        view = request_view(self.wire, request.json if isinstance(request.json, dict) else {})
        name = script_of(view)
        if not view.has_tools or name not in self.scripts:
            self._record(script=name, step=None, kind="side")
            return Step(text="ok")
        steps = {int(match.group(2)) for call in view.calls
                 if (match := CALL_ID.fullmatch(call["id"] or "")) and match.group(1) == name}
        index = max(steps) + 1 if steps else 0
        script = self.scripts[name]
        value = script[index] if index < len(script) else None
        if callable(value):
            value = value(Conversation(name, view))
        if value is None or isinstance(value, str):
            text = value if isinstance(value, str) else f"NI-RUA-E2E-DONE {name}"
            if value is None:
                self.done[name].set()
            self._record(script=name, step=index, kind="text", text=text)
            return Step(text=text)
        prefix = "toolu" if self.wire == "anthropic" else "call"
        calls = [self.host.tool_call(call, f"{prefix}_e2e_{name}_{index}_{k}") for k, call in enumerate(value)]
        self._record(script=name, step=index, kind="tools", tools=[call.tool for call in value])
        return Step(tool_calls=calls)


# The conformance sequence (design 20 section 5): run_begin -> task_plan -> robot_observe (camera images) -> one long
# robot_action -> two parallel read calls in one assistant message -> run_finish with a summary -> final text.
TASK = "Report the colour of the cube on the table."
PLAN = ("Task understanding: report the colour of the cube on the table. Approach + completion: observe both "
        "cameras, move the jaw once, read the capabilities and the robot information, then finish.")
SUMMARY = ("Host conformance sequence finished: briefing read, plan submitted, both cameras observed, jaw moved to "
           "0.3, capabilities and robot information read. Task success was not judged.")
SEQUENCE = ("run_begin", "task_plan", "robot_observe", "robot_action", "robot_capabilities", "robot_information",
            "run_finish")
OBSERVATION_ID = re.compile(r'"observation_id":"([^"]+)"')


def conformance_script():
    def action(conversation):
        observed = conversation.result(2)
        found = [match.group(1) for text in (observed.texts if observed else []) for match in
                 [OBSERVATION_ID.search(text)] if match]
        if not found:
            return "NI-RUA-E2E-ABORT: the robot_observe result carried no observation_id"
        return [ScriptCall("robot_action", {"mode": "direct", "observation_id": found[0], "timeout_s": 60, "stages": [
            {"id": "jaw", "kind": "gripper", "arm": "right", "goal_normalized": 0.3}]})]
    return [
        [ScriptCall("run_begin", {})],
        [ScriptCall("task_plan", {"plan": PLAN, "ledger": []})],
        [ScriptCall("robot_observe", {})],
        action,
        [ScriptCall("robot_capabilities", {"detail": "summary"}), ScriptCall("robot_information", {})],
        [ScriptCall("run_finish", {"summary": SUMMARY})],
    ]


WAKE_SLEEP_S = 10.0
WAKE_COMMAND = f"sleep {WAKE_SLEEP_S:g}; echo ni-rua-wake-done"


def wake_script(host):
    """One background shell task with the host's own tool, then the turn ends (spike 21 W)."""
    return [[host.background_call(WAKE_COMMAND)]]


# ---------------------------------------------------------------------------------------------- hosts
@dataclass(frozen=True)
class Upstream:
    """The model endpoint one host is configured with.

    ``wire``: ``anthropic_messages`` | ``openai_responses``; ``base_url``: the wire's base URL in its usual convention
    (Anthropic without ``/v1``, as ``ANTHROPIC_BASE_URL``; Responses with ``/v1``, as an OpenAI base URL), no trailing
    ``/``; ``key``: the credential (the placeholder ``FAKE_KEY`` for the scripted model; a real key only in live mode:
    given to the host process alone, never to the installer, never printed); ``model``: the model id the host requests
    (None: the host's own default; only the scripted Claude Code path uses it); ``provider``: the provider entry's id
    (Codex ``model_providers.<id>``, OpenCode ``providers.<id>``); ``label``: its display name; ``secret``: ``key`` is a
    real credential (live mode)."""
    wire: str
    base_url: str
    key: str = field(repr=False)
    model: str | None
    provider: str
    label: str
    secret: bool = False

    def secrets(self):
        """Byte strings that must not stay in kept evidence: the key and its last 20 characters (Claude Code records a
        key's suffix when one is approved); empty for the placeholder."""
        return [self.key.encode(), self.key[-20:].encode()] if self.secret else []


class HostProcess:
    """Base of the three host drivers: ``target`` (``HostTarget``), ``home`` (``HostHome``), ``model``
    (``ConformanceModel``, None in live mode), ``upstream`` (``Upstream``), ``log_dir`` (process output files)."""
    wire = "anthropic"
    wake_marker = None       # a text in the model request the host sends after the background task exited

    def __init__(self, target, home, log_dir):
        self.target, self.home, self.log_dir = target, home, Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.binary = target.path
        self.model = self.upstream = None
        self.processes, self.files = [], []

    def attach(self, model):
        """Drive this host with the scripted loopback ``model`` (``ConformanceModel``)."""
        self.model = model
        self.upstream = self.fake_upstream(model.base_url)

    def use_upstream(self, upstream):
        """Live mode: send the host's model requests to ``upstream`` (``live_upstream``); no scripted model."""
        self.model, self.upstream = None, upstream

    def fake_upstream(self, base_url):
        """The ``Upstream`` of the scripted model served at ``base_url`` (``http://127.0.0.1:<port>``)."""
        raise NotImplementedError

    def _credential(self, credential):
        """Whether an environment / config gets the key: always for the placeholder; for a real key only when the
        process is the host itself (``credential=True``), never the installer or ``doctor``."""
        return credential or not self.upstream.secret

    def _watch(self, process, deadline, max_requests, requests):
        """Wait for ``process`` to exit; ``requests()`` counts the host's model requests so far. Returns
        ``{"end": "exited", "code"}``, or kills the process group and returns ``{"end": "timeout" | "request_cap"}``."""
        while process.poll() is None:
            count = requests()
            if count > max_requests or time.monotonic() >= deadline:
                self.kill()
                return dict(end="request_cap" if count > max_requests else "timeout", requests=count)
            time.sleep(1.0)
        return dict(end="exited", code=process.returncode)

    def _spawn(self, command, name, **kwargs):
        """Start ``command`` in its own process group in the home's ``work`` directory; stdout (unless given) and
        stderr go to ``<log_dir>/<name>.stdout`` / ``.stderr``."""
        if "stdout" not in kwargs:
            kwargs["stdout"] = open(self.log_dir / f"{name}.stdout", "a")
            self.files.append(kwargs["stdout"])
        stderr = open(self.log_dir / f"{name}.stderr", "a")
        self.files.append(stderr)
        process = subprocess.Popen(command, cwd=self.home.work, stderr=stderr, start_new_session=True, **kwargs)
        self.processes.append(process)
        return process

    def kill(self):
        """SIGKILL every process group still running, then close the log files (idempotent)."""
        for process in self.processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(30)
            for stream in (process.stdin, process.stdout):
                if stream is not None and not stream.closed:
                    try:
                        stream.close()
                    except OSError:
                        pass
        for stream in self.files:
            stream.close()
        self.files = []

    def tool_call(self, call, call_id):
        """The host's ``ToolCall`` for a ``ScriptCall`` (NI-RUA tools get the host's prefix or namespace)."""
        raise NotImplementedError



class ClaudeHost(HostProcess):
    """``claude -p --input-format stream-json --output-format stream-json --verbose`` in the default permission mode
    (no permission flag); the model is ``ANTHROPIC_BASE_URL`` = the fake model with a placeholder key, or in live mode
    the provider's Anthropic endpoint and key with ``--model`` and the same model for every Claude Code model role
    (host mode's ``_MODEL_VARS`` of ``claude_code.options``)."""
    wake_marker = "<task-notification>"
    prefix = "mcp__plugin_ni_rua_ni_rua__"

    def fake_upstream(self, base_url):
        return Upstream("anthropic_messages", base_url, FAKE_KEY, None, "fake", "fake")

    def env(self, credential=True):
        upstream = self.upstream
        extra = dict(ANTHROPIC_BASE_URL=upstream.base_url)
        if self._credential(credential):
            extra["ANTHROPIC_API_KEY"] = upstream.key
        if upstream.model:
            from ni_rua.cortex.agents.claude_code.options import _MODEL_VARS    # host mode: one model for every role
            extra.update(dict.fromkeys(_MODEL_VARS, upstream.model))
        return self.home.env(**extra)

    def tool_call(self, call, call_id):
        return ToolCall(call.tool if call.native else self.prefix + call.tool, call.arguments, call_id)

    def background_call(self, command):
        return ScriptCall("Bash", {"command": command, "run_in_background": True,
                                   "description": "Wait in the background"}, native=True)

    def install(self, port):
        return self.home.installer("install", "claude", "--claude-bin", str(self.binary), "--port", str(port),
                                   env=self.env(credential=False))

    def open(self):
        self.lines, self.results, self.condition = [], [], threading.Condition()
        model = ["--model", self.upstream.model] if self.upstream.model else []
        self.process = self._spawn([str(self.binary), "-p", "--input-format", "stream-json", "--output-format",
                                    "stream-json", "--verbose", *model], "claude", stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, env=self.env(), text=True)
        threading.Thread(target=self._read, daemon=True, name="claude-stream").start()
        return self

    def _read(self):
        with open(self.log_dir / "claude.stream.jsonl", "a") as log:
            for line in self.process.stdout:
                log.write(line)
                try:
                    message = json.loads(line)
                except ValueError:
                    continue
                with self.condition:
                    self.lines.append((time.time(), message))
                    if message.get("type") == "result":
                        self.results.append((time.time(), message))
                    self.condition.notify_all()

    def send(self, text):
        self.process.stdin.write(json.dumps({"type": "user", "message": {"role": "user", "content": text},
                                             "parent_tool_use_id": None, "session_id": ""}) + "\n")
        self.process.stdin.flush()

    def wait_results(self, count, timeout):
        deadline = time.monotonic() + timeout
        with self.condition:
            while len(self.results) < count:
                if self.process.poll() is not None or time.monotonic() >= deadline:
                    raise AssertionError(f"claude produced {len(self.results)} of {count} results "
                                         f"(exit {self.process.poll()})")
                self.condition.wait(1)
            return self.results[count - 1]

    def turn(self, text, timeout, done=None):
        """Send one user message and wait for its ``result``; returns that result message (``done`` is unused: the
        result marks the turn end)."""
        count = len(self.results) + 1
        self.send(text)
        return self.wait_results(count, timeout)[1]

    def live_turn(self, text, timeout_s, max_requests):
        """Live mode: send ``text`` and wait for its ``result``; ``{"end": "completed", "result": message}``, or the
        process group is killed after ``timeout_s`` (s) or once more than ``max_requests`` model responses arrived
        (``{"end": "timeout" | "request_cap", "requests"}``), or ``{"end": "exited", "code"}``."""
        count = len(self.results) + 1
        deadline = time.monotonic() + timeout_s
        self.send(text)
        with self.condition:
            while len(self.results) < count:
                requests = claude_live_facts([message for _, message in self.lines])["requests"]
                if self.process.poll() is not None or requests > max_requests or time.monotonic() >= deadline:
                    break
                self.condition.wait(1)
            else:
                return dict(end="completed", result=self.results[count - 1][1])
        if self.process.poll() is not None:
            return dict(end="exited", code=self.process.returncode)
        self.kill()
        return dict(end="request_cap" if requests > max_requests else "timeout", requests=requests)

    def live_facts(self):
        return claude_live_facts([message for _, message in self.lines])

    def close(self):
        """End the session like a user closing stdin: Claude Code finishes and exits (killed after 60 s)."""
        if getattr(self, "process", None) is not None and self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(60)
            except subprocess.TimeoutExpired:
                pass
        self.kill()


class CodexHost(HostProcess):
    """``codex exec --json --skip-git-repo-check`` in the default approval and sandbox mode (no such flag). The model
    provider comes only from ``-c`` overrides on the command line (``config.toml`` holds only what the installer and
    Codex wrote): provider ``ni_rua_e2e`` on the Responses wire at the fake model's ``/v1`` (live mode: provider
    ``ni_rua_live`` at the profile's Responses base URL, the key in ``NI_RUA_E2E_LIVE_KEY`` of the Codex process)."""
    wire = "responses"
    namespace = "mcp__ni_rua"

    def fake_upstream(self, base_url):
        return Upstream("openai_responses", f"{base_url}/v1", FAKE_KEY, "fake-model", "ni_rua_e2e", "fake")

    @property
    def key_variable(self):
        return "NI_RUA_E2E_LIVE_KEY" if self.upstream.secret else "NI_RUA_E2E_FAKE_KEY"

    def env(self, credential=True):
        return self.home.env(**({self.key_variable: self.upstream.key} if self._credential(credential) else {}))

    def tool_call(self, call, call_id):
        return ToolCall(call.tool, call.arguments, call_id, namespace=None if call.native else self.namespace)

    def install(self, port):
        return self.home.installer("install", "codex", "--codex-bin", str(self.binary), "--port", str(port),
                                   env=self.env(credential=False))

    def overrides(self):
        upstream = self.upstream
        provider = f"model_providers.{upstream.provider}"
        return ["-c", f'model_provider="{upstream.provider}"', "-c", f'model="{upstream.model}"', "-c",
                f'{provider}.name="{upstream.label}"', "-c", f'{provider}.base_url="{upstream.base_url}"', "-c",
                f'{provider}.wire_api="responses"', "-c", f'{provider}.env_key="{self.key_variable}"']

    def open(self):
        return self

    def turn(self, text, timeout, done=None):
        """One ``codex exec`` run of ``text``; returns its exit code (the JSON events are in ``codex.stdout``; ``done``
        is unused: the process exit marks the turn end)."""
        process = self._spawn([str(self.binary), "exec", "--json", "--skip-git-repo-check", *self.overrides(), text],
                              "codex", stdin=subprocess.DEVNULL, env=self.env())
        try:
            return process.wait(timeout)
        except subprocess.TimeoutExpired:
            self.kill()
            raise AssertionError(f"codex exec did not finish within {timeout} s")

    def rollout(self):
        """Records of every Codex session rollout file in ``CODEX_HOME/sessions`` (oldest file first)."""
        records = []
        for path in sorted((self.home.home / ".codex" / "sessions").rglob("rollout-*.jsonl")):
            for line in path.read_text(errors="replace").splitlines():
                try:
                    records.append(json.loads(line))
                except ValueError:
                    pass
        return records

    def events(self):
        """The ``codex exec --json`` events written so far (``codex.stdout``)."""
        path = self.log_dir / "codex.stdout"
        events = []
        for line in (path.read_text(errors="replace").splitlines() if path.is_file() else []):
            try:
                events.append(json.loads(line))
            except ValueError:
                pass
        return events

    def live_turn(self, text, timeout_s, max_requests):
        """Live mode: one ``codex exec`` run of ``text`` (``HostProcess._watch``: ``{"end": "exited", "code"}``, or
        killed after ``timeout_s`` s / more than ``max_requests`` model responses)."""
        process = self._spawn([str(self.binary), "exec", "--json", "--skip-git-repo-check", *self.overrides(), text],
                              "codex", stdin=subprocess.DEVNULL, env=self.env())
        return self._watch(process, time.monotonic() + timeout_s, max_requests,
                           lambda: codex_live_facts([], self.rollout())["requests"])

    def live_facts(self):
        return codex_live_facts(self.events(), self.rollout())

    def close(self):
        self.kill()


class OpenCodeHost(HostProcess):
    """``opencode serve`` driven through its ``/api`` (``ni_rua.cortex.agents.opencode.client.OpenCodeClient``), with
    no ``permissions`` in its config (the default). The fake model is an ``anthropic-compatible`` provider ``fake``
    seeded into the global ``opencode.json`` before the installer runs (the installer adds ``plugins`` and
    ``tool_output`` to it). Live mode: the provider ``live`` with the profile's wire (the OpenCode package of that wire,
    ``WIRE_PACKAGES`` of ``ni_rua.cortex.agents.opencode.config``), base URL and key, in a mode-0600 ``opencode.json``
    of the isolated home."""
    wake_marker = 'state="completed"'
    prefix = "ni_rua_"

    def fake_upstream(self, base_url):
        return Upstream("anthropic_messages", base_url, FAKE_KEY, "fake-model", "fake", "fake")

    def env(self):
        self.password = getattr(self, "password", None) or secrets.token_urlsafe(12)
        return self.home.env(OPENCODE_PASSWORD=self.password)

    def tool_call(self, call, call_id):
        return ToolCall(call.tool if call.native else self.prefix + call.tool, call.arguments, call_id)

    def background_call(self, command):
        return ScriptCall("shell", {"command": command, "background": True,
                                    "description": "Wait in the background"}, native=True)

    @property
    def config_path(self):
        return self.home.home / "xdg" / "config" / "opencode" / "opencode.json"

    def install(self, port):
        from ni_rua.cortex.agents.opencode.config import (FALLBACK_CONTEXT_LIMIT_TOKENS, OUTPUT_LIMIT_CEILING_TOKENS,
                                                          WIRE_PACKAGES)
        upstream = self.upstream
        # OpenCode's packages append /messages and /responses to baseURL: the Anthropic base gets its /v1 here.
        base_url = upstream.base_url + ("/v1" if upstream.wire == "anthropic_messages" else "")
        # Live: host mode's limits for a model without catalogue limits (tokens; the provider lists none).
        limit = {"context": 1000000, "output": 32000} if not upstream.secret else {
            "context": FALLBACK_CONTEXT_LIMIT_TOKENS, "output": OUTPUT_LIMIT_CEILING_TOKENS}
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        if upstream.secret:
            self.config_path.touch(mode=0o600)
            self.config_path.chmod(0o600)
        self.config_path.write_text(json.dumps({"model": f"{upstream.provider}/{upstream.model}", "providers": {
            upstream.provider: {
                "name": upstream.label, "package": WIRE_PACKAGES[upstream.wire],
                "settings": {"baseURL": base_url, "apiKey": upstream.key},
                "models": {upstream.model: {"name": upstream.model, "capabilities": {
                    "tools": True, "input": ["text", "image"], "output": ["text"]}, "limit": limit}}}}}, indent=2))
        return self.home.installer("install", "opencode", "--opencode-bin", str(self.binary), "--port", str(port),
                                   env=self.env())

    def open(self, timeout=120):
        from ni_rua.cortex.agents.opencode.client import OpenCodeClient, OpenCodeError
        port = free_port()
        self._spawn([str(self.binary), "serve", "--port", str(port), "--hostname", "127.0.0.1"], "opencode",
                    env=self.env())
        self.client = OpenCodeClient(f"http://127.0.0.1:{port}", self.password, timeout_s=60)

        def healthy():
            try:
                return self.client.health(timeout=2) is not None
            except OpenCodeError:
                return False
        wait_for(healthy, timeout, "opencode serve")

        # Plugins load per location at first use; a reload makes the location load them now.
        def connected():
            status = self.client.mcp_status()
            if (status.get("ni_rua") or {}).get("status") == "connected":
                return True
            if not status:
                self.client.reload(timeout_s=60)
            return False
        wait_for(connected, timeout, "OpenCode's ni_rua MCP server to connect", interval=1.0)
        self.session = self.client.create_session("ni_rua host e2e")
        return self

    def idle(self):
        return self.session not in self.client.status()

    def turn(self, text, timeout, done=None):
        """Prompt the session and wait until ``done`` (a ``threading.Event``, e.g. the script's) is set and the session
        is idle again."""
        self.client.prompt(self.session, text)
        deadline = time.monotonic() + timeout
        if done is not None and not done.wait(timeout):
            raise AssertionError(f"OpenCode did not finish the script within {timeout} s")
        wait_for(self.idle, max(5.0, deadline - time.monotonic()), "the OpenCode session to become idle", 0.5)

    def live_turn(self, text, timeout_s, max_requests):
        """Live mode: prompt the session and wait for the ``idle`` message that ends the run. A permission request (the
        default mode asks for some actions) is answered ``reject``, as no human is there, and recorded in
        ``self.rejected``; in 2.2.7 a reject ends the execution without an ``idle`` message. Returns ``{"end":
        "completed", "outcome"}`` (the idle message's outcome), ``{"end": "stopped"}`` (the session became inactive
        with no idle message, for two polls 1 s apart), or interrupts and kills after ``timeout_s`` s / more than
        ``max_requests`` assistant steps (``{"end": "timeout" | "request_cap", "requests"}``). The session timeline
        is written to ``opencode.messages.json``."""
        self.rejected = []
        before = len(self.client.messages(self.session))
        deadline = time.monotonic() + timeout_s
        self.client.prompt(self.session, text)
        inactive = 0
        try:
            while True:
                self.timeline = self.client.messages(self.session)
                fresh = self.timeline[before:]
                idle = [message for message in fresh if message.get("type") == "idle"]
                if idle:
                    return dict(end="completed", outcome=idle[-1].get("outcome"))
                requests = opencode_live_facts(fresh)["requests"]
                if requests > max_requests or time.monotonic() >= deadline:
                    try:
                        self.client.interrupt(self.session, timeout_s=10)
                    finally:
                        self.kill()
                    return dict(end="request_cap" if requests > max_requests else "timeout", requests=requests)
                for request in self.client.permissions(self.session):
                    self.rejected.append({key: request.get(key) for key in ("action", "resources") if key in request})
                    self.client.permission_reply(self.session, request["id"], "reject")
                inactive = inactive + 1 if requests and self.idle() else 0
                if inactive >= 2:
                    return dict(end="stopped")
                time.sleep(1.0)
        finally:
            (self.log_dir / "opencode.messages.json").write_text(json.dumps(getattr(self, "timeline", []), indent=1))

    def live_facts(self):
        facts = opencode_live_facts(getattr(self, "timeline", []))
        facts["permission_denials"] = list(getattr(self, "rejected", []))
        return facts

    def close(self):
        self.kill()


HOST_CLASSES = {"claude": ClaudeHost, "codex": CodexHost, "opencode": OpenCodeHost}


# ---------------------------------------------------------------------------------------------- live mode (opt-in)
# The operator task of a live session (the gateway's --prompt; the model receives it only through run_begin).
LIVE_TASK = ("Live interface check of the SO-101 arm. Call robot_observe and look at its camera images. Write the line "
             "CUBE_COLOUR: <one English colour word> for the cube on the table (CUBE_COLOUR: unknown if you received "
             "no image; do not guess). Then move the end effector 5 mm straight up (+z in so101_base) with one "
             "robot_action, keeping its orientation and the gripper as they are, timeout_s=60, and confirm the move "
             "from the action receipt (requested and measured end-effector z). Finish with run_finish; its summary "
             "contains the CUBE_COLOUR line and the measured z.")
# The only user message a live host gets: it names neither run_begin nor the task (those come from NI-RUA).
LIVE_USER_MESSAGE = "Operate the robot through the ni_rua MCP tools."
# The answer line LIVE_TASK asks for. The underscore is required, so prose such as "the cube colour from the camera"
# is not read as an answer; the first match counts.
LIVE_COLOUR_LINE = re.compile(r"CUBE_COLOU?R\W+([A-Za-z]+)", re.I)
# The provider wire each host speaks (first = default; OpenCode has a package for both).
LIVE_WIRES = {"claude": ("anthropic_messages",), "codex": ("openai_responses",),
              "opencode": ("anthropic_messages", "openai_responses")}


def live_profile_path(value):
    """Path of a provider profile named by ``value``: a path (contains ``/`` or ends in ``.local.json``) or a profile
    id in the checkout's ``configs/profiles/``."""
    if "/" in value or value.endswith(".local.json"):
        return Path(value).expanduser().resolve()
    return REPOSITORY / "configs" / "profiles" / f"{value}.local.json"


def live_upstream(profile_path, host, model, wire=None):
    """The ``Upstream`` of ``host`` for model id ``model`` through the provider profile at ``profile_path`` on
    ``wire`` (default: the host's first ``LIVE_WIRES`` entry). The key comes from ``profiles.credential``, the only
    reader of a profile key; it stays in the returned object (not in its repr). Raises ``ValueError`` (no key value)
    for a wire the host cannot speak or the profile does not serve."""
    from ni_rua.cortex.agents import profiles
    wire = wire or LIVE_WIRES[host][0]
    if wire not in LIVE_WIRES[host]:
        raise ValueError(f"{host} cannot use the {wire} wire (it speaks {', '.join(LIVE_WIRES[host])})")
    profile = profiles.load_profile(profile_path)
    base_url, headers = profiles.credential(profile, wire)
    key = next(iter(headers.values())).removeprefix("Bearer ")
    provider = "ni_rua_live" if host == "codex" else "live"
    return Upstream(wire, base_url.rstrip("/"), key, model, provider, "live", secret=True)


def _short(value, limit=300):
    text = value if isinstance(value, str) else json.dumps(value)
    return (text.strip().splitlines() or [""])[0][:limit]


def _result_text(content):
    """The text of an MCP / Anthropic tool-result content (string or blocks)."""
    return " ".join(block.get("text", "") for block in _blocks(content) if block.get("type") == "text")


def _call(name, prefix, error=None):
    """One entry of a host's tool-call sequence: ``{"tool": "ni_rua:<registry name>" | "<native tool>", "error"}``."""
    return dict(tool=("ni_rua:" + name[len(prefix):]) if name.startswith(prefix) else name, error=error)


def claude_live_facts(messages):
    """Facts of a live Claude Code session from its stream-json ``messages`` (in order).

    Returns ``{"calls": [{"tool", "error"}] in call order ("ni_rua:<tool>" for the plugin's tools, else the native
    name; "error" = the first line of an ``is_error`` tool result), "requests": distinct assistant message ids (one
    per model response), "served_models": the ``model`` the provider answered with, "configured_model" and
    "permission_mode" (the ``init`` message), "usage" (the ``result`` message's ``usage``, ``modelUsage``,
    ``total_cost_usd``, ``num_turns``; None before it), "texts": the assistant's text blocks, "final_text" (``result``),
    "permission_denials" (tool names), "errors" (error results and an error ``result``)}``."""
    calls, by_id, responses, texts, errors = [], {}, {}, [], []
    init = result = None
    for message in messages:
        kind = message.get("type")
        if kind == "system" and message.get("subtype") == "init":
            init = message
        elif kind == "assistant":
            body = message.get("message") or {}
            if body.get("id"):
                responses.setdefault(body["id"], body.get("model"))
            for block in _blocks(body.get("content")):
                if block.get("type") == "tool_use":
                    by_id[block.get("id")] = entry = _call(block.get("name") or "", ClaudeHost.prefix)
                    calls.append(entry)
                elif block.get("type") == "text" and block.get("text", "").strip():
                    texts.append(block["text"])
        elif kind == "user":
            for block in _blocks((message.get("message") or {}).get("content")):
                if block.get("type") == "tool_result" and block.get("is_error"):
                    entry = by_id.get(block.get("tool_use_id"))
                    text = _short(_result_text(block.get("content")))
                    if entry is not None:
                        entry["error"] = text
                    errors.append(f"{entry['tool'] if entry else '?'}: {text}")
        elif kind == "result":
            result = message
    if result is not None and (result.get("is_error") or result.get("subtype") != "success"):
        errors.append(f"result {result.get('subtype')}: {_short(result.get('result') or '')}")
    usage = None if result is None else {key: result.get(key) for key in (
        "usage", "modelUsage", "total_cost_usd", "num_turns", "duration_api_ms")}
    return dict(calls=calls, requests=len(responses), served_models=sorted({m for m in responses.values() if m}),
                configured_model=(init or {}).get("model"), permission_mode=(init or {}).get("permissionMode"),
                usage=usage, texts=texts, final_text=(result or {}).get("result"),
                permission_denials=[denial.get("tool_name") for denial in (result or {}).get("permission_denials")
                                    or []], errors=errors)


def codex_live_facts(events, rollout):
    """Facts of a live ``codex exec --json`` run from its ``events`` and its session ``rollout`` records.

    Returns ``{"calls": [{"tool", "error"}] in start order (``mcp_tool_call`` items as "ni_rua:<tool>" for the
    ``ni_rua`` server, else "<server>:<tool>"; every other non-message item by its type, e.g. ``command_execution``),
    "requests": distinct ``response_id`` of the rollout's ``token_usage_record`` (one per model response),
    "served_models": None (Codex records no served model id), "configured_model", "approval_policy", "sandbox" (the
    rollout's ``turn_context``), "usage" (the ``turn.completed`` usage; None before it), "texts" (``agent_message``
    items), "final_text" (the last of them), "permission_denials": [], "errors" (``error`` items and events,
    ``turn.failed``, failed tool items)}``."""
    calls, order, texts, errors = {}, [], [], []
    usage = None
    for event in events:
        kind, item = event.get("type"), event.get("item") or {}
        if kind in ("item.started", "item.completed") and item.get("id"):
            item_type = item.get("type")
            if item_type in ("agent_message", "reasoning"):
                if kind == "item.completed" and item_type == "agent_message" and item.get("text", "").strip():
                    texts.append(item["text"])
                continue
            if item_type == "error":
                if kind == "item.completed":
                    errors.append(_short(item.get("message") or ""))
                continue
            if item_type == "mcp_tool_call":
                tool = (("ni_rua:" if item.get("server") == "ni_rua" else f"{item.get('server')}:")
                        + str(item.get("tool")))
            else:
                tool = item_type
            if item["id"] not in calls:
                calls[item["id"]] = dict(tool=tool, error=None)
                order.append(item["id"])
            if kind == "item.completed":
                result, failure = item.get("result") or {}, item.get("error")
                if failure or result.get("isError") or result.get("is_error") or item.get("status") == "failed":
                    message = failure.get("message") if isinstance(failure, dict) else failure
                    text = _short(message or _result_text(result.get("content")) or item.get("status") or "error")
                    calls[item["id"]]["error"] = text
                    errors.append(f"{tool}: {text}")
        elif kind == "turn.completed":
            usage = event.get("usage")
        elif kind in ("turn.failed", "error"):
            errors.append(f"{kind}: {_short(event.get('error') or event.get('message') or event)}")
    responses = {(record.get("payload") or {}).get("response_id") for record in rollout
                 if record.get("type") == "token_usage_record"} - {None}
    context = next((record.get("payload") or {} for record in reversed(rollout)
                    if record.get("type") == "turn_context"), {})
    return dict(calls=[calls[key] for key in order], requests=len(responses), served_models=None,
                configured_model=context.get("model"), approval_policy=context.get("approval_policy"),
                sandbox=(context.get("sandbox_policy") or {}).get("type"), usage=usage, texts=texts,
                final_text=texts[-1] if texts else None, permission_denials=[], errors=errors)


def opencode_live_facts(messages):
    """Facts of a live OpenCode 2.2.7 session from its timeline ``messages`` (``OpenCodeClient.messages``).

    Every ``assistant`` message is one model step (one request). Returns ``{"calls": [{"tool", "error"}] in order
    ("ni_rua:<tool>" for ``ni_rua_`` tools, else the native name; "error" = the failed state's error text), "requests":
    the assistant messages, "served_models": None (OpenCode records the configured model ref, not the provider's
    answer), "configured_model" (``model.id``), "usage" (summed ``tokens`` and ``cost`` of the steps), "texts",
    "final_text" (the last text of the last step with text), "permission_denials": [] (the driver adds its rejects),
    "errors" (step errors and failed tool calls)}``."""
    calls, texts, errors, models = [], [], [], []
    usage = dict(input=0, output=0, reasoning=0, cache_read=0, cache_write=0, cost=0.0)
    requests = 0
    for message in messages:
        if message.get("type") != "assistant":
            continue
        requests += 1
        model = (message.get("model") or {}).get("id")
        if model and model not in models:
            models.append(model)
        tokens = message.get("tokens") or {}
        for key in ("input", "output", "reasoning"):
            usage[key] += tokens.get(key) or 0
        usage["cache_read"] += (tokens.get("cache") or {}).get("read") or 0
        usage["cache_write"] += (tokens.get("cache") or {}).get("write") or 0
        usage["cost"] += message.get("cost") or 0
        for item in message.get("content") or []:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "tool":
                state = item.get("state") or {}
                error = None
                if state.get("status") == "error" or state.get("error"):
                    error = _short(state.get("error") or _result_text(state.get("content")) or "error")
                calls.append(_call(item.get("name") or "", OpenCodeHost.prefix, error))
                if error:
                    errors.append(f"{calls[-1]['tool']}: {error}")
            elif item.get("type") == "text" and (item.get("text") or "").strip():
                texts.append(item["text"])
        if message.get("error"):
            errors.append(f"step: {_short(message['error'])}")
    usage["cost"] = round(usage["cost"], 6)
    return dict(calls=calls, requests=requests, served_models=None, configured_model=models[0] if models else None,
                usage=usage if requests else None, texts=texts, final_text=texts[-1] if texts else None,
                permission_denials=[], errors=errors)


def tool_payload(store, record):
    """The archived typed result (``kind``, ``data``, ``success``) of one ``tool_result`` record of ``store``."""
    _, data = store.resolve_artifact(record["raw_ref"], allowed_scopes=("observer",))
    return json.loads(data)


def ee_z(store, observation_id):
    """Measured end-effector z (m, so101_base) of a saved observation."""
    reference = store.read("observation", observation_id)["robot_state_ref"]
    state = json.loads(store.resolve_artifact(reference, allowed_scopes=("actor",))[1])
    return state["arms"]["right"]["ee_pose"]["position_m"][2]


def files_containing(root, needles):
    """Files under ``root`` (no symlinks followed) whose bytes contain any of ``needles`` (byte strings)."""
    found = []
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() and not path.is_symlink():
            data = path.read_bytes()
            if any(needle and needle in data for needle in needles):
                found.append(path)
    return found


def scrub_files(root, needles):
    """Overwrite every occurrence of each of ``needles`` in the files under ``root`` with as many ``*`` (same length,
    so databases keep their layout); returns the files changed. Longer needles first, so a key and its suffix are
    both removed."""
    changed = []
    for path in files_containing(root, needles):
        data = path.read_bytes()
        for needle in sorted(needles, key=len, reverse=True):
            data = data.replace(needle, b"*" * len(needle))
        path.write_bytes(data)
        changed.append(path)
    return changed


# ---------------------------------------------------------------------------------------------- Codex schemas
def codex_projection_problems(published, delivered, path=""):
    """Differences between a published input schema and the one Codex delivered that the known Codex projection does
    not explain (``[]`` = only the known stripping). Known: keywords of ``CODEX_DROPPED_KEYWORDS`` removed at any
    depth; a node reduced to ``{}`` when it carried a combinator Codex drops (``oneOf`` / ``anyOf`` / ``if``);
    ``"properties": {}`` added to an object node without properties; ``type`` added to a node that declared none,
    inferred from its ``enum`` values (``{"enum": ["right"]}`` -> ``{"type": "string", "enum": ["right"]}``).
    Everything Codex keeps must equal the published value, every published property name must survive (no parameter
    dropped) and no other keyword may be added."""
    problems = []
    if not isinstance(published, dict) or not isinstance(delivered, dict):
        return [] if published == delivered else [f"{path or '/'}: {delivered!r} != {published!r}"]
    if delivered == {} and published.keys() & {"oneOf", "anyOf", "if", "not"}:
        return []
    for key, value in delivered.items():
        where = f"{path}/{key}"
        if key not in published:
            inferred_object = published.get("type") == "object" or "required" in published or "properties" in published
            if not (key == "properties" and value == {} and inferred_object) \
                    and not (key == "type" and (
                        _inferred_type(published.get("enum")) == value
                        or (value == "object" and inferred_object))):
                problems.append(f"{where}: added by the host")
        elif key == "properties":
            missing = sorted(set(published[key]) - set(value))
            if missing:
                problems.append(f"{where}: properties dropped: {', '.join(missing)}")
            for name, schema in value.items():
                if name not in published[key]:
                    problems.append(f"{where}/{name}: property added by the host")
                else:
                    problems += codex_projection_problems(published[key][name], schema, f"{where}/{name}")
        elif key in ("items", "additionalProperties") and isinstance(value, dict):
            problems += codex_projection_problems(published[key], value, where)
        elif key in ("allOf", "anyOf", "oneOf") and isinstance(value, list) and isinstance(published[key], list) \
                and len(value) == len(published[key]):
            for index, (left, right) in enumerate(zip(published[key], value)):
                problems += codex_projection_problems(left, right, f"{where}[{index}]")
        elif value != published[key]:
            problems.append(f"{where}: {json.dumps(value)[:120]} != published {json.dumps(published[key])[:120]}")
    for key in published.keys() - delivered.keys():
        if key not in CODEX_DROPPED_KEYWORDS:
            problems.append(f"{path}/{key}: dropped by the host (not a known Codex projection keyword)")
    return problems


def _inferred_type(values):
    """The JSON type shared by every value of an ``enum`` list (``"string"``, ``"integer"``, ...), else None."""
    names = {bool: "boolean", int: "integer", float: "number", str: "string"}
    kinds = {names.get(type(value)) for value in values} if isinstance(values, list) and values else {None}
    return kinds.pop() if len(kinds) == 1 else None


def dropped_keywords(published, delivered, counts=None):
    """``{keyword: occurrences}`` the host dropped (report only)."""
    counts = {} if counts is None else counts
    if isinstance(published, dict) and isinstance(delivered, dict):
        for key, value in published.items():
            if key not in delivered:
                counts[key] = counts.get(key, 0) + 1
            elif key == "properties" and isinstance(value, dict):
                for name, schema in value.items():
                    dropped_keywords(schema, (delivered.get(key) or {}).get(name), counts)
            elif key in ("items", "additionalProperties"):
                dropped_keywords(value, delivered.get(key), counts)
    return counts


if __name__ == "__main__":
    gateway_main()
