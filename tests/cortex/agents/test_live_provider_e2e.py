"""Opt-in LIVE end-to-end test: the real gateway, an MCP agent backend (Claude Code or OpenCode) and a real provider.

What runs for real: the gateway CLI and ``serve`` (``ni_rua.apps.gateway``: config load, run registration, Workbench,
HTTP routes, console commands, shutdown and finalization), the Harness with the full registry actor catalogue, the
agent session (``NI_RUA_LIVE_E2E_BACKEND``: ``ClaudeCodeSession`` with the pinned ``claude-agent-sdk`` and its bundled
CLI, or ``OpenCodeSession`` with the pinned ``opencode serve`` 2.2.7), the model relay, the provider's model listing,
and the provider named by the agent tool's own config file. Only the robot is synthetic:
``support.so101_scene`` keeps the real ``so101_real`` backend and SO-101 adapter (kinematics, IK, motion execution,
receipts) and replaces the serial bus with ``FakeArmDriver`` and the two cameras with a rendered scene (a checkered
table with one cube of a randomly chosen colour). Every action reaches it through ``Harness.execute``.

Scenario, driven like the web console over loopback HTTP:

1. The gateway starts on model 1 of ``NI_RUA_LIVE_E2E_MODELS`` with the startup prompt (observe, name the cube
   colour, move the end effector 5 mm up), as an operator's ``ni-rua --agent-backend <backend> --model ...``.
2. For every further model, and once more on the same model when only one is listed: claim the writer lease,
   ``set_model`` through ``POST /v1/control``, then a ``prompt`` (observe, name the colour, move 5 mm back).

Handoff scenario (``NI_RUA_LIVE_E2E_HANDOFF`` set; it replaces the model-switch scenario): the gateway starts on
``NI_RUA_LIVE_E2E_BACKEND`` through the profile ``NI_RUA_LIVE_E2E_PROFILE`` on model 1 as above; every further model of
``NI_RUA_LIVE_E2E_MODELS`` is applied with ``apply_agent_settings`` and gets a ``prompt`` turn; then, for every hop of
``NI_RUA_LIVE_E2E_HANDOFF``, the console sends ``handoff_run`` (writer lease) to that backend, profile, model and
wire with the next turn prompt. The current run is finalised and the successor run starts in the same gateway process
on the same port; its startup turn is asserted like every other turn. At the end the gateway is stopped and the chain
is checked in the archive: every predecessor ``summary.json`` has ``status: stopped``, ``successor_run_id`` and
``successor_handoff.status: handed_off``; every successor ``manifest.json`` has ``handoff.predecessor_run_id`` /
``request_id`` and the requested backend and profile; ``run.handoff_requested`` / ``run.handoff_started`` are in the
two runs; ``run.turn_settings`` of every turn names the backend, profile, wire and model of that turn;
``summary.served_models`` of each run lists only that run's models; every run has ``cleanup_status: completed``; no
key of any profile of the chain is in any file. One ``LIVE_E2E_HANDOFF {...}`` line per hop reports the timing: the
``handoff_run`` call, acceptance to the successor's ``run.handoff_started`` (Workbench built) and to its
``run.started`` (robot and agent started, first turn admitted, manifest written), from the archived event times (UTC).

Asserted after each turn: the agent offered the model the full registry catalogue (``geometry_query``,
``robot_action``, ...; Claude Code: its ``init`` tool list; OpenCode: the tools of the first main request the relay
forwarded, recorded as ``agent.native_tools``);
the turn completed; ``robot_observe`` returned camera images; the last ``CUBE_COLOUR:`` line of the turn's messages
(final answer or commentary) names the rendered colour; a ``robot_action`` returned a typed ``action`` result whose
receipt is ``completed``; no model transport block. At the end: ``served_models`` lists every model, and the console's model control carries the provider
catalogue's suggestions. A failure prints the relay's operator evidence (``http_status`` and provider ``message`` of
every refused request, the transport block) and the tail of the CLI log. One ``LIVE_E2E_TURN {...}`` JSON line per
turn reports model, status, seconds, tool calls, served models and request latency.

Configuration: the actor features are the SO-101 run flags of 2026-11-13 (``USER_RUN_FEATURES``) with two changes,
because their services are not part of this test: ``remote.enabled=false`` (no SSH tunnel or GPU host) and
``features.vggt_endpoint`` pointing to a closed loopback port, so every VGGT-backed part (object table, point cloud,
height) reports ``unavailable: <reason>`` exactly as with the service down, while the tool catalogue and the actor
instructions stay those of the user's run. ``features.grasp_memory_dir`` and the archive root are private to the run.
A ``configs/ni-rua.local.json`` of the checkout still applies underneath (as for the gateway itself).

Environment (the test is skipped, with the reason, unless the first one is set):

* ``NI_RUA_LIVE_E2E_PROFILE``: path of a NI-RUA provider profile (``configs/profiles/<id>.local.json``, schema
  ``ni_rua.provider_profile.v1``; a path, never a key): ``upstream.source=profile``, so the run takes the profile's
  wire, base URL, auth scheme, compat opt-ins, Claude Code variables and OpenCode ``thinking_display`` exactly as the
  gateway does for a profile run configuration. Additionally asserted: ``manifest.agent_backend.upstream_profile``
  names the profile and wire, and the profile's key appears in no file of the run directory or the gateway logs
  (the test reads the key in-process for that scan and never prints it).
* ``NI_RUA_LIVE_E2E_WIRE``: ``upstream.wire_api`` of a profile run (needed for OpenCode with a profile that serves
  both ``anthropic_messages`` and ``openai_responses``; default unset = the only wire the backend can use).
* ``NI_RUA_LIVE_E2E_SETTINGS`` (instead of a profile): path of the agent tool's provider file in its official format,
  a path, never a key (the key is read by the gateway only). ``claude_code``: a Claude Code settings file
  (``env.ANTHROPIC_BASE_URL`` plus ``env.ANTHROPIC_AUTH_TOKEN`` or ``env.ANTHROPIC_API_KEY``, provider switches such as
  ``CLAUDE_CODE_THINKING_DISPLAY_UPDATES``; ``upstream.source=claude_settings``). ``opencode``: an OpenCode config
  file (V2 ``providers.<id>.{package, settings.{baseURL, apiKey}}`` or V1; ``upstream.source=opencode_config``).
* ``NI_RUA_LIVE_E2E_BACKEND``: ``claude_code`` (default) or ``opencode``. ``opencode`` also needs
  ``NI_RUA_OPENCODE_EXECUTABLE`` (the pinned 2.2.7 binary, ``agent.opencode.executable``) and ``rg``.
* ``NI_RUA_LIVE_E2E_MODELS``: comma-separated model ids (default ``kimi-k2.6``).
* ``NI_RUA_LIVE_E2E_EXPECT_REFUSED``: comma-separated models of the list that the provider is known to refuse (a
  documented provider limit). When such a model's turn fails, the test asserts instead that the console reports a
  retryable transport block with the provider's HTTP status and message, and recovers like an operator: ``set_model``
  to the next model, ``retry_model_requests``, ``prompt``. Its turn is not counted in ``served_models``.
* ``NI_RUA_LIVE_E2E_PROVIDER``: ``agent.<backend>.upstream.provider`` of a settings-file run (Claude Code: a label
  recorded with the run; OpenCode: the provider id in the file, optional when it declares one; default unset). A
  profile run's provider is the profile id.
* ``NI_RUA_LIVE_E2E_CONFIG``: optional JSON object deep-merged into the generated NI-RUA config, e.g. provider
  compatibility switches ``{"agent": {"<backend>": {"upstream": {...}}}}`` or a diagnostic
  ``{"features": {"disabled_tools": [...]}}`` (the catalogue check then reports the missing tool).
* ``NI_RUA_LIVE_E2E_HANDOFF``: the handoff scenario (needs ``NI_RUA_LIVE_E2E_PROFILE``). Hops separated by ``;``,
  each ``<backend>:<profile>:<model>[:<wire>]``: ``backend`` ``codex`` | ``claude_code`` | ``opencode``; ``profile`` a
  profile id or file in the directory of ``NI_RUA_LIVE_E2E_PROFILE`` (the run's ``agent.profiles_dir``; a path must
  not contain ``:``); ``model`` the successor's model id; ``wire`` ``anthropic_messages`` | ``openai_responses``
  (default: the only wire the backend can use with that profile). Every backend of the chain must be installed (an
  ``opencode`` hop needs ``NI_RUA_OPENCODE_EXECUTABLE`` and ``rg``). Example:
  ``opencode:provider-a:deepseek-flash;codex:provider-b:gpt-6.1-terra``.
* ``NI_RUA_LIVE_E2E_EFFORT``: ``model.effort`` (default ``high``, the user's run); also the effort of every hop.
* ``NI_RUA_LIVE_E2E_TURN_TIMEOUT_S``: per-turn wait in s (default 900).
* ``NI_RUA_LIVE_E2E_CUBE_COLOUR``: one of ``support.so101_scene.COLOURS`` (default: random).
* ``NI_RUA_LIVE_E2E_RUN_ROOT``: keep the run under this directory (default: a temporary directory, removed).

Run from the checkout root::

    NI_RUA_LIVE_E2E_PROFILE=configs/profiles/<id>.local.json NI_RUA_LIVE_E2E_MODELS=<model> \
        PYTHONPATH=tests mamba run -n ni-rua python -m unittest -v test_live_provider_e2e
    NI_RUA_LIVE_E2E_SETTINGS=/path/claude.provider.json NI_RUA_LIVE_E2E_MODELS=kimi-k2.6,qwen3.8-flash \
        PYTHONPATH=tests mamba run -n ni-rua python -m unittest -v test_live_provider_e2e
    NI_RUA_LIVE_E2E_BACKEND=opencode NI_RUA_LIVE_E2E_SETTINGS=/path/opencode.provider.json \
        NI_RUA_OPENCODE_EXECUTABLE=~/.local/share/ni-rua/opencode-2.2.7/opencode \
        NI_RUA_LIVE_E2E_MODELS=qwen3.8-flash,deepseek-flash PYTHONPATH=tests mamba run -n ni-rua python -m unittest \
        -v test_live_provider_e2e
    NI_RUA_LIVE_E2E_PROFILE=configs/profiles/<id>.local.json NI_RUA_LIVE_E2E_MODELS=<model 1>,<model 2> \
        NI_RUA_LIVE_E2E_HANDOFF='opencode:<id 2>:<model 3>;codex:<id 3>:<model 4>' \
        NI_RUA_OPENCODE_EXECUTABLE=~/.local/share/ni-rua/opencode-2.2.7/opencode \
        PYTHONPATH=tests mamba run -n ni-rua python -m unittest -v test_live_provider_e2e

This calls a real, charged model provider; it never runs in the default suite.
"""

NI_RUA_TEST_TIER = "component"

import http.client
import json
import os
import random
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from ni_rua.core.storage.store import open_read_store
from support import so101_scene
from support.host_e2e import ee_z, tool_payload
from support.paths import REPO_ROOT, TESTS_ROOT

REPOSITORY = REPO_ROOT
PROFILE = os.environ.get("NI_RUA_LIVE_E2E_PROFILE")
WIRE = os.environ.get("NI_RUA_LIVE_E2E_WIRE") or None
SETTINGS = os.environ.get("NI_RUA_LIVE_E2E_SETTINGS")
BACKEND = os.environ.get("NI_RUA_LIVE_E2E_BACKEND") or "claude_code"
OPENCODE_EXECUTABLE = os.environ.get("NI_RUA_OPENCODE_EXECUTABLE")
# Per backend: upstream source of the provider file, and the prefix the agent puts before registry tool names.
UPSTREAM_SOURCES = {"claude_code": "claude_settings", "opencode": "opencode_config"}
TOOL_PREFIXES = {"claude_code": "mcp__ni_rua__", "opencode": "ni_rua_"}
MODELS = [item.strip() for item in os.environ.get("NI_RUA_LIVE_E2E_MODELS", "kimi-k2.6").split(",") if item.strip()]
REFUSED = {item.strip() for item in os.environ.get("NI_RUA_LIVE_E2E_EXPECT_REFUSED", "").split(",") if item.strip()}
TURN_TIMEOUT_S = float(os.environ.get("NI_RUA_LIVE_E2E_TURN_TIMEOUT_S", "900"))
EFFORT = os.environ.get("NI_RUA_LIVE_E2E_EFFORT", "high")
HANDOFF = os.environ.get("NI_RUA_LIVE_E2E_HANDOFF") or None
HANDOFF_BACKENDS = ("codex", "claude_code", "opencode")
WIRES = ("anthropic_messages", "openai_responses")
STARTUP_TIMEOUT_S = 300.0
SHUTDOWN_TIMEOUT_S = 240.0


def profile_reference(text, directory):
    """``(profile id, profile path)`` of a hop's ``<profile>``: an id in ``directory`` or a file in it."""
    path = Path(text).expanduser()
    if "/" in text or text.endswith(".local.json"):
        path = path.resolve()
        if path.parent != directory:
            raise ValueError(f"NI_RUA_LIVE_E2E_HANDOFF profile {text!r} is not in {directory} (the run's "
                             "agent.profiles_dir, the directory of NI_RUA_LIVE_E2E_PROFILE)")
        return path.name.removesuffix(".local.json"), path
    return text, directory / f"{text}.local.json"


def parse_hops(text, directory):
    """The hops of ``NI_RUA_LIVE_E2E_HANDOFF``: ``[{backend, profile, path, model, wire}]`` (module docstring)."""
    hops = []
    for part in (item.strip() for item in text.split(";")):
        if not part:
            continue
        pieces = part.split(":")
        if len(pieces) < 3 or not all(pieces[:3]):
            raise ValueError(f"NI_RUA_LIVE_E2E_HANDOFF hop {part!r} is not <backend>:<profile>:<model>[:<wire>]")
        backend, profile, rest = pieces[0], pieces[1], pieces[2:]
        wire = rest.pop() if len(rest) > 1 and rest[-1] in WIRES else None
        if backend not in HANDOFF_BACKENDS:
            raise ValueError(f"NI_RUA_LIVE_E2E_HANDOFF hop {part!r}: backend must be one of {', '.join(HANDOFF_BACKENDS)}")
        identifier, path = profile_reference(profile, directory)
        if not path.is_file():
            raise ValueError(f"NI_RUA_LIVE_E2E_HANDOFF hop {part!r}: no profile file {path}")
        hops.append(dict(backend=backend, profile=identifier, path=path, model=":".join(rest), wire=wire))
    if not hops:
        raise ValueError("NI_RUA_LIVE_E2E_HANDOFF names no hop")
    return hops


def _runtime_problem(backend):
    from ni_rua.cortex.agents import factory
    if backend == "codex":
        return None    # the pinned Codex SDK and CLI are dependencies of the environment
    if backend == "opencode":
        import shutil
        runtime = factory.opencode_runtime_problem(OPENCODE_EXECUTABLE) or (
            None if shutil.which("rg") else "ripgrep (rg) is not installed")
        return f"OpenCode runtime unavailable: {runtime}" if runtime else None
    runtime = factory.claude_code_runtime_problem()
    return f"Claude Code runtime unavailable: {runtime}" if runtime else None


def _problem():
    if not (PROFILE or SETTINGS):
        return ("set NI_RUA_LIVE_E2E_PROFILE to a NI-RUA provider profile, or NI_RUA_LIVE_E2E_SETTINGS to the agent "
                "tool's provider file (Claude Code settings or OpenCode config), to run the live provider end-to-end "
                "test (calls a real, charged model provider)")
    if PROFILE and SETTINGS:
        return "set only one of NI_RUA_LIVE_E2E_PROFILE and NI_RUA_LIVE_E2E_SETTINGS"
    if BACKEND not in UPSTREAM_SOURCES:
        return f"NI_RUA_LIVE_E2E_BACKEND must be one of {', '.join(UPSTREAM_SOURCES)}, not {BACKEND!r}"
    backends = [BACKEND]
    if HANDOFF:
        if not PROFILE:
            return "NI_RUA_LIVE_E2E_HANDOFF needs NI_RUA_LIVE_E2E_PROFILE (handoff_run selects provider profiles)"
        try:
            hops = parse_hops(HANDOFF, Path(PROFILE).expanduser().resolve().parent)
        except ValueError as error:
            return str(error)
        backends += [hop["backend"] for hop in hops if hop["backend"] not in backends]
    for backend in backends:
        problem = _runtime_problem(backend)
        if problem:
            return problem
    return None


PROBLEM = _problem()
# features of the SO-101 run configuration of 2026-11-13 (~/.cache/ni-rua/so101/ni-rua.json); see the module doc.
USER_RUN_FEATURES = {
    "observation_profile": "rgb_proprio_calibrated_v1", "external_perception": True, "sandbox_enabled": False,
    "motion_review": False, "vggt_endpoint": "http://127.0.0.1:8331", "vggt_camera_id": "fixed",
    "vggt_support_z_m": None, "held_perception": False, "vggt_height": True, "vggt_height_mode": "optional",
    "vggt_timeout_s": 60.0, "object_table": True, "object_table_max_objects": 20, "object_table_max_height_m": 0.3,
    "object_table_head_exclude_px": "200,405,440,480", "scene3d": True, "scene3d_voxel_m": 0.01,
    "scene3d_z_above_m": 0.6, "scene3d_z_below_m": 0.05, "object_table_observe": True, "scene3d_observe": True,
    "obs3d_timeout_s": 60.0, "obs3d_max_objects": 10, "obs3d_action_overlay": False}
# Registry tools the real SO-101 catalogue offers the actor (the CLI must not drop any of them).
EXPECTED_TOOLS = ("geometry_query", "robot_action", "robot_observe", "robot_capabilities", "robot_information",
                  "task_plan", "robot_wait", "history_view_observation", "program_run", "vision_compare")
COLOUR_WORDS = so101_scene.COLOUR_WORDS
COLOUR_LINE = re.compile(r"CUBE[_ ]COLOU?R\W*([A-Za-z]+)", re.I)
TERMINAL_AGENT_STATES = ("completed", "failed", "interrupted", "disconnected", "closed")
PROMPT = ("Live interface check of the SO-101 arm. {observe} and look at its camera images. On a line of its own "
          "write CUBE_COLOUR: <one English colour word> for the cube on the table (write CUBE_COLOUR: unknown if you "
          "received no image; do not guess). Then move the end effector 5 mm straight {direction} ({axis} in "
          "so101_base) with robot_action, keeping its orientation and the gripper as they are, timeout_s=60, at most "
          "one movement. Report the requested and the measured end-effector z.")


def turn_prompt(index):
    up = index % 2 == 0
    return PROMPT.format(observe="Call robot_observe" if index == 0 else "Call robot_observe again",
                         direction="up" if up else "down", axis="+z" if up else "-z")


def merge(base, overlay):
    """``base`` with the JSON object ``overlay`` merged in (objects recursively, every other value replaced)."""
    if not isinstance(overlay, dict):
        raise ValueError("NI_RUA_LIVE_E2E_CONFIG must be a JSON object")
    result = dict(base)
    for key, value in overlay.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def closed_loopback_port():
    """A loopback port nothing listens on (bound once, then released)."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class Gateway:
    """The gateway subprocess (``python -m support.so101_scene``) and its loopback HTTP API."""

    def __init__(self, root, config_path, scene_path, first_model, prompt):
        self.root, self.run_dir = root, root / "run"
        self.stdout_path, self.stderr_path = root / "gateway.stdout.log", root / "gateway.stderr.log"
        env = dict(os.environ, **{so101_scene.SCENE_ENV: str(scene_path)})
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(TESTS_ROOT), str(REPOSITORY),
                                                          os.environ.get("PYTHONPATH")]))
        command = [sys.executable, "-m", "support.so101_scene", "--config", str(config_path),
                   "--agent-backend", BACKEND, "--model", first_model, "--run-dir", str(self.run_dir),
                   "--port", "0", "--no-remote", "--prompt", prompt]
        with self.stdout_path.open("w") as stdout, self.stderr_path.open("w") as stderr:
            self.process = subprocess.Popen(command, cwd=REPOSITORY, env=env, stdout=stdout, stderr=stderr,
                                            start_new_session=True)
        self.base = self.origin = None
        self.lease = None

    def wait_listening(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            match = re.search(r"Workbench http://127\.0\.0\.1:(\d+);", self.stdout_path.read_text())
            if match:
                self.base = self.origin = f"http://127.0.0.1:{match.group(1)}"
                return
            if self.process.poll() is not None:
                raise AssertionError(f"gateway exited ({self.process.returncode}) before listening:\n{self.tail()}")
            time.sleep(0.2)
        raise AssertionError(f"gateway not listening after {timeout} s:\n{self.tail()}")

    def tail(self, lines=40):
        text = []
        for path in (self.stdout_path, self.stderr_path):
            try:
                text.append(f"--- {path.name}\n" + "\n".join(path.read_text().splitlines()[-lines:]))
            except OSError:
                pass
        return "\n".join(text)

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=60) as response:
            return json.loads(response.read())

    def runs(self):
        """``[(port, run directory)]`` of every run this gateway process served so far (stdout lines)."""
        return [(int(port), Path(root)) for port, root in re.findall(r"Workbench http://127\.0\.0\.1:(\d+); run (\S+)",
                                                                       self.stdout_path.read_text())]

    def post(self, path, body):
        request = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), method="POST",
                                         headers={"Content-Type": "application/json", "Origin": self.origin})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read() or b"{}")

    def control(self, command, request_id=None, **fields):
        """``POST /v1/control`` as the console sends it: claim or renew the writer lease, then the command."""
        claim = dict(request_id="claim-" + os.urandom(6).hex(), command="claim_input", client_id="live-e2e-console")
        if self.lease:
            claim["writer_token"] = self.lease
        status, body = self.post("/v1/control", claim)
        if status != 200:
            raise AssertionError(f"claim_input refused ({status}): {body}")
        self.lease = body["writer_token"]
        return self.post("/v1/control", dict(request_id=request_id or command + "-" + os.urandom(6).hex(),
                                              command=command, client_id="live-e2e-console", writer_token=self.lease,
                                              **fields))

    def wait(self, predicate, timeout, what, interval=1.0):
        deadline = time.monotonic() + timeout
        snapshot = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise AssertionError(f"gateway exited ({self.process.returncode}) while waiting for {what}:\n"
                                     f"{self.tail()}")
            try:
                snapshot = self.get("/v1/run")
            except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException):
                snapshot = None    # also while a handoff swaps the run behind the port
            if snapshot is not None and predicate(snapshot):
                return snapshot
            time.sleep(interval)
        raise AssertionError(f"timed out after {timeout} s waiting for {what}; last snapshot: {snapshot}")

    def stop(self):
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(SHUTDOWN_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(30)
                raise AssertionError(f"gateway did not stop within {SHUTDOWN_TIMEOUT_S} s after SIGTERM")
        return self.process.returncode


def read_store(run_dir, function):
    with open_read_store(run_dir, live=True) as store:
        return function(store)


def has_image(items):
    return any(isinstance(item, dict) and (item.get("type") in ("inputImage", "image")
                                           or str(item.get("imageUrl", "")).startswith("data:image/"))
               for item in items or [])


@unittest.skipIf(PROBLEM is not None, PROBLEM)
class LiveGatewayAgentTest(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        variable = "NI_RUA_LIVE_E2E_PROFILE" if PROFILE else "NI_RUA_LIVE_E2E_SETTINGS"
        settings = Path(PROFILE or SETTINGS).expanduser()
        if not settings.is_file():
            self.fail(f"{variable} does not name a readable file: {settings}")
        self.settings = settings.resolve()
        parent = os.environ.get("NI_RUA_LIVE_E2E_RUN_ROOT")
        if parent:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            self.root = Path(parent).expanduser().resolve() / f"{stamp}-{'-'.join(MODELS)}"[:120]
            self.root.mkdir(parents=True)
        else:
            temporary = tempfile.TemporaryDirectory(prefix="ni-rua-live-e2e-")
            self.addCleanup(temporary.cleanup)
            self.root = Path(temporary.name)
        self.colour = os.environ.get("NI_RUA_LIVE_E2E_CUBE_COLOUR") or random.SystemRandom().choice(
            sorted(so101_scene.COLOURS))
        deployment, self.scene_path = so101_scene.write_scene(self.root / "scene",
                                                              so101_scene.scene_document(self.colour))
        if PROFILE:
            upstream = dict(source="profile", settings=str(self.settings), wire_api=WIRE)
        else:
            upstream = dict(source=UPSTREAM_SOURCES[BACKEND], settings=str(self.settings),
                            provider=os.environ.get("NI_RUA_LIVE_E2E_PROVIDER") or None)
        agent = {"backend": BACKEND, BACKEND: dict(upstream=upstream)}
        self.hops = parse_hops(HANDOFF, self.settings.parent) if HANDOFF else []
        if BACKEND == "opencode" or any(hop["backend"] == "opencode" for hop in self.hops):
            # A handoff to OpenCode starts from this configuration (apply_profile), so it carries the executable.
            agent.setdefault("opencode", {})["executable"] = str(Path(OPENCODE_EXECUTABLE).expanduser())
        if PROFILE:
            agent["profiles_dir"] = str(self.settings.parent)   # handoff_run and the console read profiles here
        features = dict(USER_RUN_FEATURES, vggt_endpoint=f"http://127.0.0.1:{closed_loopback_port()}",
                        grasp_memory_dir=str(self.root / "grasp-memory"))
        self.config_path = self.root / "ni-rua.json"
        config = dict(
            version=1, model=dict(name=MODELS[0], effort=EFFORT),
            backend=dict(environment="so101_real", robot_model_id="so101", deployment=str(deployment)),
            features=features, agent=agent,
            run=dict(experiment_root=str(self.root / "exp"), category="live_e2e", task="desk_free_task", port=0),
            remote=dict(enabled=False))
        self.config_path.write_text(json.dumps(merge(config, json.loads(os.environ.get("NI_RUA_LIVE_E2E_CONFIG")
                                                                        or "{}")), indent=2))
        source = f"profile {self.settings.name} wire {WIRE or 'default'}" if PROFILE else f"settings {self.settings.name}"
        chain = "; handoff " + " -> ".join(f"{hop['backend']}+{hop['profile']}:{hop['model']}" for hop in self.hops) \
            if self.hops else ""
        print(f"\nLIVE_E2E run root {self.root}; backend {BACKEND}; {source}; cube colour {self.colour}; "
              f"models {MODELS}{chain}", flush=True)

    # ---- evidence ---------------------------------------------------------------------------------------------
    def evidence(self, gateway, after=0):
        """Operator evidence of the relay and the agent after event ``after`` (never model context)."""
        lines = []
        try:
            lines.append("console model_transport: " + json.dumps(gateway.get("/v1/console/state")["model_transport"]))
        except Exception as error:  # noqa: BLE001 - evidence is best effort
            lines.append(f"console state unavailable: {type(error).__name__}: {error}")

        def collect(store):
            for event in store.events_by_type(["model.model_request_finished", "model.model_transport_blocked",
                                               "agent.turn_completed", "agent.connection_lost",
                                               "agent.model_change_rejected", "agent.model_changed",
                                               "agent.model_declared", "agent.model_declare_failed",
                                               "agent.permission_auto_rejected", "agent.turn_verdict",
                                               "agent.model_image_input_unsupported"], after=after, limit=1000):
                payload = event["payload"]
                if event["type"] == "model.model_request_finished":
                    if payload.get("http_status") == 200:
                        continue
                    error = payload.get("response_error") or {}
                    lines.append(f"request refused: http_status={payload.get('http_status')} "
                                 f"model={payload.get('requested_model')} classification={error.get('classification')} "
                                 f"code={error.get('code')} type={error.get('type')} message={error.get('message')!r} "
                                 f"error_type={payload.get('error_type')}")
                elif event["type"] == "model.model_transport_blocked":
                    evidence = (payload.get("error") or {}).get("evidence") or {}
                    lines.append("transport blocked: " + json.dumps({key: evidence.get(key) for key in (
                        "http_status", "classification", "code", "type", "message")}))
                else:
                    lines.append(f"{event['type']}: {json.dumps(payload)[:400]}")
        try:
            read_store(gateway.run_dir, collect)
        except Exception as error:  # noqa: BLE001
            lines.append(f"store unavailable: {type(error).__name__}: {error}")
        log = gateway.run_dir / "private" / "agent.log"
        if log.is_file():
            lines.append("--- agent log tail\n" + "\n".join(log.read_text(errors="replace").splitlines()[-15:]))
        return "\n".join(lines)

    def check(self, gateway, condition, message, after=0):
        if not condition:
            self.fail(f"{message}\n=== evidence ===\n{self.evidence(gateway, after)}")

    # ---- per-turn facts ---------------------------------------------------------------------------------------
    def turn_facts(self, gateway, turn_id, after):
        items = [item for item in gateway.get("/v1/items")["items"] if item.get("turn_id") == turn_id]

        def collect(store):
            results = []
            for record in store.records("tool_result"):
                if (record.get("correlation") or {}).get("turn_id") != turn_id or record.get("tool") != "robot_action":
                    continue
                payload = tool_payload(store, record)
                receipt = (payload.get("data") or {}).get("receipt") or {}
                dz = None
                try:
                    before = store.read("action_request", payload["data"]["action_id"])["based_on_observation_id"]
                    dz = round(ee_z(store, receipt["result_observation_id"]) - ee_z(store, before), 4)
                except (KeyError, TypeError, ValueError):
                    pass
                results.append(dict(kind=payload.get("kind"), success=payload.get("success"),
                                    status=receipt.get("status"), batch_status=receipt.get("batch_status"),
                                    measured_dz_m=dz))
            requests = [event["payload"] for event in store.events_by_type(["model.model_request_finished"],
                                                                            after=after, limit=1000)]
            return results, requests, store.read_state("runtime", "model_requests") or {}
        actions, requests, model_requests = read_store(gateway.run_dir, collect)
        return items, actions, requests, model_requests

    @staticmethod
    def final_answer(items):
        texts = [item.get("text") or "" for item in items
                 if item.get("type") == "agentMessage" and item.get("phase") == "final_answer"]
        return "\n".join(texts) if texts else None

    def native_tools(self, gateway, backend=BACKEND):
        """Tool names the agent really offers the model, with its registry prefix: Claude Code's ``init`` message;
        OpenCode's first main request as the relay forwarded it (``agent.native_tools``)."""
        def collect(store):
            if backend == "opencode":
                for event in store.events_by_type(["agent.native_tools"], limit=10):
                    payload = event["payload"]
                    return list(payload.get("enabled") or []) + [TOOL_PREFIXES[backend] + name
                                                                for name in payload.get("registry") or []]
                return None
            for event in store.events_by_type(["agent.native"], limit=1000):
                if event["payload"].get("kind") != "SystemMessage":
                    continue
                _, data = store.resolve_artifact(event["payload"]["raw_ref"], allowed_scopes=("observer",))
                record = json.loads(data)
                if record.get("subtype") == "init":
                    return list((record.get("data") or {}).get("tools") or [])
            return None
        return read_store(gateway.run_dir, collect)

    def catalogue_check(self, gateway, backend):
        """The registry catalogue of the current run is the real one and the agent offers all of it (after the run's
        first turn, before its checks). Codex receives the registry as native dynamic tools (no CLI tool list to
        read), so for Codex only the registry itself is checked."""
        manifest = json.loads((gateway.run_dir / "manifest.json").read_text())
        registry = [tool["name"] for tool in manifest["tool_catalog"]["tools"] if tool["available"]]
        offered = native = None
        if backend in TOOL_PREFIXES:
            tools = self.native_tools(gateway, backend) or []
            prefix = TOOL_PREFIXES[backend]
            offered = sorted(name.removeprefix(prefix) for name in tools if name.startswith(prefix))
            native = sorted(name for name in tools if not name.startswith(prefix))
        print("LIVE_E2E_CATALOGUE " + json.dumps(dict(backend=backend, registry=registry, cli_offered=offered,
                                                      native=native)), flush=True)
        if offered is not None:
            self.check(gateway, set(registry) <= set(offered),
                       f"the agent does not offer {sorted(set(registry) - set(offered))} (offered: {offered})")
        self.check(gateway, set(EXPECTED_TOOLS) <= set(registry),
                   f"registry catalogue lacks {sorted(set(EXPECTED_TOOLS) - set(registry))}: {registry}")

    def run_turn(self, gateway, index, model, turn_id, started, after, report, before_checks=None):
        snapshot = gateway.wait(lambda s: s.get("turn_id") == turn_id and s.get("agent_status") in TERMINAL_AGENT_STATES,
                                TURN_TIMEOUT_S, f"turn {index + 1} ({model}) to end")
        seconds = round(time.monotonic() - started, 1)
        items, actions, requests, model_requests = self.turn_facts(gateway, turn_id, after)
        tools = [item.get("tool") or item.get("command") or item.get("type") for item in items
                 if item.get("type") in ("dynamicToolCall", "mcpToolCall", "commandExecution", "fileChange")]
        observes = [item for item in items if item.get("type") == "dynamicToolCall" and item.get("tool") == "robot_observe"]
        answer = self.final_answer(items)
        # The colour line may be in any message of the turn (models often write it before the move, as commentary).
        named = COLOUR_LINE.findall("\n".join(item.get("text") or "" for item in items
                                              if item.get("type") == "agentMessage"))
        elapsed = [payload.get("elapsed_s") for payload in requests if isinstance(payload.get("elapsed_s"), (int, float))]
        line = dict(turn=index + 1, model=model, turn_id=turn_id, status=snapshot.get("agent_status"),
                    seconds=seconds, tool_calls=tools, robot_actions=actions,
                    requests=len(requests), http_statuses=sorted({payload.get("http_status") for payload in requests},
                                                                 key=str),
                    request_s_mean=round(sum(elapsed) / len(elapsed), 2) if elapsed else None,
                    request_s_max=round(max(elapsed), 2) if elapsed else None,
                    served_models=model_requests.get("served_models"),
                    transport_blocked=model_requests.get("transport_blocked"),
                    cube_colour_answer=named[-1] if named else None, expected_colour=self.colour,
                    answer=(answer or "")[-600:])
        report.append(line)
        print("LIVE_E2E_TURN " + json.dumps(line), flush=True)
        if before_checks is not None:
            before_checks()
        if model in REFUSED and snapshot.get("agent_status") == "failed":
            # A documented provider limit: the refusal must be explicit and recoverable from the console.
            transport = gateway.get("/v1/console/state")["model_transport"]
            line["refusal"] = {key: transport.get(key) for key in ("http_status", "type", "message")}
            print("LIVE_E2E_REFUSED " + json.dumps(dict(model=model, **line["refusal"])), flush=True)
            self.check(gateway, transport.get("blocked") and transport.get("manual_retry_allowed")
                       and isinstance(transport.get("http_status"), int) and transport.get("message"),
                       f"turn {index + 1} ({model}): the expected provider refusal is not reported to the operator "
                       f"as a retryable block with the provider message: {transport}", after)
            return "refused"
        self.check(gateway, snapshot.get("agent_status") == "completed",
                   f"turn {index + 1} ({model}) ended {snapshot.get('agent_status')}", after)
        self.check(gateway, observes and all(item.get("status") == "completed" for item in observes)
                   and any(has_image(item.get("contentItems")) for item in observes),
                   f"turn {index + 1} ({model}): robot_observe did not return camera images "
                   f"({[(item.get('status'), len(item.get('contentItems') or [])) for item in observes]})", after)
        self.check(gateway, bool(named) and named[-1].lower() in COLOUR_WORDS[self.colour],
                   f"turn {index + 1} ({model}): cube colour {named[-1] if named else None!r} != {self.colour!r}; "
                   f"answer: {answer!r}", after)
        self.check(gateway, any(action["kind"] == "action" and action["status"] == "completed" for action in actions),
                   f"turn {index + 1} ({model}): no robot_action with a completed typed receipt: {actions}", after)
        self.check(gateway, not model_requests.get("transport_blocked"),
                   f"turn {index + 1} ({model}): model transport blocked", after)
        return "completed"

    # ---- the scenario -----------------------------------------------------------------------------------------
    @unittest.skipIf(HANDOFF, "NI_RUA_LIVE_E2E_HANDOFF is set: the handoff scenario runs instead")
    def test_observe_move_and_switch_models_through_the_console(self):
        first_prompt = turn_prompt(0)
        gateway = Gateway(self.root, self.config_path, self.scene_path, MODELS[0], first_prompt)
        self.addCleanup(lambda: gateway.process.poll() is None and gateway.stop())
        started = time.monotonic()
        gateway.wait_listening(STARTUP_TIMEOUT_S)
        snapshot = gateway.wait(lambda s: s.get("status") != "starting", STARTUP_TIMEOUT_S,
                                "the gateway start (startup turn admitted)")
        self.check(gateway, snapshot["status"] == "running" and snapshot["agent_backend"] == BACKEND
                   and snapshot.get("turn_id"), f"startup did not reach a running {BACKEND} turn: {snapshot}")
        report = []
        outcome = self.run_turn(gateway, 0, MODELS[0], snapshot["turn_id"], started, 0, report,
                                before_checks=lambda: self.catalogue_check(gateway, BACKEND))

        follow = MODELS[1:] or MODELS[:1]
        for index, model in enumerate(follow, start=1):
            after = read_store(gateway.run_dir, lambda store: store.event_cursor())
            status, body = gateway.control("set_model", model=model)
            self.check(gateway, status == 200 and body.get("status") == "applied",
                       f"set_model {model} refused ({status}): {body}", after)
            if outcome == "refused":
                # The console recovery: switch first (applied at the next prompt), then lift the block.
                status, body = gateway.control("retry_model_requests")
                self.check(gateway, status == 200 and body.get("status") == "retry_authorized",
                           f"retry_model_requests refused ({status}): {body}", after)
            started = time.monotonic()
            status, body = gateway.control("prompt", prompt=turn_prompt(index))
            self.check(gateway, status == 200 and body.get("status") == "running" and body.get("turn_id"),
                       f"console prompt refused ({status}): {body}", after)
            outcome = self.run_turn(gateway, index, model, body["turn_id"], started, after, report)

        state = gateway.get("/v1/console/state")
        control = state["settings"]["runtime"]["model"]
        options = {item["value"]: item for item in control["options"]}
        self.check(gateway, outcome == "refused" or not state["model_transport"]["blocked"],
                   f"model transport blocked: {state['model_transport']}")
        # The suggestions are the provider listing the session read (``agent.model_catalogue``): every usable listed
        # model is offered, labelled with its display name when the provider gives one (some listings, e.g. a bare
        # Anthropic-style gateway, carry ids only), and every model of the run is among them.
        listing = read_store(gateway.run_dir, lambda store: [event["payload"] for event in store.events_by_type(
            ["agent.model_catalogue"], limit=10)])
        listed = {item["id"]: item.get("name") for item in (listing[-1]["models"] if listing else [])}
        print("LIVE_E2E_LISTING " + json.dumps(dict(status=listing[-1]["status"] if listing else None, listed=len(listed),
                                                    source=listing[-1].get("source") if listing else None,
                                                    origin=listing[-1].get("origin") if listing else None)), flush=True)
        self.check(gateway, bool(listing) and listing[-1]["status"] in ("ok", "truncated") and set(listed) <= set(options)
                   and all(options[model]["label"] != model for model, name in listed.items() if name and name != model)
                   and all(model in options for model in MODELS) and len(options) > len(MODELS) + 4,
                   f"console model suggestions are not the provider catalogue: {control['options']}")
        served = report[-1]["served_models"] or {}
        completed = [line["model"] for line in report if line.get("refusal") is None]
        self.check(gateway, all(model in served for model in completed),
                   f"served_models {served} lack one of {completed}")

        self.assertEqual(gateway.stop(), 0, gateway.tail())
        summary = json.loads((gateway.run_dir / "summary.json").read_text())
        agent = read_store(gateway.run_dir, lambda store: store.read_state("runtime", "agent")) or {}
        print("LIVE_E2E_SUMMARY " + json.dumps(dict(
            backend=BACKEND, status=summary.get("status"), cleanup_status=summary.get("cleanup_status"),
            served_models=summary.get("served_models"), clean_shutdown=agent.get("clean_shutdown"),
            cli_version=agent.get("cli_version"), sdk_version=agent.get("sdk_version"), turns=len(report))), flush=True)
        self.assertEqual(summary.get("cleanup_status"), "completed", summary.get("cleanup"))
        self.assertIs(agent.get("clean_shutdown"), True, agent)
        if PROFILE:
            self.check_profile_run(gateway)

    def check_profile_run(self, gateway):
        """A profile run: the manifest names the profile and its wire (names only), and the profile's key is in no
        file of the run directory or the gateway logs. The key is read here only for the scan and never printed."""
        from ni_rua.cortex.agents import profiles
        profile = profiles.load_profile(self.settings)
        manifest = json.loads((gateway.run_dir / "manifest.json").read_text())
        recorded = (manifest.get("agent_backend") or {}).get("upstream_profile") or {}
        wire = recorded.get("wire_api")
        print("LIVE_E2E_PROFILE " + json.dumps(dict(upstream_profile=recorded, upstream_provider=(
            manifest.get("agent_backend") or {}).get("upstream_provider"))), flush=True)
        self.assertEqual((recorded.get("id"), sorted(recorded.get("wires") or [])), (profile.id, sorted(profile.wires)))
        self.assertIn(wire, profile.wires)
        if WIRE:
            self.assertEqual(wire, WIRE)
        _base_url, headers = profiles.credential(profile, wire)
        key = next(iter(headers.values())).removeprefix("Bearer ").encode()
        leaks = [str(path) for path in self.root.rglob("*")      # run directory, gateway logs, scene, archive
                 if path.is_file() and not path.is_symlink() and key in path.read_bytes()]
        del key, headers
        self.assertEqual(leaks, [], "the profile key was found in these files")

    # ---- the handoff scenario ---------------------------------------------------------------------------------
    @unittest.skipUnless(HANDOFF, "set NI_RUA_LIVE_E2E_HANDOFF (with NI_RUA_LIVE_E2E_PROFILE) for the handoff scenario")
    def test_handoff_chain_through_the_console(self):
        from ni_rua.cortex.agents import profiles
        first = profiles.load_profile(self.settings)
        gateway = Gateway(self.root, self.config_path, self.scene_path, MODELS[0], turn_prompt(0))
        self.addCleanup(lambda: gateway.process.poll() is None and gateway.stop())
        started = time.monotonic()
        gateway.wait_listening(STARTUP_TIMEOUT_S)
        snapshot = gateway.wait(lambda s: s.get("status") != "starting", STARTUP_TIMEOUT_S,
                                "the gateway start (startup turn admitted)")
        self.check(gateway, snapshot["status"] == "running" and snapshot["agent_backend"] == BACKEND
                   and snapshot.get("turn_id"), f"startup did not reach a running {BACKEND} turn: {snapshot}")
        report = []
        run = dict(run_id=snapshot["run_id"], directory=gateway.run_dir, backend=BACKEND, profile=first.id, wire=WIRE,
                   models=[MODELS[0]], turns=[snapshot["turn_id"]], request_id=None, polled=None)
        runs = [run]
        self.run_turn(gateway, 0, MODELS[0], snapshot["turn_id"], started, 0, report,
                      before_checks=lambda: self.catalogue_check(gateway, BACKEND))
        index = 1
        for model in MODELS[1:]:
            # A live model change of the running session: one apply_agent_settings receipt, then a turn.
            after = read_store(gateway.run_dir, lambda store: store.event_cursor())
            status, body = gateway.control("apply_agent_settings", model=model)
            result = (body.get("results") or {}).get("model") or {}
            self.check(gateway, status == 200 and body.get("status") == "applied" and result.get("status") == "applied",
                       f"apply_agent_settings model={model} refused ({status}): {body}", after)
            started = time.monotonic()
            status, body = gateway.control("prompt", prompt=turn_prompt(index))
            self.check(gateway, status == 200 and body.get("status") == "running" and body.get("turn_id"),
                       f"console prompt refused ({status}): {body}", after)
            run["models"].append(model)
            run["turns"].append(body["turn_id"])
            self.run_turn(gateway, index, model, body["turn_id"], started, after, report)
            index += 1
        for hop in self.hops:
            previous = run
            request_id = "handoff-" + os.urandom(6).hex()
            called = time.monotonic()
            status, receipt = gateway.control("handoff_run", request_id=request_id, backend=hop["backend"],
                                              profile=hop["profile"], wire_api=hop["wire"], model=hop["model"],
                                              effort=EFFORT, prompt=turn_prompt(index))
            accepted = time.monotonic()
            self.check(gateway, status == 200 and receipt.get("status") == "handoff_accepted"
                       and receipt.get("successor_run_id"), f"handoff_run to {hop['backend']} + {hop['profile']} "
                       f"({hop['model']}) refused ({status}): {receipt}")
            successor = receipt["successor_run_id"]
            gateway.lease = None    # a writer lease belongs to one run: the successor starts without one
            snapshot = gateway.wait(lambda s, old=previous["run_id"]: s.get("run_id") != old
                                    and s.get("status") != "starting", STARTUP_TIMEOUT_S,
                                    f"successor run {successor} to start", interval=0.25)
            ready = time.monotonic()
            link = snapshot.get("handoff") or {}
            self.check(gateway, snapshot.get("run_id") == successor,
                       f"run {snapshot.get('run_id')} answers instead of the successor {successor} "
                       f"(fallback: {link.get('fallback')}):\n{gateway.tail()}")
            directory = next((root for _, root in reversed(gateway.runs()) if root.name == successor), None)
            self.assertIsNotNone(directory, f"no stdout line names the successor run {successor}")
            gateway.run_dir = directory
            self.check(gateway, snapshot["status"] == "running" and snapshot["agent_backend"] == hop["backend"]
                       and snapshot.get("turn_id") and link.get("predecessor_run_id") == previous["run_id"]
                       and link.get("request_id") == request_id and link.get("fallback") is None,
                       f"the successor did not start as requested: {snapshot}")
            polled = dict(handoff_run_s=round(accepted - called, 2), accepted_to_running_polled_s=round(
                ready - accepted, 2))
            run = dict(run_id=successor, directory=directory, backend=hop["backend"], profile=hop["profile"],
                       wire=hop["wire"], models=[hop["model"]], turns=[snapshot["turn_id"]], request_id=request_id,
                       polled=polled)
            runs.append(run)
            self.run_turn(gateway, index, hop["model"], snapshot["turn_id"], ready, 0, report,
                          before_checks=lambda backend=hop["backend"]: self.catalogue_check(gateway, backend))
            options = gateway.get("/v1/console/agent-options")
            current, last = options.get("current") or {}, (options.get("handoff") or {}).get("last") or {}
            self.check(gateway, (current.get("backend"), current.get("profile"), current.get("model"))
                       == (hop["backend"], hop["profile"], hop["model"]) and last.get("status") == "completed"
                       and last.get("successor_run_id") == successor,
                       f"agent-options of the successor: current {current}, handoff.last {last}")
            index += 1
        self.assertEqual(gateway.stop(), 0, gateway.tail())
        self.check_handoff_chain(runs, [first.path] + [hop["path"] for hop in self.hops])

    def check_handoff_chain(self, runs, profile_paths):
        """The archive of a finished handoff chain (module docstring), the timing lines, and the key scan."""
        from ni_rua.cortex.agents import profiles
        facts = []
        for run in runs:
            directory = run["directory"]
            summary = json.loads((directory / "summary.json").read_text())
            manifest = json.loads((directory / "manifest.json").read_text())
            with open_read_store(directory) as store:
                def times(kind):
                    return [event["occurred_at"] for event in store.events_by_type([kind], limit=10)]
                facts.append(dict(
                    summary=summary, manifest=manifest, agent=store.read_state("runtime", "agent") or {},
                    settings=[event["payload"] for event in store.events_by_type(["run.turn_settings"], limit=100)],
                    requested=[event["payload"] for event in store.events_by_type(["run.handoff_requested"], limit=10)],
                    requested_at=times("run.handoff_requested"), handoff_started_at=times("run.handoff_started"),
                    run_started_at=times("run.started")))

        def seconds(start, end):
            parse = lambda value: datetime.fromisoformat(value.replace("Z", "+00:00"))
            return round((parse(end) - parse(start)).total_seconds(), 2) if start and end else None

        chain = []
        for position, (run, fact) in enumerate(zip(runs, facts)):
            summary, manifest, settings = fact["summary"], fact["manifest"], fact["settings"]
            agent_backend = manifest.get("agent_backend") or {}
            row = dict(run_id=run["run_id"], backend=run["backend"], profile=run["profile"], models=run["models"],
                       status=summary.get("status"), cleanup_status=summary.get("cleanup_status"),
                       served_models=summary.get("served_models"), successor_run_id=summary.get("successor_run_id"),
                       predecessor_run_id=(manifest.get("handoff") or {}).get("predecessor_run_id"),
                       turn_settings=[{key: item.get(key) for key in ("origin", "backend", "profile", "wire_api",
                                                                      "model", "effort")} for item in settings])
            if position:
                accepted_at = (facts[position - 1]["requested_at"] or [None])[0]
                row["timing_s"] = dict(run["polled"], accepted_to_successor_built_s=seconds(
                    accepted_at, (fact["handoff_started_at"] or [None])[0]), accepted_to_successor_ready_s=seconds(
                    accepted_at, (fact["run_started_at"] or [None])[0]))
                print("LIVE_E2E_HANDOFF " + json.dumps(dict(
                    predecessor=runs[position - 1]["run_id"], successor=run["run_id"],
                    to=f"{run['backend']}+{run['profile']}:{run['models'][0]}", **row["timing_s"])), flush=True)
            chain.append(row)
            self.assertEqual((summary.get("status"), summary.get("cleanup_status")), ("stopped", "completed"),
                             (run["run_id"], summary.get("cleanup")))
            self.assertEqual((agent_backend.get("kind"), (agent_backend.get("upstream_profile") or {}).get("id")),
                             (run["backend"], run["profile"]), run["run_id"])
            self.assertEqual([(item.get("turn_id"), item.get("backend"), item.get("profile"), item.get("model"))
                              for item in settings],
                             [(turn, run["backend"], run["profile"], model)
                              for turn, model in zip(run["turns"], run["models"])], run["run_id"])
            if run["wire"]:
                self.assertEqual({item.get("wire_api") for item in settings}, {run["wire"]}, run["run_id"])
            served = summary.get("served_models") or {}
            self.assertTrue(set(served) <= set(run["models"]) and all(model in served for model in run["models"]),
                            f"run {run['run_id']} served {served}, its models are {run['models']}")
            # A model changed after the first turn makes the run a mixed condition (summary.agent_settings.mixed).
            self.assertIs((summary.get("agent_settings") or {}).get("mixed"), len(run["models"]) > 1, run["run_id"])
            if run["backend"] in TOOL_PREFIXES:
                self.assertIs(fact["agent"].get("clean_shutdown"), True, (run["run_id"], fact["agent"]))
            if position + 1 < len(runs):
                successor = runs[position + 1]
                self.assertEqual((summary.get("successor_run_id"), (summary.get("successor_handoff") or {}).get(
                    "status")), (successor["run_id"], "handed_off"), run["run_id"])
                self.assertEqual([(item.get("request_id"), item.get("successor_run_id")) for item in fact["requested"]],
                                 [(successor["request_id"], successor["run_id"])], run["run_id"])
            else:
                self.assertIsNone(summary.get("successor_run_id"), run["run_id"])
            if position:
                link = manifest.get("handoff") or {}
                self.assertEqual((link.get("predecessor_run_id"), link.get("request_id"), link.get("fallback")),
                                 (runs[position - 1]["run_id"], run["request_id"], None), run["run_id"])
                self.assertEqual(len(fact["handoff_started_at"]), 1, run["run_id"])
        print("LIVE_E2E_CHAIN " + json.dumps(chain), flush=True)
        # No key of any profile of the chain in any file (run directories, gateway logs, scene, archive). The keys are
        # read here only for the scan and never printed.
        leaks = []
        for path in dict.fromkeys(profile_paths):
            profile = profiles.load_profile(path)
            _base_url, headers = profiles.credential(profile, next(iter(profile.wires)))
            key = next(iter(headers.values())).removeprefix("Bearer ").encode()
            leaks += [f"{profile.id}: {item}" for item in self.root.rglob("*")
                      if item.is_file() and not item.is_symlink() and key in item.read_bytes()]
            del key, headers
        self.assertEqual(leaks, [], "a profile key was found in these files")

if __name__ == "__main__":
    unittest.main()
