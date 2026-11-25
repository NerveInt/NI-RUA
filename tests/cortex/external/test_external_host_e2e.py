"""Opt-in host end-to-end conformance of the external agent backend (design 20 section 5, "Per-host end-to-end").

For each host binary (``support.host_e2e.TARGETS``: Claude Code 2.1.321 and the user's Claude Code, Codex 0.160.0
and the user's Codex, OpenCode 2.2.7):

1. Setup. A fresh ``mktemp -d`` home with only the host's config-dir variables; the plugin installed by the real
   installer (``node agent-plugin/bin/cli.mjs install <host> --port <port>``) and nothing else; the host in its default
   permission / approval mode. Only the loopback fake model is configured besides: ``ANTHROPIC_BASE_URL`` (Claude
   Code), ``-c`` overrides on the ``codex exec`` command line (Codex), a ``fake`` provider in the global
   ``opencode.json`` before the installer runs (OpenCode).
2. Gateway. The real gateway process ``--agent-backend external`` over the SO-101 scene double on that port. No real
   arm, ever.
3. Model. The host runs headless (``claude -p`` stream-json, ``codex exec --json``, ``opencode serve`` API) against a
   scripted fake model that replays ``run_begin`` -> ``task_plan`` -> ``robot_observe`` (two 640x480 camera frames) ->
   one long ``robot_action`` -> two parallel reads in one assistant message -> ``run_finish`` -> final text.
4. Assertions against the captured model requests and the gateway's MCP raw log, HTTP header log, Store and
   ``summary.json``: full catalogue (Codex: only its known schema projection), description and text-result lengths,
   ``initialize.instructions``, image bytes, non-read-only calls in the default mode, the long call, the ``Origin``
   header, ``tool_call`` turn ids, the ``run_finish`` final answer, ``agent.client_attached`` and the summary's null
   fields after shutdown.
5. ``doctor`` against the live gateway (pinned binaries).
6. Operator wake (Claude Code 2.1.321, OpenCode 2.2.7): the model starts ``sleep 10`` as a background shell task with
   the host's own tool and ends the turn; the host must send a new model request by itself once it exited.

The user binaries run items 1-4 without the long call. One ``HOST_E2E {...}`` line per host reports every result and
the wall time.

Environment (the module is skipped, with the reason, unless the first one is set):

* ``NI_RUA_HOST_E2E=1``: run it (several minutes; starts real host binaries, never a real model provider or arm).
* ``NI_RUA_HOST_E2E_LONG=1``: the long call lasts 330 s on Claude Code 2.1.321 (past its 300 s MCP idle timer) and
  90 s on Codex 0.160.0 and OpenCode 2.2.7 (the scene's test-only ``execute_hold_s``); otherwise it is a normal short
  call.
* ``NI_RUA_HOST_E2E_HOSTS``: comma-separated subset of the target keys (default all five).
* ``NI_RUA_HOST_E2E_ROOT``: keep every home, gateway run and model request under this directory (default: a
  temporary directory, removed).
* ``NI_RUA_HOST_E2E_CLAUDE_PINNED``, ``..._CLAUDE_USER``, ``..._CODEX_PINNED``, ``..._CODEX_USER``,
  ``NI_RUA_HOST_E2E_OPENCODE``: binary paths (defaults: the NI-RUA environment's bundled Claude Code and Codex,
  ``~/.local/bin/claude``, ``/opt/homebrew/bin/codex``, ``~/.local/share/ni-rua/opencode-2.2.7/opencode``). A missing
  binary skips that host with the reason.

Run from the checkout root::

    NI_RUA_HOST_E2E=1 [NI_RUA_HOST_E2E_LONG=1] PYTHONPATH=tests mamba run -n ni-rua python -m unittest -v \
        test_agent_external_host_e2e

Live mode (``ExternalHostLiveTest``; a real, charged model provider; never in the normal suite). Same setup per pinned
host (isolated home, the real installer, the real gateway over the scene double, the default permission mode with only
the installer's settings), but the host talks to the provider of a provider profile instead of the scripted model, and
nothing is scripted. The gateway's ``--prompt`` is ``host_e2e.LIVE_TASK`` (name the cube colour as a ``CUBE_COLOUR:``
line, move the end effector 5 mm up and confirm it from the receipt, finish with ``run_finish``); the host gets only
``host_e2e.LIVE_USER_MESSAGE`` ("Operate the robot through the ni_rua MCP tools."), which names neither ``run_begin``
nor the task, so the run checks that the model follows ``initialize.instructions`` and the tool descriptions by itself.
The key goes only to the host process: ``ANTHROPIC_API_KEY`` (Claude Code, with ``ANTHROPIC_BASE_URL``, ``--model``
and the same model for every Claude Code model role), the ``env_key`` variable of a ``-c`` provider entry (Codex), a
provider entry in the isolated home's mode-0600 ``opencode.json`` (OpenCode); never to the installer or the gateway.
Checks, from the gateway's MCP raw log and Store and the host's own transcript (Claude Code stream-json, ``codex exec
--json`` and its session rollout, the OpenCode session timeline): the first NI-RUA call is ``run_begin`` (calls before
it and ``external_briefing_required`` refusals reported); ``robot_observe`` returned images; a ``robot_action``
returned a typed ``action`` result with a ``completed`` receipt (measured end-effector dz reported); ``run_finish`` was
served, its summary (short ids resolved, as the archive keeps full ids) is the turn's final answer and that turn closed
``completed``; the first ``CUBE_COLOUR:`` line (the summary, else the host's texts) names the rendered colour; the host
ended the session by itself; a clean gateway shutdown; the key is in no gateway file and, with
``NI_RUA_HOST_E2E_ROOT``, the kept evidence is scrubbed of it (same-length overwrite) and re-scanned. One
``HOST_E2E_LIVE {...}`` line per host records the model's own tool-call sequence, the request count, the host's token
usage, the served model id (Claude Code only; Codex and OpenCode record none), refusals, permission denials, errors and
the wall time. A model that does the task imperfectly fails a check; that is a result, not a test bug.

* ``NI_RUA_HOST_E2E_LIVE=1``: run it (with the next variables).
* ``NI_RUA_HOST_E2E_LIVE_PROFILE``: a provider profile id in ``configs/profiles/`` or the path of a
  ``<id>.local.json``; Claude Code needs its ``anthropic_messages`` wire, Codex its ``openai_responses`` wire.
* ``NI_RUA_HOST_E2E_LIVE_CLAUDE_MODEL``, ``..._CODEX_MODEL``, ``..._OPENCODE_MODEL``: the provider's model id per host;
  a host without one is skipped with the reason. ``NI_RUA_HOST_E2E_LIVE_OPENCODE_WIRE``: ``anthropic_messages``
  (default) or ``openai_responses``.
* ``NI_RUA_HOST_E2E_LIVE_TIMEOUT_S`` (default 600 s) and ``NI_RUA_HOST_E2E_LIVE_MAX_REQUESTS`` (default 40 model
  responses): the host is killed past either, and the session-end check fails.
* ``NI_RUA_HOST_E2E_LIVE_CUBE_COLOUR``: one of ``support.so101_scene.COLOURS`` (default random per host).
* ``NI_RUA_HOST_E2E_HOSTS`` and ``NI_RUA_HOST_E2E_ROOT`` as above (only the three pinned keys have a live test).
"""

NI_RUA_TEST_TIER = "component"

import io
import json
import os
import random
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from support import host_e2e, so101_scene
from support.host_e2e import (CODEX_PLUGIN_SUFFIX, HOST_CLASSES, PLUGIN_CLI, SEQUENCE, SUMMARY, TARGETS, TASK,
                              ConformanceModel, ExternalGateway, HostHome, codex_projection_problems, dropped_keywords,
                              free_port, mcp_exchanges, node_executable, request_view, result_parts, strings_in,
                              wait_for)

ENABLED = os.environ.get("NI_RUA_HOST_E2E") == "1"
LONG = os.environ.get("NI_RUA_HOST_E2E_LONG") == "1"
KEEP_ROOT = os.environ.get("NI_RUA_HOST_E2E_ROOT")
SELECTED = [key.strip() for key in (os.environ.get("NI_RUA_HOST_E2E_HOSTS") or ",".join(TARGETS)).split(",")
            if key.strip()]
# Self-reported MCP clientInfo.name of each host (spike 21: initialize).
CLIENT_NAMES = {"claude": "claude-code", "codex": "codex-mcp-client", "opencode": "opencode"}
# (step, call) of every scripted NI-RUA call in host_e2e.conformance_script, in SEQUENCE order.
SCRIPT_CALLS = dict(zip(SEQUENCE, [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1), (5, 0)]))
PARALLEL_READS = ("robot_capabilities", "robot_information")       # step 4: one assistant message, two calls
CAMERA_SIZE = (640, 480)
CLAUDE_IDLE_TIMEOUT_S = 300.0


# Live mode (module docstring, "Live mode").
LIVE = os.environ.get("NI_RUA_HOST_E2E_LIVE") == "1"
LIVE_PROFILE = os.environ.get("NI_RUA_HOST_E2E_LIVE_PROFILE")
LIVE_MODEL_VARIABLES = {"claude": "NI_RUA_HOST_E2E_LIVE_CLAUDE_MODEL", "codex": "NI_RUA_HOST_E2E_LIVE_CODEX_MODEL",
                        "opencode": "NI_RUA_HOST_E2E_LIVE_OPENCODE_MODEL"}
LIVE_OPENCODE_WIRE = os.environ.get("NI_RUA_HOST_E2E_LIVE_OPENCODE_WIRE") or None
LIVE_TIMEOUT_S = float(os.environ.get("NI_RUA_HOST_E2E_LIVE_TIMEOUT_S") or 600)
LIVE_MAX_REQUESTS = int(os.environ.get("NI_RUA_HOST_E2E_LIVE_MAX_REQUESTS") or 40)
LIVE_COLOUR = os.environ.get("NI_RUA_HOST_E2E_LIVE_CUBE_COLOUR") or None


def _setup_problem():
    if node_executable() is None:
        return "node is not on PATH (the plugin installer is a Node script)"
    if not PLUGIN_CLI.is_file():
        return f"{PLUGIN_CLI} is missing"
    unknown = [key for key in SELECTED if key not in TARGETS]
    if unknown:
        return f"NI_RUA_HOST_E2E_HOSTS names unknown targets {unknown}; known: {', '.join(TARGETS)}"
    return None


def _problem():
    if not ENABLED:
        return ("set NI_RUA_HOST_E2E=1 to run the host end-to-end conformance of the external backend (starts the "
                "real Claude Code / Codex / OpenCode binaries in isolated homes against a scripted loopback model)")
    return _setup_problem()


def _live_problem():
    if not LIVE:
        return ("set NI_RUA_HOST_E2E_LIVE=1 with NI_RUA_HOST_E2E_LIVE_PROFILE and the per-host model variables to run "
                "one real-model session per pinned host against the external backend (calls a real, charged model "
                "provider)")
    if not LIVE_PROFILE:
        return ("NI_RUA_HOST_E2E_LIVE_PROFILE is not set (a provider profile id in configs/profiles/ or the path of a "
                "<id>.local.json file)")
    path = host_e2e.live_profile_path(LIVE_PROFILE)
    if not path.is_file():
        return f"NI_RUA_HOST_E2E_LIVE_PROFILE: no provider profile file {path}"
    if LIVE_COLOUR and LIVE_COLOUR not in so101_scene.COLOURS:
        return f"NI_RUA_HOST_E2E_LIVE_CUBE_COLOUR must be one of {', '.join(sorted(so101_scene.COLOURS))}"
    if LIVE_OPENCODE_WIRE and LIVE_OPENCODE_WIRE not in host_e2e.LIVE_WIRES["opencode"]:
        return f"NI_RUA_HOST_E2E_LIVE_OPENCODE_WIRE must be one of {', '.join(host_e2e.LIVE_WIRES['opencode'])}"
    return _setup_problem()


PROBLEM = _problem()
LIVE_PROBLEM = _live_problem()


def image_size(data):
    from PIL import Image as PilImage
    with PilImage.open(io.BytesIO(data)) as image:
        return image.size


class HostRunMixin:
    """Evidence root and per-column checks shared by the conformance and the live classes (``self.row``: the report
    row of the current host)."""

    def make_root(self, key):
        if KEEP_ROOT:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            root = Path(KEEP_ROOT).expanduser().resolve() / f"{stamp}-{key}"
            root.mkdir(parents=True)
            return root
        directory = tempfile.TemporaryDirectory(prefix=f"ni-rua-host-e2e-{key}-")
        self.addCleanup(directory.cleanup)
        return Path(directory.name).resolve()

    def check(self, column, function):
        """Run one check as a subTest; the report row's ``column`` gets its finding or ``FAIL: <first line>``."""
        with self.subTest(check=column):
            try:
                value = function()
            except Exception as error:
                self.row[column] = "FAIL: " + (str(error).splitlines() or [type(error).__name__])[0][:300]
                raise
            self.row[column] = value or "ok"


@unittest.skipIf(PROBLEM is not None, PROBLEM)
class ExternalHostConformanceTest(HostRunMixin, unittest.TestCase):
    maxDiff = None
    rows = []

    @classmethod
    def tearDownClass(cls):
        for row in cls.rows:
            print("HOST_E2E " + json.dumps(row), flush=True)

    def test_claude_pinned(self):
        self.conformance("claude-pinned")

    def test_claude_user(self):
        self.conformance("claude-user")

    def test_codex_pinned(self):
        self.conformance("codex-pinned")

    def test_codex_user(self):
        self.conformance("codex-user")

    def test_opencode_pinned(self):
        self.conformance("opencode-pinned")

    # ---- one host ----------------------------------------------------------------------------------------------
    def conformance(self, key):
        if key not in SELECTED:
            self.skipTest(f"{key} not in NI_RUA_HOST_E2E_HOSTS")
        target = TARGETS[key]
        if target.path is None:
            self.skipTest(f"{key}: no executable at {os.environ.get(target.variable) or target.default} "
                          f"(set {target.variable})")
        started = time.monotonic()
        version = target.version()
        root = self.make_root(key)
        hold = target.long_call_s if LONG and target.long_call_s else 0.0
        self.row = row = dict(host=key, version=version, named_version=target.named_version, long_call_s=hold or None)
        self.rows.append(row)
        print(f"\nHOST_E2E {key}: {target.path} ({version}); evidence in {root}", flush=True)

        host = HOST_CLASSES[target.host](target, HostHome(root, target.host), root / "host")
        model = ConformanceModel(host).start()
        self.addCleanup(model.close)
        host.attach(model)
        model.add("main", host_e2e.conformance_script())
        if target.wake:
            model.add("wake", host_e2e.wake_script(host))
        port = free_port()
        code, installed = host.install(port)
        (root / "install.log").write_text(installed)
        self.assertEqual(code, 0, f"installer failed:\n{installed}")

        gateway = ExternalGateway(root / "gateway", port, prompt=TASK, execute_hold_s=hold)
        self.addCleanup(gateway.kill)
        gateway.start()
        self.addCleanup(host.kill)
        host.open()
        timeout = hold + 600.0
        try:
            outcome = host.turn(model.prompt("main", "Run the NI-RUA host conformance sequence."), timeout,
                                done=model.done["main"])
        except AssertionError as error:
            raise AssertionError(f"{error}\nmodel answers: {json.dumps(model.served)[-3000:]}\n{gateway.tail()}")
        wake = self.operator_wake(target, host, model) if target.wake else None
        doctor_at = time.time()
        doctor = host.home.doctor(target.path, port, env=host.env()) if target.pinned else None
        host.close()
        returncode = gateway.stop()
        row["wall_time_s"] = round(time.monotonic() - started, 1)
        if KEEP_ROOT:
            self.keep_evidence(root, model)
        self.check_all(target, host, model, gateway, outcome=outcome, wake=wake, doctor=doctor, doctor_at=doctor_at,
                       gateway_returncode=returncode, hold=hold)

    @staticmethod
    def keep_evidence(root, model):
        with (root / "model-requests.jsonl").open("w") as stream:
            for request in model.requests():
                stream.write(json.dumps(dict(t=request.received_at, path=request.path,
                                             headers=dict(request.headers.items()), body=request.json)) + "\n")
        (root / "model-served.json").write_text(json.dumps(model.served, indent=2))

    def operator_wake(self, target, host, model):
        """Turn 2: a background ``sleep`` with the host's own shell tool, then the turn ends; returns ``{"started",
        "ended", "notified"}`` (POSIX s: the model's shell call was served, the turn ended, the host's own new model
        request arrived) or ``{"error"}``."""
        try:
            host.turn(model.prompt("wake", "Start the background wait, then end this turn."), 300,
                      done=model.done["wake"])
            ended = time.time()
            results = len(getattr(host, "results", []))

            def notified():
                for request in model.requests():
                    if request.received_at > ended and host.wake_marker in request_view(
                            model.wire, request.json).last_user_text:
                        return request
                return None
            request = wait_for(notified, WAKE_WAIT_S, "the host's own model request after the background task")
            if isinstance(host, host_e2e.ClaudeHost):
                host.wait_results(results + 1, 120)       # the turn the host started by itself
            else:
                wait_for(host.idle, 120, "the OpenCode session to become idle after the wake")
            started = next(entry["t"] for entry in model.served if entry["script"] == "wake"
                           and entry["kind"] == "tools")
            return dict(started=started, ended=ended, notified=request.received_at)
        except AssertionError as error:
            return dict(error=str(error))

    # ---- checks ------------------------------------------------------------------------------------------------
    def check_all(self, target, host, model, gateway, *, outcome, wake, doctor, doctor_at, gateway_returncode, hold):
        views = model.views()
        exchanges = mcp_exchanges(gateway.mcp_records())
        served = model.served
        main_views = [(request, view) for request, view in views if host_e2e.script_of(view) == "main"]
        self.assertTrue(main_views, f"no model request of the conformance script; served: {served}")
        first_request, first = main_views[0]
        final = main_views[-1][1]
        prefix = "toolu" if model.wire == "anthropic" else "call"
        sequence_done = all(f"{prefix}_e2e_main_{step}_{k}" in final.results for step, k in SCRIPT_CALLS.values())
        self.assertTrue(sequence_done, "the model did not receive every result of the sequence; served: "
                        + json.dumps([entry for entry in served if entry.get("script") == "main"])
                        + f"\ngateway:\n{gateway.tail()}")
        published = {tool["name"]: tool for tool in next(
            exchange.result["tools"] for exchange in exchanges if exchange.method == "tools/list" and exchange.result)}
        initialize = next(exchange for exchange in exchanges if exchange.method == "initialize" and exchange.result)
        calls = {}
        for name in SEQUENCE:
            found = [exchange for exchange in exchanges if exchange.tool == name]
            calls[name] = found[0] if len(found) == 1 else None

        self.check("tools", lambda: self.check_tools(target, host, published, main_views))
        self.check("lengths", lambda: self.check_texts(host, initialize, first_request, first, final, calls, prefix))
        self.check("image", lambda: self.check_images(final, calls, prefix))
        self.check("permissions", lambda: self.check_permissions(published, calls, final, prefix))
        self.check("long call", lambda: self.check_long_call(target, calls, final, prefix, hold))
        self.check("Origin", lambda: self.check_origin(gateway, doctor_at))
        self.check("Store", lambda: self.check_store(target, gateway, gateway_returncode, self.row["version"]))
        if doctor is not None:
            self.check("doctor", lambda: self.check_doctor(target, doctor, gateway.mcp_port, len(published)))
        else:
            self.row["doctor"] = "n/a (user version: items 1-4)"
        if wake is not None:
            self.check("wake", lambda: self.check_wake(wake))
        else:
            self.row["wake"] = "n/a"
        self.check("turn", lambda: self.check_turn(target, outcome))

    def check_turn(self, target, outcome):
        """The host ended the conformance turn normally: Claude Code's ``result`` is ``success``, ``codex exec``
        exited 0 (OpenCode: the session went idle after the final text, checked while driving it)."""
        if target.host == "claude":
            self.assertEqual((outcome.get("subtype"), outcome.get("is_error")), ("success", False), outcome)
            return "result success"
        if target.host == "codex":
            self.assertEqual(outcome, 0, "codex exec exit code")
            return "codex exec exit 0"
        return "session idle after the final text"

    def check_tools(self, target, host, published, main_views):
        """Every published tool with its full schema in every main request; descriptions as the gateway sent them."""
        details = []
        for request, view in main_views:
            if target.host == "codex":
                namespace = view.namespaces.get(host.namespace)
                self.assertIsNotNone(namespace, f"no {host.namespace} namespace tool in the request")
                delivered = {tool["name"]: tool for tool in namespace.get("tools") or []}
            else:
                delivered = {name[len(host.prefix):]: tool for name, tool in view.tools.items()
                             if name.startswith(host.prefix)}
            self.assertEqual(sorted(delivered), sorted(published), "tools dropped or added")
            for name, tool in published.items():
                got = delivered[name]
                self.assertFalse(got.get("defer_loading"), f"{name} deferred")
                if target.host == "codex":
                    self.assertEqual(got["description"], tool["description"] + CODEX_PLUGIN_SUFFIX, name)
                    self.assertEqual(codex_projection_problems(tool["inputSchema"], got["parameters"]), [], name)
                else:
                    self.assertEqual(got["description"], tool["description"], name)
                    self.assertEqual(got["input_schema"], tool["inputSchema"], name)
        if target.host == "codex":
            first = main_views[0][1].namespaces[host.namespace]["tools"]
            schema = sum(len(json.dumps(published[t["name"]]["inputSchema"])) for t in first)
            kept = sum(len(json.dumps(t["parameters"])) for t in first)
            counts = {}
            for tool in first:
                dropped_keywords(published[tool["name"]]["inputSchema"], tool["parameters"], counts)
            details.append(f"Codex schema projection {schema} -> {kept} chars; dropped "
                           + ", ".join(f"{k} x{v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:6]))
        return f"{len(published)}/{len(published)} tools in {len(main_views)} requests" + (
            "; " + "; ".join(details) if details else "")

    def check_texts(self, host, initialize, first_request, first, final, calls, prefix):
        """``initialize.instructions`` in the first request; every text item of every scripted result whole."""
        instructions = initialize.result.get("instructions")
        self.assertTrue(instructions, "the gateway sent no initialize.instructions")
        if host.wire == "responses":
            self.assertEqual(first.namespaces[host.namespace].get("description"), instructions)
        else:
            self.assertTrue(any(instructions in text for text in strings_in(first_request.json)),
                            "initialize.instructions missing from the first model request")
        longest = (0, None)
        for name, (step, k) in SCRIPT_CALLS.items():
            exchange = calls[name]
            self.assertIsNotNone(exchange, f"{name}: not exactly one tools/call at the gateway")
            sent, _ = result_parts(exchange.result)
            got = final.results[f"{prefix}_e2e_main_{step}_{k}"]
            for index, text in enumerate(sent):
                self.assertIn(text, got.texts, f"{name} text item {index} ({len(text)} chars) not delivered whole; "
                              f"delivered lengths {[len(t) for t in got.texts]}")
                longest = max(longest, (len(text), name))
        briefing = result_parts(calls["run_begin"].result)[0][0]
        return f"instructions present; briefing {len(briefing)} chars, longest result {longest[1]} {longest[0]} chars"

    def check_images(self, final, calls, prefix):
        """Image bytes of every scripted result identical; the observation carries camera-sized frames."""
        count = 0
        for name, (step, k) in SCRIPT_CALLS.items():
            _, sent = result_parts(calls[name].result)
            got = final.results[f"{prefix}_e2e_main_{step}_{k}"].images
            self.assertEqual([image.sha256 for image in got], [image.sha256 for image in sent], name)
            count += len(sent)
        _, observed = result_parts(calls["robot_observe"].result)
        sizes = [image_size(image.data) for image in observed]
        self.assertTrue(observed and all(size == CAMERA_SIZE for size in sizes), sizes)
        return (f"{count} images byte-identical; robot_observe "
                + ", ".join(f"{image.media_type} {size[0]}x{size[1]} {len(image.data)} B"
                            for image, size in zip(observed, sizes)))

    def check_permissions(self, published, calls, final, prefix):
        """Every non-read-only scripted call reached the gateway and was served (default permission mode)."""
        reached = []
        for name, (step, k) in SCRIPT_CALLS.items():
            if published[name]["annotations"].get("readOnlyHint"):
                continue
            exchange = calls[name]
            self.assertIsNotNone(exchange, f"{name} never reached the gateway")
            self.assertFalse(exchange.result.get("isError"), f"{name}: {exchange.result}")
            self.assertFalse(final.results[f"{prefix}_e2e_main_{step}_{k}"].is_error, name)
            reached.append(name)
        return "served: " + ", ".join(reached)

    def check_long_call(self, target, calls, final, prefix, hold):
        exchange = calls["robot_action"]
        step, k = SCRIPT_CALLS["robot_action"]
        texts = final.results[f"{prefix}_e2e_main_{step}_{k}"].texts
        self.assertFalse(exchange.result.get("isError"))
        receipt = json.loads(texts[[i for i, text in enumerate(texts) if '"receipt"' in text][0]])["receipt"]
        self.assertEqual(receipt.get("status"), "completed",
                         f"receipt {receipt.get('status')}: {receipt.get('reason')}")
        if not hold:
            return f"n/a (short call {exchange.duration_s:.1f} s)"
        self.assertGreaterEqual(exchange.duration_s, hold)
        if target.key == "claude-pinned":
            self.assertGreater(exchange.duration_s, CLAUDE_IDLE_TIMEOUT_S)
        return f"{exchange.duration_s:.0f} s robot_action survived, completed receipt delivered"

    def check_origin(self, gateway, doctor_at):
        """Report every ``Origin`` the host sent; no host request was refused (403)."""
        records = [record for record in gateway.http_records() if record["t"] < doctor_at]
        self.assertTrue(records, "no MCP HTTP request in the header log")
        origins = sorted({next((v for k, v in record["headers"].items() if k.lower() == "origin"), "absent")
                          for record in records})
        refused = [record for record in records if record["status"] == 403]
        agents = sorted({next((v for k, v in record["headers"].items() if k.lower() == "user-agent"), "absent")
                         for record in records})
        self.assertEqual(refused, [], "the Origin / Host check refused host requests")
        return f"Origin {', '.join(origins)} on {len(records)} requests (User-Agent {', '.join(agents)}); no 403"

    def check_store(self, target, gateway, returncode, version):
        self.assertEqual(returncode, 0, gateway.tail())

        def collect(store):
            records = store.records("tool_call")
            rows = store.states(store.TRANSCRIPT_KIND)
            attached = [event["payload"] for event in store.events_by_type(["agent.client_attached"], limit=100)]
            finals = {}
            for row in rows:
                if row.get("origin") == "run_finish_summary":
                    finals[row["id"]] = store.completed_final_texts(row["thread_id"], row["turn_id"])
            return records, rows, attached, finals
        records, rows, attached, finals = gateway.read_store(collect)
        tools = [record["params"]["tool"] for record in records if record["params"]["tool"] in SEQUENCE]
        # the two reads of one assistant message may run in either order (a host can call them in parallel)
        self.assertEqual([PARALLEL_READS if tool in PARALLEL_READS else tool for tool in tools],
                         [PARALLEL_READS if tool in PARALLEL_READS else tool for tool in SEQUENCE], tools)
        self.assertEqual(sorted(tools), sorted(SEQUENCE))
        self.assertEqual([record["params"]["tool"] for record in records if not record["correlation"].get("turn_id")],
                         [], "tool_call rows without a turn id")
        summaries = [row for row in rows if row.get("origin") == "run_finish_summary"]
        self.assertEqual([(row["text"], row["phase"], row["status"]) for row in summaries],
                         [(SUMMARY, "final_answer", "completed")])
        self.assertEqual(list(finals.values()), [[SUMMARY]])
        clients = [payload["client_info"] for payload in attached]
        mine = [client for client in clients if client.get("name") == CLIENT_NAMES[target.host]]
        self.assertTrue(mine, f"no agent.client_attached from {CLIENT_NAMES[target.host]}: {clients}")
        self.assertEqual({client.get("version") for client in mine}, {version}, clients)
        self.assertTrue(all(client.get("self_reported") is True for client in clients), clients)
        summary = gateway.summary()
        self.assertIsNotNone(summary, "no summary.json after shutdown")
        from ni_rua.cortex.agents.external.session import SUMMARY_NULL_FIELDS
        self.assertEqual({key: summary.get(key, "<missing>") for key in SUMMARY_NULL_FIELDS},
                         dict.fromkeys(SUMMARY_NULL_FIELDS))
        self.assertEqual((summary["status"], summary["agent_backend"], summary["agent_turn_status"], summary["cleanup_status"]),
                         ("completed", "external", "completed", "completed"))
        client = mine[-1]
        return (f"{len(records)} tool_call rows with turn ids; final answer = run_finish summary (origin "
                f"run_finish_summary); clientInfo {client['name']} {client['version']} (self-reported); summary null "
                f"fields {', '.join(SUMMARY_NULL_FIELDS)}")

    def check_doctor(self, target, doctor, port, count):
        code, output = doctor
        expected = f"[ok] gateway on 127.0.0.1:{port}: MCP initialize + tools/list answered, {count} tools, " \
                   "instructions present"
        self.assertIn(expected, output)
        self.assertIn("[ok] gateway offers run_begin (external backend)", output)
        listed = {"claude": "[ok] claude: ni_rua@ni_rua listed", "codex": "[ok] codex: ni_rua@ni_rua listed",
                  "opencode": "[ok] opencode: config "}[target.host]
        self.assertIn(listed, output)
        self.assertEqual(code, 0, output)
        return f"connected, {count} tools, instructions present"

    def check_wake(self, wake):
        self.assertNotIn("error", wake, wake.get("error"))
        self.assertGreaterEqual(wake["notified"] - wake["started"], host_e2e.WAKE_SLEEP_S)
        return (f"new model request by itself {wake['notified'] - wake['ended']:.1f} s after the turn ended, "
                f"{wake['notified'] - wake['started']:.1f} s after the background sleep {host_e2e.WAKE_SLEEP_S:g} "
                "was started")


WAKE_WAIT_S = 120.0


@unittest.skipIf(LIVE_PROBLEM is not None, LIVE_PROBLEM)
class ExternalHostLiveTest(HostRunMixin, unittest.TestCase):
    """One real-model session per pinned host (module docstring, "Live mode"). A model that does the task imperfectly
    fails a check: that is a finding about the model and the contract wording, reported in the ``HOST_E2E_LIVE``
    row."""
    maxDiff = None
    rows = []

    @classmethod
    def tearDownClass(cls):
        for row in cls.rows:
            print("HOST_E2E_LIVE " + json.dumps(row), flush=True)

    def test_live_claude_pinned(self):
        self.live("claude-pinned")

    def test_live_codex_pinned(self):
        self.live("codex-pinned")

    def test_live_opencode_pinned(self):
        self.live("opencode-pinned")

    def live(self, key):
        if key not in SELECTED:
            self.skipTest(f"{key} not in NI_RUA_HOST_E2E_HOSTS")
        target = TARGETS[key]
        if target.path is None:
            self.skipTest(f"{key}: no executable at {os.environ.get(target.variable) or target.default} "
                          f"(set {target.variable})")
        variable = LIVE_MODEL_VARIABLES[target.host]
        model_id = os.environ.get(variable)
        if not model_id:
            self.skipTest(f"{key}: set {variable} to a model id the provider serves")
        upstream = host_e2e.live_upstream(host_e2e.live_profile_path(LIVE_PROFILE), target.host, model_id,
                                          LIVE_OPENCODE_WIRE if target.host == "opencode" else None)
        secrets = upstream.secrets()
        colour = LIVE_COLOUR or random.SystemRandom().choice(sorted(so101_scene.COLOURS))
        started = time.monotonic()
        version = target.version()
        root = self.make_root(f"live-{key}")
        if KEEP_ROOT:
            # Registered first, so it runs last: after the host and the gateway were killed, also when a step failed.
            self.addCleanup(host_e2e.scrub_files, root, secrets)
        self.row = row = dict(host=key, version=version, model=model_id, wire=upstream.wire, cube_colour=colour,
                              started_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        self.rows.append(row)
        print(f"\nHOST_E2E_LIVE {key}: {target.path} ({version}), model {model_id} on {upstream.wire}; cube {colour}; "
              f"evidence in {root}", flush=True)

        host = HOST_CLASSES[target.host](target, HostHome(root, target.host), root / "host")
        host.use_upstream(upstream)
        port = free_port()
        code, installed = host.install(port)
        (root / "install.log").write_text(installed)
        self.assertEqual(code, 0, f"installer failed:\n{installed}")
        if target.host == "opencode":
            self.assertEqual(host.config_path.stat().st_mode & 0o777, 0o600, "opencode.json with the key is not 0600")
        gateway = ExternalGateway(root / "gateway", port, prompt=host_e2e.LIVE_TASK, colour=colour)
        self.addCleanup(gateway.kill)
        gateway.start()
        self.addCleanup(host.kill)
        host.open()
        turn_started = time.monotonic()
        outcome = host.live_turn(host_e2e.LIVE_USER_MESSAGE, LIVE_TIMEOUT_S, LIVE_MAX_REQUESTS)
        row["session_s"] = round(time.monotonic() - turn_started, 1)
        facts = host.live_facts()
        host.close()
        returncode = gateway.stop()
        row["wall_time_s"] = round(time.monotonic() - started, 1)
        row["session_end"] = {name: value for name, value in outcome.items() if name != "result"}
        row.update(host_calls=[call["tool"] + (" (error)" if call["error"] else "") for call in facts["calls"]],
                   requests=facts["requests"], usage=facts["usage"], configured_model=facts["configured_model"],
                   served_models=facts["served_models"], permission_denials=facts["permission_denials"],
                   host_errors=facts["errors"], **{name: facts[name] for name in ("permission_mode", "approval_policy",
                                                                                  "sandbox") if name in facts})
        self.check_live(target, gateway, returncode, facts, outcome, colour, root, secrets)

    # ---- checks ------------------------------------------------------------------------------------------------
    def check_live(self, target, gateway, returncode, facts, outcome, colour, root, secrets):
        exchanges = [exchange for exchange in mcp_exchanges(gateway.mcp_records()) if exchange.method == "tools/call"]
        store = gateway.read_store(self.store_facts)
        self.row["ni_rua_calls"] = [exchange.tool + (" (error)" if (exchange.result or {}).get("isError") else "")
                                    for exchange in exchanges]
        self.row["refusals"] = store["refusals"]
        finished = [exchange for exchange in exchanges if exchange.tool == "run_finish" and exchange.result
                    and not exchange.result.get("isError")]
        summary_text = ((finished[0].request.get("params") or {}).get("arguments") or {}).get("summary") \
            if finished else None
        self.check("run_begin first", lambda: self.check_first(exchanges, store, facts))
        self.check("observe images", lambda: self.check_observe(exchanges))
        self.check("action completed", lambda: self.check_action(store))
        self.check("run_finish", lambda: self.check_finish(exchanges, store, summary_text, gateway))
        self.check("colour", lambda: self.check_colour(summary_text, facts, colour))
        self.check("session end", lambda: self.check_end(target, outcome))
        self.check("gateway", lambda: self.check_gateway(gateway, returncode))
        self.check("key", lambda: self.check_key(root, gateway, secrets))

    @staticmethod
    def store_facts(store):
        """Refusals, robot_action receipts (with the measured end-effector dz), run_finish transcript rows, the final
        texts of their turns and every turn's closing status, read from the sealed Store."""
        refusals = [dict(code=event["payload"].get("code"), tool=event["payload"].get("tool"))
                    for event in store.events_by_type(["agent.tool_refused"], limit=1000)]
        actions = []
        for record in store.records("tool_result"):
            if record.get("tool") != "robot_action":
                continue
            payload = host_e2e.tool_payload(store, record)
            receipt = (payload.get("data") or {}).get("receipt") or {}
            dz = None
            try:
                before = store.read("action_request", payload["data"]["action_id"])["based_on_observation_id"]
                dz = round(host_e2e.ee_z(store, receipt["result_observation_id"]) - host_e2e.ee_z(store, before), 4)
            except (KeyError, TypeError, ValueError):
                pass
            actions.append(dict(kind=payload.get("kind"), status=receipt.get("status"),
                                batch_status=receipt.get("batch_status"), measured_dz_m=dz))
        rows = [row for row in store.states(store.TRANSCRIPT_KIND) if row.get("origin") == "run_finish_summary"]
        finals = {row["turn_id"]: store.completed_final_texts(row["thread_id"], row["turn_id"]) for row in rows}
        turns = {(event.get("correlation") or {}).get("turn_id"): event["payload"].get("status")
                 for event in store.events_by_type(["agent.turn_completed"], limit=1000)}
        # The archive keeps full ids: the summary's short ids (obs_4) are resolved before run_finish records it, by the
        # presentation's own restore (read-only here).
        from ni_rua.core.presentation.actor_presentation import ActorPresentation
        presentation = ActorPresentation(store)
        finishes = []
        for record in store.records("tool_call"):
            if record["params"]["tool"] == "run_finish":
                argument = (record["params"].get("arguments") or {}).get("summary")
                finishes.append(dict(argument=argument, restored=presentation.restore(argument),
                                     turn_id=record["correlation"].get("turn_id")))
        return dict(refusals=refusals, actions=actions, rows=rows, finals=finals, turns=turns, finishes=finishes)

    def check_first(self, exchanges, store, facts):
        """The model's first NI-RUA call is run_begin (it was told only to use the NI-RUA tools)."""
        self.assertTrue(exchanges, "the model called no NI-RUA tool")
        names = [exchange.tool for exchange in exchanges]
        before = names.index("run_begin") if "run_begin" in names else None
        briefing = [item for item in store["refusals"] if item["code"] == "external_briefing_required"]
        ni_rua_calls = [index for index, call in enumerate(facts["calls"]) if call["tool"].startswith("ni_rua:")]
        native = [call["tool"] for call in facts["calls"][:ni_rua_calls[0] if ni_rua_calls else len(facts["calls"])]]
        detail = (f"{before if before is not None else 'all'} NI-RUA calls before run_begin; {len(briefing)} "
                  f"external_briefing_required refusals; native calls before: {native or 'none'}")
        self.assertEqual(names[0], "run_begin", f"first NI-RUA call {names[0]}; {detail}")
        return "yes; " + detail

    def check_observe(self, exchanges):
        observes = [exchange for exchange in exchanges if exchange.tool == "robot_observe"]
        images = [result_parts(exchange.result)[1] for exchange in observes
                  if exchange.result and not exchange.result.get("isError")]
        self.assertTrue(any(images), f"{len(observes)} robot_observe calls, none returned an image")
        sizes = [f"{image.media_type} {image_size(image.data)[0]}x{image_size(image.data)[1]}"
                 for image in next(group for group in images if group)]
        return f"{len(observes)} calls; first images: {', '.join(sizes)}"

    def check_action(self, store):
        actions = store["actions"]
        completed = [action for action in actions if action["kind"] == "action" and action["status"] == "completed"]
        self.assertTrue(completed, f"no robot_action with a completed typed receipt: {actions}")
        return (f"{len(actions)} robot_action results; completed receipt, measured end-effector dz "
                f"{completed[0]['measured_dz_m']} m")

    def check_finish(self, exchanges, store, summary_text, gateway):
        """run_finish was served; its summary is the turn's final answer (origin run_finish_summary) and the turn
        closed completed."""
        calls = [exchange for exchange in exchanges if exchange.tool == "run_finish"]
        self.assertTrue(calls, "run_finish was never called")
        self.assertIsNotNone(summary_text, "run_finish was refused: "
                             + "; ".join(_short_result(exchange.result) for exchange in calls))
        recorded = [item for item in store["finishes"] if item["argument"] == summary_text]
        self.assertTrue(recorded, f"the served run_finish has no tool_call row: {store['finishes']}")
        final, turn = recorded[0]["restored"], recorded[0]["turn_id"]
        rows = [row for row in store["rows"] if row.get("turn_id") == turn]
        self.assertEqual([(row["text"], row["phase"], row["status"]) for row in rows],
                         [(final, "final_answer", "completed")], store["rows"])
        self.assertEqual(store["finals"][turn], [final])
        self.assertEqual(store["turns"].get(turn), "completed", store["turns"])
        summary = gateway.summary() or {}
        expanded = "" if final == summary_text else "; short ids expanded to full ids in the archive"
        return (f"{len(calls)} call(s); final answer = run_finish summary ({len(summary_text)} chars{expanded}); its "
                f"turn closed completed; summary.json agent_turn_status {summary.get('agent_turn_status')}")

    def check_colour(self, summary_text, facts, colour):
        """The CUBE_COLOUR line (the run_finish summary first, else the host's own assistant texts) names the
        rendered colour."""
        for source, texts in (("run_finish summary", [summary_text or ""]),
                              ("host transcript", facts["texts"] + [facts.get("final_text") or ""])):
            named = host_e2e.LIVE_COLOUR_LINE.findall("\n".join(texts))
            if named:
                self.assertIn(named[0].lower(), so101_scene.COLOUR_WORDS[colour],
                              f"CUBE_COLOUR {named[0]} in the {source}; rendered {colour}")
                return f"{named[0]} = rendered {colour} (in the {source})"
        self.fail(f"no CUBE_COLOUR line in the run_finish summary or the host transcript (rendered {colour})")

    def check_end(self, target, outcome):
        """The host ended the session by itself (no timeout, no request cap)."""
        if target.host == "claude":
            result = outcome.get("result") or {}
            self.assertEqual((outcome["end"], result.get("subtype"), result.get("is_error")),
                             ("completed", "success", False), outcome.get("end"))
            return "result success"
        if target.host == "codex":
            self.assertEqual((outcome["end"], outcome.get("code")), ("exited", 0), outcome)
            return "codex exec exit 0"
        self.assertEqual((outcome["end"], outcome.get("outcome")), ("completed", "succeeded"), outcome)
        return "idle, outcome succeeded"

    def check_gateway(self, gateway, returncode):
        self.assertEqual(returncode, 0, gateway.tail())
        summary = gateway.summary()
        self.assertIsNotNone(summary, "no summary.json after shutdown")
        self.assertEqual((summary["agent_backend"], summary["cleanup_status"]), ("external", "completed"))
        return "exit 0; summary.json cleanup completed"

    def check_key(self, root, gateway, secrets):
        """The key reached no file of the gateway (run directory, logs); the kept evidence is scrubbed of it (the
        scrub runs again at cleanup)."""
        self.assertEqual(host_e2e.files_containing(gateway.root, secrets), [], "the provider key is in gateway files")
        if not KEEP_ROOT:
            return "not in any gateway file; evidence removed"
        host_e2e.scrub_files(root, secrets)
        self.assertEqual(host_e2e.files_containing(root, secrets), [], "the key is still in the kept evidence")
        return "not in any gateway file; kept evidence scrubbed and re-scanned: no file contains it"


def _short_result(result):
    texts, _ = result_parts(result)
    return (" ".join(texts).strip().splitlines() or ["<no text>"])[0][:200]

if __name__ == "__main__":
    unittest.main()
