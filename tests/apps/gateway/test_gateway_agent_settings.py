"""Gateway agent settings (design 16 WP-D1): agent options, compatibility matrix, model listings, dry-run check,
profile save/remove, apply_agent_settings, run.turn_settings, runtime/agent_settings, presets and the summary block.

Synthetic only: FixtureRobot, FakeAgentSession, a loopback model listing; the Claude Code runtime probe is patched and
OpenCode has no executable (its real probe answers ``opencode_executable_not_set`` without running anything). Every
key is fake; tests assert it never reaches a response, an event, a Store record or any file of the run directory.
"""
import json
import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aiohttp.test_utils import TestClient, TestServer

from ni_rua.apps.evaluation.episode import agent_settings_summary
from ni_rua.apps.gateway.extensions import agent_settings
from ni_rua.apps.gateway.workbench import Workbench
from ni_rua.core.config import Config
from ni_rua.cortex.agents import factory
from ni_rua.cortex.agents.base import AgentCapabilities, AgentCapabilityError
from ni_rua.cortex.agents.profiles import apply_profile
from support.agent_fixtures import fake_session, fixture_harness, terminal_double
from support.fixture_robot import FixtureRobot
from support.agent_fixtures import Listing
from support.paths import REPO_ROOT, isolate_home


def setUpModule():
    isolate_home()   # a fresh Workbench opens the grasp memory store at its '~' default


NI_RUA_TEST_TIER = "component"
REPOSITORY = REPO_ROOT
FAKE_KEY = "sk-test-FAKE-d1-0123456789abcdefKEY"
OTHER_KEY = "sk-test-OTHER-d1-9876543210zyxwvuKEY"
LISTING = {"object": "list", "data": [
    {"id": "kimi-k2.6", "name": "Kimi K2.6", "protocols": ["MESSAGES", "RESPONSES"],
     "capabilities": {"tools": True, "image_input": True, "context_window": 256000,
                      "reasoning": {"supported": True, "effort": ["low", "high", "max"]}}},
    {"id": "gpt-a", "name": "GPT A", "protocols": ["RESPONSES"], "capabilities": {"tools": True, "image_input": True}},
    {"id": "gpt-b", "name": "GPT B", "protocols": ["RESPONSES"], "capabilities": {"tools": True}},
]}


def profile_document(wires, *, key=FAKE_KEY, **extra):
    document = {"schema": "ni_rua.provider_profile.v1", "name": extra.pop("name", "Fixture"), "wires": wires, **extra}
    if key is not None:
        document["credential"] = {"api_key": key}
    return document


class Proxy:
    """Relay double with the service-tier knob (probe on the current model; per-model probes)."""

    def __init__(self, supported=True):
        self.service_tier, self.service_tier_probes, self.probed, self.supported = None, {}, [], supported
        self.error = self.served = None

    def probe_service_tier(self, model, tier):
        self.probed.append((model, tier))
        if self.error:
            return dict(tier=tier, supported=None, error=self.error)
        served = (self.served or tier) if self.supported else None
        result = dict(tier=tier, supported=self.supported, served_tier=served, error=None,
                      rejection=None if self.supported else "HTTP 400 unsupported service_tier")
        self.service_tier_probes[tier] = result
        return result

    def select_model(self, model):
        return self.service_tier_probes


class AgentSettingsFixture(unittest.IsolatedAsyncioTestCase):
    """A running Workbench whose actor goes through a provider profile (``backend`` / ``profile`` per subclass)."""
    backend, profile, model = "codex", "resp", "gpt-a"
    capabilities = AgentCapabilities(live_effort=True, service_tier=True)
    effort_choices = None

    async def asyncSetUp(self):
        agent_settings.reset_runtime_probes()
        self.addCleanup(agent_settings.reset_runtime_probes)
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.listing = Listing(LISTING)
        self.addCleanup(self.listing.close)
        self.profiles_dir = self.root / "profiles"
        self.profiles_dir.mkdir()
        base = self.listing.base
        self.write("anthro", profile_document({"anthropic_messages": {"base_url": base}},
                                              defaults={"model": "kimi-k2.6", "effort": "high"}, name="Anthro"))
        self.write("resp", profile_document({"openai_responses": {"base_url": base + "/v1"}}, name="Resp"))
        self.write("both", profile_document({"anthropic_messages": {"base_url": base},
                                             "openai_responses": {"base_url": base + "/v1"}}, name="Both"))
        self.write("nokey", profile_document({"anthropic_messages": {"base_url": base}}, key=None, name="No key"))
        (self.profiles_dir / "broken.local.json").write_text("{ not json")
        probe = patch.object(factory, "claude_code_runtime_problem", return_value=None)
        probe.start()
        self.addCleanup(probe.stop)
        config = Config.from_dict({
            "model": {"name": self.model, "effort": "high"},
            "features": {"grasp_memory_dir": str(self.root / "memory"), "observation_profile": "rgb_proprio_v1",
                         "motion_review": False},
            "agent": {"profiles_dir": str(self.profiles_dir)}})
        config = apply_profile(config, self.backend, self.profiles_dir / f"{self.profile}.local.json")
        options = dict(backend_kind=self.backend, model=self.model, provider=self.profile)
        if self.effort_choices is not None:
            options["effort_choices"] = self.effort_choices
        create, created = fake_session(self.capabilities, **options)
        robot = FixtureRobot(self.root)
        robot.close = lambda: None
        with patch("ni_rua.apps.gateway.workbench.create_session", side_effect=create):
            self.workbench = Workbench(self.root / "run", robot=robot, config=config)
        self.session, = created
        if self.capabilities.service_tier:
            self.session.model_proxy = Proxy()
        self.workbench.harness.start()
        self.workbench.starting, self.workbench.status = False, "running"
        self.session.start([], "instructions")
        self.client = TestClient(TestServer(self.workbench.app()))
        await self.client.start_server()
        self.addAsyncCleanup(self.workbench.close)
        self.addAsyncCleanup(self.client.close)
        self.origin = str(self.client.make_url("")).rstrip("/")
        status, owner = await self.post("/v1/control", dict(request_id="claim", command="claim_input",
                                                            client_id="fixture"))
        self.assertEqual(status, 200)
        self.lease = dict(client_id="fixture", writer_token=owner["writer_token"])

    def write(self, profile_id, document):
        path = self.profiles_dir / f"{profile_id}.local.json"
        path.write_text(json.dumps(document))
        os.chmod(path, 0o600)
        return path

    async def post(self, path, body):
        response = await self.client.post(path, headers={"Origin": self.origin}, json=body)
        return response.status, await response.json()

    async def get(self, path):
        response = await self.client.get(path)
        if response.status == 200:
            self.assertEqual(response.headers.get("Cache-Control"), "no-store", path)
        return response.status, await response.json()

    async def control(self, identifier, command, **fields):
        return await self.post("/v1/control", dict(request_id=identifier, command=command, **self.lease, **fields))

    def events(self, kind):
        return [event["payload"] for event in self.workbench.store.events_by_type([kind])]

    def assertNoKey(self, value):
        text = value if isinstance(value, str) else json.dumps(value)
        for key in (FAKE_KEY, OTHER_KEY):
            self.assertNotIn(key, text)
            self.assertNotIn(key[8:22], text)

    def run_files_without_keys(self):
        """Every file under the run directory (Store database, WAL, records, artifacts) holds neither test key."""
        checked = 0
        for path in (self.workbench.root).rglob("*"):
            if path.is_file() and not path.is_symlink():
                data = path.read_bytes()
                for key in (FAKE_KEY, OTHER_KEY):
                    self.assertNotIn(key.encode(), data, path)
                checked += 1
        self.assertGreater(checked, 0)


class AgentOptionsTests(AgentSettingsFixture):
    """Codex run through the Responses-only profile ``resp``."""

    async def test_options_list_backends_profiles_and_matrix_reasons_without_keys(self):
        status, body = await self.get("/v1/console/agent-options")
        self.assertEqual(status, 200)
        self.assertNoKey(body)
        backends = {item["id"]: item for item in body["backends"]}
        self.assertEqual(list(backends), ["codex", "claude_code", "opencode"])
        self.assertEqual((backends["codex"]["available"], backends["claude_code"]["available"]), (True, True))
        self.assertEqual((backends["opencode"]["available"], backends["opencode"]["reason"]),
                         (False, "backend_unavailable:opencode_executable_not_set"))
        self.assertEqual(backends["opencode"]["effort_choices_by_wire"]["openai_responses"],
                         ["none", "minimal", "low", "medium", "high", "xhigh"])
        self.assertEqual(backends["opencode"]["thinking_display_wires"], ["anthropic_messages"])
        self.assertTrue(backends["codex"]["current"])
        self.assertEqual((backends["claude_code"]["live_effort"], backends["codex"]["service_tier"]), (False, True))

        profiles = {item["id"]: item for item in body["profiles"]}
        self.assertEqual(sorted(profiles), ["anthro", "both", "broken", "nokey", "resp"])
        self.assertFalse(profiles["broken"]["valid"])
        self.assertEqual(profiles["broken"]["error"], "profile_unreadable")
        self.assertIn("broken.local.json", profiles["broken"]["error_message"])
        self.assertEqual((profiles["resp"]["error"], profiles["resp"]["error_message"]), (None, None))
        self.assertEqual((profiles["nokey"]["key_present"], profiles["resp"]["key_present"]), (False, True))
        self.assertEqual((profiles["resp"]["in_use"], profiles["anthro"]["in_use"]), (True, False))
        self.assertNotIn("credential", json.dumps(profiles))

        cells = {(cell["backend"], cell["profile"]): cell for cell in body["matrix"]}
        self.assertEqual(len(cells), 15)
        expected = {
            ("codex", "resp"): None, ("codex", "both"): None, ("codex", "anthro"): "no_common_wire",
            ("codex", "nokey"): "no_common_wire", ("codex", "broken"): "profile_invalid:profile_unreadable",
            ("claude_code", "anthro"): None, ("claude_code", "both"): None,
            ("claude_code", "resp"): "no_common_wire", ("claude_code", "nokey"): "profile_missing_key",
            ("claude_code", "broken"): "profile_invalid:profile_unreadable"}
        for key, reason in expected.items():
            self.assertEqual((cells[key]["compatible"], cells[key]["reason"]), (reason is None, reason), key)
        for profile in profiles:
            self.assertEqual(cells[("opencode", profile)]["reason"],
                             "backend_unavailable:opencode_executable_not_set")
            self.assertIn("运行时不可用", cells[("opencode", profile)]["message"])
        self.assertEqual(cells[("codex", "both")]["wires"], ["openai_responses"])
        self.assertIn("anthropic_messages", cells[("claude_code", "resp")]["message"])
        self.assertEqual(cells[("claude_code", "anthro")]["checks"],
                         [dict(wire="anthropic_messages", ok=True, reason=None, message=None, effort="high")])

        current = body["current"]
        self.assertEqual({key: current[key] for key in ("backend", "profile", "provider", "wire_api", "model",
                                                         "effort", "service_tier", "upstream_source")},
                         dict(backend="codex", profile="resp", provider="resp", wire_api="openai_responses",
                              model="gpt-a", effort="high", service_tier="default", upstream_source="profile"))
        self.assertEqual((current["revision"], current["mixed"], current["turn_count"]), (0, False, 0))
        # Reasoning of the running selection, with the listing entry read through the profile's key.
        reasoning = body["reasoning"]
        self.assertEqual((reasoning["backend"], reasoning["model"]["id"], reasoning["model"]["listed"]),
                         ("codex", "gpt-a", True))
        self.assertEqual((reasoning["effort"]["command"], reasoning["service_tier"]["available"]), ("set_effort", True))
        self.assertEqual(self.listing.requests[-1]["headers"]["authorization"], "Bearer " + FAKE_KEY)
        listed = {(item["profile"], item["wire_api"]): item for item in body["catalogues"]}
        self.assertEqual((listed[("resp", "openai_responses")]["status"], listed[("resp", "openai_responses")]["count"]),
                         ("ok", 3))
        self.assertIsNone(listed[("anthro", "anthropic_messages")]["status"])   # never read for the options
        self.assertEqual(len(self.listing.requests), 1)
        self.assertEqual(body["handoff"]["allowed"], False)
        self.assertIn("handoff_not_available", [item["code"] for item in body["handoff"]["blockers"]])
        self.assertEqual(body["profile_schema"]["schema"], "ni_rua.provider_profile.v1")
        self.assertIn("credential.api_key", [field["name"] for field in body["profile_schema"]["fields"]])

    async def test_matrix_is_cached_per_profile_version_and_gate_reasons_are_verbatim(self):
        calls = []
        real = factory.backend_requirements

        def counting(candidate):
            calls.append(candidate.agent.backend)
            return real(candidate)
        with patch.object(factory, "backend_requirements", side_effect=counting):
            await self.get("/v1/console/agent-options")
            first = len(calls)
            await self.get("/v1/console/agent-options")
            self.assertEqual(len(calls), first, "unchanged profiles reuse the cached dry runs")
            path = self.profiles_dir / "anthro.local.json"
            os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 5_000_000_000))
            await self.get("/v1/console/agent-options")
            self.assertEqual(calls[first:], ["claude_code"], "a changed profile file is checked again")
        # A configuration the launch gate refuses for one backend: the cell carries its sentence verbatim.
        self.workbench.config = self.workbench.config.with_overrides({"features": {"visual_review_model": "gpt-6.1-sol"}})
        self.workbench._agent_check_cache = {}
        _, body = await self.get("/v1/console/agent-options")
        cell = next(cell for cell in body["matrix"] if (cell["backend"], cell["profile"]) == ("claude_code", "anthro"))
        self.assertEqual(cell["reason"], "gate:evidence_upstream_required_for_visual_review_model")
        self.assertTrue(cell["message"].startswith("evidence_upstream_required_for_visual_review_model:"))

    async def test_runtime_probes_are_not_repeated_for_known_selections(self):
        with patch.object(factory, "claude_code_runtime_problem", return_value=None) as probe:
            agent_settings.reset_runtime_probes()
            await self.get("/v1/console/agent-options")
            first = probe.call_count     # the availability probe plus the launch gate of the two Claude Code cells
            self.assertEqual(first, 3)
            await self.get("/v1/console/agent-options")
            self.assertEqual(probe.call_count, first)
            selection = dict(backend="claude_code", profile="anthro", model="kimi-k2.6", effort="max")
            await self.post("/v1/console/agent-options/check", selection)
            self.assertEqual(probe.call_count, first + 1)        # a new exact selection runs the launch gate once
            await self.post("/v1/console/agent-options/check", selection)
            self.assertEqual(probe.call_count, first + 1)
        with patch.object(factory, "claude_code_runtime_problem", return_value="claude_code_sdk_missing:ImportError"):
            agent_settings.reset_runtime_probes()
            _, body = await self.post("/v1/console/agent-options/check", selection)
            self.assertEqual((body["ok"], body["reason"]), (False, "backend_unavailable:claude_code_sdk_missing:ImportError"))
            _, options = await self.get("/v1/console/agent-options")
        cell = next(cell for cell in options["matrix"] if (cell["backend"], cell["profile"]) == ("claude_code", "both"))
        self.assertEqual(cell["reason"], "backend_unavailable:claude_code_sdk_missing:ImportError")

    async def test_profile_models_endpoint_caches_refreshes_and_refuses(self):
        status, body = await self.get("/v1/console/profiles/both/models?wire=anthropic_messages")
        self.assertEqual((status, body["status"], body["profile"], body["wire_api"]),
                         (200, "ok", "both", "anthropic_messages"))
        self.assertEqual([model["id"] for model in body["models"]], ["kimi-k2.6", "gpt-a", "gpt-b"])
        self.assertEqual(body["models"][0]["reasoning_efforts"], ["low", "high", "max"])
        self.assertEqual(self.listing.requests[-1]["headers"]["authorization"], "Bearer " + FAKE_KEY)
        self.assertNoKey(body)
        count = len(self.listing.requests)
        _, again = await self.get("/v1/console/profiles/both/models?wire=anthropic_messages")
        self.assertEqual((len(self.listing.requests), again["origin"]), (count, "memory"))
        _, fresh = await self.get("/v1/console/profiles/both/models?wire=anthropic_messages&refresh=1")
        self.assertEqual((len(self.listing.requests), fresh["origin"]), (count + 1, "network"))
        _, single = await self.get("/v1/console/profiles/resp/models")       # one wire: ?wire= may be left out
        self.assertEqual(single["wire_api"], "openai_responses")
        self.assertEqual((await self.client.get("/v1/console/profiles/missing/models")).status, 404)
        status, body = await self.get("/v1/console/profiles/resp/models?wire=anthropic_messages")
        self.assertEqual((status, body["error"].split(":")[0]), (400, "profile_wire_not_served"))
        status, body = await self.get("/v1/console/profiles/nokey/models?wire=anthropic_messages")
        self.assertEqual((body["status"], body["reason"]), ("unavailable", "credential_unavailable:profile_missing_key"))
        status, _ = await self.get("/v1/console/profiles/Bad%20Id/models")
        self.assertEqual(status, 400)
        disk = list((self.profiles_dir / ".cache").glob("*.json"))
        self.assertTrue(disk)
        for path in disk:
            self.assertNoKey(path.read_text())

    async def test_check_is_a_side_effect_free_dry_run_with_reasoning(self):
        before = (self.workbench.store.event_cursor(), len(self.workbench.store.records("control_request")))
        status, body = await self.post("/v1/console/agent-options/check", dict(
            backend="claude_code", profile="anthro", wire_api="anthropic_messages", model="kimi-k2.6", effort="max"))
        self.assertEqual((status, body["ok"], body["reason"], body["wire_api"]), (200, True, None, "anthropic_messages"))
        self.assertEqual((body["reasoning"]["effort"]["applies"], body["reasoning"]["effort"]["command"]),
                         ("next_session", "handoff_run"))
        cases = [
            (dict(backend="claude_code", profile="anthro", effort="none"), "gate:agent_effort_unsupported"),
            (dict(backend="claude_code", profile="resp"), "no_common_wire"),
            (dict(backend="claude_code", profile="nokey"), "profile_missing_key"),
            (dict(backend="claude_code", profile="broken"), "profile_invalid:profile_unreadable"),
            (dict(backend="claude_code", profile="ghost"), "profile_not_found"),
            (dict(backend="claude_code"), "profile_required"),
            (dict(backend="opencode", profile="both"), "backend_unavailable:opencode_executable_not_set"),
            (dict(backend="robot", profile="both"), "unknown_agent_backend"),
            (dict(backend="codex", profile="both", model="bad model"), "invalid_model"),
            (dict(backend="claude_code", profile="both", wire_api="openai_responses"), "wire_unsupported"),
        ]
        for request, reason in cases:
            status, body = await self.post("/v1/console/agent-options/check", request)
            self.assertEqual((status, body["ok"]), (200, False), request)
            self.assertTrue(body["reason"].startswith(reason), (request, body))
            self.assertTrue(body["message"], request)
            self.assertNoKey(body)
        status, body = await self.post("/v1/console/agent-options/check", dict(backend="codex", profile="both"))
        self.assertEqual((body["ok"], body["wire_api"]), (True, "openai_responses"))
        self.assertEqual(body["reasoning"]["service_tier"]["available"], True)
        self.assertEqual((self.workbench.store.event_cursor(), len(self.workbench.store.records("control_request"))),
                         before)

    async def test_console_state_carries_the_agent_selection_and_listing_capabilities(self):
        self.session.model_options = lambda: [dict(value="gpt-a", name=None, image_input=None),
                                              dict(value="unlisted-x", name=None, image_input=None)]
        await self.get("/v1/console/agent-options")     # reads the running profile's listing into the cache
        state = await (await self.client.get("/v1/console/state")).json()
        agent = state["settings"]["agent"]
        self.assertEqual((agent["backend"], agent["profile"], agent["wire_api"], agent["model"]),
                         ("codex", "resp", "openai_responses", "gpt-a"))
        self.assertIsNone(agent["served"])
        options = {item["value"]: item for item in state["settings"]["runtime"]["model"]["options"]}
        self.assertEqual((options["gpt-a"]["name"], options["gpt-a"]["image_input"]), ("GPT A", True))
        self.assertEqual(options["unlisted-x"], dict(value="unlisted-x", label="unlisted-x", image_input=None))
        self.assertEqual(options["gpt-a"]["protocols"], ["RESPONSES"])
        self.assertNoKey(state)

    async def test_served_model_is_reported_from_the_relay_events(self):
        store = self.workbench.store
        store.event("model.model_request_finished", dict(kind="model_request_finished", logical_call_id="c1",
                                                          served_model="gpt-a-2026"), source="private_model_transport")
        store.set_state("runtime", "model_requests", dict(served_models={"gpt-a-2026": 1}))
        _, body = await self.get("/v1/console/agent-options")
        self.assertEqual((body["served"]["model"], body["served"]["models"]), ("gpt-a-2026", {"gpt-a-2026": 1}))
        self.assertTrue(body["served"]["at"].endswith("Z"))


# Keys the console Agent bar reads (design 16 section 15; web/agent-settings-model.js and agent-settings-view.js).
FRONTEND_KEYS = dict(
    top=("current", "backends", "profiles", "matrix", "reasoning", "handoff", "served", "presets"),
    current=("backend", "profile", "provider", "wire_api", "model", "effort", "service_tier", "thinking_display",
             "changed_at"),
    backend=("id", "label", "available", "reason", "message", "wires", "live_effort", "service_tier", "effort_choices"),
    profile=("id", "name", "valid", "error", "error_message", "wires", "compat", "backends", "defaults",
             "key_present", "in_use"),
    profile_wire=("base_url", "host", "auth_scheme", "models_url"),
    matrix=("backend", "profile", "compatible", "reason", "message", "wires"),
    effort=("value", "command", "applies", "choices", "model_source"),
    effort_choice=("value", "backend", "model"),
    service_tier=("available", "value", "command", "applies", "choices"),
    tier_choice=("value", "label", "probe"),
    handoff=("allowed", "blockers", "disable_torque_on_disconnect", "last"),
    served=("model", "at"),
    models=("status", "reason", "fetched_at", "age_s", "stale", "models"),
    model=("id", "name", "protocols", "tools", "image_input", "context_window", "max_output_tokens", "reasoning",
           "reasoning_efforts"),
    check=("ok", "reason", "message", "reasoning"))


class FrontendContractTests(AgentSettingsFixture):
    """Every key the WP-E Agent bar reads is in the real payloads (shapes of design 16 section 15)."""

    def assertKeys(self, value, group):
        self.assertIsInstance(value, dict, group)
        missing = [key for key in FRONTEND_KEYS[group] if key not in value]
        self.assertEqual(missing, [], f"{group} lacks {missing}")

    async def test_agent_options_models_check_and_receipts_have_every_key_the_frontend_reads(self):
        store = self.workbench.store
        store.event("model.model_request_finished", dict(kind="model_request_finished", served_model="gpt-a"),
                    source="private_model_transport")
        _, body = await self.get("/v1/console/agent-options")
        self.assertKeys(body, "top")
        self.assertKeys(body["current"], "current")
        for backend in body["backends"]:
            self.assertKeys(backend, "backend")
            self.assertIsInstance(backend["effort_choices"], list)
        for profile in body["profiles"]:
            self.assertKeys(profile, "profile")
            self.assertIsInstance(profile["error"], (str, type(None)))
            for entry in profile["wires"].values():
                self.assertKeys(entry, "profile_wire")
            if profile["valid"]:
                self.assertIn("env", profile["backends"]["claude_code"])
                self.assertEqual(sorted(profile["defaults"]), ["effort", "model"])
        for cell in body["matrix"]:
            self.assertKeys(cell, "matrix")
        self.assertKeys(body["reasoning"]["effort"], "effort")
        for choice in body["reasoning"]["effort"]["choices"]:
            self.assertKeys(choice, "effort_choice")
        self.assertKeys(body["reasoning"]["service_tier"], "service_tier")
        for choice in body["reasoning"]["service_tier"]["choices"]:
            self.assertKeys(choice, "tier_choice")
        self.assertIn("available", body["reasoning"]["thinking_display"])
        self.assertKeys(body["handoff"], "handoff")
        for blocker in body["handoff"]["blockers"]:
            self.assertEqual(sorted(blocker), ["code", "message"])
        self.assertKeys(body["served"], "served")
        self.assertIsInstance(body["presets"], dict)
        _, models = await self.get("/v1/console/profiles/both/models?wire=anthropic_messages")
        self.assertKeys(models, "models")
        for model in models["models"]:
            self.assertKeys(model, "model")
        _, check = await self.post("/v1/console/agent-options/check", dict(
            backend="codex", profile="both", wire_api="openai_responses", model="gpt-a", effort="high"))
        self.assertKeys(check, "check")
        status, receipt = await self.control("apply", "apply_agent_settings", model="gpt-b")
        self.assertEqual((status, receipt["results"]["model"]["status"]), (200, "applied"))
        status, refused = await self.control("bad", "apply_agent_settings", backend="robot")
        self.assertEqual(status, 400)
        self.assertIn("error", refused)
        status, handoff = await self.control("handoff", "handoff_run", backend="claude_code", profile="anthro",
                                             wire_api="anthropic_messages", model="kimi-k2.6", effort="high")
        self.assertEqual((status, handoff["error"].split(":")[0]), (400, "handoff_not_available"))
        _, after = await self.get("/v1/console/agent-options")
        self.assertEqual((after["presets"]["resp/codex/gpt-b"]["effort"], after["current"]["model"]), ("high", "gpt-b"))

    async def test_the_frontend_model_module_digests_the_real_payload(self):
        """web/agent-settings-model.js (the Agent bar's pure state module) run by node on the real payload."""
        node = shutil.which("node")
        if node is None:
            self.skipTest("node is not installed")
        _, options = await self.get("/v1/console/agent-options")
        _, models = await self.get("/v1/console/profiles/both/models?wire=openai_responses")
        state = await (await self.client.get("/v1/console/state")).json()
        payload = self.root / "payload.json"
        payload.write_text(json.dumps(dict(options=options, models=models, state=state)))
        script = """
import fs from 'node:fs';
import * as M from './agent-settings-model.js';
const { options, models, state } = JSON.parse(fs.readFileSync(process.argv[1], 'utf8'));
const o = M.normalizeOptions(options, { consoleState: state });
const display = M.draftFrom(o.current);
const out = {
  source: o.source, current: o.current,
  backends: M.backendRows(o, display).map((row) => [row.id, row.disabled, row.reason]),
  profiles: M.profileRows(o, display, o.current).map((row) => [row.id, row.disabled, row.reason, row.keyText, row.inUse]),
  cell: M.cell(o, 'claude_code', 'nokey'), broken: M.cell(o, 'codex', 'broken'),
  wire: M.wireFor(o, 'opencode', 'both', {}),
  efforts: M.effortRows(o.reasoning.effort, o.current.effort).map((row) => [row.value, row.disabled]),
  tiers: M.tierRows(o.reasoning.service_tier, o.current.service_tier).map((row) => [row.value, row.label]),
  models: M.modelRows(models.models, { wire: 'openai_responses', current: o.current.model }).rows.map((row) => [row.id, row.disabled]),
  summary: M.catalogueSummary(models), handoff: o.handoff,
};
console.log(JSON.stringify(out));
"""
        result = subprocess.run([node, "--input-type=module", "-e", script, str(payload)], cwd=REPOSITORY / "web",
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        out = json.loads(result.stdout)
        self.assertEqual(out["source"], "agent-options")
        self.assertEqual((out["current"]["backend"], out["current"]["profile"], out["current"]["model"]),
                         ("codex", "resp", "gpt-a"))
        backends = {row[0]: row for row in out["backends"]}
        self.assertTrue(backends["opencode"][1])
        self.assertIn("opencode_executable_not_set", backends["opencode"][2])
        profiles = {row[0]: row for row in out["profiles"]}
        self.assertEqual((profiles["resp"][1], profiles["resp"][4]), (False, True))
        self.assertTrue(profiles["anthro"][1])
        self.assertIn("anthropic_messages", profiles["anthro"][2])
        self.assertTrue(profiles["broken"][1])
        self.assertNotIn("[object Object]", profiles["broken"][2])
        self.assertIn("broken.local.json", profiles["broken"][2])
        self.assertEqual(out["broken"]["reason"], "profile_unreadable")
        self.assertEqual(out["cell"]["reason"], "profile_missing_key")
        self.assertEqual(out["wire"], "anthropic_messages")    # two shared wires: anthropic_messages first
        self.assertIn(["high", False], out["efforts"])
        self.assertEqual([tier[0] for tier in out["tiers"]], ["default", "priority", "ultrafast"])
        self.assertEqual(out["models"], [["kimi-k2.6", False], ["gpt-a", False], ["gpt-b", False]])
        self.assertIn("个模型", out["summary"]["text"])
        self.assertEqual(out["handoff"]["allowed"], False)
        self.assertNoKey(result.stdout)


class ProfileWriteTests(AgentSettingsFixture):
    async def save(self, body, lease=True):
        return await self.post("/v1/console/profiles", dict(self.lease if lease else {}, **body))

    async def test_save_writes_the_file_returns_no_key_and_never_archives_it(self):
        document = profile_document({"anthropic_messages": {"base_url": "https://gw.example/anthropic"}},
                                    key=OTHER_KEY, name="New gateway",
                                    backends={"claude_code": {"env": {"CLAUDE_CODE_THINKING_DISPLAY_UPDATES": "0"}}})
        status, body = await self.save(dict(id="newgw", document=document, keep_key=False))
        self.assertEqual((status, body["id"], body["valid"], body["key_present"], body["in_use"]),
                         (200, "newgw", True, True, False))
        self.assertNoKey(body)
        path = self.profiles_dir / "newgw.local.json"
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual(json.loads(path.read_text())["credential"]["api_key"], OTHER_KEY)
        # Edit with an empty key field keeps the stored key.
        document = dict(document, name="Renamed", credential={"api_key": ""})
        status, body = await self.save(dict(id="newgw", document=document, keep_key=True))
        self.assertEqual((status, body["name"], body["key_present"]), (200, "Renamed", True))
        self.assertEqual(json.loads(path.read_text())["credential"]["api_key"], OTHER_KEY)
        # Key-only replacement of an existing profile.
        status, body = await self.save(dict(id="nokey", api_key=OTHER_KEY))
        self.assertEqual((status, body["key_present"]), (200, True))
        saved = self.events("operator.profile_saved")
        self.assertEqual([(item["id"], item["key_changed"], item["created"]) for item in saved],
                         [("newgw", True, True), ("newgw", False, False), ("nokey", True, False)])
        self.assertEqual(saved[0]["wires"], ["anthropic_messages"])
        self.assertNoKey(saved)
        _, options = await self.get("/v1/console/agent-options")
        self.assertNoKey(options)
        self.assertEqual(self.workbench.store.records("control_request"), [
            record for record in self.workbench.store.records("control_request") if record.get("command") ==
            "claim_input"])
        self.run_files_without_keys()

    async def test_writes_need_the_lease_same_origin_and_a_valid_document(self):
        document = profile_document({"anthropic_messages": {"base_url": "https://gw.example"}}, key=OTHER_KEY)
        status, body = await self.save(dict(id="x1", document=document), lease=False)
        self.assertEqual((status, body["error"]), (409, "input_owner_conflict"))
        response = await self.client.post("/v1/console/profiles", headers={"Origin": "http://evil.example"},
                                          json=dict(self.lease, id="x1", document=document))
        self.assertEqual(response.status, 403)
        status, body = await self.save(dict(id="Bad Id", document=document))
        self.assertEqual((status, body["error"].split(":")[0]), (400, "profile_invalid_id"))
        status, body = await self.save(dict(id="x1", document=dict(document, schema="other")))
        self.assertEqual((status, body["error"].split(":")[0]), (400, "profile_schema"))
        self.assertNoKey(body)
        status, body = await self.save(dict(id="x1", document="not an object"))
        self.assertEqual((status, body["error"].split(":")[0]), (400, "profile_document_required"))
        self.assertFalse((self.profiles_dir / "x1.local.json").exists())
        self.run_files_without_keys()

    async def test_remove_renames_and_refuses_the_profile_in_use(self):
        status, body = await self.post("/v1/console/profiles/anthro/remove", dict(self.lease))
        self.assertEqual((status, body["status"]), (200, "removed"))
        self.assertTrue(body["renamed_to"].startswith("anthro.local.json.removed-"))
        self.assertTrue((self.profiles_dir / body["renamed_to"]).is_file())
        self.assertFalse((self.profiles_dir / "anthro.local.json").exists())
        status, body = await self.post("/v1/console/profiles/resp/remove", dict(self.lease))
        self.assertEqual((status, body["error"].split(":")[0]), (400, "profile_in_use"))
        status, body = await self.post("/v1/console/profiles/ghost/remove", dict(self.lease))
        self.assertEqual((status, body["error"].split(":")[0]), (400, "profile_not_found"))
        status, _ = await self.post("/v1/console/profiles/both/remove", {})
        self.assertEqual(status, 409)
        self.assertEqual([item["id"] for item in self.events("operator.profile_removed")], ["anthro"])


class ApplySettingsTests(AgentSettingsFixture):
    """Codex-shaped run: model, effort (live) and service tier all apply in the running session."""

    async def test_three_fields_one_receipt_and_the_three_existing_events(self):
        status, body = await self.control("apply", "apply_agent_settings", model="gpt-b", effort="low",
                                          service_tier="priority")
        self.assertEqual((status, body["status"]), (200, "applied"))
        self.assertEqual([body["results"][field]["status"] for field in ("model", "effort", "service_tier")],
                         ["applied"] * 3)
        self.assertEqual((body["results"]["model"]["applies"], body["results"]["effort"]["applies"],
                          body["results"]["service_tier"]["applies"]),
                         ("next_agent_turn", "next_codex_turn", "next_model_request"))
        self.assertEqual(body["results"]["service_tier"]["probe"]["supported"], True)
        self.assertEqual((self.session.model, self.session.effort, self.session.model_proxy.service_tier),
                         ("gpt-b", "low", "priority"))
        self.assertEqual(self.session.model_proxy.probed, [("gpt-b", "priority")])
        for kind, key, value in (("operator.model_changed", "model", "gpt-b"),
                                 ("operator.effort_changed", "effort", "low"),
                                 ("operator.service_tier_changed", "tier", "priority")):
            event, = self.events(kind)
            self.assertEqual((event[key], event["request_id"], event["backend"], event["profile"], event["wire_api"]),
                             (value, "apply", "codex", "resp", "openai_responses"))
        applied, = self.events("operator.agent_settings_applied")
        self.assertEqual((applied["request_id"], applied["status"], sorted(applied["results"])),
                         ("apply", "applied", ["effort", "model", "service_tier"]))
        state = self.workbench.store.read_state("runtime", "agent_settings")
        self.assertEqual((state["revision"], state["model"], state["effort"], state["service_tier"], state["mixed"]),
                         (3, "gpt-b", "low", "priority", False))
        self.assertTrue(state["changed_at"].endswith("Z"))
        presets = json.loads((self.profiles_dir / "presets.json").read_text())["presets"]
        self.assertEqual({key: (value["effort"], value["service_tier"]) for key, value in presets.items()},
                         {"resp/codex/gpt-b": ("low", "priority")})
        _, options = await self.get("/v1/console/agent-options")
        self.assertEqual(options["presets"]["resp/codex/gpt-b"]["effort"], "low")
        self.assertEqual((options["current"]["revision"], options["current"]["model"]), (3, "gpt-b"))
        # The same request again is idempotent through /v1/control (same receipt, nothing re-applied).
        self.assertEqual(await self.control("apply", "apply_agent_settings", model="gpt-b", effort="low",
                                            service_tier="priority"), (status, body))
        self.assertEqual(len(self.events("operator.model_changed")), 1)

    async def test_refused_model_leaves_effort_and_tier_untouched(self):
        def refuse(model):
            raise AgentCapabilityError(f"opencode_model_not_declared:{model}: start a new run with --model {model}")
        self.session.set_model = refuse
        status, body = await self.control("apply", "apply_agent_settings", model="glm-x", effort="low",
                                          service_tier="priority")
        self.assertEqual((status, body["status"]), (200, "rejected"))
        self.assertEqual((body["results"]["model"]["status"], body["results"]["model"]["reason"]),
                         ("rejected", "opencode_model_not_declared"))
        self.assertEqual([body["results"][field]["status"] for field in ("effort", "service_tier")],
                         ["skipped", "skipped"])
        self.assertEqual((self.session.model, self.session.effort, self.session.model_proxy.service_tier),
                         ("gpt-a", "high", None))
        self.assertEqual(self.session.model_proxy.probed, [])
        for kind in ("operator.model_changed", "operator.effort_changed", "operator.service_tier_changed"):
            self.assertEqual(self.events(kind), [], kind)
        status, body = await self.control("bad", "apply_agent_settings", model="bad model", effort="low")
        self.assertEqual((body["results"]["model"]["reason"], body["results"]["effort"]["status"]),
                         ("invalid_model", "skipped"))

    async def test_later_failure_is_partial_and_unchanged_fields_are_not_reapplied(self):
        self.session.model_proxy.supported = False
        status, body = await self.control("apply", "apply_agent_settings", model="gpt-a", effort="medium",
                                          service_tier="ultrafast")
        self.assertEqual((status, body["status"]), (200, "partial"))
        self.assertEqual((body["results"]["model"]["status"], body["results"]["model"]["changed"]), ("applied", False))
        self.assertEqual((body["results"]["effort"]["status"], body["results"]["service_tier"]["status"]),
                         ("applied", "unsupported"))
        self.assertEqual(self.events("operator.model_changed"), [])
        self.assertEqual(self.session.effort, "medium")

    async def test_accepted_tier_reports_what_the_provider_actually_served(self):
        self.session.model_proxy.served = "default"        # accepted priority, served default (seen on upstream)
        status, body = await self.control("tier", "apply_agent_settings", service_tier="priority")
        result = body["results"]["service_tier"]
        self.assertEqual((status, result["status"], result["tier"], result["served_tier"]),
                         (200, "applied", "priority", "default"))
        self.assertIn("实际以 default 档位服务", result["message"])
        _, options = await self.get("/v1/console/agent-options")
        self.assertEqual((options["reasoning"]["service_tier"]["value"], options["reasoning"]["service_tier"]["served_tier"]),
                         ("priority", "default"))
        self.assertEqual((options["served"]["service_tier"], options["served"]["service_tier_source"]),
                         ("default", "probe"))
        priority = next(choice for choice in options["reasoning"]["service_tier"]["choices"]
                        if choice["value"] == "priority")
        self.assertEqual(priority["probe"]["served_tier"], "default")

    async def test_responses_listing_ids_are_filled_from_the_native_codex_catalogue(self):
        self.listing.body = dict(LISTING, data=LISTING["data"] + [{"id": "gpt-6.1-sol", "protocols": ["RESPONSES"]}])
        _, body = await self.get("/v1/console/profiles/resp/models?refresh=1")
        sol = next(model for model in body["models"] if model["id"] == "gpt-6.1-sol")
        self.assertEqual((sol["name"], sol["image_input"]), ("GPT-6.1-Sol", True))
        self.assertTrue(sol["context_window"] and sol["reasoning_efforts"])
        listed = next(model for model in body["models"] if model["id"] == "kimi-k2.6")
        self.assertEqual(listed["reasoning_efforts"], ["low", "high", "max"])     # listing claims are kept
        _, anthropic = await self.get("/v1/console/profiles/both/models?wire=anthropic_messages&refresh=1")
        self.assertIsNone(next(model for model in anthropic["models"] if model["id"] == "gpt-6.1-sol")["name"])
        state = await (await self.client.get("/v1/console/state")).json()
        options = {item["value"]: item for item in state["settings"]["runtime"]["model"]["options"]}
        self.assertEqual(options["gpt-6.1-sol"]["label"], "gpt-6.1-sol")   # listing name absent: id as label
        self.assertEqual(options["gpt-6.1-sol"]["name"], "GPT-6.1-Sol")   # native fill on a Codex run
        self.assertEqual(sorted(options), ["gpt-6.1-sol", "gpt-a", "gpt-b", "kimi-k2.6"])   # usable on Responses

    async def test_unconfirmed_tier_probe_is_not_an_unknown_outcome(self):
        self.session.model_proxy.error = "timeout: no answer"
        status, body = await self.control("tier", "apply_agent_settings", effort="low", service_tier="priority")
        self.assertEqual((status, body["status"]), (200, "partial"))
        self.assertEqual((body["results"]["service_tier"]["status"], self.session.model_proxy.service_tier),
                         ("unknown", None))
        self.assertEqual(self.workbench.status, "running")
        self.assertEqual(self.events("control.status_unknown"), [])

    async def test_session_scoped_changes_answer_handoff_required_with_the_exact_diff(self):
        status, body = await self.control("switch", "apply_agent_settings", backend="claude_code", profile="anthro",
                                          wire_api="anthropic_messages", model="kimi-k2.6", effort="high",
                                          service_tier="default")
        self.assertEqual((status, body["status"], body["reason"]), (200, "handoff_required", "handoff_required"))
        self.assertEqual(body["session_fields"], ["backend", "profile", "wire_api"])
        self.assertEqual(body["diff"], {"backend": {"current": "codex", "requested": "claude_code"},
                                        "profile": {"current": "resp", "requested": "anthro"},
                                        "wire_api": {"current": "openai_responses", "requested": "anthropic_messages"},
                                        "model": {"current": "gpt-a", "requested": "kimi-k2.6"}})
        self.assertEqual((body["results"], body["handoff"]), ({}, dict(command="handoff_run", available=False)))
        self.assertEqual(self.session.model, "gpt-a")
        self.assertEqual(self.events("operator.model_changed"), [])
        event, = self.events("operator.agent_settings_applied")
        self.assertEqual(event["status"], "handoff_required")
        status, body = await self.control("same", "apply_agent_settings", backend="codex", profile="resp",
                                          model="gpt-b")
        self.assertEqual((body["status"], body["results"]["model"]["status"]), ("applied", "applied"))
        for bad in (dict(backend="robot"), dict(profile="Bad Id"), dict(wire_api="openai_chat"),
                    dict(thinking_display="full")):
            status, body = await self.control("bad-" + next(iter(bad)), "apply_agent_settings", **bad)
            self.assertEqual(status, 400, bad)
        status, body = await self.control("empty", "apply_agent_settings")
        self.assertEqual((status, body["error"].split(":")[0]), (400, "apply_agent_settings_requires_a_setting"))

    async def test_lease_is_required(self):
        response = await self.client.post("/v1/control", headers={"Origin": self.origin}, json=dict(
            request_id="nolease", command="apply_agent_settings", model="gpt-b", client_id="other", writer_token="x"))
        self.assertEqual(response.status, 409)
        self.assertEqual(self.session.model, "gpt-a")

    async def test_unknown_outcome_is_reported_like_control_dispatch(self):
        def lost(model):
            raise RuntimeError("opencode_config_reload_unconfirmed")
        self.session.set_model = lost
        status, body = await self.control("lost", "apply_agent_settings", model="gpt-b", effort="low")
        self.assertEqual((status, body["status"], body["results"]["model"]["status"]), (409, "unknown", "unknown"))
        self.assertEqual(body["results"]["effort"]["status"], "skipped")
        self.assertEqual(self.workbench.status, "unknown")


class TurnSettingsTests(AgentSettingsFixture):
    async def prompt(self, identifier):
        status, body = await self.control(identifier, "prompt", prompt="task")
        self.assertEqual((status, body["status"]), (200, "running"))
        self.session.complete()
        return body["turn_id"]

    async def test_every_prompt_records_its_settings_and_a_mid_run_change_marks_the_run_mixed(self):
        store = self.workbench.store
        first = await self.prompt("p1")
        store.set_state("runtime", "model_requests", dict(served_models={"gpt-a": 2}))
        await self.control("m", "apply_agent_settings", model="gpt-b")
        second = await self.prompt("p2")
        store.set_state("runtime", "model_requests", dict(served_models={"gpt-a": 2, "gpt-b-0731": 3}))
        turns = self.events("run.turn_settings")
        self.assertEqual([(turn["turn_id"], turn["origin"], turn["model"]) for turn in turns],
                         [(first, "prompt", "gpt-a"), (second, "prompt", "gpt-b")])
        self.assertEqual({key: turns[0][key] for key in ("backend", "profile", "provider", "wire_api", "effort",
                                                          "service_tier", "revision")},
                         dict(backend="codex", profile="resp", provider="resp", wire_api="openai_responses",
                              effort="high", service_tier="default", revision=0))
        self.assertEqual(turns[1]["served_models_at_admission"], {"gpt-a": 2})
        state = store.read_state("runtime", "agent_settings")
        self.assertEqual((state["turn_count"], state["last_turn_id"], state["mixed"]), (2, second, True))
        summary = agent_settings_summary(store)
        self.assertTrue(summary["mixed"])
        self.assertEqual((summary["initial"]["model"], summary["final"]["model"]), ("gpt-a", "gpt-b"))
        change, = summary["changes"]
        self.assertEqual((change["field"], change["previous"], change["value"], change["after_first_turn"]),
                         ("model", "gpt-a", "gpt-b", True))
        self.assertEqual([turn["served_models"] for turn in summary["turns"]], [{"gpt-a": 2}, {"gpt-b-0731": 3}])
        self.assertNoKey(summary)

    async def test_a_run_without_live_changes_is_not_mixed(self):
        await self.prompt("p1")
        await self.prompt("p2")
        summary = agent_settings_summary(self.workbench.store)
        self.assertEqual((summary["mixed"], summary["changes"], len(summary["turns"])), (False, [], 2))
        self.assertEqual(summary["initial"], summary["final"])
        await self.control("same", "apply_agent_settings", model="gpt-a")      # unchanged value: no change
        self.assertFalse(agent_settings_summary(self.workbench.store)["mixed"])

    async def test_a_change_before_the_first_turn_is_not_mixed(self):
        await self.control("early", "apply_agent_settings", effort="low")
        await self.prompt("p1")
        summary = agent_settings_summary(self.workbench.store)
        self.assertEqual((summary["mixed"], summary["changes"][0]["after_first_turn"], summary["initial"]["effort"]),
                         (False, False, "low"))


class ClaudeCodeSettingsTests(AgentSettingsFixture):
    """Claude Code run (no live effort, no service tier) through the Anthropic profile."""
    backend, profile, model = "claude_code", "anthro", "kimi-k2.6"
    capabilities = AgentCapabilities()
    effort_choices = ("low", "medium", "high", "xhigh", "max")

    async def test_effort_change_needs_a_new_session_and_model_stays_live(self):
        status, body = await self.control("effort", "apply_agent_settings", effort="low")
        self.assertEqual((body["status"], body["session_fields"], body["diff"]),
                         ("handoff_required", ["effort"], {"effort": {"current": "high", "requested": "low"}}))
        self.assertEqual(self.session.effort, "high")
        status, body = await self.control("model", "apply_agent_settings", model="glm-5", service_tier="default")
        self.assertEqual((body["status"], body["results"]["model"]["status"]), ("applied", "applied"))
        self.assertEqual(body["results"]["service_tier"], dict(status="applied", changed=False, service_tier="default"))
        _, options = await self.get("/v1/console/agent-options")
        self.assertEqual((options["current"]["service_tier"], options["reasoning"]["effort"]["applies"],
                          options["reasoning"]["service_tier"]["available"]), (None, "next_session", False))
        self.assertEqual(options["reasoning"]["model"]["id"], "glm-5")

    async def test_set_effort_for_the_next_session_does_not_mark_the_run_mixed(self):
        await self.control("p1", "prompt", prompt="task")
        self.session.complete()
        status, body = await self.control("e", "set_effort", effort="low")
        self.assertEqual((body["status"], body["applies"]), ("applied", "next_session_start"))
        current = self.workbench.agent_settings_current()
        self.assertEqual((current["effort"], current["effort_next_session"]), ("high", "low"))
        await self.control("p2", "prompt", prompt="task")
        turns = self.events("run.turn_settings")
        self.assertEqual([turn["effort"] for turn in turns], ["high", "high"])
        self.assertEqual(turns[1]["effort_next_session"], "low")
        summary = agent_settings_summary(self.workbench.store)
        self.assertEqual((summary["mixed"], summary["changes"][0]["applies"]), (False, "next_session_start"))


class OpenCodeResponsesSettingsTests(AgentSettingsFixture):
    """OpenCode 在 openai_responses 协议线下开放 service_tier 切换。"""
    backend, profile, wire_api, model = "opencode", "resp", "openai_responses", "gpt-a"
    capabilities = AgentCapabilities(live_effort=True, service_tier=True)
    effort_choices = ("none", "minimal", "low", "medium", "high", "xhigh")

    async def asyncSetUp(self):
        probe = patch.object(factory, "opencode_runtime_problem", return_value=None)
        probe.start()
        self.addCleanup(probe.stop)
        await super().asyncSetUp()
        self.session.wire_api = self.wire_api

    async def test_opencode_responses_wire_allows_service_tier(self):
        _, options = await self.get("/v1/console/agent-options")
        self.assertTrue(options["reasoning"]["service_tier"]["available"])
        status, body = await self.control("tier", "apply_agent_settings", service_tier="priority")
        self.assertEqual((status, body["status"], body["results"]["service_tier"]["status"]),
                         (200, "applied", "applied"))
        self.assertEqual(self.session.model_proxy.service_tier, "priority")


class OpenCodeAnthropicSettingsTests(AgentSettingsFixture):
    """OpenCode 在 anthropic_messages 协议线下继续拒绝 service_tier。"""
    backend, profile, wire_api, model = "opencode", "anthro", "anthropic_messages", "kimi-k2.6"
    capabilities = AgentCapabilities(live_effort=True, service_tier=False)
    effort_choices = ("low", "medium", "high", "xhigh", "max")

    async def asyncSetUp(self):
        probe = patch.object(factory, "opencode_runtime_problem", return_value=None)
        probe.start()
        self.addCleanup(probe.stop)
        await super().asyncSetUp()
        self.session.wire_api = self.wire_api

    async def test_opencode_anthropic_wire_refuses_service_tier(self):
        _, options = await self.get("/v1/console/agent-options")
        self.assertFalse(options["reasoning"]["service_tier"]["available"])
        self.assertEqual(options["reasoning"]["service_tier"]["reason"], "service_tier_unsupported_by_backend")
        status, body = await self.control("tier", "set_service_tier", tier="priority")
        self.assertEqual((status, body["status"], body["reason"]),
                         (200, "unsupported", "service_tier_unsupported_by_backend"))


class StartupTurnTests(unittest.IsolatedAsyncioTestCase):
    async def test_startup_turn_records_its_settings(self):
        temporary_dir = self.enterContext(tempfile.TemporaryDirectory())
        create, created = fake_session(AgentCapabilities())
        robot = type("Robot", (), dict(close=lambda self: None, execute=None, reset=None, evaluate=None))()
        with patch("ni_rua.apps.gateway.workbench.create_session", side_effect=create):
            workbench = Workbench(Path(temporary_dir) / "run", robot=robot, visual_review_model=None,
                                  config=Config.from_dict({"features": {"self_briefing": False, "terminal": False}}))
        session, = created
        workbench.harness = fixture_harness(workbench)
        with patch("ni_rua.apps.gateway.workbench.Terminal", return_value=terminal_double()):
            await workbench.start("Smoke instruction.")
        self.addAsyncCleanup(workbench.close)
        turn, = [event["payload"] for event in workbench.store.events_by_type(["run.turn_settings"])]
        self.assertEqual((turn["turn_id"], turn["origin"], turn["backend"], turn["model"], turn["profile"]),
                         (session.turn_id, "startup", "fake_agent", session.model, None))
        self.assertEqual(workbench.store.read_state("runtime", "agent_settings")["turn_count"], 1)
        summary = json.loads(json.dumps(agent_settings_summary(workbench.store)))
        self.assertEqual((summary["mixed"], summary["initial"]["model"], len(summary["turns"])),
                         (False, session.model, 1))

    def test_record_failure_never_breaks_an_admitted_turn(self):
        workbench = Workbench.__new__(Workbench)
        workbench.record_turn_settings("turn-1", "prompt")   # no session, no store: silently unrecorded


if __name__ == "__main__":
    unittest.main()
