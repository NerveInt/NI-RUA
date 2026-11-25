"""Merged cortex tests:
- tests/test_agent_opencode_effort.py: OpenCode 2.x reasoning effort (design 11 section 2.3): declared model variants per wire and the launch gate.
- tests/test_opencode_config_upstream.py: agent.opencode.upstream.source = "opencode_config": endpoint, key and wire from an OpenCode config file in the
- tests/test_opencode_startup_health_timeout.py: D1 regression: the first startup readiness request of a starting OpenCode server can stall for far longer than
"""

NI_RUA_TEST_TIER = "fast"

import unittest
from unittest import mock
from ni_rua.core.config import Config, ConfigError
from ni_rua.cortex.agents import factory
from ni_rua.cortex.agents.opencode.config import (EFFORT_CHOICES, EFFORT_WIRE_FIELDS, OPENCODE_INSTALL_HINT,
                                                  OPENCODE_VERSION, OpenCodeConfigError, build_config,
                                                  effort_variants, model_spec)
import json
import tempfile
from pathlib import Path
from cortex.opencode import test_opencode_session_unit as oc_unit
from ni_rua.cortex.agents.opencode.config import RELAY_PLACEHOLDER_KEY
from ni_rua.cortex.agents.opencode.upstream_file import configured_wire, read_opencode_config_upstream
import time
from ni_rua.cortex.agents.opencode import config as opencode_config
from ni_rua.cortex.agents.opencode.client import OpenCodeClient, OpenCodeTransportError
from support.fake_opencode import PIN, PASSWORD, FakeOpenCodeServer
from cortex.opencode.test_opencode_session_unit import SessionCase

# ==== tests/test_agent_opencode_effort.py ====

QUIET = {"self_briefing": False, "visual_review_model": None}
ANTHROPIC = {"wire_api": "anthropic_messages", "base_url": "http://127.0.0.1:9", "api_key_env": "K"}
RESPONSES = {"wire_api": "openai_responses", "base_url": "http://127.0.0.1:9", "api_key_env": "K"}


class EffortVariantsTests(unittest.TestCase):
    def test_anthropic_messages_variants(self):
        variants = effort_variants("anthropic_messages")
        self.assertEqual([item["id"] for item in variants], ["low", "medium", "high", "xhigh", "max"])
        for item in variants:
            # no thinking.display by default: non-Anthropic gateways refuse the field
            self.assertEqual(item, {"id": item["id"], "body": {"thinking": {"type": "adaptive"},
                                                               "output_config": {"effort": item["id"]}}})
        self.assertEqual(EFFORT_WIRE_FIELDS["anthropic_messages"], "output_config.effort")

    def test_thinking_display_is_an_explicit_knob(self):
        for display in ("summarized", "omitted"):
            for item in effort_variants("anthropic_messages", thinking_display=display):
                self.assertEqual(item["body"]["thinking"], {"type": "adaptive", "display": display})
        with self.assertRaisesRegex(OpenCodeConfigError, "^opencode_thinking_display_invalid$"):
            effort_variants("anthropic_messages", thinking_display="full")
        with self.assertRaisesRegex(OpenCodeConfigError, "^opencode_thinking_display_requires_anthropic_messages$"):
            effort_variants("openai_responses", thinking_display="summarized")

    def test_openai_responses_variants(self):
        variants = effort_variants("openai_responses")
        self.assertEqual([item["id"] for item in variants], ["none", "minimal", "low", "medium", "high", "xhigh"])
        for item in variants:
            # store:false and include:[reasoning.encrypted_content] are sent by the package itself; no summary
            self.assertEqual(item, {"id": item["id"], "body": {"reasoning": {"effort": item["id"]}}})
        self.assertEqual(EFFORT_WIRE_FIELDS["openai_responses"], "reasoning.effort")

    def test_choices_match_the_variants_and_other_wires_are_refused(self):
        self.assertEqual(set(EFFORT_CHOICES), {"anthropic_messages", "openai_responses"})
        for wire, choices in EFFORT_CHOICES.items():
            self.assertEqual(tuple(item["id"] for item in effort_variants(wire)), choices)
        for wire in ("openai_chat", "bogus"):
            with self.assertRaisesRegex(OpenCodeConfigError, f"^opencode_effort_unsupported_wire:{wire}$"):
                effort_variants(wire)

    def test_variants_are_independent_copies(self):
        first = effort_variants("anthropic_messages", thinking_display="omitted")
        first[2]["body"]["thinking"]["type"] = "changed"
        first[2]["body"]["output_config"]["effort"] = "changed"
        again = effort_variants("anthropic_messages", thinking_display="omitted")
        self.assertEqual(again[2]["body"], {"thinking": {"type": "adaptive", "display": "omitted"},
                                            "output_config": {"effort": "high"}})
        self.assertIsNot(first[0]["body"]["thinking"], first[1]["body"]["thinking"])

    def test_every_declared_model_carries_them(self):
        for wire in EFFORT_CHOICES:
            config = build_config(models=[model_spec("m"), model_spec("n")], wire_api=wire,
                                  relay_base_url="http://127.0.0.1:1", instructions_path="/x/NI-RUA.md")
            for name, entry in config["providers"]["relay"]["models"].items():
                with self.subTest(wire=wire, model=name):
                    self.assertEqual(entry["variants"], effort_variants(wire))
                    for key in ("reasoning", "options", "temperature", "tool_call", "modalities", "cost"):
                        self.assertNotIn(key, entry)       # 1.19.5 model keys are gone

    def test_responses_verbosity_in_model_entry_build_config_and_add_model(self):
        """验证 openai_responses 线写入 settings.textVerbosity，anthropic_messages 线与 verbosity=None 不写入。"""
        spec_a = model_spec("gpt-a")
        spec_b = model_spec("gpt-b")

        entry_resp = opencode_config.model_entry(spec_a, wire_api="openai_responses", verbosity="low")
        self.assertEqual(entry_resp["settings"], {"textVerbosity": "low"})
        self.assertNotIn("settings", opencode_config.model_entry(spec_a, wire_api="openai_responses", verbosity=None))
        self.assertNotIn("settings", opencode_config.model_entry(spec_a, wire_api="anthropic_messages", verbosity="low"))

        cfg = build_config(
            models=[spec_a],
            wire_api="openai_responses",
            relay_base_url="http://127.0.0.1:1",
            instructions_path="/x/NI-RUA.md",
            verbosity="medium",
        )
        self.assertEqual(cfg["providers"]["relay"]["models"]["gpt-a"]["settings"], {"textVerbosity": "medium"})
        updated = opencode_config.add_model(cfg, spec_b, wire_api="openai_responses", verbosity="medium")
        self.assertEqual(updated["providers"]["relay"]["models"]["gpt-b"]["settings"], {"textVerbosity": "medium"})

        cfg_anthro = build_config(
            models=[spec_a],
            wire_api="anthropic_messages",
            relay_base_url="http://127.0.0.1:1",
            instructions_path="/x/NI-RUA.md",
            verbosity="medium",
        )
        self.assertNotIn("settings", cfg_anthro["providers"]["relay"]["models"]["gpt-a"])


class EffortGateTests(unittest.TestCase):
    def requirements(self, effort, upstream):
        config = Config.from_dict({"features": QUIET, "model": {"effort": effort},
                                   "agent": {"backend": "opencode", "opencode": {"upstream": upstream}}})
        with mock.patch.object(factory, "opencode_runtime_problem", return_value=None) as probe:
            try:
                return factory.backend_requirements(config)
            finally:
                self.probed = probe.called

    def test_each_wire_accepts_exactly_its_tiers(self):
        all_efforts = ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra")
        for name, upstream, wire in (("anthropic", ANTHROPIC, "anthropic_messages"),
                                     ("responses", RESPONSES, "openai_responses"),
                                     ("codex_home", {"source": "codex_home"}, "openai_responses")):
            for effort in all_efforts:
                with self.subTest(wire=name, effort=effort):
                    if effort in EFFORT_CHOICES[wire]:
                        self.assertIsNone(self.requirements(effort, upstream))
                        self.assertTrue(self.probed)
                    else:
                        with self.assertRaisesRegex(ConfigError, f"^agent_effort_unsupported:opencode:{effort}: "):
                            self.requirements(effort, upstream)
                        self.assertFalse(self.probed)     # refused before the runtime probe

    def test_an_incomplete_upstream_is_reported_before_the_effort(self):
        with self.assertRaisesRegex(ConfigError, "^opencode_upstream_incomplete"):
            self.requirements("ultra", {})

    def test_openai_chat_stays_refused(self):
        # user decision D1: no relay Chat Completions adapter for OpenCode
        with self.assertRaisesRegex(ConfigError, "^opencode_upstream_wire_unsupported:openai_chat"):
            self.requirements("high", dict(RESPONSES, wire_api="openai_chat"))

    def test_runtime_refusal_names_the_2x_package(self):
        config = Config.from_dict({"features": QUIET, "model": {"effort": "high"},
                                   "agent": {"backend": "opencode", "opencode": {"upstream": ANTHROPIC}}})
        with self.assertRaises(ConfigError) as ctx:
            factory.backend_requirements(config)             # executable not set
        message = str(ctx.exception)
        self.assertTrue(message.startswith("agent_backend_unavailable:opencode: opencode_executable_not_set; "))
        self.assertIn(OPENCODE_INSTALL_HINT, message)
        self.assertIn(f"@opencode/cli-<target>@{OPENCODE_VERSION}", message)
        self.assertIn(f"opencode-{OPENCODE_VERSION}/opencode", message)
        self.assertNotIn("opencode-ai", message)

    def test_ripgrep_fallback_and_runtime_precheck(self):
        import os
        import shutil
        from ni_rua.cortex.agents.opencode import config as oc_cfg

        root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()

        def unlock():
            for path in root.rglob("*"):
                if path.is_dir() and not path.is_symlink():
                    os.chmod(path, 0o700)

        self.addCleanup(unlock)
        bin_dir = root / "opencode-bin"
        bin_dir.mkdir()
        exe = bin_dir / "opencode"
        exe.write_text(f"#!/bin/sh\necho 'opencode v{OPENCODE_VERSION}'\n", encoding="utf-8")
        exe.chmod(0o755)
        fake_home = root / "home"
        home_rg = fake_home / ".local" / "share" / "opencode" / "bin" / "rg"
        home_rg.parent.mkdir(parents=True)
        home_rg.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        home_rg.chmod(0o755)

        # 1) PATH 无 rg 时自动回退命中 ~/.local/share/opencode/bin/rg，且 build_launch 通过私有 bin 前置到子进程 PATH
        self.assertEqual(
            oc_cfg.resolve_ripgrep(env={"PATH": "", "HOME": str(fake_home)}, executable=str(exe)),
            str(home_rg),
        )
        run_dir = root / "run"
        run_dir.mkdir()
        with mock.patch.dict(os.environ, {"PATH": "", "HOME": str(fake_home)}):
            self.assertIsNone(factory.opencode_runtime_problem(str(exe)))
            launch = oc_cfg.build_launch(
                run_dir,
                executable=str(exe),
                models=[oc_cfg.model_spec("claude-sonnet-4-5")],
                wire_api="anthropic_messages",
                relay_base_url="http://127.0.0.1:9999",
                instructions_text="INSTR",
                port=4567,
                password="secret",
                managed_config_paths=[],
            )
        self.assertEqual(launch.rg_path, str(home_rg))
        private_bin = launch.dirs["private/opencode/bin"]
        self.assertTrue(launch.env["PATH"].startswith(str(private_bin) + ":"))
        self.assertEqual(shutil.which("rg", path=launch.env["PATH"]), str(private_bin / "rg"))
        self.assertEqual(os.path.realpath(private_bin / "rg"), os.path.realpath(home_rg))

        # 2) executable 同目录 rg 优先于 ~/.local/share/opencode/bin/rg
        sibling_rg = bin_dir / "rg"
        sibling_rg.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        sibling_rg.chmod(0o755)
        self.assertEqual(
            oc_cfg.resolve_ripgrep(env={"PATH": "", "HOME": str(fake_home)}, executable=str(exe)),
            str(sibling_rg),
        )

        # 3) 全部候选路径均无可执行 rg（含非可执行文件）时，opencode_runtime_problem 与 backend_requirements 提前拦截
        sibling_rg.unlink()
        home_rg.chmod(0o644)
        config = Config.from_dict({
            "features": QUIET,
            "model": {"effort": "high"},
            "agent": {"backend": "opencode", "opencode": {"executable": str(exe), "upstream": ANTHROPIC}},
        })
        with mock.patch.dict(os.environ, {"PATH": "", "HOME": str(fake_home)}):
            self.assertEqual(factory.opencode_runtime_problem(str(exe)), "opencode_ripgrep_missing")
            with self.assertRaisesRegex(ConfigError, "^agent_backend_unavailable:opencode: opencode_ripgrep_missing;"):
                factory.backend_requirements(config)

# ==== tests/test_opencode_config_upstream.py ====

KEY = "sk-test-opencode-config-key"
BASE = "https://gateway.example/gateway/v1"
UPSTREAM_QUIET = {"self_briefing": False, "visual_review_model": None}


def v1(package, base=BASE, key=KEY, provider="gw", **extra):
    options = {"apiKey": key} if key is not None else {}
    if base is not None:
        options["baseURL"] = base
    return {"$schema": "https://opencode.ai/config.json", "provider": {provider: {
        "npm": package, "name": "Gateway", "options": options, "models": {"kimi-k2.6": {}}, **extra}}}


def v2(package, base=BASE, key=KEY, provider="gw"):
    settings = {"apiKey": key} if key is not None else {}
    if base is not None:
        settings["baseURL"] = base
    return {"providers": {provider: {"package": package, "name": "Gateway", "settings": settings}}}


class FileCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()

    def write(self, value, name="opencode.local.json"):
        path = self.root / name
        path.write_text(value if isinstance(value, str) else json.dumps(value))
        return path

    def refused(self, value, code, **options):
        path = self.write(value)
        with self.assertRaises(ValueError) as raised:
            read_opencode_config_upstream(path, **options)
        message = str(raised.exception)
        self.assertTrue(message.startswith(f"{code}:{path}"), message)
        self.assertNotIn(KEY, message)
        return message


class ParseTests(FileCase):
    def test_both_shapes_and_both_packages_map_to_the_relay_convention(self):
        cases = [(v1("@ai-sdk/anthropic"), "v1", "anthropic_messages", "https://gateway.example/gateway",
                  {"x-api-key": KEY}),
                 (v1("@ai-sdk/openai"), "v1", "openai_responses", BASE, {"Authorization": "Bearer " + KEY}),
                 (v2("aisdk:@ai-sdk/anthropic"), "v2", "anthropic_messages", "https://gateway.example/gateway",
                  {"x-api-key": KEY}),
                 (v2("aisdk:@ai-sdk/openai", base=BASE + "/"), "v2", "openai_responses", BASE,
                  {"Authorization": "Bearer " + KEY}),
                 (v2("@ai-sdk/openai"), "v2", "openai_responses", BASE, {"Authorization": "Bearer " + KEY})]
        for document, shape, wire, base, headers in cases:
            with self.subTest(shape=shape, wire=wire, package=json.dumps(document)[-80:]):
                upstream = read_opencode_config_upstream(self.write(document))
                self.assertEqual((upstream.provider_id, upstream.shape, upstream.wire_api, upstream.base_url,
                                  upstream.headers), ("gw", shape, wire, base, headers))
                self.assertNotIn(KEY, repr(upstream))

    def test_sdk_default_base_urls(self):
        anthropic = read_opencode_config_upstream(self.write(v1("@ai-sdk/anthropic", base=None)))
        openai = read_opencode_config_upstream(self.write(v2("aisdk:@ai-sdk/openai", base=None)))
        self.assertEqual((anthropic.base_url, openai.base_url), ("https://api.anthropic.com", "https://api.openai.com/v1"))

    def test_provider_selection(self):
        two = {"provider": {"a": v1("@ai-sdk/anthropic")["provider"]["gw"],
                            "b": v1("@ai-sdk/openai")["provider"]["gw"]}}
        message = self.refused(two, "opencode_config_provider_ambiguous")
        self.assertIn("a, b", message)
        self.assertEqual(read_opencode_config_upstream(self.write(two), provider="b").wire_api, "openai_responses")
        self.refused(two, "opencode_config_provider_missing", provider="c")
        self.refused({"provider": {}}, "opencode_config_no_provider")
        self.refused({"model": "x"}, "opencode_config_no_provider")
        self.refused({**v1("@ai-sdk/anthropic"), **v2("aisdk:@ai-sdk/anthropic")}, "opencode_config_mixed_shapes")

    def test_wire_from_explicit_setting_or_package(self):
        compatible = v1("@ai-sdk/openai-compatible")
        self.refused(compatible, "opencode_config_wire_unknown")
        self.refused(v2(None), "opencode_config_wire_unknown")
        self.assertEqual(read_opencode_config_upstream(self.write(compatible), wire_api="openai_responses").wire_api,
                         "openai_responses")
        self.refused(compatible, "opencode_config_wire_unsupported", wire_api="openai_chat")

    def test_key_must_be_literal_and_errors_never_show_it(self):
        self.refused(v1("@ai-sdk/anthropic", key=None), "opencode_config_missing_key")
        self.refused(v2("aisdk:@ai-sdk/anthropic", key="  "), "opencode_config_missing_key")
        for key in ("{env:GATEWAY_API_KEY}", "{file:~/.secrets/key}", "prefix-{env:X}"):
            message = self.refused(v1("@ai-sdk/openai", key=key), "opencode_config_key_substitution_unsupported")
            self.assertIn("provider.gw.options.apiKey", message)
        self.refused(v2("aisdk:@ai-sdk/openai", key=KEY + " tail"), "opencode_config_invalid_key")

    def test_base_url_checks(self):
        self.refused(v1("@ai-sdk/anthropic", base="https://gateway.example/gateway"),
                     "opencode_config_base_url_unsupported")
        for base in ("ftp://gateway.example/v1", "https://user:pw@gateway.example/v1", "https://gw.example/v1?x=1",
                     "{env:BASE}/v1", 5):
            self.refused(v1("@ai-sdk/openai", base=base), "opencode_config_invalid_base_url")

    def test_unreadable_files(self):
        self.refused("// comment\n{}", "opencode_config_unreadable")
        self.refused("[]", "opencode_config_unreadable")
        with self.assertRaisesRegex(ValueError, "^opencode_config_unreadable"):
            read_opencode_config_upstream(self.root / "missing.json")
        self.refused({"provider": {"gw": "text"}}, "opencode_config_provider_invalid")

        # Malformed JSON with embedded apiKey retains no __context__.doc secret plaintext.
        secret_path = self.root / "secret_broken.json"
        secret_key = "sk-opencode-plaintext-secret-777"
        secret_path.write_text('{"providers": {"gw": {"settings": {"apiKey": "' + secret_key + '" broken')
        with self.assertRaisesRegex(ValueError, "^opencode_config_unreadable:") as caught:
            read_opencode_config_upstream(secret_path)
        self.assertIsNone(caught.exception.__context__)
        self.assertIsNone(caught.exception.__cause__)
        self.assertNotIn(secret_key, str(caught.exception))


class ConfigAndGateTests(FileCase):
    def config(self, upstream, effort="high"):
        return Config.from_dict({"agent": {"backend": "opencode", "opencode": {
            "executable": "/opt/opencode", "upstream": upstream}}, "model": {"name": "kimi-k2.6", "effort": effort},
            "features": UPSTREAM_QUIET})

    def test_relative_path_resolution_and_no_key_in_the_archived_config(self):
        self.write(v2("aisdk:@ai-sdk/anthropic"))
        config_path = self.write({"agent": {"backend": "opencode", "opencode": {"upstream": {
            "source": "opencode_config", "settings": "opencode.local.json"}}}}, name="ni-rua.local.json")
        config = Config.load(config_path, environ={})
        upstream = config.agent.opencode.upstream
        self.assertEqual(Path(upstream.settings), self.root / "opencode.local.json")
        self.assertNotIn(KEY, json.dumps(config.to_dict()))
        self.assertEqual(configured_wire(upstream), "anthropic_messages")

    def test_file_sources_are_actor_only(self):
        path = self.write(v1("@ai-sdk/openai"))
        for source in ("opencode_config", "claude_settings"):
            with self.assertRaisesRegex(ConfigError, "^evidence_upstream_source_unsupported:" + source):
                Config.from_dict({"agent": {"backend": "opencode", "evidence": {"client": "relay", "upstream": {
                    "source": source, "settings": str(path), "wire_api": "openai_responses",
                    "base_url": "https://x.example/v1"}}}})
        with mock.patch.object(factory, "claude_code_runtime_problem", return_value=None):
            with self.assertRaisesRegex(ConfigError, "^claude_code_upstream_requires_anthropic_messages"):
                factory.backend_requirements(Config.from_dict({"agent": {"backend": "claude_code", "claude_code": {
                    "upstream": {"source": "opencode_config", "settings": str(path)}}}, "features": UPSTREAM_QUIET}))

    def test_settings_path_is_required(self):
        with self.assertRaisesRegex(ConfigError, "^upstream_opencode_config_requires_settings"):
            Config.from_dict({"agent": {"opencode": {"upstream": {"source": "opencode_config"}}}})

    def test_launch_gate_reads_the_file_and_effort_follows_the_file_wire(self):
        anthropic = self.write(v1("@ai-sdk/anthropic"), name="anthropic.json")
        openai = self.write(v2("aisdk:@ai-sdk/openai"), name="openai.json")
        broken = self.write(v1("@ai-sdk/openai", key="{env:K}"), name="broken.json")
        with mock.patch.object(factory, "opencode_runtime_problem", return_value=None):
            self.assertIsNone(factory.backend_requirements(self.config(
                {"source": "opencode_config", "settings": str(anthropic)}, effort="max")))
            self.assertIsNone(factory.backend_requirements(self.config(
                {"source": "opencode_config", "settings": str(openai)}, effort="minimal")))
            with self.assertRaisesRegex(ConfigError, "^agent_effort_unsupported:opencode:minimal"):
                factory.backend_requirements(self.config({"source": "opencode_config", "settings": str(anthropic)},
                                                         effort="minimal"))
            with self.assertRaisesRegex(ConfigError, "^opencode_config_key_substitution_unsupported"):
                factory.backend_requirements(self.config({"source": "opencode_config", "settings": str(broken)}))


class SessionTests(oc_unit.SessionCase):
    def opencode_file(self, document):
        path = self.dir / "opencode.local.json"
        path.write_text(json.dumps(document))
        return {"source": "opencode_config", "settings": str(path), "provider": "gw"}

    def test_session_reads_the_file_and_the_key_stays_with_the_host(self):
        upstream = self.opencode_file(v2("aisdk:@ai-sdk/anthropic", base="http://127.0.0.1:9/v1"))
        session = self.make(self.config(upstream=upstream))
        self.assertEqual(session.wire_api, "anthropic_messages")
        session.start(oc_unit.TOOLS, "Be careful.")
        self.assertEqual(session.model_proxy.upstream_url, "http://127.0.0.1:9")
        self.assertEqual(session.model_proxy._upstream_auth_headers, {"x-api-key": KEY})
        launch = session.launch
        generated = json.loads(launch.config_path.read_text())       # the generated OpenCode 2.x config file
        self.assertEqual(generated["providers"]["relay"]["settings"]["apiKey"], RELAY_PLACEHOLDER_KEY)
        self.assertNotIn(KEY, launch.config_path.read_text())
        self.assertNotIn(KEY, json.dumps(launch.env))
        self.assertNotIn(KEY, json.dumps(session.provenance()))
        self.assertEqual(self.store.read_state("runtime", "agent")["wire_api"], "anthropic_messages")
        self.assertNotIn(KEY, json.dumps(self.store.events(limit=10000)))
        self.assertNotIn(KEY, json.dumps(self.config(upstream=upstream).to_dict()))

    def test_responses_file_on_the_session(self):
        upstream = self.opencode_file(v1("@ai-sdk/openai", base="http://127.0.0.1:9/v1/"))
        session = self.make(self.config(upstream=upstream, effort="medium"))
        self.assertEqual(session._upstream(), ("http://127.0.0.1:9/v1", {"Authorization": "Bearer " + KEY}))
        self.assertEqual(session.wire_api, "openai_responses")

# ==== tests/test_opencode_startup_health_timeout.py ====

def poll_ready(client, budget_s, per_request_s, interval_s=0.02):
    """The startup contract: poll ``health`` with a per-request timeout capped by the remaining budget until it
    returns the server info; transport errors and the 503 ``None`` mean "not ready yet". Returns the info or None
    when the budget ran out."""
    deadline = time.monotonic() + budget_s
    while True:
        remaining = deadline - time.monotonic()
        try:
            info = client.health(timeout=max(0.001, min(per_request_s, remaining)))
        except OpenCodeTransportError:
            info = None
        if info is not None:
            return info
        if time.monotonic() >= deadline:
            return None
        time.sleep(interval_s)


class ClientStartupPollTest(unittest.TestCase):
    def setUp(self):
        self.fake = FakeOpenCodeServer().start()
        self.addCleanup(self.fake.close)
        # The client's own timeout equals the whole startup budget, as in production (30 s == 30 s).
        self.client = OpenCodeClient(self.fake.url, PASSWORD, timeout_s=2.0)

    def info_requests(self):
        return [item for item in self.fake.requests if item["path"] == "/api/info"]

    def test_per_request_timeout_caps_a_stalled_first_poll(self):
        self.fake.health_stalls, self.fake.health_stall_s = 1, 3.0
        started = time.monotonic()
        with self.assertRaises(OpenCodeTransportError):
            self.client.health(timeout=0.3)
        self.assertLess(time.monotonic() - started, 1.5)
        self.assertEqual(self.client.health()["version"], PIN)

    def test_stalled_first_poll_does_not_exhaust_the_startup_budget(self):
        self.fake.health_stalls, self.fake.health_stall_s = 1, 3.0
        info = poll_ready(self.client, budget_s=2.0, per_request_s=0.3)
        self.assertEqual(info["version"], PIN)
        self.assertGreaterEqual(len(self.info_requests()), 2)
        # Negative control: when one poll may take the whole budget, the stalled request uses all of it.
        self.fake.health_stalls = 1
        self.assertIsNone(poll_ready(self.client, budget_s=0.6, per_request_s=0.6))

    def test_503_while_starting_is_not_ready_and_not_an_error(self):
        self.fake.starting_polls = 3
        for _ in range(3):
            self.assertIsNone(self.client.health(timeout=0.5))
        self.assertEqual(self.client.health(timeout=0.5)["version"], PIN)

    def test_stall_then_starting_then_ready(self):
        self.fake.health_stalls, self.fake.health_stall_s = 1, 3.0
        self.fake.starting_polls = 3          # the stalled request consumes one of them unanswered
        info = poll_ready(self.client, budget_s=2.0, per_request_s=0.3)
        self.assertEqual(info["version"], PIN)
        self.assertGreaterEqual(len(self.info_requests()), 4)


@unittest.skipUnless(opencode_config.OPENCODE_VERSION == PIN,
                     f"OpenCodeSession still targets opencode {opencode_config.OPENCODE_VERSION}; the fake speaks the "
                     f"{PIN} API (session port: work packages OC2-A/OC2-D)")
class StartupHealthStallRegressionTest(SessionCase):
    def test_stalled_first_health_request_does_not_exhaust_startup_budget(self):
        # Mirrors the production shape of the bug: the per-request timeout used for a readiness poll equals the
        # whole startup budget (30 s == 30 s in production). Here both are 2.0 s, and the fake stalls the first
        # `GET /api/info` for longer than that (3.0 s) and then reports "starting" (503) once, so a poll without a
        # per-poll timeout would consume the full 2.0 s budget on that one request and `_await_health` would raise
        # `opencode_startup_timeout` before a healthy poll ever got a chance to run.
        self.client_timeout = 2.0
        self.fake.health_stalls = 1
        self.fake.health_stall_s = 3.0
        self.fake.starting_polls = 2
        session = self.started(startup_timeout_s=2.0, health_poll_request_timeout_s=0.3, health_poll_s=0.02)
        self.assertEqual(session.state, "idle")
        info_requests = [item for item in self.fake.requests if item["path"] == "/api/info"]
        self.assertGreaterEqual(len(info_requests), 3)

if __name__ == "__main__":
    unittest.main()
