"""Merged cortex tests:
- tests/test_agent_opencode_config.py: OpenCode 2.x launch builder (``opencode/config.py``, design 11 sections 1.1-1.4, 1.10, 2.1-2.4): config document,
- tests/test_agent_opencode_effort_smoke.py: Real-binary check of OpenCode reasoning effort per wire and tier (review T11 task 2; design 11 §2.3).
"""

NI_RUA_TEST_TIER = "component"

import base64
import copy
import http.client
import json
import os
import signal
import socket
import stat
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock
from ni_rua.cortex.agents import isolation as iso
from ni_rua.cortex.agents.model_catalogue import CatalogueModel
from ni_rua.cortex.agents.opencode import config as cfg
from ni_rua.cortex.agents.opencode.client import routes_from_openapi
import shutil
from support.fake_model_servers import (FakeAnthropicServer, FakeResponsesServer, Step, ToolCall, Usage, text_step)
from ni_rua.core.config import Config
from ni_rua.core.storage.store import Store
from ni_rua.cortex.agents import factory
from ni_rua.cortex.agents.opencode.config import EFFORT_CHOICES
from support.paths import REPO_ROOT

# ==== tests/test_agent_opencode_config.py ====

EXECUTABLE = os.environ.get("NI_RUA_OPENCODE_EXECUTABLE")
PIN = cfg.OPENCODE_VERSION


def fake_binary(directory: Path, output: str = f"opencode v{PIN}") -> str:
    path = directory / "opencode"
    path.write_text(f"#!/bin/sh\necho '{output}'\n")
    path.chmod(0o755)
    return str(path)


def spec(model_id="claude-sonnet-4-5", **kw):
    return cfg.model_spec(model_id, **kw)


class LaunchCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()
        self.exe = fake_binary(self.root)
        self.rg = self.exe  # any executable file stands in for rg
        self.addCleanup(self._unlock)

    def _unlock(self):
        for path in self.run_dir.rglob("*"):
            if path.is_dir() and not path.is_symlink():
                os.chmod(path, 0o700)

    def launch(self, **kw):
        args = dict(executable=self.exe, models=[spec()], wire_api="anthropic_messages",
                    relay_base_url="http://127.0.0.1:9999", instructions_text="INSTR ni_rua_", port=4567,
                    password="secret-test-password", mcp_url="http://127.0.0.1:8888/mcp",
                    mcp_headers={"Authorization": "Bearer t"}, ripgrep=self.rg, managed_config_paths=[],
                    base_env={"ANTHROPIC_API_KEY": "host", "TZ": "UTC"})
        args.update(kw)
        return cfg.build_launch(self.run_dir, **args)


class BuildTests(LaunchCase):
    def test_config_document_is_the_native_v2_shape(self):
        launch = self.launch()
        c = launch.config
        self.assertEqual(json.loads(launch.config_path.read_text()), c)
        self.assertEqual((c["update"], c["share"], c["snapshots"]), ("disable", "disabled", False))
        self.assertEqual(c["model"], "relay/claude-sonnet-4-5")
        self.assertEqual(c["instructions"], [str(launch.instructions_path)])
        self.assertEqual(c["tool_output"], {"max_lines": 100000, "max_bytes": 16000000})
        self.assertEqual(c["media"], {"image": {"auto_resize": False, "max_width": 1000000, "max_height": 1000000,
                                                "max_base64_bytes": 1073741824}})
        self.assertEqual(c["experimental"], {"todo": True})
        # denies only, after "read * allow" (closes the shipped read *.env -> ask); never a global allow (F7)
        self.assertEqual(c["permissions"], [
            {"action": "read", "resource": "*", "effect": "allow"},
            {"action": "webfetch", "resource": "*", "effect": "deny"},
            {"action": "websearch", "resource": "*", "effect": "deny"},
            {"action": "question", "resource": "*", "effect": "deny"}])
        self.assertEqual(c["policies"], [{"action": "provider.use", "resource": "*", "effect": "deny"},
                                         {"action": "provider.use", "resource": "relay", "effect": "allow"}])
        provider = c["providers"]["relay"]
        self.assertEqual(provider["package"], "@opencode/ai/providers/anthropic-compatible")
        self.assertEqual(provider["name"], "relay")
        self.assertEqual(provider["settings"], {"baseURL": "http://127.0.0.1:9999/v1", "apiKey": "ni-rua-local-relay",
                                                "timeout": False, "headerTimeout": False, "chunkTimeout": False})
        model = provider["models"]["claude-sonnet-4-5"]
        self.assertEqual(model["name"], "claude-sonnet-4-5")
        self.assertEqual(model["capabilities"], {"tools": True, "input": ["text", "image"], "output": ["text"]})
        self.assertEqual(model["limit"], {"context": 200000, "output": 32000})
        self.assertEqual(model["variants"], cfg.effort_variants("anthropic_messages"))
        self.assertEqual(c["mcp"], {"servers": {"ni_rua": {
            "type": "remote", "url": "http://127.0.0.1:8888/mcp", "headers": {"Authorization": "Bearer t"},
            "oauth": False, "codemode": False, "timeout": {"startup": 30000, "catalog": 30000, "execution": 600000}}}})
        # 1.19.5 keys are gone; lsp / formatter / compaction stay at the native defaults
        for key in ("provider", "permission", "enabled_providers", "small_model", "agent", "autoupdate", "snapshot",
                    "compaction", "lsp", "formatter"):
            self.assertNotIn(key, c)

    def test_wires_subagents_models_and_refusals(self):
        responses = self.launch(wire_api="openai_responses").config["providers"]["relay"]
        self.assertEqual(responses["package"], "@opencode/ai/providers/openai-compatible/responses")
        self.assertEqual(responses["models"]["claude-sonnet-4-5"]["variants"], cfg.effort_variants("openai_responses"))
        alone = self.launch(subagents=False).config
        self.assertEqual(alone["permissions"][-1], {"action": "subagent", "resource": "*", "effect": "deny"})
        self.assertNotIn("subagent", json.dumps(self.launch().config["permissions"]))
        two = self.launch(models=[spec("a"), spec("b", output_limit_tokens=7)]).config
        self.assertEqual(two["model"], "relay/a")
        self.assertEqual(cfg.declared_model_ids(two), ["a", "b"])
        self.assertEqual(two["providers"]["relay"]["models"]["b"]["limit"]["output"], 7)
        self.assertNotIn("mcp", self.launch(mcp_url=None).config)
        for kw, code in (({"wire_api": "bogus"}, "opencode_unsupported_wire:bogus"),
                         ({"wire_api": "openai_chat"}, "opencode_unsupported_wire:openai_chat"),   # user decision D1
                         ({"models": []}, "opencode_model_required"),
                         ({"models": [spec("a"), spec("a")]}, "opencode_model_declared_twice")):
            with self.subTest(kw=kw), self.assertRaisesRegex(cfg.OpenCodeConfigError, f"^{code}$"):
                self.launch(**kw)

    def test_thinking_display_reaches_every_declared_model(self):
        c = self.launch(models=[spec("a"), spec("b")], thinking_display="summarized").config
        for entry in c["providers"]["relay"]["models"].values():
            self.assertEqual({item["body"]["thinking"]["display"] for item in entry["variants"]}, {"summarized"})
        with self.assertRaisesRegex(cfg.OpenCodeConfigError, "^opencode_thinking_display_requires_anthropic_messages$"):
            self.launch(wire_api="openai_responses", thinking_display="summarized")

    def test_files_dirs_env(self):
        launch = self.launch()
        self.assertEqual(launch.argv, (self.exe, "serve", "--hostname", "127.0.0.1", "--port", "4567"))
        self.assertEqual(launch.cwd, self.run_dir.resolve() / "actor")
        text = launch.instructions_path
        self.assertEqual(text.read_text(), "INSTR ni_rua_")
        self.assertEqual(stat.S_IMODE(text.stat().st_mode), 0o444)
        config_dir = launch.dirs[cfg.CONFIG_DIR]
        self.assertEqual(config_dir, self.run_dir.resolve() / "private/opencode/xdg/config/opencode")
        self.assertEqual(launch.config_path, config_dir / "opencode.json")
        self.assertEqual(stat.S_IMODE(config_dir.stat().st_mode), 0o555)
        self.assertEqual(stat.S_IMODE(launch.config_path.stat().st_mode), 0o444)
        # nothing but the config file (2.2.7 writes nothing into the config dir; no .gitignore workaround needed)
        self.assertEqual([path.name for path in config_dir.iterdir()], ["opencode.json"])
        self.assertNotIn("private/opencode/cfgdir", launch.dirs)
        self.assertEqual(stat.S_IMODE(launch.dirs["private/opencode/xdg/data"].stat().st_mode), 0o700)
        bin_dir = launch.dirs["private/opencode/bin"]
        self.assertEqual(os.path.realpath(bin_dir / "rg"), os.path.realpath(self.rg))
        env = launch.env
        self.assertTrue(env["PATH"].startswith(str(bin_dir) + ":"))
        self.assertEqual(env["HOME"], str(launch.cwd))
        self.assertEqual(env["TMPDIR"], str(launch.dirs["private/tmp"]))
        self.assertEqual(env["OPENCODE_PASSWORD"], "secret-test-password")
        # one config dir: OPENCODE_CONFIG_DIR replaces the global dir and equals $XDG_CONFIG_HOME/opencode
        self.assertEqual(env["OPENCODE_CONFIG_DIR"], str(config_dir))
        self.assertEqual(env["XDG_CONFIG_HOME"], str(config_dir.parent))
        self.assertEqual(env["XDG_DATA_HOME"], str(launch.dirs["private/opencode/xdg/data"]))
        for flag in ("OPENCODE_DISABLE_PROJECT_CONFIG", "OPENCODE_DISABLE_CLAUDE_CODE", "OPENCODE_DISABLE_AUTOUPDATE",
                     "OPENCODE_DISABLE_MODELS_FETCH"):
            self.assertEqual(env[flag], "1")
        for gone in ("OPENCODE_PURE", "OPENCODE_DISABLE_DEFAULT_PLUGINS", "OPENCODE_DISABLE_EXTERNAL_SKILLS",
                     "OPENCODE_DISABLE_LSP_DOWNLOAD", "OPENCODE_DISABLE_EMBEDDED_WEB_UI", "OPENCODE_CONFIG_CONTENT",
                     "OPENCODE_SERVER_PASSWORD", "ANTHROPIC_API_KEY", "OPENCODE_TEST_MANAGED_CONFIG_DIR"):
            self.assertNotIn(gone, env)
        self.assertEqual(env["TZ"], "UTC")
        self.assertEqual(launch.models, (spec(),))
        self.assertEqual((launch.wire_api, launch.thinking_display), ("anthropic_messages", None))
        marked = self.launch(instance_id="agent_x").env
        self.assertEqual(marked[iso.AGENT_MARKER_ENV], "agent_x")

    def test_relaunch_on_same_run_dir_replaces_the_read_only_config(self):
        self.launch()
        again = self.launch(instructions_text="CHANGED", models=[spec("other")])
        self.assertEqual(again.instructions_path.read_text(), "CHANGED")
        self.assertEqual(json.loads(again.config_path.read_text())["model"], "relay/other")
        self.assertEqual(stat.S_IMODE(again.dirs[cfg.CONFIG_DIR].stat().st_mode), 0o555)

    def test_sandbox_launch(self):
        launch = self.launch(sandbox=True)
        self.assertEqual(launch.argv[0], cfg.SANDBOX_EXECUTABLE)
        self.assertEqual(launch.rg_path, cfg.SANDBOX_RIPGREP)
        self.assertFalse((launch.dirs["private/opencode/bin"] / "rg").exists())
        for name in ("PATH", "HOME", "TMPDIR"):
            self.assertNotIn(name, launch.env)
        spec_ = cfg.sandbox_spec(launch, harness_source="/snap")
        root = self.run_dir.resolve() / "private/opencode/xdg"
        self.assertEqual(spec_["state_dirs"], [(root / name, root / name) for name in ("data", "cache", "state")])
        self.assertEqual(spec_["executables"], [(launch.executable, cfg.SANDBOX_EXECUTABLE)])
        self.assertEqual(spec_["read_paths"], [(root / "config", root / "config"),
                                               (launch.instructions_path, launch.instructions_path),
                                               ("/snap", cfg.SANDBOX_HARNESS_SOURCE)])
        self.assertEqual(spec_["env"], launch.env)
        self.assertIsNot(spec_["env"], launch.env)

    def test_version_mismatch_and_probe(self):
        self.assertEqual(cfg.probe_cli_version(self.exe), PIN)
        for index, output in enumerate(("opencode v2.2.6", "1.19.5", "opencode v2.2.7-beta.1", "")):
            directory = self.root / f"x{index}"
            directory.mkdir()
            with self.subTest(output=output), self.assertRaises(iso.VersionMismatch) as ctx:
                self.launch(executable=fake_binary(directory, output))
            self.assertEqual(str(ctx.exception), "opencode_version_mismatch")
        with self.assertRaisesRegex(cfg.OpenCodeConfigError, "^opencode_executable_invalid$"):
            self.launch(executable="relative/opencode")

    def test_probe_cli_version_uses_isolated_home(self):
        # Without HOME/XDG_*, OpenCode resolves the invoking user's real passwd home and writes there; the probe runs
        # with a private, disposable HOME, every XDG_*_HOME and TMPDIR (Bun extracts native libraries there).
        captured = {}

        def fake_run(argv, **kwargs):
            captured.update(kwargs, argv=argv)
            return subprocess.CompletedProcess(argv, 0, stdout=f"opencode v{PIN}\n", stderr="")

        with mock.patch.object(cfg.subprocess, "run", side_effect=fake_run):
            self.assertEqual(cfg.probe_cli_version(self.exe), PIN)
        self.assertEqual(captured["argv"], [self.exe, "--version"])
        env = captured["env"]
        home = env["HOME"]
        self.assertTrue(os.path.isabs(home))
        self.assertNotIn(home, (os.path.expanduser("~"), os.environ.get("HOME", "")))
        self.assertEqual(env["TMPDIR"], home)
        for key in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME"):
            self.assertTrue(env[key].startswith(home + os.sep), f"{key}={env[key]!r} not under {home!r}")
        self.assertFalse(os.path.isdir(home))
        with mock.patch.object(cfg.subprocess, "run", side_effect=subprocess.TimeoutExpired("x", 1)):
            with self.assertRaisesRegex(cfg.OpenCodeConfigError, "^opencode_version_probe_failed:TimeoutExpired$"):
                cfg.probe_cli_version(self.exe)

    def test_managed_config_check(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(cfg.OpenCodeConfigError) as ctx:
                self.launch(managed_config_paths=[directory])
            self.assertEqual(str(ctx.exception), "opencode_managed_config_present")
        cfg.check_managed_config([str(self.root / "absent")])
        with mock.patch.object(cfg.os.path, "lexists",
                               side_effect=lambda p: p.endswith("/alice/ai.opencode.managed.plist")):
            with self.assertRaises(cfg.OpenCodeConfigError):
                cfg.check_managed_config(user="alice")
        self.assertEqual(cfg.MANAGED_CONFIG_PATHS, ("/etc/opencode", "/Library/Application Support/opencode",
                                                    "/Library/Managed Preferences/ai.opencode.managed.plist"))

    def test_ripgrep_missing(self):
        with mock.patch.dict(os.environ, {"HOME": str(self.root / "nothing")}):
            with self.assertRaises(cfg.OpenCodeConfigError) as ctx:
                cfg.resolve_ripgrep(search_path=str(self.root / "nothing"))
        self.assertEqual(str(ctx.exception), "opencode_ripgrep_missing")
        self.assertEqual(cfg.resolve_ripgrep(self.rg), self.rg)

    def test_tui_attach_command(self):
        launch = self.launch()
        argv, env = cfg.tui_attach_command(launch, server_url="http://127.0.0.1:4567", session_id="ses_abc",
                                           directory=launch.cwd)
        self.assertEqual(argv, [launch.executable, "--server", "http://127.0.0.1:4567", "--session", "ses_abc",
                                str(launch.cwd)])
        self.assertNotIn("attach", argv)
        self.assertEqual(env, launch.env)
        self.assertIsNot(env, launch.env)
        self.assertNotIn("secret-test-password", " ".join(argv))
        for bad in ("", "  ", None):
            with self.assertRaisesRegex(ValueError, "^opencode_tui_session_required$"):
                cfg.tui_attach_command(launch, server_url="http://x", session_id=bad, directory=launch.cwd)


class VersionAndHealthTests(unittest.TestCase):
    def test_parse_cli_version(self):
        for output, expected in ((f"opencode v{PIN}\n", PIN), (PIN, PIN), (f"v{PIN}", PIN),
                                 (f"\n\n  opencode v{PIN}  \nextra", PIN), ("opencode v2.3.0-beta.1", "2.3.0-beta.1"),
                                 ("", ""), (None, ""), ("garbage text", "garbage text"),
                                 ("opencode version 2.2.7", "opencode version 2.2.7")):
            with self.subTest(output=output):
                self.assertEqual(cfg.parse_cli_version(output), expected)

    def test_check_health_version_reads_api_info(self):
        info = {"version": PIN, "pid": 123, "urls": ["http://127.0.0.1:1"], "paths": {"tmp": "/t"}}
        self.assertEqual(cfg.check_health_version(info), PIN)
        for bad in ({"version": "1.19.5", "pid": 1}, {"version": f"v{PIN}"}, {"pid": 1}, {}, None, [PIN]):
            with self.subTest(info=bad), self.assertRaises(iso.VersionMismatch):
                cfg.check_health_version(bad)


class LimitsTests(unittest.TestCase):
    def test_resolve_limits_precedence(self):
        big = CatalogueModel(id="m", context_window=262144, max_output_tokens=131072)
        small = CatalogueModel(id="m", context_window=64000, max_output_tokens=16384)
        exact = CatalogueModel(id="m", max_output_tokens=32000)
        empty = CatalogueModel(id="m")
        cases = (
            ({}, (200000, 32000, "fallback", "fallback")),
            ({"catalogue_model": None}, (200000, 32000, "fallback", "fallback")),
            ({"catalogue_model": empty}, (200000, 32000, "fallback", "fallback")),
            ({"catalogue_model": big}, (262144, 32000, "catalogue", "catalogue_capped")),
            ({"catalogue_model": small}, (64000, 16384, "catalogue", "catalogue")),
            ({"catalogue_model": exact}, (200000, 32000, "fallback", "catalogue")),
            # an explicit value is used exactly, above the ceiling and above or below the catalogue
            ({"output_limit_tokens": 100000, "catalogue_model": small}, (64000, 100000, "catalogue", "explicit")),
            ({"output_limit_tokens": 8, "catalogue_model": big}, (262144, 8, "catalogue", "explicit")),
            ({"context_limit_tokens": 1000000, "output_limit_tokens": 32768},
             (1000000, 32768, "explicit", "explicit")),
        )
        for kw, (context, output, context_source, output_source) in cases:
            with self.subTest(kw=kw):
                self.assertEqual(cfg.resolve_limits(**kw), {"context": context, "output": output, "source": {
                    "context": context_source, "output": output_source}})

    def test_invalid_explicit_limits_are_refused(self):
        for kw in ({"context_limit_tokens": 0}, {"output_limit_tokens": -1}, {"output_limit_tokens": True},
                   {"context_limit_tokens": 1.5}, {"output_limit_tokens": "32000"}):
            with self.subTest(kw=kw), self.assertRaisesRegex(cfg.OpenCodeConfigError, "^opencode_limit_invalid:"):
                cfg.resolve_limits(**kw)

    def test_catalogue_values_that_are_not_positive_ints_are_ignored(self):
        odd = mock.Mock(context_window=0, max_output_tokens=True)
        self.assertEqual(cfg.resolve_limits(catalogue_model=odd)["source"], {"context": "fallback", "output": "fallback"})

    def test_model_spec_and_input_modalities(self):
        self.assertEqual(spec("m").input_modalities, ["text", "image"])
        self.assertEqual(spec("m", catalogue_model=CatalogueModel(id="m", image_input=True)).input_modalities,
                         ["text", "image"])
        text_only = spec("m", catalogue_model=CatalogueModel(id="m", image_input=False))
        self.assertEqual(text_only.input_modalities, ["text"])
        self.assertEqual(text_only.record(), {"id": "m", "limits": cfg.resolve_limits(), "input": ["text"]})
        self.assertEqual(cfg.model_entry(text_only, wire_api="anthropic_messages")["capabilities"]["input"], ["text"])
        self.assertEqual(spec("moonshotai/kimi-k2.6").id, "moonshotai/kimi-k2.6")
        for bad in ("", "a#b", "a b", " a", "a\n", None, 3):
            with self.subTest(model=bad), self.assertRaisesRegex(cfg.OpenCodeConfigError, "^opencode_model_id_invalid$"):
                spec(bad)


class RewriteTests(LaunchCase):
    def test_add_model_is_pure_and_declares_variants(self):
        launch = self.launch(thinking_display="omitted")
        original = copy.deepcopy(launch.config)
        extra = spec("added", catalogue_model=CatalogueModel(id="added", image_input=False, context_window=64000,
                                                              max_output_tokens=16384))
        updated = cfg.add_model(launch.config, extra, wire_api="anthropic_messages", thinking_display="omitted")
        self.assertEqual(launch.config, original)
        self.assertEqual(cfg.declared_model_ids(updated), ["claude-sonnet-4-5", "added"])
        self.assertEqual(updated["model"], original["model"])
        self.assertEqual(updated["providers"]["relay"]["models"]["added"], {
            "name": "added", "capabilities": {"tools": True, "input": ["text"], "output": ["text"]},
            "limit": {"context": 64000, "output": 16384},
            "variants": cfg.effort_variants("anthropic_messages", thinking_display="omitted")})
        replaced = cfg.add_model(updated, spec("added", output_limit_tokens=5), wire_api="anthropic_messages")
        self.assertEqual(cfg.declared_model_ids(replaced), ["claude-sonnet-4-5", "added"])
        self.assertEqual(replaced["providers"]["relay"]["models"]["added"]["limit"]["output"], 5)
        for broken in ({}, {"providers": {}}, {"providers": {"relay": {"models": []}}}):
            with self.assertRaisesRegex(cfg.OpenCodeConfigError, "^opencode_config_provider_missing$"):
                cfg.add_model(broken, extra, wire_api="anthropic_messages")

    def test_write_config_replaces_atomically_and_restores_the_read_only_dir(self):
        launch = self.launch()
        directory = launch.dirs[cfg.CONFIG_DIR]
        updated = cfg.add_model(launch.config, spec("added"), wire_api="anthropic_messages")
        cfg.write_config(launch.config_path, updated)
        self.assertEqual(json.loads(launch.config_path.read_text()), updated)
        self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o555)
        self.assertEqual(stat.S_IMODE(launch.config_path.stat().st_mode), 0o444)
        self.assertEqual([path.name for path in directory.iterdir()], ["opencode.json"])

    def test_write_config_failure_keeps_the_old_file_and_leaves_no_temporary(self):
        launch = self.launch()
        directory = launch.dirs[cfg.CONFIG_DIR]
        before = launch.config_path.read_text()
        with mock.patch.object(cfg.os, "replace", side_effect=OSError("disk")):
            with self.assertRaisesRegex(OSError, "disk"):
                cfg.write_config(launch.config_path, {"x": 1})
        self.assertEqual(launch.config_path.read_text(), before)
        self.assertEqual([path.name for path in directory.iterdir()], ["opencode.json"])
        self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o555)

    def test_write_config_in_a_writable_dir_keeps_its_mode(self):
        directory = self.root / "plain"
        directory.mkdir(mode=0o750)
        os.chmod(directory, 0o750)
        cfg.write_config(directory / "opencode.json", {"a": "é"})
        self.assertEqual(json.loads((directory / "opencode.json").read_text(encoding="utf-8")), {"a": "é"})
        self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o750)


class ServedModelsTests(unittest.TestCase):
    @staticmethod
    def config(*specs, wire="anthropic_messages", display=None):
        return cfg.build_config(models=list(specs) or [spec("m")], wire_api=wire, relay_base_url="http://127.0.0.1:1",
                                instructions_path="/x/NI-RUA.md", thinking_display=display)

    @staticmethod
    def served(config):
        """What ``GET /api/model`` reported for a generated config (shape measured on 2.2.7)."""
        out = []
        for model_id, entry in config["providers"]["relay"]["models"].items():
            out.append({"id": model_id, "modelID": model_id, "providerID": "relay", "name": model_id,
                        "capabilities": dict(entry["capabilities"], temperature=True), "limit": dict(entry["limit"]),
                        "variants": [dict(item) for item in entry["variants"]], "status": "active", "enabled": True})
        return out

    def problems(self, served, config):
        with self.assertRaisesRegex(cfg.OpenCodeConfigError, "^opencode_served_models_mismatch$") as ctx:
            cfg.check_served_models(served, config)
        return ctx.exception.problems

    def test_exact_declaration_passes(self):
        for wire in cfg.EFFORT_CHOICES:
            config = self.config(spec("a"), spec("b", catalogue_model=CatalogueModel(id="b", image_input=False)),
                                 wire=wire)
            self.assertIsNone(cfg.check_served_models(self.served(config), config))

    def test_every_difference_is_reported(self):
        config = self.config(spec("a"), spec("b"))
        served = self.served(config)
        self.assertEqual(self.problems(served[:1], config), ["missing:b"])
        self.assertEqual(self.problems(served + [{"id": "big-pickle", "providerID": "opencode"}], config),
                         ["unexpected:opencode/big-pickle"])
        self.assertEqual(self.problems(served + [dict(served[0], id="c")], config), ["unexpected:relay/c"])
        changed = copy.deepcopy(served)
        changed[0]["limit"]["output"] = 4096
        changed[1]["capabilities"]["input"] = ["text"]
        self.assertEqual(self.problems(changed, config), ["input:b", "limit:a"])
        extra_variant = copy.deepcopy(served)
        extra_variant[0]["variants"].append({"id": "default", "body": {}})
        self.assertEqual(self.problems(extra_variant, config), ["variants:a"])
        displayed = copy.deepcopy(served)
        displayed[1]["variants"][0]["body"]["thinking"]["display"] = "summarized"
        self.assertEqual(self.problems(displayed, config), ["variants:b"])
        self.assertEqual(self.problems(None, config), ["malformed", "missing:a", "missing:b"])
        self.assertEqual(self.problems([1] + served, config), ["malformed"])


class ToolsAndProvenanceTests(LaunchCase):
    def test_native_tool_ids(self):
        names = ["ni_rua_observe", "shell", "read", "todowrite", "read", None, 3, "subagent", "ni_rua_wait_s"]
        self.assertEqual(cfg.native_tool_ids(names), ["read", "shell", "subagent", "todowrite"])
        self.assertEqual(cfg.native_tool_ids(["x_a", "b"], tool_name_prefix="x_"), ["b"])
        self.assertEqual(cfg.native_tool_ids([]), [])

    def test_provenance_block(self):
        primary = spec("claude-sonnet-4-5", catalogue_model=CatalogueModel(id="c", context_window=262144,
                                                                            max_output_tokens=131072))
        launch = self.launch(models=[primary], thinking_display="summarized")
        block = cfg.provenance_block(launch, route_list_sha256="abc", wire_api="anthropic_messages",
                                     model="claude-sonnet-4-5", effort="high", instructions_sha256="d",
                                     enabled_tools=["write", "shell"])
        self.assertEqual(block["cli"], {"version": PIN, "executable": launch.executable, "sha256": launch.sha256})
        self.assertEqual((block["kind"], block["sdk"], block["native_model_id"], block["provider_alias"]),
                         ("opencode", None, "claude-sonnet-4-5", "relay"))
        self.assertEqual(block["provider_package"], "@opencode/ai/providers/anthropic-compatible")
        self.assertEqual(block["route_list_sha256"], "abc")
        self.assertEqual(block["native_effort"], {
            "variant": "high", "body": {"thinking": {"type": "adaptive", "display": "summarized"},
                                        "output_config": {"effort": "high"}},
            "thinking_display": "summarized", "declared_variants": ["low", "medium", "high", "xhigh", "max"],
            "wire_field": "output_config.effort"})
        self.assertEqual(block["effective_limits"], {"context": 262144, "output": 32000,
                                                     "source": {"context": "catalogue", "output": "catalogue_capped"}})
        self.assertEqual(block["declared_models"], [primary.record()])
        reasons = {item["name"]: item["reason"] for item in block["native_tools"]["disabled"]}
        self.assertEqual(reasons, {"webfetch": "online_access", "websearch": "online_access",
                                   "question": "waits_for_human"})
        self.assertEqual(block["native_tools"]["enabled"], ["shell", "write"])
        settings = block["native_settings"]
        self.assertEqual(settings["flags"], {"OPENCODE_DISABLE_PROJECT_CONFIG": "1", "OPENCODE_DISABLE_CLAUDE_CODE": "1",
                                             "OPENCODE_DISABLE_AUTOUPDATE": "1", "OPENCODE_DISABLE_MODELS_FETCH": "1"})
        self.assertEqual(settings["config_channel"], "config_file")
        self.assertEqual(settings["media_image"], cfg.MEDIA_IMAGE)
        self.assertEqual(settings["experimental"], {"todo": True})
        self.assertIs(settings["mcp_codemode"], False)
        self.assertEqual(settings["provider_timeouts"], {"timeout": False, "chunkTimeout": False, "headerTimeout": False})
        for gone in ("compaction_prune", "title_agent", "config_dirs_read_only"):
            self.assertNotIn(gone, settings)
        text = json.dumps(block)
        self.assertNotIn("secret-test-password", text)
        self.assertNotIn("Bearer", text)

    def test_provenance_after_a_model_switch_and_without_effort(self):
        launch = self.launch(subagents=False)
        added = spec("added", output_limit_tokens=777)
        block = cfg.provenance_block(launch, route_list_sha256=None, wire_api="openai_responses", model="added",
                                     subagents=False, models=list(launch.models) + [added])
        self.assertIsNone(block["native_effort"])
        self.assertIsNone(block["native_tools"]["enabled"])
        self.assertIn({"name": "subagent", "reason": "subagents_disabled"}, block["native_tools"]["disabled"])
        self.assertEqual(block["effective_limits"]["output"], 777)
        self.assertEqual([item["id"] for item in block["declared_models"]], ["claude-sonnet-4-5", "added"])
        self.assertEqual(block["provider_package"], "@opencode/ai/providers/openai-compatible/responses")
        effort = cfg.provenance_block(launch, route_list_sha256=None, wire_api="openai_responses", model="unknown",
                                      effort="none")
        self.assertIsNone(effort["effective_limits"])
        self.assertEqual(effort["native_effort"]["body"], {"reasoning": {"effort": "none"}})
        self.assertIsNone(effort["native_effort"]["thinking_display"])


class RouteListTests(unittest.TestCase):
    def test_checked_in_list(self):
        routes = cfg.load_expected_routes()
        self.assertEqual(len(routes), 170)
        self.assertEqual(routes, sorted(set(routes)))
        for needed in ("GET /api/info", "POST /api/session", "GET /api/session/{sessionID}",
                       "POST /api/session/{sessionID}/prompt", "POST /api/session/{sessionID}/interrupt",
                       "POST /api/session/{sessionID}/model", "GET /api/session/active",
                       "GET /api/session/{sessionID}/message", "GET /api/event", "GET /api/model", "GET /api/mcp",
                       "POST /api/location/reload"):
            self.assertIn(needed, routes)
        for gone in ("GET /global/health", "GET /doc", "POST /session/{sessionID}/prompt_async"):
            self.assertNotIn(gone, routes)
        data = json.loads(Path(cfg._ROUTES_FILE).read_text())
        self.assertEqual((data["opencode_version"], data["route_count"]), (PIN, 170))
        self.assertEqual(data["source"], "GET /openapi.json of the pinned binary")

    def test_digest_and_mismatch(self):
        routes = cfg.load_expected_routes()
        digest = cfg.check_route_list(routes)
        self.assertEqual(digest, cfg.route_list_digest(reversed(routes)))
        self.assertEqual(len(digest), 64)
        with self.assertRaises(cfg.OpenCodeConfigError) as ctx:
            cfg.check_route_list(routes[1:] + ["GET /new"])
        self.assertEqual(str(ctx.exception), "opencode_route_list_mismatch")
        self.assertEqual((ctx.exception.added, ctx.exception.removed), (["GET /new"], [routes[0]]))

    def test_package_data_declared(self):
        text = (REPO_ROOT / "pyproject.toml").read_text()
        self.assertIn('"ni_rua.cortex.agents.opencode" = ["opencode_routes.json"]', text)


def _free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@unittest.skipUnless(EXECUTABLE, "NI_RUA_OPENCODE_EXECUTABLE not set (pinned opencode binary)")
class RealBinaryTests(unittest.TestCase):
    """The generated config launched with the pinned binary (design 11 section 5, OC2-A acceptance). Raw HTTP, not
    ``OpenCodeClient``, so this test checks the config module alone."""

    def api(self, method, path, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        headers = {"Authorization": "Basic " + base64.b64encode(b"opencode:pw-real").decode()}
        if body is not None:
            headers["Content-Type"] = "application/json"
        try:
            conn.request(method, path, body=json.dumps(body) if body is not None else None, headers=headers)
            response = conn.getresponse()
            raw = response.read()
        finally:
            conn.close()
        return response.status, (json.loads(raw) if raw.strip() else None)

    def test_serve_with_generated_config(self):
        from support.fake_model_servers import FakeAnthropicServer, text_step
        fake = FakeAnthropicServer([text_step("A1"), text_step("A2")], default=text_step("D")).start()
        self.addCleanup(fake.close)
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self.port = _free_port()
            marker = f"oc2a-config-test-{os.getpid()}"
            launch = cfg.build_launch(
                run_dir, executable=EXECUTABLE, models=[spec("probe-model")], wire_api="anthropic_messages",
                relay_base_url=fake.base_url, instructions_text="x", port=self.port, password="pw-real",
                instance_id=marker)
            log = open(run_dir / "serve.log", "wb")
            process = subprocess.Popen(launch.argv, cwd=launch.cwd, env=launch.env, stdout=log,
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            try:
                deadline, info = time.monotonic() + 30, None
                while time.monotonic() < deadline and info is None:
                    self.assertIsNone(process.poll(), (run_dir / "serve.log").read_text()[-500:])
                    try:
                        status, body = self.api("GET", "/api/info")
                        info = body if status == 200 else None
                    except OSError:
                        pass
                    if info is None:
                        time.sleep(0.05)
                self.assertEqual(cfg.check_health_version(info), PIN)
                status, document = self.api("GET", "/openapi.json")
                self.assertEqual(cfg.check_route_list(routes_from_openapi(document)),
                                 cfg.route_list_digest(cfg.load_expected_routes()))
                status, models = self.api("GET", "/api/model")
                cfg.check_served_models(models["data"], launch.config)       # only relay/probe-model, as declared
                status, created = self.api("POST", "/api/session", {
                    "title": "ni-rua-actor", "model": {"id": "probe-model", "providerID": "relay", "variant": "high"}})
                session_id = created["data"]["id"]
                self.api("POST", f"/api/session/{session_id}/prompt", {"text": "go"})
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline and not fake.generation_requests():
                    time.sleep(0.05)
                first = fake.generation_requests()[0].json
                tools = cfg.native_tool_ids([tool["name"] for tool in first["tools"]])
                for denied in ("webfetch", "websearch", "question"):
                    self.assertNotIn(denied, tools)
                for kept in ("todowrite", "subagent", "shell", "read", "skill", "execute"):
                    self.assertIn(kept, tools)
                self.assertEqual((first["max_tokens"], first["output_config"]), (32000, {"effort": "high"}))
                self.assertNotIn("display", first["thinking"])
                updated = cfg.add_model(launch.config, spec("added-live", output_limit_tokens=7777),
                                        wire_api="anthropic_messages")
                cfg.write_config(launch.config_path, updated)
                deadline, listed = time.monotonic() + 5, []
                while time.monotonic() < deadline and "added-live" not in listed:
                    status, models = self.api("GET", "/api/model")
                    listed = [item["id"] for item in models["data"]]
                    time.sleep(0.05)
                cfg.check_served_models(models["data"], updated)
            finally:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except OSError:
                    pass
                process.wait(10)
                log.close()
                stray = [item.pid for item in iso.processes_with_marker(marker).found]
                for pid in stray:           # only processes of this launch (marker match)
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except OSError:
                        pass
                self.assertEqual(stray, [])
                os.chmod(launch.dirs[cfg.CONFIG_DIR], 0o700)

# ==== tests/test_agent_opencode_effort_smoke.py ====

SMOKE_EXECUTABLE = os.environ.get("NI_RUA_OPENCODE_EXECUTABLE")
PROBLEM = (factory.opencode_runtime_problem(SMOKE_EXECUTABLE) if SMOKE_EXECUTABLE else
           "set NI_RUA_OPENCODE_EXECUTABLE to the pinned opencode 2.2.7 platform binary")
if PROBLEM is None and shutil.which("rg") is None:
    PROBLEM = "ripgrep (rg) is not installed"
IDENTITY = dict(run_id="effort", episode_id="ep", env_instance_id="env", env_generation=1)
KEY_ENV = "NI_RUA_SMOKE_OPENCODE_EFFORT_KEY"
OUTPUT_LIMIT = 8000
CONTEXT_LIMIT = 40000        # compaction above 40000 minus OpenCode's 15% buffer; the after-task step reports more
SIGNATURE = "fake-signature-effort"
# The last user text of OpenCode 2.2.7's compaction request, and a summary in its template.
COMPACTION_MARKS = ("You MUST summarize the conversation above into a structured summary",
                    "The previous response did not fill in the required summary template")
SUMMARY = ("## Objective\n- Effort smoke.\n\n## Requirements\n- (none)\n\n## Decisions\n- (none)\n\n## Work State\n"
           "### Completed\n- delegated\n\n### Active\n- (none)\n\n### Blocked\n- (none)\n\n## Next Move\n1. answer\n\n"
           "## Relevant Files\n(none)\n\n## Important Context\n- (none)\n")


class Router:
    """Classifies each request (``compaction``: OpenCode's summary instruction is the last user text - the 2.x summary
    request keeps the tools; ``child``: a session header other than the main session; else ``main``) and answers:
    main step 1 of a turn -> think + ``subagent`` call, step 2 (after the report) -> a native ``glob`` call with an
    overflowing usage, later steps -> the final text; child -> a short report; compaction -> a summary in OpenCode's
    template. ``reset()`` starts a turn."""

    def __init__(self, wire):
        self.wire, self.main_session, self.seen, self.steps = wire, None, [], 0

    def reset(self):
        self.seen.clear()
        self.steps = 0

    def __call__(self, request):
        body = request.json if isinstance(request.json, dict) else {}
        last = last_user_text(body, self.wire)
        kind = ("compaction" if any(mark in last for mark in COMPACTION_MARKS) else
                "child" if request.header("x-opencode-session-id") != self.main_session else "main")
        self.seen.append((kind, request))
        if kind == "compaction":
            return text_step(SUMMARY)
        if kind == "child":
            return text_step("child report")
        self.steps += 1
        if self.steps == 1:
            return Step(thinking="I should delegate.", thinking_signature=SIGNATURE,
                        tool_calls=[ToolCall("subagent", {"agent": "general", "description": "explore",
                                                          "prompt": "look around"})])
        if self.steps == 2:
            return Step(text="AFTER-TASK", usage=Usage(input_tokens=CONTEXT_LIMIT),
                        tool_calls=[ToolCall("glob", {"pattern": "*.nothing-here"})])
        return text_step("FINAL")


def last_user_text(body, wire):
    items = body.get("messages") if wire == "anthropic_messages" else body.get("input")
    for item in reversed(items or []):
        if item.get("role") != "user":
            continue
        content = item.get("content")
        if isinstance(content, str):
            return content
        return " ".join(block.get("text", "") for block in content or [] if isinstance(block, dict))
    return ""


def effort_field(body, wire):
    """The tier the request carries in the wire's field, plus the companion fields (None when absent)."""
    if wire == "anthropic_messages":
        return (body.get("output_config") or {}).get("effort"), body.get("thinking"), body.get("max_tokens")
    reasoning = body.get("reasoning") or {}
    return reasoning.get("effort"), (reasoning.get("summary"), body.get("include")), body.get("max_output_tokens")


@unittest.skipIf(PROBLEM is not None, f"OpenCode runtime unavailable: {PROBLEM}")
class OpenCodeEffortSmoke(unittest.TestCase):
    maxDiff = None

    def run_wire(self, wire, server_class, model):
        from ni_rua.cortex.agents.opencode.session import OpenCodeSession
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name).resolve() / "run"
        root.mkdir()
        router = Router(wire)
        server = server_class([], default=router).start()
        self.addCleanup(server.close)
        environment = mock.patch.dict(os.environ, {KEY_ENV: "host-secret-effort"})
        environment.start()
        self.addCleanup(environment.stop)
        store = Store(root, IDENTITY)
        self.addCleanup(store.close)
        config = Config.from_dict({
            "agent": {"backend": "opencode", "opencode": {
                "executable": SMOKE_EXECUTABLE, "context_limit_tokens": CONTEXT_LIMIT, "output_limit_tokens": OUTPUT_LIMIT,
                "upstream": {"wire_api": wire, "base_url": server.base_url, "api_key_env": KEY_ENV}}},
            "model": {"name": model, "effort": "low"},
            "features": {"self_briefing": False, "visual_review_model": None}})
        self.assertIsNone(factory.backend_requirements(config))
        session = OpenCodeSession(root, store, lambda params: {"success": True, "contentItems": []}, config=config)
        self.addCleanup(session.close)
        session.start([], "Effort smoke instructions.")
        router.main_session = session.thread_id
        self.assertEqual(session.effort_choices, EFFORT_CHOICES[wire])
        for index, tier in enumerate(EFFORT_CHOICES[wire]):
            with self.subTest(wire=wire, tier=tier):
                router.reset()
                session.set_effort(tier)
                session.run_turn(f"DELEGATE ({tier})")
                self.assertTrue(session.turn_done.wait(120), "turn did not end")
                self.assertEqual(session.state, "completed", session.turn_result)
                self.assertEqual(session.completed_final_answer(), "FINAL")
                kinds = [kind for kind, _ in router.seen]
                self.assertTrue({"main", "child", "compaction"} <= set(kinds), kinds)
                for kind, request in router.seen:
                    effort, companion, limit = effort_field(request.json, wire)
                    self.assertEqual(effort, tier, (kind, request.json))
                    if wire == "anthropic_messages":
                        self.assertEqual(companion, {"type": "adaptive"}, kind)
                    else:
                        self.assertEqual(companion, (None, ["reasoning.encrypted_content"]), kind)
                    if kind != "compaction":      # 2.2.7 sends its summary request with its own max_tokens (1024)
                        self.assertEqual(limit, OUTPUT_LIMIT, kind)
                if index == 0:
                    self.assert_reasoning_replayed(wire, [request for kind, request in router.seen if kind == "main"])
        native = session.provenance()["native_effort"]
        self.assertEqual((native["variant"], native["declared_variants"]),
                         (EFFORT_CHOICES[wire][-1], list(EFFORT_CHOICES[wire])))
        switches = [event["payload"] for event in store.events_by_type(["agent.model_changed"], limit=100)]
        previous, expected = "low", []                       # model.effort at start: the session's first variant
        for tier in EFFORT_CHOICES[wire]:
            if tier != previous:
                expected.append(tier)
            previous = tier
        self.assertEqual([item["variant"] for item in switches], expected)
        report = session.close()
        self.assertTrue(report["clean_shutdown"], report)

    def assert_reasoning_replayed(self, wire, main_requests):
        """The step that thought and called ``subagent`` is replayed with its reasoning in the next main request."""
        after_tool = next(request.json for request in main_requests
                          if '"tool_result"' in json.dumps(request.json) or '"function_call_output"' in
                          json.dumps(request.json))
        if wire == "anthropic_messages":
            assistant = [message for message in after_tool["messages"] if message.get("role") == "assistant"][-1]
            thinking = [block for block in assistant["content"] if block.get("type") == "thinking"]
            self.assertEqual([(block.get("thinking"), block.get("signature")) for block in thinking],
                             [("I should delegate.", SIGNATURE)])
        else:
            reasoning = [item for item in after_tool["input"] if item.get("type") == "reasoning"]
            self.assertEqual([item.get("encrypted_content") for item in reasoning], [SIGNATURE])

    def test_anthropic_messages_every_tier(self):
        self.run_wire("anthropic_messages", FakeAnthropicServer, "claude-effort-model")

    def test_openai_responses_every_tier(self):
        self.run_wire("openai_responses", FakeResponsesServer, "gpt-effort-model")

if __name__ == "__main__":
    unittest.main()
