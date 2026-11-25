"""Hosted mode and plugin mode coexist: an ``ni-rua-plugin`` install into the user's own host config homes does not
change a hosted run (docs/development/agent-plugin.md, "Two modes, not mutually exclusive").

Each test writes what ``agent-plugin/bin/cli.mjs`` writes into a fake user config home (never the real one), then
builds the hosted backend's launch and asserts it ignores every installed entry and leaves the user files untouched.
No agent starts and no model is contacted.
"""
import inspect
import json
import os
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

from ni_rua.core.config import ClaudeCodeConfig, Config
from ni_rua.cortex.agents import mcp_server, mcp_session
from ni_rua.cortex.agents.claude_code import options as claude_opts
from ni_rua.cortex.agents.opencode import config as opencode_cfg
from ni_rua.cortex.agents.codex.session import CodexSession
from ni_rua.core.storage.store import Store
from support.fake_claude_sdk import requires_sdk
from support.agent_fixtures import FAKE_CLI, SESSION, write_cli, fake_opencode_version_binary as fake_binary, spec

NI_RUA_TEST_TIER = "packaging"
PLUGIN_DIR = "/home/user/.local/share/ni-rua/agent-plugin/0.1.0"
PLUGIN_ENV = {"MAX_MCP_OUTPUT_TOKENS": "2000000", "MCP_TOOL_TIMEOUT": "3600000"}
CODEX_PLUGIN_TOML = (
    '[marketplaces.ni_rua]\nsource_type = "local"\nsource = "' + PLUGIN_DIR + '"\n\n'
    '[plugins."ni_rua@ni_rua"]\nenabled = true\n')


class CodexHostedIgnoresPlugin(unittest.TestCase):
    def test_private_codex_home_has_no_plugin_marketplace_or_skill(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        root = Path(directory).resolve()
        source = root / "codex-home-source"
        (source / "plugins" / "cache" / "ni_rua" / "ni_rua" / "0.1.0").mkdir(parents=True)
        (source / "plugins" / "cache" / "ni_rua" / "ni_rua" / "0.1.0" / "plugin.json").write_text("{}")
        (source / "skills" / "ni-rua-robot").mkdir(parents=True)
        (source / "skills" / "ni-rua-robot" / "SKILL.md").write_text("operator skill")
        (source / "config.toml").write_text(
            'model_provider = "fixture"\n\n[model_providers.fixture]\nname = "Fixture"\n'
            'base_url = "https://fixture.invalid/v1"\nwire_api = "responses"\nrequires_openai_auth = true\n\n'
            + CODEX_PLUGIN_TOML)
        (source / "auth.json").write_text(json.dumps({"OPENAI_API_KEY": "sk-fixture"}))
        before = (source / "config.toml").read_bytes()
        relay = mock.MagicMock(base_url="http://127.0.0.1:1/v1")
        relay.start.return_value = relay
        for target, value in (("ModelRequestProxy", dict(return_value=relay)),
                              ("snapshot_actor_source", dict(return_value=root / "actor-source"))):
            patcher = mock.patch("ni_rua.cortex.agents.codex.session." + target, **value)
            patcher.start()
            self.addCleanup(patcher.stop)
        # 屏蔽后台模型列表 SSL 拉取线程，避免多进程测试退出时触发 OpenSSL 析构 SIGSEGV
        cat_patcher = mock.patch.object(CodexSession, "_start_model_catalogue", return_value=None)
        cat_patcher.start()
        self.addCleanup(cat_patcher.stop)
        config = Config.from_dict({"model": {"name": "gpt-6.1-sol", "effort": "high", "codex_home": str(source)},
                                   "features": {"sandbox_enabled": False}})
        store = Store(root / "run", dict(run_id="run", episode_id="e", env_instance_id="i", env_generation=1))
        self.addCleanup(store.close)
        session = CodexSession(store.root, store, lambda _: {}, config=config)
        self.addCleanup(session.close)
        self.assertEqual(session._prepare(), "fixture")

        private = session.codex_home
        self.assertEqual(private, store.root / "private" / "codex-home")
        text = (private / "config.toml").read_text()
        parsed = tomllib.loads(text)
        self.assertIs(parsed["features"]["plugins"], False)
        for key in ("plugins", "marketplaces"):
            self.assertNotIn(key, parsed)
        self.assertNotIn("ni_rua@ni_rua", text)
        self.assertNotIn(PLUGIN_DIR, text)
        for name in ("plugins", "skills"):
            self.assertFalse((private / name).exists(), name)
        self.assertEqual(sorted(path.name for path in private.iterdir()),
                         sorted(["config.toml", "auth.json", "models.json"]))
        self.assertEqual(session.env["CODEX_HOME"], str(private))
        self.assertNotEqual(session.env["CODEX_HOME"], str(source))
        self.assertEqual((source / "config.toml").read_bytes(), before)
        self.assertTrue((source / "plugins" / "cache").is_dir() and (source / "skills").is_dir())


@requires_sdk
class ClaudeHostedIgnoresPlugin(unittest.TestCase):
    def test_launch_uses_private_config_and_no_setting_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            user = root / "user-claude"
            user.mkdir()
            settings = {"enabledPlugins": {"ni_rua@ni_rua": True},
                        "extraKnownMarketplaces": {"ni_rua": {"source": {"source": "directory", "path": PLUGIN_DIR}}},
                        "env": dict(PLUGIN_ENV), "permissions": {"allow": ["mcp__plugin_ni_rua_ni_rua__*"]}}
            (user / "settings.json").write_text(json.dumps(settings))
            before = (user / "settings.json").read_bytes()
            run = root / "run"
            run.mkdir()
            cli = write_cli(root, FAKE_CLI)
            runtime = claude_opts.resolve_runtime(cli, {"PATH": os.defpath})
            host = {"CLAUDE_CONFIG_DIR": str(user), "HOME": str(root / "user-home"), **PLUGIN_ENV}
            with mock.patch.dict(os.environ, host):
                launch = claude_opts.build_launch(
                    run, model="claude-test-1", effort="high", thread_id=SESSION, instructions="x\n",
                    relay_base_url="http://127.0.0.1:41000", mcp_url="http://127.0.0.1:41001/mcp",
                    mcp_headers={"Authorization": "Bearer tok"}, runtime=runtime)
            private = str(run / "private" / "claude-config")
            self.assertEqual(launch.env["CLAUDE_CONFIG_DIR"], private)
            self.assertEqual(launch.options.setting_sources, [])
            self.assertIs(launch.options.strict_mcp_config, True)
            self.assertEqual(launch.options.mcp_servers["ni_rua"]["url"], "http://127.0.0.1:41001/mcp")
            self.assertNotEqual(launch.env["HOME"], host["HOME"])
            self.assertTrue(launch.env["HOME"].startswith(str(run)))
            for value in launch.env.values():
                self.assertNotIn(str(user), value)
                self.assertNotIn(host["HOME"], value)
            # the installer's env values reach the launch only where NI-RUA itself sets them (its own table), never
            # from the user's settings: the user file is not read, so nothing named in it is forwarded
            self.assertEqual(launch.env["MCP_TOOL_TIMEOUT"], claude_opts._MCP_FLAGS["MCP_TOOL_TIMEOUT"])
            self.assertFalse(any("ni_rua@ni_rua" in value or "plugin_ni_rua" in value for value in launch.env.values()))
            self.assertFalse(getattr(launch.options, "plugins", None))
            self.assertEqual((user / "settings.json").read_bytes(), before)


class OpenCodeHostedIgnoresPlugin(unittest.TestCase):
    def test_launch_uses_private_xdg_and_config_without_plugins(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        root = Path(directory).resolve()
        user_xdg = root / "user-xdg"
        (user_xdg / "opencode").mkdir(parents=True)
        global_config = {"plugins": [PLUGIN_DIR], "tool_output": {"max_lines": 100000, "max_bytes": 16000000}}
        (user_xdg / "opencode" / "opencode.json").write_text(json.dumps(global_config))
        before = (user_xdg / "opencode" / "opencode.json").read_bytes()
        run = root / "run"
        run.mkdir()
        self.addCleanup(lambda: [os.chmod(p, 0o700) for p in run.rglob("*") if p.is_dir() and not p.is_symlink()])
        exe = fake_binary(root)
        launch = opencode_cfg.build_launch(
            run, executable=exe, models=[spec()], wire_api="anthropic_messages",
            relay_base_url="http://127.0.0.1:9999", instructions_text="INSTR", port=4567, password="pw",
            mcp_url="http://127.0.0.1:8888/mcp", mcp_headers={"Authorization": "Bearer t"}, ripgrep=exe,
            managed_config_paths=[],
            base_env={"XDG_CONFIG_HOME": str(user_xdg), "OPENCODE_CONFIG_DIR": str(user_xdg / "opencode")})
        private_config = run / opencode_cfg.CONFIG_DIR
        self.assertEqual(launch.env["OPENCODE_CONFIG_DIR"], str(private_config))
        self.assertEqual(launch.env["XDG_CONFIG_HOME"], str(private_config.parent))
        for name in ("XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME"):
            self.assertTrue(launch.env[name].startswith(str(run / "private")), name)
        for value in launch.env.values():
            self.assertNotIn(str(user_xdg), value)
        written = json.loads(launch.config_path.read_text())
        self.assertEqual(launch.config_path.parent, private_config)
        for document in (launch.config, written):
            self.assertNotIn("plugins", document)
            self.assertNotIn("plugin", document)
            self.assertNotIn(PLUGIN_DIR, json.dumps(document))
        self.assertEqual(launch.env["OPENCODE_DISABLE_PROJECT_CONFIG"], "1")
        self.assertEqual((user_xdg / "opencode" / "opencode.json").read_bytes(), before)


class HostedNeverBindsThePluginPort(unittest.TestCase):
    def test_mcp_server_default_port_is_ephemeral_and_only_external_passes_one(self):
        self.assertEqual(inspect.signature(mcp_server.NiRuaMcpServer.__init__).parameters["port"].default, 0)
        self.assertEqual(inspect.signature(mcp_session.McpToolBridge.__init__).parameters["server_options"].default,
                         None)
        session = mock.MagicMock()
        with mock.patch.object(mcp_session, "NiRuaMcpServer") as server:
            mcp_session.McpToolBridge(session, [])                    # what the hosted sessions do
            self.assertNotIn("port", server.call_args.kwargs)
        from ni_rua.cortex.agents.claude_code import session as claude_session
        from ni_rua.cortex.agents.opencode import session as opencode_session
        for module in (claude_session, opencode_session):
            source = inspect.getsource(module)
            self.assertNotIn("server_options", source)
            self.assertNotIn("mcp_port", source)
        from ni_rua.cortex.agents.external import session as external_session
        self.assertIn("port=self.mcp_port", inspect.getsource(external_session))

    def test_hosted_requirements_do_not_probe_the_plugin_port(self):
        from ni_rua.cortex.agents import factory
        from ni_rua.core.config import ConfigError
        for backend in ("codex", "claude_code", "opencode"):
            with self.subTest(backend), mock.patch.object(factory, "external_port_problem") as probe:
                try:
                    factory.backend_requirements(Config.from_dict(
                        {"agent": {"backend": backend}, "model": {"name": "m", "effort": "high"}}))
                except ConfigError:
                    pass                                              # a missing runtime is not the port
                probe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
