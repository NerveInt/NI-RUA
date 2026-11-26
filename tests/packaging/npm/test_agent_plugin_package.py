"""The npm agent plugin (agent-plugin/): single-source generated manifest, Codex tool map, and host entry parity."""

import importlib.util
import json
from pathlib import Path
import re
import tomllib
import unittest

from support.paths import REPO_ROOT

NI_RUA_TEST_TIER = "packaging"
REPOSITORY = REPO_ROOT
PLUGIN = REPOSITORY / "agent-plugin"
SCRIPT = REPOSITORY / "scripts" / "gen_agent_plugin_codex_mcp.py"


def load_generator():
    spec = importlib.util.spec_from_file_location("gen_agent_plugin_codex_mcp", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(relative):
    return json.loads((PLUGIN / relative).read_text())


class CodexToolMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generator = load_generator()
        cls.registry_tools = cls.generator.registry_actor_tools()
        cls.entry = read_json(".codex-mcp.json")["mcpServers"]["ni_rua"]

    def test_generated_files_match_generator_output_exactly(self):
        for rel, expected in self.generator.render_files().items():
            with self.subTest(file=rel):
                self.assertEqual((PLUGIN / rel).read_text(), expected)

    def test_map_covers_the_real_startup_catalogues(self):
        from support import embodiment_golden as golden

        for embodiment, table in (("x5", golden.X5_STARTUP), ("so101", golden.SO101_STARTUP)):
            scenario = next(iter(table))
            names = {tool["name"] for tool in golden.run_startup(embodiment, scenario)["tools"]}
            self.assertGreater(len(names), 10, embodiment)
            self.assertEqual(sorted(names - set(self.entry["tools"])), [], f"{embodiment}: tools without output_token_limit")

    def test_every_registry_actor_tool_has_an_output_limit(self):
        missing = sorted(self.registry_tools - set(self.entry["tools"]))
        self.assertEqual(missing, [], "tools without output_token_limit in agent-plugin/.codex-mcp.json")
        self.assertEqual(set(self.entry["tools"]), self.registry_tools)
        limits = {tool["output_token_limit"] for tool in self.entry["tools"].values()}
        self.assertEqual(limits, {self.generator.OUTPUT_TOKEN_LIMIT})
        self.assertGreaterEqual(self.generator.OUTPUT_TOKEN_LIMIT * 4, 3 * self.generator.LARGEST_KNOWN_RESULT_CHARS)

    def test_bracket_tools_come_from_the_registry(self):
        self.assertLessEqual({"run_begin", "run_finish"}, self.registry_tools)
        self.assertLessEqual({"run_begin", "run_finish"}, set(self.entry["tools"]))

    def test_generator_imports_the_checkout_it_lives_in(self):
        import ni_rua

        self.assertEqual(Path(ni_rua.__file__).resolve().parents[1], REPOSITORY)


class HostEntryAgreementTests(unittest.TestCase):
    def test_single_source_manifest_drives_entries_and_plugin_runtime(self):
        from ni_rua.cortex.agents import plugin_contract
        from ni_rua.cortex.agents.claude_code.options import _MCP_FLAGS
        from ni_rua.cortex.agents.opencode.config import MEDIA_IMAGE, TOOL_OUTPUT

        manifest = read_json("manifest.generated.json")
        self.assertEqual(manifest["schema"], plugin_contract.SCHEMA)
        self.assertEqual(manifest["default_port"], plugin_contract.default_port())
        self.assertEqual(manifest["web_port"], plugin_contract.default_web_port())
        self.assertEqual(manifest["mcp_url"], f"http://127.0.0.1:{manifest['default_port']}/mcp")
        self.assertEqual([item["name"] for item in manifest["tool_definitions"]], manifest["tool_names"])
        for item in manifest["tool_definitions"]:
            self.assertTrue(item["description"])
            self.assertIsInstance(item["inputSchema"], dict)
            self.assertEqual(item["inputSchema"].get("type"), "object")
            for forbidden in ("anyOf", "oneOf", "allOf", "if", "then", "else"):
                self.assertNotIn(forbidden, item["inputSchema"], item["name"])
        expected_env = {k: v for k, v in _MCP_FLAGS.items() if k != "CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT"}
        self.assertEqual(manifest["claude_env"], expected_env)
        self.assertEqual(manifest["opencode_tool_output"], TOOL_OUTPUT)
        self.assertEqual(manifest["opencode_media_image"], MEDIA_IMAGE)
        self.assertEqual(manifest["verified_hosts"], plugin_contract.verified_hosts())
        self.assertEqual(manifest["codex_direct_catalog"], plugin_contract.CODEX_DIRECT_CATALOG_FILE)
        self.assertEqual(manifest["codex_code_mode_only_models"], plugin_contract.codex_code_mode_only_models())
        direct_catalog = read_json(plugin_contract.CODEX_DIRECT_CATALOG_FILE)
        self.assertTrue(direct_catalog["models"])
        self.assertTrue(all(model.get("tool_mode") == "direct" for model in direct_catalog["models"]))
        self.assertTrue(all(model.get("supports_search_tool") is False for model in direct_catalog["models"]))
        hooks = read_json("hooks/hooks.json")
        self.assertEqual(hooks["hooks"]["SessionStart"][0]["matcher"], "compact")
        self.assertTrue((PLUGIN / "hooks" / "on-compact.mjs").is_file())

        claude = read_json(".mcp.json")["mcpServers"]
        codex = read_json(".codex-mcp.json")["mcpServers"]
        self.assertEqual(list(claude), [manifest["server_name"]])
        self.assertEqual(list(codex), [manifest["server_name"]])
        self.assertEqual(
            claude["ni_rua"],
            {"command": "node", "args": ["${CLAUDE_PLUGIN_ROOT}/bin/cli.mjs", "mcp"], "alwaysLoad": True},
        )
        self.assertEqual(codex["ni_rua"]["command"], "ni-rua-plugin")
        self.assertEqual(codex["ni_rua"]["args"], ["mcp"])
        self.assertNotIn("url", codex["ni_rua"])
        for key, value in manifest["codex_timeouts"].items():
            self.assertEqual(codex["ni_rua"][key], value, key)
        self.assertEqual(codex["ni_rua"]["default_tools_approval_mode"], "approve")
        self.assertEqual(manifest["opencode_timeout_ms"]["execution"] // 1000, codex["ni_rua"]["tool_timeout_sec"])
        self.assertEqual(manifest["opencode_timeout_ms"]["startup"] // 1000, codex["ni_rua"]["startup_timeout_sec"])

        cli_text = (PLUGIN / "bin" / "cli.mjs").read_text()
        opencode_text = (PLUGIN / "opencode" / "plugin.js").read_text()
        self.assertIn("manifest.generated.json", cli_text)
        self.assertIn("manifest.generated.json", opencode_text)
        self.assertNotIn("DEFAULT_PORT = 8790", cli_text)
        self.assertNotIn("MAX_MCP_OUTPUT_TOKENS:", cli_text)
        self.assertNotIn("MCP_URL =", opencode_text)

    def test_versions_and_names_agree(self):
        package = read_json("package.json")
        manifest = read_json("manifest.generated.json")
        pyproject = tomllib.loads((REPOSITORY / "pyproject.toml").read_text())
        self.assertEqual(package["name"], "@nerveint/ni-rua-plugin")
        self.assertNotIn("private", package)
        self.assertEqual(package.get("publishConfig"), {"access": "public"})
        self.assertEqual(package["version"], pyproject["project"]["version"])
        self.assertEqual(manifest["version"], package["version"])
        self.assertEqual(read_json(".claude-plugin/plugin.json")["version"], package["version"])
        self.assertEqual(read_json(".codex-plugin/plugin.json")["version"], package["version"])
        for plugin_manifest in (".claude-plugin/plugin.json", ".codex-plugin/plugin.json"):
            self.assertEqual(read_json(plugin_manifest)["name"], "ni_rua")
        marketplace = read_json(".claude-plugin/marketplace.json")
        self.assertEqual(marketplace["name"], "ni_rua")
        self.assertEqual([item["name"] for item in marketplace["plugins"]], ["ni_rua"])
        self.assertEqual(read_json(".codex-plugin/plugin.json")["mcpServers"], "./.codex-mcp.json")
        self.assertEqual((PLUGIN / "LICENSE").read_text(), (REPOSITORY / "LICENSE").read_text())

    def test_package_files_exist_and_exclude_tests(self):
        package = read_json("package.json")
        self.assertIn("manifest.generated.json", package["files"])
        for entry in package["files"]:
            self.assertTrue((PLUGIN / entry).exists(), entry)
        self.assertFalse([entry for entry in package["files"] if entry.startswith("test")])
        self.assertEqual(package["bin"], {"ni-rua-plugin": "bin/cli.mjs"})
        self.assertEqual(package["exports"], {".": "./opencode/plugin.js"})
        self.assertNotIn("dependencies", package)

    def test_skills_carry_the_frontmatter_each_host_requires(self):
        for name in ("ni-rua-setup", "ni-rua-robot", "ni-rua-experiments"):
            text = (PLUGIN / "skills" / name / "SKILL.md").read_text()
            match = re.match(r"---\nname: (.+)\ndescription: (.+)\n---\n", text)
            self.assertIsNotNone(match, name)
            self.assertEqual(match.group(1), name)
            self.assertGreater(len(match.group(2)), 40)

    def test_setup_skill_covers_onboarding_calibration_and_safety_guardrails(self):
        text = (PLUGIN / "skills" / "ni-rua-setup" / "SKILL.md").read_text()
        for needle in (
            "ni-rua-plugin doctor",
            "ni-rua-so101 init-config",
            "ni-rua-so101 cameras",
            "ni-rua-so101 charuco-board",
            "ni-rua-so101 commission",
            "/calibration",
            "/console",
            "--dry-run",
            "12V",
            "Missing motor IDs",
            "dialout",
            "gateway.log",
            "ni-rua-robot",
        ):
            self.assertIn(needle, text)

    def test_actor_skill_states_the_required_rules_and_no_robot_numbers(self):
        text = (PLUGIN / "skills" / "ni-rua-robot" / "SKILL.md").read_text()
        for needle in ("run_begin", "run_finish", "does NOT stop the arm", "estop", "compaction", "Online search"):
            self.assertIn(needle, text)
        body = text.split("---\n", 2)[2]
        self.assertIsNone(re.search(r"\d+(\.\d+)?\s*(mm|cm|m|rad|deg|s)\b", body), "no robot numbers in the actor skill")

    def test_operator_skill_points_at_the_checkout_rules_and_real_flags(self):
        text = (PLUGIN / "skills" / "ni-rua-experiments" / "SKILL.md").read_text()
        agents_text = (REPOSITORY / "AGENTS.md").read_text()
        section = "实验评测、60 并发队列与防选集偏差"
        self.assertIn(f"## {section}", agents_text)
        for needle in (
            "AGENTS.md",
            section,
            "docs/development/experiment-library.md",
            "--no-wake-enabled",
            "--wake-thread",
            "--wake-codex",
            "--wake-enabled",
            "ni-rua-report",
            "ni-rua-audit",
            "2.1.321",
            "2.2.7",
        ):
            self.assertIn(needle, text)
        queue = (REPOSITORY / "ni_rua" / "apps" / "evaluation" / "queue.py").read_text()
        for flag in ("--wake-thread", "--wake-codex", "--wake-enabled", "--wait-timeout-s", "--output"):
            self.assertIn(f'"{flag}"', queue, flag)


if __name__ == "__main__":
    unittest.main()
