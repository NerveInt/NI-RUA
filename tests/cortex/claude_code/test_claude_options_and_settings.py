"""Merged cortex tests:
- tests/test_agent_claude_options.py: Claude Code launch builder: options, clean environment, ``env -i`` launcher, pins, provenance (design rev 3 s.8).
- tests/test_claude_settings_env.py: agent.claude_code.upstream.source = "claude_settings": the settings file's non-credential ``env`` entries reach the
- tests/test_claude_settings_upstream.py: agent.claude_code.upstream.source = "claude_settings": endpoint and key come from a Claude Code settings file in
"""

NI_RUA_TEST_TIER = "component"

import json
import os
import stat
import subprocess
import tempfile
import asyncio
import unittest
from pathlib import Path
from unittest import mock
from ni_rua.core.config import ClaudeCodeConfig
from ni_rua.cortex.agents.claude_code import options as opts
from ni_rua.cortex.agents.factory import NATIVE_EFFORT_CHOICES
from ni_rua.cortex.agents.isolation import VersionMismatch
from support.fake_claude_sdk import requires_sdk
from support.fake_model_servers import FakeAnthropicServer, text_step, tool_step
from ni_rua.core.config import Config, ConfigError
from ni_rua.cortex.agents import factory
from ni_rua.cortex.agents.claude_code.options import read_claude_settings_upstream

# ==== tests/test_agent_claude_options.py ====

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


@requires_sdk
class LaunchCheck(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()
        self.cli = write_cli(self.root)
        self.runtime = opts.resolve_runtime(self.cli, {"PATH": os.defpath})

    def build(self, **overrides):
        arguments = dict(model="claude-test-1", effort="high", thread_id=SESSION, instructions="Be careful.\n",
                         relay_base_url="http://127.0.0.1:41000", mcp_url="http://127.0.0.1:41001/mcp",
                         mcp_headers={"Authorization": "Bearer tok"}, runtime=self.runtime)
        arguments.update(overrides)
        return opts.build_launch(self.run_dir, **arguments)

    # ----- options -----
    def test_options_follow_the_isolation_table(self):
        launch = self.build()
        o = launch.options
        # default-allow policy: no tool allowlist of any kind, the CLI's own default set minus the denylist
        self.assertIsNone(o.tools)
        self.assertEqual(o.allowed_tools, [])
        self.assertEqual(o.disallowed_tools, ["WebSearch", "WebFetch", "AskUserQuestion", "EnterPlanMode",
                                              "ExitPlanMode"])
        self.assertEqual((o.permission_mode, o.setting_sources, o.strict_mcp_config),
                         ("bypassPermissions", [], True))
        self.assertEqual(o.system_prompt, {"type": "preset", "preset": "claude_code"})
        self.assertEqual(o.mcp_servers, {"ni_rua": {"type": "http", "url": "http://127.0.0.1:41001/mcp",
                                                    "headers": {"Authorization": "Bearer tok"}}})
        self.assertEqual((o.model, o.effort, o.fallback_model), ("claude-test-1", "high", None))
        self.assertEqual(json.loads(o.settings), {"includeCoAuthoredBy": False})
        self.assertEqual(o.max_buffer_size, 32 * 1024 * 1024)
        self.assertEqual(Path(o.cwd), launch.actor_dir)
        self.assertEqual(o.cli_path, os.fspath(launch.launcher))
        self.assertIsNone(o.can_use_tool)
        self.assertEqual(o.extra_args, {"append-system-prompt-file": os.fspath(launch.instructions_file),
                                        "replay-user-messages": None})   # slash commands are NOT disabled
        self.assertEqual(o.env, {})              # the environment lives in the launcher, never in options.env

    def test_session_id_fresh_and_resume_are_exclusive(self):
        fresh = self.build().options
        self.assertEqual((fresh.session_id, fresh.resume), (SESSION, None))
        resumed = self.build(resume=True)
        self.assertEqual((resumed.options.session_id, resumed.options.resume, resumed.thread_id, resumed.resumed),
                         (None, SESSION, SESSION, True))
        with self.assertRaisesRegex(ValueError, "claude_code_session_id_must_be_uuid"):
            self.build(thread_id="not-a-uuid")

    def test_subagents_are_on_by_default_and_denied_under_both_names_when_off(self):
        self.assertTrue(ClaudeCodeConfig().subagents)
        on = self.build()
        self.assertNotIn("Agent", on.options.disallowed_tools)
        self.assertNotIn("Task", on.options.disallowed_tools)
        off = self.build(config=ClaudeCodeConfig(subagents=False))
        # ``Agent`` in model requests, ``Task`` in the init message: either removes both, both are listed
        self.assertEqual(off.options.disallowed_tools[-2:], ["Agent", "Task"])
        self.assertIsNone(off.options.tools)
        self.assertEqual(off.options.allowed_tools, [])

    def test_disallowed_tools_carry_reasons(self):
        on = self.build()
        self.assertEqual(on.disallowed_tools, tuple(
            dict(name=name, reason=reason) for name, reason in (
                ("WebSearch", "online_access"), ("WebFetch", "online_access"), ("AskUserQuestion", "waits_for_human"),
                ("EnterPlanMode", "waits_for_human"), ("ExitPlanMode", "waits_for_human"))))
        off = self.build(config=ClaudeCodeConfig(subagents=False))
        self.assertEqual([item for item in off.disallowed_tools if item["reason"] == "subagents_disabled"],
                         [dict(name="Agent", reason="subagents_disabled"),
                          dict(name="Task", reason="subagents_disabled")])
        # nothing that is scheduling, background, skills, slash commands or worktrees is removed
        names = {item["name"] for item in off.disallowed_tools}
        for kept in ("CronCreate", "CronDelete", "CronList", "ScheduleWakeup", "Monitor", "Skill", "SlashCommand",
                     "TaskStop", "Bash", "EnterWorktree"):
            self.assertNotIn(kept, names)

    def test_hooks_and_stderr_pass_through(self):
        sink = []
        hooks = {"PreCompact": []}
        launch = self.build(hooks=hooks, stderr=sink.append)
        self.assertIs(launch.options.hooks, hooks)
        launch.options.stderr("line")
        self.assertEqual(sink, ["line"])

    # ----- environment -----
    def test_environment_is_explicit_and_complete(self):
        with mock.patch.dict(os.environ, {"AWS_SECRET_ACCESS_KEY": "leak", "ANTHROPIC_API_KEY": "real-key",
                                          "HTTPS_PROXY": "http://x", "MY_VAR": "v"}):
            launch = self.build()
        env = launch.env
        self.assertNotIn("leak", json.dumps(env))
        self.assertNotIn("real-key", json.dumps(env))
        self.assertEqual(set(env) & {"MY_VAR", "HTTPS_PROXY", "AWS_SECRET_ACCESS_KEY"}, set())
        self.assertEqual((env["HOME"], env["TMPDIR"], env["LANG"], env["PATH"]),
                         (os.fspath(self.run_dir / "actor"), os.fspath(self.run_dir / "private" / "tmp"), "C.UTF-8",
                          os.defpath))
        self.assertEqual(env["CLAUDE_CONFIG_DIR"], os.fspath(self.run_dir / "private" / "claude-config"))
        expected = {
            "ANTHROPIC_BASE_URL": "http://127.0.0.1:41000", "ANTHROPIC_API_KEY": "ni-rua-local-relay",
            "CLAUDE_CODE_GZIP_REQUEST_BODIES": "0", "CLAUDE_CODE_MAX_RETRIES": "0",
            "CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK": "1", "CLAUDE_STREAM_IDLE_TIMEOUT_MS": "1800000",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "DISABLE_TELEMETRY": "1", "DISABLE_ERROR_REPORTING": "1",
            "DISABLE_AUTOUPDATER": "1", "CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL": "1",
            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0", "CLAUDE_CODE_DISABLE_WEB_FETCH": "1", "ENABLE_TOOL_SEARCH": "false",
            "MAX_MCP_OUTPUT_TOKENS": "2000000", "MCP_TOOL_TIMEOUT": "3600000", "CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT": "0",
            "CLAUDE_CODE_MCP_AUTO_BACKGROUND_MS": "0", "CLAUDE_CODE_MAX_MCP_DESCRIPTION_LENGTH": "1000000",
            "CLAUDE_CODE_EMIT_SESSION_STATE_EVENTS": "1", "CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS": "1",
        }
        for key, value in expected.items():
            self.assertEqual(env.get(key), value, key)
        for key in ("ANTHROPIC_SMALL_FAST_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL",
                    "ANTHROPIC_DEFAULT_OPUS_MODEL", "CLAUDE_CODE_SUBAGENT_MODEL"):
            self.assertEqual(env[key], "claude-test-1", key)
        # native defaults stay: memory, CLAUDE.md discovery, background tasks, cron; no sandbox flag as non-root
        for key in ("CLAUDE_CODE_DISABLE_CLAUDE_MDS", "CLAUDE_CODE_DISABLE_AUTO_MEMORY",
                    "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS", "CLAUDE_CODE_DISABLE_CRON", "CLAUDE_AUTO_BACKGROUND_TASKS",
                    "IS_SANDBOX", "CLAUDE_CODE_BUBBLEWRAP"):
            self.assertNotIn(key, env, key)
        # nothing of the CLI writes below the shared host /tmp
        self.assertEqual(env["CLAUDE_CODE_TMPDIR"], os.fspath(self.run_dir / "private" / "tmp"))
        self.assertEqual(env["XDG_RUNTIME_DIR"], os.fspath(self.run_dir / "private" / "run"))
        self.assertNotIn("CLAUDE_CODE_AUTO_COMPACT_WINDOW", env)
        # the three SDK-injected variables are NOT in env (the launcher re-exports them)
        for key in opts.REEXPORT_VARS:
            self.assertNotIn(key, env)
        self.assertGreater(opts.STREAM_IDLE_TIMEOUT_MS, 0)

    def test_mcp_description_cap_is_lifted_and_recorded(self):
        launch = self.build()
        self.assertEqual(launch.env["CLAUDE_CODE_MAX_MCP_DESCRIPTION_LENGTH"], "1000000")
        self.assertEqual(launch.native_settings["env"]["CLAUDE_CODE_MAX_MCP_DESCRIPTION_LENGTH"], "1000000")

    def test_host_git_context_is_switched_off_recorded_and_owned(self):
        # Information boundary: without it the CLI puts the gitStatus block of any git repository above the actor
        # directory (branch, git user, untracked host paths, commit subjects) into the first user message
        # (reproduced in test_agent_prompt_parity.ClaudeCodeHostRepositoryBoundary).
        launch = self.build()
        self.assertEqual(launch.env["CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS"], "1")
        self.assertEqual(launch.native_settings["env"]["CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS"], "1")
        # the environment variable is the switch (it wins over a settings value); the settings stay as they were
        self.assertEqual(json.loads(launch.options.settings), {"includeCoAuthoredBy": False})
        self.assertIn("CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS", opts.NI_RUA_OWNED_ENV)
        with self.assertRaisesRegex(ValueError, "^claude_settings_env_reserved:CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS$"):
            self.build(settings_env={"CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS": "0"})
        settings = self.root / "provider-settings.json"
        settings.write_text(json.dumps({"env": {"ANTHROPIC_API_KEY": "k", "CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS": "0"}}))
        with self.assertRaisesRegex(ValueError, "claude_settings_env_reserved:.*CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS"):
            opts.read_claude_settings_env(settings)

    def test_native_memories_false_sets_only_the_auto_memory_switch(self):
        env = self.build(native_memories=False).env
        self.assertEqual(env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"], "1")
        self.assertNotIn("CLAUDE_CODE_DISABLE_CLAUDE_MDS", env)
        self.assertNotIn("CLAUDE_CODE_DISABLE_AUTO_MEMORY", self.build(native_memories=True).env)
        from ni_rua.cortex.agents.claude_code.session import ClaudeCodeSession
        self.assertTrue(ClaudeCodeSession.capabilities.native_memories)
        self.assertTrue(ClaudeCodeSession.capabilities.turn_images)
        self.assertEqual(
            ClaudeCodeSession.__new__(ClaudeCodeSession).host_tool_output_truncation_limit(),
            (opts.TOOL_RESULT_TEXT_LIMIT_CHARS, None, True),
        )

    def test_is_sandbox_only_for_uid_zero_and_recorded(self):
        with mock.patch.object(opts, "_running_as_root", return_value=True):
            root = self.build()
        self.assertEqual(root.env["IS_SANDBOX"], "1")
        self.assertTrue(root.native_settings["is_sandbox_env"])
        self.assertEqual(root.native_settings["env"]["IS_SANDBOX"], "1")
        with mock.patch.object(opts, "_running_as_root", return_value=False):
            other = self.build()
        self.assertNotIn("IS_SANDBOX", other.env)
        self.assertFalse(other.native_settings["is_sandbox_env"])

    def test_stream_idle_timeout_covers_the_relay_timeout(self):
        self.assertEqual(self.build(relay_timeout_s=1800).env["CLAUDE_STREAM_IDLE_TIMEOUT_MS"], "1800000")
        with self.assertRaisesRegex(ValueError, "relay_timeout_exceeds_claude_stream_idle_timeout"):
            self.build(relay_timeout_s=1801)

    def test_auto_compact_window_is_passed_when_configured(self):
        launch = self.build(config=ClaudeCodeConfig(auto_compact_window_tokens=150000))
        self.assertEqual(launch.env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"], "150000")
        self.assertEqual(launch.native_settings["auto_compact_window_tokens"], 150000)

    def test_private_dirs_and_files(self):
        launch = self.build()
        for path in (self.run_dir / "private", launch.config_dir, launch.tmp_dir, self.run_dir / "private" / "run"):
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o700, path)
        self.assertTrue((self.run_dir / "actor").is_dir())
        self.assertEqual(stat.S_IMODE(launch.launcher.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(launch.instructions_file.stat().st_mode), 0o600)
        self.assertEqual(launch.instructions_file.read_text(), launch.instructions_text)

    def test_instructions_file_is_presented_text_plus_naming_note(self):
        launch = self.build(instructions="Alpha line.\nBeta line.\n\n")
        note = ("Tool names in these instructions are written without the prefix 'mcp__ni_rua__' that your tool "
                "list shows; they refer to the same tools.")
        self.assertEqual(launch.instructions_text, "Alpha line.\nBeta line.\n\n" + note + "\n")
        self.assertEqual(opts.compose_instructions(""), note + "\n")
        import hashlib
        self.assertEqual(launch.instructions_sha256, hashlib.sha256(launch.instructions_text.encode()).hexdigest())

    def test_symlinked_private_dir_is_refused(self):
        (self.run_dir / "elsewhere").mkdir()
        os.symlink(self.run_dir / "elsewhere", self.run_dir / "private")
        with self.assertRaisesRegex(ValueError, "private_dir_symlink"):
            self.build()

    # ----- launcher -----
    def test_launcher_exposes_only_the_given_env_plus_three_sdk_variables(self):
        launch = self.build()
        parent = {"PATH": "/usr/bin:/bin", "HOME": "/nonexistent", "SECRET_THING": "boom",
                  "ANTHROPIC_API_KEY": "real", "CLAUDE_CODE_ENTRYPOINT": "sdk-py",
                  "CLAUDE_AGENT_SDK_VERSION": "0.2.180", "CLAUDE_CODE_SDK_READS_SESSION_STATE": "1",
                  "CLAUDECODE": "1"}
        output = subprocess.run([os.fspath(launch.launcher), "--print"], env=parent, capture_output=True, text=True,
                                check=True).stdout
        seen = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
        seen.pop("_", None)
        seen.pop("PWD", None)
        seen.pop("SHLVL", None)
        expected = dict(launch.env, CLAUDE_CODE_ENTRYPOINT="sdk-py", CLAUDE_AGENT_SDK_VERSION="0.2.180",
                        CLAUDE_CODE_SDK_READS_SESSION_STATE="1")
        self.assertEqual(seen, expected)
        # a re-exported variable the SDK did not set is not invented
        parent.pop("CLAUDE_CODE_SDK_READS_SESSION_STATE")
        output = subprocess.run([os.fspath(launch.launcher)], env=parent, capture_output=True, text=True,
                                check=True).stdout
        self.assertNotIn("CLAUDE_CODE_SDK_READS_SESSION_STATE", output)

    def test_launcher_accepts_the_sdk_version_probe(self):
        launch = self.build()
        completed = subprocess.run([os.fspath(launch.launcher), "--version"], capture_output=True, text=True)
        self.assertEqual(completed.stdout.strip(), "2.1.321 (Claude Code)")

    # ----- effort -----
    def test_effort_mapping(self):
        self.assertEqual(opts.effort_choices(), NATIVE_EFFORT_CHOICES["claude_code"])
        self.assertEqual(opts.effort_choices(), ("low", "medium", "high", "xhigh", "max"))
        for effort in opts.effort_choices():
            self.assertEqual(opts.native_effort(effort), effort)
            self.assertEqual(self.build(effort=effort).options.effort, effort)
        for effort in ("none", "minimal", "ultra", None):
            with self.assertRaisesRegex(ValueError, "agent_effort_unsupported:claude_code"):
                opts.native_effort(effort)

    def test_invalid_model(self):
        for model in ("", None):
            with self.assertRaisesRegex(ValueError, "invalid_model"):
                self.build(model=model)

    # ----- provenance -----
    def test_provenance_block_contents(self):
        launch = self.build()
        block = opts.provenance_block(
            launch, model="claude-test-1", counting="delivered_usable_generation_completions_v1",
            upstream_provider="anthropic", init_tools=["Bash", "Read", "TaskStop", "mcp__ni_rua__robot_observe"],
            init_model="claude-test-1", capabilities={"steer": True}, decision_reminder_channel="post_compaction")
        json.dumps(block)
        self.assertEqual(block["kind"], "claude_code")
        self.assertEqual(block["sdk"], {"name": "claude-agent-sdk", "version": "0.2.180"})
        self.assertEqual(block["cli"], {"version": "2.1.321", "executable": os.fspath(self.cli.resolve()),
                                        "sha256": self.runtime.sha256})
        self.assertEqual((block["wire_api"], block["upstream_provider"], block["model_configured"], block["effort"],
                          block["native_effort"], block["isolation"], block["context_delivery"],
                          block["evidence_client"], block["session_resumed"]),
                         ("anthropic_messages", "anthropic", "claude-test-1", "high",
                          {"effort": "high", "live": False, "applies": "session_start"}, "disabled",
                          "delta", "relay", False))
        full_block = opts.provenance_block(launch, model="claude-test-1",
                                           counting="delivered_usable_generation_completions_v1",
                                           context_delivery="full")
        self.assertEqual(full_block["context_delivery"], "full")
        self.assertEqual(block["tool_transport"]["tool_name_prefix"], "mcp__ni_rua__")
        self.assertEqual(block["tool_transport"]["kind"], "mcp_streamable_http")
        channel = block["prompt_channel"]
        self.assertEqual((channel["instructions_channel"], channel["native_base_prompt"], channel["tool_name_prefix"],
                          channel["instructions_sha256"], channel["naming_note"]),
                         ("append_system_prompt_file", "preserved", "mcp__ni_rua__", launch.instructions_sha256,
                          opts.naming_note()))
        self.assertEqual(block["native_tools"]["enabled"], ["Bash", "Read", "TaskStop"])   # read back from init
        self.assertEqual(block["native_tools"]["disabled"][:2], [{"name": "WebSearch", "reason": "online_access"},
                                                                 {"name": "WebFetch", "reason": "online_access"}])
        self.assertEqual(opts.native_tools_block(launch, ["mcp__ni_rua__x", "Task", "Monitor"])["enabled"],
                         ["Task", "Monitor"])
        self.assertEqual((block["native_model_id"], block["capabilities"], block["decision_reminder_channel"]),
                         ("claude-test-1", {"steer": True}, "post_compaction"))

    def test_provenance_without_init_leaves_enabled_unknown_and_records_effective_settings(self):
        launch = self.build(config=ClaudeCodeConfig(subagents=True))
        block = opts.provenance_block(launch, model="m", counting="c")
        self.assertIsNone(block["native_tools"]["enabled"])   # never a hand-written list
        self.assertEqual(len(block["native_tools"]["disabled"]), 5)
        self.assertNotIn("native_model_id", block)
        settings = block["native_settings"]
        self.assertEqual(settings["env"]["CLAUDE_CODE_MAX_RETRIES"], "0")
        self.assertEqual(settings["env"]["CLAUDE_CODE_EMIT_SESSION_STATE_EVENTS"], "1")
        self.assertNotIn("CLAUDE_CODE_DISABLE_BACKGROUND_TASKS", settings["env"])
        self.assertNotIn("CLAUDE_CODE_TMPDIR", settings["env"])   # run layout is not recorded
        self.assertEqual((settings["options"]["tools"], settings["is_sandbox_env"], settings["native_memory"]),
                         ("cli_default", False, True))
        self.assertEqual(settings["reexported_env"], list(opts.REEXPORT_VARS))
        self.assertEqual((settings["options"]["permission_mode"], settings["options"]["setting_sources"],
                          settings["options"]["strict_mcp_config"], settings["options"]["can_use_tool"]),
                         ("bypassPermissions", [], True, False))
        self.assertEqual(settings["stream_idle_timeout_ms"], 1800000)

    def test_provenance_carries_no_run_paths_ports_or_secrets(self):
        launch = self.build()
        text = json.dumps(opts.provenance_block(launch, model="m", counting="c"))
        for forbidden in (os.fspath(self.run_dir), "41000", "41001", "Bearer", "tok\""):
            self.assertNotIn(forbidden, text)


@requires_sdk
class RuntimeCheck(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()

    def test_explicit_cli_must_report_the_pinned_version(self):
        cli = write_cli(self.root)
        runtime = opts.resolve_runtime(cli, {"PATH": os.defpath})
        self.assertEqual((runtime.sdk_version, runtime.cli_version, runtime.executable),
                         ("0.2.180", "2.1.321", os.fspath(cli.resolve())))
        self.assertEqual(len(runtime.sha256), 64)
        other = write_cli(self.root, FAKE_CLI.replace("2.1.321", "2.1.322"), name="other")
        with self.assertRaisesRegex(VersionMismatch, "claude_code_version_mismatch") as raised:
            opts.resolve_runtime(other, {"PATH": os.defpath})
        self.assertEqual((raised.exception.actual, raised.exception.pinned), ("2.1.322", "2.1.321"))

    def test_probe_output_forms(self):
        cli = write_cli(self.root)
        self.assertEqual(opts.resolve_runtime(cli, probe=lambda exe, env: "2.1.321\n").cli_version, "2.1.321")
        for bad in ("", "  \n", "claude 2.1.321", "v2.1.321 (Claude Code)"):
            with self.assertRaises(VersionMismatch, msg=repr(bad)):
                opts.resolve_runtime(cli, probe=lambda exe, env, bad=bad: bad)

    def test_probe_receives_the_clean_environment_and_real_path(self):
        cli = write_cli(self.root)
        seen = {}
        opts.resolve_runtime(cli, {"PATH": "/x", "ONLY": "1"},
                             probe=lambda exe, env: seen.update(exe=exe, env=env) or "2.1.321 (Claude Code)")
        self.assertEqual((seen["exe"], seen["env"]), (cli.resolve(), {"PATH": "/x", "ONLY": "1"}))

    def test_failed_probe_and_bad_paths(self):
        broken = write_cli(self.root, "#!/bin/sh\nexit 3\n", name="broken")
        with self.assertRaisesRegex(RuntimeError, "claude_code_cli_version_probe_failed:exit_3"):
            opts.resolve_runtime(broken, {"PATH": os.defpath})
        with self.assertRaisesRegex(ValueError, "claude_code_cli_not_found"):
            opts.resolve_runtime(self.root / "missing", {"PATH": os.defpath})
        with self.assertRaisesRegex(ValueError, "claude_code_cli_path_must_be_absolute"):
            opts.resolve_runtime("relative/claude")
        notexec = self.root / "plain"
        notexec.write_text("x")
        with self.assertRaisesRegex(ValueError, "claude_code_cli_not_found"):
            opts.resolve_runtime(notexec)

    def test_sdk_pin_mismatch_is_a_hard_error(self):
        cli = write_cli(self.root)
        with mock.patch.object(opts, "_installed_sdk_version", return_value="0.2.181"):
            with self.assertRaisesRegex(VersionMismatch, "claude_code_version_mismatch"):
                opts.resolve_runtime(cli, {"PATH": os.defpath})

    def test_missing_sdk_distribution_is_reported(self):
        import importlib.metadata
        with mock.patch.object(opts.importlib.metadata, "version", side_effect=importlib.metadata.PackageNotFoundError):
            with self.assertRaisesRegex(RuntimeError, "claude_code_sdk_missing"):
                opts._installed_sdk_version()

    def test_bundled_cli_cli_version_constant_matches_the_sdk(self):
        bundled, version = opts._bundled_cli()
        self.assertEqual(version, opts.CLI_VERSION)
        self.assertEqual(opts.SDK_VERSION, "0.2.180")
        self.assertEqual(bundled.name, "claude")

    def test_real_bundled_cli_reports_the_pinned_version(self):
        bundled, _ = opts._bundled_cli()
        if not bundled.is_file():
            self.skipTest("bundled CLI not present")
        runtime = opts.resolve_runtime(None, {"PATH": os.defpath, "HOME": os.fspath(self.root)})
        self.assertEqual((runtime.cli_version, runtime.executable), ("2.1.321", os.fspath(bundled.resolve())))
        self.assertEqual(len(runtime.sha256), 64)


DENIED = {"WebSearch", "WebFetch", "AskUserQuestion", "EnterPlanMode", "ExitPlanMode", "Agent", "Task"}


def _runtime_problem():
    from ni_rua.cortex.agents import factory
    return factory.claude_code_runtime_problem()


@unittest.skipIf(_runtime_problem() is not None, f"pinned Claude Code SDK/CLI unavailable: {_runtime_problem()}")
class RealBinaryPermissionPolicy(unittest.TestCase):
    """Starts the pinned CLI through the real launcher and options against loopback fakes only (no model, no network):
    the init tool list holds none of the denied tools, and a scripted native ``Bash`` call and an ``mcp__ni_rua__*``
    call both run under ``bypassPermissions`` with nothing prompted or denied."""

    def test_subagents_on_by_default_keeps_the_subagent_tool(self):
        _, seen, _, _ = self.drive(ClaudeCodeConfig())
        self.assertIn("Task", seen["init"]["tools"])
        self.assertFalse(DENIED - {"Agent", "Task"} & set(seen["init"]["tools"]))

    def test_default_tool_set_minus_denylist_and_no_permission_prompt(self):
        launch, seen, server, mcp_calls = self.drive(ClaudeCodeConfig(subagents=False))
        tools = seen["init"]["tools"]
        self.assertIn("mcp__ni_rua__look", tools)
        native = opts.native_tools_block(launch, tools)
        self.assertTrue(native["enabled"], "the CLI default tool set is read back from init, never hand-written")
        self.assertIn("Bash", native["enabled"])
        self.assertFalse(DENIED & set(tools), sorted(DENIED & set(tools)))
        self.assertEqual(seen["init"]["permissionMode"], "bypassPermissions")
        # every request of the CLI lists the same: no denied tool reaches the model
        for request in server.generation_requests():
            listed = {tool["name"] for tool in (request.json.get("tools") or [])}
            self.assertFalse(DENIED & listed, sorted(DENIED & listed))
        self.assertEqual(seen["permission_messages"], [])
        self.assertEqual(getattr(seen["result"], "permission_denials", None) or [], [])
        self.assertFalse(seen["result"].is_error, seen["result"])
        self.assertEqual(len(seen["results"]), 2)
        self.assertFalse(any(block.is_error for block in seen["results"]), seen["results"])
        self.assertIn("BASH-OK-4417", json.dumps([block.content for block in seen["results"]]))
        self.assertEqual(mcp_calls, [("look", {"x": 1})])

    def drive(self, config):
        from claude_agent_sdk import ClaudeSDKClient, ResultMessage, SystemMessage, ToolResultBlock, UserMessage
        from ni_rua.core.presentation.actor_presentation import ActorPresentation
        from ni_rua.core.storage.store import Store
        from ni_rua.cortex.agents.mcp_server import NiRuaMcpServer
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name).resolve()
        run_dir = root / "run"
        run_dir.mkdir()
        store = Store(root / "store", dict(run_id="r", episode_id="e", env_instance_id="env", env_generation=1))
        self.addCleanup(store.close)
        mcp_calls = []

        def call_tool(name, arguments, **_):
            mcp_calls.append((name, arguments))
            return {"success": True, "contentItems": [{"type": "inputText", "text": "scene: red cube"}]}

        schema = {"type": "object", "properties": {"x": {"type": "number", "description": "x in m, default 0"}}}
        mcp = NiRuaMcpServer(
            tools=[{"type": "function", "name": "look", "description": "Read-only view.", "inputSchema": schema}],
            presentation=ActorPresentation(store), call_tool=call_tool, raw_log=lambda *_: None,
            on_delivery_failure=lambda *_: None, parallel_hints={"look": True}).start()
        self.addCleanup(mcp.close)

        def router(request):
            body = request.json if isinstance(request.json, dict) else {}
            if not body.get("tools"):
                return text_step("side-ok")
            results = sum(1 for message in body.get("messages") or [] if isinstance(message.get("content"), list)
                          for block in message["content"] if block.get("type") == "tool_result")
            if results == 0:
                return tool_step("Bash", {"command": "echo BASH-OK-4417", "description": "echo a marker"},
                                 id="toolu_bash_1")
            if results == 1:
                return tool_step("mcp__ni_rua__look", {"x": 1}, id="toolu_look_1")
            return text_step("FINAL")

        server = FakeAnthropicServer([], default=router, gate_timeout_s=60).start()
        self.addCleanup(server.close)
        launch = opts.build_launch(
            run_dir, model="claude-opus-5-5", effort="low", thread_id=SESSION, instructions="Be careful.",
            relay_base_url=server.base_url, mcp_url=mcp.url, mcp_headers=mcp.headers, config=config)
        seen = dict(init=None, results=[], result=None, denials=[], permission_messages=[])

        async def drive():
            async with ClaudeSDKClient(launch.options) as client:
                await client.query("Run the scripted tools, then answer.")
                async for message in client.receive_response():
                    if isinstance(message, SystemMessage):
                        if message.subtype == "init":
                            seen["init"] = message.data
                        if "permission" in message.subtype:
                            seen["permission_messages"].append(message.subtype)
                    elif isinstance(message, UserMessage) and isinstance(message.content, list):
                        seen["results"] += [block for block in message.content if isinstance(block, ToolResultBlock)]
                    elif isinstance(message, ResultMessage):
                        seen["result"] = message

        asyncio.run(asyncio.wait_for(drive(), 120))
        return launch, seen, server, mcp_calls


class NoSdkNeeded(unittest.TestCase):
    def test_constants_and_note(self):
        self.assertEqual(opts.SUBAGENT_TOOL_NAMES, ("Agent", "Task"))
        self.assertFalse(opts.LIVE_EFFORT)
        self.assertEqual(opts.TOOL_NAME_PREFIX, "mcp__ni_rua__")
        self.assertIn("'mcp__ni_rua__'", opts.naming_note())

# ==== tests/test_claude_settings_env.py ====

SETTINGS_ENV_TOKEN = "sk-test-claude-settings-env-token"
SECRET_HEADER = "x-provider-secret: header-value-never-recorded"
SETTINGS_ENV_SESSION = "9e7c96f2-1111-4222-8333-444455556666"
SETTINGS_ENV_FAKE_CLI = '#!/bin/sh\nif [ "$1" = "--version" ]; then echo "2.1.321 (Claude Code)"; exit 0; fi\nexec /usr/bin/env\n'
QUIET = {"self_briefing": False, "visual_review_model": None}


class SettingsCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()

    def write(self, name, env, **extra):
        path = self.root / name
        path.write_text(json.dumps({"env": env, **extra}))
        return path


class ReadSettingsEnvTests(SettingsCase):
    def test_forwards_provider_entries_never_credentials(self):
        path = self.write("tr.json", {"ANTHROPIC_BASE_URL": "https://gw.example/gateway", "ANTHROPIC_AUTH_TOKEN": SETTINGS_ENV_TOKEN,
                                      "ANTHROPIC_API_KEY": SETTINGS_ENV_TOKEN, "CLAUDE_CODE_THINKING_DISPLAY_UPDATES": "0",
                                      "ANTHROPIC_CUSTOM_HEADERS": SECRET_HEADER})
        self.assertEqual(opts.read_claude_settings_env(path), {"CLAUDE_CODE_THINKING_DISPLAY_UPDATES": "0",
                                                                "ANTHROPIC_CUSTOM_HEADERS": SECRET_HEADER})
        self.assertEqual(opts.read_claude_settings_env(self.write("plain.json", {"ANTHROPIC_AUTH_TOKEN": SETTINGS_ENV_TOKEN})), {})

    def test_ni_rua_owned_names_are_refused_naming_file_and_variable(self):
        for name in ("CLAUDE_CONFIG_DIR", "CLAUDE_CODE_MAX_RETRIES", "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_MODEL",
                     "PATH", "HOME", "LANG", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_DISABLE_AUTO_MEMORY",
                     "CLAUDE_STREAM_IDLE_TIMEOUT_MS", "IS_SANDBOX", "CLAUDE_CODE_EMIT_SESSION_STATE_EVENTS"):
            with self.subTest(name=name):
                path = self.write("owned.json", {"ANTHROPIC_AUTH_TOKEN": SETTINGS_ENV_TOKEN, name: "value-never-shown"})
                with self.assertRaises(ValueError) as raised:
                    opts.read_claude_settings_env(path)
                message = str(raised.exception)
                self.assertTrue(message.startswith(f"claude_settings_env_reserved:{path}: {name} "), message)
                self.assertNotIn("value-never-shown", message)
                self.assertNotIn(SETTINGS_ENV_TOKEN, message)

    def test_values_must_be_strings_and_names_shell_names(self):
        for env, code in (({"CLAUDE_CODE_THINKING_DISPLAY_UPDATES": 0}, "claude_settings_env_not_string"),
                          ({"CLAUDE_CODE_MAX_OUTPUT_TOKENS": None}, "claude_settings_env_not_string"),
                          ({"BAD-NAME": "1"}, "claude_settings_env_invalid_name")):
            with self.subTest(code=code):
                with self.assertRaises(ValueError) as raised:
                    opts.read_claude_settings_env(self.write("bad.json", {"ANTHROPIC_AUTH_TOKEN": SETTINGS_ENV_TOKEN, **env}))
                self.assertTrue(str(raised.exception).startswith(code), str(raised.exception))

    def test_launch_gate_refuses_an_owned_name(self):
        good = self.write("good.json", {"ANTHROPIC_AUTH_TOKEN": SETTINGS_ENV_TOKEN, "CLAUDE_CODE_THINKING_DISPLAY_UPDATES": "0"})
        bad = self.write("bad.json", {"ANTHROPIC_AUTH_TOKEN": SETTINGS_ENV_TOKEN, "CLAUDE_CODE_MAX_RETRIES": "3"})
        with mock.patch.object(factory, "claude_code_runtime_problem", return_value=None):
            for path, expected in ((good, None), (bad, "claude_settings_env_reserved")):
                config = Config.from_dict({"agent": {"backend": "claude_code", "claude_code": {"upstream": {
                    "source": "claude_settings", "settings": str(path)}}}, "model": {"name": "m"}, "features": QUIET})
                if expected is None:
                    self.assertIsNone(factory.backend_requirements(config))
                else:
                    with self.assertRaisesRegex(ConfigError, "^" + expected):
                        factory.backend_requirements(config)

    def test_archived_config_holds_only_the_path(self):
        self.write("claude.local.json", {"ANTHROPIC_AUTH_TOKEN": SETTINGS_ENV_TOKEN, "ANTHROPIC_CUSTOM_HEADERS": SECRET_HEADER})
        config_path = self.root / "ni-rua.json"
        config_path.write_text(json.dumps({"agent": {"backend": "claude_code", "claude_code": {"upstream": {
            "source": "claude_settings", "settings": "claude.local.json"}}}}))
        dumped = json.dumps(Config.load(config_path, environ={}).to_dict())
        self.assertNotIn(SETTINGS_ENV_TOKEN, dumped)
        self.assertNotIn(SECRET_HEADER, dumped)


@requires_sdk
class LaunchForwardingTests(SettingsCase):
    def setUp(self):
        super().setUp()
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()
        cli = self.root / "fake-claude"
        cli.write_text(SETTINGS_ENV_FAKE_CLI)
        cli.chmod(0o755)
        self.runtime = opts.resolve_runtime(cli, {"PATH": os.defpath})

    def build(self, **overrides):
        arguments = dict(model="kimi-k2.6", effort="high", thread_id=SETTINGS_ENV_SESSION, instructions="Be careful.\n",
                         relay_base_url="http://127.0.0.1:41000", mcp_url="http://127.0.0.1:41001/mcp",
                         mcp_headers={"Authorization": "Bearer tok"}, runtime=self.runtime)
        arguments.update(overrides)
        return opts.build_launch(self.run_dir, **arguments)

    def test_forwarded_values_reach_env_and_launcher_names_only_in_provenance(self):
        forwarded = {"CLAUDE_CODE_THINKING_DISPLAY_UPDATES": "0", "ANTHROPIC_CUSTOM_HEADERS": SECRET_HEADER}
        launch = self.build(settings_env=forwarded)
        for name, value in forwarded.items():
            self.assertEqual(launch.env[name], value)
        script = launch.launcher.read_text()
        self.assertIn("'CLAUDE_CODE_THINKING_DISPLAY_UPDATES=0'", script)
        self.assertEqual(launch.native_settings["settings_env"],
                         ["ANTHROPIC_CUSTOM_HEADERS", "CLAUDE_CODE_THINKING_DISPLAY_UPDATES"])
        recorded = json.dumps(launch.native_settings)
        self.assertNotIn(SECRET_HEADER, recorded)
        self.assertNotIn("CLAUDE_CODE_THINKING_DISPLAY_UPDATES", json.dumps(launch.native_settings["env"]))
        self.assertEqual(launch.env["ANTHROPIC_API_KEY"], opts.PLACEHOLDER_API_KEY)   # never the provider key
        self.assertEqual(self.build().native_settings["settings_env"], [])

    def test_build_launch_refuses_owned_or_credential_names(self):
        for name in ("CLAUDE_CODE_MAX_RETRIES", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, f"^claude_settings_env_reserved:{name}$"):
                self.build(settings_env={name: "x"})

    def test_every_variable_ni_rua_sets_is_owned(self):
        """Drift guard: a new variable in build_launch must be added to NI_RUA_OWNED_ENV (or a settings file could
        override it)."""
        with mock.patch.object(opts, "_running_as_root", return_value=True):
            launch = self.build(native_memories=False, instance_id="agent-1",
                                config=opts.ClaudeCodeConfig(auto_compact_window_tokens=200000))
        self.assertEqual(set(launch.env) - opts.NI_RUA_OWNED_ENV, set())

    def test_sandbox_launch_carries_forwarded_values_into_the_namespace(self):
        created = {}

        class Sandbox:
            def __init__(self, *args, **keywords):
                created.update(keywords)

            def prepare(self):
                return ["/usr/bin/env", "-i", "/usr/bin/bwrap", "--", "/usr/bin/python3", "-c", "pass"], {}

            def close(self):
                pass

        plan = opts.SandboxPlan(ipc=self.root / "ipc", ports=(41000, 41001))
        with mock.patch.object(opts, "AgentSandbox", Sandbox):
            launch = self.build(settings_env={"CLAUDE_CODE_THINKING_DISPLAY_UPDATES": "0"}, sandbox=plan)
        self.assertEqual(created["env"]["CLAUDE_CODE_THINKING_DISPLAY_UPDATES"], "0")
        self.assertIn("CLAUDE_CODE_THINKING_DISPLAY_UPDATES", launch.native_settings["sandbox"]["sandbox_env"])


@requires_sdk
class SessionForwardingTests(SettingsCase):
    def test_session_reads_the_forwarded_env_from_the_settings_file(self):
        from ni_rua.core.storage.store import Store
        from ni_rua.cortex.agents.claude_code.session import ClaudeCodeSession
        path = self.write("tr.json", {"ANTHROPIC_BASE_URL": "http://127.0.0.1:9", "ANTHROPIC_AUTH_TOKEN": SETTINGS_ENV_TOKEN,
                                      "CLAUDE_CODE_THINKING_DISPLAY_UPDATES": "0"})
        run = self.root / "run"
        run.mkdir()
        store = Store(run, dict(run_id="r", episode_id="e", env_instance_id="i", env_generation=1))
        self.addCleanup(store.close)
        config = Config.from_dict({"agent": {"backend": "claude_code", "claude_code": {"upstream": {
            "source": "claude_settings", "settings": str(path)}}}, "model": {"name": "m"}, "features": QUIET})
        session = ClaudeCodeSession(run, store, lambda params: None, config=config)
        self.assertEqual(session._settings_env(), {"CLAUDE_CODE_THINKING_DISPLAY_UPDATES": "0"})
        explicit = Config.from_dict({"agent": {"backend": "claude_code"}, "model": {"name": "m"}, "features": QUIET})
        self.assertEqual(ClaudeCodeSession(run, store, lambda params: None, config=explicit)._settings_env(), {})

# ==== tests/test_claude_settings_upstream.py ====

UPSTREAM_TOKEN = "sk-test-claude-settings-token"


class ClaudeSettingsUpstreamTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value))
        return path

    def test_official_format_bearer_token_and_api_key(self):
        path = self.write("claude.json", {"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:3000/",
                                                   "ANTHROPIC_AUTH_TOKEN": UPSTREAM_TOKEN}, "model": "opus"})
        self.assertEqual(read_claude_settings_upstream(path),
                         ("http://127.0.0.1:3000", {"Authorization": "Bearer " + UPSTREAM_TOKEN}))
        path = self.write("key.json", {"env": {"ANTHROPIC_API_KEY": UPSTREAM_TOKEN}})
        self.assertEqual(read_claude_settings_upstream(path), ("https://api.anthropic.com", {"x-api-key": UPSTREAM_TOKEN}))

    def test_errors_name_the_file_never_the_key(self):
        for name, value, code in (("nokey.json", {"env": {"ANTHROPIC_BASE_URL": "http://x"}}, "claude_settings_missing_key"),
                                  ("noenv.json", {"model": "opus"}, "claude_settings_missing_env")):
            with self.assertRaises(ValueError) as raised:
                read_claude_settings_upstream(self.write(name, value))
            self.assertTrue(str(raised.exception).startswith(code), str(raised.exception))
        with self.assertRaises(ValueError) as raised:
            read_claude_settings_upstream(self.root / "missing.json")
        self.assertTrue(str(raised.exception).startswith("claude_settings_unreadable"))

    def test_config_resolves_relative_path_and_never_archives_the_key(self):
        self.write("claude.local.json", {"env": {"ANTHROPIC_AUTH_TOKEN": UPSTREAM_TOKEN}})
        config_path = self.write("ni-rua.json", {"agent": {"backend": "claude_code", "claude_code": {"upstream": {
            "source": "claude_settings", "settings": "claude.local.json"}}}})
        config = Config.load(config_path, environ={})
        self.assertEqual(Path(config.agent.claude_code.upstream.settings), (self.root / "claude.local.json").resolve())
        self.assertNotIn(UPSTREAM_TOKEN, json.dumps(config.to_dict()))

    def test_settings_path_is_required(self):
        with self.assertRaises(ConfigError):
            Config.from_dict({"agent": {"claude_code": {"upstream": {"source": "claude_settings"}}}})

    def test_launch_gate_reads_the_file(self):
        good = self.write("good.json", {"env": {"ANTHROPIC_AUTH_TOKEN": UPSTREAM_TOKEN}})
        bad = self.write("bad.json", {"env": {}})
        quiet = {"self_briefing": False, "visual_review_model": None}
        with mock.patch.object(factory, "claude_code_runtime_problem", return_value=None):
            config = Config.from_dict({"agent": {"backend": "claude_code", "claude_code": {"upstream": {
                "source": "claude_settings", "settings": str(good)}}}, "model": {"name": "m"}, "features": quiet})
            self.assertIsNone(factory.backend_requirements(config))
            config = Config.from_dict({"agent": {"backend": "claude_code", "claude_code": {"upstream": {
                "source": "claude_settings", "settings": str(bad)}}}, "model": {"name": "m"}, "features": quiet})
            with self.assertRaisesRegex(ConfigError, "^claude_settings_missing_key"):
                factory.backend_requirements(config)

    def test_session_upstream_uses_the_settings_file(self):
        from ni_rua.cortex.agents.claude_code.session import ClaudeCodeSession
        from ni_rua.core.storage.store import Store
        path = self.write("claude.json", {"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:9", "ANTHROPIC_AUTH_TOKEN": UPSTREAM_TOKEN}})
        run = self.root / "run"
        run.mkdir()
        store = Store(run, dict(run_id="r", episode_id="e", env_instance_id="i", env_generation=1))
        self.addCleanup(store.close)
        config = Config.from_dict({"agent": {"backend": "claude_code", "claude_code": {"upstream": {
            "source": "claude_settings", "settings": str(path)}}}, "model": {"name": "m"},
            "features": {"self_briefing": False, "visual_review_model": None}})
        with mock.patch.dict(os.environ, {}, clear=False):
            session = ClaudeCodeSession(run, store, lambda params: None, config=config)
            self.assertEqual(session._upstream(), ("http://127.0.0.1:9", {"Authorization": "Bearer " + UPSTREAM_TOKEN}))

    def test_malformed_settings_json_retains_no_context_doc_with_secret_plaintext(self):
        broken = self.root / "broken_settings.json"
        secret = "sk-ant-super-secret-plaintext-888"
        broken.write_text('{"env": {"ANTHROPIC_API_KEY": "' + secret + '" invalid')
        with self.assertRaisesRegex(ValueError, "^claude_settings_unreadable:") as caught:
            opts.read_claude_settings_upstream(broken)
        self.assertIsNone(caught.exception.__context__)
        self.assertIsNone(caught.exception.__cause__)
        self.assertNotIn(secret, str(caught.exception))

if __name__ == "__main__":
    unittest.main()
