"""Consolidated tests:

- test_agent_queue_conditions: Queue case keys, frozen settings and schema drift for agent_backend; library and report condition fields.
- test_experiment_hook: No model/GPU calls: verify main-thread delivery, deduplication and kernel wait.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from ni_rua.apps.evaluation.report import summarize
from ni_rua.apps.library.experiment_comparisons import ExperimentGroups
from ni_rua.apps.library.experiments import Catalog, atomic_json, register_run
from ni_rua.core.config import Config
from ni_rua.core.storage.store import Store
from scripts import benchmark_queue as queue
import importlib.util
import subprocess
import sys
from support.host_capabilities import linux_proc, require
from support.paths import REPO_ROOT


NI_RUA_TEST_TIER = "component"

# --- test_agent_queue_conditions.py ---
class QueueAgentTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.checkout = self.root / 'checkout'
        (self.checkout / 'ni_rua/apps/gateway').mkdir(parents=True)
        (self.checkout / 'ni_rua/core').mkdir(parents=True)
        (self.checkout / 'ni_rua/apps/gateway/cli.py').write_text("parser.add_argument('--config')\nConfig.load()\n")
        self.case = dict(cwd=str(self.checkout), run_dir=str(self.root / 'run'), task='stack_bowls',
                         seed=0, layout_id=0, title='agent', tag=[])

    def test_case_keys_become_cli_flags(self):
        command = queue.command({**self.case, 'agent_backend': 'claude_code', 'model': 'claude-x', 'effort': 'medium'}, 1)
        for flag, value in (('--agent-backend', 'claude_code'), ('--model', 'claude-x'), ('--effort', 'medium')):
            self.assertEqual(command[command.index(flag) + 1], value)
        plain = queue.command(self.case, 1)
        for flag in ('--agent-backend', '--model', '--effort'):
            self.assertNotIn(flag, plain)

    def test_invalid_case_values_are_rejected(self):
        for update in ({'agent_backend': 'gemini'}, {'agent_backend': ''}, {'agent_backend': None}, {'model': ''},
                       {'effort': 'extreme'}, {'effort': 3}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                queue.command({**self.case, **update}, 0)

    def test_frozen_settings_stay_unchanged_for_codex_and_record_env_name_otherwise(self):
        self.assertEqual(queue.agent_settings(Config(), 'codex'), {'agent_backend': 'codex'})
        settings = queue.agent_settings(Config(), 'claude_code')
        self.assertEqual(settings, {'agent_backend': 'claude_code', 'agent_upstream_api_key_env': 'ANTHROPIC_API_KEY'})
        config = Config.from_dict({'agent': {'backend': 'opencode', 'opencode': {'upstream': {'api_key_env': 'OC_KEY'}}}})
        self.assertEqual(queue.agent_settings(config, 'opencode')['agent_upstream_api_key_env'], 'OC_KEY')
        self.assertNotIn('agent', queue.configured_manifest({'cases': [self.case]}, Config())['ni_rua_config'])

    def _old_schema_checkout(self):
        config_dir = REPO_ROOT / 'ni_rua/core/config'
        source = "\n".join(p.read_text() for p in sorted(config_dir.glob('*.py')))
        old = source.replace('class AgentConfig(Section):', 'class RetiredSectionConfig(Section):').replace(
            '    agent: AgentConfig = field(default_factory=AgentConfig)\n', '')
        self.assertNotIn('agent: AgentConfig', old)
        (self.checkout / 'ni_rua/core/config.py').write_text(old)

    def test_old_checkout_schema_rejects_agent_backend_case_without_downgrading(self):
        self._old_schema_checkout()
        manifest = queue.configured_manifest({'cases': [{**self.case, 'agent_backend': 'codex'}]}, Config())
        output = self.root / 'out'
        with patch.object(queue, 'sample_gpus') as sample, patch.object(queue.subprocess, 'Popen') as launch, \
                self.assertRaisesRegex(ValueError, 'unified_config_incompatible:agent'):
            queue.run_queue(manifest, output)
        sample.assert_not_called()
        launch.assert_not_called()
        self.assertFalse(output.exists())

    def test_old_checkout_schema_rejects_non_default_agent_section(self):
        self._old_schema_checkout()
        config = Config.from_dict({'agent': {'backend': 'claude_code'}})
        manifest = queue.configured_manifest({'cases': [self.case]}, config)
        with patch.object(queue, 'sample_gpus'), self.assertRaisesRegex(ValueError, 'unified_config_incompatible:agent'):
            queue.run_queue(manifest, self.root / 'out2')

    def test_codex_default_projection_ignores_old_checkout_without_agent(self):
        self._old_schema_checkout()
        schema = queue.runtime_config_schema(self.checkout)
        document = Config().to_dict()
        self.assertEqual(queue.project_config_for_runtimes(document, [schema]), document)

    def test_new_checkout_schema_declares_agent_config(self):
        schema = queue.runtime_config_schema(REPO_ROOT)
        self.assertIn('AgentConfig', schema)
        projected = queue.project_config_for_runtimes(Config.from_dict({'agent': {'backend': 'opencode'}}).to_dict(), [schema])
        self.assertEqual(projected['agent']['backend'], 'opencode')


class ConditionFieldTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = self.root / 'benchmark' / 'panel' / 'manifest.json'
        self.row = dict(case_id='pick__standard__g0__l0', task='pick', variant='standard', eval_seed=0, layout_id=0)

    def groups(self, config, case_extra=None):
        run = self.root / 'benchmark' / 'panel' / 'runs' / 'one'
        register_run(run, category='benchmark', task='pick', run_id='one')
        atomic_json(run / 'summary.json', dict(normalized_score=1, task_success=True, valid_episode=True,
                                               scored_episode=True, status='completed'))
        case = dict(run_dir=str(run), cwd='/source/x', task='pick', seed=0, layout_id=0, tag=['cand', self.row['case_id']],
                    **(case_extra or {}))
        manifest = dict(benchmark_group='t', method='paired', protocol_cases=[self.row], cases=[case],
                        experiment_panels={'cand': [self.row['case_id']]})
        if config is not None:
            manifest['ni_rua_config'] = config
        atomic_json(self.path, manifest)
        catalog = Catalog(self.root)
        catalog.scan()
        return {g['condition']: g for g in ExperimentGroups(self.root).snapshot(catalog)['groups']}['cand']

    def test_old_archive_without_agent_section_reads_as_codex(self):
        self.assertEqual(self.groups({'model': {'name': 'm'}})['conditions']['agent_backend'], 'codex')
        self.assertEqual(self.groups(None)['conditions']['agent_backend'], 'codex')

    def test_group_reads_configured_backend_and_case_override(self):
        config = Config.from_dict({'agent': {'backend': 'claude_code'}}).to_dict()
        self.assertEqual(self.groups(config)['conditions']['agent_backend'], 'claude_code')
        self.assertEqual(self.groups(config, {'agent_backend': 'opencode'})['conditions']['agent_backend'], 'opencode')
        self.assertEqual(self.groups(Config().to_dict(), {'agent_backend': 'opencode'})['conditions']['agent_backend'], 'opencode')


class ReportConditionTests(unittest.TestCase):
    def report(self, manifest):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Store(root, dict(run_id='r', episode_id='e', env_instance_id='i', env_generation=1)).close()
            (root / 'manifest.json').write_text(json.dumps(dict(run_id='r', **manifest)))
            (root / 'summary.json').write_text(json.dumps(dict(status='unknown')))
            return summarize(root)['conditions']

    def test_old_manifest_is_codex_with_no_cli_record(self):
        conditions = self.report(dict(codex_sdk='0.160.0'))
        self.assertEqual((conditions['agent_backend'], conditions['agent_cli'], conditions['codex_sdk']), ('codex', None, '0.160.0'))

    def test_manifest_block_supplies_backend_and_cli(self):
        cli = dict(version='1.19.5', executable='/x/opencode', sha256='ab')
        conditions = self.report(dict(agent_backend=dict(kind='opencode', cli=cli)))
        self.assertEqual((conditions['agent_backend'], conditions['agent_cli']), ('opencode', '1.19.5'))
        before = self.report(dict(codex_sdk='0.160.0', codex_cli='0.160.0'))
        after = self.report(dict(codex_sdk='0.160.0', codex_cli='0.160.0', agent_backend=dict(
            kind='codex', cli=dict(version='0.160.0', executable='/machine/specific/codex', sha256=None))))
        self.assertEqual((before['agent_backend'], before['agent_cli']), (after['agent_backend'], after['agent_cli']))
        conditions = self.report(dict(agent_backend=dict(kind='claude_code', sdk='0.2.180')))
        self.assertEqual((conditions['agent_backend'], conditions['agent_cli']), ('claude_code', None))


# --- test_experiment_hook.py ---
spec = importlib.util.spec_from_file_location("experiment_hook", REPO_ROOT / "scripts/experiment_hook.py")
hook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hook)
THREAD = "01900000-0000-7000-8000-000000000001"


def fake_codex(root):
    calls = root / "calls.jsonl"
    codex = root / "fake-codex"
    codex.write_text(f'''#!{sys.executable}
import json,sys
from pathlib import Path
with Path({str(calls)!r}).open('a') as stream: stream.write(json.dumps(sys.argv[1:])+'\\n')
print('notification accepted')
print('fixture stderr', file=sys.stderr)
raise SystemExit(9 if Path({str(root / 'fail')!r}).exists() else 0)
''')
    codex.chmod(0o700)
    return codex, calls


class CompletionHookTests(unittest.TestCase):
    @require(linux_proc, "binds processes by /proc/<pid>/stat start ticks")
    def test_pidfd_adoption_accepts_packaged_entrypoints_and_rejects_unrelated_commands(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "queue"
            output.mkdir()
            package = root / "ni_rua/apps/evaluation"
            package.mkdir(parents=True)
            (root / "ni_rua/__init__.py").write_text("")
            (root / "ni_rua/apps/__init__.py").write_text("")
            (package / "__init__.py").write_text("")
            source = "import sys,time\nprint('ready', flush=True)\nsys.stdin.readline()\ntime.sleep(.1)\n"
            (package / "queue.py").write_text(source)
            for name in ("ni-rua-queue", "unrelated.py"):
                (root / name).write_text(source)
            commands = ((["-u", "-m", "ni_rua.apps.evaluation.queue"], True),
                        ([str(root / "ni-rua-queue")], True),
                        (["-m", "unrelated", "ni_rua.apps.evaluation.queue"], False),
                        ([str(root / "unrelated.py"), "benchmark_queue.py"], False))
            for command, accepted in commands:
                with self.subTest(command=command):
                    process = subprocess.Popen([sys.executable, *command, "--output", str(output)], cwd=root,
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
                    try:
                        self.assertEqual(process.stdout.readline().strip(), "ready")
                        ticks = hook.process_identity(process.pid)
                        if accepted:
                            with self.assertRaisesRegex(RuntimeError, "reused"):
                                hook.wait_queue(output, process.pid, ticks + 1)
                            with self.assertRaisesRegex(RuntimeError, "output differs"):
                                hook.wait_queue(root / "other", process.pid, ticks)
                            process.stdin.write("exit\n")
                            process.stdin.flush()
                            hook.wait_queue(output, process.pid, ticks)
                            self.assertEqual(process.wait(timeout=5), 0)
                        else:
                            with self.assertRaisesRegex(RuntimeError, "not an NI-RUA benchmark queue"):
                                hook.wait_queue(output, process.pid, ticks)
                    finally:
                        if process.poll() is None:
                            process.stdin.write("exit\n")
                            process.stdin.flush()
                        process.wait(timeout=5)
                        process.stdin.close()
                        process.stdout.close()

    def test_stopped_queue_can_report_unconfirmed_cleanup_without_claiming_process_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            codex, calls = fake_codex(root)
            for status in ("stop_unconfirmed", "exited"):
                output = root / status
                output.mkdir()
                hook.save_json(output / "state.json", {"status": "failed", "cleanup_stop_unconfirmed": True,
                    "cases": [{"status": status, "queue_cleanup": {"native_close_confirmed": False}}]})
                receipt = hook.notify_completion(output, THREAD, codex, queue_stopped=True)
                self.assertEqual(receipt["status"], "sent")
                prompt = (output / "completion-hook/prompt.txt").read_text()
                self.assertIn("实验调度已结束", prompt)
                self.assertIn("正常关闭未确认或发生强制回收", prompt)
                self.assertNotIn("进程已退出", prompt)
            self.assertEqual(len(calls.read_text().splitlines()), 2)
            active = root / "active"
            active.mkdir()
            hook.save_json(active / "state.json", {"status": "running", "cases": [{"status": "running"}]})
            with self.assertRaisesRegex(RuntimeError, "no terminal"):
                hook.notify_completion(active, THREAD, codex, queue_stopped=True)

    def test_existing_thread_notification_and_uncertain_delivery_never_repeat(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            codex, calls = fake_codex(root)
            output = root / "queue"
            output.mkdir()
            hook.save_json(output / "state.json", {"status": "failed", "cases": [
                {"status": "exited", "returncode": 0, "summary": {"status": "completed"}},
                {"status": "exited", "returncode": 1, "summary": None},
                {"status": "exited", "returncode": 0},
                {"status": "not_started"}]})
            receipt = hook.notify_completion(output, THREAD, codex, remote="unix:///tmp/main-server.sock")
            self.assertEqual(receipt["status"], "sent")
            argv = json.loads(calls.read_text().splitlines()[0])
            self.assertEqual(argv[:4], ["queue", "--thread", THREAD, "--message"])
            self.assertEqual(argv[5:], ["--remote", "unix:///tmp/main-server.sock"])
            prompt = argv[4]
            self.assertIn(str(output.resolve()), prompt)
            self.assertEqual(prompt, f"实验已结束，结果目录：{output}")
            self.assertEqual(len(prompt.splitlines()), 1)
            artifacts = output / "completion-hook"
            self.assertEqual((artifacts / "prompt.txt").read_text(), prompt)
            self.assertEqual(hook.read_json(artifacts / "command.json")["argv"], [str(codex), *argv])
            self.assertEqual((artifacts / "stdout.log").read_text().strip(), "notification accepted")
            self.assertEqual((artifacts / "stderr.log").read_text().strip(), "fixture stderr")
            self.assertTrue(hook.notify_completion(output, THREAD, codex)["already_claimed"])
            with self.assertRaisesRegex(ValueError, "different thread"):
                hook.notify_completion(output, "01900000-0000-7000-8000-000000000002", codex)
            self.assertEqual(len(calls.read_text().splitlines()), 1)

            interrupted = root / "interrupted"
            interrupted.mkdir()
            hook.save_json(interrupted / ".main-session-wake.json", {"thread_id": THREAD, "status": "delivery_unknown"})
            self.assertEqual(hook.notify_completion(interrupted, THREAD, codex)["status"], "delivery_unknown")
            self.assertEqual(len(calls.read_text().splitlines()), 1)

            failure = root / "failure"
            failure.mkdir()
            hook.save_json(failure / "state.json", {"status": "gpu_wait_timeout", "cases": [{"status": "not_started"}]})
            (root / "fail").touch()
            receipt = hook.notify_completion(failure, THREAD, codex)
            self.assertEqual((receipt["status"], receipt["returncode"]), ("delivery_unknown", 9))
            self.assertTrue(hook.notify_completion(failure, THREAD, codex)["already_claimed"])
            self.assertEqual(len(calls.read_text().splitlines()), 2)
            (root / "fail").unlink()

            missing = root / "missing"
            missing.mkdir()
            receipt = hook.notify_completion(missing, THREAD, root / "absent-codex", process_exited=True)
            self.assertEqual((receipt["status"], receipt["returncode"]), ("delivery_unknown", None))
            self.assertIn("实验队列进程已退出", (missing / "completion-hook/prompt.txt").read_text())
            self.assertIn("state_error", hook.read_json(missing / "completion-hook/queue-state.json"))
            self.assertTrue(hook.notify_completion(missing, THREAD, codex)["already_claimed"])
            self.assertEqual(len(calls.read_text().splitlines()), 2)

    @require(linux_proc, "binds processes by /proc/<pid>/stat start ticks")
    def test_pidfd_identity_wait_and_active_state_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "queue"
            output.mkdir()
            codex, calls = fake_codex(root)
            hook.save_json(output / "state.json", {"status": "finished", "cases": [{"status": "running"}]})
            with self.assertRaisesRegex(RuntimeError, "possibly active"):
                hook.notify_completion(output, THREAD, codex)
            hook.save_json(output / "state.json", {"status": "running", "cases": [{"status": "not_started"}]})
            with self.assertRaisesRegex(RuntimeError, "no terminal"):
                hook.notify_completion(output, THREAD, codex)
            self.assertFalse(calls.exists())
            self.assertFalse((output / ".main-session-wake.json").exists())

            script = root / "benchmark_queue.py"
            script.write_text("import sys,time\nprint('ready', flush=True)\nsys.stdin.readline()\ntime.sleep(.1)\n")
            process = subprocess.Popen([sys.executable, str(script), "--output", str(output)],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            try:
                self.assertEqual(process.stdout.readline().strip(), "ready")
                ticks = hook.process_identity(process.pid)
                with self.assertRaisesRegex(RuntimeError, "reused"):
                    hook.wait_queue(output, process.pid, ticks + 1)
                with self.assertRaisesRegex(RuntimeError, "output differs"):
                    hook.wait_queue(root / "other", process.pid, ticks)
                process.stdin.write("exit\n")
                process.stdin.flush()
                hook.wait_queue(output, process.pid, ticks)
                self.assertEqual(process.wait(timeout=5), 0)
                # Queue exited without updating its record: notify for diagnosis, not success.
                receipt = hook.notify_completion(output, THREAD, codex, process_exited=True)
                self.assertEqual(receipt["status"], "sent")
                prompt = (output / "completion-hook/prompt.txt").read_text()
                self.assertIn("记录不完整、部分实验停止未确认", prompt)
                with self.assertRaisesRegex(RuntimeError, "no terminal"):
                    hook.wait_queue(output, process.pid, ticks)
                hook.save_json(output / "state.json", {"status": "finished", "cases": [{"status": "exited"}]})
                hook.wait_queue(output, process.pid, ticks)
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
                process.stdin.close()
                process.stdout.close()



if __name__ == "__main__":
    unittest.main()
