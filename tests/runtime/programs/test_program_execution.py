"""
- test_program_output_delivery.py: Repeated log reads stay small without concealing live status or new context.
- test_program_result.py: Large log endings and fixed-skill outcomes survive model presentation.
- test_program_services.py: Program broker dependencies are explicit, dynamic and action-owner scoped.
- test_programs.py: Real isolated subprocess/client checks; synthetic robot, no GPU or model.
"""
from copy import deepcopy
import json
import unittest
from ni_rua.core.presentation.program_output_delivery import ProgramOutputDelivery
from ni_rua.runtime.programs.runner import ProgramRunner
from support.motion_fixtures import DualArmHarnessBase
import base64
import io
from pathlib import Path
import random
import tempfile
from PIL import Image
from ni_rua.runtime.programs.program_result import program_result
from ni_rua.core.storage.store import Store
from dataclasses import FrozenInstanceError
from ni_rua.runtime.harness import Harness
from ni_rua.runtime.programs.runner import ProgramRunner, ProgramServices
from ni_rua.core.tool_output import ToolOutput
from ni_rua.runtime.tools.registry import ToolSpec
from support.fixture_robot import FixtureRobot
from support.program_fixture import program_services
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch
from ni_rua.runtime.programs.runner import ProgramRunner, model_wait, program_remaining
from support.host_capabilities import program_isolation, require

NI_RUA_TEST_TIER = "component"


# --- Folded from tests/test_program_output_delivery.py ---

class ProgramOutputDeliveryTests(unittest.TestCase):
    def test_repetition_preserves_changed_status_and_independent_streams(self):
        delivery = ProgramOutputDelivery()
        result = dict(job_id='job', status='running', stdout_text='a' * 1000,
            stdout_ref='stdout1', stdout_truncated=True, stdout_excerpt={'byte_size': 2000, 'truncated': True},
            stderr_text='b' * 1000, stderr_ref='stderr1')
        original = deepcopy(result)
        first = delivery.project(result, epoch=['thread', 0], call_id='first', detail='summary')
        self.assertEqual(first, result)
        self.assertEqual(delivery.project(result, epoch=['thread', 0], call_id='first', detail='summary'), first)
        result.update(status='awaiting_review', pending_review={'review_id': 'r1'}, stderr_text='new failure' * 100)
        repeated = delivery.project(result, epoch=['thread', 0], call_id='poll', detail='summary')
        self.assertNotIn('stdout_text', repeated)
        self.assertNotIn('stdout_excerpt', repeated)
        self.assertNotIn('stdout_truncated', repeated)
        self.assertEqual(repeated['stdout_delivery']['previous_tool_call_id'], 'first')
        self.assertEqual(repeated['status'], 'awaiting_review')
        self.assertEqual(repeated['pending_review'], result['pending_review'])
        self.assertEqual(repeated['stderr_text'], result['stderr_text'])
        self.assertEqual(result['stdout_text'], original['stdout_text'])
        full = delivery.project(result, epoch=['thread', 0], call_id='full', detail='full')
        self.assertEqual(full, result)
        restored = delivery.project(result, epoch=['thread', 1], call_id='after-compaction', detail='summary')
        self.assertEqual(restored, result)

    def test_new_artifacts_jobs_and_excerpt_settings_are_not_suppressed(self):
        delivery = ProgramOutputDelivery()
        result = dict(job_id='job', stdout_text='same ends' * 100, stdout_ref='a',
            stdout_excerpt={'byte_size': 5000, 'source_complete': True})
        delivery.project(result, epoch=1, call_id='first', detail='summary')
        for changes in ({'stdout_ref': 'changed-middle'}, {'job_id': 'other-job'},
                {'stdout_excerpt': {'byte_size': 5000, 'source_complete': False}}):
            with self.subTest(changes=changes):
                changed = dict(result, **changes)
                self.assertIn('stdout_text', delivery.project(changed, epoch=1, call_id='next', detail='summary'))
        for epoch, call_id in ((None, 'call'), (1, None)):
            self.assertEqual(delivery.project(result, epoch=epoch, call_id=call_id, detail='summary'), result)
        # epoch=None 与过时纪元（epoch != live_epoch）直接返回深拷贝完整结果，且绝不清空活跃纪元 _streams 基线
        delivery.project(result, epoch='e-live', call_id='live-1', detail='summary', live_epoch='e-live')
        stale = delivery.project(result, epoch='e-old', call_id='stale-1', detail='summary', live_epoch='e-live')
        self.assertEqual(stale, result)
        self.assertIsNot(stale, result)
        iso = delivery.project(result, epoch=None, call_id='iso-1', detail='summary', live_epoch='e-live')
        self.assertEqual(iso, result)
        self.assertIsNot(iso, result)
        unchanged = delivery.project(result, epoch='e-live', call_id='live-2', detail='summary', live_epoch='e-live')
        self.assertEqual(unchanged['stdout_delivery']['status'], 'unchanged')
        self.assertEqual(unchanged['stdout_delivery']['previous_tool_call_id'], 'live-1')
        short = dict(job_id='short', stdout_text='done', stderr_text='')
        for call_id in ('a', 'b'):
            self.assertEqual(delivery.project(short, epoch=1, call_id=call_id, detail='summary'), short)


class ProgramOutputHarnessTests(unittest.TestCase):
    setUp = DualArmHarnessBase.setUp

    def call(self, name, arguments, call_id):
        wire = self.h.handle_tool(dict(threadId=self.h.thread_id, turnId='turn', callId=call_id,
            tool=name, arguments=arguments))
        self.assertTrue(wire['success'])
        return json.loads(wire['contentItems'][0]['text'])

    def test_legacy_and_program_readers_do_not_consume_actor_delivery(self):
        from ni_rua.runtime.tools.program_tools import program_display
        self.h.thread_id = 'program-log-reader-isolation'
        self.h.programs = ProgramRunner(self.h.program_services())
        self.addCleanup(self.h.programs.close)
        legacy = self.h.dispatch('program_run', {'source': "print('diagnostic ' * 100)", 'timeout_s': 10},
            {'thread_id': self.h.thread_id, 'turn_id': 'turn', 'tool_call_id': 'legacy-only-read'})
        data = json.loads(legacy['contentItems'][0]['text'])
        self.assertIn('stdout_text', data)
        self.assertEqual(program_display(self.h, data, 'summary', {'program_id': 'program',
            'actor_log_delivery': True, 'tool_call_id': 'program-only-read'}), data)
        polled = self.call('program_status', {'job_id': data['job_id']}, 'first-actor-read')
        self.assertIn('stdout_text', polled)
        self.assertNotIn('stdout_delivery', polled)
        self.assertEqual(self.robot.executions, 0)

    def test_program_reply_names_program_status_as_its_detail_entry_and_the_call_returns_the_full_result(self):
        from support.reply_defaults import with_reply_features
        self.h.config = with_reply_features(self.h.config, reply_static_once=True)
        self.h.thread_id = 'program-detail-entry'
        self.h.programs = ProgramRunner(self.h.program_services())
        self.addCleanup(self.h.programs.close)
        self.assertLessEqual({'program_run', 'program_status'}, self.h.tool_registry.loaded_tool_names(self.h))
        started = self.call('program_run', {'source': "print('begin|' + 'x' * 20000 + '|end')", 'timeout_s': 10}, 'start')
        self.assertNotIn('full_result', started)
        entry = started['detail']
        self.assertEqual(entry, dict(tool='program_status', arguments=dict(job_id=started['job_id'], detail='full')))
        # the call exactly as the reply gives it returns the complete stdout the summary only bounded
        full = self.call(entry['tool'], entry['arguments'], 'read-detail')
        self.assertEqual(full['stdout_text'], 'begin|' + 'x' * 20000 + '|end\n')
        self.assertEqual((full['job_id'], full['status']), (started['job_id'], 'succeeded'))
        self.assertEqual(self.robot.executions, 0)

    def test_failed_actor_display_does_not_mark_logs_as_delivered(self):
        from unittest.mock import patch
        self.h.thread_id = 'program-log-display-failure'
        self.h.programs = ProgramRunner(self.h.program_services())
        self.addCleanup(self.h.programs.close)
        for method in ('_present_part', 'present'):
            with self.subTest(failed_method=method):
                previous_ids = {job['job_id'] for job in self.store.states('program')}
                params = dict(threadId=self.h.thread_id, turnId='turn', callId='failed-' + method,
                    tool='program_run', arguments={'source': "print('important diagnostic ' * 100)", 'timeout_s': 10})
                with patch.object(self.h.actor_output, method, side_effect=ValueError('synthetic display failure')):
                    if method == 'present':
                        with self.assertRaisesRegex(ValueError, 'synthetic display failure'):
                            self.h.handle_tool(params)
                    else:
                        self.assertFalse(self.h.handle_tool(params)['success'])
                job = next(job for job in self.store.states('program') if job['job_id'] not in previous_ids)
                polled = self.call('program_status', {'job_id': job['job_id']}, 'poll-' + method)
                self.assertIn('important diagnostic', polled['stdout_text'])
                self.assertNotIn('stdout_delivery', polled)
                self.assertEqual(polled['status'], 'succeeded')
                self.assertEqual(polled['returncode'], 0)
        self.assertEqual(self.robot.executions, 0)

    def test_real_program_start_poll_full_and_context_recovery(self):
        self.h.thread_id = 'program-log-thread'
        self.h.programs = ProgramRunner(self.h.program_services())
        self.addCleanup(self.h.programs.close)
        started = self.call('program_run', {'source': "print('begin|' + 'x' * 9000 + '|end-diagnostic')", 'timeout_s': 10}, 'start')
        self.assertEqual(started['status'], 'succeeded')
        self.assertFalse(started['stdout_truncated'])
        self.assertIn('end-diagnostic', started['stdout_text'])
        args = {'job_id': started['job_id']}
        polled = self.call('program_status', args, 'poll')
        self.assertEqual(polled['stdout_delivery']['status'], 'unchanged')
        self.assertNotIn('stdout_text', polled)
        self.assertNotIn('stdout_excerpt', polled)
        self.assertNotIn('stdout_truncated', polled)
        self.assertEqual(polled['status'], 'succeeded')
        self.assertEqual(polled['returncode'], 0)
        self.assertEqual(self.call('program_status', dict(args, detail='full'), 'full')['stdout_text'], started['stdout_text'])
        for index, reset in enumerate((
                lambda: self.store.set_state('runtime', 'image_context_epoch', 'compacted'),
                lambda: self.store.set_state('runtime', 'image_context_epoch', {'source': 'tool_reply_delivery_uncertain'}),
                lambda: self.store.set_state('runtime', 'codex', {'app_server_instance_id': 'reconnected'}))):
            reset()
            restored = self.call('program_status', args, 'reset-' + str(index))
            self.assertEqual(restored['stdout_text'], started['stdout_text'])
        self.assertEqual(len(self.store.states('program')), 1)
        self.assertEqual(self.robot.executions, 0)

    def test_isolated_program_status_returns_full_without_disturbing_main_baseline(self):
        self.h.thread_id = 'program-iso-thread'
        self.h.programs = ProgramRunner(self.h.program_services())
        self.addCleanup(self.h.programs.close)
        started = self.call('program_run', {'source': "print('iso|' + 'y' * 2000)", 'timeout_s': 10}, 'start-main')
        args = {'job_id': started['job_id']}

        # isolated 子链读取同一 job 必须拿到完整 stdout_text，且不得污染主链基线
        iso_wire = self.h.handle_tool(dict(
            threadId=self.h.thread_id, turnId='turn', callId='poll-isolated',
            tool='program_status', arguments=args, contextScope='isolated',
        ))
        iso = json.loads(iso_wire['contentItems'][0]['text'])
        self.assertEqual(iso['stdout_text'], started['stdout_text'])
        self.assertNotIn('stdout_delivery', iso)

        # 后续主链轮询依然命中 start-main 基线
        polled = self.call('program_status', args, 'poll-main')
        self.assertEqual(polled['stdout_delivery']['status'], 'unchanged')
        self.assertEqual(polled['stdout_delivery']['previous_tool_call_id'], 'start-main')



# --- Folded from tests/test_program_result.py ---

class ProgramResultTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = Store(Path(directory.name), {})
        self.addCleanup(self.store.close)

    def job(self, stdout, stderr=b"", **extra):
        return dict(job_id="program-test", status="failed", returncode=1, stop_confirmed=False,
                    calls=3, timeout_s=120, timeout_semantics="work_wall_time_excluding_own_sol_requests",
                    model_wait_s=10, finished_at="2026-10-05T00:00:00Z",
                    stdout_ref=self.store.put_artifact(stdout, "program_stdout", "text/plain"),
                    stderr_ref=self.store.put_artifact(stderr, "program_stderr", "text/plain"),
                    stdout_text=stdout[:8192].decode(errors="replace"), stdout_truncated=len(stdout)>8192,
                    stderr_text=stderr[:8192].decode(errors="replace"), stderr_truncated=len(stderr)>8192,
                    **extra)

    def test_arbitrary_stdout_stays_text_and_retains_terminal_result(self):
        raw = b"start\n" + b"x"*40000 + b'\n{"status":"completed","grasp":true}'
        errors = b"Traceback\n" + b"line\n"*7000 + b"RuntimeError: action stop unconfirmed"
        job = self.job(raw, errors, source_sha256="source", input_digest="input", correlation=dict(program_id="p"))
        before = deepcopy(job)
        result = program_result(self.store, job)
        self.assertIn('"grasp":true', result["stdout_text"])
        self.assertFalse(result["stdout_truncated"])
        self.assertNotIn("stdout_excerpt", result)
        self.assertNotIn("stderr_excerpt", result)
        self.assertTrue(result["stderr_text"].endswith("action stop unconfirmed"))
        self.assertNotIn("skill_output", result)
        for key in ("status", "returncode", "stop_confirmed", "calls", "model_wait_s", "timeout_semantics"):
            self.assertEqual(result[key], job[key])
        bounded = program_result(self.store, job, max_output_bytes=4096)
        self.assertTrue(bounded["stdout_truncated"])
        self.assertEqual(bounded["stdout_excerpt"]["byte_size"], len(raw))
        self.assertEqual(result["full_result"], dict(tool="program_status", job_id=job["job_id"], detail="full"))
        self.assertEqual(job, before)
        full = program_result(self.store, job, detail="full")
        self.assertEqual(full["stdout_text"].encode(), raw)
        self.assertEqual(full["stderr_text"].encode(), errors)
        self.assertFalse(full["stdout_truncated"])
        self.assertEqual(full["correlation"], job["correlation"])

    def test_default_preserves_middle_diagnostics_and_all_text_in_both_streams(self):
        for size in (10240, 32768, 32769, 100000):
            raw = b"x" * size + "\n中间唯一诊断: stop unconfirmed\n".encode() + b"y" * size
            result = program_result(self.store, self.job(raw, raw))
            for name in ("stdout", "stderr"):
                with self.subTest(size=size, stream=name):
                    self.assertFalse(result[name + "_truncated"])
                    self.assertNotIn(name + "_excerpt", result)
                    self.assertEqual(result[name + "_text"].encode(), raw)

    def test_long_image_data_urls_leave_diagnostics_visible_and_full_exact(self):
        image = Image.frombytes("RGB", (128, 128), random.Random(0).randbytes(128 * 128 * 3))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        url = b"data:image/png;base64," + base64.b64encode(buffer.getvalue())
        self.assertGreater(len(url), 10240)
        before = "失败诊断: status=stopped executed_native_steps=4\n".encode()
        after = "\nRuntimeError: 停止未确认; observation unavailable\n".encode()
        raw = before + url + after
        job = self.job(raw, raw)
        original_job = deepcopy(job)
        result = program_result(self.store, job)
        for name in ("stdout", "stderr"):
            with self.subTest(stream=name):
                text = result[name + "_text"]
                self.assertTrue(text.startswith(before.decode()))
                self.assertTrue(text.endswith(after.decode()))
                self.assertIn("inline image/png data URL omitted", text)
                self.assertNotIn(url[100:200].decode(), text)
                self.assertTrue(result[name + "_truncated"])
                excerpt = result[name + "_excerpt"]
                self.assertEqual(excerpt["presentation"], "inline_image_data_urls_omitted")
                self.assertEqual(excerpt["byte_ranges_source"], "presentation_utf8")
                self.assertEqual(excerpt["byte_size"], len(text.encode()))
                self.assertEqual(excerpt["byte_ranges"], [[0, len(text.encode())]])
                self.assertEqual(excerpt["omitted_bytes"], 0)
                self.assertEqual(excerpt["original_byte_size"], len(raw))
                self.assertEqual(excerpt["removed_media_count"], 1)
                self.assertEqual(excerpt["removed_media_bytes"], len(url))
                self.assertEqual(excerpt["removed_media_byte_ranges"], [[len(before), len(before) + len(url)]])
                self.assertEqual(excerpt["removed_media_byte_ranges_source"], "original_log_bytes")
                self.assertTrue(excerpt["source_complete"])
                self.assertEqual(self.store.resolve_artifact(job[name + "_ref"])[1], raw)
        self.assertEqual(job, original_job)
        full = program_result(self.store, job, detail="full")
        self.assertEqual(full["stdout_text"].encode(), raw)
        self.assertEqual(full["stderr_text"].encode(), raw)
        self.assertFalse(full["stdout_truncated"])

    def test_nonstandard_or_incomplete_media_and_nonimage_urls_stay_opaque(self):
        for payload in (b"AAA", b"AAAA=", b"____", b"AAAA@BBBB", b"AAAA\\/BBBB"):
            raw = b'before "data:image/png;base64,' + payload + b'" after: status=failed'
            with self.subTest(payload=payload):
                result = program_result(self.store, self.job(raw))
                self.assertEqual(result["stdout_text"].encode(), raw)
                self.assertNotIn("stdout_excerpt", result)
                self.assertFalse(result["stdout_truncated"])
        raw = b'data:application/octet-stream;base64,AAAA'
        result = program_result(self.store, self.job(raw))
        self.assertEqual(result["stdout_text"].encode(), raw)
        raw = b'metadata:image/png;base64,AAAA'
        result = program_result(self.store, self.job(raw))
        self.assertEqual(result["stdout_text"].encode(), raw)
        incomplete = dict(stdout_text="data:image/png;base64,AAAA", stdout_truncated=True)
        result = program_result(self.store, incomplete)
        self.assertEqual(result["stdout_text"], incomplete["stdout_text"])
        self.assertNotIn("presentation", result["stdout_excerpt"])
        self.assertFalse(result["stdout_excerpt"]["source_complete"])

    def test_multibyte_plain_output_keeps_original_byte_ranges(self):
        raw = ("汉" * 12000 + "\n错误: stop unconfirmed").encode()
        result = program_result(self.store, self.job(raw))
        self.assertFalse(result["stdout_truncated"])
        self.assertNotIn("stdout_excerpt", result)
        self.assertEqual(result["stdout_text"].encode(), raw)
        self.assertTrue(result["stdout_text"].endswith("错误: stop unconfirmed"))
        bounded = program_result(self.store, self.job(raw), max_output_bytes=4096)
        excerpt = bounded["stdout_excerpt"]
        self.assertEqual(excerpt["byte_size"], len(raw))
        self.assertEqual(excerpt["byte_ranges"], [[0, 2048], [len(raw) - 2048, len(raw)]])
        self.assertNotIn("byte_ranges_source", excerpt)
        self.assertEqual(program_result(self.store, self.job(raw), detail="full")["stdout_text"].encode(), raw)

    def test_media_ranges_stay_auditable_with_explicit_internal_excerpt_budget(self):
        prefix = "前段诊断: ".encode() + b"a" * 20000 + b"\n"
        url = b"data:image/png;base64," + base64.b64encode(b"x" * 30000)
        suffix = b"\n" + b"b" * 20000 + "\n末段诊断: stop unconfirmed".encode()
        raw = prefix + url + suffix
        result = program_result(self.store, self.job(raw), max_output_bytes=32768)
        excerpt = result["stdout_excerpt"]
        size = excerpt["byte_size"]
        self.assertGreater(size, 32768)
        self.assertEqual(excerpt["byte_ranges"], [[0, 16384], [size - 16384, size]])
        self.assertEqual(excerpt["omitted_bytes"], size - 32768)
        self.assertEqual(excerpt["byte_ranges_source"], "presentation_utf8")
        self.assertEqual(excerpt["removed_media_byte_ranges"], [[len(prefix), len(prefix) + len(url)]])
        self.assertEqual(excerpt["original_byte_size"], len(raw))
        self.assertTrue(result["stdout_text"].startswith("前段诊断: "))
        self.assertTrue(result["stdout_text"].endswith("末段诊断: stop unconfirmed"))
        self.assertEqual(program_result(self.store, self.job(raw), detail="full")["stdout_text"].encode(), raw)

    def test_plain_json_arrays_and_unknown_diagnostics_are_not_filtered(self):
        value = dict(status="unknown", failure_feedback=dict(witness=[.123456789] * 400),
            new_diagnostic=dict(reason="execution_uncertain", state=[0, 1, 2]),
            contentItems=[dict(type="inputImage", imageUrl="ordinary opaque value")])
        raw = json.dumps(value).encode()
        self.assertLess(len(raw), 10240)
        result = program_result(self.store, self.job(raw))
        self.assertEqual(result["stdout_text"].encode(), raw)
        self.assertNotIn("skill_output", result)
        self.assertNotIn("stdout_excerpt", result)

    def test_output_scope_and_kind_remain_checked(self):
        for kind, scope, error in (("program_stdout", "private", "artifact_access_scope_denied"),
                                   ("other", "actor", "program_output_artifact_kind_mismatch")):
            ref = self.store.put_artifact(b"opaque output", kind, "text/plain", scope)
            with self.subTest(kind=kind, scope=scope), self.assertRaisesRegex(ValueError, error):
                program_result(self.store, dict(stdout_ref=ref))

    def test_known_nested_evidence_omits_metadata_in_summary_and_full(self):
        receipt = dict(schema_version='0.1', run_id='fixture', env_generation=1, status='unknown',
            stop_confirmed=False, executed_native_steps=None, plan_revision=5,
            failure_feedback=dict(schema_version='diagnostic-v1', reason='stop unconfirmed',
                                  error_m=0.123456789012345),
            stage_results={'version': dict(status='partial', revision=3)},
            diagnostic=dict(version='business-version', revision=4))
        expected = {key: value for key, value in receipt.items()
                    if key not in {'schema_version', 'run_id', 'env_generation'}}
        expected['failure_feedback'] = {key: value for key, value in receipt['failure_feedback'].items()
                                        if key != 'schema_version'}
        job = self.job(b'complete middle diagnostics',
            action_summary=dict(recent_actions=[receipt] * 10),
            pending_review=dict(review_id='review', preview=receipt))
        original = deepcopy(job)
        for detail in ('summary', 'full'):
            shown = program_result(self.store, job, detail=detail)
            self.assertEqual(shown['action_summary']['recent_actions'], [expected] * 10)
            self.assertEqual(shown['pending_review']['preview'], expected)
            self.assertEqual(shown['stdout_text'], 'complete middle diagnostics')
        self.assertEqual(job, original)



# --- Folded from tests/test_program_services.py ---

class ProgramServicesTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.store = Store(root / 'run', dict(run_id='services', episode_id='ep',
            env_instance_id='env', env_generation=1))
        self.addCleanup(self.store.close)
        self.harness = Harness(self.store, FixtureRobot(root))
        self.harness.start()
        self.services = self.harness.program_services()
        self.runner = ProgramRunner(self.services)
        self.addCleanup(self.runner.close)

    def test_explicit_port_rejects_runtime_owner_and_exposes_no_robot_or_registry(self):
        with self.assertRaisesRegex(TypeError, 'requires ProgramServices'):
            ProgramRunner(self.harness)
        self.assertIsInstance(self.services, ProgramServices)
        self.assertIs(self.runner.store, self.store)
        for name in ('harness', 'robot', 'registry', 'tool_registry', 'active_context', 'cancel'):
            self.assertFalse(hasattr(self.services, name), name)
        self.assertFalse(hasattr(self.runner, 'harness'))
        with self.assertRaises(FrozenInstanceError):
            self.services.invoke = None

    def test_dynamic_tool_registration_role_effect_and_disable_are_server_bound(self):
        calls = []
        def invoke(context, arguments, correlation):
            calls.append(arguments)
            return ToolOutput(arguments).response()
        definition = dict(type='function', name='late_program_tool', description='Port regression.',
            inputSchema=dict(type='object', properties={'execute': {'type': 'boolean'}},
                required=['execute'], additionalProperties=False))
        self.harness.tool_registry.register(ToolSpec(definition, invoke,
            effect=lambda arguments: 'physical' if arguments['execute'] else 'read',
            output_schemas={'data': definition['inputSchema']}))
        self.runner._validate_tool('late_program_tool')
        self.runner._validate_arguments('late_program_tool', {'execute': False})
        self.assertEqual(self.runner._tool_effect('late_program_tool', {'execute': True}), 'physical')
        self.assertEqual(self.runner._invoke('late_program_tool', {'execute': False}, {}).data, {'execute': False})
        with self.assertRaises(ValueError):
            self.runner._invoke('late_program_tool', {'execute': False, 'unknown': 1}, {})
        self.harness.tool_registry.register(ToolSpec(dict(definition, name='actor_only'), invoke,
            roles=frozenset({'actor'}), output_schemas={'data': definition['inputSchema']}))
        with self.assertRaisesRegex(ValueError, 'program_tool_not_permitted'):
            self.runner._validate_tool('actor_only')
        self.harness.disabled_tools = frozenset({'late_program_tool'})
        with self.assertRaisesRegex(ValueError, 'tool_disabled_by_config'):
            self.runner._invoke('late_program_tool', {'execute': False}, {})
        self.assertEqual(calls, [{'execute': False}])

    def test_observation_snapshot_and_cancellation_never_lend_mutable_state(self):
        original = self.harness.latest['observation_id']
        snapshot = self.services.observation()
        snapshot['observation_id'] = 'changed-by-caller'
        self.assertEqual(self.harness.latest['observation_id'], original)
        self.harness.latest = None
        self.assertIsNone(self.services.observation()['observation_id'])
        self.harness.active_context = {'program_id': 'other-program'}
        self.assertFalse(self.services.cancel_action('my-program'))
        self.assertFalse(self.harness.cancel.is_set())
        self.assertTrue(self.services.action_active('other-program'))
        self.assertTrue(self.services.cancel_action('other-program'))
        self.assertTrue(self.harness.cancel.is_set())
        self.assertTrue(self.harness.accepting)



# --- Folded from tests/test_programs.py ---

class ProgramCheck(unittest.TestCase):
    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_opencv_uses_one_thread_inside_bounded_program(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        temp = temp_dir.name
        if True:
            store = Store(Path(temp) / 'run', dict(run_id='test', episode_id='ep', env_instance_id='env', env_generation=1))
            self.addCleanup(store.close)
            runner = ProgramRunner(program_services(SimpleNamespace(store=store)))
            self.addCleanup(runner.close)
            source = '''import cv2, numpy as np, resource
assert cv2.getNumThreads() == 1, cv2.getNumThreads()
assert resource.getrlimit(resource.RLIMIT_AS)[0] == 1024**3
image = np.zeros((480, 640, 3), dtype=np.uint8)
image[240:280, 300:350] = (0, 0, 255)
ok, encoded = cv2.imencode('.png', image)
assert ok
decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
count, labels, stats, centers = cv2.connectedComponentsWithStats((gray > 0).astype(np.uint8))
assert count == 2 and stats[1, cv2.CC_STAT_AREA] == 2000
scores = cv2.matchTemplate(gray, gray[230:290, 290:360], cv2.TM_SQDIFF)
assert cv2.minMaxLoc(scores)[2] == (290, 230)
assert np.allclose(np.linalg.svd(np.eye(3), compute_uv=False), 1)
print('opencv_and_linalg_completed')
'''
            result = runner.run(dict(source=source, timeout_s=30),
                                dict(thread_id='thread', turn_id='turn', tool_call_id='opencv'))
            self.assertEqual(result['status'], 'succeeded', result.get('stderr_text'))
            self.assertIn('opencv_and_linalg_completed', result['stdout_text'])
            self.assertNotIn("Can't spawn new thread", result.get('stderr_text', ''))

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_real_client_cannot_nest_manipulation_or_other_program_tools(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        temp = temp_dir.name
        if True:
            store = Store(Path(temp) / 'run', dict(run_id='test', episode_id='ep', env_instance_id='env', env_generation=1))
            self.addCleanup(store.close)
            calls = []
            harness = SimpleNamespace(store=store, lock=threading.RLock(), cancel=threading.Event(), active_context=None)
            def dispatch(name, args, correlation):
                calls.append(name)
                return dict(success=True, contentItems=[])
            harness.dispatch = dispatch
            runner = ProgramRunner(program_services(harness))
            self.addCleanup(runner.close)
            source = '''import json
from ni_rua_client import NiRuaClient
client = NiRuaClient()
denied = []
for tool in ("program_run", "program_status", "skill_run", "manipulation_run"):
    try:
        client.call(tool, {"parameters": {"kind": "help"}})
    except RuntimeError as error:
        assert str(error) == "program_tool_not_permitted", str(error)
        denied.append(tool)
    else:
        raise AssertionError("nested program tool was admitted: " + tool)
client.call("robot_capabilities")
print(json.dumps(denied))
'''
            result = runner.run(dict(source=source, timeout_s=5, max_calls=1),
                                dict(thread_id='thread', turn_id='turn', tool_call_id='nested'))
            self.assertEqual(result['status'], 'succeeded', result.get('stderr_text'))
            self.assertEqual(json.loads(result['stdout_text']),
                             ['program_run', 'program_status', 'skill_run', 'manipulation_run'])
            self.assertEqual(calls, ['robot_capabilities'])
            self.assertEqual(result['calls'], 1)

    def test_only_own_sol_wait_pauses_work_clock_and_restores_time_after_failure(self):
        clock = [100.0]
        entry = dict(cancel=threading.Event(), deadline=105.0)
        unrelated = dict(cancel=threading.Event(), deadline=105.0)
        with patch("ni_rua.runtime.programs.runner.time.monotonic", side_effect=lambda: clock[0]):
            with model_wait(entry):
                clock[0] = 1000.0
                self.assertEqual(program_remaining(entry), 5.0)
                self.assertEqual(program_remaining(unrelated), -895.0)
            self.assertEqual(program_remaining(entry), 5.0)
            self.assertEqual(entry["model_wait_s"], 900.0)
            self.assertIsNone(entry["model_wait_started_at"])
            clock[0] += 2
            self.assertEqual(program_remaining(entry), 3.0)
            with self.assertRaisesRegex(RuntimeError, "fixture"):
                with model_wait(entry):
                    clock[0] += 100
                    raise RuntimeError("fixture")
            self.assertEqual(program_remaining(entry), 3.0)
            self.assertEqual(entry["model_wait_s"], 1000.0)
            clock[0] += 4
            with self.assertRaisesRegex(ValueError, "program_channel_expired"):
                with model_wait(entry):
                    self.fail("Expired work budget admitted a model call")

    def test_model_wait_never_ignores_explicit_cancellation(self):
        entry = dict(cancel=threading.Event(), deadline=time.monotonic()+10)
        with model_wait(entry):
            entry["cancel"].set()
            self.assertTrue(entry["cancel"].is_set())
        with self.assertRaisesRegex(ValueError, "program_channel_expired"):
            with model_wait(entry):
                self.fail("Cancelled program admitted another call")

    def test_artifact_broker_enforces_profile_on_json_and_geometry_kinds(self):
        from ni_rua.core.observation_policy import ACTOR_PROFILE, LEGACY_ACTOR_PROFILE
        temp = self.enterContext(tempfile.TemporaryDirectory())
        store = Store(Path(temp)/'run', dict(run_id='test', episode_id='ep', env_instance_id='env', env_generation=1))
        self.addCleanup(store.close)
        refs = {
            'calibration': store.put_artifact(b'{"K":[[1,0,0],[0,1,0],[0,0,1]]}', 'camera_calibration', 'application/json'),
            'untyped_calibration': store.put_artifact(b'{"nested":{"K":[[1,0,0],[0,1,0],[0,0,1]]}}', 'computed', 'application/json'),
            'calibrated_overlay': store.put_artifact(b'calibrated projected pixels', 'geometry_overlay', 'image/jpeg'),
            'derived_text': store.put_artifact(b'actor-derived geometry fixture', 'scene_geometry', 'text/plain', 'internal'),
            'nested': store.put_artifact(b'{"nested":{"object_poses":{}}}', 'computed', 'application/json'),
            'typed_geometry': store.put_artifact(b'{"points_m":[[0,0,1]]}', 'scene_job_result', 'application/json', 'internal'),
            'rgb': store.put_artifact(b'original RGB bytes', 'rgb', 'image/jpeg'),
        }
        entry = dict(cancel=threading.Event(), deadline=time.monotonic()+30, call_lock=threading.RLock())
        for profile in (LEGACY_ACTOR_PROFILE, ACTOR_PROFILE):
            runner = ProgramRunner(program_services(SimpleNamespace(store=store, observation_profile=profile)))
            try:
                for key in ('calibration', 'calibrated_overlay'):
                    if profile == LEGACY_ACTOR_PROFILE:
                        with self.assertRaisesRegex(ValueError, 'calibrated_artifact'):
                            runner.dispatch(entry, {'artifact_ref': refs[key]})
                    else:
                        self.assertIn('data_base64', runner.dispatch(entry, {'artifact_ref': refs[key]}))
                if profile == LEGACY_ACTOR_PROFILE:
                    with self.assertRaisesRegex(ValueError, 'privileged_data'):
                        runner.dispatch(entry, {'artifact_ref': refs['untyped_calibration']})
                else:
                    self.assertIn('data_base64', runner.dispatch(entry, {'artifact_ref': refs['untyped_calibration']}))
                for key in ('typed_geometry', 'derived_text'):
                    self.assertIn('data_base64', runner.dispatch(entry, {'artifact_ref': refs[key]}))
                with self.assertRaisesRegex(ValueError, 'privileged_data'):
                    runner.dispatch(entry, {'artifact_ref': refs['nested']})
                self.assertIn('data_base64', runner.dispatch(entry, {'artifact_ref': refs['rgb']}))
            finally:
                runner.close()

    def test_artifact_owner_metadata_projection_preserves_execution_and_originals(self):
        from ni_rua.core.contracts import canonical
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        temp = temp_dir.name
        if True:
            identity = dict(run_id='test', episode_id='ep', env_instance_id='env', env_generation=1)
            store = Store(Path(temp) / 'run', identity)
            self.addCleanup(store.close)
            runner = ProgramRunner(program_services(SimpleNamespace(store=store)))
            self.addCleanup(runner.close)
            entry = dict(cancel=threading.Event(), deadline=time.monotonic()+30, call_lock=threading.RLock())
            business = dict(version='operator-selected', revision=7, run_id='business-value')
            stage = dict(schema_version='0.1', **identity, path=dict(position_m=[1.23456789012345]),
                         diagnostics=business)
            request = dict(schema_version='0.1', **identity, action_id='test-action', type='BatchAction',
                           stages=[dict(id='reach', kind='move', action=stage)])
            receipt = dict(schema_version='0.1', **identity, action_id='test-action', status='unknown',
                           execution_uncertain=True, stop_confirmed=False, executed_native_steps=None,
                           batch_revision=9, plan_revision=4,
                           stage_results={'reach': dict(status='partial', position_error_m=0.123456789012345)},
                           diagnostics=business)
            store.save('action_request', 'test-action', request)
            store.save('action_receipt', 'test-action', receipt)
            cases = [
                ('unified_action_result', dict(request=request, receipt=receipt)),
                ('program_action_evidence', [receipt]),
                ('manipulation_receipt', receipt),
                ('candidate_trajectory', dict(schema_version='0.1', identity=identity,
                    plan_revision=4, samples=[business])),
                ('computed', dict(schema_version='domain-format', version='domain-version',
                    request=request, receipt=receipt)),
            ]
            for kind, value in cases:
                with self.subTest(kind=kind):
                    raw = canonical(value).encode()
                    ref = store.put_artifact(raw, kind, 'application/json')
                    before = store.resolve_artifact(ref)
                    reply = runner.dispatch(entry, dict(artifact_ref=ref))
                    shown = json.loads(base64.b64decode(reply['data_base64']))
                    self.assertEqual(store.resolve_artifact(ref), before)
                    if kind == 'computed':
                        self.assertEqual(shown, value)
                    elif kind == 'candidate_trajectory':
                        self.assertEqual(shown, dict(plan_revision=4, samples=[business]))
                    else:
                        record = shown['receipt'] if kind == 'unified_action_result' else (
                            shown[0] if isinstance(shown, list) else shown)
                        for key in ('schema_version', *identity):
                            self.assertNotIn(key, record)
                        self.assertEqual(record['stage_results'], receipt['stage_results'])
                        self.assertEqual(record['diagnostics'], business)
                        self.assertEqual(record['status'], 'unknown')
                        self.assertIsNone(record['executed_native_steps'])
                        self.assertFalse(record['stop_confirmed'])
                        self.assertTrue(record['execution_uncertain'])
                        self.assertEqual((record['batch_revision'], record['plan_revision']), (9, 4))
                        if kind == 'unified_action_result':
                            for owner in (shown['request'], shown['request']['stages'][0]['action']):
                                self.assertFalse(set(owner) & {'schema_version', *identity})
                            self.assertEqual(shown['request']['stages'][0]['action']['diagnostics'], business)

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_real_client_budgets_isolation_and_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            store=Store(Path(temp)/'run',dict(run_id='test',episode_id='ep',env_instance_id='env',env_generation=1))
            calls=[]
            harness=SimpleNamespace(store=store,lock=threading.RLock(),cancel=threading.Event(),active_context=None)
            def dispatch(name,args,correlation):
                calls.append((name,correlation))
                return {'success':True,'contentItems':[{'type':'inputText','text':json.dumps({'test':True})}]}
            harness.dispatch=dispatch
            runner=ProgramRunner(program_services(harness))
            identity=dict(thread_id='thread',turn_id='turn',tool_call_id='call',item_id='item')
            try:
                secret=Path(temp)/'observer-secret'; secret.write_text('hidden')
                source="from ni_rua_client import NiRuaClient\nclient=NiRuaClient()\nprint(client.call('robot_capabilities'))\ntry:\n open("+repr(str(secret))+").read()\n raise AssertionError('observer leaked')\nexcept PermissionError:\n print('isolation_passed')\n"
                args=dict(source=source,timeout_s=5,max_calls=1)
                result=runner.run(args,identity)
                self.assertEqual(result['status'],'succeeded',store.resolve_artifact(result['stderr_ref'])[1])
                self.assertEqual(result['source_sha256'],__import__('hashlib').sha256(source.encode()).hexdigest())
                self.assertIn(b'isolation_passed',store.resolve_artifact(result['stdout_ref'])[1])
                self.assertEqual(calls[0][1]['item_id'],'item')
                self.assertEqual(runner.run(args,identity),result)
                self.assertEqual(len(calls),1)
                numerical=runner.run(dict(source="import numpy as np\nfrom pathlib import Path\nimport os\nassert Path.cwd() == Path(os.environ['HOME'])\nPath('relative-result.txt').write_text('job-local')\nassert float(np.linalg.norm([3.,4.])) == 5.\nprint('numpy_within_memory_budget')",timeout_s=10),dict(identity,tool_call_id='numerical'))
                self.assertEqual(numerical['status'],'succeeded',store.resolve_artifact(numerical['stderr_ref'])[1])
                self.assertIn(b'numpy_within_memory_budget',store.resolve_artifact(numerical['stdout_ref'])[1])
                guarded=runner.run(dict(source="import socket,os\ntry:\n socket.socket(socket.AF_UNIX)\n raise AssertionError('Unix socket allowed')\nexcept PermissionError: pass\ntry:\n os.fork()\n raise AssertionError('fork allowed')\nexcept PermissionError: pass\ntry:\n os.kill(os.getppid(),0)\n raise AssertionError('cross process signal allowed')\nexcept PermissionError: pass\nprint('network_process_guards_passed')"),dict(identity,tool_call_id='guarded'))
                self.assertEqual(guarded['status'],'succeeded',store.resolve_artifact(guarded['stderr_ref'])[1])
                reused=runner.run(dict(source_ref=result['source_ref'],timeout_s=5,max_calls=1),dict(identity,tool_call_id='reused'))
                self.assertEqual(reused['status'],'succeeded')
                limited=runner.run(dict(source="from ni_rua_client import NiRuaClient\nc=NiRuaClient()\nc.call('robot_capabilities')\nc.call('robot_capabilities')",max_calls=1),dict(identity,tool_call_id='limited'))
                self.assertEqual(limited['status'],'succeeded')
                self.assertEqual(limited['calls'],2)
                timed=runner.run(dict(source='import time\ntime.sleep(20)',timeout_s=1),dict(identity,tool_call_id='timed'))
                self.assertEqual(timed['status'],'cancelled')
                self.assertTrue(timed['stop_confirmed'])
                denied=runner.run(dict(source="from ni_rua_client import NiRuaClient\nNiRuaClient().call('observer_evaluate')"),dict(identity,tool_call_id='denied'))
                self.assertEqual(denied['status'],'failed')
            finally:
                runner.close();store.close()

    def test_reconcile_and_episode_isolated_unresolved_jobs(self):
        from dataclasses import replace

        temp = self.enterContext(tempfile.TemporaryDirectory())
        identity = dict(run_id='run-rec', episode_id='ep-1', env_instance_id='env-1', env_generation=1)
        store = Store(Path(temp) / 'run', identity)
        self.addCleanup(store.close)
        active_flag = {'active': True}
        base_services = program_services(SimpleNamespace(store=store))
        services = replace(
            base_services,
            sandbox_enabled=False,
            action_active=lambda jid: active_flag['active'] and jid == 'job-unknown-1',
            current_identity=lambda: dict(identity),
        )
        runner = ProgramRunner(services)
        self.addCleanup(runner.close)

        # Seed two unknown jobs: one in ep-1 (with live handle), one in ep-old (prior episode)
        fake_proc = SimpleNamespace(poll=lambda: None)
        fake_broker = SimpleNamespace(is_alive=lambda: True)
        store.set_state('program', 'job-unknown-1', {
            'job_id': 'job-unknown-1',
            'identity': dict(identity),
            'status': 'unknown',
            'stop_confirmed': False,
            'reason': 'program_broker_operation_stop_unconfirmed',
        })
        runner._unresolved_handles['job-unknown-1'] = {
            'pgid': None,
            'process': fake_proc,
            'broker': fake_broker,
            'no_process_spawned': False,
        }
        store.set_state('program', 'job-old-ep', {
            'job_id': 'job-old-ep',
            'identity': dict(run_id='run-rec', episode_id='ep-0', env_instance_id='env-1', env_generation=0),
            'status': 'unknown',
            'stop_confirmed': False,
            'reason': 'gateway_restarted_without_program_acknowledgement',
        })

        # Episode filtering & acknowledgement filtering
        self.assertEqual(runner.unresolved_jobs(identity=identity), ['job-unknown-1'])
        self.assertEqual(runner.unresolved_jobs(identity=identity, acknowledged=['program:job-unknown-1']), [])
        self.assertEqual(set(runner.unresolved_jobs()), {'job-unknown-1', 'job-old-ep'})

        # While child/broker/action still active, reconcile keeps it in still_unresolved
        first_rec = runner.reconcile('job-unknown-1')
        self.assertEqual(first_rec, {'reconciled': [], 'still_unresolved': ['job-unknown-1']})

        # Once process exited, broker exited, and active action cleared -> reconcile confirms late stop while keeping status=='unknown'
        fake_proc.poll = lambda: 0
        fake_broker.is_alive = lambda: False
        active_flag['active'] = False
        second_rec = runner.reconcile('job-unknown-1')
        self.assertEqual(second_rec, {'reconciled': ['job-unknown-1'], 'still_unresolved': []})
        saved = store.read_state('program', 'job-unknown-1')
        self.assertEqual(saved['status'], 'unknown')
        self.assertTrue(saved['late_stop_confirmed'])
        self.assertIsInstance(saved['late_stop_confirmed_at_ns'], int)
        self.assertEqual(runner.unresolved_jobs(identity=identity), [])
        events = [e for e in store.events() if e['type'] == 'program.stop_confirmed_late']
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['payload']['job_id'], 'job-unknown-1')

    def test_broker_typed_incomplete_action_gate_and_on_incomplete_action_modes(self):
        from dataclasses import replace
        from ni_rua.core.observation_policy import ACTOR_PROFILE
        from ni_rua.core.tool_output import ToolResponse

        temp = self.enterContext(tempfile.TemporaryDirectory())
        identity = dict(run_id='run-gate', episode_id='ep-1', env_instance_id='env-1', env_generation=1)
        store = Store(Path(temp) / 'run', identity)
        self.addCleanup(store.close)

        dispatched_calls = []

        def fake_dispatch(name, args, correlation):
            # Strict check: allow_after_failure MUST NEVER leak into tool arguments!
            self.assertNotIn('allow_after_failure', args)
            dispatched_calls.append((name, dict(args)))
            if name == 'robot_observe':
                host.latest = {'observation_id': 'obs-live-2', **identity}
                resp = ToolResponse(
                    [dict(type='inputText', text='{"observation_id":"obs-live-2"}')],
                    data=dict(observation_id='obs-live-2', robot_state_ref='artifact:s', cameras=[], **identity),
                    kind='observation',
                )
                resp.captured_live = True
                return resp
            if name == 'robot_action' and args.get('mode', 'preview') == 'preview':
                return ToolResponse(
                    [dict(type='inputText', text='{"status":"planned","plan_id":"p-ro"}')],
                    data=dict(status='planned', plan_id='p-ro'),
                    kind='plan',
                )
            idx = sum(1 for t, a in dispatched_calls if t == 'robot_action' and a.get('mode') == 'direct')
            if idx in (1, 3):
                # Deliberately non-English checkpoint text: broker gate uses ONLY typed status/physics_advanced!
                rcpt = dict(
                    action_id=f'act-inc-{idx}',
                    status='stopped' if idx == 1 else 'failed',
                    reason='observation_checkpoint' if idx == 1 else 'not_settled',
                    checkpoint_reason='jaw_not_settled' if idx == 1 else None,
                    checkpoint_reason_code='jaw_not_settled' if idx == 1 else None,
                    failed_stage_id=None if idx == 1 else 'insert',
                    pending_stage_ids=['lift'],
                    physics_advanced=True,
                )
                return ToolResponse(
                    [dict(type='inputText', text='{}')],
                    data=dict(request=dict(action_id=rcpt['action_id']), receipt=rcpt),
                    kind='action',
                    data_projection='action_result',
                )
            rcpt = dict(action_id=f'act-ok-{idx}', status='completed', reason=None, physics_advanced=True)
            return ToolResponse(
                [dict(type='inputText', text='{}')],
                data=dict(request=dict(action_id=rcpt['action_id']), receipt=rcpt),
                kind='action',
                data_projection='action_result',
            )

        host = SimpleNamespace(
            store=store,
            lock=threading.RLock(),
            cancel=threading.Event(),
            active_context=None,
            latest={'observation_id': 'obs-live-1'},
            observation_profile=ACTOR_PROFILE,
            dispatch=fake_dispatch,
        )
        services = replace(program_services(host), sandbox_enabled=False, current_identity=lambda: dict(identity))
        runner = ProgramRunner(services)
        self.addCleanup(runner.close)

        stage = [{'id': 's1', 'kind': 'gripper', 'arm': 'left', 'goal_normalized': 0.0}]
        script = (
            "import json\n"
            "from ni_rua_client import NiRuaClient\n"
            "c = NiRuaClient()\n"
            f"stage = {stage!r}\n"
            "# 1) Incomplete stopped action latches broker gate\n"
            "r1 = c.action(stage, observation_id='obs-live-1')\n"
            "# Read-only preview is STILL permitted while latched\n"
            "prev = c.call('robot_action', {'mode': 'preview', 'observation_id': 'obs-live-1', 'stages': stage})\n"
            "assert prev['kind'] == 'plan'\n"
            "# Blind physical continuation is blocked by broker with typed error payload\n"
            "blocked_code = None\n"
            "blocked_status = None\n"
            "try:\n"
            "    c.action(stage, observation_id='obs-live-1')\n"
            "except RuntimeError as exc:\n"
            "    blocked_code = getattr(exc, 'code', None)\n"
            "    blocked_status = (getattr(exc, 'incomplete_action', None) or {}).get('status')\n"
            "# Live robot_observe clears latch\n"
            "c.call('robot_observe', {})\n"
            "r2 = c.action(stage, observation_id='obs-live-2')\n"
            "assert r2['data']['receipt']['status'] == 'completed'\n"
            "# 2) Failed action latches again; acknowledge_failure() survives a read-only preview and permits next physical call\n"
            "r3 = c.action(stage, observation_id='obs-live-2')\n"
            "assert r3['data']['receipt']['status'] == 'failed'\n"
            "c.acknowledge_failure()\n"
            "c.call('robot_action', {'mode': 'preview', 'observation_id': 'obs-live-2', 'stages': stage})\n"
            "r4 = c.action(stage, observation_id='obs-live-2')\n"
            "assert r4['data']['receipt']['status'] == 'completed'\n"
            "print(json.dumps({'blocked_code': blocked_code, 'blocked_status': blocked_status}))\n"
        )
        job_block = runner.run(
            dict(source=script, allow_direct_actions=True, timeout_s=20),
            dict(thread_id='t', turn_id='u', tool_call_id='gate-block'),
        )
        self.assertEqual(job_block['status'], 'succeeded', job_block.get('stderr_text'))
        self.assertEqual(job_block['on_incomplete_action'], 'block')
        out = json.loads(job_block['stdout_text'].strip())
        self.assertEqual(out, {'blocked_code': 'program_blind_continuation_blocked', 'blocked_status': 'stopped'})

        # 3) on_incomplete_action='allow' permits direct continuation without observe or allow_after_failure
        dispatched_calls.clear()
        allow_script = (
            "from ni_rua_client import NiRuaClient\n"
            "c = NiRuaClient()\n"
            f"stage = {stage!r}\n"
            "r1 = c.action(stage, observation_id='obs-live-1')\n"
            "assert r1['data']['receipt']['status'] == 'stopped'\n"
            "r2 = c.action(stage, observation_id='obs-live-1')\n"
            "assert r2['data']['receipt']['status'] == 'completed'\n"
        )
        job_allow = runner.run(
            dict(source=allow_script, allow_direct_actions=True, timeout_s=20),
            dict(thread_id='t', turn_id='u', tool_call_id='gate-allow'),
            on_incomplete_action='allow',
        )
        self.assertEqual(job_allow['status'], 'succeeded', job_allow.get('stderr_text'))
        self.assertEqual(job_allow['on_incomplete_action'], 'allow')


if __name__ == "__main__":
    unittest.main()
