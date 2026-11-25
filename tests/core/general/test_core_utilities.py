"""Host-capability gating and scheduling advisory contract checks."""
import json
import os
import unittest
from unittest import mock
from unittest.mock import patch

from support import host_capabilities as caps
from support.motion_fixtures import setup_dual_arm_harness

NI_RUA_TEST_TIER = "fast"


def _capability(available, detail="fixture detail"):
    return lambda: caps.Capability("fixture_capability", available, detail)


def _run(case_class):
    result = unittest.TestResult()
    unittest.defaultTestLoader.loadTestsFromTestCase(case_class).run(result)
    return result


class RequireTests(unittest.TestCase):
    def test_available_capability_leaves_the_test_unchanged(self):
        def test_body(self):
            pass
        self.assertIs(caps.require(_capability(True), "fixture reason")(test_body), test_body)

    def test_missing_capability_skips_with_reason_and_probe_detail(self):
        with mock.patch.dict(os.environ, {caps.REQUIRE_ENV: ""}):
            class Gated(unittest.TestCase):
                @caps.require(_capability(False, "no fixture kernel"), "fixture reason")
                def test_gated(self):
                    raise AssertionError("gated body must not run")

                def test_ungated(self):
                    pass
        result = _run(Gated)
        self.assertEqual(result.testsRun, 2)
        self.assertEqual((result.failures, result.errors), ([], []))
        self.assertEqual(len(result.skipped), 1)
        self.assertEqual(result.skipped[0][1], "requires host capability fixture_capability: fixture reason "
                                               "[probe: no fixture kernel]")

    def test_strict_mode_fails_a_gated_method_instead_of_skipping(self):
        with mock.patch.dict(os.environ, {caps.REQUIRE_ENV: "1"}):
            class Gated(unittest.TestCase):
                @caps.require(_capability(False), "fixture reason")
                def test_gated(self):
                    pass
        result = _run(Gated)
        self.assertEqual((len(result.failures), result.errors, result.skipped), (1, [], []))
        self.assertIn("MissingHostCapability", result.failures[0][1])
        self.assertIn(caps.REQUIRE_ENV + "=1 forbids skipping", result.failures[0][1])

    def test_strict_mode_fails_a_gated_class_in_set_up_class(self):
        with mock.patch.dict(os.environ, {caps.REQUIRE_ENV: "1"}):
            @caps.require(_capability(False), "fixture reason")
            class Gated(unittest.TestCase):
                def test_one(self):
                    pass

                def test_two(self):
                    pass
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(Gated)
        result = unittest.TestResult()
        unittest.TestSuite([suite]).run(result)
        self.assertEqual(result.testsRun, 0)
        self.assertEqual(len(result.errors), 1)
        self.assertIn("setUpClass", str(result.errors[0][0]))
        self.assertIn("MissingHostCapability", result.errors[0][1])

    def test_expect_selects_the_branch_and_strict_mode_forbids_the_fallback(self):
        with mock.patch.dict(os.environ, {caps.REQUIRE_ENV: ""}):
            self.assertIs(caps.expect(_capability(True), "fixture reason"), True)
            self.assertIs(caps.expect(_capability(False, "no fixture kernel"), "fixture reason"), False)
        with mock.patch.dict(os.environ, {caps.REQUIRE_ENV: "1"}):
            self.assertIs(caps.expect(_capability(True), "fixture reason"), True)
            with self.assertRaisesRegex(caps.MissingHostCapability, "fixture_capability: fixture reason "
                                        r"\[probe: no fixture kernel\].*forbids the fallback"):
                caps.expect(_capability(False, "no fixture kernel"), "fixture reason")

    def test_strict_flag_values(self):
        for value, expected in (("1", True), ("true", True), ("on", True), ("0", False), ("", False)):
            with self.subTest(value=value), mock.patch.dict(os.environ, {caps.REQUIRE_ENV: value}):
                self.assertIs(caps.required(), expected)


class ProbeTests(unittest.TestCase):
    def test_crashing_probe_reports_missing_instead_of_raising(self):
        @caps._probe
        def broken_probe():
            raise OSError("fixture failure")
        result = broken_probe()
        self.assertFalse(result)
        self.assertEqual(result.name, "broken_probe")
        self.assertIn("fixture failure", result.detail)
        self.assertIs(broken_probe(), result)  # cached

    def test_every_probe_returns_a_described_result_and_composites_are_consistent(self):
        for probe in caps.PROBES:
            with self.subTest(probe=probe.__name__):
                result = probe()
                self.assertIsInstance(result, caps.Capability)
                self.assertEqual(result.name, probe.__name__)
                self.assertTrue(result.detail)
                self.assertTrue(probe.__doc__)
        for composite in (caps.program_isolation, caps.native_replay_sandbox):
            if composite():
                self.assertTrue(caps.landlock() and caps.seccomp(), composite.__name__)
        if caps.program_isolation():
            self.assertTrue(caps.linux_runtime_tools())
            self.assertGreaterEqual(caps.landlock_abi(), 4)


class SchedulingFeedbackTests(unittest.TestCase):
    setUp = setup_dual_arm_harness

    def test_failed_dispatch_exposes_advice_without_claiming_physical_state_or_retry(self):
        args = dict(observation_id=self.h.latest['observation_id'], native_steps=5)
        receipt = dict(status='failed', reason='planning_failed', executed_steps=0)
        def failed(action, correlation):
            return dict(request=action, receipt=receipt)
        with patch.object(self.h, 'execute', side_effect=failed) as execute:
            reply = self.h.dispatch('robot_wait', args, {'tool_call_id': 'failed'})
        result = json.loads(reply['contentItems'][0]['text'])
        self.assertEqual(result['receipt']['status'], 'failed')
        self.assertEqual(result['scheduling_advisory']['source_kind'], 'advisory')
        self.assertIn('if one holds an object', result['scheduling_advisory']['suggestion'])
        execute.assert_called_once()
        self.assertEqual(self.robot.executions, 0)

    def test_success_off_and_program_calls_do_not_read_plan_or_emit_advice(self):
        with patch.object(self.h.task_planning, 'current', side_effect=AssertionError('unnecessary read')):
            self.assertIsNone(self.h._scheduling_advisory({'status': 'completed'}, {}))
            self.assertIsNone(self.h._scheduling_advisory({'status': 'failed'}, {'program_id': 'nested'}))
            for flag in ('task_scheduling', 'manipulation_protocol'):
                original = self.h.config
                self.h.config = original.with_overrides({'features': {flag: False}})
                self.assertIsNone(self.h._scheduling_advisory({'status': 'failed'}, {}))
                self.h.config = original

    def test_unavailable_plan_does_not_hide_the_failed_action(self):
        with patch.object(self.h.task_planning, 'current', side_effect=KeyError('unavailable')):
            result = self.h._scheduling_advisory({'status': 'rejected'}, {})
        self.assertEqual(result['kind'], 'scheduling_review')
        self.assertNotIn('dependency_free_plan_stages_to_reassess', result)


if __name__ == "__main__":
    unittest.main()
