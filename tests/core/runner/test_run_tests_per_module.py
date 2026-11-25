"""scripts/run_tests_per_module.py: a hanging or crashing module is reported as its own failure and never blocks
the run; failures, skips and baseline differences are reported by test id. The runner is copied into a temporary
checkout whose tests/ holds small fixture modules, and runs them in real child interpreters."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest

from support.paths import REPO_ROOT

NI_RUA_TEST_TIER = "component"
RUNNER = REPO_ROOT / "scripts" / "run_tests_per_module.py"
FIXTURES = {
    "test_fixture_pass": """
        import unittest
        class Passing(unittest.TestCase):
            def test_ok(self):
                pass
            @unittest.skip("requires host capability fixture_probe: fixture reason")
            def test_skipped(self):
                pass
            def test_cases(self):
                for case in (1, 2):
                    with self.subTest(case=case):
                        pass
    """,
    "test_fixture_fail": """
        import unittest
        class Failing(unittest.TestCase):
            def test_fails(self):
                self.assertEqual(1, 2)
            def test_errors(self):
                raise RuntimeError("fixture error")
    """,
    "test_fixture_hang": """
        import time, unittest
        class Hanging(unittest.TestCase):
            def test_hangs(self):
                time.sleep(60)
    """,
    "test_fixture_crash": """
        import ctypes, unittest
        class Crashing(unittest.TestCase):
            def test_segfaults(self):
                ctypes.string_at(0)
    """,
}


class PerModuleRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        root = Path(cls.temporary.name)
        (root / "scripts").mkdir()
        shutil.copy(RUNNER, root / "scripts" / RUNNER.name)
        (root / "tests").mkdir()
        for name, source in FIXTURES.items():
            (root / "tests" / f"{name}.py").write_text(textwrap.dedent(source))
        cls.baseline = root / "baseline.json"
        cls.baseline.write_text(json.dumps({"test_fixture_pass": {
            "code": 0, "results": ["FAIL test_fixture_pass.Passing.test_ok",
                                   "ERROR test_fixture_pass.Passing.test_cases (case=2)"], "seconds": 1}}))
        cls.report = root / "report.json"
        cls.completed = subprocess.run(
            [sys.executable, str(root / "scripts" / RUNNER.name), "--jobs", "4", "--timeout", "4",
             "--report", str(cls.report), "--baseline", str(cls.baseline)],
            capture_output=True, text=True, timeout=120)
        cls.result = json.loads(cls.report.read_text())

    def test_hang_and_crash_are_reported_per_module_without_blocking_the_run(self):
        modules = self.result["modules"]
        self.assertEqual(modules["test_fixture_hang"]["status"], "timeout")
        self.assertIn("test_hangs", modules["test_fixture_hang"]["tail"])  # stack dumped before the kill
        self.assertEqual(modules["test_fixture_crash"]["status"], "crashed")
        self.assertLess(modules["test_fixture_crash"]["returncode"], 0)
        self.assertIn("Segmentation fault", modules["test_fixture_crash"]["tail"])
        self.assertEqual(modules["test_fixture_pass"]["status"], "passed")
        self.assertLess(modules["test_fixture_hang"]["seconds"], 30)

    def test_counts_ids_skips_and_exit_status(self):
        totals = self.result["totals"]
        self.assertEqual((totals["modules"], totals["failures"], totals["errors"], totals["timeouts"],
                          totals["crashes"], totals["skipped"]), (4, 1, 1, 1, 1, 1))
        failing = self.result["modules"]["test_fixture_fail"]
        self.assertEqual(list(failing["failures"]), ["test_fixture_fail.Failing.test_fails"])
        self.assertIn("RuntimeError: fixture error", failing["errors"]["test_fixture_fail.Failing.test_errors"])
        self.assertEqual(self.result["modules"]["test_fixture_pass"]["skips"],
                         {"test_fixture_pass.Passing.test_skipped": "requires host capability fixture_probe: "
                                                                    "fixture reason"})
        self.assertEqual(self.completed.returncode, 1, self.completed.stdout + self.completed.stderr)
        self.assertIn("FAILURE: test_fixture_fail.Failing.test_fails", self.completed.stdout)
        self.assertIn("TIMEOUT: test_fixture_hang", self.completed.stdout)
        self.assertIn("CRASHED: test_fixture_crash", self.completed.stdout)

    def test_baseline_comparison_separates_new_failures_from_fixed_tests(self):
        output = self.completed.stdout
        self.assertIn("NEW FAIL: test_fixture_fail.Failing.test_fails", output)
        self.assertIn("NEW timeout: test_fixture_hang", output)
        self.assertIn("PASSES: test_fixture_pass.Passing.test_ok (was FAIL)", output)
        self.assertIn("PASSES: test_fixture_pass.Passing.test_cases (case=2) (was ERROR)", output)


SLOW_FIXTURE = """
import os, pathlib, subprocess, sys, time, unittest
class Slow(unittest.TestCase):
    def test_waits(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])  # stays in the group
        marker = pathlib.Path(os.environ["RUNNER_FIXTURE_MARKERS"]) / (__name__ + ".pids")
        marker.write_text(f"{os.getpid()} {child.pid}")
        time.sleep(120)
"""


def _gone(pid, deadline_s=10):
    """True once ``pid`` has exited. Under scripts/run_tests.py this test process is a Linux child subreaper, so a
    grandchild orphaned by the killed module is re-parented here and stays a zombie until this process reaps it."""
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        with contextlib.suppress(ChildProcessError):  # not our child: its own parent reaps it
            os.waitpid(pid, os.WNOHANG)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(.05)
    return False


class RunnerInterruptionTests(unittest.TestCase):
    """Ctrl-C or SIGTERM: no further module starts, running modules are killed with their process groups, a partial
    report marked interrupted is written, and the exit status is 128 + the signal number."""

    def interrupt(self, signum):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        (root / "scripts").mkdir()
        shutil.copy(RUNNER, root / "scripts" / RUNNER.name)
        (root / "tests").mkdir()
        for name in ("test_fixture_slow_a", "test_fixture_slow_b"):
            (root / "tests" / f"{name}.py").write_text(SLOW_FIXTURE)
        markers, report = root / "markers", root / "report.json"
        markers.mkdir()
        runner = subprocess.Popen(
            [sys.executable, str(root / "scripts" / RUNNER.name), "--jobs", "1", "--timeout", "300",
             "--report", str(report)],
            env=dict(os.environ, RUNNER_FIXTURE_MARKERS=str(markers)), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True,
            # Ctrl-C reaches a runner whose SIGINT is not ignored (a background job may inherit SIG_IGN).
            preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
        try:
            marker = markers / "test_fixture_slow_a.pids"
            deadline = time.monotonic() + 60
            while not marker.exists() or len(marker.read_text().split()) < 2:
                self.assertLess(time.monotonic(), deadline, "the first module never started")
                time.sleep(.05)
            worker, grandchild = map(int, marker.read_text().split())
            runner.send_signal(signum)
            output, _ = runner.communicate(timeout=60)
        finally:
            if runner.poll() is None:
                runner.kill()
                runner.communicate()
        return runner.returncode, json.loads(report.read_text()), output, worker, grandchild, markers

    def test_sigint_and_sigterm_stop_kill_report_and_exit_non_zero(self):
        for signum in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signal=signum.name):
                code, report, output, worker, grandchild, markers = self.interrupt(signum)
                self.assertEqual(code, 128 + signum, output)
                self.assertEqual(report["interrupted"], signum.name)
                self.assertEqual(report["modules"]["test_fixture_slow_a"]["status"], "interrupted")
                self.assertEqual(report["not_run"], ["test_fixture_slow_b"])
                self.assertFalse((markers / "test_fixture_slow_b.pids").exists())
                self.assertEqual(report["totals"]["interrupted"], 1)
                self.assertIn(f"INTERRUPTED by {signum.name}", output)
                self.assertTrue(_gone(worker), "module process survived the interrupt")
                self.assertTrue(_gone(grandchild), "a process in the module's group survived the interrupt")


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_tests_per_module_under_test", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BaselineComparisonTests(unittest.TestCase):
    def test_a_module_that_crashed_or_timed_out_before_is_reported_by_its_current_status(self):
        runner = _load_runner()
        baseline = {"modules": {name: {"status": status, "returncode": -11, "seconds": 1, "tail": ""}
                                for name, status in (("test_now_passes", "crashed"), ("test_now_fails", "timeout"),
                                                     ("test_removed", "crashed"))}}
        current = {"modules": {
            "test_now_passes": {"status": "passed", "passed": ["test_now_passes.Case.test_ok"], "failures": {},
                                "errors": {}, "skips": {}, "unexpected_successes": []},
            "test_now_fails": {"status": "failed", "passed": [], "failures": {"test_now_fails.Case.test_x": "tb"},
                               "errors": {}, "skips": {}, "unexpected_successes": []}}}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            runner.compare(baseline, current)
        lines = output.getvalue()
        self.assertIn("PASSES: test_now_passes (was crashed)", lines)
        self.assertIn("NOW FAILED: test_now_fails (was timeout)", lines)
        self.assertIn("NEW FAIL: test_now_fails.Case.test_x", lines)
        self.assertIn("NOT RUN: test_removed (was crashed)", lines)


class NestedZoneDiscoveryTests(unittest.TestCase):
    def test_discovers_nested_wave2_zone_modules_with_or_without_init(self):
        with tempfile.TemporaryDirectory() as root_str:
            root = Path(root_str)
            (root / "scripts").mkdir()
            shutil.copy(RUNNER, root / "scripts" / RUNNER.name)
            (root / "tests" / "robot" / "so101").mkdir(parents=True)
            (root / "tests" / "robot" / "so101" / "test_ns_zone.py").write_text(
                "import unittest\nclass NsZone(unittest.TestCase):\n    def test_ok(self): self.assertTrue(True)\n",
                encoding="utf-8",
            )
            (root / "tests" / "core" / "config").mkdir(parents=True)
            (root / "tests" / "core" / "__init__.py").write_text("", encoding="utf-8")
            (root / "tests" / "core" / "config" / "__init__.py").write_text("", encoding="utf-8")
            (root / "tests" / "core" / "config" / "test_pkg_zone.py").write_text(
                "import unittest\nclass PkgZone(unittest.TestCase):\n    def test_ok(self): self.assertTrue(True)\n",
                encoding="utf-8",
            )
            report = root / "report.json"
            proc = subprocess.run(
                [sys.executable, str(root / "scripts" / RUNNER.name), "--jobs", "2", "--report", str(report)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            data = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(sorted(data["modules"]), ["core.config.test_pkg_zone", "robot.so101.test_ns_zone"])
            self.assertEqual(data["totals"]["tests"], 2)

    def test_cached_timings_reject_booleans_non_finite_negatives_and_huge_ints(self):
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as tmp:
            timings = Path(tmp) / ".ni-rua-test-timings.json"
            timings.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "modules": {
                            "valid_dict": {"elapsed_s": 12.5, "tests": 3},
                            "valid_zero": {"elapsed_s": 0, "tests": 1},
                            "valid_num": 3.25,
                            "bool_true": {"elapsed_s": True},
                            "bool_raw": False,
                            "negative": {"elapsed_s": -0.1},
                            "too_large_float": {"elapsed_s": 1e9},
                            "huge_int": {"elapsed_s": 10**400},
                            "str_val": {"elapsed_s": "5.0"},
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                runner._load_cached_timings(timings),
                {"valid_dict": 12.5, "valid_zero": 0.0, "valid_num": 3.25},
            )
            nan_inf = Path(tmp) / "nan_inf.json"
            nan_inf.write_text(
                '{"version":1,"modules":{"ok":1.5,"nan":NaN,"inf":Infinity,"-inf":-Infinity}}',
                encoding="utf-8",
            )
            self.assertEqual(runner._load_cached_timings(nan_inf), {"ok": 1.5})


if __name__ == "__main__":
    unittest.main()
