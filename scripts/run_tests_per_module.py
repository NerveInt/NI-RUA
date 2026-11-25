"""Run the CPU test suite one module per process, in parallel, with a per-module timeout.

Every ``tests/test_*.py`` module runs in its own interpreter (the one running this script, normally the ``ni-rua``
environment), so a hang, a segfault or state leaked by one module cannot block or poison the others. A module that
exceeds ``--timeout`` is killed with its whole process group and reported as a timeout; a module whose interpreter
dies without writing its results (signal, ``os._exit``, interpreter abort) is reported as a crash. Both carry the tail
of the module's output, which includes a ``faulthandler`` stack dump of every thread (written shortly before the
timeout, or at the crash).

Usage (from the repository root)::

    mamba run -n ni-rua python scripts/run_tests_per_module.py --jobs 6 --timeout 900 --report /tmp/cpu.json
    mamba run -n ni-rua python scripts/run_tests_per_module.py test_config test_benchmark_queue
    mamba run -n ni-rua python scripts/run_tests_per_module.py --baseline /tmp/cpu.json --report /tmp/cpu-new.json

The JSON report lists, per module, the status (``passed``, ``failed``, ``timeout``, ``crashed``), counts, failing test
ids with their tracebacks, skip reasons, and passing ids (used by ``--baseline`` to tell newly passing tests from
newly skipped ones). ``--baseline`` also accepts the older ``{module: {"results": ["FAIL <id>", ...]}}`` format.
The exit status is 1 when any test failed or errored, or any module timed out or crashed, else 0.

Ctrl-C (SIGINT) or SIGTERM stops the run: no further module starts, the running modules are killed with their
process groups (status ``interrupted``), a partial report marked ``"interrupted": "SIGINT"|"SIGTERM"`` lists them and
the modules that never ran (``not_run``), and the exit status is 130 or 143. Like a timeout, the kill reaches only
processes that stayed in the module's process group; a child that started its own session is not reached.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import fnmatch
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
TAIL_CHARACTERS = 4000
TRACEBACK_CHARACTERS = 6000
# Seconds before the timeout at which the child dumps every thread's stack (so a hang shows where it waits).
STACK_DUMP_LEAD = 10.0
# Set when the run is interrupted; a module starts only under _LIVE_LOCK while it is clear, and every running module
# process is in _LIVE, so the interrupt handler can kill all of them and no new one can start afterwards.
_STOP = threading.Event()
_LIVE, _LIVE_LOCK = set(), threading.Lock()


class _Interrupted(Exception):
    """Raised in the main thread by the SIGTERM handler (SIGINT raises KeyboardInterrupt)."""


def _worker(module, result_path, timeout):
    """Child process: run one test module verbosely on stdout/stderr and write its outcome as JSON."""
    import faulthandler
    import unittest

    # The import path of ``python -m unittest discover -s tests`` run from the root: the tests directory, then
    # the checkout (instead of this script's directory). The environment is left alone, so subprocesses started
    # by tests see the same variables as under unittest discover.
    sys.path[0:1] = [str(ROOT / "tests"), str(ROOT)]
    faulthandler.enable(all_threads=True)
    if timeout > 0:
        faulthandler.dump_traceback_later(max(1.0, timeout - min(STACK_DUMP_LEAD, timeout / 10)), exit=False)

    class Result(unittest.TextTestResult):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.passed = []

        def addSuccess(self, test):
            super().addSuccess(test)
            self.passed.append(test.id())

        def addSubTest(self, test, subtest, err):
            super().addSubTest(test, subtest, err)
            if err is None:  # Passing subtests have their own ids, so a baseline can match them.
                self.passed.append(subtest.id())

    suite = unittest.defaultTestLoader.loadTestsFromName(module)
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2, resultclass=Result).run(suite)
    faulthandler.cancel_dump_traceback_later()
    outcome = {
        "tests": result.testsRun,
        "passed": result.passed,
        "failures": {test.id(): text[-TRACEBACK_CHARACTERS:] for test, text in result.failures},
        "errors": {test.id(): text[-TRACEBACK_CHARACTERS:] for test, text in result.errors},
        "skips": {test.id(): reason for test, reason in result.skipped},
        "expected_failures": [test.id() for test, _ in result.expectedFailures],
        "unexpected_successes": [test.id() for test in result.unexpectedSuccesses],
    }
    Path(result_path).write_text(json.dumps(outcome))
    sys.stdout.flush()
    sys.stderr.flush()


def _tail(path):
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:
        return ""
    return text[-TAIL_CHARACTERS:]


def _kill_group(process):
    """SIGKILL the child's process group (the child and anything it started that did not leave the group)."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def run_module(module, args, log_dir):
    """Run ``module`` in a fresh interpreter; return its report entry."""
    log_path, result_path = log_dir / f"{module}.log", log_dir / f"{module}.result.json"
    result_path.unlink(missing_ok=True)
    command = [args.python, str(Path(__file__).resolve()), "--worker", module, "--result-file", str(result_path),
               "--timeout", str(args.timeout)]
    started = time.monotonic()
    with open(log_path, "wb") as log:
        with _LIVE_LOCK:
            if _STOP.is_set():
                return None  # The run was interrupted before this module started.
            # A new session makes the child a process-group leader, so a timeout can kill everything it started.
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            _LIVE.add(process)
        try:
            returncode = process.wait(timeout=args.timeout if args.timeout > 0 else None)
            timed_out = False
        except subprocess.TimeoutExpired:
            _kill_group(process)
            returncode = process.wait()
            timed_out = True
        finally:
            with _LIVE_LOCK:
                _LIVE.discard(process)
    # Processes the module left behind in its group would outlive the run; stop them and say so.
    try:
        os.killpg(process.pid, 0)
        leaked = True
    except (ProcessLookupError, PermissionError):
        leaked = False
    if leaked:
        _kill_group(process)
    entry = {"returncode": returncode, "seconds": round(time.monotonic() - started, 1), "leaked_processes": leaked,
             "log": str(log_path) if args.log_dir else None}
    outcome = None
    if not timed_out and result_path.exists():
        try:
            outcome = json.loads(result_path.read_text())
        except ValueError:
            outcome = None
    if timed_out:
        entry.update(status="timeout", tail=_tail(log_path))
    elif outcome is None and _STOP.is_set():
        entry.update(status="interrupted", tail=_tail(log_path))
    elif outcome is None:
        entry.update(status="crashed", tail=_tail(log_path))
    else:
        entry.update(outcome)
        if returncode != 0:  # The interpreter died after its tests finished (e.g. a crash at shutdown).
            entry.update(status="crashed", tail=_tail(log_path))
        elif outcome["failures"] or outcome["errors"] or outcome["unexpected_successes"]:
            entry["status"] = "failed"
        else:
            entry["status"] = "passed"
    if not args.log_dir:
        log_path.unlink(missing_ok=True)
    result_path.unlink(missing_ok=True)
    return entry


def failing_ids(report):
    """{test or module id: kind} for a report of this script or of the older per-module format."""
    failing = {}
    modules = report.get("modules", report) if isinstance(report, dict) else {}
    for module, entry in modules.items():
        if not isinstance(entry, dict):
            continue
        if "status" in entry:
            if entry["status"] in ("timeout", "crashed"):
                failing[module] = entry["status"]
            failing.update({test: "FAIL" for test in entry.get("failures", {})})
            failing.update({test: "ERROR" for test in entry.get("errors", {})})
            failing.update({test: "UNEXPECTED SUCCESS" for test in entry.get("unexpected_successes", [])})
        else:  # Older format: {"code": ..., "results": ["FAIL <id>", ...]}.
            if entry.get("code") == "timeout":
                failing[module] = "timeout"
            for line in entry.get("results", []):
                kind, _, test = line.partition(" ")
                failing[test] = kind
    return failing


def compare(baseline, report):
    """Print failures new since ``baseline`` and baseline failures that now pass, skip, or did not run.

    A baseline failure is a test id, or a module name for a module that timed out or crashed; the current state of
    such a module is its status (a module that crashed before and passes now ``PASSES``)."""
    before, after = failing_ids(baseline), failing_ids(report)
    passed = {test for entry in report["modules"].values() for test in entry.get("passed", [])}
    skipped = {test for entry in report["modules"].values() for test in entry.get("skips", {})}
    new = sorted(set(after) - set(before))
    fixed = sorted(set(before) - set(after))
    print(f"\nCompared with baseline: {len(new)} new failing, {len(fixed)} no longer failing")
    for test in new:
        print(f"  NEW {after[test]}: {test}")
    for test in fixed:
        module = report["modules"].get(test)
        if module is not None:
            state = "passes" if module["status"] == "passed" else "now " + module["status"]
        else:
            state = "passes" if test in passed else "skipped" if test in skipped else "not run"
        print(f"  {state.upper()}: {test} (was {before[test]})")


def summarize(report):
    totals = report["totals"]
    print(f"\n{totals['modules']} modules, {totals['tests']} tests, {totals['skipped']} skipped, "
          f"{totals['failures']} failures, {totals['errors']} errors, {totals['unexpected_successes']} unexpected "
          f"successes, {totals['timeouts']} timeouts, {totals['crashes']} crashes "
          f"({totals['seconds']} s wall clock)")
    if report.get("interrupted"):
        print(f"\nINTERRUPTED by {report['interrupted']}: {len(report['not_run'])} modules not run")
    for module, entry in report["modules"].items():
        if entry["status"] in ("timeout", "crashed", "interrupted"):
            print(f"\n{entry['status'].upper()}: {module} (exit {entry['returncode']}, {entry['seconds']} s); "
                  f"output tail:\n{entry['tail']}")
    for kind in ("failures", "errors", "unexpected_successes"):
        for module, entry in report["modules"].items():
            for test in entry.get(kind, []):
                print(f"{kind.rstrip('s').upper().replace('_', ' ')}: {test}")
    reasons = {}
    for entry in report["modules"].values():
        for reason in entry.get("skips", {}).values():
            reasons[reason] = reasons.get(reason, 0) + 1
    if reasons:
        print("\nSkips by reason:")
        for reason, count in sorted(reasons.items(), key=lambda item: (-item[1], item[0])):
            print(f"  {count:4} {reason}")
    leaked = [module for module, entry in report["modules"].items() if entry.get("leaked_processes")]
    if leaked:
        print(f"Modules that left processes behind (killed): {', '.join(leaked)}")


_EXCLUDED_DIRS = frozenset({"__pycache__", ".git", ".claude", "worktrees", "node_modules", ".venv", "venv"})
_ZONE_DIRS = frozenset({"core", "robot", "perception", "cortex", "runtime", "apps", "packaging"})
_LPT_PRIORITIES = {
    "robot.so101.test_so101_benches": 35.0,
    "apps.evaluation.test_benchmark_group": 25.0,
    "cortex.claude_code.test_claude_code_conformance": 20.0,
    "cortex.external.test_external_workbench": 19.5,
    "tests.packaging.sandbox.test_codex_sandbox": 19.0,
    "tests.packaging.sandbox.test_agent_sandbox": 18.5,
    "tests.packaging.sandbox.test_agent_claude_sandbox": 18.0,
    "tests.packaging.sandbox.test_agent_opencode_sandbox": 17.5,
    "cortex.opencode.test_opencode_session_unit": 17.0,
    "core.config.test_config": 16.5,
    "cortex.claude_code.test_claude_session_unit": 16.0,
    "apps.evaluation.test_benchmark_queue": 15.5,
    "robot.calibration.test_so101_calibration_runner": 15.0,
    "apps.gateway.test_gateway_handoff": 14.5,
    "robot.embodiment.test_embodiment_perception_evidence": 14.0,
    "runtime.harness.test_handoff_and_candidates": 13.5,
    "cortex.claude_code.test_claude_options_and_settings": 13.0,
    "apps.gateway.test_gateway_agent_settings": 12.5,
    "runtime.tools.test_tool_registry": 12.0,
    "robot.so101.test_so101_grip_squeeze": 11.8,
    "robot.so101.test_so101_backend": 11.5,
    "robot.so101.test_so101_camera_and_recalibration": 11.2,
    "apps.gateway.test_gateway_core": 11.0,
    "apps.gateway.test_gateway_console": 10.8,
    "runtime.programs.test_program_isolation": 10.5,
    "runtime.programs.test_program_review": 10.0,
    "core.runner.test_parallel_runner": 9.5,
    "runtime.tools.test_robotic_view_isolation": 9.0,
    "robot.embodiment.test_embodiment_x5_golden": 8.5,
}


def _parse_valid_timing_seconds(value):
    if type(value) not in (int, float):
        return None
    if isinstance(value, int) and not (0 <= value < 1_000_000_000):
        return None
    fv = float(value)
    if not math.isfinite(fv) or not (0.0 <= fv < 1e9):
        return None
    return fv


def _load_cached_timings(timings_file=None):
    timings_file = Path(timings_file) if timings_file is not None else (ROOT / ".ni-rua-test-timings.json")
    if not timings_file.is_file():
        return {}
    try:
        raw = json.loads(timings_file.read_text(encoding="utf-8"))
        modules = raw.get("modules") if isinstance(raw, dict) else None
        if not isinstance(modules, dict):
            return {}
        out = {}
        for mod, entry in modules.items():
            if not isinstance(mod, str):
                continue
            raw_val = entry.get("elapsed_s") if isinstance(entry, dict) else entry
            parsed = _parse_valid_timing_seconds(raw_val)
            if parsed is not None:
                out[mod] = parsed
        return out
    except (OSError, ValueError):
        return {}


def _is_zone_subdir(root_path, tests_root, subdir_name):
    if subdir_name in ("fixtures", "support"):
        return False
    if root_path == tests_root and subdir_name in _ZONE_DIRS:
        return True
    return root_path.parent == tests_root and root_path.name in _ZONE_DIRS


def _discover_modules():
    tests_root = ROOT / "tests"
    modules = []
    for root, dirs, files in os.walk(tests_root):
        root_path = Path(root)
        in_zone = (
            (root_path.parent == tests_root and root_path.name in _ZONE_DIRS)
            or (root_path.parent.parent == tests_root and root_path.parent.name in _ZONE_DIRS
                and root_path.name not in ("fixtures", "support"))
        )
        if root_path != tests_root and not (root_path / "__init__.py").is_file() and not in_zone:
            dirs[:] = []
            continue
        dirs[:] = sorted(
            d for d in dirs
            if d not in _EXCLUDED_DIRS
            and ((root_path / d / "__init__.py").is_file() or _is_zone_subdir(root_path, tests_root, d))
        )
        for fname in sorted(files):
            if fnmatch.fnmatch(fname, "test_*.py"):
                rel = (root_path / fname).relative_to(tests_root).with_suffix("")
                if rel.parts and rel.parts[0] == "packaging" and not (tests_root / "packaging" / "__init__.py").is_file():
                    modules.append("tests." + ".".join(rel.parts))
                else:
                    modules.append(".".join(rel.parts))
    return sorted(modules)


def _match_module(module, pattern):
    return (
        fnmatch.fnmatch(module, pattern)
        or fnmatch.fnmatch(module.removeprefix("tests."), pattern)
        or fnmatch.fnmatch(module.rsplit(".", 1)[-1], pattern)
    )


def select_modules(patterns):
    modules = _discover_modules()
    if not patterns:
        return modules
    chosen = [module for module in modules if any(_match_module(module, pattern) for pattern in patterns)]
    missing = [pattern for pattern in patterns if not any(_match_module(module, pattern) for module in modules)]
    if missing:
        raise SystemExit(f"no test module matches: {', '.join(missing)}")
    return chosen


def revision():
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              timeout=30).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True,
                               timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return {"head": head or None, "dirty": bool(dirty)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("modules", nargs="*", help="test module names or glob patterns (default: every "
                        "tests/test_*.py)")
    parser.add_argument("-j", "--jobs", "--workers", dest="jobs", type=int, default=4,
                        help="modules run in parallel (default: 4)")
    parser.add_argument("--timeout", type=float, default=900, help="seconds one module may run before it is killed "
                        "and reported as a timeout; 0 disables (default: 900)")
    parser.add_argument("--report", type=Path, help="write the JSON report to this file")
    parser.add_argument("--baseline", type=Path, help="earlier JSON report: print new failures and baseline "
                        "failures that now pass")
    parser.add_argument("--log-dir", type=Path, help="keep each module's full output as <module>.log here "
                        "(default: temporary, deleted)")
    parser.add_argument("--python", default=sys.executable, help="interpreter for the modules (default: this one)")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--result-file", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        _worker(args.worker, args.result_file, args.timeout)
        return 0
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    modules = select_modules(args.modules)
    if baseline:
        # Longest first (durations from the baseline) so the slowest modules do not finish last.
        seconds = {module: _parse_valid_timing_seconds(entry.get("seconds")) or 0.0
                   for module, entry in baseline.get("modules", baseline).items()
                   if isinstance(entry, dict)}
        modules.sort(key=lambda module: -(seconds.get(module) or 0.0))
    else:
        cached = _load_cached_timings()
        modules.sort(key=lambda module: -(cached.get(module, _LPT_PRIORITIES.get(module, 0.0))))
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    started = time.monotonic()
    temporary = None
    if args.log_dir:
        args.log_dir.mkdir(parents=True, exist_ok=True)
        log_dir = args.log_dir
    else:
        temporary = tempfile.TemporaryDirectory(prefix="ni-rua-tests-")
        log_dir = Path(temporary.name)
    print(f"Running {len(modules)} modules with {args.python}, {args.jobs} in parallel, timeout {args.timeout:g} s",
          flush=True)
    results, interrupted = {}, None

    def record(module, entry):
        results[module] = entry
        detail = (f"{entry['tests']} tests, {len(entry['skips'])} skipped, "
                  f"{len(entry['failures'])} failures, {len(entry['errors'])} errors"
                  if "tests" in entry else f"exit {entry['returncode']}")
        print(f"[{len(results)}/{len(modules)}] {entry['status']:8} {module}: {detail} ({entry['seconds']} s)",
              flush=True)

    def on_sigterm(signum, frame):
        raise _Interrupted("SIGTERM")

    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.signal(signal.SIGTERM, on_sigterm)
    pool, futures = concurrent.futures.ThreadPoolExecutor(args.jobs), {}
    try:
        try:
            for module in modules:
                futures[pool.submit(run_module, module, args, log_dir)] = module
            for future in concurrent.futures.as_completed(futures):
                record(futures[future], future.result())
        except (KeyboardInterrupt, _Interrupted) as error:
            interrupted = "SIGINT" if isinstance(error, KeyboardInterrupt) else str(error)
            # Finish the cleanup even if the signal is repeated.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            with _LIVE_LOCK:
                _STOP.set()
                live = list(_LIVE)
            pool.shutdown(wait=False, cancel_futures=True)
            for process in live:
                _kill_group(process)
            pool.shutdown(wait=True)  # The killed modules return at once and are reported as interrupted.
            for future, module in futures.items():
                if module not in results and future.done() and not future.cancelled() and future.result():
                    record(module, future.result())
    finally:
        pool.shutdown(wait=True)
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(signal.SIGINT, previous_sigint)
        if temporary:
            temporary.cleanup()
    ordered = {module: results[module] for module in sorted(results)}
    totals = {
        "modules": len(ordered),
        "tests": sum(entry.get("tests", 0) for entry in ordered.values()),
        "skipped": sum(len(entry.get("skips", {})) for entry in ordered.values()),
        "failures": sum(len(entry.get("failures", {})) for entry in ordered.values()),
        "errors": sum(len(entry.get("errors", {})) for entry in ordered.values()),
        "unexpected_successes": sum(len(entry.get("unexpected_successes", [])) for entry in ordered.values()),
        "timeouts": sum(entry["status"] == "timeout" for entry in ordered.values()),
        "crashes": sum(entry["status"] == "crashed" for entry in ordered.values()),
        "interrupted": sum(entry["status"] == "interrupted" for entry in ordered.values()),
        "seconds": round(time.monotonic() - started, 1),
    }
    report = {"started_at": started_at, "revision": revision(), "python": args.python, "jobs": args.jobs,
              "timeout": args.timeout, "interrupted": interrupted,
              "not_run": sorted(module for module in modules if module not in results),
              "totals": totals, "modules": ordered}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=1))
    summarize(report)
    if baseline is not None:
        compare(baseline, report)
    if args.report:
        print(f"Report: {args.report}")
    if interrupted:
        return 128 + getattr(signal, interrupted)
    bad = totals["failures"] + totals["errors"] + totals["unexpected_successes"] + totals["timeouts"] \
        + totals["crashes"]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
