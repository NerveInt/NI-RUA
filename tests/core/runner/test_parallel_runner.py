"""并行测试运行器契约测试：环境隔离、夹具边界、标准流抗干扰与崩溃恢复。"""
from __future__ import annotations

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
RUNNER_SCRIPT = REPO_ROOT / "scripts" / "run_tests.py"


class ParallelRunnerContractTests(unittest.TestCase):
    @staticmethod
    def _cleanup_record_files(*paths: Path) -> None:
        for path in paths:
            if path.is_dir():
                candidates = list(path.glob("*.json"))
            elif path.is_file():
                candidates = [path]
            else:
                continue
            for rec_path in candidates:
                try:
                    raw = rec_path.read_text(encoding="utf-8").strip()
                    if not raw:
                        continue
                    rec = json.loads(raw) if raw.startswith("{") else {"pids": [int(raw)]}
                except (OSError, ValueError, TypeError):
                    continue
                if not isinstance(rec, dict):
                    continue
                my_pid = os.getpid()
                pids: list[object] = list(rec.get("pids") or [])
                for k in ("pid", "guard_pid", "orphan_pid", "fork_pid", "refork_pid", "exec_pid"):
                    if rec.get(k) is not None:
                        pids.append(rec[k])
                for raw_pid in pids:
                    try:
                        pid = int(raw_pid)
                    except (ValueError, TypeError):
                        continue
                    if pid <= 1 or pid == my_pid:
                        continue
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except OSError:
                        pass
                    try:
                        os.waitpid(pid, os.WNOHANG)
                    except OSError:
                        pass
                td = rec.get("tmpdir")
                if isinstance(td, str) and td:
                    shutil.rmtree(td, ignore_errors=True)

    @classmethod
    def _terminate_runner_proc(cls, proc: subprocess.Popen[str], *record_paths: Path) -> None:
        try:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except OSError:
                        proc.kill()
                    proc.wait(timeout=3)
        finally:
            for stream in (proc.stdout, proc.stderr):
                if stream is not None:
                    stream.close()
            if record_paths:
                cls._cleanup_record_files(*record_paths)

    def _run_runner(
        self,
        *args: str,
        env_overrides: dict[str, str] | None = None,
        cwd: Path | None = None,
        record_paths: tuple[Path, ...] = (),
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        if env_overrides:
            env.update(env_overrides)
        proc = subprocess.Popen(
            [sys.executable, str(RUNNER_SCRIPT), *args],
            cwd=str(cwd or REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            out, err = proc.communicate(timeout=30)
            return subprocess.CompletedProcess(proc.args, proc.returncode, out, err)
        except BaseException:
            self._terminate_runner_proc(proc, *record_paths)
            raise
        finally:
            self._terminate_runner_proc(proc)

    def test_worker_isolation_short_tmpdir_thread_env_and_stdout_noise_immunity(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="ni_rua_iso_test_"))
        self.addCleanup(shutil.rmtree, root, True)
        suite_dir = root / "suite"
        out_dir = root / "records"
        suite_dir.mkdir()
        out_dir.mkdir()
        self.addCleanup(self._cleanup_record_files, out_dir)

        for idx in range(4):
            (suite_dir / f"test_mod_{idx}.py").write_text(
                textwrap.dedent(
                    f"""
                    import json, os, socket, tempfile, time, unittest
                    from pathlib import Path

                    class IsoTest{idx}(unittest.TestCase):
                        def test_env_and_unix_socket(self):
                            # 直接向 fd 1 写入非 JSON 字节，验证独立控制通道不受污染
                            os.write(1, b"RAW_FD1_NOISE_{idx}\\n")
                            print("PLAIN_PRINT_NOISE_{idx}")
                            time.sleep(0.08)
                            tmp = os.environ["TMPDIR"]
                            self.assertLessEqual(len(tmp), 24)
                            self.assertEqual(os.environ["XDG_RUNTIME_DIR"], tmp)
                            self.assertEqual(os.environ["OMP_NUM_THREADS"], "1")
                            self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"], "1")
                            self.assertEqual(os.environ["MKL_NUM_THREADS"], "1")
                            self.assertEqual(os.environ["NUMEXPR_NUM_THREADS"], "1")
                            self.assertEqual(os.environ["RAYON_NUM_THREADS"], "2")
                            self.assertEqual(os.environ["TOKIO_WORKER_THREADS"], "2")
                            self.assertEqual(os.environ["PYTHONDONTWRITEBYTECODE"], "1")
                            # 验证 64 字符长文件名 AF_UNIX 套接字在深层临时目录下不触碰 108 字节上限
                            with tempfile.TemporaryDirectory() as nested:
                                daemon = Path(nested) / "ipc" / "daemon"
                                daemon.mkdir(parents=True)
                                sock_path = daemon / ("a" * 64)
                                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                                    s.bind(str(sock_path))
                            rec = Path({str(out_dir)!r}) / "rec_{idx}.json"
                            rec_tmp = rec.with_suffix(".tmp")
                            rec_tmp.write_text(json.dumps({{"pid": os.getpid(), "tmpdir": tmp}}), encoding="utf-8")
                            os.replace(rec_tmp, rec)
                    """
                ),
                encoding="utf-8",
            )

        # 即使父环境带有超长 TMPDIR，Worker 仍须分配短路径独立目录并在退出后清理干净
        long_parent_tmp = root / ("very_long_parent_tmpdir_" * 4)
        long_parent_tmp.mkdir()
        proc = self._run_runner(
            "-s",
            str(suite_dir),
            "-j",
            "3",
            env_overrides={"TMPDIR": str(long_parent_tmp)},
            record_paths=(out_dir,),
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("Ran 4 tests", proc.stdout)

        records = [json.loads((out_dir / f"rec_{i}.json").read_text()) for i in range(4)]
        pids = {r["pid"] for r in records}
        tmpdirs = {r["tmpdir"] for r in records}
        self.assertGreaterEqual(len(pids), 2)
        self.assertEqual(len(pids), len(tmpdirs))
        for td in tmpdirs:
            self.assertFalse(Path(td).exists(), f"Worker 临时目录未清理: {td}")

    def test_module_and_class_fixtures_and_setupclass_skip_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as root_str:
            root = Path(root_str)
            suite_dir = root / "suite"
            log_file = root / "events.jsonl"
            suite_dir.mkdir()

            (suite_dir / "test_mod_fixture.py").write_text(
                textwrap.dedent(
                    f"""
                    import json, unittest
                    from pathlib import Path
                    LOG = Path({str(log_file)!r})

                    def _emit(tag):
                        with LOG.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(tag) + "\\n")

                    def setUpModule():
                        _emit("mod_up")

                    def tearDownModule():
                        _emit("mod_down")

                    class ModCaseA(unittest.TestCase):
                        def test_1(self): _emit("mod_a1")
                        def test_2(self): _emit("mod_a2")

                    class ModCaseB(unittest.TestCase):
                        def test_1(self): _emit("mod_b1")
                    """
                ),
                encoding="utf-8",
            )

            (suite_dir / "test_cls_fixture.py").write_text(
                textwrap.dedent(
                    f"""
                    import json, unittest
                    from pathlib import Path
                    LOG = Path({str(log_file)!r})

                    def _emit(tag):
                        with LOG.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(tag) + "\\n")

                    class ClsFixture(unittest.TestCase):
                        @classmethod
                        def setUpClass(cls):
                            _emit("cls_up")

                        @classmethod
                        def tearDownClass(cls):
                            _emit("cls_down")

                        def test_1(self): _emit("cls_1")
                        def test_2(self): _emit("cls_2")

                    class SkippedSetupClass(unittest.TestCase):
                        @classmethod
                        def setUpClass(cls):
                            raise unittest.SkipTest("missing optional external asset")

                        def test_never_runs_1(self): raise AssertionError("unreachable")
                        def test_never_runs_2(self): raise AssertionError("unreachable")
                    """
                ),
                encoding="utf-8",
            )
            (suite_dir / "test_mod_skipped.py").write_text(
                'import unittest\nraise unittest.SkipTest("import-time module skip")\n',
                encoding="utf-8",
            )

            proc = self._run_runner("-s", str(suite_dir), "-j", "4")
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("OK (skipped=2)", proc.stdout)
            events = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(events.count("mod_up"), 1)
            self.assertEqual(events.count("mod_down"), 1)
            self.assertEqual(events.count("cls_up"), 1)
            self.assertEqual(events.count("cls_down"), 1)
            self.assertLess(events.index("mod_up"), events.index("mod_a1"))
            self.assertGreater(events.index("mod_down"), events.index("mod_b1"))
            self.assertLess(events.index("cls_up"), events.index("cls_1"))
            self.assertGreater(events.index("cls_down"), events.index("cls_2"))

            # 精确方法过滤在含 setUpClass 及 -j 1 模式下均只运行目标单测，不得误跑同模块/同类其他测试
            for jobs_arg in ("2", "1"):
                log_file.unlink()
                single = self._run_runner("-s", str(suite_dir), "-j", jobs_arg, "test_cls_fixture.ClsFixture.test_1")
                self.assertEqual(single.returncode, 0, single.stdout + single.stderr)
                self.assertIn("Ran 1 tests", single.stdout)
                single_events = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]
                self.assertEqual(single_events, ["cls_up", "cls_1", "cls_down"])

    def test_worker_crash_is_reported_and_sibling_units_complete_on_replacement_worker(self) -> None:
        with tempfile.TemporaryDirectory() as root_str:
            suite_dir = Path(root_str) / "suite"
            marker = Path(root_str) / "survived.txt"
            crash_meta = Path(root_str) / "crash_meta.json"
            refork_ready_file = Path(root_str) / "refork_ready.txt"
            refork_child_file = Path(root_str) / "refork_grandchild.txt"
            suite_dir.mkdir()
            # 注册后置兜底清理：若被测清理逻辑回归导致下方断言失败，在测试退出时兜底杀灭孤儿并删目录
            cleanup_copy_dir = Path(tempfile.mkdtemp(prefix="test_crash_guard_"))
            self.addCleanup(shutil.rmtree, cleanup_copy_dir, True)
            self.addCleanup(self._cleanup_record_files, cleanup_copy_dir)

            (suite_dir / "test_a_crash.py").write_text(
                textwrap.dedent(
                    f"""
                    import ctypes, json, os, shutil, signal, subprocess, sys, time, unittest
                    from pathlib import Path

                    def _atomic_write(dest: Path, text: str) -> None:
                        tmp = dest.with_suffix(dest.suffix + ".tmp")
                        tmp.write_text(text, encoding="utf-8")
                        os.replace(tmp, dest)

                    class CrashTest(unittest.TestCase):
                        def test_hard_exit(self):
                            if sys.platform != "linux":
                                same_pg = subprocess.Popen(
                                    [
                                        sys.executable,
                                        "-S",
                                        "-c",
                                        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)",
                                    ],
                                    cwd="/",
                                    env={{}},
                                    stdin=subprocess.DEVNULL,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL,
                                    close_fds=True,
                                )
                                payload = json.dumps({{
                                    "orphan_pid": same_pg.pid,
                                    "tmpdir": os.environ["TMPDIR"],
                                }})
                                _atomic_write(Path({str(crash_meta)!r}), payload)
                                _atomic_write(Path({str(cleanup_copy_dir / 'crash.json')!r}), payload)
                                os.write(2, b"simulated fatal crash\\n")
                                os._exit(137)
                            # 1) 清空环境 + 根目录 + DEVNULL + 独立会话且忽略 SIGTERM 的孤儿进程
                            orphan = subprocess.Popen(
                                [
                                    sys.executable,
                                    "-S",
                                    "-c",
                                    "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)",
                                ],
                                cwd="/",
                                env={{}},
                                stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                start_new_session=True,
                                close_fds=True,
                            )
                            # 2) 原生 libc.fork() 子进程继承控制 socket FD 且忽略 SIGTERM，验证不会阻塞父进程
                            libc = ctypes.CDLL(None)
                            libc.fork.restype = ctypes.c_int
                            fork_pid = int(libc.fork())
                            if fork_pid < 0:
                                raise OSError("native fork failed")
                            if fork_pid == 0:
                                os.setsid()
                                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                                time.sleep(60)
                                os._exit(0)
                            # 3) 清理期收到 SIGTERM 时立即再次 fork+setsid 派生二级忽略 SIGTERM 子进程，验证收敛回收不挂死
                            refork_code = (
                                "import ctypes, os, signal, time\\n"
                                "from pathlib import Path\\n"
                                "def _aw(p, t):\\n"
                                "    tp = p.with_suffix(p.suffix + '.tmp')\\n"
                                "    tp.write_text(t, encoding='utf-8')\\n"
                                "    os.replace(tp, p)\\n"
                                "def _on_term(_s, _f):\\n"
                                "    libc = ctypes.CDLL(None)\\n"
                                "    libc.fork.restype = ctypes.c_int\\n"
                                "    gpid = int(libc.fork())\\n"
                                "    if gpid == 0:\\n"
                                "        os.setsid()\\n"
                                "        signal.signal(signal.SIGTERM, signal.SIG_IGN)\\n"
                                "        time.sleep(60)\\n"
                                "        os._exit(0)\\n"
                                "    if gpid > 1:\\n"
                                f"        _aw(Path({str(refork_child_file)!r}), str(gpid))\\n"
                                f"        _aw(Path({str(cleanup_copy_dir / 'gpid.json')!r}), str(gpid))\\n"
                                "    os._exit(0)\\n"
                                "signal.signal(signal.SIGTERM, _on_term)\\n"
                                f"_aw(Path({str(refork_ready_file)!r}), 'ready')\\n"
                                "time.sleep(60)\\n"
                            )
                            refork_proc = subprocess.Popen(
                                [sys.executable, "-S", "-c", refork_code],
                                cwd="/",
                                env={{}},
                                stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                start_new_session=True,
                                close_fds=True,
                            )
                            ready_deadline = time.monotonic() + 3.0
                            while not Path({str(refork_ready_file)!r}).is_file() and time.monotonic() < ready_deadline:
                                time.sleep(0.01)
                            payload = json.dumps({{
                                "orphan_pid": orphan.pid,
                                "fork_pid": fork_pid,
                                "refork_pid": refork_proc.pid,
                                "tmpdir": os.environ["TMPDIR"],
                            }})
                            _atomic_write(Path({str(crash_meta)!r}), payload)
                            _atomic_write(Path({str(cleanup_copy_dir / 'crash.json')!r}), payload)
                            os.write(2, b"simulated fatal crash\\n")
                            os._exit(137)
                    """
                ),
                encoding="utf-8",
            )
            (suite_dir / "test_b_ok.py").write_text(
                textwrap.dedent(
                    f"""
                    import unittest
                    from pathlib import Path

                    class SurviveTest(unittest.TestCase):
                        def test_still_runs(self):
                            Path({str(marker)!r}).write_text("ok", encoding="utf-8")
                    """
                ),
                encoding="utf-8",
            )

            proc = self._run_runner("-s", str(suite_dir), "-j", "1", record_paths=(cleanup_copy_dir,))
            self.assertEqual(proc.returncode, 1)
            self.assertIn("Worker 进程异常退出 (returncode=137)", proc.stdout)
            self.assertIn("simulated fatal crash", proc.stdout)
            self.assertTrue(marker.is_file(), "崩溃补位 Worker 未继续执行剩余测试单元")
            meta = json.loads(crash_meta.read_text(encoding="utf-8"))
            self.assertFalse(Path(meta["tmpdir"]).exists(), f"崩溃 Worker 临时目录未清理: {meta['tmpdir']}")
            if sys.platform == "linux":
                for leaked_key in ("orphan_pid", "fork_pid", "refork_pid"):
                    with self.assertRaises(ProcessLookupError):
                        os.kill(int(meta[leaked_key]), 0)
                self.assertTrue(refork_child_file.is_file(), "SIGTERM 清理阶段二次 fork 未触发")
                gpid = int(refork_child_file.read_text(encoding="utf-8").strip())
                self.assertGreater(gpid, 1)
                with self.assertRaises(ProcessLookupError):
                    os.kill(gpid, 0)
            else:
                with self.assertRaises(ProcessLookupError):
                    os.kill(int(meta["orphan_pid"]), 0)

    def test_subtest_unexpected_success_import_error_and_zero_match_exit_codes(self) -> None:
        with tempfile.TemporaryDirectory() as root_str:
            suite_dir = Path(root_str) / "suite"
            suite_dir.mkdir()

            (suite_dir / "test_outcomes.py").write_text(
                textwrap.dedent(
                    """
                    import unittest

                    class OutcomeChecks(unittest.TestCase):
                        def test_subtest_failure(self):
                            for v in (1, 2):
                                with self.subTest(v=v):
                                    self.assertEqual(v, 1)

                        @unittest.expectedFailure
                        def test_unexpected_pass(self):
                            self.assertEqual(1, 1)

                        def test_clean_pass(self):
                            self.assertTrue(True)
                    """
                ),
                encoding="utf-8",
            )

            # 0 匹配测试须对齐 Python 3.12 unittest 返回码 5
            zero = self._run_runner("-s", str(suite_dir), "-k", "__no_match__")
            self.assertEqual(zero.returncode, 5)
            self.assertIn("NO TESTS RAN", zero.stdout)

            # subTest 失败返回 1
            sub_fail = self._run_runner("-s", str(suite_dir), "-k", "test_subtest_failure", "-j", "2")
            self.assertEqual(sub_fail.returncode, 1)
            self.assertIn("FAIL:", sub_fail.stdout)

            # unexpectedSuccess 返回 1
            unexp = self._run_runner("-s", str(suite_dir), "-k", "test_unexpected_pass", "-j", "2")
            self.assertEqual(unexp.returncode, 1)
            self.assertIn("UNEXPECTED SUCCESS:", unexp.stdout)

            # 文件路径目标归一化与纯通过返回 0
            clean = self._run_runner(
                "-s",
                str(suite_dir),
                str(suite_dir / "test_outcomes.py"),
                "-k",
                "test_clean_pass",
            )
            self.assertEqual(clean.returncode, 0, clean.stdout + clean.stderr)

            # 语法错误文件发现期报错返回 1，绝不静默漏测
            (suite_dir / "test_broken_syntax.py").write_text("def broken(:\n", encoding="utf-8")
            broken = self._run_runner("-s", str(suite_dir), "-j", "2")
            self.assertEqual(broken.returncode, 1)
            self.assertIn("ERROR:", broken.stdout)

    def test_failfast_target_normalization_and_jobs_env_validation(self) -> None:
        with tempfile.TemporaryDirectory() as root_str:
            tests_dir = Path(root_str) / "tests"
            marker = Path(root_str) / "after_failfast.txt"
            tests_dir.mkdir()

            (tests_dir / "test_alpha.py").write_text(
                textwrap.dedent(
                    """
                    import unittest

                    class AlphaTests(unittest.TestCase):
                        def test_fail_immediately(self):
                            self.fail("boom")
                    """
                ),
                encoding="utf-8",
            )
            (tests_dir / "test_omega.py").write_text(
                textwrap.dedent(
                    f"""
                    import unittest
                    from pathlib import Path

                    class OmegaTests(unittest.TestCase):
                        def test_should_not_run_on_failfast(self):
                            Path({str(marker)!r}).write_text("ran", encoding="utf-8")
                    """
                ),
                encoding="utf-8",
            )

            # 单进程顺序与多进程同分片 failfast：首个方法失败后，同模块/同分片内后续方法亦不得执行
            intra_marker = Path(root_str) / "intra_shard_after_failfast.txt"
            (tests_dir / "test_alpha.py").write_text(
                textwrap.dedent(
                    f"""
                    import unittest
                    from pathlib import Path

                    class AlphaTests(unittest.TestCase):
                        def test_01_fail_immediately(self):
                            self.fail("boom")

                        def test_02_same_shard_should_not_run(self):
                            Path({str(intra_marker)!r}).write_text("ran_same_shard", encoding="utf-8")
                    """
                ),
                encoding="utf-8",
            )
            for jobs_val in ("1", "2"):
                ff = self._run_runner("-s", str(tests_dir), "-j", jobs_val, "-f", "test_alpha")
                self.assertEqual(ff.returncode, 1)
                self.assertFalse(intra_marker.exists(), f"-j {jobs_val} -f 未阻断同分片后续方法执行")
                self.assertIn("Ran 1 tests", ff.stdout)

            ff_all = self._run_runner("-s", str(tests_dir), "-j", "1", "-f")
            self.assertEqual(ff_all.returncode, 1)
            self.assertFalse(marker.exists())
            self.assertFalse(intra_marker.exists())

            # -j 2 -f 跨 Worker 并发执行时，任一 Worker 失败必须立即中止另一活跃 Worker 并清理其子孙与临时目录
            ff_par_root = Path(tempfile.mkdtemp(prefix="ni_rua_ff_test_"))
            self.addCleanup(shutil.rmtree, ff_par_root, True)
            ff_suite = ff_par_root / "suite"
            ff_rec_dir = ff_par_root / "records"
            ff_suite.mkdir()
            ff_rec_dir.mkdir()
            self.addCleanup(self._cleanup_record_files, ff_rec_dir)
            peer_rec_file = ff_rec_dir / "peer.json"
            (ff_suite / "test_slow_peer.py").write_text(
                textwrap.dedent(
                    f"""
                    import json, os, subprocess, sys, time, unittest
                    from pathlib import Path

                    class SlowPeerTest(unittest.TestCase):
                        def test_hold_peer(self):
                            child = subprocess.Popen(
                                [sys.executable, "-c", "import time; time.sleep(30)"],
                                start_new_session=True,
                            )
                            self.addCleanup(lambda: (child.kill(), child.wait()))
                            rec_path = Path({str(peer_rec_file)!r})
                            tmp_path = rec_path.with_suffix(".json.tmp")
                            tmp_path.write_text(
                                json.dumps({{"tmpdir": os.environ["TMPDIR"], "pids": [os.getpid(), child.pid]}}),
                                encoding="utf-8",
                            )
                            os.replace(tmp_path, rec_path)
                            time.sleep(30)
                    """
                ),
                encoding="utf-8",
            )
            (ff_suite / "test_fast_fail.py").write_text(
                textwrap.dedent(
                    f"""
                    import time, unittest
                    from pathlib import Path

                    class FastFailTest(unittest.TestCase):
                        def test_wait_peer_then_fail(self):
                            rec_path = Path({str(peer_rec_file)!r})
                            deadline = time.monotonic() + 5.0
                            while not rec_path.is_file() and time.monotonic() < deadline:
                                time.sleep(0.01)
                            self.assertTrue(rec_path.is_file(), "对端 Worker 未并发启动")
                            self.fail("trigger cross-worker failfast")
                    """
                ),
                encoding="utf-8",
            )
            ff_t0 = time.monotonic()
            ff_par = self._run_runner("-s", str(ff_suite), "-j", "2", "-f", record_paths=(ff_rec_dir,))
            ff_elapsed = time.monotonic() - ff_t0
            self.assertEqual(ff_par.returncode, 1, ff_par.stdout + ff_par.stderr)
            self.assertLess(ff_elapsed, 10.0, "-j 2 -f 未及时中断并发运行的慢速 Worker")
            peer_rec = json.loads(peer_rec_file.read_text(encoding="utf-8"))
            self.assertFalse(Path(peer_rec["tmpdir"]).exists(), f"-f 中止后残留临时目录: {peer_rec['tmpdir']}")
            for pid in peer_rec["pids"]:
                with self.assertRaises(ProcessLookupError):
                    os.kill(int(pid), 0)

            # 支持带 tests. 前缀的目标归一化加载
            prefixed = self._run_runner("-s", str(tests_dir), "tests.test_omega", env_overrides={"NI_RUA_TEST_JOBS": "2"})
            self.assertEqual(prefixed.returncode, 0, prefixed.stdout + prefixed.stderr)
            self.assertTrue(marker.is_file())

            # 支持传入子包目录路径作为目标，且在非仓库工作目录下保留调用方工作目录并解析相对路径夹具
            fixtures_dir = Path(root_str) / "fixtures"
            fixtures_dir.mkdir()
            (fixtures_dir / "input.json").write_text('{"token": "caller-cwd-ok"}', encoding="utf-8")
            (tests_dir / "__init__.py").write_text("", encoding="utf-8")
            subpkg = tests_dir / "sub"
            subpkg.mkdir()
            (subpkg / "__init__.py").write_text("", encoding="utf-8")
            (subpkg / "test_sub.py").write_text(
                textwrap.dedent(
                    """
                    import json, unittest
                    from pathlib import Path

                    class SubTests(unittest.TestCase):
                        def test_sub_one(self): self.assertTrue(True)
                        def test_sub_two(self):
                            data = json.loads(Path("fixtures/input.json").read_text(encoding="utf-8"))
                            self.assertEqual(data["token"], "caller-cwd-ok")
                    """
                ),
                encoding="utf-8",
            )
            dir_run = self._run_runner("-s", "tests", "tests/sub", "-j", "2", cwd=Path(root_str))
            self.assertEqual(dir_run.returncode, 0, dir_run.stdout + dir_run.stderr)
            self.assertIn("Ran 2 tests", dir_run.stdout)

            # 不传 -s 直接指定仓库外目录、仓库外包目录及仓库外单文件路径，Worker 均须同步导入根并成功执行
            ext_plain = Path(root_str) / "external_plain"
            ext_plain.mkdir()
            (ext_plain / "test_ext_plain.py").write_text(
                "import unittest\nclass ExtPlain(unittest.TestCase):\n    def test_ok(self): self.assertEqual(2 + 2, 4)\n",
                encoding="utf-8",
            )
            ext_plain_run = self._run_runner(str(ext_plain), "-j", "2")
            self.assertEqual(ext_plain_run.returncode, 0, ext_plain_run.stdout + ext_plain_run.stderr)
            self.assertIn("Ran 1 tests", ext_plain_run.stdout)

            ext_pkg_holder = Path(root_str) / "external_holder"
            ext_pkg = ext_pkg_holder / "extpkg"
            ext_pkg.mkdir(parents=True)
            (ext_pkg / "__init__.py").write_text("VALUE = 42\n", encoding="utf-8")
            (ext_pkg / "test_ext_pkg.py").write_text(
                "import unittest\nfrom extpkg import VALUE\nclass ExtPkg(unittest.TestCase):\n    def test_val(self): self.assertEqual(VALUE, 42)\n",
                encoding="utf-8",
            )
            ext_pkg_run = self._run_runner(str(ext_pkg), "-j", "2")
            self.assertEqual(ext_pkg_run.returncode, 0, ext_pkg_run.stdout + ext_pkg_run.stderr)
            self.assertIn("Ran 1 tests", ext_pkg_run.stdout)

            ext_file_run = self._run_runner(str(ext_pkg / "test_ext_pkg.py"), "-j", "2")
            self.assertEqual(ext_file_run.returncode, 0, ext_file_run.stdout + ext_file_run.stderr)
            self.assertIn("Ran 1 tests", ext_file_run.stdout)

            # 多个仓库外同名测试文件或目录必须明确报错拒绝，绝不能因模块名碰撞串测导致假通过
            dup_a = Path(root_str) / "dup_a"
            dup_b = Path(root_str) / "dup_b"
            dup_a.mkdir()
            dup_b.mkdir()
            (dup_a / "test_common.py").write_text(
                "import unittest\nclass Common(unittest.TestCase):\n    def test_one(self): self.assertTrue(True)\n",
                encoding="utf-8",
            )
            (dup_b / "test_common.py").write_text(
                "import unittest\nclass Common(unittest.TestCase):\n    def test_one(self): self.fail('must not be shadowed')\n",
                encoding="utf-8",
            )
            dup_files = self._run_runner(str(dup_a / "test_common.py"), str(dup_b / "test_common.py"), "-j", "2")
            self.assertEqual(dup_files.returncode, 1)
            self.assertIn("测试目标模块名冲突", dup_files.stdout)

            dup_dirs = self._run_runner(str(dup_a), str(dup_b), "-j", "2")
            self.assertEqual(dup_dirs.returncode, 1)
            self.assertIn("测试目标模块名冲突", dup_dirs.stdout)

            # 跨目录混合选择时，另一目录内未被选中的同名文件（无论顶层模块还是子包模块）绝不得遮蔽已选中的失败测试文件
            (dup_a / "test_other_a.py").write_text(
                "import test_common, unittest\nclass OtherA(unittest.TestCase):\n    def test_ok(self): self.assertTrue(hasattr(test_common, 'Common'))\n",
                encoding="utf-8",
            )
            pkg_root_a = Path(root_str) / "pkg_root_a"
            pkg_root_b = Path(root_str) / "pkg_root_b"
            (pkg_root_a / "pkg").mkdir(parents=True)
            (pkg_root_b / "pkg").mkdir(parents=True)
            (pkg_root_a / "pkg" / "__init__.py").write_text("", encoding="utf-8")
            (pkg_root_b / "pkg" / "__init__.py").write_text("", encoding="utf-8")
            (pkg_root_a / "pkg" / "test_common.py").write_text(
                "import unittest\nclass PkgCommon(unittest.TestCase):\n    def test_pkg_fail(self): self.fail('pkg_a failure must not be shadowed')\n",
                encoding="utf-8",
            )
            (pkg_root_b / "pkg" / "test_common.py").write_text(
                "import unittest\nVAL = 7\nclass PkgCommon(unittest.TestCase):\n    def test_pkg_fail(self): self.assertTrue(True)\n",
                encoding="utf-8",
            )
            (pkg_root_b / "pkg" / "test_other.py").write_text(
                "import unittest\nfrom pkg import test_common as tc\nclass PkgOther(unittest.TestCase):\n    def test_pkg_ok(self): self.assertEqual(tc.VAL, 7)\n",
                encoding="utf-8",
            )
            for jobs_arg in ("1", "2"):
                for ordered_targets in (
                    (str(dup_b / "test_common.py"), str(dup_a / "test_other_a.py")),
                    (str(dup_a / "test_other_a.py"), str(dup_b / "test_common.py")),
                ):
                    mixed = self._run_runner("-s", str(dup_b), *ordered_targets, "-j", jobs_arg)
                    self.assertEqual(mixed.returncode, 1, mixed.stdout + mixed.stderr)
                    self.assertIn("Ran 2 tests", mixed.stdout)
                    self.assertIn("must not be shadowed", mixed.stdout)

                for ordered_pkg_targets in (
                    (str(pkg_root_a / "pkg" / "test_common.py"), str(pkg_root_b / "pkg" / "test_other.py")),
                    (str(pkg_root_b / "pkg" / "test_other.py"), str(pkg_root_a / "pkg" / "test_common.py")),
                ):
                    mixed_pkg = self._run_runner("-s", str(pkg_root_a), *ordered_pkg_targets, "-j", jobs_arg)
                    self.assertEqual(mixed_pkg.returncode, 1, mixed_pkg.stdout + mixed_pkg.stderr)
                    self.assertIn("Ran 2 tests", mixed_pkg.stdout)
                    self.assertIn("pkg_a failure must not be shadowed", mixed_pkg.stdout)

            # 同包同时选中两个模块文件时，不得误逐父包缓存或丢失 setUpModule 原子边界
            pkg_fix_log = Path(root_str) / "pkg_fix.jsonl"
            (subpkg / "test_pkg_mod_fix.py").write_text(
                textwrap.dedent(
                    f"""
                    import json, unittest
                    from pathlib import Path
                    LOG = Path({str(pkg_fix_log)!r})
                    def setUpModule():
                        with LOG.open("a", encoding="utf-8") as f: f.write(json.dumps("up") + "\\n")
                    def tearDownModule():
                        with LOG.open("a", encoding="utf-8") as f: f.write(json.dumps("down") + "\\n")
                    class FixA(unittest.TestCase):
                        def test_1(self): self.assertTrue(True)
                    class FixB(unittest.TestCase):
                        def test_2(self): self.assertTrue(True)
                    """
                ),
                encoding="utf-8",
            )
            sibling_pkg = self._run_runner(
                "-s",
                str(tests_dir),
                str(subpkg / "test_pkg_mod_fix.py"),
                str(subpkg / "test_sub.py"),
                "-j",
                "2",
                cwd=Path(root_str),
            )
            self.assertEqual(sibling_pkg.returncode, 0, sibling_pkg.stdout + sibling_pkg.stderr)
            self.assertIn("Ran 4 tests", sibling_pkg.stdout)
            self.assertEqual(
                [json.loads(x) for x in pkg_fix_log.read_text(encoding="utf-8").splitlines()],
                ["up", "down"],
            )

            serial_single = self._run_runner("-s", str(tests_dir), "-j", "1", "sub.test_sub.SubTests.test_sub_one")
            self.assertEqual(serial_single.returncode, 0, serial_single.stdout + serial_single.stderr)
            self.assertIn("Ran 1 tests", serial_single.stdout)

            # 非法并发数必须明确报错拒绝
            bad_jobs = self._run_runner("-s", str(tests_dir), "-j", "0")
            self.assertNotEqual(bad_jobs.returncode, 0)
            bad_env = self._run_runner("-s", str(tests_dir), env_overrides={"NI_RUA_TEST_JOBS": "0"})
            self.assertNotEqual(bad_env.returncode, 0)

    def test_repeated_sigterm_cleans_worker_tmpdirs_and_descendants(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="ni_rua_sig_test_"))
        self.addCleanup(shutil.rmtree, root, True)
        suite_dir = root / "suite"
        rec_dir = root / "records"
        suite_dir.mkdir()
        rec_dir.mkdir()
        self.addCleanup(self._cleanup_record_files, rec_dir)

        # 注入截断损坏记录文件，验证兜底清理遇到半写/空文件时跳过并继续清理其余合法记录
        (rec_dir / "w_corrupt.json").write_text("{", encoding="utf-8")
        (rec_dir / "w_empty.json").write_text("", encoding="utf-8")

        for idx in range(3):
            (suite_dir / f"test_slow_{idx}.py").write_text(
                textwrap.dedent(
                    f"""
                    import json, os, subprocess, sys, time, unittest
                    from pathlib import Path

                    class SlowSigTest{idx}(unittest.TestCase):
                        def test_hold_children(self):
                            same_pg = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
                            new_sess = subprocess.Popen(
                                [sys.executable, "-c", "import time; time.sleep(30)"],
                                start_new_session=True,
                            )
                            self.addCleanup(lambda: (new_sess.kill(), new_sess.wait()))
                            self.addCleanup(lambda: (same_pg.kill(), same_pg.wait()))
                            rec_path = Path({str(rec_dir)!r}, "w_{idx}.json")
                            tmp_path = rec_path.with_suffix(".json.tmp")
                            tmp_path.write_text(
                                json.dumps({{
                                    "tmpdir": os.environ["TMPDIR"],
                                    "pids": [os.getppid(), os.getpid(), same_pg.pid, new_sess.pid],
                                }}),
                                encoding="utf-8",
                            )
                            os.replace(tmp_path, rec_path)
                            time.sleep(30)
                    """
                ),
                encoding="utf-8",
            )

        proc = subprocess.Popen(
            [sys.executable, str(RUNNER_SCRIPT), "-s", str(suite_dir), "-j", "3"],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 10.0
            while len(list(rec_dir.glob("w_[0-2].json"))) < 3 and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(len(list(rec_dir.glob("w_[0-2].json"))), 3, "Worker 未在预期时间内全部启动")

            # 连续多次发送 SIGTERM，验证清理阶段与退出收尾均不被后续信号打断且无进程/目录残留
            for delay in (0.003, 0.01):
                if proc.poll() is None:
                    proc.send_signal(signal.SIGTERM)
                time.sleep(delay)
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
            proc.communicate(timeout=10)
            self.assertEqual(proc.returncode, 128 + signal.SIGTERM)

            for rec_path in rec_dir.glob("w_[0-2].json"):
                rec = json.loads(rec_path.read_text(encoding="utf-8"))
                self.assertFalse(Path(rec["tmpdir"]).exists(), f"中断后残留临时目录: {rec['tmpdir']}")
                for pid in rec["pids"]:
                    with self.assertRaises(ProcessLookupError):
                        os.kill(pid, 0)
        finally:
            self._terminate_runner_proc(proc, rec_dir)

        # 验证 _cleanup_record_files 在存在损坏记录且含非法/自身 PID (-1, 0, 1, self) 时依然安全跳过并清理合法目录
        tail_tmp = Path(tempfile.mkdtemp(prefix="ni_rua_tail_cleanup_"))
        self.addCleanup(shutil.rmtree, tail_tmp, True)
        (rec_dir / "z_tail.json").write_text(
            json.dumps({"tmpdir": str(tail_tmp), "pids": [-1, 0, 1, os.getpid()]}),
            encoding="utf-8",
        )
        self._cleanup_record_files(rec_dir)
        self.assertFalse(tail_tmp.exists(), "_cleanup_record_files 被前置损坏记录阻断")
        (rec_dir / "z_tail.json").unlink()

        # 进一步验证主控进程被 SIGKILL 强杀绕过 finally 时，Linux Worker 守护进程通过 PR_SET_PDEATHSIG 自清目录与孤儿
        if sys.platform == "linux":
            import ctypes

            libc = ctypes.CDLL(None, use_errno=True)
            prev_subreaper = ctypes.c_int(0)
            if libc.prctl(37, ctypes.byref(prev_subreaper), 0, 0, 0) == 0 and prev_subreaper.value == 0:
                if libc.prctl(36, 1, 0, 0, 0) == 0:
                    self.addCleanup(libc.prctl, 36, 0, 0, 0, 0)

            for old_rec in rec_dir.glob("w_[0-2].json"):
                old_rec.unlink()
            kill_proc = subprocess.Popen(
                [sys.executable, str(RUNNER_SCRIPT), "-s", str(suite_dir), "-j", "2"],
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            try:
                kill_deadline = time.monotonic() + 10.0
                while len(list(rec_dir.glob("w_[0-2].json"))) < 2 and time.monotonic() < kill_deadline:
                    time.sleep(0.02)
                self.assertEqual(len(list(rec_dir.glob("w_[0-2].json"))), 2, "Worker 未在预期时间内全部启动")
                kill_proc.kill()
                kill_proc.communicate(timeout=5)
                recs = [json.loads(p.read_text(encoding="utf-8")) for p in rec_dir.glob("w_[0-2].json")]
                reap_deadline = time.monotonic() + 3.0
                while time.monotonic() < reap_deadline:
                    still_dirty = False
                    for rec in recs:
                        if Path(rec["tmpdir"]).exists():
                            still_dirty = True
                        for pid in rec["pids"]:
                            try:
                                os.waitpid(int(pid), os.WNOHANG)
                            except OSError:
                                pass
                            try:
                                os.kill(int(pid), 0)
                                still_dirty = True
                            except ProcessLookupError:
                                pass
                    if not still_dirty:
                        break
                    time.sleep(0.02)
                for rec in recs:
                    self.assertFalse(Path(rec["tmpdir"]).exists(), f"主控 SIGKILL 后残留临时目录: {rec['tmpdir']}")
                    for pid in rec["pids"]:
                        with self.assertRaises(ProcessLookupError):
                            os.kill(int(pid), 0)
            finally:
                self._terminate_runner_proc(kill_proc, rec_dir)

    def test_spawn_worker_base_exception_interrupt_leaves_no_leaks(self) -> None:
        from contextlib import redirect_stdout
        import importlib.util
        import io
        from unittest.mock import patch

        spec = importlib.util.spec_from_file_location("_ni_rua_run_tests_mod", RUNNER_SCRIPT)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        self.addCleanup(sys.modules.pop, spec.name, None)
        spec.loader.exec_module(mod)

        def _fallback_box_cleanup(box: dict[str, object]) -> None:
            sock_obj = box.get("sock")
            if sock_obj is not None and hasattr(sock_obj, "close"):
                try:
                    sock_obj.close()
                except OSError:
                    pass
            for key in ("pid", "exec_pid"):
                raw = box.get(key)
                if isinstance(raw, int) and raw > 1:
                    try:
                        os.kill(raw, signal.SIGKILL)
                    except OSError:
                        pass
                    try:
                        os.waitpid(raw, os.WNOHANG)
                    except OSError:
                        pass
            td = box.get("tmpdir")
            if isinstance(td, Path):
                shutil.rmtree(td, ignore_errors=True)

        captured: dict[str, object] = {}
        self.addCleanup(_fallback_box_cleanup, captured)

        def _wait_exec_child(guard_pid: int) -> int | None:
            wait_deadline = time.monotonic() + 2.0
            while time.monotonic() < wait_deadline:
                kids = mod._direct_children_of(guard_pid)
                if kids:
                    return next(iter(kids))
                time.sleep(0.005)
            return None

        def _interrupt_after_popen(worker_id, proc, sock, tmpdir, log_path):
            captured["pid"] = proc.pid
            captured["exec_pid"] = _wait_exec_child(proc.pid)
            captured["sock"] = sock
            captured["tmpdir"] = tmpdir
            raise SystemExit(143)

        registry: dict[int, object] = {}
        with patch.object(mod, "_WorkerHandle", side_effect=_interrupt_after_popen):
            with self.assertRaises(SystemExit) as ctx:
                mod._spawn_worker(1, REPO_ROOT / "tests", None, registry=registry)
        self.assertEqual(ctx.exception.code, 143)
        self.assertEqual(registry, {})
        tmpdir = captured["tmpdir"]
        assert isinstance(tmpdir, Path)
        self.assertFalse(tmpdir.exists(), f"启动中断泄漏临时目录: {tmpdir}")
        sock = captured["sock"]
        self.assertEqual(sock.fileno(), -1)
        if sys.platform == "linux":
            self.assertIsNotNone(captured["exec_pid"], "Worker 守护进程未按预期派生执行器子进程")
        for check_pid in (captured["pid"], captured["exec_pid"]):
            if check_pid is None:
                continue
            with self.assertRaises(ProcessLookupError):
                os.kill(int(check_pid), 0)

        # 进一步验证 Popen 构造函数内部在 W_guard 已 fork 出 W_exec 后、返回赋值给 proc 前遭遇中断时亦无二次收养僵尸残留
        real_popen = subprocess.Popen
        popen_captured: dict[str, object] = {}
        self.addCleanup(_fallback_box_cleanup, popen_captured)

        class _InterruptingPopen(real_popen):  # type: ignore[misc]
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                popen_captured["pid"] = self.pid
                popen_captured["exec_pid"] = _wait_exec_child(self.pid) if sys.platform == "linux" else None
                popen_captured["tmpdir"] = Path(kwargs["env"]["TMPDIR"])
                self.returncode = -9
                raise KeyboardInterrupt("interrupted before Popen returned")

        with patch.object(mod.subprocess, "Popen", _InterruptingPopen):
            with self.assertRaises(KeyboardInterrupt):
                mod._spawn_worker(2, REPO_ROOT / "tests", None, registry=registry)
        self.assertEqual(registry, {})
        popen_tmpdir = popen_captured["tmpdir"]
        assert isinstance(popen_tmpdir, Path)
        self.assertFalse(popen_tmpdir.exists(), f"Popen 构造期中断泄漏临时目录: {popen_tmpdir}")
        if sys.platform == "linux":
            self.assertIsNotNone(popen_captured["exec_pid"], "Worker 守护进程未按预期派生执行器子进程")
        for check_pid in (popen_captured["pid"], popen_captured["exec_pid"]):
            if check_pid is None:
                continue
            with self.assertRaises(ProcessLookupError):
                os.kill(int(check_pid), 0)

        # 验证 Worker 在发出 ready 后立即 EOF 退出或派发阶段抛 BrokenPipeError 时，不抛未捕获异常且待跑分片由替补 Worker 完整执行
        eof_dir = Path(tempfile.mkdtemp(prefix="ni_rua_ready_eof_"))
        self.addCleanup(shutil.rmtree, eof_dir, True)
        for u_idx in (1, 2):
            mod_name = f"test_ready_eof_{u_idx}"
            self.addCleanup(sys.modules.pop, mod_name, None)
            (eof_dir / f"{mod_name}.py").write_text(
                f"import unittest\nclass U{u_idx}(unittest.TestCase):\n    def test_ok(self): self.assertTrue(True)\n",
                encoding="utf-8",
            )
        saved_sys_path = sys.path[:]
        self.addCleanup(lambda: setattr(sys, "path", saved_sys_path))

        real_spawn = mod._spawn_worker

        # 场景 1: 首个 Worker 写出 ready 帧后立即 os._exit(91)（触发 ready+EOF 同批到达）
        ready_eof_box: dict[str, object] = {}
        self.addCleanup(_fallback_box_cleanup, ready_eof_box)
        popen_calls = 0

        def _first_worker_exits_after_ready(cmd, *args, **kwargs):
            nonlocal popen_calls
            popen_calls += 1
            if popen_calls == 1:
                cmd = [
                    sys.executable,
                    "-c",
                    'import os; fd = int(os.environ["NI_RUA_TEST_WORKER_FD"]); '
                    'os.write(fd, b\'{"event":"ready"}\\n\'); os._exit(91)',
                ]
            p = real_popen(cmd, *args, **kwargs)
            if popen_calls == 1:
                ready_eof_box["pid"] = p.pid
                ready_eof_box["tmpdir"] = Path(kwargs["env"]["TMPDIR"])
            return p

        out_eof = io.StringIO()
        with patch.object(mod.subprocess, "Popen", side_effect=_first_worker_exits_after_ready):
            with redirect_stdout(out_eof):
                rc_eof = mod.main(["-s", str(eof_dir), "-j", "1", "-q"])
        text_eof = out_eof.getvalue()
        self.assertEqual(rc_eof, 1, text_eof)
        self.assertIn("parallel_runner.worker_bootstrap", text_eof)
        self.assertIn("returncode=91", text_eof)
        self.assertIn("Ran 2 tests", text_eof)
        self.assertNotIn("incomplete_schedule", text_eof)
        self.assertFalse(Path(ready_eof_box["tmpdir"]).exists())  # type: ignore[arg-type]

        # 场景 2: 首个 Worker 正常 ready 后在派发首条 run 指令时触发 BrokenPipeError（验证 pending 回退与替补重派）
        bpipe_box: dict[str, object] = {}
        self.addCleanup(_fallback_box_cleanup, bpipe_box)
        spawn_calls = 0

        def _first_handle_broken_pipe(worker_id, s_dir, t_dir, registry=None):
            nonlocal spawn_calls
            spawn_calls += 1
            h = real_spawn(worker_id, s_dir, t_dir, registry=registry)
            if spawn_calls == 1:
                bpipe_box["pid"] = h.process.pid
                bpipe_box["tmpdir"] = h.tmpdir
                orig_send = h.send

                def _fail_on_run(payload):
                    if payload.get("op") == "run":
                        mod._signal_worker_group(h.process, signal.SIGKILL)
                        raise BrokenPipeError("simulated dispatch broken pipe")
                    return orig_send(payload)

                h.send = _fail_on_run  # type: ignore[method-assign]
            return h

        out_bpipe = io.StringIO()
        with patch.object(mod, "_spawn_worker", side_effect=_first_handle_broken_pipe):
            with redirect_stdout(out_bpipe):
                rc_bpipe = mod.main(["-s", str(eof_dir), "-j", "1", "-q"])
        text_bpipe = out_bpipe.getvalue()
        self.assertEqual(rc_bpipe, 1, text_bpipe)
        self.assertIn("parallel_runner.worker_bootstrap", text_bpipe)
        self.assertIn("Ran 2 tests", text_bpipe)
        self.assertNotIn("incomplete_schedule", text_bpipe)
        self.assertFalse(Path(bpipe_box["tmpdir"]).exists())  # type: ignore[arg-type]


class ZonedSelectionAndDagTests(unittest.TestCase):
    """验证 --zone/--subzone/--tier/--changed/--changed-scope/--explain 与 .ni-rua-test-timings.json。"""

    @staticmethod
    def _write_tree(root: Path, files: dict[str, str]) -> None:
        for rel, body in files.items():
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")

    @staticmethod
    def _git(repo: Path, *args: str) -> None:
        subprocess.run(
            ["git", *args],
            cwd=str(repo),
            check=True,
            capture_output=True,
            text=True,
        )

    def _run(self, repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(RUNNER_SCRIPT), "-s", str(repo / "tests"), *args],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_zone_subzone_tier_explain_and_nested_zone_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as root_str:
            # 外层带 `.git` 的祖先目录（TMPDIR 在别的检出内、残留的 `/tmp/.git`）不得越过就近的 `ni_rua` 项目树。
            (Path(root_str) / ".git").mkdir()
            repo = Path(root_str) / "proj"
            self._write_tree(
                repo,
                {
                    "ni_rua/__init__.py": "",
                    "ni_rua/apps/__init__.py": "",
                    "ni_rua/apps/gateway/__init__.py": "",
                    "ni_rua/apps/gateway/server.py": "PORT = 1\n",
                    "tests/core/config/test_ns_config.py": """
                        import unittest
                        class NsConfigTest(unittest.TestCase):
                            def test_ok(self): self.assertTrue(True)
                    """,
                    "tests/robot/so101/__init__.py": "",
                    "tests/robot/so101/test_so101_ns.py": """
                        import unittest
                        NI_RUA_TEST_TIER = "component"
                        class So101NsTest(unittest.TestCase):
                            def test_ok(self): self.assertTrue(True)
                    """,
                    "tests/test_flat_http.py": """
                        import unittest
                        from http.server import HTTPServer
                        from ni_rua.apps.gateway.server import PORT
                        def _dummy(): return HTTPServer(("127.0.0.1", 0), None)
                        class FlatHttpTest(unittest.TestCase):
                            def test_ok(self): self.assertEqual(PORT, 1)
                    """,
                    "tests/test_codex_sandbox.py": """
                        import unittest
                        class SandboxPkgTest(unittest.TestCase):
                            def test_ok(self): self.assertTrue(True)
                    """,
                },
            )

            # Wave 2 嵌套目录（含无 __init__.py 命名空间包）在全量发现下必须全部运行
            full = self._run(repo, "-j", "2")
            self.assertEqual(full.returncode, 0, full.stdout + full.stderr)
            self.assertIn("Ran 4 tests", full.stdout)

            # --explain --zone / --subzone / --tier 不执行测试且准确筛选
            z_robot = self._run(repo, "--explain", "--zone", "robot")
            self.assertEqual(z_robot.returncode, 0, z_robot.stdout + z_robot.stderr)
            self.assertIn("robot.so101.test_so101_ns [robot.so101 | component]", z_robot.stdout)
            self.assertNotIn("test_ns_config", z_robot.stdout)
            self.assertNotIn("test_flat_http", z_robot.stdout)

            sz = self._run(repo, "--explain", "--subzone", "robot.so101")
            self.assertEqual(sz.returncode, 0)
            self.assertIn("选中 Python 测试模块 (1)", sz.stdout)
            self.assertIn("robot.so101.test_so101_ns", sz.stdout)

            t_fast = self._run(repo, "--explain", "--tier", "fast")
            self.assertEqual(t_fast.returncode, 0)
            self.assertIn("core.config.test_ns_config [core.config | fast]", t_fast.stdout)
            self.assertNotIn("test_so101_ns", t_fast.stdout)
            self.assertNotIn("test_flat_http", t_fast.stdout)
            self.assertNotIn("test_codex_sandbox", t_fast.stdout)

            t_pkg = self._run(repo, "--explain", "--zone", "packaging")
            self.assertEqual(t_pkg.returncode, 0)
            self.assertIn("test_codex_sandbox [packaging.sandbox | packaging]", t_pkg.stdout)

            run_core = self._run(repo, "--zone", "core", "--tier", "fast", "-j", "1")
            self.assertEqual(run_core.returncode, 0, run_core.stdout + run_core.stderr)
            self.assertIn("Ran 1 tests", run_core.stdout)

            bad_zone = self._run(repo, "--zone", "unknown_zone")
            self.assertNotEqual(bad_zone.returncode, 0)

    def test_changed_hub_bounded_dag_barrel_guard_docs_js_and_timings(self) -> None:
        with tempfile.TemporaryDirectory() as root_str:
            repo = Path(root_str)
            self._write_tree(
                repo,
                {
                    ".gitignore": ".ni-rua-test-timings.json\n.ni-rua-test-timings.json.tmp\n",
                    "docs/README.md": "# docs\n",
                    "web/check.mjs": "console.log('web ok');\n",
                    "agent-plugin/package.json": '{"name":"p","scripts":{"test":"node -e \\"console.log(1)\\""}}\n',
                    "ni_rua/__init__.py": "",
                    "ni_rua/core/__init__.py": "from .leaf import val\nfrom .unrelated import other\n",
                    "ni_rua/core/leaf.py": "val = 1\n",
                    "ni_rua/core/unrelated.py": "other = 2\n",
                    "ni_rua/core/mid.py": "from .leaf import val\nmid_val = val + 1\n",
                    "ni_rua/runtime/__init__.py": "",
                    "ni_rua/runtime/harness.py": "from ni_rua.core.mid import mid_val\nclass Hub:\n    v = mid_val\n",
                    "ni_rua/apps/__init__.py": "",
                    "ni_rua/apps/downstream.py": "from ni_rua.core.mid import mid_val\ndown = mid_val + 1\n",
                    "ni_rua/apps/far_consumer.py": "from ni_rua.apps.downstream import down\nfar = down + 1\n",
                    "ni_rua/apps/hub_only.py": "from ni_rua.runtime.harness import Hub\nh = Hub.v\n",
                    "ni_rua/apps/gateway/__init__.py": "",
                    "ni_rua/apps/gateway/workbench.py": "from ni_rua.runtime.harness import Hub\nclass WB:\n    w = Hub.v\n",
                    "tests/support/__init__.py": "",
                    "tests/support/helper.py": "from ni_rua.core.leaf import val\nhelper_val = val\n",
                    "tests/test_leaf.py": """
                        import unittest
                        from ni_rua.core.leaf import val
                        class T(unittest.TestCase):
                            def test_ok(self): self.assertEqual(val, 1)
                    """,
                    "tests/test_mid_fast.py": """
                        import unittest
                        from ni_rua.core.mid import mid_val
                        class T(unittest.TestCase):
                            def test_ok(self): self.assertEqual(mid_val, 2)
                    """,
                    "tests/test_mid_comp.py": """
                        import unittest
                        from ni_rua.core.mid import mid_val
                        def Store(): return None
                        class T(unittest.TestCase):
                            def test_ok(self): self.assertEqual(mid_val, 2)
                    """,
                    "tests/test_downstream_fast.py": """
                        import unittest
                        from ni_rua.apps.downstream import down
                        class T(unittest.TestCase):
                            def test_ok(self): self.assertEqual(down, 3)
                    """,
                    "tests/test_downstream_comp.py": """
                        import unittest
                        from http.server import HTTPServer
                        from ni_rua.apps.downstream import down
                        def _ref(): return HTTPServer.__name__ if False else HTTPServer(("127.0.0.1", 0), None)
                        class T(unittest.TestCase):
                            def test_ok(self): self.assertEqual(down, 3)
                    """,
                    "tests/test_far_fast.py": """
                        import unittest
                        from ni_rua.apps.far_consumer import far
                        class T(unittest.TestCase):
                            def test_ok(self): self.assertEqual(far, 4)
                    """,
                    "tests/test_far_comp.py": """
                        import unittest
                        from http.server import HTTPServer
                        from ni_rua.apps.far_consumer import far
                        def _ref(): return HTTPServer.__name__ if False else HTTPServer(("127.0.0.1", 0), None)
                        class T(unittest.TestCase):
                            def test_ok(self): self.assertEqual(far, 4)
                    """,
                    "tests/test_harness_direct.py": """
                        import unittest
                        from ni_rua.runtime.harness import Hub
                        class T(unittest.TestCase):
                            def test_ok(self): self.assertEqual(Hub.v, 2)
                    """,
                    "tests/test_workbench_direct.py": """
                        import unittest
                        from ni_rua.apps.gateway.workbench import WB
                        class T(unittest.TestCase):
                            def test_ok(self): self.assertEqual(WB.w, 2)
                    """,
                    "tests/test_hub_only.py": """
                        import unittest
                        from ni_rua.apps.hub_only import h
                        class T(unittest.TestCase):
                            def test_ok(self): self.assertEqual(h, 2)
                    """,
                    "tests/test_unrelated.py": """
                        import unittest
                        from ni_rua.core.unrelated import other
                        class T(unittest.TestCase):
                            def test_ok(self): self.assertEqual(other, 2)
                    """,
                    "tests/test_via_support.py": """
                        import unittest
                        from support.helper import helper_val
                        class T(unittest.TestCase):
                            def test_ok(self): self.assertEqual(helper_val, 1)
                    """,
                    "tests/test_gateway_console.py": """
                        import unittest
                        class T(unittest.TestCase):
                            def test_ok(self): self.assertTrue(True)
                    """,
                    "tests/test_agent_plugin.py": """
                        import unittest
                        class T(unittest.TestCase):
                            def test_ok(self): self.assertTrue(True)
                    """,
                },
            )
            self._git(repo, "init")
            self._git(repo, "config", "user.email", "test@example.com")
            self._git(repo, "config", "user.name", "Test")
            self._git(repo, "add", ".")
            self._git(repo, "commit", "-m", "init")

            # 1. docs/** 仅变更：默认返回 0，--strict-empty 返回 5
            (repo / "docs" / "README.md").write_text("# docs updated\n", encoding="utf-8")
            doc_only = self._run(repo, "--changed")
            self.assertEqual(doc_only.returncode, 0, doc_only.stdout + doc_only.stderr)
            self.assertIn("NO TESTS RAN", doc_only.stdout)
            doc_strict = self._run(repo, "--changed", "--strict-empty")
            self.assertEqual(doc_strict.returncode, 5)
            (repo / "docs" / "README.md").write_text("# docs\n", encoding="utf-8")

            # 2. 修改 leaf.py 验证 direct / auto / full 三档与 Barrel Guard + Hub Damper
            (repo / "ni_rua" / "core" / "leaf.py").write_text("val = 1  # edited\n", encoding="utf-8")

            exp_direct = self._run(repo, "--changed", "--changed-scope", "direct", "--explain")
            self.assertEqual(exp_direct.returncode, 0, exp_direct.stdout + exp_direct.stderr)
            self.assertIn("test_leaf", exp_direct.stdout)
            self.assertIn("test_via_support", exp_direct.stdout)
            self.assertNotIn("test_mid_fast", exp_direct.stdout)

            exp_auto = self._run(repo, "--changed", "--changed-scope", "auto", "--explain")
            self.assertEqual(exp_auto.returncode, 0, exp_auto.stdout + exp_auto.stderr)
            self.assertIn("test_leaf", exp_auto.stdout)
            self.assertIn("test_via_support", exp_auto.stdout)
            self.assertIn("test_mid_fast", exp_auto.stdout)
            self.assertIn("test_mid_comp", exp_auto.stdout)
            self.assertIn("test_downstream_fast", exp_auto.stdout)
            self.assertIn("test_downstream_comp", exp_auto.stdout)
            self.assertIn("test_harness_direct", exp_auto.stdout)
            self.assertIn("test_far_fast", exp_auto.stdout)
            self.assertNotIn("test_far_comp", exp_auto.stdout)
            self.assertNotIn("test_hub_only", exp_auto.stdout)
            self.assertNotIn("test_workbench_direct", exp_auto.stdout)
            self.assertNotIn("test_unrelated", exp_auto.stdout)

            # 显式传入 --tier component 时，绝不因 depth>=3 阻尼丢弃匹配的 component 测试
            exp_tier_comp = self._run(
                repo, "--changed", "--changed-scope", "auto", "--tier", "component", "--explain"
            )
            self.assertEqual(exp_tier_comp.returncode, 0, exp_tier_comp.stdout + exp_tier_comp.stderr)
            self.assertIn("test_mid_comp", exp_tier_comp.stdout)
            self.assertIn("test_downstream_comp", exp_tier_comp.stdout)
            self.assertIn("test_far_comp", exp_tier_comp.stdout)

            exp_full = self._run(repo, "--changed", "--changed-scope", "full", "--explain")
            self.assertEqual(exp_full.returncode, 0, exp_full.stdout + exp_full.stderr)
            self.assertIn("test_downstream_comp", exp_full.stdout)
            self.assertIn("test_far_comp", exp_full.stdout)
            self.assertIn("test_hub_only", exp_full.stdout)
            self.assertIn("test_workbench_direct", exp_full.stdout)

            # 实际执行 --changed 并验证 .ni-rua-test-timings.json 持久化
            run_auto = self._run(repo, "--changed", "-j", "2")
            self.assertEqual(run_auto.returncode, 0, run_auto.stdout + run_auto.stderr)
            self.assertIn("Ran 8 tests", run_auto.stdout)
            timings_file = repo / ".ni-rua-test-timings.json"
            self.assertTrue(timings_file.is_file())
            timings_data = json.loads(timings_file.read_text(encoding="utf-8"))
            self.assertIn("test_leaf", timings_data["modules"])
            self.assertIn("test_mid_fast", timings_data["modules"])

            # 3. Hub 自身被修改时允许展开其下游测试及下游 Hub 的 1-hop 测试
            (repo / "ni_rua" / "core" / "leaf.py").write_text("val = 1\n", encoding="utf-8")
            (repo / "ni_rua" / "runtime" / "harness.py").write_text(
                "from ni_rua.core.mid import mid_val\nclass Hub:\n    v = mid_val\n# edited\n",
                encoding="utf-8",
            )
            exp_hub = self._run(repo, "--changed", "--explain")
            self.assertEqual(exp_hub.returncode, 0)
            self.assertIn("test_harness_direct", exp_hub.stdout)
            self.assertIn("test_hub_only", exp_hub.stdout)
            self.assertIn("test_workbench_direct", exp_hub.stdout)

            # 4. web/** 与 agent-plugin/**（含 SKILL.md）变更映射到 JS 检查 + 对应 Python 测试
            (repo / "ni_rua" / "runtime" / "harness.py").write_text(
                "from ni_rua.core.mid import mid_val\nclass Hub:\n    v = mid_val\n",
                encoding="utf-8",
            )
            (repo / "web" / "app.js").write_text("export const x = 1;\n", encoding="utf-8")
            (repo / "agent-plugin" / "skills" / "ni-rua-robot").mkdir(parents=True)
            (repo / "agent-plugin" / "skills" / "ni-rua-robot" / "SKILL.md").write_text("# skill\n", encoding="utf-8")
            exp_js = self._run(repo, "--changed", "--explain")
            self.assertEqual(exp_js.returncode, 0)
            self.assertIn("node web/check.mjs", exp_js.stdout)
            self.assertIn("npm --prefix agent-plugin test", exp_js.stdout)
            self.assertIn("test_gateway_console", exp_js.stdout)
            self.assertIn("test_agent_plugin", exp_js.stdout)

    def test_lazy_barrels_deletes_renames_test_chains_assets_selector_targets_and_timings_safety(self) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location("_rt_timings_check", RUNNER_SCRIPT)
        assert spec is not None and spec.loader is not None
        rt_mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = rt_mod
        self.addCleanup(sys.modules.pop, spec.name, None)
        spec.loader.exec_module(rt_mod)

        with tempfile.TemporaryDirectory() as root_str:
            repo = Path(root_str)
            self._write_tree(
                repo,
                {
                    ".gitignore": ".ni-rua-test-timings.json\n.ni-rua-test-timings.*.tmp\n",
                    "pyproject.toml": '[project]\nname = "ni_rua"\n',
                    "configs/ni-rua.json": '{"sandbox_enabled": true}\n',
                    "scripts/check_package.py": "def check(): return True\n",
                    "scripts/custom_tool.py": "VAL = 9\n",
                    "services/curobo_ik/build_model.py": "MODEL = 'x5'\n",
                    "web/check.mjs": "console.log('ok');\n",
                    "web/fixture-calibration-server.py": "PORT = 8080\n",
                    "ni_rua/__init__.py": "PKG_TAG = 'v1'\n",
                    "ni_rua/core/__init__.py": (
                        "from .leaf import raw_val as aliased_val\n"
                        "from .unrelated import un_val\n"
                        "COMPUTED = aliased_val + 1\n"
                    ),
                    "ni_rua/core/leaf.py": "raw_val = 10\n",
                    "ni_rua/core/unrelated.py": "un_val = 20\n",
                    "ni_rua/core/shim.py": "SHIM = 1\n",
                    "ni_rua/core/consumer.py": "from . import shim\nC = shim.SHIM\n",
                    "ni_rua/star/__init__.py": "from .star_leaf import *\n",
                    "ni_rua/star/star_leaf.py": "def star_fn(): return 7\n",
                    "ni_rua/oldpkg/__init__.py": "OLD = 1\n",
                    "ni_rua/robot/__init__.py": "",
                    "ni_rua/robot/info.py": "INFO = 1\n",
                    "ni_rua/robot/data/briefing.txt": "robot briefing\n",
                    "ni_rua/robot/data/arm.urdf": "<robot/>\n",
                    "ni_rua/cortex/__init__.py": "",
                    "ni_rua/cortex/agents/__init__.py": "",
                    "ni_rua/cortex/agents/codex/__init__.py": """
                        from importlib import import_module
                        _SUBMODULES = frozenset({"bridge", "session"})
                        _EXPORTS = {
                            "CodexSession": ".session",
                            "BridgeTuple": (".bridge", "BridgeCls"),
                        }
                        def __getattr__(name):
                            if name in _SUBMODULES:
                                return import_module(f".{name}", __name__)
                            mod_name = _EXPORTS.get(name)
                            if mod_name is None:
                                raise AttributeError(name)
                            if isinstance(mod_name, tuple):
                                return getattr(import_module(mod_name[0], __name__), mod_name[1])
                            return getattr(import_module(mod_name, __name__), name)
                    """,
                    "ni_rua/cortex/agents/codex/session.py": "class CodexSession: pass\n",
                    "ni_rua/cortex/agents/codex/bridge.py": "class BridgeCls: pass\n",
                    "ni_rua/apps/__init__.py": "",
                    "ni_rua/apps/evaluation/__init__.py": "",
                    "ni_rua/apps/evaluation/codex_fork.py": "def slice_rollout_jsonl(): return []\n",
                    "ni_rua/apps/evaluation/checkpoint.py": """
                        _CODEX_COMPAT_EXPORTS = frozenset({"slice_rollout_jsonl"})
                        def __getattr__(name: str):
                            if name in _CODEX_COMPAT_EXPORTS:
                                from . import codex_fork
                                return getattr(codex_fork, name)
                            raise AttributeError(name)
                    """,
                    "tests/support/__init__.py": "",
                    "tests/support/sup_a.py": "from ni_rua.core.leaf import raw_val\nSUP_A = raw_val\n",
                    "tests/support/sup_b.py": "from support.sup_a import SUP_A\nSUP_B = SUP_A + 1\n",
                    "tests/test_aliased.py": """
                        import unittest
                        from ni_rua.core import aliased_val
                        class AliasedTests(unittest.TestCase):
                            def test_alias(self): self.assertEqual(aliased_val, 10)
                    """,
                    "tests/test_bare_pkg.py": """
                        import unittest
                        from ni_rua.core import COMPUTED
                        class BarePkgTests(unittest.TestCase):
                            def test_computed(self): self.assertEqual(COMPUTED, 11)
                    """,
                    "tests/test_unrelated_pkg.py": """
                        import unittest
                        from ni_rua.core import un_val
                        class UnrelatedPkgTests(unittest.TestCase):
                            def test_un(self): self.assertEqual(un_val, 20)
                    """,
                    "tests/test_star.py": """
                        import unittest
                        from ni_rua.star import star_fn
                        class StarTests(unittest.TestCase):
                            def test_star(self): self.assertEqual(star_fn(), 7)
                    """,
                    "tests/test_codex_lazy.py": """
                        import unittest
                        from ni_rua.cortex.agents.codex import CodexSession
                        class CodexLazyTests(unittest.TestCase):
                            def test_sess(self): self.assertIsNotNone(CodexSession)
                    """,
                    "tests/test_bridge_attr.py": """
                        import unittest
                        import ni_rua.cortex.agents.codex as cdx
                        class BridgeAttrTests(unittest.TestCase):
                            def test_bridge(self): self.assertIsNotNone(cdx.BridgeTuple)
                    """,
                    "tests/test_ckpt_lazy.py": """
                        import unittest
                        from ni_rua.apps.evaluation.checkpoint import slice_rollout_jsonl
                        class CkptLazyTests(unittest.TestCase):
                            def test_slice(self): self.assertEqual(slice_rollout_jsonl(), [])
                    """,
                    "tests/test_nested_imp.py": """
                        import unittest
                        class NestedImpTests(unittest.TestCase):
                            def test_conditional_import(self):
                                if True:
                                    from ni_rua.core.leaf import raw_val
                                    self.assertEqual(raw_val, 10)
                    """,
                    "tests/test_fixture_base.py": """
                        import unittest
                        from support.sup_b import SUP_B
                        SHARED_FIX = SUP_B
                        class BaseFixTests(unittest.TestCase):
                            def test_base(self): self.assertEqual(SHARED_FIX, 11)
                    """,
                    "tests/test_fixture_consumer.py": """
                        import unittest
                        import test_fixture_base
                        class ConsumerFixTests(unittest.TestCase):
                            def test_consumer(self): self.assertEqual(test_fixture_base.SHARED_FIX, 11)
                    """,
                    "tests/test_consumer_shim.py": """
                        import unittest
                        from ni_rua.core.consumer import C
                        class ShimConsumerTests(unittest.TestCase):
                            def test_c(self): self.assertEqual(C, 1)
                    """,
                    "tests/test_oldpkg_user.py": """
                        import unittest
                        import ni_rua.oldpkg
                        class OldPkgTests(unittest.TestCase):
                            def test_old(self): self.assertEqual(ni_rua.oldpkg.OLD, 1)
                    """,
                    "tests/test_pkg_priority.py": """
                        import unittest
                        from scripts.check_package import check
                        from ni_rua.core.leaf import raw_val
                        class PkgPriorityTests(unittest.TestCase):
                            def test_pkg_zone(self): self.assertTrue(check() and raw_val == 10)
                    """,
                    "tests/test_script_refs.py": """
                        import unittest
                        REFS = ("custom_tool.py", "build_model.py", "fixture-calibration-server.py")
                        class ScriptRefTests(unittest.TestCase):
                            def test_refs(self): self.assertEqual(len(REFS), 3)
                    """,
                    "tests/test_robot_asset.py": """
                        import unittest
                        from ni_rua.robot.info import INFO
                        class RobotAssetTests(unittest.TestCase):
                            def test_info(self): self.assertEqual(INFO, 1)
                    """,
                    "tests/test_config.py": """
                        import unittest
                        class ConfigTests(unittest.TestCase):
                            def test_cfg_a(self): self.assertTrue(True)
                            def test_cfg_b(self): self.assertTrue(True)
                    """,
                    "tests/test_parallel_runner.py": """
                        import unittest
                        class RunnerSelfTests(unittest.TestCase):
                            def test_self(self): self.assertTrue(True)
                    """,
                },
            )
            self._git(repo, "init")
            self._git(repo, "config", "user.email", "test@example.com")
            self._git(repo, "config", "user.name", "Test")
            self._git(repo, "add", ".")
            self._git(repo, "commit", "-m", "init")

            # Item 9: _LAYER_PRIORITY['packaging'] 不抛 KeyError 且优先级高于 core
            exp_pkg_prio = self._run(repo, "--explain", "tests.test_pkg_priority")
            self.assertEqual(exp_pkg_prio.returncode, 0, exp_pkg_prio.stdout + exp_pkg_prio.stderr)
            self.assertIn("test_pkg_priority [packaging.wheel | packaging]", exp_pkg_prio.stdout)

            # Item 8: Selector 模式支持目录、tests.*、类/方法后缀及 --explain -k
            exp_dir = self._run(repo, "--explain", "tests")
            self.assertEqual(exp_dir.returncode, 0)
            self.assertIn("test_config", exp_dir.stdout)

            exp_cls = self._run(repo, "--explain", "test_config.ConfigTests")
            self.assertEqual(exp_cls.returncode, 0)
            self.assertIn("选中 Python 测试模块 (1)", exp_cls.stdout)

            exp_bad_cls = self._run(repo, "--explain", "test_config.MissingClass")
            self.assertEqual(exp_bad_cls.returncode, 5)

            exp_k_hit = self._run(repo, "--explain", "-k", "test_cfg_a")
            self.assertEqual(exp_k_hit.returncode, 0)
            self.assertIn("选中 Python 测试模块 (1)", exp_k_hit.stdout)
            self.assertIn("test_config", exp_k_hit.stdout)

            exp_k_miss = self._run(repo, "--explain", "-k", "__no_such_method__")
            self.assertEqual(exp_k_miss.returncode, 5)

            # Selector 执行模式保留类/方法后缀，只运行 1 个方法且不覆盖完整模块耗时缓存
            timings_path = repo / ".ni-rua-test-timings.json"
            timings_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "modules": {
                            "test_config": {"elapsed_s": 42.5, "tests": 2},
                            "bad_bool": {"elapsed_s": True, "tests": 1},
                            "bad_neg": -3.0,
                            "bad_huge": 10**400,
                            "bad_over": 1e9,
                        },
                    }
                ),
                encoding="utf-8",
            )
            loaded_t = rt_mod._load_module_timings(repo, repo / "tests")
            self.assertEqual(loaded_t, {"test_config": 42.5})

            single_sel = self._run(repo, "--zone", "core", "test_config.ConfigTests.test_cfg_a", "-j", "1")
            self.assertEqual(single_sel.returncode, 0, single_sel.stdout + single_sel.stderr)
            self.assertIn("Ran 1 tests", single_sel.stdout)
            self.assertEqual(rt_mod._load_module_timings(repo, repo / "tests")["test_config"], 42.5)

            single_plain = self._run(repo, "test_config.ConfigTests.test_cfg_a", "-j", "1")
            self.assertEqual(single_plain.returncode, 0, single_plain.stdout + single_plain.stderr)
            self.assertIn("Ran 1 tests", single_plain.stdout)
            self.assertEqual(rt_mod._load_module_timings(repo, repo / "tests")["test_config"], 42.5)

            full_cfg = self._run(repo, "tests/test_config.py", "-j", "1")
            self.assertEqual(full_cfg.returncode, 0, full_cfg.stdout + full_cfg.stderr)
            self.assertIn("Ran 2 tests", full_cfg.stdout)
            self.assertLess(rt_mod._load_module_timings(repo, repo / "tests")["test_config"], 10.0)
            self.assertEqual(list(repo.glob(".ni-rua-test-timings.*.tmp")), [])

            # Item 3 + 6 + 7: 修改 leaf.py -> 覆盖 as 别名重导出、裸包导出、条件导入、多跳 support 与测试间夹具链，且不误选 unrelated
            (repo / "ni_rua" / "core" / "leaf.py").write_text("raw_val = 10  # changed\n", encoding="utf-8")
            exp_leaf = self._run(repo, "--changed", "--explain")
            self.assertEqual(exp_leaf.returncode, 0)
            for expected_mod in (
                "test_aliased",
                "test_bare_pkg",
                "test_nested_imp",
                "test_fixture_base",
                "test_fixture_consumer",
                "test_pkg_priority",
            ):
                self.assertIn(expected_mod, exp_leaf.stdout)
            self.assertNotIn("test_unrelated_pkg", exp_leaf.stdout)
            self._git(repo, "checkout", "--", "ni_rua/core/leaf.py")

            # Item 3: 惰性 __getattr__ 桶 (_EXPORTS / _CODEX_COMPAT_EXPORTS / * 导出 / import pkg; pkg.sym / 父包 __init__.py)
            (repo / "ni_rua" / "cortex" / "agents" / "codex" / "session.py").write_text(
                "class CodexSession:\n    x = 1\n", encoding="utf-8"
            )
            exp_cdx = self._run(repo, "--changed", "--explain")
            self.assertIn("test_codex_lazy", exp_cdx.stdout)
            self.assertNotIn("test_bridge_attr", exp_cdx.stdout)
            self._git(repo, "checkout", "--", "ni_rua/cortex/agents/codex/session.py")

            (repo / "ni_rua" / "cortex" / "agents" / "codex" / "bridge.py").write_text(
                "class BridgeCls:\n    y = 2\n", encoding="utf-8"
            )
            exp_br = self._run(repo, "--changed", "--explain")
            self.assertIn("test_bridge_attr", exp_br.stdout)
            self.assertNotIn("test_codex_lazy", exp_br.stdout)
            self._git(repo, "checkout", "--", "ni_rua/cortex/agents/codex/bridge.py")

            (repo / "ni_rua" / "apps" / "evaluation" / "codex_fork.py").write_text(
                "def slice_rollout_jsonl(): return [1]\n", encoding="utf-8"
            )
            exp_fk = self._run(repo, "--changed", "--explain")
            self.assertIn("test_ckpt_lazy", exp_fk.stdout)
            self._git(repo, "checkout", "--", "ni_rua/apps/evaluation/codex_fork.py")

            (repo / "ni_rua" / "star" / "star_leaf.py").write_text("def star_fn(): return 8\n", encoding="utf-8")
            exp_star = self._run(repo, "--changed", "--explain")
            self.assertIn("test_star", exp_star.stdout)
            self._git(repo, "checkout", "--", "ni_rua/star/star_leaf.py")

            # 修改父包 __init__.py 必须选中导入其子叶模块的测试
            (repo / "ni_rua" / "__init__.py").write_text("PKG_TAG = 'v2'\n", encoding="utf-8")
            exp_parent_init = self._run(repo, "--changed", "--explain")
            self.assertIn("test_aliased", exp_parent_init.stdout)
            self.assertIn("test_consumer_shim", exp_parent_init.stdout)
            self._git(repo, "checkout", "--", "ni_rua/__init__.py")

            # Item 6: 直接修改被其他测试作为夹具导入的测试文件，必须传递选中下游测试
            (repo / "tests" / "test_fixture_base.py").write_text(
                (repo / "tests" / "test_fixture_base.py").read_text(encoding="utf-8") + "# edited\n",
                encoding="utf-8",
            )
            exp_t2t = self._run(repo, "--changed", "--explain")
            self.assertIn("test_fixture_base", exp_t2t.stdout)
            self.assertIn("test_fixture_consumer", exp_t2t.stdout)
            self._git(repo, "checkout", "--", "tests/test_fixture_base.py")

            # Item 4 & 7: configs/*.json、pyproject.toml、ni_rua/** 非 .py 资源（含 .txt/.urdf）、scripts/services/web/*.py
            (repo / "configs" / "ni-rua.json").write_text('{"sandbox_enabled": false}\n', encoding="utf-8")
            exp_cfg_json = self._run(repo, "--changed", "--explain")
            self.assertIn("test_config", exp_cfg_json.stdout)
            self._git(repo, "checkout", "--", "configs/ni-rua.json")

            (repo / "pyproject.toml").write_text('[project]\nname = "ni_rua"\nversion = "0.2"\n', encoding="utf-8")
            exp_pyproj = self._run(repo, "--changed", "--explain")
            self.assertIn("test_pkg_priority", exp_pyproj.stdout)
            self.assertIn("test_config", exp_pyproj.stdout)
            self.assertIn("test_parallel_runner", exp_pyproj.stdout)
            self._git(repo, "checkout", "--", "pyproject.toml")

            (repo / "ni_rua" / "robot" / "data" / "briefing.txt").write_text("updated briefing\n", encoding="utf-8")
            exp_asset = self._run(repo, "--changed", "--explain")
            self.assertIn("test_robot_asset", exp_asset.stdout)
            self._git(repo, "checkout", "--", "ni_rua/robot/data/briefing.txt")

            for script_rel in (
                "scripts/custom_tool.py",
                "services/curobo_ik/build_model.py",
                "web/fixture-calibration-server.py",
            ):
                p_scr = repo / script_rel
                orig_scr = p_scr.read_text(encoding="utf-8")
                p_scr.write_text(orig_scr + "# changed\n", encoding="utf-8")
                exp_scr = self._run(repo, "--changed", "--explain")
                self.assertIn("test_script_refs", exp_scr.stdout, f"{script_rel} 未触发引用它的测试")
                p_scr.write_text(orig_scr, encoding="utf-8")

            # Item 5: 删除包 __init__.py、git mv 重命名源码模块、以及非 ASCII 未引用路径 (-z)
            (repo / "ni_rua" / "oldpkg" / "__init__.py").unlink()
            exp_del_pkg = self._run(repo, "--changed", "--explain")
            self.assertIn("test_oldpkg_user", exp_del_pkg.stdout)
            self._git(repo, "checkout", "--", "ni_rua/oldpkg/__init__.py")

            self._git(repo, "mv", "ni_rua/core/shim.py", "ni_rua/core/shim_v2.py")
            (repo / "docs").mkdir(exist_ok=True)
            (repo / "docs" / "中文说明.md").write_text("# 中文\n", encoding="utf-8")
            exp_ren = self._run(repo, "--changed", "--explain")
            self.assertEqual(exp_ren.returncode, 0, exp_ren.stdout + exp_ren.stderr)
            self.assertIn("docs/中文说明.md", exp_ren.stdout)
            self.assertIn("test_consumer_shim", exp_ren.stdout)

    def test_zoned_positional_targets_and_web_js_and_lockfile_changed_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "zoned_repo"
            repo.mkdir(parents=True, exist_ok=True)
            self._write_tree(
                repo,
                {
                    "pyproject.toml": '[project]\nname = "ni_rua"\nversion = "0.1"\n',
                    "requirements.lock.txt": "numpy==2.5.3\n",
                    "constraints-oss.txt": "cmeel==0.62.0\n",
                    "web/check.mjs": "console.log('ok');\n",
                    "web/agent-settings-model.js": "export const X = 1;\n",
                    "ni_rua/__init__.py": "",
                    "tests/__init__.py": "",
                    "tests/core/__init__.py": "",
                    "tests/core/config/__init__.py": "",
                    "tests/core/config/test_config.py": """
                        import unittest
                        NI_RUA_TEST_TIER = "fast"
                        class ConfigTests(unittest.TestCase):
                            def test_alpha(self): self.assertEqual(1, 1)
                            def test_beta(self): self.assertEqual(2, 2)
                    """,
                    # Note: tests/packaging/__init__.py intentionally omitted!
                    "tests/packaging/npm/__init__.py": "",
                    "tests/packaging/npm/test_agent_plugin_coexistence.py": """
                        import unittest
                        NI_RUA_TEST_TIER = "packaging"
                        class CoexistTests(unittest.TestCase):
                            def test_lock_ref(self):
                                _ = ("requirements.lock.txt", "constraints-oss.txt")
                                self.assertTrue(True)
                    """,
                    "tests/apps/__init__.py": "",
                    "tests/apps/gateway/__init__.py": "",
                    "tests/apps/gateway/test_gateway_agent_settings.py": """
                        import unittest
                        NI_RUA_TEST_TIER = "component"
                        class WebJsRefTests(unittest.TestCase):
                            def test_web_js(self):
                                _ = "web/agent-settings-model.js"
                                self.assertTrue(True)
                    """,
                },
            )
            self._git(repo, "init")
            self._git(repo, "config", "user.email", "test@example.com")
            self._git(repo, "config", "user.name", "Test")
            self._git(repo, "add", ".")
            self._git(repo, "commit", "-m", "init")

            # 1) Dotted package/zone/subzone targets & bare stems & bare stem.Class.method work in BOTH direct run and --explain!
            for spec, expected_Substring in (
                ("core", "Ran 2 tests"),
                ("core.config", "Ran 2 tests"),
                ("packaging", "Ran 1 test"),
                ("packaging.npm", "Ran 1 test"),
                ("test_config", "Ran 2 tests"),
                ("test_config.ConfigTests.test_alpha", "Ran 1 test"),
                ("test_agent_plugin_coexistence.CoexistTests.test_lock_ref", "Ran 1 test"),
            ):
                with self.subTest(spec=spec):
                    res_run = self._run(repo, spec)
                    self.assertEqual(res_run.returncode, 0, res_run.stdout + res_run.stderr)
                    self.assertIn(expected_Substring, res_run.stdout + res_run.stderr)
                    res_exp = self._run(repo, "--explain", spec)
                    self.assertEqual(res_exp.returncode, 0, res_exp.stdout + res_exp.stderr)

            # 2) web/*.js edit triggers Python tests referencing that JS asset!
            (repo / "web" / "agent-settings-model.js").write_text("export const X = 2;\n", encoding="utf-8")
            exp_web = self._run(repo, "--changed", "--explain")
            self.assertEqual(exp_web.returncode, 0, exp_web.stdout + exp_web.stderr)
            self.assertIn("test_gateway_agent_settings", exp_web.stdout)
            self._git(repo, "checkout", "--", "web/agent-settings-model.js")

            # 3) requirements.lock.txt / constraints-oss.txt are NOT treated as doc-only!
            for lock_file in ("requirements.lock.txt", "constraints-oss.txt"):
                orig = (repo / lock_file).read_text(encoding="utf-8")
                (repo / lock_file).write_text(orig + "# bump\n", encoding="utf-8")
                exp_lock = self._run(repo, "--changed", "--explain")
                self.assertEqual(exp_lock.returncode, 0, exp_lock.stdout + exp_lock.stderr)
                self.assertIn("test_agent_plugin_coexistence", exp_lock.stdout)
                self.assertIn("test_config", exp_lock.stdout)
                (repo / lock_file).write_text(orig, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
