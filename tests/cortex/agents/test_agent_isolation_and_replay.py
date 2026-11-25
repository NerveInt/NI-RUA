"""Merged cortex tests:
- tests/test_native_terminal_replay.py: Offline replay safety and cache tests; the fixture is not production display data.
- tests/test_pidfd.py: Wrapper compatibility; the real syscall probe only opens self and sends signal 0.
"""

NI_RUA_TEST_TIER = "component"

import os
import signal
import stat
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest import mock
from ni_rua.cortex.agents import isolation as iso
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import sqlite3
from unittest.mock import patch
from ni_rua.cortex.session.native_terminal_replay import CACHE_VERSION, NativeReplayError, NativeTerminalReplay
from support.host_capabilities import native_replay_sandbox, require
import ctypes
import errno
from types import SimpleNamespace
from unittest.mock import Mock, patch
from ni_rua.cortex.session import pidfd
from support.host_capabilities import pidfd_syscall_fallback, require

# ==== tests/test_agent_isolation.py ====

HOST_SECRETS = {"ANTHROPIC_API_KEY": "sk-host", "OPENAI_API_KEY": "sk-o", "AWS_SECRET_ACCESS_KEY": "x",
                "GITHUB_TOKEN": "t", "HTTPS_PROXY": "http://p", "SSH_AUTH_SOCK": "/s", "MY_PASSWORD": "pw",
                "LANG": "en_US.UTF-8", "TZ": "UTC", "PATH": "/evil", "HOME": "/evil"}


class EnvironmentTests(unittest.TestCase):
    def test_no_host_credentials_and_explicit_keys(self):
        env = iso.clean_environment(HOST_SECRETS, path_dirs=["/priv/bin"], home="/run/actor", tmpdir="/run/tmp",
                                    extra={"ANTHROPIC_API_KEY": "ni-rua-local-relay"})
        self.assertEqual(env["ANTHROPIC_API_KEY"], "ni-rua-local-relay")  # extra is the only source
        for key in ("OPENAI_API_KEY", "AWS_SECRET_ACCESS_KEY", "GITHUB_TOKEN", "HTTPS_PROXY", "SSH_AUTH_SOCK",
                    "MY_PASSWORD"):
            self.assertNotIn(key, env)
        self.assertEqual(env["PATH"], "/priv/bin:" + os.defpath)
        self.assertEqual((env["HOME"], env["TMPDIR"], env["LANG"], env["TZ"]),
                         ("/run/actor", "/run/tmp", "en_US.UTF-8", "UTC"))

    def test_ignores_os_environ(self):
        with mock.patch.dict(os.environ, {"NI_RUA_PROBE_SECRET": "leak"}):
            env = iso.clean_environment({}, home="/h", tmpdir="/t")
        self.assertEqual(set(env), {"PATH", "HOME", "TMPDIR", "LANG"})
        self.assertEqual(env["LANG"], "C.UTF-8")

    def test_validation(self):
        with self.assertRaises(ValueError):
            iso.clean_environment({"BAD-NAME": "x"}, home="/h", tmpdir="/t")
        with self.assertRaises(ValueError):
            iso.clean_environment({}, home="relative", tmpdir="/t")
        with self.assertRaises(ValueError):
            iso.clean_environment({}, home="/h", tmpdir="/t", extra={"A": "x\0y"})


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: subprocess.run(["/bin/rm", "-rf", str(self.dir)]))
        self.probe = self.dir / "probe.py"
        self.probe.write_text("import json,os,sys\nprint(json.dumps({'env':dict(os.environ),'argv':sys.argv[1:]}))\n")
        self.probe_sh = self.dir / "probe"
        self.probe_sh.write_text(f"#!/bin/sh\nexec {sys.executable} {self.probe} \"$@\"\n")
        self.probe_sh.chmod(0o700)

    def run_launcher(self, env, reexport=(), host=None, args=()):
        launcher = iso.write_env_launcher(self.dir / "launcher", self.probe_sh, env, reexport=reexport)
        self.assertEqual(stat.S_IMODE(launcher.stat().st_mode), 0o700)
        host_env = {"PATH": os.defpath, **(host or {})}
        out = subprocess.run([str(launcher), *args], env=host_env, capture_output=True, text=True, check=True).stdout
        import json
        return json.loads(out)

    def test_only_given_env_and_reexports_visible(self):
        env = {"PATH": "/usr/bin:/bin", "HOME": "/run/actor", "CLAUDE_CODE_X": "1"}
        host = {"ANTHROPIC_API_KEY": "sk-host", "CLAUDE_CODE_ENTRYPOINT": "sdk-py",
                "CLAUDE_AGENT_SDK_VERSION": "0.2.180", "UNRELATED": "u"}
        result = self.run_launcher(env, reexport=["CLAUDE_CODE_ENTRYPOINT", "CLAUDE_AGENT_SDK_VERSION",
                                                   "CLAUDE_CODE_SDK_READS_SESSION_STATE"], host=host,
                                   args=["--flag", "a b"])
        seen = {k: v for k, v in result["env"].items() if k not in ("LC_CTYPE", "__CF_USER_TEXT_ENCODING", "PWD", "SHLVL", "_")}
        self.assertEqual(seen, {**env, "CLAUDE_CODE_ENTRYPOINT": "sdk-py", "CLAUDE_AGENT_SDK_VERSION": "0.2.180"})
        self.assertEqual(result["argv"], ["--flag", "a b"])

    def test_awkward_values_roundtrip(self):
        values = {"A": "it's \"q\" $HOME `id` \\ *", "B": "line1\nline2", "C": "", "D": "  spaced  ", "E": "a=b=c",
                  "F": "$(touch /tmp/ni-rua-should-not-exist)"}
        result = self.run_launcher(values)
        for key, value in values.items():
            self.assertEqual(result["env"][key], value)

    def test_empty_reexport_value_is_preserved_and_unset_is_absent(self):
        result = self.run_launcher({"A": "1"}, reexport=["EMPTY_ONE", "MISSING_ONE"], host={"EMPTY_ONE": ""})
        self.assertEqual(result["env"]["EMPTY_ONE"], "")
        self.assertNotIn("MISSING_ONE", result["env"])

    def test_rejections(self):
        with self.assertRaises(ValueError):
            iso.write_env_launcher(self.dir / "l", "relative/cli", {})
        with self.assertRaises(ValueError):
            iso.write_env_launcher(self.dir / "l", "/bin/true", {"A": "1"}, reexport=["A"])
        with self.assertRaises(ValueError):
            iso.write_env_launcher(self.dir / "l", "/bin/true", {"A": "1"}, reexport=["bad name"])
        target = self.dir / "real"
        target.write_text("x")
        (self.dir / "link").symlink_to(target)
        with self.assertRaises(ValueError):
            iso.write_env_launcher(self.dir / "link", "/bin/true", {})
        self.assertEqual(target.read_text(), "x")


class PrivateDirTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: subprocess.run(["/bin/rm", "-rf", str(self.dir)]))

    def test_modes_nested_and_idempotent(self):
        old = os.umask(0o022)
        try:
            made = iso.private_dirs(self.dir, ["tmp", "opencode/xdg"])
            again = iso.private_dirs(self.dir, ["tmp"])
        finally:
            os.umask(old)
        self.assertEqual(made["tmp"], again["tmp"])
        for path in (made["tmp"], made["opencode/xdg"], made["opencode/xdg"].parent):
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o700)

    def test_loose_existing_dir_is_tightened(self):
        (self.dir / "tmp").mkdir(mode=0o755)
        os.chmod(self.dir / "tmp", 0o755)
        iso.private_dirs(self.dir, ["tmp"])
        self.assertEqual(stat.S_IMODE((self.dir / "tmp").stat().st_mode), 0o700)

    def test_symlink_and_bad_names_refused(self):
        outside = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: subprocess.run(["/bin/rm", "-rf", str(outside)]))
        (self.dir / "tmp").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "private_dir_symlink"):
            iso.private_dirs(self.dir, ["tmp"])
        (self.dir / "mid").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "private_dir_symlink"):
            iso.private_dirs(self.dir, ["mid/child"])
        self.assertEqual(list(outside.iterdir()), [])
        for bad in ("../x", "/abs", "", "a/../b"):
            with self.assertRaises(ValueError):
                iso.private_dirs(self.dir, [bad])
        (self.dir / "file").write_text("x")
        with self.assertRaisesRegex(ValueError, "private_dir_not_directory"):
            iso.private_dirs(self.dir, ["file"])


class PinningTests(unittest.TestCase):
    def test_sha_cache_and_invalidation(self):
        import hashlib
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "cli"
            path.write_bytes(b"one")
            with mock.patch.object(iso, "_hash_file", wraps=iso._hash_file) as spy:
                self.assertEqual(iso.file_sha256_cached(path), hashlib.sha256(b"one").hexdigest())
                iso.file_sha256_cached(path)
                self.assertEqual(spy.call_count, 1)
                stamp = path.stat().st_mtime_ns
                path.write_bytes(b"two")  # same size
                os.utime(path, ns=(stamp + 5_000_000_000, stamp + 5_000_000_000))
                self.assertEqual(iso.file_sha256_cached(path), hashlib.sha256(b"two").hexdigest())
                self.assertEqual(spy.call_count, 2)
            link = Path(raw) / "link"
            link.symlink_to(path)
            self.assertEqual(iso.file_sha256_cached(link), hashlib.sha256(b"two").hexdigest())
            with self.assertRaises(FileNotFoundError):
                iso.file_sha256_cached(Path(raw) / "missing")

    def test_version_mismatch(self):
        self.assertEqual(iso.require_exact_version("claude_code", " 2.1.321\n", "2.1.321"), "2.1.321")
        for actual in ("2.1.322", "v2.1.321", "2.1.321 (Claude Code)", None, ""):
            with self.assertRaises(iso.VersionMismatch) as caught:
                iso.require_exact_version("opencode", actual, "2.1.321")
            self.assertEqual(str(caught.exception), "opencode_version_mismatch")
            self.assertEqual(caught.exception.pinned, "2.1.321")


PROC_STAT_FAKE = {
    100: "100 (agent) S 1 100 100 0 -1 4194560 1 0 0",
    101: "101 (bash tool) S 100 101 101 0 -1 0",          # detached: own group and session
    102: "102 (sleep)) weird ) S 101 101 101 0 -1 0",    # command contains ')' and spaces
    103: "103 (node) S 100 100 100 0 -1 0",
    200: "200 (other) S 1 200 200 0 -1 0",
    201: "201 (child of other) S 200 200 200 0 -1 0",
}


class SnapshotParserTests(unittest.TestCase):
    def test_proc_stat_parser(self):
        self.assertEqual(iso.parse_proc_stat(102, PROC_STAT_FAKE[102]), iso.ProcessInfo(102, 101, 101))
        self.assertIsNone(iso.parse_proc_stat(1, "garbage"))
        self.assertIsNone(iso.parse_proc_stat(1, "1 (x) S"))

    def test_ps_parser(self):
        rows = iso.parse_ps_table("  1     0     1\n 50  1  50\nbad line\n 7 x 7\n")
        self.assertEqual(rows, [iso.ProcessInfo(1, 0, 1), iso.ProcessInfo(50, 1, 50)])

    def test_linux_proc_fake_tree(self):
        with tempfile.TemporaryDirectory() as raw:
            for pid, text in PROC_STAT_FAKE.items():
                (Path(raw) / str(pid)).mkdir()
                (Path(raw) / str(pid) / "stat").write_text(text)
            (Path(raw) / "self").mkdir()
            (Path(raw) / "cpuinfo").write_text("x")
            (Path(raw) / "999").mkdir()  # exited: no stat file
            table = iso._read_proc_table(raw)
        snapshot = iso.build_snapshot(100, table)
        self.assertEqual([m.pid for m in snapshot.members], [100, 101, 103, 102])
        self.assertEqual(snapshot.pgids, (100, 101))
        self.assertEqual(iso.build_snapshot(555, table).members, ())

    def test_cycle_safe_and_invalid_root(self):
        table = [iso.ProcessInfo(5, 6, 5), iso.ProcessInfo(6, 5, 5)]
        self.assertEqual(len(iso.build_snapshot(5, table).members), 2)
        with self.assertRaises(ValueError):
            iso.process_tree_snapshot(1)


class TerminateTreeTests(unittest.TestCase):
    def spawn_tree(self):
        script = textwrap.dedent("""
            import os, subprocess, sys, time
            detached = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
            grouped = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
            print(detached.pid, grouped.pid, flush=True)
            time.sleep(120)
        """)
        root = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True,
                                start_new_session=True)
        detached, grouped = (int(x) for x in root.stdout.readline().split())
        self.addCleanup(self.reap, root, detached, grouped)
        return root, detached, grouped

    @staticmethod
    def reap(root, *pids):
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        root.kill()
        root.wait()
        root.stdout.close()

    def test_detached_child_found_and_killed(self):
        root, detached, grouped = self.spawn_tree()
        self.assertNotEqual(os.getpgid(detached), os.getpgid(root.pid))
        snapshot = iso.process_tree_snapshot(root.pid)
        self.assertEqual({m.pid for m in snapshot.members}, {root.pid, detached, grouped})
        # a plain killpg of the root group (what the old teardown did) leaves the detached child alive
        os.killpg(os.getpgid(root.pid), signal.SIGKILL)
        root.wait()
        time.sleep(0.2)
        self.assertTrue(iso._alive(detached))
        report = iso.terminate_tree(snapshot, grace_s=1.0)
        self.assertTrue(report.confirmed, report)
        self.assertEqual(report.errors, ())
        self.assertFalse(iso._alive(detached))
        self.assertFalse(iso._alive(grouped))
        self.assertIn(snapshot.members[0].pgid, report.signalled_groups)
        self.assertEqual(report.to_dict()["confirmed"], True)

    def test_sigterm_ignoring_process_needs_sigkill(self):
        script = ("import signal,time,sys\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                  "print('ready', flush=True)\ntime.sleep(120)\n")
        proc = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True,
                                start_new_session=True)
        self.addCleanup(self.reap, proc)
        proc.stdout.readline()
        report = iso.terminate_tree(iso.process_tree_snapshot(proc.pid), grace_s=0.2)
        self.assertTrue(report.confirmed)
        self.assertEqual(report.killed, (proc.pid,))
        self.assertEqual(report.exited_after_term, ())

    def test_never_signals_caller_and_empty_snapshot(self):
        own = iso.TreeSnapshot(os.getpid(), (iso.ProcessInfo(os.getpid(), 1, os.getpgrp()),))
        report = iso.terminate_tree(own, grace_s=0)
        self.assertEqual(report.skipped_self, (os.getpid(),))
        self.assertEqual(report.signalled_groups, ())
        self.assertTrue(report.confirmed)
        empty = iso.terminate_tree(iso.TreeSnapshot(4242424, ()), grace_s=0)
        self.assertTrue(empty.confirmed)
        self.assertEqual(empty.members, ())

    def test_survivor_reported_unconfirmed(self):
        snapshot = iso.TreeSnapshot(77, (iso.ProcessInfo(77, 1, 77),))
        with mock.patch.object(iso, "_alive", return_value=True), mock.patch.object(iso.os, "killpg") as killpg:
            report = iso.terminate_tree(snapshot, grace_s=0, verify_s=0)
        self.assertFalse(report.confirmed)
        self.assertEqual(report.survivors, (77,))
        self.assertEqual([c.args for c in killpg.call_args_list], [(77, signal.SIGTERM), (77, signal.SIGKILL)])


class _Row:
    def __init__(self, pid, ppid, pgid, *, uid=None, env=None, cwd=None):
        self.info, self.uid, self.env, self.cwd = iso.ProcessInfo(pid, ppid, pgid), os.getuid() if uid is None else uid, \
            env, cwd


class MarkerSelectionTests(unittest.TestCase):
    """``select_marked``: the pure rule behind ``processes_with_marker`` (no real processes)."""

    MARK = (b"PATH=/usr/bin", iso.AGENT_MARKER_ENV.encode() + b"=agent-abc")

    def select(self, rows, **kwargs):
        kwargs.setdefault("cwd_roots", ["/runs/r1/actor"])
        kwargs.setdefault("own_pid", 50)
        with mock.patch.object(iso.os.path, "realpath", side_effect=lambda path: path):
            return {item.pid: item.reason for item in iso.select_marked(rows, "agent-abc", **kwargs)}

    def test_environment_marker_and_descendants(self):
        rows = [_Row(50, 40, 40), _Row(40, 1, 40),                                      # the gateway and its parent
                _Row(100, 50, 40, env=self.MARK),                                         # the CLI
                _Row(101, 1, 101, env=self.MARK),                                         # detached, env readable
                _Row(102, 101, 101),                                                      # its child, env unreadable
                _Row(103, 1, 103, env=(iso.AGENT_MARKER_ENV.encode() + b"=agent-other",)),  # another session
                _Row(104, 1, 104, env=self.MARK, uid=os.getuid() + 1)]                    # another user
        self.assertEqual(self.select(rows), {100: "env", 101: "env", 102: "descendant"})

    def test_cwd_fallback_only_for_unreadable_orphans_inside_the_agent_dirs(self):
        rows = [_Row(50, 1, 50),
                _Row(200, 1, 200, cwd="/runs/r1/actor"),               # macOS platform binary, orphaned: selected
                _Row(201, 1, 201, cwd="/runs/r1/actor/sub"),           # in a subdirectory: selected
                _Row(202, 300, 202, cwd="/runs/r1/actor"),             # an operator's shell (parent is a login)
                _Row(300, 1, 300, cwd="/"),
                _Row(203, 1, 203, cwd="/runs/r1/actor-old"),           # prefix only, not inside
                _Row(204, 1, 204, cwd="/runs/r1/actor", env=(b"PATH=/x",)),  # readable env without the marker
                _Row(205, 200, 200, cwd="/tmp")]                       # child of a selected orphan
        self.assertEqual(self.select(rows), {200: "cwd", 201: "cwd", 205: "descendant"})
        self.assertEqual(self.select(rows, cwd_roots=[]), {})

    def test_never_the_caller_its_ancestors_or_pid_one(self):
        rows = [_Row(1, 0, 1, env=self.MARK), _Row(30, 1, 30, env=self.MARK), _Row(50, 30, 30, env=self.MARK),
                _Row(60, 50, 30, env=self.MARK)]
        self.assertEqual(self.select(rows), {60: "env"})

    def test_marker_value_is_validated(self):
        for bad in ("", "a b", "x" * 129, None.__class__.__name__ + "\n"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                iso.agent_marker(bad)
        env = iso.clean_environment({}, home="/h", tmpdir="/t", instance_id="agent-1f")
        self.assertEqual(env[iso.AGENT_MARKER_ENV], "agent-1f")
        self.assertNotIn(iso.AGENT_MARKER_ENV, iso.clean_environment({}, home="/h", tmpdir="/t"))

    def test_linux_proc_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actor = root / "actor"
            actor.mkdir()
            for pid, stat_line, environ in ((11, "11 (perl) S 1 11 11", b"PATH=/b\0" + self.MARK[1] + b"\0"),
                                            (12, "12 (sh) S 1 12 12", None)):
                base = root / "proc" / str(pid)
                base.mkdir(parents=True)
                (base / "stat").write_text(stat_line + " 0 0 0")
                if environ is not None:
                    (base / "environ").write_bytes(environ)
                (base / "cwd").symlink_to(actor)
            rows = {row.info.pid: row for row in iso._linux_rows(root / "proc")}
        self.assertEqual(rows[11].info, iso.ProcessInfo(11, 1, 11))
        self.assertIn(self.MARK[1], rows[11].env)
        self.assertIsNone(rows[12].env)                      # unreadable environ
        self.assertEqual(rows[12].cwd, str(actor))


class MarkerTeardownTests(unittest.TestCase):
    """``processes_with_marker`` / ``terminate_agent_processes`` against real processes of this host."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.actor = Path(os.path.realpath(directory.name)) / "actor"
        self.actor.mkdir()
        self.instance = "agent-test" + os.urandom(6).hex()
        self.env = iso.clean_environment({}, home=self.actor, tmpdir=self.actor, instance_id=self.instance)

    def kill_later(self, *pids):
        for pid in pids:
            self.addCleanup(lambda pid=pid: os.kill(pid, signal.SIGKILL) if iso._alive(pid) else None)

    def leave_detached_children(self):
        """A shell that leaves three children behind and exits at once (what a foreground Bash tool call does):
        ``sleep &`` (an Apple platform binary on macOS: environment withheld), a setsid python and a plain python."""
        script = ("sleep 300 >/dev/null 2>&1 & echo $!; "
                  f"{sys.executable} -c 'import os, time; os.setsid(); time.sleep(300)' >/dev/null 2>&1 & echo $!; "
                  f"{sys.executable} -c 'import time; time.sleep(300)' >/dev/null 2>&1 & echo $!")
        shell = subprocess.run(["/bin/sh", "-c", script], env=self.env, cwd=self.actor, capture_output=True, text=True,
                               timeout=10, start_new_session=True)
        pids = [int(line) for line in shell.stdout.split()]
        self.kill_later(*pids)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not all(self.parent(pid) == 1 for pid in pids):
            time.sleep(0.02)
        return pids

    @staticmethod
    def parent(pid):
        rows = {row.pid: row for row in (iso._read_proc_table() if sys.platform.startswith("linux")
                                         else iso._read_ps_table())}
        return rows[pid].ppid if pid in rows else None

    def test_detached_children_are_found_killed_and_verified(self):
        pids = self.leave_detached_children()
        self.assertTrue(all(iso._alive(pid) for pid in pids))
        scan = iso.processes_with_marker(self.instance, cwd_roots=[self.actor])
        self.assertTrue(set(pids) <= set(scan.pids), (pids, scan.to_dict()))
        self.assertNotIn(os.getpid(), scan.pids)
        report = iso.terminate_agent_processes(iso.TreeSnapshot(0, ()), self.instance, cwd_roots=[self.actor],
                                               grace_s=2.0)
        self.assertTrue(report.confirmed, report.to_dict())
        self.assertFalse(any(iso._alive(pid) for pid in pids))
        self.assertEqual(report.remaining, ())
        self.assertNotIn(os.getpgrp(), report.termination.signalled_groups)
        self.assertEqual(iso.processes_with_marker(self.instance, cwd_roots=[self.actor]).found, ())

    def test_unmarked_processes_are_left_alone(self):
        # A child of this process in the actor dir (an operator's shell is never an orphan) and a process of another
        # session's marker are not selected.
        other = dict(self.env, **{iso.AGENT_MARKER_ENV: "agent-someone-else"})
        own_child = subprocess.Popen(["sleep", "300"], cwd=self.actor, env={"PATH": os.defpath})
        foreign = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"], env=other,
                                   start_new_session=True)
        for process in (own_child, foreign):
            self.addCleanup(process.wait)
            self.addCleanup(process.kill)
        time.sleep(0.3)
        scan = iso.processes_with_marker(self.instance, cwd_roots=[self.actor])
        self.assertFalse({own_child.pid, foreign.pid} & set(scan.pids), scan.to_dict())

    def test_scan_failure_and_survivors_make_the_receipt_unconfirmed(self):
        failed = iso.terminate_agent_processes(iso.TreeSnapshot(0, ()), self.instance,
                                               scan=mock.Mock(side_effect=OSError("ps")))
        self.assertFalse(failed.confirmed)
        self.assertEqual(failed.errors[0], "marker_scan:OSError")
        survivor = iso.MarkedProcess(4242, 1, 4242, "env")
        stubborn = iso.MarkerScan(self.instance, (survivor,), (survivor.info(),))
        with mock.patch.object(iso, "_alive", return_value=True), mock.patch.object(iso.os, "killpg"):
            report = iso.terminate_agent_processes(iso.TreeSnapshot(0, ()), self.instance, grace_s=0, verify_s=0,
                                                   scan=lambda *args, **kwargs: stubborn, max_rounds=2)
        self.assertFalse(report.confirmed)
        self.assertEqual(report.remaining, (4242,))
        self.assertEqual(len(report.rounds), 2)
        self.assertEqual(report.to_dict()["confirmed"], False)

    def test_group_signalling_rules(self):
        # cwd-only selection in a group that also holds an unselected process: signalled by pid, never killpg.
        lonely = iso.MarkedProcess(5001, 1, 6000, "cwd")
        grouped = iso.MarkedProcess(5002, 1, 7000, "cwd")
        marked = iso.MarkedProcess(5003, 1, 8000, "env")
        table = (lonely.info(), iso.ProcessInfo(5009, 1, 6000), grouped.info(), marked.info(),
                 iso.ProcessInfo(5010, 5003, 8000))
        scans = [iso.MarkerScan(self.instance, (lonely, grouped, marked), table), iso.MarkerScan(self.instance, (), ())]
        with mock.patch.object(iso, "_alive", return_value=False), mock.patch.object(iso.os, "killpg") as killpg, \
                mock.patch.object(iso.os, "kill") as kill:
            report = iso.terminate_agent_processes(iso.TreeSnapshot(0, ()), self.instance, grace_s=0, verify_s=0,
                                                   scan=lambda *args, **kwargs: scans.pop(0))
        self.assertTrue(report.confirmed)
        self.assertEqual(sorted(report.termination.signalled_groups), [7000, 8000])
        self.assertEqual(report.termination.signalled_pids, (5001,))
        self.assertEqual(sorted(c.args[0] for c in killpg.call_args_list), [7000, 8000])
        self.assertEqual([c.args[0] for c in kill.call_args_list], [5001])

    def test_terminate_tree_signal_groups_parameter(self):
        snapshot = iso.TreeSnapshot(77, (iso.ProcessInfo(77, 1, 77), iso.ProcessInfo(78, 77, 90)))
        with mock.patch.object(iso, "_alive", return_value=False), mock.patch.object(iso.os, "killpg") as killpg, \
                mock.patch.object(iso.os, "kill") as kill:
            report = iso.terminate_tree(snapshot, grace_s=0, verify_s=0, signal_groups={77})
        self.assertEqual((report.signalled_groups, report.signalled_pids), ((77,), (78,)))
        self.assertEqual([c.args for c in killpg.call_args_list], [(77, signal.SIGTERM)])
        self.assertEqual([c.args for c in kill.call_args_list], [(78, signal.SIGTERM)])

# ==== tests/test_native_terminal_replay.py ====

THREAD = "01a100bb-1592-7820-a7bc-5ed3e48aaa50"
DISPLAY = "› original archived task\n• Called robot_observe · 12ms\n• Failed robot_move_ee · 34ms"


class NativeReplayTests(unittest.TestCase):
    def test_unknown_archive_schema_fails_before_native_capture(self):
        with sqlite3.connect(self.home / 'state_5.sqlite') as db:
            db.execute('ALTER TABLE threads RENAME COLUMN reasoning_effort TO incompatible_effort')
        with patch.object(self.replay, '_capture', side_effect=AssertionError('unsupported schema launched native CLI')):
            with self.assertRaisesRegex(NativeReplayError, 'unsupported native archive schema'):
                self.replay.get(self.evidence, THREAD)
        self.assertIsNone(self.replay.peek(self.evidence, THREAD))

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.evidence = self.root / "archive"
        self.home = self.evidence / "private" / "codex-home"
        sessions = self.home / "sessions" / "2026" / "10" / "03"
        sessions.mkdir(parents=True)
        self.rollout = sessions / f"rollout-2026-10-03T15-47-04-{THREAD}.jsonl"
        self.rollout.write_text(json.dumps({"type": "session_meta", "payload": {"id": THREAD}}) + "\n")
        self.auth = self.home / "auth.json"
        self.auth.write_text('{"secret":"do-not-copy"}')
        (self.home / "config.toml").write_text('untrusted_field = "do-not-copy"\n')
        self.original_terminal = self.evidence / "terminal.bin"
        self.original_terminal.write_bytes(b"Error: account/read failed during TUI bootstrap: archived failure")
        with sqlite3.connect(self.home / "state_5.sqlite") as db:
            db.execute("CREATE TABLE threads(id TEXT, model_provider TEXT, model TEXT, reasoning_effort TEXT, rollout_path TEXT, cwd TEXT)")
            db.execute("INSERT INTO threads VALUES(?,?,?,?,?,?)",
                       (THREAD, "ni_rua_model_proxy", "gpt-6-astra", "high", str(self.rollout), str(self.evidence)))
        with sqlite3.connect(self.home / "thread_history_1.sqlite") as db:
            db.execute("CREATE TABLE thread_turns(thread_id TEXT, turn_id TEXT, status TEXT)")
            db.execute("CREATE TABLE thread_items(thread_id TEXT, item_id TEXT, item_json TEXT, rollout_ordinal INTEGER)")
            db.execute("INSERT INTO thread_turns VALUES(?,?,?)", (THREAD, "turn-original", "interrupted"))
            for ordinal, item in enumerate([
                {"type": "userMessage", "content": [{"type": "text", "text": "original archived task"}]},
                {"type": "dynamicToolCall", "tool": "robot_observe", "status": "completed", "success": True, "durationMs": 12},
                {"type": "dynamicToolCall", "tool": "robot_move_ee", "status": "failed", "success": False, "durationMs": 34},
            ]):
                db.execute("INSERT INTO thread_items VALUES(?,?,?,?)",
                           (THREAD, "item-" + str(ordinal), json.dumps(item), ordinal))
        self.fixture = self.root / "native_fixture.py"
        self.fixture.write_text('''import json,os,socket,sys,time
from pathlib import Path
assert os.isatty(0) and os.isatty(1)
home = Path(os.environ["CODEX_HOME"])
assert not (home / "auth.json").exists()
assert "do-not-copy" not in (home / "config.toml").read_text()
assert not any("OPENAI" in key or "API_KEY" in key for key in os.environ)
for family in (socket.AF_INET, socket.AF_INET6):
    try: socket.socket(family)
    except PermissionError: pass
    else: raise AssertionError("network access was not blocked")
try: Path(sys.argv[1]).write_text("forbidden")
except PermissionError: pass
else: raise AssertionError("outside-cache write was not blocked")
(home / "fixture-command.json").write_text(json.dumps(sys.argv[2:]))
os.write(1,b"\\x1b[6n")
time.sleep(.1)
print("OpenAI Codex fixture: recorded history",flush=True)
print("''' + DISPLAY.replace("\n", "\\n") + '''",flush=True)
print("Ask Codex to do anything",flush=True)
time.sleep(30)
''')
        self.replay = NativeTerminalReplay(
            self.root / "cache", command_prefix=(sys.executable, str(self.fixture), str(self.original_terminal)),
            cli_version="test-fixture", timeout=3, quiet_period=.15)

    def fingerprint(self):
        return {str(path.relative_to(self.evidence)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in self.evidence.rglob("*") if path.is_file()}

    @require(native_replay_sandbox, "replays the native CLI under the Landlock + seccomp replay sandbox")
    def test_native_bytes_are_cached_without_auth_archive_writes_or_new_turns(self):
        before = self.fingerprint()
        self.assertTrue(self.replay.available(self.evidence, THREAD))
        self.assertIsNone(self.replay.peek(self.evidence, THREAD))
        result = self.replay.get(self.evidence, THREAD)
        self.assertEqual(before, self.fingerprint())
        self.assertEqual(self.replay.peek(self.evidence, THREAD), result)
        self.assertEqual(self.replay.get(self.evidence, THREAD), result)
        status = result["status"]
        self.assertEqual((status["cols"], status["rows"]), (140, 40))
        self.assertEqual(status["exit_code"], 0)
        self.assertFalse(status["bootstrap_failed"])
        self.assertEqual(status["replay"]["kind"], "recovered_native_tui")
        self.assertFalse(status["replay"]["original_recording"])
        self.assertEqual(status["replay"]["compatibility"]["adapter"], "codex_archive_v1")
        self.assertTrue(status["replay"]["original_bootstrap_failed"])
        self.assertTrue(status["replay"]["history_unchanged"])
        self.assertEqual(status["replay"]["recorded_turns"], 1)
        self.assertEqual(status["replay"]["recorded_items"], 3)
        self.assertEqual(status["replay"]["display_validation"]["expected_tools"], 2)
        self.assertTrue(status["replay"]["display_validation"]["complete"])
        self.assertEqual(status["replay"]["capture_exit_code"], -15)
        self.assertIn(b"recorded history", result["path"].read_bytes())
        copied = result["path"].parent / "codex-home"
        self.assertFalse((copied / "auth.json").exists())
        args = json.loads((copied / "fixture-command.json").read_text())
        self.assertEqual(args, ["resume", "--no-daemon", "--no-alt-screen", "-C",
                               str(result["path"].parent / "workspace"), "-s", "read-only",
                               "-a", "never", THREAD])

    @require(native_replay_sandbox, "replays the native CLI under the Landlock + seccomp replay sandbox")
    def test_concurrent_requests_share_one_complete_capture(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(self.replay.get, self.evidence, THREAD) for _ in range(2)]
            first, second = [future.result(timeout=8) for future in futures]
        self.assertEqual(first, second)
        self.assertEqual(len(list((self.root / "cache").glob("*/metadata.json"))), 1)

    @require(native_replay_sandbox, "replays the native CLI under the Landlock + seccomp replay sandbox")
    def test_source_change_and_corrupt_cache_are_not_reused(self):
        first = self.replay.get(self.evidence, THREAD)
        metadata_path = first["path"].parent / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["cache_version"] = CACHE_VERSION - 1
        metadata_path.write_text(json.dumps(metadata))
        self.assertIsNone(self.replay.peek(self.evidence, THREAD))
        first["path"].write_bytes(b"corrupt")
        self.assertIsNone(self.replay.peek(self.evidence, THREAD))
        repaired = self.replay.get(self.evidence, THREAD)
        self.assertIn(b"recorded history", repaired["path"].read_bytes())
        self.rollout.write_text(self.rollout.read_text() + '{"type":"event_msg","payload":{}}\n')
        self.assertIsNone(self.replay.peek(self.evidence, THREAD))
        changed = self.replay.get(self.evidence, THREAD)
        self.assertNotEqual(first["path"], changed["path"])

    def test_invalid_identity_symlink_and_nested_cache_are_rejected(self):
        self.assertFalse(self.replay.available(self.evidence, "../../escape"))
        with self.assertRaises((ValueError, NativeReplayError)):
            self.replay.get(self.evidence, "../../escape")
        original = self.rollout.read_bytes()
        target = self.root / "external.jsonl"
        target.write_bytes(original)
        self.rollout.unlink(); self.rollout.symlink_to(target)
        self.assertFalse(self.replay.available(self.evidence, THREAD))
        with self.assertRaises(NativeReplayError):
            self.replay.get(self.evidence, THREAD)
        self.rollout.unlink(); self.rollout.write_bytes(original)
        nested = NativeTerminalReplay(self.evidence / "bad-cache", command_prefix=(sys.executable,), cli_version="fixture")
        self.assertFalse(nested.available(self.evidence, THREAD))
        link = self.root / "cache-link"
        link.symlink_to(self.root / "cache", target_is_directory=True)
        with self.assertRaises(NativeReplayError):
            NativeTerminalReplay(link, command_prefix=(sys.executable,), cli_version="fixture")

        # Missing codex_cli_bin never falls back to unpinned PATH codex; available() is False and _capture raises.
        with patch.dict(sys.modules, {"codex_cli_bin": None}), \
                patch("ni_rua.cortex.session.native_terminal_replay.shutil.which") as which_mock:
            unpinned = NativeTerminalReplay(self.root / "cache-nobin", cli_version="unavailable")
        which_mock.assert_not_called()
        self.assertEqual(unpinned.command_prefix, ())
        self.assertFalse(unpinned.available(self.evidence, THREAD))
        with self.assertRaisesRegex(NativeReplayError, "pinned openai-codex-cli-bin is required"):
            unpinned._capture({"thread_id": THREAD}, self.root, self.root, self.root, {})

        # _capture closes master and slave PTY FDs if ioctl fails before Popen.
        closed_pty = []
        real_close = os.close
        with patch("ni_rua.cortex.session.native_terminal_replay.pty.openpty", return_value=(901, 902)), \
                patch("ni_rua.cortex.session.native_terminal_replay.fcntl.ioctl", side_effect=OSError("bad_pty")), \
                patch("ni_rua.cortex.session.native_terminal_replay.os.close",
                      side_effect=lambda fd: closed_pty.append(fd) if fd in (901, 902) else real_close(fd)):
            with self.assertRaises(OSError):
                self.replay._capture({"thread_id": THREAD}, self.root, self.root, self.root, {})
        self.assertEqual(sorted(closed_pty), [901, 902])

    @require(native_replay_sandbox, "replays the native CLI under the Landlock + seccomp replay sandbox")
    def test_failure_and_deadline_never_publish_a_ready_cache(self):
        for program in ("print('Error: account/read failed during TUI bootstrap: unavailable')\n",
                        "import time\nprint('OpenAI Codex loading', flush=True)\ntime.sleep(30)\n"):
            with self.subTest(program=program):
                self.fixture.write_text(program)
                self.replay.timeout = .3
                with self.assertRaises(NativeReplayError):
                    self.replay.get(self.evidence, THREAD)
                self.assertIsNone(self.replay.peek(self.evidence, THREAD))
                source = self.replay._source(self.evidence, THREAD)
                directory = self.root / "cache" / source["key"]
                self.assertTrue((directory / "failure.json").is_file())
                self.assertFalse((directory / "metadata.json").exists())

    @require(native_replay_sandbox, "replays the native CLI under the Landlock + seccomp replay sandbox")
    def test_archive_history_mutation_rejects_capture(self):
        self.fixture.write_text('''import os,sqlite3,time
from pathlib import Path
with sqlite3.connect(Path(os.environ["CODEX_HOME"])/"thread_history_1.sqlite") as db:
    db.execute("INSERT INTO thread_turns VALUES(?,?,?)", ("''' + THREAD + '''", "new-turn", "completed"))
print("OpenAI Codex fixture: recorded history",flush=True)
print("''' + DISPLAY.replace("\n", "\\n") + '''",flush=True)
print("Ask Codex to do anything",flush=True)
time.sleep(30)
''')
        before = self.fingerprint()
        with self.assertRaisesRegex(NativeReplayError, "changed archived turns or items"):
            self.replay.get(self.evidence, THREAD)
        self.assertIsNone(self.replay.peek(self.evidence, THREAD))
        self.assertEqual(before, self.fingerprint())

    @require(native_replay_sandbox, "replays the native CLI under the Landlock + seccomp replay sandbox")
    def test_loading_composer_and_partial_history_do_not_finish_capture(self):
        self.fixture.write_text('''import time
print("OpenAI Codex fixture: loading Resuming session…",flush=True)
print("Ask Codex to do anything",flush=True)
time.sleep(.35)
print("› original archived task",flush=True)
print("• Called robot_observe · 12ms",flush=True)
time.sleep(.35)
print("• Failed robot_move_ee · 34ms",flush=True)
time.sleep(30)
''')
        started = time.monotonic()
        result = self.replay.get(self.evidence, THREAD)
        self.assertGreater(time.monotonic() - started, .7)
        self.assertIn("Failed robot_move_ee".encode(), result["path"].read_bytes())
        self.assertEqual(result["status"]["replay"]["display_validation"]["observed_tools"], 2)

    @require(native_replay_sandbox, "replays the native CLI under the Landlock + seccomp replay sandbox")
    def test_missing_messages_tools_or_original_tool_duration_reject_publication(self):
        outputs = ["loading Resuming session…",
                   "› original archived task\n• Called robot_observe · 12ms\n• Called robot_observe · 12ms",
                   "› original archived task\n• Called robot_observe · 12ms\n• Failed robot_move_ee · 35ms",
                   "• Called robot_observe · 12ms\n• Failed robot_move_ee · 34ms"]
        self.replay.timeout = .35
        for output in outputs:
            with self.subTest(output=output):
                self.fixture.write_text("import time\nprint('OpenAI Codex fixture',flush=True)\n"
                                       + f"print({output!r},flush=True)\n"
                                       + "print('Ask Codex to do anything',flush=True)\ntime.sleep(30)\n")
                with self.assertRaises(NativeReplayError):
                    self.replay.get(self.evidence, THREAD)
                self.assertIsNone(self.replay.peek(self.evidence, THREAD))

# ==== tests/test_pidfd.py ====

class PidfdTests(unittest.TestCase):
    def test_prefers_standard_library_without_loading_libc(self):
        with patch.object(pidfd.os, "pidfd_open", return_value=17, create=True) as opened, \
                patch.object(pidfd.signal, "pidfd_send_signal", return_value=None, create=True) as sent, \
                patch.object(pidfd.ctypes, "CDLL") as loader:
            self.assertEqual(pidfd.pidfd_open(123, 0), 17)
            self.assertIsNone(pidfd.pidfd_send_signal(17, 0))
        opened.assert_called_once_with(123, 0)
        sent.assert_called_once_with(17, 0, None, 0)
        loader.assert_not_called()

    def test_syscall_arguments_on_supported_architectures(self):
        for machine in ("x86_64", "aarch64"):
            with self.subTest(machine=machine):
                syscall = Mock(side_effect=[17, 0])
                with patch.object(pidfd.os, "pidfd_open", None, create=True), \
                        patch.object(pidfd.signal, "pidfd_send_signal", None, create=True), \
                        patch.object(pidfd.sys, "platform", "linux"), \
                        patch.object(pidfd.platform, "machine", return_value=machine), \
                        patch.object(pidfd.ctypes, "CDLL", return_value=SimpleNamespace(syscall=syscall)):
                    self.assertEqual(pidfd.pidfd_open(123), 17)
                    self.assertIsNone(pidfd.pidfd_send_signal(17, 0))
                self.assertEqual([[arg.value for arg in call.args] for call in syscall.call_args_list],
                                 [[434, 123, 0], [424, 17, 0, None, 0]])
                self.assertIs(syscall.restype, ctypes.c_long)

    def test_errno_is_preserved_without_retry_or_pid_signal(self):
        for error in (errno.ESRCH, errno.EPERM, errno.ENOSYS, errno.EBADF):
            def fail(*arguments):
                ctypes.set_errno(error)
                return -1
            for operation, args in ((pidfd.pidfd_open, (123,)), (pidfd.pidfd_send_signal, (17, 0))):
                with self.subTest(error=error, operation=operation.__name__), \
                        patch.object(pidfd.os, "pidfd_open", None, create=True), \
                        patch.object(pidfd.signal, "pidfd_send_signal", None, create=True), \
                        patch.object(pidfd.sys, "platform", "linux"), \
                        patch.object(pidfd.platform, "machine", return_value="x86_64"), \
                        patch.object(pidfd.ctypes, "CDLL", return_value=SimpleNamespace(syscall=Mock(side_effect=fail))), \
                        patch.object(pidfd.os, "kill") as kill:
                    with self.assertRaises(OSError) as caught:
                        operation(*args)
                    self.assertEqual(caught.exception.errno, error)
                    if error == errno.ESRCH:
                        self.assertIsInstance(caught.exception, ProcessLookupError)
                    kill.assert_not_called()

    def test_unknown_platform_and_architecture_fail_closed(self):
        for system, machine in (("linux", "riscv64"), ("darwin", "x86_64")):
            with self.subTest(system=system, machine=machine), \
                    patch.object(pidfd.os, "pidfd_open", None, create=True), \
                    patch.object(pidfd.signal, "pidfd_send_signal", None, create=True), \
                    patch.object(pidfd.sys, "platform", system), \
                    patch.object(pidfd.platform, "machine", return_value=machine), \
                    patch.object(pidfd.ctypes, "CDLL") as loader:
                for operation, args in ((pidfd.pidfd_open, (123,)), (pidfd.pidfd_send_signal, (17, 0))):
                    with self.assertRaises(OSError) as caught:
                        operation(*args)
                    self.assertEqual(caught.exception.errno, errno.ENOSYS)
                loader.assert_not_called()

    @require(pidfd_syscall_fallback, "opens and signals a pidfd through the raw pidfd syscall fallback")
    def test_real_fallback_self_descriptor_and_signal_zero(self):
        with patch.object(pidfd.os, "pidfd_open", None, create=True), \
                patch.object(pidfd.signal, "pidfd_send_signal", None, create=True):
            descriptor = pidfd.pidfd_open(os.getpid())
            try:
                self.assertFalse(os.get_inheritable(descriptor))
                self.assertIsNone(pidfd.pidfd_send_signal(descriptor, 0))
            finally:
                os.close(descriptor)


class ProcessHandleTests(unittest.TestCase):
    """Real child processes: exact argv, bound exit wait, refusal after exit."""

    @unittest.skipUnless(sys.platform in ("linux", "darwin"), "pidfd or kqueue required")
    def test_argv_signal_and_exit_wait_are_bound_to_one_child(self):
        import signal, subprocess
        marker = "unix:///tmp/ni-rua path with spaces.sock"
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", marker])
        handle = pidfd.open_process(child.pid)
        try:
            self.assertIn(marker, pidfd.process_argv(child.pid))
            self.assertFalse(handle.wait(0))
            handle.send_signal(signal.SIGTERM)
            self.assertTrue(handle.wait(5))  # unreaped zombie still counts as exited
            if sys.platform == "darwin":
                with self.assertRaises(ProcessLookupError):
                    handle.send_signal(signal.SIGKILL)
        finally:
            handle.close()
            self.assertEqual(child.wait(timeout=5), -signal.SIGTERM)

    @unittest.skipUnless(sys.platform in ("linux", "darwin"), "pidfd or kqueue required")
    def test_missing_process_is_reported_not_guessed(self):
        import subprocess
        child = subprocess.Popen([sys.executable, "-c", "pass"]); child.wait()
        with self.assertRaises(ProcessLookupError):
            pidfd.open_process(child.pid).close()

    def test_darwin_truncated_procargs_raises_process_lookup_error_instead_of_value_error(self):
        for payload in (b"\x01", b"\x01\x00\x00\x00no-nul-byte"):
            fake_libc = mock.MagicMock()

            def fake_sysctl(_mib, _miblen, buf, size_ptr, *_rest, data=payload):
                size_ptr._obj.value = len(data)
                if buf is not None:
                    buf.raw = data
                return 0

            fake_libc.sysctl.side_effect = fake_sysctl
            with patch.object(pidfd.sys, "platform", "darwin"), \
                    patch.object(pidfd.ctypes, "CDLL", return_value=fake_libc):
                with self.assertRaises(ProcessLookupError):
                    pidfd.process_argv(42424)

if __name__ == "__main__":
    unittest.main()
