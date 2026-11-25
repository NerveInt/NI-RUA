"""Codex bubblewrap sandbox unit, package-proxy, and item-28 boundary conformance tests."""

from io import BytesIO
import os
from pathlib import Path
import select
import shutil
import socket
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

from ni_rua.core.config import Config
from ni_rua.core.storage.store import Store
from ni_rua.cortex.agents.codex.sandbox import CodexSandbox, _UnixForwarder
from ni_rua.cortex.agents.codex.session import UNIX_SOCKET_PATH_MAX
from support.agent_conformance import SANDBOX_PROBE_MARKER, TOOLS, SandboxBoundary
from support.fake_model_servers import FakeResponsesServer, Step, ToolCall, text_step
from support.host_capabilities import bubblewrap_namespaces, require
from support.requirements import requires_bwrap, requires_codex_binary

NI_RUA_TEST_TIER = "packaging"


def short_temporary_directory():
    """A temporary root under /tmp: these tests bind the stand-in native listener at its host path
    ``<root>/ipc/daemon/<64 hex>``, which must fit in sockaddr_un whatever TMPDIR the suite runs with."""
    return tempfile.TemporaryDirectory(prefix="r-cs-", dir="/tmp")


@require(bubblewrap_namespaces, "the native Codex actor sandbox runs real bubblewrap namespaces")
class CodexSandboxTests(unittest.TestCase):
    @staticmethod
    def _python(): return CodexSandbox._find_python_runtime()

    def test_dynamic_host_tools_accept_temporary_path_wrappers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); bin_dir = root / "bin"; bin_dir.mkdir()
            real = {name: CodexSandbox._find_host_tool(name) for name in ("bwrap", "sudo", "setpriv")}
            for name, target in real.items():
                if name == "setpriv":
                    shutil.copy2(target, bin_dir / name)
                    continue
                wrapper = bin_dir / name
                wrapper.write_text(f"#!/bin/sh\nexec {target} \"$@\"\n")
                wrapper.chmod(0o755)
            actor, home, ipc, private = (root / name for name in ("actor", "codex", "ipc", "private"))
            for path in (actor, home, ipc, private): path.mkdir(mode=0o700)
            sandbox = CodexSandbox(private, ipc, actor, home, self._python(), packages=False)
            with patch.dict(os.environ, {"PATH": str(bin_dir)}):
                prefix, env = sandbox.prepare()
                self.assertIn(str(bin_dir / "bwrap"), prefix)
                if not sandbox.user_namespace:
                    self.assertIn(str(bin_dir / "sudo"), prefix)
                    self.assertIn("/opt/ni-rua-bin/setpriv", prefix)
                result = subprocess.run(prefix + [str(sandbox.python_runtime), "-c", "pass"],
                                        env={**os.environ, **env}, capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            sandbox.close()

    def test_native_socket_alias_is_repaired_only_for_this_run(self):
        with short_temporary_directory() as temp:
            root = Path(temp); actor, home, ipc, private = (root / name for name in ("actor", "codex", "ipc", "private"))
            for path in (actor, home, ipc, private): path.mkdir(mode=0o700)
            daemon = ipc / "daemon"; daemon.mkdir(mode=0o700)
            sandbox = CodexSandbox(private, ipc, actor, home, self._python(), packages=False)
            sandbox.prepare()
            native_name = "a" * 64
            listener = socket.socket(socket.AF_UNIX)
            listener.bind(str(daemon / native_name)); listener.listen(1)
            alias = ipc / "codex.sock"
            alias.symlink_to(Path(f"/tmp/codex-daemon-{os.getuid()}") / native_name)
            try:
                self.assertTrue(sandbox.repair_socket_alias(alias))
                self.assertEqual(alias.readlink(), daemon / native_name)
                client = socket.socket(socket.AF_UNIX)
                try:
                    client.connect(str(sandbox.host_socket_path))
                    accepted, _ = listener.accept(); accepted.close()
                finally:
                    client.close()
            finally:
                listener.close(); sandbox.close()
            self.assertFalse(daemon.exists())

    def test_host_socket_descriptor_is_pinned_across_daemon_parent_replacement(self):
        with short_temporary_directory() as temp:
            root = Path(temp); actor, home, ipc, private = (root / name for name in ("actor", "codex", "ipc", "private"))
            external = root / "external"; external.mkdir()
            for path in (actor, home, ipc, private): path.mkdir(mode=0o700)
            sandbox = CodexSandbox(private, ipc, actor, home, self._python(), packages=False)
            sandbox.prepare()
            daemon = ipc / "daemon"
            native_name = "b" * 64
            original = socket.socket(socket.AF_UNIX); original.bind(str(daemon / native_name)); original.listen(1)
            malicious = socket.socket(socket.AF_UNIX); malicious.bind(str(external / native_name)); malicious.listen(1)
            alias = ipc / "codex.sock"
            alias.symlink_to(Path(f"/tmp/codex-daemon-{os.getuid()}") / native_name)
            moved = ipc / "daemon-original"
            try:
                self.assertTrue(sandbox.repair_socket_alias(alias))
                daemon.rename(moved)
                daemon.symlink_to(external, target_is_directory=True)
                alias.unlink()
                alias.symlink_to(Path(f"/tmp/codex-daemon-{os.getuid()}") / native_name)
                with self.assertRaisesRegex(ValueError, "socket_directory_changed"):
                    sandbox.repair_socket_alias(alias)
                client = socket.socket(socket.AF_UNIX)
                try:
                    client.connect(str(sandbox.host_socket_path))
                    accepted, _ = original.accept(); accepted.close()
                    self.assertFalse(select.select([malicious], [], [], .1)[0],
                                     "host bridge followed actor-replaced daemon directory")
                finally:
                    client.close()
            finally:
                original.close(); malicious.close(); sandbox.close()
                if daemon.is_symlink(): daemon.unlink()
                if moved.exists(): shutil.rmtree(moved)

    def test_deep_ipc_directory_reaches_the_native_socket_without_its_long_host_path(self):
        # Codex binds <64 hex> below /tmp/codex-daemon-<uid> inside the namespace, a short path. On the host that
        # socket is <ipc>/daemon/<64 hex>, longer than sockaddr_un allows when the run's IPC directory is as deep as
        # CodexSession accepts (only <ipc>/sandbox-owner.sock must fit). The host never binds or connects that path:
        # it reaches the socket through the pinned descriptor and the run-owned alias <ipc>/codex.sock.
        with short_temporary_directory() as temp:
            root = Path(temp) / ("d" * 60)
            actor, home, ipc, private = (root / name for name in ("actor", "codex", "ipc", "private"))
            for path in (actor, home, ipc, private): path.mkdir(parents=True, mode=0o700)
            sandbox = CodexSandbox(private, ipc, actor, home, self._python(), packages=False)
            sandbox.prepare()
            native_name = "c" * 64
            daemon = ipc / "daemon"
            self.assertLessEqual(len(os.fsencode(ipc / "sandbox-owner.sock")), UNIX_SOCKET_PATH_MAX)
            self.assertGreater(len(os.fsencode(daemon / native_name)), UNIX_SOCKET_PATH_MAX)
            # The stand-in native listener binds through a short descriptor path, as Codex does through its mount.
            daemon_fd = os.open(daemon, os.O_RDONLY | os.O_DIRECTORY)
            listener = socket.socket(socket.AF_UNIX)
            try:
                listener.bind(f"/proc/self/fd/{daemon_fd}/{native_name}"); listener.listen(2)
                alias = ipc / "codex.sock"
                alias.symlink_to(Path(f"/tmp/codex-daemon-{os.getuid()}") / native_name)
                self.assertTrue(sandbox.repair_socket_alias(alias))
                for path in (sandbox.host_socket_path, alias):
                    client = socket.socket(socket.AF_UNIX)
                    try:
                        client.connect(str(path))
                        accepted, _ = listener.accept(); accepted.close()
                    finally:
                        client.close()
            finally:
                listener.close(); os.close(daemon_fd); sandbox.close()
            self.assertFalse(daemon.exists())

    def test_native_prefix_limits_filesystem_and_allows_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            actor, home, ipc, private = (root / name for name in ("actor", "codex", "ipc", "private"))
            source = root / "snapshot"
            for path in (actor, home, ipc, private, source): path.mkdir(mode=0o700)
            (actor / "actor.txt").write_text("actor-visible")
            (home / "memory.md").write_text("memory-visible")
            (source / "allowed.py").write_text("snapshot-visible")
            secret = root / "secret.txt"
            secret.write_text("host-hidden")
            sandbox = CodexSandbox(private, ipc, actor, home, self._python(),
                                   read_paths=[(source, Path("/harness"))])
            prefix, env = sandbox.prepare()
            code = ("from pathlib import Path; import os; "
                    "assert Path(os.environ['HOME'], 'actor.txt').read_text() == 'actor-visible'; "
                    "assert Path(os.environ['CODEX_HOME'], 'memory.md').read_text() == 'memory-visible'; "
                    "assert Path('/harness/allowed.py').read_text() == 'snapshot-visible'; "
                    "import sysconfig; package_dirs=[Path(sysconfig.get_path(n)) for n in ('purelib','platlib')]; "
                    "assert not any(p.exists() for p in package_dirs), [(str(p),p.exists()) for p in package_dirs]; "
                    "import subprocess; rg=Path('/opt/ni-rua-bin/rg'); "
                    "assert not rg.exists() or subprocess.run([str(rg), '-n', 'snapshot-visible', '/harness/allowed.py'], "
                    "capture_output=True).returncode == 0; "
                    "import pip, stat; assert pip.__version__; assert stat.S_IMODE(Path('/tmp').stat().st_mode) == 0o1777; "
                    "assert Path(os.environ['PIP_TARGET']).is_dir(); "
                    "import importlib.util; assert importlib.util.find_spec('synthetic_new_package') is None; "
                    "subprocess.run(['python3', '-c', \"import os; from pathlib import Path; "
                    "Path(os.environ['PIP_TARGET'], 'synthetic_new_package.py').write_text('value = 42')\"], check=True); "
                    "import synthetic_new_package; assert synthetic_new_package.value == 42; "
                    "import tempfile; f=tempfile.TemporaryFile(); f.close(); assert Path(os.environ['TMPDIR']).is_dir(); "
                    "status=Path('/proc/self/status').read_text(); "
                    "assert os.geteuid() == os.getuid(); assert 'CapEff:\\t0000000000000000' in status; "
                    "assert 'NoNewPrivs:\\t1' in status; assert set(os.getgroups()).issubset({os.getgid()}); "
                    f"assert not Path({str(secret)!r}).exists(); "
                    f"assert not Path({str(ipc / 'relays' / 'packages')!r}).exists(); "
                    "assert not Path('/var/run/docker.sock').exists(); "
                    f"assert not Path({str(Path.home())!r}).exists(); "
                    "assert not Path('/etc/shadow').exists()")
            try:
                result = subprocess.run(prefix + [str(sandbox.python_runtime), "-c", code], env={**os.environ, **env},
                                                        capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)
            finally:
                sandbox.close()
            self.assertFalse((ipc / "relays").exists())

    def test_shell_pipelines_find_their_standard_tools_inside_the_namespace(self):
        """Regression: the actor's shell had no grep, rg, python or diff, `xargs echo` had no echo, and every shell
        start printed a setlocale warning. Runs the shell the way the native agent does (bash -lc, LC_ALL=C.UTF-8)."""
        from codex_cli_bin import bundled_codex_path
        executable = bundled_codex_path().resolve()
        with short_temporary_directory() as temp:
            root = Path(temp)
            actor, home, ipc, private, source = (root / name for name in ("actor", "codex", "ipc", "private", "snapshot"))
            for path in (actor, home, ipc, private, source): path.mkdir(mode=0o700)
            (source / "allowed.py").write_text("def published_marker():\n    return 1\n")
            (actor / "notes.md").write_text("first: one\nsecond: two\n")
            sandbox = CodexSandbox(private, ipc, actor, home, executable, read_paths=[(source, Path("/harness"))])
            self.assertEqual(sandbox.rg_executable, executable.parent.parent / "codex-path" / "rg")
            prefix, env = sandbox.prepare()

            def shell(script):
                result = subprocess.run(prefix + ["/usr/bin/env", "LC_ALL=C.UTF-8", "/bin/bash", "-lc", script],
                                        env={**os.environ, **env}, capture_output=True, text=True, timeout=20)
                return result.returncode, result.stdout, result.stderr
            try:
                for script, code, stdout in (
                        ("rg -n published_marker /harness", 0, "/harness/allowed.py:1:def published_marker():\n"),
                        ("grep -R -n published_marker /harness", 0, "/harness/allowed.py:1:def published_marker():\n"),
                        ("grep -c second notes.md", 0, "1\n"),
                        ("python -c 'print(6 * 7)'", 0, "42\n"),
                        ("printf 'a\\nb\\n' > x; printf 'a\\nc\\n' > y; diff x y > d; echo $?; cmp -s x x; echo $?", 0, "1\n0\n"),
                        ("find /harness -name '*.py' | xargs echo", 0, "/harness/allowed.py\n"),
                        ("seq 3 | tac | paste -sd, -", 0, "3,2,1\n"),
                        ("for name in curl ssh sudo setsid perl; do command -v $name; done; true", 0, "")):
                    with self.subTest(script=script):
                        self.assertEqual(shell(script), (code, stdout, ""))
            finally:
                sandbox.close()

    def test_host_environment_is_cleared_before_namespace_processes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); actor, home, ipc, private = (root / name for name in ("actor", "codex", "ipc", "private"))
            for path in (actor, home, ipc, private): path.mkdir(mode=0o700)
            sandbox = CodexSandbox(private, ipc, actor, home, self._python(), packages=False)
            prefix, env = sandbox.prepare()
            env_tool = shutil.which("env", path=os.defpath)
            self.assertEqual(prefix[:2], [env_tool, "-i"])
            secret_name = "NI_RUA_SYNTHETIC_HOST_ENV_SENTINEL"
            secret_value = "synthetic-secret-must-not-cross"
            code = ("from pathlib import Path\n"
                    f"needle={secret_name + '=' + secret_value!r}.encode()\n"
                    "forbidden=False\n"
                    "for f in Path('/proc').glob('*/environ'):\n"
                    " try:\n"
                    "  if needle in f.read_bytes(): forbidden=True\n"
                    " except OSError: pass\n"
                    "assert not forbidden")
            try:
                result = subprocess.run(prefix + [str(sandbox.python_runtime), "-c", code],
                                        env={**os.environ, **env, secret_name: secret_value},
                                        capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)
            finally:
                sandbox.close()

    def test_loopback_port_is_relayed_through_run_owned_unix_socket(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); actor, home, ipc, private = (root / name for name in ("actor", "codex", "ipc", "private"))
            for path in (actor, home, ipc, private): path.mkdir(mode=0o700)
            service = socket.socket(); service.bind(("127.0.0.1", 0))
            port = service.getsockname()[1]
            received_ping = threading.Event()
            # 共享开发机上外部端口扫描器可能发送 HTTP 探测，仅对沙盒内发来的 ping 协议包应答
            def respond():
                while not received_ping.is_set():
                    try:
                        client, _ = service.accept()
                    except OSError:
                        return
                    with client:
                        client.settimeout(2)
                        try:
                            data = client.recv(4)
                        except OSError:
                            continue
                        if data == b"ping":
                            client.sendall(b"pong")
                            received_ping.set()
                            return
            sandbox = CodexSandbox(private, ipc, actor, home, self._python(), ports=[port], packages=False)
            prefix, env = sandbox.prepare()
            service.listen()
            worker = threading.Thread(target=respond, daemon=True); worker.start()
            code = ("import socket; s=socket.create_connection(('127.0.0.1', " + str(port) + ")); "
                    "s.sendall(b'ping'); assert s.recv(4)==b'pong'; s.close()")
            try:
                result = subprocess.run(prefix + [str(sandbox.python_runtime), "-c", code], env={**os.environ, **env},
                                                        capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)
                worker.join(2)
                self.assertTrue(received_ping.is_set())
            finally:
                sandbox.close(); service.close()

    def test_parent_termination_does_not_leave_native_descendant(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); actor, home, ipc, private = (root / name for name in ("actor", "codex", "ipc", "private"))
            for path in (actor, home, ipc, private): path.mkdir(mode=0o700)
            pid_file = actor / "child.pid"
            sandbox = CodexSandbox(private, ipc, actor, home, self._python(), packages=False)
            prefix, env = sandbox.prepare()
            marker = "ni-rua-sandbox-child-" + root.name
            code = (f"import os,time,pathlib; marker={marker!r}; "
                    "pathlib.Path(os.environ['HOME'],'child.pid').write_text(str(os.getpid())); "
                    "time.sleep(60)")
            process = subprocess.Popen(prefix + [str(sandbox.python_runtime), "-c", code],
                                       env={**os.environ, **env}, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL)
            try:
                deadline = __import__("time").monotonic() + 5
                while not pid_file.exists() and __import__("time").monotonic() < deadline:
                    __import__("time").sleep(.02)
                self.assertTrue(pid_file.exists(), "sandbox child did not start")
                self.assertTrue(self._marked_processes(marker), "sandbox descendant was not found from host")
                process.terminate(); process.wait(timeout=5)
                deadline = __import__("time").monotonic() + 3
                while self._marked_processes(marker) and __import__("time").monotonic() < deadline:
                    __import__("time").sleep(.02)
                self.assertFalse(self._marked_processes(marker), "sandbox child survived its parent")
            finally:
                if process.poll() is None:
                    process.kill(); process.wait()
                sandbox.close()

    def test_owner_listener_accepts_repeated_sandbox_processes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); actor, home, ipc, private = (root / name for name in ("actor", "codex", "ipc", "private"))
            for path in (actor, home, ipc, private): path.mkdir(mode=0o700)
            sandbox = CodexSandbox(private, ipc, actor, home, self._python(), packages=False)
            prefix, env = sandbox.prepare()
            try:
                for _ in range(4):
                    result = subprocess.run(prefix + [str(sandbox.python_runtime), "-c", "pass"],
                                            env={**os.environ, **env}, capture_output=True, text=True,
                                            timeout=10)
                    self.assertEqual(result.returncode, 0, result.stderr)
            finally:
                sandbox.close()

    @unittest.skipUnless(hasattr(os, "fork"), "requires fork for owner-death lifecycle probe")
    def test_owner_sigkill_closes_guardian_and_kills_native_descendant(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); actor, home, ipc, private = (root / name for name in ("actor", "codex", "ipc", "private"))
            for path in (actor, home, ipc, private): path.mkdir(mode=0o700)
            pid_file = actor / "owner-child.pid"
            sandbox = CodexSandbox(private, ipc, actor, home, self._python(), packages=False)
            prefix, env = sandbox.prepare()
            marker = "ni-rua-owner-death-child-" + root.name
            code = (f"import time,pathlib; marker={marker!r}; "
                    "pathlib.Path(__import__('os').environ['HOME'],'owner-child.pid').write_text(str(__import__('os').getpid())); "
                    "time.sleep(60)")
            read_fd, write_fd = os.pipe()
            owner_pid = os.fork()
            if owner_pid == 0:
                os.close(read_fd)
                child = subprocess.Popen(prefix + [str(sandbox.python_runtime), "-c", code],
                                         env={**os.environ, **env}, stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL)
                deadline = __import__("time").monotonic() + 5
                while not pid_file.exists() and __import__("time").monotonic() < deadline:
                    __import__("time").sleep(.02)
                os.write(write_fd, b"ready" if pid_file.exists() else b"failed")
                os.close(write_fd)
                __import__("time").sleep(60)
                os._exit(0)
            os.close(write_fd)
            try:
                self.assertTrue(select.select([read_fd], [], [], 8)[0], "owner did not start sandbox child")
                self.assertEqual(os.read(read_fd, 16), b"ready")
                os.close(read_fd)
                self.assertTrue(self._marked_processes(marker))
                # The parent process owns this listening FD. The forked owner
                # inherited the listener and accepted connections. Close the
                # test process's inherited copies after killing that owner so
                # EOF reaches the sandbox watchdog.
                os.kill(owner_pid, 9); os.waitpid(owner_pid, 0)
                sandbox.owner_listener.close()
                sandbox.owner_listener = None
                with sandbox.owner_connections_lock:
                    inherited_connections = tuple(sandbox.owner_connections)
                    sandbox.owner_connections.clear()
                for connection in inherited_connections:
                    connection.close()
                deadline = __import__("time").monotonic() + 5
                while self._marked_processes(marker) and __import__("time").monotonic() < deadline:
                    __import__("time").sleep(.02)
                self.assertFalse(self._marked_processes(marker), "native descendant survived owner SIGKILL")
            finally:
                try: os.kill(owner_pid, 9)
                except ProcessLookupError: pass
                try: os.waitpid(owner_pid, 0)
                except ChildProcessError: pass
                try: os.close(read_fd)
                except OSError: pass
                sandbox.close()

    @staticmethod
    def _marked_processes(marker):
        matches = []
        for entry in Path("/proc").glob("[0-9]*"):
            try:
                if marker.encode() in (entry / "cmdline").read_bytes(): matches.append(entry.name)
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
        return matches


class RipgrepSourceTests(unittest.TestCase):
    """Which ``rg`` the sandbox publishes (constructor only: no namespace is started)."""

    def sandbox(self, root, executable, **options):
        return CodexSandbox(root / "private", root / "ipc", root / "actor", root / "home", executable, **options)

    def test_the_distributions_own_rg_is_published_whatever_the_host_has(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            package = root / "package"
            for directory in ("bin", "codex-path"): (package / directory).mkdir(parents=True)
            executable, bundled = package / "bin" / "codex", package / "codex-path" / "rg"
            for path in (executable, bundled): path.write_text("#!/bin/sh\n"); path.chmod(0o755)
            from ni_rua.cortex.agents.codex.sandbox import distribution_ripgrep, distribution_tool_directory
            self.assertEqual(distribution_tool_directory(executable), package / "codex-path")
            self.assertEqual(distribution_ripgrep(executable), bundled)
            for host_rg in (None, str(root / "host-rg")):   # a host without ripgrep, and one with another rg
                with self.subTest(host_rg=host_rg), patch("ni_rua.cortex.session.agent_sandbox.shutil.which",
                                                         lambda name, path=None, host_rg=host_rg: host_rg):
                    self.assertEqual(self.sandbox(root, executable).rg_executable, bundled)
                    self.assertEqual(self.sandbox(root, executable, packages=False).rg_executable, bundled)

    def test_a_bundled_rg_that_is_not_executable_is_not_preferred_to_the_host_rg(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            package = root / "package"
            for directory in ("bin", "codex-path"): (package / directory).mkdir(parents=True)
            executable, bundled, host_rg = package / "bin" / "codex", package / "codex-path" / "rg", root / "host-rg"
            for path in (executable, bundled, host_rg): path.write_text("#!/bin/sh\n"); path.chmod(0o755)
            bundled.chmod(0o644)
            with patch("ni_rua.cortex.session.agent_sandbox.shutil.which", lambda name, path=None: str(host_rg)):
                self.assertEqual(self.sandbox(root, executable).rg_executable, host_rg)

    def test_an_executable_without_a_bundled_rg_falls_back_to_the_host_rg(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            (root / "bin").mkdir()
            executable, host_rg = root / "bin" / "codex", root / "host-rg"
            for path in (executable, host_rg): path.write_text("#!/bin/sh\n"); path.chmod(0o755)
            with patch("ni_rua.cortex.session.agent_sandbox.shutil.which", lambda name, path=None: str(host_rg)):
                self.assertEqual(self.sandbox(root, executable).rg_executable, host_rg)
                self.assertIsNone(self.sandbox(root, executable, packages=False).rg_executable)
            with patch("ni_rua.cortex.session.agent_sandbox.shutil.which", lambda name, path=None: None):
                self.assertIsNone(self.sandbox(root, executable).rg_executable)

    def test_unsandboxed_path_starts_with_the_distributions_tool_directory(self):
        from ni_rua.cortex.agents.codex.session import native_tool_path
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            for directory in ("with/bin", "with/codex-path", "without/bin"): (root / directory).mkdir(parents=True)
            self.assertEqual(native_tool_path(root / "with" / "bin" / "codex"),
                             str(root / "with" / "codex-path") + os.pathsep + os.defpath)
            self.assertEqual(native_tool_path(root / "without" / "bin" / "codex"), os.defpath)


class PackageProxyTests(unittest.TestCase):
    @staticmethod
    def _exchange(request, fake_upstream=None, package_port=20123):
        with tempfile.TemporaryDirectory() as temp:
            relay = _UnixForwarder(Path(temp) / "packages", 0, packages=True, package_port=package_port)
            client, server = socket.socketpair()
            try:
                if fake_upstream is None:
                    worker = threading.Thread(target=PackageProxyTests._serve_test_client,
                                              args=(relay, server), daemon=True)
                else:
                    with patch.object(_UnixForwarder, "_https_request", new=staticmethod(fake_upstream)):
                        worker = threading.Thread(target=PackageProxyTests._serve_test_client,
                                                  args=(relay, server), daemon=True)
                        worker.start()
                        client.sendall(request)
                        response = bytearray()
                        while chunk := client.recv(65536): response.extend(chunk)
                        worker.join(2)
                        return bytes(response)
                worker.start()
                client.sendall(request)
                response = bytearray()
                while chunk := client.recv(65536): response.extend(chunk)
                worker.join(2)
                return bytes(response)
            finally:
                client.close(); server.close()

    @staticmethod
    def _serve_test_client(relay, client):
        try:
            relay._package_connection(client)
        finally:
            client.close()

    class _Response:
        def __init__(self, body, status=200, headers=None):
            self.body, self.status, self.headers = BytesIO(body), status, headers or {}

        def read(self, size=-1): return self.body.read(size)
        def getheader(self, name, default=None): return self.headers.get(name, default)

    class _Connection:
        def close(self): pass

    def test_allowed_simple_index_rewrites_only_file_download_origin(self):
        html = b'<a href="https://files.pythonhosted.org/packages/x/demo-1.0-py3-none-any.whl#sha256=abc">wheel</a>'
        response = self._Response(html, headers={"Content-Type": "text/html; charset=UTF-8"})
        observed = []
        def upstream(host, path, headers):
            observed.append((host, path))
            return self._Connection(), response
        port = 20123
        request = f"GET /simple/Demo_1/ HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n".encode()
        result = self._exchange(request, upstream, package_port=port)
        self.assertIn(b"200 OK", result)
        self.assertIn(f"http://127.0.0.1:{port}/files/packages/x/demo-1.0-py3-none-any.whl".encode(), result)
        self.assertEqual(observed, [("pypi.org", "/simple/demo-1/")])

    def test_allowed_file_fetch_is_streamed_from_fixed_host(self):
        body = b"wheel-bytes"
        response = self._Response(body, headers={"Content-Length": str(len(body)),
                                                  "Content-Type": "application/octet-stream"})
        observed = []
        def upstream(host, path, headers):
            observed.append((host, path))
            return self._Connection(), response
        port = 20123
        request = f"GET /files/packages/ab/demo-1.whl HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n".encode()
        result = self._exchange(request, upstream, package_port=port)
        self.assertIn(b"200 OK", result)
        self.assertTrue(result.endswith(body))
        self.assertEqual(observed, [("files.pythonhosted.org", "/packages/ab/demo-1.whl")])

    def test_host_method_and_path_allowlist_rejections(self):
        port = 20123
        bad_requests = [
            (f"POST /simple/demo/ HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Length: 0\r\n\r\n".encode(), b"405"),
            (b"GET /simple/demo/ HTTP/1.1\r\nHost: example.com\r\n\r\n", b"405"),
            (f"GET http://example.com/ HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n".encode(), b"403"),
            (f"GET /files/packages/../../etc/passwd.whl HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n".encode(), b"403"),
            (f"GET /other/path HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n".encode(), b"404"),
        ]
        for request, status in bad_requests:
            with self.subTest(request=request.split(b" ", 1)[0:2]):
                self.assertIn(status, self._exchange(request, package_port=port))

    def test_private_dns_and_redirects_fail_closed(self):
        with patch("ni_rua.cortex.agents.codex.sandbox.socket.getaddrinfo", return_value=[(socket.AF_INET,
                socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]):
            with self.assertRaisesRegex(OSError, "dns_rejected"):
                _UnixForwarder._https_request("pypi.org", "/simple/demo/", {})

        class RawSocket:
            def settimeout(self, *_): pass
            def connect(self, *_): pass
            def close(self): pass
        class SecureContext:
            def wrap_socket(self, raw, *, server_hostname):
                self.server_hostname = server_hostname
                return raw
        class Redirect:
            status = 302
        class Connection:
            def __init__(self, host, timeout): self.host = host
            def request(self, *_args, **_kwargs): pass
            def getresponse(self): return Redirect()
            def close(self): pass
        with (patch("ni_rua.cortex.agents.codex.sandbox.socket.getaddrinfo", return_value=[(socket.AF_INET,
                    socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))]),
              patch("ni_rua.cortex.agents.codex.sandbox.socket.socket", return_value=RawSocket()),
              patch("ni_rua.cortex.agents.codex.sandbox.ssl.create_default_context", return_value=SecureContext()),
              patch("ni_rua.cortex.agents.codex.sandbox.http.client.HTTPSConnection", Connection)):
            with self.assertRaisesRegex(OSError, "redirect_rejected"):
                _UnixForwarder._https_request("pypi.org", "/simple/demo/", {})


PROBE_ID, LOOK_ID = "call_sandbox_probe_28", "call_sandbox_look_28"
EXEC_YIELD_MS = 20000      # exec_command yield time, ms: the probe finishes in about a second
INSTRUCTIONS = "Codex sandbox conformance (item 28). Use the tools when asked and answer briefly."


class CodexResponsesServer(FakeResponsesServer):
    """``FakeResponsesServer`` whose ``exec_command`` calls carry Codex's ``functions`` namespace (as the native
    shell tool is published to the model)."""

    def _items(self, turn):
        items = super()._items(turn)
        for item in items:
            if item.get("type") == "function_call" and item.get("name") == "exec_command":
                item["namespace"] = "functions"
        return items


def _output_ids(request):
    body = request.json if isinstance(request.json, dict) else {}
    return {item.get("call_id") for item in body.get("input") or ()
            if isinstance(item, dict) and item.get("type") == "function_call_output"}


@requires_codex_binary
@requires_bwrap
class CodexSandboxConformance(unittest.TestCase):
    def test_sandbox_namespace(self):
        from ni_rua.cortex.agents.factory import backend_requirements
        from ni_rua.cortex.agents.isolation import process_tree_snapshot
        from ni_rua.cortex.session.agent_sandbox import ACTOR_ISOLATION
        from ni_rua.cortex.agents.codex.session import CodexSession
        directory = self.enterContext(tempfile.TemporaryDirectory(prefix="ni-rua-c28-codex-"))
        root = Path(directory).resolve() / "run"
        root.mkdir()
        boundary = SandboxBoundary(self, root)          # before the session: it fixes XDG_RUNTIME_DIR
        probe_command = []

        def router(request):
            outputs = _output_ids(request)
            if LOOK_ID in outputs:
                return text_step("SANDBOX-DONE-28")
            if PROBE_ID in outputs:
                return Step(tool_calls=[ToolCall("look", {"x": 1.5}, LOOK_ID)])
            if not probe_command:
                return text_step("side-ok")
            return Step(tool_calls=[ToolCall("exec_command", dict(cmd=probe_command[0], yield_time_ms=EXEC_YIELD_MS,
                                                                  max_output_tokens=2000), PROBE_ID)])

        server = CodexResponsesServer([], default=router, gate_timeout_s=60).start()
        self.addCleanup(server.close)
        source = root.parent / "codex-source"
        source.mkdir()
        (source / "config.toml").write_text(
            'model_provider="fixture"\n[model_providers.fixture]\nname="Loopback fixture"\n'
            f'base_url="{server.base_url}/v1"\nwire_api="responses"\nrequires_openai_auth=false\n'
            'supports_websockets=false\n')
        store = Store(root, dict(run_id="codex-sandbox-28", episode_id="ep", env_instance_id="env", env_generation=1))
        self.addCleanup(store.close)
        config = Config.from_dict({"model": {"codex_home": str(source), "native_memories": False},
                                   "features": {"sandbox_enabled": True, "self_briefing": False,
                                                "visual_review_model": None}})
        self.assertIsNone(backend_requirements(config))
        calls = []

        def handler(params):
            calls.append(params)
            return {"success": True, "contentItems": [{"type": "inputText", "text": "scene: one red cube at x=1.5"}]}

        session = CodexSession(root, store, handler, config=config)
        closed = []

        def close():
            if closed:
                return
            closed.append(True)
            bridge = session.client._proc if session.client else None
            session.close()
            if bridge:
                for stream in (bridge.stdin, bridge.stdout, bridge.stderr):
                    if stream and not stream.closed:
                        stream.close()
        self.addCleanup(close)
        session.start([TOOLS[0]], INSTRUCTIONS)
        # (f) recorded isolation
        self.assertEqual(session.isolation, ACTOR_ISOLATION)
        self.assertEqual(store.read_state("runtime", "agent")["isolation"], ACTOR_ISOLATION)
        self.assertEqual(store.read_state("runtime", "codex")["isolation"], ACTOR_ISOLATION)
        self.assertEqual(session.provenance()["isolation"], ACTOR_ISOLATION)
        relay_port = int(session.model_proxy.base_url.rsplit(":", 1)[1].split("/")[0])
        boundary.write_probe(session.actor_dir, [relay_port])
        probe_command.append(boundary.command)
        session.run_turn("Probe the sandbox, then look at the scene.")
        self.assertTrue(session.turn_done.wait(180), "the sandboxed turn did not end")
        self.assertEqual(session.state, "completed", session.turn_result)
        # (a)-(c) measured inside the namespace by the native shell tool
        self.assertTrue(boundary.wait_for(lambda: (session.actor_dir / boundary.report_name).is_file(), 30))
        outputs = [result.text for request in server.generation_requests() for result in server.find_tool_results(request)
                   if result.call_id == PROBE_ID]
        self.assertTrue(any(SANDBOX_PROBE_MARKER in text for text in outputs), outputs)
        boundary.assert_inside(boundary.report(), root / "private" / "actor-source")
        # (d) the registry tool round trip through the dynamic-tool path and the relay
        (call,) = calls
        self.assertEqual((call["threadId"], call["tool"], call["arguments"]), (session.thread_id, "look", {"x": 1.5}))
        looked = [result.text for request in server.generation_requests() for result in server.find_tool_results(request)
                  if result.call_id == LOOK_ID]
        self.assertTrue(any("scene: one red cube at x=1.5" in text for text in looked), looked)
        # (e) teardown
        tree = [member.pid for _role, process in session.owned_processes()
                for member in process_tree_snapshot(process.pid).members]
        tracked = boundary.assert_running(tree)
        close()
        boundary.assert_closed(tracked)


if __name__ == "__main__":
    unittest.main()
