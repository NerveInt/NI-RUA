"""
- test_program_isolation.py: Kernel-enforced program boundary; no model/GPU, only owned test channels.
- test_program_limits_unsandboxed.py: program_run without the OS sandbox (the macOS default): resource limits must not abort launch.
"""
from support.paths import REPO_ROOT
from support.program_fixture import program_services
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import tomllib
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from codex_cli_bin import bundled_codex_path
from ni_rua.cortex.agents.codex.session import ACTOR_ISOLATION, CodexSession
from ni_rua.core.config import Config
from ni_rua.runtime.programs.runner import ProgramRunner
from ni_rua.core.storage.store import Store
from support.host_capabilities import bubblewrap_namespaces, program_isolation, require
import json
import sys
from unittest.mock import Mock, patch
from ni_rua.runtime.programs.isolation import CREATE_RULESET, restrict
from support.host_capabilities import landlock, require
import importlib.util
import sysconfig
from unittest import mock
from ni_rua.runtime.harness import Harness
from ni_rua.runtime.programs import isolation
from ni_rua.runtime.programs.runner import ProgramRunner, program_runtime
from runtime.harness.test_handoff_and_candidates import WaitFixtureRobot
from support.host_capabilities import address_space_limit, expect, program_isolation, require
import contextlib
from dataclasses import replace
import errno
import resource
import select
import signal
import time
from support.host_capabilities import address_space_limit, expect
from ni_rua.runtime.programs import runner as runner_module

NI_RUA_TEST_TIER = "component"


# --- Folded from tests/test_actor_isolation.py ---

"""Codex namespace and program boundaries; synthetic files, no GPU/model calls."""




IDENTITY = dict(run_id="isolation-fixture", episode_id="ep", env_instance_id="env", env_generation=1)


class ActorIsolationTest(unittest.TestCase):
    def test_disabled_sandbox_keeps_private_config_and_uses_host_execution(self):
        from ni_rua.cortex.context.actor_instructions import instructions
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = Store(root / 'run', IDENTITY)
            profile = root / 'profile'
            profile.mkdir()
            (profile / 'config.toml').write_text('model_provider="fixture"\n'
                '[model_providers.fixture]\nbase_url="http://127.0.0.1:1/v1"\n')
            config = Config.from_dict({'model': {'codex_home': str(profile)},
                                      'features': {'sandbox_enabled': False}})
            session = CodexSession(store.root, store, lambda _: {}, config=config)
            try:
                with patch('ni_rua.cortex.agents.codex.sandbox.CodexSandbox', side_effect=AssertionError('sandbox used')):
                    session._prepare()
                self.assertIsNone(session.sandbox)
                self.assertEqual(session.command_prefix, [])
                self.assertEqual(session.command[0], str(bundled_codex_path().resolve()))
                self.assertEqual(session.env['HOME'], str(session.actor_dir))
                self.assertEqual(session.env['CODEX_HOME'], str(session.codex_home))
                host_file = root / 'synthetic-host-file'
                host_file.write_text('host-visible')
                result = subprocess.run(session.command_prefix + ['python3', '-c',
                    f'from pathlib import Path; print(Path({str(host_file)!r}).read_text())'],
                    env=session.env, cwd=session.actor_dir, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), 'host-visible')
                text = instructions('observation fixture', 'handoff fixture',
                                    source_root=session.private / 'actor-source')
                self.assertIn(str(session.private / 'actor-source'), text)
                self.assertTrue((session.private / 'actor-source/ni_rua/runtime/harness.py').is_file())
                self.assertIn('without OS sandbox isolation', text)
                self.assertNotIn('inside your sandbox', text)
                self.assertNotIn('general network access are unavailable', text)
            finally:
                session.close()
                store.close()

    def test_resume_rejects_sandbox_mode_changes_before_starting_processes(self):
        for enabled, previous in ((True, 'disabled'), (False, ACTOR_ISOLATION)):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as temporary:
                store = Store(temporary, IDENTITY)
                session = CodexSession(store.root, store, lambda _: {},
                    config=Config.from_dict({'features': {'sandbox_enabled': enabled}}))
                store.set_state('runtime', 'codex', {'isolation': previous})
                try:
                    with patch.object(session, '_stop_previous_owned_processes') as stop:
                        with self.assertRaisesRegex(ValueError, 'resume_isolation_changed'):
                            session.start([], 'fixture', resume_thread_id='fixture-thread')
                        stop.assert_not_called()
                    with self.assertRaisesRegex(ValueError, 'resume_isolation_changed'):
                        session._prepare()
                finally:
                    session.close()
                    store.close()

    @require(bubblewrap_namespaces, "starts the native Codex actor inside its bubblewrap namespace sandbox")
    def test_generated_codex_namespace_allows_workspace_but_denies_host_files_and_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = Store(root / "run", IDENTITY)
            profile = root / "profile"
            profile.mkdir()
            (profile / "config.toml").write_text('model_provider = "fixture"\n'
                '[model_providers.fixture]\nbase_url = "http://127.0.0.1:1/v1"\n')
            session = CodexSession(store.root, store, lambda params: None,
                                   config=Config.from_dict({"features": {"sandbox_enabled": True},
                                                            "model": {"codex_home": str(profile)}}))
            external_file = root / "outside-run.txt"
            external_file.write_text("synthetic ordinary file")
            output = session.actor_dir / "workspace-output.txt"
            try:
                session._prepare()
                config = tomllib.loads((session.codex_home / "config.toml").read_text())
                self.assertTrue(config["features"]["shell_tool"])
                self.assertTrue(config["features"]["unified_exec"])
                self.assertEqual(config["approval_policy"], "never")
                self.assertEqual(config["sandbox_mode"], "danger-full-access")
                self.assertEqual(session.command[-5:], [str(bundled_codex_path().resolve()),
                    "--no-daemon", "app-server", "--listen", f"unix://{session.socket_path}"])
                self.assertGreater(len(session.command), 5)
                self.assertEqual(session.env["HOME"], str(session.actor_dir))
                self.assertEqual(session.env["CODEX_HOME"], str(session.codex_home))
                self.assertFalse((session.private / "isolation.json").exists())
                permitted = session.actor_dir / 'notes.txt'
                permitted.write_text('synthetic workspace note')
                shortcut = session.actor_dir / 'host-link'
                shortcut.symlink_to(external_file)
                hidden = [external_file, profile / 'config.toml', store.root / 'events.sqlite', shortcut,
                          REPO_ROOT / 'ni_rua' / 'robot' / 'backends' / 'robodojo' / 'robot_worker.py',
                          Path('/harness/ni_rua/robot/backends/robodojo/robot_worker.py')]
                host_network_namespace = os.readlink('/proc/self/ns/net')
                host_pid_namespace = os.readlink('/proc/self/ns/pid')
                with socket.socket() as local_server:
                    local_server.bind(("127.0.0.1", 0))
                    local_server.listen()
                    source = f"""from pathlib import Path
import os, socket, tempfile
assert os.readlink('/proc/self/ns/net') != {host_network_namespace!r}
assert os.readlink('/proc/self/ns/pid') != {host_pid_namespace!r}
assert Path({str(permitted)!r}).read_text() == 'synthetic workspace note'
Path({str(output)!r}).write_text('synthetic workspace write succeeded')
with tempfile.TemporaryDirectory() as temporary:
    Path(temporary, 'probe').write_text('private tmp is writable')
assert Path('/harness/ni_rua/runtime/harness.py').read_text()
for path in {list(map(str, hidden))!r}:
    try:
        Path(path).read_bytes()
    except (PermissionError, FileNotFoundError):
        continue
    raise AssertionError('host file readable: ' + path)
try:
    socket.create_connection(('127.0.0.1', {local_server.getsockname()[1]}), timeout=1)
except OSError:
    pass
else:
    raise AssertionError('host network reachable')
with socket.socket() as listener:
    listener.bind(('127.0.0.1', 0))
    listener.listen()
    assert listener.getsockname()[1] > 0
print('codex_namespace_boundary_passed')
"""
                    result = subprocess.run(session.command[:-5] + ['python3', '-c', source], env=session.env,
                                            cwd=session.actor_dir, capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
                self.assertEqual(result.stdout.strip(), b"codex_namespace_boundary_passed")
                self.assertEqual(output.read_text(), "synthetic workspace write succeeded")
            finally:
                session.close()
                if hasattr(session, "daemon_socket"):
                    session.daemon_socket.with_suffix(".lock").unlink(missing_ok=True)
                store.close()

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_program_channel_denies_observer_but_preserves_actor_derived_internal_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "run", IDENTITY)
            harness = SimpleNamespace(store=store, lock=threading.RLock(), cancel=threading.Event(), active_context=None)
            runner = ProgramRunner(program_services(harness))
            actor_ref = store.put_artifact(b"visible RGB fixture", "rgb", "image/png")
            internal_ref = store.put_artifact(b"actor-derived geometry fixture", "scene_geometry", "text/plain", "internal")
            observer_ref = store.put_artifact(b"synthetic camera truth", "camera_diagnostic", "application/json", "observer")
            _, blob = store.artifact(observer_ref)
            source = f"""from pathlib import Path
import os
from ni_rua_client import NiRuaClient
os.chdir(os.environ['HOME'])
client = NiRuaClient()
assert client.artifact({actor_ref!r}) == b'visible RGB fixture'
assert client.artifact({internal_ref!r}) == b'actor-derived geometry fixture'
try:
    client.artifact({observer_ref!r})
except RuntimeError as error:
    assert str(error) == 'artifact_access_scope_denied'
else:
    raise AssertionError('observer artifact readable through broker')
Path('truth-link').symlink_to({str(blob)!r})
for path in [{str(blob)!r}, 'truth-link', {str(store.root / 'events.sqlite')!r}]:
    try:
        Path(path).read_bytes()
    except PermissionError:
        continue
    raise AssertionError('observer data readable through filesystem')
try:
    client.call('observer_evaluate')
except RuntimeError as error:
    assert str(error) == 'program_tool_not_permitted'
else:
    raise AssertionError('observer operation callable')
print('program_boundary_passed')
"""
            try:
                result = runner.run(dict(source=source, timeout_s=10),
                                    dict(thread_id="thread", turn_id="turn", tool_call_id="isolation", item_id="item"))
                self.assertEqual(result["status"], "succeeded", store.resolve_artifact(result["stderr_ref"])[1])
                self.assertEqual(store.resolve_artifact(result["stdout_ref"])[1].strip(), b"program_boundary_passed")
            finally:
                runner.close()
                store.close()



# --- Folded from tests/test_isolation.py ---

class IsolationCheck(unittest.TestCase):
    def test_ruleset_rights_follow_kernel_abi(self):
        # Linux UAPI rights: REFER since ABI 2, TRUNCATE since 3, IOCTL_DEV since 5.
        for abi, rights in [(1, 0x1fff), (2, 0x3fff), (3, 0x7fff),
                            (4, 0x7fff), (5, 0xffff), (6, 0xffff)]:
            with self.subTest(abi=abi):
                libc = Mock()
                created = []

                def syscall(number, *args):
                    if number == CREATE_RULESET:
                        if args[-1] == 1:
                            return abi
                        attrs, size, _ = args
                        created.append((attrs._obj.handled_access_fs,
                                        attrs._obj.handled_access_net, size))
                        return 42
                    return 0

                libc.syscall.side_effect = syscall
                libc.prctl.return_value = 0
                with patch('ni_rua.runtime.programs.isolation.ctypes.CDLL', return_value=libc), \
                        patch('ni_rua.runtime.programs.isolation.os.close') as close:
                    self.assertEqual(restrict([], [], tcp_ports=[] if abi >= 4 else None), abi)
                self.assertEqual(created, [(rights, 3 if abi >= 4 else 0, 16 if abi >= 4 else 8)])
                close.assert_called_once_with(42)

    def test_tcp_isolation_still_requires_abi_four(self):
        libc = Mock()
        libc.syscall.return_value = 3
        with patch('ni_rua.runtime.programs.isolation.ctypes.CDLL', return_value=libc):
            with self.assertRaisesRegex(RuntimeError, 'ABI >=4 required'):
                restrict([], [], tcp_ports=[])
        libc.syscall.assert_called_once_with(CREATE_RULESET, 0, 0, 1)

    @require(landlock, "the program launcher denies reads through a real Landlock ruleset")
    def test_children_cannot_read_hidden_files(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            allowed = root / "actor"; allowed.mkdir()
            hidden = root / "hidden"; hidden.write_text("oracle")
            spec = root / "spec.json"
            spec.write_text(json.dumps(dict(read_paths=[p for p in ["/usr", "/bin", "/lib", "/lib64"] if Path(p).exists()],
                                            write_paths=[str(allowed)], env=dict(PATH="/usr/bin:/bin"))))
            command = f'echo permitted > "{allowed}/result"; cat "{hidden}"'
            result = subprocess.run([sys.executable, "-m", "ni_rua.runtime.programs.isolation", str(spec), "/bin/sh", "-c", command], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"Permission denied", result.stderr)
            self.assertEqual((allowed / "result").read_text(), "permitted\n")
            self.assertNotIn(b"oracle", result.stdout)
            with socket.socket() as service:
                service.bind(("127.0.0.1", 0)); service.listen()
                spec_value = json.loads(spec.read_text()); spec_value["tcp_ports"] = [443]
                spec.write_text(json.dumps(spec_value))
                program = f'import socket; socket.create_connection(("127.0.0.1", {service.getsockname()[1]}))'
                result = subprocess.run([sys.executable, "-m", "ni_rua.runtime.programs.isolation", str(spec), "/usr/bin/python3", "-c", program], capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(b"Permission denied", result.stderr)



# --- Folded from tests/test_program_isolation.py ---

@require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
class ProgramIsolationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.actor = self.root/'actor'
        self.actor.mkdir()
        runtime, packages, read_paths = program_runtime(self.root / 'runtime')
        self.spec = self.root/'isolation.json'
        self.spec.write_text(json.dumps(dict(read_paths=read_paths,write_paths=[str(self.actor),'/dev/null'],tcp_ports=[],deny_socket_creation=True,bounded_program=True,
                                            env=dict(PATH=os.defpath,HOME=str(self.actor),TMPDIR=str(self.actor),PYTHONHOME=str(runtime),PYTHONPATH=str(packages),PYTHONDONTWRITEBYTECODE='1'))))

    def tearDown(self):
        self.temporary.cleanup()

    def run_isolated(self,source,pass_fds=()):
        result = subprocess.run([sys.executable,'-m','ni_rua.runtime.programs.isolation',str(self.spec),sys.executable,'-S','-c',source],capture_output=True,pass_fds=pass_fds,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr.decode(errors='replace'))
        return result.stdout

    def test_actual_program_retains_computation_but_denies_host_and_proc(self):
        store = Store(self.root / 'run', dict(run_id='runtime-fixture', episode_id='ep', env_instance_id='env', env_generation=1))
        harness = SimpleNamespace(store=store, lock=threading.RLock(), cancel=threading.Event(), active_context=None)
        runner = ProgramRunner(program_services(harness))
        hidden = self.root / 'host-secret.txt'
        hidden.write_text('not actor data')
        host_packages = Path(importlib.util.find_spec('numpy').origin).parent.parent.resolve()
        rejected = [str(hidden), str(Path(__file__).resolve()),
                    '/proc/self/environ', '/proc/self/fd', '/proc/self/mem', '/proc/' + str(os.getpid()) + '/mem']
        rejected.extend(str(path) for path in host_packages.glob('__editable*') if path.is_file())
        source = f'''from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import cv2
assert float(np.linalg.norm([3., 4.])) == 5.
assert Image.new('RGB', (2, 2)).size == (2, 2)
assert Rotation.identity().as_matrix().shape == (3, 3)
assert cv2.cvtColor(np.zeros((2, 2, 3), dtype=np.uint8), cv2.COLOR_RGB2GRAY).shape == (2, 2)
Path('secret-link').symlink_to({str(hidden)!r})
for name in {rejected!r} + ['secret-link']:
    try:
        Path(name).read_bytes()
    except PermissionError:
        continue
    raise AssertionError('host file readable: ' + name)
print('selected_runtime_and_host_boundary_passed')
assert Path('/proc/self/fd/0').read_bytes() == b''
'''
        try:
            job = runner.run(dict(source=source, timeout_s=20),
                             dict(thread_id='thread', turn_id='turn', tool_call_id='runtime', item_id='item'))
            self.assertEqual(job['status'], 'succeeded', store.resolve_artifact(job['stderr_ref'])[1])
            self.assertIn(b'selected_runtime_and_host_boundary_passed', store.resolve_artifact(job['stdout_ref'])[1])
            policy = json.loads((store.root / 'private' / ('isolation-' + job['job_id'] + '.json')).read_text())
            self.assertNotIn(sys.prefix, policy['read_paths'])
            self.assertNotIn(str(host_packages), policy['read_paths'])
            for directory in ('/usr', '/usr/local', '/usr/bin', '/usr/lib', '/bin', '/lib', '/lib64'):
                self.assertNotIn(directory, policy['read_paths'])
        finally:
            runner.close()
            store.close()

    def test_runtime_rejects_links_into_other_host_content(self):
        stdlib = self.root / 'synthetic-stdlib'
        package = stdlib / 'encodings'
        package.mkdir(parents=True)
        hidden = self.root / 'outside.py'
        hidden.write_text('not runtime')
        (package / 'alias.py').symlink_to(hidden)
        paths = {'stdlib': str(stdlib), 'purelib': str(self.root / 'host-packages')}
        with mock.patch('ni_rua.runtime.programs.runner.sysconfig.get_path', side_effect=lambda name: paths[name]):
            with self.assertRaisesRegex((ValueError, RuntimeError), 'runtime_.*external_symlink'):
                program_runtime(self.root / 'bad-runtime')

    def test_computation_packages_follow_actual_imports_not_venv_install_destination(self):
        actual_get_path = sysconfig.get_path
        empty_site = self.root / 'empty-venv' / 'site-packages'
        empty_site.mkdir(parents=True)
        with mock.patch('ni_rua.runtime.programs.runner.sysconfig.get_path', side_effect=lambda name:
                        str(empty_site) if name == 'purelib' else actual_get_path(name)):
            runtime, packages, read_paths = program_runtime(self.root / 'inherited-runtime')
        for name in ('numpy', 'scipy', 'PIL', 'cv2'):
            actual = Path(importlib.util.find_spec(name).origin).parent.resolve()
            self.assertEqual((packages / name).resolve(), actual)
            self.assertIn(str(actual), read_paths)
            self.assertNotIn(str(actual.parent), read_paths)
        policy = json.loads(self.spec.read_text())
        policy['read_paths'] = read_paths
        policy['env'].update(PYTHONHOME=str(runtime), PYTHONPATH=str(packages))
        self.spec.write_text(json.dumps(policy))
        self.run_isolated("import numpy,scipy,cv2; from PIL import Image; assert numpy.linalg.norm([3,4])==5; assert Image.new('RGB',(1,1)).size==(1,1)")

    def test_program_cannot_redirect_host_log_publication_via_workspace_links(self):
        store = Store(self.root / 'log-run', dict(run_id='log-fixture', episode_id='ep', env_instance_id='env', env_generation=1))
        harness = SimpleNamespace(store=store, lock=threading.RLock(), cancel=threading.Event(), active_context=None)
        runner = ProgramRunner(program_services(harness))
        secret = self.root / 'private-secret.txt'
        secret.write_text('SYNTHETIC_HOST_SECRET_MUST_NOT_BE_PUBLISHED')
        # A main actor can replace any child of its writable actor mount. That
        # old programs path must no longer control gateway writes or log reads.
        (store.root / 'actor').mkdir(exist_ok=True)
        (store.root / 'actor' / 'programs').symlink_to(self.root)
        source = f'''from pathlib import Path
for name in ('stdout.txt', 'stderr.txt'):
    Path(name).symlink_to({str(secret)!r})
try:
    Path({str(runner.logs / 'attack-link')!r}).symlink_to({str(secret)!r})
except PermissionError:
    pass
else:
    raise AssertionError('program wrote host-owned log directory')
print('only_program_authored_stdout')
'''
        try:
            job = runner.run(dict(source=source, timeout_s=10),
                             dict(thread_id='thread', turn_id='turn', tool_call_id='log-symlink', item_id='item'))
            self.assertEqual(job['status'], 'succeeded', store.resolve_artifact(job['stderr_ref'])[1])
            stdout = store.resolve_artifact(job['stdout_ref'])[1]
            stderr = store.resolve_artifact(job['stderr_ref'])[1]
            self.assertEqual(stdout.strip(), b'only_program_authored_stdout')
            self.assertEqual(stderr, b'')
            self.assertNotIn(secret.read_bytes(), stdout + stderr)
            self.assertTrue(runner.root.is_relative_to(store.root / 'private'))
            self.assertFalse(any(path.is_dir() and path.name.startswith('program-') for path in self.root.iterdir()))
        finally:
            runner.close()
            store.close()

    def test_threads_and_inherited_channel_work_without_socket_creation(self):
        parent,child = socket.socketpair()
        try:
            parent.settimeout(2)
            parent.sendall(b'ping')
            self.run_isolated(f"import os,threading\nresult=[]\nt=threading.Thread(target=lambda:result.append('thread-ok'))\nt.start();t.join()\nassert result==['thread-ok']\nfd={child.fileno()}\nassert os.read(fd,4)==b'ping'\nos.write(fd,b'pong')\n",(child.fileno(),))
            self.assertEqual(parent.recv(4),b'pong')
        finally:
            parent.close();child.close()

    def test_new_unix_abstract_tcp_and_socketpair_are_denied(self):
        source="""import socket
for family in [socket.AF_UNIX,socket.AF_INET,socket.AF_INET6]:
 try:
  socket.socket(family,socket.SOCK_STREAM)
 except PermissionError:
  continue
 raise AssertionError('program created a socket')
try:
 socket.socketpair()
except PermissionError:
 pass
else:
 raise AssertionError('program created a socketpair')
"""
        self.run_isolated(source)

    def test_connect_cannot_reach_outside_path_or_abstract_namespace(self):
        names = [str(self.root/'outside-private.sock'),'\0ni-rua-isolation-'+self.root.name]
        for name in names:
            with self.subTest(abstract=name.startswith('\0')):
                listener = socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
                candidate = socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
                try:
                    listener.bind(name); listener.listen()
                    code = f"import socket\ns=socket.socket(fileno={candidate.fileno()})\ntry:\n s.connect({name!r})\nexcept PermissionError:\n pass\nelse:\n raise AssertionError('outside Unix connect succeeded')\n"
                    self.run_isolated(code,(candidate.fileno(),))
                finally:
                    candidate.close();listener.close()

    def test_fork_signals_cross_process_memory_and_fd_access_are_denied(self):
        source="""import ctypes,errno,os
try:
 child=os.fork()
except PermissionError:
 pass
else:
 if child==0: os._exit(99)
 os.waitpid(child,0)
 raise AssertionError('program forked an extra process')
libc=ctypes.CDLL(None,use_errno=True)
seccomp=ctypes.CDLL('libseccomp.so.2')
seccomp.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p]
for name,args,expected in [
 ('clone3',[0,0],errno.ENOSYS),
 ('kill',[-1,0],errno.EPERM),('tkill',[-1,0],errno.EPERM),('tgkill',[-1,-1,0],errno.EPERM),
 ('rt_sigqueueinfo',[-1,0,0],errno.EPERM),('rt_tgsigqueueinfo',[-1,-1,0,0],errno.EPERM),
 ('ptrace',[2,-1,0,0],errno.EPERM),('process_vm_readv',[-1,0,0,0,0,0],errno.EPERM),('process_vm_writev',[-1,0,0,0,0,0],errno.EPERM),
 ('pidfd_getfd',[-1,0,0],errno.EPERM),('pidfd_send_signal',[-1,0,0,0],errno.EPERM),
 ('io_uring_setup',[1,0],errno.EPERM)]:
 number=seccomp.seccomp_syscall_resolve_name(name.encode())
 if number<0: continue
 ctypes.set_errno(0)
 result=libc.syscall(ctypes.c_long(number),*(ctypes.c_long(value) for value in args))
 assert result==-1 and ctypes.get_errno()==expected,(name,result,ctypes.get_errno())
"""
        self.run_isolated(source)


class ProgramSandboxConfigTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = Store(self.root / 'run', dict(run_id='sandbox-config', episode_id='ep',
            env_instance_id='env', env_generation=1))
        self.addCleanup(self.store.close)
        self.identity = dict(thread_id='thread', turn_id='turn', tool_call_id='program')

    def runner(self, sandbox_enabled):
        config = Config.from_dict({'features': {'sandbox_enabled': sandbox_enabled,
            'observation_profile': 'rgb_proprio_v1'}})
        harness = Harness(self.store, WaitFixtureRobot(self.root), config=config)
        harness.start()
        runner = harness.programs = ProgramRunner(harness.program_services())
        self.addCleanup(runner.close)
        return harness, runner

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_enabled_sandbox_broker_computes_image_from_symlinked_interpreter(self):
        import io
        from PIL import Image

        harness, _ = self.runner(True)
        interpreter = self.root / 'venv' / 'bin' / 'python'
        interpreter.parent.mkdir(parents=True)
        interpreter.symlink_to(sys.executable)
        (interpreter.parent.parent / 'pyvenv.cfg').write_text(
            f'home = {Path(sys.executable).parent}\ninclude-system-site-packages = true\n')
        image = Image.new('RGB', (4, 4))
        image.paste((255, 0, 0), (0, 0, 2, 2))
        encoded = io.BytesIO()
        image.save(encoded, format='PNG')
        image_ref = self.store.put_artifact(encoded.getvalue(), 'rgb', 'image/png')
        observer_ref = self.store.put_artifact(b'synthetic private data', 'camera_diagnostic',
            'text/plain', 'observer')
        hidden = self.store.root / 'private' / 'host-only.txt'
        hidden.write_text('synthetic host data')
        source = f'''import io
from pathlib import Path
import numpy as np
from PIL import Image
from ni_rua_client import NiRuaClient
client = NiRuaClient()
assert client.call('robot_capabilities')['success'] is True
image = Image.open(io.BytesIO(client.artifact({image_ref!r})))
crop = np.asarray(image.crop((0, 0, 2, 2)))
assert crop.mean(axis=(0, 1)).tolist() == [255., 0., 0.]
assert np.linalg.norm([3., 4.]) == 5.
try:
    client.artifact({observer_ref!r})
except RuntimeError as error:
    assert str(error) == 'artifact_access_scope_denied'
else:
    raise AssertionError('observer artifact readable through broker')
for path in {str(hidden)!r}, {str(self.store.root / 'events.sqlite')!r}, '/proc/self/environ':
    try:
        Path(path).read_bytes()
    except PermissionError:
        continue
    raise AssertionError('private host file readable: ' + path)
print('isolated_broker_image_computation_passed')
'''
        with mock.patch('ni_rua.runtime.programs.runner.sys.executable', str(interpreter)):
            response = harness.dispatch('program_run', dict(source=source, timeout_s=20), self.identity)
        job = json.loads(response['contentItems'][0]['text'])
        self.assertEqual(job['status'], 'succeeded', job.get('stderr_text'))
        self.assertIn('isolated_broker_image_computation_passed', job['stdout_text'])
        self.assertEqual(job['calls'], 1)
        self.assertEqual(harness.robot.native_count, 0)
        self.assertTrue(job['stop_confirmed'])

    def test_disabled_sandbox_runs_real_broker_and_review_without_system_dependencies(self):
        harness, runner = self.runner(False)
        self.assertFalse(runner.services.sandbox_enabled)
        outside = self.root / 'ordinary-host-file.txt'
        outside.write_text('synthetic host content')
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            listener.listen()
            source = f'''from pathlib import Path
import resource, socket
import numpy as np
from ni_rua_client import NiRuaClient
assert Path({str(outside)!r}).read_text() == 'synthetic host content'
with socket.create_connection(('127.0.0.1', {listener.getsockname()[1]}), timeout=1):
    pass
print('rlimit_as', *resource.getrlimit(resource.RLIMIT_AS))
assert resource.getrlimit(resource.RLIMIT_FSIZE) == (8*1024**2, 8*1024**2)
assert np.linalg.norm([3., 4.]) == 5.
client = NiRuaClient()
assert client.call('robot_capabilities')['success'] is True
reply = client.call('robot_wait', {{'native_steps': 1, 'timeout_s': 1,
    'observation_id': {harness.latest['observation_id']!r}}})
assert reply['kind'] == 'action'
assert reply['data']['receipt']['status'] == 'completed'
print('broker_and_review_completed')
'''
            with (mock.patch('ni_rua.runtime.programs.runner.program_runtime', side_effect=AssertionError('program_runtime called')),
                  mock.patch('ni_rua.runtime.programs.runner.system_library_path', side_effect=AssertionError('seccomp lookup')),
                  mock.patch('ni_rua.runtime.programs.runner.runtime_library_paths', side_effect=AssertionError('ldd lookup'))):
                job = runner.run(dict(source=source, timeout_s=20), self.identity)
                self.assertEqual(job['status'], 'awaiting_review', job.get('stderr_text'))
                self.assertEqual(harness.robot.native_count, 0)
                self.assertEqual(job['pending_review']['kind'], 'robot_wait')
                final = runner.run(dict(operation='resume', job_id=job['job_id'],
                    review_id=job['pending_review']['review_id'], decision='approve'),
                    dict(self.identity, tool_call_id='approve'))
            self.assertIsNone(runner.runtime)
            self.assertEqual(final['status'], 'succeeded', final.get('stderr_text'))
            self.assertEqual(final['calls'], 2)
            self.assertEqual(harness.robot.native_count, 1)
            self.assertIn('broker_and_review_completed', final['stdout_text'])
            self.assertTrue(final['stop_confirmed'])
            spec = json.loads(next((self.store.root / 'private').glob('isolation-*.json')).read_text())
            if expect(address_space_limit, "unsandboxed programs run under a 1 GiB RLIMIT_AS cap"):  # capped
                self.assertEqual(spec['limits']['RLIMIT_AS'], 1024**3)
                self.assertIn(f'rlimit_as {1024**3} {1024**3}', final['stdout_text'])
                self.assertNotIn('unenforced_limits', spec)
            else:  # macOS: the cap is reported as unenforced, never silently dropped.
                self.assertNotIn('RLIMIT_AS', spec['limits'])
                self.assertIn('RLIMIT_AS', spec['unenforced_limits'])

        timed = runner.run(dict(source='import time\ntime.sleep(30)', timeout_s=1),
            dict(self.identity, tool_call_id='timeout'))
        self.assertEqual(timed['status'], 'cancelled')
        self.assertTrue(timed['stop_confirmed'])

    def test_enabled_sandbox_missing_dependency_fails_without_running_source(self):
        _, runner = self.runner(True)
        marker = self.root / 'must-not-run.txt'
        with mock.patch('ni_rua.runtime.programs.runner.system_library_path', side_effect=RuntimeError('missing-seccomp-fixture')):
            job = runner.run(dict(source=f'from pathlib import Path\nPath({str(marker)!r}).touch()'),
                self.identity)
        self.assertEqual(job['status'], 'failed')
        self.assertIn('missing-seccomp-fixture', job['reason'])
        self.assertTrue(job['stop_confirmed'])
        self.assertFalse(marker.exists())

    def test_enabled_launcher_rejects_landlock_or_seccomp_failure_without_exec(self):
        spec = self.root / 'launcher.json'
        spec.write_text(json.dumps(dict(read_paths=[], write_paths=[], deny_socket_creation=True,
            bounded_program=True, env={'HOME': str(self.root)})))
        for failing in ('restrict', 'restrict_program_syscalls'):
            with self.subTest(failing=failing), \
                    mock.patch.object(sys, 'argv', ['isolation.py', str(spec), sys.executable, '-c', 'pass']), \
                    mock.patch('platform.machine', return_value='x86_64'), \
                    mock.patch.object(isolation, 'restrict') as landlock, \
                    mock.patch.object(isolation, 'restrict_program_syscalls') as seccomp, \
                    mock.patch.object(isolation.os, 'execvpe') as execute:
                (landlock if failing == 'restrict' else seccomp).side_effect = RuntimeError('isolation-unavailable')
                with self.assertRaisesRegex(RuntimeError, 'isolation-unavailable'):
                    isolation.launch()
                execute.assert_not_called()

    def test_program_runtime_revalidates_external_symlinks_on_subsequent_calls(self):
        """验证每次新建 `program_runtime` 均无状态复核包目录及内部软链接，不受此前通过校验调用的缓存污染。"""
        from importlib.machinery import ModuleSpec
        from ni_rua.runtime.programs.runner import program_runtime

        pkg_root = self.root / 'site-packages' / 'numpy'
        pkg_root.mkdir(parents=True)
        init_py = pkg_root / '__init__.py'
        init_py.write_text('# synthetic clean package\n', encoding='utf-8')
        outside_file = self.root / 'outside-secret.txt'
        outside_file.write_text('outside', encoding='utf-8')
        fake_stdlib = self.root / 'fake-stdlib'
        fake_stdlib.mkdir(parents=True)

        def fake_find_spec(name):
            if name == 'numpy':
                spec = ModuleSpec('numpy', None, origin=str(init_py), is_package=True)
                spec.submodule_search_locations = [str(pkg_root)]
                return spec
            return None

        captured_so: list[set[Path]] = []

        def fake_runtime_library_paths(objects):
            captured_so.append(set(objects))
            return ()

        with (
            mock.patch('ni_rua.runtime.programs.runner.stdlib_runtime_entries', return_value=()),
            mock.patch('ni_rua.runtime.programs.runner.sysconfig.get_path', return_value=str(fake_stdlib)),
            mock.patch('ni_rua.runtime.programs.runner.importlib.util.find_spec', side_effect=fake_find_spec),
            mock.patch('ni_rua.runtime.programs.runner.system_library_path', return_value=init_py),
            mock.patch('ni_rua.runtime.programs.runner.runtime_library_paths', side_effect=fake_runtime_library_paths),
        ):
            program_runtime(self.root / 'rt-warm-1', sandbox_enabled=True)
            real_so = pkg_root / 'core.so'
            real_so.write_bytes(b'\x7fELF')
            internal_so_link = pkg_root / 'alias.so'
            internal_so_link.symlink_to(real_so)
            program_runtime(self.root / 'rt-warm-so', sandbox_enabled=True)
            self.assertIn(real_so.resolve(), captured_so[-1])
            self.assertIn(internal_so_link.resolve(), {p.resolve() for p in captured_so[-1]})

            escape_link = pkg_root / 'escaped_link.py'
            escape_link.symlink_to(outside_file)
            with self.assertRaisesRegex(ValueError, 'program_runtime_external_symlink'):
                program_runtime(self.root / 'rt-warm-2', sandbox_enabled=False)
            escape_link.unlink()

            moved_pkg = self.root / 'moved-numpy'
            pkg_root.rename(moved_pkg)
            pkg_root.symlink_to(moved_pkg, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'program_runtime_external_symlink'):
                program_runtime(self.root / 'rt-warm-3', sandbox_enabled=False)



# --- Folded from tests/test_program_limits_unsandboxed.py ---

class UnsandboxedProgramLimitTests(unittest.TestCase):
    def test_program_launches_and_reports_unenforced_limits_honestly(self):
        temp = self.enterContext(tempfile.TemporaryDirectory())
        store = Store(Path(temp) / "run", dict(run_id="test", episode_id="ep", env_instance_id="env", env_generation=1))
        self.addCleanup(store.close)
        runner = ProgramRunner(replace(program_services(SimpleNamespace(store=store)), sandbox_enabled=False))
        self.addCleanup(runner.close)
        # Where the host can cap the address space the program must run capped; elsewhere (macOS) the cap must be
        # recorded as unenforced.
        capped = expect(address_space_limit, "unsandboxed programs run under a 1 GiB RLIMIT_AS cap")
        source = ("import resource\n"
                  "assert resource.getrlimit(resource.RLIMIT_FSIZE)[0] == 8*1024*1024\n"
                  + ("assert resource.getrlimit(resource.RLIMIT_AS)[0] == 1024**3\n" if capped else "")
                  + "print('limits_ok')\n")
        result = runner.run(dict(source=source, timeout_s=30), dict(thread_id="t", turn_id="u", tool_call_id="x"))
        self.assertEqual(result["status"], "succeeded", result.get("reason"))
        spec = json.loads(next((store.root / "private").glob("isolation-*.json")).read_text())
        if capped:
            self.assertEqual(spec["limits"]["RLIMIT_AS"], 1024 ** 3)
            self.assertNotIn("unenforced_limits", spec)
        else:
            self.assertNotIn("RLIMIT_AS", spec["limits"])
            self.assertIn("RLIMIT_AS", spec["unenforced_limits"])


class _OsWithoutWaitid:
    """``os`` as on macOS, where Python has no ``os.waitid``: forces the ``waitpid(WUNTRACED)`` path on any host."""
    def __getattr__(self, name):
        if name == "waitid":
            raise AttributeError(name)
        return getattr(os, name)


class _OsWaitidAnswersEchild:
    """``os`` whose ``waitid`` answers as Linux does for a child that has exited but is not reaped (ECHILD to a
    WSTOPPED-only wait): the Linux kernel answer on any host, macOS included (no ``os.waitid`` there)."""
    def waitid(self, *args):
        raise ChildProcessError(errno.ECHILD, os.strerror(errno.ECHILD))

    def __getattr__(self, name):
        return getattr(os, name)


def exited_child(code=7):
    """A ``Popen`` child that has exited with ``code`` and is not reaped yet (a zombie)."""
    process = subprocess.Popen([sys.executable, "-c", f"import sys; sys.stdin.read(); raise SystemExit({code})"],
                               stdin=subprocess.PIPE, start_new_session=True)
    watch = _ExitWatch(process.pid)   # macOS: registered while the child still runs
    process.stdin.close()             # the child exits now
    if not watch.wait(10):
        raise AssertionError("child never exited")
    return process


class ReviewPauseConfirmationTests(unittest.TestCase):
    """``ProgramRunner._process_stopped`` confirms a program's review pause (SIGSTOP) with real child processes.

    Regression: the pause used ``os.waitid`` unconditionally, so on macOS (no ``os.waitid``) every reviewed
    ``robot_action``/``robot_wait`` call of a program failed with ``module 'os' has no attribute 'waitid'``.
    """
    def implementations(self):
        yield "host os", mock.patch.object(runner_module, "os", os)
        yield "without os.waitid", mock.patch.object(runner_module, "os", _OsWithoutWaitid())

    def test_stop_is_confirmed_with_and_without_waitid(self):
        for name, patch in self.implementations():
            with self.subTest(implementation=name), patch:
                process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                           start_new_session=True)
                try:
                    os.killpg(process.pid, signal.SIGSTOP)
                    deadline = time.monotonic() + 10
                    while not ProgramRunner._process_stopped(process):
                        self.assertLess(time.monotonic(), deadline, "stop never confirmed")
                        time.sleep(.005)
                    self.assertIsNone(process.poll())  # stopped, not exited
                finally:
                    # SIGKILL ends a stopped process too. No SIGCONT afterwards: on macOS signalling a group whose
                    # only member is an unreaped zombie fails with EPERM.
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(10)

    def test_exit_status_is_never_lost_while_waiting_for_a_stop(self):
        for name, patch in self.implementations():
            with self.subTest(implementation=name), patch:
                process = subprocess.Popen([sys.executable, "-c", "raise SystemExit(7)"], start_new_session=True)
                deadline = time.monotonic() + 10
                while process.poll() is None:
                    self.assertFalse(ProgramRunner._process_stopped(process))
                    self.assertLess(time.monotonic(), deadline, "child never exited")
                    time.sleep(.005)
                self.assertEqual(process.returncode, 7)
                self.assertEqual(process.wait(1), 7)

    def test_a_child_that_has_already_exited_is_not_stopped_and_keeps_its_exit_status(self):
        # The program exits between the pause's SIGSTOP (or the loop's poll above) and the stop check. Linux answers
        # os.waitid(WSTOPPED) on such an unreaped child with ECHILD, which escaped as ChildProcessError instead of
        # "not stopped" (seen under load in the loop above). The third implementation gives that answer on any host.
        implementations = [*self.implementations(),
                           ("waitid answering ECHILD", mock.patch.object(runner_module, "os", _OsWaitidAnswersEchild()))]
        for name, patch in implementations:
            with self.subTest(implementation=name):
                process = exited_child(7)
                with patch:
                    self.assertFalse(ProgramRunner._process_stopped(process))
                self.assertEqual(process.wait(1), 7)

    def test_a_program_that_exited_before_its_review_pause_is_reported_so(self):
        # SIGSTOP to a program that has just exited: the kernel answers (Linux: delivered to the zombie; macOS: EPERM
        # while it is unreaped, raised as "[Errno 1] Operation not permitted" before) or EPERM as XNU does, on any host.
        def xnu_killpg(pgid, signum):
            raise PermissionError(errno.EPERM, os.strerror(errno.EPERM))
        runner = ProgramRunner.__new__(ProgramRunner)   # _pause_process uses only the static helpers
        for name, killpg in (("host kernel", os.killpg), ("EPERM as on macOS", xnu_killpg)):
            with self.subTest(answer=name):
                process = exited_child(7)
                entry = dict(process=process, cancel=threading.Event())
                with mock.patch.object(runner_module, "os", _OsWithKillpg(killpg)), \
                        self.assertRaisesRegex(ValueError, "^program_exited_before_review$"):
                    runner._pause_process(entry)
                self.assertFalse(entry.get("process_paused"))
                self.assertEqual(process.wait(1), 7)

    def test_without_waitid_or_popen_wait_lock_the_pause_fails_closed_with_a_named_reason(self):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
        try:
            # A Popen stand-in without CPython's private _waitpid_lock: no waitpid that could take the exit status.
            bare = SimpleNamespace(pid=process.pid, returncode=None)
            with mock.patch.object(runner_module, "os", _OsWithoutWaitid()), \
                    self.assertRaisesRegex(ValueError, "^program_review_pause_unsupported$"):
                ProgramRunner._process_stopped(bare)
            self.assertIsNone(process.poll())
        finally:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(10)


class _OsWithKillpg:
    """``os`` whose ``killpg(pgid, signum)`` is ``killpg``; every other name is the host ``os``."""
    def __init__(self, killpg):
        self.killpg = killpg

    def __getattr__(self, name):
        return getattr(os, name)


class _ExitWatch:
    """Reports that a child process has exited while leaving it unreaped (a zombie), so ``Popen`` still collects
    its exit status. Linux: ``os.waitid(WEXITED | WNOWAIT)``. macOS (no ``os.waitid``): a kqueue ``NOTE_EXIT``
    registered at construction, while the child still runs.
    """
    def __init__(self, pid):
        self.pid = pid
        self.queue = None
        if not hasattr(os, "waitid"):
            self.queue = select.kqueue()
            self.queue.control([select.kevent(pid, select.KQ_FILTER_PROC,
                                              select.KQ_EV_ADD | select.KQ_EV_ONESHOT, select.KQ_NOTE_EXIT)], 0, 0)

    def wait(self, timeout):
        """True once the child has exited (still unreaped); False after ``timeout`` seconds without an exit."""
        if self.queue is not None:
            try:
                return bool(self.queue.control(None, 1, timeout))
            finally:
                self.queue.close()
        deadline = time.monotonic() + timeout
        while os.waitid(os.P_PID, self.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:
            if time.monotonic() >= deadline:
                return False
            time.sleep(.005)
        return True


class _RunnerCase(unittest.TestCase):
    """An unsandboxed ``ProgramRunner`` over a fresh Store."""
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.temp = Path(temp.name)
        self.store = Store(self.temp / "run", dict(run_id="test", episode_id="ep", env_instance_id="env", env_generation=1))
        self.addCleanup(self.store.close)
        self.runner = ProgramRunner(replace(program_services(SimpleNamespace(store=self.store)), sandbox_enabled=False))
        self.addCleanup(self.runner.close)

    def run_program(self, source, killpg, call_id="run"):
        """Run ``source`` to its end with ``os.killpg`` of the runner replaced by ``killpg``; return the job result."""
        with mock.patch.object(runner_module, "os", _OsWithKillpg(killpg)):
            return self.runner.run(dict(source=source, timeout_s=60),
                                   dict(thread_id="t", turn_id="u", tool_call_id=call_id))


class CancelStopSignalRaceTests(_RunnerCase):
    """An explicit cancel races a program that ends by itself (for example, right after its cancelled tool reply).

    Regression: the supervisor saw the program alive, then sent SIGTERM to its process group after the program
    had exited. macOS answers EPERM, not ESRCH, for a group whose members have all exited but are not yet reaped;
    the PermissionError skipped the ``process.wait`` that confirms the exit, so a confirmed cancel was published as
    ``failed`` ("[Errno 1] Operation not permitted"), or as ``unknown`` with ``stop_confirmed=False`` when the
    exit was not yet reapable (flaky test_operation_verification::test_explicit_program_cancel_still_interrupts_waiting_sol).
    """
    def cancel_running_program(self, source, killpg, call_id="cancel"):
        """Start ``source``, cancel it explicitly once its process runs, and return (job result, Popen)."""
        results = []
        with mock.patch.object(runner_module, "os", _OsWithKillpg(killpg)):
            worker = threading.Thread(target=lambda: results.append(self.runner.run(
                dict(source=source, timeout_s=60), dict(thread_id="t", turn_id="u", tool_call_id=call_id))))
            worker.start()
            deadline = time.monotonic() + 10
            while not self.runner.entries:
                self.assertLess(time.monotonic(), deadline, "program never started")
                time.sleep(.005)
            entry = next(iter(self.runner.entries.values()))
            self.assertTrue(entry["process_ready"].wait(10), "program process never launched")
            self.runner.cancel(entry["job_id"])
            worker.join(30)
        self.assertFalse(worker.is_alive(), "program_run never returned after cancel")
        return results[0], entry["process"]

    def test_cancel_is_confirmed_when_the_program_ends_by_itself_as_the_stop_signal_is_sent(self):
        # "exited": the whole group has exited, unreaped, when SIGTERM is sent (old result: failed, [Errno 1]).
        # "exiting": the kernel refuses SIGTERM while the program is still exiting (old result: unknown,
        # stop_confirmed=False, as in the flaky test).
        for variant in ("exited", "exiting"):
            with self.subTest(variant=variant):
                release = self.temp / ("release-" + variant)
                source = ("import os, time\n"
                          "while not os.path.exists(" + repr(str(release)) + "):\n"
                          "    time.sleep(.005)\n")
                sent, exited_first = [], []

                def killpg(pgid, signum, variant=variant, release=release, sent=sent, exited_first=exited_first):
                    sent.append(signum)
                    if signum != signal.SIGTERM or len(sent) > 1:
                        return os.killpg(pgid, signum)
                    # The supervisor has seen the program alive and decided to stop it: the program ends by itself now.
                    if variant == "exiting":
                        release.touch()
                        raise PermissionError(errno.EPERM, os.strerror(errno.EPERM))
                    watch = _ExitWatch(pgid)
                    release.touch()
                    exited_first.append(watch.wait(10))  # hold the signal until the program has exited, unreaped
                    os.killpg(pgid, signum)  # macOS raises EPERM here: every group member has exited, none is reaped.
                    raise PermissionError(errno.EPERM, os.strerror(errno.EPERM))  # Linux signals the zombie; answer as XNU.

                result, process = self.cancel_running_program(source, killpg, call_id=variant)
                if variant == "exited":
                    self.assertEqual(exited_first, [True], "the program did not exit before the stop signal")
                self.assertEqual(sent[0], signal.SIGTERM)
                self.assertEqual(result["status"], "cancelled", result.get("reason"))
                self.assertTrue(result["stop_confirmed"])
                self.assertEqual(result["returncode"], 0)  # the program's own exit status, collected, not lost
                self.assertNotIn("reason", result)
                self.assertIsNotNone(process.poll())

    def test_cancel_stays_unknown_while_a_live_program_refuses_every_stop_signal(self):
        refused = []

        def killpg(pgid, signum):
            refused.append(signum)
            raise PermissionError(errno.EPERM, os.strerror(errno.EPERM))

        result, process = self.cancel_running_program("import time\ntime.sleep(60)\n", killpg)
        self.addCleanup(process.wait, 10)
        self.addCleanup(os.killpg, process.pid, signal.SIGKILL)
        self.assertIsNone(process.poll())  # never stopped, so never reported as stopped
        self.assertEqual(refused[:2], [signal.SIGTERM, signal.SIGKILL])
        self.assertEqual(result["status"], "unknown")
        self.assertFalse(result["stop_confirmed"])
        self.assertIn("Operation not permitted", result["reason"])

    def test_cancel_is_confirmed_when_the_program_exits_as_the_sigkill_escalation_is_sent(self):
        # The program outlives SIGTERM's wait and ends by itself just as SIGKILL is sent: macOS answers EPERM for the
        # exited, unreaped leader, and any POSIX kernel answers ESRCH if another thread (e.g. _pause_process.poll)
        # reaped the exited leader first (old result: failed, "[Errno 1]" / "[Errno 3]").
        for variant, err in (("unreaped", PermissionError(errno.EPERM, os.strerror(errno.EPERM))),
                             ("reaped", ProcessLookupError(errno.ESRCH, os.strerror(errno.ESRCH)))):
            with self.subTest(variant=variant):
                release = self.temp / ("release-escalation-" + variant)
                source = ("import os, time\n"
                          "while not os.path.exists(" + repr(str(release)) + "):\n"
                          "    time.sleep(.005)\n")
                sent = []

                def killpg(pgid, signum, variant=variant, err=err, release=release, sent=sent):
                    sent.append(signum)
                    if signum == signal.SIGTERM and sent.count(signum) == 1:
                        return None               # delivered, ignored: the program keeps running past the wait
                    if signum == signal.SIGKILL and sent.count(signum) == 1:
                        watch = _ExitWatch(pgid)
                        release.touch()
                        self.assertTrue(watch.wait(10), "the program did not exit before SIGKILL")
                        if variant == "reaped":
                            entry = next(e for e in self.runner.entries.values() if e["process"].pid == pgid)
                            deadline = time.monotonic() + 5
                            while entry["process"].poll() is None and time.monotonic() < deadline:
                                time.sleep(.005)
                            self.assertIsNotNone(entry["process"].returncode)
                        raise err
                    return os.killpg(pgid, signum)

                result, process = self.cancel_running_program(source, killpg, call_id="escalation-" + variant)
                self.assertEqual(sent[:2], [signal.SIGTERM, signal.SIGKILL])
                self.assertEqual((result["status"], result["stop_confirmed"], result["returncode"]),
                                 ("cancelled", True, 0))
                self.assertNotIn("reason", result)
                self.assertIsNotNone(process.poll())

    def test_continue_process_tolerates_eperm_when_paused_program_exited_unreaped(self):
        # On macOS, _pause_process cleanup (_continue_process -> killpg(SIGCONT)) can receive EPERM when the program
        # exited right after SIGSTOP; EPERM must not mask program_exited_before_review.
        process = SimpleNamespace(pid=99999, poll=lambda: 0, returncode=0)
        entry = {"process": process, "process_paused": False, "cancel": threading.Event()}

        def killpg(pgid, signum):
            if signum == signal.SIGSTOP:
                return None  # sets process_paused = True before the loop notices poll() == 0
            if signum == signal.SIGCONT:
                raise PermissionError(errno.EPERM, os.strerror(errno.EPERM))
            raise AssertionError(signum)

        with (mock.patch.object(runner_module, "os", _OsWithKillpg(killpg)),
              mock.patch.object(ProgramRunner, "_process_stopped", return_value=False)):
            with self.assertRaisesRegex(ValueError, "program_exited_before_review"):
                self.runner._pause_process(entry)
        self.assertFalse(entry["process_paused"])


class DescendantStopTests(_RunnerCase):
    """After the program process has exited, the rest of its process group is killed (no detached actor jobs).

    Regression: macOS answers EPERM, not ESRCH, while a descendant has exited but launchd has not reaped it yet; the
    PermissionError published a program that exited 0 as ``failed`` ("[Errno 1] Operation not permitted", 5 of 400
    runs). The answer is now no verdict: the group's disappearance confirms the stop, and a group that keeps answering
    (a live descendant refusing the signal) makes the job ``unknown``.
    """
    def descendant_source(self, child):
        """A program that forks ``child`` (Python statements, run once the program itself has exited) and exits 0."""
        return ("import os, time\n"
                "leader = os.getpid()\n"
                "if os.fork() == 0:\n"
                "    while os.getppid() == leader:\n"
                "        time.sleep(.005)\n"
                + "".join("    " + line + "\n" for line in child.splitlines()))

    def test_a_descendant_that_exits_unreaped_does_not_fail_a_succeeded_program(self):
        marker, staged = self.temp / "descendant-exiting", self.temp / "descendant-exiting.tmp"
        source = self.descendant_source(f"open({str(staged)!r}, 'w').write(str(os.getpid()))\n"
                                        f"os.replace({str(staged)!r}, {str(marker)!r})\nos._exit(0)")
        sent = []

        def killpg(pgid, signum):
            sent.append(signum)
            if signum == signal.SIGKILL and sent.count(signum) == 1:
                deadline = time.monotonic() + 10
                while not marker.exists():        # the orphan is exiting: hold the signal until then
                    self.assertLess(time.monotonic(), deadline, "the descendant never exited")
                    time.sleep(.005)
                raise PermissionError(errno.EPERM, os.strerror(errno.EPERM))   # XNU's answer until it is reaped
            if signum == 0:
                # The orphan's adopter reaps it: launchd or init, or this process when scripts/run_tests.py made it a
                # Linux child subreaper (the exited orphan would otherwise stay a zombie answering the group probe).
                with contextlib.suppress(ChildProcessError):
                    os.waitpid(int(marker.read_text()), os.WNOHANG)
            return os.killpg(pgid, signum)

        result = self.run_program(source, killpg)
        self.assertEqual(sent[0], signal.SIGKILL)
        self.assertIn(0, sent)                    # the group's disappearance was checked, not assumed
        self.assertEqual((result["status"], result["returncode"], result["stop_confirmed"]), ("succeeded", 0, True))
        self.assertNotIn("reason", result)

    def test_a_live_descendant_that_refuses_the_signal_leaves_the_job_unknown(self):
        source = self.descendant_source("time.sleep(60)")
        groups = []

        def killpg(pgid, signum):
            groups.append(pgid)
            if signum in (signal.SIGKILL, 0):
                raise PermissionError(errno.EPERM, os.strerror(errno.EPERM))
            return os.killpg(pgid, signum)

        try:
            result = self.run_program(source, killpg)
        finally:
            for pgid in set(groups):
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self.assertEqual((result["status"], result["stop_confirmed"], result["returncode"]), ("unknown", False, 0))
        self.assertEqual(result["reason"], "program_descendant_stop_unconfirmed")

    def test_the_exit_status_is_kept_when_a_step_after_the_descendant_stop_fails(self):
        # returncode is recorded before the descendant stop, so a failure of a later step (here the action-ownership
        # check) still reports the program's own exit status instead of losing it.
        def action_active(identifier):
            raise RuntimeError("action ownership unavailable")
        runner = ProgramRunner(replace(self.runner.services, action_active=action_active))
        self.addCleanup(runner.close)
        result = runner.run(dict(source="raise SystemExit(3)\n", timeout_s=60),
                            dict(thread_id="t", turn_id="u", tool_call_id="late-failure"))
        self.assertEqual((result["status"], result.get("returncode")), ("failed", 3))
        self.assertIn("action ownership unavailable", result["reason"])



if __name__ == "__main__":
    unittest.main()
