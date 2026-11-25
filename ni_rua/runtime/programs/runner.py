"""Invocation-owned Python jobs with resumable main-actor action review."""
import base64
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import socket
import stat
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
from typing import Callable

from ...core.contracts import canonical, digest, utc_now
from ...core.observation_policy import LEGACY_ACTOR_PROFILE, reject_privileged, allows_depth
from ...core.storage.store import Store
from ...cortex.session.system_runtime import runtime_library_paths, stdlib_runtime_entries, system_library_path

TERMINAL = {'succeeded','failed','cancelled','unknown'}
ACTION_RESERVE_S = 5.0
_UNSET = object()
# Seconds a program stop is confirmed for: the leader's exit after SIGTERM (then SIGKILL), and after an EPERM answer the
# leader's exit or the process group's disappearance.
STOP_CONFIRM_S = 2.0


def _scan_directory_tree(source: Path, runtime_objects: set):
    """单趟无状态校验目录内软链接边界并收集 `.so` 动态库，确保每次新建运行时均实时复核文件系统状态。"""
    stack = [str(source)]
    while stack:
        current = stack.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                if entry.is_symlink():
                    target = Path(entry.path).resolve()
                    if not target.is_relative_to(source):
                        raise ValueError('program_runtime_external_symlink')
                    if entry.name.endswith('.so') and target.is_file():
                        runtime_objects.add(Path(entry.path))
                elif entry.is_dir(follow_symlinks=False):
                    stack.append(entry.path)
                elif entry.name.endswith('.so') and entry.is_file(follow_symlinks=False):
                    runtime_objects.add(Path(entry.path))


def program_runtime(root, *, sandbox_enabled=True):
    """Expose Python's runtime and selected computation packages, never its prefix.

    The private directory supplies import directory listings. Its links target
    individually allowed runtime entries; no host site-packages or .pth is read.
    Callers serialize creation and keep this directory outside actor write roots.
    """
    root = Path(root)
    library = root / 'lib' / ('python' + sysconfig.get_python_version())
    packages = library / 'site-packages'
    packages.mkdir(parents=True, exist_ok=True)
    stdlib = Path(sysconfig.get_path('stdlib')).resolve()
    executable = Path(sys.executable).resolve()
    read_paths = [str(root), str(executable)]
    runtime_objects = {executable}
    if sandbox_enabled:
        seccomp = system_library_path('libseccomp.so.2')
        runtime_objects.add(seccomp)
        read_paths.append(str(seccomp))

    def expose(source, destination, boundary):
        if source.is_symlink():
            raise ValueError('program_runtime_external_symlink')
        source = source.resolve()
        if not source.is_relative_to(boundary):
            raise ValueError('program_runtime_external_symlink')
        is_dir = source.is_dir()
        if is_dir:
            _scan_directory_tree(source, runtime_objects)
        elif source.is_file():
            runtime_objects.add(source)
        if destination.is_symlink():
            if destination.resolve() != source:
                raise ValueError('program_runtime_link_conflict')
        elif destination.exists():
            raise ValueError('program_runtime_link_conflict')
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(source, target_is_directory=is_dir)
        read_paths.append(str(source))

    module_names = set(sys.stdlib_module_names) | {sysconfig._get_sysconfigdata_name()}
    for item in stdlib_runtime_entries(stdlib, module_names):
        expose(item, library / item.relative_to(stdlib), stdlib)
    for name, libraries in (('numpy', ('numpy.libs',)), ('scipy', ('scipy.libs',)),
                            ('PIL', ('pillow.libs',)),
                            ('cv2', ('opencv_python_headless.libs', 'opencv_python.libs')),
                            ('typing_extensions', ()), ('eigenpy', ()), ('coal', ()),
                            ('pinocchio', ()), ('qpsolvers', ()), ('pink', ()),
                            ('daqp', ()), ('ruckig', ())):
        # A venv may inherit its base interpreter's site-packages. sysconfig's
        # purelib describes the venv destination, not where imports are found.
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.origin:
            continue
        origin = Path(spec.origin)
        if spec.submodule_search_locations:
            source = origin.parent
            boundary = source.parent.resolve()
            expose(source, packages / name, boundary)
            for library in libraries:
                library_source = source.parent / library
                if library_source.exists():
                    expose(library_source, packages / library, boundary)
        elif origin.name.endswith((*importlib.machinery.EXTENSION_SUFFIXES, *importlib.machinery.SOURCE_SUFFIXES)):
            expose(origin, packages / origin.name, origin.parent.resolve())
        else:
            raise ValueError('program_runtime_requires_regular_package')
    if sandbox_enabled:
        read_paths.extend(str(path) for path in runtime_library_paths(runtime_objects))
    return root, packages, sorted(set(read_paths))


@dataclass(frozen=True)
class ProgramServices:
    """Server-bound broker operations; no robot, registry, or runtime state owner."""
    store: Store
    validate_tool: Callable[[str], None]
    validate_arguments: Callable[[str, dict], None]
    invoke: Callable[[str, dict, dict], dict]
    effect: Callable[[str, dict], str]
    observation: Callable[[], dict]
    action_active: Callable[[str], bool]
    cancel_action: Callable[[str], bool]
    sandbox_enabled: bool = True
    seccomp_enabled: bool = False
    programs_dir: str | Path | None = None
    workspace_modules: tuple = ()
    # The run's EmbodimentProfile (Harness.embodiment; None = the default robot ARX-X5): request deadlines that omitted
    # timeouts are capped from (default_action_timeout_s / default_wait_timeout_s, s) and the wait-review wording.
    embodiment: object = None
    current_identity: Callable[[], dict | None] | None = None
    action_admitted: Callable[[str, str | None], bool] | None = None


class _BlindContinuationBlocked(Exception):
    def __init__(self, payload: dict):
        super().__init__(payload.get("message") or payload.get("code") or "program_blind_continuation_blocked")
        self.payload = payload


def program_remaining(entry):
    """Remaining work time, excluding this job's own model/review waits."""
    lock = entry.setdefault('budget_lock', threading.RLock())
    with lock:
        started = entry.get('model_wait_started_at')
        if started is None:
            started = entry.get('review_wait_started_at')
        return entry['deadline'] - (time.monotonic() if started is None else started)


@contextmanager
def model_wait(entry):
    """Pause only this program's work clock around its counted Sol request.

    This does not change CPU/resource limits or action timeouts. Explicit
    cancellation remains active, and unrelated programs/actor calls earn no
    additional work time.
    """
    lock = entry.setdefault('budget_lock', threading.RLock())
    with lock:
        started = time.monotonic()
        if entry['cancel'].is_set() or started >= entry['deadline']:
            raise ValueError('program_channel_expired')
        if entry.get('model_wait_started_at') is not None or entry.get('review_wait_started_at') is not None:
            raise ValueError('program_model_wait_already_active')
        entry['model_wait_started_at'] = started
    try:
        yield
    finally:
        with lock:
            elapsed = max(0.0, time.monotonic() - started)
            entry['deadline'] += elapsed
            entry['model_wait_s'] = entry.get('model_wait_s', 0.0) + elapsed
            entry['model_wait_started_at'] = None


@contextmanager
def review_wait(entry):
    """Freeze only this job's work budget while its main actor reviews a plan."""
    lock = entry.setdefault('budget_lock', threading.RLock())
    with lock:
        started = time.monotonic()
        if entry['cancel'].is_set() or started >= entry['deadline']:
            raise ValueError('program_channel_expired')
        if entry.get('review_wait_started_at') is not None or entry.get('model_wait_started_at') is not None:
            raise ValueError('program_wait_already_active')
        entry['review_wait_started_at'] = started
    try:
        yield
    finally:
        with lock:
            elapsed = max(0.0, time.monotonic() - started)
            entry['deadline'] += elapsed
            entry['review_wait_s'] = entry.get('review_wait_s', 0.0) + elapsed
            entry['review_wait_started_at'] = None

class ProgramRunner:
    def __init__(self, services: ProgramServices):
        if not isinstance(services, ProgramServices):
            raise TypeError('ProgramRunner requires ProgramServices')
        self.services, self.store = services, services.store
        private = self.store.root / 'private'
        if private.is_symlink():
            raise ValueError('program_private_directory_must_be_real')
        private.mkdir(exist_ok=True, mode=0o700)
        self.root = private / 'program-workspaces'
        self.root.mkdir(parents=True, exist_ok=True)
        self.logs = private / 'program-logs'
        self.logs.mkdir(parents=True, exist_ok=True)
        for path in (self.root.parent, self.root, self.logs):
            if path.is_symlink() or not path.is_dir():
                raise ValueError('program_private_directory_must_be_real')
            path.chmod(0o700)
        self._isolated_ws_root = None
        self._owns_isolated_ws_root = False
        ext_prog_dir = self.services.programs_dir or os.environ.get('NI_RUA_PROGRAMS_DIR')
        if ext_prog_dir:
            ext_root = Path(ext_prog_dir)
            ext_root.mkdir(parents=True, exist_ok=True, mode=0o700)
            if ext_root.is_symlink() or not ext_root.is_dir():
                raise ValueError('program_private_directory_must_be_real')
            self._isolated_ws_root = ext_root.resolve()
        elif self.services.seccomp_enabled and not self.services.sandbox_enabled:
            self._isolated_ws_root = Path(tempfile.mkdtemp(prefix='ni-rua-prog-ws-')).resolve()
            self._isolated_ws_root.chmod(0o700)
            self._owns_isolated_ws_root = True
        self.entries = {}
        self._unresolved_handles = {}
        self.last_activity_at = time.monotonic()
        self.lock = threading.RLock()
        self.runtime_lock = threading.Lock()
        self.runtime = None
        for job in self.store.states('program'):
            if job['status'] not in TERMINAL:
                job.update(status='unknown', stop_confirmed=False,
                           reason='gateway_restarted_without_program_acknowledgement')
                self.store.set_state('program',job['job_id'],job)

    def _current_identity(self):
        services = getattr(self, 'services', None)
        fn = getattr(services, 'current_identity', None) if services is not None else None
        ident = fn() if callable(fn) else None
        if not isinstance(ident, dict):
            store = getattr(self, 'store', None)
            ident = getattr(store, 'identity', None) if store is not None else None
        return dict(ident) if isinstance(ident, dict) else None

    def snapshot(self):
        """Expose lifecycle facts without lending out mutable entries or locks."""
        with self.lock:
            entries = list(self.entries.values())
        active, pending = [], []
        for entry in entries:
            with entry['condition']:
                active.append(entry['job_id'])
                job = entry['job']
                review = job.get('pending_review') or {}
                process = entry.get('process')
                if (job.get('status') == 'awaiting_review' and entry.get('process_paused')
                        and not entry['cancel'].is_set() and process is not None and process.poll() is None
                        and review.get('decision') == 'pending' and review.get('review_id')):
                    pending.append(dict(job_id=entry['job_id'], review_id=review['review_id'],
                        kind=review.get('kind'), **({'plan_id': review['plan_id']} if review.get('plan_id') else {})))
        # Unknown ACKs outlive their supervisor threads. Never infer a confirmed
        # stop from an empty entries map or erase old unresolved process ownership.
        # A job is published before its thread/entry starts. That startup window
        # is active ownership too, never permission for observer evaluation.
        active.extend(job['job_id'] for job in self.store.states('program')
                      if job['status'] not in TERMINAL and job['job_id'] not in active)
        return dict(
            active_jobs=active,
            pending_reviews=pending,
            unresolved_jobs=self.unresolved_jobs(identity=self._current_identity()),
        )

    def unresolved_jobs(self, *, identity=None, acknowledged=()):
        ack_set = set(acknowledged or ())
        out = []
        for job in self.store.states('program'):
            jid = job.get('job_id')
            if not jid:
                continue
            if not (job.get('status') == 'unknown' or job.get('stop_confirmed') is False):
                continue
            if job.get('late_stop_confirmed'):
                continue
            if jid in ack_set or f'program:{jid}' in ack_set:
                continue
            job_ident = job.get('identity')
            if isinstance(identity, dict) and isinstance(job_ident, dict):
                if (identity.get('run_id') and job_ident.get('run_id')
                        and identity['run_id'] != job_ident['run_id']):
                    continue
                if (identity.get('episode_id') and job_ident.get('episode_id')
                        and identity['episode_id'] != job_ident['episode_id']):
                    continue
            out.append(jid)
        return out

    def reconcile(self, job_id: str | None = None) -> dict:
        if job_id is not None:
            saved = self.store.read_state('program', job_id)
            if not saved:
                raise ValueError('unknown_program_job')
            targets = [saved]
        else:
            targets = list(self.store.states('program'))
        reconciled, still_unresolved = [], []
        services = getattr(self, 'services', None)
        action_active_fn = getattr(services, 'action_active', None) if services is not None else None
        for job in targets:
            jid = job.get('job_id')
            if not jid:
                continue
            if not (job.get('status') == 'unknown' or job.get('stop_confirmed') is False):
                continue
            if job.get('late_stop_confirmed'):
                continue
            with self.lock:
                live_entry = self.entries.get(jid)
                handles = getattr(self, '_unresolved_handles', None) or {}
                handle = dict(handles.get(jid) or {})
            if live_entry is not None:
                still_unresolved.append(jid)
                continue
            proc = handle.get('process')
            pgid = handle.get('pgid') if handle.get('pgid') is not None else job.get('pgid')
            broker = handle.get('broker')
            if pgid is not None:
                proc_gone = self._group_gone(pgid, 0.05, proc=proc)
            elif proc is not None:
                proc_gone = proc.poll() is not None
            elif handle.get('no_process_spawned'):
                proc_gone = True
            else:
                proc_gone = False
            broker_gone = (broker is None) or (not broker.is_alive())
            action_gone = (not action_active_fn(jid)) if callable(action_active_fn) else True
            if proc_gone and broker_gone and action_gone:
                confirmed_at_ns = time.time_ns()
                updated = deepcopy(job)
                updated['late_stop_confirmed'] = True
                updated['late_stop_confirmed_at_ns'] = confirmed_at_ns
                self.store.set_state('program', jid, deepcopy(updated))
                with self.lock:
                    self._unresolved_handles.pop(jid, None)
                self.store.event(
                    'program.stop_confirmed_late',
                    {'job_id': jid, 'late_stop_confirmed': True, 'late_stop_confirmed_at_ns': confirmed_at_ns},
                    updated.get('correlation') or {'program_id': jid},
                )
                reconciled.append(jid)
            else:
                still_unresolved.append(jid)
        return {'reconciled': reconciled, 'still_unresolved': still_unresolved}

    def continuation_state(self):
        state = self.snapshot()
        blocked = bool(state['unresolved_jobs'] or len(state['active_jobs']) != len(state['pending_reviews']))
        return dict(blocked=blocked, pending_reviews=[] if blocked else state['pending_reviews'])

    def _active_entry(self, identifier):
        with self.lock:
            entry = self.entries.get(identifier)
        if entry is None or entry['cancel'].is_set() or program_remaining(entry) <= 0:
            raise ValueError('program_channel_expired')
        return entry

    def admit_action(self, identifier, timeout_s):
        if program_remaining(self._active_entry(identifier)) < timeout_s + ACTION_RESERVE_S:
            raise ValueError('program_time_insufficient_for_action')

    def _embodiment(self):
        from ...core.action_validation import embodiment_or_default
        return embodiment_or_default(self.services.embodiment)

    def _action_timeout(self, entry, tool, arguments):
        """``arguments`` with an omitted robot_action preview/preflight/direct/execute_if_clean or robot_wait deadline
        capped to the remaining work time: min(the embodiment's default deadline in s, remaining - action reserve - dispatch margin)."""
        if 'timeout_s' in arguments:
            entry['_explicit_timeout'] = arguments['timeout_s']
            return arguments
        entry.pop('_explicit_timeout', None)
        profile = self._embodiment()
        if tool == 'robot_action' and arguments.get('mode', 'preview') in {
            'preview', 'preflight', 'direct', 'execute_if_clean',
        }:
            default = profile.default_action_timeout_s
        elif tool == 'robot_wait':
            default = profile.default_wait_timeout_s
        else:
            return arguments
        # Leave dispatch headroom; expensive planning can still fail final admission.
        margin = 6.0 if program_remaining(entry) >= 20.0 else 1.0
        available = program_remaining(entry) - ACTION_RESERVE_S - margin
        if available < .1:
            if (
                tool == 'robot_action'
                and arguments.get('mode', 'preview') in {'preview', 'preflight'}
                and not arguments.get('auto_execute_if_clean')
            ):
                return arguments  # Read-only planning need not reserve execution time.
            raise ValueError('program_time_insufficient_for_action')
        return {**arguments, 'timeout_s': min(default, available)}

    def review_controls(self, identifier):
        entry = self._active_entry(identifier)
        return dict(cancel_event=entry['cancel'], wait_scope=lambda: model_wait(entry))

    def cancel_all(self):
        with self.lock:
            identifiers = list(self.entries)
        for identifier in identifiers:
            self.cancel(identifier)

    def status(self, identifier):
        value = self.store.read_state('program',identifier)
        if not value: raise ValueError('unknown_program_job')
        result=dict(value)
        for name in ("stdout","stderr"):
            if value.get(name+"_ref"):
                info,content=self.store.resolve_artifact(value[name+"_ref"],allowed_scopes=("actor",))
                result[name+"_text"]=content[:8192].decode(errors="replace")
                result[name+"_truncated"]=len(content)>8192
        return result

    def review_content(self, identifier):
        """Original preview content for the main actor; image blocks stay images."""
        job = self.store.read_state('program', identifier)
        reference = (job or {}).get('pending_review', {}).get('preview_ref')
        if not reference:
            return []
        info, raw = self.store.resolve_artifact(reference, allowed_scopes=('actor',))
        if info.get('kind') != 'program_review_preview':
            raise ValueError('program_review_artifact_kind_mismatch')
        return json.loads(raw).get('contentItems', [])

    def _publish(self, entry, **changes):
        """Persist a coherent boundary snapshot and wake its main-actor caller."""
        from .program_review import trusted_summary
        with entry['condition']:
            job = entry['job']
            job.update(changes)
            job.update(calls=entry['calls'], model_wait_s=entry['model_wait_s'],
                       review_wait_s=entry['review_wait_s'],
                       action_summary=trusted_summary(self.store, entry['action_evidence']),
                       visual_feedback=deepcopy(entry.get('visual_feedback', [])),
                       final_observation_id=entry.get('final_observation_id'))
            job['action_summary']['review_rejections'] = sum(
                value == 'reject' for value in job.get('review_decisions', {}).values())
            if (job['action_summary'].get('status_counts') or {}).get('failed', 0) > 0:
                job['physical_action_failures_detected'] = True
            failures = entry.get('call_errors', [])
            job['call_summary'] = dict(source_kind='program_broker_tool_calls', calls=entry['calls'],
                failed_calls=len(failures), recent_errors=deepcopy(failures[-8:]))
            job['final_observation'] = (dict(observation_id=entry['final_observation_id'],
                source_kind='program_broker_tool_reply',
                delivery_semantics='Program access is not main-actor image delivery; the harness must forward this observation.')
                if entry.get('final_observation_id') else None)
            for name in ('stdout', 'stderr'):
                # Logs are host-owned, outside every model write root. Never
                # reopen a file in the program's mutable workspace as evidence.
                path = self.logs / (entry['job_id'] + '.' + name)
                try:
                    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
                except FileNotFoundError:
                    data = b''
                else:
                    with os.fdopen(descriptor, 'rb') as stream:
                        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                            raise ValueError('program_log_requires_regular_file')
                        data = stream.read()
                job[name+'_ref'] = self.store.put_artifact(data,
                    'program_'+name, 'text/plain')
            self.store.set_state('program', entry['job_id'], deepcopy(job))
            entry['condition'].notify_all()

    def _await_boundary(self, entry):
        with entry['condition']:
            while entry['job']['status'] not in TERMINAL | {'awaiting_review'}:
                entry['condition'].wait(.1)
        return self.status(entry['job_id'])

    @staticmethod
    def _continue_process(entry):
        process = entry.get('process')
        if process is not None and entry.get('process_paused'):
            try:
                os.killpg(process.pid, signal.SIGCONT)
            except (ProcessLookupError, PermissionError):
                pass
            entry['process_paused'] = False

    @staticmethod
    def _process_stopped(process):
        """True once the program process reports a stop (SIGSTOP). Never takes an exit status from ``Popen``.

        Raises ``ValueError('program_review_pause_unsupported')`` on a host without ``os.waitid`` whose ``Popen``
        has no ``_waitpid_lock`` (CPython has it on POSIX since 3.4): the stop could then not be confirmed without
        risking the exit status, so the reviewed call fails closed with that reason.
        """
        if hasattr(os, 'waitid'):
            # WSTOPPED without WEXITED reports stops only; an exit stays for Popen to collect. Linux answers ECHILD
            # for a child that has already exited (a zombie is not waitable for stops) or was reaped: not stopped.
            try:
                stopped = os.waitid(os.P_PID, process.pid, os.WSTOPPED | os.WNOHANG)
            except ChildProcessError:
                return False
            return stopped is not None and stopped.si_code == os.CLD_STOPPED
        # No os.waitid (macOS). waitpid(WUNTRACED) reports the stop but also collects an exit, so it runs under
        # Popen's own wait lock and hands an exit status to Popen instead of losing it.
        lock = getattr(process, '_waitpid_lock', None)
        if lock is None:
            raise ValueError('program_review_pause_unsupported')
        if not lock.acquire(False):
            return False  # Popen is collecting the status right now; the caller checks again.
        try:
            if process.returncode is not None:
                return False
            pid, status = os.waitpid(process.pid, os.WUNTRACED | os.WNOHANG)
            if pid and os.WIFSTOPPED(status):
                return True
            if pid:
                process.returncode = os.waitstatus_to_exitcode(status)
            return False
        finally:
            lock.release()

    @staticmethod
    def _group_gone(pgid, bound_s=0.05, proc=None):
        """True once process group ``pgid`` has no member left (``killpg(pgid, 0)`` answers ESRCH), False when it
        still answers after ``bound_s`` seconds. EPERM is no verdict: macOS answers it both while every remaining
        member has exited but is not yet reaped (an orphan waits for launchd) and for a live member that refuses."""
        if not isinstance(bound_s, (int, float)):
            proc = bound_s
            bound_s = 0.05
        if proc is not None and proc.poll() is None:
            return False
        deadline = time.monotonic() + float(bound_s)
        while True:
            if proc is not None:
                proc.poll()
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                return True
            except PermissionError:
                pass
            if time.monotonic() >= deadline:
                return False
            time.sleep(.005)

    def _pause_process(self, entry):
        process = entry['process']
        try:
            os.killpg(process.pid, signal.SIGSTOP)
            entry['process_paused'] = True
        except (ProcessLookupError, PermissionError):
            # No verdict: macOS answers EPERM for a program that has exited but is not yet reaped, ESRCH follows once
            # it is reaped. The loop decides: program_exited_before_review, else program_review_pause_unconfirmed.
            pass
        deadline = time.monotonic()+1
        try:
            while time.monotonic() < deadline:
                if entry['cancel'].is_set():
                    raise ValueError('program_channel_expired')
                if self._process_stopped(process):
                    return
                if process.poll() is not None:
                    raise ValueError('program_exited_before_review')
                time.sleep(.005)
            raise ValueError('program_review_pause_unconfirmed')
        except Exception:
            self._continue_process(entry)
            raise

    def _validate_raw_review_action(self, arguments):
        """审核投影前对原始 robot_action 请求做模式字段与 6 类预设纯校验，防止缩写请求吞没非法参数。"""
        from ..actions.unified_actions import validate_raw_action_presets
        profile = self._embodiment()
        mode = arguments.get('mode', 'preview')
        is_cached = mode == 'execute' or (
            mode == 'execute_if_clean'
            and 'plan_id' in arguments
            and not any(k in arguments for k in ('stages', 'source_plan_id', 'skip_failed_stage'))
        )
        validate_raw_action_presets(arguments, profile)
        if is_cached:
            if 'plan_id' not in arguments:
                raise ValueError('robot_action_requires_plan_id')
            allowed_cached = {
                'mode', 'plan_id', 'action_id', 'image_delivery',
                'expected_cursor', 'expected_revision',
                'observation_id', 'timeout_s', 'camera_id', 'detail',
            }
            if set(arguments) - allowed_cached:
                raise ValueError('cached_batch_commands_cannot_be_changed')

    def _clean_downgrade_reasons(self, arguments, plan, *, is_cached_plan):
        """对受审 execute_if_clean / auto_execute_if_clean 请求基于展开后 source_args 复核硬阻断原因。"""
        from types import SimpleNamespace
        if plan.get('status') == 'awaiting_observation':
            return []
        if is_cached_plan:
            cursor = plan.get('executed_batch_native_steps', 0)
            revision = plan.get('plan_revision', 0)
            if (
                plan.get('status') != 'planned'
                or type(cursor) is not int or cursor != 0
                or type(revision) is not int or revision != 0
                or bool(plan.get('last_action_id'))
            ):
                return []
        obs_meta = self.services.observation() or {}
        obs_id = obs_meta.get('observation_id')
        latest = None
        if isinstance(obs_id, str) and obs_id:
            try:
                latest = self.store.read('observation', obs_id)
            except KeyError:
                latest = {'observation_id': obs_id}
        ident = self._current_identity() or getattr(self.store, 'identity', None) or {}
        plans_map = {}
        plan_id = plan.get('plan_id')
        for pid in {plan_id, arguments.get('plan_id'), arguments.get('source_plan_id')} - {None}:
            if not isinstance(pid, str) or not pid:
                continue
            try:
                stored_rec = deepcopy(self.store.read('unified_action_plan', pid))
            except KeyError:
                stored_rec = None
            if isinstance(stored_rec, dict):
                if pid == plan_id:
                    stored_rec['summary'] = deepcopy(plan)
                    stored_rec['executable'] = plan.get('status') in ('planned', 'awaiting_observation')
                plans_map[pid] = stored_rec
        if isinstance(plan_id, str) and plan_id and plan_id not in plans_map:
            plans_map[plan_id] = dict(
                identity=deepcopy(ident),
                source_args=deepcopy(arguments) if 'stages' in arguments else {},
                executable=plan.get('status') in ('planned', 'awaiting_observation'),
                summary=deepcopy(plan),
            )
        view = SimpleNamespace(
            store=self.store,
            embodiment=self._embodiment(),
            identity=ident,
            latest=latest,
            unified_action_plans=plans_map,
        )
        from ..actions.unified_actions import preview_downgrade_reasons_for_args
        eval_args = dict(mode='execute_if_clean', plan_id=plan_id) if isinstance(plan_id, str) and plan_id else dict(arguments)
        return list(preview_downgrade_reasons_for_args(view, eval_args, summary=plan))

    @staticmethod
    def _downgraded_clean_plan_response(plan, preview_response, arguments, reasons):
        """构造零步安全降级的计划返回副本；省略 image_delivery 或传 'none' 时保持 0 图。"""
        from ...core.tool_output import ToolOutput, ToolResponse
        summary = deepcopy(plan)
        summary['downgraded_from_execute_if_clean'] = True
        summary['downgrade_reasons'] = list(reasons)
        summary.setdefault('physics_advanced', False)
        summary.setdefault('executed_native_steps', 0)
        keep_images = arguments.get('image_delivery') == 'all'
        images = []
        if keep_images and isinstance(preview_response, ToolResponse):
            images = [
                deepcopy(item) for item in preview_response.content
                if isinstance(item, dict) and item.get('type') == 'inputImage'
            ]
        elif isinstance(summary.get('trajectory_overlay'), dict) and summary['trajectory_overlay'].get('status') == 'available':
            skip_reason = (
                'image_delivery_suppressed'
                if arguments.get('image_delivery') == 'none'
                else 'execute_if_clean_zero_image_mode'
            )
            summary['trajectory_overlay'] = dict(
                status='skipped', reason=skip_reason, candidate_label=summary.get('plan_id'),
            )
        return ToolOutput(summary, extra_content=tuple(images), kind='plan', projection='action_plan').response()

    def _review_action(self, entry, arguments, correlation, identifier, record):
        from .program_review import reviewed_plan, review_execution_state
        from ...core.presentation.actor_presentation import ActorPresentation
        from ...core.tool_output import ToolResponse
        self._validate_raw_review_action(arguments)
        mode = arguments.get('mode', 'preview')
        is_clean_request = mode == 'execute_if_clean' or bool(arguments.get('auto_execute_if_clean'))
        is_cached_plan = mode == 'execute' or (
            mode == 'execute_if_clean'
            and 'plan_id' in arguments
            and not any(k in arguments for k in ('stages', 'source_plan_id', 'skip_failed_stage'))
        )
        if is_cached_plan:
            preview_args = dict(mode='inspect', plan_id=arguments.get('plan_id'))
        else:
            target_mode = 'preflight' if mode == 'preflight' else 'preview'
            preview_args = {
                k: v for k, v in {**arguments, 'mode': target_mode}.items()
                if k != 'auto_execute_if_clean'
            }
        preview_correlation = {**correlation, 'tool_call_id': identifier+'-preview'}
        self.store.event('program.call_started', {'tool': 'robot_action', 'arguments': preview_args},
            preview_correlation)
        preview = self._invoke('robot_action', preview_args, preview_correlation)
        if (
            is_clean_request
            and not is_cached_plan
            and isinstance(preview, ToolResponse)
            and preview.kind == 'plan'
            and isinstance(preview.data, dict)
            and preview.data.get('status') == 'failed'
            and preview.data.get('physics_advanced') is False
            and type(preview.data.get('executed_native_steps')) is int
            and preview.data['executed_native_steps'] == 0
        ):
            record.update(action_dispatch_attempted=True, executed_arguments=dict(arguments))
            self.store.set_state('program_call', identifier, record)
            return preview, dict(arguments)
        plan = reviewed_plan(preview)
        if is_cached_plan:
            if 'expected_cursor' in arguments and arguments['expected_cursor'] != plan.get('executed_batch_native_steps', 0):
                raise ValueError('batch_plan_cursor_changed_since_review')
            if 'expected_revision' in arguments and arguments['expected_revision'] != plan.get('plan_revision', 0):
                raise ValueError('batch_plan_revision_changed_since_review')
        if is_clean_request:
            downgrade_reasons = self._clean_downgrade_reasons(arguments, plan, is_cached_plan=is_cached_plan)
            if downgrade_reasons:
                record.update(action_dispatch_attempted=True, executed_arguments=dict(arguments))
                self.store.set_state('program_call', identifier, record)
                return self._downgraded_clean_plan_response(plan, preview, arguments, downgrade_reasons), dict(arguments)
        review_id = 'review-' + digest([entry['job_id'], identifier, plan['plan_id']])[:24]
        gate = dict(review_id=review_id, plan_id=plan['plan_id'], operation_call_id=identifier,
                    preview=plan, preview_ref=self.store.put_artifact(
                        canonical(ActorPresentation(self.store).present_response(preview)).encode(),
                        'program_review_preview', 'application/json'),
                    kind='robot_action', decision='pending', semantics='Approve through program_run resume; do not execute the plan separately.')
        self._await_review(entry, gate, correlation)
        current_resp = self._invoke('robot_action',
            dict(mode='inspect', plan_id=plan['plan_id']),
            {**correlation, 'tool_call_id': identifier+'-approved-inspect'})
        current = reviewed_plan(current_resp)
        if review_execution_state(current) != review_execution_state(plan):
            raise ValueError('program_action_plan_changed_since_review')
        if is_clean_request:
            post_reasons = self._clean_downgrade_reasons(arguments, current, is_cached_plan=True)
            if post_reasons:
                record.update(action_dispatch_attempted=True, executed_arguments=dict(arguments))
                self.store.set_state('program_call', identifier, record)
                return self._downgraded_clean_plan_response(current, current_resp, arguments, post_reasons), dict(arguments)
        # A second caller may advance even a zero-step checkpoint between inspect
        # and execute. The native cache checks cursor and revision atomically.
        execute_args = dict(
            mode='execute_if_clean' if is_clean_request else 'execute',
            plan_id=plan['plan_id'],
            expected_cursor=arguments.get('expected_cursor', plan.get('executed_batch_native_steps', 0)),
            expected_revision=arguments.get('expected_revision', plan.get('plan_revision', 0)),
        )
        if arguments.get('action_id'):
            execute_args['action_id'] = arguments['action_id']
        if 'image_delivery' in arguments:
            execute_args['image_delivery'] = arguments['image_delivery']
        if entry['cancel'].is_set() or program_remaining(entry) <= 0:
            raise ValueError('program_channel_expired')
        record.update(action_dispatch_attempted=True, executed_arguments=execute_args)
        self.store.set_state('program_call',identifier,record)
        return self._invoke('robot_action', execute_args, correlation), execute_args

    def _review_wait_action(self, entry, arguments, correlation, identifier, record):
        frozen = deepcopy(arguments)
        semantics = ('Hold measured arm positions and gripper goals while advancing the requested native steps; consumes native action budget.'
                     if self._embodiment().clock == 'simulated_steps' else
                     'Hold measured arm positions and gripper goals for the requested real-time control steps; the world keeps moving.')
        preview = dict(kind='bounded_wait', status='review_required', request=frozen,
            observation_id=frozen.get('observation_id'), semantics=semantics)
        response = dict(success=True, contentItems=[dict(type='inputText', text=canonical(preview))])
        review_id = 'review-' + digest([entry['job_id'], identifier, frozen])[:24]
        gate = dict(review_id=review_id, kind='robot_wait', operation_call_id=identifier,
            preview=preview, preview_ref=self.store.put_artifact(canonical(response).encode(),
                'program_review_preview','application/json'), decision='pending',
            semantics='Approve through program_run resume; the exact wait and original observation are retained.')
        self._await_review(entry, gate, correlation)
        if entry['cancel'].is_set() or program_remaining(entry) <= 0:
            raise ValueError('program_channel_expired')
        observed = self.services.observation()['observation_id']
        if observed is not None and observed != frozen.get('observation_id'):
            raise ValueError('stale_observation')
        record.update(action_dispatch_attempted=True, executed_arguments=frozen)
        self.store.set_state('program_call', identifier, record)
        return self._invoke('robot_wait', frozen, correlation), frozen

    def _await_review(self, entry, gate, correlation):
        with review_wait(entry):
            # Blocking the client RPC retains its Python stack; stopping its whole
            # process also prevents another Python thread from running during review.
            entry['process_ready'].wait()
            if entry['cancel'].is_set():
                raise ValueError('program_channel_expired')
            process = entry.get('process')
            if process is None or process.poll() is not None:
                raise ValueError('program_exited_before_review')
            with entry['condition']:
                if entry['cancel'].is_set():
                    raise ValueError('program_channel_expired')
                self._pause_process(entry)
                entry['review_event'].clear()
                entry['review'] = gate
                self._publish(entry, status='awaiting_review', pending_review=deepcopy(gate))
            self.store.event('program.review_requested', gate, correlation)
            entry['review_event'].wait()
            self._continue_process(entry)
            if entry['cancel'].is_set():
                raise ValueError('program_channel_expired')
            if gate['decision'] != 'approve':
                raise ValueError('program_action_review_rejected')

    def resume(self, identifier, review_id, decision, correlation):
        if correlation.get('program_id'):
            raise ValueError('program_tool_not_permitted')
        if decision not in {'approve', 'reject'}:
            raise ValueError('invalid_program_review_decision')
        saved = self.store.read_state('program', identifier)
        if not saved:
            raise ValueError('unknown_program_job')
        if saved.get('correlation', {}).get('thread_id') != correlation.get('thread_id'):
            raise ValueError('program_review_thread_mismatch')
        with self.lock:
            entry = self.entries.get(identifier)
        if entry is None:
            previous = saved.get('review_decisions', {}).get(review_id)
            if previous == decision:
                return self.status(identifier)
            raise ValueError('program_review_not_active')
        with entry['condition']:
            previous = entry['job'].get('review_decisions', {}).get(review_id)
            if previous:
                if previous != decision:
                    raise ValueError('program_review_decision_conflict')
                return self.status(identifier)
            gate = entry.get('review')
            if (entry['cancel'].is_set() or entry['job']['status'] != 'awaiting_review'
                    or not gate or gate['review_id'] != review_id):
                raise ValueError('program_review_not_active')
            gate['decision'] = decision
            entry['job'].setdefault('review_decisions', {})[review_id] = decision
            entry['job'].pop('pending_review', None)
            self._publish(entry, status='running')
            self.store.event('program.review_decided', dict(review_id=review_id, decision=decision,
                kind=gate['kind'], plan_id=gate.get('plan_id')), correlation)
            self._continue_process(entry)
            entry['review_event'].set()
        return self._await_boundary(entry)

    def broker(self, connection, entry):
        # Buffered reads avoid one syscall per byte for multi-megabyte geometry
        # artifacts; buffered writes plus flush deliver complete framed replies.
        entry.setdefault('latched_incomplete', None)
        try:
            with connection, connection.makefile("rwb") as pipe:
                while True:
                    line=pipe.readline(1024*1024+1)
                    if not line: return
                    try:
                        if len(line)>1024*1024: raise ValueError("program_request_too_large")
                        entry["_last_call_physical"] = False
                        dispatched = self.dispatch(entry, json.loads(line))
                        result = {
                            "ok": True,
                            "result": dispatched,
                            "incomplete_action_latched": deepcopy(entry.get("latched_incomplete")),
                            "physical_action": bool(entry.get("_last_call_physical")),
                        }
                    except _BlindContinuationBlocked as blocked:
                        result = {"ok": False, "error": blocked.payload}
                    except Exception as error:
                        result = {"ok": False, "error": str(error)}
                    pipe.write((canonical(result)+"\n").encode())
                    pipe.flush()
        except OSError:
            # Process cancellation may reset reads or break the buffered close
            # flush as well as an explicit write. The supervisor owns its status.
            return

    _serve = broker

    def _is_physical_action_call(self, tool, arguments):
        from .program_review import LEGACY_ACTION_TOOLS
        if tool in LEGACY_ACTION_TOOLS:
            return True
        if tool == 'robot_action':
            if 'mode' in arguments or 'auto_execute_if_clean' in arguments:
                mode = arguments.get('mode', 'preview')
                return mode in ('execute', 'direct', 'execute_if_clean') or bool(arguments.get('auto_execute_if_clean'))
            if 'operation' in arguments:
                return arguments.get('operation') not in ('preview', 'preflight', 'inspect', 'reach')
            try:
                return self._tool_effect(tool, arguments) == 'physical'
            except Exception:
                return False
        if tool in ('robot_wait', 'robot_policy_step'):
            return True
        try:
            return self._tool_effect(tool, arguments) == 'physical'
        except Exception:
            return False

    def _validate_tool(self, name):
        from .program_review import LEGACY_ACTION_TOOLS
        if name in LEGACY_ACTION_TOOLS:
            raise ValueError('program_action_requires_robot_action')
        try:
            self.services.validate_tool(name)
        except ValueError:
            raise ValueError('program_tool_not_permitted') from None

    def _validate_arguments(self, name, arguments):
        self.services.validate_arguments(name, arguments)

    def _invoke(self, name, arguments, correlation):
        return self.services.invoke(name, arguments, correlation)

    def _tool_effect(self, name, arguments):
        return self.services.effect(name, arguments)

    def _was_action_admitted(self, entry, call_id, error):
        admitted_fn = getattr(self.services, 'action_admitted', None)
        if callable(admitted_fn):
            return bool(admitted_fn(call_id, entry.get('job_id')))
        try:
            cursor = 0
            while page := self.store.events_by_type(('action.requested',), after=cursor, limit=500):
                for ev in page:
                    corr = ev.get('correlation') or {}
                    if corr.get('tool_call_id') == call_id and corr.get('program_id') in (None, entry.get('job_id')):
                        return True
                cursor = page[-1]['server_seq']
        except Exception:
            pass
        # Without an admission authority only a rejected input proves the action never ran. Any other error (an
        # internal crash included) may follow execution, so the action stays possibly admitted and is recorded as
        # unknown rather than erased; a host that tracks admission (Harness) answers exactly above.
        from ...core.tool_errors import ToolInputError
        return not isinstance(error, (ToolInputError, ValueError))

    def _requires_review(self, entry, tool, arguments):
        if entry.get('allow_direct_actions', False):
            return False
        if self._tool_effect(tool, arguments) != 'physical':
            return False
        if tool not in {'robot_wait', 'robot_action'}:
            raise ValueError('program_physical_tool_requires_direct_authorization')
        return True

    def dispatch(self, entry, value):
        from ...core.presentation.actor_presentation import ActorPresentation
        from ...core.tool_output import wire_response
        presentation = ActorPresentation(self.store)
        value = presentation.restore(value)
        if not isinstance(value, dict):
            raise ValueError("program_request_must_be_object")
        if entry["cancel"].is_set() or program_remaining(entry) <= 0:
            raise ValueError("program_channel_expired")
        # One operation at a time also makes retry idempotency concrete.
        with entry['call_lock']:
            if entry["cancel"].is_set() or program_remaining(entry) <= 0:
                raise ValueError("program_channel_expired")
            if 'artifact_ref' in value:
                info, content = self.store.resolve_artifact(value['artifact_ref'],allowed_scopes=('actor','internal'))
                if len(content)>16*1024*1024: raise ValueError('program_artifact_too_large')
                profile = self.services.observation()['observation_profile']
                kind = info.get('kind', '')
                if kind.startswith('depth') and not allows_depth(profile):
                    raise ValueError('depth_artifact_not_permitted_by_observation_profile')
                # General scene/geometry artifacts may be actor-derived estimates.
                # Block known exact-calibration products, then inspect every JSON
                # body below; a broad kind prefix would reject legal RGB reasoning.
                if profile == LEGACY_ACTOR_PROFILE and kind in {
                        'calibration', 'camera_calibration', 'camera_pose_sample', 'robot_model',
                        'camera_poses', 'geometry_overlay', 'motion_preview_overlay'}:
                    raise ValueError('calibrated_artifact_not_permitted_by_observation_profile')
                mime = info.get('mime_type', '').split(';', 1)[0].strip().lower()
                if mime == 'application/json' or mime.endswith('+json'):
                    payload = json.loads(content)
                    if kind in {'unified_action_result', 'program_action_evidence'}:
                        from ...core.observation_policy import reject_recorded_action_privileged
                        records = payload if kind == 'program_action_evidence' and isinstance(payload, list) else [payload]
                        for record in records:
                            reject_recorded_action_privileged(record, self.store, profile)
                    else:
                        reject_privileged(payload, profile)
                    # Artifact kind is producer-owned; unknown JSON keeps its business fields.
                    projection = {'unified_action_result': 'action_result',
                        'program_action_evidence': 'program_action_evidence',
                        'manipulation_receipt': 'program_action_evidence',
                        'candidate_trajectory': 'owner_record'}.get(kind)
                    if projection:
                        from ...core.presentation.actor_output import project_metadata
                        content = canonical(project_metadata(payload, projection)).encode('utf-8')
                info, content = presentation.present_artifact(info, content)
                return {'artifact':info,'data_base64':base64.b64encode(content).decode()}
            operation = value.get('operation_id')
            if not isinstance(operation,str) or not operation or len(operation)>80: raise ValueError('invalid_program_operation_id')
            identifier = entry['job_id']+'-'+digest(operation)[:16]
            previous = self.store.read_state('program_call',identifier)
            if previous:
                if previous['input_digest']!=digest(value): raise ValueError('program_operation_content_conflict')
                if previous['status']!='completed': raise ValueError('program_operation_status_unknown_do_not_retry')
                # Older completed wire records may predate the current model
                # presentation policy. Project a copy without changing the
                # archived result or dispatching the operation again.
                return presentation.present(previous['result'])
            from .program_review import action_evidence, observation_grounds_current_world
            self._validate_tool(value.get('tool'))
            if not isinstance(value.get('arguments'),dict): raise ValueError('program_arguments_require_object')
            tool = value['tool']
            arguments = value['arguments']
            is_physical = self._is_physical_action_call(tool, arguments)
            entry['_last_call_physical'] = bool(is_physical)
            allow_after_failure = bool(value.get('allow_after_failure'))
            inc_policy = entry.get('on_incomplete_action') or (entry.get('job') or {}).get('on_incomplete_action', 'block')
            if is_physical:
                if allow_after_failure:
                    entry['latched_incomplete'] = None
                    entry['_latched_incomplete_meta'] = None
                elif inc_policy == 'block' and entry.get('latched_incomplete') is not None:
                    latched = deepcopy(entry['latched_incomplete'])
                    msg = (
                        f"Previous physical action ({latched.get('action_id')}) ended incomplete "
                        f"(status={latched.get('status')}, reason={latched.get('reason')}, "
                        f"failed_stage_id={latched.get('failed_stage_id')}, "
                        f"pending_stage_ids={latched.get('pending_stage_ids')}). "
                        "Call robot_observe to ground on the current world state or pass allow_after_failure=True."
                    )
                    payload = {
                        'code': 'program_blind_continuation_blocked',
                        'message': msg,
                        'incomplete_action': latched,
                    }
                    entry.setdefault('call_errors', []).append(dict(
                        call_id=identifier,
                        tool=tool,
                        reason='program_blind_continuation_blocked',
                        incomplete_action=latched,
                    ))
                    self.store.event(
                        'program.blind_continuation_blocked',
                        payload,
                        {**(entry.get('correlation') or {}), 'program_id': entry.get('job_id'), 'tool_call_id': identifier},
                    )
                    raise _BlindContinuationBlocked(payload)
            entry['calls'] += 1
            record = {'program_id':entry['job_id'],'operation_id':operation,'input_digest':digest(value),'status':'dispatching'}
            self.store.set_state('program_call',identifier,record)
            correlation = {**entry['correlation'],'program_id':entry['job_id'],'tool_call_id':identifier}
            self.store.event('program.call_started',{'tool':value['tool'],'arguments':value['arguments']},correlation)
            try:
                arguments = value['arguments']
                self._validate_arguments(value['tool'], arguments)
                arguments = self._action_timeout(entry, value['tool'], arguments)
                if self._requires_review(entry, value['tool'], arguments):
                    review = self._review_wait_action if value['tool'] == 'robot_wait' else self._review_action
                    raw, executed_arguments = review(entry, arguments, correlation, identifier, record)
                else:
                    if self._tool_effect(value['tool'], arguments) == 'physical':
                        record.update(action_dispatch_attempted=True, executed_arguments=arguments)
                        self.store.set_state('program_call',identifier,record)
                    raw = self._invoke(value['tool'],arguments,correlation)
                    executed_arguments = arguments
                raw_ok = getattr(raw, 'success', raw.get('success', True) if isinstance(raw, dict) else True) is not False
                raw_kind = getattr(raw, 'kind', raw.get('kind') if isinstance(raw, dict) else None)
                obs_grounded = False
                obs_id_grounded = None
                obs_captured_live = False
                if tool == 'robot_observe' and raw_ok and raw_kind == 'observation':
                    obs_meta = self.services.observation() or {}
                    obs_grounded, obs_id_grounded, obs_captured_live = observation_grounds_current_world(
                        tool,
                        raw,
                        clock=self._embodiment().clock,
                        owner_observation_id=obs_meta.get('observation_id'),
                        current_identity=self._current_identity(),
                        latched_meta=entry.get('_latched_incomplete_meta'),
                    )
                    if obs_grounded:
                        entry['latched_incomplete'] = None
                        entry['_latched_incomplete_meta'] = None
                elif raw_kind == 'action':
                    raw_data = getattr(raw, 'data', raw.get('data') if isinstance(raw, dict) else None)
                    if isinstance(raw_data, dict):
                        rcpt = raw_data.get('receipt') if isinstance(raw_data.get('receipt'), dict) else raw_data
                        req_obj = raw_data.get('request') if isinstance(raw_data.get('request'), dict) else {}
                        phys_adv = rcpt.get('physics_advanced') if 'physics_advanced' in rcpt else raw_data.get('physics_advanced')
                        act_status = rcpt.get('status') if 'status' in rcpt else raw_data.get('status')
                        if phys_adv is not False and act_status not in ('completed', 'rejected'):
                            cp_reason = rcpt.get('checkpoint_reason')
                            if cp_reason is None:
                                cp_reason = raw_data.get('checkpoint_reason')
                            if cp_reason is None and isinstance(rcpt.get('checkpoint_reasons'), dict) and rcpt['checkpoint_reasons']:
                                cp_reason = next(iter(rcpt['checkpoint_reasons'].values()))
                            cp_code = rcpt.get('checkpoint_reason_code')
                            if cp_code is None:
                                cp_code = raw_data.get('checkpoint_reason_code')
                            if cp_code is None and isinstance(rcpt.get('checkpoint_reason'), str):
                                cp_code = rcpt['checkpoint_reason']
                            raw_incomplete = {
                                'action_id': rcpt.get('action_id') or raw_data.get('action_id') or req_obj.get('action_id'),
                                'status': act_status,
                                'reason': rcpt.get('reason') if 'reason' in rcpt else raw_data.get('reason'),
                                'checkpoint_reason': cp_reason,
                                'checkpoint_reason_code': cp_code,
                                'failed_stage_id': rcpt.get('failed_stage_id') if 'failed_stage_id' in rcpt else raw_data.get('failed_stage_id'),
                                'pending_stage_ids': list(
                                    (rcpt.get('pending_stage_ids') if 'pending_stage_ids' in rcpt else raw_data.get('pending_stage_ids')) or []
                                ),
                            }
                            entry['latched_incomplete'] = presentation.present(raw_incomplete)
                            raw_res_obs = rcpt.get('result_observation_id')
                            stop_ok = (
                                rcpt.get('stop_confirmed') is True
                                and not bool(rcpt.get('execution_uncertain'))
                                and act_status != 'unknown'
                            )
                            entry['_latched_incomplete_meta'] = dict(
                                result_observation_id=raw_res_obs if isinstance(raw_res_obs, str) and raw_res_obs else None,
                                stop_confirmed=stop_ok,
                            )
                contract_error = getattr(raw, 'contract_error', None)
                if contract_error is not None:
                    record['output_contract_error'] = deepcopy(contract_error)
                    entry.setdefault('call_errors', []).append(dict(call_id=identifier,
                        tool=value['tool'], reason=contract_error['code'], phase='after_handler'))
                evidence = action_evidence(self.store, value['tool'], executed_arguments, raw, identifier,
                    physical=bool(record.get('action_dispatch_attempted')))
                snapshot_updated = False
                if evidence is not None:
                    entry.setdefault('action_evidence', []).append(evidence)
                    if evidence.get('physics_advanced') is not False:
                        entry['final_observation_id'] = evidence.get('result_observation_id')
                        snapshot_updated = True
                        if entry.get('visual_feedback'):
                            # 物理动作执行后已有最新末帧观测，清除此前未执行的候选 preview 叠加图
                            entry['visual_feedback'][:] = [
                                item for item in entry['visual_feedback']
                                if not (item.get('tool') == 'robot_action'
                                        and item.get('arguments', {}).get('mode', 'preview') == 'preview')
                            ]
                is_crop_view = bool(arguments.get('camera_id') or arguments.get('focus_3d') or arguments.get('roi'))
                observed = obs_id_grounded if obs_grounded and not is_crop_view else None
                if obs_grounded and obs_id_grounded and (obs_captured_live or not is_crop_view):
                    entry['final_observation_id'] = obs_id_grounded
                    snapshot_updated = True
                if snapshot_updated and entry.get('workspace') is not None and (entry['workspace'] / 'scene3d_helpers.py').is_file():
                    self._write_spatial_snapshot(entry['workspace'], presentation, observation_id=entry.get('final_observation_id'))
                result = presentation.present_program_response(raw)
                # Action/current-observation replies already have a final-frame path.
                # Keep other visual replies (including previews and historical crops).
                if evidence is None and not observed and any(
                        item.get('type') == 'inputImage' for item in result.get('contentItems', [])):
                    visuals = entry.setdefault('visual_feedback', [])
                    if value['tool'] == 'robot_action' and arguments.get('mode', 'preview') == 'preview':
                        # 候选搜索循环中仅保留最新一次 robot_action preview 的叠加图，避免向主模型重复推送已淘汰候选图
                        visuals[:] = [item for item in visuals
                                      if not (item.get('tool') == 'robot_action'
                                              and item.get('arguments', {}).get('mode', 'preview') == 'preview')]
                    visuals.append(dict(call_id=identifier,
                        tool=value['tool'], arguments=deepcopy(arguments),
                        reply_ref=self.store.put_artifact(canonical(result).encode(),
                            'program_visual_reply', 'application/json')))
                record.update(status='completed',result=result)
                self.store.set_state('program_call',identifier,record)
                if entry.get('visual_feedback') and entry['visual_feedback'][-1]['call_id'] == identifier:
                    self._publish(entry)
                self.store.event('program.call_completed',{'tool':value['tool'],'operation_id':operation},correlation)
                return result
            except Exception as error:
                from ...core.tool_errors import error_payload
                from ...core.tool_output import ToolResponse, StructuredText
                from ..harness import _internal_error_payload, _wrap_internal_runtime_error, is_internal_runtime_error_compat
                entry.setdefault('call_errors', []).append(dict(call_id=identifier,
                    tool=value.get('tool'), reason=str(error)[:512]))
                if is_internal_runtime_error_compat(error):
                    failure = _internal_error_payload(
                        _wrap_internal_runtime_error(str(value.get('tool') or ''), error),
                        value.get('tool'),
                        value.get('arguments'),
                    )
                else:
                    failure = error_payload(error, tool=value.get('tool'), arguments=value.get('arguments'))
                record.update(status='completed', result=presentation.present_program_response(
                    ToolResponse([StructuredText(failure)], data=failure, kind='error', success=False)))
                if record.get('action_dispatch_attempted'):
                    if self._was_action_admitted(entry, identifier, error):
                        evidence = action_evidence(self.store, value['tool'], record['executed_arguments'],
                            record['result'], identifier, physical=True)
                        if evidence is not None and (not entry.get('action_evidence')
                                or entry['action_evidence'][-1]['call_id'] != identifier):
                            entry.setdefault('action_evidence', []).append(evidence)
                            entry['final_observation_id'] = evidence.get('result_observation_id')
                        if (
                            evidence is not None
                            and evidence.get('physics_advanced') is not False
                            and evidence.get('status') not in ('completed', 'rejected', 'not_executed')
                        ):
                            entry['latched_incomplete'] = presentation.present({
                                'action_id': evidence.get('action_id'),
                                'status': evidence.get('status', 'unknown'),
                                'reason': evidence.get('reason') or str(error),
                                'checkpoint_reason': evidence.get('checkpoint_reason'),
                                'checkpoint_reason_code': evidence.get('checkpoint_reason_code'),
                                'failed_stage_id': evidence.get('failed_stage_id'),
                                'pending_stage_ids': list(evidence.get('pending_stage_ids') or []),
                            })
                            entry['_latched_incomplete_meta'] = dict(
                                result_observation_id=None,
                                stop_confirmed=False,
                            )
                    else:
                        record['action_dispatch_attempted'] = False
                self.store.set_state('program_call',identifier,record)
                self.store.event('program.call_failed',{'tool':value['tool'],'operation_id':operation,
                    'reason':str(error)},correlation)
                return record['result']

    def cancel(self, identifier):
        with self.lock:
            entry = self.entries.get(identifier)
        if entry:
            entry['cancel'].set()
            if entry.get('review_event'):
                with entry['condition']:
                    self._continue_process(entry)
                    entry['review_event'].set()
            self.services.cancel_action(identifier)
        return self.status(identifier)

    def run(self, arguments, correlation, *, on_incomplete_action=_UNSET):
        raw_inline = (
            correlation.get('_raw_inline_source')
            if isinstance(correlation, dict) and isinstance(arguments, dict) and 'source' in arguments
            else None
        )
        if isinstance(correlation, dict) and '_raw_inline_source' in correlation:
            correlation = {k: v for k, v in correlation.items() if k != '_raw_inline_source'}
        if correlation.get('program_id'):
            raise ValueError('program_tool_not_permitted')
        operation = arguments.get('operation', 'start')
        if operation == 'resume':
            if on_incomplete_action is not _UNSET or (set(arguments) - {'operation', 'job_id', 'review_id', 'decision'}):
                raise ValueError('program_resume_cannot_change_invocation')
            return self.resume(arguments.get('job_id'), arguments.get('review_id'),
                               arguments.get('decision'), correlation)
        if operation != 'start':
            raise ValueError('invalid_program_operation')
        if on_incomplete_action is _UNSET:
            inc_policy = arguments.get('on_incomplete_action', 'block')
        else:
            if 'on_incomplete_action' in arguments and arguments['on_incomplete_action'] != on_incomplete_action:
                raise ValueError('invalid_on_incomplete_action')
            inc_policy = on_incomplete_action
        if inc_policy not in ('block', 'allow'):
            raise ValueError('invalid_on_incomplete_action')
        source = raw_inline if isinstance(raw_inline, str) else arguments.get('source')
        if source is None and arguments.get('source_ref'):
            info, content = self.store.resolve_artifact(arguments['source_ref'],allowed_scopes=('actor',))
            if info['kind']!='program_source': raise ValueError('program_source_ref_kind_mismatch')
            source=content.decode()
        timeout = arguments.get('timeout_s',180)
        allow_direct = arguments.get('allow_direct_actions', False)
        if not isinstance(source,str) or not source.strip() or len(source.encode())>128*1024: raise ValueError('invalid_program_source')
        if type(timeout) not in (int,float) or not 1<=timeout<=300: raise ValueError('invalid_program_budget')
        if type(allow_direct) is not bool: raise ValueError('invalid_program_action_authorization')
        # max_calls is accepted for old clients, but is never an execution limit.
        if 'max_calls' in arguments and (type(arguments['max_calls']) is not int or arguments['max_calls']<1):
            raise ValueError('invalid_program_budget')
        digest_args = (
            dict(arguments, on_incomplete_action=inc_policy)
            if on_incomplete_action is not _UNSET and 'on_incomplete_action' not in arguments and inc_policy != 'block'
            else arguments
        )
        source_bytes = source.encode()
        source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        identifier = 'program-'+digest([correlation['thread_id'],correlation['turn_id'],correlation['tool_call_id']])[:24]
        existing = self.store.read_state('program',identifier)
        if existing:
            if existing['input_digest']!=digest(digest_args): raise ValueError('program_id_content_conflict')
            if isinstance(raw_inline, str) and existing.get('source_sha256') and existing['source_sha256'] != source_sha256:
                raise ValueError('program_id_content_conflict')
            return self.status(identifier)
        ws_parent = self._isolated_ws_root if self._isolated_ws_root is not None else self.root
        workspace = ws_parent / identifier
        workspace.mkdir()
        (workspace/'main.py').write_text(source)
        shutil.copyfile(Path(__file__).with_name('ni_rua_client.py'),workspace/'ni_rua_client.py')
        if self.services.workspace_modules:
            from ...core.presentation.actor_presentation import ActorPresentation
            from ...perception.external.scene3d import copy_workspace_modules
            if copy_workspace_modules(self.services.workspace_modules, workspace, profile=self.services.embodiment):
                self._write_spatial_snapshot(workspace, ActorPresentation(self.store))
        source_ref = self.store.put_artifact(source_bytes,'program_source','text/x-python')
        ident = self._current_identity()
        if not isinstance(ident, dict) and isinstance(correlation, dict) and ('run_id' in correlation or 'episode_id' in correlation):
            ident = {k: correlation[k] for k in ('run_id', 'episode_id', 'env_instance_id', 'env_generation') if k in correlation}
        job = {'job_id':identifier,'identity':deepcopy(ident) if isinstance(ident, dict) else None,'status':'running','source_ref':source_ref,'source_sha256':source_sha256,'input_digest':digest(digest_args),'correlation':correlation,'created_at':utc_now(),'timeout_s':timeout,'timeout_semantics':'work_wall_time_excluding_own_model_and_main_review_waits','call_limit':None,'allow_direct_actions':allow_direct,
               'on_incomplete_action':inc_policy,'action_authorization_scope':'this_program_invocation_only','review_decisions':{}}
        self.store.set_state('program',identifier,job)
        self.store.event('program.started',job,correlation)
        entry = {'job_id':identifier,'correlation':correlation,'cancel':threading.Event(),'call_lock':threading.Lock(),'calls':0,
                 'deadline':time.monotonic()+timeout,'budget_lock':threading.RLock(),'model_wait_started_at':None,'model_wait_s':0.0,
                 'review_wait_started_at':None,'review_wait_s':0.0,'allow_direct_actions':allow_direct,
                 'on_incomplete_action':inc_policy,'latched_incomplete':None,'_latched_incomplete_meta':None,
                 'condition':threading.Condition(threading.RLock()),'review_event':threading.Event(),
                 'process_ready':threading.Event(),'process_paused':False,'workspace':workspace,'job':job,'action_evidence':[]}
        with self.lock: self.entries[identifier]=entry
        worker = threading.Thread(target=self._execute_guarded, args=(entry,), daemon=True)
        entry['worker'] = worker
        worker.start()
        return self._await_boundary(entry)

    def _write_spatial_snapshot(self, workspace, presentation, *, observation_id=_UNSET):
        """向含 `scene3d_helpers.py` 的沙盒工作区原子写入与当前有效观测对齐、经脱敏校验的空间快照。"""
        snap_path = workspace / '_spatial_snapshot.json'
        cloud_path = workspace / '_spatial_cloud.npz'
        obs_meta = self.services.observation()
        live_obs_id = obs_meta.get('observation_id') if observation_id is _UNSET else observation_id
        profile = obs_meta.get('observation_profile', LEGACY_ACTOR_PROFILE)
        raw_snap = self.store.read_state('perception', 'spatial_snapshot') if live_obs_id else None
        if not live_obs_id or not isinstance(raw_snap, dict) or raw_snap.get('observation_id') != live_obs_id:
            snap_path.unlink(missing_ok=True)
            cloud_path.unlink(missing_ok=True)
            return

        has_cloud = False
        cloud_ref = raw_snap.get('cloud_artifact')
        if isinstance(cloud_ref, str) and cloud_ref:
            info, content = self.store.resolve_artifact(cloud_ref, allowed_scopes=('actor', 'internal'))
            if len(content) <= 7 * 1024 * 1024:
                _, safe_bytes = presentation.present_artifact(info, content)
                tmp_cloud = workspace / '_spatial_cloud.npz.tmp'
                tmp_cloud.write_bytes(safe_bytes)
                tmp_cloud.replace(cloud_path)
                has_cloud = True
        if not has_cloud:
            cloud_path.unlink(missing_ok=True)

        rs = deepcopy(raw_snap.get('robot_state')) if isinstance(raw_snap.get('robot_state'), dict) else None
        if rs is not None and profile == LEGACY_ACTOR_PROFILE:
            for arm_state in (rs.get('arms') or {}).values():
                if isinstance(arm_state, dict) and isinstance(arm_state.get('gripper'), dict):
                    arm_state['gripper'].pop('position_m', None)

        payload = {
            'observation_id': live_obs_id,
            'object_table': deepcopy(raw_snap.get('object_table')),
            'table_z': raw_snap.get('table_z'),
            'robot_state': rs,
            'cavity_depths_m': deepcopy(raw_snap.get('cavity_depths_m') or {}),
            'surface_slices_m': deepcopy(raw_snap.get('surface_slices_m') or {}),
            'persisted_objects': deepcopy(raw_snap.get('persisted_objects') or []),
            'has_cloud': has_cloud,
        }
        for opt_key in (
            'field_rev',
            'calib_rev',
            'view_coverage',
            'uncertainty_xyz_m',
            'newly_revealed_ids',
        ):
            if opt_key in raw_snap:
                payload[opt_key] = deepcopy(raw_snap[opt_key])
        reject_privileged(payload, profile)
        scrubbed = presentation.present(payload)
        tmp_snap = workspace / '_spatial_snapshot.json.tmp'
        tmp_snap.write_text(canonical(scrubbed), encoding='utf-8')
        tmp_snap.replace(snap_path)

    def _execute_guarded(self, entry):
        try:
            self._execute(entry)
        except Exception as error:
            # Includes failures before Popen (socket/spec/log setup). Never leave
            # the main actor waiting forever on a supervisor which has exited.
            entry['cancel'].set()
            entry['process_ready'].set()
            entry['review_event'].set()
            self._continue_process(entry)
            for channel in entry.get('channel_sockets', ()):
                try: channel.shutdown(socket.SHUT_RDWR)
                except OSError: pass
                channel.close()
            broker = entry.get('broker')
            if broker:
                broker.join(1)
            process = entry.get('process')
            stopped = process is None or process.poll() is not None
            if not stopped:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(2)
                    stopped = True
                except (OSError, subprocess.TimeoutExpired):
                    pass
            active = self.services.cancel_action(entry['job_id'])
            stopped = stopped and not active and not (broker and broker.is_alive())
            self._publish(entry, status='failed' if stopped else 'unknown', reason=str(error),
                          stop_confirmed=bool(stopped), finished_at=utc_now())
            with self.lock:
                if not stopped:
                    self._unresolved_handles[entry['job_id']] = {
                        'pgid': process.pid if process is not None else None,
                        'process': process,
                        'broker': broker,
                        'no_process_spawned': process is None,
                    }
                self.entries.pop(entry['job_id'], None)

    def _execute(self, entry):
        """Own the subprocess until terminal acknowledgement, across review yields."""
        identifier, workspace, job = entry['job_id'], entry['workspace'], entry['job']
        correlation = entry['correlation']
        parent,child=socket.socketpair()
        entry['channel_sockets'] = (parent, child)
        broker=threading.Thread(target=self.broker,args=(parent,entry),daemon=True)
        entry['broker'] = broker
        broker.start()
        if self.services.sandbox_enabled or (self.services.seccomp_enabled and sys.platform.startswith('linux')):
            with self.runtime_lock:
                if self.runtime is None:
                    self.runtime = program_runtime(self.store.root / 'private' / 'program-runtime',
                        sandbox_enabled=True)
                runtime, packages, read_paths = self.runtime
            env = {'PATH':os.defpath,'HOME':str(workspace),'TMPDIR':str(workspace),
                   'PYTHONHOME':str(runtime),'PYTHONPATH':str(workspace)+os.pathsep+str(packages),
                   'PYTHONDONTWRITEBYTECODE':'1','NI_RUA_PROGRAM_FD':str(child.fileno())}
        else:
            read_paths = []
            py_path = str(workspace)
            if os.environ.get('PYTHONPATH'):
                py_path = py_path + os.pathsep + os.environ['PYTHONPATH']
            env = {'PATH':os.defpath,'HOME':str(workspace),'TMPDIR':str(workspace),
                   'PYTHONPATH':py_path,
                   'PYTHONDONTWRITEBYTECODE':'1','NI_RUA_PROGRAM_FD':str(child.fileno())}
        from ...core.config import DEFAULT_SUBPROCESS_THREAD_ENV
        env.update(DEFAULT_SUBPROCESS_THREAD_ENV)
        limits = {'RLIMIT_FSIZE':8*1024*1024,'RLIMIT_AS':1024*1024*1024}
        unenforced = {}
        if sys.platform == 'darwin':
            # macOS setrlimit(RLIMIT_AS) fails ("current limit exceeds maximum limit") and would abort
            # every launch in preexec_fn; record the memory cap as unenforced instead of hiding it.
            limits.pop('RLIMIT_AS')
            unenforced['RLIMIT_AS'] = 'macOS cannot cap process address space; 1 GiB memory limit not enforced'
        spec = self.store.root/'private'/('isolation-'+identifier+'.json')
        spec.parent.mkdir(exist_ok=True)
        spec.write_text(canonical({'sandbox_enabled':self.services.sandbox_enabled,'seccomp_enabled':self.services.seccomp_enabled,'read_paths':read_paths,'write_paths':[str(workspace),'/dev/null','/dev/urandom'],'tcp_ports':[],'deny_socket_creation':True,'bounded_program':True,'limits':limits,'env':env,
                                   **({'unenforced_limits':unenforced} if unenforced else {})}))
        spec.chmod(0o600)
        output, errors = self.logs/(identifier+'.stdout'), self.logs/(identifier+'.stderr')
        process = None
        def _apply_limits():
            for name, value in limits.items():
                limit = getattr(resource, name, None)
                if limit is not None:
                    _, hard = resource.getrlimit(limit)
                    cap = int(value) if hard == resource.RLIM_INFINITY else min(int(value), hard)
                    resource.setrlimit(limit, (cap, cap))
        try:
            with os.fdopen(os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), 'wb') as stdout, \
                    os.fdopen(os.open(errors, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), 'wb') as stderr:
                if self.services.sandbox_enabled:
                    process = subprocess.Popen([sys.executable,str(Path(__file__).with_name('isolation.py')),str(spec),sys.executable,'-S',str(workspace/'main.py')],cwd=Path(__file__).resolve().parents[3],stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True,pass_fds=(child.fileno(),))
                elif self.services.seccomp_enabled and sys.platform.startswith('linux'):
                    process = subprocess.Popen([sys.executable,str(Path(__file__).with_name('isolation.py')),str(spec),sys.executable,str(workspace/'main.py')],cwd=str(workspace),stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True,pass_fds=(child.fileno(),))
                else:
                    process = subprocess.Popen([sys.executable,str(workspace/'main.py')],cwd=str(workspace),env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True,pass_fds=(child.fileno(),),preexec_fn=_apply_limits)
                self.last_activity_at = time.monotonic()
                output_sizes = (0, 0)
                child.close()
                entry['process']=process
                entry['process_ready'].set()
                while process.poll() is None:
                    sizes = (os.fstat(stdout.fileno()).st_size, os.fstat(stderr.fileno()).st_size)
                    if sizes != output_sizes:
                        self.last_activity_at = time.monotonic()
                        output_sizes = sizes
                    if entry['cancel'].is_set() or program_remaining(entry) <= 0:
                        entry['cancel'].set()
                        self.cancel(identifier)
                        self._continue_process(entry)
                        # The program can end by itself after the poll above (e.g. on its cancelled reply). macOS
                        # answers EPERM, not ESRCH, while every group member has exited but none is reaped yet.
                        # Neither answer is a stop confirmation: process.wait is, and a live group that also
                        # refuses SIGKILL raises below and stays unknown.
                        try: os.killpg(process.pid,signal.SIGTERM)
                        except (ProcessLookupError, PermissionError): pass
                        try: process.wait(STOP_CONFIRM_S)
                        except subprocess.TimeoutExpired:
                            try: os.killpg(process.pid,signal.SIGKILL)
                            except (ProcessLookupError, PermissionError) as refused:
                                # ESRCH when another thread just reaped the exited leader; on macOS, EPERM while the
                                # exited leader is still unreaped. Only process.wait confirms the stop; a live program
                                # refusing SIGKILL raises and stays unknown.
                                try: process.wait(STOP_CONFIRM_S)
                                except subprocess.TimeoutExpired: raise refused from None
                            else: process.wait()
                        break
                    time.sleep(.02)
                self.last_activity_at = time.monotonic()
                job.update(returncode=process.returncode)   # recorded before the descendant stop below
                # Kill descendants even after a parent exits: no detached actor jobs. EPERM is no verdict (macOS
                # answers it while a descendant has exited but launchd has not reaped it): only the group's
                # disappearance confirms the stop; a group still answering is a live descendant refusing the signal.
                descendants_stopped = True
                try: os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                except PermissionError: descendants_stopped = self._group_gone(process.pid, STOP_CONFIRM_S)
                # A cancelled action owns a separate backend operation; wait for its acknowledgement.
                deadline=time.monotonic()+45
                while self.services.action_active(identifier) and time.monotonic()<deadline: time.sleep(.05)
                unknown=self.services.action_active(identifier) or not descendants_stopped
                final_status='unknown' if unknown else ('cancelled' if entry['cancel'].is_set() else 'succeeded' if process.returncode==0 else 'failed')
                job.update(returncode=process.returncode,stop_confirmed=not unknown,finished_at=utc_now())
                if not descendants_stopped:
                    job.update(reason='program_descendant_stop_unconfirmed')
        except Exception as error:
            final_status='unknown' if process and process.poll() is None else 'failed'
            job.update(reason=str(error),stop_confirmed=not bool(process and process.poll() is None),finished_at=utc_now())
        finally:
            child.close()
            entry['cancel'].set()
            entry['process_ready'].set()
            self._continue_process(entry)
            entry['review_event'].set()
            try: parent.shutdown(socket.SHUT_RD)
            except OSError: pass
            broker.join(45)
            if broker.is_alive():
                final_status='unknown'
                job.update(stop_confirmed=False,reason='program_broker_operation_stop_unconfirmed')
            job.pop('pending_review', None)
            self._publish(entry, status=final_status)
            self.store.event('program.'+final_status,deepcopy(job),correlation)
            with self.lock:
                if final_status == 'unknown':
                    self._unresolved_handles[identifier] = {
                        'pgid': process.pid if process is not None else None,
                        'process': process,
                        'broker': broker,
                        'no_process_spawned': process is None,
                    }
                self.entries.pop(identifier,None)

    def close(self):
        with self.lock:
            entries = list(self.entries.values())
        self.cancel_all()
        deadline = time.monotonic()+50
        for entry in entries:
            worker = entry.get('worker')
            if worker and worker is not threading.current_thread():
                worker.join(max(0.0, deadline-time.monotonic()))
        if getattr(self, '_owns_isolated_ws_root', False) and getattr(self, '_isolated_ws_root', None) is not None:
            shutil.rmtree(self._isolated_ws_root, ignore_errors=True)
        return self.snapshot()
