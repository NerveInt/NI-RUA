"""Config-driven remote GPU services: connect over SSH, reuse or start each service, keep health flowing, release on exit.

`RemoteManager` is built from the validated `remote` config section (ni_rua/core/config.py: RemoteConfig). It wraps the
tunnel `Supervisor` (ni_rua/apps/remote/tunnel.py) and adds the remote side:

* per service: health through the forward first; a healthy service is reused (never stopped by us unless we started it);
  otherwise the configured start command runs detached on the host (GPU chosen by free memory when asked for), and health
  is polled until the service answers or the startup timeout runs out;
* a monitor thread reopens a dropped tunnel with backoff and refreshes health every `health_interval_s`;
* `status()` is the operator view (tunnel, per-service availability and reason, origin reused/started, GPU, UTC time).
  It, and the provenance record file, never enter model context; neither carries hashes of any kind in `status()`.

Nothing here touches the robot. A service that is unavailable only means perception reports unavailable/unknown.
"""
from __future__ import annotations

import asyncio
import copy
import re
import shlex
import subprocess
import sys
import threading
import time
from collections import namedtuple

from .tunnel import HEALTH_FIELDS, Supervisor, utc_now

Result = namedtuple("Result", "returncode stdout stderr")
LOG_TAIL_LINES = 12
REASON_LIMIT_CHARS = 400


class RemoteError(RuntimeError):
    pass


class SshRunner:
    """Run a bash script on the host over ssh (BatchMode: never prompts). The script goes through stdin, so no quoting layer."""

    def __init__(self, host, ssh="ssh", connect_timeout_s=10.0):
        self.host, self.ssh, self.connect_timeout_s = host, ssh, connect_timeout_s

    def run(self, script, timeout_s):
        argv = [self.ssh, "-o", "BatchMode=yes", "-o", f"ConnectTimeout={int(self.connect_timeout_s)}", self.host, "bash -s"]
        try:
            done = subprocess.run(argv, input=script, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return Result(124, "", f"ssh command exceeded {timeout_s:g} s")
        except OSError as error:
            return Result(127, "", str(error))
        return Result(done.returncode, done.stdout, done.stderr)


def sh_path(value):
    """Shell word for a remote path or value: a leading ~ is the remote home, everything else is quoted literally."""
    if value == "~":
        return '"$HOME"'
    if value.startswith("~/"):
        return '"$HOME"/' + shlex.quote(value[2:])
    return shlex.quote(value)


def env_exports(env, gpu=None):
    lines = []
    for item in env:
        name, value = item.split("=", 1)
        lines.append(f"export {name}={sh_path(value)}")
    if gpu is not None:
        lines.append(f"export CUDA_VISIBLE_DEVICES={shlex.quote(gpu)}")
    return lines


def workdir_line(service):
    if service.workdir is None:
        return []
    return [f"cd {sh_path(service.workdir)} 2>/dev/null || {{ echo 'NI_RUA_ERR workdir_missing'; exit 3; }}"]


def markers(stdout):
    """`NI_RUA_KEY value` lines printed by the scripts below, as {KEY: value}."""
    found = {}
    for line in stdout.splitlines():
        if line.startswith("NI_RUA_"):
            key, _, value = line.partition(" ")
            found[key] = value.strip()
    return found


GPU_QUERY_SCRIPT = "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits\n"


def parse_gpus(stdout):
    """[(index, free_mib)] from `nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits`; bad lines skipped."""
    gpus = []
    for line in stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            gpus.append((int(parts[0]), int(parts[1])))
    return gpus


def pick_gpu(gpus, spec, min_free_mib):
    """(CUDA_VISIBLE_DEVICES value, None) or (None, reason). Every chosen GPU must have min_free_mib free."""
    if not gpus:
        return None, "no_gpu_reported_by_nvidia_smi"
    free = dict(gpus)
    if spec == "auto":
        index, best = max(gpus, key=lambda item: (item[1], -item[0]))
        if best < min_free_mib:
            return None, f"no_gpu_with_{min_free_mib}_MiB_free: most free is GPU {index} with {best} MiB"
        return str(index), None
    chosen = [int(item) for item in spec.split(",")]
    for index in chosen:
        if index not in free:
            return None, f"gpu_{index}_not_present"
        if free[index] < min_free_mib:
            return None, f"gpu_{index}_has_{free[index]}_MiB_free_below_required_{min_free_mib}"
    return spec, None


def start_script(service, log_dir, gpu):
    """Detached start: refuses a busy remote port, runs `command` under setsid nohup, prints NI_RUA_PID and NI_RUA_LOG."""
    command = shlex.quote(service.command)
    wrapper = shlex.quote('bash -c "$1"; echo $? > "$2"')
    return "\n".join([
        "set -u", *env_exports(service.env, gpu), *workdir_line(service),
        f"if ss -ltn 2>/dev/null | awk 'NR>1{{print $4}}' | grep -Eq '(^|:){service.remote_port}$'; then "
        "echo 'NI_RUA_ERR remote_port_busy_but_not_healthy'; exit 4; fi",
        f"LOGDIR={sh_path(log_dir)}",
        "mkdir -p \"$LOGDIR\" || { echo 'NI_RUA_ERR log_dir_unwritable'; exit 5; }",
        f"LOG=\"$LOGDIR/{service.name}.{service.remote_port}.start.log\"",
        "rm -f \"$LOG.exit\"",
        f"setsid nohup bash -c {wrapper} _ {command} \"$LOG.exit\" >> \"$LOG\" 2>&1 < /dev/null &",
        "echo \"NI_RUA_PID $!\"", "echo \"NI_RUA_LOG $LOG\"", ""])


def launch_state_script(pid, log):
    """Whether the detached start command is still alive and the exit code it left behind, if any."""
    return "\n".join([
        f"LOG={shlex.quote(log)}", f"pid={int(pid)}",
        "if [ -s \"$LOG.exit\" ]; then echo \"NI_RUA_EXIT $(cat \"$LOG.exit\")\"; fi",
        "if kill -0 \"$pid\" 2>/dev/null; then echo 'NI_RUA_ALIVE 1'; else echo 'NI_RUA_ALIVE 0'; fi", ""])


def tail_script(log):
    return f"tail -n {LOG_TAIL_LINES} {shlex.quote(log)} 2>/dev/null\n"


def stop_script(service, pid):
    """Stop what this process started: the configured stop command, else the process group of the detached start command."""
    lines = ["set -u", *env_exports(service.env), *workdir_line(service)]
    if service.stop:
        lines += [f"bash -c {shlex.quote(service.stop)}", "echo \"NI_RUA_RC $?\""]
    else:
        polls = max(1, int(service.stop_timeout_s * 2))
        lines += [
            f"pid={int(pid)}",
            "if kill -0 \"$pid\" 2>/dev/null && [ \"$(ps -o sid= -p \"$pid\" | tr -d ' ')\" = \"$pid\" ]; then",
            "  kill -TERM -- \"-$pid\"",
            f"  for i in $(seq 1 {polls}); do kill -0 \"$pid\" 2>/dev/null || break; sleep 0.5; done",
            "  if kill -0 \"$pid\" 2>/dev/null; then kill -KILL -- \"-$pid\"; echo 'NI_RUA_KILLED 1'; else echo 'NI_RUA_STOPPED 1'; fi",
            "else echo 'NI_RUA_NOT_RUNNING 1'; fi"]
    return "\n".join(lines + [""])


def revision_script(repo):
    return (f"git -C {sh_path(repo)} rev-parse HEAD && git -C {sh_path(repo)} status --porcelain -- services | head -1\n")


def compact(text, limit=REASON_LIMIT_CHARS):
    return re.sub(r"\s+", " ", text).strip()[-limit:]


class RemoteManager:
    @staticmethod
    def _console(text):
        try:
            print(f"[remote {utc_now()}] {text}", file=sys.stderr, flush=True)
        except (OSError, ValueError):   # dead tty / closed pipe after a hangup must not abort tunnel/service teardown
            pass

    def __init__(self, config, *, record=None, say=None, runner=None, supervisor_factory=Supervisor):
        self.config, self.record_path = config, record
        self.say = say or self._console
        self.runner = runner or SshRunner(config.host, config.ssh, config.connect_timeout_s)
        self.supervisor_factory = supervisor_factory
        self.supervisor = None
        self.lifecycle = "created"   # created -> starting -> running -> stopped
        self.tunnel_error = None
        self._lock = threading.RLock()
        self._halt = threading.Event()
        self._thread = None
        self._stopped = False
        self._services = {item.name: dict(
            name=item.name, remote_port=item.remote_port, local_port=item.effective_local_port, available=False,
            reason="not_checked_yet", origin="none", started_by_us=False, gpu=None, remote_pid=None, remote_log=None,
            phase="pending", failure=None, blocked=False, checked_at=None, health={}) for item in config.services}

    # ---- lifecycle -------------------------------------------------------------------------------------------------
    def start(self):
        """Connect, ensure every service (blocking, bounded by startup_timeout_s per service), start the monitor.

        Raises RemoteError only when config.required is true and a service is not healthy afterwards.
        """
        config = self.config
        urls = {item.name: f"http://127.0.0.1:{item.effective_local_port}{item.health_path}" for item in config.services}
        forwards = [(item.effective_local_port, item.remote_port) for item in config.services]
        with self._lock:
            self.lifecycle = "starting"
        self.supervisor = self.supervisor_factory(
            config.host, forwards, urls, record=self.record_path, ssh=config.ssh, interval_s=config.health_interval_s,
            connect_timeout_s=config.connect_timeout_s)
        self.supervisor.say = self.say
        if self._halt.is_set():
            self.supervisor.stopping = True
            return
        if config.remote_repo:
            self.supervisor.record["remote_checkout"] = self._revision()
        self._bring_up()
        if self._halt.is_set():
            return
        with self._lock:
            self.lifecycle = "running"
        self._thread = threading.Thread(target=self._monitor, name="remote-monitor", daemon=True)
        self._thread.start()
        failed = {name: item["reason"] for name, item in self._services.items() if not item["available"]}
        if failed and config.required and not self._halt.is_set():
            raise RemoteError("required remote services unavailable: " + "; ".join(f"{name}: {reason}" for name, reason in failed.items()))

    def abort(self):
        """Ask a running start()/monitor to stop waiting (signal handler safe)."""
        self._halt.set()
        if self.supervisor is not None:
            self.supervisor.stopping = True

    def wait(self):
        self._halt.wait()

    def stop(self):
        """Idempotent teardown: stop the monitor, stop services we started (only with stop_on_exit), close our tunnel."""
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
        self.abort()
        if self._thread is not None:
            # Each SSH/health operation is bounded. Wait for the owner to finish before closing
            # its tunnel or stopping services, otherwise a late start can leak past teardown.
            self._thread.join()
        if self.supervisor is None:
            with self._lock:
                self.lifecycle = "stopped"
            return
        for item in self.config.services:
            state = self._services[item.name]
            if not state["started_by_us"]:
                continue
            if self.config.stop_on_exit:
                self._stop_service(item, state)
            else:
                self.say(f"{item.name}: left running on {self.config.host} (remote.stop_on_exit is false)")
                self._note(item.name, stop="left_running")
        try:
            # The record keeps the last health seen while serving; a check after the stop would only record the outage.
            for name, notes in self.supervisor.service_notes.items():
                self.supervisor.record["services"].setdefault(name, {}).update(notes)
        finally:
            self.supervisor.close()
            with self._lock:
                self.lifecycle = "stopped"

    # ---- status ------------------------------------------------------------------------------------------------------
    def status(self):
        """Operator view of the tunnel and every service; JSON-ready; no hashes, nothing model-facing."""
        with self._lock:
            services = {name: {key: copy.deepcopy(value) for key, value in item.items() if key not in ("phase", "failure", "blocked")}
                        for name, item in self._services.items()}
            sup = self.supervisor
            if sup is None:
                tunnel = "not_started"
            elif self.lifecycle == "stopped":
                tunnel = "closed"
            else:
                tunnel = ("external" if sup.external else "up") if sup.tunnel_alive() else "down"
            if self.lifecycle in ("created", "starting"):
                overall = "starting"
            elif self.lifecycle == "stopped":
                overall = "stopped"
            else:
                up = [item["available"] for item in services.values()]
                overall = "ready" if all(up) else "degraded" if any(up) else "unavailable"
            checked = [item["checked_at"] for item in services.values() if item["checked_at"]]
            return {"enabled": True, "host": self.config.host, "state": overall, "required": self.config.required,
                    "stop_on_exit": self.config.stop_on_exit,
                    "tunnel": {"state": tunnel, "restarts": sup.restarts if sup else 0, "error": self.tunnel_error,
                               "forwards": list(sup.record["forwards"]) if sup else []},
                    "services": services, "checked_at": max(checked) if checked else None}

    # ---- internals ---------------------------------------------------------------------------------------------------
    def _set(self, name, **values):
        with self._lock:
            self._services[name].update(values)

    def _note(self, name, **values):
        self.supervisor.service_notes.setdefault(name, {}).update(values)

    def _revision(self):
        done = self.runner.run(revision_script(self.config.remote_repo), self.config.command_timeout_s)
        lines = done.stdout.splitlines()
        if done.returncode != 0 or not lines:
            return None
        return {"repo": self.config.remote_repo, "revision": lines[0].strip(), "services_dirty": len(lines) > 1}

    def _apply(self, health):
        """Fold one round of health answers into the service states."""
        now = utc_now()
        with self._lock:
            for name, answer in health.items():
                state = self._services[name]
                state["checked_at"], state["available"] = now, bool(answer.get("available")) and not state["blocked"]
                state["health"] = {key: answer[key] for key in HEALTH_FIELDS if key in answer}
                if state["available"]:
                    state["reason"] = None
                    if state["origin"] == "none":
                        state["origin"] = "reused"   # answered without us starting anything
                else:
                    state["reason"] = state["failure"] or answer.get("reason") or "health_not_ok"
                if state["blocked"]:
                    state["reason"] = state["failure"]

    def _bring_up(self):
        sup = self.supervisor
        if not sup.tunnel_alive():
            try:
                sup.open()
                self.tunnel_error = None
            except RuntimeError as error:
                self.tunnel_error = str(error)
                self.say(f"tunnel unavailable: {error}")
                for name in self._services:
                    self._set(name, failure=f"tunnel_unavailable: {error}")
                sup.check()
                self._apply(sup.record["services"])
                return
        for item in self.config.services:
            if self._services[item.name]["phase"] == "pending":
                self._ensure(item)
        sup.check()
        self._apply(sup.record["services"])

    def _fail(self, item, reason):
        self.say(f"{item.name}: {reason}")
        self._set(item.name, phase="failed", failure=reason)
        self._note(item.name, start_failure=reason)

    def _ensure(self, item):
        sup, name, config = self.supervisor, item.name, self.config
        if self._halt.is_set():
            return
        self._set(name, failure=None)
        answer = sup.health_fn(sup.health_urls[name])
        if answer.get("available"):
            if item.reuse_if_healthy:
                self.say(f"{name}: already healthy on {config.host}:{item.remote_port}; reusing (never stopped by us)")
                self._set(name, phase="ready", origin="reused")
                self._note(name, origin="reused", started_by_us=False)
            else:
                self._fail(item, "already_running_and_reuse_if_healthy_is_false")
                self._set(name, blocked=True)
            return
        if item.command is None:
            self._fail(item, "not_running_and_no_start_command_configured")
            return
        gpu = None
        if item.gpu is not None:
            probe = self.runner.run(GPU_QUERY_SCRIPT, config.command_timeout_s)
            if probe.returncode != 0:
                self._fail(item, "gpu_probe_failed: " + compact(probe.stderr or f"exit {probe.returncode}"))
                return
            gpu, refusal = pick_gpu(parse_gpus(probe.stdout), item.gpu, item.gpu_min_free_mib)
            if refusal:
                self._fail(item, refusal)
                return
        if self._halt.is_set():
            return
        done = self.runner.run(start_script(item, config.log_dir, gpu), config.command_timeout_s)
        found = markers(done.stdout)
        if done.returncode != 0 or "NI_RUA_PID" not in found:
            self._fail(item, found.get("NI_RUA_ERR") or "start_command_not_launched: " + compact(done.stderr or f"exit {done.returncode}"))
            return
        pid, log = int(found["NI_RUA_PID"]), found.get("NI_RUA_LOG")
        self.say(f"{name}: started on {config.host} gpu={gpu} pid={pid} log={log}")
        self._set(name, origin="started", started_by_us=True, gpu=gpu, remote_pid=pid, remote_log=log)
        self._note(name, origin="started", started_by_us=True, gpu=gpu, remote_pid=pid, remote_log=log)
        deadline = time.monotonic() + item.startup_timeout_s
        while True:
            if self._halt.is_set():
                self._fail(item, "aborted_while_starting")
                return
            if sup.health_fn(sup.health_urls[name]).get("available"):
                self.say(f"{name}: healthy")
                self._set(name, phase="ready")
                return
            if time.monotonic() >= deadline:
                self._fail(item, f"startup_timeout_after_{item.startup_timeout_s:g}s" + self._log_tail(log))
                return
            state = markers(self.runner.run(launch_state_script(pid, log), config.command_timeout_s).stdout)
            exit_code = state.get("NI_RUA_EXIT")
            if exit_code not in (None, "0"):
                self._fail(item, f"start_command_exit_{exit_code}" + self._log_tail(log))
                return
            if state.get("NI_RUA_ALIVE") == "0" and exit_code is None:
                self._fail(item, "start_process_vanished" + self._log_tail(log))
                return
            self._halt.wait(config.poll_interval_s)

    def _log_tail(self, log):
        if not log:
            return ""
        text = self.runner.run(tail_script(log), self.config.command_timeout_s).stdout
        return f"; log tail: {compact(text)}" if text.strip() else ""

    def _stop_service(self, item, state):
        done = self.runner.run(stop_script(item, state["remote_pid"]), item.stop_timeout_s + self.config.command_timeout_s)
        found = markers(done.stdout)
        outcome = (next((key.removeprefix("NI_RUA_").lower() for key in found if key != "NI_RUA_RC"), None)
                   or (f"stop_command_rc_{found['NI_RUA_RC']}" if "NI_RUA_RC" in found else f"ssh_exit_{done.returncode}"))
        self.say(f"{item.name}: stop {outcome}")
        self._note(item.name, stop=outcome, stopped_at=utc_now())

    def _tick(self):
        """One monitor round: reopen a dead tunnel, ensure services still pending, refresh health. Returns seconds to wait."""
        sup = self.supervisor
        delay = sup.step()
        if sup.tunnel_alive():
            pending = [item for item in self.config.services if self._services[item.name]["phase"] == "pending"]
            for item in pending:
                self._ensure(item)
            if pending:
                sup.check()
        self._apply(sup.record["services"])
        with self._lock:
            self.tunnel_error = None if sup.tunnel_alive() else self.tunnel_error or "tunnel_down"
        return delay

    def _monitor(self):
        delay = self.config.health_interval_s
        while not self._halt.wait(delay):
            try:
                delay = self._tick()
            except Exception as error:   # the monitor must never die: perception just stays unavailable
                self.say(f"monitor error: {error!r}")
                delay = self.config.health_interval_s


async def start_services(manager, done):
    """Run manager.start() off the event loop. A shutdown signal (`done` set) aborts the wait.

    Raises RemoteError only when remote.required is true and a service is not healthy.
    """
    task = asyncio.ensure_future(asyncio.to_thread(manager.start))
    waiter = asyncio.ensure_future(done.wait())
    try:
        await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
        if not task.done():
            manager.abort()
        await asyncio.shield(task)
    except asyncio.CancelledError:
        manager.abort()
        await asyncio.shield(task)
        raise
    finally:
        waiter.cancel()
