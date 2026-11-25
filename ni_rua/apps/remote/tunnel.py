"""Keep SSH local port forwards to a remote GPU host open and record what they serve.

The remote services stay bound to 127.0.0.1 on the server; this only forwards local loopback ports to them. The record
written with --record is operator / manifest provenance (host, forwards, health versions, timestamps in UTC) and is
never shown to the model.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Fields of GET /health kept in the record: service versions and the serving device. Counters and timings are left out.
HEALTH_FIELDS = ("status", "version", "object_table", "points", "held", "object_table_backend", "sam31", "device",
                 "cuda_visible_devices", "started")


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def forward_pairs(ports):
    """Ports as (local, remote) pairs: a bare port P forwards 127.0.0.1:P to the host's 127.0.0.1:P."""
    return [(int(item[0]), int(item[1])) if isinstance(item, (tuple, list)) else (int(item), int(item)) for item in ports]


def ssh_command(host, ports, *, ssh="ssh", alive_interval_s=15, alive_count=3, connect_timeout_s=None):
    """`ssh -N` with one `-L 127.0.0.1:L:127.0.0.1:R` per port (bare port P = L and R both P); fails fast if a local port cannot be bound."""
    command = [ssh, "-N", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
               "-o", f"ServerAliveInterval={int(alive_interval_s)}", "-o", f"ServerAliveCountMax={int(alive_count)}"]
    if connect_timeout_s is not None:
        command += ["-o", f"ConnectTimeout={int(connect_timeout_s)}"]
    for local, remote in forward_pairs(ports):
        command += ["-L", f"127.0.0.1:{local}:127.0.0.1:{remote}"]
    return command + [host]


def port_open(port, host="127.0.0.1", timeout_s=0.5):
    try:
        with socket.create_connection((host, int(port)), timeout=timeout_s):
            return True
    except OSError:
        return False


def health(url, timeout_s=5.0):
    """{available, ...HEALTH_FIELDS} or {available: False, reason} for one GET /health."""
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            answer = json.loads(response.read())
    except (urllib.error.URLError, OSError, ValueError) as error:
        return {"available": False, "reason": type(error).__name__ + ":" + str(getattr(error, "reason", error))[:160]}
    if not isinstance(answer, dict):
        return {"available": False, "reason": "invalid_health_reply"}
    out = {"available": answer.get("status") == "ok"}
    out.update({key: answer[key] for key in HEALTH_FIELDS if key in answer})
    return out


def remote_revision(host, repo, ssh="ssh", timeout_s=15):
    """`git rev-parse HEAD` and dirty flag of the service checkout on the remote host, or None."""
    try:
        done = subprocess.run([ssh, "-o", "BatchMode=yes", host,
                               f"git -C {repo} rev-parse HEAD && git -C {repo} status --porcelain -- services | head -1"],
                              capture_output=True, text=True, timeout=timeout_s)
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = done.stdout.splitlines()
    if done.returncode != 0 or not lines:
        return None
    return {"repo": repo, "revision": lines[0].strip(), "services_dirty": len(lines) > 1}


def write_record(path, record):
    if not path:
        return
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


class Supervisor:
    """One ssh process for all forwards; restarted with backoff when it exits; health checked every interval.

    The standalone CLI drives it with run(); RemoteManager drives open() / step() / check() / close() from its own thread.
    """

    def __init__(self, host, ports, health_urls, *, record=None, ssh="ssh", interval_s=30.0, repo=None, log=sys.stderr,
                 connect_timeout_s=None, health_timeout_s=5.0, health_fn=None, port_probe=None, popen=None):
        self.host, self.forwards, self.health_urls = host, forward_pairs(ports), dict(health_urls)
        self.ports = [local for local, _ in self.forwards]
        self.record_path, self.ssh, self.interval_s, self.repo, self.log = record, ssh, float(interval_s), repo, log
        self.connect_timeout_s, self.health_timeout_s = connect_timeout_s, float(health_timeout_s)
        # Resolved at call time so tests can patch the module-level helpers.
        self.health_fn = health_fn or (lambda url: health(url, self.health_timeout_s))
        self.port_probe = port_probe or port_open
        self.popen = popen or subprocess.Popen
        # Seconds the forward may take to appear: the ssh connect plus a few seconds of handshake.
        self.open_wait_s = 10.0 if connect_timeout_s is None else float(connect_timeout_s) + 5.0
        self.process, self.restarts, self.backoff, self.external = None, 0, 1.0, False
        self._stop = threading.Event()
        # name -> extra provenance (origin, gpu, ...) merged into the record entry of that service by check().
        self.service_notes = {}
        self.record = {"schema": "ni_rua.remote_services.v1", "host": host,
                       "forwards": [f"127.0.0.1:{local} -> {host}:127.0.0.1:{remote}" for local, remote in self.forwards],
                       "started_at": utc_now(), "tunnel": "starting", "services": {}}

    @property
    def stopping(self):
        return self._stop.is_set()

    @stopping.setter
    def stopping(self, value):
        self._stop.set() if value else self._stop.clear()

    def say(self, text):
        print(f"[remote {utc_now()}] {text}", file=self.log, flush=True)

    def tunnel_alive(self):
        if self.external:
            return all(self.port_probe(port) for port in self.ports)
        return self.process is not None and self.process.poll() is None

    def open(self):
        if self.stopping:
            raise RuntimeError("tunnel startup cancelled")
        self.external = False
        busy = [p for p in self.ports if self.port_probe(p)]
        if busy:   # never steal a port: something else (maybe a manual tunnel) already serves it
            if len(busy) == len(self.ports) and all(self.health_fn(url).get("available") for url in self.health_urls.values()):
                self.external = True   # an earlier tunnel to these services still forwards them: use it, never close it
                self.say(f"ports {busy} already forwarded and healthy; using the existing tunnel")
                return
            raise RuntimeError(f"local port(s) already in use: {busy}")
        self.external = False
        try:
            self.process = self.popen(ssh_command(self.host, self.forwards, ssh=self.ssh, connect_timeout_s=self.connect_timeout_s),
                                      stdin=subprocess.DEVNULL)
        except OSError as error:
            raise RuntimeError(f"ssh launch failed: {error}") from error
        deadline = time.monotonic() + self.open_wait_s
        while time.monotonic() < deadline and not self.stopping:
            if self.process.poll() is not None:
                raise RuntimeError(f"ssh exited with code {self.process.returncode}")
            if all(self.port_probe(p) for p in self.ports):
                self.say(f"tunnel up pid={self.process.pid} ports={self.ports}")
                return
            self._stop.wait(0.25)
        self._terminate()   # a half-open ssh must not linger when the caller retries
        raise RuntimeError(f"forwarded ports did not open within {self.open_wait_s:.0f} s")

    def check(self):
        services = {name: self.health_fn(url) for name, url in self.health_urls.items()}
        for name, notes in self.service_notes.items():
            if name in services:
                services[name] = {**services[name], **notes}
        tunnel = ("external" if self.external else "up") if self.tunnel_alive() else "down"
        self.record.update(tunnel=tunnel, checked_at=utc_now(), services=services, tunnel_restarts=self.restarts)
        write_record(self.record_path, self.record)
        return services

    def _terminate(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def close(self):
        """Stop the forward this object opened; a tunnel that was already there is left alone."""
        self._stop.set()
        self._terminate()
        self.record.update(tunnel="closed", closed_at=utc_now())
        write_record(self.record_path, self.record)

    def step(self):
        """One monitor iteration: reopen a dead tunnel (with backoff), then check health. Returns seconds to wait."""
        if not self.tunnel_alive():
            code = None if self.process is None else self.process.returncode
            self.say(f"tunnel down (ssh exit {code}); perception reports unavailable; reopening")
            self.check()
            self.restarts += 1
            try:
                self.open()
                self.backoff = 1.0
            except RuntimeError as error:
                delay, self.backoff = self.backoff, min(self.backoff * 2, 60.0)
                self.say(f"reopen failed: {error}; retry in {delay:.0f} s")
                return delay
        self.check()
        return self.interval_s

    def run(self, once=False):
        if self.repo:
            self.record["remote_checkout"] = remote_revision(self.host, self.repo, self.ssh)
        self.open()
        services = self.check()
        for name, item in services.items():
            self.say(f"{name}: " + json.dumps(item, sort_keys=True))
        if once:
            return services
        delay = self.interval_s
        while not self._stop.wait(delay):
            delay = self.step()
        return self.record["services"]


def run_config(args, remote):
    """`--config` mode: connect, ensure the configured services (start when needed), then monitor until a signal."""
    from .manager import RemoteError, RemoteManager
    manager = RemoteManager(remote, record=remote.record, say=lambda text: print(f"[remote {utc_now()}] {text}", file=sys.stderr, flush=True))

    def stop(*_):
        manager.abort()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    code = 0
    try:
        manager.start()
    except RemoteError as error:
        manager.say(f"failed: {error}")
        code = 2
    try:
        status = manager.status()
        print(json.dumps(status, indent=2, sort_keys=True))
        if code == 0 and args.once:
            code = 0 if all(item["available"] for item in status["services"].values()) else 1
        elif code == 0:
            manager.wait()
    finally:
        manager.stop()
    return code


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(prog="python -m ni_rua.apps.remote", description=__doc__.split("\n")[0])
    parser.add_argument("--config", help="NI-RUA JSON config whose `remote` section declares the host and services: they are "
                                         "connected, started when not running, forwarded and monitored (--host overrides remote.host)")
    parser.add_argument("--host", help="ssh host alias, e.g. gpu-server (required without --config)")
    parser.add_argument("--port", type=int, action="append", default=None,
                        help="port forwarded 127.0.0.1:P -> host 127.0.0.1:P (repeatable; default 8331 = VGGT service)")
    parser.add_argument("--health", action="append", default=None, metavar="NAME=URL",
                        help="health endpoint to check (default vggt=http://127.0.0.1:<first port>/health)")
    parser.add_argument("--record", help="provenance JSON written after every check (for the run manifest)")
    parser.add_argument("--remote-repo", help="service checkout on the host whose git revision is recorded")
    parser.add_argument("--interval", type=float, default=None, help="seconds between health checks (default 30)")
    parser.add_argument("--once", action="store_true", help="open, check, print, close (exit 0 only when all healthy)")
    args = parser.parse_args(argv)
    if args.config:
        if args.port or args.health:
            parser.error("--port/--health describe services by hand; with --config they come from the config's remote.services")
        from dataclasses import replace
        from ...core.config import Config
        try:
            remote = Config.load(args.config).remote
        except (ValueError, OSError) as error:
            parser.error(str(error))
        overrides = {key: value for key, value in dict(host=args.host, record=args.record, remote_repo=args.remote_repo,
                                                       health_interval_s=args.interval).items() if value is not None}
        try:
            remote = replace(remote, **overrides)
        except ValueError as error:
            parser.error(str(error))
        if not remote.enabled:
            parser.error(f"remote.enabled is false in {args.config}")
        return run_config(args, remote)
    if not args.host:
        parser.error("--host is required without --config")
    ports = args.port or [8331]
    urls = dict(item.split("=", 1) for item in args.health) if args.health else {
        "vggt": f"http://127.0.0.1:{ports[0]}/health"}
    supervisor = Supervisor(args.host, ports, urls, record=args.record, interval_s=30.0 if args.interval is None else args.interval,
                            repo=args.remote_repo)

    def stop(*_):
        supervisor.stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        services = supervisor.run(once=args.once)
    except RuntimeError as error:
        supervisor.say(f"failed: {error}")
        supervisor.record.update(tunnel="failed", error=str(error))
        write_record(args.record, supervisor.record)
        return 2
    finally:
        supervisor.close()
    return 0 if all(item.get("available") for item in services.values()) else 1
