"""One durable, read-only browser view of an agent backend's native TUI in a PTY.

The TUI is a separate client process attached to the live native session (Codex: ``codex resume --remote``;
OpenCode: ``opencode --server ... --session ...``). Its output is appended to ``terminal.bin`` (replayed to every
browser subscriber). Arbitrary browser input is rejected: callers may only resize the PTY or trigger server-synthesized
read-only viewport scroll sequences, so the TUI never receives free-form keystrokes or prompt submissions.
"""

import asyncio
import errno
import fcntl
import json
import os
from pathlib import Path
import pty
import signal
import struct
import subprocess
import termios
import uuid


def _claim_terminal():
    os.setsid()
    fcntl.ioctl(0, termios.TIOCSCTTY, 0)


def terminal_bootstrap_status(data):
    """Recognize the native startup failure without copying arbitrary TUI text."""
    if b"Error: account/read failed during TUI bootstrap:" not in data:
        return {"bootstrap_failed": False, "error": None}
    routing_failed = b"workspace routing discovery failed" in data
    return {
        "bootstrap_failed": True,
        "error": {
            "code": "workspace_routing_discovery_failed" if routing_failed else "account_read_bootstrap_failed",
            "message": "Native Codex TUI account initialization failed"
                       + (": workspace routing discovery failed." if routing_failed else "."),
        },
    }


# Endpoint schemes ``start`` accepts: the Codex remote TUI attaches to a unix socket or WebSocket app-server; an
# explicit ``command`` (another backend's own attach command line) may also name an HTTP server.
_REMOTE_SCHEMES = ("unix://", "ws://", "wss://")
_COMMAND_SCHEMES = _REMOTE_SCHEMES + ("http://", "https://")
SCROLL_ACTIONS = frozenset({"line_up", "line_down", "page_up", "page_down", "top", "bottom"})
MAX_SCROLL_LINES = 40


def validate_terminal_scroll(lines: int = 0, *, action: str | None = None) -> tuple[int, str | None]:
    """Validate bounded read-only viewport scroll arguments; raise ``ValueError("invalid_terminal_scroll")``."""
    if isinstance(lines, bool) or not isinstance(lines, int) or not -MAX_SCROLL_LINES <= lines <= MAX_SCROLL_LINES:
        raise ValueError("invalid_terminal_scroll")
    if action is not None:
        if not isinstance(action, str) or action not in SCROLL_ACTIONS or lines != 0:
            raise ValueError("invalid_terminal_scroll")
    elif lines == 0:
        raise ValueError("invalid_terminal_scroll")
    return lines, action


class Terminal:
    """``command_prefix``: the Codex CLI argv prefix; ``start`` runs ``command_prefix + ["resume", "--remote", uri,
    "--no-alt-screen", thread_id]`` (default ``("codex",)``). ``command``: the complete argv of another backend's
    attach TUI, run exactly as given instead (default None = the Codex command). ``env``: the exact environment of
    the TUI (None = inherit the gateway's), plus ``TERM``/``COLORTERM``."""

    def __init__(self, run_dir, cwd, command_prefix=("codex",), env=None, command=None):
        self.path = Path(run_dir) / "terminal.bin"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cwd = str(cwd)
        self.command_prefix = list(command_prefix)
        self.command = list(command) if command is not None else None
        self.env = dict(env) if env is not None else None
        self.session_id = "pty-" + uuid.uuid4().hex
        self.thread_id = None
        self.socket_uri = None
        self.cols, self.rows = 100, 32
        self.process = None
        self.master = None
        self.log = None
        self.changed = asyncio.Event()
        self.cursor = self.path.stat().st_size if self.path.exists() else 0
        self._closed = False
        self._exit_task = None
        self._diagnostic_tail = b""
        self._pending_scroll = b""
        self._scroll_writer_fd = None
        self._command = None
        self._exit_code = None
        self.bootstrap_failed = False
        self.error = None
        metadata = self.path.with_suffix(".json")
        if metadata.exists():
            saved = json.loads(metadata.read_text())
            self.session_id = saved.get("session_id", self.session_id)
            self.thread_id = saved.get("thread_id")
            self.socket_uri = saved.get("socket_uri")
            self._command = saved.get("command")
            self._exit_code = saved.get("exit_code")
            saved_cols, saved_rows = saved.get("cols", self.cols), saved.get("rows", self.rows)
            if (not isinstance(saved_cols, bool) and isinstance(saved_cols, int) and 20 <= saved_cols <= 500
                    and not isinstance(saved_rows, bool) and isinstance(saved_rows, int) and 5 <= saved_rows <= 200):
                self.cols, self.rows = saved_cols, saved_rows
        else:
            saved = {}
        if "bootstrap_failed" in saved:
            self.bootstrap_failed = saved["bootstrap_failed"]
            self.error = saved.get("error")
        elif self.path.exists():
            # Old archives predate diagnostic metadata; leave their bytes untouched.
            with self.path.open("rb") as log:
                log.seek(max(0, self.cursor - 65536))
                diagnostic = terminal_bootstrap_status(log.read())
            self.bootstrap_failed, self.error = diagnostic["bootstrap_failed"], diagnostic["error"]

    async def start(self, socket_uri, thread_id):
        """Attach exactly once, without a prompt or a new agent turn."""
        if self.process is not None:
            if (self.socket_uri, self.thread_id) != (socket_uri, thread_id):
                raise ValueError("a terminal cannot change its app-server or thread")
            return self.status()
        if self._closed:
            raise RuntimeError("terminal has been closed")
        schemes = _REMOTE_SCHEMES if self.command is None else _COMMAND_SCHEMES
        if not isinstance(socket_uri, str) or not socket_uri.startswith(schemes) or not thread_id:
            raise ValueError("a remote endpoint and explicit thread id are required")
        self.session_id = "pty-" + uuid.uuid4().hex
        self.socket_uri, self.thread_id = socket_uri, thread_id
        self._exit_code = None
        self._diagnostic_tail = b""
        self._pending_scroll = b""
        self._remove_scroll_writer()
        self.bootstrap_failed, self.error = False, None
        master, slave = pty.openpty()
        self.master = master
        env = {**(os.environ if self.env is None else self.env), "TERM": "xterm-256color", "COLORTERM": "truecolor"}
        command = list(self.command) if self.command is not None else self.command_prefix + [
            "resume", "--remote", socket_uri, "--no-alt-screen", thread_id,
        ]
        self._command = command
        reader_added = False
        try:
            self.resize(self.cols, self.rows)
            self.log = self.path.open("ab")
            self.process = subprocess.Popen(
                command, stdin=slave, stdout=slave, stderr=slave,
                cwd=self.cwd, env=env, preexec_fn=_claim_terminal,
            )
            os.close(slave)
            slave = None
            os.set_blocking(master, False)
            asyncio.get_running_loop().add_reader(master, self._receive)
            reader_added = True
            self._persist_status()
            self._exit_task = asyncio.create_task(self._wait_for_exit())
        except BaseException:
            if reader_added:
                try:
                    asyncio.get_running_loop().remove_reader(master)
                except Exception:
                    pass
            if self.process is not None and self.process.poll() is None:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    self.process.wait(timeout=2)
                except Exception:
                    pass
                self.process = None
            try:
                os.close(master)
            except OSError:
                pass
            self.master = None
            if self.log is not None:
                try:
                    self.log.close()
                except OSError:
                    pass
                self.log = None
            raise
        finally:
            if slave is not None:
                try:
                    os.close(slave)
                except OSError:
                    pass
        return self.status()

    def _persist_status(self):
        metadata = self.path.with_suffix(".json")
        temporary = metadata.with_name(f".{metadata.name}.{self.session_id}.tmp")
        temporary.write_text(json.dumps({**self.status(), "socket_uri": self.socket_uri,
                                         "command": self._command}, indent=2))
        temporary.replace(metadata)

    async def _wait_for_exit(self):
        self._exit_code = await asyncio.to_thread(self.process.wait)
        # Drain final diagnostics before publishing the terminal's exit status.
        while self.master is not None:
            before = self.cursor
            self._receive()
            if before == self.cursor:
                break
        self._persist_status()
        self.changed.set()

    def _receive(self):
        if self.master is None:
            return
        try:
            # Startup can fail before setting descriptor flags; the final drain
            # must remain nonblocking even for that partially initialized PTY.
            os.set_blocking(self.master, False)
            data = os.read(self.master, 65536)
        except BlockingIOError:
            return
        except OSError as exc:
            if exc.errno != errno.EIO:
                raise
            data = b""
        if data:
            self.log.write(data)
            self.log.flush()
            self.cursor += len(data)
            diagnostic = terminal_bootstrap_status(self._diagnostic_tail + data)
            self._diagnostic_tail = (self._diagnostic_tail + data)[-4096:]
            if diagnostic["bootstrap_failed"] and diagnostic["error"] != self.error:
                self.bootstrap_failed, self.error = True, diagnostic["error"]
                self._persist_status()
        else:
            self._pending_scroll = b""
            self._remove_scroll_writer()
            asyncio.get_running_loop().remove_reader(self.master)
            os.close(self.master)
            self.master = None
            self._persist_status()
        self.changed.set()

    def read(self, after=0, limit=65536):
        """Return immutable bytes and their exclusive end offset."""
        if not isinstance(after, int) or not 0 <= after <= self.cursor:
            raise ValueError("terminal cursor is outside the persisted stream")
        if not isinstance(limit, int) or not 1 <= limit <= 1048576:
            raise ValueError("terminal read limit must be between 1 and 1048576")
        if not self.path.exists():
            return b"", after
        with self.path.open("rb") as log:
            log.seek(after)
            data = log.read(limit)
        return data, after + len(data)

    async def stream(self, after=0):
        """Independent subscribers replay the log, then follow the same PTY."""
        while True:
            self.changed.clear()
            data, cursor = self.read(after)
            if data:
                after = cursor
                yield data, cursor
                continue
            if self._closed or (self.process is not None and self.master is None):
                if self._exit_task is not None:
                    # EOF can arrive before waitpid observes process exit. A
                    # disconnected subscriber must not cancel the shared waiter.
                    await asyncio.shield(self._exit_task)
                    if after < self.cursor:
                        continue
                return
            await self.changed.wait()

    def resize(self, cols, rows):
        if (isinstance(cols, bool) or isinstance(rows, bool)
                or not isinstance(cols, int) or not isinstance(rows, int)
                or not 20 <= cols <= 500 or not 5 <= rows <= 200):
            raise ValueError("terminal size must be 20..500 columns and 5..200 rows")
        self.cols, self.rows = cols, rows
        if self.master is not None:
            fcntl.ioctl(self.master, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))

    def _remove_scroll_writer(self):
        if self._scroll_writer_fd is not None:
            try:
                asyncio.get_running_loop().remove_writer(self._scroll_writer_fd)
            except RuntimeError:
                pass
            self._scroll_writer_fd = None

    def _drain_pending_scroll(self) -> bool:
        if not self._pending_scroll:
            self._remove_scroll_writer()
            return True
        if self.master is None:
            self._pending_scroll = b""
            self._remove_scroll_writer()
            return False
        try:
            os.set_blocking(self.master, False)
            while self._pending_scroll:
                written = os.write(self.master, self._pending_scroll)
                if written <= 0:
                    break
                self._pending_scroll = self._pending_scroll[written:]
        except BlockingIOError:
            pass
        except OSError:
            self._pending_scroll = b""
            self._remove_scroll_writer()
            return False
        if not self._pending_scroll:
            self._remove_scroll_writer()
            return True
        if self._scroll_writer_fd != self.master:
            self._remove_scroll_writer()
            try:
                asyncio.get_running_loop().add_writer(self.master, self._drain_pending_scroll)
                self._scroll_writer_fd = self.master
            except RuntimeError:
                pass
        return False

    def scroll(self, lines: int = 0, *, action: str | None = None) -> bool:
        """Write server-synthesized read-only viewport scroll sequences to a live alternate-screen TUI PTY."""
        lines, action = validate_terminal_scroll(lines, action=action)
        if (self._closed or self.master is None or self._exit_code is not None
                or (self.process is not None and self.process.poll() is not None)):
            return False
        if not self._drain_pending_scroll():
            return False
        col = max(1, min(self.cols, max(10, self.cols // 2)))
        row = max(1, min(self.rows, max(3, self.rows // 3)))
        if action == "line_up":
            unit, count = f"\x1b[<64;{col};{row}M".encode("ascii"), 1
        elif action == "line_down":
            unit, count = f"\x1b[<65;{col};{row}M".encode("ascii"), 1
        elif action == "page_up":
            unit, count = b"\x1b[5~", 1
        elif action == "page_down":
            unit, count = b"\x1b[6~", 1
        elif action == "top":
            unit, count = b"\x1b[H", 1
        elif action == "bottom":
            unit, count = b"\x1b[F", 1
        elif lines < 0:
            unit, count = f"\x1b[<64;{col};{row}M".encode("ascii"), -lines
        else:
            unit, count = f"\x1b[<65;{col};{row}M".encode("ascii"), lines
        wrote_any = False
        try:
            os.set_blocking(self.master, False)
            for _ in range(count):
                written = os.write(self.master, unit)
                if written <= 0:
                    break
                wrote_any = True
                if written < len(unit):
                    self._pending_scroll = unit[written:]
                    self._drain_pending_scroll()
                    break
        except BlockingIOError:
            pass
        except OSError:
            return wrote_any
        return wrote_any

    def status(self):
        return {
            "session_id": self.session_id, "thread_id": self.thread_id,
            "cursor": self.cursor, "pid": self.process.pid if self.process else None,
            "exit_code": self.process.poll() if self.process else self._exit_code,
            "cols": self.cols, "rows": self.rows, "mode": "read_only",
            "bootstrap_failed": self.bootstrap_failed, "error": self.error,
        }

    async def close(self):
        """Only the run supervisor closes the PTY; browser disconnects never do."""
        self._closed = True
        try:
            if self.process is not None and self.process.poll() is None:
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.to_thread(self.process.wait, timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    await asyncio.to_thread(self.process.wait)
            if self._exit_task is not None:
                await self._exit_task
            if self.master is not None:
                self._receive()
        finally:
            if self.master is not None:
                self._pending_scroll = b""
                self._remove_scroll_writer()
                try:
                    asyncio.get_running_loop().remove_reader(self.master)
                except Exception:
                    pass
                try:
                    os.close(self.master)
                except OSError:
                    pass
                self.master = None
            if self.log is not None:
                try:
                    self.log.close()
                except OSError:
                    pass
                self.log = None
            if self.process is not None:
                try:
                    self._persist_status()
                except OSError:
                    pass
            self.changed.set()
