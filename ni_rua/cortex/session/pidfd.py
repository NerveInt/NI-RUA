"""Linux pidfds without requiring recent Python or glibc wrapper symbols."""

import ctypes
import errno
import os
import platform
import signal
import sys


def _syscall(number, *arguments):
    # Linux x86_64 and asm-generic (aarch64) UAPI use these syscall numbers.
    # Reject other ABIs instead of risking a different syscall or PID fallback.
    if (sys.platform != "linux" or platform.machine() not in {"x86_64", "aarch64"}
            or ctypes.sizeof(ctypes.c_void_p) != 8):
        raise OSError(errno.ENOSYS, "pidfd syscall fallback requires Linux x86_64 or aarch64")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(number), *arguments)
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def pidfd_open(pid, flags=0):
    native = getattr(os, "pidfd_open", None)
    if native is not None:
        return native(pid, flags)
    return _syscall(434, ctypes.c_int(pid), ctypes.c_uint(flags))


def pidfd_send_signal(descriptor, sig, flags=0):
    native = getattr(signal, "pidfd_send_signal", None)
    if native is not None:
        return native(descriptor, sig, None, flags)
    _syscall(424, ctypes.c_int(descriptor), ctypes.c_int(sig), ctypes.c_void_p(), ctypes.c_uint(flags))


class ProcessHandle:
    """Exit-observable handle bound to one live process before it is signalled.

    Linux uses a pidfd, so signals and exit waits can never reach a reused PID.
    macOS has no pidfd: the handle registers a kqueue ``EVFILT_PROC/NOTE_EXIT``
    watch, so exit waits stay bound to the original process, but signals still go
    through ``os.kill(pid)``. ``send_signal`` refuses once that watch has seen an
    exit, which leaves only the microsecond window between that check and
    ``kill``. Other platforms fail closed with ``ENOSYS``.
    """

    def __init__(self, pid):
        self.pid = pid
        self._fd = self._kqueue = None
        if sys.platform == "darwin":
            import select
            self._kqueue = select.kqueue()
            try:
                self._kqueue.control([select.kevent(pid, filter=select.KQ_FILTER_PROC,
                    flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT, fflags=select.KQ_NOTE_EXIT)], 0, 0)
            except BaseException:
                self._kqueue.close()
                raise
            self._exited = False
        else:
            self._fd = pidfd_open(pid)

    def wait(self, timeout_s):
        """Return True once the bound process has exited; ``timeout_s`` is seconds."""
        import select
        if self._fd is not None:
            return bool(select.select([self._fd], [], [], timeout_s)[0])
        if not self._exited:
            self._exited = bool(self._kqueue.control(None, 1, timeout_s))
        return self._exited

    def send_signal(self, sig):
        if self._fd is not None:
            return pidfd_send_signal(self._fd, sig)
        if self.wait(0):
            raise ProcessLookupError(errno.ESRCH, "bound process already exited")
        os.kill(self.pid, sig)

    def close(self):
        if self._fd is not None:
            os.close(self._fd)
        if self._kqueue is not None:
            self._kqueue.close()


def open_process(pid):
    """Bind a handle to ``pid``; raises ProcessLookupError when it is not running."""
    return ProcessHandle(pid)


def process_argv(pid):
    """Exact argv of a running process (Linux ``/proc``, macOS ``KERN_PROCARGS2``).

    Raises FileNotFoundError/ProcessLookupError when the process is gone and
    PermissionError when it belongs to another user; never guesses from ``ps``.
    """
    if sys.platform != "darwin":
        from pathlib import Path
        return Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").split("\0")
    libc = ctypes.CDLL(None, use_errno=True)
    mib = (ctypes.c_int * 3)(1, 49, pid)  # CTL_KERN, KERN_PROCARGS2
    size = ctypes.c_size_t(0)
    if libc.sysctl(mib, 3, None, ctypes.byref(size), None, 0) != 0:
        raise ProcessLookupError(ctypes.get_errno(), "process arguments unavailable")
    buffer = ctypes.create_string_buffer(size.value)
    if libc.sysctl(mib, 3, buffer, ctypes.byref(size), None, 0) != 0:
        error = ctypes.get_errno()
        raise (PermissionError if error == errno.EPERM else ProcessLookupError)(error, os.strerror(error))
    raw = buffer.raw[:size.value]
    if len(raw) < 4:
        raise ProcessLookupError(errno.ESRCH, "process arguments unavailable")
    argc = int.from_bytes(raw[:4], sys.byteorder)
    rest = raw[4:]
    if b"\0" not in rest:
        raise ProcessLookupError(errno.ESRCH, "process arguments unavailable")
    rest = rest[rest.index(b"\0"):].lstrip(b"\0")  # skip exec path and its padding
    return [item.decode(errors="replace") for item in rest.split(b"\0")[:argc]]
