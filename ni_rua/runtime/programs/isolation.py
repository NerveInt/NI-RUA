"""Bounded program launcher with optional Linux Landlock/seccomp isolation.

Enabled isolation fails closed without requiring user namespaces.
"""

import ctypes
import errno
import os
import sys
from pathlib import Path

# Linux UAPI syscall numbers on x86_64/aarch64; other platforms fail closed.
CREATE_RULESET, ADD_RULE, RESTRICT_SELF = 444, 445, 446
EXECUTE, WRITE_FILE, READ_FILE, READ_DIR = 1, 2, 4, 8


def restrict_program_syscalls(*, deny_socket_creation=False, bounded_program=False, deny_exec=False):
    """Keep an actor on its inherited channel; filters also constrain descendants."""
    if not (deny_socket_creation or bounded_program or deny_exec):
        return
    try:
        seccomp = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    except OSError as error:
        raise RuntimeError("libseccomp.so.2 is required for bounded actor programs") from error
    allow, deny = 0x7fff0000, 0x00050000 | errno.EPERM
    class Argument(ctypes.Structure):
        _fields_ = [("arg", ctypes.c_uint32), ("op", ctypes.c_uint32),
                    ("datum_a", ctypes.c_uint64), ("datum_b", ctypes.c_uint64)]
    seccomp.seccomp_init.argtypes = [ctypes.c_uint32]
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    seccomp.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint, ctypes.POINTER(Argument)]
    seccomp.seccomp_load.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_release.argtypes = [ctypes.c_void_p]
    context = seccomp.seccomp_init(allow)
    if not context:
        raise RuntimeError("seccomp filter initialization failed")
    def rule(name, action=deny, argument=None):
        syscall = seccomp.seccomp_syscall_resolve_name(name.encode())
        if syscall < 0:
            return  # This syscall is not present on the active native architecture.
        result = seccomp.seccomp_rule_add_array(context, action, syscall, int(argument is not None),
                                               ctypes.byref(argument) if argument is not None else None)
        if result < 0:
            raise OSError(-result, "seccomp rule failed: " + name)
    try:
        if deny_socket_creation:
            for name in ("socket", "socketpair", "connect", "socketcall", "io_uring_setup", "io_uring_enter", "io_uring_register"):
                rule(name)
        if bounded_program:
            for name in ("fork", "vfork", "kill", "tkill", "tgkill", "rt_sigqueueinfo", "rt_tgsigqueueinfo",
                         "pidfd_send_signal", "ptrace", "process_vm_readv", "process_vm_writev", "pidfd_getfd"):
                rule(name)
            # glibc falls back to clone only on ENOSYS; the clone rule permits pthreads.
            rule("clone3", 0x00050000 | errno.ENOSYS)
            rule("clone", argument=Argument(0, 7, 0x10000, 0))  # SCMP_CMP_MASKED_EQ, !CLONE_THREAD.
        if deny_exec:
            for name in ("execve", "execveat", "open_by_handle_at"):
                rule(name)
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(38, 1, 0, 0, 0) < 0:  # PR_SET_NO_NEW_PRIVS (required for unprivileged seccomp_load)
            raise OSError(ctypes.get_errno(), "prctl PR_SET_NO_NEW_PRIVS failed")
        libc.prctl(4, 0, 0, 0, 0)  # PR_SET_DUMPABLE = 0
        result = seccomp.seccomp_load(context)
        if result < 0:
            raise OSError(-result, "seccomp filter load failed")
    finally:
        seccomp.seccomp_release(context)


def restrict(read_paths, write_paths, socket_paths=(), tcp_ports=None):
    libc = ctypes.CDLL(None, use_errno=True)
    abi = libc.syscall(CREATE_RULESET, 0, 0, 1)
    if abi < 1:
        raise OSError(ctypes.get_errno(), "Landlock is required for actor isolation")
    handled = (1 << 13) - 1
    # Landlock rejects a ruleset containing rights newer than the kernel ABI.
    if abi >= 2:
        handled |= 1 << 13  # REFER
    if abi >= 3:
        handled |= 1 << 14  # TRUNCATE
    if abi >= 5:
        handled |= 1 << 15  # IOCTL_DEV

    class Ruleset(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64), ("handled_access_net", ctypes.c_uint64)]

    class PathRule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]

    class PortRule(ctypes.Structure):
        _fields_ = [("allowed_access", ctypes.c_uint64), ("port", ctypes.c_uint64)]

    def check(result):
        if result < 0:
            raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))
        return result

    if tcp_ports is not None and abi < 4:
        raise RuntimeError("Landlock ABI >=4 required for actor TCP isolation")
    attrs = Ruleset(handled, 3 if tcp_ports is not None else 0)
    size = ctypes.sizeof(attrs) if abi >= 4 else 8
    ruleset = check(libc.syscall(CREATE_RULESET, ctypes.byref(attrs), size, 0))
    try:
        for paths, access in [(read_paths, EXECUTE | READ_FILE | READ_DIR), (write_paths, handled), (socket_paths, 1 << 9)]:
            for item in paths:
                path = Path(item).resolve()
                if not path.exists():
                    raise FileNotFoundError(path)
                fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
                try:
                    allowed = access if path.is_dir() else access & (EXECUTE | READ_FILE | WRITE_FILE | (1 << 14) if abi >= 3 else EXECUTE | READ_FILE | WRITE_FILE)
                    rule = PathRule(allowed, fd)
                    check(libc.syscall(ADD_RULE, ruleset, 1, ctypes.byref(rule), 0))
                finally:
                    os.close(fd)
        for port in tcp_ports or []:
            rule = PortRule(2, port)  # CONNECT_TCP only; BIND_TCP remains denied.
            check(libc.syscall(ADD_RULE, ruleset, 2, ctypes.byref(rule), 0))
        check(libc.prctl(38, 1, 0, 0, 0))  # PR_SET_NO_NEW_PRIVS
        check(libc.syscall(RESTRICT_SELF, ruleset, 0))
    finally:
        os.close(ruleset)
    return abi


def _run_confined_entry(main_py: str) -> None:
    import json
    import runpy

    raw = os.environ.pop("_NI_RUA_CONFINED_FS", "")
    if not raw:
        raise RuntimeError("missing_confined_fs_spec")
    fs_spec = json.loads(raw)
    write_roots = tuple(Path(p).resolve() for p in fs_spec.get("write_paths", ()))
    read_roots = tuple(
        dict.fromkeys(
            [
                *write_roots,
                *(Path(p).resolve() for p in fs_spec.get("read_paths", ())),
                *(
                    Path(p).resolve()
                    for p in ("/dev/null", "/dev/urandom", "/etc/ld.so.cache", "/lib", "/lib64", "/usr/lib")
                    if os.path.exists(p)
                ),
            ]
        )
    )
    write_flag_mask = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND

    def _check_path(target, *, write: bool) -> None:
        if isinstance(target, int):
            return
        try:
            text = os.fsdecode(target)
        except Exception as err:
            raise PermissionError("program_path_access_denied") from err
        resolved = (Path(os.getcwd()) / text).resolve()
        roots = write_roots if write else read_roots
        if not any(resolved == root or resolved.is_relative_to(root) for root in roots):
            raise PermissionError(f"program_path_access_denied:{text}")

    def _audit_hook(event: str, args: tuple) -> None:
        if event.startswith("ctypes.") or event in (
            "os.system",
            "os.popen",
            "os.posix_spawn",
            "os.posix_spawnp",
            "subprocess.Popen",
            "os.exec",
        ):
            raise PermissionError(f"program_operation_denied:{event}")
        if event == "open":
            path_arg = args[0] if len(args) > 0 else None
            mode = args[1] if len(args) > 1 else None
            flags = args[2] if len(args) > 2 else 0
            is_write = (
                any(ch in mode for ch in ("w", "a", "x", "+"))
                if isinstance(mode, str)
                else bool(isinstance(flags, int) and (flags & write_flag_mask))
            )
            _check_path(path_arg, write=is_write)
        elif event in ("os.listdir", "os.scandir"):
            path_arg = args[0] if args and args[0] is not None else os.getcwd()
            _check_path(path_arg, write=False)
        elif event in ("os.mkdir", "os.remove", "os.unlink", "os.rmdir", "os.chmod", "os.truncate"):
            if args:
                _check_path(args[0], write=True)
        elif event in ("os.rename", "os.replace", "os.symlink", "os.link"):
            if len(args) >= 2:
                _check_path(args[0], write=True)
                _check_path(args[1], write=True)

    main_dir = str(Path(main_py).resolve().parent)
    sys.path[:] = [main_dir, *(p for p in sys.path[1:] if p and p != main_dir)]
    restrict_program_syscalls(
        deny_socket_creation=bool(fs_spec.get("deny_socket_creation", False)),
        bounded_program=bool(fs_spec.get("bounded_program", False)),
        deny_exec=True,
    )
    sys.addaudithook(_audit_hook)
    sys.argv = [main_py]
    runpy.run_path(main_py, run_name="__main__")


def launch():
    if len(sys.argv) >= 3 and sys.argv[1] == "--confined-entry":
        _run_confined_entry(sys.argv[2])
        return
    import json
    import platform
    spec = json.loads(Path(sys.argv[1]).read_text())
    if spec.get("limits"):
        import resource
        for name, value in spec["limits"].items():
            limit = getattr(resource, name, None)
            if limit is not None:
                _, hard = resource.getrlimit(limit)
                cap = int(value) if hard == resource.RLIM_INFINITY else min(int(value), hard)
                resource.setrlimit(limit, (cap, cap))
    if spec.get("sandbox_enabled", True):
        if platform.machine() not in ("x86_64", "aarch64"):
            raise RuntimeError("unsupported_landlock_architecture")
        restrict(spec["read_paths"], spec["write_paths"], spec.get("socket_paths", []), spec.get("tcp_ports"))
        restrict_program_syscalls(deny_socket_creation=spec.get("deny_socket_creation", False), bounded_program=spec.get("bounded_program", False))
        if spec.get("bounded_program"):
            os.chdir(spec["env"]["HOME"])
        os.execvpe(sys.argv[2], sys.argv[2:], spec["env"])
    elif spec.get("seccomp_enabled", False):
        if platform.machine() not in ("x86_64", "aarch64"):
            raise RuntimeError("unsupported_seccomp_architecture")
        if spec.get("bounded_program"):
            os.chdir(spec["env"]["HOME"])
        child_env = dict(spec["env"])
        child_env["_NI_RUA_CONFINED_FS"] = json.dumps(
            {
                "read_paths": spec.get("read_paths", []),
                "write_paths": spec.get("write_paths", []),
                "deny_socket_creation": spec.get("deny_socket_creation", False),
                "bounded_program": spec.get("bounded_program", False),
            }
        )
        os.execvpe(
            sys.argv[2],
            [sys.argv[2], "-S", str(Path(__file__).resolve()), "--confined-entry", sys.argv[-1]],
            child_env,
        )
    else:
        if spec.get("bounded_program"):
            os.chdir(spec["env"]["HOME"])
        os.execvpe(sys.argv[2], sys.argv[2:], spec["env"])


if __name__ == "__main__":
    launch()
