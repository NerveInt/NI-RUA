"""零外部依赖的多进程并行 unittest 执行器。

特性：
1. 完整复用标准库 unittest 发现机制，自动保持 setUpModule/tearDownModule 与 setUpClass/tearDownClass 边界；
2. 常驻 Worker 子进程池通过独立 AF_UNIX socketpair 通信，彻底隔离测试标准输出/错误流，杜绝管道死锁与协议污染；
3. 为每个 Worker 注入受控线程环境变量（DEFAULT_SUBPROCESS_THREAD_ENV）及短路径独立 TMPDIR/XDG_RUNTIME_DIR，
   防止 256 核宿主机线程爆炸与 Linux 108 字节 AF_UNIX 路径超限；
4. 按模块历史耗时权重（LPT）动态派发分片，自动捕获子进程异常崩溃并补位继续执行。
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, deque
from collections.abc import Iterable, Sequence
import contextlib
import ctypes
from dataclasses import dataclass
import faulthandler
import fnmatch
import json
import math
import os
from pathlib import Path
import re
import selectors
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from ni_rua.core.config import DEFAULT_SUBPROCESS_THREAD_ENV  # noqa: E402
except ImportError:
    DEFAULT_SUBPROCESS_THREAD_ENV = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "OPENCV_FOR_THREADS_NUM": "1",
        "TORCH_NUM_THREADS": "2",
        "NUMBA_NUM_THREADS": "2",
        "PXR_WORK_THREAD_LIMIT": "4",
        "RAYON_NUM_THREADS": "2",
        "TOKIO_WORKER_THREADS": "2",
    }

DEFAULT_MAX_JOBS = 32

WORKER_THREAD_ENV = {
    **DEFAULT_SUBPROCESS_THREAD_ENV,
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "OPENCV_FOR_THREADS_NUM": "1",
    "TORCH_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "2",
    "TOKIO_WORKER_THREADS": "2",
    "PYTHONDONTWRITEBYTECODE": "1",
}

# 调度优先级与分片粒度提示（仅影响队列派发顺序与分片大小，绝不影响用例发现集合与断言）
_MODULE_PROFILE: dict[str, tuple[float, int]] = {
    "robot.so101.test_so101_benches": (35.0, 1),
    "apps.evaluation.test_benchmark_group": (25.0, 1),
    "runtime.tools.test_robotic_view_isolation": (24.0, 1),
    "cortex.claude_code.test_claude_code_conformance": (20.0, 1),
    "cortex.external.test_external_workbench": (19.5, 1),
    "tests.packaging.sandbox.test_codex_sandbox": (19.0, 1),
    "tests.packaging.sandbox.test_agent_sandbox": (18.5, 1),
    "tests.packaging.sandbox.test_agent_claude_sandbox": (18.0, 1),
    "tests.packaging.sandbox.test_agent_opencode_sandbox": (17.5, 1),
    "core.runner.test_parallel_runner": (16.5, 1),
    "apps.gateway.test_gateway_handoff": (16.0, 1),
    "runtime.harness.test_handoff_and_candidates": (15.5, 1),
    "cortex.opencode.test_opencode_conformance": (15.0, 1),
    "apps.gateway.test_gateway_agent_settings": (14.5, 1),
    "robot.embodiment.test_embodiment_perception_evidence": (14.0, 1),
    "apps.gateway.test_gateway_core": (13.5, 1),
    "robot.embodiment.test_embodiment_x5_golden": (13.0, 1),
    "perception.external.test_obs3d": (12.8, 1),
    "robot.embodiment.test_embodiment_schema": (12.5, 1),
    "core.config.test_config": (12.2, 1),
    "cortex.agents.test_self_briefing_and_grounding": (12.0, 1),
    "robot.calibration.test_so101_calibration_runner": (11.8, 1),
    "runtime.tools.test_stereo_guide": (11.5, 1),
    "robot.so101.test_so101_held_perception": (11.2, 1),
    "perception.external.test_vggt_height": (11.0, 1),
    "robot.embodiment.test_embodiment_prompt_cleanup": (10.8, 1),
    "perception.external.test_object_table": (10.5, 1),
    "core.storage.test_image_storage": (10.2, 1),
    "apps.evaluation.test_benchmark_queue": (10.0, 1),
    "runtime.harness.test_prefix_cache_and_on_anomaly": (9.8, 1),
    "robot.so101.test_so101_grip_squeeze": (9.6, 1),
    "robot.so101.test_so101_feedback_grip_guard": (9.4, 1),
    "cortex.agents.test_agent_workbench": (9.2, 1),
    "robot.so101.test_so101_perception_view": (9.0, 1),
    "robot.so101.test_so101_backend": (8.8, 1),
    "apps.evaluation.test_evaluation_evidence": (8.6, 1),
    "apps.gateway.test_gateway_console": (8.4, 1),
    "cortex.claude_code.test_claude_options_and_settings": (8.2, 1),
    "runtime.tools.test_tool_registry": (8.0, 1),
    "apps.gateway.test_gateway_calibration": (7.8, 1),
    "robot.embodiment.test_embodiment_profile": (7.6, 1),
    "runtime.tools.test_robotic_view": (7.4, 1),
    "runtime.programs.test_program_isolation": (11.5, 2),
    "runtime.programs.test_program_review": (11.0, 2),
    "runtime.programs.test_program_execution": (8.0, 2),
    "cortex.codex.test_codex_evidence_and_metadata": (10.5, 1),
    "cortex.codex.test_codex_config_and_provider": (10.0, 1),
    "cortex.opencode.test_opencode_session_unit": (7.2, 2),
    "robot.so101.test_so101_grasp_and_touch": (7.0, 2),
    "runtime.harness.test_task_planning_and_notes": (6.8, 2),
    "cortex.claude_code.test_claude_session_unit": (6.6, 2),
    "cortex.agents.test_agent_prompt_parity": (6.4, 2),
    "runtime.harness.test_delivery_and_terminal_gates": (6.2, 2),
    "robot.so101.test_so101_camera_and_recalibration": (6.0, 2),
    "apps.evaluation.test_queue_config_and_hooks": (5.8, 2),
    "runtime.actions.test_unified_actions": (5.6, 2),
    "runtime.harness.test_operation_memory_and_verification": (5.4, 2),
    "cortex.codex.test_codex_requests": (5.2, 2),
    "robot.so101.test_so101_motion_features": (5.0, 2),
    "robot.so101.test_so101_motion_compensation": (4.8, 2),
    "robot.so101.test_so101_path_check": (4.6, 2),
    "runtime.harness.test_harness_geometry_and_contact": (4.5, 2),
    "apps.library.test_library_and_read_models": (4.4, 2),
    "apps.library.test_experiments_and_reports": (4.3, 2),
    "apps.evaluation.test_benchmark_and_matrix": (4.2, 2),
    "cortex.relay.test_model_proxy_continuity_and_guard": (4.1, 2),
    "robot.robodojo.test_robodojo_worker": (4.0, 2),
    "robot.so101.test_so101_instructions_and_info": (3.9, 2),
    "cortex.agents.test_agent_isolation_and_replay": (3.8, 2),
    "apps.evaluation.test_stall_and_motion_review": (3.7, 3),
    "cortex.agents.test_agent_profiles": (3.6, 3),
    "cortex.agents.test_agent_contracts_fast": (3.5, 4),
    "cortex.external.test_external_config_and_session": (3.4, 4),
}

VALID_ZONES: tuple[str, ...] = ("core", "robot", "perception", "cortex", "runtime", "apps", "packaging")
VALID_TIERS: tuple[str, ...] = ("fast", "component", "packaging")
TIMINGS_FILENAME = ".ni-rua-test-timings.json"

_LAYER_PRIORITY: dict[str, int] = {
    "packaging": 6,
    "apps": 5,
    "runtime": 4,
    "cortex": 3,
    "perception": 2,
    "robot": 1,
    "core": 0,
}

# 组合根 Hub 模块：未直接改动时不向下传递闭包扩展，仅包含其直接/1-hop 测试边界
COMPOSITION_ROOT_HUBS: frozenset[str] = frozenset({
    "ni_rua.runtime.harness",
    "ni_rua.apps.gateway.workbench",
    "ni_rua.robot.backends.factory",
    "ni_rua.cortex.agents.registry",
    "ni_rua.core.contracts",
    "ni_rua.core.config",
    "ni_rua.core.config.hooks",
    "ni_rua.core.config.loader",
    "ni_rua.core.config.sections",
    "ni_rua.core.config.thresholds",
})

_FLAT_ZONE_OVERRIDES: dict[str, tuple[str, str]] = {
    "test_agent_claude_sandbox": ("packaging", "sandbox"),
    "test_agent_opencode_sandbox": ("packaging", "sandbox"),
    "test_agent_sandbox": ("packaging", "sandbox"),
    "test_codex_sandbox": ("packaging", "sandbox"),
    "test_codex_sandbox_conformance": ("packaging", "sandbox"),
    "test_agent_plugin_coexistence": ("packaging", "npm"),
    "test_agent_plugin_package": ("packaging", "npm"),
    "test_check_package": ("packaging", "wheel"),
    "test_build_package": ("packaging", "wheel"),
    "test_parallel_runner": ("core", "runner"),
    "test_run_tests_per_module": ("core", "runner"),
    "test_layer_boundaries": ("core", "boundaries"),
    "test_host_capabilities": ("core", "general"),
    "test_config": ("core", "config"),
    "test_store_close_race": ("core", "storage"),
    "test_image_archive": ("core", "storage"),
    "test_images": ("core", "storage"),
    "test_lean_compact": ("core", "presentation"),
    "test_lean_projection": ("core", "presentation"),
    "test_lean_contracts": ("core", "presentation"),
    "test_lean_evidence": ("core", "presentation"),
    "test_conformance_deny_proxy": ("cortex", "agents"),
    "test_benchmark_group": ("apps", "evaluation"),
    "test_experiment_hook": ("apps", "evaluation"),
}

_STEM_ZONE_PREFIXES: tuple[tuple[str, tuple[str, str]], ...] = (
    ("test_so101_calibration", ("robot", "calibration")),
    ("test_so101_", ("robot", "so101")),
    ("test_robodojo", ("robot", "robodojo")),
    ("test_robot_matrix", ("apps", "evaluation")),
    ("test_embodiment_", ("robot", "embodiment")),
    ("test_joint_replan", ("robot", "kinematics")),
    ("test_batch_motion", ("robot", "motion")),
    ("test_sam31_", ("perception", "external")),
    ("test_codex_", ("cortex", "codex")),
    ("test_claude_", ("cortex", "claude_code")),
    ("test_agent_claude", ("cortex", "claude_code")),
    ("test_opencode_", ("cortex", "opencode")),
    ("test_agent_opencode", ("cortex", "opencode")),
    ("test_model_proxy", ("cortex", "relay")),
    ("test_model_catalogue", ("cortex", "relay")),
    ("test_relay_", ("cortex", "relay")),
    ("test_gateway_", ("apps", "gateway")),
    ("test_console_", ("apps", "gateway")),
    ("test_benchmark_", ("apps", "evaluation")),
    ("test_queue_", ("apps", "evaluation")),
    ("test_action_", ("runtime", "actions")),
    ("test_program_", ("runtime", "programs")),
    ("test_tool_", ("runtime", "tools")),
)

_PACKAGING_TEST_STEMS: frozenset[str] = frozenset({
    "test_agent_claude_sandbox",
    "test_agent_opencode_sandbox",
    "test_agent_plugin_coexistence",
    "test_agent_plugin_package",
    "test_agent_sandbox",
    "test_codex_sandbox",
    "test_codex_sandbox_conformance",
    "test_check_package",
    "test_build_package",
})

_COMPONENT_SIGNAL_RE = re.compile(
    r"\b(?:HTTPServer|ThreadingHTTPServer|AppRunner|TCPSite|ClientSession|sqlite3\.connect|"
    r"Store|RunStore|Workbench|ProgramRunner|ModelRequestProxy|NiRuaMcpServer|McpToolBridge|"
    r"subprocess\.Popen|subprocess\.run|subprocess\.check_output|subprocess\.check_call|"
    r"multiprocessing\.Process)\s*\("
)


@dataclass(frozen=True)
class TestClassification:
    path: Path
    module_name: str
    zone: str
    subzone: str
    tier: str
    imports: frozenset[Path]


@dataclass(frozen=True)
class SelectionPlan:
    selected_tests: tuple[Path, ...]
    reasons: dict[Path, str]
    classifications: dict[Path, TestClassification]
    changed_files: tuple[str, ...]
    run_js_web: bool
    run_js_plugin: bool
    damper_skipped_count: int
    scan_elapsed_s: float


def _parse_csv_flags(values: Sequence[str] | None, *, valid: Sequence[str], flag_name: str) -> set[str]:
    if not values:
        return set()
    out: set[str] = set()
    valid_set = set(valid)
    for raw in values:
        for item in raw.split(","):
            tok = item.strip()
            if not tok:
                continue
            if tok not in valid_set:
                raise SystemExit(f"非法 {flag_name} 值: {tok!r}（可选: {', '.join(valid)}）")
            out.add(tok)
    return out


def _parse_subzone_flags(values: Sequence[str] | None) -> set[str]:
    if not values:
        return set()
    out: set[str] = set()
    for raw in values:
        for item in raw.split(","):
            tok = item.strip()
            if tok:
                out.add(tok)
    return out


def _resolve_project_root(start_dir: Path) -> Path:
    resolved = start_dir.resolve()
    if resolved.is_relative_to(REPO_ROOT.resolve()):
        return REPO_ROOT.resolve()
    # 本检出之外按就近匹配：先遇到的 `.git` 或含 `ni_rua/__init__.py` 的目录即为项目根。否则独立项目树（如测试夹具）
    # 位于另一个带 `.git` 的目录（TMPDIR 在别的检出内、宿主残留的 `/tmp/.git`）之下时，会越过自身解析到外层，
    # 致使本检出的 `ni_rua` 遮蔽该树的同名包。
    for candidate in (resolved, *resolved.parents):
        if (candidate / ".git").exists() or (candidate / "ni_rua" / "__init__.py").is_file():
            return candidate
    return resolved.parent if resolved.name == "tests" else resolved


def _parse_python_ast(text: str) -> ast.Module:
    """解析完整 Python AST（捕获条件分支、函数内联导入及惰性 __getattr__ 导出表）。"""
    try:
        return ast.parse(text)
    except SyntaxError:
        return ast.Module(body=[], type_ignores=[])


def _extract_tier_marker(tree: ast.Module) -> str | None:
    for node in tree.body:
        target_names: list[str] = []
        val_node: ast.expr | None = None
        if isinstance(node, ast.Assign):
            target_names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            val_node = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_names = [node.target.id]
            val_node = node.value
        if "NI_RUA_TEST_TIER" in target_names and isinstance(val_node, ast.Constant) and isinstance(val_node.value, str):
            marker = val_node.value.strip()
            if marker in VALID_TIERS:
                return marker
    return None


def _resolve_rel_mod_name(raw_mod: str, pkg_parts: Sequence[str]) -> str:
    if not raw_mod.startswith("."):
        return raw_mod
    dots = len(raw_mod) - len(raw_mod.lstrip("."))
    rest = raw_mod[dots:]
    up = dots - 1
    anchor = pkg_parts[: len(pkg_parts) - up] if up <= len(pkg_parts) else []
    return ".".join([*anchor, *(rest.split(".") if rest else ())])


def _attr_chain(node: ast.AST) -> list[str] | None:
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        parts.reverse()
        return parts
    return None


def _extract_str_seq(node: ast.expr | None, known: dict[str, list[str]]) -> list[str] | None:
    if node is None:
        return None
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        if all(isinstance(el, ast.Constant) and isinstance(el.value, str) for el in node.elts):
            return [el.value for el in node.elts if isinstance(el, ast.Constant) and isinstance(el.value, str)]
        return None
    if isinstance(node, ast.Dict):
        if all(isinstance(k, ast.Constant) and isinstance(k.value, str) for k in node.keys if k is not None):
            return [k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]
        return None
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"frozenset", "set", "list", "tuple"}
        and len(node.args) == 1
    ):
        arg = node.args[0]
        if isinstance(arg, ast.Name) and arg.id in known:
            return list(known[arg.id])
        return _extract_str_seq(arg, known)
    return None


def _matched_getattr_names(
    test_node: ast.expr,
    param_name: str,
    known_collections: dict[str, list[str]],
) -> list[str]:
    if (
        isinstance(test_node, ast.Compare)
        and len(test_node.ops) == 1
        and len(test_node.comparators) == 1
        and isinstance(test_node.left, ast.Name)
        and test_node.left.id == param_name
    ):
        op = test_node.ops[0]
        comp = test_node.comparators[0]
        if isinstance(op, ast.Eq) and isinstance(comp, ast.Constant) and isinstance(comp.value, str):
            return [comp.value]
        if isinstance(op, ast.In):
            if isinstance(comp, ast.Name) and comp.id in known_collections:
                return list(known_collections[comp.id])
            seq = _extract_str_seq(comp, known_collections)
            if seq is not None:
                return seq
    return []


def _extract_lazy_barrel_info(tree: ast.Module, pkg_parts: Sequence[str]) -> tuple[dict[str, str], set[str]]:
    """解析模块级惰性 __getattr__ 导出映射（如 _EXPORTS、_SUBMODULES、_CODEX_COMPAT_EXPORTS 及转发函数）。"""
    lazy_exports: dict[str, str] = {}
    wildcard_targets: set[str] = set()
    known_collections: dict[str, list[str]] = {}
    top_dicts: dict[str, ast.Dict] = {}
    getattr_fns: list[ast.FunctionDef] = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "__getattr__" and n.args.args
    ]
    getattr_referenced_names: set[str] = {
        sub.id for fn in getattr_fns for sub in ast.walk(fn) if isinstance(sub, ast.Name)
    }

    for node in tree.body:
        targets: list[str] = []
        val_node: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            val_node = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
            val_node = node.value
        if not targets or val_node is None:
            continue
        seq = _extract_str_seq(val_node, known_collections)
        if seq is not None:
            for tname in targets:
                known_collections[tname] = seq
        if isinstance(val_node, ast.Dict):
            for tname in targets:
                top_dicts[tname] = val_node

    for dname, dnode in top_dicts.items():
        upper = dname.upper()
        if dname not in getattr_referenced_names and not upper.endswith(("EXPORTS", "SUBMODULES", "MODULES")):
            continue
        for k_node, v_node in zip(dnode.keys, dnode.values):
            if not (isinstance(k_node, ast.Constant) and isinstance(k_node.value, str)):
                continue
            sym = k_node.value
            if isinstance(v_node, ast.Constant) and isinstance(v_node.value, str):
                raw_target = v_node.value
                if ":" in raw_target:
                    mod_part, attr_part = raw_target.split(":", 1)
                    resolved_mod = _resolve_rel_mod_name(mod_part, pkg_parts)
                    if resolved_mod and attr_part:
                        lazy_exports[sym] = f"{resolved_mod}.{attr_part}"
                elif raw_target.startswith(".") or "." in raw_target:
                    resolved_mod = _resolve_rel_mod_name(raw_target, pkg_parts)
                    if resolved_mod:
                        lazy_exports[sym] = f"{resolved_mod}.{sym}"
            elif (
                isinstance(v_node, (ast.Tuple, ast.List))
                and len(v_node.elts) == 2
                and all(isinstance(el, ast.Constant) and isinstance(el.value, str) for el in v_node.elts)
            ):
                mod_part = v_node.elts[0].value  # type: ignore[union-attr]
                attr_part = v_node.elts[1].value  # type: ignore[union-attr]
                if isinstance(mod_part, str) and isinstance(attr_part, str) and (
                    mod_part.startswith(".") or "." in mod_part or mod_part.isidentifier()
                ):
                    resolved_mod = _resolve_rel_mod_name(mod_part, pkg_parts)
                    if resolved_mod:
                        lazy_exports[sym] = f"{resolved_mod}.{attr_part}"

    for fn in getattr_fns:
        param_name = fn.args.args[0].arg
        for stmt in fn.body:
            if isinstance(stmt, ast.ImportFrom):
                if stmt.level > 0 and not stmt.module:
                    for alias in stmt.names:
                        if alias.name != "*":
                            submod = _resolve_rel_mod_name("." * stmt.level + alias.name, pkg_parts)
                            if submod:
                                wildcard_targets.add(submod)
                elif stmt.module:
                    raw_m = ("." * stmt.level if stmt.level > 0 else "") + stmt.module
                    submod = _resolve_rel_mod_name(raw_m, pkg_parts)
                    if submod:
                        wildcard_targets.add(submod)
            elif isinstance(stmt, ast.If):
                syms = _matched_getattr_names(stmt.test, param_name, known_collections)
                if not syms:
                    continue
                for sub_node in ast.walk(stmt):
                    if isinstance(sub_node, ast.ImportFrom):
                        if sub_node.level > 0 and not sub_node.module:
                            for alias in sub_node.names:
                                if alias.name != "*":
                                    submod = _resolve_rel_mod_name("." * sub_node.level + alias.name, pkg_parts)
                                    if submod:
                                        for s in syms:
                                            lazy_exports.setdefault(s, f"{submod}.{s}")
                        elif sub_node.module:
                            raw_m = ("." * sub_node.level if sub_node.level > 0 else "") + sub_node.module
                            submod = _resolve_rel_mod_name(raw_m, pkg_parts)
                            if submod:
                                if len(syms) == 1 and len(sub_node.names) == 1 and sub_node.names[0].name != "*":
                                    lazy_exports.setdefault(syms[0], f"{submod}.{sub_node.names[0].name}")
                                else:
                                    for s in syms:
                                        lazy_exports.setdefault(s, f"{submod}.{s}")
                    elif isinstance(sub_node, ast.Call) and sub_node.args:
                        fn_chain = _attr_chain(sub_node.func)
                        is_imp = (
                            (isinstance(sub_node.func, ast.Name) and sub_node.func.id == "import_module")
                            or (fn_chain is not None and fn_chain[-1] == "import_module")
                        )
                        if not is_imp:
                            continue
                        first_arg = sub_node.args[0]
                        if isinstance(first_arg, ast.JoinedStr):
                            # e.g. import_module(f".{name}", __name__)
                            prefix_dots = ""
                            if (
                                len(first_arg.values) == 2
                                and isinstance(first_arg.values[0], ast.Constant)
                                and isinstance(first_arg.values[0].value, str)
                                and isinstance(first_arg.values[1], ast.FormattedValue)
                                and isinstance(first_arg.values[1].value, ast.Name)
                                and first_arg.values[1].value.id == param_name
                            ):
                                prefix_dots = first_arg.values[0].value
                            if prefix_dots.startswith("."):
                                for s in syms:
                                    submod = _resolve_rel_mod_name(f"{prefix_dots}{s}", pkg_parts)
                                    if submod:
                                        lazy_exports.setdefault(s, submod)
                        elif isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                            submod = _resolve_rel_mod_name(first_arg.value, pkg_parts)
                            if submod:
                                for s in syms:
                                    lazy_exports.setdefault(s, f"{submod}.{s}")

    return lazy_exports, wildcard_targets


def _path_to_mod_key(path: Path, project_root: Path) -> tuple[str, bool]:
    rel = path.resolve().relative_to(project_root.resolve())
    parts = rel.with_suffix("").parts
    is_pkg = bool(parts and parts[-1] == "__init__")
    if is_pkg:
        parts = parts[:-1]
    return ".".join(parts), is_pkg


def _infer_zone_from_ni_rua_mod(mod: str) -> tuple[str, str] | None:
    parts = mod.split(".")
    if parts[0] == "web":
        return ("apps", "gateway")
    if parts[0] == "services":
        sub_svc = parts[1] if len(parts) > 1 else ""
        return ("robot", "kinematics") if "curobo" in sub_svc else ("perception", "external")
    if parts[0] == "scripts":
        name = parts[1] if len(parts) > 1 else ""
        if name.startswith(("benchmark", "audit_run", "queue", "summarize_mixed", "reference_", "experiment_")):
            return ("apps", "evaluation")
        if name in ("check_package", "build_package") or "agent_plugin" in name:
            return ("packaging", "wheel")
        if name.startswith("run_tests"):
            return ("core", "runner")
        if name.startswith("so101"):
            return ("robot", "so101")
        if name == "compact_observation_images":
            return ("core", "storage")
        return None
    if parts[0] != "ni_rua" or len(parts) < 2:
        return None
    zone = parts[1]
    if zone not in ("core", "robot", "perception", "cortex", "runtime", "apps"):
        return None
    sub = parts[2] if len(parts) > 2 else ""
    if zone == "core":
        subzone = sub if sub in ("storage", "config", "presentation") else ("contracts" if "contract" in sub else "general")
    elif zone == "robot":
        if sub == "embodiments" or (len(parts) > 3 and parts[2] == "model" and parts[3] == "embodiments"):
            subzone = "embodiment"
        elif sub == "calibration" or (len(parts) > 3 and "calibration" in parts[3]):
            subzone = "calibration"
        elif sub == "backends" and len(parts) > 3:
            subzone = parts[3] if parts[3] in ("so101", "robodojo") else "motion"
        elif sub == "model":
            subzone = "kinematics"
        else:
            subzone = "motion"
    elif zone == "perception":
        subzone = sub if sub in ("depth", "scene", "external") else "scene"
    elif zone == "cortex":
        if sub == "agents" and len(parts) > 3 and parts[3] in ("codex", "claude_code", "opencode", "external"):
            subzone = parts[3]
        elif sub == "relay":
            subzone = "relay"
        elif sub == "session" and len(parts) > 3:
            s3 = parts[3]
            subzone = "codex" if "codex" in s3 else ("relay" if ("model_" in s3 or "wire" in s3) else "agents")
        else:
            subzone = "agents"
    elif zone == "runtime":
        subzone = sub if sub in ("actions", "tools", "programs") else "harness"
    else:
        subzone = sub if sub in ("gateway", "evaluation", "library", "replay") else ("gateway" if sub == "remote" else "library")
    return (zone, subzone)


def _infer_test_zone(
    test_path: Path,
    tests_dir: Path,
    imported_mod_names: Sequence[str],
) -> tuple[str, str]:
    try:
        rel = test_path.resolve().relative_to(tests_dir.resolve())
        parts = rel.parts
        if len(parts) >= 2 and parts[0] in VALID_ZONES:
            return parts[0], (parts[1] if len(parts) >= 3 else "general")
    except ValueError:
        pass

    stem = test_path.stem
    if stem in _FLAT_ZONE_OVERRIDES:
        return _FLAT_ZONE_OVERRIDES[stem]

    for prefix, mapping in _STEM_ZONE_PREFIXES:
        if stem.startswith(prefix):
            return mapping

    best: tuple[str, str] | None = None
    for mod in imported_mod_names:
        cand = _infer_zone_from_ni_rua_mod(mod)
        if cand is not None and (best is None or _LAYER_PRIORITY[cand[0]] > _LAYER_PRIORITY[best[0]]):
            best = cand
    if best is not None:
        return best
    return ("core", "general")


def _infer_test_tier(test_path: Path, text: str, explicit_tier: str | None, zone: str) -> str:
    if explicit_tier in VALID_TIERS:
        return explicit_tier
    stem = test_path.stem
    if zone == "packaging" or stem in _PACKAGING_TEST_STEMS or "check_package(" in text or "build_package(" in text:
        return "packaging"
    if _COMPONENT_SIGNAL_RE.search(text):
        return "component"
    return "fast"


def _collect_repo_py_files(project_root: Path, tests_dir: Path) -> tuple[list[Path], list[Path]]:
    """收集仓库内源码文件与非测试支持文件，排除 worktrees 与虚拟环境目录。"""
    src_files: list[Path] = []
    for sub in ("ni_rua", "scripts", "services", "web"):
        base = project_root / sub
        if not base.is_dir():
            continue
        for root, dirs, files in os.walk(base):
            rp = Path(root)
            dirs[:] = sorted(d for d in dirs if d not in _EXCLUDED_DISCOVERY_DIRS)
            for fname in sorted(files):
                if fname.endswith(".py"):
                    src_files.append((rp / fname).resolve())
    support_files: list[Path] = []
    if tests_dir.is_dir():
        for root, dirs, files in os.walk(tests_dir):
            rp = Path(root)
            if _is_excluded_relpath(rp, tests_dir):
                dirs[:] = []
                continue
            dirs[:] = sorted(d for d in dirs if d not in _EXCLUDED_DISCOVERY_DIRS)
            for fname in sorted(files):
                if fname.endswith(".py") and not fname.startswith("test_"):
                    support_files.append((rp / fname).resolve())
    return src_files, support_files


@dataclass(frozen=True)
class _StaticGraphAux:
    src_files: tuple[Path, ...]
    support_files: tuple[Path, ...]
    raw_texts: dict[Path, str]
    expanded_refs: dict[Path, set[str]]
    bare_pkg_importers: dict[Path, set[Path]]


def _build_static_index(
    project_root: Path,
    tests_dir: Path,
    test_entries: Sequence[tuple[str, Path]],
    *,
    full_graph: bool,
) -> tuple[
    dict[Path, TestClassification],
    dict[Path, set[Path]],
    dict[Path, str],
    dict[str, Path],
    _StaticGraphAux,
]:
    """一次遍历完成测试域/层级分类；仅在需要 --changed DAG 时构建全仓反向依赖图。"""
    resolved_tests_dir = tests_dir.resolve()
    resolved_project_root = project_root.resolve()
    test_paths = [p.resolve() for _, p in test_entries]
    test_mod_by_path = {p.resolve(): m for m, p in test_entries}
    src_files, support_files = _collect_repo_py_files(resolved_project_root, resolved_tests_dir) if full_graph else ([], [])

    all_mods: dict[str, Path] = {}
    mod_info_by_path: dict[Path, tuple[str, bool]] = {}
    test_sub_key_by_path: dict[Path, str] = {}
    for p in src_files + support_files + test_paths:
        mod_key, is_pkg = _path_to_mod_key(p, resolved_project_root)
        all_mods[mod_key] = p
        mod_info_by_path[p] = (mod_key, is_pkg)
        if p.is_relative_to(resolved_tests_dir):
            sub_key, _ = _path_to_mod_key(p, resolved_tests_dir)
            if sub_key:
                all_mods[sub_key] = p
                test_sub_key_by_path[p] = sub_key

    raw_imports: dict[Path, tuple[str, list[str], set[str], str | None]] = {}
    raw_texts: dict[Path, str] = {}
    pkg_exports: dict[str, dict[str, str]] = {}
    wildcard_exports: dict[str, set[str]] = {}

    scan_paths = (src_files + support_files + test_paths) if full_graph else test_paths
    for p in scan_paths:
        text = p.read_text(encoding="utf-8", errors="replace")
        raw_texts[p] = text
        tree = _parse_python_ast(text)
        explicit_tier = _extract_tier_marker(tree) if p in test_mod_by_path else None
        mod_key, is_pkg = mod_info_by_path[p]
        pkg_parts = mod_key.split(".") if is_pkg else mod_key.split(".")[:-1]

        imported_mod_names: list[str] = []
        direct_refs: set[str] = set()
        bound_aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_mod_names.append(alias.name)
                    direct_refs.add(alias.name)
                    if alias.asname:
                        bound_aliases[alias.asname] = alias.name
                        if is_pkg and full_graph:
                            pkg_exports.setdefault(mod_key, {})[alias.asname] = alias.name
                    else:
                        top_name = alias.name.split(".", 1)[0]
                        bound_aliases.setdefault(top_name, top_name)
            elif isinstance(node, ast.ImportFrom):
                base = ""
                if node.level > 0:
                    up = node.level - 1
                    anchor = pkg_parts[: len(pkg_parts) - up] if up <= len(pkg_parts) else []
                    base = ".".join([*anchor, *(node.module.split(".") if node.module else ())])
                elif node.module:
                    base = node.module
                if base:
                    imported_mod_names.append(base)
                    direct_refs.add(base)
                    for alias in node.names:
                        if alias.name == "*":
                            if full_graph:
                                wildcard_exports.setdefault(mod_key, set()).add(base)
                        else:
                            exported_as = alias.asname or alias.name
                            target_ref = f"{base}.{alias.name}"
                            imported_mod_names.append(target_ref)
                            direct_refs.add(target_ref)
                            bound_aliases[exported_as] = target_ref
                            if is_pkg and full_graph:
                                pkg_exports.setdefault(mod_key, {})[exported_as] = target_ref

        if full_graph and bound_aliases:
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    chain = _attr_chain(node)
                    if chain is not None and len(chain) >= 2 and chain[0] in bound_aliases:
                        prefix = bound_aliases[chain[0]]
                        for k in range(2, len(chain) + 1):
                            attr_ref = f"{prefix}.{'.'.join(chain[1:k])}"
                            direct_refs.add(attr_ref)

        if full_graph:
            lazy_map, lazy_wildcards = _extract_lazy_barrel_info(tree, pkg_parts)
            if lazy_map:
                exp_tbl = pkg_exports.setdefault(mod_key, {})
                for sym, target_ref in lazy_map.items():
                    exp_tbl.setdefault(sym, target_ref)
                    direct_refs.add(target_ref)
            if lazy_wildcards:
                wildcard_exports.setdefault(mod_key, set()).update(lazy_wildcards)
                direct_refs.update(lazy_wildcards)
            sub_key = test_sub_key_by_path.get(p)
            if sub_key:
                if mod_key in pkg_exports:
                    pkg_exports.setdefault(sub_key, {}).update(pkg_exports[mod_key])
                if mod_key in wildcard_exports:
                    wildcard_exports.setdefault(sub_key, set()).update(wildcard_exports[mod_key])

        raw_imports[p] = (text, imported_mod_names, direct_refs, explicit_tier)

    def _resolve_ref_bundle(ref: str) -> tuple[set[Path], set[str]]:
        paths_out: set[Path] = set()
        seen_refs: set[str] = set()
        stack = [ref]
        while stack:
            cur = stack.pop()
            if cur in seen_refs:
                continue
            seen_refs.add(cur)
            for _ in range(4):
                if "." in cur:
                    pkg_part, sym = cur.rsplit(".", 1)
                    reexp = pkg_exports.get(pkg_part, {}).get(sym)
                    if reexp and reexp != cur:
                        cur = reexp
                        seen_refs.add(cur)
                        continue
                break
            if "." in cur:
                pkg_part, sym = cur.rsplit(".", 1)
                if cur not in all_mods and sym not in pkg_exports.get(pkg_part, {}):
                    for w_base in wildcard_exports.get(pkg_part, ()):
                        w_cand = f"{w_base}.{sym}"
                        if w_cand not in seen_refs:
                            stack.append(w_cand)
            if cur in all_mods:
                paths_out.add(all_mods[cur])
            elif "." in cur:
                parent = cur.rsplit(".", 1)[0]
                if parent in all_mods:
                    paths_out.add(all_mods[parent])
        return paths_out, seen_refs

    rev_graph: dict[Path, set[Path]] = {}
    expanded_refs_by_path: dict[Path, set[str]] = {}
    bare_pkg_importers: dict[Path, set[Path]] = {}
    classifications: dict[Path, TestClassification] = {}

    script_like_assets: list[tuple[Path, str, str]] = []
    script_name_counts: Counter[str] = Counter()
    if full_graph:
        for sp in src_files:
            if sp.name == "__init__.py":
                continue
            try:
                sp_rel = sp.relative_to(resolved_project_root).as_posix()
            except ValueError:
                continue
            if sp_rel.startswith(("scripts/", "services/", "web/")):
                script_like_assets.append((sp, sp.name, sp_rel))
                script_name_counts[sp.name] += 1

    for p in scan_paths:
        text, imported_mod_names, direct_refs, explicit_tier = raw_imports[p]
        resolved_deps: set[Path] = set()
        expanded_refs: set[str] = set(direct_refs)
        if full_graph:
            explicit_deps: set[Path] = set()
            for ref in direct_refs:
                dep_set, seen_r = _resolve_ref_bundle(ref)
                expanded_refs.update(seen_r)
                for dep in dep_set:
                    if dep != p:
                        explicit_deps.add(dep)

            for dep in explicit_deps:
                if dep.name == "__init__.py":
                    has_specific_child = any(
                        other != dep and other != p and other.is_relative_to(dep.parent)
                        for other in explicit_deps
                    )
                    if not has_specific_child:
                        bare_pkg_importers.setdefault(dep, set()).add(p)

            resolved_deps.update(explicit_deps)
            for dep in list(explicit_deps):
                dep_mod_key, _ = mod_info_by_path[dep]
                dep_parts = dep_mod_key.split(".")
                for k in range(1, len(dep_parts)):
                    anc_mod = ".".join(dep_parts[:k])
                    anc_path = all_mods.get(anc_mod)
                    if anc_path is not None and anc_path.name == "__init__.py" and anc_path != p:
                        resolved_deps.add(anc_path)

            for sp, sp_name, sp_rel in script_like_assets:
                if sp != p and (sp_rel in text or (script_name_counts[sp_name] == 1 and sp_name in text)):
                    resolved_deps.add(sp)

            for dep in resolved_deps:
                if dep != p:
                    rev_graph.setdefault(dep, set()).add(p)

        expanded_refs_by_path[p] = expanded_refs

        if p in test_mod_by_path:
            zone, subzone = _infer_test_zone(p, resolved_tests_dir, imported_mod_names)
            tier = _infer_test_tier(p, text, explicit_tier, zone)
            classifications[p] = TestClassification(
                path=p,
                module_name=test_mod_by_path[p],
                zone=zone,
                subzone=subzone,
                tier=tier,
                imports=frozenset(resolved_deps),
            )

    path_mod_names = {p: info[0] for p, info in mod_info_by_path.items()}
    aux = _StaticGraphAux(
        src_files=tuple(src_files),
        support_files=tuple(support_files),
        raw_texts=raw_texts,
        expanded_refs=expanded_refs_by_path,
        bare_pkg_importers=bare_pkg_importers,
    )
    return classifications, rev_graph, path_mod_names, all_mods, aux


def _collect_git_changed_files(project_root: Path, base_ref: str) -> tuple[str, ...]:
    cmds = (
        ["git", "diff", "--name-only", "--no-renames", "-z", base_ref],
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
    )
    changed: set[str] = set()
    for cmd in cmds:
        proc = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True)
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip()
            raise SystemExit(f"git 变更检测失败 ({' '.join(cmd)}): {detail}")
        for entry in proc.stdout.split("\0"):
            clean = entry.strip("\r\n")
            if clean:
                changed.add(Path(clean).as_posix())
    return tuple(sorted(changed))


def _is_doc_only_path(rel_path: str) -> bool:
    fname = Path(rel_path).name.lower()
    if rel_path.endswith(".txt") and ("requirements" in fname or "constraints" in fname):
        return False
    return (
        rel_path.startswith("docs/")
        or rel_path.endswith((".md", ".rst", ".txt"))
        or rel_path in {".gitignore", "LICENSE"}
    )


def _deleted_py_candidate_mods(rel_path: str) -> tuple[set[str], tuple[str, str] | None]:
    no_ext = rel_path[:-3]
    if no_ext.endswith("/__init__"):
        no_ext = no_ext[: -len("/__init__")]
    elif no_ext == "__init__":
        no_ext = ""
    if not no_ext:
        return set(), None
    mod = no_ext.replace("/", ".")
    cands = {mod}
    if mod.startswith("tests."):
        stripped = mod.removeprefix("tests.")
        if stripped:
            cands.add(stripped)
    pkg_leaf = mod.rsplit(".", 1) if "." in mod else None
    pair = (pkg_leaf[0], pkg_leaf[1]) if pkg_leaf is not None else None
    return cands, pair


def _select_changed_tests(
    project_root: Path,
    tests_dir: Path,
    test_entries: Sequence[tuple[str, Path]],
    *,
    base_ref: str,
    changed_scope: str,
    zone_filter: set[str],
    subzone_filter: set[str],
    tier_filter: set[str],
) -> SelectionPlan:
    t0 = time.perf_counter()
    resolved_root = project_root.resolve()
    resolved_tests_dir = tests_dir.resolve()
    classifications, rev_graph, path_mod_names, _all_mods, aux = _build_static_index(
        resolved_root, resolved_tests_dir, test_entries, full_graph=True
    )
    changed_files = _collect_git_changed_files(resolved_root, base_ref)
    test_path_set = set(classifications.keys())
    src_path_set = set(aux.src_files)
    support_path_set = set(aux.support_files)

    selected_depth: dict[Path, int] = {}
    reasons: dict[Path, str] = {}
    seed_prod: set[Path] = set()
    seed_support: list[tuple[Path, str]] = []
    non_py_fixtures: list[str] = []
    run_js_web = False
    run_js_plugin = False

    def _record_test(tpath: Path, depth: int, reason: str) -> None:
        prev = selected_depth.get(tpath)
        if prev is None or depth < prev:
            selected_depth[tpath] = depth
            reasons[tpath] = reason

    def _propagate_test_tree(start_node: Path, base_depth: int, base_chain: str) -> None:
        seen_nodes: set[Path] = {start_node}
        t_queue: deque[tuple[Path, str]] = deque([(start_node, base_chain)])
        while t_queue:
            cur_node, cur_chain = t_queue.popleft()
            for t_caller in sorted(rev_graph.get(cur_node, ())):
                if t_caller in test_path_set or t_caller.is_relative_to(resolved_tests_dir):
                    nxt_chain = f"{cur_chain} -> {t_caller.name}"
                    if t_caller in test_path_set:
                        _record_test(t_caller, base_depth, nxt_chain)
                    if t_caller not in seen_nodes:
                        seen_nodes.add(t_caller)
                        t_queue.append((t_caller, nxt_chain))

    def _seed_asset_reference(rel_path: str, label: str) -> None:
        fname = Path(rel_path).name
        for p, txt in aux.raw_texts.items():
            if rel_path in txt or (len(fname) >= 5 and fname in txt):
                if p in test_path_set:
                    _record_test(p, 0, f"{label} {rel_path}")
                    _propagate_test_tree(p, 0, f"{label} {rel_path}")
                elif p in support_path_set:
                    seed_support.append((p, rel_path))
                elif p in src_path_set:
                    seed_prod.add(p)

    for rel in changed_files:
        if rel.startswith("agent-plugin/"):
            run_js_plugin = True
            for tpath in test_path_set:
                if tpath.stem.startswith("test_agent_plugin"):
                    _record_test(tpath, 0, f"{rel} -> agent-plugin")
                    _propagate_test_tree(tpath, 0, f"{rel} -> agent-plugin")
            continue

        if rel.startswith("web/"):
            run_js_web = True
            for tpath in test_path_set:
                stem = tpath.stem
                if stem.startswith("test_gateway_console") or stem.startswith("test_console_"):
                    _record_test(tpath, 0, f"{rel} -> gateway console")
                    _propagate_test_tree(tpath, 0, f"{rel} -> gateway console")
            _seed_asset_reference(rel, "web")
            if not rel.endswith(".py"):
                continue

        rel_fname_lower = Path(rel).name.lower()
        if (
            rel in {"pyproject.toml", "environment.yml"}
            or (rel.endswith(".txt") and ("requirements" in rel_fname_lower or "constraints" in rel_fname_lower))
        ):
            for tpath, cls in classifications.items():
                if (
                    cls.tier == "packaging"
                    or cls.zone == "packaging"
                    or tpath.stem.startswith("test_config")
                    or tpath.stem in {"test_parallel_runner", "test_run_tests_per_module"}
                ):
                    _record_test(tpath, 0, f"{rel} -> packaging/config")
                    _propagate_test_tree(tpath, 0, f"{rel} -> packaging/config")
            _seed_asset_reference(rel, "config")
            continue

        if rel.startswith("configs/"):
            for seed_rel in (
                "ni_rua/core/config/__init__.py",
                "ni_rua/core/config/loader.py",
                "ni_rua/core/config/sections.py",
                "ni_rua/robot/backends/so101/deployment.py",
            ):
                cand = (resolved_root / seed_rel).resolve()
                if cand.is_file():
                    seed_prod.add(cand)
            for tpath, cls in classifications.items():
                if tpath.stem.startswith(("test_config", "test_embodiment")) or cls.subzone in {"config", "embodiment"}:
                    _record_test(tpath, 0, f"{rel} -> config/embodiment")
                    _propagate_test_tree(tpath, 0, f"{rel} -> config/embodiment")
            _seed_asset_reference(rel, "config")
            continue

        if rel.startswith("benchmarks/"):
            bg = (resolved_root / "scripts" / "benchmark_group.py").resolve()
            if bg.is_file():
                seed_prod.add(bg)
            for tpath in test_path_set:
                if tpath.stem.startswith("test_benchmark_group"):
                    _record_test(tpath, 0, f"{rel} -> benchmark_group")
                    _propagate_test_tree(tpath, 0, f"{rel} -> benchmark_group")
            _seed_asset_reference(rel, "benchmark")
            continue

        abs_path = (resolved_root / rel).resolve()
        if abs_path in test_path_set:
            _record_test(abs_path, 0, f"changed test {rel}")
            _propagate_test_tree(abs_path, 0, f"changed test {rel}")
            continue
        if rel.startswith("tests/") and rel.endswith(".py") and abs_path.is_file():
            seed_support.append((abs_path, rel))
            continue
        if rel.startswith("tests/"):
            if not rel.endswith(".py"):
                non_py_fixtures.append(rel)
                continue

        if rel.startswith("ni_rua/") and not rel.endswith(".py"):
            ni_rua_root = (resolved_root / "ni_rua").resolve()
            cur_dir = abs_path.parent
            while cur_dir.is_relative_to(ni_rua_root):
                if cur_dir.is_dir():
                    py_siblings = sorted(p.resolve() for p in cur_dir.glob("*.py") if p.is_file())
                    if py_siblings:
                        seed_prod.update(py_siblings)
                        break
                if cur_dir == ni_rua_root:
                    break
                cur_dir = cur_dir.parent
            _seed_asset_reference(rel, "asset")
            continue

        if _is_doc_only_path(rel):
            continue

        if rel == "scripts/run_tests.py":
            for tpath in test_path_set:
                if tpath.stem == "test_parallel_runner":
                    _record_test(tpath, 0, f"{rel} -> test_parallel_runner")
                    _propagate_test_tree(tpath, 0, f"{rel} -> test_parallel_runner")
        elif rel == "scripts/run_tests_per_module.py":
            for tpath in test_path_set:
                if tpath.stem == "test_run_tests_per_module":
                    _record_test(tpath, 0, f"{rel} -> test_run_tests_per_module")
                    _propagate_test_tree(tpath, 0, f"{rel} -> test_run_tests_per_module")

        if rel.endswith(".py") and abs_path.is_file():
            seed_prod.add(abs_path)
            if rel.startswith(("scripts/", "services/", "web/")):
                _seed_asset_reference(rel, "script")
        elif rel.endswith(".py"):
            # 已删除或重命名移走的 .py 文件：匹配模块全名、(parent_pkg, leaf)、相对导入及脚本名引用
            del_cands, pkg_leaf = _deleted_py_candidate_mods(rel)
            script_name = Path(rel).name if rel.startswith(("scripts/", "services/", "web/")) else ""
            for p, exp_refs in aux.expanded_refs.items():
                matched = any(
                    r in del_cands or any(r.startswith(f"{m}.") for m in del_cands)
                    for r in exp_refs
                )
                if not matched and pkg_leaf is not None:
                    parent_pkg, leaf_name = pkg_leaf
                    matched = f"{parent_pkg}.{leaf_name}" in exp_refs
                if not matched and script_name:
                    txt = aux.raw_texts.get(p, "")
                    matched = rel in txt or script_name in txt
                if not matched:
                    continue
                if p in test_path_set:
                    _record_test(p, 0, f"deleted {rel}")
                    _propagate_test_tree(p, 0, f"deleted {rel}")
                elif p in support_path_set:
                    seed_support.append((p, f"deleted {rel}"))
                elif p in src_path_set:
                    seed_prod.add(p)

    for sup_path, rel in seed_support:
        _propagate_test_tree(sup_path, 0, f"imports {rel}")

    if non_py_fixtures:
        for p in support_path_set | test_path_set:
            txt = aux.raw_texts.get(p, "")
            for fix_rel in non_py_fixtures:
                if Path(fix_rel).name in txt or fix_rel in txt:
                    if p in test_path_set:
                        _record_test(p, 0, f"fixture {fix_rel}")
                        _propagate_test_tree(p, 0, f"fixture {fix_rel}")
                    else:
                        _propagate_test_tree(p, 0, f"fixture {fix_rel} -> {p.name}")
                    break

    # 2-Stage Hub-Bounded BFS
    visited_prod: dict[Path, int] = {p: 0 for p in seed_prod}
    queue: deque[tuple[Path, int, str]] = deque(
        (p, 0, p.relative_to(resolved_root).as_posix()) for p in sorted(seed_prod)
    )

    def _enqueue_prod(prod_path: Path, next_depth: int, next_chain: str) -> None:
        if prod_path not in visited_prod or next_depth < visited_prod[prod_path]:
            visited_prod[prod_path] = next_depth
            queue.append((prod_path, next_depth, next_chain))

    while queue:
        cur_path, depth, chain = queue.popleft()
        cur_is_unchanged_hub = (
            changed_scope == "auto"
            and path_mod_names.get(cur_path, "") in COMPOSITION_ROOT_HUBS
            and cur_path not in seed_prod
        )
        for caller in sorted(rev_graph.get(cur_path, ())):
            if caller in test_path_set:
                msg = f"depth={depth}: {chain} -> {caller.name}"
                _record_test(caller, depth, msg)
                _propagate_test_tree(caller, depth, msg)
                continue
            if caller.is_relative_to(resolved_tests_dir):
                _propagate_test_tree(caller, depth, f"depth={depth}: {chain} -> {caller.name}")
                continue
            if changed_scope == "direct":
                continue
            if cur_is_unchanged_hub:
                continue
            if changed_scope == "auto" and caller.name == "__init__.py" and caller not in seed_prod:
                # Barrel Guard: 已知子模块导出已建立直连边；仅对未解析到具体子模块的裸包导入保守透传
                barrel_rel = caller.relative_to(resolved_root).as_posix()
                barrel_chain = f"{chain} -> {barrel_rel}"
                for bare_caller in sorted(aux.bare_pkg_importers.get(caller, ())):
                    if bare_caller in test_path_set:
                        b_msg = f"depth={depth}: {barrel_chain} -> {bare_caller.name}"
                        _record_test(bare_caller, depth, b_msg)
                        _propagate_test_tree(bare_caller, depth, b_msg)
                    elif bare_caller.is_relative_to(resolved_tests_dir):
                        _propagate_test_tree(bare_caller, depth, f"depth={depth}: {barrel_chain} -> {bare_caller.name}")
                    else:
                        b_rel = bare_caller.relative_to(resolved_root).as_posix()
                        _enqueue_prod(bare_caller, depth + 1, f"{barrel_chain} -> {b_rel}")
                continue
            next_depth = depth + 1
            caller_rel = caller.relative_to(resolved_root).as_posix()
            _enqueue_prod(caller, next_depth, f"{chain} -> {caller_rel}")

    damper_skipped_count = 0
    filtered: list[Path] = []
    for tpath in sorted(selected_depth, key=lambda p: classifications[p].module_name):
        cls = classifications[tpath]
        depth = selected_depth[tpath]
        if changed_scope == "auto" and not tier_filter and depth >= 3 and cls.tier != "fast":
            damper_skipped_count += 1
            continue
        if zone_filter and cls.zone not in zone_filter:
            continue
        if subzone_filter and cls.subzone not in subzone_filter and f"{cls.zone}.{cls.subzone}" not in subzone_filter:
            continue
        if tier_filter and cls.tier not in tier_filter:
            continue
        filtered.append(tpath)

    return SelectionPlan(
        selected_tests=tuple(filtered),
        reasons=reasons,
        classifications=classifications,
        changed_files=changed_files,
        run_js_web=run_js_web,
        run_js_plugin=run_js_plugin,
        damper_skipped_count=damper_skipped_count,
        scan_elapsed_s=time.perf_counter() - t0,
    )


def _select_static_filtered_tests(
    project_root: Path,
    tests_dir: Path,
    test_entries: Sequence[tuple[str, Path]],
    *,
    zone_filter: set[str],
    subzone_filter: set[str],
    tier_filter: set[str],
    include_js: bool,
) -> SelectionPlan:
    t0 = time.perf_counter()
    classifications, _rev, _pm, _am, _aux = _build_static_index(
        project_root, tests_dir, test_entries, full_graph=False
    )
    selected: list[Path] = []
    reasons: dict[Path, str] = {}
    for _, tpath in test_entries:
        resolved = tpath.resolve()
        cls = classifications[resolved]
        if zone_filter and cls.zone not in zone_filter:
            continue
        if subzone_filter and cls.subzone not in subzone_filter and f"{cls.zone}.{cls.subzone}" not in subzone_filter:
            continue
        if tier_filter and cls.tier not in tier_filter:
            continue
        selected.append(resolved)
        reasons[resolved] = f"zone={cls.zone}.{cls.subzone} tier={cls.tier}"

    run_js_pkg = include_js or ("packaging" in zone_filter)
    return SelectionPlan(
        selected_tests=tuple(selected),
        reasons=reasons,
        classifications=classifications,
        changed_files=(),
        run_js_web=run_js_pkg and (project_root / "web" / "check.mjs").is_file(),
        run_js_plugin=run_js_pkg and (project_root / "agent-plugin" / "package.json").is_file(),
        damper_skipped_count=0,
        scan_elapsed_s=time.perf_counter() - t0,
    )


def _parse_valid_timing_seconds(value: object) -> float | None:
    if type(value) not in (int, float):
        return None
    if isinstance(value, int) and not (0 <= value < 1_000_000_000):
        return None
    fv = float(value)
    if not math.isfinite(fv) or not (0.0 <= fv < 1e9):
        return None
    return fv


def _load_module_timings(project_root: Path, start_dir: Path) -> dict[str, float]:
    resolved_root = project_root.resolve()
    if not start_dir.resolve().is_relative_to(resolved_root):
        return {}
    timings_path = resolved_root / TIMINGS_FILENAME
    if not timings_path.is_file():
        return {}
    try:
        raw = json.loads(timings_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    modules = raw.get("modules") if isinstance(raw, dict) else None
    if not isinstance(modules, dict):
        return {}
    out: dict[str, float] = {}
    for mod_name, entry in modules.items():
        if not isinstance(mod_name, str):
            continue
        raw_val = entry.get("elapsed_s") if isinstance(entry, dict) else entry
        parsed = _parse_valid_timing_seconds(raw_val)
        if parsed is not None:
            out[mod_name] = parsed
    return out


def _save_module_timings(
    project_root: Path,
    start_dir: Path,
    measured: dict[str, tuple[float, int]],
) -> None:
    if not measured:
        return
    resolved_root = project_root.resolve()
    if not start_dir.resolve().is_relative_to(resolved_root):
        return
    if resolved_root != REPO_ROOT.resolve() and not (resolved_root / ".git").exists():
        return
    timings_path = resolved_root / TIMINGS_FILENAME
    existing_modules: dict[str, dict[str, object]] = {}
    if timings_path.is_file():
        try:
            raw = json.loads(timings_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and isinstance(raw.get("modules"), dict):
                for k, v in raw["modules"].items():
                    if not isinstance(k, str):
                        continue
                    raw_el = v.get("elapsed_s") if isinstance(v, dict) else v
                    valid_el = _parse_valid_timing_seconds(raw_el)
                    if valid_el is not None:
                        prev_tests = v.get("tests", 0) if isinstance(v, dict) else 0
                        safe_tests = prev_tests if type(prev_tests) is int and prev_tests >= 0 else 0
                        existing_modules[k] = {"elapsed_s": round(valid_el, 4), "tests": safe_tests}
        except (OSError, ValueError):
            existing_modules = {}
    for mod_name, (elapsed_s, count) in measured.items():
        valid_el = _parse_valid_timing_seconds(elapsed_s)
        if valid_el is not None:
            existing_modules[mod_name] = {"elapsed_s": round(valid_el, 4), "tests": int(count)}
    payload = {"version": 1, "modules": dict(sorted(existing_modules.items()))}
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(resolved_root),
            prefix=".ni-rua-test-timings.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            tmp_path = Path(fh.name)
            fh.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(tmp_path, timings_path)
    except (OSError, ValueError):
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass


def _print_explain_plan(plan: SelectionPlan, project_root: Path, *, changed_mode: bool, changed_scope: str) -> None:
    print(f"[explain] 静态扫描耗时: {plan.scan_elapsed_s:.3f}s")
    if changed_mode:
        print(f"[explain] changed_scope={changed_scope}, git 变更文件 ({len(plan.changed_files)}):")
        for rel in plan.changed_files:
            print(f"  - {rel}")
    if plan.damper_skipped_count:
        print(f"[explain] Hub 阻尼跳过的 depth>=3 非 fast 测试模块数: {plan.damper_skipped_count}")
    js_cmds: list[str] = []
    if plan.run_js_web and (project_root / "web" / "check.mjs").is_file():
        js_cmds.append("node web/check.mjs")
    if plan.run_js_plugin and (project_root / "agent-plugin" / "package.json").is_file():
        js_cmds.append("npm --prefix agent-plugin test")
    if js_cmds:
        print(f"[explain] 委托 JS 检查 ({len(js_cmds)}):")
        for cmd in js_cmds:
            print(f"  - {cmd}")
    print(f"[explain] 选中 Python 测试模块 ({len(plan.selected_tests)}):")
    for tpath in plan.selected_tests:
        cls = plan.classifications[tpath]
        reason = plan.reasons.get(tpath, "")
        print(f"  - {cls.module_name} [{cls.zone}.{cls.subzone} | {cls.tier}] <- {reason}")


def _enumerate_ast_test_cases(
    test_file: Path,
    module_name: str,
    tests_dir: Path,
    _cache: dict[Path, dict[str, set[str]]] | None = None,
) -> list[str]:
    """静态提取模块内全部 `<module>.<ClassQualName>.<test_method>` 标识符（支持跨基类继承），供 --explain -k 与类/方法后缀校验。"""
    cache = _cache if _cache is not None else {}
    resolved_tests = tests_dir.resolve()

    def _class_methods_for_file(fpath: Path, visited_files: frozenset[Path]) -> dict[str, set[str]]:
        rf = fpath.resolve()
        if rf in cache:
            return cache[rf]
        if rf in visited_files or not rf.is_file():
            return {}
        try:
            txt = rf.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return {}
        tree = _parse_python_ast(txt)
        next_visited = visited_files | {rf}

        imported_symbol_src: dict[str, tuple[Path, str]] = {}
        imported_mod_src: dict[str, Path] = {}

        def _find_test_mod_file(mod_spec: str) -> Path | None:
            stripped = mod_spec.removeprefix("tests.")
            for cand_str in (stripped, mod_spec):
                parts = cand_str.split(".")
                py_p = (resolved_tests / Path(*parts)).with_suffix(".py")
                if py_p.is_file():
                    return py_p.resolve()
            return None

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    src_p = _find_test_mod_file(alias.name)
                    if src_p is not None:
                        imported_mod_src[alias.asname or alias.name.split(".")[-1]] = src_p
                        imported_mod_src[alias.name] = src_p
            elif isinstance(node, ast.ImportFrom) and node.module:
                src_p = _find_test_mod_file(node.module)
                if src_p is not None:
                    for alias in node.names:
                        if alias.name != "*":
                            imported_symbol_src[alias.asname or alias.name] = (src_p, alias.name)

        raw_cls: dict[str, tuple[set[str], list[str]]] = {}

        def _walk_scope(stmts: Sequence[ast.stmt], prefix: str) -> None:
            for st in stmts:
                if isinstance(st, ast.ClassDef):
                    qual = f"{prefix}.{st.name}" if prefix else st.name
                    if st.bases:
                        own = {
                            fn.name
                            for fn in st.body
                            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name.startswith("test")
                        }
                        base_refs: list[str] = []
                        for b in st.bases:
                            ch = _attr_chain(b)
                            if ch is not None:
                                base_refs.append(".".join(ch))
                        raw_cls[qual] = (own, base_refs)
                    _walk_scope(st.body, qual)

        _walk_scope(tree.body, "")

        resolved_cls: dict[str, set[str]] = {}

        def _resolve_cls(qname: str, seen_q: frozenset[str]) -> set[str]:
            if qname in resolved_cls:
                return resolved_cls[qname]
            if qname in seen_q or qname not in raw_cls:
                return set()
            own_m, bases = raw_cls[qname]
            merged = set(own_m)
            nxt_seen = seen_q | {qname}
            for bref in bases:
                if bref in raw_cls:
                    merged |= _resolve_cls(bref, nxt_seen)
                elif bref in imported_symbol_src:
                    ext_file, ext_cls = imported_symbol_src[bref]
                    ext_map = _class_methods_for_file(ext_file, next_visited)
                    merged |= ext_map.get(ext_cls, set())
                elif "." in bref:
                    mod_alias, cls_part = bref.rsplit(".", 1)
                    if mod_alias in imported_mod_src:
                        ext_map = _class_methods_for_file(imported_mod_src[mod_alias], next_visited)
                        merged |= ext_map.get(cls_part, set())
            resolved_cls[qname] = merged
            return merged

        for q in list(raw_cls):
            _resolve_cls(q, frozenset())
        cache[rf] = resolved_cls
        return resolved_cls

    cls_map = _class_methods_for_file(test_file, frozenset())
    out: list[str] = []
    for qual, methods in sorted(cls_map.items()):
        for mname in sorted(methods):
            out.append(f"{module_name}.{qual}.{mname}")
    return out


def _narrow_selection_plan(
    plan: SelectionPlan,
    *,
    raw_targets: Sequence[str],
    k_patterns: Sequence[str] | None,
    start_dir: Path,
    top_level_dir: Path | None,
    project_root: Path,
    explain_mode: bool,
) -> tuple[SelectionPlan, list[str], set[str]]:
    """在选择器/explain 模式下解析目录、模块、类与方法目标，并在 --explain 下同步应用 -k 过滤。"""
    resolved_start = start_dir.resolve()
    resolved_root = project_root.resolve()
    name_pats = [f"*{pat}*" if "*" not in pat else pat for pat in k_patterns] if k_patterns else None

    matched_dirs: list[Path] = []
    full_files: set[Path] = set()
    glob_pats: list[str] = []
    partial_specs_by_file: dict[Path, list[tuple[str, str]]] = {}

    for raw_t in raw_targets:
        cand_p = Path(raw_t)
        if cand_p.is_dir():
            matched_dirs.append(cand_p.resolve())
            continue
        if cand_p.is_file():
            full_files.add(cand_p.resolve())
            continue
        if any(ch in raw_t for ch in "*?["):
            glob_pats.append(raw_t.removesuffix(".py"))
            continue

        norm, mod_name, src_file, _troot = _resolve_target_spec(raw_t, resolved_start, top_level_dir)
        if src_file is not None and (
            src_file.resolve().is_relative_to(resolved_start) or src_file.resolve().is_relative_to(resolved_root)
        ):
            r_src = src_file.resolve()
            if r_src.name == "__init__.py":
                matched_dirs.append(r_src.parent)
            elif mod_name is not None and norm != mod_name:
                suffix = norm[len(mod_name) + 1 :]
                partial_specs_by_file.setdefault(r_src, []).append((norm, suffix))
            else:
                full_files.add(r_src)
            continue

        found_dir = False
        for base_d, rel_parts in (
            (resolved_start, raw_t.removeprefix("tests.").split(".")),
            (resolved_root, raw_t.split(".")),
        ):
            if all(part and part not in {".", ".."} for part in rel_parts):
                d_cand = (base_d / Path(*rel_parts)).resolve()
                if d_cand.is_dir():
                    matched_dirs.append(d_cand)
                    found_dir = True
                    break
        if not found_dir:
            glob_pats.append(raw_t.removesuffix(".py"))

    ast_cache: dict[Path, dict[str, set[str]]] = {}
    narrowed_paths: list[Path] = []
    effective_targets: list[str] = []
    partial_target_modules: set[str] = set()

    for p in plan.selected_tests:
        rp = p.resolve()
        cls = plan.classifications[p]
        is_full = (
            rp in full_files
            or any(rp.is_relative_to(d) for d in matched_dirs)
            or any(
                fnmatch.fnmatch(cls.module_name, pat)
                or fnmatch.fnmatch(cls.module_name.removeprefix("tests."), pat)
                or fnmatch.fnmatch(f"tests.{cls.module_name}", pat)
                or fnmatch.fnmatch(p.stem, pat)
                for pat in glob_pats
            )
        )
        partials = partial_specs_by_file.get(rp, [])
        if raw_targets and not is_full and not partials:
            continue

        need_case_filter = bool((partials and not is_full) or (explain_mode and name_pats is not None))
        if need_case_filter:
            cases = _enumerate_ast_test_cases(rp, cls.module_name, resolved_start, ast_cache)
            if partials and not is_full:
                cases = [
                    fn
                    for fn in cases
                    if any(
                        fn == f"{cls.module_name}.{suf}" or fn.startswith(f"{cls.module_name}.{suf}.")
                        for _, suf in partials
                    )
                ]
            if explain_mode and name_pats is not None:
                cases = [fn for fn in cases if any(fnmatch.fnmatchcase(fn, pat) for pat in name_pats)]
            if not cases:
                continue

        narrowed_paths.append(p)
        if is_full or not raw_targets:
            effective_targets.append(str(rp))
        else:
            partial_target_modules.add(cls.module_name)
            for norm_spec, _ in partials:
                if norm_spec not in effective_targets:
                    effective_targets.append(norm_spec)

    narrowed_plan = SelectionPlan(
        selected_tests=tuple(narrowed_paths),
        reasons=plan.reasons,
        classifications=plan.classifications,
        changed_files=plan.changed_files,
        run_js_web=plan.run_js_web if not (raw_targets or (explain_mode and name_pats is not None)) else False,
        run_js_plugin=plan.run_js_plugin if not (raw_targets or (explain_mode and name_pats is not None)) else False,
        damper_skipped_count=plan.damper_skipped_count,
        scan_elapsed_s=plan.scan_elapsed_s,
    )
    return narrowed_plan, effective_targets, partial_target_modules


def _run_delegated_js_checks(plan: SelectionPlan, project_root: Path, *, quiet: bool) -> list[tuple[str, str]]:
    failures: list[tuple[str, str]] = []
    commands: list[tuple[str, list[str]]] = []
    if plan.run_js_web and (project_root / "web" / "check.mjs").is_file():
        commands.append(("js.web", ["node", "web/check.mjs"]))
    if plan.run_js_plugin and (project_root / "agent-plugin" / "package.json").is_file():
        commands.append(("js.agent_plugin", ["npm", "--prefix", "agent-plugin", "test"]))
    for label, cmd in commands:
        if not quiet:
            print(f"[js] running: {' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True)
        if proc.returncode != 0:
            output = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
            failures.append((label, f"Command {' '.join(cmd)} exited with {proc.returncode}\n{output[-4000:]}"))
    return failures


@dataclass(frozen=True)
class WorkUnit:
    unit_id: int
    label: str
    targets: tuple[str, ...]
    expected_count: int
    weight: float
    module_name: str = ""
    import_root: str | None = None
    expected_file: str | None = None


@dataclass(frozen=True)
class _ModuleMeta:
    expected_file: str | None
    import_root: str | None
    has_module_fixtures: bool


def default_jobs() -> int:
    raw = os.environ.get("NI_RUA_TEST_JOBS", "").strip()
    if raw:
        value = int(raw)
        if value < 1:
            raise ValueError(f"NI_RUA_TEST_JOBS 必须为正整数，当前值: {raw!r}")
        return value
    return min(max(1, os.cpu_count() or 4), DEFAULT_MAX_JOBS)


def configure_sys_path(start_dir: Path, top_level_dir: Path | None) -> Path:
    """对齐 unittest discover 的导入搜索路径并返回模块根路径。"""
    base_top = (top_level_dir or start_dir).resolve()
    candidates = [base_top]
    project_root = _resolve_project_root(start_dir)
    has_local_ni_rua = project_root != base_top and project_root != REPO_ROOT and (project_root / "ni_rua").is_dir()
    if has_local_ni_rua:
        candidates.append(project_root)
    tests_root = (REPO_ROOT / "tests").resolve()
    if tests_root != base_top and start_dir.resolve().is_relative_to(tests_root):
        candidates.append(tests_root)
    if REPO_ROOT != base_top:
        candidates.append(REPO_ROOT)
    for entry in reversed(candidates):
        text = str(entry)
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)
    if has_local_ni_rua:
        _evict_shadowed_by_root(project_root)
    return base_top


_PR_SET_PDEATHSIG = 1
_PR_SET_CHILD_SUBREAPER = 36
_SUBREAPER_PID: int | None = None
_SUBREAPER_BASELINE_CHILDREN: set[int] = set()


_ORIG_POPEN = subprocess.Popen


def _ps_process_table_with_pgid() -> list[tuple[int, int, int, int]]:
    """无 /proc 的 POSIX 宿主机（如 macOS）回退：通过 ps 获取 (pid, ppid, pgid, uid) 快照。"""
    if os.name != "posix":
        return []
    ps_bin = "/bin/ps" if Path("/bin/ps").is_file() else shutil.which("ps")
    if not ps_bin:
        return []
    try:
        with _ORIG_POPEN(
            [ps_bin, "-ax", "-o", "pid=,ppid=,pgid=,uid="],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ) as proc:
            out, _ = proc.communicate(timeout=2.0)
    except (OSError, subprocess.SubprocessError):
        return []
    rows: list[tuple[int, int, int, int]] = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 4 and all(part.isdigit() for part in parts[:4]):
            rows.append((int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])))
    return rows


def _ps_process_table() -> list[tuple[int, int, int]]:
    """无 /proc 的 POSIX 宿主机（如 macOS）回退：通过 ps 获取 (pid, ppid, uid) 快照。"""
    return [(pid, ppid, uid) for pid, ppid, _pgid, uid in _ps_process_table_with_pgid()]


def _direct_children_of(parent_pid: int) -> set[int]:
    """读取 /proc/<pid>/task/*/children（无 /proc 时回退 ps）获取全部直接子进程。"""
    out: set[int] = set()
    task_dir = Path(f"/proc/{parent_pid}/task")
    if task_dir.is_dir():
        try:
            for entry in os.scandir(task_dir):
                try:
                    raw = Path(entry.path, "children").read_text(encoding="utf-8")
                except OSError:
                    continue
                for tok in raw.split():
                    if tok.isdigit():
                        out.add(int(tok))
            return out
        except OSError:
            pass
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return {pid for pid, ppid, _ in _ps_process_table() if ppid == parent_pid}
    try:
        for entry in os.scandir(proc_root):
            if not entry.name.isdigit():
                continue
            try:
                stat_bytes = Path(f"/proc/{entry.name}/stat").read_bytes()
                rparen = stat_bytes.rfind(b")")
                if rparen != -1 and int(stat_bytes[rparen + 2 :].split()[1]) == parent_pid:
                    out.add(int(entry.name))
            except OSError:
                continue
    except OSError:
        pass
    return out


def _ensure_subreaper() -> None:
    """将当前进程设为 Linux child subreaper，使已退出子进程的脱离会话孤儿归属到当前进程。"""
    global _SUBREAPER_PID, _SUBREAPER_BASELINE_CHILDREN
    my_pid = os.getpid()
    if _SUBREAPER_PID == my_pid or sys.platform != "linux":
        return
    baseline = _direct_children_of(my_pid)
    try:
        rc = ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)
    except OSError:
        return
    if rc == 0:
        _SUBREAPER_PID = my_pid
        _SUBREAPER_BASELINE_CHILDREN = baseline


@contextlib.contextmanager
def _block_termination_signals():
    """在进程孵化与清理临界区内阻塞 SIGINT/SIGTERM，防止中途异步打断造成泄漏。"""
    if hasattr(signal, "pthread_sigmask"):
        old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
        try:
            yield
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
    else:
        prev_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
        prev_term = signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            yield
        finally:
            signal.signal(signal.SIGINT, prev_int)
            signal.signal(signal.SIGTERM, prev_term)


def _apply_extra_sys_path(extra_roots: Sequence[str], base_paths: Sequence[str]) -> list[str]:
    """统一父子进程的附加导入根优先级顺序。"""
    if not extra_roots:
        return list(base_paths)
    seen: set[str] = set()
    ordered: list[str] = []
    for root in extra_roots:
        if root not in seen:
            seen.add(root)
            ordered.append(root)
    for entry in base_paths:
        if entry not in seen:
            ordered.append(entry)
    return ordered


def _resolve_external_import_root(path: Path) -> Path:
    """定位仓库外测试文件或目录对应的顶层可导入根路径。"""
    pkg_root = path if path.is_dir() else path.parent
    while (pkg_root / "__init__.py").is_file() and pkg_root.parent != pkg_root:
        pkg_root = pkg_root.parent
    return pkg_root


def _file_to_module_name(file_path: Path, import_root: Path) -> str:
    rel = file_path.resolve().relative_to(import_root.resolve())
    parts = rel.parent.parts if rel.name == "__init__.py" else rel.with_suffix("").parts
    return ".".join(parts)


def _import_root_for_module_file(mod_file: Path, mod_name: str) -> Path:
    resolved = mod_file.resolve()
    steps = len(mod_name.split(".")) + (1 if resolved.name == "__init__.py" else 0)
    cur = resolved
    for _ in range(steps):
        cur = cur.parent
    return cur


def _needs_repo_pkg_prefix(file_path: Path, base_top: Path) -> bool:
    """`tests/packaging/` 为避免遮蔽 PyPI `packaging` 不放 `__init__.py`，其下模块以 `tests.packaging.*` 从仓库根导入。"""
    resolved_top = base_top.resolve()
    if resolved_top.name != "tests" or (resolved_top / "packaging" / "__init__.py").is_file():
        return False
    try:
        rel = file_path.resolve().relative_to(resolved_top)
    except ValueError:
        return False
    return len(rel.parts) >= 2 and rel.parts[0] == "packaging"


def _resolve_target_spec(
    target: str,
    base_top: Path,
    top_level_dir: Path | None,
) -> tuple[str, str | None, Path | None, Path | None]:
    if target.endswith(".py") or "/" in target or os.sep in target:
        candidate = Path(target).resolve()
        if candidate.is_relative_to(base_top):
            imp_root = base_top.parent.resolve() if _needs_repo_pkg_prefix(candidate, base_top) else base_top.resolve()
            mod = _file_to_module_name(candidate, imp_root)
            return mod, mod, (candidate if candidate.is_file() else None), imp_root
        if candidate.is_relative_to(REPO_ROOT):
            mod = _file_to_module_name(candidate, REPO_ROOT)
            return mod, mod, (candidate if candidate.is_file() else None), REPO_ROOT.resolve()
        if candidate.is_file():
            pkg_root = _resolve_external_import_root(candidate).resolve()
            mod = _file_to_module_name(candidate, pkg_root)
            return mod, mod, candidate, pkg_root
        return target, None, None, None

    if top_level_dir is None and base_top.name == "tests" and target.startswith("tests."):
        if not (base_top / "tests").exists():
            stripped = target.removeprefix("tests.")
            parts = stripped.split(".")
            for k in range(len(parts), 0, -1):
                py_cand = (base_top / Path(*parts[:k])).with_suffix(".py")
                if py_cand.is_file():
                    if _needs_repo_pkg_prefix(py_cand, base_top):
                        return target, f"tests.{'.'.join(parts[:k])}", py_cand.resolve(), base_top.parent.resolve()
                    return stripped, ".".join(parts[:k]), py_cand.resolve(), base_top.resolve()
                pkg_init = base_top / Path(*parts[:k]) / "__init__.py"
                if pkg_init.is_file():
                    if _needs_repo_pkg_prefix(pkg_init, base_top):
                        return target, f"tests.{'.'.join(parts[:k])}", pkg_init.resolve(), base_top.parent.resolve()
                    return stripped, ".".join(parts[:k]), pkg_init.resolve(), base_top.resolve()
            return stripped, None, None, base_top.resolve()

    parts = target.split(".")
    for root in (base_top.resolve(), REPO_ROOT.resolve()):
        for k in range(len(parts), 0, -1):
            py_cand = (root / Path(*parts[:k])).with_suffix(".py")
            if py_cand.is_file():
                if root == base_top.resolve() and _needs_repo_pkg_prefix(py_cand, base_top):
                    return f"tests.{target}", f"tests.{'.'.join(parts[:k])}", py_cand.resolve(), base_top.parent.resolve()
                return target, ".".join(parts[:k]), py_cand.resolve(), root
            pkg_init = root / Path(*parts[:k]) / "__init__.py"
            if pkg_init.is_file():
                if root == base_top.resolve() and _needs_repo_pkg_prefix(pkg_init, base_top):
                    return f"tests.{target}", f"tests.{'.'.join(parts[:k])}", pkg_init.resolve(), base_top.parent.resolve()
                return target, ".".join(parts[:k]), pkg_init.resolve(), root

    stem_parts = target.removeprefix("tests.").split(".") if target.startswith("tests.test_") else parts
    if stem_parts and stem_parts[0].startswith("test_") and base_top.is_dir():
        disc_root = (top_level_dir or base_top).resolve()
        matches = [
            (m_name, f_path)
            for m_name, f_path in _iter_discoverable_files(base_top.resolve(), disc_root, "test*.py")
            if f_path.stem == stem_parts[0]
        ]
        if len(matches) == 1:
            matched_mod, matched_file = matches[0]
            imp_root = (
                base_top.parent.resolve()
                if top_level_dir is None and _needs_repo_pkg_prefix(matched_file, base_top)
                else disc_root
            )
            suffix = "." + ".".join(stem_parts[1:]) if len(stem_parts) > 1 else ""
            return matched_mod + suffix, matched_mod, matched_file.resolve(), imp_root

    return target, None, None, None


def _evict_conflicting_cached_module(mod_name: str, expected_file: Path) -> None:
    expected = expected_file.resolve()
    expected_root = _import_root_for_module_file(expected, mod_name).resolve()
    parts = mod_name.split(".")

    cached_exact = sys.modules.get(mod_name)
    exact_file = getattr(cached_exact, "__file__", None) if cached_exact is not None else None
    if exact_file is not None:
        try:
            same_exact = Path(exact_file).resolve() == expected
        except OSError:
            same_exact = False
        if not same_exact:
            for key in [k for k in list(sys.modules) if k == mod_name or k.startswith(f"{mod_name}.")]:
                sys.modules.pop(key, None)

    for depth in range(1, len(parts)):
        anc_name = ".".join(parts[:depth])
        anc_mod = sys.modules.get(anc_name)
        if anc_mod is None:
            continue
        expected_pkg_dir = (expected_root / Path(*parts[:depth])).resolve()
        anc_file = getattr(anc_mod, "__file__", None)
        same_pkg = False
        if anc_file is not None:
            try:
                same_pkg = Path(anc_file).resolve().parent == expected_pkg_dir
            except OSError:
                same_pkg = False
        else:
            anc_paths = getattr(anc_mod, "__path__", ())
            try:
                same_pkg = any(Path(p).resolve() == expected_pkg_dir for p in anc_paths)
            except OSError:
                same_pkg = False
        if not same_pkg:
            for key in [k for k in list(sys.modules) if k == anc_name or k.startswith(f"{anc_name}.")]:
                sys.modules.pop(key, None)
            break


def _evict_modules_under_root(root: Path) -> None:
    resolved_root = root.resolve()
    for key, mod in list(sys.modules.items()):
        if mod is None:
            continue
        mod_file = getattr(mod, "__file__", None)
        if mod_file is not None:
            try:
                if Path(mod_file).resolve().is_relative_to(resolved_root):
                    sys.modules.pop(key, None)
            except OSError:
                pass
            continue
        mod_paths = getattr(mod, "__path__", ())
        try:
            if any(Path(p).resolve().is_relative_to(resolved_root) for p in mod_paths):
                sys.modules.pop(key, None)
        except OSError:
            continue


def _evict_shadowed_by_root(root: Path) -> None:
    resolved_root = root.resolve()
    for key, mod in list(sys.modules.items()):
        top = key.split(".")[0]
        if top != key or mod is None:
            continue
        has_dir_candidate = (resolved_root / top).is_dir()
        if not (
            (resolved_root / f"{top}.py").is_file()
            or (resolved_root / top / "__init__.py").is_file()
            or (has_dir_candidate and top != "packaging")
        ):
            continue
        mod_file = getattr(mod, "__file__", None)
        if mod_file is not None:
            try:
                if Path(mod_file).resolve().is_relative_to(resolved_root):
                    continue
            except OSError:
                pass
        else:
            mod_paths = getattr(mod, "__path__", ())
            try:
                if any(Path(p).resolve().is_relative_to(resolved_root) for p in mod_paths):
                    continue
            except OSError:
                pass
        for sub_key in [k for k in list(sys.modules) if k == top or k.startswith(f"{top}.")]:
            sys.modules.pop(sub_key, None)


_EXCLUDED_DISCOVERY_DIRS = frozenset(
    {"__pycache__", ".git", ".claude", "worktrees", "node_modules", ".venv", "venv"}
)


def _is_excluded_relpath(path: Path, base: Path) -> bool:
    try:
        parts = path.resolve().relative_to(base.resolve()).parts
    except ValueError:
        return False
    for idx, part in enumerate(parts):
        if part in {".git", "__pycache__", "node_modules", ".venv", "venv"}:
            return True
        if part == ".claude" and idx + 1 < len(parts) and parts[idx + 1] == "worktrees":
            return True
    return False


def _is_zone_package_dir(root_path: Path, resolved_dir: Path, subdir_name: str) -> bool:
    """支持 Wave 2 `tests/<zone>/<subzone>/` 嵌套目录（含或不含 __init__.py），排除 fixtures。"""
    if subdir_name in {"fixtures", "support"}:
        return False
    if root_path == resolved_dir and subdir_name in VALID_ZONES:
        return True
    if root_path.parent == resolved_dir and root_path.name in VALID_ZONES:
        return True
    return False


def _iter_discoverable_files(directory: Path, top_dir: Path, pattern: str) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    resolved_dir = directory.resolve()
    resolved_top = top_dir.resolve()
    for root, dirs, files in os.walk(resolved_dir):
        root_path = Path(root)
        if _is_excluded_relpath(root_path, resolved_dir):
            dirs[:] = []
            continue
        is_zone_dir = (
            (root_path.parent == resolved_dir and root_path.name in VALID_ZONES)
            or (root_path.parent.parent == resolved_dir and root_path.parent.name in VALID_ZONES and root_path.name not in {"fixtures", "support"})
        )
        if root_path != resolved_dir and not (root_path / "__init__.py").is_file() and not is_zone_dir:
            dirs[:] = []
            continue
        dirs[:] = sorted(
            d
            for d in dirs
            if d not in _EXCLUDED_DISCOVERY_DIRS
            and ((root_path / d / "__init__.py").is_file() or _is_zone_package_dir(root_path, resolved_dir, d))
        )
        for fname in sorted(files):
            if fnmatch.fnmatch(fname, pattern):
                fpath = (root_path / fname).resolve()
                imp_root = resolved_top.parent if _needs_repo_pkg_prefix(fpath, resolved_top) else resolved_top
                out.append((_file_to_module_name(fpath, imp_root), fpath))
    return out


def _capture_case_module_meta(suite: unittest.TestSuite, dest: dict[str, _ModuleMeta]) -> None:
    for case in _iter_cases(suite):
        mod_name = case.__class__.__module__
        if mod_name == "unittest.loader" or mod_name in dest:
            continue
        mod_obj = sys.modules.get(mod_name)
        raw_file = getattr(mod_obj, "__file__", None) if mod_obj is not None else None
        exp_file = str(Path(raw_file).resolve()) if raw_file else None
        imp_root = str(_import_root_for_module_file(Path(exp_file), mod_name)) if exp_file else None
        dest[mod_name] = _ModuleMeta(
            expected_file=exp_file,
            import_root=imp_root,
            has_module_fixtures=_has_module_fixtures(mod_name),
        )


def load_targets(
    loader: unittest.TestLoader,
    targets: list[str],
    base_top: Path,
    top_level_dir: Path | None,
    pattern: str,
) -> tuple[unittest.TestSuite, dict[str, _ModuleMeta]]:
    """按目标各自独立的导入根加载测试，固化来源元数据并拒绝同名冲突。"""
    suite = unittest.TestSuite()
    module_meta: dict[str, _ModuleMeta] = {}
    base_sys_path = sys.path[:]
    default_roots = {base_top.resolve(), REPO_ROOT.resolve(), (REPO_ROOT / "tests").resolve()}
    last_root = base_top.resolve()
    seen_modules: dict[str, Path] = {}

    def _register_module(mod_name: str, file_path: Path) -> bool:
        resolved = file_path.resolve()
        prev = seen_modules.get(mod_name)
        if prev is not None and prev != resolved:
            err = ImportError(
                f"测试目标模块名冲突: {mod_name!r} 同时指向不同文件 {str(prev)!r} 与 {str(resolved)!r}"
            )
            suite.addTest(unittest.loader._FailedTest(mod_name, err))  # type: ignore[attr-defined]
            return False
        seen_modules[mod_name] = resolved
        return True

    def _switch_discovery_root(active_root: Path | None) -> None:
        nonlocal last_root
        if active_root is None:
            return
        resolved_active = active_root.resolve()
        if resolved_active != last_root:
            if last_root not in default_roots:
                _evict_modules_under_root(last_root)
            _evict_shadowed_by_root(resolved_active)
            last_root = resolved_active

    expanded_targets: list[str] = []
    discovered_cache: list[tuple[str, Path]] | None = None
    for raw_target in targets:
        if any(ch in raw_target for ch in "*?[") and not Path(raw_target).exists():
            if discovered_cache is None:
                discovered_cache = _iter_discoverable_files(base_top, base_top, pattern)
            pat = raw_target.removesuffix(".py")
            matched = [
                mod
                for mod, _ in discovered_cache
                if (
                    fnmatch.fnmatch(mod, pat)
                    or fnmatch.fnmatch(mod.removeprefix("tests."), pat)
                    or fnmatch.fnmatch(mod.rsplit(".", 1)[-1], pat)
                )
            ]
            if matched:
                expanded_targets.extend(matched)
            else:
                err = ImportError(f"no test module matches pattern: {raw_target!r}")
                suite.addTest(unittest.loader._FailedTest(raw_target, err))  # type: ignore[attr-defined]
        else:
            expanded_targets.append(raw_target)

    for target in expanded_targets:
        candidate = Path(target)
        if candidate.is_dir():
            resolved = candidate.resolve()
            if top_level_dir is not None:
                disc_top = top_level_dir.resolve()
            elif resolved.is_relative_to(base_top):
                disc_top = base_top.resolve()
            elif resolved.is_relative_to(REPO_ROOT):
                disc_top = REPO_ROOT.resolve()
            else:
                disc_top = _resolve_external_import_root(resolved).resolve()

            dir_discoverable = _iter_discoverable_files(resolved, disc_top, pattern)
            has_conflict = False
            for mod_name, fpath in dir_discoverable:
                if not _register_module(mod_name, fpath):
                    has_conflict = True
                else:
                    _evict_conflicting_cached_module(mod_name, fpath)
            if has_conflict:
                continue

            if dir_discoverable and all(_needs_repo_pkg_prefix(fp, disc_top) for _, fp in dir_discoverable):
                extra_sub, extra_meta = load_targets(
                    loader, [str(fp) for _, fp in dir_discoverable], base_top, top_level_dir, pattern
                )
                suite.addTests(extra_sub)
                module_meta.update(extra_meta)
                continue

            _switch_discovery_root(disc_top)
            try:
                sys.path[:] = _apply_extra_sys_path([str(disc_top)], base_sys_path)
                sub = loader.discover(str(resolved), pattern=pattern, top_level_dir=str(disc_top))
            finally:
                sys.path[:] = base_sys_path

            conflict_after = False
            discovered_paths: set[Path] = set()
            for case in _iter_cases(sub):
                mod = sys.modules.get(case.__class__.__module__)
                mod_file = getattr(mod, "__file__", None) if mod is not None else None
                if mod_file and case.__class__.__module__ != "unittest.loader":
                    r_mod_file = Path(mod_file).resolve()
                    discovered_paths.add(r_mod_file)
                    if not _register_module(case.__class__.__module__, r_mod_file):
                        conflict_after = True
            if not conflict_after:
                _capture_case_module_meta(sub, module_meta)
                suite.addTests(sub)
                missing_files = [str(fp) for _, fp in dir_discoverable if fp.resolve() not in discovered_paths]
                if missing_files:
                    extra_sub, extra_meta = load_targets(loader, missing_files, base_top, top_level_dir, pattern)
                    suite.addTests(extra_sub)
                    module_meta.update(extra_meta)
        else:
            norm, mod_name, src_file, target_root = _resolve_target_spec(target, base_top, top_level_dir)
            if src_file is not None and src_file.name == "__init__.py" and norm == mod_name:
                extra_sub, extra_meta = load_targets(
                    loader, [str(src_file.parent)], base_top, top_level_dir, pattern
                )
                suite.addTests(extra_sub)
                module_meta.update(extra_meta)
                continue
            if src_file is None:
                matched_dir: Path | None = None
                for base_d, rel_parts in (
                    (base_top.resolve(), target.removeprefix("tests.").split(".")),
                    (REPO_ROOT.resolve(), target.split(".")),
                ):
                    if all(part and part not in {".", ".."} for part in rel_parts):
                        d_cand = (base_d / Path(*rel_parts)).resolve()
                        if d_cand.is_dir():
                            matched_dir = d_cand
                            break
                if matched_dir is not None:
                    extra_sub, extra_meta = load_targets(
                        loader, [str(matched_dir)], base_top, top_level_dir, pattern
                    )
                    suite.addTests(extra_sub)
                    module_meta.update(extra_meta)
                    continue
            if mod_name is not None and src_file is not None:
                if not _register_module(mod_name, src_file):
                    continue
                _evict_conflicting_cached_module(mod_name, src_file)

            _switch_discovery_root(target_root)
            try:
                if target_root is not None:
                    sys.path[:] = _apply_extra_sys_path([str(target_root)], base_sys_path)
                try:
                    sub = loader.loadTestsFromNames([norm])
                except unittest.SkipTest as exc:
                    if hasattr(unittest.loader, "_make_skipped_test"):
                        sub = unittest.loader._make_skipped_test(norm, exc, unittest.TestSuite)  # type: ignore[attr-defined]
                    else:
                        @unittest.skip(str(exc))
                        class _SkippedModule(unittest.TestCase):
                            def test_skipped(self) -> None:
                                pass
                        sub = unittest.TestSuite([_SkippedModule("test_skipped")])
                except Exception as exc:
                    sub = unittest.TestSuite([unittest.loader._FailedTest(norm, exc)])  # type: ignore[attr-defined]
            finally:
                sys.path[:] = base_sys_path

            conflict_after = False
            if mod_name is not None and src_file is not None:
                loaded_mod = sys.modules.get(mod_name)
                loaded_file = getattr(loaded_mod, "__file__", None) if loaded_mod is not None else None
                if loaded_file is not None and Path(loaded_file).resolve() != src_file.resolve():
                    err = ImportError(
                        f"测试目标来源不匹配: 期望加载 {str(src_file.resolve())!r}，实际导入了 {str(Path(loaded_file).resolve())!r}"
                    )
                    suite.addTest(unittest.loader._FailedTest(norm, err))  # type: ignore[attr-defined]
                    conflict_after = True

            if not conflict_after:
                for case in _iter_cases(sub):
                    case_mod = case.__class__.__module__
                    if case_mod == "unittest.loader":
                        continue
                    mod = sys.modules.get(case_mod)
                    mod_file = getattr(mod, "__file__", None) if mod is not None else None
                    if mod_file is None:
                        continue
                    if not _register_module(case_mod, Path(mod_file).resolve()):
                        conflict_after = True
            if not conflict_after:
                _capture_case_module_meta(sub, module_meta)
                suite.addTests(sub)
    return suite, module_meta


def _iter_cases(suite: unittest.TestSuite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _iter_cases(item)
        else:
            yield item


def _has_module_fixtures(module_name: str) -> bool:
    mod = sys.modules.get(module_name)
    return mod is not None and (hasattr(mod, "setUpModule") or hasattr(mod, "tearDownModule"))


def _has_class_fixtures(cls: type[unittest.TestCase]) -> bool:
    setup_fn = getattr(cls.setUpClass, "__func__", cls.setUpClass)
    teardown_fn = getattr(cls.tearDownClass, "__func__", cls.tearDownClass)
    return (
        setup_fn is not unittest.TestCase.setUpClass.__func__
        or teardown_fn is not unittest.TestCase.tearDownClass.__func__
    )


def build_work_units(
    suite: unittest.TestSuite,
    *,
    parallel: bool,
    module_meta: dict[str, _ModuleMeta] | None = None,
    module_timings: dict[str, float] | None = None,
) -> tuple[list[unittest.TestCase], list[WorkUnit]]:
    """拆分测试套件为可并行调度的最小安全执行单元，并绑定每个单元精确来源文件。"""
    resolved_meta: dict[str, _ModuleMeta] = dict(module_meta) if module_meta is not None else {}
    _capture_case_module_meta(suite, resolved_meta)
    timings = module_timings or {}

    loader_failures: list[unittest.TestCase] = []
    by_module: dict[str, list[unittest.TestCase]] = {}
    for case in _iter_cases(suite):
        mod_name = case.__class__.__module__
        if mod_name == "unittest.loader":
            loader_failures.append(case)
        else:
            by_module.setdefault(mod_name, []).append(case)

    units: list[WorkUnit] = []
    next_id = 1
    for mod_name, mod_cases in by_module.items():
        weight, chunk_size = _MODULE_PROFILE.get(mod_name, (1.0, 4))
        timed_s = timings.get(mod_name)
        per_case_s = (timed_s / max(1, len(mod_cases))) if timed_s is not None else None
        meta = resolved_meta.get(mod_name)
        exp_file = meta.expected_file if meta is not None else None
        imp_root = meta.import_root if meta is not None else None
        has_mod_fix = meta.has_module_fixtures if meta is not None else _has_module_fixtures(mod_name)

        if not parallel or has_mod_fix:
            mod_ids = tuple(case.id() for case in mod_cases)
            unit_weight = max(0.05, timed_s) if timed_s is not None else weight * max(1, len(mod_cases))
            units.append(
                WorkUnit(
                    unit_id=next_id,
                    label=mod_ids[0] if len(mod_ids) == 1 else mod_name,
                    targets=mod_ids,
                    expected_count=len(mod_cases),
                    weight=unit_weight,
                    module_name=mod_name,
                    import_root=imp_root,
                    expected_file=exp_file,
                )
            )
            next_id += 1
            continue

        by_class: dict[type[unittest.TestCase], list[unittest.TestCase]] = {}
        for case in mod_cases:
            by_class.setdefault(case.__class__, []).append(case)

        for cls, cls_cases in by_class.items():
            cls_target = f"{mod_name}.{cls.__qualname__}"
            ids = [case.id() for case in cls_cases]
            if _has_class_fixtures(cls):
                unit_weight = (
                    max(0.05, per_case_s * len(cls_cases))
                    if per_case_s is not None
                    else weight * max(1, min(len(cls_cases), 4))
                )
                units.append(
                    WorkUnit(
                        unit_id=next_id,
                        label=ids[0] if len(ids) == 1 else cls_target,
                        targets=tuple(ids),
                        expected_count=len(cls_cases),
                        weight=unit_weight,
                        module_name=mod_name,
                        import_root=imp_root,
                        expected_file=exp_file,
                    )
                )
                next_id += 1
                continue

            for idx in range(0, len(ids), chunk_size):
                chunk = tuple(ids[idx : idx + chunk_size])
                label = chunk[0] if len(chunk) == 1 else f"{cls_target}[{idx + 1}..{idx + len(chunk)}]"
                unit_weight = max(0.05, per_case_s * len(chunk)) if per_case_s is not None else weight
                units.append(
                    WorkUnit(
                        unit_id=next_id,
                        label=label,
                        targets=chunk,
                        expected_count=len(chunk),
                        weight=unit_weight,
                        module_name=mod_name,
                        import_root=imp_root,
                        expected_file=exp_file,
                    )
                )
                next_id += 1

    if parallel:
        units.sort(key=lambda u: (u.weight, u.expected_count), reverse=True)
    return loader_failures, units


class _CaptureResult(unittest.TestResult):
    """捕获单分片内全部 unittest 结果状态与逐项耗时。"""

    def __init__(self) -> None:
        super().__init__()
        self.buffer = True
        self.test_timings: list[list[object]] = []
        self._started_at = 0.0

    def startTest(self, test: unittest.TestCase) -> None:
        self._started_at = time.perf_counter()
        super().startTest(test)

    def stopTest(self, test: unittest.TestCase) -> None:
        elapsed = time.perf_counter() - self._started_at
        self.test_timings.append([test.id(), elapsed])
        super().stopTest(test)


def _proc_descendants_of(root_pid: int) -> set[int]:
    """扫描 /proc 获取指定 PID 的全部传递子孙进程（不含 root_pid 自身）。"""
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return set()
    by_ppid: dict[int, list[int]] = {}
    my_uid = os.getuid()
    try:
        entries = list(os.scandir(proc_root))
    except OSError:
        return set()
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            if entry.stat(follow_symlinks=False).st_uid != my_uid:
                continue
            stat_bytes = Path(f"/proc/{pid}/stat").read_bytes()
            rparen = stat_bytes.rfind(b")")
            if rparen == -1:
                continue
            fields = stat_bytes[rparen + 2 :].split()
            if len(fields) >= 2:
                by_ppid.setdefault(int(fields[1]), []).append(pid)
        except OSError:
            continue
    descendants: set[int] = set()
    stack = [root_pid]
    while stack:
        cur = stack.pop()
        for child in by_ppid.get(cur, ()):
            if child != root_pid and child not in descendants:
                descendants.add(child)
                stack.append(child)
    return descendants


def _reap_subreaper_tree() -> None:
    """在 Worker 守护父进程内收敛终止并回收全部残留子孙与并发再派生的孤儿进程。"""
    my_pid = os.getpid()
    desc = (_proc_descendants_of(my_pid) | _direct_children_of(my_pid)) - {my_pid}
    if desc:
        _signal_owned_pids(desc, signal.SIGTERM)
        deadline = time.monotonic() + 0.15
        while desc and time.monotonic() < deadline:
            for _ in range(256):
                if time.monotonic() >= deadline:
                    break
                try:
                    wpid, _ = os.waitpid(-1, os.WNOHANG)
                    if wpid == 0:
                        break
                    desc.discard(wpid)
                except ChildProcessError:
                    desc.clear()
                    break
            desc = {pid for pid in desc if _pid_alive(pid)}
            if desc and time.monotonic() < deadline:
                time.sleep(0.01)

    kill_deadline = time.monotonic() + 2.0
    while time.monotonic() < kill_deadline:
        live = (_proc_descendants_of(my_pid) | _direct_children_of(my_pid)) - {my_pid}
        if live:
            _signal_owned_pids(live, signal.SIGKILL)
        no_children = False
        for _ in range(256):
            if time.monotonic() >= kill_deadline:
                break
            try:
                wpid, _ = os.waitpid(-1, os.WNOHANG)
                if wpid == 0:
                    break
            except ChildProcessError:
                no_children = True
                break
        if no_children and not ((_proc_descendants_of(my_pid) | _direct_children_of(my_pid)) - {my_pid}):
            break
        if time.monotonic() < kill_deadline:
            time.sleep(0.005)


def _worker_exec_loop(fd: int, start_dir: Path, top_level_dir: Path | None) -> int:
    def _worker_sigterm(signum: int, _frame: object) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _worker_sigterm)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    faulthandler.enable(all_threads=True)
    _ensure_subreaper()
    tempfile.tempdir = os.environ.get("TMPDIR") or tempfile.gettempdir()
    base_top = configure_sys_path(start_dir, top_level_dir).resolve()
    default_roots = {base_top, REPO_ROOT.resolve(), (REPO_ROOT / "tests").resolve()}
    last_applied_root: Path = base_top

    active_fd = [fd]
    if hasattr(os, "register_at_fork"):
        def _close_inherited_fd_in_child() -> None:
            cur = active_fd[0]
            if cur >= 0:
                active_fd[0] = -1
                try:
                    os.close(cur)
                except OSError:
                    pass

        os.register_at_fork(after_in_child=_close_inherited_fd_in_child)

    try:
        with socket.socket(fileno=fd) as sock, sock.makefile("rwb") as channel:
            channel.write(b'{"event":"ready"}\n')
            channel.flush()
            while True:
                line = channel.readline()
                if not line:
                    return 0
                message = json.loads(line)
                if message.get("op") == "stop":
                    return 0

                channel.write((json.dumps({"event": "started", "unit_id": message["unit_id"]}) + "\n").encode("utf-8"))
                channel.flush()

                targets: list[str] = message["targets"]
                patterns: list[str] | None = message.get("patterns")
                module_name: str = message.get("module_name") or ""
                import_root: str | None = message.get("import_root")
                expected_file: str | None = message.get("expected_file")
                failfast: bool = bool(message.get("failfast"))
                saved_cwd = os.getcwd()
                saved_env = os.environ.copy()
                saved_sys_path = sys.path[:]
                t0 = time.perf_counter()
                try:
                    cur_root = Path(import_root).resolve() if import_root else None
                    if cur_root is not None:
                        cur_root_str = str(cur_root)
                        if not saved_sys_path or saved_sys_path[0] != cur_root_str:
                            sys.path[:] = _apply_extra_sys_path([cur_root_str], saved_sys_path)
                        if cur_root != last_applied_root:
                            if last_applied_root not in default_roots:
                                _evict_modules_under_root(last_applied_root)
                            _evict_shadowed_by_root(cur_root)
                            last_applied_root = cur_root
                    if module_name and expected_file:
                        _evict_conflicting_cached_module(module_name, Path(expected_file))
                    loader = unittest.TestLoader()
                    if patterns:
                        loader.testNamePatterns = patterns
                    suite = loader.loadTestsFromNames(targets)
                    if module_name and expected_file:
                        expected_resolved = Path(expected_file).resolve()
                        loaded_mod = sys.modules.get(module_name)
                        if loaded_mod is not None:
                            loaded_file = getattr(loaded_mod, "__file__", None)
                            actual_resolved = Path(loaded_file).resolve() if loaded_file else None
                            if actual_resolved != expected_resolved:
                                err = ImportError(
                                    f"Worker 模块来源不匹配: 期望 {str(expected_resolved)!r}，实际加载 {str(actual_resolved)!r}"
                                )
                                suite = unittest.TestSuite(
                                    [unittest.loader._FailedTest(module_name, err)]  # type: ignore[attr-defined]
                                )
                        elif any(c.__class__.__module__ != "unittest.loader" for c in _iter_cases(suite)):
                            err = ImportError(f"Worker 未能定位模块 {module_name!r} ({str(expected_resolved)!r})")
                            suite = unittest.TestSuite(
                                [unittest.loader._FailedTest(module_name, err)]  # type: ignore[attr-defined]
                            )
                    result = _CaptureResult()
                    result.failfast = failfast
                    unit_timeout = float(message.get("timeout") or 0.0)
                    if unit_timeout > 0:
                        lead = min(10.0, unit_timeout / 10.0)
                        faulthandler.dump_traceback_later(max(1.0, unit_timeout - lead), exit=False)
                    try:
                        suite.run(result)
                    finally:
                        if unit_timeout > 0:
                            faulthandler.cancel_dump_traceback_later()
                finally:
                    if os.getcwd() != saved_cwd:
                        os.chdir(saved_cwd)
                    sys.path[:] = saved_sys_path
                    os.environ.clear()
                    os.environ.update(saved_env)
                    tempfile.tempdir = os.environ.get("TMPDIR") or tempfile.gettempdir()
                elapsed = time.perf_counter() - t0

                payload = {
                    "event": "done",
                    "unit_id": message["unit_id"],
                    "elapsed_s": elapsed,
                    "tests_run": result.testsRun,
                    "skipped": [[t.id(), reason] for t, reason in result.skipped],
                    "failures": [[t.id(), tb] for t, tb in result.failures],
                    "errors": [[t.id(), tb] for t, tb in result.errors],
                    "expected_failures": [[t.id(), tb] for t, tb in result.expectedFailures],
                    "unexpected_successes": [t.id() for t in result.unexpectedSuccesses],
                    "test_timings": result.test_timings,
                }
                channel.write((json.dumps(payload) + "\n").encode("utf-8"))
                channel.flush()
    finally:
        active_fd[0] = -1


def _worker_main(start_dir: Path, top_level_dir: Path | None) -> int:
    if hasattr(signal, "pthread_sigmask"):
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT, signal.SIGTERM})
    fd = int(os.environ["NI_RUA_TEST_WORKER_FD"])
    os.set_inheritable(fd, False)

    if sys.platform == "linux" and hasattr(os, "fork"):
        _ensure_subreaper()
        runner_pid_raw = os.environ.get("NI_RUA_RUNNER_PID", "").strip()
        runner_pid = int(runner_pid_raw) if runner_pid_raw.isdigit() else 0
        own_tmpdir = os.environ.get("NI_RUA_WORKER_OWNERSHIP", "")
        child_pid = os.fork()
        if child_pid > 0:
            os.close(fd)

            def _force_kill_child(_signum: int, _frame: object) -> None:
                try:
                    os.kill(child_pid, signal.SIGKILL)
                except OSError:
                    pass

            def _forward_term(_signum: int, _frame: object) -> None:
                try:
                    os.kill(child_pid, signal.SIGTERM)
                except OSError:
                    pass
                try:
                    signal.setitimer(signal.ITIMER_REAL, 0.4)
                except OSError:
                    pass

            signal.signal(signal.SIGALRM, _force_kill_child)
            signal.signal(signal.SIGTERM, _forward_term)
            signal.signal(signal.SIGINT, _forward_term)
            if runner_pid > 0:
                try:
                    ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
                except OSError:
                    pass
                if os.getppid() != runner_pid:
                    _forward_term(signal.SIGTERM, None)
            while True:
                try:
                    _, status = os.waitpid(child_pid, 0)
                    break
                except InterruptedError:
                    continue
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            _reap_subreaper_tree()
            if runner_pid > 0 and os.getppid() != runner_pid and own_tmpdir:
                shutil.rmtree(own_tmpdir, ignore_errors=True)
            code = os.waitstatus_to_exitcode(status)
            os._exit(code if code >= 0 else 128 + (-code))

    return _worker_exec_loop(fd, start_dir, top_level_dir)


class _WorkerHandle:
    def __init__(
        self,
        worker_id: int,
        process: subprocess.Popen,
        sock: socket.socket,
        tmpdir: Path,
        log_path: Path,
    ) -> None:
        self.worker_id = worker_id
        self.process = process
        self.sock = sock
        self.sock.setblocking(False)
        self._rx_buf = bytearray()
        self.eof = False
        self.tmpdir = tmpdir
        self.log_path = log_path
        self.dispatched_unit: WorkUnit | None = None
        self.in_flight: WorkUnit | None = None
        self.last_unit: WorkUnit | None = None
        self.unit_started_at: float | None = None
        self.stop_sent: bool = False

    def send(self, payload: dict) -> None:
        data = (json.dumps(payload) + "\n").encode("utf-8")
        view = memoryview(data)
        while view:
            if self.eof or self.process.poll() is not None:
                raise BrokenPipeError("Worker 进程已退出")
            try:
                sent = self.sock.send(view)
                view = view[sent:]
            except BlockingIOError:
                with selectors.DefaultSelector() as wsel:
                    wsel.register(self.sock, selectors.EVENT_WRITE)
                    wsel.select(timeout=0.2)

    def drain_lines(self) -> list[bytes]:
        while not self.eof:
            try:
                chunk = self.sock.recv(65536)
            except BlockingIOError:
                break
            except OSError:
                self.eof = True
                break
            if not chunk:
                self.eof = True
                break
            self._rx_buf.extend(chunk)
        lines: list[bytes] = []
        while True:
            nl = self._rx_buf.find(b"\n")
            if nl == -1:
                break
            lines.append(bytes(self._rx_buf[: nl + 1]))
            del self._rx_buf[: nl + 1]
        return lines

    def close_io(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass

    def close(self, *, reap_descendants: bool = True) -> None:
        _close_workers((self,), reap_descendants=reap_descendants)


def _signal_worker_group(proc: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(proc.pid, sig)
    except OSError:
        try:
            proc.send_signal(sig)
        except OSError:
            pass


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    if Path("/proc").is_dir():
        try:
            data = Path(f"/proc/{pid}/stat").read_bytes()
            rparen = data.rfind(b")")
            if rparen != -1 and data[rparen + 2 : rparen + 3] == b"Z":
                return False
        except OSError:
            return False
    return True


def _reap_reparented_pids(pids: Iterable[int]) -> None:
    for pid in pids:
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass
        except OSError:
            pass


def _find_owned_pids(
    handles: Sequence[_WorkerHandle],
    *,
    known_worker_pids: Iterable[int] = (),
) -> set[int]:
    """扫描 /proc 定位归属于指定 Worker 的全部子孙进程（含清空环境且脱离会话的硬崩孤儿）。"""
    proc_root = Path("/proc")
    if not handles:
        return set()
    my_pid = os.getpid()
    my_uid = os.getuid()
    root_pids = {h.process.pid for h in handles}
    all_worker_pids = set(known_worker_pids) | root_pids
    if not proc_root.is_dir():
        children_by_ppid: dict[int, list[int]] = {}
        seed_pids: set[int] = set(root_pids)
        for pid, ppid, pgid, uid in _ps_process_table_with_pgid():
            if pid != my_pid and uid == my_uid:
                children_by_ppid.setdefault(ppid, []).append(pid)
                if pgid in root_pids and pid not in all_worker_pids:
                    seed_pids.add(pid)
        owned: set[int] = set()
        queue = deque(seed_pids)
        while queue:
            cur = queue.popleft()
            if cur in owned:
                continue
            owned.add(cur)
            for child in children_by_ppid.get(cur, ()):
                if child not in owned:
                    queue.append(child)
        owned.discard(my_pid)
        owned.difference_update(all_worker_pids)
        return owned
    adopt_reparented = _SUBREAPER_PID == my_pid and any(h.process.poll() is not None for h in handles)

    raw_dirs: set[str] = set()
    for h in handles:
        raw_dirs.add(str(h.tmpdir))
        try:
            raw_dirs.add(str(h.tmpdir.resolve()))
        except OSError:
            pass
    owned_prefixes = tuple(raw_dirs)
    owned_slashes = tuple(f"{p}/" for p in owned_prefixes)
    env_needles = tuple(
        needle
        for p in owned_prefixes
        for needle in (
            b"NI_RUA_WORKER_OWNERSHIP=" + os.fsencode(p) + b"\x00",
            b"TMPDIR=" + os.fsencode(p) + b"\x00",
        )
    )

    children_by_ppid: dict[int, list[int]] = {}
    seed_pids: set[int] = set(root_pids)

    try:
        entries = list(os.scandir(proc_root))
    except OSError:
        return set()

    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == my_pid:
            continue
        try:
            if entry.stat(follow_symlinks=False).st_uid != my_uid:
                continue
            stat_bytes = Path(f"/proc/{pid}/stat").read_bytes()
            rparen = stat_bytes.rfind(b")")
            if rparen == -1:
                continue
            fields = stat_bytes[rparen + 2 :].split()
            if len(fields) < 2:
                continue
            state = fields[0]
            ppid = int(fields[1])
            children_by_ppid.setdefault(ppid, []).append(pid)
            is_reparented_orphan = (
                adopt_reparented
                and ppid == my_pid
                and pid not in all_worker_pids
                and pid not in _SUBREAPER_BASELINE_CHILDREN
            )
            if state == b"Z":
                if is_reparented_orphan:
                    _reap_reparented_pids((pid,))
                continue
            if pid in seed_pids:
                continue
            if is_reparented_orphan:
                seed_pids.add(pid)
                continue

            try:
                env_bytes = Path(f"/proc/{pid}/environ").read_bytes()
                if any(needle in env_bytes for needle in env_needles):
                    seed_pids.add(pid)
                    continue
            except OSError:
                pass

            try:
                cwd_target = os.readlink(f"/proc/{pid}/cwd")
                if cwd_target in raw_dirs or cwd_target.startswith(owned_slashes):
                    seed_pids.add(pid)
                    continue
            except OSError:
                pass

            try:
                with os.scandir(f"/proc/{pid}/fd") as fd_iter:
                    for fd_entry in fd_iter:
                        try:
                            fd_target = os.readlink(fd_entry.path)
                        except OSError:
                            continue
                        if fd_target in raw_dirs or fd_target.startswith(owned_slashes):
                            seed_pids.add(pid)
                            break
            except OSError:
                pass
        except OSError:
            continue

    owned: set[int] = set()
    queue = deque(seed_pids)
    while queue:
        cur = queue.popleft()
        if cur in owned:
            continue
        owned.add(cur)
        for child in children_by_ppid.get(cur, ()):
            if child not in owned:
                queue.append(child)
    owned.discard(my_pid)
    owned.difference_update(all_worker_pids)
    return owned


def _signal_owned_pids(pids: Iterable[int], sig: int) -> None:
    my_pid = os.getpid()
    my_pgid = os.getpgrp()
    signaled_pgids: set[int] = set()
    for pid in pids:
        if pid <= 1 or pid == my_pid:
            continue
        try:
            pgid = os.getpgid(pid)
        except OSError:
            pgid = -1
        if pgid > 1 and pgid != my_pgid and pgid not in signaled_pgids:
            signaled_pgids.add(pgid)
            try:
                os.killpg(pgid, sig)
            except OSError:
                pass
        try:
            os.kill(pid, sig)
        except OSError:
            pass


def _close_workers(
    handles: Iterable[_WorkerHandle],
    *,
    reap_descendants: bool = True,
    registry: dict[int, _WorkerHandle] | None = None,
    known_worker_pids: Iterable[int] = (),
) -> None:
    """并发终止一组 Worker 及其归属子孙进程，完成清理后才移除登记。"""
    items = list({id(h): h for h in handles}.values())
    if not items:
        return
    all_known_pids = set(known_worker_pids) | (
        {h.process.pid for h in registry.values()} if registry is not None else set()
    )
    with _block_termination_signals():
        # Snapshot descendants before signalling h.process so non-subreaper hosts (macOS)
        # capture child ppids before a fast-exiting worker reparents them to PID 1.
        owned_pids = _find_owned_pids(items, known_worker_pids=all_known_pids) if reap_descendants else set()
        alive: list[subprocess.Popen] = []
        for h in items:
            h.close_io()
            if h.process.poll() is None:
                _signal_worker_group(h.process, signal.SIGTERM)
                alive.append(h.process)

        if owned_pids:
            _signal_owned_pids(owned_pids, signal.SIGTERM)

        had_survivors = bool(alive or owned_pids)
        deadline = time.monotonic() + 0.5
        while (alive or owned_pids) and time.monotonic() < deadline:
            alive = [p for p in alive if p.poll() is None]
            if owned_pids:
                _reap_reparented_pids(owned_pids)
                owned_pids = {pid for pid in owned_pids if _pid_alive(pid)}
            if alive or owned_pids:
                time.sleep(0.01)

        for p in alive:
            _signal_worker_group(p, signal.SIGKILL)
        if owned_pids:
            _signal_owned_pids(owned_pids, signal.SIGKILL)

        for p in alive:
            try:
                p.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass

        if reap_descendants and (had_survivors or owned_pids):
            kill_deadline = time.monotonic() + 0.5
            while time.monotonic() < kill_deadline:
                newly_found = _find_owned_pids(items, known_worker_pids=all_known_pids)
                if newly_found:
                    owned_pids |= newly_found
                    _signal_owned_pids(newly_found, signal.SIGKILL)
                if owned_pids:
                    _reap_reparented_pids(owned_pids)
                    owned_pids = {pid for pid in owned_pids if _pid_alive(pid)}
                if not owned_pids:
                    break
                time.sleep(0.01)
            if owned_pids:
                _reap_reparented_pids(owned_pids)

        for h in items:
            if h.process.poll() is None:
                try:
                    os.killpg(h.process.pid, signal.SIGKILL)
                except OSError:
                    pass
            shutil.rmtree(h.tmpdir, ignore_errors=True)
            if h.tmpdir.exists():
                shutil.rmtree(h.tmpdir, ignore_errors=True)
            if registry is not None:
                registry.pop(h.worker_id, None)


def _reap_leaked_children_since(before_children: set[int]) -> None:
    """收敛终止并回收临界区内新增的全部直接子进程及被 subreaper 二次收养的后代僵尸。"""
    my_pid = os.getpid()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        direct = _direct_children_of(my_pid) - before_children
        if not direct:
            break
        targets = set(direct)
        for dpid in direct:
            targets |= _proc_descendants_of(dpid)
        _signal_owned_pids(targets, signal.SIGKILL)
        for dpid in direct:
            try:
                os.waitpid(dpid, os.WNOHANG)
            except OSError:
                pass
        if not (_direct_children_of(my_pid) - before_children):
            break
        time.sleep(0.005)


def _spawn_worker(
    worker_id: int,
    start_dir: Path,
    top_level_dir: Path | None,
    registry: dict[int, _WorkerHandle] | None = None,
) -> _WorkerHandle:
    _ensure_subreaper()
    base_tmp = "/tmp" if Path("/tmp").is_dir() else None
    tmpdir: Path | None = None
    parent_sock: socket.socket | None = None
    child_sock: socket.socket | None = None
    proc: subprocess.Popen | None = None
    handle: _WorkerHandle | None = None
    known_pids = {h.process.pid for h in registry.values()} if registry else set()
    before_children = _direct_children_of(os.getpid()) | known_pids

    with _block_termination_signals():
        try:
            tmpdir = Path(tempfile.mkdtemp(prefix="at", dir=base_tmp))
            log_path = tmpdir / "worker.log"
            parent_sock, child_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
            env = os.environ.copy()
            env.update(WORKER_THREAD_ENV)
            tmpdir_str = str(tmpdir)
            env["TMPDIR"] = tmpdir_str
            env["TEMP"] = tmpdir_str
            env["TMP"] = tmpdir_str
            env["XDG_RUNTIME_DIR"] = tmpdir_str
            env["NI_RUA_WORKER_OWNERSHIP"] = tmpdir_str
            env["NI_RUA_TEST_WORKER_FD"] = str(child_sock.fileno())
            if threading.current_thread() is threading.main_thread():
                env["NI_RUA_RUNNER_PID"] = str(os.getpid())

            cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "-s", str(start_dir)]
            if top_level_dir is not None:
                cmd.extend(["-t", str(top_level_dir)])

            with log_path.open("wb") as log_file:
                proc = subprocess.Popen(
                    cmd,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    pass_fds=(child_sock.fileno(),),
                    process_group=0,
                )
            child_sock.close()
            child_sock = None
            handle = _WorkerHandle(worker_id, proc, parent_sock, tmpdir, log_path)
            parent_sock = None
            if registry is not None:
                registry[worker_id] = handle
            return handle
        except BaseException:
            if child_sock is not None:
                child_sock.close()
            if handle is not None:
                _close_workers(
                    (handle,),
                    reap_descendants=True,
                    registry=registry,
                    known_worker_pids=known_pids,
                )
            else:
                if parent_sock is not None:
                    parent_sock.close()
                if proc is not None:
                    _signal_worker_group(proc, signal.SIGKILL)
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
                if registry is not None:
                    registry.pop(worker_id, None)
                if tmpdir is not None:
                    shutil.rmtree(tmpdir, ignore_errors=True)
            _reap_leaked_children_since(before_children)
            raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NI-RUA 多进程并行 unittest 运行器")
    parser.add_argument("targets", nargs="*", help="可选的模块、测试类、测试方法名、测试文件或测试目录路径")
    parser.add_argument("-s", "--start-directory", default="tests", help="测试发现起始目录（默认: tests）")
    parser.add_argument("-p", "--pattern", default="test*.py", help="测试文件匹配模式（默认: test*.py）")
    parser.add_argument("-t", "--top-level-directory", default=None, help="项目顶层目录（默认跟随起始目录）")
    parser.add_argument("-k", dest="patterns", action="append", default=None, help="按名称子串或通配符过滤测试")
    parser.add_argument("-j", "--jobs", "--workers", dest="jobs", type=int, default=None, help=f"并发 Worker 进程数（默认自动按核数上限 {DEFAULT_MAX_JOBS}）")
    parser.add_argument("--timeout", type=float, default=900.0, help="单个分片超时秒数；0 表示不限时（默认: 900）")
    parser.add_argument("-f", "--failfast", action="store_true", help="遇到首个失败或错误时立即停止")
    parser.add_argument("-v", "--verbose", action="store_true", help="输出分片进度及最慢测试统计")
    parser.add_argument("-q", "--quiet", action="store_true", help="精简输出模式")
    parser.add_argument("--zone", action="append", default=None, help=f"按域过滤测试（{', '.join(VALID_ZONES)}；可重复或逗号分隔）")
    parser.add_argument("--subzone", action="append", default=None, help="按子域过滤测试（如 robot.so101 或 so101；可重复或逗号分隔）")
    parser.add_argument("--tier", action="append", default=None, help=f"按速度层级过滤测试（{', '.join(VALID_TIERS)}；可重复或逗号分隔）")
    parser.add_argument("--changed", nargs="?", const="HEAD", default=None, metavar="BASE_REF", help="基于 git diff（默认 HEAD）与 6 层有界 DAG 自动选择受影响测试")
    parser.add_argument("--changed-scope", choices=("auto", "direct", "full"), default="auto", help="变更传播范围（默认: auto）")
    parser.add_argument("--explain", "--dry-run", dest="explain", action="store_true", help="仅输出测试选择路径与计划，不执行测试")
    parser.add_argument("--strict-empty", action="store_true", help="当 --changed 命中 0 项测试时返回退出码 5 而非 0")
    parser.add_argument("--include-js", action="store_true", help="同时运行 web/check.mjs 与 agent-plugin npm test")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    start_dir = Path(args.start_directory).resolve()
    top_level_dir = Path(args.top_level_directory).resolve() if args.top_level_directory else None

    if args.worker:
        return _worker_main(start_dir, top_level_dir)

    zone_filter = _parse_csv_flags(args.zone, valid=VALID_ZONES, flag_name="--zone")
    subzone_filter = _parse_subzone_flags(args.subzone)
    tier_filter = _parse_csv_flags(args.tier, valid=VALID_TIERS, flag_name="--tier")

    jobs = args.jobs if args.jobs is not None else default_jobs()
    if jobs < 1:
        raise SystemExit(f"--jobs 必须为正整数，当前值: {jobs}")

    project_root = _resolve_project_root(start_dir)
    use_selector = bool(
        args.changed is not None
        or zone_filter
        or subzone_filter
        or tier_filter
        or args.include_js
        or args.explain
    )
    plan: SelectionPlan | None = None
    effective_targets = list(args.targets)
    partial_target_modules: set[str] = set()
    js_failures: list[tuple[str, str]] = []

    if use_selector:
        disc_top = (top_level_dir or start_dir).resolve()
        test_entries = _iter_discoverable_files(start_dir, disc_top, args.pattern)
        if args.changed is not None:
            plan = _select_changed_tests(
                project_root,
                start_dir,
                test_entries,
                base_ref=args.changed,
                changed_scope=args.changed_scope,
                zone_filter=zone_filter,
                subzone_filter=subzone_filter,
                tier_filter=tier_filter,
            )
            if args.include_js:
                plan = SelectionPlan(
                    selected_tests=plan.selected_tests,
                    reasons=plan.reasons,
                    classifications=plan.classifications,
                    changed_files=plan.changed_files,
                    run_js_web=plan.run_js_web or (project_root / "web" / "check.mjs").is_file(),
                    run_js_plugin=plan.run_js_plugin or (project_root / "agent-plugin" / "package.json").is_file(),
                    damper_skipped_count=plan.damper_skipped_count,
                    scan_elapsed_s=plan.scan_elapsed_s,
                )
        else:
            plan = _select_static_filtered_tests(
                project_root,
                start_dir,
                test_entries,
                zone_filter=zone_filter,
                subzone_filter=subzone_filter,
                tier_filter=tier_filter,
                include_js=bool(args.include_js),
            )

        if args.targets or (args.explain and args.patterns):
            plan, effective_targets, partial_target_modules = _narrow_selection_plan(
                plan,
                raw_targets=args.targets,
                k_patterns=args.patterns,
                start_dir=start_dir,
                top_level_dir=top_level_dir,
                project_root=project_root,
                explain_mode=bool(args.explain),
            )
        else:
            effective_targets = [str(p) for p in plan.selected_tests]

        if args.explain:
            _print_explain_plan(
                plan,
                project_root,
                changed_mode=(args.changed is not None),
                changed_scope=args.changed_scope,
            )
            has_js = (
                (plan.run_js_web and (project_root / "web" / "check.mjs").is_file())
                or (plan.run_js_plugin and (project_root / "agent-plugin" / "package.json").is_file())
            )
            if not plan.selected_tests and not has_js:
                return 5 if (args.strict_empty or args.changed is None) else 0
            return 0

        if plan.run_js_web or plan.run_js_plugin:
            js_failures = _run_delegated_js_checks(plan, project_root, quiet=args.quiet)
        if not effective_targets:
            if js_failures:
                for lbl, tb in js_failures:
                    print("=" * 70)
                    print(f"ERROR: {lbl}")
                    print("-" * 70)
                    print(tb.rstrip())
                print("-" * 70)
                print(f"FAILED (errors={len(js_failures)})")
                return 1
            has_js = (
                (plan.run_js_web and (project_root / "web" / "check.mjs").is_file())
                or (plan.run_js_plugin and (project_root / "agent-plugin" / "package.json").is_file())
            )
            if has_js:
                print("-" * 70)
                print("Ran 0 Python tests (JS checks passed)\n\nOK")
                return 0
            print("-" * 70)
            print("Ran 0 tests in 0.000s\n\nNO TESTS RAN")
            return 5 if (args.strict_empty or args.changed is None) else 0

    _ensure_subreaper()
    wall_t0 = time.perf_counter()
    all_handles: dict[int, _WorkerHandle] = {}
    next_worker_id = 1

    terminated_by_sig: int | None = None

    def _sigterm_handler(signum: int, _frame: object) -> None:
        nonlocal terminated_by_sig
        terminated_by_sig = signum
        raise SystemExit(128 + signum)

    prev_sigterm = signal.signal(signal.SIGTERM, _sigterm_handler)
    try:
        # 全量发现模式下先行拉起常驻 Worker，与主进程模块发现阶段重叠冷启动耗时
        prespawn = (
            min(jobs, DEFAULT_MAX_JOBS)
            if (not effective_targets and not args.patterns and not use_selector and jobs > 1)
            else 0
        )
        for _ in range(prespawn):
            _spawn_worker(next_worker_id, start_dir, top_level_dir, registry=all_handles)
            next_worker_id += 1

        base_top = configure_sys_path(start_dir, top_level_dir)
        loader = unittest.TestLoader()
        if args.patterns:
            loader.testNamePatterns = [
                f"*{pat}*" if "*" not in pat else pat for pat in args.patterns
            ]
        module_meta: dict[str, _ModuleMeta] = {}
        if effective_targets:
            suite, module_meta = load_targets(loader, effective_targets, base_top, top_level_dir, args.pattern)
        else:
            disc_top = top_level_dir or start_dir
            if start_dir == REPO_ROOT or _is_excluded_relpath(start_dir, REPO_ROOT):
                discovered = _iter_discoverable_files(start_dir, disc_top, args.pattern)
                suite, module_meta = load_targets(
                    loader, [str(p) for _, p in discovered], base_top, top_level_dir, args.pattern
                )
            else:
                suite = loader.discover(
                    str(start_dir),
                    pattern=args.pattern,
                    top_level_dir=str(top_level_dir) if top_level_dir else None,
                )
                _capture_case_module_meta(suite, module_meta)
                # 兼容不含 __init__.py 的 Wave 2 `tests/<zone>/<subzone>/` 目录
                discovered_files = {
                    Path(meta.expected_file).resolve()
                    for meta in module_meta.values()
                    if meta.expected_file
                }
                extra_zone_files = [
                    str(p)
                    for _, p in _iter_discoverable_files(start_dir, disc_top, args.pattern)
                    if p.resolve() not in discovered_files
                    and any(part in VALID_ZONES for part in p.relative_to(start_dir).parts[:-1])
                ]
                if extra_zone_files:
                    extra_suite, extra_meta = load_targets(
                        loader, extra_zone_files, base_top, top_level_dir, args.pattern
                    )
                    suite.addTests(extra_suite)
                    module_meta.update(extra_meta)

        expected_total = suite.countTestCases()
        if expected_total == 0:
            if js_failures:
                for lbl, tb in js_failures:
                    print("=" * 70)
                    print(f"ERROR: {lbl}")
                    print("-" * 70)
                    print(tb.rstrip())
                print("-" * 70)
                print(f"FAILED (errors={len(js_failures)})")
                return 1
            print("-" * 70)
            print("Ran 0 tests in 0.000s\n\nNO TESTS RAN")
            return 5 if (args.strict_empty or args.changed is None) else 0

        module_timings = _load_module_timings(project_root, start_dir)
        loader_failures, units = build_work_units(
            suite,
            parallel=jobs > 1,
            module_meta=module_meta,
            module_timings=module_timings,
        )
        tests_run = 0
        skipped: list[list[str]] = []
        failures: list[list[str]] = []
        errors: list[list[str]] = [[lbl, tb] for lbl, tb in js_failures]
        expected_failures: list[list[str]] = []
        unexpected_successes: list[str] = []
        all_timings: list[tuple[float, str]] = []
        module_elapsed: dict[str, float] = {}
        module_tests: dict[str, int] = {}
        bad_modules: set[str] = set()
        stopped_early = False

        if loader_failures:
            local_res = _CaptureResult()
            local_res.failfast = bool(args.failfast)
            unittest.TestSuite(loader_failures).run(local_res)
            tests_run += local_res.testsRun
            skipped.extend([t.id(), reason] for t, reason in local_res.skipped)
            failures.extend([t.id(), tb] for t, tb in local_res.failures)
            errors.extend([t.id(), tb] for t, tb in local_res.errors)
            expected_failures.extend([t.id(), tb] for t, tb in local_res.expectedFailures)
            unexpected_successes.extend(t.id() for t in local_res.unexpectedSuccesses)
            if args.failfast and (failures or errors or unexpected_successes):
                stopped_early = True
                units = []

        target_workers = min(jobs, len(units))
        while len(all_handles) < target_workers:
            _spawn_worker(next_worker_id, start_dir, top_level_dir, registry=all_handles)
            next_worker_id += 1

        pending = deque(units)
        completed_units = 0
        total_units = len(units)
        dot_col = 0

        def _emit_dot(mark: str) -> None:
            nonlocal dot_col
            sys.stdout.write(mark)
            dot_col += 1
            if dot_col >= 70:
                sys.stdout.write("\n")
                dot_col = 0
            sys.stdout.flush()

        with selectors.DefaultSelector() as sel:
            excess: list[_WorkerHandle] = []
            for handle in list(all_handles.values()):
                if len(sel.get_map()) < target_workers:
                    sel.register(handle.sock, selectors.EVENT_READ, handle)
                else:
                    try:
                        handle.send({"op": "stop"})
                    except OSError:
                        pass
                    excess.append(handle)
            _close_workers(excess, reap_descendants=True, registry=all_handles)

            def _is_registered(worker: _WorkerHandle) -> bool:
                if worker.sock.fileno() < 0:
                    return False
                try:
                    sel.get_key(worker.sock)
                    return True
                except KeyError:
                    return False

            def _abort_remaining() -> None:
                nonlocal stopped_early
                stopped_early = True
                pending.clear()
                for active in list(sel.get_map().values()):
                    sel.unregister(active.fileobj)
                _close_workers(list(all_handles.values()), reap_descendants=True, registry=all_handles)

            bootstrap_crashes = 0

            def _dispatch_next(worker: _WorkerHandle) -> bool:
                if worker.eof or worker.process.poll() is not None:
                    return _handle_worker_crash(worker)
                if pending and not stopped_early:
                    unit = pending.popleft()
                    try:
                        worker.send(
                            {
                                "op": "run",
                                "unit_id": unit.unit_id,
                                "targets": list(unit.targets),
                                "patterns": loader.testNamePatterns,
                                "module_name": unit.module_name,
                                "import_root": unit.import_root,
                                "expected_file": unit.expected_file,
                                "failfast": bool(args.failfast),
                                "timeout": float(args.timeout),
                            }
                        )
                    except OSError:
                        pending.appendleft(unit)
                        worker.eof = True
                        return _handle_worker_crash(worker)
                    worker.dispatched_unit = unit
                    worker.in_flight = None
                    worker.unit_started_at = time.monotonic()
                    return False
                worker.dispatched_unit = None
                worker.in_flight = None
                if _is_registered(worker):
                    sel.unregister(worker.sock)
                worker.stop_sent = True
                try:
                    worker.send({"op": "stop"})
                except OSError:
                    pass
                worker.close_io()
                return False

            def _handle_worker_crash(worker: _WorkerHandle) -> bool:
                nonlocal bootstrap_crashes, completed_units, next_worker_id
                if _is_registered(worker):
                    sel.unregister(worker.sock)
                if worker.dispatched_unit is not None:
                    pending.appendleft(worker.dispatched_unit)
                    worker.dispatched_unit = None
                crashed_unit = worker.in_flight
                prior_unit = worker.last_unit
                worker.in_flight = None
                worker.unit_started_at = None
                try:
                    rc = worker.process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    rc = None
                tail = ""
                if worker.log_path.is_file():
                    tail = worker.log_path.read_bytes()[-4096:].decode("utf-8", errors="replace").strip()
                _close_workers((worker,), reap_descendants=True, registry=all_handles)
                if crashed_unit is not None:
                    if crashed_unit.module_name:
                        bad_modules.add(crashed_unit.module_name)
                    completed_units += 1
                    detail = f"Worker 进程异常退出 (returncode={rc})"
                    if tail:
                        detail += f"\n--- worker.log ---\n{tail}"
                    errors.append([crashed_unit.label, detail])
                    if args.verbose:
                        print(
                            f"[{completed_units}/{total_units}] {crashed_unit.label} (crashed, rc={rc}) ... ERROR",
                            flush=True,
                        )
                    elif not args.quiet:
                        _emit_dot("E")
                    if args.failfast:
                        _abort_remaining()
                        return True
                elif prior_unit is not None:
                    detail = f"Worker 在分片完成后异常退出 (returncode={rc})"
                    if tail:
                        detail += f"\n--- worker.log ---\n{tail}"
                    errors.append([prior_unit.label, detail])
                    if not args.quiet and not args.verbose:
                        _emit_dot("E")
                    if args.failfast:
                        _abort_remaining()
                        return True
                else:
                    bootstrap_crashes += 1
                    detail = f"Worker 启动阶段异常退出 (returncode={rc})"
                    if tail:
                        detail += f"\n--- worker.log ---\n{tail}"
                    errors.append(["parallel_runner.worker_bootstrap", detail])
                    if not args.quiet and not args.verbose:
                        _emit_dot("E")
                    if args.failfast or bootstrap_crashes >= 2:
                        _abort_remaining()
                        return True
                if pending and not stopped_early:
                    replacement = _spawn_worker(
                        next_worker_id, start_dir, top_level_dir, registry=all_handles
                    )
                    next_worker_id += 1
                    sel.register(replacement.sock, selectors.EVENT_READ, replacement)
                return False

            def _process_worker(worker: _WorkerHandle) -> bool:
                nonlocal completed_units, tests_run
                for raw_line in worker.drain_lines():
                    msg = json.loads(raw_line)
                    event = msg.get("event")
                    if event == "ready":
                        if _dispatch_next(worker):
                            return True
                        if not _is_registered(worker):
                            return False
                        continue
                    if event == "started":
                        if worker.dispatched_unit is not None:
                            worker.in_flight = worker.dispatched_unit
                            worker.dispatched_unit = None
                        continue

                    finished_unit = worker.in_flight or worker.dispatched_unit
                    worker.last_unit = finished_unit
                    worker.dispatched_unit = None
                    worker.in_flight = None
                    worker.unit_started_at = None
                    completed_units += 1
                    tests_run += int(msg["tests_run"])
                    skipped.extend(msg["skipped"])
                    failures.extend(msg["failures"])
                    errors.extend(msg["errors"])
                    expected_failures.extend(msg["expected_failures"])
                    unexpected_successes.extend(msg["unexpected_successes"])
                    for tid, dt in msg["test_timings"]:
                        all_timings.append((float(dt), str(tid)))
                    if finished_unit is not None and finished_unit.module_name:
                        mname = finished_unit.module_name
                        module_elapsed[mname] = module_elapsed.get(mname, 0.0) + float(msg["elapsed_s"])
                        module_tests[mname] = module_tests.get(mname, 0) + int(msg["tests_run"])

                    unit_bad = bool(msg["failures"] or msg["errors"] or msg["unexpected_successes"])
                    if args.verbose and finished_unit is not None:
                        status_tag = "FAIL" if unit_bad else "ok"
                        print(
                            f"[{completed_units}/{total_units}] {finished_unit.label} "
                            f"({msg['tests_run']} 项, {msg['elapsed_s']:.2f}s) ... {status_tag}",
                            flush=True,
                        )
                    elif not args.quiet:
                        mark = (
                            "F"
                            if msg["failures"]
                            else ("E" if msg["errors"] else ("u" if msg["unexpected_successes"] else "."))
                        )
                        _emit_dot(mark)

                    if unit_bad and args.failfast:
                        _abort_remaining()
                        return True

                    if _dispatch_next(worker):
                        return True
                    if not _is_registered(worker):
                        return False

                if _is_registered(worker) and (worker.eof or worker.process.poll() is not None):
                    return _handle_worker_crash(worker)
                return False

            while sel.get_map():
                events = sel.select(timeout=0.05)
                inspected: set[int] = set()
                aborted = False
                for key, _ in events:
                    worker: _WorkerHandle = key.data
                    inspected.add(worker.worker_id)
                    if _process_worker(worker):
                        aborted = True
                        break
                if not aborted:
                    now = time.monotonic()
                    for active in list(sel.get_map().values()):
                        worker = active.data
                        if (
                            args.timeout > 0
                            and worker.unit_started_at is not None
                            and (now - worker.unit_started_at) > args.timeout
                        ):
                            timed_out_unit = worker.in_flight or worker.dispatched_unit
                            if _is_registered(worker):
                                sel.unregister(worker.sock)
                            worker.dispatched_unit = None
                            worker.in_flight = None
                            worker.unit_started_at = None
                            _close_workers((worker,), reap_descendants=True, registry=all_handles)
                            tail = ""
                            if worker.log_path.is_file():
                                tail = worker.log_path.read_bytes()[-4096:].decode("utf-8", errors="replace").strip()
                            if timed_out_unit is not None:
                                if timed_out_unit.module_name:
                                    bad_modules.add(timed_out_unit.module_name)
                                completed_units += 1
                                detail = f"分片执行超时 (>{args.timeout:g}s)"
                                if tail:
                                    detail += f"\n--- worker.log ---\n{tail}"
                                errors.append([timed_out_unit.label, detail])
                                if args.verbose:
                                    print(
                                        f"[{completed_units}/{total_units}] {timed_out_unit.label} (timeout) ... ERROR",
                                        flush=True,
                                    )
                                elif not args.quiet:
                                    _emit_dot("E")
                                if args.failfast:
                                    _abort_remaining()
                                    aborted = True
                                    break
                            if pending and not stopped_early:
                                replacement = _spawn_worker(
                                    next_worker_id, start_dir, top_level_dir, registry=all_handles
                                )
                                next_worker_id += 1
                                sel.register(replacement.sock, selectors.EVENT_READ, replacement)
                            continue
                        if worker.worker_id not in inspected and worker.process.poll() is not None:
                            if _process_worker(worker):
                                break

        if all_handles:
            if not stopped_early:
                stopping = [h for h in all_handles.values() if h.stop_sent and h.last_unit is not None]
                stop_deadline = time.monotonic() + 5.0
                while any(h.process.poll() is None for h in stopping) and time.monotonic() < stop_deadline:
                    time.sleep(0.005)
                for h in stopping:
                    rc = h.process.poll()
                    if rc != 0 and h.last_unit is not None:
                        tail = ""
                        if h.log_path.is_file():
                            tail = h.log_path.read_bytes()[-4096:].decode("utf-8", errors="replace").strip()
                        detail = (
                            f"Worker 在分片完成后异常退出 (returncode={rc})"
                            if rc is not None
                            else "Worker 在收到 stop 后未在 5s 内正常退出"
                        )
                        if tail:
                            detail += f"\n--- worker.log ---\n{tail}"
                        errors.append([h.last_unit.label, detail])
            _close_workers(list(all_handles.values()), reap_descendants=True, registry=all_handles)

        if dot_col > 0:
            sys.stdout.write("\n")
            sys.stdout.flush()

        if not stopped_early and completed_units != total_units:
            errors.append(
                [
                    "parallel_runner.incomplete_schedule",
                    f"计划分片数 ({total_units}) 与已完成分片数 ({completed_units}) 不一致",
                ]
            )

        wall_elapsed = time.perf_counter() - wall_t0
        for kind, items in (("ERROR", errors), ("FAIL", failures)):
            for tid, tb in items:
                print("=" * 70)
                print(f"{kind}: {tid}")
                print("-" * 70)
                print(tb.rstrip())
        for tid in unexpected_successes:
            print("=" * 70)
            print(f"UNEXPECTED SUCCESS: {tid}")

        if args.verbose and all_timings:
            all_timings.sort(reverse=True)
            print("-" * 70)
            print("最慢的 10 项测试:")
            for dt, tid in all_timings[:10]:
                print(f"  {dt:6.2f}s  {tid}")

        print("-" * 70)
        print(f"Ran {tests_run} tests in {wall_elapsed:.3f}s (workers={target_workers or 1})")
        print()

        ok = not (failures or errors or unexpected_successes)
        details: list[str] = []
        if failures:
            details.append(f"failures={len(failures)}")
        if errors:
            details.append(f"errors={len(errors)}")
        if skipped:
            details.append(f"skipped={len(skipped)}")
        if expected_failures:
            details.append(f"expected failures={len(expected_failures)}")
        if unexpected_successes:
            details.append(f"unexpected successes={len(unexpected_successes)}")
        headline = "OK" if ok else "FAILED"
        print(f"{headline} ({', '.join(details)})" if details else headline)
        if not stopped_early and not args.patterns and module_elapsed:
            if not use_selector and effective_targets:
                full_mod_targets: set[str] = set()
                cand_partial_mods: set[str] = set()
                for raw_t in effective_targets:
                    if Path(raw_t).exists() or any(ch in raw_t for ch in "*?["):
                        continue
                    t_norm, t_mod, _t_src, _t_root = _resolve_target_spec(raw_t, start_dir.resolve(), top_level_dir)
                    if t_mod is not None:
                        if t_norm != t_mod:
                            cand_partial_mods.add(t_mod)
                        else:
                            full_mod_targets.add(t_mod)
                partial_target_modules |= cand_partial_mods - full_mod_targets

            full_loader = unittest.TestLoader()
            measured: dict[str, tuple[float, int]] = {}
            for m, elapsed_val in module_elapsed.items():
                if m in bad_modules or m in partial_target_modules:
                    continue
                ran_count = module_tests.get(m, 0)
                if ran_count <= 0:
                    continue
                if effective_targets:
                    loaded_mod = sys.modules.get(m)
                    if loaded_mod is None:
                        continue
                    try:
                        total_mod_cases = full_loader.loadTestsFromModule(loaded_mod).countTestCases()
                    except Exception:
                        continue
                    if total_mod_cases <= 0 or ran_count < total_mod_cases:
                        continue
                measured[m] = (elapsed_val, ran_count)
            _save_module_timings(project_root, start_dir, measured)
        return 0 if ok else 1
    finally:
        if terminated_by_sig is not None:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        _close_workers(list(all_handles.values()), reap_descendants=True, registry=all_handles)
        if terminated_by_sig is None:
            signal.signal(signal.SIGTERM, prev_sigterm)


_ensure_subreaper()


if __name__ == "__main__":
    raise SystemExit(main())
