"""Stage 2 architecture, AST worker closure, N-DoF kinematics/motion, IPC, protocol, and SAM3.1 tracking tests."""
from __future__ import annotations

NI_RUA_TEST_TIER = "fast"

import ast
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest

import numpy as np

from ni_rua.core.embodiment import GripperProfile
from ni_rua.perception.external import ik_prefilter as ikp
from ni_rua.robot.backends.robodojo.robodojo import RoboDojoBackend
from ni_rua.robot.backends.so101 import contract as so101_contract
from ni_rua.robot.backends.so101.backend import SO101Backend
from ni_rua.robot.contracts import (
    CalibratableBackend,
    ClockDomain,
    EvaluatorBackend,
    OperatorBackend,
    RobotBackend,
    VideoBackend,
    clock_domain_of,
)
from ni_rua.robot.ipc import WIRE_PREFIX, WorkerProcess
from ni_rua.robot.model import calibration_correction as cal_corr
from ni_rua.robot.model import end_effectors, geometry, kinematics as kin, so101_kinematics as s101
from ni_rua.robot.motion import contact_reanchor, feedback_motion, joint_replan, motion_safety
from ni_rua.robot.motion.batch_scheduler import ArmLayout, ArmSlot, BatchScheduler, MotionControllerPort
from ni_rua.robot.motion.endpoint_ik_polish import polish_trajectory_endpoint
from support.paths import CONFIGS_ROOT, REPO_ROOT

NI_RUA_ROOT = REPO_ROOT / "ni_rua"
ROBOT_ROOT = NI_RUA_ROOT / "robot"
FORBIDDEN_UPWARD_PREFIXES = (
    "ni_rua.perception",
    "ni_rua.cortex",
    "ni_rua.runtime",
    "ni_rua.apps",
)


def _resolve_import_targets(py_file: Path) -> list[tuple[int, str]]:
    """Extract all static and dynamic ``ni_rua.*`` / top-level import targets from ``py_file`` via AST."""
    rel_to_ni_rua = py_file.relative_to(NI_RUA_ROOT)
    pkg_parts = ["ni_rua", *rel_to_ni_rua.parts[:-1]]
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    found: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level > 0:
                keep = len(pkg_parts) - (node.level - 1)
                prefix = pkg_parts[:keep]
                base = ".".join([*prefix, mod] if mod else prefix)
            else:
                base = mod
            if base:
                found.append((node.lineno, base))
            for alias in node.names:
                if alias.name != "*":
                    target = f"{base}.{alias.name}" if base else alias.name
                    found.append((node.lineno, target))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "import_module"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            arg0 = node.args[0].value
            if arg0.startswith("."):
                lvl = len(arg0) - len(arg0.lstrip("."))
                rem = arg0[lvl:]
                keep = len(pkg_parts) - (lvl - 1)
                prefix = pkg_parts[:keep]
                resolved = ".".join([*prefix, rem] if rem else prefix)
                found.append((node.lineno, resolved))
            else:
                found.append((node.lineno, arg0))
    return found


def _module_to_path(dotted: str) -> Path | None:
    """Map a ``ni_rua.*`` dotted module name to its file on disk if it exists."""
    if not dotted.startswith("ni_rua."):
        return None
    parts = dotted.split(".")[1:]
    candidate_py = NI_RUA_ROOT.joinpath(*parts).with_suffix(".py")
    if candidate_py.is_file():
        return candidate_py
    candidate_init = NI_RUA_ROOT.joinpath(*parts, "__init__.py")
    if candidate_init.is_file():
        return candidate_init
    return None


def _synthetic_chain_manifest(dof: int) -> dict:
    """Construct a well-conditioned ``dof``-joint 3D revolute manipulator manifest (``N in {5, 6, 7}``)."""
    axes = [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0] if dof == 7 else [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ]
    offsets = [
        [0.0, 0.0, 0.12],
        [0.03, 0.0, 0.10],
        [0.0, 0.02, 0.18],
        [0.16, 0.0, 0.04],
        [0.12, -0.01, 0.02],
        [0.08, 0.0, 0.03],
        [0.06, 0.0, 0.0],
    ]
    joints = []
    parent = "base_link"
    for i in range(dof):
        child = f"link{i + 1}"
        joints.append(
            {
                "name": f"joint{i + 1}",
                "type": "revolute",
                "parent": parent,
                "child": child,
                "T_parent_joint": kin.rpy_xyz_to_matrix(tuple(offsets[i]), (0.0, 0.0, 0.0)).tolist(),
                "axis": axes[i],
                "limits": {"lower": -2.8, "upper": 2.8},
            }
        )
        parent = child
    joints.append(
        {
            "name": "ee_fixed",
            "type": "fixed",
            "parent": parent,
            "child": "tcp_link",
            "T_parent_joint": kin.rpy_xyz_to_matrix((0.05, 0.0, 0.0), (0.0, 0.0, 0.0)).tolist(),
        }
    )
    return {"base_link": "base_link", "joints": joints}


class RobotArchitectureBoundaryTests(unittest.TestCase):
    """Enforce strict layer isolation and worker transitive closure inside ``ni_rua/robot/``."""

    def test_ni_rua_robot_has_zero_upward_imports_and_zero_live_reloads(self) -> None:
        violations: list[str] = []
        reload_calls: list[str] = []
        so101_to_robodojo: list[str] = []

        for py_file in sorted(ROBOT_ROOT.rglob("*.py")):
            rel = py_file.relative_to(REPO_ROOT)
            for lineno, target in _resolve_import_targets(py_file):
                if target.startswith(FORBIDDEN_UPWARD_PREFIXES):
                    violations.append(f"{rel}:{lineno} -> {target}")
                if "so101" in py_file.parts and target.startswith("ni_rua.robot.backends.robodojo"):
                    so101_to_robodojo.append(f"{rel}:{lineno} -> {target}")

            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "reload"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "importlib"
                ):
                    reload_calls.append(f"{rel}:{node.lineno}")

        self.assertEqual(violations, [], f"Forbidden upward imports in ni_rua/robot/: {violations}")
        self.assertEqual(so101_to_robodojo, [], f"SO-101 must not inherit from RoboDojo: {so101_to_robodojo}")
        self.assertEqual(reload_calls, [], f"Live importlib.reload forbidden in ni_rua/robot/: {reload_calls}")

    def test_worker_entrypoints_transitive_closure_is_self_contained(self) -> None:
        entrypoints = (
            ROBOT_ROOT / "backends" / "robodojo" / "robot_worker.py",
            ROBOT_ROOT / "backends" / "so101" / "worker.py",
        )
        for entry in entrypoints:
            self.assertTrue(entry.is_file(), str(entry))
            visited: set[Path] = set()
            queue: list[Path] = [entry]
            bad: list[str] = []
            while queue:
                cur = queue.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                # Ensure package __init__.py files along the path are also checked.
                parent = cur.parent
                while parent.is_relative_to(NI_RUA_ROOT) and (parent / "__init__.py").is_file():
                    init_py = parent / "__init__.py"
                    if init_py not in visited:
                        queue.append(init_py)
                    parent = parent.parent

                for lineno, target in _resolve_import_targets(cur):
                    if target.startswith(FORBIDDEN_UPWARD_PREFIXES):
                        bad.append(f"{cur.relative_to(REPO_ROOT)}:{lineno} -> {target}")
                        continue
                    mod_path = _module_to_path(target)
                    if mod_path is not None and mod_path not in visited:
                        queue.append(mod_path)
                    # Also resolve parent module if target named an exported symbol.
                    if mod_path is None and "." in target and target.startswith("ni_rua."):
                        parent_mod = _module_to_path(target.rsplit(".", 1)[0])
                        if parent_mod is not None and parent_mod not in visited:
                            queue.append(parent_mod)

            self.assertGreater(len(visited), 5)
            self.assertEqual(bad, [], f"Transitive upward leak from {entry.name}: {bad}")

    def test_kinematics_is_stdlib_and_numpy_only_and_geometry_is_robot_neutral(self) -> None:
        kin_file = ROBOT_ROOT / "model" / "kinematics.py"
        for lineno, target in _resolve_import_targets(kin_file):
            top = target.split(".")[0]
            self.assertIn(
                top,
                {"__future__", "collections", "dataclasses", "functools", "math", "typing", "numpy", "scipy", "pinocchio", "pink"},
                f"kinematics.py:{lineno} imports forbidden dependency {target!r}",
            )

        geom_file = ROBOT_ROOT / "model" / "geometry.py"
        for lineno, target in _resolve_import_targets(geom_file):
            self.assertNotIn("so101", target, f"geometry.py:{lineno} still imports so101: {target}")
            self.assertFalse(target.startswith("ni_rua.perception"), f"geometry.py:{lineno} -> {target}")


class ContractsAndIpcTests(unittest.TestCase):
    """Verify formal ``RobotBackend`` protocols, ``ClockDomain`` gating, and backend-neutral IPC."""

    def test_backend_protocols_and_clock_domain_exposure_sync(self) -> None:
        from ni_rua.robot.backends.so101.deployment import load_deployment
        from ni_rua.robot.embodiments import arx_x5_profile, so101_profile

        x5 = object.__new__(RoboDojoBackend)
        s1 = object.__new__(SO101Backend)

        self.assertIsInstance(x5, RobotBackend)
        self.assertIsInstance(x5, EvaluatorBackend)
        self.assertIsInstance(x5, VideoBackend)
        self.assertEqual(clock_domain_of(arx_x5_profile()), ClockDomain.SIMULATED_STEPS)

        self.assertIsInstance(s1, RobotBackend)
        self.assertIsInstance(s1, VideoBackend)
        self.assertIsInstance(s1, OperatorBackend)
        self.assertIsInstance(s1, CalibratableBackend)
        s1_prof = so101_profile(load_deployment(CONFIGS_ROOT / "so101.example.json"))
        self.assertEqual(clock_domain_of(s1_prof), ClockDomain.REAL_TIME)

        corr_dt = cal_corr.CalibrationCorrection(time_offset_ms=12.0)
        # Discrete sim-step clock domains reject non-zero hardware camera latency offsets.
        with self.assertRaises(ValueError):
            cal_corr.validate_exposure_sync(
                {"step_domain": "adapter_control_step", "sim_step": 10},
                frame={"step_domain": "adapter_control_step", "sim_step": 10},
                correction=corr_dt,
            )
        # Hardware monotonic wall domains accept bounded exposure sync metadata.
        ok = cal_corr.validate_exposure_sync(
            {"step_domain": cal_corr.HARDWARE_MONOTONIC_STEP_DOMAIN, "sim_step": 100_000_000},
            frame={"step_domain": cal_corr.HARDWARE_MONOTONIC_STEP_DOMAIN, "sim_step": 112_000_000},
            state={"step_domain": cal_corr.HARDWARE_MONOTONIC_STEP_DOMAIN, "sim_step": 100_000_000},
            correction=corr_dt,
        )
        self.assertEqual(ok["state_sync_skew_ns"], 0)

    def test_worker_process_and_serve_worker_preemptive_operator_roundtrip(self) -> None:
        worker_code = """
import json, os, sys, threading
sys.path.insert(0, %r)
from ni_rua.robot.ipc import WIRE_PREFIX, serve_worker

emit_lock = threading.Lock()
def emit(msg):
    with emit_lock:
        sys.stdout.write(WIRE_PREFIX + json.dumps(msg) + "\\n")
        sys.stdout.flush()

class TinyAdapter:
    def __init__(self):
        self.gate = threading.Event()
    def capabilities(self):
        return {"arms": ["right"]}
    def operator(self, command, **kwargs):
        if command == "unblock":
            self.gate.set()
            return {"unblocked": True}
        return {"command": command}
    def close(self):
        return {"closed": True}

def slow_op(adapter, req, emit_fn):
    emit_fn({"id": req["id"], "progress": {"phase": "started"}})
    adapter.gate.wait(timeout=5.0)
    return {"done": True}

def oom_op(adapter, req, emit_fn):
    sys.stderr.write("fatal_custom_oom_marker\\n")
    sys.stderr.flush()
    os._exit(2)

serve_worker(
    lambda _args: TinyAdapter(),
    None,
    emit,
    preemptive_operator_commands=frozenset({"unblock"}),
    operations={
        "ping": lambda adapter, req, emit_fn: {"pong": (req.get("payload") or {}).get("value")},
        "slow": slow_op,
        "oom_log": oom_op,
    },
)
""" % (str(REPO_ROOT),)

        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "tiny_worker.py"
            script.write_text(worker_code, encoding="utf-8")
            proc = WorkerProcess(
                Path(tmp) / "logs",
                log_failure_classifier=lambda text: "custom_oom" if "fatal_custom_oom_marker" in text else None,
            )
            try:
                caps = proc.start_process(
                    [sys.executable, "-u", str(script)],
                    cwd=tmp,
                    environment=dict(__import__("os").environ),
                )
                self.assertEqual(caps, {"arms": ["right"]})
                self.assertEqual(proc._request("ping", {"value": 42}, timeout=5.0)["pong"], 42)
                seen_events: list[dict] = []
                started_event = threading.Event()
                result_box: dict = {}

                def on_progress(ev: dict) -> None:
                    seen_events.append(ev)
                    started_event.set()

                def run_slow() -> None:
                    result_box["res"] = proc._request("slow", {}, timeout=5.0, progress=on_progress)

                t = threading.Thread(target=run_slow)
                t.start()
                self.assertTrue(started_event.wait(timeout=3.0))
                op_res = proc._request("operator", {"command": "unblock"}, timeout=5.0)
                t.join(timeout=5.0)
                self.assertTrue(op_res["unblocked"])
                self.assertTrue(result_box["res"]["done"])
                self.assertEqual(seen_events[0]["phase"], "started")

                with self.assertRaises(RuntimeError) as ctx:
                    proc._request("oom_log", {}, timeout=5.0)
                self.assertEqual(str(ctx.exception), "custom_oom")
            finally:
                proc.close()


class UnifiedNDofKinematicsTests(unittest.TestCase):
    """Verify ``ni_rua.robot.model.kinematics`` across 5-DoF, 6-DoF, and 7-DoF chains + parity."""

    def test_jacobian_matches_finite_differences_for_5_6_7_dof(self) -> None:
        rng = np.random.default_rng(2026)
        eps = 1e-7
        for dof in (5, 6, 7):
            with self.subTest(dof=dof):
                manifest = _synthetic_chain_manifest(dof)
                chain = kin.chain_from_manifest(manifest, tip_link="tcp_link")
                self.assertEqual(chain.dof, dof)
                q = rng.uniform(-1.1, 1.1, size=dof)
                t_tip, jac = kin.fk_jacobian(chain, q)
                self.assertEqual(jac.shape, (6, dof))

                pos_b, rot_b, jac_b, _ = kin.fk_batch(chain, q[None, :])
                np.testing.assert_allclose(pos_b[0], t_tip[:3, 3], atol=1e-12)
                np.testing.assert_allclose(rot_b[0], t_tip[:3, :3], atol=1e-12)
                np.testing.assert_allclose(jac_b[0], jac, atol=1e-12)

                jac_num = np.zeros((6, dof), dtype=np.float64)
                for j in range(dof):
                    dq = np.zeros(dof, dtype=np.float64)
                    dq[j] = eps
                    t_plus, _ = kin.fk_jacobian(chain, q + dq)
                    t_minus, _ = kin.fk_jacobian(chain, q - dq)
                    jac_num[:3, j] = (t_plus[:3, 3] - t_minus[:3, 3]) / (2.0 * eps)
                    jac_num[3:, j] = kin.rotation_log(t_plus[:3, :3] @ t_minus[:3, :3].T) / (2.0 * eps)
                np.testing.assert_allclose(jac, jac_num, atol=1e-6)

    def test_solve_ik_5_6_7_dof_and_7dof_nullspace_posture_resolution(self) -> None:
        rng = np.random.default_rng(77)
        for dof in (5, 6, 7):
            with self.subTest(dof=dof):
                chain = kin.chain_from_manifest(_synthetic_chain_manifest(dof), tip_link="tcp_link")
                q_true = rng.uniform(-0.8, 0.8, size=dof)
                t_goal = kin.fk_links(chain, q_true)["tcp_link"]
                q_seed = q_true + rng.uniform(-0.25, 0.25, size=dof)
                res = kin.solve_ik(
                    chain,
                    kin.IKTarget(T_base_tip=t_goal),
                    q_seed,
                    kin.IKOptions(position_tolerance_m=1e-4, orientation_tolerance_rad=1e-3, restarts=4),
                )
                self.assertTrue(res.converged, res.as_dict())
                self.assertLessEqual(res.position_error_m, 1e-4)
                self.assertLessEqual(res.orientation_error_rad, 1e-3)

        # Redundant 7-DoF arm: two distinct null-space posture preferences achieve the SAME 6D pose
        # while biasing the redundant self-motion toward their respective target postures.
        chain7 = kin.chain_from_manifest(_synthetic_chain_manifest(7), tip_link="tcp_link")
        q_ref = np.array([0.2, -0.4, 0.35, -0.9, 0.25, 0.5, -0.3], dtype=np.float64)
        t_goal7 = kin.fk_links(chain7, q_ref)["tcp_link"]
        posture_a = np.array([-0.8, -0.4, -0.9, -0.9, 0.25, 0.5, -0.3], dtype=np.float64)
        posture_b = np.array([0.8, -0.4, 0.9, -0.9, 0.25, 0.5, -0.3], dtype=np.float64)
        opts7 = kin.IKOptions(
            tasks=("pose", "posture"),
            posture_weight=0.25,
            max_iterations=220,
            position_tolerance_m=1e-4,
            orientation_tolerance_rad=1e-3,
            restarts=2,
        )
        res_a = kin.solve_ik(chain7, kin.IKTarget(T_base_tip=t_goal7, posture_rad=posture_a), q_ref, opts7)
        res_b = kin.solve_ik(chain7, kin.IKTarget(T_base_tip=t_goal7, posture_rad=posture_b), q_ref, opts7)
        self.assertTrue(res_a.converged and res_b.converged)
        qa = np.asarray(res_a.q_rad)
        qb = np.asarray(res_b.q_rad)
        self.assertGreater(float(np.linalg.norm(qa - qb)), 0.15)
        self.assertLess(float(np.linalg.norm(qa - posture_a)), float(np.linalg.norm(qb - posture_a)))
        self.assertLess(float(np.linalg.norm(qb - posture_b)), float(np.linalg.norm(qa - posture_b)))

    def test_so101_x5_and_endpoint_ik_polish_numeric_parity(self) -> None:
        manifest = s101.load_model()
        q5 = np.array([0.25, -0.35, 0.65, -0.30, 0.40], dtype=np.float64)
        poses_s101 = s101.fk_all(manifest, q5, gripper_rad=0.2)
        poses_geom = geometry.fk(
            manifest,
            {"joint_names": [*s101.ARM_JOINTS, s101.GRIPPER_JOINT], "q_rad": [*q5.tolist(), 0.2], "source_kind": "measurement"},
        )
        for link_name, mat in poses_s101.items():
            np.testing.assert_allclose(mat, poses_geom[link_name], atol=1e-12)

        target5 = poses_s101[s101.TCP_LINK]
        ik5 = s101.ik(manifest, target5, q5 + 0.12)
        self.assertTrue(ik5["converged"])
        self.assertLessEqual(ik5["position_error_m"], 1e-3)

        # 6-DoF X5 endpoint polish routed through kinematics.py
        x5_spec = ikp.resolve_6r_chain_spec("right")
        k_chain = x5_spec.to_kinematic_chain()
        t_wb = ikp.resolve_arm_base_transform("right")
        q6_goal = np.array([0.15, 0.55, -0.75, 0.30, -0.20, 0.10], dtype=np.float64)
        t_tip_w = kin.fk_links(k_chain, q6_goal, T_world_base=t_wb)[k_chain.tip_link]
        quat_wxyz = np.array(
            [
                0.5 * math.sqrt(max(0.0, 1.0 + np.trace(t_tip_w[:3, :3]))),
                0.0,
                0.0,
                0.0,
            ]
        )
        # Use roboprobe_pose-compatible quaternion via ikp
        from ni_rua.perception.external.field_ops import matrix_to_quat_wxyz

        qw, qx, qy, qz = matrix_to_quat_wxyz(t_tip_w[:3, :3])
        target_pose = {"position_m": t_tip_w[:3, 3].tolist(), "quaternion_xyzw": [qx, qy, qz, qw]}
        perturbed_end = q6_goal + np.array([0.008, -0.006, 0.007, -0.005, 0.006, -0.004])
        traj = np.linspace(q6_goal - 0.15, perturbed_end, 12)

        def _pose_err_fn(q_vec):
            tf = kin.fk_links(k_chain, q_vec, T_world_base=t_wb)[k_chain.tip_link]
            return kin.pose_error(tf, t_tip_w)

        _pose_err_fn.chain = k_chain  # type: ignore[attr-defined]
        _pose_err_fn.T_world_base = t_wb  # type: ignore[attr-defined]

        diag: dict = {}
        polished = polish_trajectory_endpoint(
            traj,
            limits=k_chain.joint_limits_rad,
            target=target_pose,
            pose_error=_pose_err_fn,
            diagnostic=diag,
        )
        self.assertTrue(diag["selected"], diag)
        self.assertLess(diag["after_position_error_m"], 5e-4)
        np.testing.assert_allclose(polished[0], traj[0], atol=1e-12)


class NDofMotionAndReanchorRegressionTests(unittest.TestCase):
    """Verify 5/6/7-DoF motion modules, ``ArmLayout``, and ``#44`` ``contact_reanchor`` fixes."""

    def test_motion_safety_joint_replan_and_arrival_monitor_support_5_6_7_dof(self) -> None:
        for dof in (5, 6, 7):
            with self.subTest(dof=dof):
                names = [f"j{i}" for i in range(dof)]
                types = ["revolute"] * dof
                limits = [[-10.0, 10.0] for _ in range(dof)]

                start = np.zeros(dof)
                wound_goal = np.zeros(dof)
                wound_goal[0] = math.tau + 0.25
                traj = np.linspace(start, wound_goal, 12)

                safety = motion_safety.assess_motion(
                    start,
                    traj[1:],
                    joint_names=names,
                    joint_types=types,
                    joint_limits=limits,
                )
                self.assertEqual(safety["status"], "reject")
                self.assertEqual(safety["witnesses"][0]["reason"], "periodic_winding")

                eq, turns = joint_replan.nearest_equivalent(start, wound_goal, np.asarray(limits), types)
                self.assertEqual(turns, [-1] + [0] * (dof - 1))
                np.testing.assert_allclose(eq[0], 0.25, atol=1e-12)

                diag: dict = {}
                selected = joint_replan.select_joint_replan(
                    original=traj,
                    current=start,
                    limits=np.asarray(limits),
                    joint_types=types,
                    transforms=lambda q: {
                        "tip": np.array(
                            [
                                [math.cos(q[0]), -math.sin(q[0]), 0.0, 0.0],
                                [math.sin(q[0]), math.cos(q[0]), 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    },
                    pose_error=lambda q: (0.0, 0.0),
                    resample=lambda q: q[1:],
                    plan_joint=lambda s, g: {"status": "Success", "position": np.linspace(s, g, 6)},
                    deadline=30.0,
                    diagnostic=diag,
                    clock=lambda: 0.0,
                )
                self.assertTrue(diag["selected"], diag)
                self.assertEqual(selected.shape, (6, dof))

                target_pose = {"position_m": [0.2, 0.0, 0.3], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]}
                init_state = {
                    "ee_pose": {"position_m": [0.0, 0.0, 0.3], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]},
                    "q_rad": np.zeros(dof).tolist(),
                    "qdot_rad_s": np.zeros(dof).tolist(),
                }
                mon = feedback_motion.ArrivalMonitor(target_pose, eq.tolist(), init_state)
                self.assertEqual(mon.dof, dof)
                settled_state = {
                    "ee_pose": target_pose,
                    "q_rad": eq.tolist(),
                    "qdot_rad_s": np.zeros(dof).tolist(),
                }
                self.assertEqual(mon.observe(settled_state)[0], "continue")
                self.assertEqual(mon.observe(settled_state)[0], "arrived")

    def test_arm_layout_and_batch_scheduler_support_5_6_7_dof(self) -> None:
        from ni_rua.robot.backends.so101.deployment import load_deployment
        from ni_rua.robot.embodiments import arx_x5_profile, so101_profile

        x5_layout = ArmLayout.from_profile(arx_x5_profile())
        self.assertEqual(x5_layout.row_width, 14)
        self.assertEqual((x5_layout.offset("left"), x5_layout.dof("left"), x5_layout.jaw_index("left")), (0, 6, 6))
        self.assertEqual((x5_layout.offset("right"), x5_layout.dof("right"), x5_layout.jaw_index("right")), (7, 6, 13))

        s1_prof = so101_profile(load_deployment(CONFIGS_ROOT / "so101.example.json"))
        s1_layout = ArmLayout.from_profile(s1_prof)
        self.assertEqual(s1_layout.arms, ("right",))
        self.assertEqual((s1_layout.offset("right"), s1_layout.dof("right"), s1_layout.jaw_index("right"), s1_layout.row_width), (0, 5, 5, 6))

        dual7 = ArmLayout.from_specs({"left": 7, "right": 7}, with_jaw=True)
        self.assertEqual(dual7.row_width, 16)
        row16 = [0.0] * 16
        dual7.write_arm_q(row16, "right", [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(dual7.extract_arm_q(row16, "right"), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

        class _DummyController:
            def state(self):
                return {"arms": {}}

            def capabilities(self):
                return {}

        ctrl = _DummyController()
        self.assertIsInstance(ctrl, MotionControllerPort)
        sched = BatchScheduler(ctrl, layout=dual7)
        self.assertIs(sched.layout, dual7)

    def test_contact_reanchor_respects_slow_speed_scale_and_honest_has_reanchored_rows(self) -> None:
        class _SlowAdapter:
            def _waypoint_speed_scale(self, action_dict: dict, idx: int = 0) -> float:
                return float(action_dict.get("speed_scale", 1.0))

            def _speed_step_limits(self, arm: str, speed_scale: float) -> np.ndarray:
                return np.full(6, 0.04 * float(speed_scale), dtype=float)

        adapter = _SlowAdapter()
        # 4 steps at speed_scale=0.25 -> max allowed step is 0.01 rad/step (total budget 0.04 rad).
        # With the old bug max(speed_scale, 2.0), step_limits became 0.08 rad/step and falsely admitted a 0.16 rad jump!
        orig_rows = [[0.005 * (i + 1)] + [0.0] * 13 for i in range(4)]
        next_rec = {
            "kind": "move",
            "arm": "left",
            "start": 0,
            "end": 4,
            "expected_start_q": [0.0] * 6,
            "rows": [list(r) for r in orig_rows],
            "action": {"speed_scale": 0.25},
        }
        src_rec = {"kind": "move", "arm": "left", "start": 0, "end": 4, "rows": [], "contact_stopped_early": True}
        shared_rows = [list(r) for r in orig_rows]

        rejected = contact_reanchor.reanchor_move_stage_from_contact(
            adapter,
            "left",
            "descend",
            src_rec,
            next_rec,
            shared_rows,
            [-0.14, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.8],
        )
        self.assertIsNotNone(rejected)
        self.assertEqual(rejected["status"], "rejected")
        self.assertEqual(rejected["step_limits_rad"][0], 0.01)
        self.assertEqual(shared_rows, orig_rows)
        self.assertFalse(contact_reanchor.has_reanchored_rows({"descend": src_rec, "fine_slide": next_rec}))

        # Final-step contact_stop (start_cursor == stage_end) modifies zero rows -> has_reanchored_rows is False.
        final_rec = {
            "kind": "move",
            "arm": "left",
            "start": 0,
            "end": 4,
            "rows": [list(r) for r in orig_rows],
            "contact_stopped_early": True,
        }
        modified = contact_reanchor.clamp_stage_rows_to_held_q(final_rec, shared_rows, 4, [0.1] * 6, 0)
        self.assertFalse(modified)
        self.assertFalse(contact_reanchor.has_reanchored_rows({"final": final_rec}))

        # Mid-stage clamp (start_cursor < stage_end) modifies rows -> has_reanchored_rows is True.
        modified_mid = contact_reanchor.clamp_stage_rows_to_held_q(final_rec, shared_rows, 2, [0.1] * 6, 0)
        self.assertTrue(modified_mid)
        self.assertTrue(contact_reanchor.has_reanchored_rows({"final": final_rec}))

    def test_so101_compile_action_preserves_tolerance_preset_without_default_clobber(self) -> None:
        caps = {
            "arms": ["right"],
            "frame_id": "so101_base",
            "tcp_config_id": "so101-gripper-frame-v1",
            "default_timeout_s": 30.0,
            "motion_tolerance_defaults": {"position_tolerance_m": 0.012, "orientation_tolerance_rad": 0.2},
        }
        ident = {"run_id": "r", "episode_id": "e", "env_instance_id": "i", "env_generation": 1}
        compiled = so101_contract.compile_action(
            "move",
            {
                "observation_id": "obs_1",
                "arm": "right",
                "position_m": [0.2, 0.0, 0.1],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                "tolerance_preset": "coarse",
                "stall_guard_preset": "contact_stop",
            },
            ident,
            caps,
        )
        self.assertEqual(compiled["path"]["tolerance_preset"], "coarse")
        self.assertEqual(compiled["path"]["stall_guard_preset"], "contact_stop")
        self.assertNotIn("position_tolerance_m", compiled["path"])
        self.assertNotIn("orientation_tolerance_rad", compiled["path"])


class ShimAndEndEffectorHookTests(unittest.TestCase):
    """Verify backward-compatible module shims and pluggable ``GraspReferenceHook``."""

    def test_embodiment_and_calibration_shims_preserve_module_identity(self) -> None:
        from ni_rua.robot.embodiments.so101 import gripper_cad as new_cad, profile as new_prof
        from ni_rua.robot.model import so101_gripper_geometry as old_cad, so101_kinematics as old_prof

        self.assertIs(old_cad, new_cad)
        self.assertIs(old_prof, new_prof)

    def test_custom_grasp_reference_hook_registration(self) -> None:
        class _SuctionCupHook:
            def locate(self, gripper, origin, rotation, jaw_rad=None):
                tip = origin + rotation @ np.array([0.0, 0.0, 0.075], dtype=float)
                normal = rotation[:, 2]
                return tip, normal, None

            def aperture_m(self, gripper, jaw_rad):
                return 0.0

        hook = _SuctionCupHook()
        self.assertIsInstance(hook, end_effectors.GraspReferenceHook)
        end_effectors.register_grasp_reference("_test_suction_cup", hook)
        try:
            origin = np.array([0.1, -0.2, 0.5], dtype=float)
            rot = np.eye(3, dtype=float)
            located = end_effectors.locate_grasp_reference(
                SimpleNamespace(grasp_reference="_test_suction_cup"),
                origin,
                rot,
            )
            self.assertIsNotNone(located)
            np.testing.assert_allclose(located[0], [0.1, -0.2, 0.575], atol=1e-12)
        finally:
            end_effectors._GRASP_REFERENCE_HOOKS.pop("_test_suction_cup", None)


import copy
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from services.vggt_height.sam31.video_tracking import (
    SAM3_REVISION, _adapt_session_init, _adapt_uncached_refinements, build_predictor, run_video_tracking,
)


class FakePredictor:
    def __init__(self, *, omit=(), fail=False, cleanup=True, malformed=False):
        self.requests = []
        self.omit, self.fail, self.cleanup, self.malformed = omit, fail, cleanup, malformed

    def response(self, index):
        ids = np.array([7], dtype=np.int64) if index != 1 else np.zeros(0, np.int64)
        masks = np.zeros((len(ids), 6, 8), dtype=bool)
        if len(ids):
            masks[:, 2:4, 2 + index:4 + index] = True
        if self.malformed:
            masks = masks[:, :-1, :]
        return {"frame_index": index, "outputs": {"out_obj_ids": ids, "out_binary_masks": masks,
                                                  "out_probs": np.full(len(ids), .8)}}

    def handle_request(self, request):
        self.requests.append(copy.deepcopy(request))
        if request["type"] == "start_session":
            return {"session_id": "fake-session"}
        if request["type"] == "add_prompt":
            return self.response(request["frame_index"])
        if request["type"] == "close_session":
            if isinstance(self.cleanup, Exception):
                raise self.cleanup
            return {"is_success": self.cleanup}
        raise AssertionError(request)

    def handle_stream_request(self, request):
        self.requests.append(copy.deepcopy(request))
        for index in range(3):
            if index in self.omit:
                continue
            yield self.response(index)
            if self.fail:
                raise RuntimeError("injected inference failure")


class PinnedInitMismatchPredictor(FakePredictor):
    """Reproduce the actual pinned base start_session -> multiplex init_state signature mismatch."""
    def __init__(self, *, model_error=None):
        super().__init__()
        self.initializations, self.sessions = [], {}
        owner = self

        class MultiplexModel:
            def init_state(self, resource_path, offload_video_to_cpu=False, async_loading_frames=False):
                owner.initializations.append(dict(resource_path=resource_path,
                    offload_video_to_cpu=offload_video_to_cpu, async_loading_frames=async_loading_frames))
                if model_error is not None:
                    raise model_error
                return {"frame_state": "cpu-test"}

        self.model = MultiplexModel()
        self.async_loading_frames = False

    def handle_request(self, request):
        if request["type"] == "start_session":
            self.requests.append(copy.deepcopy(request))
            # Same unconditional kwargs as pinned Sam3BasePredictor.start_session.
            state = self.model.init_state(resource_path=request["resource_path"],
                offload_video_to_cpu=request.get("offload_video_to_cpu", False),
                offload_state_to_cpu=request.get("offload_state_to_cpu", False),
                async_loading_frames=self.async_loading_frames)
            self.sessions["fake-session"] = state
            return {"session_id": "fake-session"}
        if request["type"] == "close_session":
            self.sessions.pop(request["session_id"], None)
        return super().handle_request(request)


class PinnedMissingCacheModel:
    """Exact pinned merger behavior, without importing torch or constructing a model."""
    def __init__(self):
        self.merge_calls = 0

    def _build_sam2_output(self, inference_state, frame_idx, refined_obj_id_to_mask=None):
        self.merge_calls += 1
        if frame_idx not in inference_state["cached_frame_outputs"]:
            return {}
        obj_id_to_mask = inference_state["cached_frame_outputs"][frame_idx].copy()
        if refined_obj_id_to_mask is not None:
            for obj_id, refined_mask in refined_obj_id_to_mask.items():
                assert refined_mask is not None, f"Refined mask data must be provided for obj_id {obj_id}"
                obj_id_to_mask[obj_id] = refined_mask
        return obj_id_to_mask


class SAM31VideoTrackingTests(unittest.TestCase):
    def frames(self, root):
        directory = Path(root) / "frames"
        directory.mkdir()
        for i in range(3):
            Image.fromarray(np.full((6, 8, 3), i * 30, dtype=np.uint8)).save(directory / f"{i:06d}.png")
        return directory

    def prompt(self):
        return [{"frame_index": 0, "object_id": 7, "points_px": [[4, 3], [0, 0]], "point_labels": [1, 0]}]

    def test_original_png_and_official_requests_masks_and_missing_object(self):
        with tempfile.TemporaryDirectory() as root:
            frames = self.frames(root)
            original = {p.name: p.read_bytes() for p in frames.iterdir()}
            predictor = FakePredictor()
            report = run_video_tracking(frames, self.prompt(), Path(root) / "result", predictor=predictor)
            self.assertEqual(report["status"], "completed")
            self.assertTrue(report["session_closed"])
            self.assertFalse(report["source_verified"])
            self.assertTrue(report["predictor_injected"])
            self.assertEqual(predictor.requests[0]["resource_path"], str(frames.resolve()))
            prompt = predictor.requests[1]
            self.assertEqual(prompt["points"], [[.5, .5], [0, 0]])
            self.assertEqual(prompt["obj_id"], 7)
            self.assertTrue(prompt["rel_coordinates"])
            self.assertEqual(prompt["point_labels"], [1, 0])
            stream = predictor.requests[2]
            self.assertEqual(stream["type"], "propagate_in_video")
            self.assertEqual(stream["propagation_direction"], "both")
            self.assertEqual(predictor.requests[-1]["type"], "close_session")
            self.assertEqual(report["frames"][1]["objects"][0]["mask_visibility"], "not_returned")
            self.assertEqual(report["frames"][1]["objects"][0]["physical_visibility"], "unknown")
            first = report["frames"][0]["objects"][0]
            self.assertEqual(first["mask_area_px"], 4)
            self.assertEqual(first["mask_visibility"], "mask_present")
            pixels = np.asarray(Image.open(Path(root) / "result" / first["mask_file"]))
            self.assertEqual(int((pixels > 0).sum()), 4)
            self.assertEqual(first["mask_sha256"], hashlib.sha256(
                (Path(root) / "result" / first["mask_file"]).read_bytes()).hexdigest())
            self.assertEqual(report["frame_updates"][-3]["objects"][0]["mask_sha256"], first["mask_sha256"])
            self.assertIsNone(report["material_keypoints"])
            self.assertEqual(report["contact_evidence"], "unknown")
            self.assertEqual(len(report["frame_updates"]), 4)
            self.assertEqual(report["frames_with_any_mask"], 2)
            self.assertEqual(report["object_coverage"], [{"object_id": 7, "total_frames": 3, "mask_present": 2,
                "empty_mask": 0, "not_returned": 1, "unknown": 0, "first_mask_frame_index": 0, "last_mask_frame_index": 2}])
            self.assertEqual({p.name: p.read_bytes() for p in frames.iterdir()}, original)
            self.assertEqual(json.loads((Path(root) / "result/tracking.json").read_text())["status"], "completed")

    def test_pinned_init_mismatch_reproduced_then_adapted_without_retry(self):
        predictor = PinnedInitMismatchPredictor()
        with self.assertRaisesRegex(TypeError, "offload_state_to_cpu"):
            predictor.handle_request({"type": "start_session", "resource_path": "unused", "offload_video_to_cpu": True})
        self.assertEqual(predictor.initializations, [])
        self.assertEqual(predictor.sessions, {})
        with tempfile.TemporaryDirectory() as root:
            report = run_video_tracking(self.frames(root), self.prompt(), Path(root) / "out", predictor=predictor)
        self.assertEqual(report["status"], "completed")
        self.assertTrue(report["session_closed"])
        self.assertEqual(len(predictor.initializations), 1)
        self.assertTrue(predictor.initializations[0]["offload_video_to_cpu"])
        self.assertFalse(predictor.initializations[0]["async_loading_frames"])
        self.assertEqual(predictor.sessions, {})
        self.assertEqual(report["session_init_compatibility"]["adapter"], "omit_unsupported_false_state_offload")
        self.assertFalse(report["effective_settings"]["offload_state_to_cpu"])

    def test_init_compatibility_rejects_true_preserves_real_errors_and_is_idempotent(self):
        predictor = PinnedInitMismatchPredictor()
        _adapt_session_init(predictor)
        wrapped = predictor.model.init_state
        _adapt_session_init(predictor)
        self.assertIs(predictor.model.init_state, wrapped)
        with self.assertRaisesRegex(ValueError, "does not support"):
            predictor.model.init_state(resource_path="unused", offload_state_to_cpu=True)
        self.assertFalse(predictor.initializations)
        broken = PinnedInitMismatchPredictor(model_error=TypeError("actual loader failure"))
        with tempfile.TemporaryDirectory() as root:
            report = run_video_tracking(self.frames(root), self.prompt(), Path(root) / "out", predictor=broken)
        self.assertEqual(report["status"], "failed")
        self.assertIn("actual loader failure", report["reason"])
        self.assertEqual(len(broken.initializations), 1)
        self.assertIsNone(report["session_closed"])
        self.assertFalse(broken.sessions)

    def test_supported_state_offload_signature_is_not_modified(self):
        calls = []
        class CompatibleModel:
            def init_state(self, resource_path, offload_state_to_cpu=False):
                calls.append(offload_state_to_cpu)
        predictor = types.SimpleNamespace(model=CompatibleModel())
        original = predictor.model.init_state.__func__
        self.assertEqual(_adapt_session_init(predictor)["status"], "not_required")
        self.assertIs(predictor.model.init_state.__func__, original)
        predictor.model.init_state("unused", offload_state_to_cpu=True)
        self.assertEqual(calls, [True])

    def test_uncached_refinements_reproduce_observed_frame_zero_only_bug(self):
        actual_mask = np.ones((1, 6, 8), bool)
        model = PinnedMissingCacheModel()
        state = {"cached_frame_outputs": {}}
        observed = []
        for index in (0, 0, 1, 2):
            masks = model._build_sam2_output(state, index, {7: actual_mask})
            observed.append(list(masks))
            state["cached_frame_outputs"][index] = masks  # caller owns caching, like upstream
        self.assertEqual(observed, [[], [7], [], []])
        predictor = types.SimpleNamespace(model=PinnedMissingCacheModel())
        _adapt_uncached_refinements(predictor)
        method = predictor.model._build_sam2_output
        _adapt_uncached_refinements(predictor)
        self.assertIs(method, predictor.model._build_sam2_output)
        state = {"cached_frame_outputs": {}}
        for index in (0, 0, 1, 2):
            cache_before = dict(state["cached_frame_outputs"])
            masks = method(state, index, {7: actual_mask})
            self.assertEqual(set(masks), {7})
            self.assertIs(masks[7], actual_mask)
            self.assertEqual(state["cached_frame_outputs"].keys(), cache_before.keys())
            for key in cache_before:
                self.assertIs(state["cached_frame_outputs"][key], cache_before[key])
            state["cached_frame_outputs"][index] = masks

    def test_merge_compatibility_preserves_existing_cache_empty_masks_and_exceptions(self):
        predictor = types.SimpleNamespace(model=PinnedMissingCacheModel())
        _adapt_uncached_refinements(predictor)
        merge = predictor.model._build_sam2_output
        old, new, empty = np.ones((1, 2, 2), bool), np.zeros((1, 2, 2), bool), np.zeros((1, 2, 2), bool)
        cached = {1: old, 2: old}
        state = {"cached_frame_outputs": {4: cached}}
        out = merge(state, 4, {1: new})
        self.assertEqual(predictor.model.merge_calls, 1)
        self.assertIs(out[1], new)
        self.assertIs(out[2], old)
        self.assertIs(state["cached_frame_outputs"][4], cached)
        self.assertIs(cached[1], old)
        self.assertIs(merge(state, 5, {9: empty})[9], empty)
        self.assertFalse(empty.any())  # no mask is fabricated to rescue empty real predictions
        self.assertNotIn(5, state["cached_frame_outputs"])
        self.assertEqual(merge(state, 6), {})
        self.assertEqual(predictor.model.merge_calls, 2)
        for frame in (4, 7):
            with self.assertRaises(AssertionError):
                merge(state, frame, {3: None})
        self.assertEqual(set(state["cached_frame_outputs"]), {4})
        self.assertIs(state["cached_frame_outputs"][4], cached)
        with self.assertRaises(AttributeError):
            merge(state, 8, [])
        self.assertNotIn(8, state["cached_frame_outputs"])

    def test_point_only_merge_fix_runs_through_prompt_propagation_and_coverage(self):
        class Predictor(FakePredictor):
            def __init__(self):
                super().__init__()
                self.model = PinnedMissingCacheModel()
                self.state = {"cached_frame_outputs": {}}
            def response(self, index):
                mask = np.ones((1, 6, 8), bool)
                refined = self.model._build_sam2_output(self.state, index, {7: mask})
                self.state["cached_frame_outputs"][index] = refined
                return {"frame_index": index, "outputs": {"out_obj_ids": np.asarray(list(refined), np.int64),
                    "out_binary_masks": np.concatenate(list(refined.values())) if refined else np.zeros((0,6,8), bool),
                    "out_probs": np.ones(len(refined))}}
        with tempfile.TemporaryDirectory() as root:
            report = run_video_tracking(self.frames(root), self.prompt(), Path(root) / "out", predictor=Predictor())
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["refinement_merge_compatibility"]["status"], "applied")
        self.assertEqual(report["object_coverage"][0]["mask_present"], 3)
        self.assertEqual(report["object_coverage"][0]["not_returned"], 0)
        self.assertEqual(report["contact_evidence"], "unknown")

    def test_completed_empty_or_absent_outputs_have_zero_visible_coverage(self):
        for include_empty_mask in (False, True):
            class Predictor(FakePredictor):
                def response(self, index):
                    n = int(include_empty_mask)
                    return {"frame_index": index, "outputs": {"out_obj_ids": np.array([7] if n else [], np.int64),
                        "out_binary_masks": np.zeros((n, 6, 8), bool), "out_probs": np.ones(n)}}
            with self.subTest(include_empty_mask=include_empty_mask), tempfile.TemporaryDirectory() as root:
                report = run_video_tracking(self.frames(root), self.prompt(), Path(root) / "out", predictor=Predictor())
            self.assertEqual(report["status"], "completed")  # completed computation is not tracking success
            self.assertEqual(report["frames_with_any_mask"], 0)
            row = report["object_coverage"][0]
            self.assertEqual(row["mask_present"], 0)
            self.assertEqual(row["empty_mask" if include_empty_mask else "not_returned"], 3)
            self.assertIsNone(row["first_mask_frame_index"])
            self.assertIsNone(row["last_mask_frame_index"])

    def test_text_prompt_retains_model_assigned_instance_ids(self):
        with tempfile.TemporaryDirectory() as root:
            predictor = FakePredictor()
            report = run_video_tracking(self.frames(root), [{"frame_index": 0, "text": "gripper"}],
                                        Path(root) / "out", predictor=predictor)
            self.assertEqual(predictor.requests[1]["text"], "gripper")
            self.assertNotIn("obj_id", predictor.requests[1])
            self.assertEqual(report["frames"][0]["objects"][0]["object_id"], 7)

    def test_partial_failure_saves_observed_outputs_closes_and_leaves_unknown(self):
        with tempfile.TemporaryDirectory() as root:
            predictor = FakePredictor(fail=True)
            report = run_video_tracking(self.frames(root), self.prompt(), Path(root) / "out", predictor=predictor)
            self.assertEqual(report["status"], "failed")
            self.assertTrue(report["session_closed"])
            self.assertEqual(report["frames"][0]["status"], "mask_output")
            self.assertEqual(report["frames"][2]["status"], "unknown")
            self.assertEqual(report["frames"][2]["objects"][0]["mask_visibility"], "unknown")
            self.assertEqual(report["object_coverage"][0]["unknown"], 2)
            self.assertEqual(report["object_coverage"][0]["last_mask_frame_index"], 0)
            self.assertTrue((Path(root) / "out/tracking.json").exists())

    def test_missing_frame_is_incomplete_not_invisible_object(self):
        with tempfile.TemporaryDirectory() as root:
            report = run_video_tracking(self.frames(root), self.prompt(), Path(root) / "out",
                                        predictor=FakePredictor(omit=(2,)))
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["frames"][2]["objects"][0]["mask_visibility"], "unknown")

    def test_changed_source_or_interrupt_preserves_evidence_and_closes(self):
        for mode in ("changed", "interrupt"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as root:
                frames = self.frames(root)
                class Predictor(FakePredictor):
                    def handle_stream_request(self, request):
                        yield from super().handle_stream_request(request)
                        if mode == "interrupt":
                            raise KeyboardInterrupt()
                        Image.new("RGB", (8, 6), "red").save(frames / "000002.png")
                predictor = Predictor()
                report = run_video_tracking(frames, self.prompt(), Path(root) / "out", predictor=predictor)
                self.assertEqual(report["status"], "cancelled" if mode == "interrupt" else "failed")
                self.assertTrue(report["session_closed"])
                self.assertTrue((Path(root) / "out/tracking.json").exists())

    def test_malformed_mask_and_cleanup_failures_are_not_success(self):
        for predictor in (FakePredictor(malformed=True), FakePredictor(cleanup=False),
                          FakePredictor(cleanup=RuntimeError("close failed"))):
            with self.subTest(predictor=predictor), tempfile.TemporaryDirectory() as root:
                report = run_video_tracking(self.frames(root), self.prompt(), Path(root) / "out", predictor=predictor)
                self.assertEqual(report["status"], "failed")
                self.assertEqual(predictor.requests[-1]["type"], "close_session")

    def test_invalid_prompt_fails_before_model_or_session_and_no_input_overwrite(self):
        prompts = [[], [{"frame_index": 0, "text": "gripper", "object_id": 7}],
                   [{"frame_index": 0, "object_id": 7, "points_px": [[8, 3]], "point_labels": [1]}],
                   [{"frame_index": 9, "text": "gripper"}]]
        with tempfile.TemporaryDirectory() as root:
            frames = self.frames(root)
            for prompt in prompts:
                predictor = FakePredictor()
                with self.assertRaises(ValueError):
                    run_video_tracking(frames, prompt, Path(root) / "out", predictor=predictor)
                self.assertFalse(predictor.requests)
                self.assertFalse((Path(root) / "out").exists())
            with self.assertRaises(FileExistsError):
                run_video_tracking(frames, self.prompt(), frames, predictor=FakePredictor())
            with patch("services.vggt_height.sam31.video_tracking.build_predictor") as builder:
                with self.assertRaises(ValueError):
                    run_video_tracking(frames, self.prompt(), Path(root) / "out")
                builder.assert_not_called()

    def test_gaps_or_duplicate_filenames_and_mismatched_sizes_rejected(self):
        for variant in ("gap", "duplicate", "size"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as root:
                frames = self.frames(root)
                if variant == "gap":
                    (frames / "000001.png").unlink()
                elif variant == "duplicate":
                    (frames / "0.png").write_bytes((frames / "000000.png").read_bytes())
                else:
                    Image.new("RGB", (9, 6)).save(frames / "000001.png")
                predictor = FakePredictor()
                with self.assertRaises(ValueError):
                    run_video_tracking(frames, self.prompt(), Path(root) / "out", predictor=predictor)
                self.assertFalse(predictor.requests)

    def test_explicit_builder_uses_pinned_multiplex_api_without_default_download(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint = Path(root) / "sam3.1_multiplex.pt"; checkpoint.touch()
            package = types.ModuleType("sam3"); package.__file__ = str(Path(root) / "sam3/__init__.py")
            builder_module = types.ModuleType("sam3.model_builder")
            calls = []
            def factory(**kwargs):
                calls.append(kwargs)
                return "fake-built-model"
            builder_module.build_sam3_multiplex_video_predictor = factory
            with patch.dict("sys.modules", {"sam3": package, "sam3.model_builder": builder_module}), \
                    patch("services.vggt_height.sam31.video_tracking.subprocess.run", side_effect=[
                        types.SimpleNamespace(stdout=SAM3_REVISION + "\n"), types.SimpleNamespace(stdout="")]):
                result = build_predictor(checkpoint)
            self.assertEqual(result, "fake-built-model")
            self.assertEqual(calls[0]["checkpoint_path"], str(checkpoint.resolve()))
            self.assertFalse(calls[0]["compile"])
            self.assertFalse(calls[0]["warm_up"])
            self.assertFalse(calls[0]["async_loading_frames"])
            with patch.dict("sys.modules", {"sam3": package}), \
                    patch("services.vggt_height.sam31.video_tracking.subprocess.run",
                          return_value=types.SimpleNamespace(stdout="different-revision")):
                with self.assertRaises(ValueError):
                    build_predictor(checkpoint)


if __name__ == "__main__":
    unittest.main()
