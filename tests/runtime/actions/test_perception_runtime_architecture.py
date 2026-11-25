"""Architecture, layering, collaborator, and embodiment-parameterization regression tests for Stages 3 & 4."""
from __future__ import annotations

import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from dataclasses import replace
from ni_rua.core.config import Config
from ni_rua.core.embodiment import (
    EmbodimentProfile,
    FrozenDict,
    RobotDescription,
)
from ni_rua.core.enrichment import Enrichment, EnrichmentRequest
from ni_rua.core.episode_state import EpisodeStateLedger, StateSlot
from ni_rua.core.storage.store import Store
from ni_rua.perception.external.obs3d import Obs3dEnricher
from ni_rua.perception.external.scene3d import copy_workspace_modules, workspace_modules
from ni_rua.perception.focus_3d import project_sphere_to_camera
from ni_rua.perception.world_model import WorldObservation3D, get_world_observation, record_world_observation
from ni_rua.robot.embodiments.so101.profile import ARM_JOINTS as SO101_ARM_JOINTS, GRIPPER_JOINT as SO101_JAW, SO101
from ni_rua.robot.model.embodiments import arx_x5_profile
from ni_rua.robot.model.kinematics import IKOptions, IKTarget, KinematicChainSpec, chain_from_manifest, fk_jacobian, solve_ik
from ni_rua.robot.motion.batch_scheduler import ArmLayout, ArmSlot, BatchScheduler
from ni_rua.robot.motion.reachability import Serial6RChainSpec, resolve_6r_chain_spec
from ni_rua.runtime.actions.rollout_tracker import ProcessRolloutTracker
from ni_rua.runtime.decision_context import DecisionContextBridge, REMINDER_IN_COMPACT_REQUEST, REMINDER_IN_TOOL_RESULT
from ni_rua.runtime.harness import Harness
from ni_rua.runtime.observation_pipeline import ObservationPipeline, _action_focus_roi
from ni_rua.runtime.tools import builtins
from ni_rua.runtime.tools.context import ToolAvailability, ToolContext, runtime_port
from ni_rua.runtime.tools.plugins.robotic_view import _unavailable as robotic_view_unavailable, make_robotic_view_tool, read_robotic_view
from support.host_capabilities import expect, program_isolation
from support.paths import REPO_ROOT

NI_RUA_TEST_TIER = "component"


NI_RUA_ROOT = REPO_ROOT / "ni_rua"

LAYER_RANK = {
    "core": 0,
    "robot": 1,
    "perception": 2,
    "cortex": 3,
    "runtime": 4,
    "apps": 5,
}


def _custom_7dof_profile() -> EmbodimentProfile:
    base = arx_x5_profile()
    gripper = replace(
        base.gripper,
        empty_close_aperture_m=0.012,
        full_open_aperture_m=0.08,
        grasp_reference_offset_m=0.105,
        fingertip_beyond_reference_m=0.010,
    )
    return replace(
        base,
        robot=RobotDescription(robot_model_id="panda_custom", frame_id="panda_base", tcp_config_id="panda-hand-v1"),
        tcp_link="panda_hand",
        tcp_label="panda_hand flange",
        arms=("main",),
        max_simultaneous_arms=1,
        arm_joint_names=FrozenDict({"main": tuple(f"j{i}" for i in range(1, 8))}),
        jaw_joint_names=FrozenDict({"main": ("finger_left", "finger_right")}),
        gripper=gripper,
        pose_inputs=frozenset({"tcp_quaternion"}),
        supports=frozenset({"stall_guard"}),
        clock="real_time",
        per_request_step_limit=None,
        cameras=("cam_overview", "cam_main_wrist"),
        overview_camera="cam_overview",
        wrist_cameras=FrozenDict({"main": "cam_main_wrist"}),
        camera_examples=("cam_overview", "cam_main_wrist"),
        handoff=False,
        home_check=None,
        policy_step=False,
    )


class LayeringArchitectureTests(unittest.TestCase):
    def test_zero_upward_imports_across_all_five_lower_layers(self):
        violations: list[str] = []
        for path in sorted(NI_RUA_ROOT.rglob("*.py")):
            rel = path.relative_to(NI_RUA_ROOT).as_posix()
            parts = rel.split("/")
            if len(parts) < 2:
                continue
            src_layer = parts[0]
            if src_layer not in LAYER_RANK or src_layer == "apps":
                continue
            src_rank = LAYER_RANK[src_layer]
            is_pkg = rel.endswith("/__init__.py")
            mod_parts = ["ni_rua", *(p for p in rel[:-3].split("/") if p and p != "__init__")]
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                targets: list[str] = []
                if isinstance(node, ast.Import):
                    targets.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    if node.level == 0:
                        if node.module:
                            targets.append(node.module)
                            targets.extend(f"{node.module}.{a.name}" for a in node.names if a.name != "*")
                    else:
                        anchor = list(mod_parts if is_pkg else mod_parts[:-1])
                        for _ in range(node.level - 1):
                            if anchor:
                                anchor.pop()
                        if node.module:
                            anchor.extend(node.module.split("."))
                        base = ".".join(anchor)
                        targets.append(base)
                        targets.extend(f"{base}.{a.name}" for a in node.names if a.name != "*")
                elif (
                    isinstance(node, ast.Call)
                    and (
                        (isinstance(node.func, ast.Name) and node.func.id == "__import__")
                        or (isinstance(node.func, ast.Attribute) and node.func.attr == "import_module")
                    )
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
                    targets.append(node.args[0].value)

                for target in targets:
                    if not target.startswith("ni_rua."):
                        continue
                    dst_layer = target.split(".")[1]
                    if dst_layer in LAYER_RANK and LAYER_RANK[dst_layer] > src_rank:
                        violations.append(f"{rel}:{getattr(node, 'lineno', 0)}: {src_layer} -> {dst_layer} ({target})")

        self.assertEqual(violations, [], msg="\n".join(violations))


class CoreAndCompatPolishTests(unittest.TestCase):
    def test_import_ni_rua_defers_jsonschema_and_tool_errors_handles_missing_jsonschema(self):
        script = (
            "import sys\n"
            "import ni_rua\n"
            "assert 'jsonschema' not in sys.modules, 'jsonschema imported eagerly'\n"
            "sys.modules['jsonschema'] = None\n"
            "from ni_rua.core.tool_errors import ToolInputError, error_payload\n"
            "payload = error_payload(ToolInputError('bad_field', field_path='/x', actual=1))\n"
            "assert payload['code'] == 'bad_field'\n"
            "generic = error_payload(ValueError('plain_value_error'))\n"
            "assert generic['code'] == 'plain_value_error'\n"
            "print('NO_JSONSCHEMA_OK')\n"
        )
        res = subprocess.run(
            [sys.executable, "-P", "-c", script],
            cwd=str(REPO_ROOT),
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT), "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        self.assertIn("NO_JSONSCHEMA_OK", res.stdout)

    def test_import_ni_rua_skips_missing_hook_providers_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp) / "ni_rua"
            pkg.mkdir(parents=True)
            shutil.copyfile(NI_RUA_ROOT / "__init__.py", pkg / "__init__.py")
            shutil.copytree(NI_RUA_ROOT / "core", pkg / "core")
            x5_defaults = NI_RUA_ROOT / "robot" / "embodiments" / "arx_x5" / "defaults.json"
            if x5_defaults.is_file():
                dst_defaults = pkg / "robot" / "embodiments" / "arx_x5" / "defaults.json"
                dst_defaults.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(x5_defaults, dst_defaults)
            script = (
                "import ni_rua\n"
                "from ni_rua.core.config import Config\n"
                "cfg = Config()\n"
                "assert cfg.features is not None\n"
                "print('MINIMAL_SUBSET_OK')\n"
            )
            res = subprocess.run(
                [sys.executable, "-P", "-c", script],
                cwd=tmp,
                env={**os.environ, "PYTHONPATH": tmp, "PYTHONDONTWRITEBYTECODE": "1"},
                capture_output=True,
                text=True,
                timeout=20,
            )
            self.assertEqual(res.returncode, 0, msg=res.stderr)
            self.assertIn("MINIMAL_SUBSET_OK", res.stdout)

    def test_cortex_and_so101_internal_compat_shims_are_retired(self):
        import importlib.util

        for retired in (
            "ni_rua.cortex._compat",
            "ni_rua.cortex.session.codex",
            "ni_rua.cortex.session.codex_sandbox",
            "ni_rua.cortex.session.model_proxy",
            "ni_rua.robot.model.so101_calibration",
            "ni_rua.robot.model.so101_evidence_config",
            "ni_rua.robot.model.so101_gravity",
        ):
            self.assertIsNone(importlib.util.find_spec(retired), retired)


class PerceptionGeometryAndSandboxTests(unittest.TestCase):
    def test_focus_3d_excludes_behind_camera_and_distortion_cone_outliers(self):
        K = np.array([[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        T_wc = np.eye(4, dtype=np.float64)
        dist = np.array([-0.25, 0.08, 0.0, 0.0, 0.0], dtype=np.float64)

        behind = project_sphere_to_camera([0.0, 0.0, -0.30], 0.05, K, T_wc, 640, 480, D=dist)
        self.assertFalse(behind["in_frustum"])
        self.assertLess(behind["depth_m"], 0.0)

        straddling = project_sphere_to_camera([0.04, 0.0, 0.08], 0.06, K, T_wc, 640, 480, D=dist)
        self.assertTrue(straddling["in_frustum"])
        x, y, w, h = straddling["crop_xywh_px"]
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(x + w, 640)
        self.assertLessEqual(y + h, 480)

    def test_action_focus_roi_respects_visibility_mask_from_geometry_project(self):
        K = [[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]]
        T_wc = np.eye(4, dtype=np.float64).tolist()
        camera = {"camera_id": "cam_head", "width": 640, "height": 480}
        metadata_offscreen = {
            "camera_calibration": {"cameras": {"cam_head": {"K": K, "distortion": [0, 0, 0, 0, 0]}}},
            "camera_pose_sample": {"cameras": {"cam_head": {"T_world_camera": T_wc}}},
            "robot_state": {
                "arms": {
                    "left": {"grasp_pose": {"position_m": [5.0, 0.0, 0.5]}},
                    "right": {"grasp_pose": {"position_m": [-5.0, 0.0, 0.5]}},
                }
            },
        }
        self.assertIsNone(_action_focus_roi(metadata_offscreen, camera, {"stages": []}))

        metadata_one_visible = deepcopy(metadata_offscreen)
        metadata_one_visible["robot_state"]["arms"]["right"]["grasp_pose"]["position_m"] = [0.0, 0.0, 0.5]
        roi = _action_focus_roi(metadata_one_visible, camera, {"stages": []}, patch_px=128)
        self.assertIsNotNone(roi)
        # Centred on right arm [320, 240], unaffected by off-screen left arm at +5 m!
        self.assertEqual(roi, [256, 176, 128, 128])

    def test_copy_workspace_modules_is_standalone_and_sandbox_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            mods = workspace_modules(SimpleNamespace(config=Config.from_dict({"features": {"scene3d": True}})))
            self.assertTrue(copy_workspace_modules(mods, tmp))
            staged_files = {p.name for p in Path(tmp).iterdir() if p.is_file()}
            for expected in ("scene3d_helpers.py", "kinematics.py", "_reachability.py", "_pose_math_core.py", "_x5_kinematics.json"):
                self.assertIn(expected, staged_files)
            script = (
                "import sys\n"
                "import scene3d_helpers as s3\n"
                "assert not any(k.startswith('ni_rua') for k in sys.modules)\n"
                "assert not hasattr(s3, 'PointCloudSpatialField')\n"
                "res = s3.check_reachability('left', [0.0, 0.25, 0.85], [0.0, 0.0, 0.0])\n"
                "assert isinstance(res, dict) and 'reachable' in res\n"
                "print('SANDBOX_HELPERS_OK', res['reachable'])\n"
            )
            proc = subprocess.run(
                [sys.executable, "-P", "-c", script],
                cwd=tmp,
                env={**os.environ, "PYTHONPATH": tmp, "PYTHONDONTWRITEBYTECODE": "1"},
                capture_output=True,
                text=True,
                timeout=20,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertIn("SANDBOX_HELPERS_OK", proc.stdout)

    def test_copy_workspace_modules_non_x5_profile_omits_x5_kinematics_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            mods = workspace_modules(SimpleNamespace(config=Config.from_dict({"features": {"scene3d": True}})))
            self.assertTrue(copy_workspace_modules(mods, tmp, profile=_custom_7dof_profile()))
            staged_files = {p.name for p in Path(tmp).iterdir() if p.is_file()}
            self.assertIn("scene3d_helpers.py", staged_files)
            self.assertIn("_pose_math_core.py", staged_files)
            self.assertNotIn("_x5_kinematics.json", staged_files)
            script = (
                "import sys\n"
                "import scene3d_helpers as s3\n"
                "assert not any(k.startswith('ni_rua') for k in sys.modules)\n"
                "pt = s3.triangulate_rays(\n"
                "    320, 240, [[400,0,320],[0,400,240],[0,0,1]], [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],\n"
                "    300, 240, [[400,0,320],[0,400,240],[0,0,1]], [[1,0,0,0.05],[0,1,0,0],[0,0,1,0],[0,0,0,1]],\n"
                ")\n"
                "assert pt['status'] == 'ok'\n"
                "print('NON_X5_SANDBOX_OK')\n"
            )
            proc = subprocess.run(
                [sys.executable, "-P", "-c", script],
                cwd=tmp,
                env={**os.environ, "PYTHONPATH": tmp, "PYTHONDONTWRITEBYTECODE": "1"},
                capture_output=True,
                text=True,
                timeout=20,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertIn("NON_X5_SANDBOX_OK", proc.stdout)

    def test_active_perception_modules_have_zero_pose_skill_imports(self):
        targets = (
            NI_RUA_ROOT / "perception" / "external" / "object_table.py",
            NI_RUA_ROOT / "perception" / "external" / "obs3d.py",
            NI_RUA_ROOT / "perception" / "external" / "scene3d.py",
            NI_RUA_ROOT / "perception" / "external" / "held_perception.py",
            NI_RUA_ROOT / "runtime" / "actions" / "grasp_preview_3d.py",
        )
        for path in targets:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertNotIn("pose_skill", alias.name, msg=str(path))
                elif isinstance(node, ast.ImportFrom):
                    self.assertNotIn("pose_skill", node.module or "", msg=str(path))
                    for alias in node.names:
                        self.assertNotEqual("pose_skill", alias.name, msg=str(path))

        script = (
            "import sys\n"
            "import ni_rua.perception.external.object_table as ot\n"
            "import ni_rua.perception.external.obs3d as o3d\n"
            "import ni_rua.perception.external.scene3d as s3d\n"
            "import ni_rua.perception.external.held_perception as hp\n"
            "import ni_rua.runtime.actions.grasp_preview_3d as gp\n"
            "ot.grasp_hints_from([{'id':'o1','centre_m':[0.0,0.0,0.78],'top_z_m':0.80,'bottom_z_m':0.76,'length_m':0.12,'width_m':0.03,'yaw_deg':20.0,'axis_tilt_deg':85.0,'elongation':4.0}])\n"
            "hp.held_record(object(), 'right')\n"
            "assert 'ni_rua.perception.external.pose_skill' not in sys.modules\n"
            "print('ZERO_POSE_SKILL_OK')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-P", "-c", script],
            cwd=str(REPO_ROOT),
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT), "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("ZERO_POSE_SKILL_OK", proc.stdout)

    def test_world_observation_3d_shared_across_obs3d_focus3d_and_unified_actions(self):
        from ni_rua.perception.external import object_table as ot, obs3d
        from ni_rua.perception.focus_3d import resolve_object_target
        from ni_rua.runtime.actions.unified_actions import _extract_measured_support_z_m

        rt = SimpleNamespace(episode_state=EpisodeStateLedger("ep-wobs"))
        table = {
            "columns": list(ot.COLUMNS),
            "table_z": 0.764,
            "rows": [["o1", [0.12, -0.08, 0.785], 0.805, 0.764, 0.10, 0.04, 15.0, 88.0, "table", "ok"]],
            "grasp_hints": {"o1": {"cavity_depth_m": 0.025}},
        }
        obs3d._remember(rt, "obs-w1", "object_table", (table, None, {"value": {"camera_id": "cam_head"}, "diagnostics": {"objects": [{"id": "o1"}]}}))
        obs3d._remember(rt, "obs-w1", "scene3d", ({"artifact": "artifact:cloud1", "points": 1200, "table_z": 0.766}, {}))

        wobs = get_world_observation(rt, "obs-w1")
        self.assertIsInstance(wobs, WorldObservation3D)
        self.assertEqual(wobs.cloud_artifact_ref, "artifact:cloud1")
        self.assertAlmostEqual(wobs.cavity_depths_m["o1"], 0.025, places=5)
        self.assertEqual(obs3d.attached_table(rt, "obs-w1")["rows"][0][0], "o1")
        self.assertEqual(ot.attached_table(rt, "obs-w1")["rows"][0][0], "o1")
        target = resolve_object_target(rt, {"observation_id": "obs-w1"}, "o1")
        self.assertEqual(target["centre_m"], [0.12, -0.08, 0.785])
        self.assertAlmostEqual(_extract_measured_support_z_m(rt, "obs-w1"), 0.766, places=5)


class RobotBlindSpotRegressionTests(unittest.TestCase):
    def test_solve_ik_with_non_identity_world_base_matches_world_target(self):
        T_wb = np.eye(4, dtype=np.float64)
        T_wb[:3, 3] = [0.12, -0.08, 0.765]
        T_wb[:3, :3] = Rotation.from_euler("z", 25.0, degrees=True).as_matrix()

        origins = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], 6, axis=0)
        origins[1, :3, 3] = [0.0, 0.0, 0.12]
        origins[2, :3, 3] = [0.22, 0.0, 0.0]
        origins[3, :3, 3] = [0.20, 0.0, 0.0]
        origins[4, :3, 3] = [0.06, 0.0, 0.0]
        origins[5, :3, 3] = [0.05, 0.0, 0.0]
        axes = np.array(
            [[0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0]],
            dtype=np.float64,
        )
        bounds = np.column_stack((np.full(6, -2.8), np.full(6, 2.8)))
        chain = KinematicChainSpec(
            joint_names=tuple(f"j{i}" for i in range(6)),
            joint_types=("revolute",) * 6,
            parent_transforms=origins,
            joint_axes=axes,
            joint_limits_rad=bounds,
            tip_transform=np.eye(4, dtype=np.float64),
            base_link="base",
            tip_link="link6",
            link_names=tuple(f"link{i + 1}" for i in range(6)),
            T_world_base=T_wb,
            rest_posture_rad=np.zeros(6),
        )
        q_true = np.array([0.25, -0.35, 0.55, -0.2, 0.3, 0.4], dtype=np.float64)
        T_world_goal, _ = fk_jacobian(chain, q_true)
        target = IKTarget.from_pose(
            T_world_goal[:3, 3],
            Rotation.from_matrix(T_world_goal[:3, :3]).as_quat(),
            T_world_base=T_wb,
        )
        for opts in (
            IKOptions(tasks=("pose",), damping="adaptive_lm"),
            IKOptions(tasks=("position", "orientation"), damping="adaptive_lm"),
            IKOptions(tasks=("pose",), damping="error_scaled"),
        ):
            sol = solve_ik(chain, target, q_true + 0.08, options=opts)
            self.assertTrue(sol.converged, msg=f"{opts} failed: pos_err={sol.position_error_m}, ori_err={sol.orientation_error_rad}")
            T_world_sol, _ = fk_jacobian(chain, sol.q_rad)
            np.testing.assert_allclose(T_world_sol[:3, 3], T_world_goal[:3, 3], atol=1e-3)

    def test_error_scaled_dls_refines_null_space_posture_from_pose_converged_seed(self):
        origins = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], 7, axis=0)
        for i in range(1, 7):
            origins[i, :3, 3] = [0.12, 0.0, 0.0]
        axes = np.array(
            [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            dtype=np.float64,
        )
        bounds = np.column_stack((np.full(7, -3.0), np.full(7, 3.0)))
        chain = KinematicChainSpec(
            joint_names=tuple(f"j{i}" for i in range(7)),
            joint_types=("revolute",) * 7,
            parent_transforms=origins,
            joint_axes=axes,
            joint_limits_rad=bounds,
            tip_transform=np.eye(4, dtype=np.float64),
            base_link="base",
            tip_link="link7",
            link_names=tuple(f"link{i + 1}" for i in range(7)),
            rest_posture_rad=np.zeros(7),
        )
        q_seed = np.array([0.5, -0.8, 0.3, 0.2, -0.1, 0.15, 0.0], dtype=np.float64)
        T_goal, _ = fk_jacobian(chain, q_seed)
        posture_q = np.zeros(7, dtype=np.float64)
        target = IKTarget(T_base_tip=T_goal, posture_rad=posture_q)
        opts = IKOptions(tasks=("pose", "posture"), damping="error_scaled", posture_weight=0.2, restarts=0)
        sol = solve_ik(chain, target, q_seed, options=opts)
        self.assertTrue(sol.converged)
        self.assertLess(float(np.linalg.norm(sol.q_rad - posture_q)), float(np.linalg.norm(q_seed - posture_q)) - 1e-3)

    def test_chain_from_manifest_folds_omitted_joint_origin_and_reachability_preserves_tcp_and_prismatic(self):
        t_step = np.eye(4, dtype=np.float64)
        t_step[:3, 3] = [0.10, 0.0, 0.0]
        t_cam = np.eye(4, dtype=np.float64)
        t_cam[:3, 3] = [0.05, 0.02, 0.07]
        manifest = {
            "base_link": "base",
            "tcp_link": "link6",
            "joints": [
                {"name": "j0", "type": "prismatic", "parent": "base", "child": "link1", "axis": [0, 0, 1], "T_parent_joint": t_step.tolist(), "limits": {"lower": -0.2, "upper": 0.2}},
                {"name": "j1", "type": "revolute", "parent": "link1", "child": "link2", "axis": [0, 1, 0], "T_parent_joint": t_step.tolist(), "limits": {"lower": -2.0, "upper": 2.0}},
                {"name": "j2", "type": "revolute", "parent": "link2", "child": "link3", "axis": [0, 1, 0], "T_parent_joint": t_step.tolist(), "limits": {"lower": -2.0, "upper": 2.0}},
                {"name": "j3", "type": "revolute", "parent": "link3", "child": "link4", "axis": [1, 0, 0], "T_parent_joint": t_step.tolist(), "limits": {"lower": -2.0, "upper": 2.0}},
                {"name": "j4", "type": "revolute", "parent": "link4", "child": "link5", "axis": [0, 1, 0], "T_parent_joint": t_step.tolist(), "limits": {"lower": -2.0, "upper": 2.0}},
                {"name": "j5", "type": "revolute", "parent": "link5", "child": "link6", "axis": [1, 0, 0], "T_parent_joint": t_step.tolist(), "limits": {"lower": -2.0, "upper": 2.0}},
                {"name": "cam_fixed", "type": "fixed", "parent": "link6", "child": "cam_wrist", "T_parent_joint": t_cam.tolist()},
            ],
        }
        spec = resolve_6r_chain_spec("right", robot_model=manifest)
        self.assertIsInstance(spec, Serial6RChainSpec)
        # Trailing fixed cam_wrist must NOT be folded into link6 tip_transform!
        np.testing.assert_allclose(spec.flange_fixed_transform, np.eye(4), atol=1e-12)
        self.assertEqual(spec.joint_types[0], "prismatic")

        # Overriding joint_limits_rad must preserve prismatic joint_types!
        custom_limits = [(-0.15, 0.15)] + [(-1.5, 1.5)] * 5
        spec_override = resolve_6r_chain_spec("right", robot_model=manifest, joint_limits_rad=custom_limits)
        self.assertEqual(spec_override.to_kinematic_chain().joint_types[0], "prismatic")

        # Omitting j0 from joint_names on a traced path folds j0's T_parent_joint into j1!
        sub_chain = chain_from_manifest(manifest, joint_names=("j1", "j2", "j3", "j4", "j5"), tip_link="link6")
        np.testing.assert_allclose(sub_chain.parent_transforms[0, :3, 3], [0.20, 0.0, 0.0], atol=1e-12)

    def test_batch_scheduler_execute_3arg_and_single_arm_layout_compilation(self):
        import threading
        from support.motion_fixtures import batch, move
        from support.motion_fixtures import preview_adapter

        adapter, _, dispatched = preview_adapter()
        scheduler = BatchScheduler(adapter)
        action = batch([move("m1", arm="left", angle=-0.125)])
        plan = scheduler.preview(action)
        self.assertEqual(plan["status"], "planned")
        receipt = scheduler.execute(dict(action, batch_plan_id=plan["plan_id"]), lambda _: None, threading.Event())
        self.assertEqual(receipt["status"], "completed")
        self.assertGreater(len(dispatched), 0)

        # Also verify single-arm 5-DoF ArmLayout slot compilation directly (zero hardcoded 0/7 offsets)
        single_layout = ArmLayout(slots={"main": ArmSlot(arm="main", joints=slice(0, 5), jaw=5)}, row_width=6)
        self.assertEqual(single_layout.arms, ("main",))
        self.assertEqual(single_layout.slot("main").joints, slice(0, 5))
        self.assertEqual(single_layout.slot("main").jaw, 5)


class RuntimeCollaboratorsAndToolContextTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.identity = dict(run_id="r1", episode_id="ep1", env_instance_id="env1", env_generation=1)
        self.store = Store(self.root / "store", dict(self.identity))
        self.addCleanup(self.store.close)
        from support.fixture_robot import FixtureRobot

        self.robot = FixtureRobot(self.root)
        self.harness = Harness(self.store, self.robot)

    def test_episode_state_ledger_scoping_and_causal_basis_invalidation(self):
        run_slot = StateSlot("run_counter", lambda: {"n": 0}, scope="run")
        ep_slot = StateSlot("ep_items", list, scope="episode")
        ledger = EpisodeStateLedger(self.identity)
        ledger.get(run_slot)["n"] = 7
        ledger.get(ep_slot).append("o1")
        ledger.observation_published("obs-1", make_current=True)
        self.assertTrue(ledger.is_basis("obs-1"))

        # Unconfirmed action breaks causal continuity for obs-1!
        ledger.action_admitted("act-1")
        ledger.action_settled(continuous=False)
        self.assertFalse(ledger.is_basis("obs-1"))

        # Fresh observation after break establishes a new valid basis while obs-1 stays invalid!
        ledger.observation_published("obs-2", make_current=True)
        self.assertFalse(ledger.is_basis("obs-1"))
        self.assertTrue(ledger.is_basis("obs-2"))

        # Episode transition resets episode slots while keeping run slots!
        next_id = {**self.identity, "episode_id": "ep2", "env_generation": 2}
        ledger.begin_episode(next_id)
        self.assertEqual(ledger.get(run_slot)["n"], 7)
        self.assertEqual(ledger.get(ep_slot), [])
        self.assertFalse(ledger.is_basis("obs-2"))

    def test_harness_collaborators_and_zero_monkey_stashing_in_instance_dict(self):
        self.assertIsInstance(self.harness.episode_state, EpisodeStateLedger)
        self.assertIsInstance(self.harness.decision_context, DecisionContextBridge)
        self.assertIsInstance(self.harness.observation_pipeline, ObservationPipeline)
        self.assertEqual(Harness.delivery_channel, "codex_tool")
        self.assertIs(Harness.decision_reminder_channel, REMINDER_IN_COMPACT_REQUEST)

        initial_keys = set(self.harness.__dict__.keys())
        self.harness.delivery_channel = "agent_tool"
        self.harness.decision_reminder_channel = REMINDER_IN_TOOL_RESULT
        self.harness.unified_action_plans["p1"] = {"status": "planned"}
        self.harness._action_state_ledger["entries"] = [{"action_id": "a1"}]
        self.harness._persistent_obstacles["b1"] = {"id": "b1"}
        self.harness._obs3d_memo = ("obs-1", {"object_table": ({}, None, {})})
        self.harness._virtual_spatial_field = {"dummy": True}
        self.assertEqual(set(self.harness.__dict__.keys()), initial_keys)
        self.assertEqual(self.harness.decision_context.delivery_channel, "agent_tool")

    def test_builtins_bind_retired_and_tool_context_enforces_physical_effect_boundary(self):
        self.assertFalse(hasattr(builtins, "_bind"))
        self.harness.start()
        read_ctx = ToolContext.from_runtime(self.harness, "robot_capabilities", "read", {"tool_call_id": "c1"})
        self.assertIsNone(read_ctx.execute)
        self.assertFalse(hasattr(read_ctx, "robot"))
        self.assertFalse(hasattr(read_ctx, "store"))
        self.assertIsNotNone(read_ctx.observations)
        self.assertIsNotNone(read_ctx.records)
        port = runtime_port(read_ctx)
        with self.assertRaisesRegex(PermissionError, "tool_effect_forbids_physical_execution"):
            _ = port.execute
        with self.assertRaisesRegex(PermissionError, "tool_effect_forbids_physical_execution"):
            _ = port.policy_step

    def test_obs3d_enricher_integrates_with_observation_pipeline_without_upward_imports(self):
        obs = self.harness.start()
        enricher = Obs3dEnricher()
        req = EnrichmentRequest(observation=obs, correlation={"tool_call_id": "t1"}, audience="actor", reply="observe")
        res = enricher.enrich(self.harness, req)
        self.assertIsInstance(res, Enrichment)
        self.assertEqual(res.parts, ())

    def test_robotic_view_generalized_to_non_x5_7dof_embodiment_and_rejects_so101(self):
        so101_prof = replace(
            arx_x5_profile(),
            robot=SO101,
            tcp_link="gripper_frame_link",
            arms=("main",),
            max_simultaneous_arms=1,
            arm_joint_names=FrozenDict({"main": SO101_ARM_JOINTS}),
            jaw_joint_names=FrozenDict({"main": (SO101_JAW,)}),
            pose_inputs=frozenset({"tcp_quaternion"}),
            supports=frozenset({"stall_guard"}),
            wrist_cameras=FrozenDict({"main": "wrist"}),
            cameras=("fixed", "wrist"),
            camera_examples=("fixed", "wrist"),
            overview_camera="fixed",
            handoff=False,
            home_check=None,
            policy_step=False,
        )
        so_avail = ToolAvailability(
            Config().features, frozenset(), "rgb_proprio_calibrated_v1", None,
            False, False, False, False, True, embodiment=so101_prof,
        )
        self.assertEqual(robotic_view_unavailable(so_avail, {}), "robotic_view_requires_arx_x5_link6_wrist_camera")

        panda = _custom_7dof_profile()
        panda_avail = ToolAvailability(
            Config().features, frozenset(), "rgb_proprio_calibrated_v1", {"tcp_config_id": "panda-hand-v1"},
            False, False, False, False, True, embodiment=panda,
        )
        self.assertIsNone(robotic_view_unavailable(panda_avail, {}, profile=panda))

        store = Store(self.root / "panda_store", dict(self.identity))
        self.addCleanup(store.close)
        robot = SimpleNamespace(
            embodiment_profile=lambda: panda,
            observe=Mock(side_effect=AssertionError("no observe")),
            execute=Mock(side_effect=AssertionError("no execute")),
        )
        cfg = Config.from_dict({"features": {"observation_profile": "rgb_proprio_calibrated_v1", "motion_review": False}})
        holder: dict[str, Harness] = {}
        spec = make_robotic_view_tool(lambda args, corr: read_robotic_view(holder["h"], args, corr), profile=panda)
        h = Harness(store, robot, config=cfg, tool_specs=(spec,))
        holder["h"] = h
        h.capabilities = {"frame_id": "panda_base", "tcp_config_id": "panda-hand-v1"}

        T_hand = np.eye(4, dtype=np.float64)
        T_hand[:3, 3] = [0.35, 0.05, 0.42]
        T_mount = np.eye(4, dtype=np.float64)
        T_mount[:3, 3] = [0.04, 0.0, 0.02]
        T_cam = T_hand @ T_mount
        model_doc = {
            "base_link": "panda_base",
            "joints": [
                {"name": "j_hand", "type": "fixed", "parent": "panda_base", "child": "panda_hand", "T_parent_joint": T_hand.tolist()},
                {"name": "j_cam", "type": "fixed", "parent": "panda_hand", "child": "cam_link", "T_parent_joint": T_mount.tolist()},
            ],
        }
        cal_doc = {
            "reference_frame_id": "panda_base",
            "cameras": {
                "cam_main_wrist": {
                    "arm": "main",
                    "link_name": "cam_link",
                    "T_link_camera": np.eye(4).tolist(),
                    "T_world_base": np.eye(4).tolist(),
                    "K": [[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]],
                    "distortion": [],
                    "image_size": [64, 64],
                }
            },
        }
        state_doc = {
            "sim_step": 1,
            "step_domain": "control_step",
            "arms": {
                "main": {
                    "source_kind": "measurement",
                    "q_rad": [0.0] * 7,
                    "joint_names": [f"j{i}" for i in range(1, 8)],
                    "ee_pose": {
                        "source_kind": "measurement",
                        "frame_id": "panda_base",
                        "tcp_config_id": "panda-hand-v1",
                        "position_m": T_hand[:3, 3].tolist(),
                        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                    "gripper": {
                        "joint_names": ["finger_left", "finger_right"],
                        "position_m": [0.04, 0.04],
                        "position_source_kind": "measurement",
                        "command_normalized": 1.0,
                        "command_source_kind": "command",
                    },
                }
            },
        }
        pose_doc = {
            "frame_id": "f1",
            "T_world_camera": T_cam.tolist(),
            "T_world_link": T_cam.tolist(),
            "T_world_base": np.eye(4).tolist(),
            "source_kind": "robot_kinematics",
            "sim_step": 1,
            "step_domain": "control_step",
        }
        _, png = cv2.imencode(".png", np.zeros((64, 64, 3), np.uint8))
        put_json = lambda d, k: store.put_artifact(json.dumps(d).encode(), k, "application/json", "actor")
        bundle = {
            "schema_version": "0.1",
            **self.identity,
            "world_kind": "observed",
            "observation_id": "obs-panda-1",
            "observation_profile": "rgb_proprio_calibrated_v1",
            "sim_step": 1,
            "step_domain": "control_step",
            "clock_domain": "env1/generation-1",
            "calibration_id": "cal-panda",
            "robot_model_id": "panda_custom",
            "robot_model_ref": put_json(model_doc, "robot_model"),
            "robot_state_ref": put_json(state_doc, "robot_state"),
            "cameras": [
                {
                    "camera_id": "cam_main_wrist",
                    "frame_id": "f1",
                    "width": 64,
                    "height": 64,
                    "sim_step": 1,
                    "step_domain": "control_step",
                    "calibration_ref": put_json(cal_doc, "calibration"),
                    "pose_sample_ref": put_json(pose_doc, "camera_pose"),
                    "rgb_ref": store.put_artifact(png.tobytes(), "rgb", "image/png"),
                    "pixel_transform": "identity",
                }
            ],
        }
        store.save("observation", "obs-panda-1", bundle)
        h.scene.import_observation(bundle)
        h.latest = bundle

        reply = h.invoke(
            "robotic_view",
            {
                "observation_id": "obs-panda-1",
                "camera_id": "cam_main_wrist",
                "eye_position_m": [0.35, 0.05, 0.50],
                "target_position_m": [0.35, 0.05, 0.10],
                "up_vector": [1.0, 0.0, 0.0],
            },
            {"tool_call_id": "panda-view-1"},
            role="actor",
        )
        self.assertTrue(reply.success, msg=reply.contract_error)
        self.assertEqual(reply.data["tcp_config_id"], "panda-hand-v1")
        self.assertEqual(reply.data["frame_id"], "panda_base")
        self.assertEqual(reply.data["robot_action_preview"]["arguments"]["stages"][0]["arm"], "main")

    def test_process_rollout_tracker_scales_slip_thresholds_from_empty_close_aperture(self):
        def make_trace(arm_name: str, apertures: list[float]) -> tuple[dict, list[dict]]:
            samples = [
                {
                    "sim_step": idx,
                    "state": {
                        "sim_step": idx,
                        "arms": {
                            arm_name: {
                                "stage_id": "carry",
                                "ee_pose": {
                                    "position_m": [0.20 + 0.02 * idx, 0.0, 0.80],
                                    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                                },
                                "gripper": {
                                    "position_source_kind": "measurement",
                                    "position_m": [ap / 2.0, ap / 2.0],
                                    "command_normalized": 0.0,
                                },
                            }
                        },
                    },
                }
                for idx, ap in enumerate(apertures)
            ]
            return samples[0]["state"], samples[1:]

        # Aperture starts holding at 20.0 mm and collapses to 12.5 mm during tight-closed transport:
        # - For panda (empty_close_aperture_m = 12.0 mm), 12.5 mm < 12.0 + 1.2 = 13.2 mm -> slip_detected is True!
        # - For X5 (empty_close_aperture_m = 0.0 mm), 12.5 mm >= 0.0 + 1.2 = 1.2 mm -> slip_detected is False!
        panda = _custom_7dof_profile()
        action_panda = {"type": "BatchAction", "stages": [{"id": "carry", "kind": "move", "arm": "main"}]}
        tracker_panda = ProcessRolloutTracker(action=action_panda, runtime=SimpleNamespace(embodiment=panda, config=Config()))
        before_panda, trace_panda = make_trace("main", [0.020, 0.020, 0.0125, 0.0125])
        traj_panda = tracker_panda.finalize(
            before_state=before_panda,
            actual_trace=trace_panda,
            planned_trace=None,
            receipt={"status": "completed"},
            start_observation=None,
            terminal_observation=None,
            store=None,
        )
        self.assertTrue(traj_panda["co_motion"]["main"]["slip_detected"])

        action_x5 = {"type": "BatchAction", "stages": [{"id": "carry", "kind": "move", "arm": "left"}]}
        tracker_x5 = ProcessRolloutTracker(action=action_x5, runtime=SimpleNamespace(embodiment=None, config=Config()))
        before_x5, trace_x5 = make_trace("left", [0.020, 0.020, 0.0125, 0.0125])
        traj_x5 = tracker_x5.finalize(
            before_state=before_x5,
            actual_trace=trace_x5,
            planned_trace=None,
            receipt={"status": "completed"},
            start_observation=None,
            terminal_observation=None,
            store=None,
        )
        self.assertFalse(traj_x5["co_motion"]["left"]["slip_detected"])

    def test_depth_geometry_forwards_non_env_origin_profile_frame_id(self):
        from io import BytesIO
        from ni_rua.core.contracts import canonical
        from ni_rua.perception.depth.depth_geometry import DepthGeometryTools
        from ni_rua.perception.scene.store import SceneStore

        panda = _custom_7dof_profile()
        self.assertEqual(panda.frame_id, "panda_base")
        ident = {
            "run_id": "run-depth-panda",
            "episode_id": "ep-1",
            "env_instance_id": "env-1",
            "env_generation": 1,
        }
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = Store(Path(tmp.name) / "store", ident)
        self.addCleanup(store.close)
        scene = SceneStore(Path(tmp.name) / "scene", store.resolve_artifact, store.put_artifact)
        tools = DepthGeometryTools(scene, profile=panda)
        self.assertEqual(tools.geometry.frame_id, "panda_base")

        width, height = 64, 48
        depth = np.ones((height, width), dtype=np.float32) * 1.25
        valid = np.ones((height, width), dtype=bool)
        stream = BytesIO()
        np.savez_compressed(stream, depth=depth, valid=valid)
        depth_ref = store.put_artifact(stream.getvalue(), "depth", "application/x-npz")
        _, png = cv2.imencode(".png", np.full((height, width, 3), 120, dtype=np.uint8))
        rgb_ref = store.put_artifact(png.tobytes(), "rgb", "image/png")

        # Omit top-level reference_frame_id so GeometryTools.camera falls back to profile.frame_id ("panda_base").
        calibration = {
            "cameras": {
                "cam_overview": {
                    "K": [[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]],
                    "distortion": {"model": "opencv", "coefficients": []},
                    "image_size": [width, height],
                }
            }
        }
        cal_ref = store.put_artifact(canonical(calibration).encode(), "calibration", "application/json")
        state = {
            "source_kind": "measurement",
            "sim_step": 1,
            "step_domain": "control_step",
            "arms": {},
            "camera_poses": {
                "cam_overview": {
                    "T_world_camera": np.eye(4).tolist(),
                    "frame_id": "depth-f1",
                    "sim_step": 1,
                    "step_domain": "control_step",
                    "source_kind": "measurement",
                }
            },
        }
        state_ref = store.put_artifact(canonical(state).encode(), "robot_state", "application/json")
        obs = {
            "schema_version": "0.1",
            **ident,
            "world_kind": "observed",
            "data_origin": "synthetic_fixture",
            "observation_id": "obs-depth-panda-1",
            "sim_step": 1,
            "step_domain": "control_step",
            "clock_domain": "panda/gen-1",
            "calibration_id": "panda-cal-1",
            "observation_profile": "rgbd_proprio_calibrated_v1",
            "robot_state_ref": state_ref,
            "cameras": [
                {
                    "camera_id": "cam_overview",
                    "frame_id": "depth-f1",
                    "width": width,
                    "height": height,
                    "sim_step": 1,
                    "step_domain": "control_step",
                    "rgb_ref": rgb_ref,
                    "calibration_ref": cal_ref,
                    "pixel_transform": "identity",
                    "depth_ref": depth_ref,
                    "depth_kind": "z",
                    "depth_unit": "m",
                    "depth_clip_m": [0.01, 10.0],
                    "depth_shape": [height, width],
                }
            ],
        }
        scene.import_observation(obs)

        res = tools.execute(
            {
                **ident,
                "kind": "depth_points",
                "observation_id": "obs-depth-panda-1",
                "camera_id": "cam_overview",
                "pixels": [[32, 24]],
            }
        )
        self.assertEqual(res["status"], "supported")
        self.assertEqual(res["frame_id"], "panda_base")
        np.testing.assert_allclose(res["value"]["points"][0]["point_m"], [0.0, 0.0, 1.25], atol=1e-6)

        with self.assertRaisesRegex(ValueError, "depth_requires_env_origin"):
            tools.execute(
                {
                    **ident,
                    "kind": "depth_points",
                    "observation_id": "obs-depth-panda-1",
                    "camera_id": "cam_overview",
                    "frame_id": "env_origin",
                    "pixels": [[32, 24]],
                }
            )

    def test_program_runner_profile_staging_and_custom_6r_kinematics_handling(self):
        """回归验证 #152 §1.2: ProgramRunner 传递 profile，无 physical.json 的非 X5 6R 跳过 ik_prefilter，有 facts 时统一读写 _kinematics.json。"""
        from unittest.mock import patch
        from ni_rua.core.observation_policy import ACTOR_PROFILE
        from ni_rua.robot.embodiments.arx_x5.profile import physical_facts
        from ni_rua.runtime.programs.runner import ProgramRunner, ProgramServices

        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        ident = dict(run_id="run-stage-p2", episode_id="ep-1", env_instance_id="env-1", env_generation=1)
        store = Store(Path(temp_dir.name) / "run", ident)
        self.addCleanup(store.close)

        registry = builtins.create_registry()
        services = ProgramServices(
            store=store,
            validate_tool=lambda name: registry.spec(name, role="program"),
            validate_arguments=lambda name, arguments: registry.validate(name, arguments, role="program"),
            invoke=lambda name, arguments, correlation: {"success": True, "contentItems": []},
            effect=lambda name, arguments: registry.spec(name, role="program").effect_for(arguments),
            observation=lambda: {"observation_id": None, "observation_profile": ACTOR_PROFILE},
            action_active=lambda identifier: False,
            cancel_action=lambda identifier: False,
            sandbox_enabled=expect(program_isolation, "program_run runs under the Landlock + seccomp sandbox"),
            workspace_modules=("scene3d_helpers.py",),
            embodiment=_custom_7dof_profile(),
        )
        runner = ProgramRunner(services)
        self.addCleanup(runner.close)
        res = runner.run(
            {
                "source": "import json, os\nimport scene3d_helpers as s3\nprint(json.dumps(sorted(os.listdir('.'))))\n",
                "timeout_s": 20,
            },
            {"thread_id": "t-prof", "turn_id": "u-1", "tool_call_id": "c-1"},
        )
        self.assertEqual(res["status"], "succeeded", res.get("stderr_text"))
        ws_files = set(json.loads(res["stdout_text"]))
        self.assertIn("scene3d_helpers.py", ws_files)
        self.assertNotIn("_x5_kinematics.json", ws_files)
        self.assertNotIn("ik_prefilter.py", ws_files)

        # Custom non-X5 6R profile WITHOUT packaged physical.json -> omits ik_prefilter & raises clean RuntimeError
        custom_6r = replace(
            arx_x5_profile(),
            robot=RobotDescription(robot_model_id="ur5e-6r", frame_id="ur_base", tcp_config_id="ur5e-tcp-v1"),
            pose_inputs=frozenset({"grasp_angles", "tcp_quaternion"}),
        )
        mods = workspace_modules(SimpleNamespace(config=Config.from_dict({"features": {"scene3d": True}})))
        ws_no_facts = Path(temp_dir.name) / "ws_ur5e_no_facts"
        ws_no_facts.mkdir()
        self.assertTrue(copy_workspace_modules(mods, ws_no_facts, profile=custom_6r))
        staged_no_facts = {p.name for p in ws_no_facts.iterdir() if p.is_file()}
        self.assertNotIn("ik_prefilter.py", staged_no_facts)
        self.assertNotIn("_x5_kinematics.json", staged_no_facts)
        proc_no_facts = subprocess.run(
            [
                sys.executable,
                "-P",
                "-c",
                (
                    "import scene3d_helpers as s3\n"
                    "try:\n"
                    "    s3.check_reachability('left', [0.2, 0.1, 0.8], [0, 0, 0])\n"
                    "except RuntimeError as exc:\n"
                    "    assert 'not staged for this embodiment profile' in str(exc), exc\n"
                    "    print('CLEAN_RUNTIME_ERROR')\n"
                ),
            ],
            cwd=str(ws_no_facts),
            env={**os.environ, "PYTHONPATH": str(ws_no_facts), "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(proc_no_facts.returncode, 0, msg=proc_no_facts.stderr)
        self.assertIn("CLEAN_RUNTIME_ERROR", proc_no_facts.stdout)

        # Custom non-X5 6R profile WITH packaged physical.json -> stages _kinematics.json and _reachability reads it
        x5_facts = deepcopy(physical_facts("arx_x5"))
        ws_with_facts = Path(temp_dir.name) / "ws_ur5e_with_facts"
        ws_with_facts.mkdir()
        with patch("ni_rua.robot.embodiments.arx_x5.profile.physical_facts", return_value=x5_facts):
            self.assertTrue(copy_workspace_modules(mods, ws_with_facts, profile=custom_6r))
        staged_with_facts = {p.name for p in ws_with_facts.iterdir() if p.is_file()}
        self.assertIn("_kinematics.json", staged_with_facts)
        self.assertNotIn("_x5_kinematics.json", staged_with_facts)
        proc_with_facts = subprocess.run(
            [
                sys.executable,
                "-P",
                "-c",
                (
                    "import scene3d_helpers as s3\n"
                    "chk = s3.check_reachability('left', [0.0, 0.25, 0.85], [0.0, 0.0, 0.0])\n"
                    "assert isinstance(chk, dict) and 'reachable' in chk\n"
                    "print('CUSTOM_6R_KIN_OK')\n"
                ),
            ],
            cwd=str(ws_with_facts),
            env={**os.environ, "PYTHONPATH": str(ws_with_facts), "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(proc_with_facts.returncode, 0, msg=proc_with_facts.stderr)
        self.assertIn("CUSTOM_6R_KIN_OK", proc_with_facts.stdout)

    def test_world_observation_3d_surface_slices_are_deeply_immutable(self):
        """回归验证 #152 §2.5: WorldObservation3D 与 ObjectRow3D 的 surface_slices_m 深度冻结，禁止原位篡改缓存快照。"""
        from ni_rua.perception.external.pointcloud_field import PointCloudSpatialField

        rt = SimpleNamespace(episode_state=EpisodeStateLedger())
        table = {
            "columns": ["id", "centre_m", "top_z", "bottom_z", "length", "width", "yaw_deg", "tilt_deg", "support", "quality"],
            "table_z": 0.765,
            "rows": [["o1", [0.20, 0.00, 0.805], 0.845, 0.765, 0.080, 0.080, 0.0, 89.0, "table", "ok"]],
        }
        raw_objs = [
            {
                "id": "o1",
                "cavity_depth_m": 0.060,
                "surface_slices_m": [
                    {"z_m": 0.780, "hull_xy_m": [[0.16, -0.04], [0.24, -0.04], [0.24, 0.04], [0.16, 0.04]]},
                    {"z_m": 0.840, "hull_xy_m": np.array([[0.16, -0.04], [0.24, -0.04], [0.24, 0.04], [0.16, 0.04]])},
                ],
            }
        ]
        wobs = record_world_observation(rt, WorldObservation3D.from_parts("obs-imm-1", table=table, raw_objects=raw_objs))
        row = wobs.row("o1")
        self.assertIsNotNone(row)
        self.assertIsInstance(hash(row), int)
        self.assertIsInstance(hash(wobs.surface_slices_m), int)

        with self.assertRaises(TypeError):
            wobs.surface_slices_m["o1"] = ()  # type: ignore[index]
        with self.assertRaises(TypeError):
            row.surface_slices_m[0]["z_m"] = -999.0  # type: ignore[index]
        with self.assertRaises(TypeError):
            row.surface_slices_m[0]["hull_xy_m"][0] = (99.0, 99.0)  # type: ignore[index]
        with self.assertRaises(TypeError):
            row.surface_slices_m[0]["hull_xy_m"][0][0] = 99.0  # type: ignore[index]

        cached = get_world_observation(rt, "obs-imm-1")
        self.assertEqual(cached.row("o1").surface_slices_m[0]["z_m"], 0.780)
        self.assertEqual(cached.raw_objects()[0]["surface_slices_m"][0]["z_m"], 0.780)
        pcf = PointCloudSpatialField(objects=table, surface_slices_m=dict(cached.surface_slices_m), table_z=0.765)
        self.assertIn("o1", pcf._obj_slice_trees)

    def test_rollout_tracker_forwards_custom_profile_approach_axis_layout_and_self_store(self):
        """回归验证 #152 §2.10: ProcessRolloutTracker.finalize 保留 self.store 并透传非 X5 approach_axis_local 与单臂 7-DoF ArmLayout。"""
        from ni_rua.core.contracts import canonical
        from ni_rua.runtime.actions.rollout_tracker import _extract_planned_step_map, _extract_tcp_and_angles

        panda_z_prof = replace(
            _custom_7dof_profile(),
            gripper=replace(_custom_7dof_profile().gripper, approach_axis_local=(0.0, 0.0, 1.0), grasp_reference_offset_m=0.105),
        )
        # TCP at [0.2, -0.1, 0.5] with identity orientation and +Z approach_axis_local -> grasp centre at z=0.605 (NOT x=0.305)
        tcp_fk, _ = _extract_tcp_and_angles(
            {"link6_pose": {"position_m": [0.20, -0.10, 0.50], "rotation_matrix": np.eye(3).tolist()}},
            profile=panda_z_prof,
        )
        np.testing.assert_allclose(tcp_fk, [0.20, -0.10, 0.605], atol=1e-6)
        tcp_ee, _ = _extract_tcp_and_angles(
            {"ee_pose": {"position_m": [0.20, -0.10, 0.50], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0], "tcp_config_id": "panda-hand-v1"}},
            profile=panda_z_prof,
        )
        np.testing.assert_allclose(tcp_ee, [0.20, -0.10, 0.605], atol=1e-6)

        # Single-arm 7-DoF planned_trace row has 8 columns (7 joints + jaw at index 7)
        step_map = _extract_planned_step_map(
            None,
            {
                "actions": [[0.0] * 7 + [0.05], [0.1] * 7 + [0.05]],
                "row_contexts": [{"main": {"stage_id": "grasp", "phase": "gripper"}}, {"main": {"stage_id": "lift", "phase": "motion"}}],
            },
            arms_order=("main",),
            profile=panda_z_prof,
        )
        self.assertAlmostEqual(step_map[1]["main"]["jaw_goal_normalized"], 0.05, places=6)
        self.assertEqual(step_map[2]["main"]["stage_id"], "lift")

        # ProcessRolloutTracker.finalize retains self.store when store=None and resolves terminal robot_state_ref with profile
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        ident = dict(run_id="run-rt-store", episode_id="ep-1", env_instance_id="env-1", env_generation=1)
        store = Store(Path(temp_dir.name) / "run", ident)
        self.addCleanup(store.close)

        img0 = np.zeros((120, 160, 3), dtype=np.uint8)
        img1 = np.zeros((120, 160, 3), dtype=np.uint8)
        img0[52:68, 72:88] = 255
        img1[52:68, 78:94] = 255
        rgb0_ref = store.put_artifact(cv2.imencode(".png", img0)[1].tobytes(), "rgb", "image/png")
        rgb1_ref = store.put_artifact(cv2.imencode(".png", img1)[1].tobytes(), "rgb", "image/png")
        cal_ref = store.put_artifact(
            canonical({"cameras": {"cam_overview": {"K": [[120.0, 0.0, 80.0], [0.0, 120.0, 60.0], [0.0, 0.0, 1.0]]}}}).encode(),
            "calibration",
            "application/json",
        )
        pose_ref = store.put_artifact(
            canonical({"cameras": {"cam_overview": {"T_world_camera": np.eye(4).tolist()}}}).encode(),
            "pose_sample",
            "application/json",
        )
        term_state_ref = store.put_artifact(
            canonical(
                {
                    "arms": {
                        "main": {
                            "ee_pose": {"position_m": [0.0, 0.0, 0.395], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0], "tcp_config_id": "panda-hand-v1"},
                            "gripper": {"position_m": [0.012, 0.012], "goal_normalized": 0.0},
                        }
                    }
                }
            ).encode(),
            "robot_state",
            "application/json",
        )
        start_obs = {
            "observation_id": "obs-s",
            "cameras": [{"camera_id": "cam_overview", "calibration_ref": cal_ref, "pose_sample_ref": pose_ref, "rgb_ref": rgb0_ref}],
        }
        term_obs = {
            "observation_id": "obs-t",
            "robot_state_ref": term_state_ref,
            "cameras": [{"camera_id": "cam_overview", "calibration_ref": cal_ref, "pose_sample_ref": pose_ref, "rgb_ref": rgb1_ref}],
        }
        tracker = ProcessRolloutTracker(
            action={"stages": [{"id": "grasp", "arm": "main", "kind": "move", "gripper_goal_normalized": 0.0}]},
            store=store,
            profile=panda_z_prof,
        )
        pt = tracker.finalize(
            before_state={"sim_step": 0, "arms": {"main": {"q_rad": [0.0] * 7, "gripper": {"position_m": [0.035, 0.035], "goal_normalized": 1.0}}}},
            actual_trace=[
                {"sim_step": 1, "state": {"arms": {"main": {"q_rad": [0.02] * 7, "gripper": {"position_m": [0.012, 0.012], "goal_normalized": 0.0}}}}},
                {"sim_step": 2, "state": {"arms": {"main": {"q_rad": [0.05] * 7, "gripper": {"position_m": [0.012, 0.012], "goal_normalized": 0.0}}}}},
            ],
            planned_trace={"actions": [[0.02] * 7 + [0.0], [0.05] * 7 + [0.0]], "row_contexts": [{"main": {"stage_id": "grasp"}}, {"main": {"stage_id": "lift"}}]},
            receipt={"execution_start_observation_id": "obs-s", "result_observation_id": "obs-t"},
            start_observation=start_obs,
            terminal_observation=term_obs,
        )
        self.assertIsNotNone(pt)
        self.assertIn("main", pt["co_motion"])
        self.assertIsNotNone(pt["co_motion"]["main"].get("fingertip_flow_px"))
        self.assertGreater(pt["co_motion"]["main"]["fingertip_flow_px"], 0.5)



if __name__ == "__main__":
    unittest.main()
