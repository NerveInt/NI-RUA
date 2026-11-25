"""AST layer boundary enforcement and Stage 1 core kernel purification checks."""
from __future__ import annotations

import ast
import contextlib
import dataclasses
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from support.paths import REPO_ROOT

NI_RUA_TEST_TIER = "fast"
CORE_ROOT = REPO_ROOT / "ni_rua" / "core"
FORBIDDEN_CORE_TARGETS = frozenset({"robot", "perception", "cortex", "runtime", "apps"})
STDLIB_MODULES = frozenset(getattr(sys, "stdlib_module_names", ()))


def _resolve_relative_targets(file_path: Path, node: ast.ImportFrom) -> list[tuple[str, ...]]:
    rel_parts = file_path.relative_to(REPO_ROOT).parts
    pkg_parts = rel_parts[:-1]
    up = node.level - 1
    if up > len(pkg_parts):
        return []
    base = pkg_parts[: len(pkg_parts) - up]
    if node.module:
        return [base + tuple(node.module.split("."))]
    return [base + (alias.name,) for alias in node.names]


def _upward_imports(tree: ast.AST, file_path: Path) -> list[tuple[int, str]]:
    """Return `(lineno, description)` for any module-level or nested upward import in `file_path`."""
    violations: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if len(parts) >= 2 and parts[0] == "ni_rua" and parts[1] in FORBIDDEN_CORE_TARGETS:
                    violations.append((node.lineno, f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom):
            if node.level >= 1:
                targets = _resolve_relative_targets(file_path, node)
            else:
                module_parts = tuple(node.module.split(".")) if node.module else ()
                if module_parts == ("ni_rua",):
                    targets = [("ni_rua", alias.name) for alias in node.names]
                else:
                    targets = [module_parts]
            for parts in targets:
                if len(parts) >= 2 and parts[0] == "ni_rua" and parts[1] in FORBIDDEN_CORE_TARGETS:
                    violations.append((node.lineno, f"from {'.'.join(parts)}"))
        elif isinstance(node, ast.Call):
            fn = node.func
            is_dynamic = (
                (isinstance(fn, ast.Name) and fn.id == "__import__")
                or (isinstance(fn, ast.Attribute) and fn.attr == "import_module")
            )
            if is_dynamic and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                parts = node.args[0].value.split(".")
                if len(parts) >= 2 and parts[0] == "ni_rua" and parts[1] in FORBIDDEN_CORE_TARGETS:
                    violations.append((node.lineno, f"dynamic import {node.args[0].value}"))
    return violations


def _imported_top_modules(tree: ast.AST, file_path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level >= 1:
                for parts in _resolve_relative_targets(file_path, node):
                    if parts:
                        modules.add(parts[0])
            elif node.module:
                modules.add(node.module.split(".", 1)[0])
    return modules


class CoreLayerBoundaryTests(unittest.TestCase):
    def test_ni_rua_core_has_zero_upward_imports_module_and_deferred(self):
        py_files = sorted(CORE_ROOT.rglob("*.py"))
        self.assertTrue(py_files)
        all_violations: list[str] = []
        for path in py_files:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for lineno, desc in _upward_imports(tree, path):
                all_violations.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {desc}")
        self.assertEqual(all_violations, [])

    def test_core_embodiment_is_stdlib_only_leaf(self):
        path = CORE_ROOT / "embodiment.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        self.assertEqual(_upward_imports(tree, path), [])
        top_modules = _imported_top_modules(tree, path)
        self.assertNotIn("ni_rua", top_modules)
        self.assertNotIn("numpy", top_modules)
        if STDLIB_MODULES:
            self.assertLessEqual(top_modules, STDLIB_MODULES | {"__future__"})

    def test_core_pose_math_is_stdlib_and_numpy_only_and_batch_motion_avoids_perception(self):
        pose_path = CORE_ROOT / "pose_math.py"
        pose_tree = ast.parse(pose_path.read_text(encoding="utf-8"), filename=str(pose_path))
        top_modules = _imported_top_modules(pose_tree, pose_path)
        self.assertNotIn("ni_rua", top_modules)
        if STDLIB_MODULES:
            self.assertLessEqual(top_modules, STDLIB_MODULES | {"__future__", "numpy", "scipy"})

        batch_path = REPO_ROOT / "ni_rua" / "robot" / "motion" / "batch_motion.py"
        batch_tree = ast.parse(batch_path.read_text(encoding="utf-8"), filename=str(batch_path))
        for node in ast.walk(batch_tree):
            if isinstance(node, ast.ImportFrom) and node.level >= 1:
                for parts in _resolve_relative_targets(batch_path, node):
                    self.assertFalse(len(parts) >= 2 and parts[:2] == ("ni_rua", "perception"), parts)
            elif isinstance(node, ast.ImportFrom) and node.module:
                self.assertFalse(node.module.startswith("ni_rua.perception"), node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertFalse(alias.name.startswith("ni_rua.perception"), alias.name)

    def test_core_pose_math_quaternion_helpers_load_without_scipy_rigid_transform(self):
        # A robot worker runs in the simulator's interpreter, whose SciPy may predate RigidTransform (1.16);
        # batch_motion imports the quaternion helpers there.
        import importlib.util
        import types
        import scipy.spatial.transform as real

        older = types.ModuleType("scipy.spatial.transform")
        older.__dict__.update({name: value for name, value in vars(real).items() if name != "RigidTransform"})
        spec = importlib.util.spec_from_file_location("_pose_math_on_older_scipy", CORE_ROOT / "pose_math.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"scipy.spatial.transform": older}):
            spec.loader.exec_module(module)
            rotated = module.quat_wxyz_to_matrix(module.quat_multiply((1.0, 0.0, 0.0, 0.0), module.TOP_DOWN_QUAT_WXYZ))
            with self.assertRaises(ImportError):
                module.se3_exp([0.0, 0.0, 0.1, 0.0, 0.0, 0.2])
        self.assertEqual(rotated.shape, (3, 3))
        self.assertAlmostEqual(float(rotated[2, 0]), -1.0, places=9)

    def test_perception_pose_math_reexports_core_pose_math(self):
        from ni_rua.core import pose_math as core_pm
        from ni_rua.perception.external import pose_math as ext_pm

        for name in (
            "TOP_DOWN_QUAT_WXYZ", "wrap180", "wrap90", "axis_difference_deg", "closure_yaws_deg",
            "quat_multiply", "quat_conjugate", "quat_normalize", "quat_wxyz_to_matrix",
            "angles_deg_to_quat_wxyz", "quat_wxyz_to_angles_deg",
        ):
            self.assertIs(getattr(ext_pm, name), getattr(core_pm, name), name)

    def test_embodiment_v1_byte_identical_and_v2_round_trip(self):
        from ni_rua.core.embodiment import (
            ArmProfile, EndEffectorSpec, FrozenDict, KinematicGroupSpec,
            PROFILE_RECORD_SCHEMA, PROFILE_RECORD_SCHEMA_V2,
            profile_from_record, profile_record, profile_to_record,
        )
        from ni_rua.robot.model import embodiments as e

        self.assertIs(profile_record, profile_to_record)
        v1_rec = profile_record(e.ARX_X5_PROFILE)
        self.assertEqual(v1_rec["schema"], PROFILE_RECORD_SCHEMA)
        self.assertNotIn("arm_profiles", v1_rec["profile"])
        self.assertEqual(profile_from_record(json.loads(json.dumps(v1_rec))), e.ARX_X5_PROFILE)

        x5 = e.ARX_X5_PROFILE
        arm_specs = {
            arm: ArmProfile(
                arm=arm,
                kinematics=KinematicGroupSpec(
                    joint_names=x5.arm_joint_names[arm],
                    tip_link=x5.tcp_link,
                    tip_label=x5.tcp_label,
                    tcp_config_id=x5.tcp_config_id,
                    base_link="base_link",
                    rest_posture_rad=(0.0,) * len(x5.arm_joint_names[arm]),
                ),
                end_effector=EndEffectorSpec(
                    jaw_joint_names=x5.jaw_joint_names[arm],
                    gripper=x5.gripper,
                ),
                wrist_camera=x5.wrist_cameras.get(arm),
            )
            for arm in x5.arms
        }
        v2_profile = dataclasses.replace(x5, arm_profiles=FrozenDict(arm_specs))
        v2_rec = profile_to_record(v2_profile)
        self.assertEqual(v2_rec["schema"], PROFILE_RECORD_SCHEMA_V2)
        self.assertIn("arm_profiles", v2_rec["profile"])
        restored = profile_from_record(json.loads(json.dumps(v2_rec)))
        self.assertEqual(restored, v2_profile)
        self.assertEqual(hash(restored), hash(v2_profile))

    def test_config_ignore_local_and_cli_no_local_config_flag(self):
        from ni_rua.apps.gateway.cli import main
        from ni_rua.core.config import Config

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local = root / "ni-rua.local.json"
            local.write_text(json.dumps({"model": {"name": "local-override-model"}}))
            explicit = root / "explicit.json"
            explicit.write_text(json.dumps({"features": {"enable_wait": False}}))

            with patch("ni_rua.core.config.LOCAL_CONFIG_PATH", local):
                with_local = Config.load(explicit, environ={})
                without_local = Config.load(explicit, environ={}, ignore_local=True)
                self.assertEqual(with_local.model.name, "local-override-model")
                self.assertNotEqual(without_local.model.name, "local-override-model")
                self.assertFalse(without_local.features.enable_wait)

                out = io.StringIO()
                with patch.dict(os.environ, {}, clear=True), \
                        patch("sys.argv", ["ni_rua", "--config", str(explicit), "--no-local-config", "--print-config"]), \
                        patch("ni_rua.apps.gateway.cli.serve", new_callable=AsyncMock) as serve, \
                        contextlib.redirect_stdout(out):
                    main()
                serve.assert_not_called()
                printed = json.loads(out.getvalue())
                self.assertEqual(printed["model"]["name"], without_local.model.name)
                self.assertFalse(printed["features"]["enable_wait"])

    def test_actor_source_curated_without_private_modules(self):
        from ni_rua.cortex.session.actor_source import SOURCE_FILES

        self.assertIn("core/embodiment.py", SOURCE_FILES)
        self.assertIn("core/pose_math.py", SOURCE_FILES)
        for entry in SOURCE_FILES:
            self.assertFalse(entry.startswith("robot/backends/"), entry)
            self.assertFalse(entry.startswith("apps/"), entry)
            self.assertNotIn("profiles.py", entry)


if __name__ == "__main__":
    unittest.main()
