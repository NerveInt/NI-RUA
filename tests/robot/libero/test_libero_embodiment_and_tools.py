"""Pure-Python unit tests for simulation-only ``libero-franka-panda``, ``tool_search``, and RL confinement."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import tempfile
import unittest

import numpy as np

from ni_rua.core.config import Config
from ni_rua.core.embodiment import profile_from_record, profile_record
from ni_rua.core.storage.store import Store
from ni_rua.cortex.agents.opencode import config as opencode_config
from ni_rua.cortex.agents.opencode.session import OpenCodeSession, TOOL_NAME_PREFIX, WIRES
from ni_rua.cortex.context import startup_grounding
from ni_rua.perception.external.scene3d import _ik_staging, ik_prefilter_staged
from ni_rua.robot.backends.factory import ENVIRONMENT_BACKENDS, create_backend
from ni_rua.robot.backends.libero import ENVIRONMENT, LiberoBackend, contract
from ni_rua.robot.embodiments.libero_franka_panda import (
    ARM_ID,
    ARM_JOINTS,
    GRIPPER_JOINTS,
    LIBERO_FRANKA_PANDA,
    TCP_LINK,
    fk,
    fk_all,
    grasp_contact,
    ik,
    ik_position_priority,
    joint_limits_rad,
    libero_franka_panda_profile,
    load_model,
    pose_convention,
)
from ni_rua.apps.gateway.workbench import URDF_FILE_CLAUSE, URDF_INLINE_CLAUSE
from ni_rua.robot.model.embodiments import ARX_X5_PROFILE
from ni_rua.robot.model.robot_information import (
    robot_geometry_path,
    robot_information,
    robot_resource_path,
    robot_resources,
)
from ni_rua.runtime.harness import Harness
from ni_rua.runtime.programs.runner import ProgramRunner, ProgramServices
from ni_rua.runtime.tools.plugins import configured_tool_specs
from ni_rua.runtime.tools.tool_search import TOOL_SEARCH_NAME

REPO_ROOT = Path(__file__).resolve().parents[3]


class LiberoSimulationOnlyIdentityTests(unittest.TestCase):
    def test_simulation_only_labels_present_everywhere(self):
        info = robot_information("libero-franka-panda", section="all")
        self.assertTrue(info["available"])
        identity = info["reference"]["identity"]
        self.assertIs(identity["simulation_only"], True)
        self.assertEqual(identity["simulator"], "LIBERO (robosuite MuJoCo)")
        self.assertIn("SIMULATION-ONLY", info["briefing"].upper())
        self.assertIn(URDF_INLINE_CLAUSE, info["briefing"])

        default_info = robot_information("libero-franka-panda", section="default")
        self.assertTrue(default_info["available"])
        self.assertIs(default_info["reference"]["identity"]["simulation_only"], True)
        self.assertEqual(default_info["briefing"].count(URDF_INLINE_CLAUSE), 1)
        rewritten = default_info["briefing"].replace(URDF_INLINE_CLAUSE, URDF_FILE_CLAUSE)
        self.assertNotIn(URDF_INLINE_CLAUSE, rewritten)
        self.assertIn("<robot", default_info["urdf"])

        for section in ("default", "kinematics", "geometry", "dynamics", "control", "all"):
            sec_doc = robot_information("libero-franka-panda", section=section)
            self.assertTrue(sec_doc["available"], section)

        resources = robot_resources("libero-franka-panda")
        self.assertTrue(resources["available"])
        zip_path = robot_resource_path("libero-franka-panda", "geometry_model")
        self.assertTrue(zip_path.is_file())
        self.assertLess(zip_path.stat().st_size, 5 * 1024 * 1024)
        self.assertTrue(robot_geometry_path("libero-franka-panda").is_file())

        example_raw = (REPO_ROOT / "configs" / "libero.example.json").read_text(encoding="utf-8")
        self.assertIn("simulation-only", example_raw.lower())
        self.assertIn("mujoco", example_raw.lower())
        example_cfg = Config.load(REPO_ROOT / "configs" / "libero.example.json", environ={})
        self.assertEqual(example_cfg.backend.environment, "libero")
        self.assertEqual(example_cfg.backend.robot_model_id, "libero-franka-panda")
        self.assertIn("simulation-only", (example_cfg.run.title or "").lower())
        self.assertIn("simulation-only", example_cfg.run.tag)
        self.assertTrue(example_cfg.agent.opencode.tool_search)

        readme_en = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        readme_zh = (REPO_ROOT / "README.zh-CN.md").read_text(encoding="utf-8")
        self.assertIn("libero-franka-panda", readme_en)
        self.assertIn("simulation-only", readme_en.lower())
        self.assertIn("libero-franka-panda", readme_zh)
        self.assertIn("simulation-only", readme_zh)

        motion_txt = contract.actor_motion_instructions()
        self.assertIn("SIMULATION-ONLY", motion_txt)
        self.assertIn("NOT calibrated for physical Franka hardware", motion_txt)
        self.assertIn("local +-X is the parallel-jaw opening line", motion_txt)
        self.assertIn('local +Y points toward the "wrist" wrist camera', motion_txt)

        from ni_rua.robot.backends.libero.worker import _parse_args

        parsed_worker = _parse_args(
            [
                "--socket",
                "/tmp/libero_test.sock",
                "--max-episodes",
                "50",
                "--render-backend",
                "egl",
                "--egl-device-id",
                "0",
                "--lp-num-threads",
                "1",
            ]
        )
        self.assertEqual(parsed_worker.max_episodes, 50)
        self.assertEqual(parsed_worker.render_backend, "egl")
        self.assertEqual(parsed_worker.egl_device_id, 0)
        self.assertEqual(parsed_worker.lp_num_threads, 1)


class LiberoEmbodimentAndKinematicsTests(unittest.TestCase):
    def test_profile_roundtrip_and_7dof_kinematics(self):
        profile = libero_franka_panda_profile()
        self.assertEqual(profile.robot, LIBERO_FRANKA_PANDA)
        self.assertEqual(profile.arms, ("arm0",))
        self.assertEqual(profile.arm_joint_names["arm0"], ARM_JOINTS)
        self.assertEqual(profile.jaw_joint_names["arm0"], GRIPPER_JOINTS)
        self.assertEqual(profile.cameras, ("agentview", "wrist"))
        self.assertEqual(profile.overview_camera, "agentview")
        self.assertEqual(dict(profile.wrist_cameras), {"arm0": "wrist"})
        self.assertEqual(profile.max_stages, 32)
        self.assertEqual(profile.stages_per_arm, 32)
        self.assertAlmostEqual(profile.gripper.empty_close_aperture_m, 0.0010)
        self.assertAlmostEqual(profile.gripper.full_open_aperture_m, 0.080)
        self.assertIsNone(profile.home_check)
        self.assertFalse(profile.handoff)
        self.assertFalse(profile.policy_step)

        restored = profile_from_record(profile_record(profile))
        self.assertEqual(restored, profile)

        # Spawn orientation [0, 0, -90] deg == quaternion_xyzw [sqrt(0.5), sqrt(0.5), 0, 0]
        q_wxyz = pose_convention.angles_to_quat(0.0, 0.0, -90.0)
        q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
        s2 = np.sqrt(0.5)
        self.assertTrue(min(np.linalg.norm(q_xyzw - [s2, s2, 0, 0]), np.linalg.norm(q_xyzw + [s2, s2, 0, 0])) < 1e-6)
        self.assertEqual(
            tuple(round(v, 4) for v in pose_convention.quat_to_angles(q_wxyz)),
            (0.0, 0.0, -90.0),
        )

        # Contact presets
        self.assertEqual(dict(profile.gripper.contact_sections), {"pad_center": 0.0, "finger_middle": 0.0176, "fingertip": -0.0093})
        spec_mid = grasp_contact.resolve_contact_spec({"contact_region": "finger_middle"})
        self.assertAlmostEqual(spec_mid["contact_depth_m"], 0.0176)

        # 7-DoF FK -> IK round-trip via ni_rua.robot.model.kinematics
        model = load_model()
        q_nom = [0.0, -0.161, 0.0, -2.445, 0.0, 2.227, 0.7854]
        T_nom = fk(model, q_nom, link=TCP_LINK)
        seed = [q + 0.08 for q in q_nom]
        sol = ik(model, T_nom, seed, position_tolerance_m=1e-5, orientation_tolerance_rad=1e-4)
        self.assertTrue(sol["converged"])
        self.assertLess(sol["position_error_m"], 1e-4)
        self.assertLess(sol["orientation_error_rad"], 1e-3)

        pp = ik_position_priority(model, T_nom, seed, position_tolerance_m=1e-5)
        self.assertLess(pp["position_error_m"], 1e-4)

        all_links = fk_all(model, q_nom, finger_q_m=(0.04, -0.04))
        self.assertIn("panda_hand", all_links)
        self.assertIn("panda_leftfinger", all_links)
        self.assertIn("panda_rightfinger", all_links)
        self.assertEqual(len(joint_limits_rad(model)), 7)

        # 7-DoF Panda must NOT stage ARX-X5 6R IK prefilter
        eff_prof, supports_6r_ik, _ = _ik_staging(profile)
        self.assertEqual(eff_prof, profile)
        self.assertFalse(supports_6r_ik)
        self.assertFalse(ik_prefilter_staged(profile))
        self.assertEqual(
            startup_grounding.select_overview_camera(
                {"cameras": [{"camera_id": "wrist"}, {"camera_id": "agentview"}]},
                profile=profile,
            ),
            "agentview",
        )

    def test_factory_and_contract_validation(self):
        self.assertIn(ENVIRONMENT, ENVIRONMENT_BACKENDS)
        cfg = Config.load(REPO_ROOT / "configs" / "libero.example.json", environ={})
        backend = create_backend(config=cfg, python="/bin/true")
        self.assertIsInstance(backend, LiberoBackend)

        identity = {"run_id": "r1", "episode_id": "ep1", "env_instance_id": "i1", "env_generation": 1}
        caps = {
            "frame_id": "franka_base",
            "tcp_config_id": "libero-panda-grip-site-v1",
            "limits_profile": "libero-osc-20hz-v1",
            "arms": ["arm0"],
            "gripper_phases": ["after_motion", "during_motion"],
        }
        grasp_act = contract.compile_action(
            "grasp",
            {
                "arm": "arm0",
                "observation_id": "obs_1",
                "grasp_position_m": [0.15, 0.0, 0.08],
                "angles_deg": [0.0, 0.0, 90.0],
                "contact_region": "finger_middle",
                "gripper_goal_normalized": 0.35,
                "gripper_motion": {"start_fraction": 0.2, "end_fraction": 0.9},
                "max_upward_dz_m": 0.05,
                "max_downward_dz_m": 0.02,
                "stall_guard_preset": "contact_stop",
            },
            identity,
            caps,
        )
        arm, _ = contract.validate_action(grasp_act)
        self.assertEqual(arm, ARM_ID)

        # 32 stages accepted; 33 rejected; dual-arm "left" rejected
        stages_32 = [
            {
                "id": f"s{idx}",
                "arm": "arm0",
                "kind": "move",
                "action": contract.compile_action(
                    "move",
                    {
                        "arm": "arm0",
                        "observation_id": "obs_1",
                        "position_m": [0.1 + 0.002 * idx, 0.0, 0.2],
                        "quaternion_xyzw": [1.0, 0.0, 0.0, 0.0],
                    },
                    identity,
                    caps,
                ),
            }
            for idx in range(32)
        ]
        batch_32 = {
            "schema_version": "0.1",
            **identity,
            "world_kind": "observed",
            "type": "BatchAction",
            "action_id": "act_32",
            "based_on_observation_id": "obs_1",
            "mode": "absolute",
            "frame_id": "franka_base",
            "tcp_config_id": "libero-panda-grip-site-v1",
            "limits_profile": "libero-osc-20hz-v1",
            "timeout_s": 120.0,
            "stages": stages_32,
        }
        self.assertEqual(contract.validate_action(batch_32)[0], "arm0")
        with self.assertRaisesRegex(ValueError, "batch_requires_one_to_thirty_two_stages"):
            contract.validate_action({**batch_32, "stages": [*stages_32, dict(stages_32[0], id="s32")]})
        with self.assertRaisesRegex(ValueError, "unsupported_arm:left"):
            contract.compile_action(
                "move",
                {"arm": "left", "observation_id": "obs_1", "position_m": [0.1, 0.0, 0.2], "quaternion_xyzw": [1, 0, 0, 0]},
                identity,
                caps,
            )


class ToolSearchAndConfinementTests(unittest.TestCase):
    def test_native_tool_search_unlock_and_static_opencode_projection(self):
        cfg = Config.load(REPO_ROOT / "configs" / "libero.example.json", environ={})
        specs = configured_tool_specs(cfg)
        spec_names = [s.name for s in specs]
        self.assertIn(TOOL_SEARCH_NAME, spec_names)

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "store", {"run_id": "r1"})
            try:
                backend = create_backend(config=cfg, python="/bin/true")
                harness = Harness(
                    robot=backend,
                    store=store,
                    config=cfg,
                    tool_specs=specs,
                )
                self.assertEqual(harness.embodiment, libero_franka_panda_profile())
                harness.enable_function_tool_search()
                all_tools = harness.tool_registry.catalog(harness, role="actor")
                eager_names = [t["name"] for t in harness.eager_actor_tools()]
                deferred = [t["name"] for t in all_tools if t.get("deferLoading")]
                self.assertTrue(len(deferred) >= 5)
                target_cold = deferred[0]
                self.assertNotIn(target_cold, eager_names)
                self.assertIsNotNone(harness.on_demand_tools_section())

                # Calling a deferred tool before tool_search raises structured deferred_tool_not_loaded
                with self.assertRaisesRegex(Exception, "deferred_tool_not_loaded"):
                    harness.invoke(target_cold, {}, {"tool_call_id": "c1"}, role="actor")

                # Calling tool_search unlocks the deferred tool and returns its schema
                searched = harness.invoke(
                    TOOL_SEARCH_NAME,
                    {"names": [target_cold]},
                    {"tool_call_id": "c2"},
                    role="actor",
                )
                self.assertTrue(searched.success)
                self.assertIn(target_cold, harness.unlocked_deferred_tools)

                # Verify COLD_TOOLS consistency under layered_v2 even with a pre-built full_v1 ToolRegistry
                from ni_rua.runtime.tools.builtins import create_registry
                from ni_rua.runtime.tools.catalog_guide import COLD_TOOLS

                cfg_v2 = cfg.with_overrides({"features": {"tool_catalog_profile": "layered_v2"}})
                prebuilt_reg = create_registry(extra_specs=specs)
                store_v2 = Store(Path(tmp) / "store_v2", {"run_id": "r2"})
                try:
                    harness_v2 = Harness(
                        robot=create_backend(config=cfg_v2, python="/bin/true"),
                        store=store_v2,
                        config=cfg_v2,
                        tool_specs=specs,
                    )
                    harness_v2.tool_registry = prebuilt_reg
                    harness_v2.enable_function_tool_search()
                    loaded_v2 = harness_v2.tool_registry.loaded_tool_names(harness_v2, role="actor")
                    for cold_name in COLD_TOOLS:
                        self.assertNotIn(cold_name, loaded_v2)
                        with self.assertRaisesRegex(Exception, "deferred_tool_not_loaded"):
                            harness_v2.invoke(cold_name, {}, {"tool_call_id": f"v2_{cold_name}"}, role="actor")
                    unlock_v2 = harness_v2.invoke(
                        TOOL_SEARCH_NAME,
                        {"names": sorted(COLD_TOOLS)},
                        {"tool_call_id": "v2_unlock"},
                        role="actor",
                    )
                    self.assertTrue(unlock_v2.success)
                    loaded_after = harness_v2.tool_registry.loaded_tool_names(harness_v2, role="actor")
                    for cold_name in COLD_TOOLS:
                        self.assertIn(cold_name, loaded_after)
                finally:
                    store_v2.close()

                # OpenCodeSession._project_request strips deferred tools + denied native tools deterministically
                dummy = object.__new__(OpenCodeSession)
                dummy._deferred_tool_names = frozenset(TOOL_NAME_PREFIX + n for n in deferred)
                dummy._denied_native_tools = frozenset(("shell", "bash", "execute", "subagent", "task"))
                dummy._tool_names = frozenset(t["name"] for t in all_tools)
                dummy.presentation = harness.actor_presentation
                dummy._egress_unknown_part = lambda _t: None
                dummy.wire = WIRES["openai_responses"](client="opencode")
                mcp_wire_tools = [
                    *[
                        {"type": "function", "name": TOOL_NAME_PREFIX + t["name"], "parameters": {"type": "object"}}
                        for t in all_tools
                    ],
                    {"type": "function", "name": "bash", "parameters": {"type": "object"}},
                    {"type": "function", "name": "subagent", "parameters": {"type": "object"}},
                    {"type": "function", "name": "read", "parameters": {"type": "object"}},
                ]
                proj1 = dummy._project_request({"model": "qwen", "input": [], "tools": list(mcp_wire_tools)})
                proj2 = dummy._project_request({"model": "qwen", "input": [], "tools": list(mcp_wire_tools)})
                proj_names = [t["name"] for t in proj1["tools"]]
                self.assertEqual(proj1["tools"], proj2["tools"])
                self.assertIn(TOOL_NAME_PREFIX + TOOL_SEARCH_NAME, proj_names)
                self.assertIn("read", proj_names)
                self.assertNotIn("bash", proj_names)
                self.assertNotIn("subagent", proj_names)
                self.assertNotIn(TOOL_NAME_PREFIX + target_cold, proj_names)
            finally:
                store.close()

    def test_opencode_rl_confinement_permissions_and_seccomp_program_runner(self):
        spec = opencode_config.model_spec("Qwen3.8-27B", context_limit_tokens=131072, output_limit_tokens=16384)
        with tempfile.TemporaryDirectory() as tmp:
            actor_dir = Path(tmp) / "actor"
            actor_dir.mkdir()
            instr = Path(tmp) / "NI-RUA.md"
            instr.write_text("instructions", encoding="utf-8")
            doc = opencode_config.build_config(
                models=[spec],
                wire_api="openai_responses",
                relay_base_url="http://127.0.0.1:9000",
                instructions_path=instr,
                subagents=False,
                allow_shell=False,
                allow_external_read=False,
                actor_dir=actor_dir,
            )
            perms = doc["permissions"]
            self.assertIn({"action": "external_directory", "resource": "*", "effect": "deny"}, perms)
            self.assertIn({"action": "shell", "resource": "*", "effect": "deny"}, perms)
            self.assertIn({"action": "execute", "resource": "*", "effect": "deny"}, perms)
            self.assertIn({"action": "subagent", "resource": "*", "effect": "deny"}, perms)
            self.assertIn({"action": "read", "resource": f"{actor_dir.resolve()}/*", "effect": "allow"}, perms)

            # Verify standalone libseccomp confinement in Harness/ProgramRunner (sandbox_enabled=False, program_seccomp_enabled=True)
            cfg = Config.load(REPO_ROOT / "configs" / "libero.example.json", environ={})
            self.assertFalse(cfg.features.sandbox_enabled)
            self.assertTrue(cfg.features.program_seccomp_enabled)
            store = Store(
                Path(tmp) / "seccomp_store",
                {"run_id": "r_sec", "episode_id": "ep_sec", "env_instance_id": "i1", "env_generation": 1},
            )
            try:
                harness = Harness(
                    robot=None,
                    store=store,
                    config=cfg,
                )
                runner = ProgramRunner(harness.program_services())
                try:
                    started = runner.run(
                        {
                            "source": (
                                "import ctypes, numpy as np, os, socket\n"
                                "from pathlib import Path\n"
                                "Path('local_ok.txt').write_text(str(int(np.arange(4).sum())))\n"
                                "local_ok = Path('local_ok.txt').read_text() == '6'\n"
                                "sock_blocked = fork_blocked = store_blocked = list_blocked = ctypes_blocked = False\n"
                                "try:\n"
                                "    socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
                                "except PermissionError:\n"
                                "    sock_blocked = True\n"
                                "try:\n"
                                "    os.fork()\n"
                                "except PermissionError:\n"
                                "    fork_blocked = True\n"
                                "try:\n"
                                f"    open({str(store.root / 'events.sqlite')!r}, 'rb')\n"
                                "except PermissionError:\n"
                                "    store_blocked = True\n"
                                "try:\n"
                                f"    os.listdir({str(store.root)!r})\n"
                                "except PermissionError:\n"
                                "    list_blocked = True\n"
                                "try:\n"
                                "    ctypes.CDLL(None)\n"
                                "except PermissionError:\n"
                                "    ctypes_blocked = True\n"
                                "print(f'SECCOMP_CHECK local={local_ok} sock={sock_blocked} fork={fork_blocked} store={store_blocked} list={list_blocked} ctypes={ctypes_blocked}')\n"
                            ),
                            "timeout_s": 15,
                        },
                        {"thread_id": "t1", "turn_id": "u1", "tool_call_id": "c_sec"},
                    )
                    self.assertEqual(started["status"], "succeeded", started)
                    self.assertIn(
                        "SECCOMP_CHECK local=True sock=True fork=True store=True list=True ctypes=True",
                        started.get("stdout_text", ""),
                    )
                finally:
                    runner.close()
            finally:
                store.close()

    def test_opencode_only_runtime_imports_without_codex_sdk(self):
        import subprocess
        import sys

        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import builtins, importlib\n"
                    "_real_import = builtins.__import__\n"
                    "def _no_codex(name, *a, **kw):\n"
                    "    if name == 'openai_codex' or name.startswith('openai_codex.'):\n"
                    "        raise ModuleNotFoundError(name)\n"
                    "    return _real_import(name, *a, **kw)\n"
                    "builtins.__import__ = _no_codex\n"
                    "for mod in (\n"
                    "    'ni_rua.apps.gateway.workbench',\n"
                    "    'ni_rua.apps.gateway.routes',\n"
                    "    'ni_rua.cortex.agents.opencode.session',\n"
                    "    'ni_rua.cortex.agents.mcp_server',\n"
                    "    'ni_rua.cortex.agents.evidence',\n"
                    "    'ni_rua.runtime.harness',\n"
                    "    'ni_rua.robot.backends.libero',\n"
                    "):\n"
                    "    importlib.import_module(mod)\n"
                    "print('NO_CODEX_OK')\n"
                ),
            ],
            cwd=str(REPO_ROOT),
            env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO_ROOT)},
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("NO_CODEX_OK", proc.stdout)

    def test_libero_vs_x5_obs3d_and_actor_instructions_have_zero_leaks(self):
        from ni_rua.cortex.context.actor_instructions import (
            CALIBRATED_OBSERVATION_INSTRUCTIONS,
            OPERATION_NOTES,
            instructions as build_instructions,
            operation_notes,
            task_notes as current_task_notes,
        )
        from ni_rua.cortex.context.task_notes import resolve_task_notes
        from ni_rua.perception.external import obs3d
        from ni_rua.robot.backends.libero.contract import actor_motion_instructions
        from ni_rua.robot.model.embodiments import ARX_X5_PROFILE, LIBERO_FRANKA_PANDA_PROFILE

        features = Config.from_dict(
            {
                "features": {
                    "external_perception": True,
                    "object_table": True,
                    "object_table_observe": True,
                    "scene3d": True,
                    "scene3d_observe": True,
                }
            }
        ).features

        # 1. obs3d.developer_instructions on LIBERO must never leak ARX-X5 6R IK helpers, obstacle_object_ids, or Head-camera
        libero_obs3d = obs3d.developer_instructions(features, LIBERO_FRANKA_PANDA_PROFILE)
        x5_obs3d = obs3d.developer_instructions(features, ARX_X5_PROFILE)
        for forbidden in (
            "Head-camera",
            "obstacle_object_ids",
            "check_source_to_dest",
            "check_reachability",
            "solve_reachable_pose",
            "prefilter_stages",
            "plan_joint_safe_return",
            "min_bimanual_clearance_m",
            "6R kinematic",
            "allow_jaw_symmetry",
            "allow_approach_tilt",
        ):
            self.assertNotIn(forbidden, libero_obs3d, f"LIBERO obs3d leaked {forbidden!r}")
        for expected in ("Agentview-camera", "pad_z_m", "80 mm", "s3.top_z", "s3.triangulate_rays", "label_px"):
            self.assertIn(expected, libero_obs3d)
        # ARX-X5 keeps its full 6R IK & obstacle_object_ids text intact
        for expected_x5 in (
            "Head-camera",
            "obstacle_object_ids",
            "s3.check_source_to_dest",
            "s3.prefilter_stages",
            "min_bimanual_clearance_m",
        ):
            self.assertIn(expected_x5, x5_obs3d)

        # 2. operation_notes / task_notes on LIBERO must never advertise rejected or dual-arm/6R-IK features
        libero_notes = operation_notes(LIBERO_FRANKA_PANDA_PROFILE)
        self.assertEqual(operation_notes(ARX_X5_PROFILE), OPERATION_NOTES)
        self.assertEqual(operation_notes(None), OPERATION_NOTES)
        for forbidden in (
            "joint_travel_preset",
            "max_joint_travel_rad",
            "max_backtracking_rad",
            "obstacle_object_ids",
            "check_source_to_dest",
            "plan_joint_safe_return",
            "handoff",
            "two arms",
            "both arms",
            "allow_jaw_symmetry",
            "allow_approach_tilt",
        ):
            self.assertNotIn(forbidden, libero_notes, f"LIBERO operation_notes leaked {forbidden!r}")
        for expected in ("mode='execute_if_clean'", "mode='reach'", "stall_guard_preset='contact_stop'", "max_upward_dz_m"):
            self.assertIn(expected, libero_notes)
        resolved = current_task_notes(resolve_task_notes(default_enabled=True), profile=LIBERO_FRANKA_PANDA_PROFILE)
        self.assertEqual(resolved["text"], libero_notes)

        # 3. _local_source_instructions gates shell / external-read wording cleanly on allow_shell / allow_external_read
        confined = build_instructions(
            CALIBRATED_OBSERVATION_INSTRUCTIONS,
            None,
            source_root="/secret/run/private/actor-source",
            embodiment_motion=actor_motion_instructions(LIBERO_FRANKA_PANDA_PROFILE),
            profile=LIBERO_FRANKA_PANDA_PROFILE,
            allow_shell=False,
            allow_external_read=False,
        )
        self.assertNotIn("Use native shell", confined)
        self.assertNotIn("/secret/run/private/actor-source", confined)
        self.assertIn("native shell tools are disabled", confined)
        self.assertIn("external directory reads are disabled", confined)
        self.assertNotIn("s3.check_source_to_dest", confined)

        # Default allow_shell=True, allow_external_read=True preserves exact legacy source instructions
        unconfined_x5 = build_instructions(
            CALIBRATED_OBSERVATION_INSTRUCTIONS,
            None,
            source_root="/tmp/actor-source",
            profile=ARX_X5_PROFILE,
        )
        self.assertIn("Use native shell and file tools", unconfined_x5)
        self.assertIn("Read the public Harness implementation under /tmp/actor-source.", unconfined_x5)

    def test_workbench_opencode_libero_startup_grounding_and_confinement(self):
        import asyncio
        import threading
        from types import SimpleNamespace
        from unittest.mock import Mock
        import cv2
        from ni_rua.apps.gateway.workbench import Workbench
        from ni_rua.robot.model.embodiments import LIBERO_FRANKA_PANDA_PROFILE

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            fake_exe = Path(tmp) / "opencode"
            fake_exe.write_text("#!/bin/sh\necho 2.2.7\n", encoding="utf-8")
            fake_exe.chmod(0o755)
            img = np.full((128, 128, 3), 120, dtype=np.uint8)
            img_path = Path(tmp) / "agentview.png"
            self.assertTrue(cv2.imwrite(str(img_path), img))

            class FakeLiberoRobot:
                def embodiment_profile(self):
                    return LIBERO_FRANKA_PANDA_PROFILE

                def start(self):
                    return {
                        "robot_model": LIBERO_FRANKA_PANDA_PROFILE.robot_model_id,
                        "frame_id": LIBERO_FRANKA_PANDA_PROFILE.frame_id,
                        "tcp_config_id": LIBERO_FRANKA_PANDA_PROFILE.tcp_config_id,
                        "data_origin": "libero_mujoco",
                        "task_instruction": "Pick up the mug.",
                        "capabilities": {"arms": [ARM_ID]},
                    }

                def actor_motion_instructions(self):
                    return contract.actor_motion_instructions(LIBERO_FRANKA_PANDA_PROFILE)

                def observe(self):
                    return {
                        "sim_step": 0,
                        "step_domain": "libero_control_step",
                        "instruction": "Pick up the mug.",
                        "robot_state": {
                            "frame_id": LIBERO_FRANKA_PANDA_PROFILE.frame_id,
                            "tcp_config_id": LIBERO_FRANKA_PANDA_PROFILE.tcp_config_id,
                            "sim_step": 0,
                            "step_domain": "libero_control_step",
                            "arms": {
                                ARM_ID: {
                                    "joint_names": list(ARM_JOINTS),
                                    "q_rad": [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.7],
                                    "ee_pose": {
                                        "position_m": [0.1, 0.0, 0.9],
                                        "quaternion_xyzw": [1.0, 0.0, 0.0, 0.0],
                                    },
                                    "gripper": {
                                        "goal_normalized": 1.0,
                                        "command_normalized": 1.0,
                                        "joint_names": list(GRIPPER_JOINTS),
                                        "position_m": [0.04, 0.04],
                                    },
                                }
                            },
                        },
                        "camera_calibration": {
                            "cameras": {
                                "agentview": {
                                    "width": 128,
                                    "height": 128,
                                    "K": [[100.0, 0.0, 64.0], [0.0, 100.0, 64.0], [0.0, 0.0, 1.0]],
                                }
                            }
                        },
                        "camera_pose_sample": {
                            "cameras": {
                                "agentview": {
                                    "T_world_camera": [
                                        [1.0, 0.0, 0.0, 0.5],
                                        [0.0, -1.0, 0.0, 0.0],
                                        [0.0, 0.0, -1.0, 1.3],
                                        [0.0, 0.0, 0.0, 1.0],
                                    ]
                                }
                            }
                        },
                        "robot_model": {"robot_model_id": LIBERO_FRANKA_PANDA_PROFILE.robot_model_id},
                        "cameras": [
                            {
                                "camera_id": "agentview",
                                "mime_type": "image/png",
                                "png_path": str(img_path),
                                "width": 128,
                                "height": 128,
                            }
                        ],
                    }

                def close(self):
                    pass

            cfg = Config.from_dict(
                {
                    "backend": {"environment": "libero", "robot_model_id": "libero-franka-panda"},
                    "agent": {
                        "backend": "opencode",
                        "opencode": {
                            "executable": str(fake_exe),
                            "allow_shell": False,
                            "allow_external_read": False,
                            "upstream": {
                                "wire_api": "openai_responses",
                                "base_url": "http://127.0.0.1:9999/v1",
                                "api_key_env": "FAKE_KEY",
                                "provider": "fake",
                            },
                        },
                    },
                    "model": {"name": "fake-vlm", "effort": "medium"},
                    "features": {
                        "sandbox_enabled": False,
                        "terminal": False,
                        "motion_review": False,
                        "external_perception": False,
                        "startup_inline_grounding": True,
                        "visual_review_model": None,
                        "tool_catalog_profile": "layered_v3",
                    },
                }
            )
            wb = Workbench(root, robot=FakeLiberoRobot(), config=cfg)
            self.addCleanup(wb.store.close)
            captured = {}
            done = threading.Event()
            done.set()

            class FakeOpenCode(OpenCodeSession):
                def __init__(self):
                    self.thread_id = "ses-libero-1"
                    self.turn_id = "turn-1"
                    self.state = "completed"
                    self.turn_done = done
                    self.model = "fake-vlm"
                    self.provider = "fake"
                    self.effort = "medium"
                    self.price_card = None
                    self.cli_executable = Path("/fixture/opencode")
                    self.command_prefix = []
                    self.actor_dir = root / "actor"
                    self.actor_dir.mkdir(parents=True, exist_ok=True)
                    self.socket_path = None
                    self.env = {}
                    self.process = SimpleNamespace(pid=-1)
                    self.model_proxy = SimpleNamespace(limit_notified=False, request_count=1)
                    self.client = SimpleNamespace(prompt=Mock())
                    self.presentation = wb.session.presentation
                    self._models = [opencode_config.model_spec("fake-vlm")]
                    self.close = Mock()

                def start(self, tools, instructions, **kwargs):
                    captured["tools"] = tools
                    captured["instructions"] = instructions
                    return self.thread_id

                def run_turn(self, prompt, *, images=()):
                    captured["prompt"] = prompt
                    captured["images"] = tuple(images)
                    return self.turn_id

                def provenance(self):
                    return {"kind": "opencode", "provider": "fake", "model": self.model}

            wb.session = FakeOpenCode()
            asyncio.run(wb.start("Pick up the mug.", benchmark=True))

            self.assertIsNotNone(wb.startup_grounding)
            self.assertEqual(wb.startup_grounding["image_kind"], "raw_rgb")
            self.assertEqual(len(captured["images"]), 1)
            self.assertEqual(captured["images"][0]["type"], "image")
            self.assertTrue(captured["images"][0]["url"].startswith("data:image/"))
            self.assertIn("obs_1", captured["prompt"])
            self.assertNotIn("Use native shell", captured["instructions"])
            self.assertNotIn("private/actor-source", captured["instructions"])
            self.assertNotIn("joint_travel_preset", captured["prompt"])
            self.assertNotIn("obstacle_object_ids", captured["prompt"])
            self.assertNotIn("Head-camera", captured["instructions"])
            self.assertNotIn("check_source_to_dest", captured["instructions"])


if __name__ == "__main__":
    unittest.main()
