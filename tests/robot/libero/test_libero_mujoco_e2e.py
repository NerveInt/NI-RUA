"""End-to-end live LIBERO MuJoCo tests for simulation-only ``libero-franka-panda``.

Exercises standalone ``--stdio`` subprocess launch, pooled ``--socket`` daemon resets,
7-DoF FK/IK parity against MuJoCo ``gripper0_grip_site``, 512x512 RGB/RGB-D + calibrated
camera extrinsics, continuous ``PandaGripper`` aperture control, ``>12``-stage batches,
``stall_guard_preset`` / ``max_upward_dz_m`` / ``max_downward_dz_m`` / ``stage_grip_guard`` /
``carry_slip_guard``, ``EvaluatorBackend`` + ``VideoBackend``, LIBERO-PRO / LIBERO-Plus
perturbations, and all 18 actor tools through ``Harness`` with zero ``internal_runtime_error``s.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest

import numpy as np

from ni_rua.core.config import Config
from ni_rua.core.observation_policy import reject_privileged
from ni_rua.core.storage.store import Store
from ni_rua.robot.backends.factory import create_backend
from ni_rua.robot.backends.libero import LiberoBackend, LiberoWorkerClient
from ni_rua.robot.embodiments.libero_franka_panda import TCP_LINK, fk, load_model
from ni_rua.runtime.harness import Harness
from ni_rua.runtime.programs.runner import ProgramRunner
from ni_rua.runtime.tools.plugins import configured_tool_specs
from ni_rua.runtime.tools.tool_search import TOOL_SEARCH_NAME

REPO_ROOT = Path(__file__).resolve().parents[3]


def _find_libero_python() -> str | None:
    candidates = [
        os.environ.get("NI_RUA_LIBERO_PYTHON"),
        str(Path("~/miniforge3/envs/libero/bin/python").expanduser()),
        str(Path("~/conda/envs/sim-libero/bin/python").expanduser()),
    ]
    for raw in candidates:
        if raw and Path(raw).is_file() and os.access(raw, os.X_OK):
            return str(Path(raw).resolve())
    return None


LIBERO_PYTHON = _find_libero_python()


def _quat_angle_rad(q1_xyzw: list[float], q2_xyzw: list[float]) -> float:
    a = np.asarray(q1_xyzw, dtype=np.float64)
    b = np.asarray(q2_xyzw, dtype=np.float64)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    dot = min(1.0, abs(float(np.dot(a, b))))
    return float(2.0 * np.arccos(dot))


@unittest.skipUnless(LIBERO_PYTHON is not None, "LIBERO MuJoCo Python environment not installed on this host")
class LiberoMuJoCoEndToEndTests(unittest.TestCase):
    _daemon_dir: tempfile.TemporaryDirectory | None = None
    _daemon_proc: subprocess.Popen | None = None
    _socket_path: Path | None = None

    @classmethod
    def setUpClass(cls) -> None:
        assert LIBERO_PYTHON is not None
        cls._daemon_dir = tempfile.TemporaryDirectory(prefix="ni_rua_libero_sock_")
        sock = Path(cls._daemon_dir.name) / "worker.sock"
        env = os.environ.copy()
        env.setdefault("MUJOCO_GL", "egl")
        env.setdefault("PYOPENGL_PLATFORM", "egl")
        env["EAI_SIM_MAX_EPISODES"] = "2"
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (str(REPO_ROOT), env.get("PYTHONPATH", "")) if p
        )
        cls._daemon_proc = subprocess.Popen(
            [
                LIBERO_PYTHON,
                "-u",
                "-m",
                "ni_rua.robot.backends.libero.worker",
                "--socket",
                str(sock),
                "--run-dir",
                str(Path(cls._daemon_dir.name) / "daemon_default"),
                "--lp-num-threads",
                "1",
            ],
            cwd=str(REPO_ROOT),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                if cls._daemon_proc.poll() is not None:
                    err = cls._daemon_proc.stderr.read().decode("utf-8", errors="replace") if cls._daemon_proc.stderr else ""
                    raise RuntimeError(f"libero socket daemon exited early: {err}")
                if sock.exists():
                    try:
                        hello = LiberoWorkerClient.hello(sock, timeout=5.0)
                        if hello.get("ok"):
                            cls._socket_path = sock
                            break
                    except Exception:
                        pass
                time.sleep(0.05)
            else:
                raise TimeoutError("Timed out waiting for libero socket daemon")
        except Exception:
            cls._daemon_proc.kill()
            cls._daemon_proc.wait(timeout=5.0)
            cls._daemon_dir.cleanup()
            raise

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._socket_path is not None and cls._socket_path.exists():
            try:
                LiberoWorkerClient.shutdown(cls._socket_path, timeout=5.0)
            except Exception:
                pass
        if cls._daemon_proc is not None:
            try:
                cls._daemon_proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                cls._daemon_proc.kill()
                cls._daemon_proc.wait(timeout=5.0)
        if cls._daemon_dir is not None:
            cls._daemon_dir.cleanup()

    def _call_id(self, prefix: str, idx: int) -> dict[str, str]:
        return {"thread_id": "t_e2e", "turn_id": f"turn_{idx}", "tool_call_id": f"{prefix}_{idx}"}

    def test_stdio_subprocess_fk_rgbd_evaluator_and_video(self) -> None:
        assert LIBERO_PYTHON is not None
        cfg = Config.load(REPO_ROOT / "configs" / "libero.example.json", environ={})
        model = load_model()
        with tempfile.TemporaryDirectory(prefix="ni_rua_libero_stdio_") as tmp:
            run_dir = Path(tmp) / "run"
            backend = create_backend(
                config=cfg,
                run_dir=run_dir,
                python=LIBERO_PYTHON,
                task="libero_spatial:0",
                seed=0,
                layout_id=0,
                observation_profile="rgbd_proprio_calibrated_v1",
                camera_resolution=512,
            )
            self.assertIsInstance(backend, LiberoBackend)
            try:
                caps = backend.start()
                self.assertTrue(caps["simulation_only"])
                self.assertEqual(caps["robot_model_id"], "libero-franka-panda")
                self.assertEqual(caps["camera_resolution"], [512, 512])
                self.assertAlmostEqual(caps["scene_calibration"]["table_z_m"], -0.012, places=3)

                obs = backend.observe()
                self.assertTrue(obs["instruction"])
                cams = {c["camera_id"]: c for c in obs["cameras"]}
                self.assertEqual(set(cams), {"agentview", "wrist"})
                for cid, entry in cams.items():
                    self.assertEqual((entry["width"], entry["height"]), (512, 512), cid)
                    self.assertTrue(Path(entry["rgb_path"]).is_file(), cid)
                    with np.load(entry["depth_path"]) as npz:
                        depth = npz["depth"]
                    self.assertEqual(depth.shape, (512, 512), cid)
                    self.assertTrue(np.isfinite(depth).all(), cid)
                    self.assertGreater(float(depth.max()), 0.1, cid)

                # Verify 7-DoF URDF FK matches MuJoCo gripper0_grip_site to < 1e-4 m and < 1e-3 rad
                arm_state = obs["robot_state"]["arms"]["arm0"]
                q_rad = arm_state["q_rad"]
                T_urdf = fk(model, q_rad, link=TCP_LINK)
                mj_pos = np.asarray(arm_state["ee_pose"]["position_m"], dtype=np.float64)
                pos_err = float(np.linalg.norm(T_urdf[:3, 3] - mj_pos))
                self.assertLess(pos_err, 1e-4, f"FK position error {pos_err} >= 1e-4 m")

                from scipy.spatial.transform import Rotation

                urdf_quat = Rotation.from_matrix(T_urdf[:3, :3]).as_quat().tolist()
                ori_err = _quat_angle_rad(urdf_quat, arm_state["ee_pose"]["quaternion_xyzw"])
                self.assertLess(ori_err, 1e-3, f"FK orientation error {ori_err} >= 1e-3 rad")

                # Video recording + partial gripper + evaluator + observer snapshot
                rec_info = backend.record(True)
                self.assertTrue(rec_info["enabled"])

                identity = {"run_id": "r_stdio", "episode_id": "ep_stdio", "env_instance_id": "i1", "env_generation": 1}
                move_act = backend.compile_action(
                    "move",
                    {
                        "arm": "arm0",
                        "action_id": "act_stdio_1",
                        "observation_id": "obs_stdio",
                        "position_m": [float(mj_pos[0] + 0.03), float(mj_pos[1]), float(mj_pos[2] + 0.02)],
                        "quaternion_xyzw": arm_state["ee_pose"]["quaternion_xyzw"],
                        "gripper_goal_normalized": 0.35,
                    },
                    identity,
                    caps,
                )
                receipt = backend.execute(move_act)
                self.assertEqual(receipt["status"], "completed", receipt)

                obs_after = backend.observe()
                raw_pos = obs_after["robot_state"]["arms"]["arm0"]["gripper"]["position_m"]
                measured_ap = float(raw_pos[0]) + abs(float(raw_pos[1])) if isinstance(raw_pos, list) else float(raw_pos)
                self.assertGreater(measured_ap, 0.020)
                self.assertLess(measured_ap, 0.036)

                eval_res = backend.evaluate(final_check=True)
                self.assertEqual(eval_res["audience"], "observer_only")
                self.assertIn("task_success", eval_res)
                self.assertTrue(Path(eval_res["evaluation_path"]).is_file())

                snap = backend.observer_snapshot()
                self.assertEqual(snap["audience"], "observer_only")
                self.assertGreater(len(snap["objects"]), 0)

                if shutil.which("ffmpeg"):
                    vid = backend.save_video()
                    self.assertEqual(len(vid["videos"]), 2)
                    for item in vid["videos"]:
                        p = Path(item["video_path"])
                        self.assertTrue(p.is_file() and p.stat().st_size > 0)
                    self.assertTrue(Path(vid["composite"]["video_path"]).is_file())
            finally:
                backend.close()

    def test_socket_daemon_fast_reset_table_z_and_perturbations(self) -> None:
        assert self._socket_path is not None
        cfg = Config.load(REPO_ROOT / "configs" / "libero.example.json", environ={})

        with tempfile.TemporaryDirectory(prefix="ni_rua_libero_pool_") as tmp:
            hello_before = LiberoWorkerClient.hello(self._socket_path)
            self.assertEqual(hello_before["status"], "ok")
            self.assertEqual(hello_before["max_episodes"], 2)
            self.assertFalse(hello_before["active_episode"])
            ep_before = int(hello_before["episodes_served"])

            # 1. Verify scene table_z_m across libero_spatial (table -0.012) and libero_object (floor 0.000)
            b_spatial = LiberoBackend(
                config=cfg,
                run_dir=Path(tmp) / "ep1",
                socket_path=self._socket_path,
                task="libero_spatial:0",
                seed=0,
                layout_id=0,
            )
            try:
                caps1 = b_spatial.start()
                hello_during = LiberoWorkerClient.hello(self._socket_path)
                self.assertTrue(hello_during["active_episode"])
                self.assertEqual(hello_during["episodes_served"], ep_before + 1)
                self.assertAlmostEqual(caps1["scene_calibration"]["table_z_m"], -0.012, places=3)
                obs_sp = b_spatial.observe()
                unperturbed_cam = np.asarray(
                    obs_sp["camera_pose_sample"]["cameras"]["agentview"]["T_world_camera"], dtype=np.float64
                )
                unperturbed_q = list(obs_sp["robot_state"]["arms"]["arm0"]["q_rad"])
            finally:
                close_ep1 = b_spatial.close()
                self.assertTrue(close_ep1["closed"])
                self.assertTrue(close_ep1["pooled_socket"])
                self.assertEqual(close_ep1["episodes_served"], ep_before + 1)

            # 2. Fast same-BDDL reset WITH LIBERO-Plus / LIBERO-PRO perturbations & language_override
            t0 = time.monotonic()
            b_pert = LiberoBackend(
                config=cfg,
                run_dir=Path(tmp) / "ep2",
                socket_path=self._socket_path,
                task="libero_spatial:0",
                seed=0,
                layout_id=0,
                language_override="perturbed instruction: place the black bowl onto the plate",
                perturbation={
                    "robot_init_qpos_delta": [0.06, -0.04, 0.03, 0.0, 0.0, 0.04, -0.05],
                    "camera_orbit": {
                        "camera": "agentview",
                        "pivot": [0.0, 0.0, 0.8],
                        "yaw_deg": 18.0,
                        "pitch_deg": 6.0,
                        "distance_scale": 1.08,
                    },
                    "lighting": {
                        "ambient_scale": 0.85,
                        "diffuse_scale": 0.90,
                        "pos_delta": [0.05, -0.05, 0.03],
                    },
                },
            )
            try:
                caps_pert = b_pert.start()
                reset_dt = time.monotonic() - t0
                self.assertLess(reset_dt, 1.0, f"Pooled same-BDDL reset took {reset_dt:.3f}s")
                self.assertEqual(
                    b_pert.language_instruction,
                    "perturbed instruction: place the black bowl onto the plate",
                )
                pert_obs = b_pert.observe()
                pert_cam = np.asarray(
                    pert_obs["camera_pose_sample"]["cameras"]["agentview"]["T_world_camera"], dtype=np.float64
                )
                self.assertAlmostEqual(float(np.linalg.det(pert_cam[:3, :3])), 1.0, places=5)
                self.assertGreater(float(np.linalg.norm(pert_cam[:3, 3] - unperturbed_cam[:3, 3])), 0.05)

                pert_q = pert_obs["robot_state"]["arms"]["arm0"]["q_rad"]
                self.assertAlmostEqual(pert_q[0] - unperturbed_q[0], 0.06, places=3)
            finally:
                close_ep2 = b_pert.close()
                self.assertTrue(close_ep2["needs_recycle"])
            hello_after_2 = LiberoWorkerClient.hello(self._socket_path)
            self.assertTrue(hello_after_2["needs_recycle"])
            self.assertGreaterEqual(hello_after_2["episodes_served"], 2)

            # 3. Verify next unperturbed reset restores pristine camera & robot state cleanly
            b_clean = LiberoBackend(
                config=cfg,
                run_dir=Path(tmp) / "ep3",
                socket_path=self._socket_path,
                task="libero_spatial:0",
                seed=0,
                layout_id=0,
            )
            try:
                b_clean.start()
                clean_obs = b_clean.observe()
                clean_cam = np.asarray(
                    clean_obs["camera_pose_sample"]["cameras"]["agentview"]["T_world_camera"], dtype=np.float64
                )
                self.assertLess(float(np.linalg.norm(clean_cam - unperturbed_cam)), 1e-6)
            finally:
                b_clean.close()

            # 4. Verify libero_object floor scene table_z_m == 0.0
            b_obj = LiberoBackend(
                config=cfg,
                run_dir=Path(tmp) / "ep4",
                socket_path=self._socket_path,
                task="libero_object:0",
                seed=0,
                layout_id=0,
            )
            try:
                caps_obj = b_obj.start()
                self.assertAlmostEqual(caps_obj["scene_calibration"]["table_z_m"], 0.0, places=3)
            finally:
                b_obj.close()

    def test_harness_all_18_tools_motion_guards_and_zero_internal_errors(self) -> None:
        assert self._socket_path is not None
        cfg = Config.load(REPO_ROOT / "configs" / "libero.example.json", environ={})
        specs = configured_tool_specs(cfg)

        with tempfile.TemporaryDirectory(prefix="ni_rua_libero_harness_") as tmp:
            store = Store(
                Path(tmp) / "store",
                {"run_id": "r_h1", "episode_id": "ep_h1", "env_instance_id": "inst_1", "env_generation": 1},
            )
            backend = LiberoBackend(
                config=cfg,
                run_dir=Path(tmp) / "run",
                socket_path=self._socket_path,
                task="libero_spatial:0",
                seed=0,
                layout_id=0,
            )
            harness = Harness(
                robot=backend,
                store=store,
                config=cfg,
                task_plan_required=True,
                tool_specs=specs,
            )
            runner = ProgramRunner(harness.program_services())
            harness.programs = runner
            seq = 0

            def call(name: str, args: dict) -> object:
                nonlocal seq
                seq += 1
                return harness.invoke(name, args, self._call_id(name, seq), role="actor")

            def latest_arm_state() -> dict:
                rs = json.loads(store.resolve_artifact(harness.latest["robot_state_ref"])[1])
                return rs["arms"]["arm0"]

            def receipt_of(resp: object) -> dict:
                data = resp.data
                return data["receipt"] if isinstance(data.get("receipt"), dict) else data

            try:
                harness.start()
                harness.enable_function_tool_search()

                catalog_names = {t["name"] for t in harness.tool_registry.catalog(harness, role="actor")}
                expected_18 = {
                    "task_plan",
                    "robot_observe",
                    "robot_capabilities",
                    "robot_information",
                    "robot_contact_overlay",
                    "robot_action_status",
                    "robot_wait",
                    "history_list_actions",
                    "history_view_observation",
                    "history_read_trace",
                    "history_annotate_action",
                    "geometry_query",
                    "program_run",
                    "program_status",
                    "vision_compare",
                    "robot_action",
                    "operation_memory",
                    "tool_search",
                }
                self.assertEqual(catalog_names, expected_18)

                # 1. Unlock all 5 deferred tools via tool_search
                deferred = [
                    "robot_capabilities",
                    "robot_information",
                    "history_read_trace",
                    "history_annotate_action",
                    "vision_compare",
                ]
                ts_res = call(TOOL_SEARCH_NAME, {"names": deferred})
                self.assertEqual(set(ts_res.data["unlocked_tools"]), set(deferred))

                # 2. robot_information + robot_capabilities + operation_memory
                for section in ("default", "kinematics", "geometry", "dynamics", "control", "hardware", "all"):
                    ri = call("robot_information", {"section": section})
                    self.assertTrue(ri.data["available"], section)
                rc = call("robot_capabilities", {"detail": "full"})
                self.assertTrue(rc.success)
                om = call("operation_memory", {"request": {"kind": "help"}})
                self.assertTrue(om.success)

                # 3. robot_observe (obs_1) & verify zero privileged leakage
                obs1_resp = call("robot_observe", {})
                reject_privileged(obs1_resp.data, cfg.features.observation_profile)
                obs1 = harness.latest
                reject_privileged(obs1, cfg.features.observation_profile)
                obs1_id = obs1["observation_id"]
                ee0 = latest_arm_state()["ee_pose"]
                p0 = list(ee0["position_m"])
                q0 = list(ee0["quaternion_xyzw"])

                # 4. task_plan (submit + read + update)
                tp_sub = call(
                    "task_plan",
                    {
                        "operation": "submit",
                        "observation_id": obs1_id,
                        "plan": "Task understanding: pick up the black bowl and place it on the plate. Approach + completion: grasp bowl rim, lift cleanly, lower onto plate, release.",
                        "ledger": [],
                    },
                )
                self.assertTrue(tp_sub.success)
                self.assertTrue(call("task_plan", {"operation": "read"}).success)

                # 5. geometry_query (camera_pose, fk on agentview + wrist, project_points of live TCP, pixel_ray)
                gq_cam = call(
                    "geometry_query",
                    {"request": {"kind": "camera_pose", "observation_id": obs1_id, "camera_id": "agentview"}},
                )
                T_wc = np.asarray(gq_cam.data["value"]["T_world_camera"], dtype=np.float64)
                self.assertAlmostEqual(float(np.linalg.det(T_wc[:3, :3])), 1.0, places=5)

                gq_fk = call(
                    "geometry_query",
                    {"request": {"kind": "fk", "observation_id": obs1_id, "camera_id": "wrist"}},
                )
                self.assertTrue(gq_fk.success)
                self.assertLess(gq_fk.data["value"]["translation_residual_m"], 1e-4)
                self.assertLess(gq_fk.data["value"]["rotation_residual_rad"], 1e-3)

                gq_proj = call(
                    "geometry_query",
                    {
                        "request": {
                            "kind": "project_points",
                            "observation_id": obs1_id,
                            "camera_id": "agentview",
                            "points_m": [p0],
                        }
                    },
                )
                uv = gq_proj.data["value"]["pixels"][0]
                self.assertTrue(40.0 <= uv[0] <= 472.0 and 40.0 <= uv[1] <= 472.0, f"Unexpected gripper pixel {uv}")

                gq_ray = call(
                    "geometry_query",
                    {
                        "request": {
                            "kind": "pixel_ray",
                            "observation_id": obs1_id,
                            "camera_id": "agentview",
                            "pixel": [256, 256],
                        }
                    },
                )
                self.assertTrue(gq_ray.success)

                # 6. robot_contact_overlay on arm0
                ov_resp = call(
                    "robot_contact_overlay",
                    {
                        "observation_id": obs1_id,
                        "camera_id": "agentview",
                        "arms": ["arm0"],
                        "selections": {"arm0": {"contact_region": "finger_middle"}},
                    },
                )
                self.assertTrue(ov_resp.success)

                # 7. robot_action mode="reach" (reachable vs unreachable, zero physics steps)
                reach_resp = call(
                    "robot_action",
                    {
                        "mode": "reach",
                        "observation_id": obs1_id,
                        "stages": [
                            {
                                "id": "ok_pose",
                                "kind": "move",
                                "arm": "arm0",
                                "position_m": [p0[0] + 0.04, p0[1], p0[2] - 0.02],
                                "quaternion_xyzw": q0,
                            },
                            {
                                "id": "too_far",
                                "kind": "move",
                                "arm": "arm0",
                                "position_m": [0.98, 0.78, 1.05],
                                "quaternion_xyzw": q0,
                            },
                        ],
                    },
                )
                self.assertTrue(reach_resp.data["stage_results"]["ok_pose"]["ok"])
                self.assertFalse(reach_resp.data["stage_results"]["too_far"]["ok"])

                # 8. robot_action mode="preflight" with grasp_angles
                pre_resp = call(
                    "robot_action",
                    {
                        "mode": "preflight",
                        "observation_id": obs1_id,
                        "stages": [
                            {
                                "id": "grasp_pre",
                                "kind": "move",
                                "arm": "arm0",
                                "grasp_position_m": [p0[0] + 0.02, p0[1], p0[2] - 0.03],
                                "angles_deg": [0.0, 0.0, -90.0],
                                "contact_region": "finger_middle",
                            }
                        ],
                    },
                )
                self.assertTrue(pre_resp.success)

                # 9. >12 stages (16 stages!) preview -> execute with partial gripper 0.35 + speed_scale
                stages_16 = []
                for idx in range(15):
                    dx = 0.004 * (idx + 1)
                    stages_16.append(
                        {
                            "id": f"m{idx}",
                            "kind": "move",
                            "arm": "arm0",
                            "position_m": [p0[0] + dx, p0[1], p0[2]],
                            "quaternion_xyzw": q0,
                            "speed_scale": 1.4,
                            "tolerance_preset": "coarse",
                        }
                    )
                stages_16.append(
                    {
                        "id": "partial_grip",
                        "kind": "gripper",
                        "arm": "arm0",
                        "goal_normalized": 0.35,
                    }
                )
                prev_16 = call(
                    "robot_action",
                    {"mode": "preview", "observation_id": obs1_id, "stages": stages_16},
                )
                plan_id = prev_16.data["plan_id"]
                exec_16 = call(
                    "robot_action",
                    {
                        "mode": "execute",
                        "plan_id": plan_id,
                        "expected_cursor": 0,
                        "expected_revision": prev_16.data.get("plan_revision", 0),
                    },
                )
                self.assertEqual(receipt_of(exec_16)["status"], "completed", exec_16.data)
                act_16_id = exec_16.data["action_id"]
                obs_after_16 = harness.latest["observation_id"]
                raw_ap_16 = latest_arm_state()["gripper"]["position_m"]
                ap_after_16 = float(raw_ap_16[0]) + abs(float(raw_ap_16[1])) if isinstance(raw_ap_16, list) else float(raw_ap_16)
                self.assertTrue(0.020 <= ap_after_16 <= 0.036, f"Expected ~0.028 m, got {ap_after_16}")

                # 10. robot_action_status, mode="inspect", robot_wait, vision_compare, history_*
                self.assertTrue(call("robot_action_status", {"action_id": act_16_id}).success)
                self.assertTrue(
                    call("robot_action", {"mode": "inspect", "action_id": act_16_id, "detail": "full"}).success
                )
                wait_resp = call("robot_wait", {"observation_id": obs_after_16, "native_steps": 3})
                self.assertEqual(receipt_of(wait_resp)["status"], "completed")
                obs_after_wait = harness.latest["observation_id"]

                vc_resp = call(
                    "vision_compare",
                    {
                        "before_observation_id": obs1_id,
                        "after_observation_id": obs_after_wait,
                        "camera_id": "agentview",
                        "roi": [64, 64, 384, 384],
                    },
                )
                self.assertTrue(vc_resp.success)
                self.assertTrue(call("history_list_actions", {}).success)
                self.assertTrue(call("history_view_observation", {"observation_id": obs1_id}).success)
                self.assertTrue(
                    call("history_annotate_action", {"action_id": act_16_id, "note": "verified 16-stage partial grip"}).success
                )
                self.assertTrue(call("history_read_trace", {"action_id": act_16_id, "kind": "actual_trace"}).success)

                # 11. max_downward_dz_m guard (stops early when descending past allowed downward window)
                cur_ee = latest_arm_state()["ee_pose"]
                cur_p = list(cur_ee["position_m"])
                dz_resp = call(
                    "robot_action",
                    {
                        "observation_id": harness.latest["observation_id"],
                        "auto_execute_if_clean": True,
                        "stages": [
                            {
                                "id": "drop_guarded",
                                "kind": "move",
                                "arm": "arm0",
                                "position_m": [cur_p[0], cur_p[1], cur_p[2] - 0.12],
                                "quaternion_xyzw": q0,
                                "max_downward_dz_m": 0.015,
                            }
                        ],
                    },
                )
                self.assertIn("downward_dz", json.dumps(dz_resp.data))

                # 12. stall_guard_preset="contact_stop" + max_upward_dz_m on table contact (z = -0.025 < table -0.012)
                contact_resp = call(
                    "robot_action",
                    {
                        "observation_id": harness.latest["observation_id"],
                        "auto_execute_if_clean": True,
                        "stages": [
                            {
                                "id": "above_clear_table",
                                "kind": "move",
                                "arm": "arm0",
                                "position_m": [0.35, -0.25, 0.06],
                                "quaternion_xyzw": q0,
                                "tolerance_preset": "relaxed",
                            },
                            {
                                "id": "touch_table",
                                "kind": "move",
                                "arm": "arm0",
                                "position_m": [0.35, -0.25, -0.025],
                                "quaternion_xyzw": q0,
                                "stall_guard_preset": "contact_stop",
                                "max_upward_dz_m": 0.08,
                                "tolerance_preset": "compliant",
                            },
                            {
                                "id": "lift_clear",
                                "kind": "move",
                                "arm": "arm0",
                                "position_m": [0.35, -0.25, 0.14],
                                "quaternion_xyzw": q0,
                                "tolerance_preset": "relaxed",
                            },
                        ],
                    },
                )
                self.assertEqual(receipt_of(contact_resp)["status"], "completed", contact_resp.data)

                # 13. stage_grip_guard (grip_guard_preset="non_empty" pauses/stops on empty air close!)
                grip_guard_resp = call(
                    "robot_action",
                    {
                        "observation_id": harness.latest["observation_id"],
                        "auto_execute_if_clean": True,
                        "stages": [
                            {
                                "id": "close_empty_air",
                                "kind": "gripper",
                                "arm": "arm0",
                                "goal_normalized": 0.0,
                                "grip_guard_preset": "non_empty",
                            },
                            {
                                "id": "never_reached_lift",
                                "kind": "move",
                                "arm": "arm0",
                                "position_m": [0.35, -0.25, 0.18],
                                "quaternion_xyzw": q0,
                            },
                        ],
                    },
                )
                self.assertIn(
                    receipt_of(grip_guard_resp).get("batch_status"),
                    ("awaiting_observation", "stopped", "paused"),
                    grip_guard_resp.data,
                )
                self.assertIn("aperture_below_min_m", json.dumps(grip_guard_resp.data))

                # 14. carry_slip_guard (pauses mid-move when jaw slips below min_aperture_m after closing stage!)
                slip_resp = call(
                    "robot_action",
                    {
                        "observation_id": harness.latest["observation_id"],
                        "auto_execute_if_clean": True,
                        "carry_slip_guard": True,
                        "carry_slip_scope": "until_release",
                        "min_aperture_m": 0.020,
                        "slip_consecutive_steps": 1,
                        "stages": [
                            {
                                "id": "arm_carry_guard",
                                "kind": "gripper",
                                "arm": "arm0",
                                "goal_normalized": 0.35,
                                "min_aperture_m": 0.020,
                            },
                            {
                                "id": "transfer_with_dropped_obj",
                                "kind": "move",
                                "arm": "arm0",
                                "position_m": [0.35, -0.10, 0.22],
                                "quaternion_xyzw": q0,
                                "gripper_goal_normalized": 0.0,
                                "gripper_motion": {"start_fraction": 0.0, "end_fraction": 0.2},
                            },
                        ],
                    },
                )
                self.assertIn(
                    receipt_of(slip_resp).get("batch_status"),
                    ("awaiting_observation", "stopped", "paused"),
                    slip_resp.data,
                )
                self.assertIn("carry_slip", json.dumps(slip_resp.data))

                # 15. program_run (NiRuaClient inside seccomp-confined Python) + program_status
                prog_resp = call(
                    "program_run",
                    {
                        "source": (
                            "from ni_rua_client import NiRuaClient\n"
                            "client = NiRuaClient()\n"
                            "caps = client.call('robot_capabilities', {'detail': 'full'})\n"
                            "print('PROG_ROBOT', caps['data']['robot_model_id'])\n"
                        ),
                        "timeout_s": 20,
                    },
                )
                self.assertEqual(prog_resp.data["status"], "succeeded", prog_resp.data)
                self.assertIn("PROG_ROBOT libero-franka-panda", json.dumps(prog_resp.data))
                job_id = prog_resp.data["job_id"]
                ps_resp = call("program_status", {"job_id": job_id})
                self.assertTrue(ps_resp.success)

                # ZERO internal runtime errors across the entire 18-tool session!
                self.assertEqual(harness.internal_runtime_errors, [])
            finally:
                runner.close()
                backend.close()
                store.close()

    def test_live_workbench_opencode_2_2_7_tool_search_and_program_run(self) -> None:
        import asyncio
        from unittest import mock

        from support.fake_model_servers import FakeResponsesServer, text_step, tool_step
        from ni_rua.apps.gateway.workbench import Workbench

        opencode_exe = (
            os.environ.get("NI_RUA_OPENCODE_EXECUTABLE")
            or str(Path("~/.local/share/ni-rua/opencode-2.2.7/opencode").expanduser())
        )
        if not Path(opencode_exe).is_file() or not os.access(opencode_exe, os.X_OK) or not shutil.which("rg"):
            self.skipTest("Pinned opencode 2.2.7 binary or rg not installed on this host")

        assert self._socket_path is not None
        script = [
            tool_step("ni_rua_tool_search", {"names": ["robot_information", "robot_capabilities"]}),
            tool_step("ni_rua_robot_information", {"section": "default"}),
            tool_step(
                "ni_rua_program_run",
                {
                    "source": (
                        "from ni_rua_client import NiRuaClient\n"
                        "c = NiRuaClient()\n"
                        "obs = c.call('robot_observe', {})\n"
                        "print('WB_PROG_OBS', bool(obs.get('success')))\n"
                    ),
                    "timeout_s": 20,
                },
            ),
            text_step("Verified simulation-only libero-franka-panda via OpenCode 2.2.7 + Workbench."),
        ]

        with FakeResponsesServer(script) as server, tempfile.TemporaryDirectory(prefix="ni_rua_wb_oc_") as tmp:
            jail_dir = Path(tmp) / "prog_jail"
            cfg = Config.load(REPO_ROOT / "configs" / "libero.example.json", environ={}).with_overrides(
                {
                    "features": {
                        "terminal": False,
                        "record_video": False,
                        "export_video": False,
                        "programs_dir": str(jail_dir),
                    },
                    "model": {"name": "Qwen3.8-27B", "effort": "medium"},
                    "agent": {
                        "backend": "opencode",
                        "opencode": {
                            "executable": opencode_exe,
                            "subagents": False,
                            "tool_search": True,
                            "allow_shell": False,
                            "allow_external_read": False,
                            "context_limit_tokens": 110592,
                            "output_limit_tokens": 16384,
                            "upstream": {
                                "source": "explicit",
                                "wire_api": "openai_responses",
                                "base_url": server.base_url,
                                "api_key_env": "TEST_OC_UPSTREAM_KEY",
                            },
                        },
                    },
                }
            )
            backend = LiberoBackend(
                config=cfg,
                run_dir=Path(tmp) / "robot_run",
                socket_path=self._socket_path,
                task="libero_spatial:0",
                seed=0,
                layout_id=0,
            )

            async def _drive() -> None:
                wb = Workbench(Path(tmp) / "wb_run", robot=backend, visual_review_model=None, config=cfg)
                try:
                    await wb.start(None, benchmark=True)
                    finished = await asyncio.to_thread(wb.session.turn_done.wait, 60.0)
                    self.assertTrue(finished, "OpenCode 2.2.7 turn did not finish within 60s")
                    self.assertEqual(wb.session.turn_result.get("status"), "completed", wb.session.turn_result)
                    self.assertTrue((wb.root / "actor" / "robot.urdf").is_file())
                    self.assertEqual(wb.harness.internal_runtime_errors, [])
                finally:
                    await wb.close()

            with mock.patch.dict(os.environ, {"TEST_OC_UPSTREAM_KEY": "placeholder-key"}):
                asyncio.run(_drive())

            post_reqs = [r for r in server.requests if r.method == "POST"]
            self.assertEqual(len(post_reqs), 4)
            req_bodies = [json.loads(r.body.decode("utf-8")) for r in post_reqs]
            t0_tools = req_bodies[0]["tools"]
            t0_names = [t["name"] for t in t0_tools]
            self.assertIn("ni_rua_tool_search", t0_names)
            self.assertIn("ni_rua_robot_observe", t0_names)
            self.assertIn("ni_rua_program_run", t0_names)
            self.assertNotIn("ni_rua_robot_information", t0_names)
            self.assertNotIn("bash", t0_names)
            self.assertNotIn("shell", t0_names)
            self.assertNotIn("subagent", t0_names)
            # Zero prefix forks across all 4 turns!
            for idx in (1, 2, 3):
                self.assertEqual(req_bodies[idx]["tools"], t0_tools, f"Turn {idx} tools mutated!")


if __name__ == "__main__":
    unittest.main()
