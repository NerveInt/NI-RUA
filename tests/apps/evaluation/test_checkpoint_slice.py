"""针对中间态切片（checkpoint slice）、感知重算与 Codex 原生 thread/fork 的单元测试。"""

import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from ni_rua.apps.evaluation import checkpoint, codex_fork, turn_probe
from ni_rua.apps.evaluation.checkpoint import (
    _truncate_sqlite_store,
    extract_prefix_commands,
    prepare_checkpoint_slice,
    prepare_store_slice,
)
from ni_rua.apps.evaluation.codex_fork import (
    ForkedCodexSession,
    _to_rollout_output,
    find_rollout_path,
    prepare_codex_rollout_slice,
    require_codex_archive,
    slice_rollout_jsonl,
)
from ni_rua.robot.backends.robodojo.native_replay import capture_mujoco_snapshot, restore_mujoco_snapshot
from ni_rua.apps.evaluation.turn_probe import (
    ProbeRobotAdapter,
    _check_slice_assertions,
    _extract_tool_error,
    _resolve_cutpoint_ids,
    probe_perception_batch,
    probe_perception_slice,
)

NI_RUA_TEST_TIER = "component"
from ni_rua.perception.external.object_table import grasp_hints_from
from ni_rua.core.storage.store import Store, open_read_store


class CheckpointSliceUnitTests(unittest.TestCase):
    def open_slice_runtime(self, recorded_features, overrides=None):
        """``turn_probe._open_slice_runtime`` of an archive whose effective_config.json records ``recorded_features``."""
        from ni_rua.core.config import Config
        directory = self.enterContext(tempfile.TemporaryDirectory())
        work_dir = Path(directory)
        identity = dict(run_id="r", episode_id="e", env_instance_id="i", env_generation=0)
        Store(work_dir, identity).close()
        (work_dir / "effective_config.json").write_text(json.dumps(dict(Config().to_dict(), features=recorded_features)))
        cfg, store, _, _ = turn_probe._open_slice_runtime(work_dir, overrides)
        self.addCleanup(store.close)
        return cfg.features

    def test_slice_of_an_archive_older_than_the_reply_switches_restores_them_off_unless_overridden(self):
        from ni_rua.core.config import FEATURE_ABSENT_VALUES as REPLY_DELIVERY_ABSENT
        from support.reply_defaults import DEFAULT_REPLY_FEATURES
        self.assertEqual(set(REPLY_DELIVERY_ABSENT), set(DEFAULT_REPLY_FEATURES) - {"actor_payload_profile"})
        recorded = {"sandbox_enabled": False}  # none of the reply keys, tool_catalog_profile or robot_urdf_delivery was recorded
        features = self.open_slice_runtime(recorded)
        self.assertEqual({name: getattr(features, name) for name in REPLY_DELIVERY_ABSENT}, REPLY_DELIVERY_ABSENT)
        self.assertEqual((features.tool_catalog_profile, features.robot_urdf_delivery), ("full_v1", "inline"))
        self.assertIs(features.sandbox_enabled, False)
        # a config override is applied after the restoration and can turn a switch back on
        features = self.open_slice_runtime(recorded, {"features": {"reply_same_as": True, "action_reply_profile": "summary_first"}})
        self.assertEqual({name: getattr(features, name) for name in REPLY_DELIVERY_ABSENT},
                         dict(REPLY_DELIVERY_ABSENT, reply_same_as=True, action_reply_profile="summary_first"))
        # an archive that recorded the keys keeps its values, including those that equal today's defaults
        recorded_on = dict(DEFAULT_REPLY_FEATURES, reply_float32=False)
        features = self.open_slice_runtime(recorded_on)
        self.assertEqual({name: getattr(features, name) for name in REPLY_DELIVERY_ABSENT},
                         {name: recorded_on[name] for name in REPLY_DELIVERY_ABSENT})
        # one recorded key does not drag the missing ones to today's defaults
        features = self.open_slice_runtime({"reply_float32": True})
        self.assertEqual({name: getattr(features, name) for name in REPLY_DELIVERY_ABSENT},
                         dict(REPLY_DELIVERY_ABSENT, reply_float32=True))
        # (b) an archive from after tool_catalog_profile existed but before robot_urdf_delivery keeps its profile and
        # restores the URDF inline; the reverse (file recorded, no profile) keeps file and restores full_v1
        features = self.open_slice_runtime({"tool_catalog_profile": "compact_v1"})
        self.assertEqual((features.tool_catalog_profile, features.robot_urdf_delivery), ("compact_v1", "inline"))
        features = self.open_slice_runtime({"robot_urdf_delivery": "file"})
        self.assertEqual((features.tool_catalog_profile, features.robot_urdf_delivery), ("full_v1", "file"))
        features = self.open_slice_runtime({"tool_catalog_profile": "layered_v3", "robot_urdf_delivery": "file"})
        self.assertEqual((features.tool_catalog_profile, features.robot_urdf_delivery), ("layered_v3", "file"))

    def test_slice_rollout_truncates_and_replaces_output_and_strips_unpaired_calls(self):
        """验证 rollout 裁剪能精确截断到目标 call_id、原地替换输出并剔除同批次未完成的并行调用。"""
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src_rollout.jsonl"
            dst = Path(tmp) / "dst_rollout.jsonl"
            lines = [
                {
                    "timestamp": "2026-10-26T00:00:00Z",
                    "type": "session_meta",
                    "payload": {
                        "id": "019a0000-0000-7000-8000-000000000001",
                        "dynamic_tools": [{"name": "old_tool"}],
                        "base_instructions": {"text": "old instructions"},
                    },
                },
                {
                    "timestamp": "2026-10-26T00:00:01Z",
                    "type": "response_item",
                    "payload": {"type": "function_call", "name": "geometry_query", "arguments": "{}", "call_id": "call_1"},
                },
                {
                    "timestamp": "2026-10-26T00:00:01Z",
                    "type": "response_item",
                    "payload": {"type": "function_call", "name": "robot_observe", "arguments": "{}", "call_id": "call_parallel_unanswered"},
                },
                {
                    "timestamp": "2026-10-26T00:00:02Z",
                    "type": "response_item",
                    "payload": {"type": "function_call_output", "call_id": "call_1", "output": [{"type": "inputText", "text": "old_output"}]},
                },
                {
                    "timestamp": "2026-10-26T00:00:03Z",
                    "type": "response_item",
                    "payload": {"type": "function_call_output", "call_id": "call_parallel_unanswered", "output": [{"type": "inputText", "text": "later"}]},
                },
                {
                    "timestamp": "2026-10-26T00:00:04Z",
                    "type": "response_item",
                    "payload": {"type": "function_call", "name": "robot_action", "arguments": "{}", "call_id": "call_2"},
                },
            ]
            src.write_text("\n".join(json.dumps(x) for x in lines) + "\n")

            new_out = {"success": True, "contentItems": [{"type": "inputText", "text": "recomputed_v2_output"}]}
            info = slice_rollout_jsonl(
                src,
                dst,
                until_call_id="call_1",
                replacement_output=new_out,
                dynamic_tools=[{"name": "new_tool"}],
                developer_instructions="new dev instructions",
            )
            self.assertEqual(info["source_thread_id"], "019a0000-0000-7000-8000-000000000001")
            self.assertTrue(info["replaced_output"])

            kept = [json.loads(line) for line in dst.read_text().splitlines() if line.strip()]
            # 仅保留 session_meta + call_1 + call_1_output（剔除了未在切点前回答的 call_parallel_unanswered 及后续 call_2）
            self.assertEqual(len(kept), 3)
            self.assertEqual(kept[0]["payload"]["dynamic_tools"], [{"name": "new_tool"}])
            self.assertEqual(kept[0]["payload"]["base_instructions"]["text"], "new dev instructions")
            self.assertEqual(kept[1]["payload"]["call_id"], "call_1")
            self.assertEqual(kept[2]["payload"]["call_id"], "call_1")
            self.assertEqual(kept[2]["payload"]["output"], [{"type": "input_text", "text": "recomputed_v2_output"}])

    def test_sqlite_truncation_and_prefix_commands_extraction(self):
        """验证 SQLite 截断与前置控制序列提取。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            identity = {"run_id": "r1", "episode_id": "ep1", "env_instance_id": "env1", "env_generation": 1}
            store = Store(root, identity)
            store.set_state("runtime", "identity", identity)

            obs1 = {"observation_id": "obs-1", "robot_state": {"sim_step": 0, "arm_joint_positions": {"left": [0.0] * 6}}}
            store.save("observation", "obs-1", obs1)
            store.event("observation.published", {}, {"observation_id": "obs-1"})

            trace_ref = store.put_artifact(
                json.dumps([{"sim_step": 1, "commands": {"left": {"joint_pos": [0.1] * 6}}}]).encode(),
                "dispatched_trace",
                "application/json",
                "observer",
            )
            store.save("action_request", "act-1", {"action_id": "act-1", "based_on_observation_id": "obs-1"})
            store.event("action.requested", {}, {"action_id": "act-1"})
            store.save(
                "action_receipt",
                "act-1",
                {"action_id": "act-1", "status": "completed", "result_observation_id": "obs-2", "dispatched_trace_ref": trace_ref},
            )
            obs2 = {"observation_id": "obs-2", "last_action_id": "act-1", "robot_state": {"sim_step": 1, "arm_joint_positions": {"left": [0.1] * 6}}}
            store.save("observation", "obs-2", obs2)
            store.event("observation.published", {}, {"observation_id": "obs-2", "tool_call_id": "call-cut"})

            # 切点之后的事件与动作
            store.save("action_request", "act-2", {"action_id": "act-2", "based_on_observation_id": "obs-2"})
            store.event("action.requested", {}, {"action_id": "act-2"})
            store.save("observation", "obs-3", {"observation_id": "obs-3", "robot_state": {"sim_step": 2}})
            store.event("observation.published", {}, {"observation_id": "obs-3"})

            prefix = extract_prefix_commands(store, until_action_id="act-1")
            self.assertEqual(prefix["command_count"], 1)
            self.assertEqual(prefix["target_sim_step"], 1)
            self.assertEqual(prefix["boundary_observation_id"], "obs-2")
            self.assertEqual(prefix["target_robot_state"]["arm_joint_positions"]["left"], [0.1] * 6)

            trunc = _truncate_sqlite_store(root / "events.sqlite", until_call_id="call-cut")
            self.assertTrue(trunc["truncated"])
            self.assertEqual(trunc["dropped_observations"], 1)
            self.assertEqual(trunc["dropped_actions"], 1)

    def test_grasp_hints_preserves_floating_object_height_without_silent_reanchor(self):
        """验证悬浮/叠放物体保留真实 3D 高度与 pad_z_m，不做静默桌面重投影。"""
        held_block = {
            "centre_m": [0.345, -0.215, 0.957],
            "centre_xy": [0.345, -0.215],
            "top_z_m": 0.994,
            "bottom_z_m": 0.919,
            "length_m": 0.128,
            "width_m": 0.044,
            "yaw_deg": 31.0,
            "support": "unknown",
            "flags": ["thin_or_on_support", "floating", "low_depth_confidence"],
        }
        hints = grasp_hints_from([held_block], 0.765)
        self.assertEqual(held_block["support"], "unknown")
        self.assertAlmostEqual(held_block["centre_m"][2], 0.957, places=4)
        self.assertGreater(hints["o1"]["pad_z_m"], 0.920)

    def test_prepare_checkpoint_slice_directory(self):
        """验证 prepare_checkpoint_slice 完整生成切片目录、checkpoint_spec.json、prefix_commands.jsonl 与截断后的 rollout。"""
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src_run"
            dst = Path(tmp) / "dst_slice"
            identity = {"run_id": "r1", "episode_id": "ep1", "env_instance_id": "env1", "env_generation": 1}
            store = Store(src, identity)
            store.set_state("runtime", "identity", identity)
            store.save("observation", "obs-1", {"observation_id": "obs-1", "robot_state": {"sim_step": 0}})
            store.event("observation.published", {}, {"observation_id": "obs-1", "tool_call_id": "call_cut"})
            store.close()
            (src / "manifest.json").write_text("{}\n")
            (src / "effective_config.json").write_text("{}\n")
            (src / "scene").mkdir(parents=True, exist_ok=True)
            (src / "actor").mkdir(parents=True, exist_ok=True)
            rollout_dir = src / "private" / "codex-home" / "sessions" / "2026" / "10" / "26"
            rollout_dir.mkdir(parents=True, exist_ok=True)
            rollout_file = rollout_dir / "rollout-2026-10-26T00-00-00-019a0000-0000-7000-8000-000000000001.jsonl"
            rollout_file.write_text(
                "\n".join([
                    json.dumps({"timestamp": "2026-10-26T00:00:00Z", "type": "session_meta", "payload": {"id": "019a0000-0000-7000-8000-000000000001"}}),
                    json.dumps({"timestamp": "2026-10-26T00:00:01Z", "type": "response_item", "payload": {"type": "function_call", "name": "robot_observe", "arguments": "{}", "call_id": "call_cut"}}),
                    json.dumps({"timestamp": "2026-10-26T00:00:02Z", "type": "response_item", "payload": {"type": "function_call_output", "call_id": "call_cut", "output": "ok"}}),
                    json.dumps({"timestamp": "2026-10-26T00:00:03Z", "type": "response_item", "payload": {"type": "function_call", "name": "robot_action", "arguments": "{}", "call_id": "call_later"}}),
                ]) + "\n"
            )

            res = prepare_checkpoint_slice(src, dst, until_call_id="call_cut", recompute_cutpoint=False)
            spec = res["spec"]
            self.assertEqual(spec["source_thread_id"], "019a0000-0000-7000-8000-000000000001")
            self.assertEqual(spec["rollout"]["kept_lines"], 3)
            self.assertTrue((dst / "checkpoint_spec.json").exists())
            self.assertTrue((dst / "prefix_commands.jsonl").exists())

    def test_slice_rollout_closes_open_turn_and_reindexes_ordinal(self):
        """验证在半回合切点处自动补齐 task_complete 闭合事件并重排 ordinal，消除 <turn_aborted> 合成噪音。"""
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.jsonl"
            dst = Path(tmp) / "dst.jsonl"
            lines = [
                {"ordinal": 0, "timestamp": "2026-10-26T00:00:00Z", "type": "session_meta", "payload": {"id": "tid-1"}},
                {"ordinal": 1, "timestamp": "2026-10-26T00:00:01Z", "type": "event_msg", "payload": {"type": "task_started", "turn_id": "turn-1", "started_at": 100}},
                {"ordinal": 2, "timestamp": "2026-10-26T00:00:02Z", "type": "response_item", "payload": {"type": "function_call", "name": "geometry_query", "call_id": "call_1"}},
                {"ordinal": 3, "timestamp": "2026-10-26T00:00:03Z", "type": "response_item", "payload": {"type": "function_call_output", "call_id": "call_1", "output": "ok"}},
                {"ordinal": 4, "timestamp": "2026-10-26T00:00:04Z", "type": "response_item", "payload": {"type": "function_call", "name": "robot_action", "call_id": "call_2"}},
            ]
            src.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
            info = slice_rollout_jsonl(src, dst, until_call_id="call_1", close_open_turn=True)
            self.assertTrue(info["closed_open_turn"])
            out_lines = [json.loads(x) for x in dst.read_text().splitlines() if x.strip()]
            self.assertEqual(len(out_lines), 6)
            self.assertEqual([x["ordinal"] for x in out_lines], [0, 1, 2, 3, 4, 5])
            self.assertEqual(out_lines[-1]["payload"]["type"], "task_complete")
            self.assertEqual(out_lines[-1]["payload"]["turn_id"], "turn-1")

    def test_mujoco_snapshot_capture_and_restore(self):
        """验证 MuJoCo 物理状态（qpos/qvel/ctrl/act/mocap）快照保存与毫秒级恢复。"""
        with tempfile.TemporaryDirectory() as tmp:
            snap_file = Path(tmp) / "mujoco_snapshot.npz"
            fake_data = SimpleNamespace(
                qpos=np.array([0.1, 0.2, 0.3], dtype=np.float64),
                qvel=np.array([0.01, 0.02, 0.03], dtype=np.float64),
                ctrl=np.array([1.0, 2.0], dtype=np.float64),
                act=np.array([0.5], dtype=np.float64),
                mocap_pos=np.array([[0.3, -0.1, 0.8]], dtype=np.float64),
                mocap_quat=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
                time=12.5,
            )
            forwarded = []
            fake_sim = SimpleNamespace(data=fake_data, forward=lambda: forwarded.append(True))
            adapter = SimpleNamespace(env=SimpleNamespace(sim=fake_sim), robot=None, control_step=250, native_actions_dispatched=10)

            cap = capture_mujoco_snapshot(adapter, snap_file)
            self.assertTrue(cap["saved"])

            # 清零状态后验证恢复
            fake_data.qpos[:] = 0.0
            fake_data.qvel[:] = 0.0
            fake_data.time = 0.0
            adapter.control_step = 0
            adapter.native_actions_dispatched = 0

            res = restore_mujoco_snapshot(adapter, snap_file)
            self.assertTrue(res["restored"])
            self.assertEqual(res["sim_step"], 250)
            self.assertEqual(adapter.native_actions_dispatched, 10)
            self.assertAlmostEqual(fake_data.time, 12.5)
            np.testing.assert_allclose(fake_data.qpos, [0.1, 0.2, 0.3])
            np.testing.assert_allclose(fake_data.qvel, [0.01, 0.02, 0.03])
            self.assertEqual(len(forwarded), 1)

    def test_probe_robot_adapter_and_slice_assertions(self):
        """验证 ProbeRobotAdapter 的离线批次预检结构与黄金切片断言校验器。"""
        adapter = ProbeRobotAdapter(image_settings={})
        raw = adapter.preview_batch({"stages": [{"id": "approach", "kind": "move", "arm": "right"}, {"id": "close", "kind": "gripper", "arm": "right"}]})
        self.assertEqual(raw["status"], "planned")
        self.assertEqual(raw["planned_native_steps"], 35)
        self.assertEqual(len(adapter.previewed_actions), 1)

        sample_result = {
            "new_summary": {
                "max_stereo_skew_m": 0.008,
                "labelled_count": 6,
                "floating_unknown_count": 1,
                "rows": [
                    {"id": "o1", "support": "table", "flags": []},
                    {"id": "o2", "support": "unknown", "flags": ["floating"]},
                ],
            }
        }
        checks = _check_slice_assertions(
            sample_result,
            {
                "max_skew_below": 0.015,
                "min_labelled_count": 5,
                "max_floating_unknown": 2,
                "expect_support": {"o1": "table"},
                "expect_floating": {"o1": False, "o2": True},
            },
        )
        self.assertEqual(len(checks), 6)
        self.assertTrue(all(c["passed"] for c in checks))

        # 验证双臂过近时 ProbeRobotAdapter 生成 inter_arm_proxy_intersection 风险
        risky_raw = adapter.preview_batch({
            "stages": [{
                "id": "both_close",
                "kind": "move",
                "arm": "both",
                "arms": {
                    "left": {"grasp_position_m": [0.30, 0.01, 0.80]},
                    "right": {"grasp_position_m": [0.30, -0.01, 0.80]},
                },
            }]
        })
        self.assertEqual(risky_raw["geometry_review"]["status"], "risk")
        self.assertIn("inter_arm_proxy_intersection", risky_raw["geometry_review"]["risks"])

        # 验证 turn_fork / live_slice 出现工具调用报错时自动使断言失败（避免假阳性 PASS）
        err_msg = _extract_tool_error({"success": False, "contentItems": [{"type": "inputText", "text": "cached_batch_commands_cannot_be_changed"}]})
        self.assertEqual(err_msg, "cached_batch_commands_cannot_be_changed")
        live_with_err = {
            "mode": "live_slice",
            "tool_error_count": 1,
            "executed_slice_actions": [{"action_id": "act-1", "status": "completed"}],
        }
        err_checks = _check_slice_assertions(live_with_err, {"live_completed_min": 1})
        self.assertFalse(all(c["passed"] for c in err_checks))

    def test_codex_fork_isolation_and_module_boundaries(self):
        """验证 codex_fork 承载 Codex 专属逻辑，且 checkpoint / turn_probe 顶层无 Codex 耦合。"""
        import inspect
        from ni_rua.cortex.agents.codex import bridge as codex_bridge, session as codex_session

        self.assertIs(codex_fork.CodexSession, codex_session.CodexSession)
        self.assertIs(codex_fork.build_bridge_command, codex_bridge.build_bridge_command)
        self.assertIn("build_bridge_command(", inspect.getsource(ForkedCodexSession.start_forked))
        self.assertEqual(
            _to_rollout_output({"contentItems": [{"type": "inputText", "text": "hi"}, {"type": "inputImage", "imageUrl": "data:image/png;base64,abc"}]}),
            [{"type": "input_text", "text": "hi"}, {"type": "input_image", "image_url": "data:image/png;base64,abc"}],
        )

        # 兼容重导出应与 codex_fork 保持同一对象
        for name in (
            "require_codex_archive",
            "find_rollout_path",
            "_to_rollout_output",
            "slice_rollout_jsonl",
            "ForkedCodexSession",
            "prepare_codex_rollout_slice",
        ):
            self.assertIs(getattr(checkpoint, name), getattr(codex_fork, name))

        # checkpoint.py 与 turn_probe.py 模块顶层不得存在 codex_fork / cortex.agents.codex / openai_codex 导入
        forbidden_fragments = ("codex_fork", "cortex.agents.codex", "openai_codex")
        for mod_path in (Path(checkpoint.__file__), Path(turn_probe.__file__)):
            tree = ast.parse(mod_path.read_text(encoding="utf-8"), filename=str(mod_path))
            top_imports = []
            for node in tree.body:
                if isinstance(node, ast.Import):
                    top_imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    mod_name = node.module or ""
                    top_imports.append(mod_name)
                    top_imports.extend(f"{mod_name}.{alias.name}" for alias in node.names)
            for imp in top_imports:
                for frag in forbidden_fragments:
                    self.assertNotIn(frag, imp, f"{mod_path.name} has top-level import {imp}")

        self.assertFalse(hasattr(turn_probe, "ForkedCodexSession"))
        self.assertFalse(hasattr(turn_probe, "require_codex_archive"))

    def test_prepare_store_slice_is_backend_neutral_without_rollouts(self):
        """验证 prepare_store_slice 可直接对非 Codex（如 claude_code）且无 rollout 的归档切片。"""
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "claude_run"
            dst = Path(tmp) / "store_slice"
            rejected_dst = Path(tmp) / "rejected_codex_slice"
            identity = {"run_id": "r-cc", "episode_id": "ep1", "env_instance_id": "env1", "env_generation": 1}
            store = Store(src, identity)
            try:
                store.set_state("runtime", "identity", identity)
                store.set_state("runtime", "agent", {"backend": "claude_code", "thread_id": "cc-1", "instance_id": "agent-1"})
                obs1 = {"observation_id": "obs-1", "robot_state": {"sim_step": 0, "arm_joint_positions": {"left": [0.0] * 6}}}
                store.save("observation", "obs-1", obs1)
                store.event("observation.published", {}, {"observation_id": "obs-1"})

                trace_ref = store.put_artifact(
                    json.dumps([{"sim_step": 1, "commands": {"left": {"joint_pos": [0.2] * 6}}}]).encode(),
                    "dispatched_trace",
                    "application/json",
                    "observer",
                )
                store.save("action_request", "act-1", {"action_id": "act-1", "based_on_observation_id": "obs-1"})
                store.event("action.requested", {}, {"action_id": "act-1"})
                store.save(
                    "action_receipt",
                    "act-1",
                    {"action_id": "act-1", "status": "completed", "result_observation_id": "obs-2", "dispatched_trace_ref": trace_ref},
                )
                obs2 = {"observation_id": "obs-2", "last_action_id": "act-1", "robot_state": {"sim_step": 1, "arm_joint_positions": {"left": [0.2] * 6}}}
                store.save("observation", "obs-2", obs2)
                store.event("observation.published", {}, {"observation_id": "obs-2", "tool_call_id": "call_cut"})

                # 切点之后的数据
                store.save("action_request", "act-2", {"action_id": "act-2", "based_on_observation_id": "obs-2"})
                store.event("action.requested", {}, {"action_id": "act-2"})
                store.save("observation", "obs-3", {"observation_id": "obs-3", "robot_state": {"sim_step": 2}})
                store.event("observation.published", {}, {"observation_id": "obs-3"})
            finally:
                store.close()

            (src / "manifest.json").write_text("{}\n")
            (src / "effective_config.json").write_text("{}\n")
            (src / "scene").mkdir(parents=True, exist_ok=True)
            (src / "scene" / "meta.json").write_text('{"ok": true}\n')
            (src / "actor").mkdir(parents=True, exist_ok=True)
            (src / "actor" / "notes.txt").write_text("actor note\n")

            res = prepare_store_slice(src, dst, until_call_id="call_cut", recompute_cutpoint=False)
            spec = res["spec"]
            self.assertEqual(spec["until_call_id"], "call_cut")
            self.assertEqual(spec["until_action_id"], "act-1")
            self.assertEqual(spec["prefix_command_count"], 1)
            self.assertEqual(spec["target_sim_step"], 1)
            self.assertEqual(spec["boundary_observation_id"], "obs-2")
            self.assertNotIn("rollout", spec)
            self.assertNotIn("source_thread_id", spec)
            self.assertTrue(spec["sqlite"]["truncated"])
            self.assertEqual(spec["sqlite"]["dropped_observations"], 1)
            self.assertEqual(spec["sqlite"]["dropped_actions"], 1)
            self.assertTrue((dst / "checkpoint_spec.json").exists())
            self.assertTrue((dst / "prefix_commands.jsonl").exists())
            self.assertTrue((dst / "scene" / "meta.json").exists())
            self.assertTrue((dst / "actor" / "notes.txt").exists())
            with open_read_store(dst) as sliced:
                self.assertIsNone(sliced.read_state("runtime", "agent"))

            # prepare_checkpoint_slice 仍拒绝非 Codex 归档且不创建目标目录
            with self.assertRaisesRegex(ValueError, "^checkpoint_requires_codex_archive:claude_code$"):
                prepare_checkpoint_slice(src, rejected_dst, until_call_id="call_cut")
            self.assertFalse(rejected_dst.exists())

    def test_prepare_store_slice_rejects_missing_cutpoint_and_scrubs_full_copy_runtime_state(self):
        """验证 prepare_store_slice 对缺失切点快速抛出 KeyError（不遗留半成品），且全量切片同样清理瞬态 runtime 状态。"""
        tmp = self.enterContext(tempfile.TemporaryDirectory())
        src = Path(tmp) / "src_run"
        identity = {"run_id": "r-full", "episode_id": "ep1", "env_instance_id": "env1", "env_generation": 1}
        store = Store(src, identity)
        self.addCleanup(store.close)
        store.set_state("runtime", "identity", identity)
        store.set_state("runtime", "codex", {"thread_id": "parent-thread", "app_server_instance_id": "parent-app"})
        for transient_id, payload in (
            ("agent", {"backend": "claude_code", "thread_id": "cc-1", "instance_id": "agent-1"}),
            ("app_server", {"pid": 1234}),
            ("robot", {"pid": 5678}),
            ("model_requests", {"in_flight": 1}),
            ("scene_obstacles", {"items": ["obs_a"]}),
        ):
            store.set_state("runtime", transient_id, payload)
        store.save("observation", "obs-1", {"observation_id": "obs-1", "robot_state": {"sim_step": 0}})
        store.event("observation.published", {}, {"observation_id": "obs-1", "tool_call_id": "call_1"})
        store.save("action_request", "act-1", {"action_id": "act-1", "based_on_observation_id": "obs-1"})
        store.event("action.requested", {}, {"action_id": "act-1"})
        store.save("action_receipt", "act-1", {"action_id": "act-1", "status": "completed", "result_observation_id": "obs-1"})
        store.close()

        (src / "manifest.json").write_text("{}\n")
        (src / "effective_config.json").write_text("{}\n")
        (src / "scene").mkdir(parents=True, exist_ok=True)
        (src / "scene" / "meta.json").write_text('{"ok": true}\n')
        (src / "actor").mkdir(parents=True, exist_ok=True)
        (src / "actor" / "notes.txt").write_text("actor note\n")

        missing_call_dst = Path(tmp) / "missing_call_slice"
        with self.assertRaises(KeyError):
            prepare_store_slice(src, missing_call_dst, until_call_id="call_missing")
        self.assertFalse(missing_call_dst.exists())

        preexisting_empty_dst = Path(tmp) / "preexisting_empty_slice"
        preexisting_empty_dst.mkdir()
        with self.assertRaises(KeyError):
            prepare_store_slice(src, preexisting_empty_dst, until_action_id="act_missing")
        self.assertTrue(preexisting_empty_dst.exists())
        self.assertEqual(list(preexisting_empty_dst.iterdir()), [])
        self.assertFalse((preexisting_empty_dst / "checkpoint_spec.json").exists())

        full_dst = Path(tmp) / "full_copy_slice"
        res = prepare_store_slice(src, full_dst)
        self.assertFalse(res["spec"]["sqlite"]["truncated"])
        self.assertIsNone(res["spec"]["until_call_id"])
        self.assertEqual(res["spec"]["until_action_id"], "act-1")
        self.assertTrue((full_dst / "checkpoint_spec.json").exists())
        with open_read_store(full_dst) as full_store:
            for transient_id in ("app_server", "robot", "model_requests", "scene_obstacles", "agent"):
                self.assertIsNone(full_store.read_state("runtime", transient_id), transient_id)
            self.assertEqual(full_store.read_state("runtime", "identity"), identity)
            self.assertEqual(full_store.read_state("runtime", "codex")["thread_id"], "parent-thread")

    def test_prepare_codex_rollout_slice_and_neutral_perception_probe(self):
        """验证 prepare_codex_rollout_slice 独立工作，且 turn_probe.probe_perception_slice/batch 对非 Codex 归档可用。"""
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "codex_src"
            dst = Path(tmp) / "codex_dst"
            rollout_dir = src / "private" / "codex-home" / "sessions" / "2026" / "10" / "26"
            rollout_dir.mkdir(parents=True, exist_ok=True)
            rollout_file = rollout_dir / "rollout-2026-10-26T00-00-00-tid-2.jsonl"
            rollout_file.write_text(
                "\n".join([
                    json.dumps({"timestamp": "2026-10-26T00:00:00Z", "type": "session_meta", "payload": {"id": "tid-2"}}),
                    json.dumps({"timestamp": "2026-10-26T00:00:01Z", "type": "response_item", "payload": {"type": "function_call", "name": "robot_observe", "call_id": "call_1"}}),
                    json.dumps({"timestamp": "2026-10-26T00:00:02Z", "type": "response_item", "payload": {"type": "function_call_output", "call_id": "call_1", "output": "old"}}),
                ]) + "\n"
            )
            info = prepare_codex_rollout_slice(
                src,
                dst,
                until_call_id="call_1",
                replacement_output={"contentItems": [{"type": "inputText", "text": "new"}]},
            )
            self.assertEqual(info["source_thread_id"], "tid-2")
            self.assertTrue(info["replaced_output"])
            found = find_rollout_path(dst)
            self.assertTrue(found.exists())

            # 非 Codex 归档执行 probe_perception_slice / probe_perception_batch（无任何 rollout）
            cc_run = Path(tmp) / "cc_perception"
            identity = {"run_id": "r-cc2", "episode_id": "ep1", "env_instance_id": "env1", "env_generation": 1}
            store = Store(cc_run, identity)
            try:
                store.set_state("runtime", "identity", identity)
                store.set_state("runtime", "agent", {"backend": "claude_code", "thread_id": "cc-2", "instance_id": "agent-2"})
                obs = {"observation_id": "obs-1", "robot_state": {"sim_step": 0}, "cameras": {}}
                store.save("observation", "obs-1", obs)
                store.save("tool_call", "call_obs", {"params": {"callId": "call_obs", "tool": "robot_observe", "arguments": {"observation_id": "obs-1"}}})
                raw_ref = store.put_artifact(
                    json.dumps({
                        "data": {"observation_id": "obs-1"},
                        "wire_response": {"success": True, "contentItems": [{"type": "inputText", "text": json.dumps({"observation_id": "obs-1", "object_table": {"rows": []}})}]},
                    }).encode(),
                    "tool_result_raw",
                    "application/json",
                    "observer",
                )
                store.save("tool_result", "call_obs", {"correlation": {"tool_call_id": "call_obs"}, "raw_ref": raw_ref})
            finally:
                store.close()
            (cc_run / "effective_config.json").write_text("{}\n")
            (cc_run / "manifest.json").write_text("{}\n")

            single = probe_perception_slice(cc_run, "call_obs")
            self.assertEqual(single["kind"], "obs3d_attachments")
            self.assertEqual(single["call_id"], "call_obs")
            self.assertEqual(single["tool"], "robot_observe")
            self.assertEqual(single["new_row_count"], 0)
            batch = probe_perception_batch([(cc_run, "call_obs")], max_workers=1)
            self.assertEqual(len(batch), 1)
            self.assertEqual(batch[0]["call_id"], "call_obs")

    def test_recompute_cutpoint_propagates_config_errors_and_turn_probe_cleans_up_on_error(self):
        import shutil
        from unittest.mock import patch
        from ni_rua.apps.evaluation import turn_probe
        from ni_rua.core.config import ConfigError

        tmp = self.enterContext(tempfile.TemporaryDirectory())
        src = Path(tmp) / "cfg_err_src"
        identity = {"run_id": "r-cfg", "episode_id": "ep1", "env_instance_id": "env1", "env_generation": 1}
        store = Store(src, identity)
        self.addCleanup(store.close)
        store.set_state("runtime", "identity", identity)
        store.save("observation", "obs-1", {"observation_id": "obs-1", "robot_state": {"sim_step": 0}, "cameras": {}})
        store.save(
            "tool_call",
            "call_plan",
            {"params": {"callId": "call_plan", "tool": "task_plan", "arguments": {"plan": "test"}}},
        )
        store.event("task_plan.submitted", {"plan_id": "p1"}, {"tool_call_id": "call_plan"})
        store.close()
        (src / "manifest.json").write_text("{}\n")
        (src / "effective_config.json").write_text("{}\n")

        # Unsupported tool with valid config_overrides degrades recomputed to None cleanly
        ok_dst = Path(tmp) / "ok_unsupported_tool"
        ok_res = prepare_store_slice(src, ok_dst, until_call_id="call_plan", recompute_cutpoint=True)
        self.assertIsNone(ok_res["recomputed"])

        # Invalid config_overrides (ConfigError subclassing ValueError) MUST NOT be swallowed!
        bad_dst = Path(tmp) / "bad_cfg_override"
        with self.assertRaises(ConfigError):
            prepare_store_slice(
                src,
                bad_dst,
                until_call_id="call_plan",
                recompute_cutpoint=True,
                config_overrides={"features": {"nonexistent_feature_xyz": True}},
            )

        # probe_forked_turn cleans up auto-created temp work_dir when hydration fails
        created_dirs = []
        orig_mkdtemp = tempfile.mkdtemp

        def track_mkdtemp(*args, **kwargs):
            d = orig_mkdtemp(*args, **kwargs)
            created_dirs.append(Path(d))
            return d

        with patch("ni_rua.apps.evaluation.codex_fork.require_codex_archive"), \
             patch("ni_rua.apps.evaluation.turn_probe.prepare_checkpoint_slice", return_value={"spec": {"source_thread_id": "t1"}}), \
             patch("ni_rua.apps.evaluation.turn_probe._extract_archived_next_calls", return_value=[]), \
             patch("ni_rua.apps.evaluation.turn_probe._hydrate_slice_harness", side_effect=RuntimeError("hydration_boom")), \
             patch("ni_rua.apps.evaluation.turn_probe.tempfile.mkdtemp", side_effect=track_mkdtemp):
            # Copy valid sqlite into auto work_dir via _open_slice_runtime stub that opens a real Store
            orig_open = turn_probe._open_slice_runtime

            def open_with_db(wdir, overrides):
                shutil.copy2(ok_dst / "events.sqlite", Path(wdir) / "events.sqlite")
                (Path(wdir) / "effective_config.json").write_text("{}\n")
                return orig_open(wdir, overrides)

            with patch("ni_rua.apps.evaluation.turn_probe._open_slice_runtime", side_effect=open_with_db):
                with self.assertRaisesRegex(RuntimeError, "hydration_boom"):
                    turn_probe.probe_forked_turn(ok_dst, until_call_id="call_plan", work_dir=None)
        self.assertEqual(len(created_dirs), 1)
        self.assertFalse(created_dirs[0].exists())


if __name__ == "__main__":
    unittest.main()
