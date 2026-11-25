"""
- test_collision_witness_artifacts.py: CPU fixtures exercise native witness capture, real worker IPC and actor reads.
- test_explicit_resume_and_program_guard.py: 暂停恢复校验合一、显式选择阶段续跑与 NiRuaClient 完成断言回归测试。
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch
import numpy as np
from ni_rua.robot.motion.motion_failure import archive_collision_witnesses, solver_snapshot
from ni_rua.runtime.programs.runner import ProgramRunner
from ni_rua.robot.backends.robodojo.robodojo import RobotBackend
from ni_rua.robot.backends.robodojo.robot_process import RobotProcess
from support.motion_fixtures import motion_action, replan_adapter
from runtime.harness.test_motion_failure import native_result
from support.motion_fixtures import native_fk_fixture
import runtime.actions.test_unified_actions as unified_fixture
import threading
from ni_rua.runtime.programs.ni_rua_client import ActionIncompleteError
from ni_rua.runtime.actions.unified_actions import compile_request, dispatch as typed_dispatch
from support.motion_fixtures import DualArmHarnessBase
from support.motion_fixtures import batch, move
from support.motion_fixtures import preview_adapter

NI_RUA_TEST_TIER = "component"


# --- Folded from tests/test_collision_witness_artifacts.py ---

def native_failure():
    """Keep eight FK samples from the native adapter's selected failed candidate."""
    adapter, robot, _, _ = replan_adapter(enabled=False)
    result = native_result("trajopt")
    result.js_solution.position = np.repeat(result.js_solution.position, 4, axis=2) / 4
    diagnostics = [dict(stage="trajopt", waypoint_index=0, **solver_snapshot(result, "trajopt"))]
    with native_fk_fixture(robot.arm_joints_name) as (planner, calls):
        adapter.robot.planner[robot.robot_name] = planner
        feedback = adapter._failure_feedback(motion_action(), "left", 0, diagnostics)
    assert calls["scene_reads"] == 0
    return dict(status="failed", reason="path_infeasible", failed_stage_id="reach", failure_feedback=feedback)


class WitnessWorker:
    """Transport fixture: the real worker dispatcher owns the native capture."""
    def __init__(self, arguments):
        pass

    def capabilities(self):
        return dict(ready=True)

    def preview_batch(self, action):
        return native_failure()

    def close(self, progress):
        pass


class CollisionWitnessArtifactTests(unittest.TestCase):
    setUp = unified_fixture.UnifiedActionTests.setUp
    request = unified_fixture.UnifiedActionTests.request
    preview = unified_fixture.UnifiedActionTests.preview
    native_execution = unified_fixture.UnifiedActionTests.native_execution

    def actor_call(self, arguments, call="witness"):
        self.h.thread_id = "thread"
        return self.h.handle_tool(dict(threadId="thread", turnId="turn", callId=call,
                                      tool="robot_action", arguments=arguments))

    def payload(self, reply):
        # the delivered reply read back: compact tables decoded, same_as pointers expanded
        from support.reply_defaults import reply_values
        return reply_values(reply["contentItems"])[0]

    def witness(self, result):
        return result["failure_feedback"]["candidates"][0]["constraint_witness"]

    def test_worker_preview_full_witnesses_are_readable_by_actor_program(self):
        # Use the actual backend operation method, JSONL dispatcher and reader.
        backend = RobotBackend.__new__(RobotBackend)
        RobotProcess.__init__(backend, self.store.root / "worker")
        self.addCleanup(backend.close)
        source = ("import json\nfrom ni_rua.robot.backends.robodojo.worker_protocol import serve_worker, WIRE_PREFIX\n"
                  "from runtime.programs.test_program_guards_and_witness import WitnessWorker\n"
                  "serve_worker(WitnessWorker, None, lambda value: "
                  "print(WIRE_PREFIX + json.dumps(value, allow_nan=False), flush=True))\n")
        from support.paths import REPO_ROOT, TESTS_ROOT
        root = REPO_ROOT
        environment = dict(os.environ, PYTHONPATH=os.pathsep.join((str(root), str(TESTS_ROOT))))
        backend.start_process([sys.executable, "-c", source], cwd=root, environment=environment)
        self.robot.preview_batch = backend.preview_batch
        reply = self.actor_call(self.request())
        result = self.payload(reply)
        self.assertEqual((result["status"], result["reason"]), ("failed", "path_infeasible"))
        witness = self.witness(result)
        self.assertEqual(len(witness["collision_model_witnesses"]), 8)
        self.assertEqual(witness["collision_model_witness_count"], 8)
        self.assertEqual(witness["omitted_witness_count"], 0)
        self.assertNotIn("_full_collision_model_witnesses", json.dumps(reply))
        reference = witness["collision_model_witnesses_ref"]
        self.assertTrue(reference.startswith("artifact:ref_"), reference)
        runner = ProgramRunner(self.h.program_services())
        self.h.programs = runner
        self.addCleanup(runner.close)
        source = ("import json\nfrom ni_rua_client import NiRuaClient\n"
                  f"details = json.loads(NiRuaClient().artifact({reference!r}))\n"
                  "assert details['candidate_only'] and details['sampled_only']\n"
                  "assert details['frame_id'] == 'planner_base'\n"
                  "assert details['arm'] == 'left' and details['failed_stage_id'] == 'reach'\n"
                  "hits = details['collision_model_witnesses']\n"
                  "assert len(hits) == 8\n"
                  "assert sorted(hit['trajectory_sample_index'] for hit in hits) == list(range(8))\n"
                  "assert all(hit['kind'] == 'self_collision' for hit in hits)\n"
                  "print('read all eight collision witnesses')\n")
        job = runner.run(dict(source=source, timeout_s=60),
                         dict(thread_id="thread", turn_id="turn", tool_call_id="read-witnesses"))
        self.assertEqual(job["status"], "succeeded", job.get("stderr_text"))
        self.assertIn("read all eight collision witnesses", job["stdout_text"])
        self.assertEqual(self.robot.executions, 0)

    def test_optional_archive_failure_keeps_preview_and_execution_results(self):
        original_put = self.store.put_artifact
        def fail_witness(data, kind, *args, **kwargs):
            if kind == "planner_collision_witnesses":
                raise OSError("private filesystem path must not enter actor context")
            return original_put(data, kind, *args, **kwargs)

        self.robot.preview_batch = Mock(side_effect=lambda _: native_failure())
        with patch.object(self.store, "put_artifact", side_effect=fail_witness):
            reply = self.actor_call(self.request(), "failed-preview")
        self.assert_unavailable(self.payload(reply))
        plan = self.preview()
        self.native_execution()
        self.robot.execute.return_value.update(native_failure())
        args = dict(mode="execute", plan_id=plan["plan_id"], action_id="failed-execution")
        with patch.object(self.store, "put_artifact", side_effect=fail_witness):
            reply = self.actor_call(args, "failed-execution")
        self.assert_unavailable(self.payload(reply)["receipt"])
        persisted = self.store.read("action_receipt", "failed-execution")
        self.assert_unavailable(persisted)
        self.assertEqual(persisted["completed_native_steps"], 2)
        self.assertEqual(self.actor_call(args, "failed-execution"), reply)
        self.assertEqual(self.robot.execute.call_count, 1)

    def assert_unavailable(self, result):
        self.assertEqual((result["status"], result["reason"]), ("failed", "path_infeasible"))
        witness = self.witness(result)
        self.assertEqual(len(witness["collision_model_witnesses"]), 8)
        self.assertEqual(witness["collision_model_witness_count"], 8)
        self.assertEqual(witness["collision_model_witnesses_unavailable"],
                         dict(reason="full_witness_archive_failed", error_type="OSError"))
        self.assertNotIn("collision_model_witnesses_ref", witness)
        self.assertNotIn("_full_collision_model_witnesses", json.dumps(result))
        self.assertNotIn("private filesystem", json.dumps(result))

    def test_privileged_complete_witness_is_not_archived_or_exposed(self):
        result = native_failure()
        # Only the optional full evidence is tainted, not the bounded public summary.
        self.witness(result)["_full_collision_model_witnesses"][-1]["object_poses"] = {"hidden": [1, 2, 3]}
        with patch.object(self.store, "put_artifact", wraps=self.store.put_artifact) as put:
            archive_collision_witnesses(self.store, result, self.h.observation_profile)
        put.assert_not_called()
        self.assertEqual(result["status"], "failed")
        witness = self.witness(result)
        self.assertEqual(witness["collision_model_witnesses_unavailable"]["error_type"], "ValueError")
        self.assertNotIn("object_poses", json.dumps(result))
        self.assertNotIn("_full_collision_model_witnesses", json.dumps(result))

    def test_tainted_short_and_full_witnesses_do_not_leak_through_preview(self):
        result = native_failure()
        witness = self.witness(result)
        witness['_full_collision_model_witnesses'][0]['object_poses'] = {'hidden': [1, 2, 3]}
        self.assertIn('object_poses', witness['collision_model_witnesses'][0])
        self.robot.preview_batch = Mock(return_value=result)
        reply = self.actor_call(self.request(), 'unsafe-witness-preview')
        self.assertNotIn('object_poses', json.dumps(reply))
        shown = self.payload(reply)
        self.assertEqual(shown['status'], 'failed')
        self.assertNotIn('collision_model_witnesses', self.witness(shown))
        self.assertEqual(self.witness(shown)['collision_model_witnesses_unavailable']['error_type'], 'ValueError')
        self.assertEqual(self.robot.executions, 0)

    def test_invalid_optional_evidence_does_not_erase_acknowledged_execution(self):
        plan = self.preview()
        self.native_execution()
        self.robot.execute.return_value.update(native_failure())
        self.witness(self.robot.execute.return_value)["_full_collision_model_witnesses"][-1]["evaluator_state"] = {"hidden": True}
        reply = self.actor_call(dict(mode="execute", plan_id=plan["plan_id"], action_id="invalid-evidence"))
        receipt = self.payload(reply)["receipt"]
        persisted = self.store.read("action_receipt", "invalid-evidence")
        for result in (receipt, persisted):
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["completed_native_steps"], 2)
            witness = self.witness(result)
            self.assertEqual(witness["collision_model_witnesses_unavailable"]["error_type"], "ValueError")
            self.assertNotIn("collision_model_witnesses_ref", witness)
            self.assertNotIn("evaluator_state", json.dumps(result))
            self.assertNotIn("_full_collision_model_witnesses", json.dumps(result))
        self.assertEqual(self.robot.execute.call_count, 1)

    def test_legacy_feedback_without_complete_field_stays_unchanged(self):
        result = native_failure()
        self.witness(result).pop("_full_collision_model_witnesses")
        original = deepcopy(result)
        with patch.object(self.store, "put_artifact") as put:
            archive_collision_witnesses(self.store, result, self.h.observation_profile)
        put.assert_not_called()
        self.assertEqual(result, original)



# --- Folded from tests/test_explicit_resume_and_program_guard.py ---

class ExplicitResumeAndProgramGuardTests(unittest.TestCase):
    setUp = DualArmHarnessBase.setUp

    def _stages(self):
        return [
            dict(id='approach', kind='move', arm='left', grasp_position_m=[.20, -.10, .82], angles_deg=[0, 0, 0]),
            dict(id='check', kind='observe', arm='left', reason='检查抓取对准'),
            dict(id='grasp_lift', kind='move', arm='left', grasp_position_m=[.20, -.10, .76],
                 angles_deg=[0, 0, 0], depends_on=['approach', 'check']),
            dict(id='place', kind='move', arm='left', grasp_position_m=[.28, -.05, .80],
                 angles_deg=[0, 0, 0], depends_on=['grasp_lift']),
        ]

    def test_paused_batch_tolerates_small_settling_drift_and_rejects_extra_native_steps(self):
        adapter, _, dispatched = preview_adapter()
        action = batch([
            move('one', angle=-.125),
            dict(id='look', kind='observe', reason='check'),
            move('two', angle=-.25),
        ])
        plan = adapter.preview_batch(action)
        first = adapter.execute(dict(action, batch_plan_id=plan['plan_id']), lambda _: None, threading.Event())
        self.assertEqual((first['status'], first['reason']), ('stopped', 'observation_checkpoint'))
        self.assertEqual(len(dispatched), 3)

        # 模拟暂停期间时钟推进与微小静止沉降（0.03 rad > 旧阈值 0.02 rad 且 <= 0.10 rad，6 mm > 旧阈值 5 mm 且 <= 15 mm）
        adapter.control_step += 1
        base_state = adapter.state

        def settled_state():
            value = deepcopy(base_state())
            if len(dispatched) == 3:
                value['arms']['left']['q_rad'][1] += .03
                value['arms']['left']['ee_pose']['position_m'][2] -= .006
            return value

        adapter.state = settled_state
        resumed = adapter.execute(dict(action, batch_plan_id=plan['plan_id']), lambda _: None, threading.Event())
        self.assertEqual(resumed['status'], 'completed')
        self.assertEqual(resumed['resume_validation']['status'], 'valid')

        # 模拟暂停期间发生了额外原生动作步：必须被 _validate_paused_state 拦截
        adapter2, _, _ = preview_adapter()
        plan2 = adapter2.preview_batch(action)
        first2 = adapter2.execute(dict(action, batch_plan_id=plan2['plan_id']), lambda _: None, threading.Event())
        self.assertEqual(first2['status'], 'stopped')
        adapter2.env.take_action_cnt[0] += 1
        rejected = adapter2.execute(dict(action, batch_plan_id=plan2['plan_id']), lambda _: None, threading.Event())
        self.assertEqual(rejected['status'], 'rejected')
        self.assertEqual(rejected['reason'], 'batch_plan_stale_state_requires_new_preview')
        validation = adapter2._batch_plans[plan2['plan_id']]['resume_validation']
        self.assertEqual(validation['status'], 'invalid')
        self.assertEqual(validation['reason'], 'native_action_steps_changed_during_pause')

    def test_paused_batch_first_step_jump_guard_catches_unsafe_drift(self):
        adapter, _, _ = preview_adapter()
        action = batch([
            move('one', angle=-.125),
            dict(id='look', kind='observe', reason='check'),
            move('two', angle=-.25),
        ])
        plan = adapter.preview_batch(action)
        first = adapter.execute(dict(action, batch_plan_id=plan['plan_id']), lambda _: None, threading.Event())
        self.assertEqual(first['status'], 'stopped')

        # 关节反方向漂移 0.08 rad：虽然在 0.10 rad 静态容差内，但叠加下一阶段首步位移后超过单步上限 0.10 rad
        adapter.control_step += 1
        base_state = adapter.state

        def unsafe_settled_state():
            value = deepcopy(base_state())
            value['arms']['left']['q_rad'][0] += .08
            return value

        adapter.state = unsafe_settled_state
        failed = adapter.execute(dict(action, batch_plan_id=plan['plan_id']), lambda _: None, threading.Event())
        self.assertEqual(failed['status'], 'failed')
        self.assertEqual(failed['reason'], 'batch_resume_first_joint_step_exceeds_limit')

    def test_backend_preview_batch_supports_source_plan_continuation_with_overrides(self):
        adapter, _, _ = preview_adapter()
        action = batch([
            move('one', angle=-.125),
            dict(id='look', kind='observe', reason='check'),
            move('two', angle=-.25, depends_on=['one']),
        ])
        plan = adapter.preview_batch(action)
        first = adapter.execute(dict(action, batch_plan_id=plan['plan_id']), lambda _: None, threading.Event())
        self.assertEqual(first['status'], 'stopped')

        cont_action = dict(
            type='BatchAction',
            source_plan_id=plan['plan_id'],
            start_from_stage_id='two',
            stage_overrides=dict(two=dict(action=move('two', angle=-.20)['action'])),
        )
        cont_plan = adapter.preview_batch(cont_action)
        self.assertEqual(cont_plan['status'], 'planned')
        self.assertEqual(list(cont_plan['stage_costs']), ['two'])
        self.assertEqual(cont_plan['continuation']['source_plan_id'], plan['plan_id'])
        self.assertEqual(cont_plan['continuation']['start_from_stage_id'], 'two')
        resumed = adapter.execute(dict(cont_action, batch_plan_id=cont_plan['plan_id']), lambda _: None, threading.Event())
        self.assertEqual(resumed['status'], 'completed')
        self.assertEqual(resumed['completed_stage_ids'], ['two'])

    def test_unified_action_next_action_and_explicit_stage_continuation(self):
        def preview_side_effect(action):
            stage_ids = [s['id'] for s in action['stages']]
            full = dict(
                schema_version='candidate-trajectory-v1',
                source_kind='candidate_command_trajectory_not_measurement',
                frame_id='env_origin',
                tcp_config_id='arx-x5-link6-v1',
                joint_names={'left': [], 'right': []},
                samples=[dict(native_step=0, arms={arm: dict(q_rad=[0.] * 6, jaw_goal_normalized=1., phase='initial')
                                                   for arm in ('left', 'right')})],
                geometry=dict(status='unknown'),
                inter_arm_proximity=dict(status='unknown'),
                execution_binding='exact_cached_rows; no replanning or resampling',
            )
            return dict(
                status='planned',
                plan_id='native-' + '-'.join(stage_ids),
                planned_native_steps=len(stage_ids) * 2,
                trajectory_data=full,
                trajectory_evidence=dict(available=True),
                stage_costs={sid: dict(nominal_native_steps=2) for sid in stage_ids},
            )

        self.robot.preview_batch = Mock(side_effect=preview_side_effect)
        with patch('ni_rua.runtime.actions.inspection._trajectory_image', return_value=(dict(status='unavailable'), [])):
            initial_plan = typed_dispatch(
                self.h,
                dict(observation_id=self.h.latest['observation_id'], stages=self._stages()),
                {'tool_call_id': 'preview-1'},
            ).data
        self.assertEqual(initial_plan['plan_id'], 'p1')

        # 执行到 check 阶段暂停，验证 next_action 提示信息
        self.robot.execute = Mock(return_value=dict(
            status='stopped',
            reason='observation_checkpoint',
            executed_native_steps=2,
            completed_native_steps=2,
            batch_status='awaiting_observation',
            batch_cursor=2,
            completed_stage_ids=['approach', 'check'],
            pending_stage_ids=['grasp_lift', 'place'],
            stage_results={
                'approach': dict(status='completed', completed_steps=2),
                'check': dict(status='completed', completed_steps=0),
                'grasp_lift': dict(status='pending', completed_steps=0),
                'place': dict(status='pending', completed_steps=0),
            },
            terminal_observation=self.robot.observe(),
        ))
        exec_res = typed_dispatch(
            self.h,
            dict(mode='execute', plan_id='p1', action_id='act-pause'),
            {'tool_call_id': 'exec-1'},
        ).data
        next_action = exec_res['next_action']
        self.assertEqual(next_action['completed_stage_ids'], ['approach', 'check'])
        self.assertIsNone(next_action['failed_stage_id'])
        self.assertEqual(next_action['remaining_stage_ids'], ['grasp_lift', 'place'])
        self.assertEqual(next_action['start_from_stage_id'], 'grasp_lift')
        self.assertEqual(next_action['preview_remaining']['source_plan_id'], 'p1')
        self.assertEqual(next_action['preview_remaining']['start_from_stage_id'], 'grasp_lift')

        # 1) 使用字典形式的 stage_overrides 在 preview 模式下续跑剩余阶段
        cont_args = dict(
            observation_id=self.h.latest['observation_id'],
            source_plan_id='p1',
            start_from_stage_id='grasp_lift',
            stage_overrides=dict(
                grasp_lift=dict(grasp_position_m=[.21, -.10, .75], speed_preset='slow'),
            ),
        )
        for role in ('actor', 'program'):
            self.h.tool_registry.validate('robot_action', cont_args, role=role)
        compiled = compile_request(self.h, cont_args)
        self.assertEqual([s['id'] for s in compiled['stages']], ['grasp_lift', 'place'])
        # 已完成的 approach/check 依赖被自动剥离，未完成的内部依赖保留
        self.assertEqual(compiled['stages'][0].get('depends_on', []), [])
        self.assertEqual(compiled['stages'][1]['depends_on'], ['grasp_lift'])
        self.assertEqual(compiled['stages'][0]['action']['speed_scale'], .8)

        with patch('ni_rua.runtime.actions.inspection._trajectory_image', return_value=(dict(status='unavailable'), [])):
            cont_plan = typed_dispatch(self.h, cont_args, {'tool_call_id': 'preview-2'}).data
        self.assertEqual(cont_plan['plan_id'], 'p2')
        self.assertEqual(cont_plan['continuation_of']['source_plan_id'], 'p1')
        self.assertEqual(cont_plan['continuation_of']['start_from_stage_id'], 'grasp_lift')
        self.assertEqual(cont_plan['continuation_of']['remaining_stage_ids'], ['grasp_lift', 'place'])
        self.assertEqual(cont_plan['continuation_of']['overridden_stage_ids'], ['grasp_lift'])

        # 2) 使用 stage_overrides 在 direct 模式下从 place 续跑（底层同时支持数组形式归一化）
        from ni_rua.runtime.actions.plan_continuation import normalize_stage_overrides
        self.assertEqual(
            normalize_stage_overrides([dict(id='place', grasp_position_m=[.30, -.04, .81])]),
            {'place': dict(grasp_position_m=[.30, -.04, .81])},
        )
        self.robot.execute = Mock(return_value=dict(
            status='completed',
            reason=None,
            executed_native_steps=2,
            completed_native_steps=2,
            batch_status='completed',
            batch_cursor=2,
            completed_stage_ids=['place'],
            pending_stage_ids=[],
            stage_results={'place': dict(status='completed', completed_steps=2)},
            terminal_observation=self.robot.observe(),
        ))
        direct_cont_args = dict(
            mode='direct',
            observation_id=self.h.latest['observation_id'],
            source_plan_id='p2',
            start_from_stage_id='place',
            stage_overrides={'place': dict(grasp_position_m=[.30, -.04, .81])},
        )
        for role in ('actor', 'program'):
            self.h.tool_registry.validate('robot_action', direct_cont_args, role=role)
        with patch('ni_rua.runtime.actions.inspection._trajectory_image', return_value=(dict(status='unavailable'), [])):
            direct_res = typed_dispatch(self.h, direct_cont_args, {'tool_call_id': 'direct-cont'}).data
        self.assertEqual(direct_res['receipt']['status'], 'completed')
        self.assertEqual(direct_res['receipt']['completed_stage_ids'], ['place'])

    def test_ni_rua_client_require_completed_stops_program_loop_on_incomplete_action(self):
        def preview_side_effect(action):
            full = dict(
                schema_version='candidate-trajectory-v1',
                source_kind='candidate_command_trajectory_not_measurement',
                frame_id='env_origin',
                tcp_config_id='arx-x5-link6-v1',
                joint_names={'left': [], 'right': []},
                samples=[dict(native_step=0, arms={arm: dict(q_rad=[0.] * 6, jaw_goal_normalized=1., phase='initial')
                                                   for arm in ('left', 'right')})],
                geometry=dict(status='unknown'),
                inter_arm_proximity=dict(status='unknown'),
                execution_binding='exact_cached_rows; no replanning or resampling',
            )
            return dict(
                status='planned',
                plan_id='native-plan',
                planned_native_steps=2,
                trajectory_data=full,
                trajectory_evidence=dict(available=True),
                stage_costs={'step': dict(nominal_native_steps=2)},
            )

        self.robot.preview_batch = Mock(side_effect=preview_side_effect)
        # 第一次 direct 动作返回 failed（模拟轨迹跟踪失败），验证循环不会继续发第二次动作
        self.robot.execute = Mock(return_value=dict(
            status='failed',
            reason='joint_tracking_error',
            executed_native_steps=1,
            completed_native_steps=0,
            batch_status='failed',
            batch_cursor=0,
            failed_stage_id='step',
            completed_stage_ids=[],
            pending_stage_ids=['step'],
            stage_results={'step': dict(status='failed', completed_steps=0)},
            terminal_observation=self.robot.observe(),
        ))

        runner = self.h.programs = ProgramRunner(self.h.program_services())
        self.addCleanup(runner.close)
        obs_id = self.h.latest['observation_id']
        stage = dict(id='step', kind='move', arm='left', grasp_position_m=[.2, -.1, .8], angles_deg=[0, 0, 0])
        source = (
            "import json\n"
            "from ni_rua_client import NiRuaClient, ActionIncompleteError\n"
            "client = NiRuaClient()\n"
            f"obs_id = {obs_id!r}\n"
            f"stage = {stage!r}\n"
            "executed = 0\n"
            "try:\n"
            "    for _ in range(3):\n"
            "        executed += 1\n"
            "        res = client.action([stage], observation_id=obs_id, mode='direct', require_completed=True)\n"
            "        obs_id = res['data']['receipt']['result_observation_id']\n"
            "except ActionIncompleteError as exc:\n"
            "    print(json.dumps({\n"
            "        'caught': True,\n"
            "        'executed': executed,\n"
            "        'status': exc.receipt['status'],\n"
            "        'reason': exc.receipt['reason'],\n"
            "        'start_from': exc.receipt['next_action']['start_from_stage_id'],\n"
            "    }))\n"
        )
        with patch('ni_rua.runtime.actions.inspection._trajectory_image', return_value=(dict(status='unavailable'), [])):
            job = runner.run(
                dict(source=source, allow_direct_actions=True),
                dict(thread_id='thread', turn_id='turn', tool_call_id='prog-guard'),
            )
        self.assertEqual(job['status'], 'succeeded', job.get('stderr_text'))
        payload = json.loads(job['stdout_text'].strip())
        self.assertTrue(payload['caught'])
        self.assertEqual(payload['executed'], 1)
        self.assertEqual(payload['status'], 'failed')
        self.assertEqual(payload['reason'], 'joint_tracking_error')
        self.assertEqual(payload['start_from'], 'step')
        self.assertEqual(self.robot.execute.call_count, 1)
        recent = job['action_summary']['recent_actions'][0]
        self.assertEqual(recent['plan_id'], 'p1')
        self.assertEqual(recent['next_action']['start_from_stage_id'], 'step')
        self.assertEqual(recent['next_action']['preview_remaining']['source_plan_id'], 'p1')

    def test_ni_rua_client_direct_plan_failure_allows_candidate_fallback_loop(self):
        calls = {'n': 0}

        def preview_side_effect(action):
            calls['n'] += 1
            if calls['n'] == 1:
                return dict(
                    status='failed',
                    reason='path_infeasible',
                    failed_stage_id='step',
                    failed_arm='left',
                )
            full = dict(
                schema_version='candidate-trajectory-v1',
                source_kind='candidate_command_trajectory_not_measurement',
                frame_id='env_origin',
                tcp_config_id='arx-x5-link6-v1',
                joint_names={'left': [], 'right': []},
                samples=[dict(native_step=0, arms={arm: dict(q_rad=[0.] * 6, jaw_goal_normalized=1., phase='initial')
                                                   for arm in ('left', 'right')})],
                geometry=dict(status='unknown'),
                inter_arm_proximity=dict(status='unknown'),
                execution_binding='exact_cached_rows; no replanning or resampling',
            )
            return dict(
                status='planned',
                plan_id='native-plan-2',
                planned_native_steps=2,
                trajectory_data=full,
                trajectory_evidence=dict(available=True),
                stage_costs={'step': dict(nominal_native_steps=2)},
            )

        self.robot.preview_batch = Mock(side_effect=preview_side_effect)
        self.robot.execute = Mock(return_value=dict(
            status='completed',
            reason=None,
            executed_native_steps=2,
            completed_native_steps=2,
            batch_status='completed',
            batch_cursor=2,
            completed_stage_ids=['step'],
            pending_stage_ids=[],
            stage_results={'step': dict(status='completed', completed_steps=2)},
            terminal_observation=self.robot.observe(),
        ))
        runner = self.h.programs = ProgramRunner(self.h.program_services())
        self.addCleanup(runner.close)
        obs_id = self.h.latest['observation_id']
        s1 = dict(id='step', kind='move', arm='left', grasp_position_m=[.2, -.1, .8], angles_deg=[0, 0, -90])
        s2 = dict(id='step', kind='move', arm='left', grasp_position_m=[.2, -.1, .8], angles_deg=[0, 0, 90])
        source = (
            "import json\n"
            "from ni_rua_client import NiRuaClient\n"
            "client = NiRuaClient()\n"
            f"obs_id = {obs_id!r}\n"
            f"candidates = [{s1!r}, {s2!r}]\n"
            "results = []\n"
            "for cand in candidates:\n"
            "    res = client.action([cand], observation_id=obs_id, mode='direct')\n"
            "    status = (res.get('data') or {}).get('receipt', {}).get('status')\n"
            "    results.append((res.get('kind'), status))\n"
            "    if status == 'completed':\n"
            "        break\n"
            "print(json.dumps(results))\n"
        )
        with patch('ni_rua.runtime.actions.inspection._trajectory_image', return_value=(dict(status='unavailable'), [])):
            job = runner.run(
                dict(source=source, allow_direct_actions=True),
                dict(thread_id='thread', turn_id='turn', tool_call_id='prog-fallback'),
            )
        self.assertEqual(job['status'], 'succeeded', job.get('stderr_text'))
        self.assertEqual(json.loads(job['stdout_text'].strip()), [['plan', 'failed'], ['action', 'completed']])

    def test_dispatch_accepts_obstacle_and_clearance_fields_in_preview_continuation_and_repair(self):
        captured_actions = []

        def preview_side_effect(action):
            captured_actions.append(deepcopy(action))
            stage_ids = [s['id'] for s in action['stages']]
            full = dict(
                schema_version='candidate-trajectory-v1',
                source_kind='candidate_command_trajectory_not_measurement',
                frame_id='env_origin',
                tcp_config_id='arx-x5-link6-v1',
                joint_names={'left': [], 'right': []},
                samples=[dict(native_step=0, arms={arm: dict(q_rad=[0.] * 6, jaw_goal_normalized=1., phase='initial')
                                                   for arm in ('left', 'right')})],
                geometry=dict(status='unknown'),
                inter_arm_proximity=dict(status='unknown'),
                execution_binding='exact_cached_rows; no replanning or resampling',
            )
            return dict(
                status='planned',
                plan_id='native-' + '-'.join(stage_ids),
                planned_native_steps=len(stage_ids) * 2,
                trajectory_data=full,
                trajectory_evidence=dict(available=True),
                stage_costs={sid: dict(nominal_native_steps=2) for sid in stage_ids},
            )

        self.robot.preview_batch = Mock(side_effect=preview_side_effect)
        obs_id = self.h.latest['observation_id']
        table = dict(
            columns=['id', 'centre_m', 'top_z', 'bottom_z', 'length', 'width', 'yaw_deg', 'tilt_deg', 'support', 'quality'],
            table_z=0.765,
            rows=[
                ['o1', [0.00, -0.30, 0.810], 0.860, 0.765, 0.060, 0.060, 0.0, None, 'table', 'ok'],
                ['o2', [0.12, -0.30, 0.790], 0.820, 0.765, 0.050, 0.050, 0.0, None, 'table', 'ok'],
            ],
        )
        self.h._obs3d_memo = (obs_id, {'object_table': (table, None, {})})

        stages = [
            dict(id='s1', kind='move', arm='left', grasp_position_m=[.20, -.10, .82], angles_deg=[0, 0, 0],
                 clearance_preset='low'),
            dict(id='s2', kind='move', arm='left', grasp_position_m=[.25, -.05, .84], angles_deg=[0, 0, 0],
                 depends_on=['s1']),
        ]
        preview_args = dict(
            mode='preview',
            observation_id=obs_id,
            obstacle_object_ids=['o1', 'o2'],
            obstacle_policy='persist',
            ignore_obstacle_ids=['o2'],
            target_exemption_preset='tight',
            clearance_preset='high',
            stages=stages,
        )
        with patch('ni_rua.runtime.actions.inspection._trajectory_image', return_value=(dict(status='unavailable'), [])):
            p1 = typed_dispatch(self.h, preview_args, {'tool_call_id': 'obs-prev-1'}).data
        self.assertEqual(p1['plan_id'], 'p1')
        self.assertEqual([b['id'] for b in captured_actions[-1]['review_proxies']['obstacles']], ['o1'])
        self.assertAlmostEqual(captured_actions[-1]['stages'][0]['action']['path']['clearance_height_m'], 0.09)
        self.assertAlmostEqual(captured_actions[-1]['stages'][1]['action']['path']['clearance_height_m'], 0.18)
        self.assertAlmostEqual(captured_actions[-1]['stages'][0]['action']['path']['target_exemption_radius_m'], 0.030)

        # 续跑 source_plan_id：继承原计划的 obstacle_object_ids/ignore_obstacle_ids，同时支持切换 clearance_height_m 与 target_exemption_radius_m
        cont_args = dict(
            mode='preview',
            observation_id=obs_id,
            source_plan_id='p1',
            start_from_stage_id='s1',
            target_exemption_radius_m=0.065,
            clearance_height_m=0.14,
            stage_overrides=dict(
                s1=dict(clearance_height_m=0.11),
            ),
        )
        with patch('ni_rua.runtime.actions.inspection._trajectory_image', return_value=(dict(status='unavailable'), [])):
            p2 = typed_dispatch(self.h, cont_args, {'tool_call_id': 'obs-prev-2'}).data
        self.assertEqual(p2['plan_id'], 'p2')
        self.assertEqual([b['id'] for b in captured_actions[-1]['review_proxies']['obstacles']], ['o1'])
        self.assertAlmostEqual(captured_actions[-1]['stages'][0]['action']['path']['clearance_height_m'], 0.11)
        self.assertAlmostEqual(captured_actions[-1]['stages'][1]['action']['path']['clearance_height_m'], 0.14)
        self.assertAlmostEqual(captured_actions[-1]['stages'][0]['action']['path']['target_exemption_radius_m'], 0.065)

        # 局部修复 plan_id + skip_failed_stage：同样继承并支持覆盖避障/安全高度字段
        self.h.unified_action_plans['p1']['summary']['failed_stage_id'] = 's1'
        repair_args = dict(
            mode='preview',
            observation_id=obs_id,
            plan_id='p1',
            skip_failed_stage=True,
            clearance_height_m=0.15,
            target_exemption_radius_m=0.045,
        )
        with patch('ni_rua.runtime.actions.inspection._trajectory_image', return_value=(dict(status='unavailable'), [])):
            p3 = typed_dispatch(self.h, repair_args, {'tool_call_id': 'obs-prev-3'}).data
        self.assertEqual(p3['plan_id'], 'p3')
        self.assertEqual([s['id'] for s in captured_actions[-1]['stages']], ['s2'])
        self.assertEqual([b['id'] for b in captured_actions[-1]['review_proxies']['obstacles']], ['o1'])
        self.assertAlmostEqual(captured_actions[-1]['stages'][0]['action']['path']['clearance_height_m'], 0.15)
        self.assertAlmostEqual(captured_actions[-1]['stages'][0]['action']['path']['target_exemption_radius_m'], 0.045)

    def test_broker_blocks_blind_continuation_after_stopped_checkpoint_without_english_substrings(self):
        def preview_side_effect(action):
            full = dict(
                schema_version='candidate-trajectory-v1',
                source_kind='candidate_command_trajectory_not_measurement',
                frame_id='env_origin',
                tcp_config_id='arx-x5-link6-v1',
                joint_names={'left': [], 'right': []},
                samples=[dict(native_step=0, arms={arm: dict(q_rad=[0.] * 6, jaw_goal_normalized=1., phase='initial')
                                                   for arm in ('left', 'right')})],
                geometry=dict(status='unknown'),
                inter_arm_proximity=dict(status='unknown'),
                execution_binding='exact_cached_rows; no replanning or resampling',
            )
            return dict(
                status='planned',
                plan_id='native-plan-cp',
                planned_native_steps=2,
                trajectory_data=full,
                trajectory_evidence=dict(available=True),
                stage_costs={'step': dict(nominal_native_steps=2)},
            )

        self.robot.preview_batch = Mock(side_effect=preview_side_effect)
        exec_count = {'n': 0}

        def exec_side_effect(action, progress_callback=None, cancel_flag=None):
            exec_count['n'] += 1
            if exec_count['n'] == 1:
                # Stopped checkpoint with zero legacy English substring ("Empty grasp detected" absent!)
                return dict(
                    status='stopped',
                    reason='observation_checkpoint',
                    checkpoint_reason='carry_slip_detected',
                    checkpoint_reasons={'step': '夹爪滑脱暂停'},
                    executed_native_steps=1,
                    completed_native_steps=1,
                    batch_status='awaiting_observation',
                    batch_cursor=1,
                    completed_stage_ids=[],
                    pending_stage_ids=['step'],
                    stage_results={'step': dict(status='stopped', completed_steps=1)},
                    terminal_observation=self.robot.observe(),
                )
            return dict(
                status='completed',
                reason=None,
                executed_native_steps=2,
                completed_native_steps=2,
                batch_status='completed',
                batch_cursor=2,
                completed_stage_ids=['step'],
                pending_stage_ids=[],
                stage_results={'step': dict(status='completed', completed_steps=2)},
                terminal_observation=self.robot.observe(),
            )

        self.robot.execute = Mock(side_effect=exec_side_effect)
        runner = self.h.programs = ProgramRunner(self.h.program_services())
        self.addCleanup(runner.close)
        obs_id = self.h.latest['observation_id']
        stage = dict(id='step', kind='move', arm='left', grasp_position_m=[.2, -.1, .8], angles_deg=[0, 0, 0])
        source = (
            "import json\n"
            "from ni_rua_client import NiRuaClient\n"
            "client = NiRuaClient()\n"
            f"obs_id = {obs_id!r}\n"
            f"stage = {stage!r}\n"
            "first = client.action([stage], observation_id=obs_id, mode='direct')\n"
            "next_obs = first['data']['receipt']['result_observation_id']\n"
            "blocked = None\n"
            "try:\n"
            "    client.action([stage], observation_id=next_obs, mode='direct')\n"
            "except RuntimeError as exc:\n"
            "    blocked = {'code': getattr(exc, 'code', None), 'cp': (getattr(exc, 'incomplete_action', None) or {}).get('checkpoint_reason')}\n"
            "second = client.action([stage], observation_id=next_obs, mode='direct', allow_after_failure=True)\n"
            "print(json.dumps({'blocked': blocked, 'second_status': second['data']['receipt']['status']}))\n"
        )
        with patch('ni_rua.runtime.actions.inspection._trajectory_image', return_value=(dict(status='unavailable'), [])):
            job = runner.run(
                dict(source=source, allow_direct_actions=True),
                dict(thread_id='thread', turn_id='turn', tool_call_id='prog-cp-gate'),
            )
        self.assertEqual(job['status'], 'succeeded', job.get('stderr_text'))
        payload = json.loads(job['stdout_text'].strip())
        self.assertEqual(payload['blocked'], {'code': 'program_blind_continuation_blocked', 'cp': 'carry_slip_detected'})
        self.assertEqual(payload['second_status'], 'completed')
        self.assertEqual(self.robot.execute.call_count, 2)

    def test_program_run_schema_on_incomplete_action_and_client_acknowledge_failure_on_custom_physical_plugin(self):
        from ni_rua.core.tool_output import ToolOutput, ToolResponse
        from ni_rua.runtime.tools.registry import ToolSpec, validator_for
        from ni_rua.runtime.tools.definitions import public_definitions as public_tools

        prog_tool = next(t for t in public_tools() if t['name'] == 'program_run')
        validator = validator_for(prog_tool['inputSchema'])
        self.assertEqual(
            list(validator.iter_errors({'source': 'print(1)', 'on_incomplete_action': 'allow'})),
            [],
        )
        self.assertTrue(
            list(validator.iter_errors({'source': 'print(1)', 'on_incomplete_action': 'invalid_mode'}))
        )

        # Register a custom physical plugin tool to verify broker physical_action clears client._failure_acknowledged
        call_n = {'n': 0}

        def custom_push_handler(ctx, args, correlation):
            call_n['n'] += 1
            st = 'stopped' if call_n['n'] <= 2 else 'completed'
            rcpt = dict(
                status=st,
                reason='contact_stop' if st == 'stopped' else None,
                physics_advanced=True,
                executed_native_steps=1,
            )
            return ToolResponse(
                ToolOutput(dict(receipt=rcpt)).content_parts(),
                data=dict(action_id=f'custom-{call_n["n"]}', receipt=rcpt),
                kind='action',
            )

        self.h.tool_registry.register(
            ToolSpec(
                definition={
                    'type': 'function',
                    'name': 'custom_push',
                    'description': 'Custom physical push plugin for testing.',
                    'inputSchema': {'type': 'object', 'properties': {'force_n': {'type': 'number'}}, 'additionalProperties': False},
                },
                handler=custom_push_handler,
                roles=frozenset({'actor', 'program'}),
                effect='physical',
                output_schemas={'action': {'type': 'object'}},
            )
        )
        runner = self.h.programs = ProgramRunner(self.h.program_services())
        self.addCleanup(runner.close)

        source = (
            "import json\n"
            "from ni_rua_client import NiRuaClient\n"
            "client = NiRuaClient()\n"
            "r1 = client.call('custom_push', {'force_n': 1.0})\n"
            "client.acknowledge_failure()\n"
            "# Read-only call must NOT consume _failure_acknowledged\n"
            "obs_ro = client.call('robot_observe', {'observation_id': " + repr(self.h.latest['observation_id']) + "})\n"
            "# Custom physical plugin MUST consume _failure_acknowledged (succeeding on call #2, which also stops incomplete)\n"
            "r2 = client.call('custom_push', {'force_n': 2.0})\n"
            "# Call #3 without re-acknowledging MUST now be blocked by broker!\n"
            "blocked_third = None\n"
            "try:\n"
            "    client.call('custom_push', {'force_n': 3.0})\n"
            "except RuntimeError as exc:\n"
            "    blocked_third = getattr(exc, 'code', None)\n"
            "print(json.dumps({'r2_kind': r2.get('kind'), 'blocked_third': blocked_third}))\n"
        )
        job = runner.run(
            dict(source=source, allow_direct_actions=True, on_incomplete_action='block'),
            dict(thread_id='thread', turn_id='turn', tool_call_id='prog-custom-phys'),
        )
        self.assertEqual(job['status'], 'succeeded', job.get('stderr_text'))
        out = json.loads(job['stdout_text'].strip())
        self.assertEqual(out, {'r2_kind': 'action', 'blocked_third': 'program_blind_continuation_blocked'})
        self.assertEqual(call_n['n'], 2)



if __name__ == "__main__":
    unittest.main()
