"""
- test_harness.py: CPU synthetic fixture: mamba run -n ni-rua python -m unittest discover -s tests -p test_harness.py -v.
- test_lean_payload.py: Conservative actor projections and their immutable evidence boundary.
- test_stage_label_boundary.py: Stage labels carry action identity, never permission to publish hidden state.
- test_typed_actor_output.py: Typed actor display and explicit compatibility serialization boundaries.
"""
from copy import deepcopy
import asyncio
import hashlib
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
import zlib
from ni_rua.runtime.harness import Harness
from ni_rua.apps.gateway.workbench import Workbench
from ni_rua.robot.backends.robodojo.robodojo import FRAME_ID, TCP_CONFIG_ID
from ni_rua.core.storage.store import Store
import base64
import json
from ni_rua.core.presentation.actor_payload import lean_action_result, lean_observation
from runtime.harness.test_handoff_and_candidates import WaitFixtureRobot, text_values
import threading
import time
from unittest.mock import patch
from ni_rua.core.config import Config
from ni_rua.core.contracts import canonical
from ni_rua.core.observation_policy import (ACTOR_PROFILE, LEGACY_ACTOR_PROFILE,
    DEPTH_ACTOR_PROFILE, reject_privileged)
from ni_rua.cortex.context.operation_verification import execution_inputs
from ni_rua.runtime.programs.runner import ProgramRunner
from support.fixture_robot import FixtureRobot
from support.reply_defaults import action_result
from support.motion_fixtures import periodic_pose
from support.motion_fixtures import preview_adapter
import runtime.harness.test_harness_lifecycle_and_shutdown as runtime_lifecycle_fixture
from ni_rua.core.presentation.actor_output import ActorOutput, project_metadata
from ni_rua.core.presentation.actor_presentation import ActorPresentation
from ni_rua.runtime.programs.program_review import action_evidence, reviewed_plan
from ni_rua.core.tool_output import (
    StructuredText,
    ToolOutput,
    ToolResponse,
    bounded_text,
    output_contract_error,
    resolve_detail,
    select_fields,
    validate_page,
    wire_response,
)
from ni_rua.runtime.tools.definitions import TOOLS

NI_RUA_TEST_TIER = "component"


# --- Folded from tests/test_harness.py ---

class _HarnessProtocolRobot:
    """Deterministic protocol fixture; no physics, IK, task state, or GPU."""

    def __init__(self, root, outcome="completed"):
        self.executions = 0
        self.outcome = outcome
        self.path = Path(root) / "synthetic-rgb.png"
        self.path.parent.mkdir(parents=True, exist_ok=True)

        def chunk(kind, data):
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

        self.path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
                             + chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00")) + chunk(b"IEND", b""))

    def start(self):
        return dict(data_origin="synthetic_fixture", frame_id=FRAME_ID, tcp_config_id=TCP_CONFIG_ID,
                    path_kinds=["backend_free"], max_waypoints=1, dual_arm=False, gripper_after_motion=True)

    def observe(self):
        return dict(data_origin="synthetic_fixture", world_kind="observed", sim_step=self.executions,
                    step_domain="adapter_control_step", sim_time_s=None, captured_at="2026-09-30T00:00:00Z",
                    cameras=[dict(camera_id="fixture_camera", png_path=str(self.path), width=1, height=1)],
                    robot_state=dict(data_origin="synthetic_fixture", source_kind="measurement",
                                     arms=dict(left=dict(q_rad=[0.0], ee_pose=dict(position_m=[0.009, 0, 1],
                                               quaternion_xyzw=[0, 0, 0, 1], source_kind="measurement")))))

    def execute(self, action, progress_callback=None, cancel_flag=None):
        self.executions += 1
        if progress_callback:
            progress_callback(dict(phase="started", sim_step=self.executions, data_origin="synthetic_fixture"))
        return dict(status=self.outcome, reason="synthetic_fixture", data_origin="synthetic_fixture",
                    terminal_observation=self.observe(), planned_trace=dict(data_origin="synthetic_fixture", targets=action["path"]),
                    dispatched_trace=dict(data_origin="synthetic_fixture", steps=1),
                    actual_trace=[dict(data_origin="synthetic_fixture", measured_fixture_state=self.observe()["robot_state"])])


class HarnessCheck(unittest.TestCase):
    def test_execution_contract(self):
        temporary = self.enterContext(tempfile.TemporaryDirectory())
        identity = dict(run_id="synthetic-harness", episode_id="fixture-episode", env_instance_id="fixture-env", env_generation=1)
        store = Store(Path(temporary) / "store", identity)
        self.addCleanup(store.close)
        robot = _HarnessProtocolRobot(temporary)
        harness = Harness(store, robot, max_actions=2)
        initial = harness.start()
        self.assertEqual(initial["data_origin"], "synthetic_fixture")
        instructed = harness.publish_observation(dict(robot.observe(), instruction="Pick up the visible target by 10 cm."))
        self.assertEqual(store.read("observation", instructed["observation_id"])["task_instruction"], "Pick up the visible target by 10 cm.")

        def candidate(identifier, **updates):
            action = dict(schema_version="0.1", **identity, world_kind="observed", action_id=identifier,
                          based_on_observation_id=harness.latest["observation_id"], frame_id=FRAME_ID,
                          tcp_config_id=TCP_CONFIG_ID, mode="absolute", limits_profile="simulation-slow-v1", timeout_s=10,
                          path=dict(kind="backend_free", orientation="backend_free", blend_radius_m=0,
                                    position_tolerance_m=0.005, waypoints=[dict(left=dict(position_m=[0.01, 0, 1], quaternion_xyzw=[0, 0, 0, 1]))]),
                          gripper_events=[])
            action.update(updates)
            return action

        action = candidate("fixture-action-1")
        first = harness.execute(action)
        self.assertEqual(first["receipt"]["status"], "completed")
        self.assertEqual(harness.execute(deepcopy(action)), first)
        self.assertEqual(robot.executions, 1)
        with self.assertRaisesRegex(ValueError, "action_id_content_conflict"):
            harness.execute(dict(action, timeout_s=11))
        terminal = store.read("observation", first["receipt"]["result_observation_id"])
        self.assertEqual(terminal["last_action_id"], action["action_id"])
        self.assertEqual(terminal["data_origin"], "synthetic_fixture")
        info, image = store.artifact(terminal["cameras"][0]["rgb_ref"])
        self.assertTrue(terminal["cameras"][0]["rgb_ref"].startswith("artifact:"))
        self.assertEqual(info["mime_type"], "image/jpeg")
        self.assertEqual(info["sha256"], hashlib.sha256(image.read_bytes()).hexdigest())
        delivered = harness.observation_content(terminal, dict(tool_call_id="fixture-call"))
        self.assertTrue(delivered[1]["imageUrl"].startswith("data:image/webp;base64,"))
        for identifier, updates, reason in [
            ("stale", dict(env_generation=2), "stale_environment_identity"),
            ("internal", dict(world_kind="internal"), "internal_action_cannot_execute_observed_world"),
            ("unsupported", dict(path=dict(kind="linear", waypoints=action["path"]["waypoints"])), "unsupported_capability"),
        ]:
            rejected = harness.execute(candidate(identifier, **updates))["receipt"]
            self.assertEqual(rejected["status"], "rejected")
            self.assertIn(reason, rejected["reason"])
        harness.request_stop()
        rejected = harness.execute(candidate("after-stop"))["receipt"]
        self.assertEqual(rejected["status"], "rejected")
        self.assertIn("run_stopping", rejected["reason"])
        self.assertEqual(robot.executions, 1)

        # A normally returned unknown acknowledgement must close admission too.
        unknown_store = Store(Path(temporary) / "unknown-store", identity)
        self.addCleanup(unknown_store.close)
        unknown_robot = _HarnessProtocolRobot(Path(temporary) / "unknown-robot", outcome="unknown")
        unknown = Harness(unknown_store, unknown_robot)
        unknown.start()
        unresolved = candidate("unknown")
        unresolved["based_on_observation_id"] = unknown.latest["observation_id"]
        self.assertEqual(unknown.execute(unresolved)["receipt"]["status"], "unknown")
        self.assertFalse(unknown.accepting)
        followup = candidate("after-unknown")
        followup["based_on_observation_id"] = unknown.latest["observation_id"]
        self.assertEqual(unknown.execute(followup)["receipt"]["status"], "rejected")
        self.assertEqual(unknown_robot.executions, 1)
        # Supervision fixture: no Codex process. Clearing active_action is not
        # a stop confirmation when the last worker acknowledgement is unknown.
        supervisor = Workbench.__new__(Workbench)
        supervisor.harness, supervisor.store = unknown, unknown_store
        supervisor.session = SimpleNamespace(state="completed", interrupt=lambda: None)
        supervisor.control_responses = {}
        supervisor.status = "running"
        supervisor.starting = False
        response = asyncio.run(supervisor.stop("fixture-stop"))
        self.assertEqual(response["status"], "unknown")
        self.assertEqual(asyncio.run(supervisor.stop("fixture-stop")), response)
        supervisor.replay = False
        supervisor.robot = unknown_robot
        for operation in (supervisor.observer_evaluate, supervisor.observer_video):
            with self.assertRaisesRegex(ValueError, "observer_operation_requires_idle_run"):
                operation()
        self.assertFalse(any(event["type"] == "observer.evaluated" for event in unknown_store.events()))


class RecoveryBudgetCheck(unittest.TestCase):
    def test_accepted_without_receipt_survives_fresh_generation_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            identity=dict(run_id='recovery',episode_id='old',env_instance_id='env',env_generation=1)
            store=Store(Path(temporary)/'store',identity)
            try:
                old=dict(schema_version='0.1',**identity,world_kind='observed',action_id='lost')
                store.save('action_request','lost',old)
                store.event('action.accepted',{},dict(action_id='lost'))
                identity.update(episode_id='fresh',env_generation=2)
                robot=_HarnessProtocolRobot(temporary)
                harness=Harness(store,robot,max_actions=1)
                harness.accepting=True  # Explicitly loaded fresh physical generation.
                harness.start()
                self.assertEqual(harness.action_count,1)
                harness.capabilities.update(bounded_wait=dict(max_native_steps=50),path_kinds=['backend_free','hold'])
                result=harness.dispatch('robot_wait',dict(observation_id=harness.latest['observation_id'],native_steps=5),dict(tool_call_id='new'))
                receipt=__import__('json').loads(result['contentItems'][0]['text'])['receipt']
                self.assertEqual(receipt['status'],'rejected')
                self.assertIn('action_budget_exhausted',receipt['reason'])
                self.assertEqual(robot.executions,0)
                supervisor=Workbench.__new__(Workbench)
                supervisor.identity,supervisor.harness,supervisor.store=identity,harness,store
                self.assertFalse(supervisor.current_unknown_actions())
                self.assertEqual(harness.states['lost']['status'],'unknown')
            finally:store.close()

class ResetSafetyCheck(unittest.TestCase):
    def test_preflight_and_stop_during_reset(self):
        import threading
        with tempfile.TemporaryDirectory() as temp:
            identity=dict(run_id='reset-check',episode_id='old',env_instance_id='env',env_generation=1)
            store=Store(Path(temp)/'store',identity)
            robot=_HarnessProtocolRobot(temp)
            entered,release=threading.Event(),threading.Event()
            try:
                harness=Harness(store,robot);harness.start();harness.capabilities['reset']=True
                def validate(seed,layout):
                    if layout==999:raise ValueError('reset_layout_unavailable')
                robot.validate_reset=validate
                with self.assertRaisesRegex(ValueError,'reset_layout_unavailable'):harness.reset(layout_id=999)
                self.assertTrue(harness.accepting);self.assertEqual(identity['env_generation'],1)
                def reset(seed=0,layout_id=0):
                    entered.set();release.wait(3);return {'observation':robot.observe()}
                robot.reset=reset
                results=[]
                thread=threading.Thread(target=lambda:results.append(harness.reset()))
                thread.start();self.assertTrue(entered.wait(2));harness.request_stop();release.set();thread.join(3)
                self.assertFalse(thread.is_alive());self.assertFalse(harness.accepting);self.assertTrue(harness.cancel.is_set())
                self.assertEqual(results[0]['env_generation'],2)
            finally:release.set();store.close()



class ClosedLoopRuntimeReconciliationCheck(unittest.TestCase):
    def _make_action(self, identity, obs_id, action_id, timeout_s=10):
        return dict(
            schema_version="0.1",
            **identity,
            world_kind="observed",
            action_id=action_id,
            based_on_observation_id=obs_id,
            frame_id=FRAME_ID,
            tcp_config_id=TCP_CONFIG_ID,
            mode="absolute",
            limits_profile="simulation-slow-v1",
            timeout_s=timeout_s,
            path=dict(
                kind="backend_free",
                orientation="backend_free",
                blend_radius_m=0,
                position_tolerance_m=0.005,
                waypoints=[dict(left=dict(position_m=[0.01, 0, 1], quaternion_xyzw=[0, 0, 0, 1]))],
            ),
            gripper_events=[],
        )

    def test_worker_process_orphaned_late_reply_and_harness_reconciliation(self):
        import json
        import os
        import sys
        import threading
        from ni_rua.robot.ipc import WorkerProcess

        temporary = self.enterContext(tempfile.TemporaryDirectory())
        root = Path(temporary)

        # 1) Real WorkerProcess subprocess: timeout moves request to orphaned, bounded at 128, and delivers late reply.
        gate_file = root / "release-late.flag"
        late_received = threading.Event()
        captured = []

        def on_late(orphan, message):
            captured.append((orphan, message))
            late_received.set()

        proc = WorkerProcess(root / "worker-ipc", on_late_reply=on_late)
        self.addCleanup(proc.close)
        worker_script = (
            "import json, pathlib, sys, time\n"
            "gate = pathlib.Path(sys.argv[1])\n"
            "delayed = None\n"
            "for line in sys.stdin:\n"
            "    req = json.loads(line)\n"
            "    op = req['op']\n"
            "    if op == 'ready':\n"
            "        print('NI_RUA_JSON ' + json.dumps({'id': req['id'], 'result': {'ok': True}}), flush=True)\n"
            "    elif op == 'cancel':\n"
            "        print('NI_RUA_JSON ' + json.dumps({'id': req['id'], 'result': {'confirmed': False}}), flush=True)\n"
            "    elif op == 'execute':\n"
            "        delayed = req\n"
            "    elif op == 'flush_late':\n"
            "        while not gate.exists():\n"
            "            time.sleep(0.005)\n"
            "        print('NI_RUA_JSON ' + json.dumps({'id': delayed['id'], 'progress': {'phase': 'settling'}}), flush=True)\n"
            "        print('NI_RUA_JSON ' + json.dumps({'id': delayed['id'], 'result': {'status': 'stopped', 'stop_confirmation': {'confirmed': True}}}), flush=True)\n"
            "        print('NI_RUA_JSON ' + json.dumps({'id': req['id'], 'result': {'flushed': True}}), flush=True)\n"
            "    elif op == 'close':\n"
            "        print('NI_RUA_JSON ' + json.dumps({'id': req['id'], 'result': {'closed': True}}), flush=True)\n"
            "        break\n"
        )
        proc.start_process([sys.executable, "-u", "-c", worker_script, str(gate_file)], cwd=root, environment=dict(os.environ))
        with self.assertRaises(TimeoutError):
            proc._request("execute", {"action_id": "act-ipc-1"}, timeout=0.05)
        self.assertEqual(len(proc.orphaned), 1)

        # Verify _MAX_ORPHANED = 128 eviction bound
        with proc.lock:
            for idx in range(140):
                req_key = f"synthetic-orphan-{idx}"
                while len(proc.orphaned) >= proc._MAX_ORPHANED:
                    proc.orphaned.pop(next(iter(proc.orphaned)))
                proc.orphaned[req_key] = {"request_id": req_key, "method": "execute", "params": {}, "timed_out_at_ns": idx}
        self.assertLessEqual(len(proc.orphaned), 128)
        with proc.lock:
            proc.orphaned.clear()

        # Re-issue real timed-out request and release late reply (including a late progress frame before terminal result)
        with self.assertRaises(TimeoutError):
            proc._request("execute", {"action_id": "act-ipc-2"}, timeout=0.05)
        self.assertEqual(len(proc.orphaned), 1)
        gate_file.write_text("go")
        self.assertEqual(proc._request("flush_late", {}, timeout=5.0), {"flushed": True})
        self.assertTrue(late_received.wait(2.0))
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0][0]["method"], "execute")
        self.assertEqual(captured[0][0]["params"], {"action_id": "act-ipc-2"})
        self.assertEqual(captured[0][1]["result"]["status"], "stopped")
        proc.close()
        self.assertEqual(proc.orphaned, {})

        # 2) Harness reconciliation: race during execute except block, archival invariant, and negative gates.
        identity = dict(run_id="late-run", episode_id="ep-1", env_instance_id="env-1", env_generation=1)
        store = Store(root / "store", identity)
        self.addCleanup(store.close)
        robot = FixtureRobot(root / "robot")
        harness = Harness(store, robot)
        self.assertIsNotNone(getattr(robot, "on_late_reply", None))
        harness.start()

        # Case A: Late reply arrives WHILE execute() is raising TimeoutError (in-flight race buffered & drained!)
        def race_timeout_execute(action, progress_callback=None, cancel_flag=None):
            robot.executions += 1
            obs = robot.observe()
            # Fire late reply BEFORE execute() exits into except Exception:!
            robot.on_late_reply(
                {"request_id": "r-race", "method": "execute", "params": action, "timed_out_at_ns": 1},
                {
                    "id": "r-race",
                    "ok": True,
                    "result": {
                        "status": "stopped",
                        "stop_confirmation": {"confirmed": True},
                        "terminal_observation": obs,
                    },
                },
            )
            raise TimeoutError("robot status unknown: worker response deadline; do not replay action")

        robot.execute = race_timeout_execute
        act1 = self._make_action(identity, harness.latest["observation_id"], "act-late-race")
        res1 = harness.execute(act1)
        # Archival audit invariant: stored & in-memory status MUST stay "unknown"!
        self.assertEqual(res1["receipt"]["status"], "unknown")
        self.assertEqual(store.read("action_receipt", "act-late-race")["status"], "unknown")
        self.assertEqual(harness.states["act-late-race"]["status"], "unknown")
        late_ack = harness.states["act-late-race"]["late_acknowledgement"]
        self.assertTrue(late_ack["reopened_admission"])
        self.assertEqual(late_ack["late_status"], "stopped")
        self.assertEqual(late_ack["stop_confirmation"], {"confirmed": True})
        self.assertIsNotNone(late_ack["terminal_observation_id"])
        self.assertEqual(harness.latest["observation_id"], late_ack["terminal_observation_id"])
        self.assertTrue(harness.accepting)
        self.assertFalse(harness.unresolved_execution())
        late_events = [e for e in store.events() if e["type"] == "action.late_acknowledgement"]
        self.assertEqual(len(late_events), 1)
        self.assertTrue(late_events[0]["payload"]["reopened_admission"])

        # Case B: Unconfirmed stop / missing observation / error reply keep accepting=False
        def plain_timeout(action, progress_callback=None, cancel_flag=None):
            robot.executions += 1
            raise TimeoutError("worker_timeout")

        robot.execute = plain_timeout
        act2 = self._make_action(identity, harness.latest["observation_id"], "act-unconfirmed")
        harness.execute(act2)
        self.assertFalse(harness.accepting)
        self.assertTrue(harness.unresolved_execution())

        # B1: confirmed=False -> stays closed
        robot.on_late_reply(
            {"request_id": "r-bad-1", "method": "execute", "params": act2, "timed_out_at_ns": 2},
            {"id": "r-bad-1", "ok": True, "result": {"status": "stopped", "stop_confirmation": {"confirmed": False}, "terminal_observation": robot.observe()}},
        )
        self.assertFalse(harness.accepting)
        self.assertTrue(harness.unresolved_execution())

        # B2: missing terminal_observation -> stays closed
        robot.on_late_reply(
            {"request_id": "r-bad-2", "method": "execute", "params": act2, "timed_out_at_ns": 3},
            {"id": "r-bad-2", "ok": True, "result": {"status": "stopped", "stop_confirmation": {"confirmed": True}}},
        )
        self.assertFalse(harness.accepting)

        # B3: error late reply -> stays closed
        robot.on_late_reply(
            {"request_id": "r-bad-3", "method": "execute", "params": act2, "timed_out_at_ns": 4},
            {"id": "r-bad-3", "ok": False, "error": "RuntimeError: bus fault"},
        )
        self.assertFalse(harness.accepting)

        # B4: Operator request_stop() bumped stop_revision after action started -> valid late reply reconciles action stop,
        #     but MUST NOT reopen accepting over an explicit operator stop!
        harness.request_stop()
        robot.on_late_reply(
            {"request_id": "r-stopped-by-op", "method": "execute", "params": act2, "timed_out_at_ns": 5},
            {"id": "r-stopped-by-op", "ok": True, "result": {"status": "stopped", "stop_confirmation": {"confirmed": True}, "terminal_observation": robot.observe()}},
        )
        self.assertFalse(harness.accepting)
        self.assertFalse(harness.unresolved_execution())
        self.assertFalse(harness.states["act-unconfirmed"]["late_acknowledgement"]["reopened_admission"])
        self.assertEqual(store.read("action_receipt", "act-unconfirmed")["status"], "unknown")

    def test_external_motion_and_scene_invalidation_and_dead_safety_hooks_removed(self):
        import threading
        from ni_rua.core.contracts import digest

        temporary = self.enterContext(tempfile.TemporaryDirectory())
        root = Path(temporary)
        identity = dict(run_id="ext-motion", episode_id="ep-1", env_instance_id="env-1", env_generation=1)
        store = Store(root / "store", identity)
        self.addCleanup(store.close)
        robot = FixtureRobot(root / "robot")
        harness = Harness(store, robot)

        # Deliverable 5 checks: spatial_stream stays None; dead hooks & receipt field removed.
        self.assertIsNone(harness.spatial_stream)
        self.assertFalse(hasattr(harness, "request_safety_stop"))
        self.assertFalse(hasattr(harness, "_pending_safety_stop"))

        obs0 = harness.start()
        act_ok = self._make_action(identity, obs0["observation_id"], "act-clean")
        res_ok = harness.execute(act_ok)
        self.assertNotIn("live_field_safety_stop", res_ok["receipt"])

        # Commit a tracked entity into SceneStore so invalidate_scene exercises apply_patch
        cur_obs_id = harness.latest["observation_id"]
        harness.scene.apply_patch(
            dict(
                schema_version="0.1",
                **identity,
                world_kind="internal",
                base_scene_version=0,
                frame_id=FRAME_ID,
                observation_ids=[cur_obs_id],
                builder_digest=digest("test-builder"),
                operations=[
                    dict(
                        op="upsert_geometry",
                        entity_id="cube_1",
                        source_kind="observation_inference",
                        geometry=dict(kind="box", min_m=[0.1, -0.1, 0.76], max_m=[0.15, -0.05, 0.81]),
                    )
                ],
            )
        )
        self.assertEqual(harness.scene.version(identity, 1)["entities"]["cube_1"]["state"], "tracked")

        # Populate ephemerals and clear/mismatch harness.latest
        harness._persistent_obstacles["o1"] = {"id": "o1"}
        harness._obs3d_memo = ("obs", {})
        harness._obs3d_history.append("h1") if isinstance(harness._obs3d_history, list) else harness._obs3d_history.update(k="v")
        harness._virtual_spatial_field = {"field": 1}
        harness._wrist_progress_cache = {"frames": [1]}
        harness._held_perception = {"left": {"id": "o1"}}
        harness._pose_skill_grasps["g1"] = {"pose": 1}
        harness.failed_paths.add("bad-path")
        clear_called = []
        harness.home_check = SimpleNamespace(
            clear=lambda: clear_called.append(True),
            invalidate_current=lambda reason: None,
            update_current=lambda bundle: None,
        )
        harness.latest = {"observation_id": "unimported-obs", "calibration_id": "new-calib-id"}

        # CRITICAL CONCURRENCY RULE: external_motion("begin") MUST NOT block on robot_lock!
        held_lock = threading.Event()
        release_lock = threading.Event()

        def hold_robot_lock():
            with harness.robot_lock:
                held_lock.set()
                release_lock.wait(2.0)

        locker = threading.Thread(target=hold_robot_lock, daemon=True)
        locker.start()
        self.assertTrue(held_lock.wait(1.0))
        try:
            rev_before = harness.batch_plan_revision
            harness.external_motion("begin", "torque_off_manual_motion")
            self.assertFalse(harness.accepting)
            self.assertEqual(harness.batch_plan_revision, rev_before + 1)
            self.assertIsNone(harness.latest)
            self.assertEqual(harness._persistent_obstacles, {})
            self.assertIsNone(harness._obs3d_memo)
            self.assertFalse(harness._obs3d_history)
            self.assertIsNone(harness._virtual_spatial_field)
            self.assertFalse(hasattr(harness, "_wrist_progress_cache"))
            self.assertFalse(hasattr(harness, "_held_perception"))
            self.assertEqual(harness._pose_skill_grasps, {})
            self.assertEqual(harness.failed_paths, set())
            self.assertEqual(clear_called, [True])
            head_v = harness.scene.head(identity)["scene_version"]
            self.assertEqual(harness.scene.version(identity, head_v)["entities"]["cube_1"]["state"], "dirty")
        finally:
            release_lock.set()
            locker.join(2.0)

        # external_motion("end") refreshes observation when robot has observe, and degrades cleanly on stub robot without observe
        end_res = harness.external_motion("end", "operator_go_home", result={"status": "completed"}, refresh_observation=True)
        self.assertIsNotNone(end_res["observation_id"])
        self.assertEqual(harness.latest["observation_id"], end_res["observation_id"])
        ext_events = [e for e in store.events() if e["type"] == "harness.external_motion"]
        self.assertEqual([e["payload"]["phase"] for e in ext_events], ["begin", "end"])
        self.assertEqual(ext_events[1]["payload"]["observation_id"], end_res["observation_id"])

        stub_store = Store(root / "stub-store", identity)
        self.addCleanup(stub_store.close)
        stub_harness = Harness(stub_store, SimpleNamespace())
        stub_end = stub_harness.external_motion("end", "operator_calibrate", result={"status": "calibrated"}, refresh_observation=True)
        self.assertIsNone(stub_end["observation_id"])
        self.assertIsNone(stub_harness.latest)

    def test_invalidate_scene_chunks_over_64_entities_and_late_reply_respects_operator_supersession_and_reset(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        identity = dict(run_id="run-w3", episode_id="ep-1", env_instance_id="sim-1", env_generation=1)
        store = Store(root / "store", identity)
        self.addCleanup(store.close)

        robot = FixtureRobot(root / "fixture")
        orig_start = robot.start
        robot.start = lambda: dict(orig_start(), reset=True)
        robot.reset = lambda seed=0, layout_id=0: dict(status="reset", observation=robot.observe())
        harness = Harness(store, robot)
        obs0 = harness.start()
        make_obs_raw = robot.observe

        # 1) >64 tracked entities in SceneStore must chunk cleanly in invalidate_scene without ValueError
        for batch_idx in range(2):
            head_v = harness.scene.head(harness.identity)["scene_version"] or 0
            ops = [
                dict(
                    op="upsert_geometry",
                    entity_id=f"obj_{batch_idx}_{i}",
                    source_kind="observation_inference",
                    geometry=dict(kind="box", min_m=[0.0, 0.0, 0.76], max_m=[0.05, 0.05, 0.81]),
                )
                for i in range(40)
            ]
            harness.scene.apply_patch(
                dict(
                    schema_version="0.1",
                    **harness.identity,
                    world_kind="internal",
                    base_scene_version=head_v,
                    frame_id=FRAME_ID,
                    observation_ids=[obs0["observation_id"]],
                    builder_digest="a" * 64,
                    operations=ops,
                )
            )
        head_before = harness.scene.head(harness.identity)["scene_version"]
        self.assertEqual(len(harness.scene.version(harness.identity, head_before)["entities"]), 80)
        harness.invalidate_scene(reason="chunk_test")
        head_after = harness.scene.head(harness.identity)["scene_version"]
        self.assertEqual(head_after, head_before + 2)
        self.assertTrue(
            all(v["state"] == "dirty" for v in harness.scene.version(harness.identity, head_after)["entities"].values())
        )

        # 2) Timed-out action followed by operator relax + resume MUST NOT have latest clobbered or accepting re-closed by late reply
        act1_id = "act-timeout-1"
        store.save("action_request", act1_id, dict(action_id=act1_id, **harness.identity))
        harness._action_stop_revisions[act1_id] = harness.stop_revision
        harness._action_intervention_revisions[act1_id] = harness._operator_intervention_revision
        harness.states[act1_id] = dict(status="unknown", reason="timeout")
        harness.accepting = False

        # Operator acknowledges act1_id and runs reopen_admission (resume) -> publishes fresh current observation
        harness._operator_acknowledged_actions.add(act1_id)
        obs_resumed = harness.reopen_admission({"operator_request_id": "op-resume-1"})
        self.assertTrue(harness.accepting)
        self.assertEqual(harness.latest["observation_id"], obs_resumed["observation_id"])

        # Unconfirmed late reply arrives AFTER operator resume -> must NOT re-close accepting!
        late_unconfirmed = harness._reconcile_late_execute_reply(
            {"method": "execute", "params": {"action_id": act1_id}},
            {"ok": True, "result": {"status": "stopped", "stop_confirmation": {"confirmed": False}}},
        )
        self.assertIsNotNone(late_unconfirmed)
        self.assertFalse(late_unconfirmed["reopened_admission"])
        self.assertTrue(harness.accepting)
        self.assertEqual(harness.latest["observation_id"], obs_resumed["observation_id"])

        # Confirmed late reply with stale terminal_observation arrives AFTER operator relax -> must publish historical only, keeping latest=None!
        act2_id = "act-timeout-2"
        store.save("action_request", act2_id, dict(action_id=act2_id, **harness.identity))
        harness._action_stop_revisions[act2_id] = harness.stop_revision
        harness._action_intervention_revisions[act2_id] = harness._operator_intervention_revision
        harness.states[act2_id] = dict(status="unknown", reason="timeout")
        stale_raw = make_obs_raw()
        harness.external_motion("begin", "torque_off_manual_motion")
        self.assertIsNone(harness.latest)
        self.assertFalse(harness.accepting)
        late_after_relax = harness._reconcile_late_execute_reply(
            {"method": "execute", "params": {"action_id": act2_id}},
            {
                "ok": True,
                "result": {
                    "status": "stopped",
                    "stop_confirmation": {"confirmed": True},
                    "terminal_observation": stale_raw,
                    "episode_terminal": {"ended": True, "native_action_count": 99},
                },
            },
        )
        self.assertIsNotNone(late_after_relax)
        self.assertIsNone(harness.latest)
        self.assertFalse(harness.accepting)
        self.assertFalse(harness.episode_done.is_set())
        obs_events = [
            e for e in store.events()
            if e["type"] == "observation.available" and e["payload"]["observation_id"] == late_after_relax["terminal_observation_id"]
        ]
        self.assertEqual(len(obs_events), 1)
        self.assertTrue(obs_events[0]["payload"]["historical"])

        # 3) Cross-episode reset discards late reply from prior episode without touching new episode state
        old_ep_act = "act-old-ep"
        store.save("action_request", old_ep_act, dict(action_id=old_ep_act, **harness.identity))
        harness.states[old_ep_act] = dict(status="unknown", reason="timeout")
        old_raw = make_obs_raw()
        harness.reset(seed=1, layout_id=0)
        new_ep_latest = harness.latest["observation_id"]
        cross_ep = harness._reconcile_late_execute_reply(
            {"method": "execute", "params": {"action_id": old_ep_act}},
            {
                "ok": True,
                "result": {
                    "status": "stopped",
                    "stop_confirmation": {"confirmed": True},
                    "terminal_observation": old_raw,
                },
            },
        )
        self.assertIsNone(cross_ep)
        self.assertEqual(harness.latest["observation_id"], new_ep_latest)

    def test_worker_process_lock_atomic_pending_completion_prevents_dropped_late_reply(self):
        import io
        import json
        import os
        import sys
        import threading
        from ni_rua.robot.ipc.process import WIRE_PREFIX, WorkerProcess

        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        late_seen = []
        wp = WorkerProcess(
            root,
            on_late_reply=lambda orphan, msg: late_seen.append((orphan["request_id"], msg.get("result"))),
        )

        # Simulate timeout boundary where _read sets entry[0]/entry[1] under lock right before _request's finally runs
        req_id = "req-boundary-1"
        done = threading.Event()
        result_box = {}
        wp.pending[req_id] = (done, result_box, None)
        r_fd, w_fd = os.pipe()
        proc_stub = SimpleNamespace(
            stdout=os.fdopen(r_fd, "r", encoding="utf-8"),
            poll=lambda: 0,
        )
        wp.process = proc_stub
        wp._process_generation = 1
        log_buf = io.StringIO()
        msg = {"id": req_id, "ok": True, "result": {"status": "stopped"}}
        os.write(w_fd, (WIRE_PREFIX + json.dumps(msg) + "\n").encode("utf-8"))
        os.close(w_fd)
        try:
            wp._read(proc_stub, log_buf, 1)
        finally:
            proc_stub.stdout.close()
        # Verify entry was atomically populated and marked set
        self.assertTrue(done.is_set())
        self.assertEqual(result_box["id"], req_id)
        # Now simulate _request finally block when timed_out=True after _read just populated result_box
        with wp.lock:
            popped = wp.pending.pop(req_id, None)
            self.assertIsNotNone(popped)
            self.assertTrue(done.is_set() and "id" in result_box)
        wp.on_late_reply({"request_id": req_id}, dict(result_box))
        self.assertEqual(late_seen, [(req_id, {"status": "stopped"})])


# --- Folded from tests/test_lean_payload.py ---

TRACE_KINDS = ("planned_trace", "dispatched_trace", "actual_trace", "before_state")


class LeanProjectionTests(unittest.TestCase):
    def test_action_preserves_precision_request_errors_and_unknown_diagnostics(self):
        source = dict(
            request=dict(action_id="action", based_on_observation_id="observation", timeout_s=1.123456789,
                         path=dict(waypoints=[dict(position_m=[.12345678912345678, -1e-12, 1e20])])),
            receipt=dict(status="stopped", reason="tracking_guard", completed_native_steps=3,
                         versions=dict(adapter="v1"), step_semantics=dict(native_hz=25.123456789),
                         trace_sampling=dict(stride=2, measured=True),
                         tracking_guard=dict(error_rad=.0000000123456789, limit_rad=.123456789),
                         diagnostics=dict(versions="diagnostic version", actual_trace_ref="artifact:nested",
                                          sha256="diagnostic hash", residual_m=1e-11),
                         unknown_trace_ref="artifact:future"),
            preview=dict(trace_ref="artifact:preview"),
            error=dict(code="partial_execution", value=.9876543212345678),
        )
        source["receipt"].update({kind + "_ref": "artifact:" + kind for kind in TRACE_KINDS})
        original = deepcopy(source)
        expected = deepcopy(source)
        del expected["receipt"]["versions"]
        for kind in TRACE_KINDS:
            del expected["receipt"][kind + "_ref"]
            expected["receipt"][kind + "_available"] = True
        projected = lean_action_result(source)
        self.assertEqual(projected, expected)
        self.assertEqual(source, original)
        projected["request"]["path"]["waypoints"][0]["position_m"].append(42)
        projected["receipt"]["diagnostics"]["versions"] = "changed"
        self.assertEqual(source, original)

    def test_trace_availability_distinguishes_null_absent_and_unknown_values(self):
        for kind in TRACE_KINDS:
            reference, available = kind + "_ref", kind + "_available"
            for value in (None, "artifact:abc", "https://example/trace", "", 0, False,
                          {"artifact": "abc"}, ["artifact:abc"]):
                with self.subTest(kind=kind, value=value):
                    source = dict(receipt={reference: value})
                    expected = {available: value is not None} if value is None or value == "artifact:abc" else {reference: value}
                    self.assertEqual(lean_action_result(source), dict(receipt=expected))
            self.assertEqual(lean_action_result(dict(receipt={})), dict(receipt={}))
            for conflict in (False, True, None, "unknown"):
                source = dict(receipt={reference: "artifact:abc", available: conflict})
                self.assertEqual(lean_action_result(source), source)

    def observation(self):
        camera = dict(camera_id="head", rgb_ref="artifact:rgb", calibration_ref="artifact:calibration",
                      pose_sample_ref="artifact:pose", width=640, height=480,
                      image_encoding=dict(format="webp"), sha256="hash", source_sha256="source",
                      capture_time=1.123456789123, clock_domain="sim",
                      reconstruction=dict(sha256="nested-hash", warning="lossy", quality=90))
        return dict(bundle=dict(schema_version="0.1", run_id="run", episode_id="episode",
                                env_instance_id="env", robot_state_ref="artifact:state",
                                robot_model_ref="artifact:model", cameras=[camera],
                                image_reconstruction=dict(source_sha256="retain", error=.123456789)),
                    robot_state=dict(q_rad=[.123456789123, -1e-12]),
                    camera_calibration=dict(K=[[1.123456789, 0, 320.123456789]]),
                    camera_pose_sample=dict(T_world_camera=[1e-12, .987654321]),
                    image_view=dict(camera_id="head", rgb_ref="artifact:crop", original_rgb_ref="artifact:rgb",
                                    encoding=dict(format="webp"), image_encoding="jpeg", sha256="crop-hash",
                                    source_sha256="crop-source", source_roi=[10, 20, 100, 200], display_scale=2,
                                    pixel_transform=[[2, 0, -19.5], [0, 2, -39.5], [0, 0, 1]],
                                    reconstruction=dict(warning="reconstructed", sha256="nested", quality=80)),
                    unknown=dict(rgb_ref="artifact:unknown", sha256="unknown-hash"))

    def test_observation_removes_only_known_encoding_and_resolved_references(self):
        source = self.observation()
        original = deepcopy(source)
        expected = deepcopy(source)
        del expected["bundle"]["robot_state_ref"]
        expected["bundle"]["robot_state_available"] = True
        camera = expected["bundle"]["cameras"][0]
        for key in ("image_encoding", "sha256", "source_sha256"):
            del camera[key]
        for kind in ("rgb", "calibration", "pose_sample"):
            del camera[kind + "_ref"]
            camera[kind + "_available"] = True
        view = expected["image_view"]
        for key in ("encoding", "image_encoding", "sha256", "source_sha256"):
            del view[key]
        for kind in ("rgb", "original_rgb"):
            del view[kind + "_ref"]
            view[kind + "_available"] = True
        projected = lean_observation(source)
        self.assertEqual(projected, expected)
        self.assertEqual(source, original)
        projected["robot_state"]["q_rad"].append(42)
        projected["image_view"]["reconstruction"]["warning"] = "changed"
        self.assertEqual(source, original)

    def test_expansion_required_before_state_or_calibration_reference_elision(self):
        for key in ("robot_state", "camera_calibration", "camera_pose_sample"):
            for value in (None, [], "not expanded", 7):
                with self.subTest(key=key, value=value):
                    source = self.observation()
                    source[key] = value
                    projected = lean_observation(source)
                    if key == "robot_state":
                        self.assertEqual(projected["bundle"]["robot_state_ref"], "artifact:state")
                        self.assertNotIn("robot_state_available", projected["bundle"])
                    else:
                        reference = "calibration_ref" if key == "camera_calibration" else "pose_sample_ref"
                        self.assertEqual(projected["bundle"]["cameras"][0][reference], source["bundle"]["cameras"][0][reference])
        source = self.observation()
        for key in ("robot_state", "camera_calibration", "camera_pose_sample"):
            del source[key]
        projected = lean_observation(source)
        self.assertIn("robot_state_ref", projected["bundle"])
        self.assertIn("calibration_ref", projected["bundle"]["cameras"][0])
        self.assertIn("pose_sample_ref", projected["bundle"]["cameras"][0])

    def test_second_camera_distinct_calibration_and_pose_refs_remain_addressable(self):
        source = self.observation()
        same = deepcopy(source["bundle"]["cameras"][0])
        same["camera_id"] = "same-calibration"
        different = dict(same, camera_id="wrist", calibration_ref="artifact:wrist-calibration", pose_sample_ref="artifact:wrist-pose")
        source["bundle"]["cameras"].extend([same, different])
        cameras = lean_observation(source)["bundle"]["cameras"]
        self.assertTrue(cameras[1]["calibration_available"])
        self.assertTrue(cameras[1]["pose_sample_available"])
        self.assertEqual(cameras[2]["calibration_ref"], "artifact:wrist-calibration")
        self.assertEqual(cameras[2]["pose_sample_ref"], "artifact:wrist-pose")
        self.assertNotIn("calibration_available", cameras[2])
        self.assertNotIn("pose_sample_available", cameras[2])

    def test_observation_reference_null_unknown_and_conflict_fallbacks(self):
        locations = (("bundle", "robot_state"), ("camera", "rgb"), ("camera", "calibration"),
                     ("camera", "pose_sample"), ("image_view", "rgb"), ("image_view", "original_rgb"))
        for location, kind in locations:
            source = self.observation()
            target = source["bundle"]["cameras"][0] if location == "camera" else source[location]
            del target[kind + "_ref"]
            projected = lean_observation(source)
            actual = projected["bundle"]["cameras"][0] if location == "camera" else projected[location]
            self.assertNotIn(kind + "_available", actual)
            for value in (None, "artifact:known", "opaque-reference", {"ref": "artifact:dict"}):
                for conflict in (False, True):
                    with self.subTest(location=location, kind=kind, value=value, conflict=conflict):
                        source = self.observation()
                        target = source["bundle"]["cameras"][0] if location == "camera" else source[location]
                        target[kind + "_ref"] = value
                        if conflict:
                            target[kind + "_available"] = "existing diagnostic"
                        projected = lean_observation(source)
                        actual = projected["bundle"]["cameras"][0] if location == "camera" else projected[location]
                        if not conflict and (value is None or value == "artifact:known"):
                            self.assertNotIn(kind + "_ref", actual)
                            self.assertIs(actual[kind + "_available"], value is not None)
                        else:
                            self.assertEqual(actual[kind + "_ref"], value)
                            if conflict:
                                self.assertEqual(actual[kind + "_available"], "existing diagnostic")

    def test_nonstandard_shapes_are_deep_copied_without_recursive_elision(self):
        for source in (None, [dict(versions="keep", rgb_ref="artifact:keep", value=.123456789)],
                       dict(bundle=[{"sha256": "keep"}], receipt="unknown", image_view=[{"encoding": "keep"}])):
            for project in (lean_observation, lean_action_result):
                with self.subTest(source=source, project=project.__name__):
                    self.assertEqual(project(source), source)
                    if isinstance(source, (list, dict)):
                        self.assertIsNot(project(source), source)


class LeanHarnessTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        identity = dict(run_id="lean-fixture", episode_id="ep", env_instance_id="env", env_generation=1)
        self.store = Store(self.root / "store", identity)
        self.addCleanup(self.store.close)
        self.robot = WaitFixtureRobot(self.root)
        self.harness = Harness(self.store, self.robot)
        self.harness.start()

    def test_default_lean_preserves_program_storage_and_reads_traces_by_action_id(self):
        self.assertEqual(self.harness.actor_payload_profile, "lean_v2")
        args = dict(action_id="wait", observation_id=self.harness.latest["observation_id"], native_steps=5)
        model_values = text_values(self.harness.dispatch("robot_wait", args, dict(tool_call_id="model")))
        request = self.store.read("action_request", "wait")
        receipt = self.store.read("action_receipt", "wait")
        evidence_before = deepcopy((request, receipt, self.store.records("observation")))
        self.assertEqual(model_values[0]["request"], request)
        self.assertNotIn("versions", model_values[0]["receipt"])
        self.assertEqual(model_values[0]["receipt"]["step_semantics"], receipt["step_semantics"])
        for kind in TRACE_KINDS:
            self.assertNotIn(kind + "_ref", model_values[0]["receipt"])
            self.assertTrue(model_values[0]["receipt"][kind + "_available"])
            trace = text_values(self.harness.dispatch("history_read_trace", dict(action_id="wait", kind=kind),
                                                     dict(tool_call_id="read-" + kind)))[0]
            expected = json.loads(self.store.resolve_artifact(receipt[kind + "_ref"])[1])
            self.assertEqual(trace, expected)
        model_observation = next(value for value in reversed(model_values) if "bundle" in value)
        self.assertEqual(model_observation["robot_state"]["arms"]["left"]["q_rad"], [.123456789] * 6)
        self.assertTrue(model_observation["bundle"]["robot_state_available"])
        program_values = text_values(self.harness.dispatch("robot_wait", args, dict(tool_call_id="program", program_id="p")))
        self.assertEqual(program_values[0]["request"], request)
        self.assertEqual(program_values[0]["receipt"], receipt)
        program_observation = next(value for value in reversed(program_values) if "bundle" in value)
        self.assertIn("robot_state_ref", program_observation["bundle"])
        self.assertEqual(evidence_before, (self.store.read("action_request", "wait"),
                                         self.store.read("action_receipt", "wait"), self.store.records("observation")))
        self.assertEqual(self.robot.executions, 1)

    def test_lean_delivery_keeps_actual_image_bytes_and_crop_geometry(self):
        bundle_before = deepcopy(self.harness.latest)
        for crop in ({}, dict(camera_id="cam_head", roi=[0, 0, 1, 1], scale=2)):
            with self.subTest(crop=crop):
                lean = self.harness.observation_content(self.harness.latest, dict(tool_call_id="lean"), **crop)
                full = self.harness.observation_content(self.harness.latest, dict(tool_call_id="full", program_id="p"), **crop)
                lean_images = [base64.b64decode(item["imageUrl"].split(",", 1)[1]) for item in lean if item["type"] == "inputImage"]
                full_images = [base64.b64decode(item["imageUrl"].split(",", 1)[1]) for item in full if item["type"] == "inputImage"]
                self.assertTrue(lean_images)
                self.assertEqual(lean_images, full_images)
                metadata = json.loads(lean[0]["text"])
                self.assertTrue(metadata["bundle"]["cameras"][0]["rgb_available"])
                if crop:
                    view = metadata["image_view"]
                    original = json.loads(full[0]["text"])["image_view"]
                    for key in ("camera_id", "source_roi", "display_scale", "pixel_transform", "width", "height"):
                        self.assertEqual(view[key], original[key])
                    self.assertTrue(view["rgb_available"])
                    self.assertTrue(view["original_rgb_available"])
                    self.assertNotIn("encoding", view)
        self.assertEqual(self.harness.latest, bundle_before)
        deliveries = self.store.records("delivery")
        self.assertTrue(all("rgb_ref" in row["images"][0] and "encoding" in row["images"][0] for row in deliveries))



# --- Folded from tests/test_stage_label_boundary.py ---

class StageLabelPolicyTests(unittest.TestCase):
    def receipt(self):
        return dict(status='stopped', reason='observation_checkpoint', executed_native_steps=1,
            stage_results={'depth': dict(kind='observe', status='completed', completed_steps=0)},
            checkpoint_stage_ids=['depth'], checkpoint_reasons={'depth': 'Inspect visible insertion'})

    def test_only_bound_top_level_labels_are_exempt_and_values_remain_checked(self):
        value = self.receipt()
        original = deepcopy(value)
        for profile in (ACTOR_PROFILE, LEGACY_ACTOR_PROFILE, DEPTH_ACTOR_PROFILE):
            with self.subTest(profile=profile):
                reject_privileged(value, profile, stage_ids=['depth'])
                reject_privileged({'stages': value['stage_results']}, profile, stage_ids=['depth'])
                reject_privileged({
                    'stage_costs': {'depth': dict(arm='left', kind='observe', start_native_step=1, end_native_step=1, native_steps=0)},
                    'checkpoints': [dict(cursor=1, stage_ids=['depth'], reasons={'depth': 'Inspect visible insertion'})],
                }, profile, stage_ids=['depth'])
                with self.assertRaisesRegex(ValueError, 'privileged_data'):
                    reject_privileged(value, profile)
                with self.assertRaisesRegex(ValueError, 'does_not_match'):
                    reject_privileged(value, profile, stage_ids=['reach'])
                for hidden in ({'depth': [[1.2]]}, {'evaluator_state': {'score': 100}},
                               {'stage_results': {'depth': {'kind': 'observe'}}}):
                    bad = deepcopy(value)
                    bad['stage_results']['depth'].update(hidden)
                    with self.subTest(hidden=hidden), self.assertRaisesRegex(ValueError, 'privileged_data'):
                        reject_privileged(bad, profile, stage_ids=['depth'])
                with self.assertRaisesRegex(ValueError, 'invalid_action_stage_map_value'):
                    reject_privileged({'stage_results': {'depth': [[1.2]]}}, profile, stage_ids=['depth'])
                with self.assertRaisesRegex(ValueError, 'privileged_data'):
                    reject_privileged({'extra': value}, profile, stage_ids=['depth'])
        self.assertEqual(value, original)


class StageLabelExecutionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store = Store(Path(temporary.name) / 'run', dict(run_id='labels', episode_id='ep',
            env_instance_id='env', env_generation=1))
        self.addCleanup(self.store.close)
        self.adapter, _, self.dispatched = preview_adapter()
        robot = FixtureRobot(temporary.name)
        observation = robot.observe
        robot.observe = lambda: dict(observation(), robot_state=self.adapter.state(),
            sim_step=self.adapter.control_step, camera_calibration=dict(reference_frame_id='env_origin', cameras={}),
            camera_pose_sample=dict(cameras={}), robot_model=self.adapter.robot_model)
        self.adapter.observe = robot.observe
        robot.start = lambda: dict(self.adapter.capabilities(), observer_video=False)
        robot.preview_batch = self.adapter.preview_batch
        robot.execute = lambda action, progress_callback, cancel_flag: self.adapter.execute(action, progress_callback, cancel_flag)
        self.h = Harness(self.store, robot, config=Config.from_dict(dict(features=dict(
            motion_review=False, completion_guidance='none'))))
        self.h.start()
        self.h.thread_id = 'thread'
        self.args = dict(mode='direct', observation_id=self.h.latest['observation_id'], action_id='label-action', timeout_s=10,
            stages=[dict(id='reach', kind='move', arm='left', **periodic_pose([-.025])),
                    dict(id='depth', kind='observe', arm='left', reason='Inspect visible insertion')])

    def check_checkpoint(self):
        receipt = self.store.read('action_receipt', 'label-action')
        self.assertEqual((receipt['status'], receipt['reason']), ('stopped', 'observation_checkpoint'))
        self.assertEqual(receipt['stage_results']['depth']['status'], 'completed')
        self.assertEqual(receipt['executed_native_steps'], 1)
        self.assertEqual(len(self.dispatched), 1)
        self.assertEqual(self.h.native_action_count, 1)
        self.assertTrue(self.h.accepting)
        self.assertIsNotNone(self.h.latest)
        return receipt

    def test_actor_checkpoint_replay_and_archived_artifact(self):
        params = dict(threadId='thread', turnId='turn', callId='depth-check', tool='robot_action', arguments=self.args)
        response = self.h.handle_tool(params)
        self.assertTrue(response['success'])
        presented = action_result(response['contentItems'])
        self.assertEqual(presented['stages']['depth']['status'], 'completed')
        self.assertEqual(presented['receipt']['executed_native_steps'], 1)
        receipt = self.check_checkpoint()
        action = self.store.read('action_request', 'label-action')
        execution = execution_inputs(self.store, self.h.identity, action_ids=['label-action'],
            before_observation_id=action['based_on_observation_id'],
            after_observation_id=receipt['result_observation_id'])
        self.assertEqual(execution['actions'][0]['feedback']['executed_native_steps'], 1)
        self.assertEqual(execution['status'], 'incomplete')
        self.assertEqual(self.h.execute(action)['receipt'], receipt)
        self.assertEqual(len(self.dispatched), 1)
        runner = ProgramRunner(self.h.program_services())
        self.addCleanup(runner.close)
        entry = dict(cancel=threading.Event(), deadline=time.monotonic()+30, call_lock=threading.RLock())
        result = dict(request=action, receipt=receipt, action_id='label-action')
        reference = presented['details_ref']
        self.assertIn('data_base64', runner.dispatch(entry, {'artifact_ref': reference}))
        evidence = dict(action_id='label-action', **receipt)
        reference = self.store.put_artifact(canonical([evidence]).encode(), 'program_action_evidence', 'application/json')
        self.assertIn('data_base64', runner.dispatch(entry, {'artifact_ref': reference}))
        for bad in (dict(result, request=dict(action, stages=[dict(id='evaluator_state')])),
                    dict(result, receipt=dict(receipt, stage_results={'depth': {'depth': [1.2]}}))):
            reference = self.store.put_artifact(canonical(bad).encode(), 'unified_action_result', 'application/json')
            with self.assertRaises(ValueError):
                runner.dispatch(entry, {'artifact_ref': reference})
        reference = self.store.put_artifact(canonical(receipt).encode(), 'computed', 'application/json')
        with self.assertRaisesRegex(ValueError, 'privileged_data'):
            runner.dispatch(entry, {'artifact_ref': reference})

    def test_program_checkpoint_keeps_typed_step_evidence(self):
        runner = self.h.programs = ProgramRunner(self.h.program_services())
        self.addCleanup(runner.close)
        source = ('from ni_rua_client import NiRuaClient\n'
                  'client = NiRuaClient()\n'
                  'reply = client.call("robot_action", ' + repr(self.args) + ')\n'
                  'assert reply["success"], reply\n')
        job = runner.run(dict(source=source, timeout_s=30, allow_direct_actions=True),
            dict(thread_id='thread', turn_id='turn', tool_call_id='depth-program'))
        self.assertEqual(job['status'], 'succeeded', job.get('stderr_text'))
        evidence = job['action_summary']['recent_actions'][0]
        self.assertEqual((evidence['status'], evidence['executed_native_steps']), ('stopped', 1))
        self.assertEqual(evidence['stage_results']['depth']['status'], 'completed')
        self.check_checkpoint()

    def test_metadata_named_checkpoint_keeps_actor_and_archived_stage_evidence(self):
        for label in ('hash', 'sha256'):
            with self.subTest(label=label):
                args = deepcopy(self.args)
                args.update(action_id='label-' + label, observation_id=self.h.latest['observation_id'])
                args['stages'][1]['id'] = label
                response = self.h.handle_tool(dict(threadId='thread', turnId='turn', callId=label,
                                                   tool='robot_action', arguments=args))
                self.assertTrue(response['success'])
                shown = action_result(response['contentItems'])
                receipt = self.store.read('action_receipt', args['action_id'])
                self.assertEqual(shown['stages'][label]['status'], 'completed')
                self.assertEqual(receipt['stage_results'][label]['status'], 'completed')
                self.assertEqual(shown['receipt']['executed_native_steps'], 1)


class UnsafeAcknowledgementTests(unittest.TestCase):
    setUp = runtime_lifecycle_fixture.RuntimeLifecycleTests.setUp
    action = runtime_lifecycle_fixture.RuntimeLifecycleTests.action

    def test_bad_payload_keeps_execution_facts_without_leaking_or_replaying(self):
        execute = self.robot.execute
        for status in ('completed', 'partial', 'stopped', 'unknown', 'not_a_receipt_status'):
            with self.subTest(status=status):
                expected_status = 'unknown' if status == 'not_a_receipt_status' else status
                self.harness.accepting = True
                initial = self.harness.observe()['observation_id']
                action = self.action('unsafe-' + status, initial)
                def unsafe(*args, **kwargs):
                    result = execute(*args, **kwargs)
                    result.update(status=status, executed_native_steps=1, completed_native_steps=1,
                        executed_control_steps=10, stop_confirmation=dict(clock='paused',
                            actuator_target='last_native_command_retained', settling_steps=0),
                        episode_terminal=dict(ended=False, native_action_count=self.robot.executions, native_step_limit=20),
                        extra={'evaluator_state': {'secret': 'DO_NOT_PUBLISH'}})
                    return result
                with patch.object(self.robot, 'execute', side_effect=unsafe):
                    result = self.harness.execute(action)
                receipt = result['receipt']
                self.assertEqual(receipt['status'], expected_status)
                self.assertEqual((receipt['executed_native_steps'], receipt['completed_native_steps']), (1, 1))
                self.assertEqual(receipt['executed_control_steps'], 10)
                self.assertEqual(receipt['stop_confirmation']['clock'], 'paused')
                self.assertFalse(receipt['post_execution_error']['action_replay_permitted'])
                self.assertNotIn('DO_NOT_PUBLISH', canonical(result))
                self.assertTrue(set(receipt).isdisjoint({'extra', 'planned_trace', 'actual_trace', 'before_state'}))
                self.assertIsNone(self.harness.latest)
                before = self.robot.executions
                self.assertEqual(self.harness.execute(action), result)
                self.assertEqual(self.robot.executions, before)
                self.assertEqual(self.harness.native_action_count, before)
                self.assertEqual(self.store.read('action_receipt', action['action_id']), receipt)
                self.harness.observe()
                self.assertEqual(self.harness.accepting, expected_status != 'unknown')

    def test_non_json_payload_and_hidden_stop_fields_cannot_escape_safe_ack(self):
        execute = self.robot.execute
        def unsafe(*args, **kwargs):
            result = execute(*args, **kwargs)
            result.update(executed_native_steps=1, completed_native_steps=True,
                executed_control_steps=-1, stop_confirmation={'evaluator_state': 'DO_NOT_PUBLISH'}, extra=object())
            self.harness.request_stop()
            return result
        with patch.object(self.robot, 'execute', side_effect=unsafe):
            result = self.harness.execute(self.action('bad-values'))
        receipt = result['receipt']
        self.assertEqual(receipt['status'], 'completed')
        self.assertEqual(receipt['executed_native_steps'], 1)
        self.assertNotIn('completed_native_steps', receipt)
        self.assertNotIn('executed_control_steps', receipt)
        self.assertNotIn('stop_confirmation', receipt)
        self.assertNotIn('DO_NOT_PUBLISH', canonical(result))
        self.harness.observe()
        self.assertFalse(self.harness.accepting)

    def test_serialization_failure_keeps_counts_and_rejects_malformed_stop_values(self):
        execute = self.robot.execute
        def unsafe(*args, **kwargs):
            result = execute(*args, **kwargs)
            result.update(executed_native_steps=1, extra=object(),
                stop_confirmation=dict(clock='paused', actuator_target='last_native_command_retained',
                                       settling_steps=True))
            return result
        with patch.object(self.robot, 'execute', side_effect=unsafe):
            result = self.harness.execute(self.action('not-json'))
        receipt = result['receipt']
        self.assertEqual(receipt['status'], 'completed')
        self.assertEqual(receipt['executed_native_steps'], 1)
        self.assertNotIn('stop_confirmation', receipt)
        self.assertEqual(receipt['post_execution_error']['error_type'], 'TypeError')
        self.assertEqual(receipt['post_execution_error']['message'], 'display_not_json')
        self.assertIsNone(self.harness.latest)
        self.assertEqual(self.harness.execute(self.action('not-json')), result)
        self.assertEqual(self.robot.executions, 1)

    def test_negative_execution_acknowledgements_survive_fallback_and_fresh_observe(self):
        execute = self.robot.execute
        signals = [dict(execution_uncertain=True), dict(stop_confirmed=False),
            dict(stop_confirmation=False), dict(stop_confirmation=dict(confirmed=False)),
            dict(arm_results=dict(left=dict(execution_uncertain=True))),
            dict(stage_results=dict(reach=dict(execution_uncertain=True)))]
        for index, signal in enumerate(signals):
            for unsafe_payload in (False, True):
                with self.subTest(signal=signal, unsafe_payload=unsafe_payload):
                    self.harness.accepting = True
                    initial = self.harness.observe()['observation_id']
                    action = self.action('unconfirmed-' + str(index) + '-' + str(unsafe_payload), initial)
                    def uncertain(*args, **kwargs):
                        result = execute(*args, **kwargs)
                        result.update(status='stopped', executed_native_steps=1, **signal)
                        if unsafe_payload:
                            result['extra'] = {'evaluator_state': 'DO_NOT_PUBLISH'}
                        return result
                    with patch.object(self.robot, 'execute', side_effect=uncertain):
                        result = self.harness.execute(action)
                    receipt = result['receipt']
                    self.assertEqual((receipt['status'], receipt['executed_native_steps']), ('stopped', 1))
                    for key, value in signal.items():
                        if unsafe_payload and key in ('arm_results', 'stage_results'):
                            self.assertTrue(receipt['execution_uncertain'])
                        else:
                            self.assertEqual(receipt[key], value)
                    self.assertNotIn('DO_NOT_PUBLISH', canonical(receipt))
                    fresh = self.harness.observe()
                    self.assertFalse(self.harness.accepting)
                    executions = self.robot.executions
                    self.assertEqual(self.harness.execute(action), result)
                    refused = self.harness.execute(self.action(action['action_id'] + '-after', fresh['observation_id']))
                    self.assertEqual(refused['receipt']['reason'], 'run_stopping_or_reconciliation_required')
                    self.assertEqual(self.robot.executions, executions)
                    workbench = object.__new__(Workbench)
                    workbench.harness, workbench.store, workbench.identity = self.harness, self.store, self.identity
                    self.assertTrue(workbench.current_unknown_actions())
                    restored = Harness(self.store, self.robot, config=self.harness.config)
                    self.assertFalse(restored.accepting)

    def test_confirmed_checkpoint_allows_next_action_after_fresh_observe(self):
        execute = self.robot.execute
        def checkpoint(*args, **kwargs):
            result = execute(*args, **kwargs)
            result.update(status='stopped', executed_native_steps=1,
                stop_confirmation=dict(clock='paused', actuator_target='last_native_command_retained', settling_steps=0),
                extra={'evaluator_state': 'DO_NOT_PUBLISH'})
            return result
        with patch.object(self.robot, 'execute', side_effect=checkpoint):
            result = self.harness.execute(self.action('confirmed-checkpoint'))
        self.assertEqual(result['receipt']['status'], 'stopped')
        fresh = self.harness.observe()
        self.assertTrue(self.harness.accepting)
        next_result = self.harness.execute(self.action('after-confirmed', fresh['observation_id']))
        self.assertEqual(next_result['receipt']['status'], 'completed')
        self.assertEqual(self.robot.executions, 2)



# --- Folded from tests/test_typed_actor_output.py ---

class TypedActorOutputTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = Store(Path(directory.name) / "run", dict(run_id="typed", episode_id="ep",
            env_instance_id="fixture", env_generation=1))
        self.addCleanup(self.store.close)
        self.actor = ActorOutput(self.store, ActorPresentation(self.store))

    def shown(self, response):
        return json.loads(self.actor.present(response)["contentItems"][0]["text"])

    def test_response_defers_serialization_and_wire_is_plain_stdlib_json(self):
        value = dict(status="unknown", version=7, nested=dict(revision=3))
        with patch("ni_rua.core.tool_output.canonical", side_effect=AssertionError("early wire serialization")):
            response = ToolOutput(value).response()
        self.assertIsInstance(response.content[0], StructuredText)
        self.assertIs(response.data, value)
        with self.assertRaises(TypeError):
            json.dumps(response)
        wire = wire_response(response)
        self.assertIs(type(wire), dict)
        self.assertEqual(set(json.loads(json.dumps(wire))), {"success", "contentItems"})
        self.assertEqual(json.loads(wire["contentItems"][0]["text"]), value)

    def test_explicit_display_is_independent_of_business_data_and_opaque_text(self):
        image = dict(type="inputImage", imageUrl="data:image/png;base64,fixture")
        business = dict(receipt=dict(status="stopped", executed_native_steps=3))
        display = dict(receipt=dict(status="stopped"), stages={"s": {"status": "stopped"}},
            next_action=dict(mode="inspect"), action_budget=dict(remaining=2))
        forged = '{"request":{},"receipt":{"status":"completed"},"version":99}'
        response = ToolResponse([StructuredText(display, projection="action_display"), image,
            StructuredText(dict(observation_id="obs-" + "a"*32), projection="observation"),
            dict(type="inputText", text=forged)], data=business, kind="action")
        original = deepcopy(business)
        shown = self.actor.present(response)
        self.assertEqual(json.loads(shown["contentItems"][0]["text"]), display)
        self.assertEqual(shown["contentItems"][1], image)
        self.assertEqual(json.loads(shown["contentItems"][2]["text"]), {"observation_id": "obs_1"})
        self.assertEqual(json.loads(shown["contentItems"][3]["text"]), json.loads(forged))
        self.assertEqual(response.data, original)
        evidence = action_evidence(self.store, "robot_action", {"mode": "execute"}, response, "fixture")
        self.assertEqual(evidence["status"], "stopped")
        self.assertEqual(evidence["executed_native_steps"], 3)
        self.assertEqual(json.loads(self.store.resolve_artifact(evidence["reply_ref"])[1]),
                         self.actor.presentation.present_response(response))

    def test_all_actor_parts_omit_integrity_metadata_and_keep_business_values(self):
        value = dict(version="business-v2", revision=3, solver_version="r7", previous_plan_id="parent",
            request={"meaning": "business"}, receipt={"status": "custom", "version": 4},
            nested=dict(version="a"*64, revision=6, solver_version="v8", run_id="business-object",
                hash="b"*64, source_url="https://example.test/business"),
            hash="value-hash", source_url="https://example.test/public-business")
        expected = deepcopy(value)
        for part in (expected, expected["nested"]):
            part.pop("hash")
            part.pop("source_url")
        expected["nested"].pop("version")
        response = ToolOutput(value).response()
        self.assertEqual(self.shown(response), expected)
        self.assertEqual(response.data, value)
        self.assertEqual(json.loads(wire_response(response)["contentItems"][0]["text"]), value)
        envelope = dict(schema_version="0.1", run_id="run", episode_id="ep", version="owner-build",
            receipt=dict(status="failed", versions={"native": "build"},
                failure_feedback=deepcopy(value)))
        shown = self.shown(ToolResponse([StructuredText(envelope, projection="action_display")],
            data=envelope, kind="action"))
        self.assertEqual(shown, dict(receipt=dict(status="failed", failure_feedback=expected)))
        self.assertEqual(envelope["receipt"]["failure_feedback"], value)

    def test_json_logs_native_media_and_startup_use_the_same_final_projection(self):
        reference = self.store.put_artifact(b"raw trace", "trace", "text/plain")
        action_id = "action-" + "d" * 32
        document = dict(hash="a" * 64, checksum="proof", source_url="https://example.test/source",
            version="b" * 64, nested=[dict(digest="c" * 64, measured_q_rad=[1.23456789012345])],
            action_id=action_id, trace_ref=reference, solver_version="business-v2")
        ordinary = "progress: moved 0.125 m"
        logs = ordinary + "\n" + json.dumps(document) + "\nfinished\n"
        image = dict(type="inputImage", imageUrl="data:image/png;base64,fixture",
                     hash="image proof", source_url="https://example.test/image")
        audio = dict(type="inputAudio", audioUrl="data:audio/wav;base64,fixture",
                     checksum="audio proof", source_url="https://example.test/audio")
        response = ToolOutput(dict(stdout_text=logs, stdout_excerpt=dict(text=logs)),
            extra_content=(dict(type="inputText", text=json.dumps(document)), image, audio)).response()
        original = deepcopy(response.content)
        shown = self.actor.present(response)["contentItems"]
        expected = dict(nested=[dict(measured_q_rad=[1.23456789012345])], action_id="act_1",
            trace_ref="artifact:ref_1", solver_version="business-v2")
        displayed_logs = ordinary + "\n" + canonical(expected) + "\nfinished\n"
        self.assertEqual(json.loads(shown[0]["text"]),
            dict(stdout_text=displayed_logs, stdout_excerpt=dict(text=displayed_logs)))
        self.assertEqual(json.loads(shown[1]["text"]), expected)
        self.assertEqual(shown[2], {"type": "inputImage", "imageUrl": image["imageUrl"]})
        self.assertEqual(shown[3], {"type": "inputAudio", "audioUrl": audio["audioUrl"]})
        self.assertEqual(self.actor.present(document), expected)
        self.assertEqual(response.content, original)
        program = self.actor.presentation.present_response(response)
        self.assertEqual(json.loads(program["contentItems"][1]["text"]), expected)
        self.assertEqual(self.store.resolve_artifact(reference)[1], b"raw trace")

    def test_named_operation_projection_saves_full_original_and_does_not_guess_type(self):
        value = dict(source_kind="model_operation_judgment", version="business",
            execution=dict(status="stopped", actions=[dict(action_id="a", version=2)], blockers=["partial"]),
            answer=dict(assessment="uncertain", revision=3))
        self.assertEqual(self.shown(ToolOutput(value).response()), value)
        shown = self.shown(ToolOutput(value, projection="operation_review").response())
        self.assertEqual(shown["execution"], dict(status="stopped", blockers=["partial"], action_ids=["a"]))
        self.assertEqual(shown["answer"], value["answer"])
        reference = self.actor.restore(shown["details_ref"])
        self.assertEqual(json.loads(self.store.resolve_artifact(reference)[1]), value)

    def test_contract_error_keeps_typed_receipt_and_serializes_warning_last(self):
        value = dict(receipt=dict(status="completed", executed_native_steps=4))
        response = output_contract_error(ToolOutput(value, kind="action").response(), "robot_action",
            "missing_field", "/data/missing")
        self.assertIs(response.data, value)
        self.assertFalse(response.success)
        shown = self.actor.present(response)
        self.assertFalse(shown["success"])
        warning = json.loads(shown["contentItems"][0]["text"])["output_contract_error"]
        self.assertEqual(warning["phase"], "after_handler")
        self.assertTrue(warning["physical_action_executed"])
        self.assertIn("do not automatically replay", warning["suggestion"])
        self.assertEqual(json.loads(shown["contentItems"][1]["text"]), value)
        self.assertEqual(action_evidence(self.store, "robot_action", {"mode": "execute"}, response,
            "executed")["status"], "completed")

    def test_render_failure_keeps_known_physical_receipt_and_never_suggests_replay(self):
        business = dict(receipt=dict(status="stopped", executed_native_steps=4,
            error="motor_fault", execution_uncertain=True, stop_confirmation=dict(confirmed=False),
            future_diagnostic=dict(sensor='unavailable')))
        response = ToolResponse([StructuredText(dict(bad=object())),
            dict(type="inputImage", imageUrl="data:image/png;base64,fixture")],
            kind="action", data=business)
        shown = self.actor.present(response, tool="robot_action")
        self.assertFalse(shown["success"])
        warning = json.loads(shown["contentItems"][0]["text"])["output_contract_error"]
        self.assertEqual(warning["reason"], "actor_display_unavailable")
        self.assertTrue(warning["physical_action_executed"])
        self.assertIn("do not automatically replay", warning["suggestion"])
        self.assertEqual(json.loads(shown["contentItems"][1]["text"]), business)
        self.assertEqual(shown["contentItems"][2]["type"], "inputImage")
        self.assertEqual(response.data, business)

    def test_policy_step_without_current_observation_raises_domain_value_error(self):
        tmp = self.enterContext(tempfile.TemporaryDirectory())
        identity = dict(run_id="policy-empty-obs", episode_id="ep", env_instance_id="env", env_generation=1)
        store = Store(Path(tmp) / "store", identity)
        self.addCleanup(store.close)
        harness = Harness(store, _HarnessProtocolRobot(tmp))
        harness.start()
        obs_id = harness.latest["observation_id"]
        harness.action_policy = SimpleNamespace(infer=lambda *a, **kw: dict(actions=[[0.0] * 14]))
        harness.task_planning.call(
            dict(operation="submit", plan="test", observation_id=obs_id, ledger=[]),
            harness.identity,
            harness.latest,
            {},
        )
        harness.latest = None
        with self.assertRaisesRegex(ValueError, "current_observation_unavailable"):
            harness.policy_step(
                dict(action_id="act-no-obs", observation_id=obs_id, prompt="adjust", prefix=1),
                {},
            )

    def test_wire_only_receipt_json_does_not_gain_typed_trust_or_projection(self):
        value = dict(request={"x": 1}, receipt={"status": "completed"}, nested={"version": 3})
        wire = dict(success=True, contentItems=[dict(type="inputText", text=canonical(value))])
        self.assertEqual(self.actor.present(wire), wire)
        self.assertEqual(action_evidence(self.store, "robot_action", {"mode": "execute"}, wire,
            "untyped")["status"], "unknown")
        with self.assertRaisesRegex(ValueError, "missing_plan"):
            reviewed_plan(wire)

    def test_metadata_projection_keeps_evidence_and_restores_functional_ids(self):
        original_id = "obs-" + "a" * 32
        value = dict(schema_version="0.1", observation_id=original_id,
            receipt=dict(status="failed", completed_native_steps=4, reason="tracking_divergence"),
            robot_state=dict(q_rad=[.1], source_kind="measurement"), version="internal", sha256="b" * 64)
        original = deepcopy(value)
        shown = self.shown(ToolOutput(value, projection="observation").response())
        self.assertEqual(shown["observation_id"], "obs_1")
        self.assertEqual(shown["receipt"], value["receipt"])
        self.assertEqual(shown["robot_state"], value["robot_state"])
        self.assertTrue({"schema_version", "version", "sha256"}.isdisjoint(shown))
        self.assertEqual(self.actor.restore(dict(observation_id="obs_1")), dict(observation_id=original_id))
        self.assertEqual(value, original)
        rebuilt = ActorOutput(self.store, ActorPresentation(self.store))
        self.assertEqual(rebuilt.restore(dict(source="client.call('observe', {'observation_id':'obs_1'})"))["source"],
            "client.call('observe', {'observation_id':'" + original_id + "'})")

    def test_startup_capability_projection_keeps_limits_without_old_execution_contracts(self):
        native = dict(motion_execution_mode="resampled", unified_action=dict(max_stages=32),
            arm_joint_limits_rad={"left": [[-1., 1.]]}, arm_max_step_rad={"left": [.1]},
            action_speed=dict(speed_scale=dict(min=.25, max=2.),
                waypoint_speed_scales=dict(available=True), gripper="legacy rebased jaws"),
            versions=dict(robot="old-build"), max_waypoints=8,
            motion_preview=dict(max_waypoints=1), waypoint_failure_semantics="late planning can partly move")
        original = deepcopy(native)
        shown = self.actor.present(native, projection="capabilities")
        for field in ("arm_joint_limits_rad", "arm_max_step_rad", "unified_action"):
            self.assertEqual(shown[field], native[field])
        self.assertEqual(shown["action_speed"]["speed_scale"], native["action_speed"]["speed_scale"])
        self.assertFalse({"versions", "motion_preview", "max_waypoints", "waypoint_failure_semantics"} & shown.keys())
        self.assertNotIn("waypoint_speed_scales", shown["action_speed"])
        self.assertEqual(native, original)

    def test_shared_action_metadata_removes_only_known_nested_owner_fields_and_keeps_reviews(self):
        owner = dict(schema_version="0.1", run_id="run", episode_id="episode",
                     env_instance_id="environment", env_generation=2)
        business = dict(version="semantic-v2", revision=7, source="measured", schema_version="business-schema")
        receipt = dict(owner, status="unknown", executed_native_steps=None, stop_confirmed=False,
            batch_revision=3, plan_revision=4, stage_results={"revision": deepcopy(business)},
            failure_feedback=dict(schema_version="1", reason="tracking_error", diagnostic=deepcopy(business)))
        value = dict(owner, request=dict(owner, type="BatchAction", batch_expected_revision=3,
            stages=[dict(id="revision", kind="move", action=dict(owner,
                based_on_observation_id="obs-current", timeout_s=120, business=deepcopy(business)))]),
            receipt=receipt)
        original = deepcopy(value)
        before = self.store.db.total_changes
        shown = project_metadata(value, "action_result")
        child = shown["request"]["stages"][0]["action"]
        self.assertTrue(set(owner).isdisjoint(child))
        self.assertEqual(child, dict(based_on_observation_id="obs-current", timeout_s=120, business=business))
        self.assertEqual(shown["receipt"]["stage_results"], {"revision": business})
        self.assertEqual(shown["receipt"]["failure_feedback"], dict(reason="tracking_error", diagnostic=business))
        self.assertEqual((shown["receipt"]["batch_revision"], shown["receipt"]["plan_revision"]), (3, 4))
        self.assertEqual(project_metadata([value], "action_history"), [shown])
        self.assertEqual(project_metadata([receipt] * 3, "program_action_evidence"), [shown["receipt"]] * 3)
        self.assertEqual(value, original)
        self.assertEqual(self.store.db.total_changes, before)

        review = dict(version="owner-build", execution=dict(status="stopped", blockers=["partial"],
            actions=[dict(action_id="a", status="unknown", executed_native_steps=None,
                          failure_feedback=dict(witnesses=[.123456789] * 4))]),
            answer=dict(revision=2, source="visible evidence"))
        shown_review = project_metadata(review, "operation_review")
        self.assertEqual(shown_review["execution"], review["execution"])
        self.assertEqual(shown_review["answer"], review["answer"])
        self.assertNotIn("details_ref", shown_review)
        self.assertNotIn("version", shown_review)


class ToolOutputAndReadbackTests(unittest.TestCase):
    def test_summary_order_bounded_excerpts_pagination_and_field_selection(self):
        image = dict(type="inputImage", imageUrl="data:image/png;base64,fixture")
        output = wire_response(ToolOutput(dict(status="unknown"), dict(full_ref="artifact:full"), [image]).response())
        self.assertTrue(output["success"])
        self.assertEqual(json.loads(output["contentItems"][0]["text"]), dict(status="unknown"))
        self.assertEqual(json.loads(output["contentItems"][1]["text"]), dict(full_ref="artifact:full"))
        self.assertEqual(output["contentItems"][2], image)
        self.assertEqual(len(ToolOutput(None).content_items()), 1)

        raw = b"BEGIN\n" + b"x" * 20000 + b"\nterminal failure: partial action"
        excerpt = bounded_text(raw, max_bytes=128)
        self.assertTrue(excerpt["text"].startswith("BEGIN\n"))
        self.assertTrue(excerpt["text"].endswith("terminal failure: partial action"))
        self.assertEqual(excerpt["byte_ranges"], [[0, 64], [len(raw) - 64, len(raw)]])
        self.assertEqual(bounded_text("您好")["text"], "您好")
        self.assertEqual(bounded_text("")["byte_ranges"], [[0, 0]])
        self.assertFalse(bounded_text(raw, max_bytes=None)["truncated"])

        self.assertEqual(validate_page(), (0, 25))
        for start, limit in ((True, 2), (-1, 1), (1.0, 2), (0, False), (0, 0), (0, 101)):
            with self.assertRaises(ValueError):
                validate_page(start, limit)
        self.assertEqual(resolve_detail(program=True), "full")
        self.assertEqual(resolve_detail("summary", program=True), "summary")
        with self.assertRaises(ValueError):
            resolve_detail("unknown")

        original = dict(a=[.123456789], b=None)
        selected = select_fields(original, ("a", "b", "c"))
        self.assertEqual(selected, original)
        selected["a"].append(1)
        self.assertEqual(original["a"], [.123456789])

    def test_history_and_program_dispatch_expose_full_and_paged_readbacks(self):
        tmp = self.enterContext(tempfile.TemporaryDirectory())
        root = Path(tmp)
        store = Store(root / "store", dict(run_id="output-fixture", episode_id="ep", env_instance_id="env", env_generation=1))
        self.addCleanup(store.close)
        robot = FixtureRobot(root)
        harness = Harness(store, robot)
        harness.start()

        def call(name, args, program=False):
            correlation = dict(thread_id="t", turn_id="turn", tool_call_id=name)
            if program:
                correlation["program_id"] = "program-caller"
            reply = harness.dispatch(name, args, correlation)
            self.assertTrue(reply["success"])
            return json.loads(reply["contentItems"][0]["text"])

        for name in ("robot_action_status", "history_list_actions", "history_read_trace", "program_status"):
            definition = next(item for item in TOOLS if item["name"] == name)
            self.assertEqual(definition["inputSchema"]["properties"]["detail"]["enum"], ["summary", "full"])
        trace = [dict(sim_step=i, q_rad=[i + .123456789]) for i in range(40)]
        ref = store.put_artifact(canonical(trace).encode(), "actual_trace", "application/json")
        request = dict(action_id="action", based_on_observation_id=harness.latest["observation_id"],
                       path=dict(waypoints=[dict(position_m=[.123456789] * 3)]))
        receipt = dict(status="unknown", reason="stop_unconfirmed", actual_trace_ref=ref,
                       executed_native_steps=4, completed_native_steps=3, versions=dict(worker="fixture"))
        store.save("action_request", "action", request)
        harness.states["action"] = deepcopy(receipt)
        original = dict(request=request, receipt=receipt)
        self.assertNotIn("versions", call("robot_action_status", dict(action_id="action"))["receipt"])
        self.assertEqual(call("robot_action_status", dict(action_id="action", detail="full")), original)
        self.assertEqual(call("robot_action_status", dict(action_id="action"), program=True), original)
        self.assertEqual(call("history_list_actions", dict(detail="full")), [original])
        args = dict(action_id="action", kind="actual_trace")
        self.assertEqual(len(call("history_read_trace", args)["trace"]), 25)
        self.assertEqual(call("history_read_trace", {**args, "detail": "full"}), trace)

        raw_log = b"start\n" + b"x" * 40000 + b"\nterminal evidence"
        log_ref = store.put_artifact(raw_log, "program_stdout", "text/plain")
        job = dict(job_id="program", status="cancelled", stop_confirmed=False,
                   stdout_ref=log_ref, stdout_text=raw_log[:8192].decode(), stdout_truncated=True)
        calls = []
        harness.programs = SimpleNamespace(run=lambda a, _c: (calls.append(deepcopy(a)), deepcopy(job))[1],
                                           status=lambda _: deepcopy(job))
        summary = call("program_run", dict(source="print('caller output')"))
        self.assertFalse(summary["stdout_truncated"])
        self.assertEqual(summary["stdout_text"].encode(), raw_log)
        call("program_run", dict(source="print('caller output')", detail="full"))
        self.assertEqual(calls, [dict(source="print('caller output')")] * 2)
        self.assertEqual(robot.executions, 0)

    def test_internal_runtime_crash_archives_tool_result_and_trips_circuit_breaker(self):
        import asyncio
        from ni_rua.runtime.supervision import wait_turn
        from ni_rua.runtime.tools.builtins import output_schemas
        from ni_rua.runtime.tools.registry import ToolSpec

        tmp = self.enterContext(tempfile.TemporaryDirectory())
        root = Path(tmp)
        store = Store(
            root / "store",
            dict(run_id="crash-fixture", episode_id="ep", env_instance_id="env", env_generation=1),
        )
        self.addCleanup(store.close)
        robot = FixtureRobot(root)

        mode_box = {"kind": "type_error"}

        def flaky_handler(_context, _args, _correlation):
            if mode_box["kind"] == "type_error":
                raise TypeError("unsupported operand type(s) for -: 'NoneType' and 'float'")
            if mode_box["kind"] == "ipc_type_error":
                raise RuntimeError("TypeError: remote worker crash")
            if mode_box["kind"] == "domain_value_error":
                raise ValueError("stale_observation")
            from ni_rua.core.tool_output import ToolOutput
            return ToolOutput(
                dict(
                    receipt=dict(
                        action_id="ok",
                        status="completed",
                        executed_native_steps=0,
                        physics_advanced=False,
                    )
                )
            ).response()

        spec = ToolSpec(
            dict(
                type="function",
                name="flaky_internal_tool",
                description="Fixture tool for internal crash handling.",
                inputSchema=dict(type="object", additionalProperties=True),
            ),
            flaky_handler,
            output_schemas=output_schemas("robot_wait"),
        )
        harness = Harness(store, robot, tool_specs=(spec,))
        harness.start()
        harness.thread_id = "thread-1"

        # Domain ValueError does NOT count as internal_runtime_error
        mode_box["kind"] = "domain_value_error"
        with self.assertRaisesRegex(ValueError, "stale_observation"):
            harness.handle_tool(
                dict(threadId="thread-1", turnId="turn-1", callId="c-domain", tool="flaky_internal_tool", arguments={})
            )
        self.assertEqual(len(harness.internal_runtime_errors), 0)
        self.assertIsNone(harness.internal_runtime_tripped)

        # Internal TypeError & IPC RuntimeError("TypeError: ...") archive failed tool_result and trip on 3rd crash
        for idx, kind in enumerate(("type_error", "ipc_type_error", "type_error"), start=1):
            mode_box["kind"] = kind
            with self.assertRaises(RuntimeError) as caught:
                harness.handle_tool(
                    dict(
                        threadId="thread-1",
                        turnId="turn-1",
                        callId=f"c-crash-{idx}",
                        tool="flaky_internal_tool",
                        arguments={"attempt": idx},
                    )
                )
            self.assertEqual(getattr(caught.exception, "code", None), "internal_runtime_error")
            if idx < 3:
                self.assertIsNone(harness.internal_runtime_tripped)
                self.assertTrue(harness.accepting)

        self.assertIsNotNone(harness.internal_runtime_tripped)
        self.assertFalse(harness.accepting)
        self.assertEqual(harness.internal_runtime_tripped["tool"], "flaky_internal_tool")
        self.assertEqual(harness.internal_runtime_tripped["consecutive_errors"], 3)

        # Verify tool_result records were archived for all 3 internal crashes (not skipped)
        archived_crash_results = [
            rec
            for rec in store.records("tool_result")
            if rec.get("tool") == "flaky_internal_tool"
        ]
        self.assertEqual(len(archived_crash_results), 3)
        raw_payload = json.loads(store.resolve_artifact(archived_crash_results[-1]["raw_ref"], ("observer",))[1])
        self.assertFalse(raw_payload["success"])
        self.assertEqual(raw_payload["kind"], "error")
        self.assertEqual(raw_payload["data"]["code"], "internal_runtime_error")
        self.assertIsNone(raw_payload["data"]["minimal_example"])

        internal_events = store.events_by_type(("tool.internal_runtime_error",))
        self.assertEqual(len(internal_events), 3)

        # wait_turn immediately fast-fails with reason="internal_runtime_error"
        session = SimpleNamespace(
            turn_done=threading.Event(),
            last_activity_at=time.monotonic(),
            model_proxy=None,
            model_limit_reached=threading.Event(),
        )
        finished = asyncio.run(wait_turn(session, asyncio.Event(), harness=harness))
        self.assertFalse(finished)
        self.assertEqual(harness.wait_stop_reason, "internal_runtime_error")

        # Environment reset clears per-episode circuit breaker state
        harness.capabilities["reset"] = True
        robot.reset = lambda seed=0, layout_id=0: {"observation": robot.observe()}
        harness.reset(seed=1, layout_id=0)
        self.assertIsNone(harness.internal_runtime_tripped)
        self.assertEqual(harness.consecutive_internal_runtime_errors, 0)
        self.assertTrue(harness.accepting)

    def test_post_execution_store_failure_clears_active_action_and_returns_receipt(self):
        from ni_rua.core.storage.store import StoreClosed

        tmp = self.enterContext(tempfile.TemporaryDirectory())
        root = Path(tmp)
        store = Store(
            root / "store",
            dict(run_id="settle-fixture", episode_id="ep", env_instance_id="env", env_generation=1),
        )
        self.addCleanup(store.close)
        robot = WaitFixtureRobot(root)
        harness = Harness(store, robot, enable_wait=True)
        harness.start()
        obs_id = harness.latest["observation_id"]

        orig_save = store.save

        def fail_receipt_save(kind, identifier, value):
            if kind == "action_receipt":
                raise StoreClosed("store_closed")
            return orig_save(kind, identifier, value)

        with unittest.mock.patch.object(store, "save", side_effect=fail_receipt_save):
            resp = harness.invoke(
                "robot_wait",
                dict(action_id="act-store-fail", observation_id=obs_id, native_steps=2),
                dict(tool_call_id="call-settle"),
                role="actor",
            )
        self.assertIsNone(harness.active_action)
        self.assertIsNone(harness.active_context)
        self.assertEqual(len(harness._in_flight_actions), 0)
        self.assertFalse(harness.accepting)
        self.assertEqual(resp.data["receipt"]["status"], "completed")
        self.assertEqual(
            resp.data["receipt"]["storage_persist_error"],
            dict(error_type="StoreClosed", action_replay_permitted=False),
        )

    def test_worker_ipc_internal_type_error_to_episode_summary_unscored_infrastructure_e2e(self):
        import asyncio
        import os
        import sys
        import textwrap
        from ni_rua.apps.evaluation.episode import episode_summary, record_evaluation, wait_benchmark
        from ni_rua.core.config import Config
        from ni_rua.core.tool_errors import InternalToolRuntimeError, ToolInputError, WorkerRemoteError, error_payload
        from ni_rua.cortex.agents.mcp_server import tool_bridge_error_response
        from ni_rua.cortex.supervision.actor_stall import completed_turn_activity
        from ni_rua.cortex.supervision.recovery_evidence import completed_turn_evidence
        from ni_rua.robot.ipc.process import WorkerProcess
        from ni_rua.runtime.programs.runner import ProgramRunner
        from ni_rua.runtime.tools.builtins import output_schemas
        from ni_rua.runtime.tools.registry import ToolSpec

        repo_root = str(Path(__file__).resolve().parents[3])
        tmp = self.enterContext(tempfile.TemporaryDirectory(dir="/tmp", prefix="aur-ipc-"))
        root = Path(tmp)
        identity = dict(run_id="e2e-ipc-crash", episode_id="ep-1", env_instance_id="env-1", env_generation=1)
        store = Store(root / "store", identity)
        self.addCleanup(store.close)

        worker_script = root / "ipc_worker_stub.py"
        worker_script.write_text(
            textwrap.dedent(
                f"""\
                import json
                import sys
                sys.path.insert(0, {repo_root!r})
                from ni_rua.robot.ipc.protocol import WIRE_PREFIX, serve_worker

                class StubAdapter:
                    def capabilities(self):
                        return {{"robot_model_id": "arx_x5", "arms": ["left", "right"]}}

                    def reach_batch(self, payload):
                        mode = (payload or {{}}).get("mode")
                        if mode == "type_bug":
                            raise TypeError("unsupported operand type(s) in /Users/secret/private/robot_worker.py:99")
                        if mode == "domain_infeasible":
                            raise ValueError("path_infeasible: joint limit exceeded")
                        if mode == "domain_stale":
                            raise ValueError("stale_observation")
                        return {{"verdicts": [{{"id": "s1", "reachable": True}}]}}

                    def close(self, _progress=None):
                        return None

                def emit(msg):
                    sys.stdout.write(WIRE_PREFIX + json.dumps(msg) + "\\n")
                    sys.stdout.flush()

                serve_worker(lambda _args: StubAdapter(), None, emit)
                """
            ),
            encoding="utf-8",
        )

        worker = WorkerProcess(root / "worker")
        self.addCleanup(worker.close)
        env = dict(os.environ, PYTHONPATH=repo_root)
        worker.start_process([sys.executable, "-u", str(worker_script)], cwd=root, environment=env)

        robot = FixtureRobot(root)

        def ipc_probe_handler(_context, args, _correlation):
            out = worker._request("reach_batch", payload=dict(mode=args["mode"]), timeout=10)
            return ToolOutput(
                dict(
                    receipt=dict(
                        action_id="ok",
                        status="completed",
                        executed_native_steps=0,
                        physics_advanced=False,
                        verdicts=out.get("verdicts"),
                    )
                )
            ).response()

        spec = ToolSpec(
            dict(
                type="function",
                name="ipc_worker_probe",
                description="IPC worker exception seam probe.",
                inputSchema=dict(
                    type="object",
                    required=["mode"],
                    additionalProperties=False,
                    properties=dict(mode=dict(type="string")),
                ),
            ),
            ipc_probe_handler,
            output_schemas=output_schemas("robot_wait"),
        )
        cfg = Config.from_dict({"features": {"sandbox_enabled": False}})
        harness = Harness(store, robot, observation_profile="rgb_proprio_v1", config=cfg, tool_specs=(spec,))
        harness.start()
        harness.thread_id = "thread-e2e"
        runner = ProgramRunner(harness.program_services())
        self.addCleanup(runner.close)
        harness.programs = runner

        # 1) Legitimate caller schema ValidationError -> ToolInputError (NEVER trips internal_runtime_tripped)
        with self.assertRaises(ToolInputError) as bad_schema:
            harness.handle_tool(
                dict(
                    threadId="thread-e2e",
                    turnId="turn-1",
                    callId="c-schema",
                    tool="ipc_worker_probe",
                    arguments={"unexpected_field": 1},
                )
            )
        bridge_schema = tool_bridge_error_response(
            bad_schema.exception, tool="ipc_worker_probe", arguments={"unexpected_field": 1}, store=store, call_id="c-schema"
        )
        schema_data = json.loads(bridge_schema["contentItems"][0]["text"])
        self.assertEqual(schema_data["error_category"], "input_error")
        self.assertTrue(schema_data["retryable"])

        # 2) Snake_case domain ValueErrors across IPC (path_infeasible, stale_observation) NEVER trip circuit breaker
        for call_id, dom_mode, expected_code in (
            ("c-dom-1", "domain_infeasible", "path_infeasible"),
            ("c-dom-2", "domain_stale", "stale_observation"),
        ):
            with self.assertRaises(WorkerRemoteError) as dom_err:
                harness.handle_tool(
                    dict(
                        threadId="thread-e2e",
                        turnId="turn-1",
                        callId=call_id,
                        tool="ipc_worker_probe",
                        arguments={"mode": dom_mode},
                    )
                )
            self.assertEqual(dom_err.exception.category, "input_error")
            self.assertEqual(dom_err.exception.code, expected_code)
            for formatted in (
                error_payload(dom_err.exception, tool="ipc_worker_probe", arguments={"mode": dom_mode}),
                json.loads(
                    tool_bridge_error_response(
                        dom_err.exception, tool="ipc_worker_probe", arguments={"mode": dom_mode}, store=store, call_id=call_id
                    )["contentItems"][0]["text"]
                ),
            ):
                self.assertEqual(formatted["error_category"], "input_error")
                self.assertEqual(formatted["code"], expected_code)
                self.assertTrue(formatted["retryable"])

        # 3) Actor-authored main.py script bugs inside program_run NEVER trip internal_runtime_tripped
        for idx, bad_script in enumerate(("1 / 0\n", "x = None\nx.missing_attr()\n"), start=1):
            prog_reply = harness.handle_tool(
                dict(
                    threadId="thread-e2e",
                    turnId="turn-1",
                    callId=f"c-actor-script-{idx}",
                    tool="program_run",
                    arguments={"source": bad_script},
                )
            )
            self.assertTrue(prog_reply["success"])
            prog_body = json.loads(prog_reply["contentItems"][0]["text"])
            self.assertEqual(prog_body["status"], "failed")

        self.assertEqual(len(harness.internal_runtime_errors), 0)
        self.assertIsNone(harness.internal_runtime_tripped)
        self.assertTrue(harness.accepting)

        # 4) Internal TypeError raised inside robot_worker across IPC -> WorkerRemoteError(category="internal_runtime_error")
        # -> Harness.invoke archives failed tool_result & increments circuit breaker -> trips at 3!
        for idx in (1, 2, 3):
            call_id = f"c-ipc-bug-{idx}"
            with self.assertRaises(InternalToolRuntimeError) as bug_cm:
                harness.handle_tool(
                    dict(
                        threadId="thread-e2e",
                        turnId="turn-1",
                        callId=call_id,
                        tool="ipc_worker_probe",
                        arguments={"mode": "type_bug"},
                    )
                )
            raw_worker_err = bug_cm.exception.__cause__
            self.assertIsInstance(raw_worker_err, WorkerRemoteError)
            self.assertEqual(raw_worker_err.category, "internal_runtime_error")
            self.assertEqual(raw_worker_err.remote_type, "TypeError")

            for formatted in (
                error_payload(raw_worker_err, tool="ipc_worker_probe", arguments={"mode": "type_bug"}),
                error_payload(bug_cm.exception, tool="ipc_worker_probe", arguments={"mode": "type_bug"}),
                json.loads(
                    tool_bridge_error_response(
                        raw_worker_err, tool="ipc_worker_probe", arguments={"mode": "type_bug"}, store=store, call_id=call_id
                    )["contentItems"][0]["text"]
                ),
                json.loads(
                    tool_bridge_error_response(
                        bug_cm.exception, tool="ipc_worker_probe", arguments={"mode": "type_bug"}, store=store, call_id=call_id
                    )["contentItems"][0]["text"]
                ),
            ):
                self.assertEqual(formatted["code"], "internal_runtime_error")
                self.assertEqual(formatted["error_category"], "internal_runtime_error")
                self.assertEqual(formatted["error_type"], "TypeError")
                self.assertFalse(formatted["retryable"])
                self.assertEqual(formatted["field_path"], "")
                self.assertIsNone(formatted["actual"])
                self.assertIsNone(formatted["minimal_example"])
                self.assertNotIn("/Users/secret", formatted["error"])

            if idx == 1:
                # Intervening pure-Python program_run does not wipe the consecutive internal tool-crash streak
                ok_prog = harness.handle_tool(
                    dict(
                        threadId="thread-e2e",
                        turnId="turn-1",
                        callId="c-pure-py",
                        tool="program_run",
                        arguments={"source": "print(2 + 2)\n"},
                    )
                )
                self.assertTrue(ok_prog["success"])
                self.assertEqual(harness.consecutive_internal_runtime_errors, 1)

            if idx < 3:
                self.assertIsNone(harness.internal_runtime_tripped)
                self.assertTrue(harness.accepting)

        self.assertIsNotNone(harness.internal_runtime_tripped)
        self.assertFalse(harness.accepting)
        self.assertEqual(harness.internal_runtime_tripped["tool"], "ipc_worker_probe")
        self.assertEqual(harness.internal_runtime_tripped["error_type"], "TypeError")
        self.assertEqual(harness.internal_runtime_tripped["consecutive_errors"], 3)

        # Verify archived tool_result records for all 3 IPC internal crashes
        probe_results = [r for r in store.records("tool_result") if r.get("tool") == "ipc_worker_probe"]
        self.assertEqual(len(probe_results), 3)
        for rec in probe_results:
            archived_json = json.loads(store.resolve_artifact(rec["raw_ref"], ("observer",))[1])
            self.assertFalse(archived_json["success"])
            self.assertEqual(archived_json["kind"], "error")
            self.assertEqual(archived_json["data"]["code"], "internal_runtime_error")
            self.assertEqual(archived_json["data"]["error_category"], "internal_runtime_error")
            self.assertIsNone(archived_json["data"]["minimal_example"])

        # Crashed calls earn zero novel_investigation credit in turn activity / recovery_evidence
        turn_act = completed_turn_activity(store, thread_id="thread-e2e", turn_id="turn-1", episode_id=harness.identity["episode_id"])
        turn_ev = completed_turn_evidence(
            store,
            thread_id="thread-e2e",
            turn_id="turn-1",
            episode_id=harness.identity["episode_id"],
            native_action_count=harness.native_action_count,
        )
        self.assertFalse(any(fp.startswith("ipc_worker_probe:") for fp in turn_act))
        self.assertFalse(any(fp.startswith("ipc_worker_probe:") for fp in turn_ev["activity"]))

        # 5) wait_benchmark fast-fails with wait_stop_reason="internal_runtime_error" and episode_summary -> unscored_infrastructure
        record_evaluation(
            store,
            harness.identity,
            dict(
                task_success=False,
                normalized_score=0.0,
                valid_episode=True,
                native_reward=0.0,
                native_process_score=0.0,
                success_authority="robodojo_evaluator",
            ),
        )
        done_evt = threading.Event()
        done_evt.set()
        session = SimpleNamespace(
            thread_id="thread-e2e",
            turn_id="turn-1",
            state="completed",
            turn_done=done_evt,
            turn_result={"status": "completed"},
            last_activity_at=time.monotonic(),
            model="gpt-6.1-sol",
            provider="openai",
            price_card=None,
            model_proxy=None,
            model_limit_reached=threading.Event(),
        )
        workbench = SimpleNamespace(
            session=session,
            harness=harness,
            store=store,
            identity=harness.identity,
            config=Config(),
            control_lock=asyncio.Lock(),
            has_active_programs=lambda: False,
            current_unknown_actions=lambda: [],
        )
        bench_finished = asyncio.run(wait_benchmark(workbench, asyncio.Event()))
        self.assertFalse(bench_finished)
        self.assertEqual(harness.wait_stop_reason, "internal_runtime_error")

        summary = episode_summary(
            store,
            harness.identity,
            session,
            harness,
            benchmark=True,
            final_status="failed",
            started_at=time.monotonic() - 1.0,
        )
        self.assertEqual(summary["primary_outcome"], "unscored_infrastructure")
        self.assertFalse(summary["scored_episode"])
        self.assertIsNone(summary["task_success"])
        self.assertIsNone(summary["normalized_score"])
        self.assertEqual(summary["termination_reason"], "internal_runtime_error")
        self.assertIn("internal_runtime_error", summary["infrastructure_error"])
        self.assertIn("TypeError", summary["infrastructure_error"])



if __name__ == "__main__":
    unittest.main()
