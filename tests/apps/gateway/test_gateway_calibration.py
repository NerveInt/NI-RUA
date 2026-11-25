"""Gateway endpoints of the guided calibration workflow (/v1/calibration/*, /calibration).

Synthetic only: a fake robot with an ``operator`` method records the worker ops. No serial port, camera,
Codex or model. Lease, idle/motion gating, admission handling, request validation and the route -> worker op
mapping are checked here; the worker side is covered by test_so101_calibration_runner.py.
"""

import asyncio
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest

from aiohttp.test_utils import TestClient, TestServer

from ni_rua.apps.gateway.extensions import calibration as gateway_calibration
from ni_rua.apps.gateway.extensions.calibration import calibration_badge, validate_begin, validate_step_start
from ni_rua.apps.gateway.workbench import Workbench
from ni_rua.core.config import Config
from ni_rua.robot.calibration.workflow import capture as calibration_capture, runner as calibration_runner
from ni_rua.robot.calibration.workflow import state_machine as workflow
from support.paths import isolate_home


def setUpModule():
    isolate_home()   # a fresh Workbench opens the grasp memory store at its '~' default


NI_RUA_TEST_TIER = "component"


class FakeRobot:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.calls = []
        self.results = {}
        self.block = None
        self.entered = None

    def operator(self, command, progress_callback=None, **arguments):
        if self.block is not None and command == "calib_step_start":
            if self.entered is not None:
                self.entered.set()
            self.block.wait(5)
        self.calls.append((command, arguments))
        result = self.results.get(command)
        if isinstance(result, Exception):
            raise result
        return dict(result or dict(status="ok"), command=command)

    def close(self):
        return None


from functools import lru_cache


@lru_cache(maxsize=1)
def _cached_base_config():
    return Config.load()


class CalibrationEndpoints(unittest.IsolatedAsyncioTestCase):
    @property
    def _base_config(self):
        return _cached_base_config()

    async def asyncSetUp(self):
        self.temporary_dir = self.enterContext(tempfile.TemporaryDirectory())
        root = Path(self.temporary_dir)
        self.robot_dir = root / "robot-run"
        self.robot = FakeRobot(self.robot_dir)
        self.workbench = Workbench(root / "synthetic-calibration", robot=self.robot, config=self._base_config)
        self.workbench.starting, self.workbench.status = False, "running"
        session = SimpleNamespace(thread_id="fixture-thread", turn_id=None, state="completed", effort="high")
        session.interrupt = lambda: setattr(session, "state", "interrupted")
        session.close = lambda: setattr(session, "state", "closed")
        self.workbench.session = session
        self.observations = 0

        def observe(correlation=None):
            self.observations += 1
            observation = dict(observation_id=f"obs-fresh-{self.observations}")
            self.workbench.harness.latest = observation
            return observation

        self.workbench.harness.observe = observe
        self.workbench.harness.capabilities = {"robot_model_id": "so101"}
        self.workbench.harness.accepting = True
        self.client = TestClient(TestServer(self.workbench.app()))
        await self.client.start_server()
        self.sequence = 0

    async def asyncTearDown(self):
        await self.client.close()
        await self.workbench.close()

    def origin(self):
        return {"Origin": f"http://{self.client.host}:{self.client.port}"}

    async def control(self, command, **fields):
        self.sequence += 1
        response = await self.client.post("/v1/control", json=dict(command=command, request_id=f"req-{self.sequence}", **fields),
                                          headers=self.origin())
        return response.status, await response.json()

    async def claim(self):
        status, body = await self.control("claim_input", client_id="calibration-page")
        self.assertEqual(status, 200)
        return dict(client_id="calibration-page", writer_token=body["writer_token"])

    async def post(self, path, request_id=None, **fields):
        self.sequence += 1
        response = await self.client.post("/v1/calibration/" + path, json=dict(request_id=request_id or f"cal-{self.sequence}", **fields),
                                          headers=self.origin())
        return response.status, await response.json()

    def calls(self, op=None):
        return [call for call in self.robot.calls if op is None or call[0] == op]

    async def test_xy_capture_keeps_admission_and_is_idle_gated_and_idempotent(self):
        lease = await self.claim()
        self.robot.results["calib_step_start"] = dict(status="captured", physics_advanced=False)
        fields = dict(params={"action": "xy_capture"}, **lease)
        for _ in range(2):
            code, result = await self.post("steps/S6/start", request_id="xy-once", **fields)
            self.assertEqual(code, 200, result)
            self.assertEqual(result["status"], "captured")
        self.assertEqual(len(self.calls("calib_step_start")), 1)
        self.assertTrue(self.workbench.harness.accepting)
        self.assertEqual(self.observations, 0)
        self.workbench.session.state = "running"
        code, result = await self.post("steps/S6/start", **fields)
        self.assertNotEqual(code, 200)
        self.assertEqual(len(self.calls("calib_step_start")), 1)

    async def test_xy_unknown_measurement_is_not_unknown_execution(self):
        lease = await self.claim()
        self.robot.results["calib_step_start"] = dict(status="evaluated", physics_advanced=False,
                                                     result=dict(status="unknown", passed=None))
        params = dict(action="xy_evaluate", snapshot_id="a" * 32, fixed_pixel=[1, 2], wrist_pixel=[3, 4],
                      max_xy_error_m=.005, max_reprojection_error_px=3, min_parallax_deg=1)
        code, result = await self.post("steps/S6/start", params=params, **lease)
        self.assertEqual((code, result["status"]), (200, "evaluated"))
        self.assertTrue(self.workbench.harness.accepting)
        self.assertEqual(self.observations, 0)

    async def test_xy_previews_are_frozen_and_cannot_escape_telemetry(self):
        directory = self.workbench.telemetry_directory(); directory.mkdir(parents=True, exist_ok=True)
        identifier = "a" * 32
        (directory / f"xy-{identifier}-fixed.jpg").write_bytes(b"frozen-image")
        response = await self.client.get(f"/v1/calibration/xy/{identifier}/fixed.jpg")
        self.assertEqual(response.status, 200)
        self.assertEqual(await response.read(), b"frozen-image")
        response = await self.client.get(f"/v1/calibration/xy/{identifier}/unexpected.jpg")
        self.assertEqual(response.status, 404)
        (directory / f"xy-{identifier}-wrist.jpg").symlink_to(directory / f"xy-{identifier}-fixed.jpg")
        response = await self.client.get(f"/v1/calibration/xy/{identifier}/wrist.jpg")
        self.assertEqual(response.status, 404)

    # ---- read side -------------------------------------------------------
    async def test_status_endpoints_work_without_lease_and_are_not_cached(self):
        self.robot.results["calib_workflow_status"] = dict(
            status="ok", definition={"steps": []}, run=None,
            bundle=dict(present=True, load_error=None, calibration_created_at="2026-11-09T01:00:00Z", active_version="v1", verification=dict(passed=True, max_error_m=.001)),
            staleness=dict(status="ok"))
        response = await self.client.get("/v1/calibration/state")
        body = await response.json()
        self.assertEqual((response.status, response.headers["Cache-Control"]), (200, "no-store"))
        self.assertTrue(body["available"])
        self.assertEqual(body["gateway"]["blockers"], dict(idle=[], motion=[]))
        self.assertEqual(self.calls(), [("calib_workflow_status", {})])
        body = await (await self.client.get("/v1/calibration/summary")).json()
        self.assertEqual((body["status"], body["calibrated_at"], body["active_version"]), ("valid", "2026-11-09T01:00:00Z", "v1"))
        self.assertEqual(self.calls()[-1], ("calib_workflow_status", {"summary": True}))
        self.robot.results["calib_step_status"] = dict(status="ok", step="S2", state=dict(status="running"))
        body = await (await self.client.get("/v1/calibration/steps/S2")).json()
        self.assertEqual((body["step"], self.calls()[-1]), ("S2", ("calib_step_status", {"step": "S2"})))
        self.assertEqual((await self.client.get("/v1/calibration/steps/S9")).status, 404)

    async def test_state_reports_why_calls_are_blocked_and_survives_worker_errors(self):
        self.workbench.session.state = "running"
        self.workbench.operator_gate["torque"] = "off"
        self.robot.results["calib_workflow_status"] = RuntimeError("worker_gone")
        body = await (await self.client.get("/v1/calibration/state")).json()
        self.assertEqual((body["available"], body["reason"]), (False, "worker_gone"))
        self.assertEqual(body["gateway"]["blockers"], dict(idle=["codex_turn_running"], motion=["codex_turn_running", "torque_off"]))
        body = await (await self.client.get("/v1/calibration/summary")).json()
        self.assertEqual(body["status"], "unknown")

    async def test_unsupported_backend_reports_unavailable(self):
        self.robot.operator = None
        body = await (await self.client.get("/v1/calibration/state")).json()
        self.assertEqual((body["available"], body["reason"]), (False, "operator_command_unsupported:calib_workflow_status"))
        lease = await self.claim()
        status, body = await self.post("begin", mode="first_run", **lease)
        self.assertEqual((status, body["error"]), (400, "operator_command_unsupported:calibration"))

    async def test_preview_images_are_whitelisted_and_never_follow_symlinks(self):
        directory = self.workbench.telemetry_directory()
        directory.mkdir(parents=True)
        (directory / gateway_calibration.PREVIEW_FILES["fixed"]).write_bytes(b"\xff\xd8fixed")
        response = await self.client.get("/v1/calibration/preview/fixed.jpg")
        self.assertEqual((response.status, response.content_type, await response.read()), (200, "image/jpeg", b"\xff\xd8fixed"))
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertEqual((await self.client.get("/v1/calibration/preview/wrist.jpg")).status, 404)
        self.assertEqual((await self.client.get("/v1/calibration/preview/state.jpg")).status, 404)
        (directory / "outside.jpg").write_bytes(b"x")
        (directory / gateway_calibration.PREVIEW_FILES["wrist"]).symlink_to(directory / "outside.jpg")
        self.assertEqual((await self.client.get("/v1/calibration/preview/wrist.jpg")).status, 404)

    async def test_failed_active_verification_is_not_valid_in_summary(self):
        verdict = dict(passed=False, max_error_m=.005045)
        self.robot.results["calib_workflow_status"] = dict(bundle=dict(
            present=True, active_version="active", versions=[
                dict(name="other", verification=dict(passed=True)), dict(name="active", verification=verdict)]))
        body = await (await self.client.get("/v1/calibration/summary")).json()
        self.assertEqual(body["status"], "verification_failed")
        self.assertEqual(body["verification"], verdict)

    # ---- request validation ---------------------------------------------
    async def test_http_validation_errors_never_reach_the_worker(self):
        lease = await self.claim()
        status, body = await self.post("begin", mode="recalibrate", changes=[], **lease)
        self.assertEqual((status, body["error"]), (400, "calibration_recalibrate_requires_changes"))
        status, body = await self.post("steps/S9/start", **lease)
        self.assertEqual((status, body["error"]), (400, "calibration_step_unknown:S9"))
        status, body = await self.post("rollback", version="../../etc", **lease)
        self.assertEqual((status, body["error"]), (400, "invalid_calibration_version"))
        response = await self.client.post("/v1/calibration/begin", json=[1], headers=self.origin())
        self.assertEqual(response.status, 400)
        response = await self.client.post("/v1/calibration/begin", json=dict(mode="first_run"), headers=self.origin())  # no request_id
        self.assertEqual((response.status, (await response.json())["error"]), (400, "control_requires_request_id"))
        self.assertEqual(self.calls(), [])

    # ---- lease -----------------------------------------------------------
    async def test_mutating_calls_need_the_lease(self):
        status, body = await self.post("begin", mode="first_run")  # browser origin, nobody holds control
        self.assertEqual((status, body["error"]), (400, "input_owner_claim_required"))
        lease = await self.claim()
        for path, fields in (("begin", dict(mode="first_run")), ("steps/S0/start", {}), ("apply", {}), ("rollback", {})):
            status, body = await self.post(path, client_id="other-page", writer_token="wrong", **fields)
            self.assertEqual((status, body["error"]), (409, "input_owner_conflict"), path)
        self.assertEqual(self.calls(), [])
        status, body = await self.post("begin", **lease, mode="first_run")
        self.assertEqual((status, body["status"]), (200, "ok"))
        self.assertEqual(self.calls(), [("calib_workflow_begin", dict(mode="first_run", changes=[]))])

    async def test_cancel_needs_no_lease_and_works_while_a_step_holds_the_control_lock(self):
        lease = await self.claim()
        self.robot.entered = threading.Event()
        self.robot.block = threading.Event()
        task = asyncio.create_task(self.post("steps/S2/start", **lease, params=dict(confirmed=["board_matches"])))
        self.assertTrue(await asyncio.to_thread(self.robot.entered.wait, 5))
        self.assertFalse(self.workbench.harness.accepting)
        self.assertEqual(self.workbench.calibration_workflow_state["running_step"], "S2")
        # Status and cancel are answered while the step runs.
        self.assertEqual((await self.client.get("/v1/calibration/state")).status, 200)
        status, body = await asyncio.wait_for(self.post("cancel"), 2)
        self.assertEqual((status, body["status"], body["admission"]), (200, "ok", "closed"))
        self.robot.block.set()
        status, body = await task
        # The cancel was a stop: the step's own completion must not reopen admission.
        self.assertEqual((body["step"], body["admission"]), ("S2", "closed_until_resume"))
        self.assertFalse(self.workbench.harness.accepting)
        self.assertEqual([call[0] for call in self.calls() if call[0] != "calib_workflow_status"], ["calib_step_cancel", "calib_step_start"])

    async def test_lease_commands_are_answered_while_a_step_holds_the_control_lock(self):
        # Regression (real arm, S2 on 2026-11-15): the step's POST held the control lock for its whole 74 s, the
        # lease (30 s) could not be renewed meanwhile, and the page's claim after the lapse queued behind the step
        # until it ended. The page awaited that claim in its poll loop, so progress and previews froze mid-step.
        lease = await self.claim()
        self.robot.entered = threading.Event()
        self.robot.block = threading.Event()
        task = asyncio.create_task(self.post("steps/S2/start", **lease, params=dict(confirmed=["board_matches"])))
        try:
            self.assertTrue(await asyncio.to_thread(self.robot.entered.wait, 5))
            self.assertEqual(self.workbench.calibration_workflow_state["running_step"], "S2")
            self.assertTrue(self.workbench.control_lock.locked())
            before = self.workbench.controller["expires_at"]
            status, body = await asyncio.wait_for(self.control("renew_input", **lease), 2)
            self.assertEqual((status, body["status"]), (200, "owned"))
            self.assertGreaterEqual(body["expires_at"], before)
            self.assertTrue((await (await self.client.get("/v1/run")).json())["control_busy"])
            # Another window cannot take the live lease during the step.
            status, body = await asyncio.wait_for(self.control("claim_input", client_id="other-page"), 2)
            self.assertEqual((status, body["error"]), (409, "input_owner_conflict"))
            # Nor after the holder released it mid-step (button, or pagehide on reload: the reloaded page has a new
            # client_id); the holder may claim it again.
            status, body = await asyncio.wait_for(self.control("release_input", **lease), 2)
            self.assertEqual((status, body["status"]), (200, "released"))
            for client in ("other-page", "reloaded-page"):
                status, body = await asyncio.wait_for(self.control("claim_input", client_id=client), 2)
                self.assertEqual(status, 409, client)
                self.assertTrue(body["error"].startswith("input_owner_busy_step:"), body)
            status, body = await asyncio.wait_for(self.control("claim_input", client_id="calibration-page"), 2)
            self.assertEqual((status, body["status"]), (200, "owned"))
            lease["writer_token"] = body["writer_token"]
            # The owner's lease lapses mid-step (e.g. a throttled background tab): another window is refused at once
            # (the lease never changes hands mid-command), so it cannot relax the arm under the running step either.
            self.workbench.controller["expires_at"] = 0
            self.assertIsNone((await (await self.client.get("/v1/run")).json())["controller"]["owner_client_id"])
            status, body = await asyncio.wait_for(self.control("claim_input", client_id="other-page"), 2)
            self.assertEqual(status, 409)
            self.assertTrue(body["error"].startswith("input_owner_busy_step:"), body)
            self.assertIn("hold, estop and calibration_cancel still work from any window", body["error"])
            status, body = await asyncio.wait_for(self.control("relax", client_id="other-page", confirm="relax"), 2)
            self.assertEqual((status, body["error"]), (400, "input_owner_claim_required"))
            # The owner claims its lapsed lease again at once, not after the step.
            status, body = await asyncio.wait_for(self.control("claim_input", client_id="calibration-page"), 2)
            self.assertEqual((status, body["status"]), (200, "owned"))
            self.assertFalse(task.done())
            self.assertEqual(self.calls("calib_step_start"), [])
        finally:
            self.robot.block.set()
            status, body = await task
        self.assertEqual((status, body["step"], body["status"]), (200, "S2", "ok"))
        self.assertEqual(self.calls("relax"), [])
        self.assertFalse((await (await self.client.get("/v1/run")).json())["control_busy"])
        # After the step, another window takes an expired lease as before.
        self.workbench.controller["expires_at"] = 0
        status, body = await self.control("claim_input", client_id="other-page")
        self.assertEqual((status, body["status"]), (200, "owned"))

    async def test_cancel_of_an_idle_workflow_does_not_close_admission(self):
        status, body = await self.post("cancel")
        self.assertEqual((status, body["status"]), (200, "ok"))
        self.assertNotIn("admission", body)
        self.assertTrue(self.workbench.harness.accepting)

    # ---- gating ----------------------------------------------------------
    async def test_refused_while_turn_action_or_program_runs(self):
        lease = await self.claim()
        for label, arm, disarm, code in (
                ("turn", lambda: setattr(self.workbench.session, "state", "running"),
                 lambda: setattr(self.workbench.session, "state", "completed"), "codex_turn_running"),
                ("action", lambda: setattr(self.workbench.harness, "active_action", "act-9"),
                 lambda: setattr(self.workbench.harness, "active_action", None), "action_in_progress"),
                ("program", lambda: setattr(self.workbench, "has_active_programs", lambda: True),
                 lambda: delattr(self.workbench, "has_active_programs"), "program_active")):
            arm()
            for path, fields in (("begin", dict(mode="first_run")), ("steps/S0/start", {}), ("steps/S1/start", {}),
                                 ("apply", {}), ("rollback", {})):
                status, body = await self.post(path, **lease, **fields)
                self.assertEqual((status, body["error"]), (400, "calibration_blocked:" + code), (label, path))
            disarm()
        self.assertEqual(self.calls(), [])
        self.assertTrue(self.workbench.harness.accepting)

    async def test_motion_steps_also_need_torque_on_and_acknowledged_unknowns(self):
        lease = await self.claim()
        self.workbench.operator_gate["torque"] = "off"
        status, body = await self.post("steps/S1/start", **lease)
        self.assertEqual(body["error"], "calibration_blocked:torque_off")
        status, body = await self.post("steps/S0/start", **lease)  # read-only step: no torque needed
        self.assertEqual(status, 200)
        self.workbench.operator_gate["torque"] = "on"
        self.workbench.operator_gate["unknown"] = [dict(id="operator:old", source="operator_go_home", reason="x")]
        status, body = await self.post("steps/S4/start", **lease)
        self.assertEqual(body["error"], "calibration_blocked:unknown_requires_acknowledgement_via_resume")
        self.assertEqual([call[0] for call in self.calls()], ["calib_step_start"])  # only the S0 above

    async def test_a_step_the_gateway_is_awaiting_blocks_every_other_call(self):
        lease = await self.claim()
        self.workbench.calibration_workflow_state["running_step"] = "S2"
        status, body = await self.post("apply", **lease)
        self.assertEqual(body["error"], "calibration_blocked:calibration_step_running")
        self.assertEqual(self.calls(), [])

    # ---- route -> worker op, admission ----------------------------------
    async def test_route_to_worker_op_mapping(self):
        lease = await self.claim()
        board = dict(squares_x=7, squares_y=5, square_m=0.03, marker_m=0.022, dictionary="DICT_5X5_100", thickness_m=0.002)
        await self.post("begin", **lease, mode="recalibrate", changes=["fixed_moved", "wrist_bumped"], board=board)
        await self.post("steps/S2/start", **lease, params=dict(confirmed=["board_matches", "workspace_clear"], board=board))
        await self.post("steps/S3/start", **lease, params=dict(action="solve"))
        await self.post("apply", **lease)
        await self.post("rollback", **lease, version="20261109T031500Z")
        await self.post("rollback", **lease)
        self.assertEqual(self.calls(), [
            ("calib_workflow_begin", dict(mode="recalibrate", changes=["fixed_moved", "wrist_bumped"], board=board)),
            ("calib_step_start", dict(step="S2", params=dict(confirmed=["board_matches", "workspace_clear"], board=board))),
            ("calib_step_start", dict(step="S3", params=dict(action="solve"))),
            ("calib_apply", {}),
            ("calib_rollback", dict(version="20261109T031500Z")),
            ("calib_rollback", {})])

    async def test_motion_step_closes_admission_then_reopens_with_fresh_observation(self):
        lease = await self.claim()
        self.robot.results["calib_step_start"] = dict(status="passed", physics_advanced=True)
        self.robot.entered = threading.Event()
        self.robot.block = threading.Event()
        task = asyncio.create_task(self.post("steps/S1/start", **lease))
        self.assertTrue(await asyncio.to_thread(self.robot.entered.wait, 5))
        self.assertFalse(self.workbench.harness.accepting)
        self.robot.block.set()
        status, body = await task
        self.assertEqual((status, body["status"], body["step"], body["admission"], body["observation_id"], body["world"]),
                         (200, "passed", "S1", "open", "obs-fresh-1", "objects_not_reset"))
        self.assertTrue(self.workbench.harness.accepting)
        self.assertIsNone(self.workbench.calibration_workflow_state["running_step"])

    async def test_non_motion_steps_leave_admission_alone(self):
        lease = await self.claim()
        for path in ("steps/S0/start", "steps/S3/start", "steps/S5/start"):
            status, body = await self.post(path, **lease)
            self.assertEqual((status, "admission" in body), (200, False), path)
        self.assertTrue(self.workbench.harness.accepting)
        self.assertEqual(self.observations, 0)

    async def test_apply_and_rollback_reopen_with_a_fresh_observation_only_when_open_before(self):
        lease = await self.claim()
        self.robot.results["calib_apply"] = dict(status="applied", version="v2")
        status, body = await self.post("apply", **lease)
        self.assertEqual((body["status"], body["admission"], body["observation_id"]), ("applied", "open", "obs-fresh-1"))
        self.workbench.harness.accepting = False  # stopped before: rolling back never reopens it
        self.robot.results["calib_rollback"] = dict(status="rolled_back", version="v1")
        status, body = await self.post("rollback", **lease)
        self.assertEqual((body["status"], body["admission"], body["observation_id"]), ("rolled_back", "closed_until_resume", None))
        self.assertFalse(self.workbench.harness.accepting)

    async def test_apply_and_rollback_reload_deployment_without_mutating_support_height(self):
        lease = await self.claim()
        self.robot.reloads = 0
        self.robot.reload_deployment = lambda: setattr(self.robot, "reloads", self.robot.reloads + 1)
        self.robot.results["calib_apply"] = dict(status="applied", version="v2", calibration=dict(table_z_m=-0.012))
        self.robot.results["calib_rollback"] = dict(status="rolled_back", version="v1", table_z_m=-0.025)
        for model in ("so101", "arx-x5"):
            for height in (-0.02, 0.765, None):
                with self.subTest(model=model, height=height):
                    self.workbench.config = self.workbench.config.with_overrides({
                        "backend": {"robot_model_id": model}, "features": {"vggt_support_z_m": height}})
                    before = self.workbench.config.to_dict()
                    for endpoint, expected_status in (("apply", "applied"), ("rollback", "rolled_back")):
                        reloads = self.robot.reloads
                        status, body = await self.post(endpoint, **lease)
                        self.assertEqual((status, body["status"]), (200, expected_status))
                        self.assertEqual(self.robot.reloads, reloads + 1)
                        self.assertEqual(self.workbench.config.to_dict(), before)

    async def test_real_runner_apply_and_rollback_supply_installed_table_to_gateway(self):
        from support.so101_gateway_fixtures import HAVE_ARUCO, make_workflow_adapter, start, THICKNESS
        if not HAVE_ARUCO:
            self.skipTest("OpenCV with aruco CharucoDetector required")
        directory = Path(self.temporary_dir) / "real-runner"
        directory.mkdir()
        adapter, _, _, _, camera_file = make_workflow_adapter(directory)
        try:
            adapter.operator("calib_workflow_begin", mode="first_run",
                             board=dict(adapter.workflow.default_board(), thickness_m=THICKNESS))
            for step in ("S0", "S1", "S2", "S4", "S5"):
                result = await asyncio.to_thread(start, adapter, step)
                self.assertEqual(result["status"], "passed", result)
            self.robot.operator = adapter.operator
            self.workbench.config = self.workbench.config.with_overrides({
                "backend": {"robot_model_id": "so101"}, "features": {"vggt_support_z_m": -0.02}})
            lease = await self.claim()
            for endpoint, expected_status in (("apply", "applied"), ("rollback", "rolled_back")):
                status, body = await self.post(endpoint, **lease)
                self.assertEqual((status, body["status"]), (200, expected_status), body)
                installed = json.loads(camera_file.read_text())["joint_calibration"]["table_z_m"]
                receipt = body["operator"]
                # The actual camera reload identity has no table height; the typed install receipt must.
                self.assertNotIn("table_z_m", receipt["calibration"])
                self.assertEqual(receipt["table_z_m"], installed)
                self.assertEqual((receipt["table_z_unit"], receipt["table_z_frame"]), ("m", "so101_base"))
                self.assertEqual(self.workbench.config.features.vggt_support_z_m, -0.02)
        finally:
            adapter.close()

    async def test_rejected_and_failed_steps_are_reported_verbatim(self):
        lease = await self.claim()
        self.robot.results["calib_step_start"] = dict(status="rejected", reason="readiness_unconfirmed:table_clear")
        status, body = await self.post("steps/S1/start", **lease)
        self.assertEqual((status, body["status"], body["operator"]["reason"]), (200, "rejected", "readiness_unconfirmed:table_clear"))
        self.robot.results["calib_step_start"] = dict(status="failed", reason="接触点数不足", result=dict(verdict=dict(passed=False)))
        status, body = await self.post("steps/S1/start", **lease)
        self.assertEqual((body["status"], body["operator"]["result"]["verdict"]["passed"]), ("failed", False))

    async def test_unconfirmed_step_becomes_unknown_and_blocks_until_resume(self):
        lease = await self.claim()
        self.robot.results["calib_step_start"] = dict(status="unknown", reason="stop_unconfirmed", stop_confirmation=dict(confirmed=False),
                                                      physics_advanced=True)
        status, body = await self.post("steps/S2/start", request_id="cal-unknown", **lease, params=dict(confirmed=["board_matches"]))
        self.assertEqual((body["status"], body["admission"]), ("unknown", "closed_until_resume"))
        self.assertFalse(self.workbench.harness.accepting)
        state = await (await self.client.get("/v1/console/state")).json()
        self.assertIn("operator:cal-unknown", state["resume"]["requires_acknowledgement"])
        status, body = await self.post("steps/S4/start", **lease)
        self.assertEqual(body["error"], "calibration_blocked:unknown_requires_acknowledgement_via_resume")

    async def test_worker_error_is_unknown_and_not_resent_on_retry(self):
        lease = await self.claim()
        self.robot.results["calib_step_start"] = TimeoutError("worker_timeout")
        status, body = await self.post("steps/S1/start", request_id="cal-timeout", **lease)
        self.assertEqual((status, body["status"]), (409, "unknown"))
        self.assertEqual(self.workbench.calibration_workflow_state["last"]["status"], "unknown")
        status, body = await self.post("steps/S1/start", request_id="cal-timeout", **lease)  # same request_id: saved result
        self.assertEqual((status, body["status"]), (409, "unknown"))
        self.assertEqual(len(self.calls("calib_step_start")), 1)

    async def test_page_is_served_when_built(self):
        response = await self.client.get("/calibration")
        self.assertIn(response.status, (200, 404))
        if response.status == 404:
            self.assertIn("calibration_page_not_built", await response.text())

    async def test_external_motion_emitted_only_for_motion_or_install_and_refreshes_latest_on_closed_rollback(self):
        lease = await self.claim()
        harness = self.workbench.harness

        # Non-motion read-only calls emit zero external_motion events and zero observations
        await self.post("begin", mode="first_run", **lease)
        for ro_path, extra in (
            ("steps/S0/start", {}),
            ("steps/S3/start", {}),
            ("steps/S5/start", {}),
            ("steps/S6/start", dict(params={"action": "xy_capture"})),
        ):
            status, _ = await self.post(ro_path, **lease, **extra)
            self.assertEqual(status, 200, ro_path)
        self.assertEqual(self.observations, 0)
        self.assertEqual([e for e in self.workbench.store.events(0) if e["type"] == "harness.external_motion"], [])

        # Motion step S1 when accepting=True: external_motion clears ephemerals, reopen_admission observes once
        harness._persistent_obstacles["o1"] = {"id": "o1"}
        self.robot.results["calib_step_start"] = dict(status="passed", physics_advanced=True)
        status, body = await self.post("steps/S1/start", **lease)
        self.assertEqual((status, body["admission"], body["observation_id"]), (200, "open", "obs-fresh-1"))
        self.assertEqual(harness._persistent_obstacles, {})
        self.assertEqual(self.observations, 1)

        # Rollback when was_accepting=False: refreshes harness.latest internally while HTTP observation_id stays None
        harness.accepting = False
        harness.latest = {"observation_id": "stale-before-rollback"}
        self.robot.results["calib_rollback"] = dict(status="rolled_back", version="v1")
        status, body = await self.post("rollback", **lease)
        self.assertEqual((status, body["admission"], body["observation_id"]), (200, "closed_until_resume", None))
        self.assertEqual(harness.latest, {"observation_id": "obs-fresh-2"})
        ext_events = [e["payload"] for e in self.workbench.store.events(0) if e["type"] == "harness.external_motion"]
        self.assertEqual(
            [(p["phase"], p["source"]) for p in ext_events],
            [
                ("begin", "calibration_step:S1"),
                ("end", "calibration_step:S1"),
                ("begin", "calibration_rollback"),
                ("end", "calibration_rollback"),
            ],
        )


class CalibrationUnitChecks(unittest.TestCase):
    def test_preview_names_match_the_files_the_worker_writes(self):
        self.assertEqual({"wrist": gateway_calibration.PREVIEW_FILES["wrist"], "fixed": gateway_calibration.PREVIEW_FILES["fixed"]},
                         dict(calibration_runner.OVERLAY_FILES))
        self.assertEqual(gateway_calibration.PREVIEW_FILES["fixed_corners"], calibration_capture.HANDHELD_PREVIEW_FILE)

    def test_badge_classification(self):
        valid = dict(bundle=dict(present=True, load_error=None, calibration_created_at="2026-11-09T01:00:00Z", active_version="v1", verification=dict(passed=True, max_error_m=.001)),
                     staleness=dict(status="ok", rms_px=0.4))
        self.assertEqual(calibration_badge(valid)["status"], "valid")
        self.assertEqual(calibration_badge(dict(valid, staleness=dict(status="shifted", rms_px=9)))["status"], "maybe_stale")
        self.assertEqual(calibration_badge(dict(valid, staleness=dict(status="shifted", rms_px=9)))["reason"], "board_shift_exceeds_threshold")
        badge_div = calibration_badge(dict(valid, bundle=dict(valid["bundle"], matches_active_version=False)))
        self.assertEqual((badge_div["status"], badge_div["reason"]), ("maybe_stale", "file_diverged_from_active_version"))
        badge_joint = calibration_badge(dict(valid, bundle=dict(valid["bundle"], has_joint_calibration=False)))
        self.assertEqual((badge_joint["status"], badge_joint["reason"]), ("maybe_stale", "missing_joint_calibration"))
        badge_intr = calibration_badge(dict(valid, bundle=dict(valid["bundle"], fixed_intrinsics_source="copied_from_wrist_same_model_unverified")))
        self.assertEqual((badge_intr["status"], badge_intr["reason"]), ("maybe_stale", "fixed_intrinsics_unverified"))
        # An unavailable hint (arm moving, board hidden) never marks a calibration stale.
        self.assertEqual(calibration_badge(dict(valid, staleness=dict(status="board_not_visible")))["status"], "valid")
        self.assertEqual(calibration_badge(dict(valid, bundle=dict(present=False)))["status"], "needed")
        self.assertEqual(calibration_badge(dict(valid, bundle=dict(present=True, load_error="bad file")))["status"], "needed")
        self.assertEqual(calibration_badge(dict(available=False, reason="x"))["status"], "unknown")
        self.assertEqual(calibration_badge(None)["status"], "unknown")

    def test_legacy_binding_diagnostics_do_not_grant_calibration_validity(self):
        diagnostics = {"offsets_exact": False, "joint_mapping_recorded": False,
                       "matches_legacy_six_decimal_serialization": True,
                       "max_abs_offset_difference_rad": 4.54e-7,
                       "recovery": "new_acquisition_solve_apply"}
        badge = calibration_badge({"bundle": {"present": True, "matches_active_version": False,
                                  "binding_error": "verification_calibration_mismatch:joint_offsets",
                                  "binding_diagnostics": diagnostics, "verification": {"passed": True}}})
        self.assertEqual(badge["status"], "maybe_stale")
        self.assertEqual(badge["binding_diagnostics"], diagnostics)
        self.assertTrue(badge["verification"]["passed"])  # preserved historical verdict, never current validity

    def test_verification_is_bound_to_active_version_and_missing_is_unknown(self):
        bundle = dict(present=True, active_version="active", versions=[
            dict(name="other", verification=dict(passed=False)), dict(name="active", verification=dict(passed=True))])
        self.assertEqual(calibration_badge(dict(bundle=bundle))["status"], "valid")
        for changes in (dict(active_version="missing"), dict(verification=None), dict(verification=dict(passed=None))):
            self.assertEqual(calibration_badge(dict(bundle=dict(bundle, **changes)))["status"], "unverified")
        self.assertEqual(calibration_badge(dict(bundle=dict(bundle, verification=dict(passed=False)),
                                               staleness=dict(status="shifted")))["status"], "verification_failed")

    def test_begin_validation(self):
        self.assertEqual(validate_begin(dict(mode="first_run")), dict(mode="first_run", changes=[]))
        self.assertEqual(validate_begin(dict(mode="recalibrate", changes=["fixed_moved"], board=dict(thickness_m=0.003)))["board"],
                         dict(thickness_m=0.003))
        for bad, message in ((dict(mode="all"), "invalid_calibration_mode"),
                             (dict(mode="first_run", changes=["fixed_moved"]), "calibration_first_run_takes_no_changes"),
                             (dict(mode="recalibrate"), "calibration_recalibrate_requires_changes"),
                             (dict(mode="recalibrate", changes=["nope"]), "calibration_change_unknown:nope"),
                             (dict(mode="recalibrate", changes="fixed_moved"), "invalid_calibration_changes"),
                             (dict(mode="first_run", board=[1]), "invalid_calibration_board")):
            with self.assertRaisesRegex(ValueError, message):
                validate_begin(bad)

    def test_step_start_validation(self):
        self.assertEqual(validate_step_start(dict(step="S1")), dict(step="S1", params={}))
        self.assertEqual(validate_step_start(dict(step="S2", params=dict(confirmed=["board_matches"], board={"square_m": 0.03}))),
                         dict(step="S2", params=dict(confirmed=["board_matches"], board={"square_m": 0.03})))
        self.assertEqual(validate_step_start(dict(step="S3", params=dict(action="solve")))["params"], dict(action="solve"))
        for bad, message in ((dict(step="S9"), "calibration_step_unknown:S9"),
                             (dict(step="S1", params=dict(speed=1)), "invalid_step_params"),
                             (dict(step="S1", params=dict(confirmed=["torque_on"])), "invalid_step_confirmed"),
                             (dict(step="S1", params=dict(confirmed="table_clear")), "invalid_step_confirmed"),
                             (dict(step="S4", params=dict(board={})), "step_board_only_for_S2"),
                             (dict(step="S3", params=dict(action="pause")), "invalid_step_action"),
                             (dict(step="S2", params=dict(action="capture")), "invalid_step_action")):
            with self.assertRaisesRegex(ValueError, message):
                validate_step_start(bad)


if __name__ == "__main__":
    unittest.main()
