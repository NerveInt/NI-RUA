"""Run startup rendered from the embodiment profile (design 13 work package WP3: G1, G2, G3, G7, H4, H5).

A real ``Workbench`` with ``FakeAgentSession`` (no binary, model, network or robot) over the WP0 fixture robots:
the ARX-X5 RoboDojo parent backend on a stub world and the real SO-101 parent backend over ``FakeArmDriver``.
Checked: handoff policy only for an embodiment with a cross-arm handoff, the real-time clock and TCP wording, the
default first prompt, truthful manifest / summary labels (``data_origin``, ``clock_policy``, ``runtime_protocol``,
worker guards, ``embodiment_record_id``), replay naming the archived profile record and the resume guard for any
backend's worker on any POSIX host.
"""

NI_RUA_TEST_TIER = "component"

# === tests/test_embodiment_prompt.py ===

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import unittest

from support import embodiment_golden as golden
from robot.embodiment.test_embodiment_profile import Session, TemporaryRoot, so101_robot, x5_robot

from ni_rua.apps.gateway import workbench as workbench_module
from ni_rua.apps.gateway.workbench import (CLOCK_POLICIES, DEFAULT_PROMPT, Workbench, archived_embodiment_record,
                                           default_prompt, owned_robot_worker_running)
from ni_rua.core.storage.store import Store
from ni_rua.cortex.context import actor_instructions
from ni_rua.cortex.context.task_notes import embodiment_handoff_policy, handoff_policy
from ni_rua.robot.model import embodiments as e


def start_workbench(run, robot, config, *, prompt=DEFAULT_PROMPT, resume=False):
    """Start and close one real Workbench; returns the manifest, the session inputs, the configuration records and
    the live harness profile / record id."""
    factory = Session()
    options = dict(resume=True) if resume else {}
    with mock.patch.object(workbench_module, "create_session", side_effect=factory):
        workbench = Workbench(run, robot=robot, visual_review_model=None, config=config, **options)

    async def go():
        try:
            await workbench.start(prompt, benchmark=False)
            return dict(record_id=workbench.harness.embodiment_record_id, embodiment=workbench.harness.embodiment,
                        record=workbench.embodiment_record(), identity=dict(workbench.identity),
                        records=workbench.store.record_items("configuration"))
        finally:
            await workbench.close()
    result = asyncio.run(go())
    result.update(inputs=factory.inputs, manifest=json.loads((Path(run) / "manifest.json").read_text()))
    return result


def task_prompt_record(result):
    records = [value for name, value in result["records"].items() if name.startswith("task_prompt-")]
    assert len(records) == 1, records
    return records[0]


class HandoffGateTests(unittest.TestCase):
    def test_policy_exists_only_for_an_embodiment_with_a_cross_arm_handoff(self):
        single = SimpleNamespace(handoff=False)
        for airborne, preference in ((False, None), (True, "surface"), (True, "airborne")):
            with self.subTest(airborne=airborne, preference=preference):
                self.assertIsNone(embodiment_handoff_policy(single, airborne, handoff_preference=preference))
                self.assertEqual(embodiment_handoff_policy(e.ARX_X5_PROFILE, airborne, handoff_preference=preference),
                                 handoff_policy(airborne, handoff_preference=preference))
        # The switches are validated on every embodiment.
        for profile in (single, e.ARX_X5_PROFILE):
            with self.assertRaises(ValueError):
                embodiment_handoff_policy(profile, False, handoff_preference="airborne")
            with self.assertRaises(ValueError):
                embodiment_handoff_policy(profile, 1)


class DefaultPromptTests(unittest.TestCase):
    def test_x5_rendering_is_the_published_constant(self):
        self.assertEqual(default_prompt(), DEFAULT_PROMPT)
        self.assertEqual(default_prompt(e.ARX_X5_PROFILE), DEFAULT_PROMPT)
        self.assertIn("not completion of the RoboDojo task", DEFAULT_PROMPT)

    def test_so101_prompt_names_its_one_arm_and_no_simulator(self):
        from ni_rua.robot.backends.so101.deployment import embodiment_profile, load_deployment
        text = default_prompt(embodiment_profile(load_deployment(golden.REPOSITORY / "configs" / "so101.example.json")))
        for phrase in ("RoboDojo", "simulat", "choose one arm", "evaluation state"):
            self.assertNotIn(phrase, text)
        self.assertIn('move the TCP of the only arm "right"', text)
        self.assertIn("real hardware", text)
        self.assertIn("at most 10 mm", text)


class StartupManifestTests(TemporaryRoot):
    def test_so101_run_is_real_hardware_in_real_time_without_handoff(self):
        robot, config = so101_robot(self.root)
        self.assertTrue(config.features.joint_replan)  # configured, but no SO-101 worker applies it
        result = start_workbench(self.root / "run", robot, config)
        manifest, profile = result["manifest"], result["embodiment"]
        self.assertEqual(profile.robot_model_id, "so101")
        self.assertEqual(manifest["data_origin"], "so101_real_hardware")
        self.assertEqual(manifest["clock_policy"], CLOCK_POLICIES["real_time"])
        self.assertEqual(manifest["runtime_protocol"], "so101-deployment-v1")
        self.assertFalse(manifest["tracking_guard"]["enabled"])
        self.assertFalse(manifest["joint_replan"]["enabled"])
        self.assertIsNone(manifest["handoff_policy"])
        self.assertIsNone(task_prompt_record(result)["handoff_policy"])
        self.assertEqual(manifest["embodiment_record_id"], result["record_id"])
        self.assertTrue(result["record_id"].startswith(e.EMBODIMENT_RECORD_PREFIX))
        self.assertEqual(result["record"]["profile"]["robot"]["robot_model_id"], "so101")
        # First turn: the SO-101 default prompt, also archived as the run prompt.
        prompt = result["inputs"]["prompt"]
        self.assertEqual(prompt, default_prompt(profile))
        self.assertEqual(task_prompt_record(result)["prompt"], prompt)
        self.assertEqual(manifest["effective_config"]["run"]["prompt"], prompt)
        instructions = result["inputs"]["instructions"]
        harness_part = instructions[instructions.index("Harness instructions\n"):instructions.index("## 3D Perception")]
        self.assertIn(actor_instructions.WORLD_CLOCK_SENTENCES["real_time"], harness_part)
        for phrase in ("## Run handoff policy", "simulation", "native-step", "To move both arms together",
                       "compute the required grasp_position_m", "link6"):
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, harness_part)
        # Audit labels stay out of model context.
        self.assertNotIn(result["record_id"], instructions + prompt)
        self.assertNotIn(CLOCK_POLICIES["real_time"], instructions + prompt)

    def test_explicit_prompt_is_kept_verbatim(self):
        robot, config = so101_robot(self.root)
        result = start_workbench(self.root / "run", robot, config, prompt="Inspect the scene only.")
        self.assertEqual(result["inputs"]["prompt"], "Inspect the scene only.")

    def test_x5_run_keeps_its_labels_and_gains_only_the_record_id(self):
        robot, config = x5_robot(self.root)
        result = start_workbench(self.root / "run", robot, config)
        manifest = result["manifest"]
        self.assertEqual(manifest["data_origin"], "robodojo_runtime")
        self.assertEqual(manifest["clock_policy"], "paused_between_gateway_operations")
        self.assertEqual(manifest["runtime_protocol"], "robodojo-native-v1")
        self.assertEqual(manifest["tracking_guard"]["enabled"], config.features.tracking_guard)
        self.assertEqual(manifest["joint_replan"]["enabled"], config.features.joint_replan)
        expected = handoff_policy(config.features.airborne_handoff,
                                  handoff_preference=config.features.handoff_preference)
        self.assertEqual(manifest["handoff_policy"], expected)
        self.assertEqual(task_prompt_record(result)["handoff_policy"], expected)
        self.assertEqual(manifest["embodiment_record_id"], result["record_id"])
        self.assertEqual(result["inputs"]["prompt"], DEFAULT_PROMPT)
        self.assertEqual(result["inputs"]["instructions"].count("\n## Run handoff policy\n" + expected["text"]), 1)

    def test_episode_summary_names_the_embodiment_protocol(self):
        from ni_rua.apps.evaluation.episode import episode_summary
        from ni_rua.robot.backends.so101.deployment import embodiment_profile, load_deployment
        so101 = embodiment_profile(load_deployment(golden.REPOSITORY / "configs" / "so101.example.json"))
        store = Store(self.root / "store", dict(golden.IDENTITY))
        self.addCleanup(store.close)
        session = SimpleNamespace(thread_id="thread", turn_result=None, model="m", provider="p", price_card=None)
        for harness, protocol in ((SimpleNamespace(action_count=0, embodiment=so101), "so101-deployment-v1"),
                                  (SimpleNamespace(action_count=0, embodiment=e.ARX_X5_PROFILE), "robodojo-native-v1"),
                                  (SimpleNamespace(action_count=0), "robodojo-native-v1"),
                                  (mock.Mock(action_count=0, actor_continuation_count=0, actor_stall=None,
                                             wait_stop_reason=None), "robodojo-native-v1")):
            with self.subTest(protocol=protocol, harness=type(harness).__name__):
                summary = episode_summary(store, dict(golden.IDENTITY), session, harness, benchmark=False,
                                          final_status="stopped", started_at=0.0)
                self.assertEqual(summary["runtime_protocol"], protocol)


class ReplayRecordTests(TemporaryRoot):
    def replay(self, run):
        with mock.patch.object(workbench_module, "create_session", side_effect=Session()):
            workbench = Workbench(run, robot=SimpleNamespace(close=mock.Mock()), replay=True)

        async def go():
            try:
                await workbench.start(None)
                return (workbench.harness.embodiment, workbench.harness.embodiment_record_id,
                        workbench.embodiment_record())
            finally:
                await workbench.close()
        return asyncio.run(go())

    def test_replay_names_the_archived_record_not_the_live_backend(self):
        robot, config = so101_robot(self.root)
        run = self.root / "run"
        first = start_workbench(run, robot, config)
        live, record_id, record = self.replay(run)
        self.assertEqual(live.robot_model_id, "arx-x5")  # the replay's robot double has no profile hook
        self.assertEqual(record_id, first["record_id"])
        self.assertEqual(record, first["record"])
        self.assertEqual(record["profile"]["robot"]["robot_model_id"], "so101")
        # A manifest written before it named the record: the archived record is still found.
        path = run / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest.pop("embodiment_record_id")
        path.write_text(json.dumps(manifest))
        self.assertEqual(self.replay(run)[1:], (first["record_id"], first["record"]))

    def test_archive_without_a_profile_record_has_none(self):
        store = Store(self.root / "store", dict(golden.IDENTITY))
        self.addCleanup(store.close)
        self.assertEqual(archived_embodiment_record(store), (None, None))
        self.assertEqual(archived_embodiment_record(store, "embodiment_profile-absent"),
                         ("embodiment_profile-absent", None))
        store.save("configuration", "embodiment_profile-a", {"schema": e.PROFILE_RECORD_SCHEMA, "profile": {"n": 1}})
        store.save("configuration", "embodiment_profile-b", {"schema": e.PROFILE_RECORD_SCHEMA, "profile": {"n": 2}})
        self.assertEqual(archived_embodiment_record(store)[0], "embodiment_profile-b")
        self.assertEqual(archived_embodiment_record(store, "embodiment_profile-a")[1]["profile"], {"n": 1})


class ResumeWorkerGuardTests(TemporaryRoot):
    def spawn(self, *arguments):
        """A live stand-in worker whose argv carries ``arguments``. Returns once the child prints from its own
        interpreter: Popen returns at the child's close-on-exec, before Linux has installed the new argv, so
        ``/proc/<pid>/cmdline`` can still read empty right after Popen (seen on a loaded host)."""
        process = subprocess.Popen([sys.executable, "-c", "import time; print('ready', flush=True); time.sleep(60)",
                                    *arguments], stdout=subprocess.PIPE, text=True)

        def stop():
            process.kill()
            process.wait()
            process.stdout.close()
        self.addCleanup(stop)
        self.assertEqual(process.stdout.readline(), "ready\n")
        return process

    def test_any_backend_worker_of_this_run_is_detected_on_this_host(self):
        run_dir = str(self.root / "run" / "private" / "robot")
        worker = self.spawn("-u", "-m", "ni_rua.robot.backends.so101.worker", "--run-dir", run_dir)
        self.assertTrue(owned_robot_worker_running({"pid": worker.pid, "run_dir": run_dir}, "/elsewhere"))
        self.assertTrue(owned_robot_worker_running({"pid": worker.pid}, run_dir))
        # Another run's worker, or a process that only names a file inside this run directory, is not ours.
        other = self.spawn("--run-dir", run_dir + "-other", run_dir + "/robot-worker.log")
        self.assertFalse(owned_robot_worker_running({"pid": other.pid, "run_dir": run_dir}, run_dir))
        exited = subprocess.Popen([sys.executable, "-c", "pass"])
        exited.wait()
        self.assertFalse(owned_robot_worker_running({"pid": exited.pid, "run_dir": run_dir}, run_dir))
        for state in (None, {}, {"pid": None}, {"pid": str(worker.pid)}, {"pid": -1}, {"pid": True}):
            with self.subTest(state=state):
                self.assertFalse(owned_robot_worker_running(state, run_dir))

    def test_resume_waits_for_the_previous_worker_and_releases_the_run(self):
        robot, config = so101_robot(self.root)
        run = self.root / "run"
        first = start_workbench(run, robot, config)
        worker = self.spawn("--run-dir", str(run / "private" / "robot"))
        store = Store(run, first["identity"])
        store.set_state("runtime", "robot", {"pid": worker.pid, "run_dir": str(run / "private" / "robot")})
        store.set_state("runtime", "agent", None)  # the fixture session's thread is not resumable
        store.close()
        robot, _ = so101_robot(self.root / "resumed")
        with mock.patch.object(workbench_module, "create_session", side_effect=Session()):
            with self.assertRaisesRegex(ValueError, "resume_requires_previous_owned_robot_worker_exit"):
                Workbench(run, robot=robot, visual_review_model=None, config=config, resume=True)
        worker.kill()
        worker.wait()
        # The refused attempt released the run lock and the store: the resume proceeds once the worker exited.
        resumed = start_workbench(run, robot, config, resume=True)
        self.assertEqual(resumed["identity"]["env_generation"], first["identity"]["env_generation"] + 1)

# === tests/test_embodiment_cleanup.py ===

import asyncio
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from support import embodiment_golden as golden
from robot.embodiment.test_embodiment_profile import Session, TemporaryRoot, so101_robot

from ni_rua.core.config import Config, ConfigError
from ni_rua.core.presentation.actor_payload import lean_observation, lean_v2_display
from ni_rua.core.storage.store import Store, open_read_store
from ni_rua.perception.external import embodiment_of, object_table
from ni_rua.robot.backends.so101.deployment import embodiment_profile, load_deployment
from ni_rua.robot.model import embodiments as e

SO101 = embodiment_profile(load_deployment(golden.REPOSITORY / "configs" / "so101.example.json"))
X5_ROBODOJO = e.arx_x5_profile(binding=e.get_robodojo_robot())


def observation_reply(arm, *, remaining, limit, frame_id="env_origin", tcp_config_id="arx-x5-link6-v1", observation_id="obs_7"):
    """A robot_observe reply carrying embodiment-specific TCP and clock budget fields."""
    return {
        "bundle": {"observation_id": observation_id, "sim_step": 12, "cameras": [{"camera_id": "fixed"}]},
        "robot_state": {
            "arms": {
                arm: {
                    "ee_pose": {
                        "frame_id": frame_id,
                        "tcp_config_id": tcp_config_id,
                        "position_m": [0.123456, -0.2, 0.31234],
                        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                    "gripper": {"position_m": [0.0213, 0.0215]},
                }
            }
        },
        "current_action_budget": {"native_steps_remaining": remaining, "native_steps_limit": limit},
    }


def history(replies, tool="robot_observe"):
    items = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "task"}]}]
    for index, reply in enumerate(replies):
        call_id = f"call_{index}"
        items += [
            {"type": "function_call", "name": tool, "arguments": "{}", "call_id": call_id},
            {"type": "function_call_output", "call_id": call_id, "output": [{"type": "input_text", "text": json.dumps(reply)}]},
        ]
    return items


def body(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


# ---- G3: embodiment-neutral actor_payload & CodexSession egress scrub ----------------------------------------------
class ActorPayloadEmbodimentNeutralTests(unittest.TestCase):
    def test_actor_payload_preserves_x5_and_so101_tcp_fields_without_hardcoded_names(self):
        x5_obs = observation_reply("left", remaining=900, limit=1100, frame_id=e.ARX_X5.frame_id, tcp_config_id=e.ARX_X5.tcp_config_id)
        so101_obs = observation_reply(
            "right", remaining=None, limit=None, frame_id=SO101.frame_id, tcp_config_id=SO101.tcp_config_id
        )
        for obs, arm, profile in ((x5_obs, "left", e.ARX_X5_PROFILE), (so101_obs, "right", SO101)):
            lean = lean_observation(obs)
            shown = lean_v2_display(lean, tool="robot_observe")
            ee = shown["robot_state"]["arms"][arm]["ee_pose"]
            self.assertEqual(ee["frame_id"], profile.frame_id)
            self.assertEqual(ee["tcp_config_id"], profile.tcp_config_id)
            self.assertEqual(ee["position_m"], [0.123456, -0.2, 0.31234])

    def test_codex_session_preserves_wired_embodiment_and_scrubs_egress_neutrally(self):
        try:
            from ni_rua.cortex.agents.codex.session import CodexSession
        except ImportError as error:
            self.skipTest(str(error))
        temporary = self.enterContext(tempfile.TemporaryDirectory())
        root = Path(temporary)
        (root / "source").mkdir()
        (root / "source" / "config.toml").write_text(
            'model_provider = "fixture"\n[model_providers.fixture]\nbase_url = "https://fixture.invalid/v1"\n'
        )
        store = Store(root / "run", dict(golden.IDENTITY))
        self.addCleanup(store.close)
        config = Config.from_dict({"model": {"codex_home": str(root / "source")}, "features": {"sandbox_enabled": False}})
        session = CodexSession(store.root, store, lambda _: {}, config=config)
        self.addCleanup(session.close)
        self.assertIsNone(session.embodiment)
        session.embodiment = SO101
        self.assertIs(session.embodiment, SO101)
        replies = [
            observation_reply(
                "right",
                remaining=None,
                limit=None,
                frame_id=SO101.frame_id,
                tcp_config_id=SO101.tcp_config_id,
                observation_id="observation-" + "b" * 32,
            )
        ]
        presented = session._present_model_request({"input": history(replies)})
        decoded = json.loads(presented["input"][2]["output"][0]["text"])
        self.assertEqual(decoded["robot_state"]["arms"]["right"]["ee_pose"]["tcp_config_id"], SO101.tcp_config_id)
        self.assertNotIn("link6", body(presented))
        self.assertEqual(decoded["bundle"]["observation_id"], "obs_1")


class WorkbenchWiringTests(TemporaryRoot):
    def test_workbench_wires_the_run_profile_into_the_session(self):
        from ni_rua.apps.gateway import workbench as workbench_module
        from ni_rua.apps.gateway.workbench import Workbench
        robot, config = so101_robot(self.root)
        factory = Session()
        with mock.patch.object(workbench_module, "create_session", side_effect=factory):
            workbench = Workbench(self.root / "run", robot=robot, visual_review_model=None, config=config)
        try:
            session, = factory.created
            self.assertIs(session.embodiment, workbench.harness.embodiment)
            self.assertEqual(session.embodiment.tcp_link, "gripper_frame_link")
        finally:
            asyncio.run(workbench.close())


# ---- B8 / V-D: depth_query full opening ---------------------------------------------------------------------------
def recorded_depth_request():
    """The depth_query request schema of the 5fd93c6 X5 catalogue (golden fixture, definitions/default)."""
    _, items = golden.load_fixture("x5", "catalogue")
    definition, = [item for item in items["definitions/default"] if item["name"] == "depth_query"]
    return definition["inputSchema"]["properties"]["request"]


def leaves(value, path=""):
    if isinstance(value, dict):
        return {key: item for name, child in value.items() if name != "description"
                for key, item in leaves(child, f"{path}/{name}").items()}
    if isinstance(value, list):
        return {key: item for index, child in enumerate(value) for key, item in leaves(child, f"{path}/{index}").items()}
    return {path: value}


def band(width_m, length_m=0.06, step_m=0.002):
    """A visible top-face region: a flat grid ``length_m`` x ``width_m`` (m) at z 0.8 m, long axis along x."""
    xs = np.arange(-length_m / 2, length_m / 2 + 1e-9, step_m)
    ys = np.linspace(-width_m / 2, width_m / 2, 9)
    points = np.array([[x, y, 0.8] for x in xs for y in ys])
    return dict(points=points, major=np.array([1.0, 0.0]), minor=np.array([0.0, 1.0]), center=np.array([0.0, 0.0, 0.8]))


class DepthOpeningDifferentialTests(unittest.TestCase):
    """The only X5 difference of B8: the max_opening_m bound and default 0.088807166 -> 0.088 m."""
    BASELINE_M = 0.088807166

    def setUp(self):
        from ni_rua.perception.depth import depth_geometry
        self.depth = depth_geometry
        baseline = dict(depth_geometry.REQUEST_PARAMETERS["max_opening_m"], maximum=self.BASELINE_M,
                        default=self.BASELINE_M)
        self.baseline = mock.patch.dict(depth_geometry.REQUEST_PARAMETERS, max_opening_m=baseline)

    def test_value_is_the_one_x5_full_opening(self):
        self.assertEqual(self.depth.MAX_OPENING_M, e.ARX_X5_PROFILE.gripper.full_open_aperture_m)
        self.assertEqual(self.depth.MAX_OPENING_M, 0.088)
        self.assertEqual(self.depth.APERTURE_ZERO_M, 0.000807166)    # the measured pad gap reading is unchanged

    def test_published_schema_differs_from_5fd93c6_only_in_the_bound(self):
        from ni_rua.perception.depth.depth_runtime import request_schema
        before, after = leaves(recorded_depth_request()), leaves(request_schema())
        changed = {key: (before.get(key), after.get(key)) for key in before.keys() | after.keys()
                   if before.get(key) != after.get(key)}
        field = "/oneOf/2/properties/max_opening_m/"
        self.assertEqual(changed, {field + "maximum": (self.BASELINE_M, 0.088), field + "default": (self.BASELINE_M, 0.088)})

    def grasp(self, request, region):
        try:
            return self.depth.DepthGeometryTools._grasp(request, region)
        except ValueError as error:
            return str(error)

    def test_grasp_candidates_differ_only_by_the_opening(self):
        # (request, visible width m, expected difference). Clearance default 5 mm per side.
        cases = [({"max_opening_m": 0.07}, 0.05, "identical"), ({"max_opening_m": 0.088}, 0.075, "identical"),
                 ({}, 0.05, "margin"), ({}, 0.0785, "old_only"), ({"max_opening_m": 0.0885}, 0.05, "now_refused")]
        for request, width, expected in cases:
            with self.subTest(request=request, width=width):
                with self.baseline:
                    old = self.grasp(request, band(width))
                new = self.grasp(request, band(width))
                if expected == "identical":                    # an explicit bound inside both ranges
                    self.assertEqual(old, new)
                elif expected == "now_refused":                # 0.0885 m was inside the old range only
                    self.assertTrue(old["candidates"])
                    self.assertEqual(new, "invalid_max_opening_m")
                elif expected == "margin":                     # default bound: only the reported margin moves
                    self.assertTrue(new["candidates"])
                    self.assertEqual(len(old["candidates"]), len(new["candidates"]))
                    for before, after in zip(old["candidates"], new["candidates"]):
                        self.assertAlmostEqual(before["opening_margin_m"] - after["opening_margin_m"],
                                               self.BASELINE_M - 0.088, places=12)
                        self.assertEqual({k: v for k, v in before.items() if k != "opening_margin_m"},
                                         {k: v for k, v in after.items() if k != "opening_margin_m"})
                else:                                          # 78.5 mm + 2 x 5 mm fits only the old 88.8 mm bound
                    self.assertTrue(old["candidates"])
                    self.assertEqual(new["candidates"], [])
                    self.assertEqual(old["assumptions"], new["assumptions"])


# ---- F7: features.vggt_support_z_m from the profile ---------------------------------------------------------------
CALIBRATED = {"joint_calibration": {"table_z_m": -0.018}}


class SupportHeightFromProfileTests(TemporaryRoot):
    def test_default_is_the_profile_and_x5_resolves_to_its_table(self):
        features = Config().features
        self.assertEqual(features.vggt_support_z_m, "profile")
        for profile in (None, e.ARX_X5_PROFILE, X5_ROBODOJO):
            self.assertEqual(object_table.effective_support_z_m(features, CALIBRATED, profile), 0.765)
            self.assertEqual(object_table.effective_support_z_m(features, None, profile), 0.765)

    def test_so101_default_follows_the_observation_calibration(self):
        self.assertIsNone(SO101.support_surface_z_m)
        features = Config().features
        self.assertEqual(object_table.effective_support_z_m(features, CALIBRATED, SO101), -0.018)
        for camera in (None, {}, {"joint_calibration": {"table_z_m": None}}):
            with self.subTest(camera=camera), self.assertRaisesRegex(ValueError, "observation_support_z"):
                object_table.effective_support_z_m(features, camera, SO101)

    def test_explicit_forms_keep_their_meaning_on_both_robots(self):
        for profile in (e.ARX_X5_PROFILE, SO101):
            for value in (0.765, -0.02, 0.0):
                features = Config.from_dict({"features": {"vggt_support_z_m": value}}).features
                self.assertEqual(object_table.effective_support_z_m(features, CALIBRATED, profile), value)
            observation = Config.from_dict({"features": {"vggt_support_z_m": None}}).features
            self.assertEqual(object_table.effective_support_z_m(observation, CALIBRATED, profile), -0.018)

    def test_old_configs_and_archives_keep_the_number(self):
        _, recorded = golden.load_fixture("x5", "config")
        for name in ("Config()", "effective_config/ni_rua_benchmark", "load/ni_rua"):
            archived = recorded[name]
            self.assertEqual(archived["features"]["vggt_support_z_m"], 0.765)
            loaded = Config.from_dict(archived)
            self.assertEqual(loaded.features.vggt_support_z_m, 0.765)
            self.assertEqual(Config.from_dict(loaded.to_dict()).features.vggt_support_z_m, 0.765)
            self.assertEqual(object_table.effective_support_z_m(loaded.features, CALIBRATED, SO101), 0.765)
        path = self.root / "old.json"
        path.write_text(json.dumps({"features": {"vggt_support_z_m": 0.765}}))
        self.assertEqual(golden.load_config(path).features.vggt_support_z_m, 0.765)

    def test_shipped_configurations(self):
        self.assertEqual(golden.load_config().features.vggt_support_z_m, "profile")   # configs/ni-rua.json
        self.assertEqual(json.loads((golden.REPOSITORY / "configs" / "ni-rua.example.json").read_text())
                         ["features"]["vggt_support_z_m"], "profile")
        # configs/so101_gateway.json follows the installed calibration (null: joint_calibration.table_z_m of the
        # observation), so a recalibration needs no edit here; a fixed number went stale after the 2026-11-18 one.
        self.assertIsNone(golden.load_config(golden.SO101_CONFIGS["so101_gateway"]).features.vggt_support_z_m)
        path = self.root / "so101.json"
        path.write_text(json.dumps({"backend": {"environment": "so101_real", "robot_model_id": "so101"}}))
        features = golden.load_config(path).features
        self.assertEqual(features.vggt_support_z_m, "profile")
        self.assertEqual(object_table.effective_support_z_m(features, CALIBRATED, SO101), -0.018)

    def test_validation_and_cli_forms(self):
        for invalid, code in (("observation", "invalid_config_choice"), ("Profile", "invalid_config_choice"),
                              (3.1, "config_above_maximum"), (-3.1, "config_below_minimum"),
                              (True, "invalid_config_type"), (float("nan"), "invalid_config_type")):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ConfigError, code):
                Config.from_dict({"features": {"vggt_support_z_m": invalid}})
        from ni_rua.apps.gateway.cli import main
        path = self.root / "base.json"
        path.write_text(json.dumps({"features": {"vggt_support_z_m": 0.5}}))
        for token, expected in (("profile", "profile"), ("observation", None), ("0.765", 0.765), ("-0.02", -0.02)):
            with self.subTest(token=token):
                out = io.StringIO()
                with mock.patch("sys.argv", ["ni-rua", "--config", str(path), "--print-config",
                                             "--vggt-support-z-m", token]), contextlib.redirect_stdout(out):
                    main()
                self.assertEqual(json.loads(out.getvalue())["features"]["vggt_support_z_m"], expected)


# ---- offline replay: the archived profile -------------------------------------------------------------------------
class OfflineReplayProfileTests(TemporaryRoot):
    def test_records_round_trip_for_both_robots(self):
        for profile in (e.ARX_X5_PROFILE, X5_ROBODOJO, e.arx_x5_profile(motion_execution_mode="feedback"), SO101):
            record = json.loads(json.dumps(e.profile_record(profile)))
            restored = e.profile_from_record(record)
            self.assertEqual(restored, profile)
            self.assertEqual(hash(restored), hash(profile))
            self.assertEqual(e.profile_record(restored), record)

    def test_unsupported_records_are_refused(self):
        record = e.profile_record(SO101)
        missing = json.loads(json.dumps(record))
        missing["profile"].pop("clock")
        extra = json.loads(json.dumps(record))
        extra["profile"]["gripper"]["new_field"] = 1
        shape = json.loads(json.dumps(record))
        shape["profile"]["gripper"] = "abc"
        for bad in (None, {}, {"schema": "other", "profile": record["profile"]}, missing, extra, shape,
                    {"schema": e.PROFILE_RECORD_SCHEMA, "profile": {"n": 1}}):
            with self.assertRaisesRegex(ValueError, "unsupported_embodiment_profile_record"):
                e.profile_from_record(bad)
        invalid = json.loads(json.dumps(record))
        invalid["profile"]["clock"] = "lunar"
        with self.assertRaisesRegex(ValueError, "invalid_embodiment_profile:clock"):
            e.profile_from_record(invalid)

    def recompute_runtime(self, run):
        """``checkpoint.recompute_tool_output`` up to the perception state it rebuilds: the offline runtime."""
        from ni_rua.apps.evaluation import checkpoint
        captured = []

        def capture(runtime, *args):
            captured.append(runtime)
            raise LookupError("captured")
        with mock.patch.object(checkpoint, "_hydrate_runtime_perception_state", side_effect=capture), \
                self.assertRaisesRegex(LookupError, "captured"):
            checkpoint.recompute_tool_output(run, "call-1")
        return captured[0]

    def test_recomputation_restores_the_archived_so101_profile(self):
        from ni_rua.apps.evaluation.checkpoint import archived_embodiment
        robot, config = so101_robot(self.root)
        run = self.root / "run"
        started = start_workbench(run, robot, config)
        self.assertEqual(started["manifest"]["embodiment_record_id"], started["record_id"])
        with open_read_store(run) as store:
            self.assertEqual(archived_embodiment(store, run), started["embodiment"])
        runtime = self.recompute_runtime(run)
        self.assertEqual(runtime.embodiment, started["embodiment"])
        self.assertEqual(embodiment_of(runtime).frame_id, "so101_base")
        self.assertEqual(object_table.default_camera(runtime), "fixed")
        # A manifest that does not name the record: the store's last profile record is used.
        manifest = json.loads((run / "manifest.json").read_text())
        manifest.pop("embodiment_record_id")
        (run / "manifest.json").write_text(json.dumps(manifest))
        self.assertEqual(self.recompute_runtime(run).embodiment, started["embodiment"])

    def test_archive_without_a_record_keeps_the_pre_profile_rules(self):
        from ni_rua.apps.evaluation.checkpoint import archived_embodiment
        run = self.root / "old-run"
        store = Store(run, dict(golden.IDENTITY))
        store.close()
        (run / "manifest.json").write_text(json.dumps({"schema_version": "0.1"}))
        (run / "effective_config.json").write_text(json.dumps(
            Config.from_dict({"backend": {"robot_model_id": "so101"}, "features": {"vggt_support_z_m": None}}).to_dict()))
        with open_read_store(run) as archive:
            self.assertIsNone(archived_embodiment(archive, run))
        runtime = self.recompute_runtime(run)
        self.assertFalse(hasattr(runtime, "embodiment"))
        self.assertEqual(object_table.default_camera(runtime, {"robot_model_id": "so101"}), "fixed")
        self.assertEqual(object_table.default_camera(runtime), "fixed")              # configured robot so101
        x5_runtime = SimpleNamespace(config=Config())
        self.assertEqual(object_table.default_camera(x5_runtime, {"robot_model_id": "arx-x5"}), "cam_head")
        self.assertIs(embodiment_of(runtime), e.ARX_X5_PROFILE)                       # documented default robot
        # A manifest naming a record the store lacks is treated the same way.
        (run / "manifest.json").write_text(json.dumps({"embodiment_record_id": "embodiment_profile-absent"}))
        with open_read_store(run) as archive:
            self.assertIsNone(archived_embodiment(archive, run))


# ---- leftover X5 facts on SO-101 paths ----------------------------------------------------------------------------
class LeftoverFactTests(unittest.TestCase):
    def test_geometry_batch_reports_the_run_frame(self):
        from ni_rua.runtime.tools.handlers import geometry_query
        for runtime, frame in ((SimpleNamespace(identity=dict(golden.IDENTITY), embodiment=SO101), "so101_base"),
                               (SimpleNamespace(identity=dict(golden.IDENTITY), embodiment=X5_ROBODOJO), "env_origin"),
                               (SimpleNamespace(identity=dict(golden.IDENTITY)), "env_origin")):
            response = geometry_query(runtime, {"request": {"kind": "batch", "queries": []}}, {})
            self.assertEqual(response.data["frame_id"], frame)

    def test_robotic_view_is_offered_only_for_the_x5_link6_tcp(self):
        from ni_rua.runtime.tools.context import ToolAvailability
        from ni_rua.runtime.tools.plugins.robotic_view import _unavailable

        def availability(capabilities):
            return ToolAvailability(Config().features, frozenset(), "rgb_proprio_calibrated_v1", capabilities,
                                    False, False, False, False, True)
        self.assertIsNone(_unavailable(availability(None), {}))
        self.assertIsNone(_unavailable(availability({"tcp_config_id": e.ARX_X5.tcp_config_id}), {}))
        self.assertEqual(_unavailable(availability({"tcp_config_id": e.SO101.tcp_config_id}), {}),
                         "robotic_view_requires_arx_x5_link6_wrist_camera")


if __name__ == "__main__":
    unittest.main()
