"""Actor payload lean_v1/lean_v2 projection and Codex egress scrubbing contract tests."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from ni_rua.core.config import Config
from ni_rua.core.contracts import canonical
from ni_rua.core.presentation.actor_output import ActorOutput
from ni_rua.core.presentation.actor_payload import (
    ACTOR_PAYLOAD_PROFILES,
    DEFAULT_ACTOR_PAYLOAD_PROFILE,
    lean_action_result,
    lean_observation,
    lean_v2_display,
)
from ni_rua.core.presentation.actor_presentation import ActorPresentation, metadata_findings
from ni_rua.core.storage.store import Store
from ni_rua.core.tool_output import StructuredText, ToolResponse
from ni_rua.cortex.context.actor_instructions import instructions as actor_instructions

NI_RUA_TEST_TIER = "component"


DIGEST = "0123456789abcdef" * 4
ARTIFACT = "artifact:" + "a" * 64 + "-" + "b" * 16
LONG_OBS = "observation-" + "c" * 32
LONG_ACT = "action-" + "d" * 24


class ActorPayloadLeanProjectionTests(unittest.TestCase):
    def test_retired_lean_projection_module_and_config_flags(self):
        self.assertIsNone(importlib.util.find_spec("ni_rua.core.presentation.lean_projection"))
        self.assertEqual(ACTOR_PAYLOAD_PROFILES, ("full_v1", "compact_v1", "lean_v1", "lean_v2"))
        self.assertEqual(DEFAULT_ACTOR_PAYLOAD_PROFILE, "lean_v2")
        migrated = Config.from_dict(
            {
                "version": 1,
                "features": {
                    "lean_projection": True,
                    "lean_projection_k": 2,
                    "lean_projection_block": 4,
                    "lean_request_dump": 5,
                },
            }
        )
        self.assertEqual(migrated.version, 2)
        self.assertFalse(migrated.features.lean_projection)
        for retired_key in ("lean_projection", "lean_projection_k", "lean_projection_block", "lean_request_dump"):
            self.assertNotIn(retired_key, migrated.to_dict()["features"])

    def test_lean_observation_and_lean_v2_display_preserve_measurements_and_drop_storage_refs(self):
        raw_obs = {
            "bundle": {
                "observation_id": LONG_OBS,
                "sim_step": 120,
                "robot_state_ref": ARTIFACT,
                "cameras": [
                    {
                        "camera_id": "cam_head",
                        "frame_id": "raw-12345678-1234-1234-1234-123456789abc-0",
                        "rgb_ref": ARTIFACT,
                        "calibration_ref": ARTIFACT,
                        "pose_sample_ref": ARTIFACT,
                        "sha256": DIGEST,
                        "source_sha256": DIGEST,
                        "image_encoding": {"format": "jpeg"},
                    }
                ],
            },
            "robot_state": {
                "arms": {
                    "left": {
                        "grasp_pose": {
                            "position_m": [0.123456789, -0.0, 0.87654321],
                            "angles_deg": [0.0, 0.0, 90.0],
                        }
                    }
                }
            },
            "camera_calibration": {"cameras": {"cam_head": {"fx": 288.13253784}}},
            "camera_pose_sample": {
                "cameras": {"cam_head": {"frame_id": "raw-12345678-1234-1234-1234-123456789abc-0"}}
            },
            "camera_pose_diagnostics": {},
        }
        lean_v1 = lean_observation(raw_obs)
        self.assertTrue(lean_v1["bundle"]["robot_state_available"])
        self.assertNotIn("robot_state_ref", lean_v1["bundle"])
        cam = lean_v1["bundle"]["cameras"][0]
        self.assertTrue(cam["rgb_available"])
        self.assertTrue(cam["calibration_available"])
        self.assertTrue(cam["pose_sample_available"])
        for omitted in ("rgb_ref", "calibration_ref", "pose_sample_ref", "sha256", "source_sha256", "image_encoding"):
            self.assertNotIn(omitted, cam)

        lean_v2 = lean_v2_display(lean_v1, tool="robot_observe")
        self.assertNotIn("frame_id", lean_v2["bundle"]["cameras"][0])
        self.assertNotIn("camera_pose_diagnostics", lean_v2)
        self.assertNotIn("cameras", lean_v2["camera_pose_sample"])
        pos = lean_v2["robot_state"]["arms"]["left"]["grasp_pose"]["position_m"]
        self.assertEqual(pos, [0.123457, 0.0, 0.876543])

    def test_lean_action_result_preserves_diagnostics_and_summarizes_pass_checks(self):
        result = {
            "request": {"action_id": LONG_ACT, "based_on_observation_id": LONG_OBS},
            "receipt": {
                "action_id": LONG_ACT,
                "status": "completed",
                "versions": {"worker": "1.0"},
                "planned_trace_ref": ARTIFACT,
                "actual_trace_ref": None,
                "endpoint_residual_m": 0.0012,
                "motion_safety_checks": [
                    {
                        "status": "pass",
                        "arms": {
                            "left": {
                                "policy": "joint-travel-admission-v1",
                                "status": "pass",
                                "reasons": [],
                                "witnesses": [],
                                "exceeded_budgets": {},
                                "joint_names": ["j1", "j2"],
                                "joint_travel_rad": [0.1, 0.2],
                                "thresholds": {"max_rad": 2.0},
                                "semantics": "advisory",
                                "periodic_checks": [
                                    {
                                        "joint_name": "j1",
                                        "status": "continuous_joint",
                                        "eligible": True,
                                    }
                                ],
                            }
                        },
                    }
                ],
            },
        }
        projected = lean_action_result(result)
        receipt = projected["receipt"]
        self.assertNotIn("versions", receipt)
        self.assertTrue(receipt["planned_trace_available"])
        self.assertFalse(receipt["actual_trace_available"])
        self.assertEqual(receipt["endpoint_residual_m"], 0.0012)
        self.assertEqual(receipt["motion_safety_check_summary"]["pass_count"], 1)
        self.assertEqual(receipt["motion_safety_checks"][0]["arms"]["left"]["detail"], "passed_check_summary")

    def test_actor_output_lean_v2_keeps_typed_data_exact_and_scrubs_display(self):
        temp = self.enterContext(tempfile.TemporaryDirectory())
        store = Store(Path(temp) / "run", dict(run_id="lean", episode_id="ep", env_instance_id="env", env_generation=1))
        self.addCleanup(store.close)
        presentation = ActorPresentation(store)
        out = ActorOutput(store, presentation, payload_profile="lean_v2")
        raw_data = {
            "job_id": "job-" + "f" * 24,
            "status": "succeeded",
            "measurement": 0.123456789,
            "artifact": ARTIFACT,
            "source_sha256": DIGEST,
            "result_semantics": "omitted in lean_v2 display",
        }
        response = ToolResponse([StructuredText(raw_data, "owner_record")], data=dict(raw_data), kind="program")
        wire = out.present(response, tool="program_run")
        self.assertTrue(wire["success"])
        text = wire["contentItems"][0]["text"]
        self.assertEqual(metadata_findings(text), [])
        decoded = json.loads(text)
        self.assertEqual(decoded["job_id"], "job_1")
        self.assertEqual(decoded["measurement"], 0.123457)
        self.assertTrue(decoded["artifact"].startswith("artifact:ref_"))
        self.assertNotIn("result_semantics", decoded)
        # Original typed business data stays untouched at full precision.
        self.assertEqual(response.data["measurement"], 0.123456789)
        self.assertEqual(presentation.restore("job_1"), raw_data["job_id"])

    def test_operation_memory_instruction_omitted_when_tool_disabled(self):
        with_mem = actor_instructions("obs", "handoff", operation_memory=True)
        without_mem = actor_instructions("obs", "handoff", operation_memory=False)
        self.assertIn("operation_memory separates current-task records", with_mem)
        self.assertNotIn("operation_memory separates current-task records", without_mem)
        self.assertIn("Read the public Harness implementation under /harness.", without_mem)


class CodexEgressScrubTests(unittest.TestCase):
    """CodexSession._present_model_request multi-wire egress scrubbing and native feature preservation."""

    def session(self, features=None):
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
        store = Store(
            root / "run",
            dict(run_id="lean", episode_id="episode", env_instance_id="environment", env_generation=1),
        )
        self.addCleanup(store.close)
        config = Config.from_dict({"model": {"codex_home": str(root / "source")}, "features": features or {}})
        session = CodexSession(store.root, store, lambda _: {}, config=config)
        self.addCleanup(session.close)
        return session

    def test_baseline_preserves_native_features_and_scrubs_egress_ids_and_hashes(self):
        import subprocess
        import tomllib
        from unittest import mock

        session = self.session({"sandbox_enabled": False})
        relay = mock.MagicMock(base_url="http://127.0.0.1:54321")
        relay.start.return_value = relay
        with mock.patch("ni_rua.cortex.agents.codex.session.ModelRequestProxy", return_value=relay):
            session._prepare()
        config = tomllib.loads((session.codex_home / "config.toml").read_text())
        result = subprocess.run(
            session.command[:-5] + [str(session.cli_executable), "features", "list"],
            env=session.env,
            cwd=session.actor_dir,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(config["features"]["multi_agent"])
        self.assertTrue(config["features"]["multi_agent_v2"])
        self.assertNotIn("skills", config)
        with open(config["model_catalog_json"], encoding="utf-8") as handle:
            catalog = json.load(handle)
        self.assertIn("v2", [model["multi_agent_version"] for model in catalog["models"]])

        image_data_url = "data:image/png;base64,AAAA" + DIGEST
        request = {
            "model": "gpt-fixture",
            "instructions": f"Base sha256={DIGEST} ref {ARTIFACT}",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": [
                        {
                            "type": "input_text",
                            "text": canonical(
                                {
                                    "bundle": {"observation_id": LONG_OBS, "source_sha256": DIGEST},
                                    "artifact": ARTIFACT,
                                }
                            ),
                        },
                        {"type": "input_image", "image_url": image_data_url, "detail": "high"},
                    ],
                }
            ],
        }
        presented = session._present_model_request(request)
        # Media stays inline and untouched; provenance hashes & raw IDs are scrubbed to reversible short_ids.
        self.assertEqual(presented["input"][0]["output"][1]["image_url"], image_data_url)
        self.assertNotIn(DIGEST, presented["instructions"])
        self.assertIn("artifact:ref_", presented["instructions"])
        out_text = presented["input"][0]["output"][0]["text"]
        self.assertEqual(metadata_findings(out_text), [])
        decoded = json.loads(out_text)
        self.assertEqual(decoded["bundle"]["observation_id"], "obs_1")
        self.assertEqual(session.presentation.restore("obs_1"), LONG_OBS)
        # Repeated presentation is byte-identical (prefix-cache safe).
        self.assertEqual(canonical(presented), canonical(session._present_model_request(request)))


if __name__ == "__main__":
    unittest.main()
