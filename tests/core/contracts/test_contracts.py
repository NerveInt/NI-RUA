import copy
import tempfile
import unittest

from ni_rua.core.contracts import validate_action, validate_observation
from ni_rua.core.storage.store import Store

NI_RUA_TEST_TIER = "component"


class ContractStoreCheck(unittest.TestCase):
    def test_shared_observation_profiles_are_typed_without_granting_actor_permission(self):
        from ni_rua.core.observation_policy import actor_policy
        observation = dict(schema_version="0.1", run_id="run", episode_id="ep", env_instance_id="env",
            env_generation=1, world_kind="observed", observation_id="obs", robot_state_ref="artifact:state", cameras=[])
        validate_observation(observation)  # Missing profile retains the historical RGB default.
        for profile in ("mistyped_profile", None, 1, True, [], {}):
            with self.subTest(profile=profile), self.assertRaisesRegex(ValueError, "unsupported_observation_profile"):
                validate_observation(dict(observation, observation_profile=profile))
        synthetic = dict(observation, observation_profile="rgb_proprio_calibrated_synthetic_v1",
                         calibration_id="fixture", data_origin="synthetic_fixture")
        validate_observation(synthetic)
        with self.assertRaisesRegex(ValueError, "actor_requires_supported_observation_profile"):
            actor_policy(synthetic["observation_profile"])

    def test_immutable_artifacts_events_and_generation(self):
        identity = dict(run_id="check", episode_id="ep", env_instance_id="env", env_generation=1)
        capabilities = dict(frame_id="env_origin", tcp_config_id="tcp", path_kinds=["backend_free"])
        action = dict(schema_version="0.1", **identity, action_id="a", based_on_observation_id="o",
                      frame_id="env_origin", tcp_config_id="tcp", mode="absolute", timeout_s=10,
                      path=dict(kind="backend_free", waypoints=[dict(left=dict(position_m=[0, 0, 1], quaternion_xyzw=[0, 0, 0, 1]))]))
        validate_action(action, identity, "o", capabilities)
        for key, value in [("env_generation", 2), ("world_kind", "internal"), ("based_on_observation_id", "old")]:
            bad = copy.deepcopy(action); bad[key] = value
            with self.assertRaises(ValueError): validate_action(bad, identity, "o", capabilities)
        with tempfile.TemporaryDirectory() as root:
            store = Store(root, identity)
            ref = store.put_artifact(b"rgb", "rgb", "image/png")
            self.assertEqual(store.artifact(ref)[1].read_bytes(), b"rgb")
            self.assertEqual(store.resolve_artifact(ref)[1], b"rgb")
            private_ref = store.put_artifact(b"score", "evaluation", "application/json", "evaluator")
            with self.assertRaisesRegex(ValueError, "access_scope"):
                store.resolve_artifact(private_ref)
            store.set_state("runtime", "identity", identity)
            store.save("request", "a", action); store.save("request", "a", action)
            with self.assertRaises(ValueError): store.save("request", "a", dict(action, timeout_s=11))
            store.event("action.requested", {}, {"action_id": "a"})
            store.event("action.completed", {}, {"action_id": "a"})
            self.assertEqual([e["server_seq"] for e in store.events(1)], [2])
            store.close()
            restored = Store(root, identity)
            self.assertEqual(restored.read("request", "a"), action)
            self.assertEqual(len(restored.events()), 2)
            self.assertEqual(restored.read_state("runtime", "identity"), identity)
            restored.artifact(ref)[1].write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "digest"):
                restored.resolve_artifact(ref)
            restored.close()

    def test_tool_error_four_category_decision_matrix_and_host_path_scrubbing(self):
        from ni_rua.core.tool_errors import (
            INTERNAL_BUG_TYPES,
            InternalToolRuntimeError,
            ToolInputError,
            WorkerRemoteError,
            error_payload,
            extract_domain_code,
            is_internal_runtime_error,
        )

        self.assertEqual(extract_domain_code("stale_observation: expired"), "stale_observation")
        self.assertIsNone(extract_domain_code("not enough values to unpack (expected 3, got 2)"))
        self.assertIsNone(extract_domain_code("ValueError: stale_observation"))

        internal_cases = [
            (TypeError("unsupported operand type(s) for -: 'NoneType' and 'float'"), "TypeError"),
            (AttributeError("'NoneType' object has no attribute 'q_rad'"), "AttributeError"),
            (IndexError("list index out of range"), "IndexError"),
            (ZeroDivisionError("float division by zero"), "ZeroDivisionError"),
            (ImportError("No module named 'broken_internal_pkg'"), "ImportError"),
            (ValueError("not enough values to unpack (expected 3, got 2)"), "ValueError"),
            (
                InternalToolRuntimeError(
                    "KeyError: 'missing_internal_key' in /Users/dev/secret/module.py",
                    tool="geometry_query",
                    error_type="KeyError",
                ),
                "KeyError",
            ),
            (
                WorkerRemoteError(
                    "KeyError: 'bad_worker_dict' at /home/ubuntu/run/worker.py",
                    remote_type="KeyError",
                    category="internal_runtime_error",
                    code="internal_runtime_error",
                    remote_message="'bad_worker_dict'",
                ),
                "KeyError",
            ),
        ]
        for exc, expected_type in internal_cases:
            with self.subTest(exc=repr(exc)):
                self.assertTrue(is_internal_runtime_error(exc))
                for tool_name in ("robot_action", "geometry_query", "depth_query"):
                    payload = error_payload(exc, tool=tool_name, arguments={"request": {"kind": "pixel_ray"}})
                    self.assertEqual(payload["error_category"], "internal_runtime_error")
                    self.assertEqual(payload["code"], "internal_runtime_error")
                    self.assertEqual(payload["error_type"], expected_type)
                    self.assertFalse(payload["retryable"])
                    self.assertEqual(payload["field_path"], "")
                    self.assertIsNone(payload["actual"])
                    self.assertIsNone(payload["minimal_example"])
                    self.assertNotIn("/Users/", payload["error"])
                    self.assertNotIn("/home/", payload["error"])

        self.assertTrue(all(issubclass(cls, BaseException) for cls in INTERNAL_BUG_TYPES))
        # Direct KeyError from Store lookup misses remains caller input_error (not internal bug).
        self.assertFalse(is_internal_runtime_error(KeyError("obs_missing")))
        key_payload = error_payload(KeyError("obs_missing"), tool="history_view_observation", arguments={"observation_id": "obs_missing"})
        self.assertEqual(key_payload["error_category"], "input_error")
        self.assertTrue(key_payload["retryable"])
        nested_key_payload = error_payload(
            KeyError("obs_nested"),
            tool="geometry_query",
            arguments={"request": {"observation_ids": ["obs_nested"]}},
        )
        self.assertEqual(nested_key_payload["error_category"], "input_error")
        self.assertTrue(nested_key_payload["retryable"])

        # KeyError whose missing key does NOT appear in caller arguments is classified as internal_runtime_error.
        for bad_args in ({}, {"observation_id": "obs_1"}, None):
            internal_key = error_payload(KeyError("missing_internal_dict_key"), tool="robot_action", arguments=bad_args)
            self.assertEqual(internal_key["error_category"], "internal_runtime_error")
            self.assertEqual(internal_key["code"], "internal_runtime_error")
            self.assertEqual(internal_key["error_type"], "KeyError")
            self.assertFalse(internal_key["retryable"])
            self.assertEqual(internal_key["field_path"], "")
            self.assertIsNone(internal_key["actual"])
            self.assertIsNone(internal_key["minimal_example"])

        # geometry_query / depth_query input errors point to published schema/guide without help-call bait.
        for gtool in ("geometry_query", "depth_query"):
            gpayload = error_payload(
                ValueError("pixel_out_of_bounds"),
                tool=gtool,
                arguments={"request": {"kind": "pixel_ray", "pixel": [-1, 2]}},
            )
            self.assertEqual(gpayload["error_category"], "input_error")
            self.assertNotEqual(gpayload["minimal_example"], {"request": {"kind": "help"}})
            self.assertNotIn('"help"', gpayload["suggestion"])
            self.assertIn("required fields", gpayload["suggestion"])

        # WorkerRemoteError carrying ik_service_unreachable promotes to infrastructure_error and matches code.
        ik_remote = WorkerRemoteError(
            "ValueError: ik_service_unreachable",
            remote_type="ValueError",
            category="input_error",
            code="ik_service_unreachable",
            remote_message="ik_service_unreachable",
            operation="reach_batch",
        )
        self.assertIsInstance(ik_remote, RuntimeError)
        self.assertFalse(is_internal_runtime_error(ik_remote))
        ik_payload = error_payload(ik_remote, tool="robot_action", arguments={"mode": "reach"})
        self.assertEqual(ik_payload["error_category"], "infrastructure_error")
        self.assertEqual(ik_payload["code"], "ik_service_unreachable")
        self.assertFalse(ik_payload["retryable"])
        self.assertEqual(ik_payload["field_path"], "")
        self.assertIsNone(ik_payload["actual"])
        self.assertIsNone(ik_payload["minimal_example"])

        # OSError / host path scrubbing on infrastructure_error.
        os_err = FileNotFoundError("[Errno 2] No such file: /private/var/folders/ab/run/robot-worker.log")
        self.assertFalse(is_internal_runtime_error(os_err))
        os_payload = error_payload(os_err, tool="robot_observe", arguments={})
        self.assertEqual(os_payload["error_category"], "infrastructure_error")
        self.assertFalse(os_payload["retryable"])
        self.assertNotIn("/private/var/", os_payload["error"])

        # TimeoutError / unknown_execution never claims no motion or emits retry bait.
        timeout_err = TimeoutError("robot status unknown: worker response deadline; do not replay action")
        self.assertFalse(is_internal_runtime_error(timeout_err))
        timeout_payload = error_payload(timeout_err, tool="robot_action", arguments={"mode": "execute"})
        self.assertEqual(timeout_payload["error_category"], "unknown_execution")
        self.assertEqual(timeout_payload["code"], "execution_status_unknown")
        self.assertFalse(timeout_payload["retryable"])
        self.assertEqual(timeout_payload["field_path"], "")
        self.assertIsNone(timeout_payload["actual"])
        self.assertIsNone(timeout_payload["minimal_example"])
        self.assertNotIn("no motion", timeout_payload["outcome_semantics"].lower())

        # Caller ToolInputError remains input_error with field_path and retryable=True.
        caller_err = ToolInputError("stale_observation", field_path="/observation_id", actual="obs_1")
        caller_payload = error_payload(caller_err, tool="robot_action", arguments={"observation_id": "obs_1"})
        self.assertEqual(caller_payload["error_category"], "input_error")
        self.assertEqual(caller_payload["code"], "stale_observation")
        self.assertTrue(caller_payload["retryable"])
        self.assertEqual(caller_payload["field_path"], "/observation_id")

    def test_profile_from_record_catches_embodiment_post_init_type_errors(self):
        from ni_rua.core.embodiment import profile_from_record, profile_record
        from ni_rua.robot.model import embodiments as e

        rec = profile_record(e.ARX_X5_PROFILE)
        bad = copy.deepcopy(rec)
        bad["profile"]["tracking_divergence_range_m"] = [0.01, "bad"]
        with self.assertRaisesRegex(ValueError, "^unsupported_embodiment_profile_record$"):
            profile_from_record(bad)


if __name__ == "__main__":
    unittest.main()
