"""Consolidated tests:

- test_checkout_layout: Checkout layout and OOM recovery command validation.
- test_queue_config: Unified configuration crosses queue boundaries without consulting changed files.
- test_queue_provenance: No GPU: immutable source bindings cover changed, added and removed modules.
"""

from pathlib import Path
import tempfile
import unittest
from ni_rua.apps.evaluation import queue as eval_queue
from ni_rua.apps.evaluation.provenance import CHECKOUT_LAYOUTS, checkout_environment, checkout_layout
from ni_rua.apps.evaluation.retry import recovered_command
from copy import deepcopy
import json
import sys
from unittest.mock import AsyncMock, patch
from ni_rua.core.config import Config, cli_overrides
from ni_rua.core.observation_policy import ACTOR_PROFILE, DEPTH_ACTOR_PROFILE
from scripts import benchmark_group, benchmark_queue as queue, install_library_service
from ni_rua.apps.evaluation.provenance import bind_sources, checkout_layout, verify_source
from support.paths import REPO_ROOT


NI_RUA_TEST_TIER = "fast"


# --- test_checkout_layout.py ---
class CheckoutLayoutTests(unittest.TestCase):
    def test_checkout_environment_preserves_queue_settings(self):
        environment = {"NI_RUA_GRASP_MEMORY_DIR": "/memory/case", "NI_RUA_PRICE_CARD": "/cards/a.json", "PATH": "/bin"}
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory) / "ni-rua-checkout"
            (checkout / "ni_rua/apps/gateway/cli.py").parent.mkdir(parents=True)
            (checkout / "ni_rua/apps/gateway/cli.py").write_text("")
            launched = checkout_environment(environment, checkout_layout(checkout))
            self.assertEqual(launched, environment)

    def test_oom_recovery_replays_commands_of_every_checkout_layout(self):
        for layout in CHECKOUT_LAYOUTS:
            with self.subTest(module=layout["module"]):
                entry = {"case": {"run_dir": "/runs/retry"},
                         "retry_command_template": ["/usr/bin/python", "-u", "-m", layout["module"], "--benchmark",
                                                    "--run-dir", "/runs/original", "--gpu", "1"]}
                self.assertEqual(recovered_command(entry, 3)[-4:], ["--run-dir", "/runs/retry", "--gpu", "3"])
        entry["retry_command_template"][3] = "other.gateway"
        with self.assertRaisesRegex(ValueError, "fixed-budget benchmark gateway command"):
            recovered_command(entry, 3)


# --- test_queue_config.py ---
class QueueConfigTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        checkout = self.root / "checkout"
        (checkout / "ni_rua/apps/gateway").mkdir(parents=True)
        (checkout / "ni_rua/core").mkdir(parents=True)
        (checkout / "ni_rua/apps/gateway/cli.py").write_text("parser.add_argument('--config')\nConfig.load()\n")
        self.case = dict(cwd=str(checkout), run_dir=str(self.root / "run"), task="stack_bowls",
                         seed=0, layout_id=0, title="config", tag=[])
        self.config = Config.from_dict({"features": {"enable_wait": False, "pi05": True,
                "tracking_guard": True}, "queue": {"max_parallel": 2, "oom_retry": {"max_retries": 0}}})

    def test_case_can_disable_shared_features_and_wait_can_override_both_ways(self):
        command = queue.command({**self.case, "pi05": False, "tracking_guard": False,
                                 "enable_wait": True}, 1, config=self.config,
                                config_path=self.root / "snapshot.json")
        self.assertIn("--no-pi05", command)
        self.assertIn("--no-tracking-guard", command)
        self.assertIn("--enable-wait", command)
        self.assertNotIn("--no-enable-wait", command)
        inherited = queue.command(self.case, 1, config=self.config)
        self.assertEqual(inherited.count("--no-enable-wait"), 1)

    def test_queue_command_reaches_actual_gateway_parser_with_case_overrides(self):
        from ni_rua.apps.gateway import cli as gateway
        path = self.root / "snapshot.json"
        configured = self.config.with_overrides({"features": {"joint_replan": True,
            "motion_review": True, "visual_review_model": "gpt-6.1-sol",
            "disabled_tools": ["robot_wait"], "record_video": True,
            "export_video": True, "terminal": True, "vision_compare": True,
            "default_task_notes": True, "actor_payload_profile": "lean_v1",
            "completion_guidance": "both"}, "run": {"tag": ["inherited"]}})
        for config, changes in ((Config.from_dict({}), {}), (configured, {}), (configured,
                {"pi05": False, "tracking_guard": False, "joint_replan": False,
                 "motion_review": False, "enable_wait": True, "visual_review_model": None,
                 "disabled_tools": [], "record_video": False, "export_video": False,
                 "terminal": False, "vision_compare": False, "default_task_notes": False,
                 "completion_guidance": "none",
                 "actor_payload_profile": "full_v1", "grasp_memory_dir": str(self.root / "memory"), "tag": "single"})):
            with self.subTest(changes=changes):
                path.write_text(json.dumps(config.to_dict()))
                case = {**self.case, **changes}
                command = queue.command(case, 3, config=config, config_path=path)
                with patch.object(gateway, "serve", new_callable=AsyncMock) as serve, \
                        patch.object(sys, "argv", ["gateway", *command[4:]]):
                    gateway.main()
                actual = serve.call_args.args[0]
                for key in ("enable_wait", "pi05", "tracking_guard", "joint_replan", "motion_review",
                            "visual_review_model", "record_video", "export_video", "terminal", "grasp_memory_dir",
                            "vision_compare", "default_task_notes", "actor_payload_profile", "completion_guidance"):
                    self.assertEqual(getattr(actual, key), changes.get(key, getattr(config.features, key)), key)
                self.assertEqual(actual.disabled_tools, tuple(changes.get("disabled_tools", config.features.disabled_tools)))
                self.assertEqual(actual.run_dir, self.case["run_dir"])
                self.assertEqual(actual.gpu, 3)
                self.assertEqual(actual.tag, ("single",) if changes else ())

    def test_current_default_reservation_matches_calibrated_ten_gib(self):
        config = Config.from_dict({})
        effective = queue.configured_manifest({"cases": [self.case]}, config)
        self.assertEqual(config.queue.max_parallel, 60)
        self.assertIsNone(config.queue.eligible_gpu_indices)
        self.assertEqual(config.queue.memory_per_case_mib, 10 * 1024)
        self.assertEqual(config.queue.gpu_headroom_mib, 0)
        self.assertEqual(queue.CASE_MEMORY_MIB, 10 * 1024)
        self.assertEqual(queue.INITIAL_LAUNCH_LIMIT, 60)
        self.assertEqual(queue.WINDOW_LAUNCH_LIMIT, 60)
        self.assertEqual(effective["max_parallel"], 60)
        self.assertNotIn("eligible_gpu_indices", effective)
        self.assertIsNone(effective["ni_rua_config"]["queue"]["eligible_gpu_indices"])
        self.assertEqual(effective["memory_per_case_mib"], 10 * 1024)
        self.assertEqual(config.features.actor_payload_profile, "lean_v2")
        self.assertEqual(config.features.completion_guidance, "hook")
        self.assertEqual(effective["ni_rua_config"]["features"]["completion_guidance"], "hook")

    def test_old_runtime_schema_omits_only_default_sandbox_field(self):
        current = self.config.to_dict()
        feature_fields = set(current["features"])
        feature_fields.remove("sandbox_enabled")
        class_name = lambda key: "".join(part.capitalize() for part in key.split("_")) + "Config"
        schema = [{"Config": set(current), "FeaturesConfig": feature_fields,
                   **{class_name(key): set(current[key])
                      for key in current if isinstance(current[key], dict) and key != "features"}}]
        omissions = []
        projected = queue.project_config_for_runtimes(current, schema, omissions)
        self.assertNotIn("sandbox_enabled", projected["features"])
        self.assertEqual(omissions, [{"key": "features.sandbox_enabled", "value": False,
                                      "reason": "target_schema_omits_default"}])

    def test_old_runtime_schema_rejects_explicit_sandbox_disable(self):
        current = self.config.with_overrides({"features": {"sandbox_enabled": True}}).to_dict()
        feature_fields = set(current["features"]) - {"sandbox_enabled"}
        class_name = lambda key: "".join(part.capitalize() for part in key.split("_")) + "Config"
        schema = [{"Config": set(current), "FeaturesConfig": feature_fields,
                   **{class_name(key): set(current[key])
                      for key in current if isinstance(current[key], dict) and key != "features"}}]
        with self.assertRaisesRegex(ValueError, "unified_config_incompatible:features.sandbox_enabled"):
            queue.project_config_for_runtimes(current, schema)

    def test_new_runtime_schema_keeps_explicit_sandbox_disable(self):
        current = Config.from_dict({"features": {"sandbox_enabled": False}}).to_dict()
        schema = [queue.runtime_config_schema(REPO_ROOT)]
        projected = queue.project_config_for_runtimes(current, schema)
        self.assertIs(projected["features"]["sandbox_enabled"], False)

    def test_unreviewed_schema_drift_fails_closed_even_at_default(self):
        current = Config().to_dict()
        schema = queue.runtime_config_schema(REPO_ROOT)
        schema = {name: set(fields) for name, fields in schema.items()}
        schema["FeaturesConfig"].remove("completion_guidance")
        with self.assertRaisesRegex(ValueError, "unified_config_incompatible:features.completion_guidance"):
            queue.project_config_for_runtimes(current, [schema])

    def test_queue_rejects_unsupported_sandbox_disable_before_gpu_admission(self):
        config_dir = REPO_ROOT / "ni_rua/core/config"
        source = "\n".join(p.read_text() for p in sorted(config_dir.glob("*.py")))
        (Path(self.case["cwd"]) / "ni_rua/core/config.py").write_text(
            source.replace("    sandbox_enabled: bool = False\n", ""))
        config = Config.from_dict({"features": {"sandbox_enabled": True}})
        manifest = queue.configured_manifest({"cases": [self.case]}, config)
        output = self.root / "incompatible"
        with patch.object(queue, "sample_gpus") as sample, patch.object(queue.subprocess, "Popen") as launch, \
                self.assertRaisesRegex(ValueError, "unified_config_incompatible:features.sandbox_enabled"):
            queue.run_queue(manifest, output)
        sample.assert_not_called()
        launch.assert_not_called()
        self.assertFalse(output.exists())

    def test_oom_recovery_accepts_recorded_runtime_projection(self):
        config_dir = REPO_ROOT / "ni_rua/core/config"
        source = "\n".join(p.read_text() for p in sorted(config_dir.glob("*.py")))
        (Path(self.case["cwd"]) / "ni_rua/core/config.py").write_text(
            source.replace("    sandbox_enabled: bool = False\n", ""))
        config = Config()
        manifest = queue.configured_manifest({"cases": [self.case]}, config)
        schema = queue.runtime_config_schema(self.case["cwd"])
        projected = queue.project_config_for_runtimes(manifest["ni_rua_config"], [schema])
        snapshot = self.root / "projected.json"
        snapshot.write_text(json.dumps(projected))
        entry = {"case": self.case, "attempt_number": 1, "root_run_dir": self.case["run_dir"],
                 "status": "not_started", "logical_task_id": "compat-recovery",
                 "retry_command_template": queue.command(self.case, 0, config_path=snapshot)}
        manifest["config_compatibility"] = {"status": "projected",
            "omitted_defaults": [{"key": "features.sandbox_enabled", "value": False,
                                  "reason": "target_schema_omits_default"}],
            "target_checkouts": [self.case["cwd"]]}
        with patch.object(queue, "sample_gpus", return_value={"gpus": [], "error": None}):
            self.assertEqual(queue.run_queue(manifest, self.root / "recovery", wait_timeout_s=.001,
                                             recovery={"entries": [entry], "prior_attempts": []}), 1)

    def test_runtime_schema_parser_is_static_and_finds_old_checkout_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ni_rua/apps/gateway").mkdir(parents=True)
            (root / "ni_rua/core").mkdir(parents=True)
            (root / "ni_rua/apps/gateway/cli.py").write_text("")
            (root / "ni_rua/core/config.py").write_text(
                'raise RuntimeError("the schema parser must not execute the checkout")\n'
                "class FeaturesConfig:\n    completion_guidance: str = 'hook'\n"
                "class Config:\n    features: FeaturesConfig\n")
            schema = queue.runtime_config_schema(directory)
        self.assertIsNotNone(schema)
        self.assertNotIn("sandbox_enabled", schema["FeaturesConfig"])
        self.assertIn("completion_guidance", schema["FeaturesConfig"])

    def test_new_case_boolean_flags_require_real_booleans(self):
        for name in ("vision_compare", "default_task_notes"):
            for value in (None, 0, "false"):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    queue.command({**self.case, name: value}, 0)
            for value in (True, False):
                command = queue.command({**self.case, name: value}, 0)
                flag = "--" + ("" if value else "no-") + name.replace("_", "-")
                self.assertIn(flag, command)

    def test_standalone_commands_reject_depth_profile_and_feedback_conflicts(self):
        for changes in ({"depth_skills": True}, {"depth_auto_feedback": True},
                        {"depth_event_payload": True},
                        {"observation_profile": "rgb_proprio_v1", "depth_skills": True},
                        {"observation_profile": DEPTH_ACTOR_PROFILE, "depth_event_payload": True}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "depth_"):
                queue.command({**self.case, **changes}, 0)
        for name in ("depth_skills", "depth_auto_feedback", "depth_event_payload"):
            for value in (None, 0, "false"):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    queue.command({**self.case, "observation_profile": DEPTH_ACTOR_PROFILE, name: value}, 0)

    def test_default_depth_stays_off_and_rgbd_preserves_motion_review(self):
        from ni_rua.apps.gateway import cli as gateway
        snapshot = self.root / "default-config.json"
        snapshot.write_text(json.dumps(Config().to_dict()))
        for changes in ({}, {"observation_profile": DEPTH_ACTOR_PROFILE},
                        {"observation_profile": DEPTH_ACTOR_PROFILE, "depth_skills": True,
                         "depth_auto_feedback": True, "depth_event_payload": True},
                        {"observation_profile": DEPTH_ACTOR_PROFILE, "motion_review": False}):
            # No Config object is passed: command itself must validate defaults.
            command = queue.command({**self.case, **changes}, 0, config_path=snapshot)
            with self.subTest(changes=changes), patch.object(gateway, "serve", new_callable=AsyncMock) as serve, \
                    patch.object(sys, "argv", ["gateway", *command[4:]]):
                gateway.main()
            actual = serve.call_args.args[0]
            self.assertEqual(actual.observation_profile, changes.get("observation_profile", ACTOR_PROFILE))
            self.assertEqual(actual.motion_review, changes.get("motion_review", True))
            for feature in ("depth_skills", "depth_auto_feedback", "depth_event_payload"):
                self.assertEqual(getattr(actual, feature), changes.get(feature, False))

    def test_case_profile_override_must_disable_inherited_depth_features(self):
        shared = Config.from_dict({"features": {"observation_profile": DEPTH_ACTOR_PROFILE,
            "depth_skills": True, "depth_auto_feedback": True, "depth_event_payload": True}})
        with self.assertRaisesRegex(ValueError, "depth_features_require"):
            queue.command({**self.case, "observation_profile": ACTOR_PROFILE}, 0, config=shared)
        with self.assertRaisesRegex(ValueError, "depth_event_payload_requires"):
            queue.command({**self.case, "depth_auto_feedback": False}, 0, config=shared)
        command = queue.command({**self.case, "observation_profile": ACTOR_PROFILE,
            "depth_skills": False, "depth_auto_feedback": False, "depth_event_payload": False}, 0, config=shared)
        for flag in ("--no-depth-skills", "--no-depth-auto-feedback", "--no-depth-event-payload"):
            self.assertIn(flag, command)

    def test_invalid_unconfigured_queue_is_rejected_before_output_or_gpu_admission(self):
        case = {**self.case, "depth_skills": True}
        for recovery in (None, {"entries": []}):
            output = self.root / ("invalid-recovery" if recovery else "invalid-fresh")
            with self.subTest(recovery=bool(recovery)), \
                    patch.object(queue, "queue_config", return_value=Config()), \
                    patch.object(queue, "sample_gpus") as sample, \
                    patch.object(queue.subprocess, "Popen") as launch, \
                    self.assertRaisesRegex(ValueError, "depth_features_require"):
                queue.run_queue({"cases": [case]}, output, recovery=recovery)
            sample.assert_not_called()
            launch.assert_not_called()
            self.assertFalse(output.exists())

    def test_recovery_validation_uses_archived_rgbd_defaults_for_case_flags(self):
        shared = Config.from_dict({"features": {"observation_profile": DEPTH_ACTOR_PROFILE}})
        case = {**self.case, "depth_skills": True}
        manifest = queue.configured_manifest({"cases": [case]}, shared)
        path = self.root / "rgbd-config.json"
        path.write_text(json.dumps(manifest["ni_rua_config"]))
        entry = {"case": case, "attempt_number": 1, "root_run_dir": case["run_dir"],
                 "status": "not_started", "logical_task_id": "rgbd-fixture",
                 "retry_command_template": queue.command(case, 0, config=shared, config_path=path)}
        with patch.object(queue, "sample_gpus", return_value={"gpus": [], "error": None}) as sample, \
                patch.object(queue.subprocess, "Popen") as launch:
            self.assertEqual(queue.run_queue(manifest, self.root / "rgbd-recovery", wait_timeout_s=.001,
                recovery={"entries": [entry], "prior_attempts": []}), 1)
        sample.assert_called()
        launch.assert_not_called()

    def test_manifest_queue_overrides_and_snapshots_do_not_mutate_original(self):
        manifest = {"cases": [self.case], "max_parallel": 3, "oom_retry": {"initial_delay_s": 7}}
        effective = queue.configured_manifest(manifest, self.config)
        self.assertNotIn("ni_rua_config", manifest)
        self.assertEqual(effective["max_parallel"], 3)
        self.assertEqual(effective["oom_retry"]["max_retries"], 0)
        self.assertEqual(effective["oom_retry"]["initial_delay_s"], 7)
        self.assertEqual(effective["ni_rua_config"]["queue"]["max_parallel"], 3)
        with patch.object(queue, "experiment_environment", side_effect=AssertionError("must not reload")):
            frozen = queue.queue_config(effective)
        self.assertEqual(frozen.queue.max_parallel, 3)

    def test_queue_persists_resolved_configuration_before_gpu_wait(self):
        manifest = queue.configured_manifest({"cases": [self.case]}, self.config)
        output = self.root / "queue"

        def sample():
            snapshot = json.loads((output / "effective_config.json").read_text())
            self.assertFalse(snapshot["features"]["enable_wait"])
            self.assertEqual(snapshot["queue"]["wait_timeout_s"], .001)
            return {"gpus": [], "error": None}

        with patch.object(queue, "sample_gpus", side_effect=sample), \
                patch.object(queue.subprocess, "Popen") as launch:
            self.assertEqual(queue.run_queue(manifest, output, wait_timeout_s=.001), 1)
        launch.assert_not_called()
        stored = json.loads((output / "config.json").read_text())
        self.assertEqual(stored["manifest"]["ni_rua_config"]["features"]["enable_wait"], False)

    def test_old_checkout_is_rejected_before_output_or_gpu_admission(self):
        (Path(self.case["cwd"]) / "ni_rua/apps/gateway/cli.py").write_text("# legacy gateway without config support\n")
        manifest = queue.configured_manifest({"cases": [self.case]}, self.config)
        with patch.object(queue, "sample_gpus") as sample, \
                patch.object(queue.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(ValueError, "unified_config_requires_supported_gateway"):
                queue.run_queue(manifest, self.root / "unsupported")
        sample.assert_not_called()
        launch.assert_not_called()
        self.assertFalse((self.root / "unsupported").exists())

    def test_incompatible_shared_budget_is_rejected_before_gpu_sampling(self):
        config = self.config.with_overrides({"run": {"max_actions": 3}})
        manifest = queue.configured_manifest({"cases": [self.case]}, config)
        with patch.object(queue, "sample_gpus") as sample, \
                patch.object(queue.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(ValueError, "official native"):
                queue.run_queue(manifest, self.root / "invalid")
        sample.assert_not_called()
        launch.assert_not_called()
        self.assertFalse((self.root / "invalid").exists())

    def test_group_embeds_config_and_explicit_pi05_false_wins(self):
        manifest = benchmark_group.build_manifest("gpt-as-policy-10", self.root / "runs",
                                                  pi05=False, config=self.config)
        self.assertEqual(manifest["max_parallel"], 2)
        self.assertFalse(manifest["ni_rua_config"]["features"]["pi05"])
        self.assertTrue(all(case["pi05"] is False for case in manifest["cases"]))
        self.assertFalse(manifest["ni_rua_config"]["features"]["enable_wait"])

    def test_notifications_can_be_disabled_despite_existing_thread_binding(self):
        path, manifest_path = self.root / "config.json", self.root / "manifest.json"
        path.write_text(json.dumps({"queue": {"wake_enabled": False}}))
        manifest_path.write_text(json.dumps({"cases": [self.case]}))
        binding = self.root / "thread"
        binding.write_text("invalid thread must never be read")
        with patch.object(queue.sys, "argv", ["queue", str(manifest_path), "--config", str(path),
                        "--output", str(self.root / "output")]), \
                patch.object(queue, "WAKE_THREAD_FILE", binding), \
                patch.object(queue, "EXPERIMENT_SETTINGS", self.root / "missing.json"), \
                patch.object(queue, "run_queue", return_value=0), \
                patch("scripts.experiment_hook.notify_completion") as notify:
            self.assertEqual(queue.main(), 0)
        notify.assert_not_called()

    def test_recovery_rejects_changed_configuration_snapshot_before_gpu_admission(self):
        manifest = queue.configured_manifest({"cases": [self.case]}, self.config)
        path = self.root / "original-config.json"
        path.write_text(json.dumps(self.config.with_overrides({"features": {"enable_wait": True}}).to_dict()))
        entry = {"case": self.case, "attempt_number": 1, "root_run_dir": self.case["run_dir"],
                 "retry_command_template": queue.command(self.case, 0, config=self.config, config_path=path)}
        with patch.object(queue, "sample_gpus") as sample, \
                patch.object(queue.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(ValueError, "snapshot differs"):
                queue.run_queue(manifest, self.root / "recovery", recovery={"entries": [entry]})
        sample.assert_not_called()
        launch.assert_not_called()
        self.assertFalse((self.root / "recovery").exists())

    def test_recovery_accepts_unchanged_snapshot_from_before_new_feature_fields(self):
        archived = self.config.to_dict()
        for field in ("vision_compare", "default_task_notes"):
            archived["features"].pop(field)
        path = self.root / "old-config.json"
        path.write_text(json.dumps(archived))
        manifest = {"cases": [self.case], "ni_rua_config": archived}
        entry = {"case": self.case, "attempt_number": 1, "root_run_dir": self.case["run_dir"],
                 "status": "not_started", "logical_task_id": "fixture",
                 "retry_command_template": queue.command(self.case, 0, config_path=path)}
        with patch.object(queue, "sample_gpus", return_value={"gpus": [], "error": None}) as sample:
            self.assertEqual(queue.run_queue(manifest, self.root / "old-recovery", wait_timeout_s=.001,
                recovery={"entries": [entry], "prior_attempts": []}), 1)
        sample.assert_called()

    def test_recovery_keeps_archived_notification_opt_out(self):
        config = self.config.with_overrides({"queue": {"wake_enabled": False, "wait_timeout_s": 42}})
        manifest = queue.configured_manifest({"cases": [self.case]}, config)
        binding = self.root / "thread"
        binding.write_text("invalid binding must not be read")
        with patch.object(sys, "argv", ["queue", "--recover-oom-from", str(self.root / "old"),
                    "--output", str(self.root / "new")]), \
                patch.object(queue, "WAKE_THREAD_FILE", binding), \
                patch.object(queue, "recover_oom", return_value=(manifest, {"entries": []})), \
                patch.object(queue, "run_queue", return_value=0) as run, \
                patch("scripts.experiment_hook.notify_completion") as notify:
            self.assertEqual(queue.main(), 0)
        self.assertEqual(run.call_args.args[2], 42)
        notify.assert_not_called()

    def test_service_unit_carries_resolved_library_options(self):
        unit = install_library_service.unit_text("/repo", "/python", "/runs", 9000,
            scan_interval=2, public_origin="https://library.example", task_source_root="/task source")
        self.assertIn("--scan-interval 2", unit)
        self.assertIn('--public-origin "https://library.example"', unit)
        self.assertIn('--task-source-root "/task source"', unit)

    def test_gpu_allowlist_changes_only_admission_candidates(self):
        original = dict(gpus=[dict(index=i, uuid=f"GPU-{i}") for i in range(4)], error=None, processes=[{"pid": 1}])
        before = deepcopy(original)
        value = queue.eligible_gpu_sample(original, [1, 2, 3])
        self.assertEqual([gpu["index"] for gpu in value["gpus"]], [1, 2, 3])
        self.assertEqual(value["all_observed_gpu_indices"], [0, 1, 2, 3])
        self.assertEqual(value["processes"], original["processes"])
        self.assertEqual(original, before)
        self.assertIs(queue.eligible_gpu_sample(original, None), original)
        self.assertEqual(queue.eligible_gpu_sample(original, [5])["gpus"], [])
        for invalid in ([], [True], [1, 1], [-1], {}, "1"):
            with self.assertRaises(ValueError):
                queue.eligible_gpu_sample(original, invalid)

    def test_manipulation_and_protocol_switches_roundtrip_strictly(self):
        case = dict(self.case, arm_queue=False, motion_cost_preview=True)
        args = queue.command(case, 1, config=Config())
        self.assertIn("--no-arm-queue", args)
        self.assertIn("--motion-cost-preview", args)
        cfg = Config().with_overrides(cli_overrides(case))
        self.assertFalse(cfg.features.arm_queue)
        self.assertTrue(cfg.features.motion_cost_preview)
        for key in ("arm_queue", "motion_cost_preview"):
            with self.assertRaises(ValueError):
                queue.command(dict(case, **{key: 1}), 1, config=Config())

        keys = ("bounded_continuation", "motion_safety_guard", "trajectory_preview", "manipulation_protocol")
        for key in keys:
            self.assertIs(getattr(Config().features, key), True)
            args = queue.command(dict(self.case, **{key: False}), 1, config=Config())
            self.assertIn("--no-" + key.replace("_", "-"), args)
            config = Config().with_overrides(cli_overrides({key: False}))
            self.assertIs(getattr(config.features, key), False)
            for other in keys:
                if other != key:
                    self.assertIs(getattr(config.features, other), True)
            with self.assertRaises(ValueError):
                queue.command(dict(self.case, **{key: 1}), 1, config=Config())


# --- test_queue_provenance.py ---
class SourceBindingTests(unittest.TestCase):
    def test_launch_binding_detects_runtime_changes_but_ignores_generated_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            package = root / "ni_rua"
            (package / "apps/gateway").mkdir(parents=True)
            gateway = package / "apps/gateway/cli.py"
            gateway.write_text("original = True\n")
            binding = bind_sources([{"cwd": directory}])
            expected = binding[str(root)]  # bindings are keyed by the canonical checkout path
            cache = package / "__pycache__"
            cache.mkdir()
            (cache / "cli.pyc").write_bytes(b"cache")
            self.assertEqual(verify_source(root, expected), expected["sha256"])
            for change in ("edit", "add", "remove"):
                with self.subTest(change=change):
                    extra = package / "extra.py"
                    if change == "edit":
                        gateway.write_text("original = False\n")
                    elif change == "add":
                        extra.write_text("new = True\n")
                    else:
                        gateway.unlink()
                    with self.assertRaisesRegex(ValueError, "queue_source_changed|gateway_missing"):
                        bind_sources([{"cwd": directory}], binding)
                    gateway.write_text("original = True\n")
                    extra.unlink(missing_ok=True)

    def test_symlink_and_missing_binding_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ni_rua/apps/gateway").mkdir(parents=True)
            (root / "outside.py").write_text("pass\n")
            (root / "ni_rua/apps/gateway/cli.py").symlink_to(root / "outside.py")
            with self.assertRaisesRegex(ValueError, "symlink_denied"):
                bind_sources([{"cwd": directory}])
            with self.assertRaisesRegex(ValueError, "binding_missing"):
                bind_sources([{"cwd": directory}], {})

    def test_checkout_layout_is_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            gateway, module = "ni_rua/apps/gateway/cli.py", "ni_rua.apps.gateway"
            checkout = Path(directory).resolve() / module
            (checkout / gateway).parent.mkdir(parents=True)
            (checkout / gateway).write_text("pass\n")
            self.assertEqual(checkout_layout(checkout)["module"], module)
            self.assertIn(gateway, bind_sources([{"cwd": str(checkout)}])[str(checkout)]["files"])
            self.assertIsNone(checkout_layout(directory))



if __name__ == "__main__":
    unittest.main()
