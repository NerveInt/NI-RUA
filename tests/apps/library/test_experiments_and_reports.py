"""Consolidated tests:

- test_evaluation_reports: Evidence readers preserve corrupt/missing coverage and unknown execution.
- test_experiment_producers: CPU evidence checks: passive reference recording and immutable archive reads.
- test_experiments: Synthetic evidence fixtures; real loopback HTTP/SSE/WS and archive byte checks.
- test_scoring_policy: Task limits are scored failures; native terminals and invalid evidence differ.
"""

import json
from pathlib import Path
import tempfile
import unittest
from ni_rua.core.storage.store import Store
from ni_rua.apps.library.experiments import Catalog, register_run
from ni_rua.apps.evaluation.audit import _owned_terminal_interrupts
from scripts.audit_run import audit
from scripts.benchmark_report import build_report
from copy import deepcopy
import hashlib
import sqlite3
from unittest.mock import patch
from PIL import Image
from ni_rua.apps.library.experiments import allocate_run
from ni_rua.apps.evaluation.recording import ReferenceRecorder
import asyncio
import base64
import fcntl
import shutil
from aiohttp import ClientSession, ContentTypeError, WSMsgType, WSServerHandshakeError, web
from ni_rua.apps.library.experiments import (Catalog, allocate_run, atomic_json, catalog_id, endpoint_origin,
                                 file_hash, finish_run, import_run, register_run, write_runtime)
from ni_rua.apps.library.service import Library
from ni_rua.core.storage.store import Store, open_read_store
from ni_rua.apps.library.scoring import classify_episode_result
from ni_rua.apps.library.scoring import load_effective_summary, reconcile_summary


NI_RUA_TEST_TIER = "component"

# --- test_evaluation_reports.py ---
class EvidenceReportsTest(unittest.TestCase):
    def recovery_fixture(self, root, *, thread_id="main", events=(), raw=(), aliases=()):
        store = Store(root, dict(run_id=root.name, episode_id="episode", env_instance_id="env", env_generation=1))
        for payload in events:
            store.event("benchmark.actor_turn_continuation", payload)
        for original, alias in aliases:
            store.set_state("actor_id", alias, dict(original=original, alias=alias))
        store.close()
        (root / "manifest.json").write_text(json.dumps(dict(run_id=root.name, thread_id=thread_id,
            completion_guidance=dict(recovery_guidance=dict(version="configured-v3", instructions="private prose")))))
        (root / "summary.json").write_text(json.dumps(dict(status="unknown")))
        if raw is not None:
            (root / "codex-raw.jsonl").write_text("".join(json.dumps(value)+"\n" for value in raw))

    @staticmethod
    def recovery_attempt(prompt="recovery", version="actual-v2", thread_id="main"):
        return dict(thread_id=thread_id, prompt=prompt, recovery_context=dict(version=version))

    @staticmethod
    def native_start(prompt, thread_id="main", direction="sent"):
        return dict(direction=direction, message=dict(method="turn/start",
            params=dict(threadId=thread_id, input=[dict(type="text", text=prompt)])))

    def test_recovery_configuration_does_not_count_as_an_actual_continuation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.recovery_fixture(root, raw=[self.native_start("initial task")])
            value = build_report([root])["runs"][0]["recovery_guidance"]
            self.assertEqual(value, dict(configured_version="configured-v3", status="available",
                continuation_attempt_count=0, recovery_attempt_count=0, native_turn_start_count=1,
                native_sent_continuation_count=0, native_sent_recovery_count=0,
                unmatched_continuation_attempt_count=0, versions=[]))

    def test_recovery_event_without_native_submission_remains_an_unmatched_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.recovery_fixture(root, events=[self.recovery_attempt()], raw=[self.native_start("initial task")])
            value = build_report([root])["runs"][0]["recovery_guidance"]
            self.assertEqual(value["recovery_attempt_count"], 1)
            self.assertEqual(value["native_sent_recovery_count"], 0)
            self.assertEqual(value["unmatched_continuation_attempt_count"], 1)
            self.assertEqual(value["versions"], [dict(version="actual-v2", attempt_count=1, native_sent_count=0)])

    def test_recovery_counts_exact_owned_submissions_once_and_restores_saved_aliases(self):
        original = "obs-" + "a" * 32
        prompt = "Recovery context for " + original
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.recovery_fixture(root, aliases=[(original, "obs_7")],
                events=[self.recovery_attempt(prompt), self.recovery_attempt(prompt),
                        self.recovery_attempt("foreign only", thread_id="foreign"),
                        dict(thread_id="main", prompt="generic continuation")],
                raw=[self.native_start("initial task"), self.native_start("Recovery context for obs_7", "foreign"),
                     self.native_start("Recovery context for obs_7", direction="received"),
                     self.native_start("prefix Recovery context for obs_7"),
                     self.native_start("Recovery context for obs_7"), self.native_start("foreign only"),
                     self.native_start("generic continuation")])
            before = {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}
            value = build_report([root])["runs"][0]["recovery_guidance"]
            self.assertEqual(value["continuation_attempt_count"], 3)
            self.assertEqual(value["recovery_attempt_count"], 2)
            self.assertEqual(value["native_turn_start_count"], 5)
            self.assertEqual(value["native_sent_continuation_count"], 2)
            self.assertEqual(value["native_sent_recovery_count"], 1)
            self.assertEqual(value["unmatched_continuation_attempt_count"], 1)
            self.assertEqual(value["versions"], [dict(version="actual-v2", attempt_count=2, native_sent_count=1)])
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()})
            encoded = json.dumps(value)
            for private in (prompt, original, "obs_7", "private prose", "foreign only"):
                self.assertNotIn(private, encoded)

    def test_missing_recovery_native_log_or_main_thread_preserves_unknown_counts(self):
        for thread, raw, status in (("main", None, "native_log_missing"),
                                    (None, [self.native_start("recovery")], "main_thread_unknown")):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.recovery_fixture(root, thread_id=thread, events=[self.recovery_attempt()], raw=raw)
                value = build_report([root])["runs"][0]["recovery_guidance"]
                self.assertEqual(value["status"], status)
                self.assertIsNone(value["native_turn_start_count"])
                self.assertIsNone(value["native_sent_recovery_count"])
                self.assertIsNone(value["unmatched_continuation_attempt_count"])
                if thread:
                    self.assertEqual(value["recovery_attempt_count"], 1)
                    self.assertEqual(value["versions"], [dict(version="actual-v2", attempt_count=1, native_sent_count=None)])
                else:
                    self.assertIsNone(value["recovery_attempt_count"])

    def test_malformed_native_log_is_isolated_instead_of_becoming_zero_recovery_exposure(self):
        bad_records = ("{broken", "null\n", json.dumps(dict(direction="sent", message=[])),
            json.dumps(dict(direction="sent", message=dict(method="turn/start", params=[]))),
            json.dumps(dict(direction="sent", message=dict(method="turn/start", params=dict(threadId="main", input=[None])))))
        for raw in bad_records:
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as directory:
                good, bad = Path(directory)/"good", Path(directory)/"bad"
                self.recovery_fixture(good)
                self.recovery_fixture(bad)
                (bad / "codex-raw.jsonl").write_text(raw)
                report = build_report([good, bad])
                self.assertEqual(report["coverage"]["readable"], 1)
                self.assertEqual(report["coverage"]["unreadable"], 1)
                self.assertNotIn("recovery_guidance", report["runs"][1])
                self.assertIsNone(report["runs"][1]["task_success"])

    def test_native_audit_counts_only_interrupts_bound_to_an_owned_native_terminal(self):
        cases = (
            ("private_evaluator", "native_step_limit", "observer", "turn", "turn", "2026-01-01T00:00:02Z", "2026-01-01T00:00:01Z"),
            ("private_evaluator", "not_native", "observer", "turn", "turn", "2026-01-01T00:00:02Z", "2026-01-01T00:00:01Z"),
            ("actor", "native_step_limit", "observer", "turn", "turn", "2026-01-01T00:00:02Z", "2026-01-01T00:00:01Z"),
            ("private_evaluator", "native_step_limit", "actor", "turn", "turn", "2026-01-01T00:00:02Z", "2026-01-01T00:00:01Z"),
            ("private_evaluator", "native_step_limit", "observer", "turn", "other", "2026-01-01T00:00:02Z", "2026-01-01T00:00:01Z"),
            ("private_evaluator", "native_step_limit", "observer", "turn", "turn", "2026-01-01T00:00:04Z", "2026-01-01T00:00:01Z"),
            ("private_evaluator", "native_step_limit", "observer", "turn", "turn", "2026-01-01T00:00:02Z", "2026-01-01T00:00:03Z"),
            ("private_evaluator", "malformed", "observer", "turn", "turn", "2026-01-01T00:00:02Z", "2026-01-01T00:00:01Z"),
            ("private_evaluator", "native_failure", "observer", "turn", "turn", "2026-01-01T00:00:02Z", "2026-01-01T00:00:01Z"),
            ("private_evaluator", [], "observer", "turn", "turn", "2026-01-01T00:00:02Z", "2026-01-01T00:00:01Z"),
            ("private_evaluator", {}, "observer", "turn", "turn", "2026-01-01T00:00:02Z", "2026-01-01T00:00:01Z"),
        )
        for _ in cases:
            (terminal_source, terminal_reason, terminal_scope, interrupted_turn, interrupt_turn,
             terminal_at, started_at) = _
            with self.subTest(source=terminal_source, reason=terminal_reason,
                              scope=terminal_scope, interrupt_turn=interrupt_turn,
                              terminal_at=terminal_at, started_at=started_at), tempfile.TemporaryDirectory() as directory:
                store = Store(directory, dict(run_id="terminal-interrupt", episode_id="episode",
                                              env_instance_id="environment", env_generation=1))
                terminal = (None if terminal_reason == "malformed" else dict(
                    ended=True, success=False, reason=terminal_reason,
                    native_action_count=10, native_step_limit=10))
                reference = store.put_artifact(json.dumps(terminal).encode(), "native_terminal",
                                               "application/json", terminal_scope)
                store.event("environment.ended", {"result_ref": reference}, source=terminal_source)
                completed_message = {"method": "turn/completed", "params": {
                    "threadId": "thread", "turn": {"id": "turn", "status": "interrupted"}}}
                completed_ref = store.put_artifact(json.dumps(completed_message).encode(), "codex_event",
                                                   "application/json", "observer")
                store.event("codex.turn/completed", {"raw_ref": completed_ref}, {"thread_id": "thread"},
                            source="codex_app_server")
                raw = [
                    {"direction": "received", "received_at": started_at,
                     "message": {"method": "turn/started", "params": {
                         "threadId": "thread", "turn": {"id": "turn", "status": "inProgress"}}}},
                    {"direction": "sent", "received_at": "2026-01-01T00:00:03Z", "message": {"method": "turn/interrupt",
                     "params": {"threadId": "thread", "turnId": interrupt_turn}}},
                    {"direction": "received", "received_at": "2026-01-01T00:00:04Z", "message": {"method": "turn/completed",
                     "params": {"threadId": "thread", "turn": {"id": interrupted_turn,
                                                                       "status": "interrupted"}}}},
                ]
                events = store.events()
                for event in events:
                    if event["type"] == "environment.ended":
                        event["received_at"] = terminal_at
                result = _owned_terminal_interrupts(raw, events, store, "thread")
                valid = (terminal_reason == "native_step_limit" and terminal_source == "private_evaluator"
                         and terminal_scope == "observer" and interrupt_turn == interrupted_turn
                         and terminal_at == "2026-01-01T00:00:02Z" and started_at == "2026-01-01T00:00:01Z")
                self.assertEqual(result, {"turn"} if valid else set())
                store.close()
        self.assertEqual(_owned_terminal_interrupts([None, {"message": []}], [], None, "thread"), set())

    def test_invalid_nested_summary_is_isolated_in_report_and_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good, bad = root / "good", root / "bad"
            for run in (good, bad):
                register_run(run, category="validation", task="fixture")
                Store(run, {}).close()
                (run / "manifest.json").write_text(json.dumps({"run_id": run.name}))
                (run / "summary.json").write_text(json.dumps({"status": "unknown"}))
            for field, value in (("scoring_policy", None), ("scoring_policy", []),
                                 ("scoring_policy", "legacy"), ("native_episode", "invalid")):
                with self.subTest(field=field, value=value):
                    (bad / "summary.json").write_text(json.dumps({"status": "unknown", field: value}))
                    report = build_report([good, bad])
                    self.assertEqual(report["coverage"], dict(requested=2, readable=1, unreadable=1, duplicate_run_ids=[]))
                    self.assertEqual(report["runs"][1]["report_error"]["type"], "ValueError")
                    self.assertIsNone(report["runs"][1]["task_success"])
                    self.assertIsNone(report["runs"][1]["token_accounting"]["tokens"]["total_tokens"])
                    catalog = Catalog(root)
                    catalog.scan()
                    self.assertEqual(catalog.list()["total"], 1)
                    self.assertEqual(catalog.list()["experiments"][0]["run_id"], "good")
                    self.assertEqual(len(catalog.errors), 1)
                    self.assertIn("summary_field_requires_object:" + field, catalog.errors[0]["error"])

    def test_integrity_rejects_dangling_observation_receipt_and_observer_references(self):
        sources = ("rgb_ref", "planned_trace_ref", "dispatched_trace_ref", "actual_trace_ref", "before_state_ref",
                   "evaluation_event", "video_event", "evaluation_file", "video_file")
        for source in sources:
            with self.subTest(source=source), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                identity = dict(run_id="fixture", episode_id="episode", env_instance_id="env", env_generation=1)
                store = Store(root, identity)
                reference = store.put_artifact(b"referenced evidence", "trace", "application/octet-stream",
                    "observer" if source.startswith(("evaluation", "video")) else "actor")
                store.save("observation", "obs", dict(observation_id="obs",
                    cameras=[{"rgb_ref": reference}] if source == "rgb_ref" else []))
                store.save("action_request", "action", dict(based_on_observation_id="obs"))
                store.save("action_receipt", "action", dict(status="unknown",
                    **{source: reference} if source.endswith("_ref") and source != "rgb_ref" else {}))
                if source.startswith(("evaluation", "video")):
                    video = source.startswith("video")
                    value = {"videos": [{"video_ref": reference}]} if video else {"result_ref": reference}
                    if source.endswith("event"):
                        store.event("observer.video_exported" if video else "observer.evaluated", value)
                    else:
                        (root / ("video.json" if video else "evaluation.json")).write_text(json.dumps(value))
                (root / "manifest.json").write_text(json.dumps(identity))
                store.close()
                self.assertTrue(audit(root)["passed"])
                store = Store(root, identity)
                with store.db:
                    store.db.execute("DELETE FROM artifacts WHERE id=?", (reference.removeprefix("artifact:"),))
                store.close()
                result = audit(root)
                self.assertFalse(result["passed"], result)
                self.assertFalse(result["original_bytes_complete"])
                self.assertTrue(any(reference in failure for failure in result["failures"]), result)
                self.assertEqual(result["unresolved_actions"], ["action"])

    def test_integrity_labels_verified_compacted_rgb_without_claiming_original_bytes(self):
        import io
        import random
        from PIL import Image
        from ni_rua.core.storage.image_archive import compact_images
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = Store(root, {})
            image = Image.frombytes("RGB", (128, 96), random.Random(8).randbytes(128 * 96 * 3))
            encoded = io.BytesIO()
            image.save(encoded, format="PNG")
            reference = store.put_artifact(encoded.getvalue(), "rgb", "image/png")
            store.save("observation", "obs", {"cameras": [{"camera_id": "head", "width": 128,
                "height": 96, "rgb_ref": reference}]})
            compacted = compact_images(store)
            self.assertEqual(compacted["originals_removed"], 1)
            video_path = store.artifact(compacted["segments"][0]["video_ref"])[1]
            store.close()
            (root / "manifest.json").write_text(json.dumps({"run_id": "compacted"}))
            result = audit(root)
            self.assertTrue(result["passed"], result)
            self.assertFalse(result["original_bytes_complete"])
            self.assertEqual(result["reconstructed_images"], [reference.removeprefix("artifact:")])
            self.assertEqual(result["verified_artifacts"], 1)
            video_path.unlink()
            self.assertFalse(audit(root)["passed"])

    def test_integrity_reports_invalid_compacted_image_mapping(self):
        for mapping in (["bad mapping"], "bad mapping", 1, False):
            with self.subTest(mapping=mapping), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                store = Store(root, {})
                reference = store.put_artifact(b"original RGB", "rgb", "image/png")
                metadata, path = store.artifact(reference)
                store.save("observation", "obs", {"cameras": [{"rgb_ref": reference}]})
                store.set_state("image_archive", metadata["artifact_id"], mapping)
                path.unlink()
                store.close()
                (root / "manifest.json").write_text(json.dumps({"run_id": "invalid-mapping"}))
                result = audit(root)
                self.assertFalse(result["passed"], result)
                self.assertFalse(result["original_bytes_complete"])
                self.assertEqual(result["reconstructed_images"], [])
                self.assertIn(reference + ":image_archive_mapping_requires_object", result["failures"])

    def test_bad_runs_do_not_hide_good_runs_or_become_zero_cost_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good, bad, missing = [root / name for name in ("good", "corrupt", "missing")]
            with self.subTest(run="good"):
                store = Store(good, {})
                store.close()
                (good / "manifest.json").write_text(json.dumps(dict(run_id="good", model="custom-model")))
                (good / "summary.json").write_text(json.dumps(dict(status="unknown")))
            bad.mkdir()
            (bad / "manifest.json").write_text("{broken")
            before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
            report = build_report([good, bad, missing])
            self.assertEqual(report["coverage"], dict(requested=3, readable=1, unreadable=2, duplicate_run_ids=[]))
            self.assertEqual([row["report_status"] for row in report["runs"]], ["readable", "unreadable", "unreadable"])
            self.assertIsNone(report["runs"][1]["task_success"])
            tokens = report["token_accounting"]["tokens"]["total_tokens"]
            self.assertIsNone(tokens["total"])
            self.assertEqual(tokens["unknown_runs"], 3)
            self.assertFalse(missing.exists())
            after = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
            self.assertEqual(before, after)

    def test_integrity_does_not_require_astra_or_native_logs_and_preserves_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = Store(root, {})
            reference = store.put_artifact(b"evidence", "trace", "application/octet-stream")
            store.save("observation", "obs", dict(observation_id="obs"))
            store.save("action_request", "action", dict(based_on_observation_id="obs"))
            store.save("action_receipt", "action", dict(status="unknown"))
            _, path = store.artifact(reference)
            store.close()
            (root / "manifest.json").write_text(json.dumps(dict(run_id="integrity", model="another-model")))
            result = audit(root)
            self.assertTrue(result["passed"], result)
            self.assertEqual(result["unresolved_actions"], ["action"])
            self.assertFalse(audit(root, profile="native")["passed"])
            self.assertFalse((root / "events.sqlite-wal").exists())
            self.assertFalse((root / "events.sqlite-shm").exists())
            path.write_bytes(b"tampered")
            corrupt = audit(root)
            self.assertFalse(corrupt["passed"])
            self.assertIn("artifact_digest_mismatch", corrupt["failures"][0])


# --- test_experiment_producers.py ---
class ExperimentProducerCheck(unittest.TestCase):
    def test_passive_recording_preserves_actual_receipt_and_zero_task_score(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = allocate_run(temporary, category="reference", task="fixture", title="CPU fixture")
            image = directory / "source.png"
            Image.new("RGB", (1, 1)).save(image)
            raw = {"raw_observation_id": "raw-input", "data_origin": "synthetic_fixture", "observation_profile": "rgb_proprio_v1",
                   "sim_step": 0, "step_domain": "adapter_control_step", "robot_state": {"arms": {}},
                   "cameras": [{"camera_id": "head", "frame_id": "input-frame", "width": 1, "height": 1, "png_path": str(image)}]}
            recorder = ReferenceRecorder(directory)
            try:
                recorder.start({"data_origin": "synthetic_fixture", "observation_profile": "rgb_proprio_v1"})
                first = recorder.observe(raw)
                action = {"mode": "absolute", "path": {"kind": "hold", "waypoints": []}, "timeout_s": 1}
                terminal = deepcopy(raw)
                terminal.update(raw_observation_id="raw-result", sim_step=10)
                terminal["cameras"][0]["frame_id"] = "result-frame"
                receipt = {"status": "completed", "executed_control_steps": 10, "terminal_observation": terminal,
                           "before_observation": raw, "actual_trace": [{"sim_step": 10}], "position_error_m": 0.001,
                           "episode_terminal": {"ended": True, "success": False, "reason": "native_step_limit",
                                                "native_action_count": 800, "native_step_limit": 800}}
                unchanged = deepcopy(receipt)
                calls, updates = [], []
                cancellation = object()
                class Backend:
                    def execute(self, request, *, progress_callback, cancel_flag):
                        calls.append((request, cancel_flag))
                        if request.get("reject"): raise ValueError("synthetic_backend_refusal")
                        progress_callback({"phase": "executing", "episode_terminal": {
                            "ended": False, "success": None, "native_action_count": 799, "native_step_limit": 800}})
                        return receipt
                returned = recorder.execute(Backend(), action, progress_callback=updates.append, cancel_flag=cancellation)
                self.assertIs(returned, receipt)
                self.assertEqual(calls, [(action, cancellation)])
                self.assertEqual(updates[0]["episode_terminal"]["native_action_count"], 799)
                self.assertEqual(receipt, unchanged)
                request = recorder.store.records("action_request")[0]
                recorded = recorder.store.records("action_receipt")[0]
                self.assertEqual(request["based_on_observation_id"], first["observation_id"])
                self.assertEqual(recorded["position_error_m"], receipt["position_error_m"])
                self.assertNotIn("episode_terminal", recorded)
                events = recorder.store.events()
                progress_event = next(item for item in events if item["type"] == "action.progress")
                self.assertNotIn("episode_terminal", json.dumps(progress_event))
                native_event = next(item for item in events if item["type"] == "environment.ended")
                metadata, artifact = recorder.store.artifact(native_event["payload"]["result_ref"])
                self.assertEqual(metadata["access_scope"], "observer")
                self.assertEqual(json.loads(artifact.read_text()), receipt["episode_terminal"])
                self.assertEqual(recorder.store.read("observation", recorded["result_observation_id"])["cameras"][0]["frame_id"], "result-frame")
                self.assertIsNotNone(recorded["actual_trace_ref"])
                with self.assertRaisesRegex(ValueError, "synthetic_backend_refusal"):
                    recorder.execute(Backend(), {**action, "reject": True})
                self.assertEqual(len(calls), 2)
                self.assertEqual(recorder.store.records("action_receipt")[-1]["status"], "unknown")
                recorder.record_results({"evaluation": {"episode_index": 1, "task_success": False, "normalized_score": 0}})
                prior_generation = recorder.identity["env_generation"]
                recorder.identity.update(env_generation=2, episode_id="synthetic-new-episode")
                event = next(item for item in recorder.store.events() if item["type"] == "observer.evaluated")
                self.assertEqual(event["env_generation"], prior_generation)
                recorder.finish({"errors": [], "termination_reason": "done", "loop_verified": True, "action_count": 1,
                                 "llm_calls": 2, "evaluation": {"task_success": False, "normalized_score": 0, "valid_episode": True}}, record_results=False)
                summary = json.loads((directory / "summary.json").read_text())
                self.assertEqual(summary.get("summary", summary)["normalized_score"], 0)
                self.assertIs(summary.get("summary", summary)["task_success"], False)
                runtime = json.loads((directory / "runtime.json").read_text())
                self.assertEqual(runtime["status"], "completed")
                self.assertIsNone(runtime.get("endpoint"))
                self.assertFalse((directory / "terminal.bin").exists())
            finally:
                recorder.close()

    def test_passive_recorder_binds_native_profile_and_keeps_calibration(self):
        from ni_rua.core.observation_policy import ACTOR_PROFILE, LEGACY_ACTOR_PROFILE
        from support.fixture_robot import FixtureRobot
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            robot = FixtureRobot(root)
            recorder = ReferenceRecorder(root/"calibrated", kind="validation")
            try:
                self.assertEqual(recorder.harness.observation_profile, ACTOR_PROFILE)
                self.assertFalse(recorder.harness.motion_review)
                recorder.start({"observation_profile": ACTOR_PROFILE})
                raw = robot.observe()
                raw.update(raw_observation_id="raw-calibrated-fixture", observation_profile=ACTOR_PROFILE,
                    camera_calibration={"cameras": {"fixture_camera": {"K": [[100,0,.5],[0,100,.5],[0,0,1]]}}},
                    camera_pose_sample={"cameras": {"fixture_camera": {"T_world_camera": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}}},
                    robot_model={"base_link": "base", "joints": []})
                bundle = recorder.observe(raw)
                self.assertEqual(bundle["observation_profile"], ACTOR_PROFILE)
                metadata = json.loads(recorder.harness.observation_content(bundle, {}, detail="full")[0]["text"])
                self.assertEqual(metadata["camera_calibration"], raw["camera_calibration"])
                with self.assertRaisesRegex(ValueError, "profile_changed"):
                    recorder.start({"observation_profile": LEGACY_ACTOR_PROFILE})
                recorded = recorder.store.read_state("runtime", "recording_policy")
                self.assertEqual(recorded["observation_profile"], ACTOR_PROFILE)
                self.assertFalse(recorded["motion_review"])
            finally:
                recorder.close()
            explicit = ReferenceRecorder(root/"legacy", observation_profile=LEGACY_ACTOR_PROFILE)
            try:
                with self.assertRaisesRegex(ValueError, "matching_observation_profile"):
                    explicit.start({"observation_profile": ACTOR_PROFILE})
                explicit.start({"observation_profile": LEGACY_ACTOR_PROFILE})
                self.assertEqual(explicit.harness.observation_profile, LEGACY_ACTOR_PROFILE)
            finally:
                explicit.close()

    def test_native_controller_ee_smoke_uses_calibration_without_model_review_gate(self):
        from scripts.probes import native_policy_validate as diagnostic
        from support.fixture_robot import FixtureRobot
        from ni_rua.core.observation_policy import ACTOR_PROFILE
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            class Backend(FixtureRobot):
                observation_profile = ACTOR_PROFILE
                def start(self):
                    return {**super().start(), "observation_profile": ACTOR_PROFILE}
                def observe(self):
                    raw = super().observe()
                    raw.update(observation_profile=ACTOR_PROFILE,
                        camera_calibration={"source": "synthetic calibration"},
                        camera_pose_sample={"source": "synthetic pose"},
                        robot_model={"base_link": "base", "joints": []})
                    return raw
                def preview_motion(self, action):
                    raise AssertionError("fixed controller diagnostic must not claim model review")
                def execute(self, action, progress_callback=None, cancel_flag=None):
                    return {**super().execute(action, progress_callback, cancel_flag),
                            "executed_native_steps": 1, "executed_control_steps": 10}
                def evaluate(self, **kwargs): return {}
                def save_video(self): return {}
                def close(self): pass
            backend = Backend(root/"fixture")
            output = root/"run"
            with patch.object(diagnostic, "RobotBackend", return_value=backend) as constructor, \
                 patch("sys.argv", ["native_policy_validate", "--run-dir", str(output), "--ee-smoke"]):
                diagnostic.main()
            self.assertEqual(constructor.call_args.kwargs["observation_profile"], ACTOR_PROFILE)
            report = json.loads((output/"native-policy-report.json").read_text())
            self.assertEqual(report["status"], "completed")
            self.assertFalse(report["motion_review"])
            self.assertEqual(report["cases"][0]["receipt"]["status"], "completed")
            self.assertEqual(backend.executions, 1)

    def test_read_only_store_sees_wal_and_cannot_write_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            identity = {"run_id": "fixture", "episode_id": "ep", "env_instance_id": "env", "env_generation": 1}
            writer = Store(temporary, identity)
            writer.save("test", "one", {"value": 1})
            reader = Store(temporary, identity, read_only=True)
            try:
                writer.save("test", "two", {"value": 2})
                self.assertEqual(reader.read("test", "two"), {"value": 2})
                for operation in (lambda: reader.save("test", "three", {}), lambda: reader.set_state("x", "x", {}),
                                  lambda: reader.event("unexpected", {}), lambda: reader.put_artifact(b"x", "x", "text/plain"),
                                  lambda: reader.ingest_event({}), lambda: reader.raw_codex("x", {})):
                    with self.assertRaisesRegex(ValueError, "store_is_read_only"):
                        operation()
                with self.assertRaises(sqlite3.OperationalError):
                    reader.db.execute("DELETE FROM records")
                reference = writer.put_artifact(b"fixture secret", "fixture", "text/plain")
                _, blob = writer.artifact(reference)
                outside = Path(temporary) / "outside.txt"
                outside.write_bytes(b"fixture secret")
                blob.unlink()
                blob.symlink_to(outside)
                with self.assertRaisesRegex(ValueError, "artifact_path_outside_store"):
                    reader.resolve_artifact(reference)
            finally:
                reader.close()
                writer.close()
            database = Path(temporary) / "events.sqlite"
            before = hashlib.sha256(database.read_bytes()).hexdigest()
            reader = Store(temporary, identity, read_only=True)
            try:
                self.assertEqual(len(reader.records("test")), 2)
            finally:
                reader.close()
            self.assertEqual(before, hashlib.sha256(database.read_bytes()).hexdigest())


# --- test_experiments.py ---
def hashes(root):
    return {p.relative_to(root).as_posix(): file_hash(p) for p in root.rglob('*') if p.is_file()}


def fixture(path, name):
    identity = dict(run_id=path.name, episode_id='synthetic-episode', env_instance_id='synthetic-env', env_generation=1)
    store = Store(path, identity)
    ref = store.put_artifact(b'synthetic-image-' + name.encode(), 'rgb', 'image/png')
    denied = store.put_artifact(b'private-fixture', 'secret', 'application/octet-stream', 'private')
    observation = dict(schema_version='0.1', **identity, observation_id=name, data_origin='synthetic_fixture', cameras=[dict(rgb_ref=ref)])
    store.save('observation', name, observation)
    store.save('action_request', 'action-' + name, dict(action_id='action-' + name, **identity))
    store.save('action_receipt', 'action-' + name, dict(status='completed', data_origin='synthetic_fixture'))
    store.set_state('runtime', 'identity', identity)
    store.set_state('runtime', 'capabilities', {'data_origin': 'synthetic_fixture'})
    store.event('observation.available', {'observation_id': name, 'data_origin': 'synthetic_fixture'})
    store.close()
    return ref.removeprefix('artifact:'), denied.removeprefix('artifact:')


class CatalogCheck(unittest.TestCase):
    def test_abandoned_run_with_stale_heartbeat_reads_as_unknown_not_running(self):
        # A gateway killed before finalising (e.g. terminal hangup) leaves runtime.json non-terminal and its heartbeat
        # stops. Nothing may present such a run as running or live; its end is unknown.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'catalog'
            stale = allocate_run(root, category='validation', task='pick', run_id='abandoned')
            fresh = allocate_run(root, category='validation', task='pick', run_id='running')
            write_runtime(stale, dict(status='running', pid=1))
            write_runtime(fresh, dict(status='running', pid=1))
            atomic_json(stale / 'runtime.json', {**json.loads((stale / 'runtime.json').read_text()),
                                                 'updated_at': '2020-01-01T00:00:00Z'})
            before = hashes(stale)
            catalog = Catalog(root)
            catalog.scan()
            items = {item['run_id']: item for item in catalog.list()['experiments']}
            self.assertEqual((items['abandoned']['status'], items['abandoned']['live'],
                              items['abandoned']['connection_status']), ('unknown', False, 'heartbeat_stale'))
            self.assertEqual((items['running']['status'], 'connection_status' in items['running']), ('running', False))
            self.assertIsNone(items['abandoned'].get('finished_at'))
            self.assertEqual(hashes(stale), before)   # the raw archive is never rewritten

    def test_run_times_are_evidence_based_utc_and_cached_without_changing_sealed_wal(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / 'catalog'
            run = allocate_run(root, category='validation', task='pick', run_id='sealed')
            identity = dict(run_id=run.name, env_instance_id='fixture', env_generation=1)
            writer = Store(base / 'writer', identity)
            try:
                writer.event('observation.available', {}, occurred_at='2026-10-10T01:00:00Z')
                writer.event('run.started', {}, occurred_at='2026-10-10T09:01:02+08:00')
                writer.event('run.stopped', {}, occurred_at='2026-10-10T01:02:00Z')
                for name in ('events.sqlite', 'events.sqlite-wal'):
                    shutil.copy2(writer.root / name, run / name)
            finally:
                writer.close()
            atomic_json(run / 'summary.json', dict(status='stopped', finished_at='2026-10-10T01:03:04Z'))
            before = {file.name: (file.read_bytes(), file.stat().st_mtime_ns) for file in run.iterdir() if file.is_file()}
            catalog = Catalog(root)
            with patch('ni_rua.apps.library.experiments.open_read_store', wraps=open_read_store) as opened:
                catalog.scan()
                item = catalog.list()['experiments'][0]
                self.assertEqual(item['started_at'], '2026-10-10T01:01:02.000000Z')
                self.assertEqual(item['finished_at'], '2026-10-10T01:03:04.000000Z')
                self.assertEqual(item['started_at_source'], 'events.sqlite:run.started#2')
                self.assertEqual(item['finished_at_source'], 'summary.json:finished_at')
                self.assertNotEqual(item['registered_at'], item['started_at'])
                self.assertEqual(item['time_coverage'], dict(run_count=1, started_count=1, finished_count=1))
                self.assertEqual(item['time_warnings'], [])
                self.assertFalse(catalog.scan())
                self.assertEqual(opened.call_count, 1)
                self.assertEqual(opened.call_args.kwargs, dict(live=False))
            self.assertEqual(before, {file.name: (file.read_bytes(), file.stat().st_mtime_ns) for file in run.iterdir() if file.is_file()})

    def test_import_times_come_from_original_report_not_import_lifecycle_or_mtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = allocate_run(root, category='legacy', task='pick', run_id='imported')
            metadata = json.loads((run / 'experiment.json').read_text())
            metadata.update(imported_from='/original', evidence_root='evidence', date_source='report_mtime',
                            created_at='2026-10-12T00:00:00Z', imported_at='2026-10-14T00:00:00Z')
            atomic_json(run / 'experiment.json', metadata)
            atomic_json(run / 'runtime.json', dict(finished_at='2026-10-14T00:00:01Z', status='imported'))
            atomic_json(run / 'summary.json', dict(finished_at='2026-10-14T00:00:01Z', status='imported'))
            atomic_json(run / 'evidence' / 'validation-report.json', dict(
                started_at='2026-09-30T23:59:58-04:00', finished_at='2026-10-01T00:00:03-04:00'))
            catalog = Catalog(root); catalog.scan()
            item = catalog.list()['experiments'][0]
            self.assertEqual(item['started_at'], '2026-10-01T03:59:58.000000Z')
            self.assertEqual(item['finished_at'], '2026-10-01T04:00:03.000000Z')
            self.assertEqual(item['started_at_source'], 'evidence/validation-report.json:started_at')
            self.assertIsNone(item['registered_at'])
            self.assertEqual(item['imported_at'], '2026-10-14T00:00:00.000000Z')
            atomic_json(run / 'evidence' / 'experiment.json', dict(created_at='2026-09-30T23:59:50-04:00'))
            catalog.scan()
            self.assertEqual(catalog.list()['experiments'][0]['registered_at'], '2026-10-01T03:59:50.000000Z')

    def test_missing_invalid_and_reversed_times_remain_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, report in (
                ('missing', {}), ('naive', dict(started_at='2026-10-14T00:00:00', finished_at='bad')),
                ('epoch', dict(started_at=1791538189.6109202, finished_at=True)),
                ('reversed', dict(started_at='2026-10-14T01:00:00Z', finished_at='2026-10-14T00:00:00Z'))):
                run = allocate_run(root, category='validation', task='pick', run_id=name)
                atomic_json(run / 'summary.json', dict(report, elapsed_wall_s=100))
            catalog = Catalog(root); catalog.scan()
            items = {item['run_id']: item for item in catalog.list()['experiments']}
            self.assertEqual(catalog.errors, [])
            for name in ('missing', 'naive', 'epoch'):
                self.assertIsNone(items[name]['started_at'])
                self.assertIsNone(items[name]['finished_at'])
            self.assertTrue(items['naive']['time_warnings'])
            self.assertIsNone(items['reversed']['finished_at'])
            self.assertIsNone(items['reversed']['finished_at_source'])
            self.assertIn('finished_before_started:summary.json:finished_at', items['reversed']['time_warnings'])

    def test_start_sort_and_date_filter_use_execution_time_with_unknown_last(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, start in (('early', '2026-10-10T15:59:59Z'), ('late', '2026-10-10T16:00:00Z'), ('unknown', None)):
                run = allocate_run(root, category='validation', task='pick', run_id=name)
                if start:
                    atomic_json(run / 'summary.json', dict(started_at=start))
            catalog = Catalog(root); catalog.scan()
            self.assertEqual([item['run_id'] for item in catalog.list()['experiments']], ['late', 'early', 'unknown'])
            self.assertEqual([item['run_id'] for item in catalog.list(dict(sort='oldest'))['experiments']], ['early', 'late', 'unknown'])
            items = catalog.list(dict(date_from='2026-10-11', date_to='2026-10-11'))['experiments']
            self.assertEqual([item['run_id'] for item in items], ['late'])

    def test_event_time_cache_tracks_live_wal_changes_and_ignores_unrelated_events(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = allocate_run(root, category='validation', task='pick', run_id='active')
            writer = Store(run, dict(run_id=run.name, env_instance_id='fixture', env_generation=1))
            try:
                writer.event('observation.available', {}, occurred_at='2026-10-10T01:00:00Z')
                write_runtime(run, dict(status='running'))
                catalog = Catalog(root)
                with patch('ni_rua.apps.library.experiments.open_read_store', wraps=open_read_store) as opened:
                    catalog.scan()
                    self.assertIsNone(catalog.list()['experiments'][0]['started_at'])
                    self.assertTrue(opened.call_args.kwargs['live'])
                    writer.event('run.started', {}, occurred_at='2026-10-10T01:01:02Z')
                    catalog.scan()
                    self.assertEqual(catalog.list()['experiments'][0]['started_at'], '2026-10-10T01:01:02.000000Z')
                    self.assertEqual(opened.call_count, 2)
                    atomic_json(run / 'summary.json', dict(finished_at='2026-10-10T01:02:03Z'))
                    catalog.scan()
                    self.assertEqual(opened.call_count, 2)
                    self.assertEqual(catalog.list()['experiments'][0]['finished_at'], '2026-10-10T01:02:03.000000Z')
            finally:
                writer.close()

    def test_wrong_run_events_cannot_supply_times_and_stop_event_is_not_finish(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, identity_run in (('correct', 'correct'), ('wrong', 'different-run')):
                run = allocate_run(root, category='validation', task='pick', run_id=name)
                writer = Store(run, dict(run_id=identity_run, env_instance_id='fixture', env_generation=1))
                writer.event('run.started', {}, occurred_at='2026-10-10T01:00:00Z')
                writer.event('run.stopped', {}, occurred_at='2026-10-10T02:00:00Z')
                writer.close()
            catalog = Catalog(root); catalog.scan()
            items = {item['run_id']: item for item in catalog.list()['experiments']}
            self.assertEqual(items['correct']['started_at'], '2026-10-10T01:00:00.000000Z')
            self.assertIsNone(items['correct']['finished_at'])
            self.assertIsNone(items['wrong']['started_at'])
            self.assertIsNone(items['wrong']['finished_at'])
            self.assertIn('run_started_identity_conflict', items['wrong']['time_warnings'])

    def test_discover_custom_layouts_preserving_ids_and_run_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            canonical = allocate_run(root, category='main', task='pick', run_id='canonical')
            benchmark = root / 'benchmark' / 'group' / 'batch' / 'runs' / 'case'
            experiment = root / 'experiments' / 'image-interface'
            for path in (benchmark, experiment):
                register_run(path, category='validation', task='pick')
            # Archived originals and private metadata must not become extra runs.
            for relative in ('evidence/original', 'blobs/fixture', 'private/nested'):
                register_run(benchmark / relative, category='legacy', task='pick')
            expected = {catalog_id(path.relative_to(root).as_posix()): path
                        for path in (canonical, benchmark, experiment)}
            catalog = Catalog(root)
            self.assertTrue(catalog.scan())
            self.assertEqual({key: entry['path'] for key, entry in catalog.entries.items()}, expected)
            self.assertEqual(catalog.errors, [])
            self.assertFalse(catalog.scan())
            self.assertEqual(set(catalog.entries), set(expected))
            # New externally located runs are picked up on subsequent scans.
            later = root / 'benchmark' / 'group' / 'next-batch' / 'runs' / 'case'
            register_run(later, category='benchmark', task='pick')
            self.assertTrue(catalog.scan())
            self.assertEqual(catalog.list()['total'], 4)
            self.assertEqual(catalog.get(catalog_id(later.relative_to(root).as_posix()))['path'], later)

    def test_discovery_excludes_private_paths_and_symlink_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            root = base / 'catalog'
            valid = root / 'experiments' / 'visible'
            external = base / 'external'
            for path in (valid, external, root / 'private' / 'hidden',
                         root / 'experiments' / '.aws' / 'hidden',
                         root / 'experiments' / 'credentials.json' / 'hidden',
                         root / 'experiments' / '.env' / 'hidden'):
                register_run(path, category='validation', task='pick')
            (root / 'external-link').symlink_to(external, target_is_directory=True)
            (root / 'internal-link').symlink_to(valid, target_is_directory=True)
            linked_metadata = root / 'experiments' / 'linked-metadata'
            linked_metadata.mkdir()
            (linked_metadata / 'experiment.json').symlink_to(external / 'experiment.json')
            catalog = Catalog(root)
            catalog.scan()
            self.assertEqual([entry['path'] for entry in catalog.entries.values()], [valid])
            self.assertEqual(catalog.errors, [dict(path='experiments/linked-metadata/experiment.json',
                                                 error='symlink_metadata_denied')])

    def test_catalog_import_validation_and_preserve_source(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            first = allocate_run(base / 'catalog', category='reference', task='stack_bowls', title='Synthetic first', tags=['cpu'], run_id='one')
            write_runtime(first, dict(status='running', source='recorded_stream'))
            finish_run(first, status='completed', summary=dict(task_success=False, normalized_score=0))
            catalog = Catalog(base / 'catalog')
            self.assertTrue(catalog.scan())
            listing = catalog.list({'category': 'reference', 'tag': 'cpu', 'q': 'synthetic'})
            self.assertEqual(listing['total'], 1)
            self.assertIs(listing['experiments'][0]['summary']['task_success'], False)
            self.assertEqual(listing['experiments'][0]['summary']['score'], 0)
            self.assertEqual(catalog.list({'task': 'missing'})['total'], 0)
            stable = catalog.list()['experiments'][0]['id']
            catalog.scan()
            self.assertEqual(stable, catalog.list()['experiments'][0]['id'])
            for bad in ('../x', 'x/y', '/tmp/x', '.', '..', 'x\\y'):
                with self.assertRaises(ValueError): allocate_run(base / 'catalog', category=bad, task='stack_bowls')
            for bad in ('http://example.com:42', 'http://127.0.0.1:42/private', 'http://user@127.0.0.1:42', 'http://127.0.0.1:42/?target=http://other'):
                with self.assertRaises(ValueError): endpoint_origin(bad)
            source = base / 'legacy'; source.mkdir(); fixture(source, 'legacy-observation')
            (source / 'private' / 'codex-home').mkdir(parents=True)
            (source / 'private' / 'codex-home' / 'auth.json').write_text('private credential fixture')
            (source / 'link').symlink_to(source / 'events.sqlite')
            (source / 'reference-report.json').write_text(json.dumps(dict(passed=True, evaluation=dict(task_success=False, normalized_score=0))))
            with (source / 'gateway.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(ValueError, 'unowned'): import_run(source, base / 'catalog')
            before = hashes(source)
            copied = import_run(source, base / 'catalog', category='reference', task='stack_bowls')
            self.assertEqual(before, hashes(source))
            inventory = json.loads((copied / 'import-inventory.json').read_text())
            self.assertIn('private/codex-home/auth.json', inventory['excluded']); self.assertIn('link', inventory['excluded'])
            self.assertFalse((copied / 'evidence' / 'private' / 'codex-home' / 'auth.json').exists())
            self.assertEqual(file_hash(source / 'events.sqlite'), file_hash(copied / 'evidence' / 'events.sqlite'))
            catalog.scan(); self.assertEqual(catalog.list()['total'], 2)


class LibraryCheck(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.first = allocate_run(self.root, category='main', task='stack_bowls', run_id='synthetic-first')
        self.second = allocate_run(self.root, category='reference', task='stack_bowls', run_id='synthetic-second')
        self.artifact, self.denied = fixture(self.first, 'obs-first'); fixture(self.second, 'obs-second')
        (self.first / 'terminal.bin').write_bytes(b'synthetic terminal bytes')
        atomic_json(self.first / 'terminal.json', dict(session_id='synthetic-pty', cols=80, rows=24, exit_code=0))
        self.before = hashes(self.root); self.library = Library(self.root, scan_interval=.1); self.runners = []
        self.url = await self.start_server(self.library.app()); self.client = ClientSession()
        self.id1 = catalog_id(self.first.relative_to(self.root).as_posix()); self.id2 = catalog_id(self.second.relative_to(self.root).as_posix())

    async def start_server(self, app):
        runner = web.AppRunner(app, shutdown_timeout=.2); await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0); await site.start(); self.runners.append(runner)
        return f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}'

    async def asyncTearDown(self):
        await self.client.close()
        for runner in reversed(self.runners): await runner.cleanup()

    async def get(self, path, **kwargs):
        async with self.client.get(self.url + path, **kwargs) as response:
            try: value = await response.json()
            except (ValueError, ContentTypeError): value = {'error': await response.text()}
            return response.status, value

    async def sse_event(self, response):
        lines = []
        async with asyncio.timeout(3):
            while True:
                line = await response.content.readline()
                if not line: raise AssertionError('SSE ended before expected event')
                if line == b'\n' and lines:
                    if any(line.startswith(b'data:') for line in lines): return b''.join(lines)
                    lines = []
                else: lines.append(line)

    async def test_malformed_metadata_and_symlinks_cannot_leak_other_runs(self):
        runtime = self.first / 'runtime.json'; runtime.write_text('[]')
        await self.library.refresh()
        status, listing = await self.get('/v1/experiments'); self.assertEqual(status, 200)
        self.assertEqual(listing['total'], 1); self.assertTrue(listing['errors']); runtime.unlink()
        secret = self.second / 'auth.json'; secret.write_text(json.dumps(dict(secret='synthetic-private-value')))
        runtime.symlink_to(secret)
        await self.library.refresh()
        status, listing = await self.get('/v1/experiments'); self.assertEqual(status, 200); self.assertEqual(listing['total'], 1)
        self.assertNotIn('synthetic-private-value', json.dumps(listing)); runtime.unlink()
        database = self.first / 'events.sqlite'; held_database = self.first / 'events-held.sqlite'
        database.rename(held_database); database.symlink_to(self.second / 'events.sqlite')
        status, value = await self.get(f'/runs/{self.id1}/v1/observations/latest')
        self.assertIn(status, (400,403,404)); self.assertNotIn('obs-second', json.dumps(value))
        database.unlink(); held_database.rename(database)
        await self.library.refresh()
        terminal = self.first / 'terminal.bin'; held_terminal = self.first / 'terminal-held.bin'
        terminal.rename(held_terminal); terminal.symlink_to(secret)
        try: ws = await self.client.ws_connect(self.url + f'/runs/{self.id1}/v1/terminal')
        except WSServerHandshakeError as error: self.assertIn(error.status, (400,403,404))
        else:
            await ws.close(); self.fail('terminal symlink accepted')
        terminal.unlink(); held_terminal.rename(terminal)

    async def test_malformed_nested_summary_does_not_block_library_startup(self):
        atomic_json(self.first / "summary.json", dict(status="unknown", scoring_policy=None))
        library = Library(self.root)
        url = await self.start_server(library.app())
        async with self.client.get(url + "/v1/experiments") as response:
            self.assertEqual(response.status, 200)
            listing = await response.json()
        self.assertEqual(listing["total"], 1)
        self.assertEqual(listing["experiments"][0]["id"], self.id2)
        self.assertEqual(len(listing["errors"]), 1)
        self.assertIn("summary_field_requires_object:scoring_policy", listing["errors"][0]["error"])

    async def test_experiment_groups_are_readonly_published_snapshots(self):
        manifest = self.root / 'benchmark' / 'fixture' / 'manifest.json'
        atomic_json(manifest, dict(method='fixture', protocol_cases=[dict(task='stack_bowls', variant='standard',
            eval_seed=0, layout_id=0)], cases=[dict(task='stack_bowls', seed=0, layout_id=0,
            cwd='/fixture-source', run_dir=str(self.first))]))
        await self.library.refresh()
        before = hashes(self.root)
        status, value = await self.get('/v1/experiment-groups')
        self.assertEqual(status, 200)
        self.assertEqual(value['groups'][0]['cases'][0]['run']['id'], self.id1)
        self.assertEqual(value['ungrouped_count'], 1)
        self.assertEqual(value['revision'], self.library.revision)
        self.assertIsNotNone(value['observed_at'])
        self.assertEqual(before, hashes(self.root))
        async with self.client.post(self.url + '/v1/experiment-groups', json={}) as response:
            self.assertEqual(response.status, 405)

    async def test_readonly_replay_isolated_range_scope_and_sha(self):
        reader = Store(self.first, {}, read_only=True, immutable=True)
        try:
            self.assertEqual(reader.records('observation')[0]['observation_id'], 'obs-first')
            with self.assertRaisesRegex(ValueError, 'read_only'): reader.event('forbidden', {})
        finally: reader.close()
        prefix = f'/runs/{self.id1}/v1/'
        status, observation = await self.get(prefix + 'observations/latest')
        self.assertEqual(status, 200); self.assertEqual(observation['observation_id'], 'obs-first')
        self.assertEqual((await self.get(prefix + 'observations/obs-second'))[0], 404)
        self.assertEqual((await self.get(prefix + 'runs/synthetic-second'))[0], 404)
        status, run = await self.get(prefix + 'run')
        self.assertTrue(run['read_only']); self.assertFalse(run['writable']); self.assertTrue(run['terminal_available'])
        async with self.client.post(self.url + prefix + 'control', json=dict(request_id='no-write', command='prompt')) as response: self.assertEqual(response.status,403)
        async with self.client.get(self.url + prefix + 'artifacts/' + self.artifact, headers={'Range':'bytes=0-8'}) as response:
            self.assertEqual(response.status,206); self.assertEqual(await response.read(), b'synthetic'); self.assertIn('X-Artifact-SHA256',response.headers)
        self.assertEqual((await self.get(prefix + 'artifacts/' + self.denied))[0],400)
        async with self.client.get(self.url + prefix + 'events') as response:
            event = await self.sse_event(response); self.assertIn(b'obs-first',event); self.assertNotIn(b'obs-second',event)
        async with self.client.ws_connect(self.url + prefix + 'terminal') as ws:
            ready=await ws.receive_json(); output=await ws.receive_json()
            self.assertEqual(ready['type'],'ready'); self.assertEqual(base64.b64decode(output['data']),b'synthetic terminal bytes')
        status,detail=await self.get(f'/v1/experiments/{self.id1}'); self.assertEqual(status,200); self.assertTrue(detail['replay_available'])
        self.assertEqual(self.before,hashes(self.root))
        for headers in ({'Host':'attacker.example'},{'Origin':'http://attacker.example'}):
            async with self.client.get(self.url + prefix + 'run',headers=headers) as response: self.assertEqual(response.status,403)
        async with self.client.get(self.url + f'/v1/experiments/{self.id1}/files/../../private/auth.json') as response: self.assertEqual(response.status,404)

    async def test_auto_discovery_recorded_stream_update_and_terminal(self):
        async with self.client.get(self.url + '/v1/experiments/events') as response:
            self.assertIn(b'event: snapshot',await self.sse_event(response))
            third=allocate_run(self.root,category='reference',task='pick',title='Synthetic live stream'); fixture(third,'obs-third')
            write_runtime(third,dict(status='running',source='recorded_stream'))
            self.assertIn(b'event: changed',await self.sse_event(response))
            status,listing=await self.get('/v1/experiments?category=reference&task=pick&sort=active'); self.assertEqual(listing['total'],1)
            item=listing['experiments'][0]; self.assertTrue(item['live']); self.assertTrue(item['read_only']); self.assertFalse(item['terminal_available'])
            prefix=item['view_url']+'v1/'
            async with self.client.get(self.url+prefix+'events?after=1') as stream:
                store=Store(third,dict(run_id=third.name,episode_id='synthetic-episode',env_instance_id='synthetic-env',env_generation=1))
                try:
                    store.event('fixture.updated',dict(data_origin='synthetic_fixture')); self.assertIn(b'fixture.updated',await self.sse_event(stream))
                finally: store.close()
            finish_run(third,status='completed',summary=dict(task_success=False,normalized_score=0))
            await self.library.refresh()
            status,run=await self.get(prefix+'run'); self.assertFalse(run['library']['live']); self.assertTrue(run['read_only']); self.assertEqual(run['view_mode'],'replay')

    async def test_verified_http_sse_ws_proxy_origin_and_fallback(self):
        state={'run_id':self.first.name,'origins':[],'run_headers':[]}; app=web.Application()
        async def run(request): return web.json_response(dict(run_id=state['run_id'],status='running',terminal=dict(session_id='synthetic-live-pty'),data_origin='synthetic_fixture'))
        async def control(request):
            state['origins'].append(request.headers.get('Origin')); state['run_headers'].append(request.headers.get('X-Ni-Rua-Run-ID'))
            return web.json_response(dict(status='synthetic_ack',**await request.json()))
        async def events(request):
            response=web.StreamResponse(headers={'Content-Type':'text/event-stream'}); await response.prepare(request)
            await response.write(b'id: 9\ndata: {"type":"synthetic.live","data_origin":"synthetic_fixture"}\n\n'); return response
        async def terminal(request):
            state['origins'].append(request.headers.get('Origin')); state['run_headers'].append(request.headers.get('X-Ni-Rua-Run-ID'))
            ws=web.WebSocketResponse(); await ws.prepare(request); await ws.send_json(dict(type='ready',status=dict(session_id='synthetic-live-pty')))
            async for message in ws:
                if message.type==WSMsgType.TEXT: await ws.send_json(dict(type='synthetic_resize_ack',value=json.loads(message.data)))
            return ws
        app.router.add_get('/v1/run',run); app.router.add_post('/v1/control',control); app.router.add_get('/v1/events',events); app.router.add_get('/v1/terminal',terminal)
        backend=await self.start_server(app); write_runtime(self.first,dict(status='running',endpoint=backend)); prefix=f'/runs/{self.id1}/v1/'
        await self.library.refresh()
        status,value=await self.get(prefix+'run'); self.assertTrue(value['library']['live']); self.assertTrue(value['writable'])
        async with self.client.post(self.url+prefix+'control',headers={'Origin':self.url},json=dict(request_id='synthetic-control',command='prompt')) as response: self.assertEqual(response.status,200)
        self.assertEqual(state['origins'][-1],backend); self.assertEqual(state['run_headers'][-1],self.first.name)
        async with self.client.get(self.url+prefix+'events') as response: self.assertIn(b'synthetic.live',await self.sse_event(response))
        async with self.client.ws_connect(self.url+prefix+'terminal',headers={'Origin':self.url}) as ws:
            self.assertEqual((await ws.receive_json())['type'],'ready'); await ws.send_json(dict(type='resize',cols=90,rows=30))
            self.assertEqual((await ws.receive_json())['type'],'synthetic_resize_ack')
            await ws.send_json(dict(type='scroll',lines=-3))
            scroll_ack=await ws.receive_json()
            self.assertEqual(scroll_ack,dict(type='synthetic_resize_ack',value=dict(type='scroll',lines=-3)))
            await ws.send_json(dict(type='scroll',lines=99))
            self.assertEqual(await ws.receive_json(),dict(type='error',message='invalid_terminal_scroll'))
            await ws.send_json(dict(type='input',data='forbidden'))
            self.assertEqual((await ws.receive_json())['type'],'error')
        self.assertEqual(state['origins'][-1],backend); self.assertEqual(state['run_headers'][-1],self.first.name)
        state['run_id']='synthetic-other-run'; await self.library.refresh()
        status,value=await self.get(prefix+'run'); self.assertFalse(value['library']['live'])
        self.assertEqual((await self.get(prefix+'observations/latest'))[1]['observation_id'],'obs-first')
        async with self.client.ws_connect(self.url+prefix+'terminal',headers={'Origin':self.url}) as replay_ws:
            while (await replay_ws.receive_json())['type'] != 'status':
                pass
            for bad_replay_msg in (None, [], True, 'str', dict(type='scroll', lines=-2)):
                await replay_ws.send_str(json.dumps(bad_replay_msg))
                self.assertEqual(await replay_ws.receive_json(), dict(type='error', message='replay_is_read_only'))
        async with self.client.post(self.url+prefix+'control',json=dict(request_id='blocked',command='prompt')) as response: self.assertEqual(response.status,403)


# --- test_scoring_policy.py ---
TASK_LIMITS = (
    "model_request_limit", "official_startup_no_action_timeout",
    "official_no_action_progress_timeout", "actor_repeated_inability", "actor_no_progress",
    "development_wall_budget",
)


class TaskLimitScoringChecks(unittest.TestCase):
    def snapshot(self, reason="official_no_action_progress_timeout", **changes):
        result = dict(benchmark=True, status="stopped", cleanup_status="completed",
            termination_reason=reason, valid_episode=True, task_success=False,
            normalized_score=.4, native_process_score=40., native_reward=0.,
            native_episode=dict(ended=False, native_action_count=641, native_step_limit=700),
            infrastructure_error=None,
            authority="installed RoboDojo task.run_reward / RewardManager",
            success_authority="native_final_check")
        result.update(changes)
        return result

    def assert_failure(self, result, score, process):
        self.assertEqual(result["primary_outcome"], "scored_failure")
        self.assertIs(result["scored_episode"], True)
        self.assertIs(result["task_success"], False)
        self.assertEqual(result["normalized_score"], score)
        self.assertEqual(result["native_process_score"], process)
        self.assertEqual(result["termination_authority"], "harness")
        self.assertIs(result["native_episode"]["ended"], False)

    def test_every_task_limit_retains_zero_and_partial_process_score_as_failure(self):
        for reason in TASK_LIMITS:
            for score, process in ((0., 0.), (.4, 40.)):
                with self.subTest(reason=reason, score=score):
                    original = self.snapshot(reason, normalized_score=score, native_process_score=process)
                    saved = deepcopy(original)
                    result = classify_episode_result(original, "stopped")
                    self.assert_failure(result, score, process)
                    self.assertEqual(original, saved, "effective scoring must not change source evidence")

    def test_development_wall_budget_also_scores_outside_benchmark_mode(self):
        result = classify_episode_result(self.snapshot("development_wall_budget", benchmark=False), "stopped")
        self.assertEqual(result["primary_outcome"], "scored_failure")
        self.assertFalse(result["task_success"])
        self.assertEqual(result["normalized_score"], .4)

    def test_success_snapshot_without_native_terminal_does_not_satisfy_task_completion(self):
        for reason in TASK_LIMITS:
            with self.subTest(reason=reason):
                result = classify_episode_result(self.snapshot(reason, task_success=True,
                    normalized_score=1., native_process_score=40., native_reward=1.), "stopped")
                self.assert_failure(result, .4, 40.)
                self.assertTrue(result["native_evaluation_snapshot"]["task_success"])
                self.assertEqual(result["native_evaluation_snapshot"]["normalized_score"], 1.)
                self.assertEqual(result["success_authority"], "harness_task_limit")
                self.assertEqual(result["failure_reason"]["code"], reason)
                self.assertEqual(result["failure_reason"]["authority"], "harness")
                self.assertTrue(result["failure_reason"]["label"])

    def test_binary_native_evaluation_with_explicitly_absent_process_scores_zero_failure(self):
        result = classify_episode_result(self.snapshot(task_success=True, normalized_score=1.,
                                                        native_process_score=None), "stopped")
        self.assert_failure(result, 0., None)
        self.assertTrue(result["native_evaluation_snapshot"]["task_success"])

    def test_missing_process_field_cannot_derive_failure_score_from_raw_success(self):
        original = self.snapshot(task_success=True, normalized_score=1.)
        del original["native_process_score"]
        result = classify_episode_result(original, "stopped")
        self.assertFalse(result["scored_episode"])
        self.assertIsNone(result["normalized_score"])
        self.assertFalse(result["task_success"])
        self.assertEqual(result["task_outcome"], "failure")

    def test_native_terminal_success_wins_racing_task_limit(self):
        for reason in TASK_LIMITS:
            with self.subTest(reason=reason):
                result = classify_episode_result(self.snapshot(reason, task_success=True,
                    normalized_score=1., native_process_score=40.,
                    native_episode=dict(ended=True, success=True, reason="native_success")), "stopped")
                self.assertEqual(result["primary_outcome"], "scored_success")
                self.assertTrue(result["task_success"])
                self.assertEqual(result["normalized_score"], 1.)
                self.assertEqual(result["native_process_score"], 40.)
                self.assertEqual(result["termination_authority"], "robodojo_environment")

    def test_native_terminal_failure_wins_racing_task_limit(self):
        for reason in TASK_LIMITS:
            with self.subTest(reason=reason):
                result = classify_episode_result(self.snapshot(reason,
                    native_episode=dict(ended=True, success=False, reason="native_step_limit")), "stopped")
                self.assertEqual(result["primary_outcome"], "scored_failure")
                self.assertFalse(result["task_success"])
                self.assertEqual(result["normalized_score"], .4)
                self.assertEqual(result["termination_authority"], "robodojo_environment")

    def test_native_terminal_result_survives_unknown_cleanup_status(self):
        for success, score in ((True, 1.), (False, .4)):
            with self.subTest(success=success):
                result = classify_episode_result(self.snapshot(task_success=success, normalized_score=score,
                    infrastructure_error="cleanup resource release failed",
                    native_episode=dict(ended=True, success=success,
                                        reason="native_success" if success else "native_step_limit")), "unknown")
                self.assertTrue(result["scored_episode"])
                self.assertIs(result["task_success"], success)
                self.assertEqual(result["normalized_score"], score)
                self.assertEqual(result["termination_authority"], "robodojo_environment")

    def test_infrastructure_failure_and_unknown_state_are_never_manufactured_failures(self):
        for reason in TASK_LIMITS:
            for status, changes in (
                ("unknown", {}), ("stopped", {"infrastructure_error": "robot transport disconnected"}),
                ("failed", {"infrastructure_error": "model stream disconnected"}),
            ):
                with self.subTest(reason=reason, status=status, changes=changes):
                    result = classify_episode_result(self.snapshot(reason, **changes), status)
                    self.assertEqual(result["primary_outcome"], "unscored_infrastructure")
                    self.assertFalse(result["scored_episode"])
                    self.assertIsNone(result["normalized_score"])
                    self.assertIsNone(result["task_success"])
                    self.assertEqual(result["partial_native_result"]["normalized_score"], .4)

    def test_invalid_native_evaluation_cannot_create_a_task_score(self):
        changes = [{"valid_episode": value} for value in (False, None, 1, "true")]
        changes += [{"task_success": value} for value in (None, 0, "false")]
        changes += [{"normalized_score": value} for value in
                    (None, False, "0.4", float("nan"), float("inf"), -.1, 1.1)]
        for reason in TASK_LIMITS:
            for invalid in changes:
                with self.subTest(reason=reason, invalid=invalid):
                    result = classify_episode_result(self.snapshot(reason, **invalid), "stopped")
                    self.assertFalse(result["scored_episode"])
                    self.assertIsNone(result["normalized_score"])
                    if "valid_episode" in invalid:
                        self.assertIsNone(result["task_success"])
                        self.assertEqual(result["task_outcome"], "unknown")
                    else:
                        self.assertIs(result["task_success"], False)
                        self.assertEqual(result["task_outcome"], "failure")
                        self.assertEqual(result["score_status"], "unavailable")
                        self.assertEqual(result["failure_reason"]["code"], reason)
            original = self.snapshot(reason)
            del original["normalized_score"]
            self.assertFalse(classify_episode_result(original, "stopped")["scored_episode"])
            self.assertFalse(classify_episode_result(original, "stopped")["task_success"])

    def test_external_shutdown_without_task_limit_stays_unscored(self):
        result = classify_episode_result(self.snapshot("shutdown-signal"), "stopped")
        self.assertEqual(result["primary_outcome"], "unscored_interrupted")
        self.assertFalse(result["scored_episode"])
        self.assertIsNone(result["normalized_score"])
        self.assertIsNone(result["task_success"])


class HistoricalScoringReconciliationChecks(unittest.TestCase):
    def historical(self, reason="official_no_action_progress_timeout", **changes):
        raw = dict(valid_episode=True, task_success=False, normalized_score=.4,
                   native_process_score=40., native_reward=0.,
                   success_authority="native_final_check",
                   authority="installed RoboDojo task.run_reward / RewardManager")
        summary = dict(benchmark=True, status="stopped", cleanup_status="completed",
            termination_reason=reason, termination_authority="harness", infrastructure_error=None,
            native_episode=dict(ended=False, native_action_count=641, native_step_limit=700),
            valid_episode=True, primary_outcome="unscored_harness_timeout", scored_episode=False,
            task_success=None, normalized_score=None, native_process_score=40.,
            success_authority="native_final_check", partial_native_result=raw)
        summary.update(changes)
        return summary

    def test_verified_original_bottle_snapshot_becomes_effective_failure_without_source_rewrite(self):
        original = self.historical()
        saved = deepcopy(original)
        result = reconcile_summary(original, evidence_verified=True)
        self.assertEqual(original, saved)
        self.assertEqual(result["primary_outcome"], "scored_failure")
        self.assertTrue(result["scored_episode"])
        self.assertFalse(result["task_success"])
        self.assertEqual(result["normalized_score"], .4)
        self.assertEqual(result["native_process_score"], 40.)
        self.assertFalse(result["native_episode"]["ended"])
        self.assertEqual(result["termination_authority"], "harness")
        self.assertEqual(result["partial_native_result"], saved["partial_native_result"])
        self.assertEqual(result["scoring_policy"]["version"], "native_terminal_and_task_limits_v3")
        before = result["scoring_reconciliation"]["original"]
        for key in ("primary_outcome", "scored_episode", "task_success", "normalized_score", "success_authority"):
            self.assertEqual(before[key], saved[key])
        self.assertEqual(result["failure_reason"]["code"], "official_no_action_progress_timeout")

    def test_verified_evaluation_envelope_overrides_older_partial_snapshot(self):
        original = self.historical()
        original["partial_native_result"] = dict(valid_episode=True, task_success=False,
                                                normalized_score=0., native_process_score=0.)
        evaluation = dict(result=dict(valid_episode=True, task_success=False, normalized_score=.4,
                                      native_process_score=40., native_reward=0.,
                                      success_authority="native_final_check"))
        saved_original, saved_evaluation = deepcopy(original), deepcopy(evaluation)
        result = reconcile_summary(original, evaluation, evidence_verified=True)
        self.assertEqual(result["normalized_score"], .4)
        self.assertEqual(result["native_evaluation_snapshot"]["native_process_score"], 40.)
        self.assertEqual(original, saved_original)
        self.assertEqual(evaluation, saved_evaluation)
        self.assertEqual(result["partial_native_result"], saved_original["partial_native_result"])

    def test_no_verified_evidence_keeps_original_classification(self):
        original = self.historical()
        for kwargs in ({}, {"evidence_verified": False}):
            result = reconcile_summary(original, **kwargs)
            self.assertEqual(result["primary_outcome"], "unscored_harness_timeout")
            self.assertFalse(result["scored_episode"])
            self.assertIsNone(result["task_success"])
            self.assertIsNone(result["normalized_score"])

    def test_unsafe_or_missing_cleanup_prevents_historical_reclassification(self):
        changes = ({"cleanup_status": None}, {"cleanup_status": "failed"},
                   {"status": "unknown"}, {"infrastructure_error": "robot disconnected"},
                   {"status": "failed"}, {"termination_reason": "shutdown-signal"})
        for unsafe in changes:
            with self.subTest(unsafe=unsafe):
                original = self.historical(**unsafe)
                result = reconcile_summary(original, evidence_verified=True)
                self.assertFalse(result["scored_episode"])
                self.assertIsNone(result["normalized_score"])
                self.assertIsNone(result["task_success"])
        original = self.historical()
        del original["cleanup_status"]
        self.assertFalse(reconcile_summary(original, evidence_verified=True)["scored_episode"])

    def test_missing_or_invalid_verified_evaluation_does_not_manufacture_score(self):
        for changes in ({"partial_native_result": None},
                        {"partial_native_result": {"valid_episode": False, "task_success": False,
                                                   "normalized_score": .4, "native_process_score": 40.}},
                        {"partial_native_result": {"valid_episode": True, "task_success": False,
                                                   "normalized_score": "0.4", "native_process_score": 40.}}):
            with self.subTest(changes=changes):
                result = reconcile_summary(self.historical(**changes), evidence_verified=True)
                self.assertFalse(result["scored_episode"])
                self.assertIsNone(result["normalized_score"])
                if (changes.get("partial_native_result") or {}).get("valid_episode") is True:
                    self.assertFalse(result["task_success"])
                else:
                    self.assertIsNone(result["task_success"])

    def test_verified_clean_model_budget_failure_can_reconcile(self):
        original = self.historical("model_request_limit", status="failed", model_budget_exhausted=True,
                                   primary_outcome="unscored_model_budget")
        result = reconcile_summary(original, evidence_verified=True)
        self.assertEqual(result["primary_outcome"], "scored_failure")
        self.assertFalse(result["task_success"])
        self.assertEqual(result["normalized_score"], .4)
        self.assertEqual(result["failure_reason"]["code"], "model_request_limit")

    def test_current_policy_is_rechecked_when_cleanup_becomes_unknown(self):
        for ended in (False, True):
            with self.subTest(native_ended=ended):
                native = self.historical()["partial_native_result"]
                source = {**self.historical(), **native}
                if ended:
                    source.update(task_success=True, normalized_score=1.,
                        native_episode=dict(ended=True, success=True, reason="native_success"))
                current = classify_episode_result(source, "stopped")
                self.assertTrue(current["scored_episode"])
                current.update(status="unknown", cleanup_status="failed",
                               infrastructure_error="cleanup resource release failed")
                result = reconcile_summary(current)
                if ended:
                    self.assertTrue(result["scored_episode"])
                    self.assertTrue(result["task_success"])
                    self.assertEqual(result["normalized_score"], 1.)
                else:
                    self.assertFalse(result["scored_episode"])
                    self.assertIsNone(result["task_success"])
                    self.assertIsNone(result["normalized_score"])
                    self.assertEqual(result["task_outcome"], "unknown")


class EffectiveSummaryEvidenceChecks(unittest.TestCase):
    def setUp(self):
        self.base = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def fixture(self, name="run", *, later_event=None, artifact_value=None):
        root = self.base / name
        identity = dict(run_id="recorded-bottle-fixture", episode_id="episode-1",
                        env_instance_id="native-env", env_generation=1)
        summary = HistoricalScoringReconciliationChecks().historical()
        native = {**summary["partial_native_result"], "evaluator_actions": 641,
                  "evaluation_id": "verified-current-native-evaluation"}
        recorded = {**identity, "result": native}
        store = Store(root, identity)
        try:
            store.set_state("runtime", "identity", identity)
            # Earlier physical work is allowed; later physical work invalidates
            # the evaluator snapshot even if the summary still looks terminal.
            store.event("action.requested", {"action_id": "earlier-action"})
            encoded = json.dumps(recorded if artifact_value is None else artifact_value).encode()
            reference = store.put_artifact(encoded, "native_evaluation", "application/json", "observer")
            metadata, blob = store.artifact(reference)
            evaluation = {**recorded, "result_ref": reference}
            store.event("observer.evaluated", {"result_ref": reference})
            if later_event:
                store.event(later_event, {})
            (root / "summary.json").write_text(json.dumps(summary))
            (root / "evaluation.json").write_text(json.dumps(evaluation))
        finally:
            store.close()
        return root, summary, evaluation, metadata, blob

    def assert_unreconciled(self, root):
        result = load_effective_summary(root)
        self.assertEqual(result["primary_outcome"], "unscored_harness_timeout")
        self.assertFalse(result["scored_episode"])
        self.assertIsNone(result["normalized_score"])
        self.assertIsNone(result["task_success"])
        self.assertNotIn("scoring_reconciliation", result)

    def test_valid_sqlite_artifact_and_current_evaluation_restore_original_point_four_failure(self):
        root, summary, evaluation, _, _ = self.fixture()
        before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
        result = load_effective_summary(root)
        self.assertEqual(result["primary_outcome"], "scored_failure")
        self.assertFalse(result["task_success"])
        self.assertEqual(result["normalized_score"], .4)
        self.assertEqual(result["native_process_score"], 40.)
        self.assertFalse(result["native_episode"]["ended"])
        self.assertEqual(result["failure_reason"]["code"], "official_no_action_progress_timeout")
        self.assertEqual(result["scoring_reconciliation"]["result_ref"], evaluation["result_ref"])
        self.assertEqual(result["partial_native_result"], summary["partial_native_result"])
        after = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
        self.assertEqual(after, before, "derived historical scoring must not write archived evidence")

    def test_sealed_database_read_does_not_create_wal_or_shm(self):
        root, _, _, _, _ = self.fixture()
        for suffix in ("-wal", "-shm"):
            self.assertFalse((root / ("events.sqlite" + suffix)).exists())
        self.assertTrue(load_effective_summary(root)["scored_episode"])
        for suffix in ("-wal", "-shm"):
            self.assertFalse((root / ("events.sqlite" + suffix)).exists())

    def test_historical_scoring_catalog_and_report_preserve_sealed_wal_archive(self):
        from ni_rua.apps.library.experiments import Catalog, register_run
        from ni_rua.apps.evaluation.report import summarize
        root, _, evaluation, _, _ = self.fixture()
        register_run(root, category="benchmark", task="fixture")
        store = Store(root, {})
        try:
            store.db.execute("PRAGMA wal_autocheckpoint=0")
            store.set_state("runtime", "wal_fixture", {"committed": True})
            archive = self.base / "catalog" / "archive"
            shutil.copytree(root, archive, ignore=shutil.ignore_patterns("events.sqlite-shm"))
        finally:
            store.close()
        def fingerprint():
            return {path.relative_to(archive): (path.read_bytes(), path.stat().st_mtime_ns)
                    for path in archive.rglob("*") if path.is_file()}
        before = fingerprint()
        self.assertTrue((archive / "events.sqlite-wal").exists())
        direct = load_effective_summary(archive)
        catalog = Catalog(archive.parent)
        catalog.scan()
        for summary in (direct, catalog.list()["experiments"][0]["summary"], summarize(archive)):
            self.assertEqual(summary["normalized_score"], .4)
            self.assertFalse(summary["task_success"])
            self.assertEqual(summary["scoring_reconciliation"]["result_ref"], evaluation["result_ref"])
        self.assertEqual(fingerprint(), before)

    def test_import_preserves_original_summary_and_effective_scoring(self):
        from ni_rua.apps.library.experiments import Catalog, import_run
        for current in (False, True):
            with self.subTest(current=current):
                root, original, _, _, _ = self.fixture("current" if current else "historical")
                effective = load_effective_summary(root)
                if current:
                    original = effective
                    (root / "summary.json").write_text(json.dumps(original))
                before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
                copied = import_run(root, self.base / "imports", category="benchmark", task="fixture")
                self.assertEqual(json.loads((copied / "summary.json").read_text()), original)
                self.assertEqual(json.loads((copied / "runtime.json").read_text())["status"], "stopped")
                catalog = Catalog(self.base / "imports")
                catalog.scan()
                imported = next(entry["item"]["summary"] for entry in catalog.entries.values() if entry["path"] == copied)
                for key in ("task_success", "normalized_score", "primary_outcome", "scored_episode", "failure_reason",
                            "scoring_policy", "native_episode", "cleanup_status", "native_evaluation_snapshot"):
                    self.assertEqual(imported[key], effective[key], key)
                self.assertEqual(before, {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()})

    def test_unrelated_events_do_not_hide_evaluation_but_a_later_reset_invalidates_it(self):
        root, _, evaluation, _, _ = self.fixture()
        identity = {key: evaluation[key] for key in ("run_id", "episode_id", "env_instance_id", "env_generation")}
        store = Store(root, identity)
        try:
            for _ in range(4100):
                store.event("codex.item_delta", {"text": "unrelated terminal output"})
            self.assertTrue(load_effective_summary(root)["scored_episode"])
            store.event("environment.reset_requested", {})
            self.assert_unreconciled(root)
        finally:
            store.close()

    def test_evaluation_identity_and_latest_observer_identity_must_match_database(self):
        for source in ("evaluation", "observer"):
            with self.subTest(source=source):
                root, _, evaluation, _, _ = self.fixture(source)
                if source == "evaluation":
                    evaluation["episode_id"] = "foreign-episode"
                    (root / "evaluation.json").write_text(json.dumps(evaluation))
                else:
                    with sqlite3.connect(root / "events.sqlite") as database:
                        row = database.execute("SELECT seq,body FROM events ORDER BY seq DESC LIMIT 1").fetchone()
                        event = json.loads(row[1])
                        event["env_generation"] = 2
                        database.execute("UPDATE events SET body=? WHERE seq=?", (json.dumps(event), row[0]))
                self.assert_unreconciled(root)

    def test_actions_and_resets_after_evaluation_reject_historical_backfill(self):
        for event in ("action.requested", "environment.reset_requested", "environment.recovery_reset"):
            with self.subTest(event=event):
                root, _, _, _, _ = self.fixture(event, later_event=event)
                self.assert_unreconciled(root)

    def test_native_action_counter_and_latest_evaluation_reference_must_match(self):
        for mismatch in ("counter", "reference"):
            with self.subTest(mismatch=mismatch):
                root, summary, _, _, _ = self.fixture(mismatch)
                if mismatch == "counter":
                    summary["native_episode"]["native_action_count"] = 640
                    (root / "summary.json").write_text(json.dumps(summary))
                else:
                    with sqlite3.connect(root / "events.sqlite") as database:
                        row = database.execute("SELECT seq,body FROM events ORDER BY seq DESC LIMIT 1").fetchone()
                        event = json.loads(row[1])
                        event["payload"]["result_ref"] = "artifact:another-evaluation"
                        database.execute("UPDATE events SET body=? WHERE seq=?", (json.dumps(event), row[0]))
                self.assert_unreconciled(root)

    def test_artifact_content_and_digest_mismatches_reject_backfill(self):
        for defect in ("digest", "content", "foreign_identity", "missing_blob", "array_body"):
            with self.subTest(defect=defect):
                artifact_value = None
                if defect == "content":
                    artifact_value = dict(run_id="recorded-bottle-fixture", episode_id="episode-1",
                        env_instance_id="native-env", env_generation=1,
                        result={"valid_episode": True, "task_success": False,
                                "normalized_score": .9, "native_process_score": 90., "evaluator_actions": 641})
                elif defect == "foreign_identity":
                    artifact_value = dict(run_id="foreign-run", result={})
                elif defect == "array_body":
                    artifact_value = []
                root, _, _, _, blob = self.fixture(defect, artifact_value=artifact_value)
                if defect == "digest":
                    blob.write_bytes(blob.read_bytes().replace(b'0.4', b'0.9', 1))
                elif defect == "missing_blob":
                    blob.unlink()
                self.assert_unreconciled(root)

    def test_cached_verification_does_not_hide_later_artifact_corruption(self):
        root, _, _, _, blob = self.fixture()
        self.assertTrue(load_effective_summary(root)["scored_episode"])
        blob.write_bytes(blob.read_bytes().replace(b'0.4', b'0.9', 1))
        self.assert_unreconciled(root)



if __name__ == "__main__":
    unittest.main()
