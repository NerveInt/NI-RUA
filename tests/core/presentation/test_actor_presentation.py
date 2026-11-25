"""Presentation removes documentary provenance while evidence stays usable."""

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import unittest

from ni_rua.core.presentation.actor_presentation import ActorPresentation
from ni_rua.core.storage.store import Store

NI_RUA_TEST_TIER = "component"


class ActorPresentationTests(unittest.TestCase):
    def setUp(self):
        temporary = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(temporary).resolve() / "run"  # Store and ProgramRunner use the canonical path
        self.identity = dict(run_id="presentation", episode_id="ep", env_instance_id="env", env_generation=1)
        self.store = Store(self.root, self.identity)
        self.addCleanup(self.store.close)
        self.presentation = ActorPresentation(self.store)

    def test_long_decimal_numbers_are_kept_while_hex_digests_are_removed(self):
        digest_value = "0123456789abcdef" * 4
        shown = self.presentation.present(dict(sim_time_ns="1792427364404397500", note="tick 1792427364404397500 done",
                                               values=["12345678901234567", digest_value], marker=digest_value))
        self.assertEqual(shown, dict(sim_time_ns="1792427364404397500", note="tick 1792427364404397500 done",
                                     values=["12345678901234567"]))

    def test_program_logs_name_files_relative_to_their_workspace(self):
        workspace = str(self.root) + "/private/program-workspaces/program-" + "a" * 24
        stderr = ('Traceback (most recent call last):\n  File "' + workspace + '/main.py", line 6, in <module>\n'
                  '  File "' + workspace + '/ni_rua_client.py", line 9\nOSError: ' + str(self.root) + '/private/x')
        shown = self.presentation.present(dict(stderr_text=stderr, note=str(self.root)))
        self.assertEqual(shown["stderr_text"], 'Traceback (most recent call last):\n  File "main.py", line 6, in <module>\n'
                         '  File "ni_rua_client.py", line 9\nOSError: <run>/private/x')
        self.assertNotIn(str(self.root), shown["stderr_text"])

    def test_artifact_aliases_are_stable_across_owners_and_resume(self):
        first = self.store.put_artifact(b"first", "trace", "application/octet-stream")
        second = self.store.put_artifact(b"second", "trace", "application/octet-stream")
        value = dict(trace_ref=first, refs=[first, second], nested={first: second})
        expected = dict(trace_ref="artifact:ref_1", refs=["artifact:ref_1", "artifact:ref_2"],
                        nested={"artifact:ref_1": "artifact:ref_2"})
        self.assertEqual(self.presentation.present(value), expected)
        self.assertEqual(ActorPresentation(self.store).present(value), expected)
        self.assertEqual(self.presentation.restore(expected), value)
        resumed = Store(self.root, self.identity)
        self.addCleanup(resumed.close)
        self.assertEqual(ActorPresentation(resumed).present(value), expected)
        self.assertEqual(ActorPresentation(resumed).restore(expected), value)
        self.assertEqual(self.store.resolve_artifact(first)[1], b"first")

    def test_python_source_aliases_restore_and_unknown_aliases_fail(self):
        reference = self.store.put_artifact(b"{}", "robot_model", "application/json")
        original = dict(source="data = client.artifact('" + reference + "')\nprint(data)",
                        evidence="Read " + reference + ", then inspect the measurement.")
        presented = self.presentation.present(original)
        self.assertIn("client.artifact('artifact:ref_1')", presented["source"])
        self.assertEqual(self.presentation.restore(presented), original)
        for unknown in ("artifact:ref_999", "artifact:ref_bad"):
            with self.subTest(unknown=unknown), self.assertRaisesRegex(ValueError, "unknown_actor_artifact_reference"):
                self.presentation.restore(dict(source="client.artifact('" + unknown + "')"))

    def test_concurrent_tool_owners_share_one_alias_per_reference(self):
        references = [self.store.put_artifact(str(index).encode(), "trace", "application/octet-stream")
                      for index in range(8)]
        owners = [ActorPresentation(self.store), ActorPresentation(self.store)]
        with ThreadPoolExecutor(max_workers=8) as pool:
            outputs = list(pool.map(lambda index: owners[index % 2].present(references[index % 8]),
                                    range(64)))
        self.assertEqual(len(set(outputs)), 8)
        self.assertEqual(len(self.store.states("actor_reference")), 8)
        for index, output in enumerate(outputs):
            self.assertEqual(self.presentation.restore(output), references[index % 8])

    def test_removes_hashes_and_document_sources_without_touching_physics(self):
        checksum = "a" * 64
        value = dict(sha256=checksum, sha_256=checksum, sha265=checksum, sh265=checksum,
                     md5=checksum, build_fingerprint=checksum, source_sha256=checksum, mesh_sha256=[checksum],
                     hashes={"mesh": checksum}, input_digest=checksum, hash="build",
                     digests=[checksum], checksums={"image": checksum}, responseHash=checksum,
                     source_url="https://example.test/manual", source_filename="X5A.urdf",
                     filename="robot.yml", official_sources=["https://example.test"],
                     repository="robot-repo", commit="abc123", version=checksum,
                     source_fingerprints=[dict(path="robot.py", sha256=checksum)],
                     repository_heads=dict(robot=checksum), manufacturer_repository="https://example.test/repo",
                     manual_url="https://example.test/manual", reference="official-sources.json",
                     calibration_id=checksum, sources={"urdf": "/robot/X5A.urdf"},
                     robot_state=dict(source_kind="measurement", q_rad=[1.23456789012345],
                         ee_pose=dict(source_kind="derived_from_measured_link6", frame_id="env_origin"),
                         gripper=dict(command_source_kind="command", position_source_kind="measurement")),
                     evidence=dict(observation_id=checksum, target_id="target", source_kind="actor_judgment"),
                     schema_version="0.1", robot_version="x5-v1", calibration_source="pinhole configuration",
                     path=dict(kind="backend_free", waypoints=[[1.0, 2.0, 3.0]]))
        original = deepcopy(value)
        result = self.presentation.present(value)
        expected = {key: value[key] for key in ("robot_state", "evidence", "schema_version", "robot_version", "calibration_source", "path")}
        expected = deepcopy(expected)
        expected["evidence"]["observation_id"] = "id_1"
        self.assertEqual(result, expected)
        self.assertEqual(self.presentation.restore(result["evidence"]), original["evidence"])
        result["robot_state"]["q_rad"].append(2.0)
        self.assertEqual(value, original)
        self.assertEqual(self.presentation.present(dict(version="v3", calibration_id="camera-v2")),
                         dict(version="v3", calibration_id="camera-v2"))
        self.assertEqual(self.presentation.present(dict(source="env/robot/x5.py:24-32", damping=4.0)),
                         dict(damping=4.0))

    def test_text_json_program_json_and_free_text_remove_hashes(self):
        reference = self.store.put_artifact(b"trace", "trace", "application/octet-stream")
        raw = json.dumps(dict(sha256="a" * 64, source_url="https://example.test/source",
                              trace_ref=reference, source_kind="measurement"))
        free_text = "User task: compare https://example.test and sha256 " + "b" * 64
        value = dict(contentItems=[dict(type="inputText", text=raw),
                                  dict(type="inputText", text=free_text),
                                  dict(type="inputImage", imageUrl="data:image/png;base64,AAAA")],
                     stdout_text=raw, stderr_text=raw, stdout=dict(text=raw),
                     task=free_text, source="print(" + repr(raw) + ")")
        result = self.presentation.present(value)
        expected = dict(trace_ref="artifact:ref_1", source_kind="measurement")
        for text in (result["contentItems"][0]["text"], result["stdout_text"],
                     result["stderr_text"], result["stdout"]["text"]):
            self.assertEqual(json.loads(text), expected)
        self.assertEqual(result["task"], "User task: compare https://example.test and ")
        self.assertEqual(result["contentItems"][1]["text"], result["task"])
        self.assertEqual(result["contentItems"][2], value["contentItems"][2])
        self.assertNotIn("a" * 64, result["source"])
        self.assertIn("artifact:ref_1", result["source"])

    def test_json_artifact_projection_preserves_original_integrity_and_binary_bytes(self):
        inner = self.store.put_artifact(b"pixels", "rgb", "image/png")
        raw = json.dumps(dict(sha256="a" * 64, source_url="https://example.test", image_ref=inner,
                              measured_q_rad=[1.23456789012345])).encode()
        reference = self.store.put_artifact(raw, "trace", "application/json; charset=utf-8")
        info, content = self.store.resolve_artifact(reference)
        original = deepcopy(info)
        projected_info, projected = self.presentation.present_artifact(info, content)
        for key in ("sha256", "storage_key", "byte_size"):
            self.assertNotIn(key, projected_info)
        self.assertEqual(projected_info["artifact_id"], "ref_1")
        self.assertEqual(json.loads(projected), dict(image_ref="artifact:ref_2", measured_q_rad=[1.23456789012345]))
        self.assertEqual(self.presentation.restore(projected_info)["artifact_id"], info["artifact_id"])
        self.assertEqual(info, original)
        self.assertEqual(self.store.resolve_artifact(reference), (original, raw))
        binary_info, binary = self.store.resolve_artifact(inner)
        self.assertEqual(self.presentation.present_artifact(binary_info, binary)[1], binary)

    def test_json_lines_are_projected_and_non_json_log_bytes_and_media_urls_survive(self):
        document = dict(source_url="https://example.test/proof", contentDigest="a" * 64,
                        measured_q_rad=[1.23456789012345], source_kind="measurement")
        encoded = json.dumps(document)
        ordinary = "progress: 0.125 m; code sha256(data) is an expression\r\n  1.234500  "
        logs = ordinary + "\r\n" + encoded + "\r\n" + encoded + "\nfinished\n"
        image = dict(type="inputImage", imageUrl="https://example.test/image.png",
                     source_url="https://example.test/image-proof", sha256="a" * 64)
        audio = dict(type="inputAudio", audioUrl="https://example.test/audio.wav",
                     source_url="https://example.test/audio-proof", checksum="b" * 64)
        value = dict(stdout_text=logs, stderr_excerpt=logs,
                     stdout_excerpt=dict(text=logs), stderr=dict(text=logs),
                     contentItems=[dict(type="inputText", text=logs), image, audio])
        original = deepcopy(value)
        expected = dict(measured_q_rad=[1.23456789012345], source_kind="measurement")
        rendered = ordinary + "\r\n" + json.dumps(expected, separators=(",", ":"), sort_keys=True) + "\r\n"
        rendered += json.dumps(expected, separators=(",", ":"), sort_keys=True) + "\nfinished\n"
        result = self.presentation.present(value)
        for text in (result["stdout_text"], result["stderr_excerpt"], result["stdout_excerpt"]["text"],
                     result["stderr"]["text"], result["contentItems"][0]["text"]):
            self.assertEqual(text, rendered)
        self.assertEqual(result["contentItems"][1], dict(type="inputImage", imageUrl=image["imageUrl"]))
        self.assertEqual(result["contentItems"][2], dict(type="inputAudio", audioUrl=audio["audioUrl"]))
        self.assertEqual(value, original)

    def test_hash_values_in_text_trace_subpaths_and_text_artifacts_are_removed(self):
        checksum = "0123456789abcdef" * 4
        text = "q=0.1234567890123456789; checkpoint_sha256=" + checksum + " done"
        self.assertEqual(self.presentation.present_text(text), "q=0.1234567890123456789;  done")
        self.assertEqual(self.presentation.present(dict(trace=checksum, unexpected=checksum)), {})
        self.assertEqual(self.presentation.present([checksum, dict(distance_m=0.125)]), [dict(distance_m=0.125)])
        self.assertEqual(self.presentation.present_text("commit a1b2c3d; crc32=1234abcd"), "; ")
        scientific = "q=1.2345678901234567e-05 rad; q=-1.2345678901234567E+05"
        self.assertEqual(self.presentation.present_text(scientific), scientific)
        for scientific in ("q=1234567890123456e-05", "q=1e-1234567890123456"):
            self.assertEqual(self.presentation.present_text(scientific), scientific)
        for scalar in ("1.234567890123456789e-05", "0.12345678901234567890123456789"):
            self.assertEqual(self.presentation.present_text(scalar), scalar)
        for prose in ("proof=0x" + checksum, "data:image/png;hash=" + checksum,
                      "sha256-" + checksum + " hash_" + checksum):
            self.assertNotIn(checksum, self.presentation.present_text(prose))
            self.assertNotIn("id_", self.presentation.present_text(prose))
        self.assertEqual(self.presentation.present({checksum: "unused metadata"}), {})
        reference = self.store.put_artifact(text.encode(), "program_stdout", "text/plain")
        info, raw = self.store.resolve_artifact(reference)
        _, shown = self.presentation.present_artifact(info, raw)
        self.assertNotIn(checksum.encode(), shown)
        self.assertEqual(self.store.resolve_artifact(reference)[1], raw)
        self.assertNotIn(checksum, self.presentation.present_text("/tmp/" + checksum + ".txt"))

    def test_compound_ids_and_dictionary_keys_round_trip_across_ports_and_resume(self):
        originals = ["program-" + "a" * 24 + "-" + "b" * 16,
                     "action-plugin-" + "c" * 24, "review-" + "d" * 24,
                     "batch-" + "e" * 32, "shared_experience_" + "f" * 32]
        raw = dict(ids=originals, decisions={originals[2]: dict(action_id=originals[1])},
                   source="client.call('status', job_id='" + originals[0] + "')")
        shown = self.presentation.present(raw)
        # A program call is not its job; it gets its own short-ID namespace.
        self.assertEqual(shown["ids"], ["call_1", "act_1", "review_1", "plan_1", "exp_1"])
        self.assertEqual(shown["decisions"], {"review_1": dict(action_id="act_1")})
        self.assertEqual(ActorPresentation(self.store).restore(shown), raw)
        self.assertEqual(ActorPresentation(self.store).present(raw), shown)

    def test_schema_aliasing_preserves_structure_and_media_bytes(self):
        checksum = "abcdef01" * 8
        schema = dict(type="object", properties=dict(sha256=dict(type="string", enum=[checksum],
            description="Choose a value; build sha256=" + checksum)), required=["sha256"])
        shown = self.presentation.present_data(schema)
        self.assertEqual(shown["required"], ["sha256"])
        self.assertNotIn(checksum, json.dumps(shown))
        restored = self.presentation.restore(shown)
        self.assertEqual(restored["properties"]["sha256"]["enum"], [checksum])
        self.assertEqual(shown["properties"]["sha256"]["description"], "Choose a value; build ")
        for item in (dict(type="image", url="data:image/png;base64," + checksum),
                     dict(type="inputImage", imageUrl="https://example.test/" + checksum + ".png"),
                     dict(type="inputAudio", audioUrl="data:audio/wav;base64," + checksum)):
            self.assertEqual(self.presentation.present(item), item)

    def test_required_custom_identifier_and_executable_source_share_only_reference_aliases(self):
        frame_id = "frame-" + "1234abcd" * 4
        shown = self.presentation.present(dict(frame_id=frame_id, calibration_id="c"*64,
            builder_digest="d"*64))
        self.assertEqual(shown, dict(frame_id="id_1"))
        checksum = "checksum-" + "e"*64
        source = "frame = " + repr(frame_id) + "\nchecksum = " + repr(checksum) + "\nprecision = 0.1234567890123456789\n"
        expected = source.replace(frame_id, "id_1")
        projected = ActorPresentation(self.store).present_identifiers(source)
        self.assertEqual(projected, expected)
        self.assertEqual(ActorPresentation(self.store).restore(projected), source)
        self.assertEqual(ActorPresentation(self.store).restore(shown), dict(frame_id=frame_id))

    def test_stage_labels_survive_metadata_filter_without_exempting_their_evidence(self):
        labels = ['hash', 'sha256', 'a' * 64, 'custom-stage-' + 'c' * 24]
        evidence = dict(status='failed', reason='joint_limit', error_rad=0.12345678901234567,
                        sha256='b' * 64)
        raw = dict(stage_results={label: deepcopy(evidence) for label in labels},
                   stages={label: deepcopy(evidence) for label in labels},
                   checkpoint_reasons={label: 'Inspect contact' for label in labels},
                   stage_costs={label: 2 for label in labels},
                   pending_stage_ids=labels, completed_stage_ids=labels, checkpoint_stage_ids=labels)
        shown = self.presentation.present(raw)
        expected_labels = ['hash', 'sha256', 'id_1', 'id_2']
        for key in ('stage_results', 'stages', 'checkpoint_reasons', 'stage_costs'):
            self.assertEqual(list(shown[key]), expected_labels)
        for key in ('pending_stage_ids', 'completed_stage_ids', 'checkpoint_stage_ids'):
            self.assertEqual(shown[key], expected_labels)
        for stage in shown['stage_results'].values():
            self.assertEqual(stage, {key: value for key, value in evidence.items() if key != 'sha256'})
        self.assertEqual(ActorPresentation(self.store).restore(shown)['pending_stage_ids'], labels)
        self.assertEqual(raw['stage_results']['hash'], evidence)
        self.assertEqual(json.loads(self.presentation.present_text(json.dumps(raw))), shown)

    def test_stage_dependencies_alias_before_the_referenced_stage_is_defined(self):
        labels = ['a' * 64, 'custom-stage-' + 'c' * 24]
        raw = dict(stages=[dict(id='later', depends_on=labels), *[dict(id=label) for label in labels]])
        shown = self.presentation.present(raw)
        self.assertEqual(shown['stages'][0]['depends_on'], ['id_1', 'id_2'])
        self.assertEqual([stage['id'] for stage in shown['stages'][1:]], ['id_1', 'id_2'])
        self.assertEqual(ActorPresentation(self.store).restore(shown), raw)

    def test_clock_domains_remain_distinct_reversible_and_stable_across_owners(self):
        domains = ['env-' + 'a' * 32 + '/generation-1', 'env-' + 'b' * 32 + '/generation-1',
                   'env-' + 'a' * 32 + '/generation-2']
        raw = dict(step_domain=domains[0], related=domains, sim_step=12, readable_domain='simulator/control')
        shown = self.presentation.present(raw)
        self.assertEqual(shown, dict(step_domain='clock_1', related=['clock_1', 'clock_2', 'clock_3'],
                                    sim_step=12, readable_domain='simulator/control'))
        resumed = Store(self.root, self.identity)
        self.addCleanup(resumed.close)
        owner = ActorPresentation(resumed)
        self.assertEqual(owner.present(raw), shown)
        self.assertEqual(owner.restore(shown), raw)
        self.assertEqual(json.loads(owner.present_text(json.dumps(raw))), shown)
        _, content = owner.present_artifact(dict(mime_type='text/plain'), json.dumps(raw).encode())
        self.assertEqual(json.loads(content), shown)

    def test_program_response_uses_owner_projection_without_actor_evidence_compression(self):
        from ni_rua.core.tool_output import StructuredText, ToolResponse
        review = dict(schema_version='0.1', env_instance_id='private-env',
            answer=dict(assessment='uncertain'), execution=dict(status='unknown',
                actions=[dict(action_id='a', feedback=dict(executed_native_steps=None, stop_confirmed=False))]))
        original = deepcopy(review)
        response = ToolResponse([StructuredText(review, 'operation_review')], data=review,
                                kind='operation_review', success=True)
        shown = self.presentation.present_response(response)
        value = json.loads(shown['contentItems'][0]['text'])
        self.assertEqual(value, {key: item for key, item in review.items() if key not in {'schema_version', 'env_instance_id'}})
        self.assertIs(response.data, review)
        self.assertEqual(review, original)
        self.assertNotIn('details_ref', value)

    def test_program_typed_data_uses_owner_projection_aliases_and_preserves_unknown(self):
        from ni_rua.core.tool_output import ToolOutput, output_contract_error
        raw = dict(schema_version='0.1', run_id='private-run',
            receipt=dict(action_id='action-'+'a'*32, status='failed', executed_native_steps=None,
                stop_confirmed=False, versions={'worker':'build'},
                failure_feedback=dict(schema_version='v1', error_m=0.00798123456789,
                    sha256='b'*64), pending_stage_ids=['release']),
            diagnostics={'revision':3, 'version':'business-version'})
        original = deepcopy(raw)
        response = ToolOutput(raw, kind='action', projection='action_result').response()
        response = output_contract_error(response, 'robot_action', 'image_missing', '/content')
        shown = self.presentation.present_program_response(response)
        self.assertEqual(shown['kind'], 'action')
        self.assertFalse(shown['success'])
        self.assertEqual(shown['output_contract_error']['phase'], 'after_handler')
        self.assertNotIn('run_id', shown['data'])
        self.assertNotIn('schema_version', shown['data'])
        receipt = shown['data']['receipt']
        self.assertNotIn('versions', receipt)
        self.assertEqual(receipt['failure_feedback'], {'error_m':0.00798123456789})
        self.assertIsNone(receipt['executed_native_steps'])
        self.assertFalse(receipt['stop_confirmed'])
        self.assertEqual(self.presentation.restore(receipt['action_id']), raw['receipt']['action_id'])
        self.assertEqual(shown['data']['diagnostics'], original['diagnostics'])
        self.assertEqual(raw, original)
        self.assertNotIn('data', self.presentation.present_response(response))

    def test_program_malformed_extra_keeps_executed_receipt_and_marks_omission(self):
        from ni_rua.core.tool_output import StructuredText, ToolResponse, output_contract_error
        receipt = dict(status='failed', executed_native_steps=7, stop_confirmed=True,
            pending_stage_ids=['release'], bad_diagnostic=object())
        raw = dict(receipt=receipt, extra=object(), current_observation_id='obs-after')
        response = ToolResponse([StructuredText({'status':'display_unavailable'})],
            data=raw, kind='action', data_projection='action_result')
        response = output_contract_error(response, 'robot_action', 'result_data_not_json', '/data')
        shown = self.presentation.present_program_response(response)
        self.assertEqual(shown['kind'], 'action')
        self.assertFalse(shown['success'])
        self.assertEqual(shown['data']['receipt'], {k:v for k,v in receipt.items() if k!='bad_diagnostic'})
        self.assertEqual(shown['data']['current_observation_id'], 'obs-after')
        self.assertEqual(shown['output_contract_error']['omitted_paths'],
            ['/data/receipt/bad_diagnostic', '/data/extra'])
        self.assertEqual(shown['output_contract_error']['phase'], 'after_handler')
        self.assertIn('do not automatically replay', shown['output_contract_error']['suggestion'])
        json.dumps(shown)
        self.assertIs(response.data, raw)
        self.assertIn('bad_diagnostic', raw['receipt'])
        cycle = []
        cycle.append(cycle)
        raw['extra'] = cycle
        shown = self.presentation.present_program_response(response)
        self.assertEqual(shown['data']['receipt']['executed_native_steps'], 7)
        self.assertFalse(shown['success'])
        self.assertIn('/data/extra', shown['output_contract_error']['omitted_paths'])
        json.dumps(shown)

    def test_post_execution_presentation_fallback_survives_closed_store_and_nonfinite_content(self):
        from ni_rua.core.presentation.actor_output import ActorOutput
        from ni_rua.core.presentation.observation_output import same_frame_robot_state, saved_observation_state
        from ni_rua.core.embodiment import EMBODIMENT_RECORD_PREFIX, profile_record
        from ni_rua.robot.model import embodiments as e
        from ni_rua.core.tool_output import StructuredText, ToolResponse

        temp_dir = self.enterContext(tempfile.TemporaryDirectory())
        closed_store = Store(temp_dir, dict(run_id="r", episode_id="e", env_instance_id="i", env_generation=1))
        pres = ActorPresentation(closed_store)
        actor_out = ActorOutput(closed_store, pres)
        closed_store.close()

        # DEF-09 #1: ActorOutput.present preserves action receipt when primary display fails AND Store is closed.
        receipt = {
            "status": "unknown",
            "executed_native_steps": 4,
            "stop_confirmed": False,
            "action_id": "action-1234567890abcdef12345678",
            "note": f"worker log at {closed_store.root / 'robot-worker.log'}",
        }
        bad_part = StructuredText({"nan_field": float("nan")})
        act_resp = ToolResponse([bad_part], data={"receipt": receipt}, kind="action")
        wire = actor_out.present(act_resp, tool="robot_action")
        self.assertFalse(wire["success"])
        warning = json.loads(wire["contentItems"][0]["text"])["output_contract_error"]
        self.assertEqual(warning["phase"], "after_handler")
        self.assertTrue(warning["physical_action_executed"])
        self.assertEqual(warning["receipt_status"], "unknown")
        self.assertEqual(warning["action_id"], "act_unavailable")
        self.assertIn("do not automatically replay", warning["suggestion"])
        rescued = json.loads(wire["contentItems"][1]["text"])
        self.assertEqual(rescued["receipt"]["status"], "unknown")
        self.assertEqual(rescued["receipt"]["executed_native_steps"], 4)
        self.assertNotIn(str(closed_store.root), wire["contentItems"][1]["text"])

        # DEF-09 #2: present_program_response rescues response.data['receipt'] even if response.content has NaN.
        prog_resp = ToolResponse(
            [StructuredText({"nan_display": float("nan")})],
            data={"receipt": {"status": "completed", "executed_native_steps": 5}, "observation_id": "obs-1"},
            kind="action",
            data_projection="action_result",
        )
        prog_shown = self.presentation.present_program_response(prog_resp)
        self.assertEqual(prog_shown["kind"], "action")
        self.assertFalse(prog_shown["success"])
        self.assertEqual(prog_shown["data"]["receipt"]["executed_native_steps"], 5)
        self.assertIsNotNone(prog_shown["output_contract_error"])

        # DEF-17: host <run_dir> paths outside AGENT_DIRECTORIES are scrubbed across all string fields,
        # while <run_dir>/actor remains verbatim.
        actor_file = str(self.store.root / "actor" / "script.py")
        worker_log = str(self.store.root / "robot-worker.log")
        scrubbed = self.presentation.present({"error": f"inspect {worker_log}", "workspace": actor_file})
        self.assertNotIn(str(self.store.root / "robot-worker.log"), scrubbed["error"])
        self.assertIn("<run>/robot-worker.log", scrubbed["error"])
        self.assertEqual(scrubbed["workspace"], actor_file)

        # DEF-22: same_frame_robot_state and saved_observation_state honor single-arm SO-101 & custom embodiment arms.
        import dataclasses
        from ni_rua.core.embodiment import FrozenDict

        custom_prof = dataclasses.replace(
            e.ARX_X5_PROFILE,
            arms=("arm0",),
            max_simultaneous_arms=1,
            supports=e.ARX_X5_PROFILE.supports - {"dual_arm"},
            arm_joint_names=FrozenDict(arm0=("j1", "j2", "j3", "j4", "j5")),
            jaw_joint_names=FrozenDict(arm0=("j6",)),
            wrist_cameras=FrozenDict(),
            home_check=None,
        )
        obs_so101 = {
            "observation_id": "obs-so101",
            "sim_step": 3,
            "step_domain": "so101_control_step",
            "observation_profile": "rgb_proprio_calibrated_v1",
        }
        state_so101 = {
            "sim_step": 3,
            "step_domain": "so101_control_step",
            "frame_id": "so101_base",
            "arms": {
                "arm": {
                    "joint_names": ["j1", "j2", "j3", "j4", "j5"],
                    "q_rad": [0.0] * 5,
                    "source_kind": "measurement",
                    "ee_pose": {
                        "position_m": [0.15, 0.0, 0.12],
                        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                        "frame_id": "so101_base",
                        "tcp_config_id": "so101-tcp",
                        "source_kind": "measurement",
                    },
                }
            },
        }
        direct_sf = same_frame_robot_state(obs_so101, state_so101)
        self.assertEqual(direct_sf["status"], "available")
        self.assertIn("arm", direct_sf["robot_state"]["arms"])

        so101_dir = self.enterContext(tempfile.TemporaryDirectory())
        so101_store = Store(so101_dir, dict(run_id="r2", episode_id="e2", env_instance_id="i2", env_generation=1))
        self.addCleanup(so101_store.close)
        so101_store.save("configuration", f"{EMBODIMENT_RECORD_PREFIX}custom", profile_record(custom_prof))
        state_arm0 = dict(state_so101, frame_id="custom_base", arms={"arm0": state_so101["arms"]["arm"]})
        state_ref = so101_store.put_artifact(json.dumps(state_arm0).encode("utf-8"), "robot_state", "application/json")
        so101_store.save("observation", "obs-arm0", dict(obs_so101, observation_id="obs-arm0", robot_state_ref=state_ref))
        saved_sf = saved_observation_state(so101_store, "obs-arm0")
        self.assertEqual(saved_sf["status"], "available")
        self.assertIn("arm0", saved_sf["robot_state"]["arms"])

    def test_labeled_hash_preserves_domain_error_codes_while_stripping_real_digests(self):
        from ni_rua.core.presentation.actor_presentation import metadata_findings, scrub_for_fallback

        legal_messages = [
            "commit: observation_id_must_be_current",
            "commit: observation_id_1_must_be_current",
            "COMMIT: OBSERVATION_ID_1_MUST_BE_CURRENT",
            "KeyError: hash_map_lookup_failed_for_stage",
            "digest: native_steps_remaining_exhausted",
            "checksum: gripper_aperture_below_min_m_on_left",
            "fingerprint: camera_calibration_mismatch_on_head",
            "pre_commit_validation_failed_on_arm_left",
            "input_digest_mismatch_on_cached_plan",
            "commit: deadbeef_is_not_a_domain_code",
            "commit: InvalidObservationCommitError",
            "commit: Arm0Joint1LimitExceededOnStage2",
            "commit_status=awaiting_observation_check",
            "hash-policy: reject-stale-observation-in-batch",
            "source_hash: models/arx_x5/urdf/arx_x5_calibrated",
            "code sha256(data) is an expression",
        ]
        for message in legal_messages:
            with self.subTest(legal=message):
                self.assertEqual(self.presentation.present_text(message), message)
                self.assertEqual(self.presentation.present_egress_text(message), message)
                self.assertEqual(scrub_for_fallback(message), message)
                self.assertEqual(metadata_findings(message), [])

        sha256_hex = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        digest_samples = [
            ("COMMIT: DEADBEE", "DEADBEE"),
            ("commit abcdefa", "abcdefa"),
            ("commit a1b2c3d", "a1b2c3d"),
            ("git-commit: CAFEBABE", "CAFEBABE"),
            ("crc32=1234abcd", "1234abcd"),
            ("md5: d41d8cd98f00b204e9800998ecf8427e", "d41d8cd98f00b204e9800998ecf8427e"),
            ("sha256: " + sha256_hex, sha256_hex),
            ("sha-256=" + sha256_hex, sha256_hex),
            ("checkpoint_sha256=" + sha256_hex, sha256_hex),
            ("proof=0x" + sha256_hex, sha256_hex),
            ("sha256-47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU=", "47DEQpj8HBSa"),
            ("sha256-47DEQpj8HBSa-_TImW-5JCeuQeRkm5NMpJWZG3hSuFU", "47DEQpj8HBSa"),
            ("digest=47DEQpj8HBSa9zTImW85JCeuQeRkm5NMpJWZG3hSuFU", "47DEQpj8HBSa"),
        ]
        for sample, forbidden in digest_samples:
            with self.subTest(digest=sample):
                self.assertTrue(metadata_findings(sample))
                for cleaned in (
                    self.presentation.present_text(sample),
                    self.presentation.present_egress_text(sample),
                    scrub_for_fallback(sample),
                ):
                    self.assertNotIn(forbidden, cleaned)
                    self.assertEqual(metadata_findings(cleaned), [])

    def test_stage_plans_and_stage_indexed_effective_tolerances_keep_stage_keys(self):
        from ni_rua.core.presentation.actor_presentation import metadata_findings, scrub_for_fallback

        stage_labels = ["commit", "hash", "sha256", "a" * 64]
        raw = {
            "stage_plans": {
                label: {"position_m": [0.1, 0.2, 0.3], "sha256": "b" * 64}
                for label in stage_labels
            },
            "effective_tolerances": {
                label: {"position_tolerance_m": 0.005, "orientation_tolerance_rad": 0.05, "sha256": "c" * 64}
                for label in stage_labels
            },
            "single_stage": {
                "effective_tolerances": {
                    "position_tolerance_m": 0.005,
                    "commit": "abc1234",
                    "sha256": "d" * 64,
                }
            },
        }
        shown = self.presentation.present(raw)
        expected_keys = ["commit", "hash", "sha256", "id_1"]
        self.assertEqual(list(shown["stage_plans"]), expected_keys)
        self.assertEqual(list(shown["effective_tolerances"]), expected_keys)
        for entry in shown["stage_plans"].values():
            self.assertEqual(entry, {"position_m": [0.1, 0.2, 0.3]})
        for entry in shown["effective_tolerances"].values():
            self.assertEqual(entry, {"position_tolerance_m": 0.005, "orientation_tolerance_rad": 0.05})
        # 单阶段数值容差字典不递归豁免来源元数据键
        self.assertEqual(shown["single_stage"]["effective_tolerances"], {"position_tolerance_m": 0.005})
        self.assertEqual(metadata_findings(json.dumps(shown)), [])
        fallback = scrub_for_fallback(raw)
        self.assertEqual(list(fallback["stage_plans"])[:3], ["commit", "hash", "sha256"])
        self.assertEqual(list(fallback["effective_tolerances"])[:3], ["commit", "hash", "sha256"])
        self.assertEqual(fallback["single_stage"]["effective_tolerances"], {"position_tolerance_m": 0.005})

    def test_contract_error_pins_first_across_reply_budget_and_marks_executed_actions(self):
        from ni_rua.core.presentation.actor_output import ActorOutput
        from ni_rua.core.presentation.reply_budget import fit_reply_report
        from ni_rua.core.tool_output import ToolOutput, output_contract_error

        actor_out = ActorOutput(self.store, self.presentation)
        executed = dict(
            action_id="action-" + "1" * 32,
            receipt=dict(
                action_id="action-" + "1" * 32,
                status="completed",
                executed_native_steps=6,
                completed_native_steps=6,
                motion_safety_checks=[dict(stage_id="reach", status="pass")],
            ),
            stages=dict(reach=dict(status="completed", completed_native_steps=6, grasp_contact_feedback=dict(a=1))),
            trajectory_feedback=dict(
                available=True,
                planned=dict(samples=[dict(native_step=1)]),
                measured=dict(samples=[dict(native_step=1)]),
            ),
        )
        failed_reply = output_contract_error(
            ToolOutput(executed, kind="action", projection="action_result").response(),
            "robot_action",
            "image_missing",
            "/content/1",
        )
        self.assertFalse(failed_reply.success)
        shown = actor_out.present(failed_reply, tool="robot_action")
        self.assertFalse(shown["success"])
        first_item = json.loads(shown["contentItems"][0]["text"])["output_contract_error"]
        self.assertTrue(first_item["physical_action_executed"])
        self.assertEqual(first_item["receipt_status"], "completed")
        self.assertEqual(first_item["action_id"], "act_1")
        self.assertIn("do not automatically replay", first_item["suggestion"])

        # summary_first 与窄预算重排均必须保持 output_contract_error 锁定在第 0 项
        for profile, budget in (("summary_first", None), ("summary_first", 40), ("full", 40)):
            reversed_items = list(reversed(shown["contentItems"]))
            fitted, report = fit_reply_report(reversed_items, budget, profile=profile, layout=True)
            self.assertTrue(report["applied"])
            self.assertIn("output_contract_error", json.loads(fitted[0]["text"]))
            self.assertIn("receipt", json.loads(fitted[1]["text"]))

        # 未执行（rejected / physics_advanced=False）动作不得误标 physical_action_executed
        for non_exec in (
            dict(receipt=dict(status="rejected", executed_native_steps=0)),
            dict(receipt=dict(status="failed", physics_advanced=False, executed_native_steps=0)),
        ):
            unexec_err = output_contract_error(
                ToolOutput(non_exec, kind="action", projection="action_result").response(),
                "robot_action",
                "schema_issue",
                "/data",
            )
            self.assertNotIn("physical_action_executed", unexec_err.contract_error)

    def test_single_ref_frame2_metadata_and_compact_capability_delivery(self):
        from ni_rua.core.presentation.actor_output import ActorOutput, actor_capabilities
        from ni_rua.core.presentation.capability_output import CapabilityDelivery
        from ni_rua.core.presentation.observation_output import MetadataDelivery, observation_metadata

        def make_obs(obs_id):
            cal = dict(
                K=[[397.0, 0.0, 320.0], [0.0, 397.0, 240.0], [0.0, 0.0, 1.0]],
                distortion=[0.0] * 5,
                distortion_model="pinhole_no_distortion",
                image_size=[640, 480],
                optical_axes="+x right,+y down,+z forward",
                calibration_source="public sensor calibration",
            )
            val = dict(
                bundle=dict(
                    observation_id=obs_id,
                    run_id="run",
                    episode_id="ep",
                    env_instance_id="env",
                    env_generation=1,
                    schema_version="0.1",
                    robot_model_id="arx-x5",
                    observation_profile="rgb_proprio_calibrated_v1",
                    clock_domain="env/generation-1",
                    step_domain="adapter_control_step",
                    calibration_id="a" * 64,
                    sim_step=10,
                    action_budget=dict(
                        native_steps_remaining=700,
                        semantics="official native steps; separate from tool requests and model-call budget; reserves are not included",
                        unknown_values_are_null=True,
                    ),
                    cameras=[
                        dict(
                            camera_id=name,
                            width=640,
                            height=480,
                            pixel_transform="identity",
                            control_dt_s=0.004,
                            step_domain="adapter_control_step",
                            sim_step=10,
                        )
                        for name in ("head", "left", "right")
                    ],
                ),
                robot_state=dict(arms=dict(left=dict(joint_names=["j1"], q_rad=[0.1])), step_domain="adapter_control_step", sim_step=10),
                camera_calibration=dict(profile_id="cal", cameras=dict(head=cal, left=cal, right=cal)),
                camera_pose_sample=dict(
                    sim_step=10,
                    reference_frame_id="env_origin",
                    step_domain="adapter_control_step",
                    cameras={name: dict(T_world_camera=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) for name in ("head", "left", "right")},
                ),
            )
            return observation_metadata(
                val,
                current_observation_id=obs_id,
                current_action_budget=dict(
                    native_steps_remaining=700,
                    semantics="official native steps; separate from tool requests and model-call budget; reserves are not included",
                    unknown_values_are_null=True,
                ),
                compact_geometry=True,
            )

        md = MetadataDelivery()
        ep = ({"compaction": "e1"}, "app", "tr", None)
        p1 = md.prepare(make_obs("obs-1"), epoch=ep)
        self.assertTrue(md.commit(p1))
        p2 = md.prepare(make_obs("obs-2"), epoch=ep)
        # 第 2 帧立即通过单引用 bundle 消除重复约定，同时保留首次出现的新说明文案
        self.assertEqual(set(p2.metadata["metadata_delivery"]["static_refs"]), {"camera_calibration", "robot_state", "bundle"})
        self.assertEqual(p2.metadata["metadata_delivery"]["static_refs"]["bundle"]["observation_id"], "obs-1")
        self.assertNotIn("width", p2.metadata["bundle"]["cameras"][0])
        self.assertIn("semantics", p2.metadata["metadata_delivery"])
        self.assertIn("omitted", p2.metadata["geometry_delivery"])
        self.assertTrue(md.commit(p2))
        p3 = md.prepare(make_obs("obs-3"), epoch=ep)
        self.assertEqual(set(p3.metadata["metadata_delivery"]["static_refs"]), {"camera_calibration", "robot_state", "bundle"})
        self.assertNotIn("semantics", p3.metadata["metadata_delivery"])
        self.assertNotIn("omitted", p3.metadata["geometry_delivery"])

        cd = CapabilityDelivery()
        caps = dict(frame_id="env_origin", dual_arm=True)
        actor_out = ActorOutput(self.store, self.presentation, payload_profile="lean_v2")
        from ni_rua.core.tool_output import ToolOutput

        first_raw = cd.project(caps, epoch="e1", current_action_budget={"remaining": 10})
        first_wire = actor_out.present(
            ToolOutput(first_raw, projection="capabilities").response(),
            tool="robot_capabilities",
        )
        first_caps = json.loads(first_wire["contentItems"][0]["text"])
        self.assertEqual(first_caps["capabilities_delivery"], {"status": "inline"})
        self.assertIn("stage_execution", first_caps)
        second_caps = actor_capabilities(cd.project(caps, epoch="e1", current_action_budget={"remaining": 8}))
        self.assertEqual(second_caps["capabilities_delivery"]["status"], "unchanged")
        self.assertNotIn("stage_execution", second_caps)
        self.assertEqual(second_caps["current_action_budget"], {"remaining": 8})
        self.assertEqual(
            cd.project(caps, epoch="e2", current_action_budget={"remaining": 8}, compact_inline=True)["capabilities_delivery"],
            {"status": "inline"},
        )

    def test_present_egress_data_is_idempotent_across_late_registered_actor_ids(self):
        checksum = "abcdef01" * 8
        unknown_id = "frame-1234abcd1234"
        schema = {
            "type": "object",
            "properties": {
                "sha256": {
                    "type": "string",
                    "enum": [checksum],
                    "description": f"Inspect {unknown_id}; build sha256={checksum}",
                },
                "source_url": {
                    "type": "string",
                    "title": f"Reference {unknown_id}",
                },
            },
            "required": ["sha256", "source_url"],
        }
        before = self.presentation.present_egress_data(schema)
        self.assertEqual(before["required"], ["sha256", "source_url"])
        self.assertNotIn(checksum, json.dumps(before))
        self.assertEqual(before["properties"]["sha256"]["description"], f"Inspect {unknown_id}; build ")
        self.assertEqual(before["properties"]["source_url"]["title"], f"Reference {unknown_id}")

        # 后续轮次把未知前缀裸词注册进 actor_id 后，present_egress_data 不得回溯改写早期 Schema 文本
        registered = self.presentation.present({"frame_id": unknown_id})
        self.assertRegex(registered["frame_id"], r"^id_\d+$")
        after = self.presentation.present_egress_data(schema)
        self.assertEqual(
            json.dumps(before, ensure_ascii=False, sort_keys=False, separators=(",", ":")),
            json.dumps(after, ensure_ascii=False, sort_keys=False, separators=(",", ":")),
        )
        # 恢复端仍可还原 enum 中的显式别名
        self.assertEqual(self.presentation.restore(after)["properties"]["sha256"]["enum"], [checksum])


if __name__ == "__main__":
    unittest.main()
