"""Byte-identity of the WireAdapter extraction against fixtures recorded at c4ff16a.

The fixtures were captured by running ``tests/wire_golden_fixture.py`` against the
unmodified c4ff16a source tree (see that module's docstring); this test replays the same
scenarios through the current relay and egress scrub.
"""

NI_RUA_TEST_TIER = "component"

import json
import unittest
from ni_rua.cortex.agents.codex.session import DYNAMIC_TOOL_NAMESPACE
from ni_rua.cortex.session.wire.openai_responses import (
    OpenAIResponsesWire,
    iter_sse_json_events,
    parse_sse_data,
    present_responses_request,
)
import support.wire_golden_fixture as golden

RETIRED_PRESENTATION_CASES = frozenset({"lean_projection_block", "lean_projection_trailing", "request_dump"})
UPDATED_FORWARDING_SCENARIOS = frozenset({"projection_rejected_relay_error"})

class ForwardingGoldenTest(unittest.TestCase):
    maxDiff = 4000

    @classmethod
    def setUpClass(cls):
        cls.recorded = json.loads(golden.FORWARDING_FIXTURE.read_text())
        cls.current = json.loads(json.dumps(golden.capture_forwarding(), sort_keys=True, ensure_ascii=False))

    def test_fixture_comes_from_pre_extraction_revision(self):
        self.assertEqual(self.recorded["captured_from"], {"revision": "c4ff16a"})
        self.assertEqual(set(self.current), set(self.recorded["scenarios"]))
        self.assertEqual(len(self.current), len(golden.scenarios()))

    def test_forwarded_upstream_bodies_are_byte_identical(self):
        for name, recorded in self.recorded["scenarios"].items():
            with self.subTest(scenario=name):
                current = self.current[name]["upstream"]
                self.assertEqual([item["body"] for item in current], [item["body"] for item in recorded["upstream"]])
                self.assertEqual([(item["path"], item["headers"]) for item in current],
                                 [(item["path"], item["headers"]) for item in recorded["upstream"]])

    def test_client_visible_answers_are_byte_identical(self):
        for name, recorded in self.recorded["scenarios"].items():
            with self.subTest(scenario=name):
                self.assertEqual(self.current[name]["client"], recorded["client"])

    def test_evidence_events_and_relay_state_are_unchanged(self):
        for name, recorded in self.recorded["scenarios"].items():
            if name in UPDATED_FORWARDING_SCENARIOS:
                continue
            with self.subTest(scenario=name):
                self.assertEqual(self.current[name]["state"], recorded["state"])
                self.assertEqual([event["kind"] for event in self.current[name]["events"]],
                                 [event["kind"] for event in recorded["events"]])
                self.assertEqual(self.current[name]["events"], recorded["events"])
        rejected = self.current["projection_rejected_relay_error"]
        self.assertIsNone(rejected["state"]["blocked_error"])
        self.assertEqual(
            [event["kind"] for event in rejected["events"]],
            [
                "model_call_started",
                "model_request_body_received",
                "model_request_rejected",
                "model_call_failed",
                "model_call_finished",
            ],
        )
        self.assertEqual(rejected["events"][2]["reason"], "sandbox_remote_model_tool_disabled")
        self.assertFalse(rejected["events"][3]["manual_retry_required"])

    def test_scenarios_cover_the_required_paths(self):
        scenarios = self.recorded["scenarios"]
        kinds = lambda name: [event["kind"] for event in scenarios[name]["events"]]
        self.assertIn("model_images_trimmed", kinds("image_count_trim_and_exact_reuse"))
        self.assertIn("model_images_reused", kinds("image_count_trim_and_exact_reuse"))
        self.assertIn("model_images_trimmed", kinds("image_body_bytes_trim"))
        self.assertEqual(scenarios["compaction_route_epoch"]["state"]["image_context_epoch"], "<transport>:1")
        self.assertEqual(scenarios["budget_denial"]["client"][1]["status"], 429)
        self.assertEqual(scenarios["thread_cancelled_denial"]["client"][0]["status"], 409)
        self.assertEqual(scenarios["failed_terminal_then_transport_blocked_denial"]["client"][1]["status"], 409)
        self.assertIn("model_request_retry", kinds("retryable_503_then_success"))
        self.assertIn("model_transport_recovered", kinds("transport_recovery_after_503"))
        self.assertEqual(len(scenarios["image_rejected_then_trimmed_retry"]["upstream"]), 2)
        bodies = scenarios["image_rejected_then_trimmed_retry"]["upstream"]
        self.assertLess(bodies[1]["body"]["bytes"], bodies[0]["body"]["bytes"])
        for name in ("failed_terminal_then_transport_blocked_denial", "incomplete_terminal", "worker_failed_terminal"):
            self.assertNotIn("unsafe", scenarios[name]["client"][0]["body"]["text"])


class PresentResponsesGoldenTest(unittest.TestCase):
    maxDiff = 4000

    @classmethod
    def setUpClass(cls):
        cls.recorded = json.loads(golden.PRESENTATION_FIXTURE.read_text())

    def test_codex_present_model_request_matches_pre_extraction_output(self):
        self.assertEqual(self.recorded["captured_from"], {"revision": "c4ff16a"})
        current = json.loads(json.dumps(golden.capture_presentation(), sort_keys=True, ensure_ascii=False))
        expected = {
            name: {k: v for k, v in rec.items() if k != "dumps"}
            for name, rec in self.recorded["cases"].items()
            if name not in RETIRED_PRESENTATION_CASES
        }
        self.assertEqual(set(current), set(expected))
        for name, recorded in expected.items():
            with self.subTest(case=name):
                self.assertEqual(current[name], recorded)

    def test_present_responses_request_alone_matches_old_scrub(self):
        def present(session, request):
            return present_responses_request(request, session.presentation, deferred_tools=session.deferred_tools,
                                             namespace=DYNAMIC_TOOL_NAMESPACE)
        current = json.loads(json.dumps(golden.capture_presentation(present), sort_keys=True, ensure_ascii=False))
        for name, recorded in self.recorded["cases"].items():
            if name in RETIRED_PRESENTATION_CASES:
                continue
            recorded = {key: value for key, value in recorded.items() if key != "dumps"}
            with self.subTest(case=name):
                self.assertEqual(current[name], recorded)

    def test_planted_metadata_is_scrubbed_and_protocol_fields_are_untouched(self):
        request = golden.planted_request()
        output = self.recorded["cases"]["planted_every_field"]["output"]
        items = {item.get("type"): item for item in request["input"]}
        projected = {item.get("type"): item for item in output["input"]}
        for scrubbed in (output["instructions"], output["input"][0]["content"][0]["text"],
                         projected["reasoning"]["summary"][0]["text"], projected["function_call"]["arguments"],
                         projected["custom_tool_call"]["input"], projected["custom_tool_call_output"]["output"],
                         json.dumps(projected["tool_search_call"]["arguments"]),
                         output["tools"][0]["description"], output["tools"][2]["description"]):
            self.assertNotIn(golden.DIGEST, scrubbed)
        self.assertNotIn(golden.ARTIFACT, json.dumps(output["input"]))
        self.assertIn("artifact:ref_", output["instructions"])
        for kind, field in (("reasoning", "encrypted_content"), ("reasoning", "id"), ("function_call", "call_id"),
                            ("function_call", "id"), ("compaction", "encrypted_content"), ("compaction", "id"),
                            ("custom_tool_call", "call_id"), ("tool_search_call", "call_id")):
            self.assertEqual(projected[kind][field], items[kind][field], (kind, field))
        self.assertEqual(output["input"][1]["content"][1], request["input"][1]["content"][1])
        for key in ("model", "reasoning", "include", "prompt_cache_key", "store", "stream", "metadata"):
            self.assertEqual(output[key], request[key], key)
        self.assertEqual(output["tools"][2]["format"], request["tools"][2]["format"])

    def test_scrub_is_deterministic_for_repeated_requests(self):
        from pathlib import Path
        import tempfile
        from ni_rua.core.storage.store import Store
        from ni_rua.cortex.session.egress_cache import EgressProjectionCache
        directory = self.enterContext(tempfile.TemporaryDirectory())
        store = Store(Path(directory) / "run", dict(run_id="golden", episode_id="episode",
                                                    env_instance_id="environment", env_generation=1))
        self.addCleanup(store.close)
        session = golden.fake_session(store, Path(directory), {})
        first = present_responses_request(golden.planted_request(), session.presentation,
                                          deferred_tools=session.deferred_tools, namespace="ni_rua")
        second = present_responses_request(golden.planted_request(), session.presentation,
                                           deferred_tools=session.deferred_tools, namespace="ni_rua")
        cache = EgressProjectionCache()
        cached_first = present_responses_request(golden.planted_request(), session.presentation,
                                                 deferred_tools=session.deferred_tools, namespace="ni_rua", cache=cache)
        cached_second = present_responses_request(golden.planted_request(), session.presentation,
                                                  deferred_tools=session.deferred_tools, namespace="ni_rua", cache=cache)
        self.assertEqual(json.dumps(first, sort_keys=False), json.dumps(second, sort_keys=False))
        self.assertEqual(json.dumps(first, sort_keys=False), json.dumps(cached_first, sort_keys=False))
        self.assertEqual(json.dumps(first, sort_keys=False), json.dumps(cached_second, sort_keys=False))


class SSEStreamParserConsolidationTests(unittest.TestCase):
    """Unified SSE parser coverage for CRLF, LF, CR, multi-line data:, comments, [DONE], and error semantics."""

    def setUp(self):
        self.wire = OpenAIResponsesWire()
        self.route = self.wire.route("POST", "/responses")

    def test_parse_sse_data_and_iter_sse_json_events_handle_crlf_cr_multiline_and_comments(self):
        self.assertIsNone(parse_sse_data(b""))
        self.assertIsNone(parse_sse_data(b":keepalive\r\n\r\n"))
        self.assertIsNone(parse_sse_data(b"event: ping\r\ndata: [DONE]"))
        # Optional single leading space after data: is stripped per SSE spec, preserving extra spaces and newlines.
        self.assertEqual(parse_sse_data(b'data: {"a": 1}\r\ndata:{"b": 2}'), '{"a": 1}\n{"b": 2}')
        self.assertEqual(parse_sse_data(b"data:  indented"), " indented")

        crlf_stream = (
            b":keepalive\r\n\r\n"
            b"event: response.created\r\n"
            b'data: {"type":"response.created",\r\n'
            b'data:"response":{"id":"r1","status":"in_progress"}}\r\n\r\n'
            b"\r\n"
            b"event: response.completed\r\n"
            b'data: {"type": "response.completed",\r\n'
            b'data:  "response": {"id": "r1", "model": "gpt-m", "status": "completed", "output": [], '
            b'"usage": {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4}}}\r\n\r\n'
            b"data: [DONE]\r\n\r\n"
        )
        events = list(iter_sse_json_events(crlf_stream))
        self.assertEqual([ev["type"] for ev in events], ["response.created", "response.completed"])

        cr_only_stream = crlf_stream.replace(b"\r\n", b"\r")
        self.assertEqual(list(iter_sse_json_events(cr_only_stream)), events)

        complete, evidence, terminal = self.wire.response_evidence(crlf_stream, self.route)
        self.assertTrue(complete)
        self.assertEqual(evidence["response_id"], "r1")
        self.assertEqual(evidence["served_model"], "gpt-m")
        self.assertEqual(terminal["status"], "completed")

    def test_emitted_tool_calls_and_error_semantics_preserved_across_crlf_multiline(self):
        tool_stream = (
            b":keepalive\r\n\r\n"
            b"event: response.output_item.done\r\n"
            b'data: {"type": "response.output_item.done", "output_index": 1,\r\n'
            b'data: "item": {"type": "function_call", "call_id": "c2", "name": "t2", "arguments": "{\\"k\\": 2}"}}\r\n\r\n'
            b"event: response.output_item.done\r\n"
            b'data:{"type": "response.output_item.done", "output_index": 0,\r\n'
            b'data:"item": {"type": "function_call", "call_id": "c1", "name": "t1", "arguments": "{\\"k\\": 1}"}}\r\n\r\n'
            b'data: {"type": "response.completed", "response": {"id": "r2", "status": "completed", "output": []}}\r\n\r\n'
            b"data:[DONE]\r\n\r\n"
        )
        self.assertEqual(
            self.wire.emitted_tool_calls(tool_stream),
            [("c1", "t1", {"k": 1}), ("c2", "t2", {"k": 2})],
        )

        # terminal_sse_type returns terminal types cleanly and None on invalid UTF-8 / malformed JSON.
        self.assertEqual(
            self.wire.terminal_sse_type(b'event: response.completed\r\ndata:{"type":\r\ndata: "response.completed"}'),
            "response.completed",
        )
        self.assertIsNone(self.wire.terminal_sse_type(b"data: \xff\xfe"))
        self.assertIsNone(self.wire.terminal_sse_type(b"data: {malformed"))
        self.assertIsNone(self.wire.terminal_sse_type(b":keepalive"))

        # response_evidence and emitted_tool_calls continue raising ValueError on malformed JSON in data:.
        with self.assertRaises(ValueError):
            self.wire.response_evidence(b"data: {malformed\r\n\r\n", self.route)
        with self.assertRaises(ValueError):
            self.wire.emitted_tool_calls(b"data: {malformed\r\n\r\n")

if __name__ == "__main__":
    unittest.main()
