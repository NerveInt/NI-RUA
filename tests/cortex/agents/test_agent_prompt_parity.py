"""Instruction parity of the first actor request across agent backends (work package P4-3).

Each real-binary class captures, once, the first actor generation request its backend sends upstream for the same
benchmark episode start (``support.prompt_capture``: real Workbench instruction/prompt/catalogue assembly, real
session, relay and pinned binary, fake loopback model) and pins what
``docs/development/reports/agent-backend-prompt-parity.md`` records:

* the Workbench ``instructions`` reach the model exactly once and byte-identical in the backend's channel (Codex:
  a developer message part; Claude Code: the tail of the second ``system`` block after the preserved preset;
  OpenCode 2.2.7: its third ``system`` block, or on Responses its top-level ``instructions``, after the neutralised
  ``Instructions:`` header), followed on MCP backends by exactly the one tool-name-prefix sentence;
* the first user turn is the Workbench prompt, byte-identical, with nothing in front of it;
* registry tool descriptions reach the model unchanged, and the per-backend schema transformations stay within the
  recorded sets.

Two catalogue defects found by the first version of this review are fixed and now asserted: the MCP ``tools/list``
projection removes top-level ``anyOf``/``oneOf``/``allOf`` (``task_plan``; Claude Code 2.1.321 silently drops such
tools) and Claude Code runs with ``CLAUDE_CODE_MAX_MCP_DESCRIPTION_LENGTH`` raised (default 4096 truncated
``geometry_query``, ``program_run``, ``robot_action``).

Host-repository boundary (AGENTS.md "Restricted Information Sources"; ``*HostRepositoryBoundary``): with the run
root below a planted host (a real git repository with a branch, a git user, a remote, a commit, an untracked file, an
``AGENTS.md`` and a ``CLAUDE.md``; or an empty ``.git`` directory, as a shared Linux host had in ``/tmp``) nothing of
that host reaches the model on any backend, in any request: the first one, a subagent's or child session's
(Claude Code, OpenCode), those around Claude Code's native compaction, and a second user turn. Compared with a
reference run below no planted host, only the git-derived environment lines differ (Claude Code ``Is a git
repository`` and the memory directory's name; OpenCode ``Is directory a git repo`` and the workspace root). Each
capture's lines are derived from what is above its own directory, the reference's too (a shared host's empty
``/tmp/.git`` puts it below a ``.git``; its lines are checked, not assumed), so a line that differs for any other
reason fails; on such a host a git-dependent line that names no root is the same in both runs and only a host whose
temporary directory has no ``.git`` above it (macOS) can tell it apart. Without their switch the pinned binaries do
leak the host, which each class reproduces: Claude Code's gitStatus block (branch, git user, untracked paths, recent
commit subjects; ``CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS=1`` removes it) and Codex's upward ``AGENTS.md`` discovery to
the git root (``project_root_markers = []`` stops it). OpenCode needs no switch (``OPENCODE_DISABLE_PROJECT_CONFIG``).

Skipped with the reason when a pinned runtime is absent (Codex: ``requires_codex_binary``; Claude Code:
``requires_claude_cli``; OpenCode: ``requires_opencode`` plus ``rg``). No real model, no network; the sandbox is off.
Run: ``PYTHONPATH=tests mamba run -n ni-rua python -m unittest tests.test_agent_prompt_parity -v``.
"""

NI_RUA_TEST_TIER = "component"

from collections import Counter
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock
from support.fake_model_servers import Usage, text_step, tool_step
from support.prompt_capture import analyze, capture, model_visible_text, published_tools, snapshot, text_blocks
from support.requirements import opencode_problem, requires_claude_cli, requires_codex_binary
from ni_rua.apps.gateway.workbench import Workbench
from ni_rua.core.config import Config
from ni_rua.cortex.agents.mcp_server import TOP_LEVEL_COMBINATORS, mcp_tool_catalog

BACKEND_WORDS = ("Codex", "Claude", "OpenCode", "opencode", "tool_search", "mcp__", "ni_rua_")
# Schema keywords a client may drop on its wire (validation bounds, examples, conditionals); recorded, not corrected:
# the generated field guide in every description carries bounds, units and defaults (design 04 R3, P1-10).
RESPONSES_DROPPED = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "minLength", "maxLength",
                     "minItems", "maxItems", "uniqueItems", "minProperties", "pattern", "default", "examples",
                     "const", "dependentRequired", "prefixItems", "if", "then", "else", "not", "format"}
# OpenCode on Responses rewrites const as a one-value enum and spells out implicit object/array shapes.
RESPONSES_ADDED = {"type", "properties", "enum", "additionalProperties"}
# Codex 0.160.0 additionally drops nested property descriptions and a nested oneOf of its two largest parameter
# schemas (robot_action, geometry_query.request); the tool-level description with the field guide stays intact.
CODEX_DROPPED = RESPONSES_DROPPED | {"description", "oneOf", "type"}


def actor_catalogue():
    """The registry actor catalogue of a default run (``Workbench.actor_tools`` without a Harness)."""
    return Workbench.actor_tools(SimpleNamespace(config=Config()))


class RegistrySchemaPortability(unittest.TestCase):
    # Anthropic tool input schemas do not accept a top-level anyOf/oneOf/allOf (Claude Code 2.1.321 drops the tool,
    # OpenCode on anthropic_messages forwards it to the provider). The registry keeps task_plan's top-level allOf; the
    # MCP transport projects it away and the rule stays in the description (tests/test_tool_registry_prose_bounds.py).
    def test_no_mcp_published_tool_schema_has_a_top_level_combinator(self):
        published = mcp_tool_catalog(actor_catalogue(), SimpleNamespace(present_text=str, present_data=dict))
        offending = {tool["name"]: sorted(key for key in TOP_LEVEL_COMBINATORS if key in tool["inputSchema"])
                     for tool in published}
        self.assertEqual({name: keys for name, keys in offending.items() if keys}, {})


class PromptParity:
    """Mixin: one capture per class; ``channel`` = (location, role) of the instructions block, ``first_turn`` =
    location of the first user turn."""
    backend = None
    channel = None
    first_turn = None
    native_identity = None      # first characters of the native base prompt

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.captured = capture(cls.backend)
        cls.facts = analyze(cls.captured)

    def tool(self, name):
        return next(tool for tool in self.facts["tools"] if tool["name"] == name)

    def test_instructions_delivered_once_byte_identical(self):
        facts = self.facts["instructions"]
        self.assertEqual(facts["occurrences"], 1, facts)
        self.assertTrue(facts["delivered_identical"], facts)
        self.assertEqual((facts["location"], facts["role"]), self.channel)

    def test_instructions_name_no_backend(self):
        text = self.captured.instructions.replace(self.captured.run_root, "<run>")   # sandbox off: the source path
        for word in BACKEND_WORDS:
            self.assertNotIn(word, text)

    def test_naming_note_is_the_only_addition(self):
        blocks = {location: text for location, _, text in text_blocks(self.captured)}
        text = blocks[self.facts["instructions"]["location"]]
        start = text.index(self.captured.instructions) + len(self.captured.instructions)
        if self.captured.naming_note:
            self.assertTrue(text[start:].startswith("\n\n" + self.captured.naming_note + "\n"), text[start:start + 300])
            self.assertEqual(sum(block.count(self.captured.naming_note) for block in blocks.values()), 1)
        else:
            self.assertEqual(text[start:], "")

    def test_first_user_turn_is_the_workbench_prompt(self):
        turn = self.facts["first_user_turn"]
        self.assertTrue(turn["identical"], turn)
        self.assertEqual((turn["location"], turn["role"]), (self.first_turn, "user"))

    def test_reply_delivery_notes_reach_the_model_once_in_the_robot_action_description(self):
        """Default configuration: the reply layout, the fixed reply notes, the compact encoding and how to read a same_as
        pointer are stated once,
        in the robot_action description this backend publishes to the model, and in no other tool."""
        from ni_rua.core.presentation import reply_budget, reply_compaction, reply_encoding
        from support.prompt_capture import published_tools
        registry = next(tool for tool in self.captured.tools if tool["name"] == "robot_action")["description"]
        plan = reply_compaction.static_once_plan(
            [dict(tool, description=tool["description"].replace(" Fixed reply notes", "\0").split("\0")[0])
             if tool["name"] == "robot_action" else tool for tool in self.captured.tools], None)
        self.assertLessEqual({"budget", "steps", "grasp_contact", "joint_travel", "trajectory"}, set(plan.segments))
        notes = (reply_budget.SUMMARY_FIRST_CATALOGUE_NOTE, reply_encoding.COMPACT_ENCODING_CATALOGUE_NOTE,
                 reply_compaction.SAME_AS_CATALOGUE_NOTE)
        markers = (" Execution reply layout:", " Fixed reply notes (stated once here", " Compact reply encoding:",
                   ' detail: {"tool", "arguments"} in a reply is a ready call')
        published = [tool for tool in published_tools(self.captured) if isinstance(tool["description"], str)]
        carrying = [tool for tool in published if any(marker in tool["description"] for marker in markers)]
        self.assertEqual([tool["name"] for tool in carrying], [self.captured.tool_prefix + "robot_action"])
        description = carrying[0]["description"]
        for text in (*notes, *markers):
            self.assertEqual((registry.count(text), description.count(text)), (1, 1), text[:40])

    def test_a_tool_reply_reaches_the_model_as_the_harness_shaped_it(self):
        """Default configuration, real tool call: the reply this backend hands to the model is the one
        ``Harness._shape_actor_reply`` produced (notes stated in the catalogue are gone, repeats are pointers)."""
        from ni_rua.core.presentation.reply_compaction import dangling_same_as
        from support.prompt_capture import tool_result_delivery
        shaped = []
        delivered = [text for kind, text in tool_result_delivery(self.backend, shaped=shaped) if kind == "text"]
        (presented, final), = shaped
        texts = [item["text"] for item in final if item["type"] == "inputText"]
        self.assertNotEqual(final, presented)
        self.assertIn('"read_full"', "".join(item.get("text", "") for item in presented))
        self.assertNotIn('"read_full"', "".join(texts))
        self.assertEqual(dangling_same_as(final), [])
        position = 0
        for text in texts:   # every shaped text item arrives, in order, unchanged
            self.assertIn(text, delivered[position:])
            position = delivered.index(text, position) + 1

    def test_native_base_prompt_preserved(self):
        first = text_blocks(self.captured)[0][2]
        self.assertTrue(first.startswith(self.native_identity), first[:200])

    def assert_descriptions_identical(self, names):
        for name in names:
            tool = self.tool(name)
            self.assertTrue(tool["present"], name)
            self.assertTrue(tool["description_identical"], (name, tool["published_description_chars"],
                                                            tool["registry_description_chars"]))

    def assert_schema_changes_within(self, dropped, added):
        for tool in self.facts["tools"]:
            if tool["present"]:
                # MCP transport projection: top-level combinators are removed before the client sees the schema
                projected = set(TOP_LEVEL_COMBINATORS) if tool["top_level_combinators"] else set()
                self.assertLessEqual(set(tool["schema_removed"]), dropped | projected, tool["name"])
                self.assertLessEqual(set(tool["schema_added"]), added, tool["name"])


@requires_codex_binary
class CodexPromptParity(PromptParity, unittest.TestCase):
    backend = "codex"
    channel = ("input[2].content[0]", "developer")   # second developer message, first part; no `instructions` field
    first_turn = "input[6].content[0]"                # after Codex's own <environment_context> user message
    native_identity = "You are Codex"

    def test_eager_tools_unchanged_deferred_tools_behind_tool_search(self):
        """无原生搜索支持的通道默认平铺发布全部可用工具且不暴露失联的 tool_search。"""
        all_names = [tool["name"] for tool in self.facts["tools"]]
        self.assert_descriptions_identical(all_names)
        self.assertEqual({self.tool(name)["namespace"] for name in all_names}, {"functions"})
        self.assertEqual([tool["name"] for tool in self.facts["tools"] if not tool["present"]], [])
        self.assertNotIn("tool_search", [tool["name"] for tool in self.facts["native_tools"]])
        for name in ("robot_action_status", "program_status"):
            if any(tool["name"] == name for tool in self.facts["tools"]):
                self.assertFalse(self.tool(name)["deferred"], name)
        self.assert_schema_changes_within(CODEX_DROPPED, {"type", "properties"})


# The first text of the gitStatus context the pinned Claude Code CLI adds to the first user message inside a git
# repository unless CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS=1 (claude_code/options.py ``_BOUNDARY_FLAGS``).
CLAUDE_GIT_CONTEXT = "<system-reminder>\nAs you answer the user's questions, you can use the following context:\n# gitStatus"
# Headings of that block; none may reach the model.
GIT_CONTEXT_WORDS = ("gitStatus", "Current branch", "Main branch", "Git user", "Recent commits")


@requires_claude_cli
class ClaudeCodePromptParity(PromptParity, unittest.TestCase):
    backend = "claude_code"
    channel = ("system[1]", "system")   # --append-system-prompt-file: tail of the preset's second system block
    first_turn = "messages[0].content[0]"
    native_identity = "You are Claude Code"

    def test_startup_grounding_attaches_inline_overview_image(self):
        """真 CLI 首轮 benchmark 请求携带内联多模态概览图块。"""
        self.assertEqual(self.captured.body["messages"][0]["content"][1]["type"], "image")

    def test_no_git_context(self):
        # Whatever the host has above the run (a shared Linux host had an empty /tmp/.git): see the planted hosts in
        # ClaudeCodeHostRepositoryBoundary.
        text = model_visible_text(self.captured)
        self.assertEqual([word for word in GIT_CONTEXT_WORDS if word in text], [])

    def test_instructions_follow_the_preserved_preset(self):
        facts = self.facts["instructions"]
        self.assertGreater(facts["native_before_chars"], 0)
        self.assertEqual(facts["native_after_chars"], len("\n\n" + self.captured.naming_note + "\n"))

    def test_published_schemas_are_the_registry_schemas(self):
        for tool in self.facts["tools"]:
            if tool["present"]:
                self.assertTrue(tool["schema_portable_identical"], tool["name"])

    def test_no_tool_search_tool(self):
        self.assertNotIn("ToolSearch", [tool["name"] for tool in self.facts["native_tools"]])

    # Defect (P4-3 report, "task_plan schema"): the CLI drops MCP tools whose input schema has a top-level
    # anyOf/oneOf/allOf unless a server-side flag enables its normaliser; task_plan never reaches the model, although
    # every physical action requires a submitted plan.
    def test_every_registry_tool_is_published(self):
        self.assertEqual([tool["name"] for tool in self.facts["tools"] if not tool["present"]], [])

    # Defect (P4-3 report, "description cap"): CLAUDE_CODE_MAX_MCP_DESCRIPTION_LENGTH defaults to 4096 characters;
    # longer MCP descriptions end in "... [truncated]" (geometry_query, program_run, robot_action lose most of their
    # generated field guide).
    def test_tool_descriptions_unchanged(self):
        self.assert_descriptions_identical([tool["name"] for tool in self.facts["tools"] if tool["present"]])


OPENCODE_PROBLEM = opencode_problem() or (None if shutil.which("rg") else "ripgrep (rg) is not installed")


# OpenCode 2.2.7 (the pin; the parity report's OpenCode sizes are from 1.19.5): the native base prompt starts with this
# sentence for both model families. On anthropic_messages it is system[0], system[1] names the model and system[2]
# holds the Code Mode catalogue, then "Instructions:" and the Workbench instructions; on openai_responses all of it
# is the top-level ``instructions`` field and the first input message is the Workbench prompt.
OPENCODE_IDENTITY = "You are an AI agent running in OpenCode"


@unittest.skipIf(OPENCODE_PROBLEM is not None, f"pinned OpenCode runtime unavailable: {OPENCODE_PROBLEM}")
class OpenCodeAnthropicPromptParity(PromptParity, unittest.TestCase):
    backend = "opencode_anthropic"
    channel = ("system[2]", "system")
    first_turn = "messages[0].content[0]"
    native_identity = OPENCODE_IDENTITY

    def test_instructions_header_is_neutralised(self):
        self.assertEqual(self.facts["instructions"]["header"], "Instructions:")

    def test_every_tool_published_with_unchanged_description_and_schema(self):
        self.assert_descriptions_identical([tool["name"] for tool in self.facts["tools"]])
        for tool in self.facts["tools"]:
            self.assertTrue(tool["schema_portable_identical"], tool["name"])


@unittest.skipIf(OPENCODE_PROBLEM is not None, f"pinned OpenCode runtime unavailable: {OPENCODE_PROBLEM}")
class OpenCodeResponsesPromptParity(PromptParity, unittest.TestCase):
    backend = "opencode_responses"
    channel = ("instructions", "instructions")
    first_turn = "input[0].content[0]"
    native_identity = OPENCODE_IDENTITY

    def test_instructions_header_is_neutralised(self):
        self.assertEqual(self.facts["instructions"]["header"], "Instructions:")

    def test_every_tool_published_with_unchanged_description(self):
        self.assert_descriptions_identical([tool["name"] for tool in self.facts["tools"]])
        self.assert_schema_changes_within(RESPONSES_DROPPED, RESPONSES_ADDED)


# ---- what a host keeps of one tool result (AgentSession.tool_output_text_budget_bytes) -----------------------
def _filler(size, tag):
    return "".join(f"{tag}{index:09d}" + "." * 90 for index in range(0, size + 100, 100))[:size]


_PIXEL = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJg"
          "gg==")


def _items(*sizes):
    return [dict(type="inputImage", imageUrl=_PIXEL) if size == "image"
            else dict(type="inputText", text=_filler(size, "ABCDEFGH"[index])) for index, size in enumerate(sizes)]


@requires_claude_cli
class ClaudeCodeToolResultLimit(unittest.TestCase):
    """The pinned Claude Code CLI under this harness's settings: a text result of at most 500,000 characters reaches
    the model whole; one character group more and the whole result becomes a file pointer. That number is the
    budget the session reports."""

    def test_the_session_budget_is_the_measured_limit(self):
        from ni_rua.cortex.agents.claude_code import options
        from ni_rua.cortex.agents.claude_code.session import ClaudeCodeSession
        from support.prompt_capture import tool_result_delivery
        limit = options.TOOL_RESULT_TEXT_LIMIT_CHARS
        self.assertEqual((limit, ClaudeCodeSession.tool_output_text_budget_bytes(None)), (500_000, 500_000))
        at_limit = _items(limit // 2, limit // 2)
        delivered = [text for kind, text in tool_result_delivery("claude_code", at_limit) if kind == "text"]
        self.assertEqual(delivered, [item["text"] for item in at_limit])
        over = [text for kind, text in tool_result_delivery("claude_code", _items(limit // 2, limit // 2 + 100))
                if kind == "text"]
        self.assertEqual(len(over), 1)
        self.assertTrue(over[0].startswith("<persisted-output>"), over[0][:120])
        self.assertLess(len(over[0]), 5000)


@unittest.skipIf(OPENCODE_PROBLEM is not None, f"pinned OpenCode runtime unavailable: {OPENCODE_PROBLEM}")
class OpenCodeToolResultLimit(unittest.TestCase):
    """The pinned OpenCode binary delivers MCP tool results far larger than any reply of this harness whole, text
    item by text item, so the session reports no budget (replies are ordered, nothing is replaced)."""

    def test_no_limit_up_to_six_hundred_thousand_bytes(self):
        from ni_rua.cortex.agents.opencode.session import OpenCodeSession
        from support.prompt_capture import tool_result_delivery
        self.assertIsNone(OpenCodeSession.tool_output_text_budget_bytes(None))
        items = _items(200_000, "image", 200_000, 200_000)
        delivered = tool_result_delivery("opencode_anthropic", items)
        self.assertEqual([text for kind, text in delivered if kind == "text"],
                         [item["text"] for item in items if item["type"] == "inputText"])
        self.assertEqual(sum(kind != "text" for kind, _ in delivered), 1)


class ExternalHostHasNoKnownLimit(unittest.TestCase):
    def test_external_and_base_sessions_report_no_budget(self):
        from ni_rua.cortex.agents.base import AgentSession
        from ni_rua.cortex.agents.external.session import ExternalSession
        self.assertIsNone(AgentSession.tool_output_text_budget_bytes(None))
        self.assertIs(ExternalSession.tool_output_text_budget_bytes, AgentSession.tool_output_text_budget_bytes)


# ---- host-repository boundary -------------------------------------------------------------------------------
GIT = shutil.which("git", path=os.defpath) or shutil.which("git")
# "plain": nothing planted above the run (the reference); the other two are the hosts under test. The host may still
# have a ".git" above its temporary directory (a shared Linux host had an empty /tmp/.git): every capture's git-derived
# lines are derived from what is above its own directory (``git_entry_root``), never assumed.
HOST_KINDS = ("plain", "repository", "empty_git")
ACTOR_AGENTS_MD = "ACTOR-AGENTS-MD-MARKER: written by the agent in its own directory\n"
SECOND_TURN = "SECOND-TURN-MARKER: continue with the task."
# Text Claude Code appends to the last user message of its auto-compaction summary request (spike S5).
COMPACTION_MARK = "create a detailed summary"
# Headers only a subagent's / child session's requests carry (Claude Code, OpenCode 2.2.7) or a Codex native helper's.
SUBAGENT_HEADERS = ("x-claude-code-agent-id", "x-opencode-parent-session-id")
HELPER_HEADER = "x-openai-subagent"


def plant_host(root, kind):
    """Create the host directory ``root`` of ``kind`` and return ``(directory for the run root, markers)``.

    ``"repository"``: a git repository on branch ``feat/host-branch-marker`` with a local git user, a remote, one
    commit ("HOST-COMMIT-MARKER ...") of ``AGENTS.md``, ``CLAUDE.md`` and a tracked file, plus an untracked file.
    ``"empty_git"``: an empty ``.git`` directory next to the same ``AGENTS.md`` and ``CLAUDE.md``. ``"plain"``: the
    same two files and no ``.git``. ``markers``: the planted names and texts (the full commit hash included); none may
    reach the model. The run root goes to ``root/runs``, like a checkout's ``runs/``.
    """
    root.mkdir(parents=True)
    (root / "AGENTS.md").write_text("HOST-AGENTS-MD-MARKER: instructions of the host repository\n")
    (root / "CLAUDE.md").write_text("HOST-CLAUDE-MD-MARKER: instructions of the host repository\n")
    markers = ["HOST-AGENTS-MD-MARKER", "HOST-CLAUDE-MD-MARKER"]
    if kind == "empty_git":
        (root / ".git").mkdir()
    elif kind == "repository":
        def git(*args):
            return subprocess.run([GIT, "-c", "commit.gpgsign=false", *args], cwd=root, check=True, text=True,
                                  capture_output=True, env=dict(PATH=os.defpath, HOME=str(root), GIT_CONFIG_NOSYSTEM="1",
                                                                GIT_CONFIG_GLOBAL=os.devnull)).stdout.strip()
        git("init", "-q")
        git("symbolic-ref", "HEAD", "refs/heads/feat/host-branch-marker")
        git("config", "user.name", "Host User Marker")
        git("config", "user.email", "host-user-marker@example.invalid")
        git("remote", "add", "origin", "https://example.invalid/host-remote-marker.git")
        (root / "host-tracked-marker.txt").write_text("tracked\n")
        git("add", "AGENTS.md", "CLAUDE.md", "host-tracked-marker.txt")
        git("commit", "-q", "-m", "HOST-COMMIT-MARKER internal subject")
        (root / "host-untracked-marker.txt").write_text("untracked\n")
        markers += ["host-branch-marker", "Host User Marker", "host-user-marker@", "host-remote-marker",
                    "HOST-COMMIT-MARKER", "host-tracked-marker", "host-untracked-marker", git("rev-parse", "HEAD")]
    (root / "runs").mkdir()
    return root / "runs", markers


def removed_segment(longer, shorter):
    """The one contiguous piece of ``longer`` that ``shorter`` lacks (``AssertionError`` when it is not one)."""
    start = len(os.path.commonprefix([longer, shorter]))
    end = len(os.path.commonprefix([longer[start:][::-1], shorter[start:][::-1]]))
    if longer[:start] + longer[len(longer) - end:] != shorter:
        raise AssertionError("more than one segment differs")
    return longer[start:len(longer) - end]


def _last_item(body):
    items = body.get("messages") or body.get("input") or [None]
    return json.dumps(items[-1], ensure_ascii=False)


def declares_tools(body):
    """Whether a generation request publishes tools (top-level ``tools``, or Codex's ``additional_tools`` item)."""
    return bool(body.get("tools")) or any(isinstance(item, dict) and item.get("type") == "additional_tools"
                                          for item in body.get("input") or [])


def request_kind(request):
    """``"subagent"`` (a subagent's or child session's request), ``"compaction"`` (Claude Code's summary request),
    ``"second_turn"`` (the request of the scripted second user turn), ``"actor"`` (any other main-loop generation
    request with tools) or ``"other"`` (non-generation requests, requests without tools such as a session title,
    native helpers such as Codex memory consolidation)."""
    body = request.json if isinstance(request.json, dict) else None
    if not request.is_generation or body is None or request.header(HELPER_HEADER) or not declares_tools(body):
        return "other"
    if any(request.header(name) for name in SUBAGENT_HEADERS):
        return "subagent"
    last = _last_item(body)
    if COMPACTION_MARK in last:
        return "compaction"
    if SECOND_TURN in last:
        return "second_turn"
    return "actor"


def body_text(request):
    """Everything of a request body the model can read (every field except the ``metadata`` / ``client_metadata``
    request metadata), as JSON text."""
    return json.dumps({key: value for key, value in request.json.items() if key not in ("metadata", "client_metadata")},
                      ensure_ascii=False)


def path_forms(path):
    """``path`` as the binaries may print it: as is, without / with macOS ``/private``, and dash-sanitised (Claude Code
    names per-project directories that way)."""
    path = os.fspath(path)
    forms = {path, path.removeprefix("/private"), "/private" + path if path.startswith("/var/") else path}
    return forms | {sanitised(form) for form in forms}


def normalised(text, captured, host):
    """``text`` with ``captured``'s run root as ``<run>``, its temporary root as ``<tmp>``, the host directory as
    ``<host>`` and the thread id as ``<thread>``."""
    replacements = [(form, "<run>") for form in path_forms(captured.run_root)]
    replacements += [(form, "<tmp>") for form in path_forms(Path(captured.run_root).parent)]
    replacements += [(form, "<host>") for form in path_forms(host)]
    for value, marker in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
        text = text.replace(value, marker)
    if captured.thread_id:
        text = text.replace(captured.thread_id, "<thread>")
    return text


def normalised_lines(captured, host):
    """Lines of the first actor request's model-visible text, ``normalised``."""
    return normalised(model_visible_text(captured), captured, host).splitlines()


def git_entry_root(directory):
    """The nearest directory at or above ``directory`` holding a ``.git`` entry (a directory, also an empty one, or a
    file), None when there is none. Claude Code 2.1.x takes it as its git root; OpenCode 2.2.7 asks git there."""
    directory = Path(directory)
    for candidate in (directory, *directory.parents):
        entry = candidate / ".git"
        if entry.is_dir() or entry.is_file():
            return candidate
    return None


def git_toplevel(directory):
    """``git rev-parse --show-toplevel`` run in ``directory`` (None when ``directory`` is None or git finds no
    repository from there, e.g. at an empty ``.git``): how OpenCode 2.2.7 resolves the ``git_entry_root`` it found
    (``Git.repo.discover``)."""
    if directory is None:
        return None
    result = subprocess.run([GIT, "rev-parse", "--show-toplevel"], cwd=directory, text=True, capture_output=True,
                            env=dict(PATH=os.defpath, HOME=str(directory), GIT_CONFIG_NOSYSTEM="1",
                                     GIT_CONFIG_GLOBAL=os.devnull))
    return Path(result.stdout.strip()) if result.returncode == 0 and result.stdout.strip() else None


def sanitised(path):
    """``path`` as Claude Code names a per-project directory after it (every character but ASCII letters and digits
    becomes ``-``)."""
    return re.sub(r"[^A-Za-z0-9]", "-", os.fspath(path))


class HostRepositoryBoundary:
    """Mixin: one capture per host kind (``captures[kind] = (Capture, markers)``, ``hosts[kind]`` = the host directory).

    ``first_turn``: location of the Workbench prompt in the backend's parity layout. ``actor_files``: ``capture(
    actor_files=)``. ``subagent_call``: ``(name, arguments)`` of the native subagent tool the scripted model calls on the
    first actor request in the "repository" capture (None: no subagent). ``compaction``: also drive the backend into
    its native compaction there (Claude Code). The "repository" capture then has the subagent's requests, the
    post-compaction requests and a second user turn (``SECOND_TURN``), so the boundary is checked past the first request.
    ``git_derived(captured, host)``: the pieces of a capture's ``normalised`` model-visible text that the backend
    derives from the git facts of its actor directory, ``{name: (text, whole_line)}`` (lines of the native environment
    block, or substrings such as a path named after the git root), computed here from the same facts (``git_entry_root``
    and ``git_toplevel``, never from the capture). Every capture must show its own; between a host and the "plain"
    reference only those that differ may differ, and nothing else (default: no git-derived piece at all).
    """
    backend = None
    first_turn = None
    actor_files = None
    subagent_call = None
    compaction = False
    # Claude Code compacts after two consecutive main answers reporting ~90k prompt tokens against a 100k window
    # (test_agent_claude_code_conformance, spike S5).
    HIGH_USAGE = Usage(input_tokens=2000, output_tokens=200, cache_read_input_tokens=88000)

    def git_derived(self, captured, host):
        return {}

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        temporary = tempfile.TemporaryDirectory(prefix="ni-rua-host-boundary-")
        cls.addClassCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        cls.hosts, cls.captures = {}, {}
        for kind in HOST_KINDS:
            cls.hosts[kind] = root / kind
            parent, markers = plant_host(cls.hosts[kind], kind)
            options = cls.later_requests() if kind == "repository" else {}
            cls.captures[kind] = (capture(cls.backend, parent=parent, actor_files=cls.actor_files, **options), markers)

    @classmethod
    def later_requests(cls):
        """``capture`` options of the "repository" capture: the scripted model calls ``subagent_call`` on the first
        actor request, reports high usage on the next two actor requests when ``compaction``, and once the first turn
        has ended a second user turn is submitted; the capture ends at the second turn's first request."""
        answered, state = [], {}

        def respond(request):
            kind = request_kind(request)
            if kind == "subagent":
                return text_step("SUBAGENT-REPORT")
            if kind == "compaction":
                return text_step("<analysis>Scripted.</analysis>\n<summary>Scripted summary.</summary>")
            if kind != "actor":
                return text_step("done")
            answered.append(request)
            if len(answered) == 1 and cls.subagent_call is not None:
                return tool_step(*cls.subagent_call)
            if cls.compaction and len(answered) in (2, 3):
                return tool_step("Bash", {"command": "true", "description": "No-op"}, usage=cls.HIGH_USAGE)
            return text_step("done")

        def until(requests, session):
            kinds = [request_kind(request) for request in requests]
            ready = (cls.subagent_call is None or "subagent" in kinds) and (not cls.compaction or "compaction" in kinds)
            if ready and "turn" not in state and session.state != "running":
                state["turn"] = session.run_turn(SECOND_TURN)
            return "second_turn" in kinds
        options = dict(respond=respond, until=until)
        if cls.compaction:
            options["agent_options"] = {"auto_compact_window_tokens": 100000}
        return options

    def generation_requests(self, captured):
        return [request for request in captured.requests if request_kind(request) != "other"]

    def test_nothing_of_the_host_reaches_the_model_in_any_request(self):
        for kind in ("repository", "empty_git"):
            captured, markers = self.captures[kind]
            for index, request in enumerate(self.generation_requests(captured)):
                with self.subTest(host=kind, request=index, kind=request_kind(request)):
                    text = body_text(request)
                    self.assertEqual([marker for marker in (*markers, *GIT_CONTEXT_WORDS) if marker in text], [])

    def test_the_later_capture_points_were_reached(self):
        kinds = [request_kind(request) for request in self.captures["repository"][0].requests]
        expected = {"actor", "second_turn"} | ({"subagent"} if self.subagent_call else set()) | (
            {"compaction"} if self.compaction else set())
        self.assertLessEqual(expected, set(kinds), kinds)
        if self.compaction:   # an actor request after the (first) compaction
            self.assertIn("actor", kinds[kinds.index("compaction"):], kinds)

    def test_first_user_turn_is_the_workbench_prompt_with_nothing_before_it(self):
        for kind in HOST_KINDS:
            captured, _ = self.captures[kind]
            with self.subTest(host=kind):
                turn = analyze(captured)["first_user_turn"]
                self.assertEqual((turn["identical"], turn["location"]), (True, self.first_turn))

    def derived(self, kind):
        captured = self.captures[kind][0]
        return self.git_derived(captured, self.hosts[kind])

    def test_every_capture_shows_the_git_facts_of_its_own_directory(self):
        # The reference is established, not assumed: the "plain" capture too reports what is above its directory.
        for kind in HOST_KINDS:
            lines = normalised_lines(self.captures[kind][0], self.hosts[kind])
            for name, (text, whole_line) in self.derived(kind).items():
                with self.subTest(host=kind, piece=name):
                    if whole_line:
                        self.assertIn(text, lines)
                    else:
                        self.assertIn(text, "\n".join(lines))

    def test_only_the_git_derived_lines_differ_from_the_reference_run(self):
        plain = normalised_lines(self.captures["plain"][0], self.hosts["plain"])
        reference = self.derived("plain")
        for kind in ("repository", "empty_git"):
            derived = self.derived(kind)
            lines_expected = {text: reference[name][0] for name, (text, whole_line) in derived.items()
                              if whole_line and text != reference[name][0]}
            substitutions = [(text, reference[name][0]) for name, (text, whole_line) in derived.items()
                             if not whole_line and text != reference[name][0]]
            with self.subTest(host=kind):
                inside = normalised_lines(self.captures[kind][0], self.hosts[kind])
                text = "\n".join(inside)
                for found, replacement in substitutions:
                    self.assertIn(found, text)
                    text = text.replace(found, replacement)
                inside = text.splitlines()
                only_inside = sorted((Counter(inside) - Counter(plain)).elements())
                only_plain = sorted((Counter(plain) - Counter(inside)).elements())
                self.assertEqual(only_inside, sorted(lines_expected))
                self.assertEqual(only_plain, sorted(lines_expected.values()))


@unittest.skipIf(GIT is None, "git is not installed")
@requires_claude_cli
class ClaudeCodeHostRepositoryBoundary(HostRepositoryBoundary, unittest.TestCase):
    backend = "claude_code"
    first_turn = "messages[0].content[0]"
    subagent_call = ("Agent", {"description": "Inspect the scene", "prompt": "Report what you see.",
                               "subagent_type": "general-purpose", "run_in_background": False})
    compaction = True

    def git_derived(self, captured, host):
        # Below any ".git" (also an empty one) the "# Environment" block says so, and the run-private memory directory
        # is named after the directory holding ".git" instead of the actor directory; no repository content.
        actor = Path(captured.run_root) / "actor"
        root = git_entry_root(actor)
        return {"is_git": (" - Is a git repository: " + ("true" if root else "false"), True),
                "memory": (normalised(f"/claude-config/projects/{sanitised(root or actor)}/memory/", captured, host),
                           False)}

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from ni_rua.cortex.agents.claude_code import options
        with mock.patch.dict(options._BOUNDARY_FLAGS, clear=True):   # the native default, in the same repository
            parent = cls.hosts["repository"] / "runs"
            cls.native = capture(cls.backend, parent=parent, **cls.later_requests())

    def test_without_the_switch_the_cli_leaks_the_host_repository(self):
        """The leak the switch removes, reproduced at every capture point: the boundary tests above are not vacuous."""
        blocks = {location: text for location, _, text in text_blocks(self.native)}
        block = blocks["messages[0].content[0]"]
        self.assertTrue(block.startswith(CLAUDE_GIT_CONTEXT), block[:200])
        self.assertEqual(blocks["messages[0].content[1]"], self.native.prompt)
        requests = self.generation_requests(self.native)
        kinds = [request_kind(request) for request in requests]
        after_compaction = kinds.index("compaction") + kinds[kinds.index("compaction"):].index("actor")
        for index in (0, kinds.index("subagent"), after_compaction, kinds.index("second_turn")):
            with self.subTest(request=index, kind=kinds[index]):
                text = body_text(requests[index])
                for marker in ("feat/host-branch-marker", "Host User Marker", "host-untracked-marker",
                               "HOST-COMMIT-MARKER"):
                    self.assertIn(marker, text)

    def test_the_switch_changes_only_the_git_context_and_the_bash_git_section(self):
        captured, markers = self.captures["repository"]
        ours, native = snapshot(captured), snapshot(self.native)
        system = [block for block in ours["blocks"] if block["location"].startswith("system")]
        self.assertEqual(system, [block for block in native["blocks"] if block["location"].startswith("system")])
        self.assertEqual(ours["fields"], native["fields"])
        tools = {tool["name"]: tool["description"] for tool in published_tools(captured)}
        native_tools = {tool["name"]: tool["description"] for tool in published_tools(self.native)}
        self.assertEqual(list(tools), list(native_tools))
        self.assertEqual([name for name in tools if tools[name] != native_tools[name]], ["Bash"])
        # The CLI's generic commit/PR advice (no host data) is the one other thing the switch removes.
        removed = removed_segment(native_tools["Bash"], tools["Bash"])
        self.assertTrue(removed.startswith("\n\n# Git\n"), removed[:200])
        self.assertEqual([marker for marker in markers if marker in removed], [])


@unittest.skipIf(GIT is None, "git is not installed")
@requires_codex_binary
class CodexHostRepositoryBoundary(HostRepositoryBoundary, unittest.TestCase):
    backend = "codex"
    first_turn = "input[6].content[0]"
    actor_files = {"AGENTS.md": ACTOR_AGENTS_MD}

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from ni_rua.cortex.agents.codex import session as codex
        with mock.patch.object(codex, "BOUNDARY_CONFIG_LINES", ()):   # the native default, in the same repository
            cls.native = capture(cls.backend, parent=cls.hosts["repository"] / "runs", actor_files=cls.actor_files)

    def test_without_the_switch_codex_reads_the_host_agents_md(self):
        """The leak the switch removes, reproduced: the boundary tests above are not vacuous."""
        self.assertIn("HOST-AGENTS-MD-MARKER", model_visible_text(self.native))

    def test_the_actors_own_agents_md_is_still_read(self):
        for kind in HOST_KINDS:
            with self.subTest(host=kind):
                self.assertIn(ACTOR_AGENTS_MD.strip(), model_visible_text(self.captures[kind][0]))

    def test_the_switch_changes_only_the_agents_md_block(self):
        ours, native = snapshot(self.captures["repository"][0]), snapshot(self.native)
        self.assertEqual([block["location"] for block in ours["blocks"]],
                         [block["location"] for block in native["blocks"]])
        differ = [(a["location"], a["text"], b["text"]) for a, b in zip(ours["blocks"], native["blocks"]) if a != b]
        self.assertEqual(len(differ), 1, [location for location, _, _ in differ])
        _, text, native_text = differ[0]
        self.assertTrue(text.startswith("# AGENTS.md instructions for "), text[:200])
        self.assertIn("HOST-AGENTS-MD-MARKER", removed_segment(native_text, text))
        self.assertEqual(sorted(ours["tools"], key=str), sorted(native["tools"], key=str))


OPENCODE_BOUNDARY_PROBLEM = OPENCODE_PROBLEM or (None if GIT else "git is not installed")


def opencode_git_derived(captured, host):
    """OpenCode 2.2.7 finds a repository by walking up for ".git" and asking git for its top level there
    (``Git.repo.discover``; an empty ".git" is no repository) and has no switch for it: its "<env>" block then names the
    repository root as the workspace root (outside a repository: the actor directory). No repository content."""
    actor = Path(captured.run_root) / "actor"
    top = git_toplevel(git_entry_root(actor))
    return {"is_git": ("  Is directory a git repo: " + ("yes" if top else "no"), True),
            "workspace": (normalised("  Workspace root folder: " + os.fspath(top or actor), captured, host), True)}


OPENCODE_SUBAGENT = ("subagent", {"agent": "general", "description": "Inspect the scene",
                                  "prompt": "Report what you see."})


@unittest.skipIf(OPENCODE_BOUNDARY_PROBLEM is not None, f"unavailable: {OPENCODE_BOUNDARY_PROBLEM}")
class OpenCodeAnthropicHostRepositoryBoundary(HostRepositoryBoundary, unittest.TestCase):
    backend = "opencode_anthropic"
    first_turn = "messages[0].content[0]"
    subagent_call = OPENCODE_SUBAGENT

    def git_derived(self, captured, host):
        return opencode_git_derived(captured, host)


@unittest.skipIf(OPENCODE_BOUNDARY_PROBLEM is not None, f"unavailable: {OPENCODE_BOUNDARY_PROBLEM}")
class OpenCodeResponsesHostRepositoryBoundary(HostRepositoryBoundary, unittest.TestCase):
    backend = "opencode_responses"
    first_turn = "input[0].content[0]"
    subagent_call = OPENCODE_SUBAGENT

    def git_derived(self, captured, host):
        return opencode_git_derived(captured, host)

if __name__ == "__main__":
    unittest.main()
