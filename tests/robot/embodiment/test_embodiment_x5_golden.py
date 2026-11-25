"""ARX-X5 / RoboDojo parity against the golden fixtures frozen at 5fd93c6 (design 13 §5.6, work package WP0).

Every family of ``tests/fixtures/embodiment/x5/`` (tool catalogue, instructions, startup text, compiled actions,
grip-guard decisions, effective configuration) is recaptured from the current tree through the runtime's own builders
(``support.embodiment_golden``) and must render byte-identically. The single allowance is ``APPROVED_CORRECTION``
(design 13 §6: grip-guard preset minimums 2 / 4 / 15 mm -> 1.5 / 2 / 3 mm in five robot_action strings, 21
occurrences, +7 bytes): a catalogue item may instead equal the recorded item with exactly that correction applied, and
then every catalogue item that carries robot_action must be corrected. The fixtures were recorded on
``golden.RECORDING_HOST`` (macOS arm64), where the comparison is byte-exact. On another host an item may differ from
them only by ``golden.host_noise`` (floats within 4 ULP; a recorded path whose root that host resolves differently, at a
Config path field), every such item must be pinned for that host in ``golden.HOST_TOLERANCE`` when the host is listed
there, and each family prints its host-tolerance lines on every run. The SO-101 fixtures record current SO-101
behaviour, expected to change; they are checked for provenance only (print their diff with
``python -m support.embodiment_golden --diff --embodiment so101``).

No model, network, GPU, robot or native simulator: the X5 worker capability document comes from the real
``RoboDojoAdapter.capabilities`` builder on a stub adapter, startup runs the real ``Workbench`` with ``FakeAgentSession``.
"""

NI_RUA_TEST_TIER = "component"

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from support import embodiment_golden as golden


class FixtureProvenanceTests(unittest.TestCase):
    def test_every_fixture_comes_from_the_clean_baseline_revision(self):
        for embodiment in golden.EMBODIMENTS:
            for family in golden.FAMILIES:
                with self.subTest(embodiment=embodiment, family=family):
                    header, items = golden.load_fixture(embodiment, family)
                    self.assertEqual(header["schema"], golden.SCHEMA)
                    self.assertEqual((header["embodiment"], header["family"]), (embodiment, family))
                    self.assertEqual(header["label"], golden.LABELS[embodiment])
                    self.assertTrue(header["captured_from"]["revision"].startswith(golden.BASELINE_REVISION))
                    self.assertIs(header["captured_from"]["production_tree_clean"], True)
                    self.assertTrue(items)

    def test_so101_snapshot_is_labelled_as_expected_to_change(self):
        self.assertIn("expected to change", golden.SO101_LABEL)
        self.assertIn("never asserted", golden.SO101_LABEL)
        _, startup = golden.load_fixture("so101", "startup")
        self.assertIn('Its only arm id is "right"', startup["so101_gateway/instructions"])


class ApprovedCorrectionTests(unittest.TestCase):
    """The allowance is exactly design 13 §6, checked on the recorded fixture itself."""

    @classmethod
    def setUpClass(cls):
        _, cls.catalogue = golden.load_fixture("x5", "catalogue")

    def robot_action(self, name="definitions/default"):
        return next(item for item in self.catalogue[name] if item["name"] == "robot_action")

    def test_recorded_robot_action_carries_the_old_numbers_at_exactly_five_sites(self):
        correction = golden.APPROVED_CORRECTION
        definition = self.robot_action()
        olds = [old for old, _ in correction["replacements"]]
        counts = {}
        for pointer, text in golden.string_leaves(definition):
            found = {old: text.count(old) for old in olds if old in text}
            if found:
                counts[pointer] = found
        self.assertEqual(counts, correction["sites"])
        self.assertEqual(sum(sum(found.values()) for found in counts.values()), correction["occurrences"])
        self.assertEqual(len(correction["sites"]), 5)
        self.assertEqual(correction["occurrences"], 21)
        corrected = golden.correct_definition(definition)
        self.assertEqual(len(golden.render(corrected).encode()) - len(golden.render(definition).encode()), 7)
        self.assertFalse(any(old in golden.render(corrected) for old in olds))

    def test_corrected_numbers_are_the_recorded_runtime_preset_minimums(self):
        _, guard = golden.load_fixture("x5", "guard")
        minimum = {name: bounds[0] for name, bounds in guard["preset_aperture_bounds"].items()}
        self.assertEqual((minimum["non_empty"], minimum["rigid_object"], minimum["wide_object"], minimum["thin_object"]),
                         (0.0015, 0.002, 0.003, 0.0005))
        news = dict(golden.APPROVED_CORRECTION["replacements"])
        self.assertEqual(news["non_empty [min 0.002m]"], "non_empty [min 0.0015m]")
        self.assertEqual(news["rigid_object [>=4mm]"], "rigid_object [>=2mm]")
        self.assertEqual(news["wide_object [>=15mm]"], "wide_object [>=3mm]")

    def test_correction_allows_only_its_own_bytes(self):
        name = "actor/ni_rua_benchmark"
        recorded = self.catalogue[name]
        corrected = golden.correct_catalogue(recorded)
        # The correction alone, on the recorded item: later packages' catalogue allowances (WP2 V-F) are not applied.
        result = golden.compare("x5", "catalogue", {name: corrected}, allow_correction=True, allow_allowances=False)
        self.assertEqual(result["states"][name], "corrected")
        # One more byte anywhere else is a change.
        tampered = deepcopy(corrected)
        tampered[0]["description"] += " "
        self.assertEqual(golden.compare("x5", "catalogue", {name: tampered})["states"][name], "changed")
        # A partial correction (five of the six replacements) is a change.
        partial = deepcopy(recorded)
        action = next(item for item in partial if item["name"] == "robot_action")
        for pointer in golden.APPROVED_CORRECTION["sites"]:
            text = golden.pointer_get(action, pointer)
            for old, new in golden.APPROVED_CORRECTION["replacements"][:-1]:
                text = text.replace(old, new)
            golden.pointer_set(action, pointer, text)
        self.assertEqual(golden.compare("x5", "catalogue", {name: partial})["states"][name], "changed")
        # The correction is never applied outside the catalogue family or to SO-101.
        self.assertFalse(golden.correction_applies("startup", recorded))
        self.assertEqual(golden.compare("x5", "catalogue", {name: corrected}, allow_correction=False)["states"][name],
                         "changed")


def float_leaves(value, pointer=""):
    """``[(JSON pointer, float)]`` of every float leaf, in document order."""
    if isinstance(value, dict):
        return [leaf for key, item in value.items() for leaf in float_leaves(item, f"{pointer}/{key}")]
    if isinstance(value, list):
        return [leaf for index, item in enumerate(value) for leaf in float_leaves(item, f"{pointer}/{index}")]
    return [(pointer, value)] if isinstance(value, float) else []


def ulps_away(value, count):
    for _ in range(count):
        value = math.nextafter(value, math.inf)
    return value


LINUX = "Linux x86_64"
# How a Linux host resolves the recorded roots (no symlinks): the aliases host_path_aliases() returns there.
LINUX_ALIASES = golden.host_path_aliases(resolve=lambda root: root)
ZERO = {"floats": 0, "max_ulps": 0, "paths": 0, "references": 0}


class HostToleranceTests(unittest.TestCase):
    """``host_noise``: nothing on the recording host; elsewhere float rounding within FLOAT_ULPS and the host's own
    path resolution at Config path fields, nothing else."""

    def test_ulp_distance(self):
        self.assertEqual(golden.ulp_distance(0.3, 0.30000000000000004), 1)
        self.assertEqual(golden.ulp_distance(0.25000000000000006, 0.25), 1)
        self.assertEqual(golden.ulp_distance(0.0, -0.0), 0)
        self.assertEqual(golden.ulp_distance(-1.0, ulps_away(-1.0, 3)), 3)
        self.assertGreater(golden.ulp_distance(-5e-324, 5e-324), 1)

    def test_the_recording_host_tolerates_nothing(self):
        self.assertEqual(golden.RECORDING_HOST, "Darwin arm64")
        value = {"position_m": [0.30000000000000004], "checkpoint": "/System/Volumes/Data/home/ckpt/59999"}
        self.assertEqual(golden.host_noise(value, deepcopy(value), host=golden.RECORDING_HOST), ZERO)
        for different in ({"position_m": [0.3], "checkpoint": value["checkpoint"]},
                          {"position_m": value["position_m"], "checkpoint": "/home/ckpt/59999"}):
            self.assertIsNone(golden.host_noise(value, different, host=golden.RECORDING_HOST,
                                                path_pointers={"/checkpoint"}, aliases=LINUX_ALIASES))
        self.assertEqual(golden.host_path_aliases(resolve=lambda root: golden.RECORDED_ROOT_RESOLUTION[root]), ())
        self.assertEqual(golden.unexpected_host_tolerance("config", {"load/ni_rua": "x"}, golden.RECORDING_HOST),
                         {"load/ni_rua": "x"})

    def test_floats_within_four_ulp_and_nothing_else_differ(self):
        self.assertEqual(golden.FLOAT_ULPS, 4)
        value = {"position_m": [0.30000000000000004, 0.25, 0.6499], "steps": 3, "label": "move"}

        def with_x(x):
            changed = deepcopy(value)
            changed["position_m"][0] = x
            return changed
        self.assertEqual(golden.host_noise(value, deepcopy(value), host=LINUX), ZERO)
        self.assertEqual(golden.host_noise(value, with_x(0.3), host=LINUX), dict(ZERO, floats=1, max_ulps=1))
        self.assertEqual(golden.host_noise(value, with_x(ulps_away(0.30000000000000004, 4)), host=LINUX)["max_ulps"], 4)
        for different in (with_x(ulps_away(0.30000000000000004, 5)),           # beyond the tolerance
                          with_x(0.3001),                                        # a real change
                          dict(value, steps=3.0), dict(value, steps=True),       # another JSON type
                          {"steps": 3, "position_m": value["position_m"], "label": "move"},   # key order
                          dict(value, label="move "), dict(value, position_m=value["position_m"][:2])):
            with self.subTest(different=different):
                self.assertIsNone(golden.host_noise(value, different, host=LINUX))

    def test_a_path_is_respelled_only_where_this_host_resolves_differently(self):
        self.assertEqual(dict(LINUX_ALIASES), {"/System/Volumes/Data/home": "/home", "/private/tmp": "/tmp",
                                               "/private/var": "/var", "/private/etc": "/etc"})
        recorded, linux = "/System/Volumes/Data/home/ckpt/59999", "/home/ckpt/59999"
        options = dict(host=LINUX, aliases=LINUX_ALIASES, path_pointers={"/policy/checkpoint"})
        self.assertEqual(golden.host_noise({"policy": {"checkpoint": recorded}}, {"policy": {"checkpoint": linux}},
                                           **options), dict(ZERO, paths=1))
        for before, after, pointer in (
                (recorded, "/home/ckpt/5999", "/policy/checkpoint"),              # another path
                (linux, recorded, "/policy/checkpoint"),                          # the other direction
                (recorded, linux, "/policy/python"),                              # not a declared path field
                ("x" + recorded, "x" + linux, "/policy/checkpoint"),             # not a leading prefix
                ("/private/tmpfile", "/tmpfile", "/policy/checkpoint")):          # not a whole component
            with self.subTest(before=before, after=after, pointer=pointer):
                self.assertIsNone(golden.host_noise({"policy": {pointer.rsplit("/", 1)[1]: before}},
                                                    {"policy": {pointer.rsplit("/", 1)[1]: after}}, **options))
        # A host that resolves /home like the recording host respells nothing.
        same_home = golden.host_path_aliases(resolve=lambda root: golden.RECORDED_ROOT_RESOLUTION[root]
                                             if root == "/home" else root)
        self.assertIsNone(golden.host_noise({"policy": {"checkpoint": recorded}}, {"policy": {"checkpoint": linux}},
                                            **dict(options, aliases=same_home)))
        self.assertIn("/policy/checkpoint", golden.config_path_pointers())
        self.assertIn("/model/codex_home", golden.config_path_pointers())

    def test_a_home_below_a_symlink_is_redacted_in_both_spellings(self):
        # Config resolves its path fields, so with HOME given through a symlink (macOS /tmp, a linked Linux /home) a
        # '~' default arrives in the resolved spelling; both spellings are the home.
        from ni_rua.core.config import Config
        with tempfile.TemporaryDirectory() as directory:
            real = Path(directory).resolve() / "real-home"
            real.mkdir()
            link = real.with_name("linked-home")
            link.symlink_to(real)
            with mock.patch.dict(os.environ, {"HOME": str(link)}):
                redact = golden.Redactor()
                self.assertEqual(redact([str(real / ".codex"), str(link / "exp")]), ["~/.codex", "~/exp"])
                self.assertEqual(redact(Config.from_dict({}).model.codex_home), "~/.codex")
        self.assertNotIn("/policy/server_url", golden.config_path_pointers())

    def test_a_capture_does_not_follow_links_in_the_account_home(self):
        # Config resolves '~' defaults through whatever the account keeps under its home (a linked
        # ~/.local/share/ni-rua, a bin/python link to python3.11); capture() runs under a private empty home, so the
        # recording host and every comparing host see the same spelling.
        def grasp_memory(_embodiment):
            return golden.Redactor()(golden.load_config().features.grasp_memory_dir)
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory).resolve()
            (home / ".local" / "share").mkdir(parents=True)
            (home / "elsewhere").mkdir()
            (home / ".local" / "share" / "ni-rua").symlink_to(home / "elsewhere")
            with mock.patch.dict(os.environ, {"HOME": str(home)}), \
                    mock.patch.dict(golden.CAPTURES, {"config": grasp_memory}):
                self.assertEqual(grasp_memory("x5"), "~/elsewhere/grasp-memory")   # the account's link, followed
                self.assertEqual(golden.capture("x5", "config"), "~/.local/share/ni-rua/grasp-memory")

    def test_compare_accepts_lists_and_pins_host_effects_on_recorded_items(self):
        _, recorded = golden.load_fixture("x5", "config")
        name = "load/ni_rua"
        expected, _ = golden.apply_allowances("x5", "config", name, recorded[name])
        self.assertTrue(expected["policy"]["checkpoint"].startswith("/System/Volumes/Data/home/"))  # recorded on macOS
        linux = deepcopy(expected)
        linux["policy"]["checkpoint"] = golden.host_path_spelling(expected["policy"]["checkpoint"], LINUX_ALIASES)
        result = golden.compare("x5", "config", {name: linux}, host=LINUX, aliases=LINUX_ALIASES)
        self.assertEqual((result["states"][name], result["host"][name]), ("equal", "1 path(s) in another host spelling"))
        self.assertEqual(golden.unexpected_host_tolerance("config", result["host"], LINUX), {})
        self.assertIn("load/ni_rua: 1 path(s) in another host spelling", golden.host_report(result))
        self.assertEqual(golden.compare("x5", "config", {name: linux})["states"][name],
                         "changed" if golden.current_host() == golden.RECORDING_HOST else result["states"][name])
        linux["policy"]["checkpoint"] += "0"
        self.assertEqual(golden.compare("x5", "config", {name: linux}, host=LINUX,
                                        aliases=LINUX_ALIASES)["states"][name], "changed")

        _, recorded = golden.load_fixture("x5", "compile")
        name = next(name for name, item in recorded.items()
                    if any(number == 0.30000000000000004 for _, number in float_leaves(item)))
        expected, _ = golden.apply_allowances("x5", "compile", name, recorded[name])
        pointer = next(pointer for pointer, number in float_leaves(expected) if number == 0.30000000000000004)
        rounded = deepcopy(expected)
        golden.pointer_set(rounded, pointer, 0.3)
        result = golden.compare("x5", "compile", {name: rounded}, host=LINUX)
        self.assertEqual((result["states"][name], result["host"][name]), ("equal", "1 float(s) within 1 ULP"))
        # Pinned for Linux: this item tolerates 2 floats there; 1 is another description and is reported.
        self.assertEqual(golden.unexpected_host_tolerance("compile", result["host"], LINUX),
                         {name: "1 float(s) within 1 ULP"})
        self.assertEqual(golden.unexpected_host_tolerance("compile", result["host"], "Other host"), {})
        golden.pointer_set(rounded, pointer, 0.31)
        self.assertEqual(golden.compare("x5", "compile", {name: rounded}, host=LINUX)["states"][name], "changed")


class X5GoldenTests(unittest.TestCase):
    """Recapture each family from the current tree and compare with the fixture."""

    maxDiff = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._results = {}

    def check(self, family):
        if family in self._results:
            return self._results[family]
        result = golden.compare("x5", family)
        print(golden.host_report(result), file=sys.stderr)   # every run, also when green
        allowed = {"equal", "corrected"} if family == "catalogue" else {"equal"}
        bad = {name: state for name, state in result["states"].items() if state not in allowed}
        if bad:
            self.fail("X5 changed (design 13: only APPROVED_CORRECTION may differ)\n" + golden.report(result))
        unpinned = golden.unexpected_host_tolerance(family, result["host"])
        if unpinned:
            self.fail(f"host tolerance not pinned for {result['compared_on']} (review, then add to "
                      f"golden.HOST_TOLERANCE): {json.dumps(unpinned, indent=1)}")
        self._results[family] = result
        return result

    def test_tool_catalogue(self):
        result = self.check("catalogue")
        _, recorded = golden.load_fixture("x5", "catalogue")
        carrying = [name for name, value in recorded.items() if golden.correction_applies("catalogue", value)]
        self.assertGreater(len(carrying), 10)
        states = {result["states"][name] for name in carrying}
        self.assertEqual(len(states), 1, "the approved correction must change every robot_action catalogue at once: "
                         + json.dumps({name: result["states"][name] for name in carrying}))
        uncarrying = {result["states"][name] for name in recorded if name not in carrying}
        self.assertLessEqual(uncarrying, {"equal"})

    def test_actor_instructions(self):
        self.check("instructions")

    def test_startup_text_and_first_turn(self):
        self.check("startup")

    def test_compiled_actions(self):
        self.check("compile")

    def test_grip_guard_decisions(self):
        result = self.check("guard")
        _, recorded = golden.load_fixture("x5", "guard")
        self.assertEqual(recorded["stream"]["cases"], 140483)
        self.assertEqual(result["states"]["stream"], "equal")

    def test_effective_configuration(self):
        self.check("config")


if __name__ == "__main__":
    unittest.main()
