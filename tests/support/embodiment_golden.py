"""Golden fixtures of what the ARX-X5 / RoboDojo actor sees and what the runtime decides (design 13 §5.6, WP0).

Frozen at ``BASELINE_REVISION`` (5fd93c6) before the embodiment-decoupling refactor, so every later work package
can prove "X5 unchanged" mechanically. Six families, each captured through the builders the runtime itself uses:

``catalogue``     every tool definition the actor receives: ``public_definitions(**flags)`` for the perception flag
                  combinations ``create_registry`` passes, and the registry actor / program catalogues of a real
                  Workbench start for each shipped configuration (``robot_action`` schema included);
``instructions``  ``actor_instructions.instructions`` for calibrated / legacy / depth contracts x sandbox x operation
                  memory x handoff policy, ``obs3d.developer_instructions`` per feature set, pose-skill, held-perception,
                  visual-verification and depth sentences;
``startup``       the complete startup instructions, first actor turn, tool names, published runtime capabilities and
                  the archived task-prompt record of a real ``Workbench.start`` with ``FakeAgentSession`` (no binary,
                  no model, no network) for each shipped X5 configuration;
``compile``       schema admission (``ToolRegistry.validate``), ``compile_request`` output, ``validate_batch_action``
                  dependencies and the backend validator for a fixed corpus of stage requests (every stage field at
                  its minimum, maximum and just outside, every enum value, both arms, both pose forms, batch fields,
                  stage-count limits, continuation and obstacle persistence);
``guard``         grip-guard decisions: the seeded 140k-case differential of ``stage_grip_guard`` /
                  ``after_stage_grip_guard`` (stream digests, chunk digests, the first cases in full) plus exhaustive
                  ``validate_grip_guard_fields`` / ``preset_aperture_bounds`` / ``evaluate_grip_guard`` tables;
``config``        ``Config().to_dict()``, ``motion_thresholds()``, each shipped configuration as loaded (no environment,
                  no local file) and the ``effective_config.json`` each startup scenario wrote.

Fixtures (``tests/fixtures/embodiment/<embodiment>/<family>.json``): ``x5`` is asserted byte-identical by
``tests/test_embodiment_x5_golden.py``; ``so101`` holds the same families for the fake SO-101 (``SO101_LABEL``:
current behaviour, expected to change), never asserted, so later packages can print their SO-101 diff.

Byte identity: every item is compared as ``render(value)`` (order-preserving compact JSON, UTF-8, no NaN), so key
order, number spelling and every character count. Run-specific values are redacted before rendering (``Redactor``:
temporary run root -> ``<run>``, home directory -> ``~``, repository -> ``<repo>``, ``new_id`` identifiers and artifact
references -> ordinal placeholders). The only allowed X5 difference is ``APPROVED_CORRECTION`` (design 13 §6): applied
to a recorded catalogue item it must reproduce the current item exactly; anything else fails. The fixtures were recorded
on ``RECORDING_HOST`` (macOS arm64), where the comparison stays byte-exact. On another host ``host_noise`` (see "host
tolerance") accepts exactly two host effects, floats within ``FLOAT_ULPS`` units in the last place and a recorded path
whose root that host resolves differently; ``HOST_TOLERANCE`` pins the items each known host tolerates, and
``host_report`` lists them on every run.

Fixture files are valid JSON with one item per line; texts are stored line by line and an item close to an earlier one
as line edits of it (``$derive`` / ``$edits``, see "fixture files"); ``write_fixture`` re-reads each file and checks
that every item renders byte-identically to the capture.

Regenerate (from the worktree root; refuses to overwrite existing fixtures without ``--force`` and refuses an
``ni_rua/`` / ``configs/`` tree that differs from HEAD without ``--allow-dirty``; records the source commit)::

    PYTHONPATH=tests mamba run -n ni-rua python -m support.embodiment_golden --write [--force] [--family F] [--embodiment E]

Differential report against the fixtures (prints changed items; X5 shows whether the approved correction explains
them)::

    PYTHONPATH=tests mamba run -n ni-rua python -m support.embodiment_golden --diff [--embodiment so101]
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import difflib
import functools
import hashlib
import json
import math
import os
import random
import re
import struct
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPOSITORY = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPOSITORY / "tests" / "fixtures" / "embodiment"
BASELINE_REVISION = "5fd93c6"
FAMILIES = ("catalogue", "instructions", "startup", "compile", "guard", "config")
EMBODIMENTS = ("x5", "so101")
X5_LABEL = ("ARX-X5 / RoboDojo behaviour frozen at 5fd93c6 before the embodiment-decoupling refactor; asserted "
            "byte-identical by tests/test_embodiment_x5_golden.py except APPROVED_CORRECTION (design 13 section 6)")
SO101_LABEL = ("current SO-101 behaviour at 5fd93c6, expected to change: reference for the SO-101 diff of later "
               "work packages, never asserted")
LABELS = {"x5": X5_LABEL, "so101": SO101_LABEL}
TASK = "Stack the three bowls together."
# Text values containing a newline are stored as {"$lines": [...]} so fixture diffs are line-oriented.
LINES_KEY = "$lines"

# ---- the one approved X5 difference (design 13 section 6) ------------------------------------------------------
# Grip-guard preset minimums typed in the robot_action field help (2 / 4 / 15 mm) become the runtime values
# (GRIP_GUARD_PRESETS: non_empty 0.0015 m, rigid_object 0.002 m, wide_object 0.003 m). Five published strings,
# 21 occurrences, +7 bytes per catalogue that contains robot_action. Nothing else may differ.
APPROVED_CORRECTION = dict(
    id="design-13-section-6-grip-guard-preset-minimums",
    tool="robot_action",
    replacements=(
        ("non_empty [min 0.002m]", "non_empty [min 0.0015m]"),
        ("rigid_object [min 0.004m]", "rigid_object [min 0.002m]"),
        ("wide_object [min 0.015m]", "wide_object [min 0.003m]"),
        ("non_empty [>=2mm]", "non_empty [>=1.5mm]"),
        ("rigid_object [>=4mm]", "rigid_object [>=2mm]"),
        ("wide_object [>=15mm]", "wide_object [>=3mm]"),
    ),
    # JSON-pointer path inside the robot_action definition -> {old substring: occurrences}
    sites={
        "/description": {"non_empty [min 0.002m]": 2, "rigid_object [min 0.004m]": 2, "wide_object [min 0.015m]": 2,
                         "non_empty [>=2mm]": 1, "rigid_object [>=4mm]": 1, "wide_object [>=15mm]": 1},
        "/inputSchema/properties/stages/items/properties/grip_guard_preset/description":
            {"non_empty [min 0.002m]": 1, "rigid_object [min 0.004m]": 1, "wide_object [min 0.015m]": 1},
        "/inputSchema/properties/stages/items/allOf/0/then/properties/grip_guard_preset/description":
            {"non_empty [min 0.002m]": 1, "rigid_object [min 0.004m]": 1, "wide_object [min 0.015m]": 1},
        "/inputSchema/properties/stages/items/allOf/1/then/properties/grip_guard_preset/description":
            {"non_empty [min 0.002m]": 1, "rigid_object [min 0.004m]": 1, "wide_object [min 0.015m]": 1},
        "/inputSchema/properties/grip_guard_preset/description":
            {"non_empty [min 0.002m]": 1, "rigid_object [min 0.004m]": 1, "wide_object [min 0.015m]": 1},
    },
    occurrences=21,
    byte_delta=7,
)

# ---- approved X5 allowances of later work packages (design 13 sections 8 and 10) ---------------------------------
# Every package that makes an approved X5-visible change registers it here, in its own delimited block: one record per
# change, naming the embodiment, the fixture family, the exact items and per item the exact edits, plus the total edit
# count. Two edit forms: ``edits`` {item: [(JSON pointer inside the item, old value, new value)]} (ABSENT = the key does
# not exist) and ``replacements`` {item: [(JSON pointer of a text inside the item, "" = the item itself, old substring,
# new substring, occurrences)]} (the text must hold exactly that many occurrences of the old substring, all replaced).
# compare() takes the recorded item with every applicable edit applied as the item's expectation (each old value must
# be exactly the recorded one) and reports "equal" only when the current tree reproduces that expectation byte for
# byte; the ids of the allowances used are listed in result["allowed"] and by report(). Any other byte, or a missing
# allowed change, is "changed". APPROVED_CORRECTION keeps its own mechanism above.
ABSENT = "$absent"
ALLOWANCES = []

# --- WP5 allowances ---
# V-E (design 13 sections 8 and 10, approved): the seven threshold fields no code reads (finding F3) are removed from
# FeaturesConfig and core.config.MOTION_THRESHOLD_FIELDS. Documents that still carry them load unchanged
# (core.config.RETIRED_FEATURE_FIELDS: accepted and dropped). X5 effect: family "config" only; each of the 21 items
# below loses exactly these seven keys (recorded value -> absent): 21 x 7 = 147 removals. Nothing else changes.
_WP5_RETIRED_FIELDS = (("soft_position_tolerance_m", 0.01), ("grip_thin_max_aperture_m", 0.065),
                       ("grip_rigid_max_aperture_m", 0.087), ("grip_wide_max_aperture_m", 0.0875),
                       ("max_min_grip_m", 0.088), ("closed_jaw_goal_max", 0.05),
                       ("checkpoint_empty_jaw_measured_max", 0.03))
_WP5_V_E_ITEMS = {  # item -> JSON pointer of the object holding the seven keys
    "Config()": "/features", "motion_thresholds()": "",
    "load/ni_rua": "/features", "motion_thresholds/ni_rua": "",
    "load/rgbd_feedback_v2": "/features", "motion_thresholds/rgbd_feedback_v2": "",
    "load/rgbd_numeric_v1": "/features", "motion_thresholds/rgbd_numeric_v1": "",
    "load/rgbd_skills_v3": "/features", "motion_thresholds/rgbd_skills_v3": "",
    "load/benchmark_shared_memory": "/features", "motion_thresholds/benchmark_shared_memory": "",
    "load/ni_rua_example": "/features", "motion_thresholds/ni_rua_example": "",
    "effective_config/ni_rua_benchmark": "/features", "effective_config/ni_rua_interactive": "/features",
    "effective_config/rgbd_feedback_v2": "/features", "effective_config/rgbd_numeric_v1": "/features",
    "effective_config/rgbd_skills_v3": "/features", "effective_config/legacy_rgb_proprio_v1": "/features",
    "effective_config/prompt_capture_fixture": "/features",
}
ALLOWANCES.append(dict(
    id="design-13-V-E-unread-threshold-fields", package="WP5", embodiment="x5", family="config",
    edits={item: [(f"{base}/{key}", old, ABSENT) for key, old in _WP5_RETIRED_FIELDS]
           for item, base in _WP5_V_E_ITEMS.items()},
    occurrences=147,
))
# --- end WP5 ---

# --- WP4 allowances ---
# V-D (design 13 sections 8 and 10, approved; owner of the perception hints): the ARX-X5 full opening is the one profile
# value gripper.full_open_aperture_m = 0.088 m (joint7 + joint8 travel) instead of the pad gap 0.0888 m that
# perception/external/{pose_skill,object_table,obs3d,held_perception}.py and runtime/actions/grasp_preview_3d.py typed.
# X5-visible text: "88.8 mm" -> "88 mm" in the pose_plan description and the obs3d developer instruction, once per
# item: 2 catalogue + 5 instruction + 7 startup items = 14 occurrences, -2 bytes per item. Nothing else in the goldens
# changes. The opening_check / tight / too_wide boundaries (0.0888 -> 0.088 m, not in any golden) and V-C (scene_obstacles
# pad offset 0.147 -> 0.1501 m along link6 +x, no golden item) are pinned by tests/test_embodiment_perception.py
# (X5PerceptionDifferentialTests): every X5 hint, audit and obstacle output equals the 5fd93c6 code except these.
_WP4_POSE_PLAN = ("check against the 88.8 mm maximum opening", "check against the 88 mm maximum opening")
_WP4_OBS3D = ("against the 88.8 mm max opening", "against the 88 mm max opening")
ALLOWANCES.append(dict(
    id="design-13-V-D-x5-full-opening-in-perception-text", package="WP4", embodiment="x5", family="catalogue",
    replacements={item: [("/20/description", *_WP4_POSE_PLAN, 1)]
                  for item in ("definitions/pose_skill", "definitions/perception_shipped_pose_skill")},
    occurrences=2,
))
ALLOWANCES.append(dict(
    id="design-13-V-D-x5-full-opening-in-perception-text", package="WP4", embodiment="x5", family="instructions",
    replacements={"obs3d/" + name: [("", *_WP4_OBS3D, 1)]
                  for name in ("config_default", "ni_rua_json", "vggt_default_mode", "obs3d_action_overlay",
                               "scene3d_observe_off")},
    occurrences=5,
))
ALLOWANCES.append(dict(
    id="design-13-V-D-x5-full-opening-in-perception-text", package="WP4", embodiment="x5", family="startup",
    replacements={scenario + "/instructions": [("", *_WP4_OBS3D, 1)]
                  for scenario in ("ni_rua_benchmark", "ni_rua_interactive", "rgbd_feedback_v2", "rgbd_numeric_v1",
                                   "rgbd_skills_v3", "legacy_rgb_proprio_v1", "prompt_capture_fixture")},
    occurrences=7,
))
# Not golden-visible, pinned by tests/test_embodiment_perception.py instead (exact old -> new):
WP4_DIFFERENTIAL = dict(
    v_c=dict(path="ni_rua/robot/motion/scene_obstacles.py _stage_grasp_center_m", old_m=0.147, new_m=0.1501,
             effect="waypoint grasp centre (target exemption xy) moves 3.1 mm along link6 +x"),
    v_d=dict(path="opening_check / grasp_3d_review / is_horizontally_elongated_graspable / held_perception._holding",
             old_m=0.0888, new_m=0.088, effect="opening boundaries shift by 0.8 mm; pose_plan max_opening_m 0.0888 -> 0.088"),
)
# --- end WP4 ---

# --- WP2 allowances ---
# V-F (design 13 sections 8 and 10, approved): the stage-level max_upward_dz_m help states the default windows the X5
# batch executor applies (robot/motion/batch_motion.py endpoint(): max(pos_tol, 0.022 | 0.040 | 0.018) and
# max(1.2*pos_tol, 0.025) m; vertical compliance off without contact_stop for tight/fine presets or pos_tol < 0.009 m).
# The old text omitted the max(pos_tol, ...) terms and the 0.009 m condition. One sentence of robot_action changes, at
# three sites per catalogue that carries robot_action: the stage field help in stages.items.properties and in the move
# branch stages.items.allOf[0].then, and the generated input field guide in the description. 23 items x 3 = 69.
# The batch-level max_upward_dz_m help names no defaults and is unchanged. Nothing else changes (the approved section 6
# correction keeps its own mechanism).
_WP2_V_F = (
    'Defaults when omitted: 0.040 m with stall_guard_preset="contact_stop" (0.022 m on open-jaw pre-grasp), '
    'max(1.2*pos_tol, 0.025 m) on closed-jaw moves, 0.018 m on normal open-jaw moves, and 0 m on tight moves without '
    'contact_stop.',
    'Defaults when omitted: max(pos_tol, 0.040 m) with stall_guard_preset="contact_stop" (max(pos_tol, 0.022 m) on '
    'open-jaw pre-grasp), max(1.2*pos_tol, 0.025 m) on closed-jaw moves, max(pos_tol, 0.018 m) on other open-jaw '
    'moves, and 0 m without contact_stop on tight moves or when pos_tol (the effective position tolerance) is below '
    '0.009 m.',
)
_WP2_V_F_SITES = ("/description", "/inputSchema/properties/stages/items/properties/max_upward_dz_m/description",
                  "/inputSchema/properties/stages/items/allOf/0/then/properties/max_upward_dz_m/description")
_WP2_ROBOT_ACTION_INDEX = {  # catalogue item -> index of robot_action in the recorded tool list
    **{"definitions/" + name: 18 for name in ("default", "pose_skill", "perception_shipped",
                                              "perception_shipped_pose_skill", "vggt_default_mode",
                                              "obs3d_action_overlay", "object_table_observe_only",
                                              "scene3d_observe_only", "vggt_only", "object_table_only",
                                              "scene3d_only")},
    **{f"{role}/{scenario}": index for scenario in ("ni_rua_benchmark", "ni_rua_interactive", "prompt_capture_fixture")
       for role, index in (("actor", 15), ("program", 13))},
    **{f"{role}/{scenario}": index for scenario in ("rgbd_feedback_v2", "rgbd_numeric_v1", "rgbd_skills_v3")
       for role, index in (("actor", 16), ("program", 14))},
}
ALLOWANCES.append(dict(
    id="design-13-V-F-max-upward-dz-help", package="WP2", embodiment="x5", family="catalogue",
    replacements={item: [(f"/{index}{site}", *_WP2_V_F, 1) for site in _WP2_V_F_SITES]
                  for item, index in _WP2_ROBOT_ACTION_INDEX.items()},
    occurrences=69,
))
# --- end WP2 ---

# --- WP12 allowances ---
# B8 / V-D (design 13 sections 8 and 10, approved): perception/depth/depth_geometry.MAX_OPENING_M, the largest and default
# depth_query grasp_candidates max_opening_m, is the one ARX-X5 full opening ARX_X5_PROFILE.gripper.full_open_aperture_m
# = 0.088 m instead of the pad gap 0.088807166 m. X5-visible: the depth_query definition of every catalogue that has it
# (the 11 definitions/* items, index 17; actor/rgbd_* index 15; program/rgbd_* index 13): schema maximum and default
# (2 edits) and the generated "Default: ..." / "Maximum (inclusive): ..." field-guide numbers (2 replacements), -24 bytes
# per definition, and the max_opening_m help of runtime/tools/definitions.py in the field guide and the schema
# description (2 replacements): 17 items x 6 = 102.
_WP12_DEPTH_ITEMS = {**{"definitions/" + name: 17 for name in (
                         "default", "pose_skill", "perception_shipped", "perception_shipped_pose_skill", "vggt_default_mode",
                         "obs3d_action_overlay", "object_table_observe_only", "scene3d_observe_only", "vggt_only",
                         "object_table_only", "scene3d_only")},
                     **{f"actor/{scenario}": 15 for scenario in ("rgbd_feedback_v2", "rgbd_numeric_v1", "rgbd_skills_v3")},
                     **{f"program/{scenario}": 13 for scenario in ("rgbd_feedback_v2", "rgbd_numeric_v1", "rgbd_skills_v3")}}
_WP12_DEPTH_FIELD = "/inputSchema/properties/request/oneOf/2/properties/max_opening_m/"
# definitions.py max_opening_m help (field guide and schema description), once per definition each: 17 x 2 = 34.
_WP12_DEPTH_HELP = ("bound 0.005..0.088807166 metres, default 0.088807166;", "bound 0.005..0.088 metres, default 0.088;")
ALLOWANCES.append(dict(
    id="design-13-V-D-x5-full-opening-in-depth-query", package="WP12", embodiment="x5", family="catalogue",
    edits={item: [(f"/{index}{_WP12_DEPTH_FIELD}{key}", 0.088807166, 0.088) for key in ("maximum", "default")]
           for item, index in _WP12_DEPTH_ITEMS.items()},
    replacements={item: [(f"/{index}/description", "Default: 0.088807166;", "Default: 0.088;", 1),
                         (f"/{index}/description", "Maximum (inclusive): 0.088807166.", "Maximum (inclusive): 0.088.", 1),
                         *((pointer, *_WP12_DEPTH_HELP, 1)
                           for pointer in (f"/{index}/description", f"/{index}{_WP12_DEPTH_FIELD}description"))]
                  for item, index in _WP12_DEPTH_ITEMS.items()},
    occurrences=102,
))
# F7 (design 13 section 3.6; archive-only, never model-visible): features.vggt_support_z_m defaults to "profile" (the
# run embodiment's support_surface_z_m: ARX-X5 0.765 m, SO-101 the observation's calibrated table height) instead of
# the X5 number 0.765, in FeaturesConfig and in the shipped configs/ni-rua.json / ni-rua.example.json (the base layer
# of every Config.load). X5 resolves to the same 0.765 m (effective_support_z_m); archives and configs that carry the
# number 0.765 keep it. X5 effect: family "config" only, one value in each of the 14 items below: 14 edits.
_WP12_SUPPORT_ITEMS = ("Config()", "load/ni_rua", "load/rgbd_feedback_v2", "load/rgbd_numeric_v1", "load/rgbd_skills_v3",
                       "load/benchmark_shared_memory", "load/ni_rua_example",
                       *("effective_config/" + scenario for scenario in (
                           "ni_rua_benchmark", "ni_rua_interactive", "rgbd_feedback_v2", "rgbd_numeric_v1",
                           "rgbd_skills_v3", "legacy_rgb_proprio_v1", "prompt_capture_fixture")))
ALLOWANCES.append(dict(
    id="design-13-F7-support-height-from-profile", package="WP12", embodiment="x5", family="config",
    edits={item: [("/features/vggt_support_z_m", 0.765, "profile")] for item in _WP12_SUPPORT_ITEMS},
    occurrences=14,
))
# --- end WP12 ---

# --- Stage 0B (ni_rua/main d7b748a merge) allowances ---
def _load_stage0b_allowances(filename="x5_stage0b_allowances.json"):
    path = FIXTURE_ROOT / filename
    if not path.exists():
        return []
    doc = json.loads(path.read_text(encoding="utf-8"))
    pool = doc.get("pool", [])

    def resolve(val):
        if isinstance(val, str) and val.startswith("$v:"):
            val = deepcopy(pool[int(val[3:])])
        return ABSENT if val == ABSENT else val

    out = []
    for raw in doc["records"]:
        rec = {k: v for k, v in raw.items() if k not in ("edits", "replacements")}
        if "edits" in raw:
            rec["edits"] = {
                item: [(ptr, resolve(old), resolve(new)) for ptr, old, new in edits]
                for item, edits in raw["edits"].items()
            }
        if "replacements" in raw:
            rec["replacements"] = {
                item: [(ptr, resolve(old), resolve(new), int(cnt)) for ptr, old, new, cnt in repls]
                for item, repls in raw["replacements"].items()
            }
        out.append(rec)
    return out


ALLOWANCES.extend(_load_stage0b_allowances())
# --- end Stage 0B ---

# --- Phase 2 Wave 2 (Config Schema v2 + Tolerance.max_upward_dz_m + bloat retirement) allowances ---
_W2_CONFIG_THRESHOLDS = (
    ("tight_position_tolerance_m", 0.003),
    ("default_position_tolerance_m", 0.01),
    ("relaxed_position_tolerance_m", 0.015),
    ("compliant_position_tolerance_m", 0.015),
    ("coarse_position_tolerance_m", 0.025),
    ("max_position_tolerance_m", 0.1),
    ("tight_orientation_tolerance_rad", 0.05),
    ("orientation_tolerance_rad", 0.15),
    ("relaxed_orientation_tolerance_rad", 0.2),
    ("coarse_orientation_tolerance_rad", 0.35),
    ("max_orientation_tolerance_rad", 3.14159265),
    ("contact_divergence_threshold_m", 0.004),
    ("stall_guard_strict_divergence_m", 0.01),
    ("stall_guard_contact_stop_divergence_m", 0.02),
    ("stall_guard_compliant_divergence_m", 0.02),
    ("stall_guard_min_divergence_m", 0.005),
    ("stall_guard_max_divergence_m", 0.15),
    ("stall_guard_step_velocity_threshold_m", 0.0015),
    ("stall_guard_consecutive_steps", 2),
    ("max_gripper_aperture_m", 0.088),
    ("grip_empty_threshold_mm", 0.5),
    ("grip_thin_threshold_mm", 1.5),
    ("grip_holding_threshold_mm", 3.0),
    ("grip_solid_threshold_mm", 3.5),
    ("grip_non_empty_min_aperture_m", 0.0015),
    ("grip_thin_min_aperture_m", 0.0005),
    ("grip_rigid_min_aperture_m", 0.002),
    ("grip_wide_min_aperture_m", 0.003),
    ("joint_travel_standard_max_rad", math.pi),
    ("joint_travel_standard_backtracking_rad", math.pi / 2),
    ("joint_travel_extended_max_rad", 1.5 * math.pi),
    ("joint_travel_extended_backtracking_rad", math.pi),
    ("segment_start_joint_drift_rad", 0.1),
    ("resume_max_joint_drift_rad", 0.1),
    ("resume_max_ee_drift_m", 0.015),
    ("resume_max_orientation_drift_rad", 0.1),
    ("resume_max_jaw_drift_normalized", 0.1),
    ("batch_plan_cache_size", 16),
    ("hotspot_xy_radius_m", 0.025),
    ("hotspot_z_radius_m", 0.045),
    ("ledger_max_recent_outcomes", 3),
    ("ledger_max_hotspots", 6),
    ("target_exemption_radius_m", 0.055),
    ("clearance_height_m", 0.13),
)
_W2_RETIRED_BLOAT_FLAGS = (
    ("self_briefing", False),
    ("operation_candidates", False),
    ("lean_projection", False),
    ("lean_projection_k", 1),
    ("lean_projection_block", 8),
    ("lean_compact", False),
    ("lean_request_dump", 0),
)
ALLOWANCES.append(dict(
    id="phase2-wave2-config-schema-v2-and-bloat-retirement", package="P2-W2", embodiment="x5", family="config",
    edits={
        item: [
            ("/version", 1, 2),
            *((f"/features/{key}", old, ABSENT) for key, old in (*_W2_CONFIG_THRESHOLDS, *_W2_RETIRED_BLOAT_FLAGS)),
            ("/embodiment", ABSENT, {"thresholds": {}}),
        ]
        for item in _WP12_SUPPORT_ITEMS
    },
    occurrences=14 * (2 + len(_W2_CONFIG_THRESHOLDS) + len(_W2_RETIRED_BLOAT_FLAGS)),
))
# Static default ModelConfig.name aligns with configs/ni-rua.json (AGENTS.md default-config consistency gate).
ALLOWANCES.append(dict(
    id="phase2-config-default-model-sol", package="P2", embodiment="x5", family="config",
    edits={item: [("/model/name", "gpt-6-astra", "gpt-6.1-sol")] for item in ("Config()", "effective_config/prompt_capture_fixture")},
    occurrences=2,
))
# Reply delivery defaults (FeaturesConfig and the shipped configs): budgeted_action_reply, reply_same_as,
# reply_float32, reply_static_once, reply_compact_encoding on, action_reply_profile "summary_first"
# (core/presentation/reply_budget.py, reply_compaction.py, reply_encoding.py), and actor_payload_profile "lean_v2".
# Family "config": six added keys and one changed value in each of the 14 recorded configurations.
_REPLY_DELIVERY_DEFAULTS = dict(budgeted_action_reply=True, reply_same_as=True, reply_float32=True,
                                reply_static_once=True, action_reply_profile="summary_first",
                                reply_compact_encoding=True)
ALLOWANCES.append(dict(
    id="reply-delivery-defaults-config", package="reply-delivery", embodiment="x5", family="config",
    edits={item: [("/features/actor_payload_profile", "lean_v1", "lean_v2"),
                  *((f"/features/{name}", ABSENT, default) for name, default in _REPLY_DELIVERY_DEFAULTS.items())]
           for item in _WP12_SUPPORT_ITEMS},
    occurrences=14 * (1 + len(_REPLY_DELIVERY_DEFAULTS)),
))
# The same defaults in what the actor is handed:
#   catalogue: the robot_action description of every actor catalogue states the reply layout, the fixed reply notes
#     and the compact encoding once, before its input field guide. The fixture robot of prompt_capture_fixture
#     publishes no collision_scope in its capabilities, so its note has no collision sentence. No other tool, no
#     schema and no program catalogue changes.
#   startup: lean_v2 prints the floats of the "Robot runtime capabilities" JSON in the instructions with six
#     decimals; the two tool-geometry offsets that carried float64 noise are the only numbers that change.
_REPLY_GUIDE = "\nInput field guide (use only fields applicable to the selected mode/kind):"


def _reply_delivery_notes(collision_scope):
    from ni_rua.core.presentation import reply_budget, reply_compaction, reply_encoding
    fixed = reply_compaction.STATIC_ONCE_CATALOGUE_NOTE
    if not collision_scope:
        sentence = "collision_scope: " + reply_compaction._COLLISION_NOTE + ". "
        assert fixed.count(sentence) == 1
        fixed = fixed.replace(sentence, "")
    return (reply_budget.SUMMARY_FIRST_CATALOGUE_NOTE + fixed + reply_encoding.COMPACT_ENCODING_CATALOGUE_NOTE
            + reply_compaction.SAME_AS_CATALOGUE_NOTE)


_REPLY_NOTE_ITEMS = {"actor/ni_rua_benchmark": (15, True), "actor/ni_rua_interactive": (15, True),
                     "actor/rgbd_feedback_v2": (16, True), "actor/rgbd_numeric_v1": (16, True),
                     "actor/rgbd_skills_v3": (16, True), "actor/prompt_capture_fixture": (15, False)}
ALLOWANCES.append(dict(
    id="reply-delivery-defaults-catalogue-notes", package="reply-delivery", embodiment="x5", family="catalogue",
    replacements={item: [(f"/{index}/description", _REPLY_GUIDE, _reply_delivery_notes(collision) + _REPLY_GUIDE, 1)]
                  for item, (index, collision) in _REPLY_NOTE_ITEMS.items()},
    occurrences=6,
))
ALLOWANCES.append(dict(
    id="reply-delivery-defaults-lean-v2-startup", package="reply-delivery", embodiment="x5", family="startup",
    replacements={scenario + "/instructions": [("", '"offset_m":0.12260000000000001', '"offset_m":0.1226', 1),
                                               ("", '"offset_m":0.15760000000000002', '"offset_m":0.1576', 1)]
                  for scenario in ("ni_rua_benchmark", "ni_rua_interactive", "rgbd_feedback_v2", "rgbd_numeric_v1",
                                   "rgbd_skills_v3", "legacy_rgb_proprio_v1")},
    occurrences=12,
))
# features.robot_urdf_delivery (inserted after tool_catalog_profile, the order of FeaturesConfig) is a new key in every
# serialised configuration; the recorded form states it as "inline" (RECORDED_CATALOGUE_FEATURES).
ALLOWANCES.append(dict(
    id="catalogue-profiles-robot-urdf-delivery-key", package="catalogue-profiles", embodiment="x5", family="config",
    edits={item: [("/features/robot_urdf_delivery", ABSENT, "inline", "tool_catalog_profile")] for item in _WP12_SUPPORT_ITEMS},
    occurrences=14,
))
# features.tool_catalog_profile "full_v1" -> "layered_v3" and features.robot_urdf_delivery "inline" -> "file" are the
# shipped defaults. Family "config", the seven items that show defaults (Config() and the six shipped configurations as
# loaded): two values each = 14 edits. The seven effective_config/<scenario> items state both settings explicitly
# (RECORDED_CATALOGUE_FEATURES) and stay equal to the recorded values; catalogue, instructions and startup are captured
# with that explicit recorded form too, so they need no allowance.
ALLOWANCES.append(dict(
    id="catalogue-defaults-layered-v3-file-config", package="catalogue-defaults", embodiment="x5", family="config",
    edits={item: [("/features/tool_catalog_profile", "full_v1", "layered_v3"),
                  ("/features/robot_urdf_delivery", "inline", "file")] for item in _WP12_SUPPORT_ITEMS[:7]},
    occurrences=14,
))
ALLOWANCES.append(dict(
    id="phase2-wave2-retire-lean-projection-instructions", package="P2-W2", embodiment="x5", family="instructions",
    edits={
        "lean_projection": [(
            "",
            '\nContext format: earlier tool results in this conversation appear as compact summaries, and their images '
            'as placeholders.\nThe most recent tool result appears in full, with its images, in a final '
            '"[NI-RUA latest tool feedback]" message.\nUse history_view_observation to reread an earlier saved view, '
            "or detail='full' for its complete metadata.",
            ABSENT,
        )],
        "lean_projection_block": [(
            "",
            "\nContext format: recent tool results in this conversation appear in full with their images; older "
            "results are periodically condensed to compact summaries with image placeholders, and some older results "
            "stay in full.\nUse history_view_observation to reread an earlier saved view, or detail='full' for its "
            "complete metadata.",
            ABSENT,
        )],
    },
    occurrences=2,
))
_W2_TOL_SUMMARY = (
    "Use tolerance_preset (tight, normal, relaxed, coarse, unconstrained_orientation)",
    "Use tolerance_preset (tight, normal, relaxed, compliant, coarse, unconstrained_orientation)",
)
_W2_TOL_GROUP = (
    "relaxed/compliant/seating=(0.015 m, 0.20 rad) for compliant placement",
    "relaxed=(0.015 m, 0.20 rad) for compliant placement, compliant/seating=(0.015 m, 0.20 rad, max_upward_dz_m=0.040 m) for compliant placement",
)
_W2_TOL_SITES = (
    "/description",
    "/inputSchema/properties/stages/items/properties/tolerance_preset/description",
    "/inputSchema/properties/stages/items/allOf/0/then/properties/tolerance_preset/description",
)
ALLOWANCES.append(dict(
    id="phase2-wave2-tolerance-preset-upward-dz-on-profile", package="P2-W2", embodiment="x5", family="catalogue",
    replacements={
        item: [
            (f"/{index}/description", *_W2_TOL_SUMMARY, 1),
            *((f"/{index}{site}", *_W2_TOL_GROUP, 1) for site in _W2_TOL_SITES),
        ]
        for item, index in _WP2_ROBOT_ACTION_INDEX.items()
    },
    occurrences=23 * 4,
))
# --- end Phase 2 Wave 2 ---

# --- Phase 2 Wave 3 (carry_slip_guard public schema + program_run.on_incomplete_action) allowances ---
_W3_PROGRAM_RUN_INDEX = {
    **{"definitions/" + name: 12 for name in (
        "default", "pose_skill", "perception_shipped", "perception_shipped_pose_skill", "vggt_default_mode",
        "obs3d_action_overlay", "object_table_observe_only", "scene3d_observe_only", "vggt_only",
        "object_table_only", "scene3d_only",
    )},
    "actor/ni_rua_benchmark": 12,
    "actor/ni_rua_interactive": 12,
    "actor/legacy_rgb_proprio_v1": 10,
    "actor/prompt_capture_fixture": 12,
    "actor/rgbd_feedback_v2": 12,
    "actor/rgbd_numeric_v1": 12,
    "actor/rgbd_skills_v3": 12,
}
_W3_ON_INCOMPLETE_DESC = (
    "Optional start-only broker guard policy for subsequent physical actions after an incomplete physical action "
    "(status != 'completed' and physics_advanced != false), enum 'block' (default) or 'allow'. 'block' refuses the "
    "next physical action with program_blind_continuation_blocked until live robot_observe or explicit "
    "allow_after_failure=True / client.acknowledge_failure(); 'allow' disables broker-side latching when the program "
    "inspects receipts directly. Omit on resume. Example: {'source': '...', 'allow_direct_actions': True, "
    "'on_incomplete_action': 'block'}."
)
_W3_ON_INCOMPLETE_SCHEMA = {
    "type": "string",
    "enum": ["block", "allow"],
    "default": "block",
    "description": _W3_ON_INCOMPLETE_DESC,
}
_W3_STAGE_CS_PROPS = {
    "carry_slip_guard": {
        "type": "boolean",
        "description": (
            'Move (with gripper_goal_normalized) or gripper stage: optional boolean (default false; inherits batch '
            'carry_slip_guard when set). When true on a closing jaw stage (goal < 0.5) that resolves a non-null '
            'min_aperture_m (in metres) and passes its initial close check, monitors measured fingertip aperture '
            '(metres) on every native step of subsequent same-arm move stages and pauses immediately with '
            'checkpoint_reason_code="carry_slip_aperture_below_min_m" and slip_detected_at if measured_aperture_m < '
            'min_aperture_m for slip_consecutive_steps consecutive native steps. Example: {"id":"close",'
            '"kind":"gripper","arm":"left","goal_normalized":0,"grip_guard_preset":"rigid_object",'
            '"carry_slip_guard":true,"carry_slip_scope":"until_release"}.'
        ),
    },
    "carry_slip_scope": {
        "type": "string",
        "enum": ["next_move", "until_release"],
        "description": (
            'Move (with gripper_goal_normalized) or gripper stage: optional coverage scope for carry_slip_guard=true, '
            'enum "next_move" (default, monitors only the immediately following same-arm move stage) or '
            '"until_release" (monitors all subsequent same-arm move stages across lift/transport/align until an '
            'opening jaw command with goal >= 0.5). Requires carry_slip_guard=true.'
        ),
    },
    "slip_consecutive_steps": {
        "type": "integer",
        "minimum": 1,
        "maximum": 50,
        "description": (
            "Move (with gripper_goal_normalized) or gripper stage: optional integer in [1, 50] (unit: native steps, "
            "default 1) giving the consecutive below-min_aperture_m samples required before carry_slip_guard pauses "
            "execution. Set 2..3 to debounce transient jaw sensor jitter. Requires carry_slip_guard=true."
        ),
    },
}
_W3_BATCH_CS_PROPS = {
    "carry_slip_guard": {
        "type": "boolean",
        "description": (
            'Preview/direct only: optional batch-default boolean (default false) enabling per-step fingertip aperture '
            'slip monitoring (against min_aperture_m in metres) after passing closing jaw stages (goal < 0.5); '
            'overridden by stage-level carry_slip_guard. Example: {"grip_guard_preset":"rigid_object",'
            '"carry_slip_guard":true,"carry_slip_scope":"until_release"}.'
        ),
    },
    "carry_slip_scope": {
        "type": "string",
        "enum": ["next_move", "until_release"],
        "description": (
            'Preview/direct only: optional batch-default scope for carry_slip_guard=true, enum "next_move" (default) '
            'or "until_release" (monitors subsequent same-arm moves until jaw goal >= 0.5); requires '
            'carry_slip_guard=true and overridden by stage-level carry_slip_scope.'
        ),
    },
    "slip_consecutive_steps": {
        "type": "integer",
        "minimum": 1,
        "maximum": 50,
        "description": (
            "Preview/direct only: optional batch-default integer in [1, 50] (unit: native steps, default 1) of "
            "consecutive below-min_aperture_m samples required before pausing; requires carry_slip_guard=true and "
            "overridden by stage-level slip_consecutive_steps."
        ),
    },
}
_W3_STAGE_CS_FLAT_PROPS = {
    "carry_slip_guard": _W3_STAGE_CS_PROPS["carry_slip_guard"],
    "carry_slip_scope": {
        "type": "string",
        "description": _W3_STAGE_CS_PROPS["carry_slip_scope"]["description"],
    },
    "slip_consecutive_steps": _W3_STAGE_CS_PROPS["slip_consecutive_steps"],
}
_W3_BATCH_CS_FLAT_PROPS = {
    "carry_slip_guard": _W3_BATCH_CS_PROPS["carry_slip_guard"],
    "carry_slip_scope": {
        "type": "string",
        "minLength": 1,
        "description": _W3_BATCH_CS_PROPS["carry_slip_scope"]["description"],
    },
    "slip_consecutive_steps": _W3_BATCH_CS_PROPS["slip_consecutive_steps"],
}
_W3_INITIAL_CONTACT_PROPS = {
    "entity_ids": {
        "type": "array",
        "items": {"type": "string"},
        "description": "IDs of existing obstacles in pre-existing contact.",
    },
    "evidence": {
        "type": "string",
        "minLength": 1,
        "description": "Required nonblank observed basis for the contact declaration.",
    },
}
_W3_PLACEMENT_CONTACT_PROPS = {
    "entity_id": {"type": "string", "description": "ID of the existing obstacle supporting placement."},
    "evidence": {
        "type": "string",
        "minLength": 1,
        "description": "Required nonblank observed basis for the contact declaration.",
    },
    "max_penetration_m": {
        "type": "number",
        "minimum": 0,
        "maximum": 0.003,
        "description": "Maximum declared placement penetration in metres, default 0.002, range 0..0.003.",
    },
}
_W3_INITIAL_CONTACT_DESC = "Preview/direct only: obstacle entity IDs and evidence explaining pre-existing contact."
_W3_PLACEMENT_CONTACT_DESC = (
    "Preview/direct only: intended support entity ID and evidence, with optional permitted penetration estimate."
)
_W3_HELD_OBJECT_ALLOF = {
    "if": {"required": ["held_object"]},
    "then": {
        "properties": {
            "held_object": {
                "type": "object",
                "description": (
                    "Preview/direct only: single held-object AABB; arm required for a dual-arm batch, otherwise "
                    "inferred. Choose this or held_objects."
                ),
                "properties": {
                    "arm": {
                        "type": "string",
                        "enum": ["left", "right"],
                        "description": "Holding arm; required when a single held_object is used with both arms.",
                    }
                },
            }
        }
    },
}
_W3_SETTLE_ALLOF = {
    "if": {"required": ["settle"]},
    "then": {
        "properties": {
            "settle": {
                "type": "object",
                "description": (
                    "Preview/direct only: optional bounded RGB stability check after the complete batch, never at a "
                    "checkpoint; adds separate native wait steps. Requires robot_wait and a visible ROI."
                ),
                "properties": {
                    "camera_id": {"type": "string", "description": "RGB camera ID, default cam_head."},
                    "roi": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "maxItems": 4,
                        "description": "Target pixel [x,y,width,height]; nonnegative origin, width/height at least 4, inside the image.",
                    },
                    "max_native_steps": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "description": "Maximum additional wait steps, default 25, range 1..50.",
                    },
                    "sample_steps": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "Wait steps per RGB sample, default 5, range 1..10.",
                    },
                    "stable_samples": {
                        "type": "integer",
                        "minimum": 2,
                        "maximum": 5,
                        "description": "Required consecutive quiet comparisons, default 3, range 2..5.",
                    },
                    "max_motion_px": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 5,
                        "description": "Maximum tracked-feature motion in pixels for a quiet sample, default 0.5, range 0..5; not proof of physical stability.",
                    },
                },
            }
        }
    },
}
ALLOWANCES.append(dict(
    id="phase2-wave3-program-run-on-incomplete-action", package="P2-W3", embodiment="x5", family="catalogue",
    edits={
        item: [(f"/{index}/inputSchema/properties/on_incomplete_action", ABSENT, _W3_ON_INCOMPLETE_SCHEMA)]
        for item, index in _W3_PROGRAM_RUN_INDEX.items()
    },
    replacements={
        item: [
            (
                f"/{index}/description",
                'detail is one of ["summary","full"] when supplied; only declared fields.',
                'detail is one of ["summary","full"] when supplied; on_incomplete_action is one of ["block","allow"] when supplied; only declared fields.',
                1,
            ),
            (
                f"/{index}/description",
                'artifact references retain exact saved bytes. Type: string; Allowed: ["summary","full"].',
                'artifact references retain exact saved bytes. Type: string; Allowed: ["summary","full"].\n'
                f'on_incomplete_action: {_W3_ON_INCOMPLETE_DESC} Type: string; Allowed: ["block","allow"]; Default: "block".',
                1,
            ),
        ]
        for item, index in _W3_PROGRAM_RUN_INDEX.items()
    },
    occurrences=len(_W3_PROGRAM_RUN_INDEX) * 3,
))
ALLOWANCES.append(dict(
    id="phase2-wave3-robot-action-carry-slip-guard-schema", package="P2-W3", embodiment="x5", family="catalogue",
    edits={
        item: [
            *((f"/{index}/inputSchema/properties/stages/items/properties/{k}", ABSENT, v) for k, v in _W3_STAGE_CS_FLAT_PROPS.items()),
            *((f"/{index}/inputSchema/properties/stages/items/allOf/0/then/properties/{k}", ABSENT, v) for k, v in _W3_STAGE_CS_PROPS.items()),
            *((f"/{index}/inputSchema/properties/stages/items/allOf/1/then/properties/{k}", ABSENT, v) for k, v in _W3_STAGE_CS_PROPS.items()),
            *((f"/{index}/inputSchema/properties/{k}", ABSENT, v) for k, v in _W3_BATCH_CS_FLAT_PROPS.items()),
            (f"/{index}/inputSchema/properties/initial_contact/properties", _W3_INITIAL_CONTACT_PROPS, ABSENT),
            (f"/{index}/inputSchema/properties/placement_contact/properties", _W3_PLACEMENT_CONTACT_PROPS, ABSENT),
            (
                f"/{index}/inputSchema/allOf/3",
                _W3_HELD_OBJECT_ALLOF,
                {
                    "if": {"required": ["carry_slip_scope"]},
                    "then": {"properties": {"carry_slip_scope": _W3_BATCH_CS_PROPS["carry_slip_scope"]}},
                },
            ),
            (f"/{index}/inputSchema/allOf/4", _W3_SETTLE_ALLOF, _W3_HELD_OBJECT_ALLOF),
            (
                f"/{index}/inputSchema/allOf/5",
                ABSENT,
                {
                    "if": {"required": ["initial_contact"]},
                    "then": {
                        "properties": {
                            "initial_contact": {
                                "type": "object",
                                "description": _W3_INITIAL_CONTACT_DESC,
                                "properties": _W3_INITIAL_CONTACT_PROPS,
                            }
                        }
                    },
                },
            ),
            (
                f"/{index}/inputSchema/allOf/6",
                ABSENT,
                {
                    "if": {"required": ["placement_contact"]},
                    "then": {
                        "properties": {
                            "placement_contact": {
                                "type": "object",
                                "description": _W3_PLACEMENT_CONTACT_DESC,
                                "properties": _W3_PLACEMENT_CONTACT_PROPS,
                            }
                        }
                    },
                },
            ),
            (f"/{index}/inputSchema/allOf/7", ABSENT, _W3_SETTLE_ALLOF),
        ]
        for item, index in _WP2_ROBOT_ACTION_INDEX.items()
    },
    replacements={
        item: [
            (
                f"/{index}/description",
                'if (requires native_steps), then () | if (requires held_object), then () | if (requires settle), then ()).',
                'if (requires native_steps), then () | if (requires carry_slip_scope), then (carry_slip_scope is one of ["next_move","until_release"] when supplied) | if (requires held_object), then () | if (requires initial_contact), then () | if (requires placement_contact), then () | if (requires settle), then ()).',
                1,
            ),
            (
                f"/{index}/description",
                'clearance_preset is one of ["none","low","normal","high"] when supplied; requires id, kind, arm;',
                'clearance_preset is one of ["none","low","normal","high"] when supplied; carry_slip_scope is one of ["next_move","until_release"] when supplied; requires id, kind, arm;',
                1,
            ),
            (
                f"/{index}/description",
                'kind equals "gripper"; grip_guard_preset is one of ["none","non_empty","thin_object","rigid_object","wide_object"] when supplied; gripper_speed_preset is one of ["slower","slow","normal"] when supplied; requires id, kind, arm, goal_normalized;',
                'kind equals "gripper"; grip_guard_preset is one of ["none","non_empty","thin_object","rigid_object","wide_object"] when supplied; gripper_speed_preset is one of ["slower","slow","normal"] when supplied; carry_slip_scope is one of ["next_move","until_release"] when supplied; requires id, kind, arm, goal_normalized;',
                1,
            ),
            (
                f"/{index}/description",
                'stages[].reason: Observe stage only, required nonblank text: what evidence to inspect at this checkpoint. Type: string; Pattern: "\\\\S". [stages[].kind="observe"] Type: string; Minimum characters: 1; Pattern: "\\\\S".',
                'stages[].reason: Observe stage only, required nonblank text: what evidence to inspect at this checkpoint. Type: string; Pattern: "\\\\S". [stages[].kind="observe"] Type: string; Minimum characters: 1; Pattern: "\\\\S".\n'
                f'stages[].carry_slip_guard: {_W3_STAGE_CS_PROPS["carry_slip_guard"]["description"]} Type: boolean.\n'
                f'stages[].carry_slip_scope: {_W3_STAGE_CS_PROPS["carry_slip_scope"]["description"]} Type: string. [stages[].kind="gripper" or stages[].kind="move"] Type: string; Allowed: ["next_move","until_release"].\n'
                f'stages[].slip_consecutive_steps: {_W3_STAGE_CS_PROPS["slip_consecutive_steps"]["description"]} Type: integer; Minimum (inclusive): 1; Maximum (inclusive): 50.',
                1,
            ),
            (
                f"/{index}/description",
                'initial_contact: Preview/direct only: obstacle entity IDs and evidence explaining pre-existing contact. Type: object.\n'
                'initial_contact.entity_ids: IDs of existing obstacles in pre-existing contact. Type: array.\n'
                'initial_contact.entity_ids[]: Type: string.\n'
                'initial_contact.evidence: Required nonblank observed basis for the contact declaration. Type: string; Minimum characters: 1.\n'
                'placement_contact: Preview/direct only: intended support entity ID and evidence, with optional permitted penetration estimate. Type: object.\n'
                'placement_contact.entity_id: ID of the existing obstacle supporting placement. Type: string.\n'
                'placement_contact.evidence: Required nonblank observed basis for the contact declaration. Type: string; Minimum characters: 1.\n'
                'placement_contact.max_penetration_m: Maximum declared placement penetration in metres, default 0.002, range 0..0.003. Type: number; Minimum (inclusive): 0; Maximum (inclusive): 0.003.',
                'initial_contact: Preview/direct only: obstacle entity IDs and evidence explaining pre-existing contact. Type: object.\n'
                'placement_contact: Preview/direct only: intended support entity ID and evidence, with optional permitted penetration estimate. Type: object.',
                1,
            ),
            (
                f"/{index}/description",
                'max_samples: Inspect pagination: 1..25 samples per page, default 25. Mutually exclusive with native_steps. Type: integer; Minimum (inclusive): 1; Maximum (inclusive): 25.',
                'max_samples: Inspect pagination: 1..25 samples per page, default 25. Mutually exclusive with native_steps. Type: integer; Minimum (inclusive): 1; Maximum (inclusive): 25.\n'
                f'carry_slip_guard: {_W3_BATCH_CS_PROPS["carry_slip_guard"]["description"]} Type: boolean.\n'
                f'carry_slip_scope: {_W3_BATCH_CS_PROPS["carry_slip_scope"]["description"]} Type: string; Minimum characters: 1. Type: string; Allowed: ["next_move","until_release"].\n'
                f'slip_consecutive_steps: {_W3_BATCH_CS_PROPS["slip_consecutive_steps"]["description"]} Type: integer; Minimum (inclusive): 1; Maximum (inclusive): 50.\n'
                'initial_contact.entity_ids: IDs of existing obstacles in pre-existing contact. Type: array.\n'
                'initial_contact.entity_ids[]: Type: string.\n'
                'initial_contact.evidence: Required nonblank observed basis for the contact declaration. Type: string; Minimum characters: 1.\n'
                'placement_contact.entity_id: ID of the existing obstacle supporting placement. Type: string.\n'
                'placement_contact.evidence: Required nonblank observed basis for the contact declaration. Type: string; Minimum characters: 1.\n'
                'placement_contact.max_penetration_m: Maximum declared placement penetration in metres, default 0.002, range 0..0.003. Type: number; Minimum (inclusive): 0; Maximum (inclusive): 0.003.',
                1,
            ),
        ]
        for item, index in _WP2_ROBOT_ACTION_INDEX.items()
    },
    occurrences=len(_WP2_ROBOT_ACTION_INDEX) * (19 + 6),
))
ALLOWANCES.append(dict(
    id="phase2-wave3-compile-allowed-fields-carry-slip", package="P2-W3", embodiment="x5", family="compile",
    replacements={
        **{
            item: [
                ("/schema/suggestion", "arm, checkpoint_after_gripper,", "arm, carry_slip_guard, carry_slip_scope, checkpoint_after_gripper,", 1),
                ("/schema/suggestion", "quaternion_xyzw, speed_preset,", "quaternion_xyzw, slip_consecutive_steps, speed_preset,", 1),
            ]
            for item in ("323 structure/missing_id", "326 structure/missing_arm", "328 structure/unknown_stage_field")
        },
        "324 structure/missing_kind": [
            ("/schema/suggestion", "arm, checkpoint_after_gripper,", "arm, carry_slip_guard, carry_slip_scope, checkpoint_after_gripper,", 1),
            ("/schema/suggestion", "quaternion_xyzw, reason, speed_preset,", "quaternion_xyzw, reason, slip_consecutive_steps, speed_preset,", 1),
        ],
        "333 structure/gripper_without_goal": [
            ("/schema/suggestion", "arm, checkpoint_after_gripper,", "arm, carry_slip_guard, carry_slip_scope, checkpoint_after_gripper,", 1),
            ("/schema/suggestion", "max_aperture_m, min_aperture_m.", "max_aperture_m, min_aperture_m, slip_consecutive_steps.", 1),
        ],
    },
    occurrences=10,
))
# --- end Phase 2 Wave 3 ---

# --- Reply-detail fix: robot_action inspect native_steps limit 3 -> 25 ---
# The summary_first reply leaves out measured/planned rows and names the robot_action inspect call that returns exactly
# them (their native_steps, up to 25). The public schema allowed 3 while the inspect handler already accepted 25; both now
# take core.contracts.INSPECT_STEP_LIMIT = 25. X5-visible in robot_action of the same 23 items as the WP2 allowance: the
# native_steps maxItems (2 sites: top level and allOf/2), its help text (2 sites) and the generated field-guide line (1).
_RD_OLD_HELP = "Inspect only: 1..3 distinct existing native-step IDs; mutually exclusive with pagination. Default selects first/middle/last."
_RD_NEW_HELP = ("Inspect only: 1..25 distinct existing native-step IDs; mutually exclusive with pagination. "
                "Default (no selector) selects only the first, middle and last.")
_RD_NATIVE_STEPS_SITES = ("/inputSchema/properties/native_steps", "/inputSchema/allOf/2/then/properties/native_steps")
ALLOWANCES.append(dict(
    id="reply-detail-inspect-native-steps-limit", package="reply-detail", embodiment="x5", family="catalogue",
    edits={
        item: [edit for site in _RD_NATIVE_STEPS_SITES for edit in (
            (f"/{index}{site}/maxItems", 3, 25), (f"/{index}{site}/description", _RD_OLD_HELP, _RD_NEW_HELP))]
        for item, index in _WP2_ROBOT_ACTION_INDEX.items()
    },
    replacements={
        item: [(f"/{index}/description",
                f"native_steps: {_RD_OLD_HELP} Type: array; Maximum items: 3. Type: array; Minimum items: 1; Maximum items: 3.",
                f"native_steps: {_RD_NEW_HELP} Type: array; Maximum items: 25. Type: array; Minimum items: 1; Maximum items: 25.", 1)]
        for item, index in _WP2_ROBOT_ACTION_INDEX.items()
    },
    occurrences=len(_WP2_ROBOT_ACTION_INDEX) * 5,
))
# --- end reply-detail ---

# --- robot_action_status and program_status loaded up front ---
# Replies name these two as the read of what they left out (the ``detail`` entry), and a host whose tool search does not
# work cannot load a deferLoading tool, so both lost ``deferLoading``. X5-visible: exactly one key, ``deferLoading``
# (true -> absent), of exactly those two tool definitions: robot_action_status in all 25 catalogue items (index 5, and 4
# in the two legacy_rgb_proprio_v1 items), program_status in the 18 items with the actor tool list or the definitions
# (index 13, and 11 in actor/legacy_rgb_proprio_v1; the program role does not list it). Nothing else of either
# definition, and no other tool, changes: 43 edits.
_LOADED_STATUS_SCENARIOS = ("ni_rua_benchmark", "ni_rua_interactive", "rgbd_feedback_v2", "rgbd_numeric_v1",
                            "rgbd_skills_v3", "legacy_rgb_proprio_v1", "prompt_capture_fixture")
_LOADED_STATUS_ITEMS = (
    [("definitions/" + name, 5, 13) for name in (
        "default", "pose_skill", "perception_shipped", "perception_shipped_pose_skill", "vggt_default_mode",
        "obs3d_action_overlay", "object_table_observe_only", "scene3d_observe_only", "vggt_only",
        "object_table_only", "scene3d_only")]
    + [(f"{role}/{scenario}", 4 if scenario == "legacy_rgb_proprio_v1" else 5,
        (11 if scenario == "legacy_rgb_proprio_v1" else 13) if role == "actor" else None)
       for scenario in _LOADED_STATUS_SCENARIOS for role in ("actor", "program")])
_LOADED_STATUS_EDITS = {
    item: [(f"/{index}/deferLoading", True, ABSENT) for index in (action, program) if index is not None]
    for item, action, program in _LOADED_STATUS_ITEMS}
ALLOWANCES.append(dict(
    id="status-tools-loaded-up-front", package="status-tools", embodiment="x5", family="catalogue",
    edits=_LOADED_STATUS_EDITS,
    occurrences=sum(len(edits) for edits in _LOADED_STATUS_EDITS.values()),
))
# --- end status-tools ---

# --- TI1 tool-interface refactor (W2 definitions contracts + W3 scene3d_helpers signature sync) allowances ---
ALLOWANCES.extend(_load_stage0b_allowances("x5_ti1_allowances.json"))
# --- end TI1 ---


# ---- serialisation -----------------------------------------------------------------------------------------------
def normalize(value):
    """``value`` as plain JSON data (tuples -> lists), preserving key order; raises on NaN/inf."""
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def render(value):
    """The bytes an item is compared by: order-preserving compact JSON (UTF-8, no ASCII escaping, no NaN)."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def encode(value):
    """Fixture-file form: every string with a newline becomes ``{"$lines": text.split("\\n")}`` (lossless)."""
    if isinstance(value, str) and "\n" in value:
        return {LINES_KEY: value.split("\n")}
    if isinstance(value, dict):
        if LINES_KEY in value:
            raise ValueError("reserved fixture key in captured data: " + LINES_KEY)
        return {key: encode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [encode(item) for item in value]
    return value


def decode(value):
    if isinstance(value, dict):
        if set(value) == {LINES_KEY}:
            return "\n".join(value[LINES_KEY])
        return {key: decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode(item) for item in value]
    return value


def error_text(error):
    """``"<Type>: <message>"`` of a rejected request (the form the corpus records)."""
    return f"{type(error).__name__}: {error}"


def pointer_get(value, pointer):
    for part in pointer.strip("/").split("/"):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def string_leaves(value, pointer=""):
    """``[(JSON pointer, text)]`` of every string leaf, in document order (pointer form of APPROVED_CORRECTION)."""
    if isinstance(value, dict):
        return [leaf for key, item in value.items() for leaf in string_leaves(item, f"{pointer}/{key}")]
    if isinstance(value, list):
        return [leaf for index, item in enumerate(value) for leaf in string_leaves(item, f"{pointer}/{index}")]
    return [(pointer, value)] if isinstance(value, str) else []


def pointer_set(value, pointer, item):
    parts = pointer.strip("/").split("/")
    for part in parts[:-1]:
        value = value[int(part)] if isinstance(value, list) else value[part]
    last = parts[-1]
    value[int(last) if isinstance(value, list) else last] = item


# ---- redaction ---------------------------------------------------------------------------------------------------
_NEW_ID = re.compile(r"\b([a-z][a-z0-9_]*(?:-[a-z][a-z0-9_]*)*)-([0-9a-f]{32})\b")
_ARTIFACT = re.compile(r"artifact:[0-9a-f]{64}-[0-9a-f]{16}")


class Redactor:
    """Replace run-specific text with stable markers, consistently within one capture.

    Paths: the run directory -> ``<run>`` and the temporary root around it -> ``<tmp>`` (each with its macOS
    ``/private`` twin), the repository -> ``<repo>``, the home directory (as given and symlink-resolved) -> ``~``.
    ``new_id`` identifiers (``<prefix>-<32 hex>``) -> ``<prefix#n>`` and artifact references -> ``artifact:<#n>``,
    numbered by first appearance, so equal references stay equal. ``extra`` maps further literal values (run
    identity, thread id) to markers.
    """

    def __init__(self, run_root=None, tmp_root=None, extra=None):
        self.literals = []
        for path, marker in ((run_root, "<run>"), (tmp_root, "<tmp>")):
            if path is None:
                continue
            path = str(Path(path).resolve())
            forms = {path, path.removeprefix("/private")}
            if path.startswith("/var/"):
                forms.add("/private" + path)
            self.literals += [(form, marker) for form in forms]
        # Config resolves every path field, so a home below a symlink (macOS /tmp, a Linux /home link) appears in
        # its resolved spelling; both spellings are the home.
        home = str(Path.home())
        self.literals += [(str(REPOSITORY), "<repo>")] + [(form, "~") for form in {home, os.path.realpath(home)}]
        self.literals += [(str(key), str(value)) for key, value in (extra or {}).items() if key]
        self.literals.sort(key=lambda item: len(item[0]), reverse=True)
        self.ids, self.artifacts = {}, {}

    def text(self, text):
        for value, marker in self.literals:
            text = text.replace(value, marker)
        text = _ARTIFACT.sub(lambda match: "artifact:<#%d>" % self.artifacts.setdefault(
            match.group(0), len(self.artifacts) + 1), text)

        def identifier(match):
            number = self.ids.setdefault(match.group(0), len(self.ids) + 1)
            return f"<{match.group(1)}#{number}>"
        return _NEW_ID.sub(identifier, text)

    def __call__(self, value):
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, dict):
            return {self.text(key) if isinstance(key, str) else key: self(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self(item) for item in value]
        return value


# ---- fixture robots ----------------------------------------------------------------------------------------------
X5_ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
X5_CAMERAS = (("cam_head", None), ("cam_left_wrist", "left"), ("cam_right_wrist", "right"))
# Native numbers the X5 worker reads from the simulator; fixed fixture values here (not native facts). They enter the
# capabilities through the real RoboDojoAdapter.capabilities builder, so the frozen document is the worker's code.
X5_NATIVE = dict(control_hz=25.0, physics_dt_s=0.004, native_step_ticks=10, native_step_limit=700,
                 arm_step_limit_rad=0.05, joint_limit_rad=3.14)
_PNG_1X1 = None


def _png_bytes():
    global _PNG_1X1
    if _PNG_1X1 is None:
        import struct
        import zlib

        def chunk(kind, data):
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        _PNG_1X1 = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
                    + chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00")) + chunk(b"IEND", b""))
    return _PNG_1X1


def _translation(x=0.0, y=0.0, z=0.0):
    return [[1.0, 0.0, 0.0, x], [0.0, 1.0, 0.0, y], [0.0, 0.0, 1.0, z], [0.0, 0.0, 0.0, 1.0]]


def x5_worker_capabilities(*, observation_profile, motion_execution_mode="resampled", motion_safety_guard=True,
                           trajectory_preview=True):
    """The X5 worker's ``ready`` document: ``RoboDojoAdapter.capabilities`` (the real builder) on a stub adapter
    carrying the ``X5_NATIVE`` fixture numbers instead of simulator values; ``versions`` is a fixed placeholder."""
    import numpy as np
    from ni_rua.robot.backends.robodojo.robot_worker import RoboDojoAdapter
    adapter = object.__new__(RoboDojoAdapter)
    ticks = X5_NATIVE["native_step_ticks"]
    adapter.__dict__.update(
        motion_execution_mode=motion_execution_mode, motion_safety_guard_enabled=motion_safety_guard,
        trajectory_preview_enabled=trajectory_preview, observation_profile=observation_profile, np=np,
        env=SimpleNamespace(step_lim=X5_NATIVE["native_step_limit"], dt=X5_NATIVE["physics_dt_s"]),
        control_hz=X5_NATIVE["control_hz"], native_step_ticks=ticks,
        arm_step_limits={arm: np.full(6, X5_NATIVE["arm_step_limit_rad"]) for arm in ("left", "right")},
        arm_joint_limits={arm: [[-X5_NATIVE["joint_limit_rad"], X5_NATIVE["joint_limit_rad"]]] * 6
                          for arm in ("left", "right")},
        arm_joint_names={arm: list(X5_ARM_JOINTS) for arm in ("left", "right")},
        gripper_step_limit=min(1.0, 0.2 * ticks, 0.25), versions={"fixture": "embodiment_golden"})
    adapter.eef_gripper_dwell_step = min(adapter.gripper_step_limit, 0.18)
    return RoboDojoAdapter.capabilities(adapter)


def _x5_robot_class():
    from ni_rua.robot.backends.robodojo.robodojo import RobotBackend

    class X5GoldenRobot(RobotBackend):
        """The real RoboDojo parent backend (hooks: compile_action, validate_action, and any later profile hook) with
        an in-process deterministic world instead of the Isaac worker: ``start`` returns ``x5_worker_capabilities``,
        ``observe`` a fixed dual-arm observation in the worker's shape (cam_head + two wrist cameras, link6 TCP,
        calibration for calibrated profiles, 1x1 RGB and depth). No physics, no task state; execute/preview refuse."""

        def __init__(self, root, config):
            super().__init__(robodojo_root="/robodojo-not-used", python="/python-not-used", config=config,
                             run_dir=Path(root) / "robot")
            self.root = Path(root)
            self.root.mkdir(parents=True, exist_ok=True)
            self.png = self.root / "frame.png"
            self.png.write_bytes(_png_bytes())
            self.observations = 0
            self.capabilities = None

        def start(self):
            self.capabilities = x5_worker_capabilities(
                observation_profile=self.observation_profile, motion_execution_mode=self.motion_execution_mode,
                motion_safety_guard=self.motion_safety_guard, trajectory_preview=self.trajectory_preview)
            return deepcopy(self.capabilities)

        def _state(self):
            from ni_rua.robot.model.arx_contract import FRAME_ID, TCP_CONFIG_ID
            arms = {}
            for arm, y in (("left", 0.25), ("right", -0.25)):
                arms[arm] = {"joint_names": list(X5_ARM_JOINTS), "q_rad": [0.0, 0.4, -0.6, 0.2, 0.0, 0.0],
                             "qdot_rad_s": [0.0] * 6, "source_kind": "measurement",
                             "ee_pose": {"position_m": [0.3, y, 1.0], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                                         "frame_id": FRAME_ID, "tcp_config_id": TCP_CONFIG_ID,
                                         "source_kind": "measurement"},
                             "gripper": {"goal_normalized": 1.0, "goal_source_kind": "command",
                                         "command_normalized": 1.0, "command_source_kind": "command",
                                         "joint_names": ["joint7", "joint8"], "position_m": [0.044, 0.044],
                                         "position_source_kind": "measurement", "contact_feedback": None,
                                         "contact_feedback_reason": "no contact sensor exposed"}}
            return {"arms": arms, "frame_id": FRAME_ID, "tcp_config_id": TCP_CONFIG_ID, "sim_step": 0,
                    "step_domain": "adapter_control_step"}

        def observe(self):
            from ni_rua.core.observation_policy import LEGACY_ACTOR_PROFILE, actor_robot_state, allows_depth, has_calibration
            self.observations += 1
            name = f"raw-{self.observations}"
            cameras = []
            for camera_id, _ in X5_CAMERAS:
                camera = {"camera_id": camera_id, "frame_id": f"{name}-{camera_id}", "png_path": str(self.png),
                          "width": 1, "height": 1, "pixel_transform": "identity",
                          "capture_alignment": "physics paused; native zero-delay camera delivery fence",
                          "delivered_to_actor": False}
                if allows_depth(self.observation_profile):
                    from ni_rua.robot.backends.robodojo.robot_worker import save_depth_capture
                    import numpy as np
                    camera.update(save_depth_capture(self.root / f"{name}-{camera_id}.depth.npz",
                                                     np.full((1, 1), 0.8, np.float32), (1, 1, 3), (0.01, 10.0)))
                cameras.append(camera)
            state = self._state()
            raw = {"raw_observation_id": name, "data_origin": "robodojo_runtime", "world_kind": "observed",
                   "observation_profile": self.observation_profile, "robot_model_id": "arx-x5", "cameras": cameras,
                   "robot_state": (actor_robot_state(state, LEGACY_ACTOR_PROFILE)
                                   if self.observation_profile == LEGACY_ACTOR_PROFILE else state),
                   "sim_step": 0, "step_domain": "adapter_control_step", "control_dt_s": X5_NATIVE["physics_dt_s"],
                   "sim_time_s": 0.0, "captured_at": "2026-11-13T00:00:00+00:00", "instruction": TASK,
                   "versions": {"fixture": "embodiment_golden"}}
            if has_calibration(self.observation_profile):
                calibration, poses = {}, {}
                for camera_id, arm in X5_CAMERAS:
                    calibration[camera_id] = {"K": [[100.0, 0.0, 0.5], [0.0, 100.0, 0.5], [0.0, 0.0, 1.0]],
                                              "distortion": [0.0] * 5, "distortion_model": "pinhole_no_distortion",
                                              "image_size": [1, 1], "optical_axes": "+x right,+y down,+z forward",
                                              "calibration_source": "embodiment golden fixture"}
                    poses[camera_id] = {"T_world_camera": _translation(0.1, 0.0, 1.4), "frame_id": f"{name}-{camera_id}",
                                        "source_kind": "robot_kinematics" if arm else "simulator_camera_transform"}
                    if arm:
                        calibration[camera_id].update(arm=arm, link_name="link6", T_world_base=_translation(),
                                                      T_link_camera=_translation(0.05))
                        poses[camera_id].update(arm=arm, link_name="link6", T_world_link=_translation(0.3),
                                                T_world_base=_translation())
                    else:
                        calibration[camera_id]["T_world_camera"] = _translation(0.1, 0.0, 1.4)
                raw.update(camera_calibration={"profile_id": "robodojo-arx-x5-pinhole-v1",
                                               "source_kind": "simulator_configuration",
                                               "reference_frame_id": "env_origin", "cameras": calibration},
                           robot_model={"base_link": "base_link", "joints": []},
                           camera_pose_sample={"reference_frame_id": "env_origin", "sim_step": 0,
                                               "step_domain": "adapter_control_step", "cameras": poses})
            return raw

        def preview_motion(self, *args, **kwargs):
            raise NotImplementedError("embodiment golden fixture: no native planning")

        def preview_batch(self, action):
            raise NotImplementedError("embodiment golden fixture: no native planning")

        def execute(self, action, progress_callback=None, cancel_flag=None):
            raise NotImplementedError("embodiment golden fixture: no physics")

        def record(self, enabled=True):
            return {"recording": bool(enabled)}

        def close(self):
            return {"native_close_acknowledged": True, "process_exit_confirmed": True, "errors": []}

    return X5GoldenRobot


def _so101_robot_class():
    _tests_on_path()
    from support.so101_scene import SceneBackend

    class So101GoldenBackend(SceneBackend):
        """The real parent-side ``so101_real`` backend (deployment, contract hooks, motion instructions) over an
        in-process ``SO101Adapter`` with ``FakeArmDriver``, ``FakeClock`` and 1/10-size ``FakeCameraSource`` frames
        (deterministic; no serial port, camera or real-time thread)."""

        def __init__(self, **kwargs):
            super().__init__(scene=None, **kwargs)

        def start(self):
            if self.adapter is not None:
                return self.capabilities
            from ni_rua.robot.backends.so101.driver import FakeArmDriver, FakeCameraSource, FakeClock
            from ni_rua.robot.backends.so101.worker import SO101Adapter
            self.run_dir.mkdir(parents=True, exist_ok=True)
            clock = FakeClock()
            home = self.deployment["home"]
            driver = FakeArmDriver(self.deployment, clock, q_rad=home["q_rad"],
                                   gripper_normalized=home["gripper_normalized"])
            cameras = {name: FakeCameraSource(name, clock=clock, width=item["width"], height=item["height"])
                       for name, item in self.deployment["cameras"].items()}
            self.adapter = SO101Adapter(self.deployment, self.run_dir, self.observation_profile, driver=driver,
                                        cameras=cameras, clock=clock, image_settings=self.image_settings,
                                        deployment_path=self.deployment_path, start_telemetry=False)
            self.capabilities = self._request("ready")
            return self.capabilities

    return So101GoldenBackend


def so101_deployment(directory, *, empty_close_aperture_m=None):
    """Write ``configs/so101.example.json`` with 64x48 cameras and a synthetic camera calibration file (the
    ``test_so101_feedback_grip_guard`` fixture geometry) into ``directory``; returns the deployment path."""
    import numpy as np
    from ni_rua.robot.calibration.workflow import charuco as calibration
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    document = json.loads((REPOSITORY / "configs" / "so101.example.json").read_text())
    size = {}
    for name in ("wrist", "fixed"):
        camera = document["cameras"][name]
        size[name] = [camera["width"] // 10, camera["height"] // 10]
        camera.update(width=size[name][0], height=size[name][1])
    if empty_close_aperture_m is not None:
        document["gripper"]["empty_close_aperture_m"] = empty_close_aperture_m
    K = [[50.0, 0.0, 32.0], [0.0, 50.0, 24.0], [0.0, 0.0, 1.0]]
    fixed = np.eye(4)
    fixed[:3, 3] = [0.5, 0.0, 0.4]
    calibration.write_calibration_file(directory / "camera_calibration.json", board=calibration.DEFAULT_BOARD_SPEC,
        samples=20, cameras={
            "wrist": dict(K=K, distortion=[0.0] * 5, image_size=size["wrist"], T_link_camera=np.eye(4).tolist(),
                          intrinsics_rms_px=0.3, hand_eye_residual=dict(translation_m=0.001, rotation_rad=0.002)),
            "fixed": dict(K=K, distortion=[0.0] * 5, image_size=size["fixed"], T_base_camera=fixed.tolist(),
                          intrinsics_rms_px=0.3, extrinsic_residual=dict(reprojection_px=0.5))})
    document["camera_calibration_file"] = "camera_calibration.json"
    path = directory / "deployment.json"
    path.write_text(json.dumps(document, indent=2))
    return path




# ---- configurations ----------------------------------------------------------------------------------------------
@contextlib.contextmanager
def repository_cwd():
    """Run with the repository as working directory: ``Config`` resolves relative default paths (``scene.root``)
    against it, so captures do not depend on where the tests were started."""
    previous = os.getcwd()
    os.chdir(REPOSITORY)
    try:
        yield
    finally:
        os.chdir(previous)


@contextlib.contextmanager
def private_home():
    """Run with HOME at a fresh empty directory: ``Config`` resolves every ``~`` path default, and under the account's
    own home that follows whatever links or files the account has there (a linked ``~/.local/share/ni-rua``, a
    ``bin/python`` link to ``python3.11``), so captures would depend on who runs them. The Redactor maps the private
    home back to ``~``."""
    with tempfile.TemporaryDirectory(prefix="home-") as home, mock.patch.dict(os.environ, {"HOME": home}):
        yield home


def load_config(layer=None, overrides=None):
    """``Config.load`` as the gateway runs it (``configs/ni-rua.json`` under the repository-relative ``layer`` file,
    then ``overrides``), but with an empty environment and without ``configs/ni-rua.local.json``."""
    from ni_rua.core import config as config_module
    from ni_rua.core.config import Config
    with repository_cwd(), mock.patch.object(config_module, "LOCAL_CONFIG_PATH",
                                             REPOSITORY / "configs" / "ni-rua.local.json.absent"):
        return Config.load(REPOSITORY / layer if layer else None, environ={}, overrides=overrides)


# Shipped configurations: name -> repository-relative layer over configs/ni-rua.json (None: ni-rua.json alone).
X5_CONFIGS = {
    "ni_rua": None,
    "rgbd_feedback_v2": "configs/rgbd-feedback-v2.json",
    "rgbd_numeric_v1": "configs/rgbd-numeric-v1.json",
    "rgbd_skills_v3": "configs/rgbd-skills-v3.json",
    "benchmark_shared_memory": "configs/benchmark-shared-memory.json",
    "ni_rua_example": "configs/ni-rua.example.json",
}
SO101_CONFIGS = {"so101_gateway": "configs/so101_gateway.json"}
# Startup scenarios: name -> (config layer or "<prompt-capture>", CLI-equivalent overrides, benchmark). Every scenario
# also sets self_briefing off (it would call a model), terminal off (no PTY) and grasp memory in the temporary root.
PROMPT_CAPTURE = "<prompt-capture>"
X5_STARTUP = {
    "ni_rua_benchmark": (None, {}, True),
    "ni_rua_interactive": (None, {}, False),
    "rgbd_feedback_v2": ("configs/rgbd-feedback-v2.json", {}, True),
    "rgbd_numeric_v1": ("configs/rgbd-numeric-v1.json", {}, True),
    "rgbd_skills_v3": ("configs/rgbd-skills-v3.json", {}, True),
    "legacy_rgb_proprio_v1": (None, {"features": {"observation_profile": "rgb_proprio_v1"}}, True),
    # support.prompt_capture path: Config() defaults and its calibrated ARX X5 fixture robot (ParityRobot).
    "prompt_capture_fixture": (PROMPT_CAPTURE, {}, True),
}
SO101_STARTUP = {"so101_gateway": ("configs/so101_gateway.json", {}, False)}


def _merge(base, update):
    from ni_rua.core.config import _merge as merge
    return merge(deepcopy(base), deepcopy(update))


# The goldens record the actor-facing form before features.tool_catalog_profile became layered_v3 and
# features.robot_urdf_delivery became file: the prose catalogue and the URDF inline in the startup instructions.
# Every golden scenario states these two settings explicitly (no allowance, so the recorded bytes stay the reference);
# the shipped defaults of both, and that the old settings reproduce the recorded form, are pinned by
# tests/robot/embodiment/test_embodiment_contracts.py (ReplyDeliveryDefaultsTests) and tests/core/config/test_config.py.
RECORDED_CATALOGUE_FEATURES = dict(tool_catalog_profile="full_v1", robot_urdf_delivery="inline")


def _scenario_config(layer, overrides, root, recorded_catalogue=True):
    """``recorded_catalogue=False``: leave the two settings of ``RECORDED_CATALOGUE_FEATURES`` at their defaults."""
    from ni_rua.core.config import Config
    common = {"features": {"self_briefing": False, "terminal": False, "grasp_memory_dir": str(root / "grasp-memory"),
                           **(RECORDED_CATALOGUE_FEATURES if recorded_catalogue else {})}}
    if layer == PROMPT_CAPTURE:
        common["features"]["sandbox_enabled"] = False
        return Config.from_dict(_merge(common, overrides))
    return load_config(layer, overrides=_merge(common, overrides))


def _workbench_start(config, robot, root, benchmark):
    """Run the real ``Workbench`` startup with ``FakeAgentSession``; return what it handed to the session."""
    from ni_rua.apps.gateway import workbench as workbench_module
    from ni_rua.apps.gateway.workbench import DEFAULT_PROMPT, Workbench
    from support.fake_agent import FakeAgentSession
    created, inputs = [], {}

    def create(config, run_dir, store, tool_handler, *, price_card=None, task_planning=None):
        session = FakeAgentSession(run_dir, store, tool_handler, price_card=price_card)
        start, run_turn = session.start, session.run_turn

        def recording_start(tools, instructions, *args, **kwargs):
            inputs.update(tools=deepcopy(tools), instructions=instructions)
            return start(tools, instructions, *args, **kwargs)

        def recording_turn(prompt, *args, **kwargs):
            inputs.setdefault("prompt", prompt)
            return run_turn(prompt, *args, **kwargs)
        session.start, session.run_turn = recording_start, recording_turn
        created.append(session)
        return session

    with mock.patch.object(workbench_module, "create_session", side_effect=create):
        workbench = Workbench(root / "run", robot=robot, visual_review_model=None, config=config)

    async def run():
        try:
            await workbench.start(DEFAULT_PROMPT, benchmark=benchmark)
            program_tools = workbench.harness.tool_registry.catalog(workbench.harness, role="program")
            records = [value for name, value in workbench.store.record_items("configuration").items()
                       if name.startswith("task_prompt-")]
            return program_tools, records
        finally:
            await workbench.close()
    program_tools, records = asyncio.run(run())
    if "instructions" not in inputs or "prompt" not in inputs:
        raise RuntimeError("workbench_startup_did_not_reach_the_session")
    session, = created
    effective = json.loads((root / "run" / "effective_config.json").read_text())
    extra = {workbench.identity["episode_id"]: "<episode_id>",
             workbench.identity["env_instance_id"]: "<env_instance_id>", session.thread_id: "<thread_id>"}
    return dict(inputs, program_tools=program_tools, records=records, effective_config=effective,
                capabilities=workbench.harness.capabilities, redactor=Redactor(root / "run", root, extra))


@functools.lru_cache(maxsize=None)
def run_startup(embodiment, scenario):
    """One startup scenario of ``X5_STARTUP`` / ``SO101_STARTUP`` -> redacted JSON data: what the session received
    (``instructions``, first-turn ``prompt``, actor ``tools``), the program-role catalogue ``program_tools``, the
    published ``capabilities``, the archived ``task_prompt`` records and the written ``effective_config``.
    Cached per process: the catalogue, startup and config families share one run per scenario."""
    _tests_on_path()
    layer, overrides, benchmark = (X5_STARTUP if embodiment == "x5" else SO101_STARTUP)[scenario]
    with repository_cwd(), tempfile.TemporaryDirectory(prefix="ni-rua-embodiment-golden-") as directory:
        root = Path(directory).resolve()
        if embodiment == "x5":
            config = _scenario_config(layer, overrides, root)
            if layer == PROMPT_CAPTURE:
                from support.prompt_capture import _fixture_robot
                robot = _fixture_robot(root / "robot")
            else:
                robot = _x5_robot_class()(root / "robot", config)
        else:
            deployment = so101_deployment(root / "deployment")
            config = _scenario_config(layer, _merge({"backend": {"deployment": str(deployment)}}, overrides), root)
            robot = _so101_robot_class()(config=config, run_dir=root / "run" / "private" / "robot")
        captured = _workbench_start(config, robot, root, benchmark)
        redact = captured.pop("redactor")
        result = dict(instructions=captured["instructions"], prompt=captured["prompt"], tools=captured["tools"],
                      program_tools=captured["program_tools"], capabilities=captured["capabilities"],
                      task_prompt=captured["records"], effective_config=captured["effective_config"])
        return normalize(redact(normalize(result)))


def _tests_on_path():
    tests = str(REPOSITORY / "tests")
    if tests not in sys.path:
        sys.path.insert(0, tests)


def _startup_table(embodiment):
    return X5_STARTUP if embodiment == "x5" else SO101_STARTUP


# ---- family: catalogue -------------------------------------------------------------------------------------------
# public_definitions(**flags) for the perception flag combinations create_registry passes (vggt mode, object_table,
# scene3d, observe3d = (object_table_observe, scene3d_observe, obs3d_action_overlay), pose_skill).
_SHIPPED_PERCEPTION = dict(vggt_height="optional", object_table=True, scene3d=True, observe3d=(True, True, False))
DEFINITION_FLAGS = {
    "default": {},
    "pose_skill": dict(pose_skill=True),
    "perception_shipped": _SHIPPED_PERCEPTION,
    "perception_shipped_pose_skill": dict(_SHIPPED_PERCEPTION, pose_skill=True),
    "vggt_default_mode": dict(_SHIPPED_PERCEPTION, vggt_height="default"),
    "obs3d_action_overlay": dict(_SHIPPED_PERCEPTION, observe3d=(True, True, True)),
    "object_table_observe_only": dict(_SHIPPED_PERCEPTION, observe3d=(True, False, False)),
    "scene3d_observe_only": dict(_SHIPPED_PERCEPTION, observe3d=(False, True, False)),
    "vggt_only": dict(vggt_height="optional"),
    "object_table_only": dict(object_table=True),
    "scene3d_only": dict(scene3d=True),
}


def capture_catalogue(embodiment):
    """``definitions/<flags>`` (X5 only: the shared builder) and ``actor/<scenario>`` / ``program/<scenario>``
    registry catalogues of every startup scenario: lists of complete tool definitions."""
    from ni_rua.runtime.tools.definitions import public_definitions
    items = {}
    if embodiment == "x5":
        for name, flags in DEFINITION_FLAGS.items():
            items["definitions/" + name] = normalize(public_definitions(**flags))
    for scenario in _startup_table(embodiment):
        startup = run_startup(embodiment, scenario)
        items["actor/" + scenario] = startup["tools"]
        items["program/" + scenario] = startup["program_tools"]
    return items


# ---- family: instructions ----------------------------------------------------------------------------------------
SOURCE_ROOT_PLACEHOLDER = "<run>/private/actor-source"
HANDOFF_VARIANTS = {"surface_preferred": (True, "surface"), "airborne": (True, "airborne"), "surface_only": (False, None)}
OBS3D_FEATURES = {
    "config_default": {},
    "ni_rua_json": None,
    "external_perception_off": {"external_perception": False},
    "vggt_default_mode": {"vggt_height_mode": "default"},
    "obs3d_action_overlay": {"obs3d_action_overlay": True},
    "object_table_observe_off": {"object_table_observe": False},
    "scene3d_observe_off": {"scene3d_observe": False},
    "object_table_off": {"object_table": False, "object_table_observe": False},
}


def _contracts():
    from ni_rua.cortex.context.actor_instructions import (CALIBRATED_OBSERVATION_INSTRUCTIONS,
                                                          LEGACY_OBSERVATION_INSTRUCTIONS)
    from ni_rua.perception.depth.depth_runtime import INSTRUCTIONS as DEPTH_INSTRUCTIONS
    return {"calibrated": CALIBRATED_OBSERVATION_INSTRUCTIONS, "legacy": LEGACY_OBSERVATION_INSTRUCTIONS,
            "depth": DEPTH_INSTRUCTIONS}


def capture_instructions(embodiment):
    """Startup instruction builders, rendered per variant (text items, except ``handoff_policy/*`` records).

    ``actor/<contract>/<sandbox|local>/<memory|no_memory>``: ``actor_instructions.instructions`` (local = no sandbox,
    Harness source under ``SOURCE_ROOT_PLACEHOLDER``) with the default handoff policy; ``actor_handoff/<variant>``:
    calibrated/local/memory with each handoff policy; SO-101 passes its backend ``actor_motion_instructions``
    (calibrated and legacy only: the backend refuses depth)."""
    from ni_rua.cortex.context import actor_instructions as module
    from ni_rua.cortex.context.task_notes import handoff_policy
    from ni_rua.core.config import Config
    from ni_rua.perception.external.obs3d import developer_instructions
    items, contracts = {}, _contracts()
    motion = profile = None
    if embodiment == "so101":
        # The real Workbench path (design 13 section 12, WP3 note): the SO-101 profile of the deployment, its legacy
        # observation contract, no handoff text (profile.handoff is false) and instructions(..., profile=profile).
        from ni_rua.robot.backends.so101 import contract
        from ni_rua.robot.backends.so101.deployment import embodiment_profile, validate_deployment
        document = json.loads((REPOSITORY / "configs" / "so101.example.json").read_text())
        deployment = validate_deployment(document, base=REPOSITORY / "configs")
        profile = embodiment_profile(deployment)
        motion = contract.actor_motion_instructions(deployment)
        items["so101/actor_motion_instructions"] = motion
        items["so101/grip_guard_text"] = contract.grip_guard_text(deployment)
        contracts.pop("depth")
        contracts["legacy"] = module.legacy_observation_instructions(profile)

    def handoff_text(policy):
        return policy["text"] if profile is None or profile.handoff else None
    default_handoff = handoff_text(handoff_policy(True, handoff_preference="surface"))
    for contract_name, text in contracts.items():
        for sandbox in ("sandbox", "local"):
            for memory in ("memory", "no_memory"):
                items[f"actor/{contract_name}/{sandbox}/{memory}"] = module.instructions(
                    text, default_handoff, source_root=None if sandbox == "sandbox" else SOURCE_ROOT_PLACEHOLDER,
                    operation_memory=memory == "memory", embodiment_motion=motion, profile=profile)
    for name, (airborne, preference) in HANDOFF_VARIANTS.items():
        policy = handoff_policy(airborne, handoff_preference=preference)
        items["handoff_policy/" + name] = policy
        items["actor_handoff/" + name] = module.instructions(
            contracts["calibrated"], handoff_text(policy), source_root=SOURCE_ROOT_PLACEHOLDER,
            embodiment_motion=motion, profile=profile)
    if embodiment == "x5":
        ni_rua = load_config().features
        for name, update in OBS3D_FEATURES.items():
            features = ni_rua if update is None else Config.from_dict({"features": update}).features
            items["obs3d/" + name] = developer_instructions(features)
        from ni_rua.perception.external.pose_skill import INSTRUCTIONS as pose
        from ni_rua.perception.external.held_perception import INSTRUCTION as held
        from ni_rua.cortex.context.operation_verification import INSTRUCTIONS as verification
        items.update({"pose_skill": pose, "held_perception": held, "operation_verification": verification})
        for name, text in contracts.items():
            items["observation_contract/" + name] = text
    return items


# ---- family: startup ---------------------------------------------------------------------------------------------
def capture_startup(embodiment):
    """Per startup scenario: ``<scenario>/instructions`` and ``<scenario>/prompt`` (text the session received),
    ``<scenario>/tool_names`` (the definitions are the catalogue family's ``actor/<scenario>``),
    ``<scenario>/capabilities`` (runtime capabilities after Harness.start) and ``<scenario>/task_prompt``."""
    items = {}
    for scenario in _startup_table(embodiment):
        startup = run_startup(embodiment, scenario)
        items[scenario + "/instructions"] = startup["instructions"]
        items[scenario + "/prompt"] = startup["prompt"]
        items[scenario + "/tool_names"] = [tool["name"] for tool in startup["tools"]]
        items[scenario + "/capabilities"] = startup["capabilities"]
        items[scenario + "/task_prompt"] = startup["task_prompt"]
    return items


# ---- family: compile ---------------------------------------------------------------------------------------------
# The corpus is generated once from the 5fd93c6 schema and preset tables and stored with the outcomes; tests replay
# the STORED requests (never a corpus regenerated from the current schema).
OBSERVATION_ID = "obs-golden"
DELTA = 1e-6  # "just outside" a numeric bound (m, rad, s or dimensionless, as the field)
MOVE = {"id": "m", "kind": "move", "arm": "left", "position_m": [0.3, 0.25, 0.95], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]}
GRASP = {"id": "g", "kind": "move", "arm": "left", "grasp_position_m": [0.3, 0.25, 0.8], "angles_deg": [0.0, 0.0, 0.0]}
JAW = {"id": "j", "kind": "gripper", "arm": "left", "goal_normalized": 0.0}
OBSERVE = {"id": "o", "kind": "observe", "reason": "inspect the grasp"}
BOX = {"id": "b1", "min_m": [0.4, -0.05, 0.75], "max_m": [0.45, 0.0, 0.8], "evidence": "estimated from RGB"}
_GRASP_ONLY = {"grasp_position_m", "angles_deg", "contact_region", "contact_depth_m"}
_NEEDS_JAW_GOAL = {"gripper_motion", "gripper_speed_scale", "gripper_speed_preset", "checkpoint_after_gripper"}


def _numbers(schema, integer=False):
    step = 1 if integer else DELTA
    values = []
    if "minimum" in schema:
        values += [schema["minimum"], schema["minimum"] - step]
    if "exclusiveMinimum" in schema:
        values += [schema["exclusiveMinimum"], schema["exclusiveMinimum"] + step]
    if "maximum" in schema:
        values += [schema["maximum"], schema["maximum"] + step]
    if "exclusiveMaximum" in schema:
        values += [schema["exclusiveMaximum"], schema["exclusiveMaximum"] - step]
    if not values:
        values = [1 if integer else 1.0]
    return [round(value, 12) if isinstance(value, float) else value for value in values] + ["1"]


def _field_values(field, schema, presets=()):
    """Values for one field: bounds and just outside, every enum (plus runtime ``presets`` and a bogus name), both
    booleans, malformed shapes."""
    if field in ("position_m", "grasp_position_m"):
        base = MOVE["position_m"] if field == "position_m" else GRASP["grasp_position_m"]
        return [base, base[:2], base + [0.0], ["0.3", 0.25, 0.8]]
    if field == "quaternion_xyzw":
        return [[0.0, 0.0, 0.0, 1.0], [0.0, 0.7071068, 0.0, 0.7071068], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 2.0]]
    if field == "angles_deg":
        values = [[0.0, 0.0, 0.0], [0.0, 0.0], [0.0, 0.0, "0"]]
        for index, item in enumerate(schema["prefixItems"]):
            for bound in (item["minimum"], item["maximum"]):
                for value in (bound, bound + (DELTA if bound == item["maximum"] else -DELTA)):
                    angles = [0.0, 0.0, 0.0]
                    angles[index] = round(value, 12)
                    values.append(angles)
        return values
    if field == "gripper_motion":
        return [{"start_fraction": 0.2, "end_fraction": 0.9}, {"start_fraction": 0.0, "end_fraction": 1.0},
                {"start_fraction": 0.9, "end_fraction": 0.2}, {"start_fraction": 1.0, "end_fraction": 1.0},
                {"start_fraction": 0.2}, {"start_fraction": 0.2, "end_fraction": 0.9, "ramp": 1}]
    if "enum" in schema or presets:
        names = list(dict.fromkeys([*schema.get("enum", ()), *presets]))
        return names + ["bogus"]
    kind = schema.get("type")
    if kind in ("number", "integer"):
        return _numbers(schema, integer=kind == "integer")
    if kind == "boolean":
        return [True, False, "true"]
    if kind == "string":
        return ["x", "", " "]
    if kind == "array":
        return [["o1"], [], ["o1", "o1"]]
    return [None]


def _runtime_presets():
    from ni_rua.robot.motion.grip_guard import GRIP_GUARD_PRESETS
    from ni_rua.robot.motion.motion_safety import JOINT_TRAVEL_PRESETS
    from ni_rua.robot.motion.motion_speed import ARM_PRESETS, GRIPPER_SPEED_PRESETS
    from ni_rua.robot.motion.motion_tolerance import STALL_GUARD_PRESETS, TOLERANCE_PRESETS
    from ni_rua.robot.motion.scene_obstacles import CLEARANCE_PRESETS, TARGET_EXEMPTION_PRESETS
    return {"grip_guard_preset": list(GRIP_GUARD_PRESETS), "joint_travel_preset": list(JOINT_TRAVEL_PRESETS),
            "tolerance_preset": list(TOLERANCE_PRESETS), "stall_guard_preset": list(STALL_GUARD_PRESETS),
            "clearance_preset": list(CLEARANCE_PRESETS), "target_exemption_preset": list(TARGET_EXEMPTION_PRESETS),
            "speed_preset": list(ARM_PRESETS), "gripper_speed_preset": list(GRIPPER_SPEED_PRESETS),
            "obstacle_policy": ["persist", "stateless", "clear"]}


def compile_corpus():
    """Ordered ``[(label, request)]`` of robot_action preview requests (generated from the current schema)."""
    from ni_rua.runtime.actions.robot_action_schema import ROBOT_ACTION_TOOL, _STAGE_SCHEMAS
    presets = _runtime_presets()
    cases = []

    def add(label, stages, **batch):
        cases.append((label, dict(observation_id=OBSERVATION_ID, action_id="golden-%03d" % (len(cases) + 1),
                                  stages=deepcopy(stages), **deepcopy(batch))))

    def stage(base, **fields):
        return dict(deepcopy(base), **fields)

    for arm in ("left", "right"):
        add(f"base/move_link6/{arm}", [stage(MOVE, arm=arm)])
        add(f"base/move_grasp/{arm}", [stage(GRASP, arm=arm)])
        add(f"base/gripper/{arm}", [stage(JAW, arm=arm)])
        add(f"base/move_with_jaw_goal/{arm}", [stage(MOVE, arm=arm, gripper_goal_normalized=0.0)])
        add(f"base/observe_named_arm/{arm}", [stage(MOVE, arm=arm), stage(OBSERVE, arm=arm)])
        add(f"base/pick_sequence/{arm}", [stage(GRASP, arm=arm, id="approach"), stage(JAW, arm=arm, id="close"),
                                          stage(MOVE, arm=arm, id="lift")])
    add("base/observe_only", [OBSERVE])
    add("base/observe_barrier", [MOVE, OBSERVE, stage(MOVE, id="m2", arm="right")])
    for kind, base in (("move", MOVE), ("gripper", JAW), ("observe", OBSERVE)):
        for field, schema in _STAGE_SCHEMAS[kind]["properties"].items():
            if field in ("id", "kind", "arm", "depends_on"):
                continue
            selected = GRASP if field in _GRASP_ONLY and kind == "move" else base
            extra = {"gripper_goal_normalized": 0.0} if kind == "move" and field in _NEEDS_JAW_GOAL else {}
            for value in _field_values(field, schema, presets.get(field, ())):
                add(f"stage/{kind}/{field}={render(value)}", [stage(selected, **extra, **{field: value})])
            if extra:
                add(f"stage/{kind}/{field}/without_gripper_goal",
                    [stage(selected, **{field: _field_values(field, schema, presets.get(field, ()))[0]})])
    batch_fields = ("checkpoint_after_gripper", "grip_guard_preset", "min_aperture_m", "max_aperture_m",
                    "joint_travel_preset", "max_joint_travel_rad", "max_backtracking_rad", "tolerance_preset",
                    "position_tolerance_m", "orientation_tolerance_rad", "max_upward_dz_m", "stall_guard_preset",
                    "max_tracking_divergence_m", "timeout_s", "clearance_preset", "clearance_height_m",
                    "target_exemption_preset", "target_exemption_radius_m")
    properties = ROBOT_ACTION_TOOL["inputSchema"]["properties"]
    for field in batch_fields:
        for value in _field_values(field, properties[field], presets.get(field, ())):
            add(f"batch/{field}={render(value)}", [MOVE, JAW], **{field: value})
    add("batch/stage_overrides_batch_tolerance", [stage(MOVE, tolerance_preset="tight")], tolerance_preset="coarse")
    add("batch/stage_overrides_batch_grip_guard", [stage(JAW, grip_guard_preset="none")], grip_guard_preset="rigid_object")
    add("batch/grip_guard_on_opening_jaw", [stage(JAW, goal_normalized=1.0)], grip_guard_preset="rigid_object")
    add("batch/conflicting_clearance", [MOVE], clearance_preset="low", clearance_height_m=0.1)
    add("batch/conflicting_exemption", [MOVE], target_exemption_preset="tight", target_exemption_radius_m=0.02)
    # structure
    add("structure/dual_arm", [MOVE, stage(MOVE, id="m2", arm="right")])
    add("structure/cross_arm_dependency", [MOVE, stage(MOVE, id="m2", arm="right", depends_on=["m"])])
    add("structure/unknown_dependency", [stage(MOVE, depends_on=["nope"])])
    add("structure/self_dependency", [stage(MOVE, depends_on=["m"])])
    add("structure/dependency_cycle", [stage(MOVE, depends_on=["m2"]), stage(MOVE, id="m2", arm="right", depends_on=["m"])])
    add("structure/duplicate_dependency", [MOVE, stage(MOVE, id="m2", arm="right", depends_on=["m", "m"])])
    add("structure/duplicate_ids", [MOVE, stage(JAW, id="m")])
    add("structure/id_80_chars", [stage(MOVE, id="i" * 80)])
    add("structure/id_81_chars", [stage(MOVE, id="i" * 81)])
    add("structure/empty_id", [stage(MOVE, id="")])
    add("structure/missing_id", [{key: value for key, value in MOVE.items() if key != "id"}])
    add("structure/missing_kind", [{key: value for key, value in MOVE.items() if key != "kind"}])
    add("structure/unknown_kind", [stage(MOVE, kind="joint")])
    add("structure/missing_arm", [{key: value for key, value in MOVE.items() if key != "arm"}])
    add("structure/arm_both", [stage(MOVE, arm="both")])
    add("structure/unknown_stage_field", [stage(MOVE, via_grasp_poses=[])])
    add("structure/both_pose_forms", [stage(MOVE, grasp_position_m=[0.3, 0.25, 0.8], angles_deg=[0.0, 0.0, 0.0])])
    add("structure/position_without_quaternion", [{key: value for key, value in MOVE.items() if key != "quaternion_xyzw"}])
    add("structure/grasp_without_angles", [{key: value for key, value in GRASP.items() if key != "angles_deg"}])
    add("structure/jaw_split_id_conflict", [stage(MOVE, gripper_goal_normalized=0.0), stage(JAW, id="m/motion", arm="right")])
    add("structure/gripper_without_goal", [{key: value for key, value in JAW.items() if key != "goal_normalized"}])
    add("structure/observe_extra_field", [stage(OBSERVE, position_m=[0.0, 0.0, 0.0])])
    add("structure/no_stages", [])
    add("structure/stages_not_list", {"m": MOVE})
    add("structure/32_stages_16_per_arm", [stage(MOVE, id=f"{arm}{index}", arm=arm)
                                           for index in range(16) for arm in ("left", "right")])
    add("structure/33_stages", [stage(MOVE, id=f"{arm}{index}", arm=arm)
                                for index in range(16) for arm in ("left", "right")] + [stage(JAW, id="extra")])
    add("structure/17_stages_one_arm", [stage(MOVE, id=f"left{index}") for index in range(17)])
    add("structure/16_moves_with_jaw_goals", [stage(MOVE, id=f"left{index}", gripper_goal_normalized=0.0)
                                              for index in range(16)])
    add("structure/checkpoint_without_jaw_event", [stage(MOVE, checkpoint_after_gripper=True)])
    add("continuation/unknown_source_plan", None, source_plan_id="p-unknown", start_from_stage_id="m")
    add("continuation/start_without_source", [MOVE], start_from_stage_id="m")
    add("continuation/source_and_stages", [MOVE], source_plan_id="p-unknown", start_from_stage_id="m")
    # proxies and the per-episode obstacle registry (order matters: obstacles persist across requests)
    add("proxies/held_object_single_arm", [MOVE], held_object={k: v for k, v in BOX.items() if k != "id"})
    add("proxies/held_object_with_arm", [MOVE], held_object={**{k: v for k, v in BOX.items() if k != "id"}, "arm": "left"})
    add("proxies/held_object_dual_without_arm", [MOVE, stage(MOVE, id="m2", arm="right")],
        held_object={k: v for k, v in BOX.items() if k != "id"})
    add("proxies/held_objects", [MOVE], held_objects={"left": {k: v for k, v in BOX.items() if k != "id"}})
    add("proxies/held_object_and_held_objects", [MOVE], held_object={k: v for k, v in BOX.items() if k != "id"},
        held_objects={"left": {k: v for k, v in BOX.items() if k != "id"}})
    add("proxies/contact_entity", [MOVE], obstacles=[BOX], contact_entity_id="b1", obstacle_policy="stateless")
    add("proxies/initial_contact", [MOVE], obstacles=[BOX], initial_contact={"entity_ids": ["b1"], "evidence": "touching"},
        obstacle_policy="stateless")
    add("proxies/placement_contact", [MOVE], obstacles=[BOX], obstacle_policy="stateless",
        placement_contact={"entity_id": "b1", "evidence": "support", "max_penetration_m": 0.002})
    add("obstacles/stateless", [MOVE], obstacles=[BOX], obstacle_policy="stateless")
    add("obstacles/plain_after_stateless", [MOVE])
    add("obstacles/persist", [MOVE], obstacles=[BOX])
    add("obstacles/plain_after_persist", [MOVE])
    add("obstacles/ignore_persisted", [MOVE], ignore_obstacle_ids=["b1"])
    add("obstacles/clear", [MOVE], obstacle_policy="clear")
    add("obstacles/plain_after_clear", [MOVE])
    add("obstacles/object_ids_without_table", [MOVE], obstacle_object_ids=["o1"])
    add("obstacles/invalid_policy", [MOVE], obstacle_policy="bogus")
    add("obstacles/inverted_box", [MOVE], obstacles=[dict(BOX, min_m=BOX["max_m"], max_m=BOX["min_m"])],
        obstacle_policy="stateless")
    return [(label, request if request.get("stages") is not None else
             {key: value for key, value in request.items() if key != "stages"}) for label, request in cases]


def _descriptions(schema):
    found = []
    if isinstance(schema, dict):
        for key, value in schema.items():
            if key == "description" and isinstance(value, str):
                found.append(value)
            else:
                found += _descriptions(value)
    elif isinstance(schema, list):
        for value in schema:
            found += _descriptions(value)
    return found


def _schema_outcome(registry, request):
    """``"ok"`` or the actor-visible ToolInputError fields. Field descriptions quoted in ``suggestion`` are replaced by
    ``<field description>``: their text is frozen by the catalogue family (where the approved correction applies)."""
    from ni_rua.core.tool_errors import ToolInputError
    try:
        registry.validate("robot_action", deepcopy(request), role="actor")
        return "ok"
    except ToolInputError as error:
        suggestion = error.suggestion or ""
        definition = registry.spec("robot_action", role="actor").definition
        for text in sorted(set(_descriptions(definition["inputSchema"])), key=len, reverse=True):
            suggestion = suggestion.replace(text, "<field description>")
        return dict(error=type(error).__name__, code=error.code, field_path=error.field_path,
                    actual=normalize(error.actual) if error.actual is not None else None, suggestion=suggestion,
                    minimal_example=error.minimal_example)
    except Exception as error:  # noqa: BLE001 - recorded as the outcome
        return error_text(error)


def compile_outcome(harness, request):
    """Decisions for one request: schema admission, ``compile_request`` (compiled BatchAction or error),
    ``validate_batch_action`` dependencies, and the backend's own ``validate_action``. Leaf ``new_id`` action ids
    are redacted per case."""
    from ni_rua.robot.motion.action_compilation import validate_backend_action
    from ni_rua.robot.motion.batch_contract import validate_batch_action
    from ni_rua.runtime.actions.unified_actions import compile_request
    result = dict(schema=_schema_outcome(harness.tool_registry, request))
    try:
        action = compile_request(harness, deepcopy(request))
    except Exception as error:  # noqa: BLE001
        result["compile"] = error_text(error)
        return result
    result["compile"] = action
    try:
        dependencies = validate_batch_action(deepcopy(action), harness.capabilities)
        result["dependencies"] = {key: sorted(value) for key, value in dependencies.items()}
    except Exception as error:  # noqa: BLE001
        result["dependencies"] = error_text(error)
    try:
        validate_backend_action(harness.robot, deepcopy(action), harness.capabilities)
        result["backend"] = "ok"
    except Exception as error:  # noqa: BLE001
        result["backend"] = error_text(error)
    return normalize(Redactor()(normalize(result)))


IDENTITY = dict(run_id="golden-run", episode_id="golden-episode", env_instance_id="golden-env", env_generation=1)


def compile_harness(embodiment, root):
    """A started Harness of the shipped configuration (X5: configs/ni-rua.json + ``X5GoldenRobot``; SO-101:
    configs/so101_gateway.json + ``So101GoldenBackend``) with the fixed ``IDENTITY``."""
    from ni_rua.core.storage.store import Store
    from ni_rua.runtime.harness import Harness
    _tests_on_path()
    root = Path(root)
    if embodiment == "x5":
        config = load_config(overrides={"features": {"grasp_memory_dir": str(root / "grasp-memory")}})
        robot = _x5_robot_class()(root / "robot", config)
    else:
        deployment = so101_deployment(root / "deployment", empty_close_aperture_m=SO101_EMPTY_CLOSE_M)
        config = load_config(SO101_CONFIGS["so101_gateway"], overrides={"backend": {"deployment": str(deployment)}})
        robot = _so101_robot_class()(config=config, run_dir=root / "robot")
    features = config.features
    store = Store(root / "store", dict(IDENTITY))
    harness = Harness(store, robot, actor_payload_profile=features.actor_payload_profile, enable_wait=features.enable_wait,
                      observation_profile=features.observation_profile, motion_review=features.motion_review,
                      vision_compare=features.vision_compare, config=config, task_plan_required=True)
    harness.start()
    return harness


def capture_compile(embodiment, corpus=None):
    """``<nnn> <label>`` -> ``{"request", "schema", "compile", "dependencies", "backend"}`` for every corpus case, in
    order, on one Harness (the obstacle registry carries over between cases, as in an episode). ``corpus``:
    ``[(label, request)]`` to replay (default: ``compile_corpus()``)."""
    corpus = compile_corpus() if corpus is None else corpus
    items = {}
    with tempfile.TemporaryDirectory(prefix="ni-rua-embodiment-compile-") as directory:
        harness = compile_harness(embodiment, directory)
        try:
            for index, (label, request) in enumerate(corpus, 1):
                items["%03d %s" % (index, label)] = dict(request=normalize(request), **compile_outcome(harness, request))
        finally:
            harness.store.close()
            close = getattr(harness.robot, "close", None)
            if callable(close):
                close()
    return items


def corpus_from_items(items):
    """The stored ``[(label, request)]`` of a compile fixture, in order."""
    return [(name.split(" ", 1)[1], value["request"]) for name, value in items.items()]


# ---- family: guard -----------------------------------------------------------------------------------------------
SO101_EMPTY_CLOSE_M = 0.0035  # measured empty-close gap of the fake SO-101 jaw (test_so101_feedback_grip_guard)
GUARD_SEED, GUARD_TRIALS, GUARD_CHUNK, GUARD_SAMPLE = 7, 40000, 5000, 60
GUARD_PRESETS = [None, "none", "non_empty", "thin_object", "rigid_object", "wide_object", "normal", "thin", "solid",
                 "empty_only"]
GUARD_VALUES = [None, 0.0, 0.0004, 0.0005, 0.0015, 0.0019, 0.002, 0.0021, 0.003, 0.02, 0.088]


def _gripper(embodiment):
    """Keyword arguments the backend passes to the shared grip-guard rules (X5 passes none)."""
    if embodiment == "x5":
        return {}, True
    from ni_rua.robot.backends.so101 import contract
    from ni_rua.robot.backends.so101.deployment import validate_deployment
    document = json.loads((REPOSITORY / "configs" / "so101.example.json").read_text())
    document["gripper"]["empty_close_aperture_m"] = SO101_EMPTY_CLOSE_M
    deployment = validate_deployment(document, base=REPOSITORY / "configs")
    return contract.guard_parameters(deployment), contract.post_grasp_recheck_available(deployment)


def guard_cases(embodiment):
    """The seeded differential of design 13 G-GUARD (the review-fbgrip ``diff_guard.py`` draw sequence, 40,000 trials of
    1..6 stages, 140,483 cases): per case the inputs, ``stage_grip_guard`` resolution (or error),
    ``after_stage_grip_guard`` evaluation, trigger and the armed state after it. Yields JSON records."""
    from ni_rua.robot.motion import grip_guard as g
    gripper, recheck = _gripper(embodiment)
    rng = random.Random(GUARD_SEED)

    def draw_guard():
        stage = {}
        if rng.random() < .7: stage["grip_guard_preset"] = rng.choice(GUARD_PRESETS)
        if rng.random() < .4: stage["min_aperture_m"] = rng.choice(GUARD_VALUES)
        if rng.random() < .3: stage["max_aperture_m"] = rng.choice(GUARD_VALUES)
        batch = {}
        if rng.random() < .3: batch["grip_guard_preset"] = rng.choice(GUARD_PRESETS)
        if rng.random() < .2: batch["min_aperture_m"] = rng.choice(GUARD_VALUES)
        return stage, batch
    for trial in range(GUARD_TRIALS):
        armed = {}
        for step in range(rng.randint(1, 6)):
            arm = rng.choice(["left", "right"])
            kind = rng.choice(["move", "gripper"])
            events = rng.choice([[], [], [{"goal_normalized": rng.choice([0.0, 0.3, 0.49, 0.5, 0.84, 0.85, 1.0])}], [{}]])
            stage, batch = draw_guard()
            goal = g.jaw_goal({"gripper_events": events})
            try:
                guard, error = g.stage_grip_guard(stage, batch, goal, **gripper), None
            except ValueError as raised:
                guard, error = None, str(raised)
            jaw_delta = rng.choice([None, 0.0, -0.0009, -0.001, -0.0011, -0.079, -0.08, -0.081, -0.5, 0.3])
            facts = {"jaw_delta": jaw_delta, "measured_aperture_m": rng.choice(GUARD_VALUES), "measured_jaw_before": 1,
                     "measured_jaw_after": 0, "aperture_delta_m": None}
            remaining, arm_remaining = rng.random() < .8, rng.random() < .8
            start_z, target_z = rng.choice([None, 0.1]), rng.choice([None, 0.09, 0.094, 0.095, 0.0951, 0.2])
            downward = start_z is not None and target_z is not None and (target_z - start_z) < g.DOWNWARD_MOVE_DZ_M
            evaluation, triggered = g.after_stage_grip_guard(
                armed, arm, kind, goal, guard, facts, has_remaining_stages=remaining, has_arm_remaining=arm_remaining,
                is_downward_move=downward, post_grasp_recheck=recheck, **gripper)
            yield dict(trial=trial, step=step, arm=arm, kind=kind, events=events, stage=stage, batch=batch,
                       facts=facts, has_remaining_stages=remaining, has_arm_remaining=arm_remaining,
                       start_z=start_z, target_z=target_z, guard=guard, error=error, evaluation=evaluation,
                       triggered=triggered, armed=deepcopy(armed))


def capture_guard(embodiment):
    """``stream``: case count, SHA-256 of the rendered case stream and of each ``GUARD_CHUNK`` chunk, outcome tallies;
    ``sample``: the first ``GUARD_SAMPLE`` cases in full; ``validate_grip_guard_fields`` / ``preset_aperture_bounds``
    / ``evaluate_grip_guard`` tables over every preset and boundary aperture."""
    from ni_rua.robot.motion import grip_guard as g
    gripper, _ = _gripper(embodiment)
    total, chunks, tallies, sample = hashlib.sha256(), [], {}, []
    chunk, count = hashlib.sha256(), 0
    for record in guard_cases(embodiment):
        line = (render(normalize(record)) + "\n").encode()
        total.update(line)
        chunk.update(line)
        count += 1
        if count <= GUARD_SAMPLE:
            sample.append(normalize(record))
        if count % GUARD_CHUNK == 0:
            chunks.append(chunk.hexdigest())
            chunk = hashlib.sha256()
        outcome = ("error:" + record["error"] if record["error"] else "guard" if record["guard"] else "no_guard",
                   "triggered:" + str((record["evaluation"] or {}).get("reason")) if record["triggered"]
                   else "evaluated" if record["evaluation"] is not None else "not_evaluated")
        key = " | ".join(outcome)
        tallies[key] = tallies.get(key, 0) + 1
    if count % GUARD_CHUNK:
        chunks.append(chunk.hexdigest())
    items = {"stream": dict(cases=count, sha256=total.hexdigest(), chunk_cases=GUARD_CHUNK, chunk_sha256=chunks,
                            tallies=dict(sorted(tallies.items()))),
             "sample": sample}
    # Fingertip gaps (m): below/at zero, the preset margins, the SO-101 empty close (+margin), X5 full open and just
    # above, SO-101 full open and just above.
    apertures = [None, -0.001, 0.0, 0.0004, 0.0005, 0.0015, 0.002, 0.003, 0.0035, 0.0065, 0.087, 0.088, 0.0881, 0.1,
                 0.136, 0.137]
    bounds = [None, -0.001, 0.0, 0.0015, 0.002, 0.003, 0.0065, 0.088, 0.0881, 0.137]
    validate = []
    for preset in [*GUARD_PRESETS, "bogus"]:
        for low in bounds:
            for high in bounds:
                mapping = {key: value for key, value in (("grip_guard_preset", preset), ("min_aperture_m", low),
                                                         ("max_aperture_m", high)) if value is not None}
                try:
                    outcome = g.validate_grip_guard_fields(mapping, **gripper)
                except ValueError as error:
                    outcome = str(error)
                validate.append([preset, low, high, outcome])
    items["validate_grip_guard_fields"] = validate
    items["preset_aperture_bounds"] = {str(preset): list(g.preset_aperture_bounds(
        preset, **({"empty_close_aperture_m": gripper["empty_close_aperture_m"]} if gripper else {})))
        for preset in GUARD_PRESETS}
    evaluate = []
    for preset in GUARD_PRESETS[1:]:
        for measured in apertures:
            facts = {"measured_aperture_m": measured}
            try:
                outcome = g.evaluate_grip_guard(facts, {"grip_guard_preset": preset}, **gripper)
            except ValueError as error:  # a preset this gripper does not offer (SO-101: the X5 legacy names)
                outcome = str(error)
            evaluate.append([preset, measured, outcome])
    items["evaluate_grip_guard"] = evaluate
    return normalize(items)


# ---- family: config ----------------------------------------------------------------------------------------------
def capture_config(embodiment):
    """``Config()`` (X5) and each shipped configuration as loaded (``load/<name>``), its ``motion_thresholds`` and the
    ``effective_config.json`` of each startup scenario; SO-101 adds the validated example deployment."""
    from ni_rua.core.config import Config, motion_thresholds
    items = {}
    redact = Redactor()
    if embodiment == "x5":
        items["Config()"] = Config().to_dict()
        items["motion_thresholds()"] = motion_thresholds()
    for name, layer in (X5_CONFIGS if embodiment == "x5" else SO101_CONFIGS).items():
        config = load_config(layer)
        items["load/" + name] = config.to_dict()
        items["motion_thresholds/" + name] = motion_thresholds(config)
    for scenario in _startup_table(embodiment):
        items["effective_config/" + scenario] = run_startup(embodiment, scenario)["effective_config"]
    if embodiment == "so101":
        from ni_rua.robot.backends.so101.deployment import validate_deployment
        document = json.loads((REPOSITORY / "configs" / "so101.example.json").read_text())
        items["deployment/so101.example.json"] = validate_deployment(document, base=REPOSITORY / "configs")
    return normalize(redact(normalize(items)))


CAPTURES = {"catalogue": capture_catalogue, "instructions": capture_instructions, "startup": capture_startup,
            "compile": capture_compile, "guard": capture_guard, "config": capture_config}


def capture(embodiment, family, *, recorded=None):
    """Current items of one family. ``recorded``: the decoded fixture items (compile replays their requests)."""
    with repository_cwd(), private_home():
        if family == "compile" and recorded is not None:
            return capture_compile(embodiment, corpus_from_items(recorded))
        return CAPTURES[family](embodiment)


# ---- the approved correction -------------------------------------------------------------------------------------
def correct_definition(definition):
    """``definition`` (a recorded robot_action tool definition) with ``APPROVED_CORRECTION`` applied at exactly its
    sites. Raises ``ValueError`` when the recorded definition does not hold the expected old strings and counts."""
    corrected = deepcopy(definition)
    for pointer, expected in APPROVED_CORRECTION["sites"].items():
        text = pointer_get(corrected, pointer)
        for old, count in expected.items():
            if text.count(old) != count:
                raise ValueError(f"approved_correction_site_mismatch:{pointer}:{old}:{text.count(old)}!={count}")
        for old, new in APPROVED_CORRECTION["replacements"]:
            text = text.replace(old, new)
        pointer_set(corrected, pointer, text)
    delta = len(render(corrected).encode()) - len(render(definition).encode())
    if delta != APPROVED_CORRECTION["byte_delta"]:
        raise ValueError(f"approved_correction_byte_delta:{delta}")
    return corrected


def correct_catalogue(definitions):
    """A recorded catalogue (list of tool definitions) with the approved correction applied to its robot_action."""
    return [correct_definition(item) if item.get("name") == APPROVED_CORRECTION["tool"] else deepcopy(item)
            for item in definitions]


def correction_applies(family, value):
    return (family == "catalogue" and isinstance(value, list)
            and any(isinstance(item, dict) and item.get("name") == APPROVED_CORRECTION["tool"] for item in value))


# ---- approved allowances (ALLOWANCES) ----------------------------------------------------------------------------
def check_allowance(allowance):
    """Structural check of one ALLOWANCES record: the declared count equals its edits and replacement occurrences;
    raises ``ValueError``."""
    count = sum(len(edits) for edits in allowance.get("edits", {}).values())
    count += sum(item[3] for items in allowance.get("replacements", {}).values() for item in items)
    if count != allowance["occurrences"]:
        raise ValueError(f"allowance_count_mismatch:{allowance['id']}:{count}!={allowance['occurrences']}")
    return allowance


def _parent(value, pointer):
    parts = pointer.strip("/").split("/")
    parent = pointer_get(value, "/" + "/".join(parts[:-1])) if len(parts) > 1 else value
    key = parts[-1]
    return parent, int(key) if isinstance(parent, list) else key


def _replace_text(allowance, name, expected, pointer, old, new, count):
    text = expected if pointer == "" else pointer_get(expected, pointer)
    if not isinstance(text, str) or text.count(old) != count:
        found = text.count(old) if isinstance(text, str) else None
        raise ValueError(f"allowance_replacement_mismatch:{allowance['id']}:{name}:{pointer}:{found}!={count}")
    text = text.replace(old, new)
    if pointer == "":
        return text
    pointer_set(expected, pointer, text)
    return expected


def apply_allowances(embodiment, family, name, value):
    """``(expected, ids)``: ``value`` (a recorded item) with every ALLOWANCES edit for this item applied, and the ids
    of the allowances used. Raises ``ValueError`` when a recorded old value is not exactly the declared one."""
    expected, ids = value, []
    for allowance in ALLOWANCES:
        if ((allowance["embodiment"], allowance["family"]) != (embodiment, family)
                or name not in {*allowance.get("edits", {}), *allowance.get("replacements", {})}):
            continue
        check_allowance(allowance)
        expected = deepcopy(expected) if not ids else expected
        for pointer, old, new, *position in allowance.get("edits", {}).get(name, ()):
            if pointer == "":
                if render(expected) != render(old):
                    raise ValueError(f"allowance_old_value_mismatch:{allowance['id']}:{name}:<root>")
                expected = ABSENT if new is ABSENT else deepcopy(new)
                continue
            parent, key = _parent(expected, pointer)
            present = key in parent if isinstance(parent, dict) else 0 <= key < len(parent)
            recorded = parent[key] if present else ABSENT
            if render(recorded) != render(old):
                raise ValueError(f"allowance_old_value_mismatch:{allowance['id']}:{name}:{pointer}")
            if new is ABSENT:
                del parent[key]
            elif isinstance(parent, list) and key == len(parent):
                parent.append(deepcopy(new))  # ABSENT -> value at the list end appends (edits apply in order)
            elif position and isinstance(parent, dict) and not present:
                # optional 4th element: the key the new key follows (key order is part of the rendered document)
                items = list(parent.items())
                at = [name_ for name_, _ in items].index(position[0]) + 1
                parent.clear()
                parent.update([*items[:at], (key, deepcopy(new)), *items[at:]])
            else:
                parent[key] = deepcopy(new)
        for pointer, old, new, count in allowance.get("replacements", {}).get(name, ()):
            expected = _replace_text(allowance, name, expected, pointer, old, new, count)
        ids.append(allowance["id"])
    return expected, ids


# ---- fixture files -----------------------------------------------------------------------------------------------
SCHEMA = "ni_rua.embodiment_golden.v1"
# Lossless file encodings (write_fixture re-reads every file and asserts byte identity of every item):
#   {"$lines": [...]}                               a text, one element per line;
#   {"$derive": base, "$edits": [[i, j, [...]], ...]}  an item (or catalogue pool definition) = the line view of the
#                                                   full item ``base`` with base lines i..j-1 replaced (ascending,
#                                                   non-overlapping). Text items use their text lines; JSON items use
#                                                   ``json_lines`` (one ``<path> = <leaf JSON>`` line per leaf, in
#                                                   document order, so key order survives).
DERIVE, EDITS = "$derive", "$edits"
DERIVE_WINDOW = 24  # candidate bases: the first and the last DERIVE_WINDOW full items of the same kind


def fixture_path(embodiment, family):
    return FIXTURE_ROOT / embodiment / f"{family}.json"


def _git(*arguments):
    return subprocess.run(["git", "-C", str(REPOSITORY), *arguments], check=True, capture_output=True,
                          text=True).stdout


def source_identity():
    """Commit of the checkout the captures import, and whether ``ni_rua/`` and ``configs/`` (the frozen production
    sources) differ from it. Untracked or modified test files (this builder) do not count."""
    revision = _git("rev-parse", "HEAD").strip()
    dirty = [line[3:] for line in _git("status", "--porcelain", "--", "ni_rua", "configs").splitlines() if line]
    return dict(revision=revision, production_tree_clean=not dirty, dirty_paths=dirty)


_SENTENCE = re.compile(r"(?<=\. )")


def json_lines(value, path=""):
    """One ``<path> = <leaf>`` line per leaf of a dict/list, in document order. Path segments: ``.<JSON string>`` for
    an object key, ``[<n>]`` for a list index; empty containers are leaves. A string leaf longer than 160 characters
    continues sentence by sentence on ``<path> += <JSON string>`` lines (concatenated on reading)."""
    if isinstance(value, dict) and value:
        return [line for key, item in value.items()
                for line in json_lines(item, path + "." + json.dumps(key, ensure_ascii=False))]
    if isinstance(value, list) and value:
        return [line for index, item in enumerate(value) for line in json_lines(item, f"{path}[{index}]")]
    if isinstance(value, str) and len(value) > 160:
        parts = [part for part in _SENTENCE.split(value) if part]
        return [path + " = " + render(parts[0])] + [path + " += " + render(part) for part in parts[1:]]
    return [path + " = " + render(value)]


def from_json_lines(lines):
    decoder, root = json.JSONDecoder(), None
    for line in lines:
        tokens, position = [], 0
        while line[position] != " ":
            if line[position] == ".":
                key, position = decoder.raw_decode(line, position + 1)
                tokens.append(key)
            else:
                end = line.index("]", position)
                tokens.append(int(line[position + 1:end]))
                position = end + 1
        if line.startswith(" += ", position):
            node = root
            for token in tokens[:-1]:
                node = node[token]
            node[tokens[-1] if isinstance(tokens[-1], str) else -1] += json.loads(line[position + 4:])
            continue
        leaf = json.loads(line[position + 3:])
        if root is None:
            root = {} if isinstance(tokens[0], str) else []
        node = root
        for token, following in zip(tokens, tokens[1:]):
            empty = {} if isinstance(following, str) else []
            if isinstance(token, str):
                node = node.setdefault(token, empty)
            else:
                if token == len(node):
                    node.append(empty)
                node = node[token]
        if isinstance(tokens[-1], str):
            node[tokens[-1]] = leaf
        else:
            node.append(leaf)
    return root


def _line_edits(base, lines):
    matcher = difflib.SequenceMatcher(None, base, lines, autojunk=False)
    return [[i1, i2, lines[j1:j2]] for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != "equal"]


def _apply_edits(base, edits):
    result, position = [], 0
    for start, end, lines in edits:
        result += base[position:start] + lines
        position = end
    return result + base[position:]


def _view(value):
    """``("text", lines)`` for a multi-line string, ``("json", json_lines)`` for a non-empty dict/list, else None."""
    if isinstance(value, str) and "\n" in value:
        return "text", value.split("\n")
    if isinstance(value, (dict, list)) and value:
        return "json", json_lines(value)
    return None


def _encode_mapping(mapping, same_kind=lambda a, b: True):
    """Encode each value; derive it from an earlier full value (``same_kind`` of both names) when the edits take
    less than half of its full encoding."""
    out, full = {}, {}
    for name, value in mapping.items():
        plain = encode(value)
        view = _view(value)
        best = None
        if view is not None:
            candidates = [base for base in full if same_kind(base, name) and full[base][0] == view[0]]
            for base in dict.fromkeys(candidates[:1] + candidates[-DERIVE_WINDOW:]):
                edits = _line_edits(full[base][1], view[1])
                size = len(render(edits))
                if best is None or size < best[0]:
                    best = (size, base, edits)
        if best is not None and best[0] * 2 < len(render(plain)):
            out[name] = {DERIVE: best[1], EDITS: best[2]}
        else:
            out[name] = plain
            if view is not None:
                full[name] = view
    return out


def _decode_mapping(encoded):
    out = {}
    for name, value in encoded.items():
        if isinstance(value, dict) and DERIVE in value:
            base = out[value[DERIVE]]
            kind, lines = _view(base)
            lines = _apply_edits(lines, value[EDITS])
            out[name] = "\n".join(lines) if kind == "text" else from_json_lines(lines)
        else:
            out[name] = decode(value)
    return out


def _pooled(embodiment, items):
    """Catalogue items as lists of pool keys ``<tool>#<n>`` (each distinct definition once; a variant of an earlier
    definition of the same tool is derived from it). SO-101 refers to an identical X5 definition as ``x5:<key>``."""
    shared = {}
    if embodiment != "x5" and fixture_path("x5", "catalogue").exists():
        _, x5_pool = _x5_pool()
        shared = {render(definition): "x5:" + key for key, definition in x5_pool.items()}
    pool, keys, refs = {}, {}, {}
    for name, definitions in items.items():
        refs[name] = []
        for definition in definitions:
            text = render(definition)
            if text not in shared and text not in keys:
                tool = definition["name"]
                keys[text] = f"{tool}#{sum(1 for key in pool if key.split('#')[0] == tool) + 1}"
                pool[keys[text]] = definition
            refs[name].append(shared.get(text) or keys[text])
    return refs, _encode_mapping(pool, lambda a, b: a.split("#")[0] == b.split("#")[0])


def _x5_pool():
    document = _read_document("x5", "catalogue")
    return document, _decode_mapping(document["pool"])


def _write_json(path, header, items, pool=None):
    """Valid JSON with one item per line (texts and edit lists one element per line) for readable fixture diffs."""
    lines = ["{"]
    for key, value in header.items():
        lines.append(f" {json.dumps(key)}: {render(value)},")

    def block(name, mapping, last):
        lines.append(f" {json.dumps(name)}: {{")
        entries = list(mapping.items())
        for index, (key, value) in enumerate(entries):
            comma = "," if index < len(entries) - 1 else ""
            label = json.dumps(key, ensure_ascii=False)
            if isinstance(value, dict) and (set(value) == {LINES_KEY} or EDITS in value):
                head = (f"{json.dumps(DERIVE)}: {render(value[DERIVE])}, {json.dumps(EDITS)}: [" if EDITS in value
                        else f"{json.dumps(LINES_KEY)}: [")
                elements = value.get(EDITS, value.get(LINES_KEY))
                lines.append(f"  {label}: {{{head}")
                lines.extend("   " + render(element) + ("," if i < len(elements) - 1 else "")
                             for i, element in enumerate(elements))
                lines.append("  ]}" + comma)
            else:
                lines.append(f"  {label}: {render(value)}{comma}")
        lines.append(" }" + ("" if last else ","))
    block("items", items, pool is None)
    if pool is not None:
        block("pool", pool, True)
    lines.append("}")
    text = "\n".join(lines) + "\n"
    json.loads(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_fixture(embodiment, family, *, force=False, allow_dirty=False, items=None):
    """Capture and write one fixture; refuses to overwrite without ``force`` and a dirty production tree without
    ``allow_dirty``. Re-reads the file and checks every item renders byte-identically. Returns the path."""
    path = fixture_path(embodiment, family)
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists; pass --force to overwrite a golden fixture")
    identity = source_identity()
    if not identity["production_tree_clean"] and not allow_dirty:
        raise RuntimeError("ni_rua/ or configs/ differ from HEAD (" + ", ".join(identity["dirty_paths"])
                           + "); commit first or pass --allow-dirty")
    if identity["production_tree_clean"]:
        identity.pop("dirty_paths")
    items = capture(embodiment, family) if items is None else items
    header = dict(schema=SCHEMA, embodiment=embodiment, family=family, label=LABELS[embodiment],
                  captured_from=identity,
                  regenerate=("PYTHONPATH=tests mamba run -n ni-rua python -m support.embodiment_golden --write "
                              f"--force --embodiment {embodiment} --family {family}"))
    if family == "catalogue":
        refs, pool = _pooled(embodiment, items)
        _write_json(path, header, refs, pool)
    else:
        _write_json(path, header, _encode_mapping(items))
    _read_document.cache_clear()
    _, reread = load_fixture(embodiment, family)
    if list(reread) != list(items) or any(render(reread[name]) != render(items[name]) for name in items):
        raise AssertionError(f"{path}: fixture encoding is not lossless")
    return path


@functools.lru_cache(maxsize=None)
def _read_document(embodiment, family):
    return json.loads(fixture_path(embodiment, family).read_text(encoding="utf-8"))


def load_fixture(embodiment, family):
    """``(header, items)`` of one fixture with every file encoding undone (texts, derived items, pool references)."""
    document = deepcopy(_read_document(embodiment, family))
    encoded = document.pop("items")
    pool = document.pop("pool", None)
    if pool is None:
        return document, _decode_mapping(encoded)
    resolved = _decode_mapping(pool)

    def definition(key):
        return deepcopy(_x5_pool()[1][key[3:]] if key.startswith("x5:") else resolved[key])
    return document, {name: [definition(key) for key in keys] for name, keys in encoded.items()}


# ---- host tolerance ----------------------------------------------------------------------------------------------
# The fixtures were recorded on RECORDING_HOST. There the comparison is byte-exact: host_noise tolerates nothing.
# On another host exactly two host effects are not behaviour changes; types, ints, bools, every other string, keys,
# key order and list lengths must still match byte for byte:
#
# * float rounding: the same arithmetic through another libm, BLAS or FMA contraction rounds the last bits differently
#   (recorded 0.30000000000000004 m on macOS, 0.3 on Linux). A float may differ by at most FLOAT_ULPS units in the last
#   place (ulp_distance): 4 ULP is a relative error of about 9e-16, far below any physical tolerance (1e-9 m) and above
#   the 1-2 ULP observed between the two hosts. -0.0 and 0.0 are 0 ULP apart.
# * host path resolution: Config makes every path field absolute with Path.resolve(), which follows the host's own
#   symlinks. RECORDED_ROOT_RESOLUTION is how the recording host resolved the roots a path can start from; only a root
#   that this host resolves differently (host_path_aliases) is respelled, only on the recorded side, toward this host's
#   spelling, and only at the Config path fields (config_path_pointers) of the "config" family.
#
# HOST_TOLERANCE pins, per known other host, the items it tolerates and how; any further tolerated item (or another
# description) fails there until it is reviewed and added. Every comparison lists its tolerated items (report()).
RECORDING_HOST = "Darwin arm64"
RECORDED_ROOT_RESOLUTION = {"/home": "/System/Volumes/Data/home", "/tmp": "/private/tmp", "/var": "/private/var",
                            "/etc": "/private/etc"}
FLOAT_ULPS = 4
_PATH_NOTE = "1 path(s) in another host spelling"
HOST_TOLERANCE = {
    "Linux x86_64": {
        "compile": {"022 stage/move/angles_deg=[-180.0,0.0,0.0]": "2 float(s) within 1 ULP",
                    "030 stage/move/angles_deg=[0.0,0.0,-180.0]": "1 float(s) within 1 ULP"},
        "config": {name: _PATH_NOTE for name in (
            "load/ni_rua", "load/rgbd_feedback_v2", "load/rgbd_numeric_v1", "load/rgbd_skills_v3",
            "load/benchmark_shared_memory", "load/ni_rua_example", "effective_config/ni_rua_benchmark",
            "effective_config/ni_rua_interactive", "effective_config/rgbd_feedback_v2",
            "effective_config/rgbd_numeric_v1", "effective_config/rgbd_skills_v3",
            "effective_config/legacy_rgb_proprio_v1", "effective_config/prompt_capture_fixture")},
    },
}


def current_host():
    """``"<system> <machine>"`` of this interpreter, e.g. ``"Darwin arm64"`` or ``"Linux x86_64"``."""
    import platform
    return f"{platform.system()} {platform.machine()}"


def ulp_distance(a, b):
    """Units in the last place between two finite floats (0.0 and -0.0 are the same point)."""
    def ordinal(value):
        bits = struct.unpack("<q", struct.pack("<d", value))[0]
        return bits if bits >= 0 else -(bits & 0x7FFFFFFFFFFFFFFF)
    return abs(ordinal(a) - ordinal(b))


def host_path_aliases(resolve=os.path.realpath):
    """``((recorded prefix, this host's prefix), ...)`` for every RECORDED_ROOT_RESOLUTION root this host resolves
    differently from the recording host (none on the recording host)."""
    return tuple((recorded, resolve(root)) for root, recorded in RECORDED_ROOT_RESOLUTION.items()
                 if resolve(root) != recorded)


def host_path_spelling(text, aliases):
    """A recorded path ``text`` respelled for this host: a leading recorded prefix of ``aliases`` is replaced by this
    host's prefix (whole path components only)."""
    for recorded, here in aliases:
        if text == recorded or text.startswith(recorded + "/"):
            return here + text[len(recorded):]
    return text


@functools.lru_cache(maxsize=None)
def config_path_pointers():
    """JSON-pointer patterns (``*`` = any list index) of every ``Config`` field declared ``option(path=True)``, as
    they appear in ``Config.to_dict()`` (the ``load/*`` and ``effective_config/*`` items)."""
    import dataclasses
    import typing
    from ni_rua.core.config import Config
    pointers = set()

    def walk(kind, prefix):
        hints = typing.get_type_hints(kind)
        for item in dataclasses.fields(kind):
            pointer = f"{prefix}/{item.name}"
            if item.metadata.get("path"):
                pointers.add(pointer)
                continue
            for candidate in (hints[item.name], *typing.get_args(hints[item.name])):
                if dataclasses.is_dataclass(candidate):
                    walk(candidate, pointer)
                elif typing.get_origin(candidate) is tuple:
                    for inner in typing.get_args(candidate):
                        if dataclasses.is_dataclass(inner):
                            walk(inner, pointer + "/*")
    walk(Config, "")
    return frozenset(pointers)


def _pointer_matches(pointer, patterns):
    parts = pointer.split("/")
    return any(len(parts) == len(pattern) and all(p == "*" or p == q for p, q in zip(pattern, parts))
               for pattern in (item.split("/") for item in patterns))


def host_noise(expected, current, *, host=None, path_pointers=frozenset(), aliases=None, strings=None):
    """``{"floats": n, "max_ulps": m, "paths": k, "references": r}`` when ``current`` differs from ``expected`` (plain
    JSON data, ``expected`` recorded on RECORDING_HOST) only by the host effects above, else None. Equal values give
    all-zero counts; on the recording host any difference gives None.

    host: the comparing host (default ``current_host()``). path_pointers: patterns where a recorded path may be
    respelled (default none). aliases: ``host_path_aliases()`` by default. strings: optional
    ``callable(recorded, current) -> bool`` deciding other differing strings (counted as ``references``; the evidence
    corpus passes its artifact-reference rule)."""
    counts = {"floats": 0, "max_ulps": 0, "paths": 0, "references": 0}
    if render(expected) == render(current):
        return counts
    if (current_host() if host is None else host) == RECORDING_HOST:
        return None
    aliases = host_path_aliases() if aliases is None else aliases

    def same(left, right, pointer):
        if type(left) is not type(right):
            return False
        if isinstance(left, dict):
            return list(left) == list(right) and all(same(left[key], right[key], f"{pointer}/{key}") for key in left)
        if isinstance(left, list):
            return len(left) == len(right) and all(same(a, b, f"{pointer}/{index}")
                                                   for index, (a, b) in enumerate(zip(left, right)))
        if isinstance(left, float):
            if repr(left) == repr(right):
                return True
            distance = ulp_distance(left, right)
            counts["floats"] += 1
            counts["max_ulps"] = max(counts["max_ulps"], distance)
            return distance <= FLOAT_ULPS
        if isinstance(left, str) and left != right:
            if path_pointers and _pointer_matches(pointer, path_pointers) and \
                    host_path_spelling(left, aliases) == right:
                counts["paths"] += 1
                return True
            if strings is not None and strings(left, right):
                counts["references"] += 1
                return True
            return False
        return left == right
    return counts if same(expected, current, "") else None


def describe_noise(noise):
    parts = []
    if noise["floats"]:
        parts.append(f"{noise['floats']} float(s) within {noise['max_ulps']} ULP")
    if noise["paths"]:
        parts.append(f"{noise['paths']} path(s) in another host spelling")
    if noise.get("references"):
        parts.append(f"{noise['references']} artifact reference(s) to host-equivalent content")
    return ", ".join(parts)


def unexpected_host_tolerance(family, tolerated, host=None):
    """``{item: description}`` of the tolerated items (``compare()["host"]``) that HOST_TOLERANCE does not pin for
    ``host`` (default ``current_host()``) with exactly that description. Empty on a host HOST_TOLERANCE does not
    know (its tolerated items are still listed by report())."""
    host = current_host() if host is None else host
    if host == RECORDING_HOST:
        return dict(tolerated)
    if host not in HOST_TOLERANCE:
        return {}
    pinned = HOST_TOLERANCE[host].get(family, {})
    return {name: text for name, text in tolerated.items() if pinned.get(name) != text}


# ---- comparison --------------------------------------------------------------------------------------------------
def compare(embodiment, family, current=None, *, allow_correction=None, allow_allowances=True, host=None,
            aliases=None):
    """Compare current items with the fixture. Returns ``{"states": {item: state}, "diffs": {item: text},
    "allowed": {item: [allowance ids]}}`` with state ``equal`` (equal to the recorded item, or to the recorded item
    with its approved ``ALLOWANCES`` edits applied, then listed in ``allowed``) | ``corrected`` (equal to that
    expectation after ``APPROVED_CORRECTION``) | ``changed`` | ``missing`` (recorded, not produced) | ``added``
    (produced, not recorded). ``allow_correction`` defaults to True for X5 catalogue items; ``allow_allowances``
    (default True) applies ``ALLOWANCES``. Off the recording host, ``equal`` and ``corrected`` also hold when the
    only differences are host effects (``host_noise``; ``host`` / ``aliases`` default to this host); those items are
    listed in ``"host"`` ({item: description}) together with the comparing host (``"compared_on"``)."""
    header, recorded = load_fixture(embodiment, family)
    current = capture(embodiment, family, recorded=recorded) if current is None else current
    allow_correction = (embodiment == "x5") if allow_correction is None else allow_correction
    states, diffs, allowed, tolerated = {}, {}, {}, {}
    host = current_host() if host is None else host
    noise_options = dict(host=host, aliases=aliases,
                         path_pointers=config_path_pointers() if family == "config" else frozenset())
    for name in [*recorded, *(name for name in current if name not in recorded)]:
        if name not in current:
            if name in recorded and allow_allowances:
                expected, ids = apply_allowances(embodiment, family, name, recorded[name])
                if expected is ABSENT:
                    states[name] = "equal"
                    if ids:
                        allowed[name] = ids
                    continue
            states[name] = "missing"
            continue
        if name not in recorded:
            states[name] = "added"
            continue
        before, after = recorded[name], current[name]
        expected, ids = apply_allowances(embodiment, family, name, before) if allow_allowances else (before, [])
        correctable = allow_correction and correction_applies(family, expected)
        if render(expected) == render(after):
            states[name] = "equal"
        elif correctable and render(corrected := correct_catalogue(expected)) == render(after):
            states[name] = "corrected"
        elif (noise := host_noise(expected, after, **noise_options)) is not None:
            states[name], tolerated[name] = "equal", describe_noise(noise)
        elif correctable and (noise := host_noise(corrected, after, **noise_options)) is not None:
            states[name], tolerated[name] = "corrected", describe_noise(noise)
        else:
            states[name] = "changed"
            prefix = f"expectation = recorded + allowances {', '.join(ids)}\n" if ids else ""
            diffs[name] = prefix + describe_difference(expected, after)
        if ids and states[name] != "changed":
            allowed[name] = ids
    return dict(header=header, states=states, diffs=diffs, allowed=allowed, host=tolerated, compared_on=host)


def _pretty(value):
    if isinstance(value, str):
        return value.splitlines(keepends=False)
    return json.dumps(value, ensure_ascii=False, indent=1).splitlines()


def describe_difference(before, after, *, limit=60):
    """Readable difference: per tool for catalogues, a unified line diff otherwise (at most ``limit`` lines)."""
    if (isinstance(before, list) and isinstance(after, list) and before and after
            and all(isinstance(item, dict) and "name" in item for item in before + after)):
        out = []
        old, new = {item["name"]: item for item in before}, {item["name"]: item for item in after}
        if list(old) != list(new):
            out.append(f"tool order/set: {list(old)} -> {list(new)}")
        for name in old:
            if name in new and render(old[name]) != render(new[name]):
                out.append(f"tool {name}:")
                out += _text_diff(_flatten(old[name]), _flatten(new[name]), limit)
        return "\n".join(out[:limit * 4])
    return "\n".join(_text_diff(_pretty(before), _pretty(after), limit))


def _flatten(value, path=""):
    """``path = <json>`` lines of every leaf, so a long description diff names its JSON path."""
    if isinstance(value, dict):
        return [line for key, item in value.items() for line in _flatten(item, f"{path}/{key}")] or [f"{path} = {{}}"]
    if isinstance(value, list):
        return [line for index, item in enumerate(value) for line in _flatten(item, f"{path}/{index}")] or [f"{path} = []"]
    return [f"{path} = {json.dumps(value, ensure_ascii=False)}"]


def _text_diff(before, after, limit):
    lines = list(difflib.unified_diff(before, after, "recorded", "current", n=1, lineterm=""))
    return lines[:limit] + ([f"... {len(lines) - limit} more diff lines"] if len(lines) > limit else [])


def report(result):
    counts = {}
    for state in result["states"].values():
        counts[state] = counts.get(state, 0) + 1
    allowed = result.get("allowed", {})
    lines = [f"{result['header']['embodiment']}/{result['header']['family']}: "
             + ", ".join(f"{state}={count}" for state, count in sorted(counts.items()))
             + (f" (of which {len(allowed)} by approved allowances)" if allowed else "")]
    for name, state in result["states"].items():
        if name in allowed:
            lines.append(f"  {state} by allowance: {name} ({', '.join(allowed[name])})")
        if name in result.get("host", {}):
            lines.append(f"  {state} within host tolerance: {name} ({result['host'][name]})")
        if state != "equal":
            lines.append(f"  {state}: {name}")
            if name in result["diffs"]:
                lines += ["    " + line for line in result["diffs"][name].splitlines()]
    return "\n".join(lines)


def host_report(result):
    """Where one comparison ran and every item it accepted within host tolerance (printed on every test run)."""
    tolerated, host = result.get("host", {}), result.get("compared_on") or current_host()
    exact = " (the recording host: byte-exact)" if host == RECORDING_HOST else ""
    lines = [f"{result['header']['embodiment']}/{result['header']['family']} compared on {host}{exact}: "
             f"{len(tolerated)} item(s) within host tolerance"]
    return "\n".join(lines + [f"  {name}: {text}" for name, text in tolerated.items()])


# ---- command line ------------------------------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true", help="capture and write fixtures")
    action.add_argument("--diff", action="store_true", help="compare the current tree with the fixtures")
    parser.add_argument("--embodiment", action="append", choices=EMBODIMENTS, help="default: all")
    parser.add_argument("--family", action="append", choices=FAMILIES, help="default: all")
    parser.add_argument("--force", action="store_true", help="overwrite existing fixtures (--write)")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="write although ni_rua/ or configs/ differ from HEAD (recorded in captured_from)")
    arguments = parser.parse_args(argv)
    embodiments = arguments.embodiment or list(EMBODIMENTS)
    families = arguments.family or list(FAMILIES)
    status = 0
    for embodiment in embodiments:
        for family in families:
            if arguments.write:
                path = write_fixture(embodiment, family, force=arguments.force, allow_dirty=arguments.allow_dirty)
                print(f"wrote {path.relative_to(REPOSITORY)} ({path.stat().st_size} bytes)")
            else:
                result = compare(embodiment, family)
                print(report(result))
                if any(state not in ("equal", "corrected") for state in result["states"].values()):
                    status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
