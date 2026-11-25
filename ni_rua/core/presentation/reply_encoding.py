"""Compact encoding of the JSON text items of a presented reply (``features.reply_compact_encoding``).

A pure function of one reply's content items; program callers and stored evidence never reach it. Four rewrites, in
this order, on every canonical JSON text item (images and other text pass through):

1. Derived quantities are printed with ``DERIVED_SIGNIFICANT_DIGITS`` significant digits. Only the fields of
   ``DERIVED_QUANTITIES`` are touched: finite differences of boundary measurements. Every other number (joint angles,
   poses, residuals, tolerances, apertures, measured joint velocity ``qdot_rad_s``) keeps its printed precision. A
   field that is not in the table is a measurement.

   Why 4 digits: the differences are taken over one recorded interval (``interval_s``, 0.04 s in the simulator) of
   float32 measurements, whose spacing near 1 (m or rad) is 2**-23 = 1.2e-7. A velocity therefore carries an error of
   about 1.2e-7 * sqrt(2) / 0.04 = 4e-6 (relative 4e-5 at 0.1 m/s) and an acceleration, a second difference, about
   2 * 1.2e-7 / 0.04**2 = 1.5e-4 (relative 3e-4 at 0.5 m/s^2). Rounding to 4 significant digits changes a value by
   at most 5e-4 of itself, the size of that noise; more digits print noise, fewer would cut into the signal.

2. An arm that did not move is stated once. In an object with ``samples`` (a list of objects with ``arms``), an arm
   is unchanged when in every sample
   - each number of its entry outside ``derived_motion`` differs from the same number of the first sample by at most
     ``STATIONARY_RESOLUTION * max(1, |value|)`` (one float32 step: joint angles, joint velocities, pose, jaw goal
     and jaw command), and every string, key and list length is identical, and
   - each ``DERIVED_QUANTITIES`` value of its ``derived_motion`` is at most what a change of one such step over
     that sample's ``interval_s`` gives (velocity: step / interval, acceleration: 2 * step / interval**2), so a jaw
     that moves while the joints rest keeps the arm in the samples.
   The arm's first-sample entry moves to ``unchanged_arms[arm]`` next to ``samples`` with ``"unchanged": true`` and
   ``max_abs_joint_change_rad`` (the largest joint-angle difference to the first sample that was folded away), and
   the arm leaves every sample. Its later ``derived_motion`` entries, zero to within the bound above, are not
   printed.

3. The structures of ``TABLE_STRUCTURES``, and no others, become a table ``{"columns": [...], "rows": [[...],
   ...]}``, the form the object table already uses. Nested objects are flattened into dotted column names
   (``arms.left.q_rad``); lists, empty objects and tables are cells. A listed structure is converted when it has at
   least two entries, every entry is an object, no key contains a dot, no column holds a real ``null`` in one row
   while the field is absent in another, and the table prints shorter than the original. Otherwise it stays as it is.
   - ``optional``: column-name prefixes whose fields some rows do not have; a ``null`` cell in such a column means
     "this row has no such field" (the first measured sample has no ``derived_motion``).
   - ``common``: a string column whose value is the same in every row that has it is printed once here instead of
     in each row (``source_kind``, ``tcp_config_id``, ``phase`` ...). Below an ``optional`` prefix it applies to
     the rows that have another field of that prefix. Strings are lifted only inside a table.
   - ``keyed_by``: the table stands for a map; that column holds the former keys (``effective_tolerances`` and
     ``stage_costs``, keyed by stage id).
   - ``first``: the first entry, printed whole before the rows, when it has another shape than the rest
     (``planned.samples`` starts with the measured start state; the rows are the commands).

4. Nothing else: keys, order of rows, ids and every value outside rule 1 and rule 2 are unchanged. Every other list
   of objects (task plan ledger and attention, advisory notes, policy definitions, motion-safety rows, cameras,
   stage entries, failure candidates, batch results, history entries, ledger outcomes) keeps its JSON form.

A text item that already contains an object shaped like such a table, or a ``samples`` object that already has
``unchanged_arms``, is returned unchanged, so the encoding is always reversible (the decoder used by the tests is in
``tests/core/presentation/test_actor_payload.py``).
"""
from __future__ import annotations

import json

from ..contracts import canonical

# Field name -> kind, valid inside a ``derived_motion`` object only (runtime/actions/trajectory_feedback.py computes
# all of them as finite differences at measurement boundaries). The single source of what counts as derived.
DERIVED_PARENT = "derived_motion"
DERIVED_QUANTITIES = {
    "joint_velocity_rad_s": "velocity",
    "ee_linear_velocity_m_s": "velocity",
    "ee_angular_velocity_rad_s": "velocity",
    "jaw_velocity_m_s": "velocity",
    "joint_acceleration_rad_s2": "acceleration",
    "ee_linear_acceleration_m_s2": "acceleration",
    "ee_angular_acceleration_rad_s2": "acceleration",
    "jaw_acceleration_m_s2": "acceleration",
}
DERIVED_SIGNIFICANT_DIGITS = 4
STATIONARY_RESOLUTION = 2.0 ** -23  # spacing of float32 values in [1, 2): one step of a stored measurement

TABLE_KEYS = frozenset({"columns", "rows", "optional", "common", "keyed_by", "first"})
UNCHANGED_ARMS_KEY = "unchanged_arms"
KEY_COLUMN = "id"

# The structures printed as tables: (trailing keys of the path from the root of the text item, list indices left
# out) -> (kind, why it is listed). The path decides, not the tool: the same structure in a program's action
# summary, a wait reply or a history entry is encoded the same way. Chosen from a census of all tool replies of two
# baseline arms (94 episodes): mostly numbers, regular shape (cell fill >= 0.85), no long free text, saving >= ~25 %.
# A structure that is not listed here is never converted.
#   kind "list": a list of objects; "list_after_first": the same with a first entry of another shape kept whole
#   under "first"; "map": an object whose values are objects, converted with a ``KEY_COLUMN`` column for the keys.
TABLE_STRUCTURES = {
    ("trajectory_feedback", "measured", "samples"): ("list", "largest block of an action reply; per-arm vectors, one row per sample"),
    ("trajectory_feedback", "planned", "samples"): ("list_after_first", "command rows of one shape after the measured start state"),
    ("compact_3d_trajectory", "stages"): ("list", "one row per stage, numbers and short ids"),
    ("process_trajectory", "event_keyframes"): ("list", "one row per keyframe, same fields in every row"),
    ("collision_model_witnesses",): ("list", "sphere pairs of a refused plan, same numeric fields in every row"),
    ("periodic_checks",): ("list", "one row per joint, same fields"),
    ("trajectory_evidence", "key_poses"): ("list", "per-arm poses of a preview, same fields"),
    ("object_state_delta", "moved"): ("list", "one row per object, numbers and ids"),
    ("object_state_delta", "appeared"): ("list", "one row per object, numbers and ids"),
    ("object_state_delta", "vacated"): ("list", "one row per object, numbers and ids"),
    ("effective_tolerances",): ("map", "stage id -> the same tolerance fields"),
    ("stage_costs",): ("map", "stage id -> the same cost fields"),
}

COMPACT_ENCODING_CATALOGUE_NOTE = (
    ' Compact reply encoding: trajectory_feedback measured/planned samples, compact_3d_trajectory.stages, '
    "process_trajectory.event_keyframes, object_state_delta moved/appeared/vacated, collision_model_witnesses, "
    "periodic_checks, trajectory_evidence.key_poses and the per-stage maps effective_tolerances and stage_costs are "
    'printed as a table {"columns", "rows"}: row i gives the values of the columns in order, a dotted column name '
    'is a nested field (arms.<arm>.q_rad), "common" holds fields that have the same value in every row, a null cell '
    'in a column under an "optional" prefix means that row has no such field, "keyed_by" names the column that '
    'holds the former map keys (stage ids), and "first" is a first entry of another shape given whole before the '
    "rows. unchanged_arms[arm] next to samples is the entry of an arm whose joints, pose and "
    "jaw stayed within one float32 step in every sample (max_abs_joint_change_rad is the largest difference); it "
    "applies to every sample. Fields inside derived_motion ("
    + ", ".join(DERIVED_QUANTITIES) + ") are finite differences of boundary measurements and are printed with "
    + str(DERIVED_SIGNIFICANT_DIGITS) + " significant digits; all other numbers keep their full precision.")


def _is_number(value):
    return type(value) in (int, float)


def _is_table(value):
    return (isinstance(value, dict) and "columns" in value and "rows" in value and set(value) <= TABLE_KEYS
            and isinstance(value["columns"], list) and isinstance(value["rows"], list))


def _already_encoded(value):
    if isinstance(value, dict):
        if _is_table(value) or (UNCHANGED_ARMS_KEY in value and "samples" in value):
            return True
        return any(_already_encoded(item) for item in value.values())
    if isinstance(value, list):
        return any(_already_encoded(item) for item in value)
    return False


def _round_significant(value):
    if isinstance(value, list):
        return [_round_significant(item) for item in value]
    if type(value) is float and value == value and value not in (float("inf"), float("-inf")):
        return float(f"{value:.{DERIVED_SIGNIFICANT_DIGITS}g}")
    return value


def _round_derived(node, inside=False):
    if isinstance(node, list):
        return [_round_derived(item) for item in node]
    if not isinstance(node, dict):
        return node
    return {key: (_round_significant(item) if inside and key in DERIVED_QUANTITIES
                  else _round_derived(item, key == DERIVED_PARENT and isinstance(item, dict)))
            for key, item in node.items()}


def _within_step(value, reference):
    """Same shape, strings identical, numbers within one float32 step of ``reference``; the largest difference."""
    if _is_number(value) and _is_number(reference):
        difference = abs(value - reference)
        return difference if difference <= STATIONARY_RESOLUTION * max(1.0, abs(reference)) else None
    if isinstance(value, dict) and isinstance(reference, dict):
        if list(value) != list(reference):
            return None
        pairs = [(value[key], reference[key]) for key in value]
    elif isinstance(value, list) and isinstance(reference, list):
        if len(value) != len(reference):
            return None
        pairs = list(zip(value, reference))
    else:
        return 0.0 if type(value) is type(reference) and value == reference else None
    largest = 0.0
    for item, other in pairs:
        difference = _within_step(item, other)
        if difference is None:
            return None
        largest = max(largest, difference)
    return largest


def _derived_at_rest(derived):
    if not isinstance(derived, dict):
        return False
    interval = derived.get("interval_s")
    if not _is_number(interval) or not interval > 0:
        return False
    bounds = dict(velocity=STATIONARY_RESOLUTION / interval, acceleration=2 * STATIONARY_RESOLUTION / interval ** 2)
    for key, kind in DERIVED_QUANTITIES.items():
        values = derived.get(key, [])
        values = values if isinstance(values, list) else [values]
        if not all(_is_number(item) and abs(item) <= bounds[kind] for item in values):
            return False
    return True


def _unchanged_arm(samples, arm):
    """``max_abs_joint_change_rad`` when ``arm`` is unchanged over ``samples`` (module docstring, rule 2), else None."""
    entries = [sample["arms"].get(arm) for sample in samples]
    if not all(isinstance(entry, dict) for entry in entries) or not isinstance(entries[0].get("q_rad"), list):
        return None
    first = {key: item for key, item in entries[0].items() if key != DERIVED_PARENT}
    largest = 0.0
    for entry in entries:
        if DERIVED_PARENT in entry and not _derived_at_rest(entry[DERIVED_PARENT]):
            return None
        if _within_step({key: item for key, item in entry.items() if key != DERIVED_PARENT}, first) is None:
            return None
        difference = _within_step(entry.get("q_rad"), first["q_rad"])
        largest = max(largest, difference)
    return largest


def _fold_unchanged_arms(node):
    if isinstance(node, list):
        return [_fold_unchanged_arms(item) for item in node]
    if not isinstance(node, dict):
        return node
    node = {key: _fold_unchanged_arms(item) for key, item in node.items()}
    samples = node.get("samples")
    if (not isinstance(samples, list) or len(samples) < 2
            or not all(isinstance(sample, dict) and isinstance(sample.get("arms"), dict) for sample in samples)):
        return node
    folded = {}
    for arm in samples[0]["arms"]:
        change = _unchanged_arm(samples, arm)
        if change is not None:
            folded[arm] = dict(samples[0]["arms"][arm], unchanged=True, max_abs_joint_change_rad=change)
    if not folded:
        return node
    node["samples"] = [dict(sample, arms={arm: entry for arm, entry in sample["arms"].items() if arm not in folded})
                       for sample in samples]
    node[UNCHANGED_ARMS_KEY] = folded
    return node


def _flatten(element, prefix, out):
    """Dotted leaf paths of one list element into ``out``; False when a key cannot be a column name."""
    for key, item in element.items():
        if not isinstance(key, str) or "." in key or not key:
            return False
        path = prefix + key
        if isinstance(item, dict) and item and not _is_table(item):
            if not _flatten(item, path + ".", out):
                return False
        else:
            out[path] = item
    return True


def _optional_prefixes(columns, optional):
    """Shortest dotted prefixes that cover exactly the ``optional`` columns."""
    prefixes = []
    for column in columns:
        if column not in optional or any(column == prefix or column.startswith(prefix + ".") for prefix in prefixes):
            continue
        parts = column.split(".")
        for size in range(1, len(parts) + 1):
            prefix = ".".join(parts[:size])
            if all(other in optional for other in columns if other == prefix or other.startswith(prefix + ".")):
                prefixes.append(prefix)
                break
    return prefixes


def _table(elements):
    """The table of a list of objects, or None when the list is not same-shaped (module docstring, rule 3)."""
    if len(elements) < 2 or not all(isinstance(element, dict) for element in elements):
        return None
    rows = []
    for element in elements:
        flat = {}
        if not _flatten(element, "", flat):
            return None
        rows.append(flat)
    columns = sorted({column for row in rows for column in row})
    if not columns:
        return None
    optional = {column for column in columns if any(column not in row for row in rows)}
    if any(row.get(column, 0) is None for column in optional for row in rows):
        return None  # a real null next to an absent field: a null cell would be ambiguous
    prefixes = _optional_prefixes(columns, optional)
    owner = {column: next((prefix for prefix in prefixes if column == prefix or column.startswith(prefix + ".")), None)
             for column in columns}
    common = {}
    for column in columns:
        values = [row[column] for row in rows if column in row]
        if isinstance(values[0], str) and all(type(value) is str and value == values[0] for value in values):
            common[column] = values[0]
    # Below an optional prefix a lifted string must be recoverable: the rows that have it are exactly the rows that
    # have another, printed field of that prefix.
    for column in [column for column in common if owner[column] is not None]:
        printed = [other for other in columns if owner[other] == owner[column] and other not in common]
        if not printed or any((column in row) != any(other in row for other in printed) for row in rows):
            del common[column]
    kept = [column for column in columns if column not in common]
    table = dict(columns=kept, rows=[[row.get(column) for column in kept] for row in rows])
    if common:
        table["common"] = common
    if prefixes:
        table["optional"] = prefixes
    return table


def _structure(path):
    """``TABLE_STRUCTURES`` kind of the value at ``path`` (keys from the root, list indices left out), or None."""
    for suffix, (kind, _) in TABLE_STRUCTURES.items():
        if path[-len(suffix):] == suffix:
            return kind
    return None


def _shortest(original, candidates):
    """The shortest-printing candidate that is shorter than ``original`` (first wins a tie), else ``original``."""
    best, size = original, len(canonical(original))
    for candidate in candidates:
        if candidate is not None and len(canonical(candidate)) < size:
            best, size = candidate, len(canonical(candidate))
    return best


def _tabulate(node, path=()):
    if isinstance(node, dict):
        node = {key: _tabulate(item, path + (key,)) for key, item in node.items()}
        if (_structure(path) == "map" and len(node) >= 2
                and all(isinstance(item, dict) and KEY_COLUMN not in item for item in node.values())):
            table = _table([{KEY_COLUMN: key, **item} for key, item in node.items()])
            if table is not None and KEY_COLUMN in table["columns"]:
                return _shortest(node, [dict(table, keyed_by=KEY_COLUMN)])
        return node
    if not isinstance(node, list):
        return node
    node = [_tabulate(item, path) for item in node]
    kind = _structure(path)
    if kind == "list":
        return _shortest(node, [_table(node)])
    if kind == "list_after_first":
        rest = _table(node[1:]) if len(node) >= 3 and isinstance(node[0], dict) else None
        return _shortest(node, [_table(node), None if rest is None else dict(rest, first=node[0])])
    return node


def encode_compact_value(value):
    """One parsed JSON text item -> its compact encoding (module docstring). Pure and deterministic."""
    if _already_encoded(value):
        return value
    return _tabulate(_fold_unchanged_arms(_round_derived(value)))


def encode_compact_items(items):
    """Content items of one reply with every canonical JSON text item in compact encoding; other items unchanged."""
    result = []
    for item in items:
        if isinstance(item, dict) and item.get("type") in ("inputText", "input_text") and isinstance(item.get("text"), str):
            try:
                value = json.loads(item["text"])
            except ValueError:
                value = None
            if isinstance(value, (dict, list)) and canonical(value) == item["text"]:
                text = canonical(encode_compact_value(value))
                if text != item["text"]:
                    item = dict(item, text=text)
        result.append(item)
    return result
