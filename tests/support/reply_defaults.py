"""The actor-facing configuration before the reply-delivery and catalogue defaults changed, for tests that pin that form.

``LEGACY_REPLY_FEATURES``: every reply-delivery switch off, the ``lean_v1`` payload profile, the ``full_v1`` prose tool
catalogue and the URDF inline in the startup instructions. A run configured with these ``features`` delivers
instructions, catalogue and replies in the earlier form byte for byte
(``robot.embodiment.test_embodiment_contracts.ReplyDeliveryDefaultsTests``). ``DEFAULT_REPLY_FEATURES`` is the shipped
default of the same settings (``layered_v3`` catalogue, URDF as the file ``robot.urdf``).
"""
import dataclasses
import json
import struct
from copy import deepcopy

from ni_rua.core.presentation import reply_encoding
from ni_rua.core.presentation.reply_compaction import SAME_AS_KEY, SAME_AS_PREFIX
from ni_rua.core.presentation.reply_encoding import DERIVED_PARENT, DERIVED_QUANTITIES, STATIONARY_RESOLUTION, UNCHANGED_ARMS_KEY

LEGACY_REPLY_FEATURES = dict(
    actor_payload_profile="lean_v1", budgeted_action_reply=False, reply_same_as=False, reply_float32=False,
    reply_static_once=False, action_reply_profile="full", reply_compact_encoding=False,
    tool_catalog_profile="full_v1", robot_urdf_delivery="inline")
DEFAULT_REPLY_FEATURES = dict(
    actor_payload_profile="lean_v2", budgeted_action_reply=True, reply_same_as=True, reply_float32=True,
    reply_static_once=True, action_reply_profile="summary_first", reply_compact_encoding=True,
    tool_catalog_profile="layered_v3", robot_urdf_delivery="file")


def with_reply_features(config, **features):
    """``config`` with the legacy reply form, then ``features`` on top of it."""
    return dataclasses.replace(config, features=dataclasses.replace(
        config.features, **{**LEGACY_REPLY_FEATURES, **features}))


# ---- reading a delivered reply back (test-side inverses of the reply shaping) -----------------------------------
def float32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def resolve_same_as(root):
    """Expand every same_as pointer of a compacted reply against the reply itself (RFC 6901, document order)."""
    def lookup(pointer):
        node = root
        for token in pointer.split("/")[1:]:
            token = token.replace("~1", "/").replace("~0", "~")
            node = node[int(token)] if isinstance(node, list) else node[token]
        return expand(node)

    def expand(node):
        if isinstance(node, dict):
            if set(node) == {SAME_AS_KEY}:
                return lookup(node[SAME_AS_KEY])
            return {key: expand(value) for key, value in node.items()}
        if isinstance(node, list):
            return [expand(item) for item in node]
        if isinstance(node, str) and node.startswith(SAME_AS_PREFIX):
            return lookup(node[len(SAME_AS_PREFIX):])
        return node

    return expand(root)


def decode_compact_value(node):
    """Inverse of ``reply_encoding.encode_compact_value`` (test-side): tables back to lists of objects and
    ``unchanged_arms`` back into every sample. A restored arm entry keeps ``unchanged`` / ``max_abs_joint_change_rad``
    so that ``compact_encoding_loss`` can tell it from an entry that was printed per sample."""
    if isinstance(node, list):
        return [decode_compact_value(item) for item in node]
    if not isinstance(node, dict):
        return node
    if reply_encoding._is_table(node):
        optional, common = node.get("optional", []), node.get("common", {})

        def owner(column):
            return next((prefix for prefix in optional if column == prefix or column.startswith(prefix + ".")), None)

        def put(element, column, value):
            *parents, leaf = column.split(".")
            for part in parents:
                element = element.setdefault(part, {})
            element[leaf] = value

        elements = []
        for row in node["rows"]:
            element, present = {}, set()
            for column, cell in zip(node["columns"], row):
                if cell is None and owner(column) is not None:
                    continue
                present.add(owner(column))
                put(element, column, cell)
            for column, value in common.items():
                if owner(column) is None or owner(column) in present:
                    put(element, column, value)
            elements.append(decode_compact_value(element))
        if "keyed_by" in node:
            return {element.pop(node["keyed_by"]): element for element in elements}
        if "first" in node:
            elements.insert(0, decode_compact_value(node["first"]))
        return elements
    result = {key: decode_compact_value(item) for key, item in node.items() if key != UNCHANGED_ARMS_KEY}
    if UNCHANGED_ARMS_KEY in node:
        result["samples"] = [dict(sample, arms=dict(sample["arms"], **deepcopy(node[UNCHANGED_ARMS_KEY])))
                             for sample in result["samples"]]
    return result


def compact_encoding_loss(original, decoded, loss=None, derived=False):
    """Compare a reply value with the decoding of its compact encoding. Every value must be equal except the two
    documented rewrites, whose size is returned: ``derived_relative`` (largest relative change of a rounded
    ``DERIVED_QUANTITIES`` number), ``unchanged_abs`` (largest change of a number of an arm stated once) and
    ``unchanged_derived_abs`` (largest magnitude of a derived number not printed for such an arm). Any other
    difference raises AssertionError."""
    loss = dict(derived_relative=0.0, unchanged_abs=0.0, unchanged_derived_abs=0.0, unchanged_arms=0) if loss is None else loss
    if isinstance(original, dict) and isinstance(decoded, dict):
        if decoded.get("unchanged") is True and "max_abs_joint_change_rad" in decoded and "unchanged" not in original:
            loss["unchanged_arms"] += 1
            stated = {key: item for key, item in decoded.items() if key not in ("unchanged", "max_abs_joint_change_rad")}
            measured = {key: item for key, item in original.items() if key != DERIVED_PARENT}
            difference = reply_encoding._within_step(measured, {key: stated[key] for key in measured})
            assert difference is not None and list(measured) == [key for key in stated if key != DERIVED_PARENT], (original, decoded)
            loss["unchanged_abs"] = max(loss["unchanged_abs"], difference)
            assert difference <= decoded["max_abs_joint_change_rad"] or difference <= STATIONARY_RESOLUTION * 4
            for key, value in (original.get(DERIVED_PARENT) or {}).items():
                if key in DERIVED_QUANTITIES:
                    loss["unchanged_derived_abs"] = max([loss["unchanged_derived_abs"], *map(abs, value)])
            return loss
        assert set(original) == set(decoded), (sorted(original), sorted(decoded))
        for key in original:
            compact_encoding_loss(original[key], decoded[key], loss, derived and key in DERIVED_QUANTITIES or key == DERIVED_PARENT)
        return loss
    if isinstance(original, list) and isinstance(decoded, list):
        assert len(original) == len(decoded), (original, decoded)
        for first, second in zip(original, decoded):
            compact_encoding_loss(first, second, loss, derived)
        return loss
    if derived is True and type(original) is float and type(decoded) is float and original != decoded:
        loss["derived_relative"] = max(loss["derived_relative"], abs(decoded - original) / abs(original))
        return loss
    assert type(original) is type(decoded) and original == decoded, (original, decoded)
    return loss


def reply_values(content_items):
    """The JSON text items of a delivered reply as values: ``same_as`` pointers expanded, compact tables decoded."""
    values = []
    for item in content_items:
        if item.get("type") in ("inputText", "input_text"):
            try:
                value = json.loads(item["text"])
            except ValueError:
                continue
            values.append(decode_compact_value(resolve_same_as(value)))
    return values


def action_result(content_items):
    """The execution result of a delivered action reply as one value, whatever its layout: the result item with the
    per-stage item (``{"action_id", "stages"}``) and the trajectory item (``{"action_id", "trajectory_feedback"}``)
    merged back in. Blocks given by reference stay as their stand-ins (``stages_by_reference`` is kept too)."""
    values = reply_values(content_items)
    result = dict(next(value for value in values if isinstance(value, dict) and isinstance(value.get("receipt"), dict)))
    for value in values:
        if not isinstance(value, dict) or "receipt" in value:
            continue
        if set(value) <= {"action_id", "stages", "stages_by_reference"} and ("stages" in value or "stages_by_reference" in value):
            result["stages"] = dict(value.get("stages", {}), **result.get("stages", {}))
            if "stages_by_reference" in value:
                result["stages_by_reference"] = value["stages_by_reference"]
        elif set(value) == {"action_id", "trajectory_feedback"}:
            result["trajectory_feedback"] = value["trajectory_feedback"]
    return result


def observation_value(content_items):
    """The post-action / current observation item of a delivered reply (the last JSON item with ``bundle``)."""
    return next(value for value in reversed(reply_values(content_items)) if isinstance(value, dict) and "bundle" in value)


# ---- "nothing necessary is lost": compare a delivered reply with the legacy presentation of the same result ------
_DETAIL_RENAMED = ("full_result", "read_full", "detail_access", "next_action")
_LEAN_V2_PROSE = ("result_semantics", "timeout_semantics")
_DISPLAY_QUANTUM = 5.0000001e-7   # lean_v2 prints six decimals


def _leaves(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _leaves(item)
    elif isinstance(value, list):
        for item in value:
            yield from _leaves(item)
    else:
        yield value


def _fixed(key, value, plan):
    """``value`` under ``key`` is entirely fixed notes of ``plan`` (what reply_static_once leaves out)."""
    if isinstance(value, str):
        return (key, value) in plan.strings
    if type(value) is bool:
        return (key, value) in plan.flags
    if isinstance(value, list):
        return bool(value) and ((key, tuple(value)) in plan.lists if all(isinstance(item, str) for item in value) else False)
    if isinstance(value, dict):
        return bool(value) and all(_fixed(name, item, plan) for name, item in value.items())
    return False


def _raw_frame(key, value):
    from ni_rua.core.presentation.actor_payload import _raw_frame_id
    return isinstance(key, str) and ((key.endswith("frame_id") and _raw_frame_id(value))
                                     or (key.endswith("frame_ids") and isinstance(value, list) and value
                                         and all(map(_raw_frame_id, value))))


def lost_values(legacy, delivered, plan, path="", derived=False, tool=None):
    """Paths of ``legacy`` (a value of the legacy presentation) that ``delivered`` (the same value read back from the
    delivered reply with ``reply_values`` / ``action_result``) does not carry, apart from what the reply settings
    document: fixed notes of ``plan``, blocks given by reference (their stand-in must be there), read entries renamed
    to ``detail``, the raw frame ids and empty residues lean_v2 leaves out, six-decimal display and 4-digit derived
    quantities, and an arm stated once. An empty list means every other value arrived."""
    lost = []
    if isinstance(legacy, dict) and isinstance(delivered, dict):
        if delivered.get("omitted") == "by_reference":
            return lost
        if delivered.get("unchanged") is True and "max_abs_joint_change_rad" in delivered:
            measured = {key: item for key, item in legacy.items() if key != DERIVED_PARENT}
            return lost + lost_values(measured, {key: delivered.get(key) for key in measured}, plan, path, tool=tool)
        for key, item in legacy.items():
            here = f"{path}/{key}"
            if key in delivered.get("__by_reference__", ()):
                continue
            if key in delivered:
                lost += lost_values(item, delivered[key], plan, here, derived or key == DERIVED_PARENT, tool)
            elif (_fixed(key, item, plan) or _raw_frame(key, item)
                  or (isinstance(item, list) and item and all(isinstance(entry, str) and (key, entry) in plan.strings
                                                              for entry in item)) or (key in _DETAIL_RENAMED and "detail" in delivered)
                  or key + "_by_reference" in delivered or (key == "camera_pose_diagnostics" and item == {})
                  or (key in _LEAN_V2_PROSE and tool in ("program_run", "program_status") and path == "")
                  or (key == "cameras" and isinstance(item, dict) and all(
                      not lost_values(camera, {}, plan, here, tool=tool) for camera in item.values()))
                  or (isinstance(item, dict) and not lost_values(item, {}, plan, here, tool=tool) and item != {})):
                continue
            else:
                lost.append(here)
        return lost
    if isinstance(legacy, list) and isinstance(delivered, list):
        if len(legacy) != len(delivered):
            key = path.rsplit("/", 1)[-1]
            kept = [item for item in legacy if not (isinstance(item, str) and (key, item) in plan.strings)]
            if len(kept) != len(delivered):
                return lost + [path + f"[{len(legacy)}!={len(delivered)}]"]
            legacy = kept
        for index, (first, second) in enumerate(zip(legacy, delivered)):
            lost += lost_values(first, second, plan, f"{path}/{index}", derived, tool)
        return lost
    if isinstance(delivered, dict) and delivered.get("omitted") == "by_reference":
        return lost
    if type(legacy) is float and type(delivered) in (int, float):
        tolerance = _DISPLAY_QUANTUM + (5.0001e-4 * abs(legacy) if derived else 0.0)
        return lost if abs(legacy - delivered) <= tolerance else lost + [f"{path}:{legacy!r}!={delivered!r}"]
    return lost if type(legacy) is type(delivered) and legacy == delivered else lost + [f"{path}:{legacy!r}!={delivered!r}"]


def reply_losses(legacy_items, delivered_items, plan, tool=None):
    """``lost_values`` over a whole reply: the legacy presentation's JSON text items against the delivered reply.
    The execution result is compared as one value (``action_result``); every other legacy item with the delivered
    item that shares most top-level keys with it."""
    legacy = [json.loads(item["text"]) for item in legacy_items if item.get("type") in ("inputText", "input_text")
              and item["text"][:1] in "{["]
    delivered = [_without_partial_samples(value) for value in reply_values(delivered_items)]
    lost, used = [], set()
    for value in legacy:
        if isinstance(value, dict) and isinstance(value.get("receipt"), dict):
            other = _without_partial_samples(action_result(delivered_items))
            note = other.get("stages_by_reference")
            if isinstance(note, dict) and isinstance(other.get("stages"), dict):
                for name in note.get("stage_ids", ()):
                    if name in other["stages"]:   # some fields of the stage are given by reference
                        other["stages"][name] = dict(other["stages"][name], __by_reference__=note.get("fields", ()))
                    else:                         # the whole stage entry is
                        other["stages"][name] = dict(omitted="by_reference")
            lost += lost_values(value, other, plan, tool=tool)
            continue
        # every other item: the delivered item (each used once) that carries it best
        candidates = [(lost_values(value, other, plan, tool=tool), index) for index, other in enumerate(delivered)
                      if index not in used and type(other) is type(value)
                      and not (isinstance(other, dict) and isinstance(other.get("receipt"), dict))]
        if not candidates:
            lost.append("<item missing>")
            continue
        missing, index = min(candidates, key=lambda candidate: len(candidate[0]))
        used.add(index)
        lost += missing
    return lost


def _without_partial_samples(value):
    """A samples list of which only the window around an exception is inline counts as given by reference."""
    if isinstance(value, dict):
        result = {key: _without_partial_samples(item) for key, item in value.items()}
        if "samples_by_reference" in result and isinstance(result.get("samples"), list):
            result["samples"] = result["samples_by_reference"]
        return result
    if isinstance(value, list):
        return [_without_partial_samples(item) for item in value]
    return value
