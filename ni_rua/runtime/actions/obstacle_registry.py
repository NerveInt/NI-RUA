"""Runtime session obstacle registry binding ``pose_skill.attached_table`` into pure ``scene_obstacles`` math."""
from __future__ import annotations

from ...core.episode_state import StateSlot
from ...perception.external import pose_skill as _pose_skill
from ...robot.motion.scene_obstacles import (
    CLEARANCE_PRESETS,
    DEFAULT_CLEARANCE_HEIGHT_M,
    DEFAULT_TARGET_EXEMPTION_RADIUS_M,
    MAX_PERSISTENT_OBSTACLES,
    OBSTACLE_CLEARANCE_FIELDS,
    STAGE_CLEARANCE_FIELDS,
    TARGET_EXEMPTION_PRESETS,
    merge_and_persist_obstacles as _merge_and_persist_obstacles,
    obstacles_from_object_table as _obstacles_from_object_table,
    record_placed_obstacles,
    resolve_stage_clearance,
    resolve_target_exemption_radius,
)

PERSISTENT_OBSTACLES_SLOT: StateSlot[dict] = StateSlot("persistent_obstacles", dict, scope="episode")
_UNSET = object()


def obstacles_from_object_table(
    harness,
    observation_id,
    object_ids=None,
    *,
    min_height_above_table_m=0.020,
    object_table=_UNSET,
    report=None,
):
    """Convert cached 3D ``object_table`` rows attached to ``observation_id`` into world-frame AABB obstacles (``m``)."""
    resolver = (lambda _h, _oid: object_table) if object_table is not _UNSET else _pose_skill.attached_table
    return _obstacles_from_object_table(
        harness,
        observation_id,
        object_ids=object_ids,
        min_height_above_table_m=min_height_above_table_m,
        table_resolver=resolver,
        report=report,
    )


def merge_and_persist_obstacles(harness, args, compiled_stages=None, *, object_table=_UNSET, report=None):
    """Manage per-episode persistent obstacles and merge ``obstacle_object_ids`` from ``object_table``."""
    resolver = (lambda _h, _oid: object_table) if object_table is not _UNSET else _pose_skill.attached_table
    return _merge_and_persist_obstacles(
        harness,
        args,
        compiled_stages=compiled_stages,
        table_resolver=resolver,
        report=report,
    )


__all__ = [
    "CLEARANCE_PRESETS",
    "DEFAULT_CLEARANCE_HEIGHT_M",
    "DEFAULT_TARGET_EXEMPTION_RADIUS_M",
    "MAX_PERSISTENT_OBSTACLES",
    "OBSTACLE_CLEARANCE_FIELDS",
    "STAGE_CLEARANCE_FIELDS",
    "TARGET_EXEMPTION_PRESETS",
    "merge_and_persist_obstacles",
    "obstacles_from_object_table",
    "record_placed_obstacles",
    "resolve_stage_clearance",
    "resolve_target_exemption_radius",
]
