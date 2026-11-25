"""Immutable per-observation 3D world model (`WorldObservation3D`).

Provides one canonical, immutable per-observation representation of:
- detected 3D `object_table` rows (`ObjectRow3D`),
- cavity depths and layered convex-hull surface slices (`cavity_depths_m`, `surface_slices_m`),
- calibrated camera descriptors (`CalibratedCameraRef`),
- dense `scene3d` point-cloud artifact metadata (`cloud_artifact_ref`, `cloud_points`, `cloud_table_z_m`), and
- public held-object state (`held_by_arm`, `held_meta`, `robot_state`),

shared across `object_table.py`, `obs3d.py`, `focus_3d.py`, `grasp_preview_3d.py`, and `unified_actions.py`.
Contains zero upward imports (`core` + stdlib + `numpy` only).
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
import math
from typing import Any

import numpy as np

from ..core.embodiment import FrozenDict
from ..core.episode_state import StateSlot, slot_get, slot_peek, slot_set

DEFAULT_TABLE_COLUMNS: tuple[str, ...] = (
    "id",
    "centre_m",
    "top_z",
    "bottom_z",
    "length",
    "width",
    "yaw_deg",
    "tilt_deg",
    "support",
    "quality",
    "label_px",
)

WORLD_OBS3D_MEMO_SLOT: StateSlot = StateSlot("world_obs3d_memo", lambda: None, scope="episode")
WORLD_OBS3D_HISTORY_SLOT: StateSlot[dict] = StateSlot("world_obs3d_history", dict, scope="episode")
# Legacy slot descriptors mirrored for zero-import compatibility with stub SimpleNamespace fixtures.
_LEGACY_OBS3D_MEMO_SLOT: StateSlot = StateSlot("obs3d_memo", lambda: None, scope="episode")
_LEGACY_OBS3D_HISTORY_SLOT: StateSlot[dict] = StateSlot("obs3d_history", dict, scope="episode")

MAX_WORLD_HISTORY: int = 8


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _opt_float(value: Any) -> float | None:
    return float(value) if _is_num(value) else None


def _freeze_slice_value(value: Any) -> Any:
    """Recursively freeze slice dict/list/ndarray entries into hashable immutable structures."""
    if isinstance(value, dict):
        return FrozenDict({str(k): _freeze_slice_value(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_slice_value(v) for v in value)
    if isinstance(value, np.ndarray):
        return _freeze_slice_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _freeze_slices(slices_raw: Any) -> tuple[FrozenDict, ...]:
    if not isinstance(slices_raw, (list, tuple)) or not slices_raw:
        return ()
    return tuple(_freeze_slice_value(s) for s in slices_raw if isinstance(s, dict))


@dataclass(frozen=True)
class CalibratedCameraRef:
    """Immutable per-camera reference metadata captured in an observation bundle."""

    camera_id: str
    frame_id: str | None = None
    width: int | None = None
    height: int | None = None
    calibration_ref: str | None = None
    pose_sample_ref: str | None = None
    rgb_ref: str | None = None


@dataclass(frozen=True)
class ObjectRow3D:
    """Immutable 3D object row parsed from a model-visible `object_table`."""

    id: str
    centre_m: tuple[float, float, float] | None
    top_z: float | None = None
    bottom_z: float | None = None
    length: float | None = None
    width: float | None = None
    yaw_deg: float | None = None
    tilt_deg: float | None = None
    support: str | None = None
    quality: str | None = None
    label_px: tuple[int, int] | None = None
    label: str | None = None
    cavity_depth_m: float | None = None
    surface_slices_m: tuple[FrozenDict, ...] = ()
    uncertainty_m: float | None = None

    def __post_init__(self) -> None:
        if self.surface_slices_m and (
            not isinstance(self.surface_slices_m, tuple)
            or any(not isinstance(s, FrozenDict) for s in self.surface_slices_m)
        ):
            object.__setattr__(self, "surface_slices_m", _freeze_slices(self.surface_slices_m))

    @property
    def is_held(self) -> bool:
        return str(self.support or "").startswith("held_") or self.quality == "held_fk_depth"

    @property
    def held_by_arm(self) -> str | None:
        sup = str(self.support or "")
        if sup.startswith("held_") and len(sup) > 5:
            return sup[5:]
        return None

    @property
    def height_m(self) -> float | None:
        if self.top_z is not None and self.bottom_z is not None:
            return max(0.0, float(self.top_z) - float(self.bottom_z))
        return None

    def to_row_dict(self) -> dict[str, Any]:
        """Return a mutable row dict matching `object_table.COLUMNS` keys."""
        out: dict[str, Any] = dict(
            id=self.id,
            centre_m=list(self.centre_m) if self.centre_m is not None else None,
            top_z=self.top_z,
            bottom_z=self.bottom_z,
            length=self.length,
            width=self.width,
            yaw_deg=self.yaw_deg,
            tilt_deg=self.tilt_deg,
            support=self.support,
            quality=self.quality,
            label_px=list(self.label_px) if self.label_px is not None else None,
        )
        if self.label is not None:
            out["label"] = self.label
        return out


@dataclass(frozen=True)
class WorldObservation3D:
    """Immutable per-observation 3D world snapshot shared across perception and action review."""

    observation_id: str
    camera_id: str | None = None
    table_z_m: float | None = None
    columns: tuple[str, ...] = DEFAULT_TABLE_COLUMNS
    rows: tuple[ObjectRow3D, ...] = ()
    cavity_depths_m: FrozenDict = field(default_factory=lambda: FrozenDict({}))
    surface_slices_m: FrozenDict = field(default_factory=lambda: FrozenDict({}))
    calibrated_cameras: FrozenDict = field(default_factory=lambda: FrozenDict({}))
    cloud_artifact_ref: str | None = None
    cloud_points: int | None = None
    cloud_table_z_m: float | None = None
    held_by_arm: FrozenDict = field(default_factory=lambda: FrozenDict({}))
    _table_payload: dict[str, Any] | None = field(default=None, repr=False, compare=False)
    _raw_objects: tuple[dict[str, Any], ...] = field(default=(), repr=False, compare=False)
    _cloud_payload: dict[str, Any] | None = field(default=None, repr=False, compare=False)
    _robot_state: dict[str, Any] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.surface_slices_m, FrozenDict) or any(
            not isinstance(v, tuple) or any(not isinstance(s, FrozenDict) for s in v)
            for v in self.surface_slices_m.values()
        ):
            object.__setattr__(
                self,
                "surface_slices_m",
                FrozenDict(
                    {
                        str(k): _freeze_slices(v)
                        for k, v in dict(self.surface_slices_m or {}).items()
                        if isinstance(v, (list, tuple))
                    }
                ),
            )

    @classmethod
    def from_parts(
        cls,
        observation_id: str,
        *,
        table: dict[str, Any] | None = None,
        raw_objects: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
        camera_id: str | None = None,
        bundle: dict[str, Any] | None = None,
        cloud: dict[str, Any] | None = None,
        robot_state: dict[str, Any] | None = None,
    ) -> WorldObservation3D:
        """Build an immutable `WorldObservation3D` from `object_table` / `scene3d` / `ObservationBundle` parts."""
        cols = tuple(table.get("columns") or DEFAULT_TABLE_COLUMNS) if isinstance(table, dict) else DEFAULT_TABLE_COLUMNS
        table_z = None
        if isinstance(table, dict):
            tz_cand = table.get("table_z") if "table_z" in table else table.get("support_z_m")
            table_z = _opt_float(tz_cand)

        hints = (table.get("grasp_hints") if isinstance(table, dict) and isinstance(table.get("grasp_hints"), dict) else {}) or {}
        spatial_meta = (table.get("spatial_meta") if isinstance(table, dict) and isinstance(table.get("spatial_meta"), dict) else {}) or {}

        raw_by_id: dict[str, dict[str, Any]] = {}
        raw_tuple: tuple[dict[str, Any], ...] = ()
        if isinstance(raw_objects, (list, tuple)):
            copied_raw = [deepcopy(item) for item in raw_objects if isinstance(item, dict)]
            raw_tuple = tuple(copied_raw)
            for idx, item in enumerate(copied_raw):
                oid = str(item.get("track_id") or item.get("id") or f"o{idx + 1}")
                raw_by_id[oid] = item

        cav_map: dict[str, float] = {}
        slices_map: dict[str, tuple[FrozenDict, ...]] = {}
        for oid, h in hints.items():
            if isinstance(h, dict) and _is_num(h.get("cavity_depth_m")):
                cav_map[str(oid)] = float(h["cavity_depth_m"])
        for oid, r_obj in raw_by_id.items():
            if _is_num(r_obj.get("cavity_depth_m")):
                cav_map.setdefault(oid, float(r_obj["cavity_depth_m"]))
            s_raw = r_obj.get("surface_slices_m")
            if isinstance(s_raw, (list, tuple)) and s_raw:
                slices_map[oid] = _freeze_slices(s_raw)

        parsed_rows: list[ObjectRow3D] = []
        held_arms: dict[str, str] = {}
        if isinstance(table, dict) and isinstance(table.get("rows"), list):
            for raw_row in table["rows"]:
                if isinstance(raw_row, (list, tuple)) and len(raw_row) >= 2:
                    d = dict(zip(cols, raw_row))
                elif isinstance(raw_row, dict):
                    d = dict(raw_row)
                else:
                    continue
                oid = str(d.get("id") or "")
                if not oid:
                    continue
                c_raw = d.get("centre_m")
                centre = (
                    (float(c_raw[0]), float(c_raw[1]), float(c_raw[2]))
                    if isinstance(c_raw, (list, tuple)) and len(c_raw) == 3 and all(_is_num(v) for v in c_raw)
                    else None
                )
                unc = None
                sm = spatial_meta.get(oid)
                if isinstance(sm, dict) and _is_num(sm.get("uncertainty_m")):
                    unc = float(sm["uncertainty_m"])
                raw_px = d.get("label_px")
                if raw_px is None and oid in raw_by_id:
                    raw_px = raw_by_id[oid].get("label_px")
                label_px = (
                    (int(round(float(raw_px[0]))), int(round(float(raw_px[1]))))
                    if isinstance(raw_px, (list, tuple)) and len(raw_px) == 2 and all(_is_num(v) for v in raw_px)
                    else None
                )
                row_obj = ObjectRow3D(
                    id=oid,
                    centre_m=centre,
                    top_z=_opt_float(d.get("top_z", d.get("top_z_m"))),
                    bottom_z=_opt_float(d.get("bottom_z", d.get("bottom_z_m"))),
                    length=_opt_float(d.get("length", d.get("length_m"))),
                    width=_opt_float(d.get("width", d.get("width_m"))),
                    yaw_deg=_opt_float(d.get("yaw_deg")),
                    tilt_deg=_opt_float(d.get("tilt_deg")),
                    support=str(d["support"]) if d.get("support") is not None else None,
                    quality=str(d["quality"]) if d.get("quality") is not None else None,
                    label_px=label_px,
                    label=str(d["label"]) if d.get("label") is not None else None,
                    cavity_depth_m=cav_map.get(oid),
                    surface_slices_m=slices_map.get(oid, ()),
                    uncertainty_m=unc,
                )
                parsed_rows.append(row_obj)
                if row_obj.held_by_arm:
                    held_arms[row_obj.held_by_arm] = oid

        if isinstance(table, dict) and isinstance(table.get("held"), dict):
            for oid, h_meta in table["held"].items():
                if isinstance(h_meta, dict) and isinstance(h_meta.get("arm"), str):
                    held_arms.setdefault(h_meta["arm"], str(oid))

        cams: dict[str, CalibratedCameraRef] = {}
        if isinstance(bundle, dict) and isinstance(bundle.get("cameras"), list):
            for cam in bundle["cameras"]:
                if not isinstance(cam, dict) or not cam.get("camera_id"):
                    continue
                cid = str(cam["camera_id"])
                cams[cid] = CalibratedCameraRef(
                    camera_id=cid,
                    frame_id=cam.get("frame_id"),
                    width=int(cam["width"]) if isinstance(cam.get("width"), int) else None,
                    height=int(cam["height"]) if isinstance(cam.get("height"), int) else None,
                    calibration_ref=cam.get("calibration_ref"),
                    pose_sample_ref=cam.get("pose_sample_ref"),
                    rgb_ref=cam.get("rgb_ref"),
                )

        cloud_ref = cloud.get("artifact") if isinstance(cloud, dict) and isinstance(cloud.get("artifact"), str) else None
        cloud_pts = int(cloud["points"]) if isinstance(cloud, dict) and isinstance(cloud.get("points"), int) else None
        cloud_tz = _opt_float(cloud.get("table_z")) if isinstance(cloud, dict) else None
        eff_cam_id = camera_id or (table.get("camera_id") if isinstance(table, dict) and isinstance(table.get("camera_id"), str) else None)

        return cls(
            observation_id=str(observation_id),
            camera_id=eff_cam_id,
            table_z_m=table_z,
            columns=cols,
            rows=tuple(parsed_rows),
            cavity_depths_m=FrozenDict(cav_map),
            surface_slices_m=FrozenDict(slices_map),
            calibrated_cameras=FrozenDict(cams),
            cloud_artifact_ref=cloud_ref,
            cloud_points=cloud_pts,
            cloud_table_z_m=cloud_tz,
            held_by_arm=FrozenDict(held_arms),
            _table_payload=deepcopy(table) if isinstance(table, dict) else None,
            _raw_objects=raw_tuple,
            _cloud_payload=deepcopy(cloud) if isinstance(cloud, dict) else None,
            _robot_state=deepcopy(robot_state) if isinstance(robot_state, dict) else None,
        )

    def with_updates(
        self,
        *,
        table: dict[str, Any] | None = None,
        raw_objects: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
        camera_id: str | None = None,
        bundle: dict[str, Any] | None = None,
        cloud: dict[str, Any] | None = None,
        robot_state: dict[str, Any] | None = None,
    ) -> WorldObservation3D:
        """Return a new `WorldObservation3D` merging new table or cloud attachments immutably."""
        eff_table = table if isinstance(table, dict) else self._table_payload
        eff_raw = raw_objects if raw_objects is not None else self._raw_objects
        eff_cloud = cloud if isinstance(cloud, dict) else self._cloud_payload
        eff_rs = robot_state if isinstance(robot_state, dict) else self._robot_state
        built = WorldObservation3D.from_parts(
            self.observation_id,
            table=eff_table,
            raw_objects=eff_raw,
            camera_id=camera_id or self.camera_id,
            bundle=bundle,
            cloud=eff_cloud,
            robot_state=eff_rs,
        )
        if not built.calibrated_cameras and self.calibrated_cameras:
            return replace(built, calibrated_cameras=self.calibrated_cameras)
        return built

    @property
    def has_table(self) -> bool:
        return self._table_payload is not None

    def table_dict(self) -> dict[str, Any] | None:
        """Return a defensive copy of the attached `object_table` dict, or `None`."""
        return deepcopy(self._table_payload) if self._table_payload is not None else None

    def raw_objects(self) -> list[dict[str, Any]] | None:
        """Return diagnostic object dicts (`contour_px`, `surface_slices_m`, etc.) when available."""
        return [deepcopy(item) for item in self._raw_objects] if self._raw_objects else None

    def cloud_dict(self) -> dict[str, Any] | None:
        return deepcopy(self._cloud_payload) if self._cloud_payload is not None else None

    def robot_state_dict(self) -> dict[str, Any] | None:
        return deepcopy(self._robot_state) if self._robot_state is not None else None

    def row(self, object_id: str) -> ObjectRow3D | None:
        """Look up a typed `ObjectRow3D` by `object_id`."""
        target = str(object_id)
        for item in self.rows:
            if item.id == target:
                return item
        return None

    def row_dict(self, object_id: str) -> dict[str, Any] | None:
        """Look up a single row dict matching `object_table.COLUMNS` by `object_id`."""
        found = self.row(object_id)
        return found.to_row_dict() if found is not None else None

    def measured_support_z_m(self) -> float | None:
        """Extract measured support-plane height in `m` without hardcoded fallback constants."""
        if self.cloud_table_z_m is not None:
            return float(self.cloud_table_z_m)
        if self.table_z_m is not None:
            return float(self.table_z_m)
        table_bzs = sorted(
            float(r.bottom_z)
            for r in self.rows
            if r.support == "table" and r.bottom_z is not None and math.isfinite(r.bottom_z)
        )
        if table_bzs:
            return table_bzs[len(table_bzs) // 2]
        return None

    def match_graspable_row(self, gx: float, gy: float, max_opening_m: float) -> tuple[dict[str, Any] | None, float]:
        """Match the closest non-held 3D object row within grasp reach that fits `max_opening_m`."""
        best_row: ObjectRow3D | None = None
        best_dist = math.inf
        for r in self.rows:
            if r.is_held or r.centre_m is None or r.length is None or r.width is None or r.yaw_deg is None:
                continue
            if r.width > max_opening_m:
                continue
            cx, cy = r.centre_m[0], r.centre_m[1]
            dist = math.hypot(float(gx) - cx, float(gy) - cy)
            reach = max(0.5 * r.length + 0.035, 0.5 * r.width + 0.035, 0.065)
            if dist <= reach and dist < best_dist:
                best_row, best_dist = r, dist
        return (best_row.to_row_dict() if best_row is not None else None), best_dist

    def envelope_target(
        self,
        object_id: str,
        *,
        min_radius_m: float = 0.02,
        max_radius_m: float = 0.40,
    ) -> dict[str, Any] | None:
        """Resolve `centre_m` and adaptive bounding-sphere `envelope_radius_m` for `focus_3d`."""
        r = self.row(object_id)
        if r is None:
            return None
        if r.centre_m is None:
            raise ValueError(f"focus_3d_object_missing_centre_m:{object_id}")
        l_m = max(float(r.length) if r.length is not None else 0.04, 0.004)
        w_m = max(float(r.width) if r.width is not None else 0.04, 0.004)
        h_m = max(float(r.height_m) if r.height_m is not None else 0.04, 0.004)
        half_diag = 0.5 * math.sqrt(l_m * l_m + w_m * w_m + h_m * h_m)
        env_r = float(np.clip(1.25 * half_diag, min_radius_m, max_radius_m))
        return dict(
            object_id=r.id,
            centre_m=list(r.centre_m),
            envelope_radius_m=env_r,
        )


def _hydrate_from_legacy_entry(obs_id: str, entry: Any) -> WorldObservation3D | None:
    if not isinstance(entry, dict):
        return None
    ot_entry = entry.get("object_table")
    table = None
    raw_objects = None
    cam_id = None
    if isinstance(ot_entry, tuple) and ot_entry:
        table = ot_entry[0] if isinstance(ot_entry[0], dict) else None
        if len(ot_entry) >= 3 and isinstance(ot_entry[2], dict):
            raw_objects = (ot_entry[2].get("diagnostics") or {}).get("objects")
            cam_id = (ot_entry[2].get("value") or {}).get("camera_id")
    elif isinstance(ot_entry, dict):
        table = ot_entry

    s3_entry = entry.get("scene3d")
    cloud = None
    if isinstance(s3_entry, tuple) and s3_entry:
        cloud = s3_entry[0] if isinstance(s3_entry[0], dict) else None
    elif isinstance(s3_entry, dict):
        cloud = s3_entry

    if table is None and cloud is None:
        return None
    return WorldObservation3D.from_parts(
        obs_id,
        table=table,
        raw_objects=raw_objects,
        camera_id=cam_id,
        cloud=cloud,
    )


def record_world_observation(runtime: Any, world_obs: WorldObservation3D) -> WorldObservation3D:
    """Persist an immutable `WorldObservation3D` in the episode ledger (bounded LRU of 8)."""
    try:
        slot_set(runtime, WORLD_OBS3D_MEMO_SLOT, (world_obs.observation_id, world_obs))
        history = slot_get(runtime, WORLD_OBS3D_HISTORY_SLOT)
        if not isinstance(history, dict):
            history = {}
            slot_set(runtime, WORLD_OBS3D_HISTORY_SLOT, history)
        history[world_obs.observation_id] = world_obs
        while len(history) > MAX_WORLD_HISTORY:
            history.pop(next(iter(history)))
    except Exception as error:
        event_fn = getattr(getattr(runtime, "store", None), "event", None)
        if callable(event_fn):
            try:
                event_fn(
                    "world_model.record_failed",
                    {"observation_id": world_obs.observation_id, "error": f"{type(error).__name__}: {str(error)[:160]}"},
                    {},
                    world_kind="internal",
                )
            except Exception:
                pass
    return world_obs


def get_world_observation(runtime: Any, observation_id: str | None = None) -> WorldObservation3D | None:
    """Retrieve the `WorldObservation3D` for `observation_id` (or latest when `None`), hydrating legacy test stubs seamlessly."""
    if runtime is None:
        return None

    # First check legacy memo/history if a test mutated _obs3d_memo directly without going through record_world_observation.
    legacy_memo = slot_peek(runtime, _LEGACY_OBS3D_MEMO_SLOT, None, legacy_attr="_obs3d_memo")
    legacy_history = slot_peek(runtime, _LEGACY_OBS3D_HISTORY_SLOT, None, legacy_attr="_obs3d_history")

    w_memo = slot_peek(runtime, WORLD_OBS3D_MEMO_SLOT, None)
    w_history = slot_peek(runtime, WORLD_OBS3D_HISTORY_SLOT, None)

    if observation_id is None:
        if isinstance(w_memo, tuple) and len(w_memo) == 2 and isinstance(w_memo[1], WorldObservation3D):
            return w_memo[1]
        if isinstance(legacy_memo, tuple) and len(legacy_memo) == 2 and legacy_memo[0]:
            return _hydrate_from_legacy_entry(str(legacy_memo[0]), legacy_memo[1])
        return None

    target_id = str(observation_id)
    if isinstance(w_memo, tuple) and len(w_memo) == 2 and w_memo[0] == target_id and isinstance(w_memo[1], WorldObservation3D):
        # Verify legacy memo wasn't overwritten in-place by a test stub for a different table
        w_cand: WorldObservation3D = w_memo[1]
        if isinstance(legacy_memo, tuple) and len(legacy_memo) == 2 and legacy_memo[0] == target_id:
            hydrated = _hydrate_from_legacy_entry(target_id, legacy_memo[1])
            if hydrated is not None and hydrated.has_table and not w_cand.has_table:
                return hydrated
            if hydrated is not None and hydrated.cloud_artifact_ref and not w_cand.cloud_artifact_ref:
                return w_cand.with_updates(cloud=hydrated.cloud_dict())
        return w_cand

    if isinstance(w_history, dict) and target_id in w_history and isinstance(w_history[target_id], WorldObservation3D):
        return w_history[target_id]

    if isinstance(legacy_memo, tuple) and len(legacy_memo) == 2 and str(legacy_memo[0]) == target_id:
        return _hydrate_from_legacy_entry(target_id, legacy_memo[1])
    if isinstance(legacy_history, dict) and target_id in legacy_history:
        return _hydrate_from_legacy_entry(target_id, legacy_history[target_id])
    return None


def iter_world_observations(runtime: Any, preferred_observation_id: str | None = None) -> list[WorldObservation3D]:
    """Yield cached `WorldObservation3D` snapshots ordered: exact match first, then newest-to-oldest."""
    out: list[WorldObservation3D] = []
    seen: set[str] = set()

    def _add(item: WorldObservation3D | None) -> None:
        if item is not None and item.has_table and item.observation_id not in seen:
            seen.add(item.observation_id)
            out.append(item)

    if preferred_observation_id:
        _add(get_world_observation(runtime, preferred_observation_id))

    w_memo = slot_peek(runtime, WORLD_OBS3D_MEMO_SLOT, None)
    if isinstance(w_memo, tuple) and len(w_memo) == 2 and isinstance(w_memo[1], WorldObservation3D):
        _add(w_memo[1])
    w_history = slot_peek(runtime, WORLD_OBS3D_HISTORY_SLOT, None)
    if isinstance(w_history, dict):
        for obs_id in reversed(list(w_history.keys())):
            val = w_history.get(obs_id)
            if isinstance(val, WorldObservation3D):
                _add(val)

    legacy_memo = slot_peek(runtime, _LEGACY_OBS3D_MEMO_SLOT, None, legacy_attr="_obs3d_memo")
    if isinstance(legacy_memo, tuple) and len(legacy_memo) == 2 and legacy_memo[0]:
        _add(_hydrate_from_legacy_entry(str(legacy_memo[0]), legacy_memo[1]))
    legacy_history = slot_peek(runtime, _LEGACY_OBS3D_HISTORY_SLOT, None, legacy_attr="_obs3d_history")
    if isinstance(legacy_history, dict):
        for obs_id in reversed(list(legacy_history.keys())):
            _add(_hydrate_from_legacy_entry(str(obs_id), legacy_history[obs_id]))

    return out


def attached_table(runtime: Any, observation_id: str | None) -> dict[str, Any] | None:
    """Return the model-visible `object_table` dict attached to `observation_id`, or `None` when absent."""
    if not observation_id:
        return None
    world_obs = get_world_observation(runtime, observation_id)
    return world_obs.table_dict() if world_obs is not None else None
