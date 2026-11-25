"""兼容重导出门面：向量化 OBB 距离核与批量抓取净空评估已收敛至 `pose_math.py`、`geometry_ops.py` 与 `clearance_oracle.py`。"""
from __future__ import annotations

if __package__:
    from .pose_math import angles_deg_to_matrices_batch
    from .geometry_ops import (
        batch_obbs_to_slabs_min_distance,
        obb_sat_max_gap_batch,
        obb_signed_distance_batch,
        obb_signed_distance_pairs,
    )
    from .clearance_oracle import evaluate_grasp_clearance_batch
else:
    from pose_math import angles_deg_to_matrices_batch  # type: ignore[no-redef]
    from geometry_ops import (  # type: ignore[no-redef]
        batch_obbs_to_slabs_min_distance,
        obb_sat_max_gap_batch,
        obb_signed_distance_batch,
        obb_signed_distance_pairs,
    )
    from clearance_oracle import evaluate_grasp_clearance_batch  # type: ignore[no-redef]


def evaluate_grasp_batch(field, target, positions_m, angles_deg, **kwargs):
    """批量评估候选三维抓取位姿 `(positions_m, angles_deg)` 的客观可夹持性与邻物净空（兼容 `field=` 关键字参数）。

    参数:
        field: `VirtualSpatialField` / `PointCloudSpatialField` 实例或物体映射。
        target: 目标 `VirtualObject` 实例。
        positions_m: 候选抓取位置数组 `(G, 3)`，单位 `m`。
        angles_deg: 候选姿态角数组 `[pitch_deg, roll_deg, yaw_deg]` `(G, 3)`，单位 `deg`。
        **kwargs: 透传给 `evaluate_grasp_clearance_batch` 的可选几何评估参数（单位 `m`）。
    """
    return evaluate_grasp_clearance_batch(field, target, positions_m, angles_deg, **kwargs)

__all__ = [
    "angles_deg_to_matrices_batch",
    "batch_obbs_to_slabs_min_distance",
    "evaluate_grasp_batch",
    "obb_sat_max_gap_batch",
    "obb_signed_distance_batch",
    "obb_signed_distance_pairs",
]
