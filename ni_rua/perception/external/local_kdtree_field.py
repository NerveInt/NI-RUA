"""兼容重导出门面：局部坐标系 `LocalSurfaceKDTree` 与实测残差不确定度算子已收敛至 `geometry_ops.py`。"""
from __future__ import annotations

if __package__:
    from .geometry_ops import (
        DEFAULT_SURFACE_VOXEL_RADIUS_M,
        MIN_CAVITY_DEPTH_M,
        LocalSurfaceKDTree,
        derive_object_spatial_meta,
        derive_table_spatial_meta,
    )
else:
    from geometry_ops import (  # type: ignore[no-redef]
        DEFAULT_SURFACE_VOXEL_RADIUS_M,
        MIN_CAVITY_DEPTH_M,
        LocalSurfaceKDTree,
        derive_object_spatial_meta,
        derive_table_spatial_meta,
    )

__all__ = [
    "DEFAULT_SURFACE_VOXEL_RADIUS_M",
    "MIN_CAVITY_DEPTH_M",
    "LocalSurfaceKDTree",
    "derive_object_spatial_meta",
    "derive_table_spatial_meta",
]
