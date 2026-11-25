"""Lowest SO-101 gripper mesh point at a measured joint state (used for table touch-off constraints)."""
from __future__ import annotations

import functools
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from . import gripper_cad as geometry
from . import profile as kin

CONTACT_LINKS = ("gripper_link", "moving_jaw_so101_v1_link")


@functools.lru_cache(maxsize=1)
def _link_vertices() -> dict[str, np.ndarray]:
    path = geometry._PACKAGE / "geometry_model.zip"
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read(next(n for n in archive.namelist() if n.endswith(".urdf"))))
        return {
            name: np.unique(geometry._link_meshes(root, archive, name).reshape(-1, 3), axis=0)
            for name in CONTACT_LINKS
        }


def lowest_gripper_point(model, q_rad, gripper_rad: float = 0.0) -> dict:
    """Return link, local point (``m``), position_m (base XYZ, ``m``), and z_m of the lowest CAD vertex."""
    poses = kin.fk_all(model, q_rad, gripper_rad=gripper_rad)
    best = None
    for link, vertices in _link_vertices().items():
        world = (poses[link][:3, :3] @ vertices.T).T + poses[link][:3, 3]
        i = int(np.argmin(world[:, 2]))
        if best is None or world[i, 2] < best["z_m"]:
            best = {
                "link": link,
                "point_in_link_m": vertices[i].tolist(),
                "position_m": world[i].tolist(),
                "z_m": float(world[i, 2]),
            }
    return best
