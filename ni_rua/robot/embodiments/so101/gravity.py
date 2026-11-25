"""Static gravity torques of the SO-101 arm joints from the packaged URDF inertials.

Units: kg, m, rad, N·m. Frame: ``base_link`` (``so101_base``) with gravity along its -Z axis (the base stands level
on the table; a tilted base is not modelled). Backed by C++ Pinocchio RNEA (``pinocchio_backend.pinocchio_gravity_torques``);
no serial bus, camera or LeRobot.

Sign convention: ``gravity_torques`` returns, per arm joint, the torque gravity exerts about the joint's URDF axis
(positive = gravity pushes the joint toward larger angles). It equals ``-dU/dq`` of the potential energy
``U = sum(mass * g * z_com)``. A position servo that holds the joint still supplies the opposite torque, so a pure-P
servo of stiffness ``K`` (N·m/rad) rests at ``goal - measured = -gravity_torque / K`` (``feedforward_offset_rad``).

The link masses and centres of mass are the URDF's CAD values (``<inertial>``); they are not weighed on this unit and
do not include a held object.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np

from . import profile as kinematics

#: Standard gravity (m/s^2), along -Z of ``base_link``.
GRAVITY_M_S2 = 9.80665


def load_inertials(urdf_path=None) -> dict[str, dict]:
    """``{link: {"mass_kg": float, "com_m": [x, y, z]}}`` of every URDF link with an ``<inertial>`` element.

    ``com_m``: the centre of mass in the link's own frame (m; the ``<inertial><origin xyz>``). ``urdf_path``: optional
    path to an SO-101 URDF file; default ``None`` reads the packaged assembly member that ``profile.load_model``
    reads. Example: ``load_inertials()["upper_arm_link"]["mass_kg"]`` -> 0.103."""
    if urdf_path is not None:
        data = Path(urdf_path).read_bytes()
    else:
        package = kinematics._PACKAGE
        native = json.loads((package / "physical.json").read_text(encoding="utf-8"))["geometry"]["native_model"]
        with zipfile.ZipFile(package / f"{native['resource_id']}.zip") as archive:
            data = archive.read(native["assembly_member"])
    inertials = {}
    for link in ET.fromstring(data).findall("link"):
        inertial = link.find("inertial")
        mass = None if inertial is None else inertial.find("mass")
        if mass is None:
            continue
        origin = inertial.find("origin")
        com = [float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()]
        value = float(mass.get("value"))
        if len(com) != 3 or not np.isfinite([value, *com]).all() or value < 0:
            raise ValueError(f"so101_inertial_invalid:{link.get('name')}")
        inertials[link.get("name")] = {"mass_kg": value, "com_m": com}
    return inertials


def gravity_torques(model, inertials, q_arm_rad, *, gripper_rad: float = 0.0) -> np.ndarray:
    """Gravity torque about each arm joint axis (N·m, order ``ARM_JOINTS``) at one joint configuration.

    ``model``: ``profile.load_model()``; ``inertials``: ``load_inertials()``; ``q_arm_rad``: five arm joint
    angles (rad); ``gripper_rad``: jaw joint angle (rad, default 0; it only moves the 12 g jaw link). Computed via
    exact C++ Pinocchio RNEA (``ni_rua.robot.model.pinocchio_backend.pinocchio_gravity_torques``), matching
    ``sum over links below j of a . ((com - p) x (0, 0, -m g))`` to ``< 1e-14 N·m``.
    Example: ``gravity_torques(load_model(), load_inertials(), [0, -1, 1, 0.6, 0])``."""
    from ...model.pinocchio_backend import pinocchio_gravity_torques

    return pinocchio_gravity_torques(model, inertials, q_arm_rad, gripper_rad=gripper_rad)


def feedforward_offset_rad(torques_nm, stiffness_nm_per_rad, max_offset_rad) -> np.ndarray:
    """Joint goal offset (rad per arm joint) that lets a pure-P position servo hold the commanded angle under gravity.

    ``torques_nm``: ``gravity_torques`` (N·m). ``stiffness_nm_per_rad``: five values, the identified servo position
    stiffness of each arm joint (N·m of restoring torque per rad of position error), or ``None`` for a joint without
    feedforward (offset 0). ``max_offset_rad``: bound of every offset's magnitude (rad). Result:
    ``clip(-torque / stiffness, -max_offset_rad, max_offset_rad)``. Example: torque -0.4 N·m (gravity pulls toward
    smaller angles), stiffness 10 N·m/rad -> +0.04 rad on the goal."""
    offsets = np.zeros(len(torques_nm))
    for index, (torque, stiffness) in enumerate(zip(torques_nm, stiffness_nm_per_rad)):
        if stiffness is not None:
            offsets[index] = -float(torque) / float(stiffness)
    return np.clip(offsets, -float(max_offset_rad), float(max_offset_rad))
