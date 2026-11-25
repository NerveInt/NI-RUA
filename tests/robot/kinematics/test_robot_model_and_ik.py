"""``services/curobo_ik/build_model.shrink_limits`` against the packaged SO-101 URDF (cuRobo is not imported)."""

NI_RUA_TEST_TIER = "fast"

# === tests/test_curobo_ik_build_model.py ===

import importlib.util
import json
import re
import unittest
import zipfile
from pathlib import Path
from support.paths import REPO_ROOT

ROOT = REPO_ROOT
PACKAGE = ROOT / "ni_rua" / "robot" / "model" / "robot_information_data" / "so101"
spec = importlib.util.spec_from_file_location("curobo_ik_build_model", ROOT / "services" / "curobo_ik" / "build_model.py")
build_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_model)


def packaged_urdf():
    native = json.loads((PACKAGE / "physical.json").read_text(encoding="utf-8"))["geometry"]["native_model"]
    with zipfile.ZipFile(PACKAGE / f"{native['resource_id']}.zip") as archive:
        return archive.read(native["assembly_member"]).decode("utf-8")


def limits_of(text):
    out = {}
    for block in re.findall(r'<joint name="[^"]+" type="revolute">.*?</joint>', text, flags=re.S):
        name = re.search(r'name="([^"]+)"', block).group(1)
        lower, upper = re.search(r'lower="([-0-9.eE+]+)" upper="([-0-9.eE+]+)"', block).groups()
        out[name] = (float(lower), float(upper))
    return out


class ShrinkLimitsTests(unittest.TestCase):
    def setUp(self):
        self.urdf = packaged_urdf()
        self.original = limits_of(self.urdf)

    def test_module_imports_without_curobo(self):
        for name in ("curobo", "torch", "yaml"):
            self.assertNotIn(name, vars(build_model))

    def test_arm_joints_shrink_on_both_sides_gripper_untouched(self):
        text, limits = build_model.shrink_limits(self.urdf, 0.05)
        self.assertEqual(set(limits), set(build_model.ARM_JOINTS))
        shrunk = limits_of(text)
        self.assertIn("gripper", self.original)
        self.assertEqual(shrunk["gripper"], self.original["gripper"])
        for name in build_model.ARM_JOINTS:
            lower, upper = self.original[name]
            self.assertAlmostEqual(shrunk[name][0], lower + 0.05, places=9)
            self.assertAlmostEqual(shrunk[name][1], upper - 0.05, places=9)
            self.assertEqual(limits[name], list(shrunk[name]))

    def test_matches_planner_limits(self):
        from ni_rua.robot.model import so101_kinematics as kin
        _, limits = build_model.shrink_limits(self.urdf, 0.05)
        planner = kin.joint_limits_rad(kin.load_model())
        for name in build_model.ARM_JOINTS:
            self.assertAlmostEqual(limits[name][0], planner[name][0] + 0.05, places=6)
            self.assertAlmostEqual(limits[name][1], planner[name][1] - 0.05, places=6)

    def test_only_limit_attributes_change(self):
        text, _ = build_model.shrink_limits(self.urdf, 0.05)
        strip = lambda t: re.sub(r'lower="[^"]+" upper="[^"]+"', "", t)
        self.assertEqual(strip(text), strip(self.urdf))
        self.assertEqual(limits_of(text).keys(), self.original.keys())

    def test_zero_margin_is_identity_on_limits(self):
        _, limits = build_model.shrink_limits(self.urdf, 0.0)
        for name in build_model.ARM_JOINTS:
            self.assertEqual(tuple(limits[name]), self.original[name])

    def test_too_large_margin_exits(self):
        with self.assertRaises(SystemExit) as caught:
            build_model.shrink_limits(self.urdf, 10.0)
        self.assertIn("leaves no range", str(caught.exception))

    def test_urdf_without_arm_joints_exits(self):
        with self.assertRaises(SystemExit) as caught:
            build_model.shrink_limits("<robot></robot>", 0.05)
        self.assertIn("lacks arm joints", str(caught.exception))
        missing = re.sub(r'<joint name="wrist_roll" type="revolute">.*?</joint>', "", self.urdf, flags=re.S)
        with self.assertRaises(SystemExit) as caught:
            build_model.shrink_limits(missing, 0.05)
        self.assertIn("wrist_roll", str(caught.exception))

# === tests/test_embodiments.py ===

from dataclasses import FrozenInstanceError, fields
from pathlib import Path
import unittest

from ni_rua.robot.model.arx_contract import FRAME_ID, TCP_CONFIG_ID
from ni_rua.robot.model.embodiments import ARX_X5, RobotDescription, get_robodojo_robot


class EmbodimentTests(unittest.TestCase):
    def test_default_preserves_arx_control_and_deployment_paths(self):
        binding = get_robodojo_robot()
        self.assertIs(binding, get_robodojo_robot("arx-x5"))
        self.assertIs(binding.robot, ARX_X5)
        self.assertEqual(ARX_X5.frame_id, FRAME_ID)
        self.assertEqual(ARX_X5.tcp_config_id, TCP_CONFIG_ID)
        for field, path in {
            "environment_config_path": "env_cfg/arx_x5.yml",
            "urdf_path": "Assets/Robots/x5/X5A.urdf",
            "robot_config_path": "Assets/Robots/x5/robot_config.yml",
            "curobo_config_path": "Assets/Robots/x5/curobo.yml",
            "layout_directory": "Eval_Layout/RoboDojo/arx_x5",
        }.items():
            self.assertEqual(getattr(binding, field), Path(path))

    def test_unknown_or_malformed_selection_never_falls_back(self):
        for selection in ("unknown", "ARX-X5", "", None, [], {}):
            with self.subTest(selection=selection), self.assertRaisesRegex(
                ValueError, "unsupported_robodojo_robot",
            ):
                get_robodojo_robot(selection)

    def test_robot_description_has_no_environment_assets_and_binding_is_immutable(self):
        self.assertEqual({field.name for field in fields(RobotDescription)},
                         {"robot_model_id", "frame_id", "tcp_config_id"})
        with self.assertRaises(FrozenInstanceError):
            ARX_X5.tcp_config_id = "another-tcp"
        with self.assertRaises(FrozenInstanceError):
            get_robodojo_robot().urdf_path = Path("another.urdf")

# === tests/test_robot_information.py ===

from copy import deepcopy
from collections import Counter
import io
import json
import re
import struct
import unittest
import xml.etree.ElementTree as ET
import zipfile

from PIL import Image

from ni_rua.robot.model.robot_information import (
    ROBOT_INFORMATION_REGISTRY,
    ROBOT_INFORMATION_SECTIONS,
    robot_geometry_path,
    robot_information,
    robot_information_text,
    robot_resource_path,
    robot_resources,
)


def objects(value):
    yield value
    if isinstance(value, dict):
        for item in value.values():
            yield from objects(item)
    elif isinstance(value, list):
        for item in value:
            yield from objects(item)


class RobotInformationTests(unittest.TestCase):
    def test_full_reference_retains_complete_numerical_body(self):
        result = robot_information(section="all")
        self.assertTrue(result["available"])
        self.assertEqual(result["robot_id"], "arx-x5")
        self.assertGreater(len(result["briefing"]), 6000)
        for phrase in ("modified", "current RoboDojo episode calibration", "0.1501", "0.1226", "0.1576",
                       "qcmd7=qcmd8", "0.807 mm", "88.807 mm", "25 Hz", "0.05 rad",
                       "1.25 rad/s", "retained grasp", "measured state"):
            self.assertIn(phrase, result["briefing"])
        ref = result["reference"]
        self.assertEqual(set(ref), {"identity", "kinematics", "geometry", "dynamics",
                                    "control", "limitations", "hardware"})
        self.assertEqual(ref["identity"]["native_action_dimensions"], 14)
        joints = {j["name"]: j for j in ref["kinematics"]["joints"]}
        self.assertEqual(len(joints), 10)
        self.assertEqual(sum(j["type"] == "revolute" for j in joints.values()), 6)
        self.assertEqual(sum(j["type"] == "prismatic" for j in joints.values()), 2)
        self.assertEqual(joints["joint3"]["origin"],
                         {"xyz_m": [-.264, 0., 0.], "rpy_rad": [3.1416, 0., 0.]})
        self.assertEqual(joints["joint4"]["origin"]["xyz_m"], [.245, 0., -.056])
        self.assertEqual(joints["joint6"]["declared_limits"]["lower"], -3.14)
        self.assertEqual(joints["joint8"]["axis"], [0., -1., 0.])
        self.assertEqual(ref["kinematics"]["pose_interface"]["grasp_contact_offsets_m"],
                         {"pad_center": .1501, "finger_middle": .1226, "fingertip": .1576})

        dynamics = ref["dynamics"]
        links = {link["name"]: link["inertial"] for link in dynamics["declared_links"]}
        self.assertEqual(len(links), 11)
        self.assertIsNone(links["camera"])
        self.assertIsNone(links["camera_base"])
        self.assertAlmostEqual(sum(link["mass_kg"] for link in links.values() if link), 3.454278)
        self.assertEqual(links["link3"]["inertia_kg_m2"],
                         dict(ixx=.00036, ixy=-.00003, ixz=-.00039,
                              iyy=.00423, iyz=.00001, izz=.0042))
        for link in links.values():
            if link is not None:
                self.assertEqual(set(link["inertia_kg_m2"]), {"ixx", "ixy", "ixz", "iyy", "iyz", "izz"})
                self.assertEqual(len(link["origin"]["xyz_m"]), 3)
        self.assertEqual(len(dynamics["authored_rigid_bodies"]), 11)
        self.assertEqual(len(dynamics["authored_joints"]), 10)
        self.assertEqual(len(dynamics["authored_rigid_bodies"]["camera"]["diagonal_inertia_kg_m2"]), 3)

    def test_geometry_retains_mesh_bounds_gap_and_all_planner_spheres(self):
        geometry = robot_information(section="geometry")["reference"]["geometry"]
        links = {link["name"]: link for link in geometry["links"]}
        self.assertEqual(len(links), 11)
        self.assertAlmostEqual(links["link2"]["visual_and_collision"][0]["mesh_local_min_m"][0], -.292)
        gap = geometry["distal_flat_pad_gap"]
        self.assertAlmostEqual(gap["gap_q7_q8_zero_m"], .000807165976524353)
        self.assertAlmostEqual(gap["gap_q7_q8_0_044_m"], .08880716597652435)
        self.assertIn("actual physical joint positions", gap["q7_q8_units"])
        self.assertEqual(geometry["collision_approximations"]["link3"]["approximation"], "convexDecomposition")
        self.assertTrue(geometry["collision_approximations"]["camera"]["enabled"])
        spheres = geometry["planning_collision"]["collision_spheres"]
        self.assertEqual({k: len(v) for k, v in spheres.items()},
                         dict(base_link=3, camera=3, camera_base=6, link1=13,
                              link2=12, link3=13, link4=4, link5=3, link6=8))
        self.assertAlmostEqual(spheres["base_link"][0]["radius"], .043082091957330704)
        self.assertEqual(set(geometry["finger_geometry"]), {"link7", "link8"})

    def test_default_and_authored_settings_are_not_conflated_with_measurements(self):
        ref = robot_information(section="all")["reference"]
        dynamics = ref["dynamics"]
        authored = dynamics["authored_articulation_defaults"]
        configured = dynamics["configured_actuators_and_physics"]
        self.assertFalse(authored["enabled_self_collisions"])
        self.assertTrue(configured["enabled_self_collisions"])
        self.assertEqual(authored["solver_position_iteration_count"], 32)
        self.assertEqual(configured["solver_position_iteration_count"], 8)
        self.assertEqual(configured["arm"]["velocity_limit_sim"], 5.)
        self.assertEqual(configured["arm"]["stiffness"], 4400.)
        self.assertEqual(configured["gripper"]["damping"], 100.)
        self.assertTrue(configured["disable_gravity"])
        control = ref["control"]
        self.assertEqual(control["default_harness"]["physics_dt_s"], .004)
        self.assertEqual(control["default_harness"]["native_control_hz"], 25.)
        self.assertEqual(control["default_harness"]["default_resampled_command_rate_bound_rad_s"], 1.25)
        self.assertEqual(control["gripper"]["native_target_endpoints_m"], [-.01, .044])
        preview = control["preview_gripper_mapping"]
        self.assertNotEqual(preview["preview_command_to_joint_m"], preview["native_command_to_joint_m"])
        self.assertIn("measured", preview["interpretation"])
        self.assertEqual(control["default_spawn_joint_positions"]["joint7"], .044)
        self.assertAlmostEqual(control["planner_joint_space"]["default_joint_position"][-1], .022)

    def test_every_section_is_free_of_provenance_and_file_metadata(self):
        forbidden = re.compile(r"(?:^|_)(?:source|provenance|sha\d+|hash|fingerprint|commit|repository|"
                               r"filename|file_path|prim_path|schema_version|format_version|"
                               r"observed_date|observed_utc|triangle_count|vertex_count|bytes|"
                               r"archived_file|saved_as|url)", re.I)
        for robot_id in ROBOT_INFORMATION_REGISTRY:
            for section in ROBOT_INFORMATION_SECTIONS:
                with self.subTest(robot_id=robot_id, section=section):
                    result = robot_information(robot_id, section)
                    members = {obj["assembly_member"] for obj in objects(result)
                               if isinstance(obj, dict) and "assembly_member" in obj}
                    for obj in objects(result):
                        if isinstance(obj, dict):
                            for key in obj:
                                self.assertIsNone(forbidden.search(key), key)
                        elif isinstance(obj, str):
                            self.assertNotRegex(obj, r"(?i)\b[0-9a-f]{32,}\b|https?://|/home/|/tmp/|/X5A/")
                            if obj not in members and obj != result.get("urdf"):
                                self.assertNotRegex(obj, r"(?i)\b\w+\.(?:py|json|txt|urdf|usd|stl|yml|yaml|pdf|step)\b")
                    self.assertNotRegex(json.dumps(result), r"(?i)\bsha256\b|\bfingerprint\b|\bprovenance\b")

    def test_unknown_robot_and_section_are_explicitly_unavailable(self):
        for robot_id in ("unknown", "x5", "arx-x5-2025", "../arx_x5", "", None, []):
            with self.subTest(robot_id=robot_id):
                result = robot_information(robot_id)
                self.assertFalse(result["available"])
                self.assertEqual(result["error"], "robot_information_unavailable")
                self.assertNotIn("reference", result)
                self.assertNotIn("briefing", result)
                self.assertIsNone(robot_geometry_path(robot_id))
        for section in ("missing", "../physical", None, []):
            with self.subTest(section=section):
                result = robot_information(section=section)
                self.assertFalse(result["available"])
                self.assertNotIn("reference", result)

    def test_partial_sections_all_and_deep_copies(self):
        default = robot_information()
        full = robot_information(section="all")
        for section in ("kinematics", "geometry", "dynamics", "control"):
            partial = robot_information(section=section)
            self.assertEqual(set(partial["reference"]), {section})
            self.assertEqual(partial["reference"][section], full["reference"][section])
            self.assertNotIn("briefing", partial)
        self.assertEqual(full["briefing"], default["briefing"])
        self.assertEqual(full["reference"]["hardware"],
                         robot_information(section="hardware")["reference"]["hardware"])
        self.assertNotIn("hardware", default["reference"])
        baseline = deepcopy(full)
        full["reference"]["kinematics"]["joints"][0]["axis"][0] = 123
        full["reference"]["geometry"]["planning_collision"]["collision_spheres"]["link1"].clear()
        full["reference"]["hardware"]["product_families"].clear()
        self.assertEqual(robot_information(section="all"), baseline)
        self.assertEqual(robot_information(), default)

    def test_default_keeps_briefing_native_geometry_and_one_physical_representation(self):
        default = robot_information()
        full = robot_information(section="all")
        self.assertEqual(default["briefing"], full["briefing"])
        compact = default["reference"]
        detailed = full["reference"]
        self.assertEqual(set(compact), {"identity", "kinematics", "geometry", "control"})
        self.assertEqual(compact["geometry"], detailed["geometry"]["native_model"])
        self.assertEqual(compact["geometry"]["resource_id"], "geometry_model")
        self.assertNotIn("joints", compact["kinematics"])
        with zipfile.ZipFile(robot_resource_path(resource_id="geometry_model")) as archive:
            self.assertEqual(default["urdf"].encode("utf-8"), archive.read("X5A.urdf"))
        self.assertEqual(len(ET.fromstring(default["urdf"]).findall("joint")), 10)
        control = compact["control"]
        self.assertEqual(control["configured_actuators_and_physics"],
                         detailed["dynamics"]["configured_actuators_and_physics"])
        self.assertEqual(control["gripper"]["native_joint_target_m"],
                         detailed["control"]["gripper"]["native_joint_target_m"])
        self.assertEqual(compact["kinematics"]["pose_interface"]["grasp_contact_offsets_m"],
                         dict(pad_center=.1501, finger_middle=.1226, fingertip=.1576))
        # Alternate unit/precision representations and raw numerical inventories
        # must not creep back into the initial JSON. They are available on demand.
        forbidden = {"links", "joint_units", "declared_position_limits", "declared_limits",
                     "axis", "origin", "inertial", "visual_and_collision",
                     "authored_joints", "authored_rigid_bodies", "declared_links", "planning_collision",
                     "native_frame_matrices", "finger_geometry", "pad_center_offset_m", "tool_offset_m",
                     "native_gripper_command", "native_target_endpoints_m", "limit_units", "default_mounts"}
        for obj in objects(compact):
            if isinstance(obj, dict):
                self.assertFalse(set(obj) & forbidden)
        compact["geometry"].clear()
        compact["kinematics"]["pose_interface"]["grasp_contact_offsets_m"]["pad_center"] = 999
        control["gripper"].clear()
        self.assertEqual(robot_information()["reference"]["geometry"], detailed["geometry"]["native_model"])
        self.assertEqual(robot_information()["reference"]["kinematics"]["pose_interface"]["grasp_contact_offsets_m"]["pad_center"], .1501)

    def test_startup_text_delivers_verbatim_urdf_once_without_mutation(self):
        record = robot_information()
        original = deepcopy(record)
        text = robot_information_text(record)
        metadata, urdf = text.removeprefix("Robot physical information\n").split("\n\nRobot URDF\n", 1)
        self.assertEqual(urdf, record["urdf"])
        self.assertEqual(text.count(record["urdf"]), 1)
        self.assertNotIn("urdf", json.loads(metadata))
        self.assertEqual(json.loads(metadata), {k: v for k, v in record.items() if k != "urdf"})
        self.assertEqual(record, original)
        unknown = robot_information("unregistered")
        self.assertNotIn("Robot URDF", robot_information_text(unknown))
        self.assertNotIn("urdf", unknown)

    def test_geometry_illustration_is_packaged_without_metadata(self):
        path = robot_geometry_path()
        self.assertIsNotNone(path)
        self.assertEqual(path.parent.name, "arx_x5")
        with Image.open(path) as illustration:
            self.assertEqual(illustration.format, "PNG")
            self.assertGreaterEqual(illustration.width, 1000)
            self.assertEqual(illustration.info, {})

    def test_optional_component_resources_have_no_model_facing_paths(self):
        catalog = robot_resources()
        self.assertTrue(catalog["available"])
        expected_solids = dict(base_step=1, end_step=3, gripper_1_step=1, gripper_2_step=1)
        self.assertEqual({entry["resource_id"] for entry in catalog["resources"]}, set(expected_solids) | {"geometry_model"})
        for entry in catalog["resources"]:
            self.assertEqual(set(entry), {"resource_id", "name", "format", "description"})
            if entry["resource_id"] == "geometry_model":
                self.assertEqual(entry["format"], "ZIP")
                continue
            self.assertIn("not a complete arm", entry["description"])
            self.assertIn("do not represent the modified simulated fingers", entry["description"])
            path = robot_resource_path(resource_id=entry["resource_id"])
            self.assertIsNotNone(path)
            data = path.read_bytes()
            self.assertTrue(data.startswith(b"ISO-10303-21;"))
            self.assertTrue(data.rstrip().endswith(b"END-ISO-10303-21;"))
            self.assertEqual(data.count(b"MANIFOLD_SOLID_BREP"), expected_solids[entry["resource_id"]])
            # A truncated or damaged transfer must not leave dangling entity references.
            defined = set(re.findall(rb"#(\d+)\s*=", data))
            referenced = set(re.findall(rb"#(\d+)", data))
            self.assertEqual(defined, referenced)
        encoded = json.dumps(catalog)
        self.assertNotRegex(encoded, r"(?i)https?://|/home/|sha256|\b[0-9a-f]{32,}\b|\.STEP")
        catalog["resources"][0]["description"] = "changed"
        self.assertNotEqual(robot_resources(), catalog)
        for robot_id in ("unknown", "x5", None, []):
            self.assertFalse(robot_resources(robot_id)["available"])
            self.assertIsNone(robot_resource_path(robot_id, "base_step"))
        for resource_id in ("unknown", "../../physical", "base_step.STEP", None, []):
            self.assertIsNone(robot_resource_path(resource_id=resource_id))

    def test_cad_documentation_is_removed_without_losing_geometry_or_units(self):
        # Recorded physical entity counts also detect incomplete component transfers.
        kinds = ("CARTESIAN_POINT", "DIRECTION", "EDGE_CURVE", "ADVANCED_FACE",
                 "VERTEX_POINT", "CLOSED_SHELL", "MANIFOLD_SOLID_BREP")
        expected = {
            "base_step": (3446, (597, 646, 289, 102, 188, 1, 1)),
            "end_step": (11789, (2266, 2173, 938, 384, 591, 3, 3)),
            "gripper_1_step": (1573, (268, 271, 130, 51, 86, 1, 1)),
            "gripper_2_step": (1573, (268, 271, 130, 51, 86, 1, 1)),
        }
        documentary = {
            "APPROVAL", "APPROVAL_DATE_TIME", "APPROVAL_PERSON_ORGANIZATION",
            "APPROVAL_ROLE", "APPROVAL_STATUS", "CALENDAR_DATE", "CC_DESIGN_APPROVAL",
            "CC_DESIGN_DATE_AND_TIME_ASSIGNMENT", "CC_DESIGN_PERSON_AND_ORGANIZATION_ASSIGNMENT",
            "CC_DESIGN_SECURITY_CLASSIFICATION", "COORDINATED_UNIVERSAL_TIME_OFFSET",
            "DATE_AND_TIME", "DATE_TIME_ROLE", "LOCAL_TIME", "ORGANIZATION", "PERSON",
            "PERSON_AND_ORGANIZATION", "PERSON_AND_ORGANIZATION_ROLE",
            "SECURITY_CLASSIFICATION", "SECURITY_CLASSIFICATION_LEVEL",
        }
        for resource_id, (total, physical_counts) in expected.items():
            with self.subTest(resource_id=resource_id):
                text = robot_resource_path(resource_id=resource_id).read_text(encoding="ascii")
                self.assertIn("FILE_NAME ('', '', (''), (''), '', '', '');", text)
                self.assertIn("FILE_SCHEMA (( 'CONFIG_CONTROL_DESIGN' ));", text)
                self.assertNotRegex(text, r"\d{4}-\d{2}-\d{2}T|https?://|[A-Z]:\\|/home/|/tmp/")
                entities = re.findall(r"(?m)^#\d+\s*=", text)
                counts = Counter(re.findall(r"(?m)^#\d+\s*=\s*([A-Z_0-9]+)", text))
                self.assertEqual(len(entities), total)
                self.assertEqual(tuple(counts[kind] for kind in kinds), physical_counts)
                self.assertFalse(set(counts) & documentary)
                self.assertEqual(text.count("SI_UNIT ( .MILLI., .METRE. )"), 1)
                self.assertEqual(text.count("SI_UNIT ( $, .RADIAN. )"), 1)
                self.assertEqual(text.count("SI_UNIT ( $, .STERADIAN. )"), 1)
                labels = set(re.findall(r"'(?:[^']|'')*'", text))
                allowed = {
                    "''", "'NONE'", "'1'", "'STEP AP203'", "'CONFIG_CONTROL_DESIGN'",
                    "'config_control_design'", "'international standard'", "'WORKASPACE'",
                    "'configuration controlled 3d designs of mechanical parts and assemblies'",
                    "'design'", "'detail'", "'detailed design'", "'distance_accuracy_value'",
                    "'mechanical'", "'solid_1'", "'solid_2'", "'solid_3'",
                    "'" + resource_id.removesuffix("_step") + "'",
                }
                self.assertFalse(labels - allowed, labels - allowed)

    def test_native_geometry_archive_contains_every_assembly_mesh(self):
        path = robot_resource_path(resource_id="geometry_model")
        self.assertIsNotNone(path)
        self.assertLess(path.stat().st_size, 16*1024*1024)
        with zipfile.ZipFile(path) as archive:
            self.assertIsNone(archive.testzip())
            urdf = archive.read("X5A.urdf")
            self.assertNotIn(b"<!--", urdf)
            root = ET.fromstring(urdf)
            links = {link.get("name") for link in root.findall("link")}
            self.assertEqual(links, {"base_link", "camera", "camera_base"} | {f"link{i}" for i in range(1, 9)})
            self.assertEqual(len(root.findall("joint")), 10)
            meshes = {node.get("filename") for node in root.findall(".//mesh")}
            self.assertEqual(set(archive.namelist()), meshes | {"X5A.urdf"})
            for member in meshes:
                self.assertTrue(member.startswith("meshes/") and ".." not in member)
                data = archive.read(member)
                if member.endswith(".STL"):
                    self.assertEqual(data[:80], b"\0"*80)
                    count = struct.unpack_from("<I", data, 80)[0]
                    self.assertEqual(len(data), 84 + 50*count)
                    self.assertGreater(count, 0)
                else:
                    magic, version, length, json_size, kind = struct.unpack_from("<5I", data)
                    self.assertEqual((magic, version, length, kind), (0x46546c67, 2, len(data), 0x4e4f534a))
                    model = json.loads(data[20:20+json_size])
                    self.assertNotIn("generator", model["asset"])
                    self.assertNotIn("copyright", model["asset"])
                    self.assertFalse(any("uri" in buffer for buffer in model["buffers"]))
            joints = {joint.get("name"): joint for joint in root.findall("joint")}
            self.assertEqual(float(joints["joint3"].find("origin").get("xyz").split()[0]), -.264)
            self.assertEqual(float(joints["joint7"].find("limit").get("upper")), .044)

# === tests/test_trajectory_evidence.py ===

from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from ni_rua.robot.model.geometry import fk
from ni_rua.robot.motion.trajectory_evidence import build_trajectory_evidence, compact_trajectory_evidence, nominal_timeline


def hardware():
    joints=[]
    for i in range(1,7):
        t=np.eye(4);t[0,3]=.1
        joints.append(dict(name=f'joint{i}',type='revolute',parent='base_link' if i==1 else f'link{i-1}',
                           child=f'link{i}',axis=[0,0,1],T_parent_joint=t.tolist()))
    for i in (7,8):
        joints.append(dict(name=f'joint{i}',type='prismatic',parent='link6',child=f'link{i}',
                           axis=[0,1 if i==7 else -1,0],T_parent_joint=np.eye(4).tolist()))
    return dict(model=dict(base_link='base_link',joints=joints,source_sha256='public-fixture'),
        spheres={f'link{i}':[dict(center=[0,0,0],radius=.03)] for i in range(1,7)},sphere_buffer_m=0,
        jaws=dict(gripper_scale=[-.01,.044],gripper_move=dict(base='joint7',sign=1,mimic=['joint8',1.,0.]),
                  gripper_joints_name=['joint7','joint8']),provenance={'kind':'public-hardware-fixture'})


def public_fixture(separation=3.):
    h=hardware();state={'arms':{}}
    for arm,base in [('left',0),('right',separation)]:
        state['arms'][arm]=dict(source_kind='measurement',joint_names=[f'joint{i}' for i in range(1,7)],q_rad=[0.]*6,
            ee_pose=dict(frame_id='env_origin',tcp_config_id='arx-x5-link6-v1',source_kind='measurement',
                         position_m=[base+.6,0,0],quaternion_xyzw=[0,0,0,1]),
            gripper=dict(joint_names=['joint7','joint8'],position_source_kind='measurement',position_m=[.044,.044]))
    adapter=SimpleNamespace(control_hz=25,last_dispatched_goals={'left':1.,'right':1.})
    row=[0.]*6+[1.]+[0.]*6+[1.]
    compiled={a:(np.zeros((2,6)),[list(row),list(row)],[],None) for a in ('left','right')}
    action=dict(frame_id='env_origin',tcp_config_id='arx-x5-link6-v1')
    return adapter,state,compiled,action,h


class TrajectoryEvidenceTests(unittest.TestCase):
    def test_public_fk_bases_and_model_near_approach_have_witness_not_collision_claim(self):
        for separation,expected in [(3.,'no_near_approach_at_sampled_boundaries'),(.5,'model_near_approach')]:
            adapter,state,compiled,action,h=public_fixture(separation)
            # No env, articulation or planner attribute exists on this adapter.
            with patch('ni_rua.robot.motion.trajectory_evidence._hardware',return_value=h):
                short,full=build_trajectory_evidence(adapter,state,compiled,'paired',action)
            self.assertEqual(short['geometry']['status'],'available')
            self.assertEqual(short['inter_arm_proximity']['status'],expected)
            self.assertTrue(short['inter_arm_proximity']['missing_finger_coverage'])
            self.assertAlmostEqual(full['T_world_bases']['right'][0][3],separation)
            self.assertAlmostEqual(full['samples'][0]['arms']['left']['link_positions_m']['link6'][0],.6)
            self.assertIn('not actual collision',short['inter_arm_proximity']['semantics'])
            self.assertEqual(len(full['samples']),3)

    def test_missing_relative_calibration_returns_unknown_without_fallback(self):
        adapter,state,compiled,action,h=public_fixture()
        state['arms']['right']['ee_pose']['frame_id']='right_local_unregistered'
        with patch('ni_rua.robot.motion.trajectory_evidence._hardware',return_value=h):
            short,full=build_trajectory_evidence(adapter,state,compiled,'paired',action)
        self.assertEqual(short['geometry']['status'],'unknown')
        self.assertEqual(short['inter_arm_proximity']['status'],'unknown')
        self.assertNotIn('link_positions_m',full['samples'][0]['arms']['left'])
        self.assertNotIn('link6_pose',full['samples'][0]['arms']['left'])
        self.assertEqual(short['anomalies'][0]['kind'],'public_geometry_unknown')

    def test_missing_spheres_preserves_fk_but_proximity_unknown(self):
        adapter,state,compiled,action,h=public_fixture();h['spheres']=None
        with patch('ni_rua.robot.motion.trajectory_evidence._hardware',return_value=h):
            short,full=build_trajectory_evidence(adapter,state,compiled,'paired',action)
        self.assertEqual(short['geometry']['status'],'available')
        self.assertEqual(short['inter_arm_proximity']['status'],'unknown')
        self.assertIn('link_positions_m',full['samples'][-1]['arms']['left'])
        self.assertEqual(short['anomalies'][0]['kind'],'inter_arm_proximity_unknown')

    def test_compact_evidence_bounds_dense_near_path_and_preserves_exact_artifact(self):
        adapter,state,compiled,action,h=public_fixture(.5)
        for arm in compiled:
            positions,motion,jaws,schedule=compiled[arm]
            compiled[arm]=(positions,motion*40,jaws,schedule)
        with patch('ni_rua.robot.motion.trajectory_evidence._hardware',return_value=h):
            short,full=build_trajectory_evidence(adapter,state,compiled,'paired',action)
        original=deepcopy(full)
        self.assertEqual(short,compact_trajectory_evidence(full))
        self.assertEqual(full,original)
        self.assertLessEqual(len(short['key_poses']),6)
        self.assertEqual(short['q_unit'],'rad')
        self.assertEqual(short['joint_names'],full['joint_names'])
        self.assertEqual(len(full['samples']),81)
        self.assertEqual(short['inter_arm_proximity']['near_boundary_count'],81)
        self.assertEqual(full['inter_arm_proximity']['near_native_steps'],list(range(81)))
        self.assertEqual(short['anomalies'],[dict(kind='model_near_approach',native_step=0)])
        for forbidden in ('link_positions_m','center_m','radius_m','near_native_steps','samples'):
            self.assertNotIn('"'+forbidden+'"',json.dumps(short))
        for selected in short['key_poses']:
            for arm in ('left','right'):
                self.assertEqual(selected['arms'][arm]['q_rad'],full['samples'][selected['native_step']]['arms'][arm]['q_rad'])
        pose=short['key_poses'][0]['arms']['left']['link6_pose']
        self.assertEqual(pose,dict(position_m=[.6,0.,0.],rotation_matrix=np.eye(3).tolist()))
        self.assertIn('center_m',full['inter_arm_proximity']['witness']['left'])
        self.assertEqual(full['samples'][0]['arms']['left']['q_rad'],state['arms']['left']['q_rad'])


class KinematicsPinkAndDlsFallbackTests(unittest.TestCase):
    """Verify ``Pinocchio + Pink + DAQP`` primary path and analytical-Jacobian LM fallback parity."""

    VBOOL_MSG = "No to_python (by-value) converter found for C++ type: std::vector<bool, std::allocator<bool> >"
    PRIMAL_MSG = "solve() got an unexpected keyword argument 'primal_start'"

    def setUp(self) -> None:
        from ni_rua.robot.embodiments.arx_x5 import arx_x5_chain
        from ni_rua.robot.model import kinematics as kin

        self.kin = kin
        self.x5 = arx_x5_chain()
        orig_cache = kin._PINK_DAQP_CACHE
        self.addCleanup(setattr, kin, "_PINK_DAQP_CACHE", orig_cache)

    def test_pink_daqp_primary_and_dls_fallbacks_converge_sub_0_1mm_on_x5_and_7dof(self) -> None:
        from contextlib import ExitStack

        kin = self.kin
        x5 = self.x5
        q_true = np.array([0.18, 0.72, 0.84, -0.45, 0.26, -0.35], dtype=np.float64)
        q_seed = q_true + np.array([0.09, -0.08, 0.07, -0.09, 0.08, -0.10], dtype=np.float64)
        T_goal = kin.fk_links(x5, q_true)[x5.tip_link]
        posture_pref = q_true.copy()

        kin._PINK_DAQP_CACHE = None
        self.assertTrue(kin._pink_daqp_usable())
        self.assertEqual(kin.ik_backend(), "pink_daqp")

        modes = [
            ("pink_daqp_primary", ()),
            ("probe_false", (patch("ni_rua.robot.model.kinematics._pink_daqp_usable", return_value=False),)),
            (
                "pink_configuration_vbool",
                (patch.object(kin, "_PINK_DAQP_CACHE", True), patch("pink.Configuration", side_effect=TypeError(self.VBOOL_MSG))),
            ),
            (
                "pink_limit_vbool",
                (patch.object(kin, "_PINK_DAQP_CACHE", True), patch("pink.limits.ConfigurationLimit", side_effect=TypeError(self.VBOOL_MSG))),
            ),
            (
                "daqp_primal_start",
                (patch.object(kin, "_PINK_DAQP_CACHE", True), patch("pink.solve_ik", side_effect=TypeError(self.PRIMAL_MSG))),
            ),
        ]

        for mode_name, patches in modes:
            for tasks in (("pose",), ("pose", "posture"), ("position", "orientation")):
                with self.subTest(mode=mode_name, tasks=tasks), ExitStack() as stack:
                    for p in patches:
                        stack.enter_context(p)
                    res = kin.solve_ik(
                        x5,
                        kin.IKTarget(T_base_tip=T_goal, posture_rad=posture_pref),
                        q_seed,
                        kin.IKOptions(
                            tasks=tasks,
                            position_tolerance_m=5e-5,
                            orientation_tolerance_rad=5e-4,
                            max_iterations=120,
                            restarts=2,
                        ),
                    )
                    self.assertTrue(res.converged, (mode_name, tasks, res.as_dict()))
                    self.assertLess(res.position_error_m, 1e-4)
                    self.assertLess(res.orientation_error_rad, 1e-3)
                    self.assertGreater(res.iterations, 0)
                    self.assertTrue(np.all(res.q_rad >= x5.joint_limits_rad[:, 0] - 1e-9))
                    self.assertTrue(np.all(res.q_rad <= x5.joint_limits_rad[:, 1] + 1e-9))
                    if mode_name in ("pink_configuration_vbool", "pink_limit_vbool", "daqp_primal_start"):
                        self.assertIs(kin._PINK_DAQP_CACHE, False)

            with self.subTest(mode=mode_name, api="solve_ik_batch"), ExitStack() as stack:
                for p in patches:
                    stack.enter_context(p)
                rng = np.random.default_rng(11)
                seeds = np.vstack([q_seed, q_true + rng.uniform(-0.22, 0.22, size=(5, 6))])
                qb = kin.solve_ik_batch(
                    x5,
                    T_goal[:3, 3],
                    T_goal[:3, :3],
                    seeds,
                    pos_tol_m=1e-4,
                    rot_tol_rad=1e-3,
                    max_iters=40,
                    posture_rad=posture_pref,
                    posture_weight=0.05,
                )
                self.assertEqual(qb.shape, (6, 6))
                T_ach = kin.fk_links(x5, qb[0])[x5.tip_link]
                pe, oe = kin.pose_error(T_ach, T_goal)
                self.assertLess(pe, 1e-4)
                self.assertLess(oe, 1e-3)

        self.assertTrue(kin._pink_daqp_usable())

        # Verify 7-DoF redundant null-space posture resolution under forced DLS fallback.
        joints7 = [
            {
                "name": f"j{i}",
                "type": "revolute",
                "parent": "base_link" if i == 0 else f"link{i}",
                "child": f"link{i + 1}",
                "T_parent_joint": kin.rpy_xyz_to_matrix(
                    (0.12 if i > 0 else 0.0, 0.0, 0.15 if i == 0 else 0.0), (0.0, 0.0, 0.0)
                ).tolist(),
                "axis": [[0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]][i],
                "limits": [-2.8, 2.8],
            }
            for i in range(7)
        ]
        joints7.append(
            {
                "name": "tip_fixed",
                "type": "fixed",
                "parent": "link7",
                "child": "tcp_link",
                "T_parent_joint": kin.rpy_xyz_to_matrix((0.08, 0.0, 0.0), (0.0, 0.0, 0.0)).tolist(),
            }
        )
        chain7 = kin.chain_from_manifest({"base_link": "base_link", "joints": joints7}, tip_link="tcp_link")
        q_ref = np.array([0.2, -0.4, 0.35, -0.9, 0.25, 0.5, -0.3], dtype=np.float64)
        t_goal7 = kin.fk_links(chain7, q_ref)["tcp_link"]
        posture_a = np.array([-0.8, -0.4, -0.9, -0.9, 0.25, 0.5, -0.3], dtype=np.float64)
        posture_b = np.array([0.8, -0.4, 0.9, -0.9, 0.25, 0.5, -0.3], dtype=np.float64)
        opts7 = kin.IKOptions(
            tasks=("pose", "posture"),
            posture_weight=0.25,
            max_iterations=220,
            position_tolerance_m=1e-4,
            orientation_tolerance_rad=1e-3,
            restarts=2,
        )
        with patch("ni_rua.robot.model.kinematics._pink_daqp_usable", return_value=False):
            res_a = kin.solve_ik(chain7, kin.IKTarget(T_base_tip=t_goal7, posture_rad=posture_a), q_ref, opts7)
            res_b = kin.solve_ik(chain7, kin.IKTarget(T_base_tip=t_goal7, posture_rad=posture_b), q_ref, opts7)
        self.assertTrue(res_a.converged and res_b.converged)
        self.assertLess(res_a.position_error_m, 1e-4)
        self.assertLess(res_b.position_error_m, 1e-4)
        self.assertGreater(float(np.linalg.norm(res_a.q_rad - res_b.q_rad)), 0.15)
        self.assertLess(float(np.linalg.norm(res_a.q_rad - posture_a)), float(np.linalg.norm(res_b.q_rad - posture_a)))

    def test_unrelated_type_error_propagation_1dof_5dof_prismatic_and_active_set_qp(self) -> None:
        import math
        import sys
        from ni_rua.robot.model import so101_kinematics as s101

        kin = self.kin
        x5 = self.x5
        q_ref = np.array([0.15, 0.65, 0.75, -0.35, 0.20, -0.25], dtype=np.float64)
        T_goal = kin.fk_links(x5, q_ref)[x5.tip_link]

        # 1. Unrelated TypeError must propagate in Configuration, solve_ik inner loop, AND cold capability probe.
        for patch_target in ("pink.Configuration", "pink.solve_ik"):
            with patch.object(kin, "_PINK_DAQP_CACHE", True), patch(patch_target, side_effect=TypeError("unrelated_caller_bug")):
                with self.assertRaisesRegex(TypeError, "unrelated_caller_bug"):
                    kin.solve_ik(x5, T_goal, q_ref + 0.05)
                with self.assertRaisesRegex(TypeError, "unrelated_caller_bug"):
                    kin.solve_ik_batch(x5, T_goal[:3, 3], T_goal[:3, :3], (q_ref + 0.05)[None, :])

        def _plain_broken_config(*_args, **_kwargs):
            raise TypeError("plain_cold_probe_bug")

        with patch.object(kin, "_PINK_DAQP_CACHE", None), patch("pink.Configuration", new=_plain_broken_config):
            with self.assertRaisesRegex(TypeError, "plain_cold_probe_bug"):
                kin._pink_daqp_usable()
            self.assertIsNone(kin._PINK_DAQP_CACHE)

        import qpsolvers

        with patch.object(kin, "_PINK_DAQP_CACHE", None), patch(
            "pink.solve_ik", side_effect=qpsolvers.exceptions.SolverNotFound("daqp missing")
        ):
            self.assertFalse(kin._pink_daqp_usable())
        with patch.object(kin, "_PINK_DAQP_CACHE", None), patch(
            "pink.solve_ik", side_effect=qpsolvers.exceptions.ProblemError("bad_qp_setup")
        ):
            with self.assertRaisesRegex(qpsolvers.exceptions.ProblemError, "bad_qp_setup"):
                kin._pink_daqp_usable()
            self.assertIsNone(kin._PINK_DAQP_CACHE)

        # 2. Active-set box QP with KKT constraint release solves coupled & ill-conditioned 5D..7D QPs to exact KKT optimum.
        H_coupled = np.array([[1.0, 3.0], [3.0, 10.0]], dtype=np.float64)
        g_coupled = np.array([0.0, -1.0], dtype=np.float64)
        dq_opt = kin._box_lm_step(
            H_coupled,
            g_coupled,
            np.array([-1.0, -0.8], dtype=np.float64),
            np.array([1.0, 0.1], dtype=np.float64),
            1e-6,
            1.0,
        )
        np.testing.assert_allclose(dq_opt, [1.0, -0.4], atol=1e-4)
        A_coupled = H_coupled + 1e-6 * np.eye(2)
        self.assertLess(float(0.5 * (dq_opt @ (A_coupled @ dq_opt)) - (g_coupled @ dq_opt)), -0.299)

        rng_qp = np.random.default_rng(1234)
        for dof_qp in (5, 6, 7):
            for _ in range(50):
                M = rng_qp.standard_normal((dof_qp, dof_qp)) * np.array([10.0 ** (-i * 0.6) for i in range(dof_qp)])
                H_qp = M.T @ M
                g_qp = rng_qp.uniform(-2.0, 2.0, size=dof_qp)
                lo_qp = -rng_qp.uniform(0.0, 0.5, size=dof_qp)
                hi_qp = rng_qp.uniform(0.0, 0.5, size=dof_qp)
                cap_qp = float(rng_qp.uniform(0.05, 0.4))
                lam_qp = 1e-6
                x_qp = kin._box_lm_step(H_qp, g_qp, lo_qp, hi_qp, lam_qp, cap_qp)
                lo_eff = np.maximum(lo_qp, -cap_qp)
                hi_eff = np.minimum(hi_qp, cap_qp)
                self.assertTrue(np.all(x_qp >= lo_eff - 1e-12) and np.all(x_qp <= hi_eff + 1e-12))
                r_qp = g_qp - ((H_qp + lam_qp * np.eye(dof_qp)) @ x_qp)
                for j in range(dof_qp):
                    if hi_eff[j] - lo_eff[j] <= 1e-15:
                        continue
                    if x_qp[j] <= lo_eff[j] + 1e-11:
                        self.assertLessEqual(float(r_qp[j]), 1e-9)
                    elif x_qp[j] >= hi_eff[j] - 1e-11:
                        self.assertGreaterEqual(float(r_qp[j]), -1e-9)
                    else:
                        self.assertAlmostEqual(float(r_qp[j]), 0.0, places=9)

        # Locked joints (lower == upper) return exact bound without division-by-zero.
        locked_chain = x5.with_limits(np.column_stack([q_ref, q_ref]))
        with patch("ni_rua.robot.model.kinematics._pink_daqp_usable", return_value=False):
            locked_res = kin.solve_ik(locked_chain, T_goal, q_ref)
        self.assertTrue(locked_res.converged)
        np.testing.assert_allclose(locked_res.q_rad, q_ref, atol=1e-12)

        # 3. 1-DoF chain with Pinocchio installed AND forced DLS fallback (EigenPy 1D Jacobian normalization) + wrap_then_clamp.
        one_dof = kin.KinematicChainSpec(
            joint_names=("j1",),
            joint_types=("continuous",),
            parent_transforms=np.eye(4, dtype=np.float64)[None, :, :],
            joint_axes=np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
            joint_limits_rad=np.array([[-math.pi, math.pi]], dtype=np.float64),
            tip_transform=kin.rpy_xyz_to_matrix((0.30, 0.0, 0.0), (0.0, 0.0, 0.0)),
            base_link="base_link",
            tip_link="tip_link",
            link_names=("link1",),
        )
        T_18 = kin.fk_links(one_dof, [0.18])["tip_link"]
        _, jac_1d = kin.fk_jacobian(one_dof, [0.0])
        self.assertEqual(jac_1d.shape, (6, 1))
        with patch("ni_rua.robot.model.kinematics._pink_daqp_usable", return_value=False):
            res_1d = kin.solve_ik(
                one_dof,
                T_18,
                [0.0],
                kin.IKOptions(max_iterations=200, max_step_rad=0.01, limit_policy="wrap_then_clamp", position_tolerance_m=1e-4, orientation_tolerance_rad=1e-3),
            )
            qb_1d = kin.solve_ik_batch(one_dof, T_18[:3, 3], T_18[:3, :3], np.array([[0.0]]), pos_tol_m=1e-4, rot_tol_rad=1e-3, max_iters=36, max_step_rad=0.01)
        self.assertTrue(res_1d.converged)
        self.assertAlmostEqual(float(res_1d.q_rad[0]), 0.18, places=3)
        self.assertAlmostEqual(float(qb_1d[0, 0]), 0.18, places=3)

        # 4. 5-DoF SO-101 under DLS fallback + mixed revolute/prismatic pure-NumPy vs Pinocchio Jacobian parity.
        s_manifest = s101.load_model()
        q5 = np.array([0.22, -0.35, 0.62, -0.28, 0.35], dtype=np.float64)
        T5 = s101.fk(s_manifest, q5)
        with patch("ni_rua.robot.model.kinematics._pink_daqp_usable", return_value=False):
            sol5 = s101.ik(s_manifest, T5, q5 + 0.10, restarts=2)
            sol5_pp = s101.ik_position_priority(s_manifest, T5, q5 + 0.10, restarts=2)
        self.assertTrue(sol5["converged"])
        self.assertLessEqual(sol5_pp["position_error_m"], 1e-3)

        prismatic_chain = kin.KinematicChainSpec(
            joint_names=("j_rev", "j_slide", "j_yaw"),
            joint_types=("revolute", "prismatic", "revolute"),
            parent_transforms=np.stack(
                [np.eye(4), kin.rpy_xyz_to_matrix((0.1, 0.0, 0.05), (0.0, 0.0, 0.0)), kin.rpy_xyz_to_matrix((0.0, 0.0, 0.1), (0.0, 0.0, 0.0))]
            ),
            joint_axes=np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64),
            joint_limits_rad=np.array([[-2.0, 2.0], [0.0, 0.3], [-2.0, 2.0]], dtype=np.float64),
            tip_transform=kin.rpy_xyz_to_matrix((0.05, 0.0, 0.0), (0.0, 0.0, 0.0)),
        )
        q_rp = np.array([0.4, 0.12, -0.3], dtype=np.float64)
        t_wb = kin.rpy_xyz_to_matrix((0.12, -0.24, 0.76), (0.05, -0.04, 0.30))
        tip_pin, jac_pin = kin.fk_jacobian(prismatic_chain, q_rp, T_world_base=t_wb)
        pos_b_pin, rot_b_pin, jac_b_pin, orig_b_pin = kin.fk_batch(prismatic_chain, q_rp[None, :])
        with patch.dict(sys.modules, {"pinocchio": None, "pink": None}):
            tip_np, jac_np = kin.fk_jacobian(prismatic_chain, q_rp, T_world_base=t_wb)
            pos_b_np, rot_b_np, jac_b_np, orig_b_np = kin.fk_batch(prismatic_chain, q_rp[None, :])
            res_np = kin.solve_ik(
                x5,
                T_goal,
                q_ref + 0.05,
                kin.IKOptions(damping="error_scaled", position_tolerance_m=1e-4, orientation_tolerance_rad=1e-3),
            )
            qb_np = kin.solve_ik_batch(x5, T_goal[:3, 3], T_goal[:3, :3], (q_ref + 0.05)[None, :], pos_tol_m=1e-4, rot_tol_rad=1e-3)
        np.testing.assert_allclose(tip_np, tip_pin, atol=1e-12)
        np.testing.assert_allclose(jac_np, jac_pin, atol=1e-12)
        np.testing.assert_allclose(pos_b_np, pos_b_pin, atol=1e-12)
        np.testing.assert_allclose(rot_b_np, rot_b_pin, atol=1e-12)
        np.testing.assert_allclose(jac_b_np, jac_b_pin, atol=1e-12)
        np.testing.assert_allclose(orig_b_np, orig_b_pin, atol=1e-12)
        self.assertTrue(res_np.converged)
        self.assertLess(res_np.position_error_m, 1e-4)
        pe_b, oe_b = kin.pose_error(kin.fk_links(x5, qb_np[0])[x5.tip_link], T_goal)
        self.assertLess(pe_b, 1e-4)
        self.assertLess(oe_b, 1e-3)

        # 5. Control parameter validation & explicit zero-iteration budget contract.
        for bad_kwargs in (
            dict(max_step_rad=0.0),
            dict(max_step_rad=-0.1),
            dict(max_step_rad=float("nan")),
            dict(max_iterations=-1),
            dict(posture_weight=-0.1),
        ):
            with self.assertRaises(ValueError):
                kin.IKOptions(**bad_kwargs)
        zero_res = kin.solve_ik(x5, T_goal, q_ref + 0.1, kin.IKOptions(max_iterations=0))
        self.assertEqual(zero_res.iterations, 0)
        self.assertFalse(zero_res.converged)
        zero_batch = kin.solve_ik_batch(x5, T_goal[:3, 3], T_goal[:3, :3], q_ref[None, :], max_iters=0)
        np.testing.assert_allclose(zero_batch[0], q_ref, atol=1e-12)
        with self.assertRaises(ValueError):
            kin.solve_ik_batch(x5, np.array([np.nan, 0.0, 0.0]), T_goal[:3, :3], q_ref[None, :])
        with self.assertRaises(ValueError):
            kin.solve_ik_batch(x5, T_goal[:3, 3], T_goal[:3, :3], q_ref[None, :], max_step_rad=-0.1)


if __name__ == "__main__":
    unittest.main()
