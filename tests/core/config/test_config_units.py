"""Native config, garment runtime, and remote config unit checks."""

import ast
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ni_rua.core.config import Config, ConfigError, RemoteConfig, cli_overrides
from support.paths import CONFIGS_ROOT, REPO_ROOT

NI_RUA_TEST_TIER = "fast"


class NativeConfigCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configured_root = os.environ.get("NI_RUA_ROBODOJO_ROOT")
        roots = [Path(configured_root).expanduser()] if configured_root else [
            Path.home() / "code/RoboDojo", Path.home() / "code/robodojo"]
        root = next((path for path in roots if (path / "utils/pipeline_utils.py").is_file()), None)
        python = Path(os.environ.get("NI_RUA_ROBODOJO_PYTHON",
            str(Path.home() / "miniforge3/envs/RoboDojo/bin/python"))).expanduser()
        if root is None or not python.is_file():
            raise unittest.SkipTest("installed RoboDojo config and Python required; no simulator is launched")
        code = r'''
import csv
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
project = Path(sys.argv[2])
sys.path.insert(0, str(project))
sys.path.insert(0, str(root))
from omegaconf import OmegaConf
from utils.load_file import load_yaml
from utils.pipeline_utils import process_config, process_randomization
from ni_rua.robot.backends.robodojo.robot_worker import native_eval_config, configure_depth_capture
from ni_rua.core.observation_policy import DEPTH_ACTOR_PROFILE

with (project / "benchmarks/robodojo-42-15/cases.csv").open(newline="") as stream:
    rows = list(csv.DictReader(stream))
tasks = list(dict.fromkeys(row["task"] + ("_random" if row["variant"] == "random" else "") for row in rows))
index = load_yaml(root / "task/RoboDojo/config/_task.yml")
base = load_yaml(root / "env_cfg/arx_x5.yml")
records = {}
for task in tasks:
    config = native_eval_config(root, task, seed=3, layout_id=2)
    rgbd = native_eval_config(root, task, seed=3, layout_id=2, observation_profile=DEPTH_ACTOR_PROFILE)
    settings = {**index.get("common", {}), **index["tasks"].get(task, {})}
    expected_robot = load_yaml(root / "env_cfg/robot" / (settings.get("robot_config", "dual_x5") + ".yml"))
    if not settings.get("robot_self_collision", True):
        for robot in expected_robot["robots"]:
            robot["enabled_self_collisions"] = False
    official = OmegaConf.create({
        **{section: load_yaml(root / "env_cfg" / section / f"{name}.yml")
           for section, name in base["config"].items()},
        "task_env": load_yaml(root / "task/RoboDojo/config" / f"{task}.yml"),
        "eval_cfg": base,
    })
    official = process_randomization(official)
    official, _ = process_config(official, task_name=task)
    official.sim.scene.num_envs = 1
    official.sim.seed = [2]
    records[task] = dict(
        config=OmegaConf.to_container(config, resolve=True),
        rgbd_config=OmegaConf.to_container(rgbd, resolve=True),
        disabled_config=OmegaConf.to_container(configure_depth_capture(rgbd), resolve=True),
        expected_robot=expected_robot,
        expected_sim=OmegaConf.to_container(official.sim, resolve=True),
        expected_scene=OmegaConf.to_container(official.scene, resolve=True),
        expected_task=OmegaConf.to_container(official.task_env, resolve=True),
        expected_stabilization=settings.get("data_source", "datagen") == "teleop",
        expected_frequency=base["observation"]["collect_freq"],
    )
print(json.dumps(dict(records=records, isaac_imported=any(
    name == "isaacsim" or name.startswith("isaacsim.") or name.startswith("isaaclab.")
    for name in sys.modules))))
'''
        project = REPO_ROOT
        result = subprocess.run([str(python), "-c", code, str(root), str(project)],
            cwd=project, text=True, capture_output=True, timeout=60)
        if result.returncode:
            raise AssertionError("native config subprocess failed: " + result.stderr)
        cls.result = json.loads(result.stdout)

    def test_support_arms_match_upstream_task_registry(self):
        records = self.result["records"]
        self.assertEqual(len(records), 54)
        for task, record in records.items():
            with self.subTest(task=task):
                robots = record["config"]["robot"]
                self.assertEqual(robots, record["expected_robot"])
                supports = [robot for robot in robots["robots"] if robot.get("type") == "support"]
                if task in ("imitate_sorting_sequence", "make_kong", "play_tic_tac_toe"):
                    self.assertEqual(len(supports), 1)
                    self.assertEqual(supports[0]["robot_name"], "franka")
                else:
                    self.assertEqual(supports, [])

    def test_teleop_stabilization_and_ordinary_task_physics(self):
        for task, record in self.result["records"].items():
            with self.subTest(task=task):
                enabled = record["config"]["sim"].get("physx", {}).get("enable_stabilization", False)
                self.assertEqual(enabled, record["expected_stabilization"])
        self.assertTrue(self.result["records"]["classify_objects"]["expected_stabilization"])
        self.assertFalse(self.result["records"]["arrange_largest_number"]["expected_stabilization"])

    def test_simulation_scene_and_objects_match_official_defaults_for_every_task(self):
        records = self.result["records"]
        for task, record in records.items():
            with self.subTest(task=task):
                self.assertEqual(record["config"]["sim"], record["expected_sim"])
                self.assertEqual(record["config"]["scene"], record["expected_scene"])
                self.assertEqual(record["config"]["task_env"], record["expected_task"])
        for task in ("pour_liquid_into_cup", "pour_liquid_into_cup_random", "pour_by_language"):
            with self.subTest(fluid_task=task):
                config = records[task]["config"]
                self.assertTrue(config["task_env"]["Fluid"])
                self.assertEqual(config["sim"].get("device", "cpu"), "cpu")
                self.assertFalse(config["sim"].get("use_fabric", False))

    def test_selected_case_camera_and_observation_boundary_survive_native_setup(self):
        self.assertFalse(self.result["isaac_imported"])
        for task, record in self.result["records"].items():
            with self.subTest(task=task):
                config = record["config"]
                self.assertEqual(config["sim"]["seed"], [2])
                self.assertEqual(config["sim"]["scene"]["num_envs"], 1)
                self.assertEqual(config["eval_cfg"]["seed"], 3)
                self.assertEqual(config["eval_cfg"]["num_envs"], 1)
                self.assertEqual(config["eval_cfg"]["eval_num"], 1)
                self.assertEqual(config["eval_cfg"]["task_name"], task)
                self.assertEqual(config["eval_cfg"]["description"], {})
                self.assertEqual(config["deploy_cfg"], {"port": 0, "policy_name": "NI-RUA"})
                self.assertEqual(config["camera"]["default_frequency"], record["expected_frequency"])

    def test_depth_is_explicit_and_forced_off_when_profile_changes_back(self):
        for task, record in self.result["records"].items():
            with self.subTest(task=task):
                for key in ("config", "disabled_config"):
                    config = record[key]
                    self.assertFalse(config["eval_cfg"]["observation"]["vision"]["depth"])
                    self.assertFalse(config["eval_cfg"]["observation"]["vision"]["approximate_depth"])
                    for capture in config["camera"]["annotator"].values():
                        self.assertFalse(any(isinstance(item, dict) and item.get("type") in
                            ("depth", "distance_to_image_plane", "distance_to_camera") for item in capture.values()))
                config = record["rgbd_config"]
                self.assertTrue(config["eval_cfg"]["observation"]["vision"]["depth"])
                self.assertFalse(config["eval_cfg"]["observation"]["vision"]["approximate_depth"])
                cameras = [name for name, value in config["camera"].items() if isinstance(value, dict) and "camera" in value]
                self.assertGreater(len(cameras), 0)
                for camera in cameras:
                    self.assertEqual(config["camera"]["annotator"][camera]["distance_to_image_plane_capture"]["type"],
                                     "distance_to_image_plane")


class ArrayView:
    def __init__(self, values): self.values = np.asarray(values)
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.values
    def squeeze(self, axis): return ArrayView(self.values.squeeze(axis))


class NativeGarmentRuntimeCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(Config.load().backend.root)
        source = root / "env/scene_manager/objects/garment.py"
        if not source.is_file():
            raise unittest.SkipTest("installed native garment source required; no simulator is launched")
        tree = ast.parse(source.read_text())
        methods = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                   and node.name in ("_get_initial_info", "sample_mesh_vertices")]
        cls.namespace = dict(np=np)
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), cls.namespace)

    def test_cuda_cloth_registers_before_binding_and_reads_live_particles(self):
        events = []
        particle_positions = ArrayView([[[1., 2., 3.], [4., 5., 6.]]])
        sim_view = object()
        def initialize(view):
            self.assertEqual(events, ["register"])
            self.assertIs(view, sim_view)
            events.append("bind")
        def positions():
            self.assertEqual(events, ["register", "bind"])
            events.append("read")
            return particle_positions
        omni, physx = ModuleType("omni"), ModuleType("omni.physx")
        omni.physx = physx
        physx.get_physx_interface = lambda: SimpleNamespace(
            force_load_physics_from_usd=lambda: events.append("register"))
        obj = SimpleNamespace(_device="cuda:0", _cloth_prim_view=SimpleNamespace(
            initialize=initialize, get_world_positions=positions))
        manager = SimpleNamespace(get_physics_sim_view=lambda: sim_view)
        with patch.dict(self.namespace, SimulationManager=manager), patch.dict(
                "sys.modules", {"omni": omni, "omni.physx": physx}):
            self.namespace["_get_initial_info"](obj)
        self.assertEqual(events, ["register", "bind", "read"])
        self.assertIs(obj.initial_points_positions, particle_positions)

    def test_cuda_sampling_returns_native_pose_and_unmodified_live_vertices(self):
        positions = ArrayView([[[1., 2., 3.], [4., 5., 6.]]])
        position, orientation = ArrayView([.2, .3, .4]), ArrayView([.5, .5, .5, .5])
        obj = SimpleNamespace(_device="cuda:0", get_world_pose=lambda: (position, orientation),
            _cloth_prim_view=SimpleNamespace(get_world_positions=lambda: positions))
        world, mesh, pos, quat = self.namespace["sample_mesh_vertices"](obj)
        np.testing.assert_array_equal(world, positions.values[0])
        self.assertIs(mesh, world)
        self.assertIs(pos, position)
        self.assertIs(quat, orientation)


SERVICE = {"name": "vggt", "remote_port": 8331, "command": "bash start.sh", "gpu": "auto", "env": ["VGGT_HOME=~/models/vggt"],
           "workdir": "~/code/NI-RUA/services/vggt_height", "stop": "bash stop.sh"}


def remote(**fields):
    return {"remote": {"enabled": True, "host": "gpu-host.example", "services": [dict(SERVICE)], **fields}}


class RemoteConfigTest(unittest.TestCase):
    def test_default_is_disabled_and_absent_from_snapshots(self):
        config = Config()
        self.assertEqual(config.remote, RemoteConfig())
        self.assertFalse(config.remote.enabled)
        self.assertNotIn("remote", config.to_dict())     # X5 / simulation manifests stay byte-identical
        self.assertEqual(Config.from_dict({}).remote, RemoteConfig())

    def test_documented_defaults(self):
        config = Config.from_dict(remote())
        service = config.remote.services[0]
        self.assertEqual((service.local_port, service.effective_local_port, service.health_path, service.reuse_if_healthy),
                         (None, 8331, "/health", True))
        self.assertEqual((service.gpu_min_free_mib, service.startup_timeout_s), (15360, 300.0))
        self.assertEqual((config.remote.required, config.remote.stop_on_exit, config.remote.health_interval_s), (False, False, 30.0))

    def test_file_round_trip_resolves_the_record_path_next_to_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ni-rua.json"
            path.write_text(json.dumps({**remote(record="remote.json", required=True, stop_on_exit=True)}))
            config = Config.load(path)
            self.assertEqual(Path(config.remote.record).name, "remote.json")
            self.assertEqual(Path(config.remote.record).parent.resolve(), Path(tmp).resolve())
            self.assertEqual(config.remote.services[0].env, ("VGGT_HOME=~/models/vggt",))
            self.assertTrue(config.remote.required and config.remote.stop_on_exit)
            self.assertEqual(Config.from_dict(config.to_dict()).remote, config.remote)

    def test_shipped_example_is_a_valid_overlay(self):
        example = CONFIGS_ROOT / "so101.remote.example.json"
        config = Config.load(example)
        self.assertTrue(config.remote.enabled)
        self.assertEqual(config.remote.services[0].effective_local_port, 8331)   # = the default features.vggt_endpoint

    def test_bad_values_are_rejected(self):
        cases = {
            "enabled without host": {"enabled": True, "services": [SERVICE]},
            "enabled without services": {"enabled": True, "host": "h"},
            "unknown key": {"enabled": True, "host": "h", "services": [SERVICE], "retry": 3},
            "unknown service key": {"enabled": True, "host": "h", "services": [{**SERVICE, "ssh_key": "x"}]},
            "service without name": {"enabled": True, "host": "h", "services": [{"remote_port": 8331}]},
            "service without port": {"enabled": True, "host": "h", "services": [{"name": "vggt"}]},
            "bad name": {"enabled": True, "host": "h", "services": [{**SERVICE, "name": "a b"}]},
            "port range": {"enabled": True, "host": "h", "services": [{**SERVICE, "remote_port": 70000}]},
            "port type": {"enabled": True, "host": "h", "services": [{**SERVICE, "remote_port": "8331"}]},
            "health path": {"enabled": True, "host": "h", "services": [{**SERVICE, "health_path": "health"}]},
            "env format": {"enabled": True, "host": "h", "services": [{**SERVICE, "env": ["NOEQUALS"]}]},
            "gpu format": {"enabled": True, "host": "h", "services": [{**SERVICE, "gpu": "best"}]},
            "gpu without command": {"enabled": True, "host": "h", "services": [{**SERVICE, "command": None}]},
            "negative memory": {"enabled": True, "host": "h", "services": [{**SERVICE, "gpu_min_free_mib": 0}]},
            "timeout": {"enabled": True, "host": "h", "services": [{**SERVICE, "startup_timeout_s": 0}]},
            "duplicate names": {"enabled": True, "host": "h", "services": [SERVICE, {**SERVICE, "remote_port": 8333}]},
            "duplicate local ports": {"enabled": True, "host": "h", "services": [SERVICE, {**SERVICE, "name": "b", "remote_port": 8333, "local_port": 8331}]},
            "interval": {"enabled": True, "host": "h", "services": [SERVICE], "health_interval_s": 0},
            "empty host": {"enabled": True, "host": " ", "services": [SERVICE]},
            "services type": {"enabled": True, "host": "h", "services": SERVICE},
        }
        for label, section in cases.items():
            with self.subTest(label), self.assertRaises(ConfigError):
                Config.from_dict({"remote": section})

    def test_vggt_endpoint_must_be_served_by_a_forward(self):
        with self.assertRaises(ConfigError) as caught:
            Config.from_dict({"remote": {"enabled": True, "host": "h", "services": [{**SERVICE, "remote_port": 8331, "local_port": 18331}]}})
        self.assertIn("remote_service_missing_for_vggt_endpoint", str(caught.exception))
        Config.from_dict({"remote": {"enabled": True, "host": "h", "services": [{**SERVICE, "local_port": 18331}]},
                          "features": {"vggt_endpoint": "http://127.0.0.1:18331"}})
        # a remote endpoint that is not forwarded locally, or perception switched off, needs no matching forward
        Config.from_dict({"remote": {"enabled": True, "host": "h", "services": [{**SERVICE, "local_port": 18331}]},
                          "features": {"external_perception": False}})

    def test_cli_override_toggles_enabled_only(self):
        self.assertEqual(cli_overrides({"remote": False}), {"remote": {"enabled": False}})
        config = Config.from_dict(remote())
        self.assertFalse(config.with_overrides(cli_overrides({"remote": False})).remote.enabled)
        with self.assertRaises(ConfigError):
            Config().with_overrides(cli_overrides({"remote": True}))     # no host/services declared


if __name__ == "__main__":
    unittest.main()
