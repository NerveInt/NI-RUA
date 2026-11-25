"""
- test_action_compilation.py: Backend compiler selection through Harness and the public batch builder.
- test_actor_source.py: The actor sees published Harness source, never an entire checkout.
"""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch
from ni_rua.robot.model.embodiments import ARX_X5_PROFILE
from ni_rua.robot.motion import action_compilation, arx_actions
from ni_rua.runtime.harness import Harness
from ni_rua.runtime.actions.unified_actions import compile_request
from ni_rua.runtime.actions.unified_actions import dispatch
from ni_rua.core.storage.store import Store
from support.fixture_robot import FixtureRobot
import os
import subprocess
import sys
from ni_rua.cortex.session.actor_source import RETIRED_SOURCE_FILES, SOURCE_FILES, snapshot_actor_source

NI_RUA_TEST_TIER = "component"


# --- Folded from tests/test_action_compilation.py ---

class CustomRobot(FixtureRobot):
    def start(self):
        return dict(super().start(), robot_model_id='custom', frame_id='custom-world',
            tcp_config_id='custom-tcp', limits_profile='custom-limits', path_kinds=['backend_free', 'hold'])

    @staticmethod
    def compile_action(kind, args, identity, capabilities):
        if kind != 'move':
            raise ValueError('custom_fixture_requires_link_pose')
        return dict(schema_version='0.1', **identity, world_kind='observed',
            action_id=args.get('action_id', 'custom-leaf'), based_on_observation_id=args['observation_id'],
            frame_id=capabilities['frame_id'], tcp_config_id=capabilities['tcp_config_id'],
            mode='absolute', limits_profile=capabilities['limits_profile'], timeout_s=args['timeout_s'],
            path=dict(kind='backend_free', waypoints=[{args['arm']: dict(position_m=args['position_m'],
                quaternion_xyzw=args['quaternion_xyzw'])}]), gripper_events=[])

    def validate_action(self, action):
        self.validations = getattr(self, 'validations', 0) + 1
        if action['tcp_config_id'] != 'custom-tcp':
            raise ValueError('custom_fixture_wrong_tcp')

    def preview_batch(self, action):
        self.previews = getattr(self, 'previews', 0) + 1
        return dict(status='planned', plan_id='custom-native', planned_native_steps=1,
            trajectory_data=dict(samples=[]))

    def execute(self, action, progress_callback=None, cancel_flag=None):
        self.executions += 1
        return dict(status='completed', executed_native_steps=1, completed_native_steps=1,
            batch_status='completed', batch_cursor=len(action['stages']),
            completed_stage_ids=[stage['id'] for stage in action['stages']], pending_stage_ids=[],
            stage_results={}, terminal_observation=self.observe())


class ActionCompilationTests(unittest.TestCase):
    def setUp(self):
        self.harness = object.__new__(Harness)
        # Harness.__init__ resolves the run's embodiment before anything reads it; a bare instance names it here.
        self.harness.embodiment = ARX_X5_PROFILE
        self.harness.identity = dict(run_id='run', episode_id='ep', env_instance_id='env', env_generation=1)
        self.harness.capabilities = dict(frame_id='env_origin', tcp_config_id='arx-x5-link6-v1')
        self.harness.robot = SimpleNamespace()
        self.harness.config = SimpleNamespace(features=SimpleNamespace(default_motion_speed_scale=1.))
        self.pose = dict(arm='left', grasp_position_m=[.2, -.1, .8], angles_deg=[0, 0, 0])

    def test_non_arx_backend_owns_wait_and_grasp_compilation_paths(self):
        self.harness.capabilities = dict(frame_id='custom-world', tcp_config_id='custom-tcp', robot_model_id='custom')
        compiled = {'custom': 'compiled'}
        self.harness.robot = SimpleNamespace(compile_action=Mock(return_value=compiled))
        args = {'observation_id': 'obs'}
        with patch('ni_rua.robot.motion.arx_actions.compile_action', side_effect=AssertionError('ARX must not run')):
            self.assertIs(self.harness._action_from_args('robot_wait', args), compiled)
            self.assertIs(action_compilation.compile_action(self.harness.robot, 'grasp', args,
                self.harness.identity, self.harness.capabilities), compiled)
        self.harness.robot.compile_action.assert_has_calls([
            call('wait', args, self.harness.identity, self.harness.capabilities),
            call('grasp', args, self.harness.identity, self.harness.capabilities)])
        self.assertEqual(self.harness.robot.compile_action.call_count, 2)

    def test_unknown_or_conflicting_calibration_requires_explicit_compiler(self):
        for caps in ({}, {'robot_model_id': 'arx-x5'}, {'tcp_config_id': 'custom'},
                dict(self.harness.capabilities, robot_model_id='custom')):
            self.harness.capabilities = caps
            with self.subTest(capabilities=caps):
                with self.assertRaisesRegex(ValueError, '^action_compiler_unavailable:compile_action$'):
                    self.harness._action_from_args('robot_wait', {})

    def test_declared_legacy_x5_and_native_hooks_preserve_requests_and_errors(self):
        from ni_rua.robot.backends.robodojo.robodojo import RobotBackend
        single_args = dict(self.pose, observation_id='obs', action_id='single', contact_region='finger_middle')
        original = deepcopy(single_args)
        for robot in (SimpleNamespace(), object.__new__(RobotBackend)):
            self.harness.robot = robot
            self.assertEqual(action_compilation.compile_action(robot, 'grasp', single_args,
                    self.harness.identity, self.harness.capabilities),
                arx_actions.compile_action('grasp', single_args, self.harness.identity, self.harness.capabilities))
            with self.assertRaisesRegex(ValueError, '^dual_arm_progression_requires_dual_arm_motion$'):
                self.harness._action_from_args('robot_wait', {'dual_arm_progression': 'independent'})
        self.assertEqual(single_args, original)

    def test_batch_uses_backend_compiler_for_motion_and_split_jaw(self):
        self.harness.robot.compile_action = Mock(side_effect=arx_actions.compile_action)
        request = dict(observation_id='obs', action_id='batch', stages=[dict(self.pose,
            id='reach', kind='move', gripper_goal_normalized=0.)])
        original = deepcopy(request)
        result = compile_request(self.harness, request)
        self.assertEqual([call.args[0] for call in self.harness.robot.compile_action.call_args_list], ['grasp', 'gripper'])
        self.assertEqual([stage['id'] for stage in result['stages']], ['reach/motion', 'reach'])
        self.assertEqual(request, original)

    def custom_harness(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        store = Store(Path(temporary.name) / 'store', self.harness.identity)
        self.addCleanup(store.close)
        robot = CustomRobot(temporary.name)
        harness = Harness(store, robot)
        harness.start()
        request = dict(observation_id=harness.latest['observation_id'], stages=[dict(id='reach',
            kind='move', arm='left', position_m=[.2, 0., 1.], quaternion_xyzw=[0., 0., 0., 1.])])
        return harness, robot, request

    def test_non_arx_public_preview_and_cached_execution_use_same_validation(self):
        harness, robot, request = self.custom_harness()
        with patch('ni_rua.runtime.actions.inspection._trajectory_image', return_value=({'status': 'unavailable'}, [])):
            preview = dispatch(harness, request, {'tool_call_id': 'preview'})
            self.assertEqual(preview.data['status'], 'planned')
            harness.publish_observation(robot.observe())
            result = dispatch(harness, dict(mode='execute', plan_id=preview.data['plan_id']),
                {'tool_call_id': 'execute'})
        self.assertEqual(result.data['receipt']['status'], 'completed')
        self.assertEqual((robot.previews, robot.executions), (1, 1))
        self.assertEqual(robot.validations, 3)

    def test_non_arx_batch_rejects_invalid_children_and_missing_native_validator(self):
        from ni_rua.core.contracts import validate_action
        harness, robot, request = self.custom_harness()
        compiled = compile_request(harness, dict(request, action_id='batch'))
        for field, value, reason in (
                ('based_on_observation_id', 'stale', 'stale_observation'),
                ('env_generation', 999, 'stale_environment_identity'),
                ('type', 'BatchAction', 'batch_stage_requires_ee_action')):
            action = deepcopy(compiled)
            action['stages'][0]['action'][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, reason):
                validate_action(action, harness.identity, harness.latest['observation_id'], harness.capabilities)
        action = deepcopy(compiled)
        action['stages'][0]['action']['path']['waypoints'][0]['left']['position_m'][0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'position_m'):
            validate_action(action, harness.identity, harness.latest['observation_id'], harness.capabilities)
        robot.validate_action = None
        with self.assertRaisesRegex(ValueError, 'action_validator_unavailable'):
            dispatch(harness, request, {})
        rejected = harness.execute(compiled)['receipt']
        self.assertEqual(rejected['status'], 'rejected')
        self.assertIn('action_validator_unavailable', rejected['reason'])
        self.assertEqual(robot.executions, 0)
        self.assertFalse(hasattr(robot, 'previews'))


class SingleArmRobot(CustomRobot):
    """Declares one arm; generic validation must reject the other id."""
    def start(self):
        return dict(super().start(), arms=['right'], dual_arm=False)


class SingleArmCapabilityTests(unittest.TestCase):
    def harness(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        identity = dict(run_id='run', episode_id='ep', env_instance_id='env', env_generation=1)
        store = Store(Path(temporary.name) / 'store', identity)
        self.addCleanup(store.close)
        robot = SingleArmRobot(temporary.name)
        harness = Harness(store, robot)
        harness.start()
        return harness, robot

    def request(self, harness, arm):
        return dict(observation_id=harness.latest['observation_id'], stages=[dict(id='reach',
            kind='move', arm=arm, position_m=[.2, 0., 1.], quaternion_xyzw=[0., 0., 0., 1.])])

    def test_declared_arm_previews_and_executes(self):
        harness, robot = self.harness()
        with patch('ni_rua.runtime.actions.inspection._trajectory_image', return_value=({'status': 'unavailable'}, [])):
            preview = dispatch(harness, self.request(harness, 'right'), {'tool_call_id': 'preview'})
            self.assertEqual(preview.data['status'], 'planned')
            harness.publish_observation(robot.observe())
            result = dispatch(harness, dict(mode='execute', plan_id=preview.data['plan_id']), {'tool_call_id': 'execute'})
        self.assertEqual(result.data['receipt']['status'], 'completed')
        self.assertEqual(robot.executions, 1)

    def test_undeclared_arm_is_rejected_before_any_backend_call(self):
        harness, robot = self.harness()
        with self.assertRaisesRegex(ValueError, 'robot_action_stage_requires_arm'):
            compile_request(harness, self.request(harness, 'left'))
        from ni_rua.core.contracts import validate_action
        compiled = compile_request(harness, dict(self.request(harness, 'right'), action_id='b'))
        leaf = deepcopy(compiled['stages'][0]['action'])
        leaf['path']['waypoints'] = [{'left': leaf['path']['waypoints'][0]['right']}]
        with self.assertRaisesRegex(ValueError, 'unsupported_capability:arms'):
            validate_action(leaf, harness.identity, harness.latest['observation_id'], harness.capabilities)
        self.assertEqual(robot.executions, 0)
        self.assertFalse(hasattr(robot, 'previews'))

    def test_robot_action_calibration_gate_follows_backend_declaration(self):
        from ni_rua.runtime.tools.builtins import unavailable
        def runtime(unified):
            return SimpleNamespace(observation_profile='rgb_proprio_v1', capabilities=dict(unified_action=unified),
                features=SimpleNamespace(motion_execution_mode='resampled'))
        self.assertEqual(unavailable('robot_action', runtime({'available': True}), {}),
                         'tool_unavailable_without_permitted_calibration')
        self.assertIsNone(unavailable('robot_action', runtime({'available': True, 'requires_camera_calibration': False}), {}))
        # Only robot_action is opened; image-geometry tools keep the gate.
        self.assertEqual(unavailable('robot_contact_overlay', runtime({'requires_camera_calibration': False}), {}),
                         'tool_unavailable_without_permitted_calibration')

    def test_capability_arms_default_and_invalid_declarations(self):
        from ni_rua.core.action_validation import capability_arms
        self.assertEqual(capability_arms(None), ('left', 'right'))
        self.assertEqual(capability_arms({}), ('left', 'right'))
        self.assertEqual(capability_arms({'arms': ['right']}), ('right',))
        for bad in ([], 'right', ['right', 'right'], [''], [1]):
            with self.subTest(arms=bad), self.assertRaisesRegex(ValueError, 'invalid_capability:arms'):
                capability_arms({'arms': bad})



# --- Folded from tests/test_actor_source.py ---

FORBIDDEN_SUBSTRINGS = (
    "backends/",
    "apps/",
    "profiles.py",
    "native_evaluator",
    "scene_truth",
    "roboprobe_source",
    "model_proxy",
    "relay/",
    "evaluation/",
)


class ActorSourceTest(unittest.TestCase):
    def test_snapshot_is_exact_and_rejects_unpublished_files_and_links(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "snapshot"
            snapshot_actor_source(destination)
            self.assertEqual({path.relative_to(destination).as_posix() for path in destination.rglob("*")
                              if path.is_file()}, {"ni_rua/" + name for name in SOURCE_FILES})
            path = destination / "ni_rua" / "robot/backends/robodojo/robot_worker.py"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("unpublished backend")
            with self.assertRaisesRegex(ValueError, "unpublished_files"):
                snapshot_actor_source(destination)
            path.unlink()
            path = destination / "ni_rua" / "runtime/harness.py"
            path.unlink()
            target = Path(directory) / "host-file"
            target.write_text("host sentinel")
            path.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "unpublished_files"):
                snapshot_actor_source(destination)
            self.assertEqual(target.read_text(), "host sentinel")

    def test_source_files_strictly_exclude_backends_apps_profiles_and_observers(self):
        self.assertEqual(len(SOURCE_FILES), len(set(SOURCE_FILES)))
        for relative in SOURCE_FILES:
            for forbidden in FORBIDDEN_SUBSTRINGS:
                self.assertNotIn(forbidden, relative, msg=f"{relative} matched forbidden {forbidden!r}")

    def test_every_published_module_imports_cleanly_from_isolated_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "harness"
            snapshot_actor_source(destination)
            modules = []
            for relative in SOURCE_FILES:
                if not relative.endswith(".py"):
                    continue
                parts = ["ni_rua", *(p for p in relative[:-3].split("/") if p and p != "__init__")]
                modules.append(".".join(parts))
            script = (
                "import importlib, sys\n"
                "sys.modules['jsonschema'] = None\n"
                "import ni_rua\n"
                "del sys.modules['jsonschema']\n"
                f"for mod in {modules!r}:\n"
                "    importlib.import_module(mod)\n"
                f"print('OK', len({modules!r}))\n"
            )
            env = {**os.environ, "PYTHONPATH": str(destination), "PYTHONDONTWRITEBYTECODE": "1"}
            result = subprocess.run(
                [sys.executable, "-P", "-c", script],
                cwd=directory,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn(f"OK {len(modules)}", result.stdout)

    def test_catalogue_profiles_render_from_the_isolated_snapshot(self):
        """The profile renderer is imported lazily by definitions; the published snapshot must carry it and it must run."""
        self.assertIn("runtime/tools/catalog_guide.py", SOURCE_FILES)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "harness"
            snapshot_actor_source(destination)
            self.assertTrue((destination / "ni_rua" / "runtime" / "tools" / "catalog_guide.py").is_file())
            script = (
                "from ni_rua.runtime.tools.catalog_guide import PROFILES, PROSE_PROFILES, with_guide\n"
                "item = dict(name='t', description='d', inputSchema=dict(type='object', properties=dict("
                "x=dict(type='number', minimum=0, description='X.'))))\n"
                "print(len([with_guide(dict(item), p)['description'] for p in PROFILES if p not in PROSE_PROFILES]))\n"
            )
            env = {**os.environ, "PYTHONPATH": str(destination), "PYTHONDONTWRITEBYTECODE": "1"}
            result = subprocess.run([sys.executable, "-P", "-c", script], cwd=directory, env=env,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            from ni_rua.runtime.tools.catalog_guide import PROFILES, PROSE_PROFILES
            self.assertEqual(int(result.stdout.strip()), len(PROFILES) - len(PROSE_PROFILES))

    def test_rollout_tracker_snapshot_closure_has_zero_trajectory_evidence_dependency(self):
        import ast
        from ni_rua.runtime.actions import rollout_tracker

        src = Path(rollout_tracker.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        imported = [
            node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
        ]
        self.assertFalse(any("trajectory_evidence" in mod for mod in imported), imported)
        self.assertEqual(rollout_tracker._DEFAULT_ARMS, ("left", "right"))

    def test_snapshot_prunes_retired_source_files_on_resume_while_rejecting_symlinks_and_unpublished(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "snapshot"
            snapshot_actor_source(destination)
            for relative in RETIRED_SOURCE_FILES:
                retired = destination / "ni_rua" / relative
                retired.parent.mkdir(parents=True, exist_ok=True)
                retired.write_text("# retired pre-ad01185 snapshot module\n")
            snapshot_actor_source(destination)
            for relative in RETIRED_SOURCE_FILES:
                self.assertFalse((destination / "ni_rua" / relative).exists(), relative)
            self.assertEqual(
                {path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file()},
                {"ni_rua/" + name for name in SOURCE_FILES},
            )

            # A symlink at a retired path must never be unlinked or followed; snapshot must fail closed.
            host_sentinel = Path(directory) / "host-sentinel"
            host_sentinel.write_text("protected")
            retired_link = destination / "ni_rua" / "cortex/context/legacy_task_planning.py"
            retired_link.symlink_to(host_sentinel)
            with self.assertRaisesRegex(ValueError, "unpublished_files"):
                snapshot_actor_source(destination)
            self.assertTrue(retired_link.is_symlink())
            self.assertEqual(host_sentinel.read_text(), "protected")
            retired_link.unlink()

            # Arbitrary unpublished files alongside retired files must still fail closed.
            (destination / "ni_rua" / "cortex/context/operation_candidates.py").write_text("# retired\n")
            unpublished = destination / "ni_rua" / "cortex/context/self_briefing.py"
            unpublished.write_text("# never published\n")
            with self.assertRaisesRegex(ValueError, "unpublished_files"):
                snapshot_actor_source(destination)
            # A refused snapshot keeps the retired file too: nothing is removed before the decision.
            self.assertTrue((destination / "ni_rua" / "cortex/context/operation_candidates.py").is_file())

    def test_so101_wildcard_import_succeeds_in_isolated_snapshot_while_lazy_gravity_stays_importable(self):
        import ni_rua.robot.embodiments.so101 as so101_pkg
        from ni_rua.robot.embodiments.so101 import gravity

        self.assertNotIn("gravity", so101_pkg.__all__)
        self.assertTrue(hasattr(gravity, "__file__"))

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "harness"
            snapshot_actor_source(destination)
            script = (
                "from ni_rua.robot.embodiments.so101 import *\n"
                "import ni_rua.robot.embodiments.so101 as pkg\n"
                "assert 'gravity' not in pkg.__all__\n"
                "print('SO101_STAR_OK', len(pkg.__all__))\n"
            )
            env = {**os.environ, "PYTHONPATH": str(destination), "PYTHONDONTWRITEBYTECODE": "1"}
            result = subprocess.run(
                [sys.executable, "-P", "-c", script],
                cwd=directory,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("SO101_STAR_OK", result.stdout)



if __name__ == "__main__":
    unittest.main()
