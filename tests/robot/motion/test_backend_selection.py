"""Environment/robot combinations fail before starting a worker."""

NI_RUA_TEST_TIER = "fast"

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from ni_rua.robot.backends.factory import create_backend
from ni_rua.core.config import Config
from ni_rua.robot.backends.robodojo.robodojo import RobotBackend


class BackendSelectionTests(unittest.TestCase):
    def test_old_configuration_selects_same_backend_without_starting_it(self):
        config = Config.from_dict({'backend': {'root': '/tmp/robodojo', 'python': '/tmp/python'}})
        with patch.object(RobotBackend, 'start_process', side_effect=AssertionError('unexpected launch')):
            backend = create_backend(config=config, task='stack_bowls', seed=2, layout_id=3)
        self.assertIsInstance(backend, RobotBackend)
        self.assertEqual(backend.robot_binding.robot.robot_model_id, 'arx-x5')
        self.assertEqual((backend.root, backend.python, backend.task, backend.seed, backend.layout_id),
                         # Configured paths are canonical (/tmp is a symlink on macOS).
                         (str(Path('/tmp/robodojo').resolve()), str(Path('/tmp/python').resolve()), 'stack_bowls', 2, 3))
        self.assertEqual(Config.from_dict(config.to_dict()), config)

    def test_environment_and_robot_must_be_a_supported_pair(self):
        for selection, error in [
            ({'environment': 'unknown'}, 'unsupported_environment'),
            ({'environment': 'robodojo', 'robot_model_id': 'other-arm'}, 'unsupported_robodojo_robot'),
        ]:
            with self.subTest(selection=selection), \
                    patch.object(RobotBackend, 'start_process', side_effect=AssertionError('unexpected launch')), \
                    self.assertRaisesRegex(ValueError, error):
                create_backend(config=Config.from_dict({'backend': selection}))

    def test_parent_passes_binding_and_reset_uses_its_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Config.from_dict({'backend': {'root': directory, 'python': '/tmp/python'}})
            backend = create_backend(config=config, run_dir=Path(directory) / 'worker', task='stack_bowls')
            with patch.object(backend, 'start_process', return_value={'ready': True}) as launch:
                self.assertEqual(backend.start(), {'ready': True})
            command = launch.call_args.args[0]
            self.assertEqual(command[command.index('--robot-model-id') + 1], 'arx-x5')
            layout = Path(directory).resolve() / 'Assets/Eval_Layout/RoboDojo/arx_x5/2/stack_bowls_3.json'
            layout.parent.mkdir(parents=True)
            layout.write_text('{}')
            self.assertEqual(backend.validate_reset(seed=2, layout_id=3), layout)
            with self.assertRaisesRegex(ValueError, 'reset_layout_unavailable'):
                backend.validate_reset(seed=2, layout_id=4)

    def test_direct_robodojo_callers_cannot_ignore_environment_selection(self):
        with self.assertRaisesRegex(ValueError, 'unsupported_robodojo_environment'):
            RobotBackend(config=Config.from_dict({'backend': {'environment': 'other-simulator'}}))


if __name__ == "__main__":
    unittest.main()
