"""Operator configuration parsing, precedence and archived-run behavior."""
from support.program_fixture import program_services
from dataclasses import FrozenInstanceError
import contextlib
import io
import json
import os
from pathlib import Path
import pwd
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from ni_rua.core.config import Config, ConfigError
from ni_rua.apps.gateway.workbench import Workbench
from ni_rua.apps.gateway.cli import main
from ni_rua.runtime.harness import Harness
from ni_rua.core.observation_policy import ACTOR_POLICY, ACTOR_PROFILE, DEPTH_ACTOR_PROFILE
from ni_rua.runtime.programs.runner import ProgramRunner
from ni_rua.core.storage.store import Store
from support.fixture_robot import WaitFixtureRobot
from support.paths import ABSENT_LOCAL_CONFIG, CONFIGS_ROOT, REPO_ROOT, isolate_home

NI_RUA_TEST_TIER = "component"


def setUpModule():
    isolate_home()   # Workbench tests below open the grasp memory store at its '~' default


from copy import deepcopy
from functools import lru_cache


@lru_cache(maxsize=1)
def _cached_default_config():
    return Config()


@lru_cache(maxsize=1)
def _cached_example_raw():
    return json.loads((CONFIGS_ROOT / 'ni-rua.example.json').read_text())


@lru_cache(maxsize=1)
def _cached_ablation_base_config():
    return Config.from_dict({"features": {"motion_review": False}, "model": {"native_memories": False}})


class ConfigTests(unittest.TestCase):
    @property
    def default_config(self):
        return _cached_default_config()

    @property
    def example_raw(self):
        return deepcopy(_cached_example_raw())

    def setUp(self):
        temporary = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(temporary).resolve()
        self.path = self.root / 'config.json'

    def save(self, value):
        self.path.write_text(json.dumps(value))
        return self.path

    def test_precedence_and_path_origins(self):
        self.save({'features': {'enable_wait': False}, 'model': {'codex_home': 'auth'},
                   'images': {'reuse_exact_images': False}, 'run': {'gpu': 2}})
        env = {'CODEX_HOME': '/old/auth', 'NI_RUA_CONFIG': str(self.path),
               'NI_RUA_IMAGE_SETTINGS': '{"transport_format":"jpeg","reuse_exact_images":true}'}
        resolved = Config.load(environ=env, overrides={'run': {'gpu': 0}, 'features': {'enable_wait': True}})
        self.assertEqual(resolved.run.gpu, 0)
        self.assertTrue(resolved.features.enable_wait)
        self.assertEqual(resolved.model.codex_home, str(self.root / 'auth'))
        self.assertEqual(resolved.images.transport_format, 'jpeg')
        self.assertFalse(resolved.images.reuse_exact_images)
        self.assertEqual(Config.load(self.path, environ={}, overrides={'model': {'codex_home': 'local-auth'}}).model.codex_home,
                         str(Path.cwd() / 'local-auth'))

    def test_support_height_observation_mode_survives_config_and_cli_layers(self):
        self.save({'features': {'vggt_support_z_m': None}})
        loaded = Config.load(self.path, environ={})
        self.assertIsNone(loaded.features.vggt_support_z_m)
        self.assertIsNone(Config.from_dict(loaded.to_dict()).features.vggt_support_z_m)
        for argv_extra, expected in (([], None), (['--vggt-support-z-m', '-0.02'], -0.02),
                                     (['--vggt-support-z-m', '0.765'], .765),
                                     (['--vggt-support-z-m', 'observation'], None)):
            with self.subTest(argv=argv_extra):
                out = io.StringIO()
                with patch('sys.argv', ['ni_rua', '--config', str(self.path), '--print-config', *argv_extra]), \
                     contextlib.redirect_stdout(out):
                    main()
                self.assertEqual(json.loads(out.getvalue())['features']['vggt_support_z_m'], expected)
        self.assertEqual(loaded.with_overrides({'features': {'vggt_support_z_m': .765}}).features.vggt_support_z_m, .765)
        for invalid in (True, 'observation', 'None', float('nan'), 3.1):
            with self.subTest(invalid=invalid), self.assertRaises(ConfigError):
                Config.from_dict({'features': {'vggt_support_z_m': invalid}})

    def test_snapshot_is_immutable_and_environment_independent(self):
        config = Config.from_dict({'run': {'tag': ['paired']}, 'features': {'disabled_tools': ['program_run']}})
        with self.assertRaises(FrozenInstanceError):
            config.features.enable_wait = False
        with patch.dict(os.environ, {'NI_RUA_IMAGE_SETTINGS': 'broken JSON', 'CODEX_HOME': '/different/auth'}):
            restored = Config.from_dict(config.to_dict())
        self.assertEqual(restored, config)
        self.assertIsInstance(restored.features.disabled_tools, tuple)
        changed = config.with_overrides({'features': {'enable_wait': False}})
        self.assertTrue(config.features.enable_wait)
        self.assertFalse(changed.features.enable_wait)

    def test_sandbox_switch_is_strict_and_preserved_in_snapshots(self):
        self.assertFalse(Config.from_dict({}).features.sandbox_enabled)
        config = Config.load(self.save({'features': {'sandbox_enabled': True}}), environ={})
        self.assertTrue(config.features.sandbox_enabled)
        self.assertTrue(Config.from_dict(config.to_dict()).features.sandbox_enabled)
        disabled = Config.load(self.save({'features': {'sandbox_enabled': False}}), environ={})
        self.assertFalse(disabled.features.sandbox_enabled)
        self.assertFalse(Config.from_dict(disabled.to_dict()).features.sandbox_enabled)
        for value in ('false', 0, None):
            with self.subTest(value=value), self.assertRaises(ConfigError):
                Config.from_dict({'features': {'sandbox_enabled': value}})

    def test_gateway_repository_default_uses_local_codex_home_without_reinterpreting_archives(self):
        # '~' expands from HOME (Path.expanduser; the password database only when HOME is unset), so the cleared
        # environment keeps an explicit HOME: the expectation must not depend on whether the test process's HOME
        # matches the password-database home (it does not on a host whose test run overrides HOME).
        home = self.root / 'home'
        for explicit in (None, {'features': {'enable_wait': False}}):
            with self.subTest(explicit=explicit):
                out = io.StringIO()
                argv = ['ni_rua', '--print-config']
                if explicit is not None:
                    argv.extend(['--config', str(self.save(explicit))])
                with patch.dict(os.environ, {'HOME': str(home), 'CODEX_HOME': '/unrelated/native-home',
                        'NI_RUA_PRICE_CARD': '/unrelated/old-price-card.json'}, clear=True), \
                        patch('ni_rua.core.config.LOCAL_CONFIG_PATH', self.root / 'absent.local.json'), \
                        patch('sys.argv', argv), \
                        patch('ni_rua.apps.gateway.cli.serve', new_callable=AsyncMock) as serve, \
                        contextlib.redirect_stdout(out):
                    main()
                serve.assert_not_called()
                config = json.loads(out.getvalue())
                model = config['model']
                self.assertEqual((model['name'], model['effort']), ('gpt-6.1-sol', 'high'))
                self.assertEqual(model['codex_home'], str(home / '.codex'))
                self.assertIsNone(model['price_card'])
                self.assertTrue(model['native_memories'])
                if explicit is not None:
                    self.assertFalse(config['features']['enable_wait'])
        with patch.dict(os.environ, {'HOME': str(home)}):
            archived = Config.from_dict({'version': 1})
        self.assertEqual(archived.model.name, 'gpt-6.1-sol')
        self.assertEqual(archived.model.codex_home, str(home / '.codex'))
        self.assertTrue(archived.model.native_memories)

    def test_local_defaults_and_explicit_overrides_resolve_each_file_origin(self):
        local = self.root / 'local/ni-rua.local.json'
        local.parent.mkdir()
        local.write_text(json.dumps({'model': {'codex_home': 'provider', 'effort': 'low'}}))
        with patch('ni_rua.core.config.LOCAL_CONFIG_PATH', local):
            defaults = Config.load(environ={})
            self.assertEqual(defaults.model.name, 'gpt-6.1-sol')
            self.assertEqual(defaults.model.effort, 'low')
            self.assertEqual(defaults.model.codex_home, str(local.parent / 'provider'))
            self.save({'features': {'enable_wait': False}})
            partial = Config.load(environ={'NI_RUA_CONFIG': str(self.path)})
            self.assertFalse(partial.features.enable_wait)
            self.assertEqual(partial.model, defaults.model)
            self.save({'model': {'codex_home': 'explicit-provider', 'effort': 'medium'}})
            explicit = Config.load(self.path, environ={'NI_RUA_CONFIG': str(self.root / 'missing.json')})
            self.assertEqual(explicit.model.codex_home, str(self.root / 'explicit-provider'))
            self.assertEqual(explicit.model.effort, 'medium')
            override = Config.load(self.path, environ={}, overrides={'model': {'codex_home': 'cli-provider'}})
            self.assertEqual(override.model.codex_home, str(Path.cwd() / 'cli-provider'))

    def test_missing_local_file_preserves_public_defaults(self):
        with patch('ni_rua.core.config.LOCAL_CONFIG_PATH', self.root / 'missing.local.json'):
            config = Config.load(environ={})
        model = config.model
        self.assertEqual(config.queue.max_parallel, 60)
        self.assertIsNone(config.queue.eligible_gpu_indices)
        self.assertEqual(config.queue.memory_per_case_mib, 10240)
        self.assertEqual(config.queue.gpu_headroom_mib, 0)
        self.assertEqual(Config().queue.max_parallel, 60)
        self.assertIsNone(self.default_config.queue.eligible_gpu_indices)
        self.assertIsNone(Config.from_dict({'queue': {'max_parallel': None}}).queue.max_parallel)
        self.assertEqual(Config.from_dict({'queue': {'eligible_gpu_indices': [0, 2, 7]}}).queue.eligible_gpu_indices, (0, 2, 7))
        for bad in ([], [-1], [1, 1], [True]):
            with self.assertRaises(ConfigError):
                Config.from_dict({'queue': {'eligible_gpu_indices': bad}})
        self.assertEqual((model.name, model.effort), ('gpt-6.1-sol', 'high'))
        self.assertEqual(model.verbosity, 'high')
        self.assertEqual(self.default_config.model.verbosity, 'high')
        self.assertEqual(model.codex_home, str(Path('~/.codex').expanduser().resolve()))
        self.assertFalse(config.features.sandbox_enabled)
        self.assertFalse(self.default_config.features.sandbox_enabled)
        self.assertTrue(config.features.external_perception)
        self.assertTrue(config.features.vggt_height)
        self.assertTrue(config.features.object_table)
        self.assertTrue(config.features.scene3d)
        self.assertTrue(config.features.object_table_observe)
        self.assertTrue(config.features.scene3d_observe)
        self.assertFalse(config.features.persistent_object_tracking)
        self.assertFalse(config.features.multiview_scene3d)
        self.assertFalse(self.default_config.features.persistent_object_tracking)
        self.assertFalse(self.default_config.features.multiview_scene3d)
        self.assertFalse(config.features.held_perception)
        self.assertEqual(config.features.action_trajectory_delivery, 'automatic')
        self.assertEqual(config.images.action_feedback_policy, 'all')
        self.assertFalse(Config.from_dict({'features': {'external_perception': True}}).features.held_perception)
        self.assertFalse(Config.from_dict({'features': {'external_perception': True}}).features.persistent_object_tracking)
        self.assertFalse(Config.from_dict({'features': {'external_perception': True}}).features.multiview_scene3d)
        self.assertTrue(Config.from_dict({'features': {'external_perception': True, 'held_perception': True}}).features.held_perception)
        self.assertTrue(Config.from_dict({'features': {'external_perception': True, 'persistent_object_tracking': True}}).features.persistent_object_tracking)
        self.assertTrue(Config.from_dict({'features': {'external_perception': True, 'multiview_scene3d': True}}).features.multiview_scene3d)
        self.assertFalse(config.features.self_briefing)
        self.assertTrue(config.features.startup_inline_grounding)
        self.assertEqual(config.features.startup_grounding_views, 'overview')
        self.assertEqual(config.features.grip_non_empty_min_aperture_m, 0.0015)
        self.assertEqual(config.features.grip_rigid_min_aperture_m, 0.002)
        self.assertEqual(config.features.grip_wide_min_aperture_m, 0.003)
        # 校验 configs/ni-rua.example.json 完整覆盖 Config().to_dict() 全部字段且与默认开关一致
        example_raw = self.example_raw
        default_raw = self.default_config.to_dict()
        self.assertEqual(set(example_raw.keys()), set(default_raw.keys()))
        for section, values in default_raw.items():
            if isinstance(values, dict):
                self.assertEqual(set(example_raw[section].keys()) - set(values.keys()), set(), section)
                self.assertLessEqual(set(values.keys()) - set(example_raw[section].keys()),
                                     {'vggt_camera_id', 'deployment'}, section)
        example_cfg = Config.from_dict(example_raw)
        self.assertFalse(example_cfg.features.sandbox_enabled)
        self.assertEqual(example_cfg.model.verbosity, 'high')
        self.assertFalse(example_cfg.features.persistent_object_tracking)
        self.assertFalse(example_cfg.features.multiview_scene3d)
        self.assertEqual(example_cfg.queue, self.default_config.queue)

    def test_local_file_is_strictly_validated_before_explicit_overrides(self):
        local = self.root / 'ni-rua.local.json'
        self.save({'model': {'effort': 'high'}})
        for document in ('broken JSON', '[]', '{"model":{"api_key":"synthetic-secret"}}',
                         '{"model":{"effort":false}}'):
            with self.subTest(document=document), patch('ni_rua.core.config.LOCAL_CONFIG_PATH', local):
                local.write_text(document)
                with self.assertRaises(ConfigError):
                    Config.load(self.path, environ={}, overrides={'model': {'effort': 'high'}})

    def test_suite_reads_neither_the_checkout_local_file_nor_the_account_home(self):
        # Zone package import: Config.load() skips the checkout's git-ignored configs/ni-rua.local.json; this
        # module's HOME is private, so the default grasp memory store is not the one of the account running the suite.
        from ni_rua.core import config as config_module
        self.assertEqual(Path(config_module.LOCAL_CONFIG_PATH), ABSENT_LOCAL_CONFIG)
        self.assertNotEqual(ABSENT_LOCAL_CONFIG, CONFIGS_ROOT / 'ni-rua.local.json')
        self.assertFalse(ABSENT_LOCAL_CONFIG.exists())
        home = Path(os.environ['HOME']).resolve()
        self.assertNotEqual(home, Path(pwd.getpwuid(os.getuid()).pw_dir).resolve())
        self.assertEqual(Path(Config.from_dict({}).features.grasp_memory_dir), home / '.local/share/ni-rua/grasp-memory')

    def test_archived_config_and_queue_ignore_changed_local_defaults(self):
        from scripts.benchmark_queue import queue_config
        local = self.root / 'ni-rua.local.json'
        local.write_text(json.dumps({'model': {'codex_home': 'original-provider'}}))
        with patch('ni_rua.core.config.LOCAL_CONFIG_PATH', local):
            frozen = Config.load(environ={})
            snapshot = frozen.to_dict()
            local.write_text('broken JSON')
            self.assertEqual(Config.from_dict(snapshot), frozen)
            with patch.object(Config, 'load', side_effect=AssertionError('archive must not reload')):
                self.assertEqual(queue_config({'ni_rua_config': snapshot}), frozen)
            self.assertEqual(Config.from_dict({}).model.name, 'gpt-6.1-sol')
            self.assertEqual(Config.from_dict({}).model.codex_home, str(Path('~/.codex').expanduser().resolve()))

    def test_packaged_defaults_without_checkout_still_accept_explicit_config(self):
        package = self.root / 'package'
        package.mkdir()
        (package / 'default_config.json').write_text(json.dumps({'model': {'name': 'package-model'}}))
        self.save({'model': {'effort': 'medium'}})
        with patch('ni_rua.core.config.__file__', str(package / 'config.py')), \
                patch('ni_rua.core.config.LOCAL_CONFIG_PATH', self.root / 'absent.local.json'):
            self.assertEqual(Config.load(environ={}).model.name, 'package-model')
            explicit = Config.load(self.path, environ={})
        self.assertEqual((explicit.model.name, explicit.model.effort), ('package-model', 'medium'))

    def test_gateway_explicit_model_preserves_old_configuration_and_relative_home(self):
        self.save({'model': {'name': 'gpt-6-astra', 'effort': 'xhigh',
                             'codex_home': 'archived-auth', 'price_card': None}})
        out = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), \
                patch('sys.argv', ['ni_rua', '--config', str(self.path), '--print-config']), \
                patch('ni_rua.apps.gateway.cli.serve', new_callable=AsyncMock) as serve, \
                contextlib.redirect_stdout(out):
            main()
        serve.assert_not_called()
        model = json.loads(out.getvalue())['model']
        self.assertEqual((model['name'], model['effort']), ('gpt-6-astra', 'xhigh'))
        self.assertEqual(model['codex_home'], str(self.root / 'archived-auth'))
        self.assertIsNone(model['price_card'])

    def test_darwin_home_paths_skip_realpath_but_keep_the_resolved_form(self):
        from ni_rua.core.config import sections
        home_path = Path('/home/ckpt/run/../model')
        with patch.object(sections.sys, 'platform', 'darwin'), \
                patch.object(Path, 'resolve', side_effect=AssertionError('resolve must not touch /home on Darwin')):
            self.assertEqual(sections._absolute_path(home_path), '/System/Volumes/Data/home/ckpt/model')
        escaping = Path('/home/../../../../bin/sh')  # climbs out of /home: real resolution, never the lexical form
        with patch.object(sections.sys, 'platform', 'darwin'):
            self.assertEqual(sections._absolute_path(escaping), str(escaping.resolve()))
        with patch.object(sections.sys, 'platform', 'darwin'):
            self.assertEqual(sections._absolute_path(Path('/tmp/ni-rua-x')), str(Path('/tmp/ni-rua-x').resolve()))
        with patch.object(sections.sys, 'platform', 'linux'):
            self.assertEqual(sections._absolute_path(home_path), str(home_path.resolve()))

    def test_strict_schema_types_and_bounds(self):
        self.assertEqual(Config.from_dict({'version': 1}).version, 2)
        self.assertEqual(Config.from_dict({'version': 2}).version, 2)
        for value in ([], {'version': 0}, {'version': 3}, {'version': True}, {'unknown': {}},
                      {'features': {'typo': True}},
                      {'features': {'enable_wait': 1}}, {'features': {'enable_wait': None}},
                      {'features': {'disabled_tools': 'robot_wait'}}, {'images': {'jpeg_quality': 0}},
                      {'run': {'gpu': True}}, {'run': {'wall_budget_s': float('nan')}},
                      {'model': {'name': ''}}, {'queue': {'oom_retry': {'max_retries': 21}}},
                      {'queue': {'oom_retry': {'initial_delay_s': 0}}}):
            with self.subTest(value=value), self.assertRaises(ConfigError):
                Config.from_dict(value)

    def test_invalid_file_not_masked_by_explicit_override(self):
        self.save({'features': {'enable_wait': 'yes'}})
        with self.assertRaises(ConfigError):
            Config.load(self.path, environ={}, overrides={'features': {'enable_wait': True}})

    def test_legacy_profile_disables_calibration_dependent_tools(self):
        config = Config.from_dict({'features': {'observation_profile': 'rgb_proprio_v1'}})
        self.assertFalse(config.features.motion_review)
        self.assertTrue(config.features.enable_wait)

    def test_depth_features_default_off_and_require_explicit_sensor_profile(self):
        defaults = Config.from_dict({})
        self.assertEqual(defaults.features.observation_profile, ACTOR_PROFILE)
        for feature in ("depth_skills", "depth_auto_feedback", "depth_event_payload"):
            self.assertFalse(getattr(defaults.features, feature))
            with self.subTest(feature=feature):
                with self.assertRaisesRegex(ConfigError, "depth_features_require_rgbd"):
                    Config.from_dict({"features": {feature: True}})
                enabled = Config.from_dict({"features": {"observation_profile": DEPTH_ACTOR_PROFILE, feature: True,
                    **({"depth_auto_feedback": True} if feature == "depth_event_payload" else {})}})
                self.assertTrue(getattr(enabled.features, feature))
        with self.assertRaisesRegex(ConfigError, "requires_depth_auto_feedback"):
            Config.from_dict({"features": {"observation_profile": DEPTH_ACTOR_PROFILE, "depth_event_payload": True}})
        rgbd = Config.from_dict({"features": {"observation_profile": DEPTH_ACTOR_PROFILE}})
        self.assertTrue(rgbd.features.motion_review)
        self.assertFalse(rgbd.features.depth_skills)
        self.assertFalse(rgbd.features.depth_auto_feedback)
        self.assertFalse(rgbd.features.depth_event_payload)

    def test_cli_file_and_explicit_false_are_resolved_before_serve(self):
        self.save({'run': {'task': 'configured', 'tag': ['from-file']},
                   'features': {'tracking_guard': True, 'visual_review_model': 'gpt-6.1-sol', 'enable_wait': False}})
        with patch('sys.argv', ['ni_rua', '--config', str(self.path), '--enable-wait',
                                '--no-tracking-guard', '--no-visual-review-model', '--clear-tags']), \
                patch('ni_rua.apps.gateway.cli.serve', new_callable=AsyncMock) as serve:
            main()
        actual = serve.call_args.args[0]
        self.assertEqual(actual.task, 'configured')
        self.assertTrue(actual.enable_wait)
        self.assertFalse(actual.tracking_guard)
        self.assertIsNone(actual.visual_review_model)
        self.assertEqual(actual.tag, ())

    def test_print_config_starts_nothing_and_rejects_unknown_tools(self):
        self.save({'features': {'terminal': False}})
        out = io.StringIO()
        with patch('sys.argv', ['ni_rua', '--config', str(self.path), '--print-config']), \
                patch('ni_rua.apps.gateway.cli.serve', new_callable=AsyncMock) as serve, contextlib.redirect_stdout(out):
            main()
        serve.assert_not_called()
        self.assertFalse(json.loads(out.getvalue())['features']['terminal'])
        self.save({'features': {'disabled_tools': ['misspelled_tool']}})
        with patch('sys.argv', ['ni_rua', '--config', str(self.path), '--print-config']), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main()

    def test_tool_disabled_in_direct_and_program_dispatch(self):
        identity = dict(run_id='config', episode_id='episode', env_instance_id='env', env_generation=1)
        store = Store(self.root / 'run', identity)
        self.addCleanup(store.close)
        config = Config.from_dict({'features': {'disabled_tools': ['robot_wait', 'robot_capabilities'], 'observation_profile': 'rgb_proprio_v1'}})
        harness = Harness(store, WaitFixtureRobot(self.root), config=config)
        harness.start()
        with self.assertRaisesRegex(ValueError, 'tool_disabled_by_config:robot_wait'):
            harness.dispatch('robot_wait', {'native_steps': 1, 'observation_id': harness.latest['observation_id']}, {'tool_call_id': 'direct'})
        runner = ProgramRunner(program_services(harness))
        try:
            result = runner.run({'source': "from ni_rua_client import NiRuaClient\nr = NiRuaClient().call('robot_capabilities')\nassert r['success'] is False\nassert 'tool_disabled_by_config:robot_capabilities' in r['contentItems'][0]['text']", 'timeout_s': 5}, {'thread_id': 'thread', 'turn_id': 'turn', 'tool_call_id': 'program'})
            self.assertEqual(result['status'], 'succeeded', store.resolve_artifact(result['stderr_ref'])[1])
            self.assertEqual(harness.robot.native_count, 0)
        finally:
            runner.close()


class ArchivedConfigTests(unittest.IsolatedAsyncioTestCase):
    async def test_fresh_workbench_consumes_one_config_and_persists_effective_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = Config.from_dict({'features': {'observation_profile': 'rgb_proprio_v1', 'enable_wait': False,
                                                   'disabled_tools': ['program_run']},
                                       'run': {'task': 'configured', 'seed': 3, 'layout_id': 2},
                                       'images': {'reuse_exact_images': False}, 'model': {'name': 'configured-model'}})
            with patch.object(Config, 'load', side_effect=AssertionError('unexpected reload')):
                workbench = Workbench(root / 'new', config=config)
            try:
                self.assertEqual((workbench.task, workbench.seed, workbench.layout_id), ('configured', 3, 2))
                self.assertNotIn('robot_wait', [tool['name'] for tool in workbench.actor_tools()])
                self.assertNotIn('program_run', [tool['name'] for tool in workbench.actor_tools()])
                self.assertEqual(workbench.session.model, 'configured-model')
                snapshot = Config.from_dict(json.loads((root / 'new/effective_config.json').read_text()))
                self.assertEqual(snapshot, workbench.config)
                self.assertEqual(workbench.harness.image_settings, snapshot.images.to_dict())
            finally:
                await workbench.close()

    async def test_resume_and_replay_restore_snapshot_instead_of_new_flags(self):
        for mode in ('resume', 'replay'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                saved = Config.from_dict({'features': {'enable_wait': False, 'motion_review': False,
                                                       'pi05': True, 'disabled_tools': ['program_run'],
                                                       'completion_guidance': 'none'},
                                          'run': {'task': 'archived', 'seed': 7, 'layout_id': 3},
                                          'model': {'name': 'archived-model'}, 'images': {'transport_format': 'jpeg'}})
                manifest = dict(schema_version='0.1', run_id='run', episode_id='ep', env_instance_id='env', env_generation=1,
                                actor_information_policy=ACTOR_POLICY, observation_profile=ACTOR_PROFILE,
                                effective_config=saved.to_dict())
                (root / 'manifest.json').write_text(json.dumps(manifest))
                incoming = Config.from_dict({'features': {'enable_wait': True}, 'run': {'task': 'different'}})
                workbench = Workbench(root, config=incoming, **{mode: True})
                try:
                    self.assertFalse(workbench.enable_wait)
                    self.assertEqual(workbench.completion_guidance, 'none')
                    self.assertTrue(workbench.pi05_enabled)
                    self.assertEqual((workbench.task, workbench.seed, workbench.layout_id), ('archived', 7, 3))
                    self.assertEqual(workbench.session.model, 'archived-model')
                    self.assertNotIn('program_run', [tool['name'] for tool in workbench.actor_tools()])
                    self.assertEqual(workbench.config.images.transport_format, 'jpeg')
                    self.assertEqual(json.loads((root / 'manifest.json').read_text()), manifest)
                finally:
                    await workbench.close()

    async def test_an_archive_keeps_its_reply_form_whatever_the_current_defaults_are(self):
        """Resume / replay: an effective_config written before the reply-delivery settings existed restores the form
        it ran with (every setting absent = off), an archive that states them keeps what it states, and neither is
        overridden by the configuration of the resuming process."""
        from ni_rua.core.config import FEATURE_ABSENT_VALUES
        from support.reply_defaults import DEFAULT_REPLY_FEATURES, LEGACY_REPLY_FEATURES
        self.assertEqual(dict(FEATURE_ABSENT_VALUES, actor_payload_profile="lean_v1"), LEGACY_REPLY_FEATURES)
        current = Config().to_dict()
        old = json.loads(json.dumps(current))
        for name in FEATURE_ABSENT_VALUES:
            del old["features"][name]
        old["features"]["actor_payload_profile"] = "lean_v1"          # what such an archive recorded
        # (b) written after tool_catalog_profile existed but before robot_urdf_delivery and the reply switches: the
        # recorded catalogue profile is kept, the missing keys restore the form of the time (URDF inline).
        catalogue_only = json.loads(json.dumps(old))
        catalogue_only["features"]["tool_catalog_profile"] = "layered_v1"
        # (c) explicit full_v1 / inline written by a run that chose them after the defaults changed
        explicit_old = json.loads(json.dumps(current))
        explicit_old["features"].update(tool_catalog_profile="full_v1", robot_urdf_delivery="inline")
        mixed = json.loads(json.dumps(current))
        mixed["features"].update(reply_float32=False, action_reply_profile="full", actor_payload_profile="full_v1")
        cases = dict(old=(old, LEGACY_REPLY_FEATURES), current=(current, DEFAULT_REPLY_FEATURES),
                     catalogue_only=(catalogue_only, dict(LEGACY_REPLY_FEATURES, tool_catalog_profile="layered_v1")),
                     explicit_old=(explicit_old, dict(DEFAULT_REPLY_FEATURES, tool_catalog_profile="full_v1",
                                                      robot_urdf_delivery="inline")),
                     mixed=(mixed, dict(DEFAULT_REPLY_FEATURES, reply_float32=False, action_reply_profile="full",
                                        actor_payload_profile="full_v1")))
        for mode in ('resume', 'replay'):
            for name, (archived, expected) in cases.items():
                with self.subTest(mode=mode, archive=name), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    manifest = dict(schema_version='0.1', run_id='run', episode_id='ep', env_instance_id='env',
                                    env_generation=1, actor_information_policy=ACTOR_POLICY,
                                    observation_profile=ACTOR_PROFILE, effective_config=archived)
                    (root / 'manifest.json').write_text(json.dumps(manifest))
                    incoming = Config.from_dict({'features': dict(DEFAULT_REPLY_FEATURES)})
                    workbench = Workbench(root, config=incoming, **{mode: True})
                    try:
                        features = workbench.config.features
                        self.assertEqual({key: getattr(features, key) for key in expected}, expected)
                        self.assertEqual(workbench.harness.actor_payload_profile, expected["actor_payload_profile"])
                        action = next(tool for tool in workbench.actor_tools() if tool['name'] == 'robot_action')
                        self.assertEqual("Fixed reply notes" in action['description'], expected["reply_static_once"])
                        self.assertEqual("Execution reply layout" in action['description'],
                                         expected["action_reply_profile"] == "summary_first")
                    finally:
                        await workbench.close()

    def test_a_frozen_runtime_without_the_reply_fields_only_accepts_the_values_that_mean_absent(self):
        """Queue projection onto an older checkout: the reply-delivery fields may be left out only with the value an
        older runtime behaves as (off / full); the current defaults fail closed instead of being dropped silently."""
        from ni_rua.apps.evaluation import queue
        from ni_rua.core.config import FEATURE_ABSENT_VALUES
        schema = {name: set(fields) for name, fields in queue.runtime_config_schema(REPO_ROOT).items()}
        schema["FeaturesConfig"] -= set(FEATURE_ABSENT_VALUES)
        absent = Config.from_dict({"features": dict(FEATURE_ABSENT_VALUES)}).to_dict()
        omissions = []
        projected = queue.project_config_for_runtimes(absent, [schema], omissions)
        self.assertEqual(set(FEATURE_ABSENT_VALUES) & set(projected["features"]), set())
        self.assertEqual({item["key"]: item["value"] for item in omissions},
                         {"features." + name: value for name, value in FEATURE_ABSENT_VALUES.items()})
        for name in FEATURE_ABSENT_VALUES:
            with self.subTest(name=name):
                document = json.loads(json.dumps(absent))
                document["features"][name] = Config().to_dict()["features"][name]     # the current default: on
                with self.assertRaisesRegex(ValueError, "unified_config_incompatible:features." + name):
                    queue.project_config_for_runtimes(document, [schema])
        # a runtime that has the fields takes the document as it is
        self.assertEqual(queue.project_config_for_runtimes(Config().to_dict(), [queue.runtime_config_schema(REPO_ROOT)])
                         ["features"]["action_reply_profile"], "summary_first")

    def test_a_frozen_runtime_that_lacks_a_catalogue_value_reports_incompatible_rather_than_receiving_or_dropping_it(self):
        """A checkout whose tool_catalog_profile / robot_urdf_delivery fields exist but do not list the new default
        value (layered_v3 / file) must be reported, not sent the value and not run without it; the values it lists
        pass, and a checkout that lists no choices is not second-guessed."""
        import tempfile
        from ni_rua.apps.evaluation import queue
        document = Config().to_dict()
        self.assertEqual((document["features"]["tool_catalog_profile"], document["features"]["robot_urdf_delivery"]),
                         ("layered_v3", "file"))

        # the choices are read statically from the checkout's FeaturesConfig (never executed)
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (root / "ni_rua" / "apps" / "gateway").mkdir(parents=True)
        (root / "ni_rua" / "apps" / "gateway" / "cli.py").write_text("")
        (root / "ni_rua" / "core").mkdir(parents=True)
        (root / "ni_rua" / "core" / "config.py").write_text(
            "class FeaturesConfig:\n"
            "    tool_catalog_profile: str = option('full_v1', choices=('full_v1', 'guide_only_v1', 'layered_v1'), cli=True)\n"
            "    robot_urdf_delivery: str = 'inline'\n"
            "class Config:\n    features: FeaturesConfig\n")
        parsed = queue.runtime_config_schema(root)
        self.assertEqual(parsed["FeaturesConfig.tool_catalog_profile#choices"], {"full_v1", "guide_only_v1", "layered_v1"})
        self.assertNotIn("FeaturesConfig.robot_urdf_delivery#choices", parsed)   # no listed choices: not second-guessed

        def checkout(profile_choices, urdf_choices):
            """The current schema with the choices a checkout of that age lists for the two settings."""
            schema = {name: set(fields) for name, fields in queue.runtime_config_schema(REPO_ROOT).items()}
            for name, choices in (("tool_catalog_profile", profile_choices), ("robot_urdf_delivery", urdf_choices)):
                schema.pop(f"FeaturesConfig.{name}#choices", None)
                if choices is not None:
                    schema[f"FeaturesConfig.{name}#choices"] = set(choices)
            return schema

        old = checkout(("full_v1", "guide_only_v1", "layered_v1"), ("inline",))
        for name, other in (("tool_catalog_profile", "robot_urdf_delivery"), ("robot_urdf_delivery", "tool_catalog_profile")):
            single = json.loads(json.dumps(document))   # only this setting has a value the checkout does not list
            single["features"][other] = {"tool_catalog_profile": "full_v1", "robot_urdf_delivery": "inline"}[other]
            with self.subTest(name=name), self.assertRaisesRegex(
                    ValueError, "unified_config_incompatible:features." + name + "=" + document["features"][name]):
                queue.project_config_for_runtimes(single, [old])
        # the values the old checkout lists are passed on unchanged
        listed = json.loads(json.dumps(document))
        listed["features"].update(tool_catalog_profile="layered_v1", robot_urdf_delivery="inline")
        self.assertEqual(queue.project_config_for_runtimes(listed, [old])["features"]["tool_catalog_profile"], "layered_v1")
        # one incompatible checkout among several is enough
        with self.assertRaisesRegex(ValueError, "unified_config_incompatible:features.tool_catalog_profile"):
            queue.project_config_for_runtimes(document, [queue.runtime_config_schema(REPO_ROOT), old])
        # a checkout that knows the values, or does not list its choices, takes the document
        new = checkout(("full_v1", "layered_v3"), ("inline", "file"))
        self.assertEqual(queue.project_config_for_runtimes(document, [new]), document)
        unlisted = checkout(None, None)
        self.assertEqual(queue.project_config_for_runtimes(document, [unlisted]), document)


class LatestFeatureConfigTests(unittest.IsolatedAsyncioTestCase):
    def test_latest_main_defaults_and_cli_overrides(self):
        config = Config.from_dict({})
        self.assertEqual(config.features.actor_payload_profile, 'lean_v2')
        self.assertTrue(config.features.default_task_notes)
        self.assertTrue(config.features.vision_compare)
        self.assertEqual(config.features.completion_guidance, 'hook')
        self.assertEqual(config.queue.memory_per_case_mib, 10240)
        self.assertIsNone(config.queue.eligible_gpu_indices)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'config.json'
            path.write_text(json.dumps({'features': {'default_task_notes': False, 'vision_compare': False,
                                                    'actor_payload_profile': 'full_v1'}}))
            for flags, expected in (([], False), (['--default-task-notes', '--vision-compare'], True)):
                with patch('sys.argv', ['ni_rua', '--config', str(path), *flags]), \
                        patch('ni_rua.apps.gateway.cli.serve', new_callable=AsyncMock) as serve:
                    main()
                args = serve.call_args.args[0]
                self.assertEqual(args.default_task_notes, expected)
                self.assertEqual(args.vision_compare, expected)
                self.assertEqual(args.actor_payload_profile, 'full_v1')
                self.assertEqual(args.completion_guidance, 'hook')
                self.assertFalse(args.benchmark)  # Default hook must not block development launches.

    async def test_older_unified_snapshot_does_not_enable_new_prompt_notes(self):
        for mode in ('resume', 'replay'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                saved = Config.from_dict({'features': {'motion_review': False, 'actor_payload_profile': 'full_v1'}}).to_dict()
                del saved['features']['default_task_notes']
                del saved['features']['vision_compare']
                del saved['features']['completion_guidance']
                manifest = dict(schema_version='0.1', run_id='run', episode_id='ep', env_instance_id='env', env_generation=1,
                                actor_information_policy=ACTOR_POLICY, observation_profile=ACTOR_PROFILE,
                                task_notes='archived exact notes', task_notes_source='archived source',
                                effective_config=saved)
                (root / 'manifest.json').write_text(json.dumps(manifest))
                workbench = Workbench(root, config=Config.from_dict({}), **{mode: True})
                try:
                    self.assertFalse(workbench.default_task_notes)
                    self.assertTrue(workbench.vision_compare)
                    self.assertTrue(workbench.harness.vision_compare)
                    self.assertEqual(workbench.actor_payload_profile, 'full_v1')
                    self.assertEqual(workbench.completion_guidance, 'none')
                    self.assertIsNone(workbench.harness.home_check)
                    self.assertEqual(workbench.archived_task_notes['text'], 'archived exact notes')
                    self.assertEqual(json.loads((root / 'manifest.json').read_text()), manifest)
                finally:
                    await workbench.close()

    async def test_config_disables_comparison_in_harness_and_model_tools(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Config.from_dict({'features': {'vision_compare': False, 'default_task_notes': False,
                                                    'motion_review': False}})
            workbench = Workbench(Path(temporary) / 'run', config=config)
            try:
                self.assertFalse(workbench.default_task_notes)
                self.assertFalse(workbench.harness.vision_compare)
                available = {tool['name'] for tool in workbench.actor_tools()}
                self.assertNotIn('vision_compare', available)
                self.assertNotIn('skill_run', available)
                with self.assertRaisesRegex(ValueError, 'vision_compare_not_enabled_for_run'):
                    workbench.harness.dispatch('vision_compare', {}, {'tool_call_id': 'disabled'})
            finally:
                await workbench.close()

    async def test_legacy_resolved_notes_remain_separate_from_custom_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            custom = 'x' * 8000
            combined = 'archived default notes\n\n' + custom
            manifest = dict(schema_version='0.1', run_id='run', episode_id='ep', env_instance_id='env', env_generation=1,
                            actor_information_policy=ACTOR_POLICY, observation_profile=ACTOR_PROFILE,
                            benchmark=True, task_notes=combined, task_notes_source='default; custom',
                            task_notes_config=dict(default_enabled=True, default_version='archived-version',
                                                   custom_text=custom, custom_source='custom'))
            (root / 'manifest.json').write_text(json.dumps(manifest))
            workbench = Workbench(root, replay=True, config=Config.from_dict({}))
            try:
                self.assertEqual(workbench.config.run.task_notes, custom)
                self.assertEqual(workbench.config.run.task_notes_source, 'custom')
                self.assertEqual(workbench.archived_task_notes['text'], combined)
                self.assertEqual(workbench.archived_task_notes['config']['default_version'], 'archived-version')
                # The archived custom settings can still be reused for a fresh episode.
                workbench.config.with_overrides({'run': {'replay': False}}).validate_launch()
            finally:
                await workbench.close()

    def test_motion_thresholds_unified_in_config_and_propagated_to_consumers(self):
        from ni_rua.core.config import MOTION_THRESHOLD_FIELDS, motion_thresholds
        from ni_rua.robot.backends.robodojo.robodojo import RobotBackend

        repo_json = json.loads((CONFIGS_ROOT / 'ni-rua.json').read_text())
        self.assertEqual(repo_json['version'], 2)
        self.assertEqual(repo_json['embodiment'], {'thresholds': {}})
        defaults = Config()
        for field_name in MOTION_THRESHOLD_FIELDS:
            self.assertNotIn(field_name, repo_json['features'], field_name)
            self.assertIn(field_name, defaults.embodiment.thresholds, field_name)
        custom = Config.from_dict({
            'features': {
                'default_position_tolerance_m': 0.006,
                'soft_position_tolerance_m': 0.012,  # retired by design 13 V-E: accepted on load, then dropped
                'relaxed_position_tolerance_m': 0.014,
                'compliant_position_tolerance_m': 0.020,
                'grip_empty_threshold_mm': 0.8,
                'grip_holding_threshold_mm': 4.0,
                'orientation_tolerance_rad': 0.15,
            }
        })
        thresholds = motion_thresholds(custom)
        self.assertEqual(thresholds['default_position_tolerance_m'], 0.006)
        self.assertNotIn('soft_position_tolerance_m', thresholds)
        self.assertEqual(thresholds['relaxed_position_tolerance_m'], 0.014)
        self.assertEqual(custom.embodiment.thresholds.default_position_tolerance_m, 0.006)
        self.assertEqual(custom.features.default_position_tolerance_m, 0.006)
        robot = RobotBackend(config=custom)
        self.assertEqual(robot.motion_thresholds['orientation_tolerance_rad'], 0.15)
        self.assertEqual(robot.motion_thresholds['default_position_tolerance_m'], 0.006)


# ======================================================================
# 外部感知主开关与子开关隔离测试 (迁自 test_external_perception_switch.py)
# ======================================================================


def _catalog_runtime(config):
    from types import SimpleNamespace
    return SimpleNamespace(
        disabled_tools=(),
        observation_profile="rgb_proprio_calibrated_v1",
        motion_review=True,
        enable_wait=True,
        vision_compare=True,
        visual_reviewer=True,
        action_policy=None,
        capabilities=None,
        programs=True,
        config=config,
        features=config.features,
    )


class ExternalPerceptionSwitchTest(unittest.TestCase):
    def test_default_enables_external_perception_stack(self):
        from ni_rua.core.contracts import canonical
        from ni_rua.perception.external import object_table, obs3d, scene3d, vggt_height
        from ni_rua.runtime.tools.builtins import create_registry

        cfg = Config()
        self.assertTrue(cfg.features.external_perception)
        for flag in ("vggt_height", "object_table", "scene3d", "object_table_observe", "scene3d_observe"):
            self.assertTrue(getattr(cfg.features, flag), flag)
        self.assertFalse(cfg.features.held_perception)
        self.assertFalse(cfg.features.pose_skill)
        self.assertFalse(cfg.features.persistent_object_tracking)
        self.assertFalse(cfg.features.multiview_scene3d)
        self.assertEqual(cfg.features.vggt_height_mode, "optional")
        self.assertEqual(vggt_height.mode(cfg), "optional")
        self.assertTrue(object_table.enabled(cfg))
        self.assertTrue(scene3d.enabled(cfg))
        self.assertEqual(obs3d.flags(cfg), (True, True))
        self.assertEqual(scene3d.workspace_modules(cfg), ("scene3d_helpers.py",))
        rt = _catalog_runtime(cfg)
        catalog = {item["name"]: item for item in create_registry(runtime=rt).catalog(rt, role="actor")}
        self.assertNotIn("pose_plan", catalog)
        self.assertIn("object_table", catalog["geometry_query"]["description"])
        self.assertIn("scene3d", catalog["geometry_query"]["description"])
        self.assertNotIn("reanchor_tracks", catalog["geometry_query"]["description"])
        self.assertIn("VGGT", catalog["robot_observe"]["description"])
        dev_prompt = obs3d.developer_instructions(cfg.features)
        self.assertIn("## 3D Perception (VGGT + SAM)", dev_prompt)
        self.assertIn("object_table", dev_prompt)
        self.assertIn("scene3d_helpers", dev_prompt)
        self.assertNotIn("reanchor_tracks", dev_prompt)
        cfg_track = cfg.with_overrides({"features": {"persistent_object_tracking": True}})
        rt_track = _catalog_runtime(cfg_track)
        catalog_track = {item["name"]: item for item in create_registry(runtime=rt_track).catalog(rt_track, role="actor")}
        self.assertIn("reanchor_tracks", catalog_track["geometry_query"]["description"])
        self.assertIn("reanchor_tracks", obs3d.developer_instructions(cfg_track.features))
        _ = canonical

    def test_explicit_false_restores_native_standalone_mode(self):
        from ni_rua.core.contracts import canonical
        from ni_rua.perception.external import obs3d
        from ni_rua.runtime.tools.builtins import create_registry

        cfg = Config().with_overrides({
            "features": {
                "persistent_object_tracking": True,
                "multiview_scene3d": True,
                "held_perception": True,
            }
        }).with_overrides({"features": {"external_perception": False,
                                        # create_registry() below has no config: it publishes the prose catalogue
                                        "tool_catalog_profile": "full_v1"}})
        self.assertFalse(cfg.features.external_perception)
        for flag in (
            "vggt_height", "object_table", "scene3d", "object_table_observe", "scene3d_observe",
            "held_perception", "pose_skill", "persistent_object_tracking", "multiview_scene3d",
        ):
            self.assertFalse(getattr(cfg.features, flag), flag)
        self.assertEqual(cfg.features.vggt_height_mode, "optional")
        self.assertEqual(obs3d.developer_instructions(cfg.features), "")
        rt = _catalog_runtime(cfg)
        self.assertEqual(
            canonical(create_registry(runtime=rt).catalog(rt, role="actor")),
            canonical(create_registry().catalog(rt, role="actor")),
        )

    def test_direct_dataclass_construction_and_reversible_toggle(self):
        from ni_rua.core.config import FeaturesConfig

        feat = FeaturesConfig(external_perception=True)
        self.assertTrue(feat.vggt_height)
        self.assertTrue(feat.object_table)
        self.assertTrue(feat.scene3d)
        self.assertTrue(feat.object_table_observe)
        self.assertTrue(feat.scene3d_observe)
        self.assertFalse(feat.held_perception)
        self.assertFalse(feat.pose_skill)
        self.assertFalse(feat.persistent_object_tracking)
        self.assertFalse(feat.multiview_scene3d)
        self.assertEqual(feat.vggt_height_mode, "optional")

        feat_held_on = FeaturesConfig(
            external_perception=True,
            held_perception=True,
            persistent_object_tracking=True,
            multiview_scene3d=True,
        )
        self.assertTrue(feat_held_on.held_perception)
        self.assertTrue(feat_held_on.persistent_object_tracking)
        self.assertTrue(feat_held_on.multiview_scene3d)

        feat_off = FeaturesConfig(
            external_perception=False,
            persistent_object_tracking=True,
            multiview_scene3d=True,
        )
        for flag in (
            "vggt_height", "object_table", "scene3d", "object_table_observe", "scene3d_observe",
            "held_perception", "pose_skill", "persistent_object_tracking", "multiview_scene3d",
        ):
            self.assertFalse(getattr(feat_off, flag), flag)

        cfg_on = Config.from_dict({"features": {"external_perception": True}})
        self.assertFalse(cfg_on.features.persistent_object_tracking)
        self.assertFalse(cfg_on.features.multiview_scene3d)
        cfg_off = cfg_on.with_overrides({"features": {"external_perception": False}})
        self.assertFalse(cfg_off.features.external_perception)
        for flag in (
            "vggt_height", "object_table", "scene3d", "object_table_observe", "scene3d_observe",
            "held_perception", "pose_skill", "persistent_object_tracking", "multiview_scene3d",
        ):
            self.assertFalse(getattr(cfg_off.features, flag), flag)
        self.assertEqual(cfg_off.features.vggt_height_mode, "optional")

    def test_master_switch_respects_explicit_sub_switch_override(self):
        cfg = Config.from_dict({"features": {"external_perception": True, "pose_skill": False, "vggt_height_mode": "default"}})
        self.assertTrue(cfg.features.external_perception)
        self.assertTrue(cfg.features.object_table)
        self.assertTrue(cfg.features.object_table_observe)
        self.assertFalse(cfg.features.pose_skill)
        self.assertEqual(cfg.features.vggt_height_mode, "default")


# ======================================================================
# RGB 视觉比对开关与归档恢复测试 (迁自 test_vision_compare_flag.py)
# ======================================================================


class VisionCompareFlagTests(unittest.TestCase):
    def test_queue_preserves_legacy_commands_and_emits_explicit_ablation(self):
        from scripts import benchmark_queue

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = dict(cwd=str(REPO_ROOT), run_dir=str(root / "run"),
                        task="general_pickup", seed=0, layout_id=0, title="RGB test", tag=["fixture"])
            self.assertNotIn("--vision-compare", benchmark_queue.command(case, 0))
            for enabled, flag in ((True, "--vision-compare"), (False, "--no-vision-compare")):
                self.assertIn(flag, benchmark_queue.command({**case, "vision_compare": enabled}, 0))
            for invalid in (0, "false", None):
                with self.assertRaisesRegex(ValueError, "Invalid vision_compare"):
                    benchmark_queue.command({**case, "vision_compare": invalid}, 0)

    def test_disabled_dispatch_blocks_direct_and_program_calls_before_work(self):
        import threading
        import time
        from support.fixture_robot import FixtureRobot

        temporary = self.enterContext(tempfile.TemporaryDirectory())
        store = Store(Path(temporary) / "store", dict(run_id="run", episode_id="ep", env_instance_id="env", env_generation=1))
        self.addCleanup(store.close)
        robot = FixtureRobot(temporary)
        harness = Harness(store, robot, vision_compare=False)
        harness.start()
        self.assertFalse(harness.capabilities["vision_compare_enabled"])
        with patch("ni_rua.perception.visual_feedback.compare_observations") as compare:
            with self.assertRaisesRegex(ValueError, "vision_compare_not_enabled_for_run"):
                harness.dispatch("vision_compare", {}, {"tool_call_id": "direct"})
            runner = ProgramRunner(program_services(harness))
            try:
                entry = dict(cancel=threading.Event(), deadline=time.monotonic() + 10,
                             call_lock=threading.RLock(), job_id="program-test", calls=0,
                             max_calls=1, correlation={"tool_call_id": "parent"})
                result = runner.dispatch(entry, dict(operation_id="comparison", tool="vision_compare", arguments={}))
                self.assertFalse(result["success"])
                self.assertEqual(json.loads(result["contentItems"][0]["text"])["error"], "vision_compare_not_enabled_for_run")
            finally:
                runner.close()
            compare.assert_not_called()
        self.assertEqual(robot.executions, 0)
        harness.vision_compare = True
        with patch("ni_rua.perception.visual_feedback.compare_observations", return_value={"fixture": True}) as compare:
            self.assertTrue(harness.dispatch("vision_compare", dict(before_observation_id="before",
                after_observation_id="after", camera_id="camera", roi=[0, 0, 1, 1]), {"tool_call_id": "enabled"})["success"])
            compare.assert_called_once()


class VisionCompareArchiveTests(unittest.IsolatedAsyncioTestCase):
    async def test_resume_and_replay_use_saved_setting_and_legacy_default(self):
        for mode in ("resume", "replay"):
            for saved in (None, False, True):
                with self.subTest(mode=mode, saved=saved), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary) / "run"
                    root.mkdir()
                    manifest = dict(schema_version="0.1", run_id="run", episode_id="ep", env_instance_id="env", env_generation=1,
                                    actor_information_policy=ACTOR_POLICY, observation_profile=ACTOR_PROFILE)
                    if saved is not None:
                        manifest["vision_compare"] = saved
                    (root / "manifest.json").write_text(json.dumps(manifest))
                    workbench = Workbench(root, **{mode: True}, vision_compare=not saved)
                    try:
                        expected = True if saved is None else saved
                        self.assertEqual(workbench.vision_compare, expected)
                        self.assertEqual(workbench.harness.vision_compare, expected)
                        self.assertEqual("vision_compare" in {tool["name"] for tool in workbench.actor_tools()}, expected)
                        self.assertEqual(json.loads((root / "manifest.json").read_text()), manifest)
                        self.assertIsNone(workbench.robot.process)
                    finally:
                        await workbench.close()


# ======================================================================
# 运行时消费者配置传播测试 (迁自 test_runtime_config.py)
# ======================================================================


class RuntimeConfigTest(unittest.TestCase):
    def setUp(self):
        self.temporary = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(self.temporary).resolve()

    def store(self):
        store = Store(self.root / "run", dict(run_id="config", episode_id="episode",
                      env_instance_id="environment", env_generation=1))
        self.addCleanup(store.close)
        return store

    def test_explicit_config_reaches_all_runtime_consumers_without_environment_reload(self):
        from ni_rua.apps.library.service import Library
        from ni_rua.apps.library.task_review import TaskReviewService
        from ni_rua.cortex.agents.codex.session import CodexSession
        from ni_rua.robot.backends.robodojo.robodojo import RobotBackend

        config = Config.from_dict({
            "run": {"task": "stack_bowls", "gpu": 2, "seed": 8, "layout_id": 3},
            "features": {"tracking_guard": True, "joint_replan": True, "observation_profile": "rgb_proprio_v1"},
            "backend": {"root": str(self.root / "official"), "python": str(self.root / "python")},
            "model": {"name": "fixture-model", "effort": "high", "codex_home": str(self.root / "codex")},
            "images": {"transport_format": "jpeg", "reuse_exact_images": False},
            "library": {"root": str(self.root / "library"), "scan_interval": 3},
        })
        store = self.store()
        with patch.object(Config, "load", side_effect=AssertionError("unexpected environment reload")):
            robot = RobotBackend(config=config)
            session = CodexSession(store.root, store, lambda _: {}, config=config)
            library = Library(config=config)
            review = TaskReviewService(config=config)
        self.assertEqual((robot.gpu_id, robot.seed, robot.layout_id), (2, 8, 3))
        self.assertEqual((robot.root, robot.python), (str(self.root / "official"), str(self.root / "python")))
        self.assertTrue(robot.tracking_guard)
        self.assertTrue(robot.joint_replan)
        self.assertEqual(robot.observation_profile, "rgb_proprio_v1")
        self.assertEqual(robot.image_settings, session.image_settings)
        self.assertFalse(session.image_settings["reuse_exact_images"])
        self.assertEqual((session.model, session.effort), ("fixture-model", "high"))
        self.assertEqual(library.scan_interval, 3)
        self.assertEqual(library.catalog.root, self.root / "library")
        self.assertEqual(library.task_reviews.root, review.root)
        self.assertEqual(review.root, self.root / "official")

    def test_explicit_constructor_arguments_override_config_even_when_false_or_zero(self):
        from ni_rua.cortex.agents.codex.session import CodexSession
        from ni_rua.robot.backends.robodojo.robodojo import RobotBackend

        config = Config.from_dict({"run": {"gpu": 2, "seed": 8},
                                   "features": {"tracking_guard": True, "joint_replan": True},
                                   "model": {"name": "configured-model"}})
        robot = RobotBackend(config=config, gpu_id=0, seed=0, tracking_guard=False, joint_replan=False)
        self.assertEqual((robot.gpu_id, robot.seed), (0, 0))
        self.assertFalse(robot.tracking_guard)
        self.assertFalse(robot.joint_replan)
        store = self.store()
        session = CodexSession(store.root, store, lambda _: {}, model="explicit-model", config=config)
        self.assertEqual(session.model, "explicit-model")

    def test_saved_image_policy_wins_over_new_config_and_legacy_missing_reuse_stays_disabled(self):
        from ni_rua.core.storage.images import image_settings, store_image_settings
        from ni_rua.cortex.agents.codex.session import CodexSession

        store = self.store()
        old = image_settings({"transport_format": "jpeg", "reuse_exact_images": False})
        old.pop("reuse_exact_images")
        store.set_state("runtime", "image_settings", old)
        config = Config.from_dict({"images": {"transport_format": "webp", "reuse_exact_images": True}})
        resolved = store_image_settings(store, config.images.to_dict())
        self.assertEqual(resolved["transport_format"], "jpeg")
        self.assertFalse(resolved["reuse_exact_images"])
        session = CodexSession(store.root, store, lambda _: {}, config=config)
        self.assertEqual(session.image_settings, resolved)

    def test_backend_explicit_none_flags_remain_invalid(self):
        from ni_rua.robot.backends.robodojo.robodojo import RobotBackend

        config = Config.from_dict({})
        for flag in ("tracking_guard", "joint_replan"):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "invalid_" + flag):
                RobotBackend(config=config, **{flag: None})

    def test_visual_helper_codex_home_overrides_ambient_home(self):
        from ni_rua.apps.reference.reference_api import ResponsesClient

        configured = self.root / "configured"
        configured.mkdir()
        (configured / "config.toml").write_text('model_provider="fixture"\n[model_providers.fixture]\nbase_url="https://configured.invalid/v1"\nenv_key="CONFIG_FIXTURE_TOKEN"\n')
        (configured / "auth.json").write_text('{"OPENAI_API_KEY":"synthetic-fixture"}')
        with patch.dict(os.environ, {"CODEX_HOME": str(self.root / "absent")}), \
                patch.object(ResponsesClient, "__init__", return_value=None) as constructor:
            ResponsesClient.from_codex_config(self.root / "helper", codex_home=configured, model="gpt-6.1-sol")
        constructor.assert_called_once_with("https://configured.invalid/v1", "synthetic-fixture",
                                             self.root / "helper", model="gpt-6.1-sol", effort="high")

    def test_geometry_service_and_child_use_frozen_images_without_environment_reload(self):
        from unittest.mock import MagicMock
        from ni_rua.perception.scene.scene_service import SceneService, _compute
        from ni_rua.robot.model.geometry import GeometryTools

        settings = Config.from_dict({"images": {"transport_format": "jpeg", "jpeg_quality": 61}}).images.to_dict()
        request = dict(run_id="geometry", episode_id="episode", env_instance_id="environment", env_generation=1)
        connection = MagicMock()
        guard = MagicMock()
        guard.prctl.return_value = 0
        with patch.object(Config, "load", side_effect=AssertionError("unexpected environment reload")):
            service = SceneService(self.root / "scene", image_settings=settings)
            self.addCleanup(service.store.close)
            geometry = GeometryTools(service.scene, settings=service.image_settings)
            self.assertEqual(geometry.image_settings, settings)
            with patch("ni_rua.perception.scene.scene_service.ctypes.CDLL", return_value=guard), \
                    patch.object(GeometryTools, "execute", return_value={"status": "fixture"}):
                _compute(str(service.root), request, connection, os.getppid(), service.image_settings)
        connection.send.assert_called_once_with({"result": {"status": "fixture"}})

    def test_image_settings_reads_unified_file_and_legacy_environment(self):
        from ni_rua.core.storage.images import image_settings

        path = self.root / "config.json"
        path.write_text(json.dumps({"images": {"webp_quality": 71}}))
        with patch.dict(os.environ, {"NI_RUA_CONFIG": str(path),
                                     "NI_RUA_IMAGE_SETTINGS": '{"webp_quality":30,"reuse_exact_images":false}'}):
            settings = image_settings()
        self.assertEqual(settings["webp_quality"], 71)
        self.assertFalse(settings["reuse_exact_images"])

    def test_library_print_config_applies_cli_overrides_without_starting_server(self):
        from ni_rua.apps.library.service import main as library_main

        path = self.root / "config.json"
        path.write_text(json.dumps({"library": {"port": 8790, "scan_interval": 2,
                                                "root": str(self.root / "catalog")}}))
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch("ni_rua.apps.library.service.serve", side_effect=AssertionError("server started")):
            library_main(["--config", str(path), "--port", "8791", "--print-config"])
        resolved = json.loads(output.getvalue())
        self.assertEqual(resolved["library"]["port"], 8791)
        self.assertEqual(resolved["library"]["scan_interval"], 2)
        self.assertEqual(resolved["library"]["root"], str(self.root / "catalog"))


# ======================================================================
# 消融配置、实验开关默认关闭与历史归档反序列化兼容性 (迁自 test_operation_ablation_config.py 等)
# ======================================================================

ABLATION_FLAGS = ("task_scheduling", "preview_intent", "recovery_evidence")
DEFAULT_FALSE_EXPERIMENTAL_FLAGS = (
    "robotic_view",
    "stereo_guide",
    "pose_skill",
    "held_perception",
)


class OperationAblationConfigTests(unittest.TestCase):
    @property
    def base_config(self):
        return _cached_ablation_base_config()

    @property
    def default_config(self):
        return _cached_default_config()

    def setUp(self):
        temporary = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(temporary)
        self.config_path = self.root / "config.json"
        self.config = self.base_config

    def case(self, **features):
        return dict(cwd=str(REPO_ROOT),
                    run_dir=str(self.root / "not-started"), task="build_tower",
                    seed=1, layout_id=0, title="Synthetic ablation config",
                    tag=["config-test"], **features)

    def resolve_cli(self, args):
        output = io.StringIO()
        with patch("sys.argv", ["ni_rua", *args, "--print-config"]), \
                patch("ni_rua.apps.gateway.cli.serve", new_callable=AsyncMock) as serve, \
                contextlib.redirect_stdout(output):
            main()
        serve.assert_not_called()
        return Config.from_dict(json.loads(output.getvalue()))

    def test_fresh_run_defaults_remove_candidate_and_purpose_but_keep_motion_evidence(self):
        from types import SimpleNamespace
        from ni_rua.cortex.context.task_notes import default_notes_for
        from ni_rua.cortex.context.task_planning import PLANNING_INSTRUCTIONS

        expected = dict(task_scheduling=True, preview_intent=False, recovery_evidence=True)
        effective = self.resolve_cli([])
        self.assertEqual({name: getattr(effective.features, name) for name in ABLATION_FLAGS}, expected)
        self.assertFalse(effective.features.operation_candidates)
        self.assertTrue(effective.features.trajectory_preview)
        self.assertEqual(effective.features.action_trajectory_delivery, "automatic")
        self.assertTrue(effective.features.bounded_continuation)
        # Reply delivery: the static defaults, the shipped configuration and the CLI agree (AGENTS.md default gate).
        from support.reply_defaults import DEFAULT_REPLY_FEATURES
        shipped = json.loads((CONFIGS_ROOT / "ni-rua.json").read_text())["features"]
        example = json.loads((CONFIGS_ROOT / "ni-rua.example.json").read_text())["features"]
        for name, value in DEFAULT_REPLY_FEATURES.items():
            self.assertEqual((getattr(Config().features, name), getattr(effective.features, name), shipped[name],
                              example[name]), (value,) * 4, name)
        # (tool_catalog_profile and robot_urdf_delivery are part of DEFAULT_REPLY_FEATURES: layered_v3 / file)
        self.assertIs(self.resolve_cli(["--no-budgeted-action-reply"]).features.budgeted_action_reply, False)
        self.assertTrue(effective.features.arm_queue)
        workbench = SimpleNamespace(config=effective, pi05_enabled=False,
            enable_wait=True, visual_review_model=None, motion_review=False,
            observation_profile=ACTOR_PROFILE)
        tools = {item["name"]: item for item in Workbench.actor_tools(workbench)}
        action = tools["robot_action"]["inputSchema"]["properties"]
        self.assertIn("stages", action)
        self.assertEqual(
            set(action["mode"]["enum"]),
            {"preview", "preflight", "execute", "inspect", "direct", "execute_if_clean"},
        )
        for retired in ("robot_preview_cost", "robot_move_dual_grasp", "robot_execute_queue", "robot_inspect_trajectory"):
            self.assertNotIn(retired, tools)
        plan = tools["task_plan"]["inputSchema"]["properties"]["plan"]
        self.assertEqual(plan["type"], "string")
        self.assertNotIn("properties", plan)
        for text in (default_notes_for(), PLANNING_INSTRUCTIONS):
            self.assertNotIn("candidate_operations", text)

    def test_each_leave_one_out_flag_roundtrips_through_queue_and_actual_cli(self):
        from scripts.benchmark_queue import command, queue_config

        self.config_path.write_text(json.dumps(self.config.to_dict()))
        full = {name: True for name in ABLATION_FLAGS}
        # The reply-delivery settings are all CLI flags and queue case options; a case can state either value.
        from support.reply_defaults import DEFAULT_REPLY_FEATURES, LEGACY_REPLY_FEATURES
        switches = [name for name, value in DEFAULT_REPLY_FEATURES.items() if value is True]
        for name in switches:
            flag = "--" + name.replace("_", "-")
            self.assertIs(getattr(self.resolve_cli(["--no-" + flag[2:]]).features, name), False)
            self.assertIs(getattr(self.resolve_cli(["--no-" + flag[2:], flag]).features, name), True)
        choices = dict(action_reply_profile="full", actor_payload_profile="lean_v1", tool_catalog_profile="guide_only_v1",
                       robot_urdf_delivery="inline")
        for name, value in choices.items():
            self.assertEqual(getattr(self.resolve_cli(["--" + name.replace("_", "-"), value]).features, name), value)
        for selected in (dict(LEGACY_REPLY_FEATURES, tool_catalog_profile="guide_only_v1"),
                         dict(DEFAULT_REPLY_FEATURES, tool_catalog_profile="full_v1")):
            case = command(self.case(**selected), 2,
                           config=queue_config(dict(ni_rua_config=self.config.to_dict())), config_path=self.config_path)
            features = self.resolve_cli(case[case.index("ni_rua.apps.gateway") + 1:]).features
            self.assertEqual({name: getattr(features, name) for name in selected}, selected)
        for disabled in (None, *ABLATION_FLAGS):
            with self.subTest(disabled=disabled):
                selected = {name: name != disabled for name in ABLATION_FLAGS}
                manifest = dict(ni_rua_config=self.config.to_dict())
                args = command(self.case(**selected, action_trajectory_delivery="automatic"),
                               2, config=queue_config(manifest), config_path=self.config_path)
                gateway_args = args[args.index("ni_rua.apps.gateway") + 1:]
                effective = self.resolve_cli(gateway_args)
                self.assertEqual({name: getattr(effective.features, name) for name in ABLATION_FLAGS}, selected)
                self.assertEqual(effective.features.action_trajectory_delivery, "automatic")
                self.assertIs(effective.features.budgeted_action_reply, True)
                self.assertEqual(effective.run.gpu, 2)
                self.assertTrue(effective.run.benchmark)
                self.assertIsNone(effective.run.max_actions)
                self.assertIsNone(effective.run.wall_budget_s)
                self.assertEqual({name for name in ABLATION_FLAGS if selected[name] != full[name]},
                                 {disabled} if disabled else set())

    def test_explicit_cli_true_and_delivery_override_false_config_values(self):
        saved = self.config.with_overrides({"features": {
            **{name: False for name in ABLATION_FLAGS}, "action_trajectory_delivery": "diagnostic"}})
        self.config_path.write_text(json.dumps(saved.to_dict()))
        effective = self.resolve_cli(["--config", str(self.config_path),
            *["--" + name.replace("_", "-") for name in ABLATION_FLAGS],
            "--action-trajectory-delivery", "automatic"])
        self.assertTrue(all(getattr(effective.features, name) for name in ABLATION_FLAGS))
        self.assertEqual(effective.features.action_trajectory_delivery, "automatic")
        untouched = self.resolve_cli(["--config", str(self.config_path)])
        self.assertTrue(all(not getattr(untouched.features, name) for name in ABLATION_FLAGS))
        self.assertEqual(untouched.features.action_trajectory_delivery, "diagnostic")

    def test_new_settings_reject_wrong_types_before_any_queue_dispatch(self):
        from scripts.benchmark_queue import command

        for name in ABLATION_FLAGS:
            for invalid in (0, 1, None, "false"):
                with self.subTest(name=name, invalid=invalid):
                    with self.assertRaises(ConfigError):
                        Config.from_dict({"features": {name: invalid}})
                    with self.assertRaises(ValueError):
                        command(self.case(**{name: invalid}), 0, config=self.config)
        for invalid in (None, True, "all", ""):
            with self.subTest(delivery=invalid):
                with self.assertRaises(ConfigError):
                    Config.from_dict({"features": {"action_trajectory_delivery": invalid}})
                with self.assertRaises(ValueError):
                    command(self.case(action_trajectory_delivery=invalid), 0, config=self.config)

    def test_actor_schema_switches_are_independent_and_do_not_mutate_global_tools(self):
        from copy import deepcopy
        from types import SimpleNamespace
        from ni_rua.runtime.tools.definitions import TOOLS

        original = deepcopy(TOOLS)
        for disabled in (None, *ABLATION_FLAGS):
            with self.subTest(disabled=disabled):
                flags = {name: name != disabled for name in ABLATION_FLAGS}
                config = self.config.with_overrides({"features": flags})
                workbench = SimpleNamespace(config=config, pi05_enabled=False,
                    enable_wait=True, visual_review_model=None, motion_review=False,
                    observation_profile=ACTOR_PROFILE)
                tools = {item["name"]: item for item in Workbench.actor_tools(workbench)}
                action = tools["robot_action"]["inputSchema"]["properties"]
                self.assertIn("stages", action)
                self.assertIn("inspect", action["mode"]["enum"])
                plan = tools["task_plan"]["inputSchema"]["properties"]["plan"]
                self.assertEqual(plan["type"], "string")
                for retired in ("robot_preview_cost", "robot_move_dual_grasp", "robot_execute_queue", "robot_inspect_trajectory"):
                    self.assertNotIn(retired, tools)
        self.assertEqual(TOOLS, original)

    def test_default_false_experimental_flags_validate_strict_booleans_and_roundtrip(self):
        defaults = self.default_config
        for flag in DEFAULT_FALSE_EXPERIMENTAL_FLAGS:
            self.assertFalse(getattr(defaults.features, flag), flag)
            for invalid in (0, 1, None, "false", "true"):
                with self.subTest(flag=flag, invalid=invalid), self.assertRaises(ConfigError):
                    Config.from_dict({"features": {flag: invalid}})
        enabled = Config.from_dict({
            "features": {
                "external_perception": True,
                **{flag: True for flag in DEFAULT_FALSE_EXPERIMENTAL_FLAGS},
            }
        })
        restored = Config.from_dict(enabled.to_dict())
        for flag in DEFAULT_FALSE_EXPERIMENTAL_FLAGS:
            self.assertTrue(getattr(restored.features, flag), flag)


class OperationAblationArchivedConfigTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls._older_saved = Config.from_dict({"features": {"motion_review": False}}).to_dict()
        for name in (*ABLATION_FLAGS, "action_trajectory_delivery"):
            cls._older_saved["features"].pop(name)
        cls._mixed_selected = dict(task_scheduling=True,
                                   preview_intent=True, recovery_evidence=False)
        cls._mixed_saved = Config.from_dict({"features": {"motion_review": False,
            **cls._mixed_selected, "action_trajectory_delivery": "diagnostic"}}).to_dict()
        cls._incoming_cache = {}
        cls._exp_all_true = {flag: True for flag in DEFAULT_FALSE_EXPERIMENTAL_FLAGS}
        cls._exp_all_false = {flag: False for flag in DEFAULT_FALSE_EXPERIMENTAL_FLAGS}
        cls._exp_mixed_rv_only = {**cls._exp_all_false, "robotic_view": True, "stereo_guide": False}
        cls._exp_mixed_sg_only = {**cls._exp_all_false, "robotic_view": False, "stereo_guide": True}

        def make_saved(flag_values, *, omit_experimental=False):
            cfg = Config.from_dict({
                "features": {
                    "motion_review": False,
                    "sandbox_enabled": False,
                    "external_perception": True,
                    **flag_values,
                }
            }).to_dict()
            if omit_experimental:
                for flag in DEFAULT_FALSE_EXPERIMENTAL_FLAGS:
                    cfg["features"].pop(flag, None)
            return cfg

        cls._exp_scenarios = (
            ("all_true_vs_incoming_false", make_saved(cls._exp_all_true), cls._exp_all_false, cls._exp_all_true),
            ("all_false_vs_incoming_true", make_saved(cls._exp_all_false), cls._exp_all_true, cls._exp_all_false),
            ("missing_keys_vs_incoming_true", make_saved(cls._exp_all_false, omit_experimental=True), cls._exp_all_true, cls._exp_all_false),
            ("no_effective_config_vs_incoming_true", None, cls._exp_all_true, cls._exp_all_false),
            ("rv_true_sg_false", make_saved(cls._exp_mixed_rv_only), cls._exp_mixed_sg_only, cls._exp_mixed_rv_only),
            ("rv_false_sg_true", make_saved(cls._exp_mixed_sg_only), cls._exp_mixed_rv_only, cls._exp_mixed_sg_only),
        )

    @classmethod
    def _incoming_for_ablation(cls, expected, delivery):
        key = (tuple(sorted(expected.items())), delivery)
        cached = cls._incoming_cache.get(key)
        if cached is None:
            cached = Config.from_dict({"features": {"motion_review": False,
                **{name: not expected[name] for name in ABLATION_FLAGS},
                "action_trajectory_delivery": "diagnostic" if delivery == "automatic" else "automatic"}})
            cls._incoming_cache[key] = cached
        return cached

    @classmethod
    def _incoming_for_experimental(cls, incoming_flags):
        key = tuple(sorted(incoming_flags.items()))
        cached = cls._incoming_cache.get(key)
        if cached is None:
            cached = Config.from_dict({
                "features": {
                    "motion_review": False,
                    "sandbox_enabled": False,
                    "external_perception": True,
                    **incoming_flags,
                }
            })
            cls._incoming_cache[key] = cached
        return cached

    async def check_archive(self, saved, expected, *, mode, delivery="automatic"):
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = dict(schema_version="0.1", run_id="run", episode_id="ep",
                env_instance_id="env", env_generation=1,
                actor_information_policy=ACTOR_POLICY, observation_profile=ACTOR_PROFILE,
                task_notes="Exact archived notes", task_notes_source="archived-fixture")
            if saved is not None:
                manifest["effective_config"] = saved
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            before = path.read_bytes()
            incoming = self._incoming_for_ablation(expected, delivery)
            workbench = Workbench(root, config=incoming, robot=SimpleNamespace(close=lambda: None), **{mode: True})
            try:
                self.assertEqual({name: getattr(workbench.config.features, name) for name in ABLATION_FLAGS}, expected)
                self.assertEqual(workbench.config.features.action_trajectory_delivery, delivery)
                self.assertEqual(workbench.harness.config.features, workbench.config.features)
                self.assertEqual(workbench.archived_task_notes["text"], "Exact archived notes")
                self.assertEqual(path.read_bytes(), before)
            finally:
                await workbench.close()

    async def test_older_unified_archive_does_not_silently_enable_new_features(self):
        for mode in ("resume", "replay"):
            with self.subTest(mode=mode):
                await self.check_archive(self._older_saved, {name: False for name in ABLATION_FLAGS}, mode=mode)

    async def test_legacy_archive_without_effective_config_keeps_new_flags_off(self):
        for mode in ("resume", "replay"):
            with self.subTest(mode=mode):
                await self.check_archive(None, {name: False for name in ABLATION_FLAGS}, mode=mode)

    async def test_current_archive_preserves_its_mixed_ablation_and_delivery_mode(self):
        for mode in ("resume", "replay"):
            with self.subTest(mode=mode):
                await self.check_archive(self._mixed_saved, self._mixed_selected, mode=mode, delivery="diagnostic")

    async def test_archived_experimental_flags_roundtrip_under_resume_and_replay(self):
        from types import SimpleNamespace

        for mode in ("resume", "replay"):
            for label, saved, incoming_flags, expected_flags in self._exp_scenarios:
                with self.subTest(mode=mode, scenario=label), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    manifest = dict(
                        schema_version="0.1",
                        run_id="run",
                        episode_id="ep",
                        env_instance_id="env",
                        env_generation=1,
                        actor_information_policy=ACTOR_POLICY,
                        observation_profile=ACTOR_PROFILE,
                    )
                    if saved is not None:
                        manifest["effective_config"] = saved
                    path = root / "manifest.json"
                    path.write_text(json.dumps(manifest))
                    before = path.read_bytes()
                    incoming = self._incoming_for_experimental(incoming_flags)
                    workbench = Workbench(
                        root,
                        config=incoming,
                        robot=SimpleNamespace(close=lambda: None),
                        **{mode: True},
                    )
                    try:
                        self.assertEqual(workbench.root, root.resolve())
                        self.assertEqual(workbench.store.root, root.resolve())
                        for flag, expected_val in expected_flags.items():
                            self.assertIs(getattr(workbench.config.features, flag), expected_val, (mode, label, flag))
                        for role in ("actor", "program"):
                            names = {
                                tool["name"]
                                for tool in workbench.harness.tool_registry.catalog(workbench.harness, role=role)
                            }
                            self.assertIs("robotic_view" in names, expected_flags["robotic_view"], (mode, label, role))
                            self.assertIs("stereo_guide" in names, expected_flags["stereo_guide"], (mode, label, role))
                            self.assertIs("pose_plan" in names, expected_flags["pose_skill"], (mode, label, role))
                        self.assertEqual(path.read_bytes(), before)
                    finally:
                        await workbench.close()


class ConfigV2AndDeclarativeCliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def test_v2_embodiment_thresholds_mapping_sparse_to_dict_and_validation(self):
        from dataclasses import replace
        from ni_rua.core.config import (
            EmbodimentConfig,
            MOTION_THRESHOLD_FIELDS,
            RETIRED_FEATURE_FIELDS,
            motion_thresholds,
        )

        repo_root = REPO_ROOT
        defaults_doc = json.loads((repo_root / "ni_rua/robot/embodiments/arx_x5/defaults.json").read_text())
        self.assertEqual(tuple(defaults_doc["thresholds"]), MOTION_THRESHOLD_FIELDS)
        self.assertEqual(defaults_doc["preset_upward_dz_m"], {"compliant": 0.04, "seating": 0.04})

        cfg = Config()
        self.assertEqual(cfg.version, 2)
        self.assertEqual(cfg.embodiment.to_dict(), {"thresholds": {}})
        self.assertEqual(cfg.to_dict()["embodiment"], {"thresholds": {}})
        self.assertEqual(len(cfg.embodiment.thresholds), 44)
        self.assertEqual(dict(cfg.embodiment.thresholds), defaults_doc["thresholds"])
        self.assertEqual(cfg.embodiment.thresholds.tight_position_tolerance_m, 0.003)
        self.assertEqual(cfg.features.tight_position_tolerance_m, 0.003)
        with self.assertRaises(TypeError):
            cfg.embodiment.thresholds["tight_position_tolerance_m"] = 0.01
        with self.assertRaises(TypeError):
            cfg.embodiment.thresholds.tight_position_tolerance_m = 0.01
        with self.assertRaises(AttributeError):
            _ = cfg.embodiment.thresholds.nonexistent_threshold

        sparse = Config.from_dict({"embodiment": {"thresholds": {"tight_position_tolerance_m": 0.004, "batch_plan_cache_size": 32}}})
        self.assertEqual(sparse.to_dict()["embodiment"], {"thresholds": {"tight_position_tolerance_m": 0.004, "batch_plan_cache_size": 32}})
        self.assertEqual(sparse.embodiment.thresholds.tight_position_tolerance_m, 0.004)
        self.assertEqual(sparse.embodiment.thresholds["batch_plan_cache_size"], 32)
        self.assertEqual(sparse.embodiment.thresholds.default_position_tolerance_m, 0.01)
        self.assertEqual(sparse.features.tight_position_tolerance_m, 0.004)
        self.assertEqual(motion_thresholds(sparse)["tight_position_tolerance_m"], 0.004)
        self.assertEqual(motion_thresholds(sparse.embodiment)["tight_position_tolerance_m"], 0.004)
        self.assertEqual(motion_thresholds(sparse.features)["tight_position_tolerance_m"], 0.004)
        self.assertEqual(Config.from_dict(sparse.to_dict()), sparse)

        # Replacing embodiment on a frozen Config must never mutate the source Config's features view.
        replaced = replace(sparse, embodiment=EmbodimentConfig(thresholds={"tight_position_tolerance_m": 0.008}))
        self.assertEqual(sparse.features.tight_position_tolerance_m, 0.004)
        self.assertEqual(replaced.features.tight_position_tolerance_m, 0.008)

        # Retired bloat flags return read-only inactive defaults and stay absent from serialized snapshots.
        expected_retired_defaults = {
            "self_briefing": False,
            "operation_candidates": False,
            "lean_projection": False,
            "lean_compact": False,
            "lean_projection_k": 1,
            "lean_projection_block": 8,
            "lean_request_dump": 0,
        }
        for name, default_val in expected_retired_defaults.items():
            self.assertIn(name, RETIRED_FEATURE_FIELDS)
            self.assertEqual(getattr(cfg.features, name), default_val)
            self.assertNotIn(name, cfg.to_dict()["features"])

        invalid_cases = (
            ({"embodiment": {"unknown_subkey": {}}}, "unknown_config_keys:config.embodiment"),
            ({"embodiment": {"thresholds": {"unknown_thr": 0.1}}}, "unknown_config_keys:EmbodimentConfig.thresholds"),
            ({"embodiment": {"thresholds": "not_a_dict"}}, "invalid_config_type:EmbodimentConfig.thresholds"),
            ({"embodiment": {"thresholds": {"tight_position_tolerance_m": True}}}, "invalid_config_type:EmbodimentConfig.thresholds.tight_position_tolerance_m"),
            ({"embodiment": {"thresholds": {"tight_position_tolerance_m": float("nan")}}}, "invalid_config_type:EmbodimentConfig.thresholds.tight_position_tolerance_m"),
            ({"embodiment": {"thresholds": {"batch_plan_cache_size": 16.0}}}, "invalid_config_type:EmbodimentConfig.thresholds.batch_plan_cache_size"),
            ({"embodiment": {"thresholds": {"tight_position_tolerance_m": 0.0001}}}, "config_below_minimum:EmbodimentConfig.thresholds.tight_position_tolerance_m"),
            ({"embodiment": {"thresholds": {"tight_position_tolerance_m": 0.5}}}, "config_above_maximum:EmbodimentConfig.thresholds.tight_position_tolerance_m"),
            ({"embodiment": {"thresholds": {"grip_wide_min_aperture_m": 0.09, "max_gripper_aperture_m": 0.088}}}, "config_above_maximum:EmbodimentConfig.thresholds.grip_wide_min_aperture_m"),
        )
        for bad_doc, pattern in invalid_cases:
            with self.subTest(bad_doc=bad_doc), self.assertRaisesRegex(ConfigError, pattern):
                Config.from_dict(bad_doc)

    def test_migrate_v1_to_v2_archive_and_layer_parity(self):
        from copy import deepcopy
        from ni_rua.core.config import migrate_v1_to_v2, motion_thresholds

        v1_doc = {
            "version": 1,
            "features": {
                "enable_wait": False,
                "tight_position_tolerance_m": 0.004,
                "max_gripper_aperture_m": 0.092,
                "self_briefing": True,
                "operation_candidates": True,
                "lean_projection": True,
                "lean_projection_k": 3,
                "lean_projection_block": 16,
                "lean_compact": True,
                "lean_request_dump": 10,
                "soft_position_tolerance_m": 0.01,
            },
            "embodiment": {"thresholds": {"tight_position_tolerance_m": 0.007}},
        }
        original = deepcopy(v1_doc)
        migrated = migrate_v1_to_v2(v1_doc)
        self.assertEqual(v1_doc, original)  # input never mutated
        self.assertEqual(migrated["version"], 2)
        self.assertEqual(migrated["features"], {"enable_wait": False})
        # Explicit embodiment.thresholds takes precedence over legacy features threshold.
        self.assertEqual(
            migrated["embodiment"]["thresholds"],
            {"tight_position_tolerance_m": 0.007, "max_gripper_aperture_m": 0.092},
        )

        for bad_ver in (0, 3, -1, "1", True, None):
            with self.subTest(version=bad_ver), self.assertRaises(ConfigError):
                migrate_v1_to_v2({"version": bad_ver})

        v1_file = self.root / "archived_v1.json"
        v1_file.write_text(json.dumps({
            "version": 1,
            "features": {"default_position_tolerance_m": 0.008, "self_briefing": True, "lean_compact": True},
        }))
        with patch("ni_rua.core.config.LOCAL_CONFIG_PATH", self.root / "absent.local.json"):
            loaded = Config.load(v1_file, environ={}, overrides={"features": {"relaxed_position_tolerance_m": 0.018}})
        self.assertEqual(loaded.version, 2)
        self.assertEqual(loaded.embodiment.thresholds.default_position_tolerance_m, 0.008)
        self.assertEqual(loaded.embodiment.thresholds.relaxed_position_tolerance_m, 0.018)
        overridden = loaded.with_overrides({"features": {"coarse_position_tolerance_m": 0.03, "operation_candidates": True}})
        self.assertEqual(overridden.embodiment.thresholds.coarse_position_tolerance_m, 0.03)
        self.assertEqual(motion_thresholds(overridden)["default_position_tolerance_m"], 0.008)

    def test_single_source_cli_and_queue_metadata_parity(self):
        from ni_rua.apps.evaluation import queue
        from ni_rua.apps.gateway import cli as gateway_cli
        from ni_rua.core.config import CLI_FIELDS, CLI_OPTIONS, QUEUE_CASE_OPTIONS, QUEUE_CLI_OPTIONS, namespace_values

        parser = gateway_cli.build_parser()
        parser_flags = set(parser._option_string_actions)
        for spec in CLI_OPTIONS:
            self.assertIn(spec.flag, parser_flags, spec.dest)
            self.assertEqual(CLI_FIELDS[spec.dest], (spec.section, spec.field_name))
            if spec.kind == "bool":
                self.assertIn("--no-" + spec.flag[2:], parser_flags, spec.dest)
        self.assertIn("--config", parser_flags)
        self.assertIn("--no-local-config", parser_flags)
        self.assertIn("--print-config", parser_flags)
        self.assertIn("--clear-tags", parser_flags)
        self.assertIn("--no-visual-review-model", parser_flags)
        for retired_flag in ("--self-briefing", "--operation-candidates", "--lean-projection", "--lean-compact"):
            self.assertNotIn(retired_flag, parser_flags)

        ns_vals = namespace_values(Config())
        for spec in CLI_OPTIONS:
            self.assertIn(spec.dest, ns_vals)
        self.assertEqual({spec.dest for spec in QUEUE_CLI_OPTIONS},
                         {"wait_timeout_s", "wake_enabled", "wake_thread", "wake_codex", "wake_remote"})
        self.assertNotIn("operation_candidates", {spec.dest for spec in QUEUE_CASE_OPTIONS})
        self.assertNotIn("self_briefing", {spec.dest for spec in QUEUE_CASE_OPTIONS})

        # Queue runtime projection downgrades version 2 -> 1 and omits empty embodiment.thresholds for v1 checkouts,
        # while failing closed if non-empty threshold overrides target a v1 checkout lacking EmbodimentConfig.
        v2_doc = Config().to_dict()
        v1_schema = {
            name: set(fields_set)
            for name, fields_set in queue.runtime_config_schema(REPO_ROOT).items()
            if name != "EmbodimentConfig"
        }
        v1_schema["Config"].discard("embodiment")
        omissions = []
        projected = queue.project_config_for_runtimes(v2_doc, [v1_schema], omissions)
        self.assertEqual(projected["version"], 1)
        self.assertNotIn("embodiment", projected)
        self.assertEqual(omissions, [{"key": "embodiment", "value": {"thresholds": {}}, "reason": "target_schema_omits_default"}])

        overridden_v2 = Config.from_dict({"embodiment": {"thresholds": {"tight_position_tolerance_m": 0.004}}}).to_dict()
        with self.assertRaisesRegex(ValueError, "unified_config_incompatible:embodiment"):
            queue.project_config_for_runtimes(overridden_v2, [v1_schema])

        # Historical v1 runtime schema that declares MOTION_THRESHOLD_FIELDS on FeaturesConfig (pre-ad01185 checkouts).
        from ni_rua.core.config import MOTION_THRESHOLD_FIELDS, RETIRED_FEATURE_FIELDS
        from ni_rua.core.config.thresholds import _DEFAULT_MOTION_THRESHOLDS

        self.assertEqual(set(queue._RETIRED_FEATURE_DEFAULTS), set(RETIRED_FEATURE_FIELDS))
        historical_v1_schema = {
            name: set(fields_set)
            for name, fields_set in v1_schema.items()
        }
        historical_v1_schema["FeaturesConfig"] = set(v1_schema["FeaturesConfig"]) | set(MOTION_THRESHOLD_FIELDS)
        v2_runtime_schema = queue.runtime_config_schema(REPO_ROOT)

        # (a) Raw v1 archived config with default retired flags & default motion thresholds projects onto v2 schema.
        raw_v1_doc = {
            "version": 1,
            "features": {
                "enable_wait": True,
                "self_briefing": False,
                "operation_candidates": False,
                "lean_projection": False,
                "lean_projection_k": 1,
                "lean_projection_block": 8,
                "lean_compact": False,
                "lean_request_dump": 0,
                "soft_position_tolerance_m": 0.01,
                "tight_position_tolerance_m": 0.003,
            },
        }
        raw_omissions = []
        projected_raw_v1 = queue.project_config_for_runtimes(raw_v1_doc, [v2_runtime_schema], raw_omissions)
        self.assertEqual(projected_raw_v1["features"], {"enable_wait": True})
        self.assertTrue(any(item["key"] == "features.self_briefing" for item in raw_omissions))
        self.assertTrue(any(item["key"] == "features.tight_position_tolerance_m" for item in raw_omissions))
        for bad_raw in (
            {"version": 1, "features": {"self_briefing": True}},
            {"version": 1, "features": {"self_briefing": 0}},
            {"version": 1, "features": {"lean_request_dump": False}},
            {"version": 1, "features": {"tight_position_tolerance_m": 0.004}},
        ):
            with self.subTest(bad_raw=bad_raw), self.assertRaisesRegex(ValueError, "unified_config_incompatible:features\\."):
                queue.project_config_for_runtimes(bad_raw, [v2_runtime_schema])

        # (b) Config.from_dict(historical_v1_ni_rua_json).to_dict() (44 default thresholds in embodiment.thresholds)
        # projects cleanly onto [historical_v1_schema] and [v1_schema].
        historical_v1_ni_rua_json = {
            "version": 1,
            "features": {
                **{k: v for k, v in v2_doc["features"].items()},
                **dict(_DEFAULT_MOTION_THRESHOLDS),
                "self_briefing": False,
                "operation_candidates": False,
            },
        }
        migrated_snapshot = Config.from_dict(historical_v1_ni_rua_json).to_dict()
        self.assertEqual(len(migrated_snapshot["embodiment"]["thresholds"]), len(MOTION_THRESHOLD_FIELDS))
        for target_schema in (historical_v1_schema, v1_schema):
            mig_omissions = []
            projected_mig = queue.project_config_for_runtimes(migrated_snapshot, [target_schema], mig_omissions)
            self.assertEqual(projected_mig["version"], 1)
            self.assertNotIn("embodiment", projected_mig)
            self.assertFalse(set(MOTION_THRESHOLD_FIELDS) & set(projected_mig["features"]))
            self.assertEqual(mig_omissions[0]["key"], "embodiment")
            self.assertEqual(mig_omissions[0]["reason"], "target_schema_omits_default")

        # (c) Non-default embodiment.thresholds translate back into features.* on historical v1 schemas that declare them.
        v1_custom_thr = dict(historical_v1_ni_rua_json)
        v1_custom_thr["features"] = {**historical_v1_ni_rua_json["features"], "tight_position_tolerance_m": 0.004}
        migrated_custom = Config.from_dict(v1_custom_thr).to_dict()
        trans_omissions = []
        projected_custom = queue.project_config_for_runtimes(migrated_custom, [historical_v1_schema], trans_omissions)
        self.assertEqual(projected_custom["version"], 1)
        self.assertNotIn("embodiment", projected_custom)
        self.assertEqual(projected_custom["features"]["tight_position_tolerance_m"], 0.004)
        self.assertNotIn("default_position_tolerance_m", projected_custom["features"])
        self.assertEqual(trans_omissions[0]["reason"], "translated_to_v1_features")
        with self.assertRaisesRegex(ValueError, "unified_config_incompatible:embodiment\\.thresholds\\.tight_position_tolerance_m"):
            queue.project_config_for_runtimes(migrated_custom, [v1_schema])

        # evidence_path_warnings checks visual_review_model across backends.
        codex_rev = Config.from_dict({"features": {"visual_review_model": "gpt-6.1-sol"}})
        claude_rev = codex_rev.with_overrides({"agent": {"backend": "claude_code"}})
        warnings = queue.evidence_path_warnings([codex_rev, claude_rev])
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]["kind"], "comparison_evidence_path_mismatch")

    def test_context_delivery_defaults_align_with_codex_delta(self):
        # 四后端 context_delivery 默认均对齐 Codex 为 'delta'，显式配置 'full' 可切回，且 subagents=True 不再强覆写
        default_cfg = Config()
        self.assertEqual(default_cfg.agent.context_delivery, "delta")
        self.assertEqual(default_cfg.agent.claude_code.context_delivery, "delta")
        self.assertEqual(default_cfg.agent.opencode.context_delivery, "delta")
        self.assertEqual(default_cfg.agent.external.context_delivery, "delta")
        self.assertTrue(default_cfg.agent.claude_code.subagents)
        self.assertTrue(default_cfg.agent.opencode.subagents)
        self.assertNotIn("agent", default_cfg.to_dict())

        # 静态默认值与 configs/ni-rua.json 门禁保持 100% 一致
        repo_ni_rua = json.loads((CONFIGS_ROOT / "ni-rua.json").read_text())
        self.assertEqual(Config.from_dict({}), Config.load(environ={}, ignore_local=True))
        self.assertEqual(Config.from_dict({}), Config.from_dict(repo_ni_rua))

        for backend in ("claude_code", "opencode"):
            for subagents in (True, False):
                for configured in ("delta", "full"):
                    with self.subTest(backend=backend, subagents=subagents, configured=configured):
                        cfg = Config.from_dict({
                            "agent": {
                                "backend": backend,
                                backend: {"subagents": subagents, "context_delivery": configured},
                            }
                        })
                        section = getattr(cfg.agent, backend)
                        self.assertEqual(section.effective_context_delivery, configured)
                        self.assertEqual(cfg.agent.context_delivery, configured)
                        self.assertEqual(Config.from_dict(cfg.to_dict()), cfg)

        # ExternalConfig.context_delivery 默认 'delta' 且默认节被 _omit_agent_defaults 省略，显式 'full' 完整往返
        ext_default = Config.from_dict({"agent": {"backend": "external"}})
        self.assertEqual(ext_default.agent.context_delivery, "delta")
        self.assertNotIn("external", ext_default.to_dict()["agent"])
        ext_full = Config.from_dict({"agent": {"backend": "external", "external": {"context_delivery": "full"}}})
        self.assertEqual(ext_full.agent.external.context_delivery, "full")
        self.assertEqual(ext_full.agent.context_delivery, "full")
        self.assertEqual(ext_full.to_dict()["agent"]["external"]["context_delivery"], "full")
        self.assertEqual(Config.from_dict(ext_full.to_dict()), ext_full)
        with self.assertRaisesRegex(ConfigError, "invalid_config_choice:ExternalConfig.context_delivery"):
            Config.from_dict({"agent": {"external": {"context_delivery": "both"}}})
