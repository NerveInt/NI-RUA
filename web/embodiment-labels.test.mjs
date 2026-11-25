import assert from 'node:assert/strict';
import test from 'node:test';
import { environmentLabel, isSo101, runtimeDescription } from './task-labels.js';
import { cameraVideos, singleArmLayout } from './experiment-media.js';
import { newRunCommand } from './new-run-command.js';

globalThis.location = { href: 'http://127.0.0.1/', origin: 'http://127.0.0.1' };

test('new SO-101 manifest is labelled as the real arm', () => {
  const record = { manifest: { data_origin: 'so101_real_hardware', runtime_protocol: 'so101-deployment-v1', clock_policy: 'real_time_not_paused' } };
  assert.equal(environmentLabel(record), 'SO-101 真机');
  assert.match(runtimeDescription(record), /真实时间/);
});

test('old SO-101 archive saying robodojo_runtime is told apart by capabilities', () => {
  for (const capabilities of [{ data_origin: 'so101_real_hardware' }, { backend: 'so101_lerobot' }]) {
    const record = { manifest: { data_origin: 'robodojo_runtime', runtime_protocol: 'robodojo-native-v1', clock_policy: 'paused_between_gateway_operations', capabilities } };
    assert.equal(isSo101(record), true);
    assert.equal(environmentLabel(record), 'SO-101 真机');
    assert.doesNotMatch(runtimeDescription(record), /暂停时钟|仿真/);
  }
  assert.equal(environmentLabel({ runtime: { capabilities: { backend: 'so101_lerobot' } }, manifest: { data_origin: 'robodojo_runtime' } }), 'SO-101 真机');
});

test('RoboDojo and other records are labelled exactly as before', () => {
  const robodojo = { manifest: { data_origin: 'robodojo_runtime', runtime_protocol: 'robodojo-native-v1', clock_policy: 'paused_between_gateway_operations', capabilities: { backend: 'robodojo_curobo' } } };
  assert.equal(environmentLabel(robodojo), 'RoboDojo 仿真环境');
  assert.equal(runtimeDescription(robodojo), '');
  assert.equal(environmentLabel({ environment: 'roboprobe_sim' }), 'RoboProbe 仿真环境');
  assert.equal(environmentLabel({ data_origin: 'synthetic_fixture' }), '合成测试环境');
  assert.equal(environmentLabel({ manifest: { data_origin: 'unknown' } }), 'unknown');
  assert.equal(environmentLabel({}), '环境未注明');
});

const video = (name) => ({ name, url: `/media/${name}`, mime_type: 'video/mp4' });

test('SO-101 fixed and wrist videos are listed; RoboDojo roles are unchanged', () => {
  const arm = cameraVideos([video('fixed.mp4'), video('wrist.mp4')], []);
  assert.deepEqual(Object.keys(arm).sort(), ['fixed', 'wrist']);
  assert.equal(arm.fixed.url, '/media/fixed.mp4');
  assert.equal(singleArmLayout(arm).length, 2);
  const indexed = cameraVideos([{ ...video('a.mp4'), camera_id: 'wrist' }], []);
  assert.deepEqual(Object.keys(indexed), ['wrist']);
  const sim = cameraVideos([video('cam_head.mp4'), video('cam_left_wrist.mp4'), video('cam_right_wrist.mp4')], []);
  assert.deepEqual(Object.keys(sim).sort(), ['head', 'left', 'right']);
  assert.equal(singleArmLayout(sim), null);
  assert.equal(singleArmLayout({}), null);
});

test('new-run command: RoboDojo console text is unchanged, SO-101 gets the real-arm config', () => {
  const base = { model: 'gpt-6.1-sol', observationProfile: 'rgb_proprio_calibrated_v1' };
  const old = ['mamba run -n ni-rua python -m ni_rua.apps.gateway \\', '  --config configs/ni-rua.json \\',
    '  --model gpt-6.1-sol --observation-profile rgb_proprio_calibrated_v1 \\', '  --port 8781 --category so101 --title "<任务名>" \\',
    '  --prompt "调用 robot_capabilities 与 robot_observe，报告看到的场景与机械臂状态；不要移动。"'].join('\n');
  assert.equal(newRunCommand({ ...base, environment: 'robodojo_curobo' }), old);
  assert.equal(newRunCommand({ ...base, environment: undefined }), old);
  const arm = newRunCommand({ ...base, environment: 'so101_real', category: 'bench' });
  assert.match(arm, /--config configs\/so101_gateway\.json/);
  assert.match(arm, /--category bench/);
  assert.doesNotMatch(arm, /configs\/ni-rua\.json/);
});
