import assert from 'node:assert/strict';
import test from 'node:test';
import {
  perceptionObjects,
  matchesPerception,
  formatPerceptionSummary,
  formatObjectCentre,
  formatObjectDimensions,
  formatObjectRotation,
  formatQualityBadge,
  formatDeltaSummary,
  formatRemoteStatusSummary,
} from './perception-model.js';

test('formatPerceptionSummary formats count, scale, and table height', () => {
  assert.equal(formatPerceptionSummary(null), '暂无 3D 感知数据');
  assert.equal(
    formatPerceptionSummary({ objects: [{ id: 'o1' }, { id: 'o2' }], scale: 1.04, table: { table_z: 0.765 } }),
    '已检测 2 个物体 · VGGT 尺度 1.04 · 桌面 z=0.765m'
  );
  assert.equal(
    formatPerceptionSummary({ table: { columns: ['id'], rows: [['o1']] } }),
    '已检测 1 个物体'
  );
});

test('formatObjectCentre formats coordinates to 3 decimals', () => {
  assert.equal(formatObjectCentre(null), '-');
  assert.equal(formatObjectCentre([0.1, 0.25, 0.7891]), '[0.100, 0.250, 0.789]');
});

test('formatObjectDimensions converts metres to millimetres', () => {
  assert.equal(formatObjectDimensions(null), '-');
  assert.equal(formatObjectDimensions({ length_m: 0.08, width_m: 0.05, height_m: 0.12 }), '80 × 50 × 120');
  assert.equal(formatObjectDimensions({ length_m: 0.08, width_m: 0.05, top_z_m: 0.88, bottom_z_m: 0.78 }), '80 × 50 × 100');
});

test('formatObjectRotation formats yaw and tilt angles in degrees', () => {
  assert.equal(formatObjectRotation(null), '-');
  assert.equal(formatObjectRotation({ yaw_deg: 45.2, axis_tilt_deg: 0 }), '45° / 0°');
  assert.equal(formatObjectRotation({ yaw_deg: -30.8, axis_tilt_deg: 12.1 }), '-31° / 12°');
});

test('formatQualityBadge checks quality ok vs flags', () => {
  assert.deepEqual(formatQualityBadge({ quality: 'ok', flags: [] }), { text: '正常', ok: true });
  assert.deepEqual(formatQualityBadge({ quality: 'low', flags: ['floating', 'low_depth_confidence'] }), {
    text: '悬空, 低置信度',
    ok: false,
  });
});

test('actual motion delta schema preserves missing measurements', () => {
  assert.equal(formatDeltaSummary(null), null);
  assert.equal(formatDeltaSummary({ moved: [{ before_id: 'o1', after_id: 'o2', delta_m: [.015, -.004, .002] }], unchanged_count: 0 }),
    '动作前后位移估计 · o1 → o2: Δ=(+15, -4, +2) mm | 未变 0 个');
  assert.match(formatDeltaSummary({ moved: [{ before_id: 'o1' }] }), /未知/);
  assert.equal(formatDeltaSummary({ vacated: [], unmatched_under_arm: [{ before_id: 'o1', covered: .93 }] }),
    '动作前后位移估计 · 原位置被手臂遮挡且未匹配 1 个');
  assert.equal(formatDeltaSummary({ unrelated: {} }), null);
});

test('unknown values are never converted to zero, and columns are named', () => {
  assert.equal(formatObjectRotation({ yaw_deg: 10 }), '10° / -');
  assert.equal(formatObjectDimensions({ top_z_m: .1, bottom_z_m: null }), '- × - × -');
  assert.equal(formatObjectCentre([null, 0, .2]), '-');
  assert.deepEqual(perceptionObjects({ table: { columns: ['label', 'id'], rows: [['cup', 'o1']] } }), [{ id: 'o1', label: 'cup' }]);
  assert.match(formatPerceptionSummary({ diagnostics: { object_table: { status: 'unavailable' } } }), /无可用/);
});

test('overlays bind both camera and observation', () => {
  const p = { observation_id: 'O1', camera_id: 'fixed' };
  assert.equal(matchesPerception(p, 'O1', 'fixed'), true);
  assert.equal(matchesPerception(p, 'O1', 'wrist'), false);
  assert.equal(matchesPerception(p, 'O2', 'fixed'), false);
});

test('formatRemoteStatusSummary handles online, starting, and offline states', () => {
  assert.deepEqual(formatRemoteStatusSummary({ enabled: false }), {
    label: '感知: 未启用',
    state: 'offline',
    text: '未配置或未启用',
  });
  assert.deepEqual(
    formatRemoteStatusSummary({
      enabled: true,
      services: {
        vggt: { available: true, health: { device: 'NVIDIA RTX 4090' } },
      },
    }),
    { label: 'VGGT 在线', state: 'online', text: 'VGGT 健康检查通过；SAM 状态以返回的后端信息为准' }
  );
  assert.deepEqual(
    formatRemoteStatusSummary({
      enabled: true,
      state: 'starting',
      services: { vggt: { available: false } },
    }),
    { label: 'VGGT+SAM 启动中', state: 'warning', text: '正在拉起服务' }
  );
  assert.deepEqual(
    formatRemoteStatusSummary({
      enabled: true,
      state: 'degraded',
      services: { vggt: { available: false, reason: 'ConnectionRefused' } },
    }),
    { label: 'VGGT+SAM 离线', state: 'offline', text: 'ConnectionRefused' }
  );
});

test('archived service readiness is historical only', () => {
  const summary = formatRemoteStatusSummary({ archive: true, enabled: true, services: { vggt: { available: true } } });
  assert.match(summary.label, /历史/);
  assert.equal(summary.state, 'warning');
});
