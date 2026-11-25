import assert from 'node:assert/strict';
import test from 'node:test';
import { blockerText, boardToForm, criterionRow, criterionRule, currentStep, describeReason, formatDiffRow, formToBoard, nextStep, phaseFraction,
  phaseText, planPreview, stalenessText, stepView } from './calibration-model.js';

const definition = {
  statuses: { not_started: '未开始', running: '进行中', passed: '通过', failed: '失败', stale: '已过期' },
  steps: [{ id: 'S1', readiness: [{ id: 'table_clear', label: '桌面区域已清空' }] }, { id: 'S2', readiness: [] }, { id: 'S4', readiness: [] }, { id: 'S5', readiness: [] }],
  changes: [{ id: 'fixed_moved', steps: ['S4', 'S5'] }, { id: 'wrist_bumped', steps: ['S2', 'S4', 'S5'] }],
};
const run = { plan: ['S2', 'S4', 'S5'], steps: { S1: { status: 'passed' }, S2: { status: 'passed' }, S4: { status: 'failed' }, S5: { status: 'stale' } } };

test('criterion rows show measured numbers against the threshold in display units', () => {
  const row = criterionRow({ label: '手眼平移 RMS', value: 0.0031, threshold: 0.005, op: '<=', unit: 'mm', scale: 1000, passed: true });
  assert.deepEqual(row, { label: '手眼平移 RMS', measured: '3.10 mm', limit: '≤ 5.00 mm', passed: true, info: false });
  assert.equal(criterionRow({ label: 'x', value: null, threshold: 8, op: '>=', unit: '', scale: 1, passed: false }).measured, '—');
  assert.equal(criterionRow({ label: 'x', value: true, op: 'true', unit: '', scale: 1, passed: true }).limit, '必须为是');
  assert.equal(criterionRow({ label: 'x', value: 4.1, op: 'info', unit: 'px', scale: 1, passed: true }).limit, '仅供参考');
  assert.equal(criterionRow({ label: 'n', value: 5, threshold: 5, op: '==', unit: '', scale: 1, passed: true }).measured, '5');
  assert.equal(criterionRule({ label: 'x', op: '<=', threshold_value: 1.0, unit: 'px', scale: 1 }), '≤ 1.000 px');
  assert.equal(criterionRule({ label: 'n', op: '==', threshold_value: null, threshold_label: '计划点数', unit: '', scale: 1 }), '= 计划点数');
  assert.equal(criterionRule({ label: 'n', op: '>=', threshold_value: 1, threshold_label: null, unit: '', scale: 1 }), '≥ 1');
});

test('diff rows show metres as millimetres and keep rad/px', () => {
  assert.deepEqual(formatDiffRow({ label: '桌面高度', unit: 'm', old: 0.0012, new: -0.0005 }), { label: '桌面高度', old: '1.20 mm', new: '-0.50 mm' });
  assert.deepEqual(formatDiffRow({ label: 'pan', unit: 'rad', old: 0.01, new: null }), { label: 'pan', old: '0.0100 rad', new: '—' });
});

test('plan preview is the union of the ticked changes in step order', () => {
  assert.deepEqual(planPreview(definition, new Set(['wrist_bumped', 'fixed_moved'])), ['S2', 'S4', 'S5']);
  assert.deepEqual(planPreview(definition, new Set()), []);
});

test('stepper view and navigation follow the run plan', () => {
  assert.deepEqual(stepView(definition, run, 'S4'), { status: 'failed', inPlan: true, label: '失败', kind: 'bad' });
  assert.equal(stepView(definition, run, 'S1').inPlan, false);
  assert.equal(stepView(definition, run, 'S1').label, '通过 · 本轮不需要');
  assert.equal(currentStep(run), 'S4');
  assert.equal(nextStep(run, 'S4'), 'S5');
  assert.equal(nextStep(run, 'S5'), null);
  assert.equal(currentStep(null), null);
});

test('reason codes become Chinese text, unknown codes stay verbatim', () => {
  assert.equal(describeReason('readiness_unconfirmed:table_clear', definition), '请先勾选确认：桌面区域已清空');
  assert.equal(describeReason('calibration_blocked:torque_off,codex_turn_running', definition), `${blockerText(['torque_off'])[0]}；${blockerText(['codex_turn_running'])[0]}`);
  assert.equal(describeReason('input_owner_conflict'), '另一个浏览器窗口正在控制这台机械臂');
  assert.match(describeReason('input_owner_busy_step: a long command (e.g. a calibration step or go_home) is running'), /结束前不能换窗口取得控制；紧急停止、hold 和取消标定步骤在任何窗口都可用/);
  assert.equal(describeReason('接触点数不足'), '接触点数不足');
  assert.equal(describeReason(null), '');
});

test('progress text and fraction', () => {
  assert.equal(phaseText({ phase: 'sample', pose_index: 3, pose_count: 13, kept_count: 2, corners: 24 }), '拍摄并检测标定板 · 位姿 3/13 · 已保留 2 · 角点 24');
  assert.equal(phaseFraction({ pose_index: 3, pose_count: 12 }), 0.25);
  assert.equal(phaseFraction({ point_index: 1, point_count: 4 }), 0.25);
  assert.equal(phaseFraction({ phase: 'solving' }), null);
});

test('board form round trips mm <-> m without float noise', () => {
  const board = { squares_x: 7, squares_y: 5, square_m: 0.03, marker_m: 0.022, thickness_m: 0.0015, dictionary: 'DICT_5X5_100' };
  const form = boardToForm(board);
  assert.deepEqual([form.square_mm, form.marker_mm, form.thickness_mm], [30, 22, 1.5]);
  assert.deepEqual(formToBoard({ ...form, square_mm: '25.5' }), { ...board, square_m: 0.0255 });
});

test('staleness text', () => {
  assert.match(stalenessText({ status: 'shifted', rms_px: 9.5, threshold_px: 4 }), /9\.500 px 超过 4\.000 px/);
  assert.equal(stalenessText(null), '尚未检查');
});
