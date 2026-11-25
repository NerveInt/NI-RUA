// Pure formatting and state helpers of the /calibration wizard (no DOM, no network).
// Typed data from /v1/calibration/* in, display strings out; display text is never parsed back.
import { isEn, localeTag, pick } from './shared/i18n.js';
import {
  CALIB_READINESS_EN,
  CALIB_STATUSES_EN,
  THRESHOLD_LABELS_EN,
  criterionLabelEn,
  diffLabelEn,
  translateStepFailureReasonEn,
} from './i18n/calibration.js';

export const STATUS_KIND = { not_started: '', running: 'warn', passed: 'ok', failed: 'bad', stale: 'warn' };

// mm per m: unit conversion for the board form only.
const MM_PER_M = 1000;

export const BLOCKERS = {
  codex_turn_running: 'Codex 正在运行：请等它结束，或先到操作台中断轮次',
  action_in_progress: '机械臂正在执行动作，请等它结束',
  program_active: '有程序正在运行，请等它结束',
  calibration_step_running: '上一个标定步骤还在运行',
  unknown_requires_acknowledgement_via_resume: '存在 unknown 状态：请到操作台检查机械臂并「恢复准入」后再来',
  torque_off: '机械臂已松力矩：请先到操作台「上力矩」',
};
const BLOCKERS_EN = {
  codex_turn_running: 'Agent turn is running: wait for it to finish, or interrupt it on the Console first',
  action_in_progress: 'Robot action is in progress; wait for it to finish',
  program_active: 'A program is currently running; wait for it to finish',
  calibration_step_running: 'Previous calibration step is still running',
  unknown_requires_acknowledgement_via_resume: 'Unknown state detected: inspect the arm and run Resume on the Console first',
  torque_off: 'Arm torque is off: enable Torque On on the Console first',
};

const PHASES = {
  bus: '读取舵机总线', cameras: '检查相机', board_check: '检查标定板', moving: '移动中', descending: '下降触碰桌面', contact: '已接触桌面',
  sample: '拍摄并检测标定板', parking: '回到 home', loading: '读取数据', solving: '求解中', cross_check: '交叉校验', installing: '写入并重新加载',
};
const PHASES_EN = {
  bus: 'Reading servo bus', cameras: 'Checking cameras', board_check: 'Checking board', moving: 'Moving',
  descending: 'Descending to touch table', contact: 'Table contact reached',
  sample: 'Capturing & detecting board', parking: 'Returning home', loading: 'Loading data',
  solving: 'Solving', cross_check: 'Cross-checking', installing: 'Writing & reloading',
};

const ERRORS = {
  input_owner_conflict: '另一个浏览器窗口正在控制这台机械臂',
  input_owner_claim_required: '需要先取得控制',
  input_owner_busy_step: '一个长命令（如标定步骤或回到 home）正在为持有控制的窗口运行，结束前不能换窗口取得控制；紧急停止、hold 和取消标定步骤在任何窗口都可用',
  control_requires_request_id: '请求缺少 request_id',
};
const ERRORS_EN = {
  input_owner_conflict: 'Another browser window holds control of this arm',
  input_owner_claim_required: 'Control lease required first',
  input_owner_busy_step: 'A long command (e.g. a calibration step or go_home) is running for the controlling window; wait until it finishes before claiming here. E-STOP, hold, and step cancel remain available in any window',
  control_requires_request_id: 'Request missing request_id',
  calibration_run_not_started: 'Calibration run has not been started yet',
  calibration_step_running: 'Another calibration step is currently running',
  operator_cancel: 'Cancelled by operator',
  S3_capture_not_started: 'S3 capture has not started yet',
  xy_snapshot_mismatch: 'XY snapshot does not match current frozen frame',
  candidate_stale_or_S5_not_passed: 'Candidate solve is stale or S5 has not passed',
  rollback_target_is_active: 'Target version is already active',
};

export function blockerText(codes) {
  const table = isEn() ? BLOCKERS_EN : BLOCKERS;
  return (codes || []).map((code) => table[code] || code);
}

// Reason / error codes the gateway and worker return as "code" or "code:detail".
export function describeReason(reason, definition) {
  if (!reason) return '';
  const [code, detail] = String(reason).split(/:(.*)/s);
  const en = isEn();
  const labels = (ids) => (ids || '').split(',').map((id) => {
    if (en && CALIB_READINESS_EN[id]) return CALIB_READINESS_EN[id];
    for (const step of definition?.steps || []) for (const check of step.readiness || []) if (check.id === id) return check.label;
    return id;
  }).join(pick('、', ', '));
  if (code === 'calibration_blocked') return blockerText((detail || '').split(',')).join(pick('；', '; '));
  if (code === 'readiness_failed') return pick(`准备检查未通过：${labels(detail)}`, `Readiness check failed: ${labels(detail)}`);
  if (code === 'readiness_unconfirmed') return pick(`请先勾选确认：${labels(detail)}`, `Confirm required checks first: ${labels(detail)}`);
  const table = en ? ERRORS_EN : ERRORS;
  if (table[code]) return table[code];
  return en ? translateStepFailureReasonEn(reason) : String(reason);
}

const num = (value, digits) => Number(value).toFixed(digits);
const DIGITS = { mm: 2, px: 3, rad: 4, s: 2, m: 4 };
export function formatMeasure(value, unit = '', scale = 1) {
  if (value === true) return pick('是', 'Yes');
  if (value === false) return pick('否', 'No');
  if (value == null || typeof value !== 'number' || !Number.isFinite(value)) return '—';
  const scaled = value * scale;
  if (!unit) return Number.isInteger(scaled) ? String(scaled) : num(scaled, 3);
  return `${num(scaled, DIGITS[unit] ?? 3)} ${unit}`;
}

const OP_TEXT = { '<=': '≤', '>=': '≥', '==': '=' };
// One criterion of a step verdict -> { label, measured, limit, passed, info }.
export function criterionRow(row, stepId = null) {
  const info = row.op === 'info';
  let limit;
  if (info) limit = pick('仅供参考', 'Info only');
  else if (row.op === 'true') limit = pick('必须为是', 'Must be Yes');
  else limit = `${OP_TEXT[row.op] || row.op} ${formatMeasure(row.threshold, row.unit, row.scale)}`;
  const label = isEn() ? criterionLabelEn(row, stepId) : row.label;
  return { label, measured: formatMeasure(row.value, row.unit, row.scale), limit, passed: row.passed, info };
}

// Pass criteria shown before a step runs (definition criteria, thresholds from the deployment config).
export function criterionRule(criterion) {
  if (criterion.threshold_value == null && criterion.threshold_label) {
    const target = isEn() ? (THRESHOLD_LABELS_EN[criterion.threshold_label] || criterion.threshold_label) : criterion.threshold_label;
    return `${OP_TEXT[criterion.op] || criterion.op} ${target}`;
  }
  return criterionRow({ ...criterion, value: null, threshold: criterion.threshold_value, passed: true, label: criterion.label }).limit;
}

// Old-vs-new row of the review screen. Lengths in m are shown in mm.
export function formatDiffRow(row) {
  const unit = row.unit === 'm' ? 'mm' : row.unit;
  const scale = row.unit === 'm' ? MM_PER_M : 1;
  const label = isEn() ? diffLabelEn(row) : row.label;
  return { label, old: formatMeasure(row.old, unit, scale), new: formatMeasure(row.new, unit, scale) };
}

// Steps a recalibration would redo for the ticked changes (display of the server mapping `definition.changes`; the server
// builds the real plan on begin).
export function planPreview(definition, selected) {
  const wanted = new Set();
  for (const change of definition?.changes || []) if (selected.has(change.id)) for (const step of change.steps) wanted.add(step);
  return (definition?.steps || []).map((step) => step.id).filter((id) => wanted.has(id));
}

// { label, kind, inPlan, status } of a step in the stepper.
export function stepView(definition, run, stepId) {
  const record = run?.steps?.[stepId];
  const inPlan = !run || run.plan.includes(stepId);
  const status = record?.status || 'not_started';
  const statusText = isEn() ? (CALIB_STATUSES_EN[status] || definition?.statuses?.[status] || status) : definition.statuses[status];
  return {
    status,
    inPlan,
    label: inPlan ? statusText : `${statusText} · ${pick('本轮不需要', 'Not needed')}`,
    kind: STATUS_KIND[status] || '',
  };
}

// First step of the plan after `stepId` that has not passed (or the first one with `stepId` null).
export function nextStep(run, stepId) {
  if (!run) return null;
  const plan = run.plan;
  const from = stepId == null ? 0 : plan.indexOf(stepId) + 1;
  return plan.slice(from).find((id) => run.steps[id]?.status !== 'passed') || null;
}

// First unfinished step of the plan, for opening the wizard on a run in progress.
export function currentStep(run) {
  return nextStep(run, null);
}

export function phaseText(update) {
  if (!update) return '';
  const phases = isEn() ? PHASES_EN : PHASES;
  const parts = [phases[update.phase] || update.phase];
  if (update.pose_index != null) parts.push(pick(`位姿 ${update.pose_index}/${update.pose_count}`, `Pose ${update.pose_index}/${update.pose_count}`));
  if (update.point_index) parts.push(pick(`点 ${update.point_index}/${update.point_count}`, `Point ${update.point_index}/${update.point_count}`));
  if (update.kept_count != null) parts.push(pick(`已保留 ${update.kept_count}`, `Kept ${update.kept_count}`));
  if (update.corners != null && typeof update.corners !== 'object') parts.push(pick(`角点 ${update.corners}`, `Corners ${update.corners}`));
  if (update.samples != null) parts.push(pick(`样本 ${update.samples}`, `Samples ${update.samples}`));
  return parts.join(' · ');
}

// Fraction (0..1) of a progress update, or null when it has no count.
export function phaseFraction(update) {
  if (update?.pose_count) return update.pose_index / update.pose_count;
  if (update?.point_count && update.point_index != null) return update.point_index / update.point_count;
  return null;
}

// Board form (mm and counts) <-> board spec (m).
export function boardToForm(board) {
  if (!board) return null;
  return { squares_x: board.squares_x, squares_y: board.squares_y, square_mm: board.square_m * MM_PER_M, marker_mm: board.marker_m * MM_PER_M,
    thickness_mm: board.thickness_m * MM_PER_M, dictionary: board.dictionary };
}

export function formToBoard(form) {
  const metres = (mm) => Number((Number(mm) / MM_PER_M).toFixed(6));
  return { squares_x: Number(form.squares_x), squares_y: Number(form.squares_y), square_m: metres(form.square_mm), marker_m: metres(form.marker_mm),
    thickness_m: metres(form.thickness_mm), dictionary: String(form.dictionary).trim() };
}

// Local-time string of an ISO-8601 UTC time.
export function localTime(iso) {
  const time = Date.parse(iso);
  return Number.isFinite(time) ? new Date(time).toLocaleString(isEn() ? localeTag() : undefined) : '—';
}

export function stalenessText(staleness) {
  if (!staleness) return pick('尚未检查', 'Not checked yet');
  switch (staleness.status) {
    case 'ok':
      return pick(
        `两个相机看到的标定板与标定一致（偏差 ${formatMeasure(staleness.rms_px, 'px')}）`,
        `Board poses in both cameras agree with calibration (RMS ${formatMeasure(staleness.rms_px, 'px')})`,
      );
    case 'shifted':
      return pick(
        `两个相机看到的标定板与标定不一致：偏差 ${formatMeasure(staleness.rms_px, 'px')} 超过 ${formatMeasure(staleness.threshold_px, 'px')}，相机或标定板可能被移动`,
        `Board mismatch across cameras: RMS ${formatMeasure(staleness.rms_px, 'px')} exceeds ${formatMeasure(staleness.threshold_px, 'px')} (camera or board may have moved)`,
      );
    case 'board_not_visible':
      return pick(
        '当前两个相机没有同时看到标定板，无法检查',
        'Board is not simultaneously visible in both cameras; cannot check',
      );
    default:
      return pick(
        `暂时无法检查${staleness.reason ? `（${staleness.reason}）` : ''}`,
        `Check unavailable${staleness.reason ? ` (${staleness.reason})` : ''}`,
      );
  }
}
