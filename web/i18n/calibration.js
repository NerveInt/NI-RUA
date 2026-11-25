// /calibration SO-101 标定向导词典及后端 /v1/calibration/state 结构化字段全量英文映射。
import { COMMON_TEXT } from './common.js';

export const CALIBRATION_I18N = {
  title: 'NI-RUA · SO-101 Calibration Wizard',
  text: {
    ...COMMON_TEXT,
    'SO-101 标定向导 · 仅操作员可见': 'SO-101 Calibration · Operator Only',
    '标定步骤': 'Calibration steps',
    '标定首页': 'Calibration Home',
    '正在读取标定状态…': 'Loading calibration state…',
  },
};

export const CALIB_STATUSES_EN = Object.freeze({
  not_started: 'Not started',
  running: 'Running',
  passed: 'Passed',
  failed: 'Failed',
  stale: 'Stale',
});

export const CALIB_MODES_EN = Object.freeze({
  first_run: {
    title: 'First-Time Full Calibration',
    description: 'First-time setup or full reset: complete S0–S6 in order.',
  },
  recalibrate: {
    title: 'Recalibrate (Select What Changed)',
    description: 'Re-run only the steps affected by physical changes.',
  },
});

export const CALIB_CHANGES_EN = Object.freeze({
  fixed_moved: 'Fixed camera moved',
  fixed_replaced: 'Fixed camera replaced / refocused',
  wrist_bumped: 'Wrist camera bumped',
  table_or_base_moved: 'Table changed or robot base moved',
  board_changed: 'Calibration board changed',
  everything: 'Redo everything',
});

export const CALIB_READINESS_EN = Object.freeze({
  torque_on: 'Arm torque is enabled',
  telemetry_fresh: 'Joint telemetry is fresh',
  cameras_streaming: 'Both cameras streaming with expected resolution',
  fixed_streaming: 'Fixed camera streaming with expected resolution',
  board_both_visible: 'Calibration board visible in both cameras',
  touches_available: 'Table touch data available (current S1 or active calibration)',
  joint_inputs_ready: 'S4 passed in this run and is not stale',
  wrist_calibration_available: 'Wrist camera calibration available (current S2 or active calibration)',
  fixed_calibration_available: 'Usable fixed-camera intrinsics available (current S3 or active version; failed/stale cannot fall back)',
  review_ready: 'S5 passed in this run and is not stale',
  applied: 'Applied version, loaded calibration, and live joint mapping match',
  touches_physically_confirmed: 'Post-capture images or external measurements reviewed: physical table contact and contact location match the geometric model (encoder lag or raw load alone is insufficient)',
  table_clear: 'Table area cleared (calibration board and objects removed)',
  workspace_clear: 'Robot motion envelope clear of hands and obstacles',
  board_matches: 'Board parameters match the physical board (squares, marker size, thickness)',
  error: 'Readiness check failed',
});

export const CALIB_STEPS_EN = Object.freeze({
  S0: {
    title: 'Device Self-Check',
    purpose: 'Verify the servo bus, both cameras, and joint telemetry before starting any calibration motion. Read-only; the arm does not move.',
    operator: [
      'Turn on the 12 V servo power supply and confirm the USB cable is connected',
      'Confirm both wrist and fixed cameras are connected with unobstructed lenses',
    ],
    machine: 'Reads the power-on preflight report; checks live servo voltage, temperature, and torque state; grabs a fresh frame from each camera to verify resolution; reads joint angles once to confirm telemetry freshness.',
    outputs: ['Self-check report'],
  },
  S1: {
    title: 'Joint Zero & Table Height',
    purpose: 'Collect stall candidates and full trajectories across multiple tabletop points to estimate FK/CAD table height. Passing capture does not prove physical contact; review independent evidence before S5.',
    operator: [
      'Remove the calibration board and all objects from the tabletop in front of the arm',
      'Keep hands clear of the arm and stay within reach of E-STOP',
    ],
    machine: 'Moves above each touch point, closes the gripper, descends in small steps while recording command, measured, and stall evidence (stall is not ground-truth contact; raw load is not force), then lifts and returns home.',
    outputs: ['Contact joint angles & contact points', 'Nominal table height'],
  },
  S2: {
    title: 'Wrist Intrinsics & Hand-Eye',
    purpose: 'Calibrate wrist camera intrinsics (focal length, distortion) and its mounting pose relative to the gripper (hand-eye).',
    operator: [
      'Place the calibration board flat on the table, visible to both cameras, and secure it in place',
      'Verify that the board parameters below match your physical board (editable)',
      'Do not move the board once capture starts',
    ],
    machine: 'Moves the arm at reduced speed through a grid of poses, capturing synchronized frames from both cameras once settled at each pose, then returns home and solves wrist intrinsics and hand-eye.',
    outputs: ['Wrist camera intrinsics', 'Hand-eye T_link_camera', 'Board views (for S5)'],
  },
  S3: {
    title: 'Fixed Camera Intrinsics',
    purpose: 'Calibrate the fixed camera’s own intrinsics. A static board gives only one perspective, so hold and tilt the board across positions and distances in front of the fixed camera.',
    operator: [
      'Hold the board in front of the fixed camera, pausing ~1 second at different image regions',
      'Vary distance (near, mid, far) and tilt angles',
      'Once coverage is sufficient, click "Solve", then place the board flat back on the table (needed for S4)',
    ],
    machine: 'Automatically selects steady, non-redundant views from the fixed camera stream, reports spatial/tilt coverage and live hints, and solves fixed-camera intrinsics. The arm does not move.',
    outputs: ['Fixed camera intrinsics'],
  },
  S4: {
    title: 'Fixed Camera Extrinsics',
    purpose: 'Solve the fixed camera pose in the robot base frame: with a static board on the table, the wrist kinematic chain provides the board pose while the fixed camera observes it simultaneously.',
    operator: [
      'Place the calibration board flat on the table, visible to both cameras, and secure it in place',
      'Do not move the board once capture starts',
    ],
    machine: 'Moves the arm through several poses, captures synchronized frames on both cameras, and solves fixed-camera extrinsics and chain reprojection error using current (or S3) fixed intrinsics.',
    outputs: ['Fixed camera extrinsics T_base_camera', 'Board views (for S5)'],
  },
  S5: {
    title: 'Joint Solve & Cross-Check',
    purpose: 'Combine board views, table touch points, and the "board top = table + board thickness" prior into one joint least-squares solve for joint zero offsets, hand-eye, and table height, followed by cross-checks.',
    operator: [
      'Inspect post-capture images or external measurements for each touch point to confirm physical table contact and contact geometry (never tick based on lag or load alone)',
      'Confirm and wait for the solve to finish (the arm does not move)',
    ],
    machine: 'Runs the joint optimization and checks board-to-table height agreement, touch residuals, fixed-camera chain reprojection, and joint zero offset bounds. Explains any failing gate.',
    outputs: ['Joint zero offset_rad', 'Hand-eye transform', 'Table height', 'Fixed camera extrinsics (with updated offsets)'],
  },
  S6: {
    title: 'Apply & Verify',
    purpose: 'Apply the new calibration and verify FK/CAD height consistency at held-out tabletop points. Shares contact detection and CAD assumptions (not an independent physical accuracy proof); configured thresholds (e.g. 5 mm) apply.',
    operator: [
      'Review the old-vs-new parameter diff and click "Apply New Calibration"',
      'Remove the calibration board and all objects from the table before starting verification touches',
      'Roll back to the previous version in one click if verification fails',
    ],
    machine: 'Writes a new version, switches active bundle, reloads camera calibration, updates joint offsets in the deployment file (with backup), then touches verification points to compare measured contact Z against the new table height.',
    outputs: ['Active calibration version', 'Verification touch results'],
  },
});

export const CALIB_CRITERIA_EN = Object.freeze({
  'S0:preflight_ok': 'Power-on preflight passed',
  'S0:power_ok': 'Servo voltage & temperature within preflight bounds',
  'S0:torque_ok': 'Torque enabled',
  'S0:telemetry_age_s': 'Joint telemetry age',
  'S0:cameras_ok': 'All cameras streaming at configured resolution',
  'S1:contacts': 'Stall-candidate touch points captured (pending physical review)',
  'S2:kept': 'Valid samples kept',
  'S2:wrist_intrinsics_rms_px': 'Wrist intrinsics RMS',
  'S2:hand_eye_translation_rms_m': 'Hand-eye translation RMS',
  'S2:hand_eye_rotation_rms_rad': 'Hand-eye rotation RMS',
  'S3:views': 'Kept views',
  'S3:fixed_intrinsics_rms_px': 'Fixed-camera intrinsics RMS',
  'S4:fixed_samples': 'Samples with board visible in fixed camera',
  'S4:fixed_chain_reprojection_px': 'Fixed-camera chain reprojection RMS (info; gated in S5 with solved offsets)',
  'S5:joint_reprojection_px': 'Wrist reprojection RMS',
  'S5:board_height_error_m': 'Board height − (table + thickness)',
  'S5:max_touch_residual_m': 'Max touch residual',
  'S5:fixed_chain_reprojection_px': 'Fixed-camera chain reprojection RMS',
  'S5:max_joint_offset_rad': 'Max joint zero |offset_rad|',
  'S6:contacts': 'Successful verification touch points',
  'S6:max_verify_error_m': 'Max verification error |contact Z − table Z|',
  // 无步骤前缀时的通用回退（按 id）
  preflight_ok: 'Power-on preflight passed',
  power_ok: 'Servo voltage & temperature within preflight bounds',
  torque_ok: 'Torque enabled',
  telemetry_age_s: 'Joint telemetry age',
  cameras_ok: 'All cameras streaming at configured resolution',
  kept: 'Valid samples kept',
  wrist_intrinsics_rms_px: 'Wrist intrinsics RMS',
  hand_eye_translation_rms_m: 'Hand-eye translation RMS',
  hand_eye_rotation_rms_rad: 'Hand-eye rotation RMS',
  views: 'Kept views',
  fixed_intrinsics_rms_px: 'Fixed-camera intrinsics RMS',
  fixed_samples: 'Samples with board visible in fixed camera',
  joint_reprojection_px: 'Wrist reprojection RMS',
  board_height_error_m: 'Board height − (table + thickness)',
  max_touch_residual_m: 'Max touch residual',
  fixed_chain_reprojection_px: 'Fixed-camera chain reprojection RMS',
  max_joint_offset_rad: 'Max joint zero |offset_rad|',
  max_verify_error_m: 'Max verification error |contact Z − table Z|',
});

// 中文标准名称反查表（兼容单测或仅有 label 的旧判定行）
const CRITERIA_ZH_TO_EN = Object.freeze({
  '上电自检报告通过': 'Power-on preflight passed',
  '舵机电压与温度在部署 preflight 范围内': 'Servo voltage & temperature within preflight bounds',
  '力矩已开启': 'Torque enabled',
  '关节读数延迟': 'Joint telemetry age',
  '每个相机出图且尺寸等于部署配置': 'All cameras streaming at configured resolution',
  '完成阻滞候选采集的点数（待物理审核）': 'Stall-candidate touch points captured (pending physical review)',
  '有效样本数': 'Valid samples kept',
  '腕部内参 RMS': 'Wrist intrinsics RMS',
  '手眼平移 RMS': 'Hand-eye translation RMS',
  '手眼旋转 RMS': 'Hand-eye rotation RMS',
  '保留视图数': 'Kept views',
  '固定相机内参 RMS': 'Fixed-camera intrinsics RMS',
  '固定相机看到标定板的样本数': 'Samples with board visible in fixed camera',
  '固定相机链路重投影 RMS（参考，S5 按新零位复核门限）': 'Fixed-camera chain reprojection RMS (info; gated in S5 with solved offsets)',
  '腕部重投影 RMS': 'Wrist reprojection RMS',
  '板面高度 − (桌面 + 板厚)': 'Board height − (table + thickness)',
  '最大触碰残差': 'Max touch residual',
  '固定相机链路重投影 RMS': 'Fixed-camera chain reprojection RMS',
  '最大关节零位 |offset_rad|': 'Max joint zero |offset_rad|',
  '接触成功的验证点数': 'Successful verification touch points',
  '最大验证误差 |接触高度 − 桌面高度|': 'Max verification error |contact Z − table Z|',
});

export function criterionLabelEn(row, stepId = null) {
  if (stepId && row?.id && CALIB_CRITERIA_EN[`${stepId}:${row.id}`]) return CALIB_CRITERIA_EN[`${stepId}:${row.id}`];
  if (row?.id && CALIB_CRITERIA_EN[row.id]) {
    if (row.id === 'fixed_chain_reprojection_px' && row.op === 'info') return CALIB_CRITERIA_EN['S4:fixed_chain_reprojection_px'];
    return CALIB_CRITERIA_EN[row.id];
  }
  return CRITERIA_ZH_TO_EN[row?.label] || row?.label || '';
}

export const THRESHOLD_LABELS_EN = Object.freeze({
  '计划点数': 'planned points',
});

const DIFF_KEY_EN = Object.freeze({
  table_z_m: 'Table height',
  'wrist.hand_eye_residual.translation_m': 'Hand-eye translation RMS',
  'wrist.hand_eye_residual.rotation_rad': 'Hand-eye rotation RMS',
  'fixed.extrinsic_residual.reprojection_px': 'Fixed-camera chain reprojection',
  'fixed.intrinsics_rms_px': 'Fixed-camera intrinsics RMS',
});

const DIFF_LABEL_ZH_TO_EN = Object.freeze({
  '桌面高度': 'Table height',
  '手眼平移 RMS': 'Hand-eye translation RMS',
  '手眼旋转 RMS': 'Hand-eye rotation RMS',
  '固定相机链路重投影': 'Fixed-camera chain reprojection',
  '固定相机内参 RMS': 'Fixed-camera intrinsics RMS',
});

export function diffLabelEn(row) {
  const key = row?.key || '';
  if (DIFF_KEY_EN[key]) return DIFF_KEY_EN[key];
  let match;
  if ((match = /^joint_offset\.(.+)$/.exec(key))) return `${match[1]} zero offset_rad`;
  if ((match = /^([^.]+)\.([^.]+)\.translation_change_m$/.exec(key))) return `${match[1]} ${match[2]} translation Δ`;
  if ((match = /^([^.]+)\.([^.]+)\.rotation_change_rad$/.exec(key))) return `${match[1]} ${match[2]} rotation Δ`;
  if ((match = /^([^.]+)\.K\.(.+)$/.exec(key))) return `${match[1]} focal length ${match[2]}`;
  if (DIFF_LABEL_ZH_TO_EN[row?.label]) return DIFF_LABEL_ZH_TO_EN[row.label];
  return row?.label || key;
}

const GRID_CELL_EN = Object.freeze({
  '画面左上角': 'top-left',
  '画面上方中间': 'top-center',
  '画面右上角': 'top-right',
  '画面左侧中间': 'middle-left',
  '画面中央': 'center',
  '画面右侧中间': 'middle-right',
  '画面左下角': 'bottom-left',
  '画面下方中间': 'bottom-center',
  '画面右下角': 'bottom-right',
});

const GUIDANCE_EXACT_EN = Object.freeze({
  '采集已足够，可以求解': 'Coverage is sufficient — ready to solve',
  '覆盖足够，可以求解': 'Coverage is sufficient — ready to solve',
  '未检测到标定板，请把标定板完整放到固定相机前': 'Board not detected — hold the full board in view of the fixed camera',
  '板子拿稳': 'Hold the board steady',
  '这个位置已采过，换个位置': 'This region is already sampled — move to another spot',
  '再靠近一些': 'Move the board closer',
  '再远一些': 'Move the board farther away',
  '倾斜一点': 'Tilt the board slightly more',
  '倾斜标定板': 'Tilt the calibration board',
});

export function translateGuidanceEn(hint) {
  if (!hint) return '';
  if (GUIDANCE_EXACT_EN[hint]) return GUIDANCE_EXACT_EN[hint];
  if (hint.startsWith('还缺：')) {
    const items = hint.slice(3).split('、').map((cell) => GRID_CELL_EN[cell] || cell);
    return `Still needed: ${items.join(', ')}`;
  }
  return hint;
}

export function translateReadinessDetailEn(detail) {
  if (!detail) return '';
  return String(detail)
    .replace(/\(需要 ≥ (\d+)\)/g, '(requires ≥ $1)')
    .replace('fixture: 未满足', 'fixture: not satisfied');
}

export function translateStepFailureReasonEn(reason) {
  if (!reason) return '';
  return String(reason)
    .split(/;\s*/)
    .map((part) => {
      const match = /^(.+?) 未通过$/.exec(part);
      if (!match) return part;
      return `${CRITERIA_ZH_TO_EN[match[1]] || match[1]} failed`;
    })
    .join('; ');
}

// 按语言生成本地化后的标定定义视图，不修改服务端原始对象。
export function localizeDefinition(definition, en = false) {
  if (!definition || !en) return definition;
  return {
    ...definition,
    statuses: { ...definition.statuses, ...CALIB_STATUSES_EN },
    modes: Object.fromEntries(
      Object.entries(definition.modes || {}).map(([key, value]) => [key, CALIB_MODES_EN[key] ? { ...value, ...CALIB_MODES_EN[key] } : value]),
    ),
    changes: (definition.changes || []).map((change) => ({
      ...change,
      label: CALIB_CHANGES_EN[change.id] || change.label,
    })),
    steps: (definition.steps || []).map((step) => {
      const override = CALIB_STEPS_EN[step.id] || {};
      return {
        ...step,
        ...override,
        readiness: (step.readiness || []).map((check) => ({
          ...check,
          label: CALIB_READINESS_EN[check.id] || check.label,
        })),
        criteria: (step.criteria || []).map((criterion) => ({
          ...criterion,
          label: criterionLabelEn(criterion, step.id),
          threshold_label: THRESHOLD_LABELS_EN[criterion.threshold_label] || criterion.threshold_label,
        })),
      };
    }),
  };
}
