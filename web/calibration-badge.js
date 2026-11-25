// Calibration status badge shared by /console and /calibration. Operator-only.
// Formats gateway summaries and classifies raw state for the homepage/older gateways.
import { isEn, onLocaleChange, pick } from './shared/i18n.js';

const MINUTE = 60, HOUR = 60 * MINUTE, DAY = 24 * HOUR; // seconds, unit conversion only
export const BADGE_KIND = { valid: 'ok', needed: 'bad', maybe_stale: 'warn', unknown: '', verification_failed: 'bad', unverified: 'warn' };
const LABELS_ZH = { valid: '标定有效', needed: '需要标定', maybe_stale: '可能已过期', unknown: '标定状态未知', verification_failed: '高度验证未通过', unverified: '标定已加载，未验收' };
const LABELS_EN = { valid: 'Calibrated', needed: 'Needs calib', maybe_stale: 'Possibly stale', unknown: 'Calib unknown', verification_failed: 'Height check failed', unverified: 'Loaded · unverified' };
const ZH_LABEL_SET = new Set(Object.values(LABELS_ZH));

function statusLabel(status, rawLabel) {
  if (isEn()) {
    if (rawLabel && !ZH_LABEL_SET.has(rawLabel)) return rawLabel;
    return LABELS_EN[status] || LABELS_EN.unknown;
  }
  return rawLabel || LABELS_ZH[status] || LABELS_ZH.unknown;
}

// Read the exact active version, never the newest/history-first entry. null is unknown evidence.
export function activeVerification(bundle = {}) {
  return Object.hasOwn(bundle, 'verification') ? bundle.verification :
    bundle.versions?.find((version) => version.name === bundle.active_version)?.verification ?? null;
}

// Also supports a gateway started before verification was included in /summary.
export function summaryFromState(state) {
  const bundle = state?.bundle;
  if (!bundle || state.available === false) return { status: 'unknown', label: statusLabel('unknown') };
  const verification = activeVerification(bundle);
  let status;
  if (!bundle.present || bundle.load_error) status = 'needed';
  else if (bundle.matches_active_version === false || bundle.has_joint_calibration === false ||
    bundle.fixed_intrinsics_source === 'copied_from_wrist_same_model_unverified') status = 'maybe_stale';
  else if (verification?.passed === false) status = 'verification_failed';
  else if (state.staleness?.status === 'shifted') status = 'maybe_stale';
  else if (verification?.passed !== true) status = 'unverified';
  else status = 'valid';
  return { status, label: statusLabel(status), verification, active_version: bundle.active_version,
    calibrated_at: bundle.calibration_created_at, staleness: state.staleness };
}

// "刚刚 / N分钟前 / N小时前 / N天前" (or English equivalent) from an ISO-8601 UTC time; null when missing/unparsable.
export function relativeTime(iso, now = Date.now()) {
  const time = Date.parse(iso);
  if (!Number.isFinite(time)) return null;
  const seconds = Math.max(0, Math.round((now - time) / 1000));
  if (seconds < MINUTE) return pick('刚刚', 'just now');
  if (seconds < HOUR) return pick(`${Math.floor(seconds / MINUTE)}分钟前`, `${Math.floor(seconds / MINUTE)}m ago`);
  if (seconds < DAY) return pick(`${Math.floor(seconds / HOUR)}小时前`, `${Math.floor(seconds / HOUR)}h ago`);
  return pick(`${Math.floor(seconds / DAY)}天前`, `${Math.floor(seconds / DAY)}d ago`);
}

// { text, kind, title } for one /v1/calibration/summary body (kind: ok | bad | warn | '').
export function badgeView(summary, now = Date.now()) {
  const status = summary?.status || 'unknown';
  const label = statusLabel(status, summary?.label);
  const ago = status === 'valid' || status === 'maybe_stale' ? relativeTime(summary.calibrated_at, now) : null;
  const detail = [];
  if (summary?.active_version) detail.push(pick(`当前版本 ${summary.active_version}`, `Active version ${summary.active_version}`));
  if (status === 'maybe_stale') detail.push(pick('相机或标定板位置与标定不一致，建议重新标定', 'Camera or board pose shifted from calibration; recalibration recommended'));
  if (status === 'verification_failed') detail.push(pick('当前已加载版本的 S6 高度验收失败', 'S6 height verification failed for the active version'));
  if (status === 'unverified') detail.push(pick('当前版本没有通过 S6 高度验收的记录', 'Active version has no passing S6 height verification record'));
  if (Number.isFinite(summary?.verification?.max_error_m)) detail.push(pick(`高度验收最大误差 ${(summary.verification.max_error_m * 1000).toFixed(3)} mm`, `Max height verification error ${(summary.verification.max_error_m * 1000).toFixed(3)} mm`));
  if (status === 'needed' && summary?.reason) detail.push(summary.reason);
  if (status === 'unknown' && summary?.reason) detail.push(summary.reason);
  const sep = pick('；', '; ');
  return { text: ago ? `${label} · ${ago}` : label, kind: BADGE_KIND[status] ?? '', title: detail.join(sep) || pick('打开标定向导', 'Open calibration wizard') };
}

export async function refreshBadge(node) {
  let summary;
  try {
    const response = await fetch('/v1/calibration/summary', { cache: 'no-store' });
    summary = response.ok ? await response.json() : { status: 'unknown', reason: `${response.status}` };
    if (response.ok && !Object.hasOwn(summary, 'verification') && summary.status !== 'unknown') {
      const stateResponse = await fetch('/v1/calibration/state', { cache: 'no-store' });
      summary = stateResponse.ok ? summaryFromState(await stateResponse.json()) : { status: 'unknown', reason: `${stateResponse.status}` };
    }
  } catch (error) { summary = { status: 'unknown', reason: error.message }; }
  if (node) node._lastCalibSummary = summary;
  applyBadge(node, summary);
  return summary;
}

export function applyBadge(node, summary) {
  if (node) node._lastCalibSummary = summary;
  const view = badgeView(summary);
  node.textContent = view.text;
  node.title = view.title;
  node.className = `pill ${view.kind}`;
}

// Display refresh period only; the worker throttles its own staleness check.
export const BADGE_REFRESH_MS = 10000;
export function mountCalibrationBadge(node) {
  refreshBadge(node);
  onLocaleChange(() => { if (node?._lastCalibSummary !== undefined) applyBadge(node, node._lastCalibSummary); });
  return setInterval(() => refreshBadge(node), BADGE_REFRESH_MS);
}
