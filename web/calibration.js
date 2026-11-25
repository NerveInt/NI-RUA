// SO-101 标定向导（/calibration）：仅操作员可见，由 GET /v1/calibration/state 类型化状态驱动并支持中英双语就地切换。
import './design-tokens.css';
import './workspace-navigation.css';
import { collapseHeaderToEstop } from './sticky-header.js';
import { applyBadge, relativeTime, badgeView, summaryFromState } from './calibration-badge.js';
import { mountXY } from './calibration-xy.js';
import { blockerText, boardToForm, criterionRow, criterionRule, currentStep, describeReason, formatDiffRow, formatMeasure, formToBoard,
  localTime, nextStep, phaseFraction, phaseText, planPreview, stalenessText, stepView } from './calibration-model.js';
import { byId as $, el, textNode as text } from './shared/dom.js';
import { requestJSON } from './shared/api-client.js';
import { createLeaseClient } from './shared/lease-client.js';
import { initLocale, isEn, localeTag, mountLanguageSwitch, onLocaleChange, pick, registerPageI18n } from './shared/i18n.js';
import { CALIBRATION_I18N, localizeDefinition, translateGuidanceEn, translateReadinessDetailEn } from './i18n/calibration.js';

initLocale();
registerPageI18n(CALIBRATION_I18N, document);
mountLanguageSwitch(document.querySelector('.head-state'));

const getJSON = (path, options) => requestJSON(path, options, { cache: 'no-store' });
const leaseClient = createLeaseClient({
  requestJSON: getJSON,
  onClaimError: (error) => setMessage('bad', pick(`未能取得控制 · ${describeReason(error.message)}`, `Failed to claim control · ${describeReason(error.message)}`)),
});
const clientId = leaseClient.clientId;
const ensureLease = leaseClient.ensureLease;
let runInfo, S;
let view = 'start', selected = null, mounted = null, inFlight = null, message = null, openedRun = null;
let renewing = false;
let pollError = null;
let connectionLost = null;
const confirmed = {}; // step id -> 当前页已勾选的操作员准备项集合
const changeSelection = new Set();
const detailsOpenState = new Map();
let boardDraft = null;
const def = () => localizeDefinition(S?.definition, isEn());

const POLL_MS = 1000, POLL_BUSY_MS = 500, PREVIEW_MS = 700;
const POLL_TIMEOUT_MS = 8000;

// 每次逻辑尝试复用同一 request_id，防断网重发机械臂动作。
function send(path, fields = {}, { lease = true } = {}) {
  return leaseClient.sendIdempotent(path, fields, { lease, reclaimOnExpired: true });
}

function setMessage(kind, content) { message = { kind, text: content, at: new Date() }; renderBanners(); }

function statusText(status) {
  const zh = {
    captured: '画面已冻结，请标注同一固定指尖',
    evaluated: 'XY 检查已记录，请查看本点结果',
    started: '已开始新的标定',
    ok: '完成',
    passed: '这一步通过',
    failed: '这一步未通过',
    rejected: '被拒绝（机械臂没有动）',
    applied: '已应用新标定',
    rolled_back: '已回滚',
    cancelled: '已取消并保持当前位置',
    idle: '当前没有正在运行的步骤',
    running: '已开始',
    unknown: 'UNKNOWN：无法确认机械臂是否已停下，请先到现场检查，不要重复发送',
  };
  const en = {
    captured: 'Frames frozen — click the same fixed fingertip in both images',
    evaluated: 'XY check recorded — review the point result below',
    started: 'New calibration run started',
    ok: 'Completed',
    passed: 'Step passed',
    failed: 'Step failed',
    rejected: 'Rejected (arm did not move)',
    applied: 'New calibration applied',
    rolled_back: 'Rolled back',
    cancelled: 'Cancelled and holding current position',
    idle: 'No calibration step is currently running',
    running: 'Started',
    unknown: 'UNKNOWN: could not confirm whether the arm stopped — inspect in person and do not resend',
  };
  return (isEn() ? en : zh)[status] || status;
}

function reportOutcome(label, result) {
  const operator = result.operator || {};
  const kind = result.status === 'passed' || result.status === 'applied' || result.status === 'rolled_back' || result.status === 'started' ? 'ok'
    : result.status === 'unknown' ? 'bad' : 'warn';
  const reason = describeReason(operator.reason, def());
  const admission = result.admission === 'open'
    ? pick(' · 动作准入已重新开放', ' · action admission reopened')
    : result.admission && result.admission !== 'closed'
      ? pick(' · 动作准入保持关闭（需要在操作台恢复）', ' · action admission remains closed (resume on Console)')
      : '';
  setMessage(kind, `${label} · ${statusText(result.status)}${reason ? ` · ${reason}` : ''}${admission}`);
}

async function act(label, path, fields, options) {
  setMessage('info', pick(`${label} · 等待后端回执…`, `${label} · waiting for backend receipt…`));
  try {
    const result = await send(path, fields, options);
    reportOutcome(label, result);
    return result;
  } catch (error) {
    const unknown = error.body?.status === 'unknown' || error.status >= 500;
    const errText = unknown
      ? statusText('unknown')
      : error.status === undefined
        ? pick('网络中断，结果未确认：再次点击会沿用同一个请求，不会重复执行', 'Network interrupted, outcome unconfirmed: clicking again reuses the same request_id without re-executing')
        : pick('失败', 'failed');
    setMessage(unknown ? 'bad' : 'warn', `${label} · ${errText} · ${describeReason(error.message, def())}`);
    return null;
  } finally { refresh(); }
}

// ---------- 轮询循环 ----------
async function refresh() {
  try { if ((await refreshOnce()) && pollError) { pollError = null; renderBanners(); } }
  catch (error) { pollError = error; console.error(error); renderBanners(); }
}
async function refreshOnce() {
  const startedAt = Date.now();
  let state, run;
  try {
    const signal = AbortSignal.timeout(POLL_TIMEOUT_MS);
    [state, run] = await Promise.all([getJSON('/v1/calibration/state', { signal }), getJSON('/v1/run', { signal })]);
  } catch (error) {
    if (error.status === undefined) {
      connectionLost = {
        reason: error.name === 'TimeoutError' ? pick(`${POLL_TIMEOUT_MS / 1000} s 内无回应`, `no response within ${POLL_TIMEOUT_MS / 1000} s`) : error.message,
        since: connectionLost?.since ?? new Date(),
      };
      renderBanners();
      return false;
    }
    connectionLost = null; S = { available: false, reason: error.message }; render();
    return false;
  }
  connectionLost = null; S = state; runInfo = run;
  if (leaseClient.token && startedAt > leaseClient.claimedAt && runInfo.controller?.owner_client_id !== clientId) leaseClient.clearToken();
  if (!runInfo.control_busy) leaseClient.claimRefusedBusy = false;
  if (!leaseClient.token && !runInfo.controller?.owner_client_id && !leaseClient.claimRefusedBusy) ensureLease({ quiet: true });
  openCurrentRun();
  render();
  return true;
}
function openCurrentRun() {
  const run = S?.run;
  if (!run || run.closed || openedRun === run.run_id) return;
  openedRun = run.run_id;
  const step = currentStep(run);
  if (step || Object.values(run.steps).some((record) => record.status !== 'not_started')) { view = 'step'; selected = step || run.plan[run.plan.length - 1]; }
}
async function loop() {
  try { await refresh(); } finally { setTimeout(loop, inFlight || S?.running_step ? POLL_BUSY_MS : POLL_MS); }
}

// ---------- 顶栏与横幅 ----------
function renderLease() {
  const owner = runInfo?.controller?.owner_client_id;
  const owned = Boolean(leaseClient.token && owner === clientId);
  const other = Boolean(owner && owner !== clientId);
  $('lease-state').textContent = owned
    ? pick('本窗口持有控制', 'Control held')
    : other
      ? pick('另一窗口持有控制 · 只读', 'Other window · read-only')
      : pick('监看模式', 'Monitor only');
  $('claim').hidden = owned; $('claim').disabled = other; $('release').hidden = !owned;
}
function renderBanners() {
  const banners = [];
  const timeFmt = (d) => d.toLocaleTimeString(localeTag(), { hour12: false });
  if (S && S.available === false) {
    banners.push(['bad', pick(
      `标定服务不可用：${describeReason(S.reason, def())}。机械臂后端需要是 SO-101 真机，并且工作进程在运行。`,
      `Calibration service unavailable: ${describeReason(S.reason, def())}. Robot backend must be a live SO-101 arm with its worker running.`,
    )]);
  }
  if (connectionLost) {
    banners.push(['bad', pick(
      `与网关的连接中断（${connectionLost.reason}），正在重试 · 下面是 ${timeFmt(connectionLost.since)} 前最后收到的状态`,
      `Connection to gateway lost (${connectionLost.reason}), retrying · showing last state received before ${timeFmt(connectionLost.since)}`,
    )]);
  }
  if (pollError) {
    banners.push(['bad', pick(
      `页面刷新出错，下面的显示可能不是最新：${pollError.message}`,
      `Page refresh error (display below may be stale): ${pollError.message}`,
    )]);
  }
  if (message) banners.push([message.kind === 'info' ? '' : message.kind, `${timeFmt(message.at)} · ${message.text}`]);
  if (runInfo && !runInfo.accepting_actions) {
    banners.push(['warn', pick(
      '动作准入已关闭：标定步骤结束后，如需让 Codex 继续动作，请到操作台「恢复准入」。',
      'Action admission is closed: after calibration finishes, click "Resume" on the Console before running Agent actions.',
    )]);
  }
  $('banners').replaceChildren(...banners.map(([kind, content]) => el('div', { className: `banner ${kind}`, textContent: content })));
}
const ownedByOther = () => Boolean(runInfo?.controller?.owner_client_id && runInfo.controller.owner_client_id !== clientId);
const blockersFor = (motion) => (S?.gateway?.blockers ? S.gateway.blockers[motion ? 'motion' : 'idle'] : []);

// ---------- 步骤导航 ----------
function renderStepper() {
  const definition = def();
  $('home-link').classList.toggle('active', view === 'start');
  if (!definition) { $('stepper').replaceChildren(); return; }
  $('stepper').replaceChildren(...definition.steps.map((step) => {
    const item = stepView(definition, S.run, step.id);
    const button = el('button', { type: 'button', className: `${item.status} ${item.inPlan ? '' : 'skipped'} ${view === 'step' && selected === step.id ? 'active' : ''}` },
      el('b', { textContent: step.id }), el('span', { textContent: step.title }), el('em', { textContent: item.label }));
    button.onclick = () => { view = 'step'; selected = step.id; render(); };
    return el('li', {}, button);
  }));
}

// ---------- 通用小组件 ----------
function pill(label, kind) { return el('span', { className: `pill ${kind || ''}`, textContent: label }); }
function section(title, ...content) { return el('section', {}, el('h3', { textContent: title }), ...content); }

function previewFigure(name, caption) {
  const image = el('img', { alt: caption, hidden: true });
  const missing = el('span', { className: 'missing', textContent: pick('暂无画面', 'No frame') });
  const figure = el('figure', {}, el('div', { className: 'frame' }, image, missing), el('figcaption', { textContent: caption }));
  figure.dataset.preview = name;
  figure.refresh = () => {
    const probe = new Image();
    probe.onload = () => { image.src = probe.src; image.hidden = false; missing.hidden = true; };
    probe.onerror = () => { image.hidden = true; missing.hidden = false; };
    probe.src = `/v1/calibration/preview/${name}.jpg?t=${Date.now()}`;
  };
  return figure;
}
setInterval(() => { for (const figure of document.querySelectorAll('[data-preview]')) figure.refresh(); }, PREVIEW_MS);

function resultCard(record, definition, step) {
  const result = record.result || {};
  const verdict = result.verdict;
  const rows = (verdict?.criteria || []).map((c) => criterionRow(c, step));
  const passed = record.status === 'passed';
  const card = el('div', { className: `card result ${record.status}` },
    el('div', { className: 'head-row' },
      el('h2', { textContent: step === 'S6' ? pick('触桌高度验证结果', 'Table-Height Verification Result') : pick('结果', 'Result') }),
      pill(definition.statuses[record.status], passed ? 'ok' : record.status === 'failed' ? 'bad' : 'warn'),
      el('span', { className: 'meta', textContent: record.finished_at ? `${localTime(record.finished_at)} ${pick(`（${relativeTime(record.finished_at)}）`, `(${relativeTime(record.finished_at)})`)}` : '' })));
  if (record.reason && !passed) card.append(el('p', { className: 'fail', textContent: describeReason(record.reason, definition) }));
  if (rows.length) {
    const headers = isEn() ? ['Criterion', 'Measured', 'Requirement', ''] : ['项目', '测得', '要求', ''];
    card.append(el('div', { className: 'table-wrap' }, el('table', {}, el('thead', {}, el('tr', {}, ...headers.map((label) => el('th', { textContent: label })))),
      el('tbody', {}, ...rows.map((row) => el('tr', {}, el('td', { textContent: row.label }), el('td', { className: 'num', textContent: row.measured }),
        el('td', { className: 'num', textContent: row.limit }), el('td', { className: row.info ? '' : row.passed ? 'pass' : 'fail', textContent: row.info ? pick('参考', 'info') : row.passed ? '✓' : '✗' })))))));
  }
  const touches = result.touches || [];
  if (touches.length) {
    const touchHeaders = isEn() ? ['#', 'xy (m)', 'Outcome', 'Contact Z', 'Trigger'] : ['#', 'xy (m)', '结果', '接触高度', '触发'];
    card.append(section(pick('触碰点', 'Touch Points'), el('div', { className: 'table-wrap' }, el('table', {}, el('thead', {}, el('tr', {}, ...touchHeaders.map((label) => el('th', { textContent: label })))),
      el('tbody', {}, ...touches.map((touch) => el('tr', {}, el('td', { textContent: String(touch.index) }), el('td', { className: 'num', textContent: (touch.xy || []).map((v) => formatMeasure(v, 'm')).join(', ') }),
        el('td', { textContent: touch.status || '—' }), el('td', { className: 'num', textContent: formatMeasure(touch.contact_z_m, 'mm', 1000) }),
        el('td', { textContent: touch.trigger || '—' }))))))));
  }
  const detail = { ...result }; delete detail.verdict;
  if (Object.keys(detail).length) {
    const detailKey = `detail:${step}`;
    const isOpen = detailsOpenState.get(detailKey) ?? false;
    const detailsEl = el('details', isOpen ? { open: true } : {}, el('summary', { textContent: pick('详细数据（原始记录）', 'Detailed Data (Raw Record)') }), el('pre', { className: 'mono', textContent: JSON.stringify(detail, null, 2) }));
    detailsEl.ontoggle = () => { detailsOpenState.set(detailKey, detailsEl.open); };
    card.append(detailsEl);
  }
  const next = nextStep(S.run, step);
  const action = el('p', {});
  if (passed) {
    const nextTitle = next ? definition.steps.find((s) => s.id === next)?.title : '';
    action.append(text(next
      ? pick(`这一步通过。下一步：${next} ${nextTitle}。 `, `Step passed. Next: ${next} ${nextTitle}. `)
      : pick('这一步通过，本轮没有后续步骤了。 ', 'Step passed; all planned steps in this run are complete. ')));
    if (next) {
      const go = el('button', { type: 'button', className: 'primary', textContent: pick(`前往 ${next}`, `Go to ${next}`) });
      go.onclick = () => { selected = next; render(); };
      action.append(go);
    }
  } else if (record.status === 'stale') {
    action.append(text(pick('上游步骤已重新做过，这一步的结果需要重做。', 'An upstream step was re-run; this step is stale and must be repeated.')));
  } else {
    action.append(text(step === 'S6'
      ? pick('高度验收未通过：新标定仍已加载，没有自动回滚。请先核对测量结果，再决定重新验证或回滚到上一版本。', 'Height verification failed: the new calibration remains loaded (no automatic rollback). Review the measurements, then re-verify or roll back to the previous version.')
      : pick('未通过：排除上面的问题后，可以点上面的「重新执行」再来一次；旧标定不受影响。', 'Step failed: resolve the issue above and click "Re-run" to try again; active calibration is unaffected.')));
  }
  card.append(action);
  return card;
}

// ---------- 标定首页 ----------
function mountStart() {
  const definition = def();
  const status = el('div', { id: 'start-status' });
  const resume = el('div', { id: 'start-resume' });
  const history = el('div', { id: 'start-history' });
  const firstRun = el('div', { className: 'card' }, el('h2', { textContent: definition.modes.first_run.title }), el('p', { textContent: definition.modes.first_run.description }),
    el('p', { className: 'meta', textContent: `${pick('步骤：', 'Steps: ')}${definition.steps.map((step) => `${step.id} ${step.title}`).join(' → ')}` }));
  const startFirst = el('button', { type: 'button', className: 'primary', textContent: pick('开始首次完整标定', 'Start First-Time Full Calibration') });
  startFirst.onclick = () => beginRun('first_run', []);
  firstRun.append(el('div', { className: 'row' }, startFirst));
  const preview = el('p', { className: 'meta' });
  const startRe = el('button', { type: 'button', className: 'primary', textContent: pick('开始重新标定', 'Start Recalibration') });
  const list = el('ul', { className: 'change-list' }, ...definition.changes.map((change) => {
    const box = el('input', { type: 'checkbox', checked: changeSelection.has(change.id) });
    box.onchange = () => { box.checked ? changeSelection.add(change.id) : changeSelection.delete(change.id); updatePreview(); };
    return el('li', {}, el('label', {}, box, text(`${change.label}`)));
  }));
  const updatePreview = () => {
    const plan = planPreview(definition, changeSelection);
    preview.textContent = plan.length
      ? pick(`会重做：${plan.join(' → ')}`, `Steps to re-run: ${plan.join(' → ')}`)
      : pick('勾选哪里变了，下面会列出需要重做的步骤', 'Check what changed above to preview the steps to re-run');
    startRe.disabled = !plan.length;
  };
  startRe.onclick = () => beginRun('recalibrate', [...changeSelection]);
  const recal = el('div', { className: 'card' }, el('h2', { textContent: definition.modes.recalibrate.title }), el('p', { textContent: definition.modes.recalibrate.description }), list, preview,
    el('div', { className: 'row' }, startRe));
  updatePreview();
  const board = boardForm('start');
  const boardCard = el('div', { className: 'card' }, board.node);
  $('content').replaceChildren(status, resume, el('div', { className: 'cards' }, firstRun, recal), boardCard, history);
  return { status, resume, history, startFirst, startRe, board };
}

function beginRun(mode, changes) {
  openedRun = null;
  view = 'step';
  const boardFormValue = boardDraft || boardToForm(S.default_board || S.run?.board);
  const board = boardFormValue ? formToBoard(boardFormValue) : undefined;
  const payload = mode === 'first_run' ? { mode } : { mode, changes };
  if (board) payload.board = board;
  act(pick('开始标定', 'Begin calibration'), '/v1/calibration/begin', payload).then((result) => {
    if (result?.status === 'started') { boardDraft = null; for (const key of Object.keys(confirmed)) delete confirmed[key]; }
    else view = 'start';
  });
}

function renderStart() {
  const definition = def();
  if (mounted?.key !== 'start') mounted = Object.assign(mountStart(), { key: 'start' });
  const { status, resume, history, startFirst, startRe, board } = mounted;
  board?.update();
  const bundle = S.bundle || {};
  const summary = summaryFromState(S), badge = badgeView(summary);
  const blocked = ownedByOther() || blockersFor(false).length > 0 || Boolean(inFlight);
  startFirst.disabled = blocked;
  startRe.disabled = blocked || !planPreview(definition, changeSelection).length;
  const kv = (label, value) => [el('dt', { textContent: label }), el('dd', { className: 'mono', textContent: value })];
  const residuals = Object.entries(bundle.residuals || {}).map(([key, value]) => `${key} ${typeof value === 'number' ? formatMeasure(value, '') : JSON.stringify(value)}`).join(' · ');
  const verText = summary.verification
    ? `${summary.verification.passed === true ? pick('通过', 'Passed') : summary.verification.passed === false ? pick('未通过', 'Failed') : pick('未完成', 'Incomplete')} · ${pick('最大误差', 'max error')} ${formatMeasure(summary.verification.max_error_m, 'mm', 1000)}`
    : pick('未验收', 'Unverified');
  status.replaceChildren(el('div', { className: 'card' }, el('div', { className: 'head-row' }, el('h2', { textContent: pick('当前标定', 'Active Calibration') }),
    pill(badge.text, badge.kind)),
    el('dl', { className: 'kv' },
      ...kv(pick('标定时间', 'Calibrated at'), bundle.calibration_created_at ? localTime(bundle.calibration_created_at) : '—'),
      ...kv(pick('当前版本', 'Active version'), bundle.active_version || (bundle.present ? pick('未纳入版本库（旧文件）', 'Unversioned legacy file') : '—')),
      ...kv(pick('桌面高度', 'Table height'), formatMeasure(bundle.table_z_m, 'mm', 1000)),
      ...kv(pick('残差', 'Residuals'), residuals || '—'),
      ...kv(pick('检查', 'Staleness check'), stalenessText(S.staleness)),
      ...kv(pick('高度验收', 'Height check'), verText),
      ...(bundle.load_error ? kv(pick('加载错误', 'Load error'), bundle.load_error) : []))));
  const run = S.run;
  const unfinished = run && !run.closed && run.plan.some((id) => run.steps[id].status !== 'passed');
  resume.replaceChildren(...(run && !run.closed ? [el('div', { className: 'card' },
    el('h2', { textContent: unfinished ? pick('有一次标定正在进行', 'Calibration Run in Progress') : pick('上一次标定已完成', 'Previous Calibration Run Completed') }),
    el('p', { className: 'meta', textContent: `${definition.modes[run.mode]?.title || run.mode} · ${pick('开始于', 'started')} ${localTime(run.created_at)} · ${run.plan.map((id) => `${id} ${definition.statuses[run.steps[id].status]}`).join(pick('，', ', '))}` }),
    (() => {
      const go = el('button', { type: 'button', className: 'primary', textContent: unfinished ? pick('继续', 'Continue') : pick('查看', 'Inspect') });
      go.onclick = () => { view = 'step'; selected = currentStep(run) || run.plan[run.plan.length - 1]; render(); };
      return el('div', { className: 'row' }, go);
    })())] : []));
  const versions = bundle.versions || [];
  const histHeaders = isEn() ? ['Version', 'Created at', 'Source', 'Table Z', 'Verification', ''] : ['版本', '创建时间', '来源', '桌面高度', '验证', ''];
  history.replaceChildren(el('div', { className: 'card' }, el('h2', { textContent: pick('版本历史', 'Version History') }), versions.length ? el('div', { className: 'table-wrap' }, el('table', {},
    el('thead', {}, el('tr', {}, ...histHeaders.map((label) => el('th', { textContent: label })))),
    el('tbody', {}, ...versions.map((version) => {
      const active = version.name === bundle.active_version;
      const rollback = el('button', { type: 'button', textContent: pick('回滚到此版本', 'Roll back to this version'), disabled: active || blocked });
      rollback.onclick = () => rollbackTo(version.name);
      const verification = version.verification;
      return el('tr', {},
        el('td', { className: 'mono', textContent: version.name + (active ? pick('（当前）', ' (active)') : '') }),
        el('td', { textContent: localTime(version.created_at) }),
        el('td', { textContent: `${version.source || '—'}${version.mode ? ` · ${definition.modes[version.mode]?.title || version.mode}` : ''}` }),
        el('td', { className: 'num', textContent: formatMeasure(version.table_z_m, 'mm', 1000) }),
        el('td', { className: verification ? (verification.passed ? 'pass' : 'fail') : '', textContent: verification ? `${verification.passed ? pick('通过', 'Passed') : pick('未通过', 'Failed')} · ${formatMeasure(verification.max_error_m, 'mm', 1000)}` : pick('未验证', 'Unverified') }),
        el('td', {}, rollback));
    })))) : el('p', { className: 'meta', textContent: pick('还没有版本记录。完成一次标定并应用后，这里会列出每个版本，可一键回滚。', 'No calibration versions recorded yet. Once you complete and apply a calibration, versions appear here for one-click rollback.') })));
}

function rollbackTo(version) {
  const promptMsg = pick(
    `回滚到版本 ${version}？机械臂不会动，但相机标定和关节零位会切换回该版本，之后的观察使用它。`,
    `Roll back to version ${version}? The arm will not move, and camera calibration plus joint zero offsets will switch back to ${version} for subsequent observations.`,
  );
  if (!window.confirm(promptMsg)) return;
  act(pick(`回滚到 ${version}`, `Roll back to ${version}`), '/v1/calibration/rollback', { version });
}

// ---------- 步骤详情页 ----------
function formatBoardSpecLine(b) {
  if (!b) return '';
  return pick(
    `${b.squares_x}×${b.squares_y} · 方格 ${(b.square_m * 1000).toFixed(1)} mm · 标记 ${(b.marker_m * 1000).toFixed(1)} mm · 板厚 ${(b.thickness_m * 1000).toFixed(1)} mm · ${b.dictionary}`,
    `${b.squares_x}×${b.squares_y} · square ${(b.square_m * 1000).toFixed(1)} mm · marker ${(b.marker_m * 1000).toFixed(1)} mm · thickness ${(b.thickness_m * 1000).toFixed(1)} mm · ${b.dictionary}`,
  );
}

function boardForm(stepId) {
  const fields = [
    ['squares_x', pick('横向方格数', 'Columns (squares_x)'), 1],
    ['squares_y', pick('纵向方格数', 'Rows (squares_y)'), 1],
    ['square_mm', pick('方格边长 (mm)', 'Square size (mm)'), 0.01],
    ['marker_mm', pick('标记边长 (mm)', 'Marker size (mm)'), 0.01],
    ['thickness_mm', pick('板厚 (mm)', 'Board thickness (mm)'), 0.01],
  ];
  const inputs = {};
  const grid = el('div', { className: 'form-grid' }, ...fields.map(([key, label, step]) => {
    inputs[key] = el('input', { type: 'number', step: String(step), min: '0' });
    inputs[key].oninput = () => { boardDraft = Object.fromEntries(Object.entries(inputs).map(([name, input]) => [name, input.value])); boardDraft.dictionary = inputs.dictionary.value; };
    return el('label', {}, text(label), inputs[key]);
  }), (() => { inputs.dictionary = el('input', { type: 'text' }); inputs.dictionary.oninput = inputs.square_mm.oninput; return el('label', {}, text(pick('标记字典', 'ArUco dictionary')), inputs.dictionary); })());
  const update = () => {
    if (boardDraft) { for (const [key, input] of Object.entries(inputs)) if (document.activeElement !== input) input.value = boardDraft[key]; return; }
    const source = stepId === 'start' ? (S.default_board || S.run?.board) : (S.run?.board || S.default_board);
    const form = boardToForm(source);
    if (form) for (const [key, input] of Object.entries(inputs)) if (document.activeElement !== input) input.value = form[key];
  };
  return {
    node: section(
      pick('标定板参数（与实物不一致会得到错误的标定）', 'Calibration Board Parameters (mismatch yields wrong scale)'),
      grid,
      el('p', { className: 'meta', textContent: pick('默认值来自当前标定文件；板厚是标定板放在桌面上时板面比桌面高出的厚度。开始这一步时会一起发送。', 'Defaults come from the active calibration file; thickness is the height of the board top surface above the tabletop. Sent when starting the step.') }),
    ),
    update,
  };
}

function mountStep(stepId) {
  const definition = def();
  const step = definition.steps.find((item) => item.id === stepId);
  const refs = { step, confirm: confirmed[stepId] || (confirmed[stepId] = new Set()) };
  const head = el('div', { className: 'head-row' }, el('h2', { textContent: `${step.id} · ${step.title}` }));
  refs.statusPill = pill('', '');
  head.append(refs.statusPill, pill(step.motion ? pick('机械臂会动', 'Arm moves') : pick('机械臂不动', 'Arm stationary'), step.motion ? 'warn' : ''));
  const intro = el('div', { className: 'card' }, head,
    section(pick('这一步做什么', 'Purpose'), el('p', { textContent: step.purpose }),
      el('details', {}, el('summary', { textContent: pick('机器具体会怎么做', 'What the robot does') }), el('p', { textContent: step.machine }))),
    section(pick('你需要做', 'Operator Checklist'), el('ol', {}, ...step.operator.map((line) => el('li', { textContent: line })))),
    section(pick('通过标准', 'Pass Criteria'), el('ul', {}, ...step.criteria.map((criterion) => el('li', { textContent: `${criterion.label}${pick('：', ': ')}${criterionRule(criterion)}` })))));
  const parts = [intro];
  if (stepId === 'S2') {
    refs.board = boardForm(stepId);
    intro.append(refs.board.node);
  } else {
    const b = S.run?.board || S.default_board;
    refs.boardSpec = el('p', { className: 'meta mono', textContent: formatBoardSpecLine(b) });
    intro.append(refs.boardSpec);
  }
  refs.review = stepId === 'S6' ? el('div', { className: 'card' }) : null;
  if (refs.review) parts.push(refs.review);
  if (stepId === 'S6') {
    refs.xy = mountXY({ el, act, defaults: S.xy_defaults,
      blocked: () => !S?.run || S.run.closed || ownedByOther() || Boolean(inFlight) || blockersFor(false).length > 0 });
    parts.push(refs.xy.node);
  }
  refs.autoChecks = el('ul', { className: 'checks' });
  refs.operatorChecks = el('ul', { className: 'checks' }, ...step.readiness.filter((check) => check.kind === 'operator').map((check) => {
    const box = el('input', { type: 'checkbox', checked: refs.confirm.has(check.id) });
    box.onchange = () => { box.checked ? refs.confirm.add(check.id) : refs.confirm.delete(check.id); renderStep(); };
    return el('li', {}, el('label', {}, box, text(check.label)));
  }));
  parts.push(el('div', { className: 'card' }, el('h2', { textContent: pick('准备检查', 'Readiness Checks') }), refs.autoChecks, refs.operatorChecks));
  const names = stepId === 'S3'
    ? [['fixed_corners', pick('固定相机（角点：绿色 = 拿稳，橙色 = 在动）', 'Fixed camera (corners: green = steady, orange = moving)')]]
    : stepId === 'S5'
      ? []
      : [
        ['wrist', pick('wrist 腕部相机（绿圈 = 检测到的标定板角点）', 'wrist camera (green circles = detected ChArUco corners)')],
        ['fixed', pick('fixed 固定相机（绿圈 = 角点；橙/紫圈 = 触碰点/验证点，红十字 = 指尖）', 'fixed camera (green = corners; orange/purple = touch/verify points; red cross = fingertip)')],
      ];
  if (names.length) {
    const frames = el('div', { className: 'frames' }, ...names.map(([name, caption]) => previewFigure(name, caption)));
    parts.push(el('div', { className: 'card' }, el('h2', { textContent: pick('实时画面', 'Live Camera Preview') }), frames));
    for (const figure of frames.children) figure.refresh();
  }
  if (stepId === 'S3') {
    refs.hhBar = el('progress', { max: 1, value: 0 }); refs.hhText = el('p', { className: 'meta mono' }); refs.hhGrid = el('table', { className: 'hh-grid' });
    refs.hhBuckets = el('p', { className: 'meta mono', style: 'white-space:pre-line' }); refs.hhGuidance = el('ul', { className: 'guidance' });
    parts.push(el('div', { className: 'card' }, el('h2', { textContent: pick('采集进度', 'Capture Coverage') }), refs.hhBar, refs.hhText, el('div', { className: 'hh-view' }, refs.hhGuidance, el('div', {}, refs.hhGrid, refs.hhBuckets))));
  }
  refs.hint = el('p', { className: 'meta' });
  refs.start = el('button', { type: 'button', className: 'primary' });
  refs.stop = el('button', { type: 'button', textContent: pick('停止采集', 'Stop Capture') });
  refs.solve = el('button', { type: 'button', className: 'primary', textContent: pick('求解', 'Solve') });
  refs.cancel = el('button', { type: 'button', className: 'warn', textContent: pick('取消这一步（机械臂保持当前位置）', 'Cancel Step (hold current arm position)') });
  refs.start.onclick = () => startStep(refs, 'capture');
  refs.stop.onclick = () => startStep(refs, 'stop');
  refs.solve.onclick = () => startStep(refs, 'solve');
  refs.cancel.onclick = () => act(pick('取消', 'Cancel'), '/v1/calibration/cancel', {}, { lease: false });
  refs.progressText = el('p', { className: 'meta mono' }); refs.progressBar = el('progress', { max: 1, value: 0 });
  parts.push(el('div', { className: 'card' }, el('h2', { textContent: pick('操作与进度', 'Controls & Progress') }),
    el('div', { className: 'row' }, refs.start, ...(stepId === 'S3' ? [refs.stop, refs.solve] : []), refs.cancel), refs.hint, refs.progressBar, refs.progressText));
  refs.result = el('div', {});
  parts.push(refs.result);
  $('content').replaceChildren(...parts);
  return refs;
}

function startStep(refs, action) {
  const step = refs.step;
  const params = { confirmed: [...refs.confirm] };
  if (step.id === 'S2') params.board = formToBoard(boardDraft || boardToForm(S.run?.board || S.default_board));
  if (step.id === 'S3') { params.action = action; delete params.board; }
  const label = step.id === 'S3'
    ? {
      capture: pick('S3 开始采集', 'S3 Start Capture'),
      stop: pick('S3 停止采集', 'S3 Stop Capture'),
      solve: pick('S3 求解', 'S3 Solve'),
    }[action]
    : `${step.id} ${step.title}`;
  const isLong = !(step.id === 'S3' && action !== 'solve');
  if (isLong) inFlight = { step: step.id };
  act(label, `/v1/calibration/steps/${step.id}/start`, { params }).finally(() => { inFlight = null; });
  render();
}

function renderReview(refs) {
  const run = S.run, candidate = run?.candidate;
  const definition = def();
  const s5 = run?.steps?.S5;
  const body = [el('h2', { textContent: pick('复核与应用', 'Review & Apply') })];
  if (!candidate || s5?.status !== 'passed' || candidate.stale) {
    body.push(el('p', { textContent: s5?.status === 'passed'
      ? pick('求解结果已过期（上游步骤重做过），请重新执行 S5。', 'Solved candidate is stale (an upstream step was re-run); please run S5 again.')
      : pick('先完成 S5（联合求解）。通过后，这里会显示新旧数值对比，确认后才会写入。', 'Complete S5 (Joint Solve) first. Once passed, the old-vs-new parameter diff will appear here before writing.') }));
  } else {
    body.push(el('p', { className: 'meta', textContent: pick(
      '新数值还没有写入。确认下面的对比没问题后点「应用」：会存成新版本并立即生效，关节零位写入部署文件（自动备份），机械臂不会动。',
      'New parameters have not been written yet. Review the diff below and click "Apply New Calibration": saves a new version, activates it immediately, and writes joint zero offsets to the deployment file (with backup) without moving the arm.',
    ) }));
    body.push(resultCardSummary(s5));
    const diffHeaders = isEn() ? ['Parameter', 'Old', 'New'] : ['项目', '旧', '新'];
    body.push(el('div', { className: 'table-wrap' }, el('table', {}, el('thead', {}, el('tr', {}, ...diffHeaders.map((label) => el('th', { textContent: label })))),
      el('tbody', {}, ...(candidate.diff || []).map(formatDiffRow).map((row) => el('tr', {}, el('td', { textContent: row.label }), el('td', { className: 'num', textContent: row.old }), el('td', { className: 'num', textContent: row.new })))))));
    if (run.applied_version) {
      body.push(el('p', { className: 'pass', textContent: pick(
        `已应用：版本 ${run.applied_version}${run.previous_version ? `（上一个版本 ${run.previous_version}）` : ''}`,
        `Applied: version ${run.applied_version}${run.previous_version ? ` (previous: ${run.previous_version})` : ''}`,
      ) }));
      if (run.previous_version) {
        const back = el('button', { type: 'button', className: 'danger', textContent: pick('回滚到上一个版本', 'Roll back to previous version'), disabled: ownedByOther() || blockersFor(false).length > 0 });
        back.onclick = () => rollbackTo(run.previous_version);
        body.push(el('div', { className: 'row' }, back));
      }
    } else {
      const box = el('input', { type: 'checkbox', checked: refs.reviewed || false });
      const apply = el('button', { type: 'button', className: 'primary', textContent: pick('应用新标定', 'Apply New Calibration'), disabled: !refs.reviewed || ownedByOther() || blockersFor(false).length > 0 || Boolean(inFlight) });
      box.onchange = () => { refs.reviewed = box.checked; renderStep(); };
      apply.onclick = () => act(pick('应用新标定', 'Apply New Calibration'), '/v1/calibration/apply', {});
      body.push(el('div', { className: 'row' }, el('label', {}, box, text(pick(' 我已核对新旧数值', ' I have reviewed the old-vs-new parameter values'))), apply));
    }
  }
  refs.review.replaceChildren(...body);
}
function resultCardSummary(record) {
  const rows = (record.result?.verdict?.criteria || []).map((c) => criterionRow(c, 'S5'));
  return el('details', { open: true }, el('summary', { textContent: pick('S5 交叉校验结果', 'S5 Cross-Check Summary') }), el('ul', {}, ...rows.map((row) =>
    el('li', {}, text(`${row.label}${pick('：', ': ')}${row.measured} (${row.limit}) `), el('span', { className: row.info ? '' : row.passed ? 'pass' : 'fail', textContent: row.info ? pick('参考', 'info') : row.passed ? '✓' : '✗' })))));
}

function renderStep() {
  const definition = def();
  const key = `step:${selected}:${S.run?.run_id || ''}`;
  if (mounted?.key !== key) mounted = Object.assign(mountStep(selected), { key });
  const refs = mounted, step = definition.steps.find((item) => item.id === selected) || refs.step, run = S.run;
  refs.step = step;
  const record = run?.steps?.[step.id];
  const status = record?.status || 'not_started';
  refs.statusPill.textContent = definition.statuses[status];
  refs.statusPill.className = `pill ${status === 'passed' ? 'ok' : status === 'failed' ? 'bad' : status === 'running' || status === 'stale' ? 'warn' : ''}`;
  refs.board?.update();
  if (refs.boardSpec) {
    const b = run?.board || S.default_board;
    if (b) refs.boardSpec.textContent = formatBoardSpecLine(b);
  }
  if (refs.review) renderReview(refs);
  refs.xy?.update(run);
  const live = S.readiness?.[step.id] || {};
  const autoChecks = step.readiness.filter((check) => check.kind === 'auto');
  refs.autoChecks.replaceChildren(...autoChecks.map((check) => {
    const state = live[check.id];
    const kind = state?.ok === true ? 'ok' : state?.ok === false ? 'bad' : 'unknown';
    const detail = state?.detail ? (isEn() ? translateReadinessDetailEn(state.detail) : state.detail) : '';
    return el('li', {}, el('span', { className: `tick ${kind}`, textContent: kind === 'ok' ? '✓' : kind === 'bad' ? '✗' : '·' }), text(check.label),
      el('span', { className: 'meta', textContent: detail ? ` ${detail}` : '' }));
  }));
  const running = status === 'running';
  const capturing = step.id === 'S3' && S.handheld?.status === 'running';
  const busy = Boolean(inFlight) || Boolean(S.running_step);
  const blockers = blockersFor(step.motion);
  const boardChanged = Boolean(boardDraft) || (S.default_board && S.run?.board && (S.default_board.squares_x !== S.run.board.squares_x || S.default_board.squares_y !== S.run.board.squares_y || S.default_board.square_m !== S.run.board.square_m || S.default_board.marker_m !== S.run.board.marker_m));
  const autoOk = autoChecks.every((check) => live[check.id]?.ok === true || (step.id === 'S2' && check.id === 'board_both_visible' && boardChanged));
  const operatorOk = step.readiness.filter((check) => check.kind === 'operator').every((check) => refs.confirm.has(check.id));
  const startable = Boolean(run) && !run.closed && run.plan.includes(step.id);
  let hint = '';
  if (!run) hint = pick('先在「标定首页」开始一次标定。', 'Start a calibration run on Calibration Home first.');
  else if (!run.plan.includes(step.id)) hint = pick('这一步不在本轮计划里（本轮选择的变化不影响它）；仍可手动执行，但不是必须。', 'This step is not in the current plan (unaffected by selected changes); you may still run it manually.');
  else if (ownedByOther()) hint = pick('另一个浏览器窗口正在控制。', 'Another browser window holds control.');
  else if (blockers.length) hint = blockerText(blockers).join(pick('；', '; '));
  else if (!autoOk) hint = pick('准备检查还有红色项，先处理它们。', 'Resolve failing readiness checks (red ✗) above before starting.');
  else if (!operatorOk) hint = pick('勾选确认上面的现场检查项后才能开始。', 'Check all operator confirmation boxes above before starting.');
  const allowManual = Boolean(run) && !run.closed;
  const blocked = !allowManual || ownedByOther() || blockers.length > 0 || busy || (step.id === 'S3' ? capturing || !autoOk : !autoOk) || !operatorOk;
  refs.start.textContent = step.id === 'S3'
    ? (status === 'passed' || status === 'failed' ? pick('重新采集', 'Recapture') : pick('开始采集', 'Start Capture'))
    : status === 'not_started'
      ? (step.id === 'S6' ? pick('开始验证触碰', 'Start Verification Touches') : pick('开始', 'Start'))
      : pick('重新执行', 'Re-run');
  refs.start.disabled = blocked || (step.id === 'S3' && running && capturing);
  refs.stop.disabled = !capturing || ownedByOther();
  refs.solve.disabled = ownedByOther() || busy || !(step.id === 'S3' && S.handheld?.ready_to_solve && (running || status === 'failed'));
  refs.cancel.disabled = !(S.running_step || capturing || inFlight);
  refs.hint.textContent = hint || (startable ? (step.motion
    ? pick('点击开始：机械臂会自动运动，期间动作准入关闭，完成后自动重新开放。随时可以按 E-STOP。', 'Click Start: the arm will move automatically while action admission stays closed, reopening upon completion. E-STOP is available at any time.')
    : pick('点击开始（机械臂不动）。', 'Click Start (arm remains stationary).')) : '');
  const progress = (record?.progress || []).slice(-1)[0];
  refs.progressText.textContent = running || inFlight?.step === step.id
    ? (progress ? phaseText(progress) : pick('已发送，等待工作进程…', 'Sent, waiting for worker…'))
    : progress && status !== 'not_started'
      ? `${pick('最后阶段：', 'Last phase: ')}${phaseText(progress)}`
      : pick('尚未运行', 'Not started yet');
  const fraction = phaseFraction(progress);
  refs.progressBar.value = fraction ?? (running ? 0 : status === 'passed' ? 1 : 0);
  if (step.id === 'S3') renderHandheld(refs);
  refs.result.replaceChildren(...(record && status !== 'not_started' && status !== 'running' ? [resultCard(record, definition, step.id)] : []));
}

function renderHandheld(refs) {
  const capture = S.handheld;
  const coverage = capture?.coverage;
  refs.hhGrid.replaceChildren(...[0, 1, 2].map((row) => el('tr', {}, ...[0, 1, 2].map((col) => {
    const count = coverage?.grid?.[row]?.[col] ?? 0;
    return el('td', { className: count > 0 ? 'done' : 'missing', textContent: String(count) });
  }))));
  refs.hhBuckets.textContent = coverage
    ? pick(
      `距离（远/中/近） ${['far', 'mid', 'near'].map((key) => coverage.distance[key]).join(' / ')}\n倾斜（正对/小/大） ${['frontal', 'small', 'large'].map((key) => coverage.tilt[key]).join(' / ')}`,
      `Distance (far/mid/near) ${['far', 'mid', 'near'].map((key) => coverage.distance[key]).join(' / ')}\nTilt (frontal/small/large) ${['frontal', 'small', 'large'].map((key) => coverage.tilt[key]).join(' / ')}`,
    )
    : '—';
  refs.hhBar.value = capture?.target_views ? Math.min(1, capture.kept / capture.target_views) : 0;
  const last = capture?.last;
  refs.hhText.textContent = capture
    ? pick(
      `${capture.status} · 已保留 ${capture.kept}/${capture.target_views}（至少 ${capture.min_views} 个才能求解）· 帧 ${capture.frames}${capture.dropped_wrong_size ? ` · 丢弃尺寸不符 ${capture.dropped_wrong_size}` : ''}${last?.corners != null ? ` · 角点 ${last.corners}` : ''}${capture.reason ? ` · ${capture.reason}` : ''}`,
      `${capture.status} · kept ${capture.kept}/${capture.target_views} (min ${capture.min_views} to solve) · frames ${capture.frames}${capture.dropped_wrong_size ? ` · dropped wrong size ${capture.dropped_wrong_size}` : ''}${last?.corners != null ? ` · corners ${last.corners}` : ''}${capture.reason ? ` · ${capture.reason}` : ''}`,
    )
    : pick('尚未开始采集', 'Capture not started');
  refs.hhGuidance.replaceChildren(...(capture?.guidance || []).map((hint) => el('li', {
    className: capture.ready_to_solve ? 'ok' : '',
    textContent: isEn() ? translateGuidanceEn(hint) : hint,
  })));
}

// ---------- 顶层渲染与语言切换响应 ----------
function render() {
  applyBadge($('calib-badge'), summaryFromState(S));
  renderLease();
  renderBanners();
  renderStepper();
  if (!S?.definition) {
    mounted = null;
    $('content').replaceChildren(el('p', {
      className: 'meta',
      textContent: S
        ? pick(`标定向导暂时无法使用：${describeReason(S.reason)}`, `Calibration wizard unavailable: ${describeReason(S.reason)}`)
        : pick('正在读取标定状态…', 'Loading calibration state…'),
    }));
    return;
  }
  if (view === 'step' && !selected) selected = currentStep(S.run) || S.definition.steps[0].id;
  if (view === 'step') renderStep(); else renderStart();
}

onLocaleChange(() => {
  const prevReviewed = mounted?.reviewed;
  mounted = null;
  render();
  if (prevReviewed && mounted) { mounted.reviewed = true; renderStep(); }
});

$('home-link').onclick = () => { view = 'start'; render(); };
$('claim').onclick = async () => { await ensureLease(); await refresh(); };
$('release').onclick = async () => {
  try { await send('/v1/control', { command: 'release_input' }); } catch (error) { setMessage('warn', pick(`释放失败 · ${error.message}`, `Release failed · ${error.message}`)); }
  leaseClient.clearToken(); await refresh();
};
$('estop').onclick = () => act('E-STOP', '/v1/control', { command: 'estop' }, { lease: false });
collapseHeaderToEstop(document.querySelector('header.bar'), $('estop'));
setInterval(async () => {
  if (!leaseClient.token || renewing) return;
  renewing = true;
  try { await send('/v1/control', { command: 'renew_input' }); } catch (error) { leaseClient.clearToken(); } finally { renewing = false; }
}, 10000);
window.addEventListener('pagehide', () => {
  if (!leaseClient.token) return;
  fetch('/v1/control', { method: 'POST', keepalive: true, headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ command: 'release_input', request_id: crypto.randomUUID(), client_id: clientId, writer_token: leaseClient.token }) }).catch(() => {});
});

loop();
