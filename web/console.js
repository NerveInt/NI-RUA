// 本地真机操作员工作台（/console）：纯原生 JS，复用网关 /v1/control、控制租约、SSE 事件与正式观测记录。
import './design-tokens.css';
import { mountNativeTui } from './console-tui.js';
import { backendName, agentStatus, terminalAbsent, applyAgentNames } from './phase-model.js';
import './workspace-navigation.css';
import { mountPerceptionView } from './perception-view.js';
import { mountCalibrationBadge } from './calibration-badge.js';
import { describeReason } from './calibration-model.js';
import { newRunCommand } from './new-run-command.js';
import { mountAgentSettings } from './agent-settings-view.js';
import { isExternalBackend } from './agent-settings-model.js';
import './agent-settings.css';
import { collapseHeaderToEstop } from './sticky-header.js';
import { byId as $, el, replaceIfChanged, setPill, setText } from './shared/dom.js';
import { requestJSON } from './shared/api-client.js';
import { createLeaseClient } from './shared/lease-client.js';
import { getLocale, initLocale, localeTag, mountLanguageSwitch, onLocaleChange, pick, registerPageI18n } from './shared/i18n.js';
import { CONSOLE_I18N } from './i18n/console.js';

initLocale();

const getJSON = (path, options) => requestJSON(path, options, { cache: 'no-store' });
const leaseClient = createLeaseClient({
  requestJSON: getJSON,
  onClaimError: (error) => {
    const code = /^[a-z_]+(?=:|$)/.exec(String(error.message))?.[0];
    setStatus(pick(
      `未能取得控制 · ${describeReason(error.message)}${code ? `（${code}）` : ''}`,
      `Failed to claim control · ${describeReason(error.message)}${code ? ` (${code})` : ''}`,
    ));
  },
});
const clientId = leaseClient.clientId;
let run, consoleState, telemetry, lastObservation = null, lastItems = null, eventCursor = 0, events;
const perceptionView = mountPerceptionView({ getJSON });
registerPageI18n(CONSOLE_I18N, document);
mountLanguageSwitch(document.querySelector('.head-state'));
// Agent 栏：backend · profile · model · effort · speed。
const agentBar = mountAgentSettings($('agent-bar'), {
  getJSON, control, ensureLease: leaseClient.ensureLease,
  lease: () => { const owner = run?.controller?.owner_client_id; return { owned: Boolean(leaseClient.token && owner === clientId), other: Boolean(owner && owner !== clientId) }; },
  credentials: () => ({ client_id: clientId, writer_token: leaseClient.token }),
  refreshConsole: () => refreshRun(),
});
let perceptionObservationId, observationRefreshVersion = 0, videoSaving = false, videoRendered;

// 每次逻辑尝试复用同一 request_id，确保未确认响应重试时幂等重放而非重发物理动作。
function control(command, fields = {}, { lease = true, key = command } = {}) {
  return leaseClient.sendIdempotent('/v1/control', { command, ...fields }, { lease, key });
}

const fmt = (value, digits = 3) => (typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : value == null ? '—' : String(value));
const fmtList = (value, digits = 3) => (Array.isArray(value) ? `[${value.map((item) => fmt(item, digits)).join(', ')}]` : '—');
function setStatus(text) { $('control-status').textContent = `${new Date().toLocaleTimeString(localeTag(), { hour12: false })} · ${text}`; }

// ---------- 实时遥测（仅操作员可见） ----------
async function refreshTelemetry() {
  try { telemetry = await getJSON('/v1/console/telemetry'); }
  catch (error) { telemetry = { available: false, stale: true, reason: `telemetry_request_failed: ${error.message}` }; }
  renderTelemetry();
  renderBanners();
}

function renderTelemetry() {
  if (!telemetry) return;
  const state = telemetry?.state;
  const stale = !telemetry?.available || telemetry.stale;
  setText($('telemetry-meta'), telemetry?.available
    ? `seq ${state.seq} · captured_at ${state.captured_at} · age ${fmt(telemetry.age_s, 1)} s · ${stale ? `STALE (${telemetry.reason})` : 'live'} · connected ${state.connected}`
    : pick(`telemetry 不可用 · ${telemetry?.reason || '未知'}`, `telemetry unavailable · ${telemetry?.reason || 'unknown'}`));
  for (const figure of document.querySelectorAll('.cameras figure')) {
    const id = figure.dataset.camera;
    const health = state?.cameras?.[id];
    const cameraStale = stale || !health?.ok;
    figure.classList.toggle('stale', cameraStale);
    setText(figure.querySelector('.cam-meta'), health
      ? `${health.ok ? 'ok' : 'FAIL'} · fps ${fmt(health.fps, 1)} · last_frame_at ${health.last_frame_at || '—'}${health.error ? ` · ${health.error}` : ''}`
      : pick('无相机健康数据', 'No camera health data'));
    // 仅在遥测新鲜时拉取新帧，否则将上一帧置灰并标注 STALE。
    if (!stale && health?.ok) {
      const image = figure.querySelector('img');
      const next = new Image();
      next.onload = () => { image.src = next.src; };
      next.src = `/v1/console/cameras/${encodeURIComponent(id)}.jpg?t=${Date.now()}`;
    }
  }
  const torque = state?.torque_enabled;
  setPill($('torque'), `torque ${stale ? 'stale' : torque === true ? 'on' : torque === false ? 'OFF' : 'unknown'}`, stale || torque == null ? 'unknown' : torque ? 'ok' : 'warn');
  const phase = state?.motion?.phase;
  setPill($('phase'), `phase ${stale ? 'stale' : phase || 'unknown'}`, stale || !phase ? 'unknown' : phase === 'fault' ? 'bad' : phase === 'idle' || phase === 'holding' ? 'ok' : 'warn');
  const show = (value) => (stale ? `${value} (stale)` : value);
  setText($('tcp-pos'), show(fmtList(state?.tcp_position_m, 4)));
  setText($('tcp-quat'), show(fmtList(state?.tcp_quaternion_xyzw, 4)));
  setText($('gripper'), show(fmt(state?.gripper_normalized, 3)));
  setText($('active-action'), show(state?.motion?.active_action_id || '—'));
  setText($('tracking'), show(fmt(state?.motion?.tracking_error_rad, 4)));
  renderJoints(state, telemetry?.joint_limits_rad || consoleState?.joint_limits_rad, stale);
  // 电机行按关节名复用节点，签名含当前语言以便仅在空状态切换语言时刷新文案，常规轮询零重建。
  const motors = Object.entries(state?.motor || {});
  const body = $('motors').tBodies[0];
  const motorSig = motors.length ? JSON.stringify(motors.map(([name]) => name)) : `empty:${getLocale()}:${Boolean(telemetry?.available)}`;
  replaceIfChanged(body, motorSig, () => (motors.length ? motors.map(([name]) => el('tr', {},
    el('td', { className: 'mono', textContent: name }), el('td'), el('td'), el('td')))
    : [el('tr', { className: 'is-empty' }, Object.assign(el('td', {
      textContent: telemetry?.available
        ? pick('没有电机数据', 'No motor data')
        : pick('遥测不可用，暂无电机数据', 'Telemetry unavailable; no motor data'),
    }), { colSpan: 4 }))]));
  motors.forEach(([, motor], index) => {
    const cells = body.rows[index]?.cells;
    if (!cells || cells.length < 4) return;
    setText(cells[1], fmt(motor?.temperature_c, 1)); setText(cells[2], fmt(motor?.load, 2)); setText(cells[3], fmt(motor?.voltage_v, 2));
  });
  const faults = state?.faults || [];
  replaceIfChanged($('faults'), JSON.stringify(faults), () => faults.map((fault) => el('div', { className: 'fault', textContent: `fault · ${fault}` })));
}

// 关节行按关节集合构建一次，轮询只更新游标位置与数值。
function renderJoints(state, limits, stale) {
  const names = Object.keys(state?.q_rad || limits || {});
  const container = $('joints');
  if (replaceIfChanged(container, JSON.stringify(names), () => names.map((name) => el('div', { className: 'joint' },
    el('span', { className: 'mono', textContent: name }), el('div', { className: 'track' }, el('span', { className: 'measured' }), el('span', { className: 'goal' })),
    el('span', { className: 'mono val' }))))) container.removeAttribute('aria-busy');
  names.forEach((name, index) => {
    const row = container.children[index];
    if (!row) return;
    const q = state?.q_rad?.[name];
    const goal = state?.q_goal_rad?.[name];
    const [low, high] = limits?.[name] || [];
    const known = Number.isFinite(low) && Number.isFinite(high) && high > low;
    const position = (value) => `${Math.min(100, Math.max(0, ((value - low) / (high - low)) * 100))}%`;
    const track = row.children[1];
    const title = limits?.[name] ? `limit [${fmt(low)}, ${fmt(high)}] rad` : pick('关节限位未知', 'Joint limits unknown');
    if (track.title !== title) track.title = title;
    const [measured, target] = track.children;
    measured.hidden = !(known && Number.isFinite(q));
    target.hidden = !(known && Number.isFinite(goal));
    if (!measured.hidden && measured.style.left !== position(q)) measured.style.left = position(q);
    if (!target.hidden && target.style.left !== position(goal)) target.style.left = position(goal);
    const near = known && Number.isFinite(q) && Math.min(q - low, high - q) < 0.1 * (high - low);
    row.classList.toggle('near', near);
    setText(row.children[2], `q ${fmt(q)} · goal ${fmt(goal)} rad${stale ? ' (stale)' : ''}`);
  });
}

// ---------- 运行与 Agent 状态 ----------
function renderRunView() {
  if (!run || !consoleState) return;
  setText($('run-id'), `run ${run.run_id}`);
  applyAgentNames(document, backendName(run));
  document.querySelector('.native-tui').hidden = terminalAbsent(run);
  const turn = agentStatus(run) || 'unknown';
  setPill($('turn'), `turn ${turn}`, turn === 'running' ? 'warn' : ['unknown', 'disconnected', 'failed'].includes(turn) ? 'unknown' : 'ok');
  setPill($('admission'), `admission ${run.accepting_actions ? 'open' : 'closed'}`, run.accepting_actions ? 'ok' : 'warn');
  setText($('run-status'), `${run.status} · motion ${run.motion_status} · active_action ${run.active_action || '—'} · actions ${run.action_count}`);
  setText($('thread'), `${run.thread_id || '—'} / ${run.turn_id || '—'}`);
  const tokens = run.token_accounting?.tokens;
  setText($('tokens'), tokens ? Object.entries(tokens).filter(([, value]) => value != null).map(([key, value]) => `${key} ${value}`).join(' · ') : pick('未提供', 'Not reported'));
  const plan = consoleState.task_plan;
  setText($('plan'), plan ? `revision ${plan.revision}\n${JSON.stringify(plan.plan, null, 2)}${plan.attention?.length ? `\nattention: ${JSON.stringify(plan.attention)}` : ''}` : pick('尚无 task_plan', 'No task_plan yet'));
  renderReceipts(consoleState.recent_actions || []);
  renderSettings();
  renderPending(run.pending_requests || []);
  if (!videoSaving) renderVideoExport(consoleState.video_export);
  renderAck();
  renderLease();
  renderBanners();
}

async function refreshRun() {
  const startedAt = Date.now();
  const previousRunId = run?.run_id;
  try {
    [run, consoleState] = await Promise.all([getJSON('/v1/run'), getJSON('/v1/console/state')]);
  } catch (error) { setStatus(pick(`读取运行状态失败 · ${error.message}`, `Failed to load run state · ${error.message}`)); return; }
  if (previousRunId && run.run_id !== previousRunId) resetForNewRun();
  if (leaseClient.token && startedAt > leaseClient.claimedAt && run.controller?.owner_client_id !== clientId) leaseClient.clearToken();
  if (!run.control_busy) leaseClient.claimRefusedBusy = false;
  if (!leaseClient.token && !run.controller?.owner_client_id && !leaseClient.claimRefusedBusy) leaseClient.ensureLease({ quiet: true });
  renderRunView();
  agentBar.update({ run, consoleState, startedAt });
}

function resetForNewRun() {
  eventCursor = 0;
  events?.close();
  connectEvents();
  $('items').replaceChildren();
  for (const id of ['items', 'pending']) delete $(id).dataset.signature;
  delete $('receipts').tBodies[0].dataset.signature;
  delete $('obs-images').dataset.id;
  perceptionObservationId = undefined;
  videoRendered = undefined;
  lastObservation = null;
  lastItems = null;
  refreshItems();
  refreshObservation();
}

const RECEIPT_COLUMNS = ['action_id', 'type', 'status', 'reason', 'residuals'];
function renderReceipts(actions) {
  const cell = (index, props, ...children) => { const node = el('td', props, ...children); node.dataset.label = RECEIPT_COLUMNS[index]; return node; };
  const sig = actions.length ? JSON.stringify(actions) : `empty:${getLocale()}`;
  replaceIfChanged($('receipts').tBodies[0], sig, () => (actions.length ? actions.map((action) => {
    const residuals = Object.entries(action).filter(([key]) => key.includes('residual') || key.endsWith('_error_m') || key.endsWith('_error_rad'));
    return el('tr', {}, cell(0, { className: 'mono' }, el('a', { href: `/v1/actions/${encodeURIComponent(action.action_id)}`, target: '_blank', rel: 'noopener', textContent: action.action_id })),
      cell(1, { className: 'mono', textContent: action.type || '—' }),
      cell(2, { className: `status-${action.status}`, textContent: action.status || 'unknown' }),
      cell(3, { textContent: action.reason || '' }),
      cell(4, { className: 'mono', textContent: residuals.length ? residuals.map(([key, value]) => `${key}=${typeof value === 'object' ? JSON.stringify(value) : fmt(value, 4)}`).join(' ') : '—' }));
  }) : [el('tr', { className: 'is-empty' }, Object.assign(el('td', { textContent: pick('尚无动作回执', 'No action receipts yet') }), { colSpan: 5 }))]));
}

// 速度上限、新建 run 要素与生效配置。
function renderSettings() {
  const runtime = consoleState.settings.runtime;
  const fresh = consoleState.settings.new_run;
  const currentModel = runtime.model?.value || fresh.model || '';
  const speedCap = runtime.speed_scale_cap?.value ?? runtime.speed_scale_cap?.default ?? 0.9;
  if (document.activeElement !== $('speed')) $('speed').value = speedCap;
  const facts = [['model', currentModel || fresh.model], ['observation_profile', fresh.observation_profile],
    ['backend.environment', fresh.environment], ['backend.robot_model_id', fresh.robot_model_id]];
  replaceIfChanged($('new-run'), JSON.stringify(facts), () => facts.flatMap(([key, value]) => [el('dt', { textContent: key }), el('dd', { className: 'mono', textContent: value ?? '—' })]));
  setText($('new-run-cmd'), newRunCommand({ model: currentModel || fresh.model, observationProfile: fresh.observation_profile,
    environment: fresh.environment, category: consoleState.effective_config?.run?.category }));
  setText($('config'), JSON.stringify(consoleState.effective_config, null, 2));
}

// 最近一次保存任务视频回执（仅操作员可见）。
function renderVideoExport(state) {
  const renderKey = state ? `${getLocale()}:${state.request_id}` : null;
  if (!state || videoRendered === renderKey) return;
  videoRendered = renderKey;
  const composite = state.composite || {};
  const time = state.saved_at ? `${new Date(state.saved_at).toLocaleTimeString(localeTag(), { hour12: false })} · ` : '';
  const rearm = state.recording_rearmed === false
    ? pick(` · 注意：后续任务的录像没能重新开启（${state.rearm_error || '原因未知'}）`, ` · Warning: recording could not be re-armed for subsequent tasks (${state.rearm_error || 'unknown reason'})`)
    : '';
  const parts = [];
  if (state.status === 'complete' && composite.video_ref?.startsWith('artifact:')) {
    const url = `/v1/artifacts/${encodeURIComponent(composite.video_ref.slice('artifact:'.length))}`;
    const name = `${run?.run_id || 'run'}-${state.request_id.slice(0, 8)}-combined.mp4`;
    parts.push(
      pick(
        `${time}已保存合成视频 · ${(composite.camera_order || []).join(' + ')} · ${composite.width}×${composite.height} · ${composite.frames} 帧 · ${fmt(composite.duration_s, 1)} s · ${composite.segments} 个动作 · `,
        `${time}Saved composite video · ${(composite.camera_order || []).join(' + ')} · ${composite.width}×${composite.height} · ${composite.frames} frames · ${fmt(composite.duration_s, 1)} s · ${composite.segments} actions · `,
      ),
      el('a', { href: url, download: name, textContent: pick('下载 MP4', 'Download MP4') }), ' · ',
      el('a', { href: url, target: '_blank', rel: 'noopener', textContent: pick('新标签页播放', 'Play in new tab') }),
      el('br'), el('span', { className: 'mono', textContent: composite.video_path || '' }),
    );
    if (composite.missing_cameras?.length) {
      parts.push(el('br'), pick(
        `缺少画面的相机：${composite.missing_cameras.map((camera) => `${camera.camera_id} (${camera.status})`).join(', ')}`,
        `Cameras missing frames: ${composite.missing_cameras.map((camera) => `${camera.camera_id} (${camera.status})`).join(', ')}`,
      ));
    }
  } else if (state.status === 'empty') {
    parts.push(pick(
      `${time}没有可保存的动作录像：上次保存之后还没有执行过动作。`,
      `${time}No action recordings to save: no actions have executed since the last save.`,
    ));
  } else {
    parts.push(pick(
      `${time}保存失败 · ${state.error || composite.error || '原因未知'}${state.recording?.directory ? ` · 单相机录像目录 ${state.recording.directory}` : ''}`,
      `${time}Save failed · ${state.error || composite.error || 'unknown reason'}${state.recording?.directory ? ` · per-camera directory ${state.recording.directory}` : ''}`,
    ));
  }
  if (rearm) parts.push(rearm);
  $('video-status').replaceChildren(...parts);
}

$('save-video').onclick = async () => {
  if (!(await leaseClient.ensureLease())) { $('video-status').textContent = pick('另一个浏览器窗口正在控制，未保存', 'Another browser window holds control; video not saved'); return; }
  videoSaving = true; renderLease();
  $('video-status').textContent = pick(
    '正在封存录像并拼接多路画面…录像越长耗时越久（约为视频时长的 1/10）',
    'Sealing recordings and stitching camera streams… longer clips take proportionally longer (~1/10 of video duration)',
  );
  try { videoRendered = undefined; renderVideoExport(await control('save_video')); }
  catch (error) {
    $('video-status').textContent = pick(
      `保存未完成 · ${error.message === 'observer_operation_requires_idle_run' ? `需要 ${backendName(run)} 空闲且没有动作在执行` : error.message}`,
      `Save incomplete · ${error.message === 'observer_operation_requires_idle_run' ? `requires ${backendName(run)} idle with no active action` : error.message}`,
    );
  }
  videoSaving = false;
  await refreshRun();
};

function renderAck() {
  const required = consoleState?.resume?.requires_acknowledgement || [];
  $('ack').hidden = required.length === 0;
  const items = consoleState?.operator_gate?.unknown || [];
  $('ack-list').replaceChildren(...items.map((item) => el('li', { className: 'mono', textContent: `${item.id} · ${item.source} · ${item.reason || ''}` })));
  if (!required.length) $('ack-check').checked = false;
}

function transportReason(transport) {
  return [transport.http_status ? `HTTP ${transport.http_status}` : null, transport.classification, transport.code || transport.type,
    transport.message ? `“${transport.message}”` : null, transport.error_type].filter(Boolean).join(' · ') || 'unknown';
}

function renderBanners() {
  const banners = [];
  const unknown = consoleState?.operator_gate?.unknown || [];
  if (unknown.length || run?.status === 'unknown') {
    const ids = unknown.length ? unknown.map((item) => item.id).join(', ') : 'run status unknown';
    banners.push(['gate-unknown', 'unknown', pick(
      `UNKNOWN：${ids} — 机械臂是否停止/动作是否执行无法确认。请到现场检查，确认后在「运行控制」中勾选并 resume（会先获取新观察）。`,
      `UNKNOWN: ${ids} — whether the arm stopped or executed the action is unconfirmed. Inspect the robot in person, then check the box under Run Controls and click Resume (captures a fresh observation first).`,
    )]);
  }
  if (consoleState?.operator_gate?.torque === 'off') {
    banners.push(['torque-off', 'stale', pick(
      'torque OFF：力矩已关闭，机械臂可自由移动。需要 torque_on + resume 才能再次运动。',
      'torque OFF: servo torque is disabled and the arm can move freely. Run torque_on + resume before moving again.',
    )]);
  }
  if (consoleState?.operator_gate?.torque === 'unknown') {
    banners.push(['torque-unknown', 'unknown', pick(
      'torque 状态 unknown：最近一次力矩命令未确认。',
      'torque state unknown: the last torque command was not confirmed.',
    )]);
  }
  if (telemetry && (!telemetry.available || telemetry.stale)) {
    banners.push(['telemetry', 'stale', pick(
      `遥测 STALE / 不可用（${telemetry.reason}${telemetry.age_s != null ? `，age ${fmt(telemetry.age_s, 1)} s` : ''}）：画面与关节数值不是实时的。`,
      `Telemetry STALE / unavailable (${telemetry.reason}${telemetry.age_s != null ? `, age ${fmt(telemetry.age_s, 1)} s` : ''}): camera frames and joint values are not live.`,
    )]);
  }
  if (consoleState && !consoleState.operator_supported) {
    banners.push(['operator', 'info', pick(
      '当前后端不支持 operator 命令（hold / relax / torque_on / go_home / set_runtime_limits）。stop 仍可用；请准备物理断电开关。',
      'Current robot backend does not support operator commands (hold / relax / torque_on / go_home / set_runtime_limits). stop remains available; keep a physical power switch ready.',
    )]);
  }
  const transport = consoleState?.model_transport;
  if (transport?.blocked) {
    banners.push(['transport', 'unknown', pick(
      `模型请求已被中继阻断：${transportReason(transport)}。${transport.manual_retry_allowed ? '可先在 Agent 栏换一个能处理该请求的模型，再点「设置」里的「重试模型请求」（不会自动重试）。' : '该错误不允许手动重试，需要新建 run。'}`,
      `Model request blocked by relay: ${transportReason(transport)}. ${transport.manual_retry_allowed ? 'Select a model capable of handling this request in the Agent bar, then click "Retry model request" under Settings (never retried automatically).' : 'Manual retry is not permitted for this error; start a new run.'}`,
    )]);
  }
  const modelSetting = consoleState?.settings?.runtime?.model;
  if (modelSetting?.image_input === false) {
    banners.push(['image-input', 'stale', pick(
      `当前模型 ${modelSetting.value} 在供应商目录中标注为不支持图像输入：带图像的观察会被供应商拒绝（例如 HTTP 400 IMAGE_INPUT_UNSUPPORTED）。可在 Agent 栏换成支持图像的模型。`,
      `Current model ${modelSetting.value} is marked in the provider catalogue as lacking image input: observations with images will be rejected (e.g. HTTP 400 IMAGE_INPUT_UNSUPPORTED). Switch to a vision-capable model in the Agent bar.`,
    )]);
  }
  if (run && !run.accepting_actions && !unknown.length) {
    banners.push(['admission', 'info', pick(
      `动作准入已关闭${consoleState?.resume?.blockers?.length ? `（resume 前需：${consoleState.resume.blockers.join(', ')}）` : '，可点「恢复准入 resume」'}。`,
      `Action admission is closed${consoleState?.resume?.blockers?.length ? ` (required before resume: ${consoleState.resume.blockers.join(', ')})` : '; click "Resume" to reopen'}.`,
    )]);
  }
  // 按固定类别键就地更新横幅文本，避免 200 ms 遥测轮询引起节点重建或动画闪烁。
  const container = $('banners');
  const existing = new Map([...container.children].map((node) => [node.dataset.key, node]));
  const nodes = banners.map(([key, kind, text]) => {
    const node = existing.get(key) || el('div');
    node.dataset.key = key;
    const name = `banner ${kind}`;
    if (node.className !== name) node.className = name;
    setText(node, text);
    return node;
  });
  if (nodes.length !== container.children.length || nodes.some((node, index) => container.children[index] !== node)) container.replaceChildren(...nodes);
}

function renderLease() {
  const owner = run?.controller?.owner_client_id;
  const owned = Boolean(leaseClient.token && owner === clientId);
  setText($('lease-state'), owned
    ? pick('本窗口持有控制', 'Control held')
    : owner
      ? pick('另一窗口持有控制 · 只读', 'Other window · read-only')
      : pick('监看模式', 'Monitor only'));
  $('claim').hidden = owned; $('claim').disabled = Boolean(owner && owner !== clientId); $('release').hidden = !owned;
  for (const button of document.querySelectorAll('[data-lease]')) button.disabled = Boolean(owner && owner !== clientId);
  const transport = consoleState?.model_transport;
  $('model-transport-row').hidden = !transport?.blocked;
  if (transport?.blocked) {
    setText($('model-transport-status'), pick(`模型请求被阻断：${transportReason(transport)}`, `Model request blocked: ${transportReason(transport)}`));
    $('retry-model').disabled = Boolean(owner && owner !== clientId) || !transport.manual_retry_allowed;
  }
  const idle = agentStatus(run) !== 'running' && !run?.active_action;
  const external = isExternalBackend(run, consoleState);
  $('prompt-form').hidden = external;
  $('external-task-note').hidden = !external;
  const steering = agentStatus(run) === 'running';
  setText($('send'), steering ? pick('插话（加入当前轮次 steer）', 'Steer (inject into active turn)') : pick('发送新任务', 'Send Task'));
  const otherWindow = Boolean(owner && owner !== clientId);
  $('send').disabled = otherWindow || (!steering && (!run?.accepting_actions || !idle));
  setText($('prompt-status'), otherWindow
    ? pick('另一个浏览器窗口正在控制这台机械臂', 'Another browser window holds control of this arm')
    : steering
      ? pick(
        `${backendName(run)} 正在运行：内容会立刻加入当前轮次，模型下一步就会看到；紧急情况请用 E-STOP / hold`,
        `${backendName(run)} is running: your message is injected into the active turn immediately; use E-STOP / hold for emergencies`,
      )
      : pick(
        `${backendName(run)} 空闲：内容作为新任务发送到当前 thread`,
        `${backendName(run)} is idle: sends as a new task to the current thread`,
      ));
  $('go-home').disabled = !owned || !idle;
  $('save-video').disabled = otherWindow || !idle || videoSaving;
  const blockers = consoleState?.resume?.blockers || [];
  const needsAck = (consoleState?.resume?.requires_acknowledgement || []).length > 0;
  $('resume').disabled = !owned || blockers.length > 0 || (needsAck && !$('ack-check').checked);
  $('torque-on').disabled = !owned || !consoleState?.operator_supported;
  $('relax').disabled = !owned || !consoleState?.operator_supported;
  $('hold').disabled = !consoleState?.operator_supported;
}

// ---------- 模型可见观察（与实时预览严格区分） ----------
function renderObservationView(observation) {
  $('camera-heading').textContent = pick('感知画面 · 正式观察快照', 'Perception View · Formal Observation Snapshot');
  if (!observation || !observation.observation_id) {
    $('obs-meta').textContent = pick('尚无模型可见观察', 'No model-visible observation yet');
    $('obs-images').replaceChildren();
    return;
  }
  const obsDomKey = `${getLocale()}:${observation.observation_id}`;
  if ($('obs-images').dataset.id === obsDomKey) return;
  $('obs-images').dataset.id = obsDomKey;
  $('obs-meta').textContent = pick(
    `observation_id ${observation.observation_id} · ${observation.step_domain || ''} ${observation.sim_step ?? ''} · 这是模型看到的画面，不是实时预览`,
    `observation_id ${observation.observation_id} · ${observation.step_domain || ''} ${observation.sim_step ?? ''} · exact frame delivered to the model (not live preview)`,
  );
  $('obs-images').replaceChildren(...(observation.cameras || []).filter((camera) => camera.rgb_ref?.startsWith('artifact:')).map((camera) =>
    el('figure', {}, el('img', { src: `/v1/artifacts/${encodeURIComponent(camera.rgb_ref.slice(9))}?view=image`, alt: pick(`模型可见 ${camera.camera_id}`, `Model-visible ${camera.camera_id}`) }),
      el('figcaption', { className: 'mono', textContent: camera.camera_id }))));
}

async function refreshObservation() {
  const version = ++observationRefreshVersion;
  let observation;
  try { observation = await getJSON('/v1/observations/latest'); } catch { observation = null; }
  if (version !== observationRefreshVersion) return;
  lastObservation = observation;
  if (perceptionObservationId !== observation?.observation_id) {
    const previous = $('camera-select').value;
    $('camera-select').replaceChildren(...(observation?.cameras || []).map((camera) => el('option', { value: camera.camera_id, textContent: camera.camera_id })));
    if (observation?.cameras?.some((camera) => camera.camera_id === previous)) $('camera-select').value = previous;
    else if (observation?.cameras?.some((camera) => camera.camera_id === 'fixed')) $('camera-select').value = 'fixed';
    perceptionObservationId = observation?.observation_id;
  }
  perceptionView.syncObservationPerception(observation);
  perceptionView.showCamera();
  renderObservationView(observation);
}

// ---------- 模型消息与工具调用 ----------
function describeItem(item) {
  if (item.type === 'agentMessage') return item.text || '';
  if (item.type === 'dynamicToolCall' || item.type === 'mcpToolCall') return `${item.tool || item.name || ''} ${JSON.stringify(item.arguments || {}).slice(0, 300)} → ${item.result?.status || item.status || ''}`;
  if (item.type === 'commandExecution') return `${item.command || ''} → ${item.exitCode ?? item.status ?? ''}`;
  if (item.type === 'reasoning') return (item.summary || []).join(' ');
  if (item.type === 'error') return item.error?.message || 'error';
  return item.status || '';
}
async function refreshItems() {
  try {
    const value = await getJSON('/v1/items');
    eventCursor = Math.max(eventCursor, value.cursor || 0);
    lastItems = (value.items || []).slice(-40).reverse();
    const rows = lastItems.map((item) => [`${item.type} · ${item.status || ''}`, describeItem(item)]);
    replaceIfChanged($('items'), JSON.stringify(rows), () => rows.map(([kind, text]) => el('li', {}, el('span', { className: 'kind', textContent: kind }), document.createTextNode(text))));
  } catch (error) {
    delete $('items').dataset.signature;
    $('items').replaceChildren(el('li', { textContent: pick(`读取记录失败 · ${error.message}`, `Failed to load items · ${error.message}`) }));
  }
}

function renderPending(requests) {
  const signature = `${getLocale()}:${JSON.stringify(requests.map((request) => request.request_id))}`;
  if ($('pending').dataset.signature === signature) return;
  $('pending').dataset.signature = signature;
  $('pending').replaceChildren(...requests.map((request) => {
    const kindLabel = request.method?.includes('requestApproval')
      ? pick('原生审批', 'Native Approval')
      : pick('原生提问', 'Native Prompt');
    const section = el('section', { className: 'pending-request' }, el('h3', { textContent: `${kindLabel} · ${request.request_id}` }),
      el('pre', { className: 'mono', textContent: JSON.stringify(request.params, null, 2) }));
    const form = el('form');
    const questions = request.params?.questions;
    let input;
    if (Array.isArray(questions)) {
      input = questions.map((question) => { const field = el('input', { required: true, placeholder: question.question || question.id }); form.append(el('label', { textContent: question.question || question.id }), field); return [question.id, field]; });
    } else if (request.method?.includes('requestApproval') && request.method !== 'item/permissions/requestApproval') {
      input = el('select', {}, ...(request.params?.availableDecisions || ['decline', 'accept']).map((decision) => el('option', { value: JSON.stringify(decision), textContent: typeof decision === 'string' ? decision : JSON.stringify(decision) })));
      form.append(input);
    } else { input = el('textarea', { value: '{}' }); form.append(input); }
    const status = el('p', { className: 'meta' });
    form.append(el('button', { type: 'submit', textContent: pick('回复 (reply)', 'Reply (reply)'), className: 'primary' }), status);
    form.onsubmit = async (event) => {
      event.preventDefault();
      try {
        const response = Array.isArray(input) ? { answers: Object.fromEntries(input.map(([id, field]) => [id, { answers: [field.value] }])) }
          : input.tagName === 'SELECT' ? { decision: JSON.parse(input.value) } : JSON.parse(input.value);
        const result = await control('reply', { native_request_id: request.request_id, response }, { key: `reply-${request.request_id}` });
        status.textContent = pick(`已回复 · ${result.status || ''}`, `Replied · ${result.status || ''}`);
      } catch (error) { status.textContent = pick(`回复未确认 · ${error.message}`, `Reply unconfirmed · ${error.message}`); }
    };
    section.append(form);
    return section;
  }));
}

// ---------- 控制租约 ----------
const ensureLease = leaseClient.ensureLease;
$('claim').onclick = async () => { await ensureLease(); await refreshRun(); };
$('release').onclick = async () => {
  try { await control('release_input'); } catch (error) { setStatus(pick(`释放失败 · ${error.message}`, `Release failed · ${error.message}`)); }
  leaseClient.clearToken(); await refreshRun();
};
setInterval(async () => {
  if (!leaseClient.token) return;
  try { await control('renew_input', {}, { key: `renew-${Date.now()}` }); }
  catch (error) { leaseClient.clearToken(); setStatus(pick(`控制租约失效 · ${error.message}`, `Control lease expired · ${error.message}`)); renderLease(); }
}, 10000);
window.addEventListener('pagehide', () => {
  if (!leaseClient.token) return;
  fetch('/v1/control', { method: 'POST', keepalive: true, headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ command: 'release_input', request_id: crypto.randomUUID(), client_id: clientId, writer_token: leaseClient.token }) }).catch(() => {});
});

// ---------- 控制命令 ----------
async function runCommand(label, command, fields = {}, options = {}) {
  if (options.lease !== false && !(await ensureLease())) { setStatus(pick(`${label} · 另一个浏览器窗口正在控制`, `${label} · another browser window holds control`)); return; }
  setStatus(pick(`${label} · 等待后端回执…`, `${label} · waiting for backend receipt…`));
  try {
    const result = await control(command, fields, options);
    const worldNote = result.world === 'objects_not_reset' ? pick(' · 物体未复位', ' · objects not reset') : '';
    setStatus(`${label} · ${result.status}${result.admission ? ` · admission ${result.admission}` : ''}${worldNote}`);
  } catch (error) {
    const failLabel = error.status === 409 && error.body?.status === 'unknown'
      ? pick('UNKNOWN（勿重复发送，先检查机械臂）', 'UNKNOWN (do not resend; inspect arm first)')
      : pick('失败', 'failed');
    setStatus(`${label} · ${failLabel} · ${error.message}`);
  }
  await refreshRun();
}

$('estop').onclick = () => runCommand('E-STOP', 'estop', {}, { lease: false });
$('hold').onclick = () => runCommand('hold', 'hold', {}, { lease: false });
$('stop').onclick = () => runCommand('stop', 'stop', {}, { lease: false });
$('torque-on').onclick = () => runCommand('torque_on', 'torque_on');
$('go-home').onclick = () => runCommand('go_home', 'go_home');
$('relax').onclick = () => { $('relax-confirm').hidden = false; };
$('relax-no').onclick = () => { $('relax-confirm').hidden = true; };
$('relax-yes').onclick = async () => { $('relax-confirm').hidden = true; await runCommand('relax', 'relax', { confirm: 'relax' }); };
$('ack-check').onchange = renderLease;
$('resume').onclick = () => runCommand('resume', 'resume', { acknowledge_unknown: $('ack-check').checked ? consoleState?.resume?.requires_acknowledgement || [] : [] });
$('speed-form').onsubmit = (event) => { event.preventDefault(); runCommand('set_runtime_limits', 'set_runtime_limits', { speed_scale_cap: Number($('speed').value) }); };
$('retry-model').onclick = () => runCommand('retry_model_requests', 'retry_model_requests');
$('prompt-form').onsubmit = async (event) => {
  event.preventDefault();
  const prompt = $('prompt').value.trim();
  if (!prompt || isExternalBackend(run, consoleState)) return;
  $('send').disabled = true;
  if (!(await ensureLease())) { $('prompt-status').textContent = pick('另一个浏览器窗口正在控制，发送未执行', 'Another browser window holds control; prompt not sent'); await refreshRun(); return; }
  try {
    const steering = agentStatus(run) === 'running';
    const result = await control(steering ? 'steer' : 'prompt', { prompt }, { key: 'prompt' });
    $('prompt-status').textContent = steering
      ? pick(`已插话到当前轮次 · turn ${result.turn_id}`, `Injected into active turn · turn ${result.turn_id}`)
      : pick(`已发送 · thread ${result.thread_id} · turn ${result.turn_id}`, `Sent · thread ${result.thread_id} · turn ${result.turn_id}`);
    $('prompt').value = '';
  } catch (error) {
    $('prompt-status').textContent = pick(
      `发送未确认 · ${error.message}（相同内容重试沿用同一 request_id）`,
      `Send unconfirmed · ${error.message} (retrying the same text reuses request_id)`,
    );
  }
  await refreshRun();
};

// 切换语言时就地重绘所有缓存状态，不触发额外网络轮询。
onLocaleChange(() => {
  if (run) applyAgentNames(document, backendName(run));
  renderTelemetry();
  renderRunView();
  renderObservationView(lastObservation);
});

// ---------- 轮询与事件流 ----------
function connectEvents() {
  const source = new EventSource(`/v1/events?after=${eventCursor}`);
  events = source;
  let pending;
  source.onmessage = (message) => {
    try { eventCursor = Math.max(eventCursor, JSON.parse(message.data).server_seq || 0); } catch { /* 保留当前游标 */ }
    clearTimeout(pending);
    pending = setTimeout(() => { refreshItems(); refreshRun(); refreshObservation(); }, 250);
  };
  source.addEventListener('gap', () => { refreshItems(); refreshRun(); });
}

Promise.all([refreshRun(), refreshItems(), refreshObservation(), refreshTelemetry()]).finally(() => {
  delete document.body.dataset.boot;
  connectEvents();
});
setInterval(refreshTelemetry, 200);
setInterval(refreshRun, 1500);
setInterval(refreshObservation, 3000);

mountNativeTui($('tui'), $('tui-status'), {
  pageUp: $('tui-page-up'),
  pageDown: $('tui-page-down'),
  bottom: $('tui-bottom'),
  isReplay: () => run?.status === 'replay' || run?.view_mode === 'replay' || run?.read_only === true,
});
collapseHeaderToEstop(document.querySelector('header.bar'), $('estop'));
mountCalibrationBadge($('calib-badge'));
$('prompt').addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); $('prompt-form').requestSubmit(); }
});
ensureLease({ quiet: true }).then(refreshRun);
