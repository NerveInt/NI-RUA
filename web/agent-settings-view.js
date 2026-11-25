// 控制台 Agent 栏视图（设计 16 §9）：后端 · Profile · 模型 · 推理强度 · 速度、浮层/移动端抽屉、会话变更托盘、Profile 管理与切换进度遮罩。
import * as M from './agent-settings-model.js';
import { createCombobox, createListbox } from './ui-combobox.js';
import { getLocale, isEn, localeTag, onLocaleChange, pick } from './shared/i18n.js';
import { translateAgentBackendText } from './i18n/console.js';

const PHONE = '(max-width: 560px)';
const OPTIONS_TTL_MS = 15000;
const CATALOGUE_TTL_MS = 60000;
const CHECK_DEBOUNCE_MS = 220;
const POP_WIDTH = { backend: 360, profile: 400, model: 500, effort: 320, speed: 320, settings: 540 };
const COMMANDS = { model: 'set_model', effort: 'set_effort', service_tier: 'set_service_tier' };
const NS = 'http://www.w3.org/2000/svg';
const ICONS = {
  backend: 'M9 3v3M15 3v3M9 18v3M15 18v3M3 9h3M3 15h3M18 9h3M18 15h3M7.5 6h9A1.5 1.5 0 0 1 18 7.5v9a1.5 1.5 0 0 1-1.5 1.5h-9A1.5 1.5 0 0 1 6 16.5v-9A1.5 1.5 0 0 1 7.5 6zM10 10h4v4h-4z',
  profile: 'M8 11a4 4 0 1 1 0 8 4 4 0 0 1 0-8zM10.9 12.1 19 4M16 7l2.5 2.5M13.6 9.4l2 2',
  model: 'M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3zM4.3 7.7 12 12l7.7-4.3M12 12v8.6',
  effort: 'M4.6 18a8.5 8.5 0 1 1 14.8 0M12 14l4-4.5M12 14h.01',
  speed: 'M13 3 5 13.5h6L10 21l8-10.5h-6L13 3z',
  settings: 'M4 7h9M17 7h3M4 17h3M11 17h9M15 5v4M9 15v4',
  caret: 'M7 10l5 5 5-5',
  check: 'M5 12.5l4.5 4.5L19 7.5',
  close: 'M6.5 6.5l11 11M17.5 6.5l-11 11',
  refresh: 'M19.5 12a7.5 7.5 0 1 1-2.2-5.3M19.5 4.5v4.2h-4.2',
  search: 'M10.5 4a6.5 6.5 0 1 1 0 13 6.5 6.5 0 0 1 0-13zM20 20l-4.8-4.8',
  image: 'M4.5 5h15A1.5 1.5 0 0 1 21 6.5v11a1.5 1.5 0 0 1-1.5 1.5h-15A1.5 1.5 0 0 1 3 17.5v-11A1.5 1.5 0 0 1 4.5 5zM3 15.5l5-5 4.5 4.5 2.5-2.5 6 6M15.5 8.5h.01',
  tools: 'M14.7 6.3a4 4 0 0 0 5 5L21 12.6l-8.4 8.4a2 2 0 0 1-2.8-2.8L18.2 9.8M14.7 6.3 12.4 4a4 4 0 0 0-5.6 5.6l2.3 2.3',
  warning: 'M12 4 21.5 20h-19L12 4zM12 10v4.5M12 17.2h.01',
  plus: 'M12 5v14M5 12h14',
  edit: 'M4 20h4L19 9l-4-4L4 16v4zM13.5 6.5l4 4',
  trash: 'M5 7h14M10 7V5h4v2M7 7l1 13h8l1-13',
  arrow: 'M5 12h14M13 6l6 6-6 6',
  back: 'M19 12H5M11 6l-6 6 6 6',
  retry: 'M4.5 12a7.5 7.5 0 1 0 2.2-5.3M4.5 4.5v4.2h4.2',
};

function icon(name, className = 'ab-ico') {
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('class', className);
  svg.setAttribute('aria-hidden', 'true');
  svg.setAttribute('focusable', 'false');
  const path = document.createElementNS(NS, 'path');
  path.setAttribute('d', ICONS[name]);
  svg.append(path);
  return svg;
}
function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value == null || value === false) continue;
    if (key === 'class') node.className = value;
    else if (key === 'text') node.textContent = value;
    else if (key === 'dataset') Object.assign(node.dataset, value);
    else if (key.startsWith('on') && typeof value === 'function') node.addEventListener(key.slice(2), value);
    else if (key === 'value') node.value = value;
    else if (key === 'checked') node.checked = Boolean(value);
    else node.setAttribute(key, value === true ? '' : String(value));
  }
  node.append(...children.flat(Infinity).filter((child) => child != null && child !== false));
  return node;
}
const setText = (node, text) => { const value = text ?? ''; if (node.textContent !== value) node.textContent = value; };
const setAttr = (node, name, value) => {
  if (value == null || value === false) { if (node.hasAttribute(name)) node.removeAttribute(name); }
  else if (node.getAttribute(name) !== String(value)) node.setAttribute(name, String(value));
};
const focusables = (root) => [...root.querySelectorAll('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])')]
  .filter((node) => !node.disabled && !node.closest('[hidden],[inert]') && node.offsetParent !== null);
function trapTab(event, root) {
  if (event.key !== 'Tab') return;
  const nodes = focusables(root);
  if (!nodes.length) return;
  const first = nodes[0], last = nodes[nodes.length - 1];
  if (event.shiftKey && (document.activeElement === first || !root.contains(document.activeElement))) { event.preventDefault(); last.focus(); }
  else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
}
const clock = (date = new Date()) => date.toLocaleTimeString(localeTag(), { hour12: false });
const badgeNode = (badge) => h('span', { class: 'ab-badge', 'data-tone': badge.tone, title: badge.title },
  badge.kind === 'image' ? icon(badge.tone === 'warn' ? 'warning' : 'image', 'ab-ico ab-ico--s') : badge.kind === 'tools' ? icon('tools', 'ab-ico ab-ico--s') : null, badge.text);
const trServer = (text, code = null) => (isEn() ? translateAgentBackendText(text, code) : text);

export function mountAgentSettings(root, api) {
  const phone = matchMedia(PHONE);
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  const host = root.closest('.panel');
  host?.classList.add('ab-host');
  const s = {
    run: null, consoleState: null, snapshotAt: 0,
    raw: null, rawStatus: null, rawError: null, rawAt: 0,
    draft: null, intents: {}, catalogues: new Map(), check: { key: null, state: 'idle', result: null },
    open: null, chip: null, query: '', notice: null,
    settingsView: 'list', form: null, formMode: null, formBusy: false, formMessage: null, formErrors: {}, formWarnings: {}, removing: null, removeBusy: false,
    handoff: null, banner: null, alert: null, lastApplied: null,
  };

  // ---------- 静态骨架节点 ----------
  const chips = {};
  const chip = (field, label, iconName) => {
    const node = h('button', { type: 'button', class: 'ab-chip', 'data-field': field, 'aria-haspopup': 'dialog', 'aria-expanded': 'false', 'aria-controls': 'ab-pop' },
      h('span', { class: 'ab-chip-cap' }, h('span', { class: 'ab-chip-label', text: label }), h('span', { class: 'ab-chip-scope' })),
      h('span', { class: 'ab-chip-main' }, icon(iconName), h('span', { class: 'ab-chip-value' }), h('span', { class: 'ab-chip-meta' }),
        h('span', { class: 'ab-chip-state', 'aria-hidden': 'true' }), icon('caret', 'ab-ico ab-caret')));
    node.addEventListener('click', () => togglePop(field, node));
    node.addEventListener('keydown', (event) => { if (event.key === 'ArrowDown' && !event.altKey) { event.preventDefault(); if (s.open !== field) togglePop(field, node); } });
    chips[field] = node;
    return node;
  };
  const statusDot = h('span', { class: 'ab-status-dot', 'aria-hidden': 'true' });
  const statusText = h('p', { class: 'ab-status-text', role: 'status', 'aria-live': 'polite' });
  const alertText = h('p', { class: 'ab-alert', role: 'alert' });
  const statusActions = h('span', { class: 'ab-status-actions' });
  const statusTime = h('time', { class: 'ab-status-time' });
  const banner = h('div', { class: 'ab-banner', role: 'alert', hidden: true });
  const trayDiff = h('ul', { class: 'ab-diff' });
  const trayCheck = h('p', { class: 'ab-tray-check', role: 'status', 'aria-live': 'polite' });
  const trayRetryLabel = h('span', {});
  const trayRetry = h('button', { type: 'button', class: 'ab-btn ab-btn--s ab-btn--ghost', hidden: true, onclick: () => { s.check = { ...s.check, key: null }; scheduleCheck(); render(); } }, icon('retry'), trayRetryLabel);
  const trayNote = h('p', { class: 'ab-tray-note' });
  const trayBlockers = h('ul', { class: 'ab-tray-blockers' });
  const discardButton = h('button', { type: 'button', class: 'ab-btn', onclick: () => { s.draft = null; s.handoffNote = null; s.handoffFields = null; s.check = { key: null, state: 'idle', result: null }; render(); } });
  const switchButton = h('button', { type: 'button', class: 'ab-btn ab-btn--primary', onclick: () => openHandoffDialog() });
  const trayHeadTitle = h('strong', {});
  const tray = h('section', { class: 'ab-tray', 'data-open': 'false', inert: true },
    h('div', { class: 'ab-tray-clip' }, h('div', { class: 'ab-tray-inner' },
      h('div', { class: 'ab-tray-head' }, trayHeadTitle, trayCheck, trayRetry),
      trayDiff, trayBlockers, h('div', { class: 'ab-tray-foot' }, trayNote, h('div', { class: 'ab-tray-actions' }, discardButton, switchButton)))));
  root.classList.add('ab');
  root.dataset.loading = 'true';
  const chipsRow = h('div', { class: 'ab-chips', role: 'group' },
    chip('backend', '', 'backend'), chip('profile', 'Profile', 'profile'), chip('model', '', 'model'),
    chip('effort', '', 'effort'), chip('speed', '', 'speed'), chip('settings', '', 'settings'));
  const statusRow = h('div', { class: 'ab-status' }, statusDot, h('div', { class: 'ab-status-body' }, statusText, alertText), statusActions, statusTime);
  // External agent 只读信息卡
  const extDot = h('span', { class: 'ab-status-dot', 'aria-hidden': 'true' });
  const extState = h('span', { class: 'ab-ext-state' });
  const extClient = h('dd', { class: 'ab-ext-value' });
  const extLast = h('dd', { class: 'ab-ext-value' });
  const extEndpoint = h('code', { class: 'ab-ext-endpoint', tabindex: '0' });
  const extCopy = h('button', { type: 'button', class: 'ab-btn ab-btn--s ab-btn--ghost', onclick: async () => {
    const text = extEndpoint.textContent;
    try { await navigator.clipboard.writeText(text); setText(extCopy, pick('已复制', 'Copied')); }
    catch { const range = document.createRange(); range.selectNodeContents(extEndpoint); getSelection()?.removeAllRanges(); getSelection()?.addRange(range); setText(extCopy, pick('已选中', 'Selected')); }
    clearTimeout(extCopy.timer); extCopy.timer = setTimeout(() => setText(extCopy, pick('复制', 'Copy')), 1600);
  } });
  const extDtClient = h('dt', {});
  const extDtLast = h('dt', {});
  const extDtEndpoint = h('dt', {});
  const extNote = h('p', { class: 'ab-ext-note' });
  const extCard = h('section', { class: 'ab-ext', 'aria-label': 'External agent', hidden: true },
    h('div', { class: 'ab-ext-head' }, extDot, h('strong', { text: 'External agent' }), extState),
    h('dl', { class: 'ab-ext-rows' },
      extDtClient, extClient,
      extDtLast, extLast,
      extDtEndpoint, h('dd', { class: 'ab-ext-value ab-ext-endpoint-row' }, extEndpoint, extCopy)),
    extNote);
  root.replaceChildren(banner, chipsRow, extCard, tray, statusRow);

  // 浮层 / 移动端底部抽屉
  const popTitle = h('h3', { id: 'ab-pop-title', class: 'ab-pop-title' });
  const popSub = h('p', { class: 'ab-pop-sub' });
  const popTools = h('div', { class: 'ab-pop-tools' });
  const popProgress = h('div', { class: 'ab-progress', hidden: true, 'aria-hidden': 'true' });
  const popBody = h('div', { class: 'ab-pop-body' });
  const popGrip = h('div', { class: 'ab-pop-grip', 'aria-hidden': 'true' }, h('span'));
  const popCloseBtn = h('button', { type: 'button', class: 'ab-icon-btn ab-pop-close', onclick: () => closePop() }, icon('close'));
  const pop = h('section', { id: 'ab-pop', class: 'ab-pop', role: 'dialog', 'aria-labelledby': 'ab-pop-title', hidden: true },
    popGrip, h('header', { class: 'ab-pop-head' }, h('div', { class: 'ab-pop-titles' }, popTitle, popSub), popTools, popCloseBtn),
    popProgress, popBody);
  const scrim = h('div', { class: 'ab-scrim', hidden: true, onclick: () => closePop({ restoreFocus: true }) });

  // 非模态 handoff 确认框（保持顶部急停按钮始终可点）
  const dialogDiff = h('ul', { class: 'ab-diff ab-diff--stack' });
  const dialogFacts = h('ul', { class: 'ab-facts' });
  const dialogPrompt = h('textarea', { id: 'ab-handoff-prompt', rows: 3, maxlength: 16000 });
  const dialogError = h('p', { class: 'ab-dialog-error', role: 'alert' });
  const dialogConfirm = h('button', { type: 'button', class: 'ab-btn ab-btn--primary', onclick: () => confirmHandoff() });
  const dialogCancel = h('button', { type: 'button', class: 'ab-btn', onclick: () => closeHandoffDialog() });
  const dialogTitle = h('h3', { id: 'ab-dialog-title' });
  const dialogCloseBtn = h('button', { type: 'button', class: 'ab-icon-btn', onclick: () => closeHandoffDialog() }, icon('close'));
  const dialogPromptLabel = h('label', { class: 'ab-field-label', for: 'ab-handoff-prompt' });
  const dialog = h('dialog', { class: 'ab-dialog', 'aria-labelledby': 'ab-dialog-title' },
    h('header', { class: 'ab-dialog-head' }, dialogTitle, dialogCloseBtn),
    h('div', { class: 'ab-dialog-body' }, dialogDiff, dialogFacts, dialogPromptLabel, dialogPrompt, dialogError),
    h('footer', { class: 'ab-dialog-foot' }, dialogCancel, dialogConfirm));
  const dialogScrim = h('div', { class: 'ab-scrim ab-scrim--dialog', hidden: true });
  dialog.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') { event.preventDefault(); closeHandoffDialog(); }
    trapTab(event, dialog);
  });
  dialog.addEventListener('cancel', (event) => { event.preventDefault(); closeHandoffDialog(); });

  // Handoff 切换进度遮罩
  const overlayStepDefs = [['收尾当前 run', 'Finalizing run'], ['启动新 run', 'Starting new run'], ['就绪', 'Ready']];
  const overlayStepSpans = [];
  const overlaySteps = overlayStepDefs.map(([,], index) => {
    const labelSpan = h('span', {});
    overlayStepSpans.push(labelSpan);
    return h('li', { 'data-step': index + 1 }, h('span', { class: 'ab-step-dot' }), labelSpan);
  });
  const overlayNote = h('p', { class: 'ab-ov-note', role: 'status', 'aria-live': 'polite' });
  const overlayElapsed = h('p', { class: 'ab-ov-elapsed' });
  const overlayDismissBtn = h('button', { type: 'button', class: 'ab-btn', onclick: () => { overlay.hidden = true; } });
  const overlayActions = h('div', { class: 'ab-ov-actions', hidden: true }, overlayDismissBtn);
  const overlayTitle = h('h3', { id: 'ab-ov-title' });
  const overlay = h('div', { class: 'ab-overlay', hidden: true },
    h('div', { class: 'ab-ov-card', role: 'group', 'aria-labelledby': 'ab-ov-title' },
      overlayTitle, h('ol', { class: 'ab-steps' }, overlaySteps), overlayNote, overlayElapsed, overlayActions));
  const toast = h('div', { class: 'ab-toast', role: 'status', 'aria-live': 'polite', hidden: true });
  document.body.append(scrim, pop, dialogScrim, dialog, toast);
  (host || root).append(overlay);

  // 就地同步所有骨架静态标签（初始化与切换语言时执行，不引起轮询节点重建）
  function syncStaticLabels() {
    setText(chips.backend.querySelector('.ab-chip-label'), pick('后端', 'Backend'));
    setText(chips.profile.querySelector('.ab-chip-label'), 'Profile');
    setText(chips.model.querySelector('.ab-chip-label'), pick('模型', 'Model'));
    setText(chips.effort.querySelector('.ab-chip-label'), pick('推理强度', 'Effort'));
    setText(chips.speed.querySelector('.ab-chip-label'), pick('速度', 'Speed'));
    setText(chips.settings.querySelector('.ab-chip-label'), pick('管理', 'Manage'));
    setText(chips.settings.querySelector('.ab-chip-value'), pick('设置', 'Settings'));
    setAttr(chipsRow, 'aria-label', pick('Agent 设置', 'Agent settings'));
    setAttr(statusTime, 'title', pick('最近一次应用的时间', 'Last applied at'));
    const newSessionLabel = pick('需要新会话', 'Requires new session');
    setAttr(tray, 'aria-label', newSessionLabel);
    setText(trayHeadTitle, newSessionLabel);
    setText(trayRetryLabel, pick('重新检查', 'Recheck'));
    setText(discardButton, pick('放弃', 'Discard'));
    const switchText = pick('切换并新建 run', 'Switch & New Run');
    setText(switchButton, switchText);
    setAttr(extEndpoint, 'title', pick('本地 MCP 地址：在你自己的 Agent 里把它添加为 MCP 服务器', 'Local MCP endpoint: register this URL as an MCP server in your own Agent'));
    setText(extCopy, pick('复制', 'Copy'));
    setText(extDtClient, pick('最近 initialize 的客户端（自报，未验证）', 'Last initialize client (self-reported, unverified)'));
    setText(extDtLast, pick('最近工具调用', 'Last tool call'));
    setText(extDtEndpoint, pick('MCP 地址', 'MCP endpoint'));
    setText(extNote, pick('NI-RUA 不启动这个 Agent，也不能切换它的后端、模型或强度；任务在你自己的 Agent 里下达。', 'NI-RUA neither launches this Agent nor switches its backend, model, or effort; send tasks in your own Agent.'));
    setAttr(popCloseBtn, 'aria-label', pick('关闭', 'Close'));
    setText(dialogTitle, switchText);
    setAttr(dialogCloseBtn, 'aria-label', pick('取消', 'Cancel'));
    setText(dialogPromptLabel, pick('新 run 的首条指令（可选）', 'Initial prompt for new run (optional)'));
    setAttr(dialogPrompt, 'placeholder', pick('留空 = 只观察，不执行任务', 'Leave empty = observe only, no initial task'));
    setText(dialogCancel, pick('取消', 'Cancel'));
    if (!dialogConfirm.disabled) setText(dialogConfirm, switchText);
    setText(overlayTitle, pick('正在切换到新 run', 'Switching to new run'));
    overlayStepDefs.forEach(([zh, en], index) => setText(overlayStepSpans[index], pick(zh, en)));
    setText(overlayDismissBtn, pick('关闭提示（继续等待）', 'Dismiss (keep waiting)'));
  }
  syncStaticLabels();

  const header = document.querySelector('header.bar');
  let headerFrame = 0, headerValue = '';
  const syncHeader = () => {
    headerFrame = 0;
    const value = `${Math.max(0, Math.round(header?.getBoundingClientRect().bottom || 0))}px`;
    if (value !== headerValue) { headerValue = value; document.documentElement.style.setProperty('--ab-header-h', value); }
  };
  const scheduleHeader = () => { if (!headerFrame) headerFrame = requestAnimationFrame(syncHeader); };
  if (header) { new ResizeObserver(scheduleHeader).observe(header); addEventListener('scroll', scheduleHeader, { passive: true }); }
  syncHeader();

  // ---------- 派生状态 ----------
  function options() {
    if (s.raw && s.rawStatus === 200) return M.normalizeOptions(s.raw, { consoleState: s.consoleState });
    return M.legacyOptions(s.consoleState, s.run);
  }
  const leaseOther = () => Boolean(api.lease?.().other);
  function effortControl(o) {
    if (s.draft && s.check.result?.reasoning?.effort && s.check.key === JSON.stringify(M.checkBody(s.draft))) return s.check.result.reasoning.effort;
    if (s.draft && s.draft.backend !== o.current.backend) {
      const backend = o.backends.find((item) => item.id === s.draft.backend);
      if (Array.isArray(backend?.effort_choices)) return { choices: backend.effort_choices.map((value) => ({ value, backend: true, model: null })), applies: backend.live_effort ? 'next_turn' : 'next_session' };
    }
    return o.reasoning?.effort || null;
  }
  function thinkingControl(o) {
    if (s.draft && s.check.result?.reasoning?.thinking_display) return s.check.result.reasoning.thinking_display;
    return o.reasoning?.thinking_display || { available: false };
  }
  const effortScope = (o) => (o.source === 'legacy' ? 'next_turn' : s.handoffFields?.includes('effort') ? 'next_session' : M.scopeKind(effortControl(o)?.applies));
  function display(o) {
    const current = o.current;
    const live = (field) => M.displayValue(current[field], s.intents[field]);
    if (s.draft) return { ...s.draft, service_tier: live('service_tier').value };
    return { ...current, model: live('model').value, effort: live('effort').value, service_tier: live('service_tier').value };
  }
  function catalogueFor(o, d) {
    if (o.source === 'legacy') return o.legacyModels;
    if (!d.profile || !d.wire_api) return M.legacyOptions(s.consoleState, s.run).legacyModels;
    return s.catalogues.get(`${d.profile}|${d.wire_api}`) || { loading: true, models: [] };
  }
  const sharedWires = (o, d) => M.commonWires(o.backends.find((item) => item.id === d.backend), o.profiles.find((item) => item.id === d.profile));

  // ---------- 服务端读取 ----------
  let optionsInflight = null;
  function refreshOptions() {
    if (optionsInflight) return optionsInflight;
    optionsInflight = api.getJSON('/v1/console/agent-options').then((value) => {
      if (value && Array.isArray(value.backends) && value.current) { s.raw = value; s.rawStatus = 200; s.rawError = null; } else { s.raw = null; s.rawStatus = 404; s.rawError = null; }
    },
      (error) => { s.rawStatus = error.status || 0; s.rawError = error.status === 404 ? null : error.message || 'network_error'; if (error.status === 404) s.raw = null; })
      .finally(() => { s.rawAt = Date.now(); optionsInflight = null; render(); });
    return optionsInflight;
  }
  async function loadCatalogue(profile, wire, { refresh = false } = {}) {
    if (!profile || !wire) return;
    const key = `${profile}|${wire}`;
    const entry = s.catalogues.get(key) || { models: [] };
    if (entry.loading && !refresh) return;
    if (!refresh && entry.loadedAt && !entry.error && Date.now() - entry.loadedAt < CATALOGUE_TTL_MS) return;
    s.catalogues.set(key, { ...entry, loading: true, error: null });
    render();
    try {
      const value = await api.getJSON(`/v1/console/profiles/${encodeURIComponent(profile)}/models?wire=${encodeURIComponent(wire)}&refresh=${refresh ? 1 : 0}`);
      s.catalogues.set(key, { ...value, models: Array.isArray(value.models) ? value.models : [], loading: false, error: null, loadedAt: Date.now() });
    } catch (error) {
      s.catalogues.set(key, { ...entry, loading: false, error: error.message || 'network_error', loadedAt: Date.now() });
    }
    render();
  }
  let checkTimer = 0;
  function scheduleCheck() {
    clearTimeout(checkTimer);
    if (!s.draft) { s.check = { key: null, state: 'idle', result: null }; return; }
    if (!s.draft.profile) { s.check = { key: null, state: 'incomplete', result: null }; return; }
    const body = M.checkBody(s.draft);
    const key = JSON.stringify(body);
    if (s.check.key === key && s.check.state !== 'error') return;
    s.check = { key, state: 'loading', result: s.check.result };
    checkTimer = setTimeout(async () => {
      let next;
      try { next = { key, state: 'done', result: await api.getJSON('/v1/console/agent-options/check', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }) }; }
      catch (error) { next = { key, state: 'error', result: { ok: false, reason: error.body?.error || 'check_failed', message: error.body?.message || error.message } }; }
      if (s.draft && JSON.stringify(M.checkBody(s.draft)) === key) { s.check = next; render(); }
    }, CHECK_DEBOUNCE_MS);
  }

  // ---------- 写入与提交 ----------
  function setDraft(next) {
    const o = options();
    const diff = M.sessionDiff(o.current, next, { effortScope: effortScope(o) });
    const opening = !s.draft;
    s.draft = next && diff.fields.length && diff.handoff ? next : null;
    if (s.draft && opening && Date.now() - s.rawAt > 2000) refreshOptions();
    if (!s.draft) { s.handoffNote = null; s.handoffFields = null; }
    scheduleCheck();
    render();
  }
  const base = (o) => s.draft || M.draftFrom(o.current);

  async function applyLive(fields) {
    const o = options();
    s.alert = null;
    if (leaseOther()) { s.alert = pick('另一窗口持有控制，未应用', 'Another window holds control; not applied'); render(); return; }
    const entries = Object.entries(fields);
    const now = Date.now();
    for (const [field, value] of entries) s.intents[field] = M.createIntent(field, value, { now, previous: o.current[field] });
    const ids = Object.fromEntries(entries.map(([field]) => [field, s.intents[field].id]));
    const mine = (field) => s.intents[field]?.id === ids[field];
    render();
    if (!(await api.ensureLease())) {
      for (const [field] of entries) if (mine(field)) s.intents[field] = M.intentReceipt(s.intents[field], { status: 'rejected', message: pick('另一窗口持有控制', 'Another window holds control') });
      s.alert = pick('另一窗口持有控制，未应用', 'Another window holds control; not applied');
      render();
      return;
    }
    const turnId = String(s.run?.turn_id ?? '');
    if (o.source === 'legacy') {
      let stopped = false;
      for (const [field, value] of entries) {
        if (!mine(field)) continue;
        if (stopped) { s.intents[field] = M.intentReceipt(s.intents[field], { status: 'skipped' }); continue; }
        try {
          const receipt = await api.control(COMMANDS[field], field === 'service_tier' ? { tier: value } : { [field]: value }, { key: `agent-${field}` });
          if (mine(field)) s.intents[field] = M.intentReceipt(s.intents[field], receipt, { turnId });
          stopped = receipt.status !== 'applied';
        } catch (error) {
          if (mine(field)) s.intents[field] = M.intentError(s.intents[field], error);
          stopped = true;
        }
      }
    } else {
      try {
        const receipt = await api.control('apply_agent_settings', fields, { key: `agent-apply:${Object.keys(fields).sort().join(',')}` });
        if (M.isHandoffRequired(receipt)) {
          for (const [field] of entries) if (mine(field)) delete s.intents[field];
          s.notice = null;
          s.handoffFields = Array.isArray(receipt.session_fields) ? receipt.session_fields : [];
          setDraft(M.draftFromHandoffRequired(M.draftFrom(options().current), receipt));
          s.alert = null;
          const fieldText = (receipt.session_fields || []).map((field) => M.fieldLabel(field)).join(pick('、', ', '));
          s.handoffNote = trServer(receipt.message) || pick(`${fieldText} 需要新会话`, `${fieldText} requires a new session`);
          render();
          return;
        }
        for (const [field] of entries) {
          if (!mine(field)) continue;
          const result = receipt.results?.[field] || { status: receipt.status === 'applied' ? 'applied' : 'skipped', message: receipt.message };
          s.intents[field] = M.intentReceipt(s.intents[field], result, { turnId });
        }
      } catch (error) {
        for (const [field] of entries) if (mine(field)) s.intents[field] = M.intentError(s.intents[field], error);
      }
    }
    const rejected = entries.map(([field]) => s.intents[field]).filter((intent) => intent?.state === 'rejected' && ids[intent.field] === intent.id);
    if (rejected.length) {
      s.alert = rejected.map((intent) => pick(
        `${M.fieldLabel(intent.field)} ${intent.value} 未应用：${intent.message}`,
        `${M.fieldLabel(intent.field)} ${intent.value} not applied: ${intent.message}`,
      )).join(pick('；', '; '));
    }
    if (entries.some(([field]) => s.intents[field]?.state === 'confirmed')) s.lastApplied = new Date();
    render();
    refreshOptions();
    api.refreshConsole?.();
  }
  function retryIntent(intent) { applyLive({ [intent.field]: intent.value }); }
  function dismissIntent(field) { delete s.intents[field]; s.alert = null; render(); }

  function chooseBackend(id) {
    const o = options();
    if (id === o.current.backend && !s.draft) { closePop(); return; }
    const next = M.selectBackend(o, base(o), id);
    setDraft(next);
    if (!next.profile) {
      s.notice = pick(`请选择与 ${M.backendLabel(id)} 兼容的 profile`, `Select a profile compatible with ${M.backendLabel(id)}`);
      openPop('profile', chips.profile);
    } else closePop();
  }
  function chooseProfile(id) {
    const o = options();
    s.notice = null;
    setDraft(M.selectProfile(o, base(o), id));
    closePop();
  }
  function chooseModel(id, { requiresWire = null } = {}) {
    const o = options();
    s.query = '';
    if (s.draft || requiresWire) {
      const draft = base(o);
      const preset = M.presetFor(o, { profile: draft.profile, backend: draft.backend, model: id });
      setDraft({ ...draft, model: id, wire_api: requiresWire || draft.wire_api, effort: preset?.effort || draft.effort });
      closePop();
      return;
    }
    closePop();
    if (id !== display(o).model || s.intents.model?.state === 'unknown') applyLive({ model: id });
  }
  function chooseEffort(value) {
    const o = options();
    closePop();
    if (s.draft || (effortScope(o) === 'next_session' && o.source !== 'legacy')) { setDraft({ ...base(o), effort: value }); return; }
    if (value !== display(o).effort || s.intents.effort?.state === 'unknown') applyLive({ effort: value });
  }
  function chooseTier(value) {
    const o = options();
    closePop();
    if (value !== display(o).service_tier || s.intents.service_tier?.state === 'unknown') applyLive({ service_tier: value });
  }

  // ---------- 浮层管理 ----------
  let popKind = null, popParts = null, closeTimer = 0;
  function togglePop(kind, anchor) { if (s.open === kind) closePop(); else openPop(kind, anchor); }
  function openPop(kind, anchor) {
    clearTimeout(closeTimer);
    if (s.open && s.chip) s.chip.setAttribute('aria-expanded', 'false');
    s.alert = null;
    s.open = kind; s.chip = anchor; s.query = '';
    if (kind !== 'profile') s.notice = null;
    if (kind === 'settings') { s.settingsView = 'list'; s.removing = null; }
    anchor.setAttribute('aria-expanded', 'true');
    buildPop(kind);
    pop.hidden = false;
    pop.dataset.state = 'closed';
    place();
    paint();
    requestAnimationFrame(() => { pop.dataset.state = 'open'; });
    focusPop();
    const o = options(), d = display(o);
    if (kind === 'model' && o.source !== 'legacy') loadCatalogue(d.profile, d.wire_api);
    if (Date.now() - s.rawAt > 5000) refreshOptions();
  }
  function focusPop() {
    if (!popParts) return;
    if (popParts.combobox) popParts.combobox.input.focus({ preventScroll: true });
    else if (popParts.listbox) popParts.listbox.focusSelected();
    else focusables(popBody)[0]?.focus({ preventScroll: true });
  }
  function closePop({ restoreFocus = true } = {}) {
    if (!s.open) return;
    const anchor = s.chip;
    s.open = null; s.chip = null; popKind = null; popParts = null; s.notice = null;
    anchor?.setAttribute('aria-expanded', 'false');
    pop.dataset.state = 'closed';
    scrim.hidden = true;
    const finish = () => { if (!s.open) { pop.hidden = true; popBody.replaceChildren(); popTools.replaceChildren(); } };
    if (reducedMotion.matches) finish(); else closeTimer = setTimeout(finish, 180);
    if (restoreFocus && anchor && !anchor.disabled) anchor.focus({ preventScroll: true });
    render();
  }
  function place() {
    if (!s.open) return;
    const sheet = phone.matches;
    pop.classList.toggle('ab-pop--sheet', sheet);
    scrim.hidden = !sheet;
    if (sheet) { for (const key of ['left', 'top', 'bottom', 'width', 'maxHeight']) pop.style[key] = ''; return; }
    const rect = s.chip.getBoundingClientRect();
    const top = Number.parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--ab-header-h')) || 0;
    const width = Math.min(POP_WIDTH[s.open] || 380, innerWidth - 16);
    const left = Math.max(8, Math.min(rect.left, innerWidth - width - 8));
    const below = innerHeight - rect.bottom - 14, above = rect.top - top - 14;
    const up = below < 280 && above > below;
    Object.assign(pop.style, { width: `${width}px`, left: `${left}px`, maxHeight: `${Math.max(180, Math.min(640, up ? above : below))}px`,
      top: up ? '' : `${rect.bottom + 6}px`, bottom: up ? `${innerHeight - rect.top + 6}px` : '' });
    pop.dataset.placement = up ? 'top' : 'bottom';
  }
  let placeFrame = 0;
  const schedulePlace = () => { if (s.open && !placeFrame) placeFrame = requestAnimationFrame(() => { placeFrame = 0; place(); }); };
  addEventListener('resize', schedulePlace);
  addEventListener('scroll', schedulePlace, { passive: true, capture: true });
  phone.addEventListener('change', schedulePlace);
  document.addEventListener('pointerdown', (event) => {
    if (!s.open || pop.contains(event.target) || s.chip?.contains(event.target) || dialog.contains(event.target)) return;
    if (event.target === scrim) return;
    closePop({ restoreFocus: false });
  }, true);
  pop.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && !event.defaultPrevented) {
      event.preventDefault();
      if (s.open === 'settings' && s.settingsView === 'form') { leaveForm(); return; }
      closePop();
    }
    if (pop.classList.contains('ab-pop--sheet')) trapTab(event, pop);
  });
  pop.addEventListener('focusout', (event) => {
    const next = event.relatedTarget;
    if (s.open && !pop.classList.contains('ab-pop--sheet') && next && !pop.contains(next) && !s.chip?.contains(next) && !dialog.contains(next)) closePop({ restoreFocus: false });
  });
  let drag = null;
  popGrip.addEventListener('pointerdown', (event) => { if (!pop.classList.contains('ab-pop--sheet')) return; drag = { y: event.clientY, dy: 0 }; popGrip.setPointerCapture(event.pointerId); pop.classList.add('is-dragging'); });
  popGrip.addEventListener('pointermove', (event) => { if (!drag) return; drag.dy = Math.max(0, event.clientY - drag.y); pop.style.transform = `translateY(${drag.dy}px)`; });
  const endDrag = () => { if (!drag) return; const dismiss = drag.dy > 90; drag = null; pop.classList.remove('is-dragging'); pop.style.transform = ''; if (dismiss) closePop(); };
  popGrip.addEventListener('pointerup', endDrag);
  popGrip.addEventListener('pointercancel', endDrag);

  function buildPop(kind) {
    popKind = kind;
    popTools.replaceChildren();
    popBody.replaceChildren();
    pop.dataset.kind = kind;
    if (kind === 'backend' || kind === 'profile' || kind === 'effort' || kind === 'speed') {
      const listLabel = {
        backend: pick('后端', 'Backend'),
        profile: pick('供应商 Profile', 'Provider Profile'),
        effort: pick('推理强度', 'Effort'),
        speed: pick('速度', 'Speed'),
      }[kind];
      const listbox = createListbox({
        label: listLabel,
        className: 'ab-list',
        onSelect: (item) => ({ backend: chooseBackend, profile: chooseProfile, effort: chooseEffort, speed: chooseTier })[kind](item.key),
        onDisabled: (item) => { s.alert = pick(`${item.label} 不可选：${item.reason || '不兼容'}`, `${item.label} unavailable: ${item.reason || 'incompatible'}`); render(); },
      });
      const note = h('p', { class: 'ab-pop-note', role: 'status' });
      const foot = h('div', { class: 'ab-pop-foot' });
      popBody.append(note, listbox.element, foot);
      popParts = { listbox, note, foot };
      if (kind === 'profile') {
        foot.append(
          h('button', { type: 'button', class: 'ab-btn ab-btn--ghost', onclick: () => { openPop('settings', chips.settings); } }, icon('settings'), pick('管理 profile', 'Manage profiles')),
          h('button', { type: 'button', class: 'ab-btn ab-btn--ghost', onclick: () => { openPop('settings', chips.settings); enterForm(null); } }, icon('plus'), pick('新建 profile', 'New profile')),
        );
      }
    } else if (kind === 'model') {
      const combobox = createCombobox({
        label: pick('搜索或直接输入模型名', 'Search or enter model ID'),
        placeholder: pick('搜索或直接输入模型名…', 'Search or enter model ID…'),
        className: 'ab-list',
        onInput: (text) => { s.query = text; paint(); },
        onSelect: (item) => (item.key === '__free__' ? chooseModel(item.value) : chooseModel(item.key, { requiresWire: item.requiresWire })),
        onSubmitText: (text) => { if (M.MODEL_ID.test(text)) chooseModel(text); else { s.alert = pick('模型名格式无效', 'Invalid model ID format'); paint(); } },
        onDisabled: (item) => { s.alert = pick(`${item.label} 不可选：${item.reason}`, `${item.label} unavailable: ${item.reason}`); render(); },
      });
      if (s.query) combobox.input.value = s.query;
      const refresh = h('button', { type: 'button', class: 'ab-btn ab-btn--ghost ab-btn--s', onclick: () => { const d = display(options()); loadCatalogue(d.profile, d.wire_api, { refresh: true }); } }, icon('refresh'), h('span', { text: pick('刷新', 'Refresh') }));
      popTools.append(refresh);
      const summary = h('p', { class: 'ab-cat-summary', role: 'status', 'aria-live': 'polite' });
      const retry = h('button', { type: 'button', class: 'ab-btn ab-btn--s', hidden: true, onclick: () => { const d = display(options()); loadCatalogue(d.profile, d.wire_api, { refresh: true }); } }, icon('retry'), pick('重试', 'Retry'));
      const skeleton = h('div', { class: 'ab-skeleton', 'aria-hidden': 'true' }, Array.from({ length: 5 }, () => h('div', { class: 'ab-skel-row' }, h('span'), h('span'))));
      popBody.append(h('div', { class: 'ab-search' }, icon('search'), combobox.input),
        h('div', { class: 'ab-cat-line' }, summary, retry), skeleton, combobox.listbox.element,
        h('p', { class: 'ab-pop-hint', 'aria-hidden': 'true', text: pick('↑ ↓ 选择 · 回车应用 · 可直接输入任意模型名 · Esc 关闭', '↑ ↓ select · Enter apply · or type any model ID · Esc close') }));
      popParts = { combobox, listbox: combobox.listbox, summary, retry, skeleton, refresh };
    } else if (kind === 'settings') {
      popParts = { settings: true };
    }
  }

  function paintPop(o, d) {
    if (!s.open || !popParts) return;
    const current = o.current;
    const curTagText = pick('当前', 'Current');
    popProgress.hidden = true;
    if (popKind === 'backend') {
      setText(popTitle, pick('后端', 'Backend'));
      setText(popSub, pick('切换后端需要新会话：当前 run 收尾，本页启动新 run', 'Switching backend requires a new session: finalizes current run and starts a new run here'));
      const rows = M.backendRows(o, d);
      popParts.listbox.setItems(rows.map((row) => ({ key: row.id, label: row.label, disabled: row.disabled, selected: row.selected, reason: row.reason,
        render: (li) => li.append(h('span', { class: 'ab-opt-check' }, row.selected ? icon('check') : null),
          h('span', { class: 'ab-opt-main' }, h('span', { class: 'ab-opt-title', text: row.label }),
            h('span', { class: 'ab-opt-sub' }, row.wires.length ? row.wires.join(' · ') : null, row.id === current.backend ? h('span', { class: 'ab-tag', text: curTagText }) : null),
            row.reason ? h('span', { class: `ab-opt-reason${row.disabled ? '' : ' is-note'}`, text: row.reason }) : null)) })),
      { signature: JSON.stringify([getLocale(), rows]) });
      setText(popParts.note, o.source === 'legacy' ? pick('当前网关版本只能查看后端；切换需要更新网关', 'Current gateway is read-only for backends; update gateway to switch') : '');
    } else if (popKind === 'profile') {
      setText(popTitle, pick('供应商 Profile', 'Provider Profile'));
      setText(popSub, pick(`用于 ${M.backendLabel(d.backend)} · 切换需要新会话`, `For ${M.backendLabel(d.backend)} · switching requires a new session`));
      const rows = M.profileRows(o, d, current);
      popParts.listbox.setItems(rows.map((row) => ({ key: row.id, label: row.name, disabled: row.disabled, selected: row.selected, reason: row.reason,
        render: (li) => li.append(h('span', { class: 'ab-opt-check' }, row.selected ? icon('check') : null),
          h('span', { class: 'ab-opt-main' }, h('span', { class: 'ab-opt-title' }, row.name, h('span', { class: 'ab-opt-id', text: row.id })),
            h('span', { class: 'ab-opt-sub' }, row.wires.map((wire) => h('span', { class: 'ab-wire', title: wire.host || '' }, wire.wire, wire.host ? h('small', { text: wire.host }) : null)),
              h('span', { class: `ab-key${row.keyPresent ? '' : ' is-missing'}`, text: row.keyText }), row.current ? h('span', { class: 'ab-tag', text: curTagText }) : null),
            row.catalogue && !row.disabled ? h('span', { class: 'ab-opt-reason is-note', 'data-tone': row.catalogue.tone === 'warn' ? 'warn' : null, text: row.catalogue.text }) : null,
            row.notes.map((note) => h('span', { class: 'ab-opt-reason', text: note })),
            row.reason ? h('span', { class: 'ab-opt-reason', text: row.reason }) : null)) })),
      { signature: JSON.stringify([getLocale(), rows]) });
      setText(popParts.note, s.notice || (o.source === 'legacy'
        ? pick('当前网关版本不提供 profile 列表', 'Current gateway does not list profiles')
        : rows.length ? '' : pick('还没有 profile，点「新建 profile」粘贴一个密钥', 'No profiles yet — click "New profile" to paste an API key')));
      popParts.note.dataset.tone = s.notice ? 'warn' : '';
    } else if (popKind === 'effort') {
      const control = effortControl(o);
      const draftMode = s.draft || (effortScope(o) === 'next_session' && o.source !== 'legacy');
      setText(popTitle, pick('推理强度', 'Reasoning Effort'));
      setText(popSub, `${draftMode ? pick('需要新会话', 'Requires new session') : M.scopeLabel(control?.applies) || pick('下一轮生效', 'Applies next turn')}${control?.model_source === 'catalogue' ? pick(' · 模型支持情况来自供应商目录', ' · model support from provider catalogue') : ''}`);
      const rows = M.effortRows(control, d.effort);
      popParts.listbox.setItems(rows.map((row) => ({ key: row.value, label: row.value, disabled: row.disabled, selected: row.selected, reason: row.note,
        render: (li) => li.append(h('span', { class: 'ab-opt-check' }, row.selected ? icon('check') : null),
          h('span', { class: 'ab-opt-main' }, h('span', { class: 'ab-opt-title ab-mono', text: row.value }),
            row.description ? h('span', { class: 'ab-opt-sub', text: row.description }) : null,
            row.note ? h('span', { class: `ab-opt-reason${row.disabled ? '' : ' is-note'}`, 'data-tone': row.tone, text: row.note }) : null)) })),
      { signature: JSON.stringify([getLocale(), rows]) });
      const next = !s.draft && o.current.effort_next_session
        ? pick(
          `已为下次会话选择 ${o.current.effort_next_session}；当前会话仍为 ${o.current.effort ?? '默认'}。`,
          `Selected ${o.current.effort_next_session} for next session; current session remains ${o.current.effort ?? 'default'}.`,
        )
        : '';
      setText(popParts.note, rows.length ? next || trServer(control?.note) || '' : pick('当前组合没有可选的推理强度', 'No selectable reasoning efforts for this combination'));
    } else if (popKind === 'speed') {
      const control = o.reasoning?.service_tier;
      const served = M.servedTier(o, d.service_tier);
      setText(popTitle, pick('速度 service_tier', 'Speed service_tier'));
      setText(popSub, served.text || pick('下一次模型请求生效，运行中也可切换', 'Applies on next model request (including mid-turn)'));
      const rows = M.tierRows(control, d.service_tier);
      popParts.listbox.setItems(rows.map((row) => ({ key: row.value, label: row.label, disabled: row.disabled, selected: row.selected, reason: row.note,
        render: (li) => li.append(h('span', { class: 'ab-opt-check' }, row.selected ? icon('check') : null),
          h('span', { class: 'ab-opt-main' }, h('span', { class: 'ab-opt-title' }, row.label, h('span', { class: 'ab-opt-id', text: row.value })),
            row.description ? h('span', { class: 'ab-opt-sub', text: row.description }) : null,
            h('span', { class: `ab-opt-reason${row.disabled ? '' : ' is-note'}`, 'data-tone': row.tone, text: row.note }))) })),
      { signature: JSON.stringify([getLocale(), rows]) });
      setText(popParts.note, served.state === 'mismatch' ? pick('供应商接受了请求的档位，但探测响应报告的实际档位不同；以每次响应为准。', 'Provider accepted the requested tier, but the probe reported a different served tier; actual responses govern.') : '');
      popParts.note.dataset.tone = served.state === 'mismatch' ? 'warn' : '';
    } else if (popKind === 'model') paintModelPop(o, d);
    else if (popKind === 'settings') paintSettings(o, d);
  }

  function paintModelPop(o, d) {
    const profile = o.profiles.find((item) => item.id === d.profile);
    setText(popTitle, s.draft ? pick('模型 · 新会话', 'Model · New Session') : pick('模型', 'Model'));
    setText(popSub, [profile?.name || (o.source === 'legacy' || !d.profile ? pick('配置文件', 'Config file') : d.profile), d.wire_api].filter(Boolean).join(' · ') + (s.draft ? '' : pick(' · 下一轮生效', ' · applies next turn')));
    const catalogue = catalogueFor(o, d);
    const { rows, freeText } = M.modelRows(catalogue?.models || [], { wire: d.wire_api, sharedWires: sharedWires(o, d), query: s.query, selected: d.model, current: s.draft ? s.draft.model : o.current.model });
    const summary = M.catalogueSummary(catalogue);
    setText(popParts.summary, summary.text);
    popParts.summary.dataset.tone = summary.tone;
    popParts.retry.hidden = !summary.retry || o.source === 'legacy';
    popParts.refresh.hidden = o.source === 'legacy' || !d.profile;
    const loadingEmpty = catalogue?.loading && !(catalogue.models || []).length;
    popParts.skeleton.hidden = !loadingEmpty;
    popProgress.hidden = !(catalogue?.loading && (catalogue.models || []).length);
    const items = rows.map((row) => ({ key: row.id, label: row.id, disabled: row.disabled, selected: row.selected, reason: row.reason, requiresWire: row.requiresWire,
      render: (li) => li.append(h('span', { class: 'ab-opt-check' }, row.selected ? icon('check') : null),
        h('span', { class: 'ab-opt-main' }, h('span', { class: 'ab-opt-title ab-mono', text: row.id }), row.name ? h('span', { class: 'ab-opt-sub', text: row.name }) : null,
          row.reason ? h('span', { class: `ab-opt-reason${row.disabled ? '' : ' is-note'}`, text: row.reason }) : null),
        ...(row.badges.length ? [h('span', { class: 'ab-badges' }, row.badges.map(badgeNode))] : [])) }));
    if (freeText) {
      items.push({ key: '__free__', value: freeText.id, label: freeText.id, disabled: !freeText.valid, selected: false, reason: freeText.note,
        render: (li) => { li.classList.add('ab-opt--free'); li.append(h('span', { class: 'ab-opt-check' }, h('kbd', { text: '↵' })),
          h('span', { class: 'ab-opt-main' }, h('span', { class: 'ab-opt-title' }, pick('使用 ', 'Use '), h('span', { class: 'ab-mono', text: `“${freeText.id}”` })),
            h('span', { class: `ab-opt-reason${freeText.valid ? ' is-note' : ''}`, text: freeText.note }))); } });
    }
    popParts.combobox.setItems(items, { signature: JSON.stringify([getLocale(), rows.map((row) => [row.id, row.selected, row.disabled, row.reason, row.badges.length]), freeText, catalogue?.status]) });
    if (!rows.length && !freeText && !loadingEmpty) {
      popParts.listbox.element.dataset.empty = s.query
        ? pick('没有匹配的模型；回车可直接使用输入的模型名', 'No matching models; press Enter to use typed model ID directly')
        : pick('供应商没有返回模型列表，可以直接输入模型名', 'Provider returned no model list; type any model ID directly');
    } else delete popParts.listbox.element.dataset.empty;
  }

  // ---------- Profile 管理与高级会话选项 ----------
  let settingsSignature = null;
  function paintSettings(o, d) {
    if (s.settingsView === 'form') {
      setText(popTitle, s.formMode === 'edit' ? pick(`编辑 profile · ${s.form?.id}`, `Edit profile · ${s.form?.id}`) : pick('新建 profile', 'New profile'));
      setText(popSub, pick('保存到 configs/profiles/<id>.local.json（git 忽略，0600）', 'Saved to configs/profiles/<id>.local.json (gitignored, mode 0600)'));
      paintFormState();
      return;
    }
    setText(popTitle, pick('Profile 与高级设置', 'Profiles & Advanced Settings'));
    setText(popSub, pick('密钥只写入本机，任何接口都不会再返回', 'API keys are written locally only and never returned by any endpoint'));
    const signature = JSON.stringify([getLocale(), o.source, o.profiles, o.matrix, [...o.catalogues.values()], s.raw?.profiles_dir, s.removing, s.removeBusy, s.formMessage, d.backend, d.profile, d.wire_api, d.thinking_display, sharedWires(o, d), thinkingControl(o), leaseOther()]);
    if (signature === settingsSignature && popBody.childElementCount) return;
    settingsSignature = signature;
    const keep = document.activeElement && popBody.contains(document.activeElement) ? document.activeElement.dataset.focusKey : null;
    popBody.replaceChildren(settingsList(o, d), advancedSection(o, d),
      h('p', { class: 'ab-pop-foot ab-pop-foot--plain' }, h('button', { type: 'button', class: 'ab-link', onclick: () => {
        closePop({ restoreFocus: false });
        const config = document.getElementById('config')?.closest('details');
        if (config) { config.open = true; config.scrollIntoView({ block: 'center', behavior: reducedMotion.matches ? 'auto' : 'smooth' }); config.querySelector('summary')?.focus(); }
      } }, pick('查看生效配置（只读）', 'View effective config (read-only)'))));
    if (keep) popBody.querySelector(`[data-focus-key="${CSS.escape(keep)}"]`)?.focus({ preventScroll: true });
  }
  function settingsList(o, d) {
    const readOnly = leaseOther();
    const section = h('section', { class: 'ab-section', 'aria-labelledby': 'ab-profiles-title' },
      h('div', { class: 'ab-section-head' }, h('h4', { id: 'ab-profiles-title', text: pick('供应商 Profile', 'Provider Profiles') }),
        h('button', { type: 'button', class: 'ab-btn ab-btn--s', 'data-focus-key': 'new', disabled: readOnly || o.source === 'legacy', onclick: () => enterForm(null) }, icon('plus'), pick('新建 profile', 'New profile'))));
    if (s.formMessage) section.append(h('p', { class: 'ab-form-msg', 'data-tone': s.formMessage.tone, role: 'status', text: s.formMessage.text }));
    if (o.source === 'legacy') {
      section.append(h('p', { class: 'ab-empty', text: pick('当前网关版本不提供 profile 接口。更新网关后可在这里新建、编辑和移除 profile。', 'Current gateway does not expose profile endpoints. Update the gateway to create, edit, and remove profiles here.') }));
      return section;
    }
    if (s.raw?.profiles_dir) section.append(h('p', { class: 'ab-help ab-dir' }, pick('目录 ', 'Directory '), h('code', { class: 'ab-mono', text: s.raw.profiles_dir })));
    if (!o.profiles.length) {
      section.append(h('p', { class: 'ab-empty', text: pick('还没有 profile。新建一个并粘贴供应商密钥即可在页面上切换。', 'No profiles yet. Create one and paste a provider API key to switch on this page.') }));
      return section;
    }
    const list = h('ul', { class: 'ab-profiles' });
    for (const profile of o.profiles) {
      const inUse = profile.in_use === true || profile.id === o.current.profile;
      const confirming = s.removing === profile.id;
      const error = M.profileError(profile);
      const catalogue = error ? null : M.catalogueBrief(o, profile.id);
      const displayName = profile.name || profile.id;
      list.append(h('li', { class: 'ab-profile', 'data-invalid': error ? 'true' : null },
        h('div', { class: 'ab-profile-main' },
          h('div', { class: 'ab-profile-name' }, h('strong', { text: displayName }), h('span', { class: 'ab-opt-id', text: profile.id }),
            inUse ? h('span', { class: 'ab-tag ab-tag--accent', text: pick('当前 run 使用中', 'In use by active run') }) : null,
            error ? h('span', { class: 'ab-tag ab-tag--bad', text: pick('无效', 'Invalid') }) : null),
          error ? null : h('div', { class: 'ab-opt-sub' }, Object.entries(profile.wires || {}).map(([wire, value]) => h('span', { class: 'ab-wire' }, wire, value?.host ? h('small', { text: value.host }) : null)),
            h('span', { class: `ab-key${profile.key_present ? '' : ' is-missing'}`, text: M.keyStatus(profile) })),
          error ? h('p', { class: 'ab-opt-reason ab-error-line' }, h('code', { class: 'ab-code', text: error.code }), h('span', { text: error.text })) : null,
          error ? null : h('ul', { class: 'ab-compat', 'aria-label': pick(`${displayName} 可用于`, `${displayName} compatibility`) }, M.profileCompatibility(o, profile.id).map((row) =>
            h('li', { 'data-ok': row.compatible ? (row.partial ? 'partial' : 'true') : 'false', title: row.message || null },
              h('span', { class: 'ab-compat-mark', 'aria-hidden': 'true', text: row.compatible ? '✓' : '✕' }), row.label,
              h('span', { class: 'ab-sr', text: row.compatible ? pick('：可用', ': compatible') : pick(`：不可用（${row.message || row.reason}）`, `: incompatible (${row.message || row.reason})`) })))),
          error ? null : M.profileCompatibility(o, profile.id).filter((row) => row.message).map((row) => h('p', { class: 'ab-opt-reason is-note ab-compat-why', text: `${row.label}${pick('：', ': ')}${row.message}` })),
          catalogue ? h('p', { class: 'ab-opt-reason is-note', 'data-tone': catalogue.tone === 'warn' ? 'warn' : null, text: catalogue.text }) : null),
        h('div', { class: 'ab-profile-actions' },
          confirming ? [
            h('span', { class: 'ab-confirm-text', text: pick('移除？文件会改名保留', 'Remove? File will be renamed and kept') }),
            h('button', { type: 'button', class: 'ab-btn ab-btn--s ab-btn--danger', 'data-focus-key': `confirm-${profile.id}`, disabled: s.removeBusy, onclick: () => removeProfile(profile.id) }, s.removeBusy ? pick('移除中…', 'Removing…') : pick('确认移除', 'Confirm')),
            h('button', { type: 'button', class: 'ab-btn ab-btn--s', 'data-focus-key': `cancel-${profile.id}`, onclick: () => { s.removing = null; render(); } }, pick('取消', 'Cancel')),
          ] : [
            h('button', { type: 'button', class: 'ab-btn ab-btn--s ab-btn--ghost', 'data-focus-key': `edit-${profile.id}`, disabled: readOnly, onclick: () => enterForm(profile), 'aria-label': pick(`编辑 ${displayName}`, `Edit ${displayName}`) }, icon('edit'), pick('编辑', 'Edit')),
            h('button', { type: 'button', class: 'ab-btn ab-btn--s ab-btn--ghost', 'data-focus-key': `remove-${profile.id}`, disabled: readOnly || inUse,
              title: inUse ? pick('当前 run 正在使用，不能移除', 'In use by active run; cannot remove') : null, 'aria-label': pick(`移除 ${displayName}`, `Remove ${displayName}`), onclick: () => { s.removing = profile.id; s.formMessage = null; render(); } }, icon('trash'), pick('移除', 'Remove')),
            inUse ? h('span', { class: 'ab-opt-reason is-note', text: pick('使用中，不能移除', 'In use; cannot remove') }) : null,
          ])));
    }
    section.append(list);
    return section;
  }

  function segmented(name, label, choices, value, onChange, { disabled = false } = {}) {
    const group = h('div', { class: 'ab-seg', role: 'radiogroup', 'aria-label': label });
    let selected = choices.some((item) => item.value === value) ? value : choices[0]?.value;
    const sync = () => buttons.forEach((button, index) => { const on = choices[index].value === selected; button.setAttribute('aria-checked', String(on)); button.tabIndex = on ? 0 : -1; });
    const buttons = choices.map((choice) => h('button', { type: 'button', role: 'radio', class: 'ab-seg-btn', disabled: disabled || choice.disabled, 'data-focus-key': `${name}-${choice.value}`,
      onclick: () => { if (choice.value !== selected) { selected = choice.value; sync(); onChange(choice.value); } } }, choice.label));
    sync();
    group.addEventListener('keydown', (event) => {
      const index = buttons.indexOf(document.activeElement);
      if (index < 0) return;
      const delta = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[event.key];
      if (!delta) return;
      event.preventDefault();
      const next = buttons[(index + delta + buttons.length) % buttons.length];
      next.focus();
      next.click();
    });
    group.append(...buttons);
    return group;
  }

  function advancedSection(o, d) {
    const section = h('section', { class: 'ab-section', 'aria-labelledby': 'ab-adv-title' },
      h('div', { class: 'ab-section-head' }, h('h4', { id: 'ab-adv-title', text: pick('高级', 'Advanced') }), h('span', { class: 'ab-tag', text: pick('需要新会话', 'Requires new session') })));
    const wires = sharedWires(o, d);
    const thinking = thinkingControl(o);
    const readOnly = leaseOther() || o.source === 'legacy';
    let any = false;
    if (d.backend === 'opencode' && wires.length > 1) {
      any = true;
      const wireLabel = pick('接口 wire', 'Wire protocol');
      section.append(h('div', { class: 'ab-field' }, h('span', { class: 'ab-field-label', text: wireLabel }),
        segmented('wire', wireLabel, wires.map((wire) => ({ value: wire, label: wire })), d.wire_api, (wire) => setDraft({ ...base(o), wire_api: wire }), { disabled: readOnly }),
        h('p', { class: 'ab-help', text: pick('默认使用所选模型列出的接口（anthropic_messages 优先）。', 'Defaults to the wire listed for the selected model (anthropic_messages preferred).') })));
    }
    if (thinking?.available) {
      any = true;
      const choices = M.thinkingChoices(thinking);
      const help = M.profileHelp(o.profileSchema).thinking_display;
      const tdLabel = pick('思考显示 thinking_display', 'Thinking display (thinking_display)');
      section.append(h('div', { class: 'ab-field' }, h('span', { class: 'ab-field-label', text: tdLabel }),
        segmented('thinking', tdLabel, choices, d.thinking_display ?? null, (value) => setDraft({ ...base(o), thinking_display: value }), { disabled: readOnly }),
        h('p', { class: 'ab-help', text: [trServer(thinking.applies_label) || M.scopeLabel(thinking.applies), help?.text].filter(Boolean).join(' · ') })));
    }
    if (!any) {
      section.append(h('p', { class: 'ab-empty', text: o.source === 'legacy'
        ? pick('当前网关版本没有高级会话选项。', 'No advanced session options on the current gateway.')
        : pick(`${M.backendLabel(d.backend)} 在当前接口上没有额外的会话选项。`, `${M.backendLabel(d.backend)} has no extra session options on the current wire.`) }));
    }
    return section;
  }

  // Profile 表单：每次打开仅构建一次，不打断用户输入；字段说明直接消费网关 profile_schema.fields。
  let formNodes = null;
  function enterForm(profile) {
    s.settingsView = 'form';
    s.formMode = profile ? 'edit' : 'new';
    s.form = profile ? M.profileFormFromInfo(profile) : M.emptyProfileForm();
    s.formKeyStored = profile?.key_present === true;
    s.formFingerprint = profile ? M.keyStatus(profile) : null;
    s.formErrors = {}; s.formWarnings = {}; s.formBusy = false; s.formMessage = null;
    settingsSignature = null;
    buildForm();
    render();
    requestAnimationFrame(() => (s.formMode === 'edit' ? formNodes.inputs.api_key : formNodes.inputs.name).focus({ preventScroll: true }));
  }
  function leaveForm() {
    if (formNodes?.inputs.api_key) formNodes.inputs.api_key.value = '';
    s.form = null; formNodes = null; s.settingsView = 'list'; settingsSignature = null;
    render();
    requestAnimationFrame(() => popBody.querySelector('[data-focus-key="new"]')?.focus({ preventScroll: true }));
  }
  function buildForm() {
    const form = s.form;
    const inputs = {}, errors = {};
    const schema = options().profileSchema;
    const help = M.profileHelp(schema);
    const choices = M.profileChoices(schema);
    const helpNode = (key, id = null, extra = null) => {
      const entry = key === 'id' ? { field: null, text: M.profileIdHelp() } : help[key];
      if (!entry && !extra) return null;
      return h('p', { class: 'ab-help', id }, entry?.field ? h('code', { class: 'ab-help-field', text: entry.field }) : null, extra, entry?.text || null);
    };
    const field = (key, label, control, { wide = true, extra = null } = {}) => {
      errors[key] = h('p', { class: 'ab-field-error', id: `ab-err-${key}`, role: 'alert' });
      control.setAttribute('aria-describedby', `ab-help-${key} ab-err-${key}`);
      control.addEventListener('input', () => clearFieldError(key));
      return h('div', { class: `ab-field${wide ? '' : ' ab-field--half'}` }, h('label', { class: 'ab-field-label', for: control.id, text: label }), control,
        helpNode(key, `ab-help-${key}`, extra), errors[key]);
    };
    const text = (key, attrs = {}) => {
      const input = h('input', { id: `ab-f-${key}`, type: 'text', autocomplete: 'off', spellcheck: 'false', value: form[key] ?? '', ...attrs });
      input.addEventListener('input', () => { form[key] = input.value; if (key === 'name' && !form.idEdited && s.formMode === 'new') { form.id = M.slugify(input.value); inputs.id.value = form.id; } if (key === 'id') form.idEdited = true; });
      inputs[key] = input;
      return input;
    };
    const nameInput = text('name', { maxlength: 60, placeholder: pick('例如 Example Gateway', 'e.g. Example Gateway') });
    const idInput = text('id', { maxlength: 40, placeholder: 'example-gateway', readonly: s.formMode === 'edit' ? true : null, class: 'ab-mono' });
    const key = h('input', { id: 'ab-f-api_key', type: 'password', autocomplete: 'new-password', spellcheck: 'false', 'data-lpignore': 'true',
      placeholder: s.formKeyStored ? pick('已保存密钥 · 留空则保留', 'Key saved · leave empty to keep') : pick('粘贴供应商密钥', 'Paste provider API key') });
    key.addEventListener('input', () => { form.api_key = key.value; });
    inputs.api_key = key;
    const wireRows = M.WIRE_ORDER.filter((wire) => choices.wires.includes(wire)).map((wire) => {
      const entry = form.wires[wire];
      const enabled = h('input', { type: 'checkbox', id: `ab-f-wire-${wire}`, checked: entry.enabled });
      const url = h('input', { type: 'url', id: `ab-f-url-${wire}`, class: 'ab-mono', autocomplete: 'off', spellcheck: 'false', value: entry.base_url,
        placeholder: wire === 'anthropic_messages' ? 'https://gateway.example/anthropic' : 'https://gateway.example/v1', 'aria-label': `${wire} base_url` });
      const listing = h('input', { type: 'url', id: `ab-f-models-${wire}`, class: 'ab-mono', autocomplete: 'off', spellcheck: 'false', value: entry.models_url || '',
        placeholder: pick('models_url（可选，留空 = 由 base_url 推导）', 'models_url (optional; empty = derived from base_url)'), 'aria-label': `${wire} models_url` });
      const scheme = segmented(`auth-${wire}`, `${wire} auth_scheme`, choices.auth_schemes.map((value) => ({ value, label: value })), entry.auth_scheme, (value) => { entry.auth_scheme = value; });
      const sync = () => { entry.enabled = enabled.checked; url.disabled = !enabled.checked; listing.disabled = !enabled.checked; for (const button of scheme.querySelectorAll('button')) button.disabled = !enabled.checked; row.dataset.enabled = String(enabled.checked); };
      enabled.addEventListener('change', () => { sync(); clearFieldError(`wire.${wire}`); clearFieldError(`models.${wire}`); clearFieldError('wires'); });
      url.addEventListener('input', () => { entry.base_url = url.value; clearFieldError(`wire.${wire}`); });
      listing.addEventListener('input', () => { entry.models_url = listing.value; clearFieldError(`models.${wire}`); });
      errors[`wire.${wire}`] = h('p', { class: 'ab-field-error', role: 'alert' });
      errors[`models.${wire}`] = h('p', { class: 'ab-field-error', role: 'alert' });
      const row = h('div', { class: 'ab-wire-row' }, h('label', { class: 'ab-check', for: enabled.id }, enabled, h('span', { class: 'ab-mono', text: wire })), scheme, url, errors[`wire.${wire}`], listing, errors[`models.${wire}`]);
      sync();
      inputs[`wire.${wire}`] = url;
      inputs[`models.${wire}`] = listing;
      return row;
    });
    errors.wires = h('p', { class: 'ab-field-error', role: 'alert' });
    const depth = h('input', { id: 'ab-f-depth', type: 'number', min: 3, step: 1, inputmode: 'numeric', value: form.compat.tool_schema_max_depth, placeholder: pick('不限制', 'No limit') });
    depth.addEventListener('input', () => { form.compat.tool_schema_max_depth = depth.value; });
    inputs.tool_schema_max_depth = depth;
    const images = segmented('images', 'compat.tool_result_images', choices.tool_result_images.map((value) => ({ value, label: value })), form.compat.tool_result_images, (value) => { form.compat.tool_result_images = value; });
    const env = h('textarea', { id: 'ab-f-env', rows: 3, class: 'ab-mono', spellcheck: 'false', placeholder: 'CLAUDE_CODE_MAX_OUTPUT_TOKENS=32000' });
    env.value = form.claude_env;
    env.addEventListener('input', () => { form.claude_env = env.value; });
    inputs.claude_env = env;
    const defaultThinkingText = pick('默认（不发送）', 'Default (not sent)');
    const thinking = segmented('form-thinking', 'backends.opencode.thinking_display', M.thinkingChoices({ choices: choices.thinking_display }).map((item) => ({ value: item.value ?? '', label: item.value == null ? defaultThinkingText : item.label })),
      form.thinking_display, (value) => { form.thinking_display = value; });
    const effortSelect = h('select', { id: 'ab-f-default_effort', class: 'ab-mono' }, h('option', { value: '', text: pick('不预填', 'None') }),
      [...new Set([...choices.efforts, ...(form.default_effort && !choices.efforts.includes(form.default_effort) ? [form.default_effort] : [])])].map((value) => h('option', { value, text: value })));
    effortSelect.value = form.default_effort || '';
    effortSelect.addEventListener('change', () => { form.default_effort = effortSelect.value; clearFieldError('default_effort'); });
    inputs.default_effort = effortSelect;
    const message = h('p', { class: 'ab-form-msg', role: 'status', 'aria-live': 'polite' });
    const save = h('button', { type: 'submit', class: 'ab-btn ab-btn--primary' }, pick('保存 profile', 'Save profile'));
    const groupHelp = (keys) => keys.map((item) => helpNode(item)).filter(Boolean);
    const element = h('form', { class: 'ab-form', novalidate: true },
      h('button', { type: 'button', class: 'ab-link ab-back', onclick: () => leaveForm() }, icon('back'), pick('返回 profile 列表', 'Back to profiles')),
      h('div', { class: 'ab-grid' }, field('name', pick('名称', 'Name'), nameInput, { wide: false }), field('id', 'ID', idInput, { wide: false })),
      field('api_key', pick('密钥', 'API key'), key, { extra: s.formKeyStored ? `${s.formFingerprint}${pick('。', '. ')}` : null }),
      h('fieldset', { class: 'ab-fieldset' }, h('legend', { class: 'ab-field-label', text: pick('接口与地址', 'Wires & Endpoints') }), groupHelp(['wires']), wireRows,
        h('details', { class: 'ab-details ab-details--help' }, h('summary', { text: pick('base_url / auth_scheme / models_url 的约定', 'base_url / auth_scheme / models_url conventions') }), groupHelp(['base_url', 'auth_scheme', 'models_url'])), errors.wires),
      h('fieldset', { class: 'ab-fieldset' }, h('legend', { class: 'ab-field-label', text: pick('兼容选项', 'Compatibility Options') }),
        h('div', { class: 'ab-field' }, h('span', { class: 'ab-field-label', text: pick('工具结果中的图像', 'Images in tool results') }), images, helpNode('tool_result_images')),
        field('tool_schema_max_depth', pick('schema 最大嵌套深度', 'Max tool schema nesting depth'), depth)),
      h('div', { class: 'ab-grid' }, field('default_model', pick('默认模型（可选）', 'Default model (optional)'), text('default_model', { class: 'ab-mono', placeholder: 'qwen3.8-flash' }), { wide: false }),
        field('default_effort', pick('默认推理强度（可选）', 'Default effort (optional)'), effortSelect, { wide: false })),
      h('details', { class: 'ab-details' }, h('summary', { text: pick('高级：Claude Code 环境变量、OpenCode 思考显示', 'Advanced: Claude Code env & OpenCode thinking display') }),
        field('claude_env', pick('Claude Code 环境变量（每行 NAME=VALUE）', 'Claude Code environment variables (one NAME=VALUE per line)'), env),
        h('div', { class: 'ab-field' }, h('span', { class: 'ab-field-label', text: pick('OpenCode 思考显示', 'OpenCode thinking display') }), thinking, helpNode('thinking_display'))),
      message,
      h('div', { class: 'ab-form-actions' }, h('button', { type: 'button', class: 'ab-btn', text: pick('取消', 'Cancel'), onclick: () => leaveForm() }), save));
    element.addEventListener('submit', (event) => { event.preventDefault(); submitProfile(); });
    popBody.replaceChildren(element);
    formNodes = { element, inputs, errors, message, save };
  }
  function clearFieldError(key) {
    if (!s.formErrors[key] && !s.formWarnings[key]) return;
    s.formErrors = { ...s.formErrors, [key]: undefined };
    s.formWarnings = { ...s.formWarnings, [key]: undefined };
    if (s.formMessage?.tone === 'bad' && !Object.values(s.formErrors).some(Boolean)) s.formMessage = null;
    paintFormState();
  }
  function paintFormState() {
    if (!formNodes) return;
    for (const [key, node] of Object.entries(formNodes.errors)) setText(node, s.formErrors[key] || s.formWarnings[key] || '');
    for (const [key, node] of Object.entries(formNodes.errors)) node.dataset.tone = s.formErrors[key] ? 'bad' : s.formWarnings[key] ? 'warn' : '';
    for (const [key, input] of Object.entries(formNodes.inputs)) setAttr(input, 'aria-invalid', s.formErrors[key] ? 'true' : null);
    setText(formNodes.message, s.formMessage?.text || '');
    formNodes.message.dataset.tone = s.formMessage?.tone || '';
    formNodes.save.disabled = s.formBusy || leaseOther();
    setText(formNodes.save, s.formBusy ? pick('正在保存…', 'Saving…') : pick('保存 profile', 'Save profile'));
  }
  async function submitProfile() {
    const request = M.profileRequest(s.form, { editing: s.formMode === 'edit', keyStored: s.formKeyStored });
    s.formErrors = request.errors; s.formWarnings = request.warnings; s.formMessage = null;
    if (!request.ok) {
      s.formMessage = { tone: 'bad', text: pick('请先修正标出的字段', 'Fix the highlighted fields first') };
      paintFormState();
      const first = Object.keys(request.errors)[0];
      (formNodes.inputs[first] || formNodes.inputs[first === 'wires' ? 'wire.anthropic_messages' : 'name'])?.focus();
      return;
    }
    if (!(await api.ensureLease())) { s.formMessage = { tone: 'bad', text: pick('另一窗口持有控制，未保存', 'Another window holds control; not saved') }; paintFormState(); return; }
    s.formBusy = true; paintFormState();
    try {
      const info = await api.getJSON('/v1/console/profiles', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...api.credentials(), ...request.body }) });
      formNodes.inputs.api_key.value = '';
      s.form.api_key = '';
      const id = info?.id || request.body.id;
      s.formBusy = false;
      s.formMessage = {
        tone: 'ok',
        text: pick(
          `已保存 ${info?.name || id} · ${M.keyStatus(info || { key_present: true })}（密钥不会再显示）`,
          `Saved ${info?.name || id} · ${M.keyStatus(info || { key_present: true })} (key will not be shown again)`,
        ),
      };
      s.form = null; formNodes = null; s.settingsView = 'list'; settingsSignature = null;
      await refreshOptions();
      requestAnimationFrame(() => popBody.querySelector(`[data-focus-key="edit-${CSS.escape(id)}"]`)?.focus({ preventScroll: true }));
    } catch (error) {
      s.formBusy = false;
      const errMsg = trServer(error.body?.message || error.body?.error || error.message, error.body?.error);
      s.formMessage = { tone: 'bad', text: pick(`未保存：${errMsg}`, `Not saved: ${errMsg}`) };
      paintFormState();
    }
  }
  async function removeProfile(id) {
    if (!(await api.ensureLease())) { s.formMessage = { tone: 'bad', text: pick('另一窗口持有控制，未移除', 'Another window holds control; not removed') }; render(); return; }
    s.removeBusy = true; render();
    try {
      await api.getJSON(`/v1/console/profiles/${encodeURIComponent(id)}/remove`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(api.credentials()) });
      s.formMessage = { tone: 'ok', text: pick(`已移除 ${id}（文件已改名保留）`, `Removed ${id} (file renamed and kept)`) };
    } catch (error) {
      const errMsg = trServer(error.body?.message || error.body?.error || error.message, error.body?.error);
      s.formMessage = { tone: 'bad', text: pick(`未移除 ${id}：${errMsg}`, `Failed to remove ${id}: ${errMsg}`) };
    }
    s.removeBusy = false; s.removing = null;
    await refreshOptions();
    requestAnimationFrame(() => popBody.querySelector('[data-focus-key="new"]')?.focus({ preventScroll: true }));
  }

  // ---------- 会话切换（handoff） ----------
  function populateHandoffDialog() {
    const o = options(), d = s.draft;
    if (!d) return;
    const diff = M.sessionDiff(o.current, d, { effortScope: effortScope(o) });
    dialogDiff.replaceChildren(...diffItems(o, d, diffFields(diff)));
    const runtime = s.consoleState?.settings?.runtime?.speed_scale_cap;
    const cap = runtime?.value ?? null, deflt = runtime?.default ?? null;
    const torque = o.handoff.disable_torque_on_disconnect;
    dialogFacts.replaceChildren(
      h('li', {}, h('strong', { text: pick('当前 run', 'Current run') }), pick(
        ` ${s.run?.run_id || '—'} 会收尾并归档（summary 记录后继 run），随后在本页启动新 run。`,
        ` ${s.run?.run_id || '—'} will finalize and archive (recording the successor run in summary) before starting the new run on this page.`,
      )),
      h('li', {}, h('strong', { text: pick('机械臂', 'Robot arm') }), torque === true
        ? pick(' 网关断开时部署会松力矩：请先托住机械臂。', ' Deployment disables torque on disconnect: support the arm first.')
        : torque === false
          ? pick(' 保持力矩，切换本身不发出任何运动。', ' Holds torque; switching issues no motion.')
          : pick(' 切换本身不发出任何运动；断开时的力矩行为取决于部署配置。', ' Switching issues no motion; disconnect torque behavior follows deployment config.')),
      h('li', {}, h('strong', { text: pick('沿用', 'Carried over') }),
        cap != null
          ? pick(
            ` 速度上限 ${Number(cap).toFixed(2)}${deflt != null && cap < deflt ? '（已调低，会在新 run 中重新应用）' : '（部署默认）'}；`,
            ` Speed cap ${Number(cap).toFixed(2)}${deflt != null && cap < deflt ? ' (lowered; reapplied in new run)' : ' (deployment default)'}; `,
          )
          : pick(' 速度上限；', ' Speed cap; '),
        pick('不沿用观察、task_plan 和对话历史。', 'observations, task_plan, and conversation history are not carried over.')),
    );
  }
  function openHandoffDialog() {
    if (!s.draft) return;
    closePop({ restoreFocus: false });
    populateHandoffDialog();
    setText(dialogError, '');
    dialogConfirm.disabled = false;
    setText(dialogConfirm, pick('切换并新建 run', 'Switch & New Run'));
    dialogScrim.hidden = false;
    dialog.show();
    dialogPrompt.focus({ preventScroll: true });
    requestAnimationFrame(() => { dialog.dataset.state = 'open'; });
  }
  function closeHandoffDialog({ restoreFocus = true } = {}) {
    if (!dialog.open) return;
    dialog.dataset.state = 'closed';
    dialogScrim.hidden = true;
    dialog.close();
    if (restoreFocus) (switchButton.isConnected && !root.querySelector('.ab-tray[inert]') ? switchButton : chips.backend).focus({ preventScroll: true });
  }
  const diffFields = (diff) => ['backend', 'profile', 'model', 'effort', 'wire_api', 'thinking_display'].filter((field) => diff.fields.includes(field) || ['backend', 'profile', 'model'].includes(field));
  function diffItems(o, d, fields) {
    const name = {
      backend: M.backendLabel,
      profile: (id) => (id == null ? pick('配置文件', 'Config file') : o.profiles.find((item) => item.id === id)?.name || id),
    };
    return fields.map((field) => {
      const from = o.current[field] ?? null, to = d[field] ?? null;
      const show = (value) => (name[field] ? name[field](value) : value == null ? (field === 'wire_api' ? pick('待定', 'Pending') : pick('默认', 'Default')) : String(value));
      const changed = from !== to;
      return h('li', { class: 'ab-diff-item', 'data-changed': String(changed) }, h('span', { class: 'ab-diff-key', text: M.fieldLabel(field) }),
        changed ? [h('span', { class: 'ab-diff-from', text: show(from) }), icon('arrow', 'ab-ico ab-ico--s'), h('span', { class: 'ab-diff-to', text: to == null && field === 'profile' ? pick('请选择', 'Select') : show(to) })]
          : h('span', { class: 'ab-diff-same', text: show(to) }));
    });
  }
  async function confirmHandoff() {
    const d = s.draft;
    if (!d) return;
    if (!(await api.ensureLease())) { setText(dialogError, pick('另一窗口持有控制，未切换', 'Another window holds control; did not switch')); return; }
    dialogConfirm.disabled = true;
    setText(dialogConfirm, pick('正在提交…', 'Submitting…'));
    setText(dialogError, '');
    try {
      const receipt = await api.control('handoff_run', M.handoffBody(d, dialogPrompt.value), { key: 'agent-handoff' });
      if (receipt.status !== 'handoff_accepted' || !receipt.successor_run_id) throw Object.assign(new Error(receipt.message || receipt.reason || `handoff_${receipt.status || 'unconfirmed'}`), { status: 400, body: receipt });
      dialogPrompt.value = '';
      closeHandoffDialog({ restoreFocus: false });
      s.draft = null; s.handoffNote = null; s.handoffFields = null; s.check = { key: null, state: 'idle', result: null }; s.intents = {};
      startHandoff(receipt.successor_run_id);
    } catch (error) {
      const unknown = !error.status || error.status >= 500 || (error.status === 409 && error.body?.status === 'unknown');
      const errMsg = trServer(error.body?.message || error.body?.error || error.message, error.body?.error);
      setText(dialogError, unknown
        ? pick(
          `未确认：网关没有回执（${error.body?.reason || error.message}）。再次点击会沿用同一 request_id，不会重复切换。`,
          `Unconfirmed: no gateway receipt (${error.body?.reason || error.message}). Clicking again reuses request_id without duplicate handoff.`,
        )
        : pick(`未切换：${errMsg}`, `Did not switch: ${errMsg}`));
      dialogConfirm.disabled = false;
      setText(dialogConfirm, unknown ? pick('重试（同一 request_id）', 'Retry (same request_id)') : pick('切换并新建 run', 'Switch & New Run'));
    }
  }
  let handoffTimer = 0;
  function startHandoff(successorRunId) {
    s.handoff = { fromRunId: s.run?.run_id || null, successorRunId, startedAt: Date.now(), sawDown: false, reachable: true, runId: s.run?.run_id || null };
    overlay.hidden = false;
    overlayActions.hidden = true;
    requestAnimationFrame(() => { overlay.dataset.state = 'open'; });
    paintOverlay();
    pollHandoff();
  }
  async function pollHandoff() {
    clearTimeout(handoffTimer);
    const handoff = s.handoff;
    if (!handoff || handoff.done) return;
    try {
      const run = await api.getJSON('/v1/run');
      handoff.reachable = true; handoff.runId = run.run_id;
      handoff.run = run;
    } catch {
      handoff.reachable = false; handoff.sawDown = true;
    }
    if (s.handoff !== handoff || handoff.done) return;
    const progress = M.handoffProgress(handoff, { runId: handoff.runId, reachable: handoff.reachable });
    handoff.phase = progress.phase; handoff.step = progress.step;
    paintOverlay();
    if (progress.phase === 'ready') return finishHandoff(true);
    if (progress.phase === 'fallback') return finishHandoff(false);
    handoffTimer = setTimeout(pollHandoff, 1000);
  }
  async function finishHandoff(success) {
    const handoff = s.handoff;
    handoff.done = true;
    clearTimeout(handoffTimer);
    await api.refreshConsole?.();
    await refreshOptions();
    if (success) {
      showToast(pick(`已切换到新 run ${handoff.successorRunId}`, `Switched to new run ${handoff.successorRunId}`));
      s.banner = null;
    } else {
      const reason = trServer(M.fallbackReason(handoff.run, s.raw)) || pick('原因未记录', 'reason unrecorded');
      s.banner = pick(
        `新 run ${handoff.successorRunId} 没能启动，已用原来的设置重新启动（run ${handoff.runId}）：${reason}`,
        `New run ${handoff.successorRunId} failed to start; restarted with previous settings (run ${handoff.runId}): ${reason}`,
      );
    }
    setTimeout(() => {
      overlay.dataset.state = 'closed';
      setTimeout(() => { if (s.handoff === handoff) { s.handoff = null; overlay.hidden = true; } render(); }, reducedMotion.matches ? 0 : 200);
    }, success ? 500 : 0);
    render();
  }
  function paintOverlay() {
    const handoff = s.handoff;
    if (!handoff) return;
    const elapsed = Math.round((Date.now() - handoff.startedAt) / 1000);
    overlaySteps.forEach((node, index) => { node.dataset.state = index + 1 < handoff.step || handoff.phase === 'ready' ? 'done' : index + 1 === handoff.step ? 'active' : 'todo'; });
    const notes = {
      finalizing: pick('正在收尾当前 run：关闭动作准入、导出录像、关闭会话与存储。', 'Finalizing current run: closing action admission, exporting videos, and closing session & store.'),
      starting: handoff.reachable
        ? pick('新 run 正在启动…', 'Starting new run…')
        : pick('网关暂时不可达，这是切换中的正常现象；页面会继续等待。', 'Gateway is momentarily unreachable (expected during handoff); continuing to wait.'),
      ready: pick('新 run 已就绪。', 'New run is ready.'),
      fallback: pick('新 run 没能启动，网关已用原来的设置重新启动。', 'New run failed to start; gateway restarted with previous settings.'),
      timeout: pick('网关未响应，请查看终端。页面不会自动采取任何操作，仍在继续等待。', 'Gateway has not responded; check the terminal. Taking no automatic action and continuing to wait.'),
    };
    setText(overlayNote, notes[handoff.phase || 'finalizing']);
    overlayNote.dataset.tone = handoff.phase === 'timeout' ? 'bad' : '';
    setText(overlayElapsed, pick(`已用 ${elapsed} s · 新 run ${handoff.successorRunId}`, `Elapsed ${elapsed} s · new run ${handoff.successorRunId}`));
    overlayActions.hidden = handoff.phase !== 'timeout';
  }
  function showToast(text) {
    setText(toast, text);
    toast.hidden = false;
    requestAnimationFrame(() => { toast.dataset.state = 'open'; });
    clearTimeout(showToast.timer);
    showToast.timer = setTimeout(() => { toast.dataset.state = 'closed'; setTimeout(() => { toast.hidden = true; }, 220); }, 4200);
  }

  // ---------- 主渲染 ----------
  let frame = 0;
  function render() { if (!frame) frame = requestAnimationFrame(() => { frame = 0; paint(); }); }
  function paintChip(field, { value, meta = null, scope = '', note = false, state = 'idle', changed = false, hidden = false, disabled = false, label }) {
    const node = chips[field];
    node.hidden = hidden;
    node.disabled = disabled;
    setText(node.querySelector('.ab-chip-value'), value);
    setText(node.querySelector('.ab-chip-scope'), scope);
    const metaNode = node.querySelector('.ab-chip-meta');
    const metaKey = JSON.stringify([getLocale(), meta]);
    if (metaNode.dataset.key !== metaKey) { metaNode.dataset.key = metaKey; metaNode.replaceChildren(...(meta || []).map(badgeNode)); }
    setAttr(node, 'data-state', state);
    setAttr(node, 'data-changed', changed ? 'true' : null);
    setAttr(node, 'data-note', note ? 'true' : null);
    setAttr(node, 'aria-label', label);
  }
  function paintExternal(view) {
    const on = Boolean(view);
    chipsRow.hidden = on; statusRow.hidden = on; extCard.hidden = !on;
    if (on) { root.dataset.mode = 'external'; root.dataset.readonly = 'true'; delete root.dataset.loading; setAttr(root, 'aria-busy', null); }
    else if (root.dataset.mode === 'external') root.dataset.mode = 'live';
    if (!on) return;
    const key = `${getLocale()}:${JSON.stringify(view)}`;
    if (extCard.dataset.key === key) return;
    extCard.dataset.key = key;
    extCard.dataset.waiting = String(view.waiting);
    extCard.dataset.replay = String(view.replay);
    extDot.dataset.tone = view.lastCallAt && !view.replay ? 'ok' : 'muted';
    setText(extState, M.externalStateText(view));
    setText(extClient, view.waiting ? pick('尚无 initialize', 'No initialize yet') : M.externalClientRowText(view));
    setText(extLast, M.lastCallText(view.lastCallAt) ?? pick('尚无工具调用', 'No tool calls yet'));
    setAttr(extLast, 'title', view.lastCallAt);
    setText(extEndpoint, view.endpoint ?? (view.replay ? pick('—（回放，没有 MCP 服务）', '— (replay, no MCP server)') : pick('—（MCP 服务尚未启动）', '— (MCP server not started yet)')));
    extCopy.hidden = !view.endpoint;
  }
  function paint() {
    const external = M.externalView(s.consoleState, s.raw, s.run);
    paintExternal(external);
    if (external) { if (s.open) closePop({ restoreFocus: false }); return; }
    const o = options();
    const d = display(o);
    const current = o.current;
    const other = leaseOther();
    const diff = M.sessionDiff(current, s.draft, { effortScope: effortScope(o) });
    const changed = (field) => Boolean(s.draft) && diff.fields.includes(field);
    const stateOf = (field) => (s.draft && changed(field) ? 'draft' : M.displayValue(current[field], s.intents[field]).state);
    const STATE_WORDS = isEn()
      ? { pending: ', applying', confirmed: ', applied', unknown: ', unconfirmed', draft: ', staged' }
      : { pending: '，正在应用', confirmed: '，已应用', unknown: '，未确认', draft: '，待切换' };
    const newSessionShort = pick('新会话', 'New session');
    const nextTurnShort = pick('下一轮', 'Next turn');

    const backendText = M.backendLabel(d.backend || current.backend);
    paintChip('backend', {
      value: backendText, scope: newSessionShort, state: stateOf('backend'), changed: changed('backend'), disabled: other,
      label: pick(`后端：${backendText}（切换需要新会话${STATE_WORDS[stateOf('backend')] || ''}）`, `Backend: ${backendText} (requires new session${STATE_WORDS[stateOf('backend')] || ''})`),
    });
    const profile = o.profiles.find((item) => item.id === d.profile);
    const profileText = d.profile == null
      ? (s.draft && changed('profile') ? pick('请选择', 'Select') : o.source === 'legacy' ? pick('配置文件', 'Config file') : pick('配置文件绑定', 'Config-bound'))
      : profile?.name || d.profile;
    paintChip('profile', {
      value: profileText, scope: newSessionShort, state: stateOf('profile'), changed: changed('profile'), disabled: other,
      label: pick(`供应商 Profile：${profileText}（切换需要新会话${STATE_WORDS[stateOf('profile')] || ''}）`, `Provider Profile: ${profileText} (requires new session${STATE_WORDS[stateOf('profile')] || ''})`),
    });
    const catalogue = catalogueFor(o, d);
    const entry = (catalogue?.models || []).find((model) => model.id === d.model);
    const modelMeta = entry ? M.modelBadges(entry).filter((badge) => badge.kind === 'image' || badge.kind === 'context') : null;
    const modelState = stateOf('model');
    paintChip('model', {
      value: d.model || '—', meta: modelMeta, scope: s.draft ? newSessionShort : nextTurnShort, state: modelState, changed: changed('model'), disabled: other,
      label: pick(
        `模型：${d.model || '未设置'}（${s.draft ? '随新会话生效' : '下一轮生效'}${STATE_WORDS[modelState] || ''}）`,
        `Model: ${d.model || 'unset'} (${s.draft ? 'applies in new session' : 'applies next turn'}${STATE_WORDS[modelState] || ''})`,
      ),
    });
    if (s.open !== 'model' && entry === undefined && d.profile && d.wire_api && o.source !== 'legacy' && !s.catalogues.has(`${d.profile}|${d.wire_api}`)) loadCatalogue(d.profile, d.wire_api);

    const effort = effortControl(o);
    const effortSession = o.source !== 'legacy' && (Boolean(s.draft) || effortScope(o) === 'next_session');
    const effortState = stateOf('effort');
    const nextEffort = !s.draft && current.effort_next_session && current.effort_next_session !== d.effort ? current.effort_next_session : null;
    const defaultWord = pick('默认', 'Default');
    paintChip('effort', {
      value: d.effort || defaultWord,
      scope: nextEffort ? pick(`下次 ${nextEffort}`, `Next ${nextEffort}`) : effortSession ? newSessionShort : M.scopeShort(effort?.applies) || nextTurnShort,
      state: effortState, changed: changed('effort'),
      note: Boolean(nextEffort),
      hidden: !effort, disabled: other || !(effort?.choices || []).length,
      label: pick(
        `推理强度：${d.effort || '默认'}（${effortSession ? '需要新会话' : M.scopeLabel(effort?.applies) || '下一轮生效'}${STATE_WORDS[effortState] || ''}${nextEffort ? `；下次会话 ${nextEffort}` : ''}）`,
        `Effort: ${d.effort || 'default'} (${effortSession ? 'requires new session' : M.scopeLabel(effort?.applies) || 'applies next turn'}${STATE_WORDS[effortState] || ''}${nextEffort ? `; next session ${nextEffort}` : ''})`,
      ),
    });
    const tier = o.reasoning?.service_tier;
    const tierState = stateOf('service_tier');
    const tierText = M.tierLabel(d.service_tier);
    const served = tierState === 'idle' || tierState === 'confirmed' ? M.servedTier(o, d.service_tier) : { state: 'none' };
    paintChip('speed', {
      value: tierText, state: tierState, hidden: !tier?.available || Boolean(s.draft && s.draft.backend !== current.backend), disabled: other,
      scope: served.state === 'mismatch' ? pick(`实际 ${M.tierLabel(served.served)}`, `Served ${M.tierLabel(served.served)}`) : pick('下次请求', 'Next req'),
      note: served.state === 'mismatch',
      label: pick(
        `速度：${tierText}（下一次模型请求生效${STATE_WORDS[tierState] || ''}${served.text ? `；${served.text}` : ''}）`,
        `Speed: ${tierText} (applies next model request${STATE_WORDS[tierState] || ''}${served.text ? `; ${served.text}` : ''})`,
      ),
    });
    paintChip('settings', { value: pick('设置', 'Settings'), scope: '', state: 'idle', disabled: false, label: pick('Profile 管理与高级设置', 'Manage profiles & advanced settings') });
    root.dataset.readonly = other ? 'true' : 'false';
    root.dataset.mode = s.draft ? 'draft' : 'live';
    banner.hidden = !s.banner;
    const bannerSig = `${getLocale()}:${s.banner || ''}`;
    if (s.banner && banner.dataset.text !== bannerSig) {
      banner.dataset.text = bannerSig;
      banner.replaceChildren(icon('warning'), h('span', { text: s.banner }), h('button', { type: 'button', class: 'ab-icon-btn', 'aria-label': pick('关闭提示', 'Dismiss'), onclick: () => { s.banner = null; render(); } }, icon('close')));
    }
    const booting = s.rawStatus === null || !s.consoleState;
    if (booting) root.dataset.loading = 'true'; else delete root.dataset.loading;
    setAttr(root, 'aria-busy', booting ? 'true' : null);
    paintTray(o, d, diff);
    paintStatus(o, d, other);
    paintPop(o, d);
    if (s.handoff) paintOverlay();
  }
  function paintTray(o, d, diff) {
    const open = Boolean(s.draft) && diff.handoff;
    tray.dataset.open = String(open);
    if (open) tray.removeAttribute('inert'); else tray.setAttribute('inert', '');
    if (!open) return;
    const fields = diffFields(diff);
    const key = JSON.stringify([getLocale(), fields, s.draft, o.current, o.profiles.map((item) => [item.id, item.name])]);
    if (trayDiff.dataset.key !== key) { trayDiff.dataset.key = key; trayDiff.replaceChildren(...diffItems(o, d, fields)); }
    const check = s.check;
    let checkText = '', tone = 'muted';
    if (!s.draft.profile) { checkText = pick('请选择 profile', 'Select a profile'); tone = 'warn'; }
    else if (check.state === 'loading') { checkText = pick('正在检查组合…', 'Checking combination…'); tone = 'muted'; }
    else if (check.state === 'done' && check.result?.ok) { checkText = pick('组合可用', 'Combination valid'); tone = 'ok'; }
    else if (check.result && check.result.ok === false) {
      const msg = trServer(check.result.message, check.result.reason) || M.reasonText(check.result.reason);
      checkText = pick(`不可用：${msg}`, `Unavailable: ${msg}`);
      tone = 'bad';
    }
    setText(trayCheck, checkText);
    trayCheck.dataset.tone = tone;
    trayCheck.dataset.loading = String(check.state === 'loading');
    trayRetry.hidden = check.state !== 'error';
    const blockers = o.handoff.allowed ? [] : o.handoff.blockers;
    const blockerKey = JSON.stringify([getLocale(), blockers]);
    if (trayBlockers.dataset.key !== blockerKey) {
      trayBlockers.dataset.key = blockerKey;
      trayBlockers.replaceChildren(...blockers.map((item) => h('li', {}, icon('warning', 'ab-ico ab-ico--s'), h('span', { text: item.message || item.code }), item.message ? h('code', { text: item.code }) : null)));
    }
    const torque = o.handoff.disable_torque_on_disconnect;
    setText(trayNote, pick(
      `当前 run 会收尾并归档，随后在本页启动新 run。${torque === true ? '网关断开时会松力矩，请先托住机械臂。' : '机械臂保持力矩，不会运动。'}`,
      `Current run will finalize and archive before starting a new run here. ${torque === true ? 'Torque is disabled on disconnect — support the arm first.' : 'Arm holds torque without moving.'}`,
    ));
    const ready = Boolean(s.draft.profile) && check.state === 'done' && check.result?.ok === true && o.handoff.allowed && !leaseOther();
    switchButton.disabled = !ready;
    setAttr(switchButton, 'title', ready ? null : !o.handoff.allowed ? pick('当前不能切换：见上方原因', 'Cannot switch right now: see blockers above') : pick('等待组合检查通过', 'Waiting for combination check to pass'));
  }
  function paintStatus(o, d, other) {
    const current = o.current;
    const intents = LIVE.map((field) => s.intents[field]).filter(Boolean);
    const pending = intents.filter((intent) => intent.state === 'pending');
    const unknown = intents.filter((intent) => intent.state === 'unknown');
    const confirmed = intents.filter((intent) => intent.state === 'confirmed');
    let text = '', tone = 'ok';
    const actions = [];
    const describe = (intent) => `${M.fieldLabel(intent.field)} ${intent.field === 'service_tier' ? M.tierLabel(intent.value) : intent.value}`;
    const listSep = pick('、', ', ');
    if (s.rawStatus === null) {
      text = pick('正在读取 Agent 选项（首次读取会检查各后端的运行时与组合，约需数秒）…', 'Loading Agent options (initial check probes backend runtimes and combinations; takes a few seconds)…');
      tone = 'busy';
    } else if (pending.length) {
      text = pick(`正在应用 ${pending.map(describe).join(listSep)} · 等待后端回执…`, `Applying ${pending.map(describe).join(listSep)} · waiting for backend receipt…`);
      tone = 'busy';
    } else if (unknown.length) {
      text = pick(
        `未确认：${unknown.map(describe).join(listSep)}（${unknown[0].message}）· 重试沿用同一 request_id`,
        `Unconfirmed: ${unknown.map(describe).join(listSep)} (${unknown[0].message}) · retry reuses request_id`,
      );
      tone = 'warn';
      for (const intent of unknown) {
        actions.push(
          h('button', { type: 'button', class: 'ab-btn ab-btn--s', onclick: () => retryIntent(intent) }, icon('retry'), pick('重试', 'Retry')),
          h('button', { type: 'button', class: 'ab-btn ab-btn--s ab-btn--ghost', text: pick('放弃', 'Discard'), onclick: () => dismissIntent(intent.field) }),
        );
      }
    } else if (other) {
      text = pick('另一窗口持有控制 · 只读', 'Another window holds control · read-only');
      tone = 'muted';
    } else if (s.rawError && o.source === 'legacy') {
      text = pick(`Agent 选项读取失败（${s.rawError}）· 仅显示当前设置`, `Failed to load Agent options (${s.rawError}) · showing current settings only`);
      tone = 'warn';
      actions.push(h('button', { type: 'button', class: 'ab-btn ab-btn--s', onclick: () => refreshOptions() }, icon('retry'), pick('重试', 'Retry')));
    } else if (confirmed.length) {
      const scope = (intent) => (o.source === 'legacy' && M.scopeKind(intent.applies) === 'next_session'
        ? pick('下次会话启动时生效', 'applies on next session start')
        : M.scopeLabel(intent.applies) || pick('下一轮生效', 'applies next turn'));
      const notes = confirmed.map((intent) => intent.note).filter(Boolean);
      text = pick(
        `已应用 ${confirmed.map((intent) => `${describe(intent)}（${scope(intent)}）`).join(listSep)}${notes.length ? ` · ${notes.join('；')}` : ''}`,
        `Applied ${confirmed.map((intent) => `${describe(intent)} (${scope(intent)})`).join(listSep)}${notes.length ? ` · ${notes.join('; ')}` : ''}`,
      );
      tone = notes.length ? 'warn' : 'ok';
    } else if (s.draft) {
      text = s.handoffNote
        ? pick(`${s.handoffNote} · 已列在下方，确认后切换并新建 run`, `${s.handoffNote} · staged below; confirm to switch & start a new run`)
        : pick('已选择新的会话设置，确认后切换并新建 run', 'New session settings staged below; confirm to switch & start a new run');
      tone = 'accent';
    } else {
      const served = o.served?.model;
      const stale = served && o.served?.at && current.changed_at && o.served.at < current.changed_at;
      const mismatch = served && served !== current.model && !stale;
      const profileName = current.profile == null ? null : o.profiles.find((item) => item.id === current.profile)?.name || current.profile;
      const tier = M.servedTier(o, d.service_tier);
      const sessionPrefix = pick('当前会话', 'Active session');
      text = [
        sessionPrefix,
        [M.backendLabel(current.backend), profileName, current.wire_api].filter(Boolean).join(' · '),
        served
          ? (mismatch
            ? pick(`实际服务模型 ${served}（与请求的 ${current.model} 不同）`, `served model ${served} (differs from requested ${current.model})`)
            : stale
              ? pick(`上一轮服务模型 ${served}`, `previous-turn served model ${served}`)
              : pick(`实际服务模型 ${served}`, `served model ${served}`))
          : null,
        tier.text ? `${pick('速度：', 'Speed: ')}${tier.text}` : null,
        current.effort_next_session
          ? pick(
            `推理强度 ${current.effort_next_session} 在下次会话生效（当前会话仍为 ${current.effort ?? '默认'}）`,
            `Effort ${current.effort_next_session} applies next session (current session remains ${current.effort ?? 'default'})`,
          )
          : null,
      ].filter(Boolean).join(' · ').replace(`${sessionPrefix} · `, `${sessionPrefix} `);
      tone = mismatch || tier.state === 'mismatch' ? 'warn' : 'ok';
    }
    if (!s.draft && !pending.length && !unknown.length && !other && o.source !== 'legacy') {
      const preset = M.presetFor(o, { profile: current.profile, backend: current.backend, model: d.model });
      const fields = {};
      if (preset?.effort && preset.effort !== d.effort && effortScope(o) !== 'next_session') fields.effort = preset.effort;
      if (preset?.service_tier && o.reasoning?.service_tier?.available && preset.service_tier !== d.service_tier) fields.service_tier = preset.service_tier;
      if (Object.keys(fields).length) {
        const desc = Object.entries(fields).map(([field, value]) => `${M.fieldLabel(field)} ${field === 'service_tier' ? M.tierLabel(value) : value}`).join(' · ');
        actions.push(h('button', { type: 'button', class: 'ab-btn ab-btn--s ab-btn--ghost', onclick: () => applyLive(fields) },
          pick(`套用该模型上次的设置：${desc}`, `Apply saved preset for this model: ${desc}`)));
      }
    }
    setText(statusText, s.alert ? '' : text);
    setText(alertText, s.alert || '');
    statusDot.dataset.tone = s.alert ? 'bad' : tone;
    const actionKey = JSON.stringify([getLocale(), actions.map((node) => node.textContent), s.alert]);
    if (statusActions.dataset.key !== actionKey) {
      statusActions.dataset.key = actionKey;
      if (s.alert) actions.push(h('button', { type: 'button', class: 'ab-icon-btn', 'aria-label': pick('关闭提示', 'Dismiss'), onclick: () => { s.alert = null; render(); } }, icon('close')));
      statusActions.replaceChildren(...actions);
    }
    const stamp = s.lastApplied || null;
    setText(statusTime, stamp ? clock(stamp) : '');
  }
  const LIVE = M.LIVE_FIELDS;

  function refreshLocale() {
    syncStaticLabels();
    settingsSignature = null;
    if (s.open) {
      if (s.open === 'settings' && s.settingsView === 'form' && s.form) buildForm();
      else buildPop(s.open);
    }
    if (dialog.open) populateHandoffDialog();
    paint();
  }
  onLocaleChange(refreshLocale);

  setInterval(() => {
    let dirty = false;
    for (const field of LIVE) {
      const next = M.expireIntent(s.intents[field]);
      if (next !== s.intents[field]) { s.intents[field] = next; dirty = true; }
    }
    if (dirty) render();
  }, 1000);

  render();
  return {
    update({ run, consoleState, startedAt = Date.now() }) {
      const previous = s.run?.run_id;
      s.run = run; s.consoleState = consoleState; s.snapshotAt = startedAt;
      if (previous && run?.run_id && run.run_id !== previous) {
        s.intents = {}; s.draft = null; s.handoffNote = null; s.handoffFields = null; s.check = { key: null, state: 'idle', result: null };
        if (s.handoff && !s.handoff.done) pollHandoff(); else if (!s.handoff) showToast(pick(`网关已切换到 run ${run.run_id}`, `Gateway switched to run ${run.run_id}`));
        refreshOptions();
      }
      const runKey = JSON.stringify([run?.agent_status ?? run?.codex_status, run?.active_action, run?.accepting_actions, run?.status, run?.motion_status]);
      if (runKey !== s.runKey) { const changedRun = s.runKey !== undefined; s.runKey = runKey; if (changedRun && s.draft) refreshOptions(); }
      const o = options();
      for (const field of LIVE) {
        const next = M.settleIntent(s.intents[field], { serverValue: o.current[field], turnId: String(run?.turn_id ?? ''), snapshotAt: startedAt });
        if (next !== s.intents[field]) { if (next) s.intents[field] = next; else delete s.intents[field]; }
      }
      if (!optionsInflight && Date.now() - s.rawAt > OPTIONS_TTL_MS) refreshOptions();
      render();
    },
    refreshOptions,
    refreshLocale,
  };
}
