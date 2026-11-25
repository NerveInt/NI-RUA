// Pure state and derivations of the console Agent bar (backend · profile · model · effort · speed).
// No DOM; node-tested in agent-settings-model.test.mjs. Server shapes: design 16 §5, §6, §7.1, §8
// (GET /v1/console/agent-options, /profiles/{id}/models, POST .../check, apply_agent_settings, handoff_run).
// Nothing here decides for the backend: the server's dry run (`check`) and receipts are the judge.
import { backendKind } from './phase-model.js';
import { isEn, localeTag, pick } from './shared/i18n.js';
import { HANDOFF_BLOCKER_EN, translateAgentBackendText } from './i18n/console.js';

export const BACKEND_LABELS = Object.freeze({ codex: 'Codex', claude_code: 'Claude Code', opencode: 'OpenCode', external: 'External agent' });
// Preferred wire when a (backend, profile) pair shares several (design §5: anthropic_messages first).
export const WIRE_ORDER = Object.freeze(['anthropic_messages', 'openai_responses']);
// Catalogue `protocols` value of each wire (ni_rua/cortex/agents/model_catalogue.py WIRE_PROTOCOLS).
export const WIRE_PROTOCOLS = Object.freeze({ anthropic_messages: 'MESSAGES', openai_responses: 'RESPONSES', openai_chat: 'COMPLETIONS' });
// Same rule as console.MODEL_ID: letter/digit first, then letters, digits and . _ : / @ + - [ ], at most 200 chars.
export const MODEL_ID = /^[A-Za-z0-9][A-Za-z0-9._:/@+[\]-]{0,199}$/;
// Profile id = file stem of configs/profiles/<id>.local.json (design §3.1).
export const PROFILE_ID = /^[a-z0-9][a-z0-9_-]{0,39}$/;
export const PROFILE_SCHEMA = 'ni_rua.provider_profile.v1';
// A live change without a receipt after this long is shown as unconfirmed and the server value wins again (§9.3).
export const PENDING_TIMEOUT_MS = 10000;
// A handoff whose successor never answers: the overlay says so after this long and takes no action (§9.3).
export const HANDOFF_TIMEOUT_MS = 120000;
export const LIVE_FIELDS = Object.freeze(['model', 'effort', 'service_tier']);
export const SESSION_FIELDS = Object.freeze(['backend', 'profile', 'wire_api', 'thinking_display']);
export const FIELD_LABELS = Object.freeze({ backend: '后端', profile: 'Profile', wire_api: '接口', model: '模型', effort: '推理强度',
  service_tier: '速度', thinking_display: '思考显示' });
const FIELD_LABELS_EN = Object.freeze({ backend: 'Backend', profile: 'Profile', wire_api: 'Wire', model: 'Model', effort: 'Effort',
  service_tier: 'Speed', thinking_display: 'Thinking display' });
export const fieldLabel = (field) => (isEn() ? FIELD_LABELS_EN[field] : FIELD_LABELS[field]) || field;
export const TIER_LABELS = Object.freeze({ default: '正常', priority: 'Fast · priority', ultrafast: 'Ultrafast' });
const TIER_LABELS_EN = Object.freeze({ default: 'Default', priority: 'Fast · priority', ultrafast: 'Ultrafast' });
const REASONS = Object.freeze({
  http_401: 'HTTP 401 · 密钥被拒绝', http_403: 'HTTP 403 · 无权限', http_404: 'HTTP 404 · 列表地址不存在', http_429: 'HTTP 429 · 请求过多',
  timeout: '请求超时', URLError: '网络不可达', invalid_json: '返回的不是 JSON', invalid_shape: '返回格式无法识别',
  page_cap: '超过分页上限', cursor_repeated: '分页游标重复', network: '网络错误', not_found: '接口不存在',
  // handoff_run refusals of one selection that the dry run (POST .../check) also reports (ni_rua/apps/gateway/handoff.py)
  price_card_mismatch: '本 run 的价格表没有新 provider / 模型的价格', handoff_profile_key_refused: '供应商拒绝了该 profile 的密钥',
});
const REASONS_EN = Object.freeze({
  http_401: 'HTTP 401 · API key rejected', http_403: 'HTTP 403 · Forbidden', http_404: 'HTTP 404 · Listing URL not found',
  http_429: 'HTTP 429 · Rate limited', timeout: 'Request timed out', URLError: 'Network unreachable',
  invalid_json: 'Response is not valid JSON', invalid_shape: 'Unrecognized response shape',
  page_cap: 'Pagination cap exceeded', cursor_repeated: 'Repeated pagination cursor', network: 'Network error',
  not_found: 'Endpoint not found', price_card_mismatch: 'Run price card lacks pricing for the new provider/model',
  handoff_profile_key_refused: 'Provider refused this profile’s API key',
});

export const backendLabel = (id) => BACKEND_LABELS[id] || id || '—';
export const reasonText = (code) => {
  if (!code) return '';
  const table = isEn() ? REASONS_EN : REASONS;
  const base = String(code).split(':')[0];
  return table[code] || table[base] || (isEn() ? translateAgentBackendText(code, base) : String(code));
};
const serverText = (text, code = null) => (isEn() ? translateAgentBackendText(text, code) : text);

// The scope of a setting, from the server's `applies` word (old receipts used prose such as "next Codex turn").
export function scopeKind(applies) {
  const text = String(applies || '').toLowerCase();
  if (!text) return null;
  if (text.includes('session') || text.includes('new_run')) return 'next_session';
  if (text.includes('request')) return 'next_model_request';
  if (text.includes('turn')) return 'next_turn';
  if (text === 'live' || text === 'now') return 'live';
  return null;
}
const SCOPE_LABELS = { live: '立即生效', next_turn: '下一轮生效', next_model_request: '下次请求生效', next_session: '需要新会话' };
const SCOPE_LABELS_EN = { live: 'Applies immediately', next_turn: 'Applies next turn', next_model_request: 'Applies next request', next_session: 'Requires new session' };
export const scopeLabel = (applies) => (isEn() ? SCOPE_LABELS_EN : SCOPE_LABELS)[scopeKind(applies)] || '';
// Compact form for the chip captions (the chip's accessible name carries the full label).
const SCOPE_SHORT = { live: '立即', next_turn: '下一轮', next_model_request: '下次请求', next_session: '新会话' };
const SCOPE_SHORT_EN = { live: 'Live', next_turn: 'Next turn', next_model_request: 'Next req', next_session: 'New session' };
export const scopeShort = (applies) => (isEn() ? SCOPE_SHORT_EN : SCOPE_SHORT)[scopeKind(applies)] || '';

export function formatTokens(value) {
  if (!Number.isFinite(value) || value <= 0) return null;
  if (value >= 1e6) return `${Number((value / 1e6).toFixed(value % 1e6 ? 1 : 0))}M`;
  if (value >= 1000) return `${Math.round(value / 1000)}k`;
  return String(value);
}

export function relativeAge(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return null;
  if (seconds < 45) return pick('刚刚', 'just now');
  if (seconds < 90) return pick('1 分钟前', '1m ago');
  if (seconds < 3600) return pick(`${Math.round(seconds / 60)} 分钟前`, `${Math.round(seconds / 60)}m ago`);
  if (seconds < 86400) return pick(`${Math.round(seconds / 3600)} 小时前`, `${Math.round(seconds / 3600)}h ago`);
  return pick(`${Math.round(seconds / 86400)} 天前`, `${Math.round(seconds / 86400)}d ago`);
}

// ---------- external agent (the user's own Claude Code / Codex / OpenCode drives the run over MCP) ----------
export const EXTERNAL_BACKEND = 'external';
export function isExternalBackend(run, consoleState, raw) {
  const kind = consoleState?.settings?.agent?.backend ?? raw?.current?.backend ?? backendKind(run);
  return kind === EXTERNAL_BACKEND;
}
export const DOCTOR_CLIENT_NAME = 'ni-rua-plugin-doctor';
export function externalView(consoleState, raw, run) {
  if (!isExternalBackend(run, consoleState, raw)) return null;
  const source = consoleState?.settings?.agent?.external ?? raw?.current?.external ?? null;
  const text = (value) => (typeof value === 'string' && value.trim() ? value.trim() : null);
  const client = source?.client && typeof source.client === 'object' ? { name: text(source.client.name), version: text(source.client.version) } : null;
  return {
    endpoint: text(source?.endpoint),
    client: client && (client.name || client.version) ? client : client ? { name: null, version: null } : null,
    lastCallAt: text(source?.last_call_at),
    waiting: !client,
    replay: source?.replay === true,
    probe: client?.name === DOCTOR_CLIENT_NAME,
  };
}
export const externalClientText = (client) => (!client ? '' : [client.name, client.version].filter(Boolean).join(' ') || pick('未报告名称', 'unreported name'));
export const externalClientRowText = (view) => (view.probe
  ? pick(`${externalClientText(view.client)}（doctor 探测，不是 Agent）`, `${externalClientText(view.client)} (doctor probe, not an agent)`)
  : externalClientText(view.client));
export function externalStateText(view, now = new Date()) {
  if (!view) return '';
  if (view.replay) return pick('已结束的运行（回放）', 'Finished run (replay)');
  if (view.waiting) return pick('等待 initialize（尚无客户端发来）', 'Waiting for initialize (no client yet)');
  const who = view.probe ? pick('doctor 探测', 'doctor probe') : externalClientText(view.client);
  const call = lastCallText(view.lastCallAt, now);
  return pick(
    `最近 initialize：${who}（自报） · ${call ? `最近工具调用 ${call}` : '尚无工具调用'}`,
    `Last initialize: ${who} (self-reported) · ${call ? `last tool call ${call}` : 'no tool calls yet'}`,
  );
}
export function lastCallText(iso, now = new Date()) {
  if (!iso) return null;
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  const tag = localeTag();
  const time = date.toLocaleTimeString(tag, { hour12: false });
  return date.toDateString() === now.toDateString() ? time : `${date.toLocaleDateString(tag)} ${time}`;
}

// ---------- server snapshot ----------
function blockers(list) {
  return (Array.isArray(list) ? list : []).map((item) => {
    if (typeof item === 'string') return { code: item, message: isEn() ? (HANDOFF_BLOCKER_EN[item] || null) : null };
    const code = item?.code || item?.reason || 'blocked';
    const message = isEn() ? (HANDOFF_BLOCKER_EN[code] || serverText(item?.message, code) || null) : (item?.message || null);
    return { code, message };
  });
}
export function normalizeOptions(raw, { consoleState } = {}) {
  const runtime = consoleState?.settings?.runtime || {};
  const current = { backend: null, profile: null, wire_api: null, provider: null, model: null, effort: null, service_tier: 'default', thinking_display: null, ...(raw?.current || {}) };
  const backends = Array.isArray(raw?.backends) ? raw.backends : [];
  const liveEffort = backends.find((item) => item.id === current.backend)?.live_effort !== false && scopeKind(raw?.reasoning?.effort?.applies) !== 'next_session';
  if (runtime.model?.value) current.model = runtime.model.value;
  if (runtime.effort && runtime.effort.value !== undefined) {
    if (liveEffort) current.effort = runtime.effort.value;
    else if (runtime.effort.value != null && runtime.effort.value !== current.effort) current.effort_next_session = runtime.effort.value;
    else if (runtime.effort.value === current.effort) delete current.effort_next_session;
  }
  if (runtime.service_tier?.value) current.service_tier = runtime.service_tier.value;
  if (current.service_tier == null) current.service_tier = 'default';
  const catalogues = new Map((Array.isArray(raw?.catalogues) ? raw.catalogues : []).filter((item) => item?.profile && item.wire_api).map((item) => [`${item.profile}|${item.wire_api}`, item]));
  return {
    source: 'agent-options', current, backends,
    profiles: Array.isArray(raw?.profiles) ? raw.profiles : [],
    matrix: Array.isArray(raw?.matrix) ? raw.matrix : [],
    reasoning: raw?.reasoning || {},
    handoff: { allowed: raw?.handoff?.allowed === true, blockers: blockers(raw?.handoff?.blockers),
      disable_torque_on_disconnect: raw?.handoff?.disable_torque_on_disconnect ?? null, last: raw?.handoff?.last || null },
    served: raw?.served || null, presets: raw?.presets || {}, legacyModels: null, catalogues,
    profileSchema: raw?.profile_schema && typeof raw.profile_schema === 'object' ? raw.profile_schema : null,
  };
}

export function legacyOptions(consoleState, run) {
  const runtime = consoleState?.settings?.runtime || {};
  const backend = backendKind(run);
  const tier = runtime.service_tier;
  const models = (runtime.model?.options || []).filter((option) => typeof option?.value === 'string').map((option) => {
    let label = String(option.label || option.value).replace(/ · 无图像输入$/, '');
    if (label.endsWith(` (${option.value})`)) label = label.slice(0, -option.value.length - 3);
    return { id: option.value, name: label && label !== option.value ? label : null, protocols: null, tools: null,
      image_input: typeof option.image_input === 'boolean' ? option.image_input : null, context_window: null, max_output_tokens: null,
      reasoning: null, reasoning_efforts: null };
  });
  return {
    source: 'legacy',
    current: { backend, profile: null, wire_api: null, provider: null, model: runtime.model?.value ?? consoleState?.settings?.new_run?.model ?? null,
      effort: runtime.effort?.value ?? null, service_tier: tier?.value || 'default', thinking_display: null },
    backends: [{ id: backend, label: backendLabel(backend), available: true, wires: [] }], profiles: [], matrix: [],
    reasoning: {
      effort: { value: runtime.effort?.value ?? null, command: 'set_effort', applies: scopeKind(runtime.effort?.applies) || 'next_turn', model_source: null,
        choices: (runtime.effort?.choices || []).map((value) => ({ value, backend: true, model: null })) },
      service_tier: backend === 'codex' && tier ? { available: true, value: tier.value || 'default', command: 'set_service_tier', applies: 'next_model_request',
        choices: (tier.choices || []).map((value) => ({ value, probe: tier.probes?.[value] ?? null })) } : { available: false, reason: 'backend_has_no_service_tier' },
      thinking_display: { available: false, reason: 'gateway_without_agent_options' },
    },
    handoff: { allowed: false, blockers: [{ code: 'gateway_without_agent_options', message: pick('当前网关版本不提供后端 / profile 切换', 'Current gateway version does not provide backend / profile switching') }], disable_torque_on_disconnect: null, last: null },
    served: null, presets: {}, legacyModels: { status: models.length ? 'ok' : 'not_read', reason: null, models, stale: false, age_s: null },
    catalogues: new Map(), profileSchema: null,
  };
}

// ---------- combination matrix (design §5) ----------
const profileWires = (profile) => Object.keys(profile?.wires || {});
export const shortenPaths = (text) => (text == null ? text : String(text).replace(/(?:~|\.{1,2})?\/(?:[^\s:/'"()（）]+\/)+([^\s:/'"()（）]+)/g, '$1'));
export function profileError(profile) {
  if (!profile || profile.valid !== false) return null;
  const code = typeof profile.error === 'string' ? profile.error : profile.error?.code || 'profile_invalid';
  let text = shortenPaths(profile.error_message || (typeof profile.error === 'object' ? profile.error?.message : null) || '') || '';
  if (text.startsWith(`${code}:`)) text = text.slice(code.length + 1).trim();
  if (isEn() && text) text = serverText(text, code);
  return { code, text: text || reasonText(code) || pick('profile 无效', 'Invalid profile') };
}
export function commonWires(backend, profile) {
  const offered = new Set(profileWires(profile));
  return WIRE_ORDER.concat((backend?.wires || []).filter((wire) => !WIRE_ORDER.includes(wire)))
    .filter((wire) => (backend?.wires || []).includes(wire) && offered.has(wire));
}
export function cell(options, backendId, profileId) {
  const backend = options.backends.find((item) => item.id === backendId);
  if (!backend) return { compatible: false, reason: 'backend_unknown', message: pick(`未知后端 ${backendId}`, `Unknown backend ${backendId}`) };
  if (backend.available === false) {
    const rawMsg = backend.message ? serverText(backend.message, backend.reason) : null;
    return { compatible: false, reason: backend.reason || 'backend_unavailable', message: rawMsg || pick(`${backendLabel(backendId)} 运行时不可用`, `${backendLabel(backendId)} runtime unavailable`) };
  }
  if (profileId == null) return { compatible: options.source === 'legacy', reason: options.source === 'legacy' ? null : 'profile_required', message: options.source === 'legacy' ? null : pick('请选择 profile', 'Select a profile') };
  const profile = options.profiles.find((item) => item.id === profileId);
  if (!profile) return { compatible: false, reason: 'profile_unknown', message: pick(`未知 profile ${profileId}`, `Unknown profile ${profileId}`) };
  if (profile.valid === false) { const error = profileError(profile); return { compatible: false, reason: error.code, message: `${error.code} · ${error.text}` }; }
  const entry = options.matrix.find((item) => item.backend === backendId && item.profile === profileId);
  if (entry) {
    const msg = entry.compatible === false ? (serverText(shortenPaths(entry.message), entry.reason) || reasonText(entry.reason)) : null;
    return { compatible: entry.compatible !== false, reason: entry.compatible === false ? entry.reason || 'incompatible' : null, message: msg };
  }
  if (commonWires(backend, profile).length) return { compatible: true, reason: null, message: null };
  return {
    compatible: false,
    reason: 'no_common_wire',
    message: pick(
      `该 profile 只提供 ${profileWires(profile).join('、') || '（无接口）'}，${backendLabel(backendId)} 需要 ${(backend.wires || []).join(' 或 ')}`,
      `Profile only serves ${profileWires(profile).join(', ') || '(no wires)'}; ${backendLabel(backendId)} requires ${(backend.wires || []).join(' or ')}`,
    ),
  };
}

export function wireFor(options, backendId, profileId, { explicit = null, model = null } = {}) {
  const backend = options.backends.find((item) => item.id === backendId);
  const profile = options.profiles.find((item) => item.id === profileId);
  const shared = commonWires(backend, profile);
  if (explicit && shared.includes(explicit)) return explicit;
  const protocols = Array.isArray(model?.protocols) ? model.protocols : null;
  if (protocols) {
    const listed = shared.find((wire) => protocols.includes(WIRE_PROTOCOLS[wire]));
    if (listed) return listed;
  }
  return shared[0] || null;
}

export function backendRows(options, display) {
  const valid = options.profiles.filter((profile) => profile.valid !== false);
  return options.backends.map((backend) => {
    const here = cell(options, backend.id, display.profile);
    const anywhere = options.source === 'legacy' || valid.some((profile) => cell(options, backend.id, profile.id).compatible);
    const unavailable = backend.available === false;
    const disabled = unavailable || !anywhere || options.source === 'legacy';
    let reason = null;
    if (unavailable) reason = here.message;
    else if (options.source === 'legacy') reason = backend.id === display.backend ? null : pick('当前网关不支持切换后端', 'Current gateway does not support switching backends');
    else if (!anywhere) reason = pick('没有可用于该后端的 profile', 'No compatible profile for this backend');
    else if (!here.compatible && display.profile != null) reason = pick(`${here.message} · 选择后需换 profile`, `${here.message} · requires selecting another profile`);
    return { id: backend.id, label: backend.label || backendLabel(backend.id), wires: backend.wires || [], selected: backend.id === display.backend,
      disabled, reason, code: unavailable ? backend.reason : !here.compatible ? here.reason : null };
  });
}

export function profileRows(options, display, current) {
  return options.profiles.map((profile) => {
    const here = cell(options, display.backend, profile.id);
    return { id: profile.id, name: profile.name || profile.id, wires: profileWires(profile).map((wire) => ({ wire, host: profile.wires[wire]?.host || null })),
      keyPresent: profile.key_present === true, keyText: keyStatus(profile), selected: profile.id === display.profile, current: profile.id === current.profile,
      inUse: profile.in_use === true || profile.id === current.profile, disabled: !here.compatible, reason: here.compatible ? null : here.message, code: here.reason,
      error: profileError(profile), catalogue: catalogueBrief(options, profile.id),
      notes: here.compatible ? cellChecks(options, display.backend, profile.id).filter((row) => !row.ok).map((row) => pick(`${row.wire} 不可用：${row.message}`, `${row.wire} unavailable: ${row.message}`)) : [] };
  });
}

export function cellChecks(options, backendId, profileId) {
  const entry = options.matrix.find((item) => item.backend === backendId && item.profile === profileId);
  return (Array.isArray(entry?.checks) ? entry.checks : []).map((row) => ({ wire: row.wire, ok: row.ok === true, effort: row.effort ?? null,
    reason: row.ok ? null : row.reason || 'check_failed', message: row.ok ? null : serverText(shortenPaths(row.message), row.reason) || reasonText(row.reason) }));
}
export function profileCompatibility(options, profileId) {
  return options.backends.map((backend) => {
    const verdict = cell(options, backend.id, profileId);
    const failing = verdict.compatible ? cellChecks(options, backend.id, profileId).filter((row) => !row.ok) : [];
    const sep = pick('；', '; ');
    const colon = pick('：', ': ');
    return { backend: backend.id, label: backend.label || backendLabel(backend.id), compatible: verdict.compatible, reason: verdict.reason,
      message: verdict.compatible ? (failing.length ? failing.map((row) => `${row.wire}${colon}${row.message}`).join(sep) : null) : verdict.message, partial: failing.length > 0 };
  });
}
export function catalogueBrief(options, profileId) {
  const entries = [...(options.catalogues?.values?.() || [])].filter((item) => item.profile === profileId);
  if (!entries.length) return null;
  const parts = entries.map((item) => {
    const age = relativeAge(item.age_s);
    const wire = entries.length > 1 ? `${item.wire_api} ` : '';
    if (item.status == null) return { tone: 'muted', text: `${wire}${pick('模型列表未读取', 'Model list not fetched')}` };
    if (item.status === 'unavailable') {
      return {
        tone: 'warn',
        text: pick(
          `${wire}模型列表不可用：${reasonText(item.reason || item.failure) || '未知原因'}${item.count ? ` · 缓存 ${item.count} 个` : ''}`,
          `${wire}Model list unavailable: ${reasonText(item.reason || item.failure) || 'unknown reason'}${item.count ? ` · ${item.count} cached` : ''}`,
        ),
      };
    }
    if (item.status === 'not_read') return { tone: 'muted', text: `${wire}${pick('不提供模型列表', 'No model listing')}` };
    return {
      tone: item.status === 'truncated' ? 'warn' : 'muted',
      text: pick(
        `${wire}${item.count ?? 0} 个模型${age ? ` · ${age}` : ''}${item.stale ? ' · 已过期' : ''}`,
        `${wire}${item.count ?? 0} models${age ? ` · ${age}` : ''}${item.stale ? ' · stale' : ''}`,
      ),
    };
  });
  return { tone: parts.some((part) => part.tone === 'warn') ? 'warn' : 'muted', text: parts.map((part) => part.text).join(' · ') };
}

export function keyStatus(profile) {
  if (profile?.key_present !== true) return pick('未保存密钥', 'No key saved');
  return typeof profile.key_fingerprint === 'string' && profile.key_fingerprint
    ? pick(`密钥已保存 · ${profile.key_fingerprint}`, `Key saved · ${profile.key_fingerprint}`)
    : pick('密钥已保存', 'Key saved');
}

// ---------- draft of a session-scoped change ----------
export function draftFrom(current) {
  return { backend: current.backend, profile: current.profile, wire_api: current.wire_api, model: current.model, effort: current.effort, thinking_display: current.thinking_display ?? null };
}
export function selectBackend(options, draft, backendId) {
  const next = { ...draft, backend: backendId };
  if (!cell(options, backendId, draft.profile).compatible) return { ...next, profile: null, wire_api: null };
  return { ...next, wire_api: wireFor(options, backendId, draft.profile, { explicit: draft.wire_api }) };
}
export function selectProfile(options, draft, profileId) {
  const profile = options.profiles.find((item) => item.id === profileId);
  const next = { ...draft, profile: profileId, model: profile?.defaults?.model || draft.model, effort: profile?.defaults?.effort || draft.effort };
  return { ...next, wire_api: wireFor(options, draft.backend, profileId, { explicit: draft.wire_api }), thinking_display: profile?.backends?.opencode?.thinking_display ?? draft.thinking_display ?? null };
}
export function sessionDiff(current, draft, { effortScope = null } = {}) {
  if (!draft) return { fields: [], handoff: false };
  const fields = ['backend', 'profile', 'wire_api', 'model', 'effort', 'thinking_display'].filter((field) => (draft[field] ?? null) !== (current[field] ?? null));
  const handoff = fields.some((field) => SESSION_FIELDS.includes(field)) || (fields.includes('effort') && effortScope === 'next_session');
  return { fields, handoff };
}
export function handoffBody(draft, prompt) {
  const body = { backend: draft.backend, profile: draft.profile, wire_api: draft.wire_api, model: draft.model, effort: draft.effort };
  if (draft.backend === 'opencode' && draft.wire_api === 'anthropic_messages') body.thinking_display = draft.thinking_display ?? null;
  const text = typeof prompt === 'string' ? prompt.trim() : '';
  if (text) body.prompt = text;
  return body;
}
export const checkBody = (draft) => ({ backend: draft.backend, profile: draft.profile, wire_api: draft.wire_api, model: draft.model, effort: draft.effort });
export const presetFor = (options, { profile, backend, model }) => options.presets?.[`${profile}/${backend}/${model}`] || null;

// ---------- reasoning controls (generated from the server payload, design §7.1) ----------
export function effortRows(effort, value) {
  return (effort?.choices || []).map((choice) => {
    const refused = choice.backend === false || choice.selectable === false;
    const rawNote = choice.note ? serverText(choice.note) : null;
    const note = refused
      ? rawNote || pick('后端不支持', 'Unsupported by backend')
      : choice.model === false
        ? pick('模型未声明支持 · 由供应商裁决', 'Undeclared by model · decided by provider')
        : choice.model === true
          ? pick('模型声明支持', 'Declared by model')
          : rawNote;
    return { value: choice.value, selected: choice.value === value, disabled: refused, note, description: choice.description || null,
      tone: refused ? 'muted' : choice.model === false ? 'warn' : choice.model === true ? 'ok' : null };
  });
}
export const tierLabel = (value) => {
  const table = isEn() ? TIER_LABELS_EN : TIER_LABELS;
  return table[value] || value || table.default;
};
export function servedTier(options, requested) {
  const tier = options?.reasoning?.service_tier;
  const served = tier?.served_tier ?? options?.served?.service_tier ?? null;
  if (!tier?.available || !requested || requested === 'default') return { requested: requested || 'default', served: null, state: 'none', text: null };
  if (served == null) return { requested, served: null, state: 'unconfirmed', text: pick(`请求 ${tierLabel(requested)} · 供应商实际档位未确认`, `Requested ${tierLabel(requested)} · served tier unconfirmed`) };
  if (served !== requested) return { requested, served, state: 'mismatch', text: pick(`请求 ${tierLabel(requested)} · 供应商实际以 ${tierLabel(served)} 服务`, `Requested ${tierLabel(requested)} · served as ${tierLabel(served)}`) };
  return { requested, served, state: 'match', text: pick(`请求 ${tierLabel(requested)} · 供应商确认以 ${tierLabel(served)} 服务`, `Requested ${tierLabel(requested)} · confirmed as ${tierLabel(served)}`) };
}
export function tierRows(tier, value) {
  return (tier?.choices || []).map((choice) => {
    const probe = choice.probe;
    const refused = choice.selectable === false || probe?.supported === false;
    let note;
    if (choice.value === 'default') note = pick('标准档位', 'Standard tier');
    else if (!probe) note = pick('首次选择会先检测供应商是否支持', 'Probed for provider support on first selection');
    else if (probe.supported === false) note = pick(`已检测：供应商拒绝${probe.rejection ? `（${probe.rejection}）` : ''}`, `Probed: provider rejected${probe.rejection ? ` (${probe.rejection})` : ''}`);
    else if (probe.served_tier && probe.served_tier !== choice.value) note = pick(`已检测：供应商接受，但实际以 ${tierLabel(probe.served_tier)} 服务`, `Probed: accepted, served as ${tierLabel(probe.served_tier)}`);
    else if (probe.served_tier) note = pick('已检测：供应商接受并以该档位服务', 'Probed: accepted and served at this tier');
    else note = pick('已检测：供应商接受', 'Probed: accepted by provider');
    const rowLabel = (isEn() ? TIER_LABELS_EN[choice.value] : TIER_LABELS[choice.value]) || serverText(choice.label) || choice.value;
    return { value: choice.value, label: rowLabel, selected: choice.value === value, disabled: refused && choice.value !== value,
      note, description: choice.description || null, tone: probe ? (refused ? 'warn' : probe.served_tier && probe.served_tier !== choice.value ? 'warn' : 'ok') : null };
  });
}
export function thinkingChoices(control) {
  const defaultLabel = pick('默认（不发送）', 'Default (not sent)');
  const choices = (control?.choices || []).map((item) => (item && typeof item === 'object' ? item : { value: item }))
    .map((item) => ({
      value: item.value ?? null,
      label: item.label ? serverText(item.label) : (item.value == null ? defaultLabel : String(item.value)),
      disabled: item.selectable === false,
    }));
  if (!choices.some((item) => item.value === null)) choices.unshift({ value: null, label: defaultLabel, disabled: false });
  return choices.filter((item, index) => choices.findIndex((other) => other.value === item.value) === index);
}

// ---------- model catalogue rows (design §6) ----------
export function modelBadges(model) {
  const badges = [];
  badges.push(model.image_input === true
    ? { kind: 'image', tone: 'ok', text: pick('图像', 'Vision') }
    : model.image_input === false
      ? { kind: 'image', tone: 'warn', text: pick('无图像输入', 'No vision') }
      : { kind: 'image', tone: 'muted', text: pick('图像未声明', 'Vision undeclared') });
  if (model.tools === true) badges.push({ kind: 'tools', tone: 'neutral', text: pick('工具', 'Tools') });
  else if (model.tools == null) badges.push({ kind: 'tools', tone: 'muted', text: pick('工具未声明', 'Tools undeclared') });
  const context = formatTokens(model.context_window);
  if (context) badges.push({ kind: 'context', tone: 'neutral', text: context, title: pick(`上下文窗口 ${model.context_window} tokens`, `Context window ${model.context_window} tokens`) });
  const output = formatTokens(model.max_output_tokens);
  if (output) badges.push({ kind: 'output', tone: 'neutral', text: pick(`输出 ${output}`, `Out ${output}`), title: pick(`最大输出 ${model.max_output_tokens} tokens`, `Max output ${model.max_output_tokens} tokens`) });
  if (Array.isArray(model.reasoning_efforts) && model.reasoning_efforts.length) badges.push({ kind: 'effort', tone: 'neutral', text: `${pick('强度', 'Effort')} ${model.reasoning_efforts.join(' · ')}` });
  else if (model.reasoning === false) badges.push({ kind: 'effort', tone: 'muted', text: pick('无推理', 'No reasoning') });
  return badges;
}
const wiresOf = (protocols) => Object.keys(WIRE_PROTOCOLS).filter((wire) => protocols.includes(WIRE_PROTOCOLS[wire]));
export function modelRows(models, { wire = null, sharedWires = [], query = '', selected = null, current = null } = {}) {
  const needle = query.trim().toLowerCase();
  const protocol = WIRE_PROTOCOLS[wire];
  const listed = new Set();
  const rows = [];
  for (const model of models || []) {
    if (!model?.id || listed.has(model.id)) continue;
    listed.add(model.id);
    if (needle && !model.id.toLowerCase().includes(needle) && !String(model.name || '').toLowerCase().includes(needle)) continue;
    let disabled = false, reason = null, requiresWire = null;
    if (model.tools === false) { disabled = true; reason = pick('不支持工具调用，agent 无法驱动', 'Tool calling unsupported; cannot drive agent'); }
    else if (protocol && Array.isArray(model.protocols) && !model.protocols.includes(protocol)) {
      requiresWire = sharedWires.find((other) => other !== wire && model.protocols.includes(WIRE_PROTOCOLS[other])) || null;
      if (requiresWire) reason = pick(`需要新会话 · 改用 ${requiresWire}`, `Requires new session · switch to ${requiresWire}`);
      else {
        disabled = true;
        const avail = wiresOf(model.protocols).join(' / ') || model.protocols.join(' / ') || pick('其他接口', 'another wire');
        reason = pick(`仅在 ${avail} 上提供`, `Available only on ${avail}`);
      }
    }
    rows.push({ id: model.id, name: model.name && model.name !== model.id ? model.name : null, model, badges: modelBadges(model),
      selected: model.id === selected, current: model.id === current, disabled, reason, requiresWire, unlisted: false });
  }
  if (current && !listed.has(current) && (!needle || current.toLowerCase().includes(needle))) {
    rows.unshift({ id: current, name: null, model: { id: current }, badges: [], selected: current === selected, current: true, disabled: false,
      reason: pick('不在供应商列表中', 'Not in provider listing'), requiresWire: null, unlisted: true });
  }
  const text = query.trim();
  let freeText = null;
  if (text && !listed.has(text) && text !== current) {
    freeText = MODEL_ID.test(text)
      ? { id: text, valid: true, note: pick('不在供应商列表中，由供应商裁决', 'Not in provider listing; decided by provider') }
      : { id: text, valid: false, note: pick('模型名格式无效：字母或数字开头，只含字母、数字和 . _ : / @ + - [ ]，最多 200 字符', 'Invalid model ID: must start with letter/digit and contain only letters, digits, and . _ : / @ + - [ ] (max 200 chars)') };
  }
  return { rows, freeText };
}

export function catalogueSummary(catalogue) {
  if (!catalogue || (catalogue.loading && !catalogue.models?.length)) return { tone: 'muted', text: pick('正在读取模型列表…', 'Loading model list…') };
  const count = catalogue.models?.length || 0;
  const age = relativeAge(catalogue.age_s);
  if (catalogue.error) {
    return {
      tone: 'bad',
      text: pick(
        `读取失败 · ${catalogue.error}${count ? ' · 显示上次的列表' : ''}`,
        `Fetch failed · ${catalogue.error}${count ? ' · showing previous list' : ''}`,
      ),
      retry: true,
    };
  }
  switch (catalogue.status) {
    case 'ok':
      return count
        ? { tone: 'muted', text: pick(`${count} 个模型${age ? ` · 更新于 ${age}` : ''}`, `${count} models${age ? ` · updated ${age}` : ''}`) }
        : { tone: 'muted', text: pick('供应商没有返回模型列表，可以直接输入模型名', 'Provider returned no model list; type any model ID directly') };
    case 'truncated':
      return { tone: 'warn', text: pick(`列表不完整：${reasonText(catalogue.reason) || 'truncated'} · 共 ${count} 个`, `List truncated: ${reasonText(catalogue.reason) || 'truncated'} · ${count} total`) };
    case 'unavailable':
      return {
        tone: 'bad',
        text: pick(
          `模型列表不可用：${reasonText(catalogue.reason) || '未知原因'}${count && catalogue.stale ? ` · 显示缓存于 ${age || '之前'}的列表` : ''}`,
          `Model list unavailable: ${reasonText(catalogue.reason) || 'unknown reason'}${count && catalogue.stale ? ` · showing list cached ${age || 'earlier'}` : ''}`,
        ),
        retry: true,
      };
    case 'not_read':
      return { tone: 'muted', text: pick('该来源不提供模型列表，可以直接输入模型名', 'This source provides no model list; type any model ID directly') };
    default:
      return { tone: 'muted', text: count ? pick(`${count} 个模型`, `${count} models`) : pick('没有模型列表，可以直接输入模型名', 'No model list; type any model ID directly') };
  }
}

// ---------- live change state machine (design §9.3) ----------
let intentSerial = 0;
export function createIntent(field, value, { now = Date.now(), previous = null } = {}) {
  return { id: ++intentSerial, field, value, previous, state: 'pending', startedAt: now, message: null, applies: null, turnId: null };
}
export function intentReceipt(intent, result, { turnId = null, now = Date.now() } = {}) {
  const status = result?.status;
  if (status === 'applied') return { ...intent, state: 'confirmed', applies: result.applies || null, turnId, settledAt: now, message: null,
    note: result.message ? serverText(result.message) : null, servedTier: result.served_tier ?? null };
  const rawMsg = result?.message ? serverText(result.message, result?.reason) : null;
  const message = rawMsg || result?.reason || result?.error || (status === 'skipped' ? pick('前一项被拒绝，本项未执行', 'Skipped because an earlier setting was rejected') : status ? pick(`未应用（${status}）`, `Not applied (${status})`) : pick('未应用', 'Not applied'));
  return { ...intent, state: 'rejected', settledAt: now, message, code: result?.reason || status || null };
}
export function isHandoffRequired(receipt) { return receipt?.status === 'handoff_required'; }
export function draftFromHandoffRequired(base, receipt) {
  const next = { ...base };
  for (const [field, change] of Object.entries(receipt?.diff || {})) {
    if (field in next || SESSION_FIELDS.includes(field) || ['model', 'effort'].includes(field)) next[field] = change?.requested ?? null;
  }
  return next;
}
export function intentError(intent, error, { now = Date.now() } = {}) {
  const unknown = !error?.status || (error.status === 409 && error.body?.status === 'unknown') || error.status >= 500;
  if (unknown) return { ...intent, state: 'unknown', settledAt: now, message: error?.body?.reason || error?.message || 'network_error' };
  const raw = error.body?.message || error.body?.error || error.message;
  return { ...intent, state: 'rejected', settledAt: now, message: serverText(raw, error.body?.error), code: error.body?.error || null };
}
export function expireIntent(intent, now = Date.now(), timeoutMs = PENDING_TIMEOUT_MS) {
  return intent?.state === 'pending' && now - intent.startedAt >= timeoutMs
    ? { ...intent, state: 'unknown', settledAt: now, message: pick(`超过 ${Math.round(timeoutMs / 1000)} 秒没有回执`, `No receipt after ${Math.round(timeoutMs / 1000)} s`) }
    : intent;
}
export function settleIntent(intent, { serverValue, turnId = '', snapshotAt = Infinity } = {}) {
  if (!intent || intent.state === 'pending' || intent.state === 'rejected') return intent;
  if (snapshotAt < (intent.settledAt ?? 0)) return intent;
  if (intent.state === 'unknown') return serverValue === intent.value ? null : intent;
  if (serverValue !== intent.value) return serverValue == null ? intent : null;
  return intent.turnId != null && String(turnId ?? '') !== String(intent.turnId) ? null : intent;
}
export function displayValue(serverValue, intent) {
  if (intent?.state === 'pending' || intent?.state === 'confirmed') return { value: intent.value, state: intent.state };
  if (intent?.state === 'unknown') return { value: serverValue, state: 'unknown', intended: intent.value };
  if (intent?.state === 'rejected') return { value: serverValue, state: 'rejected', intended: intent.value };
  return { value: serverValue, state: 'idle' };
}

// ---------- handoff progress (design §4.2, §9.3) ----------
export function handoffProgress(handoff, { runId = null, reachable = true, now = Date.now() } = {}) {
  const down = handoff.sawDown || !reachable;
  if (runId && runId === handoff.successorRunId) return { phase: 'ready', step: 3 };
  if (runId && runId !== handoff.fromRunId) return { phase: 'fallback', step: 3 };
  if (now - handoff.startedAt >= HANDOFF_TIMEOUT_MS) return { phase: 'timeout', step: down ? 2 : 1 };
  return { phase: down ? 'starting' : 'finalizing', step: down ? 2 : 1 };
}
export function fallbackReason(run, options) {
  const fallback = run?.handoff?.fallback || options?.handoff?.last?.fallback || options?.handoff?.last;
  return fallback?.reason || fallback?.message || null;
}

// ---------- provider profile form (design §3.2; one canonical format, key inline, §13) ----------
export const PROFILE_FORM_FIELDS = Object.freeze({
  name: 'name', api_key: 'credential.api_key', wires: 'wires.<wire>', base_url: 'wires.<wire>.base_url', auth_scheme: 'wires.<wire>.auth_scheme',
  models_url: 'wires.<wire>.models_url', tool_result_images: 'compat.tool_result_images', tool_schema_max_depth: 'compat.tool_schema_max_depth',
  claude_env: 'backends.claude_code.env', thinking_display: 'backends.opencode.thinking_display', default_model: 'defaults.model', default_effort: 'defaults.effort',
});
export const PROFILE_ID_HELP = '文件名 <id>.local.json：小写字母或数字开头，可含 _ 和 -，最多 40 个字符；保存后不可改。';
export const profileIdHelp = () => pick(
  PROFILE_ID_HELP,
  'File name <id>.local.json: starts with lowercase letter or digit, may contain _ and - (max 40 chars); immutable once saved.',
);
export function profileHelp(schema) {
  const fields = new Map((Array.isArray(schema?.fields) ? schema.fields : []).filter((item) => item?.name && item.help).map((item) => [item.name, String(item.help)]));
  const help = {};
  for (const [control, field] of Object.entries(PROFILE_FORM_FIELDS)) if (fields.has(field)) help[control] = { field, text: fields.get(field) };
  return help;
}
export function profileChoices(schema) {
  const list = (value, fallback) => (Array.isArray(value) && value.length ? value.filter((item) => typeof item === 'string') : fallback);
  return { wires: list(schema?.wires, [...WIRE_ORDER]), auth_schemes: list(schema?.auth_schemes, ['bearer', 'x-api-key']),
    tool_result_images: list(schema?.tool_result_images, ['inline', 'user_blocks']), thinking_display: list(schema?.thinking_display, ['summarized', 'omitted']),
    efforts: list(schema?.efforts, []) };
}
export function slugify(name) {
  return String(name || '').toLowerCase().normalize('NFKD').replace(/[^a-z0-9]+/g, '-').replace(/^[-_]+|[-_]+$/g, '').slice(0, 40).replace(/[-_]+$/, '');
}
export function emptyProfileForm() {
  return { id: '', name: '', api_key: '', idEdited: false,
    wires: { anthropic_messages: { enabled: true, base_url: '', auth_scheme: 'bearer', models_url: '' }, openai_responses: { enabled: false, base_url: '', auth_scheme: 'bearer', models_url: '' } },
    compat: { tool_result_images: 'inline', tool_schema_max_depth: '' }, claude_env: '', thinking_display: '', default_model: '', default_effort: '' };
}
export function profileFormFromInfo(info) {
  const form = emptyProfileForm();
  form.id = info.id; form.name = info.name || info.id; form.idEdited = true;
  for (const wire of WIRE_ORDER) {
    const entry = info.wires?.[wire];
    form.wires[wire] = { enabled: Boolean(entry), base_url: entry?.base_url || '', auth_scheme: entry?.auth_scheme || 'bearer', models_url: entry?.models_url || '' };
  }
  form.compat = { tool_result_images: info.compat?.tool_result_images || 'inline', tool_schema_max_depth: info.compat?.tool_schema_max_depth ?? '' };
  const env = info.backends?.claude_code?.env;
  form.claude_env = env && typeof env === 'object' ? Object.entries(env).map(([key, value]) => `${key}=${value}`).join('\n') : '';
  form.thinking_display = info.backends?.opencode?.thinking_display || '';
  form.default_model = info.defaults?.model || '';
  form.default_effort = info.defaults?.effort || '';
  return form;
}
export function profileRequest(form, { editing = false, keyStored = false } = {}) {
  const errors = {}, warnings = {};
  const id = String(form.id || '').trim();
  if (!PROFILE_ID.test(id)) errors.id = pick('ID 需小写字母或数字开头，只含小写字母、数字、_ 和 -，最多 40 个字符', 'ID must start with a lowercase letter or digit and contain only a–z, 0–9, _ and - (max 40 chars)');
  const name = String(form.name || '').trim() || id;
  if (name.length > 60) errors.name = pick('名称最多 60 个字符', 'Name cannot exceed 60 characters');
  const key = String(form.api_key || '').trim();
  const keepKey = editing && keyStored && !key;
  if (!keepKey && !key) errors.api_key = pick('请粘贴密钥', 'Paste an API key');
  else if (key && !/^[\x21-\x7e]+$/.test(key)) errors.api_key = pick('密钥只能包含可打印 ASCII 字符，且不含空格', 'API key must contain printable ASCII characters without spaces');
  const wires = {};
  for (const wire of WIRE_ORDER) {
    const entry = form.wires?.[wire];
    if (!entry?.enabled) continue;
    const url = String(entry.base_url || '').trim();
    let parsed = null;
    try { parsed = new URL(url); } catch { /* reported below */ }
    if (!parsed || !/^https?:$/.test(parsed.protocol)) { errors[`wire.${wire}`] = pick('请填写 http(s) 地址', 'Enter an http(s) URL'); continue; }
    const path = parsed.pathname.replace(/\/+$/, '');
    if (wire === 'anthropic_messages' && path.endsWith('/v1')) warnings[`wire.${wire}`] = pick('Anthropic Messages 的地址通常不含 /v1', 'Anthropic Messages base_url usually omits /v1');
    if (wire === 'openai_responses' && !path.endsWith('/v1')) warnings[`wire.${wire}`] = pick('OpenAI Responses 的地址通常以 /v1 结尾', 'OpenAI Responses base_url usually ends with /v1');
    wires[wire] = { base_url: url.replace(/\/+$/, ''), auth_scheme: entry.auth_scheme === 'x-api-key' ? 'x-api-key' : 'bearer' };
    const listing = String(entry.models_url || '').trim();
    if (listing) {
      let parsedListing = null;
      try { parsedListing = new URL(listing); } catch { /* reported below */ }
      if (!parsedListing || !/^https?:$/.test(parsedListing.protocol)) errors[`models.${wire}`] = pick('模型列表地址需要 http(s)，或留空', 'models_url must be http(s), or leave empty');
      else wires[wire].models_url = listing;
    }
  }
  if (!Object.keys(wires).length && !Object.keys(errors).some((field) => field.startsWith('wire.'))) errors.wires = pick('至少启用一个接口', 'Enable at least one wire');
  const depthText = String(form.compat?.tool_schema_max_depth ?? '').trim();
  const depth = depthText === '' ? null : Number(depthText);
  if (depth !== null && (!Number.isInteger(depth) || depth < 3)) errors.tool_schema_max_depth = pick('需要 ≥ 3 的整数，或留空', 'Must be an integer ≥ 3, or leave empty');
  const env = {};
  for (const [index, raw] of String(form.claude_env || '').split('\n').entries()) {
    const line = raw.trim();
    if (!line || line.startsWith('#')) continue;
    const match = /^([A-Za-z_][A-Za-z0-9_]*)=(.*)$/.exec(line);
    if (!match) { errors.claude_env = pick(`第 ${index + 1} 行需要 NAME=VALUE`, `Line ${index + 1} must be NAME=VALUE`); break; }
    env[match[1]] = match[2];
  }
  const thinking = ['summarized', 'omitted'].includes(form.thinking_display) ? form.thinking_display : null;
  const defaults = {};
  const model = String(form.default_model || '').trim();
  if (model && !MODEL_ID.test(model)) errors.default_model = pick('模型名格式无效', 'Invalid model ID format');
  else if (model) defaults.model = model;
  const effort = String(form.default_effort || '').trim();
  if (effort) defaults.effort = effort;
  const document = { schema: PROFILE_SCHEMA, name, wires,
    compat: { tool_result_images: form.compat?.tool_result_images === 'user_blocks' ? 'user_blocks' : 'inline', ...(depth !== null ? { tool_schema_max_depth: depth } : {}) },
    backends: { claude_code: { env }, opencode: { thinking_display: thinking } }, ...(Object.keys(defaults).length ? { defaults } : {}) };
  if (!keepKey) document.credential = { api_key: key };
  return { ok: Object.keys(errors).length === 0, errors, warnings, body: { id, document, keep_key: keepKey } };
}
