import assert from 'node:assert/strict';
import test from 'node:test';
import * as M from './agent-settings-model.js';

const backends = [
  { id: 'codex', label: 'Codex', available: true, wires: ['openai_responses'], effort_choices: ['low', 'high'] },
  { id: 'claude_code', label: 'Claude Code', available: true, wires: ['anthropic_messages'] },
  { id: 'opencode', label: 'OpenCode', available: true, wires: ['anthropic_messages', 'openai_responses'] },
];
const profiles = [
  { id: 'tr', name: 'Provider Alpha', valid: true, wires: { anthropic_messages: { host: 'alpha.example' } }, key_present: true, defaults: { model: 'qwen', effort: 'high' } },
  { id: 'oc', name: 'OpenCode Console', valid: true, wires: { anthropic_messages: { host: 'oc.example' }, openai_responses: { host: 'oc.example' } }, key_present: true },
  { id: 'mz', name: 'Provider Beta', valid: true, wires: { openai_responses: { host: 'beta.example' } }, key_present: true, key_fingerprint: 'sk-…7f3a' },
  { id: 'bad', name: 'Broken', valid: false, error: 'profile_key_missing', error_message: 'profile 缺少 credential.api_key', wires: {}, key_present: false },
];
const raw = {
  current: { backend: 'claude_code', profile: 'tr', wire_api: 'anthropic_messages', model: 'qwen', effort: 'high', service_tier: 'default' },
  backends, profiles, matrix: [], reasoning: { effort: { applies: 'next_session', choices: [{ value: 'high', backend: true, model: null }] } },
  handoff: { allowed: true, blockers: [] }, presets: { 'tr/claude_code/kimi': { effort: 'max' } },
};
const options = (extra = {}) => M.normalizeOptions({ ...raw, ...extra });

test('scope words map onto the three scopes, including the prose of older receipts', () => {
  assert.equal(M.scopeKind('next_turn'), 'next_turn');
  assert.equal(M.scopeKind('next Codex turn'), 'next_turn');
  assert.equal(M.scopeKind('next_agent_turn'), 'next_turn');
  assert.equal(M.scopeKind('next_session_start'), 'next_session');
  assert.equal(M.scopeKind('next session start'), 'next_session');
  assert.equal(M.scopeKind('next model request (also mid-turn)'), 'next_model_request');
  assert.equal(M.scopeKind(undefined), null);
  assert.equal(M.scopeLabel('next_session'), '需要新会话');
  assert.equal(M.scopeLabel('next_turn'), '下一轮生效');
  assert.equal(M.scopeShort('next model request (also mid-turn)'), '下次请求');
  assert.equal(M.scopeShort('next_session_start'), '新会话');
});

test('token and age formatting', () => {
  assert.equal(M.formatTokens(256000), '256k');
  assert.equal(M.formatTokens(1000000), '1M');
  assert.equal(M.formatTokens(1500000), '1.5M');
  assert.equal(M.formatTokens(null), null);
  assert.equal(M.relativeAge(10), '刚刚');
  assert.equal(M.relativeAge(120), '2 分钟前');
  assert.equal(M.relativeAge(3 * 3600), '3 小时前');
  assert.equal(M.relativeAge(undefined), null);
});

test('the options snapshot takes live values from the fresher console poll', () => {
  const liveRaw = { ...raw, current: { ...raw.current, backend: 'opencode' }, reasoning: { effort: { applies: 'next_turn', choices: [] } } };
  const value = M.normalizeOptions(liveRaw, { consoleState: { settings: { runtime: { model: { value: 'kimi' }, effort: { value: 'max' }, service_tier: { value: 'priority' } } } } });
  assert.equal(value.source, 'agent-options');
  assert.deepEqual([value.current.model, value.current.effort, value.current.service_tier], ['kimi', 'max', 'priority']);
  assert.equal(value.current.profile, 'tr');
  assert.deepEqual(M.normalizeOptions({ handoff: { allowed: false, blockers: ['turn_running', { code: 'benchmark_run', message: '评测 run' }] } }).handoff.blockers,
    [{ code: 'turn_running', message: null }, { code: 'benchmark_run', message: '评测 run' }]);
});

test('an older gateway without agent-options keeps today\'s live controls and blocks backend/profile changes', () => {
  const consoleState = { settings: { runtime: {
    model: { value: 'gpt-6-mini', options: [{ value: 'gpt-6.1-sol', label: 'GPT 6.1 (gpt-6.1-sol)' }, { value: 'text-only', label: 'Text (text-only) · 无图像输入', image_input: false }, { value: 'gpt-6-mini', label: 'gpt-6-mini' }] },
    effort: { value: 'high', choices: ['low', 'high'], applies: 'next session start' },
    service_tier: { value: 'default', choices: ['default', 'priority'], probes: { priority: { supported: true } } } }, new_run: {} } };
  const claude = M.legacyOptions(consoleState, { agent_backend: 'claude_code' });
  assert.equal(claude.source, 'legacy');
  assert.equal(claude.current.backend, 'claude_code');
  assert.deepEqual(claude.legacyModels.models.map((model) => [model.id, model.name, model.image_input]), [['gpt-6.1-sol', 'GPT 6.1', null], ['text-only', 'Text', false], ['gpt-6-mini', null, null]]);
  assert.equal(claude.reasoning.effort.applies, 'next_session');
  assert.equal(claude.reasoning.service_tier.available, false, 'service tier is Codex-only');
  assert.equal(claude.handoff.allowed, false);
  const codex = M.legacyOptions(consoleState, {});
  assert.equal(codex.reasoning.service_tier.available, true);
  assert.deepEqual(codex.reasoning.service_tier.choices[1], { value: 'priority', probe: { supported: true } });
  assert.ok(M.backendRows(codex, codex.current).every((row) => row.disabled), 'legacy gateway: backend is read-only');
});

test('combination cells: runtime, profile validity, the server matrix, then the shared wire', () => {
  const unavailable = M.normalizeOptions({ ...raw, backends: [{ ...backends[1], available: false, reason: 'backend_unavailable:claude_code_sdk_missing', message: 'Claude Code 运行时未安装' }] });
  assert.deepEqual(M.cell(unavailable, 'claude_code', 'tr'), { compatible: false, reason: 'backend_unavailable:claude_code_sdk_missing', message: 'Claude Code 运行时未安装' });
  const o = options();
  assert.equal(M.cell(o, 'claude_code', 'bad').reason, 'profile_key_missing');
  assert.equal(M.cell(o, 'claude_code', 'bad').message, 'profile_key_missing · profile 缺少 credential.api_key', 'the code stays visible next to the sentence');
  const derived = M.cell(o, 'claude_code', 'mz');
  assert.equal(derived.reason, 'no_common_wire');
  assert.match(derived.message, /只提供 openai_responses，Claude Code 需要 anthropic_messages/);
  const matrix = options({ matrix: [{ backend: 'codex', profile: 'mz', compatible: false, reason: 'gate:x', message: 'gate says no' }] });
  assert.deepEqual(M.cell(matrix, 'codex', 'mz'), { compatible: false, reason: 'gate:x', message: 'gate says no' }, 'the server matrix wins');
  assert.equal(M.cell(o, 'opencode', 'oc').compatible, true);
});

test('backend rows: an incompatible profile does not trap the operator; unavailable or profile-less backends are disabled with a reason', () => {
  const o = options();
  const rows = Object.fromEntries(M.backendRows(o, o.current).map((row) => [row.id, row]));
  assert.equal(rows.codex.disabled, false, 'Codex has compatible profiles (oc, mz)');
  assert.match(rows.codex.reason, /选择后需换 profile/);
  assert.equal(rows.claude_code.selected, true);
  const lonely = M.normalizeOptions({ ...raw, profiles: [profiles[0]] });
  const codex = M.backendRows(lonely, lonely.current).find((row) => row.id === 'codex');
  assert.equal(codex.disabled, true);
  assert.equal(codex.reason, '没有可用于该后端的 profile');
});

test('profile rows expose the incompatibility reason as text and never a key', () => {
  const o = options();
  const rows = Object.fromEntries(M.profileRows(o, o.current, o.current).map((row) => [row.id, row]));
  assert.equal(rows.mz.disabled, true);
  assert.match(rows.mz.reason, /只提供 openai_responses/);
  assert.equal(rows.bad.disabled, true);
  assert.equal(rows.tr.current, true);
  assert.equal(rows.mz.keyText, '密钥已保存 · sk-…7f3a');
  assert.equal(rows.bad.keyText, '未保存密钥');
  assert.equal(M.keyStatus({ key_present: true }), '密钥已保存');
});

test('choosing a backend or profile builds a draft; an incompatible pair clears the profile instead of guessing one', () => {
  const o = options();
  const draft = M.draftFrom(o.current);
  const codex = M.selectBackend(o, draft, 'codex');
  assert.equal(codex.profile, null);
  assert.equal(codex.wire_api, null);
  const withProfile = M.selectProfile(o, codex, 'mz');
  assert.equal(withProfile.wire_api, 'openai_responses');
  const opencode = M.selectBackend(o, draft, 'opencode');
  assert.deepEqual([opencode.profile, opencode.wire_api], ['tr', 'anthropic_messages']);
  const defaults = M.selectProfile(o, { ...draft, model: 'other', effort: 'low' }, 'tr');
  assert.deepEqual([defaults.model, defaults.effort], ['qwen', 'high'], 'profile defaults pre-fill model and effort');
});

test('wire choice: explicit when shared, else the model\'s listed wire, else anthropic_messages first', () => {
  const o = options();
  assert.equal(M.wireFor(o, 'opencode', 'oc'), 'anthropic_messages');
  assert.equal(M.wireFor(o, 'opencode', 'oc', { explicit: 'openai_responses' }), 'openai_responses');
  assert.equal(M.wireFor(o, 'opencode', 'oc', { model: { protocols: ['RESPONSES'] } }), 'openai_responses');
  assert.equal(M.wireFor(o, 'opencode', 'oc', { explicit: 'openai_chat' }), 'anthropic_messages');
  assert.equal(M.wireFor(o, 'claude_code', 'mz'), null);
});

test('session diff: a backend/profile change or a session-scoped effort needs a handoff, a model alone does not', () => {
  const current = options().current;
  const draft = M.draftFrom(current);
  assert.deepEqual(M.sessionDiff(current, { ...draft, model: 'kimi' }), { fields: ['model'], handoff: false });
  assert.deepEqual(M.sessionDiff(current, { ...draft, backend: 'opencode', model: 'kimi' }), { fields: ['backend', 'model'], handoff: true });
  assert.equal(M.sessionDiff(current, { ...draft, effort: 'max' }, { effortScope: 'next_session' }).handoff, true);
  assert.equal(M.sessionDiff(current, { ...draft, effort: 'max' }, { effortScope: 'next_turn' }).handoff, false);
  assert.deepEqual(M.sessionDiff(current, null), { fields: [], handoff: false });
});

test('handoff body carries thinking_display only where it exists and the prompt only when typed', () => {
  const draft = { backend: 'opencode', profile: 'oc', wire_api: 'anthropic_messages', model: 'kimi', effort: 'high', thinking_display: 'summarized' };
  assert.deepEqual(M.handoffBody(draft, '  pick the cube  '), { ...draft, prompt: 'pick the cube' });
  const responses = M.handoffBody({ ...draft, wire_api: 'openai_responses' }, '   ');
  assert.equal('thinking_display' in responses, false);
  assert.equal('prompt' in responses, false);
  assert.deepEqual(M.checkBody(draft), { backend: 'opencode', profile: 'oc', wire_api: 'anthropic_messages', model: 'kimi', effort: 'high' });
  assert.deepEqual(M.presetFor(options(), { profile: 'tr', backend: 'claude_code', model: 'kimi' }), { effort: 'max' });
});

test('reasoning controls are generated from the server payload: backend refusals disabled, unreported model support marked', () => {
  const rows = M.effortRows({ choices: [{ value: 'low', backend: true, model: true }, { value: 'xhigh', backend: true, model: false }, { value: 'none', backend: false, model: null }] }, 'xhigh');
  assert.deepEqual(rows.map((row) => [row.value, row.disabled, row.note, row.selected]), [
    ['low', false, '模型声明支持', false], ['xhigh', false, '模型未声明支持 · 由供应商裁决', true], ['none', true, '后端不支持', false]]);
  const tiers = M.tierRows({ choices: [{ value: 'default' }, { value: 'priority', probe: { supported: true } }, { value: 'ultrafast', probe: null }] }, 'priority');
  assert.deepEqual(tiers.map((row) => [row.label, row.note, row.selected]), [['正常', '标准档位', false], ['Fast · priority', '已检测：供应商接受', true], ['Ultrafast', '首次选择会先检测供应商是否支持', false]]);
});

test('model badges never draw an unreported capability as supported', () => {
  const unknown = M.modelBadges({ id: 'x' });
  assert.deepEqual(unknown.map((badge) => [badge.text, badge.tone]), [['图像未声明', 'muted'], ['工具未声明', 'muted']]);
  const full = M.modelBadges({ id: 'k', image_input: true, tools: true, context_window: 256000, max_output_tokens: 32000, reasoning_efforts: ['low', 'high', 'max'] });
  assert.deepEqual(full.map((badge) => badge.text), ['图像', '工具', '256k', '输出 32k', '强度 low · high · max']);
  assert.equal(M.modelBadges({ id: 't', image_input: false })[0].text, '无图像输入');
});

test('model rows: other-wire models are disabled with the wire, or need a new session when the pair shares it; free text always works', () => {
  const models = [
    { id: 'qwen', name: 'Qwen', protocols: ['MESSAGES'], tools: true, image_input: true },
    { id: 'gpt-x', protocols: ['RESPONSES'], tools: true },
    { id: 'no-tools', tools: false },
    { id: 'qwen', name: 'duplicate' },
  ];
  const { rows, freeText } = M.modelRows(models, { wire: 'anthropic_messages', sharedWires: ['anthropic_messages'], query: '', selected: 'qwen', current: 'legacy-model' });
  assert.deepEqual(rows.map((row) => row.id), ['legacy-model', 'qwen', 'gpt-x', 'no-tools']);
  assert.equal(rows[0].reason, '不在供应商列表中');
  assert.equal(rows[1].selected, true);
  assert.deepEqual([rows[2].disabled, rows[2].reason], [true, '仅在 openai_responses 上提供']);
  assert.equal(rows[3].disabled, true);
  assert.equal(freeText, null);
  const shared = M.modelRows(models, { wire: 'anthropic_messages', sharedWires: ['anthropic_messages', 'openai_responses'] }).rows.find((row) => row.id === 'gpt-x');
  assert.deepEqual([shared.disabled, shared.requiresWire], [false, 'openai_responses']);
  const typed = M.modelRows(models, { wire: 'anthropic_messages', query: 'my-custom-model' });
  assert.deepEqual(typed.rows, []);
  assert.deepEqual(typed.freeText, { id: 'my-custom-model', valid: true, note: '不在供应商列表中，由供应商裁决' });
  assert.equal(M.modelRows(models, { query: 'bad id' }).freeText.valid, false);
  assert.deepEqual(M.modelRows(models, { query: 'QW' }).rows.map((row) => row.id), ['qwen'], 'search matches id or name, case-insensitive');
  assert.ok(M.MODEL_ID.test('claude-x[1m]') && M.MODEL_ID.test('vendor/model') && !M.MODEL_ID.test('-x') && !M.MODEL_ID.test('a b'));
});

test('catalogue states: loading, empty, truncated, unavailable with a stale list, fetch error', () => {
  assert.equal(M.catalogueSummary(undefined).text, '正在读取模型列表…');
  assert.equal(M.catalogueSummary({ loading: true, models: [] }).text, '正在读取模型列表…');
  assert.equal(M.catalogueSummary({ status: 'ok', models: [] }).text, '供应商没有返回模型列表，可以直接输入模型名');
  assert.match(M.catalogueSummary({ status: 'truncated', reason: 'page_cap', models: [{}] }).text, /列表不完整：超过分页上限/);
  const stale = M.catalogueSummary({ status: 'unavailable', reason: 'http_401', stale: true, age_s: 3 * 3600, models: [{}] });
  assert.equal(stale.text, '模型列表不可用：HTTP 401 · 密钥被拒绝 · 显示缓存于 3 小时前的列表');
  assert.equal(stale.retry, true);
  assert.equal(M.catalogueSummary({ error: 'boom', models: [{}] }).text, '读取失败 · boom · 显示上次的列表');
});

test('live change: the pending value survives a poll, then a receipt confirms until a new turn starts', () => {
  const intent = M.createIntent('model', 'kimi', { now: 1000, previous: 'qwen' });
  assert.deepEqual(M.displayValue('qwen', intent), { value: 'kimi', state: 'pending' }, 'poll value never overwrites pending');
  assert.equal(M.settleIntent(intent, { serverValue: 'qwen', snapshotAt: 5000 }), intent, 'pending is never settled by a poll');
  const confirmed = M.intentReceipt(intent, { status: 'applied', applies: 'next_turn' }, { turnId: 't1', now: 2000 });
  assert.equal(confirmed.state, 'confirmed');
  assert.deepEqual(M.displayValue('qwen', confirmed), { value: 'kimi', state: 'confirmed' });
  assert.equal(M.settleIntent(confirmed, { serverValue: 'qwen', turnId: 't1', snapshotAt: 1500 }), confirmed, 'a poll started before the receipt is no evidence');
  assert.equal(M.settleIntent(confirmed, { serverValue: 'kimi', turnId: 't1', snapshotAt: 2500 }), confirmed, 'stays "next turn" until a new turn');
  assert.equal(M.settleIntent(confirmed, { serverValue: 'kimi', turnId: 't2', snapshotAt: 2500 }), null);
  assert.equal(M.settleIntent(confirmed, { serverValue: 'other', turnId: 't1', snapshotAt: 2500 }), null, 'replaced by another change');
});

test('live change: a refusal reverts with its message; 409 unknown, network errors and silence become unconfirmed', () => {
  const intent = M.createIntent('model', 'bad-x', { now: 0 });
  const rejected = M.intentReceipt(intent, { status: 'rejected', message: 'agent_model_rejected:bad-x' });
  assert.deepEqual(M.displayValue('qwen', rejected), { value: 'qwen', state: 'rejected', intended: 'bad-x' });
  assert.equal(rejected.message, 'agent_model_rejected:bad-x');
  assert.equal(M.intentReceipt(intent, { status: 'skipped' }).message, '前一项被拒绝，本项未执行');
  assert.equal(M.intentReceipt(intent, { status: 'unsupported', message: '不支持 ultrafast' }).state, 'rejected');
  const http400 = M.intentError(intent, Object.assign(new Error('x'), { status: 400, body: { error: 'agent_model_rejected:http400-x' } }));
  assert.deepEqual([http400.state, http400.message], ['rejected', 'agent_model_rejected:http400-x']);
  const unknown = M.intentError(intent, Object.assign(new Error('x'), { status: 409, body: { status: 'unknown', reason: 'dispatch_acknowledgement_missing' } }));
  assert.deepEqual([unknown.state, unknown.message], ['unknown', 'dispatch_acknowledgement_missing']);
  assert.equal(M.intentError(intent, new TypeError('Failed to fetch')).state, 'unknown');
  assert.equal(M.expireIntent(intent, M.PENDING_TIMEOUT_MS - 1), intent);
  const silent = M.expireIntent(intent, M.PENDING_TIMEOUT_MS);
  assert.deepEqual([silent.state, M.displayValue('qwen', silent).value], ['unknown', 'qwen'], 'the server value wins again after the timeout');
  assert.equal(M.settleIntent(silent, { serverValue: 'bad-x', snapshotAt: M.PENDING_TIMEOUT_MS + 1 }), null, 'resolved once the server shows the value');
  assert.equal(M.settleIntent(silent, { serverValue: 'qwen', snapshotAt: M.PENDING_TIMEOUT_MS + 1 }), silent);
});

test('handoff progress: finalizing, listener down, successor ready, fallback run, silent gateway', () => {
  const handoff = { fromRunId: 'run-1', successorRunId: 'run-2', startedAt: 0, sawDown: false };
  assert.deepEqual(M.handoffProgress(handoff, { runId: 'run-1', reachable: true, now: 500 }), { phase: 'finalizing', step: 1 });
  assert.deepEqual(M.handoffProgress(handoff, { runId: null, reachable: false, now: 900 }), { phase: 'starting', step: 2 });
  assert.deepEqual(M.handoffProgress({ ...handoff, sawDown: true }, { runId: 'run-1', reachable: true, now: 900 }), { phase: 'starting', step: 2 });
  assert.deepEqual(M.handoffProgress(handoff, { runId: 'run-2', now: 3000 }), { phase: 'ready', step: 3 });
  assert.deepEqual(M.handoffProgress(handoff, { runId: 'run-3', now: 3000 }), { phase: 'fallback', step: 3 });
  assert.deepEqual(M.handoffProgress({ ...handoff, sawDown: true }, { runId: null, reachable: false, now: M.HANDOFF_TIMEOUT_MS }), { phase: 'timeout', step: 2 });
  assert.equal(M.fallbackReason({ handoff: { fallback: { failed_run_id: 'run-2', reason: 'successor_start_failed' } } }), 'successor_start_failed');
  assert.equal(M.fallbackReason({}, { handoff: { last: { fallback: { reason: 'from options' } } } }), 'from options');
});

test('profile form: the key is sent only in the request body, kept on edit when left empty, and every field is validated', () => {
  const form = M.emptyProfileForm();
  Object.assign(form, { id: 'example', name: 'Example Gateway', api_key: 'sk-test-SECRET-123' });
  form.wires.anthropic_messages.base_url = 'https://gateway.example/anthropic/';
  form.wires.openai_responses = { enabled: true, base_url: 'https://gateway.example/v1', auth_scheme: 'x-api-key' };
  form.compat.tool_schema_max_depth = '13';
  form.claude_env = '# comment\nCLAUDE_CODE_MAX_OUTPUT_TOKENS=32000\n';
  form.thinking_display = 'summarized';
  const request = M.profileRequest(form);
  assert.equal(request.ok, true, JSON.stringify(request.errors));
  assert.deepEqual(request.body.document, {
    schema: 'ni_rua.provider_profile.v1', name: 'Example Gateway', credential: { api_key: 'sk-test-SECRET-123' },
    wires: { anthropic_messages: { base_url: 'https://gateway.example/anthropic', auth_scheme: 'bearer' }, openai_responses: { base_url: 'https://gateway.example/v1', auth_scheme: 'x-api-key' } },
    compat: { tool_result_images: 'inline', tool_schema_max_depth: 13 },
    backends: { claude_code: { env: { CLAUDE_CODE_MAX_OUTPUT_TOKENS: '32000' } }, opencode: { thinking_display: 'summarized' } } });
  assert.equal(request.body.keep_key, false);
  const edit = M.profileRequest({ ...form, api_key: '' }, { editing: true, keyStored: true });
  assert.equal(edit.ok, true);
  assert.equal(edit.body.keep_key, true);
  assert.equal('credential' in edit.body.document, false, 'an empty key on edit keeps the stored one');
  assert.equal(M.profileRequest({ ...form, api_key: '' }).errors.api_key, '请粘贴密钥');
  const broken = M.profileRequest({ ...form, id: 'Bad ID', api_key: 'has space', compat: { tool_schema_max_depth: '2' }, claude_env: 'not an assignment',
    wires: { anthropic_messages: { enabled: true, base_url: 'ftp://x' }, openai_responses: { enabled: false } } });
  assert.deepEqual(Object.keys(broken.errors).sort(), ['api_key', 'claude_env', 'id', 'tool_schema_max_depth', 'wire.anthropic_messages']);
  assert.ok(!JSON.stringify([broken.errors, broken.warnings]).includes('has space'), 'error text never repeats the key');
  const none = M.profileRequest({ ...form, wires: { anthropic_messages: { enabled: false }, openai_responses: { enabled: false } } });
  assert.equal(none.errors.wires, '至少启用一个接口');
  const conventions = M.profileRequest({ ...form, wires: { anthropic_messages: { enabled: true, base_url: 'https://g.example/v1' }, openai_responses: { enabled: true, base_url: 'https://g.example/api' } } });
  assert.deepEqual(Object.keys(conventions.warnings).sort(), ['wire.anthropic_messages', 'wire.openai_responses']);
  assert.equal(conventions.ok, true, 'base-URL conventions are warnings, the server is the judge');
});

test('profile form helpers: id from name, edit form from ProfileInfo (no key field filled)', () => {
  assert.equal(M.slugify('Example Gateway (EU)'), 'example-gateway-eu');
  assert.equal(M.slugify('  --ProviderAlpha--  '), 'provideralpha');
  assert.ok(M.PROFILE_ID.test(M.slugify('A'.repeat(80))));
  const form = M.profileFormFromInfo({ id: 'tr', name: 'Provider Alpha', key_present: true, wires: { anthropic_messages: { host: 'alpha.example', base_url: 'https://alpha.example/a', auth_scheme: 'x-api-key' } },
    compat: { tool_result_images: 'user_blocks', tool_schema_max_depth: 13 }, backends: { claude_code: { env: { A: '1' } }, opencode: { thinking_display: 'omitted' } }, defaults: { model: 'qwen' } });
  assert.equal(form.api_key, '');
  assert.deepEqual(form.wires.anthropic_messages, { enabled: true, base_url: 'https://alpha.example/a', auth_scheme: 'x-api-key', models_url: '' });
  assert.equal(form.wires.openai_responses.enabled, false);
  assert.deepEqual([form.compat.tool_result_images, form.compat.tool_schema_max_depth, form.claude_env, form.thinking_display, form.default_model], ['user_blocks', 13, 'A=1', 'omitted', 'qwen']);
});

// ---------- WP-F: the D1 follow-ups (design 16 §16) against the real gateway shapes ----------
test('Claude Code effort: the console poll reports the next-session effort, never the running one', () => {
  const rawCc = { ...raw, current: { ...raw.current, effort: 'high' } };
  const poll = { settings: { runtime: { effort: { value: 'xhigh' } } } };
  const value = M.normalizeOptions(rawCc, { consoleState: poll });
  assert.equal(value.current.effort, 'high', 'Claude Code has live_effort false on the real gateway');
  assert.equal(value.current.effort_next_session, 'xhigh');
  const same = M.normalizeOptions({ ...rawCc, current: { ...rawCc.current, effort_next_session: 'max' } }, { consoleState: { settings: { runtime: { effort: { value: 'high' } } } } });
  assert.equal(same.current.effort, 'high');
  assert.equal('effort_next_session' in same.current, false, 'a poll showing the running value again clears the next-session note');
  const live = M.normalizeOptions({ ...rawCc, current: { ...rawCc.current, backend: 'codex' }, reasoning: { effort: { applies: 'next_turn', choices: [] } } }, { consoleState: poll });
  assert.deepEqual([live.current.effort, live.current.effort_next_session], ['xhigh', undefined], 'live-effort backends take the poll value');
  const noTier = M.normalizeOptions({ ...raw, current: { ...raw.current, service_tier: null } });
  assert.equal(noTier.current.service_tier, 'default', 'a backend without the knob is "default"');
});

test('thinking_display: the server list already holds the default, so it is never doubled', () => {
  const server = { available: true, choices: [{ value: null, label: '不发送（默认）', selectable: true }, { value: 'summarized', label: 'summarized（摘要）', selectable: true }, { value: 'omitted', label: 'omitted', selectable: false }] };
  assert.deepEqual(M.thinkingChoices(server).map((item) => [item.value, item.label, item.disabled]),
    [[null, '不发送（默认）', false], ['summarized', 'summarized（摘要）', false], ['omitted', 'omitted', true]]);
  assert.equal(M.thinkingChoices({ choices: ['summarized', 'omitted'] }).filter((item) => item.value === null).length, 1, 'older payload: default added once');
  assert.equal(M.thinkingChoices({ choices: [null, 'summarized', null] }).length, 2);
});

test('speed: requested vs served tier, honestly worded; refused tiers are unselectable', () => {
  const tier = (extra) => ({ reasoning: { service_tier: { available: true, ...extra } }, served: null });
  assert.deepEqual(M.servedTier(tier({ served_tier: 'default' }), 'priority'), { requested: 'priority', served: 'default', state: 'mismatch', text: '请求 Fast · priority · 供应商实际以 正常 服务' });
  assert.equal(M.servedTier(tier({ served_tier: 'priority' }), 'priority').state, 'match');
  assert.equal(M.servedTier(tier({}), 'priority').state, 'unconfirmed');
  assert.equal(M.servedTier({ reasoning: { service_tier: { available: true } }, served: { service_tier: 'default' } }, 'priority').state, 'mismatch', 'served.service_tier is the fallback source');
  assert.equal(M.servedTier(tier({ served_tier: 'default' }), 'default').state, 'none', 'nothing to claim for the default tier');
  assert.equal(M.servedTier({ reasoning: { service_tier: { available: false } } }, 'priority').state, 'none');
  const rows = M.tierRows({ choices: [{ value: 'default', label: '正常', selectable: true }, { value: 'priority', label: 'Fast', description: '2x speed', probe: { supported: true, served_tier: 'default' } },
    { value: 'ultrafast', label: 'Ultrafast', selectable: false, probe: { supported: false, rejection: 'HTTP 400' } }] }, 'default');
  assert.deepEqual(rows.map((row) => [row.value, row.disabled, row.note, row.tone]), [
    ['default', false, '标准档位', null], ['priority', false, '已检测：供应商接受，但实际以 正常 服务', 'warn'], ['ultrafast', true, '已检测：供应商拒绝（HTTP 400）', 'warn']]);
  assert.equal(rows[1].description, '2x speed');
  const confirmed = M.intentReceipt(M.createIntent('service_tier', 'priority'), { status: 'applied', applies: 'next_model_request', served_tier: 'default', message: '供应商接受了 priority，但探测请求实际以 default 档位服务' });
  assert.deepEqual([confirmed.state, confirmed.servedTier, confirmed.note], ['confirmed', 'default', '供应商接受了 priority，但探测请求实际以 default 档位服务']);
});

test('effort rows use the server note, description and selectable flag', () => {
  const rows = M.effortRows({ choices: [{ value: 'none', backend: true, model: false, selectable: true, note: '模型未声明支持' }, { value: 'low', backend: true, model: null, description: 'Fast responses', selectable: true },
    { value: 'max', backend: true, model: null, selectable: false, note: '当前后端不支持' }] }, 'low');
  assert.deepEqual(rows.map((row) => [row.value, row.disabled, row.note, row.description]), [
    ['none', false, '模型未声明支持 · 由供应商裁决', null], ['low', false, null, 'Fast responses'], ['max', true, '当前后端不支持', null]]);
});

test('apply_agent_settings answering handoff_required becomes a session draft, not "not applied" fields', () => {
  const receipt = { status: 'handoff_required', session_fields: ['effort'], results: {}, diff: { effort: { current: 'high', requested: 'max' }, service_tier: { current: 'default', requested: 'priority' } } };
  assert.equal(M.isHandoffRequired(receipt), true);
  assert.equal(M.isHandoffRequired({ status: 'applied', results: {} }), false);
  const base = M.draftFrom(options().current);
  const draft = M.draftFromHandoffRequired(base, receipt);
  assert.equal(draft.effort, 'max');
  assert.equal('service_tier' in draft, false, 'live-only fields are not part of a session draft');
  assert.deepEqual(M.sessionDiff(options().current, draft, { effortScope: 'next_session' }), { fields: ['effort'], handoff: true });
});

test('profiles[].error is a string code plus error_message; absolute paths are shortened to the file name', () => {
  const broken = { id: 'broken', valid: false, error: 'profile_no_wires', error_message: 'profile_no_wires:/private/tmp/x/profiles/broken.local.json: declare at least one wire in wires (anthropic_messages, openai_responses)' };
  assert.deepEqual(M.profileError(broken), { code: 'profile_no_wires', text: 'broken.local.json: declare at least one wire in wires (anthropic_messages, openai_responses)' });
  assert.equal(M.profileError({ id: 'ok', valid: true }), null);
  assert.deepEqual(M.profileError({ valid: false, error: 'profile_unreadable' }), { code: 'profile_unreadable', text: 'profile_unreadable' });
  assert.equal(M.shortenPaths('see /a/b/c/tr.local.json and ./x/y.json'), 'see tr.local.json and y.json');
  const o = options({ profiles: [...profiles, broken] });
  assert.match(M.cell(o, 'codex', 'broken').message, /^profile_no_wires · broken\.local\.json: declare/);
  const shortMatrix = options({ matrix: [{ backend: 'codex', profile: 'mz', compatible: false, reason: 'profile_invalid:x', message: '无效：x:/abs/dir/mz.local.json: bad' }] });
  assert.equal(M.cell(shortMatrix, 'codex', 'mz').message, '无效：x:mz.local.json: bad');
});

test('matrix checks and cached catalogues describe each profile without a request', () => {
  const o = options({
    matrix: [{ backend: 'opencode', profile: 'oc', compatible: true, wires: ['anthropic_messages'], checks: [{ wire: 'anthropic_messages', ok: true }, { wire: 'openai_responses', ok: false, reason: 'gate:x', message: 'gate refuses /p/oc.local.json' }] },
      { backend: 'codex', profile: 'oc', compatible: true, wires: ['openai_responses'], checks: [{ wire: 'openai_responses', ok: true }] }],
    catalogues: [{ profile: 'oc', wire_api: 'anthropic_messages', status: 'ok', count: 3, age_s: 120, stale: false }, { profile: 'oc', wire_api: 'openai_responses', status: 'unavailable', reason: 'timeout', count: 2 },
      { profile: 'tr', wire_api: 'anthropic_messages', status: null, count: null }],
  });
  assert.deepEqual(M.cellChecks(o, 'opencode', 'oc').map((row) => [row.wire, row.ok, row.message]), [['anthropic_messages', true, null], ['openai_responses', false, 'gate refuses oc.local.json']]);
  const compat = Object.fromEntries(M.profileCompatibility(o, 'oc').map((row) => [row.backend, row]));
  assert.deepEqual([compat.opencode.compatible, compat.opencode.partial, compat.opencode.message], [true, true, 'openai_responses：gate refuses oc.local.json']);
  assert.equal(compat.claude_code.compatible, true);
  assert.deepEqual(M.catalogueBrief(o, 'oc'), { tone: 'warn', text: 'anthropic_messages 3 个模型 · 2 分钟前 · openai_responses 模型列表不可用：请求超时 · 缓存 2 个' });
  assert.deepEqual(M.catalogueBrief(o, 'tr'), { tone: 'muted', text: '模型列表未读取' });
  assert.equal(M.catalogueBrief(o, 'mz'), null);
  const row = M.profileRows(o, { ...o.current, backend: 'opencode' }, o.current).find((item) => item.id === 'oc');
  assert.deepEqual(row.notes, ['openai_responses 不可用：gate refuses oc.local.json']);
});

test('profile form help is generated from profile_schema.fields (the single contract)', () => {
  const schema = { fields: [{ name: 'name', help: 'Display name.' }, { name: 'credential.api_key', help: 'The provider key.' }, { name: 'wires.<wire>.models_url', help: 'Optional listing URL.' }, { name: 'unrelated', help: 'x' }],
    auth_schemes: ['bearer', 'x-api-key'], efforts: ['low', 'high'], thinking_display: ['summarized', 'omitted'] };
  assert.deepEqual(M.profileHelp(schema), { name: { field: 'name', text: 'Display name.' }, api_key: { field: 'credential.api_key', text: 'The provider key.' }, models_url: { field: 'wires.<wire>.models_url', text: 'Optional listing URL.' } });
  assert.deepEqual(M.profileHelp(null), {}, 'no schema, no invented help');
  assert.deepEqual(M.profileChoices(schema).efforts, ['low', 'high']);
  assert.deepEqual(M.profileChoices(null).auth_schemes, ['bearer', 'x-api-key']);
  for (const field of Object.values(M.PROFILE_FORM_FIELDS)) assert.match(field, /^(name|credential\.|wires\.|compat\.|backends\.|defaults\.)/);
});

test('profile form keeps an explicit models_url (editing never drops it) and validates it', () => {
  const form = M.profileFormFromInfo({ id: 'oc', name: 'OpenCode Console', key_present: true, wires: { openai_responses: { base_url: 'https://o.example/v1', auth_scheme: 'bearer', models_url: 'https://o.example/inference/v1/models' } } });
  assert.equal(form.wires.openai_responses.models_url, 'https://o.example/inference/v1/models');
  const edit = M.profileRequest(form, { editing: true, keyStored: true });
  assert.equal(edit.ok, true, JSON.stringify(edit.errors));
  assert.deepEqual(edit.body.document.wires.openai_responses, { base_url: 'https://o.example/v1', auth_scheme: 'bearer', models_url: 'https://o.example/inference/v1/models' });
  form.wires.openai_responses.models_url = 'ftp://nope';
  assert.equal(M.profileRequest(form, { editing: true, keyStored: true }).errors['models.openai_responses'], '模型列表地址需要 http(s)，或留空');
});

// ---------- external backend: read-only Agent bar ----------
const externalPayload = (client, lastCallAt = null) => ({
  backend: 'external', profile: null, provider: null, wire_api: null, model: null, effort: null, service_tier: null, thinking_display: null, upstream_source: null,
  revision: 0, changed_at: null, turn_count: 0, mixed: false,
  external: { endpoint: 'http://127.0.0.1:8790/mcp', client, last_call_at: lastCallAt },
});
const externalRun = { agent_backend: 'external', agent_status: 'running' };

test('external agent: waiting state when no client has connected', () => {
  const consoleState = { settings: { agent: externalPayload(null) } };
  const view = M.externalView(consoleState, null, externalRun);
  assert.deepEqual(view, { endpoint: 'http://127.0.0.1:8790/mcp', client: null, lastCallAt: null, waiting: true, replay: false, probe: false });
  assert.equal(M.externalStateText(view), '等待 initialize（尚无客户端发来）');
  assert.equal(M.lastCallText(view.lastCallAt), null);
  assert.equal(M.externalClientText(view.client), '');
});

test('external agent: self-reported client and last call', () => {
  const consoleState = { settings: { agent: externalPayload({ name: 'claude-code', version: '2.1.3', self_reported: true }, '2026-11-15T08:30:00Z') } };
  const view = M.externalView(consoleState, null, externalRun);
  assert.equal(view.waiting, false);
  assert.equal(M.externalClientText(view.client), 'claude-code 2.1.3');
  assert.equal(view.lastCallAt, '2026-11-15T08:30:00Z');
  const now = new Date('2026-11-15T08:31:00Z');
  assert.match(M.lastCallText(view.lastCallAt, now), /\d{1,2}:\d{2}:\d{2}$/);
  assert.equal(M.lastCallText('not-a-time'), 'not-a-time');
  assert.equal(M.externalClientText({ name: null, version: null }), '未报告名称');
  assert.equal(M.externalClientText({ name: 'opencode', version: null }), 'opencode');
});

test('external agent: the state line states facts and never says connected', () => {
  const now = new Date('2026-11-15T08:31:00Z');
  const base = externalPayload({ name: 'claude-code', version: '2.1.3', self_reported: true });
  const noCall = M.externalView({ settings: { agent: base } }, null, externalRun);
  assert.equal(M.externalStateText(noCall, now), '最近 initialize：claude-code 2.1.3（自报） · 尚无工具调用');
  const called = M.externalView({ settings: { agent: externalPayload(base.external.client, '2026-11-15T08:30:00Z') } }, null, externalRun);
  assert.match(M.externalStateText(called, now), /^最近 initialize：claude-code 2\.1\.3（自报） · 最近工具调用 \d{1,2}:\d{2}:\d{2}$/);
  for (const view of [noCall, called]) assert.doesNotMatch(M.externalStateText(view, now), /已连接|连接/);
});

test('external agent: the doctor probe is labelled as a probe by its one documented client name', () => {
  const doctor = { name: M.DOCTOR_CLIENT_NAME, version: '0.1.0', self_reported: true };
  const view = M.externalView({ settings: { agent: externalPayload(doctor) } }, null, externalRun);
  assert.equal(M.DOCTOR_CLIENT_NAME, 'ni-rua-plugin-doctor');
  assert.equal(view.probe, true);
  assert.equal(M.externalClientRowText(view), 'ni-rua-plugin-doctor 0.1.0（doctor 探测，不是 Agent）');
  assert.match(M.externalStateText(view), /doctor 探测/);
  const other = M.externalView({ settings: { agent: externalPayload({ name: 'doctor', version: '1', self_reported: true }) } }, null, externalRun);
  assert.equal(other.probe, false);
});

test('external agent: a replay of an archived run is not "waiting"', () => {
  const payload = externalPayload(null);
  payload.external = { endpoint: null, client: null, last_call_at: null, replay: true };
  const view = M.externalView({ settings: { agent: payload } }, null, externalRun);
  assert.equal(view.replay, true);
  assert.equal(M.externalStateText(view), '已结束的运行（回放）');
});

test('external agent: payload from agent-options is used when the console state has none yet', () => {
  const view = M.externalView(null, { current: externalPayload({ name: 'codex', version: '0.9', self_reported: true }) }, externalRun);
  assert.equal(view.client.name, 'codex');
  const noEndpoint = M.externalView({ settings: { agent: { ...externalPayload(null), external: { endpoint: null, client: null, last_call_at: null } } } }, null, externalRun);
  assert.equal(noEndpoint.endpoint, null);
});

test('host backends and an unloaded console are never external', () => {
  assert.equal(M.externalView(null, null, null), null);
  assert.equal(M.externalView({ settings: { agent: { backend: 'claude_code' } } }, null, { agent_backend: 'claude_code' }), null);
  assert.equal(M.externalView(null, { current: raw.current }, { agent_backend: 'codex' }), null);
  assert.equal(M.isExternalBackend({ agent_backend: 'external' }, null, null), true);
  // the console payload wins over a stale run snapshot
  assert.equal(M.isExternalBackend({ agent_backend: 'external' }, { settings: { agent: { backend: 'codex' } } }), false);
});

test('external agent: normalized options carry no selectable controls and no handoff', () => {
  const o = M.normalizeOptions({ current: externalPayload(null), backends: [], profiles: [], matrix: [], reasoning: {},
    handoff: { allowed: false, blockers: [{ code: 'handoff_unsupported_by_backend', message: null }] } });
  assert.equal(o.current.backend, 'external');
  assert.equal(M.backendLabel('external'), 'External agent');
  assert.equal(o.handoff.allowed, false);
  assert.deepEqual(o.handoff.blockers.map((item) => item.code), ['handoff_unsupported_by_backend']);
  assert.equal(o.current.model, null);
  assert.equal(o.current.profile, null);
  assert.equal(o.current.effort, null);
});
