// Browser check of the whole operator console (WP-F): layout, safety header, polling stability, skeletons, focus,
// reduced motion, and the Agent bar follow-ups of design 16 §16, at 1440 / 768 / 390 px in light and dark.
// Never port 8781. Build first (npm --prefix web run build), then either
//   * the SYNTHETIC fixture:  mamba run -n ni-rua python web/fixture-server.py --port 18811
//       NI_RUA_URL=http://127.0.0.1:18811 node web/check-console-shell.mjs
//   * a REAL gateway (Workbench.app() over a fake robot and agent, no hardware, no model):
//       NI_RUA_TARGET=real NI_RUA_REAL_MODE=claude_code|codex|opencode NI_RUA_URL=http://127.0.0.1:<port> node web/check-console-shell.mjs
//   * the REAL supervisor (server.serve over a fake robot and agents, started on Claude Code + a profile serving
//     anthropic_messages, with a second profile "Provider Beta" serving openai_responses): NI_RUA_REAL_MODE=handoff runs every
//     section, then a real handoff to Codex from the Agent bar; NI_RUA_REAL_MODE=handoff-fallback (the successor fails
//     to start) runs only the handoff and expects the fallback banner.
// PLAYWRIGHT_MODULE / NI_RUA_BROWSER_EXECUTABLE / NI_RUA_SCREENSHOT_DIR as in check-agent-settings.mjs.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const url = process.env.NI_RUA_URL || 'http://127.0.0.1:18811';
assert.notEqual(new URL(url).port, '8781', 'never run browser checks against the robot gateway');
const target = process.env.NI_RUA_TARGET || 'fixture';
const realMode = process.env.NI_RUA_REAL_MODE || 'claude_code';
const handoffMode = realMode.startsWith('handoff');
const mode = handoffMode ? 'claude_code' : realMode;   // the backend the gateway starts on
const handoffOnly = realMode === 'handoff-fallback';
const fixture = target === 'fixture';
const output = process.env.NI_RUA_SCREENSHOT_DIR || '/tmp/ni-rua-console-shell';
const prefix = fixture ? 'fixture-' : `real-${realMode}-`;
await fs.mkdir(output, { recursive: true });
const checks = [];
const clients = new Map();   // client_id of each window (diagnostics of a lease that was not given back)
const check = (name, condition) => { assert.ok(condition, `${prefix}${name}`); checks.push(name); };
const knobs = async (body) => (fixture ? (await fetch(`${url}/fixture/agent`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })).json() : null);
const browser = await chromium.launch({ headless: true, ...(process.env.NI_RUA_BROWSER_EXECUTABLE ? { executablePath: process.env.NI_RUA_BROWSER_EXECUTABLE } : {}) });
const VIEWPORTS = [['desktop', { width: 1440, height: 1000 }], ['tablet', { width: 768, height: 1000 }], ['phone', { width: 390, height: 844 }]];

// A control command from node with its own short lease (before a page holds one): claim, command, release.
async function withLease(command, fields) {
  const post = (body) => fetch(`${url}/v1/control`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ request_id: crypto.randomUUID(), ...body }) }).then((response) => response.json());
  const client = `shell-check-${crypto.randomUUID()}`;
  let token, answer;
  for (let attempt = 0; attempt < 40 && !token; attempt += 1) {   // a closed window's lease may still be expiring
    answer = await post({ command: 'claim_input', client_id: client });
    token = answer.writer_token;
    if (!token && attempt === 0) console.warn('lease held by', clients.get((await (await fetch(`${url}/v1/run`)).json()).controller?.owner_client_id) || 'unknown window', answer);
    if (!token) await new Promise((resolve) => setTimeout(resolve, 500));
  }
  assert.ok(token, `node lease: ${JSON.stringify(answer)} controller ${JSON.stringify((await (await fetch(`${url}/v1/run`)).json()).controller)} windows ${JSON.stringify([...clients])}`);
  const result = await post({ command, ...fields, client_id: client, writer_token: token });
  await post({ command: 'release_input', client_id: client, writer_token: token });
  return result;
}

async function openConsole(viewport, colorScheme, { reducedMotion = 'no-preference', delay = null } = {}) {
  const context = await browser.newContext({ viewport, colorScheme, reducedMotion });
  const page = await context.newPage();
  page.setDefaultTimeout(20000);
  const seen = { errors: [] };
  page.on('pageerror', (error) => seen.errors.push(error.message));
  page.on('request', (request) => { if (request.url().endsWith('/v1/control') && (request.postData() || '').includes('claim_input')) clients.set(JSON.parse(request.postData()).client_id, `${viewport.width}-${colorScheme}-${reducedMotion}-${Boolean(delay)}`); });
  if (delay) for (const [pattern, ms] of Object.entries(delay)) await page.route(pattern, async (route) => { await new Promise((resolve) => setTimeout(resolve, ms)); await route.continue(); });
  await page.goto(`${url}/console`);
  return { page, context, seen };
}
const ready = async (page) => {
  await page.waitForFunction(() => document.querySelector('.ab') && !document.querySelector('.ab').dataset.loading && !document.body.dataset.boot);
  await page.waitForFunction(() => document.getElementById('lease-state').textContent.includes('本窗口'));
};
async function close({ page, context }) {
  // Leaving the page fires the console's pagehide handler, which releases the lease (keepalive), so the next window
  // can take it at once (a real gateway has no fixture reset). Clicking "release" would be re-claimed by the next poll.
  const released = page.waitForRequest((request) => request.url().endsWith('/v1/control') && (request.postData() || '').includes('"release_input"'), { timeout: 3000 }).catch(() => null);
  await page.unrouteAll({ behavior: 'ignoreErrors' }).catch(() => {});
  await page.goto('about:blank').catch((error) => console.warn(`leaving the page failed: ${error.message}`));
  await released;
  for (let attempt = 0; attempt < 30; attempt += 1) {
    const run = await (await fetch(`${url}/v1/run`)).json().catch(() => ({}));
    if (!run.controller?.owner_client_id) break;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  await context.close();
}
const shot = (page, name, options = {}) => page.screenshot({ path: path.join(output, `${prefix}${name}.png`), ...options });
const noOverflow = (page) => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1);
const estopReachable = (page) => page.evaluate(() => {
  const box = document.getElementById('estop').getBoundingClientRect();
  if (box.top < -1 || box.bottom > innerHeight + 1 || box.width < 40) return false;
  return document.elementFromPoint(box.left + box.width / 2, box.top + box.height / 2)?.closest('#estop') !== null;
});
const chip = (page, field) => page.locator(`.ab-chip[data-field=${field}]`);
const option = (page, text) => page.locator('#ab-pop [role=option]').filter({ hasText: text }).first();
const options = async () => (await fetch(`${url}/v1/console/agent-options`)).json();
async function scrollToAgent(page) {
  await page.evaluate(() => {
    const panel = document.querySelector('.panel.agent');
    const header = document.querySelector('header.bar').getBoundingClientRect().bottom;
    scrollTo(0, panel.getBoundingClientRect().top + scrollY - Math.max(0, header) - 70);
  });
  await page.waitForTimeout(250);
}

try {
  await knobs({ reset: true });
  const snapshot = await options();
  check('the gateway serves the D1 agent-options shape', Array.isArray(snapshot.backends) && snapshot.current && Array.isArray(snapshot.catalogues)
    && Array.isArray(snapshot.profile_schema?.fields) && snapshot.matrix.every((cell) => Array.isArray(cell.checks)));

  // ---------- layout and safety header at every width, both themes ----------
  for (const [name, viewport] of handoffOnly ? [] : VIEWPORTS) {
    for (const scheme of ['light', 'dark']) {
      const session = await openConsole(viewport, scheme);
      const { page, seen } = session;
      await ready(page);
      await page.waitForTimeout(400);
      const tag = `${name} ${scheme}`;
      check(`${tag}: no horizontal page scroll`, await noOverflow(page));
      check(`${tag}: E-STOP visible and clickable without scrolling`, await estopReachable(page));
      check(`${tag}: every panel fits the viewport`, await page.evaluate(() => [...document.querySelectorAll('main > .panel')].every((panel) => panel.getBoundingClientRect().right <= innerWidth + 1)));
      await shot(page, `${name}-${scheme}-overview`);
      if (name !== 'desktop' || scheme === 'light') await shot(page, `${name}-${scheme}-full`, { fullPage: true });
      // Polling must not rebuild unchanged rows (2 console polls + ~15 telemetry polls).
      const churn = await page.evaluate(async () => {
        const watched = { joints: '#joints', motors: '#motors tbody', receipts: '#receipts tbody', banners: '#banners', chips: '.ab-chips', facts: '#new-run', items: '#items', faults: '#faults' };
        const counts = Object.fromEntries(Object.keys(watched).map((key) => [key, 0]));
        const observers = Object.entries(watched).map(([key, selector]) => {
          const node = document.querySelector(selector);
          const observer = new MutationObserver((records) => { counts[key] += records.filter((record) => record.type === 'childList').length; });
          if (node) observer.observe(node, { childList: true });
          return observer;
        });
        await new Promise((resolve) => setTimeout(resolve, 3300));
        observers.forEach((observer) => observer.disconnect());
        return counts;
      });
      check(`${tag}: polling leaves unchanged rows alone ${JSON.stringify(churn)}`, Object.values(churn).every((count) => count === 0));
      // Scrolled to the bottom: the E-STOP row stays pinned; on wrapped headers the rows above it scroll away.
      await page.evaluate(() => scrollTo(0, document.documentElement.scrollHeight));
      await page.waitForTimeout(350);
      check(`${tag}: E-STOP stays pinned and clickable after scrolling`, await estopReachable(page));
      const header = await page.evaluate(() => ({ top: document.querySelector('header.bar').getBoundingClientRect().top, collapsible: document.querySelector('header.bar').dataset.collapsible }));
      if (name === 'desktop') check(`${tag}: one-row header stays whole`, Math.abs(header.top) < 1 && header.collapsible === 'false');
      else check(`${tag}: wrapped header collapses to its E-STOP row`, header.collapsible === 'true' && header.top < -20);
      await shot(page, `${name}-${scheme}-scrolled`);
      if (name === 'desktop' && scheme === 'light') {
        await page.evaluate(() => scrollTo(0, 0));
        await page.keyboard.press('Tab');
        const ring = await page.evaluate(() => { const style = getComputedStyle(document.activeElement); return { tag: document.activeElement.tagName, outline: style.outlineStyle, width: style.outlineWidth }; });
        check(`keyboard focus shows a visible ring (${JSON.stringify(ring)})`, ring.outline !== 'none' && Number.parseFloat(ring.width) >= 2);
      }
      // Agent bar: a popover, the model picker and the profile form, as they look on this width and theme.
      await scrollToAgent(page);
      await chip(page, 'backend').click();
      await page.waitForSelector('#ab-pop[data-state=open] [role=option]');
      await page.waitForTimeout(250);
      await shot(page, `${name}-${scheme}-agent-bar-open`);
      await page.keyboard.press('Escape');
      await page.waitForFunction(() => document.getElementById('ab-pop').hidden);
      await chip(page, 'model').click();
      await page.waitForFunction(() => document.querySelectorAll('#ab-pop [role=option]').length >= 2);
      await page.waitForTimeout(250);
      check(`${tag}: no overflow with the model picker open`, await noOverflow(page));
      if (name === 'phone') check(`${tag}: E-STOP stays clickable above the sheet`, await estopReachable(page));
      await shot(page, `${name}-${scheme}-model-picker`);
      await page.keyboard.press('Escape');
      await page.waitForFunction(() => document.getElementById('ab-pop').hidden);
      await chip(page, 'settings').click();
      await page.waitForSelector('#ab-pop[data-state=open] .ab-profiles');
      await page.locator('#ab-pop .ab-btn', { hasText: '新建 profile' }).first().click();
      await page.waitForSelector('#ab-f-name');
      await page.waitForTimeout(250);
      check(`${tag}: profile form fits`, await noOverflow(page) && await page.locator('#ab-pop').evaluate((node) => node.scrollWidth <= node.clientWidth + 1));
      await shot(page, `${name}-${scheme}-profile-form`);
      await page.locator('#ab-pop .ab-pop-close').click();
      check(`${tag}: no page errors`, seen.errors.length === 0);
      await close(session);
    }
  }

  // ---------- the narrowest supported phone (360 px, design 16 §9.2): no sideways scroll, E-STOP pinned ----------
  for (const scheme of handoffOnly ? [] : ['light', 'dark']) {
    const session = await openConsole({ width: 360, height: 740 }, scheme);
    const { page } = session;
    await ready(page);
    check(`360 ${scheme}: no horizontal page scroll`, await noOverflow(page));
    check(`360 ${scheme}: E-STOP visible without scrolling`, await estopReachable(page));
    await chip(page, 'model').click();
    await page.waitForFunction(() => document.querySelectorAll('#ab-pop [role=option]').length >= 2);
    check(`360 ${scheme}: no overflow with the model sheet open`, await noOverflow(page));
    await page.keyboard.press('Escape');
    await page.evaluate(() => scrollTo(0, document.documentElement.scrollHeight));
    await page.waitForTimeout(300);
    check(`360 ${scheme}: E-STOP pinned at the bottom of the page`, await estopReachable(page));
    await close(session);
  }

  // ---------- reduced motion ----------
  if (!handoffOnly) {
    const session = await openConsole({ width: 1440, height: 1000 }, 'light', { reducedMotion: 'reduce' });
    const { page } = session;
    await ready(page);
    await chip(page, 'model').click();
    await page.waitForSelector('#ab-pop[data-state=open]');
    const durations = await page.evaluate(() => ['.ab-chip', '.pill', '#stop', '.ab-pop', '.bar .workspace-nav a', '.joint .measured'].map((selector) => {
      const node = document.querySelector(selector);
      return node ? getComputedStyle(node).transitionDuration.split(',').every((value) => Number.parseFloat(value) === 0) : true;
    }));
    check('reduced motion: chips, pills, buttons, popover, joint markers do not animate', durations.every(Boolean));
    await page.keyboard.press('Escape');
    await close(session);
  }

  // ---------- loading skeletons: console boot and the slow first agent-options call ----------
  if (!handoffOnly) {
    const session = await openConsole({ width: 1440, height: 1000 }, 'light', { delay: { '**/v1/console/agent-options': 2600, '**/v1/console/state': 1200, '**/v1/console/telemetry': 1200 } });
    const { page } = session;
    await page.waitForSelector('.ab[data-loading=true]');
    check('boot: skeleton rows stand in for joints and receipts', await page.evaluate(() => document.body.dataset.boot === 'true'
      && document.querySelectorAll('#joints .joint.is-skeleton').length >= 3 && document.querySelectorAll('#receipts tr.is-skeleton').length >= 1));
    await shot(page, 'boot-skeleton');
    await page.waitForFunction(() => document.querySelector('.ab-status-text')?.textContent.includes('正在读取 Agent 选项'));
    check('agent bar: skeleton chips while the first agent-options call runs', await page.evaluate(() => document.querySelector('.ab').getAttribute('aria-busy') === 'true'
      && getComputedStyle(document.querySelector('.ab-chip')).pointerEvents === 'none'));
    await shot(page, 'agent-skeleton');
    await page.waitForFunction(() => !document.querySelector('.ab').dataset.loading);
    check('agent bar: real values replace the skeleton, never the console-only fallback first', await page.evaluate(() => !document.querySelector('.ab').hasAttribute('aria-busy')
      && document.querySelector('.ab-chip[data-field=profile] .ab-chip-value').textContent.length > 1 && !document.querySelector('.ab-chip[data-field=profile] .ab-chip-value').textContent.includes('配置文件')));
    check('boot: skeleton rows are gone once data arrived', await page.evaluate(() => !document.body.dataset.boot && !document.querySelector('#joints .is-skeleton, #receipts .is-skeleton, #motors .is-skeleton')));
    await close(session);
  }

  // ---------- Agent bar follow-ups (design 16 §16) ----------
  if (!handoffOnly) {
    if (mode === 'claude_code') await withLease('set_effort', { effort: 'xhigh' });   // Claude Code: applies at the next session start
    const session = await openConsole({ width: 1440, height: 1000 }, 'light');
    const { page, seen } = session;
    await ready(page);
    const fresh = await options();
    if (mode === 'claude_code') {
      check('effort_next_session: the chip keeps the running effort and notes the next one', (await chip(page, 'effort').locator('.ab-chip-value').textContent()) === fresh.current.effort
        && (await chip(page, 'effort').locator('.ab-chip-scope').textContent()) === '下次 xhigh' && fresh.current.effort_next_session === 'xhigh'
        && (await chip(page, 'effort').getAttribute('aria-label')).includes('下次会话 xhigh'));
    }
    // Profile manager: error codes, file names instead of absolute paths, compatibility per backend, cached listings.
    await chip(page, 'settings').click();
    await page.waitForSelector('#ab-pop[data-state=open] .ab-profiles');
    await page.waitForTimeout(300);
    const broken = fresh.profiles.find((profile) => profile.valid === false);
    if (broken) {
      const row = page.locator('#ab-pop .ab-profile[data-invalid=true]').first();
      const text = await row.textContent();
      check('invalid profile: its error code is shown as a code and the path is shortened to the file name', (await row.locator('.ab-code').textContent()) === broken.error
        && text.includes(`${broken.id}.local.json`) && !/\/(private|tmp|fixture)\//.test(text));
    }
    const valid = fresh.profiles.filter((profile) => profile.valid !== false);
    check('every valid profile lists its compatibility with every backend', await page.locator('#ab-pop .ab-compat li').count() === valid.length * fresh.backends.length);
    const listed = fresh.catalogues.find((item) => item.status === 'ok' && item.count);
    if (listed) check('cached model listings are summarised per profile', (await page.locator('#ab-pop .ab-profile', { hasText: listed.profile }).first().textContent()).includes(`${listed.count} 个模型`));
    await shot(page, 'profile-manager');
    // Profile form: help text comes from profile_schema.fields, with the contract field name.
    await page.locator('#ab-pop .ab-btn', { hasText: '新建 profile' }).first().click();
    await page.waitForSelector('#ab-f-name');
    const help = new Map(fresh.profile_schema.fields.map((field) => [field.name, field.help]));
    const shown = await page.evaluate(() => [...document.querySelectorAll('#ab-pop .ab-help-field')].map((node) => [node.textContent, node.parentElement.textContent.slice(node.textContent.length)]));
    check(`profile form help is generated from the contract (${shown.length} fields)`, shown.length >= 10
      && shown.every(([name, text]) => help.has(name) && text.endsWith(help.get(name)))
      && ['name', 'credential.api_key', 'compat.tool_result_images', 'compat.tool_schema_max_depth', 'defaults.effort', 'wires.<wire>.models_url'].every((name) => shown.some(([field]) => field === name)));
    check('defaults.effort is a choice of the contract efforts', JSON.stringify(await page.locator('#ab-f-default_effort option').evaluateAll((nodes) => nodes.map((node) => node.value)))
      === JSON.stringify(['', ...fresh.profile_schema.efforts]));
    check('auth scheme choices come from the contract', JSON.stringify(await page.locator('#ab-pop .ab-seg[aria-label="anthropic_messages auth_scheme"] [role=radio]').evaluateAll((nodes) => nodes.map((node) => node.textContent)))
      === JSON.stringify(fresh.profile_schema.auth_schemes));
    await page.locator('#ab-pop .ab-details:not(.ab-details--help) > summary').click();
    await shot(page, 'profile-form-contract-help', { fullPage: false });
    await page.locator('#ab-pop .ab-pop-close').click();

    // thinking_display: the server's list already contains the default; it must appear once.
    if (fixture) {
      await chip(page, 'backend').click();
      await option(page, 'OpenCode').click();
      await page.waitForFunction(() => document.querySelector('.ab-tray-check')?.textContent.includes('组合可用'));
    }
    if (fixture || mode === 'opencode') {
      await chip(page, 'settings').click();
      await page.waitForSelector('#ab-pop .ab-seg[aria-label^="思考显示"]');
      await page.waitForTimeout(300);
      const radios = await page.locator('#ab-pop .ab-seg[aria-label^="思考显示"] [role=radio]').evaluateAll((nodes) => nodes.map((node) => node.textContent));
      check(`thinking_display offers the default exactly once (${radios.join(' | ')})`, radios.length === 3 && radios.filter((text) => text.includes('默认')).length === 1);
      await shot(page, 'thinking-display');
      await page.locator('#ab-pop .ab-pop-close').click();
      if (fixture) await page.locator('.ab-tray .ab-btn', { hasText: '放弃' }).click();
    }

    // Speed: requested vs served, honestly (real Codex gateway: the fake relay accepts priority and serves default).
    if (mode === 'codex' && !fixture) {
      await chip(page, 'speed').click();
      await option(page, 'Fast').click();
      await page.waitForFunction(() => document.querySelector('.ab-status-text').textContent.includes('已应用 速度'));
      await page.waitForFunction(() => document.querySelector('.ab-chip[data-field=speed] .ab-chip-scope').textContent === '实际 正常');
      check('speed chip: requested Fast, the provider served the default tier (badge and accessible name)',
        (await chip(page, 'speed').locator('.ab-chip-value').textContent()).startsWith('Fast') && (await chip(page, 'speed').getAttribute('aria-label')).includes('供应商实际以 正常 服务'));
      check('the receipt note says what the provider served', (await page.locator('.ab-status-text').textContent()).includes('实际以 default 档位服务'));
      await chip(page, 'speed').click();
      await page.waitForSelector('#ab-pop[data-state=open] [role=option]');
      await page.waitForTimeout(300);
      check('speed popover: the probed tier is marked as served differently', (await option(page, 'Fast').textContent()).includes('实际以 正常 服务')
        && (await page.locator('#ab-pop .ab-pop-sub').textContent()).includes('供应商实际以 正常 服务'));
      await shot(page, 'speed-served-tier');
      await page.keyboard.press('Escape');
    }
    check('no page errors in the follow-up checks', seen.errors.length === 0);
    await close(session);
  }

  // ---------- apply_agent_settings answering handoff_required (fixture: a stale snapshot claims live effort) ----------
  if (fixture) {
    await knobs({ reset: true });
    await knobs({ stale_effort_scope: true });
    const session = await openConsole({ width: 1440, height: 1000 }, 'light');
    const { page, seen } = session;
    await ready(page);
    check('stale snapshot: effort looks live', (await chip(page, 'effort').locator('.ab-chip-scope').textContent()) === '下一轮');
    const controls = [];
    page.on('request', (request) => { if (request.url().endsWith('/v1/control') && request.method() === 'POST') controls.push(request.postData() || ''); });
    await chip(page, 'effort').click();
    await option(page, 'max').click();
    await page.waitForSelector('.ab-tray[data-open=true]');
    check('handoff_required: the request was sent once through apply_agent_settings', controls.filter((body) => body.includes('"command":"apply_agent_settings"')).length === 1);
    check('handoff_required: the diff is offered as a session change, nothing reported as not applied',
      (await page.locator('.ab-tray .ab-diff').textContent()).includes('推理强度') && (await page.locator('.ab-tray .ab-diff').textContent()).includes('max')
      && (await page.locator('.ab-status-text').textContent()).includes('需要新会话') && (await page.locator('.ab-alert').textContent()) === ''
      && (await chip(page, 'effort').getAttribute('data-state')) === 'draft');
    await page.waitForFunction(() => document.querySelector('.ab-tray-check').textContent.includes('组合可用'));
    await shot(page, 'handoff-required-tray');
    check('no page errors (handoff_required)', seen.errors.length === 0);
    await close(session);
    await knobs({ reset: true });
  }
  // ---------- a real handoff from the Agent bar (D2: successor run on the same port, or one fallback run) ----------
  if (handoffMode && !fixture) {
    const session = await openConsole({ width: 1440, height: 1000 }, 'light');
    const { page, seen } = session;
    await ready(page);
    const before = await (await fetch(`${url}/v1/run`)).json();
    // Choose the change at once, possibly while the startup turn still runs (blocker agent_turn_running): the tray must
    // pick up the cleared blocker as soon as the run state changes, well before the bar's 15 s options refresh.
    const blockedAtStart = !(await options()).handoff.allowed;
    await chip(page, 'backend').click();
    await option(page, 'Codex').click();
    await page.waitForSelector('#ab-pop[data-kind=profile][data-state=open]');
    await option(page, 'Provider Beta').click();
    await page.waitForFunction(() => document.querySelector('.ab-tray-check')?.textContent.includes('组合可用'), null, { timeout: 30000 });
    if (blockedAtStart) check('handoff: the tray names the blocker while the turn runs', (await page.locator('.ab-tray-blockers').textContent()).includes('需要 Agent 空闲')
      || (await options()).handoff.allowed);
    const clearedAt = Date.now();
    await page.waitForFunction(() => !document.querySelector('.ab-tray .ab-btn--primary').disabled, null, { timeout: 9000 });
    check(`handoff: the switch enables once the blocker clears (${blockedAtStart ? 'was blocked' : 'not blocked'}, ${Date.now() - clearedAt} ms)`, true);
    check('handoff: the tray lists the change and the real dry run passes', (await page.locator('.ab-tray .ab-diff').textContent()).includes('Codex')
      && (await chip(page, 'model').locator('.ab-chip-value').textContent()) === 'gpt-6.1-sol' && await page.locator('.ab-tray .ab-btn--primary').isEnabled());
    await scrollToAgent(page);
    await shot(page, 'handoff-tray');
    await page.locator('.ab-tray .ab-btn--primary').click();
    await page.waitForSelector('dialog.ab-dialog[open]');
    await page.waitForTimeout(250);
    check('handoff dialog: torque held, nothing moves, E-STOP clickable', (await page.locator('.ab-facts').textContent()).includes('不发出任何运动') && await estopReachable(page));
    await shot(page, 'handoff-dialog');
    const receipt = page.waitForResponse((response) => response.url().endsWith('/v1/control') && (response.request().postData() || '').includes('"handoff_run"'));
    await page.locator('.ab-dialog-foot .ab-btn--primary').click();
    const body = await (await receipt).json();
    check(`handoff_run accepted by the real gateway (${JSON.stringify(body).slice(0, 160)})`, body.status === 'handoff_accepted' && typeof body.successor_run_id === 'string');
    await page.waitForSelector('.ab-overlay:not([hidden])');
    await page.waitForTimeout(150);
    await shot(page, 'handoff-overlay');
    if (realMode === 'handoff') {
      await page.waitForFunction((id) => document.getElementById('run-id').textContent.includes(id), body.successor_run_id, { timeout: 60000 });
      await page.waitForFunction(() => document.querySelector('.ab-chip[data-field=backend] .ab-chip-value').textContent === 'Codex', null, { timeout: 30000 });
      check('handoff: the page follows the successor run on the same port', (await (await fetch(`${url}/v1/run`)).json()).run_id === body.successor_run_id
        && body.successor_run_id !== before.run_id);
      check('handoff: the bar shows the new session (Codex · Provider Beta · gpt-6.1-sol)', (await chip(page, 'profile').locator('.ab-chip-value').textContent()) === 'Provider Beta'
        && (await chip(page, 'model').locator('.ab-chip-value').textContent()) === 'gpt-6.1-sol' && await chip(page, 'speed').isVisible());
      await page.waitForFunction(() => document.querySelector('.ab-overlay').hidden, null, { timeout: 10000 });
      check('handoff: the toast names the successor run', (await page.locator('.ab-toast').textContent()).includes(body.successor_run_id));
      await page.waitForFunction(() => document.getElementById('lease-state').textContent.includes('本窗口'));
      check('handoff: the lease is taken again on the new run', true);
      await shot(page, 'handoff-done');
    } else {
      await page.waitForSelector('.ab-banner:not([hidden])', { timeout: 60000 });
      const run = await (await fetch(`${url}/v1/run`)).json();
      check(`fallback: the banner names the recorded reason (${(await page.locator('.ab-banner').textContent()).slice(0, 120)})`,
        (await page.locator('.ab-banner').textContent()).includes(run.handoff?.fallback?.reason?.slice(0, 40) || '\u0000') && (await page.locator('.ab-banner').getAttribute('role')) === 'alert');
      check('fallback: the page is on the fallback run with the old backend', run.run_id !== body.successor_run_id && run.run_id !== before.run_id
        && (await chip(page, 'backend').locator('.ab-chip-value').textContent()) === 'Claude Code');
      await shot(page, 'handoff-fallback-banner');
    }
    check('no page errors in the handoff', seen.errors.length === 0);
    await close(session);
  }
  console.log(checks.map((name) => `  ok ${name}`).join('\n'));
  console.log(`PASS: ${checks.length} checks (${prefix.slice(0, -1)}); screenshots in ${output}`);
} finally {
  await knobs({ reset: true }).catch(() => {});
  await browser.close();
}
