// Browser check of the console Agent bar against the SYNTHETIC fixture (web/fixture-server.py, never port 8781).
// Build first (npm --prefix web run build), start the fixture, then:
//   NI_RUA_URL=http://127.0.0.1:18795 PLAYWRIGHT_MODULE=/path/to/playwright/index.mjs node web/check-agent-settings.mjs
// Optional: NI_RUA_BROWSER_EXECUTABLE (e.g. a system Chrome), NI_RUA_SCREENSHOT_DIR.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const url = process.env.NI_RUA_URL || 'http://127.0.0.1:18795';
assert.notEqual(new URL(url).port, '8781', 'never run browser checks against the robot gateway');
const output = process.env.NI_RUA_SCREENSHOT_DIR || '/tmp/ni-rua-agent-settings-browser';
await fs.mkdir(output, { recursive: true });
const SECRET = 'sk-test-PLAYWRIGHT-SECRET-0001';
const checks = [];
const check = (name, condition) => { assert.ok(condition, name); checks.push(name); };
const knobs = async (body) => (await fetch(`${url}/fixture/agent`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })).json();
const browser = await chromium.launch({ headless: true, ...(process.env.NI_RUA_BROWSER_EXECUTABLE ? { executablePath: process.env.NI_RUA_BROWSER_EXECUTABLE } : {}) });

async function openConsole(viewport, colorScheme, { reset = true, setup = null } = {}) {
  if (reset) await knobs({ reset: true });
  if (setup) await knobs(setup);
  const context = await browser.newContext({ viewport, colorScheme, reducedMotion: 'no-preference' });
  const page = await context.newPage();
  page.setDefaultTimeout(15000);
  const seen = { errors: [], controls: [], profiles: [], failedRuns: 0 };
  page.on('pageerror', (error) => seen.errors.push(error.message));
  page.on('request', (request) => {
    if (request.method() !== 'POST') return;
    if (request.url().endsWith('/v1/control')) seen.controls.push(request.postData() || '');
    if (request.url().includes('/v1/console/profiles')) seen.profiles.push({ url: request.url(), body: request.postData() || '' });
  });
  page.on('requestfailed', (request) => { if (request.url().endsWith('/v1/run')) seen.failedRuns += 1; });
  await page.goto(`${url}/console`);
  await page.waitForFunction(() => document.querySelector('.ab-chip[data-field=model] .ab-chip-value')?.textContent.length > 1);
  await page.waitForFunction(() => document.getElementById('lease-state').textContent.includes('本窗口'));
  return { page, context, seen };
}
const chip = (page, field) => page.locator(`.ab-chip[data-field=${field}]`);
const chipValue = (page, field) => chip(page, field).locator('.ab-chip-value').textContent();
const noOverflow = (page) => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1);
const focusedIn = (page, selector) => page.evaluate((value) => Boolean(document.activeElement?.closest(value)), selector);
const option = (page, text) => page.locator('#ab-pop [role=option]').filter({ hasText: text }).first();
async function scrollToBar(page) {
  await page.evaluate(() => {
    const panel = document.querySelector('.panel.agent');
    const header = document.querySelector('header.bar').getBoundingClientRect().height;
    scrollTo(0, panel.getBoundingClientRect().top + scrollY - header - 8);
  });
  await page.waitForTimeout(250);
}
async function shot(page, name) { await page.screenshot({ path: path.join(output, `${name}.png`) }); }
async function waitStatus(page, text) { await page.waitForFunction((value) => document.querySelector('.ab-status-text').textContent.includes(value), text); }
async function waitAlert(page, text) { await page.waitForFunction((value) => document.querySelector('.ab-alert').textContent.includes(value), text); }

try {
  // ---------- desktop 1440, light: keyboard, optimistic receipts, rollback, free text, profiles, handoff ----------
  {
    const { page, context, seen } = await openConsole({ width: 1440, height: 1000 }, 'light');
    check('existing console controls are still present', await page.locator('#estop').count() === 1 && await page.locator('#speed-form').count() === 1
      && await page.locator('#claim').count() === 1 && await page.locator('#retry-model').count() === 1);
    check('old model / effort / tier rows are replaced by the Agent bar', await page.locator('#model-form, #effort-form, #tier-row, #model-options').count() === 0);
    check('bar shows backend, profile, model, effort', (await chipValue(page, 'backend')) === 'Claude Code' && (await chipValue(page, 'profile')) === 'Provider Alpha'
      && (await chipValue(page, 'model')) === 'qwen3.8-flash' && (await chipValue(page, 'effort')) === 'high');
    check('scopes are labelled', (await chip(page, 'model').locator('.ab-chip-scope').textContent()) === '下一轮'
      && (await chip(page, 'model').getAttribute('aria-label')).includes('下一轮生效')
      && (await chip(page, 'effort').locator('.ab-chip-scope').textContent()) === '新会话' && (await chip(page, 'backend').locator('.ab-chip-scope').textContent()) === '新会话');
    check('speed chip only for backends with a service tier', await chip(page, 'speed').isHidden());
    check('desktop 1440: no horizontal overflow', await noOverflow(page));
    await page.waitForFunction(() => document.querySelector('.ab-chip[data-field=model] .ab-chip-meta').childElementCount > 0);
    await scrollToBar(page);
    await shot(page, 'desktop-light-bar');

    // Keyboard-only model selection: Enter opens, focus lands in the combobox, arrows move, Enter applies.
    await chip(page, 'model').focus();
    await page.keyboard.press('Enter');
    await page.waitForSelector('#ab-pop[data-state=open] [role=option]');
    check('model popover is an anchored popover on desktop', !(await page.locator('#ab-pop').evaluate((node) => node.classList.contains('ab-pop--sheet'))));
    check('focus moves into the combobox', await page.evaluate(() => document.activeElement?.getAttribute('role') === 'combobox' && document.activeElement.getAttribute('aria-expanded') === 'true'));
    await page.waitForFunction(() => document.querySelectorAll('#ab-pop [role=option]').length >= 6);
    check('capability badges: image, tools, context, efforts, unknown shown as unknown', (await option(page, 'kimi-k2.6').textContent()).includes('强度 low · high · max')
      && (await option(page, 'glm-text-only').textContent()).includes('无图像输入') && (await option(page, 'mystery-model').textContent()).includes('图像未声明')
      && (await option(page, 'qwen3.8-flash').textContent()).includes('256k'));
    const disabled = option(page, 'gpt-x');
    check('a disabled row exposes its reason as text', (await disabled.getAttribute('aria-disabled')) === 'true' && (await disabled.textContent()).includes('仅在 openai_responses 上提供'));
    await page.keyboard.press('End');
    check('End reaches the disabled row (aria-activedescendant)', await page.evaluate(() => document.getElementById(document.activeElement.getAttribute('aria-activedescendant'))?.textContent.includes('gpt-x')));
    const controlsBefore = seen.controls.length;
    await page.keyboard.press('Enter');
    await waitAlert(page, '仅在 openai_responses 上提供');
    check('Enter on a disabled row sends nothing and says why', seen.controls.length === controlsBefore);
    await page.keyboard.press('Home');
    await page.keyboard.press('ArrowDown');
    await page.keyboard.press('Enter');
    await waitStatus(page, '已应用 模型 deepseek-flash');
    check('keyboard selection applied through apply_agent_settings', seen.controls.some((body) => body.includes('"command":"apply_agent_settings"') && body.includes('"model":"deepseek-flash"')));
    check('focus returns to the opening chip', await page.evaluate(() => document.activeElement?.dataset.field === 'model'));
    await page.waitForFunction(() => document.getElementById('ab-pop').hidden);
    check('popover closed after the selection', true);

    // Optimistic value survives a poll; the receipt then confirms it.
    await knobs({ apply_delay_s: 3 });
    await chip(page, 'model').click();
    await option(page, 'kimi-k2.6').click();
    await page.waitForFunction(() => { const node = document.querySelector('.ab-chip[data-field=model]'); return node.dataset.state === 'pending' && node.querySelector('.ab-chip-value').textContent === 'kimi-k2.6'; }, null, { timeout: 500 });
    check('optimistic value shown at once (next frame) with a pending state', true);
    await page.waitForResponse((response) => response.url().endsWith('/v1/console/state'));
    await page.waitForResponse((response) => response.url().endsWith('/v1/console/state'));
    await page.waitForTimeout(80);
    check('optimistic value survives a poll that still reports the old model', (await chipValue(page, 'model')) === 'kimi-k2.6' && (await chip(page, 'model').getAttribute('data-state')) === 'pending');
    await waitStatus(page, '已应用 模型 kimi-k2.6');
    check('receipt confirms the value', (await chip(page, 'model').getAttribute('data-state')) === 'confirmed');
    await knobs({ apply_delay_s: 0 });

    // Refused model: reverts and keeps the backend's message.
    await chip(page, 'model').click();
    await page.locator('#ab-pop input[role=combobox]').fill('bad-model');
    await page.keyboard.press('Enter');
    await waitAlert(page, 'agent_model_rejected:bad-model');
    await page.waitForFunction(() => document.querySelector('.ab-chip[data-field=model] .ab-chip-value').textContent === 'kimi-k2.6');
    check('rejected value reverts with the message (role=alert)', (await page.locator('.ab-alert').getAttribute('role')) === 'alert');

    // Manual model id: free text row, applied, noted as not listed.
    await chip(page, 'model').click();
    await page.locator('#ab-pop input[role=combobox]').fill('my-custom-model');
    check('free-text row offers the typed id', (await option(page, '使用').textContent()).includes('不在供应商列表中，由供应商裁决'));
    await page.keyboard.press('Enter');
    await waitStatus(page, '已应用 模型 my-custom-model');
    check('manual model id applies', (await chipValue(page, 'model')) === 'my-custom-model');

    // Lost connection: unconfirmed; retry reuses the request_id; dismiss restores the server value.
    await chip(page, 'model').click();
    await page.locator('#ab-pop input[role=combobox]').fill('net-model');
    await page.keyboard.press('Enter');
    await waitStatus(page, '未确认');
    await page.locator('.ab-status-actions .ab-btn', { hasText: '重试' }).click();
    await waitStatus(page, '未确认');
    const netBodies = seen.controls.filter((body) => body.includes('"model":"net-model"')).map((body) => JSON.parse(body).request_id);
    check('retry after an unconfirmed send reuses the same request_id', netBodies.length === 2 && netBodies[0] === netBodies[1]);
    await page.locator('.ab-status-actions .ab-btn', { hasText: '放弃' }).click();
    await page.waitForFunction(() => document.querySelector('.ab-chip[data-field=model] .ab-chip-value').textContent === 'my-custom-model');
    check('dismissing an unconfirmed change shows the server value', true);

    // Another window holding the lease: the bar is read-only.
    const observerContext = await browser.newContext({ viewport: { width: 1200, height: 900 } });
    const observer = await observerContext.newPage();
    await observer.goto(`${url}/console`);
    await observer.waitForFunction(() => document.querySelector('.ab-status-text')?.textContent.includes('另一窗口持有控制'));
    check('another window renders the bar read-only', await observer.locator('.ab-chip[data-field=model]').isDisabled() && await observer.locator('.ab-chip[data-field=backend]').isDisabled());
    await observerContext.close();

    // Profile manager: create (key pasted once, never shown again), edit (empty key keeps it), remove.
    await chip(page, 'settings').click();
    await page.locator('#ab-pop .ab-btn', { hasText: '新建 profile' }).click();
    await page.locator('#ab-f-name').fill('Example Gateway');
    check('id is derived from the name', (await page.locator('#ab-f-id').inputValue()) === 'example-gateway');
    check('key field is a password field', (await page.locator('#ab-f-api_key').getAttribute('type')) === 'password');
    await page.locator('.ab-form-actions .ab-btn--primary').click();
    check('client-side validation names the missing key', (await page.locator('#ab-err-api_key').textContent()).includes('请粘贴密钥'));
    await page.locator('#ab-f-api_key').fill(SECRET);
    await page.locator('#ab-f-url-anthropic_messages').fill('https://gateway.example/anthropic');
    await scrollToBar(page);
    await shot(page, 'desktop-light-profile-form');
    await page.locator('.ab-form-actions .ab-btn--primary').click();
    await page.waitForFunction(() => document.querySelector('#ab-pop .ab-form-msg')?.textContent.includes('已保存'));
    check('saved profile is listed with the key status, not the key', (await page.locator('#ab-pop .ab-profile', { hasText: 'Example Gateway' }).textContent()).includes('密钥已保存'));
    check('the key is nowhere in the page after saving', !(await page.content()).includes(SECRET));
    await page.locator('#ab-pop .ab-profile', { hasText: 'Example Gateway' }).locator('.ab-btn', { hasText: '编辑' }).click();
    check('edit form never shows the stored key', (await page.locator('#ab-f-api_key').inputValue()) === '' && (await page.locator('#ab-f-api_key').getAttribute('placeholder')).includes('留空则保留'));
    check('edit form keeps the id fixed', (await page.locator('#ab-f-id').getAttribute('readonly')) !== null);
    await page.locator('.ab-form-actions .ab-btn--primary').click();
    await page.waitForFunction(() => document.querySelector('#ab-pop .ab-form-msg')?.textContent.includes('已保存'));
    const saves = seen.profiles.filter((item) => item.url.endsWith('/v1/console/profiles')).map((item) => JSON.parse(item.body));
    check('profile writes go to /v1/console/profiles with the lease fields', saves.length === 2 && saves.every((body) => body.client_id && body.writer_token && body.id === 'example-gateway'));
    check('first save carries the key, the edit keeps it (keep_key, no credential)', saves[0].document.credential.api_key === SECRET && saves[1].keep_key === true && !('credential' in saves[1].document));
    check('no /v1/control body ever contains the key', seen.controls.every((body) => !body.includes(SECRET)));
    check('in-use profile cannot be removed', await page.locator('#ab-pop .ab-profile', { hasText: 'Provider Alpha' }).locator('.ab-btn', { hasText: '移除' }).isDisabled());
    await page.locator('#ab-pop .ab-profile', { hasText: 'Example Gateway' }).locator('.ab-btn', { hasText: '移除' }).click();
    await page.locator('#ab-pop .ab-btn', { hasText: '确认移除' }).click();
    await page.waitForFunction(() => document.querySelector('#ab-pop .ab-form-msg')?.textContent.includes('已移除'));
    check('removed profile disappears', await page.locator('#ab-pop .ab-profile', { hasText: 'Example Gateway' }).count() === 0);
    await page.keyboard.press('Escape');

    // Session change, keyboard only: backend via arrows, profile via type-ahead.
    await chip(page, 'backend').focus();
    await page.keyboard.press('Enter');
    await page.waitForSelector('#ab-pop[data-state=open] [role=listbox]');
    check('backend listbox takes focus on the selected row', await page.evaluate(() => document.activeElement?.getAttribute('role') === 'listbox'
      && document.getElementById(document.activeElement.getAttribute('aria-activedescendant'))?.textContent.includes('Claude Code')));
    await page.keyboard.press('ArrowDown');
    await page.keyboard.press('Enter');
    await page.waitForSelector('.ab-tray[data-open=true]');
    check('tray lists the session change', (await page.locator('.ab-tray').textContent()).includes('Claude Code') && (await page.locator('.ab-tray').textContent()).includes('OpenCode'));
    await chip(page, 'profile').focus();
    await page.keyboard.press('Enter');
    await page.waitForSelector('#ab-pop[data-state=open] [role=listbox]');
    check('incompatible profiles are disabled with the reason', (await option(page, 'Broken Example').getAttribute('aria-disabled')) === 'true'
      && (await option(page, 'Broken Example').textContent()).includes('credential.api_key'));
    await page.keyboard.type('o');
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => document.querySelector('.ab-tray-check').textContent.includes('组合可用'));
    check('profile chosen by type-ahead; profile defaults pre-fill the model', (await chipValue(page, 'profile')) === 'OpenCode Console' && (await chipValue(page, 'model')) === 'kimi-k2.6');
    check('switch button enabled only after the dry run passes', await page.locator('.ab-tray .ab-btn--primary').isEnabled());
    await scrollToBar(page);
    await shot(page, 'desktop-light-tray');
    await page.locator('.ab-tray .ab-btn--primary').click();
    await page.waitForSelector('dialog.ab-dialog[open]');
    check('confirmation dialog focuses the prompt', await page.evaluate(() => document.activeElement?.id === 'ab-handoff-prompt'));
    for (let index = 0; index < 5; index += 1) await page.keyboard.press('Tab');
    check('focus stays trapped in the dialog', await focusedIn(page, 'dialog.ab-dialog'));
    check('dialog states the speed cap carry-over and that nothing moves', (await page.locator('.ab-facts').textContent()).includes('速度上限 0.90') && (await page.locator('.ab-facts').textContent()).includes('不发出任何运动'));
    check('E-STOP stays clickable while the dialog is open', await page.evaluate(() => { const button = document.getElementById('estop').getBoundingClientRect(); return document.elementFromPoint(button.left + button.width / 2, button.top + button.height / 2)?.closest('#estop') !== null; }));
    await page.locator('#ab-handoff-prompt').fill('观察桌面');
    await page.waitForTimeout(300);
    await shot(page, 'desktop-light-handoff-dialog');
    await page.locator('.ab-dialog-foot .ab-btn--primary').click();
    await page.waitForSelector('.ab-overlay:not([hidden])');
    const handoff = JSON.parse(seen.controls.find((body) => body.includes('"command":"handoff_run"')));
    check('handoff_run carries the exact selection and the prompt', handoff.backend === 'opencode' && handoff.profile === 'opencode-console' && handoff.wire_api === 'anthropic_messages'
      && handoff.model === 'kimi-k2.6' && handoff.prompt === '观察桌面' && handoff.thinking_display === null);
    await page.waitForFunction(() => document.querySelector('.ab-ov-note').textContent.includes('网关暂时不可达'));
    check('overlay survives connection errors', seen.failedRuns > 0 && await page.locator('.ab-overlay').isVisible());
    await shot(page, 'desktop-light-handoff-overlay');
    await page.waitForFunction(() => document.querySelector('.ab-toast')?.textContent.includes('synthetic-web-2'), null, { timeout: 20000 });
    await page.waitForFunction(() => document.getElementById('run-id').textContent.includes('synthetic-web-2'));
    check('handoff ends on the successor run id', (await chipValue(page, 'backend')) === 'OpenCode');
    await page.waitForFunction(() => document.querySelector('.ab-overlay').hidden);
    check('effort is live on OpenCode', (await chip(page, 'effort').locator('.ab-chip-scope').textContent()) === '下一轮');
    await chip(page, 'effort').click();
    await option(page, 'max').click();
    await waitStatus(page, '已应用 推理强度 max');
    check('live effort applied after the handoff', (await chipValue(page, 'effort')) === 'max');

    // Successor fails to start: fallback run on the old settings, banner with the recorded reason.
    await knobs({ handoff_outcome: 'fallback', handoff_down_s: 1.0 });
    await chip(page, 'backend').click();
    await option(page, 'Claude Code').click();
    await page.waitForFunction(() => document.querySelector('.ab-tray-check').textContent.includes('组合可用'));
    await page.locator('.ab-tray .ab-btn--primary').click();
    await page.locator('.ab-dialog-foot .ab-btn--primary').click();
    await page.waitForSelector('.ab-banner:not([hidden])', { timeout: 20000 });
    check('fallback shows the banner with the reason', (await page.locator('.ab-banner').textContent()).includes('successor_start_failed') && (await page.locator('.ab-banner').getAttribute('role')) === 'alert');
    check('fallback run comes back on the old backend', (await chipValue(page, 'backend')) === 'OpenCode');
    check('no page errors on desktop', seen.errors.length === 0);
    await page.close(); await context.close();
  }

  // ---------- desktop dark and tablet: layout, anchored popovers, screenshots ----------
  for (const [name, viewport, scheme] of [['desktop-dark', { width: 1440, height: 1000 }, 'dark'], ['tablet-light', { width: 768, height: 1000 }, 'light'], ['tablet-dark', { width: 768, height: 1000 }, 'dark']]) {
    const blocked = name === 'tablet-dark';
    const { page, context, seen } = await openConsole(viewport, scheme, { setup: blocked ? { blockers: ['turn_running'] } : { handoff_down_s: 0.6, handoff_finalize_s: 0.3 } });
    check(`${name}: no horizontal overflow`, await noOverflow(page));
    await scrollToBar(page);
    await shot(page, `${name}-bar`);
    await chip(page, 'model').click();
    await page.waitForSelector('#ab-pop[data-state=open] [role=option]');
    await page.waitForFunction(() => document.querySelectorAll('#ab-pop [role=option]').length >= 6);
    check(`${name}: popover anchored, not a sheet`, !(await page.locator('#ab-pop').evaluate((node) => node.classList.contains('ab-pop--sheet'))));
    check(`${name}: popover stays inside the viewport`, await page.locator('#ab-pop').evaluate((node) => { const box = node.getBoundingClientRect(); return box.left >= 0 && box.right <= innerWidth; }));
    await page.waitForTimeout(200);
    await shot(page, `${name}-model`);
    await page.keyboard.press('Escape');
    await chip(page, 'backend').click();
    await option(page, 'Codex').click();
    await page.waitForSelector('#ab-pop[data-state=open] [role=option]');
    await option(page, 'Provider Beta').click();
    await page.waitForFunction(() => document.querySelector('.ab-tray-check').textContent.includes('组合可用'));
    check(`${name}: no overflow with the tray open`, await noOverflow(page));
    await chip(page, 'model').click();
    await page.waitForFunction(() => document.querySelector('.ab-cat-summary')?.textContent.includes('模型列表不可用'));
    check(`${name}: unavailable catalogue shows the stale list and a retry`, (await page.locator('.ab-cat-summary').textContent()).includes('缓存于 3 小时前')
      && await page.locator('#ab-pop .ab-btn', { hasText: '重试' }).isVisible());
    await scrollToBar(page);
    await shot(page, `${name}-catalogue-error`);
    await page.locator('#ab-pop .ab-btn', { hasText: '重试' }).click();
    await page.waitForFunction(() => document.querySelector('.ab-cat-summary').textContent.includes('个模型'));
    check(`${name}: retry recovers the catalogue`, true);
    await page.keyboard.press('Escape');
    if (blocked) {
      check(`${name}: handoff blockers are listed and the switch stays disabled`, (await page.locator('.ab-tray-blockers').textContent()).includes('需要 Agent 空闲')
        && await page.locator('.ab-tray .ab-btn--primary').isDisabled());
    } else if (name === 'tablet-light') {
      // Hand off to Codex, then the speed chip: a refused tier reverts with the provider's words, a supported one applies.
      await page.locator('.ab-tray .ab-btn--primary').click();
      await page.locator('.ab-dialog-foot .ab-btn--primary').click();
      await page.waitForFunction(() => document.getElementById('run-id').textContent.includes('synthetic-web-2'), null, { timeout: 20000 });
      await page.waitForFunction(() => !document.querySelector('.ab-chip[data-field=speed]').hidden);
      check(`${name}: speed chip appears for Codex with its scope`, (await chip(page, 'speed').locator('.ab-chip-scope').textContent()) === '下次请求');
      await chip(page, 'speed').click();
      check(`${name}: tiers say whether they were probed`, (await option(page, 'Ultrafast').textContent()).includes('首次选择会先检测'));
      await option(page, 'Ultrafast').click();
      await waitAlert(page, '不支持 ultrafast');
      await page.waitForFunction(() => document.querySelector('.ab-chip[data-field=speed] .ab-chip-value').textContent === '正常');
      check(`${name}: unsupported tier reverts with the message`, true);
      await chip(page, 'speed').click();
      await option(page, 'Fast').click();
      await waitStatus(page, '已应用 速度 Fast · priority');
      check(`${name}: supported tier applies`, (await chipValue(page, 'speed')) === 'Fast · priority');
    }
    check(`${name}: no page errors`, seen.errors.length === 0);
    await page.close(); await context.close();
  }

  // ---------- phone 390: bottom sheet, E-STOP reachable, focus management, stacked tray buttons ----------
  for (const scheme of ['light', 'dark']) {
    const { page, context, seen } = await openConsole({ width: 390, height: 844 }, scheme);
    check(`phone ${scheme}: no horizontal overflow`, await noOverflow(page));
    check(`phone ${scheme}: model chip spans the full width`, await page.evaluate(() => {
      const chip = document.querySelector('.ab-chip[data-field=model]').getBoundingClientRect();
      const bar = document.querySelector('.ab-chips').getBoundingClientRect();
      return Math.abs(chip.width - bar.width) < 2;
    }));
    await scrollToBar(page);
    await shot(page, `phone-${scheme}-bar`);
    await chip(page, 'model').click();
    await page.waitForSelector('#ab-pop[data-state=open] [role=option]');
    await page.waitForTimeout(300);
    check(`phone ${scheme}: popover is a bottom sheet`, await page.locator('#ab-pop').evaluate((node) => node.classList.contains('ab-pop--sheet')
      && Math.abs(node.getBoundingClientRect().bottom - innerHeight) < 2 && node.getBoundingClientRect().width >= innerWidth - 1));
    check(`phone ${scheme}: E-STOP stays visible and clickable above the sheet`, await page.evaluate(() => {
      const box = document.getElementById('estop').getBoundingClientRect();
      return box.bottom > 0 && document.elementFromPoint(box.left + box.width / 2, box.top + box.height / 2)?.closest('#estop') !== null;
    }));
    check(`phone ${scheme}: sheet has a visible close button and a drag handle`, await page.locator('#ab-pop .ab-pop-close').isVisible() && await page.locator('#ab-pop .ab-pop-grip').isVisible());
    for (let index = 0; index < 8; index += 1) await page.keyboard.press('Tab');
    check(`phone ${scheme}: focus is trapped in the sheet`, await focusedIn(page, '#ab-pop'));
    check(`phone ${scheme}: no overflow with the sheet open`, await noOverflow(page));
    await shot(page, `phone-${scheme}-sheet`);
    await page.keyboard.press('Escape');
    await page.waitForFunction(() => document.getElementById('ab-pop').hidden);
    check(`phone ${scheme}: Esc closes the sheet and focus returns to the chip`, await page.evaluate(() => document.activeElement?.dataset.field === 'model'));
    await chip(page, 'model').click();
    await page.waitForSelector('#ab-pop[data-state=open] [role=option]');
    await option(page, 'deepseek-flash').click();
    await waitStatus(page, '已应用 模型 deepseek-flash');
    check(`phone ${scheme}: tapping a row applies it`, (await chipValue(page, 'model')) === 'deepseek-flash');
    await chip(page, 'backend').click();
    await option(page, 'OpenCode').click();
    await page.waitForFunction(() => document.querySelector('.ab-tray-check').textContent.includes('组合可用'));
    check(`phone ${scheme}: tray buttons stack with the primary on top`, await page.evaluate(() => {
      const primary = document.querySelector('.ab-tray .ab-btn--primary').getBoundingClientRect();
      const other = document.querySelector('.ab-tray-actions .ab-btn:not(.ab-btn--primary)').getBoundingClientRect();
      return primary.top < other.top && primary.height >= 44;
    }));
    check(`phone ${scheme}: no overflow with the tray open`, await noOverflow(page));
    await scrollToBar(page);
    await shot(page, `phone-${scheme}-tray`);
    await chip(page, 'settings').click();
    await page.waitForSelector('#ab-pop[data-state=open] .ab-profiles');
    await page.locator('#ab-pop .ab-btn', { hasText: '新建 profile' }).click();
    await page.waitForSelector('#ab-f-name');
    await page.waitForTimeout(250);
    check(`phone ${scheme}: profile form fits`, await noOverflow(page) && await page.locator('#ab-pop').evaluate((node) => node.scrollWidth <= node.clientWidth + 1));
    await shot(page, `phone-${scheme}-profile-form`);
    await page.locator('#ab-pop .ab-pop-close').click();
    check(`phone ${scheme}: no page errors`, seen.errors.length === 0);
    await page.close(); await context.close();
  }
  // ---------- external backend: read-only Agent bar ----------
  for (const [viewport, label] of [[{ width: 1440, height: 900 }, 'desktop'], [{ width: 390, height: 844 }, 'phone']]) {
    await knobs({ reset: true });
    await knobs({ external: { endpoint: 'http://127.0.0.1:8790/mcp', client: null, last_call_at: null } });
    const context = await browser.newContext({ viewport, colorScheme: 'light', reducedMotion: 'no-preference' });
    const page = await context.newPage();
    page.setDefaultTimeout(15000);
    const errors = [], controls = [];
    page.on('pageerror', (error) => errors.push(error.message));
    page.on('request', (request) => { if (request.method() === 'POST' && request.url().endsWith('/v1/control')) controls.push(request.postData() || ''); });
    await page.goto(`${url}/console`);
    await page.waitForSelector('.ab-ext:not([hidden])');
    const text = () => page.locator('.ab-ext').innerText();
    check(`external ${label}: waiting state with the MCP endpoint as plain text`, (await text()).includes('等待 initialize') && !(await text()).includes('已连接') && (await page.locator('.ab-ext-endpoint').textContent()) === 'http://127.0.0.1:8790/mcp');
    check(`external ${label}: no backend / profile / model / effort / tier chips and no handoff`, await page.locator('.ab-chips').isHidden() && await page.locator('.ab-status').isHidden()
      && await page.locator('.ab-ext button').count() === 1 && await page.locator('.ab-tray[data-open=true]').count() === 0);
    check(`external ${label}: prompt form hidden, one sentence says the task is given in the user's own agent`, await page.locator('#prompt-form').isHidden() && (await page.locator('#external-task-note').isVisible()));
    check(`external ${label}: safety controls stay`, await page.locator('#estop').isVisible() && await page.locator('#hold').count() === 1 && await page.locator('#go-home').count() === 1 && await page.locator('#relax').count() === 1);
    check(`external ${label}: no overflow`, await noOverflow(page));
    await shot(page, `external-${label}-waiting`);
    const before = await page.locator('.ab-ext').evaluate((node) => { node.dataset.mark = 'kept'; return node.dataset.key; });
    await knobs({ external: { endpoint: 'http://127.0.0.1:8790/mcp', client: { name: 'claude-code', version: '2.1.3', self_reported: true }, last_call_at: '2026-11-15T08:30:00Z' } });
    await page.waitForFunction(() => document.querySelector('.ab-ext-value')?.textContent.includes('claude-code 2.1.3'));
    check(`external ${label}: self-reported client and last tool call shown, never "connected"`, (await text()).includes('自报') && (await text()).includes('claude-code 2.1.3') && (await text()).includes('最近工具调用') && !(await text()).includes('尚无工具调用') && !(await text()).includes('已连接'));
    check(`external ${label}: same card node is updated in place`, await page.locator('.ab-ext').evaluate((node) => node.dataset.mark === 'kept') && before !== await page.locator('.ab-ext').getAttribute('data-key'));
    await page.waitForTimeout(3500);
    check(`external ${label}: an unchanged payload does not rebuild the card`, await page.locator('.ab-ext').evaluate((node) => node.dataset.mark === 'kept'));
    check(`external ${label}: only lease claims reach /v1/control (no prompt, steer, settings or handoff), no page errors`, controls.every((body) => body.includes('"command":"claim_input"')) && errors.length === 0);
    await shot(page, `external-${label}-connected`);
    await knobs({ external: { endpoint: 'http://127.0.0.1:8790/mcp', client: { name: 'ni-rua-plugin-doctor', version: '0.1.0', self_reported: true }, last_call_at: null } });
    await page.waitForFunction(() => document.querySelector('.ab-ext')?.innerText.includes('doctor 探测'));
    check(`external ${label}: doctor is labelled a probe, with no tool call yet`, (await text()).includes('不是 Agent') && (await text()).includes('尚无工具调用') && !(await text()).includes('已连接'));
    await knobs({ external: { endpoint: null, client: null, last_call_at: null, replay: true } });
    await page.waitForFunction(() => document.querySelector('.ab-ext')?.innerText.includes('已结束的运行（回放）'));
    check(`external ${label}: replay of an archived run is not waiting and not "MCP not started"`, !(await text()).includes('等待') && !(await text()).includes('尚未启动') && (await text()).includes('回放') && await noOverflow(page));
    await shot(page, `external-${label}-replay`);
    await page.close(); await context.close();
  }
  console.log(`PASS: ${checks.length} checks; screenshots in ${output}`);
} finally {
  await knobs({ reset: true }).catch(() => {});
  await browser.close();
}
