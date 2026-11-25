// Browser check of the /calibration wizard and the /console calibration badge against the synthetic calibration
// fixture (web/fixture-calibration-server.py: the real gateway over a fake SO-101 worker; no hardware).
//
//   mamba run -n ni-rua python web/fixture-calibration-server.py --port 8791 &
//   NI_RUA_URL=http://127.0.0.1:8791 PLAYWRIGHT_MODULE=/path/to/playwright/index.mjs \
//   NI_RUA_BROWSER_EXECUTABLE="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
//   NI_RUA_SCREENSHOT_DIR=/tmp/calibration-check node web/check-calibration.mjs
//
// Console messages about the Codex terminal WebSocket are ignored (the fixture has no PTY); HTTP >= 400 responses (the
// fixture has no observation, and a page that reloads may meet its own previous writer lease) are listed, not failed.
// The script resets the fixture first, so it can be rerun against the same process. It never uses the real gateway.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
const engines = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const url = process.env.NI_RUA_URL || 'http://127.0.0.1:8791';
assert.ok(!new URL(url).port || new URL(url).port !== '8781', 'refusing to run against port 8781 (real gateway)');
const output = process.env.NI_RUA_SCREENSHOT_DIR || 'runs/web-calibration-check';
await fs.mkdir(output, { recursive: true });

const browser = await engines[process.env.NI_RUA_BROWSER || 'chromium'].launch({ headless: true,
  ...(process.env.NI_RUA_BROWSER_EXECUTABLE ? { executablePath: process.env.NI_RUA_BROWSER_EXECUTABLE } : {}) });
const errors = [], checks = [], failures = [];
const check = (name, ok = true) => { assert.ok(ok, name); checks.push(name); };
const fixture = async (endpoint, body) => {
  const response = await fetch(new URL(endpoint, url), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body ?? {}) });
  assert.equal(response.status, 200, endpoint);
};
const scenario = (body) => fixture('/fixture/calibration', body);

const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
context.on('page', (page) => {
  page.on('pageerror', (error) => errors.push(`pageerror ${error.message}`));
  page.on('console', (message) => { if (message.type() === 'error' && !message.text().startsWith('Failed to load resource') && !message.text().includes('/v1/terminal')) errors.push(`console ${message.text()}`); });
  page.on('response', (response) => { if (response.status() >= 400) failures.push(`${response.status()} ${response.request().method()} ${new URL(response.url()).pathname} ${(response.request().postData() || "").slice(0, 90)}`); });
});
const page = await context.newPage(); page.setDefaultTimeout(15000);
const shot = (name, target = page) => target.screenshot({ path: path.join(output, `${name}.png`), fullPage: true });
const content = () => page.locator('#content');
const status = (label) => page.locator('#content .head-row .pill').filter({ hasText: label });

try {
  // The run's explicit board must survive begin clearing the form draft and a page reload.
  // Exercise both sides of the former 3 mm default coincidence.
  for (const [defaultMm, explicitMm] of [[3.2, 3], [3, 3.2]]) {
    await fixture('/fixture/reset');
    await scenario({ default_board_thickness_m: defaultMm / 1000 });
    await page.goto(`${url}/calibration`);
    await page.waitForSelector('.form-grid input');
    check(`new run form uses ${defaultMm} mm configured default`, Number(await page.locator('.form-grid input').nth(4).inputValue()) === defaultMm);
    await page.locator('.form-grid input').nth(4).fill(String(explicitMm));
    await page.locator('.change-list label', { hasText: '腕部相机被碰' }).locator('input').check();
    await page.getByRole('button', { name: '开始重新标定' }).click();
    await page.waitForSelector('#stepper button.active');
    check(`S2 keeps explicit ${explicitMm} mm after begin`, Number(await page.locator('.form-grid input').nth(4).inputValue()) === explicitMm);
    await page.reload();
    await page.waitForSelector('#stepper button.active');
    check(`S2 keeps explicit ${explicitMm} mm after refresh`, Number(await page.locator('.form-grid input').nth(4).inputValue()) === explicitMm);
    for (const box of await page.locator('#content .checks input[type=checkbox]').all()) await box.check();
    const request = page.waitForRequest(r => r.url().endsWith('/v1/calibration/steps/S2/start') && r.method() === 'POST');
    await page.getByRole('button', { name: '开始', exact: true }).click();
    check(`S2 submits explicit ${explicitMm} mm`, (await request).postDataJSON().params.board.thickness_m === explicitMm / 1000);
    await status('通过').first().waitFor();
  }

  // ---------- live progress through a lapsed writer lease (regression: real-arm S2, 2026-11-15) ----------
  // A real S2 runs ~74 s and its POST holds the gateway's control lock. The 30 s writer lease lapsed mid-step, the
  // page's re-claim queued behind the step and its poll loop waited on it, so progress and the preview (redrawn by each
  // state poll) froze about halfway while the arm kept moving. Both halves of the fix are checked against the real
  // gateway: it answers the renewal the page now sends during the step; and with the holder's claims stalled until the
  // step ends (stall_claims_during_step, as the gateway did before), a lapsed lease (/fixture/expire-lease) must not
  // stop the page's poll. A /console window meanwhile is refused (input_owner_busy_step) and must not claim again
  // while the step holds control.
  await fixture('/fixture/reset');
  await scenario({ step_seconds: 20, stall_claims_during_step: true });
  await page.goto(`${url}/calibration`);
  await page.locator('.change-list label', { hasText: '腕部相机被碰' }).locator('input').check();
  await page.getByRole('button', { name: '开始重新标定' }).click();
  await page.waitForSelector('#stepper button.active');
  for (const box of await page.locator('#content .checks input[type=checkbox]').all()) await box.check();
  const poseIndex = () => Number(/位姿 (\d+)\/21/.exec(document.querySelector('#content progress + p.meta')?.textContent || '')?.[1] ?? -1);
  const previewDigest = () => {
    const image = document.querySelector('[data-preview="wrist"] img');
    if (!image?.complete || !image.naturalWidth) return null;
    const canvas = Object.assign(document.createElement('canvas'), { width: image.naturalWidth, height: image.naturalHeight });
    const context = canvas.getContext('2d');
    context.drawImage(image, 0, 0);
    return context.getImageData(0, 0, canvas.width, 24).data.reduce((sum, value) => (sum * 31 + value) >>> 0, 7); // label row
  };
  const controlCommand = (request) => request.method() === 'POST' && request.url().endsWith('/v1/control') ? request.postDataJSON()?.command : undefined;
  const runningStep = async () => (await (await fetch(new URL('/v1/calibration/state', url))).json()).running_step;
  let stepStarted = false;
  const renewal = page.waitForResponse((response) => stepStarted && controlCommand(response.request()) === 'renew_input', { timeout: 15000 });
  stepStarted = true;
  await page.getByRole('button', { name: '开始', exact: true }).click();
  const renewed = await renewal;
  check('the page renews its writer lease while S2 runs and the gateway answers at once',
    renewed.status() === 200 && (await renewed.json()).status === 'owned' && await runningStep() === 'S2');
  const poseAtLapse = await page.evaluate(poseIndex);
  const previewAtLapse = await page.evaluate(previewDigest);
  await fixture('/fixture/expire-lease');
  const consolePage = await context.newPage();
  const consoleClaims = [], consoleRefusals = [];
  consolePage.on('request', (request) => { if (controlCommand(request) === 'claim_input') consoleClaims.push(request); });
  consolePage.on('response', async (response) => {
    if (controlCommand(response.request()) === 'claim_input') consoleRefusals.push(`${response.status()} ${(await response.json().catch(() => ({}))).error || ''}`);
  });
  await consolePage.goto(`${url}/console`);
  await Promise.all([
    page.waitForFunction(([read, from]) => new Function(`return (${read})()`)() >= from + 6, [poseIndex.toString(), poseAtLapse], { timeout: 7000 }),
    consolePage.waitForTimeout(4000)]);
  check('S2 progress keeps moving while the re-claim of the lapsed lease is stalled', true);
  check('/console is refused once (input_owner_busy_step) and does not claim again during the step',
    consoleClaims.length === 1 && consoleRefusals.length === 1 && consoleRefusals[0].startsWith('409 input_owner_busy_step') && await runningStep() === 'S2');
  await consolePage.close();
  await page.waitForFunction(([read, from]) => { const now = new Function(`return (${read})()`)(); return now !== null && now !== from; },
    [previewDigest.toString(), previewAtLapse], { timeout: 5000 });
  check('S2 live preview keeps refreshing after the writer lease lapsed mid-step', true);
  await status('通过').first().waitFor();
  await page.waitForFunction(() => document.querySelector('#lease-state')?.textContent === '本窗口持有控制');
  check('the page holds the writer lease again after the step', true);
  await shot('00-S2-live-through-lease-lapse');

  // A state poll that gets no answer within the page's timeout shows "connection lost, retrying" and keeps the step view.
  await scenario({ stall_claims_during_step: false, state_delay_s: 12 });
  await page.locator('#banners .banner.bad', { hasText: '与网关的连接中断' }).waitFor({ timeout: 12000 });
  check('a stalled state poll shows "connection lost, retrying" and keeps the last step view',
    (await page.locator('#banners').textContent()).includes('正在重试') && await page.locator('#content .head-row h2', { hasText: 'S2' }).count() === 1);
  await shot('00b-connection-lost');
  await scenario({ state_delay_s: 0 });
  await page.locator('#banners .banner', { hasText: '与网关的连接中断' }).waitFor({ state: 'detached', timeout: 12000 });
  check('the banner clears once a poll is answered again', true);

  await fixture('/fixture/reset');
  // ---------- start screen ----------
  await page.goto(`${url}/calibration`);
  await page.waitForSelector('#content .cards .card');
  check('start screen shows exactly two mode cards', await page.locator('#content .cards > .card').count() === 2);
  check('change checklist comes from the server (6 entries)', await page.locator('.change-list input[type=checkbox]').count() === 6);
  check('current calibration card is valid', await page.locator('#start-status').getByText(/标定有效/).count() > 0);
  check('version history lists the imported version', (await content().textContent()).includes('imported_existing_file'));
  check('recalibrate disabled until a change is ticked', await page.getByRole('button', { name: '开始重新标定' }).isDisabled());
  await page.locator('.change-list label', { hasText: '固定相机被移动' }).locator('input').check();
  check('plan preview follows the ticked change', (await content().textContent()).includes('会重做：S4 → S5 → S6'));
  await page.locator('.change-list label', { hasText: '腕部相机被碰' }).locator('input').check();
  check('plan preview is the union of changes', (await content().textContent()).includes('会重做：S2 → S4 → S5 → S6'));
  await shot('01-start');

  // ---------- recalibrate mode: stepper marks the steps that are not needed ----------
  await page.getByRole('button', { name: '开始重新标定' }).click();
  await page.waitForSelector('#stepper button.skipped');
  check('recalibrate plan skips S0/S1/S3', await page.locator('#stepper button.skipped').count() === 3);
  check('stepper opens on the first planned step (S2)', (await page.locator('#stepper button.active b').textContent()) === 'S2');
  check('board form shown on S2', await page.locator('.form-grid input').count() === 6);
  await shot('02-recalibrate-S2');

  // ---------- first run ----------
  await page.locator('#home-link').click();
  await page.getByRole('button', { name: '开始首次完整标定' }).click();
  await page.waitForFunction(() => document.querySelectorAll('#stepper li').length === 7 && !document.querySelector('#stepper button.skipped'));
  check('stepper has 7 steps and none skipped', await page.locator('#stepper li').count() === 7 && await page.locator('#stepper button.skipped').count() === 0);
  await page.locator('#stepper button', { hasText: 'S0' }).click();
  const text = await content().textContent();
  for (const needle of ['这一步做什么', '你需要做', '通过标准', '准备检查', '关节读数延迟：≤', '机械臂不动']) check(`S0 view shows ${needle}`, text.includes(needle));
  await shot('03-S0-before');
  await page.getByRole('button', { name: '开始', exact: true }).click();
  await status('通过').first().waitFor();
  await page.waitForFunction(() => document.querySelectorAll('.result td.pass').length >= 5);
  check('S0 result table has passing rows', await page.locator('.result td.pass').count() >= 5);
  check('stepper marks S0 passed', (await page.locator('#stepper button', { hasText: 'S0' }).getAttribute('class')).includes('passed'));
  await shot('04-S0-passed');

  // S1 fails first (failure rendering), then passes on retry
  await scenario({ fail_steps: ['S1'] });
  await page.locator('#stepper button', { hasText: 'S1' }).click();
  check('motion step is labelled', (await content().textContent()).includes('机械臂会动'));
  check('start disabled until operator checks are ticked', await page.getByRole('button', { name: '开始', exact: true }).isDisabled());
  for (const box of await page.locator('#content .checks input[type=checkbox]').all()) await box.check();
  await page.getByRole('button', { name: '开始', exact: true }).click();
  await page.waitForSelector('.progress-live, progress');
  await status('失败').first().waitFor();
  check('failed step shows a red result row and a reason', await page.locator('.result td.fail').count() > 0 && await page.locator('.result .fail').count() > 1);
  check('failed step offers a re-run', await page.getByRole('button', { name: '重新执行' }).count() === 1);
  check('touch table rendered', await page.locator('.result table').count() >= 2);
  check('banner reports failure', (await page.locator('#banners').textContent()).includes('这一步未通过'));
  await shot('05-S1-failed');
  await scenario({ fail_steps: [] });
  await page.getByRole('button', { name: '重新执行' }).click();
  await status('通过').first().waitFor();

  // S2 with board form
  await page.locator('#stepper button', { hasText: 'S2' }).click();
  await page.locator('.form-grid input').first().waitFor();
  check('board form is filled from server data (mm)', (await page.locator('.form-grid input').nth(2).inputValue()) === '30');
  await page.locator('.form-grid input').nth(4).fill('3');
  for (const box of await page.locator('#content .checks input[type=checkbox]').all()) await box.check();
  await page.getByRole('button', { name: '开始', exact: true }).click();
  await page.waitForSelector('#content progress');
  await shot('06-S2-running');
  await status('通过').first().waitFor();
  check('S2 live preview image slots exist', await page.locator('[data-preview]').count() === 2);

  // S3 hand-held capture
  await page.locator('#stepper button', { hasText: 'S3' }).click();
  await page.getByRole('button', { name: '开始采集' }).click();
  await page.waitForFunction(() => document.querySelector('.hh-grid td.done'));
  check('S3 coverage grid and guidance render', await page.locator('.hh-grid td').count() === 9 && await page.locator('.guidance li').count() > 0);
  await page.waitForFunction(() => [...document.querySelectorAll('button')].some((b) => b.textContent === '求解' && !b.disabled));
  await shot('07-S3-capture');
  await page.getByRole('button', { name: '求解', exact: true }).click();
  await status('通过').first().waitFor();

  // S4, S5
  for (const id of ['S4']) {
    await page.locator('#stepper button', { hasText: id }).click();
    for (const box of await page.locator('#content .checks input[type=checkbox]').all()) await box.check();
    await page.getByRole('button', { name: '开始', exact: true }).click();
    await status('通过').first().waitFor();
  }
  await page.locator('#stepper button', { hasText: 'S5' }).click();
  check('S5 requires post-capture contact evidence review', await page.getByRole('button', { name: '开始', exact: true }).isDisabled());
  for (const box of await page.locator('#content .checks input[type=checkbox]').all()) await box.check();
  await page.getByRole('button', { name: '开始', exact: true }).click();
  await status('通过').first().waitFor();
  check('S5 result lists the joint-solve criteria', (await content().textContent()).includes('最大触碰残差'));

  // stale: re-running S2 invalidates S4 and S5
  await page.locator('#stepper button', { hasText: 'S2' }).click();
  for (const box of await page.locator('#content .checks input[type=checkbox]').all()) await box.check();
  await page.getByRole('button', { name: '重新执行' }).click();
  await page.waitForFunction(() => document.querySelectorAll('#stepper button.stale').length === 2);
  check('S4 and S5 become stale', await page.locator('#stepper button.stale').count() === 2);
  await page.locator('#stepper button', { hasText: 'S5' }).click();
  check('stale step explains it must be redone', (await content().textContent()).includes('上游步骤已重新做过'));
  await shot('08-S5-stale');
  for (const id of ['S4', 'S5']) {
    await page.locator('#stepper button', { hasText: id }).click();
    for (const box of await page.locator('#content .checks input[type=checkbox]').all()) await box.check();
    await page.getByRole('button', { name: '重新执行' }).click();
    await status('通过').first().waitFor();
  }

  // ---------- S6 review / apply ----------
  await page.locator('#stepper button', { hasText: 'S6' }).click();
  await page.getByRole('heading', { name: '复核与应用' }).waitFor();
  check('review shows the old/new diff table', await page.locator('.card:has(h2:text("复核与应用")) tbody tr').count() >= 8);
  check('lengths in m are shown in mm', (await content().textContent()).includes('-17.90 mm'));
  check('apply disabled until the numbers are acknowledged', await page.getByRole('button', { name: '应用新标定' }).isDisabled());
  await shot('09-S6-review');
  await page.getByText('我已核对新旧数值').click();
  await page.getByRole('button', { name: '应用新标定' }).click();
  await page.getByText('已应用：版本').waitFor();
  check('rollback to previous version offered after apply', await page.getByRole('button', { name: '回滚到上一个版本' }).count() === 1);
  for (const box of await page.locator('#content .checks input[type=checkbox]').all()) await box.check();
  await page.getByRole('button', { name: '开始验证触碰' }).click();
  await status('通过').first().waitFor();
  await shot('10-S6-applied-verified');

  // Independent XY annotation uses frozen images and leaves the height verdict intact.
  const xy = page.locator('.card').filter({ has: page.getByRole('heading', { name: 'XY 相机—指尖一致性检查' }) });
  await xy.getByRole('button', { name: '冻结当前画面（不移动）' }).click();
  await page.waitForFunction(() => [...document.querySelectorAll('img[alt$="冻结画面"]')].every(im => im.naturalWidth > 0));
  check('XY evaluation waits for both annotations', await xy.getByRole('button', { name: '检查本点 XY' }).isDisabled());
  for (const image of await xy.locator('img').all()) await image.click({ position: { x: 120, y: 100 } });
  await xy.getByRole('button', { name: '检查本点 XY' }).click();
  await xy.getByText(/本点超差.*12.00 mm/).waitFor();
  check('XY mismatch is visible without replacing the S6 height pass', await status('通过').count() > 0);
  await shot('10b-S6-xy');

  // Failed re-verification invalidates the active version's badge and home card.
  await scenario({ fail_steps: ['S6'] });
  await page.getByRole('button', { name: '重新执行', exact: true }).click();
  await status('失败').first().waitFor();
  await page.waitForFunction(() => document.querySelector('#calib-badge')?.textContent.includes('高度验证未通过'));
  check('failed active S6 is not green in the header', await page.locator('#calib-badge.bad').count() === 1);
  check('S6 failure explains the new calibration remains applied', (await page.locator('.result').textContent()).includes('没有自动回滚'));
  await scenario({ fail_steps: [] });

  // ---------- rollback list ----------
  await page.locator('#home-link').click();
  await page.waitForSelector('#start-history table');
  check('home reports failed active height verification', (await page.locator('#start-status').textContent()).includes('高度验证未通过'));
  check('home reports verification error with units', (await page.locator('#start-status').textContent()).includes('11.00 mm'));
  await shot('10c-S6-failed-home');
  check('history lists the applied version as current plus the imported one', await page.locator('#start-history tbody tr').count() === 2);
  check('active version cannot be rolled back to', await page.locator('#start-history tbody tr').first().locator('button').isDisabled());
  page.once('dialog', (dialog) => dialog.accept());
  await page.locator('#start-history tbody tr').nth(1).getByRole('button', { name: '回滚到此版本' }).click();
  await page.waitForFunction(() => document.querySelector('#start-history tbody tr:nth-child(2) td')?.textContent.includes('（当前）'));
  check('rollback makes the imported version current', true);
  check('rollback refreshes header and home from the same state', (await page.locator('#calib-badge').textContent()).includes('标定有效'));
  await shot('11-rollback-done');

  // ---------- states: stale badge, needed badge, readiness red, unknown receipt, lease ----------
  await scenario({ staleness: 'shifted' });
  await page.reload(); await page.waitForSelector('#calib-badge.warn');
  check('shifted staleness shows the "可能已过期" badge', (await page.locator('#calib-badge').textContent()).includes('可能已过期'));
  await scenario({ staleness: 'ok', bundle_present: false });
  await page.reload(); await page.waitForSelector('#calib-badge.bad');
  await page.locator('#home-link').click(); await page.waitForSelector('#start-status');
  check('missing calibration shows the "需要标定" badge and card', (await page.locator('#calib-badge').textContent()).includes('需要标定') && await page.getByText('需要标定').count() >= 2);
  await shot('12-needed');
  await scenario({ bundle_present: true, bad_readiness: ['torque_on', 'telemetry_fresh'] });
  await page.locator('#home-link').click();
  await page.getByRole('button', { name: '开始首次完整标定' }).click();
  // The new run opens on S0 once a refresh sees it; select S1 only after that.
  await page.waitForFunction(() => document.querySelector('#stepper button.active b')?.textContent === 'S0'
    && [...document.querySelectorAll('#stepper em')].every((label) => label.textContent === '未开始'));
  await page.locator('#stepper button', { hasText: 'S1' }).click();
  await page.waitForSelector('.tick.bad');
  check('red automatic readiness items disable start and say why', await page.locator('.tick.bad').count() === 2 && (await content().textContent()).includes('准备检查还有红色项'));
  await shot('13-readiness-red');
  await scenario({ bad_readiness: [], unknown_step: 'S0' });
  await page.locator('#stepper button', { hasText: 'S0' }).click();
  await page.getByRole('button', { name: '开始', exact: true }).click();
  await page.waitForFunction(() => document.querySelector('#banners .banner.bad')?.textContent.includes('UNKNOWN'));
  check('unconfirmed receipt is shown as UNKNOWN, not as success', true);
  await shot('14-unknown');

  // ---------- narrow layout ----------
  await scenario({ unknown_step: null });
  await page.setViewportSize({ width: 375, height: 800 });
  await page.locator('#home-link').click();
  await page.waitForSelector('#content .cards');
  const overflow = async () => {
    const { extra, wide } = await page.evaluate(() => {
      const width = document.documentElement.clientWidth;
      return { extra: document.documentElement.scrollWidth - width,
        wide: [...document.querySelectorAll('body *')].filter((node) => node.getBoundingClientRect().right > width + 1 && !node.closest('.table-wrap'))
          .slice(0, 5).map((node) => `${node.tagName}.${node.className}`) };
    });
    if (extra > 1) console.error('horizontal overflow from:', wide.join(', '));
    return extra;
  };
  check('start screen has no horizontal page scroll at 375 px', await overflow() <= 1);
  await shot('15-narrow-start');
  await page.locator('#stepper button', { hasText: 'S6' }).click();
  check('step screen has no horizontal page scroll at 375 px', await overflow() <= 1);
  await shot('16-narrow-step');
  await page.locator('#stepper button', { hasText: 'S2' }).click();
  check('board form has no horizontal page scroll at 375 px', await overflow() <= 1);

  // ---------- console badge ----------
  await page.setViewportSize({ width: 1280, height: 900 });
  await scenario({ bundle_present: true, staleness: 'ok' });
  await page.goto(`${url}/console`);
  await page.waitForSelector('#calib-badge.ok');
  check('console badge shows the calibration state and links to the wizard', (await page.locator('#calib-badge').textContent()).includes('标定有效') && (await page.locator('#calib-badge').getAttribute('href')) === '/calibration');
  for (const label of ['重新标定相机', '重载标定文件', '手持标定固定相机']) check(`console no longer has the loose "${label}" button`, await page.getByRole('button', { name: label }).count() === 0);
  await shot('17-console');

  console.log('HTTP >= 400 responses:', JSON.stringify([...new Set(failures)]));
  if (errors.length) console.error([...new Set(errors)].join('\n'));
  check('no JavaScript errors on any page', errors.length === 0);
} finally {
  await fs.writeFile(path.join(output, 'calibration-report.json'), JSON.stringify({ checked_at: new Date().toISOString(), url, checks, errors }, null, 2));
  await shot('99-final').catch(() => {});
  await browser.close();
}
console.log(`${checks.length} checks passed; screenshots in ${output}`);
if (errors.length) { console.error(errors.join('\n')); process.exitCode = 1; }
