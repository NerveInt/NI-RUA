// Read-only navigation checks against the SYNTHETIC calibration gateway.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const origin = process.env.NI_RUA_URL || 'http://127.0.0.1:18793';
assert.notEqual(new URL(origin).port, '8781', 'never run navigation tests against the robot');
const output = process.env.NI_RUA_SCREENSHOT_DIR || '/tmp/ni-rua-workspace-browser';
await fs.mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true, ...(process.env.NI_RUA_BROWSER_EXECUTABLE ? { executablePath: process.env.NI_RUA_BROWSER_EXECUTABLE } : {}) });
const page = await browser.newPage({ viewport: { width: 1440, height: 1050 }, colorScheme: 'light' });
const errors = [], controls = [], missingAssets = [];
page.on('pageerror', (error) => errors.push(error.message));
page.on('request', (request) => { if (request.url().endsWith('/v1/control') && request.method() === 'POST') controls.push(request.postDataJSON().command); });
page.on('response', (response) => { if (/\.(js|css)(\?|$)/.test(response.url()) && response.status() >= 400) missingAssets.push(response.url()); });
const observation = { observation_id: 'fixture-perception', env_generation: 1, step_domain: 'fixture', sim_step: 0,
  cameras: ['fixed', 'wrist'].map((camera_id) => ({ camera_id, frame_id: 'fixture-' + camera_id, rgb_ref: 'artifact:fixture-' + camera_id })) };
await page.route('**/v1/observations/latest', (route) => route.fulfill({ json: observation }));
await page.route('**/v1/observations/fixture-perception/perception', (route) => route.fulfill({ json: {
  observation_id: observation.observation_id, camera_id: 'fixed', overlay_ref: 'artifact:fixture-overlay',
  objects: [{ id: 'o1', label: '合成测试物体', bbox_px: [120, 100, 260, 240], centre_m: [.2, -.03, .02], length_m: .03, width_m: .02, height_m: .04, quality: 'ok' }],
  table: { columns: ['id'], rows: [['o1']], table_z: 0 }, scene3d: { points: 1000 },
  diagnostics: { object_table: { status: 'conditional' }, scene3d: { status: 'conditional' } },
} }));
await page.route('**/v1/artifacts/fixture-*', (route) => route.fulfill({ contentType: 'image/svg+xml', body: '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480"><rect width="640" height="480" fill="#b7c4bd"/><rect x="120" y="100" width="140" height="140" fill="#cf9342"/><text x="20" y="35">Synthetic saved observation</text></svg>' }));
await page.route('**/v1/remote/status', (route) => route.fulfill({ json: { enabled: true, services: { vggt: { available: true, health: { version: 'fixture', object_table_backend: 'sam31' } } } } }));
try {
  await page.goto(origin + '/');
  await page.waitForURL('**/console');
  await page.waitForSelector('#calib-badge.ok');
  assert.equal(await page.locator('.live .cameras figure').count(), 2, 'original live cameras preserved');
  assert.equal(await page.locator('#estop').count(), 1);
  assert.equal(await page.locator('#tui .xterm').count(), 1, 'one original console terminal');
  assert.equal(await page.locator('#tui .xterm-helper-textarea').evaluate((node) => getComputedStyle(node).opacity), '0', 'bundled xterm CSS survives handwritten console styles');
  assert.equal(await page.locator('#perception-panel').count(), 1, 'one shared perception view');
  await page.waitForSelector('#perception-objects-body .interactive-row');
  assert.equal(await page.locator('#camera-select').inputValue(), 'fixed');
  await page.click('#view-mode-overlay');
  await page.waitForSelector('#perception-image:not([hidden])');
  assert.equal(await page.locator('#camera-svg-overlay').isVisible(), true);
  await page.evaluate(() => scrollTo(0, 0));
  await page.screenshot({ path: path.join(output, 'console-desktop.png'), fullPage: true });
  const nav = page.getByRole('navigation', { name: '工作区导航' });
  await nav.getByRole('link', { name: '标定', exact: true }).click();
  await page.waitForURL('**/calibration');
  await page.click('#home-link');
  await page.waitForSelector('#start-status');
  assert.equal(await nav.locator('[aria-current=page]').innerText(), '标定');
  await page.screenshot({ path: path.join(output, 'calibration-desktop.png'), fullPage: true });
  await nav.getByRole('link', { name: '轨迹与记录', exact: true }).click();
  await page.waitForURL('**/index.html');
  await page.waitForSelector('#workspace-nav:not([hidden])');
  assert.equal(await page.locator('#phase-video').count(), 1, 'original review remains available');
  await page.waitForFunction(() => document.querySelector('#run-status').textContent.includes('running'));
  assert.ok(page.url().endsWith('/index.html'), 'explicit review does not redirect');
  await nav.getByRole('link', { name: '真机工作台', exact: true }).click();
  await page.waitForURL('**/console');
  await page.setViewportSize({ width: 375, height: 900 });
  await page.waitForSelector('#calib-badge');
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'console fits mobile viewport');
  await page.screenshot({ path: path.join(output, 'console-mobile.png'), fullPage: true });
  await nav.getByRole('link', { name: '标定', exact: true }).click();
  await page.click('#home-link');
  await page.waitForSelector('#start-status');
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'calibration fits mobile viewport');
  await page.screenshot({ path: path.join(output, 'calibration-mobile.png'), fullPage: true });
  // Compatibility with an already-running gateway serving the old root route:
  // intercept only its HTML and let current assets + run identity choose console.
  const reviewHTML = await (await page.request.get(origin + '/index.html')).text();
  await page.route(origin + '/', (route) => route.fulfill({ contentType: 'text/html', body: reviewHTML }));
  await page.goto(origin + '/');
  await page.waitForURL('**/console');
  assert.deepEqual(missingAssets, []);
  assert.deepEqual(errors, []);
  assert.deepEqual(controls.filter((command) => !['claim_input', 'renew_input', 'release_input'].includes(command)), [], 'navigation starts no motion/calibration/inference');
  console.log('PASS: SO101 home, existing-gateway fallback, three-way navigation, original console controls/terminal, shared perception, explicit review, desktop/mobile, CSS assets; no physical commands');
} finally { await browser.close(); }
