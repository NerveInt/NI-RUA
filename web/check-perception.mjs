// Synthetic browser regression: no robot, service calls or writes to a gateway.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assets = process.env.NI_RUA_PREVIEW_ASSETS;
assert.ok(assets, 'Build assets and set NI_RUA_PREVIEW_ASSETS');
const output = process.env.NI_RUA_SCREENSHOT_DIR || '/tmp/ni-rua-perception-browser';
await fs.mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true, ...(process.env.NI_RUA_BROWSER_EXECUTABLE ? { executablePath: process.env.NI_RUA_BROWSER_EXECUTABLE } : {}) });
const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
const errors = [], writes = [];
page.on('pageerror', (error) => errors.push(error.message));
const observation = (id) => ({ observation_id: id, env_generation: 1, sim_step: 0, cameras: ['fixed', 'wrist'].map((camera_id) => ({ camera_id, frame_id: id + camera_id, rgb_ref: `artifact:${id}-${camera_id}` })) });
const perception = (id) => ({ observation_id: id, camera_id: 'fixed', overlay_ref: `artifact:${id}-overlay`, backend: 'sam31',
  objects: [{ id: 'o1', label: '<img src=x onerror=window.injected=true>', bbox_px: [100, 80, 240, 220], contour_px: [[100, 80], [240, 100], [220, 220]], centre_m: [.1, .2, .03], length_m: .04, width_m: .03, height_m: .02, yaw_deg: 20, axis_tilt_deg: null, quality: 'ok', support: 'table' }],
  table: { columns: ['id'], rows: [['o1']], table_z: 0 }, scene3d: { points: 3000 }, diagnostics: { object_table: { status: 'conditional' }, scene3d: { status: 'conditional' } },
  delta: { moved: [{ before_id: 'o2', after_id: 'o1', delta_m: [.01, -.002, .003] }], unchanged_count: 0 } });
let mode = 'replay', delayed = null, delayedSeen, lateReady = false, failOverlay = false;
const prefix = '/runs/fixture/v1/';
await page.route('**/*', async (route) => {
  const request = route.request(), url = new URL(request.url()), endpoint = url.pathname.split('/v1/')[1];
  if (!['GET', 'HEAD'].includes(request.method())) { writes.push(request.url()); return route.abort(); }
  const json = (value, status = 200) => route.fulfill({ status, json: value });
  if (!endpoint) {
    const name = url.pathname.endsWith('/') ? 'index.html' : path.basename(url.pathname);
    if (!['index.html', 'app.js', 'app.css', 'styles.css'].includes(name)) return json({}, 404);
    return route.fulfill({ path: path.join(assets, name) });
  }
  if (endpoint === 'run') return json({ run_id: 'fixture', status: mode, view_mode: mode, read_only: true, terminal_available: false, library: { live: mode === 'live' } });
  if (endpoint === 'remote/status') return json({ enabled: true, archive: mode === 'replay', host: '<img src=x onerror=window.injected=true>', services: { vggt: { available: true, health: { version: 'fixture-vggt', object_table_backend: 'sam31' } } } });
  if (endpoint === 'observations/latest') return json(observation('O3'));
  if (endpoint.startsWith('observations/')) {
    const id = endpoint.split('/')[1];
    if (!endpoint.endsWith('/perception')) return json(observation(id));
    if (id === 'O1' && delayed) { delayedSeen?.(); await delayed; }
    if (id === 'O3' && !lateReady) return json({ error: 'perception_not_found' }, 404);
    return json(perception(id));
  }
  if (endpoint.startsWith('artifacts/')) {
    if (failOverlay && endpoint.endsWith('overlay')) return json({}, 404);
    return route.fulfill({ contentType: 'image/svg+xml', body: '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480"><rect width="640" height="480" fill="#a4b9a8"/><rect x="100" y="80" width="140" height="140" fill="#d3913c"/><text x="20" y="40">Synthetic perception fixture</text></svg>' });
  }
  if (endpoint === 'events') return route.fulfill({ contentType: 'text/event-stream', body: ': synthetic\n\n' });
  const payloads = { observations: { observations: [] }, items: { items: [], cursor: 0 }, requests: { requests: [] }, 'scene/head': {}, 'scene/versions': { versions: [] } };
  return json(payloads[endpoint] || {}, payloads[endpoint] ? 200 : 404);
});
try {
  await page.goto('http://127.0.0.1:18792/runs/fixture/');
  await page.waitForFunction(() => typeof window.selectObservationById === 'function');
  await page.evaluate(() => window.selectObservationById('O2', 'fixed'));
  await page.waitForSelector('#perception-image:not([hidden])');
  await page.waitForSelector('#camera-svg-overlay polygon');
  assert.match(await page.locator('#perception-objects-body').innerText(), /20° \/ -/);
  assert.match(await page.locator('#perception-delta-banner').innerText(), /\+10, -2, \+3/);
  assert.equal(await page.evaluate(() => window.injected), undefined);
  const sizes = await page.evaluate(() => {
    const img = document.querySelector('#perception-image').getBoundingClientRect();
    const svg = document.querySelector('#camera-svg-overlay').getBoundingClientRect();
    return { x: Math.abs(img.x - svg.x), y: Math.abs(img.y - svg.y), w: Math.abs(img.width - svg.width), h: Math.abs(img.height - svg.height) };
  });
  assert.ok(Object.values(sizes).every((v) => v < 1), JSON.stringify(sizes));
  await page.screenshot({ path: path.join(output, 'perception-overlay.png'), fullPage: true });
  await page.selectOption('#camera-select', 'wrist');
  assert.equal(await page.locator('#perception-image').getAttribute('src'), null);
  assert.equal(await page.locator('#camera-svg-overlay').isVisible(), false);
  assert.match(await page.locator('#perception-image-note').innerText(), /当前相机/);
  await page.selectOption('#camera-select', 'fixed');
  await page.click('#view-mode-split');
  await page.waitForSelector('#perception-image-split:not([hidden])');
  assert.equal(await page.locator('#camera-image-split').isVisible(), true);
  await page.screenshot({ path: path.join(output, 'perception-split.png'), fullPage: true });
  let release;
  delayed = new Promise((resolve) => { release = resolve; });
  const requestSeen = new Promise((resolve) => { delayedSeen = resolve; });
  await page.evaluate(() => window.selectObservationById('O1', 'fixed'));
  await requestSeen;
  assert.equal(await page.locator('#perception-objects-body .interactive-row').count(), 0);
  assert.equal(await page.locator('#perception-image').getAttribute('src'), null);
  await page.evaluate(() => window.selectObservationById('O2', 'fixed'));
  await page.waitForFunction(() => document.querySelector('#perception-image').getAttribute('src')?.includes('O2-overlay'));
  release(); delayed = null;
  await page.waitForLoadState('networkidle');
  assert.match(await page.locator('#perception-image').getAttribute('src'), /O2-overlay/);
  await page.click('#perception-service-btn');
  await page.waitForSelector('#perception-modal[open]');
  assert.match(await page.locator('#vggt-details').innerText(), /历史快照/);
  assert.match(await page.locator('#sam-details').innerText(), /未独立探测/);
  assert.equal(await page.locator('#remote-conn-details img').count(), 0);
  assert.equal(await page.evaluate(() => window.injected), undefined);
  await page.click('#close-perception-modal');
  failOverlay = true;
  await page.evaluate(() => window.selectObservationById('O4', 'fixed'));
  await page.waitForSelector('#camera-image:not([hidden])');
  assert.equal(await page.locator('#perception-image').isVisible(), false);
  await page.waitForFunction(() => document.querySelector('#perception-image-note').textContent.includes('读取失败'));
  failOverlay = false; mode = 'live';
  await page.reload();
  await page.waitForFunction(() => document.querySelector('#frame-meta').textContent.includes('O3'));
  assert.equal(await page.locator('#perception-objects-body .interactive-row').count(), 0);
  lateReady = true;
  await page.waitForSelector('#perception-objects-body .interactive-row', { timeout: 8000 });
  assert.equal(await page.locator('#perception-objects-body .interactive-row').count(), 1);
  assert.deepEqual(errors, []); assert.deepEqual(writes, []);
  console.log('PASS: overlay, contour alignment, split, camera binding, stale response, late result, unknown values, failed image, archived health, escaped labels; no mutations');
} finally { await browser.close(); }
