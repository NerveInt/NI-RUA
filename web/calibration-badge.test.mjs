import assert from 'node:assert/strict';
import test from 'node:test';
import { badgeView, relativeTime, summaryFromState, refreshBadge } from './calibration-badge.js';

const now = Date.parse('2026-11-09T12:00:00Z');

test('active S6 verdict overrides file presence and never uses another version', () => {
  const bundle = { present: true, active_version: 'active', versions: [
    { name: 'other', verification: { passed: true } },
    { name: 'active', verification: { passed: false, max_error_m: .005045 } },
  ] };
  const summary = summaryFromState({ bundle });
  assert.equal(summary.status, 'verification_failed');
  assert.equal(badgeView(summary).kind, 'bad');
  assert.match(badgeView(summary).title, /5.045 mm/);
  assert.equal(summaryFromState({ bundle: { ...bundle, verification: null } }).status, 'unverified');
  assert.equal(summaryFromState({ bundle: { ...bundle, active_version: 'missing' } }).status, 'unverified');
  assert.equal(summaryFromState({ bundle: { ...bundle, active_version: 'other' } }).status, 'valid');
  assert.equal(summaryFromState({ bundle: { ...bundle, present: false } }).status, 'needed');
});

test('old gateway valid summary is checked against its read-only state', async (t) => {
  const paths = [];
  t.mock.method(globalThis, 'fetch', async (path) => {
    paths.push(path);
    return { ok: true, json: async () => path.endsWith('/summary') ? { status: 'valid' } :
      { bundle: { present: true, active_version: 'v1', versions: [{ name: 'v1', verification: { passed: false } }] } } };
  });
  const node = {};
  assert.equal((await refreshBadge(node)).status, 'verification_failed');
  assert.deepEqual(paths, ['/v1/calibration/summary', '/v1/calibration/state']);
  assert.equal(node.textContent, '高度验证未通过');
});

test('unavailable legacy verification never retains a green badge', async (t) => {
  t.mock.method(globalThis, 'fetch', async (path) => ({ ok: path.endsWith('/summary'), status: 503, json: async () => ({ status: 'valid' }) }));
  const node = {};
  assert.equal((await refreshBadge(node)).status, 'unknown');
  assert.equal(node.className, 'pill ');
});

test('relative time buckets', () => {
  assert.equal(relativeTime('2026-11-09T11:59:40Z', now), '刚刚');
  assert.equal(relativeTime('2026-11-09T11:30:00Z', now), '30分钟前');
  assert.equal(relativeTime('2026-11-09T09:00:00Z', now), '3小时前');
  assert.equal(relativeTime('2026-11-07T12:00:00Z', now), '2天前');
  assert.equal(relativeTime('2026-11-09T12:00:30Z', now), '刚刚', 'a clock slightly ahead never shows a negative age');
  assert.equal(relativeTime(null, now), null);
  assert.equal(relativeTime('not a time', now), null);
});

test('badge text follows the gateway status', () => {
  const valid = badgeView({ status: 'valid', label: '标定有效', calibrated_at: '2026-11-09T09:00:00Z', active_version: 'v1' }, now);
  assert.deepEqual([valid.text, valid.kind], ['标定有效 · 3小时前', 'ok']);
  assert.match(valid.title, /v1/);
  const stale = badgeView({ status: 'maybe_stale', label: '可能已过期', calibrated_at: '2026-11-09T09:00:00Z' }, now);
  assert.deepEqual([stale.text, stale.kind], ['可能已过期 · 3小时前', 'warn']);
  const needed = badgeView({ status: 'needed', label: '需要标定', calibrated_at: null }, now);
  assert.deepEqual([needed.text, needed.kind], ['需要标定', 'bad']);
  assert.deepEqual([badgeView(null, now).text, badgeView(null, now).kind], ['标定状态未知', '']);
});
