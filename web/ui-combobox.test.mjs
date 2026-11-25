import assert from 'node:assert/strict';
import test from 'node:test';
import { createTypeahead, nextIndex, typeaheadIndex } from './ui-combobox.js';

test('arrow, page and Home/End keys move within bounds; disabled rows stay reachable', () => {
  assert.equal(nextIndex(0, -1, 'ArrowDown'), -1);
  assert.equal(nextIndex(5, -1, 'ArrowDown'), 0);
  assert.equal(nextIndex(5, -1, 'ArrowUp'), 4);
  assert.equal(nextIndex(5, 2, 'ArrowDown'), 3);
  assert.equal(nextIndex(5, 4, 'ArrowDown'), 4, 'no wrap past the end');
  assert.equal(nextIndex(5, 0, 'ArrowUp'), 0, 'no wrap past the start');
  assert.equal(nextIndex(5, 3, 'Home'), 0);
  assert.equal(nextIndex(5, 1, 'End'), 4);
  assert.equal(nextIndex(20, 2, 'PageDown'), 10);
  assert.equal(nextIndex(20, 18, 'PageDown'), 19);
  assert.equal(nextIndex(20, 5, 'PageUp'), 0);
  assert.equal(nextIndex(20, -1, 'PageUp'), 12);
  assert.equal(nextIndex(5, 2, 'Tab'), 2, 'other keys keep the index');
});

test('type-ahead: one repeated letter cycles, a longer buffer refines, case-insensitive, wraps around', () => {
  const labels = ['Codex', 'Claude Code', 'OpenCode', 'claude-x'];
  assert.equal(typeaheadIndex(labels, -1, 'c'), 0);
  assert.equal(typeaheadIndex(labels, 0, 'c'), 1);
  assert.equal(typeaheadIndex(labels, 1, 'cc'), 3, 'repeated letter keeps cycling');
  assert.equal(typeaheadIndex(labels, 3, 'c'), 0, 'wraps');
  assert.equal(typeaheadIndex(labels, 0, 'cla'), 1, 'refines from the current option');
  assert.equal(typeaheadIndex(labels, 1, 'claude-'), 3);
  assert.equal(typeaheadIndex(labels, 0, 'O'), 2);
  assert.equal(typeaheadIndex(labels, 0, 'zz'), -1);
  assert.equal(typeaheadIndex([], 0, 'a'), -1);
});

test('the type-ahead buffer resets after a pause', () => {
  let now = 0;
  const type = createTypeahead({ timeoutMs: 500, clock: () => now });
  assert.equal(type('c'), 'c');
  now = 200; assert.equal(type('l'), 'cl');
  now = 800; assert.equal(type('o'), 'o');
});
