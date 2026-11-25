import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { attachTerminalScroll, isAlternateBuffer, isLiveTerminalStatus } from './terminal-scroll.js';

const webRoot = join(dirname(fileURLToPath(import.meta.url)), '..');

function makeHarness({ bufferType = 'normal', live = true } = {}) {
  let activeType = bufferType;
  let isLiveState = live;
  const listeners = new Map();
  const attrs = new Map();
  const localCalls = [];
  const remoteCalls = [];
  const flushes = [];

  const container = {
    clientHeight: 480,
    hasAttribute: (name) => attrs.has(name),
    setAttribute: (name, value) => attrs.set(name, value),
    addEventListener: (type, fn, opts) => listeners.set(type, { fn, opts }),
    removeEventListener: (type) => listeners.delete(type),
  };
  const terminal = {
    buffer: { get active() { return { type: activeType }; } },
    scrollLines: (n) => localCalls.push(['scrollLines', n]),
    scrollPages: (n) => localCalls.push(['scrollPages', n]),
    scrollToTop: () => localCalls.push(['scrollToTop']),
    scrollToBottom: () => localCalls.push(['scrollToBottom']),
  };
  const buttons = { pageUp: {}, pageDown: {}, bottom: {} };
  const control = attachTerminalScroll({
    container,
    terminal,
    isLive: () => isLiveState,
    sendScroll: (payload) => remoteCalls.push(payload),
    buttons,
    scheduleFlush: (cb) => flushes.push(cb),
  });

  const dispatch = (type, init, { downstreamXtermMouseCancel = false } = {}) => {
    let prevented = false;
    let stopped = false;
    const event = {
      preventDefault() { prevented = true; },
      stopPropagation() { stopped = true; },
      ...init,
    };
    listeners.get(type)?.fn(event);
    if (!stopped && downstreamXtermMouseCancel) {
      event.preventDefault();
      event.stopPropagation();
    }
    return { prevented, stopped };
  };

  return {
    attrs,
    listeners,
    localCalls,
    remoteCalls,
    buttons,
    control,
    setBufferType: (next) => { activeType = next; },
    setLive: (next) => { isLiveState = next; },
    flushAll: () => { while (flushes.length) flushes.shift()(); },
    dispatch,
  };
}

test('normal buffer leaves wheel events to native xterm scrollback and routes buttons/keys locally', () => {
  const h = makeHarness({ bufferType: 'normal', live: true });
  assert.equal(h.attrs.get('tabindex'), '0');
  assert.equal(h.listeners.get('wheel').opts.capture, true);
  assert.equal(h.listeners.get('wheel').opts.passive, false);
  assert.equal(isAlternateBuffer({ buffer: { active: { type: 'normal' } } }), false);
  assert.equal(isAlternateBuffer({ buffer: { active: { type: 'alternate' } } }), true);

  const wheel = h.dispatch('wheel', { deltaY: -120, deltaMode: 0 });
  h.flushAll();
  assert.equal(wheel.prevented, false);
  assert.equal(wheel.stopped, false);
  assert.deepEqual(h.remoteCalls, []);

  h.buttons.pageUp.onclick();
  h.buttons.pageDown.onclick();
  h.buttons.bottom.onclick();
  assert.deepEqual(h.remoteCalls, []);
  assert.deepEqual(h.localCalls, [['scrollPages', -1], ['scrollPages', 1], ['scrollToBottom']]);

  h.localCalls.length = 0;
  const key = h.dispatch('keydown', { key: 'ArrowUp' });
  assert.equal(key.prevented, true);
  assert.deepEqual(h.localCalls, [['scrollLines', -1]]);
  assert.deepEqual(h.remoteCalls, []);
});

test('live alternate buffer coalesces pixel/line/page wheel deltas, resets on reversal, and clamps to [-40, 40]', () => {
  const h = makeHarness({ bufferType: 'alternate', live: true });

  // Sub-tick trackpad gesture still prevents outer page scroll while accumulating.
  const sub = h.dispatch('wheel', { deltaY: -18, deltaMode: 0 });
  assert.equal(sub.prevented, true);
  assert.equal(sub.stopped, true);
  h.flushAll();
  assert.deepEqual(h.remoteCalls, []);

  // Second gesture crosses -32px threshold -> sends -1 tick.
  h.dispatch('wheel', { deltaY: -20, deltaMode: 0 });
  h.flushAll();
  assert.deepEqual(h.remoteCalls, [{ lines: -1 }]);

  // Direction reversal clears opposite remainder immediately and coalesces within one frame.
  h.remoteCalls.length = 0;
  h.dispatch('wheel', { deltaY: 40, deltaMode: 0 });
  h.dispatch('wheel', { deltaY: 64, deltaMode: 0 });
  h.flushAll();
  assert.deepEqual(h.remoteCalls, [{ lines: 3 }]);

  // Huge burst clamps to [-40, 40].
  h.remoteCalls.length = 0;
  h.dispatch('wheel', { deltaY: -5000, deltaMode: 0 });
  h.flushAll();
  assert.deepEqual(h.remoteCalls, [{ lines: -40 }]);

  // DOM_DELTA_LINE (mode 1) and DOM_DELTA_PAGE (mode 2).
  h.remoteCalls.length = 0;
  h.dispatch('wheel', { deltaY: 3, deltaMode: 1 }); // 42px -> 1 tick
  h.flushAll();
  assert.deepEqual(h.remoteCalls, [{ lines: 1 }]);

  h.remoteCalls.length = 0;
  h.dispatch('wheel', { deltaY: -1, deltaMode: 2 }); // -480px -> -15 ticks
  h.flushAll();
  assert.deepEqual(h.remoteCalls, [{ lines: -15 }]);

  // Pinch-zoom (ctrlKey) stops propagation before xterm's mouse listener can preventDefault(), preserving zoom.
  h.remoteCalls.length = 0;
  const zoom = h.dispatch('wheel', { deltaY: -100, deltaMode: 0, ctrlKey: true }, { downstreamXtermMouseCancel: true });
  assert.equal(zoom.prevented, false);
  assert.equal(zoom.stopped, true);
  h.flushAll();
  assert.deepEqual(h.remoteCalls, []);
});

test('live alternate buffer routes toolbar buttons and scroll keys remotely without hijacking modified shortcuts', () => {
  const h = makeHarness({ bufferType: 'alternate', live: true });

  h.buttons.pageUp.onclick();
  h.buttons.pageDown.onclick();
  h.buttons.bottom.onclick();
  assert.deepEqual(h.remoteCalls, [
    { action: 'page_up' },
    { action: 'page_down' },
    { action: 'bottom' },
  ]);
  assert.deepEqual(h.localCalls, []);

  h.remoteCalls.length = 0;
  for (const key of ['ArrowUp', 'ArrowDown', 'PageUp', 'PageDown', 'Home', 'End']) {
    const res = h.dispatch('keydown', { key });
    assert.equal(res.prevented, true, key);
  }
  assert.deepEqual(h.remoteCalls, [
    { action: 'line_up' },
    { action: 'line_down' },
    { action: 'page_up' },
    { action: 'page_down' },
    { action: 'top' },
    { action: 'bottom' },
  ]);

  // Modified keys (Ctrl/Cmd/Shift) and regular typing keys are untouched.
  h.remoteCalls.length = 0;
  assert.equal(h.dispatch('keydown', { key: 'ArrowUp', ctrlKey: true }).prevented, false);
  assert.equal(h.dispatch('keydown', { key: 'PageUp', shiftKey: true }).prevented, false);
  assert.equal(h.dispatch('keydown', { key: 'Enter' }).prevented, false);
  assert.equal(h.dispatch('keydown', { key: 'a' }).prevented, false);
  assert.deepEqual(h.remoteCalls, []);
});

test('exited or replay alternate buffer never sends WebSocket scroll frames or traps page scrolling', () => {
  assert.equal(isLiveTerminalStatus({ pid: 501, exit_code: null }), true, 'live PTY with positive pid');
  assert.equal(isLiveTerminalStatus({ pid: null, exit_code: null }), false, 'standalone Gateway replay (pid: null, exit_code: null) is not live');
  assert.equal(isLiveTerminalStatus({ pid: 501, exit_code: null }, { replay: true }), false, 'explicit replay mode overrides live status');
  assert.equal(isLiveTerminalStatus({ pid: 501, exit_code: null, replay: { kind: 'original_pty' } }), false, 'library archive replay.kind is not live');
  assert.equal(isLiveTerminalStatus({ pid: 501, exit_code: 0 }), false, 'exited PTY is not live');
  assert.equal(isLiveTerminalStatus(null), false, 'missing status before ready is not live');

  let status = { pid: null, exit_code: null };
  let replayFlag = false;
  const h = makeHarness({ bufferType: 'alternate', live: false });
  // Also verify via dynamic isLive wired to isLiveTerminalStatus:
  const sent = [];
  const listeners = new Map();
  const buttons = { pageUp: {}, pageDown: {}, bottom: {} };
  attachTerminalScroll({
    container: {
      clientHeight: 480,
      hasAttribute: () => true,
      addEventListener: (t, fn) => listeners.set(t, fn),
    },
    terminal: { buffer: { active: { type: 'alternate' } } },
    isLive: () => isLiveTerminalStatus(status, { replay: replayFlag }),
    sendScroll: (p) => sent.push(p),
    buttons,
    scheduleFlush: (cb) => cb(),
  });
  // Standalone Gateway replay (pid: null, exit_code: null): wheel does not preventDefault, buttons send nothing.
  let defaultPrevented = false;
  let propagationStopped = false;
  listeners.get('wheel')({
    deltaY: -120,
    deltaMode: 0,
    preventDefault() { defaultPrevented = true; },
    stopPropagation() { propagationStopped = true; },
  });
  assert.equal(defaultPrevented, false);
  assert.equal(propagationStopped, true);
  buttons.pageUp.onclick();
  assert.deepEqual(sent, []);

  // Once a live PTY status arrives (pid > 0, exit_code: null, replay: false), remote scroll activates.
  status = { pid: 1234, exit_code: null };
  buttons.pageUp.onclick();
  assert.deepEqual(sent, [{ action: 'page_up' }]);

  // If run status reports replay=true, remote scroll shuts back off.
  replayFlag = true;
  sent.length = 0;
  buttons.bottom.onclick();
  assert.deepEqual(sent, []);

  const wheel = h.dispatch('wheel', { deltaY: -120, deltaMode: 0 }, { downstreamXtermMouseCancel: true });
  h.flushAll();
  assert.equal(wheel.prevented, false, 'capture stopPropagation shields page scroll from xterm mouse wheel cancel');
  assert.equal(wheel.stopped, true);
  h.buttons.pageUp.onclick();
  h.buttons.bottom.onclick();
  assert.equal(h.dispatch('keydown', { key: 'PageUp' }).prevented, false);
  assert.deepEqual(h.remoteCalls, []);
});

test('console and workbench HTML/CSS expose accessible scroll controls and vertical resize', () => {
  const consoleHtml = readFileSync(join(webRoot, 'console.html'), 'utf8');
  const consoleTuiJs = readFileSync(join(webRoot, 'console-tui.js'), 'utf8');
  const indexHtml = readFileSync(join(webRoot, 'index.html'), 'utf8');
  const consoleCss = readFileSync(join(webRoot, 'console.css'), 'utf8');
  const stylesCss = readFileSync(join(webRoot, 'styles.css'), 'utf8');

  for (const id of ['tui-page-up', 'tui-page-down', 'tui-bottom']) {
    assert.match(consoleHtml, new RegExp(`id="${id}"`), `console.html missing #${id}`);
  }
  assert.match(consoleHtml, /id="tui"[^>]*tabindex="0"/);
  assert.match(consoleTuiJs, /isLiveTerminalStatus/);

  for (const id of ['terminal-page-up', 'terminal-page-down', 'terminal-bottom']) {
    assert.match(indexHtml, new RegExp(`id="${id}"`), `index.html missing #${id}`);
  }
  assert.match(indexHtml, /id="terminal"[^>]*tabindex="0"/);

  assert.match(consoleCss, /#tui\{[^}]*resize:vertical/);
  assert.match(consoleCss, /@media\(pointer:coarse\)\{[\s\S]*?\.tui-scroll button\{min-height:44px/);
  assert.match(stylesCss, /#terminal\{[^}]*resize:vertical/);
});
