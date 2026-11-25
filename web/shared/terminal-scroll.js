// Shared read-only viewport scrolling for xterm.js terminals across /console and Workbench.
// Normal buffer (e.g. Codex --no-alt-screen): xterm.js scrolls its local scrollback buffer natively.
// Alternate buffer (e.g. OpenCode full-screen TUI): wheel, scroll keys and toolbar buttons send bounded
// server-synthesized scroll commands over /v1/terminal while the PTY is live.

const MAX_SCROLL_LINES = 40;
const PIXELS_PER_TICK = 32;
const LINE_HEIGHT_PX = 14;

const KEY_ACTIONS = Object.freeze({
  ArrowUp: { remote: 'line_up', local: (terminal) => terminal.scrollLines(-1) },
  ArrowDown: { remote: 'line_down', local: (terminal) => terminal.scrollLines(1) },
  PageUp: { remote: 'page_up', local: (terminal) => terminal.scrollPages(-1) },
  PageDown: { remote: 'page_down', local: (terminal) => terminal.scrollPages(1) },
  Home: { remote: 'top', local: (terminal) => terminal.scrollToTop() },
  End: { remote: 'bottom', local: (terminal) => terminal.scrollToBottom() },
});

export function isAlternateBuffer(terminal) {
  return terminal?.buffer?.active?.type === 'alternate';
}

export function isLiveTerminalStatus(status, { replay = false } = {}) {
  if (!status || replay || status.replay?.kind) return false;
  if (status.exit_code !== undefined && status.exit_code !== null) return false;
  return Number.isInteger(status.pid) && status.pid > 0;
}

export function attachTerminalScroll({
  container,
  terminal,
  isLive,
  sendScroll,
  buttons = {},
  scheduleFlush = (callback) => (typeof requestAnimationFrame === 'function' ? requestAnimationFrame(callback) : queueMicrotask(callback)),
}) {
  let pixelRemainder = 0;
  let pendingLines = 0;
  let flushScheduled = false;

  const useRemoteScroll = () => isAlternateBuffer(terminal) && Boolean(isLive?.());

  const resetAccumulator = () => {
    pixelRemainder = 0;
    pendingLines = 0;
  };

  const flushPending = () => {
    flushScheduled = false;
    if (!pendingLines) return;
    const lines = Math.max(-MAX_SCROLL_LINES, Math.min(MAX_SCROLL_LINES, pendingLines));
    pendingLines = 0;
    if (lines !== 0 && useRemoteScroll()) sendScroll({ lines });
  };

  const triggerAction = (remoteAction, localFn) => {
    resetAccumulator();
    if (useRemoteScroll()) {
      sendScroll({ action: remoteAction });
    } else if (!isAlternateBuffer(terminal)) {
      localFn(terminal);
    }
  };

  const onWheel = (event) => {
    if (event.ctrlKey || event.metaKey) {
      // Stop xterm's mouse-tracking wheel handler from calling preventDefault() on pinch/Ctrl-zoom.
      if (isAlternateBuffer(terminal)) event.stopPropagation();
      return;
    }
    if (!useRemoteScroll()) {
      resetAccumulator();
      // In exited/replay alternate-screen archives, stop xterm's mouse-tracking listener from swallowing page scroll.
      if (isAlternateBuffer(terminal)) event.stopPropagation();
      return;
    }
    if (!Number.isFinite(event.deltaY) || event.deltaY === 0) return;
    event.preventDefault();
    event.stopPropagation();
    const mode = event.deltaMode ?? 0;
    const pageHeight = container?.clientHeight || 360;
    const deltaPx = mode === 1 ? event.deltaY * LINE_HEIGHT_PX : mode === 2 ? event.deltaY * pageHeight : event.deltaY;
    if ((pixelRemainder > 0 && deltaPx < 0) || (pixelRemainder < 0 && deltaPx > 0)) pixelRemainder = 0;
    pixelRemainder += deltaPx;
    const ticks = Math.trunc(pixelRemainder / PIXELS_PER_TICK);
    if (ticks === 0) return;
    pixelRemainder -= ticks * PIXELS_PER_TICK;
    pendingLines = Math.max(-MAX_SCROLL_LINES, Math.min(MAX_SCROLL_LINES, pendingLines + ticks));
    if (!flushScheduled) {
      flushScheduled = true;
      scheduleFlush(flushPending);
    }
  };

  const onKeyDown = (event) => {
    if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
    const mapping = KEY_ACTIONS[event.key];
    if (!mapping) return;
    if (!useRemoteScroll() && isAlternateBuffer(terminal)) return;
    event.preventDefault();
    event.stopPropagation();
    triggerAction(mapping.remote, mapping.local);
  };

  if (container && !container.hasAttribute?.('tabindex')) container.setAttribute?.('tabindex', '0');
  container?.addEventListener?.('wheel', onWheel, { capture: true, passive: false });
  container?.addEventListener?.('keydown', onKeyDown, { capture: true });

  const pageUp = () => triggerAction('page_up', (t) => t.scrollPages(-1));
  const pageDown = () => triggerAction('page_down', (t) => t.scrollPages(1));
  const bottom = () => triggerAction('bottom', (t) => t.scrollToBottom());

  if (buttons.pageUp) buttons.pageUp.onclick = pageUp;
  if (buttons.pageDown) buttons.pageDown.onclick = pageDown;
  if (buttons.bottom) buttons.bottom.onclick = bottom;

  return {
    pageUp,
    pageDown,
    bottom,
    reset: resetAccumulator,
    dispose() {
      resetAccumulator();
      container?.removeEventListener?.('wheel', onWheel, { capture: true });
      container?.removeEventListener?.('keydown', onKeyDown, { capture: true });
    },
  };
}
