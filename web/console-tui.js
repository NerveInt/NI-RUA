// Read-only native TUI for the operator console (same /v1/terminal stream as the workbench).
import { Terminal } from '@xterm/xterm';
import { FitAddon } from '@xterm/addon-fit';
import '@xterm/xterm/css/xterm.css';
import { attachTerminalScroll, isLiveTerminalStatus } from './shared/terminal-scroll.js';
import { onLocaleChange, pick } from './shared/i18n.js';

export function mountNativeTui(container, status, options = {}) {
  const terminal = new Terminal({ disableStdin: true, scrollback: 5000, fontSize: 12, convertEol: false,
    fontFamily: 'ui-monospace, "SF Mono", Menlo, monospace',
    theme: { background: '#141b18', foreground: '#d2dfd6', selectionBackground: '#34533e' } });
  const fit = new FitAddon();
  terminal.loadAddon(fit);
  terminal.open(container);
  let socket, cursor = 0, session, terminalStatus = null, writes = Promise.resolve();
  let stateKind = 'connecting', stateDetail = null;
  const paintStatus = () => {
    switch (stateKind) {
      case 'connecting': status.textContent = pick('连接中', 'Connecting'); break;
      case 'connected': status.textContent = pick('已连接 · 只读', 'Connected · read-only'); break;
      case 'replay': status.textContent = pick('历史终端 · 只读', 'Archived terminal · read-only'); break;
      case 'live': status.textContent = pick('实时 · 只读', 'Live · read-only'); break;
      case 'exited': status.textContent = pick(`原生 TUI 已退出 (${stateDetail ?? '未知'})`, `Native TUI exited (${stateDetail ?? 'unknown'})`); break;
      case 'render_error': status.textContent = pick(`终端渲染异常 · ${stateDetail}`, `Terminal render error · ${stateDetail}`); break;
      case 'error': status.textContent = stateDetail || pick('终端尚未就绪', 'Terminal not ready yet'); break;
      case 'reconnecting': status.textContent = pick('断开 · 正在重连', 'Disconnected · reconnecting'); break;
    }
  };
  onLocaleChange(paintStatus);
  const root = container.closest?.('.native-tui');
  const scroll = attachTerminalScroll({
    container,
    terminal,
    isLive: () => socket?.readyState === WebSocket.OPEN && isLiveTerminalStatus(terminalStatus, { replay: Boolean(options.isReplay?.()) }),
    sendScroll: (payload) => {
      if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: 'scroll', ...payload }));
    },
    buttons: {
      pageUp: options.pageUp ?? root?.querySelector('#tui-page-up'),
      pageDown: options.pageDown ?? root?.querySelector('#tui-page-down'),
      bottom: options.bottom ?? root?.querySelector('#tui-bottom'),
    },
  });
  const size = () => {
    try { fit.fit(); } catch { return; }
    if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: 'resize',
      cols: Math.min(500, Math.max(20, terminal.cols)), rows: Math.min(200, Math.max(5, terminal.rows)) }));
  };
  new ResizeObserver(size).observe(container);
  const connect = () => {
    if (socket && socket.readyState <= WebSocket.OPEN) return;
    const url = new URL('/v1/terminal', location.href);
    url.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    url.searchParams.set('after', String(cursor));
    if (session) url.searchParams.set('session_id', session);
    const connection = new WebSocket(url);
    socket = connection;
    connection.onopen = () => { stateKind = 'connected'; stateDetail = null; paintStatus(); size(); };
    connection.onmessage = ({ data }) => {
      let message; try { message = JSON.parse(data); } catch { return; }
      if (message.type === 'ready' || message.type === 'status') {
        const value = message.status || message;
        session = value.session_id || session;
        terminalStatus = value;
        const isReplay = Boolean(value.replay?.kind || options.isReplay?.() || (value.pid == null && value.exit_code == null && value.cursor > 0));
        if (isReplay) { stateKind = 'replay'; stateDetail = null; }
        else if (value.exit_code != null) { stateKind = 'exited'; stateDetail = value.exit_code; }
        else { stateKind = 'live'; stateDetail = null; }
        paintStatus();
      } else if (message.type === 'output') {
        writes = writes.then(() => new Promise((resolve) => {
          if (message.cursor <= cursor) return resolve();
          if (message.after > cursor) { connection.close(); return resolve(); }
          const bytes = Uint8Array.from(atob(message.data), (c) => c.charCodeAt(0)).slice(Math.max(0, cursor - message.after));
          terminal.write(bytes, () => { cursor = message.cursor; resolve(); });
        })).catch((error) => {
          stateKind = 'render_error'; stateDetail = error?.message || error; paintStatus();
        });
      } else if (message.type === 'gap') {
        writes = writes.then(() => { scroll.reset(); terminal.reset(); cursor = 0; session = undefined; terminalStatus = null; connection.close(); }).catch((error) => {
          stateKind = 'render_error'; stateDetail = error?.message || error; paintStatus();
        });
      } else if (message.type === 'exit') {
        terminalStatus = { ...(terminalStatus || {}), exit_code: message.exit_code ?? -1 };
        stateKind = 'exited'; stateDetail = message.exit_code ?? null; paintStatus();
      } else if (message.type === 'error') {
        stateKind = 'error'; stateDetail = message.message || null; paintStatus();
      }
    };
    connection.onclose = () => { if (connection === socket) { stateKind = 'reconnecting'; paintStatus(); setTimeout(connect, 1500); } };
  };
  connect();
  return terminal;
}
