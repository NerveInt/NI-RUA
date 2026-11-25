import { buildPhases, mapActionVideo, clipTransition, isNativeAgentEvent } from './phase-model.js';
import { getLocale, onLocaleChange, pick } from './shared/i18n.js';

const $ = (id) => document.getElementById(id);
const cameraName = (id) => ({
  cam_head: pick('头部 · 主视角', 'Head · Main View'),
  head: pick('头部 · 主视角', 'Head · Main View'),
  cam_left_wrist: pick('左腕', 'Left Wrist'),
  cam_right_wrist: pick('右腕', 'Right Wrist'),
  fixed: pick('固定 · 主视角', 'Fixed · Main View'),
  wrist: pick('腕部', 'Wrist'),
})[id] || id;
const toolName = (item) => item?.tool || item?.name || item?.type || pick('Harness 动作', 'Harness Action');
const messageText = (item) => {
  if (!item) return '';
  const parts = [item.text, ...(Array.isArray(item.summary) ? item.summary : []), ...(Array.isArray(item.content) ? item.content : [])];
  return [...new Set(parts.map((part) => typeof part === 'string' ? part : part?.text || '').filter(Boolean))].join('\n');
};
function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}
function details(title, value, open = false) {
  const element = node('details'); element.open = open;
  element.append(node('summary', title), node('pre', typeof value === 'string' ? value : JSON.stringify(value, null, 2)));
  return element;
}

// Review uses recorded identities and PTS only. It never submits run controls.
export function createPhaseReview({ read, artifactURL, terminal, inspectEvent, terminalReady, onPhaseChange, agentName = () => 'Codex' }) {
  const events = new Map(), actions = new Map(), actionVersions = new Map(), pending = new Set(), observationCache = new Map();
  const actionFailures = new Map();
  const navNodes = new Map(), activityNodes = new Map();
  let items = [], records = [], phases = [], selectedId, userSelected = false, timer, refreshing = false, refreshAgain = false;
  let segments = [], segmentIndex = 0, clip, selectionToken = 0, contextSignature, mediaSignature, pendingSeek = false, advancing = false;
  let autoTerminal = true, retryTimer, drawTimer, seeking = false, eventCursor = 0, expectedCursor = 0;
  let windowVersion = 0;
  let transitionNotice = '';
  const video = $('phase-video');
  const frameCallbacks = typeof video.requestVideoFrameCallback === 'function';

  function schedule() { clearTimeout(timer); timer = setTimeout(refresh, 120); }
  function clipsFor(phase, camera = $('phase-camera').value) {
    const clips = [];
    const sources = [...records].reverse();
    const preferred = sources.findIndex((record) => record.entry.camera_id === camera && artifactURL(record.entry.video_ref) === clip?.url);
    if (preferred > 0) sources.unshift(...sources.splice(preferred, 1));
    for (const entry of phase.actions || []) {
      if (!entry.action) continue;
      for (const record of sources) {
        if (record.entry.camera_id !== camera) continue;
        const mapping = mapActionVideo(record, entry.action, entry.result);
        if (mapping.reason || mapping.pause) continue;
        const url = artifactURL(record.entry.video_ref);
        if (!url) continue;
        clips.push({ ...mapping, record, actionId: entry.action_id, url }); break;
      }
    }
    return clips;
  }
  function noClipReason(phase) {
    if (eventCursor < expectedCursor) return pick(`正在读取阶段对应的事件与录像（${eventCursor}/${expectedCursor}）…`, `Loading correlated phase events and recordings (${eventCursor}/${expectedCursor})…`);
    if (!phase.actions?.length) return pick('此阶段没有机器人运动录像；观察、等待与思考阶段保留文字，连播时自动跳过。', 'No robot motion video in this phase; observation, wait, and reasoning phases retain text records and are skipped during continuous playback.');
    if (phase.actions.some((entry) => !entry.action)) return pick('动作证据尚未读取完成或不可用。', 'Action evidence is still loading or unavailable.');
    if (!records.length) return pick('这个运行尚未导出录像。', 'No exported video recordings for this run yet.');
    const matching = records.filter((record) => record.entry.camera_id === $('phase-camera').value);
    if (!matching.length) return pick('本运行没有保存这个视角的录像。', 'No recording saved for this camera view in this run.');
    const reasons = phase.actions.flatMap((entry) => matching.map((record) => mapActionVideo(record, entry.action, entry.result)));
    if (reasons.some((mapping) => mapping.pause)) return pick('此阶段仿真步未前进，没有运动片段；连播时自动跳过。', 'Simulation step did not advance in this phase (no motion clip); skipped during continuous playback.');
    return `${pick('没有可核对的阶段片段', 'No verifiable phase clip')} · ${reasons.find((mapping) => mapping.reason)?.reason || pick('映射未记录', 'mapping unrecorded')}`;
  }
  function syncCameras() {
    const ids = [...new Set(records.map((record) => record.entry.camera_id))];
    ids.sort((a, b) => (a.includes('head') ? -1 : b.includes('head') ? 1 : a.localeCompare(b)));
    if (!ids.length) return;
    const signature = `${getLocale()}:${JSON.stringify(ids)}`, select = $('phase-camera');
    if (select.dataset.signature === signature) return;
    const previous = select.value; select.dataset.signature = signature;
    select.replaceChildren(...ids.map((id) => { const option = node('option', cameraName(id)); option.value = id; return option; }));
    if (ids.includes(previous)) select.value = previous;
  }
  function seedRecordedActions() {
    const latest = new Map(), requests = new Map();
    for (const event of events.values()) {
      if (event.correlation?.action_id && event.type?.startsWith('action.')) latest.set(event.correlation.action_id, event);
      if (event.type === 'action.requested' && event.payload?.request?.action_id === event.correlation?.action_id) requests.set(event.correlation.action_id, event.payload.request);
    }
    for (const map of [actions, actionVersions, actionFailures]) for (const id of map.keys()) if (!latest.has(id)) map.delete(id);
    for (const [id, event] of latest) {
      const request = requests.get(id), receipt = event.payload, result = observationCache.get(receipt?.result_observation_id);
      if (request && result && ['action.completed', 'action.failed', 'action.stopped'].includes(event.type) && receipt?.execution_interval && receipt.status) {
        actions.set(id, { action: { request, receipt }, result }); actionVersions.set(id, event.event_id); actionFailures.delete(id);
      }
    }
    return latest;
  }
  async function hydrate() {
    const version = windowVersion;
    const latest = seedRecordedActions();
    const queue = [...latest].filter(([id, event]) => actionVersions.get(id) !== event.event_id && !pending.has(id));
    async function worker() {
      while (queue.length) {
        const [id, event] = queue.shift();
        if (actionVersions.get(id) === event.event_id) continue;
        pending.add(id);
        try {
          const action = await read(`/v1/actions/${encodeURIComponent(id)}`, { signal: AbortSignal.timeout(15000) });
          if (version !== windowVersion) return;
          let result, resultError;
          const resultId = action.receipt?.result_observation_id || action.result_observation_id;
          if (resultId) {
            try {
              if (!observationCache.has(resultId)) {
                const observation = await read(`/v1/observations/${encodeURIComponent(resultId)}`, { signal: AbortSignal.timeout(15000) });
                if (version !== windowVersion) return;
                observationCache.set(resultId, observation);
                if (observationCache.size > 2000) observationCache.delete(observationCache.keys().next().value);
              }
              result = observationCache.get(resultId);
            } catch (error) { resultError = error; }
          }
          if (actionVersions.get(id) === event.event_id) continue;
          actions.set(id, { action, result });
          if (resultError) throw resultError;
          actionVersions.set(id, event.event_id); actionFailures.delete(id);
        } catch (error) {
          if (version !== windowVersion) return;
          if (actionVersions.get(id) === event.event_id) continue;
          const previous = actionFailures.get(id);
          const attempts = (previous?.version === event.event_id ? previous.attempts : 0) + 1;
          actionFailures.set(id, { version: event.event_id, attempts, message: error.message });
          if (attempts >= 3) actionVersions.set(id, event.event_id);
          else { clearTimeout(retryTimer); retryTimer = setTimeout(schedule, 1000); }
        }
        finally { pending.delete(id); clearTimeout(drawTimer); drawTimer = setTimeout(draw, 100); }
      }
    }
    await Promise.all(Array.from({ length: Math.min(2, queue.length) }, worker));
  }
  function draw() {
    seedRecordedActions();
    const previous = phases.find((value) => value.id === selectedId);
    phases = buildPhases([...events.values()], items, actions, agentName());
    syncCameras(); renderNavigation();
    let phase = phases.find((value) => value.id === selectedId);
    if (!userSelected) phase = phases.find((value) => clipsFor(value).length) || phase;
    if (!phase && previous?.actions.length) phase = phases.find((value) => value.actions.some((entry) => entry.action_id === previous.actions[0].action_id));
    phase ||= phases[0];
    if (phase) select(phase.id, { automatic: true });
    else {
      $('phase-empty').textContent = pick('尚无可用的阶段记录。', 'No phase records available yet.');
      $('phase-count').textContent = pick('尚无阶段', 'No phases yet');
      onPhaseChange?.(null, { index: -1, total: 0 });
    }
  }
  async function refresh() {
    draw();
    if (refreshing) { refreshAgain = true; return; }
    refreshing = true;
    try {
      await hydrate();
      draw();
    } finally { refreshing = false; if (refreshAgain) { refreshAgain = false; schedule(); } }
  }
  function renderNavigation() {
    const ids = new Set(phases.map((phase) => phase.id));
    for (const map of [navNodes, activityNodes]) for (const [id, element] of map) if (!ids.has(id)) { element.remove(); map.delete(id); }
    $('phase-activity').querySelector('.empty')?.remove();
    let playable = 0;
    phases.forEach((phase, index) => {
      const count = clipsFor(phase).length; if (count) playable++;
      let button = navNodes.get(phase.id);
      if (!button) {
        button = node('button'); button.type = 'button'; button.dataset.phaseId = phase.id;
        button.onclick = () => select(phase.id); navNodes.set(phase.id, button);
      }
      const title = `${index + 1}. ${phase.title}`;
      const awaiting = eventCursor < expectedCursor || phase.actions.some((entry) => !entry.action && (actionFailures.get(entry.action_id)?.attempts || 0) < 3);
      const badge = count
        ? pick(`${count} 段视频`, `${count} ${count === 1 ? 'clip' : 'clips'}`)
        : awaiting
          ? pick('读取中', 'Loading')
          : pick('无视频', 'No video');
      button.textContent = `${title} · ${badge}`;
      button.title = title; button.classList.toggle('has-video', Boolean(count));
      if ($('phase-nav').children[index] !== button) $('phase-nav').insertBefore(button, $('phase-nav').children[index] || null);
      let li = activityNodes.get(phase.id);
      if (!li) {
        li = node('li'); const action = node('button'); action.type = 'button'; action.dataset.phaseId = phase.id;
        action.onclick = () => { select(phase.id); $('phase-title').scrollIntoView({ block: 'center', behavior: 'smooth' }); };
        li.append(action); activityNodes.set(phase.id, li);
      }
      const summary = messageText(phase.before?.at(-1)) || phase.item?.arguments?.language || phase.item?.arguments?.prompt || '';
      li.firstChild.replaceChildren(
        node('span', String(index + 1).padStart(2, '0'), 'phase-number'),
        node('span', title.slice(title.indexOf(' ') + 1), 'phase-name'),
        node('span', count || awaiting ? badge : pick('无视频 · 连播跳过', 'No video · skipped'), 'phase-badge'),
        node('span', typeof summary === 'string' ? summary.slice(0, 180) : '', 'phase-summary'),
      );
      if ($('phase-activity').children[index] !== li) $('phase-activity').insertBefore(li, $('phase-activity').children[index] || null);
    });
    $('phase-count').textContent = pick(
      `${phases.length} 个阶段 · ${playable} 个含视频 · ${cameraName($('phase-camera').value)}${eventCursor < expectedCursor ? ` · 事件读取中 ${eventCursor}/${expectedCursor}` : ''}`,
      `${phases.length} phases · ${playable} with video · ${cameraName($('phase-camera').value)}${eventCursor < expectedCursor ? ` · loading events ${eventCursor}/${expectedCursor}` : ''}`,
    );
  }
  function select(id, { automatic = false, play = false, preserveTime = false, part = 0, notice } = {}) {
    const phase = phases.find((value) => value.id === id); if (!phase) return;
    const changed = selectedId !== id;
    selectedId = id; if (!automatic) userSelected = true;
    const index = phases.indexOf(phase);
    $('phase-title').textContent = pick(`阶段 ${index + 1} / ${phases.length} · ${phase.title}`, `Phase ${index + 1} / ${phases.length} · ${phase.title}`);
    $('phase-previous').disabled = index <= 0; $('phase-next').disabled = index >= phases.length - 1;
    for (const [key, button] of navNodes) {
      if (key === id) button.setAttribute('aria-current', 'step'); else button.removeAttribute('aria-current');
    }
    for (const [key, li] of activityNodes) {
      if (key === id) li.firstChild.setAttribute('aria-current', 'step'); else li.firstChild.removeAttribute('aria-current');
    }
    if (changed) {
      const nav = $('phase-nav'), button = navNodes.get(id), list = $('phase-activity'), row = activityNodes.get(id);
      if (button) {
        const bounds = button.getBoundingClientRect(), viewport = nav.getBoundingClientRect();
        if (bounds.left < viewport.left || bounds.right > viewport.right) nav.scrollLeft += bounds.left - viewport.left - (nav.clientWidth - bounds.width) / 2;
      }
      if (row) {
        const bounds = row.getBoundingClientRect(), viewport = list.getBoundingClientRect();
        if (bounds.top < viewport.top || bounds.bottom > viewport.bottom) list.scrollTop += bounds.top - viewport.top - (list.clientHeight - bounds.height) / 2;
      }
    }
    const previousClip = clip;
    segments = clipsFor(phase);
    const signature = JSON.stringify([getLocale(), id, segments.map((entry) => [entry.url, entry.start.pts_s, entry.end.pts_s])]);
    renderContext(phase);
    onPhaseChange?.(phase, { index, total: phases.length });
    if (mediaSignature !== signature || !automatic) {
      mediaSignature = signature;
      renderParts();
      if (segments.length) {
        const retained = automatic && !changed && previousClip ? segments.findIndex((entry) => entry.actionId === previousClip.actionId && entry.url === previousClip.url
          && entry.start.pts_s === previousClip.start.pts_s && entry.end.pts_s === previousClip.end.pts_s) : -1;
        showClip(retained >= 0 ? retained : Math.min(part, segments.length - 1), { play, preserveTime: preserveTime || retained >= 0, notice });
      }
      else {
        selectionToken++; clip = null; pendingSeek = false; transitionNotice = ''; video.pause(); video.hidden = true; video.removeAttribute('src'); video.load();
        delete video.dataset.startPts; delete video.dataset.endPts;
        $('phase-empty').hidden = false; $('phase-playing').hidden = true;
      }
    }
    if (!segments.length) {
      $('phase-empty').textContent = noClipReason(phase);
      $('phase-position').textContent = eventCursor < expectedCursor
        ? pick('正在核对事件与录像', 'Verifying events and recordings')
        : pick('无视频片段 · 阶段文字与工具记录仍可查看', 'No video clip · phase text and tool records remain available');
    }
    $('phase-play').disabled = !segments.length;
    if (changed) { autoTerminal = true; jumpTerminal(phase, true); }
  }
  function renderParts() {
    $('phase-segments').replaceChildren(...segments.map((entry, index) => {
      const button = node('button', pick(`片段 ${index + 1} · ${(entry.end.pts_s - entry.start.pts_s).toFixed(2)} 秒`, `Clip ${index + 1} · ${(entry.end.pts_s - entry.start.pts_s).toFixed(2)} s`));
      button.type = 'button'; button.onclick = () => showClip(index); return button;
    }));
    $('phase-segments').hidden = segments.length < 2;
  }
  function showClip(index, { play = false, preserveTime = false, notice } = {}) {
    clip = segments[index]; if (!clip) return;
    segmentIndex = index;
    const current = clip;
    if (notice !== undefined) transitionNotice = notice;
    else if (!preserveTime) transitionNotice = '';
    video.hidden = false; $('phase-empty').hidden = true; $('phase-playing').hidden = false;
    const phaseIndex = phases.findIndex((phase) => phase.id === selectedId);
    $('phase-playing').textContent = pick(`阶段 ${phaseIndex + 1} · 片段 ${index + 1}/${segments.length}`, `Phase ${phaseIndex + 1} · Clip ${index + 1}/${segments.length}`);
    video.dataset.startPts = current.start.pts_s; video.dataset.endPts = current.end.pts_s;
    video.dataset.phaseId = selectedId; video.dataset.actionId = current.actionId;
    for (const [i, button] of [...$('phase-segments').children].entries()) button.setAttribute('aria-pressed', String(i === index));
    updatePosition();
    if (preserveTime) return;
    const token = ++selectionToken;
    pendingSeek = true; video.pause();
    const seek = () => {
      if (token !== selectionToken) return;
      video.currentTime = current.start.pts_s;
      pendingSeek = false;
      if (play) video.play().catch(() => { $('phase-position').textContent += pick(' · 点击播放继续', ' · click Play to continue'); });
    };
    const url = new URL(current.url, location.href).href;
    if (video.src === url && video.readyState >= 1) seek();
    else { video.onloadedmetadata = seek; if (video.src !== url) video.src = url; video.load(); }
  }
  function updatePosition() {
    if (!clip) return;
    $('phase-position').textContent = pick(
      `片段 ${segmentIndex + 1}/${segments.length} · 录像 ${clip.start.pts_s.toFixed(3)}–${clip.end.pts_s.toFixed(3)} 秒 · 当前 ${video.currentTime.toFixed(3)} 秒${transitionNotice ? ' · ' + transitionNotice : ''}`,
      `Clip ${segmentIndex + 1}/${segments.length} · PTS ${clip.start.pts_s.toFixed(3)}–${clip.end.pts_s.toFixed(3)} s · now ${video.currentTime.toFixed(3)} s${transitionNotice ? ' · ' + transitionNotice : ''}`,
    );
    $('phase-position').title = `${clip.domain} ${clip.start.sim_step}–${clip.end.sim_step} · ${clip.actionId}${clip.record.entry.timing ? ' · ' + clip.record.entry.timing : ''}`;
  }
  function nextClip() {
    const current = phases.findIndex((phase) => phase.id === selectedId);
    if (segmentIndex + 1 < segments.length) return { phase: phases[current], part: segmentIndex + 1, clip: segments[segmentIndex + 1] };
    if (!$('phase-continuous').checked) return null;
    for (const phase of phases.slice(current + 1)) {
      const candidates = clipsFor(phase);
      if (candidates.length) return { phase, part: 0, clip: candidates[0] };
    }
    return null;
  }
  function finishClip(time = video.currentTime) {
    if (!clip || pendingSeek || advancing || seeking || video.seeking) return;
    advancing = true;
    try {
      while (clip && time >= clip.end.pts_s) {
        const next = nextClip();
        if (!next) {
          video.pause();
          if (video.currentTime > clip.end.pts_s + 0.03) video.currentTime = clip.end.pts_s;
          transitionNotice = pick('本阶段播放结束', 'Phase playback finished'); updatePosition(); return;
        }
        const transition = clipTransition(clip, next.clip);
        const notice = transition.continuous ? '' : transition.kind === 'gap'
          ? pick(`已跳过未关联片段（录像 ${transition.from.toFixed(3)}–${transition.to.toFixed(3)} 秒）`, `Skipped unassociated interval (${transition.from.toFixed(3)}–${transition.to.toFixed(3)} s)`)
          : transition.kind === 'source' ? pick('已切换录像文件，按该文件的记录时间定位', 'Switched video file; seeking by recorded PTS')
            : pick('片段边界不连续，已定位下一片段', 'Discontinuous clip boundary; sought to next clip');
        video.dataset.transition = transition.kind;
        const options = { play: !transition.continuous, preserveTime: transition.continuous, notice };
        if (next.phase.id === selectedId) showClip(next.part, options);
        else select(next.phase.id, { ...options, part: next.part });
        if (!transition.continuous) return;
      }
    } finally { advancing = false; }
  }
  video.addEventListener('timeupdate', () => {
    updatePosition();
    if (!frameCallbacks && !video.paused) finishClip();
  });
  if (frameCallbacks) {
    const frame = (_now, metadata) => {
      if (!video.paused) finishClip(metadata.mediaTime);
      video.requestVideoFrameCallback(frame);
    };
    video.requestVideoFrameCallback(frame);
  }
  video.addEventListener('ended', () => finishClip());
  video.addEventListener('play', () => {
    userSelected = true;
    if (clip && !pendingSeek && (video.currentTime < clip.start.pts_s || video.currentTime >= clip.end.pts_s)) video.currentTime = clip.start.pts_s;
  });
  video.addEventListener('seeking', () => { seeking = true; });
  video.addEventListener('seeked', () => {
    try {
    if (!clip || pendingSeek || advancing) return;
    const time = video.currentTime;
    if (time >= clip.start.pts_s - 0.025 && time <= clip.end.pts_s + 0.025) { updatePosition(); return; }
    for (const phase of phases) {
      const candidates = clipsFor(phase), part = candidates.findIndex((entry) => entry.url === clip.url && time >= entry.start.pts_s && time < entry.end.pts_s);
      if (part >= 0) { select(phase.id, { preserveTime: true, part }); return; }
    }
    video.pause(); video.currentTime = clip.start.pts_s;
    $('phase-position').textContent = pick('这个时间没有对应阶段，已回到当前片段起点。', 'No phase corresponds to this timestamp; returned to clip start.');
    } finally { seeking = false; }
  });
  video.addEventListener('error', () => {
    if (!clip) return;
    $('phase-empty').hidden = false; $('phase-empty').replaceChildren(node('span', pick('录像无法在浏览器中播放。', 'Video cannot be played in this browser.')));
    const download = node('a', pick('下载原始录像', 'Download raw MP4')); download.href = clip.url; download.download = `${clip.record.entry.camera_id}.mp4`;
    $('phase-empty').append(download); video.hidden = true; $('phase-playing').hidden = true;
    $('phase-position').textContent = pick(`媒体读取或解码失败 · code ${video.error?.code || '未知'}`, `Media read/decode failed · code ${video.error?.code || 'unknown'}`);
  });

  function appendMessages(parent, heading, messages) {
    const section = node('section', undefined, 'phase-context-block'); section.append(node('h3', heading));
    if (!messages?.length) section.append(node('p', pick('没有保存此处的说明或思考记录。', 'No message or reasoning records saved here.'), 'meta'));
    for (const item of messages || []) {
      const text = messageText(item);
      const label = item.type === 'reasoning'
        ? pick('已记录的思考摘要', 'Recorded reasoning summary')
        : item.type === 'userMessage'
          ? pick('用户指令', 'User prompt')
          : pick(`${agentName()} 说明`, `${agentName()} message`);
      section.append(node('p', label, 'phase-text-label'), node('pre', text || pick('此条记录未保存可见文本。', 'No visible text saved for this item.')));
    }
    parent.append(section);
  }
  function renderContext(phase) {
    const signature = JSON.stringify([getLocale(), phase.id, phase.items, phase.before, phase.after, phase.prompt, phase.events.map((event) => event.event_id), phase.actions, [...actionFailures]]);
    if (contextSignature === signature) return; contextSignature = signature;
    const content = $('phase-context'); content.replaceChildren();
    const prompt = messageText(phase.prompt);
    content.append(details(pick('本轮任务 / Prompt', 'Turn Task / Prompt'), prompt || pick('此阶段没有关联到已保存的用户指令。', 'No saved user prompt is correlated with this phase.'), true));
    appendMessages(content, pick(`执行前 · ${agentName()} 的说明与已记录思考`, `Before · ${agentName()} Messages & Reasoning`), phase.before);
    const current = node('section', undefined, 'phase-context-block phase-tool');
    current.append(node('h3', pick(`当前阶段 · ${phase.title}`, `Current Phase · ${phase.title}`)));
    if (phase.kind === 'codex') {
      appendMessages(current, pick(`${agentName()} 已保存的记录`, `Saved ${agentName()} records`), phase.items);
    } else if (phase.item) {
      const item = phase.item;
      current.append(node('p', `${toolName(item)} · ${item.status || pick('状态未提供', 'status unspecified')}`, 'phase-tool-name'));
      const args = item.arguments;
      const instruction = args && typeof args === 'object' && ['language', 'prompt', 'instruction', 'code', 'script'].filter((key) => args[key] !== undefined);
      if (instruction?.length) for (const key of instruction) current.append(details(pick(`提交给工具的 ${key}`, `Tool argument · ${key}`), args[key], true));
      if (item.command) current.append(details(pick('执行命令', 'Command'), item.command, true));
      if (args !== undefined) current.append(details(pick('完整工具参数', 'Full tool arguments'), args, !instruction?.length));
      const text = messageText(item); if (text) current.append(node('pre', text));
      for (const [key, zhTitle, enTitle] of [
        ['result', '工具结果', 'Tool result'],
        ['contentItems', '工具返回内容', 'Tool content items'],
        ['aggregatedOutput', '命令输出', 'Command output'],
        ['error', '错误', 'Error'],
      ]) {
        if (item[key] !== undefined && item[key] !== null) current.append(details(pick(zhTitle, enTitle), item[key]));
      }
      const jump = node('button', pick(`查看这条 ${agentName()} 原始记录`, `Jump to raw ${agentName()} item`)); jump.type = 'button';
      jump.onclick = () => {
        const target = [...document.querySelectorAll('#native-items > li')].find((li) => li.dataset.itemId === item.id);
        if (target) { document.querySelector('.native-panel').open = true; target.querySelector('details').open = true; target.scrollIntoView({ block: 'center', behavior: 'smooth' }); }
      };
      current.append(jump);
    }
    for (const entry of phase.actions || []) {
      const action = entry.action;
      current.append(node('p', `${entry.action_id} · ${action?.receipt?.status || pick('证据读取中或不可用', 'evidence loading or unavailable')}`, 'meta'));
      if (action) current.append(details(pick('Harness 动作请求与执行回执', 'Harness action request & execution receipt'), action));
      const failure = actionFailures.get(entry.action_id);
      if (failure) {
        current.append(node('p', pick(
          `部分证据读取失败：${failure.message}${failure.attempts < 3 ? ' · 正在重试' : ' · 可点击上方“刷新阶段记录”重试'}`,
          `Partial evidence read failed: ${failure.message}${failure.attempts < 3 ? ' · retrying' : ' · click "Refresh Phases" above to retry'}`,
        ), 'meta'));
      }
    }
    const obs3dEvent = phase.events.find((event) => event.type === 'geometry.obs3d_attached');
    if (obs3dEvent?.payload) {
      const p = obs3dEvent.payload;
      const ot = p.object_table || {};
      const s3 = p.scene3d || {};
      const perceptionSection = node('section', undefined, 'phase-context-block phase-perception');
      perceptionSection.append(node('h3', pick('3D 空间感知 · VGGT & SAM 几何估计', '3D Spatial Perception · VGGT & SAM Geometry')));
      const text = pick(
        `检测到 ${ot.rows ?? '未知数量'} 个物体 · 状态: ${ot.status || '未知'} · 耗时: ${p.latency_s ? p.latency_s + 's' : '未知'}` + (s3.points ? ` · 3D 点云: ${Number(s3.points).toLocaleString()} 点` : ''),
        `${ot.rows ?? 'unknown'} objects · status: ${ot.status || 'unknown'} · latency: ${p.latency_s ? p.latency_s + 's' : 'unknown'}` + (s3.points ? ` · 3D cloud: ${Number(s3.points).toLocaleString()} pts` : ''),
      );
      perceptionSection.append(node('p', text, 'meta'));
      if (p.observation_id) {
        const link = node('button', pick('在左侧观察中查看此帧感知叠加图', 'View perception overlay for this frame on the left'));
        link.type = 'button';
        link.onclick = () => {
          const obsId = p.observation_id;
          if (obsId && typeof window.selectObservationById === 'function') {
            window.selectObservationById(obsId, p.camera_id);
          }
        };
        perceptionSection.append(link);
      }
      current.append(perceptionSection);
    }
    const relevantEvents = phase.events.filter((event) => !isNativeAgentEvent(event.type));
    if (relevantEvents.length) {
      const list = node('div', undefined, 'phase-evidence-links');
      for (const event of relevantEvents) {
        const button = node('button', `#${event.server_seq} · ${event.type}`); button.type = 'button';
        button.onclick = () => inspectEvent(event, button); list.append(button);
      }
      const eventDetails = node('details');
      eventDetails.append(node('summary', pick(`本阶段 Harness 事件与证据（${relevantEvents.length}）`, `Harness Events & Evidence in Phase (${relevantEvents.length})`)), list);
      current.append(eventDetails);
    }
    content.append(current);
    appendMessages(content, pick(`执行后 · ${agentName()} 的反馈与下一步说明`, `After · ${agentName()} Feedback & Next Steps`), phase.after);
    content.append(node('p', pick('按同一轮次的记录顺序展示前后文；没有保存的思考内容不作补写。', 'Context is shown in exact recorded order within the same turn; missing reasoning is never synthesized.'), 'meta'));
  }

  async function jumpTerminal(phase, automatic = false) {
    if (!phase) return;
    const id = phase.id;
    $('terminal-phase-label').textContent = pick(`当前：${phase.title}`, `Current: ${phase.title}`);
    $('phase-terminal-jump').disabled = false;
    $('phase-terminal-status').textContent = pick('正在匹配终端原文…', 'Matching terminal output…');
    const synchronized = await terminalReady();
    if (id !== selectedId) return;
    if (!synchronized) {
      $('phase-terminal-status').textContent = pick('终端历史仍在同步；读取完成后再匹配本阶段原文。', 'Terminal history is still syncing; will match once complete.');
      return;
    }
    const values = [phase.item?.command, phase.item?.arguments?.language, phase.item?.arguments?.prompt, messageText(phase.item), ...[...(phase.before || [])].reverse().map(messageText)];
    const normalize = (value) => value.replace(/\s+/g, '');
    const queries = values.filter((value) => typeof value === 'string').flatMap((value) => value.split('\n')).map((value) => normalize(value)).filter((value) => value.length >= 18).map((value) => value.slice(0, 100));
    const buffer = terminal.buffer.active;
    const lines = Array.from({ length: buffer.length }, (_, index) => buffer.getLine(index)?.translateToString(true) || '');
    for (const query of queries) {
      const matches = [];
      for (let index = 0; index < lines.length; index++) {
        const candidate = normalize(lines.slice(index, index + 4).join(''));
        const offset = candidate.indexOf(query);
        if (offset >= 0 && offset < normalize(lines[index]).length) matches.push(index);
      }
      if (matches.length === 1) {
        terminal.scrollToLine(Math.max(0, matches[0] - 2)); terminal.selectLines(matches[0], Math.min(matches[0] + 2, buffer.length - 1));
        $('phase-terminal-status').textContent = pick('已定位唯一匹配的可见原文（文本匹配）。', 'Located unique matching visible text in terminal.');
        autoTerminal = false; return;
      }
    }
    $('phase-terminal-status').textContent = pick('终端未找到唯一对应原文；请查看左侧本阶段的结构化记录。旧日志未保存阶段位置索引。', 'No unique match in terminal output; inspect structured phase records on the left. Legacy logs lack byte-offset indices.');
    if (!automatic) autoTerminal = false;
  }
  $('phase-terminal-jump').onclick = () => jumpTerminal(phases.find((phase) => phase.id === selectedId));
  $('phase-previous').onclick = () => { const index = phases.findIndex((phase) => phase.id === selectedId); if (index > 0) select(phases[index - 1].id); };
  $('phase-next').onclick = () => { const index = phases.findIndex((phase) => phase.id === selectedId); if (index < phases.length - 1) select(phases[index + 1].id); };
  $('phase-play').onclick = () => showClip(0, { play: true });
  $('phase-camera').onchange = () => { const playing = !video.paused; renderNavigation(); if (selectedId) select(selectedId, { play: playing }); };
  $('phase-retry').onclick = () => { for (const id of actionFailures.keys()) actionVersions.delete(id); actionFailures.clear(); schedule(); };
  function refreshLocale() {
    contextSignature = undefined;
    mediaSignature = undefined;
    delete $('phase-camera').dataset.signature;
    if (transitionNotice) transitionNotice = '';
    draw();
    updatePosition();
  }
  onLocaleChange(refreshLocale);
  schedule();
  return {
    receive(event) {
      events.set(event.event_id, event);
      const trimmed = events.size > 5000;
      if (trimmed) events.delete(events.keys().next().value);
      eventCursor = Math.max(eventCursor, event.server_seq || 0); schedule();
      return trimmed;
    },
    replace(value, nativeItems) {
      windowVersion++; events.clear(); actions.clear(); actionVersions.clear(); actionFailures.clear(); observationCache.clear(); pending.clear();
      for (const event of value) events.set(event.event_id, event);
      items = nativeItems; eventCursor = expectedCursor = Math.max(0, ...value.map((event) => event.server_seq || 0));
      selectedId = undefined; userSelected = false; schedule();
    },
    setItems(value, cursor) { items = value; expectedCursor = Math.max(expectedCursor, cursor || 0); schedule(); },
    setObservations(value) {
      observationCache.clear();
      for (const observation of value.slice(-2000)) observationCache.set(observation.observation_id, observation);
    },
    setVideos(value) { records = [...value]; schedule(); },
    terminalUpdated() { if (autoTerminal && selectedId) jumpTerminal(phases.find((phase) => phase.id === selectedId), true); },
    refreshLocale,
  };
}
