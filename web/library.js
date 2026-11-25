// 永久实验库主入口（/library.html）：支持实验组矩阵对比、全部运行检索、详情回放及中英双语就地切换。
import { renderTaskReview } from './task-review.js';
import { taskLabel, environmentLabel, runtimeDescription } from './task-labels.js';
import { setupLibraryScroll } from './library-scroll.js';
import { metrics } from './experiment-metrics.js';
import { cameraVideos, localURL, singleArmLayout } from './experiment-media.js';
import { setupComparisons } from './comparison-view.js';
import { runTimeText } from './experiment-time.js';
import { byId as $ } from './shared/dom.js';
import { requestJSON } from './shared/api-client.js';
import { getLocale, initLocale, isEn, localeTag, mountLanguageSwitch, onLocaleChange, pick, registerPageI18n } from './shared/i18n.js';
import { LIBRARY_I18N, translateLibraryServerTextEn } from './i18n/library.js';

initLocale();
registerPageI18n(LIBRARY_I18N, document);
mountLanguageSwitch(document.querySelector('header'), { className: 'lang-switch' });

const labelsZh = {
  categories: { main: '主实验', benchmark: '基准评测', acceptance: '验收验证', imported: '导入记录', native: 'NI-RUA 持续会话', harness: 'NI-RUA 持续会话', robot_validation: '机器人验证', validation: '机器人验证', reference: '参考策略', roboprobe: '参考策略', geometry: '几何与局部场景', scene: '几何与局部场景', synthetic: '合成夹具' },
  statuses: { new: '新建', starting: '启动中', running: '进行中', idle: '等待任务', waiting: '等待中', stopping: '正在停止', stopped: '已停止', completed: '已完成', succeeded: '已完成', failed: '失败', cancelled: '已取消', replay: '历史记录', recorded: '已归档', imported: '已导入', disconnected: '连接中断', unknown: '状态未知', unavailable: '暂不可用', stale: '记录已过期' },
};
const labelsEn = {
  categories: { main: 'Main Experiment', benchmark: 'Benchmark', acceptance: 'Acceptance', imported: 'Imported', native: 'Persistent Session', harness: 'Persistent Session', robot_validation: 'Robot Validation', validation: 'Robot Validation', reference: 'Reference Policy', roboprobe: 'Reference Policy', geometry: 'Geometry & Scene', scene: 'Geometry & Scene', synthetic: 'Synthetic Fixture' },
  statuses: { new: 'New', starting: 'Starting', running: 'Running', idle: 'Idle', waiting: 'Waiting', stopping: 'Stopping', stopped: 'Stopped', completed: 'Completed', succeeded: 'Completed', failed: 'Failed', cancelled: 'Cancelled', replay: 'Archived', recorded: 'Recorded', imported: 'Imported', disconnected: 'Disconnected', unknown: 'Unknown', unavailable: 'Unavailable', stale: 'Stale' },
};
const label = (group, value) => (isEn() ? labelsEn : labelsZh)[group]?.[value] || value || pick('未提供', 'Unspecified');
const rows = new Map(), reportNodes = new Map(), mediaNodes = new Map();
let selectedId, detailVersion = 0, offset = 0, limit = 20, total = 0, appliedSearch = new URLSearchParams();
let refreshing = false, refreshAgain = false, refreshFailed = false, refreshTimer, lastRefresh, libraryRevision, reportSignature, mediaSignature;
let lastFacets = null, lastExperiments = [], currentDetailRecord = null;
let selectedWrist = 'left', currentCameras = {};
const filterNames = ['q', 'category', 'status', 'task', 'tag', 'date_from', 'date_to', 'sort'];

function renderContext(node, record) {
  const environment = environmentLabel(record), task = taskLabel(record.task), description = runtimeDescription(record);
  const signature = JSON.stringify([getLocale(), environment, task, record.task, description]);
  if (node.dataset.signature === signature) return;
  node.dataset.signature = signature;
  const environmentTag = document.createElement('span'); environmentTag.className = 'environment-label'; environmentTag.textContent = environment; if (description) environmentTag.title = description;
  const taskTag = document.createElement('span'); taskTag.className = 'task-label'; taskTag.textContent = task; taskTag.title = record.task || pick('任务未注明', 'Task unspecified');
  node.replaceChildren(environmentTag, taskTag);
}

function link(text, reference) {
  const url = localURL(reference); if (!url) return null;
  const node = document.createElement('a'); node.textContent = text; node.href = url; return node;
}
const read = (path) => requestJSON(path);
function saveURL() {
  const query = new URLSearchParams(appliedSearch);
  const current = new URLSearchParams(location.search);
  for (const name of ['lang', 'view', 'scope', 'group', 'compare', 'reference', 'case', 'camera']) if (current.has(name)) query.set(name, current.get(name));
  if (offset) query.set('offset', offset);
  if (limit !== 20) query.set('limit', limit);
  if (selectedId) query.set('selected', selectedId);
  history.replaceState(null, '', `${location.pathname}${query.size ? '?' + query : ''}`);
  sessionStorage.setItem('ni-rua-library-url', location.pathname + location.search);
}
function restoreURL() {
  const query = new URLSearchParams(location.search);
  for (const name of filterNames) {
    const value = query.get(name) ?? (name === 'sort' ? 'active' : '');
    if ($(name).tagName === 'SELECT' && value && !Array.from($(name).options).some((option) => option.value === value)) {
      const option = document.createElement('option'); option.value = value; option.textContent = label(name === 'category' ? 'categories' : 'statuses', value); $(name).append(option);
    }
    $(name).value = value; if (value) appliedSearch.set(name, value);
  }
  offset = Math.max(0, Number.parseInt(query.get('offset'), 10) || 0);
  limit = [20, 50, 100].includes(Number(query.get('limit'))) ? Number(query.get('limit')) : 20;
  $('limit').value = limit; selectedId = query.get('selected') || undefined;
}
function applyFilters() {
  if (!$('library-filters').reportValidity()) return;
  if ($('date_from').value && $('date_to').value && $('date_from').value > $('date_to').value) {
    $('library-errors').hidden = false; $('library-errors').textContent = pick('开始日期不能晚于结束日期。', 'Start date cannot be later than end date.'); return;
  }
  appliedSearch = new URLSearchParams();
  for (const name of filterNames) if ($(name).value.trim()) appliedSearch.set(name, $(name).value.trim());
  offset = 0; saveURL(); refreshLibrary();
}
function facets(values) {
  lastFacets = values;
  for (const [id, key, zhEmpty, enEmpty] of [['category', 'categories', '全部分类', 'All categories'], ['status', 'statuses', '全部状态', 'All statuses']]) {
    const select = $(id), previous = select.value;
    if (document.activeElement === select) continue;
    const choices = [...new Set([...(values?.[key] || []), ...(previous ? [previous] : [])])];
    const sig = `${getLocale()}:${JSON.stringify(choices)}`;
    if (sig === select.dataset.choices) continue;
    select.dataset.choices = sig;
    const all = document.createElement('option'); all.value = ''; all.textContent = pick(zhEmpty, enEmpty);
    select.replaceChildren(all, ...choices.map((value) => { const option = document.createElement('option'); option.value = value; option.textContent = label(key, value); return option; }));
    select.value = previous;
  }
  for (const [id, key] of [['task-choices', 'tasks'], ['tag-choices', 'tags']]) {
    const choices = values?.[key] || [];
    const sig = `${getLocale()}:${JSON.stringify(choices)}`;
    if ($(id).dataset.choices === sig) continue;
    $(id).dataset.choices = sig;
    $(id).replaceChildren(...choices.map((value) => { const option = document.createElement('option'); option.value = value; if (key === 'tasks') option.label = taskLabel(value); return option; }));
  }
}
function renderRows(experiments) {
  lastExperiments = experiments;
  const visible = new Set(experiments.map((record) => record.id));
  for (const [id, row] of rows) if (!visible.has(id)) { row.item.remove(); rows.delete(id); }
  experiments.forEach((record, index) => {
    let row = rows.get(record.id);
    if (!row) {
      const item = document.createElement('li'); item.className = 'experiment-row'; item.dataset.experimentId = record.id;
      const title = document.createElement('button'); title.type = 'button'; title.className = 'experiment-title'; title.onclick = () => selectExperiment(record.id, true);
      const status = document.createElement('div'); status.className = 'experiment-status';
      const context = document.createElement('div'); context.className = 'experiment-context';
      const meta = document.createElement('p'); meta.className = 'meta'; const result = document.createElement('p'); result.className = 'experiment-metrics';
      const tags = document.createElement('div'); tags.className = 'experiment-tags';
      item.append(title, context, status, meta, result, tags); row = { item, title, context, status, meta, result, tags }; rows.set(record.id, row);
    }
    row.context.setAttribute('aria-label', pick('环境与任务', 'Environment & task'));
    row.item.classList.toggle('selected', record.id === selectedId); row.title.setAttribute('aria-pressed', String(record.id === selectedId));
    const signature = `${getLocale()}:${JSON.stringify(record)}`;
    if (row.signature !== signature) {
      row.signature = signature; row.title.textContent = record.title || record.task || record.run_id || record.id;
      renderContext(row.context, record);
      const chip = document.createElement('span'); chip.className = `status-chip ${record.live ? 'live' : ['failed', 'unknown'].includes(record.status) ? record.status : ''}`;
      chip.textContent = `${label('statuses', record.status)}${record.live ? pick(' · 实时', ' · Live') : ''}`;
      const category = document.createElement('span'); category.textContent = label('categories', record.category); row.status.replaceChildren(chip, category);
      row.meta.textContent = `${record.task || pick('任务未提供', 'Task unspecified')} · ${record.run_id || record.id}\n${runTimeText(record)} ${pick('（北京时间 UTC+8）', '(UTC+8)')}`;
      row.result.textContent = metrics(record.evaluator_available === false ? { ...record.summary, evaluator_available: false } : record.summary);
      const recordTags = record.tags || [];
      const tagSig = `${getLocale()}:${JSON.stringify(recordTags)}`;
      if (row.tags.dataset.signature !== tagSig) {
        row.tags.dataset.signature = tagSig;
        row.tags.replaceChildren(...recordTags.map((value) => {
          const button = document.createElement('button'); button.type = 'button'; button.textContent = value;
          button.setAttribute('aria-label', pick(`筛选标签 ${value}`, `Filter by tag ${value}`));
          button.onclick = () => { $('tag').value = value; applyFilters(); }; return button;
        }));
      }
    }
    const current = $('experiment-list').children[index]; if (current !== row.item) $('experiment-list').insertBefore(row.item, current || null);
  });
  $('empty-library').hidden = experiments.length > 0;
}
function paintPaginationCounts() {
  $('result-count').textContent = pick(
    `${total} 个实验${total ? ` · ${offset + 1}–${Math.min(total, offset + limit)}` : ''}`,
    `${total} runs${total ? ` · ${offset + 1}–${Math.min(total, offset + limit)}` : ''}`,
  );
  $('page-count').textContent = pick(
    `第 ${Math.floor(offset / limit) + 1} / ${Math.max(1, Math.ceil(total / limit))} 页`,
    `Page ${Math.floor(offset / limit) + 1} / ${Math.max(1, Math.ceil(total / limit))}`,
  );
}
async function refreshLibrary() {
  if (document.hidden) return;
  if (refreshing) { refreshAgain = true; return; } refreshing = true;
  const search = new URLSearchParams(appliedSearch); search.set('offset', offset); search.set('limit', limit); const requestSignature = search.toString();
  try {
    if (comparisons.isActive()) {
      const data = await comparisons.refresh();
      refreshFailed = !data;
      if (data) showConnection(data);
      return;
    }
    const data = await read(`/v1/experiments?${search}`);
    const current = new URLSearchParams(appliedSearch); current.set('offset', offset); current.set('limit', limit);
    if (current.toString() !== requestSignature) { refreshAgain = true; return; }
    total = data.total;
    if (offset >= total && offset > 0) { offset = Math.max(0, Math.ceil(total / limit) - 1) * limit; saveURL(); refreshAgain = true; return; }
    facets(data.facets); renderRows(data.experiments || []);
    paintPaginationCounts();
    $('previous-page').disabled = offset === 0; $('next-page').disabled = offset + limit >= total;
    showConnection(data);
    $('library-errors').hidden = !(data.errors?.length);
    $('library-errors').textContent = data.errors?.length
      ? pick(
        `${data.errors.length} 个目录读取异常：${data.errors.map((error) => typeof error === 'string' ? error : error.error || error.message || JSON.stringify(error)).join('；')}`,
        `${data.errors.length} directory read error(s): ${data.errors.map((error) => typeof error === 'string' ? error : error.error || error.message || JSON.stringify(error)).join('; ')}`,
      )
      : '';
    if (selectedId) selectExperiment(selectedId);
  } catch (error) {
    refreshFailed = true;
    const savedAt = lastRefresh ? lastRefresh.toLocaleTimeString(localeTag(), { hour12: false }) : '';
    $('library-connection').textContent = pick(
      `实验库暂不可用 · ${error.message}${savedAt ? ` · 保留 ${savedAt} 的记录` : ''}`,
      `Library unavailable · ${error.message}${savedAt ? ` · retaining snapshot from ${savedAt}` : ''}`,
    );
  }
  finally { refreshing = false; if (refreshAgain) { refreshAgain = false; scheduleRefresh(); } }
}
function showConnection(data) {
  refreshFailed = false;
  if (data) {
    libraryRevision = data.revision;
    lastRefresh = data.observed_at ? new Date(data.observed_at) : new Date();
  }
  if (lastRefresh) {
    $('library-connection').textContent = pick(
      `自动更新 · ${lastRefresh.toLocaleTimeString(localeTag(), { hour12: false })}`,
      `Auto-updating · ${lastRefresh.toLocaleTimeString(localeTag(), { hour12: false })}`,
    );
  }
}
function scheduleRefresh() { clearTimeout(refreshTimer); refreshTimer = setTimeout(refreshLibrary, 250); }
function clearDetailAssets() {
  for (const entry of mediaNodes.values()) entry.node.querySelector('video')?.pause();
  reportNodes.clear(); mediaNodes.clear(); currentCameras = {}; reportSignature = undefined; mediaSignature = undefined;
  $('detail-reports').replaceChildren(); $('detail-media').replaceChildren();
}
function paintExperimentDetail(record) {
  if (!record) return;
  $('detail-placeholder').hidden = true; $('detail-content').hidden = false;
  $('detail-title').textContent = record.title || record.task || record.run_id || record.id;
  renderContext($('detail-context'), record);
  const modeText = record.live
    ? (record.view_mode === 'live_recording' ? pick('实时记录 · 只读', 'Live recording · read-only') : pick('实时运行', 'Live run'))
    : pick('历史记录 · 只读', 'Archived · read-only');
  $('detail-state').textContent = `${label('categories', record.category)} · ${label('statuses', record.status)} · ${modeText}`;
  $('detail-meta').textContent = `${record.task || pick('任务未提供', 'Task unspecified')} · ${record.run_id || record.id}\n${runTimeText(record)}\n${pick('北京时间（UTC+8）', 'Beijing Time (UTC+8)')}`;
  $('detail-metrics').textContent = metrics(record.summary, record.task_review);
  renderTaskReview($('detail-task-review'), record.task_review);
  $('detail-capabilities').textContent = record.replay_available
    ? `${record.terminal_available ? pick('保留原生终端、观察与事件', 'Native terminal, observations, and events retained') : pick('无原生终端记录；查看已保存观察与事件', 'No native terminal log; saved observations and events available')}${record.live ? pick('；切换历史记录不影响运行', '; inspecting history does not affect the live run') : pick('；输入与运行修改已关闭', '; input and run mutations disabled')}`
    : pick('没有可回放的原生协议日志；可查看保存的报告与摄像机录像。', 'No replayable native protocol log; saved reports and camera videos remain available.');
  const links = [];
  if (record.replay_available || record.live && record.view_mode === 'live') {
    const linkLabel = record.live
      ? (record.view_mode === 'live_recording' ? pick('查看实时记录', 'View Live Recording') : pick('打开直播工作台', 'Open Live Workbench'))
      : pick('打开只读回放', 'Open Read-Only Replay');
    const node = link(linkLabel, record.view_url);
    if (node) links.push(node);
  }
  const evidence = link(pick('查看证据清单', 'View Evidence Manifest'), record.evidence_url); if (evidence) links.push(evidence);
  if (record.trajectory_download?.status === 'available') {
    const download = link(pick('下载原始轨迹', 'Download Raw Trajectory'), record.trajectory_download.url);
    if (download) { download.download = `${record.id}-trajectory.zip`; links.push(download); }
    const help = document.createElement('span'); help.className = 'meta';
    help.textContent = pick('可用离线重放工具选择分辨率生成视频（docs/development/hd-replay.md）', 'Render custom-resolution video with offline HD Replay CLI (docs/development/hd-replay.md)');
    links.push(help);
  } else if (record.trajectory_download?.reason) {
    const unavailable = document.createElement('span'); unavailable.className = 'meta';
    const reason = isEn() ? translateLibraryServerTextEn(record.trajectory_download.reason) : record.trajectory_download.reason;
    unavailable.textContent = pick(`轨迹下载暂不可用：${reason}`, `Trajectory download unavailable: ${reason}`);
    links.push(unavailable);
  }
  const linkSignature = JSON.stringify([getLocale(), links.map((node) => [node.textContent, node.href])]);
  if ($('detail-links').dataset.signature !== linkSignature) { $('detail-links').dataset.signature = linkSignature; $('detail-links').replaceChildren(...links); }
  $('detail-raw').querySelector('pre').textContent = JSON.stringify({ id: record.id, run_id: record.run_id, kind: record.kind, tags: record.tags, summary: record.summary, metadata: record.metadata, runtime: record.runtime, manifest: record.manifest }, null, 2);
  const reports = record.reports || [], media = cameraVideos(record.media || [], reports, currentCameras);
  const nextReports = `${getLocale()}:${JSON.stringify(reports)}`, nextMedia = `${getLocale()}:${JSON.stringify(media)}`;
  if (nextReports !== reportSignature) { reportSignature = nextReports; renderReports(reports); }
  if (nextMedia !== mediaSignature) { mediaSignature = nextMedia; renderMedia(media); }
}
async function selectExperiment(id, userSelected = false) {
  const changed = selectedId !== id; selectedId = id; $('close-detail').hidden = false;
  if (changed) {
    currentDetailRecord = null;
    clearDetailAssets();
    $('detail-title').textContent = pick('正在读取实验', 'Loading run');
    $('detail-content').hidden = true;
    $('detail-placeholder').hidden = false;
    $('detail-placeholder').textContent = pick('正在读取状态与保存的证据', 'Loading run state and saved evidence');
  }
  saveURL(); for (const [key, row] of rows) { row.item.classList.toggle('selected', key === id); row.title.setAttribute('aria-pressed', String(key === id)); }
  const version = ++detailVersion;
  try {
    const record = await read(`/v1/experiments/${encodeURIComponent(id)}`); if (version !== detailVersion || id !== selectedId) return;
    currentDetailRecord = record;
    paintExperimentDetail(record);
    if (userSelected && innerWidth <= 760) $('experiment-detail').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (error) {
    if (version !== detailVersion) return;
    $('detail-placeholder').hidden = false;
    $('detail-placeholder').textContent = pick(`实验详情暂不可用 · ${error.message}`, `Run details unavailable · ${error.message}`);
  }
}
function syncDetails(id, nodes, records, signatureFor, build) {
  const keys = new Set(records.map((record) => JSON.stringify([record.name, record.url])));
  for (const [key, entry] of nodes) if (!keys.has(key)) { entry.node.querySelector('video')?.pause(); entry.node.remove(); nodes.delete(key); }
  records.forEach((record, index) => {
    const key = JSON.stringify([record.name, record.url]), signature = signatureFor(record); let entry = nodes.get(key);
    if (!entry || entry.signature !== signature) {
      const node = build(record); node.open = entry?.node.open || false;
      entry?.node.querySelector('video')?.pause(); entry?.node.replaceWith(node); entry = { node, signature }; nodes.set(key, entry);
    }
    const current = $(id).children[index]; if (current !== entry.node) $(id).insertBefore(entry.node, current || null);
  });
}
function renderReports(reports) {
  syncDetails('detail-reports', reportNodes, reports, (report) => `${getLocale()}:${JSON.stringify(report)}`, (report) => {
    const details = document.createElement('details'), summary = document.createElement('summary');
    summary.textContent = `${pick('报告', 'Report')} · ${report.name || pick('未命名', ' unnamed')}`;
    details.append(summary);
    const download = link(pick('打开原始报告', 'Open Raw Report'), report.url); if (download) { download.target = '_blank'; download.rel = 'noopener'; details.append(download); }
    if (report.content !== undefined && report.content !== null) { const pre = document.createElement('pre'); pre.textContent = typeof report.content === 'string' ? report.content : JSON.stringify(report.content, null, 2); details.append(pre); }
    return details;
  });
}

function cameraCard(role, plainTitle) {
  const node = document.createElement('article'); node.className = 'camera-card'; node.dataset.cameraRole = role;
  const heading = document.createElement('div'); heading.className = 'camera-heading';
  const title = document.createElement('h3');
  title.textContent = plainTitle || (role === 'head' ? pick('头部摄像机 · 主视角', 'Head Camera · Main View') : pick('辅助视角', 'Auxiliary View'));
  heading.append(title);
  let select;
  if (role === 'wrist') {
    select = document.createElement('select'); select.setAttribute('aria-label', pick('辅助摄像机', 'Auxiliary camera'));
    for (const [value, zhText, enText] of [['left', '左腕摄像机', 'Left Wrist Camera'], ['right', '右腕摄像机', 'Right Wrist Camera']]) {
      const option = document.createElement('option'); option.value = value; option.textContent = pick(zhText, enText); select.append(option);
    }
    select.onchange = () => { selectedWrist = select.value; renderMedia(currentCameras); }; heading.append(select);
  }
  const content = document.createElement('div'); content.className = 'camera-content';
  const status = document.createElement('p'); status.className = 'detail-media-status'; status.setAttribute('role', 'status');
  const download = document.createElement('a'); download.className = 'camera-download'; download.textContent = pick('下载录像', 'Download MP4'); download.hidden = true;
  node.append(heading, content, status, download);
  return { node, content, status, download, select };
}
function renderCamera(entry, record, name) {
  const source = localURL(record?.url);
  entry.download.hidden = !source;
  if (source) { entry.download.href = source; entry.download.download = String(record.name || '').split('/').at(-1); }
  const sourceKey = `${getLocale()}:${source || ''}`;
  if (entry.source === sourceKey) return;
  entry.source = sourceKey; entry.content.querySelector('video')?.pause(); entry.content.replaceChildren();
  if (!source) {
    const empty = document.createElement('p'); empty.className = 'camera-empty'; empty.textContent = pick(`暂无${name}录像`, `No ${name} recording yet`);
    entry.content.append(empty); entry.status.textContent = pick('录像保存后会自动显示。', 'Recording appears automatically once saved.'); return;
  }
  const player = document.createElement('video'); player.controls = true; player.playsInline = true; player.preload = 'metadata';
  player.setAttribute('aria-label', pick(`${name}录像`, `${name} recording`));
  entry.status.textContent = pick('正在读取录像信息', 'Loading video metadata');
  player.onloadedmetadata = () => {
    const dur = Number.isFinite(player.duration) ? player.duration.toFixed(2) : pick('未知', 'unknown');
    entry.status.textContent = `${player.videoWidth}×${player.videoHeight} · ${pick(`时长 ${dur} 秒`, `duration ${dur} s`)}`;
  };
  player.onerror = () => { entry.status.textContent = pick('录像无法播放或浏览器不支持此格式，可下载原文件查看。', 'Video cannot be played in this browser; download the raw file instead.'); };
  player.src = source; entry.content.append(player);
}
function renderMedia(cameras) {
  currentCameras = cameras;
  const single = singleArmLayout(cameras);
  const wanted = new Set(single ? single.map((card) => card.role) : ['head', 'wrist']);
  for (const [role, entry] of mediaNodes) if (!wanted.has(role)) { entry.node.querySelector('video')?.pause(); entry.node.remove(); mediaNodes.delete(role); }
  if (single) {
    for (const card of single) {
      let entry = mediaNodes.get(card.role);
      if (!entry) { entry = cameraCard(card.role, card.title); mediaNodes.set(card.role, entry); $('detail-media').append(entry.node); }
      renderCamera(entry, cameras[card.camera], card.name);
    }
    return;
  }
  if (!cameras[selectedWrist] && cameras[selectedWrist === 'left' ? 'right' : 'left']) selectedWrist = selectedWrist === 'left' ? 'right' : 'left';
  for (const role of ['head', 'wrist']) {
    let entry = mediaNodes.get(role);
    if (!entry) { entry = cameraCard(role); mediaNodes.set(role, entry); $('detail-media').append(entry.node); }
    if (entry.select) {
      for (const option of entry.select.options) option.disabled = !cameras[option.value];
      entry.select.value = selectedWrist; entry.select.disabled = !cameras.left && !cameras.right;
    }
    const camera = role === 'head' ? 'head' : selectedWrist;
    const camName = isEn()
      ? { head: 'Head Camera', left: 'Left Wrist Camera', right: 'Right Wrist Camera' }[camera]
      : { head: '头部摄像机', left: '左腕摄像机', right: '右腕摄像机' }[camera];
    renderCamera(entry, cameras[camera], camName);
  }
}
let refreshingMedia = false;
async function refreshSelectedMedia() {
  if (document.hidden || comparisons.isActive() || !selectedId || refreshingMedia || $('detail-content').hidden) return;
  const id = selectedId, version = detailVersion; refreshingMedia = true;
  try {
    const record = await read(`/v1/experiments/${encodeURIComponent(id)}/media`);
    if (document.hidden || comparisons.isActive() || id !== selectedId || version !== detailVersion) return;
    const cameras = cameraVideos(record.media || [], record.reports || [], currentCameras), signature = `${getLocale()}:${JSON.stringify(cameras)}`;
    if (signature !== mediaSignature) { mediaSignature = signature; renderMedia(cameras); }
  } catch { /* 保留当前媒体并等待下次刷新重试 */ }
  finally { refreshingMedia = false; }
}
$('library-filters').onsubmit = (event) => { event.preventDefault(); applyFilters(); };
$('reset-filters').onclick = () => { $('library-filters').reset(); applyFilters(); };
$('refresh-library').onclick = refreshLibrary;
$('previous-page').onclick = () => { offset = Math.max(0, offset - limit); saveURL(); refreshLibrary(); };
$('next-page').onclick = () => { if (offset + limit >= total) return; offset += limit; saveURL(); refreshLibrary(); };
$('limit').onchange = () => { limit = Number($('limit').value); offset = 0; saveURL(); refreshLibrary(); };
$('close-detail').onclick = () => {
  selectedId = undefined; currentDetailRecord = null; detailVersion++; clearDetailAssets(); saveURL(); $('detail-content').hidden = true;
  $('detail-title').textContent = pick('选择一个实验', 'Select a Run'); $('detail-placeholder').hidden = false;
  $('detail-placeholder').textContent = pick(
    '从左侧选择实验，直接播放头部主视角与腕部辅助视角录像，并查看评测结果。',
    'Select a run on the left to play synchronized main and wrist camera recordings and inspect evaluation evidence.',
  );
  $('close-detail').hidden = true;
  for (const row of rows.values()) { row.item.classList.remove('selected'); row.title.setAttribute('aria-pressed', 'false'); }
};
const comparisons = setupComparisons({ read, openRun: (id) => selectExperiment(id, true), onViewChange: refreshLibrary });

onLocaleChange(() => {
  comparisons.refreshLocale();
  if (lastFacets) facets(lastFacets);
  if (lastExperiments.length) renderRows(lastExperiments);
  if (total || lastRefresh) paintPaginationCounts();
  showConnection(null);
  if (currentDetailRecord && selectedId === currentDetailRecord.id) {
    for (const entry of mediaNodes.values()) { entry.node.querySelector('video')?.pause(); entry.node.remove(); }
    mediaNodes.clear(); reportSignature = undefined; mediaSignature = undefined;
    paintExperimentDetail(currentDetailRecord);
  }
});

setupLibraryScroll();
restoreURL(); refreshLibrary();
const events = new EventSource('/v1/experiments/events');
function eventRefresh(event) {
  if (document.hidden) return;
  let revision;
  try { revision = JSON.parse(event.data).revision; } catch { /* 无版本号通知直接复检 */ }
  clearTimeout(refreshTimer);
  refreshTimer = setTimeout(() => { if (revision === undefined || revision !== libraryRevision) refreshLibrary(); }, 250);
}
events.addEventListener('snapshot', eventRefresh); events.addEventListener('changed', eventRefresh); events.onmessage = eventRefresh;
events.onerror = () => { $('library-connection').textContent = pick('更新通知连接中断 · 每 5 秒核对实验库', 'Event stream disconnected · polling library every 5 s'); };
setInterval(() => { if (!document.hidden && (refreshFailed || events.readyState !== EventSource.OPEN)) refreshLibrary(); }, 5000);
setInterval(refreshSelectedMedia, 15000);
document.addEventListener('visibilitychange', () => { if (!document.hidden) refreshLibrary(); });
window.addEventListener('pagehide', () => events.close());
