import './perception-view.css';
import {
  perceptionObjects,
  matchesPerception,
  formatPerceptionSummary,
  formatObjectCentre,
  formatObjectDimensions,
  formatObjectRotation,
  formatQualityBadge,
  formatDeltaSummary,
  formatRemoteStatusSummary,
} from './perception-model.js';
import { onLocaleChange, pick, registerPageI18n } from './shared/i18n.js';
import { COMMON_TEXT } from './i18n/common.js';

const VIEWER_HTML = `
<div class="panel-bar">
            <h1 id="camera-heading">机械臂 · 实时观察</h1>
            <div class="camera-view-mode-tabs" role="group" aria-label="画面显示模式">
              <button id="view-mode-raw" type="button" class="view-mode-tab active" title="原始 RGB 相机画面">原始画面</button>
              <button id="view-mode-overlay" type="button" class="view-mode-tab" title="VGGT + SAM 感知检测与分割轮廓叠加">感知叠加</button>
              <button id="view-mode-split" type="button" class="view-mode-tab" title="左右并排对比原图与感知检测图">并排对比</button>
            </div>
            <select id="camera-select" aria-label="选择相机"></select>
            <button id="return-live" hidden>返回实时</button>
          </div>
          <div class="perception-view-options">
            <label class="toggle-label"><input id="toggle-bboxes" type="checkbox" checked> 交互标签框</label>
            <span id="perception-quick-badge" class="meta"></span>
          </div>
          <div id="camera-stage">
            <div id="camera-stage-split" class="camera-stage-split" hidden>
              <div class="split-pane">
                <span class="split-title">原始画面 (Raw)</span>
                <img id="camera-image-split" alt="原始画面">
              </div>
              <div class="split-pane">
                <span class="split-title">感知检测 (VGGT + SAM)</span>
                <img id="perception-image-split" alt="感知叠加画面">
              </div>
            </div>
            <div id="camera-stage-single" class="camera-stage-single">
              <img id="camera-image" alt="尚无已发布的机器人画面" hidden>
              <img id="perception-image" alt="VGGT + SAM 感知叠加画面" hidden>
              <svg id="camera-svg-overlay" class="camera-svg" viewBox="0 0 640 480" preserveAspectRatio="xMidYMid meet" hidden></svg>
            </div>
            <p id="perception-image-note" hidden></p>
            <p id="camera-empty">等待已发布的机器人观察</p>
          </div>
          <div id="frame-meta" class="meta">观察与媒体将按 ID 关联</div>
`;
const DETAILS_HTML = `
<details id="perception-panel" open class="perception-panel">
            <summary>3D 空间感知 · VGGT & SAM 分割与几何估计</summary>
            <div class="perception-body">
              <p class="meta">VGGT 单目几何估计依赖桌面尺度和标定；并非独立深度测量。SAM 框与轮廓仅对应标注的相机和观察。</p>
              <div id="perception-meta-bar" class="perception-meta-bar">
                <span id="perception-status-pill" class="perception-pill">未检测到感知数据</span>
                <span id="perception-timing-info" class="meta-item"></span>
                <span id="perception-plane-info" class="meta-item"></span>
                <span id="perception-cloud-info" class="meta-item"></span>
              </div>
              <div id="perception-delta-banner" class="perception-delta" hidden></div>
              <div class="perception-table-wrapper">
                <table id="perception-objects-table" class="perception-table">
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>SAM 类别标签</th>
                      <th>3D 中心 [x, y, z] (m)</th>
                      <th>尺寸 (长×宽×高 mm)</th>
                      <th>偏角 (Yaw / Tilt)</th>
                      <th>支撑关系</th>
                      <th>质量与标志</th>
                    </tr>
                  </thead>
                  <tbody id="perception-objects-body">
                    <tr><td colspan="7" class="empty">暂无当前观察的检测物体</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
          </details>
`;
const SERVICE_HTML = `
<button id="perception-service-btn" class="perception-btn" type="button" aria-expanded="false" title="VGGT & SAM 远端模型服务状态"><span class="status-dot"></span><span id="perception-service-label">VGGT+SAM 检查中</span></button><dialog id="perception-modal" class="perception-modal" aria-label="VGGT 与 SAM 远端视觉模型服务详情">
      <div class="modal-header">
        <h2>VGGT & SAM · 远端视觉模型服务</h2>
        <button id="close-perception-modal" type="button" class="close-btn" aria-label="关闭">×</button>
      </div>
      <div class="modal-body">
        <div class="modal-grid">
          <div class="service-card">
            <h3>VGGT-Omega · 3D 单目深度与几何</h3>
            <div id="vggt-details" class="card-details">正在读取...</div>
          </div>
          <div class="service-card">
            <h3>SAM · 实例分割后端</h3>
            <div id="sam-details" class="card-details">正在读取...</div>
          </div>
        </div>
        <div class="modal-section">
          <h3>宿主机与连接状态</h3>
          <div id="remote-conn-details" class="card-details">正在读取...</div>
        </div>
      </div>
    </dialog>
`;

/** Mount the read-only saved-observation viewer once in a page. */
export function mountPerceptionView({ getJSON, apiURL = (path) => path }) {
  const $ = (id) => document.getElementById(id);
  const viewerHost = $('perception-viewer-host');
  const detailsHost = $('perception-details-host');
  const serviceHost = $('perception-service-host');
  viewerHost.innerHTML = VIEWER_HTML;
  detailsHost.innerHTML = DETAILS_HTML;
  serviceHost.innerHTML = SERVICE_HTML;
  const perceptionDict = { text: COMMON_TEXT };
  registerPageI18n(perceptionDict, viewerHost);
  registerPageI18n(perceptionDict, detailsHost);
  registerPageI18n(perceptionDict, serviceHost);

  let displayedObservation = null;
  const artifactURL = (reference) => typeof reference === 'string' && reference.startsWith('artifact:')
    ? apiURL(`/v1/artifacts/${encodeURIComponent(reference.slice(9))}`) : null;
  let currentPerception = null;
  let perceptionLoadFailed = false;
  let remoteLoadFailed = false;
  let viewMode = 'raw';
  let showBBoxes = true;
  let remoteStatus = null;
  let perceptionFetchVersion = 0;

  function safeArtifactURL(reference) {
    if (!reference || typeof reference !== 'string') return null;
    const id = reference.startsWith('artifact:') ? reference.slice('artifact:'.length) : reference;
    return apiURL(`/v1/artifacts/${encodeURIComponent(id)}`);
  }

  function setViewMode(mode) {
    viewMode = mode;
    ['raw', 'overlay', 'split'].forEach((m) => {
      const btn = $(`view-mode-${m}`);
      if (btn) btn.classList.toggle('active', m === mode);
    });
    const single = $('camera-stage-single');
    const split = $('camera-stage-split');
    if (single && split) {
      single.hidden = mode === 'split';
      split.hidden = mode !== 'split';
    }
    showCamera();
  }

  const rawBtn = $('view-mode-raw');
  if (rawBtn) rawBtn.onclick = () => setViewMode('raw');
  const overlayBtn = $('view-mode-overlay');
  if (overlayBtn) overlayBtn.onclick = () => setViewMode('overlay');
  const splitBtn = $('view-mode-split');
  if (splitBtn) splitBtn.onclick = () => setViewMode('split');
  const bboxCheckbox = $('toggle-bboxes');
  if (bboxCheckbox) {
    bboxCheckbox.onchange = (e) => {
      showBBoxes = e.target.checked;
      renderSvgOverlay();
    };
  }

  function highlightObject(targetId) {
    document.querySelectorAll('#perception-objects-body tr.interactive-row').forEach((tr) => {
      tr.classList.toggle('highlighted', tr.dataset.objId === targetId);
    });
    document.querySelectorAll('#camera-svg-overlay g').forEach((g) => {
      g.classList.toggle('highlighted', g.dataset.objId === targetId);
    });
  }

  function renderSvgOverlay() {
    const svg = $('camera-svg-overlay');
    if (!svg) return;
    svg.replaceChildren();

    const img = viewMode === 'overlay' && !$('perception-image').hidden ? $('perception-image') : $('camera-image');
    if (!matchesPerception(currentPerception, displayedObservation?.observation_id, $('camera-select').value) || !showBBoxes || viewMode === 'split' || !currentPerception?.objects?.length || !img || img.hidden || !img.naturalWidth) {
      svg.setAttribute('hidden', '');
      return;
    }

    svg.setAttribute('viewBox', `0 0 ${img.naturalWidth} ${img.naturalHeight}`);
    svg.removeAttribute('hidden');
    const bounds = img.getBoundingClientRect(), parent = svg.parentElement.getBoundingClientRect();
    const scale = Math.min(bounds.width / img.naturalWidth, bounds.height / img.naturalHeight);
    const width = img.naturalWidth * scale, height = img.naturalHeight * scale;
    Object.assign(svg.style, { left: `${bounds.left - parent.left + (bounds.width - width) / 2}px`,
      top: `${bounds.top - parent.top + (bounds.height - height) / 2}px`, width: `${width}px`, height: `${height}px` });

    currentPerception.objects.forEach((obj, idx) => {
      if (!obj.bbox_px || obj.bbox_px.length !== 4) return;
      const [x0, y0, x1, y1] = obj.bbox_px;
      const w = x1 - x0;
      const h = y1 - y0;
      const id = obj.id || `o${idx + 1}`;

      const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      g.dataset.objId = id;

      const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      rect.setAttribute('class', 'bbox-rect');
      rect.setAttribute('x', String(x0));
      rect.setAttribute('y', String(y0));
      rect.setAttribute('width', String(Math.max(w, 2)));
      rect.setAttribute('height', String(Math.max(h, 2)));
      g.append(rect);
      if (Array.isArray(obj.contour_px) && obj.contour_px.length > 2 && obj.contour_px.every((point) => Array.isArray(point) && point.length === 2 && point.every(Number.isFinite))) {
        const contour = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        contour.setAttribute('points', obj.contour_px.map((point) => point.join(',')).join(' '));
        contour.setAttribute('class', 'bbox-rect');
        g.append(contour);
      }

      const labelText = `${id}: ${obj.label || 'obj'}${obj.label_score !== null && obj.label_score !== undefined ? ' ' + Math.round(obj.label_score * 100) + '%' : ''}`;
      const textWidth = Math.max(labelText.length * 8, 44);
      const labelBg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      labelBg.setAttribute('class', 'bbox-label-bg');
      labelBg.setAttribute('x', String(x0));
      labelBg.setAttribute('y', String(Math.max(0, y0 - 18)));
      labelBg.setAttribute('width', String(textWidth));
      labelBg.setAttribute('height', '18');
      labelBg.setAttribute('rx', '3');
      g.append(labelBg);

      const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      text.setAttribute('class', 'bbox-text');
      text.setAttribute('x', String(x0 + 4));
      text.setAttribute('y', String(Math.max(13, y0 - 5)));
      text.textContent = labelText;
      g.append(text);

      if (obj.label_px && obj.label_px.length === 2) {
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('class', 'bbox-point');
        circle.setAttribute('cx', String(obj.label_px[0]));
        circle.setAttribute('cy', String(obj.label_px[1]));
        circle.setAttribute('r', '3');
        g.append(circle);
      }

      g.onmouseenter = () => highlightObject(id);
      g.onmouseleave = () => highlightObject(null);

      svg.append(g);
    });
  }

  function renderPerceptionData(perception) {
    const quickBadge = $('perception-quick-badge');
    const statusPill = $('perception-status-pill');
    const timingInfo = $('perception-timing-info');
    const planeInfo = $('perception-plane-info');
    const cloudInfo = $('perception-cloud-info');
    const deltaBanner = $('perception-delta-banner');
    const tbody = $('perception-objects-body');

    if (!perception) {
      if (quickBadge) quickBadge.textContent = pick('暂无 3D 感知数据', 'No 3D perception data');
      if (statusPill) {
        statusPill.textContent = perceptionLoadFailed
          ? pick('感知记录未就绪或读取失败', 'Perception record not ready or failed to load')
          : pick('未运行外部感知', 'External perception not run');
        statusPill.className = 'perception-pill';
      }
      if (timingInfo) timingInfo.textContent = '';
      if (planeInfo) planeInfo.textContent = '';
      if (cloudInfo) cloudInfo.textContent = '';
      if (deltaBanner) deltaBanner.hidden = true;
      if (tbody) {
        const td = document.createElement('td');
        td.colSpan = 7;
        td.className = 'empty';
        td.textContent = pick('当前观察未运行外部 3D 感知或服务未开启', 'External 3D perception was not run or is disabled for this observation');
        const tr = document.createElement('tr');
        tr.append(td);
        tbody.replaceChildren(tr);
      }
      renderSvgOverlay();
      return;
    }

    const objects = perceptionObjects(perception);

    if (quickBadge) {
      quickBadge.textContent = formatPerceptionSummary(perception);
    }

    if (statusPill) {
      const components = perception.diagnostics || {};
      const labels = [
        ['object_table', pick('物体表', 'object_table')],
        ['scene3d', pick('点云', 'scene3d')],
      ].map(([key, label]) => {
        const component = components[key];
        return `${label}: ${component?.status || pick('未记录', 'unrecorded')}${component?.reason ? ' (' + component.reason + ')' : ''}`;
      });
      statusPill.textContent = `${perception.camera_id || pick('相机未知', 'unknown camera')} · ${pick('几何估计', 'geometry')} · ${labels.join(' · ')}`;
      statusPill.className = 'perception-pill';
    }

    if (timingInfo) {
      const t = perception.service_timing || perception.details?.service_timing || {};
      const parts = [];
      if (t.vggt_s !== undefined && t.vggt_s !== null) parts.push(`VGGT ${Math.round(t.vggt_s * 1000)}ms`);
      if (t.sam_s !== undefined && t.sam_s !== null) parts.push(`SAM ${Math.round(t.sam_s * 1000)}ms`);
      if (t.text_s !== undefined && t.text_s !== null) parts.push(`Text ${Math.round(t.text_s * 1000)}ms`);
      if (t.total_s !== undefined && t.total_s !== null) parts.push(pick(`总计 ${Math.round(t.total_s * 1000)}ms`, `Total ${Math.round(t.total_s * 1000)}ms`));
      else if (perception.latency_s) parts.push(pick(`总计 ${Math.round(perception.latency_s * 1000)}ms`, `Total ${Math.round(perception.latency_s * 1000)}ms`));
      timingInfo.textContent = parts.length ? `⏱ ${parts.join(' | ')}` : '';
    }

    if (planeInfo) {
      const z = perception.table?.table_z ?? perception.details?.support_fit?.table_z_m;
      planeInfo.textContent = z !== undefined && z !== null ? pick(`基准桌面: z=${z}m`, `Table plane: z=${z}m`) : '';
    }

    if (cloudInfo) {
      const pts = perception.scene3d?.points;
      cloudInfo.textContent = pts ? pick(`3D 点云: ${Number(pts).toLocaleString()} 点`, `3D cloud: ${Number(pts).toLocaleString()} pts`) : '';
    }

    if (deltaBanner) {
      const text = formatDeltaSummary(perception.delta);
      if (text) {
        deltaBanner.textContent = text;
        deltaBanner.hidden = false;
      } else {
        deltaBanner.hidden = true;
      }
    }

    if (tbody) {
      tbody.replaceChildren();
      if (objects.length > 0) {
        objects.forEach((obj, idx) => {
          const tr = document.createElement('tr');
          tr.className = 'interactive-row';
          tr.dataset.objId = obj.id || `o${idx + 1}`;

          const tdId = document.createElement('td');
          tdId.className = 'obj-id';
          tdId.textContent = obj.id || `o${idx + 1}`;

          const tdLabel = document.createElement('td');
          if (obj.label) {
            const badge = document.createElement('span');
            badge.className = 'label-badge';
            badge.textContent = obj.label;
            tdLabel.append(badge);
            if (obj.label_score !== undefined && obj.label_score !== null) {
              const score = document.createElement('span');
              score.className = 'score-badge';
              score.textContent = `${Math.round(obj.label_score * 100)}%`;
              tdLabel.append(score);
            }
          } else {
            tdLabel.textContent = pick('通用物体 (auto)', 'Generic object (auto)');
          }

          const tdCentre = document.createElement('td');
          tdCentre.className = 'mono';
          tdCentre.textContent = formatObjectCentre(obj.centre_m);

          const tdDim = document.createElement('td');
          tdDim.className = 'mono';
          tdDim.textContent = formatObjectDimensions(obj);

          const tdRot = document.createElement('td');
          tdRot.textContent = formatObjectRotation(obj);

          const tdSupp = document.createElement('td');
          tdSupp.textContent = obj.support === 'table'
            ? pick('桌面支撑', 'Table')
            : (obj.support ? pick(`叠放于 ${obj.support}`, `On ${obj.support}`) : pick('未知', 'Unknown'));

          const tdQuality = document.createElement('td');
          const badgeInfo = formatQualityBadge(obj);
          const qBadge = document.createElement('span');
          qBadge.className = `quality-badge ${badgeInfo.ok ? 'ok' : 'low'}`;
          qBadge.textContent = badgeInfo.text;
          tdQuality.append(qBadge);

          tr.append(tdId, tdLabel, tdCentre, tdDim, tdRot, tdSupp, tdQuality);
          tr.onmouseenter = () => highlightObject(obj.id);
          tr.onmouseleave = () => highlightObject(null);
          tbody.append(tr);
        });
      } else {
        const td = document.createElement('td');
        td.colSpan = 7;
        td.className = 'empty';
        td.textContent = pick('当前记录没有物体行；请查看上方感知状态', 'No object rows in current record; check perception status above');
        const tr = document.createElement('tr');
        tr.append(td);
        tbody.append(tr);
      }
    }

    renderSvgOverlay();
  }

  async function syncObservationPerception(observation) {
    displayedObservation = observation;
    const token = ++perceptionFetchVersion;
    if (currentPerception?.observation_id !== observation?.observation_id) {
      currentPerception = null;
      perceptionLoadFailed = false;
      renderPerceptionData(null);
    }
    if (!observation?.observation_id) return;
    try {
      const data = await getJSON(`/v1/observations/${encodeURIComponent(observation.observation_id)}/perception`);
      if (token !== perceptionFetchVersion || displayedObservation?.observation_id !== observation.observation_id) return;
      currentPerception = data.observation_id === observation.observation_id ? data : null;
      perceptionLoadFailed = false;
      renderPerceptionData(currentPerception);
      showCamera();
    } catch {
      if (token !== perceptionFetchVersion) return;
      currentPerception = null;
      perceptionLoadFailed = true;
      renderPerceptionData(null);
      showCamera();
    }
  }

  let rawImageFailed = false;
  function showCamera() {
    const observation = displayedObservation;
    const camera = observation?.cameras?.find((item) => item.camera_id === $('camera-select').value);
    const rawUrl = artifactURL(camera?.rgb_ref);
    const image = $('camera-image'), perceptionImage = $('perception-image');
    const rawSplit = $('camera-image-split'), perceptionSplit = $('perception-image-split');
    const empty = $('camera-empty');
    const matches = matchesPerception(currentPerception, observation?.observation_id, camera?.camera_id);
    const overlayUrl = matches ? safeArtifactURL(currentPerception.overlay_ref) : null;
    const setImage = (img, url) => {
      const src = url ? url + '?view=image' : null;
      if (src !== img.getAttribute('src')) {
        if (img === image) rawImageFailed = false;
        img.hidden = true;
        if (src) img.setAttribute('src', src); else img.removeAttribute('src');
      }
    };
    const update = () => {
      const rawReady = Boolean(rawUrl && image.complete && image.naturalWidth && !rawImageFailed);
      const overlayReady = Boolean(overlayUrl && perceptionImage.complete && perceptionImage.naturalWidth);
      image.hidden = !rawReady || viewMode === 'split' || (viewMode === 'overlay' && overlayReady);
      perceptionImage.hidden = !overlayReady || viewMode !== 'overlay';
      rawSplit.hidden = !(rawUrl && rawSplit.complete && rawSplit.naturalWidth);
      perceptionSplit.hidden = !(overlayUrl && perceptionSplit.complete && perceptionSplit.naturalWidth);
      empty.hidden = Boolean(rawUrl) && !rawImageFailed;
      empty.textContent = rawImageFailed
        ? pick('媒体读取失败 · 未沿用上一观察画面', 'Failed to load media · previous frame was not reused')
        : observation
          ? pick('该观察没有可访问的 RGB 媒体', 'No accessible RGB media for this observation')
          : pick('等待机器人发布观察', 'Waiting for published robot observation');
      const note = $('perception-image-note');
      note.hidden = viewMode === 'raw' || overlayReady;
      note.textContent = !matches
        ? pick('当前相机没有对应感知叠加图', 'No perception overlay for the selected camera')
        : !overlayUrl
          ? pick('此观察未保存叠加图', 'No overlay image saved for this observation')
          : pick('叠加图加载中或读取失败', 'Overlay loading or failed to read');
      renderSvgOverlay();
    };
    for (const img of [image, perceptionImage, rawSplit, perceptionSplit]) {
      img.onload = update;
      img.onerror = () => { if (img === image) rawImageFailed = true; update(); };
    }
    setImage(image, rawUrl); setImage(rawSplit, rawUrl);
    setImage(perceptionImage, overlayUrl); setImage(perceptionSplit, overlayUrl);
    image.alt = camera ? `${camera.camera_id} · ${observation.observation_id} · ${camera.frame_id}` : pick('无画面', 'No frame');
    update();
    $('frame-meta').textContent = observation
      ? `${observation.observation_id} · generation ${observation.env_generation} · ${observation.step_domain || pick('步域未提供', 'step domain unspecified')} ${observation.sim_step ?? pick('未知', 'unknown')} · ${camera?.frame_id || pick('画面缺失', 'missing frame')}${camera?.image_reconstruction ? pick(' · 视频重建（有损）', ' · video-reconstructed (lossy)') : ''}`
      : pick('观察与媒体将按 ID 关联', 'Observations and media are linked by exact ID');
  }
  new ResizeObserver(renderSvgOverlay).observe($('camera-stage-single'));

  function paintRemoteBadge() {
    const btn = $('perception-service-btn');
    const label = $('perception-service-label');
    if (remoteLoadFailed) {
      if (btn) btn.className = 'perception-btn offline';
      if (label) label.textContent = pick('感知: 连接失败', 'Perception: Unreachable');
      renderRemoteModalContent(null);
      return;
    }
    const summary = formatRemoteStatusSummary(remoteStatus);
    if (btn) btn.className = `perception-btn ${summary.state}`;
    if (label) label.textContent = summary.label;
    renderRemoteModalContent(remoteStatus);
  }

  async function updateRemoteStatus() {
    try {
      const data = await getJSON('/v1/remote/status');
      remoteStatus = data;
      remoteLoadFailed = false;
      paintRemoteBadge();
    } catch {
      remoteStatus = null;
      remoteLoadFailed = true;
      paintRemoteBadge();
    }
  }

  function renderRemoteModalContent(data) {
    const vggt = data?.services?.vggt, health = vggt?.health || {};
    const render = (id, rows) => {
      const target = $(id);
      if (!target) return;
      target.replaceChildren(...rows.map(([label, value]) => {
        const row = document.createElement('div'); row.className = 'metric-row';
        const key = document.createElement('span'); key.className = 'metric-label'; key.textContent = label;
        const text = document.createElement('span'); text.className = 'metric-value'; text.textContent = value ?? pick('未记录', 'Not recorded');
        row.append(key, text); return row;
      }));
    };
    render('vggt-details', [
      [pick('状态来源', 'Status source'), data?.archive ? pick('历史快照，当前状态未知', 'Archived snapshot; live health unknown') : pick('当前健康检查', 'Live health check')],
      [pick('VGGT 状态', 'VGGT status'), vggt?.available === true ? pick('可用', 'Available') : vggt?.available === false ? pick('不可用', 'Unavailable') : pick('未知', 'Unknown')],
      [pick('原因', 'Reason'), vggt?.reason],
      [pick('模型版本', 'Model version'), health.version],
      [pick('加速设备', 'Device'), health.device],
      [pick('启动时间', 'Started at'), health.started],
    ]);
    render('sam-details', [
      [pick('实例分割后端', 'Segmentation backend'), health.object_table_backend],
      [pick('SAM 版本', 'SAM version'), health.sam31],
      [pick('健康状态', 'Health note'), pick('未独立探测；以上为 VGGT 服务报告的后端信息', 'Not probed independently; backend metadata reported by VGGT service')],
    ]);
    render('remote-conn-details', [
      [pick('远端主机', 'Remote host'), data?.host],
      [pick('隧道状态', 'Tunnel state'), data?.tunnel?.state],
      [pick('端口转发', 'Port forwards'), data?.tunnel?.forwards?.join(pick('；', '; '))],
      [pick('服务进程 PID', 'Service PID'), vggt?.remote_pid ?? health.pid],
    ]);
  }

  const modal = $('perception-modal');
  const openBtn = $('perception-service-btn');
  const closeBtn = $('close-perception-modal');
  if (openBtn && modal) {
    openBtn.onclick = () => {
      updateRemoteStatus();
      if (typeof modal.showModal === 'function') modal.showModal();
      else modal.setAttribute('open', '');
    };
  }
  if (closeBtn && modal) {
    closeBtn.onclick = () => {
      if (typeof modal.close === 'function') modal.close();
      else modal.removeAttribute('open');
    };
  }
  updateRemoteStatus();
  setInterval(updateRemoteStatus, 6000);
  $('camera-select').onchange = showCamera;

  function refreshLocale() {
    renderPerceptionData(currentPerception);
    showCamera();
    paintRemoteBadge();
  }
  onLocaleChange(refreshLocale);
  return { showCamera, syncObservationPerception, setViewMode, refreshLocale };
}
