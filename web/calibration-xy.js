import { pick } from './shared/i18n.js';

// Frozen-frame operator annotation. Coordinates always refer to the original RGB.
export function imagePixel(event, image) {
  const r = image.getBoundingClientRect();
  if (!image.naturalWidth || !image.naturalHeight || !r.width || !r.height) return null;
  const u = (event.clientX - r.left) * image.naturalWidth / r.width;
  const v = (event.clientY - r.top) * image.naturalHeight / r.height;
  return u >= 0 && v >= 0 && u < image.naturalWidth && v < image.naturalHeight ? [u, v] : null;
}

// Deployment-provided SI/px/deg thresholds -> display values. No independent UI defaults.
export function xyThresholdFields(defaults = {}) {
  return [
    ['max_xy_error_m', pick('XY 容差（mm）', 'XY tolerance (mm)'), defaults.max_xy_error_m == null ? '' : defaults.max_xy_error_m * 1000, .1],
    ['max_reprojection_error_px', pick('重投影上限（px）', 'Max reprojection (px)'), defaults.max_reprojection_error_px ?? '', .1],
    ['min_parallax_deg', pick('最小视差角（deg）', 'Min parallax (deg)'), defaults.min_parallax_deg ?? '', .1],
  ];
}

export function mountXY({ el, act, blocked, defaults }) {
  let id = null, points = {};
  const capture = el('button', { type: 'button', textContent: pick('冻结当前画面（不移动）', 'Freeze current frames (no motion)') });
  const evaluate = el('button', { type: 'button', textContent: pick('检查本点 XY', 'Evaluate XY at this point') });
  const inputs = {};
  const controls = el('div', { className: 'row' });
  for (const [key, label, value, step] of xyThresholdFields(defaults)) {
    inputs[key] = el('input', { type: 'number', min: step, step, value });
    controls.append(el('label', {}, label, inputs[key]));
  }
  const images = {}, labels = {}, marks = {};
  const frames = el('div', { className: 'frames' });
  const unpickedLabel = (name) => pick(`${name}：尚未选点`, `${name}: no point selected`);
  const coordLabel = (name, point) => pick(`${name}：[${point.map(x => x.toFixed(1)).join(', ')}] px`, `${name}: [${point.map(x => x.toFixed(1)).join(', ')}] px`);
  for (const name of ['fixed', 'wrist']) {
    const image = images[name] = el('img', { alt: pick(`${name} 冻结画面`, `${name} frozen frame`), style: 'width:100%;display:block;cursor:crosshair' });
    const marker = marks[name] = el('span', { textContent: '+', hidden: true,
      style: 'position:absolute;color:#ff2727;font-size:26px;transform:translate(-50%,-50%);pointer-events:none;line-height:1' });
    labels[name] = el('figcaption', { textContent: unpickedLabel(name) });
    image.onclick = (event) => {
      const point = imagePixel(event, image);
      if (!point || !id) return;
      points[name] = point;
      marker.style.left = `${100 * point[0] / image.naturalWidth}%`;
      marker.style.top = `${100 * point[1] / image.naturalHeight}%`;
      marker.hidden = false;
      labels[name].textContent = coordLabel(name, point);
      evaluate.disabled = blocked() || !points.fixed || !points.wrist;
    };
    frames.append(el('figure', {}, el('div', { style: 'position:relative' }, image, marker), labels[name]));
  }
  const results = el('div', { className: 'meta', style: 'white-space:pre-line' });
  capture.onclick = () => act(pick('冻结 XY 检查画面', 'Freeze XY frames'), '/v1/calibration/steps/S6/start', { params: { action: 'xy_capture' } });
  evaluate.onclick = () => act(pick('检查 XY 一致性', 'Evaluate XY consistency'), '/v1/calibration/steps/S6/start', { params: {
    action: 'xy_evaluate', snapshot_id: id, fixed_pixel: points.fixed, wrist_pixel: points.wrist,
    max_xy_error_m: Number(inputs.max_xy_error_m.value) / 1000,
    max_reprojection_error_px: Number(inputs.max_reprojection_error_px.value),
    min_parallax_deg: Number(inputs.min_parallax_deg.value),
  } });
  const node = el('div', { className: 'card' }, el('h2', { textContent: pick('XY 相机—指尖一致性检查', 'XY Camera–Fingertip Consistency Check') }),
    el('p', { textContent: pick('任务结束、机械臂静止后冻结画面。在两图中点击同一个固定夹指内侧最前端中心；遮挡或看不清时不要提交。红色 + 是你的选点。',
      'Freeze frames while the arm is stationary. Click the exact inner front tip center of the fixed jaw in both images; do not submit if occluded or unclear. Red + marks your click.') }),
    el('p', { className: 'meta', textContent: pick('在工作区不同位置、高度和姿态重复检查。本工具不移动机械臂、不改标定。单点通过不代表整台机械臂准确；人工选点和共享标定误差尚未量化。',
      'Repeat across different positions, heights, and poses. This tool neither moves the arm nor alters calibration. Passing at one point does not prove full-workspace accuracy; manual click and shared calibration errors remain unquantified.') }),
    controls, el('div', { className: 'row' }, capture, evaluate), frames, results);
  return { node, update(run) {
    const next = run?.xy_capture?.snapshot_id;
    if (next && next !== id) {
      id = next; points = {};
      for (const name of ['fixed', 'wrist']) {
        images[name].src = `/v1/calibration/xy/${id}/${name}.jpg`;
        marks[name].hidden = true; labels[name].textContent = unpickedLabel(name);
      }
    }
    capture.disabled = blocked(); evaluate.disabled = blocked() || !points.fixed || !points.wrist;
    results.textContent = (run?.xy_verifications || []).map((r, i) => {
      const delta = r.observed_minus_predicted_m?.map(v => (v * 1000).toFixed(2)).join(', ');
      const verdict = r.status === 'passed'
        ? pick('本点条件通过', 'Passed at this point')
        : r.status === 'failed'
          ? pick('本点超差', 'Out of tolerance')
          : pick('无法判定', 'Inconclusive');
      return `${i + 1}. ${verdict}${delta ? ` · ΔXYZ [${delta}] mm · XY ${(r.xy_error_m * 1000).toFixed(2)} mm` : ` · ${r.reason || ''}`}`;
    }).join('\n') || pick('尚无 XY 结果；触桌高度验证与 XY 检查分别记录。', 'No XY results yet; table-height verification and XY checks are recorded separately.');
  } };
}
