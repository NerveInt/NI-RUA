import { pick } from './shared/i18n.js';

export function localURL(reference) {
  if (typeof reference !== 'string' || !reference) return null;
  try { const url = new URL(reference, location.href); return url.origin === location.origin && ['http:', 'https:'].includes(url.protocol) ? url.href : null; }
  catch { return null; }
}

export function cameraVideos(media, reports, currentCameras = {}) {
  const aliases = { cam_head: 'head', head: 'head', head_camera: 'head', camera_head: 'head', cam_left_wrist: 'left', left_wrist: 'left', left_wrist_camera: 'left', cam_right_wrist: 'right', right_wrist: 'right', right_wrist_camera: 'right',
    // SO-101 real arm: one fixed overview camera and one wrist camera (deployment camera ids).
    fixed: 'fixed', fixed_camera: 'fixed', wrist: 'wrist', wrist_camera: 'wrist' };
  const cameraId = (value) => aliases[String(value || '').toLowerCase()];
  const indexed = reports.flatMap((report) => Array.isArray(report.content?.videos) ? report.content.videos : []);
  const candidates = { head: [], left: [], right: [], fixed: [], wrist: [] };
  for (const mediaRecord of media) {
    if (!localURL(mediaRecord.url) || !(mediaRecord.mime_type?.startsWith('video/') || /\.(mp4|webm|mov)(?:\?|$)/i.test(mediaRecord.name || mediaRecord.url))) continue;
    const basename = String(mediaRecord.name || '').split('/').at(-1).replace(/\.[^.]+$/, '');
    const descriptor = indexed.find((video) => video.url === mediaRecord.url || typeof video.video_path === 'string' && mediaRecord.name && (video.video_path === mediaRecord.name || video.video_path.endsWith('/' + mediaRecord.name)));
    const explicit = cameraId(mediaRecord.camera_id) || cameraId(descriptor?.camera_id), camera = explicit || cameraId(basename);
    if (camera) candidates[camera].push({ record: mediaRecord, priority: explicit ? 0 : 1 });
  }
  const selected = {};
  for (const [camera, choices] of Object.entries(candidates)) {
    choices.sort((a, b) => a.priority - b.priority || String(a.record.name).localeCompare(String(b.record.name)) || a.record.url.localeCompare(b.record.url));
    // File inventories can change order or gain unrelated media while playing.
    // Retain the chosen source as long as it is still a valid camera recording.
    const existing = choices.find(({ record }) => record.url === currentCameras[camera]?.url);
    if (choices.length) {
      const chosen = (existing || choices[0]).record;
      selected[camera] = { name: chosen.name, url: chosen.url, mime_type: chosen.mime_type };
    }
  }
  return selected;
}

// Card layout of a run's recordings. The RoboDojo robot has head / left-wrist / right-wrist cameras (default layout,
// null here); the SO-101 has a fixed overview camera and one wrist camera and gets two plain cards.
export function singleArmLayout(cameras = {}) {
  if (cameras.head || cameras.left || cameras.right || !(cameras.fixed || cameras.wrist)) return null;
  return [
    { role: 'fixed', camera: 'fixed', title: pick('固定摄像机 · 主视角', 'Fixed Camera · Main View'), name: pick('固定摄像机', 'Fixed Camera') },
    { role: 'wrist-camera', camera: 'wrist', title: pick('腕部摄像机', 'Wrist Camera'), name: pick('腕部摄像机', 'Wrist Camera') },
  ];
}
