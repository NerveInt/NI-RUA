/**
 * Pure functions for formatting and projecting VGGT & SAM perception results and remote telemetry.
 */
import { pick } from './shared/i18n.js';

const finite = (value) => typeof value === 'number' && Number.isFinite(value);

export function perceptionObjects(perception) {
  if (perception?.objects?.length) return perception.objects;
  const columns = perception?.table?.columns || [];
  return (perception?.table?.rows || []).map((row) => Object.fromEntries(columns.map((key, index) => [key, row[index]])));
}

export function matchesPerception(perception, observationId, cameraId) {
  return Boolean(perception && perception.observation_id === observationId && perception.camera_id === cameraId);
}

export function formatPerceptionSummary(perception) {
  if (!perception) return pick('暂无 3D 感知数据', 'No 3D perception data');
  if (!perception.table && !perception.objects?.length) {
    return perception.scene3d
      ? pick('已保存 VGGT 点云 · 无物体表', 'VGGT point cloud saved · no object table')
      : pick('当前观察无可用感知结果', 'No perception results for current observation');
  }
  const count = perceptionObjects(perception).length;
  const scale = perception.scale !== undefined && perception.scale !== null
    ? pick(` · VGGT 尺度 ${perception.scale}`, ` · VGGT scale ${perception.scale}`)
    : '';
  const tableZ = perception.table?.table_z !== undefined && perception.table?.table_z !== null
    ? pick(` · 桌面 z=${perception.table.table_z}m`, ` · table z=${perception.table.table_z}m`)
    : '';
  return pick(`已检测 ${count} 个物体${scale}${tableZ}`, `${count} ${count === 1 ? 'object' : 'objects'} detected${scale}${tableZ}`);
}

export function formatObjectCentre(centre) {
  if (!Array.isArray(centre) || centre.length < 3 || !centre.slice(0, 3).every(finite)) return '-';
  return `[${centre.slice(0, 3).map((v) => Number(v).toFixed(3)).join(', ')}]`;
}

export function formatObjectDimensions(obj) {
  if (!obj) return '-';
  const l = finite(obj.length_m) ? Math.round(Number(obj.length_m) * 1000) : '-';
  const w = finite(obj.width_m) ? Math.round(Number(obj.width_m) * 1000) : '-';
  const h = finite(obj.height_m)
    ? Math.round(Number(obj.height_m) * 1000)
    : (finite(obj.top_z_m) && finite(obj.bottom_z_m)
      ? Math.round((Number(obj.top_z_m) - Number(obj.bottom_z_m)) * 1000)
      : '-');
  return `${l} × ${w} × ${h}`;
}

export function formatObjectRotation(obj) {
  if (!obj) return '-';
  const yaw = finite(obj.yaw_deg) ? `${Math.round(Number(obj.yaw_deg))}°` : '-';
  const tilt = finite(obj.axis_tilt_deg) ? `${Math.round(Number(obj.axis_tilt_deg))}°` : '-';
  return `${yaw} / ${tilt}`;
}

export function formatQualityBadge(obj) {
  if (!obj) return { text: pick('未知', 'Unknown'), ok: false };
  const flags = Array.isArray(obj.flags) ? obj.flags.filter(Boolean) : [];
  const isOk = obj.quality === 'ok' && flags.length === 0;
  if (isOk) return { text: pick('正常', 'OK'), ok: true };
  const flagText = flags.map((f) => {
    switch (f) {
      case 'floating': return pick('悬空', 'floating');
      case 'floating_rescaled_to_table': return pick('悬空校准', 'rescaled to table');
      case 'low_depth_confidence': return pick('低置信度', 'low depth confidence');
      case 'contact_mismatch': return pick('接触间隙过大', 'contact gap');
      case 'height_above_max': return pick('高度超限', 'height above max');
      default: return f;
    }
  }).join(', ');
  return { text: flagText || obj.quality || pick('未知', 'Unknown'), ok: false };
}

export function formatDeltaSummary(delta) {
  if (!delta || typeof delta !== 'object') return null;
  const parts = (delta.moved || []).map((item) => {
    const vector = item.delta_m;
    const displacement = Array.isArray(vector) && vector.length === 3 && vector.every(finite)
      ? vector.map((v) => { const mm = Math.round(v * 1000); return `${mm >= 0 ? '+' : ''}${mm}`; }).join(', ')
      : pick('未知', 'unknown');
    return `${item.before_id || '?'} → ${item.after_id || '?'}: Δ=(${displacement}) mm`;
  });
  if (delta.vacated?.length) parts.push(pick(`原位置不再匹配 ${delta.vacated.length} 个`, `vacated ${delta.vacated.length}`));
  if (delta.unmatched_under_arm?.length) parts.push(pick(`原位置被手臂遮挡且未匹配 ${delta.unmatched_under_arm.length} 个`, `occluded under arm ${delta.unmatched_under_arm.length}`));
  if (delta.appeared?.length) parts.push(pick(`新匹配 ${delta.appeared.length} 个`, `newly matched ${delta.appeared.length}`));
  if (finite(delta.unchanged_count)) parts.push(pick(`未变 ${delta.unchanged_count} 个`, `unchanged ${delta.unchanged_count}`));
  return parts.length ? `${pick('动作前后位移估计', 'Motion delta estimate')} · ${parts.join(' | ')}` : null;
}

export function formatRemoteStatusSummary(status) {
  if (status?.archive) {
    return {
      label: pick('感知: 历史服务记录', 'Perception: Archived'),
      state: 'warning',
      text: pick('保存时的状态，当前健康状态未知', 'Recorded state at archive time; live health unknown'),
    };
  }
  if (!status || !status.enabled) {
    return {
      label: pick('感知: 未启用', 'Perception: Disabled'),
      state: 'offline',
      text: pick('未配置或未启用', 'Not configured or disabled'),
    };
  }
  const vggt = status.services?.vggt;
  if (vggt?.available) {
    const backend = vggt.health?.object_table_backend;
    return {
      label: pick(`VGGT 在线${backend ? ' · 后端 ' + backend : ''}`, `VGGT Online${backend ? ' · ' + backend : ''}`),
      state: 'online',
      text: pick('VGGT 健康检查通过；SAM 状态以返回的后端信息为准', 'VGGT health check passed; SAM status follows reported backend metadata'),
    };
  }
  if (status.state === 'starting') {
    return {
      label: pick('VGGT+SAM 启动中', 'VGGT+SAM Starting'),
      state: 'warning',
      text: pick('正在拉起服务', 'Starting service'),
    };
  }
  return {
    label: pick('VGGT+SAM 离线', 'VGGT+SAM Offline'),
    state: 'offline',
    text: vggt?.reason || pick('服务不可用', 'Service unavailable'),
  };
}
