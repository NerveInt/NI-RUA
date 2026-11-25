import { getLocale, isEn, pick } from './shared/i18n.js';
import { translateFailureReasonEn, translateTaskReviewTextEn } from './i18n/workbench.js';

const node = (tag, text, className) => {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
};
const number = (value) => (typeof value === 'number' && Number.isFinite(value) ? String(Number(value.toFixed(6))) : pick('未记录', 'Not recorded'));
const labels = {
  scored: '已评测',
  unscored: '未完成正式评测',
  unknown: '评测状态未确认',
  partial: '仅保存部分评测结果',
  pending: '等待正式评测',
  unavailable: '未保存可用评测结果',
};
const labelsEn = {
  scored: 'Formally Evaluated',
  unscored: 'Not Formally Scored',
  unknown: 'Evaluation Status Unconfirmed',
  partial: 'Partial Evaluation Snapshot',
  pending: 'Awaiting Evaluation',
  unavailable: 'No Usable Evaluation Saved',
};
const trServer = (value) => (isEn() ? translateTaskReviewTextEn(value) : value);

function disclosure(title, value) {
  const details = node('details');
  details.append(node('summary', title), node('pre', typeof value === 'string' ? value : JSON.stringify(value, null, 2)));
  return details;
}
function section(title) {
  const result = node('section', undefined, 'task-review-section'); result.append(node('h3', title)); return result;
}
function sourceLink(source) {
  if (typeof source.url !== 'string') return null;
  try {
    const url = new URL(source.url, location.href);
    if (url.origin !== location.origin || !['http:', 'https:'].includes(url.protocol)) return null;
    const rawLabel = source.label || source.path || pick('查看原始证据', 'View raw evidence');
    const link = node('a', trServer(rawLabel)); link.href = url.href; link.target = '_blank'; link.rel = 'noopener'; return link;
  } catch { return null; }
}
export function renderTaskReview(container, review, message) {
  const signature = JSON.stringify([getLocale(), review, message]);
  if (container.dataset.signature === signature) return;
  container.dataset.signature = signature;
  container.replaceChildren(node('h2', pick('任务要求与评分', 'Task Requirements & Scoring')));
  if (!review) {
    container.append(node('p', message || pick('此运行尚未提供可核对的任务与评分记录。', 'No verifiable task and scoring record is available for this run yet.'), 'meta'));
    return;
  }
  const en = isEn();
  container.dataset.taskId = review.task_id || '';
  const description = section(pick('官方任务描述', 'Official Task Instruction'));
  description.append(node('p', review.description?.text || pick('未保存官方任务说明。', 'No official task instruction saved.'), 'task-description'));
  const kind = (en
    ? {
      recorded_instruction: 'Official task instruction recorded in this run',
      official_template: 'Official task instruction template',
      recorded_observation: 'Official task instruction recorded in this run',
      installed_task_source: 'Official task instruction template from installed source',
    }
    : {
      recorded_instruction: '本次运行保存的官方任务指令',
      official_template: '官方任务指令模板',
      recorded_observation: '本次运行保存的官方任务指令',
      installed_task_source: '当前安装版本的官方任务指令模板',
    })[review.description?.source_kind];
  if (kind) description.append(node('p', kind, 'meta'));
  container.append(description);

  const scoring = review.scoring || {}, criteria = section(pick('每一个得分点', 'Scoring Criteria Breakdown'));
  if (scoring.rule_source?.status !== 'matched' && scoring.rule_source?.message) criteria.append(node('p', trServer(scoring.rule_source.message), 'score-evidence-note'));
  if (scoring.aggregation) criteria.append(node('p', trServer(scoring.aggregation), 'task-aggregation'));
  const list = node('ol', undefined, 'score-criteria');
  for (const point of scoring.criteria || []) {
    const row = node('li'); row.dataset.criterionId = point.id || '';
    const heading = node('div', undefined, 'score-criterion-heading');
    heading.append(node('strong', trServer(point.label) || point.id || pick('评分项', 'Criterion')));
    if (point.score_label || typeof point.score === 'number') {
      heading.append(node('span', point.score_label ? trServer(point.score_label) : pick(`${number(point.score)} 分`, `${number(point.score)} pts`), 'score-value'));
    }
    row.append(heading);
    if (point.description) row.append(node('p', trServer(point.description)));
    if (point.variant) row.append(node('p', pick(`适用：${point.variant}`, `Variant: ${trServer(point.variant)}`), 'meta'));
    row.append(node('p', point.recorded_result == null
      ? pick('本次命中情况：未保存此得分点的单独判定。', 'Recorded verdict: per-criterion verdict was not persisted.')
      : pick(`本次记录：${typeof point.recorded_result === 'object' ? JSON.stringify(point.recorded_result) : point.recorded_result}`, `Recorded: ${typeof point.recorded_result === 'object' ? JSON.stringify(point.recorded_result) : point.recorded_result}`), 'criterion-result'));
    list.append(row);
  }
  if (list.children.length) criteria.append(list);
  else criteria.append(node('p', pick('没有可核对的分项评分规则。', 'No per-item scoring rules available to verify.'), 'meta'));
  if (scoring.completion_conditions?.length) {
    criteria.append(node('h4', pick('任务成功条件', 'Task Completion Conditions')));
    const conditions = node('ul'); for (const condition of scoring.completion_conditions) conditions.append(node('li', trServer(condition))); criteria.append(conditions);
  }
  container.append(criteria);

  const result = review.result || {}, actual = section(pick('我们的分数 · 如何得到', 'Recorded Score & Derivation'));
  actual.dataset.resultStatus = result.status || 'unknown';
  const summary = node('div', undefined, 'task-score-summary');
  const statusMap = en ? labelsEn : labels;
  summary.append(node('strong', result.evaluator_available === false
    ? pick('无官方评分器 · 不计分', 'No Official Evaluator · Unscored')
    : statusMap[result.status] || result.status || pick('评测状态未确认', 'Evaluation Status Unconfirmed')));
  const snapshotLabel = ['unscored', 'partial'].includes(result.status) ? pick(' · 局部记录', ' · partial snapshot') : '';
  const values = [
    [pick('归一化得分（0–1）', 'Normalized score (0–1)'), result.status === 'unscored' ? (result.task_success === false ? pick('不可用', 'N/A') : pick('不计分', 'Unscored')) : number(result.normalized_score)],
    [pick(`原生过程分（0–${scoring.max_score ?? 100}）${snapshotLabel}`, `Native process score (0–${scoring.max_score ?? 100})${snapshotLabel}`), scoring.mode === 'binary_success' && result.native_process_score == null ? pick('无独立过程分', 'No separate process score') : number(result.native_process_score)],
    [pick(`原生奖励（native_reward）${snapshotLabel}`, `Native reward (native_reward)${snapshotLabel}`), number(result.native_reward)],
    [pick('任务成功', 'Task success'), result.task_success === true ? pick('是', 'Yes') : result.task_success === false ? pick('否', 'No') : pick('未确认', 'Unconfirmed')],
  ];
  if (result.failure_reason?.label) values.push([pick('失败原因', 'Failure reason'), en ? translateFailureReasonEn(result.failure_reason) : result.failure_reason.label]);
  const dl = node('dl'); for (const [key, value] of values) dl.append(node('dt', key), node('dd', value)); summary.append(dl); actual.append(summary);
  for (const explanation of result.explanation || []) actual.append(node('p', trServer(explanation), 'score-explanation'));
  if (result.breakdown_status === 'not_recorded') {
    actual.append(node('p', pick('归档只保存了汇总值，没有保存各评分项的命中明细；不能用总分反推哪些条件达成。', 'Archive saved only aggregate totals without per-criterion verdicts; individual predicate completion cannot be inferred from the total score.'), 'score-evidence-note'));
  }
  if (result.partial_native_result != null) actual.append(disclosure(pick('部分评测记录', 'Partial evaluation record'), result.partial_native_result));
  container.append(actual);

  const source = scoring.rule_source || {}, provenance = section(pick('规则与证据来源', 'Rule & Evidence Provenance'));
  if (source.message) provenance.append(node('p', trServer(source.message), source.status === 'matched' ? 'meta' : 'score-evidence-note'));
  const links = node('div', undefined, 'task-review-sources');
  for (const record of review.sources || []) { const link = sourceLink(record); if (link) links.append(link); }
  if (links.children.length) provenance.append(links);
  const code = scoring.source_code || [];
  if (code.length) {
    const details = node('details'); details.append(node('summary', pick('核对官方评分函数与规则版本', 'Inspect official reward functions & rule version')));
    for (const entry of code) { details.append(node('p', `${entry.title || entry.path}${entry.line ? pick(` · 第 ${entry.line} 行`, ` · line ${entry.line}`) : ''}`, 'meta'), node('pre', entry.text || '')); }
    details.append(disclosure(pick('规则版本指纹', 'Rule source fingerprint'), source)); provenance.append(details);
  }
  container.append(provenance);
}
