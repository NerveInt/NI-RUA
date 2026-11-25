// 前端双语国际化核心与各页面模型层中英双语单元测试
import assert from 'node:assert/strict';
import test from 'node:test';
import {
  DEFAULT_LOCALE,
  STORAGE_KEY,
  getLocale,
  initLocale,
  mountLanguageSwitch,
  normalizeLocale,
  registerPageI18n,
  resolveLocale,
  setLocale,
  toggleLocale,
} from './i18n.js';
import { CONSOLE_I18N, HANDOFF_BLOCKER_EN, translateAgentBackendText } from '../i18n/console.js';
import {
  CALIB_CHANGES_EN,
  CALIB_CRITERIA_EN,
  CALIB_MODES_EN,
  CALIB_READINESS_EN,
  CALIB_STEPS_EN,
  diffLabelEn,
  localizeDefinition,
  translateGuidanceEn,
} from '../i18n/calibration.js';
import { translateFailureReasonEn, translateTaskReviewTextEn } from '../i18n/workbench.js';
import { translateLibraryServerTextEn } from '../i18n/library.js';
import { criterionRow, criterionRule, describeReason, formatDiffRow, formatMeasure, phaseText, stalenessText, stepView } from '../calibration-model.js';
import { badgeView, relativeTime, summaryFromState } from '../calibration-badge.js';
import * as AgentModel from '../agent-settings-model.js';
import { formatDeltaSummary, formatObjectRotation, formatPerceptionSummary, formatQualityBadge, formatRemoteStatusSummary } from '../perception-model.js';
import { applyAgentNames, buildPhases, mapActionVideo } from '../phase-model.js';
import { environmentLabel, runtimeDescription, taskLabel } from '../task-labels.js';
import { metrics } from '../experiment-metrics.js';
import { groupTimeText, runTimeText } from '../experiment-time.js';
import { familyKey, familyLabel } from '../comparison-model.js';
import { newRunCommand } from '../new-run-command.js';

function createMemoryStorage(initial = {}) {
  const map = new Map(Object.entries(initial));
  return {
    getItem: (key) => (map.has(key) ? map.get(key) : null),
    setItem: (key, value) => { map.set(key, String(value)); },
    dump: () => Object.fromEntries(map),
  };
}

// 构造轻量级模拟 DOM，覆盖 TreeWalker 文本遍历与属性双向切换。
function createMockDocument(htmlLang = 'zh-Hans', docTitle = 'NI-RUA · 真机工作台') {
  class MockNode {
    constructor(nodeType) { this.nodeType = nodeType; this.parentElement = null; this.ownerDocument = doc; }
  }
  class MockText extends MockNode {
    constructor(value) { super(3); this.nodeValue = value; }
    get textContent() { return this.nodeValue; }
    set textContent(v) { this.nodeValue = v; }
  }
  class MockElement extends MockNode {
    constructor(tagName, attrs = {}) {
      super(1);
      this.tagName = tagName.toUpperCase();
      this.attributes = new Map();
      this.dataset = {};
      this.children = [];
      this.childNodes = [];
      this.className = '';
      this.id = '';
      for (const [k, v] of Object.entries(attrs)) this.setAttribute(k, v);
    }
    setAttribute(name, value) {
      const str = String(value);
      this.attributes.set(name, str);
      if (name === 'id') this.id = str;
      if (name === 'class') this.className = str;
      if (name.startsWith('data-')) {
        const camel = name.slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase());
        this.dataset[camel] = str;
      }
    }
    getAttribute(name) { return this.attributes.has(name) ? this.attributes.get(name) : null; }
    hasAttribute(name) { return this.attributes.has(name); }
    removeAttribute(name) { this.attributes.delete(name); }
    append(...nodes) {
      for (const item of nodes) {
        const child = typeof item === 'string' ? new MockText(item) : item;
        child.parentElement = this;
        this.childNodes.push(child);
        if (child.nodeType === 1) this.children.push(child);
      }
      return this;
    }
    insertBefore(node, before) {
      node.parentElement = this;
      const idx = this.childNodes.indexOf(before);
      if (idx >= 0) this.childNodes.splice(idx, 0, node);
      else this.childNodes.push(node);
      if (node.nodeType === 1) {
        const eIdx = this.children.indexOf(before);
        if (eIdx >= 0) this.children.splice(eIdx, 0, node);
        else this.children.push(node);
      }
      return node;
    }
    get textContent() { return this.childNodes.map((n) => n.textContent).join(''); }
    set textContent(v) {
      this.children = [];
      this.childNodes = [new MockText(String(v))];
      this.childNodes[0].parentElement = this;
    }
    matchesSelector(sel) {
      return sel.split(',').map((s) => s.trim()).some((part) => {
        if (part.startsWith('#')) return this.id === part.slice(1);
        if (part.startsWith('.')) return this.className.split(/\s+/).includes(part.slice(1));
        if (part === '[data-agent-name]') return Object.hasOwn(this.dataset, 'agentName');
        if (part.startsWith('[') && part.endsWith(']')) return this.attributes.has(part.slice(1, -1));
        if (part === 'pre#config') return this.tagName === 'PRE' && this.id === 'config';
        return this.tagName === part.toUpperCase();
      });
    }
    closest(selector) {
      for (let cur = this; cur; cur = cur.parentElement) {
        if (cur.nodeType === 1 && cur.matchesSelector(selector)) return cur;
      }
      return null;
    }
    querySelectorAll(selector) {
      const results = [];
      const visit = (el) => {
        for (const child of el.children) {
          if (child.matchesSelector(selector)) results.push(child);
          visit(child);
        }
      };
      visit(this);
      return results;
    }
    querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  }
  const doc = {
    nodeType: 9,
    title: docTitle,
    documentElement: null,
    body: null,
    createElement: (tag) => new MockElement(tag),
    getElementById(id) { return this.documentElement.querySelector(`#${id}`); },
    querySelector(sel) { return this.documentElement.querySelector(sel); },
    querySelectorAll(sel) { return this.documentElement.querySelectorAll(sel); },
    createTreeWalker(root) {
      const texts = [];
      const walk = (n) => {
        for (const child of n.childNodes || []) {
          if (child.nodeType === 3) texts.push(child);
          else if (child.nodeType === 1) walk(child);
        }
      };
      walk(root);
      let index = -1;
      return { nextNode: () => texts[++index] || null };
    },
  };
  doc.documentElement = new MockElement('html', { lang: htmlLang });
  doc.body = new MockElement('body');
  doc.documentElement.append(doc.body);
  return { doc, MockElement, MockText };
}

test('resolveLocale honors URL ?lang priority, persists to localStorage, and defaults to zh-CN', (t) => {
  t.after(() => { setLocale('zh-CN', { persist: false, syncUrl: false }); });

  assert.equal(normalizeLocale('en'), 'en');
  assert.equal(normalizeLocale('en-US'), 'en');
  assert.equal(normalizeLocale('zh-CN'), 'zh-CN');
  assert.equal(normalizeLocale('zh-Hans'), 'zh-CN');
  assert.equal(normalizeLocale('fr'), null);
  assert.equal(normalizeLocale(''), null);

  // 无参数与存储时必须严格默认回退 'zh-CN'
  const emptyStore = createMemoryStorage();
  assert.equal(resolveLocale({ search: '', storage: emptyStore }), DEFAULT_LOCALE);
  assert.equal(DEFAULT_LOCALE, 'zh-CN');

  // URL ?lang=en 优先于已存储的 zh-CN，并立即写入 localStorage['ni-rua.lang']
  const store = createMemoryStorage({ [STORAGE_KEY]: 'zh-CN' });
  assert.equal(resolveLocale({ search: '?lang=en', storage: store }), 'en');
  assert.equal(store.getItem(STORAGE_KEY), 'en');

  // 无 URL 参数时读取 localStorage['ni-rua.lang']
  assert.equal(resolveLocale({ search: '?view=runs', storage: store }), 'en');

  // URL ?lang=zh-CN 覆盖并持久化回中文
  assert.equal(resolveLocale({ search: '?lang=zh-CN', storage: store }), 'zh-CN');
  assert.equal(store.getItem(STORAGE_KEY), 'zh-CN');

  // 非法 URL 参数忽略并回退 localStorage
  store.setItem(STORAGE_KEY, 'en');
  assert.equal(resolveLocale({ search: '?lang=invalid', storage: store }), 'en');
});

test('registerPageI18n and mountLanguageSwitch translate static DOM reversibly without touching skipped blocks', (t) => {
  const prevDoc = globalThis.document;
  t.after(() => {
    globalThis.document = prevDoc;
    setLocale('zh-CN', { persist: false, syncUrl: false });
  });

  const { doc, MockElement } = createMockDocument('zh-Hans', 'NI-RUA · 真机工作台');
  globalThis.document = doc;

  const headState = new MockElement('div', { class: 'head-state' });
  const heading = new MockElement('span');
  heading.append('  SO-101 真机工作台  ');
  const stopBtn = new MockElement('button', { title: '中断 Codex 轮次并关闭动作准入。不需要租约。' });
  stopBtn.append('中断轮次', new MockElement('small'));
  stopBtn.children[0].append('stop');
  const promptLabel = new MockElement('label');
  const agentSpan = new MockElement('span', { 'data-agent-name': '' });
  agentSpan.append('Codex');
  promptLabel.append('给 ', agentSpan, ' 的指令：空闲时开始新任务，运行中可随时插话');
  const configPre = new MockElement('pre', { id: 'config' });
  configPre.append('运行控制');
  doc.body.append(headState, heading, stopBtn, promptLabel, configPre);

  initLocale({ search: '', storage: createMemoryStorage() });
  registerPageI18n(CONSOLE_I18N, doc);
  const switchBtn = mountLanguageSwitch(headState);
  applyAgentNames(doc, 'Claude Code');

  assert.equal(getLocale(), 'zh-CN');
  assert.equal(doc.documentElement.getAttribute('lang'), 'zh-Hans');
  assert.equal(switchBtn.textContent, 'EN');
  assert.equal(heading.textContent, '  SO-101 真机工作台  ');
  assert.equal(stopBtn.getAttribute('title'), '中断 Claude Code 轮次并关闭动作准入。不需要租约。');

  // 切换到英文：更新 lang、标题、按钮、空白保留文本与带 data-agent-name 的混合子节点，且不改 pre#config
  toggleLocale();
  applyAgentNames(doc, 'Claude Code');
  assert.equal(getLocale(), 'en');
  assert.equal(doc.documentElement.getAttribute('lang'), 'en');
  assert.equal(doc.title, 'NI-RUA · Operator Console');
  assert.equal(switchBtn.textContent, '中文');
  assert.equal(heading.textContent, '  SO-101 Console  ');
  assert.equal(stopBtn.childNodes[0].nodeValue, 'Stop Turn');
  assert.equal(stopBtn.children[0].textContent, 'stop');
  assert.equal(stopBtn.getAttribute('title'), 'Interrupt Claude Code turn and close action admission. No lease required.');
  assert.equal(promptLabel.textContent, 'Prompt for Claude Code: start a new task when idle, or steer while running');
  assert.equal(configPre.textContent, '运行控制');

  // 切回简体中文：100% 恢复原中文文本、html[lang] 与页面 title
  toggleLocale();
  applyAgentNames(doc, 'Claude Code');
  assert.equal(getLocale(), 'zh-CN');
  assert.equal(doc.documentElement.getAttribute('lang'), 'zh-Hans');
  assert.equal(doc.title, 'NI-RUA · 真机工作台');
  assert.equal(switchBtn.textContent, 'EN');
  assert.equal(heading.textContent, '  SO-101 真机工作台  ');
  assert.equal(stopBtn.childNodes[0].nodeValue, '中断轮次');
  assert.equal(stopBtn.getAttribute('title'), '中断 Claude Code 轮次并关闭动作准入。不需要租约。');
  assert.equal(promptLabel.textContent, '给 Claude Code 的指令：空闲时开始新任务，运行中可随时插话');
});

test('calibration i18n maps S0–S6 steps, readiness, criteria, threshold_label, diff keys, guidance, and badge', (t) => {
  t.after(() => { setLocale('zh-CN', { persist: false, syncUrl: false }); });

  const rawDef = {
    statuses: { not_started: '未开始', running: '进行中', passed: '通过', failed: '未通过', stale: '需重做' },
    modes: {
      first_run: { title: '首次完整标定', description: '第一次使用…' },
      recalibrate: { title: '重新标定（勾选哪里变了）', description: '只重做受影响步骤。' },
    },
    changes: Object.keys(CALIB_CHANGES_EN).map((id) => ({ id, label: '中文变更', steps: ['S4', 'S5', 'S6'] })),
    steps: Object.keys(CALIB_STEPS_EN).map((id) => ({
      id,
      title: '中文标题',
      purpose: '中文目的',
      operator: ['中文操作'],
      machine: '中文机器说明',
      outputs: ['中文产物'],
      readiness: [{ id: 'torque_on', kind: 'auto', label: '机械臂已上力矩' }, { id: 'table_clear', kind: 'operator', label: '桌面区域已清空' }],
      criteria: id === 'S1'
        ? [{ id: 'contacts', label: '完成阻滞候选采集的点数（待物理审核）', op: '==', threshold_value: null, threshold_label: '计划点数' }]
        : [{ id: 'preflight_ok', label: '上电自检报告通过', op: 'true', threshold_value: true }],
    })),
  };

  setLocale('en', { persist: false, syncUrl: false });
  const enDef = localizeDefinition(rawDef, true);
  assert.equal(enDef.modes.first_run.title, CALIB_MODES_EN.first_run.title);
  assert.equal(enDef.changes[0].label, CALIB_CHANGES_EN[enDef.changes[0].id]);
  for (const step of enDef.steps) {
    assert.ok(!/[\u4e00-\u9fff]/.test(step.title), `step ${step.id} title should be English`);
    assert.ok(!/[\u4e00-\u9fff]/.test(step.purpose), `step ${step.id} purpose should be English`);
    assert.equal(step.readiness[0].label, CALIB_READINESS_EN.torque_on);
  }
  const s1Rule = criterionRule(enDef.steps.find((s) => s.id === 'S1').criteria[0]);
  assert.equal(s1Rule, '= planned points');

  const cRow = criterionRow({ id: 'fixed_chain_reprojection_px', label: '固定相机链路重投影 RMS', value: 1.2, threshold: 6.0, op: '<=', unit: 'px', passed: true }, 'S5');
  assert.equal(cRow.label, CALIB_CRITERIA_EN['S5:fixed_chain_reprojection_px']);
  assert.equal(formatMeasure(true), 'Yes');
  assert.equal(formatDiffRow({ key: 'joint_offset.shoulder_pan', label: 'shoulder_pan 零位 offset_rad', old: 0.01, new: -0.02, unit: 'rad' }).label, 'shoulder_pan zero offset_rad');
  assert.equal(diffLabelEn({ key: 'wrist.T_link_camera.translation_change_m', label: '手眼平移变化' }), 'wrist T_link_camera translation Δ');
  assert.equal(translateGuidanceEn('还缺：画面左上角、画面右下角'), 'Still needed: top-left, bottom-right');
  assert.equal(describeReason('readiness_failed:torque_on', enDef), 'Readiness check failed: Arm torque is enabled');
  assert.equal(stepView(enDef, { plan: ['S4'], steps: { S0: { status: 'passed' } } }, 'S0').label, 'Passed · Not needed');
  assert.equal(phaseText({ phase: 'sample', pose_index: 2, pose_count: 8, kept_count: 2 }), 'Capturing & detecting board · Pose 2/8 · Kept 2');
  assert.match(stalenessText({ status: 'ok', rms_px: 1.25 }), /Board poses in both cameras agree/);

  const now = Date.parse('2026-11-20T12:00:00Z');
  assert.equal(relativeTime('2026-11-20T11:55:00Z', now), '5m ago');
  const summary = summaryFromState({
    available: true,
    bundle: { present: true, active_version: 'v3', calibration_created_at: '2026-11-20T11:00:00Z', verification: { passed: true, max_error_m: 0.0021 } },
  });
  const badge = badgeView(summary, now);
  assert.equal(badge.text, 'Calibrated · 1h ago');
  assert.match(badge.title, /Active version v3/);

  // 切回中文验证零破坏
  setLocale('zh-CN', { persist: false, syncUrl: false });
  assert.equal(formatMeasure(true), '是');
  assert.equal(badgeView(summaryFromState({ available: false }), now).text, '标定状态未知');
});

test('console, perception, workbench, and library models produce idiomatic English and restore zh-CN cleanly', (t) => {
  t.after(() => { setLocale('zh-CN', { persist: false, syncUrl: false }); });

  setLocale('en', { persist: false, syncUrl: false });

  // Agent settings & console helpers
  assert.equal(AgentModel.fieldLabel('effort'), 'Effort');
  assert.equal(AgentModel.tierLabel('default'), 'Default');
  assert.equal(AgentModel.scopeLabel('next_turn'), 'Applies next turn');
  assert.equal(translateAgentBackendText('该 profile 只提供 openai_responses，Claude Code 需要 anthropic_messages'), 'Profile only serves openai_responses; Claude Code requires anthropic_messages');
  assert.equal(translateAgentBackendText('未知文案', 'agent_turn_running'), HANDOFF_BLOCKER_EN.agent_turn_running);
  const validation = AgentModel.profileRequest(AgentModel.emptyProfileForm());
  assert.equal(validation.ok, false);
  assert.match(validation.errors.id, /lowercase letter or digit/);
  assert.match(newRunCommand({ model: 'gpt-5.4', observationProfile: 'formal_rgb', environment: 'so101_real' }), /<task-name>[\s\S]*Call robot_capabilities and robot_observe/);

  // Perception helpers
  assert.equal(formatPerceptionSummary({ objects: [{ id: 'o1' }], table: { table_z: 0.012 } }), '1 object detected · table z=0.012m');
  assert.equal(formatObjectRotation({ yaw_deg: 15, axis_tilt_deg: 4 }), '15° / 4°');
  assert.equal(formatQualityBadge({ quality: 'low', flags: ['floating', 'low_depth_confidence'] }).text, 'floating, low depth confidence');
  assert.match(formatDeltaSummary({ moved: [{ before_id: 'o1', after_id: 'o1', delta_m: [0.02, -0.01, 0] }], appeared: ['o2'] }), /Motion delta estimate · o1 → o1: Δ=\(\+20, -10, \+0\) mm \| newly matched 1/);
  assert.equal(formatRemoteStatusSummary({ enabled: false }).label, 'Perception: Disabled');

  // Phase & workbench helpers
  const phases = buildPhases(
    [],
    [{ id: 'u1', type: 'userMessage', thread_id: 't1', turn_id: 'turn1', text: 'pick block' }],
    new Map(),
  );
  assert.equal(phases[0].title, 'Task prompt');
  assert.match(mapActionVideo({ entry: {}, identity: {} }, null).reason, /Action and recording differ in run \/ episode \/ generation/);
  assert.equal(translateFailureReasonEn({ code: 'model_request_limit', label: '模型调用额度耗尽' }), 'Model request budget exhausted');
  assert.match(translateTaskReviewTextEn('评分条件组 2'), /Scoring Criterion Group 2/);

  // Library helpers
  assert.equal(taskLabel('hang_mugs'), 'Hang Mugs');
  assert.equal(taskLabel('hang_mugs_random'), 'Hang Mugs (Random)');
  assert.equal(environmentLabel({ environment: 'so101_real' }), 'SO-101 Real Robot');
  assert.match(runtimeDescription({ environment: 'so101_real' }), /SO-101 real hardware · SO-101 deployment protocol/);
  assert.equal(metrics({ evaluator_available: false, action_count: 4 }), 'No official evaluator for this environment; run is unscored');
  assert.equal(runTimeText({ started_at: '2026-11-20T04:00:00Z', finished_at: '2026-11-20T04:01:05Z' }), 'Start: 2026-11-20 12:00:00\nEnd: 2026-11-20 12:01:05');
  assert.match(groupTimeText({ started_at: '2026-11-20T04:00:00Z', finished_at: '2026-11-20T04:10:00Z', time_coverage: { run_count: 2, started_count: 2, finished_count: 2 } }), /Earliest start: 2026-11-20 12:00:00/);
  assert.equal(familyKey({}), '其他独立实验');
  assert.equal(familyLabel({}), 'Other Independent Runs');
  assert.match(translateLibraryServerTextEn('缺少完整的原生轨迹归档'), /Complete native trajectory archive is missing/);

  // 恢复默认中文核验
  setLocale('zh-CN', { persist: false, syncUrl: false });
  assert.equal(taskLabel('hang_mugs'), '悬挂马克杯');
  assert.equal(familyLabel({}), '其他独立实验');
});
