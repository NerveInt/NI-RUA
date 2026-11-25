// 前端双语国际化核心模块：零外部依赖，默认回退简体中文（zh-CN），支持静态 DOM 可逆双向翻译与就地语言切换。

export const STORAGE_KEY = 'ni-rua.lang';
export const DEFAULT_LOCALE = 'zh-CN';
const TRANSLATABLE_ATTRS = ['title', 'placeholder', 'aria-label', 'alt'];
const SKIP_TAGS = new Set(['SCRIPT', 'STYLE', 'NOSCRIPT', 'TEXTAREA', 'CODE']);

// 规范化语言标识：仅接受 en 与 zh-CN，非法值返回 null。
export function normalizeLocale(raw) {
  if (typeof raw !== 'string') return null;
  const value = raw.trim().toLowerCase();
  if (!value) return null;
  if (value === 'en' || value.startsWith('en-') || value.startsWith('en_')) return 'en';
  if (value === 'zh-cn' || value === 'zh-hans' || value === 'zh' || value.startsWith('zh-')) return 'zh-CN';
  return null;
}

function defaultStorage() {
  try { return globalThis.localStorage ?? null; } catch { return null; }
}

// 解析当前生效语言：URL ?lang= 优先级最高（并持久化到 localStorage），其次 localStorage，默认回退 'zh-CN'。
export function resolveLocale({
  search = globalThis.location?.search ?? '',
  storage = defaultStorage(),
} = {}) {
  if (typeof search === 'string' && search) {
    const fromUrl = normalizeLocale(new URLSearchParams(search).get('lang'));
    if (fromUrl) {
      try { storage?.setItem?.(STORAGE_KEY, fromUrl); } catch { /* 忽略受限存储环境错误 */ }
      return fromUrl;
    }
  }
  try {
    const fromStorage = normalizeLocale(storage?.getItem?.(STORAGE_KEY));
    if (fromStorage) return fromStorage;
  } catch { /* 忽略受限存储环境错误 */ }
  return DEFAULT_LOCALE;
}

let currentLocale = resolveLocale();
const listeners = new Set();
const registeredRoots = new Map();
const switchButtons = new Set();
const textCache = new WeakMap();
const attrCache = new WeakMap();
const docMetaCache = new WeakMap();

export const getLocale = () => currentLocale;
export const isEn = () => currentLocale === 'en';
export const pick = (zh, en) => (currentLocale === 'en' ? en : zh);
export const localeTag = () => (currentLocale === 'en' ? 'en-US' : 'zh-CN');

// 重新按当前全局环境初始化语言（页面启动或单测切换模拟环境时调用）。
export function initLocale(options) {
  currentLocale = resolveLocale(options);
  return currentLocale;
}

export function onLocaleChange(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function splitWhitespace(raw) {
  const match = /^(\s*)([\s\S]*?)(\s*)$/.exec(raw);
  return match ? [match[1], match[2], match[3]] : ['', raw, ''];
}

function shouldSkipElement(element) {
  if (!element) return true;
  if (SKIP_TAGS.has(element.tagName)) return true;
  if (element.closest?.('.xterm, #tui, #terminal, pre#config, pre#run-data, pre#inspector-data, pre#scene-data, #agent-bar, #ab-pop, .ab-dialog, .ab-overlay, #lang-switch, #calib-badge')) return true;
  return false;
}

function activeAgentName(doc) {
  const marker = doc?.querySelector?.('[data-agent-name]');
  return marker?.textContent?.trim() || null;
}

function applyAttributeValue(element, attr, template, doc) {
  const datasetKey = attr === 'title' ? 'agentTitle' : attr === 'aria-label' ? 'agentAria' : null;
  if (datasetKey && element.dataset && (template.includes('Codex') || element.dataset[datasetKey] !== undefined)) {
    element.dataset[datasetKey] = template;
    const agent = activeAgentName(doc);
    element.setAttribute(attr, agent && template.includes('Codex') ? template.replaceAll('Codex', agent) : template);
    return;
  }
  element.setAttribute(attr, template);
}

// 对指定根节点执行可逆静态翻译：首轮自动缓存原中文文本与属性，切回 zh-CN 完整恢复原文。
export function translateSubtree(root, dictionary = {}, locale = currentLocale) {
  if (!root) return;
  const textMap = dictionary.text || {};
  const attrMap = dictionary.attr || textMap;
  const exactMap = dictionary.exactText || {};
  const doc = root.ownerDocument || (root.nodeType === 9 ? root : globalThis.document);

  if (doc && (root === doc || root === doc.documentElement || root === doc.body)) {
    let meta = docMetaCache.get(doc);
    if (!meta) {
      meta = {
        htmlLang: doc.documentElement?.getAttribute?.('lang') || 'zh-Hans',
        title: doc.title || '',
      };
      docMetaCache.set(doc, meta);
    }
    if (doc.documentElement) {
      doc.documentElement.setAttribute('lang', locale === 'en' ? 'en' : meta.htmlLang);
    }
    if (dictionary.title) {
      if (doc.title === meta.title || doc.title === dictionary.title) {
        doc.title = locale === 'en' ? dictionary.title : meta.title;
      }
    }
  }

  const scope = root.nodeType === 9 ? root.body || root.documentElement : root;
  if (!scope) return;

  // 翻译并缓存元素属性（title / placeholder / aria-label / alt）
  const elements = [
    ...(scope.nodeType === 1 ? [scope] : []),
    ...(typeof scope.querySelectorAll === 'function' ? scope.querySelectorAll('[title],[placeholder],[aria-label],[alt]') : []),
  ];
  for (const element of elements) {
    if (shouldSkipElement(element)) continue;
    let saved = attrCache.get(element);
    for (const attr of TRANSLATABLE_ATTRS) {
      const datasetKey = attr === 'title' ? 'agentTitle' : attr === 'aria-label' ? 'agentAria' : null;
      const current = element.dataset?.[datasetKey] ?? element.getAttribute?.(attr);
      if (current == null) continue;
      let record = saved?.get(attr);
      if (!record) {
        const [, core, ] = splitWhitespace(current);
        const translatedCore = attrMap[core] ?? textMap[core];
        if (translatedCore === undefined) continue;
        const [lead, , trail] = splitWhitespace(current);
        record = { zh: current, en: `${lead}${translatedCore}${trail}` };
        if (!saved) { saved = new Map(); attrCache.set(element, saved); }
        saved.set(attr, record);
      }
      const liveTemplate = element.dataset?.[datasetKey] ?? element.getAttribute(attr);
      if (liveTemplate === record.zh || liveTemplate === record.en || element.dataset?.[datasetKey] !== undefined) {
        applyAttributeValue(element, attr, locale === 'en' ? record.en : record.zh, doc);
      }
    }
  }

  // 遍历文本节点（跳过代码块、终端与动态 JSON 区），保留前后空白与子元素事件绑定
  if (doc?.createTreeWalker) {
    const walker = doc.createTreeWalker(scope, 4 /* NodeFilter.SHOW_TEXT */);
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      if (shouldSkipElement(node.parentElement)) continue;
      let record = textCache.get(node);
      if (!record) {
        const raw = node.nodeValue || '';
        if (Object.hasOwn(exactMap, raw)) {
          record = { zh: raw, en: exactMap[raw] };
          textCache.set(node, record);
        } else {
          const [lead, core, trail] = splitWhitespace(raw);
          if (!core || !Object.hasOwn(textMap, core)) continue;
          record = { zh: raw, en: `${lead}${textMap[core]}${trail}` };
          textCache.set(node, record);
        }
      }
      if (node.nodeValue === record.zh || node.nodeValue === record.en) {
        node.nodeValue = locale === 'en' ? record.en : record.zh;
      }
    }
  }
}

// 注册整页或局部容器的静态词典，初始化时若为英文则立即翻译，后续切换语言自动双向重放。
export function registerPageI18n(dictionary, root = globalThis.document) {
  if (!root) return;
  registeredRoots.set(root, dictionary);
  // 首轮无论当前语言都先扫描建档（zh-CN 下保持原文不变，但锁定原始中文缓存，避免后续动态覆盖前丢失基准）
  translateSubtree(root, dictionary, currentLocale);
}

function updateSwitchButton(button) {
  if (!button) return;
  const toEn = currentLocale !== 'en';
  button.textContent = toEn ? 'EN' : '中文';
  const label = toEn ? 'Switch to English' : '切换到简体中文';
  button.title = label;
  button.setAttribute('aria-label', label);
  button.dataset.lang = currentLocale;
}

// 切换当前语言并同步持久化、更新静态 DOM、语言按钮与动态视图回调。
export function setLocale(nextLocale, { persist = true, syncUrl = true } = {}) {
  const normalized = normalizeLocale(nextLocale) || DEFAULT_LOCALE;
  const changed = normalized !== currentLocale;
  currentLocale = normalized;
  if (persist) {
    try { defaultStorage()?.setItem?.(STORAGE_KEY, currentLocale); } catch { /* 忽略存储异常 */ }
  }
  if (syncUrl && globalThis.location?.href && typeof globalThis.history?.replaceState === 'function') {
    try {
      const url = new URL(globalThis.location.href);
      if (url.searchParams.has('lang')) {
        url.searchParams.set('lang', currentLocale);
        globalThis.history.replaceState(null, '', `${url.pathname}${url.search}${url.hash}`);
      }
    } catch { /* 忽略非标准 URL 环境 */ }
  }
  for (const [root, dictionary] of registeredRoots) {
    translateSubtree(root, dictionary, currentLocale);
  }
  for (const button of switchButtons) updateSwitchButton(button);
  if (changed) {
    for (const listener of [...listeners]) listener(currentLocale);
  }
  return currentLocale;
}

export function toggleLocale() {
  return setLocale(currentLocale === 'en' ? 'zh-CN' : 'en');
}

// 在页面顶部状态栏挂载紧凑语言切换按钮（不破坏现有布局与轮询）。
export function mountLanguageSwitch(container, { className = 'pill lang-switch', before = null } = {}) {
  if (!container || !globalThis.document) return null;
  let button = globalThis.document.getElementById('lang-switch');
  if (!button) {
    button = globalThis.document.createElement('button');
    button.id = 'lang-switch';
    button.type = 'button';
    button.className = className;
    button.onclick = () => toggleLocale();
    if (before && before.parentElement === container) container.insertBefore(button, before);
    else container.append(button);
  }
  switchButtons.add(button);
  updateSwitchButton(button);
  return button;
}
