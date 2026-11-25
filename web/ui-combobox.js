// Accessible listbox and editable combobox (WAI-ARIA 1.2 listbox / combobox patterns) with rich rows, disabled rows
// that stay reachable (their reason is part of the option text) and free-text entry. A <datalist> cannot show
// badges or disabled rows. Keyboard: ArrowUp/ArrowDown, PageUp/PageDown, Home/End, Enter, Escape, type-ahead.
// The keyboard rules are pure functions so node can test them (ui-combobox.test.mjs).

// Next active option index for a navigation key; disabled options stay reachable so their reason can be read.
export function nextIndex(count, index, key, { pageSize = 8 } = {}) {
  if (!count) return -1;
  switch (key) {
    case 'ArrowDown': return index < 0 ? 0 : Math.min(count - 1, index + 1);
    case 'ArrowUp': return index < 0 ? count - 1 : Math.max(0, index - 1);
    case 'Home': return 0;
    case 'End': return count - 1;
    case 'PageDown': return Math.min(count - 1, Math.max(0, index) + pageSize);
    case 'PageUp': return Math.max(0, (index < 0 ? count : index) - pageSize);
    default: return index;
  }
}

// Type-ahead: the next label starting with the typed buffer. A buffer of one repeated character cycles through the
// labels starting with it; a longer buffer refines the match from the current option.
export function typeaheadIndex(labels, start, buffer) {
  const text = String(buffer || '').toLowerCase();
  if (!text || !labels.length) return -1;
  const repeated = [...text].every((char) => char === text[0]);
  const needle = repeated ? text[0] : text;
  const from = repeated ? start + 1 : Math.max(start, 0);
  for (let step = 0; step < labels.length; step += 1) {
    const index = (((from + step) % labels.length) + labels.length) % labels.length;
    if (String(labels[index] || '').toLowerCase().startsWith(needle)) return index;
  }
  return -1;
}

export function createTypeahead({ timeoutMs = 500, clock = () => Date.now() } = {}) {
  let buffer = '', last = -Infinity;
  return (char) => {
    const now = clock();
    buffer = now - last > timeoutMs ? char : buffer + char;
    last = now;
    return buffer;
  };
}

const isPrintable = (event) => event.key.length === 1 && event.key !== ' ' && !event.ctrlKey && !event.metaKey && !event.altKey;
let serial = 0;

// items: [{ key, label (type-ahead text), disabled, selected, render(li) }]. `owner` holds focus and
// aria-activedescendant (the listbox itself, or a combobox input).
export function createListbox({ label, onSelect, onDisabled, className = '' } = {}) {
  const id = `ui-listbox-${++serial}`;
  const element = document.createElement('ul');
  element.id = id;
  element.className = `ui-listbox ${className}`.trim();
  element.setAttribute('role', 'listbox');
  if (label) element.setAttribute('aria-label', label);
  let items = [], nodes = [], active = -1, signature = null, owner = element;
  const typeahead = createTypeahead();
  element.tabIndex = 0;

  function setActive(index, { scroll = true } = {}) {
    if (nodes[active]) nodes[active].classList.remove('is-active');
    active = index;
    const node = nodes[active];
    if (node) {
      node.classList.add('is-active');
      owner.setAttribute('aria-activedescendant', node.id);
      if (scroll) node.scrollIntoView({ block: 'nearest' });
    } else owner.removeAttribute('aria-activedescendant');
  }
  function choose(index) {
    const item = items[index];
    if (!item) return false;
    if (item.disabled) { onDisabled?.(item); return true; }
    onSelect?.(item);
    return true;
  }
  function setItems(next, { signature: nextSignature = null, keepActive = true } = {}) {
    const activeKey = items[active]?.key;
    if (nextSignature !== null && nextSignature === signature && next.length === items.length) {
      items = next;
      return;
    }
    signature = nextSignature;
    items = next;
    nodes = items.map((item, index) => {
      const node = document.createElement('li');
      node.id = `${id}-${index}`;
      node.className = 'ui-option';
      node.setAttribute('role', 'option');
      node.setAttribute('aria-selected', item.selected ? 'true' : 'false');
      if (item.disabled) node.setAttribute('aria-disabled', 'true');
      node.dataset.key = item.key;
      item.render(node);
      node.addEventListener('pointerdown', (event) => event.preventDefault());   // keep focus in the owner
      node.addEventListener('click', () => { setActive(index, { scroll: false }); choose(index); });
      node.addEventListener('pointermove', () => { if (active !== index) setActive(index, { scroll: false }); });
      return node;
    });
    element.replaceChildren(...nodes);
    const kept = keepActive && activeKey != null ? items.findIndex((item) => item.key === activeKey) : -1;
    active = -1;
    setActive(kept, { scroll: false });
  }
  function handleKey(event, { allowHomeEnd = true, typeAhead = true } = {}) {
    if (['ArrowDown', 'ArrowUp', 'PageDown', 'PageUp'].includes(event.key) || (allowHomeEnd && ['Home', 'End'].includes(event.key))) {
      event.preventDefault();
      setActive(nextIndex(items.length, active, event.key));
      return true;
    }
    if (event.key === 'Enter' || (event.key === ' ' && owner === element)) {
      if (active < 0) return false;
      event.preventDefault();
      return choose(active);
    }
    if (typeAhead && isPrintable(event)) {
      const match = typeaheadIndex(items.map((item) => item.label), active, typeahead(event.key));
      if (match >= 0) { event.preventDefault(); setActive(match); return true; }
    }
    return false;
  }
  element.addEventListener('keydown', (event) => { if (owner === element) handleKey(event); });
  element.addEventListener('focus', () => {
    if (owner === element && active < 0 && items.length) setActive(Math.max(0, items.findIndex((item) => item.selected)));
  });
  return {
    id, element, setItems, handleKey, setActive, choose,
    get active() { return active; },
    get items() { return items; },
    activeItem: () => items[active] || null,
    setOwner(node) { owner = node; element.tabIndex = node === element ? 0 : -1; },
    focusSelected() {
      const index = items.findIndex((item) => item.selected);
      setActive(index >= 0 ? index : items.length ? 0 : -1);
      element.focus({ preventScroll: true });
    },
  };
}

// Editable combobox: a text input (role=combobox) that filters a listbox and accepts free text with Enter.
export function createCombobox({ label, placeholder = '', onSelect, onInput, onSubmitText, onDisabled, className = '' } = {}) {
  const listbox = createListbox({ label, onSelect, onDisabled, className });
  const input = document.createElement('input');
  input.type = 'text';
  input.className = 'ui-combobox-input';
  input.setAttribute('role', 'combobox');
  input.setAttribute('aria-autocomplete', 'list');
  input.setAttribute('aria-expanded', 'true');
  input.setAttribute('aria-controls', listbox.id);
  if (label) input.setAttribute('aria-label', label);
  input.placeholder = placeholder;
  input.autocomplete = 'off';
  input.spellcheck = false;
  input.setAttribute('autocapitalize', 'off');
  listbox.setOwner(input);
  input.addEventListener('input', () => { listbox.setActive(-1, { scroll: false }); onInput?.(input.value); });
  input.addEventListener('keydown', (event) => {
    if (event.isComposing) return;
    // Home/End move the caret while typing; they move through the list once an option is active or the field is empty.
    const allowHomeEnd = listbox.active >= 0 || input.value === '';
    if (listbox.handleKey(event, { allowHomeEnd, typeAhead: false })) return;
    if (event.key === 'Enter') {
      event.preventDefault();
      const text = input.value.trim();
      if (text) onSubmitText?.(text);
    } else if (event.key === 'Escape' && input.value) {
      event.preventDefault();
      event.stopPropagation();
      input.value = '';
      onInput?.('');
    }
  });
  return { input, listbox, setItems: listbox.setItems, get value() { return input.value; }, set value(text) { input.value = text; } };
}
