// Shared imperative DOM helpers for the NI-RUA web workspaces.

export const byId = (id) => document.getElementById(id);

export function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  Object.assign(node, props);
  node.append(...children.filter((child) => child != null));
  return node;
}

export const textNode = (value) => document.createTextNode(value);

export function setText(node, text) {
  const value = text == null ? '' : String(text);
  if (node.textContent !== value) node.textContent = value;
}

export function setPill(node, text, kind) {
  setText(node, text);
  const name = `pill ${kind || ''}`;
  if (node.className !== name) node.className = name;
}

export function replaceIfChanged(node, signature, build) {
  if (node.dataset.signature === signature) return false;
  node.dataset.signature = signature;
  node.replaceChildren(...build());
  return true;
}
