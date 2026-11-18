// DOM helpers: templates are the only way markup is created, so structure lives in index.html.

export const $ = (selector, root = document) => root.querySelector(selector);

/** A fresh copy of a <template id="..."> element's first child. */
export function cloneTemplate(id) {
  const tpl = document.getElementById(id);
  if (!(tpl instanceof HTMLTemplateElement)) throw new Error(`missing template #${id}`);
  const node = tpl.content.firstElementChild;
  if (!node) throw new Error(`template #${id} is empty`);
  return node.cloneNode(true);
}

/** The element marked data-slot="name" inside root (or root itself). */
export function slot(root, name) {
  if (root.dataset?.slot === name) return root;
  const el = root.querySelector(`[data-slot="${name}"]`);
  if (!el) throw new Error(`missing slot "${name}"`);
  return el;
}

/** Set text of several slots at once: fill(el, {label: "x", meta: "y"}). */
export function fill(root, values) {
  for (const [name, value] of Object.entries(values)) slot(root, name).textContent = value ?? "";
  return root;
}

export function clear(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
  return el;
}

export function el(tag, { className, text, attrs } = {}, children = []) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  if (attrs) for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  for (const child of children) if (child) node.appendChild(child);
  return node;
}

export const fmt = {
  int: (n) => (Number.isFinite(n) ? Math.round(n).toLocaleString() : "–"),
  num: (n, digits = 3) => (Number.isFinite(n) ? Number(n).toFixed(digits) : "–"),
  time: (date = new Date()) => date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
  chars: (n) => (n >= 1000 ? `${(n / 1000).toFixed(1)}k chars` : `${n} chars`),
  truncate: (s, n) => (s && s.length > n ? `${s.slice(0, n - 1)}…` : s ?? ""),
};

/** "Qwen3-4B-Instruct-2507" from a model id, a folder, or a Hugging Face cache snapshot path. */
export function modelName(id) {
  const path = String(id ?? "").replace(/\\/g, "/").replace(/\/+$/, "");
  const hub = path.match(/models--([^/]+)/);
  return hub ? hub[1].split("--").pop() : path.split("/").pop();
}
