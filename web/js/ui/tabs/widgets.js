// Small builders shared by the tabs. All markup comes from templates in index.html.

import { cloneTemplate, el, fill, slot } from "../../core/dom.js";

export function card(title, aside = "") {
  const root = fill(cloneTemplate("tpl-card"), { title, aside });
  return { root, body: slot(root, "body"), aside: slot(root, "aside") };
}

export function statusLine(state, text) {
  const line = fill(cloneTemplate("tpl-status-line"), { text });
  line.dataset.state = state;
  return line;
}

export function field({ label, help = "", control }) {
  const root = fill(cloneTemplate("tpl-field"), { label, help });
  slot(root, "control").appendChild(control);
  return root;
}

export function input(type, value = "", attrs = {}) {
  const node = el("input", { attrs: { type, spellcheck: "false", autocomplete: "off", ...attrs } });
  node.value = value ?? "";
  return node;
}

export function select(options, value) {
  const node = el("select");
  for (const { value: v, label } of options) node.appendChild(el("option", { text: label, attrs: { value: v } }));
  node.value = value;
  return node;
}

export function button(text, { primary = false, danger = false, small = true } = {}) {
  const cls = ["btn", small && "btn-small", primary && "btn-primary", danger && "btn-danger"].filter(Boolean).join(" ");
  return el("button", { className: cls, text, attrs: { type: "button" } });
}

export function row(...children) {
  const node = el("div", { className: "row" });
  for (const c of children) if (c) node.appendChild(c);
  return node;
}

export function para(text) {
  return el("p", { text });
}

/** Run an async action with its button disabled; toast errors. */
export async function busy(btn, hud, action, label = "") {
  const old = btn.textContent;
  btn.disabled = true;
  if (label) btn.textContent = label;
  try {
    return await action();
  } catch (err) {
    hud.toast(err.message, "error");
    return undefined;
  } finally {
    btn.disabled = false;
    btn.textContent = old;
  }
}
