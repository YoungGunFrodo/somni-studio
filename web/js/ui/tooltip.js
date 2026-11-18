// The hover card: what a node is, its name, and one line of context.

import { $, fill, fmt, slot } from "../core/dom.js";

const OFFSET = 16;

function subtitle(node) {
  const a = node.attrs ?? {};
  switch (node.kind) {
    case "memory":
      return fmt.truncate(a.summary, 140);
    case "tag":
      return `${fmt.int(a.count)} memories${a.virtual ? " · grouped by subject (untagged)" : ""}`;
    case "file":
      return `memory file · ${fmt.chars(a.chars)} · ${a.memories} memories · click to open`;
    case "expert":
      return a.routed ? `routed · threshold ${fmt.num(a.threshold)}` : "always on";
    case "module":
      return `${a.block} · layer ${a.layer}`;
    default:
      return a.role ?? "";
  }
}

export class Tooltip {
  #el = $("#tooltip");
  #host;

  constructor(host) {
    this.#host = host;
  }

  show(node, clientX, clientY) {
    if (!node) {
      this.hide();
      return;
    }
    fill(this.#el, { kind: node.kind, label: node.label, sub: subtitle(node) });
    slot(this.#el, "kind").dataset.kind = node.kind;
    this.#el.hidden = false;
    const box = this.#host.getBoundingClientRect();
    const w = this.#el.offsetWidth;
    const h = this.#el.offsetHeight;
    let x = clientX - box.left + OFFSET;
    let y = clientY - box.top + OFFSET;
    if (x + w > box.width - 8) x = clientX - box.left - w - OFFSET;
    if (y + h > box.height - 8) y = clientY - box.top - h - OFFSET;
    this.#el.style.transform = `translate(${Math.max(8, x)}px, ${Math.max(8, y)}px)`;
  }

  hide() {
    this.#el.hidden = true;
  }
}
