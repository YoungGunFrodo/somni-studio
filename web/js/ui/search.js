// Find a memory, tag or file by name; matches are highlighted in the scene, Enter flies to one.

import { $, clear, cloneTemplate, fill, slot } from "../core/dom.js";

const DEBOUNCE_MS = 90;

export class Search {
  #input = $("#search");
  #list = $("#search-results");
  #results = [];
  #active = 0;
  #timer = null;

  constructor({ store, onHighlight, onPick }) {
    this.store = store;
    this.onHighlight = onHighlight;
    this.onPick = onPick;
    this.#input.addEventListener("input", () => {
      clearTimeout(this.#timer);
      this.#timer = setTimeout(() => this.#run(), DEBOUNCE_MS);
    });
    this.#input.addEventListener("keydown", (e) => this.#key(e));
    this.#input.addEventListener("blur", () => setTimeout(() => (this.#list.hidden = true), 150));
    this.#input.addEventListener("focus", () => (this.#list.hidden = !this.#results.length));
    document.addEventListener("keydown", (e) => {
      if (e.key === "/" && !["INPUT", "TEXTAREA"].includes(document.activeElement?.tagName)) {
        e.preventDefault();
        this.#input.focus();
      }
    });
  }

  clear() {
    this.#input.value = "";
    this.#run();
  }

  #run() {
    this.#results = this.store.search(this.#input.value);
    this.#active = 0;
    this.#render();
    this.onHighlight(this.#results.length ? new Set(this.#results.map((n) => n.id)) : null);
  }

  #render() {
    clear(this.#list);
    this.#results.forEach((node, i) => {
      const li = fill(cloneTemplate("tpl-search-result"), { kind: node.kind, label: node.label });
      slot(li, "kind").dataset.kind = node.kind;
      li.classList.toggle("active", i === this.#active);
      li.querySelector("button").addEventListener("mousedown", (e) => {
        e.preventDefault();
        this.#choose(i);
      });
      this.#list.appendChild(li);
    });
    this.#list.hidden = !this.#results.length;
  }

  #choose(i) {
    const node = this.#results[i];
    if (!node) return;
    this.#list.hidden = true;
    this.onPick(node.id);
  }

  #key(e) {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      const n = this.#results.length;
      if (!n) return;
      this.#active = (this.#active + (e.key === "ArrowDown" ? 1 : n - 1)) % n;
      this.#render();
    } else if (e.key === "Enter") {
      e.preventDefault();
      this.#choose(this.#active);
    } else if (e.key === "Escape") {
      this.clear();
      this.#input.blur();
    }
  }
}
