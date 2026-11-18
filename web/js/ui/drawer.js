// The left rail and the drawer its tabs open in. Each tab is an object with
//   title: string, render(body): Promise|void, and optional onEvent(type, event) / hide().
// A tab renders when opened and refreshes itself from live events while it is open.

import { $, clear } from "../core/dom.js";

export class Drawer {
  #drawer = $("#drawer");
  #body = $("#drawer-body");
  #tabs = new Map();
  #current = null;

  constructor({ onChange }) {
    this.onChange = onChange;
    for (const btn of document.querySelectorAll(".rail-btn")) {
      btn.addEventListener("click", () => this.open(btn.dataset.tab));
    }
    $("#drawer-close").addEventListener("click", () => this.close());
  }

  register(name, tab) {
    this.#tabs.set(name, tab);
  }

  get current() {
    return this.#current;
  }

  get isOpen() {
    return !this.#drawer.hidden;
  }

  async open(name) {
    if (name === "brain" || !this.#tabs.has(name)) {
      this.close();
      return;
    }
    if (this.#current && this.#current !== name) this.#tabs.get(this.#current).hide?.();
    this.#current = name;
    const tab = this.#tabs.get(name);
    $("#drawer-title").textContent = tab.title;
    this.#mark(name);
    this.#drawer.hidden = false;
    document.body.classList.add("drawer-open");
    clear(this.#body);
    this.onChange?.();
    await tab.render(this.#body);
  }

  close() {
    if (this.#current) this.#tabs.get(this.#current).hide?.();
    this.#current = null;
    this.#drawer.hidden = true;
    document.body.classList.remove("drawer-open");
    this.#mark("brain");
    this.onChange?.();
  }

  /** Re-render the open tab (after an action changed what it shows). */
  async refresh() {
    if (!this.#current) return;
    const tab = this.#tabs.get(this.#current);
    const scroll = this.#body.scrollTop;
    clear(this.#body);
    await tab.render(this.#body);
    this.#body.scrollTop = scroll;
  }

  /** Forward a live event to the open tab. */
  dispatch(type, event) {
    if (this.#current) this.#tabs.get(this.#current).onEvent?.(type, event);
  }

  #mark(name) {
    for (const btn of document.querySelectorAll(".rail-btn")) btn.classList.toggle("active", btn.dataset.tab === name);
  }
}
