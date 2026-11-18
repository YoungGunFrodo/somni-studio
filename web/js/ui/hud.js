// Top bar (home, live dot, stats, engine), warnings banner, legend and toasts.

import { $, clear, cloneTemplate, el, fill, fmt, modelName, slot } from "../core/dom.js";
import { LEGEND } from "../scene/palette.js";

const TOAST_MS = 4200;

const ENGINE_TEXT = {
  unloaded: "model not loaded",
  loading: "loading model…",
  ready: "ready",
  busy: "thinking…",
  error: "model error",
};

export class Hud {
  #warningsKey = null;

  constructor({ onEngineToggle, onSleepToggle }) {
    this.engineButton = $("#engine-toggle");
    this.engineButton.addEventListener("click", () => onEngineToggle(this.engineState));
    this.engineState = "unloaded";
    this.sleepState = "idle";
    this.sleepButton = $("#sleep-toggle");
    this.sleepButton.addEventListener("click", () => onSleepToggle(this.asleep));
    this.#renderLegend();
  }

  setConnected(connected) {
    $("#live-dot").className = `dot ${connected ? "live" : "down"}`;
  }

  setGraphMeta(meta) {
    const brain = meta.brain ?? {};
    const memory = meta.memory ?? {};
    $("#home-path").textContent = meta.home ?? "";
    $("#home-chip").title = `Somni home: ${meta.home}${meta.read_only ? " (read-only)" : ""}`;
    const stats = {
      memories: memory.store?.memories,
      tags: memory.tags,
      files: memory.store?.files,
      experts: brain.experts?.length,
      layers: brain.layers,
    };
    for (const [key, value] of Object.entries(stats)) $(`[data-stat="${key}"]`).textContent = fmt.int(value ?? NaN);
    const warnings = meta.warnings ?? [];
    const key = warnings.join("\n");
    const box = $("#warnings");
    if (key === this.#warningsKey) return;           // unchanged: keep a dismissal
    this.#warningsKey = key;
    clear(box);
    for (const w of warnings) box.appendChild(el("p", { text: w }));
    const close = el("button", { className: "btn icon-btn btn-small warnings-close", text: "×",
      attrs: { type: "button", "aria-label": "Dismiss", title: "Dismiss" } });
    close.addEventListener("click", () => (box.hidden = true));
    box.prepend(close);
    box.hidden = !warnings.length;
  }

  get asleep() {
    return this.sleepState === "sleeping" || this.sleepState === "waking";
  }

  setSleep(state) {
    this.sleepState = state;
    document.body.classList.toggle("sleeping", this.asleep);
    $("#sleep-toggle-text").textContent = state === "waking" ? "Waking…" : this.asleep ? "Wake up" : "Sleep now";
    this.sleepButton.disabled = state === "waking";
    this.engineButton.disabled = this.asleep || this.engineState === "loading" || this.engineState === "busy";
    const pill = $("#engine-pill");
    if (this.asleep) {
      pill.dataset.state = "asleep";
      $("#engine-text").textContent = state === "waking" ? "waking up…" : "asleep · learning";
    } else if (pill.dataset.state === "asleep") {
      this.setEngine({ state: this.engineState, provider: this.provider, read_only: this.readOnly });
    }
  }

  setEngine(status) {
    this.engineState = status.state;
    this.provider = status.provider;
    this.readOnly = status.read_only;
    if (this.asleep) return;
    const pill = $("#engine-pill");
    pill.dataset.state = status.state;
    let text = ENGINE_TEXT[status.state] ?? status.state;
    if (status.state === "ready" && status.model) text = `${modelName(status.model)} · ready`;
    if (status.state === "error" && status.detail) text = status.detail;
    $("#engine-text").textContent = text;
    pill.title = [status.provider, status.model, status.dtype, status.detail, status.experts_dir].filter(Boolean).join(" · ");
    const loaded = status.state === "ready" || status.state === "busy";
    this.engineButton.textContent = loaded ? "Unload" : status.state === "loading" ? "Loading…" : "Load model";
    this.engineButton.disabled = status.state === "loading" || status.state === "busy";
    $("#chat-mode").textContent = `${status.provider === "openai" ? "server" : "local model"}${status.read_only ? " · read-only" : " · learning"}`;
  }

  toast(text, level = "info") {
    const node = fill(cloneTemplate("tpl-toast"), { text });
    node.dataset.level = level;
    $("#toasts").appendChild(node);
    setTimeout(() => {
      node.classList.add("leaving");
      setTimeout(() => node.remove(), 450);
    }, TOAST_MS);
  }

  #renderLegend() {
    const list = $("#legend");
    for (const item of LEGEND) {
      const li = fill(cloneTemplate("tpl-legend-item"), { label: item.label });
      slot(li, "swatch").style.color = `#${item.colour.toString(16).padStart(6, "0")}`;
      list.appendChild(li);
    }
  }
}
