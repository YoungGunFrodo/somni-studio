// Settings dialog, generated from the server's settings object so it never drifts from it.

import { $, clear, cloneTemplate, el, fill, slot } from "../core/dom.js";
import { api } from "../net/api.js";

const SECTION_TITLES = { somni: "Somni", provider: "Model", graph: "Graph", server: "Server" };
const SECTION_ORDER = ["somni", "provider", "graph", "server"];

const HELP = {
  "somni.repo": "Folder of the Somni framework checkout (contains sleepai/).",
  "somni.home": "A Somni home: memory.db, checkpoints/, adapters/.",
  "somni.adapter": "Optional expert folder to run instead of the active checkpoint's.",
  "somni.base_model": "Hugging Face model id or folder. Empty: the home's config.json.",
  "somni.read_only": "On: chat never writes to the memory store. Off: every question is logged as an episode the next sleep learns from.",
  "somni.use_tags": "Recall also scans subject tags and linked memories.",
  "provider.kind": "somni: the in-process model with live brain telemetry. openai: an OpenAI-compatible server (LM Studio, llama-server); only memory lights up.",
  "provider.url": "Server address for the openai provider.",
  "provider.model": "Model id sent to the server (LM Studio needs it).",
  "provider.max_new_tokens": "Longest answer, in tokens.",
  "provider.dtype": "Model precision. auto: fp16 on GPUs without fast bf16 (e.g. RX 6000 series, about 4.7x faster there), bf16 where the hardware has it.",
  "graph.poll_seconds": "How often the store and checkpoints are checked for changes.",
  "graph.max_memories": "Most memories drawn; the most used are kept.",
  "server.host": "127.0.0.1 keeps the studio on this computer. Takes effect after a restart.",
  "server.port": "Takes effect after a restart.",
};

export class SettingsDialog {
  #dialog = $("#settings");
  #body = $("#settings-body");
  #error = $("#settings-error");
  #original = null;
  #choices = {};
  #types = {};

  constructor({ hud, onSaved }) {
    this.hud = hud;
    this.onSaved = onSaved;
    $("#open-settings").addEventListener("click", () => this.open());
    $("#settings-form").addEventListener("submit", (e) => {
      if (e.submitter?.value !== "save") return;      // Cancel closes the dialog natively
      e.preventDefault();
      this.#save();
    });
  }

  async open() {
    try {
      const data = await api.settings();
      this.#original = data.settings;
      this.#choices = data.choices ?? {};
      this.#types = data.types ?? {};
      $("#settings-path").textContent = data.path;
      this.#error.textContent = "";
      this.#render();
      this.#dialog.showModal();
    } catch (err) {
      this.hud.toast(err.message, "error");
    }
  }

  #render() {
    clear(this.#body);
    for (const section of SECTION_ORDER) {
      const values = this.#original[section];
      if (!values) continue;
      const fs = fill(cloneTemplate("tpl-setting-section"), { title: SECTION_TITLES[section] ?? section });
      const grid = slot(fs, "fields");
      for (const [key, value] of Object.entries(values)) grid.appendChild(this.#field(section, key, value));
      this.#body.appendChild(fs);
    }
  }

  #field(section, key, value) {
    const path = `${section}.${key}`;
    const row = fill(cloneTemplate("tpl-setting-field"), { label: key, help: HELP[path] ?? "" });
    const type = this.#types[path] ?? typeof value;
    let input;
    if (this.#choices[path]) {
      input = el("select");
      for (const choice of this.#choices[path]) input.appendChild(el("option", { text: choice, attrs: { value: choice } }));
      input.value = value;
    } else if (type === "bool") {
      input = el("input", { attrs: { type: "checkbox" } });
      input.checked = value;
    } else if (type === "int" || type === "float") {
      input = el("input", { attrs: { type: "number", step: type === "int" ? "1" : "any" } });
      input.value = String(value);
    } else {
      input = el("input", { attrs: { type: "text", spellcheck: "false" } });
      input.value = value;
    }
    input.dataset.path = path;
    input.dataset.type = type;
    slot(row, "control").appendChild(input);
    return row;
  }

  #collect() {
    const patch = {};
    for (const input of this.#body.querySelectorAll("[data-path]")) {
      const [section, key] = input.dataset.path.split(".");
      let value;
      switch (input.dataset.type) {
        case "bool": value = input.checked; break;
        case "int": value = Number.parseInt(input.value, 10); break;
        case "float": value = Number.parseFloat(input.value); break;
        default: value = input.value;
      }
      if (typeof value === "number" && !Number.isFinite(value)) throw new Error(`${input.dataset.path} must be a number`);
      if (value !== this.#original[section][key]) (patch[section] ??= {})[key] = value;
    }
    return patch;
  }

  async #save() {
    this.#error.textContent = "";
    let patch;
    try {
      patch = this.#collect();
    } catch (err) {
      this.#error.textContent = err.message;
      return;
    }
    if (!Object.keys(patch).length) {
      this.#dialog.close();
      return;
    }
    const button = $("#settings-save");
    button.disabled = true;
    try {
      const res = await api.saveSettings(patch);
      this.#dialog.close();
      this.hud.toast(res.note || "settings saved", "good");
      this.onSaved(res.settings);
    } catch (err) {
      this.#error.textContent = err.message;
    } finally {
      button.disabled = false;
    }
  }
}
