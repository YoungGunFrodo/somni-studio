// The conversation panel. Events for an answer can arrive before POST /api/chat returns its
// request id (a fast server), so events for unknown ids are buffered and replayed.

import { $, cloneTemplate, fill, fmt, slot } from "../core/dom.js";
import { api } from "../net/api.js";

const MAX_TEXTAREA_PX = 140;
const BUFFER_TTL_MS = 30_000;
const SOURCE_TITLES = { search: "found by keyword search", tags: "found through subject tags", file: "memory file" };
const ALWAYS_RECALLED = new Set(["identity", "profile"]);   // in every answer: not news

export class Chat {
  #log = $("#chat-log");
  #input = $("#chat-input");
  #messages = new Map();          // request id -> {root, body, chips, meta, text, chipFor}
  #early = new Map();             // request id -> [events] that arrived first

  constructor({ socket, hud, onChipHover, onChipClick, toolLabel = (name) => name }) {
    this.hud = hud;
    this.toolLabel = toolLabel;
    this.onChipHover = onChipHover;
    this.onChipClick = onChipClick;
    $("#chat-form").addEventListener("submit", (e) => {
      e.preventDefault();
      this.#send();
    });
    this.#input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        this.#send();
      }
    });
    this.#input.addEventListener("input", () => this.#autosize());
    $("#chat-collapse").addEventListener("click", () => $("#chat").classList.toggle("collapsed"));
    for (const type of ["chat.start", "recall", "token", "chat.done", "chat.error", "tool.call", "tool.result"]) {
      socket.on(type, (event) => this.#route(event));
    }
    this.#loadHistory();
  }

  /** Earlier exchanges from the episode log, so a reload does not wipe the conversation. */
  async #loadHistory() {
    let messages = [];
    try {
      ({ messages } = await api.conversation());
    } catch {
      return;                                       // read-only or offline: start empty
    }
    if (!messages.length || this.#messages.size) return;
    $("#chat-empty").hidden = true;
    for (const m of messages) {
      const time = fmt.time(new Date(m.at * 1000));
      const asked = this.#append("user", m.question, time);
      asked.root.classList.add("history");
      const answered = this.#append("assistant", m.answer, time);
      answered.root.classList.add("history");
      answered.meta.title = `part of conversation #${m.episode_id} (sleep learns from whole conversations)`;
    }
  }

  async #send() {
    const question = this.#input.value.trim();
    if (!question) return;
    this.#input.value = "";
    this.#autosize();
    await this.sendText(question);
  }

  /** Ask a question (typed, or heard by the microphone). */
  async sendText(question) {
    question = question.trim();
    if (!question) return;
    $("#chat-empty").hidden = true;
    this.#append("user", question);
    const msg = this.#append("assistant", "");
    msg.body.classList.add("streaming");
    try {
      const { request_id: id } = await api.ask(question);
      this.#messages.set(id, msg);
      for (const event of this.#early.get(id)?.events ?? []) this.#apply(msg, event);
      this.#early.delete(id);
    } catch (err) {
      this.#fail(msg, err.message);
    }
  }

  #append(role, text, time = fmt.time()) {
    const root = cloneTemplate("tpl-message");
    root.dataset.role = role;
    fill(root, { role: role === "user" ? "you" : "somni", time, body: text });
    this.#log.appendChild(root);
    this.#scroll();
    return { root, body: slot(root, "body"), chips: slot(root, "chips"), meta: slot(root, "meta"), text: "", chipFor: new Map() };
  }

  #route(event) {
    const msg = this.#messages.get(event.request_id);
    if (msg) {
      this.#apply(msg, event);
      return;
    }
    const entry = this.#early.get(event.request_id) ?? { at: Date.now(), events: [] };
    entry.events.push(event);
    this.#early.set(event.request_id, entry);
    for (const [id, e] of this.#early) if (Date.now() - e.at > BUFFER_TTL_MS) this.#early.delete(id);
  }

  #apply(msg, event) {
    switch (event.type) {
      case "recall":
        for (const item of event.items) if (!ALWAYS_RECALLED.has(item.source)) this.#chip(msg, item);
        break;
      case "token":
        msg.text = event.replace ? event.text : msg.text + event.text;
        msg.body.textContent = msg.text;
        this.#scroll();
        break;
      case "chat.done":
        this.#done(msg, event);
        break;
      case "chat.error":
        this.#fail(msg, event.message);
        break;
      case "tool.call":
        this.#toolChip(msg, event);
        break;
      case "tool.result":
        this.#toolChip(msg, event);
        break;
      default:
        break;
    }
  }

  #chip(msg, item) {
    if (msg.chipFor.has(item.node_id)) return msg.chipFor.get(item.node_id);
    const chip = fill(cloneTemplate("tpl-chip"), { label: item.label });
    chip.dataset.source = item.source;
    chip.title = `${item.label} — ${SOURCE_TITLES[item.source] ?? item.source}`;
    chip.addEventListener("mouseenter", () => this.onChipHover(item.node_id, true));
    chip.addEventListener("mouseleave", () => this.onChipHover(item.node_id, false));
    chip.addEventListener("click", () => this.onChipClick(item.node_id));
    msg.chips.appendChild(chip);
    msg.chipFor.set(item.node_id, chip);
    return chip;
  }

  #toolChip(msg, event) {
    let chip = msg.chipFor.get(`call:${event.call_id}`);
    if (!chip) {
      chip = fill(cloneTemplate("tpl-chip"), { label: this.toolLabel(event.name) });
      chip.dataset.source = "tool";
      chip.disabled = true;
      msg.chips.appendChild(chip);
      msg.chipFor.set(`call:${event.call_id}`, chip);
    }
    if (event.type === "tool.call") {
      chip.classList.add("working");
      chip.title = `${event.name} ${JSON.stringify(event.arguments ?? {})}`;
    } else {
      chip.classList.remove("working");
      chip.classList.add(event.ok ? "ok" : "failed");
      slot(chip, "label").textContent = `${this.toolLabel(event.name)}: ${event.summary}`;
    }
  }

  #done(msg, event) {
    msg.body.classList.remove("streaming");
    msg.text = event.text || msg.text;
    msg.body.textContent = msg.text || "(no answer)";
    const notes = [`${(event.elapsed_ms / 1000).toFixed(1)}s`];
    if (event.used_node) {
      const recalled = (event.recall ?? []).find((r) => r.node_id === event.used_node);
      const chip = this.#chip(msg, recalled ?? { node_id: event.used_node, source: "search", label: "cited memory" });
      chip.classList.add("used");
      notes.push(`cited ${recalled?.label ?? "a memory"}`);
    }
    if (event.caveat) notes.push("answered from a disputed memory");
    if (!event.episode_id) notes.push("read-only: not logged");
    msg.meta.textContent = notes.join(" · ");
    msg.meta.title = event.episode_id
      ? `part of conversation #${event.episode_id}: tonight's sleep learns from the whole conversation`
      : "";
    this.#scroll();
  }

  #fail(msg, message) {
    msg.body.classList.remove("streaming");
    msg.root.classList.add("error");
    msg.body.textContent = message;
    this.hud.toast(message, "error");
  }

  #autosize() {
    this.#input.style.height = "auto";
    this.#input.style.height = `${Math.min(MAX_TEXTAREA_PX, this.#input.scrollHeight)}px`;
  }

  #scroll() {
    this.#log.scrollTop = this.#log.scrollHeight;
  }
}
