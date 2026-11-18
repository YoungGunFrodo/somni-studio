// Sleep tab: sleep now, watch it work, wake it, and approve or reject what it learned (morning review).

import { cloneTemplate, el, fill, fmt } from "../../core/dom.js";
import { api } from "../../net/api.js";
import { busy, button, card, para, row, statusLine } from "./widgets.js";

const STAGES = [
  ["formation", "Memories"],
  ["training", "Training"],
  ["dreaming", "Dreaming"],
  ["evaluation", "Testing"],
];

const STATE_TEXT = {
  idle: "Awake",                                   // also after you stop a sleep
  sleeping: "Asleep: learning from recent conversations",
  waking: "Waking up at the next checkpoint…",
  finished: "Awake: the last sleep finished",
  failed: "Awake: the last sleep did not finish",
};

function gateDetail(g) {
  const parts = [];
  if (g.current !== null && g.current !== undefined) parts.push(`now ${fmt.num(g.current, 2)}`);
  if (g.candidate !== null && g.candidate !== undefined) parts.push(`after ${fmt.num(g.candidate, 2)}`);
  if (g.note) parts.push(g.note);
  return parts.join(" · ");
}

export class SleepTab {
  title = "Sleep";
  #log = null;
  #stages = null;           // stage name -> element
  #bar = null;
  #dreamLine = null;
  #seen = new Set();
  #clock = null;
  #startedAt = 0;
  #timer = null;

  #tick() {
    clearTimeout(this.#timer);
    if (!this.#clock) return;
    const secs = Math.max(0, Math.round((Date.now() - this.#startedAt) / 1000));
    const h = Math.floor(secs / 3600), m = Math.floor((secs % 3600) / 60), s = secs % 60;
    this.#clock.textContent = `asleep ${h ? `${h}h ` : ""}${m}m ${String(s).padStart(2, "0")}s`;
    this.#timer = setTimeout(() => this.#tick(), 1000);
  }

  constructor({ hud, refresh }) {
    this.hud = hud;
    this.refresh = refresh;
  }

  async render(body) {
    this.hide();
    const s = await api.sleep();
    const asleep = s.state === "sleeping" || s.state === "waking";

    const now = card("Now");
    now.body.appendChild(statusLine(s.state, STATE_TEXT[s.state] ?? s.state));
    if (asleep && s.started_at) {
      this.#clock = now.aside;
      this.#startedAt = s.started_at * 1000;
      this.#tick();
    }
    if (s.summary) now.body.appendChild(para(s.summary));
    if (s.relearn && !asleep) {
      now.body.appendChild(para(`The next sleep relearns every memory for ${s.relearn.to} (after a model switch). `
        + "It takes longer than a normal night."));
    }
    if (!asleep) {
      now.body.appendChild(para("Sleep turns recent conversations into memories, trains a new expert on them while "
        + "every older one stays frozen, practises what it got wrong, and tests itself. It uses the whole GPU, so "
        + "chat pauses until it wakes. You decide afterwards whether to keep what it learned."));
      const start = button("Sleep now", { primary: true });
      start.addEventListener("click", () => busy(start, this.hud, async () => {
        await api.startSleep();
        await this.refresh();
      }, "Falling asleep…"));
      now.body.appendChild(row(start));
    } else {
      const wake = button("Wake up");
      wake.disabled = s.state === "waking";
      const stop = button("Stop now", { danger: true });
      wake.addEventListener("click", () => busy(wake, this.hud, async () => {
        await api.wake();
        await this.refresh();
      }));
      stop.addEventListener("click", () => busy(stop, this.hud, async () => {
        await api.stopSleep();
        await this.refresh();
      }));
      now.body.append(this.#progress(s.phase ?? {}), para("Wake up stops at the next safe point. Stop now ends "
        + "it immediately; nothing half-trained is ever used."), row(wake, stop));
    }
    body.appendChild(now.root);

    for (const c of s.review ?? []) body.appendChild(this.#reviewCard(c, s.last_report));
    if (s.last_report && !(s.review ?? []).length) body.appendChild(this.#reportCard(s.last_report));

    const log = card("Sleep log", asleep ? "live" : "");
    this.#log = el("pre", { className: "log", text: (s.log ?? []).join("\n") });
    log.body.appendChild(this.#log);
    body.appendChild(log.root);
    this.#log.scrollTop = this.#log.scrollHeight;
  }

  #progress(phase) {
    const box = el("div", { className: "card-body" });
    const line = el("div", { className: "phase-line" });
    this.#stages = new Map();
    this.#seen = new Set();
    for (const [name, label] of STAGES) {
      const step = el("span", { className: "phase-step", text: label });
      this.#stages.set(name, step);
      line.appendChild(step);
    }
    const bar = el("div", { className: "meter-bar" });
    this.#bar = el("span");
    bar.appendChild(this.#bar);
    this.#dreamLine = el("p", { text: "" });
    box.append(line, bar, this.#dreamLine);
    if (phase.phase) this.#markStage(phase.phase);
    if (phase.total) this.#bar.style.width = `${Math.round((100 * phase.step) / phase.total)}%`;
    return box;
  }

  #markStage(name) {
    if (!this.#stages?.has(name)) return;
    this.#seen.add(name);
    for (const [stage, element] of this.#stages) {
      element.classList.toggle("now", stage === name);
      element.classList.toggle("done", this.#seen.has(stage) && stage !== name);
    }
  }

  #reviewCard(candidate, report) {
    const c = card("Morning review", candidate.checkpoint);
    c.body.appendChild(statusLine(candidate.passed ? "connected" : "error",
      `${fmt.int(candidate.memories)} memories learned · the tests ${candidate.passed ? "passed" : "did not pass"}`));
    if (report && report.candidate === candidate.checkpoint) c.body.appendChild(this.#gates(report));
    c.body.appendChild(para(candidate.passed
      ? "Keep it to make what it learned part of the model. Reject it to throw this night away; the memories stay in the store and are tried again."
      : "Somni recommends rejecting this night: a test shows it got worse at something. The memories stay in the store."));
    const keep = button("Keep what it learned", { primary: candidate.passed });
    const reject = button("Reject", { danger: !candidate.passed });
    keep.addEventListener("click", () => busy(keep, this.hud, async () => {
      await api.review(candidate.checkpoint, "approve");
      this.hud.toast("adopted: the new expert answers from the next question on", "good");
      await this.refresh();
    }));
    reject.addEventListener("click", () => busy(reject, this.hud, async () => {
      await api.review(candidate.checkpoint, "reject");
      this.hud.toast("rejected");
      await this.refresh();
    }));
    c.body.appendChild(row(keep, reject));
    return c.root;
  }

  #reportCard(report) {
    const c = card("Last dream report", `sleep ${report.sleep_no}`);
    c.body.appendChild(para(`${report.status} · ${fmt.int(report.memories_formed)} memories formed · `
      + `${fmt.int(report.learned)} trained`));
    if (report.gates?.length) c.body.appendChild(this.#gates(report));
    const full = button("Full report");
    full.addEventListener("click", () => busy(full, this.hud, async () => {
      const { text } = await api.sleepReport();
      full.replaceWith(el("pre", { className: "log", text }));
    }));
    c.body.appendChild(row(full));
    return c.root;
  }

  #gates(report) {
    const list = el("ul", { className: "gate-list" });
    for (const g of report.gates ?? []) {
      const li = fill(cloneTemplate("tpl-gate-row"), { name: g.name, detail: gateDetail(g) });
      li.querySelector(".status-dot").dataset.state = g.passed ? "ok" : "bad";
      list.appendChild(li);
    }
    return list;
  }

  onEvent(type, event) {
    if (type === "sleep.log" && this.#log) {
      const atBottom = this.#log.scrollTop + this.#log.clientHeight >= this.#log.scrollHeight - 8;
      const text = this.#log.textContent;
      const kept = event.replace && text ? text.slice(0, text.lastIndexOf("\n") + 1) : text + (text ? "\n" : "");
      this.#log.textContent = kept + event.line;
      if (atBottom) this.#log.scrollTop = this.#log.scrollHeight;
    } else if (type === "sleep.status") {
      this.refresh();
    } else if (type === "sleep.phase" && this.#stages) {
      if (event.state === "start" || event.state === "asking") {
        this.#markStage(event.phase);
        if (event.phase !== "dreaming" || event.state === "start") this.#bar.style.width = "0%";
      }
      if (event.state === "progress" && event.total) {
        this.#markStage(event.phase);
        this.#bar.style.width = `${Math.round((100 * event.step) / event.total)}%`;
      }
      if (event.phase === "dreaming" && event.state === "asking") {
        this.#dreamLine.textContent = `Dreaming about ${event.memories?.length ?? 0} memories (${event.questions} questions)…`;
      }
      if (event.phase === "dreaming" && event.state === "checked") {
        this.#dreamLine.textContent = `Dream: ${event.right} of ${event.questions} answers right; it practises the rest.`;
      }
      if (event.phase === "formation" && event.state === "end") {
        this.#dreamLine.textContent = `Formed ${(event.created ?? []).length} memories.`;
      }
    }
  }

  hide() {
    this.#log = null;
    this.#stages = null;
    this.#clock = null;
    clearTimeout(this.#timer);
  }
}
