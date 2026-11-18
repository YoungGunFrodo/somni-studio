// Turns live model events into light. Everything eases in and out (see GraphView and Strands), and
// per-layer activity is smoothed over frames, so the picture breathes with the model instead of
// flickering:
//   recall    recalled memories glow, beams carry them to the recall hub, a spark enters the input
//   route     each expert that switched on for this question lights its whole strand
//   activity  the spine streams and each block glows with how much it changed the token
//   token     the output head pulses softly
//   done      the cited memory glows gold; experts and spine settle back to rest

import { BEAM_COLOURS } from "../scene/effects.js";

const HUB = "hub:recall";
const TOOLS = "hub:tools";
const TOOL_COLOUR = 0xffa86b;
const TOOL_OK = 0x4fd6a0;
const TOOL_FAILED = 0xff6f7f;
const EMBED = "brain:embed";
const HEAD = "brain:head";
const PEAK_DECAY = 0.996;          // per activity frame: slowly forget old peaks
const SMOOTH = 0.35;               // share of a new activity frame mixed into the running value
const RELEASE_MS = 1400;           // the brain settles this long after the answer

const LEVELS = {
  recallHit: 0.9,
  usedMemory: 1.15,
  hub: 0.8,
  embed: 0.7,
  head: 0.45,
  layer: [0.12, 0.6],              // [floor, extra at full activity]
  module: 0.22,
  expertOn: [0.12, 0.18],          // held glow of an expert's nodes: [base, extra by routing score]
  expertOff: 0.06,
  expertActivity: 0.25,
};

export class LiveLights {
  #peaks = new Map();
  #smooth = new Map();
  #expertsOn = new Map();          // expert index -> routing strength 0..1
  #release = null;

  constructor({ view, effects, strands, store }) {
    this.view = view;
    this.effects = effects;
    this.strands = strands;
    this.store = store;
  }

  bind(socket) {
    socket.on("chat.start", () => this.#start());
    socket.on("recall", (e) => this.#recall(e));
    socket.on("route", (e) => this.#route(e));
    socket.on("activity", (e) => this.#activity(e));
    socket.on("token", () => this.view.pulse(HEAD, LEVELS.head));
    socket.on("chat.done", (e) => this.#done(e));
    socket.on("chat.error", () => this.#finish());
    socket.on("tool.call", (e) => this.#toolCall(e));
    socket.on("tool.result", (e) => this.#toolResult(e));
  }

  #toolCall({ name }) {
    const tool = `tool:${name}`;
    this.view.pulse(HEAD, 0.8);
    this.view.pulse(TOOLS, 0.9);
    this.effects.beam(this.#pos(HEAD), this.#pos(TOOLS), TOOL_COLOUR);
    this.effects.beam(this.#pos(TOOLS), this.#pos(tool), TOOL_COLOUR, { delay: 0.35 });
    setTimeout(() => this.view.pulse(tool, 1.1), 700);
  }

  #toolResult({ name, ok }) {
    const tool = `tool:${name}`;
    this.view.pulse(tool, ok ? 0.9 : 1.2);
    this.effects.beam(this.#pos(tool), this.#pos(HEAD), ok ? TOOL_OK : TOOL_FAILED, { delay: 0.1 });
  }

  #pos(id) {
    return this.view.positionOf(id);
  }

  #start() {
    clearTimeout(this.#release);
    this.view.clearLevels("expert");
    this.#expertsOn.clear();
    this.#smooth.clear();
    this.strands.reset();
    this.strands.setSpine(0.35);
  }

  #recall({ items: all }) {
    const items = all.filter((i) => i.source !== "identity");       // in every answer: not news
    const hub = this.#pos(HUB);
    items.forEach((item, i) => {
      const delay = i * 0.08;
      this.view.pulse(item.node_id, LEVELS.recallHit);
      this.effects.beam(this.#pos(item.node_id), hub, BEAM_COLOURS[item.source] ?? BEAM_COLOURS.search, { delay });
    });
    if (items.length) {
      this.view.pulse(HUB, LEVELS.hub);
      const lag = items.length * 0.08 + 0.45;
      this.effects.spark(hub, this.#pos(EMBED), 0xffd36b, { delay: lag, seconds: 0.7, size: 8 });
      setTimeout(() => this.view.pulse(EMBED, LEVELS.embed), (lag + 0.7) * 1000);
    }
  }

  #route({ experts }) {
    for (const ex of experts) {
      const ref = ex.threshold ?? ex.score ?? 1;
      const strength = Math.max(0, Math.min(1, (ex.score ?? ref) / ((ref || 1) * 2)));
      if (ex.on) {
        this.#expertsOn.set(ex.index, strength);
        this.strands.setExpert(ex.index, 0.45 + 0.4 * strength);
      }
      for (const { node_id, value } of ex.layers) {
        const ratio = Math.max(0, Math.min(1, value / ((ref || 1) * 1.5)));
        if (ex.on) this.view.setLevel(node_id, LEVELS.expertOn[0] + LEVELS.expertOn[1] * ratio);
        else this.view.pulse(node_id, LEVELS.expertOff);
      }
    }
  }

  #activity({ phase, layers }) {
    if (phase === "read") this.view.pulse(EMBED, LEVELS.embed);
    const modules = this.store.meta.brain?.modules ?? [];
    let total = 0;
    for (const { node_id, value } of layers) {
      const peak = Math.max((this.#peaks.get(node_id) ?? 0) * PEAK_DECAY, value, 1e-4);
      this.#peaks.set(node_id, peak);
      const smooth = (this.#smooth.get(node_id) ?? 0) * (1 - SMOOTH) + (value / peak) * SMOOTH;
      this.#smooth.set(node_id, smooth);
      total += smooth;
      if (smooth < 0.03) continue;
      this.view.pulse(node_id, LEVELS.layer[0] + LEVELS.layer[1] * smooth);
      const layer = node_id.slice(node_id.lastIndexOf(":") + 1);
      for (const name of modules) this.view.pulse(`brain:module:${layer}:${name}`, LEVELS.module * smooth);
      for (const k of this.#expertsOn.keys()) this.view.pulse(`expert:${k}:${layer}`, LEVELS.expertActivity * smooth);
    }
    if (layers.length) this.strands.setSpine(0.3 + 0.6 * (total / layers.length));
  }

  #done(event) {
    if (event.used_node) {
      this.view.pulse(event.used_node, LEVELS.usedMemory);
      this.effects.beam(this.#pos(event.used_node), this.#pos(HUB), BEAM_COLOURS.used);
    }
    this.#finish();
  }

  #finish() {
    clearTimeout(this.#release);
    this.#release = setTimeout(() => {
      this.view.clearLevels("expert");
      this.#expertsOn.clear();
      this.strands.reset();
    }, RELEASE_MS);
  }
}
