// Sleep, made visible. The sleep process narrates each stage ("sleep.phase" events) and the
// brain shows it:
//
//   formation   memories shimmer while conversations become memories; new ones flare as they form
//   training    the next expert grows up the tower as a violet strand, step by step
//   dreaming    it quizzes itself: each memory it dreams about sends a violet beam into the brain;
//               answers it got right glow softly, ones it got wrong glow amber and are practised
//   evaluation  a scan sweeps up and down the tower while the tests run
//
// Everything here is driven by what the sleep actually does: the memories in the beams are the
// ones Somni asked itself about, and the strand grows with its real training steps.

const DREAM = 0xb796ff;
const MISSED = 0xffb35c;
const HUB = "hub:recall";
const EMBED = "brain:embed";
const STEP_MS = { shimmer: 240, dream: 170, scan: 55 };

export class Dreams {
  #timers = new Map();
  #queue = [];
  #scan = 0;
  #scanDir = 1;

  constructor({ view, effects, strands, store }) {
    this.view = view;
    this.effects = effects;
    this.strands = strands;
    this.store = store;
  }

  bind(socket) {
    socket.on("sleep.phase", (e) => this.phase(e));
    socket.on("sleep.status", (e) => {
      if (e.state !== "sleeping" && e.state !== "waking") this.stop();
    });
  }

  /** On page load while a sleep is running: pick up where it is. */
  resume(sleep) {
    if (!sleep || (sleep.state !== "sleeping" && sleep.state !== "waking")) return;
    const phase = sleep.phase ?? {};
    if (phase.phase) this.phase({ phase: phase.phase, state: "start" });
    if (phase.total) this.phase({ phase: phase.phase, state: "progress", step: phase.step, total: phase.total });
  }

  phase(e) {
    const key = `${e.phase}:${e.state}`;
    switch (key) {
      case "formation:start":
        this.#loop("shimmer", () => this.#shimmer());
        break;
      case "formation:end":
        this.#stop("shimmer");
        for (const id of e.created ?? []) this.view.pulse(`mem:${id}`, 0.9);
        break;
      case "training:start":
        this.#ghost(0.03);
        break;
      case "training:end":
        this.#ghost(1, 0.55);
        break;
      case "dreaming:start":
        document.body.classList.add("dreaming");
        this.strands.setDreaming(true);
        this.#ghost(1, 0.5);
        break;
      case "dreaming:asking":
        this.#queue = (e.memories ?? []).map((id) => `mem:${id}`).filter((id) => this.store.get(id));
        this.#loop("dream", () => this.#dream());
        break;
      case "dreaming:checked":
        for (const [id, right] of e.results ?? []) this.#result(`mem:${id}`, right);
        break;
      case "dreaming:end":
        this.#stop("dream");
        document.body.classList.remove("dreaming");
        this.strands.setDreaming(false);
        break;
      case "evaluation:start":
        this.#loop("scan", () => this.#scanStep());
        break;
      case "evaluation:end":
        this.#stop("scan");
        break;
      default:
        if (e.state === "progress" && e.total) {
          // training steps: the strand grows; during dreaming they are practice, so it pulses
          if (e.phase === "dreaming") this.strands.setGhost({ level: 0.5 + 0.4 * Math.random() });
          else this.#ghost(Math.max(0.03, e.step / e.total));
        }
    }
  }

  stop() {
    for (const name of [...this.#timers.keys()]) this.#stop(name);
    this.#queue = [];
    document.body.classList.remove("dreaming");
    this.strands.setDreaming(false);
    this.strands.hideGhost();
  }

  // ------------------------------------------------------------ pieces
  #loop(name, fn) {
    this.#stop(name);
    this.#timers.set(name, setInterval(fn, STEP_MS[name]));
  }

  #stop(name) {
    clearInterval(this.#timers.get(name));
    this.#timers.delete(name);
  }

  #memories() {
    return [...this.store.nodes.values()].filter((n) => n.kind === "memory");
  }

  #shimmer() {
    const all = this.#memories();
    for (let i = 0; i < Math.min(6, all.length); i++) {
      this.view.pulse(all[Math.floor(Math.random() * all.length)].id, 0.25);
    }
  }

  #dream() {
    if (!this.#queue.length) {
      this.#stop("dream");
      return;
    }
    const id = this.#queue.shift();
    this.#queue.push(id);                            // keep cycling until the answers are checked
    const from = this.view.positionOf(id);
    const hub = this.view.positionOf(HUB);
    this.view.pulse(id, 0.7);
    this.effects.beam(from, hub, DREAM);
    if (Math.random() < 0.35) this.effects.spark(hub, this.view.positionOf(EMBED), DREAM, { seconds: 0.8, size: 7 });
  }

  #result(id, right) {
    if (!this.store.get(id)) return;
    if (right) {
      this.view.pulse(id, 0.45);
    } else {
      this.view.pulse(id, 1.0);
      this.effects.beam(this.view.positionOf(id), this.view.positionOf(HUB), MISSED);
    }
  }

  #scanStep() {
    const layers = this.store.meta.brain?.layers ?? 0;
    if (!layers) return;
    this.view.pulse(`brain:layer:${this.#scan}`, 0.75);
    this.#scan += this.#scanDir;
    if (this.#scan <= 0 || this.#scan >= layers - 1) this.#scanDir *= -1;
    this.#scan = Math.max(0, Math.min(layers - 1, this.#scan));
  }

  /** Grow the next expert's strand to `reach` (0..1) with a glow `level`. */
  #ghost(reach, level = 0.6) {
    if (!this.strands.hasGhost) {
      const points = this.#nextExpertPath();
      if (!points.length) return;
      this.strands.showGhost(points, DREAM);
    }
    this.strands.setGhost({ reach, level });
  }

  #nextExpertPath() {
    const brain = this.store.meta.brain ?? {};
    const geo = brain.geometry;
    if (!geo) return [];
    const k = (brain.experts ?? []).length;            // the new expert's index
    const radius = geo.expert_radius + geo.expert_step * k;
    const layers = [...this.store.nodes.values()].filter((n) => n.kind === "layer")
      .sort((a, b) => a.attrs.layer - b.attrs.layer);
    return layers.map((n, i) => {
      const a = geo.expert_phase * k + geo.expert_twist * i;
      return { x: n.x + radius * Math.cos(a), y: n.y, z: n.z + radius * Math.sin(a) };
    });
  }
}
