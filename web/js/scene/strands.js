// Living strands through the brain: one glowing ribbon per expert (each night of learning winds
// around the model), and a central spine from input to output. Particles stream along them: slowly
// and faintly at rest, quickly and brightly while that part of the model is working.

import * as THREE from "three";

const PARTICLES_PER_LAYER = 4;
const IDLE = { opacity: 0.55, tube: 0.2, speed: 0.014 };
const ACTIVE = { opacity: 1.0, tube: 0.5, speed: 0.1 };
const EASE_PER_SECOND = 3.5;            // how fast a strand brightens and dims (no flicker)

function sparkTexture() {
  const size = 64;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d");
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  g.addColorStop(0, "rgba(255,255,255,1)");
  g.addColorStop(0.3, "rgba(255,255,255,0.5)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

class Strand {
  constructor(points, colour, { radius, particleSize, texture }) {
    this.curve = new THREE.CatmullRomCurve3(points, false, "centripetal");
    this.level = 0;
    this.target = 0;
    this.phase = Math.random();
    this.group = new THREE.Group();

    this.tube = new THREE.Mesh(
      new THREE.TubeGeometry(this.curve, Math.max(32, points.length * 8), radius, 8, false),
      new THREE.MeshBasicMaterial({ color: colour, transparent: true, opacity: IDLE.tube, depthWrite: false,
        blending: THREE.AdditiveBlending, toneMapped: false }));
    this.group.add(this.tube);

    const count = Math.max(24, Math.round(points.length * PARTICLES_PER_LAYER));
    this.offsets = Float32Array.from({ length: count }, (_, i) => (i + Math.random() * 0.6) / count);
    this.positions = new Float32Array(count * 3);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(this.positions, 3));
    this.points = new THREE.Points(geo, new THREE.PointsMaterial({
      color: colour, map: texture, size: particleSize, transparent: true, opacity: IDLE.opacity,
      depthWrite: false, blending: THREE.AdditiveBlending, toneMapped: false, sizeAttenuation: true }));
    this.points.frustumCulled = false;
    this.group.add(this.points);
    this.tmp = new THREE.Vector3();
    this.reach = 1;               // how much of the strand exists (a strand being trained grows)
  }

  setReach(fraction) {
    this.reach = Math.max(0.001, Math.min(1, fraction));
    const index = this.tube.geometry.index;
    if (index) this.tube.geometry.setDrawRange(0, Math.ceil((index.count * this.reach) / 6) * 6);
  }

  tick(dt) {
    this.level += (this.target - this.level) * Math.min(1, dt * EASE_PER_SECOND);
    const k = this.level;
    const speed = IDLE.speed + (ACTIVE.speed - IDLE.speed) * k;
    this.phase = (this.phase + dt * speed) % 1;
    this.points.material.opacity = IDLE.opacity + (ACTIVE.opacity - IDLE.opacity) * k;
    this.tube.material.opacity = IDLE.tube + (ACTIVE.tube - IDLE.tube) * k;
    for (let i = 0; i < this.offsets.length; i++) {
      this.curve.getPointAt(((this.offsets[i] + this.phase) % 1) * this.reach, this.tmp);
      this.positions[i * 3] = this.tmp.x;
      this.positions[i * 3 + 1] = this.tmp.y;
      this.positions[i * 3 + 2] = this.tmp.z;
    }
    this.points.geometry.attributes.position.needsUpdate = true;
  }

  dispose() {
    this.tube.geometry.dispose();
    this.tube.material.dispose();
    this.points.geometry.dispose();
    this.points.material.dispose();
  }
}

export class Strands {
  #root = new THREE.Group();
  #experts = new Map();         // expert index -> Strand
  #spine = null;
  #ghost = null;                // the expert being trained right now (during sleep)
  #dreaming = false;
  #texture = sparkTexture();

  constructor(stage) {
    stage.scene.add(this.#root);
    stage.onTick((dt) => {
      this.#spine?.tick(dt);
      this.#ghost?.tick(dt);
      for (const s of this.#experts.values()) s.tick(dt);
    });
  }

  /** Show the next expert growing up the tower (sleep: training). `points` from its geometry. */
  showGhost(points, colour = 0xb796ff) {
    if (this.#ghost) return;
    const path = points.map((p) => new THREE.Vector3(p.x, p.y, p.z));
    this.#ghost = new Strand(path, new THREE.Color(colour), { radius: 0.75, particleSize: 3.6,
      texture: this.#texture });
    this.#ghost.setReach(0.02);
    this.#ghost.target = 0.6;
    this.#root.add(this.#ghost.group);
  }

  setGhost({ reach, level } = {}) {
    if (!this.#ghost) return;
    if (reach !== undefined) this.#ghost.setReach(reach);
    if (level !== undefined) this.#ghost.target = Math.max(0, Math.min(1, level));
  }

  hideGhost() {
    if (!this.#ghost) return;
    this.#root.remove(this.#ghost.group);
    this.#ghost.dispose();
    this.#ghost = null;
  }

  get hasGhost() {
    return this.#ghost !== null;
  }

  /** Dreaming: every strand drifts slowly, as if the whole brain were breathing. */
  setDreaming(on) {
    this.#dreaming = on;
    for (const s of this.#experts.values()) s.target = on ? 0.35 : 0;
    if (this.#spine) this.#spine.target = on ? 0.25 : 0;
  }


  rebuild(nodes) {
    for (const s of [this.#spine, ...this.#experts.values()]) {
      if (!s) continue;
      this.#root.remove(s.group);
      s.dispose();
    }
    this.#experts.clear();
    this.#spine = null;

    const byExpert = new Map();
    const layers = [];
    let embed = null, head = null;
    for (const n of nodes.values()) {
      if (n.kind === "expert") {
        if (!byExpert.has(n.attrs.expert)) byExpert.set(n.attrs.expert, { hue: n.hue, nodes: [] });
        byExpert.get(n.attrs.expert).nodes.push(n);
      } else if (n.kind === "layer") layers.push(n);
      else if (n.kind === "embed") embed = n;
      else if (n.kind === "head") head = n;
    }
    if (embed && head && layers.length) {
      layers.sort((a, b) => a.attrs.layer - b.attrs.layer);
      const pts = [embed, ...layers, head].map((n) => new THREE.Vector3(n.x, n.y, n.z));
      this.#spine = new Strand(pts, new THREE.Color(0x9fb4ff), { radius: 0.6, particleSize: 3.2, texture: this.#texture });
      this.#root.add(this.#spine.group);
    }
    for (const [k, { hue, nodes: list }] of byExpert) {
      list.sort((a, b) => a.attrs.layer - b.attrs.layer);
      if (list.length < 2) continue;
      const colour = new THREE.Color().setHSL(hue ?? 0.42, 0.8, 0.62);
      const strand = new Strand(list.map((n) => new THREE.Vector3(n.x, n.y, n.z)), colour,
        { radius: 0.75, particleSize: 3.6, texture: this.#texture });
      this.#experts.set(k, strand);
      this.#root.add(strand.group);
    }
  }

  /** 0..1: how busy an expert is (eased; never flickers). */
  setExpert(index, level) {
    const s = this.#experts.get(index);
    if (s) s.target = Math.max(0, Math.min(1, level));
  }

  setSpine(level) {
    if (this.#spine) this.#spine.target = Math.max(0, Math.min(1, level));
  }

  reset() {
    this.setSpine(this.#dreaming ? 0.25 : 0);
    for (const s of this.#experts.values()) s.target = this.#dreaming ? 0.35 : 0;
  }
}
