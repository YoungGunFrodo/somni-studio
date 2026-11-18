// Short-lived effects: beams from recalled memories to the recall hub, and sparks that carry
// recalled memory along the bridge into the model's input.

import * as THREE from "three";

const BEAM_POINTS = 48;
const BEAM_DRAW_S = 0.45;
const BEAM_HOLD_S = 0.9;
const BEAM_FADE_S = 1.6;
const SPARK_S = 0.85;
const MAX_BEAMS = 48;
const GATHER_S = 1.05;           // a new node's material falling in
const FLASH_S = 1.5;
const RING_S = 1.7;
const MOTES = 16;
const MAX_BIRTHS = 12;           // at once (a sleep can add dozens: the rest just grow in)

export const BEAM_COLOURS = {
  search: 0x7aa2ff,
  tags: 0xb796ff,
  file: 0xe8ecff,
  used: 0xffd36b,
};

function curve(from, to) {
  const mid = new THREE.Vector3().addVectors(from, to).multiplyScalar(0.5);
  mid.y += from.distanceTo(to) * 0.28;
  return new THREE.QuadraticBezierCurve3(from.clone(), mid, to.clone());
}

export class Effects {
  #stage;
  #beams = [];
  #sparks = [];
  #births = [];
  #sparkTexture = Effects.#makeSparkTexture();

  constructor(stage) {
    this.#stage = stage;
    stage.onTick((dt) => this.#tick(dt));
  }

  /** A beam that draws itself from `from` to `to`, holds, then fades. */
  beam(from, to, colour = BEAM_COLOURS.search, { delay = 0 } = {}) {
    if (!from || !to) return;
    if (this.#beams.length >= MAX_BEAMS) this.#dispose(this.#beams.shift());
    const points = curve(from, to).getPoints(BEAM_POINTS - 1);
    const geo = new THREE.BufferGeometry().setFromPoints(points);
    geo.setDrawRange(0, 0);
    const mat = new THREE.LineBasicMaterial({ color: colour, transparent: true, opacity: 1, depthWrite: false,
      blending: THREE.AdditiveBlending, toneMapped: false });
    const line = new THREE.Line(geo, mat);
    line.frustumCulled = false;
    this.#stage.scene.add(line);
    this.#beams.push({ line, age: -delay });
    this.spark(from, to, colour, { delay, seconds: BEAM_DRAW_S });
  }

  /** A bright point travelling from `from` to `to`. */
  spark(from, to, colour = 0xffd36b, { delay = 0, seconds = SPARK_S, size = 7 } = {}) {
    if (!from || !to) return;
    const mat = new THREE.SpriteMaterial({ map: this.#sparkTexture, color: colour, transparent: true,
      depthWrite: false, blending: THREE.AdditiveBlending, toneMapped: false });
    const sprite = new THREE.Sprite(mat);
    sprite.scale.setScalar(size);
    sprite.visible = false;
    this.#stage.scene.add(sprite);
    this.#sparks.push({ sprite, path: curve(from, to), age: -delay, seconds });
  }

  /** A star forming where a new node appears: motes fall in, it ignites, a ring rolls outward. */
  birth(position, size = 1, colour = 0xffd36b) {
    if (!position || this.#births.length >= MAX_BIRTHS) return false;
    const group = new THREE.Group();
    group.position.copy(position);
    const motes = [];
    for (let k = 0; k < MOTES; k++) {
      const dir = new THREE.Vector3().randomDirection();
      const sprite = this.#sprite(colour, size * (0.9 + Math.random() * 0.8));
      sprite.material.opacity = 0;
      group.add(sprite);
      motes.push({ sprite, from: dir.multiplyScalar(size * (9 + Math.random() * 7)), lag: Math.random() * 0.25,
        swirl: (Math.random() - 0.5) * 2.4 });
    }
    const flash = this.#sprite(0xffffff, 0.01);
    flash.material.opacity = 0;
    group.add(flash);
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.82, 1, 64),
      new THREE.MeshBasicMaterial({ color: colour, transparent: true, opacity: 0, depthWrite: false,
        side: THREE.DoubleSide, blending: THREE.AdditiveBlending, toneMapped: false }));
    group.add(ring);
    this.#stage.scene.add(group);
    this.#births.push({ group, motes, flash, ring, size, age: 0 });
    return true;
  }

  #sprite(colour, scale) {
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: this.#sparkTexture, color: colour,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, toneMapped: false }));
    sprite.scale.setScalar(scale);
    return sprite;
  }

  #tickBirths(dt) {
    const camera = this.#stage.camera;
    this.#births = this.#births.filter((b) => {
      b.age += dt;
      for (const m of b.motes) {                    // fall inward, faster near the end, with a little swirl
        const t = Math.min(1, Math.max(0, (b.age - m.lag) / (GATHER_S - m.lag)));
        const pull = 1 - t * t * t;
        m.sprite.position.copy(m.from).multiplyScalar(pull)
          .applyAxisAngle(THREE.Object3D.DEFAULT_UP, m.swirl * t);
        m.sprite.material.opacity = t >= 1 ? 0 : Math.min(1, t * 4) * 0.85;
      }
      const lit = b.age - GATHER_S;                 // ignition
      if (lit > 0) {
        const f = lit / FLASH_S;
        b.flash.scale.setScalar(b.size * (3 + 13 * Math.min(1, f * 5)) * (1 - 0.5 * Math.min(1, f)));
        b.flash.material.opacity = f < 1 ? (f < 0.12 ? f / 0.12 : 1 - (f - 0.12) / 0.88) : 0;
        const r = lit / RING_S;
        b.ring.scale.setScalar(b.size * (1.5 + 13 * (1 - (1 - Math.min(1, r)) ** 3)));
        b.ring.material.opacity = r < 1 ? 0.75 * (1 - r) : 0;
        b.ring.quaternion.copy(camera.quaternion);
      }
      if (lit > Math.max(FLASH_S, RING_S)) {
        this.#stage.scene.remove(b.group);
        for (const m of b.motes) m.sprite.material.dispose();
        b.flash.material.dispose();
        b.ring.geometry.dispose();
        b.ring.material.dispose();
        return false;
      }
      return true;
    });
  }

  #tick(dt) {
    if (this.#births.length) this.#tickBirths(dt);
    this.#beams = this.#beams.filter((b) => {
      b.age += dt;
      if (b.age < 0) return true;
      const drawn = Math.min(1, b.age / BEAM_DRAW_S);
      b.line.geometry.setDrawRange(0, Math.ceil(drawn * BEAM_POINTS));
      const fadeAt = BEAM_DRAW_S + BEAM_HOLD_S;
      b.line.material.opacity = b.age < fadeAt ? 1 : Math.max(0, 1 - (b.age - fadeAt) / BEAM_FADE_S);
      if (b.age > fadeAt + BEAM_FADE_S) {
        this.#dispose(b);
        return false;
      }
      return true;
    });
    this.#sparks = this.#sparks.filter((s) => {
      s.age += dt;
      if (s.age < 0) return true;
      const t = s.age / s.seconds;
      if (t >= 1) {
        this.#stage.scene.remove(s.sprite);
        s.sprite.material.dispose();
        return false;
      }
      s.sprite.visible = true;
      s.sprite.position.copy(s.path.getPoint(t));
      s.sprite.material.opacity = Math.sin(Math.PI * t) * 0.9 + 0.1;
      return true;
    });
  }

  #dispose(beam) {
    this.#stage.scene.remove(beam.line);
    beam.line.geometry.dispose();
    beam.line.material.dispose();
  }

  static #makeSparkTexture() {
    const size = 64;
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = size;
    const ctx = canvas.getContext("2d");
    const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
    g.addColorStop(0, "rgba(255,255,255,1)");
    g.addColorStop(0.25, "rgba(255,255,255,0.65)");
    g.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, size, size);
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    return texture;
  }
}
