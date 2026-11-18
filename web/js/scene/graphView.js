// Draws the graph: one instanced mesh per node kind (so 100k nodes cost a handful of draw calls),
// two line batches for links, and per-node "heat" that makes nodes glow and swell when used.
//
//   heat   a pulse that fades by itself (recall hit, token, activity frame); it eases toward its
//          target instead of jumping, so a stream of pulses reads as a smooth glow, not a flicker
//   level  a held glow until cleared (an expert that is switched on for the current answer)
//   dim    brightness outside a highlight (search results, a selected node's neighbourhood)
//   born   0 -> 1 while a new node forms (it grows out of nothing, overshoots, settles); below 0 it
//          is not visible yet (the particles of its birth effect are still gathering)

import * as THREE from "three";
import { LINK_COLOURS, SCENE, nodeColour } from "./palette.js";

const SHAPES = {
  memory: () => new THREE.IcosahedronGeometry(1, 1),
  tag: () => new THREE.IcosahedronGeometry(1, 0),
  file: () => new THREE.OctahedronGeometry(1, 0),
  layer: () => new THREE.CylinderGeometry(1, 1, 0.22, 32),
  module: () => new THREE.IcosahedronGeometry(1, 0),
  expert: () => new THREE.OctahedronGeometry(1, 0),
  embed: () => new THREE.DodecahedronGeometry(1, 0),
  head: () => new THREE.DodecahedronGeometry(1, 0),
  hub: () => new THREE.IcosahedronGeometry(1, 2),
  tool: () => new THREE.BoxGeometry(1.3, 1.3, 1.3),
};
// Non-uniform scale per kind: layers are flat plates that span their ring of linear layers.
const SHAPE_SCALE = { layer: [4.2, 1, 4.2] };
const FINE_LINKS = new Set(["tagged", "source"]);   // the dense ones the "Links" toggle hides

const tmpMatrix = new THREE.Matrix4();
const tmpPos = new THREE.Vector3();
const tmpScale = new THREE.Vector3();
const tmpQuat = new THREE.Quaternion();
const tmpColour = new THREE.Color();
const BIRTH_S = 1.9;                     // from the first gathering particle to a settled node
export const BIRTH_DELAY = 0.55;         // of that, how long the node stays hidden (in units of born)

/** Scale of a forming node: nothing, then a quick swell past full size, then settling. */
function bornScale(b) {
  if (b <= 0) return 0;
  if (b >= 1) return 1;
  const c = 2.2;                         // ease-out-back overshoot
  const x = b - 1;
  return 1 + (c + 1) * x * x * x + c * x * x;
}

class NodeGroup {
  constructor(kind, nodes) {
    this.kind = kind;
    this.ids = nodes.map((n) => n.id);
    const count = nodes.length;
    const material = new THREE.MeshBasicMaterial({ color: 0xffffff, toneMapped: false });
    this.mesh = new THREE.InstancedMesh((SHAPES[kind] ?? SHAPES.memory)(), material, count);
    this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.mesh.userData.group = this;
    this.mesh.frustumCulled = false;
    this.pos = new Float32Array(count * 3);
    this.size = new Float32Array(count);
    this.base = new Float32Array(count * 3);
    this.heat = new Float32Array(count);
    this.target = new Float32Array(count);
    this.level = new Float32Array(count);
    this.dim = new Float32Array(count).fill(1);
    this.born = new Float32Array(count).fill(1);
    this.active = new Set();
    this.shape = SHAPE_SCALE[kind] ?? [1, 1, 1];
    nodes.forEach((n, i) => {
      this.pos.set([n.x, n.y, n.z], i * 3);
      this.size[i] = n.size;
      nodeColour(n, tmpColour);
      this.base.set([tmpColour.r, tmpColour.g, tmpColour.b], i * 3);
    });
    for (let i = 0; i < count; i++) this.write(i);
    this.mesh.instanceMatrix.needsUpdate = true;
    if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
    this.mesh.computeBoundingSphere();
  }

  /** Write instance i's matrix and colour from its current heat / level / dim. */
  write(i) {
    const intensity = Math.min(1.6, Math.max(this.heat[i], this.level[i]));
    const d = this.dim[i];
    const grow = 1 + SCENE.pulseScale * Math.min(1, intensity);
    const s = this.size[i] * grow * bornScale(this.born[i]);
    tmpPos.fromArray(this.pos, i * 3);
    tmpScale.set(s * this.shape[0], s * this.shape[1], s * this.shape[2]);
    this.mesh.setMatrixAt(i, tmpMatrix.compose(tmpPos, tmpQuat, tmpScale));
    const lift = 1 + intensity * SCENE.litBoost;
    const forming = this.born[i] < 1 ? (1 - Math.max(0, this.born[i])) * 0.9 : 0;   // white-hot while young
    const white = intensity * 0.35 + forming;
    tmpColour.setRGB(this.base[i * 3] * d * lift + white, this.base[i * 3 + 1] * d * lift + white,
      this.base[i * 3 + 2] * d * lift + white);
    this.mesh.setColorAt(i, tmpColour);
  }

  dispose() {
    this.mesh.geometry.dispose();
    this.mesh.material.dispose();
    this.mesh.dispose();
  }
}

export class GraphView {
  #stage;
  #root = new THREE.Group();
  #groups = new Map();          // kind -> NodeGroup
  #index = new Map();           // node id -> [group, i]
  #nodes = new Map();
  #links = new Map();
  #lines = { structural: null, fine: null };
  #linksVisible = true;
  #highlight = null;
  #markers;

  constructor(stage) {
    this.#stage = stage;
    stage.scene.add(this.#root);
    this.#markers = {
      hover: this.#makeRing(0xffffff, 0.45),
      selected: this.#makeRing(0xffd36b, 0.95),
    };
    stage.onTick((dt, t) => this.#tick(dt, t));
  }

  // ------------------------------------------------------------ building
  setGraph(nodes, links) {
    const carried = new Map();                     // keep glows across rebuilds
    for (const [id, [g, i]] of this.#index) {
      if (g.heat[i] > 0.01 || g.target[i] > 0.01 || g.level[i] > 0 || g.born[i] < 1) {
        carried.set(id, [g.heat[i], g.target[i], g.level[i], g.born[i]]);
      }
    }
    for (const g of this.#groups.values()) {
      this.#root.remove(g.mesh);
      g.dispose();
    }
    this.#groups.clear();
    this.#index.clear();
    this.#nodes = nodes;
    this.#links = links;

    const byKind = new Map();
    for (const n of nodes.values()) {
      if (!byKind.has(n.kind)) byKind.set(n.kind, []);
      byKind.get(n.kind).push(n);
    }
    for (const [kind, list] of byKind) {
      const g = new NodeGroup(kind, list);
      this.#groups.set(kind, g);
      g.ids.forEach((id, i) => this.#index.set(id, [g, i]));
      this.#root.add(g.mesh);
    }
    for (const [id, [heat, target, level, born]] of carried) {
      const hit = this.#index.get(id);
      if (!hit) continue;
      hit[0].heat[hit[1]] = heat;
      hit[0].target[hit[1]] = target;
      hit[0].level[hit[1]] = level;
      hit[0].born[hit[1]] = born;
      if (born < 1) hit[0].active.add(hit[1]);
      hit[0].active.add(hit[1]);
    }
    this.#buildLinks();
    if (this.#highlight) this.setHighlight(this.#highlight);
  }

  #buildLinks() {
    for (const key of ["structural", "fine"]) {
      const old = this.#lines[key];
      if (old) {
        this.#root.remove(old);
        old.geometry.dispose();
        old.material.dispose();
      }
    }
    const buckets = { structural: [], fine: [] };
    for (const link of this.#links.values()) {
      const a = this.#nodes.get(link.source);
      const b = this.#nodes.get(link.target);
      if (a && b) buckets[FINE_LINKS.has(link.kind) ? "fine" : "structural"].push([link, a, b]);
    }
    for (const [key, list] of Object.entries(buckets)) {
      const pos = new Float32Array(list.length * 6);
      const col = new Float32Array(list.length * 6);
      list.forEach(([link, a, b], i) => {
        pos.set([a.x, a.y, a.z, b.x, b.y, b.z], i * 6);
        const [r, g, bl, alpha] = LINK_COLOURS[link.kind] ?? [1, 1, 1, 0.1];
        const k = alpha * (this.#highlight && !(this.#highlight.has(a.id) && this.#highlight.has(b.id)) ? 0.12 : 1);
        col.set([r * k, g * k, bl * k, r * k, g * k, bl * k], i * 6);
      });
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
      const mat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false,
        blending: THREE.AdditiveBlending, toneMapped: false });
      const lines = new THREE.LineSegments(geo, mat);
      lines.frustumCulled = false;
      lines.visible = key === "structural" || this.#linksVisible;
      this.#lines[key] = lines;
      this.#root.add(lines);
    }
  }

  #makeRing(colour, opacity) {
    const mesh = new THREE.Mesh(new THREE.RingGeometry(1.35, 1.6, 48),
      new THREE.MeshBasicMaterial({ color: colour, transparent: true, opacity, depthWrite: false,
        side: THREE.DoubleSide, toneMapped: false }));
    mesh.visible = false;
    mesh.userData.node = null;
    this.#stage.scene.add(mesh);
    return mesh;
  }

  // ------------------------------------------------------------ queries
  get nodes() {
    return this.#nodes;
  }

  positionOf(id) {
    const n = this.#nodes.get(id);
    return n ? new THREE.Vector3(n.x, n.y, n.z) : null;
  }

  /** Bounding box of every node, or of one region ("brain" | "memory"). */
  bounds(region) {
    const box = new THREE.Box3();
    for (const n of this.#nodes.values()) {
      if (!region || n.region === region) box.expandByPoint(tmpPos.set(n.x, n.y, n.z));
    }
    return box;
  }

  /** The node under a ray, or null. */
  pick(raycaster) {
    const meshes = [...this.#groups.values()].map((g) => g.mesh);
    const hits = raycaster.intersectObjects(meshes, false);
    for (const hit of hits) {
      const g = hit.object.userData.group;
      if (g && hit.instanceId !== undefined && g.dim[hit.instanceId] > SCENE.dimmed + 0.01) return g.ids[hit.instanceId];
    }
    return hits.length ? hits[0].object.userData.group.ids[hits[0].instanceId] : null;
  }

  // ------------------------------------------------------------ effects
  pulse(id, amount = 1) {
    const hit = this.#index.get(id);
    if (!hit) return;
    const [g, i] = hit;
    g.target[i] = Math.max(g.target[i], amount);
    g.active.add(i);
  }

  /** A new node: hidden while its birth effect gathers, then it grows in and glows. */
  grow(id) {
    const hit = this.#index.get(id);
    if (!hit) return;
    const [g, i] = hit;
    g.born[i] = -BIRTH_DELAY;
    g.target[i] = Math.max(g.target[i], 1.3);
    g.write(i);
    g.active.add(i);
  }

  setLevel(id, value) {
    const hit = this.#index.get(id);
    if (!hit) return;
    const [g, i] = hit;
    g.level[i] = value;
    g.active.add(i);
  }

  clearLevels(kind) {
    for (const g of this.#groups.values()) {
      if (kind && g.kind !== kind) continue;
      for (let i = 0; i < g.level.length; i++) {
        if (g.level[i] > 0) {
          g.level[i] = 0;
          g.active.add(i);
        }
      }
    }
  }

  /** Dim everything outside `ids` (a Set), or restore with null. */
  setHighlight(ids) {
    this.#highlight = ids && ids.size ? ids : null;
    for (const g of this.#groups.values()) {
      for (let i = 0; i < g.ids.length; i++) {
        g.dim[i] = !this.#highlight || this.#highlight.has(g.ids[i]) ? 1 : SCENE.dimmed;
        g.write(i);
      }
      g.mesh.instanceMatrix.needsUpdate = true;
      if (g.mesh.instanceColor) g.mesh.instanceColor.needsUpdate = true;
    }
    this.#buildLinks();
  }

  setLinksVisible(on) {
    this.#linksVisible = on;
    if (this.#lines.fine) this.#lines.fine.visible = on;
  }

  setHover(id) {
    this.#placeMarker(this.#markers.hover, id);
  }

  setSelected(id) {
    this.#placeMarker(this.#markers.selected, id);
  }

  #placeMarker(marker, id) {
    const n = id ? this.#nodes.get(id) : null;
    marker.visible = Boolean(n);
    marker.userData.node = n ?? null;
    if (n) marker.position.set(n.x, n.y, n.z);
  }

  // ------------------------------------------------------------ frame
  #tick(dt, t) {
    const decay = Math.exp(-SCENE.decayPerSecond * dt);
    const ease = Math.min(1, dt * SCENE.attackPerSecond);
    for (const g of this.#groups.values()) {
      if (!g.active.size) continue;
      for (const i of g.active) {
        if (g.born[i] < 1) {
          g.born[i] = Math.min(1, g.born[i] + dt * (1 + BIRTH_DELAY) / BIRTH_S);
          if (g.born[i] <= 0) {                     // still gathering: hold the glow for the ignition
            g.write(i);
            continue;
          }
        }
        g.target[i] *= decay;
        g.heat[i] += (g.target[i] - g.heat[i]) * ease;
        if (g.heat[i] < 0.004 && g.target[i] < 0.004) g.heat[i] = g.target[i] = 0;
        g.write(i);
        if (g.heat[i] === 0 && g.level[i] === 0 && g.born[i] >= 1) g.active.delete(i);
      }
      g.mesh.instanceMatrix.needsUpdate = true;
      if (g.mesh.instanceColor) g.mesh.instanceColor.needsUpdate = true;
    }
    const camera = this.#stage.camera;
    for (const marker of Object.values(this.#markers)) {
      const n = marker.userData.node;
      if (!marker.visible || !n) continue;
      const hit = this.#index.get(n.id);
      const grow = hit ? 1 + SCENE.pulseScale * Math.min(1, Math.max(hit[0].heat[hit[1]], hit[0].level[hit[1]])) : 1;
      const breathe = marker === this.#markers.selected ? 1 + 0.08 * Math.sin(t * 3.2) : 1;
      const wide = n.kind === "layer" ? 4.2 : 1;
      marker.scale.setScalar(n.size * grow * breathe * wide * 1.15);
      marker.quaternion.copy(camera.quaternion);
    }
  }
}
