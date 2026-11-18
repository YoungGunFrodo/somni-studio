// Text in the scene (CSS2D): region titles, top-level tags, the model's landmarks.

import * as THREE from "three";
import { CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";
import { cloneTemplate, fill, modelName } from "../core/dom.js";
import { cssHue } from "./palette.js";

const LAYER_LABEL_EVERY = 4;

export class Labels {
  #group = new THREE.Group();
  #host;

  constructor(stage, host) {
    this.#host = host;
    stage.scene.add(this.#group);
  }

  set visible(on) {
    this.#host.classList.toggle("labels-off", !on);
  }

  rebuild(nodes, meta) {
    for (const obj of [...this.#group.children]) {
      obj.element?.remove();
      this.#group.remove(obj);
    }
    const brain = meta.brain ?? {};
    let brainTop = -Infinity, brainX = 0, memTop = -Infinity, memX = 0, memCount = 0;
    for (const n of nodes.values()) {
      if (n.region === "brain" && n.y > brainTop) {
        brainTop = n.y;
        brainX = n.x;
      }
      if (n.region === "memory") {
        memTop = Math.max(memTop, n.y);
        memX += n.x;
        memCount += 1;
      }
      const text = this.#textFor(n, brain);
      if (!text) continue;
      const cls = n.kind === "tag" ? "root" : n.kind === "layer" ? "minor" : "";
      const label = this.#add(text, [n.x, n.y + n.size * (n.kind === "layer" ? 1 : 1.6) + 1.2, n.z], cls);
      if (n.kind === "tag" && n.hue !== undefined) label.element.style.setProperty("--label-color", cssHue(n.hue));
      if (n.kind === "layer") label.position.x -= 34;
      if (n.kind === "expert") label.position.y += 7;          // above the top of its strand, clear of L27
    }
    if (Number.isFinite(brainTop)) {
      const model = brain.model ? ` · ${modelName(brain.model)}` : "";
      this.#add(`Model${model}`, [brainX, brainTop + 22, 0], "region");
    }
    if (memCount) this.#add("Memory", [memX / memCount, memTop + 18, 0], "region");
    else {
      const hub = nodes.get("hub:recall");
      if (hub) {
        // no memories yet: the camera frames the brain and the hub, so the hint sits by the hub
        this.#add("Memory", [hub.x, hub.y + 16, 0], "region");
        this.#add("empty until the first sleep", [hub.x, hub.y - 9, 0], "minor");
      }
    }
  }

  #textFor(n, brain) {
    switch (n.kind) {
      case "tag":
        return n.attrs?.root ? n.label : null;
      case "hub":
        return n.id === "hub:tools" ? "tools" : "recall";
      case "embed":
        return "input";
      case "head":
        return "output";
      case "layer":
        return n.attrs.layer % LAYER_LABEL_EVERY === 0 || n.attrs.layer === brain.layers - 1 ? `L${n.attrs.layer}` : null;
      case "expert":
        return n.attrs.layer === brain.layers - 1 ? `expert ${n.attrs.expert + 1}` : null;
      default:
        return null;
    }
  }

  #add(text, [x, y, z], cls) {
    const el = fill(cloneTemplate("tpl-label"), { text });
    if (cls) el.classList.add(cls);
    const obj = new CSS2DObject(el);
    obj.position.set(x, y, z);
    this.#group.add(obj);
    return obj;
  }
}
