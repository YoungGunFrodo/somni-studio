// Every colour, size and timing constant of the scene, in one place.

import * as THREE from "three";

export const SCENE = {
  background: 0x05060b,
  fogDensity: 0.0011,
  bloom: { strength: 0.9, radius: 0.5, threshold: 0.82 },
  decayPerSecond: 1.8,          // how fast a pulse fades (e^-k·t)
  attackPerSecond: 9,           // how fast a node brightens toward a pulse (no hard flashes)
  pulseScale: 0.55,             // a fully lit node grows by this fraction
  litBoost: 1.1,                // a fully lit node's colour moves this far toward white, past 1 for bloom
  dimmed: 0.03,                 // linear brightness outside a highlight (about 20% on screen)
};

// Base colours by node kind. Memory-region nodes carry a hue from their top-level tag instead.
const KIND_COLOURS = {
  embed: 0xf4f6ff,
  head: 0xf4f6ff,
  layer: 0x8fa6ff,
  module_attention: 0x6c8cff,
  module_mlp: 0x47cfc0,
  expert: 0x4fd6a0,
  hub: 0xffd36b,
  tool: 0xffa86b,
  tag: 0xb796ff,
  memory: 0x7aa2ff,
  file: 0xe8ecff,
};

export const LEGEND = [
  { label: "Layer", colour: KIND_COLOURS.layer },
  { label: "Attention", colour: KIND_COLOURS.module_attention },
  { label: "MLP", colour: KIND_COLOURS.module_mlp },
  { label: "Expert (a night)", colour: KIND_COLOURS.expert },
  { label: "Recall hub", colour: KIND_COLOURS.hub },
  { label: "Tool", colour: KIND_COLOURS.tool },
  { label: "Tag", colour: KIND_COLOURS.tag },
  { label: "Memory", colour: KIND_COLOURS.memory },
  { label: "Memory file", colour: KIND_COLOURS.file },
];

// Linear multipliers (sRGB output brightens them): condensed reads ~65%, faded ~35% on screen.
const TIER_LIGHTNESS = { protected: 1, condensed: 0.38, faded: 0.11 };

const scratch = new THREE.Color();

/** The resting colour of a node (THREE.Color, linear). */
export function nodeColour(node, out = new THREE.Color()) {
  const { kind, hue, attrs = {} } = node;
  if (kind === "module") {
    out.setHex(attrs.block === "mlp" ? KIND_COLOURS.module_mlp : KIND_COLOURS.module_attention);
  } else if (hue !== undefined && hue !== null) {
    const sat = kind === "file" ? 0.28 : kind === "tag" ? 0.78 : 0.66;
    const light = kind === "file" ? 0.66 : kind === "tag" ? 0.58 : 0.5;
    out.setHSL(hue, sat, light);
    if (kind === "memory") out.multiplyScalar(0.72);     // resting memories stay under the glow threshold
  } else {
    out.setHex(KIND_COLOURS[kind] ?? 0xffffff);
  }
  if (kind === "memory") {
    if (attrs.status === "disputed") out.lerp(scratch.setHex(0xff6f7f), 0.75);
    out.multiplyScalar(TIER_LIGHTNESS[attrs.tier] ?? 1);
  }
  return out;
}

/** CSS colour for a hue (labels of top-level tags). */
export function cssHue(hue) {
  return `hsl(${Math.round(hue * 360)} 80% 72%)`;
}

export const KIND_HEX = KIND_COLOURS;

// Link colours (additive blending: darker = fainter).
export const LINK_COLOURS = {
  flow: [0.55, 0.62, 1.0, 0.55],
  module: [0.35, 0.45, 0.9, 0.18],
  expert: [0.3, 0.84, 0.63, 0.22],
  expert_chain: [0.3, 0.84, 0.63, 0.4],
  bridge: [1.0, 0.83, 0.42, 0.55],
  tool: [1.0, 0.66, 0.42, 0.45],
  tag_tree: [0.72, 0.6, 1.0, 0.22],
  tagged: [0.48, 0.64, 1.0, 0.035],
  source: [0.9, 0.92, 1.0, 0.025],
};
