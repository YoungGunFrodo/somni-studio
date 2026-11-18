"""The brain region: the model drawn as a tower of transformer blocks with Somni's experts around it.

Drawing individual weights is impossible (a 1.7B model has 1.7 billion), so the unit is the one
Somni itself works with: each block, the linear layers inside it that experts attach to, and one
node per expert per block. Every adopted night adds a new expert, so the tower grows a new ring.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..somni.home import ExpertSet, ModelShape
from .layout import Vec, golden_hues
from .model import Graph, Link, LinkKind, Node, NodeKind, Region

EMBED_ID = "brain:embed"
HEAD_ID = "brain:head"
TOOLS_HUB_ID = "hub:tools"


def tool_id(name: str) -> str:
    return f"tool:{name}"


def layer_id(i: int) -> str:
    return f"brain:layer:{i}"


def module_id(i: int, name: str) -> str:
    return f"brain:module:{i}:{name}"


def expert_id(k: int, i: int) -> str:
    return f"expert:{k}:{i}"


@dataclass(frozen=True)
class BrainGeometry:
    centre: Vec = (-200.0, 0.0, 0.0)
    layer_spacing: float = 7.0
    module_radius: float = 9.5
    expert_radius: float = 21.0
    expert_step: float = 6.5         # each newer expert ring sits further out
    twist: float = 0.13              # radians per layer: the tower spirals slightly
    expert_twist: float = 0.34       # experts wind around the tower (about 1.5 turns over 28 layers)
    expert_phase: float = 2.4        # each newer expert starts this far further round (radians)
    tools_offset: Vec = (70.0, -10.0, 0.0)  # the tools hub, relative to the output head
    tools_radius: float = 24.0              # tools fan out around it (names show on hover)


ATTENTION = {"q_proj", "k_proj", "v_proj", "o_proj"}


def build_brain_graph(shape: ModelShape, experts: ExpertSet, geo: BrainGeometry = BrainGeometry(),
                      tools: tuple[tuple[str, str], ...] = ()) -> Graph:
    """`tools`: (name, title) of everything the assistant can call; drawn as the brain's "hands"."""
    g = Graph()
    cx, cy, cz = geo.centre
    n = shape.layers

    def y_of(i: int) -> float:
        return cy + (i - (n - 1) / 2) * geo.layer_spacing

    bottom, top = y_of(0) - geo.layer_spacing * 2.5, y_of(n - 1) + geo.layer_spacing * 2.5

    g.add_node(Node(EMBED_ID, NodeKind.EMBED, Region.BRAIN, "Input embedding", (cx, bottom, cz), 2.8,
                    attrs={"role": "prompt and recalled memories enter here"}))
    g.add_node(Node(HEAD_ID, NodeKind.HEAD, Region.BRAIN, "Output head", (cx, top, cz), 2.8,
                    attrs={"role": "next-token prediction"}))

    modules = list(experts.targets)
    for i in range(n):
        y = y_of(i)
        g.add_node(Node(layer_id(i), NodeKind.LAYER, Region.BRAIN, f"Layer {i}", (cx, y, cz), 1.8,
                        attrs={"layer": i}))
        g.add_link(Link(EMBED_ID if i == 0 else layer_id(i - 1), layer_id(i), LinkKind.FLOW))
        for j, name in enumerate(modules):
            a = 2 * math.pi * j / max(1, len(modules)) + geo.twist * i
            pos = (cx + geo.module_radius * math.cos(a), y, cz + geo.module_radius * math.sin(a))
            block = "attention" if name in ATTENTION else "mlp"
            g.add_node(Node(module_id(i, name), NodeKind.MODULE, Region.BRAIN, f"Layer {i} · {name}", pos, 0.7,
                            attrs={"layer": i, "module": name, "block": block}))
            g.add_link(Link(layer_id(i), module_id(i, name), LinkKind.MODULE, 0.5))
    g.add_link(Link(layer_id(n - 1), HEAD_ID, LinkKind.FLOW))

    hues = golden_hues(len(experts.experts), offset=0.08)
    for expert, hue in zip(experts.experts, hues, strict=True):
        k = expert.index
        radius = geo.expert_radius + geo.expert_step * k
        phase = geo.expert_phase * k     # each newer expert starts a third of a turn further round
        for i in range(n):
            a = phase + geo.expert_twist * i
            pos = (cx + radius * math.cos(a), y_of(i), cz + radius * math.sin(a))
            attrs = {"expert": k, "layer": i, "routed": expert.threshold is not None}
            if expert.threshold is not None:
                attrs["threshold"] = round(expert.threshold, 4)
            g.add_node(Node(expert_id(k, i), NodeKind.EXPERT, Region.BRAIN, f"Expert {k + 1} · layer {i}", pos,
                            1.0, hue, attrs))
            g.add_link(Link(expert_id(k, i), layer_id(i), LinkKind.EXPERT, 0.4))
            if i:
                g.add_link(Link(expert_id(k, i - 1), expert_id(k, i), LinkKind.EXPERT_CHAIN, 0.6))

    if tools:
        hx, hy, hz = cx + geo.tools_offset[0], top + geo.tools_offset[1], cz + geo.tools_offset[2]
        g.add_node(Node(TOOLS_HUB_ID, NodeKind.HUB, Region.BRAIN, "Tools", (hx, hy, hz), 2.0,
                        attrs={"role": "what the assistant can do: it reaches its tools from here"}))
        g.add_link(Link(HEAD_ID, TOOLS_HUB_ID, LinkKind.TOOL, 0.8))
        for j, (name, title) in enumerate(tools):
            a = math.pi * (0.15 + 0.7 * j / max(1, len(tools) - 1))          # an arc above the hub
            pos = (hx + geo.tools_radius * math.cos(a) * 0.9, hy + geo.tools_radius * math.sin(a), hz)
            g.add_node(Node(tool_id(name), NodeKind.TOOL, Region.BRAIN, title, pos, 1.0,
                            attrs={"tool": name}))
            g.add_link(Link(TOOLS_HUB_ID, tool_id(name), LinkKind.TOOL, 0.5))

    g.meta = {
        "model": shape.name,
        "layers": n,
        "layers_exact": shape.exact,
        "modules": modules,
        "experts": [{"index": e.index, "threshold": e.threshold, "calibration": e.calibration}
                    for e in experts.experts],
        "experts_source": experts.source,
        "experts_dir": experts.directory.name if experts.directory else None,
        "checkpoint": experts.checkpoint,
        "rank": experts.rank,
        "tools": [name for name, _title in tools],
        # how expert strands are placed, so the UI can draw the next one while it is being trained
        "geometry": {"centre": list(geo.centre), "expert_radius": geo.expert_radius,
                     "expert_step": geo.expert_step, "expert_twist": geo.expert_twist,
                     "expert_phase": geo.expert_phase},
    }
    return g
