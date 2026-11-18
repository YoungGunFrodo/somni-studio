"""The graph the UI draws: nodes with fixed 3D positions, typed links, and diffs between versions.

Positions are computed on the server (graph/layout.py), so every client sees the same picture,
a reload never reshuffles it, and the browser only renders.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Region(str, Enum):
    BRAIN = "brain"
    BRIDGE = "bridge"
    MEMORY = "memory"


class NodeKind(str, Enum):
    EMBED = "embed"          # token embedding: where the prompt (and recalled memory) enters
    LAYER = "layer"          # one transformer block
    MODULE = "module"        # a linear layer inside a block (q/k/v/o, gate/up/down)
    HEAD = "head"            # output head: where tokens leave
    EXPERT = "expert"        # one night's expert at one layer (Somni stacked experts)
    HUB = "hub"              # the recall junction between brain and memory, and the tools junction
    TOOL = "tool"            # something the assistant can do (clock, memory, home, email...)
    TAG = "tag"              # subject tag (broad to specific)
    MEMORY = "memory"        # one memory: subject / attribute / value
    FILE = "file"            # memory file: full source text (cold storage)


class LinkKind(str, Enum):
    FLOW = "flow"            # layer -> next layer
    MODULE = "module"        # layer -> its linear layers
    EXPERT = "expert"        # expert -> the layer it attaches to
    EXPERT_CHAIN = "expert_chain"   # an expert's nodes, layer to layer
    BRIDGE = "bridge"        # embedding <-> recall hub <-> top-level tags
    TAG_TREE = "tag_tree"    # broader tag -> narrower tag
    TAGGED = "tagged"        # memory -> tag
    SOURCE = "source"        # memory -> the memory file it was formed from
    TOOL = "tool"            # output head -> tools hub -> each tool


def _r(v: float) -> float:
    return round(float(v), 2)


@dataclass(frozen=True)
class Node:
    id: str
    kind: NodeKind
    region: Region
    label: str
    position: tuple[float, float, float]
    size: float = 1.0
    hue: float | None = None          # 0..1; None = the kind's own colour
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_wire(self) -> dict:
        x, y, z = self.position
        out = {"id": self.id, "kind": self.kind.value, "region": self.region.value, "label": self.label,
               "x": _r(x), "y": _r(y), "z": _r(z), "size": round(self.size, 3)}
        if self.hue is not None:
            out["hue"] = round(self.hue, 4)
        if self.attrs:
            out["attrs"] = self.attrs
        return out


@dataclass(frozen=True)
class Link:
    source: str
    target: str
    kind: LinkKind
    weight: float = 1.0

    @property
    def id(self) -> str:
        return f"{self.kind.value}:{self.source}>{self.target}"

    def to_wire(self) -> dict:
        """No id on the wire: clients derive it the same way (`kind:source>target`), which keeps
        a large graph about 40% smaller."""
        return {"source": self.source, "target": self.target, "kind": self.kind.value,
                "weight": round(self.weight, 3)}


@dataclass
class Graph:
    nodes: dict[str, Node] = field(default_factory=dict)
    links: dict[str, Link] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node

    def add_link(self, link: Link) -> None:
        if link.source in self.nodes and link.target in self.nodes:
            self.links[link.id] = link

    def merge(self, other: Graph) -> None:
        self.nodes.update(other.nodes)
        for link in other.links.values():
            self.add_link(link)

    def to_wire(self, revision: int) -> dict:
        return {"revision": revision, "meta": self.meta,
                "nodes": [n.to_wire() for n in self.nodes.values()],
                "links": [link.to_wire() for link in self.links.values()]}


@dataclass
class GraphDiff:
    added: list[dict] = field(default_factory=list)
    updated: list[dict] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    added_links: list[dict] = field(default_factory=list)
    removed_links: list[str] = field(default_factory=list)
    meta_changed: bool = False

    @property
    def empty(self) -> bool:
        return not (self.added or self.updated or self.removed or self.added_links
                    or self.removed_links or self.meta_changed)

    def to_wire(self, revision: int, meta: dict) -> dict:
        return {"revision": revision, "base_revision": revision - 1, "meta": meta,
                "added": self.added, "updated": self.updated, "removed": self.removed,
                "added_links": self.added_links, "removed_links": self.removed_links}


def diff(old: Graph, new: Graph) -> GraphDiff:
    out = GraphDiff(meta_changed=old.meta != new.meta)
    for nid, node in new.nodes.items():
        before = old.nodes.get(nid)
        if before is None:
            out.added.append(node.to_wire())
        elif before.to_wire() != node.to_wire():
            out.updated.append(node.to_wire())
    out.removed = [nid for nid in old.nodes if nid not in new.nodes]
    out.added_links = [link.to_wire() for lid, link in new.links.items() if lid not in old.links]
    out.removed_links = [lid for lid in old.links if lid not in new.links]
    return out
