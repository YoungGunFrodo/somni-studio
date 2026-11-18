"""The memory region: subject tags as a tree, memories around their most specific tag, memory
files next to the memories formed from them, and the bridge that joins it all to the brain."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass

from ..somni.store_reader import MemoryRecord, StoreSnapshot
from .brain_graph import EMBED_ID
from .layout import (
    TreeSpacing,
    Vec,
    add,
    build_tag_tree,
    centroid,
    golden_hues,
    hash_unit,
    place_leaf,
    place_tree,
    scale,
    sub,
    unit,
)
from .model import Graph, Link, LinkKind, Node, NodeKind, Region

HUB_ID = "hub:recall"
UNTAGGED = "~untagged"            # virtual root for memories without subject tags
SUMMARY_CHARS = 220

TierFn = Callable[[MemoryRecord, int], str]


def tag_id(tag: str) -> str:
    return f"tag:{tag}"


def memory_id(mid: str) -> str:
    return f"mem:{mid}"


def file_id(episode_id: int) -> str:
    return f"file:{episode_id}"


@dataclass(frozen=True)
class MemoryGeometry:
    centre: Vec = (120.0, 0.0, 0.0)
    hub: Vec = (-60.0, 0.0, 0.0)
    file_offset: float = 5.5          # files sit a little outside their memories
    spacing: TreeSpacing = TreeSpacing()


def effective_tags(m: MemoryRecord) -> tuple[str, ...]:
    """A memory's tags; an untagged memory is grouped by its subject (the first part of its key),
    so stores that never ran tagging still form clusters."""
    if m.tags:
        return m.tags
    subject = m.key.split("/", 1)[0] or "misc"
    return (UNTAGGED, f"{UNTAGGED}/{subject}")


def _tag_label(tag: str) -> str:
    if tag == UNTAGGED:
        return "untagged"
    return tag.split("/", 1)[1] if tag.startswith(UNTAGGED + "/") else tag


def _short(text: str, n: int = SUMMARY_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def build_memory_graph(snap: StoreSnapshot, tier_of: TierFn | None = None,
                       geo: MemoryGeometry = MemoryGeometry()) -> Graph:
    g = Graph()
    tags_of = {m.id: effective_tags(m) for m in snap.memories}
    tree = build_tag_tree(tags_of.values())
    tag_pos = place_tree(tree, geo.centre, tree.count, geo.spacing)
    root_hue = dict(zip(sorted(tree.roots), golden_hues(len(tree.roots)), strict=True))
    hue_of_tag = {t: root_hue[tree.root_of(t)] for t in tree.parent}

    for tag, pos in tag_pos.items():
        depth, count = tree.depth(tag), tree.count[tag]
        size = min(6.0, (1.6 if depth == 0 else 0.85) + 0.32 * math.sqrt(count))
        g.add_node(Node(tag_id(tag), NodeKind.TAG, Region.MEMORY, _tag_label(tag), pos, size, hue_of_tag[tag],
                        {"tag": tag, "count": count, "depth": depth, "root": tree.root_of(tag) == tag,
                         "virtual": tag.startswith(UNTAGGED)}))
        parent = tree.parent.get(tag)
        if parent is not None:
            g.add_link(Link(tag_id(parent), tag_id(tag), LinkKind.TAG_TREE, 1.0 / (1 + depth)))

    anchor_of = {mid: next((t for t in reversed(tags) if t in tag_pos), None) for mid, tags in tags_of.items()}
    crowd = Counter(anchor_of.values())
    mem_pos: dict[str, Vec] = {}
    for m in snap.memories:
        anchor = anchor_of[m.id]
        pos = place_leaf(tag_pos[anchor], geo.centre, m.id, crowd[anchor], geo.spacing) if anchor else geo.centre
        mem_pos[m.id] = pos
        attrs = {"memory_id": m.id, "key": m.key, "summary": _short(m.summary), "status": m.status,
                 "strength": m.strength, "uses": m.retrieval_count, "consolidated": m.consolidated,
                 "tags": list(m.tags)}
        if tier_of is not None:
            attrs["tier"] = tier_of(m, snap.today)
        size = 0.55 + 0.09 * math.sqrt(m.retrieval_count)
        g.add_node(Node(memory_id(m.id), NodeKind.MEMORY, Region.MEMORY, m.key, pos, size,
                        hue_of_tag.get(anchor) if anchor else None, attrs))
        for tag in tags_of[m.id]:
            if tag in tag_pos:
                g.add_link(Link(memory_id(m.id), tag_id(tag), LinkKind.TAGGED, 1.0 if tag == anchor else 0.35))

    members: dict[int, list[str]] = defaultdict(list)
    for mid, eid in snap.source_of.items():
        if mid in mem_pos:
            members[eid].append(mid)
    orphan_radius = geo.spacing.root_radius * 1.45
    for f in snap.files:
        mids = members.get(f.episode_id, [])
        mid_point = centroid(mem_pos[m] for m in mids)
        if mid_point is None:
            pos = add(geo.centre, scale(hash_unit(f"file:{f.episode_id}"), orphan_radius))
        else:
            pos = add(mid_point, scale(unit(sub(mid_point, geo.centre)), geo.file_offset))
        hues = Counter(hue_of_tag.get(anchor_of.get(m)) for m in mids if anchor_of.get(m))
        hue = hues.most_common(1)[0][0] if hues else None
        g.add_node(Node(file_id(f.episode_id), NodeKind.FILE, Region.MEMORY, f.title, pos, 1.15, hue,
                        {"episode_id": f.episode_id, "title": f.title, "chars": f.chars, "memories": len(mids)}))
        for mid in mids:
            g.add_link(Link(memory_id(mid), file_id(f.episode_id), LinkKind.SOURCE, 0.5))

    g.add_node(Node(HUB_ID, NodeKind.HUB, Region.BRIDGE, "Recall", geo.hub, 2.6,
                    attrs={"role": "memories and memory files recalled for a question pass through here"}))
    for root in tree.roots:
        g.add_link(Link(HUB_ID, tag_id(root), LinkKind.BRIDGE, 0.8))

    g.meta = {
        "store": dict(snap.totals),
        "drawn_memories": len(snap.memories),
        "truncated": snap.truncated,
        "today": snap.today,
        "tags": len(tag_pos),
        "roots": [_tag_label(r) for r in tree.roots],
    }
    return g


def bridge_link() -> Link:
    """Recalled memory enters the model with the prompt, at the input embedding."""
    return Link(EMBED_ID, HUB_ID, LinkKind.BRIDGE, 1.0)
