"""What the inspector panel shows for a node: the graph's own data plus a fresh read of the store."""
from __future__ import annotations

from collections import defaultdict

from ..graph.memory_graph import file_id, memory_id
from ..graph.model import Graph, NodeKind
from ..somni.store_reader import StoreReader

MAX_NEIGHBOURS = 60


class NodeNotFound(KeyError):
    pass


def neighbours(graph: Graph, node_id: str) -> dict[str, list[str]]:
    """Linked node ids grouped by link kind."""
    out: dict[str, list[str]] = defaultdict(list)
    for link in graph.links.values():
        if link.source == node_id:
            out[link.kind.value].append(link.target)
        elif link.target == node_id:
            out[link.kind.value].append(link.source)
    return {k: v[:MAX_NEIGHBOURS] for k, v in out.items()}


def _memory_detail(reader: StoreReader, graph: Graph, mid: str) -> dict:
    record = reader.memory(mid)
    if record is None:
        return {"missing": True}
    related = reader.related(mid, record.get("tags") or [])
    return {
        **{k: record.get(k) for k in ("id", "key", "type", "summary", "value", "body", "sources", "strength",
                                       "status", "supersedes", "superseded_by", "created_day", "last_used_day",
                                       "retrieval_count", "consolidated_in", "rejection_reason", "tags")},
        "file_node": file_id(record["episode_id"]) if record.get("has_file") else None,
        "related": [{"node_id": memory_id(r["id"]), "key": r["key"], "shared": r["shared"],
                     "drawn": memory_id(r["id"]) in graph.nodes} for r in related],
    }


def _file_detail(reader: StoreReader, episode_id: int) -> dict:
    record = reader.file(episode_id)
    if record is None:
        return {"missing": True}
    return {**record, "memory_nodes": [memory_id(m) for m in record.pop("memories")]}


def _tag_detail(reader: StoreReader, graph: Graph, node) -> dict:
    tag = node.attrs.get("tag", "")
    if node.attrs.get("virtual"):
        linked = neighbours(graph, node.id).get("tagged", [])
        return {"tag": node.label, "virtual": True, "count": node.attrs.get("count", len(linked)),
                "memories": [{"node_id": n, "key": graph.nodes[n].label} for n in linked if n in graph.nodes]}
    record = reader.tag(tag) or {"tag": tag, "count": 0, "memories": []}
    record["memories"] = [{"node_id": memory_id(m["id"]), "key": m["key"], "summary": m["summary"]}
                          for m in record["memories"]]
    return record


def _brain_detail(graph: Graph, node) -> dict:
    brain = graph.meta.get("brain", {})
    if node.kind is NodeKind.EXPERT:
        k = node.attrs.get("expert", 0)
        experts = brain.get("experts", [])
        expert = experts[k] if k < len(experts) else {}
        return {"expert": k, "layer": node.attrs.get("layer"), "threshold": expert.get("threshold"),
                "calibration": expert.get("calibration") or {}, "source": brain.get("experts_source"),
                "folder": brain.get("experts_dir"), "rank": brain.get("rank")}
    return {"model": brain.get("model"), **node.attrs}


def node_detail(graph: Graph, reader: StoreReader, node_id: str) -> dict:
    node = graph.nodes.get(node_id)
    if node is None:
        raise NodeNotFound(node_id)
    if node.kind is NodeKind.MEMORY:
        detail = _memory_detail(reader, graph, node.attrs["memory_id"])
    elif node.kind is NodeKind.FILE:
        detail = _file_detail(reader, int(node.attrs["episode_id"]))
    elif node.kind is NodeKind.TAG:
        detail = _tag_detail(reader, graph, node)
    elif node.region.value == "brain":
        detail = _brain_detail(graph, node)
    else:
        detail = dict(node.attrs)
    return {"node": node.to_wire(), "detail": detail, "neighbours": neighbours(graph, node_id)}
