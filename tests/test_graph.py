import asyncio
import hashlib
import time
import unittest
from collections import Counter

from somni_studio.events import EventBus, EventType
from somni_studio.graph.brain_graph import EMBED_ID, HEAD_ID, expert_id, layer_id
from somni_studio.graph.memory_graph import HUB_ID, UNTAGGED, file_id, memory_id, tag_id
from somni_studio.graph.model import LinkKind, NodeKind, diff
from somni_studio.graph.service import GraphService, build_graph
from somni_studio.somni.home import FALLBACK_LAYERS

from .helpers import HomeTestCase


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BuildGraphTest(HomeTestCase):
    def test_regions_kinds_and_counts(self):
        g = build_graph(self.settings)
        kinds = Counter(n.kind for n in g.nodes.values())
        self.assertEqual(kinds[NodeKind.MEMORY], 5)
        self.assertEqual(kinds[NodeKind.FILE], 2)
        self.assertEqual(kinds[NodeKind.LAYER], FALLBACK_LAYERS)            # model config not found
        self.assertEqual(kinds[NodeKind.MODULE], FALLBACK_LAYERS * 7)
        self.assertEqual(kinds[NodeKind.EXPERT], FALLBACK_LAYERS * 2)       # two stacked experts
        self.assertEqual(kinds[NodeKind.HUB], 2)                         # recall + tools
        self.assertGreaterEqual(kinds[NodeKind.TOOL], 5)
        for nid in (EMBED_ID, HEAD_ID, HUB_ID, layer_id(0), expert_id(1, FALLBACK_LAYERS - 1)):
            self.assertIn(nid, g.nodes)

    def test_every_link_joins_two_existing_nodes(self):
        g = build_graph(self.settings)
        for link in g.links.values():
            self.assertIn(link.source, g.nodes)
            self.assertIn(link.target, g.nodes)

    def test_memory_structure(self):
        g = build_graph(self.settings)
        mid = next(n.id for n in g.nodes.values() if n.label == "normans/named")
        links = {(link.kind, link.target) for link in g.links.values() if link.source == mid}
        self.assertIn((LinkKind.TAGGED, tag_id("normandy")), links)
        self.assertIn((LinkKind.TAGGED, tag_id("history")), links)
        self.assertTrue(any(k is LinkKind.SOURCE and t.startswith("file:") for k, t in links))
        tree = {(link.source, link.target) for link in g.links.values() if link.kind is LinkKind.TAG_TREE}
        self.assertIn((tag_id("history"), tag_id("normans")), tree)
        self.assertIn((HUB_ID, tag_id("history")), {(link.source, link.target) for link in g.links.values()
                                                     if link.kind is LinkKind.BRIDGE})
        self.assertIn((EMBED_ID, HUB_ID), {(link.source, link.target) for link in g.links.values()})

    def test_untagged_memories_are_grouped_by_subject(self):
        g = build_graph(self.settings)
        node = g.nodes[tag_id(f"{UNTAGGED}/tesla-coil")]
        self.assertTrue(node.attrs["virtual"])
        self.assertEqual(node.label, "tesla-coil")

    def test_memory_attrs_include_somni_salience_tier(self):
        g = build_graph(self.settings)
        tiers = {n.attrs.get("tier") for n in g.nodes.values() if n.kind is NodeKind.MEMORY}
        self.assertEqual(tiers, {"protected"})      # all created within the grace period

    def test_meta_reports_experts_and_warnings(self):
        g = build_graph(self.settings)
        brain = g.meta["brain"]
        self.assertEqual(brain["experts_source"], "active")
        self.assertEqual(brain["checkpoint"], "ckpt-0001")
        self.assertEqual([e["threshold"] for e in brain["experts"]], [None, 0.6])
        self.assertTrue(any("config.json" in w for w in g.meta["warnings"]))

    def test_building_never_writes_to_the_store(self):
        db = self.home / "memory.db"
        before = (_sha(db), db.stat().st_mtime_ns)
        build_graph(self.settings)
        self.assertEqual((_sha(db), db.stat().st_mtime_ns), before)

    def test_layout_is_deterministic(self):
        a, b = build_graph(self.settings), build_graph(self.settings)
        self.assertEqual({k: n.position for k, n in a.nodes.items()}, {k: n.position for k, n in b.nodes.items()})

    def test_missing_home_gives_an_empty_memory_region_and_a_warning(self):
        g = build_graph(self.settings.merged({"somni": {"home": str(self.root / "nope")}}))
        self.assertFalse(any(n.kind is NodeKind.MEMORY for n in g.nodes.values()))
        self.assertTrue(any("not found" in w for w in g.meta["warnings"]))


class DiffTest(HomeTestCase):
    def _add_memory(self):
        from sleepai.memory.store import Memory, MemoryStore
        store = MemoryStore(self.home / "memory.db")
        mem = store.insert(Memory(key="normans/leader", type="fact", summary="Rollo led the Normans.", value="Rollo",
                                  strength="verified", sources=["episode:1"], tags=["history", "normans", "rollo"]))
        store.db.close()
        return mem.id

    def test_diff_reports_exactly_the_new_memory_and_tag(self):
        before = build_graph(self.settings)
        mid = self._add_memory()
        change = diff(before, build_graph(self.settings))
        added = {n["id"] for n in change.added}
        self.assertEqual(added, {memory_id(mid), tag_id("rollo")})
        self.assertFalse(change.removed)
        self.assertTrue(any(link["source"] == memory_id(mid) for link in change.added_links))

    def test_service_publishes_a_diff_when_the_store_changes(self):
        async def scenario():
            bus = EventBus()
            bus.attach(asyncio.get_running_loop())
            q = bus.subscribe()
            service = GraphService(bus, self.settings)
            self.assertTrue(await service.refresh(force=True))
            first = await asyncio.wait_for(q.get(), 1)
            self.assertEqual(first["type"], EventType.GRAPH_SNAPSHOT.value)
            self.assertFalse(await service.refresh(), "no change, no event")
            time.sleep(0.02)
            self._add_memory()
            self.assertTrue(await service.refresh())
            event = await asyncio.wait_for(q.get(), 1)
            self.assertEqual(event["type"], EventType.GRAPH_DIFF.value)
            self.assertEqual(event["base_revision"], first["revision"])
            self.assertEqual(service.revision, first["revision"] + 1)
        asyncio.run(scenario())


class IdsTest(unittest.TestCase):
    def test_ids(self):
        self.assertEqual(file_id(3), "file:3")
        self.assertEqual(memory_id("m1"), "mem:m1")


if __name__ == "__main__":
    unittest.main()
