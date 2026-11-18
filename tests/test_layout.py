import math
import unittest

from somni_studio.graph.layout import build_tag_tree, fibonacci_sphere, golden_hues, hash_unit, norm, place_tree, sub


class TagTreeTest(unittest.TestCase):
    def test_broad_to_specific_lists_make_a_tree(self):
        tree = build_tag_tree([("history", "normans", "normandy"), ("history", "normans", "origin"),
                               ("science", "tesla")])
        self.assertEqual(set(tree.roots), {"history", "science"})
        self.assertEqual(tree.parent["normans"], "history")
        self.assertEqual(tree.parent["normandy"], "normans")
        self.assertEqual(tree.depth("origin"), 2)
        self.assertEqual(tree.root_of("origin"), "history")
        self.assertEqual(tree.count["normans"], 2)

    def test_most_common_predecessor_wins(self):
        tree = build_tag_tree([("a", "x"), ("a", "x"), ("b", "x")])
        self.assertEqual(tree.parent["x"], "a")

    def test_cycles_are_broken(self):
        tree = build_tag_tree([("a", "b"), ("b", "a")])
        self.assertTrue(tree.roots, "a cycle must still leave a root")
        for tag in ("a", "b"):            # walking up always terminates at a root
            self.assertIn(tree.root_of(tag), tree.roots)

    def test_duplicate_and_empty_tags_are_ignored(self):
        tree = build_tag_tree([("a", "", "a", "b")])
        self.assertEqual(tree.parent, {"a": None, "b": "a"})


class GeometryTest(unittest.TestCase):
    def test_hash_unit_is_stable_and_unit_length(self):
        self.assertEqual(hash_unit("memory-1"), hash_unit("memory-1"))
        self.assertNotEqual(hash_unit("memory-1"), hash_unit("memory-2"))
        self.assertAlmostEqual(norm(hash_unit("x")), 1.0, places=9)

    def test_fibonacci_sphere(self):
        pts = fibonacci_sphere(50)
        self.assertEqual(len(pts), 50)
        self.assertTrue(all(math.isclose(norm(p), 1.0, rel_tol=1e-9) for p in pts))
        self.assertEqual(fibonacci_sphere(0), [])

    def test_hues_are_distinct(self):
        hues = golden_hues(12)
        self.assertEqual(len(set(round(h, 6) for h in hues)), 12)
        self.assertTrue(all(0 <= h < 1 for h in hues))

    def test_adding_a_leaf_tag_moves_nothing_else(self):
        lists = [("history", "normans"), ("science", "tesla")]
        before_tree = build_tag_tree(lists)
        before = place_tree(before_tree, (0, 0, 0), before_tree.count)
        after_tree = build_tag_tree(lists + [("history", "rome")])
        after = place_tree(after_tree, (0, 0, 0), after_tree.count)
        self.assertEqual(before["science"], after["science"])
        self.assertEqual(before["tesla"], after["tesla"])
        self.assertIn("rome", after)

    def test_children_sit_outside_their_parent(self):
        tree = build_tag_tree([("history", "normans", "origin")])
        pos = place_tree(tree, (0, 0, 0), tree.count)
        self.assertGreater(norm(sub(pos["normans"], pos["history"])), 1.0)
        self.assertGreater(norm(sub(pos["origin"], pos["normans"])), 1.0)


if __name__ == "__main__":
    unittest.main()
