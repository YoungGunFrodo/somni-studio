"""Deterministic 3D layout.

Design goals: (1) the same store always gives the same picture; (2) a new memory appears near
its subject without moving anything else; (3) cost grows linearly with the store, so a store of
100k memories lays out as fast as it reads.

So there is no force simulation. Tags form a tree (Somni tags run broad -> specific, so the tag
before a tag in a memory's list is its parent), the tree is laid out radially outward from the
memory region's centre, and every direction comes from a hash of the node id: stable across runs
and independent of what else is in the store.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

Vec = tuple[float, float, float]

GOLDEN_ANGLE = math.pi * (3 - math.sqrt(5))


def add(a: Vec, b: Vec) -> Vec:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale(a: Vec, s: float) -> Vec:
    return (a[0] * s, a[1] * s, a[2] * s)


def norm(a: Vec) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def unit(a: Vec, fallback: Vec = (0.0, 1.0, 0.0)) -> Vec:
    n = norm(a)
    return scale(a, 1.0 / n) if n > 1e-9 else fallback


def centroid(points: Iterable[Vec]) -> Vec | None:
    pts = list(points)
    if not pts:
        return None
    return scale((sum(p[0] for p in pts), sum(p[1] for p in pts), sum(p[2] for p in pts)), 1.0 / len(pts))


def hash_unit(key: str) -> Vec:
    """A uniformly distributed unit vector chosen by `key` (stable across processes)."""
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    u = int.from_bytes(digest[:4], "little") / 2**32
    v = int.from_bytes(digest[4:], "little") / 2**32
    z = 2 * u - 1
    r = math.sqrt(max(0.0, 1 - z * z))
    phi = 2 * math.pi * v
    return (r * math.cos(phi), z, r * math.sin(phi))


def hash_fraction(key: str) -> float:
    return int.from_bytes(hashlib.blake2b(key.encode("utf-8"), digest_size=4).digest(), "little") / 2**32


def fibonacci_sphere(n: int) -> list[Vec]:
    """n well-spread unit vectors."""
    if n <= 0:
        return []
    if n == 1:
        return [(0.0, 1.0, 0.0)]
    out = []
    for i in range(n):
        y = 1 - 2 * (i + 0.5) / n
        r = math.sqrt(max(0.0, 1 - y * y))
        theta = GOLDEN_ANGLE * i
        out.append((r * math.cos(theta), y, r * math.sin(theta)))
    return out


def golden_hues(n: int, offset: float = 0.58) -> list[float]:
    """n hues spread by the golden ratio, so neighbours in the list never look alike."""
    return [(offset + i * 0.6180339887) % 1.0 for i in range(n)]


# ---------------------------------------------------------------- tag tree
@dataclass(frozen=True)
class TagTree:
    parent: dict[str, str | None]       # tag -> broader tag (None for a root)
    roots: tuple[str, ...]
    count: dict[str, int]               # tag -> memories carrying it

    def depth(self, tag: str) -> int:
        d, seen = 0, set()
        while (p := self.parent.get(tag)) is not None and tag not in seen:
            seen.add(tag)
            tag, d = p, d + 1
        return d

    def root_of(self, tag: str) -> str:
        seen = set()
        while (p := self.parent.get(tag)) is not None and tag not in seen:
            seen.add(tag)
            tag = p
        return tag


def build_tag_tree(tag_lists: Iterable[Sequence[str]]) -> TagTree:
    """Each tag's parent is the tag that most often comes right before it. A tag that comes first
    more often than it follows anything is a root. Cycles (a before b in some memories, b before a
    in others) are broken by making the more common tag a root."""
    count: Counter = Counter()
    first: Counter = Counter()
    before: dict[str, Counter] = defaultdict(Counter)
    for tags in tag_lists:
        tags = list(dict.fromkeys(t for t in tags if t))
        count.update(tags)
        if tags:
            first[tags[0]] += 1
        for prev, tag in zip(tags, tags[1:], strict=False):     # consecutive pairs
            before[tag][prev] += 1
    parent: dict[str, str | None] = {}
    for tag in count:
        votes = before.get(tag)
        if not votes:
            parent[tag] = None
            continue
        best, n = max(votes.items(), key=lambda kv: (kv[1], count[kv[0]], kv[0]))
        parent[tag] = None if first[tag] > n else best
    for tag in sorted(count, key=lambda t: (-count[t], t)):      # break cycles
        seen, cur = {tag}, parent[tag]
        while cur is not None:
            if cur in seen:
                parent[tag] = None
                break
            seen.add(cur)
            cur = parent[cur]
    roots = tuple(sorted((t for t, p in parent.items() if p is None), key=lambda t: (-count[t], t)))
    return TagTree(parent, roots, dict(count))


# ---------------------------------------------------------------- radial tree placement
@dataclass(frozen=True)
class TreeSpacing:
    root_radius: float = 74.0          # roots sit on a sphere of this radius around the centre
    child_base: tuple[float, ...] = (17.0, 7.0, 4.0)   # parent-child distance by child depth
    child_growth: tuple[float, ...] = (1.6, 0.7, 0.4)  # extra distance per sqrt(subtree size)
    outward: tuple[float, ...] = (0.75, 0.45, 0.2)     # pull of the outward direction by depth
    leaf_distance: float = 2.4         # memory -> its tag
    leaf_growth: float = 0.12          # per sqrt(memories around that tag)


def _level(values: tuple[float, ...], depth: int) -> float:
    return values[min(depth, len(values)) - 1] if depth > 0 else 0.0


def place_tree(tree: TagTree, centre: Vec, subtree_size: dict[str, int],
               spacing: TreeSpacing = TreeSpacing()) -> dict[str, Vec]:
    """Roots evenly on a sphere (ordered by name, so adding a small tag elsewhere never moves
    them); each child out from its parent along a hashed direction biased away from the centre."""
    children: dict[str, list[str]] = defaultdict(list)
    for tag, p in tree.parent.items():
        if p is not None:
            children[p].append(tag)
    pos: dict[str, Vec] = {}
    for root, direction in zip(sorted(tree.roots), fibonacci_sphere(len(tree.roots)), strict=True):
        pos[root] = add(centre, scale(direction, spacing.root_radius))
    stack = [(root, 1) for root in sorted(tree.roots)]
    while stack:
        tag, depth = stack.pop()
        here = pos[tag]
        outward = unit(sub(here, centre), hash_unit(tag))
        for child in sorted(children.get(tag, ())):
            bias = _level(spacing.outward, depth)
            direction = unit(add(scale(outward, bias), scale(hash_unit(child), 1 - bias * 0.5)), outward)
            dist = _level(spacing.child_base, depth) + _level(spacing.child_growth, depth) * math.sqrt(
                subtree_size.get(child, 1))
            pos[child] = add(here, scale(direction, dist))
            stack.append((child, depth + 1))
    return pos


def place_leaf(anchor: Vec, centre: Vec, key: str, crowd: int, spacing: TreeSpacing = TreeSpacing()) -> Vec:
    """A memory near its most specific tag, on the outer side of it."""
    outward = unit(sub(anchor, centre), hash_unit(key))
    direction = unit(add(scale(outward, 0.35), hash_unit(key)), outward)
    jitter = 0.75 + 0.5 * hash_fraction(key)
    return add(anchor, scale(direction, (spacing.leaf_distance + spacing.leaf_growth * math.sqrt(crowd)) * jitter))
