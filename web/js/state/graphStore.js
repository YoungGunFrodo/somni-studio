// The client's copy of the graph. Snapshots replace it; diffs patch it in place.
// Emits "changed" with {reason: "snapshot" | "diff", added: string[]} after either.

import { Emitter } from "../core/emitter.js";

export const linkId = (link) => `${link.kind}:${link.source}>${link.target}`;

export class GraphStore extends Emitter {
  nodes = new Map();
  links = new Map();
  meta = {};
  revision = 0;

  loadSnapshot(snapshot) {
    this.nodes = new Map(snapshot.nodes.map((n) => [n.id, n]));
    this.links = new Map(snapshot.links.map((l) => [linkId(l), l]));
    this.meta = snapshot.meta ?? {};
    this.revision = snapshot.revision;
    this.emit("changed", { reason: "snapshot", added: [] });
  }

  /** Apply a diff. Returns false when it does not follow our revision (caller must resync). */
  applyDiff(diff) {
    if (diff.base_revision !== this.revision) return false;
    for (const id of diff.removed) this.nodes.delete(id);
    for (const n of diff.added) this.nodes.set(n.id, n);
    for (const n of diff.updated) this.nodes.set(n.id, n);
    for (const id of diff.removed_links) this.links.delete(id);
    for (const l of diff.added_links) this.links.set(linkId(l), l);
    this.meta = diff.meta ?? this.meta;
    this.revision = diff.revision;
    this.emit("changed", { reason: "diff", added: diff.added.map((n) => n.id) });
    return true;
  }

  get(id) {
    return this.nodes.get(id);
  }

  /** Case-insensitive label search over memories, tags and files, best matches first. */
  search(query, limit = 12) {
    const q = query.trim().toLowerCase();
    if (q.length < 2) return [];
    const scored = [];
    for (const n of this.nodes.values()) {
      if (n.region !== "memory") continue;
      const hay = `${n.label} ${n.attrs?.summary ?? ""}`.toLowerCase();
      const at = hay.indexOf(q);
      if (at < 0) continue;
      const kindBoost = n.kind === "tag" ? 0 : n.kind === "file" ? 1 : 2;
      scored.push([at === 0 ? 0 : at < n.label.length ? 1 : 3, kindBoost, n.label.length, n]);
    }
    scored.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
    return scored.slice(0, limit).map((s) => s[3]);
  }
}
