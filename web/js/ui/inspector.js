// The side panel for the selected node. Memory files open in full, with "Open in editor".

import { $, clear, cloneTemplate, el, fill, fmt, slot } from "../core/dom.js";
import { api } from "../net/api.js";

const TITLES = {
  memory: "Memory", tag: "Tag", file: "Memory file", expert: "Expert", layer: "Layer", module: "Linear layer",
  embed: "Input", head: "Output", hub: "Recall hub",
};

export class Inspector {
  #panel = $("#inspector");
  #body = $("#inspector-body");
  #current = null;
  #token = 0;

  constructor({ store, hud, onSelect, onFocus, onClose }) {
    this.store = store;
    this.hud = hud;
    this.onSelect = onSelect;
    $("#inspector-close").addEventListener("click", () => onClose());
    $("#inspector-focus").addEventListener("click", () => this.#current && onFocus(this.#current));
  }

  get open() {
    return !this.#panel.hidden;
  }

  hide() {
    this.#panel.hidden = true;
    this.#current = null;
  }

  async show(id) {
    const node = this.store.get(id);
    if (!node) return;
    this.#current = id;
    const token = ++this.#token;
    this.#panel.hidden = false;
    const badge = $("#inspector-kind");
    badge.textContent = TITLES[node.kind] ?? node.kind;
    badge.dataset.kind = node.kind;
    $("#inspector-title").textContent = node.label;
    clear(this.#body).appendChild(el("p", { className: "chat-empty", text: "loading…" }));
    try {
      const data = await api.node(id);
      if (token !== this.#token) return;            // a newer selection won
      clear(this.#body);
      this.#render(node, data.detail ?? {}, data.neighbours ?? {});
    } catch (err) {
      if (token !== this.#token) return;
      clear(this.#body).appendChild(el("p", { className: "chat-empty", text: err.message }));
    }
  }

  // ------------------------------------------------------------ building blocks
  #section(title, ...children) {
    const sec = fill(cloneTemplate("tpl-section"), { title });
    const content = slot(sec, "content");
    for (const child of children.flat()) if (child) content.appendChild(child);
    this.#body.appendChild(sec);
    return sec;
  }

  #kv(key, value, { pill = false } = {}) {
    const row = fill(cloneTemplate("tpl-kv"), { key, value: pill ? "" : String(value ?? "–") });
    if (pill) {
      const p = el("span", { className: "status-pill", text: value ?? "–" });
      p.dataset.value = value ?? "";
      slot(row, "value").appendChild(p);
    }
    return row;
  }

  #links(rows) {
    const list = el("ul", { className: "links" });
    for (const { id, label, meta } of rows) {
      const li = fill(cloneTemplate("tpl-link-row"), { label, meta: meta ?? "" });
      const btn = li.querySelector("button");
      if (id && this.store.get(id)) btn.addEventListener("click", () => this.onSelect(id, { fly: true }));
      else btn.disabled = true;
      list.appendChild(li);
    }
    return list;
  }

  #tagChips(tags) {
    const box = el("div", { className: "tags" });
    for (const tag of tags) {
      const chip = fill(cloneTemplate("tpl-chip"), { label: tag });
      chip.dataset.source = "tags";
      const id = `tag:${tag}`;
      if (this.store.get(id)) chip.addEventListener("click", () => this.onSelect(id, { fly: true }));
      box.appendChild(chip);
    }
    return box;
  }

  #labelOf(id) {
    return this.store.get(id)?.label ?? id;
  }

  // ------------------------------------------------------------ per kind
  #render(node, d, nb) {
    if (d.missing) {
      this.#section("Not found", el("p", { text: "This item is no longer in the memory store." }));
      return;
    }
    const render = {
      memory: () => this.#memory(node, d),
      file: () => this.#file(d),
      tag: () => this.#tag(d),
      expert: () => this.#expert(d),
    }[node.kind];
    if (render) render();
    else this.#generic(node, d, nb);
  }

  #memory(node, d) {
    this.#section("Fact", el("p", { className: "insp-summary", text: d.summary }),
      d.value ? this.#kv("value", d.value) : null);
    this.#section("State",
      this.#kv("status", d.status, { pill: true }),
      node.attrs?.tier ? this.#kv("salience tier", node.attrs.tier, { pill: true }) : null,
      this.#kv("strength", d.strength),
      this.#kv("uses", fmt.int(d.retrieval_count)),
      this.#kv("created day", d.created_day),
      this.#kv("last used day", d.last_used_day < 0 ? "never" : d.last_used_day),
      this.#kv("in weights", d.consolidated_in?.length ? d.consolidated_in.join(", ") : "not yet"),
      d.supersedes ? this.#kv("corrects", d.supersedes) : null,
      d.rejection_reason ? this.#kv("note", d.rejection_reason) : null);
    if (d.tags?.length) this.#section("Tags", this.#tagChips(d.tags));
    if (d.file_node) {
      this.#section("Source", this.#links([{ id: d.file_node, label: this.#labelOf(d.file_node), meta: "open memory file" }]));
    }
    if (d.related?.length) {
      this.#section("Linked memories (2+ shared tags)",
        this.#links(d.related.map((r) => ({ id: r.node_id, label: r.key, meta: `${r.shared} shared` }))));
    }
    this.#section("Key", this.#kv("key", d.key), this.#kv("id", d.id),
      this.#kv("sources", (d.sources ?? []).join(", ")));
  }

  #file(d) {
    const reader = fill(cloneTemplate("tpl-file-reader"), { text: d.text, size: fmt.chars(d.text.length) });
    slot(reader, "open").addEventListener("click", async () => {
      try {
        const res = await api.openFile(d.episode_id);
        this.hud.toast(`opened ${res.opened.split(/[\\/]/).pop()}`, "good");
      } catch (err) {
        this.hud.toast(err.message, "error");
      }
    });
    this.#section(`Episode #${d.episode_id}${d.kind ? ` · ${d.kind}` : ""}`, reader);
    if (d.memory_nodes?.length) {
      this.#section(`Memories formed from it (${d.memory_nodes.length})`,
        this.#links(d.memory_nodes.map((id) => ({ id, label: this.#labelOf(id) }))));
    }
  }

  #tag(d) {
    const p = el("p", { text: `${fmt.int(d.count)} memories carry this tag.${d.virtual ? " (No subject tags yet: grouped by the subject in each memory's key.)" : ""}` });
    this.#section("Tag", p);
    this.#section("Memories", this.#links((d.memories ?? []).map((m) => ({ id: m.node_id, label: m.key }))));
  }

  #expert(d) {
    const explain = el("p", {
      text: "One night of sleep trained this expert while the base model and every older expert stayed frozen. "
        + "Before each answer Somni scores the question against it and switches it on only when the score reaches its threshold.",
    });
    const rows = [
      this.#kv("expert", d.expert + 1),
      this.#kv("layer", d.layer),
      this.#kv("threshold", d.threshold === null || d.threshold === undefined ? "always on" : fmt.num(d.threshold)),
      this.#kv("rank", d.rank ?? "–"),
      this.#kv("folder", d.folder ?? "–"),
      this.#kv("source", d.source === "latest" ? "newest trained (not adopted)" : d.source),
    ];
    this.#section("Expert", explain, rows);
    const cal = d.calibration ?? {};
    if (cal.memory && cal.rehearsal) {
      this.#section("Routing calibration (p5 / p50 / p95)",
        this.#kv("memory prompts", cal.memory.map((v) => fmt.num(v, 2)).join(" / ")),
        this.#kv("rehearsal prompts", cal.rehearsal.map((v) => fmt.num(v, 2)).join(" / ")),
        this.#kv("memory routed on", `${Math.round((cal.train_memory_on ?? 0) * 100)}%`),
        this.#kv("rehearsal routed on", `${Math.round((cal.train_rehearsal_on ?? 0) * 100)}%`));
    }
  }

  #generic(node, d, nb) {
    const rows = Object.entries(d).filter(([, v]) => typeof v !== "object").map(([k, v]) => this.#kv(k, v));
    this.#section(TITLES[node.kind] ?? node.kind, rows);
    const linked = Object.entries(nb).flatMap(([kind, ids]) => ids.map((id) => ({ id, label: this.#labelOf(id), meta: kind })));
    if (linked.length) this.#section("Connected to", this.#links(linked.slice(0, 40)));
  }
}
