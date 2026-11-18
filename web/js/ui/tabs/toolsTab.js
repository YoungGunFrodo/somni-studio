// Tools tab: everything the assistant can do, whether it is ready, and how it is used.

import { cloneTemplate, el, fill } from "../../core/dom.js";
import { api } from "../../net/api.js";
import { card, para } from "./widgets.js";

const INVOCATION = {
  automatic: "used automatically",
  user: "you start it",
  planned: "not yet used by the assistant",
};

export class ToolsTab {
  title = "Tools";

  constructor({ refresh }) {
    this.refresh = refresh;
  }

  async render(body) {
    const { tools } = await api.tools();
    body.appendChild(para("What the assistant can do. Memory recall and learning run on every conversation. "
      + "Tools from connections are wired up and tested here; letting the assistant call them on its own is the "
      + "next step."));
    const groups = new Map();
    for (const t of tools) {
      if (!groups.has(t.group)) groups.set(t.group, []);
      groups.get(t.group).push(t);
    }
    for (const [group, list] of groups) {
      const ready = list.filter((t) => t.ready).length;
      const box = card(group, `${ready}/${list.length} ready`);
      const ul = el("ul", { className: "tool-list" });
      for (const t of list) {
        const li = fill(cloneTemplate("tpl-tool-row"), {
          title: t.title, sub: t.ready ? t.description : `${t.description} Needs: ${t.needs}.`,
          tag: INVOCATION[t.invocation] ?? t.invocation,
        });
        li.dataset.state = t.ready ? "ready" : "not_configured";
        li.querySelector(".tool-tag").dataset.kind = t.invocation;
        li.title = t.name;
        ul.appendChild(li);
      }
      box.body.appendChild(ul);
      body.appendChild(box.root);
    }
  }

  onEvent(type) {
    if (type === "connection") this.refresh();       // a connection's tools became ready (or not)
  }
}
