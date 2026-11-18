// Connections tab: what Somni can reach (smart home, Google Home, the web, email). Each card shows
// its live status, how to set it up, what you can then say, and (if it has settings) a form.
// Secret fields are never filled in by the server; leaving one empty keeps the stored value.

import { el } from "../../core/dom.js";
import { api } from "../../net/api.js";
import { DevicesCard } from "./devicesCard.js";
import { busy, button, card, field, input, para, row, select, statusLine } from "./widgets.js";

const STATE_TEXT = { connected: "Connected", not_configured: "Not connected", error: "Problem", checking: "Checking…" };
const HOME_ASSISTANT = "home_assistant";

export class ConnectionsTab {
  title = "Connections";
  #status = new Map();          // integration id -> status line element

  constructor({ hud, refresh }) {
    this.hud = hud;
    this.refresh = refresh;
  }

  async render(body) {
    const { connections } = await api.connections();
    const connected = connections.filter((c) => c.status.state === "connected").length;
    body.appendChild(para(`Devices first, then services (${connected} of ${connections.length} connected). `
      + "Everything you enter stays on this computer; passwords and tokens are never sent back to the browser. "
      + "Once something is connected, just ask: Somni picks the right tool."));
    this.#status.clear();
    body.appendChild(new DevicesCard({ hud: this.hud }).render());
    for (const c of connections) body.appendChild(this.#card(c));
  }

  #card(c) {
    const box = card(c.title, c.category);
    const status = this.#statusLine(c.status);
    this.#status.set(c.id, status);
    box.body.append(status, para(c.description));
    if (c.status.state !== "connected" && c.setup.length) box.body.appendChild(this.#steps(c.setup));
    if (c.fields.length) this.#form(c, box.body);
    else this.#derived(c, box.body);
    if (c.examples.length) {
      const list = el("ul", { className: "examples" });
      for (const example of c.examples) list.appendChild(el("li", { text: `“${example}”` }));
      box.body.append(el("p", { className: "examples-title", text: "Then try saying:" }), list);
    }
    return box.root;
  }

  #steps(steps) {
    const list = el("ol", { className: "setup-steps" });
    for (const step of steps) list.appendChild(el("li", { text: step }));
    return list;
  }

  #form(c, body) {
    const inputs = new Map();
    for (const f of c.fields) {
      let control;
      if (f.kind === "choice") {
        control = select(f.options.map((o) => ({ value: o, label: o })), c.values[f.key] ?? f.default);
      } else {
        const type = f.kind === "secret" ? "password" : f.kind === "int" ? "number" : f.kind === "url" ? "url" : "text";
        const value = f.kind === "secret" ? "" : c.values[f.key] ?? f.default ?? "";
        const placeholder = f.kind === "secret" && c.values[`${f.key}_set`] ? "saved (leave empty to keep)" : f.placeholder;
        control = input(type, String(value), { placeholder });
      }
      inputs.set(f.key, control);
      body.appendChild(field({ label: f.label + (f.required ? "" : " (optional)"), help: f.help, control }));
    }
    const save = button(c.configured ? "Save & test" : "Connect", { primary: true });
    save.addEventListener("click", () => busy(save, this.hud, async () => {
      const values = Object.fromEntries([...inputs].map(([k, node]) => [k, node.value]));
      const { status: s } = await api.saveConnection(c.id, values);
      this.#show(c.id, s);
      this.hud.toast(`${c.title}: ${s.message}`, s.state === "connected" ? "good" : "error");
      if (s.state === "connected") await this.refresh();
    }, "Testing…"));
    const test = c.configured ? button("Test") : null;
    test?.addEventListener("click", () => busy(test, this.hud, async () => {
      const { status: s } = await api.testConnection(c.id);
      this.#show(c.id, s);
    }, "Testing…"));
    const remove = c.configured ? button("Disconnect", { danger: true }) : null;
    remove?.addEventListener("click", () => busy(remove, this.hud, async () => {
      await api.removeConnection(c.id);
      await this.refresh();
    }));
    const extras = c.id === HOME_ASSISTANT ? this.#homeAssistantHelpers(inputs.get("url")) : [];
    body.appendChild(row(save, test, ...extras, remove));
    if (c.id === HOME_ASSISTANT) {
      if (c.values.signed_in) body.appendChild(para("Signed in to Home Assistant; Somni keeps the sign-in fresh."));
      body.appendChild(this.#vmBox());
    }
  }

  /** Home Assistant's VM on this PC (Somni starts it with itself): its state, start and stop. */
  #vmBox() {
    const box = el("div", { className: "vm-box" });
    const draw = (vm) => {
      box.replaceChildren();
      if (!vm.vm) {
        if (vm.virtualbox) box.appendChild(para("No Home Assistant virtual machine on this PC."));
        return;
      }
      const running = vm.state === "running";
      box.appendChild(statusLine(running ? "connected" : "not_configured",
        `Virtual machine "${vm.vm}": ${vm.state}. Somni starts it whenever Somni starts.`));
      const toggle = button(running ? "Stop Home Assistant" : "Start Home Assistant", { danger: running });
      toggle.addEventListener("click", () => busy(toggle, this.hud, async () => {
        draw(await api.homeAssistantVmControl(running ? "stop" : "start"));
        this.hud.toast(running ? "Home Assistant is shutting down" : "Home Assistant is starting (about a minute)",
          "good");
      }, running ? "Stopping…" : "Starting…"));
      box.appendChild(row(toggle));
    };
    api.homeAssistantVm().then(draw).catch(() => {});
    return box;
  }

  /** Home Assistant: find it on the network, sign in on its own page (no token to copy). */
  #homeAssistantHelpers(urlInput) {
    const find = button("Find on my network");
    find.addEventListener("click", () => busy(find, this.hud, async () => {
      const { found } = await api.discoverHomeAssistant();
      if (!found.length) throw new Error("no Home Assistant answered on this network; type its address");
      urlInput.value = found[0];
      this.hud.toast(`found Home Assistant at ${found[0]}`, "good");
    }, "Looking…"));
    const signIn = button("Sign in with Home Assistant", { primary: true });
    signIn.addEventListener("click", () => {
      const base = urlInput.value.trim().replace(/\/+$/, "");
      if (!/^https?:\/\//.test(base)) {
        this.hud.toast("enter or find the Home Assistant address first", "error");
        return;
      }
      window.location.href = `/api/connections/home_assistant/signin?url=${encodeURIComponent(base)}`;
    });
    return [find, signIn];
  }

  /** A card whose status comes from another connection (Google Home through Home Assistant). */
  #derived(c, body) {
    const check = button("Check again");
    check.addEventListener("click", () => busy(check, this.hud, async () => {
      const { status: s } = await api.testConnection(c.id);
      this.#show(c.id, s);
    }, "Checking…"));
    body.appendChild(row(check));
  }

  #statusLine(s) {
    const label = STATE_TEXT[s.state] ?? s.state;
    const message = s.message && s.message.toLowerCase() !== label.toLowerCase() ? ` · ${s.message}` : "";
    return statusLine(s.state, label + message);
  }

  #show(id, s) {
    const old = this.#status.get(id);
    if (!old) return;
    const line = this.#statusLine(s);
    old.replaceWith(line);
    this.#status.set(id, line);
  }

  onEvent(type, event) {
    if (type === "connection") this.#show(event.id, event.status);
  }
}
