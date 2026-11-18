// The "Devices" card on the Connect tab: things Somni controls directly on the network (lights,
// plugs, speakers, displays, TVs). Scan, connect with one click, switch them, forget them.
// Everything here can also be done by asking Somni ("find my devices", "connect the kitchen display").

import { el } from "../../core/dom.js";
import { api } from "../../net/api.js";
import { busy, button, card, para, row } from "./widgets.js";

const KIND_LABEL = { light: "light", switch: "plug / switch", speaker: "speaker / display", tv: "TV" };
const EXAMPLES = ["Find my devices", "Connect the kitchen display", "Turn off the desk lamp",
  "Pause the living room display", "Say dinner is ready on the kitchen display", "Open Netflix on the Roku"];

export class DevicesCard {
  #box;
  #state = { links: [], devices: [], found: [] };

  constructor({ hud }) {
    this.hud = hud;
  }

  render() {
    this.#box = card("Devices", "Directly on your network");
    this.#box.body.append(
      para("Lights, plugs, speakers, displays and TVs that Somni controls itself: no hub, no account, nothing "
        + "leaves your network. Works with WiZ, LIFX, TP-Link Kasa, Philips Hue, Yeelight, Govee, Shelly, "
        + "Google Nest / Chromecast and Roku."),
      el("div", { className: "devices-live", text: "Loading…" }));
    this.#load();
    return this.#box.root;
  }

  async #load() {
    try {
      this.#state = await api.devices();
    } catch (err) {
      this.#state = { links: [], devices: [], found: [] };
      this.hud.toast(`devices: ${err.message}`, "error");
    }
    this.#draw();
  }

  #draw() {
    const live = this.#box.body.querySelector(".devices-live");
    live.replaceChildren();
    const { devices, links, found } = this.#state;
    if (devices.length) {
      const list = el("ul", { className: "device-list" });
      for (const d of devices) list.appendChild(this.#deviceRow(d));
      live.appendChild(list);
    } else {
      live.appendChild(para("Nothing connected yet. Scan to see what is on your network, or just ask Somni to "
        + "find your devices."));
    }
    const scan = button(found.length ? "Scan again" : "Scan my network", { primary: !links.length });
    scan.addEventListener("click", () => busy(scan, this.hud, async () => {
      this.#state = await api.scanDevices();
      this.#draw();
    }, "Scanning…"));
    live.appendChild(row(scan));
    live.appendChild(this.#foundList(found, new Set(links.map((l) => l.id))));
    live.append(el("p", { className: "examples-title", text: "Then try saying:" }),
      el("ul", { className: "examples" }, EXAMPLES.map((x) => el("li", { text: `“${x}”` }))));
  }

  #deviceRow(d) {
    const state = d.state === "unreachable" ? "not answering" : d.detail || d.state;
    const extra = [d.brightness != null && d.on ? `${d.brightness}%` : "", d.volume != null ? `volume ${d.volume}` : ""]
      .filter(Boolean).join(" · ");
    const item = el("li", { className: "device-row", attrs: { "data-state": d.state } }, [
      el("span", { className: "device-dot" }),
      el("span", { className: "device-name", text: d.name }),
      el("span", { className: "device-meta", text: [KIND_LABEL[d.kind] ?? d.kind, state, extra].filter(Boolean).join(" · ") }),
    ]);
    const actions = el("span", { className: "device-actions" });
    if (d.reachable && (d.kind === "light" || d.kind === "switch")) {
      const toggle = button(d.on ? "Turn off" : "Turn on");
      toggle.addEventListener("click", () => busy(toggle, this.hud, () => this.#control(d, "toggle")));
      actions.appendChild(toggle);
    }
    if (d.reachable && d.kind === "speaker" && d.on) {
      const stop = button("Stop");
      stop.addEventListener("click", () => busy(stop, this.hud, () => this.#control(d, "stop")));
      actions.appendChild(stop);
    }
    const forget = button("Forget", { danger: true });
    forget.title = "Disconnect (you can connect it again with a scan)";
    forget.addEventListener("click", () => busy(forget, this.hud, async () => {
      this.#state = await api.forgetDevice(d.link);
      this.#draw();
    }));
    actions.appendChild(forget);
    item.appendChild(actions);
    return item;
  }

  async #control(device, action) {
    const { device: now } = await api.controlDevice(device.id, { action });
    this.#state.devices = this.#state.devices.map((d) => (d.id === now.id ? now : d));
    this.#draw();
  }

  #foundList(found, connected) {
    const wrap = el("div", { className: "found-list" });
    const usable = found.filter((f) => f.supported && !connected.has(f.key));
    const others = found.filter((f) => !f.supported);
    if (usable.length) {
      wrap.appendChild(el("p", { className: "examples-title", text: "Found on your network" }));
      const list = el("ul", { className: "device-list" });
      for (const f of usable) {
        const connect = button("Connect", { primary: true });
        connect.addEventListener("click", () => busy(connect, this.hud, async () => {
          const result = await api.connectDevice(f.key);
          this.#state = result;
          const waiting = result.report.waiting?.[0];
          this.hud.toast(waiting ?? `connected ${f.name}`, waiting ? "error" : "good");
          this.#draw();
        }, "Connecting…"));
        list.appendChild(el("li", { className: "device-row" }, [
          el("span", { className: "device-name", text: f.name }),
          el("span", { className: "device-meta", text: [f.model, f.note].filter(Boolean).join(" · ") }),
          el("span", { className: "device-actions" }, [connect]),
        ]));
      }
      wrap.appendChild(list);
    }
    if (others.length) {
      wrap.appendChild(el("p", { className: "examples-title", text: "Also on your network (not supported yet)" }));
      wrap.appendChild(el("ul", { className: "device-list muted" }, others.map((f) => el("li", {
        className: "device-row",
        text: `${f.name}${f.model ? ` · ${f.model}` : ""}`,
      }))));
    }
    return wrap;
  }
}
