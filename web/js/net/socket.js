// Live events over one WebSocket, reconnecting with backoff. Emits each event under its `type`,
// plus "connection" with {connected: boolean} whenever the link goes up or down.

import { Emitter } from "../core/emitter.js";

const MIN_DELAY_MS = 500;
const MAX_DELAY_MS = 8000;

export class LiveSocket extends Emitter {
  #url;
  #ws = null;
  #delay = MIN_DELAY_MS;
  #timer = null;

  constructor(path = "/ws") {
    super();
    const scheme = location.protocol === "https:" ? "wss:" : "ws:";
    this.#url = `${scheme}//${location.host}${path}`;
  }

  connect() {
    clearTimeout(this.#timer);
    const ws = new WebSocket(this.#url);
    this.#ws = ws;
    ws.addEventListener("open", () => {
      this.#delay = MIN_DELAY_MS;
      this.emit("connection", { connected: true });
    });
    ws.addEventListener("message", (msg) => {
      let event;
      try {
        event = JSON.parse(msg.data);
      } catch {
        return;
      }
      if (event && typeof event.type === "string") this.emit(event.type, event);
    });
    ws.addEventListener("close", () => {
      if (this.#ws !== ws) return;
      this.emit("connection", { connected: false });
      this.#timer = setTimeout(() => this.connect(), this.#delay);
      this.#delay = Math.min(this.#delay * 2, MAX_DELAY_MS);
    });
  }
}
