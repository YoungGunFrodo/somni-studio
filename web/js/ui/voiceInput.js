// Voice input in the chat panel: the mic button (push to talk), the hands-free toggle, a level
// meter and a one-line status. Utterances go to /api/listen; what comes back becomes a message.

import { $ } from "../core/dom.js";
import { Listener } from "../live/listener.js";
import { api } from "../net/api.js";

const STATUS_CLEAR_MS = 4000;
const STORAGE_KEY = "somni.handsfree";

export class VoiceInput {
  #listener = new Listener();
  #status = $("#voice-status");
  #mic = $("#mic-button");
  #handsfree = $("#handsfree-toggle");
  #clear = null;
  #wakeWord = "somni";

  constructor({ hud, send, speaker }) {
    this.hud = hud;
    this.send = send;
    this.speaker = speaker;
    this.#mic.addEventListener("click", () => this.#push());
    this.#handsfree.addEventListener("click", () => this.#toggleHandsFree());
    this.#listener.on("state", (s) => this.#render(s));
    this.#listener.on("level", (level) => this.#mic.style.setProperty("--level", level.toFixed(3)));
    this.#listener.on("speech-start", () => {
      if (this.#listener.mode === "push") this.speaker.stop();       // talking over it stops it
    });
    this.#listener.on("utterance", (u) => this.#hear(u));
  }

  /** Called with /api/listen status (from the hello event and the Voice tab). */
  configure(status) {
    this.#wakeWord = status.wake_word || this.#wakeWord;
    const usable = status.enabled && status.ready;
    this.#mic.disabled = this.#handsfree.disabled = !usable;
    this.#mic.title = usable ? "Talk (click, speak, pause)" : status.problem || "listening is switched off";
    this.#handsfree.title = usable ? `Hands-free: say "${this.#title()}, …"` : this.#mic.title;
    if (usable && this.#wanted() && this.#listener.mode === "off") this.#toggleHandsFree(true);
    if (!usable && this.#listener.mode !== "off") this.#listener.setHandsFree(false);
  }

  /** The assistant is speaking: hands-free listening pauses so it never hears itself. */
  set assistantSpeaking(on) {
    this.#listener.paused = on;
  }

  #title() {
    return this.#wakeWord.charAt(0).toUpperCase() + this.#wakeWord.slice(1);
  }

  #wanted() {
    try {
      return localStorage.getItem(STORAGE_KEY) === "1";
    } catch {
      return false;
    }
  }

  #remember(on) {
    try {
      localStorage.setItem(STORAGE_KEY, on ? "1" : "0");
    } catch {
      /* private mode: just not remembered */
    }
  }

  async #push() {
    try {
      await this.#listener.push();
    } catch (err) {
      this.hud.toast(`microphone: ${err.message}`, "error");
    }
  }

  async #toggleHandsFree(force) {
    const on = force ?? this.#listener.mode !== "handsfree";
    try {
      await this.#listener.setHandsFree(on);
      this.#remember(on);
      if (on) this.#say(`Hands-free: say "${this.#title()}, …" to talk.`);
    } catch (err) {
      this.hud.toast(`microphone: ${err.message}`, "error");
    }
  }

  #render({ mode, recording }) {
    this.#mic.classList.toggle("recording", recording);
    this.#mic.setAttribute("aria-pressed", String(mode === "push"));
    this.#handsfree.setAttribute("aria-pressed", String(mode === "handsfree" || this.#listener.mode === "handsfree"));
    if (mode === "push" && recording) this.#say("Listening… (pause when you are done)", false);
  }

  async #hear({ wav, mode }) {
    this.#say("Understanding…", false);
    try {
      const result = await api.listen(wav, mode);
      if (mode === "wake" && result.heard_wake && !result.command) {
        this.#listener.arm();
        this.#say("Yes? I'm listening.");
        return;
      }
      const text = (result.command || "").trim();
      if (!text) {
        if (mode === "talk") this.#say("I didn't catch that.");
        else this.#say("");                          // not addressed to us: stay quiet
        return;
      }
      this.#say(`Heard: “${text}”`);
      this.send(text);
    } catch (err) {
      this.#say("");
      this.hud.toast(`listening: ${err.message}`, "error");
    }
  }

  #say(text, autoClear = true) {
    clearTimeout(this.#clear);
    this.#status.textContent = text;
    this.#status.hidden = !text;
    if (text && autoClear) this.#clear = setTimeout(() => (this.#status.hidden = true), STATUS_CLEAR_MS);
  }
}
