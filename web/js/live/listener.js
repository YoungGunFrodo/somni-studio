// The microphone: capture, voice-activity detection, and the two ways of talking.
//
//   push to talk  click the mic, speak, it stops by itself after a pause (or click again)
//   hands-free    always listening; only what follows the wake word ("Somni, ...") is sent.
//                 Saying just "Somni" and pausing works too: the next sentence is the command.
//
// Detection is energy-based with an adaptive noise floor, which is cheap and good enough to cut
// utterances; Whisper (on the server) decides what was actually said. While the assistant is
// speaking, hands-free listening pauses so it never hears itself.
//
// Emits: "state" {mode, recording}, "level" (0..1), "utterance" {wav: Blob, mode}.

import { Emitter } from "../core/emitter.js";

const RATE = 16000;
const FRAME_S = 0.032;
const START_FRAMES = 3;            // ~100 ms above threshold starts an utterance
const END_SILENCE_S = { push: 1.3, handsfree: 0.8 };
const PREROLL_S = 0.35;            // audio kept from just before speech started
const MAX_UTTERANCE_S = 20;
const MIN_UTTERANCE_S = 0.35;
const MIN_THRESHOLD = 0.012;
const FLOOR_RISE = 0.02;           // how fast the noise floor follows quiet sound
const ARMED_S = 8;                 // after "Somni" alone, the next sentence is the command

export class Listener extends Emitter {
  #ctx = null;
  #stream = null;
  #node = null;
  #mode = "off";                   // off | push | handsfree
  #recording = false;
  #chunks = [];
  #preroll = [];
  #above = 0;
  #silence = 0;
  #length = 0;
  #floor = 0.004;
  #paused = false;
  #armedUntil = 0;
  #returnTo = "off";               // where push-to-talk goes back to (off, or hands-free)

  get mode() {
    return this.#mode;
  }

  get recording() {
    return this.#recording;
  }

  /** Hands-free listening pauses while the assistant talks. */
  set paused(on) {
    this.#paused = on;
    if (on && this.#mode === "handsfree" && this.#recording) this.#reset();
  }

  arm() {
    this.#armedUntil = performance.now() + ARMED_S * 1000;
  }

  get armed() {
    return performance.now() < this.#armedUntil;
  }

  async push() {
    if (this.#mode === "push") {                // second click: finish now
      this.#finish();
      return;
    }
    const wasHandsFree = this.#mode === "handsfree";
    await this.#open();
    this.#mode = "push";
    this.#returnTo = wasHandsFree ? "handsfree" : "off";
    this.#reset();
    this.#recording = true;                      // record from the click, not from the first loud frame
    this.#state();
  }

  async setHandsFree(on) {
    if (on) {
      await this.#open();
      this.#mode = "handsfree";
      this.#reset();
    } else {
      this.#mode = "off";
      this.#reset();
      this.#close();
    }
    this.#state();
  }

  async #open() {
    if (this.#ctx) return;
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("this browser cannot use a microphone here");
    this.#stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    this.#ctx = new AudioContext();
    await this.#ctx.audioWorklet.addModule("/js/live/mic-worklet.js");
    const source = this.#ctx.createMediaStreamSource(this.#stream);
    this.#node = new AudioWorkletNode(this.#ctx, "somni-capture");
    this.#node.port.onmessage = (e) => this.#frame(downsample(e.data, this.#ctx.sampleRate));
    source.connect(this.#node);
  }

  #close() {
    this.#node?.disconnect();
    this.#stream?.getTracks().forEach((t) => t.stop());
    this.#ctx?.close();
    this.#ctx = this.#stream = this.#node = null;
  }

  #reset() {
    this.#recording = false;
    this.#chunks = [];
    this.#above = this.#silence = this.#length = 0;
  }

  #state() {
    this.emit("state", { mode: this.#mode, recording: this.#recording, armed: this.armed });
  }

  #frame(samples) {
    const rms = Math.sqrt(samples.reduce((s, x) => s + x * x, 0) / samples.length);
    const threshold = Math.max(MIN_THRESHOLD, this.#floor * 3.2);
    this.emit("level", Math.min(1, rms / (threshold * 4)));
    if (this.#mode === "off" || (this.#paused && this.#mode === "handsfree")) return;

    const loud = rms > threshold;
    if (!this.#recording) {
      if (!loud) this.#floor += (Math.min(rms, this.#floor * 2) - this.#floor) * FLOOR_RISE;
      this.#preroll.push(samples);
      while (this.#preroll.length * FRAME_S > PREROLL_S) this.#preroll.shift();
      this.#above = loud ? this.#above + 1 : 0;
      if (this.#above >= START_FRAMES) {
        this.#recording = true;
        this.#chunks = [...this.#preroll];
        this.#length = this.#chunks.length * FRAME_S;
        this.#silence = 0;
        this.emit("speech-start");
        this.#state();
      }
      return;
    }
    this.#chunks.push(samples);
    this.#length += FRAME_S;
    this.#silence = loud ? 0 : this.#silence + FRAME_S;
    const heardSomething = this.#length - this.#silence > MIN_UTTERANCE_S;
    if ((this.#silence >= END_SILENCE_S[this.#mode] && heardSomething) || this.#length >= MAX_UTTERANCE_S) {
      this.#finish();
    } else if (this.#mode === "push" && this.#silence >= 6 && !heardSomething) {
      this.#finish();                            // clicked, then said nothing
    }
  }

  #finish() {
    const mode = this.#mode;
    const chunks = this.#chunks;
    const spoken = this.#length - this.#silence;
    this.#reset();
    if (mode === "push") {
      this.#mode = this.#returnTo;
      if (this.#mode === "off") this.#close();
    }
    this.#state();
    if (chunks.length && spoken > MIN_UTTERANCE_S) {
      const sendMode = mode === "push" || this.armed ? "talk" : "wake";
      this.#armedUntil = 0;
      this.emit("utterance", { wav: encodeWav(chunks), mode: sendMode });
    }
  }
}

function downsample(samples, rate) {
  if (rate === RATE) return samples;
  const ratio = rate / RATE;
  const out = new Float32Array(Math.floor(samples.length / ratio));
  for (let i = 0; i < out.length; i++) {
    const start = Math.floor(i * ratio);
    const end = Math.min(samples.length, Math.floor((i + 1) * ratio));
    let sum = 0;
    for (let j = start; j < end; j++) sum += samples[j];
    out[i] = sum / Math.max(1, end - start);        // box filter: enough for speech at 16 kHz
  }
  return out;
}

function encodeWav(chunks) {
  const length = chunks.reduce((n, c) => n + c.length, 0);
  const buffer = new ArrayBuffer(44 + length * 2);
  const view = new DataView(buffer);
  const text = (offset, s) => [...s].forEach((ch, i) => view.setUint8(offset + i, ch.charCodeAt(0)));
  text(0, "RIFF");
  view.setUint32(4, 36 + length * 2, true);
  text(8, "WAVE");
  text(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, RATE, true);
  view.setUint32(28, RATE * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  text(36, "data");
  view.setUint32(40, length * 2, true);
  let offset = 44;
  for (const chunk of chunks) {
    for (const s of chunk) {
      view.setInt16(offset, Math.max(-1, Math.min(1, s)) * 0x7fff, true);
      offset += 2;
    }
  }
  return new Blob([buffer], { type: "audio/wav" });
}
