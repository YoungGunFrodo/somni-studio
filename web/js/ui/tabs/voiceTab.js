// Voice tab: speak answers aloud, choose a voice and speed, hear a preview.

import { api } from "../../net/api.js";
import { busy, button, card, field, input, para, row, select, statusLine } from "./widgets.js";

const PREVIEW_TEXT = "Good evening, sir. All systems are running normally, and the house is quiet.";
// A measured, produced AI-butler sound: a touch of a deeper British voice, the assistant effect, unhurried.
const BUTLER = { blend: "bm_george", blend_amount: 0.25, effect: "assistant", speed: 0.93, pause: 0.18 };
const EFFECTS = [
  { value: "none", label: "None: the plain voice" },
  { value: "assistant", label: "Assistant: a faint metallic sheen and a small room, like a house AI" },
];

export class VoiceTab {
  title = "Voice";

  constructor({ hud, saveSettings, speaker, onListenChanged }) {
    this.hud = hud;
    this.saveSettings = saveSettings;
    this.speaker = speaker;
    this.onListenChanged = onListenChanged;
  }

  async render(body) {
    const s = await api.speech();

    const state = card("Speech");
    state.body.appendChild(statusLine(s.ready ? "ready" : "error",
      s.ready ? `ready · Kokoro on ${s.threads} CPU threads` : s.problem));
    state.body.appendChild(para("Answers are spoken sentence by sentence while the model is still writing. "
      + "Speech runs on the CPU, so it does not slow down the model on the GPU."));
    const enabled = input("checkbox");
    enabled.checked = s.enabled;
    enabled.addEventListener("change", () => this.saveSettings({ voice: { enabled: enabled.checked } })
      .then(() => this.speaker.setEnabled(enabled.checked))
      .catch((err) => this.hud.toast(err.message, "error")));
    state.body.appendChild(field({ label: "Speak answers aloud", control: enabled }));
    body.appendChild(state.root);

    const voice = card("Voice");
    const voices = s.voices.length ? s.voices : [{ id: s.voice, label: s.voice }];
    const pick = select(voices.map((v) => ({ value: v.id, label: v.label })), s.voice);
    const speed = input("range", String(s.speed), { min: "0.7", max: "1.4", step: "0.05" });
    const speedLabel = document.createElement("span");
    const showSpeed = () => (speedLabel.textContent = `${Number(speed.value).toFixed(2)}×`);
    showSpeed();
    speed.addEventListener("input", showSpeed);
    const blend = select([{ value: "", label: "None" }, ...voices.map((v) => ({ value: v.id, label: v.label }))],
      s.blend || "");
    const amount = input("range", String(s.blend_amount ?? 0), { min: "0", max: "0.8", step: "0.05" });
    const amountLabel = document.createElement("span");
    const effect = select(EFFECTS, s.effect || "none");
    const pause = input("range", String(s.pause ?? 0), { min: "0", max: "0.8", step: "0.02" });
    const pauseLabel = document.createElement("span");
    const showAll = () => {
      showSpeed();
      amountLabel.textContent = `${Math.round(Number(amount.value) * 100)}%`;
      pauseLabel.textContent = `${Number(pause.value).toFixed(2)} s`;
    };
    showAll();
    for (const node of [amount, pause]) node.addEventListener("input", showAll);
    const style = () => ({ voice: pick.value, speed: Number(speed.value), blend: blend.value,
      blend_amount: blend.value ? Number(amount.value) : 0, effect: effect.value, pause: Number(pause.value) });
    const text = input("text", PREVIEW_TEXT);
    const preview = button("Preview");
    const butler = button("AI butler preset");
    butler.title = "Keeps your voice; adds a little of a deeper British voice, the assistant effect and unhurried pacing";
    const save = button("Use this voice", { primary: true });
    preview.disabled = save.disabled = butler.disabled = !s.ready;
    const play = async () => {
      const { clip } = await api.previewSpeech(text.value, style());
      this.speaker.playNow(api.clipUrl(clip));
    };
    preview.addEventListener("click", () => busy(preview, this.hud, play, "Speaking…"));
    butler.addEventListener("click", () => busy(butler, this.hud, async () => {
      if (pick.value === BUTLER.blend) blend.value = "bm_lewis";
      else blend.value = BUTLER.blend;
      amount.value = String(BUTLER.blend_amount);
      effect.value = BUTLER.effect;
      speed.value = String(BUTLER.speed);
      pause.value = String(BUTLER.pause);
      showAll();
      await play();
    }, "Speaking…"));
    save.addEventListener("click", () => busy(save, this.hud, async () => {
      await this.saveSettings({ voice: style() });
      this.hud.toast("voice saved", "good");
    }));
    voice.body.append(field({ label: "Voice", control: pick }), field({ label: "Speed", control: row(speed, speedLabel) }),
      field({ label: "Blend with", help: "Mix in a second voice to change the character.", control: blend }),
      field({ label: "Blend amount", control: row(amount, amountLabel) }),
      field({ label: "Effect", control: effect }),
      field({ label: "Pause between sentences", control: row(pause, pauseLabel) }),
      field({ label: "Preview text", control: text }), row(preview, butler, save));
    body.appendChild(voice.root);
    body.appendChild(await this.#listening());
  }

  async #listening() {
    const s = await api.listenStatus();
    const box = card("Listening", "speech-to-text on the CPU");
    box.body.appendChild(statusLine(s.ready ? "ready" : "error",
      s.ready ? `ready · Whisper ${s.model} on ${s.threads} CPU threads` : s.problem));
    box.body.appendChild(para("Click the microphone in the chat to talk; it stops when you pause. Or switch on "
      + `hands-free (the ear) and start with "${s.wake_word}": "${s.wake_word}, turn on the kitchen light". `
      + "It does not listen while it is speaking, and audio never leaves this computer."));
    const enabled = input("checkbox");
    enabled.checked = s.enabled;
    const model = select([
      { value: "tiny.en", label: "tiny.en: fastest, least accurate" },
      { value: "base.en", label: "base.en: fast and accurate (recommended)" },
      { value: "small.en", label: "small.en: most accurate, 3x slower" },
    ], s.model);
    const wake = input("text", s.wake_word, { maxlength: "40" });
    const save = button("Save", { primary: true });
    save.addEventListener("click", () => busy(save, this.hud, async () => {
      await this.saveSettings({ listen: { enabled: enabled.checked, model: model.value,
        wake_word: wake.value.trim().toLowerCase() } });
      const status = await api.listenStatus();
      this.onListenChanged?.(status);
      this.hud.toast(status.ready || !status.enabled ? "listening saved" : status.problem, status.ready ? "good" : "error");
    }));
    box.body.append(field({ label: "Use the microphone", control: enabled }),
      field({ label: "Recognition model", control: model,
        help: "Models are downloaded with scripts/fetch_listen.py." }),
      field({ label: "Wake word", control: wake, help: "A short, unusual word works best." }), row(save));
    return box.root;
  }
}
