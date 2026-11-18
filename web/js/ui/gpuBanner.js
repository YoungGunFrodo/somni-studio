// "Paused while <game> uses the graphics card": shown when the GPU guard steps aside, with the two
// ways out (resume now, or never pause for that program again).

import { $ } from "../core/dom.js";
import { api } from "../net/api.js";

export class GpuBanner {
  #banner = $("#gpu-banner");
  #text = $("#gpu-banner-text");
  #wasPaused = false;

  constructor({ hud }) {
    this.hud = hud;
    $("#gpu-resume").addEventListener("click", () => this.#resume(false));
    $("#gpu-ignore").addEventListener("click", () => this.#resume(true));
  }

  show(status) {
    if (!status) return;
    const paused = Boolean(status.paused);
    this.#banner.hidden = !paused;
    document.body.classList.toggle("gpu-paused", paused);
    if (paused) {
      const name = status.by?.name ?? "another program";
      const what = [status.resumes?.model && "the model", status.resumes?.sleep && "sleep"].filter(Boolean).join(" and ");
      this.#text.textContent = `Paused: ${name} is using the graphics card${what ? ` (${what} will come back)` : ""}. `
        + `I'll resume ${Math.round(status.resume_after_s)} s after it's done.`;
      $("#gpu-ignore").textContent = `Always ignore ${name}`;
      if (!this.#wasPaused) this.hud.toast(`Stepped aside for ${name}`);
    } else if (this.#wasPaused) {
      this.hud.toast("The graphics card is free again: back to work", "good");
    }
    this.#wasPaused = paused;
  }

  async #resume(ignore) {
    try {
      this.show(await api.gpuResume(ignore));
    } catch (err) {
      this.hud.toast(err.message, "error");
    }
  }
}
