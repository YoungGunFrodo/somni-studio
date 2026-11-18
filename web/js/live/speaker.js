// Plays spoken answers: sentences arrive as "speech" events (possibly out of order, possibly with a
// failed one) and are played in order. A new answer interrupts the previous one.

const RETRY_MS = 120;

export class Speaker {
  #audio = new Audio();
  #request = null;          // the answer being spoken
  #clips = new Map();       // seq -> url | null (failed)
  #next = 0;
  #playing = false;
  #enabled = true;

  constructor({ button, onToggle, onPlaying = () => {} }) {
    this.button = button;
    this.onPlaying = onPlaying;
    this.#audio.preload = "auto";
    this.#audio.addEventListener("ended", () => this.#advance());
    this.#audio.addEventListener("error", () => this.#advance());
    button.addEventListener("click", () => onToggle(!this.#enabled));
  }

  get enabled() {
    return this.#enabled;
  }

  setEnabled(on) {
    this.#enabled = Boolean(on);
    this.button.setAttribute("aria-pressed", String(this.#enabled));
    this.button.title = this.#enabled ? "Speaking answers aloud (click to mute)" : "Muted (click to speak answers)";
    if (!this.#enabled) this.stop();
  }

  /** A new answer started: forget the old one. */
  begin(requestId) {
    this.stop();
    this.#request = requestId;
  }

  /** A "speech" event from the server. */
  handle(event, clipUrl) {
    if (!this.#enabled) return;
    if (event.request_id !== this.#request) {
      if (this.#request !== null) return;          // a stale answer
      this.#request = event.request_id;
    }
    this.#clips.set(event.seq, event.clip ? clipUrl(event.clip) : null);
    if (!this.#playing) this.#play();
  }

  /** Play one clip right away (voice preview). */
  playNow(url) {
    this.stop();
    this.#request = "preview";
    this.#clips.set(0, url);
    this.#play();
  }

  stop() {
    this.#audio.pause();
    this.#audio.removeAttribute("src");
    this.#clips.clear();
    this.#next = 0;
    this.#playing = false;
    this.#request = null;
    this.button.classList.remove("speaking");
    this.onPlaying(false);
  }

  #play() {
    if (!this.#clips.has(this.#next)) {
      this.#playing = false;
      this.button.classList.remove("speaking");
      this.onPlaying(false);
      return;
    }
    const url = this.#clips.get(this.#next);
    if (url === null) {                             // that sentence failed: skip it
      this.#advance();
      return;
    }
    this.#playing = true;
    this.button.classList.add("speaking");
    this.onPlaying(true);
    this.#audio.src = url;
    this.#audio.play().catch(() => {
      // Autoplay can be blocked until the page has been clicked once; try the next sentence later.
      setTimeout(() => this.#advance(), RETRY_MS);
    });
  }

  #advance() {
    this.#clips.delete(this.#next);
    this.#next += 1;
    this.#play();
  }
}
