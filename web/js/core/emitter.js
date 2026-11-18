// Minimal typed-by-convention event emitter.

export class Emitter {
  #handlers = new Map();

  on(type, handler) {
    if (!this.#handlers.has(type)) this.#handlers.set(type, new Set());
    this.#handlers.get(type).add(handler);
    return () => this.off(type, handler);
  }

  off(type, handler) {
    this.#handlers.get(type)?.delete(handler);
  }

  emit(type, payload) {
    for (const handler of this.#handlers.get(type) ?? []) {
      try {
        handler(payload);
      } catch (err) {
        console.error(`[somni] handler for "${type}" failed`, err);
      }
    }
  }
}
