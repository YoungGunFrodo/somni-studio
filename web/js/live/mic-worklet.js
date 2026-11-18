// Audio worklet: hands raw microphone samples to the page in ~32 ms batches.
// (Runs on the audio thread; kept tiny on purpose.)

const BATCH = 1536;

class Capture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(BATCH);
    this.filled = 0;
  }

  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel) return true;
    let i = 0;
    while (i < channel.length) {
      const take = Math.min(channel.length - i, BATCH - this.filled);
      this.buffer.set(channel.subarray(i, i + take), this.filled);
      this.filled += take;
      i += take;
      if (this.filled === BATCH) {
        this.port.postMessage(this.buffer.slice(0));
        this.filled = 0;
      }
    }
    return true;
  }
}

registerProcessor("somni-capture", Capture);
