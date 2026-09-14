// Captures mic audio and emits 16kHz mono 16-bit PCM chunks.
// Mirrors the bidi-demo recorder contract: onChunk(base64Pcm) per chunk.

const TARGET_RATE = 16000;

export class AudioRecorder {
  constructor(onChunk) {
    this.onChunk = onChunk;
    this.ctx = null;
    this.stream = null;
    this.node = null;
  }

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
    this.ctx = new AudioContext({ sampleRate: TARGET_RATE });

    // AudioWorklet keeps capture off the main thread; ScriptProcessor is
    // deprecated and drops frames under load.
    const workletCode = `
      class PcmTap extends AudioWorkletProcessor {
        process(inputs) {
          const ch = inputs[0][0];
          if (ch) this.port.postMessage(new Float32Array(ch));
          return true;
        }
      }
      registerProcessor('pcm-tap', PcmTap);
    `;
    const blobUrl = URL.createObjectURL(
      new Blob([workletCode], { type: "application/javascript" })
    );
    await this.ctx.audioWorklet.addModule(blobUrl);
    URL.revokeObjectURL(blobUrl);

    const source = this.ctx.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.ctx, "pcm-tap");
    this.node.port.onmessage = (e) => this.onChunk(this._encode(e.data));
    source.connect(this.node);
    // Worklet must reach a destination to be pulled; a muted gain avoids echo.
    const mute = this.ctx.createGain();
    mute.gain.value = 0;
    this.node.connect(mute).connect(this.ctx.destination);
  }

  _encode(float32) {
    const pcm = new Int16Array(float32.length);
    for (let i = 0; i < float32.length; i++) {
      const s = Math.max(-1, Math.min(1, float32[i]));
      pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    let bin = "";
    const bytes = new Uint8Array(pcm.buffer);
    for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
    return btoa(bin);
  }

  stop() {
    if (this.node) this.node.disconnect();
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    if (this.ctx) this.ctx.close();
    this.node = this.stream = this.ctx = null;
  }
}
