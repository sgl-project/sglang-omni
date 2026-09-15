// SPDX-License-Identifier: Apache-2.0
// Continuous area resampling; fractional sample positions survive worklet blocks.
export class PCMFramer {
  constructor(inputRate, outputRate = 16000, frameSamples = 1280) {
    if (!(inputRate >= outputRate && outputRate > 0 && frameSamples > 0)) throw new Error('Invalid audio rates');
    this.ratio = inputRate / outputRate;
    this.remaining = this.ratio;
    this.sum = 0;
    this.offset = 0;
    this.frame = new Int16Array(frameSamples);
  }
  push(input, emit) {
    for (const value of input) {
      let available = 1;
      while (available > 1e-9) {
        const take = Math.min(available, this.remaining);
        this.sum += value * take;
        this.remaining -= take;
        available -= take;
        if (this.remaining < 1e-9) {
          const x = Math.max(-1, Math.min(1, this.sum / this.ratio));
          this.frame[this.offset++] = Math.round(x * (x < 0 ? 32768 : 32767));
          this.remaining = this.ratio;
          this.sum = 0;
          if (this.offset === this.frame.length) {
            const ready = this.frame;
            this.frame = new Int16Array(ready.length);
            this.offset = 0;
            emit(ready);
          }
        }
      }
    }
  }
}

export function encodePCM(samples) {
  const bytes = new Uint8Array(samples.length * 2);
  const view = new DataView(bytes.buffer);
  samples.forEach((value, i) => view.setInt16(i * 2, value, true));
  let binary = '';
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary);
}

export function decodePCM(encoded) {
  const raw = atob(encoded);
  if (raw.length % 2) throw new Error('Invalid PCM16 audio');
  const bytes = Uint8Array.from(raw, c => c.charCodeAt(0));
  const view = new DataView(bytes.buffer);
  return Float32Array.from({length: raw.length / 2}, (_, i) => view.getInt16(i * 2, true) / 32768);
}

export class PCMPlayer {
  constructor(context, onPlayed) {
    this.context = context;
    this.onPlayed = onPlayed;
    this.analyser = context.createAnalyser();
    this.analyser.fftSize = 256;
    this.analyser.connect(context.destination);
    this.sources = new Set();
    this.endAt = 0;
    this.ends = new Map();
    this.rebuffers = 0;
  }
  get queuedSeconds() { return Math.max(0, this.endAt - this.context.currentTime); }
  enqueue(pcm, rate, event) {
    if (!pcm.length) return;
    if (this.queuedSeconds > 8) throw new Error('播放积压超过 8 秒，请结束后重新开始。');
    const buffer = this.context.createBuffer(1, pcm.length, rate);
    buffer.copyToChannel(pcm, 0);
    const source = this.context.createBufferSource();
    source.buffer = buffer;
    source.connect(this.analyser);
    const epoch = event.sglang?.epoch ?? 0;
    const key = `${epoch}/${event.response_id}/${event.item_id}`;
    const playedEndMs = (this.ends.get(key) || 0) + pcm.length * 1000 / rate;
    this.ends.set(key, playedEndMs);
    const item = {source, cancelled: false};
    this.sources.add(item);
    // Reserve jitter headroom only at startup/underrun. Adding a lead to
    // every packet punches gaps into an otherwise continuous waveform.
    const now = this.context.currentTime;
    const underrun = this.endAt > 0 && this.endAt < now + 0.005;
    if (underrun) this.rebuffers++;
    const at = this.endAt > now + 0.005 ? this.endAt : now + 0.48;
    this.endAt = at + buffer.duration;
    source.onended = () => {
      this.sources.delete(item);
      source.disconnect();
      if (!item.cancelled) this.onPlayed?.(event, playedEndMs);
    };
    source.start(at);
  }
  clear() {
    for (const item of this.sources) {
      item.cancelled = true;
      try { item.source.stop(); } catch { /* Already ended. */ }
      item.source.disconnect();
    }
    this.sources.clear();
    this.ends.clear();
    this.endAt = 0;
  }
  async close() {
    this.clear();
    this.analyser.disconnect();
    await this.context.close();
  }
}
