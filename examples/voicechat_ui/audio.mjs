// SPDX-License-Identifier: Apache-2.0
// Continuous area resampling; fractional sample positions survive worklet blocks.
export class PCMFramer {
    constructor(inputRate, outputRate = 16000, frameSamples = 1280) {
        if (!(inputRate >= outputRate && outputRate > 0 && frameSamples > 0))
            throw new Error("Invalid audio rates");
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
                    this.frame[this.offset++] = Math.round(
                        x * (x < 0 ? 32768 : 32767),
                    );
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
    let binary = "";
    for (const byte of bytes) binary += String.fromCharCode(byte);
    return btoa(binary);
}

export function decodePCM(encoded) {
    const raw = atob(encoded);
    if (raw.length % 2) throw new Error("Invalid PCM16 audio");
    const bytes = Uint8Array.from(raw, (c) => c.charCodeAt(0));
    const view = new DataView(bytes.buffer);
    return Float32Array.from(
        { length: raw.length / 2 },
        (_, i) => view.getInt16(i * 2, true) / 32768,
    );
}

// Stateful windowed-sinc conversion. Browser BufferSource resampling restarts
// its filter at every packet; convert continuously before creating sources.
export class PlaybackResampler {
    constructor(sourceRate, targetRate) {
        this.ratio = sourceRate / targetRate;
        this.half = 16;
        this.phases = 1024;
        this.buffer = new Float32Array(0);
        this.start = 0;
        this.total = 0;
        this.produced = 0;
        const cutoff = Math.min(1, targetRate / sourceRate) * 0.94;
        this.weights = Array.from({ length: this.phases }, (_, phase) => {
            const f = phase / this.phases;
            const w = Float64Array.from({ length: 32 }, (_, k) => {
                const distance = k - 15 - f;
                const x = Math.PI * cutoff * distance;
                const sinc = Math.abs(x) < 1e-12 ? 1 : Math.sin(x) / x;
                return (
                    cutoff *
                    sinc *
                    (0.5 + 0.5 * Math.cos((Math.PI * distance) / this.half))
                );
            });
            const sum = w.reduce((a, b) => a + b, 0);
            return w.map((x) => x / sum);
        });
    }
    push(pcm) {
        const joined = new Float32Array(this.buffer.length + pcm.length);
        joined.set(this.buffer);
        joined.set(pcm, this.buffer.length);
        this.total += pcm.length;
        const values = [];
        while (
            Math.floor(this.produced * this.ratio) + this.half <
            this.total
        ) {
            const position = this.produced * this.ratio;
            const base = Math.floor(position);
            const w =
                this.weights[
                    Math.min(
                        this.phases - 1,
                        Math.floor((position - base) * this.phases),
                    )
                ];
            let value = 0;
            for (let k = 0; k < w.length; k++) {
                const index = base + k - 15;
                if (index >= 0) value += joined[index - this.start] * w[k];
            }
            values.push(value);
            this.produced++;
        }
        const keep = Math.max(0, Math.floor(this.produced * this.ratio) - 15);
        this.buffer = joined.slice(keep - this.start);
        this.start = keep;
        return Float32Array.from(values);
    }
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
    get queuedSeconds() {
        return Math.max(0, this.endAt - this.context.currentTime);
    }
    enqueue(pcm, rate, event) {
        if (!pcm.length) return;
        if (this.queuedSeconds > 8)
            throw new Error("播放积压超过 8 秒，请结束后重新开始。");
        const targetRate = this.context.sampleRate;
        const epoch = event.sglang?.epoch ?? 0;
        const key = `${epoch}/${event.response_id}/${event.item_id}`;
        if (this.streamKey !== key || this.sourceRate !== rate) {
            this.resampler =
                rate === targetRate
                    ? null
                    : new PlaybackResampler(rate, targetRate);
            this.streamKey = key;
            this.sourceRate = rate;
        }
        const inputEndMs =
            (this.ends.get(key) || 0) + (pcm.length * 1000) / rate;
        this.ends.set(key, inputEndMs);
        const samples = this.resampler ? this.resampler.push(pcm) : pcm;
        if (!samples.length) return;
        const buffer = this.context.createBuffer(1, samples.length, targetRate);
        buffer.copyToChannel(samples, 0);
        const source = this.context.createBufferSource();
        source.buffer = buffer;
        source.connect(this.analyser);
        const playedEndMs = this.resampler
            ? Math.min(
                  inputEndMs,
                  (this.resampler.produced * 1000) / targetRate,
              )
            : inputEndMs;
        const item = { source, cancelled: false };
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
            try {
                item.source.stop();
            } catch {
                /* Already ended. */
            }
            item.source.disconnect();
        }
        this.sources.clear();
        this.ends.clear();
        this.resampler = null;
        this.streamKey = null;
        this.endAt = 0;
    }
    async close() {
        this.clear();
        this.analyser.disconnect();
        await this.context.close();
    }
}
