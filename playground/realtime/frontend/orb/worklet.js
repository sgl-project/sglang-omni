class CaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const config = options.processorOptions || {};
    this.sourceRate = config.sourceRate || sampleRate;
    this.targetRate = config.targetRate || 16000;
    this.frameSamples = config.frameSamples || 1280;
    this.input = [];
    this.sourcePosition = 0;
    this.frame = new Int16Array(this.frameSamples);
    this.frameOffset = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;
    for (let index = 0; index < channel.length; index += 1) this.input.push(channel[index]);
    const step = this.sourceRate / this.targetRate;
    while (this.sourcePosition <= this.input.length - 2) {
      const left = Math.floor(this.sourcePosition);
      const weight = this.sourcePosition - left;
      const sample = this.input[left] * (1 - weight) + this.input[left + 1] * weight;
      const clipped = Math.max(-1, Math.min(1, sample));
      this.frame[this.frameOffset] = clipped < 0 ? clipped * 32768 : clipped * 32767;
      this.frameOffset += 1;
      this.sourcePosition += step;
      if (this.frameOffset === this.frameSamples) {
        let energy = 0;
        for (const value of this.frame) energy += (value / 32768) ** 2;
        const frame = this.frame;
        this.port.postMessage({ type: "frame", frame, rms: Math.sqrt(energy / frame.length) }, [frame.buffer]);
        this.frame = new Int16Array(this.frameSamples);
        this.frameOffset = 0;
      }
    }
    const discard = Math.max(0, Math.floor(this.sourcePosition) - 1);
    if (discard) {
      this.input.splice(0, discard);
      this.sourcePosition -= discard;
    }
    return true;
  }
}

class PlaybackProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const config = options.processorOptions || {};
    this.initialSamples = config.initialSamples || Math.round(sampleRate * 0.3);
    this.maxSamples = config.maxSamples || Math.round(sampleRate * 2.5);
    this.keepSamples = config.keepSamples || Math.round(sampleRate * 1.5);
    // Adaptive jitter target: each underrun raises it (up to maxJitterSamples),
    // a quiet stretch lowers it back. With maxJitterSamples == initialSamples
    // (the default) it stays fixed, which suits models that answer in 1 s chunks.
    this.target = this.initialSamples;
    this.maxJitterSamples = this.initialSamples;
    this.sinceUnderrun = 0;
    // Backlog (s) beyond the target that starts a gentle / fast catch-up and a skip.
    this.catchUp = { slow: 0.2, fast: 0.7, skip: 1.5, slowRate: 1.12, fastRate: 1.25 };
    this.dropped = 0;
    this.rate = 1;
    this.frameLen = Math.round(sampleRate * 0.02);
    this.hopOut = Math.round(sampleRate * 0.01);
    this.hopIn = this.hopOut;
    this.tolerance = Math.round(sampleRate * 0.006);
    this.stretch = null;
    this.outBuf = [];
    this.outPos = 0;
    this.queue = [];
    this.headOffset = 0;
    this.queuedSamples = 0;
    this.started = false;
    this.lastSample = 0;
    this.fadeRemaining = 0;
    this.fadeLength = Math.max(1, Math.round(sampleRate * 0.01));
    this.playedSamples = 0;
    this.statusCountdown = 0;
    this.statusEnergy = 0;
    this.statusSamples = 0;
    // An underrun is the queue running dry while a response is still open; the
    // gap after response.done (flush) is the model going quiet, not a stall.
    this.responseOpen = false;
    this.underruns = 0;
    this.port.onmessage = ({ data }) => {
      if (data.type === "push" && data.samples) {
        this.queue.push(data.samples);
        this.queuedSamples += data.samples.length;
        // A network stall lets audio pile up while the server keeps generating in
        // real time; without a cap the backlog never shrinks and playback drifts
        // behind the transcript. Skip ahead by dropping the oldest samples.
        if (this.queuedSamples > this.maxSamples) this.trim(this.keepSamples);
      } else if (data.type === "config") {
        // Sent once the server grants its unit size; tunes buffering per model.
        if (data.initialSamples) this.initialSamples = this.target = data.initialSamples;
        if (data.maxJitterSamples) this.maxJitterSamples = data.maxJitterSamples;
        if (data.maxSamples) this.maxSamples = data.maxSamples;
        if (data.keepSamples) this.keepSamples = data.keepSamples;
        if (data.catchUp) this.catchUp = { ...this.catchUp, ...data.catchUp };
      } else if (data.type === "trim") {
        this.trim(data.keepSamples || 0);
      } else if (data.type === "open") {
        this.responseOpen = true;
      } else if (data.type === "clear") {
        this.queue = [];
        this.headOffset = 0;
        this.queuedSamples = 0;
        this.started = false;
        this.fadeRemaining = this.fadeLength;
        this.stretch = null;
        this.rate = 1;
        this.outBuf = [];
        this.outPos = 0;
      } else if (data.type === "flush") {
        this.responseOpen = false;
        if (this.queuedSamples > 0) this.started = true;
      } else if (data.type === "reset") {
        this.responseOpen = false;
        this.target = this.initialSamples;
        this.sinceUnderrun = 0;
        this.underruns = 0;
        this.queue = [];
        this.headOffset = 0;
        this.queuedSamples = 0;
        this.started = false;
        this.fadeRemaining = this.fadeLength;
        this.playedSamples = 0;
        this.stretch = null;
        this.rate = 1;
        this.outBuf = [];
        this.outPos = 0;
      }
    };
  }

  trim(keepSamples) {
    while (this.queuedSamples > keepSamples && this.queue.length) {
      const chunk = this.queue[0];
      const remaining = chunk.length - this.headOffset;
      const excess = this.queuedSamples - keepSamples;
      if (remaining <= excess) {
        this.queue.shift();
        this.headOffset = 0;
        this.queuedSamples -= remaining;
        this.dropped += remaining;
      } else {
        this.headOffset += excess;
        this.queuedSamples -= excess;
        this.dropped += excess;
      }
    }
    this.fadeRemaining = this.fadeLength;
    this.stretch = null;
  }

  // Seconds-old speech is stale in a duplex talk, so a large backlog is skipped instead of stretched.
  catchUpRate() {
    const { slow, fast, skip, slowRate, fastRate } = this.catchUp;
    const backlog = (this.queuedSamples - this.target) / sampleRate;
    if (backlog > skip) {
      this.trim(this.target + Math.round(sampleRate * fast));
      return fastRate;
    } else if (backlog > fast) return fastRate;
    else if (backlog > slow) return slowRate;
    else if (this.rate > 1 && backlog > slow / 2) return this.rate;
    else return 1;
  }

  peek(count) {
    if (this.queuedSamples < count) return null;
    const view = new Float32Array(count);
    let filled = 0;
    let offset = this.headOffset;
    for (let index = 0; index < this.queue.length && filled < count; index += 1) {
      const chunk = this.queue[index];
      const take = Math.min(chunk.length - offset, count - filled);
      view.set(chunk.subarray(offset, offset + take), filled);
      filled += take;
      offset = 0;
    }
    return view;
  }

  consume(count) {
    let left = count;
    while (left > 0 && this.queue.length) {
      const chunk = this.queue[0];
      const remaining = chunk.length - this.headOffset;
      if (remaining <= left) {
        this.queue.shift();
        this.headOffset = 0;
        left -= remaining;
        this.queuedSamples -= remaining;
      } else {
        this.headOffset += left;
        this.queuedSamples -= left;
        left = 0;
      }
    }
  }

  fill(count) {
    while (this.outBuf.length - this.outPos < count) {
      const rate = this.catchUpRate();
      if (rate !== this.rate) this.enterRate(rate);
      if (this.rate === 1) {
        const need = count - (this.outBuf.length - this.outPos);
        const take = Math.min(need, this.queuedSamples);
        if (take <= 0) return;
        const view = this.peek(take);
        for (let index = 0; index < take; index += 1) this.outBuf.push(view[index]);
        this.consume(take);
      } else if (!this.stretchFrame()) return;
    }
  }

  enterRate(rate) {
    if (this.rate > 1 && this.stretch) {
      // Leave stretch mode at the natural continuation of the last frame.
      this.consume(Math.min(this.queuedSamples, this.stretch.prevStart + this.hopOut));
      this.stretch = null;
    }
    this.rate = rate;
    this.hopIn = Math.round(this.hopOut * rate);
  }

  stretchFrame() {
    const N = this.frameLen;
    const Hs = this.hopOut;
    if (!this.stretch) {
      const frame = this.peek(N);
      if (!frame) return false;
      for (let index = 0; index < Hs; index += 1) this.outBuf.push(frame[index]);
      this.stretch = { tail: frame.slice(Hs, N), prevStart: 0, inPos: this.hopIn };
      return true;
    }
    const state = this.stretch;
    const nominal = Math.round(state.inPos);
    const lo = Math.max(0, nominal - this.tolerance);
    const hi = nominal + this.tolerance;
    const view = this.peek(hi + N);
    if (!view) return false;
    let best = nominal;
    let bestScore = -Infinity;
    const tail = state.tail;
    for (let start = lo; start <= hi; start += 1) {
      let dot = 0;
      let energy = 1e-9;
      for (let index = 0; index < Hs; index += 1) {
        const sample = view[start + index];
        dot += sample * tail[index];
        energy += sample * sample;
      }
      const score = dot / Math.sqrt(energy);
      if (score > bestScore) {
        bestScore = score;
        best = start;
      }
    }
    for (let index = 0; index < Hs; index += 1) {
      const weight = index / Hs;
      this.outBuf.push(tail[index] * (1 - weight) + view[best + index] * weight);
    }
    state.tail = view.slice(best + Hs, best + N);
    state.prevStart = best;
    state.inPos += this.hopIn;
    const release = Math.max(0, Math.min(state.prevStart, Math.round(state.inPos) - this.tolerance));
    if (release > 0) {
      this.consume(release);
      state.prevStart -= release;
      state.inPos -= release;
    }
    return true;
  }

  takeSample() {
    if (this.outPos >= this.outBuf.length) return null;
    const value = this.outBuf[this.outPos];
    this.outPos += 1;
    this.playedSamples += 1;
    if (this.outPos >= 4096) {
      this.outBuf = this.outBuf.slice(this.outPos);
      this.outPos = 0;
    }
    return value;
  }

  process(_inputs, outputs) {
    const output = outputs[0][0];
    if (!this.started && this.queuedSamples >= this.target) this.started = true;
    this.sinceUnderrun += output.length;
    if (this.target > this.initialSamples && this.sinceUnderrun > sampleRate * 10) {
      this.target = Math.max(this.initialSamples, this.target - Math.round(sampleRate * 0.05));
      this.sinceUnderrun = 0;
    }
    if (this.started) this.fill(output.length);
    for (let index = 0; index < output.length; index += 1) {
      let value = 0;
      if (this.fadeRemaining > 0) {
        value = this.lastSample * (this.fadeRemaining / this.fadeLength);
        this.fadeRemaining -= 1;
      } else if (this.started) {
        const next = this.takeSample();
        if (next === null) {
          this.started = false;
          if (this.responseOpen) {
            this.underruns += 1;
            this.target = Math.min(this.maxJitterSamples, this.target + Math.round(sampleRate * 0.1));
            this.sinceUnderrun = 0;
          }
        } else value = next;
      }
      output[index] = value;
      this.lastSample = value;
      this.statusEnergy += value * value;
      this.statusSamples += 1;
    }
    this.statusCountdown -= output.length;
    if (this.statusCountdown <= 0) {
      this.statusCountdown = Math.round(sampleRate / 10);
      this.port.postMessage({
        type: "status",
        droppedMs: Math.round((this.dropped / sampleRate) * 1000),
        rate: this.rate,
        queueMs: this.queuedSamples / sampleRate * 1000,
        targetMs: this.target / sampleRate * 1000,
        playedMs: this.playedSamples / sampleRate * 1000,
        buffering: !this.started,
        underruns: this.underruns,
        outputRms: this.statusSamples ? Math.sqrt(this.statusEnergy / this.statusSamples) : 0,
      });
      this.statusEnergy = 0;
      this.statusSamples = 0;
    }
    return true;
  }
}

registerProcessor("capture-processor", CaptureProcessor);
registerProcessor("playback-processor", PlaybackProcessor);
