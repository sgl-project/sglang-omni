// Protocol state for one native full-duplex realtime session; no DOM, no audio.
// app.js wires it to the page; the Node tests drive it with a fake socket.

// The server grants the input rate in session.updated; this is only the fallback.
export const DEFAULT_INPUT_RATE = 16000;
export const PACKET_MS = 80;
const MAX_UNITS_SHOWN = 200;

// Image wire names; the server declares image input in session.updated and the page follows.
export const IMAGE = {
  GRANTED_FIELD: "input_modalities",
  MODALITY: "image",
  FORMAT_FIELD: "input_image_format",
  FRAME_EVENT: "sglang.input_image.append",
  PAYLOAD_FIELD: "image",
  TIME_FIELD: "t_ms",
  ACK_EVENT: "sglang.input_image.accepted",
};
// The model rescales frames to about 448x448, so a smaller short side would only be upscaled.
export const IMAGE_DEFAULTS = { types: ["image/jpeg"], max_bytes: 512 * 1024, max_frames_per_unit: 1, max_short_side: 448 };

export function bytesToBase64(bytes) {
  let binary = "";
  const stride = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += stride) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + stride));
  }
  return btoa(binary);
}

export function base64ToBytes(encoded) {
  const binary = atob(encoded);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return bytes;
}

export class DuplexSession {
  // transport(event) sends one JSON-serialisable client event; now() returns ms.
  // extension is sent as session.sglang (sampling, reference_audio, max_slice_nums, ...).
  constructor({ transport, now, outputModalities = ["audio"], instructions = "", extension = null, listeners = {} }) {
    this.transport = transport;
    this.now = now;
    this.outputModalities = outputModalities;
    this.instructions = instructions;
    this.extension = extension;
    this.listeners = listeners;
    this.state = "connecting";
    this.sessionId = null;
    this.granted = null;
    this.eventSequence = 0;
    this.trace = [];
    this.packets = [];
    this.nextSeq = 0;
    this.sentSamples = 0;
    this.pending = new Int16Array(0);
    this.inputPaused = false;
    this.abandoned = new Set();
    this.units = [];
    this.unitAudio = new Map();
    this.responseOpen = false;
    this.transcript = "";
    this.lastLatencyMs = null;
    this.errors = [];
    this.drained = null;
    this.closeReason = null;
    // Camera frames: one per native unit, stamped on the audio media clock.
    this.image = null;
    this.frameSource = null;
    this.frameBusy = false;
    this.frameGeneration = 0;
    this.lastFrameUnit = -1;
    this.frames = [];
    this.framesSkipped = 0;
  }

  // Granted image parameters ({format, maxBytes, maxShortSide}) or null.
  get imageGranted() {
    return this.image !== null;
  }

  // Media clock (ms) of the newest input sample: the clock of t_start_ms.
  get inputClockMs() {
    return ((this.sentSamples + this.pending.length) * 1000) / this.inputRate;
  }

  get frameStats() {
    let accepted = 0;
    let rejected = 0;
    for (const frame of this.frames) {
      if (frame.status === "accepted") accepted += 1;
      else if (frame.status === "rejected") rejected += 1;
    }
    return { sent: this.frames.length, accepted, rejected, skipped: this.framesSkipped };
  }

  get unitMs() {
    return this.granted ? this.granted.native_unit_ms : null;
  }

  // Input PCM rate the server granted; the microphone is resampled to it.
  get inputRate() {
    const format = this.granted && this.granted.input_audio_format;
    return format && format.rate ? format.rate : DEFAULT_INPUT_RATE;
  }

  get packetSamples() {
    return Math.round((this.inputRate * PACKET_MS) / 1000);
  }

  get outputRate() {
    const format = this.granted && this.granted.output_audio_format;
    return format && format.rate ? format.rate : 24000;
  }

  emit(name, payload) {
    const listener = this.listeners[name];
    if (listener) listener(payload);
  }

  record(direction, event, extra = null) {
    this.trace.push({ direction, time_s: this.now() / 1000, event, ...extra });
  }

  // traced(event) replaces what the trace keeps (e.g. without the JPEG body).
  send(type, payload = {}, traced = null) {
    const event = { type, event_id: `web-${++this.eventSequence}`, ...payload };
    if (traced) this.record("send", traced.event(event), traced.extra);
    else this.record("send", event);
    this.transport(event);
    return event;
  }

  // Microphone or file PCM16 at 16 kHz, any length; sent as 80 ms packets.
  pushInput(samples) {
    if (this.state !== "ready") return 0;
    const merged = new Int16Array(this.pending.length + samples.length);
    merged.set(this.pending);
    merged.set(samples, this.pending.length);
    let offset = 0;
    let sent = 0;
    const size = this.packetSamples;
    while (merged.length - offset >= size) {
      this.queuePacket(merged.slice(offset, offset + size));
      offset += size;
      sent += 1;
    }
    this.pending = merged.slice(offset);
    this.pump();
    this.maybeCaptureFrame();
    return sent;
  }

  // source() resolves to {base64, bytes} or null (no frame ready); null turns frames off.
  setFrameSource(source) {
    this.frameSource = source;
    this.stopFrames();
  }

  // Drops any capture in flight; called when the microphone stops.
  stopFrames() {
    this.frameGeneration += 1;
    this.frameBusy = false;
  }

  // One frame per native unit: capture on the first input push at or past
  // k x unit on the media clock. A capture still running when the next
  // boundary passes makes that unit skip its frame; nothing is queued.
  maybeCaptureFrame() {
    if (!this.image || !this.frameSource || this.frameBusy || this.inputPaused || this.state !== "ready") return;
    const clockMs = this.inputClockMs;
    const unit = Math.floor(clockMs / this.unitMs);
    if (unit <= this.lastFrameUnit) return;
    this.lastFrameUnit = unit;
    this.frameBusy = true;
    const generation = this.frameGeneration;
    const tMs = Math.round(clockMs);
    Promise.resolve()
      .then(() => this.frameSource())
      .then((frame) => {
        if (generation !== this.frameGeneration) return;
        this.frameBusy = false;
        if (frame) this.sendFrame(frame, tMs);
        else this.skipFrame(unit);
      })
      .catch((error) => {
        if (generation !== this.frameGeneration) return;
        this.frameBusy = false;
        this.emit("warning", `camera frame: ${error.message}`);
      });
  }

  // The server binds a frame to unit floor(t_ms / unit) and rejects it once the
  // audio completing that unit has arrived, so a grab that finished too late is dropped.
  sendFrame(frame, tMs) {
    if (!this.image || this.state !== "ready") return null;
    const unit = Math.floor(tMs / this.unitMs);
    if (this.inputClockMs >= (unit + 1) * this.unitMs || frame.bytes > this.image.maxBytes) {
      this.skipFrame(unit);
      return null;
    }
    // seq is local (UI and trace only); the wire event carries t_ms alone.
    const seq = this.frames.length;
    const traceFrame = { seq, unit, t_ms: tMs, bytes: frame.bytes };
    const event = this.send(IMAGE.FRAME_EVENT, { [IMAGE.PAYLOAD_FIELD]: frame.base64, sglang: { [IMAGE.TIME_FIELD]: tMs } }, {
      event: (sent) => ({ ...sent, [IMAGE.PAYLOAD_FIELD]: `<${frame.bytes} bytes>` }),
      extra: { frame: traceFrame },
    });
    const record = { seq, unit, tMs, bytes: frame.bytes, eventId: event.event_id, sendMs: this.now(), status: "sent", ackLatencyMs: null, error: null };
    this.frames.push(record);
    this.emit("frame", { ...this.frameStats, last: record });
    return record;
  }

  skipFrame(unit) {
    this.framesSkipped += 1;
    this.emit("frame", { ...this.frameStats, skippedUnit: unit });
  }

  frameFor(eventId) {
    return eventId ? this.frames.find((frame) => frame.eventId === eventId) || null : null;
  }

  queuePacket(pcm) {
    const seq = this.nextSeq++;
    const tStartMs = (this.sentSamples * 1000) / this.inputRate;
    this.sentSamples += pcm.length;
    this.packets.push({ seq, tStartMs, samples: pcm.length, pcm, sendMs: null, eventId: null, acked: false });
  }

  pump() {
    if (this.inputPaused || this.state !== "ready") return;
    for (const packet of this.packets) {
      if (packet.eventId !== null) continue;
      const event = this.send("input_audio_buffer.append", {
        audio: bytesToBase64(new Uint8Array(packet.pcm.buffer, packet.pcm.byteOffset, packet.pcm.byteLength)),
        sglang: { seq: packet.seq, t_start_ms: packet.tStartMs },
      });
      packet.eventId = event.event_id;
      packet.sendMs = this.now();
    }
  }

  endInput() {
    if (this.state !== "ready") return;
    if (this.pending.length) {
      this.queuePacket(this.pending);
      this.pending = new Int16Array(0);
    }
    this.pump();
    this.send("sglang.input_audio.end");
    this.state = "ending";
    this.emit("state", this.state);
  }

  close() {
    if (["closing", "closed"].includes(this.state)) return;
    this.send("session.close");
    this.state = "closing";
    this.emit("state", this.state);
  }

  // Everything this session sent and received, audio included, for replaying a report offline.
  exportTrace() {
    return JSON.stringify({ session_id: this.sessionId, granted: this.granted, close_reason: this.closeReason, trace: this.trace });
  }

  packetFor(eventId) {
    return this.packets.find((packet) => packet.eventId === eventId) || null;
  }

  // Last input packet whose audio overlaps [.., endMs): the unit could not complete earlier.
  lastPacketBefore(endMs) {
    let found = null;
    for (const packet of this.packets) {
      if (packet.tStartMs < endMs - 1e-6 && packet.sendMs !== null) found = packet;
      else if (packet.tStartMs >= endMs) break;
    }
    return found;
  }

  handle(event) {
    this.record("receive", event);
    const extension = event.sglang || {};
    switch (event.type) {
      case "session.created": {
        this.sessionId = event.session && event.session.id;
        this.state = "negotiating";
        const session = { output_modalities: this.outputModalities };
        if (this.instructions) session.instructions = this.instructions;
        if (this.extension && Object.keys(this.extension).length) session.sglang = this.extension;
        this.send("session.update", { session });
        break;
      }
      case "session.updated": {
        this.granted = event.session.sglang.granted;
        for (const rejection of this.granted.rejections || []) {
          this.emit("warning", `setting ${rejection.field} not applied: ${rejection.reason}`);
        }
        const inputRate = this.granted.input_audio_format && this.granted.input_audio_format.rate;
        if (!(inputRate > 0)) {
          this.fail(`server granted no usable input rate (${inputRate})`);
          break;
        }
        if (this.state === "negotiating") this.state = "ready";
        const modalities = this.granted[IMAGE.GRANTED_FIELD];
        if (Array.isArray(modalities) && modalities.includes(IMAGE.MODALITY)) {
          const format = { ...IMAGE_DEFAULTS, ...(this.granted[IMAGE.FORMAT_FIELD] || {}) };
          const types = Array.isArray(format.types) && format.types.length ? format.types : IMAGE_DEFAULTS.types;
          this.image = { format: types[0], maxBytes: format.max_bytes, maxShortSide: IMAGE_DEFAULTS.max_short_side };
        } else {
          this.image = null;
          this.stopFrames();
        }
        this.emit("granted", this.granted);
        this.emit("image", this.image);
        break;
      }
      case "sglang.input_audio.accepted": {
        const packet = this.packetFor(event.client_event_id);
        if (packet) {
          // Acked packets are kept for latency lookups only; drop their PCM.
          packet.acked = true;
          packet.pcm = null;
        }
        break;
      }
      case IMAGE.ACK_EVENT: {
        // The ack names the frame by its event_id (client_event_id also accepted).
        const frame = this.frameFor(event.client_event_id) || this.frameFor(event.event_id);
        if (frame && frame.status === "sent") {
          frame.status = "accepted";
          frame.ackLatencyMs = this.now() - frame.sendMs;
          this.trace.at(-1).frame = { seq: frame.seq, unit: frame.unit, t_ms: frame.tMs, bytes: frame.bytes, ack_latency_ms: frame.ackLatencyMs };
          this.emit("frame", { ...this.frameStats, last: frame });
        }
        break;
      }
      case "response.created":
        this.responseOpen = true;
        this.emit("response", { open: true, id: event.response && event.response.id });
        break;
      case "response.output_audio.delta": {
        const bytes = base64ToBytes(event.delta || "");
        const unitId = extension.unit_id || "unknown";
        this.unitAudio.set(unitId, (this.unitAudio.get(unitId) || 0) + bytes.byteLength / 2);
        this.emit("audio", bytes);
        break;
      }
      case "response.output_text.delta":
      case "response.output_audio_transcript.delta":
        this.transcript += event.delta || "";
        this.emit("text", event.delta || "");
        break;
      case "response.done":
        this.responseOpen = false;
        if (this.transcript && !this.transcript.endsWith("\n")) this.transcript += "\n";
        this.emit("response", { open: false, id: event.response && event.response.id, status: event.response && event.response.status });
        break;
      case "sglang.unit.done":
        this.completeUnit(event);
        break;
      case "sglang.input_audio.ended":
        this.emit("state", this.state);
        break;
      case "sglang.input_audio.drained":
        this.drained = event;
        this.emit("drained", event);
        this.close();
        break;
      case "error":
        this.handleError(event);
        break;
      case "session.closed":
        this.state = "closed";
        this.closeReason = event.reason || "unknown";
        this.emit("state", this.state);
        break;
      default:
        break;
    }
    return event;
  }

  completeUnit(event) {
    const extension = event.sglang || {};
    const unitId = event.unit_id || extension.unit_id;
    const media = extension.media_time || null;
    const audioSamples = this.unitAudio.get(unitId) || 0;
    this.unitAudio.delete(unitId);
    // The shared runtime reports `decision` when server timing is on; otherwise infer from audio.
    const decision = event.decision && event.decision !== "unknown"
      ? event.decision
      : audioSamples > 0 ? "speak" : "listen";
    let latencyMs = null;
    if (media) {
      const packet = this.lastPacketBefore(media.t_start_ms + Math.max(media.duration_ms, 1e-3));
      if (packet) latencyMs = this.now() - packet.sendMs;
    }
    if (latencyMs !== null) this.lastLatencyMs = latencyMs;
    const unit = {
      unitId,
      decision,
      inferred: !event.decision || event.decision === "unknown",
      audioMs: (audioSamples * 1000) / this.outputRate,
      mediaStartMs: media ? media.t_start_ms : null,
      mediaDurationMs: media ? media.duration_ms : null,
      latencyMs,
      computeMs: Number.isFinite(event.compute_ms) ? event.compute_ms : null,
    };
    this.units.push(unit);
    if (this.units.length > MAX_UNITS_SHOWN) this.units.splice(0, this.units.length - MAX_UNITS_SHOWN);
    if (this.inputPaused) {
      this.inputPaused = false;
      this.pump();
    }
    this.emit("unit", unit);
  }

  handleError(event) {
    const error = event.error || {};
    const extension = event.sglang || {};
    if (!extension.fatal && error.event_id && this.abandoned.has(error.event_id)) return;
    const frame = this.frameFor(error.event_id);
    if (!extension.fatal && frame) {
      // invalid_state (late, second frame, after input end), buffer_overflow (too far
      // ahead) or not_supported: count it and keep going; audio is unaffected.
      frame.status = "rejected";
      frame.error = error.code || "error";
      this.trace.at(-1).frame = { seq: frame.seq, unit: frame.unit, t_ms: frame.tMs, bytes: frame.bytes, rejected: frame.error };
      this.errors.push(error);
      this.emit("frame", { ...this.frameStats, last: frame });
      return;
    }
    const packet = error.event_id ? this.packetFor(error.event_id) : null;
    if (!extension.fatal && packet && error.code === "buffer_overflow") {
      // pressure_policy "reject": rewind to the first unacked packet and resend
      // after the next unit completes. Appends already in flight will be
      // rejected as non-contiguous; their errors are ignored.
      this.inputPaused = true;
      for (const pending of this.packets) {
        if (!pending.acked && pending.eventId !== null) {
          this.abandoned.add(pending.eventId);
          pending.eventId = null;
          pending.sendMs = null;
        }
      }
      this.emit("warning", `append seq ${packet.seq} rejected (buffer_overflow); resending after the next unit`);
      return;
    }
    this.errors.push(error);
    if (extension.fatal) {
      this.state = "error";
      this.emit("state", this.state);
    }
    this.emit("warning", `${error.code || "error"}: ${error.message || "unknown error"}`);
  }

  fail(message) {
    this.errors.push({ code: "client", message });
    this.emit("warning", message);
    this.close();
  }

  traceJsonl() {
    return this.trace.map((row) => JSON.stringify(row)).join("\n") + (this.trace.length ? "\n" : "");
  }
}
