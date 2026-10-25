// Protocol state for one realtime session (native full duplex or turn based); no DOM, no audio.
// app.js wires it to the page; the Node tests drive it with a fake socket.

// The server grants the input rate in session.updated; this is only the fallback.
export const DEFAULT_INPUT_RATE = 16000;
export const PACKET_MS = 80;
const MAX_UNITS_SHOWN = 200;
const MAX_REPLIES_KEPT = 20;

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

// How much of a reply's text the listener has heard: the largest text_end whose audio ends by playedMs.
export function heardTextEnd(progress, playedMs) {
  let textEnd = 0;
  for (const point of progress) {
    if (point.audioMs <= playedMs) textEnd = Math.max(textEnd, point.textEnd);
  }
  return textEnd;
}

// note (Haoling Pu): the server counts text_end in code points, while String.slice counts UTF-16 code units.
function codeUnitOffset(text, codePoints) {
  let offset = 0;
  let count = 0;
  for (const char of text) {
    if (count === codePoints) break;
    offset += char.length;
    count += 1;
  }
  return offset;
}

export class DuplexSession {
  // transport(event) sends one JSON-serialisable client event; now() returns ms.
  constructor({ transport, now, outputModalities = ["audio"], listeners = {} }) {
    this.transport = transport;
    this.now = now;
    this.outputModalities = outputModalities;
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
    this.responseId = null;
    // Turn-based servers report the user's speech from their VAD.
    this.userSpeaking = false;
    this.transcript = "";
    this.lastLatencyMs = null;
    this.errors = [];
    this.drained = null;
    this.closeReason = null;
    // Per response: its text, item, the server's sglang.response.progress points and how much of its audio has played.
    this.replies = new Map();
    this.lastReplyId = null;
  }

  reply(responseId) {
    return this.replies.get(responseId) || null;
  }

  // Characters of a reply's text the listener has heard; fixed once the reply stops playing.
  heardEnd(responseId) {
    const reply = this.reply(responseId);
    return reply ? Math.min(heardTextEnd(reply.progress, reply.playedMs), reply.text.length) : 0;
  }

  // What the caption shows of a reply: the heard part and the part still to play. A reply that ended
  // other than completed, or that the player stopped, keeps only what was heard: the prefix the
  // server's history kept once it reports one, else the progress the player passed, and nothing when
  // none of its audio played. A reply without progress from the server is shown whole until it is cut.
  captionParts(responseId) {
    const reply = this.reply(responseId);
    if (!reply) return { heard: "", pending: "" };
    const cut = (reply.status !== null && reply.status !== "completed") || reply.stopped;
    if (cut && reply.playedMs <= 0) return { heard: "", pending: "" };
    let end;
    if (cut && reply.keptEnd !== null) {
      end = codeUnitOffset(reply.text, reply.keptEnd);
    } else if (reply.progress.length) {
      end = codeUnitOffset(reply.text, this.heardEnd(responseId));
    } else if (cut) {
      end = 0;
    } else {
      end = reply.text.length;
    }
    return { heard: reply.text.slice(0, end), pending: cut ? "" : reply.text.slice(end) };
  }

  // The player's count of a reply's audio; a cut reply tells the server how much was heard.
  played(responseId, playedMs, cut) {
    const reply = this.reply(responseId);
    if (!reply || reply.stopped) return;
    reply.playedMs = Math.max(reply.playedMs, playedMs);
    if (cut) {
      reply.stopped = true;
      if (this.state === "ready" && this.granted.supports_truncate && reply.itemId) {
        this.send("conversation.item.truncate", { item_id: reply.itemId, content_index: 0, audio_end_ms: reply.playedMs });
      }
    }
    this.emit("played", responseId);
  }

  get turnBased() {
    return Boolean(this.granted) && this.granted.interaction === "turn_based";
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

  send(type, payload = {}) {
    const event = { type, event_id: `web-${++this.eventSequence}`, ...payload };
    this.record("send", event);
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
    return sent;
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
        this.send("session.update", { session: { output_modalities: this.outputModalities } });
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
        this.emit("granted", this.granted);
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
      case "input_audio_buffer.speech_started":
      case "input_audio_buffer.speech_stopped":
        this.userSpeaking = event.type === "input_audio_buffer.speech_started";
        break;
      case "response.created":
        this.responseOpen = true;
        this.responseId = (event.response && event.response.id) || null;
        this.replies.set(this.responseId, { text: "", itemId: null, progress: [], playedMs: 0, stopped: false, status: null, keptEnd: null });
        this.lastReplyId = this.responseId;
        if (this.replies.size > MAX_REPLIES_KEPT) this.replies.delete(this.replies.keys().next().value);
        this.emit("response", { open: true, id: this.responseId });
        break;
      case "response.output_audio.delta": {
        const bytes = base64ToBytes(event.delta || "");
        const unitId = extension.unit_id || "unknown";
        this.unitAudio.set(unitId, (this.unitAudio.get(unitId) || 0) + bytes.byteLength / 2);
        const reply = this.reply(event.response_id);
        if (reply) reply.itemId = event.item_id;
        this.emit("audio", { bytes, responseId: event.response_id || this.responseId });
        break;
      }
      case "response.output_text.delta":
      case "response.output_audio_transcript.delta":
        this.transcript += event.delta || "";
        if (this.reply(event.response_id)) this.reply(event.response_id).text += event.delta || "";
        this.emit("text", event.delta || "");
        break;
      case "sglang.response.progress": {
        const reply = this.reply(event.response_id);
        if (reply) reply.progress.push({ textEnd: event.text_end, audioMs: event.audio_ms });
        this.emit("progress", event.response_id);
        break;
      }
      case "sglang.response.heard": {
        const reply = this.reply(event.response_id);
        if (reply) reply.keptEnd = event.text_end;
        this.emit("progress", event.response_id);
        break;
      }
      case "response.done": {
        const reply = this.reply(event.response && event.response.id);
        if (reply) reply.status = event.response.status || null;
        this.responseOpen = false;
        this.responseId = null;
        if (this.transcript && !this.transcript.endsWith("\n")) this.transcript += "\n";
        this.emit("response", { open: false, id: event.response && event.response.id, status: event.response && event.response.status });
        break;
      }
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
