// When a reply's audio starts against the picture of a plugin output port that follows replies (params.follows_replies), for every
// renderer of such a format. "hold": the reply's audio waits until a renderer shows the reply's first frame, so the voice never
// leads the picture, at the cost of the model's visual latency before the first word. "free": the audio plays on arrival and the
// picture follows when it comes. An interrupted reply drops its held audio with its frames, by response id.
// No DOM, no audio: app.js hands it the playback worklet's postMessage; the Node tests drive it with a fake clock.

export const AV_SYNC_POLICIES = ["hold", "free"];
// A reply whose first frame is later than its port's params.reply_media_wait_ms plays unheld; a port that names none waits this long.
export const DEFAULT_REPLY_MEDIA_WAIT_MS = 1500;

export class ReplyMediaGate {
  // policy: one of AV_SYNC_POLICIES; now(): ms; post(message, transfer): the playback worklet's postMessage.
  constructor({ policy, now, post }) {
    if (!AV_SYNC_POLICIES.includes(policy)) throw new Error(`unknown A/V sync policy ${policy}`);
    this.policy = policy;
    this.now = now;
    this.post = post;
    this.reset();
  }

  reset() {
    this.isGating = false;
    this.waitMs = DEFAULT_REPLY_MEDIA_WAIT_MS;
    // responseId -> { sinceMs, samples, ended }: replies whose audio waits for their first frame.
    this.held = new Map();
    this.shownReplies = new Set();
  }

  // The plugin output ports session.updated advertised; draws(format) says whether the page renders a format. With several such
  // ports a reply waits for the longest of their waits.
  follow(ports, draws) {
    const gating = ports.filter((port) => port.params && port.params.follows_replies && draws(port.format));
    this.isGating = this.policy === "hold" && gating.length > 0;
    this.waitMs = gating.length > 0 ? Math.max(...gating.map((port) => port.params.reply_media_wait_ms ?? DEFAULT_REPLY_MEDIA_WAIT_MS)) : DEFAULT_REPLY_MEDIA_WAIT_MS;
  }

  audio(samples, responseId) {
    const held = this.held.get(responseId);
    if (held) held.samples.push(samples);
    else if (this.isGating && responseId && !this.shownReplies.has(responseId)) this.held.set(responseId, { sinceMs: this.now(), samples: [samples], ended: false });
    else this.post({ type: "push", samples, responseId }, [samples.buffer]);
  }

  // The reply ended: a cancelled one stops with its held audio, a completed one plays out once released.
  end(responseId, status) {
    const held = this.held.get(responseId);
    if (status === "cancelled") {
      this.held.delete(responseId);
      this.post({ type: "cancel", responseId });
    } else if (held) held.ended = true;
    else this.post({ type: "flush", responseId });
  }

  // A renderer showed the reply's first frame.
  shown(responseId) {
    this.shownReplies.add(responseId);
    const held = this.held.get(responseId);
    if (!held) return;
    this.held.delete(responseId);
    for (const samples of held.samples) this.post({ type: "push", samples, responseId }, [samples.buffer]);
    if (held.ended) this.post({ type: "flush", responseId });
  }

  // Called every animation frame: a reply held past the wait plays without its picture.
  tick() {
    const now = this.now();
    for (const [responseId, held] of this.held) {
      if (now - held.sinceMs >= this.waitMs) this.shown(responseId);
    }
  }
}
