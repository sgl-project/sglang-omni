// When a reply's audio starts against the picture of a plugin output port that follows replies (params.follows_replies), for every
// renderer of such a format. "hold": the reply's audio waits until a renderer shows the reply's first frame, so the voice never
// leads the picture, at the cost of the model's visual latency before the first word. "free": the audio plays on arrival and the
// picture follows when it comes. An interrupted reply drops its held audio with its frames, by response id.
// A turnless reply stream (a native full-duplex model: one reply for the whole session, its frames in no turn) has no first frame to
// wait for, so "hold" aligns it instead: audio and pictures share one media clock, anchored at the first reply audio's arrival plus the
// align delay, the port's params.reply_lookahead_ms (how much reply audio past a frame's time the node hears before it makes the frame;
// its reply_media_wait_ms when it names none) at most its reply_media_wait_ms, plus the renderers' playout delay. Reply audio plays at
// its place on that clock, the reply audio received before it, and a renderer given the gate as its clock shows frame t at origin + t.
// No DOM, no audio: app.js hands it the playback worklet's postMessage; the Node tests drive it with a fake clock.

export const AV_SYNC_POLICIES = ["hold", "free"];
// A reply whose first frame is later than its port's params.reply_media_wait_ms plays unheld; a port that names none waits this long.
export const DEFAULT_REPLY_MEDIA_WAIT_MS = 1500;
// A renderer plays a plugin port's media this long after its first chunk arrived, a jitter buffer for later chunks that arrive a little
// late; a reply's first frame shows that much later, so a held reply's audio may wait that much longer.
export const DEFAULT_PLAYOUT_DELAY_MS = 100;

// The sample rate of the samples the page hands audio() when follow() names none.
const DEFAULT_SAMPLE_RATE = 24000;

// The playout delay a page's ?playout_delay_ms= names; the default when it names none or not a number of ms at least 0.
export function playoutDelayFrom(params) {
  const value = params.get("playout_delay_ms");
  const ms = value === null || value.trim() === "" ? NaN : Number(value);
  return Number.isFinite(ms) && ms >= 0 ? ms : DEFAULT_PLAYOUT_DELAY_MS;
}

export class ReplyMediaGate {
  // policy: one of AV_SYNC_POLICIES; now(): ms; post(message, transfer): the playback worklet's postMessage; playoutDelayMs: the
  // renderers' playout delay, added to every wait.
  constructor({ policy, now, post, playoutDelayMs = DEFAULT_PLAYOUT_DELAY_MS }) {
    if (!AV_SYNC_POLICIES.includes(policy)) throw new Error(`unknown A/V sync policy ${policy}`);
    this.policy = policy;
    this.now = now;
    this.post = post;
    this.playoutDelayMs = playoutDelayMs;
    this.reset();
  }

  reset() {
    this.isGating = false;
    this.isAligning = false;
    this.waitMs = DEFAULT_REPLY_MEDIA_WAIT_MS + this.playoutDelayMs;
    this.alignDelayMs = this.waitMs;
    this.sampleRate = DEFAULT_SAMPLE_RATE;
    // responseId -> { sinceMs, samples, ended }: replies whose audio waits for their first frame.
    this.held = new Map();
    this.shownReplies = new Set();
    // Aligning: the shared media clock's origin (null until anchored), the place of the next reply audio on it, the audio not yet due
    // ({ atMs, samples, responseId }, in order), the completed replies whose flush waits for their last audio, and when a renderer first
    // asked for the origin.
    this.originMs = null;
    this.positionMs = 0;
    this.queue = [];
    this.ending = new Set();
    this.firstAskedMs = null;
  }

  // The plugin output ports session.updated advertised; draws(format) says whether the page renders a format. With several such
  // ports a reply waits for the longest of their waits, plus the renderers' playout delay. turnless: the session streams one reply
  // with no turns (a native full-duplex model), which "hold" aligns rather than gates; sampleRate: the rate of the samples audio() gets.
  follow(ports, draws, { turnless = false, sampleRate = DEFAULT_SAMPLE_RATE } = {}) {
    const gating = ports.filter((port) => port.params && port.params.follows_replies && draws(port.format));
    const following = this.policy === "hold" && gating.length > 0;
    this.isGating = following && !turnless;
    this.isAligning = following && turnless;
    this.sampleRate = sampleRate;
    const waitOf = (port) => port.params.reply_media_wait_ms ?? DEFAULT_REPLY_MEDIA_WAIT_MS;
    this.waitMs = this.playoutDelayMs + (gating.length > 0 ? Math.max(...gating.map(waitOf)) : DEFAULT_REPLY_MEDIA_WAIT_MS);
    this.alignDelayMs = gating.length > 0 ? this.playoutDelayMs + Math.max(...gating.map((port) => Math.min(port.params.reply_lookahead_ms ?? waitOf(port), waitOf(port)))) : this.waitMs;
  }

  // The renderers' media clock origin given their own proposal (first chunk arrival plus playout delay, minus its stamp): theirs unless
  // aligning; aligning, the first reply audio's anchor, null while there is none, and the renderer's own once no reply audio came within the wait.
  mediaOrigin(proposedMs) {
    if (!this.isAligning) return proposedMs;
    if (this.originMs === null) {
      const now = this.now();
      if (this.firstAskedMs === null) this.firstAskedMs = now;
      if (now - this.firstAskedMs >= this.waitMs) this.originMs = proposedMs;
    }
    return this.originMs;
  }

  audio(samples, responseId) {
    if (this.isAligning) {
      this.place();
      this.queue.push({ atMs: this.originMs + this.positionMs, samples, responseId });
      this.positionMs += (samples.length * 1000) / this.sampleRate;
      this.release();
      return;
    }
    const held = this.held.get(responseId);
    if (held) held.samples.push(samples);
    else if (this.isGating && responseId && !this.shownReplies.has(responseId)) this.held.set(responseId, { sinceMs: this.now(), samples: [samples], ended: false });
    else this.post({ type: "push", samples, responseId }, [samples.buffer]);
  }

  // The page dropped samples of the reply stream (a local mute): aligning, they keep their place on the media clock.
  skip(numSamples) {
    if (!this.isAligning) return;
    this.place();
    this.positionMs += (numSamples * 1000) / this.sampleRate;
  }

  // The first reply audio anchors the media clock: its place plays the align delay after its arrival.
  place() {
    if (this.originMs === null) this.originMs = this.now() + this.alignDelayMs - this.positionMs;
  }

  // Aligning: post the reply audio now due, and a completed reply's flush after its last audio.
  release() {
    const now = this.now();
    while (this.queue.length && this.queue[0].atMs <= now) {
      const { samples, responseId } = this.queue.shift();
      this.post({ type: "push", samples, responseId }, [samples.buffer]);
      if (this.ending.has(responseId) && !this.queue.some((entry) => entry.responseId === responseId)) {
        this.ending.delete(responseId);
        this.post({ type: "flush", responseId });
      }
    }
  }

  // The reply ended: a cancelled one stops with its held audio, a completed one plays out once released.
  end(responseId, status) {
    if (this.isAligning) {
      if (status === "cancelled") {
        this.queue = this.queue.filter((entry) => entry.responseId !== responseId);
        this.post({ type: "cancel", responseId });
      } else if (this.queue.some((entry) => entry.responseId === responseId)) this.ending.add(responseId);
      else this.post({ type: "flush", responseId });
      return;
    }
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

  // Called every animation frame: a reply held past the wait plays without its picture; aligned audio plays when due.
  tick() {
    if (this.isAligning) this.release();
    const now = this.now();
    for (const [responseId, held] of this.held) {
      if (now - held.sinceMs >= this.waitMs) this.shown(responseId);
    }
  }
}
