import assert from "node:assert/strict";
import test from "node:test";

import { DEFAULT_PLAYOUT_DELAY_MS, DEFAULT_REPLY_MEDIA_WAIT_MS, ReplyMediaGate, playoutDelayFrom } from "./reply_media_gate.js";

const FOLLOWING = { modality: "motion", format: "test_motion", params: { fps: 30, timeline: "input", follows_replies: true } };

function gate(policy, ports = [FOLLOWING], draws = () => true, options = { playoutDelayMs: 0 }) {
  const clock = { ms: 0 };
  const posted = [];
  const mediaGate = new ReplyMediaGate({ policy, now: () => clock.ms, post: (message) => posted.push([message.type, message.responseId]), ...options });
  mediaGate.follow(ports, draws);
  return { clock, posted, mediaGate };
}

test("hold: a reply's audio waits until its first frame shows, then plays and plays out once the reply ended", () => {
  const { posted, mediaGate } = gate("hold");
  mediaGate.audio(new Float32Array(4), "resp_1");
  mediaGate.audio(new Float32Array(4), "resp_1");
  mediaGate.end("resp_1", "completed");
  assert.deepEqual(posted, []);
  mediaGate.shown("resp_1");
  mediaGate.audio(new Float32Array(4), "resp_1");
  assert.deepEqual(posted, [["push", "resp_1"], ["push", "resp_1"], ["flush", "resp_1"], ["push", "resp_1"]]);
});

test("free: a reply's audio plays on arrival whatever the picture does", () => {
  const { posted, mediaGate } = gate("free");
  mediaGate.audio(new Float32Array(4), "resp_1");
  mediaGate.end("resp_1", "completed");
  assert.deepEqual(posted, [["push", "resp_1"], ["flush", "resp_1"]]);
});

test("hold: a cancelled reply drops its held audio, and one whose frame never shows plays after the default wait", () => {
  const { clock, posted, mediaGate } = gate("hold");
  mediaGate.audio(new Float32Array(4), "resp_1");
  mediaGate.end("resp_1", "cancelled");
  mediaGate.shown("resp_1");
  assert.deepEqual(posted, [["cancel", "resp_1"]]);
  mediaGate.audio(new Float32Array(4), "resp_2");
  clock.ms = DEFAULT_REPLY_MEDIA_WAIT_MS - 1;
  mediaGate.tick();
  assert.deepEqual(posted, [["cancel", "resp_1"]]);
  clock.ms = DEFAULT_REPLY_MEDIA_WAIT_MS;
  mediaGate.tick();
  assert.deepEqual(posted, [["cancel", "resp_1"], ["push", "resp_2"]]);
});

test("hold gates no audio without a drawn port that follows replies, nor audio outside a reply", () => {
  const unfollowed = { ...FOLLOWING, params: { fps: 30 } };
  for (const [ports, draws, responseId] of [[[unfollowed], () => true, "resp_1"], [[FOLLOWING], () => false, "resp_1"], [[FOLLOWING], () => true, null]]) {
    const { posted, mediaGate } = gate("hold", ports, draws);
    mediaGate.audio(new Float32Array(4), responseId);
    assert.deepEqual(posted, [["push", responseId]]);
  }
});

test("hold: a port's reply_media_wait_ms sets how long a reply waits, the longest one when several ports gate", () => {
  const port = (format, wait) => ({ ...FOLLOWING, format, params: { ...FOLLOWING.params, reply_media_wait_ms: wait } });
  for (const [ports, waitMs] of [[[port("test_motion", 1000)], 1000], [[port("test_motion", 1000), port("test_video", 800)], 1000], [[port("test_motion", 1000), FOLLOWING], DEFAULT_REPLY_MEDIA_WAIT_MS]]) {
    const { clock, posted, mediaGate } = gate("hold", ports);
    mediaGate.audio(new Float32Array(4), "resp_1");
    clock.ms = waitMs - 1;
    mediaGate.tick();
    assert.deepEqual(posted, [], String(waitMs));
    clock.ms = waitMs;
    mediaGate.tick();
    assert.deepEqual(posted, [["push", "resp_1"]], String(waitMs));
  }
});

test("hold: a reply waits its port's wait plus the renderers' playout delay, 100 ms unless the page names another", () => {
  const port = { ...FOLLOWING, params: { ...FOLLOWING.params, reply_media_wait_ms: 1000 } };
  assert.equal(DEFAULT_PLAYOUT_DELAY_MS, 100);
  for (const [options, waitMs] of [[{}, 1100], [{ playoutDelayMs: 250 }, 1250], [{ playoutDelayMs: 0 }, 1000]]) {
    const { clock, posted, mediaGate } = gate("hold", [port], () => true, options);
    mediaGate.audio(new Float32Array(4), "resp_1");
    clock.ms = waitMs - 1;
    mediaGate.tick();
    assert.deepEqual(posted, [], String(waitMs));
    clock.ms = waitMs;
    mediaGate.tick();
    assert.deepEqual(posted, [["push", "resp_1"]], String(waitMs));
  }
});

test("a page's ?playout_delay_ms names the playout delay, the default when absent or not a number of ms at least 0", () => {
  for (const [search, delayMs] of [["", 100], ["?playout_delay_ms=0", 0], ["?playout_delay_ms=180", 180], ["?playout_delay_ms=", 100], ["?playout_delay_ms=-1", 100], ["?playout_delay_ms=x", 100]]) {
    assert.equal(playoutDelayFrom(new URLSearchParams(search)), delayMs, search);
  }
});

// A turnless reply stream at 1 kHz, so a delta's sample count is its length in ms.
function aligning(policy, ports, options = {}) {
  const clock = { ms: 0 };
  const posted = [];
  const mediaGate = new ReplyMediaGate({ policy, now: () => clock.ms, post: (message) => posted.push([message.type, message.responseId, clock.ms]), ...options });
  mediaGate.follow(ports, () => true, { turnless: true, sampleRate: 1000 });
  return { clock, posted, mediaGate };
}

const LOOKING = { ...FOLLOWING, params: { ...FOLLOWING.params, reply_media_wait_ms: 1000, reply_lookahead_ms: 800 } };

test("hold aligns a turnless reply stream: its audio plays at its place on one media clock, the port's lookahead plus the playout delay behind its arrival", () => {
  const { clock, posted, mediaGate } = aligning("hold", [LOOKING]);
  assert.equal(mediaGate.isAligning, true);
  assert.equal(mediaGate.isGating, false);
  assert.equal(mediaGate.alignDelayMs, 800 + DEFAULT_PLAYOUT_DELAY_MS);
  clock.ms = 50;
  mediaGate.audio(new Float32Array(160), "resp_s");
  assert.equal(mediaGate.originMs, 950);
  clock.ms = 130;
  mediaGate.audio(new Float32Array(80), "resp_s");
  clock.ms = 949;
  mediaGate.tick();
  assert.deepEqual(posted, []);
  clock.ms = 950;
  mediaGate.tick();
  assert.deepEqual(posted, [["push", "resp_s", 950]]);
  // The second delta's place is the 160 ms received before it, however early or late it arrived.
  clock.ms = 1109;
  mediaGate.tick();
  assert.equal(posted.length, 1);
  clock.ms = 1110;
  mediaGate.tick();
  assert.deepEqual(posted.at(-1), ["push", "resp_s", 1110]);
  // A delta that arrives after its place plays at once.
  clock.ms = 1500;
  mediaGate.audio(new Float32Array(80), "resp_s");
  assert.deepEqual(posted.at(-1), ["push", "resp_s", 1500]);
});

test("the align delay is the lookahead at most the wait, plus the playout delay; a port that names no lookahead aligns at its wait", () => {
  const port = (format, params) => ({ ...FOLLOWING, format, params: { ...FOLLOWING.params, ...params } });
  for (const [ports, options, delayMs] of [
    [[port("m", { reply_media_wait_ms: 1000, reply_lookahead_ms: 405 })], {}, 505],
    [[port("m", { reply_media_wait_ms: 1000, reply_lookahead_ms: 2000 })], {}, 1100],
    [[port("m", { reply_media_wait_ms: 1000 })], {}, 1100],
    [[port("m", {})], {}, DEFAULT_REPLY_MEDIA_WAIT_MS + 100],
    [[port("m", { reply_media_wait_ms: 1000, reply_lookahead_ms: 800 }), port("v", { reply_media_wait_ms: 1000, reply_lookahead_ms: 405 })], {}, 900],
    [[port("m", { reply_media_wait_ms: 1000, reply_lookahead_ms: 800 })], { playoutDelayMs: 250 }, 1050],
  ]) {
    assert.equal(aligning("hold", ports, options).mediaGate.alignDelayMs, delayMs, JSON.stringify(ports));
  }
});

test("free plays a turnless reply stream on arrival, and a turn-based one still gates by reply", () => {
  const free = aligning("free", [LOOKING]);
  assert.equal(free.mediaGate.isAligning, false);
  free.mediaGate.audio(new Float32Array(80), "resp_s");
  assert.deepEqual(free.posted, [["push", "resp_s", 0]]);
  assert.equal(free.mediaGate.mediaOrigin(123), 123);
  const { mediaGate } = gate("hold", [LOOKING]);
  assert.equal(mediaGate.isGating, true);
  assert.equal(mediaGate.isAligning, false);
  assert.equal(mediaGate.mediaOrigin(123), 123);
});

test("aligning, the renderers' clock waits for the first reply audio and then is its clock; with no audio within the wait it is theirs", () => {
  const { clock, mediaGate } = aligning("hold", [LOOKING]);
  assert.equal(mediaGate.mediaOrigin(200), null);
  clock.ms = 300;
  assert.equal(mediaGate.mediaOrigin(200), null);
  mediaGate.audio(new Float32Array(160), "resp_s");
  assert.equal(mediaGate.mediaOrigin(200), 300 + 900);
  const silent = aligning("hold", [LOOKING]);
  assert.equal(silent.mediaGate.mediaOrigin(200), null);
  silent.clock.ms = 1099;
  assert.equal(silent.mediaGate.mediaOrigin(200), null);
  silent.clock.ms = 1100;
  assert.equal(silent.mediaGate.mediaOrigin(200), 200);
});

test("aligning, muted audio keeps its place, a completed reply flushes after its last audio and a cancelled one drops what is not due", () => {
  const { clock, posted, mediaGate } = aligning("hold", [LOOKING]);
  mediaGate.skip(160);
  assert.equal(mediaGate.originMs, 900);
  mediaGate.audio(new Float32Array(80), "resp_s");
  mediaGate.end("resp_s", "completed");
  clock.ms = 1059;
  mediaGate.tick();
  assert.deepEqual(posted, []);
  clock.ms = 1060;
  mediaGate.tick();
  assert.deepEqual(posted, [["push", "resp_s", 1060], ["flush", "resp_s", 1060]]);
  mediaGate.audio(new Float32Array(80), "resp_t");
  mediaGate.end("resp_t", "cancelled");
  clock.ms = 5000;
  mediaGate.tick();
  assert.deepEqual(posted.slice(2), [["cancel", "resp_t", 1060]]);
  mediaGate.end("resp_u", "completed");
  assert.deepEqual(posted.at(-1), ["flush", "resp_u", 5000]);
});
