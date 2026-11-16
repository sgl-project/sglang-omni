import assert from "node:assert/strict";
import test from "node:test";

import { DEFAULT_REPLY_MEDIA_WAIT_MS, ReplyMediaGate } from "./reply_media_gate.js";

const FOLLOWING = { modality: "motion", format: "test_motion", params: { fps: 30, timeline: "input", follows_replies: true } };

function gate(policy, ports = [FOLLOWING], draws = () => true) {
  const clock = { ms: 0 };
  const posted = [];
  const mediaGate = new ReplyMediaGate({ policy, now: () => clock.ms, post: (message) => posted.push([message.type, message.responseId]) });
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
