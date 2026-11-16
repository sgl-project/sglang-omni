import test from "node:test";
import assert from "node:assert/strict";

import { bytesToBase64 } from "./session.js";
import { DEFAULT_PLAYOUT_DELAY_MS, FEATURES_PER_FRAME, MotionView, SMPLH_PARENTS, jointPositions } from "./smplh_flame_motion.js";

function chunk(tStartMs, frames, responseId = null, fill = () => 0) {
  const values = new Float32Array(frames * FEATURES_PER_FRAME);
  values.forEach((_, index) => { values[index] = fill(index % FEATURES_PER_FRAME); });
  return { t_start_ms: tStartMs, response_id: responseId, payload_b64: bytesToBase64(new Uint8Array(values.buffer)) };
}

test("at rest the hands hang below the shoulders and every joint has a parent in the tree", () => {
  const positions = jointPositions(new Float32Array(FEATURES_PER_FRAME));
  assert.equal(SMPLH_PARENTS.length, 52);
  assert.ok(SMPLH_PARENTS.every((parent, joint) => parent < joint));
  assert.ok(positions[22][0] > positions[16][0] && positions[37][0] < positions[17][0]);
});

test("a bent elbow moves the wrist and the hand after it, not the shoulder", () => {
  const rest = jointPositions(new Float32Array(FEATURES_PER_FRAME));
  const frame = new Float32Array(FEATURES_PER_FRAME);
  frame[3 * 18 + 2] = 1.2;
  const bent = jointPositions(frame);
  assert.deepEqual(bent[16], rest[16]);
  assert.notDeepEqual(bent[20], rest[20]);
  assert.notDeepEqual(bent[24], rest[24]);
});

test("frames play on the media clock from the first chunk's arrival, and a reply is shown once its first frame is due", () => {
  let now = 1000;
  const shown = [];
  const view = new MotionView(null, { now: () => now, onReplyShown: (id) => shown.push(id), playoutDelayMs: 0 });
  view.push(chunk(0, 24));
  now = 1300;
  view.push(chunk(800, 24, "resp_1", (index) => (index === 0 ? 0.5 : 0)));
  assert.equal(view.due()[0], 0);
  now = 1000 + 799;
  view.due();
  assert.deepEqual(shown, []);
  now = 1000 + 800;
  assert.equal(view.due()[0], 0.5);
  assert.deepEqual(shown, ["resp_1"]);
});

test("an interrupted reply's motion freezes on its last shown pose: its frames not yet due and later chunks of it are dropped", () => {
  let now = 1000;
  const view = new MotionView(null, { now: () => now, playoutDelayMs: 0 });
  view.push(chunk(0, 24, "resp_1", () => 0.25));
  now = 1000 + 400;
  assert.equal(view.due()[0], 0.25);
  view.drop("resp_1");
  view.push(chunk(800, 24, "resp_1", () => 0.5));
  view.push(chunk(1600, 24, null, () => 0.75));
  now = 1000 + 1599;
  assert.equal(view.due(), null);
  now = 1000 + 1600;
  assert.equal(view.due()[0], 0.75);
});

test("by default motion plays 100 ms after the first chunk's arrival, so a chunk up to 100 ms behind its pace plays every frame on time", () => {
  let now = 1000;
  const shown = [];
  const view = new MotionView(null, { now: () => now, onReplyShown: (id) => shown.push(id) });
  assert.equal(DEFAULT_PLAYOUT_DELAY_MS, 100);
  view.push(chunk(0, 24, null, () => 0.25));
  assert.equal(view.due(), null);
  now = 1100;
  assert.equal(view.due()[0], 0.25);
  // The next chunk is due at 1900 and arrives at 1890, 90 ms behind the first chunk's pace: its first frame is not skipped.
  now = 1890;
  view.due();
  view.push(chunk(800, 24, "resp_1", (index) => (index === 0 ? 0.5 : 0)));
  assert.deepEqual(shown, []);
  now = 1900;
  assert.equal(view.due()[0], 0.5);
  assert.deepEqual(shown, ["resp_1"]);
});

test("given a clock, frames play on its origin, and none plays while it holds the origin back", () => {
  let now = 1000;
  let origin = null;
  const proposed = [];
  const view = new MotionView(null, { now: () => now, playoutDelayMs: DEFAULT_PLAYOUT_DELAY_MS, clock: { mediaOrigin: (ms) => { proposed.push(ms); return origin; } } });
  view.push(chunk(0, 24, null, (index) => (index === 0 ? 0.5 : 0)));
  now = 5000;
  assert.equal(view.due(), null);
  assert.deepEqual(proposed, [1000 + DEFAULT_PLAYOUT_DELAY_MS]);
  origin = 5100;
  assert.equal(view.due(), null);
  now = 5100;
  assert.equal(view.due()[0], 0.5);
});
