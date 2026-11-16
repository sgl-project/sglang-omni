import test from "node:test";
import assert from "node:assert/strict";

import { bytesToBase64 } from "./session.js";
import { FEATURES_PER_FRAME, SMPLX_PARENTS, SmplxMotionView, decodeFrames, jointPositions } from "./smplx_motion.js";

function chunk(tStartMs, frames, durationMs, responseId = null, fill = () => 0) {
  const values = new Float32Array(frames * FEATURES_PER_FRAME);
  values.forEach((_, index) => { values[index] = fill(index % FEATURES_PER_FRAME); });
  return { t_start_ms: tStartMs, duration_ms: durationMs, response_id: responseId, payload_b64: bytesToBase64(new Uint8Array(values.buffer)) };
}

test("the SMPL-X tree has a parent before every joint, and at rest the wrists reach out past the shoulders", () => {
  assert.equal(SMPLX_PARENTS.length, 55);
  assert.ok(SMPLX_PARENTS.every((parent, joint) => parent < joint));
  const positions = jointPositions(new Float32Array(FEATURES_PER_FRAME));
  assert.ok(positions[20][0] > positions[16][0] && positions[21][0] < positions[17][0]);
  assert.equal(decodeFrames(chunk(0, 3, 100).payload_b64).length, 3);
});

test("a chunk's frames span its duration, play from the first chunk's arrival plus the playout delay, and a reply is shown at its first frame", () => {
  let now = 1000;
  const shown = [];
  const view = new SmplxMotionView(null, { now: () => now, onReplyShown: (id) => shown.push(id), playoutDelayMs: 100 });
  view.push(chunk(0, 15, 500));
  view.push(chunk(500, 15, 500, "resp_1", (index) => (index === 66 ? 0.2 : 0)));
  now = 1100 + 499;
  assert.equal(view.due()[66], 0);
  assert.deepEqual(shown, []);
  now = 1100 + 500;
  assert.ok(Math.abs(view.due()[66] - 0.2) < 1e-6);
  assert.deepEqual(shown, ["resp_1"]);
});

test("an interrupted reply's motion freezes: its frames not yet due and its later chunks are dropped", () => {
  let now = 1000;
  const view = new SmplxMotionView(null, { now: () => now, playoutDelayMs: 0 });
  view.push(chunk(0, 15, 500, "resp_1", () => 0.25));
  now = 1000 + 100;
  assert.equal(view.due()[0], 0.25);
  view.drop("resp_1");
  view.push(chunk(500, 15, 500, "resp_1", () => 0.5));
  view.push(chunk(1000, 15, 500, null, () => 0.75));
  now = 1000 + 999;
  assert.equal(view.due(), null);
  now = 1000 + 1000;
  assert.equal(view.due()[0], 0.75);
});
