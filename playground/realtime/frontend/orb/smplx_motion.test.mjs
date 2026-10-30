import test from "node:test";
import assert from "node:assert/strict";

import { bytesToBase64 } from "./session.js";
import {
  FEATURES_PER_FRAME,
  REST_OFFSETS_M,
  SMPLX_PARENTS,
  SkeletonView,
  jointPositions,
} from "./smplx_motion.js";

const LEFT_ELBOW = 18;
const LEFT_WRIST = 20;

function payload(frames) {
  const values = new Float32Array(frames.length * FEATURES_PER_FRAME);
  frames.forEach((frame, index) => values.set(frame, index * FEATURES_PER_FRAME));
  return bytesToBase64(new Uint8Array(values.buffer));
}

function restFrame(root = [0, 1.3, 0]) {
  const frame = new Float32Array(FEATURES_PER_FRAME);
  frame.set(root, 265);
  return frame;
}

test("every joint but the root has a parent earlier in SMPL-X order", () => {
  assert.equal(SMPLX_PARENTS.length, 55);
  assert.equal(REST_OFFSETS_M.length, 55);
  SMPLX_PARENTS.forEach((parent, joint) => assert.ok(joint === 0 ? parent === -1 : parent >= 0 && parent < joint));
});

test("a bent elbow swings the wrist about the elbow and leaves the shoulder chain alone", () => {
  const rest = jointPositions(restFrame());
  const bent = restFrame();
  bent.set([0, 0, Math.PI / 2], 3 * LEFT_ELBOW);
  const posed = jointPositions(bent);
  const forearm = REST_OFFSETS_M[LEFT_WRIST];
  assert.deepEqual(posed[LEFT_ELBOW], rest[LEFT_ELBOW]);
  const swung = [posed[LEFT_WRIST][0] - posed[LEFT_ELBOW][0], posed[LEFT_WRIST][1] - posed[LEFT_ELBOW][1]];
  assert.ok(Math.abs(swung[0] + forearm[1]) < 1e-6 && Math.abs(swung[1] - forearm[0]) < 1e-6);
});

test("frames play on the chunks' media clock from the first chunk's arrival", () => {
  let nowMs = 1000;
  const view = new SkeletonView(null, { now: () => nowMs });
  view.push({ format: "smplx_motion", t_start_ms: 0, duration_ms: 80, payload_b64: payload([restFrame([0, 1, 0]), restFrame([0, 2, 0])]) });
  view.push({ format: "smplx_motion", t_start_ms: 80, duration_ms: 80, payload_b64: payload([restFrame([0, 3, 0]), restFrame([0, 4, 0])]) });
  assert.equal(view.due()[266], 1);
  nowMs = 1085;
  assert.equal(view.due()[266], 3);
  assert.equal(view.due(), null);
  assert.equal(view.queue.length, 1);
});
