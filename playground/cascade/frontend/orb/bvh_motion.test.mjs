import test from "node:test";
import assert from "node:assert/strict";

import { bytesToBase64 } from "./session.js";
import { BvhMotionView, bvhJointPositions, bvhSkeletonOf } from "./bvh_motion.js";

const SKELETON = { fps: 30, features: 3 + 4 * 3, parents: [-1, 0, 1], offsets: [[0, 0, 0], [0, 0, 20], [0, 0, 30]], up: "z" };

function row(root = [0, 0, 100], quaternions = [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]) {
  return [...root, ...quaternions.flat()];
}

function chunk(tStartMs, rows, responseId = null) {
  const values = new Float32Array(rows.flat());
  return { t_start_ms: tStartMs, response_id: responseId, payload_b64: bytesToBase64(new Uint8Array(values.buffer)) };
}

test("at rest each joint sits at its parent plus its offset, the root at its position", () => {
  const positions = bvhJointPositions(row(), SKELETON);
  assert.deepEqual(positions, [[0, 0, 100], [0, 0, 120], [0, 0, 150]]);
});

test("a joint's rotation turns its children's offsets, not its own", () => {
  const half = Math.SQRT1_2;
  const positions = bvhJointPositions(row([0, 0, 100], [[1, 0, 0, 0], [half, half, 0, 0], [1, 0, 0, 0]]), SKELETON);
  assert.deepEqual(positions[1], [0, 0, 120]);
  assert.ok(Math.abs(positions[2][1] + 30) < 1e-6 && Math.abs(positions[2][2] - 120) < 1e-6);
});

test("rows play on the media clock at the port's rate, a held empty row keeps the last pose, and a reply shows once its first row is due", () => {
  let now = 1000;
  const shown = [];
  const view = new BvhMotionView(null, { now: () => now, playoutDelayMs: 0, onReplyShown: (id) => shown.push(id), skeleton: () => SKELETON });
  view.push(chunk(0, [row([1, 0, 100])]));
  view.push({ t_start_ms: 33.3, response_id: null, payload_b64: "" });
  view.push(chunk(66.7, [row([2, 0, 100])], "resp_1"));
  assert.equal(view.due()[0], 1);
  now = 1040;
  assert.equal(view.due(), null);
  assert.deepEqual(shown, []);
  now = 1070;
  assert.equal(view.due()[0], 2);
  assert.deepEqual(shown, ["resp_1"]);
});

test("rows wait for the session's granted skeleton, and the skeleton is the bvh_motion port's params", () => {
  const view = new BvhMotionView(null, { now: () => 0, playoutDelayMs: 0, skeleton: () => null });
  view.push(chunk(0, [row()]));
  assert.equal(view.queue.length, 0);
  assert.equal(bvhSkeletonOf([{ format: "jpeg_frame", params: {} }, { format: "bvh_motion", params: SKELETON }]), SKELETON);
  assert.equal(bvhSkeletonOf([]), null);
});
