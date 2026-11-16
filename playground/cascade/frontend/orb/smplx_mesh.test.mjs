import test from "node:test";
import assert from "node:assert/strict";

import { bytesToBase64 } from "./session.js";
import { MotionView } from "./smplh_flame_motion.js";
import { SMPLX_PARENTS, SmplxMotionView } from "./smplx_motion.js";
import { NUM_JOINTS, attachSmplxMesh, parseMeshAsset, skinVertices, smplhFlameFramePose, smplxFramePose } from "./smplx_mesh.js";

// The blob playground/cascade/app.py serves: a uint32 header length, the JSON header padded to 4 bytes, then each array 4-byte aligned.
function pack(arrays) {
  const header = {};
  let offset = 0;
  for (const [name, { dtype, shape, data }] of Object.entries(arrays)) {
    header[name] = { dtype, shape, offset };
    offset += Math.ceil(data.byteLength / 4) * 4;
  }
  let text = JSON.stringify(header);
  text += " ".repeat((4 - ((text.length + 4) % 4)) % 4);
  const bytes = new Uint8Array(4 + text.length + offset);
  new DataView(bytes.buffer).setUint32(0, text.length, true);
  bytes.set(new TextEncoder().encode(text), 4);
  for (const [name, { data }] of Object.entries(arrays)) {
    bytes.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), 4 + text.length + header[name].offset);
  }
  return bytes.buffer;
}

// One joint per SMPL-X joint along +x, one vertex 1 cm above each joint skinned to it alone, and one expression direction moving vertex 0 up.
function syntheticAsset() {
  const joints = new Float32Array(NUM_JOINTS * 3);
  const vertices = new Float32Array(NUM_JOINTS * 3);
  for (let joint = 0; joint < NUM_JOINTS; joint += 1) {
    joints.set([0.1 * joint, 0.5, 0], 3 * joint);
    vertices.set([0.1 * joint, 0.51, 0], 3 * joint);
  }
  const skinIndices = new Uint8Array(NUM_JOINTS * 4);
  const skinWeights = new Float32Array(NUM_JOINTS * 4);
  for (let joint = 0; joint < NUM_JOINTS; joint += 1) {
    skinIndices[4 * joint] = joint;
    skinWeights[4 * joint] = 1;
  }
  // float16 0.5 is 0x3800; the first expression direction moves vertex 0 along y.
  const expressionDirs = new Uint16Array(3 * 2);
  expressionDirs[1 * 2] = 0x3800;
  return pack({
    vertices: { dtype: "f4", shape: [NUM_JOINTS, 3], data: vertices },
    faces: { dtype: "u4", shape: [1, 3], data: new Uint32Array([0, 1, 2]) },
    joints: { dtype: "f4", shape: [NUM_JOINTS, 3], data: joints },
    parents: { dtype: "i4", shape: [NUM_JOINTS], data: Int32Array.from(SMPLX_PARENTS) },
    skin_indices: { dtype: "u1", shape: [NUM_JOINTS, 4], data: skinIndices },
    skin_weights: { dtype: "f4", shape: [NUM_JOINTS, 4], data: skinWeights },
    expression_vertices: { dtype: "u4", shape: [1], data: new Uint32Array([0]) },
    expression_dirs: { dtype: "f2", shape: [1, 3, 2], data: expressionDirs },
  });
}

function close(actual, expected, message) {
  assert.ok(Math.abs(actual - expected) < 1e-5, `${message}: ${actual} vs ${expected}`);
}

test("the server's blob decodes into typed arrays, float16 expression directions widened to float32", () => {
  const asset = parseMeshAsset(syntheticAsset());
  assert.equal(asset.vertices.length, NUM_JOINTS * 3);
  assert.ok(asset.parents instanceof Int32Array && asset.parents[0] === -1);
  assert.ok(asset.skinIndices instanceof Uint8Array);
  assert.equal(asset.influences, 4);
  assert.equal(asset.numExpressions, 2);
  assert.deepEqual([...asset.expressionDirs], [0, 0, 0.5, 0, 0, 0]);
});

test("at rest the mesh is the template with the root joint at the origin, and expressions move only their vertices", () => {
  const asset = parseMeshAsset(syntheticAsset());
  const rest = skinVertices(asset, new Float32Array(165), null);
  for (let index = 0; index < rest.length; index += 3) {
    close(rest[index], asset.vertices[index], "x");
    close(rest[index + 1], asset.vertices[index + 1] - 0.5, "y");
  }
  const smiling = skinVertices(asset, new Float32Array(165), Float32Array.from([0.2, 0]));
  close(smiling[1], rest[1] + 0.1, "vertex 0 moves by 0.5 x 0.2");
  close(smiling[4], rest[4], "vertex 1 stays");
});

test("a joint's rotation turns its own and its descendants' vertices about its rest place and leaves the rest of the tree", () => {
  const asset = parseMeshAsset(syntheticAsset());
  const pose = new Float32Array(165);
  // The left shoulder (16) turns a quarter about z: the elbow (18), a child, swings about the shoulder.
  pose[3 * 16 + 2] = Math.PI / 2;
  const posed = skinVertices(asset, pose, null);
  const shoulder = [1.6, 0];
  const elbowRest = [1.8, 0.01];
  const turned = [shoulder[0] - (elbowRest[1] - shoulder[1]), shoulder[1] + (elbowRest[0] - shoulder[0])];
  close(posed[3 * 18], turned[0], "elbow x");
  close(posed[3 * 18 + 1], turned[1], "elbow y");
  // The shoulder's own vertex, 1 cm above it, turns to 1 cm left of it.
  close(posed[3 * 16], 1.6 - 0.01, "shoulder vertex x");
  // The right shoulder (17), on another branch, stays.
  close(posed[3 * 17], 1.7, "right shoulder x");
  close(posed[3 * 17 + 1], 0.01, "right shoulder y");
});

test("an smplh_flame frame maps SMPL-H's body and hands onto SMPL-X's joints and FLAME's expression, jaw and eyes onto the face", () => {
  const frame = new Float32Array(264).map((_, index) => index + 1);
  const { pose, expression } = smplhFlameFramePose(frame);
  assert.equal(pose.length, 165);
  assert.deepEqual([...expression], [...frame.subarray(156, 256)]);
  assert.deepEqual([...pose.subarray(0, 66)], [...frame.subarray(0, 66)]);
  // jaw about x; left eye x, y; right eye x, y (SMPL-X joints 22-24); FLAME's head rotation (256-258) is not used.
  assert.deepEqual([...pose.subarray(66, 75)], [frame[259], 0, 0, frame[260], frame[261], 0, frame[262], frame[263], 0]);
  assert.deepEqual([...pose.subarray(75, 165)], [...frame.subarray(66, 156)]);
  const smplx = smplxFramePose(new Float32Array(271).map((_, index) => index));
  assert.equal(smplx.pose.length, 165);
  assert.equal(smplx.expression[0], 165);
  assert.equal(smplx.expression.length, 100);
});

test("without the server's model the views keep their stick figures; with it both draw each due frame on the mesh", async () => {
  const views = [new MotionView(null), new SmplxMotionView(null)];
  assert.equal(await attachSmplxMesh({ canvas: null, views, fetchAsset: async () => ({ ok: false }) }), null);
  assert.ok(views.every((view) => view.mesh === null));
  const shown = [];
  const renderer = { show: (pose, expression) => shown.push({ pose: [...pose], expression: expression && [...expression] }) };
  const attached = await attachSmplxMesh({
    canvas: "canvas",
    views,
    fetchAsset: async () => ({ ok: true, arrayBuffer: async () => syntheticAsset() }),
    loadRenderer: async (canvas, asset) => {
      assert.equal(canvas, "canvas");
      assert.equal(asset.vertices.length, NUM_JOINTS * 3);
      return renderer;
    },
  });
  assert.equal(attached, renderer);
  assert.ok(views.every((view) => view.mesh === renderer));

  let now = 0;
  const smplx = new SmplxMotionView(null, { now: () => now, playoutDelayMs: 0 });
  smplx.mesh = renderer;
  const values = new Float32Array(2 * 271).fill(0.25);
  smplx.push({ t_start_ms: 0, duration_ms: 66, payload_b64: bytesToBase64(new Uint8Array(values.buffer)) });
  smplx.draw();
  assert.equal(shown.length, 1);
  assert.equal(shown[0].pose.length, 165);
  assert.equal(shown[0].expression.length, 100);

  const smplh = new MotionView(null, { now: () => now, playoutDelayMs: 0 });
  smplh.mesh = renderer;
  smplh.push({ t_start_ms: 0, payload_b64: bytesToBase64(new Uint8Array(new Float32Array(264).fill(0.5).buffer)) });
  smplh.draw();
  assert.equal(shown.length, 2);
  assert.equal(shown[1].pose[66], 0.5);
  // the left eye's x rotation, and no z
  assert.equal(shown[1].pose[69], 0.5);
  assert.equal(shown[1].pose[71], 0);
  assert.equal(shown[1].expression.length, 100);
});
