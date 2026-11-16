// The smplx_motion output format (MIBURI's layout, also StreamTalk's) drawn on the SMPL-X mesh when the page has the user's SMPL-X model
// (smplx_mesh.js), otherwise as a stick figure: the SMPL-X joint tree with hand-set rest offsets of plausible adult proportions, not the SMPL-X
// model's own joints, and a bar for the jaw.
import { base64ToBytes } from "./session.js";
import { MotionView, axisAngleToMatrix } from "./smplh_flame_motion.js";
import { smplxFramePose } from "./smplx_mesh.js";

export const SMPLX_MOTION = "smplx_motion";
export const FEATURES_PER_FRAME = 271;
// The rate a chunk without a duration plays at; a chunk's own duration_ms over its frames wins.
export const DEFAULT_MOTION_FPS = 30;
const NUM_JOINTS = 55;
// SMPL-X joint 22 is the jaw; its rotation about x opens the mouth.
const JAW = 3 * 22;
const JAW_OPEN_RAD = 0.3;

// Parent of each joint in SMPL-X joint order: body 0-21, jaw and eyes 22-24, left then right hand (index, middle, pinky, ring, thumb; three each).
export const SMPLX_PARENTS = [
  -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 15, 15, 15,
  20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 20, 37, 38,
  21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53,
];

// Offset of each joint from its parent in meters at rest: x to the body's left, y up, z forward.
const LEFT_HAND = [
  [0.09, -0.01, 0.025], [0.035, 0, 0], [0.025, 0, 0],
  [0.095, -0.01, 0.005], [0.035, 0, 0], [0.027, 0, 0],
  [0.08, -0.015, -0.035], [0.022, 0, 0], [0.018, 0, 0],
  [0.088, -0.012, -0.015], [0.03, 0, 0], [0.024, 0, 0],
  [0.025, -0.015, 0.03], [0.03, -0.005, 0.02], [0.025, 0, 0.015],
];
export const REST_OFFSETS_M = [
  [0, 0, 0], [0.06, -0.09, 0], [-0.06, -0.09, 0], [0, 0.11, -0.02], [0.04, -0.38, 0], [-0.04, -0.38, 0],
  [0, 0.13, 0], [-0.01, -0.4, -0.04], [0.01, -0.4, -0.04], [0, 0.05, 0.02], [0.02, -0.06, 0.12], [-0.02, -0.06, 0.12],
  [0, 0.21, -0.03], [0.07, 0.11, -0.01], [-0.07, 0.11, -0.01], [0, 0.09, 0.05], [0.11, 0.03, -0.02], [-0.11, 0.03, -0.02],
  [0.26, -0.01, -0.02], [-0.26, -0.01, -0.02], [0.25, 0.01, 0], [-0.25, 0.01, 0],
  [0, -0.01, 0.05], [0.03, 0.06, 0.08], [-0.03, 0.06, 0.08],
  ...LEFT_HAND,
  ...LEFT_HAND.map(([x, y, z]) => [-x, y, z]),
];

export function decodeFrames(payloadBase64) {
  const bytes = base64ToBytes(payloadBase64);
  const values = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  const frames = [];
  for (let start = 0; start + FEATURES_PER_FRAME <= values.length; start += FEATURES_PER_FRAME) {
    frames.push(values.subarray(start, start + FEATURES_PER_FRAME));
  }
  return frames;
}

function multiply(a, b) {
  const out = new Array(9);
  for (let row = 0; row < 3; row += 1) {
    for (let column = 0; column < 3; column += 1) {
      out[row * 3 + column] = a[row * 3] * b[column] + a[row * 3 + 1] * b[3 + column] + a[row * 3 + 2] * b[6 + column];
    }
  }
  return out;
}

// Position of every joint of one frame, the root at the origin, each joint rotated by its parents' poses.
export function jointPositions(frame) {
  const rotations = [];
  const positions = [];
  for (let joint = 0; joint < NUM_JOINTS; joint += 1) {
    const local = axisAngleToMatrix(frame[3 * joint], frame[3 * joint + 1], frame[3 * joint + 2]);
    const parent = SMPLX_PARENTS[joint];
    if (parent < 0) {
      rotations.push(local);
      positions.push([0, 0, 0]);
    } else {
      const rotation = rotations[parent];
      const [x, y, z] = REST_OFFSETS_M[joint];
      const base = positions[parent];
      positions.push([
        base[0] + rotation[0] * x + rotation[1] * y + rotation[2] * z,
        base[1] + rotation[3] * x + rotation[4] * y + rotation[5] * z,
        base[2] + rotation[6] * x + rotation[7] * y + rotation[8] * z,
      ]);
      rotations.push(multiply(rotation, local));
    }
  }
  return positions;
}

// MotionView's media clock, playout delay, reply gate and interrupt drop, over SMPL-X frames; a chunk's frames span its duration_ms.
export class SmplxMotionView extends MotionView {
  push(event) {
    if (event.response_id && this.dropped.has(event.response_id)) return;
    if (this.proposedOriginMs === null) this.proposedOriginMs = this.now() + this.playoutDelayMs - event.t_start_ms;
    const frames = decodeFrames(event.payload_b64);
    const frameMs = event.duration_ms > 0 && frames.length ? event.duration_ms / frames.length : 1000 / DEFAULT_MOTION_FPS;
    frames.forEach((frame, index) => {
      this.queue.push({ mediaMs: event.t_start_ms + index * frameMs, frame, responseId: event.response_id });
    });
  }

  posed(frame) {
    return smplxFramePose(frame);
  }

  draw() {
    const frame = this.due();
    if (frame === null) return;
    if (this.mesh) {
      const { pose, expression } = this.posed(frame);
      this.mesh.show(pose, expression);
      return;
    }
    const positions = jointPositions(frame);
    this.canvas.hidden = false;
    const context = this.canvas.getContext("2d");
    const { width, height } = this.canvas;
    const scale = height / 1.6;
    context.clearRect(0, 0, width, height);
    context.strokeStyle = "#3f8f86";
    context.lineWidth = 3;
    context.beginPath();
    for (let joint = 1; joint < NUM_JOINTS; joint += 1) {
      const [x, y] = positions[joint];
      const [parentX, parentY] = positions[SMPLX_PARENTS[joint]];
      context.moveTo(width / 2 - parentX * scale, height * 0.7 - parentY * scale);
      context.lineTo(width / 2 - x * scale, height * 0.7 - y * scale);
    }
    context.stroke();
    const jawOpen = Math.min(Math.max(frame[JAW] / JAW_OPEN_RAD, 0), 1);
    context.fillStyle = "#3f8f86";
    context.fillRect(width - 14, height * (1 - jawOpen) - 4, 8, height * jawOpen);
  }
}

// The renderer the page registers: registerPluginRenderer(SMPLX_MOTION, smplxMotionRenderer(view)).
export function smplxMotionRenderer(view) {
  return (event) => view.push(event);
}
