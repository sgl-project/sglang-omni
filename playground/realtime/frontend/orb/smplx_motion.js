// The smplx_motion output format drawn as a stick figure: the SMPL-X joint tree with hand-set rest offsets of plausible adult proportions, not the SMPL-X model's own joints; a mesh needs the user's SMPL-X model files.
import { base64ToBytes } from "./session.js";

export const SMPLX_MOTION = "smplx_motion";
export const FEATURES_PER_FRAME = 271;
export const MOTION_FPS = 25;
const NUM_JOINTS = 55;
const POSE_FEATURES = 3 * NUM_JOINTS;
const EXPRESSION_FEATURES = 100;
const ROOT_TRANSLATION = POSE_FEATURES + EXPRESSION_FEATURES;

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
  for (let start = 0; start < values.length; start += FEATURES_PER_FRAME) {
    frames.push(values.subarray(start, start + FEATURES_PER_FRAME));
  }
  return frames;
}

// Rodrigues: an axis-angle vector as a row-major 3x3 rotation.
export function axisAngleToMatrix(x, y, z) {
  const angle = Math.hypot(x, y, z);
  if (angle < 1e-8) return [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const [u, v, w] = [x / angle, y / angle, z / angle];
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const t = 1 - c;
  return [
    t * u * u + c, t * u * v - s * w, t * u * w + s * v,
    t * u * v + s * w, t * v * v + c, t * v * w - s * u,
    t * u * w - s * v, t * v * w + s * u, t * w * w + c,
  ];
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

// World position of every joint of one frame: root at its translation, each joint rotated by its parents' poses.
export function jointPositions(frame) {
  const rotations = [];
  const positions = [];
  for (let joint = 0; joint < NUM_JOINTS; joint += 1) {
    const local = axisAngleToMatrix(frame[3 * joint], frame[3 * joint + 1], frame[3 * joint + 2]);
    const parent = SMPLX_PARENTS[joint];
    if (parent < 0) {
      rotations.push(local);
      positions.push([frame[ROOT_TRANSLATION], frame[ROOT_TRANSLATION + 1], frame[ROOT_TRANSLATION + 2]]);
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

// Frames queue on the chunks' media clock and play from the first chunk's arrival; draw() paints the frame due now, seen from the front.
export class SkeletonView {
  constructor(canvas, { now = () => performance.now() } = {}) {
    this.canvas = canvas;
    this.now = now;
    this.queue = [];
    this.originMs = null;
  }

  push(event) {
    if (this.originMs === null) this.originMs = this.now() - event.t_start_ms;
    decodeFrames(event.payload_b64).forEach((frame, index) => {
      this.queue.push({ mediaMs: event.t_start_ms + (index * 1000) / MOTION_FPS, frame });
    });
  }

  due() {
    if (this.originMs === null) return null;
    const mediaMs = this.now() - this.originMs;
    let frame = null;
    while (this.queue.length && this.queue[0].mediaMs <= mediaMs) frame = this.queue.shift().frame;
    return frame;
  }

  draw() {
    const frame = this.due();
    if (frame === null) return;
    const positions = jointPositions(frame);
    this.canvas.hidden = false;
    const context = this.canvas.getContext("2d");
    const { width, height } = this.canvas;
    const scale = height / 2.2;
    const [rootX, rootY] = positions[0];
    context.clearRect(0, 0, width, height);
    context.strokeStyle = "#3f8f86";
    context.lineWidth = 3;
    context.beginPath();
    for (let joint = 1; joint < NUM_JOINTS; joint += 1) {
      const [x, y] = positions[joint];
      const [parentX, parentY] = positions[SMPLX_PARENTS[joint]];
      context.moveTo(width / 2 - (parentX - rootX) * scale, height * 0.55 - (parentY - rootY) * scale);
      context.lineTo(width / 2 - (x - rootX) * scale, height * 0.55 - (y - rootY) * scale);
    }
    context.stroke();
  }
}
