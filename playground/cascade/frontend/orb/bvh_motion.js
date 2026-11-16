// The bvh_motion output format drawn as a stick figure of the port's own skeleton: each row is the root position, one unit quaternion
// (w, x, y, z) per joint relative to its parent, then face channels; the port's params carry the joint names, parents, rest offsets and
// which axis is up, what a BVH HIERARCHY holds. Empty rows (held frames) keep the previous pose. Seen from the front: the horizontal axis
// is x, the vertical one the skeleton's up axis.
import { base64ToBytes } from "./session.js";
import { MotionView } from "./smplh_flame_motion.js";

export const BVH_MOTION = "bvh_motion";
const AXES = { x: 0, y: 1, z: 2 };

export function quaternionToMatrix(w, x, y, z) {
  const n = Math.hypot(w, x, y, z) || 1;
  [w, x, y, z] = [w / n, x / n, y / n, z / n];
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
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

// Position of every joint of one row: the root at its position, each joint at its parent plus the parent's rotation of its rest offset.
export function bvhJointPositions(row, skeleton) {
  const rotations = [];
  const positions = [];
  skeleton.parents.forEach((parent, joint) => {
    const q = 3 + 4 * joint;
    const local = quaternionToMatrix(row[q], row[q + 1], row[q + 2], row[q + 3]);
    if (parent < 0) {
      rotations.push(local);
      positions.push([row[0], row[1], row[2]]);
    } else {
      const rotation = rotations[parent];
      const [x, y, z] = skeleton.offsets[joint];
      const base = positions[parent];
      positions.push([
        base[0] + rotation[0] * x + rotation[1] * y + rotation[2] * z,
        base[1] + rotation[3] * x + rotation[4] * y + rotation[5] * z,
        base[2] + rotation[6] * x + rotation[7] * y + rotation[8] * z,
      ]);
      rotations.push(multiply(rotation, local));
    }
  });
  return positions;
}

// The MotionView's media clock and reply gate over bvh_motion chunks; skeleton() returns the port's params once the session granted them.
export class BvhMotionView extends MotionView {
  constructor(canvas, { skeleton = () => null, ...options } = {}) {
    super(canvas, options);
    this.skeleton = skeleton;
  }

  push(event) {
    if (event.response_id && this.dropped.has(event.response_id)) return;
    const skeleton = this.skeleton();
    if (!skeleton) return;
    if (this.proposedOriginMs === null) this.proposedOriginMs = this.now() + this.playoutDelayMs - event.t_start_ms;
    const bytes = event.payload_b64 ? base64ToBytes(event.payload_b64) : new Uint8Array(0);
    const values = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
    const frameMs = 1000 / skeleton.fps;
    const rows = Math.floor(values.length / skeleton.features);
    if (rows === 0) {
      this.queue.push({ mediaMs: event.t_start_ms, frame: null, responseId: event.response_id || null });
      return;
    }
    for (let index = 0; index < rows; index += 1) {
      const frame = values.subarray(index * skeleton.features, (index + 1) * skeleton.features);
      this.queue.push({ mediaMs: event.t_start_ms + index * frameMs, frame, responseId: event.response_id || null });
    }
  }

  // The newest pose due now; a due held frame keeps the previous one.
  due() {
    const origin = this.origin();
    if (origin === null) return null;
    const now = this.now();
    let frame = null;
    while (this.queue.length && origin + this.queue[0].mediaMs <= now) {
      const entry = this.queue.shift();
      if (entry.frame) frame = entry.frame;
      if (entry.responseId && !this.shown.has(entry.responseId)) {
        this.shown.add(entry.responseId);
        this.onReplyShown(entry.responseId);
      }
    }
    return frame;
  }

  draw() {
    const frame = this.due();
    const skeleton = this.skeleton();
    if (frame === null || !skeleton) return;
    const positions = bvhJointPositions(frame, skeleton);
    const up = AXES[skeleton.up || "y"];
    const across = up === 0 ? 1 : 0;
    const heights = positions.map((p) => p[up]);
    const low = Math.min(...heights);
    const span = Math.max(Math.max(...heights) - low, 1e-6);
    this.canvas.hidden = false;
    const context = this.canvas.getContext("2d");
    const { width, height } = this.canvas;
    const scale = (height * 0.85) / span;
    const rootAcross = positions[0][across];
    context.clearRect(0, 0, width, height);
    context.strokeStyle = "#3f8f86";
    context.lineWidth = 3;
    context.beginPath();
    skeleton.parents.forEach((parent, joint) => {
      if (parent < 0) return;
      const [a, b] = [positions[parent], positions[joint]];
      context.moveTo(width / 2 + (a[across] - rootAcross) * scale, height * 0.95 - (a[up] - low) * scale);
      context.lineTo(width / 2 + (b[across] - rootAcross) * scale, height * 0.95 - (b[up] - low) * scale);
    });
    context.stroke();
  }
}

// The renderer the page registers: registerPluginRenderer(BVH_MOTION, bvhMotionRenderer(view)).
export function bvhMotionRenderer(view) {
  return (event) => view.push(event);
}

// The bvh_motion port's params among the session's granted plugin ports, or null.
export function bvhSkeletonOf(ports) {
  const port = (ports || []).find((candidate) => candidate.format === BVH_MOTION);
  return port ? port.params || null : null;
}
