// The SMPL-X body mesh, skinned in the browser: smplx_motion frames (MIBURI's layout, StreamTalk's) and smplh_flame_motion frames (ELLSA 2) posed on the
// neutral template by linear blend skinning. The model data comes from the playground server at /smplx/mesh.bin, built from the user's own
// SMPL-X file (playground/cascade/app.py --smplx-model; SMPL-X is non-commercial and not shipped); without it the views keep their stick figures.
// Drawing it takes three.js, which smplx_mesh_view.js imports only once the model arrived; this module is plain math and runs under node --test.
import { axisAngleToMatrix } from "./smplh_flame_motion.js";

export const MESH_URL = "/smplx/mesh.bin";
export const NUM_JOINTS = 55;
export const POSE_FEATURES = 3 * NUM_JOINTS;
export const NUM_EXPRESSIONS = 100;
const JAW = 22;
const LEFT_EYE = 23;
const RIGHT_EYE = 24;
// An smplh_flame_motion frame's FLAME code follows its 156 SMPL-H pose values.
const FLAME_START = 156;

function float16ToFloat32(bits) {
  const out = new Float32Array(bits.length);
  for (let index = 0; index < bits.length; index += 1) {
    const h = bits[index];
    const sign = h & 0x8000 ? -1 : 1;
    const exponent = (h >> 10) & 0x1f;
    const fraction = h & 0x3ff;
    if (exponent === 0) out[index] = sign * 2 ** -14 * (fraction / 1024);
    else if (exponent === 31) out[index] = fraction ? NaN : sign * Infinity;
    else out[index] = sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
  }
  return out;
}

const TYPED = { f4: Float32Array, u4: Uint32Array, i4: Int32Array, u1: Uint8Array, f2: Uint16Array };

// The blob app.py serves: a uint32 header length, a JSON header of { name: { dtype, shape, offset } }, then the arrays from the next 4-byte boundary.
export function parseMeshAsset(buffer) {
  const headerLength = new DataView(buffer).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, headerLength)));
  const base = 4 + headerLength;
  const arrays = {};
  for (const [name, { dtype, shape, offset }] of Object.entries(header)) {
    const Typed = TYPED[dtype];
    if (!Typed) throw new Error(`unknown dtype ${dtype} of ${name}`);
    const length = shape.reduce((product, size) => product * size, 1);
    const view = new Typed(buffer.slice(base + offset, base + offset + length * Typed.BYTES_PER_ELEMENT));
    arrays[name] = dtype === "f2" ? float16ToFloat32(view) : view;
  }
  return {
    vertices: arrays.vertices,
    faces: arrays.faces,
    joints: arrays.joints,
    parents: arrays.parents,
    skinIndices: arrays.skin_indices,
    skinWeights: arrays.skin_weights,
    influences: header.skin_indices.shape[1],
    expressionVertices: arrays.expression_vertices || null,
    expressionDirs: arrays.expression_dirs || null,
    numExpressions: header.expression_dirs ? header.expression_dirs.shape[2] : 0,
  };
}

// The skinning transform of every joint for one pose (55 axis-angle rotations, each relative to its parent): row-major 3x4 [R | t] taking a rest
// vertex to its posed place, the root joint kept at the origin. The joint's world transform G chains [R | rest offset from the parent] down the
// tree from the root's [R | J_root]; skinning applies G after taking the joint's rest place J off the vertex, so t = G_t - G_R J - J_root.
export function skinningTransforms(pose, joints, parents) {
  const world = new Float64Array(NUM_JOINTS * 12);
  const out = new Float32Array(NUM_JOINTS * 12);
  for (let joint = 0; joint < NUM_JOINTS; joint += 1) {
    const local = axisAngleToMatrix(pose[3 * joint], pose[3 * joint + 1], pose[3 * joint + 2]);
    const parent = parents[joint];
    const g = joint * 12;
    const j = [joints[3 * joint], joints[3 * joint + 1], joints[3 * joint + 2]];
    for (let row = 0; row < 3; row += 1) {
      if (parent < 0) {
        world[g + row * 4] = local[row * 3];
        world[g + row * 4 + 1] = local[row * 3 + 1];
        world[g + row * 4 + 2] = local[row * 3 + 2];
        world[g + row * 4 + 3] = j[row];
      } else {
        const p = parent * 12;
        for (let column = 0; column < 3; column += 1) {
          world[g + row * 4 + column] = world[p + row * 4] * local[column] + world[p + row * 4 + 1] * local[3 + column] + world[p + row * 4 + 2] * local[6 + column];
        }
        world[g + row * 4 + 3] = world[p + row * 4] * (j[0] - joints[3 * parent]) + world[p + row * 4 + 1] * (j[1] - joints[3 * parent + 1])
          + world[p + row * 4 + 2] * (j[2] - joints[3 * parent + 2]) + world[p + row * 4 + 3];
      }
    }
    for (let row = 0; row < 3; row += 1) {
      for (let column = 0; column < 3; column += 1) out[g + row * 4 + column] = world[g + row * 4 + column];
      out[g + row * 4 + 3] = world[g + row * 4 + 3] - (world[g + row * 4] * j[0] + world[g + row * 4 + 1] * j[1] + world[g + row * 4 + 2] * j[2]) - joints[row];
    }
  }
  return out;
}

// The posed vertices of one frame into out (Float32Array of 3 per vertex): the template plus the expression blend shapes (expression may be null),
// then each vertex moved by its largest skinning weights.
export function skinVertices(asset, pose, expression, out = new Float32Array(asset.vertices.length)) {
  const { vertices, joints, parents, skinIndices, skinWeights, influences } = asset;
  const transforms = skinningTransforms(pose, joints, parents);
  let rest = vertices;
  if (expression && asset.expressionVertices) {
    rest = Float32Array.from(vertices);
    const { expressionVertices, expressionDirs, numExpressions } = asset;
    const count = Math.min(numExpressions, expression.length);
    for (let index = 0; index < expressionVertices.length; index += 1) {
      const vertex = expressionVertices[index];
      for (let axis = 0; axis < 3; axis += 1) {
        const dirs = (index * 3 + axis) * numExpressions;
        let sum = 0;
        for (let k = 0; k < count; k += 1) sum += expressionDirs[dirs + k] * expression[k];
        rest[vertex * 3 + axis] += sum;
      }
    }
  }
  const count = rest.length / 3;
  for (let vertex = 0; vertex < count; vertex += 1) {
    const x = rest[3 * vertex];
    const y = rest[3 * vertex + 1];
    const z = rest[3 * vertex + 2];
    let px = 0;
    let py = 0;
    let pz = 0;
    for (let k = 0; k < influences; k += 1) {
      const weight = skinWeights[vertex * influences + k];
      if (weight === 0) continue;
      const m = skinIndices[vertex * influences + k] * 12;
      px += weight * (transforms[m] * x + transforms[m + 1] * y + transforms[m + 2] * z + transforms[m + 3]);
      py += weight * (transforms[m + 4] * x + transforms[m + 5] * y + transforms[m + 6] * z + transforms[m + 7]);
      pz += weight * (transforms[m + 8] * x + transforms[m + 9] * y + transforms[m + 10] * z + transforms[m + 11]);
    }
    out[3 * vertex] = px;
    out[3 * vertex + 1] = py;
    out[3 * vertex + 2] = pz;
  }
  return out;
}

// An smplx_motion frame (271 values) as the mesh's pose and expression.
export function smplxFramePose(frame) {
  return { pose: frame.subarray(0, POSE_FEATURES), expression: frame.subarray(POSE_FEATURES, POSE_FEATURES + NUM_EXPRESSIONS) };
}

// An smplh_flame_motion frame (264 values) on the SMPL-X body: SMPL-H's body (joints 0-21) and hands (22-51) are SMPL-X's joints 0-21 and 25-54,
// and the FLAME code drives the face the way ELLSA 2's own renderer does (evaluation/render.py load_face): SMPL-X 2020's 100 expression
// coefficients are FLAME 2020's, so its entries 0-99 are the expression as they are; entry 103 turns the jaw about x, and entries 104-105 and
// 106-107 are the left and right eyes' rotations about x and y. Its head rotation (100-102) repeats the body's neck and head and is left out.
export function smplhFlameFramePose(frame) {
  const pose = new Float32Array(POSE_FEATURES);
  pose.set(frame.subarray(0, 66), 0);
  pose.set(frame.subarray(66, 156), 75);
  pose[3 * JAW] = frame[FLAME_START + 103];
  pose.set(frame.subarray(FLAME_START + 104, FLAME_START + 106), 3 * LEFT_EYE);
  pose.set(frame.subarray(FLAME_START + 106, FLAME_START + 108), 3 * RIGHT_EYE);
  return { pose, expression: frame.subarray(FLAME_START, FLAME_START + NUM_EXPRESSIONS) };
}

// Fetch the model blob and hand each view (MotionView, SmplxMotionView) a mesh renderer; the views keep their stick figures when the server has no model.
export async function attachSmplxMesh({
  canvas,
  views,
  fetchAsset = () => fetch(MESH_URL),
  loadRenderer = (target, asset) => import("./smplx_mesh_view.js").then(({ SmplxMeshView }) => new SmplxMeshView(target, asset)),
}) {
  try {
    const response = await fetchAsset();
    if (!response.ok) return null;
    const renderer = await loadRenderer(canvas, parseMeshAsset(await response.arrayBuffer()));
    views.forEach((view) => { view.mesh = renderer; });
    return renderer;
  } catch (error) {
    console.warn("SMPL-X mesh unavailable, drawing a stick figure:", error);
    return null;
  }
}
