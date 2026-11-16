// Draws the skinned SMPL-X mesh (smplx_mesh.js) with three.js, which the page maps in its import map; the vertices are skinned on the CPU each
// frame and the normals recomputed, seen from the front with the root at the origin.
import * as THREE from "three";

import { skinVertices } from "./smplx_mesh.js";

const SKIN = 0xc9a48a;
const FIT_MARGIN = 1.15;
// Half the width kept in view, in meters: arms out to the side in a gesture reach about this far from the spine.
const HALF_WIDTH_M = 1.0;
const FOV_DEG = 28;

export class SmplxMeshView {
  constructor(canvas, asset) {
    this.canvas = canvas;
    this.asset = asset;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(globalThis.devicePixelRatio || 1, 2));
    this.scene = new THREE.Scene();
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x445566, 1.6));
    const key = new THREE.DirectionalLight(0xffffff, 1.8);
    key.position.set(0.6, 1.2, 2.0);
    this.scene.add(key);
    this.positions = new Float32Array(asset.vertices.length);
    skinVertices(asset, new Float32Array(165), null, this.positions);
    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute("position", new THREE.BufferAttribute(this.positions, 3));
    this.geometry.setIndex(new THREE.BufferAttribute(asset.faces, 1));
    this.geometry.computeVertexNormals();
    this.mesh = new THREE.Mesh(this.geometry, new THREE.MeshStandardMaterial({ color: SKIN, roughness: 0.75, metalness: 0 }));
    this.scene.add(this.mesh);
    // Frame the rest body from the knees up, where co-speech gestures happen.
    this.geometry.computeBoundingBox();
    const { min, max } = this.geometry.boundingBox;
    const bottom = min.y + 0.35 * (max.y - min.y);
    this.fitHeight = (max.y - bottom) * FIT_MARGIN;
    this.centerY = (max.y + bottom) / 2;
    this.camera = new THREE.PerspectiveCamera(FOV_DEG, 3 / 4, 0.05, 20);
    this.place();
  }

  // Back the camera off until both the body's height and HALF_WIDTH_M either side of it fit the canvas.
  place() {
    const tangent = Math.tan(THREE.MathUtils.degToRad(FOV_DEG / 2));
    const distance = Math.max(this.fitHeight / 2 / tangent, HALF_WIDTH_M / (tangent * this.camera.aspect));
    this.camera.position.set(0, this.centerY, distance);
    this.camera.lookAt(0, this.centerY, 0);
  }

  resize() {
    const width = this.canvas.clientWidth || this.canvas.width;
    const height = this.canvas.clientHeight || this.canvas.height;
    if (this.width === width && this.height === height) return;
    this.width = width;
    this.height = height;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.place();
  }

  // Pose the mesh (55 axis-angle rotations, an optional 100 expression coefficients) and draw it.
  show(pose, expression) {
    this.canvas.hidden = false;
    this.resize();
    skinVertices(this.asset, pose, expression, this.positions);
    this.geometry.attributes.position.needsUpdate = true;
    this.geometry.computeVertexNormals();
    this.renderer.render(this.scene, this.camera);
  }
}
