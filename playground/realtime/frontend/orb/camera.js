// Camera preview and on-demand grabs; a 448 px JPEG per 1 s unit adds about 40 KB/s next to 32 KB/s of audio.
import { bytesToBase64 } from "./session.js";

export const CAMERA_CONSTRAINTS = { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 5 } };
// Tried in order until the encoded frame fits the granted max_bytes; then the size shrinks.
export const QUALITY_STEPS = [0.7, 0.5, 0.35];

export class Camera {
  constructor(preview) {
    this.preview = preview;
    this.stream = null;
    this.canvas = null;
  }

  get active() {
    return Boolean(this.stream);
  }

  async start() {
    if (this.stream) return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) throw new Error("camera capture needs Chrome on http://localhost or HTTPS");
    const stream = await navigator.mediaDevices.getUserMedia({ video: CAMERA_CONSTRAINTS, audio: false });
    this.stream = stream;
    this.preview.muted = true;
    this.preview.srcObject = stream;
    await this.preview.play().catch(() => {});
  }

  stop() {
    if (this.stream) this.stream.getTracks().forEach((track) => track.stop());
    this.stream = null;
    this.preview.srcObject = null;
  }

  async encode(width, height, format, quality) {
    this.canvas.width = width;
    this.canvas.height = height;
    this.canvas.getContext("2d").drawImage(this.preview, 0, 0, width, height);
    const blob = await new Promise((resolve) => this.canvas.toBlob(resolve, format, quality));
    return blob ? new Uint8Array(await blob.arrayBuffer()) : null;
  }

  // One encoded frame under maxBytes, or null while the camera has no frame yet.
  async grab({ format, maxBytes, maxShortSide }) {
    const video = this.preview;
    if (!this.stream || video.readyState < 2 || !video.videoWidth || !video.videoHeight) return null;
    if (!this.canvas) this.canvas = document.createElement("canvas");
    let scale = Math.min(1, maxShortSide / Math.min(video.videoWidth, video.videoHeight));
    for (let attempt = 0; attempt < 6; attempt += 1) {
      const width = Math.max(1, Math.round(video.videoWidth * scale));
      const height = Math.max(1, Math.round(video.videoHeight * scale));
      for (const quality of QUALITY_STEPS) {
        const bytes = await this.encode(width, height, format, quality);
        if (!bytes || !this.stream) return null;
        if (bytes.byteLength <= maxBytes) return { base64: bytesToBase64(bytes), bytes: bytes.byteLength, width, height, quality };
      }
      scale *= 0.75;
    }
    return null;
  }
}
