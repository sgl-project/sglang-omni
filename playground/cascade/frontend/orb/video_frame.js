// The jpeg_frame output format drawn on a canvas: one JPEG per frame; an empty payload keeps the previous picture.
import { base64ToBytes } from "./session.js";

export const JPEG_FRAME = "jpeg_frame";

// Frames queue on the chunks' media clock, the Session input timeline the node stamps them on, and play from the first frame's arrival;
// a frame arriving after its time shows on arrival, and when several are due only the newest is drawn.
// onReplyShown(responseId) fires once, when the first frame of a reply is due: the page's media gate plays that reply's audio from then on,
// so voice and lips start together.
// drop(responseId) ends an interrupted reply's picture where it was cut: its frames not yet due, and any that still arrive, are dropped,
// and the last picture stays until the next listening frame is due.
export class VideoFrameView {
  // decode(bytes) resolves to something drawImage takes; the page passes createImageBitmap over a JPEG Blob.
  constructor(canvas, { now = () => performance.now(), onReplyShown = () => {}, decode = null } = {}) {
    this.canvas = canvas;
    this.now = now;
    this.onReplyShown = onReplyShown;
    this.decode = decode || ((bytes) => createImageBitmap(new Blob([bytes], { type: "image/jpeg" })));
    this.reset();
  }

  push(event) {
    if (event.response_id && this.dropped.has(event.response_id)) return;
    if (this.originMs === null) this.originMs = this.now() - event.t_start_ms;
    const bytes = event.payload_b64 ? base64ToBytes(event.payload_b64) : new Uint8Array(0);
    this.queue.push({ atMs: this.originMs + event.t_start_ms, bytes, responseId: event.response_id || null });
  }

  // The newest picture due now, or null when no new picture is due; a due empty frame keeps the previous one.
  due() {
    const now = this.now();
    let bytes = null;
    while (this.queue.length && this.queue[0].atMs <= now) {
      const entry = this.queue.shift();
      if (entry.bytes.length) bytes = entry.bytes;
      if (entry.responseId && !this.shown.has(entry.responseId)) {
        this.shown.add(entry.responseId);
        this.onReplyShown(entry.responseId);
      }
    }
    return bytes;
  }

  drop(responseId) {
    this.dropped.add(responseId);
    this.queue = this.queue.filter((entry) => entry.responseId !== responseId);
  }

  reset() {
    this.queue = [];
    this.originMs = null;
    this.shown = new Set();
    this.dropped = new Set();
    // The newest decode wins; an older one finishing later is not drawn.
    this.sequence = 0;
  }

  draw() {
    const bytes = this.due();
    if (bytes === null) return;
    const sequence = ++this.sequence;
    this.decode(bytes).then((picture) => {
      if (sequence !== this.sequence) return;
      this.canvas.width = picture.width;
      this.canvas.height = picture.height;
      this.canvas.getContext("2d").drawImage(picture, 0, 0);
      this.canvas.hidden = false;
      if (picture.close) picture.close();
    }).catch(() => {});
  }
}

// The renderer the page registers: registerPluginRenderer(JPEG_FRAME, jpegFrameRenderer(view)).
export function jpegFrameRenderer(view) {
  return (event) => view.push(event);
}
