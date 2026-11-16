import test from "node:test";
import assert from "node:assert/strict";

import { DEFAULT_PLAYOUT_DELAY_MS, VideoFrameView } from "./video_frame.js";

function frame(name, tStartMs, responseId = null) {
  return { t_start_ms: tStartMs, duration_ms: 40, response_id: responseId, payload_b64: name ? btoa(name) : "" };
}

function view(now, options = { playoutDelayMs: 0 }) {
  const shown = [];
  const frames = new VideoFrameView(null, { now: () => now.ms, onReplyShown: (id) => shown.push([id, now.ms]), ...options });
  const due = () => {
    const bytes = frames.due();
    return bytes === null ? null : new TextDecoder().decode(bytes);
  };
  return { frames, shown, due };
}

test("frames play on their media clock from the first frame's arrival, the newest due one drawn", () => {
  const now = { ms: 1000 };
  const { frames, due } = view(now);
  for (const [name, tStartMs] of [["a", 200], ["b", 240], ["c", 280]]) frames.push(frame(name, tStartMs));
  assert.equal(due(), "a");
  now.ms = 1039;
  assert.equal(due(), null);
  now.ms = 1040;
  assert.equal(due(), "b");
  now.ms = 1200;
  assert.equal(due(), "c");
});

test("a late frame shows on arrival and an empty frame keeps the previous picture", () => {
  const now = { ms: 0 };
  const { frames, due } = view(now);
  frames.push(frame("a", 0));
  assert.equal(due(), "a");
  now.ms = 120;
  frames.push(frame("", 40));
  frames.push(frame("c", 80));
  assert.equal(due(), "c");
  frames.push(frame("", 120));
  assert.equal(due(), null);
});

test("a reply is shown once, when its first frame is due, held frames included", () => {
  const now = { ms: 0 };
  const { frames, shown, due } = view(now);
  frames.push(frame("idle", 0));
  frames.push(frame("", 40, "resp_1"));
  frames.push(frame("talk", 80, "resp_1"));
  due();
  assert.deepEqual(shown, []);
  now.ms = 40;
  due();
  now.ms = 80;
  assert.equal(due(), "talk");
  assert.deepEqual(shown, [["resp_1", 40]]);
});

test("an interrupted reply keeps its last shown picture: its frames not yet due and later ones of it are dropped", () => {
  const now = { ms: 0 };
  const { frames, due } = view(now);
  frames.push(frame("t0", 0, "resp_1"));
  frames.push(frame("t1", 40, "resp_1"));
  assert.equal(due(), "t0");
  frames.drop("resp_1");
  frames.push(frame("t2", 80, "resp_1"));
  frames.push(frame("idle", 120));
  now.ms = 119;
  assert.equal(due(), null);
  now.ms = 120;
  assert.equal(due(), "idle");
});

test("draw decodes the due picture and only the newest decode reaches the canvas", async () => {
  const now = { ms: 0 };
  const drawn = [];
  const resolvers = [];
  const canvas = { hidden: true, getContext: () => ({ drawImage: (picture) => drawn.push(picture.name) }) };
  const frames = new VideoFrameView(canvas, {
    now: () => now.ms,
    playoutDelayMs: 0,
    decode: (bytes) => new Promise((resolve) => resolvers.push(() => resolve({ name: new TextDecoder().decode(bytes), width: 4, height: 2 }))),
  });
  frames.push(frame("a", 0));
  frames.draw();
  now.ms = 40;
  frames.push(frame("b", 40));
  frames.draw();
  resolvers[1]();
  resolvers[0]();
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.deepEqual(drawn, ["b"]);
  assert.equal(canvas.hidden, false);
  assert.equal(canvas.width, 4);
});

test("by default frames play 100 ms after the first frame's arrival, so a frame up to 100 ms late shows on its time", () => {
  const now = { ms: 1000 };
  const { frames, shown, due } = view(now, {});
  assert.equal(DEFAULT_PLAYOUT_DELAY_MS, 100);
  frames.push(frame("a", 200, "resp_1"));
  assert.equal(due(), null);
  now.ms = 1099;
  assert.equal(due(), null);
  now.ms = 1100;
  assert.equal(due(), "a");
  assert.deepEqual(shown, [["resp_1", 1100]]);
  // b is due at 1140 and arrives 90 ms behind the first frame's pace, c 110 ms behind.
  now.ms = 1130;
  frames.push(frame("b", 240));
  now.ms = 1140;
  assert.equal(due(), "b");
  now.ms = 1190;
  frames.push(frame("c", 280));
  assert.equal(due(), "c");
});

test("given a clock, frames play on its origin, and none plays while it holds the origin back", () => {
  const now = { ms: 1000 };
  let origin = null;
  const { frames, due } = view(now, { playoutDelayMs: 0, clock: { mediaOrigin: () => origin } });
  frames.push(frame("a", 0));
  frames.push(frame("b", 40));
  now.ms = 3000;
  assert.equal(due(), null);
  origin = 2980;
  assert.equal(due(), "a");
  now.ms = 3020;
  assert.equal(due(), "b");
});
