import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

import { DEFAULT_PLAYOUT_DELAY_MS, ReplyMediaGate, playoutDelayFrom } from "../../../realtime/frontend/orb/reply_media_gate.js";
import { DuplexSession, drawsPluginFormat, registerPluginRenderer } from "./session.js";
import { MotionView, SMPLH_FLAME_MOTION, smplhFlameMotionRenderer } from "./smplh_flame_motion.js";
import { JPEG_FRAME, VideoFrameView, jpegFrameRenderer } from "./video_frame.js";
import { SMPLX_MOTION, SmplxMotionView, smplxMotionRenderer } from "./smplx_motion.js";

const appSource = readFileSync(new URL("./app.js", import.meta.url), "utf8").replace(/^import .+;$/gm, "");
// The settings panel is the realtime page's module, served under this page's path by playground/cascade/app.py.
const settingsSource = readFileSync(new URL("../../../realtime/frontend/orb/settings.js", import.meta.url), "utf8");
const pageSource = readFileSync(new URL("./index.html", import.meta.url), "utf8");

// The page with a settings panel that answers the given session options; locks records what the page asked the panel to lock
// and panelOptions what it constructed the panel with; posted is every message the page sent the playback worklet.
function createApp(sessionOptions, search = "") {
  const meshes = [];
  const sockets = [];
  const posted = [];
  const locks = [];
  const panelOptions = [];
  const elements = new Map();
  const clock = { ms: 12345 };
  const context = vm.createContext({
    DuplexSession,
    drawsPluginFormat,
    registerPluginRenderer,
    ReplyMediaGate,
    playoutDelayFrom,
    MotionView,
    SMPLH_FLAME_MOTION,
    smplhFlameMotionRenderer,
    SMPLX_MOTION,
    SmplxMotionView,
    smplxMotionRenderer,
    attachSmplxMesh: (options) => { meshes.push(options); return Promise.resolve(null); },
    JPEG_FRAME,
    VideoFrameView,
    jpegFrameRenderer,
    window: { DEMO_WS_URL: "ws://localhost/v1/realtime", addEventListener() {} },
    location: { search },
    URLSearchParams,
    performance: { now: () => clock.ms },
    navigator: { mediaDevices: { getUserMedia: async () => { throw new Error("No microphone in playback test"); } } },
    document: {
      documentElement: { dataset: {} },
      addEventListener() {},
      getElementById: (id) => {
        if (!elements.has(id)) elements.set(id, { dataset: {}, addEventListener() {}, setAttribute() {} });
        return elements.get(id);
      },
    },
    matchMedia: () => ({ matches: false }),
    localStorage: { getItem() { return null; } },
    setTimeout() {},
    clearTimeout() {},
    setInterval() {},
    requestAnimationFrame() {},
    ResizeObserver: class { observe() {} },
    Orb: class {},
    Settings: class {
      constructor(ui, options) { panelOptions.push(options); }
      async init() {}
      lock(locked) { locks.push(locked); }
      sessionOptions() { return sessionOptions; }
      micConstraint() { return {}; }
    },
    AudioContext: class {
      constructor({ sampleRate }) {
        this.sampleRate = sampleRate;
        this.audioWorklet = { addModule: async () => {} };
        this.destination = {};
      }
      async resume() {}
      async close() {}
      createAnalyser() { return {}; }
    },
    AudioWorkletNode: class {
      constructor() {
        this.port = { onmessage: null, postMessage: (message) => posted.push(message) };
      }
      connect() { return this; }
    },
    WebSocket: class {
      static OPEN = 1;
      static CLOSING = 2;
      constructor() {
        this.readyState = 0;
        this.listeners = {};
        this.sent = [];
        sockets.push(this);
      }
      addEventListener(name, listener) { this.listeners[name] = listener; }
      send(text) { this.sent.push(JSON.parse(text)); }
      open() {
        this.readyState = 1;
        this.listeners.open();
        this.receive({ type: "session.created", session: { id: "test-session" } });
      }
      receive(event) { this.listeners.message({ data: JSON.stringify(event) }); }
      async close() { this.readyState = 3; await this.listeners.close({ code: 1000 }); }
    },
  });
  vm.runInContext(appSource, context);
  return {
    meshes,
    sockets,
    locks,
    panelOptions,
    posted,
    clock,
    run: (source) => vm.runInContext(source, context),
    connect: () => vm.runInContext("connect()", context),
    settled: () => vm.runInContext("eventChain", context),
  };
}

async function firstSessionUpdate(app) {
  await app.connect();
  const socket = app.sockets.at(-1);
  socket.open();
  await app.settled();
  return { socket, session: socket.sent.find((event) => event.type === "session.update").session };
}

test("the page has every element and every session.js export the shared settings module uses", async () => {
  const usedIds = [...new Set([...settingsSource.matchAll(/\bui\.([A-Za-z]+)/g)].map((match) => match[1]))];
  const pageIds = new Set([...pageSource.matchAll(/\bid="([A-Za-z]+)"/g)].map((match) => match[1]));
  const lookedUpIds = new Set([...appSource.match(/const ids = \[([^\]]*)\]/)[1].matchAll(/"([A-Za-z]+)"/g)].map((match) => match[1]));
  assert.deepEqual(usedIds.filter((id) => !pageIds.has(id)), []);
  assert.deepEqual(usedIds.filter((id) => !lookedUpIds.has(id)), []);
  const importedNames = [...settingsSource.matchAll(/^import \{([^}]+)\} from "\.\/session\.js";/gm)].flatMap((match) => match[1].split(",").map((name) => name.trim()));
  const session = await import("./session.js");
  assert.ok(importedNames.length > 0);
  assert.deepEqual(importedNames.filter((name) => typeof session[name] !== "function"), []);
});

test("the page offers both motion views the SMPL-X mesh on its own canvas", () => {
  const app = createApp({ instructions: "", outputModalities: ["audio"], extension: {}, turnDetection: null });
  assert.equal(app.meshes.length, 1);
  assert.equal(app.meshes[0].views.length, 2);
  assert.ok(app.meshes[0].views.every((view) => view instanceof MotionView));
  assert.ok(/id="meshView"/.test(pageSource) && /"three": "https:\/\/cdn\.jsdelivr\.net\/npm\/three@[0-9.]+\//.test(pageSource));
});

test("the page offers no text-only replies, since its captions and truncate follow audio playback", () => {
  const app = createApp({ instructions: "", outputModalities: ["audio"], extension: {}, turnDetection: null });
  assert.equal(app.panelOptions.length, 1);
  assert.equal(app.panelOptions[0].offersTextOnly, false);
});

test("the page sends only output_modalities while the panel is empty and the chosen settings otherwise, locked during the call", async () => {
  const empty = createApp({ instructions: "", outputModalities: ["audio"], extension: {}, turnDetection: null });
  const { socket, session } = await firstSessionUpdate(empty);
  assert.deepEqual(session, { output_modalities: ["audio"] });
  assert.equal(empty.locks.at(-1), true);
  await socket.close();
  assert.equal(empty.locks.at(-1), false);

  const filled = createApp({
    instructions: "Answer in one short English sentence.",
    outputModalities: ["audio"],
    extension: { sampling: { temperature: 0.2, max_new_tokens_per_unit: 128 } },
    turnDetection: { type: "semantic_vad", eagerness: "high" },
  });
  assert.deepEqual((await firstSessionUpdate(filled)).session, {
    output_modalities: ["audio"],
    instructions: "Answer in one short English sentence.",
    audio: { input: { turn_detection: { type: "semantic_vad", eagerness: "high" } } },
    sglang: { sampling: { temperature: 0.2, max_new_tokens_per_unit: 128 } },
  });
});

test("a reply's audio waits for its first frame of a drawn port that follows replies, and plays on arrival with ?av_sync=free", async () => {
  registerPluginRenderer("test_frame", () => {});
  const port = { modality: "video", format: "test_frame", params: { fps: 25, timeline: "input", follows_replies: true } };
  const pushes = (app) => app.posted.filter((message) => message.type === "push").map((message) => message.responseId);
  for (const [search, beforeShown] of [["", []], ["?av_sync=free", ["resp_1"]]]) {
    const app = createApp({ instructions: "", outputModalities: ["audio"], extension: {}, turnDetection: null }, search);
    const { socket } = await firstSessionUpdate(app);
    socket.receive({ type: "session.updated", session: { sglang: { granted: { input_audio_format: { rate: 16000 }, output_audio_format: { rate: 24000 } }, plugin_output_ports: [port] } } });
    socket.receive({ type: "response.created", response: { id: "resp_1" } });
    socket.receive({ type: "response.output_audio.delta", response_id: "resp_1", item_id: "item_1", delta: Buffer.alloc(480).toString("base64") });
    await app.settled();
    assert.deepEqual(pushes(app), beforeShown, search);
    app.run('mediaGate.shown("resp_1")');
    assert.deepEqual(pushes(app), ["resp_1"], search);
  }
});

test("with the ELLSA 2 motion port, a reply's audio waits for the reply's first motion frame, and an interrupt drops the reply's motion", async () => {
  const app = createApp({ instructions: "", outputModalities: ["audio"], extension: {}, turnDetection: null });
  const { socket } = await firstSessionUpdate(app);
  const motion = { modality: "motion", format: "smplh_flame_motion", params: { fps: 30, timeline: "input", follows_replies: true, reply_media_wait_ms: 1000 } };
  socket.receive({ type: "session.updated", session: { sglang: { granted: { input_audio_format: { rate: 16000 }, output_audio_format: { rate: 24000 } }, plugin_output_ports: [motion] } } });
  socket.receive({ type: "response.created", response: { id: "resp_1" } });
  socket.receive({ type: "response.output_audio.delta", response_id: "resp_1", item_id: "item_1", delta: Buffer.alloc(480).toString("base64") });
  await app.settled();
  const pushes = () => app.posted.filter((message) => message.type === "push" && message.responseId === "resp_1");
  assert.equal(pushes().length, 0);
  assert.equal(app.run("mediaGate.waitMs"), 1000 + DEFAULT_PLAYOUT_DELAY_MS);
  const frames = Buffer.from(new Float32Array(24 * 264).buffer).toString("base64");
  socket.receive({ type: "sglang.output_chunk", response_id: "resp_1", turn: 1, modality: "motion", format: "smplh_flame_motion", t_start_ms: 0, duration_ms: 800, payload_b64: frames });
  socket.receive({ type: "sglang.output_chunk", response_id: "resp_1", turn: 1, modality: "motion", format: "smplh_flame_motion", t_start_ms: 800, duration_ms: 800, payload_b64: frames });
  await app.settled();
  assert.equal(pushes().length, 0);
  app.clock.ms += DEFAULT_PLAYOUT_DELAY_MS - 1;
  app.run("motionView.due()");
  assert.equal(pushes().length, 0);
  app.clock.ms += 1;
  app.run("motionView.due()");
  assert.equal(pushes().length, 1);
  socket.receive({ type: "response.done", response: { id: "resp_1", status: "cancelled" } });
  await app.settled();
  assert.equal(app.run('motionView.queue.filter((entry) => entry.responseId === "resp_1").length'), 0);
});

test("with the AVTR-1 video port, a reply's audio waits for the reply's first frame, ?av_sync=free plays it on arrival, and an interrupt drops the reply's frames", async () => {
  const video = { modality: "video", format: "jpeg_frame", params: { fps: 25, timeline: "input", follows_replies: true, reply_media_wait_ms: 1000, width: 640, height: 360 } };
  for (const [search, beforeShown] of [["", 0], ["?av_sync=free", 1]]) {
    const app = createApp({ instructions: "", outputModalities: ["audio"], extension: {}, turnDetection: null }, search);
    const { socket } = await firstSessionUpdate(app);
    socket.receive({ type: "session.updated", session: { sglang: { granted: { input_audio_format: { rate: 16000 }, output_audio_format: { rate: 24000 } }, plugin_output_ports: [video] } } });
    socket.receive({ type: "response.created", response: { id: "resp_1" } });
    socket.receive({ type: "response.output_audio.delta", response_id: "resp_1", item_id: "item_1", delta: Buffer.alloc(480).toString("base64") });
    await app.settled();
    const pushes = () => app.posted.filter((message) => message.type === "push" && message.responseId === "resp_1");
    assert.equal(pushes().length, beforeShown, search);
    assert.equal(app.run("mediaGate.waitMs"), 1000 + DEFAULT_PLAYOUT_DELAY_MS);
    const jpeg = Buffer.from("jpeg").toString("base64");
    socket.receive({ type: "sglang.output_chunk", response_id: "resp_1", turn: 1, modality: "video", format: "jpeg_frame", t_start_ms: 0, duration_ms: 40, payload_b64: jpeg });
    socket.receive({ type: "sglang.output_chunk", response_id: "resp_1", turn: 1, modality: "video", format: "jpeg_frame", t_start_ms: 40, duration_ms: 40, payload_b64: jpeg });
    await app.settled();
    assert.equal(pushes().length, beforeShown, search);
    app.run("videoView.due()");
    assert.equal(pushes().length, beforeShown, search);
    app.clock.ms += DEFAULT_PLAYOUT_DELAY_MS;
    app.run("videoView.due()");
    assert.equal(pushes().length, 1, search);
    socket.receive({ type: "response.done", response: { id: "resp_1", status: "cancelled" } });
    await app.settled();
    assert.equal(app.run('videoView.queue.filter((entry) => entry.responseId === "resp_1").length'), 0);
  }
});

test("?playout_delay_ms sets the renderers' playout delay and the media gate's wait with it", async () => {
  const motion = { modality: "motion", format: "smplh_flame_motion", params: { fps: 30, timeline: "input", follows_replies: true, reply_media_wait_ms: 1000 } };
  for (const [search, delayMs] of [["", DEFAULT_PLAYOUT_DELAY_MS], ["?playout_delay_ms=250", 250], ["?playout_delay_ms=0", 0], ["?playout_delay_ms=-5", DEFAULT_PLAYOUT_DELAY_MS]]) {
    const app = createApp({ instructions: "", outputModalities: ["audio"], extension: {}, turnDetection: null }, search);
    const { socket } = await firstSessionUpdate(app);
    socket.receive({ type: "session.updated", session: { sglang: { granted: { input_audio_format: { rate: 16000 }, output_audio_format: { rate: 24000 } }, plugin_output_ports: [motion] } } });
    await app.settled();
    assert.equal(app.run("mediaGate.waitMs"), 1000 + delayMs, search);
    assert.equal(app.run("motionView.playoutDelayMs"), delayMs, search);
    assert.equal(app.run("videoView.playoutDelayMs"), delayMs, search);
  }
});

test("with a native full-duplex model, the reply audio and the motion play on one clock, the port's lookahead plus the playout delay behind the audio, and ?av_sync=free plays the audio on arrival", async () => {
  const motion = { modality: "motion", format: "smplh_flame_motion", params: { fps: 30, timeline: "input", follows_replies: true, reply_media_wait_ms: 1000, reply_lookahead_ms: 800 } };
  const frames = Buffer.from(new Float32Array(24 * 264).buffer).toString("base64");
  for (const search of ["", "?av_sync=free"]) {
    const app = createApp({ instructions: "", outputModalities: ["audio"], extension: {}, turnDetection: null }, search);
    const { socket } = await firstSessionUpdate(app);
    socket.receive({ type: "session.updated", session: { sglang: { granted: { interaction: "native", native_full_duplex: true, input_audio_format: { rate: 24000 }, output_audio_format: { rate: 24000 } }, plugin_output_ports: [motion] } } });
    await app.settled();
    const start = app.clock.ms;
    socket.receive({ type: "response.created", response: { id: "resp_s" } });
    // 160 ms of agent audio, the first unit's.
    socket.receive({ type: "response.output_audio.delta", response_id: "resp_s", item_id: "item_s", delta: Buffer.alloc(2 * 3840).toString("base64") });
    await app.settled();
    const pushes = () => app.posted.filter((message) => message.type === "push" && message.responseId === "resp_s").length;
    if (search) {
      assert.equal(app.run("mediaGate.isAligning"), false);
      assert.equal(pushes(), 1, search);
      continue;
    }
    assert.equal(app.run("mediaGate.isAligning"), true);
    assert.equal(pushes(), 0);
    // The first motion chunk arrives one block later; it plays on the audio's clock, not from its own arrival.
    app.clock.ms += 900;
    socket.receive({ type: "sglang.output_chunk", response_id: null, turn: null, modality: "motion", format: "smplh_flame_motion", t_start_ms: 0, duration_ms: 800, payload_b64: frames });
    await app.settled();
    assert.equal(app.run("motionView.origin()"), start + 800 + DEFAULT_PLAYOUT_DELAY_MS);
    app.clock.ms = start + 800 + DEFAULT_PLAYOUT_DELAY_MS - 1;
    app.run("mediaGate.tick()");
    assert.equal(pushes(), 0);
    assert.equal(app.run("motionView.due()"), null);
    app.clock.ms += 1;
    app.run("mediaGate.tick()");
    assert.equal(pushes(), 1);
    assert.notEqual(app.run("motionView.due()"), null);
  }
});
