import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { setImmediate as nextEventLoopTurn } from "node:timers/promises";
import vm from "node:vm";

import { DuplexSession } from "./session.js";

const appSource = readFileSync(new URL("./app.js", import.meta.url), "utf8").replace(/^import .+;$/gm, "");
// The settings panel is the realtime page's module, served under this page's path by playground/cascade/app.py.
const settingsSource = readFileSync(new URL("../../../realtime/frontend/orb/settings.js", import.meta.url), "utf8");
const pageSource = readFileSync(new URL("./index.html", import.meta.url), "utf8");

// The page with a settings panel that answers the given session options; locks records what the page asked the panel to lock
// and panelOptions what it constructed the panel with.
function createApp(sessionOptions, microphone = null) {
  const sockets = [];
  const locks = [];
  const panelOptions = [];
  const elements = new Map();
  const playbackCommands = [];
  const microphonePorts = [];
  const context = vm.createContext({
    DuplexSession,
    window: { DEMO_WS_URL: "ws://localhost/v1/realtime", addEventListener() {} },
    location: { search: "" },
    URLSearchParams,
    URL: { createObjectURL: () => "blob:preview", revokeObjectURL() {} },
    createImageBitmap: async () => ({ width: 1, height: 1, close() {} }),
    performance: { now: () => 12345 },
    navigator: { mediaDevices: { getUserMedia: async () => {
      if (microphone) return microphone;
      throw new Error("No microphone in playback test");
    } } },
    document: {
      documentElement: { dataset: {} },
      addEventListener() {},
      createElement: () => ({ dataset: {} }),
      getElementById: (id) => {
        if (!elements.has(id)) elements.set(id, { dataset: {}, listeners: {}, textContent: "", replaceChildren() {}, addEventListener(name, listener) { this.listeners[name] = listener; }, setAttribute() {} });
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
      async refreshDevices() {}
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
      createMediaStreamSource() { return { connect() { return this; } }; }
      createGain() { return { gain: { value: 1 }, connect() { return this; } }; }
    },
    AudioWorkletNode: class {
      constructor(context, processorName) {
        this.port = { onmessage: null, postMessage(message) { playbackCommands.push(message); } };
        if (processorName === "capture-processor") microphonePorts.push(this.port);
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
    sockets,
    locks,
    panelOptions,
    elements,
    playbackCommands,
    microphonePorts,
    evaluate: (source) => vm.runInContext(source, context),
    connect: () => vm.runInContext("connect()", context),
    settled: async () => {
      await vm.runInContext("eventChain", context);
      await nextEventLoopTurn();
    },
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

test("the page offers no text-only replies, since its captions and truncate follow audio playback", () => {
  const app = createApp({ instructions: "", outputModalities: ["audio"], extension: {}, turnDetection: null });
  assert.equal(app.panelOptions.length, 1);
  assert.equal(app.panelOptions[0].offersTextOnly, false);
});

test("microphone permission failure clears the accepted image so recording can be retried", async () => {
  const app = createApp({ instructions: "", outputModalities: ["audio"], extension: {}, turnDetection: null });
  const { socket } = await firstSessionUpdate(app);
  socket.receive({ type: "session.updated", session: { sglang: { granted: { input_audio_format: { rate: 16000 }, input_modalities: ["audio", "image"], input_image_format: { max_bytes: 100 }, client_commit: true } } } });
  await app.settled();
  const imageFile = app.elements.get("imageFile");
  imageFile.files = [{ name: "cup.png", size: 4, type: "image/png", arrayBuffer: async () => new Uint8Array([137, 80, 78, 71]).buffer }];
  await imageFile.listeners.change();
  for (let attempt = 0; attempt < 2; attempt += 1) {
    const recording = app.elements.get("questionBtn").listeners.click();
    const imageEvent = socket.sent.at(-1);
    assert.equal(imageEvent.type, "sglang.input_image.append");
    socket.receive({ type: "sglang.input_image.accepted", client_event_id: imageEvent.event_id, unit_id: "unit_0" });
    await app.settled();
    await recording;
    assert.equal(socket.sent.at(-1).type, "input_audio_buffer.clear");
    assert.equal(app.elements.get("imageStatus").textContent, "Ready");
  }
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

test("VAD image sessions acknowledge uploads before opening a continuous microphone and clear playback on speech", async () => {
  let stopped = false;
  const app = createApp({ outputModalities: ["audio"] }, { getTracks: () => [{ stop() { stopped = true; } }] });
  const { socket } = await firstSessionUpdate(app);
  app.evaluate('settings.capabilities = { input_modalities: ["audio", "image"], input_image_format: { max_bytes: 100 }, client_commit: true, turn_control: ["server_vad", null] }');
  const imageFile = app.elements.get("imageFile");
  imageFile.files = [{ name: "cup.png", size: 4, type: "image/png", arrayBuffer: async () => new Uint8Array([137, 80, 78, 71]).buffer }];
  await imageFile.listeners.change();
  socket.receive({ type: "session.updated", session: { sglang: { granted: {
    interaction: "turn_based", input_audio_format: { rate: 16000 }, input_modalities: ["audio", "image"],
    input_image_format: { max_bytes: 100 }, client_commit: true, turn_control: ["server_vad", null],
  } } } });
  await app.settled();
  const imageEvent = socket.sent.at(-1);
  assert.equal(imageEvent.type, "sglang.input_image.append");
  assert.equal(app.microphonePorts.length, 0);
  socket.receive({ type: "sglang.input_image.accepted", client_event_id: imageEvent.event_id, unit_id: "unit_0" });
  await app.settled();
  assert.equal(app.microphonePorts.length, 1);
  assert.equal(app.elements.get("questionBtn").hidden, true);
  assert.equal(app.elements.get("imageTools").hidden, false);
  assert.equal(app.elements.get("muteBtn").hidden, false);
  socket.receive({ type: "response.created", response: { id: "reply-1" } });
  await app.settled();
  app.microphonePorts[0].onmessage({ data: { type: "frame", frame: new Int16Array(1280).fill(42).buffer } });
  assert.equal(socket.sent.at(-1).type, "input_audio_buffer.append");
  assert.equal(stopped, false);
  socket.receive({ type: "input_audio_buffer.speech_started" });
  await app.settled();
  assert.equal(app.playbackCommands.at(-1).type, "clear");
  assert.equal(socket.sent.some((event) => event.type === "input_audio_buffer.commit"), false);
  await socket.close();
  assert.equal(stopped, true);
});

test("a failed microphone in a live image session can be retried from the microphone control", async () => {
  const app = createApp({ outputModalities: ["audio"] });
  const { socket } = await firstSessionUpdate(app);
  app.evaluate('settings.capabilities = { input_modalities: ["audio", "image"], input_image_format: { max_bytes: 100 }, client_commit: true, turn_control: ["server_vad", null] }');
  const imageFile = app.elements.get("imageFile");
  imageFile.files = [{ name: "cup.png", size: 4, type: "image/png", arrayBuffer: async () => new Uint8Array([137, 80, 78, 71]).buffer }];
  await imageFile.listeners.change();
  socket.receive({ type: "session.updated", session: { sglang: { granted: {
    interaction: "turn_based", input_audio_format: { rate: 16000 }, input_modalities: ["audio", "image"],
    input_image_format: { max_bytes: 100 }, client_commit: true, turn_control: ["server_vad", null],
  } } } });
  await app.settled();
  const firstImage = socket.sent.at(-1);
  assert.equal(firstImage.type, "sglang.input_image.append");
  socket.receive({ type: "sglang.input_image.accepted", client_event_id: firstImage.event_id, unit_id: "unit_0" });
  await app.settled();
  assert.equal(app.elements.get("imageStatus").textContent, "Ready");
  assert.equal(app.elements.get("muteBtn").disabled, false);
  assert.equal(app.elements.get("muteBtn").title, "Start microphone");
  const initialClearCount = socket.sent.filter((event) => event.type === "input_audio_buffer.clear").length;
  app.elements.get("muteBtn").listeners.click();
  await app.settled();
  const retriedImage = socket.sent.at(-1);
  assert.equal(retriedImage.type, "sglang.input_image.append");
  assert.notEqual(retriedImage.event_id, firstImage.event_id);
  socket.receive({ type: "sglang.input_image.accepted", client_event_id: retriedImage.event_id, unit_id: "unit_0" });
  await app.settled();
  assert.equal(socket.sent.filter((event) => event.type === "input_audio_buffer.clear").length, initialClearCount + 1);
});

test("images picked together are sent in order, only new picks are sent later, and the tray keeps the granted window", async () => {
  const app = createApp({ outputModalities: ["audio"] }, { getTracks: () => [{ stop() {} }] });
  const { socket } = await firstSessionUpdate(app);
  const imageFormat = { max_bytes: 100, max_frames_per_unit: 2 };
  socket.receive({ type: "session.updated", session: { sglang: { granted: {
    interaction: "turn_based", input_audio_format: { rate: 16000 }, input_modalities: ["audio", "image"],
    input_image_format: imageFormat, client_commit: true, turn_control: ["server_vad", null],
  } } } });
  await app.settled();
  const imageFile = app.elements.get("imageFile");
  const png = (name, marker) => ({ name, size: 5, type: "image/png", arrayBuffer: async () => new Uint8Array([137, 80, 78, 71, marker]).buffer });
  const pick = async (...files) => {
    const sentBefore = socket.sent.length;
    imageFile.files = files;
    const picking = imageFile.listeners.change();
    await app.settled();
    const appended = socket.sent.slice(sentBefore).filter((event) => event.type === "sglang.input_image.append");
    for (const event of appended) socket.receive({ type: "sglang.input_image.accepted", client_event_id: event.event_id, unit_id: "unit_0" });
    await picking;
    return appended.map((event) => Buffer.from(event.image, "base64")[4]);
  };
  assert.deepEqual(await pick(png("a.png", 1), png("b.png", 2), png("c.png", 3)), [2, 3]);
  assert.equal(app.elements.get("warning").textContent, "Only the last 2 images are kept");
  assert.equal(app.elements.get("imageName").textContent, "2 images");
  assert.equal(app.elements.get("imageStatus").textContent, "Received");
  assert.deepEqual(await pick(png("d.png", 4)), [4]);
  assert.equal(app.evaluate("images.map((image) => image.name).join()"), "c.png,d.png");
});
