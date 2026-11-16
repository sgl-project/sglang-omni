import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

import { ReplyMediaGate } from "../../../realtime/frontend/orb/reply_media_gate.js";
import { DuplexSession, drawsPluginFormat, registerPluginRenderer } from "./session.js";

const appSource = readFileSync(new URL("./app.js", import.meta.url), "utf8").replace(/^import .+;$/gm, "");
// The settings panel is the realtime page's module, served under this page's path by playground/cascade/app.py.
const settingsSource = readFileSync(new URL("../../../realtime/frontend/orb/settings.js", import.meta.url), "utf8");
const pageSource = readFileSync(new URL("./index.html", import.meta.url), "utf8");

// The page with a settings panel that answers the given session options; locks records what the page asked the panel to lock
// and panelOptions what it constructed the panel with; posted is every message the page sent the playback worklet.
function createApp(sessionOptions, search = "") {
  const sockets = [];
  const posted = [];
  const locks = [];
  const panelOptions = [];
  const elements = new Map();
  const context = vm.createContext({
    DuplexSession,
    drawsPluginFormat,
    ReplyMediaGate,
    window: { DEMO_WS_URL: "ws://localhost/v1/realtime", addEventListener() {} },
    location: { search },
    URLSearchParams,
    performance: { now: () => 12345 },
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
    sockets,
    locks,
    panelOptions,
    posted,
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
