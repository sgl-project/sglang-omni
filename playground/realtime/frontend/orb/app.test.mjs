import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

import { DuplexSession } from "./session.js";

const appSource = readFileSync(new URL("./app.js", import.meta.url), "utf8").replace(/^import .+;$/gm, "");

function createApp({ actualSampleRate, loadModule = async () => {} } = {}) {
  const nodes = [];
  const sockets = [];
  const elements = new Map();
  let microphoneRequests = 0;
  const context = vm.createContext({
    DuplexSession,
    window: { DEMO_WS_URL: "ws://localhost/v1/realtime", addEventListener() {} },
    location: { search: "" },
    URLSearchParams,
    performance: { now: () => 12345 },
    navigator: { mediaDevices: { getUserMedia: async () => { microphoneRequests += 1; throw new Error("No microphone in playback test"); } } },
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
    Camera: class { stop() {} },
    Orb: class {},
    Settings: class {
      async init() {}
      lock() {}
      sessionOptions() { return {}; }
      micConstraint() { return {}; }
    },
    AudioContext: class {
      constructor({ sampleRate }) {
        this.sampleRate = actualSampleRate || sampleRate;
        this.audioWorklet = { addModule: () => loadModule(sampleRate) };
        this.destination = {};
      }
      async resume() {}
      async close() {}
      createAnalyser() { return {}; }
    },
    AudioWorkletNode: class {
      constructor(audioContext) {
        this.context = audioContext;
        this.messages = [];
        this.port = { onmessage: null, postMessage: (message) => this.messages.push(message) };
        nodes.push(this);
      }
      connect() { return this; }
    },
    WebSocket: class {
      static OPEN = 1;
      static CLOSING = 2;
      constructor() {
        this.readyState = 0;
        this.listeners = {};
        sockets.push(this);
      }
      addEventListener(name, listener) { this.listeners[name] = listener; }
      send() {}
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
    nodes,
    sockets,
    get microphoneRequests() { return microphoneRequests; },
    get session() { return vm.runInContext("session", context); },
    connect: () => vm.runInContext("connect()", context),
    settled: () => vm.runInContext("eventChain", context),
  };
}

function grant(socket, unitMilliseconds, outputSampleRate = 24000) {
  socket.receive({ type: "session.updated", session: { sglang: { granted: {
    native_unit_ms: unitMilliseconds,
    input_audio_format: { rate: 16000 },
    output_audio_format: { rate: outputSampleRate },
  } } } });
}

function status(generation) {
  return { type: "status", generation, droppedMs: 10, rate: 1, queueMs: 500, targetMs: 300, playedMs: 1000, buffering: false, underruns: 2, outputRms: 0.2, responseOpen: true, responseId: "unattributed" };
}

test("reused nodes restore each granted profile and attribute status only to its session", async () => {
  const app = createApp({ actualSampleRate: 48000 });
  let previousSession;
  let previousGeneration;
  let previousHandler;
  for (const unitMilliseconds of [200, 1000, 200]) {
    await app.connect();
    const socket = app.sockets.at(-1);
    socket.open();
    grant(socket, unitMilliseconds);
    await app.settled();
    assert.equal(app.nodes.length, 1);
    const node = app.nodes[0];
    const configuration = node.messages.findLast((message) => message.type === "config");
    const streaming = unitMilliseconds === 200;
    assert.equal(configuration.initialSamples, streaming ? 19200 : 14400);
    assert.equal(configuration.maxJitterSamples, streaming ? 43200 : 14400);
    assert.equal(configuration.maxSamples, streaming ? 168000 : 120000);
    assert.equal(configuration.keepSamples, streaming ? 105600 : 72000);
    assert.deepEqual({ ...configuration.catchUp }, streaming
      ? { slow: 0.5, fast: 1.0, skip: 2.0, slowRate: 1.06, fastRate: 1.12 }
      : { slow: 0.2, fast: 0.7, skip: 1.5, slowRate: 1.12, fastRate: 1.25 });
    assert.equal(configuration.nativeUnitSamples, streaming ? 9600 : 48000);
    const generation = node.messages.findLast((message) => message.type === "reset").generation;
    if (previousSession) {
      const previousRows = previousSession.trace.length;
      node.port.onmessage({ data: status(previousGeneration) });
      previousHandler({ data: status(previousGeneration) });
      assert.equal(previousSession.trace.length, previousRows);
      assert.equal(app.session.trace.filter((row) => row.direction === "local").length, 0);
    }
    node.port.onmessage({ data: status(generation) });
    const row = JSON.parse(app.session.exportTrace()).trace.at(-1);
    assert.equal(row.direction, "local");
    assert.equal(row.time_s, 12.345);
    assert.equal(row.event.type, "playback.status");
    assert.equal(row.event.generation, generation);
    assert.equal(row.event.underruns, 2);
    assert.equal(row.event.responseOpen, true);
    assert.equal("responseId" in row.event, false);
    socket.receive({ type: "response.output_audio.delta", response_id: "reply", delta: Buffer.alloc(48000, 32).toString("base64") });
    await app.settled();
    assert.equal(node.messages.at(-1).samples.length, 48000);
    const resets = node.messages.filter((message) => message.type === "reset").length;
    grant(socket, unitMilliseconds);
    await app.settled();
    assert.equal(node.messages.filter((message) => message.type === "reset").length, resets);
    previousSession = app.session;
    previousGeneration = generation;
    previousHandler = node.port.onmessage;
    await socket.close();
  }
});

test("audio waits for a replacement node to be configured and old node status is ignored", async () => {
  const moduleReady = Promise.withResolvers();
  const moduleLoading = Promise.withResolvers();
  const app = createApp({ loadModule: (sampleRate) => {
    if (sampleRate !== 48000) return Promise.resolve();
    moduleLoading.resolve();
    return moduleReady.promise;
  } });
  await app.connect();
  const socket = app.sockets[0];
  socket.open();
  grant(socket, 1000);
  await app.settled();
  const oldNode = app.nodes[0];
  const oldHandler = oldNode.port.onmessage;
  const oldGeneration = oldNode.messages.findLast((message) => message.type === "reset").generation;
  grant(socket, 1000, 48000);
  socket.receive({ type: "response.created", response: { id: "reply" } });
  socket.receive({ type: "response.output_audio.delta", response_id: "reply", delta: Buffer.alloc(96000, 32).toString("base64") });
  await moduleLoading.promise;
  assert.equal(oldNode.port.onmessage, null);
  assert.ok(oldNode.messages.every((message) => message.type !== "push"));
  moduleReady.resolve();
  await app.settled();
  const node = app.nodes.at(-1);
  assert.notEqual(node, oldNode);
  const messages = node.messages.map((message) => message.type);
  assert.ok(messages.indexOf("config") < messages.indexOf("push"));
  assert.ok(messages.indexOf("reset") < messages.indexOf("push"));
  assert.equal(node.messages.at(-1).samples.length, 48000);
  const rows = app.session.trace.length;
  oldHandler({ data: status(oldGeneration) });
  assert.equal(app.session.trace.length, rows);
});

test("a grant finishing during reconnect cannot configure or start the next session", async () => {
  const moduleReady = Promise.withResolvers();
  const moduleLoading = Promise.withResolvers();
  const app = createApp({ loadModule: (sampleRate) => {
    if (sampleRate !== 48000) return Promise.resolve();
    moduleLoading.resolve();
    return moduleReady.promise;
  } });
  await app.connect();
  const oldSocket = app.sockets[0];
  oldSocket.open();
  grant(oldSocket, 200, 48000);
  oldSocket.receive({ type: "response.output_audio.delta", response_id: "old", delta: "" });
  await moduleLoading.promise;
  await oldSocket.close();
  const reconnect = app.connect();
  moduleReady.resolve();
  await reconnect;
  assert.equal(app.microphoneRequests, 0);
  assert.ok(app.nodes.every((node) => node.messages.every((message) => !["config", "push"].includes(message.type))));
  const socket = app.sockets.at(-1);
  socket.open();
  grant(socket, 1000);
  await app.settled();
  assert.equal(app.microphoneRequests, 1);
  const node = app.nodes.at(-1);
  assert.equal(node.context.sampleRate, 24000);
  assert.equal(node.messages.find((message) => message.type === "config").nativeUnitSamples, 24000);
});
