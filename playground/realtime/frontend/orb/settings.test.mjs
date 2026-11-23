import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

import { DuplexSession, bytesToBase64 } from "./session.js";

const settingsSource = readFileSync(new URL("./settings.js", import.meta.url), "utf8").replace(/^import .+;$/gm, "").replace(/^export /gm, "");
// One capabilities response per backend, as /v1/realtime/capabilities returns it.
const CAPABILITIES = JSON.parse(readFileSync(new URL("./capabilities.fixtures.json", import.meta.url), "utf8"));
const CAPABILITIES_URL = "/v1/realtime/capabilities";
const PRESET_PROMPT = "扮演一个具有以上声音特征的助手。你处于双工模式，可以一边听、一边说。";
const PRESET_SAMPLES = [0, 0.5, -0.5, 1];
const PRESETS = {
  audio_duplex: [{ id: "chinese_call", order: 1, name: "中文通话", description: "", system_prompt: PRESET_PROMPT, ref_audio: { path: "assets/minicpmo/ref_audio/ref_minicpm_signature.wav", name: "ref_minicpm_signature.wav", data: null } }],
  omni: [],
};
const PRESET_AUDIO = { ref_audio: { name: "ref_minicpm_signature.wav", data: Buffer.from(new Float32Array(PRESET_SAMPLES).buffer).toString("base64"), duration: PRESET_SAMPLES.length / 16000 } };
const UI_IDS = ["settingsForm", "settingsLocked", "presetRow", "setPreset", "setPrompt", "voiceRow", "setVoice", "voicePreview", "voiceInfo", "voiceFile", "voiceNameRow", "setVoiceName", "outputRow", "setOutput", "setMic", "turnRow", "setTurnType", "silenceRow", "setSilence", "eagernessRow", "setEagerness", "advanced", "greedyRow", "setGreedy", "samplingFields", "sliceRow", "setSlices", "settingsReset"];

class FakeElement {
  constructor(id) {
    this.id = id;
    this.children = [];
    this.listeners = {};
    this.hidden = false;
    this.disabled = false;
    this.checked = false;
    this.value = "";
    this.textContent = "";
    this.files = null;
    this.label = null;
  }
  addEventListener(name, listener) { this.listeners[name] = listener; }
  replaceChildren(...nodes) { this.children = nodes; }
  append(...nodes) { this.children.push(...nodes); }
  add(option) { this.children.push(option); }
  get options() { return this.children; }
  closest() { if (!this.label) this.label = new FakeElement("label"); return this.label; }
  querySelectorAll() { return []; }
  click() {}
}

class Option {
  constructor(text, value) { this.text = text; this.value = value; }
}

// The settings module in a page whose server answers the given capabilities and the MiniCPM-o presets; pageOptions are the page's own constructor options.
async function createSettings(capabilities, { storage = new Map(), pageOptions = {} } = {}) {
  const ui = Object.fromEntries(UI_IDS.map((id) => [id, new FakeElement(id)]));
  const requests = [];
  const context = vm.createContext({
    bytesToBase64,
    atob,
    btoa,
    console,
    Option,
    document: { createElement: (tag) => new FakeElement(tag) },
    navigator: {},
    localStorage: { getItem: (key) => (storage.has(key) ? storage.get(key) : null), setItem: (key, value) => storage.set(key, value) },
    fetch: async (url) => {
      requests.push(url);
      const bodies = { [CAPABILITIES_URL]: capabilities, "/api/presets": PRESETS, "/api/presets/audio_duplex/chinese_call/audio": PRESET_AUDIO };
      return url in bodies ? { ok: true, json: async () => bodies[url] } : { ok: false, json: async () => ({}) };
    },
  });
  vm.runInContext(settingsSource, context);
  const Settings = vm.runInContext("Settings", context);
  const settings = new Settings(ui, { capabilitiesUrl: CAPABILITIES_URL, defaultInstructions: "", ...pageOptions });
  await settings.init();
  return { settings, ui, requests, storage };
}

function renderedFields(ui) {
  return ui.samplingFields.children.map((label) => {
    const [caption, input] = label.children;
    return { name: input.name, label: caption.textContent, placeholder: input.placeholder, min: input.min };
  });
}

function optionValues(select) {
  return select.options.map((option) => option.value);
}

function choose(ui, id, value) {
  ui[id].value = value;
  return ui[id].listeners.change();
}

function typeSampling(ui, name, value) {
  ui.samplingFields.listeners.input({ target: { name, value } });
}

function typePrompt(ui, prompt) {
  ui.setPrompt.value = prompt;
  ui.setPrompt.listeners.input();
}

// The session.update the page sends for the current settings, through the same DuplexSession app.js uses, as it goes on the wire.
function sessionUpdate(settings) {
  const sent = [];
  const options = settings.sessionOptions();
  const session = new DuplexSession({ transport: (event) => sent.push(JSON.stringify(event)), now: () => 0, outputModalities: options.outputModalities, instructions: options.instructions, extension: options.extension, turnDetection: options.turnDetection });
  session.handle({ type: "session.created", session: { id: "s" } });
  return JSON.parse(sent[0]).session;
}

function pcm16Of(wavBase64) {
  const bytes = Buffer.from(wavBase64, "base64");
  assert.equal(bytes.toString("latin1", 0, 4), "RIFF");
  return [...new Int16Array(bytes.buffer, bytes.byteOffset + 44, (bytes.length - 44) / 2)];
}

test("MiniCPM-o renders every declared knob with a neutral placeholder and sends the first preset's prompt and voice", async () => {
  const { settings, ui, requests } = await createSettings(CAPABILITIES.minicpm_o);
  assert.deepEqual(renderedFields(ui), [
    { name: "temperature", label: "Temperature", placeholder: "default", min: "0" },
    { name: "top_k", label: "Top K", placeholder: "default", min: "-1" },
    { name: "top_p", label: "Top P", placeholder: "default", min: "0.01" },
    { name: "repetition_penalty", label: "Repetition penalty", placeholder: "default", min: "1" },
    { name: "listen_prob_scale", label: "Listen scale", placeholder: "default", min: "0" },
    { name: "force_listen_count", label: "Initial listen units", placeholder: "default", min: "0" },
    { name: "max_new_tokens_per_unit", label: "Tokens per 1 s unit", placeholder: "default", min: "1" },
    { name: "repetition_window_size", label: "Repetition window", placeholder: "default", min: "1" },
    { name: "talker_temperature", label: "Voice temperature", placeholder: "default", min: "0" },
    { name: "talker_repetition_penalty", label: "Voice repetition penalty", placeholder: "default", min: "1" },
  ]);
  assert.equal(ui.greedyRow.hidden, false);
  assert.equal(ui.voiceRow.hidden, false);
  assert.equal(ui.voiceNameRow.hidden, true);
  assert.equal(ui.turnRow.hidden, true);
  assert.equal(ui.presetRow.hidden, false);
  assert.ok(requests.includes("/api/presets"));

  const granted = sessionUpdate(settings);
  assert.deepEqual(Object.keys(granted).sort(), ["instructions", "output_modalities", "sglang"]);
  assert.equal(granted.instructions, PRESET_PROMPT);
  assert.deepEqual(granted.output_modalities, ["audio"]);
  assert.equal(granted.sglang.reference_audio.media_type, "audio/wav");
  assert.deepEqual(pcm16Of(granted.sglang.reference_audio.data), [0, 16384, -16384, 32767]);

  await choose(ui, "setVoice", "default");
  typePrompt(ui, "Answer briefly.");
  ui.setGreedy.checked = true;
  ui.setGreedy.listeners.change();
  typeSampling(ui, "temperature", "0.5");
  typeSampling(ui, "top_k", "40");
  typeSampling(ui, "force_listen_count", "0");
  typeSampling(ui, "max_new_tokens_per_unit", "");
  assert.deepEqual(sessionUpdate(settings), {
    output_modalities: ["audio"],
    instructions: "Answer briefly.",
    sglang: { sampling: { greedy: true, temperature: 0.5, top_k: 40, force_listen_count: 0 } },
  });
});

test("Qwen3-Omni gets its own sampling knobs and server VAD, no preset and no voice", async () => {
  const { settings, ui, requests } = await createSettings(CAPABILITIES.qwen3_omni);
  assert.deepEqual(renderedFields(ui), [
    { name: "temperature", label: "Temperature", placeholder: "default", min: "0" },
    { name: "max_new_tokens", label: "Reply token limit", placeholder: "default", min: "1" },
    { name: "talker_max_new_tokens", label: "Voice token limit", placeholder: "default", min: "1" },
    { name: "talker_temperature", label: "Voice temperature", placeholder: "default", min: "0" },
    { name: "talker_top_k", label: "Voice top K", placeholder: "default", min: "-1" },
    { name: "talker_top_p", label: "Voice top P", placeholder: "default", min: "0.01" },
    { name: "talker_repetition_penalty", label: "Voice repetition penalty", placeholder: "default", min: "1" },
    { name: "seed", label: "Seed", placeholder: "default", min: "0" },
  ]);
  assert.equal(ui.greedyRow.hidden, true);
  assert.equal(ui.voiceRow.hidden, true);
  assert.equal(ui.voiceNameRow.hidden, true);
  assert.equal(ui.presetRow.hidden, true);
  assert.equal(requests.includes("/api/presets"), false);
  assert.equal(ui.turnRow.hidden, false);
  assert.deepEqual(optionValues(ui.setTurnType), ["", "server_vad"]);
  assert.equal(ui.silenceRow.hidden, true);
  assert.deepEqual(sessionUpdate(settings), { output_modalities: ["audio"] });

  await choose(ui, "setTurnType", "server_vad");
  assert.equal(ui.silenceRow.hidden, false);
  assert.equal(ui.eagernessRow.hidden, true);
  assert.equal(ui.setSilence.value, "800");
  typePrompt(ui, "Reply in one sentence.");
  typeSampling(ui, "temperature", "0.3");
  typeSampling(ui, "max_new_tokens", "256");
  typeSampling(ui, "talker_top_k", "-1");
  typeSampling(ui, "seed", "7");
  assert.deepEqual(sessionUpdate(settings), {
    output_modalities: ["audio"],
    instructions: "Reply in one sentence.",
    audio: { input: { turn_detection: { type: "server_vad", silence_duration_ms: 800 } } },
    sglang: { sampling: { temperature: 0.3, max_new_tokens: 256, talker_top_k: -1, seed: 7 } },
  });
});

test("the streaming cascade labels the token limit as a reply limit and offers both turn detectors", async () => {
  const { settings, ui, requests } = await createSettings(CAPABILITIES.streaming_cascade);
  assert.deepEqual(renderedFields(ui), [
    { name: "temperature", label: "Temperature", placeholder: "default", min: "0" },
    { name: "max_new_tokens_per_unit", label: "Reply token limit", placeholder: "default", min: "1" },
  ]);
  assert.equal(ui.greedyRow.hidden, true);
  assert.equal(ui.voiceRow.hidden, true);
  assert.equal(ui.voiceNameRow.hidden, true);
  assert.equal(ui.sliceRow.hidden, true);
  assert.equal(ui.presetRow.hidden, true);
  assert.equal(requests.includes("/api/presets"), false);
  assert.deepEqual(optionValues(ui.setTurnType), ["", "server_vad", "semantic_vad"]);
  assert.deepEqual(sessionUpdate(settings), { output_modalities: ["audio"] });

  // A bare server_vad inherits the profile's VAD pause, 160 ms on base-clone, so the pick prefills the VAD node's own 800 ms and the user may edit it.
  await choose(ui, "setTurnType", "server_vad");
  assert.equal(ui.setSilence.value, "800");
  assert.deepEqual(sessionUpdate(settings).audio, { input: { turn_detection: { type: "server_vad", silence_duration_ms: 800 } } });
  ui.setSilence.value = "1200";
  ui.setSilence.listeners.input();
  assert.equal(sessionUpdate(settings).audio.input.turn_detection.silence_duration_ms, 1200);

  await choose(ui, "setTurnType", "semantic_vad");
  assert.equal(ui.silenceRow.hidden, true);
  assert.equal(ui.eagernessRow.hidden, false);
  await choose(ui, "setEagerness", "high");
  typeSampling(ui, "temperature", "0.2");
  typeSampling(ui, "max_new_tokens_per_unit", "128");
  assert.deepEqual(sessionUpdate(settings), {
    output_modalities: ["audio"],
    audio: { input: { turn_detection: { type: "semantic_vad", eagerness: "high" } } },
    sglang: { sampling: { temperature: 0.2, max_new_tokens_per_unit: 128 } },
  });

  // Back to the deployment default: nothing about turns is sent, whatever was typed before.
  await choose(ui, "setTurnType", "");
  assert.equal(ui.eagernessRow.hidden, true);
  assert.equal("audio" in sessionUpdate(settings), false);
});

test("PersonaPlex offers its packaged voices by name and its own sampling knobs", async () => {
  const { settings, ui, requests } = await createSettings(CAPABILITIES.personaplex);
  assert.deepEqual(renderedFields(ui), [
    { name: "temperature", label: "Temperature", placeholder: "default", min: "0" },
    { name: "top_k", label: "Top K", placeholder: "default", min: "-1" },
    { name: "top_p", label: "Top P", placeholder: "default", min: "0.01" },
    { name: "min_p", label: "Min P", placeholder: "default", min: "0" },
    { name: "repetition_penalty", label: "Repetition penalty", placeholder: "default", min: "1" },
    { name: "audio_temperature", label: "Audio temperature", placeholder: "default", min: "0" },
    { name: "audio_top_k", label: "Audio top K", placeholder: "default", min: "-1" },
    { name: "seed", label: "Seed", placeholder: "default", min: "0" },
  ]);
  assert.equal(ui.voiceRow.hidden, true);
  assert.equal(ui.voiceNameRow.hidden, false);
  assert.deepEqual(optionValues(ui.setVoiceName), ["", ...CAPABILITIES.personaplex.voices]);
  assert.equal(ui.turnRow.hidden, true);
  assert.equal(ui.presetRow.hidden, true);
  assert.equal(requests.includes("/api/presets"), false);
  assert.deepEqual(sessionUpdate(settings), { output_modalities: ["audio"] });

  await choose(ui, "setVoiceName", "NATM1");
  typePrompt(ui, "You are a travel agent.");
  typeSampling(ui, "min_p", "0.05");
  typeSampling(ui, "audio_top_k", "250");
  typeSampling(ui, "seed", "3");
  assert.deepEqual(sessionUpdate(settings), {
    output_modalities: ["audio"],
    instructions: "You are a travel agent.",
    sglang: { voice: "NATM1", sampling: { min_p: 0.05, audio_top_k: 250, seed: 3 } },
  });
});

test("a page that offers no text-only replies hides the choice and sends audio even with a stored text output", async () => {
  const storage = new Map([[`orb.settings.v4:${CAPABILITIES.streaming_cascade.model}`, JSON.stringify({ output: "text" })]]);
  const { settings, ui } = await createSettings(CAPABILITIES.streaming_cascade, { storage, pageOptions: { offersTextOnly: false } });
  assert.equal(ui.outputRow.hidden, true);
  assert.deepEqual(sessionUpdate(settings), { output_modalities: ["audio"] });
  await choose(ui, "setTurnType", "server_vad");
  assert.deepEqual(sessionUpdate(settings), {
    output_modalities: ["audio"],
    audio: { input: { turn_detection: { type: "server_vad", silence_duration_ms: 800 } } },
  });
});

test("a declared sampling key the page does not know still gets a numeric control under its own name", async () => {
  const capabilities = { ...CAPABILITIES.personaplex, sampling_parameters: ["temperature", "mystery_knob"] };
  const { settings, ui } = await createSettings(capabilities);
  assert.deepEqual(renderedFields(ui).map(({ name, label }) => ({ name, label })), [
    { name: "temperature", label: "Temperature" },
    { name: "mystery_knob", label: "mystery_knob" },
  ]);
  typeSampling(ui, "mystery_knob", "1.5");
  assert.deepEqual(sessionUpdate(settings).sglang, { sampling: { mystery_knob: 1.5 } });
});

test("saved settings are kept per model and an entry without a model is ignored", async () => {
  const storage = new Map([["orb.settings.v3", JSON.stringify({ presetId: "", prompt: "old prompt", voice: { source: "default" }, output: "audio", mic: "", sampling: { temperature: "0.1" }, sliceNums: "" })]]);
  const first = await createSettings(CAPABILITIES.minicpm_o, { storage });
  assert.equal(first.ui.setPrompt.value, PRESET_PROMPT);
  assert.equal(sessionUpdate(first.settings).sglang.sampling, undefined);
  typeSampling(first.ui, "temperature", "0.5");
  assert.deepEqual([...storage.keys()], ["orb.settings.v3", "orb.settings.v4:openbmb/MiniCPM-o-4_5"]);

  const other = await createSettings(CAPABILITIES.streaming_cascade, { storage });
  assert.equal(other.ui.setPrompt.value, "");
  assert.deepEqual(sessionUpdate(other.settings), { output_modalities: ["audio"] });
  typePrompt(other.ui, "Answer in English.");

  const reloaded = await createSettings(CAPABILITIES.minicpm_o, { storage });
  assert.equal(reloaded.ui.setPrompt.value, PRESET_PROMPT);
  assert.equal(sessionUpdate(reloaded.settings).sglang.sampling.temperature, 0.5);
  const otherReloaded = await createSettings(CAPABILITIES.streaming_cascade, { storage });
  assert.equal(sessionUpdate(otherReloaded.settings).instructions, "Answer in English.");
});
