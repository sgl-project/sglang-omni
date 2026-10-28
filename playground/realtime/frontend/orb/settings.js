// Session settings for the orb page: presets, system prompt, voice, output, microphone,
// turn detection and the sampling knobs the server declares. Every control follows
// /v1/realtime/capabilities, so a model that lacks a feature simply has no control
// for it. Settings are applied when the next session starts and kept in localStorage.
import { bytesToBase64 } from "./session.js";

// v3: settings stored by v1 carried the persona prompt that stalled audio-only calls; v2 defaulted to the English preset.
// Note (ruoyu): v4 keys the entry by deployment model, so settings saved for one backend do not reach another.
const STORE_PREFIX = "orb.settings.v4";
const REFERENCE_RATE = 16000;
// The server rejects a reference WAV over 1 MiB; 30 s of 16 kHz PCM16 stays under it.
const MAX_REFERENCE_S = 30;
// Note (ruoyu): an empty field is not sent, so the deployment's own default applies; the placeholder says so instead of quoting one backend's number.
const DEFAULT_PLACEHOLDER = "default";
// Note (ruoyu): ranges follow SamplingConfig in sglang_omni/serve/realtime/schema.py; a declared key missing here still gets a plain numeric input.
const SAMPLING_FIELDS = [
  { key: "temperature", label: "Temperature", step: "0.05", min: "0", max: "2" },
  { key: "top_k", label: "Top K", step: "1", min: "-1" },
  { key: "top_p", label: "Top P", step: "0.01", min: "0.01", max: "1" },
  { key: "min_p", label: "Min P", step: "0.01", min: "0", max: "1" },
  { key: "repetition_penalty", label: "Repetition penalty", step: "0.05", min: "1" },
  { key: "listen_prob_scale", label: "Listen scale", step: "0.1", min: "0" },
  { key: "force_listen_count", label: "Initial listen units", step: "1", min: "0" },
  { key: "max_new_tokens_per_unit", label: "Tokens per 1 s unit", step: "1", min: "1" },
  { key: "max_new_tokens", label: "Reply token limit", step: "1", min: "1" },
  { key: "repetition_window_size", label: "Repetition window", step: "1", min: "1" },
  { key: "talker_temperature", label: "Voice temperature", step: "0.05", min: "0", max: "2" },
  { key: "talker_repetition_penalty", label: "Voice repetition penalty", step: "0.05", min: "1" },
  { key: "talker_top_k", label: "Voice top K", step: "1", min: "-1" },
  { key: "talker_top_p", label: "Voice top P", step: "0.01", min: "0.01", max: "1" },
  { key: "talker_max_new_tokens", label: "Voice token limit", step: "1", min: "1" },
  { key: "audio_temperature", label: "Audio temperature", step: "0.05", min: "0", max: "2" },
  { key: "audio_top_k", label: "Audio top K", step: "1", min: "-1" },
  { key: "seed", label: "Seed", step: "1", min: "0" },
];
// Note (ruoyu): the cascade maps max_new_tokens_per_unit to the chat completion's max_tokens, so on a turn-based backend it bounds the whole reply, not one unit.
const TURN_BASED_LABELS = { max_new_tokens_per_unit: "Reply token limit" };
const INTEGER_FIELDS = new Set(["top_k", "force_listen_count", "max_new_tokens_per_unit", "max_new_tokens", "repetition_window_size", "talker_top_k", "talker_max_new_tokens", "audio_top_k", "seed"]);
const TURN_TYPE_LABELS = { server_vad: "Silence ends the turn (server VAD)", semantic_vad: "End-of-turn model (semantic VAD)" };
const EAGERNESS_LEVELS = ["low", "medium", "high"];
// Note (ruoyu): the VAD node's own default in sglang_omni/models/vad/node.py. A profile that runs its vad stage with a shorter
// pause for an end-of-turn model (base-clone: 160 ms) would otherwise cut a bare server_vad pick at every short pause.
const SERVER_VAD_SILENCE_MS = 800;

// Float32 mono samples at 16 kHz -> the {media_type, data} WAV the server accepts.
function referenceWav(samples) {
  const pcm = new DataView(new ArrayBuffer(44 + samples.length * 2));
  const text = (offset, value) => [...value].forEach((character, index) => pcm.setUint8(offset + index, character.charCodeAt(0)));
  text(0, "RIFF"); pcm.setUint32(4, 36 + samples.length * 2, true); text(8, "WAVE");
  text(12, "fmt "); pcm.setUint32(16, 16, true); pcm.setUint16(20, 1, true); pcm.setUint16(22, 1, true);
  pcm.setUint32(24, REFERENCE_RATE, true); pcm.setUint32(28, REFERENCE_RATE * 2, true);
  pcm.setUint16(32, 2, true); pcm.setUint16(34, 16, true);
  text(36, "data"); pcm.setUint32(40, samples.length * 2, true);
  for (let index = 0; index < samples.length; index += 1) {
    const value = Math.max(-1, Math.min(1, samples[index]));
    pcm.setInt16(44 + index * 2, Math.round(value * (value < 0 ? 32768 : 32767)), true);
  }
  return { media_type: "audio/wav", data: bytesToBase64(new Uint8Array(pcm.buffer)) };
}

function decodeFloat32(encoded) {
  const bytes = Uint8Array.from(atob(encoded), (character) => character.charCodeAt(0));
  return new Float32Array(bytes.buffer);
}

function encodeFloat32(samples) {
  return bytesToBase64(new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength));
}

// Any audio file the browser can decode -> mono float32 at 16 kHz, at most MAX_REFERENCE_S.
async function fileToReference(file) {
  const context = new AudioContext();
  try {
    const decoded = await context.decodeAudioData(await file.arrayBuffer());
    const frames = Math.min(Math.ceil(decoded.duration * REFERENCE_RATE), MAX_REFERENCE_S * REFERENCE_RATE);
    const offline = new OfflineAudioContext(1, frames, REFERENCE_RATE);
    const source = offline.createBufferSource();
    source.buffer = decoded;
    source.connect(offline.destination);
    source.start();
    const rendered = await offline.startRendering();
    return { samples: rendered.getChannelData(0), duration: rendered.duration };
  } finally {
    context.close();
  }
}

function load(storeKey) {
  try {
    return JSON.parse(localStorage.getItem(storeKey)) || {};
  } catch {
    return {};
  }
}

export class Settings {
  constructor(ui, { capabilitiesUrl, defaultInstructions, editableInstructions = true, offersTextOnly = true }) {
    this.ui = ui;
    this.capabilitiesUrl = capabilitiesUrl;
    this.defaultInstructions = defaultInstructions;
    this.editableInstructions = editableInstructions;
    this.offersTextOnly = offersTextOnly;
    this.capabilities = null;
    this.storeKey = null;
    this.presets = [];
    this.state = this.defaults();
    this.presetVoice = null;
    this.locked = false;
  }

  defaults() {
    return { presetId: "", prompt: this.defaultInstructions, voice: { source: "default" }, voiceName: "", output: "audio", mic: "", sampling: {}, sliceNums: "", turn: { type: "", silenceMs: "", eagerness: "" } };
  }

  save() {
    if (!this.storeKey) return;
    try {
      localStorage.setItem(this.storeKey, JSON.stringify(this.state));
    } catch {
      // Storage full or blocked (an uploaded voice is large); the settings still apply to this page.
    }
  }

  async init() {
    try {
      const response = await fetch(this.capabilitiesUrl, { cache: "no-store" });
      if (response.ok) this.capabilities = await response.json();
    } catch {
      this.capabilities = null;
    }
    if (this.capabilities) {
      this.storeKey = `${STORE_PREFIX}:${this.capabilities.model}`;
      this.state = { ...this.defaults(), ...load(this.storeKey) };
    }
    if (this.offersPresets) {
      try {
        // Playground presets: {mode: [{id, name, description, system_prompt, ref_audio}]}.
        const response = await fetch("/api/presets", { cache: "no-store" });
        if (response.ok) {
          const byMode = await response.json();
          this.presets = Object.entries(byMode).flatMap(([mode, list]) => list.map((preset) => ({
            ...preset,
            key: `${mode}/${preset.id}`,
            label: mode === "omni" ? `${preset.name} · video` : preset.name,
            mode,
          })));
        }
      } catch {
        this.presets = [];
      }
    }
    if (!this.state.presetId && this.presets.length) {
      // Presets come sorted by their order field; the first one is the deployment's default.
      await this.applyPreset(this.presets[0].key);
    } else if (this.state.voice.source === "preset") {
      await this.loadPresetVoice(this.state.presetId);
    }
    this.render();
    this.bind();
  }

  // Note (ruoyu): a preset is a duplex prompt plus a reference voice, so it only fits a native deployment that takes reference audio; a name match would break under --model-name.
  get offersPresets() {
    return Boolean(this.capabilities && this.capabilities.native_full_duplex && this.capabilities.supports_reference_audio);
  }

  get supportsVoice() {
    return Boolean(this.capabilities && this.capabilities.supports_reference_audio);
  }

  get voiceNames() {
    return (this.capabilities && this.capabilities.voices) || [];
  }

  get samplingKeys() {
    return (this.capabilities && this.capabilities.sampling_parameters) || [];
  }

  get turnTypes() {
    return ((this.capabilities && this.capabilities.turn_control) || []).filter((type) => type in TURN_TYPE_LABELS);
  }

  get samplingFields() {
    const turnBased = Boolean(this.capabilities && this.capabilities.interaction === "turn_based");
    return this.samplingKeys.filter((key) => key !== "greedy").map((key) => {
      const field = SAMPLING_FIELDS.find((known) => known.key === key) || { key, label: key, step: "any" };
      return turnBased && key in TURN_BASED_LABELS ? { ...field, label: TURN_BASED_LABELS[key] } : field;
    });
  }

  async loadPresetVoice(key) {
    this.presetVoice = null;
    const preset = this.presets.find((entry) => entry.key === key);
    if (!preset || !preset.ref_audio) return;
    const response = await fetch(`/api/presets/${preset.mode}/${preset.id}/audio`);
    if (!response.ok) return;
    const body = await response.json();
    if (body.ref_audio) this.presetVoice = { name: body.ref_audio.name, data: body.ref_audio.data, duration: body.ref_audio.duration };
  }

  async applyPreset(key) {
    const preset = this.presets.find((entry) => entry.key === key);
    if (!preset) return;
    this.state.presetId = key;
    if (preset.system_prompt) this.state.prompt = preset.system_prompt;
    await this.loadPresetVoice(key);
    this.state.voice = this.presetVoice ? { source: "preset" } : { source: "default" };
    this.save();
  }

  currentVoice() {
    if (!this.supportsVoice) return null;
    if (this.state.voice.source === "preset") return this.presetVoice;
    if (this.state.voice.source === "upload") return this.state.voice.data ? this.state.voice : null;
    return null;
  }

  turnDetection() {
    const { type, silenceMs, eagerness } = this.state.turn;
    if (!this.turnTypes.includes(type)) return null;
    const detection = { type };
    const silenceDurationMs = parseInt(silenceMs, 10);
    if (type === "server_vad" && Number.isFinite(silenceDurationMs)) detection.silence_duration_ms = silenceDurationMs;
    if (type === "semantic_vad" && EAGERNESS_LEVELS.includes(eagerness)) detection.eagerness = eagerness;
    return detection;
  }

  // What the next session sends: instructions, output modalities, turn detection and session.sglang.
  sessionOptions() {
    const extension = {};
    const sampling = {};
    if (this.samplingKeys.includes("greedy") && this.state.sampling.greedy) sampling.greedy = true;
    for (const { key } of this.samplingFields) {
      const raw = this.state.sampling[key];
      if (raw === undefined || raw === "") continue;
      const value = INTEGER_FIELDS.has(key) ? parseInt(raw, 10) : parseFloat(raw);
      if (Number.isFinite(value)) sampling[key] = value;
    }
    if (Object.keys(sampling).length) extension.sampling = sampling;
    if (this.voiceNames.includes(this.state.voiceName)) extension.voice = this.state.voiceName;
    const voice = this.currentVoice();
    if (voice) extension.reference_audio = referenceWav(decodeFloat32(voice.data));
    const image = this.capabilities && this.capabilities.input_image_format;
    if (image && this.state.sliceNums) extension.max_slice_nums = Math.min(parseInt(this.state.sliceNums, 10), image.max_slice_nums || 1);
    const textOnly = this.offersTextOnly && this.state.output === "text" && this.capabilities && (this.capabilities.output_modalities || []).includes("text");
    return {
      instructions: this.editableInstructions ? (this.state.prompt || "").trim() : "",
      outputModalities: textOnly ? ["text"] : ["audio"],
      extension,
      turnDetection: this.turnDetection(),
    };
  }

  micConstraint() {
    return this.state.mic ? { deviceId: { exact: this.state.mic } } : {};
  }

  async refreshDevices() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
    const devices = (await navigator.mediaDevices.enumerateDevices()).filter((device) => device.kind === "audioinput");
    const select = this.ui.setMic;
    select.replaceChildren(new Option("System default", ""));
    devices.forEach((device, index) => {
      if (device.deviceId && device.deviceId !== "default") select.add(new Option(device.label || `Microphone ${index + 1}`, device.deviceId));
    });
    select.value = [...select.options].some((option) => option.value === this.state.mic) ? this.state.mic : "";
  }

  lock(locked) {
    this.locked = locked;
    this.ui.settingsForm.querySelectorAll("input, select, textarea, button").forEach((element) => { element.disabled = locked; });
    this.ui.settingsLocked.hidden = !locked;
  }

  render() {
    const ui = this.ui;
    ui.presetRow.hidden = !this.presets.length;
    ui.setPreset.replaceChildren(...this.presets.map((preset) => new Option(preset.label, preset.key)), new Option("Custom", ""));
    ui.setPreset.value = this.presets.some((preset) => preset.key === this.state.presetId) ? this.state.presetId : "";
    ui.setPrompt.value = this.state.prompt || "";
    ui.setPrompt.closest("label").hidden = !this.editableInstructions;
    ui.voiceRow.hidden = !this.supportsVoice;
    ui.setVoice.replaceChildren(
      new Option("Model default", "default"),
      ...(this.presetVoice ? [new Option(`Preset: ${this.presetVoice.name}`, "preset")] : []),
      new Option(this.state.voice.source === "upload" && this.state.voice.name ? `Uploaded: ${this.state.voice.name}` : "Upload a recording…", "upload"),
    );
    ui.setVoice.value = this.state.voice.source === "preset" && !this.presetVoice ? "default" : this.state.voice.source;
    const voice = this.currentVoice();
    ui.voicePreview.hidden = !voice;
    ui.voiceInfo.textContent = voice ? `${voice.duration ? voice.duration.toFixed(1) : "?"} s reference` : "";
    ui.voiceNameRow.hidden = !this.voiceNames.length;
    ui.setVoiceName.replaceChildren(new Option("Deployment default", ""), ...this.voiceNames.map((name) => new Option(name, name)));
    ui.setVoiceName.value = this.voiceNames.includes(this.state.voiceName) ? this.state.voiceName : "";
    const outputs = (this.capabilities && this.capabilities.output_modalities) || ["audio"];
    ui.outputRow.hidden = !this.offersTextOnly || !outputs.includes("text");
    ui.setOutput.value = this.state.output;
    const turnTypes = this.turnTypes;
    const turnType = turnTypes.includes(this.state.turn.type) ? this.state.turn.type : "";
    ui.turnRow.hidden = !turnTypes.length;
    ui.setTurnType.replaceChildren(new Option("Deployment default", ""), ...turnTypes.map((type) => new Option(TURN_TYPE_LABELS[type], type)));
    ui.setTurnType.value = turnType;
    ui.silenceRow.hidden = turnType !== "server_vad";
    ui.setSilence.value = this.state.turn.silenceMs;
    ui.eagernessRow.hidden = turnType !== "semantic_vad";
    ui.setEagerness.value = this.state.turn.eagerness;
    const image = this.capabilities && this.capabilities.input_image_format;
    ui.sliceRow.hidden = !image || !(image.max_slice_nums > 1);
    if (image) {
      ui.setSlices.replaceChildren(new Option("Server default", ""), ...Array.from({ length: image.max_slice_nums || 1 }, (_, index) => new Option(index === 0 ? "1 (fastest)" : String(index + 1), String(index + 1))));
      ui.setSlices.value = this.state.sliceNums || "";
    }
    ui.greedyRow.hidden = !this.samplingKeys.includes("greedy");
    ui.setGreedy.checked = Boolean(this.state.sampling.greedy);
    ui.samplingFields.replaceChildren(...this.samplingFields.map((field) => {
      const label = document.createElement("label");
      label.className = "field inline";
      const input = Object.assign(document.createElement("input"), { type: "number", name: field.key, step: field.step, placeholder: DEFAULT_PLACEHOLDER, value: this.state.sampling[field.key] ?? "" });
      if (field.min !== undefined) input.min = field.min;
      if (field.max !== undefined) input.max = field.max;
      label.append(Object.assign(document.createElement("span"), { textContent: field.label }), input);
      return label;
    }));
    ui.advanced.hidden = !this.samplingKeys.length && ui.sliceRow.hidden;
    this.refreshDevices().catch(() => {});
    this.lock(this.locked);
  }

  bind() {
    const ui = this.ui;
    ui.setPreset.addEventListener("change", async () => {
      if (ui.setPreset.value) await this.applyPreset(ui.setPreset.value);
      else { this.state.presetId = ""; this.save(); }
      this.render();
    });
    ui.setPrompt.addEventListener("input", () => {
      this.state.prompt = ui.setPrompt.value;
      // Editing the prompt makes it a custom setup; the voice stays as chosen.
      if (this.state.presetId) { this.state.presetId = ""; ui.setPreset.value = ""; }
      this.save();
    });
    ui.setVoice.addEventListener("change", () => {
      if (ui.setVoice.value === "upload") { ui.voiceFile.click(); return; }
      this.state.voice = { source: ui.setVoice.value };
      this.save();
      this.render();
    });
    ui.voiceFile.addEventListener("change", async () => {
      const file = ui.voiceFile.files && ui.voiceFile.files[0];
      ui.voiceFile.value = "";
      if (!file) { this.render(); return; }
      try {
        const { samples, duration } = await fileToReference(file);
        this.state.voice = { source: "upload", name: file.name, data: encodeFloat32(samples), duration };
        this.save();
      } catch (error) {
        ui.voiceInfo.textContent = `Could not read ${file.name}: ${error.message}`;
      }
      this.render();
    });
    ui.voicePreview.addEventListener("click", async () => {
      const voice = this.currentVoice();
      if (!voice) return;
      const context = new AudioContext({ sampleRate: REFERENCE_RATE });
      const samples = decodeFloat32(voice.data);
      const buffer = context.createBuffer(1, samples.length, REFERENCE_RATE);
      buffer.copyToChannel(samples, 0);
      const source = context.createBufferSource();
      source.buffer = buffer;
      source.connect(context.destination);
      source.onended = () => context.close();
      source.start();
    });
    ui.setVoiceName.addEventListener("change", () => { this.state.voiceName = ui.setVoiceName.value; this.save(); });
    ui.setOutput.addEventListener("change", () => { this.state.output = ui.setOutput.value; this.save(); });
    ui.setMic.addEventListener("change", () => { this.state.mic = ui.setMic.value; this.save(); });
    ui.setTurnType.addEventListener("change", () => {
      this.state.turn.type = ui.setTurnType.value;
      if (ui.setTurnType.value === "server_vad" && this.state.turn.silenceMs === "") this.state.turn.silenceMs = String(SERVER_VAD_SILENCE_MS);
      this.save();
      this.render();
    });
    ui.setSilence.addEventListener("input", () => { this.state.turn.silenceMs = ui.setSilence.value; this.save(); });
    ui.setEagerness.addEventListener("change", () => { this.state.turn.eagerness = ui.setEagerness.value; this.save(); });
    ui.setSlices.addEventListener("change", () => { this.state.sliceNums = ui.setSlices.value; this.save(); });
    ui.setGreedy.addEventListener("change", () => { this.state.sampling.greedy = ui.setGreedy.checked; this.save(); });
    ui.samplingFields.addEventListener("input", (event) => {
      if (!event.target.name) return;
      this.state.sampling[event.target.name] = event.target.value;
      this.save();
    });
    ui.settingsReset.addEventListener("click", async () => {
      this.state = this.defaults();
      if (this.presets.length) await this.applyPreset(this.presets[0].key);
      this.save();
      this.render();
    });
    navigator.mediaDevices?.addEventListener?.("devicechange", () => this.refreshDevices().catch(() => {}));
  }
}
