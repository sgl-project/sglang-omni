// Voice page for a native full-duplex /v1/realtime session, proxied by playground/realtime/app.py.
import { DuplexSession } from "./session.js";
import { Camera } from "./camera.js";
import { Orb } from "./orb.js";
import { Settings } from "./settings.js";

const JITTER_MS = 300;
const STATUS_POLL_MS = 5000;
const SESSION_CAP_S = window.DEMO_SESSION_CAP_S || 600;
// RMS below the floor reads as silence; above it the level rises to 1 at about -14 dBFS.
const LEVEL_FLOOR = 0.008;
const LEVEL_GAIN = 5;
const MIC_TALK_LEVEL = 0.12;
const MODEL_TALK_LEVEL = 0.04;
const CAPTION_FADE_MS = 5000;
// Output quieter than this counts as silence when deciding to end a local interrupt.
const QUIET_RMS = 0.01;
const UNMUTE_QUIET_MS = 400;
// The stop button is offered while the model has been audible this recently.
const INTERRUPT_WINDOW_MS = 1500;
// Speech has gaps between syllables; a mood holds this long after the last loud frame.
const MODEL_HOLD_MS = 700;
const MIC_HOLD_MS = 400;

const ids = ["brandName", "statusPill", "statusText", "transcriptBtn", "traceBtn", "themeBtn", "warning", "selfView", "cameraPreview", "frameCount", "orb", "orbCanvas", "statusLine", "caption", "note", "startBtn", "dock", "cameraBtn", "cameraOn", "cameraOff", "muteBtn", "micOn", "micOff", "interruptBtn", "endBtn", "hint", "sheet", "sheetClose", "transcript", "settingsBtn", "settingsPanel", "settingsClose", "settingsForm", "settingsLocked", "presetRow", "setPreset", "setPrompt", "voiceRow", "setVoice", "voicePreview", "voiceInfo", "voiceFile", "outputRow", "setOutput", "setMic", "advanced", "greedyRow", "setGreedy", "samplingFields", "sliceRow", "setSlices", "settingsReset"];
const ui = Object.fromEntries(ids.map((id) => [id, document.getElementById(id)]));

function wsUrl() {
  const base = window.DEMO_WS_URL || "";
  const token = new URLSearchParams(location.search).get("token");
  if (!token) return base;
  return `${base}${base.includes("?") ? "&" : "?"}token=${encodeURIComponent(token)}`;
}

const CLOSE_MESSAGES = {
  4401: "This demo needs an access link with a token.",
  4408: "The 10-minute limit was reached. Thanks for trying the demo!",
  4429: "Someone else is talking to the model right now, please try again in a minute.",
};

let socket = null;
let session = null;
// The most recent session, kept after it closes so its trace can still be downloaded.
let lastSession = null;
let connecting = false;
let startingMic = false;
let micGeneration = 0;
let captureContext = null;
let captureStream = null;
let captureNode = null;
let micAnalyser = null;
let playbackContext = null;
let playbackNode = null;
let playbackAnalyser = null;
let playbackRate = 24000;
let eventChain = Promise.resolve();
let remoteBusy = false;
let serverUp = null;
let sessionStart = null;
let muted = false;
let mutedResponse = false;
// Quiet output heard since a local interrupt; enough of it ends the mute.
let mutedQuietMs = 0;
// A response is open but has not produced audio yet.
let awaitingAudio = false;
let lastSpokeAt = 0;
let lastModelLoudAt = -Infinity;
let lastMicLoudAt = -Infinity;
let warnTimer = null;
// The camera is offered only when the server declares image input in
// session.updated; window.DEMO_CAMERA = false hides it even then.
const OFFER_CAMERA = window.DEMO_CAMERA !== false;
const camera = new Camera(ui.cameraPreview);
let startingCamera = false;
const orb = new Orb(ui.orbCanvas);
// /v1/realtime/capabilities next to the WebSocket endpoint, unless config.js names it.
const capabilitiesUrl = window.DEMO_CAPABILITIES_URL || (window.DEMO_WS_URL || "").replace(/^ws/, "http").replace(/\/v1\/realtime.*$/, "/v1/realtime/capabilities");
const settings = new Settings(ui, { capabilitiesUrl, defaultInstructions: window.DEMO_INSTRUCTIONS || "", editableInstructions: window.DEMO_EDITABLE_INSTRUCTIONS !== false });
const darkQuery = matchMedia("(prefers-color-scheme: dark)");

// Deployment-specific text comes from config.js, so one page serves any duplex model.
if (window.DEMO_MODEL_NAME) {
  ui.brandName.textContent = window.DEMO_MODEL_NAME;
  document.title = `${window.DEMO_MODEL_NAME} Live`;
}
if (window.DEMO_NOTE) ui.note.textContent = window.DEMO_NOTE;

function isOpen() {
  return Boolean(socket) && socket.readyState === WebSocket.OPEN;
}

function isReady() {
  return Boolean(session) && session.state === "ready" && isOpen();
}

function inCall() {
  return isOpen() || connecting;
}

function isDark() {
  const forced = document.documentElement.dataset.theme;
  return forced ? forced === "dark" : darkQuery.matches;
}

function warn(message, sticky = false) {
  clearTimeout(warnTimer);
  ui.warning.textContent = message;
  ui.warning.hidden = !message;
  if (message && !sticky) warnTimer = setTimeout(() => { ui.warning.hidden = true; }, 7000);
}

function setButtons() {
  const call = inCall();
  ui.startBtn.hidden = call;
  ui.dock.hidden = !call;
  ui.note.hidden = call;
  ui.caption.hidden = !call;
  ui.startBtn.disabled = remoteBusy || serverUp === false;
  ui.orb.setAttribute("aria-label", call ? "Show transcript" : "Start conversation");
  ui.muteBtn.disabled = !captureStream;
  ui.muteBtn.setAttribute("aria-pressed", String(muted));
  ui.muteBtn.setAttribute("aria-label", muted ? "Unmute microphone" : "Mute microphone");
  ui.muteBtn.title = muted ? "Unmute microphone" : "Mute microphone";
  ui.micOn.hidden = muted;
  ui.micOff.hidden = !muted;
  const offerCamera = OFFER_CAMERA && Boolean(session && session.imageGranted && isReady());
  ui.cameraBtn.hidden = !offerCamera;
  ui.cameraBtn.disabled = startingCamera;
  ui.cameraBtn.setAttribute("aria-pressed", String(camera.active));
  ui.cameraBtn.setAttribute("aria-label", camera.active ? "Turn camera off" : "Turn camera on");
  ui.cameraBtn.title = camera.active ? "Turn camera off" : "Turn camera on";
  ui.cameraOn.hidden = !camera.active;
  ui.cameraOff.hidden = camera.active;
  ui.selfView.hidden = !camera.active;
  settings.lock(call);
}

// Level (0..1) from an analyser's current time-domain window.
const levelBuffer = new Float32Array(2048);
function levelOf(analyser) {
  if (!analyser) return 0;
  const samples = levelBuffer.subarray(0, analyser.fftSize);
  analyser.getFloatTimeDomainData(samples);
  let energy = 0;
  for (let index = 0; index < samples.length; index += 1) energy += samples[index] * samples[index];
  const rms = Math.sqrt(energy / samples.length);
  return Math.min(1, Math.max(0, (rms - LEVEL_FLOOR) * LEVEL_GAIN) ** 0.7);
}

// What the orb and the status line show, derived every frame.
function mood(micLevel, modelLevel, now) {
  if (session && session.state === "error") return { mood: "error", level: 0, line: "Something went wrong" };
  if (connecting || (isOpen() && !isReady())) return { mood: "thinking", level: 0, line: "Connecting…" };
  if (!isOpen()) {
    if (serverUp === false) return { mood: "idle", level: 0, line: "The demo is offline right now" };
    if (remoteBusy) return { mood: "idle", level: 0, line: "Someone else is talking to it, try again soon" };
    return { mood: "idle", level: 0, line: "" };
  }
  if (now - lastModelLoudAt < MODEL_HOLD_MS) return { mood: "speaking", level: modelLevel, line: "" };
  if (muted) return { mood: "idle", level: 0, line: "Microphone muted" };
  if (!captureStream) return { mood: "idle", level: 0, line: startingMic ? "Allow the microphone to start" : "Microphone off" };
  if (now - lastMicLoudAt < MIC_HOLD_MS) return { mood: "user", level: micLevel, line: "Listening" };
  if (session.responseOpen && awaitingAudio) return { mood: "thinking", level: 0, line: "Thinking…" };
  return { mood: "listening", level: micLevel, line: "Listening" };
}

function render(now) {
  const micLevel = muted ? 0 : levelOf(micAnalyser);
  const modelLevel = levelOf(playbackAnalyser);
  if (modelLevel > MODEL_TALK_LEVEL) lastModelLoudAt = now;
  if (micLevel > MIC_TALK_LEVEL) lastMicLoudAt = now;
  const state = mood(micLevel, modelLevel, now);
  if (state.mood === "speaking") lastSpokeAt = now;
  orb.frame(now, state.level, state.mood);
  if (ui.statusLine.textContent !== state.line) ui.statusLine.textContent = state.line;
  ui.caption.dataset.fade = String(!(session && session.responseOpen) && now - lastSpokeAt > CAPTION_FADE_MS);
  requestAnimationFrame(render);
}

function tick() {
  ui.interruptBtn.disabled = mutedResponse || !isOpen() || performance.now() - lastModelLoudAt > INTERRUPT_WINDOW_MS;
  let label;
  let pill;
  if (isOpen() && sessionStart !== null) {
    const left = Math.max(0, SESSION_CAP_S - Math.floor((performance.now() - sessionStart) / 1000));
    label = `${Math.floor(left / 60)}:${String(left % 60).padStart(2, "0")}`;
    pill = "live";
    if (left === 0) {
      sessionStart = null;
      closeNow().catch((error) => warn(error.message));
      warn("Session time limit reached. Start a new conversation to continue.");
    }
  } else if (inCall()) {
    label = "connecting";
    pill = "connecting";
  } else if (serverUp === null) {
    label = "checking…";
    pill = "idle";
  } else if (serverUp === false) {
    label = "offline";
    pill = "error";
  } else if (remoteBusy) {
    label = "busy";
    pill = "busy";
  } else {
    label = "available";
    pill = "idle";
  }
  ui.statusText.textContent = label;
  ui.statusPill.dataset.state = pill;
  ui.hint.textContent = isOpen() && sessionStart !== null ? "Just talk. Interrupt it any time by speaking." : "";
}

async function pollStatus() {
  if (!window.DEMO_STATUS_URL) {
    serverUp = true;
    return;
  }
  try {
    const response = await fetch(window.DEMO_STATUS_URL, { cache: "no-store" });
    const body = await response.json();
    serverUp = Boolean(body.ok);
    remoteBusy = Boolean(body.busy) && !isOpen();
  } catch {
    serverUp = false;
  }
  setButtons();
}

function renderTranscript() {
  const text = session ? session.transcript : "";
  const paragraphs = text.split("\n").filter((line) => line.trim());
  // Audio-only models never send text; the transcript control appears with the first words.
  ui.transcriptBtn.hidden = !paragraphs.length;
  if (!paragraphs.length) {
    ui.transcript.innerHTML = '<p class="empty">What the model says will appear here.</p>';
    ui.caption.textContent = "";
    return;
  }
  ui.transcript.replaceChildren(...paragraphs.map((line) => {
    const paragraph = document.createElement("p");
    paragraph.textContent = line;
    return paragraph;
  }));
  ui.transcript.scrollTop = ui.transcript.scrollHeight;
  // The caption follows the answer being spoken, or the last one once it ends.
  ui.caption.textContent = paragraphs.at(-1);
}

function renderFrames() {
  if (!session) return;
  const { sent, accepted, rejected } = session.frameStats;
  ui.frameCount.textContent = `${accepted}/${sent} frames${rejected ? ` · ${rejected} rejected` : ""}`;
}

function setSheet(open) {
  ui.sheet.dataset.open = String(open);
  ui.transcriptBtn.setAttribute("aria-expanded", String(open));
}

async function startCamera() {
  if (startingCamera || camera.active || !session || !session.imageGranted) return;
  startingCamera = true;
  const activeSession = session;
  setButtons();
  try {
    await camera.start();
    if (session !== activeSession || !isReady()) {
      camera.stop();
      return;
    }
    const params = activeSession.image;
    activeSession.setFrameSource(() => camera.grab(params));
    renderFrames();
  } finally {
    startingCamera = false;
    setButtons();
  }
}

function stopCamera() {
  if (session) session.setFrameSource(null);
  camera.stop();
  setButtons();
}

async function ensurePlayback(rate) {
  if (playbackContext && playbackRate === rate) {
    await playbackContext.resume();
    return;
  }
  if (playbackContext) await playbackContext.close().catch(() => {});
  playbackRate = rate;
  playbackContext = new AudioContext({ sampleRate: rate, latencyHint: "interactive" });
  await playbackContext.audioWorklet.addModule("worklet.js");
  playbackNode = new AudioWorkletNode(playbackContext, "playback-processor", {
    outputChannelCount: [1],
    processorOptions: { initialSamples: Math.max(1, Math.round(playbackContext.sampleRate * JITTER_MS / 1000)) },
  });
  playbackNode.connect(playbackContext.destination);
  // A side tap for the orb; it measures exactly what reaches the speaker.
  playbackAnalyser = playbackContext.createAnalyser();
  playbackAnalyser.fftSize = 2048;
  playbackNode.connect(playbackAnalyser);
  await playbackContext.resume();
}

function rmsOf(samples) {
  let energy = 0;
  for (let index = 0; index < samples.length; index += 1) energy += samples[index] * samples[index];
  return samples.length ? Math.sqrt(energy / samples.length) : 0;
}

// Sub-second streaming units stall on every network hiccup, so they get a deeper adaptive jitter buffer.
const STREAMING_UNIT_MS = 500;
const STREAMING_PLAYBACK = { jitterMs: 400, maxJitterMs: 900, maxMs: 3500, keepMs: 2200, catchUp: { slow: 0.5, fast: 1.0, skip: 2.0, slowRate: 1.06, fastRate: 1.12 } };

function configurePlayback(unitMs) {
  if (!playbackNode || !(unitMs > 0) || unitMs >= STREAMING_UNIT_MS) return;
  const samples = (ms) => Math.round((playbackContext.sampleRate * ms) / 1000);
  const { jitterMs, maxJitterMs, maxMs, keepMs, catchUp } = STREAMING_PLAYBACK;
  playbackNode.port.postMessage({ type: "config", initialSamples: samples(jitterMs), maxJitterSamples: samples(maxJitterMs), maxSamples: samples(maxMs), keepSamples: samples(keepMs), catchUp });
}

function pcm16ToFloat(bytes, sourceRate, targetRate) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const source = new Float32Array(bytes.byteLength >> 1);
  for (let index = 0; index < source.length; index += 1) source[index] = view.getInt16(index * 2, true) / 32768;
  if (targetRate === sourceRate || source.length < 2) return source;
  const target = new Float32Array(Math.max(1, Math.round(source.length * targetRate / sourceRate)));
  const ratio = sourceRate / targetRate;
  for (let index = 0; index < target.length; index += 1) {
    const position = Math.min(index * ratio, source.length - 1);
    const left = Math.floor(position);
    const right = Math.min(left + 1, source.length - 1);
    const weight = position - left;
    target[index] = source[left] * (1 - weight) + source[right] * weight;
  }
  return target;
}

function sessionListeners() {
  return {
    state: (state) => {
      if (state !== "ready") stopCamera();
      if (state === "closed" || state === "error") stopInput();
      setButtons();
    },
    granted: async () => {
      sessionStart = performance.now();
      await ensurePlayback(session.outputRate);
      configurePlayback(session.unitMs);
      setButtons();
      // One tap starts everything: the microphone follows the grant.
      startMic().catch((error) => warn(`Microphone: ${error.message}`, true));
    },
    audio: (bytes) => {
      if (!playbackNode) return;
      awaitingAudio = false;
      const samples = pcm16ToFloat(bytes, session.outputRate, playbackContext.sampleRate);
      // After a local interrupt, drop what the model is still saying. Models that
      // answer in turns lift the mute at the next response.created; models that
      // stream one response for the whole session lift it once they fall quiet.
      if (mutedResponse) {
        mutedQuietMs = rmsOf(samples) < QUIET_RMS ? mutedQuietMs + (samples.length * 1000) / playbackContext.sampleRate : 0;
        if (mutedQuietMs < UNMUTE_QUIET_MS) return;
        mutedResponse = false;
      }
      playbackNode.port.postMessage({ type: "push", samples }, [samples.buffer]);
    },
    response: ({ open }) => {
      if (open) mutedResponse = false;
      awaitingAudio = open;
      setButtons();
      if (playbackNode) playbackNode.port.postMessage({ type: open ? "open" : "flush" });
      // A new answer means whatever is still queued belongs to the previous one.
      if (open && playbackNode) playbackNode.port.postMessage({ type: "trim", keepSamples: Math.round(playbackContext.sampleRate * 0.5) });
      renderTranscript();
    },
    text: renderTranscript,
    unit: (unit) => {
      // The model stopped talking (a unit without audio): what is still queued was
      // generated before that decision, keep at most one unit of it.
      if (unit && unit.decision === "listen" && playbackNode) playbackNode.port.postMessage({ type: "trim", keepSamples: Math.round(playbackContext.sampleRate * 1.0) });
    },
    drained: () => {},
    image: () => setButtons(),
    frame: renderFrames,
    warning: (message) => warn(message),
  };
}

async function connect() {
  if (connecting || isOpen()) return;
  if (!window.DEMO_WS_URL) throw new Error("This page is not configured with a server address.");
  connecting = true;
  await stopInput();
  stopCamera();
  sessionStart = null;
  muted = false;
  session = null;
  renderTranscript();
  warn("");
  if (playbackNode) playbackNode.port.postMessage({ type: "reset" });
  setButtons();
  let activeSocket = null;
  try {
    // Open playback inside the click so Chrome's autoplay policy lets it run.
    await ensurePlayback(playbackRate);
    activeSocket = new WebSocket(wsUrl());
    socket = activeSocket;
    const options = settings.sessionOptions();
    const activeSession = new DuplexSession({
      transport: (event) => activeSocket.send(JSON.stringify(event)),
      now: () => performance.now(),
      outputModalities: options.outputModalities,
      instructions: options.instructions,
      extension: options.extension,
      listeners: sessionListeners(),
    });
    session = activeSession;
    lastSession = activeSession;
    ui.traceBtn.hidden = false;
    eventChain = Promise.resolve();
    activeSocket.addEventListener("open", () => {
      if (socket !== activeSocket) return;
      connecting = false;
      remoteBusy = false;
      setButtons();
    });
    activeSocket.addEventListener("message", (message) => {
      eventChain = eventChain.then(async () => {
        if (socket !== activeSocket) return;
        activeSession.handle(JSON.parse(message.data));
      }).catch((error) => warn(`Server event: ${error.message}`));
    });
    activeSocket.addEventListener("error", () => {
      if (socket !== activeSocket) return;
      warn("Could not reach the demo server. It may be offline; please try again later.", true);
    });
    activeSocket.addEventListener("close", async (event) => {
      if (socket !== activeSocket) return;
      connecting = false;
      socket = null;
      await stopInput();
      stopCamera();
      if (CLOSE_MESSAGES[event.code]) warn(CLOSE_MESSAGES[event.code], true);
      if (event.code === 4429) remoteBusy = true;
      if (playbackNode) playbackNode.port.postMessage({ type: "reset" });
      setButtons();
      pollStatus();
    });
  } catch (error) {
    connecting = false;
    if (activeSocket && activeSocket.readyState < WebSocket.CLOSING) activeSocket.close();
    socket = null;
    warn(error.message, true);
    setButtons();
  }
}

function feedInput(frame) {
  if (!isReady()) return;
  // Muting sends silence rather than stopping, so the unit clock and camera frames keep going.
  if (muted) frame.fill(0);
  session.pushInput(frame);
}

async function startMic() {
  if (startingMic || captureStream || !isReady()) return;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) throw new Error("Microphone capture needs Chrome on an HTTPS page.");
  startingMic = true;
  const generation = micGeneration;
  const activeSocket = socket;
  setButtons();
  let pendingStream = null;
  let pendingContext = null;
  const stale = () => generation !== micGeneration || socket !== activeSocket || !isReady();
  try {
    pendingStream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true, ...settings.micConstraint() }, video: false });
    // Device names are only visible once the microphone permission is granted.
    settings.refreshDevices().catch(() => {});
    if (stale()) throw new Error("the session changed while the microphone permission was pending");
    pendingContext = new AudioContext({ latencyHint: "interactive" });
    await pendingContext.audioWorklet.addModule("worklet.js");
    if (stale()) throw new Error("the session changed while the microphone was starting");
    const source = pendingContext.createMediaStreamSource(pendingStream);
    const pendingNode = new AudioWorkletNode(pendingContext, "capture-processor", {
      processorOptions: { sourceRate: pendingContext.sampleRate, targetRate: session.inputRate, frameSamples: session.packetSamples },
    });
    const silent = pendingContext.createGain();
    silent.gain.value = 0;
    source.connect(pendingNode).connect(silent).connect(pendingContext.destination);
    const analyser = pendingContext.createAnalyser();
    analyser.fftSize = 2048;
    source.connect(analyser);
    await pendingContext.resume();
    if (stale()) throw new Error("the session changed while the microphone was starting");
    captureStream = pendingStream;
    captureContext = pendingContext;
    captureNode = pendingNode;
    micAnalyser = analyser;
    pendingStream = null;
    pendingContext = null;
    pendingNode.port.onmessage = ({ data }) => {
      if (data.type !== "frame" || captureNode !== pendingNode) return;
      feedInput(new Int16Array(data.frame));
    };
  } finally {
    if (pendingStream) pendingStream.getTracks().forEach((track) => track.stop());
    if (pendingContext) await pendingContext.close().catch(() => {});
    startingMic = false;
    setButtons();
  }
}

async function stopInput() {
  micGeneration += 1;
  // Frames ride on the microphone clock; drop any grab still in flight.
  if (session) session.stopFrames();
  if (captureStream) captureStream.getTracks().forEach((track) => track.stop());
  captureStream = null;
  captureNode = null;
  micAnalyser = null;
  if (captureContext) await captureContext.close().catch(() => {});
  captureContext = null;
  setButtons();
}

async function closeNow() {
  await stopInput();
  stopCamera();
  if (playbackNode) playbackNode.port.postMessage({ type: "clear" });
  if (session && isOpen()) session.close();
  else if (socket) socket.close();
  setButtons();
}

function interrupt() {
  mutedResponse = true;
  mutedQuietMs = 0;
  if (playbackNode) playbackNode.port.postMessage({ type: "clear" });
  setButtons();
}

const start = () => connect().catch((error) => warn(error.message, true));
ui.startBtn.addEventListener("click", start);
ui.orb.addEventListener("click", () => {
  if (inCall()) setSheet(!ui.transcriptBtn.hidden && ui.sheet.dataset.open !== "true");
  else if (!ui.startBtn.disabled) start();
});
ui.muteBtn.addEventListener("click", () => {
  muted = !muted;
  setButtons();
});
ui.cameraBtn.addEventListener("click", () => (camera.active ? stopCamera() : startCamera().catch((error) => warn(`Camera: ${error.message}`))));
ui.interruptBtn.addEventListener("click", interrupt);
ui.endBtn.addEventListener("click", () => closeNow().catch((error) => warn(error.message)));
ui.transcriptBtn.addEventListener("click", () => setSheet(ui.sheet.dataset.open !== "true"));
ui.sheetClose.addEventListener("click", () => setSheet(false));
ui.themeBtn.addEventListener("click", () => {
  const next = isDark() ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("theme", next); } catch {}
});
function setPanel(open) {
  ui.settingsPanel.dataset.open = String(open);
  ui.settingsBtn.setAttribute("aria-expanded", String(open));
  if (open) setSheet(false);
}
ui.settingsBtn.addEventListener("click", () => setPanel(ui.settingsPanel.dataset.open !== "true"));
ui.settingsClose.addEventListener("click", () => setPanel(false));
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    setSheet(false);
    setPanel(false);
  }
});
try {
  const saved = localStorage.getItem("theme");
  if (saved) document.documentElement.dataset.theme = saved;
} catch {}
window.addEventListener("beforeunload", () => { if (session && isOpen()) session.close(); });
setInterval(tick, 250);
setInterval(pollStatus, STATUS_POLL_MS);
pollStatus();
setButtons();
tick();
requestAnimationFrame(render);
settings.init().catch((error) => warn(`Settings: ${error.message}`));

ui.traceBtn.addEventListener("click", () => {
  if (!lastSession) return;
  const blob = new Blob([lastSession.exportTrace()], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `duplex-trace-${lastSession.sessionId || "session"}.json`;
  link.click();
  URL.revokeObjectURL(link.href);
});
