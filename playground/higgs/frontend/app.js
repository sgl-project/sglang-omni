// SPDX-License-Identifier: Apache-2.0
// Higgs Audio v3 TTS playground — vanilla JS frontend.

// ---------------------------------------------------------------------------
// Inline control tokens (mirror docs/cookbook/higgs_tts.md tables).
// ---------------------------------------------------------------------------

const TOKEN_CATEGORIES = {
  emotion: [
    ["elation", "Elation / joy"],
    ["amusement", "Amusement / playful laughter"],
    ["enthusiasm", "Enthusiasm / excitement"],
    ["determination", "Determination / firmness"],
    ["pride", "Pride / confidence"],
    ["contentment", "Calm satisfaction"],
    ["affection", "Warmth / affection"],
    ["relief", "Relief"],
    ["contemplation", "Thoughtful / reflective"],
    ["confusion", "Confused"],
    ["surprise", "Surprised"],
    ["awe", "Awe / wonder"],
    ["longing", "Longing / yearning"],
    ["arousal", "Heightened desire"],
    ["anger", "Anger"],
    ["fear", "Fear"],
    ["disgust", "Disgust"],
    ["bitterness", "Bitterness"],
    ["sadness", "Sadness"],
    ["shame", "Shame"],
    ["helplessness", "Helplessness"],
  ],
  style: [
    ["singing", "Singing"],
    ["shouting", "Shouting / projected voice"],
    ["whispering", "Whisper"],
  ],
  sfx: [
    ["cough", "Cough"],
    ["laughter", "Laughter"],
    ["crying", "Crying"],
    ["screaming", "Screaming"],
    ["burping", "Burping"],
    ["humming", "Humming"],
    ["sigh", "Sigh"],
    ["sniff", "Sniff"],
    ["sneeze", "Sneeze"],
  ],
  prosody: [
    ["speed_very_slow", "~0.65× speed"],
    ["speed_slow", "~0.85× speed"],
    ["speed_fast", "~1.2× speed"],
    ["speed_very_fast", "~1.4× speed"],
    ["pitch_low", "~−3 semitones"],
    ["pitch_high", "~+2.5 semitones"],
    ["pause", "~400–700 ms pause"],
    ["long_pause", "~700–1500 ms pause"],
    ["expressive_high", "More expressive delivery"],
    ["expressive_low", "Flatter delivery"],
  ],
};

const CATEGORY_ORDER = ["emotion", "style", "sfx", "prosody"];

// ---------------------------------------------------------------------------
// DOM helpers
// ---------------------------------------------------------------------------

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const textInput = $("#text-input");
const refAudio = $("#ref-audio");
const refAudioUrl = $("#ref-audio-url");
const refText = $("#ref-text");
const temperature = $("#temperature");
const topP = $("#top-p");
const topK = $("#top-k");
const maxNewTokens = $("#max-new-tokens");
const seed = $("#seed");

const synthButton = $("#synth-button");
const synthLabel = synthButton.querySelector(".primary-label");
const streamToggle = $("#stream-toggle");
const statusEl = $("#status");
const statusText = statusEl.querySelector(".status-text");
const finalAudio = $("#final-audio");
const finalAudioActions = $("#final-audio-actions");
const muteButton = $("#mute-button");
const refAudioName = $("#ref-audio-name");
const refAudioClear = $("#ref-audio-clear");
const refAudioRecord = $("#ref-audio-record");
const refAudioRecordLabel = refAudioRecord.querySelector("span");
const refRecordRow = $("#ref-record-row");
const refRecordTimer = $("#ref-record-timer");
const refAudioPreviewWrap = $("#ref-audio-preview-wrap");
const refAudioPreview = $("#ref-audio-preview");
const insertToast = $("#insert-toast");
let insertToastTimer = null;
const themeToggle = $("#theme-toggle");
const envBadge = $("#env-badge");
const envText = envBadge.querySelector(".env-text");

const historyList = $("#history");

// --- Web Audio plumbing for live streaming playback ----
//
// Setting <audio>.src to a new blob URL on every incoming chunk reloads
// the element, which never finishes loading before the next chunk arrives —
// so playback only kicks in once the stream completes. Instead we convert
// each PCM chunk into an AudioBuffer and schedule it on the AudioContext
// timeline.

let audioCtx = null;
let muteGain = null;
let nextStartTime = 0;
let scheduledSources = [];
let playbackGeneration = 0;
let playbackQueue = Promise.resolve();
let muted = false;
let synthesisInFlight = false;

const PCM_DEFAULT_SAMPLE_RATE = 24000;
const PCM_DEFAULT_CHANNELS = 1;
const PCM_DEFAULT_BIT_DEPTH = 16;
const WAV_FORMAT_PCM = 1;

function ensureAudioCtx() {
  if (!audioCtx) {
    const Ctor = window.AudioContext || window.webkitAudioContext;
    audioCtx = new Ctor();
    muteGain = audioCtx.createGain();
    muteGain.gain.value = muted ? 0 : 1;
    muteGain.connect(audioCtx.destination);
  }
  if (audioCtx.state === "suspended") audioCtx.resume();
  return audioCtx;
}

function stopScheduledPlayback() {
  playbackGeneration += 1;
  playbackQueue = Promise.resolve();
  for (const src of scheduledSources) {
    try { src.stop(); } catch {}
    try { src.disconnect(); } catch {}
  }
  scheduledSources = [];
  nextStartTime = audioCtx ? audioCtx.currentTime : 0;
}

function beginPlaybackSession() {
  stopScheduledPlayback();
  return playbackGeneration;
}

function enqueuePcmChunk(pcmBytes, format, generation) {
  playbackQueue = playbackQueue
    .catch(() => {})
    .then(() => schedulePcmChunk(pcmBytes, format, generation));
}

function pcmBytesToAudioBuffer(ctx, pcmBytes, format) {
  const sampleWidth = format.bitsPerSample / 8;
  const frameCount = Math.floor(pcmBytes.length / format.blockAlign);
  const audioBuffer = ctx.createBuffer(
    format.channels,
    frameCount,
    format.sampleRate,
  );
  const view = new DataView(
    pcmBytes.buffer,
    pcmBytes.byteOffset,
    pcmBytes.byteLength,
  );

  for (let frame = 0; frame < frameCount; frame++) {
    const frameOffset = frame * format.blockAlign;
    for (let channel = 0; channel < format.channels; channel++) {
      const offset = frameOffset + channel * sampleWidth;
      audioBuffer.getChannelData(channel)[frame] = view.getInt16(offset, true) / 32768;
    }
  }
  return audioBuffer;
}

async function schedulePcmChunk(pcmBytes, format, generation) {
  if (generation !== playbackGeneration) return;
  const ctx = ensureAudioCtx();
  const audioBuffer = pcmBytesToAudioBuffer(ctx, pcmBytes, format);
  if (generation !== playbackGeneration) return;

  const source = ctx.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(muteGain);
  source.onended = () => {
    scheduledSources = scheduledSources.filter((src) => src !== source);
  };

  const now = ctx.currentTime;
  const startAt = Math.max(nextStartTime, now + 0.02);
  scheduledSources.push(source);
  source.start(startAt);
  nextStartTime = startAt + audioBuffer.duration;
}

muteButton.addEventListener("click", () => {
  muted = !muted;
  if (muteGain) muteGain.gain.value = muted ? 0 : 1;
  muteButton.setAttribute("aria-pressed", muted ? "true" : "false");
  muteButton.title = muted ? "Unmute live playback" : "Mute live playback";
});

// --- per-clip audio actions: 0.5× / 1× / 1.5× + download --
// Exposes the same things the browser's three-dot menu used to hide,
// as visible icons next to the player.
const PLAYBACK_RATES = [0.5, 1, 1.5];

const DOWNLOAD_ICON_SVG = `
<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor"
     stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
  <polyline points="7 10 12 15 17 10"/>
  <line x1="12" y1="15" x2="12" y2="3"/>
</svg>`.trim();

function wireAudioActions(audioEl, actionsEl, downloadName) {
  actionsEl.replaceChildren();
  const speedButtons = [];

  for (const rate of PLAYBACK_RATES) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "audio-action speed" + (rate === 1 ? " active" : "");
    btn.textContent = `${rate}×`;
    btn.title = `Play at ${rate}× speed`;
    btn.setAttribute("aria-label", `Set playback speed to ${rate}×`);
    btn.addEventListener("click", () => {
      audioEl.playbackRate = rate;
      for (const other of speedButtons) other.classList.remove("active");
      btn.classList.add("active");
    });
    speedButtons.push(btn);
    actionsEl.appendChild(btn);
  }
  audioEl.playbackRate = 1;

  const dl = document.createElement("button");
  dl.type = "button";
  dl.className = "audio-action download";
  dl.title = "Download WAV";
  dl.setAttribute("aria-label", "Download WAV");
  dl.innerHTML = DOWNLOAD_ICON_SVG;
  dl.addEventListener("click", () => {
    if (!audioEl.src) return;
    const a = document.createElement("a");
    a.href = audioEl.src;
    a.download = downloadName();
    document.body.appendChild(a);
    a.click();
    a.remove();
  });
  actionsEl.appendChild(dl);
}

function timestampedFilename(prefix = "higgs") {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return (
    `${prefix}-${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}` +
    `-${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}.wav`
  );
}

wireAudioActions(finalAudio, finalAudioActions, () => timestampedFilename("higgs-final"));

// --- theme toggle (light / dark, persisted) -----------
const THEME_KEY = "higgs-playground-theme";
function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  try { localStorage.setItem(THEME_KEY, theme); } catch {}
  themeToggle.title =
    theme === "dark" ? "Switch to light mode" : "Switch to dark mode";
}
(function bootTheme() {
  let saved;
  try { saved = localStorage.getItem(THEME_KEY); } catch {}
  if (!saved) {
    saved = window.matchMedia &&
            window.matchMedia("(prefers-color-scheme: light)").matches
      ? "light" : "dark";
  }
  applyTheme(saved);
})();
themeToggle.addEventListener("click", () => {
  const current = document.documentElement.getAttribute("data-theme");
  applyTheme(current === "dark" ? "light" : "dark");
});

// --- backend health check (one-shot on load) ----------
(async function checkBackend() {
  try {
    const resp = await fetch("/healthz", { cache: "no-store" });
    if (!resp.ok) throw new Error();
    const data = await resp.json();
    if (data.backend === "ok") {
      envBadge.classList.add("ok");
      envText.textContent = "backend · ready";
    } else {
      envBadge.classList.add("down");
      envText.textContent = "backend · down";
    }
  } catch {
    envBadge.classList.add("down");
    envText.textContent = "backend · n/a";
  }
})();

// --- reference audio: upload, record, preview ------------
let recordedRefFile = null;
let recordedRefUrl = null;
let micRecorder = null;
let micStream = null;
let micChunks = [];
let recordStartedAt = null;
let recordTimerInterval = null;

function clearRecordedRefPreviewUrl() {
  if (recordedRefUrl) {
    URL.revokeObjectURL(recordedRefUrl);
    recordedRefUrl = null;
  }
}

function stopMicStream() {
  if (micStream) {
    micStream.getTracks().forEach((track) => track.stop());
    micStream = null;
  }
}

function clearRecordTimer() {
  if (recordTimerInterval) {
    clearInterval(recordTimerInterval);
    recordTimerInterval = null;
  }
  recordStartedAt = null;
  refRecordTimer.textContent = "0:00";
}

function formatRecordElapsed(ms) {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

function updateRefAudioLabel() {
  const uploaded = refAudio.files && refAudio.files[0];
  if (uploaded) {
    refAudioName.textContent = uploaded.name;
    refAudioName.classList.add("has-file");
    return;
  }
  if (recordedRefFile) {
    refAudioName.textContent = recordedRefFile.name;
    refAudioName.classList.add("has-file");
    return;
  }
  refAudioName.textContent = "No reference selected";
  refAudioName.classList.remove("has-file");
}

function clearRecordedRef() {
  recordedRefFile = null;
  clearRecordedRefPreviewUrl();
  refAudioPreview.pause();
  refAudioPreview.removeAttribute("src");
  refAudioPreview.load();
  refAudioPreviewWrap.classList.add("hidden");
}

function clearUploadedRef() {
  refAudio.value = "";
}

function clearReferenceAudio() {
  clearUploadedRef();
  clearRecordedRef();
  updateRefAudioLabel();
}

function setRecordingUi(isRecording) {
  refRecordRow.classList.toggle("hidden", !isRecording);
  refAudioRecord.classList.toggle("recording", isRecording);
  refAudioRecord.disabled = synthesisInFlight;
  refAudioRecord.setAttribute("aria-pressed", isRecording ? "true" : "false");
  refAudioRecord.title = isRecording
    ? "Stop recording"
    : "Record reference audio from microphone";
  if (refAudioRecordLabel) {
    refAudioRecordLabel.textContent = isRecording ? "Stop" : "Record";
  }
}

async function startReferenceRecording() {
  if (synthesisInFlight) return;
  if (micRecorder && micRecorder.state === "recording") return;

  clearUploadedRef();
  clearRecordedRef();

  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    micChunks = [];
    micRecorder = new MediaRecorder(micStream);
    micRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size) micChunks.push(event.data);
    };
    micRecorder.onstop = () => {
      stopMicStream();
      clearRecordTimer();
      setRecordingUi(false);

      const blob = new Blob(micChunks, { type: "audio/webm" });
      micChunks = [];
      if (!blob.size) {
        updateRefAudioLabel();
        return;
      }

      recordedRefFile = new File(
        [blob],
        `reference-${Date.now()}.webm`,
        { type: "audio/webm" },
      );
      clearRecordedRefPreviewUrl();
      recordedRefUrl = URL.createObjectURL(blob);
      refAudioPreview.src = recordedRefUrl;
      refAudioPreviewWrap.classList.remove("hidden");
      updateRefAudioLabel();
    };

    micRecorder.start();
    recordStartedAt = performance.now();
    refRecordTimer.textContent = "0:00";
    recordTimerInterval = setInterval(() => {
      if (recordStartedAt === null) return;
      refRecordTimer.textContent = formatRecordElapsed(
        performance.now() - recordStartedAt,
      );
    }, 250);
    setRecordingUi(true);
  } catch (error) {
    stopMicStream();
    clearRecordTimer();
    setRecordingUi(false);
    setStatus(
      `Microphone access failed: ${error.message || "permission denied"}`,
      "error",
    );
  }
}

function stopReferenceRecording() {
  if (micRecorder && micRecorder.state === "recording") {
    micRecorder.stop();
  }
}

refAudio.addEventListener("change", () => {
  const file = refAudio.files && refAudio.files[0];
  if (file) {
    clearRecordedRef();
  }
  updateRefAudioLabel();
});

refAudioClear.addEventListener("click", () => {
  if (micRecorder && micRecorder.state === "recording") {
    stopReferenceRecording();
  }
  clearReferenceAudio();
});

refAudioRecord.addEventListener("click", () => {
  if (micRecorder && micRecorder.state === "recording") {
    stopReferenceRecording();
    return;
  }
  startReferenceRecording();
});

// ---------------------------------------------------------------------------
// Token picker
// ---------------------------------------------------------------------------

function renderTokenTabs() {
  const tabsContainer = $("#token-tabs");
  tabsContainer.innerHTML = "";
  CATEGORY_ORDER.forEach((category, i) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "token-tab" + (i === 0 ? " active" : "");
    btn.dataset.category = category;
    btn.textContent = category;
    btn.addEventListener("click", () => {
      $$(".token-tab").forEach((t) => t.classList.remove("active"));
      btn.classList.add("active");
      renderTokenGrid(category);
    });
    tabsContainer.appendChild(btn);
  });
}

function renderTokenGrid(category) {
  const grid = $("#token-grid");
  grid.replaceChildren();
  for (const [name, desc] of TOKEN_CATEGORIES[category]) {
    const literal = `<|${category}:${name}|>`;

    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "token-chip";
    chip.dataset.category = category;
    chip.title = `Insert ${literal}`;

    // textContent (not innerHTML) — the literal contains "<|…|>" which the
    // HTML parser treats as a malformed tag and silently drops, leaving
    // chips blank and unclickable. Use DOM construction to keep them text.
    const nameSpan = document.createElement("span");
    nameSpan.className = "token-name";
    nameSpan.textContent = literal;

    const descSpan = document.createElement("span");
    descSpan.className = "token-desc";
    descSpan.textContent = desc;

    chip.append(nameSpan, descSpan);
    chip.addEventListener("click", () => {
      insertTokenAtCursor(literal);
      // Two visible confirmations so the click never goes unnoticed:
      //   1. the chip itself flashes green and scales up briefly
      //   2. a transient toast appears next to the textarea label
      chip.classList.add("flash");
      setTimeout(() => chip.classList.remove("flash"), 420);
      showInsertToast(literal);
    });
    grid.appendChild(chip);
  }
}

function showInsertToast(literal) {
  insertToast.textContent = `✓ inserted ${literal}`;
  insertToast.classList.add("show");
  if (insertToastTimer) clearTimeout(insertToastTimer);
  insertToastTimer = setTimeout(() => {
    insertToast.classList.remove("show");
    insertToastTimer = null;
  }, 1400);
}

function insertTokenAtCursor(token) {
  const start = textInput.selectionStart ?? textInput.value.length;
  const end = textInput.selectionEnd ?? textInput.value.length;
  const before = textInput.value.slice(0, start);
  const after = textInput.value.slice(end);

  // Add a leading space if there isn't whitespace already before the cursor,
  // and a trailing space so the next typed character doesn't merge into the
  // token visually. Avoid double-spacing.
  const leading = before && !/\s$/.test(before) ? " " : "";
  const trailing = after && !/^\s/.test(after) ? " " : "";
  const insert = `${leading}${token}${trailing}`;

  textInput.value = before + insert + after;
  const cursor = (before + insert).length;
  textInput.focus();
  textInput.setSelectionRange(cursor, cursor);
}

// ---------------------------------------------------------------------------
// Form helpers
// ---------------------------------------------------------------------------

function resolveReferenceAudioFile() {
  if (refAudio.files && refAudio.files[0]) {
    return refAudio.files[0];
  }
  return recordedRefFile;
}

function buildFormData() {
  const fd = new FormData();
  fd.append("text", textInput.value);
  const refFile = resolveReferenceAudioFile();
  if (refFile) {
    fd.append("ref_audio", refFile);
  }
  fd.append("ref_audio_url", refAudioUrl.value || "");
  fd.append("ref_text", refText.value || "");
  fd.append("temperature", temperature.value || "");
  fd.append("top_p", topP.value || "");
  fd.append("top_k", topK.value || "");
  fd.append("max_new_tokens", maxNewTokens.value || "");
  fd.append("seed", seed.value || "");
  return fd;
}

function setStatus(text, kind = "") {
  statusText.textContent = text;
  statusEl.classList.remove("success", "error", "busy");
  if (kind) statusEl.classList.add(kind);
}

function lockButton(busy) {
  synthButton.disabled = busy;
  synthLabel.textContent = busy ? "Synthesizing…" : "Synthesize";
  streamToggle.disabled = busy;
  refAudioRecord.disabled = busy;
}

// ---------------------------------------------------------------------------
// History
// ---------------------------------------------------------------------------

function renderHistoryEmpty() {
  historyList.innerHTML =
    '<li class="history-empty">No synthesis yet — your generated clips will appear here.</li>';
}

function appendHistory({ text, audioUrl, meta }) {
  // Drop the empty-state placeholder on first append.
  const placeholder = historyList.querySelector(".history-empty");
  if (placeholder) placeholder.remove();

  const li = document.createElement("li");
  li.className = "history-item";

  const textDiv = document.createElement("div");
  textDiv.className = "history-text";
  textDiv.textContent = text;
  li.appendChild(textDiv);

  if (audioUrl) {
    const row = document.createElement("div");
    row.className = "audio-row";

    const audio = document.createElement("audio");
    audio.controls = true;
    audio.setAttribute(
      "controlslist",
      "nodownload noplaybackrate noremoteplayback",
    );
    audio.src = audioUrl;

    const actions = document.createElement("div");
    actions.className = "audio-actions";

    row.append(audio, actions);
    li.appendChild(row);

    wireAudioActions(audio, actions, () => timestampedFilename("higgs"));
  }

  if (meta) {
    const metaDiv = document.createElement("div");
    metaDiv.className = "history-meta";
    metaDiv.textContent = meta;
    li.appendChild(metaDiv);
  }

  historyList.prepend(li);
}

$("#clear-history").addEventListener("click", () => {
  renderHistoryEmpty();
});

// ---------------------------------------------------------------------------
// Synthesize — single button, streaming controlled by the toggle.
// ---------------------------------------------------------------------------

synthButton.addEventListener("click", async () => {
  if (synthesisInFlight) return;
  if (!textInput.value.trim()) {
    setStatus("Please enter some text to synthesize.", "error");
    return;
  }
  synthesisInFlight = true;
  finalAudio.pause();
  finalAudio.removeAttribute("src");
  finalAudio.load();
  try {
    if (streamToggle.checked) {
      await runStreaming();
    } else {
      await runNonStreaming();
    }
  } finally {
    synthesisInFlight = false;
  }
});

async function runNonStreaming() {
  const inputText = textInput.value;
  stopScheduledPlayback();
  lockButton(true);
  setStatus("Submitting request…", "busy");

  const started = performance.now();
  try {
    const resp = await fetch("/api/synthesize", {
      method: "POST",
      body: buildFormData(),
    });
    if (!resp.ok) {
      const err = await resp.text();
      throw new Error(`HTTP ${resp.status}: ${err}`);
    }
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    finalAudio.src = url;
    finalAudio.pause();
    const elapsed = ((performance.now() - started) / 1000).toFixed(2);
    const sizeKb = (blob.size / 1024).toFixed(0);
    const meta = `${elapsed}s total | ${sizeKb} KB`;
    setStatus(meta, "success");
    appendHistory({ text: inputText, audioUrl: url, meta });
  } catch (exc) {
    setStatus(`Request failed: ${exc.message}`, "error");
  } finally {
    lockButton(false);
  }
}

async function runStreaming() {
  const inputText = textInput.value;
  lockButton(true);
  setStatus("Connecting to speech stream…", "busy");

  // Resume / create the AudioContext synchronously within the user gesture
  // chain so browsers don't refuse to start playback.
  ensureAudioCtx();
  const playbackRun = beginPlaybackSession();

  const started = performance.now();
  let chunkCount = 0;
  let firstAudioAt = null;
  const pcmChunks = [];
  let pendingBytes = new Uint8Array(0);

  try {
    const resp = await fetch("/api/synthesize/stream", {
      method: "POST",
      body: buildFormData(),
    });
    if (!resp.ok) {
      const err = await resp.text();
      throw new Error(`HTTP ${resp.status}: ${err}`);
    }

    const reader = resp.body.getReader();
    const streamFormat = pcmFormatFromHeaders(resp.headers);

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      const aligned = alignPcmChunk(
        value,
        pendingBytes,
        streamFormat.blockAlign,
      );
      pendingBytes = aligned.pending;
      if (aligned.chunk.length === 0) {
        continue;
      }

      chunkCount += 1;
      pcmChunks.push(aligned.chunk);

      if (firstAudioAt === null) {
        firstAudioAt = (performance.now() - started) / 1000;
      }

      enqueuePcmChunk(aligned.chunk, streamFormat, playbackRun);

      setStatus(
        `Streaming · chunk ${chunkCount} · first audio ${firstAudioAt.toFixed(2)}s`,
        "busy",
      );
    }

    if (pendingBytes.length !== 0) {
      throw new Error("PCM stream ended with a partial audio frame.");
    }
    if (pcmChunks.length === 0) {
      throw new Error("No audio was returned.");
    }

    // Provide the full combined WAV in the Final audio bar so the user can
    // scrub / re-listen / download. The live playback is already happening
    // in the Web Audio graph and continues independently.
    const finalBytes = writeWav(
      streamFormat,
      pcmChunks,
      pcmChunks.reduce((total, chunk) => total + chunk.length, 0),
    );
    const finalUrl = URL.createObjectURL(
      new Blob([finalBytes], { type: "audio/wav" }),
    );
    finalAudio.src = finalUrl;

    const elapsed = ((performance.now() - started) / 1000).toFixed(2);
    const ftf =
      firstAudioAt !== null ? ` · first audio ${firstAudioAt.toFixed(2)}s` : "";
    const meta = `${elapsed}s total · ${chunkCount} chunks${ftf}`;
    setStatus(meta, "success");
    appendHistory({ text: inputText, audioUrl: finalUrl, meta });
  } catch (exc) {
    stopScheduledPlayback();
    setStatus(`Request failed: ${exc.message}`, "error");
  } finally {
    lockButton(false);
    // Indicator is driven by the scheduler's own timer — don't force it off
    // here, the still-queued audio should keep the bars vibing until done.
  }
}

// ---------------------------------------------------------------------------
// PCM / WAV utilities (browser-side)
// ---------------------------------------------------------------------------

function pcmFormatFromHeaders(headers) {
  const sampleRate = Number(headers.get("x-sample-rate")) || PCM_DEFAULT_SAMPLE_RATE;
  const channels = Number(headers.get("x-channels")) || PCM_DEFAULT_CHANNELS;
  const bitsPerSample =
    Number(headers.get("x-bit-depth")) || PCM_DEFAULT_BIT_DEPTH;
  const sampleWidth = bitsPerSample / 8;
  return {
    audioFormat: WAV_FORMAT_PCM,
    channels,
    sampleRate,
    byteRate: sampleRate * channels * sampleWidth,
    blockAlign: channels * sampleWidth,
    bitsPerSample,
  };
}

function alignPcmChunk(chunk, pendingBytes, blockAlign) {
  let merged = chunk;
  if (pendingBytes.length !== 0) {
    merged = new Uint8Array(pendingBytes.length + chunk.length);
    merged.set(pendingBytes, 0);
    merged.set(chunk, pendingBytes.length);
  }

  const alignedLength = merged.length - (merged.length % blockAlign);
  return {
    chunk: merged.subarray(0, alignedLength),
    pending: merged.subarray(alignedLength),
  };
}

function writeWav(fmt, pcms, total) {
  const header = 44;
  const out = new Uint8Array(header + total);
  const view = new DataView(out.buffer);

  // RIFF
  out[0] = 0x52; out[1] = 0x49; out[2] = 0x46; out[3] = 0x46;
  view.setUint32(4, 36 + total, true);
  out[8] = 0x57; out[9] = 0x41; out[10] = 0x56; out[11] = 0x45;

  // fmt
  out[12] = 0x66; out[13] = 0x6d; out[14] = 0x74; out[15] = 0x20;
  view.setUint32(16, 16, true);
  view.setUint16(20, fmt.audioFormat, true);
  view.setUint16(22, fmt.channels, true);
  view.setUint32(24, fmt.sampleRate, true);
  view.setUint32(28, fmt.byteRate, true);
  view.setUint16(32, fmt.blockAlign, true);
  view.setUint16(34, fmt.bitsPerSample, true);

  // data
  out[36] = 0x64; out[37] = 0x61; out[38] = 0x74; out[39] = 0x61;
  view.setUint32(40, total, true);

  let offset = header;
  for (const pcm of pcms) {
    out.set(pcm, offset);
    offset += pcm.length;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

renderTokenTabs();
renderTokenGrid(CATEGORY_ORDER[0]);
