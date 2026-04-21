(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const connectBtn = $("connect");
  const disconnectBtn = $("disconnect");
  const pushToTalkBtn = $("push-to-talk");
  const clearLogBtn = $("clear-log");
  const statusEl = $("status");
  const logEl = $("log");
  const conversationEl = $("conversation");
  const instructionsEl = $("instructions");
  const userPromptEl = $("user-prompt");
  const sendTextBtn = $("send-text");
  const audioModeAutoEl = $("audio-mode-auto");
  const audioModePushEl = $("audio-mode-push");
  const audioModeHelpEl = $("audio-mode-help");
  const micLevelFillEl = $("mic-level-fill");
  const micLevelTextEl = $("mic-level-text");
  const remoteLevelFillEl = $("remote-level-fill");
  const remoteLevelTextEl = $("remote-level-text");

  let ws = null;
  let localStream = null;
  let sessionId = null;
  let inputAudioMode = "vad";
  let pushToTalkActive = false;
  let pushToTalkKeyActive = false;
  let assistantMessages = new Map();

  let captureAudioContext = null;
  let captureSource = null;
  let captureProcessor = null;
  let captureSink = null;
  let captureSampleRate = 16000;

  let playbackAudioContext = null;
  let playbackCursor = 0;
  let playbackGeneration = 0;
  let playbackSources = new Set();
  let outputSampleRate = 24000;
  let remoteLevel = 0;
  let remoteLevelRaf = 0;

  function getApiBase() {
    if (
      typeof window !== "undefined" &&
      Object.prototype.hasOwnProperty.call(window, "SGLANG_OMNI_API_BASE")
    ) {
      return String(window.SGLANG_OMNI_API_BASE || "").trim().replace(/\/$/, "");
    }
    return window.location.origin;
  }

  function buildWebSocketUrl() {
    const base = new URL(getApiBase(), window.location.href);
    base.protocol = base.protocol === "https:" ? "wss:" : "ws:";
    base.pathname = "/v1/realtime/ws";
    base.search = "";
    return base.toString();
  }

  function setStatus(text) {
    statusEl.textContent = text;
  }

  function log(message, payload) {
    const stamp = new Date().toLocaleTimeString();
    const line = `[${stamp}] ${message}`;
    const body = payload ? `\n${JSON.stringify(payload, null, 2)}` : "";
    logEl.textContent += `${line}${body}\n\n`;
    logEl.scrollTop = logEl.scrollHeight;
  }

  function scrollConversationToBottom() {
    conversationEl.scrollTop = conversationEl.scrollHeight;
  }

  function addConversationEmptyState() {
    if (conversationEl.querySelector(".conversation-empty")) {
      return;
    }
    const empty = document.createElement("div");
    empty.className = "conversation-empty";
    empty.textContent = "Conversation history will appear here.";
    conversationEl.appendChild(empty);
  }

  function clearConversation() {
    assistantMessages = new Map();
    conversationEl.textContent = "";
    addConversationEmptyState();
  }

  function createConversationMessage(role, text, pending) {
    const container = document.createElement("div");
    container.className = `message message-${role}${pending ? " pending" : ""}`;

    const roleEl = document.createElement("div");
    roleEl.className = "message-role";
    roleEl.textContent = role === "user" ? "User" : "Assistant";

    const contentEl = document.createElement("div");
    contentEl.className = "message-content";
    contentEl.textContent = text;

    container.appendChild(roleEl);
    container.appendChild(contentEl);
    return { container, contentEl };
  }

  function appendConversationMessage(role, text, pending = false) {
    const emptyState = conversationEl.querySelector(".conversation-empty");
    if (emptyState) {
      emptyState.remove();
    }
    const message = createConversationMessage(role, text, pending);
    conversationEl.appendChild(message.container);
    scrollConversationToBottom();
    return message;
  }

  function ensureAssistantMessage(responseId) {
    if (!responseId) {
      return null;
    }
    const existing = assistantMessages.get(responseId);
    if (existing) {
      return existing;
    }
    const message = appendConversationMessage("assistant", "", true);
    const entry = {
      container: message.container,
      contentEl: message.contentEl,
      rawText: "",
      hasVisibleText: false,
      hasAudio: false,
    };
    assistantMessages.set(responseId, entry);
    return entry;
  }

  function appendAssistantDelta(responseId, delta) {
    if (!responseId || typeof delta !== "string" || delta.length === 0) {
      return;
    }
    const entry = ensureAssistantMessage(responseId);
    if (!entry) {
      return;
    }
    entry.rawText += delta;
    entry.contentEl.textContent = entry.rawText;
    if (entry.rawText.trim().length > 0) {
      entry.hasVisibleText = true;
    }
    scrollConversationToBottom();
  }

  function noteAssistantAudio(responseId) {
    if (!responseId) {
      return;
    }
    const entry = ensureAssistantMessage(responseId);
    if (!entry) {
      return;
    }
    entry.hasAudio = true;
    if (!entry.hasVisibleText) {
      entry.contentEl.textContent = "(streaming audio)";
    }
    scrollConversationToBottom();
  }

  function finalizeAssistantMessage(responseId, text) {
    if (!responseId) {
      return;
    }
    const entry = ensureAssistantMessage(responseId);
    if (!entry) {
      return;
    }
    if (typeof text === "string" && text.length > 0) {
      entry.rawText = text;
      entry.contentEl.textContent = text;
      entry.hasVisibleText = text.trim().length > 0;
    }
    if (!entry.hasVisibleText) {
      entry.contentEl.textContent = entry.hasAudio
        ? "(audio response)"
        : "(no text output)";
    }
    entry.container.classList.remove("pending");
    scrollConversationToBottom();
  }

  function updateMicLevel(level) {
    const clamped = Math.max(0, Math.min(1, level));
    const percent = Math.round(clamped * 100);
    micLevelFillEl.style.width = `${percent}%`;
    micLevelTextEl.textContent = `${percent}%`;
  }

  function updateRemoteLevel(level) {
    const clamped = Math.max(0, Math.min(1, level));
    const percent = Math.round(clamped * 100);
    remoteLevelFillEl.style.width = `${percent}%`;
    remoteLevelTextEl.textContent = `${percent}%`;
  }

  function tickRemoteLevelDecay() {
    remoteLevel = Math.max(remoteLevel * 0.92, 0);
    updateRemoteLevel(remoteLevel);
    if (remoteLevel <= 0.01) {
      remoteLevel = 0;
      updateRemoteLevel(0);
      remoteLevelRaf = 0;
      return;
    }
    remoteLevelRaf = requestAnimationFrame(tickRemoteLevelDecay);
  }

  function bumpRemoteLevel(level) {
    remoteLevel = Math.max(remoteLevel, Math.max(0, Math.min(1, level)));
    updateRemoteLevel(remoteLevel);
    if (!remoteLevelRaf) {
      remoteLevelRaf = requestAnimationFrame(tickRemoteLevelDecay);
    }
  }

  function clearPlaybackQueue() {
    playbackGeneration += 1;
    playbackCursor = 0;
    playbackSources.forEach((source) => {
      try {
        source.stop();
      } catch (_) {
        // Ignore already-ended nodes.
      }
    });
    playbackSources.clear();
    remoteLevel = 0;
    updateRemoteLevel(0);
  }

  function computeRms(samples) {
    if (!samples || samples.length === 0) {
      return 0;
    }
    let sum = 0;
    for (let i = 0; i < samples.length; i += 1) {
      const sample = samples[i];
      sum += sample * sample;
    }
    return Math.sqrt(sum / samples.length);
  }

  function float32ToPcm16Bytes(samples) {
    const pcm = new Int16Array(samples.length);
    for (let i = 0; i < samples.length; i += 1) {
      const sample = Math.max(-1, Math.min(1, samples[i]));
      pcm[i] = sample < 0 ? Math.round(sample * 32768) : Math.round(sample * 32767);
    }
    return pcm.buffer;
  }

  function pcm16ToFloat32(buffer) {
    const int16 = new Int16Array(buffer);
    const out = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i += 1) {
      out[i] = int16[i] / 32768;
    }
    return out;
  }

  async function ensurePlaybackContext() {
    if (!playbackAudioContext) {
      const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
      if (!AudioContextCtor) {
        throw new Error("Web Audio API is not supported");
      }
      playbackAudioContext = new AudioContextCtor();
    }
    if (playbackAudioContext.state === "suspended") {
      await playbackAudioContext.resume();
    }
    return playbackAudioContext;
  }

  async function playAssistantAudioChunk(buffer) {
    if (!(buffer instanceof ArrayBuffer) || buffer.byteLength === 0) {
      return;
    }
    const audioContext = await ensurePlaybackContext();
    const samples = pcm16ToFloat32(buffer);
    if (samples.length === 0) {
      return;
    }

    const audioBuffer = audioContext.createBuffer(1, samples.length, outputSampleRate);
    audioBuffer.copyToChannel(samples, 0);

    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContext.destination);

    const generation = playbackGeneration;
    const startAt = Math.max(audioContext.currentTime + 0.04, playbackCursor || 0);
    source.start(startAt);
    playbackCursor = startAt + audioBuffer.duration;
    playbackSources.add(source);
    source.onended = () => {
      playbackSources.delete(source);
      if (generation !== playbackGeneration) {
        return;
      }
      if (playbackSources.size === 0 && playbackAudioContext) {
        playbackCursor = Math.max(playbackAudioContext.currentTime, 0);
      }
    };

    bumpRemoteLevel(Math.min(computeRms(samples) * 4.0, 1.0));
  }

  function canSendControlEvent() {
    return Boolean(ws && ws.readyState === WebSocket.OPEN);
  }

  function sendControlEvent(payload) {
    if (!canSendControlEvent()) {
      log("control event skipped", {
        type: payload && payload.type ? payload.type : "unknown",
        reason: "websocket not open",
      });
      return false;
    }
    ws.send(JSON.stringify(payload));
    return true;
  }

  function updatePushToTalkUi() {
    const manualMode = inputAudioMode === "manual";
    pushToTalkBtn.disabled = !(manualMode && canSendControlEvent());
    pushToTalkBtn.classList.toggle("hidden", !manualMode);
    pushToTalkBtn.classList.toggle("active", pushToTalkActive);
    pushToTalkBtn.textContent = pushToTalkActive
      ? "Release To Commit"
      : "Hold To Talk";
  }

  function updateTextPromptUi() {
    const connected = canSendControlEvent();
    userPromptEl.disabled = !connected;
    sendTextBtn.disabled = !(connected && userPromptEl.value.trim());
  }

  function updateAudioModeHelp() {
    if (inputAudioMode === "manual") {
      audioModeHelpEl.textContent =
        "Push To Talk captures continuously but only commits audio while you hold the button or space bar.";
      return;
    }
    audioModeHelpEl.textContent =
      "Auto VAD streams microphone PCM continuously and lets the server detect utterance boundaries.";
  }

  function setInputAudioMode(mode) {
    inputAudioMode = mode === "manual" ? "manual" : "vad";
    audioModeAutoEl.checked = inputAudioMode === "vad";
    audioModePushEl.checked = inputAudioMode === "manual";
    if (inputAudioMode !== "manual") {
      pushToTalkActive = false;
      pushToTalkKeyActive = false;
    }
    updateAudioModeHelp();
    updatePushToTalkUi();
  }

  function sendSessionUpdate() {
    const session = {
      instructions: instructionsEl.value.trim(),
      audio: {
        input_mode: inputAudioMode,
      },
    };
    return sendControlEvent({
      type: "session.update",
      session,
    });
  }

  function sendInputAudioFormat() {
    return sendControlEvent({
      type: "input_audio_format",
      sample_rate: captureSampleRate,
      encoding: "pcm16le",
    });
  }

  async function startAudioCapture(stream) {
    stopAudioCapture();

    const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextCtor) {
      throw new Error("Web Audio API is not supported");
    }

    captureAudioContext = new AudioContextCtor({ sampleRate: 16000 });
    if (captureAudioContext.state === "suspended") {
      await captureAudioContext.resume();
    }
    captureSampleRate = captureAudioContext.sampleRate;

    captureSource = captureAudioContext.createMediaStreamSource(stream);
    captureProcessor = captureAudioContext.createScriptProcessor(2048, 1, 1);
    captureSink = captureAudioContext.createGain();
    captureSink.gain.value = 0;

    captureProcessor.onaudioprocess = (event) => {
      const input = event.inputBuffer.getChannelData(0);
      const copy = new Float32Array(input.length);
      copy.set(input);
      updateMicLevel(Math.min(computeRms(copy) * 4.0, 1.0));
      if (!canSendControlEvent()) {
        return;
      }
      ws.send(float32ToPcm16Bytes(copy));
    };

    captureSource.connect(captureProcessor);
    captureProcessor.connect(captureSink);
    captureSink.connect(captureAudioContext.destination);
  }

  function stopAudioCapture() {
    if (captureProcessor) {
      try {
        captureProcessor.disconnect();
      } catch (_) {
        // Ignore node teardown errors.
      }
      captureProcessor.onaudioprocess = null;
      captureProcessor = null;
    }
    if (captureSink) {
      try {
        captureSink.disconnect();
      } catch (_) {
        // Ignore node teardown errors.
      }
      captureSink = null;
    }
    if (captureSource) {
      try {
        captureSource.disconnect();
      } catch (_) {
        // Ignore node teardown errors.
      }
      captureSource = null;
    }
    if (captureAudioContext) {
      captureAudioContext.close().catch(() => {});
      captureAudioContext = null;
    }
    updateMicLevel(0);
  }

  function handleServerEvent(event) {
    if (!event || typeof event.type !== "string") {
      return;
    }

    if (event.type === "session.created") {
      sessionId = event.session_id || null;
      outputSampleRate =
        (event.audio && Number(event.audio.output_sample_rate)) || outputSampleRate;
      clearConversation();
      setStatus(sessionId ? `Connected (${sessionId})` : "Connected");
      disconnectBtn.disabled = false;
      updatePushToTalkUi();
      updateTextPromptUi();
      return;
    }

    if (event.type === "conversation.item.created") {
      const item = event.item || {};
      if (item.role === "user" && typeof item.content === "string") {
        appendConversationMessage("user", item.content);
      }
      return;
    }

    if (event.type === "response.created") {
      ensureAssistantMessage(event.response_id);
      return;
    }

    if (event.type === "response.output_text.delta") {
      appendAssistantDelta(event.response_id, event.delta);
      return;
    }

    if (event.type === "response.done") {
      finalizeAssistantMessage(event.response_id, event.text);
      return;
    }

    if (event.type === "response.output_audio.delta") {
      noteAssistantAudio(event.response_id);
      return;
    }

    if (event.type === "response.cancelled") {
      clearPlaybackQueue();
      const entry = assistantMessages.get(event.response_id);
      if (entry) {
        entry.container.classList.remove("pending");
      }
      return;
    }

    if (event.type === "session.updated") {
      const session = event.session || {};
      const audio = session.audio || {};
      if (typeof audio.input_mode === "string") {
        setInputAudioMode(audio.input_mode);
      }
      return;
    }

    if (event.type === "output_audio_buffer.cleared") {
      clearPlaybackQueue();
      return;
    }
  }

  function beginPushToTalk(source) {
    if (inputAudioMode !== "manual" || pushToTalkActive) {
      return;
    }
    if (!sendControlEvent({ type: "input_audio_buffer.start" })) {
      updatePushToTalkUi();
      return;
    }
    pushToTalkActive = true;
    updatePushToTalkUi();
    log("push-to-talk started", { source });
  }

  function commitPushToTalk(source) {
    if (inputAudioMode !== "manual" || !pushToTalkActive) {
      return;
    }
    pushToTalkActive = false;
    updatePushToTalkUi();
    if (!sendControlEvent({ type: "input_audio_buffer.commit" })) {
      return;
    }
    log("push-to-talk committed", { source });
  }

  function submitTextPrompt() {
    const text = userPromptEl.value.trim();
    if (!text) {
      updateTextPromptUi();
      return;
    }
    if (
      !sendControlEvent({
        type: "conversation.item.create",
        item: { role: "user", content: text },
      })
    ) {
      updateTextPromptUi();
      return;
    }
    if (!sendControlEvent({ type: "response.create" })) {
      updateTextPromptUi();
      return;
    }
    userPromptEl.value = "";
    updateTextPromptUi();
    log("text prompt sent", { chars: text.length });
  }

  async function connect() {
    if (ws) {
      return;
    }
    connectBtn.disabled = true;
    setStatus("Requesting microphone...");

    localStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        channelCount: 1,
      },
      video: false,
    });
    await startAudioCapture(localStream);
    await ensurePlaybackContext();

    const socket = new WebSocket(buildWebSocketUrl());
    socket.binaryType = "arraybuffer";
    ws = socket;

    socket.addEventListener("open", () => {
      sendInputAudioFormat();
      sendSessionUpdate();
      setStatus("Connected");
      disconnectBtn.disabled = false;
      updatePushToTalkUi();
      updateTextPromptUi();
      log("websocket opened", {
        url: buildWebSocketUrl(),
        capture_sample_rate: captureSampleRate,
      });
    });

    socket.addEventListener("message", (event) => {
      if (typeof event.data === "string") {
        try {
          const payload = JSON.parse(event.data);
          handleServerEvent(payload);
          log("server event", payload);
        } catch (err) {
          log("server message", { raw: event.data });
        }
        return;
      }
      playAssistantAudioChunk(event.data).catch((err) => {
        log("audio playback error", { message: String(err) });
      });
    });

    socket.addEventListener("error", () => {
      log("websocket error", { url: buildWebSocketUrl() });
    });

    socket.addEventListener("close", (event) => {
      if (ws === socket) {
        log("websocket closed", {
          code: event.code,
          reason: event.reason || "",
        });
        disconnect(true).catch((err) => {
          log("disconnect error", { message: String(err) });
        });
      }
    });
  }

  async function disconnect(fromRemote = false) {
    const socket = ws;
    ws = null;

    if (!fromRemote && socket) {
      try {
        socket.close();
      } catch (_) {
        // Ignore close errors.
      }
    }

    stopAudioCapture();
    if (localStream) {
      localStream.getTracks().forEach((track) => track.stop());
      localStream = null;
    }
    if (playbackAudioContext) {
      clearPlaybackQueue();
      try {
        await playbackAudioContext.close();
      } catch (_) {
        // Ignore context close errors.
      }
      playbackAudioContext = null;
    } else {
      clearPlaybackQueue();
    }

    sessionId = null;
    pushToTalkActive = false;
    pushToTalkKeyActive = false;
    disconnectBtn.disabled = true;
    connectBtn.disabled = false;
    setStatus("Idle");
    updatePushToTalkUi();
    updateTextPromptUi();
    clearConversation();
  }

  connectBtn.addEventListener("click", async () => {
    try {
      await connect();
    } catch (err) {
      log("connect error", { message: String(err) });
      await disconnect(true);
    }
  });

  disconnectBtn.addEventListener("click", () => {
    disconnect(false).catch((err) => {
      log("disconnect error", { message: String(err) });
    });
  });

  pushToTalkBtn.addEventListener("mousedown", (event) => {
    event.preventDefault();
    beginPushToTalk("button");
  });
  pushToTalkBtn.addEventListener("mouseup", (event) => {
    event.preventDefault();
    commitPushToTalk("button");
  });
  pushToTalkBtn.addEventListener("mouseleave", () => {
    commitPushToTalk("button-leave");
  });
  pushToTalkBtn.addEventListener("touchstart", (event) => {
    event.preventDefault();
    beginPushToTalk("touch");
  });
  pushToTalkBtn.addEventListener("touchend", (event) => {
    event.preventDefault();
    commitPushToTalk("touch");
  });

  window.addEventListener("keydown", (event) => {
    if (inputAudioMode !== "manual") {
      return;
    }
    if (event.code !== "Space" || event.repeat) {
      return;
    }
    const tagName = event.target && event.target.tagName ? event.target.tagName : "";
    if (tagName === "TEXTAREA" || tagName === "INPUT") {
      return;
    }
    event.preventDefault();
    pushToTalkKeyActive = true;
    beginPushToTalk("space");
  });

  window.addEventListener("keyup", (event) => {
    if (inputAudioMode !== "manual") {
      return;
    }
    if (event.code !== "Space" || !pushToTalkKeyActive) {
      return;
    }
    event.preventDefault();
    pushToTalkKeyActive = false;
    commitPushToTalk("space");
  });

  window.addEventListener("blur", () => {
    if (inputAudioMode !== "manual") {
      return;
    }
    pushToTalkKeyActive = false;
    commitPushToTalk("window-blur");
  });

  audioModeAutoEl.addEventListener("change", () => {
    if (!audioModeAutoEl.checked) {
      return;
    }
    setInputAudioMode("vad");
    if (canSendControlEvent()) {
      sendSessionUpdate();
    }
  });

  audioModePushEl.addEventListener("change", () => {
    if (!audioModePushEl.checked) {
      return;
    }
    setInputAudioMode("manual");
    if (canSendControlEvent()) {
      sendSessionUpdate();
    }
  });

  clearLogBtn.addEventListener("click", () => {
    logEl.textContent = "";
  });

  sendTextBtn.addEventListener("click", () => {
    submitTextPrompt();
  });

  userPromptEl.addEventListener("input", () => {
    updateTextPromptUi();
  });

  userPromptEl.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" || event.shiftKey) {
      return;
    }
    event.preventDefault();
    submitTextPrompt();
  });

  updateMicLevel(0);
  updateRemoteLevel(0);
  clearConversation();
  setInputAudioMode("vad");
  updatePushToTalkUi();
  updateTextPromptUi();
})();
