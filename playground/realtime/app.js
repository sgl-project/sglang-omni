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
  const cameraEl = $("camera");
  const audioModeAutoEl = $("audio-mode-auto");
  const audioModePushEl = $("audio-mode-push");
  const audioModeHelpEl = $("audio-mode-help");
  const localVideoEl = $("local-video");
  const micLevelFillEl = $("mic-level-fill");
  const micLevelTextEl = $("mic-level-text");
  const remoteLevelFillEl = $("remote-level-fill");
  const remoteLevelTextEl = $("remote-level-text");
  const remoteAudioEl = $("remote-audio");

  let pc = null;
  let dc = null;
  let localStream = null;
  let sessionId = null;
  let micAudioContext = null;
  let micAnalyser = null;
  let micSource = null;
  let micLevelData = null;
  let micMeterRaf = 0;
  let remoteAudioContext = null;
  let remoteAnalyser = null;
  let remoteSource = null;
  let remoteLevelData = null;
  let remoteMeterRaf = 0;
  let pushToTalkActive = false;
  let pushToTalkKeyActive = false;
  let inputAudioMode = "vad";
  let assistantMessages = new Map();

  function getRtcConfiguration() {
    const config =
      typeof window !== "undefined" && window.SGLANG_OMNI_ICE_CONFIG
        ? window.SGLANG_OMNI_ICE_CONFIG
        : null;
    if (!config || !Array.isArray(config.urls) || config.urls.length === 0) {
      return undefined;
    }
    return {
      iceServers: [
        {
          urls: config.urls,
          username: config.username || undefined,
          credential: config.credential || undefined,
        },
      ],
    };
  }

  function getApiBase() {
    if (
      typeof window !== "undefined" &&
      Object.prototype.hasOwnProperty.call(window, "SGLANG_OMNI_API_BASE")
    ) {
      return String(window.SGLANG_OMNI_API_BASE || "").trim().replace(/\/$/, "");
    }
    return "";
  }

  function waitForIceGatheringComplete(peerConnection, timeoutMs = 5000) {
    if (!peerConnection || peerConnection.iceGatheringState === "complete") {
      return Promise.resolve();
    }

    return new Promise((resolve) => {
      let timeoutId = 0;

      const cleanup = () => {
        peerConnection.removeEventListener("icegatheringstatechange", onStateChange);
        if (timeoutId) {
          window.clearTimeout(timeoutId);
        }
      };

      const onStateChange = () => {
        if (peerConnection.iceGatheringState === "complete") {
          cleanup();
          resolve();
        }
      };

      peerConnection.addEventListener("icegatheringstatechange", onStateChange);
      timeoutId = window.setTimeout(() => {
        cleanup();
        resolve();
      }, timeoutMs);
    });
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
      hasText: false,
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
    if (!entry.hasText) {
      entry.contentEl.textContent = delta;
      entry.hasText = true;
    } else {
      entry.contentEl.textContent += delta;
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
    if (!entry.hasText && typeof text === "string" && text.length > 0) {
      entry.contentEl.textContent = text;
      entry.hasText = true;
    }
    if (!entry.hasText) {
      entry.contentEl.textContent = "(no text output)";
    }
    entry.container.classList.remove("pending");
    scrollConversationToBottom();
  }

  function handleServerEvent(event) {
    if (!event || typeof event.type !== "string") {
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
    if (event.type === "response.cancelled" && typeof event.response_id === "string") {
      const entry = assistantMessages.get(event.response_id);
      if (entry) {
        entry.container.classList.remove("pending");
      }
    }
  }

  function canSendControlEvent() {
    return Boolean(dc && dc.readyState === "open");
  }

  function sendControlEvent(payload) {
    if (!canSendControlEvent()) {
      log("control event skipped", {
        type: payload && payload.type ? payload.type : "unknown",
        reason: "data channel not open",
      });
      return false;
    }
    dc.send(JSON.stringify(payload));
    return true;
  }

  function updateLocalVideoPreview(stream) {
    const hasVideoTrack =
      stream && typeof stream.getVideoTracks === "function" && stream.getVideoTracks().length > 0;
    if (!hasVideoTrack) {
      localVideoEl.srcObject = null;
      localVideoEl.classList.add("hidden");
      return;
    }

    localVideoEl.srcObject = stream;
    localVideoEl.classList.remove("hidden");
    const playPromise = localVideoEl.play();
    if (playPromise && typeof playPromise.catch === "function") {
      playPromise.catch(() => {});
    }
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

  function updatePushToTalkUi() {
    const manualMode = inputAudioMode === "manual";
    pushToTalkBtn.disabled = !(manualMode && pc && canSendControlEvent());
    pushToTalkBtn.classList.toggle("hidden", !manualMode);
    pushToTalkBtn.classList.toggle("active", pushToTalkActive);
    pushToTalkBtn.textContent = pushToTalkActive
      ? "Release To Commit"
      : "Hold To Talk";
  }

  function updateTextPromptUi() {
    const connected = Boolean(pc && canSendControlEvent());
    userPromptEl.disabled = !connected;
    sendTextBtn.disabled = !(connected && userPromptEl.value.trim());
  }

  function updateAudioModeHelp() {
    if (inputAudioMode === "manual") {
      audioModeHelpEl.textContent =
        "Push To Talk only captures while you hold the button or space bar.";
      return;
    }
    audioModeHelpEl.textContent =
      "Auto VAD streams continuously and lets the server detect utterance boundaries.";
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

  function sendSessionModeUpdate() {
    return sendControlEvent({
      type: "session.update",
      session: {
        audio: {
          input_mode: inputAudioMode,
        },
      },
    });
  }

  function stopMicLevelMeter() {
    if (micMeterRaf) {
      cancelAnimationFrame(micMeterRaf);
      micMeterRaf = 0;
    }
    if (micSource) {
      try {
        micSource.disconnect();
      } catch (_) {
        // Ignore analyzer teardown errors.
      }
      micSource = null;
    }
    if (micAnalyser) {
      try {
        micAnalyser.disconnect();
      } catch (_) {
        // Ignore analyzer teardown errors.
      }
      micAnalyser = null;
    }
    if (micAudioContext) {
      micAudioContext.close().catch(() => {});
      micAudioContext = null;
    }
    micLevelData = null;
    updateMicLevel(0);
  }

  function stopRemoteLevelMeter() {
    if (remoteMeterRaf) {
      cancelAnimationFrame(remoteMeterRaf);
      remoteMeterRaf = 0;
    }
    if (remoteSource) {
      try {
        remoteSource.disconnect();
      } catch (_) {
        // Ignore analyzer teardown errors.
      }
      remoteSource = null;
    }
    if (remoteAnalyser) {
      try {
        remoteAnalyser.disconnect();
      } catch (_) {
        // Ignore analyzer teardown errors.
      }
      remoteAnalyser = null;
    }
    if (remoteAudioContext) {
      remoteAudioContext.close().catch(() => {});
      remoteAudioContext = null;
    }
    remoteLevelData = null;
    updateRemoteLevel(0);
  }

  async function startMicLevelMeter(stream) {
    stopMicLevelMeter();

    const hasAudioTrack =
      stream && typeof stream.getAudioTracks === "function" && stream.getAudioTracks().length > 0;
    if (!hasAudioTrack) {
      return;
    }

    const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextCtor) {
      log("mic level meter unavailable", { message: "Web Audio API is not supported" });
      return;
    }

    micAudioContext = new AudioContextCtor();
    if (micAudioContext.state === "suspended") {
      await micAudioContext.resume();
    }
    micAnalyser = micAudioContext.createAnalyser();
    micAnalyser.fftSize = 2048;
    micAnalyser.smoothingTimeConstant = 0.85;
    micSource = micAudioContext.createMediaStreamSource(stream);
    micSource.connect(micAnalyser);
    micLevelData = new Float32Array(micAnalyser.fftSize);

    const tick = () => {
      if (!micAnalyser || !micLevelData) {
        return;
      }
      micAnalyser.getFloatTimeDomainData(micLevelData);
      let sum = 0;
      for (let i = 0; i < micLevelData.length; i += 1) {
        const sample = micLevelData[i];
        sum += sample * sample;
      }
      const rms = Math.sqrt(sum / micLevelData.length);
      updateMicLevel(Math.min(rms * 4.0, 1.0));
      micMeterRaf = requestAnimationFrame(tick);
    };

    tick();
  }

  async function startRemoteLevelMeter(stream) {
    stopRemoteLevelMeter();

    const hasAudioTrack =
      stream && typeof stream.getAudioTracks === "function" && stream.getAudioTracks().length > 0;
    if (!hasAudioTrack) {
      return;
    }

    const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextCtor) {
      log("remote level meter unavailable", { message: "Web Audio API is not supported" });
      return;
    }

    remoteAudioContext = new AudioContextCtor();
    if (remoteAudioContext.state === "suspended") {
      await remoteAudioContext.resume();
    }
    remoteAnalyser = remoteAudioContext.createAnalyser();
    remoteAnalyser.fftSize = 2048;
    remoteAnalyser.smoothingTimeConstant = 0.85;
    remoteSource = remoteAudioContext.createMediaStreamSource(stream);
    remoteSource.connect(remoteAnalyser);
    remoteLevelData = new Float32Array(remoteAnalyser.fftSize);

    const tick = () => {
      if (!remoteAnalyser || !remoteLevelData) {
        return;
      }
      remoteAnalyser.getFloatTimeDomainData(remoteLevelData);
      let sum = 0;
      for (let i = 0; i < remoteLevelData.length; i += 1) {
        const sample = remoteLevelData[i];
        sum += sample * sample;
      }
      const rms = Math.sqrt(sum / remoteLevelData.length);
      updateRemoteLevel(Math.min(rms * 4.0, 1.0));
      remoteMeterRaf = requestAnimationFrame(tick);
    };

    tick();
  }

  function beginPushToTalk(source) {
    if (inputAudioMode !== "manual") {
      return;
    }
    if (pushToTalkActive) {
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
    if (inputAudioMode !== "manual") {
      return;
    }
    if (!pushToTalkActive) {
      return;
    }
    pushToTalkActive = false;
    updatePushToTalkUi();
    if (!sendControlEvent({ type: "input_audio_buffer.commit" })) {
      return;
    }
    log("push-to-talk committed", { source });
  }

  function ensureRemoteAudioPlaying() {
    if (!remoteAudioEl || !remoteAudioEl.srcObject) {
      return;
    }
    const playPromise = remoteAudioEl.play();
    if (playPromise && typeof playPromise.catch === "function") {
      playPromise.catch(() => {});
    }
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
    if (pc) return;
    connectBtn.disabled = true;
    setStatus("Requesting media...");

    localStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        channelCount: 1,
      },
      video: cameraEl.checked,
    });
    updateLocalVideoPreview(localStream);
    await startMicLevelMeter(localStream);

    const rtcConfiguration = getRtcConfiguration();
    pc = new RTCPeerConnection(rtcConfiguration);
    dc = pc.createDataChannel("events");
    dc.addEventListener("open", () => {
      sendSessionModeUpdate();
      updatePushToTalkUi();
      updateTextPromptUi();
    });
    dc.addEventListener("close", () => {
      pushToTalkActive = false;
      updatePushToTalkUi();
      updateTextPromptUi();
    });
    dc.addEventListener("message", (event) => {
      try {
        const payload = JSON.parse(event.data);
        handleServerEvent(payload);
        log("server event", payload);
      } catch (err) {
        log("server message", { raw: event.data });
      }
    });

    const remoteStream = new MediaStream();
    remoteAudioEl.autoplay = true;
    remoteAudioEl.srcObject = remoteStream;
    pc.addEventListener("track", (event) => {
      if (event.track.kind === "audio") {
        remoteStream.addTrack(event.track);
        startRemoteLevelMeter(remoteStream).catch(() => {});
        ensureRemoteAudioPlaying();
      }
    });
    pc.addEventListener("connectionstatechange", () => {
      log("peer connection state", { state: pc.connectionState });
      setStatus(`RTC ${pc.connectionState}`);
    });

    localStream.getTracks().forEach((track) => pc.addTrack(track, localStream));

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    await waitForIceGatheringComplete(pc);

    const localDescription = pc.localDescription;
    if (!localDescription) {
      throw new Error("Local SDP offer was not created");
    }

    const offerUrl = `${getApiBase()}/v1/realtime/webrtc/offer`;

    let response;
    try {
      response = await fetch(offerUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sdp: localDescription.sdp,
          type: localDescription.type,
          instructions: instructionsEl.value.trim(),
          input_audio_mode: inputAudioMode,
        }),
      });
    } catch (err) {
      throw new Error(`Failed to POST ${offerUrl}: ${String(err)}`);
    }
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(
        `Offer failed via ${offerUrl} with ${response.status}: ${detail || response.statusText}`
      );
    }

    const answer = await response.json();
    sessionId = answer.session_id;
    await pc.setRemoteDescription(answer);
    clearConversation();
    setStatus(`Connected (${sessionId})`);
    log("session created", answer);

    disconnectBtn.disabled = false;
    updatePushToTalkUi();
    updateTextPromptUi();
  }

  async function disconnect() {
    const closingSessionId = sessionId;
    disconnectBtn.disabled = true;
    setStatus("Disconnecting...");

    if (dc) {
      dc.close();
      dc = null;
    }
    if (pc) {
      pc.close();
      pc = null;
    }
    if (localStream) {
      localStream.getTracks().forEach((track) => track.stop());
      localStream = null;
    }
    updateLocalVideoPreview(null);
    stopMicLevelMeter();
    stopRemoteLevelMeter();
    sessionId = null;
    pushToTalkActive = false;
    pushToTalkKeyActive = false;
    try {
      remoteAudioEl.pause();
    } catch (_) {
      // Ignore media-element teardown errors.
    }
    remoteAudioEl.srcObject = null;
    remoteAudioEl.removeAttribute("src");
    remoteAudioEl.load();
    clearConversation();
    connectBtn.disabled = false;
    setStatus("Idle");
    updatePushToTalkUi();
    updateTextPromptUi();

    if (closingSessionId) {
      log("local disconnect complete", {
        sessionId: closingSessionId,
        cleanup: "backend session cleanup is handled by peer connection state change",
      });
    }
  }

  connectBtn.addEventListener("click", async () => {
    try {
      await connect();
    } catch (err) {
      log("connect error", { message: String(err) });
      await disconnect();
    }
  });

  disconnectBtn.addEventListener("click", () => {
    disconnect().catch((err) => log("disconnect error", { message: String(err) }));
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
    if (pc && canSendControlEvent()) {
      sendSessionModeUpdate();
    }
  });
  audioModePushEl.addEventListener("change", () => {
    if (!audioModePushEl.checked) {
      return;
    }
    setInputAudioMode("manual");
    if (pc && canSendControlEvent()) {
      sendSessionModeUpdate();
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
