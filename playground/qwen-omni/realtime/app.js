/**
 * WIRE SERVICE — sglang-omni /v1/realtime
 *
 * Captures mic → 16 kHz mono PCM16 via AudioWorklet → base64-encodes →
 * sends `input_audio_buffer.append`. Server VAD is always on; auto-commit
 * fires on speech_stopped. Text-only mode renders the assistant reply and
 * user transcript. Text + audio mode renders the assistant reply and queues
 * its 24 kHz PCM16 audio for gapless playback.
 *
 * Vanilla — no framework, no build step, no error handling. Per house
 * style: if something fails, the browser console gets the exception.
 */

(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);

  // ─────────────────────  DOM refs  ─────────────────────
  const wsUrlEl       = $("ws-url");
  const outputModeEl  = $("output-mode");
  const instructionsEl = $("instructions");
  const connectBtn    = $("connect");
  const disconnectBtn = $("disconnect");
  const statusEl      = $("status");
  const statusDotEl   = $("status-dot");
  const livePillEl    = $("live-pill");
  const liveTextEl    = livePillEl.querySelector(".live-text");

  const micStartBtn   = $("mic-start");
  const micStopBtn    = $("mic-stop");
  const clearBufferBtn = $("clear-buffer");
  const micStatusEl   = $("mic-status");
  const oscilloCanvas = $("oscilloscope");
  const oscilloCtx    = oscilloCanvas.getContext("2d");

  const transcriptsEl = $("transcripts");
  const mastheadModeEl = $("masthead-mode");
  const mastheadTaglineEl = $("masthead-tagline");
  const transcriptsLabelEl = $("transcripts-label");
  const transcriptsTaglineEl = $("transcripts-tagline");

  // ─────────────────────  State  ─────────────────────
  let ws = null;
  let audioCtx = null;
  let micStream = null;
  let workletNode = null;
  let analyserNode = null;
  let drawRaf = 0;
  let playbackCtx = null;
  let nextPlaybackTime = 0;
  let activeModalities = null;
  let sessionReady = false;
  let turnCounter = 0;
  // Each turn card is keyed by the audio item_id minted at speech_started /
  // committed.
  const turnCards = new Map();           // item_id → DOM node
  // response.text.delta events have no item_id link to the audio. Server
  // serializes turns, so we maintain a FIFO of audio item_ids queued for a
  // response and bind one to each response.created.
  const pendingAudioForResponse = [];    // queue of item_ids awaiting response
  let respondingTurnItemId = null;       // item_id of the response currently streaming
  const TARGET_SR = 16000;
  const OUTPUT_SR = 24000;

  // ─────────────────────  Status helpers  ─────────────────────

  function setStatus(text, mode = "") {
    statusEl.textContent = text;
    statusDotEl.className = "status-dot" + (mode ? " " + mode : "");
  }

  function setLive(on) {
    if (on) {
      livePillEl.classList.add("on");
      liveTextEl.textContent = "ON THE WIRE";
    } else {
      livePillEl.classList.remove("on");
      liveTextEl.textContent = "OFFLINE";
    }
  }

  function setMicStatus(text) {
    micStatusEl.textContent = text;
  }

  // ─────────────────────  WebSocket  ─────────────────────

  function wsSend(payload) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify(payload));
  }

  function wantsAudioOutput() {
    return outputModeEl.value === "text-audio";
  }

  function connectionWantsAudio() {
    return activeModalities
      ? activeModalities.includes("audio")
      : wantsAudioOutput();
  }

  function selectedModalities() {
    return wantsAudioOutput() ? ["text", "audio"] : ["text"];
  }

  function sendSessionUpdate(modalities = activeModalities || selectedModalities()) {
    // turn_detection is fixed server-side (always server_vad with defaults);
    // output modalities follow the mode selected before opening the wire.
    const wantsAudio = modalities.includes("audio");
    const session = {
      modalities,
      input_audio_format: "pcm16",
      instructions: instructionsEl.value,
    };
    if (wantsAudio) {
      session.output_audio_format = "pcm16";
    }
    wsSend({
      type: "session.update",
      session,
    });
  }

  connectBtn.addEventListener("click", () => {
    const url = wsUrlEl.value.trim();
    activeModalities = selectedModalities();
    sessionReady = false;
    stopPlayback();
    if (connectionWantsAudio()) {
      ensurePlaybackContext();
    }
    outputModeEl.disabled = true;
    ws = new WebSocket(url);
    setStatus("Opening line…");

    ws.onopen = () => {
      setStatus("Negotiating output mode…");
      setLive(true);
      connectBtn.disabled = true;
      disconnectBtn.disabled = false;
      micStartBtn.disabled = true;
      sendSessionUpdate(activeModalities);
    };

    ws.onmessage = (ev) => {
      handleServerEvent(JSON.parse(ev.data));
    };

    ws.onclose = () => {
      setStatus("Standing by");
      setLive(false);
      connectBtn.disabled = false;
      disconnectBtn.disabled = true;
      outputModeEl.disabled = false;
      micStartBtn.disabled = true;
      micStopBtn.disabled = true;
      clearBufferBtn.disabled = true;
      stopMic();
      stopPlayback();
      activeModalities = null;
      sessionReady = false;
      ws = null;
    };

    ws.onerror = () => {
      setStatus("Wire error", "error");
    };
  });

  disconnectBtn.addEventListener("click", () => {
    if (ws) ws.close();
  });

  // ─────────────────────  Microphone  ─────────────────────

  micStartBtn.addEventListener("click", () => startMic());
  micStopBtn.addEventListener("click", () => {
    stopMic();
    clearTurns();
  });

  function clearTurns() {
    turnCards.clear();
    turnCounter = 0;
    pendingAudioForResponse.length = 0;
    respondingTurnItemId = null;
    transcriptsEl.innerHTML =
      '<p class="empty-state">The wire is quiet. Open it, then speak.</p>';
  }

  clearBufferBtn.addEventListener("click", () =>
    wsSend({ type: "input_audio_buffer.clear" }),
  );

  async function startMic() {
    // Explicit constraints: force mono, request 16 kHz (browser may
    // ignore but the AudioContext resamples regardless), and turn on
    // echo / noise / auto-gain so the engine sees a clean signal.
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: TARGET_SR,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    audioCtx = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: TARGET_SR,
    });

    const source = audioCtx.createMediaStreamSource(micStream);

    // Mix any stereo input to mono inside the worklet so we never
    // accidentally send only one (potentially silent) channel.
    const workletCode = `
      class Forwarder extends AudioWorkletProcessor {
        process(inputs) {
          const channels = inputs[0];
          if (!channels || !channels[0]) return true;
          if (channels.length === 1) {
            this.port.postMessage(channels[0]);
          } else {
            const n = channels[0].length;
            const mono = new Float32Array(n);
            const c = channels.length;
            for (let i = 0; i < n; i++) {
              let sum = 0;
              for (let k = 0; k < c; k++) sum += channels[k][i];
              mono[i] = sum / c;
            }
            this.port.postMessage(mono);
          }
          return true;
        }
      }
      registerProcessor('forwarder', Forwarder);
    `;
    const blob = new Blob([workletCode], { type: "application/javascript" });
    await audioCtx.audioWorklet.addModule(URL.createObjectURL(blob));

    workletNode = new AudioWorkletNode(audioCtx, "forwarder");
    workletNode.port.onmessage = (e) => onAudioFrame(e.data);

    analyserNode = audioCtx.createAnalyser();
    analyserNode.fftSize = 1024;
    analyserNode.smoothingTimeConstant = 0.6;

    source.connect(workletNode);
    source.connect(analyserNode);

    setMicStatus("Channel hot");
    micStartBtn.disabled = true;
    micStopBtn.disabled = false;
    clearBufferBtn.disabled = false;
    drawScope();
  }

  function stopMic() {
    if (workletNode) { workletNode.disconnect(); workletNode = null; }
    if (analyserNode) { analyserNode.disconnect(); analyserNode = null; }
    if (audioCtx) { audioCtx.close(); audioCtx = null; }
    if (micStream) {
      micStream.getTracks().forEach((t) => t.stop());
      micStream = null;
    }
    if (drawRaf) {
      cancelAnimationFrame(drawRaf);
      drawRaf = 0;
    }
    clearScope();
    setMicStatus("Channel cold");
    micStopBtn.disabled = true;
    clearBufferBtn.disabled = true;
    if (ws && ws.readyState === WebSocket.OPEN) {
      micStartBtn.disabled = false;
    }
  }

  function onAudioFrame(float32) {
    const pcm16 = new Int16Array(float32.length);
    for (let i = 0; i < float32.length; i++) {
      const s = Math.max(-1, Math.min(1, float32[i]));
      pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    const b64 = bytesToBase64(new Uint8Array(pcm16.buffer));
    wsSend({ type: "input_audio_buffer.append", audio: b64 });
  }

  function bytesToBase64(bytes) {
    let binary = "";
    const chunkSize = 0x8000;
    for (let i = 0; i < bytes.length; i += chunkSize) {
      binary += String.fromCharCode.apply(
        null,
        bytes.subarray(i, i + chunkSize),
      );
    }
    return btoa(binary);
  }

  function base64ToBytes(encoded) {
    const binary = atob(encoded);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    return bytes;
  }

  function ensurePlaybackContext() {
    if (!playbackCtx || playbackCtx.state === "closed") {
      playbackCtx = new (window.AudioContext || window.webkitAudioContext)();
      nextPlaybackTime = 0;
    }
    if (playbackCtx.state === "suspended") {
      playbackCtx.resume();
    }
    return playbackCtx;
  }

  function queueAudioDelta(encoded) {
    const bytes = base64ToBytes(encoded);
    if (bytes.byteLength % 2 !== 0) {
      throw new Error("Received an odd-length PCM16 audio delta");
    }

    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const samples = new Float32Array(bytes.byteLength / 2);
    for (let i = 0; i < samples.length; i++) {
      samples[i] = view.getInt16(i * 2, true) / 0x8000;
    }

    const ctx = ensurePlaybackContext();
    const buffer = ctx.createBuffer(1, samples.length, OUTPUT_SR);
    buffer.copyToChannel(samples, 0);
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);

    const startAt = Math.max(ctx.currentTime + 0.02, nextPlaybackTime);
    source.start(startAt);
    nextPlaybackTime = startAt + buffer.duration;
  }

  function stopPlayback() {
    if (playbackCtx) {
      playbackCtx.close();
      playbackCtx = null;
    }
    nextPlaybackTime = 0;
  }

  // ─────────────────────  Oscilloscope  ─────────────────────

  function clearScope() {
    const w = oscilloCanvas.width;
    const h = oscilloCanvas.height;
    oscilloCtx.clearRect(0, 0, w, h);
  }

  function drawScope() {
    if (!analyserNode) return;
    const w = oscilloCanvas.width;
    const h = oscilloCanvas.height;
    const buf = new Uint8Array(analyserNode.fftSize);
    analyserNode.getByteTimeDomainData(buf);

    oscilloCtx.clearRect(0, 0, w, h);

    // Faint baseline.
    oscilloCtx.strokeStyle = "rgba(26, 24, 20, 0.18)";
    oscilloCtx.lineWidth = 1;
    oscilloCtx.beginPath();
    oscilloCtx.moveTo(0, h / 2);
    oscilloCtx.lineTo(w, h / 2);
    oscilloCtx.stroke();

    // Waveform — vermilion ink.
    oscilloCtx.strokeStyle = "#c5392b";
    oscilloCtx.lineWidth = 1.5;
    oscilloCtx.lineJoin = "round";
    oscilloCtx.beginPath();
    const slice = w / buf.length;
    let x = 0;
    for (let i = 0; i < buf.length; i++) {
      const v = (buf[i] - 128) / 128.0;
      const y = h / 2 + v * (h / 2) * 0.95;
      if (i === 0) oscilloCtx.moveTo(x, y);
      else oscilloCtx.lineTo(x, y);
      x += slice;
    }
    oscilloCtx.stroke();

    drawRaf = requestAnimationFrame(drawScope);
  }

  // ─────────────────────  Server events  ─────────────────────

  function handleServerEvent(evt) {
    switch (evt.type) {
      case "session.created":
        setStatus("Negotiating output mode…");
        return;

      case "session.updated":
        if (
          !evt.session ||
          JSON.stringify(evt.session.modalities) !== JSON.stringify(activeModalities)
        ) {
          setStatus("Output mode negotiation failed", "error");
          if (ws) ws.close();
          return;
        }
        sessionReady = true;
        micStartBtn.disabled = false;
        setStatus(
          `${connectionWantsAudio() ? "Text + audio" : "Text only"} · ` +
            `session ${evt.session.id.slice(0, 12)}…`,
          "connected",
        );
        return;

      case "input_audio_buffer.speech_started":
        ensureTurn(evt.item_id);
        setTurnMeta(evt.item_id, `started ${ms(evt.audio_start_ms)}`);
        return;

      case "input_audio_buffer.speech_stopped":
        setTurnMeta(evt.item_id, `stopped ${ms(evt.audio_end_ms)}`);
        return;

      case "input_audio_buffer.committed":
        ensureTurn(evt.item_id);
        setTurnMeta(evt.item_id, "committed · awaiting reply");
        // Queue this turn for the next response.created — server processes
        // commits serially so FIFO is correct.
        pendingAudioForResponse.push(evt.item_id);
        return;

      // ── Pass 1: assistant reply (streams first) ──
      case "response.created":
        respondingTurnItemId = pendingAudioForResponse.shift() || null;
        if (respondingTurnItemId) {
          setTurnMeta(respondingTurnItemId, "replying");
        }
        return;

      case "response.text.delta":
        if (respondingTurnItemId) {
          appendToBody(respondingTurnItemId, "assistant-body", evt.delta || "");
        }
        return;

      case "response.text.done":
        if (respondingTurnItemId) {
          setTurnMeta(
            respondingTurnItemId,
            connectionWantsAudio()
              ? "reply streaming"
              : "reply done · transcribing",
          );
        }
        return;

      case "response.audio.delta":
        if (connectionWantsAudio() && evt.delta) {
          queueAudioDelta(evt.delta);
        }
        return;

      case "response.audio.done":
        if (respondingTurnItemId) {
          setTurnMeta(respondingTurnItemId, "reply streaming");
        }
        return;

      case "response.done":
        if (respondingTurnItemId) {
          const responseStatus =
            (evt.response && evt.response.status) || "completed";
          const node = turnCards.get(respondingTurnItemId);
          if (node) node.dataset.state = responseStatus;
          if (connectionWantsAudio() || responseStatus !== "completed") {
            setTurnMeta(
              respondingTurnItemId,
              responseStatus === "completed" ? "complete" : responseStatus,
            );
          }
        }
        respondingTurnItemId = null;
        return;

      // In text-only mode, render the incremental user transcript.
      case "conversation.item.input_audio_transcription.delta":
        if (!connectionWantsAudio()) {
          appendToBody(evt.item_id, "user-body", evt.delta || "");
        }
        return;

      case "conversation.item.input_audio_transcription.completed": {
        if (connectionWantsAudio()) return;
        const node = ensureTurn(evt.item_id);
        const responseFailed =
          node.dataset.state === "failed" || node.dataset.state === "cancelled";
        if (!responseFailed) node.dataset.state = "completed";
        const body = node.querySelector(".user-body");
        const final = evt.transcript || (body && body.textContent) || "";
        if (body) {
          if (final.trim()) {
            body.textContent = final;
          } else {
            body.textContent = "[silent — empty transcript]";
            body.style.fontStyle = "italic";
            body.style.color = "var(--ink-faint)";
          }
        }
        if (!responseFailed) setTurnMeta(evt.item_id, "complete");
        return;
      }

      case "error":
        setStatus("error: " + (evt.error && evt.error.code), "error");
        return;

      default:
        return;
    }
  }

  function ensureTurn(itemId) {
    let node = turnCards.get(itemId);
    if (node) return node;

    const empty = transcriptsEl.querySelector(".empty-state");
    if (empty) empty.remove();

    turnCounter += 1;
    const serial = "№ " + String(turnCounter).padStart(3, "0");

    node = document.createElement("article");
    node.className = "utterance";
    node.dataset.state = "in-progress";
    const meta = document.createElement("div");
    meta.className = "utterance-meta";
    for (const [className, text] of [
      ["serial", serial],
      ["ts", nowTime()],
      ["state", "opening"],
    ]) {
      const span = document.createElement("span");
      span.className = className;
      span.textContent = text;
      meta.appendChild(span);
    }
    node.appendChild(meta);

    const assistantRole = document.createElement("p");
    assistantRole.className = "utterance-role";
    assistantRole.textContent = "Assistant";
    node.appendChild(assistantRole);

    const assistantBody = document.createElement("p");
    assistantBody.className = "utterance-body assistant-body";
    node.appendChild(assistantBody);

    if (!connectionWantsAudio()) {
      const userRole = document.createElement("p");
      userRole.className = "utterance-role";
      userRole.textContent = "You said";
      node.appendChild(userRole);

      const userBody = document.createElement("p");
      userBody.className = "utterance-body user-body";
      node.appendChild(userBody);
    }
    transcriptsEl.appendChild(node);
    transcriptsEl.scrollTop = transcriptsEl.scrollHeight;
    turnCards.set(itemId, node);
    return node;
  }

  function appendToBody(itemId, bodyClass, text) {
    const node = ensureTurn(itemId);
    const body = node.querySelector("." + bodyClass);
    if (body) body.textContent += text;
  }

  function setTurnMeta(itemId, stateText) {
    const node = turnCards.get(itemId);
    if (!node) return;
    const stateEl = node.querySelector(".utterance-meta .state");
    if (stateEl) stateEl.textContent = stateText;
  }

  function ms(n) { return typeof n === "number" ? n + "ms" : ""; }

  function nowTime() {
    return new Date().toLocaleTimeString("en-GB", { hour12: false });
  }

  // ─────────────────────  Misc UI  ─────────────────────

  function updatePresentation() {
    if (wantsAudioOutput()) {
      mastheadModeEl.textContent = "LIVE AUDIO RESPONSES";
      mastheadTaglineEl.textContent =
        "A demonstration of /v1/realtime — voice in, streamed text and voice " +
        "out. Server VAD detects when you stop; the engine streams its spoken answer.";
      transcriptsLabelEl.textContent = "Assistant Responses";
      transcriptsTaglineEl.textContent =
        "Each response is set in Newsreader Italic, lifted from the wire as " +
        "the engine emits it.";
    } else {
      mastheadModeEl.textContent = "A LIVE TRANSCRIPT";
      mastheadTaglineEl.textContent =
        "A demonstration of /v1/realtime — voice in, text out. " +
        "Server VAD detects when you stop; the engine answers, then " +
        "transcribes what you said.";
      transcriptsLabelEl.textContent = "The Transcript";
      transcriptsTaglineEl.textContent =
        "Each turn shows the assistant reply followed by the verbatim " +
        "transcript of what you said.";
    }
    clearTurns();
  }

  outputModeEl.addEventListener("change", () => {
    stopPlayback();
    updatePresentation();
  });
  instructionsEl.addEventListener("change", () => sendSessionUpdate());
  updatePresentation();
})();
