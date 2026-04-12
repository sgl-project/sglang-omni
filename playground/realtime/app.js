(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const connectBtn = $("connect");
  const disconnectBtn = $("disconnect");
  const clearLogBtn = $("clear-log");
  const statusEl = $("status");
  const logEl = $("log");
  const instructionsEl = $("instructions");
  const cameraEl = $("camera");
  const textOutputEl = $("text-output");
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
  let remoteAudioDiagnosticsAttached = false;
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

  function describeTrack(track) {
    return {
      kind: track.kind,
      id: track.id,
      enabled: track.enabled,
      muted: track.muted,
      readyState: track.readyState,
      label: track.label || "",
    };
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
      playPromise.catch((err) => {
        log("local video play notice", { message: String(err) });
      });
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

    log("mic level meter ready", {
      sampleRate: micAudioContext.sampleRate,
      fftSize: micAnalyser.fftSize,
    });
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

    log("remote level meter ready", {
      sampleRate: remoteAudioContext.sampleRate,
      fftSize: remoteAnalyser.fftSize,
    });
    tick();
  }

  function ensureRemoteAudioPlaying(reason) {
    if (!remoteAudioEl || !remoteAudioEl.srcObject) {
      return;
    }
    const playPromise = remoteAudioEl.play();
    if (playPromise && typeof playPromise.catch === "function") {
      playPromise.catch((err) => {
        const message = String(err);
        if (!message.includes("aborted")) {
          log("remote audio play notice", { reason, message });
        }
      });
    }
  }

  function attachRemoteAudioDiagnostics() {
    if (remoteAudioDiagnosticsAttached) {
      return;
    }
    remoteAudioDiagnosticsAttached = true;

    remoteAudioEl.addEventListener("loadedmetadata", () => {
      log("remote audio loadedmetadata", {
        readyState: remoteAudioEl.readyState,
        paused: remoteAudioEl.paused,
      });
      ensureRemoteAudioPlaying("loadedmetadata");
    });
    remoteAudioEl.addEventListener("canplay", () => {
      log("remote audio canplay", {
        readyState: remoteAudioEl.readyState,
        paused: remoteAudioEl.paused,
      });
      ensureRemoteAudioPlaying("canplay");
    });
    remoteAudioEl.addEventListener("playing", () => {
      log("remote audio playing", {
        currentTime: remoteAudioEl.currentTime,
        readyState: remoteAudioEl.readyState,
      });
    });
    remoteAudioEl.addEventListener("pause", () => {
      log("remote audio pause", {
        currentTime: remoteAudioEl.currentTime,
        ended: remoteAudioEl.ended,
      });
    });
    remoteAudioEl.addEventListener("error", () => {
      const mediaError = remoteAudioEl.error;
      log("remote audio error", {
        code: mediaError ? mediaError.code : null,
        message: mediaError ? mediaError.message : null,
      });
    });
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
    log("local media acquired", {
      tracks: localStream.getTracks().map(describeTrack),
    });
    updateLocalVideoPreview(localStream);
    await startMicLevelMeter(localStream);
    attachRemoteAudioDiagnostics();

    const rtcConfiguration = getRtcConfiguration();
    pc = new RTCPeerConnection(rtcConfiguration);
    log("rtc configuration", {
      iceServers:
        rtcConfiguration && Array.isArray(rtcConfiguration.iceServers)
          ? rtcConfiguration.iceServers.map((server) => ({
              urls: server.urls,
            }))
          : [],
    });
    dc = pc.createDataChannel("events");
    dc.addEventListener("open", () => log("data channel open"));
    dc.addEventListener("close", () => log("data channel close"));
    dc.addEventListener("error", (event) => {
      log("data channel error", { message: String(event.error || event) });
    });
    dc.addEventListener("message", (event) => {
      try {
        log("server event", JSON.parse(event.data));
      } catch (err) {
        log("server message", { raw: event.data });
      }
    });

    const remoteStream = new MediaStream();
    remoteAudioEl.autoplay = true;
    remoteAudioEl.srcObject = remoteStream;
    pc.addEventListener("track", (event) => {
      log("remote track received", {
        kind: event.track.kind,
        id: event.track.id,
        streams: event.streams.map((stream) => stream.id),
      });
      if (event.track.kind === "audio") {
        remoteStream.addTrack(event.track);
        startRemoteLevelMeter(remoteStream).catch((err) => {
          log("remote level meter error", { message: String(err) });
        });
        ensureRemoteAudioPlaying("remote-track");
      }
    });
    pc.addEventListener("connectionstatechange", () => {
      log("peer connection state", { state: pc.connectionState });
      setStatus(`RTC ${pc.connectionState}`);
    });
    pc.addEventListener("iceconnectionstatechange", () => {
      log("ice connection state", { state: pc.iceConnectionState });
    });
    pc.addEventListener("icegatheringstatechange", () => {
      log("ice gathering state", { state: pc.iceGatheringState });
    });
    pc.addEventListener("signalingstatechange", () => {
      log("signaling state", { state: pc.signalingState });
    });
    pc.addEventListener("icecandidateerror", (event) => {
      log("ice candidate error", {
        address: event.address,
        port: event.port,
        url: event.url,
        errorCode: event.errorCode,
        errorText: event.errorText,
      });
    });

    localStream.getTracks().forEach((track) => pc.addTrack(track, localStream));

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    const offerUrl = `${getApiBase()}/v1/realtime/webrtc/offer`;
    log("creating realtime offer", {
      offerUrl,
      camera: Boolean(cameraEl.checked),
      outputText: Boolean(textOutputEl.checked),
    });

    let response;
    try {
      response = await fetch(offerUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sdp: offer.sdp,
          type: offer.type,
          instructions: instructionsEl.value.trim(),
          output_text: Boolean(textOutputEl.checked),
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
    setStatus(`Connected (${sessionId})`);
    log("session created", answer);

    disconnectBtn.disabled = false;
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
    try {
      remoteAudioEl.pause();
    } catch (_) {
      // Ignore media-element teardown errors.
    }
    remoteAudioEl.srcObject = null;
    remoteAudioEl.removeAttribute("src");
    remoteAudioEl.load();
    connectBtn.disabled = false;
    setStatus("Idle");

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

  clearLogBtn.addEventListener("click", () => {
    logEl.textContent = "";
  });

  updateMicLevel(0);
  updateRemoteLevel(0);
})();
