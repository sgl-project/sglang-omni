(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const connectBtn = $("connect");
  const disconnectBtn = $("disconnect");
  const pushToTalkBtn = $("push-to-talk");
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
  const localCaptureTextEl = $("local-capture-text");
  const localCaptureAudioEl = $("local-capture-audio");
  const localCaptureDownloadEl = $("local-capture-download");
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
  let pushToTalkActive = false;
  let pushToTalkKeyActive = false;
  let localCaptureSupported = false;
  let localCaptureRecorder = null;
  let localCaptureStream = null;
  let localCaptureChunks = [];
  let localCaptureMimeType = "";
  let localCaptureActive = false;
  let localCaptureStartedAt = 0;
  let localCaptureObjectUrl = null;

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

  function releaseLocalCaptureUrl() {
    if (localCaptureObjectUrl) {
      URL.revokeObjectURL(localCaptureObjectUrl);
      localCaptureObjectUrl = null;
    }
  }

  function setLocalCaptureStatus(text) {
    localCaptureTextEl.textContent = text;
  }

  function resetLocalCaptureUi() {
    releaseLocalCaptureUrl();
    localCaptureAudioEl.pause();
    localCaptureAudioEl.removeAttribute("src");
    localCaptureAudioEl.load();
    localCaptureAudioEl.classList.add("hidden");
    localCaptureDownloadEl.href = "#";
    localCaptureDownloadEl.classList.add("hidden");
    setLocalCaptureStatus("No capture");
  }

  function updatePushToTalkUi() {
    pushToTalkBtn.disabled = !(pc && canSendControlEvent());
    pushToTalkBtn.classList.toggle("active", pushToTalkActive);
    pushToTalkBtn.textContent = pushToTalkActive
      ? "Release To Commit"
      : "Hold To Talk";
  }

  function encodeMonoPcm16Wav(floatSamples, sampleRate) {
    const pcm = new Int16Array(floatSamples.length);
    for (let i = 0; i < floatSamples.length; i += 1) {
      const clamped = Math.max(-1, Math.min(1, floatSamples[i]));
      pcm[i] = clamped < 0 ? Math.round(clamped * 0x8000) : Math.round(clamped * 0x7fff);
    }

    const bytesPerSample = 2;
    const dataSize = pcm.length * bytesPerSample;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    function writeAscii(offset, value) {
      for (let i = 0; i < value.length; i += 1) {
        view.setUint8(offset + i, value.charCodeAt(i));
      }
    }

    writeAscii(0, "RIFF");
    view.setUint32(4, 36 + dataSize, true);
    writeAscii(8, "WAVE");
    writeAscii(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * bytesPerSample, true);
    view.setUint16(32, bytesPerSample, true);
    view.setUint16(34, 16, true);
    writeAscii(36, "data");
    view.setUint32(40, dataSize, true);

    let offset = 44;
    for (let i = 0; i < pcm.length; i += 1) {
      view.setInt16(offset, pcm[i], true);
      offset += bytesPerSample;
    }
    return buffer;
  }

  function summarizeSamples(floatSamples) {
    if (!floatSamples.length) {
      return { rms: 0, peak: 0 };
    }
    let energy = 0;
    let peak = 0;
    for (let i = 0; i < floatSamples.length; i += 1) {
      const sample = floatSamples[i];
      const abs = Math.abs(sample);
      energy += sample * sample;
      if (abs > peak) {
        peak = abs;
      }
    }
    return {
      rms: Math.sqrt(energy / floatSamples.length),
      peak,
    };
  }

  function selectLocalCaptureMimeType() {
    if (typeof MediaRecorder === "undefined") {
      return "";
    }
    const candidates = [
      "audio/webm;codecs=opus",
      "audio/ogg;codecs=opus",
      "audio/webm",
      "audio/ogg",
    ];
    for (let i = 0; i < candidates.length; i += 1) {
      const candidate = candidates[i];
      if (typeof MediaRecorder.isTypeSupported === "function") {
        if (MediaRecorder.isTypeSupported(candidate)) {
          return candidate;
        }
      }
    }
    return "";
  }

  async function startLocalCaptureRecorder(stream) {
    stopLocalCaptureRecorder();

    const hasAudioTrack =
      stream && typeof stream.getAudioTracks === "function" && stream.getAudioTracks().length > 0;
    if (!hasAudioTrack) {
      return;
    }
    if (typeof MediaRecorder === "undefined") {
      log("local capture unavailable", { message: "MediaRecorder API is not supported" });
      return;
    }

    localCaptureSupported = true;
    localCaptureMimeType = selectLocalCaptureMimeType();
    log("local capture support ready", {
      mimeType: localCaptureMimeType || "default",
      trackLabel: stream.getAudioTracks()[0].label || "",
    });
  }

  function stopLocalCaptureRecorder() {
    localCaptureActive = false;
    localCaptureChunks = [];
    localCaptureStartedAt = 0;
    if (localCaptureRecorder) {
      if (localCaptureRecorder.state !== "inactive") {
        try {
          localCaptureRecorder.stop();
        } catch (_) {
          // Ignore recorder teardown errors.
        }
      }
      localCaptureRecorder = null;
    }
    if (localCaptureStream) {
      localCaptureStream.getTracks().forEach((track) => track.stop());
      localCaptureStream = null;
    }
    localCaptureMimeType = "";
  }

  async function decodeBlobToWav(blob) {
    const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextCtor) {
      throw new Error("Web Audio API is not supported");
    }
    const arrayBuffer = await blob.arrayBuffer();
    const context = new AudioContextCtor();
    try {
      const audioBuffer = await context.decodeAudioData(arrayBuffer.slice(0));
      const channelCount = audioBuffer.numberOfChannels;
      const sampleCount = audioBuffer.length;
      const monoSamples = new Float32Array(sampleCount);
      for (let channel = 0; channel < channelCount; channel += 1) {
        const data = audioBuffer.getChannelData(channel);
        for (let i = 0; i < sampleCount; i += 1) {
          monoSamples[i] += data[i] / channelCount;
        }
      }
      const wavBuffer = encodeMonoPcm16Wav(monoSamples, audioBuffer.sampleRate);
      return {
        wavBlob: new Blob([wavBuffer], { type: "audio/wav" }),
        sampleRate: audioBuffer.sampleRate,
        samples: monoSamples,
      };
    } finally {
      context.close().catch(() => {});
    }
  }

  function beginLocalCapture(source) {
    if (!localCaptureSupported || !localStream || localStream.getAudioTracks().length === 0) {
      log("local capture unavailable", {
        source,
        reason: "recorder support not ready",
      });
      return;
    }
    stopLocalCaptureRecorder();
    localCaptureChunks = [];
    localCaptureStartedAt = performance.now();

    const captureTrack = localStream.getAudioTracks()[0].clone();
    localCaptureStream = new MediaStream([captureTrack]);
    try {
      localCaptureRecorder = localCaptureMimeType
        ? new MediaRecorder(localCaptureStream, { mimeType: localCaptureMimeType })
        : new MediaRecorder(localCaptureStream);
    } catch (err) {
      captureTrack.stop();
      localCaptureStream = null;
      log("local capture init failed", {
        source,
        message: String(err),
      });
      setLocalCaptureStatus("Capture init failed");
      return;
    }
    localCaptureRecorder.addEventListener("dataavailable", (event) => {
      if (event.data && event.data.size > 0) {
        localCaptureChunks.push(event.data);
      }
    });
    localCaptureRecorder.addEventListener("error", (event) => {
      log("local capture recorder error", {
        source,
        message: String(event.error || event),
      });
    });
    localCaptureActive = true;
    try {
      localCaptureRecorder.start(250);
    } catch (err) {
      localCaptureActive = false;
      captureTrack.stop();
      localCaptureStream = null;
      localCaptureRecorder = null;
      log("local capture start failed", {
        source,
        message: String(err),
      });
      setLocalCaptureStatus("Capture start failed");
      return;
    }
    setLocalCaptureStatus("Recording...");
    log("local capture started", {
      source,
      mimeType: localCaptureRecorder.mimeType || localCaptureMimeType || "default",
      trackLabel: captureTrack.label || "",
    });
  }

  function finalizeLocalCapture(source) {
    if (!localCaptureActive) {
      return Promise.resolve(null);
    }
    localCaptureActive = false;
    if (!localCaptureRecorder || localCaptureRecorder.state === "inactive") {
      setLocalCaptureStatus("No capture");
      return Promise.resolve(null);
    }

    return new Promise((resolve) => {
      const recorder = localCaptureRecorder;
      const onStop = async () => {
        recorder.removeEventListener("stop", onStop);

        if (localCaptureChunks.length === 0) {
          setLocalCaptureStatus("No samples captured");
          log("local capture empty", { source });
          resolve({
            source,
            mimeType: recorder.mimeType || localCaptureMimeType || "unknown",
            chunkCount: 0,
            byteLength: 0,
          });
          return;
        }

        const blob = new Blob(localCaptureChunks, {
          type: recorder.mimeType || localCaptureMimeType || "application/octet-stream",
        });
        const chunkCount = localCaptureChunks.length;
        localCaptureChunks = [];

        let samples = null;
        let sampleRate = null;
        let playableBlob = blob;
        let extension = "bin";

        try {
          const decoded = await decodeBlobToWav(blob);
          playableBlob = decoded.wavBlob;
          samples = decoded.samples;
          sampleRate = decoded.sampleRate;
          extension = "wav";
        } catch (err) {
          const mimeType = blob.type || "";
          extension = mimeType.includes("ogg")
            ? "ogg"
            : mimeType.includes("webm")
              ? "webm"
              : "bin";
          log("local capture wav conversion failed", {
            source,
            mimeType: blob.type || "unknown",
            message: String(err),
          });
        }

        releaseLocalCaptureUrl();
        localCaptureObjectUrl = URL.createObjectURL(playableBlob);

        const timestamp = new Date().toISOString().replaceAll(":", "-");
        localCaptureAudioEl.src = localCaptureObjectUrl;
        localCaptureAudioEl.classList.remove("hidden");
        localCaptureDownloadEl.href = localCaptureObjectUrl;
        localCaptureDownloadEl.download = `local-capture-${timestamp}.${extension}`;
        localCaptureDownloadEl.classList.remove("hidden");

        const stats = samples ? summarizeSamples(samples) : { rms: null, peak: null };
        const durationMs = samples && sampleRate
          ? Math.round((samples.length / sampleRate) * 1000)
          : null;
        setLocalCaptureStatus(
          durationMs && sampleRate
            ? `${(durationMs / 1000).toFixed(2)}s @ ${sampleRate} Hz`
            : `${blob.size} bytes (${blob.type || "unknown"})`
        );

        const summary = {
          source,
          mimeType: blob.type || "unknown",
          chunkCount,
          byteLength: blob.size,
          durationMs,
          sampleRate,
          rms: stats.rms === null ? null : Number(stats.rms.toFixed(6)),
          peak: stats.peak === null ? null : Number(stats.peak.toFixed(6)),
          elapsedMs: Number((performance.now() - localCaptureStartedAt).toFixed(1)),
        };
        log("local capture ready", summary);
        localCaptureStartedAt = 0;
        stopLocalCaptureRecorder();
        resolve(summary);
      };

      recorder.addEventListener("stop", onStop);
      try {
        if (typeof recorder.requestData === "function" && recorder.state === "recording") {
          recorder.requestData();
        }
      } catch (_) {
        // Ignore recorder flush errors.
      }
      recorder.stop();
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

  function beginPushToTalk(source) {
    if (pushToTalkActive) {
      return;
    }
    if (!sendControlEvent({ type: "input_audio_buffer.start" })) {
      updatePushToTalkUi();
      return;
    }
    pushToTalkActive = true;
    beginLocalCapture(source);
    updatePushToTalkUi();
    log("push-to-talk started", { source });
  }

  async function commitPushToTalk(source) {
    if (!pushToTalkActive) {
      return;
    }
    pushToTalkActive = false;
    updatePushToTalkUi();
    if (!sendControlEvent({ type: "input_audio_buffer.commit" })) {
      return;
    }
    const captureSummary = await finalizeLocalCapture(source);
    log("push-to-talk committed", {
      source,
      localCapture: captureSummary,
    });
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
    await startLocalCaptureRecorder(localStream);
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
    dc.addEventListener("open", updatePushToTalkUi);
    dc.addEventListener("close", () => log("data channel close"));
    dc.addEventListener("close", () => {
      pushToTalkActive = false;
      updatePushToTalkUi();
    });
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
    updatePushToTalkUi();
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
    resetLocalCaptureUi();
    stopLocalCaptureRecorder();
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
    connectBtn.disabled = false;
    setStatus("Idle");
    updatePushToTalkUi();

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
    commitPushToTalk("button").catch((err) => {
      log("local capture finalize failed", { message: String(err) });
    });
  });
  pushToTalkBtn.addEventListener("mouseleave", () => {
    commitPushToTalk("button-leave").catch((err) => {
      log("local capture finalize failed", { message: String(err) });
    });
  });
  pushToTalkBtn.addEventListener("touchstart", (event) => {
    event.preventDefault();
    beginPushToTalk("touch");
  });
  pushToTalkBtn.addEventListener("touchend", (event) => {
    event.preventDefault();
    commitPushToTalk("touch").catch((err) => {
      log("local capture finalize failed", { message: String(err) });
    });
  });

  window.addEventListener("keydown", (event) => {
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
    if (event.code !== "Space" || !pushToTalkKeyActive) {
      return;
    }
    event.preventDefault();
    pushToTalkKeyActive = false;
    commitPushToTalk("space").catch((err) => {
      log("local capture finalize failed", { message: String(err) });
    });
  });
  window.addEventListener("blur", () => {
    pushToTalkKeyActive = false;
    commitPushToTalk("window-blur").catch((err) => {
      log("local capture finalize failed", { message: String(err) });
    });
  });

  clearLogBtn.addEventListener("click", () => {
    logEl.textContent = "";
  });

  updateMicLevel(0);
  updateRemoteLevel(0);
  resetLocalCaptureUi();
  updatePushToTalkUi();
})();
