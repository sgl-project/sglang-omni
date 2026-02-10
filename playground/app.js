/**
 * SGLang-Omni Playground - JavaScript frontend with API streaming.
 * Features: sys+user prompt, image/video/audio upload, webcam recording, mic recording,
 * streaming LLM output, hyperparameters (temperature, top_p, top_k), model select.
 */

(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);

  // DOM refs
  const apiUrlEl = $("api-url");
  const modelSelect = $("model-select");
  const systemPromptEl = $("system-prompt");
  const userPromptEl = $("user-prompt");
  const temperatureEl = $("temperature");
  const temperatureValueEl = $("temperature-value");
  const topPEl = $("top-p");
  const topPValueEl = $("top-p-value");
  const topKEl = $("top-k");
  const topKValueEl = $("top-k-value");
  const returnAudioEl = $("return-audio");

  const imageInput = $("image-input");
  const videoInput = $("video-input");
  const audioInput = $("audio-input");
  const audioControls = $("audio-controls");
  const audioVisualizerWrap = $("audio-visualizer-wrap");
  const audioVisualizer = $("audio-visualizer");
  const micStopBtn = $("mic-stop");
  const recordRow = $("record-row");

  const imagePreviews = $("image-previews");
  const videoPreviews = $("video-previews");
  const audioPreviews = $("audio-previews");

  const webcamToggle = $("webcam-toggle");
  const webcamPreview = $("webcam-preview");
  const webcamCapture = $("webcam-capture");
  const webcamPreviews = $("webcam-previews");

  const micToggle = $("mic-toggle");
  const micStatus = $("mic-status");
  const micPreviews = $("mic-previews");
  const audioSection = document.querySelector(".io-section.audio");

  const chatArea = $("chat-area");
  const chatPlaceholder = $("chat-placeholder");
  const messagesEl = $("messages");

  const sendBtn = $("send-btn");
  const stopBtn = $("stop-btn");
  const clearBtn = $("clear-btn");
  if (micToggle) micToggle.textContent = "Record audio";
  if (webcamToggle) webcamToggle.textContent = "Access webcam";
  if (micStatus) micStatus.classList.add("hidden");

  // State
  const state = {
    images: [],
    videos: [],
    audios: [],
    webcamVideos: [],
    micAudios: [],
    streaming: false,
    isRecording: false,
    abortController: null,
    webcamStream: null,
    webcamRecorder: null,
    webcamChunks: [],
    micRecorder: null,
    micChunks: [],
    audioStream: null,
    audioCtx: null,
    analyser: null,
    analyserData: null,
    vizAnim: 0,
  };

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  function renderMediaInMessage(container, mediaItems) {
    if (!mediaItems || mediaItems.length === 0) return;
    const wrap = document.createElement("div");
    wrap.className = "msg-media";
    mediaItems.forEach((item) => {
      if (item.kind === "image") {
        const img = document.createElement("img");
        img.src = item.url;
        img.alt = item.name || "Image";
        wrap.appendChild(img);
      } else if (item.kind === "video") {
        const vid = document.createElement("video");
        vid.src = item.url;
        vid.controls = true;
        wrap.appendChild(vid);
      } else if (item.kind === "audio") {
        const aud = document.createElement("audio");
        aud.src = item.url;
        aud.controls = true;
        wrap.appendChild(aud);
      }
    });
    container.appendChild(wrap);
  }

  function addUserMessage(text, mediaItems) {
    chatPlaceholder.classList.add("hidden");
    const msg = document.createElement("div");
    msg.className = "msg msg-user";

    const role = document.createElement("div");
    role.className = "msg-role";
    role.textContent = "User";
    msg.appendChild(role);

    renderMediaInMessage(msg, mediaItems);

    const content = document.createElement("div");
    content.className = "msg-content";
    content.textContent = text || "(No text)";
    msg.appendChild(content);

    messagesEl.appendChild(msg);
    chatArea.scrollTop = chatArea.scrollHeight;
  }

  function addAssistantMessage(text, isStreaming) {
    const msg = document.createElement("div");
    msg.className = "msg msg-assistant" + (isStreaming ? " streaming" : "");
    msg.innerHTML =
      '<div class="msg-role">Assistant</div><div class="msg-content">' +
      escapeHtml(text || "") +
      "</div>";
    messagesEl.appendChild(msg);
    chatArea.scrollTop = chatArea.scrollHeight;
    return msg;
  }

  function updateAssistantMessage(msgEl, text) {
    const contentEl = msgEl.querySelector(".msg-content");
    if (contentEl) contentEl.textContent = text;
    chatArea.scrollTop = chatArea.scrollHeight;
  }

  function addAssistantAudio(msgEl, audioUrl) {
    const wrap = msgEl.querySelector(".msg-media") || document.createElement("div");
    if (!wrap.classList.contains("msg-media")) {
      wrap.className = "msg-media";
      msgEl.appendChild(wrap);
    }
    const aud = document.createElement("audio");
    aud.src = audioUrl;
    aud.controls = true;
    wrap.appendChild(aud);
  }

  function setStreamingState(isStreaming) {
    state.streaming = isStreaming;
    sendBtn.disabled = isStreaming;
    if (stopBtn) stopBtn.classList.toggle("hidden", !isStreaming);
  }

  function revokeUrls(list) {
    list.forEach((item) => {
      if (item.url) URL.revokeObjectURL(item.url);
    });
  }

  function clearAllMedia() {
    revokeUrls(state.images);
    revokeUrls(state.videos);
    revokeUrls(state.audios);
    revokeUrls(state.webcamVideos);
    revokeUrls(state.micAudios);
    state.images = [];
    state.videos = [];
    state.audios = [];
    state.webcamVideos = [];
    state.micAudios = [];
    if (imageInput) imageInput.value = "";
    if (videoInput) videoInput.value = "";
    if (audioInput) audioInput.value = "";
    renderAllPreviews();
  }

  function clearChat() {
    messagesEl.innerHTML = "";
    chatPlaceholder.classList.remove("hidden");
  }

  function createPreviewItem(fileOrBlob, kind, nameOverride) {
    const blob = fileOrBlob instanceof Blob ? fileOrBlob : null;
    const file = fileOrBlob instanceof File ? fileOrBlob : null;
    const url = URL.createObjectURL(fileOrBlob);
    return {
      kind,
      file: file || null,
      blob: blob || null,
      url,
      name: nameOverride || (file ? file.name : "recording"),
    };
  }

  function renderPreviewList(container, list, kind) {
    if (!container) return;
    container.innerHTML = "";
    list.forEach((item, index) => {
      const wrap = document.createElement("div");
      wrap.className = "preview-wrap";
      let mediaEl;
      if (kind === "image") {
        mediaEl = document.createElement("img");
        mediaEl.src = item.url;
        mediaEl.alt = item.name || "Image";
      } else if (kind === "video") {
        mediaEl = document.createElement("video");
        mediaEl.src = item.url;
        mediaEl.muted = true;
        mediaEl.playsInline = true;
      } else {
        mediaEl = document.createElement("canvas");
        mediaEl.className = "wave-canvas";
        mediaEl.width = 400;
        mediaEl.height = 60;
        drawWaveformFromAudio(mediaEl, item);
      }
      wrap.appendChild(mediaEl);

      if (kind === "audio") {
        const audioEl = document.createElement("audio");
        audioEl.src = item.url;
        audioEl.preload = "metadata";
        audioEl.className = "preview-audio-el";
        wrap.appendChild(audioEl);

        const waveWrap = document.createElement("div");
        waveWrap.className = "audio-wave-wrap";
        const scrubber = document.createElement("div");
        scrubber.className = "audio-scrubber";
        waveWrap.appendChild(mediaEl);
        waveWrap.appendChild(scrubber);

        const timeRow = document.createElement("div");
        timeRow.className = "audio-time-row";
        const timeCurrent = document.createElement("span");
        timeCurrent.className = "audio-time-current";
        timeCurrent.textContent = "0:00";
        const timeDuration = document.createElement("span");
        timeDuration.className = "audio-time-duration";
        timeDuration.textContent = "0:00";
        timeRow.appendChild(timeCurrent);
        timeRow.appendChild(timeDuration);
        wrap.appendChild(waveWrap);
        wrap.appendChild(timeRow);

        function updateTimeAndScrubber() {
          const t = audioEl.currentTime;
          const d = audioEl.duration;
          if (Number.isFinite(d) && d > 0) {
            timeCurrent.textContent = formatTime(t);
            timeDuration.textContent = formatTime(d);
            scrubber.style.left = (t / d) * 100 + "%";
          }
        }
        audioEl.addEventListener("loadedmetadata", () => {
          timeDuration.textContent = formatTime(audioEl.duration);
        });
        audioEl.addEventListener("timeupdate", updateTimeAndScrubber);
        audioEl.addEventListener("ended", () => {
          playBtn.classList.remove("playing");
          playBtn.innerHTML = playIcon;
          scrubber.style.left = "0%";
          timeCurrent.textContent = formatTime(0);
        });

        const playIcon = "<span class=\"audio-play-icon\">▶</span>";
        const pauseIcon = "<span class=\"audio-play-icon\">❚❚</span>";
        const playBtn = document.createElement("button");
        playBtn.type = "button";
        playBtn.className = "audio-play-btn";
        playBtn.innerHTML = playIcon;
        playBtn.title = "Play";
        playBtn.addEventListener("click", () => {
          if (audioEl.paused) {
            audioEl.play().catch(() => {});
            playBtn.classList.add("playing");
            playBtn.innerHTML = pauseIcon;
            playBtn.title = "Pause";
          } else {
            audioEl.pause();
            playBtn.classList.remove("playing");
            playBtn.innerHTML = playIcon;
            playBtn.title = "Play";
          }
        });

        const controlsRow = document.createElement("div");
        controlsRow.className = "audio-preview-controls";
        const volBtn = document.createElement("button");
        volBtn.type = "button";
        volBtn.className = "audio-icon-btn";
        volBtn.title = "Volume";
        volBtn.innerHTML = "<span class=\"audio-icon\">🔊</span>";
        const speedBtn = document.createElement("button");
        speedBtn.type = "button";
        speedBtn.className = "audio-speed-btn";
        speedBtn.textContent = "1x";
        const speeds = [0.5, 1, 1.5, 2];
        let speedIdx = 1;
        speedBtn.addEventListener("click", () => {
          speedIdx = (speedIdx + 1) % speeds.length;
          const r = speeds[speedIdx];
          audioEl.playbackRate = r;
          speedBtn.textContent = r + "x";
        });
        const rewindBtn = document.createElement("button");
        rewindBtn.type = "button";
        rewindBtn.className = "audio-icon-btn";
        rewindBtn.title = "Rewind";
        rewindBtn.innerHTML = "<span class=\"audio-icon\">⏪</span>";
        rewindBtn.addEventListener("click", () => {
          audioEl.currentTime = Math.max(0, audioEl.currentTime - 5);
        });
        const ffBtn = document.createElement("button");
        ffBtn.type = "button";
        ffBtn.className = "audio-icon-btn";
        ffBtn.title = "Forward";
        ffBtn.innerHTML = "<span class=\"audio-icon\">⏩</span>";
        ffBtn.addEventListener("click", () => {
          audioEl.currentTime = Math.min(audioEl.duration || 0, audioEl.currentTime + 5);
        });
        const stopBtn = document.createElement("button");
        stopBtn.type = "button";
        stopBtn.className = "audio-icon-btn";
        stopBtn.title = "Stop";
        stopBtn.innerHTML = "<span class=\"audio-icon\">⏹</span>";
        stopBtn.addEventListener("click", () => {
          audioEl.pause();
          audioEl.currentTime = 0;
          playBtn.classList.remove("playing");
          playBtn.innerHTML = playIcon;
          playBtn.title = "Play";
          updateTimeAndScrubber();
        });
        const loopBtn = document.createElement("button");
        loopBtn.type = "button";
        loopBtn.className = "audio-icon-btn";
        loopBtn.title = "Loop";
        loopBtn.innerHTML = "<span class=\"audio-icon\">🔁</span>";
        loopBtn.addEventListener("click", () => {
          audioEl.loop = !audioEl.loop;
          loopBtn.classList.toggle("active", audioEl.loop);
        });
        controlsRow.appendChild(volBtn);
        controlsRow.appendChild(speedBtn);
        controlsRow.appendChild(rewindBtn);
        controlsRow.appendChild(playBtn);
        controlsRow.appendChild(ffBtn);
        controlsRow.appendChild(stopBtn);
        controlsRow.appendChild(loopBtn);
        wrap.appendChild(controlsRow);
      }

      const delBtn = document.createElement("button");
      delBtn.type = "button";
      delBtn.className = "del-btn";
      delBtn.textContent = "×";
      delBtn.addEventListener("click", () => {
        URL.revokeObjectURL(item.url);
        list.splice(index, 1);
        renderAllPreviews();
        updateAudioControlsVisibility();
      });
      wrap.appendChild(delBtn);
      container.appendChild(wrap);
    });
  }

  function renderAllPreviews() {
    renderPreviewList(imagePreviews, state.images, "image");
    renderPreviewList(videoPreviews, state.videos, "video");
    renderPreviewList(audioPreviews, state.audios, "audio");
    renderPreviewList(webcamPreviews, state.webcamVideos, "video");
    renderPreviewList(micPreviews, state.micAudios, "audio");
    updateAudioControlsVisibility();
  }

  function hasAnyAudio() {
    return (state.audios.length + state.micAudios.length) > 0;
  }

  function updateAudioControlsVisibility() {
    const hasAudio = hasAnyAudio();
    const recording = state.isRecording;
    if (audioControls) {
      audioControls.style.display = (!recording && !hasAudio) ? "flex" : "none";
      const recordBtn = micToggle;
      const uploadLabel = audioInput ? audioInput.parentElement : null;
      if (recordBtn) recordBtn.classList.remove("hidden");
      if (uploadLabel) uploadLabel.classList.remove("hidden");
      if (micStatus) micStatus.classList.add("hidden");
    }
    if (recordRow) recordRow.classList.toggle("hidden", !recording);
    if (audioVisualizerWrap) audioVisualizerWrap.classList.toggle("hidden", !recording);
  }

  async function initVisualizer(stream) {
    if (!audioVisualizer) return;
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    state.audioCtx = ctx;
    state.analyser = ctx.createAnalyser();
    state.analyser.fftSize = 256;
    const source = ctx.createMediaStreamSource(stream);
    source.connect(state.analyser);
    const bufferLength = state.analyser.frequencyBinCount;
    state.analyserData = new Uint8Array(bufferLength);
    drawVisualizer();
  }

  function drawVisualizer() {
    if (!audioVisualizer || !state.analyser) return;
    const canvasCtx = audioVisualizer.getContext("2d");
    const width = audioVisualizer.width;
    const height = audioVisualizer.height;
    state.vizAnim = requestAnimationFrame(drawVisualizer);
    state.analyser.getByteTimeDomainData(state.analyserData);
    canvasCtx.fillStyle = "#eef0ff";
    canvasCtx.fillRect(0, 0, width, height);
    canvasCtx.lineWidth = 2;
    canvasCtx.strokeStyle = "#5c6bc0";
    canvasCtx.beginPath();
    const sliceWidth = width / state.analyserData.length;
    let x = 0;
    for (let i = 0; i < state.analyserData.length; i++) {
      const v = state.analyserData[i] / 128.0;
      const y = (v * height) / 2;
      if (i === 0) {
        canvasCtx.moveTo(x, y);
      } else {
        canvasCtx.lineTo(x, y);
      }
      x += sliceWidth;
    }
    canvasCtx.lineTo(width, height / 2);
    canvasCtx.stroke();
  }

  function stopVisualizer() {
    if (state.vizAnim) cancelAnimationFrame(state.vizAnim);
    state.vizAnim = 0;
    if (state.audioCtx) {
      try {
        state.audioCtx.close();
      } catch (e) {}
    }
    state.audioCtx = null;
    state.analyser = null;
    state.analyserData = null;
    if (audioVisualizer) {
      const canvasCtx = audioVisualizer.getContext("2d");
      canvasCtx.clearRect(0, 0, audioVisualizer.width, audioVisualizer.height);
    }
    if (audioVisualizerWrap) audioVisualizerWrap.classList.add("hidden");
  }

  function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return m + ":" + (s < 10 ? "0" : "") + s;
  }

  async function drawWaveformFromAudio(canvas, item) {
    try {
      const arrayBuffer = await (item.blob
        ? item.blob.arrayBuffer()
        : item.file
        ? item.file.arrayBuffer()
        : fetch(item.url).then((r) => r.arrayBuffer()));
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
      const raw = audioBuffer.getChannelData(0);
      const width = canvas.width;
      const height = canvas.height;
      const canvasCtx = canvas.getContext("2d");
      canvasCtx.clearRect(0, 0, width, height);
      canvasCtx.fillStyle = "#fff";
      canvasCtx.fillRect(0, 0, width, height);
      const centerY = height / 2;
      const barCount = Math.min(width, 120);
      const barWidth = Math.max(1, (width / barCount) - 1);
      const blockSize = Math.max(1, Math.floor(raw.length / barCount));
      canvasCtx.fillStyle = "#9ca3af";
      for (let i = 0; i < barCount; i++) {
        const blockStart = i * blockSize;
        let sum = 0;
        for (let j = 0; j < blockSize && blockStart + j < raw.length; j++) {
          sum += Math.abs(raw[blockStart + j]);
        }
        const avg = sum / blockSize;
        const barHeight = Math.max(2, avg * (height / 2));
        const x = (width / barCount) * i + 1;
        canvasCtx.fillRect(x, centerY - barHeight, barWidth, barHeight);
        canvasCtx.fillRect(x, centerY, barWidth, barHeight);
      }
      canvasCtx.setLineDash([4, 4]);
      canvasCtx.strokeStyle = "#d1d5db";
      canvasCtx.lineWidth = 1;
      canvasCtx.beginPath();
      canvasCtx.moveTo(0, centerY);
      canvasCtx.lineTo(width, centerY);
      canvasCtx.stroke();
      canvasCtx.setLineDash([]);
      ctx.close();
    } catch (err) {
      console.error("Waveform draw failed", err);
    }
  }

  function updateSliderValue(slider, outputEl, decimals) {
    if (!slider || !outputEl) return;
    const val = Number(slider.value);
    outputEl.textContent = Number.isFinite(val) ? val.toFixed(decimals) : slider.value;
  }

  if (temperatureEl) {
    temperatureEl.addEventListener("input", () => updateSliderValue(temperatureEl, temperatureValueEl, 2));
    updateSliderValue(temperatureEl, temperatureValueEl, 2);
  }
  if (topPEl) {
    topPEl.addEventListener("input", () => updateSliderValue(topPEl, topPValueEl, 2));
    updateSliderValue(topPEl, topPValueEl, 2);
  }
  if (topKEl) {
    topKEl.addEventListener("input", () => updateSliderValue(topKEl, topKValueEl, 0));
    updateSliderValue(topKEl, topKValueEl, 0);
  }
  updateAudioControlsVisibility();

  // File uploads
  if (imageInput) {
    imageInput.addEventListener("change", () => {
      const files = imageInput.files ? Array.from(imageInput.files) : [];
      files.forEach((file) => state.images.push(createPreviewItem(file, "image")));
      imageInput.value = "";
      renderAllPreviews();
    });
  }

  if (videoInput) {
    videoInput.addEventListener("change", () => {
      const files = videoInput.files ? Array.from(videoInput.files) : [];
      files.forEach((file) => state.videos.push(createPreviewItem(file, "video")));
      videoInput.value = "";
      renderAllPreviews();
    });
  }

  if (audioInput) {
    audioInput.addEventListener("change", () => {
      if (hasAnyAudio()) {
        audioInput.value = "";
        return;
      }
      const files = audioInput.files ? Array.from(audioInput.files) : [];
      if (files.length === 0) return;
      const file = files[0];
      state.audios.push(createPreviewItem(file, "audio"));
      audioInput.value = "";
      renderAllPreviews();
    });
  }

  // Webcam recording
  async function startWebcam() {
    if (state.webcamStream) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      state.webcamStream = stream;
      if (webcamPreview) {
        webcamPreview.srcObject = stream;
        webcamPreview.classList.remove("hidden");
        await webcamPreview.play();
      }
      state.webcamChunks = [];
      state.webcamRecorder = new MediaRecorder(stream);
      state.webcamRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size) state.webcamChunks.push(e.data);
      };
      state.webcamRecorder.onstop = () => {
        const blob = new Blob(state.webcamChunks, { type: "video/webm" });
        const file = new File([blob], `webcam_${Date.now()}.webm`, { type: "video/webm" });
        state.webcamVideos.push(createPreviewItem(file, "video", file.name));
        renderAllPreviews();
        stopWebcamStream();
      };
      state.webcamRecorder.start();
      if (webcamToggle) webcamToggle.classList.add("recording");
      if (webcamCapture) webcamCapture.classList.remove("hidden");
      if (webcamToggle) webcamToggle.textContent = "Stop webcam";
    } catch (err) {
      console.error("Webcam access failed:", err);
      stopWebcamStream();
    }
  }

  function stopWebcamStream() {
    if (state.webcamRecorder && state.webcamRecorder.state !== "inactive") {
      state.webcamRecorder.stop();
    }
    if (state.webcamStream) {
      state.webcamStream.getTracks().forEach((t) => t.stop());
      state.webcamStream = null;
    }
    if (webcamPreview) {
      webcamPreview.pause();
      webcamPreview.srcObject = null;
      webcamPreview.classList.add("hidden");
    }
    if (webcamToggle) webcamToggle.classList.remove("recording");
    if (webcamCapture) webcamCapture.classList.add("hidden");
    if (webcamToggle) webcamToggle.textContent = "Access webcam";
  }

  if (webcamToggle) {
    webcamToggle.addEventListener("click", () => {
      if (state.webcamStream) {
        stopWebcamStream();
      } else {
        startWebcam();
      }
    });
  }

  if (webcamCapture) {
    webcamCapture.addEventListener("click", () => {
      if (state.webcamRecorder && state.webcamRecorder.state !== "inactive") {
        state.webcamRecorder.stop();
      } else {
        stopWebcamStream();
      }
    });
  }

  // Mic recording
  async function toggleMic() {
    if (hasAnyAudio()) return;
    if (state.micRecorder && state.micRecorder.state === "recording") {
      state.micRecorder.stop();
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      state.audioStream = stream;
      await initVisualizer(stream);

      state.micChunks = [];
      state.micRecorder = new MediaRecorder(stream);
      state.micRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size) state.micChunks.push(e.data);
      };
      state.micRecorder.onstop = () => {
        stopVisualizer();
        if (state.audioStream) {
          state.audioStream.getTracks().forEach((t) => t.stop());
          state.audioStream = null;
        }
        const blob = new Blob(state.micChunks, { type: "audio/webm" });
        const file = new File([blob], `mic_${Date.now()}.webm`, { type: "audio/webm" });
        state.micAudios.push(createPreviewItem(file, "audio", file.name));
        state.isRecording = false;
        renderAllPreviews();
        if (micToggle) micToggle.classList.remove("recording");
        if (micStatus) {
          micStatus.textContent = "";
          micStatus.classList.add("hidden");
        }
        updateAudioControlsVisibility();
      };
      state.isRecording = true;
      state.micRecorder.start();
    if (micToggle) micToggle.classList.add("recording");
    if (micStatus) {
      micStatus.textContent = "Recording...";
      micStatus.classList.remove("hidden");
    }
      updateAudioControlsVisibility();
  } catch (err) {
      console.error("Mic access failed:", err);
      if (micStatus) micStatus.textContent = "Mic error";
      stopVisualizer();
      state.isRecording = false;
      updateAudioControlsVisibility();
    }
  }

  if (micToggle) {
    micToggle.addEventListener("click", () => {
      toggleMic();
    });
  }

  if (micStopBtn) {
    micStopBtn.addEventListener("click", () => {
      if (state.micRecorder && state.micRecorder.state === "recording") {
        state.micRecorder.stop();
      }
    });
  }

  async function fileToDataUrl(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(reader.error);
      reader.readAsDataURL(file);
    });
  }

  async function buildMediaPayload() {
    const images = [];
    const videos = [];
    const audios = [];

    const imageFiles = state.images.map((item) => item.file).filter(Boolean);
    const videoFiles = state.videos.concat(state.webcamVideos).map((item) => item.file).filter(Boolean);
    const audioFiles = state.audios.concat(state.micAudios).map((item) => item.file).filter(Boolean);

    for (const file of imageFiles) {
      images.push(await fileToDataUrl(file));
    }
    for (const file of videoFiles) {
      videos.push(await fileToDataUrl(file));
    }
    for (const file of audioFiles) {
      audios.push(await fileToDataUrl(file));
    }

    return { images, videos, audios };
  }

  function getAllMediaForMessage() {
    return [].concat(state.images, state.videos, state.webcamVideos, state.audios, state.micAudios);
  }

  async function* streamChatCompletion(payload) {
    const apiBase = (apiUrlEl && apiUrlEl.value ? apiUrlEl.value : "").replace(/\/$/, "");
    const url = apiBase + "/v1/chat/completions";
    const controller = new AbortController();
    state.abortController = controller;

    const res = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    if (!res.ok || !res.body) {
      const text = await res.text().catch(() => "");
      throw new Error(`Request failed (${res.status}): ${text || res.statusText}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    let done = false;
    let audioB64 = "";

    while (!done) {
      const { value, done: streamDone } = await reader.read();
      if (streamDone) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() || "";
      for (const part of parts) {
        const lines = part.split("\n");
        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith("data:")) continue;
          const data = trimmed.replace(/^data:\s*/, "");
          if (data === "[DONE]") {
            done = true;
            break;
          }
          let parsed;
          try {
            parsed = JSON.parse(data);
          } catch (err) {
            continue;
          }
          const delta = parsed.choices && parsed.choices[0] && parsed.choices[0].delta;
          if (delta && delta.content) {
            yield { type: "text", value: delta.content };
          }
          if (delta && delta.audio && delta.audio.data) {
            audioB64 += delta.audio.data;
          }
        }
        if (done) break;
      }
    }

    if (audioB64) {
      yield { type: "audio", value: audioB64 };
    }
  }

  async function handleSend() {
    if (state.streaming) return;
    const userText = userPromptEl.value.trim();
    const systemText = systemPromptEl.value.trim();

    const mediaItems = getAllMediaForMessage();
    if (!userText && mediaItems.length === 0) return;

    addUserMessage(userText || "(Media only)", mediaItems);
    userPromptEl.value = "";

    const assistantMsg = addAssistantMessage("", true);
    let assistantText = "";

    setStreamingState(true);

    try {
      const mediaPayload = await buildMediaPayload();
      const messages = [];
      if (systemText) {
        messages.push({ role: "system", content: systemText });
      }
      messages.push({ role: "user", content: userText || " " });

      const payload = {
        model: modelSelect && modelSelect.value ? modelSelect.value : undefined,
        messages: messages,
        temperature: temperatureEl ? Number(temperatureEl.value) : undefined,
        top_p: topPEl ? Number(topPEl.value) : undefined,
        top_k: topKEl ? Number(topKEl.value) : undefined,
        stream: true,
        images: mediaPayload.images.length ? mediaPayload.images : undefined,
        videos: mediaPayload.videos.length ? mediaPayload.videos : undefined,
        audios: mediaPayload.audios.length ? mediaPayload.audios : undefined,
      };

      if (returnAudioEl && returnAudioEl.checked) {
        payload.modalities = ["text", "audio"];
        payload.audio = { format: "wav" };
      }

      const stream = streamChatCompletion(payload);
      for await (const chunk of stream) {
        if (chunk.type === "text") {
          assistantText += chunk.value;
          updateAssistantMessage(assistantMsg, assistantText);
        } else if (chunk.type === "audio") {
          const audioUrl = `data:audio/wav;base64,${chunk.value}`;
          addAssistantAudio(assistantMsg, audioUrl);
        }
      }
    } catch (err) {
      if (err && err.name === "AbortError") {
        if (!assistantText) updateAssistantMessage(assistantMsg, "[stopped]");
      } else {
        console.error(err);
        // Fallback: mock streaming tokens so UI keeps consistent behavior without backend
        const mock = "This is a mock streaming response because the API request failed.";
        await mockStreamTokens(mock, (chunk) => {
          assistantText += chunk;
          updateAssistantMessage(assistantMsg, assistantText);
        });
      }
    } finally {
      assistantMsg.classList.remove("streaming");
      setStreamingState(false);
    }
  }

  // Mock token streaming helper for fallback UX
  function mockStreamTokens(fullText, onChunk) {
    return new Promise((resolve) => {
      let i = 0;
      function tick() {
        if (i < fullText.length) {
          onChunk(fullText[i]);
          i += 1;
          setTimeout(tick, 20);
        } else {
          resolve();
        }
      }
      tick();
    });
  }

  function handleStop() {
    if (state.abortController) {
      state.abortController.abort();
      state.abortController = null;
    }
    setStreamingState(false);
  }

  function handleClear() {
    handleStop();
    clearChat();
    clearAllMedia();
    if (userPromptEl) userPromptEl.value = "";
  }

  if (sendBtn) sendBtn.addEventListener("click", handleSend);
  if (stopBtn) stopBtn.addEventListener("click", handleStop);
  if (clearBtn) clearBtn.addEventListener("click", handleClear);

  if (userPromptEl) {
    userPromptEl.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    });
  }
})();
