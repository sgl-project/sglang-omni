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

  const chatArea = $("chat-area");
  const chatPlaceholder = $("chat-placeholder");
  const messagesEl = $("messages");

  const sendBtn = $("send-btn");
  const stopBtn = $("stop-btn");
  const clearBtn = $("clear-btn");
  if (micToggle) micToggle.textContent = "Record audio";
  if (webcamToggle) webcamToggle.textContent = "Access webcam";

  // State
  const state = {
    images: [],
    videos: [],
    audios: [],
    webcamVideos: [],
    micAudios: [],
    streaming: false,
    abortController: null,
    webcamStream: null,
    webcamRecorder: null,
    webcamChunks: [],
    micRecorder: null,
    micChunks: [],
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
        mediaEl = document.createElement("audio");
        mediaEl.src = item.url;
        mediaEl.controls = true;
      }
      const delBtn = document.createElement("button");
      delBtn.type = "button";
      delBtn.className = "del-btn";
      delBtn.textContent = "×";
      delBtn.addEventListener("click", () => {
        URL.revokeObjectURL(item.url);
        list.splice(index, 1);
        renderAllPreviews();
      });
      wrap.appendChild(mediaEl);
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
      const files = audioInput.files ? Array.from(audioInput.files) : [];
      files.forEach((file) => state.audios.push(createPreviewItem(file, "audio")));
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
    if (state.micRecorder && state.micRecorder.state === "recording") {
      state.micRecorder.stop();
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      state.micChunks = [];
      state.micRecorder = new MediaRecorder(stream);
      state.micRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size) state.micChunks.push(e.data);
      };
      state.micRecorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(state.micChunks, { type: "audio/webm" });
        const file = new File([blob], `mic_${Date.now()}.webm`, { type: "audio/webm" });
        state.micAudios.push(createPreviewItem(file, "audio", file.name));
        renderAllPreviews();
        if (micToggle) micToggle.classList.remove("recording");
        if (micStatus) micStatus.textContent = "";
        if (micToggle) micToggle.textContent = "Record audio";
      };
      state.micRecorder.start();
      if (micToggle) micToggle.classList.add("recording");
      if (micStatus) micStatus.textContent = "Recording...";
      if (micToggle) micToggle.textContent = "Stop recording";
    } catch (err) {
      console.error("Mic access failed:", err);
      if (micStatus) micStatus.textContent = "Mic error";
    }
  }

  if (micToggle) {
    micToggle.addEventListener("click", () => {
      toggleMic();
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
        updateAssistantMessage(assistantMsg, `Error: ${err.message || err}`);
      }
    } finally {
      assistantMsg.classList.remove("streaming");
      setStreamingState(false);
    }
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
