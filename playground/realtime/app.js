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
  const remoteAudioEl = $("remote-audio");

  let pc = null;
  let dc = null;
  let localStream = null;
  let sessionId = null;

  function getApiBase() {
    const globalBase =
      typeof window !== "undefined" && window.SGLANG_OMNI_API_BASE
        ? String(window.SGLANG_OMNI_API_BASE).trim()
        : "";
    return (globalBase || "http://localhost:8000").replace(/\/$/, "");
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

  async function connect() {
    if (pc) return;
    connectBtn.disabled = true;
    setStatus("Requesting media...");

    localStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
      video: cameraEl.checked,
    });

    pc = new RTCPeerConnection();
    dc = pc.createDataChannel("events");
    dc.addEventListener("open", () => log("data channel open"));
    dc.addEventListener("message", (event) => {
      try {
        log("server event", JSON.parse(event.data));
      } catch (err) {
        log("server message", { raw: event.data });
      }
    });

    const remoteStream = new MediaStream();
    remoteAudioEl.srcObject = remoteStream;
    pc.addEventListener("track", (event) => {
      if (event.track.kind === "audio") {
        remoteStream.addTrack(event.track);
      }
    });
    pc.addEventListener("connectionstatechange", () => {
      log("peer connection state", { state: pc.connectionState });
      setStatus(`RTC ${pc.connectionState}`);
    });

    localStream.getTracks().forEach((track) => pc.addTrack(track, localStream));

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    const response = await fetch(`${getApiBase()}/v1/realtime/webrtc/offer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sdp: offer.sdp,
        type: offer.type,
        instructions: instructionsEl.value.trim(),
        output_text: Boolean(textOutputEl.checked),
      }),
    });
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(detail || `Offer failed with ${response.status}`);
    }

    const answer = await response.json();
    sessionId = answer.session_id;
    await pc.setRemoteDescription(answer);
    setStatus(`Connected (${sessionId})`);
    log("session created", answer);

    disconnectBtn.disabled = false;
  }

  async function disconnect() {
    disconnectBtn.disabled = true;
    setStatus("Disconnecting...");

    try {
      if (sessionId) {
        await fetch(
          `${getApiBase()}/v1/realtime/sessions/${encodeURIComponent(sessionId)}`,
          {
          method: "DELETE",
          }
        );
      }
    } catch (_) {
      // Best-effort cleanup.
    }

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
    sessionId = null;
    remoteAudioEl.srcObject = null;
    connectBtn.disabled = false;
    setStatus("Idle");
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
})();
