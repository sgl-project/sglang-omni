// SPDX-License-Identifier: Apache-2.0
import { PCMPlayer, encodePCM, decodePCM } from "./audio.mjs";

const $ = (id) => document.getElementById(id);
let current = null,
    ready = false,
    sequence = 0;
const replies = new Map();
const logs = [];
function log(text) {
    logs.push(`${new Date().toLocaleTimeString()}  ${text}`);
    if (logs.length > 40) logs.shift();
    $("events").textContent = logs.join("\n");
    $("events").scrollTop = $("events").scrollHeight;
}
function notice(message = "") {
    $("notice").textContent = message;
    $("notice").hidden = !message;
}
function status(text, kind = "") {
    $("status").className = `status ${kind}`;
    $("status").replaceChildren(
        Object.assign(document.createElement("i"), {}),
        document.createTextNode(text),
    );
}
function controls(active, connecting = false) {
    $("start").disabled = active || connecting || !ready;
    $("stop").disabled = !active && !connecting;
    $("mute").disabled = !active;
    $("microphone").disabled = active || connecting;
}
function send(s, type, fields = {}) {
    if (s.ws?.readyState !== WebSocket.OPEN) return false;
    s.ws.send(
        JSON.stringify({ type, event_id: `ui-${++sequence}`, ...fields }),
    );
    return true;
}
function addText(event) {
    const id = event.response_id;
    let entry = replies.get(id);
    if (!entry) {
        $("empty")?.remove();
        const box = document.createElement("article");
        box.className = "message";
        const head = document.createElement("div");
        head.className = "message-head";
        head.textContent = "VOICECHAT";
        const time = document.createElement("time");
        time.textContent = new Date().toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
        });
        head.append(time);
        const text = document.createElement("p");
        const tail = document.createElement("span");
        tail.className = "message-status";
        box.append(head, text, tail);
        $("transcript").append(box);
        entry = { text, tail };
        replies.set(id, entry);
    }
    const nearBottom =
        $("transcript").scrollHeight -
            $("transcript").scrollTop -
            $("transcript").clientHeight <
        110;
    entry.text.textContent += event.delta || "";
    $("copy").disabled = false;
    if (nearBottom) $("transcript").scrollTop = $("transcript").scrollHeight;
}
async function devices() {
    const items = await navigator.mediaDevices.enumerateDevices();
    const selected = $("microphone").value;
    $("microphone").replaceChildren(new Option("系统默认麦克风", ""));
    for (const item of items.filter(
        (d) => d.kind === "audioinput" && d.deviceId !== "default",
    )) {
        $("microphone").append(
            new Option(item.label || "麦克风", item.deviceId),
        );
    }
    if ([...$("microphone").options].some((o) => o.value === selected))
        $("microphone").value = selected;
}
async function checkService() {
    if (current) return;
    try {
        const response = await fetch("/v1/realtime/capabilities", {
            signal: AbortSignal.timeout(5000),
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const caps = await response.json();
        if (
            !caps.native_full_duplex ||
            caps.input_audio_format?.rate !== 16000 ||
            caps.output_audio_format?.rate !== 22050
        )
            throw new Error("当前服务不是 VoiceChat 原生全双工配置");
        ready = true;
        controls(false);
        status("服务就绪");
        $("connection").textContent =
            `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/v1/realtime`;
    } catch (error) {
        ready = false;
        controls(false);
        status("等待服务", "error");
        $("connection").textContent =
            `服务暂不可用：${error.message}。请检查服务或端口转发。`;
    }
}

async function start() {
    if (current || !ready) return;
    notice();
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
        notice(
            "麦克风需要安全页面。请通过 http://localhost:8097 打开，或使用 HTTPS。",
        );
        return;
    }
    const s = {
        seq: 0,
        sentSamples: 0,
        completedSamples: 0,
        muted: false,
        level: 0,
        started: 0,
        first: false,
        connecting: true,
    };
    current = s;
    controls(false, true);
    status("正在连接");
    $("call-title").textContent = "正在准备你的麦克风";
    $("call-hint").textContent = "请允许浏览器使用麦克风。";
    $("first-audio").textContent = "—";
    $("backlog").textContent = "—";
    $("buffer").textContent = "—";
    try {
        // Keep packet boundaries at the model rate; resample the continuous mix
        // to the device rate instead of resampling each small source separately.
        try {
            s.outputContext = new AudioContext({
                sampleRate: 22050,
                latencyHint: "interactive",
            });
        } catch {
            s.outputContext = new AudioContext({ latencyHint: "interactive" });
        }
        try {
            s.captureContext = new AudioContext({
                sampleRate: 16000,
                latencyHint: "interactive",
            });
        } catch {
            s.captureContext = new AudioContext({ latencyHint: "interactive" });
        }
        await Promise.all([
            s.outputContext.resume(),
            s.captureContext.resume(),
        ]);
        if (current !== s) return;
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                deviceId: $("microphone").value
                    ? { exact: $("microphone").value }
                    : undefined,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
            },
        });
        if (current !== s) {
            stream.getTracks().forEach((track) => track.stop());
            return;
        }
        s.stream = stream;
        await devices();
        await s.captureContext.audioWorklet.addModule(
            "/voicechat-assets/capture-worklet.js",
        );
        if (current !== s) return;
        s.player = new PCMPlayer(s.outputContext, () => {});
        s.ws = new WebSocket(
            `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/v1/realtime`,
        );
        s.connectTimer = setTimeout(() => {
            if (current === s && s.connecting)
                end("连接超时，请检查服务是否已启动。", true);
        }, 15000);
        s.ws.onmessage = ({ data }) => {
            if (current !== s) return;
            try {
                handle(s, JSON.parse(data));
            } catch (error) {
                end(error.message, true);
            }
        };
        s.ws.onerror = () => {
            if (current === s)
                end("WebSocket 连接失败，请检查服务和端口转发。", true);
        };
        s.ws.onclose = () => {
            if (current === s) end("连接已关闭，可以重新开始。");
        };
        stream.getAudioTracks()[0].onended = () => {
            if (current === s) end("麦克风已断开，请重新选择设备。", true);
        };
    } catch (error) {
        if (current !== s) return;
        const messages = {
            NotAllowedError:
                "没有获得麦克风权限。请在地址栏的网站权限中允许麦克风，再重试。",
            NotFoundError: "没有找到麦克风，请连接设备后重试。",
            NotReadableError: "无法读取麦克风，请检查设备是否被其他应用占用。",
        };
        await end(messages[error.name] || error.message, true);
    }
}

function beginCapture(s) {
    s.node = new AudioWorkletNode(s.captureContext, "voicechat-capture", {
        processorOptions: { targetRate: 16000, frameSamples: 1280 },
    });
    s.source = s.captureContext.createMediaStreamSource(s.stream);
    s.silent = s.captureContext.createGain();
    s.silent.gain.value = 0;
    s.source.connect(s.node);
    s.node.connect(s.silent);
    s.silent.connect(s.captureContext.destination);
    s.node.port.onmessage = ({ data }) => {
        if (current !== s || s.connecting) return;
        s.level = data.level;
        if (
            s.ws.bufferedAmount > 256000 ||
            s.sentSamples - s.completedSamples > 16000 * 9
        ) {
            end(
                "输入积压超过 9 秒，已停止本次对话以避免延迟继续增加。请重新开始。",
                true,
            );
            return;
        }
        if (
            send(s, "input_audio_buffer.append", {
                audio: encodePCM(data.pcm),
                sglang: {
                    seq: s.seq,
                    t_start_ms: (s.sentSamples * 1000) / 16000,
                },
            })
        ) {
            s.seq++;
            s.sentSamples += data.pcm.length;
        }
    };
}

function handle(s, event) {
    switch (event.type) {
        case "session.created":
            log("连接已建立，正在协商音频格式");
            send(s, "session.update", {
                session: {
                    output_modalities: ["audio"],
                    audio: {
                        input: {
                            format: { type: "audio/pcm", rate: 16000 },
                            turn_detection: null,
                        },
                        output: { format: { type: "audio/pcm", rate: 22050 } },
                    },
                },
            });
            break;
        case "session.updated": {
            const grant = event.session.sglang.granted;
            if (
                event.session.audio?.input?.format?.rate !== 16000 ||
                event.session.audio?.output?.format?.rate !== 22050 ||
                !grant.output_modalities.includes("audio")
            )
                throw new Error("服务器协商的音频格式与页面不一致");
            if (!s.connecting) break;
            clearTimeout(s.connectTimer);
            s.connecting = false;
            s.started = performance.now();
            s.limit = 240;
            beginCapture(s);
            controls(true);
            status("对话中", "active");
            $("stage").classList.add("active");
            $("call-title").textContent = "我在听，请直接说话";
            $("call-hint").textContent =
                "你可以随时开口，模型会持续接收你的声音。";
            $("mic-label").textContent = "正在收音";
            $("speaker-label").textContent = "等待回复";
            $("transcript-state").textContent = "LIVE";
            log(
                `开始收音 · 16 kHz → 22.05 kHz · 浏览器播放 ${s.outputContext.sampleRate / 1000} kHz`,
            );
            break;
        }
        case "response.output_audio.delta":
            if (!s.first) {
                s.first = true;
                $("first-audio").textContent =
                    `${((performance.now() - s.started) / 1000).toFixed(2)} s`;
            }
            s.player.enqueue(decodePCM(event.delta), 22050, event);
            break;
        case "response.output_audio_transcript.delta":
        case "response.output_text.delta":
            addText(event);
            break;
        case "response.done": {
            const item = replies.get(event.response.id);
            if (item && event.response.status === "cancelled")
                item.tail.textContent = "已打断";
            break;
        }
        case "sglang.unit.done": {
            const unit = Number(event.unit_id.replace("unit_", ""));
            if (Number.isFinite(unit))
                s.completedSamples = Math.min(s.sentSamples, (unit + 1) * 1280);
            break;
        }
        case "session.closed":
            end(
                event.reason === "session_timeout"
                    ? "本次 4 分钟会话已结束，点击开始可重新对话。"
                    : "对话已结束。",
            );
            break;
        case "error":
            throw new Error(
                event.error?.code === "buffer_overflow"
                    ? "服务端音频积压过多，已停止本次对话。请稍后重试。"
                    : event.error?.message || "服务端发生错误",
            );
    }
}

async function end(message = "", isError = false) {
    const s = current;
    if (!s) return;
    current = null;
    clearTimeout(s.connectTimer);
    if (s.node) {
        s.node.port.onmessage = null;
        s.node.disconnect();
        s.node.port.close();
    }
    s.source?.disconnect();
    s.silent?.disconnect();
    s.stream?.getTracks().forEach((track) => {
        track.onended = null;
        track.stop();
    });
    s.player?.clear();
    let socketClosed = Promise.resolve();
    if (s.ws) {
        if (s.ws.readyState === WebSocket.OPEN) {
            send(s, "session.close");
            // Allow the server to acknowledge cleanup, while bounding a stalled close.
            socketClosed = new Promise((resolve) => {
                const ws = s.ws;
                const timer = setTimeout(() => {
                    ws.close();
                    resolve();
                }, 4000);
                ws.onmessage = ({ data }) => {
                    try {
                        if (JSON.parse(data).type === "session.closed") {
                            clearTimeout(timer);
                            ws.close();
                            resolve();
                        }
                    } catch {
                        ws.close();
                        resolve();
                    }
                };
                ws.onclose = () => {
                    clearTimeout(timer);
                    resolve();
                };
            });
        } else s.ws.close();
    }
    await Promise.allSettled([
        socketClosed,
        s.captureContext?.close(),
        s.player ? s.player.close() : s.outputContext?.close(),
    ]);
    controls(false);
    status(isError ? "对话已停止" : "服务就绪", isError ? "error" : "");
    $("stage").classList.remove("active");
    $("stage").querySelector(".orb").style.transform = "";
    $("call-title").textContent = isError
        ? "本次对话已停止，请重新开始"
        : "随时，再聊一会儿";
    $("call-hint").textContent = "这段对话已结束，点击开始可以开启新的会话。";
    $("mic-label").textContent = "未开启";
    $("speaker-label").textContent = "已停止";
    $("transcript-state").textContent = "TRANSCRIPT";
    $("mute").textContent = "静音麦克风";
    $("mute").setAttribute("aria-pressed", "false");
    $("backlog").textContent = "—";
    $("buffer").textContent = "—";
    notice(message);
    log(message || "对话已结束");
}

$("start").onclick = start;
$("stop").onclick = () => end("对话已结束。");
$("mute").onclick = () => {
    const s = current;
    if (!s?.node) return;
    s.muted = !s.muted;
    s.node.port.postMessage({ muted: s.muted });
    $("mute").textContent = s.muted ? "恢复麦克风" : "静音麦克风";
    $("mute").setAttribute("aria-pressed", String(s.muted));
    $("mic-label").textContent = s.muted ? "已静音" : "正在收音";
    $("call-title").textContent = s.muted
        ? "麦克风已静音，继续听它说"
        : "我在听，请直接说话";
};
$("copy").onclick = async () => {
    try {
        await navigator.clipboard.writeText(
            [...replies.values()].map((e) => e.text.textContent).join("\n\n"),
        );
        $("copy").textContent = "已复制";
        setTimeout(() => {
            $("copy").textContent = "复制回复";
        }, 1500);
    } catch {
        notice("无法自动复制，请选中文字后复制。");
    }
};
window.addEventListener("pagehide", () => {
    if (current) {
        current.stream?.getTracks().forEach((t) => t.stop());
        current.ws?.close();
    }
});

const levels = { mic: new Array(44).fill(0), speaker: new Array(44).fill(0) };
const sound = new Float32Array(256);
function draw(id, history, value, color) {
    history.push(Math.min(1, value * 7));
    history.shift();
    const canvas = $(id),
        ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const step = canvas.width / history.length;
    ctx.fillStyle = color;
    history.forEach((v, i) => {
        const height = 2 + v * 36;
        ctx.beginPath();
        ctx.roundRect(
            i * step + 2,
            (canvas.height - height) / 2,
            step - 5,
            height,
            2,
        );
        ctx.fill();
    });
}
setInterval(() => {
    const s = current;
    let outputLevel = 0;
    if (s?.player) {
        s.player.analyser.getFloatTimeDomainData(sound);
        outputLevel = Math.sqrt(
            sound.reduce((sum, x) => sum + x * x, 0) / sound.length,
        );
    }
    draw("mic-wave", levels.mic, s?.muted ? 0 : s?.level || 0, "#78a58b");
    draw("speaker-wave", levels.speaker, outputLevel, "#c2b079");
    if (!s || s.connecting) return;
    const elapsed = Math.floor((performance.now() - s.started) / 1000);
    $("duration").textContent =
        `${String(Math.floor(elapsed / 60)).padStart(2, "0")}:${String(elapsed % 60).padStart(2, "0")} / 04:00`;
    $("backlog").textContent =
        `${Math.max(0, (s.sentSamples - s.completedSamples) / 16000).toFixed(2)} s`;
    $("buffer").textContent = `${s.player.queuedSeconds.toFixed(2)} s`;
    $("speaker-label").textContent =
        outputLevel > 0.005 ? "正在播放" : "听你说话";
    $("stage").querySelector(".orb").style.transform =
        `scale(${1 + Math.min(0.08, outputLevel * 0.7)})`;
    if (elapsed >= s.limit) end("本次会话已到时间，可以重新开始。");
}, 80);
checkService();
setInterval(() => {
    if (!ready && !current) checkService();
}, 5000);
