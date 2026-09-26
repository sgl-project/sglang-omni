// SPDX-License-Identifier: Apache-2.0
import { PCMPlayer, encodePCM, decodePCM } from "./audio.mjs";

const $ = (id) => document.getElementById(id);
let current = null;
let sequence = 0;

function controls(active, connecting = false) {
    $("start").disabled = active || connecting;
    $("stop").disabled = !active && !connecting;
    $("mute").disabled = !active;
}

function send(s, type, fields = {}) {
    if (s.ws?.readyState !== WebSocket.OPEN) return false;
    s.ws.send(
        JSON.stringify({ type, event_id: `ui-${++sequence}`, ...fields }),
    );
    return true;
}
async function start() {
    if (current) return;
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
        $("status").textContent = "请通过 localhost 或 HTTPS 打开，以使用麦克风。";
        return;
    }
    const s = {
        seq: 0,
        sentSamples: 0,
        completedSamples: 0,
        muted: false,
        connecting: true,
    };
    current = s;
    controls(false, true);
    $("status").textContent = "正在连接，请允许麦克风权限…";
    $("transcript").textContent = "";
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
        await s.captureContext.audioWorklet.addModule(
            "/voicechat-assets/capture-worklet.js",
        );
        if (current !== s) return;
        s.player = new PCMPlayer(s.outputContext);
        s.ws = new WebSocket(
            `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/v1/realtime`,
        );
        s.connectTimer = setTimeout(() => {
            if (current === s && s.connecting)
                end("连接超时，请检查服务是否已启动。");
        }, 15000);
        s.ws.onmessage = ({ data }) => {
            if (current !== s) return;
            try {
                handle(s, JSON.parse(data));
            } catch (error) {
                end(error.message);
            }
        };
        s.ws.onerror = () => {
            if (current === s)
                end("WebSocket 连接失败，请检查服务和端口转发。");
        };
        s.ws.onclose = () => {
            if (current === s) end("连接已关闭，可以重新开始。");
        };
        stream.getAudioTracks()[0].onended = () => {
            if (current === s) end("麦克风已断开，请检查设备后重新开始。");
        };
    } catch (error) {
        if (current !== s) return;
        const messages = {
            NotAllowedError:
                "没有获得麦克风权限。请在地址栏的网站权限中允许麦克风，再重试。",
            NotFoundError: "没有找到麦克风，请连接设备后重试。",
            NotReadableError: "无法读取麦克风，请检查设备是否被其他应用占用。",
        };
        await end(messages[error.name] || error.message);
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
        if (
            s.ws.bufferedAmount > 256000 ||
            s.sentSamples - s.completedSamples > 16000 * 9
        ) {
            end(
                "输入积压超过 9 秒，已停止本次对话以避免延迟继续增加。请重新开始。",
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
            s.sessionTimer = setTimeout(() => {
                if (current === s) end("已到 4 分钟上限，可重新开始。");
            }, 240000);
            beginCapture(s);
            controls(true);
            $("status").textContent = "对话中，可直接开口。";
            break;
        }
        case "response.output_audio.delta":
            s.player.enqueue(decodePCM(event.delta), 22050, event);
            break;
        case "response.output_audio_transcript.delta":
        case "response.output_text.delta":
            $("transcript").textContent += event.delta || "";
            break;
        case "sglang.unit.done": {
            const unit = Number(event.unit_id.replace("unit_", ""));
            if (Number.isFinite(unit))
                s.completedSamples = Math.min(s.sentSamples, (unit + 1) * 1280);
            break;
        }
        case "session.closed":
            end("对话已结束。");
            break;
        case "error":
            throw new Error(
                event.error?.code === "buffer_overflow"
                    ? "服务端音频积压过多，已停止本次对话。请稍后重试。"
                    : event.error?.message || "服务端发生错误",
            );
    }
}

async function end(message = "对话已结束。") {
    const s = current;
    if (!s) return;
    current = null;
    clearTimeout(s.connectTimer);
    clearTimeout(s.sessionTimer);
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
    $("status").textContent = message;
    $("mute").textContent = "静音麦克风";
    $("mute").setAttribute("aria-pressed", "false");
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
    $("status").textContent = s.muted ? "麦克风已静音，模型继续接收静音。" : "对话中，可直接开口。";
};
window.addEventListener("pagehide", () => {
    if (current) {
        current.stream?.getTracks().forEach((t) => t.stop());
        current.ws?.close();
    }
});
