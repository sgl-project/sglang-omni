# TTS 模型使用指南

本指南以 [Fish Speech S2-Pro](https://huggingface.co/fishaudio/s2-pro) 为例，演示如何在 SGLang-Omni 及兼容 OpenAI 的 API 中使用 TTS (文本转语音) 模型。

## 准备工作

```bash
docker pull frankleeeee/sglang-omni:dev
docker run -it --shm-size 32g --gpus all frankleeeee/sglang-omni:dev /bin/zsh
```

```bash
git clone https://github.com/sgl-project/sglang-omni.git
cd sglang-omni
uv venv .venv -p 3.12 && source .venv/bin/activate
uv pip install -v .
hf download fishaudio/s2-pro
```

## 启动服务器

```bash
sgl-omni serve \
  --model-path fishaudio/s2-pro \
  --config examples/configs/s2pro_tts.yaml \
  --port 8000
```

## 使用 Curl 调用

在不提供任何参考音频的情况下，直接从文本生成语音：

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"input": "Hello, how are you?"}' \
    --output output.wav
```

请注意，如果没有提供参考音频，生成的语音听起来会比较像机器发音。为了获得更自然的人声效果，请使用包含参考音频片段的**声音克隆 (Voice Cloning)** 功能。

### 声音克隆 (Voice Cloning)

下面的示例使用了来自 [`seed-tts-eval-mini`](https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini) 的样本片段。`references` 字段接受 `audio_path` (本地路径或 HTTP URL) 和 `text` (该音频的文本转录)。

1. 非流式请求

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Get the trust fund to the bank early.",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }]
  }' \
  --output output.wav
```

2. 流式请求 (Streaming)

启用流式传输，以通过服务器发送事件 (SSE) 实时接收音频块。只需设置 `"stream": true`：

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Get the trust fund to the bank early.",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }],
    "stream": true
  }'
```

服务器会返回一连串的 SSE 事件。每个事件包含一个 `audio.speech.chunk` 对象，里面是 base64 编码的音频块数据。数据流以 `data: [DONE]` 结束。

## 使用 Python 调用

### 基础 TTS

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={"input": "Hello, how are you?"},
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

### 声音克隆 (Python)

```python
REFERENCE_AUDIO = "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav"
REFERENCE_TEXT = "We asked over twenty different people, and they all said it was his."
SPEECH_INPUT = "Get the trust fund to the bank early."
```

1. 非流式请求

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "input": SPEECH_INPUT,
        "references": [{"audio_path": REFERENCE_AUDIO, "text": REFERENCE_TEXT}],
    },
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

2. 流式请求

```python
import base64, io, json, wave

import requests

payload = {
    "input": SPEECH_INPUT,
    "references": [{"audio_path": REFERENCE_AUDIO, "text": REFERENCE_TEXT}],
    "stream": True,
    "response_format": "wav",
}

chunks = []
fmt = None
with requests.post(
    "http://localhost:8000/v1/audio/speech",
    json=payload,
    stream=True,
    timeout=600,
) as stream:
    stream.raise_for_status()
    for line in stream.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data = line[len("data:"):].lstrip()
        if data == "[DONE]":
            break
        b64 = (json.loads(data).get("audio") or {}).get("data")
        if not b64:
            continue
        with wave.open(io.BytesIO(base64.b64decode(b64)), "rb") as w:
            if fmt is None:
                fmt = w.getnchannels(), w.getsampwidth(), w.getframerate()
            chunks.append(w.readframes(w.getnframes()))

assert fmt
nc, sw, fr = fmt
with wave.open("output_stream.wav", "wb") as w:
    w.setnchannels(nc)
    w.setsampwidth(sw)
    w.setframerate(fr)
    w.writeframes(b"".join(chunks))
```

## 请求参数 (Request Parameters)

下表列出了 `/v1/audio/speech` 端点接受的所有参数。

| 参数 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| `input` | string | (必填) | 要合成的文本 |
| `voice` | string | `"default"` | 声音标识符 |
| `response_format` | string | `"wav"` | 输出音频格式 |
| `speed` | float | `1.0` | 播放速度倍率 |
| `stream` | bool | `false` | 是否通过 SSE 启用流式传输 |
| `references` | list | `null` | 用于声音克隆的参考音频；每项包含 `audio_path` (本地路径 / 远程 URL) 和 `text` |
| `max_new_tokens` | int | `null` | 生成的最大 token 数 |
| `temperature` | float | `null` | 采样温度 |
| `top_p` | float | `null` | Top-p 采样 |
| `top_k` | int | `null` | Top-k 采样 |
| `repetition_penalty` | float | `null` | 重复惩罚参数 |
| `seed` | int | `null` | 用于保证可重复性的随机种子 |

## 交互式测试台 (Interactive Playground)

SGLang-Omni 自带了一个基于 Gradio 的测试台，用于进行交互式的 TTS 实验：

```bash
./playground/tts/start.sh
```

该测试台现在针对同一个 S2 Pro 后端提供了两种演示模式：

- `Non-Streaming` (非流式)：发起标准请求，并在生成完成后展示最终的 WAV 文件。
- `Streaming` (流式)：消费 `/v1/audio/speech` 的 SSE 流，从增量式的 WAV 数据块开始播放，同时还会写入一个最终合并后的 WAV 产物供检查。

启动脚本会首先启动后端，等待 `/health` 健康检查通过后，再通过以下命令启动 Gradio UI：

```bash
python -m playground.tts.app --api-base http://localhost:8000
```

你可以在 [这里](https://x.com/lmsysorg/status/2031412267213008984/video/1) 观看演示视频。**我们强烈建议使用此测试台**，因为在命令行界面 (CLI) 中很难直接与音频数据进行交互。
