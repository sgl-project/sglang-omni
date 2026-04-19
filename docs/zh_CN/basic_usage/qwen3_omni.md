# Omni 模型使用指南

本指南以 [Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) 为例，演示如何在 SGLang-Omni 及兼容 OpenAI 的 API 中使用 Omni 模型。Qwen3-Omni 支持多模态输入（文本、图像、音频），并可根据模式选择仅输出文本或同时输出文本 + 音频。

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
```

## 纯文本模式 (Text-Only Mode)

纯文本模式在单张 GPU 上运行思考器（thinker）流水线。它接受多模态输入（文本、图像、音频）并且仅产生文本输出。

### 启动服务器

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --text-only \
  --port 8008
```

### 图像和文本输入

发送一张图像附带一个文本问题，以获取文本响应。

**cURL**

```bash
curl -X POST http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": "How many cars are there in the picture?"}],
    "images": ["tests/data/cars.jpg"],
    "modalities": ["text"],
    "max_tokens": 16
  }'
```

**Python**

```python
import requests

resp = requests.post(
    "http://localhost:8008/v1/chat/completions",
    json={
        "model": "qwen3-omni",
        "messages": [{"role": "user", "content": "How many cars are there in the picture?"}],
        "images": ["tests/data/cars.jpg"],
        "modalities": ["text"],
        "max_tokens": 16,
    },
)
resp.raise_for_status()
result = resp.json()
print(result["choices"][0]["message"]["content"])
```

### 音频和图像输入

将音频文件与图像一起发送。音频中包含口头提问 ("How many cars are there in the picture?")，模型将基于这两个输入作答。

> **注意：** 当所有的语义内容都来自音频、视频或图像而不是文本时，请在 user 消息中设置 `"content": ""` (空字符串)。

**cURL**

```bash
curl -X POST http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": ""}],
    "images": ["tests/data/cars.jpg"],
    "audios": ["tests/data/query_to_cars.wav"],
    "modalities": ["text"],
    "max_tokens": 16
  }'
```

**Python**

```python
import requests

resp = requests.post(
    "http://localhost:8008/v1/chat/completions",
    json={
        "model": "qwen3-omni",
        "messages": [{"role": "user", "content": ""}],
        "images": ["tests/data/cars.jpg"],
        "audios": ["tests/data/query_to_cars.wav"],
        "modalities": ["text"],
        "max_tokens": 16,
    },
)
resp.raise_for_status()
result = resp.json()
print(result["choices"][0]["message"]["content"])
```

## 语音模式 (Speech Mode)

语音模式跨多张 GPU 运行完整的 9 阶段流水线。它会同时产生文本输出（来自思考器 thinker）和音频输出（来自对话器 talker）。

### 启动服务器 (Speech Output)

语音模式需要多张 GPU。使用示例脚本来控制 GPU 分配：

```bash
python examples/run_qwen3_omni_speech_server.py \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --gpu-thinker 0 \
  --gpu-talker 1 \
  --gpu-code-predictor 1 \
  --gpu-code2wav 1 \
  --port 8008
```

或者使用不带 `--text-only` 参数的 CLI（默认为语音模式）：

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --port 8008
```

### 图像和文本输入 (Speech Output)

发送一张图像附带文本问题，以同时获取文本和音频响应。设置 `"modalities": ["text", "audio"]` 来启用音频输出。

**cURL**

```bash
curl -X POST http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": "How many cars are there in the picture?"}],
    "images": ["tests/data/cars.jpg"],
    "modalities": ["text", "audio"],
    "max_tokens": 16
  }'
```

**Python**

```python
import base64
import requests

resp = requests.post(
    "http://localhost:8008/v1/chat/completions",
    json={
        "model": "qwen3-omni",
        "messages": [{"role": "user", "content": "How many cars are there in the picture?"}],
        "images": ["tests/data/cars.jpg"],
        "modalities": ["text", "audio"],
        "max_tokens": 16,
    },
)
resp.raise_for_status()
result = resp.json()
choice = result["choices"][0]["message"]

print(choice["content"])

audio_data = base64.b64decode(choice["audio"]["data"])
with open("output.wav", "wb") as f:
    f.write(audio_data)
```

### 音频和图像输入 (Speech Output)

发送音频文件与图像。模型听到口头提问并看到图像后，将以文本和音频的形式同时作答。

**cURL**

```bash
curl -X POST http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": ""}],
    "images": ["tests/data/cars.jpg"],
    "audios": ["tests/data/query_to_cars.wav"],
    "modalities": ["text", "audio"],
    "max_tokens": 16
  }'
```

**Python**

```python
import base64
import requests

resp = requests.post(
    "http://localhost:8008/v1/chat/completions",
    json={
        "model": "qwen3-omni",
        "messages": [{"role": "user", "content": ""}],
        "images": ["tests/data/cars.jpg"],
        "audios": ["tests/data/query_to_cars.wav"],
        "modalities": ["text", "audio"],
        "max_tokens": 16,
    },
)
resp.raise_for_status()
result = resp.json()
choice = result["choices"][0]["message"]

print(choice["content"])

audio_data = base64.b64decode(choice["audio"]["data"])
with open("output.wav", "wb") as f:
    f.write(audio_data)
```

### 视频和音频输入

发送一段视频和一段口语音频问题。模型观看视频，听取问题，然后以文本和音频作答。

**cURL**

```bash
curl -X POST http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": ""}],
    "videos": ["tests/data/draw.mp4"],
    "audios": ["tests/data/query_to_draw.wav"],
    "modalities": ["text", "audio"],
    "max_tokens": 16
  }'
```

**Python**

```python
import base64
import requests

resp = requests.post(
    "http://localhost:8008/v1/chat/completions",
    json={
        "model": "qwen3-omni",
        "messages": [{"role": "user", "content": ""}],
        "videos": ["tests/data/draw.mp4"],
        "audios": ["tests/data/query_to_draw.wav"],
        "modalities": ["text", "audio"],
        "max_tokens": 16,
    },
)
resp.raise_for_status()
result = resp.json()
choice = result["choices"][0]["message"]

print(choice["content"])

audio_data = base64.b64decode(choice["audio"]["data"])
with open("output.wav", "wb") as f:
    f.write(audio_data)
```

## 请求参数 (Request Parameters)

下表列出了针对 Qwen3-Omni，`/v1/chat/completions` 端点所接受的所有参数。

| 参数 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| `model` | string | `null` | 模型标识符 |
| `messages` | list | (必填) | 聊天消息列表，每条包含 `role` 和 `content` |
| `modalities` | list | `["text"]` | 输出模态：`["text"]` 表示仅文本，`["text", "audio"]` 表示文本和音频 |
| `images` | list | `null` | 图像文件路径列表 (本地路径或 URL) |
| `audios` | list | `null` | 音频文件路径列表 (本地路径或 URL) |
| `videos` | list | `null` | 视频文件路径列表 (本地路径或 URL) |
| `max_tokens` | int | `null` | 生成的最大 token 数 |
| `temperature` | float | `null` | 采样温度 |
| `top_p` | float | `null` | Top-p 采样 |
| `top_k` | int | `null` | Top-k 采样 |
| `repetition_penalty` | float | `null` | 重复惩罚参数 |
| `seed` | int | `null` | 用于保证可重复性的随机种子 |
| `stream` | bool | `false` | 是否通过 SSE 启用流式传输 |
| `audio` | dict | `null` | 音频输出配置，例如 `{"voice": "default", "format": "wav"}` |
| `stage_sampling` | dict | `null` | 覆盖单阶段的采样参数，例如 `{"thinker": {"temperature": 0.8}}` |
