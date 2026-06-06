# Ming-Flash-Omni-2.0

[Ming-flash-omni-2.0](https://huggingface.co/inclusionAI/Ming-flash-omni-2.0) is a
multi-modal omni model from inclusionAI built on the Bailing MoE backbone. It accepts
text, image, audio, and video input and produces text-only or text + audio output. In
SGLang-Omni it runs as a multi-stage pipeline with a dedicated talker stage for speech
output and an optional streaming-TTS variant for sub-second time-to-first-audio.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md), then
download the model (≈200 GB of MoE weights):

```bash
hf download inclusionAI/Ming-flash-omni-2.0
```

The thinker tokenizer files live in a sibling repo; the loader falls back to it
automatically, but you can pre-download it to avoid a runtime fetch:

```bash
hf download inclusionAI/Ming-flash-omni-Preview
```

## Server Configuration

`sgl-omni serve` auto-detects the Bailing MoE architecture and selects the speech
pipeline by default. Add `--text-only` to launch the text-only pipeline.

### Speech Pipeline (text + audio output)

The 7-stage speech pipeline runs the thinker and talker on separate GPUs:

```bash
sgl-omni serve \
  --model-path inclusionAI/Ming-flash-omni-2.0 \
  --thinker-gpus 0 \
  --talker-gpu 1 \
  --port 8000
```

For a single-GPU host, offload thinker weights to CPU. ~200 GB of MoE weights need
substantial offload (the example launchers default to `--cpu-offload-gb 80`):

```bash
sgl-omni serve \
  --model-path inclusionAI/Ming-flash-omni-2.0 \
  --cpu-offload-gb 80 \
  --port 8000
```

For multi-GPU thinker tensor parallelism, pass `--thinker-tp-size` together with the
GPU ranks and keep the talker on a non-overlapping GPU:

```bash
sgl-omni serve \
  --model-path inclusionAI/Ming-flash-omni-2.0 \
  --thinker-tp-size 2 --thinker-gpus 0,1 \
  --talker-gpu 2 \
  --port 8000
```

The server validates that the talker GPU does not collide with the thinker TP range and
raises a clear error if it does.

### Text-Only Pipeline

The 6-stage text pipeline disables the talker and is suitable for understanding tasks
(audio/image/video → text):

```bash
sgl-omni serve \
  --model-path inclusionAI/Ming-flash-omni-2.0 \
  --text-only \
  --port 8000
```

### Streaming-TTS Pipeline

For sub-second time-to-first-audio, launch the 8-stage streaming pipeline directly via
the example script — it inserts a segmenter between the thinker and a streaming talker
so audio starts emitting from incremental text deltas instead of waiting for the full
response:

```bash
python examples/run_ming_omni_speech_server.py \
  --model-path inclusionAI/Ming-flash-omni-2.0 \
  --gpu-thinker 0 --gpu-talker 1 \
  --enable-streaming-tts \
  --port 8000
```

## Text Input

**cURL**

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ming-omni",
    "messages": [{"role": "user", "content": "你好，请介绍一下你自己。"}],
    "modalities": ["text"],
    "max_tokens": 256
  }'
```

**Python**

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "ming-omni",
        "messages": [{"role": "user", "content": "你好，请介绍一下你自己。"}],
        "modalities": ["text"],
        "max_tokens": 256,
    },
)
resp.raise_for_status()
print(resp.json()["choices"][0]["message"]["content"])
```

## Image + Text Input

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ming-omni",
    "messages": [{"role": "user", "content": "Briefly describe this image."}],
    "images": ["tests/data/cars.jpg"],
    "modalities": ["text"],
    "max_tokens": 64
  }'
```

Images can also be passed inline with the OpenAI multi-content format:

```json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "image_url", "image_url": {"url": "tests/data/cars.jpg"}},
      {"type": "text", "text": "Briefly describe this image."}
    ]
  }],
  "modalities": ["text"]
}
```

## Audio Input

When the user query is delivered as audio, leave `content` as an empty string and pass
the clip in `audios`:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ming-omni",
    "messages": [{"role": "user", "content": ""}],
    "audios": ["tests/data/query_to_cars.wav"],
    "modalities": ["text"],
    "max_tokens": 64
  }'
```

## Speech Output

Add `"audio"` to `modalities` to receive a text + audio response. This requires the
speech pipeline (omit `--text-only`):

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ming-omni",
    "messages": [{"role": "user", "content": "请给我讲一个故事。"}],
    "modalities": ["text", "audio"],
    "max_tokens": 256
  }'
```

A Python helper that saves the returned waveform to a WAV file is available in
`examples/run_ming_omni_speech.py`.

## Input / Output Modalities

| Input | Output | Speech server | Notes |
|---|---|---|---|
| Text | Text | No | — |
| Image + text | Text | No | — |
| Audio | Text | No | `content` must be `""` when the query is spoken |
| Image + audio | Text | No | `content` must be `""` when the query is spoken |
| Video | Text | No | `content` must be `""` when query comes from video |
| Video + text | Text | No | — |
| Text | Text + Audio | **Yes** | — |
| Image + text | Text + Audio | **Yes** | — |
| Audio | Text + Audio | **Yes** | `content` must be `""` when the query is spoken |

## Generation Parameters

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `model` | string | — | Model name (`ming-omni` by default) |
| `messages` | list | (required) | OpenAI chat-completion messages |
| `modalities` | list | `["text"]` | Add `"audio"` for speech output (speech server required) |
| `images` | list | `null` | Top-level image paths or URLs |
| `audios` | list | `null` | Top-level audio paths or URLs |
| `videos` | list | `null` | Top-level video paths or URLs |
| `max_tokens` | int | `null` | Max thinker tokens |
| `temperature` | float | model default | Sampling temperature |
| `top_p` | float | model default | Top-p sampling |
| `stream` | bool | `false` | SSE streaming |

## Voice Selection

The talker ships with a `voice_name.json` preset manifest under the talker checkpoint.
Selecting a non-default voice currently requires launching with the example script and
passing `--voice <id>`; the default is `DB30`.

```bash
python examples/run_ming_omni_speech_server.py \
  --model-path inclusionAI/Ming-flash-omni-2.0 \
  --gpu-thinker 0 --gpu-talker 1 \
  --voice DB30
```

## Known Limitations

- **`modalities: ["text", "audio"]` requires the speech server.** A text-only server
  silently drops the `"audio"` request — no error is raised, the response simply has no
  audio. Launch without `--text-only` for audio output.
- **Talker GPU must not overlap with the thinker TP range.** The config raises a
  `ValueError` at startup if they collide; pick a `--talker-gpu` outside the thinker
  GPU set.
- **`content` must be `""` when the user query lives entirely in `audios`, `videos`, or
  `images`.** Leaving a stale text query in `content` causes the model to process both.
- **Streaming-TTS variant is launched through the example script.** The 8-stage
  streaming pipeline (segmenter + streaming talker) is wired through
  `examples/run_ming_omni_speech_server.py --enable-streaming-tts`; the `sgl-omni
  serve` CLI exposes the 7-stage non-streaming speech pipeline.
- **Large weight footprint.** Ming-flash-omni-2.0 ships ≈200 GB of MoE weights. On a
  single GPU, `--cpu-offload-gb` is effectively required (the example launchers default
  to `80`).
