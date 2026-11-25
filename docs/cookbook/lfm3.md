# LFM3-Audio and LFM3-Omni

Liquid AI's [LFM3-Audio-2B](https://huggingface.co/LiquidAI/LFM3-Audio-2B) takes text and 16 kHz speech in and answers with text, 24 kHz speech (LiquidCodec, 12.5 frames per second), or interleaved text and speech. [LFM3-Omni-2B](https://huggingface.co/LiquidAI/LFM3-Omni-2B) adds image and video input. Both run as a `preprocessing → generation → decode` pipeline on one GPU, with the codec decode in its own process, and are served through `/v1/audio/speech` and `/v1/chat/completions`.

The checkpoints are released under the LFM Open License v1.0.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md), then download a model:

```bash
hf download LiquidAI/LFM3-Omni-2B --revision c01718a568e53ae936e6b5e1ff5ab2de7d7f388b
```

The checkpoints ship their own encoder and codec code, which the pipeline loads with `trust_remote_code`. `examples/configs/lfm3_omni.yaml` pins the checkpoint to the snapshot it was validated against, so the launch below takes `model_path` from the config.

## Server Configuration

```bash
sgl-omni serve \
  --config examples/configs/lfm3_omni.yaml \
  --port 8000
```

For LFM3-Audio, use `examples/configs/lfm3_audio.yaml` (`LiquidAI/LFM3-Audio-2B` at revision `2d6c1f99c129b3ef07a8d168bc0ff08e2bdb93f1`).

## Text to Speech

Requests follow [TTS Model Usage](../basic_usage/tts.md); this section covers LFM3's own fields. Both models have two built-in voices, `female` (default) and `male`. `language` sets the spoken accent and defaults to `American English`; `instructions` replaces the whole system prompt.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "LiquidAI/LFM3-Omni-2B",
    "voice": "male",
    "input": "The quick brown fox jumps over the lazy dog."
  }' \
  --output output.wav
```

`temperature`, `top_k` and `repetition_penalty` control speech-frame sampling; `temperature: 0` is greedy and `seed` makes a request reproducible. Unset fields take the checkpoint's defaults (LFM3-Omni: `0.4`, no top-k, `1.2`; LFM3-Audio: `1.0`, `128`, `1.1`).

## Chat

Ask for speech with `"modalities": ["text", "audio"]`. The model answers with interleaved text and speech; the response carries both the text and the waveform. `talker_temperature`, `talker_top_k` and `talker_repetition_penalty` control the speech frames.

```python
import base64
import requests

resp = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "LiquidAI/LFM3-Omni-2B",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "modalities": ["text", "audio"],
    },
).json()
message = resp["choices"][0]["message"]
print(message["content"])
with open("answer.wav", "wb") as f:
    f.write(base64.b64decode(message["audio"]["data"]))
```

Without `audio` in `modalities` the model answers in text only, without a reasoning block. `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty` and `seed` control the text; stop sequences are not supported.

Speech, images (LFM3-Omni) and videos (LFM3-Omni) go in the message content as `input_audio` / `audio_url`, `image_url` and `video_url` parts:

```python
image = base64.b64encode(open("photo.png", "rb").read()).decode()
messages = [{
    "role": "user",
    "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
        {"type": "video_url", "video_url": {"url": "https://example.com/clip.mp4"}},
        {"type": "text", "text": "Describe the photo and the clip."},
    ],
}]
```

Videos are sampled at the checkpoint's frame rate (2 fps, 4 to 384 frames) and read without their soundtrack; send the audio as its own part. Media URLs and local paths follow the server's media policy (allowed domains and local media path).

## Speech Sampling and Speed

Speech frames are drawn by one of two samplers, set with `speech_sampler` under `stages.generation.factory` in the server config:

- `seeded` (default): draws are hashed from the request seed and frame position, so one CUDA graph covers the whole frame. The same seed reproduces a request served alone; under concurrency, bf16 numerics that vary with the batch can change its speech, and so can a retraction, whose replay recomputes the request in one prefill. Seeded output differs from the checkpoint's own `generate`.
- `exact`: each request's generator draws in the checkpoint's order, so a seed reproduces the checkpoint's `generate`. Only the depth passes run in CUDA graphs.

With `seeded`, each request's speech state lives on the GPU, so the next decode step launches before the host reads the last one (`enable_async_decode`, on by default).

SeedTTS EN (1088 samples), LFM3-Omni-2B, one H200, with the default config (`seeded` rows: mean of two runs):

| Concurrency | Sampler | WER | Latency p50 | RTF | Throughput |
|---|---|---|---|---|---|
| 8 | exact | 0.98% | 1.06 s | 0.24 | 7.5 req/s |
| 8 | seeded | 0.91% | 0.51 s | 0.12 | 15.4 req/s |
| 32 | exact | 0.88% | 3.39 s | 0.78 | 9.3 req/s |
| 32 | seeded | 0.87% | 0.76 s | 0.18 | 40.7 req/s |
| 64 | seeded | 0.95% | 1.03 s | 0.25 | 58.1 req/s |

With `--generation.engine.max_running_requests 128`, concurrency 128 serves 69.6 req/s (0.92% WER, 1.65 s p50). LFM3-Audio-2B with `seeded`: 1.02% WER and 15.7 req/s at concurrency 8, 0.92% and 41.7 req/s at 32.

The codec decode runs in its own process, so it does not share the Python interpreter with the engine's scheduler loop: against the codec in the engine's process, on an H100, that served 14.9 vs 12.8 req/s at concurrency 8, 60.7 vs 46.2 at 64 and 78.7 vs 48.9 at 128. The generation stage reserves 70% of the GPU memory and the codec 15% (`gpu_memory_fraction` under `stages.generation` and `stages.decode`).

The engine runs up to 64 requests at once (`--generation.engine.max_running_requests`). New requests wait up to 30 ms (or until 32 are waiting) to prefill together: on an H100 with the codec in the engine's process it served about 46 instead of 40 req/s at concurrency 64 and cost about 3% at 8. `prefill_coalesce_requests` and `prefill_coalesce_wait_ms` under `stages.generation.factory` change it; `prefill_coalesce_requests: 0` turns it off.

## Limitations

- Decode runs in CUDA graphs; prefill runs eagerly. No streaming audio output or voice cloning yet.
- Chunked prefill and the radix cache are off, so each prompt is prefilled in one pass; the engine refuses settings it does not support at startup.
- Audio, image and video encoders run inside the generation stage's prefill, once per prefill batch.
