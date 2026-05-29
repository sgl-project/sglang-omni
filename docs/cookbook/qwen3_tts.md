# Qwen3 TTS

[Qwen3-TTS-12Hz-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) is a discrete
multi-codebook text-to-speech model from the Qwen team. It performs fast voice cloning from a
short reference clip, supports 10 languages, and streams 24 kHz speech with low latency. The
`12Hz` in the name refers to the codec **frame rate** (12 acoustic frames per second), not the
playback sample rate. SGLang-Omni serves two checkpoints — `0.6B` and `1.7B` — through the same
`preprocessing → tts_engine → vocoder` pipeline and the OpenAI-compatible `/v1/audio/speech`
endpoint.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md).

Qwen3-TTS Base uses the upstream `qwen-tts` package, which currently pins Transformers 4.57.3.
Install it only in environments that serve Qwen3-TTS:

```bash
uv pip install transformers==4.57.3 accelerate==1.12.0 sox einops
uv pip install --no-deps qwen-tts==0.1.1
```

> Do **not** add `--upgrade` here. It pulls a newer `torch`/`numpy`/CUDA stack and breaks
> inference (mismatched cuDNN, `numba` requires NumPy ≤ 2.3). Pin only what is listed above so
> the image's existing `torch` build is left untouched.

Download a checkpoint (both repositories are public, no token required):

```bash
hf download Qwen/Qwen3-TTS-12Hz-0.6B-Base
hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base
```

## Server Configuration

The pipeline is `preprocessing → tts_engine → vocoder`.

```bash
# 0.6B
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --config examples/configs/qwen3_tts_0_6b.yaml \
  --port 8000
```

```bash
# 1.7B
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --config examples/configs/qwen3_tts_1_7b.yaml \
  --port 8000
```

## Synthesizing Speech

### Zero-shot


Qwen3-TTS does not support zero-shot synthesis.

### Voice Cloning

The `references` field accepts `audio_path` (a local path or HTTP URL) and `text` (the
transcript of that clip). Supplying the transcript enables in-context-learning (ICL) mode and
materially improves cloning quality; omitting it falls back to speaker-embedding (x-vector)
mode.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "SGLang-Omni is a great project!",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }]
  }' \
  --output output.wav
```

`ref_audio` and `ref_text` are accepted as shorthand for `references[0].audio_path` and
`references[0].text`.

#### Python

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "input": "Get the trust fund to the bank early.",
        "references": [{
            "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
            "text": "We asked over twenty different people, and they all said it was his.",
        }],
    },
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

### Language Hint

`language` biases the model toward a target language. It defaults to `auto` (let the model
detect). Supported languages are Chinese, English, Japanese, Korean, German, French, Russian,
Portuguese, Spanish, and Italian.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "今天天气不错，就该出去晒晒太阳。",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }],
    "language": "Chinese"
  }' \
  --output output.wav
```

### Streaming

Set `"stream": true` to receive audio chunks in real time over Server-Sent Events (SSE):

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

Each event carries a base64-encoded audio chunk; the stream ends with `data: [DONE]`. See the
[Higgs TTS cookbook](../cookbook/higgs_tts.md#streaming) for a full Python SSE consumer.

## Generation Parameters

| Parameter | Default | Notes |
|---|---|---|
| `input` | (required) | Text to synthesize |
| `references` | `null` | Reference clip for cloning; each item has `audio_path` and `text` |
| `ref_audio` / `ref_text` | `null` | Shorthand for `references[0].audio_path` / `references[0].text` |
| `language` | `auto` | Target-language hint (see list above) |
| `temperature` | `0.9` | Sampling temperature |
| `top_p` | `1.0` | Top-p sampling |
| `top_k` | `50` | Top-k sampling |
| `repetition_penalty` | `1.05` | Repetition penalty |
| `max_new_tokens` | `2048` | Maximum number of generated codec tokens |
| `seed` | `null` | Random seed for reproducibility |
| `stream` | `false` | Stream audio chunks over SSE |

## Model Variants

| Checkpoint | Parameters | Config |
|---|---|---|
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | 0.6B | `examples/configs/qwen3_tts_0_6b.yaml` |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | 1.7B | `examples/configs/qwen3_tts_1_7b.yaml` |

Both expose an identical request API. The 1.7B model has higher capacity (typically better
quality) at a larger memory and latency cost; the 0.6B model is lighter and faster.

## Benchmark Results

Qwen3-TTS-12Hz-0.6B-Base on Seed-TTS EN (1088 utterances, reference voice cloning from each
prompt), concurrency 16, WER scored with HF Whisper-large-v3. Hardware: 1× H200 SXM.

| Metric | Value |
|---|---|
| WER (corpus, excl. runaway outliers) | 1.07% |
| WER (per-sample median / p95) | 0.00% / 9.09% |
| WER (corpus micro-avg, raw) | 18.29% |
| Runaway samples (>50% WER) | 2 / 1088 (0.2%) |
| Latency mean / median (s) | 6.61 / 6.24 |
| RTF mean / median | 1.51 / 1.48 |
| Output throughput (tok/s) | 115.4 |
| Completed / failed requests | 1088 / 0 |

Typical output is clean (0.00% median WER, 9.09% p95). Two utterances (0.2%) ran away into a
repetition loop and generated ~164 s of looping audio up to `max_new_tokens`, which alone lifts
the raw micro-average to 18.29%; excluding those, corpus WER is 1.07%. RTF > 1 reflects the
0.6B codec pipeline at concurrency 16, not single-stream latency. The 1.7B checkpoint trades
latency for quality.

## Known Limitations

- **Reference audio recommended.** As a cloning model, Qwen3-TTS Base produces robotic speech
  without a reference clip.
- **Transcript improves cloning.** Providing `text` in `references` (ICL mode) yields better
  speaker similarity than speaker-embedding-only (x-vector) mode.
- **Language detection.** `language: auto` may misdetect for short or code-switched inputs;
  set `language` explicitly when you know the target language.
- **Rare runaway generation.** Roughly 0.2% of utterances (observed on the 0.6B checkpoint) can
  fall into a repetition loop and keep generating up to `max_new_tokens`. Raising
  `repetition_penalty` (default `1.05`) or lowering `max_new_tokens` mitigates it; the 1.7B
  checkpoint is less prone.
