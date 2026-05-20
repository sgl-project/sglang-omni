# Higgs TTS Model Usage

This guide uses [`boson-sglang/higgs-audio-v3-tts-4b-base`](https://huggingface.co/boson-sglang/higgs-audio-v3-tts-4b-base) — Higgs Audio v3 (Qwen3-4B backbone, 8 discrete codebooks × 1026 vocab, bf16) — with SGLang-Omni and the OpenAI-compatible API. The pipeline is `preprocessing → audio_encoder → tts_engine → vocoder`; the vocoder loads the public [`bosonai/higgs-audio-v2-tokenizer`](https://huggingface.co/bosonai/higgs-audio-v2-tokenizer) codec.

## Prerequisites

```bash
docker pull frankleeeee/sglang-omni:dev
docker run -it --shm-size 32g --gpus all frankleeeee/sglang-omni:dev /bin/zsh
```

```bash
git clone https://github.com/sgl-project/sglang-omni.git
cd sglang-omni
uv venv .venv -p 3.12 && source .venv/bin/activate
uv pip install -v .

# Higgs TTS model is private; export your HF token before downloading.
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
hf download boson-sglang/higgs-audio-v3-tts-4b-base
hf download bosonai/higgs-audio-v2-tokenizer
```

## Launch the Server

```bash
sgl-omni serve \
  --model-path boson-sglang/higgs-audio-v3-tts-4b-base \
  --config examples/configs/higgs_tts.yaml \
  --port 8000
```

The audio codec defaults to the public [`bosonai/higgs-audio-v2-tokenizer`](https://huggingface.co/bosonai/higgs-audio-v2-tokenizer) repo. Override per-stage if you need a different codec checkpoint:

```bash
sgl-omni serve \
  --model-path boson-sglang/higgs-audio-v3-tts-4b-base \
  --config examples/configs/higgs_tts.yaml \
  --stage-arg preprocessing.audio_codec_path=<path-or-repo-id> \
  --stage-arg vocoder.audio_codec_path=<path-or-repo-id> \
  --port 8000
```

## Use Curl

### Voice Cloning

Higgs TTS conditions on both reference audio **and** its transcript (`<|ref_text|>` segment); supplying the transcript materially improves quality versus audio-only cloning. The `references` field accepts `audio_path` (local path or HTTP URL) and `text` (transcript of that audio).

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Get the trust fund to the bank early.",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }],
    "temperature": 0.8,
    "top_k": 50,
    "max_new_tokens": 1024
  }' \
  --output output.wav
```

### Zero-shot

Without a reference, the model falls back to the `<|tts|> <|text|> ... <|audio|>` zero-shot prompt:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, how are you?"}' \
  --output output.wav
```

### Streaming

Set `"stream": true` to receive base64-encoded WAV chunks over Server-Sent Events:

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

## Use Python

### Voice Cloning

```python
import requests

REFERENCE_AUDIO = "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav"
REFERENCE_TEXT = "We asked over twenty different people, and they all said it was his."
SPEECH_INPUT = "Get the trust fund to the bank early."

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "input": SPEECH_INPUT,
        "references": [{"audio_path": REFERENCE_AUDIO, "text": REFERENCE_TEXT}],
        "temperature": 0.8,
        "top_k": 50,
        "max_new_tokens": 1024,
    },
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

### Pre-encoded Reference Codes

For high-throughput pipelines (e.g. RL rollout) where the same reference audio is reused across many requests, you can encode the reference audio offline and pass the discrete codes directly via `reference_codes` — this skips the server-side codec encode step. Shape must be `[T, num_codebooks=8]`.

```python
resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "input": SPEECH_INPUT,
        "reference_codes": codes_TN,  # [T, 8] int list, pre-delay-pattern
        "reference_text": REFERENCE_TEXT,
    },
)
```

## Request Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input` | string | (required) | Text to synthesize |
| `voice` | string | `"default"` | Voice identifier (ignored when `references` is set) |
| `response_format` | string | `"wav"` | Output audio format |
| `stream` | bool | `false` | Enable streaming via SSE |
| `references` | list | `null` | Reference audio for voice cloning; each item has `audio_path` (local path or HTTP URL) and `text` (transcript) |
| `reference_codes` | list[list[int]] | `null` | Pre-encoded discrete codes, shape `[T, 8]` — alternative to `references[0].audio_path` |
| `reference_text` | string | `null` | Transcript of reference audio when supplying `reference_codes` |
| `max_new_tokens` | int | `2048` | Maximum number of generated multi-codebook steps |
| `temperature` | float | `1.0` | Sampling temperature |
| `top_p` | float | `null` | Top-p sampling |
| `top_k` | int | `null` | Top-k sampling |
| `seed` | int | `null` | Random seed for reproducibility |

## Benchmark Results

On seed-tts en (1000 utterances) on a single A100 40GB, bf16, top_k=50, temp=0.8,
max_new_tokens=1024, scored with HF Whisper-large-v3 (fp32) for WER and
WavLM-large ECAPA-TDNN cosine similarity × 100:

| metric | value |
|---|---|
| avg WER | 0.0182 |
| avg speaker similarity | 64.81 |

Throughput (N=50/level, sequential thread pool):

| Concurrency | Mean Latency | RTF (per-req) | audio_s/s |
|---|---|---|---|
| 1 | 4637 ms | 0.526 | 1.90 |
| 16 | 7138 ms | 0.747 | 12.88 |
| 32 | 10188 ms | 0.865 | 16.94 |
