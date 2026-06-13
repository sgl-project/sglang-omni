# Voxtral TTS

[Voxtral-4B-TTS](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) is an open-weights
text-to-speech model from Mistral AI built on a Ministral-3B backbone. It generates lifelike
24 kHz speech with natural prosody across 9 languages and ships with a set of preset named
voices. In SGLang-Omni, Voxtral runs as a `preprocessing → tts_generation → vocoder` pipeline
and is served through the OpenAI-compatible `/v1/audio/speech` endpoint.


## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md), then install the Voxtral-specific tokenizer and download the model:

```bash
# Voxtral preprocessing uses Mistral's Tekken tokenizer from mistral-common.
uv pip install 'mistral_common[audio]>=1.11.0'

hf download mistralai/Voxtral-4B-TTS-2603
```

The model repository is public, so no Hugging Face token is required.

## Server Configuration

The pipeline is `preprocessing → tts_generation → vocoder`.
First startup can take several minutes while the `tts_generation` stage captures CUDA graphs.

```bash
sgl-omni serve \
  --model-path mistralai/Voxtral-4B-TTS-2603 \
  --config examples/configs/voxtral_tts.yaml \
  --port 8000
```

## Synthesizing Speech

### Zero-shot

With no voice specified, Voxtral falls back to its default voice (`cheerful_female`).

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "SGLang-Omni is a great project!"}' \
  --output output.wav
```

### Named Voices

Voxtral speaks with **preset named voices** (it does not clone from a reference clip). Select
one with the `voice` field:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Get the trust fund to the bank early.",
    "voice": "casual_male",
    "max_new_tokens": 4096
  }' \
  --output output.wav
```

The available voices ship inside the checkpoint as `voice_embedding/*.pt` files. List them
from your downloaded snapshot:

```bash
ls "$(hf download mistralai/Voxtral-4B-TTS-2603)/voice_embedding"
```

#### Python

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "input": "Get the trust fund to the bank early.",
        "voice": "casual_male",
        "max_new_tokens": 4096,
    },
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

### Streaming

Set `"stream": true` and `"response_format": "pcm"` to receive raw PCM audio
chunks in real time:

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Get the trust fund to the bank early.",
    "voice": "casual_male",
    "stream": true,
    "response_format": "pcm"
  }' \
  --output output.pcm
```

Streaming returns `audio/pcm` 16-bit mono PCM bytes with sample-rate metadata in
the response headers. See the [Higgs TTS cookbook](../cookbook/higgs_tts.md#streaming)
for a full Python raw PCM consumer.

## Request Parameters

| Parameter | Default | Notes |
|---|---|---|
| `input` | (required) | Text to synthesize |
| `voice` | `cheerful_female` | Preset voice name from the checkpoint's `voice_embedding/` directory |
| `max_new_tokens` | `4096` | Maximum number of generated acoustic tokens |
| `response_format` | `wav` | Output container (`wav`, `mp3`, `flac`, `opus`, `aac`, `pcm`) |
| `stream` | `false` | Stream raw PCM audio chunks |

> Voxtral generation is **deterministic**: the engine fixes `temperature` to `0.0`, so sampling
> parameters such as `top_p`, `top_k`, and `temperature` are not used. Reference-clip voice
> cloning (`references`) is **not** supported for Voxtral — use a preset `voice` instead.

## Benchmark Results

Seed-TTS EN (full set, 1088 utterances), bf16, `max_new_tokens=4096`,
`--no-ref-audio --voice cheerful_female`, concurrency 16, WER scored with HF
Whisper-large-v3. Hardware: 1× H200 SXM.

| Metric | Value |
|---|---|
| WER (corpus micro-avg) | 1.20% |
| WER (per-sample mean / median) | 1.22% / 0.00% |
| WER (per-sample p95 / max) | 9.09% / 42.86% |
| >50% WER samples | 0 / 1088 |
| Latency mean / median (s) | 2.94 / 2.86 |
| Latency p95 / p99 (s) | 4.56 / 5.37 |
| RTF mean / median | 0.519 / 0.541 |
| Output throughput (tok/s) | 383.7 |
| Throughput (req/s) | 5.40 |
| Completed / failed requests | 1088 / 0 |

Reproduce with the SeedTTS command documented in `benchmarks/README.md`. The Voxtral model card
also quotes ~70 ms first-audio latency at concurrency 1; the table above is a throughput-oriented
run at concurrency 16, so its RTF reflects batched load rather than the latency-optimized
single-stream figure. Output is 24 kHz.

## Known Limitations

- **Preset voices only.** Voxtral selects from named voices baked into the checkpoint; it does
  not clone an arbitrary speaker from a reference clip in this engine.
- **Deterministic decoding.** `temperature` is fixed at `0.0`; you cannot trade determinism for
  diversity through sampling parameters.
- **Language coverage.** Quality is tuned for the 9 supported languages (English, French,
  Spanish, German, Italian, Portuguese, Dutch, Arabic, Hindi).
- **Non-commercial license.** The weights are CC BY-NC 4.0; commercial use is not permitted.
