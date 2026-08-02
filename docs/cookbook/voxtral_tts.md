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

### RTX 4090 24 GB Profile

For a single RTX 4090-class 24 GB GPU, use the experimental profile bounded to
concurrency 1, the only consumer-GPU shape exercised so far:

```bash
sgl-omni serve \
  --model-path mistralai/Voxtral-4B-TTS-2603 \
  --config examples/configs/voxtral_tts_4090_24gb.yaml \
  --port 8000
```

The profile keeps BF16 generation, the default `mem_fraction_static=0.85`, CUDA
Graphs, and `torch.compile`, but caps `max_running_requests` at 1. The generation
builder derives matching CUDA Graph and compile batch caps, avoiding capture for
unqualified concurrency shapes.

The audio tokenizer uses `flash_attn` when that package is importable and
otherwise falls back to PyTorch SDPA. The RTX 4090D run exercised the SDPA
fallback. That run is functional evidence for SM89 and 24 GB capacity, not
qualification of a standard RTX 4090 or higher concurrency.

## Synthesizing Speech

### Preset Voice

Voxtral ships preset voices with the checkpoint. Use `cheerful_female` for the
default preset voice.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Voxtral-4B-TTS-2603",
    "voice": "cheerful_female",
    "input": "SGLang-Omni is a great project!"
  }' \
  --output output.wav
```

### Named Voices

Voxtral speaks with **preset named voices** (it does not clone from a reference clip). Select
one with the `voice` field:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Voxtral-4B-TTS-2603",
    "voice": "casual_male",
    "input": "Get the trust fund to the bank early.",
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
        "model": "mistralai/Voxtral-4B-TTS-2603",
        "voice": "casual_male",
        "input": "Get the trust fund to the bank early.",
        "max_new_tokens": 4096,
    },
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

### PCM Output and Stream Compatibility

Set `"response_format": "pcm"` to receive raw 24 kHz, 16-bit mono PCM bytes.
`"stream": true` is accepted for API compatibility, but the current Voxtral
vocoder does not flush incremental chunks: the response arrives when synthesis
finishes.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Voxtral-4B-TTS-2603",
    "voice": "casual_male",
    "input": "Get the trust fund to the bank early.",
    "stream": true,
    "response_format": "pcm"
  }' \
  --output output.pcm
```

The response uses `audio/pcm` with sample-rate metadata in its headers. Do not
interpret request latency as time to first audio until incremental vocoder
delivery is implemented.

## Request Parameters

| Parameter | Default | Notes |
|---|---|---|
| `model` | served model | Served model identifier |
| `input` | (required) | Text to synthesize |
| `voice` | `default` | Preset voice name from the checkpoint's `voice_embedding/` directory |
| `max_new_tokens` | `4096` | Maximum number of generated acoustic tokens |
| `response_format` | `wav` | Output container (`wav`, `mp3`, `flac`, `opus`, `aac`, `pcm`) |
| `stream` | `false` | Accepted for compatibility; audio currently arrives after full synthesis |

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

### RTX 4090D Functional Run

[Issue #1188](https://github.com/sgl-project/sglang-omni/issues/1188)
records a single-GPU functional run on an RTX 4090D (SM89, 24,564 MiB), using
BF16 and the native SDPA audio-tokenizer fallback. Output length was
nondeterministic, so real-time factor (RTF) is the comparable metric.

| Metric (concurrency 1, 20 requests) | Value |
|---|---|
| Completed / failed | 20 / 0 |
| Latency median / p95 | 1.404 s / 1.729 s |
| Audio duration median / p95 | 4.800 s / 6.080 s |
| RTF median / p95 | 0.286 / 0.296 |
| RTF min / max | 0.283 / 0.310 |

BF16 startup, preset voice synthesis, WAV output, and PCM output completed.
Reference-voice cloning was unavailable in that run. Incremental streaming,
steady-state memory, compile-memory deltas, concurrency above 1, and a
current-`main` quality sweep remain unqualified.

## Known Limitations

- **Preset voices only.** Voxtral selects from named voices baked into the checkpoint; it does
  not clone an arbitrary speaker from a reference clip in this engine.
- **Non-incremental streaming.** The API accepts `stream=true`, but Voxtral
  currently emits audio only after full synthesis.
- **Deterministic decoding.** `temperature` is fixed at `0.0`; you cannot trade determinism for
  diversity through sampling parameters.
- **Language coverage.** Quality is tuned for the 9 supported languages (English, French,
  Spanish, German, Italian, Portuguese, Dutch, Arabic, Hindi).
- **Non-commercial license.** The weights are CC BY-NC 4.0; commercial use is not permitted.
