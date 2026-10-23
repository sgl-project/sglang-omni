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
MODEL_PATH="$(hf download mistralai/Voxtral-4B-TTS-2603 --quiet)"
ls "${MODEL_PATH}/voice_embedding"
```

**Python**

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

### Streaming

Set `"stream": true` and `"response_format": "pcm"` to receive raw PCM audio
chunks in real time:

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
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

Streaming returns `audio/pcm` 16-bit mono PCM bytes with sample-rate metadata in
the response headers. See the [Higgs TTS cookbook](../cookbook/higgs_tts.md#streaming)
for a full Python raw PCM consumer.

## Request Parameters

| Parameter | Default | Notes |
|---|---|---|
| `model` | served model | Served model identifier |
| `input` | (required) | Text to synthesize |
| `voice` | `default` | Preset voice name from the checkpoint's `voice_embedding/` directory |
| `max_new_tokens` | `4096` | Maximum number of generated acoustic tokens |
| `response_format` | `wav` | Output container (`wav`, `mp3`, `flac`, `opus`, `aac`, `pcm`) |
| `stream` | `false` | Stream raw PCM audio chunks |

> Voxtral generation is **deterministic**: the engine fixes `temperature` to `0.0`, so a request
> that sets `temperature`, `top_p`, `top_k`, `repetition_penalty` or `seed` is rejected. Reference-clip voice
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

## Apple Silicon (Native MLX)

`voxtral_tts_server` from the native MLX runtime (`sglang_omni_mlx/native`) serves the same
`/v1/audio/speech` API on an Apple Silicon Mac, from the 4-bit
[`mlx-community/Voxtral-4B-TTS-2603-mlx-4bit`](https://huggingface.co/mlx-community/Voxtral-4B-TTS-2603-mlx-4bit)
checkpoint, as one C++ process without CUDA, PyTorch or Python.

```bash
sglang_omni_mlx/native/scripts/build_runtime.sh ~/omni-runtime

hf download mlx-community/Voxtral-4B-TTS-2603-mlx-4bit \
  --revision f98fc91b9cb5adc7dab56102c690458276c14c6a --local-dir voxtral-4b-tts-mlx-4bit

~/omni-runtime/bin/voxtral_tts_server --model-path voxtral-4b-tts-mlx-4bit \
  --model-name mistralai/Voxtral-4B-TTS-2603 --port 8000
```

The requests above work unchanged. The native server differs from the CUDA server in that:

- `response_format` is `wav` or `pcm`; `speed` must be `1.0`; `stream_format` must be `audio`.
- `max_new_tokens` is at most `4096` audio frames (12.5 per second, about 5.5 minutes).
- An unknown `voice` returns 400 with the list of the checkpoint's 20 preset voices.
- Started with `--model-name`, it returns 404 for any other `model`; without that flag, `model`
  is ignored, as on CUDA.
- Errors are `{"detail": "..."}`.
- Requests run one at a time and `--max-queued-requests` (default 2) more wait; the rest get
  503, and `/health` shows the queue.
- A request not finished `--request-timeout-s` (default 600) seconds after it arrives, time in
  the queue included, is cancelled: it returns 504, or its stream ends early.
- Output varies between requests: the acoustic head samples flow-matching noise.

Measured on an M4 MacBook Air (16 GB), warm: 3.3 s load, 0.55–0.85 s to the first streamed
audio, real-time factor 0.72 (mean of 18 utterances), about 3.4 GB peak MLX memory. Transcribed
back with Qwen3-ASR-0.6B, 10 English sentences give 2.6% WER and one sentence in each of the
other 8 supported languages gives 5.3% WER. With the same seed and MLX version, its audio codes
are bit-identical to the Python MLX port in
[#2788](https://github.com/sgl-project/sglang-omni/pull/2788), and so is its 16-bit PCM after
the first 10 ms, which it fades in float32 as the CUDA vocoder does. Mac CI checks five seeded
requests' codes (English, French, German, Hindi, and a `max_new_tokens` cap) against
`sglang_omni_mlx/native/ci/model_golden/voxtral-4b-tts-2603.json`.

## Known Limitations

- **Preset voices only.** Voxtral selects from named voices baked into the checkpoint; it does
  not clone an arbitrary speaker from a reference clip in this engine.
- **Deterministic decoding.** `temperature` is fixed at `0.0`; you cannot trade determinism for
  diversity through sampling parameters.
- **Language coverage.** Quality is tuned for the 9 supported languages (English, French,
  Spanish, German, Italian, Portuguese, Dutch, Arabic, Hindi).
- **Non-commercial license.** The weights are CC BY-NC 4.0; commercial use is not permitted.
