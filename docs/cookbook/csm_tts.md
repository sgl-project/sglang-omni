# Sesame CSM-1B

[CSM-1B](https://huggingface.co/sesame/csm-1b) is a conversational TTS model
from Sesame. A Llama-3.2-1B backbone predicts codebook 0 at 12.5 Hz; a 31-step
depth decoder fills the remaining 31 RVQ codes; Kyutai Mimi decodes each frame
to **24 kHz** audio. SGLang-Omni serves the HuggingFace Transformers export
[`eustlb/csm-1b`](https://huggingface.co/eustlb/csm-1b) through
`/v1/audio/speech`.

The pipeline is `preprocessing → audio_encoder → tts_engine → vocoder`.

| Component | Spec |
|---|---|
| Architecture | `CsmForConditionalGeneration` (`csm`) |
| Backbone | Llama-3.2-1B (16 L, hidden=2048, GQA 32/8) |
| Depth decoder | 31 steps, one RVQ codebook per step |
| Audio tokens | 32-codebook Mimi RVQ at 12.5 Hz |
| Output audio | 24 kHz mono |

Default sampling is **T=0.9, top_k=50, no nucleus**. Caller-set sampling
fields override these defaults; omitted fields keep the CSM defaults.
Prompt text uses speaker tag `[0]`.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md),
then download the Transformers export:

```bash
hf download eustlb/csm-1b
```

Use the Transformers shards (`transformers-*.safetensors`). The original
`sesame/csm-1b` `model.safetensors` / `ckpt.pt` Q/K weights are not
RoPE-permuted for this port.

## Launch the Server

```bash
sgl-omni serve \
  --model-path eustlb/csm-1b \
  --config examples/configs/csm_1b.yaml \
  --port 8000
```

## Synthesizing Speech

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "eustlb/csm-1b",
    "input": "The past is just a story we tell ourselves."
  }' \
  --output output.wav
```

### Python

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "eustlb/csm-1b",
        "input": "The past is just a story we tell ourselves.",
    },
)
resp.raise_for_status()
open("output.wav", "wb").write(resp.content)
```

`max_new_tokens` counts **frames** (80 ms each). The example config caps
generation at 125 frames (~10 s).

## Streaming

Add `"stream": true` to a speech request. Shared raw WAV, PCM, and SSE clients
are in [TTS Model Usage](../basic_usage/tts.md#streaming). First-chunk latency
uses `initial_codec_chunk_frames` when set; otherwise the vocoder defaults
(`stream_stride=13`, then `stream_followup_stride=6`).

## Request Parameters

| Parameter | Default | Notes |
|---|---|---|
| `input` | (required) | Non-empty text to synthesize |
| `temperature` | `0.9` | Override when set on the request |
| `top_k` | `50` | Override when set on the request |
| `top_p` | unset | No nucleus unless the caller sets it |
| `max_new_tokens` | `125` | Frames (80 ms). 125 ≈ 10 s |
| `stream` | `false` | Streams raw PCM when enabled |
| `response_format` | `wav` | Use `pcm` when `stream` is enabled |

Codebook 0 and the depth decoder share this sampling set.

## Benchmark Results

`eustlb/csm-1b` on Seed-TTS EN (1088 utterances, `--no-ref-audio`), official
sampling `temperature=0.9` / `top_k=50`, example-config frame cap 125 (~10 s).
WER scored with HF Whisper-large-v3. Hardware: 1× H200, eager decode, radix
off, CUDA graph off.

| System | Corpus WER | Mean / median | >50% WER | Completed |
|---|---|---|---|---|
| This serve path | 3.49% | 3.55% / 0.00% | 7 / 1088 | 1088 / 0 |
| HuggingFace `generate` | 3.90% | 4.06% / 0.00% | 17 / 1088 | 1088 / 0 |

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --generate-only --use-existing-server \
  --host 127.0.0.1 --port 8000 \
  --model eustlb/csm-1b \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --lang en --no-ref-audio \
  --temperature 0.9 --top-k 50 \
  --max-new-tokens 125 --max-concurrency 2
```

## Known Limitations

- CUDA graph and radix cache are off in the example config.
- Concurrency knob: raise `tts_engine.engine.max_running_requests` (example
  default `4`). The sampler pool follows `get_schedule().max_running_requests`.
  Encoder/vocoder `max_batch_size` stay at `4` unless you raise them in
  config. The Llama backbone is not tensor-parallel.
- On Hopper, the depth decoder uses FLASH / EFFICIENT / MATH SDPA. cuDNN
  aborts on the additive-mask + GQA layout.
