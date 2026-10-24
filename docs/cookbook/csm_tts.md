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
| Context | Optional previous turns as `{speaker, text, audio\|codes}` |

Default sampling is **T=0.9, top_k=50, no nucleus**. Speech-API sentinels
`0.8 / 0.8 / 30` are ignored unless those fields are set explicitly.

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

Plain TTS. `speaker` / `speaker_id` is an integer tag rendered as `[0]`, `[1]`,
… in the prompt (default `0`).

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

### Context / voice prompt

Optional previous turns go in `context` as `{speaker, text, audio}` (or
pre-encoded `codes`). The audio_encoder stage runs Mimi only when raw audio
is present.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "eustlb/csm-1b",
    "input": "How are you doing today?",
    "speaker": 0,
    "context": [{
      "speaker": 0,
      "text": "Hello, this is a voice prompt.",
      "audio": {"audio_path": "/path/to/prompt.wav"}
    }]
  }' \
  --output output.wav
```

`max_new_tokens` counts **frames** (80 ms each). The example config caps
generation at 125 frames (~10 s).

## Streaming

`stream: true` emits PCM from the vocoder as frames arrive. First-chunk
latency uses `initial_codec_chunk_frames` when set; otherwise the vocoder
defaults (`stream_stride=13`, then `stream_followup_stride=6`).

## Request Parameters

| Parameter | Default | Notes |
|---|---|---|
| `input` | (required) | Non-empty text to synthesize |
| `speaker` / `speaker_id` | `0` | Integer tag rendered as `[0]`, `[1]`, … in the prompt |
| `context` | `null` | Previous turns `{speaker, text, audio\|codes}` |
| `references` | `null` | Alias folded into `context` |
| `temperature` | `0.9` | Speech-API sentinels `0.8` are ignored unless set explicitly |
| `top_k` | `50` | Speech-API sentinel `30` is ignored unless set explicitly |
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
| This serve path | 3.48% | 3.40% / 0.00% | 7 / 1088 | 1088 / 0 |
| HuggingFace `generate` | 3.90% | 4.06% / 0.00% | 17 / 1088 | 1088 / 0 |

Serve-path latency mean 4.42 s, audio duration mean 4.11 s.

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
- Admission is `max_running_requests: 4`. The Llama backbone is not
  tensor-parallel.
- On Hopper, the depth decoder uses FLASH / EFFICIENT / MATH SDPA. cuDNN
  aborts on the additive-mask + GQA layout.
