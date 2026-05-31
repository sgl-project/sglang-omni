# Higgs TTS

[Higgs Audio v3 TTS](https://huggingface.co/boson-sglang/higgs-audio-v3-tts-4b-base)
is a chat-native text-to-speech model from Boson AI built on a Qwen3-4B backbone. It generates
24 kHz speech through 8 discrete codebooks and supports 100+ languages, voice cloning from a
reference clip, and fine-grained inline control over emotion, style, sound effects, and prosody.

## Highlights

- **Chat-native, low-latency** streaming multi-turn speech generation
- **Multilingual** — 100+ languages and dialects, 90+ with single-digit WER/CER
- **Voice clone accuracy** — high-fidelity zero-shot speaker cloning from reference clips
- **Inline control** via `<|emotion:…|>`, `<|style:…|>`, `<|sfx:…|>`, `<|prosody:…|>` tags

## Architecture

![Higgs Audio v3 Generation Architecture](../_static/image/higgs-architecture.png)

Higgs autoregressive decoder consumes interleaved text and audio tokens. Audio is encoded by the **Higgs Tokenizer** into 8 codebooks at 25 fps, staggered via a **delay pattern**, then mapped to backbone hidden states through a **multi-codebook fused embedding**. Output codes pass through a **multi-codebook fused head**, are de-delayed, and decoded back to waveform. Multi-turn generation interleaves `<|text|>…<|audio|>…` chunks so each new chunk is grounded on reference + prior chunks.

| Component | Spec |
|---|---|
| Backbone | ~4B autoregressive decoder (36 L, hidden=2560, GQA 32/8) |
| Audio tokens | 8 codebooks × 1026 vocab, delay pattern |
| Multi-codebook embedding / head | Fused single-tensor, tied with text embedding |
| Context length | 8,192 tokens (training sequence length) |

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md), then download the model:

```bash
# Higgs TTS model is private; export your HF token before downloading.
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
hf download boson-sglang/higgs-audio-v3-TTS-4B-grpo05200410999
hf download bosonai/higgs-audio-v2-tokenizer
```

## Server Configuration

The pipeline is `preprocessing → audio_encoder → tts_engine → vocoder`.

```bash
sgl-omni serve \
  --model-path boson-sglang/higgs-audio-v3-TTS-4B-grpo05200410999 \
  --config examples/configs/higgs_tts.yaml \
  --port 8000
```

## Synthesizing Speech

### Zero-shot

1. Use curl

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, how are you?"}' \
  --output output.wav
```

2. Use Python

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={"input": "Hello, how are you?"},
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

Reference output:

<audio controls>
  <source src="../_static/audio/higgs-1.wav" type="audio/wav">
</audio>

### Voice Cloning

Supplying the reference transcript (`text`) materially improves cloning quality.

1. Use curl

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Have a nice day and enjoy south california sunshine.",
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

2. Use Python

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "input": "Have a nice day and enjoy south california sunshine.",
        "references": [{
            "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
            "text": "We asked over twenty different people, and they all said it was his.",
        }],
        "temperature": 0.8,
        "top_k": 50,
        "max_new_tokens": 1024,
    },
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```
Reference input:

<audio controls>
  <source src="../_static/audio/higgs-3.wav" type="audio/wav">
</audio>

Reference output:

<audio controls>
  <source src="../_static/audio/higgs-2.wav" type="audio/wav">
</audio>

### Streaming

Unlike a standard request where you wait for the full audio to be generated before receiving anything, streaming lets you start receiving and playing audio **while generation is still in progress**. This significantly reduces time-to-first-audio, which matters for real-time or interactive use cases.

Higgs TTS implements streaming via [Server-Sent Events (SSE)](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events). Each SSE event carries a base64-encoded WAV chunk. Your client can decode and play each chunk as it arrives, rather than buffering the entire response.

Enable streaming by setting `"stream": true` in the request body. During generation, the vocoder emits incremental audio chunks; the terminal event is intentionally slim and carries metadata such as `sample_rate` and `usage` instead of repeating the full waveform. Inside the pipeline, audio chunks use the compact `audio_waveform` payload (`bytes` plus `audio_waveform_shape`, `audio_waveform_dtype`, and `sample_rate`), which the HTTP layer encodes into the SSE `audio.data` field.

1. Use curl

Set `"stream": true` in your request body:

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
The `-N` flag disables curl's output buffering so SSE events are printed as they arrive.

2. Use Python

This example decodes each chunk and writes it to a WAV file incrementally. In a real application, you would pipe the decoded bytes directly to an audio player (e.g., via `pyaudio` or `sounddevice`).

```python
import requests
import base64
import json

REFERENCE_AUDIO = "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav"
REFERENCE_TEXT = "We asked over twenty different people, and they all said it was his."
SPEECH_INPUT = "Get the trust fund to the bank early."

with requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "input": SPEECH_INPUT,
        "references": [{"audio_path": REFERENCE_AUDIO, "text": REFERENCE_TEXT}],
        "stream": True,
    },
    stream=True,
) as resp:
    resp.raise_for_status()
    with open("output_streaming.wav", "wb") as f:
        for line in resp.iter_lines():
            if not line or line == b"data: [DONE]":
                continue
            if not line.startswith(b"data: "):
                continue

            event = json.loads(line[len(b"data: "):])

            if event.get("finish_reason") == "stop":
                break

            audio_data = event.get("audio") or {}
            if audio_data.get("data"):
                chunk = base64.b64decode(audio_data["data"])
                f.write(chunk)
                # In a real app: feed `chunk` to your audio player here
```

Reference output:

<audio controls>
  <source src="../_static/audio/higgs-4.wav" type="audio/wav">
</audio>


#### What the SSE response looks like
Each event follows the standard SSE format:
```
data: {"id": "speech-...", "object": "audio.speech.chunk", "index": 0, "audio": {"data": "<base64-encoded WAV bytes>", "format": "wav", ...}, "finish_reason": null}
data: {"id": "speech-...", "object": "audio.speech.chunk", "index": 1, "audio": null, "finish_reason": "stop", "usage": {...}}
data: [DONE]
```
Audio chunks have `"finish_reason": null` and carry audio data in `audio.data`. The final metadata event has `"finish_reason": "stop"` and `"audio": null`, followed by a `[DONE]` sentinel.



### Inline Control Tokens

Embed control tokens directly in the `input` field. Tokens from different
categories can be combined:

**Demo**

1. Emotion: surprise

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "I cant believe it! <|emotion:surprise|> <|prosody:pause|> <|style:whispering|> Higgs Model and SGLang are absolutely incredible."
  }' \
  --output output.wav
```

Reference output:

<audio controls>
  <source src="../_static/audio/control-tokens-test1.wav" type="audio/wav">
</audio>

2. Prosody: speed_slow

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "<|emotion:enthusiasm|> Welcome to the show! <|prosody:pause|> <|prosody:speed_slow|> Today we have something truly special for you."
  }' \
  --output output.wav
```
Reference output:

<audio controls>
  <source src="../_static/audio/control-tokens-test2.wav" type="audio/wav">
</audio>

3. Combine them together:

Here is an example of combining emotion, prosody and style tokens together:

<details>
<summary>Commands</summary>

Part 1 — female asks:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "<|prosody:pitch_high|> <|prosody:speed_slow|> Excuse me. Can you tell me how much the shirt is?",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_103675.wav",
      "text": "Excuse me. Can you tell me how much the shirt is?"
    }],
    "temperature": 0.5,
    "top_k": 30,
    "seed": 404
  }' \
  --output part1.wav
```

Part 2 — male answers:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "<|prosody:speed_very_slow|> <|prosody:expressive_low|> Yes, it is nine fifteen.",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }],
    "temperature": 0.5,
    "top_k": 30,
    "seed": 43
  }' \
  --output part2.wav
```

Part 3 — female reads the question:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "<|prosody:speed_slow|> <|prosody:expressive_low|> Question: How much is the shirt?",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_103675.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }],
    "temperature": 0.5,
    "top_k": 30,
    "seed": 44
  }' \
  --output part3.wav
```

Concatenate (~0.6 s gap between lines):

```bash
ffmpeg -y \
  -i part1.wav -f lavfi -t 0.6 -i anullsrc=r=24000:cl=mono \
  -i part2.wav -f lavfi -t 0.6 -i anullsrc=r=24000:cl=mono \
  -i part3.wav \
  -filter_complex "[0:a][1:a][2:a][3:a][4:a]concat=n=5:v=0:a=1" \
  gaokao_listening.wav
```

</details>

Reference output:

<audio controls>
  <source src="../_static/audio/gaokao-listening.wav" type="audio/wav">
</audio>

#### Emotion

| Token | Description |
|---|---|
| `<\|emotion:elation\|>` | Elation / joy |
| `<\|emotion:amusement\|>` | Amusement / playful laughter |
| `<\|emotion:enthusiasm\|>` | Enthusiasm / excitement |
| `<\|emotion:determination\|>` | Determination / firmness |
| `<\|emotion:pride\|>` | Pride / confidence |
| `<\|emotion:contentment\|>` | Calm satisfaction |
| `<\|emotion:affection\|>` | Warmth / affection |
| `<\|emotion:relief\|>` | Relief |
| `<\|emotion:contemplation\|>` | Thoughtful / reflective |
| `<\|emotion:confusion\|>` | Confused |
| `<\|emotion:surprise\|>` | Surprised |
| `<\|emotion:awe\|>` | Awe / wonder |
| `<\|emotion:longing\|>` | Longing / yearning |
| `<\|emotion:arousal\|>` | Heightened desire |
| `<\|emotion:anger\|>` | Anger |
| `<\|emotion:fear\|>` | Fear |
| `<\|emotion:disgust\|>` | Disgust |
| `<\|emotion:bitterness\|>` | Bitterness |
| `<\|emotion:sadness\|>` | Sadness |
| `<\|emotion:shame\|>` | Shame |
| `<\|emotion:helplessness\|>` | Helplessness |

#### Style

| Token | Description |
|---|---|
| `<\|style:singing\|>` | Singing |
| `<\|style:shouting\|>` | Shouting / projected voice |
| `<\|style:whispering\|>` | Whisper |

#### Sound Effects

| Token | Description |
|---|---|
| `<\|sfx:cough\|>` | Cough |
| `<\|sfx:laughter\|>` | Laughter |
| `<\|sfx:crying\|>` | Crying |
| `<\|sfx:screaming\|>` | Screaming |
| `<\|sfx:burping\|>` | Burping |
| `<\|sfx:humming\|>` | Humming |
| `<\|sfx:sigh\|>` | Sigh |
| `<\|sfx:sniff\|>` | Sniff |
| `<\|sfx:sneeze\|>` | Sneeze |

#### Prosody

| Token | Effect |
|---|---|
| `<\|prosody:speed_very_slow\|>` | ~0.65× speed |
| `<\|prosody:speed_slow\|>` | ~0.85× speed |
| `<\|prosody:speed_fast\|>` | ~1.2× speed |
| `<\|prosody:speed_very_fast\|>` | ~1.4× speed |
| `<\|prosody:pitch_low\|>` | ~−3 semitones |
| `<\|prosody:pitch_high\|>` | ~+2.5 semitones |
| `<\|prosody:pause\|>` | ~400–700 ms pause |
| `<\|prosody:long_pause\|>` | ~700–1500 ms pause |
| `<\|prosody:expressive_high\|>` | More expressive delivery |
| `<\|prosody:expressive_low\|>` | Flatter delivery |

### Pre-encoded reference codes
For high-throughput pipelines (e.g. RL rollout) where the same reference audio is reused across many requests, you can encode the reference audio offline and pass the discrete codes directly via `reference_codes` — this skips the server-side codec encode step. Shape must be `[T, num_codebooks=8]`.

```python
# python
resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "input": SPEECH_INPUT,
        "reference_codes": codes_TN,   # [T, 8] int list, pre-delay-pattern
        "reference_text": REFERENCE_TEXT,
    },
)
```

### Request parameters

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


### Throughput

[TODO (yichi, Huapeng): This should be updated in the last minute.]

Throughput on seed-tts en (N=50 per concurrency, sequential thread pool, A100 40GB, bf16):

| Concurrency | Mean latency | RTF (per-req) | audio_s/s |
|---:|---:|---:|---:|
| 1 | 4637 ms | 0.526 | 1.90 |
| 16 | 7138 ms | 0.747 | 12.88 |
| 32 | 10188 ms | 0.865 | 16.94 |

## Evaluation Benchmarks

We report **WER / CER** (↓, %) and **WavLM speaker similarity** (↑, ×100) on three zero-shot voice-cloning benchmarks.

### Seed-TTS

| Lang | WER ↓ | SIM ↑ |
|---|---|---|
| en | 2.05 | 64.86 |
| zh | 2.00 | 70.96 |
| **macro** | **2.02** | **67.91** |

### CV3 (9 langs)

| Lang | WER ↓ | SIM ↑ |
|---|---|---|
| de | 8.62 | 65.43 |
| en | 6.73 | 60.37 |
| es | 5.03 | 68.18 |
| fr | 14.50 | 62.34 |
| it | 8.55 | 67.34 |
| ja | 7.96 | 67.91 |
| ko | 4.38 | 68.40 |
| ru | 9.38 | 66.77 |
| zh | 5.19 | 69.71 |
| **macro** | **7.82** | **66.27** |

### MiniMax-Multilingual (23 langs)

| Lang | WER ↓ | SIM ↑ |
|---|---|---|
| ar | 2.59 | 74.77 |
| cs | 4.62 | 78.80 |
| de | 0.74 | 70.65 |
| el | 1.81 | 78.02 |
| en | 1.87 | 81.32 |
| es | 3.06 | 72.78 |
| fi | 4.62 | 82.69 |
| fr | 4.70 | 70.27 |
| hi | 6.81 | 80.94 |
| id | 2.38 | 72.42 |
| it | 2.07 | 74.56 |
| ja | 3.74 | 74.23 |
| ko | 3.57 | 74.86 |
| nl | 2.10 | 73.02 |
| pl | 2.08 | 83.16 |
| pt | 2.59 | 76.52 |
| ro | 3.64 | 77.10 |
| ru | 4.66 | 74.48 |
| th | 7.59 | 77.64 |
| tr | 2.09 | 77.72 |
| uk | 2.69 | 71.79 |
| vi | 1.18 | 73.46 |
| zh | 1.65 | 74.85 |
| **macro** | **3.17** | **75.92** |
