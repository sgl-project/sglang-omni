# TTS Model Usage

This guide uses [Fish Speech S2-Pro](https://huggingface.co/fishaudio/s2-pro) as an example TTS (text-to-speech) model with SGLang-Omni and the OpenAI-compatible API. The same `/v1/audio/speech` endpoint also supports Voxtral TTS, Qwen3-TTS, and MOSS-TTS.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md), then download the model:

```bash
hf download fishaudio/s2-pro
```

Qwen3-TTS uses the upstream `qwen-tts` package. Install it without dependencies
so the SGLang-Omni Transformers 5.6 / SGLang 0.5.12.post1 stack remains in place:

```bash
uv pip install --upgrade sox einops
uv pip install --no-deps qwen-tts==0.1.1
```

## Supported TTS Models

| Model family | Example config | Request notes |
|---|---|---|
| [Fish Speech S2-Pro](../cookbook/fishaudio_s2_pro.md) | `examples/configs/s2pro_tts.yaml` | Supports plain TTS and voice cloning with `references` |
| [Voxtral TTS](../cookbook/voxtral_tts.md) | `examples/configs/voxtral_tts.yaml` | Uses `input`, `voice`, `response_format`, and `max_new_tokens`. Use `--no-ref-audio` for SeedTTS benchmarking |
| [Qwen3-TTS Base](../cookbook/qwen3_tts.md) | `examples/configs/qwen3_tts_0_6b.yaml`, `examples/configs/qwen3_tts_1_7b.yaml` | Requires reference audio through `ref_audio` or `references[0].audio_path`. `language` defaults to `auto` |
| [Qwen3-TTS CustomVoice](../cookbook/qwen3_tts.md) | `examples/configs/qwen3_tts_0_6b_customvoice.yaml` | Text-only requests use the checkpoint speaker table. Missing `voice` defaults to `Vivian` |
| [Qwen3-TTS VoiceDesign](../cookbook/qwen3_tts.md) | `examples/configs/qwen3_tts_1_7b_voicedesign.yaml` | Requires `task_type="VoiceDesign"` and non-empty `instructions`. No reference audio is required |
| [MOSS-TTS](../cookbook/moss_tts.md) | `examples/configs/moss_tts.yaml` | Voice cloning via `ref_audio` or `references[0].audio_path` (+ `text`). Duration via `${token:N}` or `token_count`. Benchmark at `--max-concurrency 8` |

## Launch the Server

```bash
sgl-omni serve \
  --model-path fishaudio/s2-pro \
  --config examples/configs/s2pro_tts.yaml \
  --port 8000
```

For Voxtral:

```bash
sgl-omni serve \
  --model-path mistralai/Voxtral-4B-TTS-2603 \
  --config examples/configs/voxtral_tts.yaml \
  --port 8000
```

For Qwen3-TTS Base:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --config examples/configs/qwen3_tts_0_6b.yaml \
  --port 8000
```

For Qwen3-TTS CustomVoice:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --config examples/configs/qwen3_tts_0_6b_customvoice.yaml \
  --port 8000
```

For Qwen3-TTS VoiceDesign:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --config examples/configs/qwen3_tts_1_7b_voicedesign.yaml \
  --port 8000
```

For MOSS-TTS:

```bash
sgl-omni serve \
  --model-path OpenMOSS-Team/MOSS-TTS-v1.5 \
  --config examples/configs/moss_tts.yaml \
  --port 8000
```

## Use Curl

Generate speech from text without any reference audio. This is valid for
Qwen3-TTS CustomVoice, Voxtral, and S2-Pro. It is not valid for Qwen3-TTS Base.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"input": "Hello, how are you?"}' \
    --output output.wav
```

Qwen3-TTS Base requires reference audio:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Get the trust fund to the bank early.",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "ref_text": "We asked over twenty different people, and they all said it was his."
  }' \
  --output output.wav
```

Qwen3-TTS VoiceDesign uses text plus voice instructions:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
      "input": "Hello, how are you?",
      "task_type": "VoiceDesign",
      "instructions": "A warm, natural young adult voice."
    }' \
    --output output.wav
```

For natural-sounding Fish Speech S2-Pro results, use Voice Cloning with a reference audio clip.

### Fish Speech Voice Cloning

The examples below use a sample clip from [`seed-tts-eval-mini`](https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini). The `references` field accepts `audio_path` (a local path, file URL, data URL, or HTTP URL) and `text` (transcript of that audio).

1. Non-streaming request

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Get the trust fund to the bank early.",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }]
  }' \
  --output output.wav
```

2. Streaming

Enable streaming to receive raw PCM audio chunks in real time. HTTP streaming
requires both `"stream": true` and `"response_format": "pcm"`:

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Get the trust fund to the bank early.",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }],
    "stream": true,
    "response_format": "pcm"
  }' \
  --output output.pcm
```

Streaming returns 16-bit mono PCM bytes (`audio/pcm`) with sample-rate metadata
in response headers. It does not include in-band JSON events, final usage, or a
terminal sentinel. When the client does not set `initial_codec_chunk_frames`,
streaming requests default to a 1-frame first vocoder chunk for lower
first-audio latency. Set `initial_codec_chunk_frames` to `0` to use the model's
steady chunk size from the start.

## Use Python

### Basic TTS

This no-reference request applies to Fish Speech S2-Pro and Voxtral TTS.

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

### OpenAI Python SDK

The endpoint is compatible with the OpenAI Python SDK when the client points to
the SGLang-Omni server:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

response = client.audio.speech.create(
    model="fishaudio/s2-pro",
    voice="default",
    input="Hello, how are you?",
    response_format="wav",
)
response.stream_to_file("output.wav")
```

### Voice Cloning

```python
REFERENCE_AUDIO = "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav"
REFERENCE_TEXT = "We asked over twenty different people, and they all said it was his."
SPEECH_INPUT = "Get the trust fund to the bank early."
```

1. Non-streaming Request

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "input": SPEECH_INPUT,
        "references": [{"audio_path": REFERENCE_AUDIO, "text": REFERENCE_TEXT}],
    },
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

2. Streaming Request

```python
import wave

import requests

payload = {
    "input": SPEECH_INPUT,
    "references": [{"audio_path": REFERENCE_AUDIO, "text": REFERENCE_TEXT}],
    "stream": True,
    "response_format": "pcm",
}

chunks = []
with requests.post(
    "http://localhost:8000/v1/audio/speech",
    json=payload,
    stream=True,
    timeout=600,
) as stream:
    stream.raise_for_status()
    sample_rate = int(stream.headers.get("x-sample-rate", 24000))
    for chunk in stream.iter_content(chunk_size=None):
        if chunk:
            chunks.append(chunk)

with wave.open("output_stream.wav", "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sample_rate or 24000)
    w.writeframes(b"".join(chunks))
```

## Request Parameters

The table below lists all parameters accepted by the `/v1/audio/speech` endpoint.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input` | string | (required) | Text to synthesize |
| `voice` | string | `"default"` | Voice identifier |
| `response_format` | string | `"wav"` | Output audio format: `wav`, `mp3`, `flac`, `pcm`, `aac`, or `opus` |
| `speed` | float | `1.0` | Playback speed multiplier from `0.25` to `4.0` |
| `stream` | bool | `false` | Enable raw PCM streaming. When true, `response_format` must be `pcm` |
| `initial_codec_chunk_frames` | int | `null` | Optional first codec chunk size for streaming TTFA tuning. Higgs TTS currently consumes this parameter first. Raw PCM speech requests default this to `1` unless the client sets a value, including `0` |
| `references` | list | `null` | Reference audio for voice cloning. Each item has `audio_path` (local path / file URL / data URL / remote URL) and `text` |
| `ref_audio` | string | `null` | Reference audio path / URL / base64 string. Equivalent to `references[0].audio_path` |
| `ref_text` | string | `null` | Transcript for `ref_audio`. Equivalent to `references[0].text` |
| `language` | string | `null` | Language hint: `Auto`, `Chinese`, `English`, `Japanese`, `Korean`, `German`, `French`, `Russian`, `Portuguese`, `Spanish`, or `Italian` |
| `task_type` | string | `null` | Qwen3-TTS task type: `Base`, `CustomVoice`, or `VoiceDesign`. Inferred as `Base` when reference audio/text is present, otherwise `CustomVoice` |
| `instructions` | string | `null` | Qwen3-TTS style or VoiceDesign instructions |
| `max_new_tokens` | int | `null` | Maximum number of generated tokens |
| `token_count` | int | `null` | Model-specific duration token target |
| `duration_tokens` | int | `null` | Alias-style duration token target for models that expose duration control |
| `x_vector_only_mode` | bool | `null` | Qwen3-TTS Base speaker-embedding mode |
| `temperature` | float | `null` | Sampling temperature |
| `top_p` | float | `null` | Top-p sampling |
| `top_k` | int | `null` | Top-k sampling |
| `repetition_penalty` | float | `null` | Repetition penalty |
| `seed` | int | `null` | Model-specific. Qwen3-TTS Base accepts request-scoped seed, Voxtral TTS currently rejects seed |

Invalid speech requests return an OpenAI-style error envelope:

```json
{
  "error": {
    "message": "stream=true requires response_format='pcm'",
    "type": "BadRequestError",
    "param": "response_format",
    "code": 400
  }
}
```

## H200 SeedTTS Benchmark Commands

Download the full SeedTTS set first:

```bash
python -m benchmarks.dataset.prepare --dataset seedtts
```

Run EN and ZH after launching the target server on port 8000. Do not add benchmark results to docs until the full H200 runs complete.

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --port 8000 \
  --output-dir results/qwen3_tts_0_6b_en \
  --lang en \
  --max-concurrency 16

python -m benchmarks.eval.benchmark_tts_seedtts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --port 8000 \
  --output-dir results/qwen3_tts_0_6b_zh \
  --lang zh \
  --max-concurrency 16

python -m benchmarks.eval.benchmark_tts_seedtts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --port 8000 \
  --output-dir results/qwen3_tts_1_7b_en \
  --lang en \
  --max-concurrency 16

python -m benchmarks.eval.benchmark_tts_seedtts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --port 8000 \
  --output-dir results/qwen3_tts_1_7b_zh \
  --lang zh \
  --max-concurrency 16

python -m benchmarks.eval.benchmark_tts_seedtts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --model mistralai/Voxtral-4B-TTS-2603 \
  --port 8000 \
  --output-dir results/voxtral_en \
  --lang en \
  --max-new-tokens 4096 \
  --max-concurrency 16 \
  --no-ref-audio \
  --voice cheerful_female

python -m benchmarks.eval.benchmark_tts_seedtts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --model mistralai/Voxtral-4B-TTS-2603 \
  --port 8000 \
  --output-dir results/voxtral_zh \
  --lang zh \
  --max-new-tokens 4096 \
  --max-concurrency 16 \
  --no-ref-audio \
  --voice cheerful_female
```

## Interactive Playground

SGLang-Omni ships with a Gradio-based playground for interactive TTS experimentation:

```bash
./playground/s2pro/start.sh
```

The playground now exposes two demo modes against the same S2 Pro backend:

- `Non-Streaming` starts a standard request and shows the final WAV after generation finishes.
- `Streaming` consumes the `/v1/audio/speech` raw PCM stream, converts incremental chunks for playback, and also writes a final combined WAV artifact for inspection.

The launcher starts the backend first, waits for `/health`, then starts the Gradio UI with:

```bash
python -m playground.s2pro.app --api-base http://localhost:8000
```

A demo play video is available [here](https://x.com/lmsysorg/status/2031412267213008984/video/1). We highly recommend using playground since audio data is hard to interact with by CLI.
