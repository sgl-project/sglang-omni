# TTS Model Usage

This guide uses [Fish Speech S2-Pro](https://huggingface.co/fishaudio/s2-pro) as an example TTS (text-to-speech) model with SGLang-Omni and the OpenAI-compatible API. The same `/v1/audio/speech` endpoint also supports Voxtral TTS, Qwen3-TTS, and MOSS-TTS.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md), then download the model:

```bash
hf download fishaudio/s2-pro
```

Qwen3-TTS uses the upstream `qwen-tts` package, which currently requires
Transformers 4.57.3. Install it only in environments that serve Qwen3-TTS:

```bash
uv pip install --upgrade transformers==4.57.3 accelerate==1.12.0 sox einops
uv pip install --no-deps qwen-tts==0.1.1
```

## Supported TTS Models

| Model family | Example config | Request notes |
|---|---|---|
| Fish Speech S2-Pro | `examples/configs/s2pro_tts.yaml` | Supports plain TTS and voice cloning with `references` |
| [Voxtral TTS](../cookbook/voxtral_tts.md) | `examples/configs/voxtral_tts.yaml` | Uses `input`, `voice`, `response_format`, and `max_new_tokens`; use `--no-ref-audio` for SeedTTS benchmarking |
| [Qwen3-TTS Base](../cookbook/qwen3_tts.md) | `examples/configs/qwen3_tts_0_6b.yaml`, `examples/configs/qwen3_tts_1_7b.yaml` | Requires reference audio through `ref_audio` or `references[0].audio_path`; `language` defaults to `auto` |
| Qwen3-TTS CustomVoice | `examples/configs/qwen3_tts_0_6b_customvoice.yaml` | Text-only requests use the checkpoint speaker table; missing `voice` defaults to `Vivian` |
| Qwen3-TTS VoiceDesign | `examples/configs/qwen3_tts_1_7b_voicedesign.yaml` | Requires `task_type="VoiceDesign"` and non-empty `instructions`; no reference audio is required |
| [MOSS-TTS](../cookbook/moss_tts.md) | `examples/configs/moss_tts.yaml` | Voice cloning via `ref_audio` or `references[0].audio_path` (+ `text`); duration via `${token:N}` or `token_count`; benchmark at `--max-concurrency 8` |

## Launch the Server

```bash
sgl-omni serve \
  --model-path fishaudio/s2-pro \
  --config examples/configs/s2pro_tts.yaml \
  --port 8000
```

Local `file://` reference audio is disabled by default. To allow local
reference files, launch with an explicit directory:

```bash
sgl-omni serve \
  --model-path fishaudio/s2-pro \
  --config examples/configs/s2pro_tts.yaml \
  --allowed-local-media-path /path/to/reference-audio \
  --port 8000
```

Remote HTTP(S) reference audio requires `--allowed-media-domain`. Repeat the
flag to allow multiple trusted media hosts.

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
REF_URI="file://$PWD/docs/_static/audio/gaokao-listening.wav"

curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d "{
    \"input\": \"This is a local reference audio test.\",
    \"ref_audio\": \"${REF_URI}\",
    \"ref_text\": \"Reference transcript for the local audio clip.\"
  }" \
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

### Voice Cloning

The examples below use an allowed local audio clip. The `references` field
accepts `audio_path` as a data URL or an allowed `file://` URI, plus `text`
(transcript of that audio).

1. Non-streaming request

```bash
REF_URI="file://$PWD/docs/_static/audio/gaokao-listening.wav"

curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d "{
    \"input\": \"This is a local reference audio test.\",
    \"references\": [{
      \"audio_path\": \"${REF_URI}\",
      \"text\": \"Reference transcript for the local audio clip.\"
    }]
  }" \
  --output output.wav
```

2. Streaming

Enable streaming to receive audio chunks in real time via Server-Sent Events
(SSE). HTTP streaming uses raw PCM chunks, so set both `"stream": true` and
`"response_format": "pcm"`:

```bash
REF_URI="file://$PWD/docs/_static/audio/gaokao-listening.wav"

curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d "{
    \"input\": \"This is a local reference audio streaming test.\",
    \"references\": [{
      \"audio_path\": \"${REF_URI}\",
      \"text\": \"Reference transcript for the local audio clip.\"
    }],
    \"stream\": true,
    \"response_format\": \"pcm\"
  }"
```

The server returns a stream of SSE events. Each event contains an
`audio.speech.chunk` object with a base64-encoded PCM chunk. The stream ends
with `data: [DONE]`.

For clients that want a continuous byte stream instead of SSE framing, request raw PCM explicitly:

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Get the trust fund to the bank early.",
    "stream": true,
    "stream_format": "audio",
    "response_format": "pcm"
  }' \
  --output output.pcm
```

Raw audio streaming returns 16-bit mono PCM bytes (`audio/pcm`) with sample-rate metadata in response headers. It does not include in-band SSE events, final usage, or a `[DONE]` sentinel. When the client does not set `initial_codec_chunk_frames`, raw PCM requests default to a 1-frame first vocoder chunk for lower first-audio latency; set `initial_codec_chunk_frames` to `0` to use the model's steady chunk size from the start.

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
from pathlib import Path

REFERENCE_AUDIO = Path("docs/_static/audio/gaokao-listening.wav").resolve().as_uri()
REFERENCE_TEXT = "Reference transcript for the local audio clip."
SPEECH_INPUT = "This is a local reference audio test."
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
import base64, json, wave

import requests

payload = {
    "input": SPEECH_INPUT,
    "references": [{"audio_path": REFERENCE_AUDIO, "text": REFERENCE_TEXT}],
    "stream": True,
    "response_format": "pcm",
}

chunks = []
sample_rate = None
with requests.post(
    "http://localhost:8000/v1/audio/speech",
    json=payload,
    stream=True,
    timeout=600,
) as stream:
    stream.raise_for_status()
    for line in stream.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data = line[len("data:"):].lstrip()
        if data == "[DONE]":
            break
        audio = json.loads(data).get("audio") or {}
        b64 = audio.get("data")
        if not b64:
            continue
        sample_rate = sample_rate or audio.get("sample_rate")
        chunks.append(base64.b64decode(b64))

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
| `stream` | bool | `false` | Enable streaming via SSE; when true, `response_format` must be `pcm` |
| `stream_format` | string | `"sse"` | Streaming transport. Use `"audio"` with `stream=true` and `response_format="pcm"` for raw PCM bytes; the response headers declare the stream sample rate, channel count, and bit depth |
| `initial_codec_chunk_frames` | int | `null` | Optional first codec chunk size for streaming TTFA tuning. Higgs TTS currently consumes this parameter first; raw PCM speech requests default this to `1` unless the client sets a value, including `0` |
| `references` | list | `null` | Reference audio for voice cloning; each item has `audio_path` (allowed HTTP(S), data URL, or allowed `file://` URI) and `text` |
| `ref_audio` | string | `null` | Reference audio as allowed HTTP(S), data URL, or an allowed `file://` URI; equivalent to `references[0].audio_path` |
| `ref_text` | string | `null` | Transcript for `ref_audio`; equivalent to `references[0].text` |
| `language` | string | `null` | Language hint: `Auto`, `Chinese`, `English`, `Japanese`, `Korean`, `German`, `French`, `Russian`, `Portuguese`, `Spanish`, or `Italian` |
| `task_type` | string | `null` | Qwen3-TTS task type: `Base`, `CustomVoice`, or `VoiceDesign`; inferred as `Base` when reference audio/text is present, otherwise `CustomVoice` |
| `instructions` | string | `null` | Qwen3-TTS style or VoiceDesign instructions |
| `max_new_tokens` | int | `null` | Maximum number of generated tokens |
| `token_count` | int | `null` | Model-specific duration token target |
| `duration_tokens` | int | `null` | Alias-style duration token target for models that expose duration control |
| `x_vector_only_mode` | bool | `null` | Qwen3-TTS Base speaker-embedding mode |
| `temperature` | float | `null` | Sampling temperature |
| `top_p` | float | `null` | Top-p sampling |
| `top_k` | int | `null` | Top-k sampling |
| `repetition_penalty` | float | `null` | Repetition penalty |
| `seed` | int | `null` | Model-specific; Qwen3-TTS Base accepts request-scoped seed, Voxtral TTS currently rejects seed |

Invalid speech requests return an OpenAI-style error envelope:

```json
{
  "error": {
    "message": "stream=true requires response_format='pcm'",
    "type": "invalid_request_error",
    "param": "response_format",
    "code": null
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
- `Streaming` consumes the `/v1/audio/speech` SSE stream, converts incremental PCM chunks for playback, and also writes a final combined WAV artifact for inspection.

The launcher starts the backend first, waits for `/health`, then starts the Gradio UI with:

```bash
python -m playground.s2pro.app --api-base http://localhost:8000
```

A demo play video is available [here](https://x.com/lmsysorg/status/2031412267213008984/video/1). We highly recommend using playground since audio data is hard to interact with by CLI.
