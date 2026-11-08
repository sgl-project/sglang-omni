# ASR Model Usage

This guide uses [Qwen3-ASR](../cookbook/qwen3_asr.md) as the example. The same
`/v1/audio/transcriptions` endpoint also serves Whisper ASR, Fun-ASR-Nano,
ARK-ASR-3B, Nemotron 3.5 ASR, Nemotron Transcribe, MOSS-Transcribe-Diarize,
and Ming-3.0-ASR. Every ASR pipeline accepts one uploaded audio file per
request and returns text.

Speech translation is a separate route. `/v1/audio/translations` is served only
by pipelines that declare audio-translation capability: Whisper ASR translates
source speech to English, and Hy-SimulMT translates Chinese to English. The
other ASR models return HTTP 400 instead of silently transcribing. See
[Audio Translations](audio_translations.md) for the support matrix and the
response formats.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md),
then download the checkpoint named in the model's cookbook. Qwen3-ASR is served
from a pinned revision:

```bash
MODEL_REVISION=7278e1e70fe206f11671096ffdd38061171dd6e5
MODEL_PATH="$(
  hf download Qwen/Qwen3-ASR-1.7B \
    --revision "${MODEL_REVISION}" \
    --quiet
)"
```

Model-specific prerequisites stay in the cookbook: the Apple Silicon and Ascend
NPU notes for Qwen3-ASR, Whisper's multilingual-checkpoint requirement, and any
packages a pipeline imports beyond the base installation.

## Supported ASR Models

| Model family | Example config | Request notes |
|---|---|---|
| [Qwen3-ASR](../cookbook/qwen3_asr.md) | `examples/configs/qwen3_asr_rtx4090.yaml`, `examples/configs/qwen3_asr_rtx5090.yaml` | Multilingual transcription with 30 language hints; chunked uploads up to one hour; live PCM through `/v1/realtime?intent=transcription` |
| [Ming-3.0-ASR](../cookbook/ming3_asr.md) | `--model-path` only | Chinese and English only; splits uploads longer than 30 s; `srt` and `vtt` rejected |
| [Fun-ASR-Nano](../cookbook/fun_asr.md) | `--model-path` only | Multilingual transcription; no translation route |
| [ARK-ASR-3B](../cookbook/arkasr.md) | `--model-path` only | Multilingual transcription; shares the LM prefill-admission gate |
| [Nemotron 3.5 ASR](../cookbook/nemotron3_5_asr.md) | `--model-path` only | Multilingual RNN-T transcription; native PCM streaming through `/v1/realtime` without VAD or manual commit |
| [Nemotron Transcribe](../cookbook/nemotron_transcribe.md) | `--model-path` only | Whole-file transcription up to one hour as one decoder request; validated on one H100 80 GB |
| [MOSS-Transcribe-Diarize](../cookbook/moss_transcribe_diarize.md) | `--model-path` only | Multi-speaker transcription, diarization and timestamps in one pass; up to ~90 minutes; `verbose_json` segments |
| [Whisper ASR](../cookbook/whisper_asr.md) | `--model-path` only | Experimental transcription and speech-to-English translation; `srt` and `vtt` subtitles with model-derived timestamps |

## Launch the Server

The ASR pipelines in this guide run one ASR stage on one GPU. Qwen3-ASR needs
only the checkpoint path, plus `--enable-realtime` when the server should
accept live audio:

```bash
sgl-omni serve \
  --model-path "${MODEL_PATH}" \
  --model-name Qwen/Qwen3-ASR-1.7B \
  --enable-realtime \
  --port 8000
```

The other models start the same way with their own `--model-path`:

```bash
sgl-omni serve --model-path openai/whisper-large-v3 --port 8000
```

Qualified per-GPU profiles and the flags a model needs beyond its checkpoint
stay in its cookbook: Qwen3-ASR's RTX 4090 and RTX 5090 configs and its
speculative-decoding options, Whisper's speculative config, and the stage
settings for Nemotron Transcribe and MOSS-Transcribe-Diarize.

## Use Curl

### Transcribe

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=Qwen/Qwen3-ASR-1.7B \
  -F file=@tests/data/query_to_cars.wav \
  -F response_format=json
```

`response_format` accepts `json` (the default), `verbose_json`, and `text`.
`verbose_json` adds duration-based usage when the pipeline can probe the audio
duration, plus per-segment detail where the model adapter emits segments:
MOSS-Transcribe-Diarize parses its speaker markup into OpenAI-style `segments`,
and Qwen3-ASR reports one segment per transcribed chunk. Models whose adapter
derives real segment timestamps also accept `srt` and `vtt`; the others return
HTTP 400 before inference.

### Stream Transcription

Set `stream=true` to receive incremental transcript text over Server-Sent
Events, and use `curl -N` to disable client-side buffering:

```bash
curl -N -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=Qwen/Qwen3-ASR-1.7B \
  -F file=@tests/data/query_to_cars.wav \
  -F language=en \
  -F response_format=json \
  -F stream=true
```

The stream carries zero or more delta events, one final transcript event, and
the SSE sentinel:

```text
data: {"type":"transcript.text.delta","delta":"..."}

data: {"type":"transcript.text.done","text":"..."}

data: [DONE]

```

Deltas are batched by the pipeline; Qwen3-ASR and Ming-3.0-ASR default to a
50 ms interval. EOS and other terminal conditions flush any buffered text
before the final transcript event. Streaming accepts the `json` and `text`
response formats only; subtitle formats return HTTP 400 when combined with
`stream=true`.

### Live PCM Transcription

A realtime WebSocket session accepts audio as it arrives instead of a finished
multipart upload. Mount it with `--enable-realtime` at server start:

```bash
sgl-omni serve \
  --model-path "${MODEL_PATH}" \
  --model-name Qwen/Qwen3-ASR-1.7B \
  --enable-realtime \
  --port 8000
```

Qwen3-ASR exposes `/v1/realtime?intent=transcription`. Configure the session,
append base64-encoded mono 16 kHz PCM16 packets with
`input_audio_buffer.append`, and finalize a segment with
`input_audio_buffer.commit` or the configured server VAD. `transcription.done`
asks for `transcription.completed`; `input_audio_buffer.clear` discards the
current segment and keeps the session open.

```json
{
  "type": "session.update",
  "session": {
    "language": "English",
    "turn_detection": {
      "type": "server_vad",
      "threshold": 0.5,
      "prefix_padding_ms": 300,
      "silence_duration_ms": 500
    }
  }
}
```

Partial results are full replacements rather than append-only deltas, and an
event with `is_final=true` is immutable. Append events are not idempotent: after
a transport failure, reconnect and restart the transcription instead of
replaying packets on the old session. The Qwen3-ASR cookbook documents the
partial rollback, segment boundaries, and language reporting in full.

Nemotron 3.5 ASR uses the native path instead of `intent=transcription`: connect
to `/v1/realtime?model=nvidia/nemotron-3.5-asr-streaming-0.6b`, send mono 16 kHz
PCM16, and read text output. That path has no VAD, manual commit, language
field, or `max_new_tokens`.

## Use Python

### Basic Transcription

```python
import requests

with open("tests/data/query_to_cars.wav", "rb") as audio_file:
    resp = requests.post(
        "http://localhost:8000/v1/audio/transcriptions",
        data={
            "model": "Qwen/Qwen3-ASR-1.7B",
            "response_format": "json",
        },
        files={"file": ("query_to_cars.wav", audio_file, "audio/wav")},
        timeout=300,
    )

resp.raise_for_status()
print(resp.json()["text"])
```

### Consume the Delta Stream

Read the SSE body incrementally and append each delta. The done event repeats
the complete transcript, so it is safe to use it as the final result:

```python
import json

import requests

with open("tests/data/query_to_cars.wav", "rb") as audio_file:
    with requests.post(
        "http://localhost:8000/v1/audio/transcriptions",
        data={
            "model": "Qwen/Qwen3-ASR-1.7B",
            "response_format": "json",
            "stream": "true",
        },
        files={"file": ("query_to_cars.wav", audio_file, "audio/wav")},
        stream=True,
        timeout=300,
    ) as stream:
        stream.raise_for_status()
        parts = []
        for line in stream.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            if event["type"] == "transcript.text.delta":
                parts.append(event["delta"])
            elif event["type"] == "transcript.text.done":
                print(event["text"])
```

### Live Audio over WebSocket

```python
import asyncio
import base64
import json

import websockets


async def main():
    async with websockets.connect(
        "ws://localhost:8000/v1/realtime?intent=transcription"
    ) as ws:
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {"language": "English"},
        }))
        print(await ws.recv())

        with open("tests/data/query_to_cars.wav", "rb") as f:
            pcm16 = f.read()
        for start in range(0, len(pcm16), 3200):
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm16[start:start + 3200]).decode(),
            }))

        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        await ws.send(json.dumps({"type": "transcription.done"}))

        while True:
            event = json.loads(await ws.recv())
            print(event)
            if event["type"] == "transcription.completed":
                break


asyncio.run(main())
```

The example assumes the upload is already mono 16 kHz PCM16; decode other
formats first, because the session does not resample packets.

## Request Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Audio file uploaded as multipart form data. One file per request |
| `model` | string | served model | Model identifier |
| `language` | string | none | Optional language hint. The accepted set is model-specific; omit it for automatic detection. An unsupported hint returns HTTP 400 |
| `prompt` | string | none | Model-specific context: vocabulary biasing for Qwen3-ASR, topic or hotword context for some other models. See the model's cookbook for its exact effect |
| `response_format` | string | `json` | `json`, `verbose_json`, or `text`. `srt` and `vtt` are accepted only by adapters with real segment timestamps |
| `temperature` | float | `0` | Sampling temperature; `0` uses greedy decoding. A negative value is rejected |
| `repetition_penalty` | float | `1.0` | Penalty in `(0, 2]` on tokens already generated. Values above `1` discourage repeats and can stop a runaway loop on noisy or silent audio, but they also lower accuracy on clean audio, so keep `1.0` unless loops occur. Model support varies: Qwen3-ASR's DFlash draft rejects values other than `1.0` |
| `max_new_tokens` | integer | stage limit | Per-request generation-token cap |
| `stream` | boolean | `false` | Return SSE transcript deltas. Supports `json` and `text` only |

### Language Hints

When `language` is omitted, a model that supports detection identifies the
spoken language before transcribing. Set an explicit hint when the language is
known, or when automatic detection is unreliable for short or ambiguous audio.
The hint is a request field rather than a prompt prefix, so the accepted codes
and canonical names differ per model: Qwen3-ASR takes 30 codes, Ming-3.0-ASR
takes only `en` and `zh`, and Nemotron 3.5 ASR has no language field at all.
Unsupported values fail the request rather than falling back to English.

## Long Audio

Pipelines that can transcribe an isolated clip accept long uploads by splitting
them, running each chunk as its own engine request, and joining the transcripts
in order. The scheduling policy is the same four `audio_chunking` flags on
every such pipeline:

| Name | Meaning |
|---|---|
| `--audio_chunking.max_audio_clip_s` | Longest clip sent to the engine in one request, and therefore the chunk length. Capped at the model's native clip limit |
| `--audio_chunking.max_concurrent_chunks` | How many chunks of one request run in the engine at once. A per-request cap so one long upload cannot crowd out other requests |
| `--audio_chunking.max_total_audio_s` | Upper limit on one upload; above it the request fails with HTTP 400. It bounds a single decoded waveform, not the total across uploads |
| `--audio_chunking.max_concurrent_long_audio_requests` | How many long uploads the server admits at once. Past the cap a new long upload gets HTTP 503 instead of queueing; short uploads are never gated |

Defaults and native limits are model-specific, and each cookbook lists the
values it ships with. Qwen3-ASR defaults to 30 s clips, eight concurrent chunks,
a one-hour upload cap, and four admitted long uploads. Whisper's 30 s clip is
the hard edge of its mel window and can only be lowered. Nemotron Transcribe
keeps a one-hour upload as a single decoder request instead of chunking.

The long-audio share is an aggregate guard. Each admitted upload holds its
decoded waveform (float32 at 16 kHz), so decoded waveforms held at once are at
most
`max_concurrent_long_audio_requests × max_total_audio_s × 16000 × 4` bytes, and
engine slots long audio can occupy together are at most
`max_concurrent_long_audio_requests × max_concurrent_chunks`. The stock
derivation keeps that product at half of `--asr.engine.max_running_requests` so
short requests always have slots left.

Streaming cannot chunk. A stream request runs as one engine request and is
bounded by the model's `max_native_clip_s`; use `stream=false` for longer
uploads.

## Audio Translations

`/v1/audio/translations` translates source speech to English and is available
only where the pipeline declares the capability. See
[Audio Translations](audio_translations.md) for the model support matrix,
response formats, and the `language` source hint. Everything else on this page
applies to transcription only.

## Benchmarking

The shared ASR benchmark sweeps concurrency on SeedTTS reference audio through
`/v1/audio/transcriptions`. It defaults to Qwen3-ASR and accepts other
checkpoints through `--model-path`:

```bash
python -m benchmarks.eval.benchmark_asr_seedtts \
  --port 8000 \
  --model-path Qwen/Qwen3-ASR-1.7B \
  --concurrencies 1,2,4,8,16,32,64 \
  --repeats 3 --warmup
```

The report includes RTF (processing time divided by audio duration) and RTFx
(successful input-audio seconds divided by wall-clock seconds), next to
throughput, latency, and corpus WER. Per-model numbers and the revisions used to
reproduce them live in each cookbook; the full flag set, GPU sampling, and CI
setup are in the [ASR benchmark runbook](../developer_reference/asr_bench.md).

## Known Limitations

- The HTTP endpoint accepts one uploaded file per request. Live audio uses the
  realtime WebSocket and requires `--enable-realtime`.
- Audio is resampled to mono 16 kHz before transcription.
- Streaming runs as one engine request and cannot chunk. It is bounded by the
  model's native clip limit.
- An upload above `max_total_audio_s` returns HTTP 400; a long upload past the
  admitted count returns HTTP 503. Short requests are never gated.
- `/v1/audio/translations` is limited to models with translation capability;
  the other ASR models return HTTP 400.
- Model-specific caps still apply, such as Qwen3-ASR's 60-second whole-upload
  limit on Apple Silicon and Whisper's 30-second mel window.
