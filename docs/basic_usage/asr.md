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
| [Fun-ASR-Nano](../cookbook/fun_asr.md) | `--model-path` only | Multilingual transcription; `language` also selects the output language; no translation route |
| [ARK-ASR-3B](../cookbook/arkasr.md) | `--model-path` only | Multilingual transcription; `language` is recorded but does not change its fixed prompt; prefill coalescing is tuned per GPU |
| [Nemotron 3.5 ASR](../cookbook/nemotron3_5_asr.md) | `--model-path` only | Multilingual RNN-T transcription; greedy decoding only; native PCM streaming through `/v1/realtime` without VAD or manual commit |
| [Nemotron Transcribe](../cookbook/nemotron_transcribe.md) | `--model-path` only | Whole-file transcription up to one hour as one decoder request; no realtime, timestamps, or translation; validated on one H100 80 GB |
| [MOSS-Transcribe-Diarize](../cookbook/moss_transcribe_diarize.md) | `--model-path` only | Multi-speaker transcription, diarization and timestamps in one pass; up to ~90 minutes; `verbose_json` speaker segments and `srt`/`vtt` from real timestamps |
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
duration, plus per-segment detail where the model adapter emits segments. The
segment granularity is model-specific, and each cookbook shows the shape its
adapter returns. Models whose adapter derives real segment timestamps also
accept `srt` and `vtt`; the others return HTTP 400 before inference.

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
multipart upload. Start the server with `--enable-realtime`, then connect to
whichever endpoint shape the pipeline implements:

- **`/v1/realtime?intent=transcription`** — Qwen3-ASR and Hy-SimulMT serve this
  shared path. Configure the session with `session.update`, append base64-encoded
  mono 16 kHz PCM16 packets, finalize a segment with `input_audio_buffer.commit`
  or server VAD, and read `transcription.segment` events until
  `transcription.completed`. Session options, partial-update semantics, and
  rollback behavior are documented in the
  [Qwen3-ASR cookbook](../cookbook/qwen3_asr.md#live-pcm-transcription).
- **`/v1/realtime?model=...`** — Nemotron 3.5 ASR uses this native path. Send
  mono 16 kHz PCM16 and read text output; it has no VAD, manual commit, language
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

A complete WebSocket client that appends PCM packets, commits a segment, and
reads the completed transcript is in the
[Qwen3-ASR cookbook](../cookbook/qwen3_asr.md#live-pcm-transcription).
The session takes PCM16 packets as they arrive, so decode other formats first.

## Request Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Audio file uploaded as multipart form data. One file per request |
| `model` | string | served model | Model identifier |
| `language` | string | none | Optional language field. Its meaning and accepted values differ per pipeline; see Language Hints below |
| `response_format` | string | `json` | `json`, `verbose_json`, or `text`. `srt` and `vtt` are accepted only by adapters with real segment timestamps |
| `temperature` | float | `0` | Sampling temperature; `0` uses greedy decoding. A negative value is rejected, and some pipelines accept greedy decoding only |
| `stream` | boolean | `false` | Return SSE transcript deltas. Supports `json` and `text` only |

Fields beyond this set belong to individual pipelines instead of the shared
schema: `prompt` means vocabulary or hotword biasing on Qwen3-ASR, Fun-ASR-Nano
and MOSS-Transcribe-Diarize, previous-text conditioning on Whisper, and an
instruction override on Nemotron Transcribe, while Nemotron 3.5 ASR rejects it.
`repetition_penalty` and `max_new_tokens` exist only on some pipelines — for
example Qwen3-ASR, Fun-ASR-Nano and MOSS-Transcribe-Diarize — with different
defaults and bounds. Each cookbook lists the fields its pipeline accepts, their
defaults, and their limits.

### Language Hints

When `language` is omitted, a pipeline that supports detection identifies the
spoken language before transcribing. Set an explicit value when the language is
known, or when automatic detection is unreliable for short or ambiguous audio.
The field is model-specific in both its values and its effect: Qwen3-ASR takes
30 codes and their canonical names, Ming-3.0-ASR takes only `en` and `zh`,
Fun-ASR-Nano uses it to select the output language, ARK-ASR records it without
changing its fixed prompt, Nemotron 3.5 ASR takes `auto` or a checkpoint locale,
and Nemotron Transcribe exposes no language field at all. Unsupported values
fail the request rather than falling back to English.

## Long Audio

Pipelines that can transcribe an isolated clip accept long uploads by splitting
them, running each chunk as its own engine request, and joining the transcripts
in order. Qwen3-ASR and Whisper ASR share this splitting path, and with it the
same four `audio_chunking` flags:

| Name | Meaning |
|---|---|
| `--audio_chunking.max_audio_clip_s` | Longest clip sent to the engine in one request, and therefore the chunk length. Capped at the model's native clip limit |
| `--audio_chunking.max_concurrent_chunks` | How many chunks of one request run in the engine at once. A per-request cap so one long upload cannot crowd out other requests |
| `--audio_chunking.max_total_audio_s` | Upper limit on one upload; above it the request fails with HTTP 400. It bounds a single decoded waveform, not the total across uploads |
| `--audio_chunking.max_concurrent_long_audio_requests` | How many long uploads the server admits at once. Past the cap a new long upload gets HTTP 503 instead of queueing; short uploads are never gated |

Defaults and native clip limits are model-specific, and each cookbook lists the
values it ships with: Qwen3-ASR chunks toward its 1,200 s native clip, while
Whisper's 30 s clip is the hard edge of its mel window and can only be lowered.
Pipelines that do not use this path describe their own handling in their
cookbooks; Nemotron Transcribe, for example, keeps a one-hour upload as a single
decoder request.

For the shared splitting path, the long-audio share is an aggregate guard.
Each admitted upload holds its decoded waveform (float32 at 16 kHz), so decoded
waveforms held at once are at most
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
- Streaming runs as one engine request and cannot chunk where chunking
  applies. It is bounded by the model's native clip limit.
- On the shared splitting path, an upload above `max_total_audio_s` returns
  HTTP 400, and a long upload past the admitted count returns HTTP 503. Short
  requests are never gated.
- `/v1/audio/translations` is limited to models with translation capability;
  the other ASR models return HTTP 400.
- Model-specific caps still apply, such as Qwen3-ASR's 60-second whole-upload
  limit on Apple Silicon and Whisper's 30-second mel window.
