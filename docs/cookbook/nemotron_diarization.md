# Nemotron 3 Diarization

[NVIDIA Nemotron 3 Diarization preview](https://huggingface.co/nvidia/Nemotron-3-Diarization-preview)
is a Sortformer model that identifies up to eight speakers in a recording,
including overlapping speech. SGLang-Omni serves it through
`/v1/audio/diarizations`, which returns speaker labels and timestamps.
Use a separate ASR model if you also need a transcript.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md).
Use an NVIDIA GPU and Python 3.12, with FFmpeg installed for compressed audio.
NeMo is not required to serve the model.

## Server Configuration

The model runs on one GPU with FP32 weights. By default, it processes one
uploaded recording at a time, but you can increase concurrency with `max_concurrency`
(see [Concurrent Requests](#concurrent-requests)).

```bash
sgl-omni serve --config examples/configs/nemotron_diarization.yaml --port 8000
```

The example downloads a pinned checkpoint revision. To use a local checkpoint,
set `model_path` in your YAML configuration:

```yaml
config_cls: NemotronDiarizationPipelineConfig
model_path: /path/to/Nemotron-3-Diarization-preview.nemo
```

You can also point `model_path` to a directory containing that file. Use
`--config` for this model; automatic discovery with `--model-path` alone does
not support the `.nemo` archive.

## Diarize Audio

Upload a recording as multipart form data. The server converts it to mono and
resamples it to 16 kHz.

```bash
curl http://localhost:8000/v1/audio/diarizations \
    -F file=@conversation.wav
```

Example response:

```json
{
  "duration": 3.5,
  "segments": [
    {"start": 0.2, "end": 2.1, "speaker": "speaker_0"},
    {"start": 1.8, "end": 3.4, "speaker": "speaker_1"}
  ]
}
```

Times are seconds from the start of the recording, with 10 ms frame resolution.
Segments are sorted by start time and can overlap when speakers talk at once.
Silence returns an empty `segments` list. Speaker labels belong to each
recording: `speaker_0` in one request is not necessarily the same person as
`speaker_0` in another.

## Request Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `file` | file | required | Audio file uploaded as multipart form data |
| `model` | string | server default | Must match the served model name if provided |
| `response_format` | string | `json` | Only `json` is supported |
| `stream` | boolean | `false` | Only `false` is supported |

`/v1/audio/diarizations` is a SGLang-Omni extension. It rejects transcription
prompts, language selection, speaker-count overrides, and generation parameters.

## Inference Profiles

The default `offline` profile processes audio in chunks while keeping speaker
context across the recording. To use smaller chunks, select `low_latency`:

```bash
sgl-omni serve --config examples/configs/nemotron_diarization.yaml \
    --diarization.factory.profile low_latency --port 8000
```

Both profiles accept a complete recording and return one response after
processing finishes. The `low_latency` setting changes the model's chunking;
use the WebSocket endpoint below for live audio input.

## Concurrent Requests

To process multiple recordings at once, increase `max_concurrency`. For example:

```bash
sgl-omni serve --config examples/configs/nemotron_diarization.yaml \
    --diarization.factory.max_concurrency 2 --port 8000
```

Active requests share model weights and use separate speaker caches and CUDA
streams. Additional requests wait in the queue. Increasing concurrency uses more
GPU memory; check memory use and throughput with your recording lengths before
raising the limit. Both inference profiles support this setting.

## Live Audio

Use `/v1/audio/diarizations/stream` to send microphone audio over a WebSocket.
The same server command supports both uploads and live sessions. Live sessions
always use the low-latency profile: 720 ms chunks with 320 ms of lookahead,
plus a small amount of audio context for feature extraction. The first update
needs about 1.06 seconds of audio, plus inference and network time.

On your laptop, install the client dependencies and run the example:

```bash
pip install websockets sounddevice
python examples/nemotron_diarization_live.py --seconds 30
```

Allow microphone access if your operating system asks. The client prints speaker
labels and timestamps as you speak, then flushes the last audio when the recording
ends. To reach a remote server, forward its port with SSH:

```bash
ssh -N -L 8000:localhost:8000 user@gpu-host
```

You can also replay a recording at its original speed:

```bash
ffmpeg -i conversation.wav -ar 16000 -ac 1 -c:a pcm_s16le live.wav
python examples/nemotron_diarization_live.py --file live.wav
```

### WebSocket Protocol

Wait for `session.ready`, then send binary messages containing mono, 16 kHz,
little-endian signed PCM16 samples. Each message can contain up to one second of
audio (32,000 bytes). The microphone example captures 100 ms blocks and combines
queued blocks into messages of up to one second when catching up after a delay.
Its local buffer holds up to five seconds of audio before reporting an error.
Wait for `audio.ack`
after each message to avoid outrunning the server. These messages contain raw
samples, without a WAV header.

| Event | Direction | Meaning |
| --- | --- | --- |
| `session.ready` | server → client | Session ID and accepted audio format |
| `diarization.update` | server → client | New finalized time range (`start`, `end`) and its `segments` |
| `audio.ack` | server → client | Audio accepted; `processed_until` is the finalized timestamp |
| `audio.end` | client → server | Flush remaining audio and finish the recording |
| `diarization.done` | server → client | Final recording duration; the server then closes the socket |
| `session.reset` | client → server | Discard buffered audio and speaker state; start a new recording |
| `error` | server → client | Error message; the server then closes the socket |

Send control events as JSON, for example `{"type": "audio.end"}`. Updates contain
absolute timestamps from the start of the session and preserve overlapping
speakers. A speaker interval can continue in the next update; concatenate adjacent
intervals with the same speaker if you need a full-recording timeline. Earlier
updates are not revised. Silence produces updates with empty segment lists.
A reset returns a new `session.ready` and restarts timestamps and speaker labels.
Disconnecting releases the session, including any buffered audio.

The server keeps up to eight live sessions by default. Increase
`--diarization.factory.max_live_sessions` to change that limit. Each session keeps
its own bounded audio buffer and speaker cache while sharing model weights.
Live sessions use the single-stage pipeline in the example configuration; process
replicas are not supported for this endpoint.
`max_concurrency` controls simultaneous inference work across uploads and live
chunks; it is separate from the number of open sessions. A connection idle for
60 seconds is closed. The server rejects excess queued audio instead of dropping
samples, so timestamps remain aligned with the audio you sent.

## Known Limitations

- Up to eight speakers per recording. Audio with more speakers is still accepted,
  but the model cannot assign a separate label to each person.
- Simultaneous inference work is limited by `max_concurrency`, which defaults to `1`.
- Disconnecting a client discards pending results and releases its live session.
  An inference call already in progress finishes before its worker starts more work.

## Tests

Run the CPU tests for checkpoint validation, timestamp handling, and the endpoint:

```bash
pytest tests/unit_test/nemotron_diarization tests/unit_test/serve/test_diarizations.py \
    tests/unit_test/serve/test_diarization_ws.py
```

The GPU integration tests in `tests/test_model/test_nemotron_diarization.py`
compare both profiles with NeMo and exercise real HTTP requests. They require a
local checkpoint and the ASR dependencies from
[the pinned NeMo source](https://github.com/NVIDIA-NeMo/Speech/tree/2c1a2f91d64566b5d391b83df42f9ab4cd810adb).

```bash
NEMOTRON_DIARIZATION_CHECKPOINT=/path/to/checkpoint \
  python -m pytest tests/test_model/test_nemotron_diarization.py -q
```

Set `NEMOTRON_DIARIZATION_AUDIO_DIR` to include additional WAV recordings in the tests.
