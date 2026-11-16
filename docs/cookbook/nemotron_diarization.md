# Nemotron 3 Diarization

[NVIDIA Nemotron 3 Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) is a Sortformer model that finds who spoke when for up to eight speakers, including overlapping speech. SGLang-Omni serves it natively, without NeMo, through `/v1/audio/diarizations` for uploaded recordings and `/v1/audio/diarizations/stream` for live audio. The model returns speaker intervals, not transcripts, so use a separate ASR model if you also need text.

## Launch

The server runs on one NVIDIA GPU and needs FFmpeg for compressed audio. It downloads only `config.json` and the `.nemo` archive from the repository and computes in FP32.

```bash
sgl-omni serve --model-path nvidia/Nemotron-3-Diarization --port 8000
```

To pin the checkpoint revision, use the example configuration. To serve a local archive, set `model_path` to the `.nemo` file or to a directory that contains `Nemotron-3-Diarization.nemo`:

```yaml
config_cls: NemotronDiarizationPipelineConfig
model_path: /path/to/Nemotron-3-Diarization.nemo
```

```bash
sgl-omni serve --config examples/configs/nemotron_diarization.yaml --port 8000
```

The first request of each new window shape compiles attention kernels, so measure warm requests separately. Results are reproducible on the same hardware, software, and inference profile.

## Diarize a Recording

Upload a recording as multipart form data. The server converts it to mono and resamples it to 16 kHz.

```bash
curl http://localhost:8000/v1/audio/diarizations -F file=@conversation.wav
```

```json
{
  "duration": 3.5,
  "segments": [
    {"start": 0.2, "end": 2.1, "speaker": "speaker_0"},
    {"start": 1.8, "end": 3.4, "speaker": "speaker_1"}
  ]
}
```

Times are seconds from the start of the recording, with 10 ms resolution. Segments are sorted by start time and overlap when speakers talk at once. Silence returns an empty `segments` list. Speaker labels belong to one recording, so `speaker_0` in one request is not necessarily the same person in another.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `file` | file | required | Audio file uploaded as multipart form data |
| `model` | string | server default | Must match the served model name if provided |
| `response_format` | string | `json` | Only `json` is supported |
| `stream` | boolean | `false` | Only `false` is supported |

The endpoint is a SGLang-Omni extension. It rejects transcription prompts, language selection, speaker-count overrides, and generation parameters, and other models reject it.

## Inference Profiles

The default `offline` profile processes 27.2 second chunks while keeping speaker context across the recording. The `low_latency` profile uses 0.72 second chunks with 0.32 seconds of lookahead, matching live audio:

```bash
sgl-omni serve --model-path nvidia/Nemotron-3-Diarization --port 8000 \
    --diarization.factory.profile low_latency
```

Both profiles return one response after the whole upload is processed.

## Concurrent Requests

The server advances each recording one chunk at a time. A new recording or live operation runs its first chunk before active recordings continue, and active recordings then finish in arrival order. These factory options control capacity:

| Option | Default | Meaning |
| --- | --- | --- |
| `max_concurrency` | `1` | Chunks that run at once, each on its own CUDA stream. It does not batch recordings. |
| `max_running_requests` | `16` | Recordings and live operations that hold model state at once. Each upload keeps its features until it finishes. |
| `max_queued_requests` | `64` | Additional uploads or new live sessions that wait. Beyond this, uploads receive HTTP 503 and new live sessions receive an `error` event. |
| `max_live_sessions` | `8` | Open live sessions. Audio and control events of an open session are never rejected, but they wait while all running slots are busy. |

Set them as `--diarization.factory.<option> <value>`. A cancelled upload releases its state after the current chunk finishes.

## Live Audio

Live sessions always use the low-latency profile. The first update needs about 1.06 seconds of audio plus inference and network time. Try it from a laptop microphone or replay a mono 16 kHz PCM16 WAV file at its original speed:

```bash
pip install websockets sounddevice
python examples/nemotron_diarization_live.py --seconds 30
python examples/nemotron_diarization_live.py --file live.wav
```

The client imports sounddevice in both modes, which needs the PortAudio library on Linux (for example `apt install libportaudio2`). Convert other recordings for replay with `ffmpeg -i input.wav -ar 16000 -ac 1 -c:a pcm_s16le live.wav`.

After `session.ready`, send binary messages of mono, 16 kHz, little-endian PCM16 samples without a WAV header, up to one second (32,000 bytes) per message, and wait for `audio.ack` after each one. Send control events as JSON, for example `{"type": "audio.end"}`.

| Event | Direction | Meaning |
| --- | --- | --- |
| `session.ready` | server to client | Session ID and accepted audio format |
| `diarization.update` | server to client | Newly finalized time range (`start`, `end`) and its `segments` |
| `audio.ack` | server to client | Audio accepted, with `processed_until` as the finalized timestamp |
| `audio.end` | client to server | Flush remaining audio and finish the recording |
| `diarization.done` | server to client | Final duration, after which the server closes the socket |
| `session.reset` | client to server | Drop buffered audio and speaker state, then start a new session |
| `error` | server to client | Error message, after which the server closes the socket |

Updates use absolute timestamps from the start of the session and are never revised. An interval can continue in the next update, so join adjacent intervals of the same speaker for a full timeline. Live sessions need the single-stage pipeline without process replicas. A connection idle for 60 seconds is closed, and disconnecting releases the session.

## Limitations

- At most eight speakers per recording. Audio with more speakers is accepted, but the model cannot label each person separately.
- Recordings are processed one chunk at a time per request, without batching across requests.
