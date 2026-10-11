# Nemotron 3.5 ASR

[Nemotron 3.5 ASR Streaming 0.6B](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
is a multilingual speech-recognition model with a FastConformer encoder and
an RNN-T decoder. SGLang-Omni supports complete-file transcription through
`/v1/audio/transcriptions` and native cache-aware PCM streaming through the shared
WebSocket session API.

## Prerequisites

Follow [Installation](../get_started/installation.md), then run the examples
from the repository root. Use the repository's pinned Transformers version;
the Nemotron compatibility implementation is included.

For Intel GPUs, follow the [Nemotron 3.5 ASR XPU recipe](../get_started/installation_xpu.md#nemotron-35-asr-speech-to-text-single-xpu).

## Server Configuration

The default pipeline runs one ASR stage on one GPU in `float32`:

```bash
sgl-omni serve \
  --model-path nvidia/nemotron-3.5-asr-streaming-0.6b \
  --port 8000
```

Tune the ASR stage with `--asr.factory.*` flags:

| Option | Default | Description |
|---|---|---|
| `dtype` | `float32` | Model dtype |
| `num_lookahead_tokens` | `3` | Encoder right context; the checkpoint supports `0`, `3`, `6`, and `13` |
| `max_batch_size` | `8` | Maximum number of requests in a scheduler batch |
| `max_batch_wait_ms` | `2.0` | Maximum wait to form a batch, in milliseconds |
| `enable_encoder_cuda_graph` | `true` | Capture steady-state streaming encoder batches at startup on CUDA |
| `encoder_graph_max_batch_size` | `max_batch_size` | Capture every exact batch size from 1 through this limit, bounded by pool capacity |
| `session_max_concurrency` | `max(4, max_batch_size)` | Concurrent hooks/ordinary requests per ASR replica |
| `max_open_sessions` | `64` | Open streams per ASR replica |
| `max_state_bytes` | `8 GiB` | Total reserved session state budget, excluding shared model weights |
| `max_pcm_bytes` | `2 MiB` | Buffered PCM per session; consumed audio is discarded except overlap |
| `max_history_tokens` | `16384` | Bounded token/duration history; excess fails the session |
| `max_text_bytes` | `2 MiB` | Combined raw/clean transcript limit |

For example:

```bash
sgl-omni serve \
  --model-path nvidia/nemotron-3.5-asr-streaming-0.6b \
  --asr.factory.num_lookahead_tokens 3 \
  --asr.factory.max_batch_size 8 \
  --asr.factory.max_batch_wait_ms 2 \
  --port 8000
```

Within each scheduler batch, complete-file requests with the same
`max_new_tokens` value share one model `generate()` call. Different token limits
are processed in separate batches to preserve each request's output limit.

## Transcribe Audio

Upload a complete audio file; the server converts it to mono 16 kHz audio:

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F model=nvidia/nemotron-3.5-asr-streaming-0.6b \
  -F file=@tests/data/query_to_cars.wav \
  -F language=auto \
  -F response_format=verbose_json
```

Use `response_format=text` for the transcript alone, or `json` for a JSON
response. `verbose_json` also includes duration and language information.
Locale tags are removed from the transcript. With `language=auto`, the
response reports a language when the model emits one unambiguous locale tag.

## Request Parameters

| Parameter | Default | Description |
|---|---|---|
| `file` | required | Audio file uploaded as multipart form data |
| `model` | server default | Model identifier |
| `language` | `auto` | Checkpoint-defined locale or language code, matched case-insensitively; `auto` enables language detection |
| `response_format` | `json` | `json`, `verbose_json`, or `text`; streaming responses accept only `json` or `text` |
| `temperature` | `0` | Greedy RNN-T decoding only; non-zero values are rejected |
| `max_new_tokens` | model default | Optional positive output-token limit |
| `prompt` | unset | Text prompts are unsupported; non-empty values are rejected |
| `stream` | `false` | Stream the HTTP response after a complete file upload; see below |

Supported language values come from the checkpoint's prompt dictionary.
Unsupported values fail before model inference. Nemotron supports transcription
only; `/v1/audio/translations` returns HTTP 400. Segment timestamps, SRT/VTT
output, and speaker diarization are not supported.

## Native Streaming

Enable realtime on the server with `--enable-realtime`, then connect to
`/v1/realtime?model=nvidia/nemotron-3.5-asr-streaming-0.6b`.
This uses the shared native path (default `intent=conversation`) for ASR text.
It does not add support for `intent=transcription`, VAD, manual commit, or
conversation generation.

The public configuration accepts 16 kHz mono PCM16 and text output. Language
is auto and decoding uses the model default; the shared schema has no language
or max_new_tokens fields. Instructions are unsupported. Ordinary file
transcription parameters above are unchanged.

Run this example with a mono 16 kHz PCM16 WAV file:

```python
import asyncio
import base64
import json
import wave

import websockets


async def transcribe(path: str) -> None:
    with wave.open(path, "rb") as audio:
        assert (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) == (1, 2, 16000)
        pcm = audio.readframes(audio.getnframes())
    url = "ws://localhost:8000/v1/realtime?model=nvidia/nemotron-3.5-asr-streaming-0.6b"
    async with websockets.connect(url) as websocket:
        assert json.loads(await websocket.recv())["type"] == "session.created"
        await websocket.send(json.dumps({
            "event_id": "configure", "type": "session.update",
            "session": {"output_modalities": ["text"]},
        }))
        assert json.loads(await websocket.recv())["type"] == "session.updated"

        async def receive() -> None:
            while True:
                event = json.loads(await websocket.recv())
                if event["type"] == "error":
                    raise RuntimeError(event)
                elif event["type"] == "response.output_text.delta":
                    print(event["delta"], end="", flush=True)
                elif event["type"] == "response.output_text.done":
                    print("\nFinal:", event["text"])
                elif event["type"] == "sglang.input_audio.drained":
                    return
                else:
                    pass

        reader = asyncio.create_task(receive())
        for sequence, offset in enumerate(range(0, len(pcm), 640)):
            await websocket.send(json.dumps({
                "event_id": f"audio-{sequence}", "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm[offset:offset + 640]).decode(),
                "sglang": {"seq": sequence},
            }))
            await asyncio.sleep(0.02)
        await websocket.send(json.dumps({
            "event_id": "end", "type": "sglang.input_audio.end",
        }))
        await reader
        await websocket.send(json.dumps({
            "event_id": "close", "type": "session.close",
        }))
        while json.loads(await websocket.recv())["type"] != "session.closed":
            pass


asyncio.run(transcribe("tests/data/query_to_draw.wav"))
```

One session is one recognition stream. EOS flushes a residual window and
emits exactly one final, including an empty final for an empty stream.
Audio after EOS is rejected. Close/disconnect cancels and releases state;
it does not flush. Clear only discards PCM still pending in the realtime
runtime, preserving an active unit and model caches already submitted to
the stage. Open a new session for another recognition stream.

The 20 ms native unit is an input operation, not a model window or a first
text latency guarantee. Deployment callers can configure native_unit_ms
on create_realtime_deployment. Short units complete without waiting for a
future window; long appends drain all currently runnable windows.

Each ASR replica owns one engine and model thread. Different sessions share
model batches while retaining separate attention, convolution and RNN-T
caches. H concurrent hook threads plus one inbox bridge thread feed that
engine; if H is smaller than max_batch_size, the public path cannot submit a
full batch. Replicas do not share batches. Offline and streaming work
alternate at batch boundaries when both are ready; a long offline forward
cannot be preempted. max_batch_wait_ms is a collection deadline, not an
end-to-end latency bound.

Model task, PCM, cache reservation, token history and text budgets are
bounded. Usage reports actual persistent cache/history/PCM bytes; batch
temporaries and shared weights are measured separately at process level.
When an explicit internal-session token limit is reached, later PCM is
validated and counted toward duration without further inference or buffering.
Budget errors after stage acceptance terminate the stream; they are not a
promise that retrying the same seq is safe. Public admission rejection follows
the shared runtime's sequence/retry contract.

### Persistent encoder state pool

Streaming uses persistent attention and convolution buffers allocated before
the model thread starts. Each session leases one slot; batches pass slot IDs to
tensor gather/write-back operations.
The encoder retains up to `sliding_window - 1` history frames per slot. Logical
frame counts and masks handle history filling and the sliding window without
growing or replacing the pool buffers. First and subsequent windows are grouped
for their different causal-convolution padding, and output rows are restored to
request order before RNN-T decoding.

Pool capacity is the smaller of `max_open_sessions` and the number of complete
session reservations that fit `max_state_bytes`. Each reservation includes the
full encoder slot, decoder state, and the configured PCM/history/text budgets.
A budget too small for one slot fails at startup. The pool occupies its full
capacity even when idle; active-session usage reports each leased slot once,
while batch temporaries and allocator overhead remain outside the state budget.

EOS releases the slot after the final window has finished. Reaching the decoder
token limit, cancellation, close, failure, and shutdown also release it. Reused
slots are cleared before another session receives them. Completed sessions keep
their transcript and RNN-T state until close, but no longer own encoder storage.
All pool access and slot recycling use the existing serialized model owner.

The pool is the streaming encoder's sole state representation on all devices.
Disable `enable_encoder_cuda_graph` for pooled eager execution. Offline
transcription continues to use model generation.

### Streaming encoder CUDA graphs

Before the model thread starts, capture every exact batch size up to
`encoder_graph_max_batch_size` (the scheduler's `max_batch_size` by default),
bounded by pool capacity. With default limits this captures B8 through B1,
largest first, using one shared graph memory pool. There is no runtime capture.
First windows and uncaptured shapes or batch sizes use the pooled eager encoder.
Capture failures fail startup; disable graphs explicitly to run entirely eager.

Each graph uses the fixed subsequent-window mel length for the configured
lookahead. History length does not add another bucket: device-side per-slot
frame counts construct the attention mask inside the graph. Runtime copies mel
features, prompt IDs, and slot IDs into static inputs. Attention and convolution
history stay in the persistent pool; their batched gathers and writes, together
with device frame-count updates, are captured operations. No per-request history
packing or post-replay cache scatter runs in Python. Pool operations still move
history data on the device; this is not a zero-copy attention implementation.

The model owner serializes all replays and eager pool access. Only one group in
each batch can replay; the batch concatenation restores request order and copies
the borrowed graph output into caller-owned storage before any subsequent replay.
Shutdown synchronizes and releases graphs before releasing pool storage.
Graph workspaces and static inputs/outputs
are additional allocations, outside `max_state_bytes`.

CPU parity and lifecycle tests do not establish GPU performance or
real-checkpoint recognition accuracy; CUDA replay tests require an NVIDIA GPU.

Setting stream=true on /v1/audio/transcriptions streams the response to a
complete uploaded file. It does not select native PCM input.
