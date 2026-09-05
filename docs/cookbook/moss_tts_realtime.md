# MOSS-TTS-Realtime

[MOSS-TTS-Realtime](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Realtime)
is a 1.7B context-aware streaming text-to-speech model from MOSI.AI and the
OpenMOSS team. It is designed for interactive voice agents: text can arrive
incrementally while audio is being generated, and successful turns retain the
model's text/audio KV context for the next turn in the same session. The model
supports 20 languages and emits **24 kHz, signed 16-bit, mono PCM** through the
[MOSS-Audio-Tokenizer](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer)
codec.

SGLang-Omni exposes two complementary APIs:

| API | Input contract | Use it for |
|---|---|---|
| `POST /v1/audio/speech` | The complete text is present when the request starts | Offline generation, evaluation, and OpenAI-compatible clients |
| `WS /v1/audio/speech/realtime` | `input.text` or `input.tokens` events can arrive while generation is running | LLM SSE deltas, live typing, and multi-turn voice agents |

The ordinary HTTP streaming mode (`stream=true`) streams audio for an already
complete input. It is not equivalent to the incremental-input realtime
WebSocket.

## Prerequisites

Install SGLang-Omni by following [Installation](../get_started/installation.md),
then download both public checkpoints:

```bash
hf download OpenMOSS-Team/MOSS-TTS-Realtime
hf download OpenMOSS-Team/MOSS-Audio-Tokenizer
```

The model processor is loaded from the trusted custom code shipped with the
checkpoint. The `websockets` package used by the client example is included in
the SGLang-Omni dependencies.

## Launch the Server

### Single GPU

The provided config colocates preprocessing, the AR engine, and the streaming
vocoder on one GPU. It enables the current performance defaults: a bfloat16
vocoder decoder and dense CUDA Graph capture for codec frame counts 1 through
12. Normal low-latency playback still follows the 1/2/3-frame ramp; a larger
captured shape is used only to consume frames that are already queued.

```bash
CUDA_VISIBLE_DEVICES=0 sgl-omni serve \
  --config examples/configs/moss_tts_realtime.yaml \
  --allowed-media-domain huggingface.co \
  --allowed-media-domain cas-bridge.xethub.hf.co \
  --port 8000
```

The media-domain flags are only needed by the remote reference-audio examples
below. The model-owned `/v1/audio/speech/realtime` endpoint is mounted
automatically; `--enable-realtime` controls the separate generic
`/v1/realtime` API and is not required here.

### Split Across Two GPUs

Use `MossTTSRealtimeSplitPipelineConfig` to keep the AR engine on the first
visible GPU and place codec encoding plus vocoding on the second visible GPU:

```yaml
# moss_tts_realtime_split.yaml
config_cls: MossTTSRealtimeSplitPipelineConfig
model_path: OpenMOSS-Team/MOSS-TTS-Realtime
codec_model_path: OpenMOSS-Team/MOSS-Audio-Tokenizer
vocoder_dtype: bfloat16
cuda_graph: true
```

```bash
CUDA_VISIBLE_DEVICES=0,1 sgl-omni serve \
  --config moss_tts_realtime_split.yaml \
  --port 8000
```

The GPU indices in the split config are relative to `CUDA_VISIBLE_DEVICES`.

### Verify the Runtime

Check readiness and inspect the effective vocoder settings:

```bash
curl -s http://localhost:8000/health

curl -s http://localhost:8000/model_info | jq '
  .stages[]
  | select(.stage == "vocoder")
  | .data
  | {
      codec_decoder_dtype,
      codec_cuda_graph_enabled,
      codec_cuda_graph_captured_frames
    }
'
```

With the example config, the expected values are `bfloat16`, `true`, and
`[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]`. CUDA Graph capture falls back to
eager execution if capture cannot be completed; `/model_info` and the server
log show the effective state.

## Complete-Input Speech API

Use the ordinary OpenAI-compatible endpoint when all text is already known.
The following request clones the supplied reference voice and returns a WAV
file:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "OpenMOSS-Team/MOSS-TTS-Realtime",
    "voice": "default",
    "input": "SGLang-Omni can serve both complete and incremental text input.",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "ref_text": "We asked over twenty different people, and they all said it was his.",
    "response_format": "wav",
    "seed": 43
  }' \
  --output output.wav
```

Reference audio is optional. Without it, the model uses its unconditioned
voice; with it, `ref_text` is recommended for a complete voice-cloning prompt.

To stream raw audio for a complete input, set `stream=true` and
`response_format=pcm`:

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "OpenMOSS-Team/MOSS-TTS-Realtime",
    "voice": "default",
    "input": "This complete sentence is decoded as streaming PCM.",
    "stream": true,
    "response_format": "pcm"
  }' \
  | ffmpeg -f s16le -ar 24000 -ac 1 -i pipe:0 output_stream.wav
```

## Incremental-Input Realtime API

The realtime endpoint uses JSON text frames for control/input events and binary
WebSocket frames for PCM audio:

```text
session.config
  -> session.configured
turn.start
  -> turn.started
input.text or input.tokens
  -> input.ack
input.done
  -> input.ack
  -> audio.start
  -> binary PCM chunks
  -> audio.done
  -> turn.done
next turn.start, or session.close
```

Audio can begin before `input.done` once enough incremental text is available,
so clients must be ready to receive binary frames while they are still sending
text.

### Complete Python Client

This example sends incremental text with one acknowledged update in flight,
collects PCM while input is still being sent, writes each turn to a WAV file,
then starts a second turn on the same WebSocket session:

```python
import asyncio
import json
import wave
from collections.abc import Iterable

import websockets


WS_URL = "ws://127.0.0.1:8000/v1/audio/speech/realtime"
MODEL = "OpenMOSS-Team/MOSS-TTS-Realtime"
SAMPLE_RATE = 24_000


async def receive_until(
    ws,
    expected_type: str,
    *,
    pcm: bytearray | None = None,
    accepted_seq_no: int | None = None,
) -> dict:
    while True:
        message = await ws.recv()
        if isinstance(message, bytes):
            if pcm is None:
                raise RuntimeError(
                    f"received binary audio while waiting for {expected_type}"
                )
            pcm.extend(message)
            continue

        event = json.loads(message)
        event_type = event.get("type")
        if event_type == "error":
            raise RuntimeError(event.get("message", "realtime TTS error"))
        if event_type != expected_type:
            continue
        if (
            accepted_seq_no is not None
            and event.get("accepted_seq_no") != accepted_seq_no
        ):
            continue
        return event


async def synthesize_turn(
    ws,
    *,
    turn_id: str,
    text_deltas: Iterable[str],
    output_path: str,
) -> None:
    await ws.send(json.dumps({"type": "turn.start", "turn_id": turn_id}))
    await receive_until(ws, "turn.started")

    pcm = bytearray()
    seq_no = 0
    for text in text_deltas:
        if not text:
            continue
        await ws.send(
            json.dumps(
                {
                    "type": "input.text",
                    "turn_id": turn_id,
                    "seq_no": seq_no,
                    "text": text,
                },
                ensure_ascii=False,
            )
        )
        await receive_until(
            ws,
            "input.ack",
            pcm=pcm,
            accepted_seq_no=seq_no,
        )
        seq_no += 1

    await ws.send(
        json.dumps(
            {
                "type": "input.done",
                "turn_id": turn_id,
                "seq_no": seq_no,
            }
        )
    )
    await receive_until(
        ws,
        "input.ack",
        pcm=pcm,
        accepted_seq_no=seq_no,
    )
    turn_done = await receive_until(ws, "turn.done", pcm=pcm)

    if not turn_done.get("committed"):
        raise RuntimeError(f"turn was not committed: {turn_done}")

    with wave.open(output_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm)


async def main() -> None:
    async with websockets.connect(WS_URL, max_size=None) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "session.config",
                    "session": {
                        "model": MODEL,
                        "voice": "default",
                        "response_format": "pcm",
                        "sample_rate": SAMPLE_RATE,
                        "temperature": 0.8,
                        "top_p": 0.6,
                        "top_k": 30,
                        "repetition_penalty": 1.1,
                        "repetition_window": 50,
                    },
                }
            )
        )
        configured = await receive_until(ws, "session.configured")
        assert configured["sample_rate"] == SAMPLE_RATE

        await synthesize_turn(
            ws,
            turn_id="turn-1",
            text_deltas=["The first ", "assistant response ", "arrives incrementally."],
            output_path="turn-1.wav",
        )
        await synthesize_turn(
            ws,
            turn_id="turn-2",
            text_deltas=["这是同一个会话里的", "第二轮语音。"],
            output_path="turn-2.wav",
        )

        await ws.send(json.dumps({"type": "session.close"}))
        await receive_until(ws, "session.closed")


asyncio.run(main())
```

Add `ref_audio`, `ref_text`, or `references` to the `session` object when voice
cloning is required. The reference is prepared once when the session is
configured and is reused by subsequent turns.

### Protocol Rules

- `seq_no` starts at `0` for every turn and increases without gaps. Exact
  retries of an already accepted event are idempotent, but reusing a sequence
  number with different content is rejected.
- A turn may use `input.text` or `input.tokens`, not both. `input.text` accepts
  arbitrary non-empty increments, including a single character.
- `input.tokens` must contain IDs from the exact tokenizer bundled with the
  served MOSS-TTS-Realtime checkpoint. Prefer `input.text` unless the upstream
  component deliberately shares that tokenizer.
- `input.done` closes only the current turn's input. No more text can be
  appended to that turn, but another unique `turn.start` can follow after
  `turn.done` on the same WebSocket.
- One WebSocket session can have only one active turn. `turn_id` values must be
  unique within the session.
- `turn.cancel` rolls back the active turn. Only a `turn.done` event with
  `committed=true` becomes reusable multi-turn context.
- `turn.start.user` is optional. When supplied, it must contain both `text` and
  `audio`; this adds the preceding user utterance and acoustics to the turn
  context.

### Voice Continuity Across Turns

A session's voice stays continuous across committed turns: the model keeps its
multi-turn KV context, and the streaming codec keeps its causal decoder state
as well. Codec state is leased to the session (not to an individual turn), so
turn N+1's audio continues exactly where turn N ended — there is no per-turn
acoustic reset.

The codec context is released and reset only on session close, idle session
expiry (`session_idle_ttl_s`), a `turn.cancel`/abort, or an internal failure.
A cancelled turn's partial audio is deliberately discarded from the causal
context so it cannot leak into the next turn.

The streaming codec pool holds one slot per held session plus per active turn
(`limits.max_held_sessions + limits.max_active_turns`, 80 by default). Idle
sessions retain their slot until close or TTL expiry, so deployments that keep
many long-idle sessions should budget the slot memory accordingly (the engine
startup log and `/model_info` report the derived codec reserve and the current
`codec_held_sessions` count).

## Bridging an LLM SSE Stream

For a voice assistant, forward the first non-empty text-bearing SSE delta
immediately. Do not wait for a sentence boundary, punctuation, a fixed
character count, or the LLM's final event. When the LLM's TTFT is reliably below
the socket idle limit, configure the WebSocket before the assistant text
arrives; otherwise open it when the first useful delta arrives.

The published checkpoint's realtime processor uses a 12-token prefill window.
SGLang-Omni starts the first AR prefill as soon as the accumulated text reaches
that tokenizer threshold, or as soon as `input.done` arrives for a shorter
non-empty turn. Sending the first delta immediately therefore reaches the
earliest possible generation point even when that first delta alone is shorter
than 12 tokens.

Recommended flow:

1. If the LLM's TTFT is reliably below 30 seconds, configure the TTS WebSocket
   ahead of the LLM request and start the TTS turn immediately before it. If
   TTFT can exceed 30 seconds, open/configure the socket and start the turn when
   the first non-empty delta arrives.
2. Send the first useful delta immediately as `input.text(seq_no=0)`.
3. Keep at most one input update awaiting `input.ack`. While it is outstanding,
   concatenate new raw SSE deltas; send the accumulated text as the next update
   when the acknowledgement arrives. This bounds protocol pressure without
   delaying the first delta.
4. Send `input.done` with the next sequence number only after the LLM stream has
   finished.
5. Play binary PCM frames as they arrive; do not wait for `audio.done`.

Preserve the raw delta order. In particular, do not repeatedly submit the full
assistant response: every `input.text` value is an append-only increment.

## Session Configuration

`session.config` accepts the following model-specific generation defaults:

| Field | Default | Notes |
|---|---:|---|
| `temperature` | `0.8` | Sampling temperature; `0` is allowed |
| `top_p` | `0.6` | Nucleus-sampling threshold |
| `top_k` | `30` | Top-k sampling |
| `repetition_penalty` | `1.1` | Repetition penalty |
| `repetition_window` | `50` | Recent-token window used by the model |
| `max_new_tokens` | model default (`1000`) | Maximum generated realtime steps |
| `seed` | unset | Non-negative deterministic sampling seed |
| `response_format` | `pcm` | The realtime endpoint accepts PCM only |
| `sample_rate` | `24000` | Fixed by the codec |

The HTTP `/v1/audio/speech` path keeps these MOSS-TTS-Realtime model defaults
unless the caller explicitly supplies a generation field.

## Deployment Overrides

Important pipeline fields and their defaults are:

| Field | Default | Meaning |
|---|---:|---|
| `vocoder_dtype` | `bfloat16` | Decoder precision; set `float32` for compatibility fallback |
| `cuda_graph` | `true` | Enable streaming vocoder CUDA Graph capture |
| `cuda_graph_frames` | `null` | Densely capture codec frame counts 1 through 12 |
| `cuda_graph_min_free_gb` | `3.0` | Minimum free memory required before capture |
| `limits.max_sessions` | `64` | Logical realtime sessions admitted by the backend |
| `limits.max_active_turns` | `16` | Concurrent turns across all WebSockets; one per WebSocket |
| `limits.input_idle_timeout_s` | `30` | Backend timeout while an active turn waits for more input |
| `limits.turn_timeout_s` | `600` | Maximum backend turn lifetime |
| `limits.session_idle_ttl_s` | `300` | Backend idle-session retention limit |

Config fields can be changed in YAML or through dotted extra CLI arguments. For
example, this runs the vocoder decoder in FP32 and disables its CUDA Graph:

```bash
sgl-omni serve \
  --config examples/configs/moss_tts_realtime.yaml \
  --vocoder-dtype float32 \
  --cuda-graph false \
  --port 8000
```

`cuda_graph_frames` accepts explicit frame counts from 1 through 25. Capturing
the dense 1-through-25 range is useful only when a deployment expects unusually
large vocoder backlogs. It increases startup capture time and graph memory, but
does not make the normal path wait for 25 frames: the scheduler always consumes
the largest captured shape that fits the frames already pending.

To increase the backend active-input timeout:

```bash
sgl-omni serve \
  --config examples/configs/moss_tts_realtime.yaml \
  --limits.input-idle-timeout-s 60 \
  --port 8000
```

## Troubleshooting

### The WebSocket Closes After an Idle Gap

The public speech-realtime WebSocket currently enforces a 30-second
application-level gap between client text frames. This is separate from the
backend `limits.input_idle_timeout_s` setting. Open the turn close to the first
LLM delta and keep client messages less than 30 seconds apart.

For a long active turn, an exact retry of the most recently acknowledged input
event (same type, `turn_id`, `seq_no`, and content) is safe and receives another
`input.ack` without appending duplicate text. If no turn has started yet, wait
to send `turn.start` until useful text is available. After `turn.done`, start
the next turn or close the session before the socket idle limit.

### Generation Does Not Start After One Character

`input.text` accepts one character immediately, but acceptance is not the same
as AR prefill. The published checkpoint waits for 12 tokenizer tokens, unless
`input.done` closes a shorter non-empty turn. Continue forwarding deltas rather
than buffering them client-side.

### Sequence Gap or Mixed Input Mode

Use `next_seq_no` from `turn.started` and `input.ack`. Do not skip sequence
numbers, and do not switch between `input.text` and `input.tokens` within a
turn.

### Text Is Rejected After `input.done`

`input.done` is final for that turn. Wait for `turn.done`, then send a new
`turn.start` with a previously unused `turn_id`.

### Reference Audio Is Rejected

Remote URLs must satisfy `--allowed-media-domain`. Local files must be under
the directory passed through `--allowed-local-media-path`. A base64 audio data
URI can also be supplied in `ref_audio`.

### CUDA Graph Is Not Active

Inspect the vocoder entry in `/model_info`. If
`codec_cuda_graph_captured_frames` is empty, check the server log for the
capture error and available-memory gate. Serving continues through the eager
vocoder path. Set `cuda_graph: false` when an eager-only deployment is desired.
