# MiniCPM-o 4.5 native full duplex

This package integrates `openbmb/MiniCPM-o-4_5` as a native SGLang-Omni
pipeline. The active path does **not** launch the official Demo, a nested model
worker, or a JSONL/RPC subprocess.

The single GPU stage combines:

- MiniCPM-V 4.5 vision and streaming Whisper perception;
- Qwen3 decoding through SGLang `ModelWorker`, scheduler, paged KV pools, and
  `StreamingSession`;
- the checkpoint's `tts.` network and Token2wav as in-process side components;
- the SGLang-Omni Realtime WebSocket and duplex epoch-fencing protocol.

The official Demo remains the behavioral reference for one-second units,
listen/speak decisions, deferred chunk finalization, barge-in, TTS KV state,
and Token2wav lookahead. Its FastAPI/WebSocket/worker layers are not copied.

## Install and launch

Token2wav is optional at package level. Install the compatible provider extra
for speech output. Run these commands from the repository root:

```bash
pip install -e '.[minicpmo-o]'
```

Then launch the checked-in recipe:

```bash
CUDA_VISIBLE_DEVICES=0 sgl-omni serve \
  --config examples/configs/minicpmo_o_4_5_duplex.yaml \
  --enable-realtime
```

When both paths are omitted, the checkpoint asset `assets/HT_ref_audio.wav` is
used for both the Token2wav prompt and the LLM system voice reference, matching
the official duplex path. Set `prompt_wav_path` and `ref_audio_path` in the
recipe to override either side independently. A session can also
send inline reference audio in `session.update.voice` before its first model
unit is submitted. Inline reference fields are base64-encoded raw 16 kHz mono
float32 little-endian samples (not WAV or PCM16), with a combined 30-second
limit.

The recipe pins `revision` so the SGLang weights, remote processor code, and
TTS assets resolve to the same checkpoint commit. For an air-gapped or fully
audited deployment, download that revision first and set `model_path` to the
local snapshot directory.

To exercise perception/text output without Token2wav, set:

```yaml
duplex_sampling:
  generate_audio: false
```

## Realtime contract

Connect to `ws://127.0.0.1:8000/v1/realtime`. Input audio is base64-encoded,
16 kHz mono PCM16:

```json
{
  "type": "input_audio_buffer.append",
  "audio": "<base64 PCM16>",
  "video_frames": []
}
```

Client chunks may be smaller than one second. The Realtime layer buffers them
and submits a model unit whenever it has exactly 16,000 samples (32,000 bytes).
Up to eight encoded video frames may accompany a unit. Useful control events
are:

- `session.update` before the first unit;
- `response.cancel` for barge-in and response-epoch fencing;
- `response.audio.playback_ack` with a monotonic `audio_end_ms`;
- `input_audio_buffer.clear` and `session.close`.

Text is emitted as `response.text.delta`. Audio is emitted as
`response.audio.delta` containing base64 PCM16 at 24 kHz. A completed model
unit produces `input_audio_buffer.processed`; a natural speaking-turn boundary
produces `response.text.done`, `response.audio.done`, and `response.done`
without closing the duplex session. Session terminal events follow normal
SGLang-Omni Realtime naming.

## Correctness constraints in the first version

- CUDA, TP=1, one live session, and model batch size 1;
- overlap scheduling, CUDA Graph, async decode, and chunked prefill disabled;
- explicit listen/speak mode only;
- 16,000 input samples per model unit, at most eight video frames, and bounded
  public/internal inline payloads;
- Whisper attention cache resets at its learned 1,500-position boundary while
  the Qwen paged-KV session remains intact. This is a hard APM encoder-cache
  reset at roughly 30 seconds, not an audio sliding window or LLM context
  rebase; quality across the real boundary still needs CUDA E2E validation;
- the LLM context has no high-watermark rebase yet. Replayable media embeddings
  are offloaded to host memory after commit and released on session close;
- a cleanup failure poisons the stage so a partially reset model cannot be
  silently reused;
- Token2wav 0.1.1 keeps its voice-prompt cache on the provider object. The
  enforced one-session limit makes this safe in the first release; supporting
  concurrent sessions requires that cache to become session-owned too.

CPU/fake-component tests cover protocol, perception chunking, request/session
bridging, TTS state, and Realtime translation. Real checkpoint loading and
multi-turn CUDA/TTS output still require validation on a supported NVIDIA
host; they are not implied by those unit tests.
