# Full-duplex audio examples

Run commands from the repository root. The native audio route uses `/v1/realtime`
with `session.update`, `input_audio_buffer.append`, `sglang.input_audio.end`, and
`session.close`. Wait for `sglang.input_audio.drained` before closing after input EOS.

## MiniCPM-o

Set `model_path` in `minicpmo.yaml` to a prepared MiniCPM-o 4.5 checkpoint, then run:

```bash
sgl-omni serve --config examples/full_duplex/minicpmo.yaml --enable-realtime
```

The `MiniCPMODuplexPipelineConfig` is also available as the `session` model variant.
The existing `text` and `speech` variants keep their ordinary request pipelines.
An optional top-level `reference_audio` path supplies the reference for both
perception and speech; the default is the checkpoint's `assets/HT_ref_audio.wav`.

The native path accepts mono PCM16 at 16 kHz and emits 24 kHz audio and text over
`/v1/realtime`. It processes one-second units with session-resident encoder,
thinker KV, sampler history, TTS, and vocoder state. Empty input EOS reaches the
speech stage to flush pending audio without an additional thinker request.

This integration uses the current shared native protocol: open, append, and close.
It does not expose the historical epoch/cancel or `sglang.microturn.done` events.
Video input is not wired into this audio path.
