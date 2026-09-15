# Native VoiceChat sessions (local integration prototype)

This opt-in pipeline connects the existing VoiceChat model implementation to
pipeline sessions (#2035), the AR streaming-session bridge (#2069), and the
shared realtime endpoint (#2070). It needs all three dependencies. The default
offline VoiceChat configuration is unchanged.

## Run

Use the checkpoint and tokenizer downloads in [the offline cookbook](nemotron_voicechat.md).
The initial configuration supports one session on one GPU, with four separate
stage processes. On B200, Talker explicitly uses `triton` attention because
the default TRTLLM context kernel does not support its head dimension of 72.

```bash
CUDA_VISIBLE_DEVICES=0 python examples/run_nemotron_voicechat_duplex.py \
  --model-path /path/to/NVIDIA-NemotronLabs-VoiceChat-11B \
  --audio /path/to/NVIDIA-NemotronLabs-VoiceChat-11B/turn_taking.wav \
  --paced --out reply.wav
```

The example checks the output sample count after **input drained**, writes a
22.05 kHz PCM WAV and a JSON timing/transcript report, and closes the session and
worker processes. EOS acceptance alone does not mean inference has finished.

To serve on localhost:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/run_nemotron_voicechat_duplex.py \
  --model-path /path/to/NVIDIA-NemotronLabs-VoiceChat-11B --serve --port 8097
```

Connect to `ws://127.0.0.1:8097/v1/realtime`. Send `session.update` with
`{"output_modalities":["audio"]}` and wait for `session.updated`. The advertised
input is mono PCM16 at 16 kHz and output is PCM16 at 22050 Hz. Send
`input_audio_buffer.append` with base64 audio and `sglang.seq` starting at zero.
The shared runtime assembles arbitrary client chunks into 1280-sample units.
Use `sglang.input_audio.end` to pad a final partial unit and drain the codec;
wait for `sglang.input_audio.drained`. Use `session.close` to release resources.
`response.cancel` fences old output while retaining accepted input and model state.
Clients must discard queued audio from the old epoch when cancellation is acknowledged.

## Execution and state

The fixed route is `perception → thinker → talker → code2wav`.

- Perception owns its causal convolution and attention caches through `SessionHooks`.
- Thinker forwards the prompt plus frame 0 once. Every later unit uses the
  previous output token and function token together with the new acoustic row.
- Talker forwards the voice prompt plus its first frame once; subsequent rows
  fuse the prior codes and current text token.
- Both AR stages use the shared bridge's native session, including its retained
  request slot and KV. A continuation appends **no new token IDs**: the prior
  unit's last sampled, unprocessed token is the next frame's input position.
  Model runners supply fused rows for exactly the uncached suffix, with a hard
  alignment check. Bounded history also allows correct prefix replay.
- Codec retains at most 16 code frames and holds back 256 samples until the next
  unit or EOS. An empty EOS drains that tail without another thinker/talker step.
- A response spans units until EOS or a cancellation epoch. Text state and audio
  conditioning survive cancellation. Closing releases all owners downstream first.

## Scope

This is native frame-driven input, with no VAD or turn commit. The model's own
listening/speaking behavior controls generated audio. Automatic server interrupt
notifications based on model markers are not implemented; client cancellation
uses the shared runtime's epoch semantics. Playback does not rewind model history.

The initial version uses the checkpoint system prompt and Aria voice. Custom
instructions are rejected. Tool execution, session resume, cross-unit overlap,
concurrent sessions, CUDA graphs, and latency optimization are outside this change.
Sessions are limited to 240 seconds; the AR adapters also enforce context bounds.
The underlying offline talker precision and classifier-free-guidance limitations
still apply. Do not interpret unit tests or a smoke run as a quality benchmark.
