# Native VoiceChat sessions (local integration prototype)

This opt-in pipeline connects the existing VoiceChat model implementation to
pipeline sessions (#2035), the AR streaming-session bridge (#2069), and the
shared realtime endpoint (#2070). It needs all three dependencies. The default
offline VoiceChat configuration is unchanged.

## Run

Use the checkpoint and tokenizer downloads in [the offline cookbook](nemotron_voicechat.md).
The initial configuration supports one session on one GPU, with four separate
stage processes. On B200, both AR stages use `triton` attention with page size 1. This avoids
64-position padding for the thinker’s single-position continuations; Talker also
requires this because the default TRTLLM context kernel does not support its
head dimension of 72.

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

### Browser conversation UI

Open **http://localhost:8097** after the server prints `VoiceChat UI`.
The example serves its HTML, CSS and JavaScript directly; no frontend build or
npm installation is required. Startup captures the fixed-size codec graph and runs two silent frames to warm
first-frame and continuation kernels, including the speech sampler graph, before
opening the port (`--no-warmup` skips the silent session).

1. Click **开始对话** and allow microphone access. Headphones help avoid acoustic feedback.
2. Speak naturally. The browser sends continuous 80 ms PCM frames and plays
   streamed output, with assistant text shown alongside it.
3. **静音麦克风** sends silence while the model continues responding.
   **打断播放** clears scheduled output and cancels the current response while
   preserving model history. **结束对话** releases the microphone and session.

Microphone access requires localhost or HTTPS. To use a Kubernetes worker,
forward its example port, then open the same localhost URL:

```bash
kubectl port-forward --context YOUR_CONTEXT -n default pod/YOUR_GPU_POD 8097:8097 --address 127.0.0.1
```

The page shows input processing backlog, queued playback duration, and time to
first audio packet (which can contain silence). It stops if input backlog exceeds
9 seconds instead of silently dropping input. Sessions last at most four minutes;
one browser session is supported at a time. This is a prototype: sustained GPU
processing can fall behind real time. Automatic interruption depends on model
behavior; the explicit interrupt button is available to test cancellation.

Playback uses a 480 ms startup/rebuffer reserve, then schedules packets
contiguously. Where supported, the output AudioContext runs at 22050 Hz so
resampling happens on the continuous mix, rather than independently per packet.
If the browser uses another device rate, a stateful 32-tap windowed-sinc converter
retains filter history across packets before scheduling buffers at that rate.
This prevents periodic clicks from restarting browser resampling at packet edges.

Browser audio tests: `node --test examples/voicechat_ui/audio.test.mjs`.
Static route tests: `pytest tests/unit_test/nemotron_voicechat/test_duplex_ui.py`.

### WebSocket clients

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
concurrent sessions, and backbone CUDA graphs are outside this change. Fixed-shape
speech sampling, the full codec window, and perception after its causal caches
reach their fixed bounds use CUDA graph replay; dynamic backbone KV remains
managed by the native session scheduler. Perception capture restores its warmup
state before replay, so capture does not consume extra input frames.
Sessions are limited to 240 seconds; the AR adapters also enforce context bounds.
The underlying offline talker precision and classifier-free-guidance limitations
still apply. Do not interpret unit tests or a smoke run as a quality benchmark.
