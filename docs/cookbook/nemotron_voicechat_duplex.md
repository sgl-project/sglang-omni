# Native VoiceChat sessions (local integration prototype)

This opt-in pipeline connects the existing VoiceChat model implementation to
pipeline sessions (#2035), the AR streaming-session bridge (#2069), and the
shared realtime endpoint (#2070). This local integration includes #2035 and #2069
from main and #2070 at 0d09ba0b. The default
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
opening the port (`--no-warmup` skips the silent session). File-input runs also
warm up first. The warmup session allows 120 seconds per unit for cold kernel
compilation; live sessions retain the standard operation timeout.

1. Click **开始对话** and allow microphone access. Headphones help avoid acoustic feedback.
2. Speak naturally. The browser sends continuous 80 ms PCM frames and plays
   streamed output, with assistant text shown alongside it.
3. **静音麦克风** sends silence while the model continues responding.
   **结束** releases the microphone and session.

Microphone access requires localhost or HTTPS. To use a Kubernetes worker,
forward its example port, then open the same localhost URL:

```bash
kubectl port-forward --context YOUR_CONTEXT -n default pod/YOUR_GPU_POD 8097:8097 --address 127.0.0.1
```

The minimal page has start, stop, and microphone mute controls, connection
status, and assistant text. It stops if input backlog exceeds 9 seconds instead
of silently dropping input. Sessions last at most four minutes;
one browser session is supported at a time. This is a prototype: sustained GPU
processing can fall behind real time. Automatic interruption depends on model
behavior. Explicit response cancellation is not supported by the current protocol.

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
The current protocol does not accept `response.cancel` or playback acknowledgements.

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
- A response spans units until EOS. State is keyed by session ID and open index;
  closing releases all owners downstream first.

## Scope

This is native frame-driven input, with no VAD or turn commit. The model's own
listening/speaking behavior controls generated audio. Automatic server interrupt
notifications based on model markers are not implemented. Playback does not rewind model history.

The initial version uses the checkpoint system prompt and Aria voice. Custom
instructions are rejected. Tool execution, session resume, cross-unit overlap,
concurrent sessions, and backbone CUDA graphs are outside this change. Fixed-shape
speech sampling, the full codec window, and perception after its causal caches
reach their fixed bounds use CUDA graph replay; dynamic backbone KV remains
managed by the native session scheduler. Perception capture restores its warmup
state before replay, so capture does not consume extra input frames.
The browser limits sessions to 240 seconds; the AR adapters enforce context bounds
for every client. The shared runtime has no wall-clock session timeout.
The underlying offline talker precision and classifier-free-guidance limitations
still apply. Do not interpret unit tests or a smoke run as a quality benchmark.

## Reading the integration

Follow one 80 ms input unit through these files:

1. `duplex_config.py`: the stage order and placement. Each stage has its own worker;
   the example places all four on one GPU.
2. `duplex_stages.py`: construct the models, runners, adapters, and schedulers.
   The existing TTS builder creates the runner, then adapters, then the scheduler.
   The offline request/result callbacks remain part of that builder contract;
   session units use the session adapter instead.
3. `duplex.py`, `PerceptionHooks.append`: convert 1280 PCM16 samples into acoustic
   features, retaining causal encoder history across units.
4. `duplex_ar.py`, `ThinkerAdapter.build` and `result`: fuse acoustic features with
   the previous text/function tokens, run one position, and publish new tokens.
   The first unit also supplies the checkpoint prompt.
5. `TalkerAdapter.build` and `result`: fuse the new text token with previous audio
   codes, then produce the next frame of codes.
6. `duplex.py`, `CodecHooks.append`: decode a bounded code window, emit only fresh
   samples, and release the held-back tail when input ends.
7. `realtime.py`: convert terminal audio/text into events for the shared WebSocket
   runtime. The browser records and plays audio; it does not schedule model stages.

### The continuation contract

A unit performs one forward and samples one token. That sampled token has not yet
been forwarded when the unit finishes. The next unit therefore appends no new
input token IDs: the shared streaming session supplies that pending position,
while the model adapter supplies its new fusion embedding. `attach_fusion_rows`
selects only the suffix not already covered by KV. Historical fusion rows remain
available if a prefix must be replayed.

The shared scheduler owns requests, token history, and KV lifetime. The adapters
own model-specific fusion history, previous model outputs, and detokenized text.
Perception and codec hooks own their causal state. Completing a unit keeps these
session states; closing the session releases them.

For review, first check continuation alignment, EOS tail drain, and session
cleanup. Then inspect CUDA graph capture and replay as an optimization of the same
computation. Existing tests cover uncached suffix selection, continuation fusion,
codec output/flush, graph replay, and WebSocket cleanup.
