# FireRedTTS3

FireRedTTS3 is a token-free continuous-latent text-to-speech family. Audio is modeled end to end as 64-dim continuous latents at 25 Hz — there is no RVQ codebook and no audio token sampler. Each autoregressive step of the Qwen3-1.7B backbone produces one latent patch (4 frames = 160 ms of audio): the backbone hidden state conditions a flow-matching DiT that solves a 10-step Euler ODE for the patch, a PatchEncoder folds the patch back into the next backbone input embedding, and a stop head decides when to end. A RedAE encoder/decoder pair converts between 24 kHz waveform and latents; the decoder runs as the streaming vocoder stage.

Because the backbone emits no token logits in the audio phase, `temperature` / `top_p` / `top_k` do not shape the audio — use the solver knobs (`n_timesteps`, `inference_cfg`, `stop_threshold`) instead. The exception is the Instruct variant's text phase (voice_design / semantic_edit), where the backbone genuinely samples text and those knobs apply.

| Component | Spec |
|---|---|
| Backbone | Qwen3 1.7B decoder (28 L, hidden=2048, GQA 16/8) |
| Acoustic head | Flow-matching DiT (11 L, hidden=1024, 16 heads, AdaLN) + PatchEncoder (8 L) |
| Latent patch | 4 frames × 64 dims (one patch = 160 ms of audio, 3840 samples) |
| Vocoder | RedAE decoder (18 L Qwen3, SWA 64) + iSTFT head |
| Sample rate | 24 kHz |
| Solver | Euler, `n_timesteps=10` with classifier-free guidance |

## Variants

One architecture, three checkpoints. The variant is detected from the checkpoint's `config.json` at launch; each variant has its own example config.

| | Base | Instruct | Speaker |
|---|---|---|---|
| Example config | `examples/configs/firered_tts3_base.yaml` | `examples/configs/firered_tts3_instruct.yaml` | `examples/configs/firered_tts3_speaker.yaml` |
| Voice condition | Reference clip: CAM++ speaker embedding + in-context prompt latents | Reference clip as in-context prompt (tts task) or edit source | Built-in speaker token; no reference audio |
| Tasks | Voice cloning only | `Base` (cloning), `VoiceDesign`, `SemanticEdit`, `AcousticEdit` | Text-only speech |
| Default CFG | 2.0 | 2.0 (tts) / 1.2 (design, edits) | 1.2 |
| `max_patches` | 400 (~64 s) | 400 | 600 (~96 s) |
| Default `flow_mode` | `compile_bf16` | `fp32` (see below) | `compile_bf16` |
| Dual-stream decode form | no | no | opt-in via `extra_params.dual_stream` |
| Pipeline stages | preprocessing → reference_encode → talker → vocoder | same as Base | preprocessing → talker → vocoder (no reference_encode) |

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md), then launch the server with the config matching your checkpoint:

```bash
sgl-omni serve \
  --model-path /path/to/FireRedTTS3-Base \
  --config examples/configs/firered_tts3_base.yaml \
  --allowed-local-media-path docs/_static/audio \
  --port 8000
```

For Instruct or Speaker, swap the checkpoint and the config for `firered_tts3_instruct.yaml` / `firered_tts3_speaker.yaml`. The example configs pin `model_path`, so `--model-path` is only needed to serve a different checkpoint location.

Every GPU stage warms its cold-start paths at startup (encoders, DiT compile, CUDA graph capture, vocoder decode shapes), so the first served requests are already warm. The DiT compile for a concurrency bucket that warmup did not cover runs lazily on first use — a one-time ~45-85 s pause per process (~45 s with a warm torch-inductor FX cache, which persists across restarts).

The examples below read local clips; allow their directories with `--allowed-local-media-path`, or fetch reference audio over HTTP with `--allowed-media-domain`.

## Synthesizing Speech

### Voice Cloning (Base and Instruct)

Base takes exactly one reference clip plus its transcript; the transcript is prefixed to the input so the model can align prompt audio with prompt text. A wrong transcript hurts cloning quality.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FireRedTTS3-Base",
    "input": "欢迎使用点点语音合成系统，这是一段零样本音色克隆测试。",
    "language": "Chinese",
    "references": [{
      "audio_path": "docs/_static/audio/ref_voice.wav",
      "text": "It was the night before my birthday. Hooray! It’s almost here! It may not be a holiday, but it’s the best day of the year."
    }],
    "seed": 42
  }' \
  --output output.wav
```

`ref_audio` / `ref_text` are accepted as a shorthand for `references[0].audio_path` / `references[0].text`. `language` defaults to `Chinese`. The API accepts `Chinese`, `English`, `Japanese`, `Korean`, `German`, `French`, `Russian`, `Portuguese`, `Spanish`, and `Italian`. `Auto` is not supported.

To allow other checkpoint language or dialect tags (such as `ZH_Sichuan`), extend the Python pipeline config class `sglang_omni.models.firered_tts3.config.FireRedTTS3PipelineConfig` and set its `additional_speech_languages: ClassVar[frozenset[str]]` to those tags. This attribute is inherited from `PipelineConfig`; it is a class attribute, not a YAML field or a stage factory setting. The stock FireRedTTS3 config does not add any languages.

### Streaming

Streaming emits raw 24 kHz 16-bit mono PCM as latent patches decode, cutting time-to-first-audio to ~80 ms warm at concurrency 1. Set `"stream": true` and `"response_format": "pcm"`:

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FireRedTTS3-Base",
    "input": "欢迎使用点点语音合成系统，这是一段零样本音色克隆测试。",
    "references": [{
      "audio_path": "docs/_static/audio/ref_voice.wav",
      "text": "It was the night before my birthday. Hooray! It’s almost here! It may not be a holiday, but it’s the best day of the year."
    }],
    "stream": true,
    "response_format": "pcm",
    "seed": 42
  }' \
  --output output.pcm

ffmpeg -f s16le -ar 24000 -ac 1 -i output.pcm output.wav
```

The WebSocket protocol at `/v1/audio/speech/stream` (session.config → input.text → input.commit / input.done) works the same way; `ref_audio` / `ref_text` / `language` / `seed` ride in the `session.config` message. See [TTS Model Usage](../basic_usage/tts.md#websocket-speech-streaming) for the full event flow.

### Instruction Tasks (Instruct)

The Instruct checkpoint adds three tasks on top of cloning. `task_type` accepts `VoiceDesign`, `SemanticEdit`, and `AcousticEdit` (case- and separator-insensitive). Omit `task_type` or use `Base` for cloning; `TTS` and `tts` are rejected with HTTP 400.

`VoiceDesign` synthesizes a voice from a text description — no reference audio, `instructions` required:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FireRedTTS3-Instruct",
    "input": "今天天气很好，我们一起去公园散步吧。",
    "task_type": "VoiceDesign",
    "instructions": "一个年轻女性的温柔嗓音，语速稍慢，带一点俏皮。",
    "seed": 42
  }' \
  --output voice_design.wav
```

`SemanticEdit` rewrites the content of an existing clip (the reference audio is the edit source, capped at 20 s); `input` may be empty:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  --data-binary @- --output semantic_edit.wav <<'JSON'
{
  "model": "FireRedTTS3-Instruct",
  "input": "",
  "task_type": "SemanticEdit",
  "instructions": "Replace 'birthday' with 'anniversary'.",
  "ref_audio": "docs/_static/audio/ref_voice.wav",
  "seed": 42
}
JSON
```

For substitutions, use `Replace 'source' with 'target'.` with single quotes
around both words. The quoted example above succeeded in a local ASR check;
omitting quotes did not perform the intended replacement on the same audio.
The `x-gen-text` response header contains edit masks, not a literal transcript
of the generated audio. The replacement word may appear only in the audio.
The heredoc preserves the instruction's single quotes in the shell command.

`AcousticEdit` keeps the content and adjusts one acoustic attribute. The instruction comes from `instructions` verbatim, or is composed from `acoustic_attribute` (`Speed` / `Pitch` / `Volume`) and a value — the top-level `speed` field doubles as the Speed value:

Raw instructions must use the trained templates: `adjust the speed to 0.8x`,
`shift the pitch by 2 steps`, or `adjust the volume to 0.8`, for example.
Free-form acoustic descriptions are unsupported.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FireRedTTS3-Instruct",
    "input": "",
    "task_type": "AcousticEdit",
    "ref_audio": "docs/_static/audio/ref_voice.wav",
    "speed": 0.8,
    "seed": 42
  }' \
  --output acoustic_edit.wav
```

VoiceDesign and SemanticEdit run a text autoregressive phase first (the model free-writes the voice plan / edited transcript, up to 200 tokens), so their first audio arrives later than a pure cloning request of the same length. Text-phase sampling defaults to the reference values (VoiceDesign: temperature 0.7, top_p 0.8, top_k 20; everything else greedy) and explicit `temperature` / `top_p` / `top_k` request fields override them.

### Speaker Variant and Dual-Stream

The Speaker variant conditions on a built-in speaker token — requests are text-only, and passing a reference audio is rejected with HTTP 400:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FireRedTTS3-Speaker",
    "input": "欢迎使用点点语音合成系统，这是一段零样本音色克隆测试。",
    "seed": 42
  }' \
  --output speaker.wav
```

The dual-stream decode form interleaves text and latent feeds (2 text tokens per patch group) and emits each patch's audio immediately — the reference's low-latency streaming inference form. Opt in per request with `extra_params.dual_stream` (serving rejects it on non-speaker checkpoints), optionally with an `extra_params.style` prefix:

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FireRedTTS3-Speaker",
    "input": "欢迎使用点点语音合成系统，这是一段零样本音色克隆测试。",
    "extra_params": {"dual_stream": true, "style": "轻松自然地讲述"},
    "stream": true,
    "response_format": "pcm",
    "seed": 42
  }' \
  --output speaker_dual.pcm
```

`dual_stream` selects the decode form and is orthogonal to `stream` (the output mode): a non-streaming dual-stream request returns one WAV. Incremental text upload mid-generation (extending the text while audio is already streaming) is not supported.

### Request Parameters

Top-level fields (the endpoint drops unknown top-level keys silently — model knobs belong in `stage_params` / `extra_params`):

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | string | served model | Served model identifier |
| `input` | string | (required) | Text to synthesize; at most 400 characters. May be empty for SemanticEdit / AcousticEdit |
| `references` | list | variant | One item with `audio_path` + `text`. Required for Base and Instruct cloning (both fields) and the edit tasks (audio only); forbidden for the Speaker variant and VoiceDesign |
| `ref_audio` / `ref_text` | string | `null` | Shorthand for `references[0]` |
| `response_format` | string | `"wav"` | Output format; `pcm` required when `stream` is true |
| `stream` | bool | `false` | Stream raw 24 kHz PCM chunks |
| `seed` | int | `null` | Per-request flow-sampler seed; fixed seed + fixed request is bit-exact |
| `language` | string | `"Chinese"` | Language tag for Base: `Chinese`, `English`, `Japanese`, `Korean`, `German`, `French`, `Russian`, `Portuguese`, `Spanish`, `Italian`. Other checkpoint tags require the pipeline config class's `additional_speech_languages` (see above); `Auto` is unsupported |
| `task_type` | string | omitted (cloning) | Instruct tasks: `Base` (cloning) / `VoiceDesign` / `SemanticEdit` / `AcousticEdit`; `TTS` and `tts` are rejected |
| `instructions` | string | `null` | VoiceDesign description, SemanticEdit instruction, or a raw AcousticEdit template |
| `speed` | float | `1.0` | Consumed natively by AcousticEdit as the Speed value ("adjust the speed to Nx"); ignored by other tasks |
| `temperature` / `top_p` / `top_k` | | task defaults | Instruct text-phase sampling overrides only; the audio phase never samples |
| `max_new_tokens` | int | derived | Engine decode-step budget; rarely needed — the pipeline derives it from `max_patches` (dual-stream: `3 * max_patches + 20`) |

Solver and stop knobs ride under `stage_params.talker`:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_timesteps` | int | `10` | Euler ODE steps per latent patch |
| `inference_cfg` | float | Base 2.0, Speaker 1.2, Instruct per task | Classifier-free guidance strength |
| `stop_threshold` | float | `0.5` | Stop-head probability that ends generation |
| `min_patches` | int | `6` | Patches generated before the stop head may fire (~1 s floor) |
| `max_patches` | int | `400` (Speaker `600`) | Patch budget; the deployment config's value caps requests |
| `seed` | int | `null` | Same as the top-level seed |

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FireRedTTS3-Base",
    "input": "欢迎使用点点语音合成系统。",
    "ref_audio": "docs/_static/audio/ref_voice.wav",
    "ref_text": "It was the night before my birthday. Hooray! It’s almost here! It may not be a holiday, but it’s the best day of the year.",
    "seed": 42,
    "stage_params": {"talker": {"n_timesteps": 10, "inference_cfg": 2.0, "max_patches": 200}}
  }' \
  --output output.wav
```

Speaker-only `extra_params` (validated at the serving layer):

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dual_stream` | bool | `false` | Dual-stream decode form (2:1 text/latent interleave); 400 on non-speaker checkpoints |
| `style` | string | `null` | Style prefix for the dual-stream prompt; applies to the dual-stream form only |

AcousticEdit composition knobs under `stage_params.preprocessing`: `acoustic_attribute` (`Speed` / `Pitch` / `Volume`) and `acoustic_value` (float). An explicit `instructions` string overrides composition entirely.

### Flow Execution Modes

`flow_mode` is a launch-time talker factory setting (`stages.talker.factory.flow_mode` in the config), selecting the DiT execution stack inside the flow loop:

| Mode | What changes | Use |
|---|---|---|
| `fp32` | nothing — the reference stack | Bit-exact deployments; the Instruct default |
| `compile_fp32` | torch.compile over the fp32 DiT | Faster fp32 |
| `bf16` | DiT matmul body in bf16, eager | (measured slower than compile_bf16; kept for A/B) |
| `compile_bf16` | bf16 body + torch.compile | The throughput stack; Base / Speaker default |

Only the DiT matmul body lowers to bf16: the Euler state, noise, CFG blend, and time schedule stay fp32. Instruct pins `fp32` because its stop head sits near the decision threshold — every non-bit-exact mode measurably flips stop trajectories on some utterances (stopping 2-3 patches early). Base and Speaker default to `compile_bf16`, which keeps the c=1 stop trajectory identical and RTF ~35% lower than fp32.

Compile cost is per process and per batch bucket: the first request(s) that hit a new concurrency shape pay a one-time ~45-85 s torch.compile (~45 s with a warm inductor FX cache, which persists across restarts). Startup warmup pre-compiles the single-request shapes.

### Compiled Flow Caches and Speaker Controls

The Flow time and convolution caches are enabled by default for `flow_mode: compile_fp32` and `compile_bf16` when Flow CUDA graphs are used (`stages.talker.factory.flow_cuda_graph: true`). Eager `fp32` / `bf16` modes leave both caches disabled by default. Explicitly enabling either cache with an eager mode is rejected.

| Environment variable | Compiled-mode default | Eager-mode default | Behavior |
|---|---|---|---|
| `FIRERED_TTS3_PRECOMPUTE_TIME` | `1` | `0` | Caches the original compiled time-branch matrix-multiply outputs for each Euler step and batch bucket |
| `FIREREDTTS3_CONV_LAYOUT` | `conv2d_channels_last` | `conv1d` | Caches channels-last convolution weights and replaces external convolution calls inside the original compiled graph |

Disable either cache independently, or both, before starting the worker:

```bash
export FIRERED_TTS3_PRECOMPUTE_TIME=0
export FIREREDTTS3_CONV_LAYOUT=conv1d
```

The original DiT graph and its compilation options are retained. Each bucket compares complete Euler Flow outputs before capture and after graph replay on a fixed zero probe and two seeded nonzero probes. The observed compiled backend and the time-only cache must each remain finite and byte-exact against the original compiled Flow. Only the final candidate containing convolution replacements may use numerical tolerances:

| Environment variable | Default | Meaning |
|---|---|---|
| `FIRERED_TTS3_FLOW_ATOL` | `0.001` | Absolute error allowance for convolution-optimized Flow output elements |
| `FIRERED_TTS3_FLOW_RTOL` | `0.01` | Relative error allowance against the original compiled Flow output |

Every output element must satisfy `abs(candidate - reference) <= atol + rtol * abs(reference)`. Both tolerances must be finite and nonnegative; shape, dtype, device, finite values, cache signatures, and operation coverage remain mandatory. Logs include maximum absolute error, RMSE, worst per-request NRMSE, and the number of elements outside the tolerance. Batch-averaged error is not an acceptance criterion. Set both variables to `0` to restore byte-for-byte convolution checks, including signed zero.

These defaults are conservative numerical screening budgets, not calibrated guarantees of perceptual quality. Capture probes run once per bucket, rather than comparing every served request against a second model. A schema mismatch or failed check retries the whole Flow with the original compiled path. Failure of that original path invokes the existing fallback behavior. Compare the same batch shape and dtype when qualifying a change; equality across different batching schedules or precisions is not required by this gate.

Before accepting a non-exact optimization, use fixed-input Flow cases and free-running synthesis with the same seeds and execution shapes. Require identical generated patch counts, PCM sample counts, and normal termination reasons; do not force patch counts or truncate audio to match. Assess content and audio quality separately on held-out utterances, including omissions, repetitions, cut-off tails, noise, and voice changes. Waveform RMSE or correlation alone cannot establish quality. Streaming PCM comparisons also require the same vocoder batch and chunk shapes; use `stages.vocoder.factory.stream_batching: false` for isolated precision qualification, and retain normal batching for performance measurements. The time-cache collection/copy checks remain byte-exact regardless of convolution tolerance.

All Flow buckets first try ordinary PyTorch/cuDNN channels-last convolution with prepacked weights, including the BF16 B4 bucket and padded B3 requests. This path does not require cuDNN frontend or a specific GPU architecture. It must pass the complete Flow precision checks and actual CUDA graph replay before replacing the validated time-only graph.

The BF16 B4 bucket doubles to eight rows for CFG. If the ordinary path fails validation on SM120, its convolution with input `[8,1024,1,12]` and weight `[1024,1024,1,3]` can retry a controlled cuDNN execution plan. This plan retains the original channels-first engine and reduction settings while reusing channels-last weights, and requires cuDNN 9.20 and frontend 1.28.0. Install the optional frontend with `pip install 'sglang-omni[firered-tts3]'` (or `pip install nvidia-cudnn-frontend==1.28.0` in an existing environment). The retry must pass the same complete Flow and replay checks against the original compiled reference. Missing dependencies, unsupported versions, deterministic execution requests, or retry failures retain the already validated time-only graph. Other architectures continue to use the ordinary convolution path when it passes validation.

On SM120, `compile_bf16` also enables DiT GEMM specialization when the validated convolution path is active. B1/B2/B4 use a specialized first FFN projection; B8/B16 use packed QKV projections. Packed QKV weights are allocated only when preparing the first B8/B16 graph, so deployments limited to small buckets do not allocate them. Packed QKV initialization failure leaves the small-batch FFN path available. The kernels require the supported eleven-layer BF16 DiT and BF16 reduced-precision reduction. Set `FIRERED_TTS3_FFN_GEMM=0` to disable them independently. Unsupported shapes or failed checks retain the validated time/convolution path.

Every added optimization, including the captured GEMM graph replay, is checked against the original compiled Flow output with the same total error budget. Passing individual incremental comparisons cannot accumulate a larger allowed error.

Each runner owns its plan and workspace. Calls and graph replays on the same runner must be serialized, as required by its static inputs and shared graph pool; concurrent runners must use separate instances. Graph entries retain their dispatch modes, cached tensors, and execution-plan resources. External cached tensors are registered with each replay stream so their allocations remain valid until queued execution finishes.

Load weights and finish dtype/device placement before constructing the Flow executor and its caches. The worker rejects weight-update requests after Flow execution is initialized; restart the worker to load different weights. Direct mutation of model parameters or modules is unsupported. Capture-time weight checks catch supported mismatches, but graph replay does not scan weights and these checks do not provide hot-update support. Each graph entry owns the time table built for its model, dtype/device, CFG batch shape, and time schedule; graph eviction releases the table when its last owner is gone.

The independent talker factory settings `control_token_fastpath` and `skip_min_patch_stop` both default to `true` and apply only to synchronous Speaker audio steps (prefill and decode) with `dual_stream: false`. The control-token fast path avoids a host/device round trip. The minimum-patch optimization skips the stop head and its scalar synchronization while `patch_count < min_patches`, when stopping is already disallowed; normal stop-head evaluation resumes at the minimum patch count. It does not change `min_patches` or `stop_threshold`. Other decode paths retain their normal behavior.

To disable the Speaker controls, merge the following values into the deployment config. Each can be disabled independently of the other control and the Flow caches:

```yaml
stages:
  talker:
    factory:
      control_token_fastpath: false
      skip_min_patch_stop: false
```

### Performance

Base voice cloning, one fixed Chinese request (~4.8 s audio), 1x L20, streaming path, warm service at the final commit. The vLLM column is the reference port's final tuned stack on 1x H20 (a longer ~7.8 s utterance); L20 and H20 measure within a few percent of each other on this stack's other TTS models.

| Metric | vLLM (H20) | SGLang-Omni (L20) |
|---|---:|---:|
| RTF, concurrency 1 | 0.309 | 0.230 |
| RTF, concurrency 8 | 0.213 | 0.342 |
| TTFP, concurrency 1 (warm) | 181-190 ms | 82 ms |

Concurrency-1 latency and TTFP beat the vLLM port clearly; vLLM still leads on c=8 RTF (its per-step engine overhead amortizes differently), so high-concurrency throughput remains the open frontier.

### Known Limitations

- Single GPU only (`tp_size=1`); one variant per deployment.
- End-to-end parity with the fp32 reference is statistical, not bitwise: any bf16 backbone (including the reference vLLM port) diverges chaotically after a few patches. At a fixed seed every request is deterministic and bit-exact against itself, and the c=1 (exact-bucket) paths stay bit-exact across the CUDA-graph and compile stacks; c>=2 batched flow output drifts within the known batched-GEMM envelope.
- Instruct VoiceDesign / SemanticEdit pay a text-AR phase (up to 200 tokens) before the first audio chunk.
- The Speaker variant takes no reference audio, and incremental text upload mid-generation is not supported.
- Concurrency above ~16 needs a VRAM budget review: each captured flow-graph batch row costs ~1.2 GB.
- `speed` is consumed only by AcousticEdit; it does not post-process other tasks' audio.
