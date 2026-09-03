# Qwen3-TTS

[Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) is a
discrete multi-codebook text-to-speech family with voice cloning, 10-language
generation, and 24 kHz audio output.

## Overview

| Item | Value |
|---|---|
| Task | TTS |
| Checkpoint(s) | `Qwen/Qwen3-TTS-12Hz-{0.6B,1.7B}-Base`, plus CustomVoice and VoiceDesign variants |
| Endpoint(s) | `/v1/audio/speech` |
| Pipeline | preprocessing → TTS engine → vocoder |
| Input / output | Text and optional reference audio → 24 kHz audio |
| Streaming | HTTP PCM or WebSocket audio output; Base checkpoints only |
| Validated hardware | H100 |

`12Hz` is the codec frame rate, not the playback sample rate.

## Prerequisites

Install SGLang-Omni by following [Installation](../get_started/installation.md).
Qwen3-TTS uses the upstream `qwen-tts` package and the system `sox` binary:

```bash
apt-get update && apt-get install -y sox
uv pip install --no-deps sox einops
uv pip install --no-deps qwen-tts==0.1.1
```

Keep `--no-deps` on both commands. Resolving `qwen-tts` would replace the
project's Transformers 5.12 / SGLang 0.5.18 stack with Transformers 4.57.3;
resolving `sox` can upgrade NumPy beyond the `numba==0.65.1` ceiling. Do not add
`onnxruntime`, which is already a project dependency and can trigger the same
NumPy conflict.

SGLang-Omni applies the required Transformers compatibility shim from
`sglang_omni/models/qwen3_tts/compat.py`. If an upstream API change produces a
`TypeError`, report it instead of installing `qwen-tts`'s Transformers pin.

## Deploy

Serve the 1.7B Base checkpoint with its checked-in default configuration:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --config examples/configs/qwen3_tts_1_7b.yaml \
  --port 8000
```

First startup can take several minutes while the TTS engine captures CUDA
Graphs.

## Send a request

Base checkpoints clone a voice from `references[0]`. Include the reference
transcript to use in-context-learning mode, which gives better speaker
similarity than speaker-embedding-only mode.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "voice": "default",
    "input": "SGLang-Omni is a great project!",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }]
  }' \
  --output output.wav
```

`ref_audio` and `ref_text` are shorthand for the first reference object's
`audio_path` and `text` fields.

## Capabilities

### Checkpoint modes

| Mode | Conditioning | Streaming |
|---|---|---|
| Base | Reference audio; transcript recommended | Yes |
| CustomVoice | Checkpoint speaker selected by `voice` | No |
| VoiceDesign | Text plus non-empty `instructions` | No |

### Language hints

`language` defaults to `auto`. You can explicitly select Chinese, English,
Japanese, Korean, German, French, Russian, Portuguese, Spanish, or Italian.
Use an explicit hint for short or code-switched input when automatic detection
is unreliable.

### Streaming

Base checkpoints support HTTP PCM and the stateful speech WebSocket;
CustomVoice and VoiceDesign remain non-streaming. See
[Streaming](../user_guide/advanced_features/streaming.md) for the shared
transport and framing contracts.

When `initial_codec_chunk_frames` is omitted, Base checkpoints use 8 frames for
the first vocoder chunk. A smaller value lowers time to first audio but can
increase playback underruns. An explicit `0` uses the steady-state stride from
the first chunk. Utterances shorter than the initial threshold arrive in the
final flush.

### Deterministic inference

Both Base sizes expose opt-in deterministic inference. It is disabled by
default because it serializes preprocessing and vocoder work. See
[Deterministic inference](../user_guide/advanced_features/deterministic_inference.md)
for the enablement and evidence contract.

## Configuration

The 0.6B Base checkpoint uses the same pipeline and request format through
`examples/configs/qwen3_tts_0_6b.yaml`. CustomVoice and VoiceDesign use their
own checked-in configs. See [TTS model usage](../basic_usage/tts.md) for those
launch commands and their text-only request fields.

### First-audio chunk ramp

For latency-sensitive deployments the whole early chunk schedule can be
configured server-side with `stream_chunk_ramp` on the vocoder stage: entry
`i` sizes streaming decode chunk `i + 1` in codec frames, and past the ramp
the steady stride takes over, so `[2, 4, 8]` yields a
`2 -> 4 -> 8 -> 8 -> ...` schedule. Set it through a pipeline config file:

```yaml
config_cls: Qwen3TTSPipelineConfig
model_path: Qwen/Qwen3-TTS-12Hz-0.6B-Base
stages:
  vocoder:
    factory:
      stream_chunk_ramp: [2, 4, 8]
```

```bash
python -m sglang_omni.cli serve --config qwen3_tts_ramp.yaml
```

Smaller early chunks lower time-to-first-audio but start playback with less
buffered audio, so the continuity cost grows with concurrency: keep
`[2, 4, 8]` to low concurrency, prefer `[4, 8]` up to moderate concurrency,
and keep the default schedule for saturated serving. The ramp is mutually
exclusive with the legacy `initial_chunk_frames` /
`stream_initial_followup_stride` options, its first entry must not exceed the
steady stride, and a per-request `initial_codec_chunk_frames` still overrides
only the first chunk.

## Generation Parameters

Non-streaming responses set `X-Finish-Reason` to `stop` after codec EOS or
`length` at `max_new_tokens`. A `length` response is decodable but may contain
an incomplete utterance.

For the complete shared request and response contract, see the
[Speech API](../user_guide/serving/speech_api.md).

## Limitations

- Base checkpoints need a reference clip for natural output; without one,
  speech is typically robotic.
- Omitting the reference transcript uses speaker-embedding-only mode and
  usually reduces cloning quality.
- `language: auto` can misdetect short or code-switched inputs.
- The 0.6B Base checkpoint has shown rare repetition loops up to
  `max_new_tokens`. Lower that limit or raise `repetition_penalty` when this
  occurs; the 1.7B checkpoint is less prone.

## Benchmark

Run the Seed-TTS benchmark against the deployed server:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --generate-only \
  --use-existing-server \
  --stream \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --port 8000
```

Follow the [benchmark methodology](../benchmarks/methodology.md) when
publishing results.

## Related documentation

- [TTS serving and request fields](../basic_usage/tts.md)
- [Speech API](../user_guide/serving/speech_api.md)
- [Streaming](../user_guide/advanced_features/streaming.md)
- [Admission control](../user_guide/advanced_features/admission_control.md)
- [Deterministic inference](../user_guide/advanced_features/deterministic_inference.md)
- [TTS process topology](../basic_usage/tts_process_topology.md)
- [MPS/DP and Qwen3-TTS weight-sharing status](../basic_usage/mps_dp.md)
- [Supported models](../supported_models.md)
- [TTS model integration](../developer_reference/tts_model_integration.md)
