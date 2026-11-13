# Qwen3-TTS

Qwen3-TTS is a multi-codebook text-to-speech model family that streams 24 kHz speech in 10 languages. The `12Hz` checkpoint name refers to the codec frame rate. SGLang-Omni serves [Base](#voice-cloning) for voice cloning, [CustomVoice](#customvoice) for built-in speakers, and [VoiceDesign](#voicedesign) for voices described in natural language through the OpenAI-compatible `/v1/audio/speech` endpoint.

<span id="quick-start"></span>

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md). The following example uses Linux with an NVIDIA GPU.

Install the Qwen3-TTS dependencies in the server environment:

```bash
apt-get update && apt-get install -y sox
uv pip install --no-deps sox einops
uv pip install --no-deps qwen-tts==0.1.1
```

See [dependency troubleshooting](qwen3_tts/reference.md#dependency-troubleshooting) if imports fail.

<span id="start-the-server"></span>

## Server Configuration

This example uses 1.7B CustomVoice:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --config examples/configs/qwen3_tts_1_7b_customvoice.yaml \
  --port 8000
```

The first startup will take several minutes for compilation and warmup. Then, in another terminal, run the following command and wait until `/health` returns HTTP 200:

```bash
curl -i http://localhost:8000/health
```

<span id="generate-speech"></span>

## Synthesizing Speech

```bash
curl --fail-with-body http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "input": "Hello! This is Qwen3-TTS running with SGLang-Omni.",
    "voice": "Ryan",
    "language": "English",
    "instructions": "Speak clearly and calmly."
  }' \
  --output output.wav
```

Open `output.wav` in an audio player. The response is WAV audio by default. For Python or the OpenAI SDK, use the [TTS client examples](../basic_usage/tts.md#python).

## Choose a model

| Goal | Checkpoint family | What to provide |
|---|---|---|
| Generate speech with a built-in speaker | [CustomVoice](#customvoice) | Text and a speaker name |
| Clone a voice | [Base](#voice-cloning) | Text and reference audio; include its transcript for better cloning |
| Describe a new voice | [VoiceDesign](#voicedesign) | Text and a voice description |

Use CustomVoice if you do not have a reference clip. For other hardware, such as Ascend NPU or Intel XPU, or for the native Apple Silicon runtime, see [platform deployment](qwen3_tts/platforms.md).

<span id="text-only-requests"></span>

## Voice modes

<span id="customvoice-checkpoints"></span>

### CustomVoice

CustomVoice uses `voice` to select a built-in speaker. The 1.7B model also supports speaking-style control through `instructions`, while the 0.6B model is primarily intended for speech synthesis with built-in speakers.

List the available speakers:

```bash
curl --fail-with-body http://localhost:8000/v1/audio/voices
```

The 0.6B checkpoint also supports built-in speakers, but reliable style control requires 1.7B. See [speaker and dialect behavior](qwen3_tts/reference.md#customvoice-behavior) for details.

### Voice cloning

Voice cloning requires a Base checkpoint with reference audio or a [precomputed speaker embedding](qwen3_tts/reference.md#precomputed-speaker-embedding). See [Voice cloning in TTS Model Usage](../basic_usage/tts.md#voice-cloning) for setup and request examples.

<span id="voicedesign-checkpoint"></span>

### VoiceDesign

VoiceDesign generates a voice from a description in `instructions` and requires the 1.7B VoiceDesign checkpoint. See [Voice design in TTS Model Usage](../basic_usage/tts.md#voice-design) for setup and request examples.

### Model variants

Use the full checkpoint ID in `--model-path` and the request's `model` field.

| Checkpoint | Config under `examples/configs/` |
|---|---|
| `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | `qwen3_tts_0_6b_customvoice.yaml` |
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | `qwen3_tts_1_7b_customvoice.yaml` |
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | `qwen3_tts_0_6b.yaml` |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | `qwen3_tts_1_7b.yaml` |
| `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | `qwen3_tts_1_7b_voicedesign.yaml` |

## Streaming

With the CustomVoice server running, send a streaming request:

```bash
curl --fail-with-body -N http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "input": "Hello! This is Qwen3-TTS running with SGLang-Omni.",
    "voice": "Ryan",
    "language": "English",
    "stream": true,
    "response_format": "wav"
  }' \
  --output streaming.wav
```

This saves the streamed audio to `streaming.wav`.

All three voice modes support streaming. For detailed examples, see [TTS streaming examples](../basic_usage/tts.md#raw-wav-and-pcm), [SSE](../basic_usage/tts.md#sse-event-framing), and [WebSocket](../basic_usage/tts.md#websocket-speech-streaming).

The default chunk schedule favors early audio. If playback stalls under load, see [first-audio chunk tuning](qwen3_tts/tuning.md#first-audio-chunk-ramp).

<span id="generation-parameters"></span>
<span id="language-hint"></span>

## Common options

With the 1.7B CustomVoice server running, customize the language, speaking style, generation limit, and sampling seed:

```bash
curl --fail-with-body http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "input": "你好，欢迎使用 SGLang-Omni。",
    "voice": "Vivian",
    "language": "Chinese",
    "instructions": "用平静、自然的语气说话。",
    "max_new_tokens": 512,
    "seed": 42
  }' \
  --output chinese-speech.wav
```

Reaching `max_new_tokens` without codec EOS fails the request; allow enough tokens for the full sentence. A fixed `seed` does not guarantee byte-identical audio; see [deterministic inference](qwen3_tts/tuning.md#deterministic-inference).

See [Qwen3-TTS parameters](qwen3_tts/reference.md#generation-parameters) for defaults and model-specific controls, and the [shared TTS API](../basic_usage/tts.md#request-parameters) for common request fields.

### Known limitations

- **Base requires a reference voice.** Provide reference audio or a speaker embedding. Include the audio transcript for better cloning quality. For text-only synthesis, use CustomVoice or VoiceDesign.
- **Speech may repeat or stop before the sentence ends.** Repetition can occur on some inputs. Set `max_new_tokens` to limit generation, but allow enough tokens for the full sentence. If generation reaches the limit without codec EOS, the request fails; discard any partial audio already received. See [finish reasons](qwen3_tts/reference.md#finish-reasons) and [benchmark results](qwen3_tts/benchmarks.md#benchmark-results).

<span id="next-steps"></span>

## Advanced Guides

<span id="optional-cutedsl-decode"></span>
<span id="deterministic-inference"></span>
<span id="overload-admission-policy"></span>
<span id="server-readiness"></span>
<span id="prefill-cuda-graphs"></span>
<span id="prefill-admission-coalescing"></span>
<span id="process-topology"></span>
<span id="codec-decoding-defaults"></span>
<span id="first-audio-chunk-ramp"></span>
<span id="precomputed-speaker-embedding"></span>
<span id="ascend-npu-baseline"></span>
<span id="apple-silicon-native-mlx"></span>
<span id="benchmarking"></span>
<span id="benchmark-results"></span>
<span id="runaway-regression"></span>
<span id="b-base"></span>
<span id="b-customvoice"></span>

- [Tuning](qwen3_tts/tuning.md): concurrency, streaming continuity, CUDA graphs, deterministic inference, and optional CuTeDSL kernels.

- [Reference](qwen3_tts/reference.md): full parameter defaults, precomputed embeddings, speaker behavior, and dependency troubleshooting.

- [Platform deployment](qwen3_tts/platforms.md): Ascend NPU and native Apple Silicon, with links to other installation guides.

- [Benchmarks](qwen3_tts/benchmarks.md): reproduce evaluations and inspect results with their hardware and sampling conditions.
