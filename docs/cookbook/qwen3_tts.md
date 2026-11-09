# Qwen3-TTS

Generate 24 kHz speech in 10 languages through the OpenAI-compatible `/v1/audio/speech` endpoint. Choose a built-in speaker, clone a reference voice, or describe a voice in words.

<span id="server-configuration"></span>

## Quick start

### Prerequisites

Follow [Installation](../get_started/installation.md) to install SGLang-Omni on Linux with an NVIDIA GPU. Run the commands below from the repository root so the example config paths resolve.

Install the Qwen3-TTS dependencies in the server environment:

```bash
apt-get update && apt-get install -y sox
uv pip install --no-deps sox einops
uv pip install --no-deps qwen-tts==0.1.1
```

Keep `--no-deps` on both lines to preserve SGLang-Omni's Transformers and NumPy versions. Install both the system `sox` binary and the Python package. See [dependency troubleshooting](qwen3_tts/reference.md#dependency-troubleshooting) if imports fail.

### Start the server

This example uses 1.7B CustomVoice, which supports built-in speakers and style instructions:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --config examples/configs/qwen3_tts_1_7b_customvoice.yaml \
  --port 8000
```

First startup can take several minutes for compilation and warmup. In another terminal, check readiness; continue when `/health` returns HTTP 200:

```bash
curl -i http://localhost:8000/health
```

### Generate speech

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

Open `output.wav` in an audio player. The response is WAV audio by default. For Python or the OpenAI SDK, use the [TTS client examples](../basic_usage/tts.md#python); pass Qwen-specific fields such as `language` and `references` through the SDK's `extra_body`.

## Choose a model

| Goal | Checkpoint family | What to provide |
|---|---|---|
| Generate speech with a built-in speaker | [CustomVoice](#customvoice-checkpoints) | Text and a speaker name |
| Clone a voice | [Base](#voice-cloning) | Text and reference audio; include its transcript for better cloning |
| Describe a new voice | [VoiceDesign](#voicedesign-checkpoint) | Text and a voice description |

Use CustomVoice, as in the quick start above, if you do not have a reference clip. Each family needs its own checkpoint; changing request fields does not switch the loaded model. For Ascend NPU, Intel XPU, or the native Apple Silicon runtime, see [platform deployment](qwen3_tts/platforms.md).

<span id="synthesizing-speech"></span>
<span id="text-only-requests"></span>

## Voice modes

### CustomVoice checkpoints

The quick start uses CustomVoice. Set `voice` to a built-in speaker; omit reference-audio fields. An omitted or `default` voice selects `Vivian`.

List the speakers available on your running server:

```bash
curl --fail-with-body http://localhost:8000/v1/audio/voices
```

The 0.6B checkpoint also supports built-in speakers, but reliable style control requires 1.7B. When switching to 0.6B, use the matching model/config pair below and omit `instructions`. See [speaker and dialect behavior](qwen3_tts/reference.md#customvoice-behavior) for details.

### Voice cloning

Base requires reference audio or a [precomputed speaker embedding](qwen3_tts/reference.md#precomputed-speaker-embedding). Include the reference transcript for better voice similarity; without it, the model uses speaker-embedding-only mode.

Stop the previous server, then launch Base:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --config examples/configs/qwen3_tts_0_6b.yaml \
  --port 8000
```

After `/health` returns HTTP 200, send a reference clip and its transcript. `audio_path` accepts an HTTP URL or a path accessible to the server:

```bash
curl --fail-with-body http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "input": "Get the trust fund to the bank early.",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }]
  }' \
  --output cloned-voice.wav
```

### VoiceDesign checkpoint

VoiceDesign needs a non-empty voice description in `instructions`. Omit `voice` and reference-audio fields.

Stop the previous server, then launch VoiceDesign:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --config examples/configs/qwen3_tts_1_7b_voicedesign.yaml \
  --port 8000
```

After `/health` returns HTTP 200, describe the voice and provide the text to speak:

```bash
curl --fail-with-body http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "input": "Hello from Qwen VoiceDesign.",
    "task_type": "VoiceDesign",
    "instructions": "A warm, natural young adult voice."
  }' \
  --output designed-voice.wav
```

### Model variants

Use the full checkpoint ID in `--model-path` and the request's `model` field. Config paths are relative to the repository root.

| Checkpoint | Config under `examples/configs/` |
|---|---|
| `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | `qwen3_tts_0_6b_customvoice.yaml` |
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | `qwen3_tts_1_7b_customvoice.yaml` |
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | `qwen3_tts_0_6b.yaml` |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | `qwen3_tts_1_7b.yaml` |
| `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | `qwen3_tts_1_7b_voicedesign.yaml` |

## Streaming

To receive audio incrementally, add these fields to a request above and use `curl -N`:

```json
{
  "stream": true,
  "response_format": "wav"
}
```

All three voice modes support streaming. WAV streaming sends one header followed by audio chunks; `response_format: "pcm"` selects raw PCM. Saving the response to a file does not play it as chunks arrive. For playback clients and other transports, see [TTS streaming examples](../basic_usage/tts.md#raw-wav-and-pcm), [SSE](../basic_usage/tts.md#sse-event-framing), and [WebSocket](../basic_usage/tts.md#websocket-speech-streaming).

The default chunk schedule favors early audio. If playback stalls under load, see [first-audio chunk tuning](qwen3_tts/tuning.md#first-audio-chunk-ramp).

<span id="generation-parameters"></span>
<span id="language-hint"></span>

## Common options

| Field | When to use it |
|---|---|
| `language` | Set the target language explicitly for short or mixed-language text; defaults to `auto` |
| `instructions` | Describe speaking style for 1.7B CustomVoice; required to describe a voice for VoiceDesign |
| `max_new_tokens` | Limit generation length in codec tokens; reaching the limit can truncate speech |
| `seed` | Control sampling randomness; byte-identical audio also requires [deterministic inference](qwen3_tts/tuning.md#deterministic-inference) |

Supported languages: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, and Italian. For example, add `"language": "Chinese"` to synthesize Chinese text.

See [Qwen3-TTS parameters](qwen3_tts/reference.md#generation-parameters) for defaults and model-specific controls, and the [shared TTS API](../basic_usage/tts.md#request-parameters) for common request fields.

### Known limitations

Base cannot synthesize from text alone: provide a reference clip or speaker embedding, or choose CustomVoice/VoiceDesign. A reference transcript improves cloning quality.

Rare repetition loops can continue until the generation limit. Base applies a text-length-based default budget; CustomVoice and VoiceDesign use their checkpoint limit unless you set `max_new_tokens`. A response with `X-Finish-Reason: length` contains audio but may be incomplete. See [finish reasons](qwen3_tts/reference.md#finish-reasons) and [benchmark results](qwen3_tts/benchmarks.md#benchmark-results).

## Next steps

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
