# Fish Audio S2-Pro

[Fish Audio S2-Pro](https://huggingface.co/fishaudio/s2-pro) is a text-to-speech model served through `/v1/audio/speech`. It supports plain TTS, voice cloning with a reference clip, and streaming audio chunks.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md).

Fish Audio uses the Descript DAC codec, which is not included in the base
`sglang-omni` package. From the SGLang-Omni repository root, install its
model-specific dependencies:

```bash
uv pip install \
  "descript-audiotools==0.7.2" \
  "descript-audio-codec==1.0.0"
```

Then download the model:

```bash
hf download fishaudio/s2-pro
```

## Experimental Apple Silicon support

Follow the [Apple Silicon installation instructions](../get_started/installation.md#macos-apple-silicon),
then install the DAC codec dependencies listed above into `.venv-apple`.
S2-Pro provides a native MLX path and a Torch/MPS compatibility path, selected
with `SGLANG_USE_MLX`. Both use the official `fishaudio/s2-pro` checkpoint with
unquantized BF16 weights; no converted MLX artifact or `mlx-audio` runtime
package is needed. `SGLANG_USE_MLX=1` off Apple Metal fails at startup instead of
silently falling back.

```bash
source .venv-apple/bin/activate
export DYLD_LIBRARY_PATH="$(brew --prefix ffmpeg@7)/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"

# Native MLX
SGLANG_USE_MLX=1 sgl-omni serve \
  --model-path fishaudio/s2-pro \
  --config examples/configs/s2pro_tts.yaml \
  --port 8000 --allowed-local-media-path .

# Alternatively, stop the server and use Torch/MPS
SGLANG_USE_MLX=0 sgl-omni serve \
  --model-path fishaudio/s2-pro \
  --config examples/configs/s2pro_tts.yaml \
  --port 8000 --allowed-local-media-path .
```

Both paths share one Apple profile: they serve one request at a time
(`max_running_requests=1`) and additional requests queue and complete in turn,
CUDA graphs, `torch.compile`, radix caching, and chunked prefill are all
disabled, the request context is bounded to 4,096 tokens, and quantized
checkpoints are rejected. MLX runs the Slow AR and the whole Fast-AR residual
chain natively and crosses only the semantic logits to CPU for the shared Fish
sampler; Torch/MPS runs both eagerly with the `torch_native` attention backend.
Plain TTS, voice cloning, streaming PCM, and the `seed` parameter use the shared
Fish request and output adapters on either path. Cross-device numerical or
voice-quality parity with CUDA has not been established.

This is an experimental compatibility profile. It has not been qualified for
real-time serving or production throughput. See the model README for validation
and the opt-in HTTP checks.

## Server Configuration

```bash
sgl-omni serve \
  --model-path fishaudio/s2-pro \
  --config examples/configs/s2pro_tts.yaml \
  --port 8000
```

## Synthesize Speech

Plain TTS:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "fishaudio/s2-pro",
    "voice": "default",
    "input": "Hello, how are you?"
  }' \
  --output output.wav
```

Voice cloning:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "fishaudio/s2-pro",
    "voice": "default",
    "input": "Get the trust fund to the bank early.",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }]
  }' \
  --output output.wav
```

Streaming:

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "fishaudio/s2-pro",
    "voice": "default",
    "input": "Get the trust fund to the bank early.",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }],
    "stream": true,
    "response_format": "pcm"
  }' \
  --output output.pcm
```

## Request Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | string | served model | Served model identifier |
| `input` | string | required | Text to synthesize |
| `voice` | string | `default` | Voice identifier for non-reference requests |
| `response_format` | string | `wav` | Output audio format |
| `speed` | float | `1.0` | Playback speed multiplier |
| `stream` | bool | `false` | Stream raw PCM audio chunks |
| `references` | list | `null` | Reference clip for voice cloning. Each item has `audio_path` and `text` |
| `ref_audio` / `ref_text` | string | `null` | Shorthand for `references[0].audio_path` and `references[0].text` |
| `max_new_tokens` | int | `2048` | Maximum generated semantic tokens |
| `temperature` | float | `0.8` | Sampling temperature |
| `top_p` | float | `0.8` | Top-p sampling |
| `top_k` | int | `30` | Top-k sampling. It must be `-1` or between `1` and `30` |
| `repetition_penalty` | float | `1.1` | Repetition penalty |

## Known Limitations

- `top_k` is constrained to `-1` or `1..30`; keep requests inside this range because invalid values currently fail the S2-Pro pipeline instead of returning a clean parameter error.
- Reference quality strongly affects cloned voice quality.
- Use streaming for interactive playback; CLI inspection of raw audio responses is awkward.
