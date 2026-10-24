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

## Server Configuration

```bash
sgl-omni serve \
  --model-path fishaudio/s2-pro \
  --config examples/configs/s2pro_tts.yaml \
  --port 8000
```

Reference audio encoding runs on GPU 0 in the preprocessing process by default,
using the checkpoint's FP32 codec. Tokenization and audio loading remain on CPU.
The preprocessing process has a 6 GiB PyTorch allocator limit; this is a memory
budget, not a reference-duration limit or a reservation of all GPU allocations.
Long references can exceed it and fail encoding.

Set `--preprocessing.total_reserve_bytes 8GiB` to increase the budget, while
leaving enough GPU memory for the autoregressive engine and vocoder. To encode
references on CPU instead, clear both the GPU placement and its budget:

```bash
sgl-omni serve \
  --model-path fishaudio/s2-pro \
  --config examples/configs/s2pro_tts.yaml \
  --preprocessing.gpu null \
  --preprocessing.total_reserve_bytes null \
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

See the [non-streaming Fish Speech voice cloning request](../basic_usage/tts.md#fish-speech-voice-cloning).

Streaming:

See the [streaming Fish Speech voice cloning request](../basic_usage/tts.md#fish-speech-voice-cloning).

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
