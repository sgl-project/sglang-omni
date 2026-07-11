# Fish Audio S2-Pro

[Fish Audio S2-Pro](https://huggingface.co/fishaudio/s2-pro) is a text-to-speech model served through `/v1/audio/speech`. It supports plain TTS, voice cloning with a reference clip, and streaming audio chunks.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md), then download the model:

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

## Synthesize Speech

Plain TTS:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, how are you?"}' \
  --output output.wav
```

Voice cloning:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
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
| `input` | string | required | Text to synthesize |
| `voice` | string | `default` | Voice identifier for non-reference requests |
| `response_format` | string | `wav` | Output audio format |
| `speed` | float | `1.0` | Playback speed multiplier |
| `stream` | bool | `false` | Stream raw PCM audio chunks |
| `references` | list | `null` | Reference clip for voice cloning; each item has `audio_path` and `text` |
| `ref_audio` / `ref_text` | string | `null` | Shorthand for `references[0].audio_path` and `references[0].text` |
| `max_new_tokens` | int | `2048` | Maximum generated semantic tokens |
| `temperature` | float | `0.8` | Sampling temperature |
| `top_p` | float | `0.8` | Top-p sampling |
| `top_k` | int | `30` | Top-k sampling; must be `-1` or between `1` and `30` |
| `repetition_penalty` | float | `1.1` | Repetition penalty |

## Known Limitations

- `top_k` is constrained to `-1` or `1..30`; keep requests inside this range because invalid values currently fail the S2-Pro pipeline instead of returning a clean parameter error.
- Reference quality strongly affects cloned voice quality.
- Use streaming for interactive playback; CLI inspection of raw audio responses is awkward.
