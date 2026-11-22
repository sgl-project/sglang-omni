# LFM3-Audio and LFM3-Omni

Liquid AI's [LFM3-Audio-2B](https://huggingface.co/LiquidAI/LFM3-Audio-2B) takes text and
16 kHz speech in and answers with text, 24 kHz speech (LiquidCodec, 12.5 frames per second),
or interleaved text and speech. [LFM3-Omni-2B](https://huggingface.co/LiquidAI/LFM3-Omni-2B)
adds image and video input. Both run as a `preprocessing → generation → decode` pipeline on
one GPU and are served through `/v1/audio/speech` and `/v1/chat/completions`.

The checkpoints are released under the LFM Open License v1.0.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md), then download a model:

```bash
hf download LiquidAI/LFM3-Omni-2B
```

The checkpoints ship their own encoder and codec code, which the pipeline loads with
`trust_remote_code`.

## Server Configuration

```bash
sgl-omni serve \
  --model-path LiquidAI/LFM3-Omni-2B \
  --config examples/configs/lfm3_omni.yaml \
  --port 8000
```

For LFM3-Audio, use `LiquidAI/LFM3-Audio-2B` with `examples/configs/lfm3_audio.yaml`.

## Text to Speech

Both models have two built-in voices, `female` (default) and `male`. `language` sets the
spoken accent and defaults to `American English`; `instructions` replaces the whole
system prompt.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "LiquidAI/LFM3-Omni-2B",
    "voice": "male",
    "input": "The quick brown fox jumps over the lazy dog."
  }' \
  --output output.wav
```

`temperature`, `top_k` and `repetition_penalty` control speech-frame sampling; `temperature: 0`
is greedy and `seed` makes a request reproducible. Unset fields take the checkpoint's defaults
(LFM3-Omni: `0.4`, no top-k, `1.2`; LFM3-Audio: `1.0`, `128`, `1.1`).

## Chat

Ask for speech with `"modalities": ["text", "audio"]`. The model answers with interleaved
text and speech; the response carries both the text and the waveform. `talker_temperature`,
`talker_top_k` and `talker_repetition_penalty` control the speech frames.

```python
import base64
import requests

resp = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "LiquidAI/LFM3-Omni-2B",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "modalities": ["text", "audio"],
    },
).json()
message = resp["choices"][0]["message"]
print(message["content"])
with open("answer.wav", "wb") as f:
    f.write(base64.b64decode(message["audio"]["data"]))
```

Without `audio` in `modalities` the model answers in text only, without a reasoning block.

Speech, images (LFM3-Omni) and videos (LFM3-Omni) go in the message content as
`input_audio` / `audio_url`, `image_url` and `video_url` parts:

```python
image = base64.b64encode(open("photo.png", "rb").read()).decode()
messages = [{
    "role": "user",
    "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
        {"type": "video_url", "video_url": {"url": "https://example.com/clip.mp4"}},
        {"type": "text", "text": "Describe the photo and the clip."},
    ],
}]
```

Videos are sampled at the checkpoint's frame rate (2 fps, 4 to 384 frames) and read without
their soundtrack; send the audio as its own part.

## Limitations

- No CUDA graphs, streaming audio output or voice cloning yet.
- Chunked prefill and the radix cache are off, so each prompt is prefilled in one pass.
