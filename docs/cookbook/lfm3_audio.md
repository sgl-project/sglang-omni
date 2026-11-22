# LFM3-Audio

[LFM3-Audio-2B](https://huggingface.co/LiquidAI/LFM3-Audio-2B) is Liquid AI's audio
foundation model. It takes text and 16 kHz speech in and answers with text, 24 kHz speech
(LiquidCodec, 12.5 frames per second), or interleaved text and speech. In SGLang-Omni it runs
as a `preprocessing → generation → decode` pipeline on one GPU and is served through
`/v1/audio/speech` and `/v1/chat/completions`.

The checkpoint is released under the LFM Open License v1.0.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md), then download the model:

```bash
hf download LiquidAI/LFM3-Audio-2B
```

The checkpoint ships its own audio encoder and codec code, which the pipeline loads with
`trust_remote_code`.

## Server Configuration

```bash
sgl-omni serve \
  --model-path LiquidAI/LFM3-Audio-2B \
  --config examples/configs/lfm3_audio.yaml \
  --port 8000
```

## Text to Speech

LFM3-Audio has two built-in voices, `female` (default) and `male`. `language` sets the
spoken accent and defaults to `American English`; `instructions` replaces the whole
system prompt.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "LiquidAI/LFM3-Audio-2B",
    "voice": "male",
    "input": "The quick brown fox jumps over the lazy dog."
  }' \
  --output output.wav
```

`temperature`, `top_k` and `repetition_penalty` control speech-frame sampling (defaults
`1.0`, `128` and `1.1`); `temperature: 0` is greedy. `seed` makes a request reproducible.

## Chat

Ask for speech with `"modalities": ["text", "audio"]`. The model answers with interleaved
text and speech; the response carries both the text and the waveform.

```python
import base64
import requests

resp = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "LiquidAI/LFM3-Audio-2B",
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

Speech input goes in the message content as an `input_audio` part (or `audio_url`):

```python
audio = base64.b64encode(open("question.wav", "rb").read()).decode()
messages = [{
    "role": "user",
    "content": [
        {"type": "input_audio", "input_audio": {"data": audio, "format": "wav"}},
        {"type": "text", "text": "Transcribe the audio."},
    ],
}]
```

## Limitations

- No CUDA graphs, streaming audio output or voice cloning yet.
- Chunked prefill and the radix cache are off, so each prompt is prefilled in one pass.
