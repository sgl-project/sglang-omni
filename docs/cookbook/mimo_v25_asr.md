# MiMo-V2.5-ASR

[MiMo-V2.5-ASR](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-ASR) is Xiaomi MiMo's
end-to-end speech recognition model. SGLang-Omni serves it through the
OpenAI-compatible `/v1/audio/transcriptions` endpoint.

Architecture (input path only):

```text
waveform (24 kHz)
  -> MiMo-Audio-Tokenizer encoder / 8-layer RVQ
  -> input-local patch encoder (4 frames -> 1 LM embedding)
  -> SGLang Qwen2 decode
  -> text
```

TTS / spoken-dialogue audio output is out of scope for this integration.

## Prerequisites

Install `sglang-omni`, then download both checkpoints:

```bash
hf download XiaomiMiMo/MiMo-Audio-Tokenizer
hf download XiaomiMiMo/MiMo-V2.5-ASR
```

The tokenizer defaults to `XiaomiMiMo/MiMo-Audio-Tokenizer`. A large GPU is
required (on the order of an 80 GB card for comfortable serving).

## Server Configuration

```bash
sgl-omni serve \
  --model-path XiaomiMiMo/MiMo-V2.5-ASR \
  --port 8000
```

## Transcribe Audio

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=XiaomiMiMo/MiMo-V2.5-ASR \
  -F file=@tests/data/query_to_cars.wav \
  -F language=zh \
  -F response_format=json
```

`language` maps to the official ASR tags:

| `language` | MiMo tag |
|---|---|
| `zh` / `chinese` | `<chinese>` |
| `en` / `english` | `<english>` |
| `auto` / omitted | no tag (auto detect) |

## Known Limitations

- First launch downloads the 1.2B audio tokenizer and the 7B ASR weights.
- Default serving uses `max_running_requests=32` with CUDA graphs disabled.
- Audio output modalities are rejected.
- Prompt templates are fixed (first template of each official pool) for
  deterministic serving; official `asr_sft` randomly samples templates.
