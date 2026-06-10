# MOSS-TTS Local

[MOSS-TTS-Local-Transformer](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer) is the **Local (depth-transformer)** variant of [MOSS-TTS](moss_tts.md). OpenMOSS recommends it for research, quality-critical, and bitrate-sensitive use.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md), then download the model:

```bash
hf download OpenMOSS-Team/MOSS-TTS-Local-Transformer
```

## Server Configuration

```bash
sgl-omni serve \
  --model-path OpenMOSS-Team/MOSS-TTS-Local-Transformer \
  --config examples/configs/moss_tts_local.yaml \
  --port 8000
```

> **`--config` is required.** The Local checkpoint declares `architectures: ["MossTTSDelayModel"]` (the same string as the 8B Delay checkpoint), so serving by `--model-path` alone resolves to the Delay pipeline. The explicit `--config` selects the Local pipeline.

## Synthesize Speech

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, how are you?"}' \
  --output output.wav
```

Voice cloning, duration control, and text markup work exactly as in [MOSS-TTS](moss_tts.md) — Local reuses its preprocessing.

## Known Limitations

- **`--config` required** — serving by `--model-path` alone resolves to the Delay pipeline (see Server Configuration).
- **Eager depth decode** — the per-frame depth loop runs outside the CUDA graph.
- **Fixed sampling** — uses the upstream Local defaults; per-request `temperature` / `top_p` / `top_k` overrides aren't plumbed yet.
- **No streaming** — `stream: true` returns the whole clip in one chunk, not incremental output.
