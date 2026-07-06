# Playground

Browser playgrounds for the three models served by SGLang-Omni.

| Subdirectory | Model | UI |
|---|---|---|
| `qwen-omni/` | Qwen3-Omni — multimodal chat (text / audio / image / video). | HTML / CSS / JS |
| `s2pro/` | S2 Pro — text-to-speech with voice cloning, streaming and non-streaming. | Gradio |
| `higgs/` | Higgs Audio v3 — multilingual TTS with inline control tokens (emotion / style / sfx / prosody) and streaming. | HTML / CSS / JS |

Each `start.sh` launches the backend, waits for `/health`, then launches the
playground UI in the foreground. `Ctrl-C` stops both. The raw `sgl-omni serve`
command is shown alongside in case you want to run the backend yourself (for
example, on a separate host) and point the UI at it.

## Qwen3-Omni

```bash
# One command: backend + UI
./playground/qwen-omni/start.sh \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct
```

```bash
# Or run the backend yourself
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --port 8000

# …then the UI separately (point it at the backend via env var)
SGLANG_OMNI_API_BASE=http://localhost:8000 \
  python playground/qwen-omni/app.py --port 7860
```

Open <http://localhost:7860> — backend at <http://localhost:8000>.

Override ports with `--port` (backend) and `--playground-port` (UI).

## S2 Pro TTS

```bash
# One command: backend + UI
./playground/s2pro/start.sh \
  --model-path fishaudio/s2-pro
```

```bash
# Or run the backend yourself
sgl-omni serve \
  --model-path fishaudio/s2-pro \
  --config examples/configs/s2pro_tts.yaml \
  --port 8000

# …then the UI separately
python -m playground.s2pro.app --api-base http://localhost:8000 --port 7899
```

Open <http://localhost:7899>. Two tabs: `Non-Streaming` (final WAV after
generation) and `Streaming` (incremental playback from raw
`/v1/audio/speech` PCM chunks).

Override ports with `--port` (backend) and `--gradio-port` (UI).

## Higgs Audio v3 TTS

```bash
# One command: backend + UI
./playground/higgs/start.sh \
  --model-path bosonai/higgs-tts-3-4b
```

```bash
# Or run the backend yourself
sgl-omni serve \
  --model-path bosonai/higgs-tts-3-4b \
  --port 8000

# …then the UI separately
python -m playground.higgs.app --api-base http://localhost:8000 --port 7860
```

Open <http://localhost:7860>. Features:

- Non-streaming and streaming tabs (incremental playback from raw PCM chunks).
- Reference audio from **microphone recording**, file upload, or URL for voice cloning.
- Inline control-token picker (clickable chips for emotion / style / sfx /
  prosody) that inserts `<|category:name|>` tokens at the cursor.

Override ports with `--port` (backend) and `--playground-port` (UI).

## SSH tunnel (remote servers / Docker)

From your local machine:

```bash
ssh -L 8000:localhost:8000 -L 7860:localhost:7860 user@host
```
