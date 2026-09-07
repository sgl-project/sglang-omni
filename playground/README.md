# Playground

Browser playgrounds for models served by SGLang-Omni.

| Subdirectory | Model | UI |
|---|---|---|
| `qwen-omni/` | Qwen3-Omni — multimodal chat (text / audio / image / video). | HTML / CSS / JS |
| `s2pro/` | S2 Pro — text-to-speech with voice cloning, streaming and non-streaming. | Gradio |
| `higgs/` | Higgs Audio v3 — multilingual TTS with inline control tokens (emotion / style / sfx / prosody) and streaming. | HTML / CSS / JS |
| `qwen3_tts_mlx/` | Qwen3-TTS Base — voice cloning on Apple Silicon via MLX. Runs in-process, no backend. | Gradio |

Each `start.sh` launches the backend, waits for `/health`, then launches the
playground UI in the foreground. `Ctrl-C` stops both. The raw `sgl-omni serve`
command is shown alongside in case you want to run the backend yourself (for
example, on a separate host) and point the UI at it.

`qwen3_tts_mlx/` is the exception: MLX is not wired into Omni's scheduler yet, so
that playground loads the checkpoint into the UI process instead of talking to a
backend over HTTP.

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

## Qwen3-TTS voice cloning on MLX (Apple Silicon)

Clone a voice from a reference clip, entirely on-device via MLX. There is no
backend process — the UI loads the model itself.

```bash
# Voice cloning needs a *-Base checkpoint: CustomVoice and VoiceDesign ship no
# speech-tokenizer encoder, so they cannot encode reference audio.
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir q3tts-base

./playground/qwen3_tts_mlx/start.sh --model-path ./q3tts-base
```

```bash
# Or launch the UI directly
python -m playground.qwen3_tts_mlx.app --model-path ./q3tts-base --port 7860
```

Open <http://localhost:7860>. Upload or record reference audio (any sample rate —
it is resampled to 24 kHz), enter its transcript and the text to speak, then
`Clone voice`. The run detail panel breaks down prefill / talker / vocoder time
and the realtime factor.

**The reference transcript must cover the whole clip.** A partial or mismatched
transcript degrades output badly in the official implementation too — it is a
conditioning requirement, not a tolerance. Trim audio and transcript together.

Repeated runs against the same reference reuse its cached codes and skip
re-encoding, so only the first request pays that cost.

Requires `pip install mlx mlx-lm` on Apple Silicon. For scripted use without a
UI, see `examples/qwen3_tts_mlx_clone.py`.
