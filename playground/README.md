# Playground

Browser playgrounds for SGLang-Omni.

| Subdirectory | Model | UI |
|---|---|---|
| `realtime/` | Native full-duplex audio/video (MiniCPM-o validated). | HTML / CSS / JS |
| `qwen-omni/` | Qwen3-Omni — multimodal chat (text / audio / image / video). | HTML / CSS / JS |
| `s2pro/` | S2 Pro — text-to-speech with voice cloning, streaming and non-streaming. | Gradio |
| `duo/` | Duo — streaming video and speech with a reference portrait. | HTML / CSS / JS |
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

## Duo realtime video and speech

Use the Duo-enabled Omni environment with [SGLang 0.5.23 and the Duo Diffusion components](https://github.com/Anmuliar/sglang/tree/duo-0.5.23) and eight CUDA GPUs. From the repository root, set these local paths in each terminal:

```bash
export DUO_MODEL_DIR="$PWD/models"
export DUO_ASSET_DIR="$PWD/assets"
export DUO_MODEL_SOURCE=/path/to/Duo/third_party/duo
export PYTHONPATH="$DUO_MODEL_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
```

For a first deployment, place the `Duo-1.0-causal-base` and `Duo-1.0-distill` releases under `$DUO_MODEL_DIR`, then prepare the codec, realization and merged intent weights once:

```bash
export DUO_CODEC_CONFIG_DIR=/path/to/Duo/duo/models/realization/modules/native_vae
./playground/duo/sgl-omni-duo prepare
./playground/duo/sgl-omni-duo merge
```

Start the complete backend in one terminal. The registered Duo pipeline owns the Qwen3-ASR worker on GPU 3, Intent TP4 on GPUs 0–3, Realization TP8 on GPUs 0–7 and four Codec tiles on GPUs 4–7.

```bash
sgl-omni serve --model-path "$DUO_MODEL_DIR" --enable-realtime --host 127.0.0.1 --port 18082
```

Once the backend is ready, start the browser gateway in a second terminal. Supply `portraits/seaside/portrait.jpg` under `$DUO_ASSET_DIR`, along with `sample/frames/*.jpg` (24 FPS), `sample/audio.pcm` (aligned 16 kHz mono PCM16) and `sample/example.mp4`. Use a clip with speech to try a spoken exchange.

```bash
# Terminal 2: check backend readiness, then launch the frontend
curl -fsS http://127.0.0.1:18082/duo/configuration && ./playground/duo/sgl-omni-duo web
```

Open <http://localhost:18081>. For a remote GPU host, forward the frontend port with `ssh -L 18081:127.0.0.1:18081 user@host`. When using the installed CUDA wheel, use `sgl-omni-duo` in place of `./playground/duo/sgl-omni-duo`.

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

## Native full duplex

See the [Full-Duplex Guide](../docs/developer_reference/full_duplex_guide.md#browser-demo) to start
microphone and camera chat with MiniCPM-o, including remote access from your browser.
