# Playground

Browser playgrounds for SGLang-Omni.

| Subdirectory | Model | UI |
|---|---|---|
| `realtime/` | Native full-duplex audio/video (MiniCPM-o validated). | HTML / CSS / JS |
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

## Native full duplex

See the [Full Duplex Interactive Model](../docs/developer_reference/full_duplex_guide.md#browser-demo) to start
microphone and camera chat with MiniCPM-o, including remote access from your browser.

## Streaming cascade with body motion

`cascade/app.py` serves the cascade page and proxies `/v1/realtime`. A
Session that publishes `smplx_motion` (StreamTalk, MIBURI) or
`smplh_flame_motion` (ELLSA 2) is drawn as a stick figure, or on the SMPL-X
body mesh when the playground is given an SMPL-X model:

```bash
python playground/cascade/app.py --api-base http://127.0.0.1:8000 --port 8080 \
    --smplx-model /path/to/SMPLX_NEUTRAL_2020.npz   # or SMPLX_MODEL_PATH=...
```

- SMPL-X is licensed for non-commercial research only
  (https://smpl-x.is.tue.mpg.de), so no model data is in this repository.
  Download `SMPLX_NEUTRAL_2020.npz` yourself under its license; the
  playground reads it from that local path at startup.
- The page fetches `/smplx/mesh.bin` (built from the model: template, joints,
  4 largest skinning weights per vertex, expression blend shapes), skins the
  mesh in the browser and draws it with three.js from jsdelivr. Shape and
  pose correctives are not applied.
- `smplx_motion` drives the body, hands, jaw and the 100 expression
  coefficients. `smplh_flame_motion` drives the body, hands and the face:
  SMPL-X 2020's 100 expression coefficients are FLAME 2020's, so its FLAME
  expression (entries 0-99), jaw (103) and eyes (104-107) apply as they are.
