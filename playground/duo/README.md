# Duo browser playground

Use an eight-GPU CUDA environment with the matching SGLang Diffusion and Duo model sources. Install the Duo-enabled SGLang-Omni wheel in the active environment. The wheel includes the launcher, model preparation scripts, browser assets and backend entry point. In a source checkout you can substitute `./playground/duo/sgl-omni-duo` for `sgl-omni-duo`. The gateway requires aiohttp, Pillow, NumPy and PyAV with libx264 for H.264 transport.

To build and install the wheel from this branch, run the following in the repository root with the target Python environment active. Install the matching SGLang Diffusion checkout and model-source dependencies as described in the Duo model guide.

```bash
python -m pip wheel --no-deps . --wheel-dir dist
python -m pip install dist/sglang_omni-*.whl
```

Set the following paths in each terminal. The model directory contains the downloaded `Duo-1.0-causal-base/` and `Duo-1.0-distill/` releases.

```bash
export DUO_MODEL_DIR="$PWD/models"
export DUO_ASSET_DIR="$PWD/assets"
export DUO_MODEL_SOURCE=/path/to/Duo/third_party/duo
export DUO_CODEC_CONFIG_DIR=/path/to/Duo/duo/models/realization/modules/native_vae
export PYTHONPATH="$DUO_MODEL_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
```

Prepare the local weights once, then start each service in its own terminal using the same environment.

```bash
sgl-omni-duo prepare
sgl-omni-duo merge
```

| Terminal | Command | Local endpoint |
|---|---|---|
| 1 | `sgl-omni serve --model-path "$DUO_MODEL_DIR" --enable-realtime --host 127.0.0.1 --port 18082` | ws://127.0.0.1:18082/v1/realtime |
| 2 | `sgl-omni-duo web` | http://localhost:18081/ |

Open the playground when `curl -fsS http://127.0.0.1:18081/status` returns `"ready": true`. For remote GPUs, forward the loopback port with `ssh -L 18081:127.0.0.1:18081 user@gpu-host`, then open the same localhost URL. Set the gateway’s `DUO_BACKEND_PORT` to the serve port. `DUO_WEB_PORT` changes the browser port. Extra command-line arguments are forwarded to the selected service. For example, `--realization-kv-reset-interval-blocks 43` enables periodic reset. The packaged model guide is installed under `share/sglang-omni/duo/docs/duo.md`.

Supply the seaside portrait at `portraits/seaside/portrait.jpg`. The example clip requires sorted JPEG files in `sample/frames/` at 24 FPS, aligned mono PCM16 at 16 kHz in `sample/audio.pcm`, and a browser preview at `sample/example.mp4`. Include at least 17 frames and 11,333 audio samples. The gateway reads the sample at startup.

Open the gateway through HTTPS or localhost, then choose the sample or allow camera and microphone access. One session can generate at a time. Omit the backend reset interval to allow a manual DiT KV reset after the first completed block. Setting `--realization-kv-reset-interval-blocks N` to a positive integer disables the button and resets before blocks N, 2N, and so on (zero-based). Qwen and Codec state continue across the reset.
