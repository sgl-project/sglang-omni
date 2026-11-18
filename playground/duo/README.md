# Duo browser playground

Start the [Duo realtime backend and ASR](../../docs/models/duo.md), then launch the gateway in the same Python environment with aiohttp, Pillow, NumPy and PyAV (libx264 for H.264 streaming).

```bash
python playground/duo/server.py --backend-url http://127.0.0.1:18082 \
  --sample-directory /path/to/sample --portrait-directory /path/to/portraits --port 18081
```

Supply the seaside portrait at `portraits/seaside/portrait.jpg`. The example clip requires sorted JPEG files in `sample/frames/` at 24 FPS, aligned mono PCM16 at 16 kHz in `sample/audio.pcm`, and a browser preview at `sample/example.mp4`. Include at least 17 frames and 11,333 audio samples. The gateway reads the sample at startup.

Open the gateway through HTTPS or localhost, then choose the sample or allow camera and microphone access. One session can generate at a time. Omit the backend reset interval to allow a manual DiT KV reset after the first completed block. Setting `--realization-kv-reset-interval-blocks N` to a positive integer disables the button and resets before blocks N, 2N, and so on (zero-based). Qwen and Codec state continue across the reset.
