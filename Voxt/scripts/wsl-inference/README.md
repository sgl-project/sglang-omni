# Serving speech (and language) models from WSL

Voxt for Windows takes speech recognition and language models from any
OpenAI-compatible endpoint (Speech models → Remote services, Language models
→ Custom). One documented deployment is sglang-omni serving the models in
WSL2 on the same PC. This page is the verified recipe for an RTX-4070-class
Windows machine; any remote endpoint works just as well.

## Prerequisites

- WSL2 with a working NVIDIA driver (`nvidia-smi` inside WSL shows the GPU).
- An sglang-omni checkout with a Python environment (`<repo>/.venv`) that has
  sglang-omni installed. The model downloads to the WSL Hugging Face cache.

## Serve the speech model

```bash
cd <sglang-omni checkout>
# Download once (see the pitfalls below — both env vars matter):
HF_HUB_DISABLE_XET=1 HF_ENDPOINT=https://hf-mirror.com \
  .venv/bin/hf download Qwen/Qwen3-ASR-1.7B
# Serve (localhost-only, conservative memory for an 8 GB card):
HF_HUB_OFFLINE=1 .venv/bin/sgl-omni serve \
  --model-path Qwen/Qwen3-ASR-1.7B --host 127.0.0.1 --port 8000 \
  --asr.engine.max_running_requests 32 --mem-fraction-static 0.7
```

Point Voxt's speech provider at `http://127.0.0.1:8000/v1` with model
`Qwen/Qwen3-ASR-1.7B` (the OpenAI-compatible preset). A language model can
be served the same way on another port and added as a Custom provider.

## Pitfalls (all hit for real on 2026-11-13)

1. **Xet silently skips the weight shards.** The download prints "✓
   Downloaded" while `blobs/` holds only the small files; the server then
   fails with "Could not resolve model architecture". Always set
   `HF_HUB_DISABLE_XET=1` (and use `HF_ENDPOINT=https://hf-mirror.com` where
   huggingface.co is slow). Verify with `du -sh
   ~/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B` — expect ~4 GB.
2. **The JIT build uses an ancient nvcc.** flashinfer's `jit/core.py`
   hardcodes `--compress-mode=size`, which nvcc older than 12.9 rejects
   (`nvcc fatal: Unknown option`). Check `nvcc --version`; if below 12.9,
   comment that flag out in
   `.venv/lib/python3.*/site-packages/flashinfer/jit/core.py` (temporary
   local patch), and point `CUDA_HOME` at the newest toolkit under
   `/usr/local`, e.g. `CUDA_HOME=/usr/local/cuda-12.6
   PATH=/usr/local/cuda-12.6/bin:$PATH`.
3. **GPU memory fraction.** If startup dies with "Loaded weights leave no
   GPU memory for the KV cache", pass `--mem-fraction-static 0.7` (the
   profiler's suggested minimum plus margin).
4. **The WSL VM shuts down when idle.** Roughly 60 s after the last wsl.exe
   session exits, the VM stops and kills the server (and wipes tmpfs /tmp).
   Host the serve command from a Windows Scheduled Task that keeps a
   foreground `wsl.exe` parent alive:

   ```bat
   wsl.exe -d Ubuntu -- bash -c "cd <repo> && HF_HUB_OFFLINE=1 CUDA_HOME=/usr/local/cuda-12.6 .venv/bin/sgl-omni serve --model-path Qwen/Qwen3-ASR-1.7B --host 127.0.0.1 --port 8000 --asr.engine.max_running_requests 32 --mem-fraction-static 0.7 >> %USERPROFILE%\asr-serve.log 2>&1"
   ```
   `schtasks /create /tn serve-asr /tr "C:\path\serve-asr.bat" /sc once /st 23:59 /f && schtasks /run /tn serve-asr`

   The task survives terminal and ssh exits and needs no desktop session.
