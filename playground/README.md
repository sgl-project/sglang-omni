# Playground

This directory contains the SGLang-Omni frontend demo and filesystem API.

## Overview

The playground is a browser UI for testing multimodal model interactions end to end.
It lets you send text, audio, image, and video inputs, then view streamed model responses in one place.
It also includes a server-side file browser so users can pick media directly from files already inside the container.

## Functions

1. Prompting and chat
- Set system prompt and user prompt.
- Multi-turn conversation history with media context.

2. Multimodal input
- Upload audio, image, and video files from local machine.
- Record audio from microphone.
- Capture video from webcam.
- Pick media files from server/container filesystem via FS API modal.

3. Generation controls
- Configure model generation parameters (temperature, top-p, top-k).
- Output modality selector (text only; audio options available when talker stage is implemented).

4. Streaming output
- Receive assistant responses in streaming mode (SSE).
- Display text output and returned media in chat history.

5. Filesystem service integration
- Browse directories and select files through `playground/fs_api.py`.
- Separate allowed filesystem scope from default browse start path.

## Components

1. Frontend static page (`index.html` + `app.js` + `styles.css`)
2. Filesystem API (`playground/fs_api.py`, default port `8001`)
3. Backend API server (`sglang_omni/serve/openai_api.py`, default port `8000`)

## Quick Start

Run everything (backend + frontend + FS API) with one command:

```bash
CUDA_VISIBLE_DEVICES=5 ./playground/start_playground.sh \
  --pipeline qwen3-omni \
  --model-id Qwen/Qwen3-Omni-30B-A3B-Instruct
```

Then open `http://localhost:3000` in your browser.

### Custom ports

```bash
./playground/start_playground.sh \
  --pipeline qwen3-omni \
  --model-id Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --backend-port 8080 \
  --frontend-port 3001
```

### SSH tunnel (for remote servers / Docker)

From your local machine:

```bash
ssh -L 3000:localhost:3000 -L 8000:localhost:8000 -L 8001:localhost:8001 user@host
```

## Start Separately

If you prefer to manage processes individually:

Backend:

```bash
CUDA_VISIBLE_DEVICES=5 sglang-omni-server \
  --pipeline qwen3-omni \
  --model-id Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --port 8000
```

Frontend + FS API:

```bash
./playground/start_playground_and_fs.sh
```

Or manually:

```bash
python -m http.server 3000 --directory playground
python -m playground.fs_api --host 0.0.0.0 --port 8001
```

## Default Ports

| Service | Port | Env var |
|---------|------|---------|
| Backend API | 8000 | `BACKEND_PORT` |
| Frontend | 3000 | `FRONTEND_PORT` |
| FS API | 8001 | `FS_PORT` |
