# Playground

This directory contains the SGLang-Omni frontend demo.

## Overview

The playground is a browser UI for testing multimodal model interactions end to end.
It lets you send text, audio, image, and video inputs, then view streamed model responses in one place.
It also includes a built-in file browser so users can pick media files from the server filesystem.

## Functions

1. Prompting and chat
- Set system prompt and user prompt.
- Multi-turn conversation history with media context.

2. Multimodal input
- Upload audio, image, and video files from local machine.
- Record audio from microphone.
- Capture video from webcam.
- Pick media files from server/container filesystem via built-in file browser.

3. Generation controls
- Configure model generation parameters (temperature, top-p, top-k).
- Output modality selector (text only; audio options available when talker stage is implemented).

4. Streaming output
- Receive assistant responses in streaming mode (SSE).
- Display text output and returned media in chat history.

## Quick Start

Run everything with one command:

```bash
./playground/start_playground.sh \
  --pipeline qwen3-omni \
  --model-id Qwen/Qwen3-Omni-30B-A3B-Instruct
```

Then open `http://localhost:8000` in your browser.

### Custom port

```bash
./playground/start_playground.sh \
  --pipeline qwen3-omni \
  --model-id Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --port 8080
```

### SSH tunnel (for remote servers / Docker)

From your local machine:

```bash
ssh -L 8000:localhost:8000 user@host
```

## Start Manually

```bash
sglang-omni-server \
  --pipeline qwen3-omni \
  --model-id Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --port 8000 \
  --serve-playground playground/
```

## Architecture

Everything is served from a single FastAPI process:

| Endpoint | Description |
|----------|-------------|
| `/` | Playground UI (index.html, app.js, styles.css) |
| `/v1/chat/completions` | Chat completions (text + audio, streaming) |
| `/v1/audio/speech` | Text-to-speech |
| `/v1/models` | List available models |
| `/v1/fs/list` | Browse server filesystem |
| `/v1/fs/file` | Download a server file |
| `/health` | Health check |
