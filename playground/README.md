# Playground

This directory contains several playground interfaces for SGLang-Omni.

| Subdirectory | Description |
|---|---|
| `web/` | Full-featured HTML/CSS/JS UI served directly by the sglang-omni server. Supports text, audio, image, video inputs and a built-in file browser. |
| `gradio/` | Lightweight Gradio app that connects to a running server via HTTP. Text chat with streaming, model selector, and generation parameter controls. |
| `realtime-ws/` | Standalone websocket realtime app with server-side VAD, text input, microphone streaming, and streamed assistant audio playback. |

## Web Playground

The web playground is embedded in the backend — a single process serves both the API and the UI.

```bash
uv pip install -v -e ".[dev]"
./playground/web/start.sh \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct
```

Then open `http://localhost:8000` in your browser.

## Realtime WebSocket Playground

Install the realtime extra before launching:

```bash
uv pip install -v -e ".[realtime]"
```

Launch the backend plus standalone frontend app with one command:

```bash
./playground/realtime-ws/start.sh [--mock] [realtime-options] [backend-options...]
```

Minimal usable commands:

```bash
# local smoke test
./playground/realtime-ws/start.sh --mock

# real model
./playground/realtime-ws/start.sh --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct
```

In normal backend mode, pass the usual speech server flags such as `--model-path`:

```bash
./playground/realtime-ws/start.sh \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct
```

Then open `http://localhost:7862`.

For a browser smoke test without loading any model, launch the mock realtime API:

```bash
./playground/realtime-ws/start.sh --mock
```

That path exercises:

- browser microphone capture over websocket PCM streaming
- server-side VAD turn detection
- automatic response start after speech stop
- streamed assistant audio playback in the browser
- text prompts over the same websocket session

The mock backend returns canned text plus a synthetic tone sequence instead of
calling the inference pipeline.

### Remote browser over SSH port forwarding

Because the transport is plain HTTP + WebSocket, standard SSH forwarding is
enough for remote browser testing.

Example:

```bash
./playground/realtime-ws/start.sh --mock
```

Forward the backend port and the frontend port from the remote machine:

```bash
ssh -L 8000:localhost:8000 -L 7862:localhost:7862 user@host
```

For the full launcher help, run:

```bash
./playground/realtime-ws/start.sh --help
```

The websocket playground:

- streams microphone PCM to the backend over `/v1/realtime/ws`
- runs server-side VAD to auto-trigger one inference turn per utterance
- supports manual push-to-talk and text prompts in the same session
- streams assistant audio back over the websocket and auto-plays it in the browser
- keeps the frontend separate from the inference API server

### Custom port

```bash
./playground/web/start.sh \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --port 8080
```

## Gradio Playground

### Install

```bash
pip install "sglang-omni[gradio]"
# or just: pip install gradio httpx
```

### Launch (one command)

`start.sh` launches the backend server, waits for it to become healthy, then starts the Gradio UI:

```bash
./playground/gradio/start.sh \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct
```

Backend runs on `http://localhost:8000`, Gradio UI on `http://localhost:7860`. Use `--port` / `--gradio-port` to change, `--share` for a public link.

### Connect to an existing server

If you already have a server running, use `app.py` directly:

```bash
python playground/gradio/app.py --api-base http://localhost:8000
```

## SSH tunnel (for remote servers / Docker)

From your local machine:

```bash
ssh -L 8000:localhost:8000 -L 7860:localhost:7860 user@host
```

## Architecture

| Endpoint | Description |
|----------|-------------|
| `/` | Web playground UI (index.html, app.js, styles.css) |
| `/v1/chat/completions` | Chat completions (text + audio, streaming) |
| `/v1/audio/speech` | Text-to-speech |
| `/v1/realtime/ws` | Realtime websocket session transport |
| `/v1/models` | List available models |
| `/v1/fs/list` | Browse server filesystem |
| `/v1/fs/file` | Download a server file |
| `/health` | Health check |
