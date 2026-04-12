# Playground

This directory contains two playground interfaces for SGLang-Omni.

| Subdirectory | Description |
|---|---|
| `web/` | Full-featured HTML/CSS/JS UI served directly by the sglang-omni server. Supports text, audio, image, video inputs and a built-in file browser. |
| `gradio/` | Lightweight Gradio app that connects to a running server via HTTP. Text chat with streaming, model selector, and generation parameter controls. |
| `realtime/` | Standalone WebRTC prototype app with server-side VAD, automatic turn triggering, and streamed assistant audio playback. |

## Web Playground

The web playground is embedded in the backend — a single process serves both the API and the UI.

```bash
uv pip install -v -e ".[dev]"
./playground/web/start.sh \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct
```

Then open `http://localhost:8000` in your browser.

## Realtime Prototype

Install the realtime extra before launching:

```bash
uv pip install -v -e ".[realtime]"
```

Launch the backend plus standalone frontend app with one command:

```bash
./playground/realtime/start.sh [--mock] [realtime-options] [serve-options...]
```

Minimal usable commands:

```bash
# local smoke test
./playground/realtime/start.sh --mock

# remote smoke test with TURN
./playground/realtime/start.sh --mock --with-turn

# real model
./playground/realtime/start.sh --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct
```

In normal backend mode, pass the usual serve flags such as `--model-path`:

```bash
./playground/realtime/start.sh \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct
```

Then open `http://localhost:7861`.

For a browser smoke test without loading any model, launch the mock realtime API:

```bash
./playground/realtime/start.sh --mock
```

That path still exercises:

- browser microphone capture over WebRTC
- server-side VAD turn detection
- automatic response start after speech stop
- streamed assistant audio playback in the browser

The mock backend returns canned text plus a synthetic tone sequence instead of
calling the inference pipeline.

### Remote browser over SSH port forwarding

SSH port forwarding only tunnels the HTTP signaling path. It does not carry the
actual WebRTC media transport, so remote browser testing usually needs a TURN
server.

Example:

```bash
./playground/realtime/start.sh --mock \
  --with-turn
```

You can also add a STUN URL with another `--ice-server`, but for tunneled or
firewalled environments TURN is the important part.

`start.sh --with-turn` now assumes `coturn` is already installed on the host
and launches the local `turnserver` binary directly. If Tailscale is installed,
it auto-detects the first Tailscale IPv4 and uses that as `--turn-host`.

If you do not want auto-detection, override it explicitly:

```bash
./playground/realtime/start.sh --mock \
  --with-turn \
  --turn-host turn.example.com \
  --turn-user demo-user \
  --turn-password demo-pass
```

For the full launcher help, run:

```bash
./playground/realtime/start.sh --help
```

If the server is behind NAT and `turn.example.com` resolves to a public IP that
is different from the machine's local interface address, also pass:

```bash
--turn-public-ip PUBLIC_IP
```

For TURN to work, the TURN host must be reachable by the browser on the TURN
listener port and on the configured relay port range.

If you are running inside WSL and exposing the demo through the Windows host,
you need to forward both:

- the TURN listener port, by default `3479`
- the relay allocation range, for example `49160-49200`

Forwarding only the TURN listener port is not enough for a normal coturn relay.

The prototype:

- uses WebRTC for microphone uplink and assistant audio playback
- runs server-side VAD to auto-trigger one inference turn per utterance
- optionally buffers a webcam track and injects recent sampled frames into the turn request
- reuses the existing request-oriented pipeline rather than modifying the coordinator
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
| `/v1/models` | List available models |
| `/v1/fs/list` | Browse server filesystem |
| `/v1/fs/file` | Download a server file |
| `/health` | Health check |
