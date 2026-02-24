# Playground

This directory contains the SGLang-Omni frontend demo and filesystem API.

## Overview

The playground is a browser UI for testing multimodal model interactions end to end.
It lets you send text, audio, image, and video inputs, then view streamed model responses in one place.
It also includes a server-side file browser so users can pick media directly from files already inside the container.

## Functions

1. Prompting and chat
- Set system prompt and user prompt.
- View conversation history and clear it.

2. Multimodal input
- Upload audio, image, and video files from local machine.
- Record audio from microphone.
- Capture video from webcam.
- Pick media files from server/container filesystem via FS API modal.

3. Generation controls
- Configure model generation parameters (for example temperature/top-p/top-k in the UI).
- Optionally request audio output when supported by backend.

4. Streaming output
- Receive assistant responses in streaming mode.
- Display text output and returned media in chat history.

5. Filesystem service integration
- Browse directories and select files through `playground/fs_api.py`.
- Separate allowed filesystem scope from default browse start path.

## Components

1. Frontend static page (`index.html` + `app.js` + `styles.css`)
2. Filesystem API (`playground/fs_api.py`, default port `8001`) (used to list the filesystem of the server that hosts sglang omni so that you can upload data from it)

## How to Start

### Start frontend + FS API together (recommended)

Run from repository root:

```bash
./playground/start_playground_and_fs.sh
```

Default ports:
- Frontend: `3000`
- FS API: `8001`

You can pass custom ports:

```bash
./playground/start_playground_and_fs.sh 3001 9001
```

### Start separately

Frontend:

```bash
python -m http.server 3000 --directory playground
```

FS API:

```bash
python -m playground.fs_api --host 0.0.0.0 --port 8001
```
