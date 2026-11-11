(api-server-quickstart)=
# Quickstart

Start Qwen3-ASR-1.7B on one CUDA GPU and transcribe a short audio clip.

## Install and start

Use Linux with a compatible NVIDIA GPU and driver. Choose Docker or uv below;
see [CUDA Installation](installation.md) for the requirements and
[other platforms](installation.md#other-platforms) for non-CUDA setups.

**Docker (recommended):** Follow [Docker installation](installation.md#docker-recommended)
to pull the image, start the container, and install SGLang-Omni inside it.
Run the launch command below in that container.

**uv:** With the {ref}`CUDA prerequisites <option-b-manual-install>` installed,
create a Python 3.12 environment on the host:

```bash
uv venv .venv -p 3.12
source .venv/bin/activate
uv pip install --prerelease=allow "sglang-omni==0.2.0"
```

From either environment, start the server:

```bash
CUDA_VISIBLE_DEVICES=0 sgl-omni serve \
  --model-path Qwen/Qwen3-ASR-1.7B \
  --model-name Qwen/Qwen3-ASR-1.7B \
  --host 127.0.0.1 \
  --port 8000
```

The first launch downloads the checkpoint. Wait for the server to finish
starting. If it does not become ready, check the server log and
{ref}`/health <health-check>`.

## Send a request

In a second terminal on the same host, download the sample WAV used in the
[Qwen3-ASR Cookbook](../cookbook/qwen3_asr.md#transcribe-audio) and upload it:

```bash
curl -fL -o query_to_cars.wav \
  https://raw.githubusercontent.com/sgl-project/sglang-omni/7fc1a31850efbc1138e1895b5a3bcbeecf3a8481/tests/data/query_to_cars.wav

curl -fsS http://localhost:8000/v1/audio/transcriptions \
  -F model=Qwen/Qwen3-ASR-1.7B \
  -F file=@query_to_cars.wav \
  -F response_format=json
```

Check the JSON response's `text` field for the spoken question, “How many
cars are there in the picture?”

Keep the server running to try [streaming transcription](examples.md#stream-a-transcript).

## API server reference

The sections below cover the HTTP server, health checks, and additional API
requests. For internals, see [API Server Design](../developer_reference/apiserver_design.md).

### What This Server Is

The API server is an adapter between HTTP clients and the internal pipeline runtime:

`HTTP request` → `FastAPI app` → `Client` → `Coordinator` → `Pipeline stages`

In other words, it does not run model logic directly. Its job is to translate OpenAI-style requests into internal requests and format the results back into HTTP responses.

### Start the Server

The installed CLI entrypoint is `sgl-omni`.

The simplest way to start the server is to provide a model path and let `sglang-omni` build the pipeline config for you:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --host 0.0.0.0 \
  --port 8000
```

The most useful flags are:

- `--model-path`: Hugging Face model ID or local model directory
- `--host`: bind address, default `0.0.0.0`
- `--port`: bind port, default `8000`
- `--model-name`: override the model name returned by `/v1/models`
- `--log-level`: logging level for the server process

If you already have a pipeline config file, you can also pass `--config path/to/config.yaml`. When the config file contains `model_path`, `--model-path` is optional and can be used as an override.

(unified-sglang-cli)=
**Unified SGLang CLI**

When SGLang-Omni is installed alongside an SGLang release that supports serve
backend plugins, the same server can be launched through the core executable:

```bash
sglang serve <model-name-or-path> --model-type omni [additional-arguments]
```

Config-only launches are supported as well:

```bash
sglang serve --model-type omni --config path/to/config.yaml
```

SGLang-Omni intentionally does not install another `sglang` executable. It
registers an `omni` serve backend with SGLang core, while `sgl-omni serve`
remains available as a backward-compatible alias. Both commands accept the same
arguments, including the positional model path, so these two launches are
equivalent:

```bash
sglang serve <model-name-or-path> --model-type omni --port 8000
sgl-omni serve <model-name-or-path> --port 8000
```

Automatic Omni model detection is not enabled yet; select the backend
explicitly with `--model-type omni`.

### Check That It Works

(health-check)=
**Health check**

```bash
curl -s http://localhost:8000/health
```

Example response:

```json
{
  "status": "healthy",
  "running": true
}
```

The server returns:

- `200` when the runtime is healthy
- `503` when the HTTP server is up but the underlying runtime reports unhealthy status

(list-the-served-model)=
**List the served model**

```bash
curl -s http://localhost:8000/v1/models
```

This endpoint returns a single-model list. The model ID comes from `--model-name` if you set it, otherwise from the pipeline name.

### Send a Minimal Chat Request

The core endpoint is `POST /v1/chat/completions`.

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 128,
    "stream": false
  }'
```

The response follows the OpenAI chat completion shape. In the common case, the text is in `choices[0].message.content`.

Besides `model` and `messages`, the most useful request fields are:

- `temperature`
- `top_p`
- `max_tokens`
- `stop`
- `seed`
- `stream`

### Streaming and Multi-modal Requests

(streaming)=
**Streaming**

Set `stream` to `true` to receive Server-Sent Events (SSE):

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [
      {"role": "user", "content": "Write a short greeting."}
    ],
    "stream": true
  }'
```

A few details matter here:

- the response type is `text/event-stream`
- the first chunk may contain only `role="assistant"`
- the stream ends with `data: [DONE]`
- `usage` is attached to the final completion chunk

(multi-modal-input-and-output)=
**Multi-modal input and output**

`sglang-omni` extends the standard OpenAI chat schema with a few extra fields:

- `images`
- `audios`
- `videos`
- `modalities`
- `audio`
- `stage_sampling`
- `stage_params`

For example, a video request with text output looks like this:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [
      {"role": "user", "content": "What is happening in this video?"}
    ],
    "videos": ["/absolute/path/to/demo.mp4"],
    "modalities": ["text"],
    "max_tokens": 128,
    "stream": false
  }'
```

The `videos`, `images`, and `audios` fields accept either local file paths or HTTP(S) URLs.

### Text-to-Speech

The server also exposes `POST /v1/audio/speech`.

```bash
curl -s http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "input": "Hello from SGLang-Omni.",
    "voice": "default",
    "response_format": "wav"
  }' \
  -o speech.wav
```

Two things to remember:

- the response body is audio bytes, not JSON
- the actual output format may differ from the requested one if the encoder falls back to another supported codec

### Common Errors

When requests fail, the server returns standard HTTP error codes:

- `400 Bad Request`: malformed request body or invalid parameters
- `422 Unprocessable Entity`: a request field fails schema validation, such as empty `messages` without top-level `audios`, `images` or `videos`, or `max_tokens` below 1 (the response names the field)
- `500 Internal Server Error`: runtime error during generation (check server logs for details)
- `503 Service Unavailable`: the request queue is full (retry later), or the runtime is not healthy (verify with `/health`)

If you see a 500 error, check the server logs for the full traceback. Common issues include:
- unsupported media formats
- out-of-memory errors
- missing model files

### Next Reading

- [API Server Design](../developer_reference/apiserver_design.md)
- [Developer Reference](../developer_reference/main.md)
