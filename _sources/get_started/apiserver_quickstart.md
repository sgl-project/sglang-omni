# API Server Quickstart

This page is the shortest path from “I have the repo” to “the API server responds to a request”.

`sglang-omni` exposes an OpenAI-compatible API server on top of its multi-stage pipeline runtime. That server is the main HTTP entry point for:

- chat completions
- streaming responses
- model listing
- health checks
- text-to-speech

If you want the internal design rather than the usage flow, see [API Server Design](../developer_reference/apiserver_design.md).

## What This Server Is

The API server is an adapter between HTTP clients and the internal pipeline runtime:

`HTTP request` → `FastAPI app` → `Client` → `Coordinator` → `Pipeline stages`

In other words, it does not run model logic directly. Its job is to translate OpenAI-style requests into internal requests and format the results back into HTTP responses.

## Start the Server

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

If you already have a pipeline config file, you can also pass `--config path/to/config.yaml`. In the current CLI, `--model-path` is still required even when `--config` is provided.

## Check That It Works

### Health check

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

### List the served model

```bash
curl -s http://localhost:8000/v1/models
```

This endpoint returns a single-model list. The model ID comes from `--model-name` if you set it, otherwise from the pipeline name.

## Send a Minimal Chat Request

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

## Streaming and Multi-modal Requests

### Streaming

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

### Multi-modal input and output

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

## Text-to-Speech

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

## Common Errors

When requests fail, the server returns standard HTTP error codes:

- `400 Bad Request`: malformed request body or invalid parameters
- `500 Internal Server Error`: runtime error during generation (check server logs for details)
- `503 Service Unavailable`: the runtime is not healthy (verify with `/health`)

If you see a 500 error, check the server logs for the full traceback. Common issues include:
- unsupported media formats
- out-of-memory errors
- missing model files

## Next Reading

- [API Server Design](../developer_reference/apiserver_design.md)
- [Architecture](../developer_reference/architecture.md)
