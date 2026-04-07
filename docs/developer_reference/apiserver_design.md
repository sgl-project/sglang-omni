# API Server Design

This page explains the API server at the level that is most useful for maintenance: where it sits in the system, which files matter, and how requests are mapped into the runtime.

If you only want to launch the server and call it, start with [API Server Quickstart](../get_started/apiserver_quickstart.md).

## Role in the System

The API server is the outer protocol layer on top of the `sglang-omni` pipeline runtime.

At a high level, the request path is:

`CLI / Python entrypoint` → `PipelineConfig` → `Pipeline Startup` → `Coordinator` → `Client` → `FastAPI`

That split keeps responsibilities clean:

- the pipeline runtime handles orchestration and execution
- the `Client` layer submits requests and assembles results
- the API server translates between HTTP/OpenAI-style payloads and those internal abstractions

## Key Files

For the current server implementation, these are the files that matter most.

| File | Role |
| --- | --- |
| `sglang_omni/serve/openai_api.py` | Defines the FastAPI app, routes, request conversion, and response formatting |
| `sglang_omni/serve/protocol.py` | Defines request and response schemas |
| `sglang_omni/serve/launcher.py` | Compiles the pipeline, starts the runtime, mounts the app, and runs Uvicorn |
| `sglang_omni/client/client.py` | Submits requests to the coordinator and aggregates text, audio, and stream results |
| `sglang_omni/cli/serve.py` | Defines the current CLI surface for `sgl-omni serve` |

If you are tracing endpoint behavior, `openai_api.py` and `client.py` are usually the best places to start.

## `create_app()` vs `launch_server()`

This is the most important distinction in the serving code.

### `create_app(client, model_name=...)`

`create_app()` only builds the FastAPI app and registers the core routes.

It does **not**:

- compile the pipeline
- start the runtime
- create the coordinator
- mount profiling routes
- run Uvicorn

Use it when you already have a live `Client` and want to embed the HTTP layer yourself.

### `launch_server(pipeline_config, ...)`

`launch_server()` is the full built-in server lifecycle.

It:

- compiles the pipeline config
- starts the runner
- creates the `Client`
- creates the FastAPI app
- mounts profiling routes
- runs Uvicorn
- stops the runtime on shutdown

Use it when you want the standard out-of-the-box server path.

## Route Surface

The current server exposes these main routes:

| Method | Path | Notes |
| --- | --- | --- |
| `GET` | `/health` | Health status from `client.health()` |
| `GET` | `/v1/models` | Single-model listing for the active pipeline |
| `POST` | `/v1/chat/completions` | Chat completions, including streaming and optional audio |
| `POST` | `/v1/audio/speech` | Text-to-speech |
| `POST` | `/start_profile` | Added by the built-in launcher |
| `POST` | `/stop_profile` | Added by the built-in launcher |

The profiling routes are only present when the server is started through `launch_server()`.

## Request Mapping

The server does not pass OpenAI-style request bodies straight into the runtime. It first converts them into internal request objects.

### Chat requests

`ChatCompletionRequest` includes standard OpenAI-style fields such as:

- `model`
- `messages`
- `temperature`
- `top_p`
- `max_tokens`
- `stop`
- `seed`
- `stream`

It also includes `sglang-omni` extensions such as:

- `images`
- `audios`
- `videos`
- `modalities`
- `audio`
- `stage_sampling`
- `stage_params`
- `request_id`

### Conversion into `GenerateRequest`

`_build_chat_generate_request()` in `openai_api.py` is the key translation point. It:

- normalizes stop sequences
- builds `SamplingParams`
- converts chat messages into internal `Message` objects
- maps per-stage sampling overrides
- stores media input and audio config in request metadata
- copies `modalities` into `output_modalities`

From that point on, the runtime works with an internal `GenerateRequest`, not the original OpenAI-style payload.

## Response Paths

### Non-streaming chat

For non-streaming chat, the path is roughly:

`chat request` → `Client.completion()` → OpenAI-style JSON response

`Client.completion()` aggregates:

- text fragments
- audio chunks
- final usage
- final finish reason

If audio is present, it is base64-encoded before being returned to the API layer.

### Streaming chat

For streaming chat, the server emits SSE events.

The important current semantics are:

- the first chunk may contain only `role="assistant"`
- text and audio are emitted as separate deltas
- the final completion chunk includes the `finish_reason`
- the stream ends with `data: [DONE]`
- `usage` is attached to the final completion chunk

### Speech / TTS

The speech route reuses the same internal request path rather than introducing a separate serving stack.

`CreateSpeechRequest` is converted into a `GenerateRequest` with:

- `output_modalities=["audio"]`
- `task="tts"` in metadata
- TTS-specific parameters stored under `tts_params`

`Client.speech()` then collects audio chunks, encodes them, and returns raw audio bytes to the HTTP layer.
