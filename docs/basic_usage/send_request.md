# Send Requests

SGLang-Omni exposes an OpenAI-compatible API server that supports:

- Chat completions
- Streaming responses
- Multi-modal input (image, audio, video)
- Text-to-speech
- Health checks and model listing

For internal design details, see [API Server Design](../developer_reference/apiserver_design.md).

## Prerequisites

- Install SGLang-Omni by following [Installation](../get_started/installation.md).

## Supported Models

| Model | Description |
| --- | --- |
| [`Qwen/Qwen3-Omni-30B-A3B-Instruct`](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) | Natively end-to-end multilingual omni-modal foundation model. Processes text, images, audio, and video, and delivers real-time streaming responses in both text and natural speech. |
| [`fishaudio/s2-pro`](https://huggingface.co/fishaudio/s2-pro) | Leading text-to-speech model with fine-grained inline control of prosody and emotion. |

## Launch the Server

The installed CLI entrypoint is `sgl-omni`. Start the server by providing a model path:

```bash
sgl-omni serve --model-path <model>
```

For example:

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

## Health Check

Use `/health` to confirm that the HTTP server is up and the runtime is healthy:

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

## Model Listing

Use `/v1/models` to see the model exposed by the current server:

```bash
curl -s http://localhost:8000/v1/models
```

This endpoint returns a single-model list. The model ID comes from `--model-name` if you set it, otherwise from the pipeline name.


## Model-Specific Guides

- [Omni Model Usage](./qwen3_omni.md)
- [TTS Model Usage](./tts_s2pro.md)
