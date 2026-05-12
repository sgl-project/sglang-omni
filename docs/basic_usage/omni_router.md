# Omni Router Usage

The SGLang-Omni Router is an external HTTP router for Omni V1 deployments. It
fronts multiple complete Omni V1 API servers and exposes one OpenAI-compatible
endpoint to clients.

Use the router when you launch more than one `sgl-omni serve` process and want
one stable endpoint for request distribution, health tracking, and worker-pool
control.

## Router Topology

The router is an external HTTP process:

```text
client
  |
  v
sgl-omni-router
  |
  +-- sgl-omni serve worker A
  +-- sgl-omni serve worker B
```

Each worker is a complete Omni V1 HTTP server. The router does not load model
weights or split a single request across workers. It selects one routable worker
for each request, forwards the original request bytes, and returns the worker
response with router diagnostic headers.

## Launch Workers and Router From YAML

For a local homogeneous pool, `sgl-omni-router` can start the worker replicas
and then start the router after all managed workers pass `/health`:

```bash
sgl-omni-router \
  --host 0.0.0.0 \
  --port 8008 \
  --launcher-config examples/configs/qwen3_omni_router.yaml \
  --policy round_robin \
  --health-failure-threshold 2 \
  --health-success-threshold 1 \
  --health-check-interval-secs 10 \
  --log-level info
```

Example launcher config:

```yaml
launcher:
  backend: local
  model_path: Qwen/Qwen3-Omni-30B-A3B-Instruct
  model_name: qwen3-omni
  num_workers: 2
  num_gpus_per_worker: 2
  worker_host: 127.0.0.1
  worker_base_port: 8011
  wait_timeout: 600
```

`backend: local` means the router process starts and manages worker
subprocesses on the same machine. The launched workers are complete Omni V1
servers started with `sgl-omni serve --version v1`; they are not partial
pipeline stages. The router waits for every managed worker to pass `/health`
before it starts accepting client traffic, and it stops those managed workers
when the router exits.

`num_gpus_per_worker` controls automatic GPU grouping. The full Qwen3-Omni
speech topology uses two logical GPUs per worker by default: thinker, image
encoder, and audio encoder run on logical GPU 0, while talker and code2wav run
on logical GPU 1. With `num_workers: 2` and `num_gpus_per_worker: 2`, the
launcher assigns `0,1` to the first worker and `2,3` to the second worker when
four CUDA devices are visible.

Set `worker_gpu_ids` only when you need explicit placement. Each entry maps one
`CUDA_VISIBLE_DEVICES` value to one worker, for example
`worker_gpu_ids: ["0,1", "2,3"]` for two full Qwen3-Omni speech workers. On a
two-GPU machine, use `num_workers: 1` for the full speech topology, or add
`worker_extra_args: "--text-only"` if you intentionally want two one-GPU
workers for text output.

Use `worker_extra_args` for public Omni V1 serve options that are specific to
the worker process, such as `--mem-fraction-static`, `--thinker-tp-size`, or
`--text-only`. These arguments are passed to `sgl-omni serve --version v1`
after the launcher-owned flags. When no memory flags are provided, Omni V1 uses
its normal auto-sizing path.

Use `worker_capabilities` when managed workers intentionally expose only part
of the Omni API surface. For example, text-only workers should not advertise
speech or audio-output support:

```yaml
launcher:
  backend: local
  model_path: Qwen/Qwen3-Omni-30B-A3B-Instruct
  model_name: qwen3-omni
  num_workers: 2
  num_gpus_per_worker: 1
  worker_extra_args: "--text-only"
  worker_capabilities:
    - chat
    - streaming
    - image_input
    - audio_input
    - video_input
```

If `worker_capabilities` is omitted and `worker_extra_args` contains
`--text-only`, the router registers the managed workers with the same text-only
capability set shown above.

## Launch Worker Servers Manually

Start each Omni V1 worker separately. The example below launches two full
Qwen3-Omni speech workers on different GPU pairs and ports:

```bash
CUDA_VISIBLE_DEVICES=0,1 sgl-omni serve \
  --version v1 \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --model-name qwen3-omni \
  --host 0.0.0.0 \
  --port 8011
```

```bash
CUDA_VISIBLE_DEVICES=2,3 sgl-omni serve \
  --version v1 \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --model-name qwen3-omni \
  --host 0.0.0.0 \
  --port 8012
```

Worker URLs passed to the router must be base URLs such as
`http://127.0.0.1:8011`. Do not include endpoint paths, query strings, or
fragments.

## Launch the Router

Start the router with the worker URLs:

```bash
sgl-omni-router \
  --host 0.0.0.0 \
  --port 8008 \
  --worker-urls http://127.0.0.1:8011 http://127.0.0.1:8012 \
  --policy round_robin \
  --health-failure-threshold 2 \
  --health-success-threshold 1 \
  --health-check-interval-secs 10 \
  --log-level info
```

## Router Arguments

The table below lists the router command-line arguments.

| Argument | Default | Description |
|---|---|---|
| `--host` | `0.0.0.0` | Host interface for the router HTTP server. |
| `--port` | `8000` | Port for the router HTTP server. |
| `--worker-urls` | not set | Space-separated Omni V1 worker base URLs for a homogeneous worker pool. |
| `--worker-config` | not set | JSON file that defines workers and optional per-worker model/capability metadata. |
| `--launcher-config` | not set | YAML file for a managed local worker pool. Do not use with `--worker-urls` or `--worker-config`. |
| `--policy` | `round_robin` | Routing policy: `round_robin`, `least_request`, or `random`. |
| `--model` | not set | Model name assigned to every worker when using `--worker-urls`. Do not use with `--worker-config`. |
| `--request-timeout-secs` | `1800` | Timeout for proxied worker requests. |
| `--max-payload-size` | `536870912` | Maximum request body size accepted by the router, in bytes. |
| `--max-connections` | `100` | Maximum number of HTTP connections used by the router's upstream worker client. |
| `--health-failure-threshold` | `3` | Consecutive failed health checks or routed request failures before a worker becomes unhealthy. |
| `--health-success-threshold` | `2` | Consecutive successful health checks before an unhealthy or unknown worker becomes healthy. |
| `--health-check-timeout-secs` | `5` | Timeout for one worker health-check request. |
| `--health-check-interval-secs` | `10` | Interval between background worker health checks. |
| `--health-check-endpoint` | `/health` | Worker endpoint used by background health checks. |
| `--log-level` | `info` | Router and Uvicorn log level. |

Routing policies:

- `round_robin`: rotates through routable workers in order.
- `least_request`: selects a routable worker with the fewest active data-plane
  requests, then round-robins among ties.
- `random`: selects a random routable worker.

Pass exactly one of `--launcher-config`, `--worker-urls`, or
`--worker-config`. Use `--worker-config` when workers serve different models or
only a subset of Omni capabilities:

```json
{
  "workers": [
    {
      "url": "http://127.0.0.1:8011",
      "model": "qwen3-omni",
      "capabilities": ["chat", "image_input", "video_input"]
    },
    {
      "url": "http://127.0.0.1:8012",
      "model": "qwen3-omni",
      "capabilities": ["chat", "audio_input", "audio_output", "speech"]
    }
  ]
}
```

Then launch with:

```bash
sgl-omni-router \
  --host 0.0.0.0 \
  --port 8008 \
  --worker-config workers.json \
  --policy least_request
```

## Check Router and Worker State

The router exposes separate process and worker-pool health surfaces:

```bash
curl -i http://127.0.0.1:8008/live
curl -i http://127.0.0.1:8008/ready
curl -i http://127.0.0.1:8008/health
curl -s http://127.0.0.1:8008/workers
curl -s http://127.0.0.1:8008/v1/models
```

The endpoints have different meanings:

- `GET /live`: the router process is running. This does not wait for workers to
  become healthy.
- `GET /ready`: at least one worker is routable. This returns `503` when all
  workers are unhealthy, dead, disabled, or still unknown.
- `GET /health`: worker-pool health summary. This returns `503` when no worker
  is routable.
- `GET /workers`: detailed worker state, including `health_state`, `disabled`,
  `routable`, `active_requests`, failure counters, and last error.
- `GET /v1/models`: merged model list from routable workers.

## Send Requests Through the Router

Point clients at the router port instead of the worker ports. The request schema
is the same OpenAI-compatible schema used by each worker server.

Image input with text output:

```bash
curl -i http://127.0.0.1:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-request-id: router-image-1" \
  -d '{
    "model": "qwen3-omni",
    "messages": [
      {"role": "user", "content": "How many cars are there in the image? Answer briefly."}
    ],
    "images": ["tests/data/cars.jpg"],
    "modalities": ["text"],
    "max_tokens": 16
  }'
```

Streaming text:

```bash
curl -N http://127.0.0.1:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-request-id: router-stream-1" \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": "Say hello briefly."}],
    "stream": true,
    "max_tokens": 16
  }'
```

The router preserves the original request body. For ordinary JSON requests, it
parses a bounded amount of request metadata for worker selection and forwards
the original bytes to the selected worker.

## Manage Workers

Add a worker at runtime:

```bash
curl -s http://127.0.0.1:8008/workers \
  -H "Content-Type: application/json" \
  -d '{"url":"http://127.0.0.1:8013","model":"qwen3-omni"}'
```

Disable a worker without deleting it:

```bash
curl -s -X PUT http://127.0.0.1:8008/workers/http%3A%2F%2F127.0.0.1%3A8013 \
  -H "Content-Type: application/json" \
  -d '{"disabled":true}'
```

Mark a worker dead for manual quarantine:

```bash
curl -s -X PUT http://127.0.0.1:8008/workers/http%3A%2F%2F127.0.0.1%3A8013 \
  -H "Content-Type: application/json" \
  -d '{"is_dead":true}'
```

Recover a manually dead worker:

```bash
curl -s -X PUT http://127.0.0.1:8008/workers/http%3A%2F%2F127.0.0.1%3A8013 \
  -H "Content-Type: application/json" \
  -d '{"is_dead":false}'
```

Delete a worker:

```bash
curl -s -X DELETE http://127.0.0.1:8008/workers/http%3A%2F%2F127.0.0.1%3A8013
```

Worker update requests are atomic. If an update returns `400`, the live worker
state is not partially changed.

## Routing Behavior

The router only selects workers that are healthy, not disabled, and capable of
serving the request.

The default worker capability set represents a complete Omni V1 replica:

- `chat`
- `speech`
- `streaming`
- `image_input`
- `audio_input`
- `video_input`
- `audio_output`

The router infers required capabilities from each request:

- `/v1/chat/completions` requires `chat`
- `stream: true` requires `streaming`
- `images`, `image`, or image message parts require `image_input`
- `audios`, `audio_inputs`, or audio message parts require `audio_input`
- `videos`, `video`, or video message parts require `video_input`
- `modalities: ["audio"]` or `audio` output fields require `audio_output`
- `/v1/audio/speech` requires `speech`, plus `streaming` for streamed speech

Register narrower worker capabilities only when a worker cannot serve one of
those request classes.

Large JSON requests are not fully parsed by the router. With a homogeneous pool
of complete Omni V1 replicas, no extra headers are needed. With mixed models,
provide a model hint. With mixed worker capabilities, provide a capability hint
when the router cannot infer a single safe worker set:

- `X-SGLang-Omni-Route-Model`: requested model for mixed-model pools
- `X-SGLang-Omni-Route-Capabilities`: comma-separated capabilities such as
  `image_input`, `audio_input`, `video_input`, `audio_output`, or `streaming`
- `X-SGLang-Omni-Route-Stream`: `true` or `false` for large streaming requests

These headers are router-only hints and are not forwarded to workers.

## Request Diagnostics

Routed responses include:

- `X-SGLang-Omni-Worker`: selected worker ID
- `X-SGLang-Omni-Request-ID`: request ID from the request headers or body, or a
  router-generated ID
- `X-SGLang-Omni-Route-Attempt`: currently `1`

Router logs include a route-completion record for buffered and streaming
requests. Each record contains the request ID, selected worker, path, stream
flag, inferred capabilities, status code, duration, and terminal outcome.

## Failure Handling

If a worker health check or routed request fails repeatedly, the worker becomes
unhealthy and leaves the routable pool. It can return to healthy after the
configured number of successful health checks.

To inspect failover behavior:

1. Stop one worker.
2. Call `GET /workers` and check its `consecutive_failures`, `health_state`, and
   `routable` fields.
3. Send another request through the router and verify that it uses a remaining
   routable worker.
4. Restart the stopped worker and wait for it to become healthy again.

For a source checkout without installed console scripts, verify the module entry
point with:

```bash
python -m sglang_omni_router.serve --help
```
