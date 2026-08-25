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
  num_gpus_per_worker: 1
  worker_host: 127.0.0.1
  worker_base_port: 8011
  worker_extra_args: "--config examples/configs/qwen3_omni_colocated_h20.yaml --colocate"
  wait_timeout: 600
```

`backend: local` means the router process starts and manages worker
subprocesses on the same machine. The launched workers are complete Omni V1
servers started with `sgl-omni serve`; they are not partial
pipeline stages. The router waits for every managed worker to pass `/health`
before it starts accepting client traffic, and it stops those managed workers
when the router exits.

`num_gpus_per_worker` controls automatic GPU grouping. The default Qwen3-Omni
router example uses colocated workers: each complete speech worker runs on one
GPU through `examples/configs/qwen3_omni_colocated_h20.yaml`. With
`num_workers: 2` and `num_gpus_per_worker: 1`, the launcher assigns GPU `0` to
the first worker and GPU `1` to the second worker when two CUDA devices are
visible.

Use `examples/configs/qwen3_omni_colocated_h200.yaml` instead for single-H200
workers.

Set `worker_gpu_ids` only when you need explicit placement. Each entry maps one
`CUDA_VISIBLE_DEVICES` value to one worker, for example
`worker_gpu_ids: ["0", "1"]` for two one-GPU colocated Qwen3-Omni workers. Use
`worker_extra_args: "--text-only"` only if you intentionally want text-output
workers instead of speech-output workers.

Use `worker_extra_args` for public Omni V1 serve options that are specific to
the worker process, such as `--mem-fraction-static`, `--thinker.tp_size`, or
`--text-only`. These arguments are passed to `sgl-omni serve`
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

For short audio-input / text-output MMSU-style workloads, use the fused
text-path Qwen3-Omni config instead of the default speech-colocated worker:

```yaml
launcher:
  backend: local
  model_path: Qwen/Qwen3-Omni-30B-A3B-Instruct
  model_name: qwen3-omni
  num_workers: 2
  num_gpus_per_worker: 1
  worker_extra_args: "--config examples/configs/qwen3_omni_mmsu.yaml --text-only"
```

This keeps preprocessing, encoders, aggregation, thinker, and decode in one
worker process while leaving the general speech-colocated topology unchanged.

## Launch Worker Servers Manually

Start each Omni V1 worker separately. The example below launches two colocated
Qwen3-Omni speech workers on different GPUs and ports:

```bash
CUDA_VISIBLE_DEVICES=0 sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --model-name qwen3-omni \
  --config examples/configs/qwen3_omni_colocated_h20.yaml \
  --colocate \
  --host 0.0.0.0 \
  --port 8011
```

```bash
CUDA_VISIBLE_DEVICES=1 sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --model-name qwen3-omni \
  --config examples/configs/qwen3_omni_colocated_h20.yaml \
  --colocate \
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
| `--max-connections` | auto: `128 x workers`, capped at `4096` | Admission bound: maximum concurrent in-flight model requests before the router fast-rejects with `503`. The upstream connection pool is sized to at least this value. Explicit values below `64 x workers` log an under-feed warning. |
| `--max-inflight` | equal to `--max-connections` | Advanced override that decouples the admission bound from `--max-connections`. The upstream pool is sized to the larger of the two. |
| `--health-failure-threshold` | `3` | Consecutive failed health checks or routed request failures before a worker becomes unhealthy. |
| `--health-success-threshold` | `2` | Consecutive successful health checks before an unhealthy or unknown worker becomes healthy. |
| `--health-check-timeout-secs` | `5` | Timeout for one worker health-check request. |
| `--health-check-interval-secs` | `10` | Interval between background worker health checks. |
| `--health-check-endpoint` | `/health` | Worker endpoint used by background health checks. |
| `--voice-owner-worker-url` | first worker with `speech` and `audio_input` | Single-process mode only. Worker that owns uploaded TTS voices. Voice management and synthesis requests that use uploaded voices stay on this worker. Without such an owner, built-in voices can still route to eligible speech workers, but voice management is unavailable. |
| `--router-state-dir` | `$SGLANG_OMNI_ROUTER_STATE_DIR`, else `$XDG_STATE_HOME/sglang-omni-router`, else `~/.local/state/sglang-omni-router` | Directory for [durable router state](#durable-router-state) (the weight-update journal), which must survive a host reboot. Mount it on a persistent volume in a container. |
| `--log-level` | `info` | Router and Uvicorn log level. |
| `--strict-limits` | off | Fail startup instead of warning when the `nofile` soft limit is too low for the resolved upstream pool size (`max(--max-connections, --max-inflight)`). |
| `--router-processes` | `1` | Number of data-plane relay processes. `1` keeps the single-process router below; `N >= 2` enables the [multi-process router](#multi-process-router-controldata-plane-split) (x86-64 Linux only, and rejected at startup together with an explicit `--policy least_request`). |
| `--shutdown-drain-secs` | `--request-timeout-secs` | Multi-process only: how long a stopping data-plane process waits for in-flight requests before cancelling them. The default matches the request timeout, so a routine shutdown never truncates a request its own timeout would have allowed. |

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
- `GET /health`: worker-pool health summary plus admission stats (`inflight`,
  `max_inflight`, `peak_inflight`, `rejected_total`). This returns `503` when no
  worker is routable.
- `GET /workers`: detailed worker state, including `health_state`, `disabled`,
  `routable`, `active_requests`, aggregate and per-service-class request
  counters, and last error.
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
- `/v1/audio/speech` and `/v1/audio/speech/batch` require `speech`;
  `/v1/audio/speech` also requires `streaming` when `stream: true` (batch speech
  does not support streaming)
- speech requests using `ref_audio` or audio-bearing `references` also require
  `audio_input`
- `/v1/audio/speech/stream` WebSocket sessions require `speech` and `streaming`,
  plus `audio_input` when configured with reference audio
- `/v1/audio/voices` management and synthesis using an uploaded voice require
  the owner worker, which has both `speech` and `audio_input`
- `/v1/audio/transcriptions` and `/v1/audio/translations` require
  `audio_input` (also for `streaming` when the `stream` form field is true.) are multipart uploads, so the router reads the `model` and `stream` form
  fields with a single linear pass that skips the uploaded file to avoid cpu
  overhead, so in a pool that mixes ASR models the request lands on a worker
  registered with that model name. When the router cannot read a field it
  falls back to `X-SGLang-Omni-Route-Model` and `X-SGLang-Omni-Route-Stream`;
  when a field and its header are both present they must agree or the router
  answers `400`. Translation support is per model, and a worker that does not
  support it answers `400`.

Register narrower worker capabilities only when a worker cannot serve one of
those request classes.

The complete TTS route set described below is currently supported by the
single-process router (`--router-processes 1`). Multi-process router data
planes retain the route surface documented in
[Multi-Process Router](#multi-process-router-controldata-plane-split).

Uploaded TTS voices are mutable worker-local state. The Router sends voice
list, upload, and delete operations to `--voice-owner-worker-url`, and pins
speech, batch, and WebSocket requests that name an uploaded voice to the same
worker. It returns `503` if that owner is unavailable instead of silently
routing the request to a worker without the voice. Once the uploaded-voice
registry is available, built-in voices remain balanced by the configured
routing policy. By default, the first routable worker with both `speech` and
`audio_input` capabilities becomes the owner and remains the owner for that
router process. Configure
`--voice-owner-worker-url` when the owner must remain stable across router
restarts. Pools without such an owner can still route built-in TTS to eligible
speech workers, but voice management returns `503`.
The owner cannot be removed or have either required capability removed through
the router's worker API while the router is running.

Voice ownership assumes one router is the writer for the pool, clients perform
voice mutations through that router, and the owner keeps its speaker directory
across restarts. Multiple independent routers or direct mutations against a
worker are not coordinated and are unsupported for uploaded voices.
The router loads its uploaded-voice registry in the background after voice
traffic begins and reconciles it after voice mutations. Until the initial load
succeeds, or while a mutation outcome is unresolved, every non-default voice
name is pinned to the owner because the router cannot safely distinguish a
built-in name from an uploaded name. `GET /health` reports the selected owner,
owner routability, registry state, and uploaded-voice count under
`voice_routing`.
The router hydrates this registry through the compact
`GET /v1/audio/voices?names_only=true` response rather than transferring stored
reference metadata.

Large JSON requests are not fully parsed by the router. With a homogeneous pool
of complete Omni V1 replicas, no extra headers are needed. With mixed models,
provide a model hint. With mixed worker capabilities, provide a capability hint
when the router cannot infer a single safe worker set:

- `X-SGLang-Omni-Route-Model`: requested model for mixed-model pools
- `X-SGLang-Omni-Route-Capabilities`: comma-separated capabilities such as
  `image_input`, `audio_input`, `video_input`, `audio_output`, or `streaming`
- `X-SGLang-Omni-Route-Stream`: `true` or `false` for large streaming requests

Speech and speech-batch JSON bodies larger than 1 MiB are conservatively pinned
to the voice owner because the router cannot fully inspect them to rule out an
uploaded voice reference. Without an eligible owner, the router conservatively
requires `audio_input`; it never weakens capability selection at the size
boundary. Route-hint headers do not override voice ownership.

These headers are router-only hints and are not forwarded to workers.

## Request Diagnostics

Routed HTTP responses include:

- `X-SGLang-Omni-Worker`: selected worker ID
- `X-SGLang-Omni-Request-ID`: request ID from the request headers or body, or a
  router-generated ID
- `X-SGLang-Omni-Route-Attempt`: currently `1`

HTTP routes emit `route_completed` after worker selection and `route_rejected`
before selection. Completion records contain the request ID, selected worker,
path, stream flag, inferred capabilities, status code, duration, and terminal
outcome.

WebSocket sessions cannot add HTTP response headers after the upgrade. Router
rejections are sent as a TTS `error` event followed by a protocol-appropriate
close code. Every accepted WebSocket emits one `tts_websocket_completed` record,
including rejections before worker selection; those records use `worker=-`.

## Overload Behavior

The router bounds its concurrent work. Once `--max-connections` in-flight model
requests are being relayed, additional model requests are rejected immediately,
before the request body is read:

- HTTP requests receive status `503`, an OpenAI-style error envelope
  (`"type": "overloaded_error"`), a `Retry-After: 1` header, and a
  `route_rejected` log with `reason=router_overloaded`
- WebSocket requests receive a TTS `error` event with
  `error_type=overloaded_error`, close code `1013`, and a terminal log with
  `outcome=router_overloaded`

Health and management endpoints (`/live`, `/ready`, `/health`, `/workers`, admin
routes) are never gated. `GET /health` reports the current in-flight level, the
peak since startup, and the total rejected count.

Sizing guidance:

- The auto default (`128 x workers`) is a divergence backstop, not a latency
  target. For large responses on a single-core router, an oversized bound
  degrades service itself; size `--max-connections` toward
  `capacity x acceptable latency` for your payload shape.
- Each in-flight request holds two file descriptors (client plus upstream). The
  router warns at startup when the `nofile` soft limit is below
  `2 x upstream pool size + headroom`, where the pool size is
  `max(--max-connections, --max-inflight)`. Raise the limit, or lower whichever
  of the two flags binds the pool (the warning names it); `--strict-limits`
  turns the warning into a startup error.
- A rejected request costs the client its keep-alive connection (the router
  responds before reading the body), so clients should back off on `503` rather
  than immediately retrying on a fresh connection.

## Failure Handling

Worker liveness is owned by the background `/health` probes. A relayed request
only marks a worker unhealthy when the router cannot get a usable response from
it: a transport-level failure (connection error or read timeout, with no HTTP
response) or a gateway status the worker returns, `502 Bad Gateway` or `504
Gateway Timeout`. Capacity backpressure and application statuses the worker
answers with itself, `429 Too Many Requests`, `503 Service Unavailable`, `408
Request Timeout`, and `500 Internal Server Error`, are counted as per-request
failures in the worker statistics but never evict a reachable worker, so one
overloaded worker or a stream of bad-input requests cannot cascade the pool into
unavailability. A worker that leaves the pool can return to healthy after the
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

## Durable Router State

`--router-state-dir` holds the control-plane state that must outlive both a
router restart and a reboot of the router host. Today that is the weight-update
journal: when an update is interrupted after some workers were already updated,
the journal keeps the affected workers disabled until an operator verifies
their weight versions and re-enables them
(`PUT /workers/{worker_id} {"disabled": false}`). Workers run on their own
hosts and outlive the router, so losing the journal would let a fresh router
re-enable a pool whose weights no longer match.

Resolution order:

1. `--router-state-dir`
2. `SGLANG_OMNI_ROUTER_STATE_DIR`
3. `$XDG_STATE_HOME/sglang-omni-router`
4. `~/.local/state/sglang-omni-router`

The directory is created owner-only (`0700`, journal file `0600`) and is keyed
by the router's `host:port`, so several routers on one machine keep separate
journals. If the directory cannot be created or written, the failure mode
depends on the mode: with `--router-processes >= 2` startup fails; the
single-process default still starts, logs a warning, and refuses weight
updates (`503`) until `--router-state-dir` points at a writable persistent
path. In neither mode is there a temp-directory fallback, which would look
durable while a reboot silently dropped the record.

If the journal itself becomes unreadable, re-enabling cannot resolve it and
every weight update stays blocked with `409`. Verify the pool's weight
versions, then drop the record explicitly:

```bash
curl -X POST http://127.0.0.1:8000/weight_update_journal/resolve \
  -H "Authorization: Bearer $SGLANG_OMNI_ADMIN_KEY" \
  -d '{"acknowledge": true}'
```

The call is admin-authenticated, requires `acknowledge` so the record cannot be
dropped by accident, reports the worker ids it dropped, and fails with `503` if
the file could not be removed. The affected workers stay disabled until they
are re-enabled one by one.

In a container, mount it on a persistent volume:

```bash
docker run -v /srv/sglang-omni-router:/var/lib/sglang-omni-router ... \
  python -m sglang_omni_router.serve \
    --router-state-dir /var/lib/sglang-omni-router \
    --worker-urls http://127.0.0.1:8011
```

What this directory does **not** manage:

- Per-run runtime files (serialized config, internal socket, worker snapshot,
  admission shared memory) stay in a temporary per-run workdir. They are
  rebuilt at startup and are meant to disappear with the process tree.
- Logs. The router writes to stdout/stderr; persist them with your container
  runtime, systemd, or log collector.

## Multi-Process Router (Control/Data-Plane Split)

> x86-64 Linux only: the shared admission seqlock relies on x86-64 store
> ordering, and any other machine is refused at startup. Enabled with
> `--router-processes N`; the default `1` keeps the
> single-process router described above. The one behavioral addition at
> `N = 1` is the weight-update journal: recovery at startup can keep workers
> from an earlier interrupted update disabled until an operator re-enables
> them.

At `N >= 2` the router runs as a small process tree:

- A **supervisor** binds the public port once and passes the listening socket
  to `N` **data-plane (DP)** processes, which accept from the shared queue and
  relay the model routes (`/generate`, `/v1/chat/completions`,
  `/v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/audio/translations`).
- One **control plane (CP)** owns the worker registry, health checks, and the
  admin surface. DPs learn the routable-worker set from a snapshot file the CP
  republishes on every state change and on a fixed keepalive cadence. Admin
  requests sent to the public port are forwarded to the CP, which enforces the
  admin key; `/v1/models`, `/live`, and `/ready` are answered by the DP
  itself, `/health` is the CP's aggregate view.
- The supervisor restarts crashed children (with a fail-closed budget for
  rapid crash loops), fences replaced processes by generation, and tears the
  tree down in order on SIGTERM. A stopping DP drains in-flight requests for
  up to `--shutdown-drain-secs` (default: `--request-timeout-secs`) before
  remaining tasks are cancelled, and the supervisor waits out that drain
  before escalating to SIGKILL.

Behavior differences to know about:

- **The admission bound is shared and soft.** The in-flight bound spans all
  DPs through a shared-memory counter array. Concurrent admission checks can
  overshoot the bound by at most `N - 1` requests. In `/health`, `inflight` is
  an instantaneous sum (not a linearizable value) and `peak_inflight` is
  best-effort. `rejected_total` is exact while every live slot stays readable;
  a DP killed mid-write loses the counters it had not yet published, so the
  total is best-effort across a crash and reclaim window.
- **`least_request` requires a single process.** It reads per-process
  counters, so `--router-processes >= 2` combined with an explicit
  `--policy least_request` fails at startup; use `round_robin` (DPs stagger
  their starting offsets to avoid herding) or `random`.
- **CP unavailability degrades before it fails.** DPs keep serving from their
  last snapshot for the stale timeout, then shed new requests with fast `503`s
  and flip `/ready`; one republish restores service. `/health` reports
  `degraded` while some DPs are missing and `503` only when nothing can serve.
- **Worker weight updates wait for the data planes.** Before broadcasting, the
  CP publishes the disabled-worker snapshot and waits until every live DP has
  acknowledged it; on timeout the update fails closed and nothing is sent.
- **Counters are aggregated.** `/workers` aggregate totals and
  `*_requests_by_class` maps are summed across DPs and stay monotonic across DP
  restarts; `active_requests` is a best-effort gauge over live DPs. Counters
  reset with the CP, matching single-process restarts.
- **File descriptors scale with `N`.** Each DP holds a full-size upstream pool
  (a skewed keep-alive client mix can pin the whole bound onto one DP, which
  must still carry it) with keep-alives split `pool / N`; the startup log
  prints the per-process and cluster fd budget for the nofile check.

CPU affinity guidance for dense deployments: pin the `N` DP processes to `N`
dedicated cores on one NUMA node; the CP and supervisor are near-idle and can
share one core; keep model workers and benchmark clients on other cores (or
the other NUMA node) so relay throughput is not contended.

## Troubleshooting

Check the Router and worker pool before restarting any process — see
[Check Router and Worker State](#check-router-and-worker-state) for the endpoint
commands and their meanings.

Use the response and health signals to choose the next step:

| Signal | Meaning | Action |
|---|---|---|
| `/live` fails | The Router process is not reachable. | Check the process, bind address, port, and startup logs. |
| `/live` is `200`, `/ready` is `503` | No worker is routable. | Inspect the worker states in `/workers`. |
| Model request: `413`, `message=payload too large` | The request body exceeds `--max-payload-size`. | Reduce the request size or, after checking memory and concurrency impact, raise the configured limit. |
| Large JSON model request: `400` requiring a route-hint header | JSON bodies larger than 1 MiB are not fully parsed, so the Router cannot infer the model or capabilities needed to select from a mixed worker pool. | Set the header named in the error message, such as `x-sglang-omni-route-model` or `x-sglang-omni-route-capabilities`; see [Routing Behavior](#routing-behavior). |
| Model request: `400`, `message` names a route-hint header | A route-hint header is malformed or disagrees with the JSON body: an empty value, an unsupported capability, or a value that conflicts with the body's `model` or `stream`. | Read the message; it names the offending header. The rejection log records the same message in `reason=`, with spaces replaced by underscores. |
| Model request: `503`, `type=overloaded_error`, `Retry-After: 1` | Router admission is full. | Back off and inspect `inflight`, `max_inflight`, and `rejected_total` in `/health`. |
| Model request: `503`, `message=no eligible upstream` | No routable worker matches the request. | Check worker routability, model, and capabilities. |
| Model request: `503` with `X-SGLang-Omni-Worker` | The selected worker returned `503`. | Inspect that worker's logs and `/health` endpoint. |
| Streaming response starts with `200` but ends early | The upstream stream failed after the HTTP status was sent. SSE responses end with an `upstream stream failed before completion` event whose body carries `"code": 502`; non-SSE streaming bodies truncate without an error frame. | Check the route-completion log for `outcome=stream_error` (a client disconnect logs `stream_cancelled` instead), then inspect the selected worker. |

For admission limits, rejection logs, and capacity guidance, see
[Overload Behavior](#overload-behavior).

Selection failures contain `reason=no_eligible_upstream` plus the inferred model
and capabilities. They occur before a worker is chosen and therefore have no
`X-SGLang-Omni-Worker` header.

A worker-returned `503` does not by itself evict the worker.

### Distinguish `502` Responses

For model requests that select a single worker, a `502` with
`X-SGLang-Omni-Worker` can come from either the Router or the selected worker.
Use the body to distinguish them:

- `{"error": {"message": "upstream request failed"}}`: the Router selected a
  worker, but a connection error or timeout prevented it from obtaining a
  response. Check the selected worker process, its port, and its `/health`
  endpoint.
- Any other body: the selected worker returned its own `502`, which the Router
  relayed. Investigate that worker's upstream dependencies.

This distinction does not apply to `/v1/models` or administrative broadcast
routes. Those routes may return a Router-generated `502` without selecting a
single worker and therefore without `X-SGLang-Omni-Worker`.

For how transport failures and worker-returned `502` or `504` responses affect
worker health, see [Failure Handling](#failure-handling).

### Inspect Worker State

Print the fields used to determine routability. `worker_id` is the
percent-encoded identifier the admin routes expect; `display_id` is the
host and port shown in logs:

```bash
curl -s http://127.0.0.1:8008/workers | python3 -c '
import json, sys
for worker in json.load(sys.stdin)["workers"]:
    print(
        "worker_id=" + worker["worker_id"],
        "display_id=" + worker["display_id"],
        "state=" + worker["health_state"],
        "disabled=" + str(worker["disabled"]),
        "routable=" + str(worker["routable"]),
        "failures=" + str(worker["consecutive_failures"]),
        "last_error=" + str(worker["last_error"]),
    )
'
```

| State | Meaning and action |
|---|---|
| `unknown` | The worker has not passed `--health-success-threshold` checks. Confirm its `/health` endpoint is reachable and wait for startup checks. |
| `unhealthy` | Failures reached `--health-failure-threshold`. Inspect `last_status_code`, `last_error`, and worker logs. Successful checks restore it automatically. |
| `dead` | The worker was manually quarantined and health probes skip it. Resolve the problem before clearing `is_dead`; the Router checks health before routing to it again. |
| `disabled: true` | The worker is administratively excluded even if healthy. Re-enable it only when it is ready for new requests. |

Also verify that `--health-check-endpoint` matches the endpoint exposed by the
worker.

For `no eligible upstream`, compare the request with each routable worker's
`model` and `capabilities` in `/workers`; see
[Routing Behavior](#routing-behavior) for the capability mapping and route-hint
headers.

Model filtering applies only when at least one candidate worker advertises a
`model`. Do not clear model metadata to work around a mismatch: if no candidate
advertises a model, the Router stops filtering by model.

### Drain and Remove a Worker

Disable the worker first so it receives no new requests, wait for
`active_requests` to reach zero, and then delete it:

```bash
(  # subshell: a failed step exits the procedure, not your shell
worker_id='http%3A%2F%2F127.0.0.1%3A8013'
max_wait_secs=1800
deadline=$((SECONDS + max_wait_secs))

curl -fsS -X PUT "http://127.0.0.1:8008/workers/${worker_id}" \
  -H "Content-Type: application/json" \
  -d '{"disabled":true}' ||
  exit 1

while true; do
  active_requests=$(
    curl -fsS http://127.0.0.1:8008/workers |
      python3 -c '
import json, sys
target = sys.argv[1]
workers = json.load(sys.stdin)["workers"]
worker = next((item for item in workers if item["worker_id"] == target), None)
print(worker["active_requests"] if worker else "missing")
' "${worker_id}"
  ) || exit 1
  case "${active_requests}" in
    0) break ;;
    missing)
      echo "worker ${worker_id} is not registered; check /workers" >&2
      exit 1
      ;;
  esac
  if (( SECONDS >= deadline )); then
    echo "timed out waiting for ${worker_id} to drain" >&2
    exit 1
  fi
  sleep 2
done

curl -fsS -X DELETE "http://127.0.0.1:8008/workers/${worker_id}"
)
```

Copy `worker_id` from `/workers` rather than constructing it manually; the
snippet in [Inspect Worker State](#inspect-worker-state) prints it in a form you
can paste directly. Raise `max_wait_secs` for workloads whose responses can
legitimately exceed the default 30-minute drain window.

Deleting a worker removes only its Router registration. Stop the worker process
separately after the drain completes.
