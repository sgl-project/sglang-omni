# Model-Free Router Fleet

CPU-only mock worker processes behind the real [Rust router](../../../../docs/basic_usage/omni_router.md).
This Cargo example needs no models, Python packages, GPUs or external services
and changes no production router behavior.

## Run

From `sglang_omni_router/rust`:

```bash
cargo build --locked --bin sgl-omni-router --example mock_fleet
target/debug/examples/mock_fleet \
  --config examples/mock_fleet/fleet.toml \
  --requests 8 --concurrency 4

# Keep the fleet running for manual clients:
target/debug/examples/mock_fleet \
  --config examples/mock_fleet/fleet.toml --mode serve
```

The sample has six workers: two text replicas, vision, omni (including realtime),
TTS/voice/speech-WebSocket, and ASR/translation. The `shared` model alias tests
modality filtering independently of model selection. For manual requests, use
the printed router URL:

```bash
curl http://127.0.0.1:30000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"shared","messages":[{"role":"user","content":"hello"}]}'
```

Workers use ephemeral loopback ports, replacing any configured `base_url`.
The launcher generates `router.toml`, runs the router's `--check-config` and waits
for readiness. Change `server.listen` if port 30000 is busy; existing listeners
are never terminated. All listeners must be loopback-only.

`run` exits nonzero on failed checks, startup failure or timeout. The runner's
`--deadline-secs` defaults to 300; individual requests time out after 15 seconds.
Each process uses two Tokio threads. Exit, SIGINT and SIGTERM stop/reap owned
children; this is not a graceful-drain test.

## Configure

Use the router TOML schema with 1-64 distinct `[[workers]]` entries and optional
`[workers.mock]` settings. Omit `base_url`; the launcher supplies it. Keep supported
combinations in correlated `[[workers.service_profiles]]` rows, as in
[`fleet.toml`](fleet.toml). Workers validate requests independently of the router.

```toml
[workers.mock]
first_packet_delay_ms = 5
chunk_interval_ms = 2
chunks = 4
chunk_bytes = 4096
health_status = 200
response_status = 200
max_request_bytes = 8388608
# disconnect_after_chunks = 1
```

Unknown fields and excessive limits are rejected. Audio is silent 24 kHz mono
16-bit PCM or WAV; do not advertise MP3/Opus/AAC/FLAC. Speech defaults to WAV;
specify a supported `response_format`. Speech WebSocket setup uses `stream_audio`,
not HTTP `stream`. Image/video inputs are not decoded; ASR and batch results are
synthetic. Uploaded voices are bounded, in-memory worker-local state.

Direct worker URLs expose **unauthenticated, local-only** controls:

| Endpoint | Purpose |
| --- | --- |
| `GET /__mock/stats` | Identity, profiles and accepted/rejected/service counts |
| `POST /__mock/control` | `{"health_status":503}` or `{"response_status":503}` injects failures; set to `200` to recover |

Faults intentionally fail positive checks. Responses carry `x-mock-worker`,
`x-mock-service`, `x-mock-modalities` and a request-ID echo; JSON/WS events also
carry mock identity. These do not implement router-generated diagnostic headers.

## Results and limits

Artifacts go to a unique `target/` directory, or a new `--output` directory:
generated config, `endpoints.json`, process/config-check logs and `report.json`.
Existing output directories are never overwritten.

`--requests` applies **per generated HTTP workload per path**, direct and routed.
Both paths use the same eligible pool, honoring enabled routes, trust domains,
correlated profiles and batch limits. Direct dispatch is round robin; routed
dispatch uses the configured policy. The report includes successful QPS/byte rate,
worker distribution, P50/P95/P99 latency and first-body-byte latency, with failures
recorded separately. There is no warmup, CPU affinity or worker CPU instrumentation.

Checks cover capability rejection, identity/request IDs, JSON/SSE/audio/multipart,
WebSocket setup and audio lengths, voice ownership, upstream `503`, health recovery,
generation `413` without dispatch (both limits must be at most 2 MiB), and realtime
`429` saturation (explicit limit of 1-32). Router metrics and worker stats are saved.

Consult the report's actual checks and `skipped`, `not_tested` and
`unsupported_router_features` lists for custom fleets. This is not full protocol
conformance or evidence for raw HTTP framing, rejected-media connection resets,
multipart zero-copy, global/HTTP-class overload or active graceful drain; the
router's socket-level tests cover those paths.

First-body-byte latency includes mock delay: it is not first audible audio or
router-internal TTFP. Synthetic throughput proves neither model quality/RTF nor
Python/Rust per-core speedup or DP3 x MPS performance.

Use release builds for performance experiments:

```bash
cargo build --release --locked --bin sgl-omni-router --example mock_fleet
target/release/examples/mock_fleet \
  --router-bin target/release/sgl-omni-router \
  --config examples/mock_fleet/fleet.toml --requests 100 --concurrency 16
```

## Tests

```bash
cargo test --example mock_fleet --locked
cargo clippy --example mock_fleet --locked -- -D warnings
```
