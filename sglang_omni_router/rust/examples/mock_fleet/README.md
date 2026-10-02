# Model-Free Router Fleet

Run independently validated mock workers behind the actual Rust router. No
Python packages, models, GPUs, downloads of model assets, or external services
are needed. The mock fleet is a Cargo example, not part of the production binary.

## Run

From `sglang_omni_router/rust`:

```bash
cargo build --locked --bin sgl-omni-router --example mock_fleet
target/debug/examples/mock_fleet \
  --config examples/mock_fleet/fleet.toml \
  --requests 8 --concurrency 4
```

The supplied fleet starts six worker processes: two text replicas, vision,
omni, TTS/voice/speech-WebSocket, and ASR/translation. Omni also serves realtime
WebSockets. Text, vision, and omni advertise a shared model ID so requests
exercise modality filtering independently of model-name selection.

Workers bind ephemeral loopback ports and announce their actual addresses.
The launcher generates `router.toml`, runs the actual router's `--check-config`,
starts it, and waits for readiness. It never connects to configured external
worker URLs; generated loopback URLs replace those values.

The router listener is explicit: edit `server.listen` if port 30000 is occupied.
Existing listeners are not terminated. The harness refuses non-loopback binds.

`run` is the default mode. It exits nonzero on failed checks, timeout or startup
failure and stops/reaps its child processes. `--deadline-secs` bounds the run
(default 300 seconds); individual client requests have 15-second timeouts.
Each process uses two Tokio runtime threads. SIGINT/SIGTERM clean up the fleet;
cleanup terminates the children and is not a test of graceful router drain.

## Manual Mode

```bash
target/debug/examples/mock_fleet \
  --config examples/mock_fleet/fleet.toml --mode serve
```

Use the router URL printed at startup:

```bash
curl http://127.0.0.1:30000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"shared","messages":[{"role":"user","content":"hello"}]}'
```

Responses include `x-mock-worker`, `x-mock-service`, `x-mock-modalities`, and an
upstream request-ID echo. JSON and WebSocket setup/events include mock identity
metadata. These headers originate at the mock, not at the router; they do not
implement the RFC's missing router-generated worker diagnostic header.

## Configure

The fleet file uses the real router TOML schema, plus an optional `mock` table
inside each worker. `base_url` may be omitted because the launcher supplies it.
Add workers by adding `[[workers]]` entries with distinct IDs. Keep each supported
combination in one correlated `[[workers.service_profiles]]` row.

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

Unknown mock fields and excessive limits are rejected. Generated audio is
24 kHz, mono, signed 16-bit silent PCM, optionally in a valid WAV container.
The mock does not encode MP3, Opus, AAC or FLAC; do not advertise these formats.
Image/video inputs are classified structurally, not decoded. ASR consumes real
multipart framing but returns deterministic text rather than transcribing.
Speech defaults to WAV in this mock; manual clients should specify a supported
`response_format`. Speech WebSocket setup uses `stream_audio`, not HTTP `stream`.

The bounded voice registry is in memory and is lost when the worker exits.
Batch responses are deterministic ordered synthetic results, not a complete
model-server implementation. Setup/config and audio-flow checks are not a full
realtime conversation or TTS application-protocol conformance suite.

Direct worker URLs in `endpoints.json` expose local test-only controls:

- `GET /__mock/stats`: identity, profiles, accepted/rejected counts, service counts.
- `POST /__mock/control` with `{"health_status":503}`: fail health probes.
- `POST /__mock/control` with `{"response_status":503}`: inject upstream failures.
- Set either status back to `200` to recover.

Controls are unauthenticated and workers must remain loopback-only. Faults in
configuration intentionally make positive checks fail; failures are not hidden.

## Results And Coverage

Artifacts are written to a unique directory under `target/`, or a new directory
specified by `--output`. Existing output directories are never overwritten.
Artifacts include the generated manifest, endpoints, config-check log, worker
logs, router log, and `report.json`.

`--requests` is the number of requests **per generated HTTP workload per path**,
not a total across the fleet. Direct-worker and routed measurements use the same
eligible worker pool. Concurrency is bounded; results are grouped by request
variant and include successful QPS, byte rate, worker distribution, P50/P95/P99
latency and client-observed first-body-byte latency. There is no warmup phase or
CPU affinity. Direct workers use round robin; routed traffic uses the configured
policy. Failure samples are retained separately from successful latency metrics.

Automated checks currently cover:

- Model/modality/format/stream selection and response identity.
- JSON, SSE completion, synthetic PCM/WAV and multipart transcription/translation.
- Direct and routed unsupported-model rejection, and an unsupported image case
  when the configuration contains a suitable text-only model.
- Request-ID propagation upstream and return downstream.
- Speech/realtime WebSocket setup, audio and stable session identity.
- Voice CRUD and named-voice HTTP synthesis on the configured owner.
- Complete generation `413` with no upstream dispatch, by exceeding both
  streamed and buffered limits when both are at most 2 MiB (the supplied example
  uses 64 KiB). Larger limits are explicitly reported as skipped.
- Upstream `503` propagation, all-workers-unhealthy rejection without dispatch,
  and readiness recovery.
- Realtime session saturation returning `429`, for configured limits of 1..=32.
- Router metrics/diagnostics and worker counters captured in the report.

Workload generation honors enabled routes, correlated profiles, route trust
domains and batch-size limits. Profiles outside the route's trust domain or for
disabled services are not included in either the direct or routed workloads.
Realtime audio byte counts are decoded payload bytes, not base64 character counts.
WebSocket PCM length and WAV headers/lengths are checked against the worker config.

The report lists skipped, untested and unsupported features. In particular this is not
proof of raw HTTP framing correctness, the known rejected-media connection-reset
regression, global/HTTP-class overload bounds, multipart zero-copy behavior,
graceful drain, or every profile/task/reference combination. The existing Rust
socket-level tests remain authoritative for those cases. Custom manifests may
not enable every scenario; inspect the report's actual check list.

Client first-body-byte time includes mock delay and is not first audible audio
or router-internal TTFP. Direct-worker baselines help expose mock/client limits,
but worker CPU saturation is not instrumented. Synthetic results do not prove
real-model quality, RTF, Python/Rust per-core speedup or DP3 x MPS performance.

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