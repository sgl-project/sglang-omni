# TTS Serving Benchmark

This benchmark validates the serving behavior of SGLang Omni's
OpenAI-compatible TTS service. It is separate from SeedTTS quality evaluation:
SeedTTS measures model quality and speed on a fixed corpus, while this
benchmark exercises API contracts, malformed requests, stateful voice
operations, streaming behavior, and high-concurrency load.

Note: before all serving APIs are integrated, enabled-but-missing contracts are
expected to fail explicitly. After those APIs are integrated,
`examples/stress.json` should pass.

## Design

```text
spec.json
  -> spec parser
  -> serving-stress scenario builder
  -> load-stage scheduler
  -> HTTP/SDK/WebSocket/voice/batch clients
  -> response classifiers
  -> results.json + manifest.json + raw JSONL + logs
```

The benchmark does not launch the target service. It reads the target address
from `spec.json`, generates a deterministic scenario matrix, runs the configured
load stages, classifies every response, and writes structured artifacts.

The harness exit code only describes the benchmark runner:

- `0`: the harness completed normally and wrote artifacts. The service can
  still fail the benchmark through `overall.passed=false`.
- non-zero: the harness hit an infrastructure or runtime error, such as an
  invalid spec, artifact-write failure, or unhandled runner exception. When the
  runner can still write artifacts, `results.json` records
  `harness_status="error"` and `harness_error`.

Use both the process exit code and `results.json`: the exit code reports
whether the harness ran correctly, while `overall.passed` reports whether the
target service passed the configured serving contracts.

## Harness Contract

The Docker entry point uses fixed input and output paths:

```text
/etc/benchmark/spec.json   # input spec
/var/benchmark/out/        # output directory
```

The default entry point is:

```bash
python -m benchmarks.eval.benchmark_tts_serving \
  --spec /etc/benchmark/spec.json \
  --out /var/benchmark/out
```

Direct Python runs require `ffmpeg` on `PATH` for compressed audio validation.
The Docker image installs `ffmpeg`.

## File Structure

```text
benchmark_tts_serving.py       # entry point under benchmarks/eval/
spec.py                        # spec schema and load-stage defaults
scenarios.py                   # deterministic scenario matrix
http_client.py                 # HTTP request dispatch
sdk_client.py                  # OpenAI SDK compatibility path
ws_client.py                   # WebSocket speech-stream path
voice_client.py                # uploaded-voice lifecycle and cache pressure
batch_client.py                # batch speech response validation
audio_validation.py            # WAV/PCM and decoded compressed-audio checks
report.py                      # results.json aggregation and coverage contract
artifacts.py                   # manifest, raw JSONL, and harness log writing
examples/                      # runnable spec examples
Dockerfile                     # standalone benchmark image
```

## Spec Reference

Top-level fields:

| Field | Description |
|-------|-------------|
| `base_url` | Target service address, for example `http://127.0.0.1:8000`. |
| `model_name` | Model id copied into OpenAI-compatible requests. |
| `test_type` | Artifact label for benchmark systems. Valid values: `engine`, `e2e`, `external`. |
| `run_id` | Optional run identifier included in artifacts. |
| `seed` | Deterministic seed for scenario ordering and open-loop arrivals. Defaults to `0`. |
| `auth.api_key_env` | Optional environment variable name containing a bearer token. |
| `params` | Benchmark profile, endpoints, load stages, timeout, and voice limits. |

Common `params` fields:

| Field | Description |
|-------|-------------|
| `profile` | Serving-stress scenario mix. Valid value: `stress`. |
| `enabled_endpoints` | Endpoint families to exercise: `speech`, `speech_stream`, `voices`, `batch`, `websocket`. |
| `total_requests` | Request count used when `load_stages` is not provided. |
| `max_concurrency` | Concurrency used when `load_stages` is not provided. |
| `load_stages` | Explicit staged load plan. |
| `timeout_s` | Per-request timeout. |
| `speaker_max_uploaded` | Expected server-side uploaded-speaker cap. |
| `voice_cache_pressure_voice_count` | Number of unique uploaded voices for cache-pressure stages. |
| `voice_speaker_cap_count` | Upload-attempt budget for speaker-cap stages. |
| `file_ref_audio` | Optional file URI sent to the target service. Required for full speech reference coverage. The target service must be launched with an allowed-local-media path that contains this file. |
| `file_ref_text` | Optional transcript for `file_ref_audio`. Defaults to the SeedTTS reference text. |

Load-stage fields:

| Field | Description |
|-------|-------------|
| `id` | Stable stage id used in scenario ids and artifacts. |
| `mode` | `closed_loop`, `open_loop`, `ramp`, `burst`, or `soak`. |
| `request_count` | Number of scheduled scenarios for the stage. |
| `max_concurrency` | Maximum in-flight requests for the stage. |
| `request_rate` | Requests per second for `open_loop` and `ramp`. Do not set this for `soak` because it is derived from `request_count / duration_s`. |
| `start_request_rate` | Initial requests per second for `ramp`. |
| `duration_s` | Wall-clock duration for `soak`. |
| `arrival_distribution` | `deterministic` or `poisson`. |
| `enabled_endpoints` | Optional per-stage endpoint override. |

## Endpoint Contracts

`enabled_endpoints` controls which API families are part of the run:

| Endpoint family | Contract |
|-----------------|----------|
| `speech` | `POST /v1/audio/speech` with response formats, task types, language handling, speed bounds, reference audio, SDK compatibility, and malformed-request classification. |
| `speech_stream` | `POST /v1/audio/speech` with raw PCM streaming and streaming error cases. |
| `batch` | `POST /v1/audio/speech/batch` with 1-32 item batches, per-item overrides, item-level success/error records, and oversized batch rejection. |
| `voices` | `GET`, `POST`, and `DELETE /v1/audio/voices` with upload formats, metadata, overwrite, delete, named voice reuse through speech and batch synthesis, speaker-cap, cleanup, race, and cache-pressure behavior. |
| `websocket` | `/v1/audio/speech/stream` with session configuration, incremental text input, binary audio frames, event ordering, client disconnect, malformed JSON, and missing-config errors. |

Voice cache-pressure scenarios require `GET /v1/audio/voices` to expose a
`cache_stats` object with `entries`, `memory_bytes`, `max_bytes`,
`eviction_count`, `hit_count`, `miss_count`, and
`delete_invalidation_counter`. The cache-pressure sequence uploads unique
voices, synthesizes with each voice, revisits older voices, deletes the created
set, and requires the miss, hit, memory, and delete-invalidation counters to
move according to those operations. If the generated traffic reaches the
advertised cache budget, `eviction_count` must move as well.

## Response Body Contracts

Malformed HTTP requests must return a structured JSON error body:

```json
{
  "error": {
    "message": "human-readable error",
    "type": "BadRequestError",
    "param": "field_name_or_null",
    "code": 400
  }
}
```

Missing voice-management resources use the same shape with `type:
"NotFoundError"` and `code: 404`. The missing-voice
`DELETE /v1/audio/voices/{name}` contract is a voice-management response
instead of the generic error envelope:

```json
{
  "success": false,
  "error": {
    "message": "voice not found"
  }
}
```

The harness accepts `error` as either a non-empty object or a non-empty string
for this delete-specific response.

## Basic Usage

Start a compatible TTS service, then run the checked-in serving-stress spec:

```bash
python -m benchmarks.eval.benchmark_tts_serving \
  --spec benchmarks/tts_serving/examples/stress.json \
  --out results/tts_serving/stress
```

The checked-in spec targets a Higgs TTS service through the `base_url` and
`model_name` fields in `examples/stress.json`. It also sends a reference clip
from `docs/_static/audio`, so launch the target service from the repo root with
`--allowed-local-media-path docs/_static/audio`. Update the spec fields, and
`auth.api_key_env` when needed, for a different target.

## Docker

Build the standalone image:

```bash
docker build -f benchmarks/tts_serving/Dockerfile \
  -t sglang-omni-tts-serving-benchmark .
```

Run the image with a spec mounted at the contract path:

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/benchmarks/tts_serving/examples/stress.json:/etc/benchmark/spec.json:ro" \
  -v "$PWD/results/tts_serving/stress:/var/benchmark/out" \
  sglang-omni-tts-serving-benchmark
```

When the target service is outside the container network, set `base_url` to an
address reachable from the container. If `file_ref_audio` is configured, the
URI must point to a file visible to the target service, not to a file inside the
benchmark container.

## Scenario Matrix

The matrix is deterministic for a given spec seed. It covers:

- speech success paths across response formats, task types, speed boundaries,
  reference-audio shapes, allowed `file://` references, SDK calls, and raw
  speech streaming
- malformed speech requests that must return the structured error envelope
- multilingual and adversarial text payloads
- batch speech creation and item-level result validation
- uploaded-voice list, upload, overwrite, delete, metadata, named speech and
  batch reuse, speaker-cap, and upload/delete race contracts
- voice cache pressure traffic with observable cache counters
- WebSocket speech-stream setup, event order, audio events, and error cases

Voice scenarios are stateful. Successful standalone uploads verify uploaded
metadata with `GET /v1/audio/voices`, delete the created voice, and list again
to prove cleanup. Lifecycle delete also verifies that using the deleted voice
for synthesis returns a structured `400 BadRequestError`. Repeated runs should
not consume persistent speaker slots or change later baselines.

Compressed response formats are decoded through `ffmpeg` before validation.
The benchmark checks decoded PCM duration and non-zero signal so container
headers or placeholder bytes cannot pass as generated audio.

## Serving-Stress Profile

`examples/stress.json` is the serving-stress example. Its stages are:

| Stage | Purpose |
|-------|---------|
| `closed-1` | Serial baseline that isolates contract failures from concurrency effects. |
| `closed-16` | Moderate closed-loop concurrency. |
| `ramp-128` | Poisson ramp from low request rate to high request rate. |
| `soak-300s` | Sustained load over a fixed duration. |
| `ws-burst-512` | WebSocket-only burst pressure. |
| `voice-cache-pressure` | Uploaded-voice cache pressure below the speaker cap. |
| `voice-speaker-cap` | State-aware speaker-cap validation. |
| `mixed-burst-512` | Full-endpoint burst with `request_count=512` and `max_concurrency=512`. |

The mixed burst intentionally matches request count and concurrency so the
client can emit the full burst without creating artificial
`load_generator_saturated` records.

Speaker-cap stages list existing uploaded voices first, upload only the names
needed to reach `speaker_max_uploaded`, and require the first overflow upload
to fail. Set `speaker_max_uploaded` to the server-side uploaded-speaker cap.

Cache-pressure stages require observable counters for entries, memory usage,
hits, misses, delete invalidation, and eviction behavior when the cache reaches
capacity. Missing counters or counters that do not match the generated traffic
fail the serving contract.

## Results

The output directory contains:

```text
results.json          # summary, coverage contract, metrics, unsupported APIs
manifest.json         # parsed-spec hash, scenario-set hash, and artifact metadata
raw/*.jsonl           # per-scenario records
logs/harness.log      # load-stage execution notes
```

`results.json` includes:

| Field | Meaning |
|-------|---------|
| `overall.passed` | End-to-end benchmark pass/fail for the configured contracts. |
| `harness_status` | Whether the harness ran successfully. |
| `overall.coverage_contract_valid` | Whether required scenario coverage was achieved. |
| `overall.load_generation_valid` | Whether the client emitted the intended load without saturation or excessive lag. |
| `metrics.status_counts` | Count of `ok`, protocol failures, unsupported contracts, and load-generator failures. |
| `metrics.endpoint_mix` | Executed scenario count by endpoint family. |
| `unsupported_contracts` | Enabled API contracts that were missing or unsupported. |
| `coverage_failures` | Coverage requirements that traffic alone could not prove. |
