# Metrics

## Backend and modality extensions

Backend collector passthrough preserves all `sglang:` metric families already
instantiated by the enabled SGLang scheduler collector, including conditional
histograms/counters. It does not instantiate or reimplement unsupported engine
features. HTTP-owned names (core request metrics and fresh engine gauges) are
filtered to prevent duplicate families. Existing backend labels are preserved;
`omni_stage` identifies the source pipeline stage without overwriting SRT's
own `stage` labels. Set the engine stage's `enable_metrics` ServerArgs override
when these backend collectors are required; the HTTP switch alone does not
enable backend feature collection. This bridge does not include arbitrary
process-global collectors or device synchronization.

ThreadedSimpleScheduler snapshots count actual running futures, including
aborted work that has not stopped, and non-aborted queued requests.
StreamingSimpleScheduler snapshots count new-request queue entries and work
being processed by its new-request/chunk handlers; a stream waiting for more
input is not called GPU compute. dLLM snapshots count unfinished admitted
staging requests and waiting requests. Their special compute/ready-step events
still require separate instrumentation; gauges do not prove batch latency.


OmniScheduler reports fresh KV-pool snapshots for `sglang:token_usage`,
`sglang:full_token_usage`, `sglang:num_used_tokens`,
`sglang:kv_available_tokens`, `sglang:kv_evictable_tokens`, and
`sglang:max_total_num_tokens`; SWA/Mamba usage is present only for those pools.
`sglang:gen_throughput` comes from the backend reporter's decode window with
its stale-window handling, not completed-request counts divided by E2E.
`sglang:cache_hit_rate` requires backend metrics to be enabled and uses backend
stats, not an inferred HTTP cache ratio. Missing backend capabilities produce
no gauge sample. Engine gauges carry `model_name` and `stage`; follower values
are not added, and removed series are cleared on successful scrapes.

OmniScheduler emits batch sizes at synchronous/asynchronous forward launch.
Its queue sample is first wait-queue entry to first instrumented launch; it is
not a sum of later retraction waits. SimpleScheduler records each compute
queue wait. These boundaries are intentionally documented separately.

`sglang_omni:stage_request_duration_seconds` covers dispatch through Stage
routing/cleanup, including scheduler waiting, with completed/error/aborted
outcomes. It applies independently of scheduler compute instrumentation and
is not the upstream SRT enum-based per-stage/device timer.

`sglang_omni:audio_output_seconds_total` sums output waveform sample lengths
at the reported sample rate on successfully completed requests, before speed
conversion/encoding. Encoded bytes or missing sample rates are not guessed.
`sglang_omni:generated_images_total` counts the final reported image list (or
single image), not diffusion steps; it requires an Images-capable result
builder. Aborted/failed requests do not add successful modality output.


Install the optional dependency with `pip install -e '.[metrics]'` and pass
`--enable-metrics` (or `--enable_metrics`) to `sgl-omni serve`. Metrics are
disabled by default. Scrape `GET /metrics` on the service port. The endpoint
has no API-key requirement; restrict access through deployment networking.

## SGLang-compatible core metrics

| Metric | Type | Unit | Labels |
| --- | --- | --- | --- |
| `sglang:e2e_request_latency_seconds` | Histogram | seconds | model_name, is_streaming |
| `sglang:time_to_first_token_seconds` | Histogram | seconds | model_name, is_streaming |
| `sglang:inter_token_latency_seconds` | Histogram | seconds | model_name |
| `sglang:prompt_tokens_total` | Counter | tokens | model_name, is_streaming |
| `sglang:generation_tokens_total` | Counter | tokens | model_name, is_streaming |
| `sglang:num_running_reqs` | Gauge | requests | model_name, stage |
| `sglang:num_queue_reqs` | Gauge | requests | model_name, stage |

Latency buckets follow the reference SGLang checkout (commit
`e310eaeaf766c207533b920f459d3d57263c5d12`). Names and types are compatible;
the multi-stage measurement boundary is necessarily Omni-specific.

- E2E starts at Client generation entry, before request construction, and ends
  when the final pipeline output reaches the Client. It includes preprocessing
  and downstream stages, but not HTTP admission before Client entry or final
  HTTP response encoding/transmission. It is not a single-stage engine timer.
- TTFT and ITL use dedicated host-side AR token events after SGLang updates
  `output_ids`. The Coordinator observes them for streaming and non-streaming
  requests alike; SSE framing and empty terminal chunks are not observations.
  IPC delivery is included, as in a tokenizer-side observation. No device
  synchronization or per-token device timestamp is added.
- The first AR stage to report tokens owns the request's token observations.
  Later stages are not added to it. These are raw backend token counts, not
  estimates derived from text length, image count or diffusion steps.
- For an interval producing N new tokens, ITL records N samples of interval/N.
  Duplicate cumulative counts do not add samples; a decrease resets the
  baseline without recording a negative increment.
- Token counters are recorded once when generation completes. If AR events
  exist, use their final count; otherwise use available final text usage.
  Missing usage produces no token sample. Pipeline exceptions and cancelled
  streams do not publish successful final usage. Explicit backend aborts are
  counted as completed/aborted requests with the available partial token count.
- Models without AR token telemetry, including image diffusion, have E2E
  observations but no fabricated TTFT/ITL observations.
- Each supported stage reports its actual scheduler state through the existing
  admin control plane. SimpleScheduler includes serial, batched and concurrent
  work; OmniScheduler includes active prefill/decode requests and its waiting
  queue. Preprocessing/admission backlog outside that queue is not folded into
  the SGLang queue gauge. Unsupported schedulers produce no synthetic zero.
- TP follower counts are not summed. Stage replicas have distinct stage labels.
  A request may exist in multiple stages: summing stage gauges is stage work,
  not a deduplicated service request count. Select the model execution stage
  when reusing a dashboard designed for a single SGLang engine.

## Export lifecycle and additional HTTP metrics

SimpleScheduler also sends host-side queue/compute events over the existing
control plane when metrics are enabled. `sglang:queue_time_seconds` measures
enqueue to compute dispatch, including batch assembly and concurrency waiting;
its buckets match SGLang's defaults, with an additional `stage` label.
`sglang_omni:scheduler_batch_size` records one sample per compute dispatch, not
per image or request result. `sglang_omni:stage_compute_duration_seconds` records
host compute lifetime with `model_name`, `stage`, and `outcome` labels
(`completed`, `error`, `aborted`). It is not synchronized device time. These
events currently cover SimpleScheduler only; other scheduler lifecycles must
be instrumented separately. Cancelled queue entries emit no compute sample.

Token-length metrics are `sglang:prompt_tokens_histogram`,
`sglang:generation_tokens_histogram`, and
`sglang:uncached_prompt_tokens_histogram`. They use SGLang's default
token-length buckets and `model_name`, recording one sample per completed
request, not per cumulative snapshot. Missing counts produce no sample.
Uncached length requires an explicitly reported cached count, including zero;
missing cache data is not treated as zero.

`sglang:cached_tokens_total` uses `model_name` and `cache_source="total"`,
matching SGLang's fallback without a source breakdown. Only positive reported
counts are added. Zero is represented by the uncached-length sample. Counts
outside `[0, prompt_tokens]` are omitted. Cache details must belong to the same
text stage as native token observations. Current usage provides no
device/host/storage breakdown, so none is inferred. Images without text usage,
failures and cancelled streams publish no successful token-length/cache data.

One HTTP process owns one app-local registry; generation workers report over
the control plane. Repeated app creation does not register global collectors.
There is no multiprocess file directory to clean. `PROMETHEUS_MULTIPROC_DIR`
is rejected before model loading because it changes Prometheus client storage;
unset it before launch. Multiple HTTP workers sharing one pipeline are not
supported by this exporter. Do not merge independently exported SRT metrics
into this registry: that would duplicate names/counts.

Scrapes refresh scheduler gauges. A timeout or failed stage returns HTTP 503,
not a successful response carrying stale values. Removed stages disappear on
the next successful scrape. Token observation state is removed on success,
failure and cancellation; late events are ignored.

Additional counters are `sglang:num_requests_total`,
`sglang:num_aborted_requests_total` and `sglang_omni:failed_requests_total`.
HTTP transport metrics are separate: `sglang_omni:http_requests_total`,
`sglang_omni:http_request_duration_seconds`, and
`sglang_omni:http_requests_inflight`. They cover validation errors and streams
through the final response body/disconnect. An HTTP 200 SSE may still contain
a model error; HTTP status is not model success. Routes use templates (or
`unmatched`), never request IDs, prompts or arbitrary paths. Health, favicon,
metrics scrapes and WebSockets are excluded.

## Current scope and deferred capabilities

The current scope is the seven core metrics, request/HTTP reliability, queue
and host Stage lifetimes, plus applicable AR token/KV/cache and modality usage.
The SGLang metric catalogue is a reference, not a requirement to implement
every producer. Conditional passthrough is not feature support or acceptance.

Backend snapshots also read the attached prefix cache's existing Radix
`metrics_collector`, when backend metrics are enabled. Shared metric objects
are deduplicated. Explicit L3 `storage_metrics_collector` traversal and native
speculative-verification/grammar request counters are deferred until the
corresponding engine capabilities exist. The generic backend bridge can still
expose existing native families; it does not enable those capabilities.

Vocoder ready-step-specific compute/batch instrumentation is deferred; its
generic Stage lifecycle remains available. Streaming handler gauges do not
claim to measure every autonomous vocoder ready-step. Startup, independent
encoder/dispatcher and SRT-specific PD telemetry require separate source and
boundary analysis. Omni PD stages exist, but their queues/transfers cannot be
assumed to have SRT bootstrap/preallocation semantics.

## Aggregation

AVG/P95/P99 are queries over Histogram samples, not separate exporter gauges.
For example, a model's ITL average over five minutes is:

```promql
sum by (model_name) (rate(sglang:inter_token_latency_seconds_sum[5m]))
/
sum by (model_name) (rate(sglang:inter_token_latency_seconds_count[5m]))
```

P95 is `histogram_quantile(0.95, sum by (le, model_name)
(rate(sglang:inter_token_latency_seconds_bucket[5m])))`; use 0.99 for P99.
Token counters are lifetime totals; use `increase(...[5m])` for a period count.
Histogram percentiles are bucket-based estimates. No observations means no
valid latency estimate, not zero latency.

GPU/NPU latency overhead, actual TP/replica deployment and monitoring-platform
label compatibility require hardware/deployment acceptance. SenseNova Images
acceptance covers E2E and scheduler gauges, not text TTFT/ITL correctness.
Advanced feature-specific metrics are not a current acceptance gate. Existing
conditional backend export does not imply every family has been validated.
