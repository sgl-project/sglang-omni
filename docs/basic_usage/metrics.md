# Prometheus metrics

Pass `--enable-metrics` when starting the server to enable collection and expose
Prometheus text at `GET /metrics`:

```bash
sgl-omni serve --model-path MODEL_ID --enable-metrics
curl http://localhost:8000/metrics
```

Metrics are disabled by default, matching SGLang server behavior.

## Inference metrics

Engine stages use SGLang's own metrics collector. SGLang-Omni enables that
collector for each engine stage and exposes its multiprocess output without
recomputing token counts or latency. Every SGLang metric includes a `stage`
label; replicated stages also include `replica`. Existing SGLang labels such as
`model_name`, `engine_type`, `tp_rank`, and `pp_rank` remain unchanged.

Metric names, buckets, and calculations follow the installed SGLang version.
See the [SGLang observability documentation](https://docs.sglang.ai/advanced_features/observability.html)
for the available inference metrics.

## Generated audio metrics

Audio output from `/v1/audio/speech` populates these metrics:

| Metric | Type | Extra label | Description |
| --- | --- | --- | --- |
| `sglang_omni:audio_ttfp_s` | Histogram | — | Time to first audio payload for streaming responses |
| `sglang_omni:audio_rtf` | Histogram | — | Engine generation time divided by output audio duration |
| `sglang_omni:audio_e2e_latency_s` | Histogram | — | HTTP request arrival to final audio response |
| `sglang_omni:audio_duration_s` | Histogram | — | Generated audio duration |
| `sglang_omni:audio_chunk_interval_s` | Histogram | — | Wall time between audio chunks |
| `sglang_omni:audio_underrun_s` | Histogram | — | Largest playback buffer underrun in a completed stream |
| `sglang_omni:audio_continuity_ok_total` | Counter | `threshold_ms` | Completed streams within each underrun threshold |

Non-streaming responses do not contribute TTFP, chunk interval, underrun, or
continuity samples. Interrupted streams can contribute TTFP and chunk intervals,
but do not contribute output duration, E2E latency, RTF, underrun, or continuity.

The endpoint also includes the standard Python process, platform, and garbage
collector metric families from `prometheus_client`.
