<!-- Thank you for your contribution! We appreciate it. The following guidelines will help improve your pull request and facilitate feedback. If anything is unclear, don't hesitate to submit your pull request and ask the maintainers for assistance. -->

## Motivation

Whisper request construction previously ran on one scheduler thread, so concurrently arriving requests were often admitted as separate batch-1 prefills. This underutilized the serving-reachable encoder batch-2 CUDA Graph added in #1412 and made CPU-side audio preprocessing a throughput bottleneck.

This PR enables bounded parallel request construction and pending-build-aware prefill coalescing for Whisper. The goal is to form batch-2 prefills under concurrent traffic without adding a fixed wait to the single-request path or changing transcription accuracy, request lifecycle behavior, tensor ownership, or the atomic encoder-prefix admission contract.

## Modifications

- Build up to two Whisper requests concurrently, with at most 16 submitted request-build futures. When `max_queued_requests` is unset, additional requests remain queued instead of encountering an implicit request-build backlog ceiling.
- Coalesce Whisper prefill at two waiting requests or a 6 ms deadline, but only while another request build is pending. A single request or a partial batch with no remaining build work is admitted immediately.
- Keep coalescing disabled after request builds drain during active decode, avoiding an unnecessary deadline wait at low concurrency.
- Serialize the mutable Whisper tokenizer prefix and prompt-token construction region so concurrent requests cannot mix language, task, or previous-context state. Audio loading and feature extraction remain outside the lock and execute concurrently.
- Expose Whisper through the shared `--prefill-coalesce-requests` and `--prefill-coalesce-wait-ms` CLI overrides.
- Add pipeline-default, scheduler-forwarding, CLI-override, and concurrent-tokenizer ownership tests.
- Document the default policy, opt-out configuration, reproducible benchmark command, accuracy results, and before/after performance.

## Related Issues

Part of https://github.com/sgl-project/sglang-omni/issues/1396 (Whisper ASR W-PR2: prefill coalescing to reduce small-batch prefill).

## Accuracy Test

Tested `openai/whisper-base` in FP16 on the 20-sample SeedTTS English subset. Each configuration ran one discarded warmup and five measured repeats at concurrency 1, 2, 4, and 8.

| Configuration | Measured requests | Successful requests | Corpus WER at c=1/2/4/8 |
|---|---:|---:|---:|
| One request-build worker, coalescing disabled | 400 | 400 | 0.0415 / 0.0415 / 0.0415 / 0.0415 |
| Two request-build workers, coalescing disabled | 400 | 400 | 0.0415 / 0.0415 / 0.0415 / 0.0415 |
| Two request-build workers, coalescing enabled | 400 | 400 | 0.0415 / 0.0415 / 0.0415 / 0.0415 |

All 1,200 measured requests completed successfully, and corpus WER was unchanged across configurations and concurrency levels. Two additional concurrent multipart requests using `tests/data/query_to_cars.wav` and `tests/data/query_to_draw.wav` returned HTTP 200 with the expected transcripts.

Focused verification:

```text
71 passed
```

This includes the complete Whisper unit-test directory plus the shared prefill coalescing gate, validation, and CLI tests. The concurrent tokenizer test also verifies that English and French requests retain their own prefix tokens while the shared mutable tokenizer region has a maximum concurrency of one.

## Benchmark & Profiling

Environment: one NVIDIA H200, `openai/whisper-base`, FP16, 20 SeedTTS English samples, one discarded warmup and five measured repeats per concurrency.

```bash
python -m benchmarks.eval.benchmark_asr_seedtts \
  --port 8000 --model-path openai/whisper-base \
  --max-samples 20 --concurrencies 1,2,4,8 \
  --repeats 5 --warmup --disable-resource-monitor \
  --output whisper_wpr2_h200.json
```

The baseline used `request_build_max_workers: 1` and `prefill_coalesce_requests: 0`. The attribution configuration used two request-build workers with coalescing disabled. The optimized configuration used two request-build workers, `prefill_coalesce_requests: 2`, a 6 ms wait, `prefill_coalesce_when_idle: true`, `prefill_coalesce_requires_pending_builds: true`, and `prefill_coalesce_after_builds_during_decode: false`.

| Concurrency | Baseline req/s | Two workers req/s | Coalesced req/s | Total gain | Coalescing gain | Baseline mean latency (s) | Coalesced mean latency (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 21.04 | 22.51 | 22.46 | +6.8% | -0.3% | 0.047 | 0.044 |
| 2 | 30.45 | 36.68 | 41.96 | +37.8% | +14.4% | 0.066 | 0.047 |
| 4 | 40.24 | 55.62 | 62.83 | +56.2% | +13.0% | 0.097 | 0.063 |
| 8 | 48.03 | 75.93 | 82.15 | +71.0% | +8.2% | 0.161 | 0.092 |

The c=1 difference between the two-worker and coalesced configurations is within measurement noise, while coalescing adds 8.2% to 14.4% throughput at c=2 through c=8 on top of the request-build parallelism gain.

Runtime evidence confirms that the intended path executed:

```text
request_build_workers=2
request_build_max_pending=16
request_build_max_pending_observed=2
Replaying Whisper encoder CUDA graph batch=2 request_batch=2
Prefill batch, #new-seq: 2, #new-token: 3008
```

## Checklist

- [x] Format your code according with pre-commit.
- [x] Add unit tests.
- [x] Update documentation / docstrings / example tutorials as needed.
- [x] Provide throughput / latency benchmark results and accuracy evaluation results as needed.
- [ ] For reviewers: If you haven't made any contributions to this PR and are only assisting with merging the main branch, please remove yourself as a co-author when merging the PR.

## CI

CI runs on self-hosted GPU runners and requires a maintainer to add the
`run-ci` label. Once labeled, every subsequent push re-triggers CI as
long as the label remains. Use `/tag-and-rerun-ci higgs` or
`/tag-and-rerun-ci moss` to select a TTS CI model, and
`/tag-and-rerun-ci fun-asr` or `/tag-and-rerun-ci qwen3-asr` to select an ASR
CI model. One selector from each family can be combined, for example
`/tag-and-rerun-ci moss fun-asr`. Draft PRs are skipped even if labeled.
