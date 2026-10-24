# MiniCPM-o bounded local video decode reuse

## Scope and mechanism

Repeated questions about the same video already benefit from encoder-result caching, but timed-video preprocessing still opens the file, samples frames, extracts source audio, and constructs audio segments again. This change reuses decoded media within one offline preprocessing worker. It does not cache model responses or change sampling, media ordering, resizing, or model computation.

The local path is checked against the current media policy on every request. Eligible source files are read into a bounded byte snapshot; full SHA-256 plus all effective video-decoding options identify the entry. The decoder consumes those same bytes, avoiding a hash/decode mismatch if the path changes. Each caller receives independent frames, audio arrays and timestamp containers. Failures are not cached. Concurrent eligible requests share a producer; cancelling a request drains its executor operation without cancelling other users of that producer.

The default configuration retains at most eight entries and 256 MiB of accounted decoded payloads; source snapshots are limited to 32 MiB and four cache-managed operations. Admission overflow and oversized source files use the existing uncached loader. These limits do not bound total process RSS or all decoder concurrency: decoder scratch, returned copies, Python/Pillow allocations and uncached work need additional memory. HTTP, data and file URLs remain on the original connector path. Disable reuse with `--preprocessing.factory.video_decode_cache_max_entries 0`.

## Evidence and experiment boundaries

Baseline main: `f31cf9e9a`. Hardware: one H100 NVL on GPU0, twelve fixed allocated CPU cores, OMP_NUM_THREADS=4; torch 2.13.0+cu130, SGLang 0.5.21, transformers 5.12.1. Checkpoint: `openbmb/MiniCPM-o-4_5` at `503e754207c94da6bb26850b4469f367c9ea3582`.

Both A and B use the same bounded-audio-padding calculation investigated in PR #3007, installed by a process-local experiment adapter. Neither arm changes it between visits. Only the production video-decode-cache capacity differs: zero for A, eight for B. This isolates incremental decode reuse rather than claiming the padding gain again. Production source in this branch does not include #3007. The benchmark is consequently a scoped integration screen, not an unmodified-main-versus-candidate claim.

Inputs are three public Video_MME_ci clips: `qvwfWyU9Gfc.mp4`, `24i4ncHuf6A.mp4`, `N1cdUjctpG8.mp4`. Requests use ordered inline video, original video audio, `video_fps=1`, `video_max_frames=64`, `video_max_pixels=401408`, temperature zero, seed 42 and a 32-token output limit. One server at a time runs A1→B1→B2→A2 on the same GPU. Each visit performs three warmups, two first-use requests for other clips, two sequential three-video windows, a six-request c3 window, a different question about the first clip, and two single-audio protection requests: 17 measured requests per visit, 68 total. Warm encoder caches and repeated inputs are intentional; do not generalize this workload to unique-video traffic.

## Incremental service results

| Metric | A1 | B1 | B2 | A2 |
|---|---:|---:|---:|---:|
| c3 throughput, six requests (req/s) | 0.9470 | 3.0426 | 2.8169 | 0.9309 |
| Warm sequential clip 1 (s) | 1.002 | 0.335 | 0.348 | 0.987 |
| Warm sequential clip 2 (s) | 1.395 | 0.373 | 0.381 | 1.357 |
| Warm sequential clip 3 (s) | 1.497 | 0.360 | 0.395 | 1.483 |
| Different question, clip 1 (s) | 0.964 | 0.309 | 0.331 | 0.949 |
| Second single-audio request (s) | 0.261 | 0.245 | 0.248 | 0.260 |
| End GPU memory (MiB) | 62059 | 63069 | 62951 | 62951 |

Both B visits outperform both A visits for the repeated-video cells; mean c3 throughput is about 3.12 times baseline. All 68 measured requests succeeded. Six requests are too few for a production p95 guarantee; per-request samples are retained rather than presenting a small interpolated tail as statistical acceptance. End GPU memory is not peak GPU memory.

First-use clip latencies were A1 2.050/2.083 s, B1 1.727/2.046 s, B2 2.029/2.070 s, A2 2.046/2.097 s. No stable cold-path speedup is claimed. The miss path adds snapshot/hash/copy work, so unique-video protection must remain in scope.

An earlier same-server ABBA on the first clip measured A medians 1.023/1.014 s and B medians 0.356/0.356 s; B's first miss was 0.998 s and was not counted as a hit. Candidate-only production integration without a decode monkeypatch measured roughly 0.39–0.40 s for subsequent requests.

## Correctness and generated-output limitation

Direct cache miss/hit comparisons on all three real clips preserve every frame pixel, source-audio float sample, timestamp and duration exactly. Repeated uncached decoding of these clips was also exact. This verifies media equivalence, not arbitrary language-model response bit-exactness.

The default nondeterministic model configuration produced one differing concurrent text response in the final B2 window; token-use records still matched. Additional repeat control showed several formulations of the same first-clip description with cache enabled, while the slower uncached admissions happened to be stable. With the Thinker's supported deterministic inference enabled identically for A and B, 18 concurrent requests per arm across three repeat windows were stable and text-identical across arms. This supports a dynamic-batch numerical-path explanation, but the default-setting response variation remains a documented limitation, not erased by the deterministic check. No perceptual, ASR, or video-QA accuracy score is claimed from this small descriptive-video workload.

Tests cover content changes at the same path, identical bytes at different paths, decode-option isolation, mutation isolation, entry and byte eviction, oversized sources/outputs, source snapshot consistency, decode failure/retry, same-content concurrent production, producer/waiter cancellation, local-policy rejection despite a cached entry and real video frame parity. Related checkpoint prompt/video/config suites and changed-file pre-commit are executed; raw logs distinguish completed passes, formatter changes, timeouts and infrastructure failures.

## Memory and local component cost

Repeated uncached real-video decode medians were approximately 542/942/1098 ms, versus 3–6 ms for full-file SHA-256 alone. Actual production-cache hits, including source reading, hashing and independent output copies, measured about 15–38 ms. Hash time alone is not reported as hit latency.

Three cached clips retain 146,400,172 accounted bytes. In separate standalone decoder processes, cached peak RSS was 1,181,786,112 bytes and uncached peak RSS 1,092,399,104 bytes; final RSS was approximately 1.118 GB versus 0.944 GB. These runs include library/allocator retention and are not full-server peak-memory measurements. They establish a real host-memory tradeoff, not a zero-cost optimization. All inputs fit the proposed resident budget; workloads exceeding it should be tested for eviction and miss rates rather than assuming the same speedup.

## Reproduction

Use the pinned checkpoint, project dependencies including decord, and the three named clips. Launch both arms from the same source and GPU/CPU environment, applying the identical audio-padding setup to both arms if validating the incremental result above. Use the same server arguments:

```bash
python -m sglang_omni.cli serve \
  --model-path "$CHECKPOINT" --host 127.0.0.1 --port 18129 \
  --thinker.factory.max_seq_len 32768 \
  --thinker.engine.mem_fraction_static 0.55 \
  --talker.engine.mem_fraction_static 0.15
```

For A append `--preprocessing.factory.video_decode_cache_max_entries 0`; for B use the configured capacity. Keep every other argument identical. Send `/v1/chat/completions` with an inline video URL containing a bare local path, `use_audio_in_video=true`, the sampling/pixel options above, and a fixed description question. Warm separately before measuring. For the concurrent cell send each of the three clips twice using three client workers and divide six completions by total cell wall time. Record first-use results separately, and retain every visit rather than selecting the best. For exact concurrent text repeatability apply the same `--thinker.engine.enable_deterministic_inference true` to both arms.

```bash
MINICPMO_CHECKPOINT="$CHECKPOINT" python -m pytest \
  tests/unit_test/minicpm_o/test_preprocessor_video.py \
  tests/unit_test/minicpm_o/test_preprocessor_prompt.py \
  tests/unit_test/minicpm_o/test_native_config.py -q
pre-commit run --all-files
```

Large raw responses, video bytes and machine-specific scripts remain in the local gitignored `.profiling-runs/minicpm-audio-graphs/` directory. The previous exact audio-encoder graph prototype was rejected after reverse-order service measurements showed only about 2.66% mean E2E improvement; no production graph code is included here.
