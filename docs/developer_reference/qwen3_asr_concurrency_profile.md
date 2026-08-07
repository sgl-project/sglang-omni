# Qwen3-ASR High-Concurrency Benchmark and Bottleneck Profile

This is the issue #1324 Q-PR2 deliverable: a reproducible current-main
benchmark of Qwen3-ASR serving at concurrency 1–64, compared against a fixed
serving baseline on identical pre-segmented audio, with enough internal
telemetry to attribute where request time goes. The numbers below are the
before-baseline for Q-PR3 (#1326, merged), Q-PR4, and Q-PR5.

## Method

- Model: `Qwen/Qwen3-ASR-1.7B`, bf16, one 80 GB GPU per server (DP=1).
- Data: full SeedTTS reference sets — EN 1088 clips, ZH 2020 clips — staged by
  `benchmarks.dataset.seedtts`; both systems receive the same files through
  the same client (`benchmarks/eval/benchmark_asr_seedtts.py`).
- Sweep: concurrency `1, 8, 16, 32, 64`, one discarded warmup pass plus three
  measured repeats per level; closed-loop client.
- Current is SGLang-Omni at the commit under review, served with the default
  Qwen3-ASR pipeline (`max_running_requests=32`,
  `request_build_max_workers=2`, `request_build_max_pending=16`, async decode
  on). Baseline is a fixed OpenAI-compatible serving stack for the same
  checkpoint with its stock configuration, on an identical GPU of the same
  host. Neither side was tuned.
- Telemetry: per-request profiler events (request build, admission queue,
  first forward, decode tail), host CPU and per-GPU utilization sampling, raw
  per-request JSONL, and an environment fingerprint are captured by the
  benchmark's `--profile-events --sample-util --fingerprint --save-raw-dir`
  flags added in this PR.

### Reproduce

```bash
# current
sgl-omni serve --model-path Qwen/Qwen3-ASR-1.7B --port 8511

python -m benchmarks.eval.benchmark_asr_seedtts \
  --port 8511 --concurrencies 1,8,16,32,64 --repeats 3 --warmup \
  --profile-events --profile-event-dir /tmp/asr_profile \
  --sample-util --util-gpu-ids <gpu> --fingerprint \
  --save-raw-dir raw-current --output asr_en_current.json

# baseline: point the same client at the baseline server's port
python -m benchmarks.eval.benchmark_asr_seedtts \
  --port <baseline-port> --concurrencies 1,8,16,32,64 --repeats 3 --warmup \
  --sample-util --util-gpu-ids <gpu> --fingerprint \
  --output asr_en_baseline.json
```

## Shared-host variance

The measurement host is shared; co-tenant load shifts between runs and shows
up directly in these host-bound workloads (see #907). Runs on a quiet host
and a loaded host are both reported: the absolute peaks move, but the
structural findings — the concurrency knee, its cause, and the stage
decomposition — are identical in every run. Current-vs-baseline comparisons
were taken back-to-back so both sides see similar external load.

## Results — current (quiet host)

SeedTTS EN, 1088 clips, mean over three repeats:

| conc | req/s | RTFx | lat mean | lat p95 | corpus WER | completed |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 11.6 | 54.8 | 0.088 s | 0.129 s | 0.0122 | 3264/3264 |
| 8 | 27.5 | 130.4 | 0.289 s | 0.437 s | 0.0122 | 3264/3264 |
| 16 | 53.0 | 251.2 | 0.304 s | 0.458 s | 0.0122 | 3264/3264 |
| 32 | 108.0 | 511.4 | 0.295 s | 0.399 s | 0.0122 | 3264/3264 |
| 64 | 103.0 | 486.8 | 0.610 s | 0.741 s | 0.0125 | 3165/3264 |

SeedTTS ZH, 2020 clips, mean over three repeats:

| conc | req/s | RTFx | lat mean | lat p95 | corpus CER | completed |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 14.3 | 66.9 | 0.070 s | 0.088 s | 0.0062 | 6060/6060 |
| 8 | 64.2 | 300.8 | 0.124 s | 0.175 s | 0.0064 | 6060/6060 |
| 16 | 96.5 | 451.9 | 0.165 s | 0.231 s | 0.0062 | 6060/6060 |
| 32 | 126.4 | 591.6 | 0.252 s | 0.353 s | 0.0062 | 6060/6060 |
| 64 | 129.2 | 604.3 | 0.489 s | 0.613 s | 0.0062 | 5993/6060 |

Reading:

- Throughput saturates at concurrency 32 and stops scaling to 64 on both
  languages while mean latency roughly doubles.
- At concurrency 64 both languages shed 1–4 % of requests with HTTP 500: a
  single worker admits at most `request_build_max_pending=16` request builds
  and `max_running_requests=32` running requests, so the 64-deep closed loop
  overflows the build backlog.
- WER/CER stay in band at every level (the concurrency-64 uptick is the
  shed-request denominator, not transcription regressions).
- Relative to the pre-#1326 reference (97.9 req/s at concurrency 32 on the
  same EN set and hardware class), async decode lifted saturation throughput
  by about 10 %.

## Results — current vs baseline (back-to-back, same host window)

SeedTTS EN, 1088 clips, mean over three repeats; the two sweeps ran
back-to-back on identical neighboring GPUs so both saw the same co-tenant
load. One baseline concurrency-1 repeat was hit by a co-tenant burst
(7.3 req/s against 22.6/21.4); the mean below includes it.

| conc | current req/s | baseline req/s | ratio | current lat mean/p95 | baseline lat mean/p95 | current completed | baseline completed |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12.0 | 17.1 | 1.42× | 0.083 / 0.102 s | 0.076 / 0.149 s | 3264/3264 | 3264/3264 |
| 8 | 41.9 | 70.3 | 1.68× | 0.191 / 0.256 s | 0.117 / 0.174 s | 3264/3264 | 3264/3264 |
| 16 | 62.6 | 120.6 | 1.93× | 0.255 / 0.346 s | 0.132 / 0.171 s | 3264/3264 | 3264/3264 |
| 32 | 79.5 | 183.8 | 2.31× | 0.401 / 0.549 s | 0.175 / 0.242 s | 3264/3264 | 3264/3264 |
| 64 | 80.9 | 264.6 | 3.27× | 0.777 / 0.947 s | 0.238 / 0.291 s | 3208/3264 | 3264/3264 |

Corpus WER stayed 0.0122–0.0125 (current) and 0.0123–0.0124 (baseline) at
every level. Reading:

- The gap is a scaling gap, not a single-request model-forward gap: baseline
  keeps scaling past current's concurrency-32 saturation and completes every
  request at 64 while current sheds 1–2 % there.
- Excluding the contended repeat, baseline is ~1.8× faster even at
  concurrency 1 (0.044 s vs 0.083 s mean), so part of the gap is fixed
  per-request serving overhead, and the rest grows with concurrency —
  consistent with the admission cap and host-dispatch costs quantified
  below.
- This reproduces the external report's direction with a wider magnitude on
  this hardware/window; the concurrency knee the roadmap describes is
  confirmed on current main.

SeedTTS ZH, 2020 clips, mean over three repeats, same back-to-back pairing
(current concurrency-1 repeats were partially contended in this window):

| conc | current req/s | baseline req/s | ratio | current lat mean/p95 | baseline lat mean/p95 | current completed | baseline completed |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.1 | 25.2 | 2.78× | 0.116 / 0.178 s | 0.039 / 0.053 s | 6060/6060 | 6060/6060 |
| 8 | 47.8 | 131.0 | 2.74× | 0.167 / 0.238 s | 0.061 / 0.083 s | 6060/6060 | 6060/6060 |
| 16 | 68.4 | 186.5 | 2.73× | 0.233 / 0.324 s | 0.085 / 0.117 s | 6060/6060 | 6060/6060 |
| 32 | 86.6 | 245.5 | 2.83× | 0.368 / 0.522 s | 0.129 / 0.164 s | 6060/6060 | 6060/6060 |
| 64 | 82.1 | 265.7 | 3.24× | 0.772 / 1.017 s | 0.238 / 0.339 s | 5961/6060 | 6060/6060 |

Corpus CER stayed 0.0061–0.0063 (current) and 0.0064–0.0066 (baseline) at
every level.

## Bottleneck decomposition (profiled passes, EN)

Per-request stage averages from the request-event profiler; encoder work runs
inside the first LM forward on this path, so "first forward" is encoder +
prefill:

| conc | build | build→queued | queued→scheduled | first forward | decode tail | total |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.6 ms | 0.7 ms | 1.1 ms | 28.0 ms | 55.3 ms | 91 ms |
| 8 | 6.0 ms | 16.5 ms | 2.0 ms | 29.5 ms | 131.8 ms | 196 ms |
| 16 | 6.1 ms | 20.0 ms | 1.9 ms | 30.1 ms | 185.6 ms | 259 ms |
| 32 | 6.5 ms | 23.1 ms | 5.9 ms | 46.2 ms | 357.7 ms | 472 ms |
| 64 | 6.3 ms | 30.1 ms | 412.8 ms | 33.3 ms | 392.6 ms | 910 ms |

GPU utilization stays between 33 % and 54 % (mean) at every level; the GPU is
never the constraint.

Dominant costs, in order:

1. **Admission queueing at concurrency 64.** The queued→scheduled wait jumps
   from ~6 ms to ~413 ms — the entire knee. Requests beyond
   `max_running_requests=32` wait; requests beyond the build backlog are
   rejected. This is the Q-PR5 target.
2. **Decode-step host dispatch.** The decode tail grows from 55 ms to
   ~360 ms per request as concurrency rises while GPU utilization falls —
   decode wall-clock is dominated by host-side per-step work shared across
   the running batch, consistent with the #907 causal profiling
   (GPU-idle-94 %, CPU-sensitivity-0.69). Q-PR5's knob sweep and any further
   host-side batching improvements attack this.
3. **First-forward stalls.** Encoder + prefill costs 28–46 ms per admitted
   request and executes on the scheduler thread and default stream, so every
   admission stalls the whole running decode batch for that long (and forces
   an async-decode drain). This is the Q-PR4 target: move the encoder ahead
   of LM admission onto its own thread/stream and batch it.
4. **Build-to-queued gap.** 16–30 ms at concurrency ≥ 8 from the FIFO
   build-result drain and admission-lock path; small but visible, also in
   Q-PR5 scope.

## Environment

Fingerprints are embedded in each result JSON produced by
`--fingerprint`: a client block (git SHA, dependency freeze hash, driver,
GPU, cached model revision — describing the benchmark process) and a server
block (what the target server reports about itself). In the runs above,
client and server shared one host and checkout, so the client git state also
identifies the server code. Raw per-request records and profiler event JSONL
live beside the result files on the measurement host.

Summary for the runs above: NVIDIA H100 80GB HBM3, driver 580.126.20,
torch 2.11.0+cu130, sglang 0.5.16, transformers 5.12.1, model snapshot
`7278e1e70fe206f11671096ffdd38061171dd6e5`, dependency freeze
`49652ce3ea5a7720…`. The current-side numbers were produced at this
branch's commit; note the fingerprint's `git.dirty` flag also counts
untracked run-output directories.
