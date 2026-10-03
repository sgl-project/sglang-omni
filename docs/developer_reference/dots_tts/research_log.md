# dots.tts Low-Fruit Optimization Research Log

Last updated: 2026-08-11.

This log records bounded, math-preserving serving optimizations found after the
larger dots.tts CUDA Graph, batching, and KV-cache work had landed. The local
microbenchmarks isolate the changed operator path; they are not end-to-end RTF
or throughput claims.

## Source of Truth

- Repository: [sgl-project/sglang-omni](https://github.com/sgl-project/sglang-omni)
- Audit base: `main` at `1bb0f15b`
- Performance roadmap: [#1367](https://github.com/sgl-project/sglang-omni/issues/1367)
- Model operators: [`dots.tts==0.2.1`](https://pypi.org/project/dots.tts/0.2.1/)
- Checkpoints: [`dots.tts-mf`](https://huggingface.co/dots-studio/dots.tts-mf)
  and [`dots.tts-soar`](https://huggingface.co/dots-studio/dots.tts-soar)

| Optimization | Pull request | Head commit | Status |
| --- | --- | --- | --- |
| Reuse one batched latent denormalization | [#1438](https://github.com/sgl-project/sglang-omni/pull/1438) | `76d0816d` | Merged |
| Stack feedback directly into the CUDA Graph buffer | [#1439](https://github.com/sgl-project/sglang-omni/pull/1439) | `eede6786` | Merged |
| Bypass redundant vocoder staging | [#1440](https://github.com/sgl-project/sglang-omni/pull/1440) | `25ae3d95` | Merged |
| Reuse the CFG null-projection bias | [#1441](https://github.com/sgl-project/sglang-omni/pull/1441) | `5f530dc5` | Merged |
| Batch compatible streaming AudioVAE steps | [#1444](https://github.com/sgl-project/sglang-omni/pull/1444) | `7f5529ad` | Merged |

No dataset, checkpoint, adapter, or model artifact was produced. The PR
branches and this log are the complete reusable state; raw benchmark logs are
transient validation output rather than source-of-truth artifacts.

## Experiment Contract

- Keep RNG, tensor shapes, request ordering, solver parameters, and model
  outputs unchanged.
- Prefer removing redundant allocations, copies, or mathematically constant
  work over introducing a new kernel or execution mode.
- Validate the owning unit-test file and run an RTX A6000 CUDA microbenchmark
  with PyTorch `2.11.0+cu130`, SGLang `0.5.16`, and `dots.tts==0.2.1`.
- Report isolated path latency only; require a full serving benchmark before
  claiming an end-to-end latency or throughput change.

## Results and Decisions

| Hypothesis | Change | Isolated result | Correctness evidence | Decision |
| --- | --- | --- | --- | --- |
| Batched MeanFlow denormalizes the same latent repeatedly | Reuse one batch result for semantic-encoder input and per-request views | At batch 16, 17 calls became 1; 558.794 to 77.494 us/step (-86.13%) | `test_flow_head.py`: 9 passed; batch-call regression | Keep |
| Graph feedback staging allocates and copies an intermediate tensor | Use `torch.stack(..., out=persistent_buffer)` | Batch 1-16: about 49-58% lower staging latency | `test_model_runner.py`: 9 passed; output-storage regression | Keep |
| Vocoder always pads singleton and equal-length buckets | Direct singleton input, one `cat` for equal lengths, retain padding for mixed lengths | Batch 1/4/8/16 assembly: 39.951/85.686/187.155/357.004 to 1.082/14.653/17.626/18.554 us | Vocoder tests: 14 passed; singleton storage regressions | Keep |
| SOAR/base computes `Linear(zeros)` every token | Expand the linear bias for the CFG null branch | Actual 1536-to-1024 projection: 118.875 to 74.577 us/append (-37.26%) | Bitwise CUDA parity; `test_flow_head.py`: 10 passed | Keep |
| Binary EOS softmax could become a logit-difference comparison | No code change | Would change rounding near a user-visible threshold | Static review | Drop |
| Tail index and mask tensors could be cached more aggressively | No code change | Frequent batch 8/16 shapes are already CUDA Graph captured; expected win is small outside fallback shapes | Static review | Drop |
| Incremental acoustic-tail KV allocation could reduce memory | No code change | Material design and lifecycle work, outside the low-fruit scope | Roadmap review | Defer |

## Failure Notes

- A local macOS test environment had mismatched Torch/Torchaudio packages and
  could not collect model-level tests. Validation moved to an isolated venv in
  the existing Aries canonical container; the shared container environment was
  not modified.
- One foreground SSH test lost its connection. The retry ran detached inside
  the same container and persisted its exit code; all nine model-runner tests
  passed.

## Reproduction Checks

Run the owning tests for each PR:

```bash
python -m pytest -q tests/unit_test/dots_tts/test_flow_head.py
python -m pytest -q tests/unit_test/dots_tts/test_model_runner.py
python -m pytest -q \
  tests/unit_test/dots_tts/test_vocoder.py \
  tests/unit_test/dots_tts/test_vocoder_streaming.py
```

Before quoting serving-level gains, run the canonical Seed-TTS benchmark from
the [dots.tts cookbook](../../cookbook/dots_tts.md#performance) on the same GPU
for both revisions.

## 2026-08-11 Streaming SeedTTS Concurrency Sweep

- Hypothesis: the merged cross-request AudioVAE batching in #1444 should retain
  its c=8 streaming gain on the full SeedTTS EN set and reveal the saturation
  point across c=1,2,4,8,16,32.
- Contract: [contract.json](runs/2026-08-11-streaming-seedtts-sweep/contract.json)
- Allocation: [rollout-plan.json](runs/2026-08-11-streaming-seedtts-sweep/rollout-plan.json)
- Execution: hyper00 and hyper01 H200 GPUs selected only when free; c=1 uses the
  first 50 samples, all other concurrency levels use all 1,088 English samples.
- Failure: the first detached coordinators exited before model setup because
  the bind-mounted checkout tripped Git's `safe.directory` ownership check.
  The launcher now scopes `safe.directory` to its three read-only Git commands;
  no benchmark request ran in the failed attempt.
- Failure: the reused hyper01 image lacked the declared `jiwer` dependency, so
  its first workers exited during module import. The launch script now installs
  `jiwer` before starting workers; no request ran in that attempt.
- Failure: the previous reusable containers used an older runtime that lacked
  `dots_tts` and `msgpack`. Both canonical containers were recreated with the
  current H200 profile image, preserving their persistent `/data` mount. The
  launcher now pins `dots.tts`, `jiwer`, and `openai-whisper`; no request ran in
  the incompatible-runtime attempts.
- Allocation correction: use only high-numbered GPUs: hyper00 GPUs 7,6,5,4 for
  c=1,2,4,8 and hyper01 GPUs 7,6 for c=16,32.
- Pause: the first valid sweep was stopped on request after c=1 had completed
  both synthesis and ASR. The retained c=1 artifacts contain 50/50 successful
  requests and 50/50 evaluated WER samples; the other concurrency runs were
  incomplete and are not used as results.
- Resume allocation:
  [resume-rollout-plan.json](runs/2026-08-11-streaming-seedtts-sweep/resume-rollout-plan.json).
  It uses only GPUs reported free immediately before launch, ordered from high
  to low: hyper00 7,6,5 for c=2,4,8 and hyper01 7,6 for c=16,32. Each host uses
  a new task-scoped timestamp container exposing only its selected GPUs.
- Canonical summary:
  [results-summary.json](runs/2026-08-11-streaming-seedtts-sweep/results-summary.json).
  Raw per-request logs remain transient artifacts on the persistent host paths
  recorded there; no reusable dataset or model artifact was produced.

| Concurrency | Samples | Success | Request QPS | Audio s/s | Mean latency (s) | Mean TTFP (s) | Mean inter-chunk (s) | Corpus WER |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 50 | 50/50 | 0.589 | 2.327 | 1.698 | 0.355 | 0.1829 | 1.241% |
| 2 | 1,088 | 1,088/1,088 | 0.904 | 3.768 | 2.212 | 0.4405 | 0.2324 | 1.281% |
| 4 | 1,088 | 1,088/1,088 | 1.342 | 5.596 | 2.978 | 0.6114 | 0.3104 | 1.231% |
| 8 | 1,088 | 1,088/1,088 | 1.479 | 6.171 | 5.406 | 2.3405 | 0.4018 | 1.264% |
| 16 | 1,088 | 1,088/1,088 | 1.489 | 6.210 | 10.719 | 7.6436 | 0.4034 | 1.231% |
| 32 | 1,088 | 1,088/1,088 | 1.443 | 6.017 | 22.038 | 18.7227 | 0.4344 | 1.315% |

- Conclusion: throughput saturates at c=8-c=16. C=16 improves audio
  throughput by only 0.63% over c=8 while mean latency nearly doubles and mean
  TTFP increases from 2.34 s to 7.64 s. C=32 reduces throughput by 3.11% versus
  c=16 while latency and TTFP worsen sharply. Use c=8 as the default
  latency/throughput tradeoff; c=16 is only useful when maximizing aggregate
  throughput regardless of per-request latency.
- Cleanup: after both hosts exited with code 0 and all summary/WER files were
  revalidated on persistent storage, the two timestamp task containers and
  their host map entries were removed. The raw results remain intact under the
  paths recorded in `results-summary.json`.
- Status: `COMPLETE`.

## 2026-08-11 Non-Streaming SeedTTS Concurrency Sweep

- Hypothesis: running the same current dots.tts deployment without `--stream`
  should establish whether the high streaming RTF values came from streaming
  semantics or from the current serving stack more generally.
- Contract:
  [contract.json](runs/2026-08-11-nonstreaming-seedtts-sweep/contract.json)
- Allocation:
  [rollout-plan.json](runs/2026-08-11-nonstreaming-seedtts-sweep/rollout-plan.json)
- Canonical summary:
  [results-summary.json](runs/2026-08-11-nonstreaming-seedtts-sweep/results-summary.json)
- Execution: commit `71022250` on free H200 GPUs, selected from high to low.
  Hyper00 used GPUs 7 and 5; hyper01 used GPUs 7 and 6. C=1 used the first 50
  samples, and c=2,4,8,16,32 each used all 1,088 English samples. The server
  used `examples/configs/dots_tts.yaml`, `max_running_requests=16`, seed 42,
  and 10 warmup requests. No c=64 run was launched.
- Failure: the first task containers used `jaxanluo/sglang-omni:dev`, whose
  bundled SGLang checkout lacked `sglang.srt.platforms.cpu`. All shards exited
  during import before any benchmark request. The failed logs remain under the
  prior persistent run root. The retry used the current H200 profile image
  `hongccc/sglang-omni:dev` with image id `374d0b1c30b2`; no failed-attempt data
  was mixed into the summary.

| Concurrency | Samples | Success | Request QPS | Audio s/s | Mean latency (s) | Mean RTF | Corpus WER |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 50 | 50/50 | 0.679 | 2.682 | 1.473 | 0.3818 | 1.241% |
| 2 | 1,088 | 1,088/1,088 | 1.162 | 4.843 | 1.721 | 0.4200 | 1.256% |
| 4 | 1,088 | 1,088/1,088 | 1.778 | 7.415 | 2.248 | 0.5471 | 1.264% |
| 8 | 1,088 | 1,088/1,088 | 3.019 | 12.587 | 2.645 | 0.6462 | 1.323% |
| 16 | 1,088 | 1,088/1,088 | 3.618 | 15.096 | 4.401 | 1.0704 | 1.348% |
| 32 | 1,088 | 1,088/1,088 | 4.271 | 17.836 | 7.410 | 1.8638 | 1.331% |

- Comparison: against the user-provided PR #1393 H100 table, request
  throughput is lower by 23.3%, 22.9%, 24.8%, 22.5%, 20.4%, and 7.7% at
  c=1,2,4,8,16,32 respectively. Mean RTF is higher by 31.7%, 30.4%, 33.8%,
  30.0%, 26.5%, and 8.3%.
- Serial check: a one-at-a-time rerun of c=1 on hyper00 GPU 7 measured 0.680
  req/s, 1.472 s latency, and 0.3812 RTF, effectively identical to the first
  sweep's 0.679 req/s, 1.473 s, and 0.3818. The c=2 continuation was interrupted
  before completion and is excluded. This rules out concurrent benchmark
  workers and host CPU pressure as the main cause of the c=1 gap.
- Two-level check: a requested c=16/c=32 priority rerun on two hyper00 GPUs
  measured 3.902/4.123 req/s, 4.080/7.675 s latency, and 0.9924/1.9343 RTF.
  Relative to the first sweep, c=16 improved 7.9% while c=32 declined 3.5%, so
  allocation/run variance exists but does not recover the historical H100
  result. A subsequent c=4/c=8 pair was stopped before valid results and is
  excluded.
- Attribution: #1393's 4.64 req/s c=16 number was measured on H100. PR #1374,
  opened from the same `525d41e3` base and merged eight minutes later, already
  measured the H200 mixed-length SeedTTS c=16 baseline at 3.694 req/s. The
  roughly 20% H100/H200 gap therefore predates #1438/#1439/#1440/#1441/#1444.
  In addition, PR #1445's `2e607bc` H200 baseline predates all five low-fruit
  PRs and measured non-streaming c=8 at 3.072 req/s; this sweep's 3.019 req/s is
  only 1.7% lower, within the documented run-to-run noise range.
- Smaller real issue: #1420 raised reference-encode `max_concurrency` to 8 while
  shipping `max_batch_size: 1`; reference encode and vocoder still serialize on
  one codec lock. #1434 measured the contention, and #1445's split-lock H200
  A/B recovers 3.91% non-streaming throughput. This is a real but partial loss,
  not an explanation for the apparent 20% regression. It also cannot explain
  c=1, whose reference path has no concurrent request to contend with.
- Decision: do not revert the low-fruit PRs based on this sweep. Treat the large
  delta as a baseline mismatch. If a new benchmark is authorized, compare
  pre-#1420, `2e607bc`, and current main sequentially on one H200 with the same
  image and dataset slice. Without more testing, continue #1445 as the bounded
  fix for the confirmed shared-lock cost.
- Cleanup: all task containers created for the initial, serial, and priority
  runs were removed after persistent outputs were checked. No other container
  or process was touched.
- Status: `COMPLETE`.

## 2026-08-11 PR #1445 H100 Split-Lock A/B

- Hypothesis: splitting reference-encode and vocoder locks should recover the
  non-streaming loss attributed to shared-lock contention without relying on a
  cross-device H100/H200 comparison.
- Contract:
  [contract.json](runs/2026-08-11-pr1445-h100-ab/contract.json)
- Summary:
  [results-summary.json](runs/2026-08-11-pr1445-h100-ab/results-summary.json)
- Source: current main `2b45073c` versus the #1445 patch ported onto current
  main as `6d54e21a`. The port includes #1444 and changes only codec lock
  ownership plus focused tests.
- Execution: H100 physical GPUs 0 and 1, full 1,088-sample SeedTTS EN set,
  non-streaming, seed 42, 10 warmups, `max_running_requests=16`, CUDA Graph max
  batch 16, and generate-only measurement. Base and candidate ran concurrently
  and swapped GPUs between rounds. C=16 used two rounds per revision; c=32 used
  four after the first two showed contradictory card-level deltas.
- Validation: the candidate passed 42 focused unit tests. All 12 benchmark runs
  completed 1,088/1,088 requests with zero failures. WER was not rerun because
  this contract isolated generation performance and the lock split does not
  change model math.

| Concurrency | Base req/s | Split req/s | Throughput delta | Base latency | Split latency | Base RTF | Split RTF |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 4.760 | 4.783 | +0.49% | 3.344 s | 3.327 s | 0.8125 | 0.8059 |
| 32 | 4.988 | 4.947 | -0.82% | 6.344 s | 6.401 s | 1.5961 | 1.6084 |

- C=16 card control: GPU 0 improved 0.47% and GPU 1 improved 0.52%, so the
  approximately 0.5% gain is repeatable but below the normal threshold for a
  material serving optimization.
- C=32 variance: base/split throughput medians are 4.973/4.973 req/s even
  though means differ by -0.82%. One split run on GPU 1 measured 4.812 req/s;
  the next on the same GPU measured 4.964 req/s. The change is effectively
  neutral at the median and increases observed run variance.
- Historical correction: current-main H100 baseline is 4.760 req/s at c=16
  and 4.988 req/s at c=32, respectively 2.6% and 12.6% above the official
  #1393 H100 reference. The earlier apparent regression came from comparing
  H200 runs with this H100 reference, not from #1438/#1439/#1440/#1441/#1444.
- Decision: #1445 should be evaluated and described as a streaming
  optimization. Its non-streaming +3.91% result does not generalize to c=16 or
  c=32. Do not spend more H100 time trying to make the c=32 mean positive; if
  the existing streaming throughput/TTFC versus ITL tradeoff is acceptable,
  rebase the patch and rerun the streaming c=8 contract.
- GitHub: the H100 result and current-main port were posted on
  [#1445](https://github.com/sgl-project/sglang-omni/pull/1445#issuecomment-5250903451).
- Cleanup: the verified raw results remain on personal persistent storage; the
  task container and its host map entry were removed.
- Status: `COMPLETE`.

## 2026-08-11 Current-Main H100 Complete Concurrency Ladder

- Hypothesis: the apparent regression against #1393 should disappear when
  current main is measured on the same H100 hardware class.
- Contract:
  [contract.json](runs/2026-08-11-main-h100-remaining-sweep/contract.json)
- Summary:
  [results-summary.json](runs/2026-08-11-main-h100-remaining-sweep/results-summary.json)
- Source: current main `2b45073c`, the same base commit used by the PR #1445
  H100 crossover run. Runtime image, model, dataset revision, server config,
  seed, warmups, and generation-only measurement are identical across all six
  concurrency levels.
- Execution: H100 physical GPUs 0 and 1 ran the same concurrency at the same
  time. C=1 used the first 50 samples; c=2/4/8 used the full 1,088-sample set.
  The earlier PR #1445 run supplies c=16 and c=32. Every row is a two-GPU mean,
  except c=32 which has four base runs from the crossover contract.
- Validation: all eight new runs completed with zero failed requests and the
  expected 50 or 1,088 result rows. WER was not rerun because these runs isolate
  generation performance.

| Concurrency | Samples | Repeats | Request QPS | Audio s/s | Mean latency | Mean RTF | QPS vs user #1393 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 50 | 2 | 0.935 | 3.726 | 1.070 s | 0.2751 | +5.65% |
| 2 | 1,088 | 2 | 1.556 | 6.493 | 1.286 s | 0.3138 | +3.18% |
| 4 | 1,088 | 2 | 2.493 | 10.407 | 1.603 s | 0.3902 | +5.41% |
| 8 | 1,088 | 2 | 3.875 | 16.173 | 2.062 s | 0.5021 | -0.54% |
| 16 | 1,088 | 2 | 4.760 | 19.859 | 3.344 s | 0.8125 | +4.72% |
| 32 | 1,088 | 4 | 4.988 | 20.818 | 6.344 s | 1.5961 | +7.83% |

- Conclusion: current main improves five of the six user-provided #1393 rows.
  C=8 is effectively flat at -0.54%. C=32 is now the throughput peak at 4.988
  req/s, but c=16 remains the better latency/throughput operating point because
  c=32 adds only 4.8% throughput while latency rises about 90%.
- Attribution: the previous H200 numbers cannot be used to claim a code
  regression against the H100 cookbook. On matching H100 hardware, current
  main is healthy and generally faster than #1393.
- Cleanup: the verified 3.7 GB raw result tree remains on personal persistent
  storage. The task container and its host map entry were removed.
- Status: `COMPLETE`.
