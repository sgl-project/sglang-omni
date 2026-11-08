# PersonaPlex Mimi encoder CUDA graph

This change adds CUDA graph replay for each 80 ms Mimi input frame. Each session owns its convolution history, transformer cache and frame offset; reset clears them in place for reuse.

This per-session batch-1 implementation overlaps with P2-1 in [#3174](https://github.com/sgl-project/sglang-omni/pull/3174), which covers batched streaming encoding and CUDA graph replay. Our experiments with this separate implementation may serve as a performance reference for that work.

The subsequent [full FDB experiment](personaplex_full_fdb_baseline.md) covers all 1,723 released calls with default LM inference and checks eager/graph equality on the existing 39-call cohort.

## Configuration

Enable it with `--mimi_encode.factory.enable_cuda_graph true` on a CUDA session server; the default is `false`. Serving-thread warmup captures one graph per session slot. Single-frame input uses replay, other supported chunk sizes use eager encoding, and empty end-of-input units emit no codes.

On H100 PCIe, each graph adds 39.94 MiB and takes 0.51 seconds to prepare. Slots share model weights and own their graph pools and state.

## Validation and standalone timing

The 2026-11-05 experiment uses an H100 PCIe 80GB at $2.89/hour, the pinned PersonaPlex model revision `fdaf4090a61cb315c138a1faee287ffd6c716309`, and PyTorch 2.14.1+cu130. Source starts at `564324173906b1311e4a82af284703814e5c5c6c` with the encoder graph changes recorded in the experiment's source archive and patch.

All 65 GPU codec, session-hook and stage tests pass, covering padding, cache wrap, interleaved sessions, eager chunks and state reuse. Local codec tests pass with 26 cases and three accelerator skips; pre-commit passes.

The real-checkpoint encoder comparison uses sample 1 from v1.5 user interruption and background speech. Inputs use the recorder's PCM16 normalization and final-frame padding. Five blocks alternate eager/graph order, producing 20 encodings and 3,350 frames. Every code matches exactly across all runs. Timing includes input upload and blocking code download, with one CPU thread and no profiler.

| Encoder mode | Median of recording-level median frame time |
|---|---:|
| Eager | 11.05 ms |
| CUDA graph | 2.56 ms |

## Full-model preflight

Greedy preflight uses the same two inputs, seed 42, voice `NATF2`, the teacher prompt, concurrency 1 and batch-invariant operations. After warmup, both encoder modes run with and without event profiling. All eight calls pass protocol validation with identical complete PCM and text for each input.

The following event-profiled measurements cover 335 input units per encoder mode. Each row comes from one completed fresh launch. Stage medians and total unit medians are calculated separately.

| Encoder mode | Median unit lifetime | Mimi encode service | LM service | Decoder service |
|---|---:|---:|---:|---:|
| Eager | 114.95 ms | 32.10 ms | 66.06 ms | 4.15 ms |
| CUDA graph | 75.28 ms | 3.65 ms | 57.08 ms | 4.10 ms |

The two unprofiled eager recordings have mean packet gaps of 115.33–116.26 ms and post-input drain times of 5.45–6.73 seconds. With graph replay, packet gaps are 79.89–80.04 ms and drain times are 82–92 ms. First audio arrives in 141–143 ms for eager and 111–117 ms for graph replay. These measurements include the full model and its process scheduling on this host.

The initial graph warmup takes 17.17 seconds to emit audio, exceeding its 14.568-second input and leaving simultaneous input/output unexercised. Subsequent eager and graph warmups pass. Logs place the delay around initial LM execution; its cause remains under investigation.

## Expanded verification

The same-host greedy comparison covers 24 v1.5 variants and 15 v1.0 calls. All calls pass protocol validation, and all 39 eager/graph pairs have identical complete PCM and text.

Across these 39 calls, median first-audio latency decreases from 139.8 to 112.3 ms and median post-input drain decreases from 5.53 seconds to 91.3 ms. The maximum drain is 19.95 seconds for eager and 3.56 seconds for graph replay. Graph runs of ICC backchannel samples 0 and 1 retain 1.49 and 3.56 seconds of drain; the other 37 graph recordings drain within 105 ms.

A fresh launch repeats both ICC inputs with 88–93 ms of drain and identical PCM/text. Profiling covers 619 units with median lifetime 74.23 ms and Mimi encode service 3.69 ms. The original two outliers remain in the cohort statistics; their trigger is unresolved.

All 39 graph recordings also match same-host offline requests with `session_reference=true`: complete PCM is byte-identical and every transcript matches. The offline warmup reply matches as well.

## Fixed-seed sampling and FDB timing

The quality-sampling comparison repeats the same 39 inputs with seed 42, text temperature 0.7 and top-k 25, and audio temperature 0.8 and top-k 250. Both modes enable batch-invariant operations. Each dataset section starts with the same 44.672-second ICC warmup. All 78 measured recordings pass protocol validation; all 39 eager/graph pairs have byte-identical complete PCM and matching text. Median first-audio latency decreases from 144.1 to 113.4 ms and median post-input drain from 5.98 seconds to 93.0 ms. The maximum graph drain is 97.5 ms.

FDB timing covers 12 v1.5 pairs and 24 windows per mode. All windows score with evaluator `3e799c45a045256f47d5f1c9cda90157e2d2ec9e`, Silero VAD 6.2.1 and PyTorch 2.11.0. Local scoring reproduces the saved Linux VAD segments and intervals. The [full FDB experiment](personaplex_full_fdb_baseline.md) adds ASR and behavior scoring.

Each cell pools the reference intervals and shows mean seconds and count. The report includes per-sample intervals, medians and 2,000 sample-bootstrap replicates with seed 42.

| Scoring window | Interval | Eager mean (count) | CUDA graph mean (count) |
|---|---|---:|---:|
| Overlap | Response | 2.480 s (21) | 0.496 s (22) |
| Overlap | Stop | 1.406 s (14) | 0.529 s (14) |
| Clean | Response | 1.495 s (12) | 0.388 s (12) |
| Clean | Stop | 0.871 s (3) | 0.668 s (3) |

FDB scoring audio uses FIFO receipt-time playout within the input window. Faster delivery changes this waveform's timing and the intervals visible before the window ends, while the complete generated PCM stays identical. The overlap response interval count therefore changes from 21 to 22. Both modes have model speech at the window end in 10 overlap and six clean samples. The user-ending-without-a-later-model-start label occurs in three eager overlap samples and two graph samples. Both modes have one overlap and nine clean samples with no stop interval. These labels remain in the summaries alongside the interval statistics.

## Further work

- [ ] Coordinate implementation and evaluation scope with the author of [#3174](https://github.com/sgl-project/sglang-omni/pull/3174).
- [ ] Evaluate #3174 with the existing 39-call consistency comparison and FDB tooling under matched inputs, sampling settings and runtime conditions.

## Evidence

`results/p0-4-encode-graph/` contains source identity, frame timings, code digests, recordings and stage metrics. `experiment-summary.json` links the greedy, offline and quality comparisons and timing summaries.

The raw archive is `mimi-encode-graph-evidence.tar.gz`, SHA-256 `93cb8519f913e831752d29ce6e67dd50651f6bde02a7c7f94db63a0d219b512a`. It contains source snapshots, tests, commands and recordings. `mimi-encode-graph-report.zip` packages the analysis and local timing results.
