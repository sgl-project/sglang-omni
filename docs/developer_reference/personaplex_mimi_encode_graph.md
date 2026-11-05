# PersonaPlex Mimi encoder CUDA graph

The session Mimi encoder can replay a CUDA graph for each 80 ms input frame. The graph runs the existing encoder operations and owns one session's convolution history, transformer cache and frame offset. A session reset clears those tensors in place before the graph is reused.

The subsequent [full FDB experiment](personaplex_full_fdb_baseline.md) covers all 1,723 released calls with default LM inference and checks eager/graph equality on the existing 39-call cohort.

## Configuration

Enable the prototype on a CUDA session server with `--mimi_encode.factory.enable_cuda_graph true`. The default is `false`. The encoder captures one graph per admitted session slot during serving-thread warmup. Whole-frame input uses graph replay; other supported chunk sizes use the existing eager encoder with the same session state. Empty end-of-input units retain their existing behavior.

On the measured H100 PCIe, a graph adds approximately 39.94 MiB of allocated GPU memory and takes 0.51 seconds to prepare in the standalone encoder experiment. Each session slot has its own graph pool and state; slots share model weights.

## Validation and standalone timing

The 2026-11-05 experiment uses an H100 PCIe 80GB at $2.89/hour, the pinned PersonaPlex model revision `fdaf4090a61cb315c138a1faee287ffd6c716309`, and PyTorch 2.14.1+cu130. Source starts at `564324173906b1311e4a82af284703814e5c5c6c` with the encoder graph changes recorded in the experiment's source archive and patch.

All 65 codec, session hook and stage tests pass on the GPU host with the final source. The graph test covers first-frame padding, attention-cache wrap, two interleaved sessions, a multi-frame eager chunk between graph replays, empty end-of-input and session reuse. Local codec tests pass with 26 cases and three accelerator skips; repository pre-commit checks pass.

The real-checkpoint encoder comparison uses sample 1 from v1.5 user interruption and background speech. Inputs use the recorder's PCM16 normalization and final-frame padding. Five blocks alternate eager/graph order, producing 20 encodings and 3,350 frames. Every code matches exactly across all runs. Timing includes input upload and blocking code download, with one CPU thread and no profiler.

| Encoder mode | Median of recording-level median frame time |
|---|---:|
| Eager | 11.05 ms |
| CUDA graph | 2.56 ms |

## Full-model preflight

The same two complete inputs are recorded with greedy sampling, seed 42, voice `NATF2`, the teacher prompt, concurrency 1 and batch-invariant operations enabled. Each completed launch warms both inputs, then records them without profiling and with request-event profiling. All eight measured recordings pass protocol validation. For each input, complete PCM and text match across both encoder modes and both profiling settings.

The following event-profiled measurements cover 335 input units per encoder mode. Each row comes from one completed fresh launch. Stage medians and total unit medians are calculated separately.

| Encoder mode | Median unit lifetime | Mimi encode service | LM service | Decoder service |
|---|---:|---:|---:|---:|
| Eager | 114.95 ms | 32.10 ms | 66.06 ms | 4.15 ms |
| CUDA graph | 75.28 ms | 3.65 ms | 57.08 ms | 4.10 ms |

The two unprofiled eager recordings have mean packet gaps of 115.33–116.26 ms and post-input drain times of 5.45–6.73 seconds. With graph replay, packet gaps are 79.89–80.04 ms and drain times are 82–92 ms. First audio arrives in 141–143 ms for eager and 111–117 ms for graph replay. These measurements include the full model and its process scheduling on this host.

The first full-model graph launch returns its first audio packet after 17.17 seconds, after the 14.568-second input ends. Its warmup report has no protocol violations and marks simultaneous input/output coverage as unexercised. The experiment retains that recording separately. The subsequent eager launch and graph relaunch pass both warmup calls. Logs place the long initial delays around the first LM prefill and first single-token step; their precise cause remains under investigation.

## Expanded verification

Both same-host greedy cohorts complete all 39 calls: 24 v1.5 overlap/clean variants and 15 v1.0 inputs. Every recording passes protocol validation, and the input hashes match the existing cohort. Complete PCM is byte-identical for all 39 eager/graph pairs, and every transcript matches.

Across these 39 calls, median first-audio latency decreases from 139.8 to 112.3 ms and median post-input drain decreases from 5.53 seconds to 91.3 ms. The maximum drain is 19.95 seconds for eager and 3.56 seconds for graph replay. Graph runs of ICC backchannel samples 0 and 1 retain 1.49 and 3.56 seconds of drain; the other 37 graph recordings drain within 105 ms.

A fresh-launch repeat of both ICC inputs completes six calls, including warmups, with 88–93 ms of drain. All four measured replies retain the original cohort's complete PCM and text. Event profiling covers 619 units with a median unit lifetime of 74.23 ms, including 3.69 ms of Mimi encode service. The original two slow calls remain in the cohort statistics; their trigger has not been identified.

All 39 graph recordings also match same-host offline requests with `session_reference=true`: complete PCM is byte-identical and every transcript matches. The offline warmup reply matches as well.

## Fixed-seed sampling and FDB timing

The quality-sampling comparison repeats the same 39 inputs with seed 42, text temperature 0.7 and top-k 25, and audio temperature 0.8 and top-k 250. Both modes enable batch-invariant operations. Each dataset section starts with the same 44.672-second ICC warmup. All 78 measured recordings pass protocol validation; all 39 eager/graph pairs have byte-identical complete PCM and matching text. Median first-audio latency decreases from 144.1 to 113.4 ms and median post-input drain from 5.98 seconds to 93.0 ms. The maximum graph drain is 97.5 ms.

FDB timing uses the 12 v1.5 samples, three per subset, with overlap and clean variants: 24 scoring windows per mode. Every window scores successfully using reference revision `3e799c45a045256f47d5f1c9cda90157e2d2ec9e` and Silero VAD 6.2.1. Scoring runs locally with PyTorch 2.11.0; a preflight reproduces the saved Linux VAD segments and intervals for the same audio. ASR and behavior labels are outside this timing experiment.

The table pools the reference implementation's per-sample intervals. Each cell shows mean seconds and interval count. These are campaign reductions; the reference defines no scalar aggregate. The artifact includes per-sample intervals, medians and 2,000 sample-bootstrap replicates with seed 42.

| Scoring window | Interval | Eager mean (count) | CUDA graph mean (count) |
|---|---|---:|---:|
| Overlap | Response | 2.480 s (21) | 0.496 s (22) |
| Overlap | Stop | 1.406 s (14) | 0.529 s (14) |
| Clean | Response | 1.495 s (12) | 0.388 s (12) |
| Clean | Stop | 0.871 s (3) | 0.668 s (3) |

FDB scoring audio uses FIFO receipt-time playout within the input window. Faster delivery changes this waveform's timing and the intervals visible before the window ends, while the complete generated PCM stays identical. The overlap response interval count therefore changes from 21 to 22. Both modes have model speech at the window end in 10 overlap and six clean samples. The user-ending-without-a-later-model-start label occurs in three eager overlap samples and two graph samples. Both modes have one overlap and nine clean samples with no stop interval. These labels remain in the summaries alongside the interval statistics.

## Evidence

Local evidence is stored under `results/p0-4-encode-graph/`. `preparation.json` identifies the source snapshot and allocation; `encoder-benchmark.json` stores every standalone frame time and code digest; `smoke-analysis.json` contains full-model output digests and stage metrics. `experiment-summary.json` links the greedy, offline and quality comparisons and both timing summaries. Large recordings and generated JSON stay in the experiment artifacts.

The complete remote archive is `mimi-encode-graph-evidence.tar.gz`, SHA-256 `93cb8519f913e831752d29ce6e67dd50651f6bde02a7c7f94db63a0d219b512a`. It includes the original source archive, the final validated source overlay and its file hashes, test logs, launch commands and recordings. Local source hashes match that overlay. The GPU was released after download and archive verification; the local timing results and compact report are packaged separately in `mimi-encode-graph-report.zip`.
