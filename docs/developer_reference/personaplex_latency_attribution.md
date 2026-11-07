# PersonaPlex latency attribution experiment

The 2026-11-04 experiment varies attention splitting, batch-invariant operations and CPU threads on one H100. Greedy processing takes about 94 ms per 80 ms input unit, creating backlog. Quality-sampling profiles run below 80 ms, with performance ordering varying across phases. The [interleaved follow-up](personaplex_interleaved_analysis.md) controls launch order and adds kernel timing.

## Configuration

Serving revision is `a44bd6b0a5771d4bb7c29a220cc0f660e6973fac`, with checkpoint `fdaf4090a61cb315c138a1faee287ffd6c716309`, voice `NATF2`, the existing teacher prompt, concurrency 1 and 80 ms PCM input units. The pinned container is `hongccc/sglang-omni@sha256:06ed306707bcf3d9e9beb5ba7ce66f09a666cac11c23455ece50c34199c0e985`. The H100 80 GB HBM3 runs driver 595.91.07, PyTorch 2.14.1+cu130 and SGLang 0.5.22. Its Xeon Platinum 8470 host resolves a 23-thread budget and 11 fallback threads per process. Recordings use local disk.

Greedy controls use sample 1 from v1.5 user interruption and background speech, repeated twice per configuration. Quality timing uses sample 1 from each of the four v1.5 subsets, both clean and overlap variants, repeated twice: 16 calls per arm. Both quality arms use seed 42, text temperature 0.7/top-k 25 and audio temperature 0.8/top-k 250. Models and inputs are already downloaded when recording starts.

The batch-invariant-off control skips SGLang's `maybe_enable_batch_invariant_mode()` registration while retaining deterministic sampling and the deterministic attention configuration. Experimental backend source snapshots and launch commands accompany each phase. These patches remain experiment artifacts.

## Greedy controls

Each row summarizes four recordings. Packet gap is the median of the per-recording mean gap; first packet and post-input drain are medians across recordings.

| Configuration | Packet gap, ms | First packet, ms | Post-input drain, s |
|---|---:|---:|---:|
| Ordinary greedy, initial | 82.77 | 116.00 | 0.505 |
| Ordinary greedy, attention split 1 | 85.97 | 125.37 | 1.090 |
| Deterministic greedy | 93.73 | 147.54 | 2.413 |
| Deterministic greedy, automatic split | 92.40 | 143.81 | 2.166 |
| Ordinary greedy, later repeat | 89.68 | 123.91 | 1.750 |
| Deterministic greedy, one CPU thread | 95.03 | 144.59 | 2.628 |
| Deterministic greedy, batch-invariant off | 87.95 | 127.51 | 1.296 |

Attention splitting alone does not reproduce the full deterministic overhead. One CPU thread yields no improvement. The ordinary configuration's later repeat also slows down, showing variation across sequential phases on this host.

Changing the attention split preserves both greedy outputs within each deterministic setting. Disabling batch-invariant operations changes both complete PCM replies to those of ordinary greedy. With batch-invariant operations disabled on both routes, the two offline `session_reference` comparisons match complete PCM byte for byte and text exactly. The earlier [39-call consistency result](personaplex_expanded_baseline.md#complete-reply-consistency) remains the broader validation for the full deterministic configuration.

## Complete stage profiles

Each profile covers the same two overlap inputs. Greedy batch-invariant-off has two repeats, totaling 670 input units; the other rows have one repeat, totaling 335 units each. Every included capture has three complete process receipts and zero dropped events. Two earlier captures with a smaller event buffer dropped events and are excluded from this table.

| Configuration | Admitted to finished P50, ms | Mimi encode service P50, ms | LM service P50, ms | Decoder service P50, ms |
|---|---:|---:|---:|---:|
| Deterministic greedy | 93.53 | 29.67 | 47.42 | 4.18 |
| Greedy, batch-invariant off | 89.11 | 29.91 | 41.65 | 3.85 |
| Deterministic quality sampling | 74.84 | 20.53 | 40.18 | 3.78 |
| Quality sampling, batch-invariant off | 65.98 | 21.09 | 31.97 | 3.43 |

Stage service is wall-clock time, including execution and scheduling. Stage and total-lifetime medians are computed separately.

Greedy deterministic processing exceeds the 80 ms cadence and accumulates 1.267 seconds of median admission wait. Quality-sampling profiles stay below that cadence. Mimi timing also varies across phases with identical input and configuration, motivating interleaved measurements.

## Seeded quality timing

All 32 recordings pass protocol validation, and all 32 FDB timing evaluations succeed. Within each arm, all eight inputs reproduce identical complete PCM and text across the two recordings. Across arms, all eight transmitted input hashes match, while none of the eight output PCM or text comparisons match. Identical seeds therefore control sampling randomness without holding generated content constant when numerical operations change.

Median drain is 0.083 seconds with deterministic operations and 1.587 seconds with batch-invariant operations off. The later profiler reverses this speed ordering, motivating repeated fresh-launch controls.

Timing below is the pooled mean over the reference evaluator's overlap intervals, with interval counts in parentheses. Each subset contains one distinct pair recorded twice; repeats do not increase the number of distinct dataset samples.

| Subset | Deterministic stop, s | Batch-invariant-off stop, s | Deterministic response, s | Batch-invariant-off response, s |
|---|---:|---:|---:|---:|
| User interruption | 0.596 (4) | 0.796 (4) | 0.404 (4) | 0.708 (4) |
| User backchannel | 0.524 (2) | 0.604 (2) | 1.748 (4) | 1.196 (4) |
| Talking to other | 0.124 (2) | 1.244 (2) | 0.452 (4) | 1.020 (4) |
| Background speech | No interval (0) | No interval (0) | 0.276 (4) | 0.852 (4) |
| All | 0.460 (8) | 0.860 (8) | 0.720 (16) | 0.944 (16) |

Scoring uses evaluator `3e799c45a045256f47d5f1c9cda90157e2d2ec9e`, receipt-time playout, SoundFile and Silero VAD 6.2.1 on CPU. These intervals reflect generated speech and delivery timing.

## Follow-up

The [interleaved experiment](personaplex_interleaved_analysis.md) repeats five fresh launches per configuration and sampling mode with CPU/GPU telemetry and kernel attribution. It leads to [Mimi encoder CUDA graphs](personaplex_mimi_encode_graph.md) and the [full FDB baseline](personaplex_full_fdb_baseline.md).

## Evidence

The raw archive is `results/p0-4-latency/latency-evidence.tar.gz`, SHA-256 `aa0209036fa2bafbe612571578afc27a50a553886bd22095f1fd318ba51864c0`. It contains recordings, offline replies, scoring receipts, profiler captures, source, commands, telemetry and setup logs.

The machine-readable report and portable report inputs are retained under `results/p0-4-latency/`. Generated JSON and raw audio remain local artifacts for attachment.

The report is `personaplex_latency_attribution.json`, SHA-256 `b6e28565f73769c9ac16cdfda133bf145078341f882a06182131206e4cc2869c`. Rebuild it without a GPU or network access:

```bash
mkdir -p /tmp/personaplex-latency-report
tar -xzf results/p0-4-latency/latency-report-inputs.tar.gz \
  -C /tmp/personaplex-latency-report
python /tmp/personaplex-latency-report/build-report.py \
  --root /tmp/personaplex-latency-report \
  --output /tmp/personaplex-latency-report/regenerated.json
cmp results/p0-4-latency/personaplex_latency_attribution.json \
  /tmp/personaplex-latency-report/regenerated.json
```

The portable bundle reproduces the report byte for byte. Verification covers 68 recordings, two PCM/text comparisons, 32 scoring receipts, 12 profiler receipts and 1,953 source hashes. The GPU was released after archive verification.
