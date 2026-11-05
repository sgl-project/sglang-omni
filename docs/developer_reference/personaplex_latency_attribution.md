# PersonaPlex latency attribution experiment

The 2026-11-04 follow-up separates attention splitting, batch-invariant operations and CPU thread settings on one H100. Greedy profiling reproduces output backlog when processing an 80 ms input unit takes about 94 ms. Seeded quality sampling also produces runs below 80 ms per unit. The observed performance ordering changes between recording phases, so the experiment retains the current deterministic configuration and identifies the measurements needed for the next optimization.

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

These are wall-clock stage service times, including execution and scheduling effects. They are not isolated GPU kernel measurements. Stage medians do not sum to the median of the complete unit lifetime.

The greedy deterministic capture has a ready-to-admitted P50 of 1.267 s: pending input accumulates before admission because processing exceeds the 80 ms input cadence. The sampled captures operate below that cadence at the median. Mimi encoding also changes substantially between phases, although its input and configuration are the same; sampling mode and host/runtime variation need separate measurement before assigning a fixed cost to the deterministic operation switch.

## Seeded quality timing

All 32 recordings pass protocol validation, and all 32 FDB timing evaluations succeed. Within each arm, all eight inputs reproduce identical complete PCM and text across the two recordings. Across arms, all eight transmitted input hashes match, while none of the eight output PCM or text comparisons match. Identical seeds therefore control sampling randomness without holding generated content constant when numerical operations change.

The deterministic arm's post-input drain P50 is 0.083 s; the batch-invariant-off arm's is 1.587 s. The later sampled profiler shows the opposite processing-speed ordering. This variation prevents selecting the faster production configuration from this campaign alone.

Timing below is the pooled mean over the reference evaluator's overlap intervals, with interval counts in parentheses. Each subset contains one distinct pair recorded twice; repeats do not increase the number of distinct dataset samples.

| Subset | Deterministic stop, s | Batch-invariant-off stop, s | Deterministic response, s | Batch-invariant-off response, s |
|---|---:|---:|---:|---:|
| User interruption | 0.596 (4) | 0.796 (4) | 0.404 (4) | 0.708 (4) |
| User backchannel | 0.524 (2) | 0.604 (2) | 1.748 (4) | 1.196 (4) |
| Talking to other | 0.124 (2) | 1.244 (2) | 0.452 (4) | 1.020 (4) |
| Background speech | No interval (0) | No interval (0) | 0.276 (4) | 0.852 (4) |
| All | 0.460 (8) | 0.860 (8) | 0.720 (16) | 0.944 (16) |

Scoring uses the pinned FDB evaluator `3e799c45a045256f47d5f1c9cda90157e2d2ec9e`, receipt-time playout, SoundFile loading and Silero VAD 6.2.1 on CPU. ASR and behavior judging are outside this timing experiment. Timing includes both generated behavior and delivery effects.

## Next experiment

The [completed interleaved experiment](personaplex_interleaved_analysis.md) repeats both configurations across five fresh launches per sampling mode and adds CPU/GPU telemetry and CUDA kernel attribution.

Keep the current deterministic implementation while profiling Mimi encoding and LM execution with GPU kernel timing and CPU scheduling/throttling measurements. Alternate configurations across fresh launches, retain seed 42 and identical inputs, and compare repeated runs with and without event profiling. This will distinguish stage execution costs from the phase-to-phase variation observed here. Any candidate change should then repeat complete-output parity on the 39-call cohort before expanding quality coverage.

## Evidence

The local raw archive is `results/p0-4-latency/latency-evidence.tar.gz`, SHA-256 `aa0209036fa2bafbe612571578afc27a50a553886bd22095f1fd318ba51864c0`. It includes original recordings, offline requests/replies, timing receipts, complete and incomplete profiler captures, backend snapshots, serving source, launch commands, GPU telemetry and setup failure logs. The offline launch was retried after removing a realtime-only seed flag; CPU scoring was rerun with the source directory on its import path.

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

The portable bundle reproduces the report byte for byte. Local verification independently checks all 68 measured recordings, both complete PCM/text comparisons, 32 scoring receipts, 12 complete profiler receipts and 1,953 source hashes. The archive contains no symbolic links; its hard links resolve to files included in the archive. The H100 was deleted after download and archive verification, and the subsequent RunPod query returned no instance.
