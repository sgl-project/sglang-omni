# PersonaPlex interleaved latency analysis

The 2026-11-04 interleaved experiment measures the cost of batch-invariant operations across fresh launches. Disabling them reduces median unit lifetime by 5.50 ms for greedy sampling and 7.12 ms for quality sampling, and changes PCM and text. All 80 primary calls keep up with 80 ms input pacing.

On this H100 NVL/AMD host, the [earlier experiment's](personaplex_latency_attribution.md) seconds-long backlog does not recur. Mimi spends about 2 ms executing kernels within 16.5–17.4 ms of stage service time. This finding led to the [encoder CUDA graph implementation](personaplex_mimi_encode_graph.md), later evaluated on [full FDB](personaplex_full_fdb_baseline.md).

## Design and validation

Serving revision is `f1cb3b19d578d8b38ef6e4da2301c45807cfbc53`; model revision is `fdaf4090a61cb315c138a1faee287ffd6c716309`. The pinned container, voice `NATF2`, teacher prompt, concurrency 1 and seed 42 match the preceding experiment. Quality sampling uses text temperature 0.7/top-k 25 and audio temperature 0.8/top-k 250. Greedy sampling sets both temperatures to zero.

The host uses H100 NVL, AMD EPYC 9654, PyTorch 2.14.1+cu130 and SGLang 0.5.22. Its CPU budget is 27 threads, with a cgroup quota of 26.16 CPUs. The preceding H100 SXM/Xeon Platinum 8470 host used 23 threads. Performance comparisons below pair configurations on this host.

Five blocks alternate sampling-mode order and configuration order. Each of the four configurations gets five fresh launches. Every launch warms both complete inputs, then records them once with request-event profiling and once without it; profiler order alternates across blocks. The inputs are sample 1 from v1.5 user interruption and background speech. This produces 80 measured recordings over two distinct inputs, plus 40 warmups. Separate kernel profiles use four-second prefixes, with one capture per configuration and one additional preflight capture.

All 80 primary calls pass protocol validation. Each input/configuration reproduces its full PCM and text across ten recordings and matches the prior host's PCM. Changing batch-invariant operations changes PCM and text in both sampling modes.

All 24 campaign phases and the preflight finish successfully. The 75 process receipts report complete event storage and zero dropped events. Ten CUDA traces contain 1,589,840 kernel events.

## Delivery and unit processing

The following client measurements exclude event profiling. Each row contains ten recordings: two inputs across five fresh launches. Gap is the median of each recording's mean audio-packet gap; first packet and drain are medians across recordings.

| Sampling | Batch-invariant operations | First packet, ms | Packet gap, ms | Post-input drain, ms |
|---|---|---:|---:|---:|
| Greedy | On | 74.93 | 80.02 | 63.91 |
| Greedy | Off | 67.83 | 80.05 | 61.77 |
| Quality | On | 75.81 | 79.99 | 61.56 |
| Quality | Off | 63.13 | 80.06 | 54.11 |

Across all 80 primary recordings, post-input drain stays below 86 ms. With request-event profiling enabled, the median paired drain change is +4.00 ms for deterministic greedy, -3.24 ms for greedy with batch-invariant operations off, +3.52 ms for deterministic quality and -1.36 ms for quality with those operations off. Event profiling does not introduce sustained backlog in these runs.

Stage measurements below are medians of the five per-launch P50 values. Each launch contributes 335 units across the two inputs. Component medians and total-lifetime medians are calculated separately and need not sum.

| Sampling | Batch-invariant operations | Admitted to finished, ms | Mimi encode service, ms | LM service, ms | Decoder service, ms |
|---|---|---:|---:|---:|---:|
| Greedy | On | 59.35 | 17.09 | 30.68 | 3.45 |
| Greedy | Off | 52.92 | 17.40 | 28.12 | 3.02 |
| Quality | On | 61.11 | 16.61 | 31.84 | 3.47 |
| Quality | Off | 53.84 | 16.52 | 29.12 | 3.03 |

All five greedy pairs favor disabling batch-invariant operations, with reductions of 4.97–6.86 ms. All five quality pairs do as well, with reductions of 5.89–8.15 ms. The 5.50/7.12 ms paired medians differ from subtracting the table's column medians because pairing preserves each block's measurements.

## GPU kernel attribution

Kernel timestamps are aligned with request-event stage windows using the trace's `baseTimeNanoseconds`. Attribution counts overlapping kernel execution once within each process and clips it to the stage window. The first unit and zero-duration end-of-input unit are excluded, leaving 98 steady input units per capture. Each table entry is the median over those units.

| Sampling | Batch-invariant operations | Mimi kernel busy, ms | LM kernel busy, ms | Decoder kernel busy, ms | LM kernel count |
|---|---|---:|---:|---:|---:|
| Greedy | On | 2.018 | 8.248 | 2.405 | 1,693 |
| Greedy | Off | 2.016 | 7.034 | 2.022 | 1,165 |
| Quality | On | 2.014 | 8.808 | 2.407 | 1,775 |
| Quality | Off | 2.015 | 7.590 | 2.023 | 1,247 |

Mimi executes 841 kernels per unit, taking about 2 ms of GPU time. Stage service takes 16.5–17.4 ms in primary runs and 14.7–26.4 ms in kernel captures. The gap includes host execution, launches, synchronization, transfers and scheduling.

Disabling batch-invariant operations removes 528 LM kernels per unit and 1.21–1.22 ms of LM GPU time; decoder GPU time falls by 0.38 ms. Dispatch and synchronization are further candidates for the larger wall-clock difference.

## CPU and GPU telemetry

One-second cgroup snapshots inside recording windows cover 10,887 periods, including 83 throttled periods (0.76%). Cumulative `throttled_time` increases by 215.93 seconds. Each sampled window contains at most four throttled periods.

Ten scheduler-counter totals decrease as threads exit. A follow-up CPU-wait analysis needs thread identities and lifetimes; the raw counters are retained.

The 1,168 GPU samples show constant SM/memory clocks of 1,785/2,619 MHz, utilization of 18% median and 30% maximum, and temperatures of 44–48°C. Sampling includes idle time between paced input units.

## Follow-up

The [Mimi encoder graph experiment](personaplex_mimi_encode_graph.md) implements replay, verifies complete-output equality on 39 calls and measures reduced backlog. The [full FDB experiment](personaplex_full_fdb_baseline.md) extends coverage to 1,723 calls with default LM inference.

## Evidence and reproduction

The raw archive is `results/p0-4-interleaved/interleaved-evidence.tar.gz`, SHA-256 `73d9924e88d293c6831fd295334e02547859a437cf48a18d34eeabb345adb3b2`. It contains recordings, launch commands, backend snapshots, event and CUDA traces, CPU/GPU telemetry and runtime conditions. The machine-readable report is `results/p0-4-interleaved/personaplex_interleaved_analysis.json`, SHA-256 `6e88e520cc244e647c23d0642d79aab8e5f26529c689a4b081f89326f1469807`. Generated JSON stays in the local attachment package.

Rebuild the report from the extracted measurements without a GPU:

```bash
mkdir -p /tmp/personaplex-interleaved-report
tar -xzf results/p0-4-interleaved/interleaved-report-inputs.tar.gz \
  -C /tmp/personaplex-interleaved-report
python /tmp/personaplex-interleaved-report/build-report.py \
  --root /tmp/personaplex-interleaved-report \
  --output /tmp/personaplex-interleaved-report/regenerated.json
cmp results/p0-4-interleaved/personaplex_interleaved_analysis.json \
  /tmp/personaplex-interleaved-report/regenerated.json
```

The H100 NVL cost $3.19/hour and was released after approximately one hour, following archive download and verification.
