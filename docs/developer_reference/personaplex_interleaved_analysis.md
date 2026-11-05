# PersonaPlex interleaved latency analysis

The 2026-11-04 interleaved experiment finds a repeatable cost for batch-invariant operations while all 80 primary recordings keep up with 80 ms input pacing. Disabling those operations reduces the per-launch median unit lifetime in every paired block: the median paired reduction is 5.50 ms for greedy sampling and 7.12 ms for quality sampling. It also changes the complete PCM and text outputs. The current deterministic implementation remains the reference configuration.

The seconds-long backlog from the [earlier attribution experiment](personaplex_latency_attribution.md) does not recur on this host. This run uses an H100 NVL and AMD CPU, so the earlier host's slowdown remains unresolved. The new measurements identify substantial time outside GPU kernel execution, particularly in Mimi encoding, as the next optimization target.

## Design and validation

Serving revision is `f1cb3b19d578d8b38ef6e4da2301c45807cfbc53`; model revision is `fdaf4090a61cb315c138a1faee287ffd6c716309`. The pinned container, voice `NATF2`, teacher prompt, concurrency 1 and seed 42 match the preceding experiment. Quality sampling uses text temperature 0.7/top-k 25 and audio temperature 0.8/top-k 250. Greedy sampling sets both temperatures to zero.

The machine has an H100 NVL, an AMD EPYC 9654 CPU, PyTorch 2.14.1+cu130 and SGLang 0.5.22. The process CPU budget resolves to 27; the cgroup quota is 2,615,625 microseconds per 100,000-microsecond period, or approximately 26.16 CPUs. The preceding machine used H100 SXM and Xeon Platinum 8470 with a 23-thread budget. These changes prevent attributing the cross-host improvement to one component.

Five blocks alternate sampling-mode order and configuration order. Each of the four configurations gets five fresh launches. Every launch warms both complete inputs, then records them once with request-event profiling and once without it; profiler order alternates across blocks. The inputs are sample 1 from v1.5 user interruption and background speech. This produces 80 measured recordings over two distinct inputs, plus 40 warmups. Separate kernel profiles use four-second prefixes, with one capture per configuration and one additional preflight capture.

All 80 primary recordings pass protocol validation. Each input/configuration combination yields identical full PCM and text across its ten recordings, including both profiler settings and all five launches. The eight input/configuration combinations also match the corresponding prior-host PCM outputs. Changing the batch-invariant setting changes both PCM and text for both inputs in both sampling modes. This campaign measures repeatability within the session route; it adds no new offline comparisons or FDB behavior scores.

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

Mimi executes 841 kernels per steady unit in every configuration. Its kernel busy time stays near 2 ms, while its primary-stage service medians are approximately 16.5–17.4 ms. In the kernel captures themselves, Mimi service medians range from 14.7 to 26.4 ms. That difference includes host execution, launch gaps, synchronization, transfers and scheduling; it is not a direct measurement of CPU compute alone.

Disabling batch-invariant operations removes 528 LM kernels per steady unit and about 1.21–1.22 ms of LM kernel busy time. Decoder kernel busy time falls by about 0.38 ms. The larger wall-clock improvement supports investigating dispatch and synchronization costs alongside arithmetic. Kernel captures are instrumented runs over short prefixes; their service times are reported separately from primary timing measurements.

## CPU and GPU telemetry

Within the primary recording windows, cgroup snapshots cover 10,887 scheduling periods, with 83 throttled periods (0.76%). The cumulative `throttled_time` counter increases by 215.93 seconds; this value is retained as a cgroup counter and is not converted into an elapsed-time percentage. There are at most four throttled periods in any sampled recording window. Measurements use one-second snapshots inside each recording window, excluding boundary fragments.

The scheduler monitor sums counters across currently existing threads. Ten process/recording comparisons have decreasing totals when threads exit, so those aggregates cannot provide a complete CPU-wait decomposition. The raw counters are retained; a follow-up scheduling trace should preserve thread identities and lifetimes. No claim about the earlier host's CPU throttling follows from these new-host measurements.

The 1,168 GPU samples inside primary recording windows show constant SM and memory clocks of 1,785 and 2,619 MHz. GPU utilization has a median of 18% and maximum of 30%; temperature ranges from 44 to 48°C. These paced, single-session measurements include time between input units. They do not indicate GPU saturation or clock variation as the source of this campaign's timing differences.

## Next implementation experiment

Prototype CUDA graph replay for the fixed-size, stateful Mimi encoding step while preserving its arithmetic and codec-state transitions. The 841 small kernels and roughly 2 ms of GPU execution per input unit make launch and synchronization overhead a concrete target. Measure the prototype with the same interleaved controls, then repeat complete PCM/text parity on the 39-call cohort before expanding quality coverage.

Keep batch-invariant operations enabled during that prototype. Their cost is measurable, and disabling them changes generated outputs. Resolve remaining host-side delays with thread-lifetime-aware scheduling traces and CUDA API correlation rather than treating the wall-clock/kernel-time difference as one cause.

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

The H100 NVL was allocated at $3.19/hour and deleted after approximately one hour, after the archive downloaded and passed checksum verification. The background retry task has been unloaded. Analysis uses the saved evidence and starts no new GPU instance.
