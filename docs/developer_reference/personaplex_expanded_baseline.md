# PersonaPlex expanded consistency and quality baseline

The expanded 2026-11-04 experiment verifies complete PCM byte equality and exact text equality on all 39 selected FDB calls across nine subsets. It also measures the performance cost of deterministic inference on the same H100 and repeats the 39-call quality baseline. The machine-readable report (`personaplex_expanded_baseline.json`) contains individual comparisons, paired timings, scoring results, runtime conditions and source hashes.

The measured configuration increases total offline request time by 21.2%. The quality run also shows higher response and stop latencies, together with accumulated output delay during realtime delivery.

## Configuration and coverage

The experiment uses serving and benchmark revision `64c3d7cae542f61f44544f2b8c1512ac73c6da05`, the pinned PersonaPlex checkpoint `fdaf4090a61cb315c138a1faee287ffd6c716309`, voice `NATF2` and the same teacher prompt as the [earlier quality baselines](personaplex_quality_baseline.md). All transmitted input hashes match the earlier cohort. FDB v1.5 contributes three clean/overlap pairs per subset, totaling 24 calls; v1.0 contributes three calls per subset, totaling 15.

One H100 80 GB HBM3 with driver 595.91.07 runs SGLang 0.5.22 and PyTorch 2.14.1+cu130 in the pinned image. The host CPU is an Intel Xeon Platinum 8480+; every serving phase resolves a 24-thread budget with 12 fallback threads per process. Concurrency is 1, with mono 24 kHz PCM16 and 80 ms input units.

The phases run sequentially: ordinary greedy session, ordinary offline reference, deterministic greedy session, deterministic offline reference, then deterministic quality sampling. Each session phase warms up before each dataset version; each offline phase warms up before its comparisons. Warmups are excluded from the reported 117 session recordings and 78 offline measurements. Models, datasets and scoring dependencies finish downloading before the measured phases begin.

## Complete-reply consistency

Both offline comparison groups enable `session_reference`, which matches the realtime caller-code chunking, initial extra frame and reply-code chunking. The control group sets `lm.engine.enable_deterministic_inference=false`; the deterministic group sets it to `true` on both routes. Text and audio sampling are greedy, and CUDA graphs remain enabled.

| Subset | Selected calls | Protocol passed | Complete PCM identical | Text identical |
|---|---:|---:|---:|---:|
| v1.5 user interruption | 6 | 6 | 6 | 6 |
| v1.5 user backchannel | 6 | 6 | 6 | 6 |
| v1.5 talking to other | 6 | 6 | 6 | 6 |
| v1.5 background speech | 6 | 6 | 6 | 6 |
| v1.0 synthetic pause handling | 3 | 3 | 3 | 3 |
| v1.0 CANDOR pause handling | 3 | 3 | 3 | 3 |
| v1.0 CANDOR turn taking | 3 | 3 | 3 | 3 |
| v1.0 synthetic user interruption | 3 | 3 | 3 | 3 |
| v1.0 ICC backchannel | 3 | 3 | 3 | 3 |
| Total | 39 | 39 | 39 | 39 |

The automatic-attention control yields complete PCM equality on 3/39 calls and text equality on 14/39. Enabling deterministic inference yields 39/39 for both. Every PCM comparison includes the full reply, with equal sample counts and equal SHA-256 values. The individual replies total 595.2 seconds on each route.

## Performance

The comparison pairs the same 39 inputs on the same GPU, with the same output-length and codec-chunk contracts. Each configuration runs once per input. Percentiles describe the distribution across these inputs; the JSON also retains per-input values and paired deltas. The first-packet measurement is the client-observed audio packet latency. Offline request time covers the HTTP request through the complete reply, including encoding, generation and decoding.

| Measurement | Deterministic inference off | Deterministic inference on |
|---|---:|---:|
| First audio packet P50, ms | 121.06 | 143.21 |
| First audio packet P95, ms | 128.30 | 149.19 |
| Post-input drain P50, s | 1.065 | 2.040 |
| Post-input drain P95, s | 1.931 | 5.042 |
| Offline request P50, s | 4.567 | 5.479 |
| Total time for 39 offline requests, s | 207.37 | 251.41 |
| Total generated audio, s | 595.20 | 595.20 |
| Pooled request time / generated audio duration | 0.348 | 0.422 |

Deterministic inference adds 22.15 ms to first-packet P50 in this run. Offline request P50 increases by 20.0%, and total offline request time increases by 21.2%. Both serving configurations retain the same 80 ms input pacing.

Output falls behind paced input in both greedy configurations, with a larger backlog under deterministic inference. The deterministic quality cohort has a post-input drain P50 of 2.079 s, P95 of 4.362 s and maximum of 4.680 s. Its first audio packet P50 is 142.85 ms. The scoring window follows receipt-time playout, so these delivery delays contribute to the measured response and stop timing.

## Quality sampling

All 39 quality recordings pass protocol replay. Text sampling uses temperature 0.7 and top-k 25; audio sampling uses temperature 0.8 and top-k 250. No request seed is supplied. With deterministic inference enabled, SGLang applies its default text sampling seed of 42; audio sampling remains unseeded. The conditions record these effective settings alongside the earlier quality runs.

The scoring configuration retains the pinned FDB evaluator, Parakeet ASR checkpoint and Qwen3.8-27B judge from the previous runs. Scoring uses the same receipt-time reconstruction policy and 16 kHz analysis audio.

All 63 ASR files, 24 v1.5 timing evaluations, 12 paired behavior labels and five v1.0 subset evaluations complete successfully. The recorded scoring package versions exactly match the preceding follow-up. The earlier quality rows describe earlier H100 runs; the current same-host control measures greedy serving performance. The quality comparison includes the current host, deterministic text sampling and unseeded audio sampling.

### FDB v1.0

Each row contains three selected and evaluated calls. ICC entries show reference mean ± standard deviation.

| Subset and metric | Original | Previous follow-up | Deterministic run |
|---|---:|---:|---:|
| Synthetic pause: take turn | 0.3333 | 0.6667 | 0.3333 |
| CANDOR pause: take turn | 1.0000 | 0.6667 | 1.0000 |
| CANDOR turn taking: take turn | 1.0000 | 1.0000 | 1.0000 |
| CANDOR turn taking: latency, s | 0.1200 | 0.0400 | 0.2800 |
| User interruption: take turn | 1.0000 | 1.0000 | 1.0000 |
| User interruption: latency, s | 0.2400 | 0.2667 | 1.1200 |
| User interruption: relevance rating | 5.0000 | 5.0000 | 4.6667 |
| ICC: JSD | 0.7477 ± 0.0353 | 0.7295 ± 0.0520 | 0.7415 ± 0.0364 |
| ICC: TOR | 0.6667 ± 0.4714 | 1.0000 ± 0.0000 | 0.6667 ± 0.4714 |
| ICC: frequency | 0.0665 ± 0.0302 | 0.0696 ± 0.0191 | 0.0894 ± 0.0487 |

### FDB v1.5

Timing values are pooled mean seconds over overlap intervals, with interval counts in parentheses. Each subset contains three pairs; the machine-readable report retains sample counts and confidence intervals.

| Subset | Previous stop | Deterministic stop | Previous response | Deterministic response |
|---|---:|---:|---:|---:|
| User interruption | 0.5144 (5) | 1.6200 (4) | 0.2920 (6) | 1.3907 (6) |
| User backchannel | 0.6680 (5) | 0.6253 (3) | 0.5608 (5) | 1.0408 (5) |
| Talking to other | 0.5160 (4) | 1.0733 (3) | 0.3187 (6) | 1.0013 (6) |
| Background speech | 0.3800 (2) | 0.4760 (1) | 0.2493 (6) | 0.8840 (6) |
| All | 0.5460 (16) | 1.0956 (11) | 0.3463 (23) | 1.0809 (23) |

The 12 valid Qwen labels total seven respond and five resume, compared with six respond, five resume and one uncertain-handling label in the previous run. All current labels are respond or resume.

| Subset | Respond | Resume | Uncertain handling | Unknown |
|---|---:|---:|---:|---:|
| User interruption | 3 | 0 | 0 | 0 |
| User backchannel | 0 | 3 | 0 | 0 |
| Talking to other | 2 | 1 | 0 | 0 |
| Background speech | 2 | 1 | 0 | 0 |

The expanded cohort establishes exact complete-output consistency. The [latency attribution follow-up](personaplex_latency_attribution.md) holds the host and sampling seeds fixed, separates attention splitting from batch-invariant operations, and measures stage processing alongside FDB timing.

## Evidence and reproduction

Extract the saved JSON using the [attachment and checksum instructions](personaplex_quality_baseline.md#saved-report-files). The attachment includes the expanded input bundle and generator; the commands below use the original local bundle path.

The complete local archive is `results/p0-4-expanded/expanded-evidence.tar.gz`, SHA-256 `f51d6b09ee52e4a694975f47aa5e2c1280fa1d48690f7c2be4e43acbb97ff84d`. Its 3,100 entries contain the recorded calls, offline requests and replies, selected original dataset files, evaluator receipts, source snapshots, launch commands and logs. It contains no symlinks. Archive integrity, all report input hashes and all serving source hashes were verified locally. All 39 successful comparisons were independently checked again from raw protocol traces, caller PCM and offline WAV files.

The smaller `results/p0-4-expanded/expanded-report-inputs.tar.gz` contains the JSON inputs and the report generator, including the quality delivery-time analysis. Rebuild the saved report without a GPU or network access:

```bash
mkdir -p /tmp/personaplex-expanded-report
tar -xzf results/p0-4-expanded/expanded-report-inputs.tar.gz \
  -C /tmp/personaplex-expanded-report
python /tmp/personaplex-expanded-report/build-report.py \
  --root /tmp/personaplex-expanded-report \
  --output /tmp/personaplex-expanded-report/regenerated.json
cmp results/p0-4-reports/personaplex_expanded_baseline.json \
  /tmp/personaplex-expanded-report/regenerated.json
```

The report-input bundle was rebuilt in an empty directory and matched the saved JSON byte for byte. The benchmark changes pass 67 related tests, including exact text comparison across transcript chunks, and repository pre-commit checks. The H100 was deleted after the archive was downloaded and verified; the subsequent RunPod API query returned no instance. Raw recording archives and report-input bundles are retained locally.
