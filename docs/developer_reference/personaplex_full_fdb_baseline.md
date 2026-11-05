# PersonaPlex full FDB baseline

The 2026-11-05 experiment records every released FDB v1.5 overlap/clean input and every v1.0 input: 1,723 calls and 7.316 hours of input audio. All calls pass protocol validation, exercise simultaneous input/output and meet the recorder's 80 ms input-pacing tolerance. The experiment uses concurrency 1, seed 42, default LM inference and the opt-in [Mimi encoder CUDA graph](personaplex_mimi_encode_graph.md).

## Configuration and coverage

Serving source is `5a37be84d797719a43daeb97b9c002db7d223eca`. The RunPod H100 PCIe 80GB costs $2.89/hour. Serving uses a CPU thread budget of 16. Serving uses SGLang 0.5.22, PyTorch 2.14.1+cu130 and NVIDIA driver 595.91.07 from image `hongccc/sglang-omni@sha256:06ed306707bcf3d9e9beb5ba7ce66f09a666cac11c23455ece50c34199c0e985`.

The PersonaPlex checkpoint is `nvidia/personaplex-7b-v1` at `fdaf4090a61cb315c138a1faee287ffd6c716309`, with voice `NATF2` and prompt “You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.” Sampling uses text temperature 0.7/top-k 25 and audio temperature 0.8/top-k 250. The server sets `--lm.engine.enable_deterministic_inference false` and `--mimi_encode.factory.enable_cuda_graph true`. Input uses mono 24 kHz PCM16 and 80 ms units.

| Release | Subset | Samples | Calls |
|---|---|---:|---:|
| v1.5 | User interruption | 200 | 400 |
| v1.5 | User backchannel | 98 | 196 |
| v1.5 | Talking to other | 100 | 200 |
| v1.5 | Background speech | 100 | 200 |
| v1.0 | Synthetic pause handling | 137 | 137 |
| v1.0 | CANDOR pause handling | 216 | 216 |
| v1.0 | CANDOR turn taking | 119 | 119 |
| v1.0 | Synthetic user interruption | 200 | 200 |
| v1.0 | ICC backchannel | 55 | 55 |
| All | | 1,225 | 1,723 |

The v1.5 release contains 98 backchannel samples; its upstream declaration lists 99. The inventory records the released files and archive hashes. The longest individual input is 94 seconds. A 44.672-second ICC warmup precedes each release's recording batch.

## Default inference and repeatability

The same-host preflight runs the existing 39 selected calls with eager and graph encoders, using default LM inference and the quality sampling settings above. All 78 recordings pass protocol validation. Every eager/graph pair has byte-identical complete PCM and identical text. The full run reproduces the same complete PCM and text for all 39 shared inputs.

| Preflight encoder | Median first audio | Median post-input drain | Maximum drain |
|---|---:|---:|---:|
| Eager | 122.9 ms | 2,459.1 ms | 8,693.1 ms |
| CUDA graph | 93.6 ms | 72.1 ms | 76.6 ms |

The first eager preflight attempt completed its warmup, then stopped before measured calls because the selected-input directory lacked release revision files. Copying those files from the validated full dataset enabled both complete preflight runs. The retained first warmup emitted audio after 12.69 seconds. The full run's two warmups emitted audio after 848 and 94 ms and passed protocol validation. The initial cold-start delay remains under investigation.

## Full-run latency and stability

| Metric across 1,723 calls | Median | P95 | Maximum |
|---|---:|---:|---:|
| First audio packet | 92.9 ms | 107.7 ms | 153.9 ms |
| Post-input drain | 71.7 ms | 82.1 ms | 148.1 ms |
| Per-call maximum audio packet gap | 123.3 ms | 163.3 ms | 200.5 ms |
| Close acknowledgement | 16.9 ms | 20.0 ms | 23.0 ms |

The server remains live across both releases. Five-second monitoring covers 7.55 hours and 5,438 GPU samples. Median GPU memory use is 26,081 MiB, with a 26,543 MiB maximum. From ten minutes after each process starts to its final sample, the server's RSS increases by 2.03 MiB and its two main workers increase by 1.38 and 2.48 MiB. The remaining large worker stays at 688.64 MiB. One main worker reaches 11,303.69 MiB during startup and ends at 3,126.39 MiB.

## Evaluation

Reference export qualifies all 498 v1.5 pairs and all 727 v1.0 calls. Scoring audio uses FIFO receipt-time playout within `[0, T_input]`, resampled to mono 16 kHz PCM16. Complete generated PCM is retained separately for equality checks.

The evaluator is pinned at `3e799c45a045256f47d5f1c9cda90157e2d2ec9e`. ASR uses Parakeet TDT 0.6B v2 at `ae9ad07059c7c739ffaf932226a8fe64ae2620b0`; timing uses Silero VAD 6.2.1. The custom behavior judge is Qwen3.8-27B at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, BF16, thinking disabled, temperature zero and a 512-token output limit. Scoring uses PyTorch/torchaudio 2.11.0+cu130 and NeMo 3.0.0.

All 1,992 v1.5 audio files and 727 v1.0 outputs complete ASR. One clean output has an empty transcript, retained in the ASR ledger. All 996 v1.5 windows produce timing results. All 498 pairs receive a valid custom Qwen label on their first attempt.

### v1.5 timing

The table pools the pinned evaluator's intervals for the overlap variants. Counts in parentheses are intervals; a recording can contribute several. These are campaign reductions of the per-sample intervals. The report retains medians and 2,000 sample-bootstrap replicates with seed 20260925.

| Subset | Mean stop latency, seconds (intervals) | Mean response latency, seconds (intervals) |
|---|---:|---:|
| User interruption | 0.622 (399) | 0.433 (354) |
| User backchannel | 0.580 (99) | 0.590 (143) |
| Talking to other | 0.581 (82) | 0.342 (194) |
| Background speech | 0.623 (77) | 0.368 (191) |
| All | 0.611 (657) | 0.425 (882) |

The pooled mean 95% intervals are 0.595–0.628 seconds for stop and 0.399–0.455 seconds for response. Clean variants have mean stop latency 0.621 seconds across 183 intervals and mean response latency 0.407 seconds across 482 intervals. Model speech reaches the scoring-window end in 350 overlap and 222 clean recordings. The user-ending-without-a-later-model-start label occurs in 99 overlap and 17 clean recordings. Zero stop intervals occur in 48 overlap and 331 clean recordings; zero response intervals occur in two overlap and 17 clean recordings. These labels remain alongside the latency statistics.

### v1.5 behavior

The custom judge classifies paired clean/overlap transcripts with the reference behavior prompt. Counts below use all released pairs; the report includes per-label Wilson 95% intervals.

| Subset | Pairs | Respond | Resume | Uncertain handling | Unknown |
|---|---:|---:|---:|---:|---:|
| User interruption | 200 | 181 | 17 | 2 | 0 |
| User backchannel | 98 | 8 | 78 | 3 | 9 |
| Talking to other | 100 | 80 | 14 | 1 | 5 |
| Background speech | 100 | 63 | 20 | 5 | 12 |
| All | 498 | 332 | 129 | 11 | 26 |

### v1.0 reference metrics

All five released subsets complete the pinned reference evaluator. The interruption relevance rating uses the 187 model responses among 200 evaluated inputs; Qwen handles all 187 requests without retries. ICC values are the reference evaluator's mean ± standard deviation.

| Subset | Evaluated | Reference results |
|---|---:|---|
| Synthetic pause handling | 137 | Average take turn: 0.7664 |
| CANDOR pause handling | 216 | Average take turn: 0.6713 |
| CANDOR turn taking | 119 | Average take turn: 0.9916; average latency: 0.1582 s |
| Synthetic user interruption | 200 | Average take turn: 0.9350; average latency: 0.2409 s; average rating: 4.5936/5 |
| ICC backchannel | 55 | JSD: 0.7080 ± 0.0621; TOR: 0.4909 ± 0.4999; frequency: 0.1068 ± 0.0690 |

### v1.0 unified metrics

The unified scorer measures pause intrusion in 69/137 synthetic pause calls and 141/216 CANDOR pause calls. CANDOR turn-taking has 70/119 early starts and 42/119 responses within one second of the reference endpoint. Of 200 interruption calls, 179 exercise interruption while the model is speaking; all 179 stop, with median stop latency 0.808 seconds and P90 1.320 seconds. Response within one second occurs in 129/179 exercised calls. The machine-readable results retain the distinct event definitions, denominators, censoring and confidence intervals.

## Evidence and scope

The local experiment directory is `results/p0-4-full/`. It retains the dataset inventory, source and model revisions, launch commands, complete recordings, scoring receipts, judge exchanges, resource samples and both 39-call comparisons. Generated JSON and audio are experiment artifacts. The measured configuration covers one fixed seed and concurrency 1 on this H100 host.

The complete evidence archive is `full-fdb-evidence.tar.gz`, SHA-256 `d84f6e774dd63c6fd38e3c5e191b0d48a7754668bee7d56d7c49bd303c7d756d`. The compact `full-fdb-report.zip` contains `full-fdb-report.json`, its source JSON bundle, a standard-library rebuild script and checksums. Its README documents offline reconstruction and byte comparison of the report.
