# PersonaPlex full FDB baseline

The 2026-11-05 experiment completes all 1,723 released FDB calls, covering 7.316 hours of audio at concurrency 1 and seed 42. The [Mimi encoder CUDA graph added in this work](personaplex_mimi_encode_graph.md) reduces median post-input drain from 2.46 seconds to 72 ms in the same-host 39-call comparison, with identical PCM and text. The full run measures its latency, stability and model behavior with default LM inference.

## Configuration and coverage

Serving source is `5a37be84d797719a43daeb97b9c002db7d223eca`. The H100 PCIe 80GB host uses a 16-thread CPU budget, SGLang 0.5.22, PyTorch 2.14.1+cu130 and driver 595.91.07. The image is `hongccc/sglang-omni@sha256:06ed306707bcf3d9e9beb5ba7ce66f09a666cac11c23455ece50c34199c0e985`.

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

The v1.5 backchannel archive contains 98 samples; the upstream declaration lists 99. The longest input is 94 seconds. Each release's batch starts with a 44.672-second ICC warmup.

## Default inference and repeatability

The preflight changes only the encoder mode on the same host. All 78 calls pass protocol validation, and all 39 pairs have identical complete PCM and text. The full run reproduces those outputs for all 39 shared inputs.

| Preflight encoder | Median first audio | Median post-input drain | Maximum drain |
|---|---:|---:|---:|
| Eager | 122.9 ms | 2,459.1 ms | 8,693.1 ms |
| CUDA graph | 93.6 ms | 72.1 ms | 76.6 ms |

An initial setup attempt stopped before measured calls because release revision files were missing; copying them fixed setup. Its warmup first packet took 12.69 seconds. The full-run warmups took 848 and 94 ms. Cold-start latency remains under investigation.

## Full-run latency and stability

| Metric across 1,723 calls | Median | P95 | Maximum |
|---|---:|---:|---:|
| First audio packet | 92.9 ms | 107.7 ms | 153.9 ms |
| Post-input drain | 71.7 ms | 82.1 ms | 148.1 ms |
| Per-call maximum audio packet gap | 123.3 ms | 163.3 ms | 200.5 ms |
| Close acknowledgement | 16.9 ms | 20.0 ms | 23.0 ms |

All 1,723 calls pass protocol validation, exercise simultaneous input/output and meet the 80 ms input-pacing tolerance. The server stays live across both releases. Monitoring covers 7.55 hours at five-second intervals: GPU memory has a median of 26,081 MiB and maximum of 26,543 MiB. After the first ten minutes, RSS grows by 2.03 MiB in the server and 1.38/2.48 MiB in its main workers. The report includes startup peaks and per-process samples.

## Evaluation

All 498 v1.5 pairs and 727 v1.0 calls qualify for scoring. Scoring uses FIFO receipt-time playout within `[0, T_input]`, resampled to mono 16 kHz PCM16; equality checks use complete generated PCM.

The evaluator is pinned at `3e799c45a045256f47d5f1c9cda90157e2d2ec9e`. ASR uses Parakeet TDT 0.6B v2 at `ae9ad07059c7c739ffaf932226a8fe64ae2620b0`; timing uses Silero VAD 6.2.1. The custom behavior judge is Qwen3.8-27B at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, BF16, thinking disabled, temperature zero and a 512-token output limit. Scoring uses PyTorch/torchaudio 2.11.0+cu130 and NeMo 3.0.0.

ASR completes all 2,719 files, including one empty clean-output transcript. All 996 v1.5 windows produce timing results, and all 498 pairs receive a valid Qwen label on the first attempt.

### v1.5 timing

The table pools overlap intervals from the reference evaluator; parentheses show interval counts. The report includes medians and 2,000 sample-bootstrap replicates with seed 20260925.

| Subset | Mean stop latency, seconds (intervals) | Mean response latency, seconds (intervals) |
|---|---:|---:|
| User interruption | 0.622 (399) | 0.433 (354) |
| User backchannel | 0.580 (99) | 0.590 (143) |
| Talking to other | 0.581 (82) | 0.342 (194) |
| Background speech | 0.623 (77) | 0.368 (191) |
| All | 0.611 (657) | 0.425 (882) |

The pooled mean 95% intervals are 0.595–0.628 seconds for stop and 0.399–0.455 seconds for response. Clean means are 0.621 seconds for stop (183 intervals) and 0.407 seconds for response (482 intervals).

| Window label | Overlap | Clean |
|---|---:|---:|
| Model speech at window end | 350 | 222 |
| User ends without a later model start | 99 | 17 |
| No stop interval | 48 | 331 |
| No response interval | 2 | 17 |

### v1.5 behavior

Qwen classifies paired transcripts with the reference behavior prompt. The report includes Wilson 95% intervals for each label.

| Subset | Pairs | Respond | Resume | Uncertain handling | Unknown |
|---|---:|---:|---:|---:|---:|
| User interruption | 200 | 181 | 17 | 2 | 0 |
| User backchannel | 98 | 8 | 78 | 3 | 9 |
| Talking to other | 100 | 80 | 14 | 1 | 5 |
| Background speech | 100 | 63 | 20 | 5 | 12 |
| All | 498 | 332 | 129 | 11 | 26 |

### v1.0 reference metrics

The interruption rating averages 187 model responses among 200 evaluated inputs. ICC values are mean ± standard deviation.

| Subset | Evaluated | Reference results |
|---|---:|---|
| Synthetic pause handling | 137 | Average take turn: 0.7664 |
| CANDOR pause handling | 216 | Average take turn: 0.6713 |
| CANDOR turn taking | 119 | Average take turn: 0.9916; average latency: 0.1582 s |
| Synthetic user interruption | 200 | Average take turn: 0.9350; average latency: 0.2409 s; average rating: 4.5936/5 |
| ICC backchannel | 55 | JSD: 0.7080 ± 0.0621; TOR: 0.4909 ± 0.4999; frequency: 0.1068 ± 0.0690 |

### v1.0 unified metrics

Pause intrusion occurs in 69/137 synthetic calls and 141/216 CANDOR calls. CANDOR turn-taking has 70/119 early starts and 42/119 responses within one second. Interruption is exercised in 179/200 calls; all 179 stop, with median latency 0.808 seconds and P90 1.320 seconds. Of those calls, 129/179 respond within one second. These measurements establish the model's behavior baseline for future turn-taking work.

## Evidence

`results/p0-4-full/` contains the inputs, revisions, commands, recordings, scoring receipts, judge exchanges and telemetry. JSON results include denominators, censoring and confidence intervals.

The raw archive is `full-fdb-evidence.tar.gz`, SHA-256 `d84f6e774dd63c6fd38e3c5e191b0d48a7754668bee7d56d7c49bd303c7d756d`. `full-fdb-report.zip` packages the report JSON, reconstruction inputs, script and checksums; its README gives rebuild commands.
