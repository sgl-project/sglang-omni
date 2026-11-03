# PersonaPlex single-call quality baseline

This experiment implements P0-4 of [the PersonaPlex serving roadmap](https://github.com/sgl-project/sglang-omni/issues/3083). It records three samples from each FDB subset at concurrency 1 and compares a greedy realtime reply with the offline reply to the same transmitted PCM.

## Configuration

The run used one RunPod H100 80 GB on 2026-11-03 UTC. Serving ran SGLang 0.5.22, PyTorch 2.14.1+cu130 and NVIDIA driver 595.91.07 from the repository's pinned CI image, `hongccc/sglang-omni@sha256:06ed306707bcf3d9e9beb5ba7ce66f09a666cac11c23455ece50c34199c0e985`.

| Input | Value |
|---|---|
| Serving source | `be967415da712799707e8988a0dbdd2291e17bd1` |
| Benchmark implementation | `b4c8d672ed931e0f58544c279234622db65f7aa6` |
| PersonaPlex checkpoint | `nvidia/personaplex-7b-v1`, revision `fdaf4090a61cb315c138a1faee287ffd6c716309` |
| Voice | `NATF2` |
| Role prompt | You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way. |
| Quality sampling | Default text temperature 0.7/top-k 25; audio temperature 0.8/top-k 250 |
| Greedy sampling | Text and audio temperature 0 |
| Session capacity | 1 |
| Input/output | Mono 24 kHz PCM16, 80 ms input units |
| FDB evaluator | `3e799c45a045256f47d5f1c9cda90157e2d2ec9e` |
| ASR | `nvidia/parakeet-tdt-0.6b-v2`, revision `ae9ad07059c7c739ffaf932226a8fe64ae2620b0` |
| Behavior judge | `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, BF16 |

FDB v1.5 uses sample IDs 1, 2 and 3 in each of `user_interruption`, `user_backchannel`, `talking_to_other` and `background_speech`: 12 pairs and 24 calls. FDB v1.0 uses three samples from each of its five subsets: 15 calls. A warmup call precedes each recording batch. The retained run manifests identify every source file and its SHA-256.

## Capture and export

All 24 v1.5 calls pass protocol replay with integer PCM padding accounting. The initial replay marked 15 calls as failing because it calculated padding by subtracting floating-point millisecond durations. For example, a 349,640-sample input at 24 kHz requires 1,720 padding samples, or 71.66666666666667 ms. Computing the remainder in PCM bytes removes the subtraction error. The corrected replay is saved separately from the original recordings and reports.

All 15 v1.0 calls pass protocol replay. Reference export produces 24 eligible v1.5 variants and 15 eligible v1.0 recordings. It verifies the original 24 kHz PCM and packet payloads before resampling scoring audio to 16 kHz. The output window follows receipt-time FIFO playout over the transmitted input duration.

An input ending between scoring samples retains the last partial sample through ceiling division. The v1.0 unified scorer checks the WAV length against the original integer input sample count and sample rate; rounding the microsecond timestamp rejected one of the exported interruption recordings.

## Quality scores

Scoring used PyTorch/torchaudio 2.11.0+cu130, NeMo 3.0.0 and Silero VAD 6.2.1. All 48 v1.5 audio files and 15 v1.0 output files were transcribed successfully. All 24 v1.5 variants produced timing results, and all 12 pairs received valid Qwen behavior labels. The [machine-readable results](personaplex_quality_baseline.json) retain the metrics, denominators, confidence intervals and artifact hashes.

### FDB v1.0

Each row contains three selected and evaluated recordings. These are the pinned reference evaluator's metric names and reductions. Qwen3.8-27B supplies the interruption relevance rating, with thinking disabled and temperature zero.

| Subset | Reference results |
|---|---|
| Synthetic pause handling | Average take turn: 0.3333 |
| CANDOR pause handling | Average take turn: 1.0000 |
| CANDOR turn taking | Average take turn: 1.0000; average latency: 0.1200 s |
| Synthetic user interruption | Average take turn: 1.0000; average latency: 0.2400 s; average rating: 5.0/5 |
| ICC backchannel | JSD: 0.7477 ± 0.0353; TOR: 0.6667 ± 0.4714; frequency: 0.0665 ± 0.0302 |

The ICC values show the reference evaluator's mean ± standard deviation. The unified v1.0 metrics additionally count 1/3 CANDOR calls starting early, 2/3 interruption calls responding within one second, and an interruption stop-latency median of 0.6587 s. The unified metrics use their own event definitions; the JSON includes both sets of results.

### FDB v1.5

The timing table pools all intervals emitted by the pinned evaluator for the overlap variants. Each subset has three pairs. Counts in parentheses are timing intervals; a recording can contribute multiple intervals.

| Subset | Mean stop latency, seconds (intervals) | Mean response latency, seconds (intervals) |
|---|---|---|
| User interruption | 0.5400 (5) | 0.3133 (6) |
| User backchannel | 0.6680 (5) | 0.5544 (5) |
| Talking to other | 0.5000 (4) | 0.3080 (6) |
| Background speech | 0.3640 (2) | 0.2493 (6) |
| All | 0.5480 (16) | 0.3477 (23) |

Qwen3.8-27B classifies paired clean/overlap transcripts using the reference behavior prompt. The following custom-judge results are counts among three valid labels per subset, with temperature zero, thinking disabled and a 512-token output limit.

| Subset | Respond | Resume | Uncertain handling | Unknown |
|---|---|---|---|---|
| User interruption | 2 | 1 | 0 | 0 |
| User backchannel | 0 | 3 | 0 | 0 |
| Talking to other | 2 | 1 | 0 | 0 |
| Background speech | 2 | 0 | 1 | 0 |
| All | 6 | 5 | 1 | 0 |

## Greedy comparison

The input is FDB v1.5 `user_interruption/1/input.wav`. The recorder resamples it to 24 kHz; the offline request receives that exact PCM in a WAV data URI.

| Comparison | Result |
|---|---|
| Cold session vs. warm session | Byte-identical, 353,280 samples each |
| Session vs. offline, complete reply | Different; first differing sample is 1 (zero-based) |
| Session prefix over caller duration vs. offline | Different |
| Offline reply length | 349,632 samples |
| Session excess length | 3,648 samples: one initial frame plus final unit padding |

The session PCM SHA-256 is `d8163b0b05b3c463a0f6dc252bf435428af3b272922505c79de4498bdf5bfa08`. The offline PCM SHA-256 is `2c0c6829d44df2541bf67fc2aa4e4cef103ffbfe0953f6bd1dfe3edb3ec32aec`. The text replies also differ. The byte-identical session/offline acceptance condition is currently failing.

The serving paths use different PCM quantization: the session's `waveform_to_pcm16` rounds scaled samples, while the offline WAV encoder truncates them. This can produce one-unit differences even for identical floating-point samples.

A component experiment on the same padded waveform compares Mimi whole-waveform encoding with successive 80 ms encode steps. It finds 19 differing codes among 1,464 codes, first at frame 65 (5.2 seconds). The saved code tensors and differing positions provide a separate input to investigating the serving-path mismatch.

## Reproduction

The [PersonaPlex cookbook](../cookbook/personaplex.md#quality-baseline) provides the recording, export and greedy comparison commands. The comparison command writes the original offline request and response, both audio outputs, PCM hashes and the first differing sample to its output directory. It exits with status 1 for the session/offline result above.
