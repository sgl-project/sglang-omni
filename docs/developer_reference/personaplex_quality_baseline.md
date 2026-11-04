# PersonaPlex single-call quality baseline

This experiment implements P0-4 of [the PersonaPlex serving roadmap](https://github.com/sgl-project/sglang-omni/issues/3083). It records three samples from each FDB subset at concurrency 1 and compares a greedy realtime reply with the offline reply to the same transmitted PCM.

The [expanded experiment](personaplex_expanded_baseline.md) verifies complete PCM and text equality on all 39 selected calls across nine subsets and records the performance cost and new quality scores. The initial [session reference experiment](#full-reply-consistency) covers two FDB inputs, a repeated call and two short frame-boundary cases. The original quality baselines below retain their serving configurations and comparison results.

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

Scoring used PyTorch/torchaudio 2.11.0+cu130, NeMo 3.0.0 and Silero VAD 6.2.1. All 48 v1.5 audio files and 15 v1.0 output files were transcribed successfully. All 24 v1.5 variants produced timing results, and all 12 pairs received valid Qwen behavior labels. The machine-readable results (`personaplex_quality_baseline.json`) retain the metrics, denominators, confidence intervals and artifact hashes.

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

The session PCM SHA-256 is `d8163b0b05b3c463a0f6dc252bf435428af3b272922505c79de4498bdf5bfa08`. The offline PCM SHA-256 is `2c0c6829d44df2541bf67fc2aa4e4cef103ffbfe0953f6bd1dfe3edb3ec32aec`. The text replies also differ. The byte-identical session/offline acceptance condition fails in this original baseline.

The original serving paths used different PCM quantization: the session's `waveform_to_pcm16` rounded scaled samples, while the offline WAV encoder truncated them. This produced one-unit differences even for identical floating-point samples. The follow-up below unifies that conversion.

A component experiment on the same padded waveform compares Mimi whole-waveform encoding with successive 80 ms encode steps. It finds 19 differing codes among 1,464 codes, first at frame 65 (5.2 seconds). The saved code tensors and differing positions provide a separate input to investigating the serving-path mismatch.

## Reproduction

The [PersonaPlex cookbook](../cookbook/personaplex.md#quality-baseline) provides the recording, export and greedy comparison commands. The comparison command writes the original offline request and response, both audio outputs, PCM hashes and the first differing sample to its output directory. It exits with status 1 for the session/offline result above.

### Saved report files

The four complete report JSONs and three reconstruction input bundles are packaged as `personaplex-p0-4-reports-20261104-v1.zip` for attachment to [PR #3469](https://github.com/sgl-project/sglang-omni/pull/3469). The ZIP has SHA-256 `6084cec44de9d69b03113de2ad417c12afb6eb95d41d61ff8c960b1a6b1b2ec5`. [Report checksums](personaplex_reports.sha256) identify the individual JSON files. Extract the attachment from the repository root:

```bash
mkdir -p results/p0-4-reports
unzip personaplex-p0-4-reports-20261104-v1.zip -d results/p0-4-reports
(cd results/p0-4-reports && shasum -a 256 -c ../../docs/developer_reference/personaplex_reports.sha256)
```

The attachment's `README.md` contains reconstruction commands for its `inputs/` bundles. The commands below use the original local artifact directories.

### Rebuilding the retained report

The report generator reads the saved scoring and comparison results without starting a model or contacting a service. Experiment conditions are recorded in `benchmarks/duplex/personaplex-baseline-conditions.json`; the generator requires the hardware, serving, generation, selection and evaluation identities. The original run used operator-recorded hardware and launch settings, with source manifests and evaluator receipts retained alongside the measurements.

```bash
python -m benchmarks.eval.benchmark_personaplex_report \
  --artifacts results/p0-4 \
  --conditions benchmarks/duplex/personaplex-baseline-conditions.json \
  --output results/p0-4-reports/personaplex_quality_baseline.json \
  --bundle results/p0-4-followup/report-inputs.tar.gz
python -m benchmarks.eval.benchmark_personaplex_report \
  --artifacts results/p0-4 \
  --conditions benchmarks/duplex/personaplex-baseline-conditions.json \
  --output results/p0-4-reports/personaplex_quality_baseline.json --check
```

The portable input bundle contains the exact source JSON files and `conditions.json`. When the artifact directory contains `diagnostics.json`, the report and bundle also include those control experiments. Extract the bundle into an empty directory, then pass that directory as `--artifacts` and its `conditions.json` as `--conditions`. The bundle and full recording archives are retained locally with this experiment.

`--check` reconstructs the complete JSON and compares parsed values, ignoring key order and whitespace. It checks source hashes as well as measurements, so replacing a source file invalidates the report even when its selected metric values happen to match. It verifies reconstruction of the same saved experiment. A new GPU recording produces a new report.

For comparisons between experiments, match the model/checkpoint, voice, prompt, sampling, dataset selection, sample rate, concurrency, evaluator identities and reconstruction policy first. Compare quality metrics together with their sample/interval counts and uncertainty, and compare greedy PCM/code equality exactly. Hardware and serving versions identify intentional changes to the experiment. Receipt timestamps, local paths, endpoint addresses and process IDs describe where and when an experiment ran; they are informational across runs. Archive and source hashes identify the evidence and are expected to change when a new experiment is recorded.

## Follow-up: locating the session/offline differences

A second H100 run on 2026-11-04 used the same GPU model, driver, image and checkpoint. Capturing intermediate tensors reproduced both original PCM hashes exactly. Diagnostics were written to the instance's local disk; the first attempt on the network volume timed out and its failed capture was retained separately.

| Layer | Observation, zero-based indices |
|---|---|
| Prepared caller waveform | Identical, including padding: 351,360 float samples |
| Prompt tokens | Identical: 90 × 17 |
| Voice prompt embeddings | Identical: 51 × 4,096 |
| Mimi caller codes | 19 different entries; first at frame 65, codebook 3 |
| Generated agent codes | First difference at frame 45, codebook 7 |
| Generated text tokens | First difference at frame 53 |

The first generated-code difference precedes the caller-code difference. A control run supplied the session's exact caller codes to the offline LM. The generated-code and text-token differences still began at frames 45 and 53. Actual LM input embeddings matched through the first 45 steps, while hidden states first differed at step 39. The session executes subsequent units as retained-history prefills; offline executes decode steps. This places the first observed generation divergence in LM execution before token selection, with matching caller codes and prompt tensors. The subsequent attention controls below locate the numerical difference.

The decoder also produces different floating-point samples while the generated code prefix still matches. Over the first 45 generated frames, the maximum absolute waveform difference was 0.0006243438. Applying the same truncation rule to both waveforms leaves the first PCM difference at sample 9, with a maximum difference of 20 PCM units in that prefix.

A decoder-only control supplies the same 184 generated code frames to both chunk schedules: two frames then one at a time for session, and one frame then groups of four for offline. Among 353,280 output samples, 119,019 PCM values differ, starting at sample 9, with a maximum difference of 21 PCM units. Repeating both schedules with CUDA graphs disabled produces exactly the same output as each schedule with graphs enabled. This isolates the decoder difference to chunk-dependent numerical computation in this experiment.

The PCM fix uses the offline encoder's truncation rule for session output. A fresh greedy session on the fixed code passes protocol replay and exactly matches truncating the captured original session waveform. Its PCM SHA-256 is `b5a516f6962d3dca08a88cd74d3e6c289e77eeef8643314a12d4824d0137912c`; its first difference from the unchanged offline output moves from sample 1 to sample 9.

The length difference follows the existing request contracts: session executes one extra LM step and emits all 184 frames, while offline generates 183 frames and trims to the caller's 349,632 samples. Full reply equality is false in this follow-up. The session reference mode below aligns the output-length contract and numerical execution of the LM and codec.

## Full-reply consistency

On 2026-11-04, a third H100 experiment achieved complete PCM byte equality and exact text equality with offline `session_reference` enabled and SGLang deterministic inference enabled on both routes. The consistency JSON (`personaplex_consistency.json`) records the runtime, source hashes, controls and individual comparisons. The serving files match local implementation commit `8a7df9f46`; the launch receipts retain the base revision plus the exact candidate patch.

### Attention controls

Two controls use the session's captured caller codes to isolate the LM. Disabling the offline LM CUDA graph makes all 183 common frames of input embeddings, hidden states, generated audio codes and text tokens exactly equal to the session. A second control retains CUDA graphs and changes only FlashAttention's `num_splits` from automatic selection (`0`) to `1`; it produces the same exact equality across all 183 common frames. The session has one additional frame under its output contract. These controls locate the observed LM divergence in attention split selection in this environment.

SGLang's supported `enable_deterministic_inference` setting selects a fixed attention split and batch-invariant operations. The end-to-end runs use that setting with the original SGLang backend restored; its SHA-256 is `efa751c504ec92fbaebd376f38a33b715b3e650df00516e11fde2ed7c1023b02`. Offline decode retains CUDA graphs. Both pipelines independently encode the actual caller waveform and run the LM.

### Reference mode and measured results

The new offline mode encodes caller audio in 80 ms chunks, generates the session's extra initial frame, decodes two frames first and one frame per chunk afterward, and returns all generated samples. The realtime first-packet behavior and default offline output-length contract keep their existing definitions. The [cookbook commands](../cookbook/personaplex.md#greedy-session-and-offline-comparison) reproduce the launch and strict comparison.

| Case | Transmitted input samples | Complete output samples, each route | PCM bytes | Text |
|---|---:|---:|---|---|
| FDB v1.5 `user_interruption/1` | 349,632 | 353,280 | Identical | Identical |
| Same input, warm repeat | 349,632 | 353,280 | Identical | Identical |
| FDB v1.5 `user_interruption/2` | 316,353 | 318,720 | Identical | Identical |
| One 80 ms input frame | 1,920 | 3,840 | Identical | Identical, empty |
| Partial second input frame | 1,940 | 5,760 | Identical | Identical, empty |

All samples are mono 24 kHz PCM16. The first input and its repeat produce SHA-256 `824015250f0901a2cf0e2111589afefb84706a51d828a6a26fac636220587c11` on both routes. The second FDB input produces `a42ecca4d01491aab19952566b4f2836e0bde2e67594fa36f6d44aabe7ba95b9`. Every comparison includes the complete output, with zero differing bytes and equal lengths.

The three full-length calls pass protocol replay. The two short calls have zero protocol violations and are marked `not_exercised` for input/output overlap coverage; their separately recorded boundary comparisons establish output length and byte equality. The tested scope is greedy generation, concurrency 1, this pinned checkpoint and this H100 runtime. Full-dataset coverage and quality scoring under the deterministic configuration are subsequent experiments; the earlier 39-call quality measurements below retain their original sampling and serving settings.

### Evidence and validation

The local archive `results/p0-4-parity/parity-evidence.tar.gz` has SHA-256 `b3d5314cf67892a606769454b8adae8b4a9390128dd3eb5e1a8b6bb7e91b3d17`. It contains 2,069 entries with no symlinks: both diagnostic tensor captures, the reference capture, server logs, five session recordings, all requests and responses, comparisons, launch scripts and the exact serving patch. Extract it with `tar -xzf results/p0-4-parity/parity-evidence.tar.gz -C results/p0-4-parity`; the summary is `parity/consistency-summary.json` and the end-to-end artifacts are under `parity/final/`.

Validation passed 90 configuration, stage, request-builder and LM-session tests, seven session-hook tests using the real Mimi codec, and 66 benchmark report, profile and artifact tests. Repository pre-commit checks pass. The regression cases cover the extra frame's context budget, matching LM inputs from the same caller frames, per-request encoder state, codec chunk scheduling and the recorded comparison mode. The evidence archive was verified locally before the RunPod H100 instance was deleted; a subsequent API query returned no instance.

## Follow-up quality run

The 2026-11-04 run uses serving revision `377a86fb97eb9d1a57eb982dd0bee3c75022ab46`, which changes session PCM quantization to truncation. All 24 v1.5 calls and 15 v1.0 calls pass protocol replay and reference export. All 63 ASR files, 24 v1.5 timing evaluations, 12 paired behavior labels and five v1.0 subset evaluations complete successfully. The follow-up JSON (`personaplex_quality_followup.json`) records the scores, conditions, layer comparisons and source hashes separately from the original baseline.

Both runs use the same sample IDs, model checkpoints, prompts, sampling settings and reference evaluator revision. Quality generation uses default sampling without an explicit seed. The OpenAI client package is 3.45.0 in the follow-up and 3.42.1 in the original run; the retained scoring environments record the complete package versions. The score differences describe these two small runs. The greedy waveform comparison above isolates the PCM conversion change.

### FDB v1.0 comparison

Each subset contains three evaluated recordings. ICC values are reference mean ± standard deviation.

| Subset and metric | Original | Follow-up |
|---|---|---|
| Synthetic pause: take turn | 0.3333 | 0.6667 |
| CANDOR pause: take turn | 1.0000 | 0.6667 |
| CANDOR turn taking: take turn | 1.0000 | 1.0000 |
| CANDOR turn taking: latency, s | 0.1200 | 0.0400 |
| User interruption: take turn | 1.0000 | 1.0000 |
| User interruption: latency, s | 0.2400 | 0.2667 |
| User interruption: relevance rating | 5.0/5 | 5.0/5 |
| ICC: JSD | 0.7477 ± 0.0353 | 0.7295 ± 0.0520 |
| ICC: TOR | 0.6667 ± 0.4714 | 1.0000 ± 0.0000 |
| ICC: frequency | 0.0665 ± 0.0302 | 0.0696 ± 0.0191 |

The follow-up unified interruption metrics count 2/3 responses within one second and a stop-latency median of 0.5627 s, compared with 2/3 and 0.6587 s in the original run.

### FDB v1.5 comparison

Timing values are pooled mean seconds over overlap intervals, with interval counts in parentheses. Each subset contains three pairs.

| Subset | Original stop | Follow-up stop | Original response | Follow-up response |
|---|---|---|---|---|
| User interruption | 0.5400 (5) | 0.5144 (5) | 0.3133 (6) | 0.2920 (6) |
| User backchannel | 0.6680 (5) | 0.6680 (5) | 0.5544 (5) | 0.5608 (5) |
| Talking to other | 0.5000 (4) | 0.5160 (4) | 0.3080 (6) | 0.3187 (6) |
| Background speech | 0.3640 (2) | 0.3800 (2) | 0.2493 (6) | 0.2493 (6) |
| All | 0.5480 (16) | 0.5460 (16) | 0.3477 (23) | 0.3463 (23) |

All per-subset Qwen behavior counts match the original table: six respond, five resume, one uncertain handling and zero unknown across 12 valid labels.

### Follow-up report reconstruction

The complete local evidence archive is `results/p0-4-followup/followup-evidence.tar.gz`, SHA-256 `39faec4c6d7337e9b64bb7d1ef70850c7c268013c2a04e10a8da061c5890f6ad`. It contains recordings, scores, diagnostic tensors, temporary capture scripts and the serving source used in the experiment. The smaller `followup-report-inputs.tar.gz` contains the JSON inputs needed to reconstruct the saved follow-up report.

```bash
mkdir -p /tmp/personaplex-followup-report
tar -xzf results/p0-4-followup/followup-report-inputs.tar.gz \
  -C /tmp/personaplex-followup-report
python -m benchmarks.eval.benchmark_personaplex_report \
  --artifacts /tmp/personaplex-followup-report \
  --conditions /tmp/personaplex-followup-report/conditions.json \
  --output results/p0-4-reports/personaplex_quality_followup.json --check
```

The fixed greedy recording's original manifest retains its pre-fix source label. The follow-up conditions record the actual serving revision, matching source-file hashes and the saved offline comparison source. The new quality recordings carry the fixed serving revision directly. The parity CLI also now passes explicit identity fields to `server_identity`; the earlier call failed before sending the offline request.

Validation includes 262 passing benchmark tests, three passing session-hook checks in the serving image, both report bundles rebuilt in empty directories, and repository pre-commit checks. The eight reference-checkout tests initially skipped locally were run with the retained pinned evaluator files. Strict byte equality remains a failing acceptance condition in both saved reports.
