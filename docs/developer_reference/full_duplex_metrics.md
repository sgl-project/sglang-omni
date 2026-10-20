# Full-Duplex-Bench v1.0 unified metrics

`unified-score` scores a Full-Duplex-Bench v1.0 reference tree with turn-taking metrics that do not depend on how a system decides to speak. It reads the same Parakeet transcripts and dataset annotations as the official evaluator, and for user interruption it also runs Silero VAD on the exported audio, so native full-duplex, turn-based and cascade systems are scored by the same rules. It needs no GPU, LLM or network access.

## Why the official numbers are only a reference

`reference-evaluate` runs the pinned official evaluator unchanged, and its numbers stay in the report as the reference column. Four of its rules measure something other than their names suggest:

1. **Talking over the user gives the best latency.** `eval_smooth_turn_taking.py` sets a negative latency, a reply that starts before the user's turn ends, to 0. On one 119-sample capture of a native full-duplex model, 39 latencies were clamped this way.
2. **Pause takeover does not look at the pauses.** `eval_pause_handling.py` never reads `pause.json`. It counts a takeover in every nonempty transcript of the whole window except one that both spans less than 1 s and has at most 3 words. Speech before the user's first pause counts, and an output of at most 3 words within 1 s, such as a takeover that the user's resumed speech cuts off, does not. On Candor pause handling, the official rule found a takeover in 30 samples of a turn-based system that answers after 500 ms of silence, while that system's transcript contained a word onset at or after the first annotated pause in 67.
3. **Non-responders leave the latency average.** Latency is averaged only over samples that took the turn, so a model too slow to answer inside the window drops out and can look faster.
4. **Not every interruption interrupts.** `eval_user_interruption.py` scores every sample alike, whether or not the model was speaking when the user cut in. On one 200-sample capture of a native full-duplex model, no speech was detected at the interruption onset in 87 samples, and a reply after R in those samples still enters the official takeover rate, latency and judge rating.

## Inputs and checks

Run it on a tree written by `reference-export` and transcribed by `reference-asr` (see the [runbook](full_duplex_bench.md)). All times are seconds on the input window `[0, T]`, where T is the input length recorded in `manifest.json`. Each transcript word has a start and an end from `output.json`; a word with a null end counts as zero-length at its start. Times are rounded to the microsecond before any comparison. This removes float noise in the files (an annotation at 6.32000000000005 against a word at 6.32), not ASR alignment error.

Loading a sample checks its times and stops with the sample id and the problem if any check fails: every word starts in `[0, T]`; the reference times E, S and R defined below lie in `[0, T]` with `S <= R`; `pause.json` lists at least one pause, each `[a, b]` with `a <= b` inside `[0, T]`, sorted and not overlapping; and no interruption word starts before R.

These annotations do not reliably label backchannels versus aborted turn starts; we therefore count every word onset, including a one-word "yeah". The rules are a strict word-timing proxy that also penalizes acknowledgments a listener might accept.

## Metrics

### Pause handling

Subsets `synthetic_pause_handling` and `candor_pause_handling`, reported separately. The official evaluator scores the whole window as one user turn, and we operationally treat the window as a non-yielding user turn. From the start of the first annotated pause to T the user is pausing or speaking again, so a model word starting in that region takes the floor too early. Each pause owns the time from its start to the next pause start, and the last pause owns the time to T.

| Metric | Definition | Better |
|---|---|---|
| `pause_intrusion_rate` | Samples with a model word starting at or after the first pause start | Lower |
| `pause_intrusion_per_pause` | Pauses whose owned time contains a model word start, over all pauses | Lower |
| `intrusion_s_per_pause_s` | Annotated pause seconds covered by ASR word spans, over all annotated pause seconds | Lower |
| `inside_pause_onset_rate` | Samples with a model word starting inside an annotated pause `[a, b)` | Context |
| `pre_pause_speech_rate` | Samples with a model word starting before the first pause start | Context |
| `carry_in_rate` | Samples with a model word that starts before the first pause start and ends after it | Context |
| `any_speech_rate` | Samples with any model word in the window | Context |

`pause_intrusion_rate` is the headline. It counts onsets up to T, not only inside the pauses, because a model whose reaction time exceeds a pause replies after the user has resumed; this needs no added reaction-time parameter. `inside_pause_onset_rate` is the narrower count of onsets inside the pauses.

Onsets before the first pause are outside this pause-conditioned observation region: they cannot react to a pause that has not started. Their samples stay in the denominator, and a model still talking at the first pause counts once it starts another word. This is a scope restriction, not a judgment that early talk is harmless; `pre_pause_speech_rate` and `carry_in_rate` report it.

`intrusion_s_per_pause_s` measures coverage of annotated pause time by ASR word spans, not acoustic occupancy. Any word overlapping a pause adds that overlap, including one that started before the first pause, and a zero-length word adds nothing. `pause_intrusion_per_pause` is an owned-interval diagnostic, not a per-pause takeover probability: one continuing utterance can place words in several owned intervals, the intervals differ in length, and the last one runs to T. `intrusion_words` in `unified-samples.jsonl` counts the intruding word onsets, not utterances or seconds.

### Smooth turn-taking

Subset `candor_turn_taking`. The reference time E is `turn_taking.json[0].timestamp[0]`, the time the official evaluator uses as the end of the user's turn. Each sample has exactly one outcome:

- `early_start`: a model word starts before E.
- `response`: no word starts before E and at least one starts at or after it. The latency is the start of the first word minus E.
- `no_response`: no transcribed word in the window.

| Metric | Definition | Better |
|---|---|---|
| `early_start_rate` | `early_start` samples | Lower |
| `response_rate_1s`, `_2s`, `_5s` | `response` samples with a latency of at most 1, 2 or 5 s | Higher |
| `no_response_rate` | `no_response` samples | Lower |
| `response_latency_s` | Median with its interval, and p90, of the latency, where a sample without a `response` never responds | Lower |

An early start never counts as a response, so talking through the end of the user's turn cannot buy a short latency, and an early acknowledgment followed by a good answer is an early start. `no_response` means no transcribed word, not every failed sample, so a system can have no silent samples and still not reach its p90.

Let F(t) be the share of samples with a `response` latency of at most t. Then `response_rate_ks` is F(k), and a latency quantile is Q(q) = inf{t : F(t) >= q}, the ceil(qN)-th smallest latency with no interpolation. The quantiles and the response rates are one curve: Q(q) is at most k exactly when `response_rate_ks` is at least q, so the median is the time by which half of all samples have responded. A quantile that falls on a sample without a `response` is reported as not reached. Placing those samples at infinity is a scoring convention for a valid response within the window, not an estimate of when a silent system would answer.

### User interruption

Subset `synthetic_user_interruption`. `interrupt.json[0].timestamp` gives the onset S and the end R of the user's interruption. `reference-asr` cuts the audio at R before transcribing, so the transcript holds only words from R on and cannot show whether the model stopped. Silero VAD on the full `output.wav` supplies that (see [Speech detection](#speech-detection)). Each sample records whether detected speech covers S (`speech_at_onset`, a segment with start <= S < end) and whether speech that began before R carries over it (`speech_at_end`, start < R < end; a reply that begins exactly at R does not). The outcome is decided in this order:

- `not_exercised`: no detected speech at S, so the interruption did not cut into detected speech. The sample keeps `speech_at_end`, so a model that starts talking during the interruption and talks past R stays visible.
- `speaking_at_end`: speech at S and at R. The words after R may continue the interrupted answer, so the sample has no latency. A model that stopped and then started a reply before R lands here too, the interruption counterpart of an early start.
- `response`: speech at S, none carried over R, and a transcribed word starts at or after R. The latency is the start of the first word minus R.
- `no_response`: speech at S, none carried over R, and no transcribed word after R.

| Metric | Definition | Better |
|---|---|---|
| `exercised_rate` | Eligible samples with speech at S | Context |
| `speaking_at_end_rate` | Exercised samples that are `speaking_at_end` | Lower |
| `response_rate_1s`, `_2s`, `_5s` | Exercised samples with a `response` within 1, 2 or 5 s | Higher |
| `no_response_rate` | Exercised samples that are `no_response` | Lower |
| `response_latency_s` | Median with its interval, and p90, over exercised samples, as for turn-taking | Lower |
| `stop_latency_s` | Median with its interval, and p90, of the time from S to the end of the segment covering S; a segment that runs to T has no observed stop | Lower |
| `speech_s_per_interruption_s` | Detected speech seconds inside [S, R] over interruption seconds, pooled over exercised samples | Lower |
| `restart_rate` | Exercised samples with a segment starting inside (S, R) | Lower |

The outcome only checks the endpoint R. A model can keep talking through most of the interruption, stop just before R and still get a `response`; `stop_latency_s`, `speech_s_per_interruption_s` and `restart_rate` show that. None of them tells a reactive stop from an answer that ended on its own, which would need a matched run with the interruption muted.

Every metric except `exercised_rate` is over exercised samples. Whether a sample is exercised depends on the system as well as the dataset, because S is fixed in input time: a model that answers briefly or late is silent at S more often. A `not_exercised` sample therefore counts as an unknown outcome in the selected bounds of those metrics, like an ineligible one; for `exercised_rate` itself it is a known no. A model silent at every S gets bounds of `[0, 1]` and is visibly unmeasured, and a terse model's lower bound drops. The bounds are worst-case completions under this fixed-onset protocol, not a bound on how a model would handle an interruption timed while it speaks. Read `exercised_rate` and the bounds with every interruption number.

### Backchannel

`icc_backchannel` has no unified metric; the official frequency and JSD are the reference. `eval_backchannel.py` has known problems. Fixing them needs new definitions with their own validation, so they are only listed here:

- It adds every VAD segment up to 3 s to the backchannel list, including the ones it classified as a takeover.
- Its takeover flag is overwritten by each segment, and the loop stops at the first segment over 3 s, which also drops the later segments from frequency and timing.
- The timing score is SciPy's Jensen-Shannon distance with the natural log, at most about 0.83, while a sample without backchannels is given 1. The average therefore mixes the share of samples without backchannels with the timing distance of the rest.

### Speech detection

The scorer runs the Silero model that `load_silero_vad()` loads, with Silero's default threshold 0.5, minimum speech 250 ms and minimum silence 100 ms, and no padding. The default 30 ms of padding widens each segment on both sides and can move a reply that starts just after R to before it. `output.wav` must hold exactly T seconds of samples. Segments come from the whole file, so near S they can depend on audio after S. A pause longer than 100 ms ends a segment, so a model pausing between sentences exactly at S counts as `not_exercised`. On one 200-sample capture of a native full-duplex model, thresholds from 0.3 to 0.7 and minimum silences from 50 to 500 ms moved `not_exercised` between 85 and 95 samples, `response_rate_2s` between 0.952 and 0.957 and the median `stop_latency_s` between 1.04 and 1.27 s, and left `speaking_at_end` at 4. `restart_rate` ran from 0.05 to 0.27, because a shorter minimum silence splits speech into more segments, so it is a diagnostic and does not rank systems.

The audio is the received output played in order. For the turn-based realtime protocol the playout drops queued audio when the server reports that the user started speaking while interruptions are on; native full-duplex traces have no such event, so audio a system has already sent plays out, and the outcome describes the system as delivered.

### Reading them together

Silence scores perfectly on pause handling, and one canned word right after E counts as a response. Read pause restraint and response rates together, from the same system configuration; neither is enough alone.

## Denominators and intervals

- Sample shares and latency quantiles are over the subset's eligible samples, and for interruption over its exercised ones. Eligible samples are those whose capture passed the `reference-export` checks (input pacing, an error-free observation window, dataset hashes that match the run). The pooled ratios divide by the pauses and the pause seconds of the eligible samples. `declared` is the full subset size.
- Ineligible samples are counted under `counts`, never dropped silently. A sample can have several reasons, so the counts in `counts.ineligible_reasons` need not sum to `ineligible`.
- An ineligible sample's outcome is unknown, and a capture failure can depend on the system. Every sample share therefore also has `selected_bounds`, `[count / selected, (count + unknown) / selected]`: its value over all selected samples with each unknown outcome taken as no, then as yes. The unknown samples are the ineligible ones, plus the `not_exercised` ones for every interruption metric except `exercised_rate`. `coverage` is eligible over selected for each subset.
- A share of samples has a 95% Wilson interval, whose endpoints are exactly 0 at a count of 0 and exactly 1 at a count of n. The pooled ratios and the median latency have a 95% percentile bootstrap interval that resamples whole samples (2000 resamples, seed 0), because the pauses of one sample are not independent. Each resample's median and the interval endpoints use the same quantile rule as above. A resample with no pause time has no `intrusion_s_per_pause_s` and is skipped. The output is deterministic.
- `min_observable_s` is the shortest time from the reference to T. T is the dataset input length, so every system has the same horizon on a sample. A response after T is not observed, so `response_rate_ks` is a lower bound when `min_observable_s` is below k. Candor turn-taking windows end about 5 s after E, so there `response_rate_5s` is close to the share that responded inside the window.
- The intervals cover dataset sampling only, not repeated generations or ASR errors. To compare two systems, pair their rows in `unified-samples.jsonl` by `sample_id` over the samples eligible in both, and for interruption exercised in both. Pairing does not remove selection bias, so report coverage, `exercised_rate` and the selected bounds as well, and for interruption how many samples eligible in both were exercised in both, only in one system, or in neither.

## Run

```bash
python -m benchmarks.eval.benchmark_duplex_v10 unified-score --tree "$OUT/reference"
```

It refuses a tree without `asr.json`, stops if the transcript or annotation of an eligible sample, or the `output.wav` of an interruption sample, differs from `asr.json` or `manifest.json`, and stops on a time that fails the checks above. It writes:

- `unified.json`: per subset, the counts, `coverage` and every metric with its interval, plus `selected_bounds` for sample shares and `min_observable_s` for turn-taking and interruption. Only the median latency has an interval, not p90. At the top are the definitions version (`fdb-v1.0-unified-4`), the hashes of `manifest.json` and `asr.json`, and the silero-vad version, model file hash and settings.
- `unified-samples.jsonl`: one row per selected sample of the four scored subsets (`icc_backchannel` has no rows) with every per-sample quantity, including the pause flags `pause_intrusion`, `inside_pause_onset`, `pre_pause_speech`, `carry_in` and `any_speech`, and for interruption the outcome, S, `speech_at_onset`, `speech_at_end`, `speech_before_onset`, `stop_s`, `speech_in_interruption_s` and `restarts`. Ineligible rows keep their sample id, subset and task and add their reasons.

It prints each subset's metrics as value, interval and count, with the bounds over all selected samples for each share. When `reference-evaluate` has written `summary.json` from the same `manifest.json` and `asr.json`, its numbers are printed below them as the reference column; if either hash differs, the column is left out and a line says which input changed. Interruption numbers rated by a judge other than the official one are labeled with that judge's model instead of `official`. `summary.json` is never written or used for scoring. Rerunning on the same tree writes the same bytes.

## Not measured

- **Why the model stops after an interruption.** `stop_latency_s` says when the speech covering S ends, not whether the model stopped because of the interruption; that needs a matched run with the interruption muted.
- **The end of the user's speech in a pause sample.** The window is treated as a non-yielding user turn. The official evaluator reads output chunks, not a user-end annotation, so it does not establish the true turn end either. A reply that starts in a short trailing silence counts as an intrusion, so not every counted onset is premature.
- **Relevance of a reply.** Neither these metrics nor the official turn-taking score check that a reply answers the user. Only the official interruption judge rating covers it, and that rating depends on the judge model the run configured.
- **ASR errors.** A missed or invented word changes the metrics it falls in. All systems share the pinned Parakeet checkpoint, which does not mean they share the same recognition error.
- **Acoustic latency.** The output is a simulated playout of the received audio.
