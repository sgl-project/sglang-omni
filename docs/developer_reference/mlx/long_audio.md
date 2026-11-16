# ASR long-recording measurements

The opt-in benchmark measures native Qwen3-ASR through HTTP or its realtime WebSocket, and Parakeet through HTTP. It reports transcription quality alongside memory and elapsed time.

## Run

Build the native runtime with sglang_omni_mlx/native/scripts/build_runtime.sh, install ci/requirements.lock into a Python 3.12 environment, and provision the frozen corpus and Qwen3-ASR-0.6B-4bit using ci/provision.py. Then run this from the repository root:

```bash
python -m sglang_omni_mlx.native.ci.benchmark_long_audio \
  --runtime-bin "$NATIVE_RUNTIME_BIN" \
  --data-root "$CI_DATA_ROOT" \
  --model mlx-community/Qwen3-ASR-0.6B-4bit \
  --output "$ASR_RESULTS_DIRECTORY" \
  --mode realtime --minutes 5 10 30 60 --languages en zh --speed 1 \
  -- --cache-limit-mb 512
```

--output must be a new directory. Flags after -- go to the server and are recorded in metadata.json. Voxt starts the server with --cache-limit-mb 512, so pass that flag to measure what the app runs; without it the server's default 2 GiB limit applies. This command paces 210 minutes of audio live. A speed greater than one is an accelerated input stress measurement: refresh decodes may be coalesced, and its accuracy and finalization latency are not equivalent to live dictation.

For Qwen3-ASR, use --mode http separately for HTTP requests. Alone it sends one whole-file request with the server's default budget of 10 tokens per audio second. This is a maximum-length stress measurement, not what Voxt sends: a repetition loop runs until that budget, so the time, memory and error rate of such a run depend on the budget. Start with five minutes before testing longer whole-file requests. Add --chunk-seconds 30 to send whole clips in requests of at most 30 seconds, each with the budget of a Voxt Final (28 tokens per audio second plus 64, within 256 to 8192). This is close to the 20 to 45 second pieces Voxt sends during a long recording, except that the cuts fall on clip boundaries instead of detected pauses.

Each case starts a dedicated server, warms it with a short clip, sends the long recording and transcribes the warmup clip again. The post-session transcript must match warmup exactly. Disconnection is covered by test_a_disconnected_stream_stops_its_decode in ci/test_server_api.py.

The benchmark reads the dedicated server's macOS physical footprint, the kernel's record of its peak footprint and its resident memory every 250 ms. By default it kills only that server once the peak exceeds 8 GiB; change --max-footprint-gib explicitly to test another budget. If the sampler cannot read or record measurements, it stops that server rather than silently running without the guard. This is a soft guard, not a hard memory limit: a large allocation can exceed the threshold before the next read. Do not rely on it to prevent system memory pressure.

## Parakeet final transcription

Parakeet's automatic mode (omit chunk_duration, or use 1199 or more, as Voxt does) keeps audio up to 30 seconds whole and splits longer audio at the quietest 100 ms in the last five seconds before that limit. The ranges are consecutive, without overlap or discarded samples; if the last range would contain less than 10 seconds, the preceding search moves earlier to keep useful tail context. Unlike the bidirectional searches of Cohere Transcribe and MOSS, a cut never exceeds the cap. It requires no VAD checkpoint.

A chunk can still produce no tokens or stop while speech continues, including in the Python MLX reference. When an unrecognized tail of at least three seconds has a mean-square amplitude of at least 1e-5 (about -50 dBFS RMS), automatic decoding retries it at most twice: first from one second before the last recognized token, or with 250 ms of leading silence for an entirely blank chunk, then with more padding. Padding only uses spare room in the 30-second input, so no retry discards the source tail, and an attempt whose input would repeat the previous decode, such as on a full 30-second blank chunk, is skipped. Recovered text merges on matching overlap tokens; a failed retry keeps the existing text.

The energy threshold is a retry gate, not a voice detector or a completeness guarantee. Noise can trigger it, and it does not detect internal gaps, missing leading speech, tails shorter than three seconds or speech below the threshold. Check deletion runs as well as WER.

Explicit chunk_duration values from 2 to 300 seconds keep fixed windows with one-second overlap and token merging, without recovery. They are compatibility and diagnostic settings: short windows can decode mid-speech blanks, and long ones use substantially more memory. The HTTP stream field selects SSE transport, not the chunking.

The server and CLI cap MLX's freed-buffer cache at 256 MiB. This is not a total memory limit: weights, active encoder arrays, the uploaded audio and queued requests come on top. The CLI clears its cache between files and reports peak_active_bytes, active_bytes and cache_bytes; these MLX counters exclude some process allocations, so use the benchmark's kernel footprint to assess a smaller Mac.

Provision mlx-community/parakeet-tdt-0.6b-v3 with ci/provision_models.py, then run the guarded benchmark. Parakeet v3 has no Chinese or realtime endpoint, so the benchmark defaults it to English and requires --mode http:

```bash
python -m sglang_omni_mlx.native.ci.benchmark_long_audio \
  --runtime-bin "$NATIVE_RUNTIME_BIN" \
  --data-root "$CI_DATA_ROOT" \
  --model-kind parakeet --model mlx-community/parakeet-tdt-0.6b-v3 \
  --output "$PARAKEET_RESULTS_DIRECTORY" \
  --mode http --minutes 1 5 15 30 --max-footprint-gib 3
```

This sends each whole recording in one request, exercising the server's own splitting and recovery; do not add --chunk-seconds when measuring that path. Compare the same seed and frozen corpus before and after a policy change.

The parakeet-tdt-0.6b-v3.json golden requests five-second overlapping windows to keep its recorded Swift parity outputs, and parakeet-tdt-0.6b-v3-final.json requests automatic mode. Keep that distinction rather than replacing the old outputs or widening their tolerance. API regressions are in ci/test_parakeet_api.py; checkpoint-free boundary and coverage tests are in native/tests/audio_chunk_test.cpp and run in Mac CI.

### Validated configuration

Measured on Apple M4 Max with 64 GiB, macOS 27.0.1, MLX 0.32.3 and the checkpoint pinned in ci/model_pins.json. English WER on the frozen 392-clip corpus fell from 23.844% with the original five-second policy to 2.548% in automatic mode: short clips were unchanged, medium clips went from 20.214% to 2.009% and long clips from 31.851% to 2.752%. The Python reference (mlx-audio 0.5.9) gives 23.128% with the same five-second windows, so the loss comes from the policy rather than the port. Explicit five-second decoding matched the original binary on all 392 clips. Chinese and mixed-language metrics remain poor because this checkpoint does not support Chinese.

The guarded command above, with the default seed 2888, gave the following. Peak footprint is the kernel process footprint in GiB (2^30 bytes), including startup and warmup.

| Requested recording | Original five-second WER | Automatic final WER | Longest deletion run, original / final | Automatic peak footprint | Request time / realtime factor |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 minute | 16.23% | 2.09% | 9 / 0 words | 2.05 GiB | 0.338 s / 0.00482 |
| 5 minutes | 23.23% | 2.73% | 21 / 1 words | 2.17 GiB | 1.220 s / 0.00405 |
| 15 minutes | 40.75% | 2.88% | 62 / 5 words | 2.33 GiB | 3.427 s / 0.00380 |
| 30 minutes | 38.97% | 2.62% | 62 / 5 words | 2.48 GiB | 6.705 s / 0.00372 |

Before recovery, 30-second chunks alone left 41- and 86-word deletion runs on the 15- and 30-minute inputs despite 7.66% and 5.43% WER; recovery cut their deleted reference words from 134 to 12 and from 156 to 15. All four cases stayed under the 3 GiB guard, exited normally and reproduced the warmup transcript. These synthetic concatenations do not measure natural meetings, and no 8 GiB Mac or system-wide memory-pressure test was run, so measure on the target machine rather than inferring an OOM guarantee.

## Input and scoring

The recording concatenates complete short and medium clips of one language from the frozen corpus. A seeded permutation uses every clip before reusing the pool. Longer recordings necessarily reuse clips; the manifest records their order and the recording's hash. These are synthetic concatenations, not natural hour-long meetings.

The benchmark sends HTTP fields for the voxt_swift audio layout and both stopping flags. These control Qwen3-ASR; Parakeet ignores them and uses its own decoding policy. Qwen3-ASR realtime sessions use the server's own defaults for layout and stopping rules, which can differ between revisions. Do not treat Qwen3-ASR HTTP versus realtime scores as a controlled same-layout comparison. Voxt shows realtime text only as a preview and inserts the text of its HTTP requests, so realtime error rates are not what the app inserts. This tool does not calculate an isolated-short-clip baseline or directly measure natural continuous meetings; differences across durations also include changing clip composition and boundary effects.

The final clip is kept whole, so actual duration is at least the requested duration and may exceed it by one clip. Every reference word remains aligned with audio actually sent. The same seed gives nested recording prefixes across durations. English WER and Chinese CER count the tokens of ci/transcript_tokens.py, which the golden checker also uses, so Chinese keeps embedded Latin words as units. Both missing tails and repeated text count as errors.

Each score also reports deleted_reference_tokens and longest_deleted_reference_token_run from the same optimal Levenshtein edit alignment as the error count. The former counts deleted reference tokens; the latter is the longest run of adjacent reference positions marked as deletions, not elapsed seconds. Units match reference_tokens: English words, or Chinese characters with embedded Latin words counted as single tokens by ci/transcript_tokens.py. Substitutions are errors but not deletions; retained or substituted reference tokens break a deletion run. Repeated text can admit more than one optimal alignment, so these counts describe the alignment chosen by RapidFuzz rather than timestamp-based speech coverage.

## Interpret results

- metadata.json records hardware, OS, source revision, working-tree state, executable hash, checkpoint file hashes, benchmark dependency versions, HTTP request fields, realtime defaults and the memory guard. The MLX version loaded by the separate native server is not discovered automatically; record it independently.
- Each input manifest records the exact source sequence and expected transcript. Each result stores the hypothesis, error numerator and denominator, deleted_reference_tokens, longest_deleted_reference_token_run, timing, memory, the recovery observation and server_exit_code. An exit before shutdown, a nonzero exit during shutdown or a shutdown timeout marks the case as failed. Realtime runs also save timestamped protocol events.
- The memory trace spans startup, warmup, the long session and recovery. Compare equivalent phases; an encoder allocation or stored transcript is not by itself a leak. The warm baseline, post-session and post-recovery values separate those phases. peak_footprint_bytes is the kernel's exact peak at the end of the session. It only grows, so it also covers startup and warmup, and it equals warmup_peak_footprint_bytes when the session set no new peak.
- HTTP results include real_time_factor, the request wall time divided by audio duration: a 6-second request for 60 seconds of audio has a factor of 0.1. In realtime mode, wall time includes the time spent sending audio at the selected speed, while the server may process audio concurrently. Finalization time is the wait from sending the last audio frame to receiving the completed transcript; it can include processing audio that accumulated earlier.
- A measured status means the protocol, scoring and recovery completed. Review the error rate, deletion count, longest deletion run, generated-token finish reason and memory trace. A low aggregate error rate can still hide a long missing span. The table above includes the deletion fields; the original hypotheses were rescored with the same alignment code used for the revised run.

Compare before/after builds with identical weights, MLX version, input hashes, pacing and hardware. Keep existing API tests and corpus goldens as correctness gates; this benchmark is an opt-in measurement, not a timing-sensitive unit-test gate. Its own tests are in native/tests/test_long_audio_benchmark.py and run against the stub server.
