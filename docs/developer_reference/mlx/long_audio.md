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

Parakeet's automatic whole-recording mode (omit chunk_duration, or use 1199 or more) keeps audio up to 30 seconds whole and splits longer audio at the quietest 100 ms in the last five seconds before that limit. The ranges are consecutive, without overlap or discarded samples. If the last range would contain less than 10 seconds, the preceding quiet search moves earlier to reserve useful tail context. This follows the low-energy boundary selection used by Cohere Transcribe and MOSS, but unlike their bidirectional searches it never exceeds the cap. It requires no VAD checkpoint.

A chunk can still produce no tokens or stop while speech continues, including in the Python MLX reference. Automatic decoding retries a trailing unrecognized span of at least three seconds when its mean-square amplitude is at least 1e-5 (about -50 dBFS RMS). The first retry includes one second of recognized context, or 250 ms of leading silence for an entirely blank chunk; a second retry varies the padding up to 500 ms. Matching overlap tokens merges recovered text. Padding is limited to spare capacity in the 30-second input budget: every encoder input remains at most 30 seconds and no retry discards the source tail. There are at most two additional decodes per chunk, with cancellation checked between decodes and decoder steps.

The energy threshold is a retry gate, not a voice detector or a completeness guarantee. Noise can trigger it. It does not detect internal gaps followed by later recognized words, missing leading speech, tails shorter than three seconds or speech below the energy threshold. A full 30-second blank input has no spare padding capacity, so recovery can repeat the original decode; nearly full inputs can also limit the variation between attempts. Failed retries retain the existing text; successful retries can revise the one-second overlap. Check deletion spans as well as WER, and retain headroom for the bounded extra work.

Explicit chunk_duration values from 2 to 300 seconds retain fixed windows with one-second overlap and token alignment/merging. These are compatibility and diagnostic settings, not the recommended final-transcription policy: short mid-speech windows can decode to blanks, and long explicit windows consume substantially more memory. The HTTP stream field selects SSE transport, not the internal chunking algorithm or incremental recognition.

The Parakeet process limits MLX's freed-buffer cache to 256 MiB for both server and CLI. This is not a total memory limit: weights, active encoder arrays, the uploaded audio and queued requests also consume memory. The CLI clears its cache between files, like Qwen3-ASR, and reports peak_active_bytes, active_bytes and cache_bytes. These MLX counters exclude some process allocations; use the benchmark's kernel physical footprint to assess a smaller Mac. Avoid unbounded request concurrency, and keep headroom for macOS and other applications. A measured footprint on a larger Mac is not a guarantee against OOM on every smaller Mac.

Provision mlx-community/parakeet-tdt-0.6b-v3 with ci/provision_models.py, then run the same guarded benchmark (Parakeet v3 has no Chinese or realtime endpoint):

```bash
python -m sglang_omni_mlx.native.ci.benchmark_long_audio \
  --runtime-bin "$NATIVE_RUNTIME_BIN" \
  --data-root "$CI_DATA_ROOT" \
  --model-kind parakeet --model mlx-community/parakeet-tdt-0.6b-v3 \
  --output "$PARAKEET_RESULTS_DIRECTORY" \
  --mode http --languages en --minutes 1 5 15 30 --max-footprint-gib 3
```

This deliberately sends each whole recording in one request, exercising the server's internal bounded splitting and recovery. Do not add --chunk-seconds when measuring that path. The guard records startup, transcription and recovery and fails if the sampled kernel peak exceeds the supplied budget; it does not impose a hard allocator limit. Compare the same seed and frozen corpus before and after a policy change.

The original parakeet-tdt-0.6b-v3.json golden explicitly requests five-second overlapping windows to retain its recorded Swift parity outputs. The separate parakeet-tdt-0.6b-v3-final.json golden requests automatic mode and records the changed final policy. Preserve that distinction rather than replacing the old outputs or weakening their tolerance. API regressions are in ci/test_parakeet_api.py; checkpoint-free boundary and coverage tests are in native/tests/audio_chunk_test.cpp and run in Mac CI.

### Validated configuration

Measured on Apple M4 Max with 64 GiB unified memory, macOS 27.0.1 and the pinned MLX/MLX Metal 0.32.3 runtime. The checkpoint is the revision and file hashes in ci/model_pins.json. English WER on the frozen 392-clip corpus changed from 23.844% with the original five-second policy to 2.548% with automatic pause-aligned chunks and bounded recovery. The 64 short English clips were unchanged; medium clips improved from 20.214% to 2.009% and long clips from 31.851% to 2.752%. Explicit five-second decoding matched the original binary on all 392 outputs on this chip. Chinese and mixed-language metrics remain poor because this checkpoint does not support Chinese. Recovery changed two unsupported-Chinese noisy clips from blank output to unrelated English text; silence and synthetic-noise clips remained empty. Do not use recovery as a language or speech detector.

The Python reference is mlx-audio 0.5.9 with MLX/MLX Metal 0.32.3, using the same BF16 checkpoint at revision ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15 and SHA-256 hashes in ci/model_pins.json. A separate same-policy comparison used explicit five-second windows with one-second overlap on all 392 clips (147 English, 173 Chinese, 64 mixed and eight non-speech):

| Implementation and policy | English WER | Chinese CER | Mixed MER |
| --- | ---: | ---: | ---: |
| Python reference, explicit five seconds | 23.128% | 99.975% | 89.717% |
| Native, explicit five seconds | 23.844% | 99.938% | 89.460% |
| Native, automatic final policy | 2.548% | 99.988% | 88.689% |

Python and native explicit-five-second text matched on 217/392 clips after stripping outer whitespace. That is distinct from the 392/392 native-before/native-after compatibility check. The Python front end casts waveform samples to BF16; native preserves float32 waveform arithmetic for Swift compatibility. This is a known arithmetic difference, not a complete localization of every differing transcript. Automatic mode intentionally changes policy, so its row is an accuracy comparison rather than a Python parity claim. Chinese remains unsupported in every row.

A single-process matched performance pass on the same 80 English medium/long clips (1,920.645 seconds of audio), with five-second windows and one-second overlap, gave:

| Implementation | Constructor load | Summed decode time | Realtime factor | Peak kernel footprint |
| --- | ---: | ---: | ---: | ---: |
| Python reference | 0.052 s | 18.448 s | 0.00961 | 2.19 GiB |
| Native explicit five seconds | 0.149 s | 14.637 s | 0.00762 | 2.01 GiB |

Both clear the MLX cache between files; native also sets its 256 MiB cache cap while Python retains the default. Constructor load excludes interpreter imports and can leave lazy work to the first decode. Python decode includes file I/O; the native timer starts after reading the WAV bytes. Peak footprint comes from macOS time -l and includes process startup. These single passes describe this configuration, not an isolated kernel-speed or cold-start guarantee.

The guarded automatic-policy command above, with the default seed 2888, gave the following matched-input comparison. Peak footprint includes startup and warmup, is the kernel process footprint (not MLX active memory or RSS), and uses GiB (2^30 bytes).

| Requested recording | Original five-second WER | Automatic final WER | Longest deletion run, original / final | Automatic peak footprint | Request time / realtime factor |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 minute | 16.23% | 2.09% | 9 / 0 words | 2.05 GiB | 0.338 s / 0.00482 |
| 5 minutes | 23.23% | 2.73% | 21 / 1 words | 2.17 GiB | 1.220 s / 0.00405 |
| 15 minutes | 40.75% | 2.88% | 62 / 5 words | 2.33 GiB | 3.427 s / 0.00380 |
| 30 minutes | 38.97% | 2.62% | 62 / 5 words | 2.48 GiB | 6.705 s / 0.00372 |

Review of the earlier 30-second policy without recovery found 41- and 86-word deletion runs, despite aggregate WER of 7.66% and 5.43% on the 15- and 30-minute inputs. With the same input hashes, bounded recovery reduced total deleted reference words from 134 to 12 and from 156 to 15; the longest run is five words in each revised case. The API suite reconstructs the two failing windows from frozen corpus clips and also checks 30.25-, 31- and 32-second short-tail recordings over JSON and SSE. These measurements do not establish complete speech coverage.

All four automatic cases passed the 3 GiB sampled guard, exited normally and reproduced the warmup transcript after the long request. A separate persistent-server run also completed 73 queued clip requests and recovered after a disconnected request under the same guard. The earlier CLI's 73-file invocation kept its measured freed-buffer cache below 256 MiB; the 256 MiB allocator setting and per-file cache clearing are unchanged. The source recording length changes the clip composition and cut positions, so WER is not monotonic in duration. These figures do not measure natural meetings, every supported language or worst-case acoustic content.

No physical 8 GiB Mac or system-wide low-memory-pressure test was performed. The 30-second encoder cap and cache bound avoid the prior unbounded CLI buffer retention, but cannot guarantee that an arbitrary upload, explicit 300-second window, concurrent queue or other applications will fit a smaller Mac. Use automatic mode, bounded client concurrency and the guarded benchmark on the target machine rather than inferring an OOM guarantee from this larger host.

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
