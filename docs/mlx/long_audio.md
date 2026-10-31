# Qwen3-ASR long-recording measurements

The opt-in benchmark measures the native Qwen3-ASR server through HTTP or its realtime WebSocket. It reports transcription quality alongside memory and elapsed time.

## Run

Build the native runtime with sglang_omni_mlx/native/scripts/build_runtime.sh, install ci/requirements.lock into a Python 3.12 environment, and provision the frozen corpus and Qwen3-ASR-0.6B-4bit using ci/provision.py. From the repository root, run python -m sglang_omni_mlx.native.ci.benchmark_long_audio --help for the required paths.

```bash
python -m sglang_omni_mlx.native.ci.benchmark_long_audio \
  --server "$NATIVE_RUNTIME_BIN/qwen3_asr_server" \
  --model-directory "$CI_DATA_ROOT/models/mlx-community_Qwen3-ASR-0.6B-4bit" \
  --corpus-directory "$CI_DATA_ROOT/corpus/v1" \
  --output "$ASR_RESULTS_DIRECTORY" \
  --mode realtime --minutes 5 10 30 60 --languages en zh --speed 1
```

Supply --server, --model-directory, --corpus-directory and a new --output directory. Use --mode realtime --minutes 5 10 30 60 --languages en zh --speed 1 for live pacing. This matrix contains 210 minutes of audio. A speed greater than one is an accelerated input stress measurement: refresh decodes may be coalesced, and its accuracy and finalization latency are not equivalent to live dictation. Use --mode http separately to measure one whole-file request; this deliberately does not implement Voxt's client-side chunking. Start with five minutes before testing longer whole-file requests.

Each case starts a dedicated server, warms it with a short clip, sends the long recording, then tests SSE disconnection on a separate five-minute recording and transcribes the warmup clip again. An observed running request is required before disconnecting. A separate natural run records time from its first heartbeat to completion; the cancelled run records time from its first heartbeat to disconnect and from disconnect to idle. The comparison subtracts the latter heartbeat-to-disconnect interval from the natural run's heartbeat-to-completion time. A missing heartbeat or an idle time no earlier than this paired estimate produces an inconclusive case and a nonzero exit. Earlier idle is evidence of cancellation, though independent runs and request completion can race. The post-cancellation transcript must match warmup exactly. Cancellation here covers HTTP SSE, not WebSocket cancellation.

The benchmark polls the dedicated server's macOS physical footprint and resident memory every 250 ms. By default it kills only that server if a sample exceeds 8 GiB; change --max-footprint-gib explicitly to test another budget. If the sampler cannot read or record measurements, it stops that server rather than silently running without the guard. This is a soft, sampled guard, not a hard memory limit: large allocations can exceed the threshold before the next sample, and short peaks can be missed. Do not rely on it to prevent system memory pressure.

## Input and scoring

The recording concatenates complete short and medium clips of one language from the frozen corpus. A seeded permutation uses every clip before reusing the pool, avoiding the single repeated utterance that can make a long-input cancellation test finish early. Longer recordings necessarily reuse clips; the manifest records their order, unique count, hashes and reuse count. These are synthetic concatenations, not natural hour-long meetings.

HTTP requests explicitly use the voxt_swift audio layout and both stopping flags. Realtime sessions use the server's own defaults for layout and stopping rules, which can differ between revisions. Do not treat HTTP versus realtime scores as a controlled same-layout comparison. This tool does not calculate an isolated-short-clip baseline or directly measure natural continuous meetings; differences across durations also include changing clip composition and boundary effects.

The final clip is kept whole, so actual duration is at least the requested duration and may exceed it by one clip. Every reference word remains aligned with audio actually sent. The same seed gives nested recording prefixes across durations. English WER and Chinese CER use the frozen-corpus checker's normalization; edit distance uses RapidFuzz to avoid quadratic Python execution time on hour-long transcripts. Both missing tails and repeated text count as errors. Chinese normalization keeps embedded Latin words as units, as in the existing golden checker.

## Interpret results

- metadata.json records hardware, OS, source revision, working-tree state, executable hash, checkpoint file hashes, benchmark dependency versions, HTTP request fields, realtime defaults and the sampled guard. The MLX version loaded by the separate native server is not discovered automatically; record it independently.
- Each input manifest records the exact source sequence and expected transcript. Each result stores the hypothesis, error numerator and denominator, timing, memory and cancellation/recovery observations. Realtime runs also save timestamped protocol events.
- The memory trace spans startup, warmup, the long session and cancellation/recovery. Compare equivalent phases; an encoder allocation or stored transcript is not by itself a leak. The warm baseline, session peak, post-session and post-recovery values separate those phases.
- HTTP real-time factor is request wall time divided by audio duration: a 6-second request for 60 seconds of audio has a factor of 0.1. In realtime mode, wall time includes the time spent sending audio at the selected speed, while the server may process audio concurrently. Finalization time is the wait from sending the last audio frame to receiving the completed transcript; it can include processing audio that accumulated earlier.
- A measured status means the protocol, scoring, recovery and paired timing measurement completed. Inconclusive means the cancellation comparison could not establish earlier idle. Review the error rate, generated-token finish reason, cancellation comparison and memory trace.

Compare before/after builds with identical weights, MLX version, input hashes, pacing and hardware. Keep existing API tests and corpus goldens as correctness gates; this benchmark is an opt-in measurement, not a timing-sensitive unit-test gate.
