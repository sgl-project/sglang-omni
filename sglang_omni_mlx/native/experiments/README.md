# Qwen3-ASR pause refresh simulation

This deterministic event simulation compares three policies using the native
admission helper in `src/qwen3_asr/realtime_refresh.h`:

- **A:** cadence-only reference.
- **B:** 300 ms pause threshold; busy hints are discarded.
- **C:** 300 ms pause threshold; one useful busy hint may be deferred.

It also prints the existing 200 ms pause arm. Each trace runs with 100 ms and
200 ms packet delivery and assumed 200 ms and 400 ms inference durations. The
targeted traces cover a busy hint that is already covered, a useful deferred
hint, a cadence refresh that supersedes a deferred hint, VAD-reported speech
resumption, Final handoff, and multiple pauses while one refresh is active.

Reproduce from the repository root:

```sh
xcrun clang++ -std=c++20 -O2 -Wall -Wextra \
  -I sglang_omni_mlx/native/src \
  sglang_omni_mlx/native/experiments/qwen3_asr_pause_refresh_experiment.cpp \
  -o /private/tmp/qwen3_asr_pause_refresh_experiment
/private/tmp/qwen3_asr_pause_refresh_experiment \
  > /private/tmp/qwen3_asr_pause_refresh_results.txt
diff -u sglang_omni_mlx/native/experiments/qwen3_asr_pause_refresh_results.txt \
  /private/tmp/qwen3_asr_pause_refresh_results.txt
```

The deferred arm reports received and valid hints, covered or insufficient
rejections, deferrals, admissions, cadence supersession, invalidation on speech
resumption or Final, additional refresh count, completed snapshot coverage at
Stop, readiness at the first modeled HeadStart opportunity, residual refresh
work, and synthetic Final queue delay. The buffer-burst traces keep sample
coordinates separate from wall time and add a Stop delay where needed to model
the interval between the final audio append and Final handoff.

The simulation models scheduling only. Its inference durations are assumptions;
coverage means the admitted refresh's audio snapshot endpoint, not transcript
quality or equality. It does not measure GPU speed, real-model latency, or
end-to-end TTLC, and its Final queue delay is a simplified FIFO model. The
checked-in output is not evidence of a performance improvement.

For real runtime experiments, deferred admission remains opt-in and requires
the 300 ms pause arm:

```sh
open --env VOXT_EXPERIMENT_ASR_PAUSE_REFRESH_MS=300 \
  --env VOXT_EXPERIMENT_ASR_DEFER_BUSY_PAUSE=1 build/darwin-arm64/Voxt.app
```

Add `VOXT_EXPERIMENT_ASR_PAUSE_REFRESH_TRACE=1` to emit content-free,
session-correlated admission and execution records. The trace is independent of
the deferred control so the discard arm can also be attributed. No model
checkpoint is needed for the simulator.
