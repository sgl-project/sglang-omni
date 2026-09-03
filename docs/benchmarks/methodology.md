# Benchmark methodology

SGLang-Omni benchmarks measure correctness and serving performance under a
declared model, dataset, hardware, software, launch, and traffic configuration.
A command in a cookbook is an entry point; its result becomes qualification
evidence only when the full run is reproducible.

## Required run identity

Record these fields with every published result:

- repository commit and dirty-worktree state;
- model identifier and resolved revision;
- dataset identifier, split, revision, and effective input hash when available;
- hardware model, count, memory, placement, and relevant host resources;
- driver, CUDA or accelerator runtime, Python, framework, and dependency
  versions;
- complete server command and checked-in config;
- benchmark command, seed, sample count, warmup, repeats, and concurrency;
- streaming mode, response format, and chunk policy;
- completed, failed, rejected, skipped, and timed-out sample counts;
- path to the machine-readable artifact.

Do not publish a hardware minimum from a benchmark. A result qualifies only the
measured configuration.

## Dataset and correctness

Pin the dataset revision and report the evaluated population after filtering.
Keep normalization, language-specific scoring, reference construction, and
excluded samples explicit.

Use task-appropriate metrics:

| Task | Typical correctness evidence |
|---|---|
| TTS | WER/CER, speaker similarity, audio validity, completion rate |
| ASR | WER/CER, diarization or timestamp metrics, completion rate |
| Omni understanding | Dataset accuracy plus failed/skipped counts |
| Speech-output Omni | Understanding accuracy plus audio WER/similarity and audio validity |

Report aggregate and tail behavior. Never remove runaway or failed samples from
the primary result without also publishing the raw count and exclusion rule.

## Warmup and repeats

Warmup requests run before the timed population and are excluded from measured
aggregates. State whether warmup happens once per server, once per concurrency,
or once per repeat. Use fresh servers when startup state, graph capture, or
cache state is part of the comparison.

Use multiple measured repeats for performance claims. Publish per-repeat
results or dispersion rather than only the best run. Interleave competing
configurations when shared-host drift could bias a sequential comparison.

## Traffic model

Always distinguish:

- **closed loop**: the client holds at most N requests in flight and sends a
  replacement after one completes;
- **open loop**: arrivals follow an offered request rate independent of
  completions;
- **sustained overshoot**: offered load intentionally remains above admitted
  capacity to measure rejection and recovery.

Concurrency is not request rate. A stepped closed-loop sweep measures behavior
at fixed in-flight limits but does not prove overload shedding under continuous
arrivals.

## Performance metrics

Use consistent definitions:

| Metric | Definition |
|---|---|
| End-to-end latency | Client wall time from request send to terminal response |
| TTFT | Time from request send to first text token |
| TTFA | Time from request send to first audio bytes |
| Throughput | Successful requests divided by measured wall time |
| Output token rate | Generated output tokens divided by request time or wall time, as named by the benchmark |
| RTF | Request latency divided by produced or processed audio duration |
| RTFx | Successful input-audio seconds divided by wall-clock seconds |

Lower RTF is faster relative to audio duration. Higher RTFx means more audio is
processed per wall-clock second. Keep unsuccessful requests out of latency and
RTF aggregates only when their counts and failure reasons are reported next to
the successful metrics.

For streaming, also record first-chunk policy, chunk cadence, client playback or
buffering behavior, and whether the final tail is included in end-to-end time.

## Resource measurements

Identify the exact server process IDs when collecting process CPU, GPU memory,
power, or utilization. If process-scoped measurement is unavailable, mark the
metric unavailable instead of silently including unrelated host workloads.
Containerized CPU measurement may require the host PID namespace.

Resource sampling should not change the traffic model. Use matched monitoring
settings across compared configurations.

## Qualification reports

Keep permanent cookbooks short: include the canonical command and links to this
methodology and the relevant report. Put large result tables, tuning sweeps,
bottleneck analysis, and current-main comparisons in a qualification report or
machine-readable artifact tied to an exact commit.

CI thresholds are regression gates, not universal performance guarantees. A CI
subset, hardware lane, and slack policy must remain visible when a CI result is
cited.

## Canonical entry points

- TTS: `python -m benchmarks.eval.benchmark_tts_seedtts`
- ASR: `python -m benchmarks.eval.benchmark_asr_seedtts`
- Omni image understanding: `python benchmarks/eval/benchmark_omni_mmmu.py`
- Omni audio understanding: `python benchmarks/eval/benchmark_omni_mmsu.py`
- Omni video understanding: `python -m benchmarks.eval.benchmark_omni_videomme`
- Omni video and audio: `python -m benchmarks.eval.benchmark_omni_videoamme`

Use `--help` on the selected entry point and the model cookbook for its pinned
revisions and canonical arguments.
