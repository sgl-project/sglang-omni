---
name: omni-gpu-deep-dive
description: Attribute GPU time in a sglang-omni stage to specific lines of sglang_omni/ python source. Wraps the vendored llm-torch-profiler-analysis backend, adding an omni source-attribution shim, a mapping/formal trace pair with steady-state gates, and the rules for reading its tables on an omni stage. Use when a stage is slower than expected and the question is "which python code is the bottleneck", not "which kernel is hot".
---

# omni-gpu-deep-dive

## What this is

The kernel analysis is not ours: `/sgl-workspace/sglang/.claude/skills/llm-torch-profiler-analysis`
already ranks kernels, finds overlap headroom, and matches fuse patterns. Omni adds
three things:

1. **Source attribution** (`scripts/omni_source_shim.py`) - the backend's path
   allowlist knows `python/sglang/`, `vllm/`, `tensorrt_llm/`, not `sglang_omni/`,
   so omni frames lose to torch frames and the kernel table names
   `torch/nn/modules/linear.py` instead of the line that called it. The shim lifts
   omni frames in both of the backend's two independent ranking ladders (kernel
   table and overlap table) by wrapping them. Parent-repo files are never modified.
2. **The trace pair and its gates** (`scripts/omni_trace_pair.py`).
3. **The workload contract and the omni reading rules** (this file) - what a
   profiled body has to look like for the numbers to mean anything, and what a
   given table column is worth on an omni stage.

## The one rule

**Two traces, two questions. Never conclude from one.**

| | CUDA graph | `with_stack` | Answers |
| --- | --- | --- | --- |
| `mapping` | off | on | *where* - which python line owns this kernel |
| `formal` | on (real serving config) | off | *how much* - the time a user actually pays |

A python stack exists only when the launch is a real python call, so graph-on can
never name your code; graph-off timings are not the ones you ship. Take location
from `mapping`, every number from `formal`, and conclude only when a kernel is
heavy in `formal` *and* attributed in `mapping`.

Measured on H200. An "iter" is one call of the workload body, so music3's row is
one 200-frame chunk (30 DiT steps):

| trace | kernels/iter | GPU ms/iter | wall ms/iter | busy % |
| --- | ---: | ---: | ---: | ---: |
| code2wav `mapping` (eager) | 836 | 5.77 | 19.96 | 28.9 |
| code2wav `formal` (graph on) | 906 | 5.94 | **6.43** | **92.3** |
| whisper encoder `mapping` (eager) | 399 | 29.07 | 30.17 | 96.3 |
| whisper encoder `formal` (graph on) | 399 | 29.03 | 29.74 | 97.6 |
| music3 DiT `mapping` (eager) | 27844 | 524.87 | 875.80 | 59.9 |
| music3 DiT `formal` (graph on) | 29044 | 527.08 | **553.35** | **95.3** |

Three workloads, three regimes, and a single trace misleads in two of them.
code2wav's `mapping` trace says "launch-bound, add CUDA graphs" - production has
had them for months (20 -> 6.4 ms); `formal` says 906 kernels per window at median
2.3 us and 92% busy, so the work left is fusion, not launches. Whisper's two
traces agree within 1% because its kernels are large - luck, not a rule. Music3's
GPU time is identical either way, yet graphs still buy 322 ms of wall per chunk: a
30-step serial loop pays launch overhead *around* heavy work, so "busy 60%" is a
gap problem and the top kernel rows are a compute problem at the same time.

## Running it

```bash
cd <repo root>
S=.claude/skills/omni-gpu-deep-dive/scripts
export PYTHONPATH=/sgl-workspace/sglang/python:.        # source sglang, not site-packages

python my_workload.py --output-dir .profiling-runs/<run>/   # see "The workload" below
python $S/analyze_omni_profile.py --framework sglang \
    --mapping-input .profiling-runs/<run>/mapping \
    --formal-input  .profiling-runs/<run>/formal \
    --output-dir    .profiling-runs/<run>/report
```

`analyze_omni_profile.py` is a passthrough: every backend flag works
(`--kernel-table-limit`, `--pid-substring`, `--merge-profiles`, single-trace
`--input`, ...). Nothing under `.profiling-runs/` is ever committed.

## The workload

You write one per stage - throwaway, ~100 lines with argparse, not committed. It
builds the module, builds one realistic input, and hands `capture_pair` two
callables running *the same work* eager and in the real serving config:

```python
capture_pair(
    output_dir=args.output_dir,
    mapping_body=lambda: encoder(features),      # eager, no graph, no compile
    formal_body=lambda: runner.run(features),    # what production runs
    iters=args.iters, warmup=args.warmup,
)
```

Rules:

- **Random weights are fine.** Attribution and kernel shapes follow the module
  graph and the input shape, not the values.
- **Capture and compile *before* `capture_pair`.** One-time work inside the
  profiled window is what the gate exists to reject.
- **One realistic shape per run.** Omni stages are bucketed; a shape nobody serves
  produces a kernel mix nobody pays for.
- **Same work on both sides.** If `formal_body` covers less than `mapping_body`,
  the shares are not comparable.

## Gates

**Before** - `capture_pair` warms up, then refuses any trace containing a Dynamo /
Inductor compile or a CUDA graph capture (`assert_steady_state`). One-time cost
charged to a steady-state kernel is the most common way a run reaches a confident
wrong answer. Warm every shape bucket until the gate passes; never subtract the
cost afterwards.

The gate rejects compiling and capturing, not *compiled* or *captured execution*:
`cudaGraphLaunch` is what a healthy formal trace is full of, and
`is_torchdynamo_compiling` is a predicate every HF forward calls. If you widen
`_COMPILE_MARKERS`, keep that distinction or the gate will block clean runs.

**After** - a change is accepted on the `formal` config only. Mapping-trace deltas
prove nothing about serving, because the graph replaces the launch path the mapping
trace measured.

Correctness before speed: fix the input and measure the *base's own* run-to-run
output variance first, or you cannot tell a regression from nondeterminism the base
already had.

Then performance, on the graph-on formal config: **A/A** first (base against base)
for the noise floor, then a **paired A/B in both orders**. If the orders disagree,
you measured drift - warm-up, clocks, other tenants - not your change.

## Reading the report

- **Kernel table, "Python location"** is the answer to "which code". Trust it.
  Multiple sites with shares mean one kernel shape is reached from several call
  sites (a GEMM used by `fc1/fc2` and by `qkv_proj`) - information, not noise.
- **`transformers/models/...` there is a real answer, not a failure.** Where a
  stage's compute lives in a vendored HF module and omni only wraps it (Code2Wav:
  omni contributes the graph runner and scheduler, transformers every kernel), the
  transformers line owns the kernel. Same for `torchaudio/`. What you cannot act on
  is a *torch runtime* frame (`torch/nn/modules/linear.py`, `torch/nn/functional.py`):
  the mapping trace lost the caller - a gap, not a finding.
- **Overlap table's "Python scope" is a majority vote**, not the kernel table's top
  site: it attributes each launch by time window and reports the most common site,
  so a kernel split `:123` 77% / `:93` 23% can show `:93`. For the line to edit,
  use the kernel table.
- **Fuse table's "Candidate fused Python path"** cites an LLM-oriented catalog of
  upstream `python/sglang/srt/...` paths. The pattern match can be real; the
  destination usually is not omni's.

## Omni-specific reality

The LLM north star - "tensor cores never idle" - is wrong for most omni stages:

- Feature extraction and vocoder stages are **memory-bound** (conv1d, STFT,
  depthwise, resample). The fix is fusion or fewer passes, not occupancy.
- Streaming vocoder windows are **launch-bound**; the kernel table looks flat and
  cheap while wall clock is launch count and gaps. Read the overlap table's gaps,
  consider CUDA graphs or coalescing.
- A stage is a **pipeline process** (preprocess / encoder / LLM / vocoder / post),
  not prefill/decode. Profile the slow stage in isolation first; for cross-stage
  handoff use `sglang_omni/profiler/views.py`, not this skill.
- **After `torch.compile`** kernels become `triton_poi_fused_*` with stacks in
  generated code. For the mapping trace, disable compile as well as graphs.

## Where this sits

Stage 2 of three, and worth little without the other two.

1. **Triage** - find the *stage* from the request event timeline, before any
   profiler. Read `py-spy` **per thread**, not aggregated: an omni stage's cost is
   usually one thread's and the aggregate hides which. For TTS, measure cold
   reference and hot reference separately - different workloads, and a mean over
   both describes neither.
2. **GPU deep dive** - this skill: one stage, one shape, the trace pair, the kernel
   table's python location. Escalate to `nsys` for SM headroom only when the report
   leaves no clear change point.
3. **Validation** - the **After** gate above.

`model-profiling` owns stage 1 and the record of the outcome: it plans the run,
gets human confirmation, tracks findings. Use it first, and go back to it with the
answer.
