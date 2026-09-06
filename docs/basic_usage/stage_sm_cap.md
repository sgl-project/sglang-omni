# Per-stage SM cap

When several GPU stages of one pipeline run as separate OS processes on one GPU behind CUDA MPS,
MPS shares the device freely between them. That is usually what you want. It is not always what
you want: if one stage issues short bursts of wide, SM-hungry kernels while another runs a
latency-sensitive autoregressive loop, the bursty stage can hold most of the SMs for milliseconds
at a time and stall the other one.

`sm_cap` puts a ceiling on one stage's SM usage, using a CUDA
[Green Context](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html).

**This is not a general speedup and there is no portable default.** On one pipeline the stage
worth capping is the vocoder and capping the engine does nothing; on another it is exactly the
reverse, and capping the wrong stage costs far more than capping the right one gains. Measure
first, with the procedure in [Choosing the cap](#choosing-the-cap).

## How it works

A preloaded bootstrap library creates a green context covering the requested number of SMs and
makes it current for the stage process before CUDA is initialized, including every thread the
process creates later. The stage is then provisioned that many SMs instead of the whole device.

Two cases documented by NVIDIA let a green context exceed its provisioned set, so treat the cap as
a strong bound rather than a hard one: with `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` set, MPS may scale
the SM set up to the client's share (serving refuses to start in that combination), and on compute
capability 9.x, loading a module that uses dynamic parallelism grants all green contexts 2 extra
shared SMs.

Nothing about scheduling changes: no new scheduler, no added synchronization, no change to the
request path. The only effect is on how many SMs one process is provisioned.

A cap is an upper bound, not a reservation. Two capped stages may cover overlapping SMs, and an
uncapped stage still sees the whole device.

## Requirements

- The capped stage must own its OS process (`--<stage>.process <name>`). A green context is
  process-wide, so stages sharing a process share one cap and may not disagree about it.
- Each GPU stage needs a `gpu_memory_fraction` once the process split makes them separate
  processes.
- A driver with the Green Context API (`cuGreenCtxCreate`), and CUDA MPS, as with any same-GPU
  multi-process serving. See [MPS data parallel](mps_dp.md).
- `sm_cap` must be a multiple of 8, the SM group size on H100 and H200. Other granularities are
  not supported yet: the group count is derived from that constant rather than negotiated with the
  driver.
- The capped process must see exactly one CUDA device, so that the cap cannot land on a different
  device than the stage runs on. Narrow `CUDA_VISIBLE_DEVICES` per worker, as
  [MPS data parallel](mps_dp.md) already does.

## Build the bootstrap

```bash
make -C tools/green_ctx
export SGLANG_OMNI_SM_CAP_BOOTSTRAP=$PWD/tools/green_ctx/libgreen_ctx_bootstrap.so
```

`SGLANG_OMNI_SM_CAP_BOOTSTRAP` must be set whenever any stage declares `sm_cap`; serving fails to
start otherwise rather than running uncapped.

## Enable it

```bash
python -m sglang_omni.cli serve --config <config> \
  --vocoder.process vocoder \
  --tts_engine.gpu_memory_fraction 0.20 \
  --vocoder.gpu_memory_fraction 0.05 \
  --vocoder.sm_cap 80
```

or in the pipeline config:

```yaml
stages:
  vocoder:
    process: vocoder
    gpu_memory_fraction: 0.05
    sm_cap: 80
```

At startup the capped process verifies that it really is running in the capped context, on its
main thread and on a freshly created one, comparing context identity rather than SM count, and
**fails closed** if either check fails. A stage that silently ignored its cap would look identical
to a working one in every metric except throughput, which is the failure this check exists to
prevent. The fresh thread matters because a library that was merely loaded, rather than preloaded,
does not interpose `pthread_create` and so cannot bind one.

## Choosing the cap

Run the pipeline at your target replica count and concurrency and measure the **one-sided capacity
curve** of each GPU stage: cap exactly one stage to `s` SMs, leave the others uncapped, record
throughput. `32, 40, 48, 56, 64, 72, 88` is enough to see the shape.

| Observation | Meaning | Action |
| --- | --- | --- |
| Capping stage X raises throughput above the uncapped baseline | X was displacing the others | cap X, size as below |
| Capping stage X leaves throughput flat | X has slack; capping it is free but pointless | no cap |
| Capping stage X lowers throughput at every size | X needs the whole device | no cap |

If no stage's curve rises above the uncapped baseline by a margin you care about, this feature is
not for your pipeline. Keep plain MPS.

**Sizing.** The top of the curve is broad and its lower edge is sharp. On the Qwen3-TTS
measurement below, four adjacent sizes (72 to 96 SM) land within 1.3% of each other, one step
lower is worth only +2.6%, and two steps lower is *worse than no cap at all*: squeeze a stage
enough and it becomes the bottleneck itself. Pick the middle of the plateau, never its lower edge.
If your curve is still rising at the largest size you measured, extend the sweep instead of
shipping the last point you happened to try.

**Do not assume a load threshold.** On the pipeline below the cap is worth +12.8% at 8 in-flight
requests per replica and +14.8% at 20. Measure your own operating point.

## Measured example

H200 (132 SM), single card, Seed-TTS EN (`benchmarks.eval.benchmark_tts_seedtts`, 1088 requests,
`--stream --response-format pcm`), closed-loop fixed concurrency, arms interleaved within one
session, throughput corrected for runaway requests. Every arm ran with 0 failed requests.

These runs predate this change and set the same `GREEN_CTX_*` variables through each stage's `env`
block, which is exactly what `sm_cap` now derives; the mechanism is identical but the numbers were
not re-measured through this surface. What was re-measured on it is correctness: a capped
Qwen3-TTS run completes with the cap verified and 0 failures. Throughput in req/s. Note that the
stage worth capping differs between pipelines:

| Pipeline | Topology | Uncapped | Capped | Delta | RTF p95 |
| --- | --- | --- | --- | --- | --- |
| Qwen3-TTS-12Hz-1.7B | 4 replicas, c80, **vocoder** cap 80 | 25.51 | 30.05 | +17.8% | 1.28 → 0.85 |
| Qwen3-TTS-12Hz-1.7B | 3 replicas, c60, **vocoder** cap 72 | 22.85 | 26.73 | +17.0% | 0.87 → 0.75 |
| MOSS-TTS-Local | 2 replicas, c32, **engine** cap 48 | 9.26 | 14.02 | +51.4% | 1.99 → 0.68 |
| Higgs-TTS | 3 replicas, c48, vocoder cap 88 | 29.30 | 29.96 | +2.3% (not significant) | 0.82 → 0.63 |

Capping the wrong stage on the same pipelines: capping the Qwen3-TTS engine wins at no size, and
capping the MOSS-TTS-Local vocoder loses 40 to 57% with RTF p95 rising from 1.99 to 5.37. Higgs is
the case where nothing is worth capping, and the one-sided curves say so in advance: both of its
stages need most of the device.

## Costs and caveats

- Splitting a stage into its own process adds one CUDA context per replica (order 1-2 GiB). The
  cap itself costs no memory.
- A cap below the plateau is a net loss.
- The bootstrap only activates in `python` processes. Helper binaries a stage spawns (for example
  `ldconfig` from `torch.inductor`) inherit `LD_PRELOAD`; initializing CUDA inside them fails and
  their non-zero exit would break the caller.
- An `LD_PRELOAD` inherited from the parent is preserved: the bootstrap is prepended to it.
  `GREEN_CTX_*` variables are derived from `sm_cap` and are rejected if set by hand, in the parent
  environment or in a stage's `env` block.
- NVIDIA documents that "a green context can be current to only one thread at a time" and that
  there is no internal synchronization for concurrent access. This bootstrap converts the green
  context with `cuGreenCtxCreate` plus `cuCtxFromGreenCtx` and makes the resulting ordinary
  `CUcontext` current on every thread, which is how a multi-threaded stage can use it at all.
  Measurements across the pipelines above ran without failures or output corruption, and the
  corruption we did see came from threads landing on *different* contexts, not from sharing one.
  Even so, this usage sits in a documented grey area and is worth a maintainer's judgement.
- The bootstrap writes its startup line to stderr. Under Nsight Systems that output is buffered,
  so do not gate a profiled run on scraping it; the in-process check above does not depend on it.
