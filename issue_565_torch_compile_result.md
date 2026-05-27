# Issue #565 — Enable `torch.compile` for Higgs TTS AR backbone

**Branch:** `feat/yichi/higgs-tts/torch_compile`
**Status:** Production patch ready. The previous `bs ≤ 12` cap is no longer needed — a one-shot eager forward before sglang's CG capture loop side-steps the upstream `sglang × inductor × CG` corruption at the first compiled+captured bs.

---

## TL;DR

`torch.compile`'s first dynamo trace during sglang's CUDA-graph capture loop allocates / initializes lazy CUDA state (cuBLAS handles, allocator pool, etc.) **inside the captured graph**. Replay of that graph then reads garbage at the slot the lazy init laid down → model emits broken logits → audio runs to `max_new_tokens` → catastrophic outliers.

**The fix:** force all lazy CUDA init *outside* the capture window by running ONE eager forward at `bs=1` after `_compile_higgs_backbone` but before `init_device_graphs()`. ~12 lines in `stages.py`, no sglang patch required.

Result vs no-compile baseline (SeedTTS-EN N=1088, single H200):

| Config | Δ vs baseline |
|---|---|
| c=16 / mr=16 / cap=16 | outliers **50 → 2** (−96 %), WER excl 1.29 → 1.31 %, QPS 8.41 → 8.49 (+1 %) |
| c=32 / mr=32 / cap=32 | outliers **50 → 4** (−92 %), WER excl 1.28 → 1.24 %, QPS 9.68 → 9.42 (−3 %) |

Note: `torch.compile` itself doesn't deliver throughput on this workload (4 B decode-bound, bs ≤ 32). The PR's value is now (a) compile is **safe** to enable across the full bs range, and (b) the eager pre-warmup as a side-effect cuts most of Higgs's inherent c≥16 outliers from ~50/run to ~2–4/run.

---

## What the patch does

Three files (one new), ~110 lines of code:

1. **`sglang_omni/models/higgs_tts/sglang_qwen3_backbone.py`** (new) — `HiggsQwen3Model` subclasses sglang's `Qwen3Model` and overrides `forward` to pick between eager `self.layers` (prefill) and compiled `self._compiled_decode_layers` (decode). `HiggsQwen3ForCausalLM` subclasses `Qwen3ForCausalLM` and swaps the inner `self.model`.
2. **`sglang_omni/models/higgs_tts/model.py`** — import the local fork; instantiate `HiggsQwen3ForCausalLM` instead of `Qwen3ForCausalLM`.
3. **`sglang_omni/models/higgs_tts/stages.py`** —
   - `_compile_higgs_backbone`: populate `_compiled_decode_layers`.
   - `_warmup_eager_pre_cg` (NEW, ~12 lines): temporarily nulls `_compiled_decode_layers` (so dispatch falls back to eager `self.layers`), calls `model_runner._dummy_run(batch_size=1)`, restores the compiled list. Runs after `_compile_higgs_backbone` and before `init_device_graphs()`.
   - `create_sglang_tts_engine_executor`: defers CG capture (`disable_cuda_graph=True` until after compile, then `init_device_graphs()`) so capture records compiled kernels.

---

## Bug — what happens without the fix

When `torch.compile` covers the largest bs in sglang's capture set, the CUDA graph for that bs returns corrupted logits on replay. Model never samples EOC → audio runs to `max_new_tokens=2048` (~80 s of audio per outlier). c=16 p99 latency goes from ~3 s (baseline) to ~19 s (no fix); c=32 p99 goes from ~5 s to ~27 s.

### Truth table (✓ = clean, ✗ = catastrophic 50–100/100)

| # | bs | compile | CG | model | extra knob | Result |
|:---:|:---:|:---:|:---:|:---:|:---|:---:|
| 1 | 16 | off | off | Higgs | eager baseline | ✓ |
| 2 | 16 | off | on  | Higgs | CG-only at bs=16 | ✓ |
| 3 | 16 | on  | off | Higgs | bs=16 not captured | ✓ |
| 4 | ≤12 | on | on | Higgs | small captured bs only | ✓ |
| 5 | 16 | on  | on | toy (SDPA+MLP+RMSNorm) | pure `torch.cuda.CUDAGraph` | ✓ |
| 6 | **16** | **on** | **on** | **Higgs** | default config | **✗** |
| 7 | 16 | on  | on | Higgs | only bs=16 captured (no smaller bs) | ✗ |
| 8 | 16 | on  | on | Higgs | `mode=default` (no max-autotune) | ✗ |
| 9 | 16 | on  | on | Higgs | `dynamic=True` (no static specialize) | ✗ |
| 10 | 16 | on | on | Higgs | skip `set_torch_compile_config()` | ✗ |

Bug fires iff: `compile=on` AND `CG=on` AND the bs being decoded is **the first one compiled inside sglang's capture loop**. Remove any single knob → clean.

### Numerical signature

Captured-replay `hidden_states` at the broken bs:

| | broken | clean |
|---|---:|---:|
| `hidden.norm` mean | **775.4** | **825.5** |
| NaN / inf | 0 | 0 |

Systematic ~6 % L2-norm drop, no NaN — consistent with "kernel reads partially stale memory", not gross corruption. Propagated through hundreds of AR decode steps it manifests as wrong text content / runaway output.

### Ruled-out hypotheses

| Hypothesis | Killed by | Why not |
|---|:---:|---|
| Multi-graph **pool sharing** across captures | row 7 | Only bs=16 captured → still ✗ |
| **Aggressive autotune** kernel selection | row 8 | `mode=default` → still ✗ |
| **Static specialization** at largest bs | row 9 | `dynamic=True` → still ✗ |
| `set_torch_compile_config` inductor tweaks | row 10 | Skip the call → still ✗ |
| **Torch-generic** CG × compile bug | row 5 | Toy model at bs=16 → bit-identical to eager |
| **Compiled kernel is wrong on its own** | row 3 | Compile-on, CG-off at bs=16 → 19 800 per-layer hidden_states bit-identical to eager |
| Higgs **shadow buffers / sampler scatter** | dump | 6 % norm drop already present in `hidden_states` *before* any Higgs-specific code touches it |

---

## How V14 (the eager pre-warmup) was reached

Six fix attempts inside sglang or in stages.py at the failing config (mr=16/cap=16/c=16, where bs=16 is captured+compiled):

| Variant | Change | Outcome |
|---|---|---|
| V1 | `cuda.synchronize() + empty_cache()` between warmup and capture | p99 = 19.0 s — no help |
| V3 | Two-phase: warmup every bs first (no capture), then capture every bs | partial — kills runaway but adds 50 mid-severity outliers |
| V5 | `torch.compile(layer, dynamic=True)` | p99 = 19.6 s — no help |
| V10 | `compile_mode="default"` | p99 = 19.4 s — no help |
| V11 | Skip `set_torch_compile_config()` entirely | p99 = 18.7 s — no help |
| V13 | Capture in ASCENDING order (largest bs last) | clean at c=16 but **catastrophic at c=1** — just moves the bug from bs=largest to bs=1 |
| **V14** | **Eager `_dummy_run(bs=1)` before `init_device_graphs()`** | **clean across c=1 / c=16 / c=32** ✓ |

V13's failure at c=1 was the key signal: the bug isn't tied to "the largest bs", it's tied to **the first bs the dynamo trace runs against in sglang's capture loop**. Whatever cold CUDA init that trace pulls in lands inside the captured graph. V14 does that cold init separately in an eager forward, then sglang's capture sees a fully warm CUDA state.

---

## Full results — Baseline vs V14

All numbers: SeedTTS-EN, N=1088, single H200, `top_k=50 t=0.8`, ASR scorer on a separate GPU.

### c=16, max_running=16, cap=16

| Metric | Baseline (no compile) | **V14** | Δ |
|---|---:|---:|---:|
| Latency mean | 2.17 s | 1.78 s | −18 % |
| Latency p99 | 3.14 s | 2.98 s | −5 % |
| QPS | 8.41 | 8.49 | +1 % |
| Output tokens mean | 104 | 123 | +18 % |
| Audio duration mean | 3.86 s | 4.64 s | +20 % |
| WER excl > 50 % | 1.29 % | 1.31 % | +0.02 pp (noise) |
| Outliers (> 50 %) | 50 | **2** | **−96 %** |

### c=32, max_running=32, cap=32

| Metric | Baseline (no compile) | **V14** | Δ |
|---|---:|---:|---:|
| Latency mean | 3.31 s | 3.31 s | 0 |
| Latency p99 | 5.38 s | 5.56 s | +3 % |
| QPS | 9.68 | 9.42 | −3 % |
| Output tokens mean | ~150 | 121 | −19 % |
| WER excl > 50 % | 1.28 % | **1.24 %** | **−3 %** (relative) |
| Outliers (> 50 %) | 50 | **4** | **−92 %** |
| WER per-sample max | 7 600 % | 2 793 % | −63 % |

### c=1, N=50 (smoke test)

| Metric | Baseline | **V14** |
|---|---:|---:|
| Latency mean | 0.79 s | 0.79 s |
| Output tokens mean | 154 | 144 |
| Audio duration mean | 5.88 s | 5.48 s |

---

## Open follow-ups

1. **`torch.compile` doesn't actually deliver throughput on this workload.** V14 is within ±3 % of baseline at the configs we tested. The PR's current value is "compile is safe to enable" + "outliers cut by ~95 %", not raw speed. Worth discussing whether to keep the compile path on by default.
2. **Root cause unknown.** V14 sidesteps the bug but doesn't *explain* what specific lazy CUDA state the first dynamo trace is laying down. A teammate with deeper inductor + CUDA-graph knowledge could pin this down and file an upstream sglang issue.
3. **`speaker_sim` not measured.** Worth adding with `--similarity-only` on the saved audio dirs.
