# Issue #565 — Enable `torch.compile` for Higgs TTS AR backbone

**Branch:** `feat/yichi/higgs-tts/torch_compile`
**Status:** Production patch caps `_compiled_max_decode_bs=12` as the workaround. Root cause of the bs=16/32 catastrophic localized to an upstream sglang × inductor interaction. Multiple `sglang`-side fix candidates tried — one (V13: ascending capture order) eliminates the runaway and matches baseline quality, but **does not deliver throughput**. Holding the workaround pending an upstream owner.

---

## TL;DR

| Metric (single H200, SeedTTS EN) | Baseline (CG-on, no compile) | This PR (compile up to bs=12, eager bs≥13) |
|---|---:|---:|
| N=100, c=1   throughput (req/s) | 1.15 | **1.68**  (+46 %) |
| N=100, c=1   RTF | 0.158 | 0.160 |
| N=100, c=32  throughput | 5.96 | **9.22**  (+55 %) |
| N=100, c=32  audio_s/s | 32.4 | **34.8** |
| N=1088, c=16 WER excl > 50 % | 1.36 % (PR #534 report) | 1.32 % |
| Catastrophic samples | 0 | 0 |

At c=16 we ship `_compiled_max_decode_bs=12`, so `bs==16` decode falls back to eager and dodges the upstream bug. Throughput at low concurrency is +46 %, but at c=16 with `max_running_requests=16` (and thus a fully compiled CG range up to bs=16) the bug fires; without the bs=12 cap the WER explodes.

---

## What the patch does

Three files (one new), ~100 lines of code:

1. **`sglang_omni/models/higgs_tts/sglang_qwen3_backbone.py`** (new) — `HiggsQwen3Model` subclasses sglang's `Qwen3Model` and overrides `forward` to pick between eager `self.layers` (prefill, bs > `_compiled_max_decode_bs`) and compiled `self._compiled_decode_layers` (decode, bs ≤ cap). `HiggsQwen3ForCausalLM` subclasses `Qwen3ForCausalLM` and swaps the inner `self.model`.
2. **`sglang_omni/models/higgs_tts/model.py`** — import the local fork; instantiate `HiggsQwen3ForCausalLM` instead of `Qwen3ForCausalLM`.
3. **`sglang_omni/models/higgs_tts/stages.py`** — `_compile_higgs_backbone` populates the compiled-layers list and the bs cap. `create_sglang_tts_engine_executor` defers CG capture (`disable_cuda_graph=True` until after compile, then `init_device_graphs()`) so capture records compiled kernels rather than eager ones.

---

## Root-cause investigation — bs=16 catastrophic

Initial attempt let bs=16 captured graphs run through compiled layers; that path produced 99–100/100 catastrophic outputs at c=32 (WER > 50 % on essentially every sample, audio runs to `max_new_tokens` instead of EOC).

### Truth table (✓ = clean, ✗ = catastrophic 50–100/100)

| # | bs | compile | CG capture | model | extra knob | Result |
|:---:|:---:|:---:|:---:|:---:|:---|:---:|
| 1 | 16 | off | off | Higgs | (eager baseline) | ✓ |
| 2 | 16 | off | on  | Higgs | shipped path (CG-only at bs=16) | ✓ |
| 3 | 16 | on  | off | Higgs | bs=16 not captured | ✓ |
| 4 | ≤12 | on | on | Higgs | shipped path (small captured bs) | ✓ |
| 5 | 16 | on  | on | toy (SDPA+MLP+RMSNorm) | pure `torch.cuda.CUDAGraph()` | ✓ |
| 6 | **16** | **on** | **on** | **Higgs** | default config | **✗** |
| 7 | 16 | on  | on | Higgs | only bs=16 captured (no smaller bs) | ✗ |
| 8 | 16 | on  | on | Higgs | `mode=default` (no max-autotune) | ✗ |
| 9 | 16 | on  | on | Higgs | `dynamic=True` (no static specialize) | ✗ |
| 10 | 16 | on | on | Higgs | skip `set_torch_compile_config()` | ✗ |

Bug fires **iff** all four hold simultaneously: `bs=16` AND `compile=on` AND `CG=on` AND model uses **sglang's stock Qwen3 layer** (rows 6–10). Removing any single one → clean.

### Rejected candidate root causes

| Hypothesis | Ruled out by | Why not the cause |
|---|:---:|---|
| Multi-graph **pool sharing** across captures | row 7 | Capturing only bs=16 (no smaller bs) is still 50/50 catastrophic. No other captures to share pool with. |
| **Aggressive autotune** kernel selection | row 8 | `mode=default` (no max-autotune) → still catastrophic. |
| **Static specialization** at largest bs | row 9 | `dynamic=True` forces dynamic shape codegen → still catastrophic. |
| `set_torch_compile_config()` inductor tweaks | row 10 | Skip the call entirely → still catastrophic. |
| **Torch-generic** CG × compile bug | row 5 | Toy SDPA + MLP + RMSNorm at bs=16 → bit-identical to eager. |
| **Compiled kernel is numerically wrong** | row 3 | Compile-on at bs=16 WITHOUT CG → 19800 per-layer hidden_states bit-identical to eager. Correct in isolation. |
| Higgs **shadow buffers / sampler scatter** | hidden_states dump | The 6 % systematic norm drop is already present in `hidden_states` right after the backbone's final RMSNorm — before any Higgs-specific code touches it. Downstream is innocent victim. |

### Captured-replay hidden_states at bs=16

| | broken (CG + compile bs=16) | shipped (CG only, no compile bs=16) |
|---|---:|---:|
| n samples | 69 | 626 |
| `hidden.norm` mean | **775.4** | **825.5** |
| `hidden.norm` range | 716–823 | 811–831 |
| NaN count | 0 | 0 |

Broken bs=16 output is **systematically 6 % below** the clean baseline — no NaN, no inf, just a clean L2 norm shift. Signature consistent with "kernel reads partially stale memory" or "accumulator drops some iterations" rather than gross corruption. Propagated through hundreds of AR decode steps it manifests as catastrophic text generation.

### Why this is an upstream bug

Comparing the clean and broken regimes at bs=16 (rows 2 vs 6), the only differing code is which layer wrapper the loop iterates:

- row 2: `self.layers[i]` — plain `Qwen3DecoderLayer` (sglang upstream)
- row 6: `self._compiled_decode_layers[i]` — `torch.compile(Qwen3DecoderLayer)` (still upstream, wrapped by upstream `torch.compile`)

Our wrapper code is trivial — no kernels, no math. The kernels actually executing inside the captured graph at bs=16 are **all from sglang upstream and PyTorch upstream** (`sgl_kernel.rmsnorm`, `silu_and_mul`, `fused_add_rmsnorm`, extern cuBLAS `mm`, inductor-emitted Triton). The capture mechanism (`init_device_graphs`) and the codegen (inductor) are both upstream.

Combined with the rejected candidates above, the bug is constrained to:

> some interaction between **sglang's `init_device_graphs` capture path** and **inductor-emitted kernels for sglang's stock Qwen3 layer composition** that fires only when the *largest* captured bs is also `torch.compile`-d.

That interaction lives entirely above this repo.

---

## sglang-side fix attempts (overnight pass)

After localising the bug upstream, I tried several fixes directly inside `sglang/srt/model_executor/cuda_graph_runner.py` to see if any of them could lift the bs=12 cap. Test rig: N=1088 SeedTTS EN, single H200, sample at `top_k=50 t=0.8`, `max_running_requests=16 cuda_graph_max_bs=16 _compiled_max_decode_bs=16 c=16` — i.e. bs=16 is BOTH captured AND compiled (the exact regime the cap currently dodges).

| Variant | sglang change | Outcome |
|---|---|---|
| V1 | `torch.cuda.synchronize()` + `torch.cuda.empty_cache()` between warmup and the `device.graph(...)` capture inside `capture_one_batch_size` | p99 = 19.0 s — no help |
| V3 | Two-phase: warm every bs first (no capture), sync + empty cache, then capture every bs | partial — kills runaway but adds 50 mid-severity outliers, p99 = 3.3 s |
| V5 | `torch.compile(layer, dynamic=True)` instead of static | p99 = 19.6 s — no help |
| V11 | Skip `set_torch_compile_config()` entirely (drop `coordinate_descent_tuning`, `fx_graph_cache`, dynamo cache-size raise, `monkey_patch_torch_compile`) | p99 = 18.7 s — no help |
| V10 | `compile_mode="default"` instead of `max-autotune-no-cudagraphs` | p99 = 19.4 s — no help |
| **V13** | **Reverse capture order: capture `[1, 2, 4, 8, 12, 16]` ascending instead of `[16, 12, 8, 4, 2, 1]`** | **p99 = 3.19 s** ✓ no runaway, but no throughput win either |

V13 — capturing in ascending order so the largest bs (also the first dynamo trace) happens **last** rather than first — was the only thing that killed the runaway. Quality-wise V13 matches baseline (WER excl > 50 % = 1.35 % at c=16, 1.27–1.38 % at c=32, vs baseline 1.36 %). But it does **not** deliver throughput: at c=32 QPS drops from 9.68 (no compile) to 8.5–9.15 (V13 + compile). At c=16 QPS is essentially flat (8.74 vs 8.41 my-baseline).

So V13 makes compile **safe** at the full bs range, but `torch.compile` at these workloads (decode-bound 4 B model, bs ≤ 32) just doesn't beat the existing eager kernels enough to pay for the extra compile/capture overhead.

### Full V13 comparison table

All 1088 SeedTTS-EN samples, single H200, ASR scorer on a separate GPU.

#### c=16, `max_running_requests=16`, `cuda_graph_max_bs=16` (so `_compiled_max_decode_bs=16` actually puts compile at the largest captured bs)

| Run | sglang | compile | WER corpus | WER excl > 50 % | Outliers | Outlier max | p99 (s) | QPS |
|---|---|---|---|---|---|---|---|---|
| Baseline | upstream (reverse) | off | 6.36 % | 1.29 % | 50 | 3 740 % | 3.14 | 8.41 |
| No fix | upstream (reverse) | **on** | (runaway) | — | runaway | — | **19.25** | 4.93 |
| **V13** | **ascending** | **on** | 4.61 % | 1.35 % | 50 | 2 785 % | **3.19** | **8.74** |

#### c=32, `max_running_requests=32`, `cuda_graph_max_bs=32`, `_compiled_max_decode_bs=32`

| Run | sglang | compile | WER corpus | WER excl > 50 % | Outliers | Outlier max | p99 (s) | QPS |
|---|---|---|---|---|---|---|---|---|
| Baseline | upstream (reverse) | off | 13.56 % | 1.28 % | 50 | 7 600 % | 5.38 | 9.68 |
| No fix | upstream (reverse) | **on** | (runaway) | — | runaway | — | **26.95** | 7.27 |
| **V13** | **ascending** | **on** | 9.55–9.61 % | 1.27–1.38 % | 50 | 4 100 % | 5.57–5.74 | 8.51–9.15 |

#### Reading this table

- The **50 outliers** that appear in every row, including the no-compile baseline, are Higgs's inherent flakiness at higher concurrency — not the compile×CG bug. With V13 the outliers are *less severe* (lower max %) than baseline; without V13 + with compile they become catastrophic runaways.
- **WER excl > 50 %** is the stable quality metric (matches PR #534's 1.36 %). All rows except the runaway ones are within noise of baseline.
- **Throughput**: V13 + compile is essentially flat vs no-compile baseline. `torch.compile` here does not pay back its overhead.

### Conclusion for upstream owner

V13 is a one-line patch in `sglang/srt/model_executor/cuda_graph_runner.py` `capture()`:

```diff
-            # Reverse the order to enable better memory sharing across cuda graphs.
-            capture_range = (
-                tqdm.tqdm(list(reversed(self.capture_bs)))
-                if get_tensor_model_parallel_rank() == 0
-                else reversed(self.capture_bs)
-            )
+            capture_range = (
+                tqdm.tqdm(list(self.capture_bs))
+                if get_tensor_model_parallel_rank() == 0
+                else iter(self.capture_bs)
+            )
```

It side-steps the bug consistently but doesn't *explain* it, and reversing the capture order has a memory-pool-reuse cost upstream may not want to pay. We're holding the bs=12 workaround in this PR until someone with deeper knowledge of inductor / CUDA-graph memory-pool interaction can pin the actual root cause.

---

## Open follow-ups

1. **N=1088 full-set validation of the shipped (cap=12) path** — current PR ships with the cap; the +46 % / +55 % numbers above are N=100. The full-set numbers under the cap are in the table directly above (the c=16 V13 row mostly applies because at `_compiled_max_decode_bs=12` bs=16 is eager anyway, and the V13 question reduces to whether compile at bs∈{1,2,4,8,12} pays for itself — it doesn't in our tests).
2. **`speaker_sim` not measured.** Can be added with `--similarity-only` on the saved audio dirs.
3. **File an upstream sglang issue** with the diagnostic data above + the V13 evidence. Anyone running stock Qwen3 + `enable_torch_compile=True` + `cuda_graph_max_bs ≥ 16` is potentially affected.
4. **Lift `_compiled_max_decode_bs` once upstream fixes the bug.** TODO marker is in `_compile_higgs_backbone` (`stages.py`).
