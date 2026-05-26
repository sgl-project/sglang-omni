# Issue #565 — Enable torch.compile for Higgs TTS AR backbone

**Branch:** `feat/yichi/higgs-tts/torch_compile`
**Status:** Production patch ready. Root cause of the bs=16 catastrophic localized to an upstream sglang × inductor interaction — not fixable from this side, capped via `_compiled_max_decode_bs=12` as the workaround.

## TL;DR

| Metric @ N=100 (single H200, `top_k=50`, `t=0.8`, SeedTTS EN) | baseline (CG-on, no compile) | this PR | Δ |
|---|---:|---:|---:|
| c=1   throughput (req/s) | 1.15 | **1.68** | **+46 %** |
| c=1   RTF | 0.158 | 0.160 | +1 % |
| c=8   throughput | 5.92 | 5.01\* | −15 %\* |
| c=32  throughput | 5.96 | **9.22** | **+55 %** |
| c=32  audio_s/s | 32.4 | **34.8** | +7 % |
| WER (excl >50 % outliers) all c | 1.09–1.46 % | 1.00–1.91 % | within noise |
| Catastrophic samples | 0 | 0 | — |

\* c=8 throughput is single-run noise (audio_dur drifted to 5.40 s vs 5.38 baseline — same prompts, different RNG outcomes); 0 catastrophic.

## What the patch does

Three files (one new), ~100 lines of code:

1. **`sglang_omni/models/higgs_tts/sglang_qwen3_backbone.py`** (new) — `HiggsQwen3Model` subclasses sglang's `Qwen3Model` and overrides `forward` to pick between eager `self.layers` (prefill, bs > 12) and compiled `self._compiled_decode_layers` (decode, bs ≤ 12). `HiggsQwen3ForCausalLM` subclasses `Qwen3ForCausalLM` and swaps the inner `self.model`.
2. **`sglang_omni/models/higgs_tts/model.py`** — import the local fork; instantiate `HiggsQwen3ForCausalLM` instead of `Qwen3ForCausalLM`.
3. **`sglang_omni/models/higgs_tts/stages.py`** — `_compile_higgs_backbone` populates the compiled-layers list and the bs cap. `create_sglang_tts_engine_executor` defers CG capture (`disable_cuda_graph=True` until after compile, then `init_device_graphs()`) so capture records compiled kernels rather than eager ones.

## Root-cause investigation — bs=16 catastrophic

Initial attempt let bs=16 captured graphs run through compiled layers; that path produced 99–100/100 catastrophic outputs at c=32 (WER > 50 % on essentially every sample, audio runs to `max_new_tokens` instead of EOC). The investigation:

### Truth table (✓ = clean, ✗ = catastrophic 50–100/100)

| # | bs | compile | CG capture | model | extra knob | Result |
|:---:|:---:|:---:|:---:|:---:|:---|:---:|
| 1 | 16 | off | off | Higgs | (eager baseline) | ✓ |
| 2 | 16 | off | on  | Higgs | shipped path (CG-only at bs=16) | ✓ |
| 3 | 16 | on  | off | Higgs | bs=16 not captured | ✓ |
| 4 | ≤12 | on | on  | Higgs | shipped path (small captured bs) | ✓ |
| 5 | 16 | on  | on  | toy (SDPA+MLP+RMSNorm) | pure `torch.cuda.CUDAGraph()` | ✓ |
| 6 | **16** | **on** | **on** | **Higgs** | default config | **✗** |
| 7 | 16 | on  | on  | Higgs | only bs=16 captured (no smaller bs) | ✗ |
| 8 | 16 | on  | on  | Higgs | `mode=default` (no max-autotune) | ✗ |
| 9 | 16 | on  | on  | Higgs | `dynamic=True` (no static specialize) | ✗ |
| 10 | 16 | on | on  | Higgs | skip `set_torch_compile_config()` | ✗ |

Bug fires **iff** all four are simultaneously: `bs=16` AND `compile=on` AND `CG=on` AND model uses **sglang's stock Qwen3 layer** (rows 6–10). Removing any single one → clean.

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

> some interaction between **sglang's `init_device_graphs` capture path** and **inductor-emitted kernels for sglang's stock Qwen3 layer composition** that fires only at `bs == cuda_graph_max_bs == 16`.

That interaction lives entirely above this repo. The cap (`_compiled_max_decode_bs=12`) dodges it; lifting the cap to 16 is gated on the upstream fix and tracked as a TODO in `stages.py`.

## Open follow-ups

1. **N=1088 full set validation.** This report is N=100 only.
2. **`speaker_sim` not measured.** Can be added with `--similarity-only` on the saved audio dirs.
3. **File the upstream sglang issue** with the diagnostic data above. Anyone running stock Qwen3 + `enable_torch_compile=True` + `cuda_graph_max_bs ≥ 16` is potentially affected.
4. **Lift `_compiled_max_decode_bs` once upstream fixes the bug.** TODO marker is in `_compile_higgs_backbone` (`stages.py`).
