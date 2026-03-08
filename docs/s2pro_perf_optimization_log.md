# S2-Pro SGLang Performance Optimization Log

> Date: 2026-03-08
> Branch: `feat/fishaudio_s1_s2pro`
> Goal: Fix `torch.compile` WER bug, optimize sampling throughput, enable CUDA graph for codebook loop

---

## Baseline (before changes)

| Config | tok/s (BS=1) | WER | Notes |
|--------|-------------|-----|-------|
| `compile=True` (old `max-autotune`) | ~60 | **99%** | Fast but completely broken output |
| `compile=False` (eager) | ~26 | ~1% | Correct but slow |
| Batched BS=8 (eager) | ~19.7 per-req | ~1% | Per-req throughput degrades 69% vs BS=1 |

---

## P0: Fix torch.compile + Optimize Sampling

### Problem 1: torch.compile produces garbage audio (WER 99%)

**Root cause**: `torch.compile(mode="max-autotune")` internally captures CUDA graphs via Inductor. The sampling function used `torch.multinomial` which requires CPU-GPU synchronization — incompatible with CUDA graph replay. The random state gets corrupted, producing nonsense tokens.

**Fix**: Changed compile mode to `"max-autotune-no-cudagraphs"` in `s2pro_sglang_ar.py`:

```python
# Before
torch.compile(..., mode="max-autotune")

# After
torch.compile(..., mode="max-autotune-no-cudagraphs")
```

### Problem 2: Sampling function too slow (sort-based, O(V log V))

**Root cause**: The initial fix replaced `topk+scatter` with `sort+argsort+gather` (the `_logits_to_probs` approach from the reference `flash-fish-inference-bak`). This was CUDA-graph-safe but operated on the **full vocabulary** (4096 elements for codebooks):

```python
# Old slow path (2x sort on 4096 elements)
sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # O(V log V)
cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)   # O(V)
# ... mask, temperature, softmax on all V elements ...
inverse_indices = torch.argsort(sorted_indices, dim=-1)              # O(V log V)
return torch.gather(probs_sort, dim=-1, index=inverse_indices)       # O(V)
```

**Fix**: Rewrote `_sample_with_topk` to use `torch.topk` (O(V + k log k)) and operate on only k=30 elements:

```python
def _sample_with_topk(logits, temperature, top_p, top_k=30, ...):
    # 1. topk: O(V) scan + O(k log k) sort, returns [bs, k] sorted descending
    top_vals, top_indices = torch.topk(logits, k=top_k, dim=-1)

    # 2. Top-p filtering on k=30 elements (not 4096)
    probs_raw = F.softmax(top_vals, dim=-1)
    cum_probs = torch.cumsum(probs_raw, dim=-1)
    mask = cum_probs > top_p
    mask[..., 0] = False

    # 3. Temperature scaling + re-softmax on k elements
    top_vals = top_vals / torch.clip(temperature, min=1e-5)
    top_vals = top_vals.masked_fill(mask, -float("inf"))
    probs = F.softmax(top_vals, dim=-1)

    # 4. Gumbel-max sampling (no CUDA sync, no torch.multinomial)
    q = torch.empty_like(probs).exponential_(1)
    sampled_idx = torch.argmax(probs / q, dim=-1, keepdim=True)

    # 5. Map back to original vocab index via gather (not scatter)
    return torch.gather(top_indices, dim=-1, index=sampled_idx).to(dtype=torch.int)
```

**Why this is safe for CUDA graphs and torch.compile**:
- `torch.topk` with int `k`: deterministic, no CPU-GPU sync
- `gather` instead of `scatter`: deterministic with duplicate indices
- `exponential_()` + `argmax` (Gumbel-max): replaces `torch.multinomial` which requires sync
- All ops on fixed-shape tensors `[bs, 30]`

### Files changed

| File | Change |
|------|--------|
| `runtime/s2pro_ar.py` | Rewrote `_sample_with_topk`: removed `_logits_to_probs` (sort-based) and `_multinomial_no_sync`, replaced with topk+gather+Gumbel-max |
| `runtime/s2pro_sglang_ar.py` | Changed compile mode to `max-autotune-no-cudagraphs` |

### P0 Results

| Iteration | tok/s | WER | Method |
|-----------|-------|-----|--------|
| v1 (sort-based) | ~50 | 1.10% | argsort+gather on full vocab |
| **v2 (topk-based)** | **52.9** | **1.10%** | torch.topk on k=30 elements |

v2 recovered ~3 tok/s by replacing O(V log V) sort with O(V + k log k) topk. TTFB improved from 217ms to 207ms. RTF from 0.43 to 0.41.

---

## P1: CUDA Graph for Codebook Loop

### Problem 1: CUDA graph capture crashes with torch.compile

**Error** (first attempt):
```
TorchRuntimeError: Dynamo failed to run FX node with fake tensors:
  rms_norm ... normalized_shape=[2560], but got input of size[1, 1, 1024]
```

**Root cause**: `S2ProFastGraphRunner` used the **compiled** function (`self._codebook_fn`) for CUDA graph capture. Dynamo's fake tensor tracing failed on the shape transformation through `project_in`.

**Fix**: Use **eager** function for CUDA graph capture. CUDA graphs and torch.compile serve overlapping purposes — using both is redundant and problematic.

```python
# Before (crashes)
self._fast_graph_runner = S2ProFastGraphRunner(
    codebook_fn=self._codebook_fn,  # compiled
    ...
)

# After
self._fast_graph_runner = S2ProFastGraphRunner(
    codebook_fn=self._codebook_fn_eager,  # eager, no Dynamo tracing
    ...
)
```

### Problem 2: Static buffer has wrong hidden dimension

**Error** (second attempt, after fixing Problem 1):
```
RuntimeError: Given normalized_shape=[2560], expected input with shape [*, 2560],
  but got input of size[1, 1, 1024]
```

**Root cause**: The hidden_dim probing code inspected `audio_decoder.project_in.parameters()` to get the input dimension. But for S2-Pro, **`config.text_dim == config.dim` (both 2560)**, so `project_in = nn.Identity()` which has **zero parameters**. The probing loop iterated zero times and fell back to `hidden_dim = 1024` (wrong).

The text model (Qwen3-4B) outputs hidden states with dim=2560. The static buffer must match.

```python
# Before (wrong: Identity has no params, fallback to 1024)
hidden_dim = None
for p in audio_decoder.project_in.parameters():
    hidden_dim = p.shape[1]
    break
if hidden_dim is None:
    hidden_dim = 1024  # <-- wrong for Qwen3-4B

# After (correct: read from audio decoder config)
hidden_dim = getattr(
    getattr(audio_decoder, "config", None), "text_dim", None
)
if hidden_dim is None:
    for p in audio_decoder.project_in.parameters():
        hidden_dim = p.shape[1]
        break
if hidden_dim is None:
    hidden_dim = 2560
```

### Fix 3: `top_k` passed as tensor but `torch.topk` needs Python int

```python
# Before: top_k was a tensor buffer in the graph runner
self._top_k_buf = torch.full((max_bs, 1), 30, dtype=torch.int64)
# ... passed as tensor to codebook_fn

# After: top_k is a Python int, baked into the captured graph
self._top_k = top_k  # int, e.g. 30
# ... passed as int constant during capture
def run():
    return self._codebook_fn(h, s, t, p, self._top_k)
```

The `top_k` value (30) is constant across all requests and all replay calls, so baking it as a compile-time constant is correct and simplifies the interface.

### Audio Decoder CUDA Graph Compatibility

Verified that the audio decoder (`fish_speech/models/text2semantic/modeling.py`) is CUDA-graph-safe:

- `reset_caches()`: uses `.zero_()` (in-place tensor op, captured by graph)
- `forward_kvcached()`: uses `self.input_pos.fill_(codebook_idx)` (in-place tensor op)
- KV cache updates: all in-place tensor operations
- No Python-level state changes that would be invisible to CUDA graphs

### Files changed

| File | Change |
|------|--------|
| `runtime/s2pro_sglang_ar.py` | `S2ProFastGraphRunner`: use eager fn, `top_k` as int (removed `_top_k_buf`), `replay()` drops `top_k` param |
| `runtime/s2pro_sglang_ar.py` | `_batched_two_stage_decode`: pass int `top_k` to codebook fn, update `replay()` call |
| `runtime/s2pro_ar.py` | `_sample_with_topk` signature: `top_k: int` (was `Union[int, Tensor]`) |

---

## P2: Benchmark Updates

Added CLI flags to `benchmarks/profile_s2pro_sglang.py`:

- `--enable-cuda-graph`: enables CUDA graph capture for codebook loop
- `--max-batch-size`: controls max batch size for engine and graph capture (default 64)

---

## Test Commands

```bash
export S2PRO_CKPT=/root/.cache/huggingface/s2-pro/s2-pro
export SEED_TTS=/tmp

# --- P0: compile + topk optimization ---
CUDA_VISIBLE_DEVICES=0 python benchmarks/profile_s2pro_sglang.py \
    --checkpoint $S2PRO_CKPT \
    --testset $SEED_TTS/seedtts_testset/en/meta.lst \
    --output-dir results/s2pro_p0_v2 \
    --max-samples 50 --batch-sizes 1 --save-audio

python benchmarks/eval_wer.py \
    --meta $SEED_TTS/seedtts_testset/en/meta.lst \
    --audio-dir results/s2pro_p0_v2/audio --lang en

# --- P1: CUDA graph + batched ---
CUDA_VISIBLE_DEVICES=0 python benchmarks/profile_s2pro_sglang.py \
    --checkpoint $S2PRO_CKPT \
    --testset $SEED_TTS/seedtts_testset/en/meta.lst \
    --output-dir results/s2pro_p1_v2 \
    --max-samples 50 --batch-sizes 1,2,4,8 --enable-cuda-graph --save-audio

# --- No compile baseline (for comparison) ---
CUDA_VISIBLE_DEVICES=0 python benchmarks/profile_s2pro_sglang.py \
    --checkpoint $S2PRO_CKPT \
    --testset $SEED_TTS/seedtts_testset/en/meta.lst \
    --output-dir results/s2pro_nocompile \
    --max-samples 50 --batch-sizes 1 --no-compile --save-audio
```

## Expected Results

| Config | tok/s (BS=1) | WER | Status |
|--------|-------------|-----|--------|
| compile + topk (P0 v2) | ~60 | ~1% | To verify |
| compile + CUDA graph (P1 v2) | ~60 (BS=1), near-linear batch scaling | ~1% | To verify |
| No compile (baseline) | ~26 | ~1% | Reference |

## Key Design Decisions

1. **topk vs sort for sampling**: `topk` is O(V + k log k) vs sort's O(V log V). For codebook V=4096, k=30, topk is ~5x fewer FLOPs. Both are CUDA-graph-safe; the old sort approach was chosen for "safety" but was unnecessarily conservative.

2. **Eager fn for CUDA graph capture**: torch.compile + CUDA graph is redundant. CUDA graphs already eliminate kernel launch overhead. Using eager avoids Dynamo's fake tensor tracing issues with complex model internals (e.g., `project_in` shape changes).

3. **top_k as int constant**: Since all requests use the same top_k value and `torch.topk` requires a Python int, baking it as a constant simplifies the code and avoids `.item()` sync issues in compiled/CUDA-graph paths.
