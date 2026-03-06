# Plan: Fix Missing Deepstack Intermediate-Layer Visual Injection in SGLang Thinker

## 1. Background: How Qwen3-Omni's Thinker Expects Visual Data

Qwen3-Omni feeds visual information into the thinker (language model) through **two complementary channels**. Both must work together for the model to fully understand visual content.

### Channel A — Input Embedding Replacement (Layer 0) — ALREADY WORKING

When a user sends a video, the preprocessor tokenizes the video frames into placeholder tokens (`<video>`, token ID 151656) in the prompt. During prefill, `_inject_multimodal_embeds()` scans the token sequence, finds these placeholders, and replaces them with the actual `video_embeds` produced by the image encoder. The result is an `input_embeds` tensor where visual positions contain real visual features instead of placeholder embeddings.

**This channel is fully implemented and working.** The entire upstream pipeline is correct:
- Image encoder produces `video_embeds` ✅
- Merge stage packs them into `thinker_inputs["model_inputs"]` ✅
- `build_sglang_thinker_request` attaches them to `req.omni_model_inputs` ✅
- `_inject_multimodal_embeds()` replaces placeholders in `input_embeds` ✅
- `_forward_with_omni_embeds()` passes `input_embeds` to `outer.model()` ✅

The model sees visual information at the input layer. This alone allows the model to produce visually-grounded responses (it can identify objects, text, scenes).

### Channel B — Deepstack Intermediate-Layer Injection — BROKEN

In addition to the flat embeddings at layer 0, the image encoder produces **deepstack visual embeddings** — a list of per-layer visual feature tensors. These are designed to be injected as additive residuals at **intermediate transformer layers** (typically layers 0, 1, 2, controlled by `deepstack_visual_indexes` in the model config).

Think of it this way: Channel A gives the model one "look" at the video at the very beginning (layer 0). Channel B gives it additional "looks" at deeper layers, with progressively refined visual features. This multi-layer injection is a core architectural design of Qwen3-Omni for robust visual understanding.

**This channel is broken.** The deepstack data is correctly produced by the upstream pipeline and arrives at `_forward_with_omni_embeds()`, but is silently dropped there — it never reaches the language model. All intermediate-layer visual injection is lost.

### How the Native SGLang Path Handles Both Channels

> **IMPORTANT FOR THE IMPLEMENTING AGENT**: The code snippets below were verified against SGLang v0.5.8 at the time of writing, but they may become outdated. **You MUST clone the exact SGLang version used by this project and verify all referenced APIs and code paths yourself before implementing.** Do not blindly trust the snippets below.
>
> ```bash
> # SGLang Omni uses SGLang 0.5.8. Clone it to verify:
> git clone --depth 1 --branch v0.5.8 https://github.com/sgl-project/sglang.git /tmp/sglang-v0.5.8
> # Then check the actual code at:
> #   python/sglang/srt/managers/mm_utils.py          — general_mm_embed_routine
> #   python/sglang/srt/models/qwen3_vl_moe.py        — Qwen3MoeLLMModel, get_deepstack_embeds
> #   python/sglang/srt/models/qwen3_vl.py             — Qwen3VLForConditionalGeneration.forward
> #   python/sglang/srt/models/qwen3_omni_moe.py       — Qwen3OmniMoeForConditionalGeneration
> ```

When SGLang runs Qwen3-Omni directly (without the sglang_omni pipeline), the `forward()` method of `Qwen3VLForConditionalGeneration` (`python/sglang/srt/models/qwen3_vl.py`, line 888) calls `general_mm_embed_routine()`:

```python
# qwen3_vl.py line 921-929:
hidden_states = general_mm_embed_routine(
    input_ids=input_ids,
    forward_batch=forward_batch,
    language_model=self.model,
    multimodal_model=self,
    positions=positions,
    use_deepstack=self.use_deepstack,   # {Modality.IMAGE: True, Modality.VIDEO: True}
    pp_proxy_tensors=pp_proxy_tensors,
)
```

Inside `general_mm_embed_routine()` (file: `python/sglang/srt/managers/mm_utils.py`, line 1045), both channels are handled:

```python
# mm_utils.py line 1091-1104:
input_embeds, other_info = embed_mm_inputs(...)    # Channel A: embedding replacement
# add for qwen3_vl deepstack
if use_deepstack:
    kwargs["input_deepstack_embeds"] = other_info["input_deepstack_embeds"]  # Channel B

# mm_utils.py line 1130-1135:
hidden_states = language_model(
    input_ids=None,
    forward_batch=forward_batch,
    input_embeds=input_embeds,      # Channel A data
    **kwargs,                        # Contains input_deepstack_embeds for Channel B
)
```

Inside the language model `Qwen3MoeLLMModel` (file: `python/sglang/srt/models/qwen3_vl_moe.py`, line 69), the deepstack data is consumed at each transformer layer:

```python
# qwen3_vl_moe.py line 90-113:
for layer_idx, layer in enumerate(self.layers[self.start_layer : self.end_layer]):
    layer_idx += self.start_layer
    # ...
    deepstack_embeds = self.get_deepstack_embeds(
        layer_idx - 1, input_deepstack_embeds      # Slice per-layer features
    )
    hidden_states, residual = layer(
        positions, hidden_states, forward_batch, residual,
        post_residual_addition=deepstack_embeds,    # Added to residual at this layer
    )
```

And `get_deepstack_embeds` (line 57) slices the concatenated tensor:

```python
# qwen3_vl_moe.py line 57-67:
def get_deepstack_embeds(self, layer_idx, input_deepstack_embeds):
    if input_deepstack_embeds is None or layer_idx not in self.deepstack_embed_to_decoder_layer:
        return None
    sep = self.hidden_size * layer_idx
    return input_deepstack_embeds[:, sep : sep + self.hidden_size]
```

Where `self.deepstack_embed_to_decoder_layer = range(3)` (line 52), meaning layers 0, 1, 2 receive deepstack residuals.

## 2. What's Broken: The Last-Mile Drop

The sglang_omni pipeline bypasses `general_mm_embed_routine` and calls the language model directly via `_forward_with_omni_embeds()`. This method implements Channel A correctly, but **drops Channel B at the last step**.

Here is the broken code in `sglang_omni/engines/omni/runtime/sglang_ar.py` (commit `d609c32`):

```python
def _forward_with_omni_embeds(self, forward_batch, input_embeds,
                               deepstack_visual_embeds=None,    # ← arrives from upstream (CORRECT)
                               visual_pos_masks=None):          # ← arrives from upstream (CORRECT)
    ...
    hidden_states = outer.model(
        input_ids=None,
        positions=positions,
        forward_batch=forward_batch,
        input_embeds=input_embeds,         # ← Channel A: passed correctly ✅
        # input_deepstack_embeds=???       # ← Channel B: MISSING ❌
    )
```

The upstream pipeline is fine — every stage correctly produces and forwards the deepstack data:

| Stage | Deepstack Status |
|-------|-----------------|
| Image encoder → `deepstack_visual_embeds_video` | ✅ Produced correctly |
| Merge → `thinker_inputs["model_inputs"]["deepstack_visual_embeds"]` | ✅ Packed correctly |
| `_inject_multimodal_embeds()` → returns `(input_embeds, ds_embeds, vis_masks)` | ✅ Extracted correctly |
| `_forward_with_omni_embeds(forward_batch, input_embeds, ds_embeds, vis_masks)` | ✅ Received correctly |
| `outer.model(input_ids=None, ..., input_embeds=input_embeds)` | ❌ **ds_embeds dropped here** |

Because `input_deepstack_embeds` is never passed, `get_deepstack_embeds()` always returns `None`, and `post_residual_addition` is never applied at any intermediate layer.

## 3. The Fix: Bridge the Last-Mile Gap

### 3.1 Scope

Only **one method** in **one file** needs to change:
- **File**: `sglang_omni/engines/omni/runtime/sglang_ar.py`
- **Method**: `_forward_with_omni_embeds()`

Do NOT touch any upstream code (encoder, merge, `_inject_multimodal_embeds`, request builder). They are all correct. Do NOT touch `outer.model()` or any SGLang internals. The only change is: convert the deepstack data into the format the language model expects, and pass it through.

### 3.2 Format Conversion

The pipeline and the language model use different formats for deepstack data. You need to bridge them:

**Pipeline format** (what `_forward_with_omni_embeds` receives):
- `deepstack_visual_embeds`: a Python list of `N` tensors, each `[num_visual_tokens, hidden_size]` — one tensor per deepstack layer
- `visual_pos_masks`: a boolean tensor `[seq_len]` — `True` at visual token positions

**SGLang format** (what `outer.model()` expects as `input_deepstack_embeds`):
- A single 2D tensor `[seq_len, hidden_size * N]`
- Layer *i*'s features occupy columns `[hidden_size*i : hidden_size*(i+1)]`
- Non-visual positions (where `visual_pos_masks` is `False`) must be zeros

Conversion steps:
1. Concatenate the list along `dim=-1` → `[num_visual_tokens, hidden_size * N]`
2. Create a zero tensor of shape `[seq_len, hidden_size * N]`
3. Scatter the visual-only data into the full-sequence tensor using `visual_pos_masks`
4. Pass the result as `input_deepstack_embeds` to `outer.model()`

### 3.3 Verification: How SGLang Consumes the Tensor

For reference (verified against SGLang v0.5.8, `python/sglang/srt/models/qwen3_vl_moe.py` line 52-67):

```python
# Qwen3MoeLLMModel:
self.deepstack_embed_to_decoder_layer = range(3)  # layers 0, 1, 2

def get_deepstack_embeds(self, layer_idx, input_deepstack_embeds):
    if input_deepstack_embeds is None or layer_idx not in self.deepstack_embed_to_decoder_layer:
        return None
    sep = self.hidden_size * layer_idx
    return input_deepstack_embeds[:, sep : sep + self.hidden_size]
    # Returns [seq_len, hidden_size] — added to residual at this transformer layer
```

## 4. Code Style Requirements

The fix **must strictly follow** the complete style guidelines defined in `/data/chenyang/.claude/agents/code-style-agent.md`. Use the Code Style Agent to write and self-review the implementation against all priority levels (P0 through P4).

## 5. Validation

The fix must pass the existing integration test:

```bash
python tests/test_video_integration.py
```

This test starts the server, sends a video with text prompts across two conversation rounds, and validates the responses. The model's output may improve in visual detail with deepstack active, but the keyword assertions will still match.

**Do NOT modify the test file.**

## 6. Summary

| What | Status |
|------|--------|
| Channel A (layer 0 embedding replacement) | ✅ Working — no changes needed |
| Upstream deepstack production (encoder → merge → request builder → `_inject_multimodal_embeds`) | ✅ Working — no changes needed |
| Channel B last-mile passthrough (`_forward_with_omni_embeds` → `outer.model()`) | ❌ **Broken — this is what you fix** |
| Language model deepstack consumption (`get_deepstack_embeds` → `post_residual_addition`) | ✅ Working — no changes needed |
