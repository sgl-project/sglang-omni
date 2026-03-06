# Debate: Is Channel B (Deepstack Intermediate-Layer Injection) Actually Broken?

## Conclusion: Channel B is already implemented and working.

`new_plan.md` claims Channel B is broken. After systematic runtime verification, I found that Channel B is fully functional. The fix was already committed in `ec215b2` and the deepstack data flows end-to-end through the pipeline.

---

## 1. What `new_plan.md` Claims

`new_plan.md` (committed at `693ca09`) states:

> **Channel B — Deepstack Intermediate-Layer Injection — BROKEN**
>
> The deepstack data is correctly produced by the upstream pipeline and arrives at `_forward_with_omni_embeds()`, but is silently dropped there — it never reaches the language model.

It describes the broken code as:

```python
hidden_states = outer.model(
    input_ids=None,
    positions=positions,
    forward_batch=forward_batch,
    input_embeds=input_embeds,
    # input_deepstack_embeds=???   # ← Channel B: MISSING
)
```

And prescribes a fix: convert the per-layer deepstack list into a concatenated 2D tensor `[seq_len, hidden_size * N]` and pass it as `input_deepstack_embeds` to `outer.model()`.

---

## 2. What I Found in the Code

### 2.1 The fix already exists at HEAD

When I read `sglang_omni/engines/omni/runtime/sglang_ar.py` at HEAD (`693ca09`, branch `img-lost`), the `_forward_with_omni_embeds()` method already contains the exact fix described in the plan (lines 470–512):

```python
ds_input = None
if deepstack_visual_embeds is not None and visual_pos_masks is not None:
    device = input_embeds.device
    dtype = input_embeds.dtype
    layer_tensors = [
        t.to(device=device, dtype=dtype) for t in deepstack_visual_embeds
    ]
    ds_input = torch.cat(layer_tensors, dim=-1)

    full_ds = torch.zeros(
        input_embeds.shape[0],
        ds_input.shape[-1],
        device=device,
        dtype=dtype,
    )
    full_ds[visual_pos_masks] = ds_input
    ds_input = full_ds

hidden_states = outer.model(
    input_ids=None,
    positions=positions,
    forward_batch=forward_batch,
    input_embeds=input_embeds,
    input_deepstack_embeds=ds_input,
)
```

This code does exactly what the plan prescribes:
1. Concatenate the per-layer list along `dim=-1` -> `[num_visual_tokens, hidden_size * N]`
2. Create a zero tensor `[seq_len, hidden_size * N]`
3. Scatter visual-only data using `visual_pos_masks`
4. Pass as `input_deepstack_embeds` to `outer.model()`

### 2.2 When was it added?

```
git log --oneline
693ca09 adds new plan to fix middle layer transfer   <-- new_plan.md updated
3cf177c adds new plan to fix middle layer transfer   <-- new_plan.md added
c251c09 upd reflection
7bab66c enlarge the test
ec215b2 add unit test                                <-- fix added HERE
d609c32 clean up comments                            <-- broken state referenced by plan
```

`git show ec215b2 -- sglang_omni/engines/omni/runtime/sglang_ar.py` confirmed this commit added the 25-line deepstack conversion and passthrough to `_forward_with_omni_embeds`.

The plan (`new_plan.md`) was committed AFTER the fix, in `3cf177c` and `693ca09`.

### 2.3 Why I initially couldn't confirm correctness

Static code analysis showed the fix was present and matched the plan's specification. However, I could not rule out these failure modes from code reading alone:

- **Serialization loss**: The pipeline passes data between stages via ZMQ + pickle + SHM relay (`DataPlaneAdapter` in `sglang_omni/pipeline/worker/data_plane.py`). Torch tensors are extracted via `_extract_tensors()`, transferred via shared memory, and restored via `_restore_tensors()`. If this serialization dropped the deepstack tensors (a list of tensors nested inside `thinker_inputs.model_inputs`), the data would be silently None at runtime.

- **Data never populated upstream**: If `merge_for_thinker` or the image encoder never produced deepstack data for some reason, the downstream code would be dead code.

- **Shape mismatch at runtime**: If `visual_pos_masks.sum()` didn't match the number of deepstack visual tokens, the scatter operation would fail silently or crash.

The `reflection.md` document (committed at `c251c09`) further reinforced doubt with this statement:

> "The deepstack passthrough edit had no measurable effect because the bug was already fixed."
> "Currently, only Channel A works. Channel B is broken."

These contradictory statements (code present but described as broken) meant I could not trust static analysis alone.

---

## 3. Runtime Verification

### 3.1 Method

I added debug print statements that write to `/tmp/deepstack_debug.log` at three checkpoints:

1. **Inside `_inject_multimodal_embeds`** (after reading `omni_inputs`): Log the keys present in `omni_inputs` and the type/shape of deepstack data.

2. **At the top of `_forward_with_omni_embeds`**: Log whether `deepstack_visual_embeds` and `visual_pos_masks` are None or populated.

3. **Just before calling `outer.model()`**: Log the final `ds_input` tensor shape and whether it contains nonzero values.

Then ran the integration test: `python tests/test_video_integration.py`

### 3.2 Results

The debug log (`/tmp/deepstack_debug.log`) for round 1:

```
[INJECT] omni_inputs keys: ['video_embeds', 'deepstack_visual_embeds', 'video_grid_thw', 'video_second_per_grid']
[INJECT] ds_embeds type=<class 'list'>, image_ds type=<class 'NoneType'>, video_ds type=<class 'NoneType'>
[INJECT] ds_embeds len=3, first shape=torch.Size([2400, 2048])
[FORWARD] deepstack_visual_embeds type=<class 'list'>
[FORWARD] visual_pos_masks type=<class 'torch.Tensor'>
[FORWARD] deepstack_visual_embeds len=3
[FORWARD] ds_input type=<class 'torch.Tensor'>, is_none=False
[FORWARD] ds_input shape=torch.Size([2416, 6144]), nonzero=1581056.00
```

Round 2 showed the same pattern with `ds_input shape=torch.Size([2590, 6144])`.

### 3.3 Interpretation

| Checkpoint | Expected if broken | Actual observed | Verdict |
|---|---|---|---|
| `omni_inputs` contains `deepstack_visual_embeds` | Key missing or value is None | Key present, value is list of 3 tensors | Data survives serialization |
| Each tensor shape | N/A | `[2400, 2048]` (2400 visual tokens, 2048 hidden_size) | Matches encoder output |
| `deepstack_visual_embeds` at `_forward_with_omni_embeds` | None | list of 3 tensors | Upstream extraction works |
| `ds_input` passed to `outer.model()` | None | `[2416, 6144]` = `[seq_len, hidden_size * 3]` | Format conversion correct |
| `ds_input` nonzero sum | 0.0 | 1,581,056.0 | Real data, not zeros |

Every checkpoint confirms Channel B is active and passing real deepstack data to the language model.

### 3.4 Verification of SGLang consumption

I also verified that the SGLang language model (`Qwen3MoeLLMModel` in `sglang/srt/models/qwen3_vl_moe.py`) correctly accepts and uses `input_deepstack_embeds`:

- `forward()` accepts `input_deepstack_embeds: Optional[torch.Tensor]` parameter (line 76)
- `self.deepstack_embed_to_decoder_layer = range(3)` (line 52) — layers 0, 1, 2 receive deepstack
- `get_deepstack_embeds(layer_idx - 1, input_deepstack_embeds)` slices the tensor at each layer (line 104)
- The sliced tensor is passed as `post_residual_addition` to each transformer layer (line 112)

The `ds_input` tensor shape `[2416, 6144]` = `[seq_len, 2048 * 3]` matches exactly what `get_deepstack_embeds` expects: it slices `[:, 0:2048]` for layer 0, `[:, 2048:4096]` for layer 1, `[:, 4096:6144]` for layer 2.

---

## 4. Why `new_plan.md` Is Wrong

### 4.1 The plan describes an outdated state

The plan references commit `d609c32` as the broken state. At that commit, `_forward_with_omni_embeds` indeed did NOT pass `input_deepstack_embeds` to `outer.model()`. The plan's diagnosis of the problem is accurate for that commit.

However, the fix was already applied in commit `ec215b2` (by a previous Claude session during investigation). The plan was written AFTER this fix, in commits `3cf177c` and `693ca09`. The plan author may not have been aware that the fix was already committed.

### 4.2 The reflection document is contradictory

`reflection.md` (section 2.3) says both:

> "During this investigation, I wrote this conversion and passthrough."

and:

> "Currently, only Channel A works. Channel B is broken."

The first statement acknowledges the fix was written. The second says it's still broken. This contradiction likely arose because the reflection was drafted before the fix was committed, then partially updated afterward. The "Channel B is broken" statement was never corrected.

### 4.3 The integration test cannot distinguish Channel A from Channel B

The test (`tests/test_video_integration.py`) checks for keywords like "airport", "train", "gate", "12", "uci" in the model response. These keywords can be produced by Channel A alone (input embedding replacement at layer 0 gives the model sufficient visual information). There is no test that specifically validates Channel B is active.

This means even if Channel B were broken, the test would still pass — which is probably why the plan's author believed it was still broken despite the test passing.

### 4.4 The "no measurable effect" observation

The reflection notes that adding the deepstack passthrough had "no measurable effect." This is expected: the keyword-based integration test measures whether the model understands the video, not whether deepstack specifically contributes. Channel A alone provides enough visual information for the test's keyword assertions. A proper Channel B validation would require either:
- Comparing model logits/hidden states with and without deepstack
- A perceptual quality test on edge cases where deepstack makes a visible difference
- Runtime instrumentation (which is what I did above)

---

## 5. Full Upstream Data Flow Verified

For completeness, here is the verified data flow:

| Stage | What happens | Status |
|---|---|---|
| **Image encoder** (`Qwen3OmniImageEncoder.forward`) | `self.visual()` returns `(video_embeds, deepstack_feature_lists)`. Output includes `deepstack_visual_embeds_video` key. | Verified in code |
| **Merge** (`build_thinker_inputs`) | `_as_tensor_list(video_out.get("deepstack_visual_embeds_video"))` extracts the list. Stored as `thinker_model_inputs["deepstack_visual_embeds"]`. | Verified in code |
| **Inter-stage transfer** (`DataPlaneAdapter`) | `_extract_tensors` recursively finds tensors in lists/dicts, transfers via SHM, `_restore_tensors` rebuilds. | Verified at runtime (data arrives intact) |
| **Request builder** (`build_sglang_thinker_request`) | `model_inputs = dict(thinker_inputs.get("model_inputs", {}))`, then `req.omni_model_inputs = model_inputs`. | Verified at runtime (`omni_inputs` has the key) |
| **Embed injection** (`_inject_multimodal_embeds`) | Reads `omni_inputs.get("deepstack_visual_embeds")`, builds `visual_pos_masks`, returns `(input_embeds, ds_embeds_out, visual_masks_out)`. | Verified at runtime (list of 3 tensors, each `[2400, 2048]`) |
| **Forward** (`_forward_with_omni_embeds`) | Concatenates list -> `[2400, 6144]`, scatters to `[2416, 6144]`, passes as `input_deepstack_embeds`. | Verified at runtime (`ds_input` shape and nonzero values confirmed) |
| **Language model** (`Qwen3MoeLLMModel.forward`) | `get_deepstack_embeds` slices per-layer features, adds as `post_residual_addition` at layers 0, 1, 2. | Verified in SGLang source code |
