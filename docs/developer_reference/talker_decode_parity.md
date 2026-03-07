# Talker Decode Parity Investigation

Last updated: 2026-03-08

This note tracks the ongoing investigation into speech quality regressions caused by
`talker_ar` decode divergence between SGLang runtime and the Hugging Face reference
implementation.

## Current summary

The current root-cause statement is now:

> The earlier "first cached decode backend is wrong" framing was a false lead for the
> current text-only repro. The stronger signal now is that the parity harness was
> feeding HF a stale step-1 input because it missed the live `trailing_text_hidden`
> updates that arrive between prefill and the first decode step.

What is already clear:

- `prefill` is close to HF.
- `code2wav` is not the primary issue.
- feedback scheduling was previously broken, but that issue has been fixed and is no
  longer the dominant cause of the bad audio.
- the old "cached attention backend is the primary suspect" statement should now be
  treated as provisional history, not the latest conclusion.
- the runtime first decode input is dominated by a live projected trailing-thinker
  chunk, not by the residual-code feedback vector.
- once HF step-1 is rebuilt with the actual live trailing chunk captured from
  runtime, the first decode input and hidden state return to close parity.
- stock HF `generate()` is not a valid gold reference for this streaming path unless
  it is wrapped to update `trailing_text_hidden` live during generation.

## What has been ruled out

### 1. `code2wav` / vocoder

Runtime-generated codec codes were fed into HF `code2wav`, and the resulting waveform
was nearly identical. This rules out vocoder mismatch as the main source of the
audible regression.

### 2. Missing feedback delivery

Earlier runs exposed a `WAITING_FEEDBACK` scheduling bug, but that path has already
been fixed. Feedback now reaches Talker, and the scheduling behavior changed as
expected after the fix.

### 3. Completely broken prefill

Observed parity before decode remains strong:

- prefill logits are close to HF
- `step-0` talker hidden cosine is about `0.999`

## Key experiments and outcomes

### Teacher-forced hidden-state parity

The most important signal is the hidden-state comparison under teacher forcing. Using
the same runtime history and matching feedback inputs still produces the same failure
pattern:

- `step-0` is close
- `step-1` diverges sharply
- later steps remain bad

Representative runs:

| Request ID | Condition | Step-0 cosine | Step-1 cosine | Step-2 cosine |
| --- | --- | ---: | ---: | ---: |
| `validate-1772962477` | default backend | `0.9993647` | `-0.0030638` | `-0.0015170` |
| `validate-1772962132` | `prefill_attention_backend=fa3`, `decode_attention_backend=fa3` | `0.9995192` | `-0.0477765` | `0.0356651` |
| `validate-1772962650` | `SGLANG_OMNI_DISABLE_TALKER_MROPE=1` | `0.9993647` | `-0.0075962` | `-0.0406702` |
| `validate-1772965863` | default backend + layer-0 q/k probe | `0.9993647` | `0.0147628` | `0.0511477` |
| `validate-1772966943` | default backend + layer-0 attention-output probe | `0.9993647` | `-0.0386472` | `0.0052035` |

Interpretation:

- backend changes affect the later token trajectory
- backend changes do not fix the first cached decode step
- request-side Talker `mrope` metadata is not the primary cause for the text-only
  repro

### Request-side Talker `mrope` disable experiment

An env-gated switch was added in
[engine_io.py](/omni/sglang-omni/sglang_omni/models/qwen3_omni/pipeline/engine_io.py#L349):

- `SGLANG_OMNI_DISABLE_TALKER_MROPE=1`

With this enabled:

- request-side `req_mrope_shape` becomes `None`
- decode behavior stays effectively unchanged
- the token path remains the default one
- hidden parity still fails at `step-1`

This lowers the priority of request-side `mrope` setup for the text-only repro.

### First decode position / rotary check

Decode still constructs 3D positions internally even when request-side multimodal
inputs are absent. That behavior comes from
[forward_batch_info.py](/omni/sglang/python/sglang/srt/model_executor/forward_batch_info.py#L671).

However, the first decode layer-0 probe shows that this is not where the text-only
parity failure begins.

For request `validate-1772965863`, SGLang vs HF at layer 0, step 1:

- `layer0_input_ln cosine = 0.9999962`
- `q_after_rope cosine = 0.9999917`
- `k_after_rope cosine = 0.9999871`

Interpretation:

- `input_layernorm` is effectively aligned
- `qk_norm` is effectively aligned
- rotary / positions are effectively aligned for the first decode step

This removes `rotary` and decode position construction from the top suspect list for
the current text-only repro.

### First decode attention-output check

For request `validate-1772966943`, SGLang vs HF at layer 0, step 1:

- `attn_output_before_o_proj cosine = 0.0359844`
- `attn_output_after_o_proj cosine = 0.3169531`

Interpretation:

- the first large mismatch appears inside the cached attention result itself
- the mismatch is present before `o_proj`
- `o_proj` is not the primary source of divergence

This was the strongest early evidence that the issue lived in the cached attention /
KV-cache path rather than in rotary, position handling, or post-attention MLP math.
Later experiments refined this conclusion further.

### Cached attention reproduction from runtime tensors

For request `validate-1772976066`, the runtime dump now includes:

- offline layer-0 `q/k/v` recompute from the runtime first-decode input
- live layer-0 attention output
- runtime KV slices used for the first decode step

The key results are:

- `runtime_vs_hf_input_ln cosine = 0.9999962`
- `runtime_vs_hf_q_after_rope cosine = 0.9999917`
- `runtime_vs_hf_k_after_rope cosine = 0.9999871`
- `runtime_offline_vs_hf_v cosine = 0.9999943`
- `runtime_live_vs_offline_k_after_rope cosine = 1.0`
- `runtime_live_vs_offline_v cosine = 1.0`

Interpretation:

- runtime live `q/k/v` exactly match the runtime offline recompute
- that offline recompute also matches the HF direct projection path extremely well
- so the runtime first-step `q/k/v` math is internally self-consistent

The next check was even more important: reconstruct layer-0 attention directly from
the runtime `q` and dumped runtime `cache_k/cache_v`.

For request `validate-1772976066`:

- `manual_runtime_cache_vs_runtime_attn_output_before_o_proj cosine = 0.9999884`
- `manual_runtime_cache_vs_runtime_attn_output_after_o_proj cosine = 0.9999907`

Interpretation:

- SGLang's first-step attention output is almost perfectly reproduced by a plain
  PyTorch softmax attention over the dumped runtime tensors
- this means the runtime attention backend is internally consistent with the runtime
  tensors it was given
- the issue is not "the backend computed the wrong answer from the same inputs"

### Actual HF live attention-input capture

The previous offline HF comparisons were still missing one important piece: the true
`q/k/v` that HF feeds into its own first decode attention during generation.

For request `validate-1772976066`, the compare script was extended to capture the
actual `query_states`, `key_states`, and `value_states` entering the HF attention
kernel on step 1.

Results:

- `runtime_q_vs_hf_live_q_after_rope cosine = 0.1636574`
- `runtime_cache_k_vs_hf_live_key_states_prefix cosine = 0.9999949`
- `runtime_cache_v_vs_hf_live_value_states_prefix cosine = 0.9999970`
- `runtime_cache_k_vs_hf_live_key_states_last cosine = 0.7532094`
- `runtime_cache_v_vs_hf_live_value_states_last cosine = -0.0112152`

Interpretation:

- the prefix KV cache still matches very well
- the current step's live HF attention inputs do **not** match the SGLang current
  step inputs
- the mismatch is concentrated in the current decode token, not the prefix cache
- therefore the latest evidence points upstream of cached attention math, into the
  construction of the current decode-step attention inputs

This is the strongest current evidence and supersedes the earlier "backend attention
must be wrong" framing.

### Live `trailing_text_hidden` capture on first decode

The next probe targeted the actual runtime construction of `feedback_input_embeds`
inside [sglang_ar.py](/omni/sglang-omni/sglang_omni/engines/omni/runtime/sglang_ar.py).

For request `validate-1772982287`, with
`SGLANG_OMNI_DUMP_TALKER_FEEDBACK_INPUTS=1`, the first decode step showed:

- `generation_steps = 1`
- `decode_batch_idx = 1`
- `trailing_len = 3`
- `thinker_chunks_done = true`
- `used_trailing_value_shape = [1024]`
- `raw_feedback_norm = 1.9763`
- `used_trailing_norm = 147.0019`
- `combined_norm = 146.7603`

Most importantly:

- `combined_feedback_input_embeds vs qk_dump.feedback_input_embeds cosine = 1.0`
- `raw_feedback + used_trailing_value vs combined cosine = 1.0`
- `qk_feedback vs used_trailing_value cosine = 0.99991`
- `qk_feedback vs raw_feedback cosine = -0.10731`

Interpretation:

- the runtime first decode input is **not** `feedback + tts_pad_embed`
- the runtime first decode input is effectively `feedback + live trailing chunk`
- because the trailing chunk norm is two orders of magnitude larger than the raw
  residual-code feedback, the first decode input is almost entirely determined by the
  live trailing chunk
- any parity harness that uses the stale prefill dump with empty
  `trailing_text_hidden` will necessarily report a catastrophic step-1 mismatch even
  if the cached decode math is correct

This also explains why the earlier compare script observed that runtime
`feedback_input_embeds` looked nothing like the code predictor output or
`code_predictor_output + tts_pad_embed`.

### HF step-1 rebuilt with the actual live trailing chunk

The final check was to take:

- HF prefill cache
- HF code predictor sampled with `manual_seed(123)`
- the **actual** `used_trailing_value` captured from runtime step 1

and rebuild the step-1 `inputs_embeds` fed into HF decode.

For request `validate-1772982287`:

- `candidate_input_vs_runtime_input cosine = 0.9999803`
- `hf_candidate_hidden_vs_runtime_hidden_step1 cosine = 0.9980757`

Interpretation:

- once HF uses the actual live runtime trailing chunk, the first decode input almost
  exactly matches runtime
- the first decode hidden state also returns to close parity
- this strongly de-prioritizes cached decode backend math, KV restore, and rotary as
  root causes for the current repro
- the larger earlier mismatch came from comparing against the wrong live decode input

### Streaming semantics vs stock HF `generate()`

The runtime streaming path currently does the following:

- Talker starts after collecting only `min_thinker_chunks = 1`
- later thinker chunks are appended live into `request.data.trailing_text_hidden`
- when thinker finishes, EOS is appended as the last trailing entry

For request `validate-1772982287`, the observed first few decode steps were:

- step 1: uses a large projected trailing chunk
- step 2: uses another large projected trailing chunk
- step 3: uses a small EOS trailing embedding
- step 4+: falls back to `tts_pad_embed`

This matches the intended `build_assistant_part()` shape:

- assistant prefill consumes the first four projected assistant-side positions
- `trailing_text_hidden` is "tokens after first 4 + tts_eos"

However, stock HF `Qwen3OmniMoeTalkerForConditionalGeneration.generate()` does **not**
update `trailing_text_hidden` inside `_update_model_kwargs_for_generation()`. It only
updates:

- `past_key_values`
- `attention_mask`
- `cache_position`
- `hidden_states`
- `generation_step`

Interpretation:

- static HF `generate()` can under-model the runtime streaming semantics
- a correct parity harness for this path must drive HF with a custom loop that
  updates `trailing_text_hidden` live, rather than passing a fixed tensor once at
  prefill time

## Current highest-probability suspects

### 1. Live trailing-thinker state was missing from the parity harness

This is the strongest current explanation for the previously observed step-1 failure.

Specifically:

- prefill dumps can show `trailing_text_hidden` as empty
- the runtime request state is then updated asynchronously by
  [talker_executor.py](/omni/sglang-omni/sglang_omni/models/qwen3_omni/components/talker_executor.py)
  via `_append_trailing_chunk()` and `_mark_thinker_done()`
- by the time the first decode runs, `trailing_text_hidden` may already contain live
  projected thinker chunks plus EOS
- the previous HF comparison did not model this live state transition

### 2. Thinker/Talker synchronization semantics

If there is still a real audio-quality issue after correcting the parity harness, the
next place to look is not cached decode math but synchronization semantics:

- when should Talker start consuming trailing thinker chunks
- whether runtime and HF reference agree on when those chunks become visible
- whether Talker should legally observe `trailing_len = 3` on the first decode step
  for this request

### 3. Residual code predictor parity as a secondary effect

This remains relevant, but is now clearly secondary for step 1:

- `runtime_hidden + manual_seed(123)` reproduces runtime residual codes exactly
- `HF prefill hidden + manual_seed(123)` still diverges on a few residual-code slots
- but the first decode input is dominated by the live trailing chunk, so that residual
  mismatch has only a small effect on the step-1 combined embedding

## Lower-priority or de-prioritized suspects

### Fused set-KV buffer path

This is now lower priority for the current repro.

Reasons:

- request `validate-1772968294` with `SGLANG_OMNI_DISABLE_TALKER_FUSED_SET_KV=1`
  did not materially improve parity
- in request `validate-1772976066`, the runtime dump reports:
  - `used_fused_qk_norm_rope = false`
  - `used_fused_set_kv_buffer = false`
- so the default text-only repro is not currently exercising fused set-KV on the
  first decode step

### Request-side Talker `mrope` setup

Still lower priority for this text-only repro.

### Prompt assembly, code2wav, and fully broken prefill

Still ruled out or substantially de-prioritized:

- prompt assembly errors
- code2wav mismatch
- fully broken prefill math

### Cached attention backend "wrong math from correct tensors"

This framing is now strongly de-prioritized for the current repro.

The runtime attention output is internally self-consistent, and once HF is given the
actual live step-1 input, the hidden-state parity mostly returns.

## Repo instrumentation added during investigation

### Decode metadata logging

[sglang_ar.py](/omni/sglang-omni/sglang_omni/engines/omni/runtime/sglang_ar.py#L1004)
logs the first decode step with:

- `input_ids`
- `positions`
- `mrope_shape`
- `mrope_last`
- feedback shape
- `generation_steps`
- `decode_batch_idx`
- `seq_len`

### Layer-0 q/k dump

[sglang_ar.py](/omni/sglang-omni/sglang_omni/engines/omni/runtime/sglang_ar.py#L1080)
adds an env-gated dump of:

- feedback input embeds
- layer-0 input after `input_layernorm`
- layer-0 `q` after rotary
- layer-0 `k` after rotary
- layer-0 `v`

Enable with:

```bash
SGLANG_OMNI_DUMP_TALKER_QK=1
```

Artifacts are written to:

- `/tmp/talker_decode_layer0_qk_<request_id>.pt`

### Layer-0 attention-output dump

[sglang_ar.py](/omni/sglang-omni/sglang_omni/engines/omni/runtime/sglang_ar.py#L1181)
and [thinker.py](/omni/sglang-omni/sglang_omni/models/qwen3_omni/thinker.py#L352)
add an env-gated dump of the actual runtime layer-0 attention output on the first
decode step:

- whether fused qk-norm / fused set-kv was used
- attention output before `o_proj`
- attention output after `o_proj`
- current-step cache location
- current-step `k/v`
- current-step KV buffer slice
- full request KV slice as seen by runtime

Enable with:

```bash
SGLANG_OMNI_DUMP_TALKER_ATTN=1
```

Artifacts are written to:

- `/tmp/talker_decode_layer0_attn_<request_id>.pt`

### Request-side Talker `mrope` disable switch

[engine_io.py](/omni/sglang-omni/sglang_omni/models/qwen3_omni/pipeline/engine_io.py#L349)
supports:

```bash
SGLANG_OMNI_DISABLE_TALKER_MROPE=1
```

This is only for diagnosis and should not be treated as a fix.

### Talker fused set-KV disable switch

[sglang_ar.py](/omni/sglang-omni/sglang_omni/engines/omni/runtime/sglang_ar.py#L1269)
supports:

```bash
SGLANG_OMNI_DISABLE_TALKER_FUSED_SET_KV=1
```

This is diagnostic only. It did not materially improve the current text-only repro.

### First decode feedback-input dump

[sglang_ar.py](/omni/sglang-omni/sglang_omni/engines/omni/runtime/sglang_ar.py)
now supports:

```bash
SGLANG_OMNI_DUMP_TALKER_FEEDBACK_INPUTS=1
```

Artifacts are written to:

- `/tmp/talker_feedback_input_<request_id>_step<generation_step>.pt`

Each dump records:

- raw feedback received from the code predictor
- whether thinker was marked done
- the current trailing length
- the actual `used_trailing_value`, if any
- `tts_pad_embed`
- the final combined first-decode input embedding

## Temporary debug artifacts used in the investigation

Current ad-hoc comparisons rely on artifacts written under `/tmp`:

- `/tmp/talker_prefill_<request_id>.pt`
- `/tmp/code_predictor_debug_<request_id>.pt`
- `/tmp/talker_prefill_logits_<request_id>.pt`
- `/tmp/talker_decode_layer0_qk_<request_id>.pt`
- `/tmp/talker_decode_layer0_attn_<request_id>.pt`
- `/tmp/talker_feedback_input_<request_id>_step<generation_step>.pt`

These are diagnostic dumps, not stable interfaces.

## Recommended next step

The next most valuable experiment is now:

1. update the parity harness so HF step-1 uses the live runtime
   `used_trailing_value` / trailing sequence instead of the stale prefill dump
2. replace stock HF `generate()` with a custom HF decode loop that updates
   `trailing_text_hidden` live each step
3. rerun the hidden-state comparison across several requests after that correction
4. if audible quality is still wrong, focus on Thinker/Talker synchronization timing:
   - when trailing chunks are appended
   - when `thinker_chunks_done` flips
   - whether first decode should already see those chunks in the reference flow

At this point, the investigation has moved away from "cached decode backend parity"
and toward "live trailing-thinker state parity and synchronization semantics".
