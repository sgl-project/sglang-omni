# Talker Decode Parity Investigation

Last updated: 2026-03-09

This note tracks the investigation into speech quality regressions caused by
`talker_ar` decode divergence between SGLang runtime and the Hugging Face reference
implementation.

## Current summary

The investigation has identified multiple concrete Talker-side bugs.

The first resolved bug was:

> Talker's shared-expert branch was consuming the wrong tensor.
>
> `self.experts(...)` uses the fused MoE path with `inplace=True`, so it overwrites
> `hidden_states` with the routed-expert output. In
> [talker.py](/omni/sglang-omni/sglang_omni/models/qwen3_omni/talker.py), the code
> was previously computing `shared_expert` and `shared_expert_gate` **after** the
> routed experts, which meant the shared branch was consuming routed output instead
> of the original MLP input. HF consumes the original hidden state for both routed
> and shared branches.
>
> The fix is to compute the shared branch first, before calling the in-place routed
> experts.

The latest additional finding is:

> Talker routed-expert parity still diverges on the fused MoE path, even when the
> same packed weights and the same exact layer input are used.
>
> On isolated layer-7 exact-runtime-input replay, the native routed-expert path is
> effectively identical to HF, while the fused routed-expert path is not. Forcing
> Talker MLP to use the native routed-expert path restores near-identical isolated
> MLP parity on layers 7-10.
>
> This is a real local parity bug, but it did **not** fully fix end-to-end greedy
> speech behavior on the fixed prompt, so there is still at least one remaining
> speech-side issue after the local MLP fix.

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
- the earlier generalized "layer-1 qk dump" probe was invalid because it still
  rebuilt `q/k/v` from `feedback_input_embeds`; it was only semantically valid for
  layer 0.
- the current reliable later-step probes are:
  - runtime `self_attn_input` captured at the real `self_attn` boundary
  - runtime attention dump tensors (`current_q/current_k/current_v`, cache slices,
    attention outputs)
- HF exact-runtime-input attention capture

What became clear after the Talker fix:

- the long-form speech stack is now aligned to HF at the stage level:
  - Talker hidden replay returns to near-identical parity on the long prompt
  - Code Predictor outputs match HF exactly on checked long-form steps
  - `code2wav` waveform reconstruction from runtime codes is effectively identical
- the apparent long-form text repetition was a validator artifact, not a model-path
  issue
  - the validator concatenated thinker text deltas plus two full-text final chunks
  - the real thinker text for the long prompt is the target sentence once
- deterministic Thinker text generation is now exact to HF on the long prompt
  - runtime and HF produced the same 19 generated token IDs, including EOS
  - runtime text, runtime final event text, decode terminal text, and HF text all
    matched byte-for-byte

At this point, no remaining deterministic HF parity gap has been reproduced on the
text-only long prompt. The next remaining parity question, if needed, is the sampled
Talker generation path rather than Thinker or forward math.

## New speech-side findings

The remaining end-to-end speech gap was narrowed further after the Talker shared-MoE
fix.

### Broken streamed assistant embedding

For runtime request `runtime-greedy-1773017247`, the first streamed Thinker chunk
used to seed the assistant speech path had:

- `token_id = 9707`
- `chunk_tensor norm = 149.75`
- true Thinker embedding row `9707` norm `= 0.8211`
- `chunk_tensor vs true embedding cosine = 0.0182`

So the streamed assistant `chunk.tensor` was not the true Thinker embedding for the
generated text token, even though Talker was projecting it through
`talker.text_projection(...)` as if it were.

This was the direct cause of the bad last prefill row:

- before the fix, the last Talker prefill row norm was `307.8962`
- after reconstructing assistant text embeddings from `token_id`, the same row
  dropped to `2.1392`, matching HF scale

The runtime change is in
[talker_executor.py](/omni/sglang-omni/sglang_omni/models/qwen3_omni/components/talker_executor.py):

- assistant prefill now reconstructs text embeddings from `token_id`
- trailing assistant chunk projection now also reconstructs embeddings from
  `token_id`
- the prompt-side captured hidden path is unchanged

### End-to-end impact

Before the assistant-embedding fix, greedy runtime speech on the long prompt
(`runtime-greedy-1773016133`) behaved like this:

- Talker ran to the `256` token cap
- no natural codec EOS
- output audio duration `20.4569s`

After the fix, greedy runtime speech (`runtime-greedy-1773017499`) improved to:

- natural codec EOS at Talker step `94`
- output audio duration `7.4969s`

HF full greedy speech on the same prompt still finishes shorter:

- HF Talker length `78`
- HF audio duration `6.1369s`

So the assistant-embedding bug was real and materially improved the runtime output,
but it did not fully close greedy speech parity.

### What still differs now

With the assistant-embedding fix in place:

- runtime and HF layer-0 Talker codec tokens now match for the first two steps
  - HF: `[1049, 196, ...]`
  - runtime: `[1049, 196, ...]`
- the first mismatch is at step `2`
  - HF token `1484`
  - runtime token `207`

The timing experiments also ruled out one remaining hypothesis:

- forcing Talker to wait until Thinker was fully done before starting prefill
  (`SGLANG_OMNI_TALKER_WAIT_FOR_THINKER_DONE=1`) produced the same `94`-token,
  `7.4969s` runtime result
- so the remaining gap is not caused by starting Talker "too early"

The strongest current evidence points at the sampled Code Predictor path:

- full 16-code row at step `0` matches HF exactly
- full 16-code row at step `1` already diverges
- that divergence happens exactly before the later Talker layer-0 sequence starts
  drifting

Official HF predictor capture on the same prompt made the next split explicit:

- step `0` predictor input hidden cosine vs runtime: `0.9999864`
- step `0` predictor sampled codes: exact match
- step `1` predictor input hidden cosine vs runtime: `0.9998630`
- step `1` predictor sampled codes: diverged despite the near-identical input

That originally looked like a sampling-state / RNG mismatch, but the later
replay split was more precise:

- runtime step-1 Talker decode input vs official HF `inputs_embeds`: exact
  (`cosine = 1.0`, `max_abs = 0.0`)
- runtime prompt-prefill `input_embeds` vs official HF prompt-prefill
  `input_embeds`: exact
- official HF-used decode positions vs runtime decode positions: exact
- Talker step-1 per-layer tensors are still very close, but not identical
  through deeper layers:
  - `decoder_effective_inputs`: worst cosine `0.9995465` at layer `12`
  - `layer_inputs`: worst cosine `0.9992045` at layer `11`
  - `mlp_inputs`: worst cosine `0.9979600` at layer `11`

The decisive residual-code replay was:

- predictor call `0`
  - runtime hidden vs HF hidden cosine: `0.9999864`
  - sampled residual codes: exact match
- predictor call `1`
  - runtime hidden vs HF hidden cosine: `0.9998630`
  - greedy residual decoding on runtime hidden and HF hidden: exact match
  - sampled residual decoding with a fresh `manual_seed(123)` on each call:
    exact match
  - sampled residual decoding with the real sequential call order:
    - runtime hidden reproduces runtime step-1 residual row
    - HF hidden reproduces official HF step-1 residual row

So the remaining mismatch is not best described as "different RNG logic" or
"wrong predictor input assembly". The higher-confidence statement is:

> The remaining speech mismatch is now a sampling-sensitive downstream effect
> of a very small Talker hidden-state drift. Input assembly, positions, and the
> residual-code sampler implementation all match HF closely enough that the
> divergence only appears once the sampled Code Predictor path sees slightly
> different hidden states.

Practical interpretation:

- deterministic parity is already much tighter than the earlier bug
- sampled RVQ code generation is still not identical because Talker hidden
  states are not bitwise-identical yet
- the remaining root-cause search belongs inside Talker forward numerics /
  kernels, not prompt assembly or `code2wav`

### Isolated Talker MLP follow-up

The next isolating replay split the Talker MLP block into its routed and shared
branches on exact captured runtime `step-1` layer inputs.

For isolated single-rank `Qwen3OmniTalker.load_weights(...)` replay:

- shared branch parity on layer `7` is already effectively exact to HF
  - `shared_pre_vs_hf cosine = 0.9999952`
  - `shared_gated_vs_hf cosine = 0.9999948`
- routed branch parity depends on which routed-expert path is used
  - `moe_forward_native(...)` vs HF routed output:
    - `max_abs ≈ 2.96e-05`
    - `mean_abs ≈ 6.31e-06`
  - fused `self.experts(...)` vs HF routed output:
    - `max_abs ≈ 0.1533`
    - `mean_abs ≈ 0.01053`

That resolves the contradiction from the earlier full-MLP replay:

- the local mismatch was not caused by the shared branch
- the local mismatch was not caused by native routed-expert math
- the local mismatch came from the fused routed-expert path actually used inside
  `TalkerSparseMoeBlock.forward(...)`

After forcing Talker to use the native routed-expert path for the routed branch,
isolated exact-runtime-input full MLP parity returned to near-identical on the
problem layers:

- layer `7`: `cosine = 0.9999912`
- layer `8`: `cosine = 0.9999931`
- layer `9`: `cosine = 0.9999880`
- layer `10`: `cosine = 0.9999917`

However, the first patched fixed-prompt greedy runtime rerun
(`runtime-greedy-1773031613`) still produced the old long speech behavior:

- runtime final text still matched the requested sentence exactly
- audio duration remained `20.4569s`
- the run did not recover the shorter HF-like greedy stop behavior

So the fused routed-expert path is a confirmed local Talker parity bug, but it is
not yet the entire end-to-end speech explanation on its own.

### Fixed-prompt greedy speech replay

A cleaner fixed-input replay was then run on the exact sentence:

> "Please speak this exact sentence once, naturally and clearly: Hello there,
> this is a longer speech validation sample generated after the Talker parity
> fix."

Both runtime and official HF produced the same Thinker text:

- `Hello there, this is a longer speech validation sample generated after the Talker parity fix.`

But the speech outputs still differed:

- official HF greedy speech:
  - duration `5.9769s`
  - Talker layer-0 length `78`
- runtime greedy speech:
  - duration `7.4969s`
  - Talker layer-0 length `94`

The fixed-prompt code-level split made the remaining issue much more precise:

- full codec frame `0`: exact match
- full codec frame `1`: already diverged
- Talker layer-0 token sequence still matched through the first two tokens
  - HF layer-0 prefix: `[1049, 196, 1484, ...]`
  - runtime layer-0 prefix: `[1049, 196, 207, ...]`

That means the first wrong thing on the fixed prompt is **not** the layer-0 token
at step `1`; it is the residual-code row attached to frame `1`.

To rule out logits-processor differences, runtime Talker decode logits were dumped
before and after codec-token suppression for the first few greedy steps:

- runtime raw Talker top-k matched HF exactly at step `0`
- runtime raw Talker top-k still matched HF exactly at step `1`
- runtime raw Talker logits diverged at step `2`

This is important because it rules out the hypothesis that the early greedy split
comes from codec suppression or repetition-penalty handling:

- the first visible greedy divergence in raw Talker logits happens **after**
  frame-`1` residual feedback has already diverged
- so the split is upstream of runtime codec suppression

The fixed-prompt predictor replay then closed the loop completely:

- predictor call `0`
  - runtime hidden cosine vs HF: `0.9999864`
  - sampled residual codes: exact match
- predictor call `1`
  - runtime hidden cosine vs HF: `0.9998630`
  - sampled residual codes: diverged
- direct HF replay with the standalone code predictor showed:
  - HF hidden at call `1` reproduces the official HF residual row
  - runtime hidden at call `1` reproduces the runtime residual row
  - both use the same seed (`123`) and the same call-`0` prefix

So the strongest current statement is now:

> The remaining speech mismatch is a predictor-sampling-sensitive consequence of
> a very small Talker hidden-state drift at predictor call `1`.
>
> It is not a codec-suppression bug, not a repetition-penalty mismatch, and not
> a separate code-predictor sampling implementation bug.

Put differently:

- prompt assembly is no longer the issue
- Talker layer-0 greedy logits are still aligned for the first two steps
- the first mismatch appears in the residual-code row for frame `1`
- that wrong residual feedback then causes raw Talker logits to split at step `2`

Current blocker for the next bisect step:

- the cleanest remaining experiment is a prompt-prefill Talker layer dump, but
  repeated attempts to rerun the full pipeline with layer probes were blocked by
  Thinker startup OOM on GPU `0` before the request reached Talker

### Prompt-prefill last-token replay

The OOM blocker only affected full-pipeline reruns. A Talker-only replay using
the saved projected prefill tensors bypassed Thinker completely and let us
capture the prompt-prefill last token directly from the local Talker model.

Artifacts:

- runtime direct prefill replay:
  `/tmp/talker_prefill_runtime_lasttoken_layers.pt`
- HF prefill replay on the same saved `input_embeds`:
  `/tmp/qwen3_hf_talker_prefill_layers.pt`

Those per-layer last-token comparisons are all very tight:

- `decoder_effective_inputs`: worst cosine `0.9998648` at layer `18`
- `layer_inputs`: worst cosine `0.9998476` at layer `18`
- `mlp_inputs`: worst cosine `0.9998192` at layer `09`
- `mlp_outputs`: worst cosine `0.9995355` at layer `09`
- `decoder_output_effective_inputs`: worst cosine `0.9998648` at layer `17`

This is the strongest current evidence that prompt prefill is no longer hiding
another large semantic bug. The remaining mismatch really is in the "tiny
numerical drift" category: local Talker and HF are extremely close through
prefill, but not bitwise-identical, and the later sampled residual-code path is
sensitive enough for that drift to matter.

### Layer-7 to layer-8 handoff on the fixed speech prompt

The next step-by-step replay tightened the remaining speech-side gap further on
the fixed prompt.

Using the exact runtime `step-1` projected Talker input and a same-run runtime
layer dump (`runtime-greedy-1773032338`):

- decoder effective input parity stays very tight through layer `7`
  - layer `6`: `cosine = 0.9999729`
  - layer `7`: `cosine = 0.9999700`
- the first clear decoder-handoff drop appears at layer `8`
  - layer `8` decoder effective input: `cosine = 0.9992896`
- the layer-8 self-attention input then amplifies that small drift
  - layer `8` `self_attn_input`: `cosine = 0.9957954`

That looked like a possible layer-8 RMSNorm issue, but the direct replay ruled
that out:

- applying HF `layer[8].input_layernorm` to the saved runtime
  `decoder_effective_inputs[8]` reproduces the saved runtime
  `self_attn_input[8]` almost exactly
  - `cosine = 0.9999938`

So layer-8 input layernorm is behaving correctly on the runtime tensor it
receives. The real drift is already present in the effective decoder input
before that layernorm runs.

The live layer-7 MLP itself was then re-checked with a dedicated runtime MLP
dump (`runtime-greedy-1773033415`), which captures the actual routed/shared
branches used in the engine:

- router logits vs HF: exact within dump precision
  - `cosine = 1.0`, `max_abs = 0.0`
- routed branch vs HF: near-identical
  - `cosine = 0.9999897`
- shared branch vs HF: near-identical
  - `cosine = 0.9999935`
- final MLP output vs HF: near-identical
  - `cosine = 0.9999915`

An isolated single-rank `Qwen3OmniTalker.load_weights(...)` replay on the same
saved layer-7 MLP input also matches HF:

- layer `7` full MLP output vs HF:
  - `cosine = 0.9999914`

That means the remaining fixed-prompt speech drift is **not** inside layer-7
MoE routing, the shared branch, or the MLP combine itself.

The strongest current statement is now:

> The remaining live speech-side drift starts in the decoder handoff from
> layer `7` into layer `8`, specifically in the post-MLP residual/effective-input
> path before layer-8 attention.

This is a narrower and more reliable target than the earlier "late-layer Talker
numerics" summary.

## Fix status

The runtime fix is in
[talker.py](/omni/sglang-omni/sglang_omni/models/qwen3_omni/talker.py).

Before the fix, request `validate-1772996403` showed:

- dumped `mlp_input` was exactly equal to `routed_output`
- `shared_gate(post_experts_hidden)` matched the runtime shared gate with cosine
  `1.0`
- `shared_gate(pre_gate_hidden)` had cosine `-1.0`

After the fix, request `validate-1772996644` showed the inverse:

- dumped `mlp_input` is now the real pre-gate MLP input
- `shared_gate(pre_gate_hidden)` matches the runtime shared gate with cosine `1.0`
- `shared_gate(post_experts_hidden)` has cosine `-1.0`

Most importantly, request `validate-1772996788` restored exact-runtime-feedback
parity. Under the `runtime_feedback_reference` replay:

- `step-1 hidden cosine = 0.9998964`
- `step-2 hidden cosine = 0.9999722`
- `step-3 hidden cosine = 0.9999335`
- `step-4 hidden cosine = 0.9998780`
- `step-5 hidden cosine = 0.9998212`
- `step-6 hidden cosine = 0.9997559`
- `step-7 hidden cosine = 0.9999368`
- `step-8 hidden cosine = 0.9998242`
- `step-9 hidden cosine = 0.9999325`

The cache-free full-sequence replay also returned to near-identical parity:

- `step-1 hidden cosine = 0.9999221`
- `step-2 hidden cosine = 0.9999806`
- `step-3 hidden cosine = 0.9999543`
- `step-4 hidden cosine = 0.9999602`

So the remaining "later-step Talker parity gap" from this repro is no longer
reproducible once the shared branch uses the correct input tensor.

## Post-fix end-to-end parity status

### Long-form speech stack parity

For long-form request `validate-long-1772998073`, exact-runtime-feedback replay
restored Talker hidden parity on the real streaming path:

- `step-0 hidden cosine = 0.9999886`
- `step-1 hidden cosine = 0.9997877`
- `step-2 hidden cosine = 0.9999772`
- `step-3 hidden cosine = 0.9996383`
- `step-4 hidden cosine = 0.9997507`

The cache-free full-sequence replay on the same long-form dump also stayed near
identical through the checked steps.

For long-form request `validate-long-1772997600`, Code Predictor was replayed
offline through the same HF wrapper used by the runtime executor:

- first checked steps: `all_codes_match = true`
- feedback embedding cosines were effectively `1.0`

For the same request, runtime `code2wav` codes were decoded again through HF
`Qwen3OmniMoeCode2Wav.chunked_decode(...)` and compared against the runtime WAV:

- waveform cosine: `1.0000986`
- `max_abs = 3.0517578125e-05`
- `rmse = 2.5130576e-05`

Interpretation:

- Talker forward parity is restored on the long prompt
- Code Predictor is matching HF
- runtime `code2wav` output is effectively identical to HF on the same codes

### Validator text artifact

The long validation helper originally printed:

- the concatenated thinker text deltas
- the thinker final full-text chunk
- the decode terminal full-text chunk

That made the final text appear repeated 3 times even when the actual thinker output
was only a single sentence. This was a validator aggregation bug, not a model or
runtime parity issue.

### Deterministic Thinker parity on the long prompt

Using the text-only pipeline and raw coordinator stream capture for request
`thinker-compare-1772999003`, the runtime Thinker produced these generated token IDs:

- `[9707, 1052, 11, 419, 374, 264, 5021, 8806, 10519, 6077, 7907, 1283, 279, 18976, 261, 49615, 5046, 13, 151645]`

The runtime decoded text was:

- `Hello there, this is a longer speech validation sample generated after the Talker parity fix.`

HF deterministic generation on the same prompt produced:

- the exact same 19 generated token IDs
- the exact same decoded text

And all runtime text surfaces matched HF:

- `runtime_delta_text == hf_generated_text`
- `runtime_final_text_from_event == hf_generated_text`
- `runtime_decode_complete_text == hf_generated_text`

Interpretation:

- Thinker deterministic generation is now exact to HF on the long prompt
- the remaining "human voice but semantically bad" complaint is no longer explained
  by a known deterministic Thinker parity gap

## Historical investigation notes

The sections below capture the narrowing process that led to the final fix.

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

### Multi-step teacher-forced decode with runtime live inputs

To test whether the remaining problem was still in Talker decode math, HF was driven
with the **actual runtime combined decode input** from
`/tmp/talker_feedback_input_<request_id>_step*.pt` for multiple steps.

For request `validate-1772982287`, using the runtime's real per-step
`combined_feedback_input_embeds`:

- step 1 hidden cosine = `0.9981078`
- step 2 hidden cosine = `0.9981378`
- step 3 hidden cosine = `0.6245142`
- step 4 hidden cosine = `0.4443898`

Interpretation:

- once the runtime live input is used, the first two decode steps return to close
  parity
- this is strong additional evidence that the original "first cached decode kernel is
  wrong" theory was mislocalized
- the remaining later-step drift is now much more likely to come from higher-level
  state evolution, such as live trailing sequence availability, generation-state
  synchronization, or reference-loop modeling gaps
- in other words, the investigation has moved from "decode math parity" to
  "streaming state parity"

Using the reconstructed live trailing sequence directly inside the HF parity harness
gives another strong step-1 / step-2 result:

- step-1 prepared input cosine = `0.9999804`
- step-1 hidden cosine = `0.9962941`
- step-2 hidden cosine = `0.9978197`

This is now the most representative reference for the current repro.

### HF full-sequence no-cache replay with exact runtime decode inputs

To check whether the remaining drift was still a cached decode artifact, the same
runtime `combined_feedback_input_embeds` sequence was replayed in HF as a
**full-sequence, no-cache** forward.

For request `validate-1772985883`:

- step 1 hidden cosine = `0.9981078`
- step 2 hidden cosine = `0.9981378`
- step 3 hidden cosine = `0.6245888`
- step 4 hidden cosine = `0.4395470`

Interpretation:

- the later drift survives even when HF does not use decode cache at all
- this further de-prioritizes "cached decode backend / KV restore" as the primary
  remaining issue
- the residual mismatch is now more consistent with one of:
  - later-step forward parity under projected talker inputs
  - position / rope semantics beyond the first two decode steps
  - a mismatch in what runtime hidden capture represents versus the HF reference

### Layer-0 step-3 probe

The layer-0 probe was then extended from only `step-1` to also capture `step-3`,
which is the first clearly drifting step under the exact-runtime-input replay.

For request `validate-1772987334`, step 3 shows:

- `layer0_input_ln cosine = 0.9999959`
- `q_after_rope cosine = 0.9999869`
- `k_after_rope cosine = 0.9999915`
- `v cosine = 0.9999969`

So the **current token's** layer-0 projection path is still extremely close.

However, the first layer attention output is already off:

- `attn_output_before_o_proj cosine = 0.2023579`
- `attn_output_after_o_proj cosine = 0.1848289`

And the live cache contents are no longer closely aligned by this point:

- `cache_k_prefix cosine = 0.9821874`
- `cache_v_prefix cosine = 0.8653094`
- `cache_k_last cosine = 0.9603992`
- `cache_v_last cosine = 0.1265397`

Interpretation:

- the step-3 mismatch still does **not** start in the current token's
  `input_ln/q/k/v/rope` path
- it shows up in the attention result, consistent with earlier positions already
  having drifted enough to perturb the layer-0 KV state
- combined with the exact-runtime-input **full-sequence no-cache** replay, this
  argues that the root cause is not "decode cache backend only"
- the more likely remaining issue is a later-step forward-parity problem that
  accumulates into the layer-0 cache/value state by step 3

That interpretation was then tightened one step further by replaying the **exact**
runtime per-step feedback inputs back into HF and comparing runtime layer-0 cache /
attention directly against that exact replay.

For request `validate-1772987334`:

- step 2 exact-feedback layer-0 cache:
  - `cache_k_all cosine = 0.9999946`
  - `cache_v_all cosine = 0.9999970`
- step 2 exact-feedback layer-0 attention:
  - `attn_output_before_o_proj cosine = 0.9999886`
  - `attn_output_after_o_proj cosine = 0.9999896`
- step 3 exact-feedback layer-0 cache:
  - `cache_k_all cosine = 0.9999942`
  - `cache_v_all cosine = 0.9999970`
- step 3 exact-feedback layer-0 attention:
  - `attn_output_before_o_proj cosine = 0.9999534`
  - `attn_output_after_o_proj cosine = 0.9999736`

At the same time, the final hidden-state parity under exact runtime inputs is still:

- step 1 hidden cosine = `0.9981078`
- step 2 hidden cosine = `0.9981378`
- step 3 hidden cosine = `0.6245142`

Interpretation:

- under an apples-to-apples exact-runtime-input replay, layer 0 is **not** where the
  real drift starts
- the apparent layer-0 attention mismatch seen earlier for later steps was caused by
  comparing runtime against a non-equivalent HF decode loop
- the remaining real divergence is now localized to **after layer 0**, i.e. in
  layer 1+ / later residual accumulation rather than in the first attention block

That statement was tightened one more step by replaying the remainder of the
runtime layer-0 block directly in HF.

For request `validate-1772990200`, step 3:

- take the runtime step-3 `feedback_input_embeds`
- add the runtime step-3 layer-0 `attn_output_after_o_proj`
- run HF layer-0 `post_attention_layernorm`
- run HF layer-0 MLP
- add the post-MLP residual

The reconstructed `layer-1` input matches the HF exact-runtime-input replay almost
perfectly:

- `layer1_input cosine = 0.9999933`
- `layer1_input max_abs = 0.00390625`

Interpretation:

- the remaining unverified part of layer 0 is now also effectively cleared
- the first real gap is downstream of the layer-0 residual/MLP boundary
- at this point, the investigation target moves from "after layer 0 somewhere" to
  "inside layer 1 attention or later"

The next parallel probe then captured runtime layer-1 attention directly.

For request `validate-1772992139`, step 3:

- runtime layer-1 attention dump was captured with
  `SGLANG_OMNI_DUMP_TALKER_ATTN_LAYER=1`
- HF exact-runtime-input layer-1 attention was captured via a separate one-off
  replay script

Comparison:

- `layer1 attn_output_before_o_proj cosine = 0.8520573`
- `layer1 attn_output_after_o_proj cosine = 0.8958437`

Interpretation:

- layer-1 attention is the first place where a material later-step mismatch has now
  been observed directly
- combined with the `layer0 -> layer1_input` replay above, the remaining primary
  suspect is no longer "generic layer-1+" but specifically **layer-1 attention /
  its live runtime inputs or KV state**

### Layer-1 step-3 `self_attn` boundary

The earlier "layer-1 qk dump" results turned out to be contaminated by a probe bug:
the dump path had only been generalized by layer index, but it still rebuilt
`q/k/v` from `feedback_input_embeds`, which is only correct for layer 0.

The reliable replacement probe now captures:

- runtime `self_attn_input` via a `self_attn` forward-pre-hook
- runtime `current_q_after_rope/current_k_after_rope/current_v` directly from the
  live attention forward
- HF exact-runtime-input `self_attn_input` and attention tensors

For request `validate-1772993744`, step 3:

- `runtime self_attn_input vs HF self_attn_input cosine = 0.8328621`
- `runtime current_q_after_rope vs HF q_after_rope cosine = 0.9523513`
- `runtime current_k_after_rope vs HF key_states[last] cosine = 0.9973885`
- `runtime current_v vs HF value_states[last] cosine = 0.7813702`
- `runtime attn_output_before_o_proj vs HF cosine = 0.8520573`
- `runtime attn_output_after_o_proj vs HF cosine = 0.8958437`
- `runtime cache_k vs HF key_states cosine = 0.9961338`
- `runtime cache_v vs HF value_states cosine = 0.9348503`

Interpretation:

- the first reliable live mismatch is **before** attention aggregation, at the
  layer-1 `self_attn` input boundary
- `k` and cache-`k` remain very close
- `v` and cache-`v` drift more than `k`
- attention output drift follows from that boundary mismatch, rather than being the
  first place where the problem appears

### HF replay from runtime `self_attn_input`

To separate "bad input" from "bad layer-1 qkv math", HF layer-1 `qkv/rope` was run
offline using the **runtime-captured** `self_attn_input` and the **runtime-captured**
 positions, then compared against the runtime live attention tensors.

For request `validate-1772993744`:

- step 1:
  - `q cosine = 0.9999816`
  - `k cosine = 0.9999994`
  - `v cosine = 0.9999999`
- step 2:
  - `q cosine = 0.9999796`
  - `k cosine = 0.9999994`
  - `v cosine = 1.0`
- step 3:
  - `q cosine = 0.9999738`
  - `k cosine = 0.9999992`
  - `v cosine = 1.0`

Interpretation:

- layer-1 `qkv/rope` math itself is effectively aligned with HF, even at step 3
- the remaining real gap is therefore **upstream of qkv**, at the formation of
  `self_attn_input`
- the most likely remaining target is now the output of
  `LayerCommunicator.prepare_attn(...)`, i.e. `input_layernorm / residual /
  communication` semantics for later decode steps

### Layer-1 `prepare_attn` narrowing

The layer-input probe was then extended to capture:

- decoder-layer `hidden_states`
- decoder-layer `residual`
- decoder-layer `effective_input = hidden_states + residual`
- final `self_attn_input`

For request `validate-1772994156`, step 3:

- `decoder_effective_input vs HF decoder_layer_input cosine = 0.9090642`
- `self_attn_input vs HF self_attn_input cosine = 0.8328621`

This shows that the mismatch is already present before `self_attn`, and that it is
further amplified by the move from decoder-layer effective input to normalized
attention input.

The final check then fed the **runtime-captured decoder effective input** into the HF
layer-1 `input_layernorm` directly and compared the result against the **runtime**
`self_attn_input`.

For request `validate-1772994156`:

- step 1: `cosine = 0.9999952`
- step 2: `cosine = 0.9999954`
- step 3: `cosine = 0.9999958`

Interpretation:

- layer-1 `input_layernorm` itself is effectively aligned with HF
- the remaining real gap is **not** in `input_layernorm`
- the remaining real gap is **not** primarily in post-layernorm communication either
- the current smallest live suspect is now the formation of the decoder-layer
  effective input itself, i.e. the later-step `hidden_states + residual` semantics
  before `prepare_attn`

### Talker hidden-capture side channel is not trustworthy for later-step parity

An attempted follow-up probe added talker-side multi-layer hidden capture and dumped
`output.extra["hidden_states"]` on step 3. That path is **not** a valid parity signal
for the current investigation.

For request `validate-1772989746`:

- the exact-runtime-input replay still shows:
  - step 1 hidden cosine = `0.9981078`
  - step 2 hidden cosine = `0.9981378`
  - step 3 hidden cosine = `0.6245142`
- the independently verified layer-0 attention/cache probe is still nearly exact:
  - `attn_output_after_o_proj cosine = 0.9999736`
  - `cache_k_all cosine = 0.9999942`
  - `cache_v_all cosine = 0.9999970`
- but the talker hidden-capture dump for the same step reports:
  - captured `embed` vs HF layer-0 hidden cosine = `-0.2013559`
  - captured `layer-1` input vs HF layer-1 hidden cosine = `-0.1565505`

This is internally inconsistent, because the runtime step-3 `combined_feedback_input`
and layer-0 `q/k/v/attn` are already known to align with HF under the exact replay.

Additional spot checks confirmed that the captured talker `embed` tensor itself does
not even match the current runtime decode input on step 3.

Interpretation:

- for talker decode, `output.extra["hidden_states"]` is currently not anchored to the
  same semantic location as the exact-runtime-input HF hidden reference
- this side channel should **not** be used to localize the remaining step-3 parity gap
- any higher-layer probe should instead use live per-layer runtime tensors gathered
  directly at the layer boundary, not the current hidden-capture side channel

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

The new trailing-event timeline for request `validate-1772985883` makes the ordering
explicit:

- `append_trailing_chunk idx=0` with `chunk_id=1`, `token_id=0` (`"!"`)
- `append_trailing_chunk idx=1` with `chunk_id=2`, `token_id=151645`
  (`"<|im_end|>"`)
- `append_tts_eos idx=2`
- only after that does Talker finish prefill / emit `step-0` token `167`
- then first decode starts with `trailing_len = 3`

Interpretation:

- the runtime is not merely consuming stale prefill state
- Talker is observing **future thinker chunks before the first decode step**
- this is now the central semantic question: whether that lookahead is intentional
  streaming behavior or an overly eager synchronization policy

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
- for requests where thinker is already done by the first decode step, passing the
  fully reconstructed trailing sequence up front may also be sufficient for parity
  checks

One additional caveat from request `validate-1772982287`:

- even after reconstructing the full trailing sequence (`trailing_len = 3`), stock HF
  `generate()` still does not match the runtime sampled token path

Interpretation:

- exact non-teacher-forced token parity is still confounded by generation-loop
  differences such as sampling/RNG state and model-kwargs evolution
- this reinforces that stock HF `generate()` is not the right final gold reference
  for the streaming Talker path

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
SGLANG_OMNI_DUMP_TALKER_QK_MAX_STEP=3
```

Artifacts are written to:

- `/tmp/talker_decode_layer0_qk_<request_id>_step<generation_step>.pt`

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
SGLANG_OMNI_DUMP_TALKER_ATTN_MAX_STEP=3
```

Artifacts are written to:

- `/tmp/talker_decode_layer0_attn_<request_id>_step<generation_step>.pt`

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

1. treat the exact-runtime-input HF full-sequence replay as the control reference,
   since it removes code predictor sampling and decode-cache effects from the test
2. focus on the boundary that forms the layer-1 decoder effective input, not on
   cached decode kernels, `qkv/rope`, or `input_layernorm`
3. compare runtime vs HF on `step-3+` for:
   - decoder-layer `hidden_states`
   - decoder-layer `residual`
   - decoder-layer `effective_input = hidden_states + residual`
4. determine whether the first mismatch enters through the residual stream itself,
   through the pre-residual hidden stream, or through their combination semantics
5. do **not** rely on the old generalized layer-1 qk dump or the hidden-capture side
   channel for this, since neither is aligned with the live later-step runtime
   boundary

At this point, the investigation has moved away from "first cached decode backend
parity", away from "layer-1 attention math", and away from `input_layernorm` itself.
The current target is now **later-step layer-1 residual / effective-input semantics
before `prepare_attn`**.
