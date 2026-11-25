# Whisper beam search with N-best output (design note)

Issue: sgl-project/sglang-omni#4331. Status: implemented as described below.

## Goal

- `beam_width` per request, on Whisper ASR stage requests and on `/v1/audio/transcriptions`.
- Return all `beam_width` hypotheses, best first, each with its text and its raw cumulative log-prob.
- Optional `beam_search_diversity_rate`, TensorRT-LLM semantics: a ranking-only bonus of `rate * (rank_within_parent % beam_width)`; returned log-probs stay raw.
- Greedy and beam requests share one server and one decode forward.

## What sglang 0.5.23 already gives us

`python/sglang/srt/beam_search/` runs a beam request as one leader `Req` plus `k-1` bare member rows in `req_to_token`. The member rows ride the normal decode batch (`append_beam_tail`), alias the leader's prompt KV, and share decoded KV on fork (reparenting remaps `req_to_token`, no KV copy). Admission, retraction, decode memory checks and the result processor already know about beam groups.

What does not fit Whisper in omni:

1. `validate_and_init` refuses encoder-decoder models, and omni never calls it: the Whisper request builder constructs the `Req` itself, so `handle_generate_request` does not run.
2. Candidates are scored on raw logits. Whisper's `suppress_tokens`, begin-suppress and timestamp rules never reach the beams. This is a correctness bug, not only a gap.
3. `append_beam_tail` / `strip_beam_tail` do not extend `encoder_lens`; decode allocation computes `encoder_lens + seq_lens`, so the tail rows break the shape (the issue author hit this).
4. Omni's runner samples and relays tokens itself (`ModelRunner.sample_next_token_ids`, `SGLangExecutionBridge.publish_next_tokens`), so sglang's pre-sample logits capture (in `tp_worker`) and its selection hook (`Scheduler._relay_forward_payload`) never run.
5. Selection semantics differ from Hugging Face: no early stopping (every group decodes to `max_new_tokens`, which for Whisper also lets repetition loops win on length-normalized score), finished hypotheses are not limited to the top `k` ranks, and there is no diversity term.

## Recommended first version: answers to the issue's four questions

**1. Speculative decoding.** Beam requests are accepted only on servers without speculative decoding. On a spec server (`whisper_large_v3_speculative.yaml`) a request with `beam_width > 1` gets HTTP 400. Spec is a server-level opt-in today, and a per-request bypass would need the spec worker to run part of a batch without drafts, which is a separate project.

**2. Async decode and CUDA graphs.**

- Encoder CUDA graph and dense-prefill CUDA graph: unchanged. Only the leader prefills, exactly like a greedy request; member rows never prefill.
- Draft CUDA graph: not reached (no beam on spec servers).
- Decode CUDA graph: member rows are extra rows of the same decode forward, so the graph bucket is chosen by `reqs + member rows`. Each member row gets its leader's `encoder_lens`, so cross-attention reads the leader's encoder KV.
- Async decode (launch-first lookahead): a batch that holds a live beam group runs synchronously, through the same `lookahead_eligible` gate that timestamped requests already use. Prefill batches with a beam leader also run synchronously. Greedy rows in such a batch lose the one-step overlap while the beam request is live. This needs a B200 measurement; a lookahead-safe version (select at launch, commit at resolve, as sglang's overlap path does) is the follow-up if the cost matters.

**3. Encoder cache.** One entry per request. The leader's prefill writes the cross-attention KV once; member rows alias the leader's `req_to_token` prefix (encoder slots plus decoder prompt), so there is no second encoder run, no second pre-LM cache entry, and no encoder KV copy. The decoded region is owned by the group and freed once at finish (`free_member_rows`).

**4. `segment_timestamps`, `detect_language`, streaming.**

- `segment_timestamps`: supported. Timestamp rules run per beam before selection, from per-beam device state (last token, penultimate token, last timestamp) that is gathered by parent index on every step. The first step, with an empty history, uses the existing host rules, including `max_initial_timestamp_index`.
- `detect_language`: rejected with `beam_width > 1` (it is a one-token decode).
- Streaming: final transcripts only. `stream=true` with `beam_width > 1` gets 400; internally a beam leader emits no partial chunks, and its terminal payload carries the N-best.
- Long audio that `/v1/audio/transcriptions` splits into chunks: rejected with `beam_width > 1` in this version, because per-chunk N-best lists do not compose into an N-best for the file.

## Scoring and selection (Hugging Face `generate` semantics)

Per step, for each group with `k = beam_width` rows (one row at the leader's prefill step):

1. `logprob = log_softmax(raw_logits)`, then mask with `suppress_tokens`, `begin_suppress_tokens` (first generated position only) and the timestamp rules. Masked tokens get `-inf`; the rest are not renormalized. This is the order HF uses (log-softmax, then logits processors).
2. Each row contributes its top `2k` tokens. Ranking key: `cum_logprob[parent] + logprob + rate * (rank_within_parent % k)`. With `rate = 0` this is exactly HF's top-`2k` over `k x vocab`.
3. Survivors: the first `k` non-EOS candidates in ranking order. They carry the raw cumulative log-prob (no bonus).
4. Finished: EOS candidates among the top `k` ranks. Score = `raw_cum / len ** length_penalty`, with `len` counting the EOS and `length_penalty = 1.0` (HF default). The best `k` finished are kept.
5. Stop when `k` hypotheses are finished and the best running beam's `raw_cum / len ** length_penalty` is not above the worst finished score (HF `early_stopping=False`), or at `max_new_tokens`, where the top `k` candidates are finished as `length`.
6. Output: all `k` hypotheses, sorted by score; each has text, `cum_logprob` (raw), `score` (length-normalized), and `finish_reason`.

Selection is a fixed-shape tensor function with no host sync, like sglang's `joint_select`; the group state machine commits on the host after the step's D2H, where early stopping is decided.

Omni's greedy path does not apply `begin_suppress_tokens` today, while HF and openai-whisper do. The beam path applies it for parity; fixing greedy is a separate change.

## API

- Whisper stage request params: `beam_width` (int, default 1) and `beam_search_diversity_rate` (float, default 0.0).
- `/v1/audio/transcriptions` form fields: the same two names.
- Validation (400): `beam_width` in `[1, 16]`; with `beam_width > 1` also reject `temperature > 0`, `stream=true`, `detect_language`, chunked long audio, `/v1/audio/translations`, and speculative servers.
- Response: OpenAI has no N-best field, so it is `sglext.n_best`, a list of `{text, cum_logprob, score, finish_reason}` in score order, present only when `beam_width > 1`. Top-level `text` stays the best hypothesis. `sglext` is the repo's existing home for fields outside the OpenAI schema.
- `json` and `verbose_json` carry `sglext.n_best`. On HTTP, model timestamps come only with `srt` and `vtt`, which are plain text, so those return the best hypothesis only; a stage request with `segment_timestamps` and `beam_width` gets the timestamped N-best.

## Implementation map

- `sglang_omni/models/whisper_asr/beam_search.py` (new): the selection function, the group state machine (HF finish rules), and per-beam logit processing (suppress, begin-suppress, timestamps).
- `OmniBeamCoordinator` (subclass of sglang's `BeamCoordinator`): validation without the encoder-decoder refusal, group creation from the request builder's params, selection on processed log-probs, and encoder-aware `append_beam_tail` / `strip_beam_tail`.
- `OmniScheduler`: install the subclass, validate and attach the group when a built request enters the queue, gate async lookahead and async prefill, suppress partial stream chunks for beam leaders, and copy the group's results into the request data at finish.
- `ModelRunner`: capture and slice the member-row logits before sampling, and run selection plus the relay stash after sampling (the omni equivalents of sglang's two hooks).
- Request builder, result adapter, `speech_to_text` form, and `protocol.py`: the params and the N-best.

## Tests and parity

- CPU unit tests: the selection function against a direct port of HF's beam step on small vocabularies (with EOS ties, diversity, max length), the group's finish rules, per-beam timestamp state, encoder-aware tail append and strip, parameter validation, and the response schema.
- GPU parity on one RTX 4090 D: whisper-large-v3 against HF `generate(num_beams=5, num_return_sequences=5)` with the same processors and `max_new_tokens`, on a few clips, with and without timestamps. Compare token sequences and order of all 5 hypotheses, and cumulative log-probs within a tolerance. Also run greedy and beam requests together on one server.
- Performance numbers for Baseten are a later step on B200.
