# SPDX-License-Identifier: Apache-2.0
"""SGLang MLX runner adapter for Fun-CosyVoice3 speech-token decoding."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from .model import SPEECH_TOKEN_SIZE

_SPEECH_IDS = mx.arange(SPEECH_TOKEN_SIZE, dtype=mx.int32)
# MLX streams are thread-local. Materialize this module-level lookup on the
# construction thread so later scheduler-thread graphs do not retain stream 0.
mx.eval(_SPEECH_IDS)


class FunCosyVoice3MlxModelRunner:
    """Customize only prompt prefill; generic MLX cache/decode stays upstream."""

    def _load_model(self) -> None:
        from sglang.srt.hardware_backend.mlx.remote_code_gate import (
            ensure_remote_code_allowed,
            resolve_model_directory,
        )

        from .model import load_cosyvoice3_mlx_model

        model_dir = resolve_model_directory(self.model_path, revision=self.revision)
        ensure_remote_code_allowed(model_dir, self.trust_remote_code)
        self.model = load_cosyvoice3_mlx_model(
            model_dir,
            quantization=self._quantization,
        )
        # Non-final chunked prefills are disabled for this stage; the wrapper's
        # decode head is intentionally the only forward entry point.
        self._trunk = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._cosyvoice3_prompt_lengths: dict[str, int] = {}
        self._cosyvoice3_min_lengths: dict[str, int] = {}
        self._cosyvoice3_repetition_penalties: dict[str, float] = {}
        self._cosyvoice3_seen_masks: dict[str, mx.array] = {}
        self._cosyvoice3_recent_tokens: dict[str, list[int]] = {}
        self._cosyvoice3_sampling_pending_tokens: mx.array | None = None

    @staticmethod
    def _request_prompt(req: Any) -> tuple[list[int], list[int]]:
        text_ids = getattr(req, "_cosyvoice3_text_token_ids", None)
        prompt_ids = getattr(req, "_cosyvoice3_prompt_speech_token_ids", None)
        if text_ids is None or prompt_ids is None:
            raise ValueError(
                "Fun-CosyVoice3 MLX request is missing raw prompt token metadata"
            )
        return list(text_ids), list(prompt_ids)

    def _constrain_logits(
        self,
        logits: mx.array,
        req_ids: list[str],
        caches: list[list[Any]],
        *,
        initial: bool = False,
        pending_tokens: mx.array | None = None,
    ) -> mx.array:
        """Apply Omni's stop and repetition constraints before sampling."""
        rows = []
        for index, req_id in enumerate(req_ids):
            row = logits[index]
            prompt_length = self._cosyvoice3_prompt_lengths.get(req_id, 0)
            if initial:
                generated_count = 0
            else:
                generated_count = max(
                    self._first_attention_cache(caches[index]).offset - prompt_length,
                    0,
                )
            if generated_count < self._cosyvoice3_min_lengths.get(req_id, 0):
                row = mx.concatenate(
                    [
                        row[:SPEECH_TOKEN_SIZE],
                        mx.full_like(row[SPEECH_TOKEN_SIZE:], -float("inf")),
                    ]
                )

            penalty = self._cosyvoice3_repetition_penalties.get(req_id, 1.0)
            if penalty != 1.0:
                seen = self._cosyvoice3_seen_masks[req_id]
                if pending_tokens is not None:
                    # A chained step is built before the predecessor is
                    # finalized. Carry that predecessor's lazy token into the
                    # graph so repetition state is still exact.
                    seen = seen | (_SPEECH_IDS == pending_tokens[index])
                speech_logits = row[:SPEECH_TOKEN_SIZE]
                adjusted = mx.where(
                    speech_logits > 0,
                    speech_logits / penalty,
                    speech_logits * penalty,
                )
                speech_logits = mx.where(seen, adjusted, speech_logits)
            else:
                seen = self._cosyvoice3_seen_masks[req_id]
                if pending_tokens is not None:
                    seen = seen | (_SPEECH_IDS == pending_tokens[index])
                speech_logits = row[:SPEECH_TOKEN_SIZE]

            row = mx.concatenate([speech_logits, row[SPEECH_TOKEN_SIZE:]])
            rows.append(row)
        return mx.stack(rows)

    def prefill_start(
        self,
        req_id: str,
        new_token_ids: list[int],
        full_token_ids: list[int],
        prefix_slot_ids: list[int],
        new_slot_ids: list[int],
        req_pool_idx: int,
        req: Any | None = None,
        needs_logits: bool = True,
        logit_edit_row: mx.array | None = None,
        logprob_spec: Any = None,
    ):
        from sglang.srt.hardware_backend.mlx.model_runner import MlxPendingPrefill

        del new_token_ids, new_slot_ids
        if req is None:
            raise ValueError(
                "Fun-CosyVoice3 MLX prefill requires its scheduler request"
            )
        if prefix_slot_ids:
            raise NotImplementedError(
                "Fun-CosyVoice3 MLX does not support radix-cache prefixes yet"
            )
        if not self.disable_radix_cache:
            raise RuntimeError("Fun-CosyVoice3 MLX requires disable_radix_cache=True")

        if self._enable_sampling:
            self._req_sampling[req_id] = self._sampling_params_for_request(req)
        self._cosyvoice3_prompt_lengths[req_id] = len(full_token_ids)
        self._cosyvoice3_min_lengths[req_id] = int(req.sampling_params.min_new_tokens)
        self._cosyvoice3_repetition_penalties[req_id] = float(
            req.sampling_params.repetition_penalty
        )
        self._cosyvoice3_seen_masks[req_id] = mx.zeros(
            (SPEECH_TOKEN_SIZE,), dtype=mx.bool_
        )
        self._cosyvoice3_recent_tokens[req_id] = []
        text_ids, prompt_ids = self._request_prompt(req)
        embeddings = self.model.build_prompt_embeddings(text_ids, prompt_ids)
        cache = self._acquire_cache()
        logits = self.model.forward_embeddings(embeddings, cache=cache)
        logits = self._constrain_logits(
            logits[:, -1, :],
            [req_id],
            [cache],
            initial=True,
        )
        lazy_token, lazy_logprobs = self._select_tokens_with_logprobs(
            logits,
            [req_id],
            [cache],
            logit_edit_row,
            logprob_spec,
        )
        del needs_logits
        return MlxPendingPrefill(
            lazy_token=lazy_token,
            cache=cache,
            req_id=req_id,
            full_token_ids=list(full_token_ids),
            req_pool_idx=req_pool_idx,
            synced_offset=0,
            lazy_logprobs=lazy_logprobs,
        )

    def _sampling_params_for_request(self, req: Any) -> Any:
        from sglang.srt.hardware_backend.mlx.sampling import (
            DEFAULT_SAMPLING_SEED,
            MlxSamplingParams,
        )

        sampling_params = req.sampling_params
        # Omni's public request seed is honored even when SGLang's global
        # deterministic-inference mode is off. The latter only supplies the
        # default seed for an otherwise unseeded request.
        seed = sampling_params.sampling_seed
        if seed is None and self._deterministic_seeding:
            seed = DEFAULT_SAMPLING_SEED
        # The shared MLX constructor warns that it ignores repetition
        # penalties. This runner applies that penalty above, so build the same
        # normalized parameter object directly and avoid a misleading warning.
        return MlxSamplingParams(
            temperature=sampling_params.temperature,
            top_k=sampling_params.top_k,
            top_p=sampling_params.top_p,
            min_p=sampling_params.min_p,
            seed=seed,
        )

    def decode_batch_start(
        self,
        req_ids: list[str],
        edit_rows: mx.array | None = None,
        logprob_spec: Any = None,
        logits_hook: Any = None,
    ):
        if len(req_ids) != 1:
            return super().decode_batch_start(
                req_ids,
                edit_rows=edit_rows,
                logprob_spec=logprob_spec,
                logits_hook=logits_hook,
            )
        from sglang.srt.hardware_backend.mlx.model_runner import MlxPendingDecode

        req_id = req_ids[0]
        cache = self._req_caches[req_id]
        input_ids = mx.array([[self._req_token_ids[req_id][-1]]], dtype=mx.int32)
        logits = self._decode_with_native_cache([cache], [input_ids])
        logits = self._constrain_logits(logits, req_ids, [cache])
        if logits_hook is not None:
            logits = self._run_logits_hook(logits, logits_hook)
        lazy_tokens, lazy_logprobs = self._select_tokens_with_logprobs(
            logits,
            req_ids,
            [cache],
            edit_rows,
            logprob_spec,
        )
        return MlxPendingDecode(
            lazy_tokens=lazy_tokens,
            req_ids=req_ids,
            caches=[cache],
            lazy_logprobs=lazy_logprobs,
            logprob_spec=logprob_spec,
            edit_rows=edit_rows,
        )

    def _recent_token_masks(
        self,
        req_ids: list[str],
        pending_tokens: mx.array | None,
    ) -> mx.array:
        """Build recent-token masks for CosyVoice's repetition-aware sampler."""
        masks = []
        for index, req_id in enumerate(req_ids):
            mask = mx.zeros((SPEECH_TOKEN_SIZE,), dtype=mx.bool_)
            recent = self._cosyvoice3_recent_tokens.get(req_id, [])
            if recent:
                mask = mask.at[mx.array(recent, dtype=mx.int32)].add(True)
            if pending_tokens is not None:
                mask = mask | (_SPEECH_IDS == pending_tokens[index])
            masks.append(mask)
        return mx.stack(masks)

    def _select_tokens_with_logprobs(
        self,
        last_logits: mx.array,
        req_ids: list[str],
        caches: list[list[Any]],
        edit_rows: mx.array | None = None,
        logprob_spec: Any = None,
    ):
        """Apply CosyVoice RAS fallback around SGLang's MLX sampler.

        The reference sampler first draws from nucleus/top-k and, when that
        candidate appeared in the recent ten-token window, redraws from the
        full distribution with that candidate masked. Keep this entirely in
        the MLX graph so chained decode remains valid.
        """
        if not self._enable_sampling:
            return super()._select_tokens_with_logprobs(
                last_logits,
                req_ids,
                caches,
                edit_rows,
                logprob_spec,
            )

        from sglang.srt.hardware_backend.mlx.sampling import (
            MlxSamplingParams,
            compute_logprobs,
            sample_tokens,
            scale_by_temperature,
        )

        params = [self._req_sampling[req_id] for req_id in req_ids]
        edited = self._edited_logits(last_logits, edit_rows)
        scaled = scale_by_temperature(edited, params)
        positions = [self._first_attention_cache(cache).offset - 1 for cache in caches]
        self._rng_key, first_key = mx.random.split(self._rng_key)
        first = sample_tokens(
            edited,
            params,
            positions,
            first_key,
            scaled=scaled,
        )

        recent_masks = self._recent_token_masks(
            req_ids,
            self._cosyvoice3_sampling_pending_tokens,
        )
        first_is_speech = first < SPEECH_TOKEN_SIZE
        repeated = first_is_speech & mx.take_along_axis(
            recent_masks,
            mx.minimum(first, SPEECH_TOKEN_SIZE - 1)[:, None],
            axis=1,
        ).squeeze(-1)

        fallback_params = [
            MlxSamplingParams(
                temperature=param.temperature,
                top_k=1 if param.is_greedy else edited.shape[-1],
                top_p=1.0,
                min_p=0.0,
                seed=param.seed,
            )
            for param in params
        ]
        fallback_mask = _SPEECH_IDS[None, :] == first[:, None]
        fallback_logits = mx.where(
            repeated[:, None] & fallback_mask,
            -float("inf"),
            edited[:, :SPEECH_TOKEN_SIZE],
        )
        fallback_logits = mx.concatenate(
            [fallback_logits, edited[:, SPEECH_TOKEN_SIZE:]], axis=1
        )
        self._rng_key, fallback_key = mx.random.split(self._rng_key)
        fallback = sample_tokens(
            fallback_logits,
            fallback_params,
            positions,
            fallback_key,
        )
        tokens = mx.where(repeated, fallback, first)

        lazy_logprobs = (
            compute_logprobs(
                last_logits=edited,
                params=params,
                tokens=tokens,
                spec=logprob_spec,
                scaled=scaled,
            )
            if logprob_spec is not None
            else None
        )
        return tokens, lazy_logprobs

    def decode_batch_start_chained(self, prev):
        if len(prev.req_ids) != 1:
            return super().decode_batch_start_chained(prev)
        from sglang.srt.hardware_backend.mlx.model_runner import MlxPendingDecode

        self._cosyvoice3_sampling_pending_tokens = prev.lazy_tokens
        try:
            logits = self._decode_with_native_cache(
                prev.caches,
                [prev.lazy_tokens[:, None]],
            )
            logits = self._constrain_logits(
                logits,
                prev.req_ids,
                prev.caches,
                pending_tokens=prev.lazy_tokens,
            )
            lazy_tokens, lazy_logprobs = self._select_tokens_with_logprobs(
                logits,
                prev.req_ids,
                prev.caches,
                prev.edit_rows,
                prev.logprob_spec,
            )
        finally:
            self._cosyvoice3_sampling_pending_tokens = None
        return MlxPendingDecode(
            lazy_tokens=lazy_tokens,
            req_ids=prev.req_ids,
            caches=prev.caches,
            lazy_logprobs=lazy_logprobs,
            logprob_spec=prev.logprob_spec,
            edit_rows=prev.edit_rows,
        )

    def prefill_finalize(self, pending) -> int:
        token_id = super().prefill_finalize(pending)
        if 0 <= token_id < SPEECH_TOKEN_SIZE:
            self._record_seen_token(pending.req_id, token_id)
        return token_id

    def decode_batch_finalize(self, pending) -> list[int]:
        token_ids = super().decode_batch_finalize(pending)
        for req_id, token_id in zip(pending.req_ids, token_ids, strict=True):
            if 0 <= token_id < SPEECH_TOKEN_SIZE:
                self._record_seen_token(req_id, token_id)
        return token_ids

    def _record_seen_token(self, req_id: str, token_id: int) -> None:
        seen = self._cosyvoice3_seen_masks[req_id] | (_SPEECH_IDS == token_id)
        mx.eval(seen)
        self._cosyvoice3_seen_masks[req_id] = seen
        recent = self._cosyvoice3_recent_tokens.setdefault(req_id, [])
        recent.append(token_id)
        del recent[:-10]

    def remove_request(self, req_id: str) -> None:
        super().remove_request(req_id)
        self._cosyvoice3_prompt_lengths.pop(req_id, None)
        self._cosyvoice3_min_lengths.pop(req_id, None)
        self._cosyvoice3_repetition_penalties.pop(req_id, None)
        self._cosyvoice3_seen_masks.pop(req_id, None)
        self._cosyvoice3_recent_tokens.pop(req_id, None)

    def clear(self) -> None:
        super().clear()
        self._cosyvoice3_sampling_pending_tokens = None
        self._cosyvoice3_prompt_lengths.clear()
        self._cosyvoice3_min_lengths.clear()
        self._cosyvoice3_repetition_penalties.clear()
        self._cosyvoice3_seen_masks.clear()
        self._cosyvoice3_recent_tokens.clear()


def make_fun_cosyvoice3_mlx_runner_class():
    """Build the runner after SGLang's MLX backend has been imported."""
    from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

    class FunCosyVoice3MlxRunner(FunCosyVoice3MlxModelRunner, MlxModelRunner):
        pass

    return FunCosyVoice3MlxRunner


__all__ = ["FunCosyVoice3MlxModelRunner", "make_fun_cosyvoice3_mlx_runner_class"]
