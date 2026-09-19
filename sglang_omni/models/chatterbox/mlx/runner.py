# SPDX-License-Identifier: Apache-2.0
"""SGLang MLX runner adapter for Chatterbox-Turbo T3 speech-token decoding."""

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx

_SPEECH_VOCAB_SIZE = 6563
_START_SPEECH_TOKEN = 6561
_SPEECH_IDS = mx.arange(_SPEECH_VOCAB_SIZE)
mx.eval(_SPEECH_IDS)


class ChatterboxT3MlxModelRunner:
    """Customize prompt prefill with the T3 conditioning prefix and apply the
    repetition penalty in the lazy graph; generic MLX cache/decode stays
    upstream."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._chatterbox_repetition_penalties: dict[str, float] = {}
        self._chatterbox_seen_masks: dict[str, mx.array] = {}

    def _load_model(self) -> None:
        from sglang.srt.hardware_backend.mlx.remote_code_gate import (
            resolve_model_directory,
        )

        from .config import ChatterboxT3MlxConfig
        from .loader import load_t3_weights
        from .model import ChatterboxT3MlxModel

        model_dir = resolve_model_directory(self.model_path, revision=self.revision)
        self.model = ChatterboxT3MlxModel(ChatterboxT3MlxConfig())
        load_t3_weights(self.model, model_dir)

        self._builtin_speaker_emb = None
        self._builtin_cond_speech_tokens = None
        conds_path = os.path.join(model_dir, "conds.pt")
        if os.path.exists(conds_path):
            from chatterbox.tts_turbo import Conditionals

            conds = Conditionals.load(conds_path, map_location="cpu")
            self._builtin_speaker_emb = mx.array(
                conds.t3.speaker_emb.detach().float().cpu().numpy()
            )
            self._builtin_cond_speech_tokens = mx.array(
                conds.t3.cond_prompt_speech_tokens.detach().cpu().numpy().astype("int32")
            )

    def _request_prompt(self, req: Any) -> tuple[mx.array, mx.array, mx.array]:
        text_tokens = getattr(req, "_chatterbox_text_tokens", None)
        if text_tokens is None:
            raise ValueError("Chatterbox MLX request is missing text tokens")
        text_tokens = mx.array([list(text_tokens)], dtype=mx.int32)

        speaker = getattr(req, "_chatterbox_speaker_emb", None)
        cond_tokens = getattr(req, "_chatterbox_cond_speech_tokens", None)
        if speaker is None:
            speaker = self._builtin_speaker_emb
            cond_tokens = self._builtin_cond_speech_tokens
            if speaker is None:
                raise ValueError(
                    "Chatterbox MLX request has no speaker and no builtin voice"
                )
        else:
            if hasattr(speaker, "detach"):
                speaker = speaker.detach().cpu().float().numpy()
            speaker = mx.array(speaker)
            cond_tokens = mx.array([list(cond_tokens)], dtype=mx.int32)
        return speaker, cond_tokens, text_tokens

    def _constrain_logits(
        self, logits: mx.array, req_ids: list[str]
    ) -> mx.array:
        """Apply the repetition penalty to already-seen speech tokens."""
        rows = []
        for index, req_id in enumerate(req_ids):
            row = logits[index]
            penalty = self._chatterbox_repetition_penalties.get(req_id, 1.0)
            if penalty != 1.0:
                seen = self._chatterbox_seen_masks[req_id]
                adjusted = mx.where(row > 0, row / penalty, row * penalty)
                row = mx.where(seen, adjusted, row)
            rows.append(row)
        return mx.stack(rows)

    def _record_seen_token(self, req_id: str, token_id: int) -> None:
        if not 0 <= token_id < _START_SPEECH_TOKEN:
            return
        seen = self._chatterbox_seen_masks[req_id] | (_SPEECH_IDS == token_id)
        mx.eval(seen)
        self._chatterbox_seen_masks[req_id] = seen

    def _select_tokens_with_logprobs(
        self,
        last_logits: mx.array,
        req_ids: list[str],
        caches: list[list[Any]],
        edit_rows: mx.array | None = None,
        logprob_spec: Any = None,
    ):
        last_logits = self._constrain_logits(last_logits, req_ids)
        return super()._select_tokens_with_logprobs(
            last_logits, req_ids, caches, edit_rows, logprob_spec
        )

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
        from sglang.srt.hardware_backend.mlx.sampling import MlxSamplingParams

        del new_token_ids, new_slot_ids
        if req is None:
            raise ValueError("Chatterbox MLX prefill requires its scheduler request")
        if prefix_slot_ids:
            raise NotImplementedError("Chatterbox MLX does not support radix prefixes yet")

        if self._enable_sampling:
            self._req_sampling[req_id] = MlxSamplingParams.from_req(
                req, deterministic_seeding=self._deterministic_seeding
            )
        self._chatterbox_repetition_penalties[req_id] = float(
            req.sampling_params.repetition_penalty
        )
        self._chatterbox_seen_masks[req_id] = mx.zeros(
            (_SPEECH_VOCAB_SIZE,), dtype=mx.bool_
        )

        speaker, cond_tokens, text_tokens = self._request_prompt(req)
        embeddings = self.model._build_inputs_embeds(speaker, cond_tokens, text_tokens)
        cache = self._acquire_cache()
        logits = self.model._forward_last_logits(embeddings, cache=cache)
        lazy_token, lazy_logprobs = self._select_tokens_with_logprobs(
            logits[:, -1, :], [req_id], [cache], logit_edit_row, logprob_spec
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

    def prefill_finalize(self, pending: Any) -> int:
        token_id = super().prefill_finalize(pending)
        self._record_seen_token(pending.req_id, int(token_id))
        return token_id

    def decode_batch_finalize(self, pending: Any) -> list[int]:
        token_ids = super().decode_batch_finalize(pending)
        for req_id, token_id in zip(pending.req_ids, token_ids):
            self._record_seen_token(req_id, int(token_id))
        return token_ids

    def remove_request(self, req_id: str) -> None:
        super().remove_request(req_id)
        self._chatterbox_repetition_penalties.pop(req_id, None)
        self._chatterbox_seen_masks.pop(req_id, None)

    def clear(self) -> None:
        super().clear()
        self._chatterbox_repetition_penalties.clear()
        self._chatterbox_seen_masks.clear()


def make_chatterbox_t3_mlx_runner_class():
    """Build the runner after SGLang's MLX backend has been imported."""
    from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

    class ChatterboxT3MlxRunner(ChatterboxT3MlxModelRunner, MlxModelRunner):
        pass

    return ChatterboxT3MlxRunner


__all__ = ["ChatterboxT3MlxModelRunner", "make_chatterbox_t3_mlx_runner_class"]
