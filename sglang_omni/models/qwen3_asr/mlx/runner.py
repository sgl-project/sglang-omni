# SPDX-License-Identifier: Apache-2.0
"""SGLang MLX runner extension for Qwen3-ASR audio prefill."""

from __future__ import annotations

import logging
import time
from typing import Any

import mlx.core as mx

from sglang_omni.model_runner.audio_mlx import AudioMlxModelRunner

logger = logging.getLogger(__name__)


class Qwen3ASRMlxModelRunner(AudioMlxModelRunner):
    """Qwen3-ASR support layered on SGLang's native MLX model runner.

    The base runner continues to own cache layout, pool sizing, radix state,
    and batched decode. This mixin only supplies the unsupported Qwen3-ASR
    model class and the multimodal first-prefill operation.
    """

    model_name = "Qwen3-ASR"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._qwen3_asr_prompt_lengths: dict[str, int] = {}
        self._qwen3_asr_repetition_penalties: dict[str, float] = {}
        self._qwen3_asr_pending_tokens: mx.array | None = None

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
        if req is None:
            raise ValueError("Qwen3-ASR MLX prefill requires its scheduler request")
        sampling_params = getattr(req, "sampling_params", None)
        repetition_penalty = float(getattr(sampling_params, "repetition_penalty", 1.0))
        prompt_lengths = getattr(self, "_qwen3_asr_prompt_lengths", None)
        if prompt_lengths is None:
            prompt_lengths = self._qwen3_asr_prompt_lengths = {}
        penalties = getattr(self, "_qwen3_asr_repetition_penalties", None)
        if penalties is None:
            penalties = self._qwen3_asr_repetition_penalties = {}
        prompt_lengths[req_id] = len(full_token_ids)
        penalties[req_id] = repetition_penalty
        try:
            return super().prefill_start(
                req_id=req_id,
                new_token_ids=new_token_ids,
                full_token_ids=full_token_ids,
                prefix_slot_ids=prefix_slot_ids,
                new_slot_ids=new_slot_ids,
                req_pool_idx=req_pool_idx,
                req=req,
                needs_logits=needs_logits,
                logit_edit_row=logit_edit_row,
                logprob_spec=logprob_spec,
            )
        except Exception:
            prompt_lengths.pop(req_id, None)
            penalties.pop(req_id, None)
            raise

    def _apply_audio_decode_constraints(
        self,
        logits: mx.array,
        req_ids: list[str],
        pending_tokens: mx.array | None = None,
    ) -> mx.array:
        """Apply SGLang repetition-penalty semantics to generated tokens."""
        prompt_lengths = getattr(self, "_qwen3_asr_prompt_lengths", {})
        penalties = getattr(self, "_qwen3_asr_repetition_penalties", {})
        token_history = getattr(self, "_req_token_ids", {})
        vocab_ids = mx.arange(logits.shape[-1], dtype=mx.int32)
        rows = []
        for index, req_id in enumerate(req_ids):
            row = logits[index]
            penalty = penalties.get(req_id, 1.0)
            if penalty == 1.0:
                rows.append(row)
                continue

            prompt_length = prompt_lengths.get(req_id, 0)
            generated_ids = token_history.get(req_id, [])[prompt_length:]
            seen = mx.zeros((logits.shape[-1],), dtype=mx.bool_)
            if generated_ids:
                seen = seen.at[mx.array(generated_ids, dtype=mx.int32)].add(True)
            if pending_tokens is not None:
                seen = seen | (vocab_ids == pending_tokens[index])
            adjusted = mx.where(row < 0, row * penalty, row / penalty)
            rows.append(mx.where(seen, adjusted, row))
        return mx.stack(rows)

    def _select_tokens_with_logprobs(
        self,
        last_logits: mx.array,
        req_ids: list[str],
        caches: list[list[Any]],
        edit_rows: mx.array | None = None,
        logprob_spec: Any = None,
    ):
        constrained_logits = self._apply_audio_decode_constraints(
            last_logits,
            req_ids,
            pending_tokens=getattr(self, "_qwen3_asr_pending_tokens", None),
        )
        return super()._select_tokens_with_logprobs(
            constrained_logits,
            req_ids,
            caches,
            edit_rows,
            logprob_spec,
        )

    def decode_batch_start_chained(self, prev):
        self._qwen3_asr_pending_tokens = prev.lazy_tokens
        try:
            return super().decode_batch_start_chained(prev)
        finally:
            self._qwen3_asr_pending_tokens = None

    def remove_request(self, req_id: str) -> None:
        super().remove_request(req_id)
        self._qwen3_asr_prompt_lengths.pop(req_id, None)
        self._qwen3_asr_repetition_penalties.pop(req_id, None)

    def clear(self) -> None:
        super().clear()
        self._qwen3_asr_pending_tokens = None
        self._qwen3_asr_prompt_lengths.clear()
        self._qwen3_asr_repetition_penalties.clear()

    def _load_model(self) -> None:
        from mlx_lm.utils import load_model
        from sglang.srt.hardware_backend.mlx.remote_code_gate import (
            ensure_remote_code_allowed,
            resolve_model_directory,
        )

        from .config import ModelConfig
        from .model import Qwen3ASRModel

        model_path = resolve_model_directory(
            self.model_path,
            revision=self.revision,
        )
        ensure_remote_code_allowed(model_path, self.trust_remote_code)
        logger.info("Loading native MLX Qwen3-ASR model: %s", model_path)
        started = time.perf_counter()
        self.model, _config = load_model(
            model_path,
            get_model_classes=lambda config: (Qwen3ASRModel, ModelConfig),
        )
        logger.info(
            "Loaded native MLX Qwen3-ASR model in %.2fs",
            time.perf_counter() - started,
        )


def make_qwen3_asr_mlx_runner_class():
    """Build the extension class after the MLX backend has been selected."""
    from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

    class Qwen3ASRMlxRunner(Qwen3ASRMlxModelRunner, MlxModelRunner):
        pass

    return Qwen3ASRMlxRunner
