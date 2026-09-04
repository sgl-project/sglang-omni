# SPDX-License-Identifier: Apache-2.0
"""SGLang MLX runner extension for Qwen3-ASR audio prefill."""

from __future__ import annotations

import logging
import time
from typing import Any

import mlx.core as mx
import torch

logger = logging.getLogger(__name__)


class Qwen3ASRMlxModelRunner:
    """Qwen3-ASR support layered on SGLang's native MLX model runner.

    The base runner continues to own cache layout, pool sizing, radix state,
    and batched decode. This mixin only supplies the unsupported Qwen3-ASR
    model class and the multimodal first-prefill operation.
    """

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

    @staticmethod
    def _audio_item(req: Any) -> Any:
        mm_inputs = req.multimodal_inputs
        if mm_inputs is None:
            raise ValueError("Qwen3-ASR MLX prefill requires multimodal inputs")
        if len(mm_inputs.mm_items) != 1:
            raise ValueError(
                "Qwen3-ASR MLX prefill requires exactly one audio item, got "
                f"{len(mm_inputs.mm_items)}"
            )
        return mm_inputs.mm_items[0]

    @staticmethod
    def _to_numpy(tensor: Any) -> Any:
        tensor = tensor.detach().cpu()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        return tensor.numpy()

    @staticmethod
    def _normalize_audio_token_ids(req: Any, token_ids: list[int]) -> list[int]:
        item = Qwen3ASRMlxModelRunner._audio_item(req)
        mm_inputs = req.multimodal_inputs
        if mm_inputs.audio_token_id is None or item.pad_value is None:
            raise ValueError(
                "Qwen3-ASR MLX prefill has incomplete audio token metadata"
            )
        audio_token_id = int(mm_inputs.audio_token_id)
        pad_value = int(item.pad_value)
        return [
            audio_token_id if int(token_id) == pad_value else int(token_id)
            for token_id in token_ids
        ]

    def _audio_prefill_inputs(
        self, req: Any, token_ids: list[int]
    ) -> tuple[mx.array, mx.array]:
        item = self._audio_item(req)
        if item.feature is None or item.feature_attention_mask is None:
            raise ValueError("Qwen3-ASR MLX prefill requires audio features and mask")

        normalized_ids = self._normalize_audio_token_ids(req, token_ids)
        audio_token_id = int(req.multimodal_inputs.audio_token_id)
        audio_positions = [
            index
            for index, token_id in enumerate(normalized_ids)
            if token_id == audio_token_id
        ]
        if not audio_positions:
            raise ValueError("Qwen3-ASR MLX prefill has no audio placeholders")
        audio_start = audio_positions[0]
        num_audio_tokens = len(audio_positions)
        if audio_positions != list(range(audio_start, audio_start + num_audio_tokens)):
            raise ValueError("Qwen3-ASR MLX audio placeholders must be contiguous")
        input_ids = mx.array([normalized_ids], dtype=mx.int32)
        input_features = mx.array(self._to_numpy(item.feature))
        feature_attention_mask = mx.array(self._to_numpy(item.feature_attention_mask))
        audio_features = self.model.get_audio_features(
            input_features, feature_attention_mask
        )
        input_embeddings = self.model._build_inputs_embeds(
            input_ids,
            audio_features,
            audio_start=audio_start,
            num_audio_tokens=num_audio_tokens,
        )
        return input_ids, input_embeddings

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

        if req is None:
            raise ValueError("Qwen3-ASR MLX prefill requires its scheduler request")
        if prefix_slot_ids:
            raise NotImplementedError(
                "Qwen3-ASR MLX audio prefill does not support a radix prefix yet"
            )
        if not self.disable_radix_cache:
            raise RuntimeError(
                "Qwen3-ASR MLX Stage A requires disable_radix_cache=True"
            )
        if logit_edit_row is not None or logprob_spec is not None:
            raise NotImplementedError(
                "Qwen3-ASR MLX audio prefill supports greedy decoding only"
            )

        input_ids, input_embeddings = self._audio_prefill_inputs(req, new_token_ids)
        cache = self._acquire_cache()
        logits = self.model._forward_last_logits(input_embeddings, cache=cache)
        # note (yexiaodong): Chunked prefill is disabled for this audio path, so
        # needs_logits is always true; retain the argument for the SGLang API.
        del needs_logits
        lazy_token = mx.argmax(logits[:, -1, :], axis=-1)
        return MlxPendingPrefill(
            lazy_token=lazy_token,
            cache=cache,
            req_id=req_id,
            # note (yexiaodong): Later decode bookkeeping requires real model
            # token IDs instead of Omni's out-of-vocabulary audio placeholder.
            full_token_ids=self._normalize_audio_token_ids(req, full_token_ids),
            req_pool_idx=req_pool_idx,
            synced_offset=0,
            lazy_logprobs=None,
        )

    def decode_batch_start(
        self,
        req_ids: list[str],
        edit_rows: mx.array | None = None,
        logprob_spec: Any = None,
        logits_hook: Any = None,
    ):
        if (
            len(req_ids) != 1
            or edit_rows is not None
            or logprob_spec is not None
            or logits_hook is not None
        ):
            return super().decode_batch_start(
                req_ids,
                edit_rows=edit_rows,
                logprob_spec=logprob_spec,
                logits_hook=logits_hook,
            )

        from sglang.srt.hardware_backend.mlx.model_runner import MlxPendingDecode

        req_id = req_ids[0]
        cache = self._req_caches[req_id]
        input_ids = mx.array(
            [[self._req_token_ids[req_id][-1]]],
            dtype=mx.int32,
        )
        lazy_logits = self._decode_with_native_cache([cache], [input_ids])
        lazy_tokens = mx.argmax(lazy_logits, axis=-1)
        return MlxPendingDecode(
            lazy_tokens=lazy_tokens,
            req_ids=[req_id],
            caches=[cache],
            lazy_logprobs=None,
            logprob_spec=None,
            edit_rows=None,
        )

    def decode_batch_start_chained(self, prev):
        if (
            len(prev.req_ids) != 1
            or prev.edit_rows is not None
            or prev.logprob_spec is not None
        ):
            return super().decode_batch_start_chained(prev)

        from sglang.srt.hardware_backend.mlx.model_runner import MlxPendingDecode

        lazy_logits = self._decode_with_native_cache(
            prev.caches,
            [prev.lazy_tokens[:, None]],
        )
        lazy_tokens = mx.argmax(lazy_logits, axis=-1)
        return MlxPendingDecode(
            lazy_tokens=lazy_tokens,
            req_ids=prev.req_ids,
            caches=prev.caches,
            lazy_logprobs=None,
            logprob_spec=None,
            edit_rows=None,
        )


def make_qwen3_asr_mlx_runner_class():
    """Build the extension class after the MLX backend has been selected."""
    from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

    class Qwen3ASRMlxRunner(Qwen3ASRMlxModelRunner, MlxModelRunner):
        pass

    return Qwen3ASRMlxRunner
