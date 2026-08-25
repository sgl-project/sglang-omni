# SPDX-License-Identifier: Apache-2.0
"""SGLang MLX runner extension for Qwen3-ASR audio prefill."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)


def _resolve_model_path(model_path: str) -> Path:
    path = Path(model_path).expanduser()
    if path.is_dir():
        return path.resolve()

    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_path))


class Qwen3ASRMlxModelRunner:
    """Qwen3-ASR support layered on SGLang's native MLX model runner.

    The base runner continues to own cache layout, pool sizing, radix state,
    and batched decode. This mixin only supplies the unsupported Qwen3-ASR
    model class and the multimodal first-prefill operation.
    """

    def _load_model(self) -> None:
        from mlx_lm.utils import load_model

        from .config import ModelConfig
        from .model import Qwen3ASRModel

        model_path = _resolve_model_path(self.model_path)
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
        return tensor.detach().cpu().numpy()

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

        input_ids, input_embeddings = self._audio_prefill_inputs(req, new_token_ids)
        cache = self._acquire_cache()
        logits = self.model._forward_last_logits(input_embeddings, cache=cache)
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
        )

    def decode_batch_start(self, req_ids: list[str]):
        if len(req_ids) != 1:
            return super().decode_batch_start(req_ids)

        from sglang.srt.hardware_backend.mlx.model_runner import MlxPendingDecode

        req_id = req_ids[0]
        cache = self._req_caches[req_id]
        input_ids = mx.array(
            [[self._req_token_ids[req_id][-1]]],
            dtype=mx.int32,
        )
        lazy_tokens = self._decode_with_native_cache([cache], [input_ids])
        return MlxPendingDecode(
            lazy_tokens=lazy_tokens,
            req_ids=[req_id],
            caches=[cache],
        )

    def decode_batch_start_chained(self, prev):
        if len(prev.req_ids) != 1:
            return super().decode_batch_start_chained(prev)

        from sglang.srt.hardware_backend.mlx.model_runner import MlxPendingDecode

        lazy_tokens = self._decode_with_native_cache(
            prev.caches,
            [prev.lazy_tokens[:, None]],
        )
        return MlxPendingDecode(
            lazy_tokens=lazy_tokens,
            req_ids=prev.req_ids,
            caches=prev.caches,
        )


def make_qwen3_asr_mlx_runner_class():
    """Build the extension class lazily so non-Apple imports need no MLX."""
    from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

    class Qwen3ASRMlxRunner(Qwen3ASRMlxModelRunner, MlxModelRunner):
        pass

    return Qwen3ASRMlxRunner
