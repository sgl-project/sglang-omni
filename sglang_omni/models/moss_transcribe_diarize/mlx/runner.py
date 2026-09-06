# SPDX-License-Identifier: Apache-2.0
"""SGLang MLX runner extension for MOSS-Transcribe-Diarize audio prefill."""

from __future__ import annotations

import logging
import time
from typing import Any

import mlx.core as mx

from sglang_omni.model_runner.audio_mlx import AudioMlxModelRunner

logger = logging.getLogger(__name__)


class MossTranscribeDiarizeMlxModelRunner(AudioMlxModelRunner):
    model_name = "MOSS-Transcribe-Diarize"

    def _load_model(self) -> None:
        from mlx_lm.utils import load_model
        from sglang.srt.hardware_backend.mlx.remote_code_gate import (
            ensure_remote_code_allowed,
            resolve_model_directory,
        )

        from .config import ModelConfig
        from .model import MossTranscribeDiarizeModel

        model_path = resolve_model_directory(
            self.model_path,
            revision=self.revision,
        )
        ensure_remote_code_allowed(model_path, self.trust_remote_code)
        logger.info("Loading native MLX MOSS-Transcribe-Diarize model: %s", model_path)
        started = time.perf_counter()
        self.model, _config = load_model(
            model_path,
            get_model_classes=lambda config: (
                MossTranscribeDiarizeModel,
                ModelConfig,
            ),
        )
        logger.info(
            "Loaded native MLX MOSS-Transcribe-Diarize model in %.2fs",
            time.perf_counter() - started,
        )

    @staticmethod
    def _item_data(item: Any, name: str) -> Any:
        value = getattr(item, name, None)
        if value is not None:
            return value
        return getattr(item, "model_specific_data", {}).get(name)

    def _audio_prefill_inputs(
        self, req: Any, token_ids: list[int]
    ) -> tuple[mx.array, mx.array]:
        item = self._audio_item(req)
        if item.feature is None:
            raise ValueError(f"{self.model_name} MLX prefill requires audio features")
        feature_lengths = self._item_data(item, "audio_feature_lengths")
        chunk_mapping = self._item_data(item, "audio_chunk_mapping")
        if feature_lengths is None or chunk_mapping is None:
            raise ValueError(
                f"{self.model_name} MLX prefill requires audio length metadata"
            )

        normalized_ids = self._normalize_audio_token_ids(req, token_ids)
        audio_token_id = int(req.multimodal_inputs.audio_token_id)
        audio_positions = [
            index
            for index, token_id in enumerate(normalized_ids)
            if token_id == audio_token_id
        ]
        if not audio_positions:
            raise ValueError(f"{self.model_name} MLX prefill has no audio placeholders")

        audio_batches = self.model.get_audio_features(
            mx.array(self._to_numpy(item.feature)),
            mx.array(self._to_numpy(feature_lengths)),
            mx.array(self._to_numpy(chunk_mapping)),
        )
        if len(audio_batches) != 1:
            raise ValueError(
                f"{self.model_name} MLX prefill requires exactly one audio"
            )
        input_ids = mx.array([normalized_ids], dtype=mx.int32)
        input_embeddings = self.model._build_inputs_embeds(
            input_ids,
            audio_batches[0],
            audio_positions=audio_positions,
        )
        return input_ids, input_embeddings


def make_moss_transcribe_diarize_mlx_runner_class():
    from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

    class MossTranscribeDiarizeMlxRunner(
        MossTranscribeDiarizeMlxModelRunner, MlxModelRunner
    ):
        pass

    return MossTranscribeDiarizeMlxRunner


__all__ = [
    "MossTranscribeDiarizeMlxModelRunner",
    "make_moss_transcribe_diarize_mlx_runner_class",
]
