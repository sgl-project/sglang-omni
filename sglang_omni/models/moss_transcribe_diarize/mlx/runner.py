# SPDX-License-Identifier: Apache-2.0
"""SGLang MLX runner extension for MOSS-Transcribe-Diarize audio prefill."""

from __future__ import annotations

import logging
import time

import mlx.core as mx
from mlx_lm.utils import load_model
from sglang.srt.hardware_backend.mlx.remote_code_gate import (
    ensure_remote_code_allowed,
    resolve_model_directory,
)
from sglang.srt.managers.schedule_batch import Req

from sglang_omni.model_runner.audio_mlx import AudioMlxModelRunner
from sglang_omni.models.moss_transcribe_diarize.mlx.config import ModelConfig
from sglang_omni.models.moss_transcribe_diarize.mlx.model import (
    MossTranscribeDiarizeModel,
)

logger = logging.getLogger(__name__)


class MossTranscribeDiarizeMlxModelRunner(AudioMlxModelRunner):
    model_name = "MOSS-Transcribe-Diarize"
    prefill_chunk_size = 2048

    def _load_model(self) -> None:  # noqa: leading-underscore  # SGLang override
        model_path = resolve_model_directory(
            self.model_path,
            revision=self.revision,
        )
        ensure_remote_code_allowed(model_path, self.trust_remote_code)
        logger.info(f"Loading native MLX MOSS-Transcribe-Diarize model: {model_path}")
        started = time.perf_counter()
        self.model, _config = load_model(
            model_path,
            get_model_classes=lambda config: (
                MossTranscribeDiarizeModel,
                ModelConfig,
            ),
        )
        logger.info(
            f"Loaded native MLX MOSS-Transcribe-Diarize model in {time.perf_counter() - started:.2f}s"
        )

    def audio_prefill_inputs(
        self, request: Req, token_ids: list[int]
    ) -> tuple[mx.array, mx.array]:
        audio_item = self.audio_item(request)

        normalized_ids = self.normalize_audio_token_ids(request, token_ids)
        audio_token_id = request.multimodal_inputs.audio_token_id
        audio_positions = [
            index
            for index, token_id in enumerate(normalized_ids)
            if token_id == audio_token_id
        ]
        audio_features = self.model.get_audio_features(
            mx.array(self.to_numpy(audio_item.feature)),
            mx.array(self.to_numpy(audio_item.audio_feature_lengths)),
        )
        input_ids = mx.array([normalized_ids], dtype=mx.int32)
        input_embeddings = self.model.build_inputs_embeds(
            input_ids,
            audio_features,
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
