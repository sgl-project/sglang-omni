# SPDX-License-Identifier: Apache-2.0
"""SGLang MLX runner extension for Qwen3-ASR audio prefill."""

from __future__ import annotations

import logging
import time

from sglang_omni.model_runner.audio_mlx import AudioMlxModelRunner

logger = logging.getLogger(__name__)


class Qwen3ASRMlxModelRunner(AudioMlxModelRunner):
    """Qwen3-ASR support layered on SGLang's native MLX model runner.

    The base runner continues to own cache layout, pool sizing, radix state,
    and batched decode. This mixin only supplies the unsupported Qwen3-ASR
    model class and the multimodal first-prefill operation.
    """

    model_name = "Qwen3-ASR"

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
