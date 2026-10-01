# SPDX-License-Identifier: Apache-2.0
"""SGLang MLX runner extension for ARK-ASR audio prefill."""

from __future__ import annotations

import logging
import time

import mlx.core as mx

from sglang_omni.model_runner.audio_mlx import AudioMlxModelRunner

logger = logging.getLogger(__name__)


class ArkasrMlxModelRunner(AudioMlxModelRunner):
    """ARK-ASR model loading and audio prefill for the native MLX runner."""

    model_name = "ARK-ASR"

    def _load_model(self) -> None:
        from mlx_lm.utils import load_model, quantize_model
        from sglang.srt.hardware_backend.mlx.remote_code_gate import (
            ensure_remote_code_allowed,
            resolve_model_directory,
        )

        from .config import ModelConfig
        from .model import ArkasrModel

        model_path = resolve_model_directory(
            self.model_path,
            revision=self.revision,
        )
        ensure_remote_code_allowed(model_path, self.trust_remote_code)
        logger.info("Loading native MLX ARK-ASR model: %s", model_path)
        started = time.perf_counter()
        self.model, config = load_model(
            model_path,
            get_model_classes=lambda config: (ArkasrModel, ModelConfig),
        )
        presets = {"mlx_q4": (4, 64), "mlx_q8": (8, 64)}
        if self._quantization in presets and "quantization" not in config:
            bits, group_size = presets[self._quantization]
            logger.info(
                "Quantizing native MLX ARK-ASR text stack: bits=%d, group_size=%d",
                bits,
                group_size,
            )
            self.model, _config = quantize_model(
                self.model,
                config,
                group_size=group_size,
                bits=bits,
            )
        mx.eval(self.model.parameters())
        logger.info(
            "Loaded native MLX ARK-ASR model in %.2fs",
            time.perf_counter() - started,
        )


def make_arkasr_mlx_runner_class():
    """Build the extension class after the MLX backend has been selected."""
    from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

    class ArkasrMlxRunner(ArkasrMlxModelRunner, MlxModelRunner):
        pass

    return ArkasrMlxRunner
