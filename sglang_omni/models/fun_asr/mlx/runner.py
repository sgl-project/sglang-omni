# SPDX-License-Identifier: Apache-2.0
"""Fun-ASR audio prefill on SGLang's native MLX worker."""

from sglang_omni.model_runner.audio_mlx import AudioMlxModelRunner


class FunASRMlxModelRunner(AudioMlxModelRunner):
    model_name = "Fun-ASR"

    def _load_model(self):
        from mlx_lm.utils import load_model
        from sglang.srt.hardware_backend.mlx.remote_code_gate import (
            ensure_remote_code_allowed,
            resolve_model_directory,
        )

        from .config import ModelConfig
        from .model import FunASRModel

        if self._quantization is not None:
            raise ValueError("Fun-ASR MLX currently requires unquantized HF weights")
        path = resolve_model_directory(self.model_path, revision=self.revision)
        ensure_remote_code_allowed(path, self.trust_remote_code)
        self.model, _ = load_model(
            path, get_model_classes=lambda config: (FunASRModel, ModelConfig)
        )


def make_fun_asr_mlx_runner_class():
    from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

    class FunASRMlxRunner(FunASRMlxModelRunner, MlxModelRunner):
        pass

    return FunASRMlxRunner
