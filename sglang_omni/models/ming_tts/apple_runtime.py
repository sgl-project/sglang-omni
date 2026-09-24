# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from sglang_omni.models.ming_tts.engine_builder import MingTtsEngineBuilder


def ming_tts_uses_mlx() -> bool:
    from sglang.srt.hardware_backend.mlx.runtime import use_mlx

    from sglang_omni.platforms import current_platform

    selected = use_mlx()
    if selected and not current_platform.is_mps():
        raise ValueError("Ming native MLX requires Apple Silicon / Metal")
    if current_platform.is_mps() and not selected:
        raise ValueError("Ming on Apple requires SGLANG_USE_MLX=1; Torch/MPS is not supported")
    return selected


class MingTtsMlxEngineBuilder(MingTtsEngineBuilder):
    def generation_defaults(self, *, dtype: str) -> dict[str, Any]:
        return {
            **super().generation_defaults(dtype=dtype),
            "max_running_requests": 1,
            "max_total_tokens": self.context_length,
            "max_prefill_tokens": self.context_length,
            "attention_backend": "torch_native",
        }

    def adjust_overrides(self, overrides: dict[str, Any]) -> None:
        super().adjust_overrides(overrides)
        if self.tp_size != 1 or self.tp_rank != 0:
            raise ValueError("Ming MLX requires TP=1")
        if int(overrides["max_running_requests"]) != 1:
            raise ValueError("Ming MLX requires max_running_requests=1")
        if not overrides["disable_cuda_graph"]:
            raise ValueError("Ming MLX requires disable_cuda_graph=true")
        if overrides.get("attention_backend") != "torch_native":
            raise ValueError("Ming MLX bookkeeping requires attention_backend=torch_native")
        for phase in ("prefill_attention_backend", "decode_attention_backend"):
            if overrides.get(phase) not in (None, "torch_native"):
                raise ValueError(f"Ming MLX bookkeeping requires {phase}=torch_native")
        if overrides.get("speculative_algorithm") is not None:
            raise ValueError("Ming MLX does not support speculative decoding")
        if int(overrides["max_total_tokens"]) < self.context_length:
            raise ValueError(
                "Ming MLX token pool must hold the complete context without retraction"
            )
        if int(overrides["max_prefill_tokens"]) < self.context_length:
            raise ValueError("Ming MLX requires a full, unsplit prefill")

    def setup_model(
        self,
        *,
        model_worker: Any,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: Any,
    ) -> None:
        from sglang_omni.models.ming_tts.tokenizer import load_ming_tts_tokenizer

        self._model_worker = model_worker
        self.tokenizer = load_ming_tts_tokenizer(
            checkpoint_dir, llm_config=self.config.llm_config
        )

    def get_model_buffer_bs(self, model: Any) -> int | None:
        return None

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        from .mlx.worker import MingTTSMlxModelRunner

        self._model_runner = MingTTSMlxModelRunner(model_worker, output_proc)
        return self._model_runner

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        return super().make_adapters(self._model_worker._mlx_runner.model)
