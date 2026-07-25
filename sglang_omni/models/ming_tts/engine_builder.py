# SPDX-License-Identifier: Apache-2.0
"""Ming-Omni-TTS SGLang engine builder."""

from __future__ import annotations

import logging
from typing import Any

from sglang_omni.scheduling.engine_factory import TtsEngineBuilder

logger = logging.getLogger(__name__)


def _coerce_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


class MingTtsEngineBuilder(TtsEngineBuilder):
    model_name = "Ming-Omni-TTS"
    context_length = 0  # resolved from the checkpoint config in pre_infra_setup

    def __init__(
        self,
        *,
        context_length: int | None = None,
        total_gpu_memory_fraction: float | None = None,
        tp_rank: int = 0,
        tp_size: int = 1,
        nccl_port: int | None = None,
    ) -> None:
        from sglang_omni.models.ming_tts.hf_config import MING_TTS_MODEL_ARCH_OVERRIDE

        tp_rank = int(tp_rank)
        tp_size = int(tp_size)
        if tp_size <= 0:
            raise ValueError(
                f"Ming-Omni-TTS tts_engine tp_size must be positive; got {tp_size}"
            )
        if tp_rank < 0 or tp_rank >= tp_size:
            raise ValueError(
                f"Ming-Omni-TTS tts_engine tp_rank={tp_rank} is out of range "
                f"for tp_size={tp_size}"
            )
        if tp_size > 1 and nccl_port is None:
            raise ValueError("Ming-Omni-TTS tts_engine TP requires nccl_port")

        self.model_arch_override = MING_TTS_MODEL_ARCH_OVERRIDE
        self.requested_context_length = context_length
        self.total_gpu_memory_fraction = total_gpu_memory_fraction
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.nccl_port = nccl_port
        self.config: Any = None
        self.tokenizer: Any = None
        self._model_runner: Any = None

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        from sglang_omni.models.ming_tts import stages as ming_stages

        self.config = ming_stages._load_ming_tts_config(checkpoint_dir)
        ming_stages._check_ming_tts_tp_backbone_config(self.config, self.tp_size)
        context_length = int(self.requested_context_length or 0)
        if context_length <= 0:
            context_length = ming_stages._resolve_context_length(self.config)
        self.context_length = int(context_length)

    def generation_defaults(self, *, dtype: str) -> dict[str, Any]:
        return {
            "max_running_requests": 8,
            "dtype": dtype,
            "disable_cuda_graph": True,
            "disable_overlap_schedule": True,
            "disable_radix_cache": True,
            "enable_torch_compile": False,
            "max_prefill_tokens": min(int(self.context_length), 8192),
            "sampling_backend": "pytorch",
            "trust_remote_code": False,
        }

    def adjust_overrides(self, overrides: dict[str, Any]) -> None:
        # context_length is supplied by build_sglang_server_args directly.
        overrides.pop("context_length", None)
        overrides["tp_size"] = self.tp_size

        if "disable_radix_cache" in overrides and not _coerce_bool(
            overrides["disable_radix_cache"],
            name="server_args_overrides.disable_radix_cache",
        ):
            raise ValueError(
                "Ming-Omni-TTS prefix/radix cache is not currently supported"
            )
        overrides["disable_radix_cache"] = True

        if (
            "chunked_prefill_size" in overrides
            and int(overrides.get("chunked_prefill_size") or 0) != 0
        ):
            raise ValueError(
                "Ming-Omni-TTS requires chunked_prefill_size=0 because generated "
                "continuous state does not have chunk rollback semantics"
            )
        overrides["chunked_prefill_size"] = 0
        if bool(overrides.get("enable_torch_compile", False)):
            raise ValueError("Ming-Omni-TTS torch.compile is not currently supported")

    def infra_kwargs(self) -> dict[str, Any]:
        return {
            "tp_rank": self.tp_rank,
            "nccl_port": self.nccl_port,
            "total_gpu_memory_fraction": self.total_gpu_memory_fraction,
        }

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
        model_worker.model_runner.model.eval()
        self.tokenizer = load_ming_tts_tokenizer(
            checkpoint_dir,
            llm_config=self.config.llm_config,
        )
        logger.info(
            "Ming AR SGLang startup: gpu_id=%s tp_rank=%s/%s "
            "total_gpu_memory_fraction=%s disable_cuda_graph=%s cuda_graph_bs=%s "
            "radix_cache=%s nccl_port=%s",
            gpu_id,
            self.tp_rank,
            self.tp_size,
            self.total_gpu_memory_fraction,
            bool(getattr(server_args, "disable_cuda_graph", False)),
            getattr(server_args, "cuda_graph_bs", None),
            not bool(getattr(server_args, "disable_radix_cache", True)),
            self.nccl_port,
        )

    def get_model_buffer_bs(self, model: Any) -> int | None:
        return int(model._decode_input_embedding.num_embeddings)

    def post_cuda_graph_setup(self, model: Any, server_args: Any) -> None:
        del server_args
        # The acoustic tail graph replays only on the owning rank; other TP
        # ranks run the backbone graph and skip latent sampling.
        if self.tp_rank != 0:
            return
        model.init_tail_graphs(
            list(self._model_worker.model_runner.graph_runner.capture_bs)
        )

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        from sglang_omni.models.ming_tts.model_runner import MingTTSModelRunner

        self._model_runner = MingTTSModelRunner(model_worker, output_proc)
        return self._model_runner

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        from sglang_omni.models.ming_tts.engine_io import (
            make_ming_tts_scheduler_adapters,
        )

        return make_ming_tts_scheduler_adapters(
            model=model,
            tokenizer=self.tokenizer,
            reset_request=self._model_runner.reset_request,
            owns_acoustic_result=self.tp_rank == 0,
        )

    def make_abort_callback(self) -> Any | None:
        return self._model_runner.reset_request
