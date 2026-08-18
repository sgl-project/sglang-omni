# SPDX-License-Identifier: Apache-2.0
"""Ming-Omni-TTS SGLang engine builder."""

from __future__ import annotations

import logging
from typing import Any

from sglang_omni.models.ming_omni.tp_utils import validate_attention_tp_config
from sglang_omni.scheduling.engine_factory import TtsEngineBuilder
from sglang_omni.scheduling.generation_batch_policy import get_decode_cuda_graph_bs

logger = logging.getLogger(__name__)


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


class MingTtsEngineBuilder(TtsEngineBuilder):
    model_name = "Ming-Omni-TTS"
    context_length = 0

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
        if self.tp_size > 1:
            llm_config = self.config.llm_config
            hidden_size = int(llm_config.hidden_size)
            head_dim = int(llm_config.head_dim)
            num_heads = int(llm_config.num_attention_heads)
            num_kv_heads = int(llm_config.num_key_value_heads)
            if min(hidden_size, head_dim, num_heads, num_kv_heads) <= 0:
                raise ValueError(
                    "Ming-Omni-TTS TP requires positive hidden/head dimensions: "
                    f"hidden_size={hidden_size}, head_dim={head_dim}, "
                    f"num_attention_heads={num_heads}, "
                    f"num_key_value_heads={num_kv_heads}"
                )
            if head_dim * num_heads != hidden_size:
                raise ValueError(
                    "Ming-Omni-TTS TP requires head_dim * num_attention_heads "
                    f"to equal hidden_size ({head_dim} * {num_heads} != {hidden_size})"
                )
            if hidden_size % self.tp_size != 0:
                raise ValueError(
                    "Ming-Omni-TTS TP requires hidden_size divisible by tp_size: "
                    f"hidden_size={hidden_size}, tp_size={self.tp_size}"
                )
            validate_attention_tp_config(
                num_attention_heads=num_heads,
                num_key_value_heads=num_kv_heads,
                tp_size=self.tp_size,
                context="Ming-Omni-TTS tts_engine",
            )
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
        overrides.pop("context_length", None)
        overrides["tp_size"] = self.tp_size

        if not _is_truthy(overrides["disable_overlap_schedule"]):
            raise ValueError(
                "Ming-Omni-TTS does not currently support SGLang overlap "
                "scheduling; set disable_overlap_schedule=true because the "
                "continuous acoustic feedback state has no overlap-safe lifecycle"
            )
        overrides["disable_overlap_schedule"] = True

        if not _is_truthy(overrides["disable_radix_cache"]):
            raise ValueError(
                "Ming-Omni-TTS requires disable_radix_cache=true because "
                "prefix/radix cache is not currently supported"
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
        if _is_truthy(overrides.get("enable_torch_compile", False)):
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
            bool(server_args.disable_cuda_graph),
            get_decode_cuda_graph_bs(server_args),
            not bool(server_args.disable_radix_cache),
            self.nccl_port,
        )

    def get_model_buffer_bs(self, model: Any) -> int | None:
        return int(model._decode_input_embedding.num_embeddings)

    def post_cuda_graph_setup(self, model: Any, server_args: Any) -> None:
        del server_args
        # Note (yzxiao): Only the acoustic owner captures tail graphs because
        # follower ranks run the backbone graph without latent sampling.
        if self.tp_rank != 0:
            return
        model.init_tail_graphs(
            list(self._model_worker.model_runner.decode_cuda_graph_runner.capture_bs)
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

    def extra_scheduler_kwargs(self) -> dict[str, Any]:
        if self.tp_rank != 0:
            return {}
        from sglang_omni.models.ming_tts.engine_io import build_ming_tts_stream_output

        return {"stream_output_builder": build_ming_tts_stream_output}

    def make_abort_callback(self) -> Any | None:
        return self._model_runner.reset_request
