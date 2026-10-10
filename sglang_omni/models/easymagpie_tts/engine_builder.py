# SPDX-License-Identifier: Apache-2.0
"""EasyMagpie SGLang AR engine builder."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from sglang.kernels.ops.mamba.triton_ops import (
    initialize_mamba_selective_state_update_backend,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.cuda_graph_config import Backend as CudaGraphBackend

from sglang_omni.models.easymagpie_tts import CAPABILITIES
from sglang_omni.models.easymagpie_tts.model_runner import EasyMagpieTTSModelRunner
from sglang_omni.models.easymagpie_tts.payload_types import MAX_TEXT_TOKENS, MAX_TOP_K
from sglang_omni.models.easymagpie_tts.request_builders import (
    apply_easymagpie_result,
    build_sglang_easymagpie_request,
    easymagpie_stream_output_builder,
)
from sglang_omni.models.easymagpie_tts.speakers import load_speaker_embeddings
from sglang_omni.scheduling.engine_factory import TtsEngineBuilder
from sglang_omni.scheduling.generation_batch_policy import (
    build_default_prefill_cuda_graph_bs,
)

EASYMAGPIE_ARCH = "EasyMagpieTTSForConditionalGeneration"
EASYMAGPIE_CONTEXT_LENGTH = 8192
DEFAULT_MAX_RUNNING_REQUESTS = 64
# Concurrent arrivals coalesce into one prefill; bigger batches run eagerly.
PREFILL_GRAPH_MAX_TOKENS = 2048

logger = logging.getLogger(__name__)


def decode_graph_batch_sizes(max_batch: int) -> list[int]:
    """Powers of two up to ``max_batch``, plus ``max_batch`` itself."""
    sizes = [size for size in (1, 2, 4, 8, 16, 32, 64) if size < max_batch]
    return [*sizes, max_batch]


def sglang_captures_mamba_prefill() -> bool:
    """Whether SGLang keeps Mamba2 prefill inside the breakable prefill graph.

    Older SGLang breaks out to eager at every Mamba layer and, at the pinned
    release, drops the talker's ``inputs_embeds`` on replay.
    """
    return hasattr(AttentionBackend, "breakable_cuda_graph_request_slots")


class EasyMagpieTTSEngineBuilder(TtsEngineBuilder):
    model_name = "EasyMagpie-TTS"
    context_length = EASYMAGPIE_CONTEXT_LENGTH
    model_arch_override = EASYMAGPIE_ARCH
    supports_breakable_prefill_cuda_graph = (
        CAPABILITIES.supports_breakable_prefill_cuda_graph
    )

    def __init__(
        self,
        *,
        max_running_requests: int = DEFAULT_MAX_RUNNING_REQUESTS,
        mem_fraction_static: float = 0.72,
        cuda_graph: bool = True,
        enable_async_decode: bool = True,
        async_decode_min_batch_size: int = 1,
    ) -> None:
        self.max_running_requests = max_running_requests
        self.mem_fraction_static = mem_fraction_static
        self.cuda_graph = cuda_graph
        self.enable_async_decode = enable_async_decode
        self.async_decode_min_batch_size = async_decode_min_batch_size

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        del checkpoint_dir
        # note (Yashwant Hayaran): the Mamba conv cache dtype is only
        # configurable through this SGLang env and defaults to bfloat16; a
        # mismatch with the model dtype breaks the causal-conv kernels.
        envs.SGLANG_MAMBA_CONV_DTYPE.set(self.dtype)

    def generation_defaults(self, *, dtype: str) -> dict[str, Any]:
        defaults = {
            "dtype": dtype,
            "max_running_requests": self.max_running_requests,
            "max_total_tokens": self.max_running_requests * EASYMAGPIE_CONTEXT_LENGTH,
            "mem_fraction_static": self.mem_fraction_static,
            # Decode state is seeded from the prompt's last row, so the prompt
            # must arrive in one prefill.
            "chunked_prefill_size": 0,
            "disable_cuda_graph": not self.cuda_graph,
            "cuda_graph_max_bs": self.max_running_requests,
            "cuda_graph_bs": decode_graph_batch_sizes(self.max_running_requests),
            "disable_overlap_schedule": True,
            "disable_radix_cache": True,
            "enable_torch_compile": False,
            "sampling_backend": "pytorch",
            "trust_remote_code": False,
        }
        if self.cuda_graph and sglang_captures_mamba_prefill():
            defaults["cuda_graph_backend_prefill"] = CudaGraphBackend.BREAKABLE
            defaults["cuda_graph_bs_prefill"] = build_default_prefill_cuda_graph_bs(
                PREFILL_GRAPH_MAX_TOKENS
            )
        else:
            defaults["disable_prefill_cuda_graph"] = True
            if self.cuda_graph:
                logger.warning(
                    "This SGLang cannot keep Mamba2 prefill inside the prefill "
                    "CUDA graph; EasyMagpie prefill runs eagerly. Upgrade SGLang "
                    "to replay it from graphs."
                )
            else:
                pass
        return defaults

    def adjust_overrides(self, overrides: dict[str, Any]) -> None:
        if int(overrides.get("tp_size", 1)) != 1:
            raise ValueError("EasyMagpie TTS supports tp_size=1 only")
        else:
            pass
        # Graph buckets and buffers must cover every request the scheduler
        # may batch, including a max_running_requests set per stage.
        max_running = int(
            overrides.get("max_running_requests", self.max_running_requests)
        )
        self.max_running_requests = max_running
        overrides["cuda_graph_max_bs"] = max_running
        overrides["cuda_graph_bs"] = decode_graph_batch_sizes(max_running)

    def customize_server_args(self, server_args: Any) -> None:
        initialize_mamba_selective_state_update_backend(server_args)

    def setup_model(
        self,
        *,
        model_worker: Any,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: Any,
    ) -> None:
        del device, gpu_id
        model_runner = model_worker.model_runner
        model = model_runner.model
        model.eval()
        model.setup_speakers(
            load_speaker_embeddings(
                Path(checkpoint_dir), model.tts_config.embedding_dim
            )
        )
        # Before graph capture, which the factory runs after setup_model. SGLang
        # may lower max_running_requests to fit memory, but the graph buckets
        # still reach the requested size.
        max_batch = max(self.max_running_requests, server_args.max_running_requests)
        model.setup_decode_state(
            num_slots=int(model_runner.req_to_token_pool.size),
            text_capacity=MAX_TEXT_TOKENS,
            max_batch=int(max_batch),
            max_top_k=MAX_TOP_K,
        )

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        return EasyMagpieTTSModelRunner(model_worker, output_proc)

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        del model
        return build_sglang_easymagpie_request, apply_easymagpie_result

    def extra_scheduler_kwargs(self) -> dict[str, Any]:
        return {
            "stream_output_builder": easymagpie_stream_output_builder,
            "enable_async_decode": self.enable_async_decode,
            "async_decode_min_batch_size": self.async_decode_min_batch_size,
        }


__all__ = [
    "DEFAULT_MAX_RUNNING_REQUESTS",
    "EASYMAGPIE_ARCH",
    "EasyMagpieTTSEngineBuilder",
    "PREFILL_GRAPH_MAX_TOKENS",
    "decode_graph_batch_sizes",
    "sglang_captures_mamba_prefill",
]
