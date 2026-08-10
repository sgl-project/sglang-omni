# SPDX-License-Identifier: Apache-2.0
"""Stage factories for LLaDA2-Uni pipeline."""

from __future__ import annotations

import logging
from typing import Any

from sglang_omni.models.llada2_uni.config import IMAGE_STAGE

logger = logging.getLogger(__name__)


def create_preprocessing_executor(
    model_path: str,
    *,
    thinker_max_seq_len: int | None = None,
):
    from sglang_omni.models.llada2_uni.components.preprocessor import LLaDA2Preprocessor
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    preprocessor = LLaDA2Preprocessor(
        model_path=model_path,
        max_seq_len=thinker_max_seq_len,
    )
    return SimpleScheduler(preprocessor)


def create_image_encoder_executor(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: Any = None,
):
    import torch

    from sglang_omni.models.llada2_uni.components.image_encoder import (
        LLaDA2ImageEncoder,
    )
    from sglang_omni.models.llada2_uni.payload_types import LLaDA2UniPipelineState
    from sglang_omni.models.llada2_uni.request_builders import (
        apply_encoder_result,
        build_encoder_request,
        merge_image_tokens_for_thinker,
    )
    from sglang_omni.models.weight_loader import resolve_dtype
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    dtype = resolve_dtype(dtype)

    model = LLaDA2ImageEncoder(model_path=model_path, device=device, dtype=dtype)

    def _encode(payload):
        state = LLaDA2UniPipelineState.from_dict(payload.data)
        request = build_encoder_request(state, stage_name=IMAGE_STAGE)

        if request.get("_skip"):
            result = request.get("_result", {})
        else:
            with torch.no_grad():
                result = model(**request)

        apply_encoder_result(state, stage_name=IMAGE_STAGE, result=result)
        merge_image_tokens_for_thinker(state)
        state.encoder_inputs.clear()
        state.encoder_outs.clear()
        payload.data = state.to_dict()
        return payload

    return SimpleScheduler(_encode)


def create_sglang_dllm_thinker_executor_from_config(
    model_path: str,
    *,
    gpu_id: int = 0,
    thinker_max_seq_len: int = 8192,
    dllm_algorithm: str = "LowConfidence",
    dllm_algorithm_config: str | None = None,
    server_args_overrides: dict[str, Any] | None = None,
):
    """Create an DllmScheduler for the LLaDA2-Uni thinker."""
    from sglang_omni.models.llada2_uni.bootstrap import create_dllm_thinker_scheduler
    from sglang_omni.scheduling.sglang_backend import build_sglang_server_args

    overrides: dict[str, Any] = {
        "attention_backend": "flashinfer",
        "disable_cuda_graph": True,
        "sampling_backend": "pytorch",
    }
    overrides.update(server_args_overrides or {})

    server_args = build_sglang_server_args(
        model_path,
        context_length=thinker_max_seq_len,
        dllm_algorithm=dllm_algorithm,
        dllm_algorithm_config=dllm_algorithm_config,
        **overrides,
    )
    logger.info(
        "create_sglang_dllm_thinker_executor_from_config: "
        "dllm_algorithm=%s, mem_fraction_static=%s",
        server_args.dllm_algorithm,
        server_args.mem_fraction_static,
    )
    return create_dllm_thinker_scheduler(server_args, gpu_id)


def create_decode_executor(model_path: str):
    from sglang_omni.models.llada2_uni.components.streaming_detokenizer import (
        create_streaming_detokenize_scheduler,
    )

    return create_streaming_detokenize_scheduler(model_path)
