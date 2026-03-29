# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-V pipeline helpers."""

from sglang_omni.models.minicpm_v.pipeline.engine_io import (
    apply_encoder_result,
    apply_llm_result,
    build_encoder_request,
    build_llm_request,
    build_sglang_llm_request,
)
from sglang_omni.models.minicpm_v.pipeline.merge import (
    build_llm_inputs,
    merge_for_llm,
)
from sglang_omni.models.minicpm_v.pipeline.next_stage import (
    AGGREGATE_STAGE,
    DECODE_STAGE,
    IMAGE_STAGE,
    LLM_STAGE,
    PREPROCESSING_STAGE,
    aggregate_next,
    decode_next,
    encoder_next,
    llm_next,
    preprocessing_next,
)
from sglang_omni.models.minicpm_v.pipeline.state_io import load_state, store_state

__all__ = [
    # Stage names
    "PREPROCESSING_STAGE",
    "IMAGE_STAGE",
    "AGGREGATE_STAGE",
    "LLM_STAGE",
    "DECODE_STAGE",
    # Routing functions
    "preprocessing_next",
    "encoder_next",
    "aggregate_next",
    "llm_next",
    "decode_next",
    # Merge functions
    "merge_for_llm",
    "build_llm_inputs",
    # Engine IO
    "build_encoder_request",
    "apply_encoder_result",
    "build_llm_request",
    "build_sglang_llm_request",
    "apply_llm_result",
    # State IO
    "load_state",
    "store_state",
]
