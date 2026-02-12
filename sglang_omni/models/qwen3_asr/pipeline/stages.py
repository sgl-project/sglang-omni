# SPDX-License-Identifier: Apache-2.0
"""Pipeline stages for Qwen3-ASR."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.engines.omni import create_ar_engine, create_encoder_engine
from sglang_omni.executors import EngineExecutor, PreprocessingExecutor
from sglang_omni.models.qwen3_asr.components.audio_encoder import Qwen3ASRAudioEncoder
from sglang_omni.models.qwen3_asr.components.preprocessor import Qwen3ASRPreprocessor
from sglang_omni.models.qwen3_asr.components.thinker import Qwen3ASRSplitThinker
from sglang_omni.models.qwen3_asr.pipeline.engine_io import (
    apply_encoder_result,
    apply_thinker_result,
    build_encoder_request,
    build_thinker_request,
)
from sglang_omni.models.qwen3_asr.pipeline.next_stage import AUDIO_STAGE, THINKER_STAGE
from sglang_omni.models.qwen3_asr.pipeline.state_io import load_state, store_state
from sglang_omni.proto import StagePayload


def create_preprocessing_executor(model_id: str) -> PreprocessingExecutor:
    preprocessor = Qwen3ASRPreprocessor(model_id=model_id)

    def _preprocess(payload: StagePayload) -> StagePayload:
        return preprocessor(payload)

    return PreprocessingExecutor(_preprocess)


def _create_encoder_executor(
    *,
    stage_name: str,
    model: torch.nn.Module,
    device: str,
) -> EngineExecutor:
    def _request_builder(payload: StagePayload):
        state = load_state(payload)
        return build_encoder_request(state, stage_name=stage_name)

    def _result_builder(payload: StagePayload, result: Any) -> StagePayload:
        state = load_state(payload)
        apply_encoder_result(state, stage_name=stage_name, result=result)
        return store_state(payload, state)

    engine = create_encoder_engine(model, device=device)
    return EngineExecutor(
        engine=engine, request_builder=_request_builder, result_builder=_result_builder
    )


def create_audio_encoder_executor(
    model_id: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
) -> EngineExecutor:
    model = Qwen3ASRAudioEncoder(model_id=model_id, device=device, dtype=dtype)
    return _create_encoder_executor(stage_name=AUDIO_STAGE, model=model, device=device)


from transformers import AutoTokenizer


def create_thinker_executor(
    model_id: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
    **kwargs: Any,
) -> EngineExecutor:
    model = Qwen3ASRSplitThinker(model_id=model_id, device=device, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    def _request_builder(payload: StagePayload):
        state = load_state(payload)
        return build_thinker_request(state, params=kwargs)

    def _result_builder(payload: StagePayload, result: Any) -> StagePayload:
        state = load_state(payload)
        apply_thinker_result(state, stage_name=THINKER_STAGE, result=result)
        return store_state(payload, state)

    engine = create_ar_engine(model, tokenizer=tokenizer, device=device)
    return EngineExecutor(
        engine=engine, request_builder=_request_builder, result_builder=_result_builder
    )
