# SPDX-License-Identifier: Apache-2.0
"""Executor factories for the Qwen3-Omni split pipeline."""

from __future__ import annotations

from transformers import AutoTokenizer
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
    Qwen3OmniMoeProcessor,
)

from sglang_omni.engines.omni import create_ar_engine
from sglang_omni.executors import EngineExecutor, FrontendExecutor
from sglang_omni.frontends import Qwen3OmniFrontend
from sglang_omni.models.qwen3_omni.engines import AsyncModuleEngine
from sglang_omni.models.qwen3_omni.hf_split import (
    Qwen3OmniAudioEncoder,
    Qwen3OmniImageEncoder,
    Qwen3OmniSplitThinker,
)
from sglang_omni.models.qwen3_omni.request_builders import (
    build_audio_encoder_request,
    build_image_encoder_request,
    build_thinker_ar_request,
)
from sglang_omni.models.qwen3_omni.result_builders import (
    build_audio_result,
    build_image_result,
    build_thinker_result,
)
from sglang_omni.proto import StagePayload


def create_frontend_executor(model_id: str) -> FrontendExecutor:
    frontend = Qwen3OmniFrontend(model_id=model_id)
    return FrontendExecutor(frontend)


def create_image_encoder_executor(
    model_id: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
) -> EngineExecutor:
    model = Qwen3OmniImageEncoder(model_id=model_id, device=device, dtype=dtype)
    engine = AsyncModuleEngine(model)
    return EngineExecutor(
        engine=engine,
        request_builder=build_image_encoder_request,
        result_builder=build_image_result,
    )


def create_audio_encoder_executor(
    model_id: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
) -> EngineExecutor:
    model = Qwen3OmniAudioEncoder(model_id=model_id, device=device, dtype=dtype)
    engine = AsyncModuleEngine(model)
    return EngineExecutor(
        engine=engine,
        request_builder=build_audio_encoder_request,
        result_builder=build_audio_result,
    )


def create_aggregate_executor() -> FrontendExecutor:
    def _identity(payload: StagePayload) -> StagePayload:
        return payload

    return FrontendExecutor(_identity)


def create_thinker_executor(
    model_id: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
    max_seq_len: int = 8192,
) -> EngineExecutor:
    model = Qwen3OmniSplitThinker(model_id=model_id, device=device, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    engine = create_ar_engine(
        model=model,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        device=device,
    )
    return EngineExecutor(
        engine=engine,
        request_builder=build_thinker_ar_request,
        result_builder=build_thinker_result,
    )


def create_decode_executor(model_id: str) -> FrontendExecutor:
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = processor.tokenizer

    def _decode(payload: StagePayload) -> StagePayload:
        data = payload.data if isinstance(payload.data, dict) else {}
        thinker_out = (data.get("engine_outputs", {}) if isinstance(data, dict) else {}).get("thinker", {})
        output_ids = thinker_out.get("output_ids", [])
        text = tokenizer.decode(output_ids, skip_special_tokens=True)
        payload.data = {"text": text, "modality": "text"}
        return payload

    return FrontendExecutor(_decode)
