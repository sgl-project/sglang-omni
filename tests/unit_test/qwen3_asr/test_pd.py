# SPDX-License-Identifier: Apache-2.0
"""ASR continuation preserves chunk-local transcription semantics."""

from dataclasses import fields
from unittest.mock import Mock

import pytest
import torch
from pydantic import ValidationError
from sglang.srt.managers.schedule_batch import MultimodalInputs
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

import sglang_omni.models.qwen3_asr.engine_builder as engine_builder
from sglang_omni.models.qwen3_asr.config import Qwen3ASRPDPipelineConfig
from sglang_omni.models.qwen3_asr.pd import ASRResumeState, Qwen3ASRPDAdapter
from sglang_omni.models.qwen3_asr.request_builders import (
    Qwen3ASRRequestData,
    make_qwen3_asr_scheduler_adapters,
)
from sglang_omni.scheduling.pd_utils import (
    DecodeContinuation,
    continuation_from_req,
    req_from_continuation,
)
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from tests.unit_test.pipeline.test_pd_utils import make_allocation, prefill_req
from tests.unit_test.qwen3_asr.test_request_builders import FakeTokenizer


@pytest.mark.parametrize("language", [None, "English"])
@pytest.mark.parametrize("prefix", ["", "Earlier words. "])
def test_asr_roundtrip_preserves_transcript(language: str | None, prefix: str) -> None:
    tokenizer = FakeTokenizer()
    adapter = Qwen3ASRPDAdapter(tokenizer=tokenizer)
    request = prefill_req()
    request.output_ids[:] = [10] if language is None else [20]
    request.omni_data = Qwen3ASRRequestData(
        **{
            field.name: getattr(request.omni_data, field.name)
            for field in fields(SGLangARRequestData)
        },
        prompt_token_ids=[10, 11, 12],
        audio_duration_s=12.5,
        language=language,
        streaming_prefix_text=prefix,
    )
    request.omni_data.stage_payload.request.inputs = torch.ones(160)
    request.omni_data.stage_payload.data = {"audio": torch.ones(160)}
    request.multimodal_inputs = MultimodalInputs(mm_items=[])
    request.multimodal_inputs.mrope_position_delta = torch.tensor([0])
    continuation = DecodeContinuation.decode(
        continuation_from_req(request, "transfer", adapter.build_state).encode()
    )
    assert continuation.stage_payload["request"]["inputs"] is None
    assert continuation.stage_payload["data"] is None
    pool = ReqToTokenPool(
        size=1, max_context_len=32, device="cpu", enable_memory_saver=False
    )
    restored = req_from_continuation(
        continuation,
        make_allocation(),
        req_to_token_pool=pool,
        state_restorer=adapter.restore_state,
    )
    assert isinstance(restored.omni_data, Qwen3ASRRequestData)
    assert restored.omni_data.req is restored
    assert restored.omni_data.output_ids is restored.output_ids
    assert restored.multimodal_inputs.mrope_position_delta.tolist() == [0]
    assert restored.multimodal_inputs.mm_items == []
    suffix = [100, 101, 20, 21, 22] if language is None else [21, 22]
    request.output_ids.extend(suffix)
    restored.output_ids.extend(suffix)
    _, result_adapter = make_qwen3_asr_scheduler_adapters(
        tokenizer=tokenizer, max_new_tokens=16
    )
    original = result_adapter(request.omni_data).data
    resumed = result_adapter(restored.omni_data).data
    for key in ("text", "language", "duration_s"):
        assert resumed[key] == original[key]
    assert resumed["text"].startswith(prefix)
    pool.free(restored)
    assert pool.available_size() == 1


def test_asr_rejects_nonzero_position_delta() -> None:
    request = prefill_req()
    request.omni_data = Qwen3ASRRequestData(
        **{
            field.name: getattr(request.omni_data, field.name)
            for field in fields(SGLangARRequestData)
        }
    )
    request.multimodal_inputs = MultimodalInputs(mm_items=[])
    request.multimodal_inputs.mrope_position_delta = torch.tensor([1])
    with pytest.raises(ValueError, match="mRoPE"):
        Qwen3ASRPDAdapter(tokenizer=FakeTokenizer()).build_state(request)


@pytest.mark.parametrize("duration", [-1.0, float("nan"), float("inf")])
def test_asr_rejects_invalid_resume_duration(duration: float) -> None:
    with pytest.raises(ValidationError):
        ASRResumeState.model_validate(
            {
                "schema": "qwen3-asr-v1",
                "audio_duration_s": duration,
                "language": None,
                "streaming_prefix_text": "",
                "engine_start_s": 0.0,
            }
        )


def test_asr_pd_retains_audio_chunking_and_streaming_strategy() -> None:
    config = Qwen3ASRPDPipelineConfig(model_path="local-model")
    stages = {stage.name: stage for stage in config.stages}
    assert config.allow_audio_chunking
    assert config.realtime_transcription is not None
    assert stages["asr"].next == "asr_decode"
    assert not stages["asr"].terminal
    assert stages["asr_decode"].terminal
    assert stages["asr_decode"].factory.enable_pre_lm_encoder is False
    assert stages["asr_decode"].factory.enable_encoder_cuda_graph is False
    for stage in stages.values():
        assert stage.engine.disable_radix_cache
        assert stage.engine.page_size == 1


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_asr_pd_uses_existing_scheduler_factory_hook(
    role: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = object.__new__(engine_builder.Qwen3ASREngineBuilder)
    builder.pd_role = role
    builder.pd_stage_name = "asr" if role == "prefill" else "asr_decode"
    builder.pd_partner_stage = "asr_decode"
    builder.tokenizer = FakeTokenizer()
    builder.make_abort_callback = Mock(return_value=None)
    builder.make_request_finished_callback = Mock(return_value=None)
    builder.extra_scheduler_callbacks = Mock(return_value={})
    stream_builder = Mock()
    builder.extra_scheduler_kwargs = Mock(
        return_value={"stream_output_builder": stream_builder}
    )
    factory = Mock()
    monkeypatch.setattr(
        engine_builder,
        "OmniPrefillScheduler" if role == "prefill" else "OmniDecodeScheduler",
        factory,
    )
    builder.make_scheduler(
        model_worker=Mock(),
        tree_cache=Mock(),
        req_to_token_pool=Mock(),
        token_to_kv_pool_allocator=Mock(),
        server_args=Mock(),
        model_config=Mock(),
        model_runner=Mock(),
        request_builder=Mock(),
        result_adapter=Mock(),
    )
    arguments = factory.call_args.kwargs
    assert arguments["stage_name"] == builder.pd_stage_name
    if role == "prefill":
        assert arguments["stream_output_builder"] is None
        assert arguments["partner_stage"] == "asr_decode"
    else:
        assert arguments["stream_output_builder"] is stream_builder
        assert arguments["emit_transferred_token"] is True
