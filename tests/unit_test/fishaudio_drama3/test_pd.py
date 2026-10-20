# SPDX-License-Identifier: Apache-2.0
"""Drama 3 continuation preserves frames consumed by the existing output adapter."""

from array import array
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.models.fishaudio_drama3 import engine_builder
from sglang_omni.models.fishaudio_drama3.engine_builder import Drama3EngineBuilder
from sglang_omni.models.fishaudio_drama3.pd import build_state, restore_state
from sglang_omni.models.fishaudio_drama3.request_builders import (
    Drama3RequestData,
    Drama3ResultAdapter,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.pd_utils import (
    DecodeContinuation,
    ReservedKV,
    continuation_from_req,
    req_from_continuation,
)
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData


def prefilled_request(first_main_token: int = 7) -> Req:
    sampling = SamplingParams(max_new_tokens=1, temperature=0.7, top_k=3, top_p=0.9)
    sampling.normalize(None)
    request = Req(
        rid="drama-pd",
        origin_input_text="",
        origin_input_ids=array("q", [3, 4]),
        sampling_params=sampling,
        vocab_size=32,
        eos_token_ids={2},
    )
    request.output_ids.append(first_main_token)
    request.omni_data = Drama3RequestData(
        req=request,
        input_ids=torch.tensor([3, 4]),
        output_ids=request.output_ids,
        stage_payload=StagePayload(
            request_id=request.rid,
            request=OmniRequest(inputs=torch.ones(3), params={"stream": True}),
            data=torch.ones(2, 3),
        ),
        frame_rows=[[3, 1, 2], [4, 2, 3], [first_main_token, 3, 4]],
        prompt_row_count=2,
        max_new_tokens=1,
        temperature=0.7,
        top_k=3,
        top_p=0.9,
    )
    return request


@pytest.mark.parametrize("first_main_token", [7, 2])
def test_handoff_preserves_complete_first_frame_and_result(first_main_token):
    source = prefilled_request(first_main_token)
    continuation = DecodeContinuation.decode(
        continuation_from_req(source, "transfer", build_state).encode()
    )
    restored = req_from_continuation(
        continuation,
        ReservedKV(slots=torch.tensor([5, 6]), page_indices=(5, 6), seq_len=2),
        req_to_token_pool=ReqToTokenPool(
            size=2, max_context_len=8, device="cpu", enable_memory_saver=False
        ),
        state_restorer=restore_state,
    )
    assert isinstance(restored.omni_data, Drama3RequestData)
    assert restored.omni_data.frame_rows == source.omni_data.frame_rows
    assert restored.omni_data.streamed_frame_count == 0
    assert restored.omni_data.temperature == source.omni_data.temperature
    assert restored.omni_data.top_k == source.omni_data.top_k
    assert restored.omni_data.top_p == source.omni_data.top_p
    assert restored.omni_data.stage_payload.data is None
    assert restored.omni_data.stage_payload.request.inputs is None
    assert isinstance(source.omni_data.stage_payload.data, torch.Tensor)
    adapter = Drama3ResultAdapter(eos_token_id=2)
    expected = adapter(source.omni_data).data
    actual = adapter(restored.omni_data).data
    assert torch.equal(actual.frames, expected.frames)
    assert actual.prompt_row_count == expected.prompt_row_count


def test_handoff_refuses_audio_already_emitted_by_prefill():
    source = prefilled_request()
    source.omni_data.streamed_frame_count = 1
    with pytest.raises(ValueError, match="must not emit"):
        build_state(source)


@pytest.mark.parametrize(
    "bad_rows",
    [
        [[3, 1, 2], [4, 2, 3]],
        [[3, 1, 2], [4, 2, 3], [8, 3, 4]],
        [[3, 1, 2], [4, 2, 3], [7, 3]],
        [[3, 1, 2], [4, 2, 3], [7, 3.5, 4]],
    ],
)
def test_restore_rejects_incomplete_or_inconsistent_frames(bad_rows):
    source = prefilled_request()
    _, resume, _ = build_state(source)
    resume["frame_rows"] = bad_rows
    base = SGLangARRequestData(
        **{
            field.name: getattr(source.omni_data, field.name)
            for field in fields(SGLangARRequestData)
        }
    )
    with pytest.raises(ValueError):
        restore_state(source, base, resume)


@pytest.mark.parametrize("role", ["prefill", "decode"])
@pytest.mark.parametrize("async_decode", [False, True])
def test_pd_builder_keeps_runner_and_async_settings(monkeypatch, role, async_decode):
    prefill_constructor = Mock()
    decode_constructor = Mock()
    monkeypatch.setattr(engine_builder, "OmniPrefillScheduler", prefill_constructor)
    monkeypatch.setattr(engine_builder, "OmniDecodeScheduler", decode_constructor)
    builder = Drama3EngineBuilder(
        context_length=64,
        max_running_requests=2,
        pd_role=role,
        pd_stage_name="tts_engine" if role == "prefill" else "tts_decode",
        pd_partner_stage="tts_decode" if role == "prefill" else None,
    )
    runner = object()
    streamer = object()
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            model=SimpleNamespace(
                config=SimpleNamespace(
                    audio_decoder_config=SimpleNamespace(
                        num_codebooks=10, vocab_size=32
                    )
                )
            )
        )
    )
    builder.make_scheduler(
        model_worker=worker,
        model_runner=runner,
        extra_scheduler_kwargs={
            "enable_async_decode": async_decode,
            "stream_output_builder": streamer,
        },
    )
    constructor = prefill_constructor if role == "prefill" else decode_constructor
    arguments = constructor.call_args.kwargs
    assert arguments["model_runner"] is runner
    assert arguments["tp_worker"] is worker
    assert arguments["enable_async_decode"] is async_decode
    if role == "prefill":
        assert arguments["stream_output_builder"] is None
        assert arguments["partner_stage"] == "tts_decode"
    else:
        assert arguments["stream_output_builder"] is streamer
        assert arguments["emit_transferred_token"] is True
        assert arguments["state_restorer"].keywords == {
            "frame_width": 11,
            "codebook_size": 32,
        }
