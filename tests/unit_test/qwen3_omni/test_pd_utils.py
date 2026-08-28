# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from sglang_omni.models.qwen3_omni.request_builders import (
    QWEN_THINKER_PD_RESUME_SCHEMA,
    make_thinker_pd_adapters,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.pd_utils import DecodeContinuation


def test_qwen_adapter_strips_prefill_tensors_and_restores_mrope() -> None:
    tokenizer = object()
    state_builder, state_restorer = make_thinker_pd_adapters(tokenizer)
    payload = StagePayload(
        request_id="request-1",
        request=OmniRequest(inputs={"text": "hello"}, params={}, metadata={}),
        data={
            "prompt": {"input_ids": torch.tensor([10, 11, 12])},
            "thinker_inputs": {"input_embeds": torch.ones(3, 4)},
            "stream_state": {"token_ids": [42], "text": "x"},
        },
    )
    source = SimpleNamespace(
        _omni_data=SimpleNamespace(stage_payload=payload),
        multimodal_inputs=SimpleNamespace(
            mrope_position_delta=torch.tensor([[3]], dtype=torch.long)
        ),
    )

    projected, resume, input_ids = state_builder(source)
    DecodeContinuation(
        request_id="request-1",
        transfer_id="transfer-1",
        origin_input_ids=input_ids,
        output_ids=[42],
        vocab_size=128,
        sampling_params={},
        stage_payload=projected,
        multimodal_resume=resume,
    ).encode()

    assert projected["request"]["inputs"] is None
    assert projected["data"] == {
        "prompt": {"input_ids": [10, 11, 12]},
        "stream_state": {"token_ids": [42], "text": "x"},
    }
    assert resume["schema"] == QWEN_THINKER_PD_RESUME_SCHEMA

    target = SimpleNamespace()
    state_restorer(target, SimpleNamespace(), resume)
    assert target.tokenizer is tokenizer
    assert target.multimodal_inputs.mrope_position_delta.tolist() == [[3]]
