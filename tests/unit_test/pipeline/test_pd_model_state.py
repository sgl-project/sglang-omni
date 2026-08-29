# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.ming_omni.bootstrap import MING_THINKER_PD_RESUME_SCHEMA
from sglang_omni.models.ming_omni.bootstrap import (
    make_thinker_pd_adapters as make_ming_pd_adapters,
)
from sglang_omni.models.qwen3_omni.request_builders import QWEN_THINKER_PD_RESUME_SCHEMA
from sglang_omni.models.qwen3_omni.request_builders import (
    make_thinker_pd_adapters as make_qwen_pd_adapters,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.pd_utils import DecodeContinuation


def _payload(data) -> StagePayload:
    return StagePayload(
        request_id="request",
        request=OmniRequest(
            inputs={"discarded": True},
            params={"stream": True},
            metadata={"output_modalities": ["text"]},
        ),
        data=data,
    )


def _continuation(payload, resume, input_ids) -> DecodeContinuation:
    return DecodeContinuation(
        request_id="request",
        transfer_id="transfer",
        origin_input_ids=input_ids,
        output_ids=[42],
        vocab_size=128,
        sampling_params={},
        stage_payload=payload,
        multimodal_resume=resume,
    )


def test_qwen_pd_state_round_trip_uses_generic_continuation_hook() -> None:
    builder, restorer = make_qwen_pd_adapters(tokenizer="tokenizer")
    source = SimpleNamespace(
        origin_input_ids=[10, 11, 12],
        _omni_data=SimpleNamespace(
            stage_payload=_payload(
                {
                    "prompt": {"input_ids": torch.tensor([10, 11, 12])},
                    "stream_state": {"offset": 3},
                    "encoder_outs": {"large": torch.ones(4)},
                }
            )
        ),
        multimodal_inputs=SimpleNamespace(
            mrope_position_delta=torch.tensor([[5]], dtype=torch.long)
        ),
    )
    payload, resume, input_ids = builder(source)
    continuation = DecodeContinuation.decode(
        _continuation(payload, resume, input_ids).encode()
    )

    assert continuation.multimodal_resume == {
        "schema": QWEN_THINKER_PD_RESUME_SCHEMA,
        "mrope_position_delta": [[5]],
    }
    assert continuation.stage_payload["data"] == {
        "prompt": {"input_ids": [10, 11, 12]},
        "stream_state": {"offset": 3},
    }
    destination = SimpleNamespace()
    restorer(destination, None, continuation.multimodal_resume)
    assert destination.tokenizer == "tokenizer"
    assert destination.multimodal_inputs.mrope_position_delta.tolist() == [[5]]


def test_qwen_pd_state_rejects_unknown_schema_and_shape() -> None:
    _, restorer = make_qwen_pd_adapters(tokenizer=None)
    with pytest.raises(ValueError, match="unsupported Qwen"):
        restorer(
            SimpleNamespace(),
            None,
            {"schema": "future", "mrope_position_delta": [[1]]},
        )
    with pytest.raises(ValueError, match="shape"):
        restorer(
            SimpleNamespace(),
            None,
            {
                "schema": QWEN_THINKER_PD_RESUME_SCHEMA,
                "mrope_position_delta": [1, 2],
            },
        )


def test_ming_pd_state_round_trip_uses_generic_continuation_hook() -> None:
    builder, restorer = make_ming_pd_adapters(tokenizer="tokenizer")
    source = SimpleNamespace(
        origin_input_ids=[20, 21, 22],
        _omni_data=SimpleNamespace(
            stage_payload=_payload(
                {
                    "prompt": {"input_ids": torch.tensor([20, 21, 22])},
                    "stream_state": {"offset": 2},
                    "thinker_inputs": {"image_embeds": torch.ones(4, 2)},
                }
            )
        ),
    )
    payload, resume, input_ids = builder(source)
    continuation = DecodeContinuation.decode(
        _continuation(payload, resume, input_ids).encode()
    )

    assert continuation.multimodal_resume == {"schema": MING_THINKER_PD_RESUME_SCHEMA}
    assert continuation.stage_payload["data"] == {
        "prompt": {"input_ids": [20, 21, 22]},
        "stream_state": {"offset": 2},
    }
    destination = SimpleNamespace()
    restorer(destination, None, continuation.multimodal_resume)
    assert destination.tokenizer == "tokenizer"
    assert destination.omni_model_inputs is None


def test_ming_pd_state_rejects_unknown_schema() -> None:
    _, restorer = make_ming_pd_adapters(tokenizer=None)
    with pytest.raises(ValueError, match="unsupported Ming"):
        restorer(SimpleNamespace(), None, {"schema": "future"})
