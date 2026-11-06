# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni thinker requests under a speculative draft's verify."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang_omni.models.qwen3_omni import request_builders
from tests.unit_test.fixtures.qwen_fakes import make_qwen_payload


def build_requests(
    monkeypatch: pytest.MonkeyPatch,
    *,
    is_speculative: bool,
    multimodal_inputs: list[SimpleNamespace | None],
) -> list[SimpleNamespace]:
    """Run the thinker's request builder over one request per multimodal input."""
    built = iter(multimodal_inputs)
    monkeypatch.setattr(
        request_builders,
        "build_sglang_thinker_request",
        lambda state, **kwargs: SimpleNamespace(
            req=SimpleNamespace(multimodal_inputs=next(built))
        ),
    )
    request_builder, _ = request_builders.make_thinker_scheduler_adapters(
        tokenizer=None,
        vocab_size=256,
        capture_prompt_hidden_states=False,
        is_speculative=is_speculative,
    )
    return [
        request_builder(make_qwen_payload(request_id=f"request-{index}")).req
        for index in range(len(multimodal_inputs))
    ]


def test_verify_positions_of_a_mixed_batch_stay_integer(monkeypatch) -> None:
    """compute_mrope_positions floors the positions but can return a float delta, and
    SGLang's verify stacks a multimodal request's delta with a text request's int64
    zero before adding it to the integer positions."""
    multimodal_request, text_request = build_requests(
        monkeypatch,
        is_speculative=True,
        multimodal_inputs=[
            SimpleNamespace(mrope_position_delta=torch.tensor([[-3.0]])),
            None,
        ],
    )
    delta = multimodal_request.multimodal_inputs.mrope_position_delta
    forward_batch = SimpleNamespace(seq_lens=torch.tensor([5, 9]))

    ForwardBatch.compute_spec_mrope_positions(
        forward_batch,
        SimpleNamespace(device="cpu"),
        SimpleNamespace(
            multimodal_inputs=[
                multimodal_request.multimodal_inputs,
                text_request.multimodal_inputs,
            ],
            spec_info=SimpleNamespace(positions=torch.tensor([5, 6, 7, 9, 10, 11])),
        ),
    )

    assert delta.dtype == torch.int64
    assert delta.tolist() == [[-3]]
    assert text_request.multimodal_inputs is None
    assert forward_batch.mrope_positions.dtype == torch.int64
    assert torch.equal(
        forward_batch.mrope_positions,
        torch.tensor([[2, 3, 4, 9, 10, 11]]).repeat(3, 1),
    )


def test_a_thinker_without_a_draft_keeps_the_delta_as_computed(monkeypatch) -> None:
    (request,) = build_requests(
        monkeypatch,
        is_speculative=False,
        multimodal_inputs=[
            SimpleNamespace(mrope_position_delta=torch.tensor([[-3.0]])),
        ],
    )

    assert request.multimodal_inputs.mrope_position_delta.dtype == torch.float32
