# SPDX-License-Identifier: Apache-2.0
"""Talker projected prefill through the Omni prefill sidecar.

before_prefill composes the projected prompt rows and attaches them to the
sidecar; SGLang dispatches the forward, so the batch stays eligible for the
breakable prefill CUDA graph. Rows composed from the wrong slice produce
speech of exactly the right duration saying the wrong thing, which every
duration-based gate passes, so per-request slicing is pinned here.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.model_runner.prefill_inputs import get_omni_prefill_inputs
from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner


def _sched_req(**data_kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(data=SimpleNamespace(**data_kwargs))


def _projected_req(
    embeds: torch.Tensor, *, prefix_len: int, extend_len: int
) -> SimpleNamespace:
    return _sched_req(
        input_embeds_are_projected=True,
        prefill_input_embeds=embeds,
        req=SimpleNamespace(
            input_embeds=None,
            prefix_indices=list(range(prefix_len)),
            extend_range=SimpleNamespace(length=extend_len),
        ),
    )


def _unprojected_req(extend_len: int) -> SimpleNamespace:
    return _sched_req(
        input_embeds_are_projected=False,
        prefill_input_embeds=None,
        req=SimpleNamespace(
            input_embeds=None,
            prefix_indices=[],
            extend_range=SimpleNamespace(length=extend_len),
        ),
    )


def _forward_batch(num_tokens: int, *, input_embeds=None) -> SimpleNamespace:
    return SimpleNamespace(
        input_embeds=input_embeds,
        replace_embeds=None,
        input_ids=torch.zeros(num_tokens, dtype=torch.long),
        rids=["r0"],
    )


def _runner(dtype: torch.dtype = torch.float32) -> QwenTalkerModelRunner:
    runner = object.__new__(QwenTalkerModelRunner)
    runner.model = SimpleNamespace(activation_dtype=dtype)
    return runner


def test_before_prefill_attaches_projected_embeds_to_the_sidecar() -> None:
    embeds = torch.randn(10, 64)
    forward_batch = _forward_batch(10)

    _runner().before_prefill(
        forward_batch,
        schedule_batch=None,
        requests=[_projected_req(embeds, prefix_len=0, extend_len=10)],
    )

    payload = get_omni_prefill_inputs(forward_batch)
    assert payload is not None
    assert torch.equal(payload.input_embeds, embeds)
    assert payload.input_embeds_are_projected is True
    # The batch's own embeds field stays untouched: a batch carrying it is
    # refused by prefill-graph admission.
    assert forward_batch.input_embeds is None


def test_sidecar_composes_the_logical_rows_for_each_request() -> None:
    first = torch.randn(12, 64)
    second = torch.randn(9, 64)
    requests = [
        _projected_req(first, prefix_len=3, extend_len=5),
        _projected_req(second, prefix_len=2, extend_len=7),
    ]
    num_tokens = 5 + 7

    sidecar_batch = _forward_batch(num_tokens)
    _runner().before_prefill(sidecar_batch, schedule_batch=None, requests=requests)

    payload = get_omni_prefill_inputs(sidecar_batch)
    assert payload is not None
    assert payload.input_embeds.shape[0] == num_tokens
    assert torch.equal(payload.input_embeds, torch.cat((first[3:8], second[2:9])))


def test_sidecar_embeds_are_converted_to_the_model_dtype() -> None:
    forward_batch = _forward_batch(4)

    _runner(dtype=torch.bfloat16).before_prefill(
        forward_batch,
        schedule_batch=None,
        requests=[
            _projected_req(
                torch.randn(4, 64, dtype=torch.float32), prefix_len=0, extend_len=4
            )
        ],
    )

    payload = get_omni_prefill_inputs(forward_batch)
    assert payload is not None
    assert payload.input_embeds.dtype is torch.bfloat16


def test_mixed_projected_and_unprojected_batch_is_still_refused() -> None:
    requests = [
        _projected_req(torch.randn(4, 64), prefix_len=0, extend_len=4),
        _unprojected_req(4),
    ]

    with pytest.raises(RuntimeError, match="cannot be batched together"):
        _runner().before_prefill(
            _forward_batch(8), schedule_batch=None, requests=requests
        )


def test_batch_carrying_sglang_input_embeds_uses_the_standard_forward() -> None:
    forward_batch = _forward_batch(5, input_embeds=torch.randn(5, 64))
    requests = [_unprojected_req(5)]

    runner = _runner()
    runner.before_prefill(forward_batch, schedule_batch=None, requests=requests)
    assert get_omni_prefill_inputs(forward_batch) is None


def test_talker_forward_accepts_the_sidecar_kwargs() -> None:
    """SGLModelRunner passes omni_prefill_rids to every sidecar model."""
    from sglang_omni.models.qwen3_omni.components.talker import Qwen3OmniTalker

    params = inspect.signature(Qwen3OmniTalker.forward).parameters
    assert "omni_prefill_rids" in params
    assert "input_embeds_are_projected" in params
