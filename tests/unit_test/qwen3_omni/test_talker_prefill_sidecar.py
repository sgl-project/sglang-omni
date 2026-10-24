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
from sglang_omni.scheduling.sglang_backend.request_data import EmbeddingSpan


def sched_req(**data_kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(
        data=SimpleNamespace(session_embedding_spans=[], **data_kwargs)
    )


def projected_req(
    embeds: torch.Tensor, *, prefix_len: int, extend_len: int
) -> SimpleNamespace:
    return sched_req(
        input_embeds_are_projected=True,
        prefill_input_embeds=embeds,
        req=SimpleNamespace(
            input_embeds=None,
            prefix_len=prefix_len,
            extend_len=extend_len,
        ),
    )


def unprojected_req(extend_len: int) -> SimpleNamespace:
    return sched_req(
        input_embeds_are_projected=False,
        prefill_input_embeds=None,
        req=SimpleNamespace(
            input_embeds=None,
            prefix_len=0,
            extend_len=extend_len,
        ),
    )


def make_forward_batch(
    num_tokens: int,
    *,
    input_embeds=None,
    rows: list[tuple[int, int]] | None = None,
) -> SimpleNamespace:
    spans = rows if rows is not None else [(0, num_tokens)]
    return SimpleNamespace(
        input_embeds=input_embeds,
        replace_embeds=None,
        input_ids=torch.zeros(num_tokens, dtype=torch.long),
        extend_prefix_lens_cpu=[prefix_len for prefix_len, _ in spans],
        extend_seq_lens_cpu=[extend_len for _, extend_len in spans],
        rids=["r0"],
    )


def make_runner(dtype: torch.dtype = torch.float32) -> QwenTalkerModelRunner:
    runner = object.__new__(QwenTalkerModelRunner)
    runner.model = SimpleNamespace(activation_dtype=dtype)
    return runner


def test_before_prefill_attaches_projected_embeds_to_the_sidecar() -> None:
    embeds = torch.randn(10, 64)
    forward_batch = make_forward_batch(10)

    make_runner().before_prefill(
        forward_batch,
        schedule_batch=None,
        requests=[projected_req(embeds, prefix_len=0, extend_len=10)],
    )

    payload = get_omni_prefill_inputs(forward_batch)
    assert payload is not None
    assert torch.equal(payload.input_embeds, embeds)
    assert payload.input_embeds_are_projected is True
    # The batch's own embeds field stays untouched: a batch carrying it is
    # refused by prefill-graph admission.
    assert forward_batch.input_embeds is None


def test_native_session_rows_follow_the_batch_span_not_the_request() -> None:
    """A native session unit's decode tail mixed into a prefill batch takes its one-token
    batch span from the spans, not the window from the request's last prefill."""
    span_rows = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    tail_request = SimpleNamespace(
        prefix_len=100,
        extend_len=4,
        extend_end=104,
        get_fill_ids=lambda: list(range(104)),
    )
    request = sched_req(
        input_embeds_are_projected=True,
        prefill_input_embeds=None,
        req=tail_request,
    )
    request.data.session_embedding_spans = [
        EmbeddingSpan(start=101, end=104, input_embeds=span_rows)
    ]
    runner = make_runner()
    runner.model.get_input_embeddings = lambda: (
        lambda token_ids: token_ids.to(torch.float32)[:, None].repeat(1, 4)
    )
    forward_batch = make_forward_batch(1, rows=[(103, 1)])

    runner.before_prefill(forward_batch, schedule_batch=None, requests=[request])

    payload = get_omni_prefill_inputs(forward_batch)
    assert payload is not None
    assert torch.equal(payload.input_embeds, span_rows[2:3])
    assert (tail_request.prefix_len, tail_request.extend_end) == (100, 104)


def test_before_prefill_preserves_unprojected_tensor_embeds() -> None:
    embeds = torch.randn(6, 64)
    request = sched_req(
        input_embeds_are_projected=False,
        prefill_input_embeds=embeds,
        req=SimpleNamespace(
            input_embeds=None,
            prefix_len=0,
            extend_len=6,
        ),
    )
    forward_batch = make_forward_batch(6)

    make_runner().before_prefill(
        forward_batch,
        schedule_batch=None,
        requests=[request],
    )

    payload = get_omni_prefill_inputs(forward_batch)
    assert payload is not None
    assert torch.equal(payload.input_embeds, embeds)
    assert payload.input_embeds_are_projected is False
    assert forward_batch.input_embeds is None


def test_sidecar_composes_the_logical_rows_for_each_request() -> None:
    first = torch.randn(12, 64)
    second = torch.randn(9, 64)
    requests = [
        projected_req(first, prefix_len=3, extend_len=5),
        projected_req(second, prefix_len=2, extend_len=7),
    ]
    num_tokens = 5 + 7

    sidecar_batch = make_forward_batch(num_tokens, rows=[(3, 5), (2, 7)])
    make_runner().before_prefill(sidecar_batch, schedule_batch=None, requests=requests)

    payload = get_omni_prefill_inputs(sidecar_batch)
    assert payload is not None
    assert payload.input_embeds.shape[0] == num_tokens
    assert torch.equal(payload.input_embeds, torch.cat((first[3:8], second[2:9])))


def test_sidecar_embeds_are_converted_to_the_model_dtype() -> None:
    forward_batch = make_forward_batch(4)

    make_runner(dtype=torch.bfloat16).before_prefill(
        forward_batch,
        schedule_batch=None,
        requests=[
            projected_req(
                torch.randn(4, 64, dtype=torch.float32), prefix_len=0, extend_len=4
            )
        ],
    )

    payload = get_omni_prefill_inputs(forward_batch)
    assert payload is not None
    assert payload.input_embeds.dtype is torch.bfloat16


def test_mixed_projected_and_unprojected_batch_is_still_refused() -> None:
    requests = [
        projected_req(torch.randn(4, 64), prefix_len=0, extend_len=4),
        unprojected_req(4),
    ]

    with pytest.raises(RuntimeError, match="cannot be batched together"):
        make_runner().before_prefill(
            make_forward_batch(8), schedule_batch=None, requests=requests
        )


def test_batch_carrying_sglang_input_embeds_uses_the_standard_forward() -> None:
    forward_batch = make_forward_batch(5, input_embeds=torch.randn(5, 64))
    requests = [unprojected_req(5)]

    runner = make_runner()
    runner.before_prefill(forward_batch, schedule_batch=None, requests=requests)
    assert get_omni_prefill_inputs(forward_batch) is None


def test_talker_forward_accepts_the_sidecar_kwargs() -> None:
    """SGLModelRunner passes omni_prefill_rids to every sidecar model."""
    from sglang_omni.models.qwen3_omni.components.talker import Qwen3OmniTalker

    params = inspect.signature(Qwen3OmniTalker.forward).parameters
    assert "omni_prefill_rids" in params
    assert "input_embeds_are_projected" in params
