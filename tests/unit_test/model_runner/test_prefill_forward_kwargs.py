# SPDX-License-Identifier: Apache-2.0
"""Tests for forwarding the private Omni prefill sidecar to the model."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
    get_omni_prefill_inputs,
)

sglang_model_runner = pytest.importorskip(
    "sglang_omni.model_runner.sglang_model_runner",
    reason="SGLang runtime dependencies are not installed",
)
SGLModelRunner = sglang_model_runner.SGLModelRunner


def _forward_batch() -> SimpleNamespace:
    return SimpleNamespace(
        input_embeds=None,
        replace_embeds=None,
        replace_positions=None,
        mm_inputs=[object(), None],
        input_ids=torch.zeros(4, dtype=torch.long),
        batch_size=2,
        rids=["request-a", "request-b"],
    )


def test_extend_forward_kwargs_bridges_sidecar_without_mutating_batch() -> None:
    runner = SGLModelRunner.__new__(SGLModelRunner)
    runner.support_pp = False
    runner.is_generation = True
    runner.dtype = torch.float32
    forward_batch = _forward_batch()
    mm_inputs = forward_batch.mm_inputs
    payload = OmniPrefillInputs(input_embeds=torch.zeros(4, 8))
    attach_omni_prefill_inputs(forward_batch, payload)

    kwargs = runner._extend_forward_kwargs(forward_batch, object())

    assert kwargs["input_embeds"] is payload.input_embeds
    assert kwargs["omni_prefill_rids"] is forward_batch.rids
    assert forward_batch.input_embeds is None
    assert forward_batch.mm_inputs is mm_inputs
    assert get_omni_prefill_inputs(forward_batch) is payload
    # Models that do not declare the flag must not receive it.
    assert "input_embeds_are_projected" not in kwargs


def test_extend_forward_kwargs_forwards_the_projected_flag_when_set() -> None:
    runner = SGLModelRunner.__new__(SGLModelRunner)
    runner.support_pp = False
    runner.is_generation = True
    runner.dtype = torch.float32
    forward_batch = _forward_batch()
    attach_omni_prefill_inputs(
        forward_batch,
        OmniPrefillInputs(
            input_embeds=torch.zeros(4, 8), input_embeds_are_projected=True
        ),
    )

    kwargs = runner._extend_forward_kwargs(forward_batch, object())

    assert kwargs["input_embeds_are_projected"] is True


def test_extend_forward_kwargs_rejects_a_sidecar_in_another_dtype() -> None:
    runner = SGLModelRunner.__new__(SGLModelRunner)
    runner.support_pp = False
    runner.is_generation = True
    runner.dtype = torch.bfloat16
    forward_batch = _forward_batch()
    attach_omni_prefill_inputs(
        forward_batch, OmniPrefillInputs(input_embeds=torch.zeros(4, 8))
    )

    with pytest.raises(RuntimeError, match="model dtype"):
        runner._extend_forward_kwargs(forward_batch, object())
