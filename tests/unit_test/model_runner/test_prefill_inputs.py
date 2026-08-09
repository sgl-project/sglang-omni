# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the private Omni prefill-input sidecar."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
    get_omni_prefill_inputs,
)


def _forward_batch(
    *,
    num_tokens: int = 4,
    replace_embeds=None,
) -> SimpleNamespace:
    return SimpleNamespace(
        input_embeds=None,
        replace_embeds=replace_embeds,
        mm_inputs=[object(), None],
        input_ids=torch.zeros(num_tokens, dtype=torch.long),
        batch_size=2,
    )


def _payload(*, rows: int = 4) -> OmniPrefillInputs:
    return OmniPrefillInputs(input_embeds=torch.zeros(rows, 8))


def test_attach_uses_private_sidecar_without_mutating_upstream_fields() -> None:
    forward_batch = _forward_batch()
    mm_inputs = forward_batch.mm_inputs
    payload = _payload()

    attach_omni_prefill_inputs(forward_batch, payload)

    assert forward_batch.input_embeds is None
    assert forward_batch.mm_inputs is mm_inputs
    assert get_omni_prefill_inputs(forward_batch) is payload


def test_attach_rejects_replace_embeds() -> None:
    forward_batch = _forward_batch(replace_embeds=torch.zeros(1, 8))

    with pytest.raises(RuntimeError, match="replace_embeds"):
        attach_omni_prefill_inputs(forward_batch, _payload())


def test_attach_rejects_token_row_mismatch() -> None:
    forward_batch = _forward_batch(num_tokens=5)

    with pytest.raises(RuntimeError, match="extend-window tokens"):
        attach_omni_prefill_inputs(forward_batch, _payload(rows=4))
