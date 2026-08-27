# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from sglang_omni.models.qwen3_omni.components.talker import Qwen3OmniMoeTalkerTextModel


def make_fake_model(
    hidden: int, max_batch_size: int = 4
) -> Qwen3OmniMoeTalkerTextModel:
    model = object.__new__(Qwen3OmniMoeTalkerTextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=hidden)
    model.codec_embedding = nn.Embedding(2, hidden)
    model._init_feedback_state(max_batch_size=max_batch_size)
    return model


def test_feedback_slots_sized_for_the_reserved_pool_row() -> None:
    model = make_fake_model(hidden=16, max_batch_size=8)

    assert model._feedback_slots.shape == (9, 16)
    assert model._feedback_slots.dtype == model._feedback_buffer.dtype
    assert model._feedback_slots.device == model._feedback_buffer.device
    assert torch.equal(model._feedback_slots, torch.zeros(9, 16))


def test_feedback_batch_buffers_keep_batch_dim() -> None:
    model = make_fake_model(hidden=16, max_batch_size=4)

    assert model._feedback_buffer.shape == (4, 16)
    assert model._feedback_mask.shape == (4,)
    assert model._feedback_mask.dtype == torch.bool


def test_feedback_slots_cover_max_req_pool_idx() -> None:
    max_running_requests = 4
    model = make_fake_model(hidden=16, max_batch_size=max_running_requests)

    row = torch.ones(16, dtype=model._feedback_slots.dtype)
    model._feedback_slots[max_running_requests] = row

    assert torch.equal(model._feedback_slots[max_running_requests], row)
