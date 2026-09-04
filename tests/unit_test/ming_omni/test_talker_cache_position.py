# SPDX-License-Identifier: Apache-2.0
"""Decode cache_position under transformers 5.x tensor-returning caches."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.ming_omni.talker.modeling_ming_omni_talker import (
    _decode_cache_position,
)


def test_raw_arange_rejects_tensor_bounds():
    """transformers 5.x StaticCache.get_seq_length() returns a 1-element
    tensor once the cache is non-empty; feeding it to arange is a TypeError.
    This is the #1114 failure mode the helper exists to prevent."""
    seen = torch.tensor([5])
    with pytest.raises(TypeError):
        torch.arange(seen, seen + 1, device="cpu")


@pytest.mark.parametrize("seen", [0, torch.tensor([5]), torch.tensor(7)])
def test_decode_cache_position_coerces_cache_length(seen):
    cache = SimpleNamespace(get_seq_length=lambda: seen)
    pos = _decode_cache_position(cache, 3, torch.device("cpu"))
    start = int(seen) if not isinstance(seen, int) else seen
    assert pos.tolist() == [start, start + 1, start + 2]
    assert pos.device.type == "cpu"
