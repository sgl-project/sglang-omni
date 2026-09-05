# SPDX-License-Identifier: Apache-2.0
"""The memoized chunked-causal whisper mask matches the remote code's loop."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers", exc_type=ImportError)

from sglang_omni.models.minicpm_o.components.audio_encoder import (  # noqa: E402
    _subsequent_chunk_mask,
    _subsequent_chunk_mask_cached,
)


def _reference(size: int, chunk_size: int) -> torch.Tensor:
    # verbatim from the checkpoint's modeling_minicpmo.subsequent_chunk_mask
    ret = torch.zeros(size, size, dtype=torch.bool)
    for i in range(size):
        ending = min((i // chunk_size + 1) * chunk_size, size)
        ret[i, :ending] = True
    return ret


@pytest.mark.parametrize(
    ("size", "chunk_size"),
    [(1, 1), (7, 3), (50, 50), (64, 7), (1500, 50), (1499, 50)],
)
def test_matches_reference_loop(size, chunk_size):
    mask = _subsequent_chunk_mask(size, chunk_size, torch.device("cpu"))
    assert mask.dtype == torch.bool
    assert torch.equal(mask, _reference(size, chunk_size))


def test_memoized_per_shape_and_device():
    _subsequent_chunk_mask_cached.cache_clear()
    first = _subsequent_chunk_mask(10, 4, torch.device("cpu"))
    again = _subsequent_chunk_mask(10, 4, torch.device("cpu"))
    other = _subsequent_chunk_mask(12, 4, torch.device("cpu"))
    assert first is again
    assert other is not first
    assert _subsequent_chunk_mask_cached.cache_info().hits == 1
