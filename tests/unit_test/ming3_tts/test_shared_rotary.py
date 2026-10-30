"""Packed rotation retains partial-head values and independent output storage."""

import pytest
import torch
from x_transformers.x_transformers import apply_rotary_pos_emb

from sglang_omni.modules.ming_audio.flow.rotary import apply_shared_rotary


@pytest.mark.parametrize("rotation_width", [8, 16])
@pytest.mark.parametrize("scale", [1.0, 0.5])
def test_shared_rotary_matches_native_and_preserves_input(
    rotation_width: int, scale: float
) -> None:
    torch.manual_seed(42)
    hidden_states = torch.randn(2, 4, 3, 16)
    before = hidden_states.clone()
    frequencies = torch.randn(2, 3, rotation_width)
    expected = apply_rotary_pos_emb(t=hidden_states, freqs=frequencies, scale=scale)
    actual = apply_shared_rotary(
        hidden_states,
        frequencies.cos(),
        frequencies.sin(),
        scale,
        materialize_output=False,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(hidden_states, before, rtol=0, atol=0)
    actual.zero_()
    torch.testing.assert_close(hidden_states, before, rtol=0, atol=0)
