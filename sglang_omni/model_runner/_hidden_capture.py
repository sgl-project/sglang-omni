# SPDX-License-Identifier: Apache-2.0
"""Packed hidden-state capture helpers.

Speech capture publishes the requested layers through
``LogitsProcessorOutput.hidden_states``: the model packs
the captured layers side by side along the last axis
(see ``Qwen3OmniThinkerForCausalLM.process_hidden_states``) so the tensor rides
CUDA-graph replay as a single ordinary output. This module holds the matching
unpack helper shared by the model runner and the output processor.
"""

from __future__ import annotations

import torch


def unpack_packed_hidden_capture(
    packed: torch.Tensor | None,
    *,
    capture_layer_count: int,
    hidden_size: int,
) -> tuple[torch.Tensor, ...] | None:
    """Split packed captured layers along the last axis.

    ``packed`` is None on steps that captured nothing (NULL capture mode);
    any other width than ``hidden_size * capture_layer_count`` means the
    capture configuration and the model disagree — fail loud instead of
    silently dropping speech hidden states.
    """
    if packed is None:
        return None
    assert packed.shape[-1] == hidden_size * capture_layer_count, (
        f"packed hidden capture width {packed.shape[-1]} != "
        f"{hidden_size} * {capture_layer_count} "
        "(hidden_size * capture layers)"
    )
    return tuple(packed.split(hidden_size, dim=-1))
