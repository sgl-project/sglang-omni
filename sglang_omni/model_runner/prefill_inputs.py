# SPDX-License-Identifier: Apache-2.0
"""Omni-owned conditioning passed to model forward after graph admission."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

_OMNI_PREFILL_INPUTS_ATTR = "_sglang_omni_prefill_inputs"


@dataclass(frozen=True)
class OmniPrefillInputs:
    """Model-dtype embeddings covering the current extend-window tokens."""

    input_embeds: torch.Tensor
    input_embeds_are_projected: bool | None = None
    input_deepstack_embeds: torch.Tensor | None = None


def attach_omni_prefill_inputs(
    forward_batch: ForwardBatch, prefill_inputs: OmniPrefillInputs
) -> None:
    """Attach conditioning without changing graph-admission inputs."""
    if forward_batch.replace_embeds is not None:
        raise RuntimeError(
            "OmniPrefillInputs conflicts with forward_batch.replace_embeds"
        )
    else:
        pass
    num_tokens = len(forward_batch.input_ids)
    if prefill_inputs.input_embeds.shape[0] != num_tokens:
        raise RuntimeError(
            "OmniPrefillInputs embeddings must cover the extend-window tokens: "
            f"embeds rows={prefill_inputs.input_embeds.shape[0]}, "
            f"batch tokens={num_tokens}"
        )
    else:
        pass
    input_deepstack_embeds = prefill_inputs.input_deepstack_embeds
    if input_deepstack_embeds is not None and (
        input_deepstack_embeds.ndim != 2
        or input_deepstack_embeds.shape[0] != num_tokens
        or input_deepstack_embeds.dtype != prefill_inputs.input_embeds.dtype
        or input_deepstack_embeds.device != prefill_inputs.input_embeds.device
    ):
        raise ValueError(
            "DeepStack inputs must be a matrix matching the embedding token rows, "
            "dtype and device"
        )
    else:
        pass
    setattr(forward_batch, _OMNI_PREFILL_INPUTS_ATTR, prefill_inputs)


def get_omni_prefill_inputs(
    forward_batch: ForwardBatch | None,
) -> OmniPrefillInputs | None:
    """Return the private Omni payload, or None when none is attached."""
    return getattr(forward_batch, _OMNI_PREFILL_INPUTS_ATTR, None)


def clear_omni_prefill_inputs(forward_batch: ForwardBatch | None) -> None:
    """Remove the private Omni payload, if present."""
    if hasattr(forward_batch, _OMNI_PREFILL_INPUTS_ATTR):
        delattr(forward_batch, _OMNI_PREFILL_INPUTS_ATTR)
    else:
        pass


__all__ = [
    "OmniPrefillInputs",
    "attach_omni_prefill_inputs",
    "clear_omni_prefill_inputs",
    "get_omni_prefill_inputs",
]
