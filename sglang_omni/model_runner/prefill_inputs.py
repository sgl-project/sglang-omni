# SPDX-License-Identifier: Apache-2.0
"""Omni-owned prefill inputs for breakable prefill CUDA graphs.

Runner-composed embeddings must remain off ForwardBatch.input_embeds
until after prefill-CUDA-graph admission because upstream rejects batches
carrying that field. They also must not reuse ForwardBatch.mm_inputs,
whose upstream contract belongs to multimodal request inputs. This module
therefore attaches the embeddings to a private, short-lived Omni sidecar.
SGLModelRunner._extend_forward_kwargs moves them into explicit model
forward kwargs after admission, and the Omni runner clears the sidecar after
the forward completes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

_OMNI_PREFILL_INPUTS_ATTR = "_sglang_omni_prefill_inputs"


@dataclass(frozen=True)
class OmniPrefillInputs:
    """Per-forward prefill conditioning composed by an omni model runner.

    input_embeds covers exactly the extend-window tokens of the batch, in
    model dtype. Request identity remains owned by ForwardBatch.rids and is
    forwarded separately by SGLModelRunner.

    input_embeds_are_projected stays None for models whose forward does
    not declare the parameter; only a runner composing embeddings already in
    model space sets it, and only then is it forwarded.
    """

    input_embeds: torch.Tensor
    input_embeds_are_projected: bool | None = None


def attach_omni_prefill_inputs(
    forward_batch: Any, prefill_inputs: OmniPrefillInputs
) -> None:
    """Attach prefill_inputs without modifying upstream-owned fields."""
    if forward_batch.replace_embeds is not None:
        raise RuntimeError(
            "OmniPrefillInputs conflicts with forward_batch.replace_embeds"
        )
    num_tokens = len(forward_batch.input_ids)
    if prefill_inputs.input_embeds.shape[0] != num_tokens:
        raise RuntimeError(
            "OmniPrefillInputs embeddings must cover the extend-window tokens: "
            f"embeds rows={prefill_inputs.input_embeds.shape[0]}, "
            f"batch tokens={num_tokens}"
        )
    setattr(forward_batch, _OMNI_PREFILL_INPUTS_ATTR, prefill_inputs)


def get_omni_prefill_inputs(forward_batch: Any) -> OmniPrefillInputs | None:
    """Return the private Omni payload, or None when none is attached."""
    return getattr(forward_batch, _OMNI_PREFILL_INPUTS_ATTR, None)


def clear_omni_prefill_inputs(forward_batch: Any) -> None:
    """Remove the private Omni payload, if present."""
    if hasattr(forward_batch, _OMNI_PREFILL_INPUTS_ATTR):
        delattr(forward_batch, _OMNI_PREFILL_INPUTS_ATTR)


__all__ = [
    "OmniPrefillInputs",
    "attach_omni_prefill_inputs",
    "clear_omni_prefill_inputs",
    "get_omni_prefill_inputs",
]
