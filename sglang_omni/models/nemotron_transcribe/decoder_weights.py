# SPDX-License-Identifier: Apache-2.0
"""Stream consolidated checkpoint tensors into the NemotronH decoder."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from sglang_omni.models.nemotron_transcribe.checkpoint_contract import (
    classify_checkpoint_name,
)


def checkpoint_to_language_model_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    """Yield NemotronH parameter names and tensors for language_model.load_weights."""
    for name, tensor in weights:
        kind, payload = classify_checkpoint_name(name)
        if kind == "split_down":
            for index in range(tensor.shape[0]):
                yield f"{payload}.experts.{index}.down_proj.weight", tensor[index].t()
        elif kind == "split_up":
            for index in range(tensor.shape[0]):
                yield f"{payload}.experts.{index}.up_proj.weight", tensor[index].t()
        elif kind == "rename":
            yield payload, tensor
        else:
            continue
