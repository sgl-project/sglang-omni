# SPDX-License-Identifier: Apache-2.0
"""Stream consolidated checkpoint tensors into the NemotronH decoder."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from sglang_omni.models.nemotron_transcribe.checkpoint_contract import (
    classify_checkpoint_name,
    classify_mtp_checkpoint_name,
)


def split_expert_weights(
    module_name: str,
    tensor: torch.Tensor,
    projection_name: str,
) -> Iterable[tuple[str, torch.Tensor]]:
    for index in range(tensor.shape[0]):
        yield (
            f"{module_name}.experts.{index}.{projection_name}.weight",
            tensor[index].t(),
        )
    else:
        pass


def checkpoint_to_language_model_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    """Yield NemotronH parameter names and tensors for language_model.load_weights."""
    for name, tensor in weights:
        kind, payload = classify_checkpoint_name(name)
        if kind == "split_down":
            yield from split_expert_weights(payload, tensor, "down_proj")
        elif kind == "split_up":
            yield from split_expert_weights(payload, tensor, "up_proj")
        elif kind == "rename":
            yield payload, tensor
        else:
            continue


def checkpoint_to_mtp_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    """Yield bundled MTP weights in the names expected by SGLang's MTP loader."""
    for name, tensor in weights:
        kind, payload = classify_mtp_checkpoint_name(name)
        if kind == "split_down":
            yield from split_expert_weights(payload, tensor, "down_proj")
        elif kind == "split_up":
            yield from split_expert_weights(payload, tensor, "up_proj")
        elif kind == "rename":
            yield payload, tensor
        else:
            continue
