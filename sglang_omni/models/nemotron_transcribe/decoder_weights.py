# SPDX-License-Identifier: Apache-2.0
"""Stream consolidated checkpoint tensors into the NemotronH decoder and its MTP draft."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import torch

from sglang_omni.models.nemotron_transcribe.checkpoint_contract import (
    classify_checkpoint_name,
    classify_mtp_checkpoint_name,
)


def checkpoint_to_language_model_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield NemotronH parameter names and tensors for language_model.load_weights."""
    for name, tensor in weights:
        yield from convert_classified_weight(*classify_checkpoint_name(name), tensor)


def checkpoint_to_mtp_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the bundled MTP tensors under the mtp. names SGLang's MTP loader expects."""
    for name, tensor in weights:
        yield from convert_classified_weight(
            *classify_mtp_checkpoint_name(name), tensor
        )


def convert_classified_weight(
    kind: str, payload: str, tensor: torch.Tensor
) -> Iterator[tuple[str, torch.Tensor]]:
    if kind == "split_down":
        for index in range(tensor.shape[0]):
            yield f"{payload}.experts.{index}.down_proj.weight", tensor[index].t()
    elif kind == "split_up":
        for index in range(tensor.shape[0]):
            yield f"{payload}.experts.{index}.up_proj.weight", tensor[index].t()
    elif kind == "rename":
        yield payload, tensor
    else:
        pass
