# SPDX-License-Identifier: Apache-2.0
"""Typed request-group metadata for atomically scheduled dLLM companions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class DllmCompanionSpec:
    """A physical companion row that must stay aligned with its primary row."""

    role: str
    input_ids: tuple[int, ...]
    left_pad_length: int = 0


@dataclass(frozen=True)
class DllmRequestGroupSpec:
    """Companion rows and algorithm parameters attached to one primary request."""

    companions: tuple[DllmCompanionSpec, ...]
    algorithm_args: Mapping[str, Any] = field(default_factory=dict)
    primary_left_pad_length: int = 0

    def validate(self, *, primary_input_length: int) -> None:
        """Validate invariants required by atomic batch-level execution."""
        if primary_input_length < 0:
            raise ValueError("primary input length must be non-negative")
        if not 0 <= self.primary_left_pad_length <= primary_input_length:
            raise ValueError(
                "dLLM primary left padding must fit inside the aligned row"
            )

        roles = ("conditional", *(companion.role for companion in self.companions))
        if any(not role or role == "conditional" for role in roles[1:]):
            raise ValueError("companion roles must be non-empty and non-primary")
        if len(set(roles)) != len(roles):
            raise ValueError("dLLM group roles must be unique")

        for companion in self.companions:
            if len(companion.input_ids) != primary_input_length:
                raise ValueError(
                    "dLLM companion rows must be physically aligned with the primary"
                )
            if not 0 <= companion.left_pad_length <= primary_input_length:
                raise ValueError(
                    "dLLM companion left padding must fit inside the aligned row"
                )


@dataclass(frozen=True)
class DllmGroupMember:
    """Scheduler-owned membership metadata stored on each physical request."""

    group_id: str
    role: str
    left_pad_length: int
    algorithm_args: Mapping[str, Any]


@dataclass(frozen=True)
class DllmForwardGroup:
    """Ordered batch metadata consumed by grouped dLLM algorithms."""

    group_id: str
    roles: tuple[str, ...]
    left_pad_lengths: tuple[int, ...]
    algorithm_args: Mapping[str, Any]


def align_cfg_request_group(
    *,
    mask_token_id: int | None,
    conditional_input_ids: tuple[int, ...],
    unconditional_input_ids: tuple[int, ...],
    no_image_input_ids: tuple[int, ...] | None = None,
    algorithm_args: Mapping[str, Any] | None = None,
    existing_left_pad_lengths: Mapping[str, int] | None = None,
) -> tuple[tuple[int, ...], DllmRequestGroupSpec]:
    """Align CFG branches once, where their physical requests are assembled."""
    branches = [
        ("conditional", tuple(conditional_input_ids)),
        ("unconditional", tuple(unconditional_input_ids)),
    ]
    if no_image_input_ids is not None:
        branches.append(("no_image", tuple(no_image_input_ids)))
    target_length = max(len(input_ids) for _, input_ids in branches)
    if any(len(input_ids) != target_length for _, input_ids in branches):
        if isinstance(mask_token_id, bool) or not isinstance(mask_token_id, int):
            raise ValueError("LLaDA2 tokenizer has no mask_token_id for CFG padding")

    existing = existing_left_pad_lengths or {}
    aligned: dict[str, tuple[int, ...]] = {}
    left_pad_lengths: dict[str, int] = {}
    for role, input_ids in branches:
        prior_pad = int(existing.get(role, 0))
        if not 0 <= prior_pad <= len(input_ids):
            raise ValueError(
                f"CFG {role} has invalid left-pad length "
                f"{prior_pad} for {len(input_ids)} input tokens"
            )
        added_pad = target_length - len(input_ids)
        aligned[role] = (int(mask_token_id),) * added_pad + input_ids
        left_pad_lengths[role] = prior_pad + added_pad

    companions = [
        DllmCompanionSpec(
            role="unconditional",
            input_ids=aligned["unconditional"],
            left_pad_length=left_pad_lengths["unconditional"],
        )
    ]
    if no_image_input_ids is not None:
        companions.append(
            DllmCompanionSpec(
                role="no_image",
                input_ids=aligned["no_image"],
                left_pad_length=left_pad_lengths["no_image"],
            )
        )
    spec = DllmRequestGroupSpec(
        companions=tuple(companions),
        algorithm_args=dict(algorithm_args or {}),
        primary_left_pad_length=left_pad_lengths["conditional"],
    )
    spec.validate(primary_input_length=target_length)
    return aligned["conditional"], spec


def apply_forward_group_padding(forward_batch: Any, group: DllmForwardGroup) -> None:
    """Attach host/device padding metadata and normalize grouped positions."""
    batch_size = int(forward_batch.batch_size)
    if not (len(group.roles) == len(group.left_pad_lengths) == batch_size):
        raise RuntimeError("CFG padding metadata must match batch size")
    if not forward_batch.forward_mode.is_extend():
        raise RuntimeError("CFG left-pad metadata requires an extend batch")

    query_lengths = torch.as_tensor(
        forward_batch.extend_seq_lens_cpu, dtype=torch.int64, device="cpu"
    )
    if query_lengths.numel() != batch_size:
        raise RuntimeError("CFG query-length metadata must match batch size")
    if bool(torch.any(query_lengths != query_lengths[0])):
        raise RuntimeError("CFG physical rows must have equal active-block lengths")

    left_pad_lengths = tuple(int(length) for length in group.left_pad_lengths)
    if any(length < 0 for length in left_pad_lengths):
        raise RuntimeError("CFG left-pad lengths must be non-negative")
    pad_tensor = torch.as_tensor(
        left_pad_lengths,
        dtype=forward_batch.seq_lens.dtype,
        device=forward_batch.seq_lens.device,
    )
    query_length = int(query_lengths[0])
    if forward_batch.positions.numel() != batch_size * query_length:
        raise RuntimeError("CFG position span does not match the aligned batch")
    positions = forward_batch.positions.view(batch_size, query_length)
    positions.sub_(pad_tensor.to(dtype=positions.dtype).unsqueeze(1)).clamp_min_(0)

    forward_batch.omni_dllm_group = group
    forward_batch.dllm_left_pad_lens_cpu = left_pad_lengths
    forward_batch.dllm_left_pad_lens = pad_tensor
