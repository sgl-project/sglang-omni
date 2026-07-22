# SPDX-License-Identifier: Apache-2.0
"""Serialize Higgs actions into the issue #780 omni rollout contract."""

from __future__ import annotations

from typing import Any

import torch

OMNI_ROLLOUT_VERSION = 1


def build_omni_rollout_trace(
    delayed_codes: torch.Tensor,
    *,
    num_codebooks: int,
    codebook_vocab_size: int,
    delayed_logprobs: torch.Tensor,
    action_mask: torch.Tensor,
    model_family: str = "higgs_tts",
    stage: str = "tts_engine",
    stream_name: str = "higgs_codes",
) -> dict[str, Any]:
    """Validate and serialize aligned sampled actions, masks, and logprobs."""
    if delayed_codes.ndim != 2:
        raise ValueError(
            f"delayed_codes must be 2-D [L, N], got shape {tuple(delayed_codes.shape)}"
        )
    L, N = delayed_codes.shape
    if N != num_codebooks:
        raise ValueError(
            f"delayed_codes has {N} codebooks but num_codebooks={num_codebooks}"
        )

    if tuple(action_mask.shape) != (L, N):
        raise ValueError(
            f"action_mask shape {tuple(action_mask.shape)} != codes shape {(L, N)}"
        )
    if tuple(delayed_logprobs.shape) != (L, N):
        raise ValueError(
            f"delayed_logprobs shape {tuple(delayed_logprobs.shape)} != "
            f"codes shape {(L, N)}"
        )
    action_mask = action_mask.to(torch.bool)
    if delayed_codes.numel() and not bool(
        ((delayed_codes >= 0) & (delayed_codes < codebook_vocab_size)).all()
    ):
        raise ValueError("Higgs rollout action is outside the codebook vocabulary")
    action_logprobs = delayed_logprobs[action_mask]
    if action_logprobs.numel() and not bool(torch.isfinite(action_logprobs).all()):
        raise ValueError("non-finite policy logprob at a sampled action position")

    delayed_logprobs = torch.where(
        action_mask,
        delayed_logprobs.to(torch.float32),
        torch.zeros_like(delayed_logprobs, dtype=torch.float32),
    )
    stream: dict[str, Any] = {
        "name": stream_name,
        "stage": stage,
        "modality": "audio",
        "action_type": "discrete",
        "layout": "codebook_2d",
        "flatten_order": "time_major",
        "shape": [int(L), int(N)],
        "vocab_size": int(codebook_vocab_size),
        "actions": delayed_codes.to(torch.long).tolist(),
        "logprobs": delayed_logprobs.tolist(),
        "action_mask": action_mask.to(torch.int64).tolist(),
        "deterministic_mask": None,
        "channel_ids": list(range(N)),
        "channel_roles": [f"codebook_{channel}" for channel in range(N)],
    }

    return {
        "version": OMNI_ROLLOUT_VERSION,
        "model_family": model_family,
        "stages": [stage],
        "total_action_count": int(action_mask.sum().item()),
        "action_streams": [stream],
        "non_action_outputs": [],
    }


__all__ = ["OMNI_ROLLOUT_VERSION", "build_omni_rollout_trace"]
