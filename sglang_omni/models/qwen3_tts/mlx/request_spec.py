# SPDX-License-Identifier: Apache-2.0
"""Translate Omni's Qwen3-TTS request data into the MLX runner's spec.

The preprocessing stage currently assembles prompt embeddings with the Torch
model, so this converts them once at admission. Replacing that with MLX-native
prompt construction removes the conversion *and* the second copy of the model on
an Apple Silicon host; until then this keeps the scheduler path working.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx
import numpy as np

from .runner import Qwen3TTSRequestSpec
from .sampling import SamplingParams

logger = logging.getLogger(__name__)


def _to_mlx(value: Any) -> mx.array | None:
    """Convert a Torch tensor / NumPy array to an ``mx.array``."""
    if value is None:
        return None
    if isinstance(value, mx.array):
        return value
    if isinstance(value, np.ndarray):
        return mx.array(value)
    detach = getattr(value, "detach", None)
    if detach is None:
        return None
    tensor = detach().cpu()
    # MLX has no bfloat16 NumPy bridge, so promote then cast back.
    if str(tensor.dtype) == "torch.bfloat16":
        return mx.array(tensor.float().numpy()).astype(mx.bfloat16)
    return mx.array(tensor.numpy())


def _with_batch_dim(value: mx.array | None) -> mx.array | None:
    """Ensure ``[1, rows, hidden]``; preprocessing squeezes the batch away."""
    if value is None:
        return None
    if value.ndim == 2:
        return value[None, :, :]
    return value


def _trailing_rows(data: Any) -> Any:
    """The unconsumed trailing-text rows, as one tensor.

    Omni stores them in a chunked device queue; the MLX runner keeps its own
    cursor, so the whole remainder is taken at once.
    """
    queue = getattr(data, "pending_text_queue", None)
    if queue is None:
        return None
    rows = getattr(queue, "rows", None)
    if rows is None:
        return None
    cursor = int(getattr(queue, "cursor", 0) or 0)
    remaining = rows[cursor:] if cursor else rows
    chunks = getattr(queue, "_chunks", None)
    if chunks:
        import torch

        remaining = torch.cat([remaining, *chunks], dim=0)
    return remaining


def _semantic_params(data: Any) -> SamplingParams:
    return SamplingParams(
        temperature=float(getattr(data, "temperature", 0.9) or 0.0),
        top_k=int(getattr(data, "top_k", 50) or 0),
        top_p=float(getattr(data, "top_p", 1.0) or 1.0),
        repetition_penalty=float(getattr(data, "repetition_penalty", 1.0) or 1.0),
    )


def _subtalker_params(data: Any) -> SamplingParams:
    if not bool(getattr(data, "subtalker_dosample", True)):
        return SamplingParams(temperature=0.0)
    return SamplingParams(
        temperature=float(getattr(data, "subtalker_temperature", 0.9) or 0.0),
        top_k=int(getattr(data, "subtalker_top_k", 50) or 0),
        top_p=float(getattr(data, "subtalker_top_p", 1.0) or 1.0),
        repetition_penalty=1.0,
    )


def build_request_spec(data: Any) -> Qwen3TTSRequestSpec | None:
    """Build the runner's spec, or ``None`` if this request has no prompt yet."""
    # Not `a or b`: these are tensors, and truth-testing a multi-element
    # tensor raises.
    raw_prompt = getattr(data, "prefill_input_embeds", None)
    if raw_prompt is None:
        raw_prompt = getattr(data, "prompt_input_embeds", None)
    prompt = _with_batch_dim(_to_mlx(raw_prompt))
    if prompt is None:
        return None

    pad_embed = _with_batch_dim(_to_mlx(getattr(data, "tts_pad_embed", None)))
    if pad_embed is None:
        raise ValueError(
            "Qwen3-TTS MLX needs tts_pad_embed to continue the text stream "
            "after the prompt"
        )
    trailing = _with_batch_dim(_to_mlx(_trailing_rows(data)))

    seed = getattr(data, "semantic_sampling_seed", None)
    return Qwen3TTSRequestSpec(
        prompt_embeds=prompt,
        trailing_text_embeds=trailing,
        pad_embed=pad_embed,
        semantic=_semantic_params(data),
        subtalker=_subtalker_params(data),
        seed=int(seed) if seed is not None else None,
    )
