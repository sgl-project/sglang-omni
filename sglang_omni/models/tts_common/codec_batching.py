# SPDX-License-Identifier: Apache-2.0
"""Waveform normalization and same-shape batch dispatch for TTS codecs."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import numpy as np
import torch

WaveformInput = torch.Tensor | np.ndarray
ItemT = TypeVar("ItemT")
KeyT = TypeVar("KeyT")
ResultT = TypeVar("ResultT")


def to_mono_3d(waveform: WaveformInput) -> torch.Tensor:
    """Normalize a tensor or ndarray waveform to mono ``[1, 1, L]``."""
    if isinstance(waveform, np.ndarray):
        wav = torch.from_numpy(np.ascontiguousarray(waveform))
    elif isinstance(waveform, torch.Tensor):
        wav = waveform
    else:
        raise TypeError(
            f"waveform must be Tensor or ndarray, got {type(waveform).__name__}"
        )

    if wav.ndim == 1:
        return wav.view(1, 1, -1)
    if wav.ndim == 2:
        return wav[:1].unsqueeze(0)
    if wav.ndim == 3:
        if wav.shape[1] != 1:
            raise ValueError(f"audio must be mono, got shape {tuple(wav.shape)}")
        return wav
    raise ValueError(f"waveform must be 1-, 2- or 3-D, got {wav.ndim}-D")


def run_bucketed_batch(
    items: list[ItemT],
    *,
    bucket_key_fn: Callable[[ItemT], KeyT],
    single_fn: Callable[[ItemT], ResultT],
    batch_fn: Callable[[list[ItemT]], list[ResultT]],
    error_label: str,
) -> list[ResultT]:
    if not items:
        return []
    if len(items) == 1:
        return [single_fn(items[0])]

    buckets: dict[KeyT, list[int]] = {}
    for i, item in enumerate(items):
        buckets.setdefault(bucket_key_fn(item), []).append(i)

    results: list[ResultT | None] = [None] * len(items)
    for indices in buckets.values():
        if len(indices) == 1:
            results[indices[0]] = single_fn(items[indices[0]])
            continue
        batch_results = batch_fn([items[i] for i in indices])
        for idx, result in zip(indices, batch_results):
            results[idx] = result

    out: list[ResultT] = []
    for i, result in enumerate(results):
        if result is None:
            raise RuntimeError(f"{error_label} did not produce result for item {i}")
        out.append(result)
    return out
