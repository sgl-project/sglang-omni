# SPDX-License-Identifier: Apache-2.0
"""Realtime media helpers shared by the WebRTC prototype."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def mono_float32(audio: Any) -> np.ndarray:
    """Normalize arbitrary mono/stereo audio into mono float32 in [-1, 1]."""
    arr = np.asarray(audio)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim > 1:
        if arr.shape[0] <= arr.shape[-1]:
            arr = arr.mean(axis=0)
        else:
            arr = arr.mean(axis=1)

    if np.issubdtype(arr.dtype, np.integer):
        scale = max(abs(np.iinfo(arr.dtype).min), np.iinfo(arr.dtype).max)
        arr = arr.astype(np.float32) / float(scale)
    else:
        arr = arr.astype(np.float32, copy=False)

    return np.clip(arr, -1.0, 1.0)


def resample_linear(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """Resample 1D audio with linear interpolation."""
    if orig_sr == target_sr:
        return audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return audio.astype(np.float32, copy=False)

    duration = audio.shape[0] / float(orig_sr)
    new_len = max(int(round(duration * target_sr)), 1)
    old_idx = np.arange(audio.shape[0], dtype=np.float64)
    new_idx = np.linspace(0.0, audio.shape[0] - 1, num=new_len, dtype=np.float64)
    return np.interp(new_idx, old_idx, audio).astype(np.float32)


def resize_rgb_frame(
    frame_rgb: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Resize an HWC RGB frame to a bounded size on CPU."""
    image = Image.fromarray(frame_rgb.astype(np.uint8), mode="RGB")
    resized = image.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)
