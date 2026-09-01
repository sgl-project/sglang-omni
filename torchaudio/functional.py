from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def resample(
    waveform: torch.Tensor,
    orig_freq: int,
    new_freq: int,
    *_,
    **__,
) -> torch.Tensor:
    orig_freq = int(orig_freq)
    new_freq = int(new_freq)
    if orig_freq == new_freq:
        return waveform
    if waveform.numel() == 0:
        return waveform

    shape = waveform.shape
    flat = waveform.reshape(-1, shape[-1]).to(dtype=torch.float32)
    new_len = max(int(round(shape[-1] * float(new_freq) / float(orig_freq))), 1)
    out = F.interpolate(
        flat.unsqueeze(1),
        size=new_len,
        mode="linear",
        align_corners=False,
    ).squeeze(1)
    return out.reshape(*shape[:-1], new_len).to(device=waveform.device)


class _FunctionalNamespace:
    @staticmethod
    def _get_sinc_resample_kernel(
        orig_freq: int,
        new_freq: int,
        gcd: int | None = None,
        *_,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **__,
    ):
        gcd = math.gcd(int(orig_freq), int(new_freq)) if gcd is None else int(gcd)
        kernel = torch.empty(0, device=device, dtype=dtype or torch.float32)
        return kernel, gcd

    @staticmethod
    def _apply_sinc_resample_kernel(
        waveform: torch.Tensor,
        orig_freq: int,
        new_freq: int,
        *_,
        **__,
    ) -> torch.Tensor:
        return resample(waveform, orig_freq, new_freq)


functional = _FunctionalNamespace()

__all__ = ["functional", "resample"]
