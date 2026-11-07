# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from sglang_omni.models.ming_omni.talker.audio_vae.istft import ISTFT, ISTFTHead
from sglang_omni.platforms.npu import overlap_add_by_shifts


def test_predict_spectrum_matches_pre_refactor_inline_math() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        head = ISTFTHead(dim=3, n_fft=8, hop_length=2)
        hidden = torch.randn(2, 5, 3)

    with torch.no_grad():
        head.out.weight[0].zero_()
        head.out.bias[0] = math.log(200.0)

    expected_projection = head.out(hidden).transpose(1, 2)
    magnitude, phase = expected_projection.chunk(2, dim=1)
    unclipped_magnitude = torch.exp(magnitude)
    expected_spectrum = torch.clip(unclipped_magnitude, max=1e2) * (
        torch.cos(phase) + 1j * torch.sin(phase)
    )

    spectrum, projection = head.predict_spectrum(hidden)

    assert bool((unclipped_magnitude > 1e2).any())
    assert projection.shape == expected_projection.shape
    assert projection.dtype == expected_projection.dtype
    assert spectrum.shape == expected_spectrum.shape
    assert spectrum.dtype == expected_spectrum.dtype
    assert torch.equal(projection, expected_projection)
    assert torch.equal(spectrum, expected_spectrum)


def test_overlap_add_components_masks_audio_and_window_envelope() -> None:
    istft = ISTFT(n_fft=8, hop_length=2, win_length=8)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1)
        spectrum = torch.complex(torch.randn(2, 5, 4), torch.randn(2, 5, 4))
    valid_frame_mask = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, False],
        ]
    )

    masked_spectrum = spectrum * valid_frame_mask.unsqueeze(1)
    inverse = torch.fft.irfft(masked_spectrum, istft.n_fft, dim=1, norm="backward")
    inverse = inverse * istft.window[None, :, None]
    output_size = (spectrum.shape[-1] - 1) * istft.hop_length + istft.win_length
    expected_numerator = F.fold(
        inverse,
        output_size=(1, output_size),
        kernel_size=(1, istft.win_length),
        stride=(1, istft.hop_length),
    )[:, 0, 0, :]
    window_frames = (
        istft.window.square()
        .expand(spectrum.shape[0], spectrum.shape[-1], -1)
        .transpose(1, 2)
    )
    window_frames = window_frames * valid_frame_mask.unsqueeze(1)
    expected_denominator = F.fold(
        window_frames,
        output_size=(1, output_size),
        kernel_size=(1, istft.win_length),
        stride=(1, istft.hop_length),
    )[:, 0, 0, :]

    numerator, denominator = istft.overlap_add_components(
        spectrum,
        valid_frame_mask=valid_frame_mask,
    )

    assert torch.equal(numerator, expected_numerator)
    assert torch.equal(denominator, expected_denominator)
    invalid_values = torch.full_like(spectrum, complex(1e6, -1e6))
    corrupted = torch.where(valid_frame_mask.unsqueeze(1), spectrum, invalid_values)
    corrupted_numerator, corrupted_denominator = istft.overlap_add_components(
        corrupted,
        valid_frame_mask=valid_frame_mask,
    )
    assert torch.equal(corrupted_numerator, numerator)
    assert torch.equal(corrupted_denominator, denominator)


@pytest.mark.parametrize(
    ("n_fft", "hop_length", "frame_count"),
    [
        (3528, 882, 6),
        (8, 3, 7),
    ],
)
def test_npu_overlap_add_matches_fold(
    n_fft: int, hop_length: int, frame_count: int
) -> None:
    """The Ascend slice-summed overlap-add has to reproduce fold sample for sample.

    Ming's n_fft=3528 faults the vector core in fold, so the NPU platform rebuilds
    the overlap-add from stride-aligned slice sums. The two parameter sets cover
    the production geometry and a window that is not a multiple of the hop, where
    the implementation pads the window tail.
    """
    istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2)
        frames = torch.randn(2, n_fft, frame_count)
        window_frames = torch.rand(1, n_fft, frame_count)
    output_size = (frame_count - 1) * hop_length + n_fft

    def folded(tensor: torch.Tensor) -> torch.Tensor:
        return F.fold(
            tensor.expand(2, -1, -1),
            output_size=(1, output_size),
            kernel_size=(1, istft.win_length),
            stride=(1, hop_length),
        )[:, 0, 0, :]

    numerator, denominator = overlap_add_by_shifts(
        frames,
        window_frames,
        hop_length=hop_length,
        win_length=istft.win_length,
    )

    torch.testing.assert_close(numerator, folded(frames), rtol=0, atol=0)
    torch.testing.assert_close(denominator, folded(window_frames), rtol=0, atol=0)
