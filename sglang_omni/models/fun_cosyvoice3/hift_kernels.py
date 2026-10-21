# SPDX-License-Identifier: Apache-2.0
"""HiFT's Snake, short time Fourier transform and its inverse as Triton kernels. Snake equals
CosyVoice's Snake bit for bit; the transforms match torch.stft and torch.istft to float32
rounding with one fixed sum order per output, so a row's result does not follow its batch."""

from __future__ import annotations

import torch

from sglang_omni.models.fun_cosyvoice3.causal_conv import HAS_TRITON

SAMPLES_PER_PROGRAM = 1024
FRAMES_PER_PROGRAM = 128
OUTPUT_SAMPLES_PER_PROGRAM = 256

if HAS_TRITON:
    import triton
    import triton.language as tl
    from sglang.kernels.numerics import div_rn_f32, mul_rn_f32
    from triton.language.extra import libdevice

    # note (ratish): sample counts change every step, one binary serves them all.
    @triton.jit(do_not_specialize=["samples", "channels"])
    def snake_kernel(
        x_ptr,
        alpha_ptr,
        inverse_alpha_ptr,
        out_ptr,
        samples,
        channels,
        BLOCK_SAMPLES: tl.constexpr,
    ):
        channel = tl.program_id(1)
        row = tl.program_id(2).to(tl.int64) * channels + channel
        sample_indices = tl.program_id(0) * BLOCK_SAMPLES + tl.arange(0, BLOCK_SAMPLES)
        is_sample = sample_indices < samples
        alpha = tl.load(alpha_ptr + channel)
        inverse_alpha = tl.load(inverse_alpha_ptr + channel)
        x = tl.load(x_ptr + row * samples + sample_indices, mask=is_sample)
        sine = libdevice.sin(mul_rn_f32(x, alpha))
        # note (ratish): each product rounds alone, as in the eager ops; an FMA would not.
        out = x + mul_rn_f32(inverse_alpha, mul_rn_f32(sine, sine))
        tl.store(out_ptr + row * samples + sample_indices, out, mask=is_sample)

    @triton.jit(do_not_specialize=["frames", "padded_samples"])
    def short_time_spectrum_kernel(
        padded_signal_ptr,
        basis_ptr,
        spectrum_ptr,
        frames,
        padded_samples,
        N_FFT: tl.constexpr,
        HOP_LENGTH: tl.constexpr,
        BINS: tl.constexpr,
        BLOCK_FRAMES: tl.constexpr,
    ):
        row = tl.program_id(1).to(tl.int64)
        frame_indices = tl.program_id(0) * BLOCK_FRAMES + tl.arange(0, BLOCK_FRAMES)
        is_frame = frame_indices < frames
        taps = tl.arange(0, N_FFT)
        frame_samples = tl.load(
            padded_signal_ptr
            + row * padded_samples
            + frame_indices[:, None] * HOP_LENGTH
            + taps[None, :],
            mask=is_frame[:, None],
            other=0.0,
        )
        for component in tl.static_range(2 * BINS):
            basis = tl.load(basis_ptr + component * N_FFT + taps)
            tl.store(
                spectrum_ptr + (row * 2 * BINS + component) * frames + frame_indices,
                tl.sum(frame_samples * basis[None, :], axis=1),
                mask=is_frame,
            )

    @triton.jit(do_not_specialize=["frames", "samples"])
    def overlap_add_inverse_kernel(
        real_ptr,
        imaginary_ptr,
        basis_ptr,
        squared_window_ptr,
        out_ptr,
        frames,
        samples,
        N_FFT: tl.constexpr,
        HOP_LENGTH: tl.constexpr,
        BINS: tl.constexpr,
        BLOCK_SAMPLES: tl.constexpr,
    ):
        row = tl.program_id(1).to(tl.int64)
        sample_indices = tl.program_id(0) * BLOCK_SAMPLES + tl.arange(0, BLOCK_SAMPLES)
        is_sample = sample_indices < samples
        padded_positions = sample_indices + N_FFT // 2
        signal = tl.zeros((BLOCK_SAMPLES,), dtype=tl.float32)
        envelope = tl.zeros((BLOCK_SAMPLES,), dtype=tl.float32)
        for overlap in tl.static_range(N_FFT // HOP_LENGTH):
            frame_indices = padded_positions // HOP_LENGTH - overlap
            tap_indices = padded_positions - frame_indices * HOP_LENGTH
            is_frame = is_sample & (frame_indices >= 0) & (frame_indices < frames)
            # note (ratish): one bin at a time, so each output sums in the same order
            # whatever the tile.
            for bin_index in tl.static_range(BINS):
                spectrum_offsets = (row * BINS + bin_index) * frames + frame_indices
                real = tl.load(real_ptr + spectrum_offsets, mask=is_frame, other=0.0)
                imaginary = tl.load(
                    imaginary_ptr + spectrum_offsets, mask=is_frame, other=0.0
                )
                cosine = tl.load(basis_ptr + bin_index * N_FFT + tap_indices)
                sine = tl.load(basis_ptr + (BINS + bin_index) * N_FFT + tap_indices)
                signal += real * cosine + imaginary * sine
            envelope += tl.load(
                squared_window_ptr + tap_indices, mask=is_frame, other=0.0
            )
        # note (ratish): torch.istft divides correctly rounded; Triton's / is approximate.
        tl.store(
            out_ptr + row * samples + sample_indices,
            div_rn_f32(signal, envelope),
            mask=is_sample,
        )

else:
    pass


def snake(
    x: torch.Tensor, alpha: torch.Tensor, inverse_alpha: torch.Tensor
) -> torch.Tensor:
    """x: (batch, channels, samples) float32; alpha and inverse_alpha: (channels,), the
    latter 1 / (alpha + 1e-9) as the module computes it."""
    batch, channels, samples = x.shape
    assert x.is_contiguous() and x.dtype == torch.float32
    out = torch.empty_like(x)
    grid = (triton.cdiv(samples, SAMPLES_PER_PROGRAM), channels, batch)
    snake_kernel[grid](
        x,
        alpha,
        inverse_alpha,
        out,
        samples,
        channels,
        BLOCK_SAMPLES=SAMPLES_PER_PROGRAM,
    )
    return out


def short_time_spectrum(
    padded_signal: torch.Tensor, basis: torch.Tensor, hop_length: int
) -> torch.Tensor:
    """padded_signal: (rows, samples) float32, reflect padded; basis: (2 * bins, n_fft),
    the windowed cosines then the windowed negative sines. Returns (rows, 2 * bins,
    frames), the real parts then the imaginary parts."""
    rows, padded_samples = padded_signal.shape
    components, n_fft = basis.shape
    assert padded_signal.is_contiguous() and padded_signal.dtype == torch.float32
    frames = (padded_samples - n_fft) // hop_length + 1
    spectrum = padded_signal.new_empty(rows, components, frames)
    grid = (triton.cdiv(frames, FRAMES_PER_PROGRAM), rows)
    short_time_spectrum_kernel[grid](
        padded_signal,
        basis,
        spectrum,
        frames,
        padded_samples,
        N_FFT=n_fft,
        HOP_LENGTH=hop_length,
        BINS=components // 2,
        BLOCK_FRAMES=FRAMES_PER_PROGRAM,
    )
    return spectrum


def overlap_add_inverse(
    real: torch.Tensor,
    imaginary: torch.Tensor,
    basis: torch.Tensor,
    squared_window: torch.Tensor,
    hop_length: int,
) -> torch.Tensor:
    """torch.istft with center=True over real, imaginary: (rows, bins, frames) float32;
    basis: (2 * bins, n_fft), the one sided inverse's windowed cosines then negative
    sines over n_fft. Returns (rows, hop_length * (frames - 1))."""
    rows, bins, frames = real.shape
    components, n_fft = basis.shape
    assert real.is_contiguous() and imaginary.is_contiguous() and components == 2 * bins
    assert real.dtype == imaginary.dtype == torch.float32
    samples = hop_length * (frames - 1)
    out = real.new_empty(rows, samples)
    grid = (triton.cdiv(samples, OUTPUT_SAMPLES_PER_PROGRAM), rows)
    overlap_add_inverse_kernel[grid](
        real,
        imaginary,
        basis,
        squared_window,
        out,
        frames,
        samples,
        N_FFT=n_fft,
        HOP_LENGTH=hop_length,
        BINS=bins,
        BLOCK_SAMPLES=OUTPUT_SAMPLES_PER_PROGRAM,
    )
    return out
