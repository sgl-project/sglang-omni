# SPDX-License-Identifier: Apache-2.0
"""HiFT's conv epilogues over channels-last activations, its short time Fourier
transform and the inverse as Triton kernels. An epilogue writes the next conv's zero
padded input in one pass and equals the eager chain it replaces bit for bit, its Snake
CosyVoice's; the transforms match torch.stft and torch.istft to float32 rounding with
one fixed sum order per output, so a row's result does not follow its batch."""

from __future__ import annotations

import torch

from sglang_omni.models.fun_cosyvoice3.causal_conv import HAS_TRITON

# note (ratish): 256 threads a program, each one 16-byte vector of float32.
ELEMENTS_PER_PROGRAM = 1024
ELEMENTWISE_WARPS = 8
FRAMES_PER_PROGRAM = 128
OUTPUT_SAMPLES_PER_PROGRAM = 256

if HAS_TRITON:
    import triton
    import triton.language as tl
    from sglang.kernels.numerics import div_rn_f32, mul_rn_f32
    from triton.language.extra import libdevice

    # note (ratish): row counts change every step, one binary serves them all.
    @triton.jit(do_not_specialize=["rows", "padded_rows", "left_padding"])
    def conv_epilogue_kernel(
        x_ptr,
        bias_ptr,
        residual_ptr,
        alpha_ptr,
        inverse_alpha_ptr,
        sum_ptr,
        padded_ptr,
        rows,
        padded_rows,
        left_padding,
        CHANNELS: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr,
        STORE_SUM: tl.constexpr,
        HAS_SNAKE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        batch = tl.program_id(1).to(tl.int64)
        offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        if HAS_SNAKE:
            output_rows = padded_rows
            row_shift = left_padding
        else:
            output_rows = rows
            row_shift = 0
        mask = offsets < output_rows * CHANNELS
        row = offsets // CHANNELS - row_shift
        channel = offsets % CHANNELS
        is_row = mask & (row >= 0) & (row < rows)
        source_offsets = (batch * rows + row) * CHANNELS + channel
        x = tl.load(x_ptr + source_offsets, mask=is_row, other=0.0)
        if HAS_BIAS:
            x = x + tl.load(bias_ptr + channel, mask=is_row, other=0.0)
        else:
            pass
        if HAS_RESIDUAL:
            x = x + tl.load(residual_ptr + source_offsets, mask=is_row, other=0.0)
        else:
            pass
        if STORE_SUM:
            tl.store(sum_ptr + source_offsets, x, mask=is_row)
        else:
            pass
        if HAS_SNAKE:
            alpha = tl.load(alpha_ptr + channel, mask=is_row, other=1.0)
            inverse_alpha = tl.load(inverse_alpha_ptr + channel, mask=is_row, other=1.0)
            sine = libdevice.sin(mul_rn_f32(x, alpha))
            # note (ratish): each product rounds alone, as in the eager ops; an FMA
            # would not.
            out = tl.where(
                is_row, x + mul_rn_f32(inverse_alpha, mul_rn_f32(sine, sine)), 0.0
            )
            tl.store(
                padded_ptr + batch * padded_rows * CHANNELS + offsets, out, mask=mask
            )
        else:
            pass

    @triton.jit(do_not_specialize=["rows", "padded_rows", "left_padding"])
    def leaky_relu_padded_kernel(
        x_ptr,
        bias_ptr,
        padded_ptr,
        rows,
        padded_rows,
        left_padding,
        slope,
        CHANNELS: tl.constexpr,
        FACTOR: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        batch = tl.program_id(1).to(tl.int64)
        offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < padded_rows * CHANNELS
        upsampled_row = offsets // CHANNELS - left_padding
        channel = offsets % CHANNELS
        is_row = mask & (upsampled_row >= 0) & (upsampled_row < rows * FACTOR)
        source_offsets = (batch * rows + upsampled_row // FACTOR) * CHANNELS + channel
        x = tl.load(x_ptr + source_offsets, mask=is_row, other=0.0)
        if HAS_BIAS:
            x = x + tl.load(bias_ptr + channel, mask=is_row, other=0.0)
        else:
            pass
        out = tl.where(is_row, tl.where(x > 0, x, x * slope), 0.0)
        tl.store(padded_ptr + batch * padded_rows * CHANNELS + offsets, out, mask=mask)

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


def snake_padded(
    x: torch.Tensor,
    alpha: torch.Tensor,
    inverse_alpha: torch.Tensor,
    left_padding: int,
    right_padding: int,
    *,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    sum_out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Snake of x + bias + residual, summed in that order, between left_padding and
    right_padding zero rows. x, residual and sum_out: (batch, rows, channels) float32,
    contiguous; sum_out, when given, receives the sum. alpha and inverse_alpha:
    (channels,), the latter 1 / (alpha + 1e-9) as the module computes it."""
    batch, rows, channels = x.shape
    assert x.is_contiguous() and x.dtype == torch.float32
    assert residual is None or residual.is_contiguous()
    padded_rows = left_padding + rows + right_padding
    padded = x.new_empty(batch, padded_rows, channels)
    grid = (triton.cdiv(padded_rows * channels, ELEMENTS_PER_PROGRAM), batch)
    conv_epilogue_kernel[grid](
        x,
        bias,
        residual,
        alpha,
        inverse_alpha,
        sum_out,
        padded,
        rows,
        padded_rows,
        left_padding,
        CHANNELS=channels,
        HAS_BIAS=bias is not None,
        HAS_RESIDUAL=residual is not None,
        STORE_SUM=sum_out is not None,
        HAS_SNAKE=True,
        BLOCK=ELEMENTS_PER_PROGRAM,
        num_warps=ELEMENTWISE_WARPS,
    )
    return padded


def add_bias_and_residual(
    x: torch.Tensor, bias: torch.Tensor, residual: torch.Tensor
) -> torch.Tensor:
    """x + bias + residual in that order; x and residual: (batch, rows, channels)
    float32, contiguous."""
    batch, rows, channels = x.shape
    assert x.is_contiguous() and residual.is_contiguous() and x.dtype == torch.float32
    sum_out = torch.empty_like(x)
    grid = (triton.cdiv(rows * channels, ELEMENTS_PER_PROGRAM), batch)
    conv_epilogue_kernel[grid](
        x,
        bias,
        residual,
        None,
        None,
        sum_out,
        None,
        rows,
        rows,
        0,
        CHANNELS=channels,
        HAS_BIAS=True,
        HAS_RESIDUAL=True,
        STORE_SUM=True,
        HAS_SNAKE=False,
        BLOCK=ELEMENTS_PER_PROGRAM,
        num_warps=ELEMENTWISE_WARPS,
    )
    return sum_out


def leaky_relu_padded(
    x: torch.Tensor,
    slope: float,
    factor: int,
    left_padding: int,
    *,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """leaky_relu(x + bias) repeated factor times along the rows, as nearest upsampling
    does, after left_padding zero rows. x: (batch, rows, channels) float32, contiguous.
    """
    batch, rows, channels = x.shape
    assert x.is_contiguous() and x.dtype == torch.float32
    padded_rows = left_padding + rows * factor
    padded = x.new_empty(batch, padded_rows, channels)
    grid = (triton.cdiv(padded_rows * channels, ELEMENTS_PER_PROGRAM), batch)
    leaky_relu_padded_kernel[grid](
        x,
        bias,
        padded,
        rows,
        padded_rows,
        left_padding,
        slope,
        CHANNELS=channels,
        FACTOR=factor,
        HAS_BIAS=bias is not None,
        BLOCK=ELEMENTS_PER_PROGRAM,
        num_warps=ELEMENTWISE_WARPS,
    )
    return padded


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
