// SPDX-License-Identifier: Apache-2.0
// Kaldi-style fbank pieces CAM++, SenseVoice and Fun-ASR-Nano share: Kaldi's
// mel filters, the symmetric Hamming window, a log fbank on MLX, and its
// frames stacked at a low frame rate.
#pragma once

#include <vector>

#include "mlx/mlx.h"

namespace omni_common {

// Kaldi's mel_bin_count triangular mel filters on its 1127 ln(1 + f / 700)
// scale from 20 Hz, for 16 kHz audio and an fft_size-point FFT:
// [mel_bin_count, fft_size / 2 + 1] with the Nyquist bin zero.
std::vector<double> KaldiMelBanks(int fft_size, int mel_bin_count);

// The FFT size KaldiFbank pads a window of window_length samples to.
int PaddedFftSize(int window_length);

// The symmetric Hamming window of size samples, built in float.
std::vector<float> SymmetricHammingWindow(int size);

// Log mel energies [frames, mel_bins] of samples scaled to the 16-bit range:
// frames of window.size() samples every hop_length samples (none for shorter
// audio), each with its mean removed, preemphasized (the first sample taking
// itself as its predecessor) and windowed, zero padded to a power of two, and
// the power spectrum through mel_filters [fft_size / 2 + 1, mel_bins], floored
// at log_floor.
mlx::core::array KaldiFbank(const std::vector<float> &samples,
                            const std::vector<float> &window, int hop_length,
                            const mlx::core::array &mel_filters,
                            float log_floor);

// Fbank frames [frames, mel_bins] stacked lfr_m at a time every lfr_n,
// [ceil(frames / lfr_n), lfr_m * mel_bins]: stack i starts (lfr_m - 1) / 2
// frames before frame i * lfr_n, the first and last frames repeated past
// either end.
mlx::core::array LowFrameRateFrames(const mlx::core::array &fbank, int lfr_m,
                                    int lfr_n);

} // namespace omni_common
