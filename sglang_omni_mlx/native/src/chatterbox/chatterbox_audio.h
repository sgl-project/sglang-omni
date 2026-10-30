// SPDX-License-Identifier: Apache-2.0
// Chatterbox-Turbo's voice-prompt audio front end: WAV decoding, polyphase
// resampling, BS.1770 loudness normalization and the flow's prompt mel.
#pragma once

#include <complex>
#include <cstddef>
#include <string_view>
#include <vector>

#include "mlx/mlx.h"

namespace chatterbox {

// A WAV file held in memory (integer PCM of 8, 16, 24 or 32 bits, IEEE float
// of 32 or 64 bits, plain or WAVE_FORMAT_EXTENSIBLE, mono or stereo) as mono
// float32 samples at sample_rate. Samples pass through PCM16 as a 16-bit
// decoder would, stereo is averaged, and other rates are resampled with
// Resample. Throws std::invalid_argument for anything else.
std::vector<float> ReadAudio(std::string_view file_bytes, int sample_rate);
// The duration of the samples ReadAudio decodes from a WAV file at
// sample_rate, read from its header alone; throws std::invalid_argument when
// ReadAudio would.
double WavDurationSeconds(std::string_view file_bytes, int sample_rate);

// Polyphase resampling through a Kaiser-windowed sinc low-pass (64 zero
// crossings) with the signal's edge values extended past both ends, filtered
// in double and rounded to float32.
std::vector<float> Resample(const std::vector<float> &samples, int from_rate,
                            int to_rate);

// Scales samples so their BS.1770 integrated loudness is target_lufs. A clip
// shorter than one 400 ms gating block, or one with no block above the gates
// (silence), is returned unchanged.
std::vector<float> NormalizeLoudness(std::vector<float> samples,
                                     int sample_rate, double target_lufs);

// The flow's prompt mel of 24 kHz samples: 80 Slaney mel bins of a 1920-point
// magnitude STFT every 480 samples, natural log, float32 [1, frames, 80].
// Throws std::invalid_argument below one hop of samples.
mlx::core::array S3GenMel(const std::vector<float> &samples_24k);

// The first size / 2 + 1 bins of the DFT of real samples, in double.
std::vector<std::complex<double>> RealFft(const std::vector<double> &samples);

// A double sum in blocks of eight that halves above 128 values, the order a
// vectorized float64 reduction adds in.
double PairwiseSum(const double *values, size_t count);

} // namespace chatterbox
