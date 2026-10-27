// SPDX-License-Identifier: Apache-2.0
// Shared WAV decoding, mel filters, Whisper features and silence detection.
#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "mlx/mlx.h"

namespace omni_common {

inline constexpr int kSampleRate = 16000;
inline constexpr int kHopLength = 160;
inline constexpr int kFftSize = 400;
inline constexpr int kMelBinCount = 128;
// Log-mel floor and the dynamic range kept below the loudest bin.
inline constexpr float kLogMelFloor = 1e-10f;
inline constexpr float kLogMelDynamicRange = 8.0f;
// 16 kHz mono PCM16 or float32 WAV to float32 samples; throws
// std::invalid_argument for anything else.
std::vector<float> DecodeWav(std::string_view wav_bytes);

// Slaney-scale, Slaney-normalized filters [frequency_bins, mel_bins], built in
// float32 in the same order as Voxt's Swift front end.
const std::vector<float> &MelFilterBank(int mel_bin_count = kMelBinCount);
const std::vector<float> &PeriodicHannWindow();

// Whisper's fixed 30 s encoder window.
inline constexpr int kWhisperWindowSampleCount = 30 * kSampleRate;

// Whisper log-mel of one 30 s window, zero padded, [frames, mel_bins].
// Note (Dayuxiaoshui): the floor is taken per window, not over the recording,
// because the reference extracts each 30 s window on its own.
mlx::core::array WhisperWindowFeatures(const float *samples, int sample_count,
                                       int mel_bin_count);

bool PeakIsSilent(const float *samples, size_t sample_count,
                  float peak_threshold);

} // namespace omni_common
