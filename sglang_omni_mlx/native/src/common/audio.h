// SPDX-License-Identifier: Apache-2.0
// Shared WAV decoding, mel filters, Whisper features and silence detection.
#pragma once

#include <cstdint>
#include <string>
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
// Mono PCM16 or float32 WAV at sample_rate to float32 samples; throws
// std::invalid_argument for anything else.
std::vector<float> DecodeWav(std::string_view wav_bytes,
                             int sample_rate = kSampleRate);
// Samples as 16-bit little-endian PCM, clipped to [-1, 1] and scaled by
// 32767 as the CUDA server's encode_pcm does.
std::string Pcm16Bytes(const std::vector<float> &samples);
// A mono 16-bit WAV file around PCM bytes.
std::string WavBytes(const std::string &pcm, int sample_rate);

// Slaney-scale, Slaney-normalized filters [frequency_bins, mel_bins], built in
// float32 in the same order as Voxt's Swift front end.
const std::vector<float> &MelFilterBank(int mel_bin_count = kMelBinCount);
const std::vector<float> &PeriodicHannWindow();
mlx::core::array SwiftPeriodicHannWindow();

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
