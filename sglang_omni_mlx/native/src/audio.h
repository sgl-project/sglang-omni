// SPDX-License-Identifier: Apache-2.0
// Qwen3-ASR audio front end: WAV decoding, log-mel and audio token counts.
#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "mlx/mlx.h"

namespace qwen3_asr {

inline constexpr int kSampleRate = 16000;
inline constexpr int kHopLength = 160;
inline constexpr int kFftSize = 400;
inline constexpr int kMelBinCount = 128;
// Qwen3-ASR encodes mel frames in chunks of 100 frames, 13 audio tokens each.
inline constexpr int kChunkFrameCount = 100;
inline constexpr int kChunkTokenCount = 13;

// How the audio part of the prompt is built. kReference follows the
// checkpoint's processor; kVoxtSwift reproduces Voxt's Swift port (one more
// mel frame, and a token count computed with float32 true division).
enum class AudioLayout { kReference, kVoxtSwift };

// 16 kHz mono PCM16 or float32 WAV to float32 samples; throws
// std::invalid_argument for anything else.
std::vector<float> DecodeWav(std::string_view wav_bytes);

// Slaney-scale, Slaney-normalized filters [frequency_bins, mel_bins], built in
// float32 in the same order as Voxt's Swift front end.
const std::vector<float> &MelFilterBank(int mel_bin_count = kMelBinCount);
const std::vector<float> &PeriodicHannWindow();

// Whisper-style log-mel in float32 on MLX, [mel_bins, frames]. The reference
// layout drops the final centered STFT frame; the Swift layout keeps it.
mlx::core::array LogMel(const std::vector<float> &samples, AudioLayout layout);

// Whisper's fixed 30 s encoder window.
inline constexpr int kWhisperWindowSampleCount = 30 * kSampleRate;

// Whisper log-mel of one 30 s window, zero padded, [frames, mel_bins].
// Note (Dayuxiaoshui): the floor is taken per window, not over the recording,
// because the reference extracts each 30 s window on its own.
mlx::core::array WhisperWindowFeatures(const float *samples, int sample_count,
                                       int mel_bin_count);

int ConvOutputFrames(int frame_count);
int ReferenceTokenCount(int frame_count);
int SwiftTokenCount(int frame_count);
int TokenCount(int frame_count, AudioLayout layout);

bool PeakIsSilent(const float *samples, size_t sample_count,
                  float peak_threshold);

} // namespace qwen3_asr
