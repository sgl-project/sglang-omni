// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ming_tts/ming_tts_audio.h"

namespace ming_tts {

// The sample rate CampPlus embeds.
inline constexpr int kSpeakerSampleRate = 16000;

// A reference recording at the rates its two encoders take: 16 kHz for
// CampPlus and AudioVAE's 44.1 kHz.
struct ReferenceAudio {
  std::vector<float> samples_16k;
  std::vector<float> samples_44k;
};

struct ReferenceConditioning {
  mlx::core::array latents;
  mlx::core::array speaker;
};

class CampPlus {
public:
  explicit CampPlus(const std::filesystem::path &path);
  mlx::core::array Encode(const std::vector<float> &samples) const;

private:
  mlx::core::array BatchNorm(const mlx::core::array &hidden,
                             const std::string &prefix) const;
  mlx::core::array Conv1d(const mlx::core::array &hidden,
                          const std::string &weight, const std::string &bias,
                          int stride = 1, int dilation = 1) const;
  omni_common::Checkpoint speaker_;
};

class MingReferenceEncoder {
public:
  explicit MingReferenceEncoder(const std::filesystem::path &directory);
  ReferenceConditioning Encode(const ReferenceAudio &audio) const;
  // The acoustic patches Encode makes of sample_count 44.1 kHz samples.
  int PatchCount(size_t sample_count) const;

private:
  CampPlus speaker_;
  MingAudioEncoder audio_;
  int alignment_samples_;
};

} // namespace ming_tts
