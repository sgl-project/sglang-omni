// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "common/campplus.h"
#include "ming_tts/ming_tts_audio.h"
#include "ming_tts/ming_tts_request.h"

namespace ming_tts {

struct ReferenceConditioning {
  mlx::core::array latents;
  mlx::core::array speaker;
};

// Whether directory has CampPlus: campplus.safetensors, as the mlx-community
// checkpoints carry it, or the official checkpoint's campplus.onnx.
bool HasCamPlusWeights(const std::filesystem::path &directory);

class MingReferenceEncoder {
public:
  explicit MingReferenceEncoder(const std::filesystem::path &directory);
  ReferenceConditioning Encode(const ReferenceAudio &audio) const;
  // The acoustic patches Encode makes of sample_count 44.1 kHz samples.
  int PatchCount(size_t sample_count) const;

private:
  omni_common::CamPlusEncoder speaker_;
  MingAudioEncoder audio_;
  int alignment_samples_;
};

} // namespace ming_tts
