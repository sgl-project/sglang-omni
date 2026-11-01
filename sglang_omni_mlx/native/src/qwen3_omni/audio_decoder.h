// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "qwen3_omni/model.h"
namespace qwen3_omni {
struct DecodedAudio {
  std::vector<float> samples;
  int sample_rate;
};
DecodedAudio decode_audio_bytes(const std::vector<uint8_t> &bytes,
                                const Cancellation &cancelled);
} // namespace qwen3_omni
