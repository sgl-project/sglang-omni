// SPDX-License-Identifier: Apache-2.0
// Fun-CosyVoice3's S3 speech tokenizer v3 (25 tokens per second): Whisper's
// 128-band log-mel, a twelve-block FSMN attention encoder and a finite scalar
// quantizer.
#pragma once

#include <vector>

#include "common/layers.h"
#include "mlx/mlx.h"

namespace cosyvoice3 {

class SpeechTokenizer {
public:
  // Reads the S3TokenizerV3 checkpoint's encoder.* and quantizer.* weights.
  explicit SpeechTokenizer(const omni_common::Checkpoint &checkpoint);

  // All speech tokens of up to 30 s of 16 kHz samples, int32 [1, tokens].
  mlx::core::array Encode(const std::vector<float> &samples_16k) const;

private:
  mlx::core::array Block(const mlx::core::array &x, int layer,
                         const mlx::core::array &attention_mask) const;

  const omni_common::Checkpoint &checkpoint_;
  // Rotary tables [positions, head dim].
  mlx::core::array rotary_cos_;
  mlx::core::array rotary_sin_;
};

} // namespace cosyvoice3
