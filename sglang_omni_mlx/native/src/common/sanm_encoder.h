// SPDX-License-Identifier: Apache-2.0
// The SAN-M encoder SenseVoice and Fun-ASR-Nano share: sinusoidal positions
// and pre-norm self-attention layers with an FSMN memory, under SenseVoice's
// weight names.
#pragma once

#include "common/layers.h"
#include "mlx/mlx.h"

namespace omni_common {

struct SanmEncoderConfig {
  // Width of the stacked features and of the first layer's input.
  int input_size = 0;
  int model_width = 0;
  int head_count = 0;
  // The first layer plus the main stack, then the stack after after_norm.
  int block_count = 0;
  int tp_block_count = 0;
  int kernel_size = 0;
  int sanm_shift = 0;
};

// The encoder's output [1, frames, model_width] of features [1, frames,
// input_size]: scaled by sqrt(model_width) plus sinusoids, then
// encoder.encoders0.0, encoder.encoders.*, encoder.after_norm,
// encoder.tp_encoders.* and encoder.tp_norm, in the features' dtype.
mlx::core::array SanmEncode(const Checkpoint &checkpoint,
                            const mlx::core::array &features,
                            const SanmEncoderConfig &config);

} // namespace omni_common
