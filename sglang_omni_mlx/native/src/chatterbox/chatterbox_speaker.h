// SPDX-License-Identifier: Apache-2.0
// Chatterbox-Turbo's CAM++ speaker encoder: Kaldi log-mel filterbank features
// of 16 kHz audio to the flow's x-vector.
#pragma once

#include <string>
#include <vector>

#include "common/layers.h"
#include "mlx/mlx.h"

namespace chatterbox {

class SpeakerEncoder {
public:
  // Reads the checkpoint's CAM++ weights under weight_prefix (with its
  // trailing dot), such as "s3gen.speaker_encoder.".
  SpeakerEncoder(const omni_common::Checkpoint &checkpoint,
                 const std::string &weight_prefix);

  // The x-vector of 16 kHz samples, float32 [1, 192].
  mlx::core::array Embed(const std::vector<float> &samples_16k) const;

private:
  const mlx::core::array &Weight(const std::string &name) const;
  mlx::core::array BatchNorm(const mlx::core::array &x,
                             const std::string &prefix) const;
  mlx::core::array ResidualBlock(const mlx::core::array &x,
                                 const std::string &prefix, int stride) const;
  mlx::core::array DenseLayer(const mlx::core::array &x,
                              const std::string &prefix, int dilation) const;

  omni_common::WeightMap weights_;
};

} // namespace chatterbox
