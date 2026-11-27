// SPDX-License-Identifier: Apache-2.0
// MeanFlow head: one audio latent from one LM hidden state, via a single-step
// flow matching sampler.
#pragma once

#include "common/layers.h"
#include "haru/haru_config.h"
#include "mlx/mlx.h"

namespace haru {

namespace mx = mlx::core;

class MeanFlowHead {
public:
  MeanFlowHead(const omni_common::Checkpoint &checkpoint,
               const HeadConfig &config);

  // LM hidden states [B, H] and float32 noise [B, D] -> normalized latents
  // [B, D].
  mx::array Sample(const mx::array &hidden, const mx::array &noise,
                   float guidance) const;

private:
  // A scalar -> [1, hidden] via sinusoidal features and a two-layer MLP.
  mx::array ScalarEmbed(const std::string &prefix, float value) const;
  // The mean velocity u(z, t, h | c): z [B, D] and summed conditioning
  // c [B, H] -> [B, D].
  mx::array Velocity(const mx::array &z, const mx::array &c) const;

  const omni_common::Checkpoint &checkpoint_;
  HeadConfig config_;
  mx::Dtype dtype_;
  mx::array frequencies_; // [128] sinusoidal frequency basis
  mx::array time_embedding_;
  mx::array interval_embedding_;
  mx::array step_embedding_;
};

} // namespace haru
