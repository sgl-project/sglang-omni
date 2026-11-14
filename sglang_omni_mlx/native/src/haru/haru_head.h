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

  // LM hidden states [B, H] and noise [B, D] -> normalized latents [B, D].
  mx::array Sample(const mx::array &hidden, const mx::array &noise,
                   float guidance) const;

private:
  // A scalar -> [1, hidden] via sinusoidal features and a two-layer MLP.
  mx::array ScalarEmbed(const std::string &prefix, float value,
                        mx::Dtype dtype) const;
  // The mean velocity u(z, t, h | c): z [B, D] and summed conditioning
  // c [B, H] -> [B, D].
  mx::array Velocity(const mx::array &z, const mx::array &c) const;

  const omni_common::Checkpoint &checkpoint_;
  HeadConfig config_;
  mx::array freqs_;  // [1, 128] sinusoidal frequency basis
};

} // namespace haru
