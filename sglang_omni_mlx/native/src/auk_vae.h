// SPDX-License-Identifier: MIT AND Apache-2.0
// Copyright (C) 2026 Tencent. All rights reserved.
// Copyright (c) 2022 NVIDIA CORPORATION.
// AuK audio VAE; see LICENSE for the MIT permission notice.
#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "layers.h"

namespace auk {

struct PosteriorStats {
  mlx::core::array mean;
  mlx::core::array log_standard_deviation;
};

class VAE {
public:
  explicit VAE(const std::filesystem::path &vae_directory);
  int HopSize() const { return hop_size_; }
  int LatentDimension() const { return latent_dimension_; }

  // Waveform [samples]; posterior, noise and normalized latents [frames,
  // channels].
  PosteriorStats Posterior(const mlx::core::array &waveform) const;
  mlx::core::array Encode(const PosteriorStats &posterior,
                          const mlx::core::array &posterior_noise) const;
  mlx::core::array Decode(const mlx::core::array &latents) const;

private:
  mlx::core::array Convolution(const mlx::core::array &hidden_states,
                               const std::string &prefix, int stride = 1,
                               int dilation = 1, bool causal = false) const;
  mlx::core::array Activation(const mlx::core::array &hidden_states,
                              const std::string &prefix) const;
  mlx::core::array ResidualBlock(const mlx::core::array &hidden_states,
                                 const std::string &prefix,
                                 const std::vector<int> &dilations) const;

  qwen3_asr::Checkpoint checkpoint_;
  int hop_size_ = 1;
  int latent_dimension_ = 0;
  bool causal_ = false;
  bool activation_causal_ = false;
  bool snake_logscale_ = false;
  std::vector<int> downsample_rates_;
  std::vector<int> upsample_rates_;
  std::vector<std::vector<int>> residual_dilations_;
};

} // namespace auk
