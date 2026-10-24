// SPDX-License-Identifier: Apache-2.0
// dots.tts checkpoint loading: the three converted safetensors (core, vocoder,
// speaker), the latent stats and the merged config, with the layout fixes
// applied at load (conv weight transposes, the speaker wrapper prefix,
// BatchNorm folding to a frozen affine). The Qwen2.5 backbone ops delegate to
// qwen3_asr::Checkpoint, which resolves the quantized matmuls.
#pragma once

#include <filesystem>
#include <string>

#include "common/layers.h"
#include "mlx/mlx.h"
#include "nlohmann/json.hpp"

namespace dots_tts {

class DotsCheckpoint {
public:
  explicit DotsCheckpoint(const std::filesystem::path &model_directory);

  bool Has(const std::string &name) const { return checkpoint_.Has(name); }
  const mlx::core::array &Weight(const std::string &name) const {
    return checkpoint_.Weight(name);
  }

  mlx::core::array BackboneLinear(const mlx::core::array &x,
                                  const std::string &prefix) const {
    return checkpoint_.Linear(x, prefix);
  }

  const qwen3_asr::Checkpoint &BackboneCheckpoint() const {
    return checkpoint_;
  }

  const mlx::core::array &latent_mean() const { return latent_mean_; }
  const mlx::core::array &latent_var() const { return latent_var_; }
  const nlohmann::json &config() const { return config_; }

private:
  nlohmann::json config_;
  qwen3_asr::Checkpoint checkpoint_;
  mlx::core::array latent_mean_{0.0f};
  mlx::core::array latent_var_{0.0f};
};

mlx::core::array Conv1dWeight(const mlx::core::array &w,
                              mlx::core::Dtype dtype);
mlx::core::array ConvTranspose1dWeight(const mlx::core::array &w,
                                       mlx::core::Dtype dtype);
mlx::core::array Conv2dWeight(const mlx::core::array &w,
                              mlx::core::Dtype dtype);
mlx::core::array FilterWeight(const mlx::core::array &f, int channels);

} // namespace dots_tts
