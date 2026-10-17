// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "layers.h"

namespace ming_tts {

struct AcousticConfig {
  int hidden_size;
  int depth;
  int num_heads;
  int intermediate_size;
  static AcousticConfig FromJson(const nlohmann::json &config);
};

struct TailOutput {
  mlx::core::array latents;
  mlx::core::array feedback;
  mlx::core::array stop_probability;
};

mlx::core::array CfmTimesteps(int steps);

class MingAcoustic {
public:
  explicit MingAcoustic(const std::filesystem::path &directory);
  TailOutput Step(const mlx::core::array &hidden,
                  const mlx::core::array &history,
                  const mlx::core::array &noise,
                  const mlx::core::array &timesteps,
                  const mlx::core::array &sde_random, float cfg, float sigma,
                  float temperature) const;
  int patch_size() const { return patch_size_; }
  int history_frames() const { return history_frames_; }
  int latent_dim() const { return latent_dim_; }

private:
  mlx::core::array Aggregate(const mlx::core::array &latents) const;
  MingAcoustic(const nlohmann::json &config,
               const std::filesystem::path &directory);
  mlx::core::array Blocks(mlx::core::array hidden, const AcousticConfig &config,
                          const std::string &prefix) const;
  mlx::core::array Dit(const mlx::core::array &latents,
                       const mlx::core::array &timestep,
                       const mlx::core::array &hidden,
                       const mlx::core::array &history) const;
  AcousticConfig dit_, aggregator_;
  int patch_size_, history_frames_, latent_dim_, llm_hidden_size_;
  qwen3_asr::Checkpoint checkpoint_;
};

} // namespace ming_tts
