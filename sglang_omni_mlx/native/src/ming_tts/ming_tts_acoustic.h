// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>
#include <vector>

#include "common/layers.h"

namespace ming_tts {

struct AcousticConfig {
  int hidden_size;
  int depth;
  int num_heads;
  static AcousticConfig FromJson(const nlohmann::json &config);
};

struct TailOutput {
  // float32 [batch, patch, latent].
  mlx::core::array latents;
  // float32 [batch].
  mlx::core::array stop_probability;
};

class MingAcoustic {
public:
  explicit MingAcoustic(const std::filesystem::path &directory);
  // A patch the CFM samples for the backbone's last hidden state [batch, 1,
  // hidden] after history [batch, history_frames, latent], and the stop head's
  // probability that it ends the speech.
  TailOutput Step(const mlx::core::array &hidden,
                  const mlx::core::array &history, float cfg, float sigma,
                  float temperature) const;
  int patch_size() const { return patch_size_; }
  int history_frames() const { return history_frames_; }
  int latent_dim() const { return latent_dim_; }
  mlx::core::array Aggregate(const mlx::core::array &latents) const;
  mlx::core::array ProjectSpeaker(const mlx::core::array &speaker) const;

private:
  MingAcoustic(const nlohmann::json &config,
               const std::filesystem::path &directory);
  mlx::core::array Blocks(mlx::core::array hidden, const AcousticConfig &config,
                          const std::string &prefix) const;
  // The DiT's time embeddings [steps, 1, hidden] of cfm_times_ but the last.
  mlx::core::array TimeEmbeddings() const;
  AcousticConfig dit_, aggregator_;
  int patch_size_, history_frames_, latent_dim_, llm_hidden_size_;
  omni_common::Checkpoint checkpoint_;
  // The CFM's sway-sampled times, from 0 to 1.
  std::vector<float> cfm_times_;
  mlx::core::array time_embeddings_;
};

} // namespace ming_tts
