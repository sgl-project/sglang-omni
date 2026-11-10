// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Tencent. All rights reserved.
#pragma once

#include <filesystem>
#include <functional>
#include <optional>

#include "common/layers.h"

namespace auk {

class DiT {
public:
  // A released checkpoint's weights and its config.yaml arch section.
  DiT(omni_common::WeightMap weights, const nlohmann::json &architecture,
      int latent_channels, mlx::core::Dtype dtype);
  // The conditioner's text layers and width, which a separate checkpoint
  // sets.
  void ValidateInputDimensions(int text_layers, int text_hidden_channels) const;

  // All Qwen hidden states [layers + 1, tokens, hidden], including embeddings.
  mlx::core::array
  FuseHiddenStates(const mlx::core::array &all_hidden_states) const;

  mlx::core::array Sample(const mlx::core::array &conditioning,
                          const std::optional<mlx::core::array> &reference,
                          int reference_length, const mlx::core::array &noise,
                          const mlx::core::array &times,
                          float guidance_strength,
                          const std::function<void()> &check_cancelled) const;

private:
  void ValidateWeights() const;
  struct PreparedReference {
    std::optional<mlx::core::array> conditional;
    std::optional<mlx::core::array> unconditional;
    std::optional<mlx::core::array> valid_frames;
    int frame_count = 0;
  };
  PreparedReference
  PrepareReference(const std::optional<mlx::core::array> &reference,
                   int reference_length, bool classifier_free_guidance) const;
  mlx::core::array ProjectText(const mlx::core::array &conditioning) const;
  mlx::core::array
  EmbedAudio(const mlx::core::array &latents,
             const std::optional<mlx::core::array> &valid_frames) const;
  mlx::core::array Predict(const mlx::core::array &latents,
                           const mlx::core::array &projected_text,
                           const mlx::core::array &timestep,
                           const PreparedReference &reference,
                           bool classifier_free_guidance) const;

  nlohmann::json config_;
  omni_common::Checkpoint checkpoint_;
  mlx::core::Dtype dtype_;
  int head_count_;
  int joint_layer_count_;
  int single_layer_count_;
  bool attention_mask_enabled_;
};

} // namespace auk
