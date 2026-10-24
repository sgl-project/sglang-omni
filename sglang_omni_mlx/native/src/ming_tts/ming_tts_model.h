// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <filesystem>
#include <vector>

#include "common/layers.h"
#include "qwen3_asr/qwen3_decoder.h"

namespace ming_tts {

qwen3_asr::WeightMap
LoadComponentWeights(const std::filesystem::path &directory,
                     const std::vector<std::string> &prefixes);
void RequireShape(const qwen3_asr::Checkpoint &checkpoint,
                  const std::string &name, const mlx::core::Shape &shape);
mlx::core::Dtype WeightDtype(const qwen3_asr::Checkpoint &checkpoint,
                             const std::string &prefix);

struct BackboneConfig {
  int vocab_size;
  int hidden_size;
  int intermediate_size;
  int moe_intermediate_size;
  int num_hidden_layers;
  int num_attention_heads;
  int num_key_value_heads;
  int head_dim;
  int num_experts;
  int num_experts_per_tok;
  int num_shared_experts;
  int first_k_dense_replace;
  int max_position_embeddings;
  float rms_norm_eps;
  float rope_theta;
  float routed_scaling_factor;
  bool norm_topk_prob;
  bool use_qkv_bias;
  bool use_bias;
  std::array<int, 3> mrope_section;

  static BackboneConfig FromJson(const nlohmann::json &config);
};

class MingBackbone {
public:
  explicit MingBackbone(const std::filesystem::path &model_directory);

  mlx::core::array EmbedTokens(const mlx::core::array &input_ids) const;
  mlx::core::array Forward(const mlx::core::array &embeddings,
                           const mlx::core::array &positions,
                           std::vector<qwen3_asr::KVCache> &caches) const;
  std::vector<qwen3_asr::KVCache> NewCaches() const;
  const BackboneConfig &config() const { return config_; }

private:
  MingBackbone(const nlohmann::json &config,
               const std::filesystem::path &model_directory);
  mlx::core::array Rotary(const mlx::core::array &x,
                          const mlx::core::array &positions) const;
  mlx::core::array Mlp(const mlx::core::array &x,
                       const std::string &prefix) const;
  mlx::core::array Moe(const mlx::core::array &x,
                       const std::string &prefix) const;

  BackboneConfig config_;
  qwen3_asr::Checkpoint checkpoint_;
  mlx::core::array inverse_frequencies_;
};

} // namespace ming_tts
