// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "common/chat_service.h"
#include "common/layers.h"
#include "common/tokenizer.h"

namespace ming3_audio {

struct ChatRequest {
  std::string system_prompt;
  std::string before_audio;
  std::string after_audio;
  std::vector<float> samples;
  int max_new_tokens = 256;
};

struct ChatResult {
  std::string text;
  std::string finish_reason = "length";
  int prompt_tokens = 0;
  int completion_tokens = 0;
};

struct LatentAttentionState {
  mlx::core::array keys;
  mlx::core::array values;
};

struct LayerCache {
  std::optional<LatentAttentionState> attention;
  std::array<std::optional<mlx::core::array>, 3> convolution;
  std::optional<mlx::core::array> recurrent;
};

class Model {
public:
  explicit Model(const std::filesystem::path &directory);
  ChatResult Generate(const ChatRequest &request,
                      const chat_service::Cancellation &cancel,
                      omni_common::WeightMap *trace = nullptr) const;
  mlx::core::array Encode(const mlx::core::array &mel,
                          omni_common::WeightMap *trace = nullptr) const;
  mlx::core::array Embed(const mlx::core::array &token_ids) const;
  mlx::core::array Forward(const mlx::core::array &embeddings,
                           std::vector<LayerCache> &caches,
                           omni_common::WeightMap *trace = nullptr) const;
  std::vector<LayerCache> NewCaches() const;
  int AudioTokenCount(int mel_frames) const;
  int max_context_tokens() const { return text_.at("max_position_embeddings"); }

private:
  Model(const std::filesystem::path &directory, const nlohmann::json &config);
  mlx::core::array RmsNorm(const mlx::core::array &hidden,
                           const std::string &prefix) const;
  mlx::core::array Mlp(const mlx::core::array &hidden,
                       const std::string &prefix) const;
  mlx::core::array Moe(const mlx::core::array &hidden,
                       const std::string &prefix) const;
  mlx::core::array DeltaAttention(const mlx::core::array &hidden,
                                  const std::string &prefix,
                                  LayerCache &cache) const;
  mlx::core::array
  LatentAttention(const mlx::core::array &hidden, const std::string &prefix,
                  std::optional<LatentAttentionState> &cache) const;

  nlohmann::json text_;
  nlohmann::json audio_;
  omni_common::Checkpoint checkpoint_;
  omni_common::Tokenizer tokenizer_;
  std::vector<int> stop_tokens_;
};

} // namespace ming3_audio
