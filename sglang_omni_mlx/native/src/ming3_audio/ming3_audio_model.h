// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <deque>
#include <filesystem>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/layers.h"
#include "common/tokenizer.h"
#include "qwen3_asr/qwen3_decoder.h"

namespace ming3_audio {

inline constexpr int kDefaultContextTokens = 8192;
enum class SpeculativeAlgorithm { kNone, kNextN };
int ParseContextLength(const std::string &value);
SpeculativeAlgorithm ParseSpeculativeAlgorithm(const std::string &value);

using Cancellation = std::function<bool()>;

class Cancelled : public std::runtime_error {
public:
  Cancelled() : std::runtime_error("chat generation cancelled") {}
};

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
  int draft_tokens = 0;
  int accepted_draft_tokens = 0;
};

struct PreparedChatRequest {
  ChatRequest request;
  std::vector<int> prompt_ids;
  int audio_start;
  int audio_tokens;
};

struct TextConfig {
  int num_hidden_layers;
  int num_attention_heads;
  int head_dim;
  int layer_group_size;
  int qk_rope_head_dim;
  int qk_nope_head_dim;
  int v_head_dim;
  int kv_lora_rank;
  int short_conv_kernel_size;
  int first_k_dense_replace;
  int num_experts;
  int num_experts_per_tok;
  int n_group;
  int topk_group;
  int max_position_embeddings;
  float rms_norm_eps;
  float rope_theta;
  float kda_lower_bound;
  float routed_scaling_factor;

  static TextConfig FromJson(const nlohmann::json &config);
};

struct AudioConfig {
  int n_audio_state;
  int n_audio_head;
  int n_audio_layer;
  std::vector<int> downsampler_kernel_size_list;
  std::vector<int> downsampler_stride_list;
  int whisper_chunk_size;

  static AudioConfig FromJson(const nlohmann::json &config);
};

class Model {
public:
  Model(const std::filesystem::path &directory, int context_tokens,
        SpeculativeAlgorithm speculative_algorithm);
  PreparedChatRequest Prepare(ChatRequest request) const;
  ChatResult Generate(const PreparedChatRequest &prepared,
                      const Cancellation &cancel) const;

private:
  // A KDA layer's inputs for one round that is not verified yet.
  struct PendingRound {
    mlx::core::array state;
    mlx::core::array queries;
    mlx::core::array keys;
    mlx::core::array values;
    mlx::core::array decay;
    mlx::core::array beta;
    std::vector<mlx::core::array> convolution_inputs;
  };

  struct LayerCache {
    qwen3_asr::KVCache attention;
    std::array<std::optional<mlx::core::array>, 3> convolution;
    std::optional<mlx::core::array> recurrent;
    bool keeps_pending_rounds = false;
    std::deque<PendingRound> pending_rounds;
  };

  struct DraftedGeneration {
    std::vector<int> token_ids;
    int draft_tokens = 0;
    int accepted_draft_tokens = 0;
  };

  Model(const std::filesystem::path &directory, const nlohmann::json &config,
        int context_tokens, SpeculativeAlgorithm speculative_algorithm);
  mlx::core::array Encode(const mlx::core::array &mel) const;
  mlx::core::array Embed(const mlx::core::array &token_ids) const;
  mlx::core::array Forward(const mlx::core::array &embeddings,
                           std::vector<LayerCache> &caches) const;
  static mlx::core::array LastPosition(const mlx::core::array &hidden);
  mlx::core::array Logits(const mlx::core::array &hidden) const;
  // Returns the last position's hidden state; earlier positions only fill the
  // cache.
  mlx::core::array DraftForward(const mlx::core::array &next_embeddings,
                                const mlx::core::array &hidden,
                                qwen3_asr::KVCache &cache) const;
  void AcceptOldestRound(std::vector<LayerCache> &caches) const;
  // Keeps the oldest pending round's anchor and undoes the dropped_positions
  // latent-attention positions after it.
  void RejectOldestDraft(std::vector<LayerCache> &caches,
                         int dropped_positions) const;
  bool IsStopToken(int token) const;
  DraftedGeneration GenerateWithDraft(const mlx::core::array &embeddings,
                                      const std::vector<int> &prompt_ids,
                                      int max_new_tokens,
                                      const Cancellation &cancel) const;
  int AudioTokenCount(int mel_frames) const;
  mlx::core::array RmsNorm(const mlx::core::array &hidden,
                           const std::string &prefix) const;
  mlx::core::array Mlp(const mlx::core::array &hidden,
                       const std::string &prefix) const;
  mlx::core::array Moe(const mlx::core::array &hidden,
                       const std::string &prefix) const;
  mlx::core::array DeltaAttention(const mlx::core::array &hidden,
                                  const std::string &prefix,
                                  LayerCache &cache) const;
  mlx::core::array LatentAttention(const mlx::core::array &hidden,
                                   const std::string &prefix,
                                   qwen3_asr::KVCache &cache) const;

  TextConfig text_;
  AudioConfig audio_;
  int context_tokens_;
  SpeculativeAlgorithm speculative_algorithm_;
  omni_common::Checkpoint checkpoint_;
  omni_common::Tokenizer tokenizer_;
  std::vector<int> stop_tokens_;
};

} // namespace ming3_audio
