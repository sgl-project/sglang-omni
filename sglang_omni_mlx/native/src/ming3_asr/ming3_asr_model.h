#pragma once

#include <array>
#include <atomic>
#include <filesystem>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "common/layers.h"

namespace ming3_asr {

void ValidateAudio(const std::vector<float> &samples);
// Finite mono 16 kHz audio longer than 200 samples produces bfloat16 [frames,
// 128].
mlx::core::array ComputeAudioFeatures(const std::vector<float> &samples);

struct MingConfig {
  explicit MingConfig(const nlohmann::json &config);

  int hidden_size;
  int vocabulary_size;
  int intermediate_size;
  int expert_intermediate_size;
  int shared_intermediate_size;
  int layer_count;
  int attention_head_count;
  int attention_head_dimension;
  int layers_per_attention_group;
  int key_value_lora_rank;
  int query_lora_rank;
  int rotary_head_dimension;
  int non_rotary_head_dimension;
  int value_head_dimension;
  int convolution_kernel_token_count;
  int expert_count;
  int selected_expert_count;
  int expert_group_count;
  int selected_expert_group_count;
  int dense_layer_count;
  int max_context_token_count;
  std::vector<int> eos_token_ids;
  int audio_token_id;
  int audio_hidden_dimension;
  int audio_attention_head_count;
  int audio_layer_count;
  int audio_context_frame_count;
  int audio_chunk_frame_count;
  float rms_norm_epsilon;
  float rope_theta;
  float routed_scaling_factor;
  float kda_lower_bound;
  bool should_normalize_audio_embeddings;
  omni_common::QuantizationConfig quantization;

  bool UsesMla(int layer_index) const;
  int AudioTokenCount(int frame_count) const;
};

struct DeltaState {
  std::array<mlx::core::array, 3> convolution;
  mlx::core::array recurrent;
};

struct LatentAttentionState {
  mlx::core::array keys;
  mlx::core::array values;
};

struct DeltaCache {
  std::optional<DeltaState> state;
};

struct LatentAttentionCache {
  std::optional<LatentAttentionState> state;
};

using LayerCache = std::variant<DeltaCache, LatentAttentionCache>;

class MingModel;

class MingCache {
public:
  const std::vector<LayerCache> &layers() const { return layers_; }

private:
  friend class MingModel;
  explicit MingCache(const MingModel *owner);

  const MingModel *owner_;
  std::vector<LayerCache> layers_;
  int token_count_ = 0;
  bool is_valid_ = true;
};

struct GatedDeltaResult {
  mlx::core::array output;
  mlx::core::array next_state;
};

GatedDeltaResult
GatedDelta(const mlx::core::array &queries, const mlx::core::array &keys,
           const mlx::core::array &values, const mlx::core::array &decay,
           const mlx::core::array &beta, const mlx::core::array &state);

class MingModel {
public:
  explicit MingModel(const std::filesystem::path &model_directory);
  MingModel(const MingModel &) = delete;
  MingModel &operator=(const MingModel &) = delete;
  MingModel(MingModel &&) = delete;
  MingModel &operator=(MingModel &&) = delete;

  const MingConfig &config() const { return config_; }
  // A request-local cache tied to this model; a failed Forward invalidates it.
  MingCache NewCache() const;
  // Vocabulary token IDs produce [1, token_count, hidden_size] in the embedding
  // dtype.
  mlx::core::array EmbedTokens(const std::vector<int> &token_ids) const;
  // Produces [1, audio_token_count, hidden_size] from finite mono 16 kHz
  // samples.
  mlx::core::array EncodeAudio(const std::vector<float> &samples,
                               const std::atomic<bool> &cancel) const;
  // Appends [1, token_count, hidden_size] to the cache and returns hidden
  // states with the same shape.
  mlx::core::array Forward(const mlx::core::array &embeddings, MingCache &cache,
                           const std::atomic<bool> &cancel) const;
  // Projects only the last token to float32 [1, 1, vocabulary_size].
  mlx::core::array LastTokenLogits(const mlx::core::array &hidden_states) const;

private:
  MingModel(const nlohmann::json &configuration,
            const std::filesystem::path &model_directory);
  void ValidateWeights() const;
  mlx::core::array EncodeAudioChunk(const mlx::core::array &audio_features,
                                    const std::atomic<bool> &cancel) const;
  mlx::core::array DeltaAttention(const mlx::core::array &hidden_states,
                                  const std::string &weight_prefix,
                                  DeltaCache &cache) const;
  mlx::core::array LatentAttention(const mlx::core::array &hidden_states,
                                   const std::string &weight_prefix,
                                   LatentAttentionCache &cache) const;
  mlx::core::array FeedForward(const mlx::core::array &hidden_states,
                               const std::string &weight_prefix,
                               int layer_index) const;
  MingConfig config_;
  omni_common::Checkpoint checkpoint_;
};

} // namespace ming3_asr
