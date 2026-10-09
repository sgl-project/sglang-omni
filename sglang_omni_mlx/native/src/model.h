// SPDX-License-Identifier: Apache-2.0
// Qwen3-ASR on MLX: audio encoder, Qwen3 text decoder and checkpoint loading.
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "audio.h"
#include "mlx/mlx.h"

namespace qwen3_asr {

struct AudioEncoderConfig {
  int num_mel_bins = 0;
  int encoder_layers = 0;
  int encoder_attention_heads = 0;
  int d_model = 0;
  int n_window = 0;
  int n_window_infer = 0;
};

struct TextDecoderConfig {
  int num_hidden_layers = 0;
  int num_attention_heads = 0;
  int num_key_value_heads = 0;
  int head_dim = 0;
  float rms_norm_eps = 0.0f;
  float rope_theta = 0.0f;
};

struct QuantizationConfig {
  int group_size = 0;
  int bits = 0;
  std::string mode = "affine";
};

// Per-layer key/value cache that grows in fixed steps.
class KVCache {
public:
  std::pair<mlx::core::array, mlx::core::array>
  UpdateAndFetch(const mlx::core::array &keys, const mlx::core::array &values);
  int offset() const { return offset_; }

private:
  std::optional<mlx::core::array> keys_;
  std::optional<mlx::core::array> values_;
  int offset_ = 0;
};

class Qwen3ASR {
public:
  // Builds the model from an MLX checkpoint directory; layers stored with
  // .scales are quantized as the checkpoint was.
  explicit Qwen3ASR(const std::filesystem::path &model_directory);

  // [mel_bins, frames] to [audio_tokens, output_dim].
  mlx::core::array EncodeAudio(const mlx::core::array &mel,
                               AudioLayout layout) const;
  // Token ids [1, length] to embeddings [1, length, hidden].
  mlx::core::array EmbedTokens(const mlx::core::array &ids) const;
  // Logits for the last position only, [vocab].
  mlx::core::array Decode(const mlx::core::array &embeddings,
                          std::vector<KVCache> &caches) const;
  std::vector<KVCache> NewCaches() const;

private:
  const mlx::core::array &Weight(const std::string &name) const;
  bool Has(const std::string &name) const;
  mlx::core::array Linear(const mlx::core::array &x,
                          const std::string &prefix) const;
  mlx::core::array LayerNorm(const mlx::core::array &x,
                             const std::string &prefix) const;
  mlx::core::array RmsNorm(const mlx::core::array &x,
                           const std::string &prefix) const;
  mlx::core::array Conv2d(const mlx::core::array &x,
                          const std::string &prefix) const;
  mlx::core::array AudioEncoderLayer(const mlx::core::array &x,
                                     int layer) const;
  mlx::core::array TextDecoderLayer(const mlx::core::array &x, int layer,
                                    KVCache &cache) const;

  AudioEncoderConfig audio_;
  TextDecoderConfig text_;
  QuantizationConfig quantization_;
  std::unordered_map<std::string, mlx::core::array> weights_;
};

} // namespace qwen3_asr
