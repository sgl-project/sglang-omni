// SPDX-License-Identifier: Apache-2.0
// The Qwen3 text decoder Qwen3-ASR and MOSS-Transcribe-Diarize share; only the
// weight-name prefix differs.
#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "layers.h"
#include "mlx/mlx.h"
#include "nlohmann/json.hpp"

namespace qwen3_asr {

struct Qwen3DecoderConfig {
  int num_hidden_layers = 0;
  int num_attention_heads = 0;
  int num_key_value_heads = 0;
  int head_dim = 0;
  float rms_norm_eps = 0.0f;
  float rope_theta = 0.0f;

  static Qwen3DecoderConfig FromJson(const nlohmann::json &text_config);
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

class Qwen3Decoder {
public:
  // Weights under prefix (layers, norm, embed_tokens); the output projection
  // is the tied embedding table.
  Qwen3Decoder(const Checkpoint &checkpoint, std::string prefix,
               Qwen3DecoderConfig config);

  // Token ids [1, length] to embeddings [1, length, hidden].
  mlx::core::array EmbedTokens(const mlx::core::array &ids) const;
  // Runs every layer over the embeddings, appending to the caches; returns
  // the last layer's hidden states.
  mlx::core::array Forward(const mlx::core::array &embeddings,
                           std::vector<KVCache> &caches) const;
  // Logits of the last position of the hidden states, [vocab].
  mlx::core::array LastLogits(const mlx::core::array &hidden_states) const;
  std::vector<KVCache> NewCaches() const;

private:
  mlx::core::array Layer(const mlx::core::array &x, int layer,
                         KVCache &cache) const;

  const Checkpoint &checkpoint_;
  std::string prefix_;
  Qwen3DecoderConfig config_;
};

} // namespace qwen3_asr
