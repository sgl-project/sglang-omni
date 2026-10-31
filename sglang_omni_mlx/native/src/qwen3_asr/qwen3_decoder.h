// SPDX-License-Identifier: Apache-2.0
// The Qwen3 text decoder Qwen3-ASR and MOSS-Transcribe-Diarize share; only the
// weight-name prefix differs. Voxtral TTS's Mistral decoder is the same
// without the query/key norms and with traditional rotary embeddings.
#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "common/layers.h"
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
  bool has_query_key_norm = true;
  // Rotates adjacent rather than half-split pairs of each head.
  bool traditional_rope = false;
  int hidden_size = 0;
  int vocab_size = 0;

  static Qwen3DecoderConfig FromJson(const nlohmann::json &text_config);
};

// Per-layer key/value cache that grows in fixed steps.
class KVCache {
public:
  std::pair<mlx::core::array, mlx::core::array>
  UpdateAndFetch(const mlx::core::array &keys, const mlx::core::array &values);
  // Drops the newest token_count positions; the next update overwrites them.
  void Trim(int token_count);
  int offset() const { return offset_; }

private:
  std::optional<mlx::core::array> keys_;
  std::optional<mlx::core::array> values_;
  int offset_ = 0;
};

class Qwen3Decoder {
public:
  // Weights under prefix (layers, norm, embed_tokens), or at the top level
  // when the prefix is empty; the output projection is the tied embedding
  // table.
  Qwen3Decoder(const omni_common::Checkpoint &checkpoint, std::string prefix,
               Qwen3DecoderConfig config);

  // Token ids [1, length] to embeddings [1, length, hidden].
  mlx::core::array EmbedTokens(const mlx::core::array &ids) const;
  // Runs every layer over the embeddings, appending to the caches; returns
  // the last layer's hidden states. The hidden states entering each layer in
  // capture_layers are appended to captured, in that order.
  mlx::core::array
  Forward(const mlx::core::array &embeddings, std::vector<KVCache> &caches,
          const std::vector<int> &capture_layers = {},
          std::vector<mlx::core::array> *captured = nullptr) const;
  // Appends every layer's keys and values of hidden_states [1, length,
  // hidden] to the caches, for later queries to attend to. The hidden states
  // are projected as given, without the layers' input norms.
  void AppendKeysValues(const mlx::core::array &hidden_states,
                        std::vector<KVCache> &caches) const;
  // Logits of the last position of the hidden states, [vocab].
  mlx::core::array LastLogits(const mlx::core::array &hidden_states) const;
  // Logits of every position of the hidden states, [length, vocab].
  mlx::core::array Logits(const mlx::core::array &hidden_states) const;
  std::vector<KVCache> NewCaches() const;
  const Qwen3DecoderConfig &config() const { return config_; }

private:
  std::string Name(const std::string &suffix) const;
  // One layer's keys, rotated at the cache's next positions, and values of x,
  // each [batch, kv_heads, length, head_dim].
  std::pair<mlx::core::array, mlx::core::array>
  KeysValues(const mlx::core::array &x, const std::string &prefix,
             const KVCache &cache) const;
  mlx::core::array Layer(const mlx::core::array &x, int layer,
                         KVCache &cache) const;

  const omni_common::Checkpoint &checkpoint_;
  std::string prefix_;
  Qwen3DecoderConfig config_;
};

} // namespace qwen3_asr
