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

  static Qwen3DecoderConfig FromJson(const nlohmann::json &text_config);
};

// Per-layer key/value cache that grows in fixed steps.
class KVCache {
public:
  std::pair<mlx::core::array, mlx::core::array>
  UpdateAndFetch(const mlx::core::array &keys, const mlx::core::array &values);
  int offset() const { return offset_; }
  // The filled keys and values, [1, kv_heads, offset, head_dim].
  std::pair<mlx::core::array, mlx::core::array> Filled() const;

private:
  std::optional<mlx::core::array> keys_;
  std::optional<mlx::core::array> values_;
  int offset_ = 0;
};

// Key/value cache for sequences decoded in lockstep. Rows are left-padded to
// the longest, so every row appends at the same index; the pad slots are
// masked out of attention.
class BatchedKVCache {
public:
  // Each row is one sequence's filled [1, kv_heads, length, head_dim]; room
  // is kept for extra_tokens appended steps.
  BatchedKVCache(
      const std::vector<std::pair<mlx::core::array, mlx::core::array>> &rows,
      int extra_tokens);
  // Appends [batch, kv_heads, 1, head_dim] and returns everything filled.
  std::pair<mlx::core::array, mlx::core::array>
  Append(const mlx::core::array &keys, const mlx::core::array &values);
  int length() const { return end_; }
  int capacity() const { return keys_.shape(2); }

private:
  mlx::core::array keys_;
  mlx::core::array values_;
  int end_ = 0;
};

class Qwen3Decoder {
public:
  // Weights under prefix (layers, norm, embed_tokens); the output projection
  // is the tied embedding table.
  Qwen3Decoder(const omni_common::Checkpoint &checkpoint, std::string prefix,
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
  // One decode step for a batch, embeddings [batch, 1, hidden]. offsets is
  // each row's true position, int32 [batch]; valid is [batch, 1, 1, capacity],
  // true where a cache slot is not padding. Returns logits [batch, vocab].
  mlx::core::array DecodeBatch(const mlx::core::array &embeddings,
                               std::vector<BatchedKVCache> &caches,
                               const mlx::core::array &offsets,
                               const mlx::core::array &valid) const;

private:
  mlx::core::array Layer(const mlx::core::array &x, int layer,
                         KVCache &cache) const;
  mlx::core::array BatchLayer(const mlx::core::array &x, int layer,
                              BatchedKVCache &cache,
                              const mlx::core::array &offsets,
                              const mlx::core::array &mask) const;

  const omni_common::Checkpoint &checkpoint_;
  std::string prefix_;
  Qwen3DecoderConfig config_;
};

} // namespace qwen3_asr
