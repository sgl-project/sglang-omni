// SPDX-License-Identifier: Apache-2.0
// The Qwen3 text decoder Qwen3-ASR and MOSS-Transcribe-Diarize share; only the
// weight-name prefix differs. Voxtral TTS's Mistral decoder is the same
// without the query/key norms and with traditional rotary embeddings.
#pragma once

#include <map>
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
  // Rotates with cosines and sines rounded to the input's dtype, and masks
  // a multi-token step with an additive mask, as Hugging Face-style MLX
  // ports do, instead of the fused rope and causal attention.
  bool rotary_tables = false;
  int hidden_size = 0;
  int vocab_size = 0;

  static Qwen3DecoderConfig FromJson(const nlohmann::json &text_config);
};

// Per-layer key/value cache that grows in fixed steps. Keys and values may
// differ in head width, as in MLA.
class KVCache {
public:
  std::pair<mlx::core::array, mlx::core::array>
  UpdateAndFetch(const mlx::core::array &keys, const mlx::core::array &values);
  // Drops the newest token_count positions; the next update overwrites them.
  void Trim(int token_count);
  int offset() const { return offset_; }
  // The cache's buffers, to evaluate; empty before the first update.
  std::vector<mlx::core::array> Arrays() const;
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
  // Each row is one sequence's filled [1, kv_heads, length, head_dim].
  explicit BatchedKVCache(
      const std::vector<std::pair<mlx::core::array, mlx::core::array>> &rows);
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
  // Weights under prefix (layers, norm, embed_tokens), or at the top level
  // when the prefix is empty; the output projection is the tied embedding
  // table.
  Qwen3Decoder(const omni_common::Checkpoint &checkpoint, std::string prefix,
               Qwen3DecoderConfig config);
  // Combine the dense, bias-free MLP input projections into one matmul.
  void PackDenseGateUp();
  // Precompute a short sequence's opening pair and subsequent one-token
  // rotary steps before decoding.
  void CacheRotary(int token_count, mlx::core::Dtype dtype);

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
  // One decode step for a batch, embeddings [batch, 1, hidden]. offsets is
  // each row's true position, int32 [batch]; pads is each row's count of
  // left-padding cache slots, int32 [batch, 1]. Returns logits [batch, vocab].
  mlx::core::array DecodeBatch(const mlx::core::array &embeddings,
                               std::vector<BatchedKVCache> &caches,
                               const mlx::core::array &offsets,
                               const mlx::core::array &pads) const;

private:
  // A step's rotary tables [1, 1, length, head_dim] and, for a step of more
  // than one token, its additive causal mask [length, cached + length].
  struct RotaryInputs {
    mlx::core::array cosines;
    mlx::core::array sines;
    std::optional<mlx::core::array> mask;
  };

  std::string Name(const std::string &suffix) const;
  mlx::core::array Layer(const mlx::core::array &x, int layer, KVCache &cache,
                         const std::optional<RotaryInputs> &rotary) const;
  mlx::core::array BatchLayer(const mlx::core::array &x, int layer,
                              BatchedKVCache &cache,
                              const mlx::core::array &offsets,
                              const mlx::core::array &mask) const;
  RotaryInputs Rotary(int length, int offset, mlx::core::Dtype dtype) const;
  // Rotates heads [batch, heads, length, head_dim] at the cache's next
  // positions: with the step's tables when given, else with the fused rope.
  mlx::core::array Rotated(const mlx::core::array &heads, const KVCache &cache,
                           const std::optional<RotaryInputs> &rotary) const;
  // One layer's keys, rotated at the cache's next positions, and values of x,
  // each [batch, kv_heads, length, head_dim].
  std::pair<mlx::core::array, mlx::core::array>
  KeysValues(const mlx::core::array &x, const std::string &prefix,
             const KVCache &cache,
             const std::optional<RotaryInputs> &rotary) const;

  const omni_common::Checkpoint &checkpoint_;
  std::string prefix_;
  Qwen3DecoderConfig config_;
  std::vector<mlx::core::array> gate_up_weights_;
  std::map<std::pair<int, int>, RotaryInputs> cached_rotary_inputs_;
};

} // namespace qwen3_asr
