// SPDX-License-Identifier: Apache-2.0
// Qwen3-ASR's text decoder: the shared Qwen3Decoder's computation with its
// quantized projections and one-token attention on the tuned kernels.
// Other models keep the shared Qwen3Decoder.
#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "common/layers.h"
#include "common/qwen3_decoder.h"
#include "mlx/mlx.h"
#include "qwen3_asr/kernels/kernels.h"

namespace qwen3_asr {

// Per-layer key/value cache that grows in fixed steps.
class Qwen3ASRKVCache {
public:
  Qwen3ASRKVCache() = default;
  // Updates write into the cache's buffer, which a copy would share.
  Qwen3ASRKVCache(const Qwen3ASRKVCache &) = delete;
  Qwen3ASRKVCache &operator=(const Qwen3ASRKVCache &) = delete;
  Qwen3ASRKVCache(Qwen3ASRKVCache &&) = default;
  Qwen3ASRKVCache &operator=(Qwen3ASRKVCache &&) = default;

  std::pair<mlx::core::array, mlx::core::array>
  UpdateAndFetch(const mlx::core::array &keys, const mlx::core::array &values);
  // Grows the caches, [batch, heads, capacity, head_dim], to hold rows of
  // rows_shape [batch, heads, tokens, head_dim] from offset() on, and returns
  // them.
  std::pair<mlx::core::array, mlx::core::array>
  Reserve(const mlx::core::Shape &rows_shape, mlx::core::Dtype dtype);
  // Takes caches holding token_count more tokens from offset() on, moves past
  // them, and returns every token's keys and values.
  std::pair<mlx::core::array, mlx::core::array>
  Commit(const mlx::core::array &keys, const mlx::core::array &values,
         int token_count);
  int offset() const { return offset_; }
  // Forgets the last token_count entries; later updates overwrite them.
  void Rewind(int token_count) { offset_ -= token_count; }

private:
  std::optional<mlx::core::array> keys_;
  std::optional<mlx::core::array> values_;
  int offset_ = 0;
};

class Qwen3ASRDecoder {
public:
  // Weights under prefix (layers, norm, embed_tokens); the output projection
  // is the tied embedding table.
  Qwen3ASRDecoder(const omni_common::Checkpoint &checkpoint, std::string prefix,
                  Qwen3DecoderConfig config);

  // Token ids [1, length] to embeddings [1, length, hidden].
  mlx::core::array EmbedTokens(const mlx::core::array &ids) const;
  // Runs every layer over the embeddings, appending to the caches; returns
  // the last layer's hidden states.
  mlx::core::array Forward(const mlx::core::array &embeddings,
                           std::vector<Qwen3ASRKVCache> &caches) const;
  // Logits of the last rows positions of the hidden states, [rows, vocab].
  mlx::core::array LastLogits(const mlx::core::array &hidden_states,
                              int rows) const;
  std::vector<Qwen3ASRKVCache> NewCaches() const;

private:
  // The layers' quantized arrays, none unless every layer is quantized.
  std::optional<std::vector<QuantizedWeights>>
  QuantizedLayers(const std::vector<std::string> &prefixes) const;
  // Each layer applied to x, RMS-normalized first by the norm at norm_prefix
  // when given; fused when quantized and the kernels allow.
  std::vector<mlx::core::array> Projections(
      const mlx::core::array &x, const std::vector<std::string> &prefixes,
      const std::optional<std::string> &norm_prefix = std::nullopt) const;
  mlx::core::array Layer(const mlx::core::array &x, int layer,
                         Qwen3ASRKVCache &cache) const;

  const omni_common::Checkpoint &checkpoint_;
  std::string prefix_;
  Qwen3DecoderConfig config_;
};

} // namespace qwen3_asr
