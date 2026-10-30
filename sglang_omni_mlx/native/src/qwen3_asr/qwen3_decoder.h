// SPDX-License-Identifier: Apache-2.0
// The Qwen3 text decoder Qwen3-ASR and MOSS-Transcribe-Diarize share; only the
// weight-name prefix differs. Voxtral TTS's Mistral decoder is the same
// without the query/key norms and with traditional rotary embeddings.
#pragma once

#include <string>
#include <vector>

#include "common/kv_cache.h"
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
                           std::vector<omni_common::KVCache> &caches) const;
  // Logits of the last position of the hidden states, [vocab].
  mlx::core::array LastLogits(const mlx::core::array &hidden_states) const;
  std::vector<omni_common::KVCache> NewCaches() const;

private:
  mlx::core::array Layer(const mlx::core::array &x, int layer,
                         omni_common::KVCache &cache) const;

  const omni_common::Checkpoint &checkpoint_;
  std::string prefix_;
  Qwen3DecoderConfig config_;
};

} // namespace qwen3_asr
