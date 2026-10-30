// SPDX-License-Identifier: Apache-2.0
#include "qwen3_asr/qwen3_decoder.h"

#include <cmath>
#include <utility>

namespace qwen3_asr {

namespace mx = mlx::core;

Qwen3DecoderConfig
Qwen3DecoderConfig::FromJson(const nlohmann::json &text_config) {
  return {text_config.at("num_hidden_layers").get<int>(),
          text_config.at("num_attention_heads").get<int>(),
          text_config.at("num_key_value_heads").get<int>(),
          text_config.at("head_dim").get<int>(),
          text_config.at("rms_norm_eps").get<float>(),
          text_config.at("rope_theta").get<float>()};
}

Qwen3Decoder::Qwen3Decoder(const omni_common::Checkpoint &checkpoint,
                           std::string prefix, Qwen3DecoderConfig config)
    : checkpoint_(checkpoint), prefix_(std::move(prefix)), config_(config) {}

mx::array Qwen3Decoder::EmbedTokens(const mx::array &ids) const {
  return checkpoint_.Embed(ids, prefix_ + ".embed_tokens");
}

mx::array Qwen3Decoder::Layer(const mx::array &x, int layer,
                              omni_common::KVCache &cache) const {
  const std::string prefix = prefix_ + ".layers." + std::to_string(layer);
  const float eps = config_.rms_norm_eps;
  const int batch = x.shape(0);
  const int length = x.shape(1);
  const int head_count = config_.num_attention_heads;
  const int kv_head_count = config_.num_key_value_heads;
  const int head_dim = config_.head_dim;
  const mx::array normed =
      checkpoint_.RmsNorm(x, prefix + ".input_layernorm", eps);
  mx::array queries =
      mx::reshape(checkpoint_.Linear(normed, prefix + ".self_attn.q_proj"),
                  {batch, length, head_count, head_dim});
  mx::array keys =
      mx::reshape(checkpoint_.Linear(normed, prefix + ".self_attn.k_proj"),
                  {batch, length, kv_head_count, head_dim});
  if (config_.has_query_key_norm) {
    queries = checkpoint_.RmsNorm(queries, prefix + ".self_attn.q_norm", eps);
    keys = checkpoint_.RmsNorm(keys, prefix + ".self_attn.k_norm", eps);
  } else {
  }
  const mx::array values =
      mx::reshape(checkpoint_.Linear(normed, prefix + ".self_attn.v_proj"),
                  {batch, length, kv_head_count, head_dim});
  queries = mx::fast::rope(mx::transpose(queries, {0, 2, 1, 3}), head_dim,
                           config_.traditional_rope, config_.rope_theta, 1.0f,
                           cache.offset());
  keys = mx::fast::rope(mx::transpose(keys, {0, 2, 1, 3}), head_dim,
                        config_.traditional_rope, config_.rope_theta, 1.0f,
                        cache.offset());
  auto [cached_keys, cached_values] =
      cache.UpdateAndFetch(keys, mx::transpose(values, {0, 2, 1, 3}));
  const mx::array attended = mx::fast::scaled_dot_product_attention(
      queries, cached_keys, cached_values,
      static_cast<float>(std::pow(static_cast<double>(head_dim), -0.5)),
      length > 1 ? "causal" : "");
  const mx::array hidden = mx::add(
      x, checkpoint_.Linear(mx::reshape(mx::transpose(attended, {0, 2, 1, 3}),
                                        {batch, length, -1}),
                            prefix + ".self_attn.o_proj"));
  const mx::array mlp_input =
      checkpoint_.RmsNorm(hidden, prefix + ".post_attention_layernorm", eps);
  return mx::add(
      hidden,
      checkpoint_.Linear(
          mx::multiply(omni_common::Silu(checkpoint_.Linear(
                           mlp_input, prefix + ".mlp.gate_proj")),
                       checkpoint_.Linear(mlp_input, prefix + ".mlp.up_proj")),
          prefix + ".mlp.down_proj"));
}

mx::array
Qwen3Decoder::Forward(const mx::array &embeddings,
                      std::vector<omni_common::KVCache> &caches) const {
  mx::array hidden = embeddings;
  for (int layer = 0; layer < config_.num_hidden_layers; ++layer) {
    hidden = Layer(hidden, layer, caches[layer]);
  }
  return hidden;
}

mx::array Qwen3Decoder::LastLogits(const mx::array &hidden_states) const {
  const int length = hidden_states.shape(1);
  const mx::array last = checkpoint_.RmsNorm(
      mx::slice(hidden_states, {0, length - 1, 0},
                {hidden_states.shape(0), length, hidden_states.shape(2)}),
      prefix_ + ".norm", config_.rms_norm_eps);
  const mx::array logits =
      checkpoint_.TiedProjection(last, prefix_ + ".embed_tokens");
  return mx::reshape(logits, {logits.shape(-1)});
}

std::vector<omni_common::KVCache> Qwen3Decoder::NewCaches() const {
  return std::vector<omni_common::KVCache>(config_.num_hidden_layers);
}

} // namespace qwen3_asr
