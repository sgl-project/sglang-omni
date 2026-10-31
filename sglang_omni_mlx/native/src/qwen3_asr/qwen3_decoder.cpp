// SPDX-License-Identifier: Apache-2.0
#include "qwen3_asr/qwen3_decoder.h"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace qwen3_asr {

namespace mx = mlx::core;

namespace {

constexpr int kKvCacheStepTokens = 256;

} // namespace

Qwen3DecoderConfig
Qwen3DecoderConfig::FromJson(const nlohmann::json &text_config) {
  // note (wilsonzheng0327): newer configs keep rope_theta under
  // rope_parameters.
  const nlohmann::json &rope = text_config.contains("rope_parameters")
                                   ? text_config.at("rope_parameters")
                                   : text_config;
  Qwen3DecoderConfig config{text_config.at("num_hidden_layers").get<int>(),
                            text_config.at("num_attention_heads").get<int>(),
                            text_config.at("num_key_value_heads").get<int>(),
                            text_config.at("head_dim").get<int>(),
                            text_config.at("rms_norm_eps").get<float>(),
                            rope.at("rope_theta").get<float>()};
  config.hidden_size = text_config.at("hidden_size").get<int>();
  config.vocab_size = text_config.at("vocab_size").get<int>();
  return config;
}

std::pair<mx::array, mx::array>
KVCache::UpdateAndFetch(const mx::array &keys, const mx::array &values) {
  const int new_token_count = keys.shape(2);
  if (!keys_.has_value() || offset_ + new_token_count > keys_->shape(2)) {
    const int step_count =
        (new_token_count + kKvCacheStepTokens - 1) / kKvCacheStepTokens;
    const mx::Shape grown = {keys.shape(0), keys.shape(1),
                             step_count * kKvCacheStepTokens, keys.shape(3)};
    const mx::array extra_keys = mx::zeros(grown, keys.dtype());
    const mx::array extra_values = mx::zeros(grown, values.dtype());
    if (!keys_.has_value()) {
      keys_ = extra_keys;
      values_ = extra_values;
    } else {
      const mx::Shape kept = {keys_->shape(0), keys_->shape(1), offset_,
                              keys_->shape(3)};
      keys_ = mx::concatenate(
          {mx::slice(*keys_, {0, 0, 0, 0}, kept), extra_keys}, 2);
      values_ = mx::concatenate(
          {mx::slice(*values_, {0, 0, 0, 0}, kept), extra_values}, 2);
    }
  } else {
  }
  const mx::Shape start = {0, 0, offset_, 0};
  const mx::Shape stop = {keys.shape(0), keys.shape(1),
                          offset_ + new_token_count, keys.shape(3)};
  keys_ = mx::slice_update(*keys_, keys, start, stop);
  values_ = mx::slice_update(*values_, values, start, stop);
  offset_ += new_token_count;
  const mx::Shape fetched = {keys_->shape(0), keys_->shape(1), offset_,
                             keys_->shape(3)};
  return {mx::slice(*keys_, {0, 0, 0, 0}, fetched),
          mx::slice(*values_, {0, 0, 0, 0}, fetched)};
}

void KVCache::Trim(int token_count) {
  assert(token_count <= offset_);
  offset_ -= token_count;
}

Qwen3Decoder::Qwen3Decoder(const omni_common::Checkpoint &checkpoint,
                           std::string prefix, Qwen3DecoderConfig config)
    : checkpoint_(checkpoint), prefix_(std::move(prefix)), config_(config) {}

std::string Qwen3Decoder::Name(const std::string &suffix) const {
  return prefix_.empty() ? suffix : prefix_ + "." + suffix;
}

mx::array Qwen3Decoder::EmbedTokens(const mx::array &ids) const {
  return checkpoint_.Embed(ids, Name("embed_tokens"));
}

mx::array Qwen3Decoder::Layer(const mx::array &x, int layer,
                              KVCache &cache) const {
  const std::string prefix = Name("layers." + std::to_string(layer));
  const float eps = config_.rms_norm_eps;
  const int batch = x.shape(0);
  const int length = x.shape(1);
  const int head_count = config_.num_attention_heads;
  const int head_dim = config_.head_dim;
  const mx::array normed =
      checkpoint_.RmsNorm(x, prefix + ".input_layernorm", eps);
  mx::array queries =
      mx::reshape(checkpoint_.Linear(normed, prefix + ".self_attn.q_proj"),
                  {batch, length, head_count, head_dim});
  if (config_.has_query_key_norm) {
    queries = checkpoint_.RmsNorm(queries, prefix + ".self_attn.q_norm", eps);
  } else {
  }
  queries = mx::fast::rope(mx::transpose(queries, {0, 2, 1, 3}), head_dim,
                           config_.traditional_rope, config_.rope_theta, 1.0f,
                           cache.offset());
  const auto [keys, values] = KeysValues(normed, prefix, cache);
  auto [cached_keys, cached_values] = cache.UpdateAndFetch(keys, values);
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

mx::array Qwen3Decoder::Forward(const mx::array &embeddings,
                                std::vector<KVCache> &caches,
                                const std::vector<int> &capture_layers,
                                std::vector<mx::array> *captured) const {
  mx::array hidden = embeddings;
  for (int layer = 0; layer < config_.num_hidden_layers; ++layer) {
    if (std::count(capture_layers.begin(), capture_layers.end(), layer) > 0) {
      captured->push_back(hidden);
    } else {
    }
    hidden = Layer(hidden, layer, caches[layer]);
  }
  return hidden;
}

std::pair<mx::array, mx::array>
Qwen3Decoder::KeysValues(const mx::array &x, const std::string &prefix,
                         const KVCache &cache) const {
  const int batch = x.shape(0);
  const int length = x.shape(1);
  const int kv_head_count = config_.num_key_value_heads;
  const int head_dim = config_.head_dim;
  mx::array keys =
      mx::reshape(checkpoint_.Linear(x, prefix + ".self_attn.k_proj"),
                  {batch, length, kv_head_count, head_dim});
  if (config_.has_query_key_norm) {
    keys = checkpoint_.RmsNorm(keys, prefix + ".self_attn.k_norm",
                               config_.rms_norm_eps);
  } else {
  }
  keys = mx::fast::rope(mx::transpose(keys, {0, 2, 1, 3}), head_dim,
                        config_.traditional_rope, config_.rope_theta, 1.0f,
                        cache.offset());
  const mx::array values =
      mx::reshape(checkpoint_.Linear(x, prefix + ".self_attn.v_proj"),
                  {batch, length, kv_head_count, head_dim});
  return {keys, mx::transpose(values, {0, 2, 1, 3})};
}

void Qwen3Decoder::AppendKeysValues(const mx::array &hidden_states,
                                    std::vector<KVCache> &caches) const {
  for (int layer = 0; layer < config_.num_hidden_layers; ++layer) {
    const auto [keys, values] = KeysValues(
        hidden_states, Name("layers." + std::to_string(layer)), caches[layer]);
    caches[layer].UpdateAndFetch(keys, values);
  }
}

mx::array Qwen3Decoder::LastLogits(const mx::array &hidden_states) const {
  const int length = hidden_states.shape(1);
  const mx::array last = checkpoint_.RmsNorm(
      mx::slice(hidden_states, {0, length - 1, 0},
                {hidden_states.shape(0), length, hidden_states.shape(2)}),
      Name("norm"), config_.rms_norm_eps);
  const mx::array logits =
      checkpoint_.TiedProjection(last, Name("embed_tokens"));
  return mx::reshape(logits, {logits.shape(-1)});
}

mx::array Qwen3Decoder::Logits(const mx::array &hidden_states) const {
  const mx::array logits = checkpoint_.TiedProjection(
      checkpoint_.RmsNorm(hidden_states, Name("norm"), config_.rms_norm_eps),
      Name("embed_tokens"));
  return mx::reshape(logits, {hidden_states.shape(1), logits.shape(-1)});
}

std::vector<KVCache> Qwen3Decoder::NewCaches() const {
  return std::vector<KVCache>(config_.num_hidden_layers);
}

} // namespace qwen3_asr
