// SPDX-License-Identifier: Apache-2.0
#include "qwen3_asr/qwen3_decoder.h"

#include <algorithm>
#include <cmath>

namespace qwen3_asr {

namespace mx = mlx::core;

namespace {

constexpr int kKvCacheStepTokens = 256;

} // namespace

Qwen3DecoderConfig
Qwen3DecoderConfig::FromJson(const nlohmann::json &text_config) {
  return {text_config.at("num_hidden_layers").get<int>(),
          text_config.at("num_attention_heads").get<int>(),
          text_config.at("num_key_value_heads").get<int>(),
          text_config.at("head_dim").get<int>(),
          text_config.at("rms_norm_eps").get<float>(),
          text_config.at("rope_theta").get<float>()};
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

std::pair<mx::array, mx::array> KVCache::Filled() const {
  const mx::Shape stop = {keys_->shape(0), keys_->shape(1), offset_,
                          keys_->shape(3)};
  return {mx::slice(*keys_, {0, 0, 0, 0}, stop),
          mx::slice(*values_, {0, 0, 0, 0}, stop)};
}

namespace {

// Left-pads each row to the longest, stacks the rows and appends room.
mx::array StackRows(const std::vector<mx::array> &rows, int extra_tokens) {
  int longest = 0;
  for (const mx::array &row : rows) {
    longest = std::max(longest, row.shape(2));
  }
  std::vector<mx::array> padded;
  for (const mx::array &row : rows) {
    const int pad = longest - row.shape(2);
    if (pad > 0) {
      padded.push_back(mx::concatenate(
          {mx::zeros({1, row.shape(1), pad, row.shape(3)}, row.dtype()), row},
          2));
    } else {
      padded.push_back(row);
    }
  }
  const mx::array stacked = mx::concatenate(padded, 0);
  return mx::concatenate(
      {stacked, mx::zeros({stacked.shape(0), stacked.shape(1), extra_tokens,
                           stacked.shape(3)},
                          stacked.dtype())},
      2);
}

} // namespace

BatchedKVCache::BatchedKVCache(
    const std::vector<std::pair<mx::array, mx::array>> &rows, int extra_tokens)
    : keys_(StackRows(
          [&] {
            std::vector<mx::array> keys;
            for (const auto &row : rows) {
              keys.push_back(row.first);
            }
            return keys;
          }(),
          extra_tokens)),
      values_(StackRows(
          [&] {
            std::vector<mx::array> values;
            for (const auto &row : rows) {
              values.push_back(row.second);
            }
            return values;
          }(),
          extra_tokens)) {
  int longest = 0;
  for (const auto &row : rows) {
    longest = std::max(longest, row.first.shape(2));
  }
  end_ = longest;
}

std::pair<mx::array, mx::array>
BatchedKVCache::Append(const mx::array &keys, const mx::array &values) {
  const mx::Shape start = {0, 0, end_, 0};
  const mx::Shape stop = {keys.shape(0), keys.shape(1), end_ + 1,
                          keys.shape(3)};
  keys_ = mx::slice_update(keys_, keys, start, stop);
  values_ = mx::slice_update(values_, values, start, stop);
  end_ += 1;
  const mx::Shape fetched = {keys_.shape(0), keys_.shape(1), end_,
                             keys_.shape(3)};
  return {mx::slice(keys_, {0, 0, 0, 0}, fetched),
          mx::slice(values_, {0, 0, 0, 0}, fetched)};
}

Qwen3Decoder::Qwen3Decoder(const omni_common::Checkpoint &checkpoint,
                           std::string prefix, Qwen3DecoderConfig config)
    : checkpoint_(checkpoint), prefix_(std::move(prefix)), config_(config) {}

mx::array Qwen3Decoder::EmbedTokens(const mx::array &ids) const {
  return checkpoint_.Embed(ids, prefix_ + ".embed_tokens");
}

mx::array Qwen3Decoder::Layer(const mx::array &x, int layer,
                              KVCache &cache) const {
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

mx::array Qwen3Decoder::Forward(const mx::array &embeddings,
                                std::vector<KVCache> &caches) const {
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

mx::array Qwen3Decoder::BatchLayer(const mx::array &x, int layer,
                                   BatchedKVCache &cache,
                                   const mx::array &offsets,
                                   const mx::array &mask) const {
  const std::string prefix = prefix_ + ".layers." + std::to_string(layer);
  const float eps = config_.rms_norm_eps;
  const int batch = x.shape(0);
  const int head_count = config_.num_attention_heads;
  const int kv_head_count = config_.num_key_value_heads;
  const int head_dim = config_.head_dim;
  const mx::array normed =
      checkpoint_.RmsNorm(x, prefix + ".input_layernorm", eps);
  mx::array queries = checkpoint_.RmsNorm(
      mx::reshape(checkpoint_.Linear(normed, prefix + ".self_attn.q_proj"),
                  {batch, 1, head_count, head_dim}),
      prefix + ".self_attn.q_norm", eps);
  mx::array keys = checkpoint_.RmsNorm(
      mx::reshape(checkpoint_.Linear(normed, prefix + ".self_attn.k_proj"),
                  {batch, 1, kv_head_count, head_dim}),
      prefix + ".self_attn.k_norm", eps);
  const mx::array values =
      mx::reshape(checkpoint_.Linear(normed, prefix + ".self_attn.v_proj"),
                  {batch, 1, kv_head_count, head_dim});
  queries = mx::fast::rope(mx::transpose(queries, {0, 2, 1, 3}), head_dim,
                           false, config_.rope_theta, 1.0f, offsets);
  keys = mx::fast::rope(mx::transpose(keys, {0, 2, 1, 3}), head_dim, false,
                        config_.rope_theta, 1.0f, offsets);
  auto [cached_keys, cached_values] =
      cache.Append(keys, mx::transpose(values, {0, 2, 1, 3}));
  const mx::array attended = mx::fast::scaled_dot_product_attention(
      queries, cached_keys, cached_values,
      static_cast<float>(std::pow(static_cast<double>(head_dim), -0.5)),
      "array", mask);
  const mx::array hidden = mx::add(
      x, checkpoint_.Linear(
             mx::reshape(mx::transpose(attended, {0, 2, 1, 3}), {batch, 1, -1}),
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

mx::array Qwen3Decoder::DecodeBatch(const mx::array &embeddings,
                                    std::vector<BatchedKVCache> &caches,
                                    const mx::array &offsets,
                                    const mx::array &valid) const {
  // The step appends one slot to every layer; attention covers them all.
  const int attended = caches[0].length() + 1;
  const mx::array mask =
      mx::slice(valid, {0, 0, 0, 0}, {valid.shape(0), 1, 1, attended});
  mx::array hidden = embeddings;
  for (int layer = 0; layer < config_.num_hidden_layers; ++layer) {
    hidden = BatchLayer(hidden, layer, caches[layer], offsets, mask);
  }
  const mx::array normed =
      checkpoint_.RmsNorm(hidden, prefix_ + ".norm", config_.rms_norm_eps);
  const mx::array logits =
      checkpoint_.TiedProjection(normed, prefix_ + ".embed_tokens");
  return mx::reshape(logits, {logits.shape(0), logits.shape(-1)});
}

std::vector<KVCache> Qwen3Decoder::NewCaches() const {
  return std::vector<KVCache>(config_.num_hidden_layers);
}

} // namespace qwen3_asr
