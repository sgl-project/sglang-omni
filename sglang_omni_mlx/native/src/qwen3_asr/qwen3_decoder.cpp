// SPDX-License-Identifier: Apache-2.0
#include "qwen3_asr/qwen3_decoder.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace qwen3_asr {

namespace mx = mlx::core;

KVCache::KVCache(int step_tokens) : step_tokens_(step_tokens) {
  if (step_tokens <= 0) {
    throw std::invalid_argument("KV cache step_tokens must be positive");
  } else {
  }
}

std::vector<mx::array> KVCache::Arrays() const {
  if (keys_.has_value()) {
    return {*keys_, *values_};
  } else {
    return {};
  }
}

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
  if (new_token_count > std::numeric_limits<int>::max() - offset_) {
    throw std::overflow_error("KV cache token count overflow");
  } else {
  }
  const auto positions = [](const mx::array &heads, int token_count) {
    return mx::Shape{heads.shape(0), heads.shape(1), token_count,
                     heads.shape(3)};
  };
  if (!keys_.has_value() || offset_ + new_token_count > keys_->shape(2)) {
    const int64_t rounded_tokens =
        (static_cast<int64_t>(new_token_count) + step_tokens_ - 1) /
        step_tokens_ * step_tokens_;
    if (rounded_tokens > std::numeric_limits<int>::max() - offset_) {
      throw std::overflow_error("KV cache capacity overflow");
    } else {
    }
    const int grown_token_count = static_cast<int>(rounded_tokens);
    const mx::array extra_keys =
        mx::zeros(positions(keys, grown_token_count), keys.dtype());
    const mx::array extra_values =
        mx::zeros(positions(values, grown_token_count), values.dtype());
    if (!keys_.has_value()) {
      keys_ = extra_keys;
      values_ = extra_values;
    } else {
      keys_ = mx::concatenate(
          {mx::slice(*keys_, {0, 0, 0, 0}, positions(*keys_, offset_)),
           extra_keys},
          2);
      values_ = mx::concatenate(
          {mx::slice(*values_, {0, 0, 0, 0}, positions(*values_, offset_)),
           extra_values},
          2);
    }
  } else {
  }
  const mx::Shape start = {0, 0, offset_, 0};
  keys_ = mx::slice_update(*keys_, keys, start,
                           positions(keys, offset_ + new_token_count));
  values_ = mx::slice_update(*values_, values, start,
                             positions(values, offset_ + new_token_count));
  offset_ += new_token_count;
  return {mx::slice(*keys_, {0, 0, 0, 0}, positions(*keys_, offset_)),
          mx::slice(*values_, {0, 0, 0, 0}, positions(*values_, offset_))};
}

void KVCache::Trim(int token_count) {
  assert(token_count <= offset_);
  offset_ -= token_count;
}

std::pair<mx::array, mx::array> KVCache::Filled() const {
  const mx::Shape stop = {keys_->shape(0), keys_->shape(1), offset_,
                          keys_->shape(3)};
  return {mx::slice(*keys_, {0, 0, 0, 0}, stop),
          mx::slice(*values_, {0, 0, 0, 0}, stop)};
}

namespace {

// Left-pads each row to the longest and stacks the rows.
mx::array StackRows(const std::vector<mx::array> &rows) {
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
  return mx::concatenate(padded, 0);
}

} // namespace

BatchedKVCache::BatchedKVCache(
    const std::vector<std::pair<mx::array, mx::array>> &rows)
    : keys_(StackRows([&] {
        std::vector<mx::array> keys;
        for (const auto &row : rows) {
          keys.push_back(row.first);
        }
        return keys;
      }())),
      values_(StackRows([&] {
        std::vector<mx::array> values;
        for (const auto &row : rows) {
          values.push_back(row.second);
        }
        return values;
      }())) {
  int longest = 0;
  for (const auto &row : rows) {
    longest = std::max(longest, row.first.shape(2));
  }
  end_ = longest;
}

std::pair<mx::array, mx::array>
BatchedKVCache::Append(const mx::array &keys, const mx::array &values) {
  // Grown a step at a time, like KVCache, so the reservation follows the
  // tokens decoded rather than the largest budget in the batch.
  if (end_ + 1 > capacity()) {
    const auto grown = [](const mx::array &heads) {
      return mx::concatenate(
          {heads, mx::zeros({heads.shape(0), heads.shape(1), kKvCacheStepTokens,
                             heads.shape(3)},
                            heads.dtype())},
          2);
    };
    keys_ = grown(keys_);
    values_ = grown(values_);
  } else {
  }
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

std::string Qwen3Decoder::Name(const std::string &suffix) const {
  return prefix_.empty() ? suffix : prefix_ + "." + suffix;
}

mx::array Qwen3Decoder::EmbedTokens(const mx::array &ids) const {
  return checkpoint_.Embed(ids, Name("embed_tokens"));
}

mx::array Qwen3Decoder::Layer(const mx::array &x, int layer, KVCache &cache,
                              const std::optional<RotaryInputs> &rotary) const {
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
  queries = Rotated(mx::transpose(queries, {0, 2, 1, 3}), cache, rotary);
  const auto [keys, values] = KeysValues(normed, prefix, cache, rotary);
  auto [cached_keys, cached_values] = cache.UpdateAndFetch(keys, values);
  const float scale =
      static_cast<float>(std::pow(static_cast<double>(head_dim), -0.5));
  std::optional<mx::array> attended;
  if (rotary.has_value() && rotary->mask.has_value()) {
    attended = mx::fast::scaled_dot_product_attention(
        queries, cached_keys, cached_values, scale, "", *rotary->mask);
  } else if (rotary.has_value()) {
    attended = mx::fast::scaled_dot_product_attention(queries, cached_keys,
                                                      cached_values, scale);
  } else {
    attended = mx::fast::scaled_dot_product_attention(
        queries, cached_keys, cached_values, scale, length > 1 ? "causal" : "");
  }
  const mx::array hidden = mx::add(
      x, checkpoint_.Linear(mx::reshape(mx::transpose(*attended, {0, 2, 1, 3}),
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
  const std::optional<RotaryInputs> rotary =
      config_.rotary_tables
          ? std::optional<RotaryInputs>(Rotary(
                embeddings.shape(1), caches[0].offset(), embeddings.dtype()))
          : std::nullopt;
  mx::array hidden = embeddings;
  // Note (Yucheng Hu): MLX frees temporaries per buffer, so eval per layer.
  const bool is_prefill = embeddings.shape(1) > 1;
  for (int layer = 0; layer < config_.num_hidden_layers; ++layer) {
    if (std::count(capture_layers.begin(), capture_layers.end(), layer) > 0) {
      captured->push_back(hidden);
    } else {
    }
    const mx::array layer_input = hidden;
    hidden = Layer(hidden, layer, caches[layer], rotary);
    if (is_prefill) {
      mx::async_eval(hidden);
      mx::eval(layer_input);
    } else {
    }
  }
  return hidden;
}

std::pair<mx::array, mx::array>
Qwen3Decoder::KeysValues(const mx::array &x, const std::string &prefix,
                         const KVCache &cache,
                         const std::optional<RotaryInputs> &rotary) const {
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
  keys = Rotated(mx::transpose(keys, {0, 2, 1, 3}), cache, rotary);
  const mx::array values =
      mx::reshape(checkpoint_.Linear(x, prefix + ".self_attn.v_proj"),
                  {batch, length, kv_head_count, head_dim});
  return {keys, mx::transpose(values, {0, 2, 1, 3})};
}

void Qwen3Decoder::AppendKeysValues(const mx::array &hidden_states,
                                    std::vector<KVCache> &caches) const {
  for (int layer = 0; layer < config_.num_hidden_layers; ++layer) {
    const auto [keys, values] =
        KeysValues(hidden_states, Name("layers." + std::to_string(layer)),
                   caches[layer], std::nullopt);
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

Qwen3Decoder::RotaryInputs Qwen3Decoder::Rotary(int length, int offset,
                                                mx::Dtype dtype) const {
  const int head_dim = config_.head_dim;
  const mx::array inverse_frequencies = mx::divide(
      mx::array(1.0f),
      mx::power(mx::array(config_.rope_theta),
                mx::divide(mx::arange(0.0, static_cast<double>(head_dim), 2.0,
                                      mx::float32),
                           mx::array(static_cast<float>(head_dim)))));
  const mx::array positions =
      mx::astype(mx::arange(offset, offset + length), mx::float32);
  const mx::array frequencies =
      mx::transpose(mx::multiply(mx::reshape(inverse_frequencies, {1, -1, 1}),
                                 mx::reshape(positions, {1, 1, -1})),
                    {0, 2, 1});
  const mx::array angles = mx::concatenate({frequencies, frequencies}, -1);
  RotaryInputs rotary{mx::expand_dims(mx::astype(mx::cos(angles), dtype), 1),
                      mx::expand_dims(mx::astype(mx::sin(angles), dtype), 1),
                      std::nullopt};
  if (length > 1) {
    // Each query sees the cached keys and the step's keys up to its own.
    const mx::array query_positions =
        mx::expand_dims(mx::arange(offset, offset + length), 1);
    const mx::array key_positions =
        mx::expand_dims(mx::arange(offset + length), 0);
    rotary.mask = mx::astype(
        mx::multiply(
            mx::astype(mx::less(query_positions, key_positions), mx::float32),
            mx::array(std::numeric_limits<float>::lowest())),
        dtype);
  } else {
  }
  return rotary;
}

mx::array
Qwen3Decoder::Rotated(const mx::array &heads, const KVCache &cache,
                      const std::optional<RotaryInputs> &rotary) const {
  if (rotary.has_value()) {
    const int half = config_.head_dim / 2;
    const mx::Shape &shape = heads.shape();
    const mx::array first =
        mx::slice(heads, {0, 0, 0, 0}, {shape[0], shape[1], shape[2], half});
    const mx::array second = mx::slice(heads, {0, 0, 0, half}, shape);
    return mx::add(
        mx::multiply(heads, rotary->cosines),
        mx::multiply(mx::concatenate({mx::negative(second), first}, -1),
                     rotary->sines));
  } else {
    return mx::fast::rope(heads, config_.head_dim, config_.traditional_rope,
                          config_.rope_theta, 1.0f, cache.offset());
  }
}

mx::array Qwen3Decoder::Logits(const mx::array &hidden_states) const {
  const mx::array logits = checkpoint_.TiedProjection(
      checkpoint_.RmsNorm(hidden_states, Name("norm"), config_.rms_norm_eps),
      Name("embed_tokens"));
  return mx::reshape(logits, {hidden_states.shape(1), logits.shape(-1)});
}

mx::array Qwen3Decoder::BatchLayer(const mx::array &x, int layer,
                                   BatchedKVCache &cache,
                                   const mx::array &offsets,
                                   const mx::array &mask) const {
  const std::string prefix = Name("layers." + std::to_string(layer));
  const float eps = config_.rms_norm_eps;
  const int batch = x.shape(0);
  const int head_count = config_.num_attention_heads;
  const int kv_head_count = config_.num_key_value_heads;
  const int head_dim = config_.head_dim;
  const mx::array normed =
      checkpoint_.RmsNorm(x, prefix + ".input_layernorm", eps);
  mx::array queries =
      mx::reshape(checkpoint_.Linear(normed, prefix + ".self_attn.q_proj"),
                  {batch, 1, head_count, head_dim});
  mx::array keys =
      mx::reshape(checkpoint_.Linear(normed, prefix + ".self_attn.k_proj"),
                  {batch, 1, kv_head_count, head_dim});
  if (config_.has_query_key_norm) {
    queries = checkpoint_.RmsNorm(queries, prefix + ".self_attn.q_norm", eps);
    keys = checkpoint_.RmsNorm(keys, prefix + ".self_attn.k_norm", eps);
  } else {
  }
  const mx::array values =
      mx::reshape(checkpoint_.Linear(normed, prefix + ".self_attn.v_proj"),
                  {batch, 1, kv_head_count, head_dim});
  queries = mx::fast::rope(mx::transpose(queries, {0, 2, 1, 3}), head_dim,
                           config_.traditional_rope, config_.rope_theta, 1.0f,
                           offsets);
  keys = mx::fast::rope(mx::transpose(keys, {0, 2, 1, 3}), head_dim,
                        config_.traditional_rope, config_.rope_theta, 1.0f,
                        offsets);
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
                                    const mx::array &pads) const {
  // The step appends one slot to every layer; attention covers them all, bar
  // each row's left padding.
  const int attended = caches[0].length() + 1;
  const mx::array mask =
      mx::reshape(mx::greater_equal(mx::arange(attended, mx::int32), pads),
                  {pads.shape(0), 1, 1, attended});
  mx::array hidden = embeddings;
  for (int layer = 0; layer < config_.num_hidden_layers; ++layer) {
    hidden = BatchLayer(hidden, layer, caches[layer], offsets, mask);
  }
  const mx::array normed =
      checkpoint_.RmsNorm(hidden, Name("norm"), config_.rms_norm_eps);
  const mx::array logits =
      checkpoint_.TiedProjection(normed, Name("embed_tokens"));
  return mx::reshape(logits, {logits.shape(0), logits.shape(-1)});
}

std::vector<KVCache> Qwen3Decoder::NewCaches() const {
  return std::vector<KVCache>(config_.num_hidden_layers);
}

std::vector<KVCache> Qwen3Decoder::NewCaches(int step_tokens) const {
  return std::vector<KVCache>(config_.num_hidden_layers, KVCache(step_tokens));
}

} // namespace qwen3_asr
