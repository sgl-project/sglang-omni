// SPDX-License-Identifier: Apache-2.0
#include "qwen3_asr/qwen3_decoder.h"

#include <cmath>
#include <limits>

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

Qwen3Decoder::Qwen3Decoder(const omni_common::Checkpoint &checkpoint,
                           std::string prefix, Qwen3DecoderConfig config)
    : checkpoint_(checkpoint), prefix_(std::move(prefix)), config_(config) {}

mx::array Qwen3Decoder::EmbedTokens(const mx::array &ids) const {
  return checkpoint_.Embed(ids, prefix_ + ".embed_tokens");
}

mx::array Qwen3Decoder::Layer(const mx::array &x, int layer, KVCache &cache,
                              const std::optional<RotaryInputs> &rotary) const {
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
  const float scale =
      static_cast<float>(std::pow(static_cast<double>(head_dim), -0.5));
  std::optional<mx::array> attended;
  if (rotary.has_value()) {
    const auto rotate = [&](const mx::array &heads) {
      const int half = head_dim / 2;
      const mx::Shape &shape = heads.shape();
      const mx::array first =
          mx::slice(heads, {0, 0, 0, 0}, {shape[0], shape[1], shape[2], half});
      const mx::array second = mx::slice(heads, {0, 0, 0, half}, shape);
      return mx::add(
          mx::multiply(heads, rotary->cosines),
          mx::multiply(mx::concatenate({mx::negative(second), first}, -1),
                       rotary->sines));
    };
    queries = rotate(mx::transpose(queries, {0, 2, 1, 3}));
    keys = rotate(mx::transpose(keys, {0, 2, 1, 3}));
    auto [cached_keys, cached_values] =
        cache.UpdateAndFetch(keys, mx::transpose(values, {0, 2, 1, 3}));
    if (rotary->mask.has_value()) {
      attended = mx::fast::scaled_dot_product_attention(
          queries, cached_keys, cached_values, scale, "", *rotary->mask);
    } else {
      attended = mx::fast::scaled_dot_product_attention(queries, cached_keys,
                                                        cached_values, scale);
    }
  } else {
    queries = mx::fast::rope(mx::transpose(queries, {0, 2, 1, 3}), head_dim,
                             config_.traditional_rope, config_.rope_theta, 1.0f,
                             cache.offset());
    keys = mx::fast::rope(mx::transpose(keys, {0, 2, 1, 3}), head_dim,
                          config_.traditional_rope, config_.rope_theta, 1.0f,
                          cache.offset());
    auto [cached_keys, cached_values] =
        cache.UpdateAndFetch(keys, mx::transpose(values, {0, 2, 1, 3}));
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
                                std::vector<KVCache> &caches) const {
  const std::optional<RotaryInputs> rotary =
      config_.rotary_tables
          ? std::optional<RotaryInputs>(Rotary(
                embeddings.shape(1), caches[0].offset(), embeddings.dtype()))
          : std::nullopt;
  mx::array hidden = embeddings;
  // Note (Yucheng Hu): MLX frees temporaries per buffer, so eval per layer.
  const bool is_prefill = embeddings.shape(1) > 1;
  std::optional<mx::array> previous_layer_output;
  for (int layer = 0; layer < config_.num_hidden_layers; ++layer) {
    hidden = Layer(hidden, layer, caches[layer], rotary);
    if (is_prefill) {
      mx::async_eval(hidden);
      if (previous_layer_output.has_value()) {
        previous_layer_output->wait();
      } else {
      }
      previous_layer_output = hidden;
    } else {
    }
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

std::vector<KVCache> Qwen3Decoder::NewCaches() const {
  return std::vector<KVCache>(config_.num_hidden_layers);
}

} // namespace qwen3_asr
