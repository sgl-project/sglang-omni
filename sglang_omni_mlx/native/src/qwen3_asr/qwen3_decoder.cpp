// SPDX-License-Identifier: Apache-2.0
#include "qwen3_asr/qwen3_decoder.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

namespace qwen3_asr {

namespace mx = mlx::core;

namespace {

constexpr int kKvCacheStepTokens = 256;

} // namespace

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
  const auto positions = [](const mx::array &heads, int token_count) {
    return mx::Shape{heads.shape(0), heads.shape(1), token_count,
                     heads.shape(3)};
  };
  if (!keys_.has_value() || offset_ + new_token_count > keys_->shape(2)) {
    const int grown_token_count = (new_token_count + kKvCacheStepTokens - 1) /
                                  kKvCacheStepTokens * kKvCacheStepTokens;
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

Qwen3Decoder::Qwen3Decoder(const omni_common::Checkpoint &checkpoint,
                           std::string prefix, Qwen3DecoderConfig config)
    : checkpoint_(checkpoint), prefix_(std::move(prefix)), config_(config) {}

std::string Qwen3Decoder::Name(const std::string &suffix) const {
  return prefix_.empty() ? suffix : prefix_ + "." + suffix;
}

void Qwen3Decoder::PackDenseGateUp() {
  std::vector<mx::array> packed;
  for (int layer = 0; layer < config_.num_hidden_layers; ++layer) {
    const std::string prefix =
        Name("layers." + std::to_string(layer) + ".mlp.");
    const std::string gate = prefix + "gate_proj";
    const std::string up = prefix + "up_proj";
    if (checkpoint_.Has(gate + ".scales") || checkpoint_.Has(up + ".scales") ||
        checkpoint_.Has(gate + ".bias") || checkpoint_.Has(up + ".bias")) {
      return;
    } else {
    }
    packed.push_back(mx::concatenate({checkpoint_.Weight(gate + ".weight"),
                                      checkpoint_.Weight(up + ".weight")},
                                     0));
  }
  mx::eval(packed);
  gate_up_weights_ = std::move(packed);
}

mx::array Qwen3Decoder::EmbedTokens(const mx::array &ids) const {
  return checkpoint_.Embed(ids, Name("embed_tokens"));
}

void Qwen3Decoder::CacheRotary(int token_count, mx::Dtype dtype) {
  std::map<std::pair<int, int>, RotaryInputs> prepared;
  std::vector<mx::array> buffers;
  const auto prepare = [&](int length, int offset) {
    RotaryInputs rotary = Rotary(length, offset, dtype);
    buffers.push_back(rotary.cosines);
    buffers.push_back(rotary.sines);
    if (rotary.mask) {
      buffers.push_back(*rotary.mask);
    } else {
    }
    prepared.emplace(std::pair{length, offset}, std::move(rotary));
  };
  prepare(2, 0);
  for (int offset = 2; offset < token_count; ++offset) {
    prepare(1, offset);
  }
  mx::eval(buffers);
  cached_rotary_inputs_ = std::move(prepared);
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
  const mx::array activated = [&] {
    if (gate_up_weights_.empty()) {
      return mx::multiply(
          omni_common::Silu(
              checkpoint_.Linear(mlp_input, prefix + ".mlp.gate_proj")),
          checkpoint_.Linear(mlp_input, prefix + ".mlp.up_proj"));
    } else {
      const auto halves = mx::split(
          mx::matmul(mlp_input, mx::transpose(gate_up_weights_[layer])), 2, -1);
      return mx::multiply(omni_common::Silu(halves[0]), halves[1]);
    }
  }();
  return mx::add(hidden,
                 checkpoint_.Linear(activated, prefix + ".mlp.down_proj"));
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
  const auto found = cached_rotary_inputs_.find({length, offset});
  if (found != cached_rotary_inputs_.end() &&
      found->second.cosines.dtype() == dtype) {
    return found->second;
  } else {
  }
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

std::vector<KVCache> Qwen3Decoder::NewCaches() const {
  return std::vector<KVCache>(config_.num_hidden_layers);
}

} // namespace qwen3_asr
