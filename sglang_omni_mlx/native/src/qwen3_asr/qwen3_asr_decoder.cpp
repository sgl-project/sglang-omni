// SPDX-License-Identifier: Apache-2.0
#include "qwen3_asr/qwen3_asr_decoder.h"

#include <cmath>

namespace qwen3_asr {

namespace mx = mlx::core;

namespace {

constexpr int kKvCacheStepTokens = 256;

} // namespace

std::pair<mx::array, mx::array>
Qwen3ASRKVCache::UpdateAndFetch(const mx::array &keys,
                                const mx::array &values) {
  const int new_token_count = keys.shape(2);
  const auto [key_cache, value_cache] = Reserve(keys.shape(), keys.dtype());
  const mx::Shape start = {0, 0, offset_, 0};
  const mx::Shape stop = {keys.shape(0), keys.shape(1),
                          offset_ + new_token_count, keys.shape(3)};
  return Commit(mx::slice_update(key_cache, keys, start, stop),
                mx::slice_update(value_cache, values, start, stop),
                new_token_count);
}

std::pair<mx::array, mx::array>
Qwen3ASRKVCache::Reserve(const mx::Shape &rows_shape, mx::Dtype dtype) {
  const int new_token_count = rows_shape[2];
  if (!keys_.has_value() || offset_ + new_token_count > keys_->shape(2)) {
    const int step_count =
        (new_token_count + kKvCacheStepTokens - 1) / kKvCacheStepTokens;
    const mx::Shape grown = {rows_shape[0], rows_shape[1],
                             step_count * kKvCacheStepTokens, rows_shape[3]};
    const mx::array extra_keys = mx::zeros(grown, dtype);
    const mx::array extra_values = mx::zeros(grown, dtype);
    if (!keys_.has_value()) {
      // Decode attention writes into the cache's buffer; zeros alone is a
      // broadcast.
      keys_ = mx::contiguous(extra_keys);
      values_ = mx::contiguous(extra_values);
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
  return {*keys_, *values_};
}

std::pair<mx::array, mx::array> Qwen3ASRKVCache::Commit(const mx::array &keys,
                                                        const mx::array &values,
                                                        int token_count) {
  keys_ = keys;
  values_ = values;
  offset_ += token_count;
  const mx::Shape fetched = {keys_->shape(0), keys_->shape(1), offset_,
                             keys_->shape(3)};
  return {mx::slice(*keys_, {0, 0, 0, 0}, fetched),
          mx::slice(*values_, {0, 0, 0, 0}, fetched)};
}

Qwen3ASRDecoder::Qwen3ASRDecoder(const omni_common::Checkpoint &checkpoint,
                                 std::string prefix, Qwen3DecoderConfig config)
    : checkpoint_(checkpoint), prefix_(std::move(prefix)), config_(config) {}

std::optional<std::vector<QuantizedWeights>> Qwen3ASRDecoder::QuantizedLayers(
    const std::vector<std::string> &prefixes) const {
  std::vector<QuantizedWeights> layers;
  for (const std::string &prefix : prefixes) {
    if (!checkpoint_.Has(prefix + ".biases") ||
        checkpoint_.Has(prefix + ".bias")) {
      return std::nullopt;
    } else {
      layers.push_back({checkpoint_.Weight(prefix + ".weight"),
                        checkpoint_.Weight(prefix + ".scales"),
                        checkpoint_.Weight(prefix + ".biases")});
    }
  }
  return layers;
}

std::vector<mx::array> Qwen3ASRDecoder::Projections(
    const mx::array &x, const std::vector<std::string> &prefixes,
    const std::optional<std::string> &norm_prefix) const {
  const omni_common::QuantizationConfig &quantization =
      checkpoint_.Quantization();
  const float eps = config_.rms_norm_eps;
  if (const auto layers = QuantizedLayers(prefixes)) {
    return QuantizedProjections(
        x, *layers, quantization.group_size, quantization.bits,
        norm_prefix.has_value()
            ? std::optional<InputNorm>(
                  {checkpoint_.Weight(*norm_prefix + ".weight"), eps})
            : std::nullopt);
  } else {
    const mx::array input =
        norm_prefix.has_value() ? checkpoint_.RmsNorm(x, *norm_prefix, eps) : x;
    std::vector<mx::array> outputs;
    for (const std::string &prefix : prefixes) {
      outputs.push_back(checkpoint_.Linear(input, prefix));
    }
    return outputs;
  }
}

mx::array Qwen3ASRDecoder::EmbedTokens(const mx::array &ids) const {
  return checkpoint_.Embed(ids, prefix_ + ".embed_tokens");
}

mx::array Qwen3ASRDecoder::Layer(const mx::array &x, int layer,
                                 Qwen3ASRKVCache &cache) const {
  const std::string prefix = prefix_ + ".layers." + std::to_string(layer);
  const float eps = config_.rms_norm_eps;
  const int batch = x.shape(0);
  const int length = x.shape(1);
  const int head_count = config_.num_attention_heads;
  const int kv_head_count = config_.num_key_value_heads;
  const int head_dim = config_.head_dim;
  const omni_common::QuantizationConfig &quantization =
      checkpoint_.Quantization();
  const std::vector<mx::array> projected =
      Projections(x,
                  {prefix + ".self_attn.q_proj", prefix + ".self_attn.k_proj",
                   prefix + ".self_attn.v_proj"},
                  prefix + ".input_layernorm");
  const float scale =
      static_cast<float>(std::pow(static_cast<double>(head_dim), -0.5));
  const mx::array attended = [&] {
    if (batch == 1 && length == 1 && head_dim == 128 &&
        DecodeAttentionFusable(projected[0], projected[1])) {
      const auto [key_cache, value_cache] = cache.Reserve(
          {batch, kv_head_count, length, head_dim}, projected[1].dtype());
      const std::vector<mx::array> attention = DecodeAttention(
          projected[0], projected[1], projected[2],
          checkpoint_.Weight(prefix + ".self_attn.q_norm.weight"),
          checkpoint_.Weight(prefix + ".self_attn.k_norm.weight"), key_cache,
          value_cache, cache.offset(), eps, config_.rope_theta, scale);
      cache.Commit(attention[1], attention[2], length);
      return attention[0];
    } else {
    }
    mx::array queries = checkpoint_.RmsNorm(
        mx::reshape(projected[0], {batch, length, head_count, head_dim}),
        prefix + ".self_attn.q_norm", eps);
    mx::array keys = checkpoint_.RmsNorm(
        mx::reshape(projected[1], {batch, length, kv_head_count, head_dim}),
        prefix + ".self_attn.k_norm", eps);
    const mx::array values =
        mx::reshape(projected[2], {batch, length, kv_head_count, head_dim});
    queries = mx::fast::rope(mx::transpose(queries, {0, 2, 1, 3}), head_dim,
                             false, config_.rope_theta, 1.0f, cache.offset());
    keys = mx::fast::rope(mx::transpose(keys, {0, 2, 1, 3}), head_dim, false,
                          config_.rope_theta, 1.0f, cache.offset());
    const auto [cached_keys, cached_values] =
        cache.UpdateAndFetch(keys, mx::transpose(values, {0, 2, 1, 3}));
    return mx::fast::scaled_dot_product_attention(
        queries, cached_keys, cached_values, scale, length > 1 ? "causal" : "");
  }();
  const std::string o_proj = prefix + ".self_attn.o_proj";
  const auto o_layer = QuantizedLayers({o_proj});
  const mx::array attended_rows =
      mx::reshape(mx::transpose(attended, {0, 2, 1, 3}), {batch, length, -1});
  const mx::array hidden =
      o_layer.has_value()
          ? QuantizedLinearAdd(attended_rows, (*o_layer)[0], x,
                               quantization.group_size, quantization.bits)
          : mx::add(x, Projections(attended_rows, {o_proj})[0]);
  const std::string mlp_norm = prefix + ".post_attention_layernorm";
  const std::string gate = prefix + ".mlp.gate_proj";
  const std::string up = prefix + ".mlp.up_proj";
  const auto gate_up = QuantizedLayers({gate, up});
  const std::string down_proj = prefix + ".mlp.down_proj";
  const auto down_layer = QuantizedLayers({down_proj});
  const mx::array activated = [&] {
    if (gate_up.has_value()) {
      return QuantizedGatedLinear(
          hidden, (*gate_up)[0], (*gate_up)[1], quantization.group_size,
          quantization.bits,
          InputNorm{checkpoint_.Weight(mlp_norm + ".weight"), eps});
    } else {
      const mx::array mlp_input = checkpoint_.RmsNorm(hidden, mlp_norm, eps);
      return mx::multiply(
          omni_common::Silu(checkpoint_.Linear(mlp_input, gate)),
          checkpoint_.Linear(mlp_input, up));
    }
  }();
  return down_layer.has_value()
             ? QuantizedLinearAdd(activated, (*down_layer)[0], hidden,
                                  quantization.group_size, quantization.bits)
             : mx::add(hidden, Projections(activated, {down_proj})[0]);
}

mx::array Qwen3ASRDecoder::Forward(const mx::array &embeddings,
                                   std::vector<Qwen3ASRKVCache> &caches) const {
  mx::array hidden = embeddings;
  for (int layer = 0; layer < config_.num_hidden_layers; ++layer) {
    hidden = Layer(hidden, layer, caches[layer]);
  }
  return hidden;
}

mx::array Qwen3ASRDecoder::LastLogits(const mx::array &hidden_states,
                                      int rows) const {
  const int length = hidden_states.shape(1);
  const mx::array last = checkpoint_.RmsNorm(
      mx::slice(hidden_states, {0, length - rows, 0},
                {hidden_states.shape(0), length, hidden_states.shape(2)}),
      prefix_ + ".norm", config_.rms_norm_eps);
  const mx::array logits = Projections(last, {prefix_ + ".embed_tokens"})[0];
  return mx::reshape(logits, {rows, logits.shape(-1)});
}

std::vector<Qwen3ASRKVCache> Qwen3ASRDecoder::NewCaches() const {
  return std::vector<Qwen3ASRKVCache>(config_.num_hidden_layers);
}

} // namespace qwen3_asr
