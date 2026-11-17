// SPDX-License-Identifier: Apache-2.0
#include "common/qwen3_decoder.h"

#include <algorithm>
#include <cassert>
#include <cmath>

#include "common/generation.h"
#include "common/kv_cache.h"

namespace omni_common {

namespace mx = mlx::core;

namespace {

// The longest input, such as a draft's verify block, that runs as one graph.
constexpr int kSingleGraphTokens = 32;

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
          {heads, mx::zeros({heads.shape(0), heads.shape(1),
                             omni_common::kKvCacheStepTokens, heads.shape(3)},
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

mx::array Qwen3Decoder::Project(const mx::array &x, const std::string &prefix,
                                bool tied_embedding) const {
  return tied_embedding ? checkpoint_.TiedProjection(x, prefix)
                        : checkpoint_.Linear(x, prefix);
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
  mx::array queries = mx::reshape(Project(normed, prefix + ".self_attn.q_proj"),
                                  {batch, length, head_count, head_dim});
  if (config_.has_query_key_norm) {
    queries = checkpoint_.RmsNorm(queries, prefix + ".self_attn.q_norm", eps);
  } else {
  }
  queries = Rotated(mx::transpose(queries, {0, 2, 1, 3}), cache);
  const auto [keys, values] = KeysValues(normed, prefix, cache);
  auto [cached_keys, cached_values] = cache.UpdateAndFetch(keys, values);
  const mx::array attended = mx::fast::scaled_dot_product_attention(
      queries, cached_keys, cached_values,
      static_cast<float>(std::pow(static_cast<double>(head_dim), -0.5)),
      /*window_size=*/-1, length > 1 ? "causal" : "");
  const mx::array hidden =
      mx::add(x, Project(mx::reshape(mx::transpose(attended, {0, 2, 1, 3}),
                                     {batch, length, -1}),
                         prefix + ".self_attn.o_proj"));
  const mx::array mlp_input =
      checkpoint_.RmsNorm(hidden, prefix + ".post_attention_layernorm", eps);
  return mx::add(
      hidden,
      Project(omni_common::Swiglu(Project(mlp_input, prefix + ".mlp.gate_proj"),
                                  Project(mlp_input, prefix + ".mlp.up_proj")),
              prefix + ".mlp.down_proj"));
}

mx::array Qwen3Decoder::Forward(const mx::array &embeddings,
                                std::vector<KVCache> &caches,
                                const std::vector<int> &capture_layers,
                                std::vector<mx::array> *captured) const {
  mx::array hidden = embeddings;
  // Note (Yucheng Hu): MLX frees temporaries per buffer, so eval per layer.
  // Note (XXXXRT666): only past kSingleGraphTokens, so a short input such as a
  // draft block or a code predictor's first step builds without waiting.
  const bool is_prefill = embeddings.shape(1) > kSingleGraphTokens;
  for (int layer = 0; layer < config_.num_hidden_layers; ++layer) {
    if (std::count(capture_layers.begin(), capture_layers.end(), layer) > 0) {
      captured->push_back(hidden);
    } else {
    }
    const mx::array layer_input = hidden;
    hidden = Layer(hidden, layer, caches[layer]);
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
                         const KVCache &cache) const {
  const int batch = x.shape(0);
  const int length = x.shape(1);
  const int kv_head_count = config_.num_key_value_heads;
  const int head_dim = config_.head_dim;
  mx::array keys = mx::reshape(Project(x, prefix + ".self_attn.k_proj"),
                               {batch, length, kv_head_count, head_dim});
  if (config_.has_query_key_norm) {
    keys = checkpoint_.RmsNorm(keys, prefix + ".self_attn.k_norm",
                               config_.rms_norm_eps);
  } else {
  }
  keys = Rotated(mx::transpose(keys, {0, 2, 1, 3}), cache);
  const mx::array values =
      mx::reshape(Project(x, prefix + ".self_attn.v_proj"),
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
  const mx::array logits = Project(last, Name("embed_tokens"), true);
  return mx::reshape(logits, {logits.shape(-1)});
}

mx::array Qwen3Decoder::Rotated(const mx::array &heads,
                                const KVCache &cache) const {
  return mx::fast::rope(heads, config_.head_dim, config_.traditional_rope,
                        config_.rope_theta, 1.0f, cache.offset());
}

mx::array Qwen3Decoder::Logits(const mx::array &hidden_states) const {
  const mx::array logits = Project(
      checkpoint_.RmsNorm(hidden_states, Name("norm"), config_.rms_norm_eps),
      Name("embed_tokens"), true);
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
      /*window_size=*/-1, "array", mask);
  const mx::array hidden = mx::add(
      x, checkpoint_.Linear(
             mx::reshape(mx::transpose(attended, {0, 2, 1, 3}), {batch, 1, -1}),
             prefix + ".self_attn.o_proj"));
  const mx::array mlp_input =
      checkpoint_.RmsNorm(hidden, prefix + ".post_attention_layernorm", eps);
  return mx::add(
      hidden, checkpoint_.Linear(
                  omni_common::Swiglu(
                      checkpoint_.Linear(mlp_input, prefix + ".mlp.gate_proj"),
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

void Qwen3Decoder::DecodePlain(
    mx::array &next_token, std::vector<KVCache> &caches,
    float repetition_penalty, const std::function<bool(int)> &commit,
    const std::atomic<bool> &cancel, const std::vector<int> &capture_layers,
    std::vector<std::vector<mx::array>> *captured) const {
  // Note (Minyi Tang): only generated tokens are penalized, as on CUDA; the
  // scales stay lazy so penalizing adds no host sync to the queued step.
  mx::array penalty_by_token_id = mx::ones({config_.vocab_size}, mx::float32);
  bool finished = false;
  while (!finished) {
    if (cancel.load()) {
      throw omni_common::GenerationCancelled();
    } else {
    }
    const mx::array token = next_token;
    // Note (Jiaxin Deng): queue the next step before reading this token, so
    // the GPU decodes while the stop rules are checked.
    const mx::array embeddings = EmbedTokens(mx::reshape(token, {1, 1}));
    mx::array logits = LastLogits(
        Forward(embeddings, caches, capture_layers,
                captured == nullptr ? nullptr : &captured->emplace_back()));
    if (repetition_penalty != 1.0f) {
      penalty_by_token_id =
          mx::put_along_axis(penalty_by_token_id, mx::reshape(token, {1}),
                             mx::array(repetition_penalty), 0);
      logits = mx::astype(logits, mx::float32);
      logits = mx::where(mx::less(logits, mx::array(0.0f)),
                         mx::multiply(logits, penalty_by_token_id),
                         mx::divide(logits, penalty_by_token_id));
    } else {
    }
    next_token = mx::argmax(logits);
    mx::async_eval({next_token});
    finished = commit(static_cast<int>(token.item<uint32_t>()));
  }
}

std::vector<KVCache> Qwen3Decoder::NewCaches() const {
  return std::vector<KVCache>(config_.num_hidden_layers);
}

} // namespace omni_common
