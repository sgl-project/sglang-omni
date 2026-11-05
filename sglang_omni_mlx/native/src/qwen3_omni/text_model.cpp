// SPDX-License-Identifier: Apache-2.0
#include "qwen3_omni/model.h"

#include <cmath>

namespace qwen3_omni {
std::pair<mx::array, mx::array> rotary_factors(const mx::array &positions,
                                               int dimensions, float theta,
                                               bool interleaved) {
  auto frequencies =
      mx::power(mx::array(theta),
                -mx::arange(0, dimensions, 2, mx::float32) / dimensions);
  auto angles = [&]() {
    if (interleaved) {
      std::vector<int> selector(dimensions / 2);
      for (int i = 0; i < dimensions / 2; ++i) {
        selector[i] = i < 60 ? i % 3 : 0;
      }
      auto selected = mx::take(mx::squeeze(positions, 1),
                               mx::array(selector.data(), {dimensions / 2}), 0);
      return mx::transpose(selected) * frequencies;
    } else {
      return mx::reshape(positions, {-1, 1}) * frequencies;
    }
  }();
  auto doubled = mx::concatenate({angles, angles}, -1);
  return {mx::cos(doubled), mx::sin(doubled)};
}
mx::array rotary(const mx::array &input,
                 const std::pair<mx::array, mx::array> &factors) {
  int dimensions = input.shape(-1);
  auto cosine =
      mx::expand_dims(mx::astype(factors.first, input.dtype()), {0, 1});
  auto sine =
      mx::expand_dims(mx::astype(factors.second, input.dtype()), {0, 1});
  auto rotated = mx::concatenate({-rows(input, dimensions / 2, dimensions, -1),
                                  rows(input, 0, dimensions / 2, -1)},
                                 -1);
  return input * cosine + rotated * sine;
}
mx::array rotary(const mx::array &input, const mx::array &positions,
                 float theta, bool interleaved) {
  return rotary(input,
                rotary_factors(positions, input.shape(-1), theta, interleaved));
}
TextModel::TextModel(Weights &weights, Json configuration, std::string prefix,
                     bool interleaved)
    : weights(weights), configuration(std::move(configuration)),
      prefix(std::move(prefix)), interleaved(interleaved) {
  const auto &config = this->configuration;
  validate_text_configuration(config);
  const int dimensions = config.at("hidden_size");
  const int head_dimensions = config.at("head_dim");
  for (int i = 0; i < config.at("num_hidden_layers"); ++i) {
    auto layer = this->prefix + ".layers." + std::to_string(i);
    for (const auto &name : {"input_layernorm", "post_attention_layernorm"}) {
      weights.require_norm(layer + "." + name, dimensions);
    }
    for (const auto &name : {"q_norm", "k_norm"}) {
      weights.require_norm(layer + ".self_attn." + name, head_dimensions);
    }
    for (const auto &name : {"q_proj", "k_proj", "v_proj"}) {
      int heads =
          config.at(std::string(name) == "q_proj" ? "num_attention_heads"
                                                  : "num_key_value_heads");
      weights.require_linear(layer + ".self_attn." + name, dimensions,
                             heads * head_dimensions);
    }
    weights.require_linear(layer + ".self_attn.o_proj",
                           config.at("num_attention_heads").get<int>() *
                               head_dimensions,
                           dimensions);
    int experts = config.value("num_experts", 0);
    auto require_feed_forward = [&](const std::string &name,
                                    int intermediate_dimensions,
                                    int expert_count) {
      weights.require_linear(name + ".gate_proj", dimensions,
                             intermediate_dimensions, false, expert_count);
      weights.require_linear(name + ".up_proj", dimensions,
                             intermediate_dimensions, false, expert_count);
      weights.require_linear(name + ".down_proj", intermediate_dimensions,
                             dimensions, false, expert_count);
    };
    if (experts > 0) {
      weights.require_linear(layer + ".mlp.gate", dimensions, experts);
      require_feed_forward(layer + ".mlp.switch_mlp",
                           config.at("moe_intermediate_size"), experts);
      int shared_dimensions =
          config.value("shared_expert_intermediate_size", 0);
      if (shared_dimensions > 0) {
        require_feed_forward(layer + ".mlp.shared_expert", shared_dimensions,
                             0);
        weights.require_linear(layer + ".mlp.shared_expert_gate", dimensions,
                               1);
      } else {
      }
    } else {
      require_feed_forward(layer + ".mlp", config.at("intermediate_size"), 0);
    }
  }
  weights.require_norm(this->prefix + ".norm", dimensions);
}
std::vector<KVCache> TextModel::make_cache() const {
  return std::vector<KVCache>(configuration.at("num_hidden_layers").get<int>());
}
mx::array
TextModel::feed_forward(const std::string &name, const mx::array &input,
                        const std::optional<mx::array> &experts) const {
  auto gated =
      omni_common::Silu(weights.linear(name + ".gate_proj", input, experts)) *
      weights.linear(name + ".up_proj", input, experts);
  return weights.linear(name + ".down_proj", gated, experts);
}
std::pair<mx::array, std::optional<mx::array>>
TextModel::forward(const mx::array &embeddings, const mx::array &positions,
                   std::vector<KVCache> &cache, const Cancellation &cancelled,
                   const std::vector<mx::array> &deepstack,
                   int capture_layer) const {
  auto hidden_states = embeddings;
  std::optional<mx::array> captured;
  const int length = embeddings.shape(1);
  const int heads = configuration.at("num_attention_heads");
  const int key_value_heads = configuration.at("num_key_value_heads");
  const int head_dimensions = configuration.at("head_dim");
  const float epsilon = configuration.at("rms_norm_eps");
  auto factors = rotary_factors(positions, head_dimensions,
                                configuration.at("rope_theta"), interleaved);
  for (int i = 0; i < static_cast<int>(cache.size()); ++i) {
    check_cancelled(cancelled);
    auto layer = prefix + ".layers." + std::to_string(i);
    auto normalized =
        weights.rms(layer + ".input_layernorm", hidden_states, epsilon);
    auto query =
        mx::reshape(weights.linear(layer + ".self_attn.q_proj", normalized),
                    {1, length, heads, head_dimensions});
    auto key =
        mx::reshape(weights.linear(layer + ".self_attn.k_proj", normalized),
                    {1, length, key_value_heads, head_dimensions});
    query = rotary(
        mx::transpose(weights.rms(layer + ".self_attn.q_norm", query, epsilon),
                      {0, 2, 1, 3}),
        factors);
    key = rotary(
        mx::transpose(weights.rms(layer + ".self_attn.k_norm", key, epsilon),
                      {0, 2, 1, 3}),
        factors);
    auto value = mx::transpose(
        mx::reshape(weights.linear(layer + ".self_attn.v_proj", normalized),
                    {1, length, key_value_heads, head_dimensions}),
        {0, 2, 1, 3});
    auto &state = cache[i];
    if (state.keys) {
      state.keys = mx::concatenate({*state.keys, key}, 2);
      state.values = mx::concatenate({*state.values, value}, 2);
    } else {
      state.keys = key;
      state.values = value;
    }
    state.offset += length;
    auto attended = mx::fast::scaled_dot_product_attention(
        query, *state.keys, *state.values,
        std::pow(float(head_dimensions), -0.5f), length > 1 ? "causal" : "");
    attended =
        mx::reshape(mx::transpose(attended, {0, 2, 1, 3}), {1, length, -1});
    hidden_states =
        hidden_states + weights.linear(layer + ".self_attn.o_proj", attended);
    auto input = weights.rms(layer + ".post_attention_layernorm", hidden_states,
                             epsilon);
    if (configuration.value("num_experts", 0) > 0) {
      auto logits = weights.linear(layer + ".mlp.gate", input);
      auto probabilities = mx::softmax(prefix.starts_with("talker")
                                           ? mx::astype(logits, mx::float32)
                                           : logits,
                                       -1, true);
      int count = configuration.at("num_experts_per_tok");
      auto indices =
          rows(mx::argpartition(-probabilities, count - 1, -1), 0, count, -1);
      auto scores = mx::take_along_axis(probabilities, indices, -1);
      scores = scores / mx::sum(scores, -1, true);
      auto expanded = mx::expand_dims(input, {2, 3});
      auto expert_output = mx::squeeze(
          feed_forward(layer + ".mlp.switch_mlp", expanded, indices), -2);
      hidden_states = hidden_states +
                      mx::sum(expert_output * mx::expand_dims(scores, -1), -2);
      if (configuration.value("shared_expert_intermediate_size", 0) > 0) {
        hidden_states = hidden_states +
                        mx::sigmoid(weights.linear(
                            layer + ".mlp.shared_expert_gate", input)) *
                            feed_forward(layer + ".mlp.shared_expert", input);
      } else {
      }
    } else {
      hidden_states = hidden_states + feed_forward(layer + ".mlp", input);
    }
    if (i < static_cast<int>(deepstack.size())) {
      hidden_states = hidden_states + deepstack[i];
    } else {
    }
    if (capture_layer == i + 1) {
      captured = hidden_states;
    } else {
    }
    if (length > 1 && i % 4 == 3) {
      mx::eval(hidden_states);
    } else {
    }
  }
  return {weights.rms(prefix + ".norm", hidden_states, epsilon), captured};
}
} // namespace qwen3_omni
