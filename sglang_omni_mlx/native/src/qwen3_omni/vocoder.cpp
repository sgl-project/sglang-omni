// SPDX-License-Identifier: Apache-2.0
#include "qwen3_omni/vocoder.h"
#include <cmath>
namespace qwen3_omni {
Vocoder::Vocoder(Weights &weights, Json configuration)
    : weights(weights), configuration(std::move(configuration)) {
  const auto &config = this->configuration;
  const int dimensions = config.at("hidden_size");
  const std::string prefix = "code2wav";
  weights.require_linear(prefix + ".code_embedding", dimensions,
                         config.at("codebook_size").get<int>() *
                             config.at("num_quantizers").get<int>());
  for (int i = 0; i < config.at("num_hidden_layers"); ++i) {
    auto layer = prefix + ".pre_transformer.layers." + std::to_string(i);
    for (const auto &name : {"input_layernorm", "post_attention_layernorm"}) {
      weights.require_norm(layer + "." + name, dimensions);
    }
    for (const auto &name : {"self_attn_layer_scale", "mlp_layer_scale"}) {
      weights.require(layer + "." + name + ".scale", {dimensions});
    }
    for (const auto &name : {"q_proj", "k_proj", "v_proj", "o_proj"}) {
      weights.require_linear(layer + ".self_attn." + name, dimensions,
                             dimensions);
    }
    for (const auto &name : {"gate_proj", "up_proj"}) {
      weights.require_linear(layer + ".mlp." + name, dimensions,
                             config.at("intermediate_size"));
    }
    weights.require_linear(layer + ".mlp.down_proj",
                           config.at("intermediate_size"), dimensions);
  }
  weights.require_norm(prefix + ".pre_transformer.norm", dimensions);
  auto require_convolution = [&](const std::string &name, int input_dimensions,
                                 int output_dimensions, int kernel) {
    weights.require(name + ".weight",
                    {output_dimensions, kernel, input_dimensions});
    weights.require(name + ".bias", {output_dimensions});
  };
  auto require_snake = [&](const std::string &name, int dimensions) {
    weights.require(name + ".alpha", {dimensions});
    weights.require(name + ".beta", {dimensions});
  };
  for (int i = 0; i < static_cast<int>(config.at("upsampling_ratios").size());
       ++i) {
    int ratio = config.at("upsampling_ratios")[i];
    auto name = prefix + ".upsample." + std::to_string(i);
    require_convolution(name + ".0.conv", dimensions, dimensions, ratio);
    require_convolution(name + ".1.dwconv.conv", 1, dimensions, 7);
    weights.require_norm(name + ".1.norm", dimensions, true);
    weights.require_linear(name + ".1.pwconv1", dimensions, dimensions * 4,
                           true);
    weights.require_linear(name + ".1.pwconv2", dimensions * 4, dimensions,
                           true);
    weights.require(name + ".1.gamma", {dimensions});
  }
  const int decoder_dimensions = config.at("decoder_dim");
  require_convolution(prefix + ".decoder.0.conv", dimensions,
                      decoder_dimensions, 7);
  int count = config.at("upsample_rates").size();
  for (int i = 0; i < count; ++i) {
    int ratio = config.at("upsample_rates")[i];
    int input_dimensions = decoder_dimensions / (1 << i);
    int output_dimensions = input_dimensions / 2;
    auto name = prefix + ".decoder." + std::to_string(i + 1) + ".block";
    require_snake(name + ".0", input_dimensions);
    require_convolution(name + ".1.conv", input_dimensions, output_dimensions,
                        2 * ratio);
    for (int residual = 2; residual <= 4; ++residual) {
      auto block = name + "." + std::to_string(residual);
      require_snake(block + ".act1", output_dimensions);
      require_snake(block + ".act2", output_dimensions);
      require_convolution(block + ".conv1.conv", output_dimensions,
                          output_dimensions, 7);
      require_convolution(block + ".conv2.conv", output_dimensions,
                          output_dimensions, 1);
    }
  }
  require_snake(prefix + ".decoder." + std::to_string(count + 1),
                decoder_dimensions / (1 << count));
  require_convolution(prefix + ".decoder." + std::to_string(count + 2) +
                          ".conv",
                      decoder_dimensions / (1 << count), 1, 7);
}
int Vocoder::samples_per_frame() const {
  int samples = 1;
  for (const auto &ratio : configuration.at("upsample_rates")) {
    samples *= ratio.get<int>();
  }
  for (const auto &ratio : configuration.at("upsampling_ratios")) {
    samples *= ratio.get<int>();
  }
  return samples;
}
mx::array Vocoder::convolution(const std::string &name, const mx::array &input,
                               int dilation, int groups) const {
  const auto &weight = weights[name + ".weight"];
  auto padded =
      mx::pad(input, {{0, 0}, {(weight.shape(1) - 1) * dilation, 0}, {0, 0}});
  return mx::conv1d(padded, weight, 1, 0, dilation, groups) +
         weights[name + ".bias"];
}
mx::array Vocoder::transpose_convolution(const std::string &name,
                                         const mx::array &input,
                                         int stride) const {
  auto result = mx::conv_transpose1d(input, weights[name + ".weight"], stride) +
                weights[name + ".bias"];
  return rows(result, 0, input.shape(1) * stride, 1);
}
mx::array Vocoder::snake(const std::string &name,
                         const mx::array &input) const {
  auto oscillation = mx::sin(input * mx::exp(weights[name + ".alpha"]));
  return input +
         oscillation * oscillation / (mx::exp(weights[name + ".beta"]) + 1e-9f);
}
mx::array Vocoder::forward(const mx::array &codes,
                           const Cancellation &cancelled) const {
  const std::string prefix = "code2wav";
  int dimensions = configuration.at("hidden_size");
  int heads = configuration.at("num_attention_heads");
  int head_dimensions = dimensions / heads;
  auto offsets =
      mx::reshape(
          mx::arange(configuration.at("num_quantizers").get<int>(), mx::int32),
          {1, -1, 1}) *
      configuration.at("codebook_size").get<int>();
  auto hidden_states = mx::mean(
      weights.embedding(prefix + ".code_embedding", codes + offsets), 1);
  int length = hidden_states.shape(1);
  auto factors = rotary_factors(mx::arange(length, mx::int32), head_dimensions,
                                configuration.at("rope_theta"), false);
  float epsilon = configuration.at("rms_norm_eps");
  for (int i = 0; i < configuration.at("num_hidden_layers"); ++i) {
    check_cancelled(cancelled);
    auto layer = prefix + ".pre_transformer.layers." + std::to_string(i);
    auto normalized =
        weights.rms(layer + ".input_layernorm", hidden_states, epsilon);
    auto project = [&](const std::string &name) {
      return mx::transpose(
          mx::reshape(weights.linear(layer + ".self_attn." + name, normalized),
                      {1, length, heads, head_dimensions}),
          {0, 2, 1, 3});
    };
    auto query = rotary(project("q_proj"), factors),
         key = rotary(project("k_proj"), factors);
    auto attended = mx::fast::scaled_dot_product_attention(
        query, key, project("v_proj"), std::pow(float(head_dimensions), -.5f),
        "causal");
    attended = mx::reshape(mx::transpose(attended, {0, 2, 1, 3}),
                           {1, length, dimensions});
    hidden_states =
        hidden_states + weights.linear(layer + ".self_attn.o_proj", attended) *
                            weights[layer + ".self_attn_layer_scale.scale"];
    auto input = weights.rms(layer + ".post_attention_layernorm", hidden_states,
                             epsilon);
    auto gated =
        omni_common::Silu(weights.linear(layer + ".mlp.gate_proj", input)) *
        weights.linear(layer + ".mlp.up_proj", input);
    hidden_states =
        hidden_states + weights.linear(layer + ".mlp.down_proj", gated) *
                            weights[layer + ".mlp_layer_scale.scale"];
  }
  hidden_states =
      weights.rms(prefix + ".pre_transformer.norm", hidden_states, epsilon);
  for (int i = 0;
       i < static_cast<int>(configuration.at("upsampling_ratios").size());
       ++i) {
    auto name = prefix + ".upsample." + std::to_string(i);
    hidden_states =
        transpose_convolution(name + ".0.conv", hidden_states,
                              configuration.at("upsampling_ratios")[i]);
    auto convolved =
        convolution(name + ".1.dwconv.conv", hidden_states, 1, dimensions);
    auto normalized = weights.norm(name + ".1.norm", convolved, 1e-6);
    hidden_states =
        hidden_states + weights.linear(name + ".1.pwconv2",
                                       omni_common::Gelu(weights.linear(
                                           name + ".1.pwconv1", normalized))) *
                            weights[name + ".1.gamma"];
  }
  hidden_states = convolution(prefix + ".decoder.0.conv", hidden_states);
  int count = configuration.at("upsample_rates").size();
  for (int i = 0; i < count; ++i) {
    check_cancelled(cancelled);
    auto name = prefix + ".decoder." + std::to_string(i + 1) + ".block";
    hidden_states = transpose_convolution(
        name + ".1.conv", snake(name + ".0", hidden_states),
        configuration.at("upsample_rates")[i]);
    int dilation = 1;
    for (int residual = 2; residual <= 4; ++residual, dilation *= 3) {
      auto block = name + "." + std::to_string(residual);
      auto convolved =
          convolution(block + ".conv1.conv",
                      snake(block + ".act1", hidden_states), dilation);
      hidden_states =
          hidden_states +
          convolution(block + ".conv2.conv", snake(block + ".act2", convolved));
    }
  }
  return mx::reshape(
      mx::clip(convolution(
                   prefix + ".decoder." + std::to_string(count + 2) + ".conv",
                   snake(prefix + ".decoder." + std::to_string(count + 1),
                         hidden_states)),
               mx::array(-1.f), mx::array(1.f)),
      {-1});
}
} // namespace qwen3_omni
