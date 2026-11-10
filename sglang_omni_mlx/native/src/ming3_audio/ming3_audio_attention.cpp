// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Antgroup and The HuggingFace Inc. team. All rights reserved.
#include "ming3_audio/ming3_audio_model.h"

#include <cmath>

namespace ming3_audio {
namespace mx = mlx::core;

namespace {
constexpr int kRecurrentTokenLimit = 64;

// Note (XXXXRT666): KDA decays each key channel, but MLX 0.32.4's
// mx::fast::gated_delta_update reads one gate per head whenever both widths
// are 128, and a loop of MLX ops costs several dispatches per token. Switch
// to gated_delta_update once the runtime's MLX includes ml-explore/mlx#4791.
// One simdgroup runs one head's value column through every token in float32,
// each lane holding a 32nd of that column's state in registers.
const mx::fast::CustomKernelFunction &DeltaRule() {
  static const auto kernel = mx::fast::metal_kernel(
      "ming3_audio_delta_rule",
      {"queries", "keys", "values", "decay", "beta", "state_in"},
      {"outputs", "state_out"}, R"(
  constexpr int lane_width = key_width / 32;
  const uint head = thread_position_in_grid.z;
  const uint column = thread_position_in_grid.y;
  const uint lane_offset = thread_index_in_simdgroup * lane_width;
  const int length = queries_shape[0];
  const device float *query = queries + head * key_width + lane_offset;
  const device float *key = keys + head * key_width + lane_offset;
  const device float *gate = decay + head * key_width + lane_offset;
  const device float *value = values + head * value_width + column;
  const device float *strength = beta + head;
  device float *output = outputs + head * value_width + column;
  const uint state_offset = (head * value_width + column) * key_width + lane_offset;
  float memory[lane_width];
  for (int i = 0; i < lane_width; ++i) {
    memory[i] = state_in[state_offset + i];
  }
  for (int t = 0; t < length; ++t) {
    float recalled = 0.0f;
    for (int i = 0; i < lane_width; ++i) {
      memory[i] *= gate[i];
      recalled += memory[i] * key[i];
    }
    const float correction = strength[0] * (value[0] - simd_sum(recalled));
    float answer = 0.0f;
    for (int i = 0; i < lane_width; ++i) {
      memory[i] += key[i] * correction;
      answer += memory[i] * query[i];
    }
    answer = simd_sum(answer);
    if (thread_index_in_simdgroup == 0) {
      output[0] = answer;
    }
    query += heads * key_width;
    key += heads * key_width;
    gate += heads * key_width;
    value += heads * value_width;
    strength += heads;
    output += heads * value_width;
  }
  for (int i = 0; i < lane_width; ++i) {
    state_out[state_offset + i] = memory[i];
  }
)");
  return kernel;
}
} // namespace

mx::array Model::LatentAttention(const mx::array &hidden,
                                 const std::string &prefix,
                                 qwen3_asr::KVCache &cache) const {
  const int length = hidden.shape(1), heads = text_.num_attention_heads;
  const int cached_tokens = cache.offset();
  const int rotary_width = text_.qk_rope_head_dim;
  const int plain_width = text_.qk_nope_head_dim;
  const int value_width = text_.v_head_dim, latent_width = text_.kv_lora_rank;
  const auto query = mx::transpose(
      mx::reshape(checkpoint_.Linear(
                      RmsNorm(checkpoint_.Linear(hidden, prefix + ".q_a_proj"),
                              prefix + ".q_a_layernorm"),
                      prefix + ".q_b_proj"),
                  {1, length, heads, plain_width + rotary_width}),
      {0, 2, 1, 3});
  const auto query_parts = mx::split(query, mx::Shape{plain_width}, -1);
  const auto compressed =
      mx::split(checkpoint_.Linear(hidden, prefix + ".kv_a_proj_with_mqa"),
                mx::Shape{latent_width}, -1);
  const auto key_value = mx::transpose(
      mx::reshape(
          checkpoint_.Linear(RmsNorm(compressed[0], prefix + ".kv_a_layernorm"),
                             prefix + ".kv_b_proj"),
          {1, length, heads, plain_width + value_width}),
      {0, 2, 1, 3});
  const auto key_value_parts = mx::split(key_value, mx::Shape{plain_width}, -1);
  const auto rotary = [&](const mx::array &input) {
    return mx::fast::rope(input, rotary_width, true, text_.rope_theta, 1.0f,
                          cached_tokens);
  };
  const auto queries =
      mx::concatenate({query_parts[0], rotary(query_parts[1])}, -1);
  const auto keys = mx::concatenate(
      {key_value_parts[0],
       mx::broadcast_to(
           rotary(mx::reshape(compressed[1], {1, 1, length, rotary_width})),
           {1, heads, length, rotary_width})},
      -1);
  const auto [cached_keys, cached_values] =
      cache.UpdateAndFetch(keys, key_value_parts[1]);
  auto attended =
      mx::transpose(mx::fast::scaled_dot_product_attention(
                        queries, cached_keys, cached_values,
                        1.0f / std::sqrt(plain_width + rotary_width),
                        /*window_size=*/-1, length > 1 ? "causal" : ""),
                    {0, 2, 1, 3});
  const auto gate = mx::astype(
      mx::sigmoid(mx::astype(checkpoint_.Linear(hidden, prefix + ".g_proj"),
                             mx::float32)),
      hidden.dtype());
  attended = mx::multiply(attended, mx::expand_dims(gate, -1));
  return checkpoint_.Linear(
      mx::reshape(attended, {1, length, heads * value_width}),
      prefix + ".dense");
}

mx::array Model::DeltaAttention(const mx::array &hidden,
                                const std::string &prefix,
                                LayerCache &cache) const {
  const int length = hidden.shape(1), heads = text_.num_attention_heads;
  const int width = text_.head_dim, kernel = text_.short_conv_kernel_size;
  std::vector<mx::array> projected;
  int projection_index = 0;
  for (const auto *name : {"q", "k", "v"}) {
    const std::string projection = prefix + "." + name;
    const auto values = checkpoint_.Linear(hidden, projection + "_proj");
    const auto history = cache.convolution[projection_index].value_or(
        mx::zeros({1, kernel - 1, heads * width}, hidden.dtype()));
    const auto padded = mx::concatenate({history, values}, 1);
    cache.convolution[projection_index++] =
        mx::slice(padded, {0, length, 0}, padded.shape());
    const auto convolved =
        mx::conv1d(mx::astype(padded, mx::float32),
                   checkpoint_.Weight(projection + "_conv1d.weight"), 1, 0, 1,
                   heads * width);
    projected.push_back(
        mx::reshape(mx::astype(omni_common::Silu(convolved), hidden.dtype()),
                    {length, heads, width}));
  }
  const auto normalize = [&](const mx::array &input) {
    const auto values = mx::astype(input, mx::float32);
    const auto normalized = mx::multiply(
        values, mx::rsqrt(mx::add(mx::sum(mx::square(values), -1, true),
                                  mx::array(1e-6f))));
    // Note (altale): Long prefills materialize normalized Q/K in the input
    // dtype.
    if (length > kRecurrentTokenLimit) {
      return mx::astype(mx::astype(normalized, input.dtype()), mx::float32);
    } else {
      return normalized;
    }
  };
  const auto queries =
      mx::multiply(normalize(projected[0]), mx::array(1.0f / std::sqrt(width)));
  const auto keys = normalize(projected[1]);
  const auto values = mx::astype(projected[2], mx::float32);
  const auto gate_logits = mx::reshape(
      mx::astype(checkpoint_.Linear(hidden, prefix + ".f_proj"), mx::float32),
      {length, heads, width});
  const auto decay = mx::exp(mx::multiply(
      mx::array(text_.kda_lower_bound),
      mx::sigmoid(mx::multiply(
          checkpoint_.Weight(prefix + ".decay_scale"),
          mx::add(gate_logits, checkpoint_.Weight(prefix + ".dt_bias"))))));
  const auto beta = mx::reshape(
      mx::sigmoid(mx::astype(checkpoint_.Linear(hidden, prefix + ".b_proj"),
                             mx::float32)),
      {length, heads});
  const auto state =
      cache.recurrent.value_or(mx::zeros({heads, width, width}, mx::float32));
  const auto recurrence = DeltaRule()(
      {queries, keys, values, decay, beta, state},
      {{length, heads, width}, state.shape()}, {mx::float32, mx::float32},
      {32, width, heads}, {32, 4, 1},
      {{"key_width", width}, {"value_width", width}, {"heads", heads}},
      std::nullopt, false, {});
  cache.recurrent = recurrence[1];
  auto normalized = mx::fast::rms_norm(
      mx::astype(mx::astype(recurrence[0], hidden.dtype()), mx::float32),
      checkpoint_.Weight(prefix + ".o_norm.weight"), text_.rms_norm_eps);
  const auto output_gate = mx::reshape(
      mx::astype(checkpoint_.Linear(hidden, prefix + ".g_proj"), mx::float32),
      {length, heads, width});
  normalized = mx::astype(mx::multiply(normalized, mx::sigmoid(output_gate)),
                          hidden.dtype());
  return checkpoint_.Linear(mx::reshape(normalized, {1, length, heads * width}),
                            prefix + ".o_proj");
}

} // namespace ming3_audio
