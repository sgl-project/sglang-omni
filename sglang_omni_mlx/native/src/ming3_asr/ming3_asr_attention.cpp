#include "ming3_asr/ming3_asr_model.h"

#include <cmath>
#include <utility>

namespace ming3_asr {

namespace mx = mlx::core;

namespace {

constexpr float kQueryKeyNormEpsilon = 1e-6f;

} // namespace

mx::array MingModel::DeltaAttention(const mx::array &hidden_states,
                                    const std::string &weight_prefix,
                                    DeltaCache &cache) const {
  const int token_count = hidden_states.shape(1);
  const int head_count = config_.attention_head_count;
  const int head_dimension = config_.attention_head_dimension;
  const int projection_dimension = head_count * head_dimension;
  const int history_token_count = config_.convolution_kernel_token_count - 1;
  std::array<mx::array, 3> convolution_history =
      cache.state.has_value()
          ? cache.state->convolution
          : std::array{mx::zeros({1, history_token_count, projection_dimension},
                                 hidden_states.dtype()),
                       mx::zeros({1, history_token_count, projection_dimension},
                                 hidden_states.dtype()),
                       mx::zeros({1, history_token_count, projection_dimension},
                                 hidden_states.dtype())};
  const auto project_convolution = [&](const std::string &projection_name,
                                       size_t projection_index) {
    auto projected_states = checkpoint_.Linear(
        hidden_states, weight_prefix + "." + projection_name + "_proj");
    projected_states = mx::concatenate(
        {convolution_history[projection_index], projected_states}, 1);
    convolution_history[projection_index] = mx::slice(
        projected_states, {0, token_count, 0}, projected_states.shape());
    projected_states = omni_common::Silu(
        mx::conv1d(projected_states,
                   checkpoint_.Weight(weight_prefix + "." + projection_name +
                                      "_conv1d.weight"),
                   1, 0, 1, projection_dimension));
    return mx::reshape(projected_states,
                       {1, token_count, head_count, head_dimension});
  };
  const std::array projected_states{project_convolution("q", 0),
                                    project_convolution("k", 1),
                                    project_convolution("v", 2)};
  auto queries = mx::astype(projected_states[0], mx::float32);
  auto keys = mx::astype(projected_states[1], mx::float32);
  queries = mx::multiply(
      queries, mx::rsqrt(mx::add(mx::sum(mx::square(queries), -1, true),
                                 mx::array(kQueryKeyNormEpsilon))));
  queries = mx::multiply(
      queries, mx::array(1.0f / std::sqrt(static_cast<float>(head_dimension))));
  keys =
      mx::multiply(keys, mx::rsqrt(mx::add(mx::sum(mx::square(keys), -1, true),
                                           mx::array(kQueryKeyNormEpsilon))));
  const auto gate = mx::reshape(
      mx::add(mx::astype(
                  checkpoint_.Linear(hidden_states, weight_prefix + ".f_proj"),
                  mx::float32),
              checkpoint_.Weight(weight_prefix + ".dt_bias")),
      {1, token_count, head_count, head_dimension});
  const auto rate = mx::reshape(
      mx::exp(mx::astype(checkpoint_.Weight(weight_prefix + ".A_log"),
                         mx::float32)),
      {head_count, 1});
  const auto decay =
      mx::exp(mx::multiply(mx::array(config_.kda_lower_bound),
                           mx::sigmoid(mx::multiply(rate, gate))));
  const auto beta = mx::sigmoid(
      mx::astype(checkpoint_.Linear(hidden_states, weight_prefix + ".b_proj"),
                 mx::float32));
  const auto state =
      cache.state.has_value()
          ? cache.state->recurrent
          : mx::zeros({1, head_count, head_dimension, head_dimension},
                      mx::float32);
  const auto recurrent =
      GatedDelta(queries, keys, projected_states[2], decay, beta, state);
  cache.state =
      DeltaState{std::move(convolution_history), recurrent.next_state};
  const auto rounded_output =
      mx::astype(recurrent.output, projected_states[2].dtype());
  auto output = mx::astype(rounded_output, mx::float32);
  output = mx::fast::rms_norm(
      output,
      mx::astype(checkpoint_.Weight(weight_prefix + ".o_norm.weight"),
                 mx::float32),
      config_.rms_norm_epsilon);
  const auto output_gate = mx::astype(
      mx::reshape(checkpoint_.Linear(hidden_states, weight_prefix + ".g_proj"),
                  output.shape()),
      mx::float32);
  output =
      mx::reshape(mx::astype(mx::multiply(output, mx::sigmoid(output_gate)),
                             hidden_states.dtype()),
                  {1, token_count, projection_dimension});
  return checkpoint_.Linear(output, weight_prefix + ".o_proj");
}

mx::array MingModel::LatentAttention(const mx::array &hidden_states,
                                     const std::string &weight_prefix,
                                     LatentAttentionCache &cache) const {
  const int token_count = hidden_states.shape(1);
  const int head_count = config_.attention_head_count;
  const int cached_token_count =
      cache.state.has_value() ? cache.state->keys.shape(2) : 0;
  auto queries = checkpoint_.Linear(
      checkpoint_.RmsNorm(
          checkpoint_.Linear(hidden_states, weight_prefix + ".q_a_proj"),
          weight_prefix + ".q_a_layernorm", config_.rms_norm_epsilon),
      weight_prefix + ".q_b_proj");
  queries = mx::transpose(
      mx::reshape(queries, {1, token_count, head_count, -1}), {0, 2, 1, 3});
  const auto query_parts =
      mx::split(queries, mx::Shape{config_.non_rotary_head_dimension}, -1);
  const auto latent_parts = mx::split(
      checkpoint_.Linear(hidden_states, weight_prefix + ".kv_a_proj_with_mqa"),
      mx::Shape{config_.key_value_lora_rank}, -1);
  auto latent_states = checkpoint_.Linear(
      checkpoint_.RmsNorm(latent_parts[0], weight_prefix + ".kv_a_layernorm",
                          config_.rms_norm_epsilon),
      weight_prefix + ".kv_b_proj");
  latent_states = mx::transpose(
      mx::reshape(latent_states, {1, token_count, head_count, -1}),
      {0, 2, 1, 3});
  const auto key_value = mx::split(
      latent_states, mx::Shape{config_.non_rotary_head_dimension}, -1);
  const auto rotated_queries =
      mx::fast::rope(query_parts[1], config_.rotary_head_dimension, true,
                     config_.rope_theta, 1.0f, cached_token_count);
  const auto rotated_keys = mx::fast::rope(
      mx::expand_dims(latent_parts[1], 1), config_.rotary_head_dimension, true,
      config_.rope_theta, 1.0f, cached_token_count);
  queries = mx::concatenate({query_parts[0], rotated_queries}, -1);
  auto keys = mx::concatenate(
      {key_value[0], mx::broadcast_to(rotated_keys, rotated_queries.shape())},
      -1);
  auto values = key_value[1];
  if (cache.state.has_value()) {
    keys = mx::concatenate({cache.state->keys, keys}, 2);
    values = mx::concatenate({cache.state->values, values}, 2);
  } else {
  }
  cache.state = LatentAttentionState{keys, values};
  auto output = mx::fast::scaled_dot_product_attention(
      queries, keys, values,
      1.0f / std::sqrt(static_cast<float>(config_.non_rotary_head_dimension +
                                          config_.rotary_head_dimension)),
      token_count > 1 ? "causal" : "");
  output = mx::transpose(output, {0, 2, 1, 3});
  const auto gate = mx::reshape(
      mx::astype(
          mx::sigmoid(mx::astype(
              checkpoint_.Linear(hidden_states, weight_prefix + ".g_proj"),
              mx::float32)),
          hidden_states.dtype()),
      {1, token_count, head_count, 1});
  return checkpoint_.Linear(
      mx::reshape(mx::multiply(output, gate),
                  {1, token_count, head_count * config_.value_head_dimension}),
      weight_prefix + ".dense");
}

} // namespace ming3_asr
