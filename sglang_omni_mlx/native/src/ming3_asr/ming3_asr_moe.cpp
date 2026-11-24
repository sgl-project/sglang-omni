#include "ming3_asr/ming3_asr_model.h"

#include <algorithm>
#include <limits>

namespace ming3_asr {

namespace mx = mlx::core;

namespace {

constexpr float kRoutingWeightEpsilon = 1e-20f;
constexpr size_t kSortedExpertAssignmentThreshold = 64;

} // namespace

mx::array MingModel::FeedForward(const mx::array &hidden_states,
                                 const std::string &weight_prefix,
                                 int layer_index) const {
  if (layer_index < config_.dense_layer_count) {
    return checkpoint_.SwiGlu(hidden_states, weight_prefix);
  } else {
  }
  const int token_count = hidden_states.shape(1);
  const int expert_group_count = config_.expert_group_count;
  const int experts_per_group = config_.expert_count / expert_group_count;
  const int selected_expert_count = config_.selected_expert_count;
  const auto routing_scores = mx::sigmoid(mx::matmul(
      mx::astype(hidden_states, mx::float32),
      mx::transpose(mx::astype(
          checkpoint_.Weight(weight_prefix + ".gate.weight"), mx::float32))));
  auto selection_scores = mx::add(
      routing_scores, checkpoint_.Weight(weight_prefix + ".gate.expert_bias"));
  if (expert_group_count > 1) {
    const auto grouped_scores =
        mx::reshape(selection_scores,
                    {1, token_count, expert_group_count, experts_per_group});
    const auto group_scores = mx::sum(
        mx::topk(grouped_scores, std::min(2, experts_per_group), -1), -1);
    const auto selected_group_indices = mx::slice(
        mx::argpartition(group_scores, -config_.selected_expert_group_count,
                         -1),
        {0, 0, expert_group_count - config_.selected_expert_group_count},
        {1, token_count, expert_group_count});
    const auto allowed_group_mask =
        mx::any(mx::equal(mx::arange(expert_group_count),
                          mx::expand_dims(selected_group_indices, -1)),
                -2);
    selection_scores = mx::reshape(
        mx::where(mx::expand_dims(allowed_group_mask, -1), grouped_scores,
                  mx::array(-std::numeric_limits<float>::infinity())),
        routing_scores.shape());
  } else {
  }
  const auto selected_expert_indices =
      mx::slice(mx::argpartition(selection_scores, -selected_expert_count, -1),
                {0, 0, config_.expert_count - selected_expert_count},
                {1, token_count, config_.expert_count});
  auto routing_weights =
      mx::take_along_axis(routing_scores, selected_expert_indices, -1);
  if (selected_expert_count > 1) {
    routing_weights =
        mx::divide(routing_weights, mx::add(mx::sum(routing_weights, -1, true),
                                            mx::array(kRoutingWeightEpsilon)));
  } else {
  }
  routing_weights =
      mx::multiply(routing_weights, mx::array(config_.routed_scaling_factor));
  auto expert_inputs =
      mx::reshape(hidden_states, {1, token_count, 1, 1, config_.hidden_size});
  auto expert_indices = selected_expert_indices;
  std::optional<mx::array> inverse_sort_order;
  const bool should_sort_expert_assignments =
      selected_expert_indices.size() >= kSortedExpertAssignmentThreshold;
  if (should_sort_expert_assignments) {
    const auto flattened_expert_indices =
        mx::reshape(selected_expert_indices, {-1});
    const auto sort_order = mx::argsort(flattened_expert_indices);
    inverse_sort_order = mx::argsort(sort_order);
    expert_indices = mx::take(flattened_expert_indices, sort_order, 0);
    expert_inputs = mx::take(
        mx::reshape(expert_inputs, {token_count, 1, config_.hidden_size}),
        mx::floor_divide(sort_order,
                         mx::array(selected_expert_count, sort_order.dtype())),
        0);
  } else {
  }
  const auto gate_projection = checkpoint_.GatherLinear(
      expert_inputs, weight_prefix + ".switch_mlp.gate_proj", expert_indices,
      should_sort_expert_assignments);
  const auto up_projection = checkpoint_.GatherLinear(
      expert_inputs, weight_prefix + ".switch_mlp.up_proj", expert_indices,
      should_sort_expert_assignments);
  const auto activated_states =
      mx::multiply(omni_common::Silu(gate_projection), up_projection);
  auto expert_outputs =
      mx::squeeze(checkpoint_.GatherLinear(
                      activated_states, weight_prefix + ".switch_mlp.down_proj",
                      expert_indices, should_sort_expert_assignments),
                  -2);
  if (should_sort_expert_assignments) {
    expert_outputs = mx::reshape(
        mx::take(expert_outputs, *inverse_sort_order, 0),
        {1, token_count, selected_expert_count, config_.hidden_size});
  } else {
  }
  const auto weighted_expert_outputs =
      mx::multiply(mx::astype(expert_outputs, mx::float32),
                   mx::expand_dims(routing_weights, -1));
  const auto routed_output =
      mx::astype(mx::sum(weighted_expert_outputs, -2), hidden_states.dtype());
  return mx::add(
      routed_output,
      checkpoint_.SwiGlu(hidden_states, weight_prefix + ".shared_experts"));
}

} // namespace ming3_asr
