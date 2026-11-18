// SPDX-License-Identifier: Apache-2.0
#include "common/fastconformer.h"

#include <cmath>
#include <vector>

namespace omni_common {

namespace mx = mlx::core;

namespace {

constexpr float kPositionTimescale = 10000.0f;

} // namespace

mx::array RelativePositionTable(int length, int width) {
  const int row_count = 2 * length - 1;
  const mx::array positions =
      mx::arange(static_cast<double>(length - 1), static_cast<double>(-length),
                 -1.0, mx::float32);
  const mx::array inverse_timescales = mx::exp(mx::multiply(
      mx::arange(0, width, 2, mx::float32),
      mx::array(static_cast<float>(-std::log(kPositionTimescale) / width))));
  const mx::array angles = mx::multiply(mx::expand_dims(positions, 1),
                                        mx::expand_dims(inverse_timescales, 0));
  return mx::reshape(mx::stack({mx::sin(angles), mx::cos(angles)}, -1),
                     {1, row_count, width});
}

mx::array ConformerFeedForward(const Checkpoint &checkpoint, const mx::array &x,
                               const std::string &prefix) {
  return checkpoint.Linear(Silu(checkpoint.Linear(x, prefix + ".linear1")),
                           prefix + ".linear2");
}

mx::array RelativePositionAttention(const Checkpoint &checkpoint,
                                    const mx::array &queries,
                                    const mx::array &keys_values,
                                    const mx::array &relative_positions,
                                    const std::string &prefix, int head_count) {
  const int query_count = queries.shape(1);
  const int key_count = keys_values.shape(1);
  const int position_count = relative_positions.shape(1);
  const int width = queries.shape(2);
  const int head_width = width / head_count;
  const float scale =
      static_cast<float>(std::pow(static_cast<double>(head_width), -0.5));
  const auto heads = [&](const mx::array &projected, int length) {
    return mx::transpose(
        mx::reshape(projected, {1, length, head_count, head_width}),
        {0, 2, 1, 3});
  };
  const mx::array query_heads =
      mx::reshape(checkpoint.Linear(queries, prefix + ".linear_q"),
                  {1, query_count, head_count, head_width});
  const mx::Dtype dtype = query_heads.dtype();
  const mx::array content_queries = mx::transpose(
      mx::add(query_heads,
              mx::astype(checkpoint.Weight(prefix + ".pos_bias_u"), dtype)),
      {0, 2, 1, 3});
  const mx::array position_queries = mx::transpose(
      mx::add(query_heads,
              mx::astype(checkpoint.Weight(prefix + ".pos_bias_v"), dtype)),
      {0, 2, 1, 3});
  const mx::array keys =
      heads(checkpoint.Linear(keys_values, prefix + ".linear_k"), key_count);
  const mx::array values =
      heads(checkpoint.Linear(keys_values, prefix + ".linear_v"), key_count);
  const mx::array positions =
      heads(checkpoint.Linear(relative_positions, prefix + ".linear_pos"),
            position_count);
  const mx::array position_scores =
      mx::multiply(RelativeShift(mx::matmul(position_queries,
                                            mx::swapaxes(positions, -2, -1)),
                                 key_count),
                   Scalar(scale, dtype));
  const mx::array attended = mx::fast::scaled_dot_product_attention(
      content_queries, keys, values, scale, /*window_size=*/-1, "array",
      position_scores);
  return checkpoint.Linear(mx::reshape(mx::transpose(attended, {0, 2, 1, 3}),
                                       {1, query_count, width}),
                           prefix + ".linear_out");
}

mx::array ConformerConvolution(const Checkpoint &checkpoint, const mx::array &x,
                               const std::string &prefix) {
  const std::vector<mx::array> halves =
      mx::split(checkpoint.Conv1d(x, prefix + ".pointwise_conv1"), 2, -1);
  const mx::array gated = mx::multiply(halves[0], mx::sigmoid(halves[1]));
  const int padding =
      (checkpoint.Weight(prefix + ".depthwise_conv.weight").shape(1) - 1) / 2;
  // Note (XXXXRT666): padded apart, since MLX's depthwise kernel only takes
  // unpadded input.
  const mx::array convolved =
      checkpoint.Conv1d(mx::pad(gated, {{0, 0}, {padding, padding}, {0, 0}}),
                        prefix + ".depthwise_conv", 1, 0, 1, gated.shape(2));
  return checkpoint.Conv1d(Silu(convolved), prefix + ".pointwise_conv2");
}

} // namespace omni_common
