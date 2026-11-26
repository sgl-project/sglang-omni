// SPDX-License-Identifier: Apache-2.0
#include "common/sanm_encoder.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace omni_common {

namespace mx = mlx::core;

namespace {

constexpr float kLayerNormEps = 1e-5f;
constexpr double kPositionTimescale = 10000.0;

mx::array EncoderLayer(const Checkpoint &checkpoint, const mx::array &x,
                       const std::string &prefix, bool residual,
                       const SanmEncoderConfig &config) {
  const std::string attention = prefix + ".self_attn";
  const int width = config.model_width;
  const int head_count = config.head_count;
  const int head_width = width / head_count;
  const int time = x.shape(1);
  const mx::array normed =
      checkpoint.LayerNorm(x, prefix + ".norm1", kLayerNormEps);
  const std::vector<mx::array> qkv =
      mx::split(checkpoint.Linear(normed, attention + ".linear_q_k_v"), 3, -1);
  // Note (Dayuxiaoshui): the FSMN memory, a depthwise convolution over the
  // values padded to keep their length, added back onto them.
  const int left_padding = (config.kernel_size - 1) / 2 + config.sanm_shift;
  const int right_padding = config.kernel_size - 1 - left_padding;
  const mx::array memory = mx::add(
      mx::conv1d(
          mx::pad(qkv[2], {{0, 0}, {left_padding, right_padding}, {0, 0}}),
          checkpoint.Weight(attention + ".fsmn_block.weight"), 1, 0, 1, width),
      qkv[2]);
  const auto heads = [&](const mx::array &projected) {
    return mx::transpose(
        mx::reshape(projected, {1, time, head_count, head_width}),
        {0, 2, 1, 3});
  };
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_width));
  const mx::array attended = mx::fast::scaled_dot_product_attention(
      heads(qkv[0]), heads(qkv[1]), heads(qkv[2]), scale);
  const mx::array attention_output = mx::add(
      checkpoint.Linear(
          mx::reshape(mx::transpose(attended, {0, 2, 1, 3}), {1, time, width}),
          attention + ".linear_out"),
      memory);
  // Note (Dayuxiaoshui): the first layer widens its input, so it has no
  // residual around attention; as in the Swift port, only a width change
  // drops it.
  const mx::array hidden =
      residual ? mx::add(x, attention_output) : attention_output;
  const mx::array expanded = checkpoint.Linear(
      checkpoint.LayerNorm(hidden, prefix + ".norm2", kLayerNormEps),
      prefix + ".feed_forward.w_1");
  const mx::array feed_forward = checkpoint.Linear(
      mx::maximum(expanded, mx::array(0.0f, expanded.dtype())),
      prefix + ".feed_forward.w_2");
  return mx::add(hidden, feed_forward);
}

} // namespace

mx::array SanmEncode(const Checkpoint &checkpoint, const mx::array &features,
                     const SanmEncoderConfig &config) {
  mx::array x = mx::multiply(
      features, Scalar(static_cast<float>(
                           std::sqrt(static_cast<double>(config.model_width))),
                       features));
  // Note (Dayuxiaoshui): sinusoids at positions 1 onward, sines then
  // cosines, in MLX ops as the Swift port builds them.
  const int time = x.shape(1);
  const int half_width = std::max(config.input_size / 2, 1);
  std::vector<float> positions(time);
  for (int index = 0; index < time; ++index) {
    positions[index] = static_cast<float>(index + 1);
  }
  const float log_increment =
      static_cast<float>(std::log(kPositionTimescale) /
                         static_cast<double>(std::max(half_width - 1, 1)));
  const mx::array inverse_timescales = mx::exp(mx::multiply(
      mx::arange(half_width, mx::float32), mx::array(-log_increment)));
  const mx::array scaled_time = mx::multiply(
      mx::expand_dims(mx::array(positions.data(), {time}, mx::float32), 1),
      mx::expand_dims(inverse_timescales, 0));
  const mx::array encoding =
      mx::concatenate({mx::sin(scaled_time), mx::cos(scaled_time)}, 1);
  x = mx::add(x, mx::astype(mx::expand_dims(encoding, 0), x.dtype()));

  x = EncoderLayer(checkpoint, x, "encoder.encoders0.0",
                   config.input_size == config.model_width, config);
  for (int layer = 0; layer < config.block_count - 1; ++layer) {
    x = EncoderLayer(checkpoint, x, "encoder.encoders." + std::to_string(layer),
                     true, config);
  }
  x = checkpoint.LayerNorm(x, "encoder.after_norm", kLayerNormEps);
  for (int layer = 0; layer < config.tp_block_count; ++layer) {
    x = EncoderLayer(checkpoint, x,
                     "encoder.tp_encoders." + std::to_string(layer), true,
                     config);
  }
  return checkpoint.LayerNorm(x, "encoder.tp_norm", kLayerNormEps);
}

} // namespace omni_common
