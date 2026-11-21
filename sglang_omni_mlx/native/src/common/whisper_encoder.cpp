// SPDX-License-Identifier: Apache-2.0
#include "common/whisper_encoder.h"

#include <cmath>

namespace omni_common {

namespace mx = mlx::core;

namespace {

constexpr float kLayerNormEps = 1e-5f;
constexpr float kRotaryBase = 10000.0f;

} // namespace

WeightMap WhisperConvMlxLayout(WeightMap weights, const std::string &prefix) {
  const std::string conv_prefix = prefix + ".conv";
  for (auto &[name, array] : weights) {
    if (name.rfind(conv_prefix, 0) == 0 && array.ndim() == 3) {
      array = mx::transpose(array, {0, 2, 1});
    } else {
    }
  }
  return weights;
}

mx::array WhisperEncoderLayer(const Checkpoint &checkpoint, const mx::array &x,
                              const std::string &layer_prefix, int head_count,
                              bool rotary_query_key) {
  const int batch = x.shape(0);
  const int length = x.shape(1);
  const int width = x.shape(2);
  const int head_dim = width / head_count;
  const mx::array normed = checkpoint.LayerNorm(
      x, layer_prefix + ".self_attn_layer_norm", kLayerNormEps);
  const auto project = [&](const std::string &name) {
    return mx::transpose(
        mx::reshape(
            checkpoint.Linear(normed, layer_prefix + ".self_attn." + name),
            {batch, length, head_count, head_dim}),
        {0, 2, 1, 3});
  };
  mx::array queries = project("q_proj");
  mx::array keys = project("k_proj");
  if (rotary_query_key) {
    // Note (AzizbekMuminoff): rotary positions turn the first half of each
    // head in adjacent pairs; the second half passes through.
    queries = mx::fast::rope(queries, head_dim / 2, true, kRotaryBase, 1.0f, 0);
    keys = mx::fast::rope(keys, head_dim / 2, true, kRotaryBase, 1.0f, 0);
  } else {
  }
  const mx::array attended = mx::fast::scaled_dot_product_attention(
      queries, keys, project("v_proj"),
      static_cast<float>(std::pow(static_cast<double>(head_dim), -0.5)));
  const mx::array hidden = mx::add(
      x, checkpoint.Linear(mx::reshape(mx::transpose(attended, {0, 2, 1, 3}),
                                       {batch, length, width}),
                           layer_prefix + ".self_attn.out_proj"));
  return mx::add(
      hidden,
      checkpoint.Linear(
          Gelu(checkpoint.Linear(
              checkpoint.LayerNorm(hidden, layer_prefix + ".final_layer_norm",
                                   kLayerNormEps),
              layer_prefix + ".fc1")),
          layer_prefix + ".fc2"));
}

} // namespace omni_common
