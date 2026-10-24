// SPDX-License-Identifier: Apache-2.0
// dots.tts neural-network primitives shared by its acoustic components.
#include "dots_tts/dots_tts_acoustics.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

#include "common/layers.h"

namespace dots_tts::nn {

namespace mx = mlx::core;

mx::array RMSNorm::operator()(const mx::array &x) const {
  return mx::fast::rms_norm(x, std::optional<mx::array>(weight), eps);
}

mx::array LayerNorm::operator()(const mx::array &x) const {
  return mx::fast::layer_norm(x, std::nullopt, std::nullopt, eps);
}

mx::array ReplicatePad(const mx::array &x, int left, int right) {
  if (left == 0 && right == 0) {
    return x;
  }
  std::vector<mx::array> pieces;
  if (left > 0) {
    pieces.push_back(
        mx::broadcast_to(mx::slice(x, {0, 0, 0}, {x.shape(0), 1, x.shape(2)}),
                         {x.shape(0), left, x.shape(2)}));
  }
  pieces.push_back(x);
  if (right > 0) {
    pieces.push_back(
        mx::broadcast_to(mx::slice(x, {0, x.shape(1) - 1, 0},
                                   {x.shape(0), x.shape(1), x.shape(2)}),
                         {x.shape(0), right, x.shape(2)}));
  }
  return mx::concatenate(pieces, 1);
}

mx::array GeluTanh(const mx::array &x) {
  const mx::Dtype in_dtype = x.dtype();
  const mx::array xf = mx::astype(x, mx::float32);
  const mx::array inner = mx::multiply(
      mx::array(0.7978845608028654f, mx::float32),
      mx::add(xf, mx::multiply(mx::array(0.044715f, mx::float32),
                               mx::multiply(xf, mx::multiply(xf, xf)))));
  const mx::array out = mx::multiply(
      mx::array(0.5f, mx::float32),
      mx::multiply(xf, mx::add(mx::array(1.0f, mx::float32), mx::tanh(inner))));
  return mx::astype(out, in_dtype);
}

mx::array Mlp::operator()(const mx::array &x) const {
  mx::array hidden = fc1(x);
  hidden = activation == Activation::GeluTanh ? GeluTanh(hidden)
                                              : qwen3_asr::Silu(hidden);
  return fc2(hidden);
}

mx::array RotaryEmbedding::Apply(const mx::array &t) const {
  return mx::fast::rope(t, dim, /*traditional=*/false,
                        std::optional<float>(theta), 1.0f, 0);
}

MultiHeadAttention::MultiHeadAttention(int hidden_size_, int num_heads_,
                                       bool qkv_bias_, bool qk_norm_,
                                       NormLayer norm_layer_, bool rotary_bias_,
                                       float rotary_theta_)
    : hidden_size(hidden_size_), num_heads(num_heads_),
      head_dim(hidden_size_ / num_heads_),
      scale(std::pow(static_cast<double>(head_dim), -0.5)), qkv_bias(qkv_bias_),
      qk_norm(qk_norm_), rotary_bias(rotary_bias_), rotary_theta(rotary_theta_),
      norm_layer(norm_layer_) {
  if (hidden_size % num_heads != 0) {
    throw std::invalid_argument(
        "dots.tts attention hidden_size must be divisible by num_heads");
  }
  if (rotary_bias) {
    rotary = RotaryEmbedding(head_dim, rotary_theta);
  }
}

mx::array
MultiHeadAttention::operator()(const mx::array &q,
                               const std::optional<mx::array> &k,
                               const std::optional<mx::array> &v,
                               const std::optional<mx::array> &mask) const {
  const mx::array &key_input = k.has_value() ? *k : q;
  const mx::array &value_input = v.has_value() ? *v : q;
  const mx::array projected_queries = q_proj(q);
  const mx::array projected_keys = k_proj(key_input);
  const mx::array projected_values = v_proj(value_input);

  const int batch_size = projected_queries.shape(0);
  const int query_length = projected_queries.shape(1);
  // note (guozhihao-224): [B, N, H*D] -> [B, H, N, D]. Each tensor splits by
  // its own length so a cross-attention query can see a longer key cache and
  // the output stays q-length.
  const auto split_heads = [&](const mx::array &projected) {
    return mx::transpose(mx::reshape(projected, {batch_size, projected.shape(1),
                                                 num_heads, head_dim}),
                         {0, 2, 1, 3});
  };
  mx::array query_heads = split_heads(projected_queries);
  mx::array key_heads = split_heads(projected_keys);
  const mx::array value_heads = split_heads(projected_values);

  if (qk_norm) {
    if (!q_norm_weight.has_value() || !k_norm_weight.has_value()) {
      throw std::runtime_error("attention qk-norm is missing norm weights");
    }
    query_heads = RMSNorm(*q_norm_weight)(query_heads);
    key_heads = RMSNorm(*k_norm_weight)(key_heads);
  }

  if (rotary_bias) {
    query_heads = rotary->Apply(query_heads);
    key_heads = rotary->Apply(key_heads);
  }

  std::optional<mx::array> attn_bias;
  if (mask.has_value()) {
    attn_bias = mx::where(
        *mask, mx::array(0.0f, mx::float32),
        mx::array(-std::numeric_limits<float>::infinity(), mx::float32));
    *attn_bias = mx::astype(*attn_bias, query_heads.dtype());
  }

  const mx::array attended =
      attn_bias.has_value()
          ? mx::fast::scaled_dot_product_attention(query_heads, key_heads,
                                                   value_heads, scale,
                                                   /*mask_mode=*/"", attn_bias)
          : mx::fast::scaled_dot_product_attention(query_heads, key_heads,
                                                   value_heads, scale);

  const mx::array merged = mx::reshape(mx::transpose(attended, {0, 2, 1, 3}),
                                       {batch_size, query_length, -1});
  return o_proj(merged);
}

} // namespace dots_tts::nn
