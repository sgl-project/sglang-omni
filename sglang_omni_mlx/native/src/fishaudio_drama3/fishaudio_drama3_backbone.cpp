// SPDX-License-Identifier: Apache-2.0
// Drama 3's backbone: gated delta and gated attention layers, each followed by
// a mixture of experts, over frame embeddings.
#include <cmath>
#include <optional>
#include <vector>

#include "fishaudio_drama3/fishaudio_drama3_model.h"

namespace fishaudio_drama3 {

namespace mx = mlx::core;

namespace {

constexpr const char *kBackbone = "backbone.model";
// Expert gathers sort their rows once a call routes this many.
constexpr int kSortedExpertRows = 64;

std::vector<mx::array> SwigluGraph(const std::vector<mx::array> &inputs) {
  const mx::array &gate = inputs[0];
  return {mx::multiply(mx::multiply(gate, mx::sigmoid(gate)), inputs[1])};
}

// silu(gate) * x, fused into one kernel.
mx::array Swiglu(const mx::array &gate, const mx::array &x) {
  static const auto compiled = mx::compile(SwigluGraph, true);
  return compiled({gate, x})[0];
}

std::vector<mx::array>
PreciseSwigluGraph(const std::vector<mx::array> &inputs) {
  const mx::array gate = mx::astype(inputs[1], mx::float32);
  const mx::array activated = mx::multiply(gate, mx::sigmoid(gate));
  return {
      mx::astype(mx::multiply(activated, mx::astype(inputs[2], mx::float32)),
                 inputs[0].dtype())};
}

// The gated norm's output: silu(gate) * normed in float32, in hidden's dtype.
mx::array PreciseSwiglu(const mx::array &hidden, const mx::array &gate,
                        const mx::array &normed) {
  static const auto compiled = mx::compile(PreciseSwigluGraph, true);
  return compiled({hidden, gate, normed})[0];
}

std::vector<mx::array> DecayGraph(const std::vector<mx::array> &inputs) {
  const mx::array &a_log = inputs[0];
  const mx::array shifted = mx::add(inputs[1], inputs[2]);
  const mx::array softplus =
      mx::logaddexp(shifted, mx::array(0, shifted.dtype()));
  return {mx::exp(mx::multiply(
      mx::negative(mx::exp(mx::astype(a_log, mx::float32))), softplus))};
}

// The gated delta rule's per-step decay exp(-exp(A_log) * softplus(a +
// dt_bias)).
mx::array Decay(const mx::array &a_log, const mx::array &a,
                const mx::array &dt_bias) {
  static const auto compiled = mx::compile(DecayGraph, true);
  return compiled({a_log, a, dt_bias})[0];
}

} // namespace

mx::array SliceLast(const mx::array &x, int begin, int end) {
  mx::Shape start(x.ndim(), 0);
  mx::Shape stop = x.shape();
  start.back() = begin;
  stop.back() = end;
  return mx::slice(x, start, stop);
}

mx::array SliceAxis(const mx::array &x, int axis, int begin, int end) {
  mx::Shape start(x.ndim(), 0);
  mx::Shape stop = x.shape();
  start[axis] = begin;
  stop[axis] = end;
  return mx::slice(x, start, stop);
}

mx::array Drama3Model::Fp32Norm(const mx::array &x, const std::string &prefix,
                                float eps) const {
  return mx::astype(mx::fast::rms_norm(mx::astype(x, mx::float32),
                                       checkpoint_.Weight(prefix + ".weight"),
                                       eps),
                    x.dtype());
}

mx::array Drama3Model::Embed(const mx::array &frames) const {
  const mx::array text =
      checkpoint_.Embed(mx::squeeze(SliceLast(frames, 0, 1), -1),
                        std::string(kBackbone) + ".embed_tokens");
  const mx::array summed =
      mx::add(text, EmbeddingSum(frames, 1, 1 + backbone_codebook_count_));
  const mx::array mixed = mx::multiply(
      summed, omni_common::Scalar(
                  std::pow(1.0 + backbone_codebook_count_, -0.5), summed));
  const mx::array is_semantic =
      mx::expand_dims(IsSemantic(mx::squeeze(SliceLast(frames, 0, 1), -1)), -1);
  return mx::where(is_semantic, mixed, text);
}

mx::array Drama3Model::GatedDeltaLayer(const mx::array &x,
                                       const std::string &prefix,
                                       LayerCache &cache) const {
  const int batch = x.shape(0);
  const int time = x.shape(1);
  const int key_dim = linear_key_head_dim_ * linear_key_head_count_;
  const int value_dim = linear_value_head_dim_ * linear_value_head_count_;
  const int convolution_dim = 2 * key_dim + value_dim;
  const mx::array qkv = checkpoint_.Linear(x, prefix + ".in_proj_qkv");
  const mx::array z = mx::reshape(
      checkpoint_.Linear(x, prefix + ".in_proj_z"),
      {batch, time, linear_value_head_count_, linear_value_head_dim_});
  const mx::array b = checkpoint_.Linear(x, prefix + ".in_proj_b");
  const mx::array a = checkpoint_.Linear(x, prefix + ".in_proj_a");

  const mx::array convolution_state =
      cache.convolution_state.has_value()
          ? *cache.convolution_state
          : mx::zeros({batch, convolution_kernel_size_ - 1, convolution_dim},
                      x.dtype());
  const mx::array convolution_input =
      mx::concatenate({convolution_state, qkv}, 1);
  const int kept = convolution_kernel_size_ - 1;
  cache.convolution_state = mx::contiguous(
      SliceAxis(convolution_input, 1, convolution_input.shape(1) - kept,
                convolution_input.shape(1)));
  const mx::array convolved = omni_common::Silu(mx::conv1d(
      convolution_input, checkpoint_.Weight(prefix + ".conv1d.weight"), 1, 0, 1,
      convolution_dim));

  const double inverse_scale = std::pow(linear_key_head_dim_, -0.5);
  // RMS normalization averages squares; L2 normalization sums them.
  const float l2_norm_eps =
      static_cast<float>(1e-6 * std::pow(inverse_scale, 2));
  mx::array q =
      mx::reshape(SliceLast(convolved, 0, key_dim),
                  {batch, time, linear_key_head_count_, linear_key_head_dim_});
  mx::array k =
      mx::reshape(SliceLast(convolved, key_dim, 2 * key_dim),
                  {batch, time, linear_key_head_count_, linear_key_head_dim_});
  const mx::array v = mx::reshape(
      SliceLast(convolved, 2 * key_dim, convolution_dim),
      {batch, time, linear_value_head_count_, linear_value_head_dim_});
  q = mx::multiply(omni_common::Scalar(std::pow(inverse_scale, 2), q.dtype()),
                   mx::fast::rms_norm(q, std::nullopt, l2_norm_eps));
  k = mx::multiply(omni_common::Scalar(inverse_scale, k.dtype()),
                   mx::fast::rms_norm(k, std::nullopt, l2_norm_eps));

  const mx::array beta = mx::sigmoid(b);
  const mx::array decay = Decay(checkpoint_.Weight(prefix + ".A_log"), a,
                                checkpoint_.Weight(prefix + ".dt_bias"));
  const mx::array state =
      cache.recurrent_state.has_value()
          ? *cache.recurrent_state
          : mx::zeros({batch, linear_value_head_count_, linear_value_head_dim_,
                       linear_key_head_dim_},
                      mx::float32);
  const std::vector<mx::array> updated =
      mx::fast::gated_delta_update(q, k, v, decay, beta, state);
  cache.recurrent_state = updated[1];

  const mx::array normed = mx::fast::rms_norm(
      updated[0], checkpoint_.Weight(prefix + ".norm.weight"), rms_norm_eps_);
  const mx::array gated = PreciseSwiglu(updated[0], z, normed);
  return checkpoint_.Linear(mx::reshape(gated, {batch, time, -1}),
                            prefix + ".out_proj");
}

mx::array Drama3Model::AttentionLayer(const mx::array &x,
                                      const std::string &prefix,
                                      qwen3_asr::KVCache &cache) const {
  const int batch = x.shape(0);
  const int time = x.shape(1);
  const mx::array projected =
      mx::reshape(checkpoint_.Linear(x, prefix + ".q_proj"),
                  {batch, time, attention_head_count_, 2 * head_dim_});
  mx::array queries = SliceLast(projected, 0, head_dim_);
  const mx::array gate = mx::reshape(
      SliceLast(projected, head_dim_, 2 * head_dim_), {batch, time, -1});
  queries = mx::transpose(Fp32Norm(queries, prefix + ".q_norm", rms_norm_eps_),
                          {0, 2, 1, 3});
  mx::array keys = mx::transpose(
      Fp32Norm(mx::reshape(checkpoint_.Linear(x, prefix + ".k_proj"),
                           {batch, time, key_value_head_count_, head_dim_}),
               prefix + ".k_norm", rms_norm_eps_),
      {0, 2, 1, 3});
  const mx::array values = mx::transpose(
      mx::reshape(checkpoint_.Linear(x, prefix + ".v_proj"),
                  {batch, time, key_value_head_count_, head_dim_}),
      {0, 2, 1, 3});
  queries = mx::fast::rope(queries, rotary_dims_, false, rope_theta_, 1.0f,
                           cache.offset());
  keys = mx::fast::rope(keys, rotary_dims_, false, rope_theta_, 1.0f,
                        cache.offset());
  const auto [cached_keys, cached_values] = cache.UpdateAndFetch(keys, values);
  const mx::array attended = mx::fast::scaled_dot_product_attention(
      queries, cached_keys, cached_values,
      static_cast<float>(std::pow(static_cast<double>(head_dim_), -0.5)),
      time > 1 ? "causal" : "");
  const mx::array output =
      mx::reshape(mx::transpose(attended, {0, 2, 1, 3}), {batch, time, -1});
  return checkpoint_.Linear(mx::multiply(output, mx::sigmoid(gate)),
                            prefix + ".o_proj");
}

mx::array Drama3Model::MixtureOfExperts(const mx::array &x,
                                        const std::string &prefix) const {
  const mx::array gates = mx::softmax(checkpoint_.Linear(x, prefix + ".gate"),
                                      std::vector<int>{-1}, true);
  const mx::array experts =
      SliceLast(mx::argpartition(gates, expert_count_ - experts_per_token_, -1),
                expert_count_ - experts_per_token_, expert_count_);
  mx::array scores = mx::take_along_axis(gates, experts, -1);
  scores = mx::divide(scores, mx::sum(scores, -1, true));

  // Rows [..., 1, 1, dim] per token, each gathered by its experts; once
  // enough rows route, they are sorted by expert so weights stream in order.
  const std::string switch_prefix = prefix + ".switch_mlp";
  const bool is_sorted = experts.size() >= kSortedExpertRows;
  mx::array rows = mx::expand_dims(x, {-2, -3});
  mx::array expert_ids = experts;
  std::optional<mx::array> inverse_order;
  if (is_sorted) {
    const mx::array flat_experts = mx::flatten(experts);
    const mx::array order = mx::argsort(flat_experts);
    inverse_order = mx::argsort(order);
    rows = mx::take(
        mx::flatten(rows, 0, -3),
        mx::floor_divide(order, mx::array(experts_per_token_, order.dtype())),
        0);
    expert_ids = mx::take(flat_experts, order);
  } else {
  }
  const auto expert_linear = [&](const mx::array &input,
                                 const std::string &name) {
    const std::string weight_prefix = switch_prefix + "." + name;
    const omni_common::QuantizationConfig &quantization =
        checkpoint_.Quantization(weight_prefix);
    return mx::gather_qmm(input, checkpoint_.Weight(weight_prefix + ".weight"),
                          checkpoint_.Weight(weight_prefix + ".scales"),
                          checkpoint_.Weight(weight_prefix + ".biases"),
                          std::nullopt, expert_ids, true,
                          quantization.group_size, quantization.bits,
                          quantization.mode, std::nullopt, is_sorted);
  };
  const mx::array up = expert_linear(rows, "up_proj");
  const mx::array gate = expert_linear(rows, "gate_proj");
  mx::array routed = expert_linear(Swiglu(gate, up), "down_proj");
  if (is_sorted) {
    mx::Shape unsorted_shape = experts.shape();
    unsorted_shape.push_back(routed.shape(-2));
    unsorted_shape.push_back(routed.shape(-1));
    routed = mx::reshape(mx::take(routed, *inverse_order, 0), unsorted_shape);
  } else {
  }
  routed = mx::squeeze(routed, -2);
  const mx::array combined =
      mx::sum(mx::multiply(routed, mx::expand_dims(scores, -1)), -2);

  const std::string shared_prefix = prefix + ".shared_expert";
  const mx::array shared = checkpoint_.Linear(
      Swiglu(checkpoint_.Linear(x, shared_prefix + ".gate_proj"),
             checkpoint_.Linear(x, shared_prefix + ".up_proj")),
      shared_prefix + ".down_proj");
  const mx::array shared_gated = mx::multiply(
      mx::sigmoid(checkpoint_.Linear(x, prefix + ".shared_expert_gate")),
      shared);
  return mx::add(combined, shared_gated);
}

mx::array Drama3Model::Backbone(const mx::array &embeddings,
                                std::vector<LayerCache> &caches) const {
  mx::array hidden = embeddings;
  for (int layer = 0; layer < layer_count_; ++layer) {
    const std::string prefix =
        std::string(kBackbone) + ".layers." + std::to_string(layer);
    const bool is_linear = (layer + 1) % 4 != 0;
    const mx::array normed =
        Fp32Norm(hidden, prefix + ".input_layernorm", rms_norm_eps_);
    const mx::array mixed =
        is_linear
            ? GatedDeltaLayer(normed, prefix + ".linear_attn", caches[layer])
            : AttentionLayer(normed, prefix + ".self_attn",
                             caches[layer].attention);
    const mx::array residual = mx::add(hidden, mixed);
    hidden = mx::add(
        residual, MixtureOfExperts(
                      Fp32Norm(residual, prefix + ".post_attention_layernorm",
                               rms_norm_eps_),
                      prefix + ".mlp"));
  }
  return Fp32Norm(hidden, std::string(kBackbone) + ".norm", rms_norm_eps_);
}

} // namespace fishaudio_drama3
