// SPDX-License-Identifier: Apache-2.0
#include "haru/haru_model.h"

#include <cmath>

namespace haru {

namespace mx = mlx::core;

namespace {

      // AudioProjector's RMS norm epsilon.
constexpr float kAudioProjectorEps = 1e-6f;

// The gated norm's output: silu(gate) * normed in float32, in hidden's dtype.
std::vector<mx::array> PreciseSwigluGraph(const std::vector<mx::array> &inputs) {
  const mx::array gate = mx::astype(inputs[1], mx::float32);
  const mx::array activated = mx::multiply(gate, mx::sigmoid(gate));
  return {mx::astype(mx::multiply(activated, mx::astype(inputs[2], mx::float32)),
                     inputs[0].dtype())};
}

mx::array PreciseSwiglu(const mx::array &hidden, const mx::array &gate,
                        const mx::array &normed) {
  static const auto compiled = mx::compile(PreciseSwigluGraph, true);
  return compiled({hidden, gate, normed})[0];
}

std::vector<mx::array> DecayGraph(const std::vector<mx::array> &inputs) {
  const mx::array &a_log = inputs[0];
  // The sum is computed in float32: softplus(a.astype(fp32) + dt_bias).
  const mx::array shifted =
      mx::add(mx::astype(inputs[1], mx::float32), mx::astype(inputs[2], mx::float32));
  const mx::array softplus =
      mx::logaddexp(shifted, mx::array(0.0f, mx::float32));
  return {mx::exp(mx::multiply(
      mx::negative(mx::exp(mx::astype(a_log, mx::float32))), softplus))};
}

// The gated delta rule's per-step decay exp(-exp(A_log) * softplus(a + dt_bias)).
mx::array Decay(const mx::array &a_log, const mx::array &a,
                const mx::array &dt_bias) {
  static const auto compiled = mx::compile(DecayGraph, true);
  return compiled({a_log, a, dt_bias})[0];
}

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

} // namespace

HaruModel::HaruModel(const std::filesystem::path &model_directory,
                     int quantize_bits)
    : config_(HaruConfig::FromFile(model_directory / "config.json")),
      checkpoint_(omni_common::LoadSafetensors(model_directory),
                  omni_common::ReadJson(model_directory / "config.json")),
      layer_count_(config_.lm.num_layers()),
      rotary_dims_(static_cast<int>(config_.lm.head_dim *
                                    config_.lm.partial_rotary_factor)),
      latent_scale_(mx::astype(checkpoint_.Weight("latent_scale"), mx::float32)),
      latent_bias_(mx::astype(checkpoint_.Weight("latent_bias"), mx::float32)) {
  if (quantize_bits == 0) {
    return;
  }
  checkpoint_.QuantizeLinear("lm.embed_tokens", 64, quantize_bits);
  for (int layer = 0; layer < layer_count_; ++layer) {
    const std::string prefix = "lm.layers." + std::to_string(layer);
    const std::string block =
        config_.lm.layer_types[layer] == kLinear ? prefix + ".linear_attn"
                                                 : prefix + ".self_attn";
    if (config_.lm.layer_types[layer] == kLinear) {
      checkpoint_.QuantizeLinear(block + ".in_proj_qkv", 64, quantize_bits);
      checkpoint_.QuantizeLinear(block + ".in_proj_z", 64, quantize_bits);
      checkpoint_.QuantizeLinear(block + ".in_proj_b", 64, quantize_bits);
      checkpoint_.QuantizeLinear(block + ".in_proj_a", 64, quantize_bits);
      checkpoint_.QuantizeLinear(block + ".out_proj", 64, quantize_bits);
    } else {
      checkpoint_.QuantizeLinear(block + ".q_proj", 64, quantize_bits);
      checkpoint_.QuantizeLinear(block + ".k_proj", 64, quantize_bits);
      checkpoint_.QuantizeLinear(block + ".v_proj", 64, quantize_bits);
      checkpoint_.QuantizeLinear(block + ".o_proj", 64, quantize_bits);
    }
    checkpoint_.QuantizeLinear(prefix + ".mlp.gate_proj", 64, quantize_bits);
    checkpoint_.QuantizeLinear(prefix + ".mlp.up_proj", 64, quantize_bits);
    checkpoint_.QuantizeLinear(prefix + ".mlp.down_proj", 64, quantize_bits);
  }
}

std::vector<LayerCache> HaruModel::NewCaches() const {
  std::vector<LayerCache> caches(layer_count_);
  for (int layer = 0; layer < layer_count_; ++layer) {
    if (config_.lm.layer_types[layer] == kLinear) {
      caches[layer].convolution_state = mx::zeros(
          {1, config_.lm.linear_conv_kernel_dim - 1, config_.lm.conv_dim()},
          checkpoint_.Weight("lm.norm.weight").dtype());
      caches[layer].recurrent_state = mx::zeros(
          {1, config_.lm.linear_num_value_heads,
           config_.lm.linear_value_head_dim, config_.lm.linear_key_head_dim},
          mx::float32);
    }
  }
  return caches;
}

mx::array HaruModel::NormGain(const std::string &prefix) const {
  return mx::add(mx::astype(checkpoint_.Weight(prefix + ".weight"), mx::float32),
                 mx::array(1.0f, mx::float32));
}

mx::array HaruModel::Fp32Norm(const mx::array &x, const std::string &prefix,
                              float eps) const {
  return mx::astype(
      mx::fast::rms_norm(mx::astype(x, mx::float32), NormGain(prefix), eps),
      x.dtype());
}

mx::array HaruModel::AudioProject(const mx::array &latents,
                                  const std::string &prefix) const {
  const mx::array scaled = mx::multiply(
      mx::add(mx::astype(latents, mx::float32), latent_bias_), latent_scale_);
  const mx::Dtype dtype = checkpoint_.Weight(prefix + ".fc1.weight").dtype();
  const mx::array h = checkpoint_.Linear(mx::astype(scaled, dtype),
                                         prefix + ".fc1");
  const mx::array normed =
// AudioProjector's RMS norm epsilon.
      mx::fast::rms_norm(h, checkpoint_.Weight(prefix + ".norm.weight"), kAudioProjectorEps);
  return checkpoint_.Linear(normed, prefix + ".fc2");
}

mx::array HaruModel::Embed(const mx::array &tokens,
                           const std::optional<mx::array> &model_latents,
                           const std::optional<mx::array> &user_latents,
                           const std::optional<mx::array> &model_mask,
                           const std::optional<mx::array> &user_mask) const {
  mx::array x = checkpoint_.Embed(tokens, "lm.embed_tokens");
  const auto add_stream = [&](const std::optional<mx::array> &latents,
                              const std::optional<mx::array> &mask,
                              const std::string &prefix) {
    if (!latents.has_value()) {
      return;
    }
    mx::array y = AudioProject(*latents, prefix);
    if (mask.has_value()) {
      y = mx::multiply(y, mx::expand_dims(mx::astype(*mask, y.dtype()), -1));
    }
    x = mx::add(x, y);
  };
  add_stream(model_latents, model_mask, "model_audio_proj");
  add_stream(user_latents, user_mask, "user_audio_proj");
  return x;
}

mx::array HaruModel::GatedDeltaLayer(const mx::array &x,
                                     const std::string &prefix,
                                     LayerCache &cache) const {
  const int batch = x.shape(0);
  const int time = x.shape(1);
  const int key_dim = config_.lm.key_dim();
  const int convolution_dim = config_.lm.conv_dim();
  const int value_heads = config_.lm.linear_num_value_heads;
  const int value_head_dim = config_.lm.linear_value_head_dim;
  const int key_head_dim = config_.lm.linear_key_head_dim;
  const int kernel = config_.lm.linear_conv_kernel_dim;

  const mx::array qkv = checkpoint_.Linear(x, prefix + ".in_proj_qkv");
  const mx::array z =
      mx::reshape(checkpoint_.Linear(x, prefix + ".in_proj_z"),
                  {batch, time, value_heads, value_head_dim});
  const mx::array b = checkpoint_.Linear(x, prefix + ".in_proj_b");
  const mx::array a = checkpoint_.Linear(x, prefix + ".in_proj_a");

  const mx::array convolution_state =
      cache.convolution_state.has_value()
          ? *cache.convolution_state
          : mx::zeros({batch, kernel - 1, convolution_dim}, x.dtype());
  const mx::array convolution_input =
      mx::concatenate({convolution_state, qkv}, 1);
  const int kept = kernel - 1;
  cache.convolution_state = mx::contiguous(
      SliceAxis(convolution_input, 1, convolution_input.shape(1) - kept,
                convolution_input.shape(1)));
  // The checkpoint stores [out, in, kernel]; MLX conv1d takes [out, kernel, in].
  const mx::array conv_weight = mx::transpose(
      checkpoint_.Weight(prefix + ".conv1d.weight"), {0, 2, 1});
  const mx::array convolved = omni_common::Silu(
      mx::conv1d(convolution_input, conv_weight, 1, 0, 1, convolution_dim));

  const double inverse_scale = std::pow(key_head_dim, -0.5);
  const float l2_norm_eps =
      static_cast<float>(1e-6 * std::pow(inverse_scale, 2));
  mx::array q =
      mx::reshape(SliceLast(convolved, 0, key_dim),
                  {batch, time, config_.lm.linear_num_key_heads, key_head_dim});
  mx::array k = mx::reshape(
      SliceLast(convolved, key_dim, 2 * key_dim),
      {batch, time, config_.lm.linear_num_key_heads, key_head_dim});
  const mx::array v =
      mx::reshape(SliceLast(convolved, 2 * key_dim, convolution_dim),
                  {batch, time, value_heads, value_head_dim});
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
          : mx::zeros({batch, value_heads, value_head_dim, key_head_dim},
                      mx::float32);
  const std::vector<mx::array> updated =
      mx::fast::gated_delta_update(q, k, v, decay, beta, state);
  cache.recurrent_state = updated[1];

  const mx::array normed = mx::fast::rms_norm(
      updated[0], checkpoint_.Weight(prefix + ".norm.weight"), config_.lm.rms_norm_eps);
  const mx::array gated = PreciseSwiglu(updated[0], z, normed);
  return checkpoint_.Linear(mx::reshape(gated, {batch, time, -1}),
                            prefix + ".out_proj");
}

mx::array HaruModel::AttentionLayer(const mx::array &x,
                                    const std::string &prefix,
                                    qwen3_asr::KVCache &cache) const {
  const int batch = x.shape(0);
  const int time = x.shape(1);
  const int heads = config_.lm.num_attention_heads;
  const int kv_heads = config_.lm.num_key_value_heads;
  const int head_dim = config_.lm.head_dim;
  const mx::array projected =
      mx::reshape(checkpoint_.Linear(x, prefix + ".q_proj"),
                  {batch, time, heads, 2 * head_dim});
  mx::array queries = SliceLast(projected, 0, head_dim);
  const mx::array gate =
      mx::reshape(SliceLast(projected, head_dim, 2 * head_dim),
                  {batch, time, -1});
  queries = mx::transpose(Fp32Norm(queries, prefix + ".q_norm", config_.lm.rms_norm_eps),
                          {0, 2, 1, 3});
  mx::array keys = mx::transpose(
      Fp32Norm(mx::reshape(checkpoint_.Linear(x, prefix + ".k_proj"),
                           {batch, time, kv_heads, head_dim}),
               prefix + ".k_norm", config_.lm.rms_norm_eps),
      {0, 2, 1, 3});
  const mx::array values = mx::transpose(
      mx::reshape(checkpoint_.Linear(x, prefix + ".v_proj"),
                  {batch, time, kv_heads, head_dim}),
      {0, 2, 1, 3});
  queries = mx::fast::rope(queries, rotary_dims_, false, config_.lm.rope_theta,
                           1.0f, cache.offset());
  keys = mx::fast::rope(keys, rotary_dims_, false, config_.lm.rope_theta, 1.0f,
                        cache.offset());
  const auto [cached_keys, cached_values] = cache.UpdateAndFetch(keys, values);
  const mx::array attended = mx::fast::scaled_dot_product_attention(
      queries, cached_keys, cached_values,
      static_cast<float>(std::pow(static_cast<double>(head_dim), -0.5)),
      time > 1 ? "causal" : "");
  const mx::array output =
      mx::reshape(mx::transpose(attended, {0, 2, 1, 3}), {batch, time, -1});
  return checkpoint_.Linear(mx::multiply(output, mx::sigmoid(gate)),
                            prefix + ".o_proj");
}

mx::array HaruModel::MLP(const mx::array &x, const std::string &prefix) const {
  const mx::array gate = checkpoint_.Linear(x, prefix + ".gate_proj");
  const mx::array up = checkpoint_.Linear(x, prefix + ".up_proj");
  return checkpoint_.Linear(mx::multiply(omni_common::Silu(gate), up),
                            prefix + ".down_proj");
}

mx::array HaruModel::Backbone(const mx::array &embeddings,
                              std::vector<LayerCache> &caches) const {
  mx::array hidden = embeddings;
  for (int layer = 0; layer < layer_count_; ++layer) {
    const std::string prefix =
        "lm.layers." + std::to_string(layer);
    const mx::array normed =
        Fp32Norm(hidden, prefix + ".input_layernorm", config_.lm.rms_norm_eps);
    const mx::array mixed =
        config_.lm.layer_types[layer] == kLinear
            ? GatedDeltaLayer(normed, prefix + ".linear_attn", caches[layer])
            : AttentionLayer(normed, prefix + ".self_attn",
                             caches[layer].attention);
    const mx::array residual = mx::add(hidden, mixed);
    hidden = mx::add(
        residual,
        MLP(Fp32Norm(residual, prefix + ".post_attention_layernorm",
                     config_.lm.rms_norm_eps),
            prefix + ".mlp"));
  }
  return Fp32Norm(hidden, "lm.norm", config_.lm.rms_norm_eps);
}

mx::array HaruModel::Logits(const mx::array &hidden) const {
  return mx::astype(checkpoint_.TiedProjection(hidden, "lm.embed_tokens"),
                    mx::float32);
}

mx::array HaruModel::LogitsFp32(const mx::array &hidden,
                                const mx::array &tokens) const {
  const mx::array rows =
      mx::astype(checkpoint_.Embed(tokens, "lm.embed_tokens"), mx::float32);
  return mx::matmul(mx::astype(hidden, mx::float32), mx::transpose(rows));
}

mx::array HaruModel::UnscaleLatents(const mx::array &normalized) const {
  return mx::subtract(
      mx::divide(mx::astype(normalized, mx::float32), latent_scale_),
      latent_bias_);
}

} // namespace haru
