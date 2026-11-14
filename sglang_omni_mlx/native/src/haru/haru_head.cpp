// SPDX-License-Identifier: Apache-2.0
#include "haru/haru_head.h"

#include <cmath>

namespace haru {

namespace mx = mlx::core;

namespace {

constexpr int kFrequencyDim = 256;
// The ScalarEmbed sinusoidal frequency base.
constexpr float kFrequencyBase = 10000.0f;

// x sliced to [begin, end) along its last axis.
mx::array SliceLast(const mx::array &x, int begin, int end) {
  mx::Shape start(x.ndim(), 0);
  mx::Shape stop = x.shape();
  start.back() = begin;
  stop.back() = end;
  return mx::slice(x, start, stop);
}

// rms norm computed in float32 with no gain, back in x's dtype.
mx::array HeadRmsNorm(const mx::array &x, float eps) {
  return mx::astype(mx::fast::rms_norm(mx::astype(x, mx::float32),
                                       std::nullopt, eps),
                    x.dtype());
}

// The sinusoidal frequency basis [128], shared by every scalar embedding.
mx::array BuildFreqs() {
  const int half = kFrequencyDim / 2;
  const mx::array index = mx::arange(0, half, mx::float32);
  return mx::exp(mx::multiply(
      mx::array(-std::log(kFrequencyBase), mx::float32),
      mx::divide(index, mx::array(static_cast<float>(half), mx::float32))));
}

} // namespace

MeanFlowHead::MeanFlowHead(const omni_common::Checkpoint &checkpoint,
                           const HeadConfig &config)
    : checkpoint_(checkpoint), config_(config), freqs_(BuildFreqs()) {}

mx::array MeanFlowHead::ScalarEmbed(const std::string &prefix, float value,
                                    mx::Dtype dtype) const {
  const mx::array args =
      mx::multiply(mx::array(value, dtype), freqs_);  // [1, 128]
  const mx::array features = mx::astype(
      mx::concatenate({mx::cos(args), mx::sin(args)}, -1), dtype);  // [1, 256]
  const mx::array h = checkpoint_.Linear(features, prefix + ".fc1");
  return checkpoint_.Linear(omni_common::Silu(h), prefix + ".fc2");
}

mx::array MeanFlowHead::Velocity(const mx::array &z, const mx::array &c) const {
  const int hidden = config_.hidden_size;
  mx::array x = checkpoint_.Linear(z, "head.latent_proj");
  const mx::array conditioned = omni_common::Silu(c);
  for (int i = 0; i < config_.num_layers; ++i) {
    const std::string prefix = "head.layers." + std::to_string(i);
    const mx::array modulation =
        checkpoint_.Linear(conditioned, prefix + ".ada");  // [B, 3*H]
    const mx::array shift = SliceLast(modulation, 0, hidden);
    const mx::array scale = SliceLast(modulation, hidden, 2 * hidden);
    const mx::array gate = SliceLast(modulation, 2 * hidden, 3 * hidden);
    const mx::array h = mx::multiply(
        HeadRmsNorm(x, config_.norm_eps),
        checkpoint_.Weight(prefix + ".norm.weight"));
    const mx::array modulated = mx::add(
        mx::multiply(h, mx::add(omni_common::Scalar(1.0f, scale.dtype()), scale)),
        shift);
    const mx::array ffn = checkpoint_.Linear(
        mx::multiply(omni_common::Silu(checkpoint_.Linear(modulated,
                                                          prefix + ".ffn.gate_proj")),
                     checkpoint_.Linear(modulated, prefix + ".ffn.up_proj")),
        prefix + ".ffn.down_proj");
    x = mx::add(x, mx::multiply(gate, ffn));
  }
  const mx::array final_modulation =
      checkpoint_.Linear(conditioned, "head.final.ada");  // [B, 2*H]
  const mx::array shift = SliceLast(final_modulation, 0, hidden);
  const mx::array scale = SliceLast(final_modulation, hidden, 2 * hidden);
  const mx::array out = mx::add(
      mx::multiply(HeadRmsNorm(x, config_.norm_eps),
                   mx::add(omni_common::Scalar(1.0f, scale.dtype()), scale)),
      shift);
  return checkpoint_.Linear(out, "head.final.proj");
}

mx::array MeanFlowHead::Sample(const mx::array &hidden, const mx::array &noise,
                               float guidance) const {
  const mx::Dtype dtype = noise.dtype();
  const int steps = config_.num_steps;  // 1
  const float step_size = 1.0f / steps;
  const mx::array time_embed = ScalarEmbed("head.t_embed", 0.0f, dtype);
  const mx::array interval =
      ScalarEmbed("head.dt_embed", config_.interval_embed, dtype);
  const mx::array step = ScalarEmbed("head.h_embed", step_size, dtype);
  const mx::array omega =
      ScalarEmbed("head.omega_embed", 1.0f - 1.0f / guidance, dtype);

  const mx::array cond = checkpoint_.Linear(hidden, "head.cond_proj");
  mx::array z = noise;
  for (int i = 0; i < steps; ++i) {
    mx::array c = mx::add(cond, time_embed);
    c = mx::add(c, interval);
    c = mx::add(c, step);
    c = mx::add(c, omega);
    z = mx::add(z, mx::multiply(omni_common::Scalar(step_size, z.dtype()),
                                Velocity(z, c)));
  }
  return z;
}

} // namespace haru
