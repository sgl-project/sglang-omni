// SPDX-License-Identifier: MIT AND Apache-2.0
// Copyright (C) 2026 Tencent. All rights reserved.
// Copyright (c) 2022 NVIDIA CORPORATION.
// AuK audio VAE; see LICENSE for the MIT permission notice.
// Alias-free activation uses the alias-free-torch formulation (Apache-2.0).
#include "auk/auk_vae.h"

#include <algorithm>
#include <stdexcept>

namespace auk {

namespace mx = mlx::core;

namespace {

constexpr int kActivationUpsampleRatio = 2;
constexpr int kActivationFilterSize = 12;
constexpr int kEncoderResidualStacks = 6;
// Note (XXXXRT666): the VAE runs in float16, Snake inside in float32; the
// released float32 weights decode to audio that transcribes the same.
constexpr mx::Dtype kDtype = mx::float16;

// Released weights with weight normalization folded and Torch convolution
// layouts moved to MLX's, in kDtype; the training-only flow is dropped. Each
// activation filter is also split into its even and odd taps, flipped for
// upsampling, for the two-phase convolutions.
omni_common::WeightMap
ConvertVAEWeights(const omni_common::WeightMap &stored,
                  const std::vector<int> &upsample_rates) {
  omni_common::WeightMap weights;
  for (const auto &[stored_name, stored_weight] : stored) {
    if (stored_name.starts_with("flow.") ||
        stored_name.ends_with(".weight_g")) {
      continue;
    } else {
    }
    std::string name = stored_name;
    mx::array weight = mx::astype(stored_weight, mx::float32);
    if (name.ends_with(".weight_v")) {
      name.resize(name.size() - 2);
      const mx::array magnitude =
          mx::astype(stored.at(name + "_g"), mx::float32);
      weight = weight *
               (magnitude / mx::sqrt(mx::sum(weight * weight, {1, 2}, true)));
    } else {
    }
    if (name.ends_with(".weight") && weight.ndim() == 3 &&
        name.starts_with("ups.")) {
      // ups.<stage>.0.weight, a transposed convolution, as its stride-1
      // phases.
      const int stage = std::stoi(name.substr(4));
      weight = omni_common::TransposedConvolutionPhases(
          mx::transpose(weight, {1, 2, 0}), upsample_rates.at(stage));
    } else if (name.ends_with(".weight") && weight.ndim() == 3) {
      weight = mx::transpose(weight, {0, 2, 1});
    } else if (name.ends_with(".filter")) {
      weight = mx::transpose(weight, {0, 2, 1});
      const int taps = weight.shape(1);
      const bool flips = name.ends_with(".upsample.filter");
      for (const int phase : {0, 1}) {
        mx::array phase_taps = mx::slice(weight, {0, phase, 0}, {1, taps, 1},
                                         {1, kActivationUpsampleRatio, 1});
        if (flips) {
          phase_taps = mx::take(
              phase_taps,
              mx::arange(phase_taps.shape(1) - 1, -1, -1, mx::int32), 1);
        } else {
        }
        weights.emplace(name + (phase == 0 ? ".even" : ".odd"),
                        mx::astype(phase_taps, kDtype));
      }
    } else {
    }
    weights.emplace(std::move(name), mx::astype(weight, kDtype));
  }
  return weights;
}

mx::array LeakyRelu(const mx::array &hidden_states, float negative_slope) {
  return mx::maximum(
      hidden_states,
      mx::multiply(hidden_states,
                   omni_common::Scalar(negative_slope, hidden_states)));
}

} // namespace

VAE::VAE(const omni_common::WeightMap &weights, const nlohmann::json &config)
    : checkpoint_(
          ConvertVAEWeights(
              weights, config.at("upsample_rates").get<std::vector<int>>()),
          nlohmann::json::object()) {
  latent_dimension_ = config.at("latent_dim").get<int>();
  causal_ = config.at("causal").get<bool>();
  activation_causal_ = config.at("act_causal").get<bool>();
  snake_logscale_ = config.at("snake_logscale").get<bool>();
  downsample_rates_ = config.at("downsample_rates").get<std::vector<int>>();
  upsample_rates_ = config.at("upsample_rates").get<std::vector<int>>();
  const auto encoder_channels =
      config.at("downsample_channels").get<std::vector<int>>();
  upsample_kernels_ =
      config.at("upsample_kernel_sizes").get<std::vector<int>>();
  const auto &upsample_kernels = upsample_kernels_;
  const auto residual_kernels =
      config.at("resblock_kernel_sizes").get<std::vector<int>>();
  residual_dilations_ =
      config.at("resblock_dilation_sizes").get<std::vector<std::vector<int>>>();
  if (!config.at("use_vae").get<bool>() || latent_dimension_ <= 0 ||
      downsample_rates_.empty() || upsample_rates_.empty() ||
      encoder_channels.size() != downsample_rates_.size() + 1 ||
      upsample_kernels.size() != upsample_rates_.size() ||
      residual_kernels.empty() ||
      residual_kernels.size() != residual_dilations_.size()) {
    throw std::invalid_argument("invalid AuK VAE configuration");
  } else {
  }
  int decoder_hop_size = 1;
  for (int rate : downsample_rates_) {
    if (rate <= 0) {
      throw std::invalid_argument("VAE downsample rates must be positive");
    } else {
      hop_size_ *= rate;
    }
  }
  int decoder_channels = config.at("upsample_initial_channel").get<int>();
  for (size_t i = 0; i < upsample_rates_.size(); ++i) {
    const int rate = upsample_rates_[i];
    const int kernel = upsample_kernels[i];
    if (rate <= 0 || kernel < rate || decoder_channels < 2 ||
        decoder_channels % 2 != 0 || (causal_ && kernel != 2 * rate)) {
      throw std::invalid_argument("invalid VAE upsampling configuration");
    } else {
      decoder_hop_size *= rate;
      decoder_channels /= 2;
    }
  }
  for (size_t i = 0; i < residual_kernels.size(); ++i) {
    const auto &dilations = residual_dilations_[i];
    if (residual_kernels[i] <= 0 || residual_kernels[i] % 2 == 0 ||
        dilations.empty() ||
        std::any_of(dilations.begin(), dilations.end(),
                    [](int dilation) { return dilation <= 0; })) {
      throw std::invalid_argument("invalid VAE residual configuration");
    } else {
    }
  }
  if (hop_size_ != decoder_hop_size ||
      std::any_of(encoder_channels.begin(), encoder_channels.end(),
                  [](int channels) { return channels <= 0; })) {
    throw std::invalid_argument("inconsistent VAE encoder/decoder dimensions");
  } else {
  }
  checkpoint_.RequireShape("global_mean", {latent_dimension_});
  checkpoint_.RequireShape("global_log_std", {latent_dimension_});
  const auto require_convolution = [&](const std::string &prefix,
                                       int input_channels, int output_channels,
                                       int kernel_size, bool bias = true) {
    checkpoint_.RequireShape(prefix + ".weight",
                             {output_channels, kernel_size, input_channels});
    if (bias) {
      checkpoint_.RequireShape(prefix + ".bias", {output_channels});
    } else {
    }
  };
  const auto require_activation = [&](const std::string &prefix, int channels) {
    checkpoint_.RequireShape(prefix + ".act.alpha", {channels});
    checkpoint_.RequireShape(prefix + ".act.beta", {channels});
    checkpoint_.RequireShape(prefix + ".upsample.filter",
                             {1, kActivationFilterSize, 1});
    checkpoint_.RequireShape(prefix + ".downsample.lowpass.filter",
                             {1, kActivationFilterSize, 1});
  };
  const std::string encoder = "audio_encoder.generator.";
  require_convolution(encoder + "0.layer", 1, encoder_channels[0], 3);
  for (size_t i = 0; i < downsample_rates_.size(); ++i) {
    const int channels = encoder_channels[i + 1];
    require_convolution(encoder + std::to_string(2 + 3 * i) + ".layer",
                        encoder_channels[i], channels,
                        2 * downsample_rates_[i]);
    for (int j = 0; j < kEncoderResidualStacks; ++j) {
      const std::string prefix =
          encoder + std::to_string(3 + 3 * i) + ".layers." + std::to_string(j);
      require_convolution(prefix + ".1", channels, channels, 3);
      require_convolution(prefix + ".3", channels, channels, 3);
    }
  }
  require_convolution(
      encoder + std::to_string(2 + 3 * downsample_rates_.size()) + ".layer",
      encoder_channels.back(), 2 * latent_dimension_, 3);
  decoder_channels = config.at("upsample_initial_channel").get<int>();
  require_convolution("conv_pre", latent_dimension_, decoder_channels, 7);
  for (size_t i = 0; i < upsample_rates_.size(); ++i) {
    // A transposed convolution, kept as its stride-1 phases.
    const std::string upsample = "ups." + std::to_string(i) + ".0";
    const int rate = upsample_rates_[i];
    checkpoint_.RequireShape(upsample + ".weight",
                             {rate * (decoder_channels / 2),
                              (upsample_kernels[i] + rate - 1) / rate,
                              decoder_channels});
    checkpoint_.RequireShape(upsample + ".bias", {decoder_channels / 2});
    decoder_channels /= 2;
    for (size_t j = 0; j < residual_kernels.size(); ++j) {
      const std::string prefix =
          "resblocks." + std::to_string(i * residual_kernels.size() + j);
      for (size_t k = 0; k < residual_dilations_[j].size(); ++k) {
        require_convolution(prefix + ".convs1." + std::to_string(k),
                            decoder_channels, decoder_channels,
                            residual_kernels[j]);
        require_convolution(prefix + ".convs2." + std::to_string(k),
                            decoder_channels, decoder_channels,
                            residual_kernels[j]);
        require_activation(prefix + ".activations." + std::to_string(2 * k),
                           decoder_channels);
        require_activation(prefix + ".activations." + std::to_string(2 * k + 1),
                           decoder_channels);
      }
    }
  }
  require_activation("activation_post", decoder_channels);
  require_convolution("conv_post", decoder_channels, 1, 7, false);
}

mx::array VAE::Convolution(const mx::array &hidden_states,
                           const std::string &prefix, int stride, int dilation,
                           bool causal) const {
  const auto &weight = checkpoint_.Weight(prefix + ".weight");
  const int extent = dilation * (weight.shape(1) - 1);
  const mx::array padded =
      causal ? mx::pad(hidden_states, {{0, 0}, {extent, 0}, {0, 0}})
             : hidden_states;
  const mx::array convolved =
      mx::conv1d(padded, weight, stride, causal ? 0 : extent / 2, dilation);
  return checkpoint_.Has(prefix + ".bias")
             ? convolved + checkpoint_.Weight(prefix + ".bias")
             : convolved;
}

mx::array VAE::Activation(const mx::array &hidden_states,
                          const std::string &prefix) const {
  const int batch = hidden_states.shape(0);
  const int channels = hidden_states.shape(2);
  const std::string upsample = prefix + ".upsample.filter";
  const int kernel_size = checkpoint_.Weight(upsample).shape(1);
  const int phase_taps = kernel_size / kActivationUpsampleRatio;
  const int padding = phase_taps - 1;
  const int left_trim = padding * kActivationUpsampleRatio +
                        (kernel_size - kActivationUpsampleRatio) / 2;
  const int right_trim = padding * kActivationUpsampleRatio +
                         (kernel_size - kActivationUpsampleRatio + 1) / 2;
  // Note (XXXXRT666): the 2x upsampling and downsampling each run as two
  // phases of stride-1 depthwise convolutions, which MLX's depthwise kernel
  // takes, instead of a transposed and a strided convolution; the sums are the
  // same.
  const auto depthwise = [&](const mx::array &input, const std::string &taps) {
    return mx::conv1d(
        input,
        mx::broadcast_to(checkpoint_.Weight(taps), {channels, phase_taps, 1}),
        1, 0, 1, channels);
  };
  const mx::array edge_padded =
      mx::pad(hidden_states, {{0, 0}, {padding, padding}, {0, 0}},
              mx::array(0.0f), "edge");
  const mx::array zero_padded =
      mx::pad(edge_padded, {{0, 0}, {phase_taps - 1, phase_taps - 1}, {0, 0}});
  const mx::array even = depthwise(zero_padded, upsample + ".even");
  const mx::array odd = depthwise(zero_padded, upsample + ".odd");
  mx::array activated = mx::multiply(
      mx::reshape(mx::stack({even, odd}, 2),
                  {batch, 2 * even.shape(1), channels}),
      omni_common::Scalar(static_cast<float>(kActivationUpsampleRatio), even));
  activated = mx::slice(activated, {0, left_trim, 0},
                        {batch, activated.shape(1) - right_trim, channels});
  activated = omni_common::Snake(
      activated, checkpoint_.Weight(prefix + ".act.alpha"),
      checkpoint_.Weight(prefix + ".act.beta"), snake_logscale_);
  const std::string downsample = prefix + ".downsample.lowpass.filter";
  const int pad_left =
      activation_causal_ ? kernel_size - 1 : (kernel_size - 1) / 2;
  const int pad_right = activation_causal_ ? 0 : kernel_size / 2;
  const mx::array padded =
      mx::pad(activated, {{0, 0}, {pad_left, pad_right}, {0, 0}},
              mx::array(0.0f), "edge");
  const int padded_length = padded.shape(1);
  const int output_length = padded_length / 2 - phase_taps + 1;
  const auto samples = [&](int phase) {
    return mx::slice(padded, {0, phase, 0}, {batch, padded_length, channels},
                     {1, kActivationUpsampleRatio, 1});
  };
  const auto phase_output = [&](int phase, const std::string &taps) {
    return mx::slice(depthwise(samples(phase), downsample + taps), {0, 0, 0},
                     {batch, output_length, channels});
  };
  return mx::add(phase_output(0, ".even"), phase_output(1, ".odd"));
}

mx::array VAE::ResidualBlock(const mx::array &hidden_states,
                             const std::string &prefix,
                             const std::vector<int> &dilations) const {
  mx::array residual_states = hidden_states;
  for (size_t i = 0; i < dilations.size(); ++i) {
    const auto first = Convolution(
        Activation(residual_states,
                   prefix + ".activations." + std::to_string(2 * i)),
        prefix + ".convs1." + std::to_string(i), 1, dilations[i], causal_);
    residual_states =
        residual_states +
        Convolution(Activation(first, prefix + ".activations." +
                                          std::to_string(2 * i + 1)),
                    prefix + ".convs2." + std::to_string(i), 1, 1, causal_);
  }
  return residual_states;
}

PosteriorStats VAE::Posterior(const mx::array &waveform) const {
  if (waveform.ndim() != 1 ||
      waveform.size() < static_cast<size_t>(hop_size_)) {
    throw std::invalid_argument(
        "VAE waveform must be one-dimensional and at least one hop long");
  } else {
  }
  mx::array hidden_states =
      mx::reshape(mx::astype(waveform, kDtype), {1, waveform.shape(0), 1});
  const std::string encoder = "audio_encoder.generator.";
  hidden_states =
      LeakyRelu(Convolution(hidden_states, encoder + "0.layer"), 0.2f);
  for (size_t i = 0; i < downsample_rates_.size(); ++i) {
    hidden_states = Convolution(hidden_states,
                                encoder + std::to_string(2 + 3 * i) + ".layer",
                                downsample_rates_[i]);
    const std::string residual =
        encoder + std::to_string(3 + 3 * i) + ".layers.";
    for (int j = 0; j < kEncoderResidualStacks; ++j) {
      const auto first =
          Convolution(LeakyRelu(hidden_states, 0.01f),
                      residual + std::to_string(j) + ".1", 1, 1 << j);
      hidden_states =
          hidden_states + Convolution(LeakyRelu(first, 0.01f),
                                      residual + std::to_string(j) + ".3");
    }
    hidden_states = LeakyRelu(hidden_states, 0.2f);
  }
  const mx::array statistics = mx::squeeze(
      Convolution(hidden_states,
                  encoder + std::to_string(2 + 3 * downsample_rates_.size()) +
                      ".layer"),
      0);
  const auto parts = mx::split(mx::astype(statistics, mx::float32), 2, -1);
  return {parts[0], parts[1]};
}

mx::array VAE::Encode(const PosteriorStats &posterior,
                      const mx::array &posterior_noise) const {
  if (posterior_noise.shape() != posterior.mean.shape()) {
    throw std::invalid_argument(
        "VAE posterior noise must have shape [frames, channels]");
  } else {
  }
  const mx::array latents =
      posterior.mean + mx::astype(posterior_noise, mx::float32) *
                           mx::exp(posterior.log_standard_deviation);
  return (latents -
          mx::astype(checkpoint_.Weight("global_mean"), mx::float32)) /
         mx::sqrt(
             mx::astype(checkpoint_.Weight("global_log_std"), mx::float32));
}

mx::array VAE::Decode(const mx::array &latents) const {
  if (latents.ndim() != 2 || latents.shape(0) == 0 ||
      latents.shape(1) != latent_dimension_) {
    throw std::invalid_argument(
        "VAE latents must have shape [frames, latent_dim]");
  } else {
  }
  mx::array hidden_states = mx::expand_dims(
      mx::astype(
          mx::astype(latents, mx::float32) *
                  mx::sqrt(mx::astype(checkpoint_.Weight("global_log_std"),
                                      mx::float32)) +
              mx::astype(checkpoint_.Weight("global_mean"), mx::float32),
          kDtype),
      0);
  hidden_states = Convolution(hidden_states, "conv_pre");
  for (size_t i = 0; i < upsample_rates_.size(); ++i) {
    const std::string prefix = "ups." + std::to_string(i) + ".0";
    const int rate = upsample_rates_[i];
    const int kernel = upsample_kernels_[i];
    hidden_states = omni_common::ConvolveTransposed(
                        hidden_states, checkpoint_.Weight(prefix + ".weight"),
                        rate, kernel, causal_ ? 0 : (kernel - rate) / 2) +
                    checkpoint_.Weight(prefix + ".bias");
    if (causal_) {
      hidden_states =
          mx::slice(hidden_states, {0, 0, 0},
                    {1, hidden_states.shape(1) - rate, hidden_states.shape(2)});
    } else {
    }
    const size_t first = i * residual_dilations_.size();
    mx::array combined =
        ResidualBlock(hidden_states, "resblocks." + std::to_string(first),
                      residual_dilations_[0]);
    for (size_t j = 1; j < residual_dilations_.size(); ++j) {
      combined =
          combined + ResidualBlock(hidden_states,
                                   "resblocks." + std::to_string(first + j),
                                   residual_dilations_[j]);
    }
    hidden_states = combined * omni_common::Scalar(
                                   1.0f / residual_dilations_.size(), combined);
  }
  return mx::reshape(
      mx::clip(
          mx::astype(Convolution(Activation(hidden_states, "activation_post"),
                                 "conv_post", 1, 1, causal_),
                     mx::float32),
          mx::array(-1.0f), mx::array(1.0f)),
      {-1});
}

} // namespace auk
