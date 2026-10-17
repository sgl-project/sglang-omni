// SPDX-License-Identifier: MIT AND Apache-2.0
// Copyright (C) 2026 Tencent. All rights reserved.
// Copyright (c) 2022 NVIDIA CORPORATION.
// AuK audio VAE; see LICENSE for the MIT permission notice.
// Alias-free activation uses the alias-free-torch formulation (Apache-2.0).
#include "auk_vae.h"

#include <algorithm>
#include <stdexcept>

namespace auk {

namespace mx = mlx::core;

namespace {

constexpr int kActivationUpsampleRatio = 2;
constexpr int kActivationFilterSize = 12;
constexpr float kSnakeEpsilon = 1e-9f;
constexpr int kEncoderResidualStacks = 6;

qwen3_asr::WeightMap LoadVAEWeights(const std::filesystem::path &directory) {
  auto weights = qwen3_asr::LoadSafetensors(directory);
  for (auto &[name, weight] : weights) {
    weight = mx::astype(weight, mx::float32);
  }
  return weights;
}

mx::array LeakyRelu(const mx::array &hidden_states, float negative_slope) {
  return mx::maximum(hidden_states, hidden_states * negative_slope);
}

} // namespace

VAE::VAE(const std::filesystem::path &vae_directory)
    : checkpoint_(LoadVAEWeights(vae_directory), nlohmann::json::object()) {
  const auto config = qwen3_asr::ReadJson(vae_directory / "config.json");
  latent_dimension_ = config.at("latent_dim").get<int>();
  causal_ = config.at("causal").get<bool>();
  activation_causal_ = config.at("act_causal").get<bool>();
  snake_logscale_ = config.at("snake_logscale").get<bool>();
  downsample_rates_ = config.at("downsample_rates").get<std::vector<int>>();
  upsample_rates_ = config.at("upsample_rates").get<std::vector<int>>();
  const auto encoder_channels =
      config.at("downsample_channels").get<std::vector<int>>();
  const auto upsample_kernels =
      config.at("upsample_kernel_sizes").get<std::vector<int>>();
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
                  [](int channels) { return channels <= 0; }) ||
      checkpoint_.Weight("global_mean").shape() !=
          mx::Shape{latent_dimension_} ||
      checkpoint_.Weight("global_log_std").shape() !=
          mx::Shape{latent_dimension_}) {
    throw std::invalid_argument("inconsistent VAE encoder/decoder dimensions");
  } else {
  }
  const auto require_shape = [&](const std::string &name,
                                 const mx::Shape &shape) {
    if (checkpoint_.Weight(name).shape() != shape) {
      throw std::invalid_argument("invalid VAE tensor shape: " + name);
    } else {
    }
  };
  const auto require_convolution = [&](const std::string &prefix,
                                       int input_channels, int output_channels,
                                       int kernel_size, bool bias = true) {
    require_shape(prefix + ".weight",
                  {output_channels, kernel_size, input_channels});
    if (bias) {
      require_shape(prefix + ".bias", {output_channels});
    } else if (checkpoint_.Has(prefix + ".bias")) {
      throw std::invalid_argument("unexpected VAE bias: " + prefix);
    } else {
    }
  };
  const auto require_activation = [&](const std::string &prefix, int channels) {
    require_shape(prefix + ".act.alpha", {channels});
    require_shape(prefix + ".act.beta", {channels});
    require_shape(prefix + ".upsample.filter", {1, kActivationFilterSize, 1});
    require_shape(prefix + ".downsample.lowpass.filter",
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
    require_convolution("ups." + std::to_string(i) + ".0", decoder_channels,
                        decoder_channels / 2, upsample_kernels[i]);
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
  const int channels = hidden_states.shape(2);
  const auto &upsample_filter = checkpoint_.Weight(prefix + ".upsample.filter");
  const int kernel_size = upsample_filter.shape(1);
  const int padding = kernel_size / kActivationUpsampleRatio - 1;
  const int left_trim = padding * kActivationUpsampleRatio +
                        (kernel_size - kActivationUpsampleRatio) / 2;
  const int right_trim = padding * kActivationUpsampleRatio +
                         (kernel_size - kActivationUpsampleRatio + 1) / 2;
  mx::array activated =
      mx::conv_transpose1d(
          mx::pad(hidden_states, {{0, 0}, {padding, padding}, {0, 0}},
                  mx::array(0.0f), "edge"),
          mx::broadcast_to(upsample_filter, {channels, kernel_size, 1}),
          kActivationUpsampleRatio, 0, 1, 0, channels) *
      static_cast<float>(kActivationUpsampleRatio);
  activated = mx::slice(activated, {0, left_trim, 0},
                        {1, activated.shape(1) - right_trim, channels});
  const auto &stored_alpha = checkpoint_.Weight(prefix + ".act.alpha");
  const auto &stored_beta = checkpoint_.Weight(prefix + ".act.beta");
  const mx::array alpha =
      snake_logscale_ ? mx::exp(stored_alpha) : stored_alpha;
  const mx::array beta = snake_logscale_ ? mx::exp(stored_beta) : stored_beta;
  activated = activated +
              mx::square(mx::sin(activated * alpha)) / (beta + kSnakeEpsilon);
  const auto &downsample_filter =
      checkpoint_.Weight(prefix + ".downsample.lowpass.filter");
  const int downsample_kernel_size = downsample_filter.shape(1);
  const int pad_left = activation_causal_ ? downsample_kernel_size - 1
                                          : (downsample_kernel_size - 1) / 2;
  const int pad_right = activation_causal_ ? 0 : downsample_kernel_size / 2;
  return mx::conv1d(mx::pad(activated, {{0, 0}, {pad_left, pad_right}, {0, 0}},
                            mx::array(0.0f), "edge"),
                    mx::broadcast_to(downsample_filter,
                                     {channels, downsample_kernel_size, 1}),
                    kActivationUpsampleRatio, 0, 1, channels);
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
      mx::reshape(mx::astype(waveform, mx::float32), {1, waveform.shape(0), 1});
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
  const auto parts = mx::split(statistics, 2, -1);
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
  return (latents - checkpoint_.Weight("global_mean")) /
         mx::sqrt(checkpoint_.Weight("global_log_std"));
}

mx::array VAE::Decode(const mx::array &latents) const {
  if (latents.ndim() != 2 || latents.shape(0) == 0 ||
      latents.shape(1) != latent_dimension_) {
    throw std::invalid_argument(
        "VAE latents must have shape [frames, latent_dim]");
  } else {
  }
  mx::array hidden_states =
      mx::expand_dims(mx::astype(latents, mx::float32) *
                              mx::sqrt(checkpoint_.Weight("global_log_std")) +
                          checkpoint_.Weight("global_mean"),
                      0);
  hidden_states = Convolution(hidden_states, "conv_pre");
  for (size_t i = 0; i < upsample_rates_.size(); ++i) {
    const std::string prefix = "ups." + std::to_string(i) + ".0";
    const auto &weight = checkpoint_.Weight(prefix + ".weight");
    const int rate = upsample_rates_[i];
    hidden_states =
        mx::conv_transpose1d(hidden_states, weight, rate,
                             causal_ ? 0 : (weight.shape(1) - rate) / 2) +
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
    hidden_states = combined / static_cast<float>(residual_dilations_.size());
  }
  return mx::reshape(
      mx::clip(Convolution(Activation(hidden_states, "activation_post"),
                           "conv_post", 1, 1, causal_),
               mx::array(-1.0f), mx::array(1.0f)),
      {-1});
}

} // namespace auk
