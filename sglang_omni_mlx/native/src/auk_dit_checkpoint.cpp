// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Tencent. All rights reserved.
#include "auk_dit.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace auk {
namespace mx = mlx::core;
using qwen3_asr::Checkpoint;

namespace {

qwen3_asr::WeightMap LoadWeights(const std::filesystem::path &directory,
                                 mx::Dtype dtype) {
  auto weights = qwen3_asr::LoadSafetensors(directory);
  for (auto &[name, weight] : weights) {
    const bool preserve_float = name == "layer_weights" ||
                                name == "layer_scale" ||
                                name == "transformer.rotary_embed.inv_freq";
    weight = mx::astype(weight, preserve_float ? mx::float32 : dtype);
  }
  return weights;
}

void RequireWeight(const Checkpoint &checkpoint, const std::string &name,
                   const mx::Shape &shape) {
  if (checkpoint.Weight(name).shape() != shape) {
    throw std::invalid_argument("AuK checkpoint has an invalid shape for " +
                                name);
  } else {
  }
}

void RequireLinear(const Checkpoint &checkpoint, const std::string &prefix,
                   int output_channels, int input_channels,
                   bool has_bias = true) {
  RequireWeight(checkpoint, prefix + ".weight",
                {output_channels, input_channels});
  if (checkpoint.Has(prefix + ".scales") ||
      checkpoint.Has(prefix + ".biases")) {
    throw std::invalid_argument("AuK quantized weights are unsupported: " +
                                prefix);
  } else if (has_bias) {
    RequireWeight(checkpoint, prefix + ".bias", {output_channels});
  } else if (checkpoint.Has(prefix + ".bias")) {
    throw std::invalid_argument("AuK bias-free layer has an unexpected bias: " +
                                prefix);
  } else {
  }
}

void RequireBlock(const Checkpoint &checkpoint, const std::string &prefix,
                  int hidden_channels, int head_channels,
                  int feed_forward_channels, bool joint) {
  for (const std::string &stream : joint ? std::vector<std::string>{"_x", "_c"}
                                         : std::vector<std::string>{""}) {
    RequireLinear(checkpoint, prefix + ".attn_norm" + stream + ".linear",
                  6 * hidden_channels, hidden_channels);
    RequireLinear(checkpoint, prefix + ".ff" + stream + ".linear_in",
                  2 * feed_forward_channels, hidden_channels, false);
    RequireLinear(checkpoint, prefix + ".ff" + stream + ".linear_out",
                  hidden_channels, feed_forward_channels, false);
  }
  for (const std::string &stream : joint ? std::vector<std::string>{"", "_c"}
                                         : std::vector<std::string>{""}) {
    RequireLinear(checkpoint, prefix + ".attn.to_qkv" + stream,
                  3 * hidden_channels, hidden_channels);
    RequireLinear(checkpoint,
                  prefix +
                      (stream.empty() ? ".attn.to_out.0" : ".attn.to_out_c"),
                  hidden_channels, hidden_channels);
    const std::string norm_prefix =
        prefix + (stream.empty() ? ".attn." : ".attn.c_");
    RequireWeight(checkpoint, norm_prefix + "q_norm.weight", {head_channels});
    RequireWeight(checkpoint, norm_prefix + "k_norm.weight", {head_channels});
  }
}

} // namespace

DiT::DiT(const std::filesystem::path &model_directory, mx::Dtype dtype)
    : config_(qwen3_asr::ReadJson(model_directory / "config.json")),
      checkpoint_(LoadWeights(model_directory, dtype), config_), dtype_(dtype),
      head_count_(config_.at("arch").at("heads").get<int>()),
      joint_layer_count_(config_.at("arch").at("num_layers").get<int>()),
      single_layer_count_(
          config_.at("arch").at("num_single_layers").get<int>()),
      attention_mask_enabled_(
          config_.at("arch").at("attn_mask_enabled").get<bool>()) {
  ValidateWeights();
}

void DiT::ValidateWeights() const {
  const auto &architecture = config_.at("arch");
  const int hidden_channels = architecture.at("dim").get<int>();
  const int head_channels = architecture.value("dim_head", 64);
  const int text_channels = architecture.at("text_hidden_dim").get<int>();
  const int latent_channels = config_.at("latent_dim").get<int>();
  const double feed_forward_channels =
      hidden_channels * architecture.at("ff_mult").get<double>();
  if ((dtype_ != mx::float32 && dtype_ != mx::bfloat16) || head_count_ <= 0 ||
      hidden_channels <= 0 ||
      hidden_channels > std::numeric_limits<int>::max() / 6 ||
      hidden_channels % head_count_ != 0 || head_channels <= 0 ||
      head_channels % 2 != 0 ||
      hidden_channels / head_count_ != head_channels || text_channels <= 0 ||
      latent_channels <= 0 || joint_layer_count_ < 0 ||
      single_layer_count_ < 0 ||
      (joint_layer_count_ == 0 && single_layer_count_ == 0) ||
      !std::isfinite(feed_forward_channels) || feed_forward_channels < 1 ||
      feed_forward_channels > std::numeric_limits<int>::max() / 2 ||
      config_.contains("quantization") ||
      config_.contains("quantization_config")) {
    throw std::invalid_argument(
        "unsupported AuK DiT architecture or precision");
  } else {
  }
  const auto &layer_weights = checkpoint_.Weight("layer_weights");
  if (layer_weights.ndim() != 1 || layer_weights.size() == 0) {
    throw std::invalid_argument(
        "AuK layer fusion requires a nonempty weight vector");
  } else {
  }
  RequireWeight(checkpoint_, "layer_scale", {1});
  RequireWeight(checkpoint_, "transformer.rotary_embed.inv_freq",
                {head_channels / 2});
  RequireLinear(checkpoint_, "transformer.txt_proj", hidden_channels,
                text_channels);
  RequireWeight(checkpoint_, "transformer.txt_norm.weight", {hidden_channels});
  RequireLinear(checkpoint_, "transformer.audio_embed.linear", hidden_channels,
                latent_channels);
  for (int layer : {0, 2}) {
    const std::string prefix =
        "transformer.audio_embed.conv_pos_embed.conv1d." +
        std::to_string(layer);
    const auto &weight = checkpoint_.Weight(prefix + ".weight");
    if (weight.ndim() != 3 || weight.shape(0) != hidden_channels ||
        weight.shape(1) <= 0 || weight.shape(1) % 2 != 1 ||
        weight.shape(2) <= 0 || hidden_channels % weight.shape(2) != 0) {
      throw std::invalid_argument(
          "AuK positional convolution has incompatible geometry: " + prefix);
    } else {
    }
    RequireWeight(checkpoint_, prefix + ".bias", {hidden_channels});
  }
  const auto &time_weight =
      checkpoint_.Weight("transformer.time_embed.time_mlp.0.weight");
  if (time_weight.ndim() != 2 || time_weight.shape(0) != hidden_channels ||
      time_weight.shape(1) < 4 || time_weight.shape(1) % 2 != 0) {
    throw std::invalid_argument(
        "AuK timestep embedding has incompatible geometry");
  } else {
  }
  RequireLinear(checkpoint_, "transformer.time_embed.time_mlp.0",
                hidden_channels, time_weight.shape(1));
  RequireLinear(checkpoint_, "transformer.time_embed.time_mlp.2",
                hidden_channels, hidden_channels);
  for (int layer = 0; layer < joint_layer_count_; ++layer) {
    RequireBlock(checkpoint_,
                 "transformer.transformer_blocks." + std::to_string(layer),
                 hidden_channels, head_channels,
                 static_cast<int>(feed_forward_channels), true);
  }
  for (int layer = 0; layer < single_layer_count_; ++layer) {
    RequireBlock(checkpoint_,
                 "transformer.single_transformer_blocks." +
                     std::to_string(layer),
                 hidden_channels, head_channels,
                 static_cast<int>(feed_forward_channels), false);
  }
  RequireLinear(checkpoint_, "transformer.norm_out.linear", 2 * hidden_channels,
                hidden_channels);
  RequireLinear(checkpoint_, "transformer.proj_out", latent_channels,
                hidden_channels);
}

void DiT::ValidateInputDimensions(int text_layers, int text_hidden_channels,
                                  int latent_channels) const {
  if (text_layers != checkpoint_.Weight("layer_weights").shape(0) ||
      text_hidden_channels !=
          checkpoint_.Weight("transformer.txt_proj.weight").shape(1) ||
      latent_channels !=
          checkpoint_.Weight("transformer.proj_out.weight").shape(0)) {
    throw std::invalid_argument(
        "AuK conditioner, DiT and VAE dimensions do not match");
  } else {
  }
}

} // namespace auk
