// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Tencent. All rights reserved.
#include "auk/auk_dit.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace auk {
namespace mx = mlx::core;
using omni_common::Checkpoint;

namespace {

omni_common::WeightMap ConvertWeights(omni_common::WeightMap weights,
                                      mx::Dtype dtype) {
  for (auto &[name, weight] : weights) {
    const bool preserve_float = name == "layer_weights" ||
                                name == "layer_scale" ||
                                name == "transformer.rotary_embed.inv_freq";
    if (name.ends_with("conv_pos_embed.conv1d.0.weight") ||
        name.ends_with("conv_pos_embed.conv1d.2.weight")) {
      weight = mx::transpose(weight, {0, 2, 1});
    } else {
    }
    weight = mx::astype(weight, preserve_float ? mx::float32 : dtype);
  }
  return weights;
}

void RequireLinear(const Checkpoint &checkpoint, const std::string &prefix,
                   int output_channels, int input_channels,
                   bool has_bias = true) {
  checkpoint.RequireShape(prefix + ".weight",
                          {output_channels, input_channels});
  if (has_bias) {
    checkpoint.RequireShape(prefix + ".bias", {output_channels});
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
    checkpoint.RequireShape(norm_prefix + "q_norm.weight", {head_channels});
    checkpoint.RequireShape(norm_prefix + "k_norm.weight", {head_channels});
  }
}

} // namespace

DiT::Config DiT::Config::Read(const nlohmann::json &architecture,
                              int latent_channels) {
  const int hidden_channels = architecture.at("dim").get<int>();
  return {
      .hidden_channels = hidden_channels,
      .head_count = architecture.at("heads").get<int>(),
      .head_channels = architecture.value("dim_head", 64),
      .text_channels = architecture.at("text_hidden_dim").get<int>(),
      .latent_channels = latent_channels,
      .feed_forward_channels = static_cast<int>(
          hidden_channels * architecture.at("ff_mult").get<double>()),
      .joint_layer_count = architecture.at("num_layers").get<int>(),
      .single_layer_count = architecture.at("num_single_layers").get<int>(),
      .attention_mask_enabled =
          architecture.at("attn_mask_enabled").get<bool>(),
  };
}

DiT::DiT(omni_common::WeightMap weights, const nlohmann::json &architecture,
         int latent_channels, mx::Dtype dtype)
    : config_(Config::Read(architecture, latent_channels)),
      checkpoint_(ConvertWeights(std::move(weights), dtype),
                  nlohmann::json::object()),
      rotary_periods_(mx::reciprocal(
          checkpoint_.Weight("transformer.rotary_embed.inv_freq"))),
      dtype_(dtype) {
  ValidateWeights();
  mx::eval(rotary_periods_);
}

void DiT::ValidateWeights() const {
  if (config_.head_count <= 0 ||
      config_.hidden_channels % config_.head_count != 0 ||
      config_.hidden_channels / config_.head_count != config_.head_channels ||
      config_.head_channels % 2 != 0) {
    throw std::invalid_argument("unsupported AuK DiT architecture");
  } else {
  }
  checkpoint_.RequireShape("layer_scale", {1});
  checkpoint_.RequireShape("transformer.rotary_embed.inv_freq",
                           {config_.head_channels / 2});
  RequireLinear(checkpoint_, "transformer.txt_proj", config_.hidden_channels,
                config_.text_channels);
  checkpoint_.RequireShape("transformer.txt_norm.weight",
                           {config_.hidden_channels});
  RequireLinear(checkpoint_, "transformer.audio_embed.linear",
                config_.hidden_channels, config_.latent_channels);
  for (int layer : {0, 2}) {
    const std::string prefix =
        "transformer.audio_embed.conv_pos_embed.conv1d." +
        std::to_string(layer);
    const auto &weight = checkpoint_.Weight(prefix + ".weight");
    if (weight.ndim() != 3 || weight.shape(0) != config_.hidden_channels ||
        weight.shape(1) <= 0 || weight.shape(1) % 2 != 1 ||
        weight.shape(2) <= 0 ||
        config_.hidden_channels % weight.shape(2) != 0) {
      throw std::invalid_argument(
          "AuK positional convolution has incompatible geometry: " + prefix);
    } else {
    }
    checkpoint_.RequireShape(prefix + ".bias", {config_.hidden_channels});
  }
  const auto &time_weight =
      checkpoint_.Weight("transformer.time_embed.time_mlp.0.weight");
  if (time_weight.ndim() != 2 ||
      time_weight.shape(0) != config_.hidden_channels ||
      time_weight.shape(1) < 4 || time_weight.shape(1) % 2 != 0) {
    throw std::invalid_argument(
        "AuK timestep embedding has incompatible geometry");
  } else {
  }
  RequireLinear(checkpoint_, "transformer.time_embed.time_mlp.0",
                config_.hidden_channels, time_weight.shape(1));
  RequireLinear(checkpoint_, "transformer.time_embed.time_mlp.2",
                config_.hidden_channels, config_.hidden_channels);
  for (int layer = 0; layer < config_.joint_layer_count; ++layer) {
    RequireBlock(checkpoint_,
                 "transformer.transformer_blocks." + std::to_string(layer),
                 config_.hidden_channels, config_.head_channels,
                 config_.feed_forward_channels, true);
  }
  for (int layer = 0; layer < config_.single_layer_count; ++layer) {
    RequireBlock(checkpoint_,
                 "transformer.single_transformer_blocks." +
                     std::to_string(layer),
                 config_.hidden_channels, config_.head_channels,
                 config_.feed_forward_channels, false);
  }
  RequireLinear(checkpoint_, "transformer.norm_out.linear",
                2 * config_.hidden_channels, config_.hidden_channels);
  RequireLinear(checkpoint_, "transformer.proj_out", config_.latent_channels,
                config_.hidden_channels);
}

void DiT::ValidateInputDimensions(int text_layers,
                                  int text_hidden_channels) const {
  if (checkpoint_.Weight("layer_weights").shape() != mx::Shape{text_layers} ||
      checkpoint_.Weight("transformer.txt_proj.weight").shape(1) !=
          text_hidden_channels) {
    throw std::invalid_argument(
        "AuK conditioner and DiT dimensions do not match");
  } else {
  }
}

} // namespace auk
