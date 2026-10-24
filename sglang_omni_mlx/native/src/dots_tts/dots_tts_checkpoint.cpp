// SPDX-License-Identifier: Apache-2.0
#include "dots_tts/dots_tts_checkpoint.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include "common/npz.h"

namespace dots_tts {

namespace mx = mlx::core;

namespace {

constexpr float kBatchNormEpsilon = 1e-5f;

std::pair<mx::array, mx::array> FoldBatchNorm(const qwen3_asr::WeightMap &w,
                                              const std::string &prefix,
                                              mx::Dtype dtype) {
  const mx::array running_mean =
      mx::astype(w.at(prefix + ".running_mean"), mx::float32);
  const mx::array running_var =
      mx::astype(w.at(prefix + ".running_var"), mx::float32);
  const mx::array inv = mx::rsqrt(
      mx::add(running_var, mx::array(kBatchNormEpsilon, mx::float32)));
  const bool affine = w.count(prefix + ".weight") > 0;
  const mx::array gamma =
      affine ? mx::astype(w.at(prefix + ".weight"), mx::float32)
             : mx::ones_like(running_mean);
  const mx::array beta = affine
                             ? mx::astype(w.at(prefix + ".bias"), mx::float32)
                             : mx::zeros_like(running_mean);
  const mx::array scale = mx::astype(mx::multiply(gamma, inv), dtype);
  const mx::array shift = mx::astype(
      mx::subtract(beta, mx::multiply(running_mean, mx::multiply(gamma, inv))),
      dtype);
  return {scale, shift};
}

qwen3_asr::WeightMap LoadWeights(const std::filesystem::path &model_directory) {
  const auto load_file = [&](const std::string &name) {
    const std::filesystem::path path = model_directory / name;
    if (!std::filesystem::exists(path)) {
      throw std::runtime_error("converted dots.tts checkpoint is missing " +
                               path.string());
    }
    return mx::load_safetensors(path.string());
  };

  qwen3_asr::WeightMap weights;
  {
    const auto [tensors, metadata] = load_file("core.safetensors");
    (void)metadata;
    for (auto &[name, array] : tensors)
      weights.insert_or_assign(name, array);
  }

  {
    const auto [tensors, metadata] = load_file("vocoder.safetensors");
    (void)metadata;
    for (auto &[name, array] : tensors)
      weights.insert_or_assign(name, array);
  }

  auto [speaker_raw, speaker_metadata] = load_file("speaker.safetensors");
  (void)speaker_metadata;
  qwen3_asr::WeightMap speaker;
  for (auto &[name, array] : speaker_raw) {
    if (name.rfind("model.", 0) != 0) {
      continue;
    }
    speaker.insert_or_assign(name.substr(std::string("model.").size()), array);
  }
  if (speaker.count("resample.kernel") > 0) {
    speaker.erase("resample.kernel");
  }

  {
    std::vector<std::string> bn_prefixes;
    for (const auto &[name, array] : speaker) {
      const std::string suffix = ".running_mean";
      if (name.size() > suffix.size() &&
          name.compare(name.size() - suffix.size(), suffix.size(), suffix) ==
              0) {
        bn_prefixes.push_back(name.substr(0, name.size() - suffix.size()));
      }
    }
    std::sort(bn_prefixes.begin(), bn_prefixes.end());
    bn_prefixes.erase(std::unique(bn_prefixes.begin(), bn_prefixes.end()),
                      bn_prefixes.end());
    for (const std::string &prefix : bn_prefixes) {
      const auto [scale, shift] = FoldBatchNorm(speaker, prefix, mx::float32);
      speaker.insert_or_assign(prefix + "_bn_scale", scale);
      speaker.insert_or_assign(prefix + "_bn_shift", shift);
    }
  }
  for (auto &[name, array] : speaker)
    weights.insert_or_assign(name, array);

  return weights;
}

nlohmann::json MergedConfig(const std::filesystem::path &model_directory) {
  nlohmann::json config = qwen3_asr::ReadJson(model_directory / "config.json");
  const nlohmann::json llm_config =
      qwen3_asr::ReadJson(model_directory / "llm_config.json");
  for (const auto &[key, value] : llm_config.items())
    config[key] = value;
  if (config.contains("quantization")) {
    nlohmann::json &quantization = config.at("quantization");
    quantization["group_size"] = quantization.value("group_size", 64);
    quantization["bits"] = quantization.value("bits", 4);
  }
  return config;
}

} // namespace

DotsCheckpoint::DotsCheckpoint(const std::filesystem::path &model_directory)
    : config_(MergedConfig(model_directory)),
      checkpoint_(LoadWeights(model_directory), config_) {
  const std::filesystem::path stats_path = model_directory / "latent_stats.npz";
  if (!std::filesystem::exists(stats_path)) {
    throw std::runtime_error("converted dots.tts checkpoint is missing " +
                             stats_path.string());
  }
  const auto stats = npz::Load(stats_path);
  const auto mean = stats.find("mean");
  const auto var = stats.find("var");
  if (mean == stats.end() || var == stats.end()) {
    throw std::runtime_error("latent_stats.npz is missing mean/var");
  }
  latent_mean_ = mx::astype(mean->second, mx::float32);
  latent_var_ = mx::astype(var->second, mx::float32);
}

mlx::core::array Conv1dWeight(const mlx::core::array &w,
                              mlx::core::Dtype dtype) {
  // torch [out, in, k] -> MLX [out, k, in].
  return mx::astype(mx::transpose(w, {0, 2, 1}), dtype);
}

mlx::core::array ConvTranspose1dWeight(const mlx::core::array &w,
                                       mlx::core::Dtype dtype) {
  // torch [in, out, k] -> MLX [out, k, in].
  return mx::astype(mx::transpose(w, {1, 2, 0}), dtype);
}

mlx::core::array Conv2dWeight(const mlx::core::array &w,
                              mlx::core::Dtype dtype) {
  // torch [out, in, kh, kw] -> MLX [out, kh, kw, in].
  return mx::astype(mx::transpose(w, {0, 2, 3, 1}), dtype);
}

mlx::core::array FilterWeight(const mlx::core::array &f, int channels) {
  // [1, 1, k] (fixed) or [C, 1, k] (per-channel) -> depthwise [C, k, 1] fp32.
  mx::array out = f;
  if (out.shape(0) == 1 && channels != 1) {
    out = mx::broadcast_to(out, {channels, 1, out.shape(2)});
  }
  out = mx::transpose(out, {0, 2, 1});
  return mx::astype(out, mx::float32);
}

} // namespace dots_tts
