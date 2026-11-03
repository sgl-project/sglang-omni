// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "auk/auk_conditioner.h"
#include "auk/auk_dit.h"
#include "auk/auk_vae.h"

namespace auk {

// The VAE's sample rate, and so every output's.
inline constexpr int kSampleRate = 24000;

struct SynthesisRequest {
  std::string instruction;
  std::vector<float> reference_audio;
  int reference_sample_rate = 24000;
  double seconds;
  std::optional<uint64_t> seed;
  bool match_reference_duration = false;
};

struct SynthesisResult {
  std::vector<float> samples;
  int sample_rate;
};

class Model {
public:
  // A released AuK or AuK-Flash checkpoint directory as published, and the
  // Qwen2.5-Omni-3B directory its conditioner runs.
  Model(const std::filesystem::path &model_directory,
        const std::filesystem::path &text_encoder_directory,
        mlx::core::Dtype dtype, float max_seconds);
  SynthesisResult Synthesize(const SynthesisRequest &request,
                             const std::function<void()> &check_cancelled);

private:
  struct SamplingRecipe {
    int steps;
    float guidance_strength;
    std::optional<float> sway_coefficient;
    std::vector<float> time_grid;
  };
  Model(const nlohmann::json &config,
        const std::filesystem::path &model_directory,
        const std::filesystem::path &text_encoder_directory,
        mlx::core::Dtype dtype, float max_seconds);
  static SamplingRecipe ReadSamplingRecipe(const nlohmann::json &config,
                                           mlx::core::Dtype dtype,
                                           float max_seconds);

  SamplingRecipe sampling_recipe;
  float max_seconds;
  Conditioner conditioner;
  DiT diffusion;
  VAE vae;
};

} // namespace auk
