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
  Model(const std::filesystem::path &model_directory, mlx::core::Dtype dtype,
        float max_seconds);
  SynthesisResult Synthesize(const SynthesisRequest &request,
                             const std::function<void()> &check_cancelled);

private:
  nlohmann::json config;
  int sample_rate;
  float max_seconds;
  Conditioner conditioner;
  DiT diffusion;
  VAE vae;
};

} // namespace auk
