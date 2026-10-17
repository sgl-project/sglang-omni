// SPDX-License-Identifier: Apache-2.0
#include "auk/auk_model.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

#include "common/audio_resample.h"

namespace auk {

namespace mx = mlx::core;

namespace {

nlohmann::json ReadConfig(const std::filesystem::path &directory,
                          mx::Dtype dtype, float max_seconds) {
  const auto config = qwen3_asr::ReadJson(directory / "config.json");
  if (config.value("model_type", std::string()) != "auk" ||
      (config.at("name") != "AuK" && config.at("name") != "AuK-Flash") ||
      config.at("sample_rate") != 24000 ||
      config.at("downsample_rate") != 480 || config.at("latent_dim") != 64 ||
      (dtype != mx::float32 && dtype != mx::bfloat16) ||
      !std::isfinite(max_seconds) || max_seconds <= 0 || max_seconds > 30) {
    throw std::invalid_argument("unsupported AuK runtime configuration");
  } else {
  }
  const int steps = config.at("nfe").get<int>();
  const float guidance = config.at("cfg_strength").get<float>();
  const auto grid = config.at("time_grid").get<std::vector<float>>();
  if (steps <= 0 || !std::isfinite(guidance) || guidance < 0 ||
      (!config.at("sway_sampling_coef").is_null() &&
       !std::isfinite(config.at("sway_sampling_coef").get<float>())) ||
      (!grid.empty() &&
       (grid.size() != static_cast<size_t>(steps + 1) || grid.front() != 0 ||
        grid.back() != 1 ||
        std::any_of(grid.begin(), grid.end(),
                    [](float time) { return !std::isfinite(time); }) ||
        std::adjacent_find(grid.begin(), grid.end(),
                           std::greater_equal<float>()) != grid.end()))) {
    throw std::invalid_argument("invalid AuK sampling recipe");
  } else {
    return config;
  }
}

} // namespace

Model::Model(const std::filesystem::path &directory, mx::Dtype dtype,
             float maximum_seconds)
    : config(ReadConfig(directory, dtype, maximum_seconds)),
      sample_rate(config.at("sample_rate").get<int>()),
      max_seconds(maximum_seconds),
      conditioner(directory / "conditioner", dtype),
      diffusion(directory / "dit", dtype), vae(directory / "vae") {
  diffusion.ValidateInputDimensions(conditioner.TextLayerCount(),
                                    conditioner.TextHiddenChannels(),
                                    vae.LatentDimension());
  if (vae.HopSize() != config.at("downsample_rate").get<int>() ||
      vae.LatentDimension() != config.at("latent_dim").get<int>()) {
    throw std::invalid_argument(
        "AuK artifact components have inconsistent dimensions");
  } else {
  }
}

SynthesisResult
Model::Synthesize(const SynthesisRequest &request,
                  const std::function<void()> &check_cancelled) {
  if (request.instruction.empty() || request.instruction.size() > 32768 ||
      !std::isfinite(request.seconds) || request.seconds <= 0 ||
      request.seconds > max_seconds || request.reference_sample_rate <= 0 ||
      (!request.reference_audio.empty() &&
       request.reference_audio.size() <
           static_cast<size_t>(
               std::ceil(request.reference_sample_rate * 0.03))) ||
      request.reference_audio.size() >
          static_cast<size_t>(request.reference_sample_rate * max_seconds) ||
      !std::all_of(request.reference_audio.begin(),
                   request.reference_audio.end(),
                   [](float sample) { return std::isfinite(sample); })) {
    throw std::invalid_argument("invalid AuK synthesis input");
  } else {
  }
  check_cancelled();
  uint64_t seed;
  if (request.seed) {
    seed = *request.seed;
  } else {
    std::random_device entropy;
    seed = (static_cast<uint64_t>(entropy()) << 32) | entropy();
  }
  const auto [reference_key, generation_key] =
      mx::random::split(mx::random::key(seed));
  std::optional<mx::array> reference_latents;
  std::optional<std::vector<float>> conditioner_audio;
  int reference_length = 0;
  if (!request.reference_audio.empty()) {
    const auto reference_audio = qwen3_asr::ResampleAudio(
        request.reference_audio, request.reference_sample_rate, sample_rate);
    conditioner_audio = qwen3_asr::ResampleAudio(
        request.reference_audio, request.reference_sample_rate, 16000);
    reference_length = static_cast<int>(reference_audio.size()) / vae.HopSize();
    const auto waveform = mx::array(reference_audio.data(),
                                    {static_cast<int>(reference_audio.size())});
    const auto posterior = vae.Posterior(waveform);
    const auto posterior_noise =
        mx::random::normal({1, posterior.mean.shape(0), vae.LatentDimension()},
                           mx::float32, reference_key);
    reference_latents = vae.Encode(posterior, mx::squeeze(posterior_noise, 0));
    reference_length = std::min(reference_length, reference_latents->shape(0));
    mx::eval(*reference_latents);
  } else {
  }
  check_cancelled();
  const auto conditioning = diffusion.FuseHiddenStates(
      conditioner.Encode(request.instruction, conditioner_audio));
  mx::eval(conditioning);
  check_cancelled();
  const int output_frames =
      request.match_reference_duration
          ? std::max(1, reference_length)
          : std::max(1, static_cast<int>(std::ceil(
                            request.seconds * sample_rate / vae.HopSize())));
  const auto noise = mx::random::normal({output_frames, vae.LatentDimension()},
                                        mx::float32, generation_key);
  const auto sway =
      config.at("sway_sampling_coef").is_null()
          ? std::nullopt
          : std::optional<float>(config.at("sway_sampling_coef").get<float>());
  const auto latents = diffusion.Sample(
      conditioning, reference_latents, reference_length, noise,
      config.at("nfe").get<int>(), config.at("cfg_strength").get<float>(), sway,
      config.at("time_grid").get<std::vector<float>>(), check_cancelled);
  if (!mx::all(mx::isfinite(latents)).item<bool>()) {
    throw std::runtime_error("AuK produced non-finite latents");
  } else {
  }
  check_cancelled();
  const auto waveform = vae.Decode(latents);
  mx::eval(waveform);
  if (!mx::all(mx::isfinite(waveform)).item<bool>()) {
    throw std::runtime_error("AuK produced non-finite audio");
  } else {
    check_cancelled();
    return {{waveform.data<float>(), waveform.data<float>() + waveform.size()},
            sample_rate};
  }
}

} // namespace auk
