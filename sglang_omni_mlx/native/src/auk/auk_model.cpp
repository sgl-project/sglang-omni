// SPDX-License-Identifier: Apache-2.0
#include "auk/auk_model.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string_view>

#include "common/audio.h"
#include "common/audio_resample.h"
#include "common/layers.h"

namespace auk {

namespace mx = mlx::core;

namespace {

constexpr int SAMPLES_PER_FRAME = 480;
constexpr int LATENT_CHANNELS = 64;
// Note (XXXXRT666): the released sampling recipes, as
// sglang_omni/models/auk/constants.py has them.
constexpr int kAukSteps = 32;
constexpr float kAukGuidanceStrength = 2.0f;
constexpr float kAukSwayCoefficient = -1.0f;
constexpr int kFlashSteps = 4;
constexpr float kFlashGuidanceStrength = 0.0f;
constexpr std::array<float, kFlashSteps + 1> kFlashTimeGrid = {
    0.0f, 0.07612049579620361f, 0.2928932309150696f, 0.6173166036605835f, 1.0f};
// The wrappers training saved the released weights under.
constexpr std::array<std::string_view, 4> kTrainingPrefixes = {
    "ema_model.", "module.", "model.", "net."};

// The model section of a released checkpoint's config.yaml.
nlohmann::json ReadCheckpointConfig(const std::filesystem::path &directory) {
  return omni_common::ReadYaml(directory / "config.yaml").at("model");
}

// The VAE's constructor arguments, whose latent_dim config.yaml interpolates
// from model.vae.latent_dim.
nlohmann::json VaeConfig(const nlohmann::json &config) {
  nlohmann::json vae_config = config.at("vae").at("model_init_kwargs");
  vae_config["latent_dim"] = config.at("vae").at("latent_dim");
  return vae_config;
}

omni_common::WeightMap
LoadCheckpointWeights(const std::filesystem::path &path) {
  const auto stored = mx::load_safetensors(path.string()).first;
  omni_common::WeightMap weights;
  for (const auto &[stored_name, weight] : stored) {
    std::string name = stored_name;
    for (const std::string_view prefix : kTrainingPrefixes) {
      if (name.starts_with(prefix)) {
        name.erase(0, prefix.size());
        break;
      } else {
      }
    }
    if (!weights.emplace(name, weight).second) {
      throw std::invalid_argument("duplicate AuK checkpoint weight: " + name);
    } else {
    }
  }
  return weights;
}

} // namespace

Model::SamplingRecipe Model::ReadSamplingRecipe(const nlohmann::json &config,
                                                mx::Dtype dtype) {
  if ((config.at("name") != "AuK" && config.at("name") != "AuK-Flash") ||
      config.at("vae").at("target_sample_rate") != kSampleRate ||
      config.at("vae").at("latent_dim") != LATENT_CHANNELS ||
      (dtype != mx::float32 && dtype != mx::bfloat16)) {
    throw std::invalid_argument("unsupported AuK runtime configuration");
  } else if (config.at("name") == "AuK-Flash") {
    mx::array times(kFlashTimeGrid.data(),
                    {static_cast<int>(kFlashTimeGrid.size())});
    mx::eval(times);
    return {times, kFlashGuidanceStrength};
  } else {
    mx::array times = mx::linspace(0, 1, kAukSteps + 1, mx::float32);
    times = times + mx::array(kAukSwayCoefficient) *
                        (mx::cos(mx::array(static_cast<float>(M_PI_2)) * times) -
                         mx::array(1.0f) + times);
    mx::eval(times);
    return {times, kAukGuidanceStrength};
  }
}

Model::Model(const std::filesystem::path &model_directory,
             const std::filesystem::path &text_encoder_directory,
             mx::Dtype dtype)
    : Model(ReadCheckpointConfig(model_directory), model_directory,
            text_encoder_directory, dtype) {}

Model::Model(const nlohmann::json &config,
             const std::filesystem::path &model_directory,
             const std::filesystem::path &text_encoder_directory,
             mx::Dtype dtype)
    : sampling_recipe(ReadSamplingRecipe(config, dtype)),
      conditioner(text_encoder_directory, dtype),
      diffusion(LoadCheckpointWeights(model_directory /
                                      (config.at("name") == "AuK-Flash"
                                           ? "auk_flash.safetensors"
                                           : "auk_base.safetensors")),
                config.at("arch"), config.at("vae").at("latent_dim").get<int>(),
                dtype),
      vae(LoadCheckpointWeights(model_directory / "vae.safetensors"),
          VaeConfig(config)) {
  diffusion.ValidateInputDimensions(conditioner.TextLayerCount(),
                                    conditioner.TextHiddenChannels());
  if (vae.HopSize() != SAMPLES_PER_FRAME) {
    throw std::invalid_argument("unsupported AuK VAE hop size");
  } else {
  }
}

SynthesisResult
Model::Synthesize(const SynthesisRequest &request,
                  const std::function<void()> &check_cancelled) {
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
    const auto reference_audio = omni_common::ResampleAudio(
        request.reference_audio, request.reference_sample_rate, kSampleRate);
    conditioner_audio = omni_common::ResampleAudio(
        request.reference_audio, request.reference_sample_rate,
        omni_common::kSampleRate);
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
                            request.seconds * kSampleRate / vae.HopSize())));
  const auto noise = mx::random::normal({output_frames, vae.LatentDimension()},
                                        mx::float32, generation_key);
  const auto latents =
      diffusion.Sample(conditioning, reference_latents, reference_length, noise,
                       sampling_recipe.times, sampling_recipe.guidance_strength,
                       check_cancelled);
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
            kSampleRate};
  }
}

} // namespace auk
