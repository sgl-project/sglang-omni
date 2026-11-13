// SPDX-License-Identifier: Apache-2.0 AND MIT
#include "ming_tts/ming_tts_acoustic.h"

#include <array>
#include <cassert>
#include <cmath>
#include <stdexcept>

#include "ming_tts/ming_tts_model.h"

namespace ming_tts {

namespace mx = mlx::core;
namespace {
constexpr const char *kDit = "flowloss.cfm.model";
constexpr const char *kAggregator = "linear_proj_audio";
constexpr float kNormEpsilon = 1e-6f;
constexpr int kHalfFrequencyEmbeddingDim = 128;
constexpr double kFrequencyBase = 10000.0;
constexpr int kTimestepScale = 1000;
constexpr int kCfmSteps = 10;
// Copyright (c) 2025 inclusionAI
// Adapted from the Ming-TTS flow-matching inference solver: its 10-step
// schedule in 32nds, sway-sampled with this coefficient.
constexpr std::array<int, kCfmSteps + 1> kCfmSchedule = {0,  2,  4,  6,  8, 12,
                                                         16, 20, 24, 28, 32};
constexpr double kSwayCoefficient = -1.0;

std::vector<float> CfmTimes() {
  std::vector<float> times;
  for (const int step : kCfmSchedule) {
    const double t = step / 32.0;
    times.push_back(static_cast<float>(
        t + kSwayCoefficient * (std::cos(M_PI / 2 * t) - 1 + t)));
  }
  return times;
}

} // namespace

AcousticConfig AcousticConfig::FromJson(const nlohmann::json &config) {
  const auto pe_heads = config.value("pe_attn_head", nlohmann::json());
  const int num_heads = config.at("num_heads");
  if (!config.value("qk_norm", nlohmann::json()).is_null() ||
      // Note (altale): Acoustic blocks do not implement speaker conditioning.
      !config.value("spk_dim", nlohmann::json()).is_null() ||
      (!pe_heads.is_null() && pe_heads != num_heads)) {
    throw std::invalid_argument("unsupported Ming acoustic configuration");
  } else {
    return {config.at("hidden_size"), config.at("depth"), num_heads};
  }
}

MingAcoustic::MingAcoustic(const std::filesystem::path &directory)
    : MingAcoustic(omni_common::ReadJson(directory / "config.json"),
                   directory) {}

MingAcoustic::MingAcoustic(const nlohmann::json &config,
                           const std::filesystem::path &directory)
    : dit_(AcousticConfig::FromJson(config.at("ditar_config"))),
      aggregator_(AcousticConfig::FromJson(config.at("aggregator_config"))),
      patch_size_(config.at("ditar_config").at("patch_size")),
      history_frames_(config.at("ditar_config").at("history_patch_size")),
      latent_dim_(config.at("audio_tokenizer_config")
                      .at("enc_kwargs")
                      .at("latent_dim")),
      llm_hidden_size_(config.at("llm_config").at("hidden_size")),
      checkpoint_(
          LoadComponentWeights(directory, {"flowloss.", "linear_proj_audio.",
                                           "stop_head.", "spk_head."}),
          config),
      cfm_times_(CfmTimes()), time_embeddings_(TimeEmbeddings()) {}

mx::array MingAcoustic::TimeEmbeddings() const {
  const std::string prefix = std::string(kDit) + ".t_embedder.time_mlp";
  const mx::array frequencies = mx::exp(mx::multiply(
      mx::arange(kHalfFrequencyEmbeddingDim, mx::float32),
      mx::array(static_cast<float>(-std::log(kFrequencyBase) /
                                   (kHalfFrequencyEmbeddingDim - 1)))));
  const mx::array phase = mx::multiply(
      mx::multiply(
          mx::array(kTimestepScale),
          mx::expand_dims(mx::array(cfm_times_.data(), {kCfmSteps}), -1)),
      frequencies);
  const mx::array embeddings = mx::expand_dims(
      checkpoint_.Linear(
          omni_common::Silu(checkpoint_.Linear(
              mx::astype(mx::concatenate({mx::sin(phase), mx::cos(phase)}, -1),
                         WeightDtype(checkpoint_, prefix + ".0")),
              prefix + ".0")),
          prefix + ".2"),
      1);
  mx::eval(embeddings);
  return embeddings;
}

mx::array MingAcoustic::ProjectSpeaker(const mx::array &speaker) const {
  return checkpoint_.Linear(
      mx::astype(speaker, WeightDtype(checkpoint_, "spk_head")), "spk_head");
}

mx::array MingAcoustic::Blocks(mx::array hidden, const AcousticConfig &config,
                               const std::string &prefix) const {
  const int batch = hidden.shape(0), length = hidden.shape(1),
            head_dim = config.hidden_size / config.num_heads;
  for (int layer = 0; layer < config.depth; ++layer) {
    const std::string block = prefix + ".blocks." + std::to_string(layer);
    const mx::array normed =
        checkpoint_.RmsNorm(hidden, block + ".norm1", kNormEpsilon);
    const auto project = [&](const std::string &projection) {
      return mx::transpose(
          mx::reshape(checkpoint_.Linear(normed, block + ".attn." + projection),
                      {batch, length, config.num_heads, head_dim}),
          {0, 2, 1, 3});
    };
    const mx::array attended = mx::fast::scaled_dot_product_attention(
        mx::fast::rope(project("to_q"), head_dim, true,
                       static_cast<float>(kFrequencyBase), 1.0f, 0),
        mx::fast::rope(project("to_k"), head_dim, true,
                       static_cast<float>(kFrequencyBase), 1.0f, 0),
        project("to_v"),
        static_cast<float>(std::pow(static_cast<double>(head_dim), -0.5)));
    hidden = mx::add(
        hidden,
        checkpoint_.Linear(mx::reshape(mx::transpose(attended, {0, 2, 1, 3}),
                                       {batch, length, config.hidden_size}),
                           block + ".attn.to_out.0"));
    hidden = mx::add(
        hidden,
        checkpoint_.Linear(
            omni_common::TanhGelu(checkpoint_.Linear(
                checkpoint_.RmsNorm(hidden, block + ".norm2", kNormEpsilon),
                block + ".mlp.ff.0.0")),
            block + ".mlp.ff.2"));
  }
  return checkpoint_.Linear(
      checkpoint_.RmsNorm(hidden, prefix + ".final_layer.norm_final",
                          kNormEpsilon),
      prefix + ".final_layer.linear");
}

mx::array MingAcoustic::Aggregate(const mx::array &latents) const {
  assert(latents.ndim() == 3 && latents.shape(2) == latent_dim_);
  const std::string prefix = kAggregator;
  const mx::array hidden = checkpoint_.Linear(
      mx::astype(latents, WeightDtype(checkpoint_, prefix + ".x_embedder")),
      prefix + ".x_embedder");
  const mx::array cls = checkpoint_.Embed(
      mx::zeros({latents.shape(0), 1}, mx::int32), prefix + ".word_embedder");
  const mx::array output =
      Blocks(mx::concatenate({cls, hidden}, 1), aggregator_, prefix);
  return mx::slice(output, {0, 0, 0}, {latents.shape(0), 1, llm_hidden_size_});
}

TailOutput MingAcoustic::Step(const mx::array &hidden, const mx::array &history,
                              float cfg, float sigma, float temperature) const {
  const int batch = hidden.shape(0);
  assert(hidden.shape() == mx::Shape({batch, 1, llm_hidden_size_}));
  assert(history.shape() == mx::Shape({batch, history_frames_, latent_dim_}));
  const std::string prefix = kDit;
  const std::string conditioning_prefix = prefix + ".c_embedder.cond_embedder";
  const std::string embedder_prefix = prefix + ".x_embedder";
  const mx::Dtype embedder_dtype = WeightDtype(checkpoint_, embedder_prefix);
  const auto doubled = [](const mx::array &x) {
    return mx::concatenate({x, x}, 0);
  };
  // The guided and unguided branches run as one batch; every CFM step shares
  // their conditioning and history.
  const mx::array conditioning = checkpoint_.Linear(
      mx::astype(mx::concatenate({hidden, mx::zeros_like(hidden)}, 0),
                 WeightDtype(checkpoint_, conditioning_prefix)),
      conditioning_prefix);
  const mx::array history_embedding = doubled(
      checkpoint_.Linear(mx::astype(history, embedder_dtype), embedder_prefix));
  mx::array sampled =
      mx::random::normal({batch, patch_size_, latent_dim_}, mx::float32);
  const float noise_scale = sigma * std::sqrt(temperature);
  const std::optional<mx::array> sde_noise =
      noise_scale > 0
          ? std::optional(mx::random::normal(
                {kCfmSteps - 1, batch, patch_size_, latent_dim_}, mx::float32))
          : std::nullopt;
  for (int step = 0; step < kCfmSteps; ++step) {
    const mx::array output = Blocks(
        mx::concatenate(
            {mx::add(conditioning, mx::slice(time_embeddings_, {step, 0, 0},
                                             {step + 1, 1, dit_.hidden_size})),
             history_embedding,
             doubled(checkpoint_.Linear(mx::astype(sampled, embedder_dtype),
                                        embedder_prefix))},
            1),
        dit_, prefix);
    const auto predictions =
        mx::split(mx::slice(output, {0, output.shape(1) - patch_size_, 0},
                            output.shape()),
                  2, 0);
    const mx::array velocity =
        mx::add(predictions[0],
                mx::multiply(mx::subtract(predictions[0], predictions[1]),
                             mx::array(cfg)));
    const float dt = cfm_times_[step + 1] - cfm_times_[step];
    sampled = mx::add(sampled, mx::multiply(mx::array(dt), velocity));
    if (sde_noise.has_value() && step + 1 < kCfmSteps) {
      sampled = mx::add(
          sampled,
          mx::multiply(mx::array(noise_scale * std::sqrt(std::abs(dt))),
                       mx::reshape(mx::slice(*sde_noise, {step, 0, 0, 0},
                                             {step + 1, batch, patch_size_,
                                              latent_dim_}),
                                   sampled.shape())));
    } else {
    }
  }
  const mx::array stop =
      mx::softmax(checkpoint_.Linear(
                      mx::astype(hidden, WeightDtype(checkpoint_, "stop_head")),
                      "stop_head"),
                  -1);
  return {sampled,
          mx::astype(
              mx::reshape(mx::slice(stop, {0, 0, 1}, {batch, 1, 2}), {batch}),
              mx::float32)};
}

} // namespace ming_tts
