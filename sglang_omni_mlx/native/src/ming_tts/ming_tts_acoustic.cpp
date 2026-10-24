// SPDX-License-Identifier: Apache-2.0 AND MIT
#include "ming_tts/ming_tts_acoustic.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <map>
#include <stdexcept>

#include "ming_tts/ming_tts_model.h"

namespace ming_tts {

namespace mx = mlx::core;
namespace {
constexpr const char *kDit = "flowloss.cfm.model";
constexpr const char *kAggregator = "linear_proj_audio";
constexpr float kNormEpsilon = 1e-6f;
constexpr int kFrequencyEmbeddingDim = 256;
constexpr int kHalfFrequencyEmbeddingDim = kFrequencyEmbeddingDim / 2;
constexpr double kFrequencyBase = 10000.0;
constexpr int kTimestepScale = 1000;

mx::array GeluApprox(const mx::array &x) {
  static const auto compiled = mx::compile(
      [](const std::vector<mx::array> &inputs) {
        const auto &x = inputs[0];
        const auto dtype = x.dtype();
        return std::vector<mx::array>{mx::multiply(
            mx::multiply(mx::array(0.5f, dtype), x),
            mx::add(
                mx::array(1.0f, dtype),
                mx::tanh(mx::multiply(
                    mx::array(static_cast<float>(std::sqrt(2.0 / M_PI)), dtype),
                    mx::add(x, mx::multiply(
                                   mx::array(0.044715f, dtype),
                                   mx::power(x, mx::array(3, dtype))))))))};
      },
      true);
  return compiled({x})[0];
}

} // namespace

AcousticConfig AcousticConfig::FromJson(const nlohmann::json &config) {
  AcousticConfig result{config.at("hidden_size"), config.at("depth"),
                        config.at("num_heads"),
                        static_cast<int>(config.at("hidden_size").get<int>() *
                                         config.value("mlp_ratio", 4.0))};
  const auto pe_heads = config.value("pe_attn_head", nlohmann::json());
  if (std::min({result.hidden_size, result.depth, result.num_heads,
                result.intermediate_size}) <= 0 ||
      result.hidden_size % result.num_heads ||
      (result.hidden_size / result.num_heads) % 2 ||
      !config.value("qk_norm", nlohmann::json()).is_null() ||
      // Note (altale): Acoustic blocks do not implement speaker conditioning.
      !config.value("spk_dim", nlohmann::json()).is_null() ||
      (!pe_heads.is_null() && pe_heads != result.num_heads)) {
    throw std::invalid_argument("unsupported Ming acoustic configuration");
  } else {
    return result;
  }
}

MingAcoustic::MingAcoustic(const std::filesystem::path &directory)
    : MingAcoustic(qwen3_asr::ReadJson(directory / "config.json"), directory) {}

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
          [&] {
            if (std::min({patch_size_, history_frames_, latent_dim_,
                          llm_hidden_size_}) <= 0) {
              throw std::invalid_argument(
                  "Ming acoustic dimensions must be positive");
            } else {
              return LoadComponentWeights(directory,
                                          {"flowloss.", "linear_proj_audio.",
                                           "stop_head.", "spk_head."});
            }
          }(),
          config) {
  const auto linear = [&](const std::string &prefix, int output_size,
                          int input_size) {
    RequireShape(checkpoint_, prefix + ".weight", {output_size, input_size});
    RequireShape(checkpoint_, prefix + ".bias", {output_size});
  };
  for (const auto &[prefix, acoustic] :
       {std::pair{std::string(kDit), dit_},
        std::pair{std::string(kAggregator), aggregator_}}) {
    linear(prefix + ".x_embedder", acoustic.hidden_size, latent_dim_);
    for (int layer = 0; layer < acoustic.depth; ++layer) {
      const std::string block = prefix + ".blocks." + std::to_string(layer);
      for (const auto *norm : {"norm1", "norm2"}) {
        RequireShape(checkpoint_, block + "." + norm + ".weight",
                     {acoustic.hidden_size});
      }
      for (const auto *projection : {"to_q", "to_k", "to_v", "to_out.0"}) {
        linear(block + ".attn." + projection, acoustic.hidden_size,
               acoustic.hidden_size);
      }
      linear(block + ".mlp.ff.0.0", acoustic.intermediate_size,
             acoustic.hidden_size);
      linear(block + ".mlp.ff.2", acoustic.hidden_size,
             acoustic.intermediate_size);
    }
    RequireShape(checkpoint_, prefix + ".final_layer.norm_final.weight",
                 {acoustic.hidden_size});
    linear(prefix + ".final_layer.linear",
           prefix == kDit ? latent_dim_ : llm_hidden_size_,
           acoustic.hidden_size);
  }
  linear(std::string(kDit) + ".t_embedder.time_mlp.0", dit_.hidden_size,
         kFrequencyEmbeddingDim);
  linear(std::string(kDit) + ".t_embedder.time_mlp.2", dit_.hidden_size,
         dit_.hidden_size);
  linear(std::string(kDit) + ".c_embedder.cond_embedder", dit_.hidden_size,
         llm_hidden_size_);
  RequireShape(checkpoint_, std::string(kAggregator) + ".word_embedder.weight",
               {1, aggregator_.hidden_size});
  linear("stop_head", 2, llm_hidden_size_);
  linear("spk_head", llm_hidden_size_, 192);
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
            GeluApprox(checkpoint_.Linear(
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

mx::array MingAcoustic::Dit(const mx::array &latents, const mx::array &timestep,
                            const mx::array &hidden,
                            const mx::array &history) const {
  const std::string prefix = kDit;
  const mx::array frequencies = mx::exp(mx::multiply(
      mx::arange(kHalfFrequencyEmbeddingDim, mx::float32),
      mx::array(static_cast<float>(-std::log(kFrequencyBase) /
                                   (kHalfFrequencyEmbeddingDim - 1)))));
  const mx::array phase = mx::multiply(
      mx::multiply(mx::array(kTimestepScale),
                   mx::expand_dims(mx::astype(timestep, mx::float32), -1)),
      frequencies);
  const std::string time_prefix = prefix + ".t_embedder.time_mlp";
  const mx::array time = checkpoint_.Linear(
      qwen3_asr::Silu(checkpoint_.Linear(
          mx::astype(mx::concatenate({mx::sin(phase), mx::cos(phase)}, -1),
                     WeightDtype(checkpoint_, time_prefix + ".0")),
          time_prefix + ".0")),
      time_prefix + ".2");
  const std::string cond_prefix = prefix + ".c_embedder.cond_embedder";
  const mx::array conditioning =
      mx::add(mx::expand_dims(time, 1),
              checkpoint_.Linear(
                  mx::astype(hidden, WeightDtype(checkpoint_, cond_prefix)),
                  cond_prefix));
  const mx::array embedded = checkpoint_.Linear(
      mx::astype(mx::concatenate({history, latents}, 1),
                 WeightDtype(checkpoint_, prefix + ".x_embedder")),
      prefix + ".x_embedder");
  return Blocks(mx::concatenate({conditioning, embedded}, 1), dit_, prefix);
}

// Copyright (c) 2025 inclusionAI
// Adapted from the Ming-TTS flow-matching inference solver.
mx::array CfmTimesteps(int steps) {
  static const std::map<int, std::vector<float>> schedules = {
      {5, {0, 2, 4, 8, 16, 32}},
      {6, {0, 2, 4, 6, 8, 16, 32}},
      {7, {0, 2, 4, 6, 8, 16, 24, 32}},
      {10, {0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32}},
      {12, {0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32}},
      {16, {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32}}};
  if (steps < 1) {
    throw std::invalid_argument("CFM requires positive steps");
  } else if (const auto found = schedules.find(steps);
             found != schedules.end()) {
    return mx::divide(mx::array(found->second.data(),
                                {static_cast<int>(found->second.size())}),
                      mx::array(32.0f));
  } else {
    return mx::linspace(0, 1, steps + 1);
  }
}

TailOutput MingAcoustic::Step(const mx::array &hidden, const mx::array &history,
                              const mx::array &noise,
                              const mx::array &timesteps,
                              const mx::array &sde_random, float cfg,
                              float sigma, float temperature) const {
  assert(hidden.shape() == mx::Shape({noise.shape(0), 1, llm_hidden_size_}));
  assert(noise.shape() ==
         mx::Shape({hidden.shape(0), latent_dim_, patch_size_}));
  assert(history.shape() ==
         mx::Shape({hidden.shape(0), history_frames_, latent_dim_}));
  assert(timesteps.ndim() == 1 && timesteps.size() >= 2);
  assert(sde_random.shape() ==
         mx::Shape({timesteps.shape(0) - 2, hidden.shape(0), patch_size_,
                    latent_dim_}));
  mx::array sampled = mx::astype(mx::transpose(noise, {0, 2, 1}), mx::float32);
  const mx::array times = mx::add(
      timesteps,
      mx::multiply(
          mx::array(-1.0f),
          mx::add(mx::subtract(
                      mx::cos(mx::multiply(
                          mx::array(static_cast<float>(M_PI / 2)), timesteps)),
                      mx::array(1.0f)),
                  timesteps)));
  const mx::array conditioning =
      mx::concatenate({hidden, mx::zeros_like(hidden)}, 0);
  const mx::array doubled_history = mx::concatenate({history, history}, 0);
  for (int step = 0; step < times.shape(0) - 1; ++step) {
    const mx::array time = mx::take(times, mx::array(step), 0);
    const mx::array dt =
        mx::subtract(mx::take(times, mx::array(step + 1), 0), time);
    const mx::array output = Dit(mx::concatenate({sampled, sampled}, 0),
                                 mx::broadcast_to(time, {2 * hidden.shape(0)}),
                                 conditioning, doubled_history);
    const auto predictions =
        mx::split(mx::slice(output, {0, output.shape(1) - patch_size_, 0},
                            output.shape()),
                  2, 0);
    const mx::array velocity =
        mx::add(predictions[0],
                mx::multiply(mx::subtract(predictions[0], predictions[1]),
                             mx::array(cfg)));
    sampled = mx::add(sampled, mx::multiply(dt, velocity));
    if (step + 2 < times.shape(0)) {
      const mx::array shift = mx::multiply(
          mx::multiply(mx::array(sigma), mx::sqrt(mx::array(temperature))),
          mx::sqrt(mx::abs(dt)));
      sampled = mx::add(
          sampled,
          mx::multiply(shift, mx::take(sde_random, mx::array(step), 0)));
    } else {
    }
  }
  const mx::array stop =
      mx::softmax(checkpoint_.Linear(
                      mx::astype(hidden, WeightDtype(checkpoint_, "stop_head")),
                      "stop_head"),
                  -1);
  return {sampled,
          mx::reshape(Aggregate(sampled), {hidden.shape(0), llm_hidden_size_}),
          mx::reshape(mx::slice(stop, {0, 0, 1}, {hidden.shape(0), 1, 2}),
                      {hidden.shape(0)})};
}

} // namespace ming_tts
