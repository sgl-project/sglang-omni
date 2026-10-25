// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Prince Canuma and contributors.
#include "music3/music3_model.h"

#include <algorithm>
#include <cmath>
#include <optional>

namespace music3 {
namespace mx = mlx::core;

mx::array Model::Convolve(const mx::array &hidden_states,
                          const std::string &prefix, int padding, int dilation,
                          int stride, bool transpose) const {
  mx::array output =
      transpose
          ? mx::conv_transpose1d(hidden_states,
                                 checkpoint_.Weight(prefix + ".weight"), stride,
                                 padding, dilation)
          : mx::conv1d(hidden_states, checkpoint_.Weight(prefix + ".weight"),
                       stride, padding, dilation);
  if (checkpoint_.Has(prefix + ".bias")) {
    return output + checkpoint_.Weight(prefix + ".bias");
  } else {
    return output;
  }
}

mx::array Model::Condition(const mx::array &hidden_states) const {
  const int frames = hidden_states.shape(1);
  const int hidden_size = config_.at("hidden_size");
  const mx::Dtype dtype =
      checkpoint_.Weight("condition_encoder.proj.weight").dtype();
  const mx::array hidden =
      mx::transpose(mx::reshape(mx::astype(hidden_states, dtype),
                                {1, frames, 8, hidden_size}),
                    {0, 2, 3, 1});
  const mx::array weights = mx::astype(
      mx::softmax(mx::astype(checkpoint_.Weight(
                                 "condition_encoder.layer_weight_logits"),
                             mx::float32),
                  0),
      dtype);
  mx::array projected = mx::sum(hidden * mx::reshape(weights, {1, 8, 1, 1}), 1);
  projected =
      mx::astype(checkpoint_.Weight("condition_encoder.layer_scale"), dtype) *
      projected;
  projected = Convolve(mx::transpose(projected, {0, 2, 1}),
                       "condition_encoder.proj", 1);
  const int latent_count =
      std::max(1, static_cast<int>(frames * 44100.0 / 24000 * 960 / 512));
  const mx::array indices = mx::astype(
      mx::arange(latent_count) *
          static_cast<float>(static_cast<double>(frames) / latent_count),
      mx::int32);
  return mx::take(projected,
                  mx::clip(indices, mx::array(0), mx::array(frames - 1)), 1);
}

mx::array Model::Velocity(const mx::array &latents, float sigma,
                          const mx::array &condition) const {
  const int channels = config_.at("dit_in_channels");
  const int heads = config_.at("dit_num_heads");
  const int head_dim = config_.at("dit_head_dim");
  const int rotary_dim = config_.at("dit_rotary_dim");
  mx::array hidden =
      mx::transpose(mx::concatenate({latents, mx::zeros_like(latents),
                                     mx::transpose(condition, {0, 2, 1})},
                                    1),
                    {0, 2, 1});
  hidden = Convolve(hidden, "transformer.preprocess_conv") + hidden;
  const mx::array timestep = mx::full({1, 1}, sigma, latents.dtype());
  const mx::array angles = mx::matmul(
      timestep * mx::array(static_cast<float>(2 * M_PI), timestep.dtype()),
      mx::transpose(checkpoint_.Weight("transformer.time_proj.weight")));
  const mx::array time_embedding = checkpoint_.Linear(
      omni_common::Silu(checkpoint_.Linear(
          mx::concatenate({mx::cos(angles), mx::sin(angles)}, -1),
          "transformer.time_embed.linear_1")),
      "transformer.time_embed.linear_2");
  hidden = mx::concatenate({mx::expand_dims(time_embedding, 1),
                            checkpoint_.Linear(hidden, "transformer.proj_in")},
                           1);
  const int length = hidden.shape(1);
  const mx::array inverse_frequencies = mx::reciprocal(mx::power(
      mx::array(10000.0f),
      mx::astype(mx::arange(0, rotary_dim, 2), mx::float32) / rotary_dim));
  mx::array frequencies = mx::outer(mx::astype(mx::arange(length), mx::float32),
                                    inverse_frequencies);
  frequencies = mx::concatenate({frequencies, frequencies}, -1);
  const mx::array cosine =
      mx::astype(mx::expand_dims(mx::cos(frequencies), 1), hidden.dtype());
  const mx::array sine =
      mx::astype(mx::expand_dims(mx::sin(frequencies), 1), hidden.dtype());
  for (int layer = 0; layer < config_.at("dit_num_layers").get<int>();
       ++layer) {
    const std::string prefix =
        "transformer.transformer_blocks." + std::to_string(layer);
    const mx::array normalized =
        checkpoint_.LayerNorm(hidden, prefix + ".norm1", 1e-5f);
    const auto project = [&](const std::string &name, bool rotary) {
      mx::array projected =
          mx::reshape(checkpoint_.Linear(normalized, prefix + ".attn." + name),
                      {1, length, heads, head_dim});
      if (rotary) {
        const mx::array left = mx::slice(projected, {0, 0, 0, 0},
                                         {1, length, heads, rotary_dim / 2});
        const mx::array right = mx::slice(projected, {0, 0, 0, rotary_dim / 2},
                                          {1, length, heads, rotary_dim});
        const mx::array rotated =
            mx::slice(projected, {0, 0, 0, 0}, {1, length, heads, rotary_dim}) *
                cosine +
            mx::concatenate({-right, left}, -1) * sine;
        projected = mx::concatenate(
            {rotated,
             mx::slice(projected, {0, 0, 0, rotary_dim}, projected.shape())},
            -1);
      } else {
      }
      return mx::transpose(projected, {0, 2, 1, 3});
    };
    const mx::array attended = mx::fast::scaled_dot_product_attention(
        project("to_q", true), project("to_k", true), project("to_v", false),
        static_cast<float>(std::pow(static_cast<double>(head_dim), -0.5)));
    hidden = hidden + checkpoint_.Linear(
                          mx::reshape(mx::transpose(attended, {0, 2, 1, 3}),
                                      {1, length, heads * head_dim}),
                          prefix + ".attn.to_out.0");
    const auto feed_forward =
        mx::split(checkpoint_.Linear(
                      checkpoint_.LayerNorm(hidden, prefix + ".norm2", 1e-5f),
                      prefix + ".ff_in"),
                  2, -1);
    hidden = hidden + checkpoint_.Linear(feed_forward[0] *
                                             omni_common::Silu(feed_forward[1]),
                                         prefix + ".ff_out");
  }
  hidden = checkpoint_.Linear(mx::slice(hidden, {0, 1, 0}, hidden.shape()),
                              "transformer.proj_out");
  hidden = Convolve(hidden, "transformer.postprocess_conv") + hidden;
  return mx::transpose(mx::reshape(hidden, {1, length - 1, channels}),
                       {0, 2, 1});
}

mx::array Model::Snake(const mx::array &hidden_states,
                       const std::string &prefix) const {
  const mx::array alpha = mx::transpose(
      mx::astype(checkpoint_.Weight(prefix + ".alpha"), hidden_states.dtype()),
      {0, 2, 1});
  // Note (Akazaakane): power and square round differently in BF16.
  return hidden_states + mx::power(mx::sin(alpha * hidden_states),
                                   mx::array(2, hidden_states.dtype())) /
                             (alpha + mx::array(1e-9f, alpha.dtype()));
}

mx::array Model::Vocoder(const mx::array &latents) const {
  const int channels = config_.at("dit_in_channels");
  mx::array hidden = mx::transpose(
      mx::reshape(latents, {2, channels / 2, latents.shape(2)}), {0, 2, 1});
  hidden =
      Convolve(Convolve(hidden, "vocoder.dec_in_proj"), "vocoder.conv_in", 3);
  const auto ratios =
      config_.at("vocoder_upsampling_ratios").get<std::vector<int>>();
  for (size_t index = 0; index < ratios.size(); ++index) {
    const std::string prefix = "vocoder.blocks." + std::to_string(index);
    const int stride = ratios[index];
    hidden = Convolve(Snake(hidden, prefix + ".snake1"), prefix + ".conv_t1",
                      (stride + 1) / 2, 1, stride, true);
    int dilation = 1;
    for (int unit = 1; unit <= 3; ++unit, dilation *= 3) {
      const std::string residual = prefix + ".res_unit" + std::to_string(unit);
      const mx::array first =
          Convolve(Snake(hidden, residual + ".snake1"), residual + ".conv1",
                   3 * dilation, dilation);
      hidden = hidden + Convolve(Snake(first, residual + ".snake2"),
                                 residual + ".conv2");
    }
  }
  return mx::reshape(mx::tanh(Convolve(Snake(hidden, "vocoder.snake_out"),
                                       "vocoder.conv_out", 3)),
                     {2, -1});
}

Result Model::Synthesize(const Request &request,
                         const std::atomic<bool> &cancelled) const {
  const mx::array frames = GenerateFrames(request, cancelled);
  mx::eval(frames);
  const int frame_count = frames.shape(1);
  std::optional<mx::array> previous_latent;
  std::optional<mx::array> previous_condition;
  std::vector<mx::array> wave_chunks;
  for (int start = 0, chunk = 0; start < frame_count;
       start += kChunkHopFrames, ++chunk) {
    CheckCancelled(cancelled);
    const int end = std::min(start + kChunkFrames, frame_count);
    mx::array condition =
        Condition(mx::slice(frames, {0, start, 0}, {1, end, frames.shape(2)}));
    const int latent_count = condition.shape(1);
    mx::array latent = mx::astype(
        mx::random::normal(
            {1, config_.at("dit_in_channels").get<int>(), latent_count},
            mx::float32, mx::random::key(AcousticSeed(request.seed, chunk))),
        condition.dtype());
    const int overlap = previous_latent.has_value()
                            ? std::min(previous_latent->shape(2), latent_count)
                            : 0;
    const mx::array noise =
        mx::slice(latent, {0, 0, 0}, {1, latent.shape(1), overlap});
    if (overlap > 0) {
      condition = mx::concatenate(
          {mx::slice(*previous_condition, {0, 0, 0},
                     {1, overlap, condition.shape(2)}),
           mx::slice(condition, {0, overlap, 0}, condition.shape())},
          1);
    } else {
    }
    const mx::array zeros = mx::zeros_like(condition);
    for (int step = 0; step < request.dit_steps; ++step) {
      CheckCancelled(cancelled);
      // Note (Akazaakane): NumPy linspace rounds before subtraction in the
      // reference scheduler.
      const auto sigma_at = [&](int index) {
        if (index == request.dit_steps) {
          return 1.0f;
        } else if (request.dit_steps == 1) {
          return 0.0f;
        } else {
          const float raw = static_cast<float>(
              1.0 + index * ((1.0 / request.dit_steps - 1.0) /
                             (request.dit_steps - 1)));
          return 1.0f - raw;
        }
      };
      const float sigma = sigma_at(step);
      if (overlap > 0) {
        const mx::array prompt =
            mx::array(static_cast<float>(1.0 - (1.0 - 1e-6) * sigma),
                      noise.dtype()) *
                noise +
            mx::array(sigma, noise.dtype()) *
                mx::slice(*previous_latent, {0, 0, 0},
                          {1, latent.shape(1), overlap});
        latent = mx::concatenate(
            {prompt, mx::slice(latent, {0, 0, overlap}, latent.shape())}, 2);
      } else {
      }
      const mx::array conditional = Velocity(latent, sigma, condition);
      mx::array velocity = conditional;
      if (request.dit_guidance != 1.0f) {
        const mx::array unconditional = Velocity(latent, sigma, zeros);
        velocity = unconditional +
                   mx::array(request.dit_guidance, unconditional.dtype()) *
                       (conditional - unconditional);
      } else {
      }
      latent =
          latent +
          mx::array(sigma_at(step + 1) - sigma, velocity.dtype()) * velocity;
      mx::eval(latent);
    }
    if (overlap > 0) {
      latent =
          mx::concatenate({mx::slice(*previous_latent, {0, 0, 0},
                                     {1, latent.shape(1), overlap}),
                           mx::slice(latent, {0, 0, overlap}, latent.shape())},
                          2);
    } else {
    }
    CheckCancelled(cancelled);
    const mx::array waveform =
        mx::clip(mx::astype(Vocoder(latent), mx::float32), mx::array(-1.0f),
                 mx::array(1.0f));
    const int left_samples = start == 0 ? 0 : 86 * kLatentHopSamples;
    const int right_samples = end == frame_count ? 0 : 258 * kLatentHopSamples;
    wave_chunks.push_back(mx::slice(waveform, {0, left_samples},
                                    {2, waveform.shape(1) - right_samples}));
    const int saved_start = std::max(0, latent_count - 2 * kOverlapLatents);
    const int saved_end = std::max(saved_start, latent_count - kOverlapLatents);
    previous_latent =
        mx::slice(latent, {0, 0, saved_start}, {1, latent.shape(1), saved_end});
    previous_condition = mx::slice(condition, {0, saved_start, 0},
                                   {1, saved_end, condition.shape(2)});
    mx::eval({wave_chunks.back(), *previous_latent, *previous_condition});
    mx::clear_cache();
    if (end == frame_count) {
      break;
    } else {
    }
  }
  CheckCancelled(cancelled);
  return {ResampleStereo(mx::concatenate(wave_chunks, 1)), frame_count};
}

} // namespace music3
