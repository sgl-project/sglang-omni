// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Tencent. All rights reserved.
#include "auk/auk_dit.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace auk {
namespace mx = mlx::core;
using qwen3_asr::Checkpoint;
using qwen3_asr::Silu;

namespace {

constexpr float kRmsEpsilon = std::numeric_limits<float>::epsilon();
constexpr float kLayerNormEpsilon = 1e-6f;

mx::array Normalize(const mx::array &hidden_states) {
  return mx::fast::layer_norm(hidden_states, std::nullopt, std::nullopt,
                              kLayerNormEpsilon);
}

mx::array ApplyRotary(const mx::array &hidden_states,
                      const mx::array &inverse_frequencies) {
  const int channels = hidden_states.shape(3);
  const mx::array angles =
      mx::multiply(mx::reshape(mx::arange(hidden_states.shape(2), mx::float32),
                               {1, 1, -1, 1}),
                   inverse_frequencies);
  mx::Shape stop = hidden_states.shape();
  const mx::array even =
      mx::slice(hidden_states, {0, 0, 0, 0}, stop, {1, 1, 1, 2});
  const mx::array odd =
      mx::slice(hidden_states, {0, 0, 0, 1}, stop, {1, 1, 1, 2});
  const mx::array cosine = mx::cos(angles), sine = mx::sin(angles);
  return mx::astype(mx::reshape(mx::stack({even * cosine - odd * sine,
                                           odd * cosine + even * sine},
                                          -1),
                                {hidden_states.shape(0), hidden_states.shape(1),
                                 hidden_states.shape(2), channels}),
                    hidden_states.dtype());
}

struct AttentionProjection {
  mx::array queries, keys, values;
};

AttentionProjection
ProjectAttention(const Checkpoint &checkpoint, const mx::array &hidden_states,
                 const std::string &prefix, const std::string &suffix,
                 const mx::array &inverse_frequencies, int head_count) {
  const auto projected = mx::split(
      checkpoint.Linear(hidden_states, prefix + ".to_qkv" + suffix), 3, -1);
  std::vector<mx::array> heads;
  for (const auto &projection : projected) {
    heads.push_back(mx::transpose(
        mx::reshape(projection,
                    {projection.shape(0), projection.shape(1), head_count, -1}),
        {0, 2, 1, 3}));
  }
  const std::string norm_prefix = prefix + (suffix.empty() ? "." : ".c_");
  return {ApplyRotary(
              checkpoint.RmsNorm(heads[0], norm_prefix + "q_norm", kRmsEpsilon),
              inverse_frequencies),
          ApplyRotary(
              checkpoint.RmsNorm(heads[1], norm_prefix + "k_norm", kRmsEpsilon),
              inverse_frequencies),
          heads[2]};
}

std::vector<mx::array>
Attention(const Checkpoint &checkpoint, const mx::array &hidden_states,
          const std::optional<mx::array> &conditioning,
          const std::string &prefix, const mx::array &inverse_frequencies,
          int head_count, const std::optional<mx::array> &valid_positions,
          bool attention_mask_enabled) {
  auto projection = ProjectAttention(checkpoint, hidden_states, prefix, "",
                                     inverse_frequencies, head_count);
  if (conditioning) {
    const auto text = ProjectAttention(checkpoint, *conditioning, prefix, "_c",
                                       inverse_frequencies, head_count);
    projection.queries = mx::concatenate({projection.queries, text.queries}, 2);
    projection.keys = mx::concatenate({projection.keys, text.keys}, 2);
    projection.values = mx::concatenate({projection.values, text.values}, 2);
  } else {
  }
  const std::optional<mx::array> mask =
      valid_positions && attention_mask_enabled
          ? std::optional(mx::expand_dims(*valid_positions, {1, 2}))
          : std::nullopt;
  const mx::array attended = mx::fast::scaled_dot_product_attention(
      projection.queries, projection.keys, projection.values,
      1.0f / std::sqrt(static_cast<float>(projection.queries.shape(3))), "",
      mask);
  const int batch_count = hidden_states.shape(0),
            frame_count = hidden_states.shape(1);
  const mx::array merged =
      mx::reshape(mx::transpose(attended, {0, 2, 1, 3}),
                  {batch_count, -1, attended.shape(1) * attended.shape(3)});
  mx::array audio = checkpoint.Linear(
      mx::slice(merged, {0, 0, 0}, {batch_count, frame_count, merged.shape(2)}),
      prefix + ".to_out.0");
  if (valid_positions) {
    audio = mx::where(mx::expand_dims(mx::slice(*valid_positions, {0, 0},
                                                {batch_count, frame_count}),
                                      -1),
                      audio, mx::zeros_like(audio));
  } else {
  }
  if (conditioning) {
    mx::array text = checkpoint.Linear(
        mx::slice(merged, {0, frame_count, 0}, merged.shape()),
        prefix + ".to_out_c");
    if (valid_positions) {
      text = mx::where(
          mx::expand_dims(mx::slice(*valid_positions, {0, frame_count},
                                    valid_positions->shape()),
                          -1),
          text, mx::zeros_like(text));
    } else {
    }
    return {audio, text};
  } else {
    return {audio};
  }
}

struct Modulation {
  mx::array normalized, attention_gate, shift, scale, feed_forward_gate;
};

Modulation Modulate(const Checkpoint &checkpoint,
                    const mx::array &hidden_states, const mx::array &timestep,
                    const std::string &prefix) {
  auto modulation = mx::split(
      mx::expand_dims(checkpoint.Linear(Silu(timestep), prefix + ".linear"), 1),
      6, -1);
  const mx::array one(1.0f, hidden_states.dtype());
  return {Normalize(hidden_states) * (one + modulation[1]) + modulation[0],
          modulation[2], modulation[3], modulation[4], modulation[5]};
}

mx::array FeedForward(const Checkpoint &checkpoint,
                      const mx::array &hidden_states,
                      const Modulation &modulation, const std::string &prefix) {
  const mx::array normalized =
      Normalize(hidden_states) *
          (mx::array(1.0f, hidden_states.dtype()) + modulation.scale) +
      modulation.shift;
  const auto gates =
      mx::split(checkpoint.Linear(normalized, prefix + ".linear_in"), 2, -1);
  return hidden_states + modulation.feed_forward_gate *
                             checkpoint.Linear(Silu(gates[0]) * gates[1],
                                               prefix + ".linear_out");
}

} // namespace

mx::array DiT::FuseHiddenStates(const mx::array &all_hidden_states) const {
  const int layer_count = checkpoint_.Weight("layer_weights").size();
  if (all_hidden_states.ndim() != 3 ||
      all_hidden_states.shape(0) != layer_count + 1) {
    throw std::invalid_argument(
        "AuK fusion requires embeddings and every Qwen layer");
  } else {
  }
  const mx::array normalized =
      mx::fast::layer_norm(mx::astype(mx::slice(all_hidden_states, {1, 0, 0},
                                                all_hidden_states.shape()),
                                      mx::float32),
                           std::nullopt, std::nullopt, 1e-5f);
  const mx::array weights = mx::reshape(
      mx::softmax(checkpoint_.Weight("layer_weights"), 0), {-1, 1, 1});
  return mx::sum(normalized * weights, 0) * checkpoint_.Weight("layer_scale");
}

mx::array DiT::ProjectText(const mx::array &conditioning) const {
  return checkpoint_.RmsNorm(
      checkpoint_.Linear(mx::expand_dims(mx::astype(conditioning, dtype_), 0),
                         "transformer.txt_proj"),
      "transformer.txt_norm", kRmsEpsilon);
}

mx::array DiT::EmbedAudio(const mx::array &latents,
                          const std::optional<mx::array> &valid_frames) const {
  const mx::array projected =
      checkpoint_.Linear(latents, "transformer.audio_embed.linear");
  mx::array positions = valid_frames
                            ? mx::where(mx::expand_dims(*valid_frames, -1),
                                        projected, mx::zeros_like(projected))
                            : projected;
  static const auto mish = mx::compile(
      [](const std::vector<mx::array> &inputs) {
        const auto &x = inputs[0];
        return std::vector<mx::array>{
            x * mx::tanh(mx::logaddexp(x, mx::array(0.0f, x.dtype())))};
      },
      true);
  for (int layer : {0, 2}) {
    const std::string prefix =
        "transformer.audio_embed.conv_pos_embed.conv1d." +
        std::to_string(layer);
    const auto &weight = checkpoint_.Weight(prefix + ".weight");
    positions = mx::conv1d(positions, weight, 1, weight.shape(1) / 2, 1,
                           positions.shape(2) / weight.shape(2)) +
                checkpoint_.Weight(prefix + ".bias");
    if (valid_frames) {
      positions = mx::where(mx::expand_dims(*valid_frames, -1), positions,
                            mx::zeros_like(positions));
    } else {
    }
    positions = mish({positions})[0];
  }
  return projected + positions;
}

DiT::PreparedReference
DiT::PrepareReference(const std::optional<mx::array> &reference,
                      int reference_length,
                      bool classifier_free_guidance) const {
  PreparedReference prepared;
  prepared.frame_count = reference ? reference->shape(0) : 0;
  if (reference_length < 0 || reference_length > prepared.frame_count) {
    throw std::invalid_argument(
        "AuK reference length exceeds its latent frames");
  } else if (prepared.frame_count > 0) {
    prepared.valid_frames = mx::expand_dims(
        mx::less(mx::arange(prepared.frame_count), mx::array(reference_length)),
        0);
    const mx::array reference_latents =
        mx::expand_dims(mx::astype(*reference, dtype_), 0);
    prepared.conditional = EmbedAudio(reference_latents, prepared.valid_frames);
    std::vector<mx::array> embeddings{*prepared.conditional,
                                      *prepared.valid_frames};
    if (classifier_free_guidance) {
      prepared.unconditional =
          EmbedAudio(mx::zeros_like(reference_latents), prepared.valid_frames);
      embeddings.push_back(*prepared.unconditional);
    } else {
    }
    mx::eval(embeddings);
  } else {
  }
  return prepared;
}

mx::array DiT::Forward(const mx::array &latents, const mx::array &conditioning,
                       float timestep,
                       const std::optional<mx::array> &reference,
                       int reference_length,
                       bool classifier_free_guidance) const {
  const auto prepared =
      PrepareReference(reference, reference_length, classifier_free_guidance);
  return Predict(latents, ProjectText(conditioning), mx::array(timestep),
                 prepared, classifier_free_guidance);
}

mx::array DiT::Predict(const mx::array &latents,
                       const mx::array &projected_text,
                       const mx::array &timestep,
                       const PreparedReference &reference,
                       bool classifier_free_guidance) const {
  const auto &time_weight =
      checkpoint_.Weight("transformer.time_embed.time_mlp.0.weight");
  const int half_frequency_count = time_weight.shape(1) / 2;
  const mx::array frequencies =
      mx::exp(mx::arange(half_frequency_count, mx::float32) *
              mx::array(static_cast<float>(-std::log(10000.0) /
                                           (half_frequency_count - 1))));
  const mx::array angles =
      mx::reshape(timestep, {1, 1}) * mx::array(1000.0f) * frequencies;
  mx::array time_embedding = checkpoint_.Linear(
      Silu(checkpoint_.Linear(
          mx::astype(mx::concatenate({mx::sin(angles), mx::cos(angles)}, -1),
                     dtype_),
          "transformer.time_embed.time_mlp.0")),
      "transformer.time_embed.time_mlp.2");
  mx::array audio =
      EmbedAudio(mx::expand_dims(mx::astype(latents, dtype_), 0), std::nullopt);
  mx::array text = projected_text;
  std::optional<mx::array> audio_mask;
  if (reference.frame_count > 0) {
    const mx::array conditional =
        mx::concatenate({*reference.conditional, audio}, 1);
    audio_mask = mx::concatenate(
        {*reference.valid_frames, mx::ones({1, latents.shape(0)}, mx::bool_)},
        1);
    if (classifier_free_guidance) {
      audio = mx::concatenate(
          {conditional, mx::concatenate({*reference.unconditional, audio}, 1)},
          0);
    } else {
      audio = conditional;
    }
  } else if (classifier_free_guidance) {
    audio = mx::concatenate({audio, audio}, 0);
  } else {
  }
  if (classifier_free_guidance) {
    text = mx::concatenate({text, mx::zeros_like(text)}, 0);
    time_embedding = mx::concatenate({time_embedding, time_embedding}, 0);
    audio_mask =
        audio_mask
            ? std::optional(mx::concatenate({*audio_mask, *audio_mask}, 0))
            : std::nullopt;
  } else {
  }
  const int token_count = text.shape(1);
  const auto &inverse_frequencies =
      checkpoint_.Weight("transformer.rotary_embed.inv_freq");
  const mx::array text_mask = mx::ones({text.shape(0), token_count}, mx::bool_);
  const std::optional<mx::array> joint_mask =
      audio_mask ? std::optional(mx::concatenate({*audio_mask, text_mask}, 1))
                 : std::nullopt;
  for (int layer = 0; layer < joint_layer_count_; ++layer) {
    const std::string prefix =
        "transformer.transformer_blocks." + std::to_string(layer);
    const auto text_modulation =
        Modulate(checkpoint_, text, time_embedding, prefix + ".attn_norm_c");
    const auto audio_modulation =
        Modulate(checkpoint_, audio, time_embedding, prefix + ".attn_norm_x");
    const auto attended = Attention(
        checkpoint_, audio_modulation.normalized, text_modulation.normalized,
        prefix + ".attn", inverse_frequencies, head_count_, joint_mask,
        attention_mask_enabled_);
    text = FeedForward(checkpoint_,
                       text + text_modulation.attention_gate * attended[1],
                       text_modulation, prefix + ".ff_c");
    audio = FeedForward(checkpoint_,
                        audio + audio_modulation.attention_gate * attended[0],
                        audio_modulation, prefix + ".ff_x");
  }
  audio = mx::concatenate({text, audio}, 1);
  const std::optional<mx::array> single_mask =
      audio_mask ? std::optional(mx::concatenate({text_mask, *audio_mask}, 1))
                 : std::nullopt;
  for (int layer = 0; layer < single_layer_count_; ++layer) {
    const std::string prefix =
        "transformer.single_transformer_blocks." + std::to_string(layer);
    const auto modulation =
        Modulate(checkpoint_, audio, time_embedding, prefix + ".attn_norm");
    const auto attended = Attention(
        checkpoint_, modulation.normalized, std::nullopt, prefix + ".attn",
        inverse_frequencies, head_count_, single_mask, attention_mask_enabled_);
    audio = FeedForward(checkpoint_,
                        audio + modulation.attention_gate * attended[0],
                        modulation, prefix + ".ff");
  }
  audio = mx::slice(audio, {0, token_count + reference.frame_count, 0},
                    audio.shape());
  const auto final_modulation = mx::split(
      mx::expand_dims(checkpoint_.Linear(Silu(time_embedding),
                                         "transformer.norm_out.linear"),
                      1),
      2, -1);
  return checkpoint_.Linear(
      Normalize(audio) * (mx::array(1.0f, dtype_) + final_modulation[0]) +
          final_modulation[1],
      "transformer.proj_out");
}

mx::array DiT::Sample(const mx::array &conditioning,
                      const std::optional<mx::array> &reference,
                      int reference_length, const mx::array &noise, int steps,
                      float guidance_strength,
                      std::optional<float> sway_coefficient,
                      const std::vector<float> &time_grid,
                      const std::function<void()> &check_cancelled) const {
  if (noise.ndim() != 2 || noise.shape(0) < 1 ||
      noise.shape(1) !=
          checkpoint_.Weight("transformer.proj_out.weight").shape(0)) {
    throw std::invalid_argument(
        "AuK noise must contain positive frames and latent channels");
  } else if (time_grid.empty() ? steps < 1 : time_grid.size() < 2) {
    throw std::invalid_argument(
        "AuK sampling requires a positive step count or a time grid");
  } else {
  }
  check_cancelled();
  mx::array times =
      time_grid.empty()
          ? mx::linspace(0, 1, steps + 1, mx::float32)
          : mx::array(time_grid.data(), {static_cast<int>(time_grid.size())});
  if (time_grid.empty() && sway_coefficient) {
    times =
        times + mx::array(*sway_coefficient) *
                    (mx::cos(mx::array(static_cast<float>(M_PI_2)) * times) -
                     mx::array(1.0f) + times);
  } else {
  }
  const bool use_guidance = guidance_strength >= 1e-5f;
  const mx::array projected_text = ProjectText(conditioning);
  const auto prepared =
      PrepareReference(reference, reference_length, use_guidance);
  mx::array latents = mx::astype(noise, mx::float32);
  for (int step = 0; step < static_cast<int>(times.size()) - 1; ++step) {
    check_cancelled();
    const mx::array timestep = mx::take(times, mx::array(step));
    mx::array velocity =
        Predict(latents, projected_text, timestep, prepared, use_guidance);
    if (use_guidance) {
      const auto predictions = mx::split(velocity, 2, 0);
      velocity =
          predictions[0] + (predictions[0] - predictions[1]) *
                               mx::array(guidance_strength, velocity.dtype());
    } else {
    }
    latents = latents + (mx::take(times, mx::array(step + 1)) - timestep) *
                            mx::squeeze(velocity, 0);
    mx::eval(latents);
  }
  check_cancelled();
  return latents;
}

} // namespace auk
