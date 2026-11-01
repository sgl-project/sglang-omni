// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
#include "auk/auk_conditioner.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "common/audio.h"

namespace auk {

namespace mx = mlx::core;

mx::array Conditioner::EncodeAudio(const mx::array &features) const {
  if (features.ndim() != 2 || features.shape(0) != omni_common::kMelBinCount ||
      features.shape(1) < 3 ||
      features.shape(1) > config_.max_audio_samples / omni_common::kHopLength) {
    throw std::invalid_argument(
        "AuK audio features must contain valid mel frames");
  } else {
  }
  const int frame_count = features.shape(1);
  const int window = config_.audio_window_frames;
  const int maximum_frames = std::min(frame_count, window);
  std::vector<mx::array> chunks;
  std::vector<int> chunk_frames;
  for (int start = 0; start < frame_count; start += window) {
    const int frames = std::min(window, frame_count - start);
    const mx::array chunk = mx::transpose(mx::slice(
        features, {0, start}, {omni_common::kMelBinCount, start + frames}));
    chunks.push_back(mx::pad(chunk, {{0, maximum_frames - frames}, {0, 0}}));
    chunk_frames.push_back(frames);
  }
  const int batch = static_cast<int>(chunks.size());
  const mx::array lengths(chunk_frames.data(), {batch, 1}, mx::int32);
  const mx::array valid = mx::less(mx::arange(maximum_frames), lengths);
  const auto dtype = checkpoint_.Weight("audio_tower.conv1.weight").dtype();
  mx::array hidden = mx::astype(mx::stack(chunks), dtype);
  hidden = mx::multiply(
      omni_common::Gelu(mx::add(
          mx::conv1d(hidden, checkpoint_.Weight("audio_tower.conv1.weight"), 1,
                     1),
          checkpoint_.Weight("audio_tower.conv1.bias"))),
      mx::expand_dims(valid, -1));
  hidden = omni_common::Gelu(mx::add(
      mx::conv1d(hidden, checkpoint_.Weight("audio_tower.conv2.weight"), 2, 1),
      checkpoint_.Weight("audio_tower.conv2.bias")));
  const int encoded_frames = hidden.shape(1);
  const int half_hidden = config_.audio_hidden_size / 2;
  const float log_increment =
      static_cast<float>(std::log(10000.0) / (half_hidden - 1));
  const mx::array inverse_timescales = mx::exp(mx::multiply(
      mx::array(-log_increment), mx::arange(half_hidden, mx::float32)));
  const mx::array positions = mx::multiply(
      mx::reshape(mx::arange(encoded_frames, mx::float32), {encoded_frames, 1}),
      inverse_timescales);
  hidden = mx::add(
      hidden,
      mx::astype(mx::concatenate({mx::sin(positions), mx::cos(positions)}, -1),
                 hidden.dtype()));
  const mx::array mask = mx::reshape(
      mx::less(mx::arange(encoded_frames),
               mx::floor_divide(mx::add(lengths, mx::array(1)), mx::array(2))),
      {batch, 1, 1, encoded_frames});
  const int head_size = config_.audio_hidden_size / config_.audio_head_count;
  for (int layer = 0; layer < config_.audio_layer_count; ++layer) {
    const std::string prefix = "audio_tower.layers." + std::to_string(layer);
    const mx::array normed =
        checkpoint_.LayerNorm(hidden, prefix + ".self_attn_layer_norm", 1e-5f);
    const auto project = [&](const std::string &name) {
      return mx::transpose(
          mx::reshape(
              checkpoint_.Linear(normed, prefix + ".self_attn." + name),
              {batch, encoded_frames, config_.audio_head_count, head_size}),
          {0, 2, 1, 3});
    };
    const mx::array attended = mx::fast::scaled_dot_product_attention(
        project("q_proj"), project("k_proj"), project("v_proj"),
        static_cast<float>(std::pow(static_cast<double>(head_size), -0.5)), "",
        mask);
    hidden = mx::add(
        hidden,
        checkpoint_.Linear(
            mx::reshape(mx::transpose(attended, {0, 2, 1, 3}),
                        {batch, encoded_frames, config_.audio_hidden_size}),
            prefix + ".self_attn.out_proj"));
    hidden = mx::add(
        hidden,
        checkpoint_.Linear(omni_common::Gelu(checkpoint_.Linear(
                               checkpoint_.LayerNorm(
                                   hidden, prefix + ".final_layer_norm", 1e-5f),
                               prefix + ".fc1")),
                           prefix + ".fc2"));
    mx::eval(hidden);
  }
  std::vector<mx::array> valid_chunks;
  for (int index = 0; index < batch; ++index) {
    const int kept = (chunk_frames[index] + 1) / 2;
    valid_chunks.push_back(
        mx::reshape(mx::slice(hidden, {index, 0, 0},
                              {index + 1, kept, config_.audio_hidden_size}),
                    {kept, config_.audio_hidden_size}));
  }
  hidden = mx::concatenate(valid_chunks, 0);
  const int paired_frames = ((frame_count + 1) / 2) / 2 * 2;
  hidden = mx::mean(
      mx::reshape(
          mx::slice(hidden, {0, 0}, {paired_frames, config_.audio_hidden_size}),
          {paired_frames / 2, 2, config_.audio_hidden_size}),
      1);
  return checkpoint_.Linear(
      checkpoint_.LayerNorm(hidden, "audio_tower.ln_post", 1e-5f),
      "audio_tower.proj");
}

} // namespace auk
