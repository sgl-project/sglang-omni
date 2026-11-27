// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
#include "auk/auk_conditioner.h"

#include <algorithm>
#include <cmath>

#include "common/audio.h"
#include "common/whisper_encoder.h"

namespace auk {

namespace mx = mlx::core;

mx::array Conditioner::EncodeAudio(const mx::array &features) const {
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
      omni_common::Gelu(checkpoint_.Conv1d(hidden, "audio_tower.conv1", 1, 1)),
      mx::expand_dims(valid, -1));
  hidden =
      omni_common::Gelu(checkpoint_.Conv1d(hidden, "audio_tower.conv2", 2, 1));
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
  for (int layer = 0; layer < config_.audio_layer_count; ++layer) {
    hidden = omni_common::WhisperEncoderLayer(
        checkpoint_, hidden, "audio_tower.layers." + std::to_string(layer),
        config_.audio_head_count, false, mask);
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
