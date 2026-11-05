// SPDX-License-Identifier: Apache-2.0
#include "qwen3_omni/encoders.h"
#include <cmath>

namespace qwen3_omni {
VisionEncoder::VisionEncoder(Weights &weights, Json configuration)
    : weights(weights), configuration(std::move(configuration)) {
  const auto &config = this->configuration;
  int dimensions = config.at("hidden_size");
  std::string prefix = "thinker.vision_tower";
  weights.require(prefix + ".patch_embed.proj.weight",
                  {dimensions, 2, 16, 16, 3});
  weights.require(prefix + ".patch_embed.proj.bias", {dimensions});
  weights.require_linear(prefix + ".pos_embed", dimensions, 2304);
  for (int i = 0; i < config.at("depth"); ++i) {
    auto layer = prefix + ".blocks." + std::to_string(i);
    for (const auto &name : {"norm1", "norm2"}) {
      weights.require_norm(layer + "." + name, dimensions, true);
    }
    weights.require_linear(layer + ".attn.qkv", dimensions, dimensions * 3,
                           true);
    weights.require_linear(layer + ".attn.proj", dimensions, dimensions, true);
    weights.require_linear(layer + ".mlp.linear_fc1", dimensions,
                           config.at("intermediate_size"), true);
    weights.require_linear(layer + ".mlp.linear_fc2",
                           config.at("intermediate_size"), dimensions, true);
  }
  for (int i = -1;
       i < static_cast<int>(config.at("deepstack_visual_indexes").size());
       ++i) {
    auto name = i < 0 ? prefix + ".merger"
                      : prefix + ".deepstack_merger_list." + std::to_string(i);
    weights.require_norm(name + ".norm", dimensions * (i < 0 ? 1 : 4), true);
    weights.require_linear(name + ".linear_fc1", dimensions * 4, dimensions * 4,
                           true);
    weights.require_linear(name + ".linear_fc2", dimensions * 4,
                           config.at("out_hidden_size"), true);
  }
}
mx::array VisionEncoder::merge(const mx::array &hidden_states,
                               int index) const {
  std::string prefix = "thinker.vision_tower";
  auto name = index < 0
                  ? prefix + ".merger"
                  : prefix + ".deepstack_merger_list." + std::to_string(index);
  int dimensions = configuration.at("hidden_size");
  auto normalized = mx::reshape(
      weights.norm(name + ".norm",
                   index < 0 ? hidden_states
                             : mx::reshape(hidden_states, {-1, dimensions * 4}),
                   1e-6),
      {-1, dimensions * 4});
  return weights.linear(
      name + ".linear_fc2",
      omni_common::Gelu(weights.linear(name + ".linear_fc1", normalized)));
}
std::pair<mx::array, std::vector<mx::array>>
VisionEncoder::forward(const mx::array &patches, const VisualGrid &grid,
                       const Cancellation &cancelled) const {
  std::string prefix = "thinker.vision_tower";
  const int dimensions = configuration.at("hidden_size");
  const int heads = configuration.at("num_heads");
  const int head_dimensions = dimensions / heads;
  auto pixels =
      mx::transpose(mx::reshape(patches, {-1, 3, 2, 16, 16}), {0, 2, 3, 4, 1});
  auto hidden_states =
      mx::reshape(mx::conv3d(pixels,
                             weights[prefix + ".patch_embed.proj.weight"],
                             {2, 16, 16}),
                  {-1, dimensions}) +
      weights[prefix + ".patch_embed.proj.bias"];
  std::vector<int> row_positions, column_positions;
  std::vector<std::vector<int>> indices(4);
  std::vector<std::vector<float>> coefficients(4);
  for (int row_block = 0; row_block < grid.height / 2; ++row_block) {
    for (int column_block = 0; column_block < grid.width / 2; ++column_block) {
      for (int row_within = 0; row_within < 2; ++row_within) {
        for (int column_within = 0; column_within < 2; ++column_within) {
          int row = row_block * 2 + row_within;
          int column = column_block * 2 + column_within;
          row_positions.push_back(row);
          column_positions.push_back(column);
          float y = float(row) * 47 / std::max(1, grid.height - 1);
          float x = float(column) * 47 / std::max(1, grid.width - 1);
          int y0 = int(y), x0 = int(x);
          int y1 = std::min(47, y0 + 1), x1 = std::min(47, x0 + 1);
          float dy = y - y0, dx = x - x0;
          int corners[] = {y0 * 48 + x0, y0 * 48 + x1, y1 * 48 + x0,
                           y1 * 48 + x1};
          float factors[] = {(1 - dy) * (1 - dx), (1 - dy) * dx, dy * (1 - dx),
                             dy * dx};
          for (int i = 0; i < 4; ++i) {
            indices[i].push_back(corners[i]);
            coefficients[i].push_back(factors[i]);
          }
        }
      }
    }
  }
  int patches_per_frame = grid.height * grid.width;
  std::optional<mx::array> absolute;
  for (int i = 0; i < 4; ++i) {
    auto embedding =
        weights.embedding(prefix + ".pos_embed",
                          mx::array(indices[i].data(), {patches_per_frame}));
    auto weighted = embedding * mx::astype(mx::array(coefficients[i].data(),
                                                     {patches_per_frame, 1}),
                                           embedding.dtype());
    absolute = absolute ? *absolute + weighted : weighted;
  }
  hidden_states = hidden_states + mx::tile(*absolute, {grid.frames, 1});
  auto frequencies = mx::power(
      mx::array(10000.f), -mx::arange(0, head_dimensions / 2, 2, mx::float32) /
                              (head_dimensions / 2));
  auto row_angles =
      mx::astype(mx::array(row_positions.data(), {patches_per_frame, 1}),
                 mx::float32) *
      frequencies;
  auto column_angles =
      mx::astype(mx::array(column_positions.data(), {patches_per_frame, 1}),
                 mx::float32) *
      frequencies;
  auto angles = mx::tile(mx::concatenate({row_angles, column_angles}, -1),
                         {grid.frames, 1});
  auto doubled = mx::concatenate({angles, angles}, -1);
  auto factors = std::make_pair(mx::cos(doubled), mx::sin(doubled));
  std::vector<mx::array> deepstack;
  auto capture_indices =
      configuration.at("deepstack_visual_indexes").get<std::vector<int>>();
  for (int i = 0; i < configuration.at("depth"); ++i) {
    check_cancelled(cancelled);
    auto layer = prefix + ".blocks." + std::to_string(i);
    auto normalized = weights.norm(layer + ".norm1", hidden_states, 1e-6);
    auto projected = mx::transpose(
        mx::reshape(weights.linear(layer + ".attn.qkv", normalized),
                    {-1, 3, heads, head_dimensions}),
        {1, 2, 0, 3});
    auto query = rotary(rows(projected, 0, 1), factors);
    auto key = rotary(rows(projected, 1, 2), factors);
    auto value = rows(projected, 2, 3);
    std::vector<mx::array> attended;
    for (int frame = 0; frame < grid.frames; ++frame) {
      int start = frame * patches_per_frame;
      int stop = start + patches_per_frame;
      attended.push_back(mx::fast::scaled_dot_product_attention(
          rows(query, start, stop, 2), rows(key, start, stop, 2),
          rows(value, start, stop, 2), std::pow(float(head_dimensions), -.5f)));
    }
    auto output =
        mx::reshape(mx::transpose(mx::concatenate(attended, 2), {0, 2, 1, 3}),
                    {-1, dimensions});
    hidden_states =
        hidden_states + weights.linear(layer + ".attn.proj", output);
    hidden_states =
        hidden_states + weights.linear(layer + ".mlp.linear_fc2",
                                       omni_common::Gelu(weights.linear(
                                           layer + ".mlp.linear_fc1",
                                           weights.norm(layer + ".norm2",
                                                        hidden_states, 1e-6))));
    auto capture = std::find(capture_indices.begin(), capture_indices.end(), i);
    if (capture != capture_indices.end()) {
      deepstack.push_back(
          merge(hidden_states, int(capture - capture_indices.begin())));
    } else {
    }
    if (i % 4 == 3) {
      mx::eval(hidden_states);
    } else {
    }
  }
  return {merge(hidden_states, -1), deepstack};
}
AudioEncoder::AudioEncoder(Weights &weights, Json configuration)
    : weights(weights), configuration(std::move(configuration)) {
  const auto &config = this->configuration;
  int dimensions = config.at("d_model");
  int channels = config.at("downsample_hidden_size");
  std::string prefix = "thinker.audio_tower";
  for (int i = 1; i <= 3; ++i) {
    auto name = prefix + ".conv2d" + std::to_string(i);
    weights.require(name + ".weight", {channels, 3, 3, i == 1 ? 1 : channels});
    weights.require(name + ".bias", {channels});
  }
  weights.require_linear(
      prefix + ".conv_out",
      channels * ((config.at("num_mel_bins").get<int>() + 7) / 8), dimensions);
  for (int i = 0; i < config.at("encoder_layers"); ++i) {
    auto layer = prefix + ".layers." + std::to_string(i);
    for (const auto &name : {"self_attn_layer_norm", "final_layer_norm"}) {
      weights.require_norm(layer + "." + name, dimensions, true);
    }
    for (const auto &name : {"q_proj", "k_proj", "v_proj", "out_proj"}) {
      weights.require_linear(layer + ".self_attn." + name, dimensions,
                             dimensions, true);
    }
    weights.require_linear(layer + ".fc1", dimensions,
                           config.at("encoder_ffn_dim"), true);
    weights.require_linear(layer + ".fc2", config.at("encoder_ffn_dim"),
                           dimensions, true);
  }
  weights.require_norm(prefix + ".ln_post", dimensions, true);
  weights.require_linear(prefix + ".proj1", dimensions, dimensions, true);
  weights.require_linear(prefix + ".proj2", dimensions, config.at("output_dim"),
                         true);
}
mx::array AudioEncoder::forward(const mx::array &mel,
                                const Cancellation &cancelled) const {
  const int dimensions = configuration.at("d_model");
  const int frame_count = mel.shape(1);
  const int chunk_frames = configuration.at("n_window").get<int>() * 2;
  const int chunks = (frame_count + chunk_frames - 1) / chunk_frames;
  const int maximum_frames = std::min(frame_count, chunk_frames);
  std::string prefix = "thinker.audio_tower";
  std::vector<mx::array> chunk_inputs;
  std::vector<int> selected_rows;
  for (int i = 0; i < chunks; ++i) {
    int count = std::min(chunk_frames, frame_count - i * chunk_frames);
    auto chunk = rows(mel, i * chunk_frames, i * chunk_frames + count, 1);
    chunk = mx::pad(chunk, {{0, 0}, {0, maximum_frames - count}});
    chunk_inputs.push_back(mx::expand_dims(chunk, {0, -1}));
    for (int j = 0; j < output_length(count); ++j) {
      selected_rows.push_back(i * output_length(maximum_frames) + j);
    }
  }
  auto input = mx::concatenate(chunk_inputs, 0);
  for (int i = 1; i <= 3; ++i) {
    auto name = prefix + ".conv2d" + std::to_string(i);
    input = omni_common::Gelu(
        mx::conv2d(input, weights[name + ".weight"], {2, 2}, {1, 1}) +
        weights[name + ".bias"]);
  }
  int width = input.shape(2);
  input = weights.linear(
      prefix + ".conv_out",
      mx::reshape(mx::transpose(input, {0, 2, 3, 1}), {chunks, width, -1}));
  auto frequencies = mx::exp(-std::log(10000.f) / float(dimensions / 2 - 1) *
                             mx::arange(dimensions / 2, mx::float32));
  auto angles =
      mx::reshape(mx::arange(width, mx::float32), {-1, 1}) * frequencies;
  input =
      input + mx::expand_dims(
                  mx::concatenate({mx::sin(angles), mx::cos(angles)}, -1), 0);
  auto hidden_states =
      mx::take(mx::reshape(input, {-1, dimensions}),
               mx::array(selected_rows.data(), {int(selected_rows.size())}), 0);
  const int heads = configuration.at("encoder_attention_heads");
  const int head_dimensions = dimensions / heads;
  const int window_frames = output_length(maximum_frames) *
                            configuration.at("n_window_infer").get<int>() /
                            chunk_frames;
  for (int i = 0; i < configuration.at("encoder_layers"); ++i) {
    check_cancelled(cancelled);
    auto layer = prefix + ".layers." + std::to_string(i);
    auto normalized =
        weights.norm(layer + ".self_attn_layer_norm", hidden_states);
    auto project = [&](const std::string &name) {
      return mx::expand_dims(
          mx::transpose(mx::reshape(weights.linear(layer + ".self_attn." + name,
                                                   normalized),
                                    {-1, heads, head_dimensions}),
                        {1, 0, 2}),
          0);
    };
    auto query = project("q_proj"), key = project("k_proj"),
         value = project("v_proj");
    std::vector<mx::array> outputs;
    for (int start = 0; start < hidden_states.shape(0);
         start += window_frames) {
      int stop = std::min(start + window_frames, hidden_states.shape(0));
      outputs.push_back(mx::fast::scaled_dot_product_attention(
          rows(query, start, stop, 2), rows(key, start, stop, 2),
          rows(value, start, stop, 2), std::pow(float(head_dimensions), -.5f)));
    }
    auto attended =
        mx::reshape(mx::transpose(mx::concatenate(outputs, 2), {0, 2, 1, 3}),
                    {-1, dimensions});
    hidden_states =
        hidden_states + weights.linear(layer + ".self_attn.out_proj", attended);
    hidden_states =
        hidden_states +
        weights.linear(
            layer + ".fc2",
            omni_common::Gelu(weights.linear(
                layer + ".fc1",
                weights.norm(layer + ".final_layer_norm", hidden_states))));
    if (i % 4 == 3) {
      mx::eval(hidden_states);
    } else {
    }
  }
  return weights.linear(prefix + ".proj2",
                        omni_common::Gelu(weights.linear(
                            prefix + ".proj1",
                            weights.norm(prefix + ".ln_post", hidden_states))));
}
} // namespace qwen3_omni
